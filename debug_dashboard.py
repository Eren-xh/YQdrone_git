"""Local, read-only PPO telemetry server and full-rate JSONL recorder.

No additional packages are needed. The simulation publishes copied values;
background threads handle disk writes and browser requests. Replay a recording::

    python debug_dashboard.py --replay runs/debug/<session>/telemetry.jsonl
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import SimpleQueue
import socket
import threading
import time
from urllib.parse import parse_qs, urlsplit

import numpy as np


PAGE = Path(__file__).resolve().parent / "web" / "dashboard.html"
OBSERVATIONS = [
    ("fx", "比力 X", "m/s²"), ("fy", "比力 Y", "m/s²"), ("fz", "比力 Z", "m/s²"),
    ("p", "角速度 p / roll", "rad/s"), ("q", "角速度 q / pitch", "rad/s"),
    ("r", "角速度 r / yaw", "rad/s"),
    ("gx", "重力方向 X", "1"), ("gy", "重力方向 Y", "1"), ("gz", "重力方向 Z", "1"),
    ("prev_T", "上一步 PPO 动作 T", "1"), ("prev_p", "上一步 PPO 动作 p", "1"),
    ("prev_q", "上一步 PPO 动作 q", "1"), ("prev_r", "上一步 PPO 动作 r", "1"),
]


def json_safe(value):
    """Produce strict JSON, including a visible null for invalid sensor data."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def session_metadata(model_path, normalizer_path, env, vec_env, episodes, seed):
    config = env.propulsion.config
    return json_safe({
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "model": str(model_path), "vecnormalize": str(normalizer_path),
        "episodes": episodes, "seed": seed,
        "control_hz": env.ctrl_freq, "control_dt": env.control_timestep,
        "max_episode_steps": env.max_episode_steps,
        "body_frame": "FRD: X 前 / Y 右 / Z 下；世界 Z 向上",
        "observations": [{"key": key, "label": label, "unit": unit} for key, label, unit in OBSERVATIONS],
        "norm_obs": vec_env.norm_obs, "clip_obs": vec_env.clip_obs,
        "max_total_thrust": env.propulsion.max_total_thrust,
        "hover_thrust": config.mass * config.gravity,
        "rate_limit": env.propulsion.rate_limit,
        "max_rpm": config.max_rpm,
        "max_motor_thrust": config.max_motor_thrust,
        "recovery_thresholds": env.RECOVERY_THRESHOLDS,
        "recovery_hold_steps": env.recovery_hold_steps,
        "reference_rotation": env._body_to_world().copy(),
    })


def make_frame(*, episode, seed, step, dt, controller, raw_obs, policy_obs,
               action, info, reward, done, reference_rotation):
    """One causal transition: input at t, command, then resulting state at t+dt.

    All resulting state comes from info, including at terminal auto-reset.
    model.predict's Box-clipped action is recorded; this is not the unbounded
    Gaussian mean inside the policy network.
    """
    gravity = np.asarray(info["gravity_body"])
    relative_rotation = np.asarray(reference_rotation).T @ np.asarray(info["body_to_world"])
    r = relative_rotation
    rpy = np.degrees([
        np.arctan2(r[2, 1], r[2, 2]),
        np.arcsin(np.clip(-r[2, 0], -1.0, 1.0)),
        np.arctan2(r[1, 0], r[0, 0]),
    ])
    clipped_action = np.clip(np.asarray(action).reshape(4), -1.0, 1.0)
    return json_safe({
        "episode": episode, "seed": seed, "step": step,
        "input_time": (step - 1) * dt, "time": step * dt,
        "controller": controller, "phase_after": info.get("phase"),
        "raw_obs": np.asarray(raw_obs).reshape(13).copy(),
        "policy_obs": np.asarray(policy_obs).reshape(13).copy(),
        "action": clipped_action,
        "action_saturated": np.abs(clipped_action) >= 0.99,
        "command": {
            "thrust": info["commanded_total_thrust"],
            "rates": info["commanded_target_rates"],
        },
        "actual": {
            "thrust": info["total_thrust"], "rates": info["angular_rate"],
            "motor_rpm": info["motor_rpm"], "desired_motor_rpm": info["desired_motor_rpm"],
            "motor_thrust": info["motor_thrust"], "torque": info["motor_torque_body"],
        },
        "state": {
            "position": info["position"], "rpy_deg": rpy,
            "tilt_deg": np.degrees(np.arccos(np.clip(gravity[2], -1.0, 1.0))),
            "rate_norm": np.linalg.norm(info["angular_rate"]),
            # Both velocity fields refer to the IMU point; an orthogonal
            # world/body rotation preserves the speed used by the task.
            "speed": np.linalg.norm(info["velocity_body"]),
        },
        "contact": {
            "armor": info.get("impact", False), "ground": info.get("ground_contact", False),
            "force": info.get("impact_force", 0.0),
            "impulse": info.get("impact_impulse", 0.0),
            "pairs": info.get("contacts", []),
        },
        "recovery": {
            "impact_seen": info.get("impact_seen", False),
            "steps": info.get("recovery_steps", 0),
            "hold": info.get("success_hold", 0),
            "criteria": info.get("recovery_criteria", {}),
            "recovered": info.get("recovered", False),
            "crashed": info.get("crashed", False),
        },
        "reward": reward, "done": done,
        "truncated": info.get("TimeLimit.truncated", False),
    })


class TelemetryHub:
    """Bounded browser history; every frame is also written to disk."""

    def __init__(self, metadata, log_root=None, max_frames=60_000):
        self.metadata = json_safe(metadata)
        self.frames = deque(maxlen=max_frames)
        self.summaries = []
        self.sequence = 0
        self.status = "running"
        self.message = ""
        self.lock = threading.Lock()
        self.queue = SimpleQueue()
        self.writer = None
        self.log_path = None
        self.writer_error = None
        self._finished = False
        if log_root is not None:
            root = Path(log_root)
            root.mkdir(parents=True, exist_ok=True)
            session_dir = root / datetime.now().strftime("eval_%Y%m%d_%H%M%S_%f")
            session_dir.mkdir()
            self.log_path = session_dir / "telemetry.jsonl"
            self.log_path.touch()
            (session_dir / "session.json").write_text(
                json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            self.writer = threading.Thread(target=self._write_records, name="xh1-telemetry-writer", daemon=True)
            self.writer.start()
            self.queue.put({"type": "session", "metadata": self.metadata})

    def _write_records(self):
        try:
            with self.log_path.open("a", encoding="utf-8") as stream:
                last_flush = time.monotonic()
                while True:
                    record = self.queue.get()
                    if record is None:
                        break
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                    if time.monotonic() - last_flush >= 0.5:
                        stream.flush()
                        last_flush = time.monotonic()
        except Exception as exc:
            self.writer_error = str(exc)

    def publish(self, frame):
        # make_frame owns all lists; the simulator will never mutate them.
        with self.lock:
            self.sequence += 1
            frame["sequence"] = self.sequence
            self.frames.append(frame)
        if self.writer is not None:
            self.queue.put({"type": "frame", **frame})

    def add_summary(self, summary):
        summary = json_safe(summary)
        with self.lock:
            self.summaries.append(summary)
        if self.writer is not None:
            self.queue.put({"type": "summary", "summary": summary})

    def snapshot(self, after=0, limit=1000):
        with self.lock:
            frames = [frame for frame in self.frames if frame["sequence"] > after][:limit]
            return {
                "frames": frames, "last_sequence": self.sequence,
                "oldest_sequence": self.frames[0]["sequence"] if self.frames else 0,
                "summaries": self.summaries.copy(), "status": self.status,
                "message": self.message, "writer_error": self.writer_error,
                "log_path": str(self.log_path) if self.log_path else None,
            }

    def finish(self, status="finished", message=""):
        if self._finished:
            return
        self._finished = True
        with self.lock:
            self.status, self.message = status, message
        if self.writer is not None:
            self.queue.put({"type": "finished", "status": status, "message": message})
            self.queue.put(None)
            self.writer.join()
            (self.log_path.parent / "summary.json").write_text(
                json.dumps(self.summaries, ensure_ascii=False, indent=2), encoding="utf-8")
            if self.writer_error:
                print(f"Telemetry recording error: {self.writer_error}")

    @classmethod
    def replay(cls, path):
        hub = None
        path = Path(path).resolve()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                kind = record.pop("type")
                if kind == "session":
                    hub = cls(record["metadata"])
                elif hub is None:
                    raise ValueError("Recording must start with a session record")
                elif kind == "frame":
                    hub.publish(record)
                elif kind == "summary":
                    hub.add_summary(record["summary"])
        if hub is None:
            raise ValueError("Recording is empty")
        hub.log_path = path
        hub.finish("replay", "已加载历史记录")
        return hub


class DashboardServer:
    def __init__(self, hub, port=8765):
        self.hub = hub
        page = PAGE.read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send_bytes(self, data, content_type, filename=None):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                if filename:
                    self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                try:
                    url = urlsplit(self.path)
                    if url.path == "/":
                        self.send_bytes(page, "text/html; charset=utf-8")
                    elif url.path == "/api/session":
                        self.send_json(hub.metadata)
                    elif url.path == "/api/state":
                        try:
                            after = max(0, int(parse_qs(url.query).get("after", ["0"])[0]))
                        except ValueError:
                            self.send_error(400, "after must be an integer")
                            return
                        self.send_json(hub.snapshot(after))
                    elif url.path == "/api/log" and hub.log_path is not None:
                        self.send_bytes(hub.log_path.read_bytes(), "application/x-ndjson", "telemetry.jsonl")
                    elif url.path == "/favicon.ico":
                        self.send_error(404)
                    else:
                        self.send_error(404)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def send_json(self, data):
                self.send_bytes(json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                                "application/json; charset=utf-8")

        class LocalHTTPServer(ThreadingHTTPServer):
            # HTTPServer's default SO_REUSEADDR permits duplicate live binds
            # on Windows, which could send a page another run's telemetry.
            allow_reuse_address = False

            def server_bind(self):
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                super().server_bind()

        try:
            self.httpd = LocalHTTPServer(("127.0.0.1", port), Handler)
        except OSError as exc:
            if exc.errno not in (errno.EADDRINUSE, 10048):
                raise
            self.httpd = LocalHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/"
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="xh1-dashboard-http", daemon=True)

    def start(self):
        self.thread.start()
        print(f"PPO live dashboard: {self.url}")
        if self.hub.log_path:
            print(f"Telemetry (all control steps): {self.hub.log_path}")

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    hub = TelemetryHub.replay(args.replay)
    server = DashboardServer(hub, args.port)
    server.start()
    try:
        if not args.no_browser:
            import webbrowser
            webbrowser.open(server.url)
        print("Replay ready. Press Ctrl+C to close the server.")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
