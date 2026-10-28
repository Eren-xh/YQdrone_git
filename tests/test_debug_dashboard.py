"""Regression checks for diagnostic timing, persistence, and read-only serving.

Run from drone_xh1: python -m unittest discover -s tests -v
"""

import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

from debug_dashboard import DashboardServer, TelemetryHub, json_safe, session_metadata
from evaluate_ppo import DEFAULT_MODEL, DEFAULT_NORMALIZER, load_policy, run_episode


class RecorderTests(unittest.TestCase):
    def test_full_log_survives_memory_eviction_and_replays_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = TelemetryHub({"schema_version": 1}, log_root=directory, max_frames=2)
            for step in range(9):
                hub.publish({"step": step + 1, "episode": 1 if step < 4 else 2})
            hub.add_summary({"episode": 1, "recovered": False})
            hub.finish()
            self.assertEqual([f["sequence"] for f in hub.snapshot()["frames"]], [8, 9])
            records = [json.loads(line) for line in hub.log_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(r["type"] == "frame" for r in records), 9)
            self.assertEqual(records[-1]["type"], "finished")
            replay = TelemetryHub.replay(hub.log_path)
            self.assertEqual([f["sequence"] for f in replay.snapshot()["frames"]], list(range(1, 10)))
            self.assertEqual({f["episode"] for f in replay.snapshot()["frames"]}, {1, 2})
            self.assertEqual(replay.summaries, hub.summaries)
            self.assertEqual(replay.status, "replay")

    def test_sensor_nonfinite_values_remain_valid_json(self):
        values = json_safe({"values": np.array([np.nan, np.inf, -np.inf, 1.0]), "flag": np.bool_(True)})
        self.assertEqual(json.loads(json.dumps(values, allow_nan=False)),
                         {"values": [None, None, None, 1.0], "flag": True})

    def test_http_incremental_polling_and_port_conflict(self):
        hub = TelemetryHub({"schema_version": 1})
        for step in range(1005):
            hub.publish({"step": step + 1})
        server = DashboardServer(hub, port=0)
        server.start()
        other = DashboardServer(hub, port=server.httpd.server_port)
        other.start()
        try:
            self.assertNotEqual(server.url, other.url)
            with urlopen(server.url + "api/state?after=0") as response:
                first = json.load(response)
            self.assertEqual(len(first["frames"]), 1000)
            with urlopen(server.url + "api/state?after=1000") as response:
                rest = json.load(response)
            self.assertEqual([f["sequence"] for f in rest["frames"]], [1001, 1002, 1003, 1004, 1005])
            with urlopen(server.url) as response:
                self.assertIn("PPO 实时调试", response.read().decode("utf-8"))
            for route, code in [("api/state?after=bad", 400), ("../drone_model/drone0.xml", 404)]:
                with self.assertRaises(HTTPError) as raised:
                    urlopen(server.url + route)
                self.assertEqual(raised.exception.code, code)
        finally:
            other.close()
            server.close()


@unittest.skipUnless(DEFAULT_MODEL.exists() and DEFAULT_NORMALIZER.exists(), "saved policy is unavailable")
class PolicyDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.vec = load_policy(DEFAULT_MODEL, DEFAULT_NORMALIZER)
        cls.env = cls.vec.venv.envs[0]
        # Discover the approach duration instead of coupling this diagnostic
        # test to the impact time of a particular physics implementation.
        cls.env.reset(seed=100)
        while cls.env.phase == cls.env.APPROACH:
            _, _, terminated, truncated, _ = cls.env.step(np.zeros(4, dtype=np.float32))
            if terminated or truncated:
                raise AssertionError("Diagnostic fixture did not reach an armor impact")
        # Stop just after impact to exercise terminal auto-reset while tilted.
        cls.env.max_episode_steps = cls.env.step_count + 3
        cls.baseline = run_episode(cls.model, cls.vec, 100, realtime=False)
        cls.meta = session_metadata(DEFAULT_MODEL, DEFAULT_NORMALIZER, cls.env, cls.vec, 2, 100)
        cls.hub = TelemetryHub(cls.meta)
        cls.traced = run_episode(cls.model, cls.vec, 100, realtime=False, telemetry=cls.hub)
        cls.frames = cls.hub.snapshot()["frames"]

    @classmethod
    def tearDownClass(cls):
        cls.vec.close()

    def test_diagnostics_do_not_change_policy_or_physics(self):
        self.assertEqual(self.baseline, self.traced)
        self.assertEqual(len(self.frames), self.traced["steps"])

    def test_input_is_exact_normalized_predict_input(self):
        for f in self.frames:
            with self.subTest(step=f["step"]):
                expected = self.vec.normalize_obs(np.array([f["raw_obs"]], dtype=np.float32))[0]
                np.testing.assert_array_equal(np.asarray(f["policy_obs"], dtype=np.float32), expected)

    def test_contact_step_is_approach_then_ppo_and_previous_action_is_causal(self):
        first_impact = next(i for i, f in enumerate(self.frames) if f["recovery"]["impact_seen"])
        self.assertEqual(self.frames[first_impact]["controller"], "APPROACH")
        self.assertEqual(self.frames[first_impact + 1]["controller"], "PPO")
        np.testing.assert_array_equal(self.frames[first_impact + 1]["raw_obs"][9:], np.zeros(4))
        for index in range(first_impact + 1, len(self.frames)):
            f = self.frames[index]
            thrust, rates = self.env.propulsion.decode_action(f["action"])
            self.assertAlmostEqual(f["command"]["thrust"], thrust)
            np.testing.assert_allclose(f["command"]["rates"], rates)
            if index > first_impact + 1:
                np.testing.assert_array_equal(f["raw_obs"][9:], self.frames[index - 1]["action"])

    def test_terminal_frame_uses_pre_reset_state(self):
        final = self.frames[-1]
        self.assertTrue(final["done"])
        self.assertTrue(final["truncated"])
        self.assertAlmostEqual(final["state"]["tilt_deg"], self.traced["final_tilt_deg"])
        reset_tilt = np.degrees(np.arccos(np.clip(self.env.gravity_body[2], -1., 1.)))
        self.assertFalse(np.isclose(final["state"]["tilt_deg"], reset_tilt, atol=1e-8))
        np.testing.assert_allclose(self.env.gravity_body, [0.0, 0.0, 1.0], atol=1e-10)
        self.assertEqual(self.env.step_count, 0)


if __name__ == "__main__":
    unittest.main()
