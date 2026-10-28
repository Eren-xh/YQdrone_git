"""Read CSV and reproduce the impact without editing training/model parameters.

Run from drone_xh1 with the mujoco environment. Branch experiments preserve
the armor and ground; an air test translates a copied terminal pose away
from the armor solely to isolate rate-control authority.
"""
from pathlib import Path
import argparse
import csv
import json
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_ppo import load_policy, DEFAULT_MODEL, DEFAULT_NORMALIZER
from debug_dashboard import json_safe


def wrench_about_com(env, generalized):
    jp = np.zeros((3, env.model.nv))
    jr = np.zeros((3, env.model.nv))
    mujoco.mj_jacSubtreeCom(env.model, env.data, jp, env.root_body_id)
    mujoco.mj_jacBody(env.model, env.data, None, jr, env.root_body_id)
    world = np.linalg.solve(np.vstack([jp, jr]).T, generalized)
    rotation = env._body_to_world()
    return np.r_[rotation.T @ world[:3], rotation.T @ world[3:]]


def run(model, vec, mode="ppo", steps=1200):
    env = vec.venv.envs[0]
    obs, _ = env.reset(seed=100)
    frames = []
    for step in range(steps):
        normalized = vec.normalize_obs(obs[None, :])
        action, _ = model.predict(normalized, deterministic=True)
        action = action[0]
        controller = env.phase
        if controller == env.RECOVERY and mode != "ppo":
            gravity = env.gravity_body.copy()
            if mode == "zero_thrust_zero_rate":
                action = np.array([-1., 0., 0., 0.], dtype=np.float32)
            else:
                thrust = env.propulsion.config.mass * 9.81
                if mode == "attitude_70pct_hover":
                    thrust *= .7
                rates = np.clip(5 * np.cross([0., 0., 1.], gravity), -6, 6)
                if mode == "rate_damping":
                    rates = np.zeros(3)
                action = np.r_[2 * thrust / env.propulsion.max_total_thrust - 1,
                               rates / env.propulsion.rate_limit].astype(np.float32)
        requested_torque = None
        unclipped = None
        if controller == env.RECOVERY:
            thrust, rates = env.propulsion.decode_action(action)
            requested_torque = env.propulsion.rate_controller(rates, env.angular_rate_body)
            unclipped = np.linalg.solve(env.propulsion.allocation_matrix(), np.r_[thrust, requested_torque])
        obs, reward, terminated, truncated, info = env.step(action)
        motor_wrench = wrench_about_com(env, env.data.qfrc_applied.copy())
        contact_wrench = wrench_about_com(env, env.data.qfrc_constraint.copy())
        rotation = env._body_to_world().copy()
        normal = np.array(env.data.geom_xmat[env.armor_geom_id]).reshape(3, 3)[:, 0]
        frame = {
            "step": step + 1, "time": (step + 1) * env.control_timestep,
            "controller": controller, "action": action.copy(),
            "rates": info["angular_rate"],
            "tilt": np.degrees(np.arccos(np.clip(info["gravity_body"][2], -1, 1))),
            "gravity_body": info["gravity_body"], "z": info["position"][2],
            "contact": info["impact"], "contact_force": info["impact_force"],
            "requested_torque": requested_torque, "unclipped_motor_thrust": unclipped,
            "actual_motor_thrust": info["motor_thrust"],
            "motor_torque": motor_wrench[3:], "contact_torque": contact_wrench[3:],
            "motor_force_world": rotation @ motor_wrench[:3],
            "contact_force_world": rotation @ contact_wrench[:3],
            "thrust_dot_plate_normal": float(np.dot(-rotation[:, 2], normal)),
            "recovered": info.get("recovered", False), "crashed": info.get("crashed", False),
        }
        frames.append(frame)
        if terminated or truncated:
            break
    return frames


def isolated_air_test(model, vec, baseline_steps):
    results = {}
    for mode in ["at_plate", "away_from_plate"]:
        # Recreate exactly the same pose, motor RPM, and zero velocity in
        # both tests; only world translation differs. The plate stays present.
        run(model, vec, steps=baseline_steps)
        env = vec.venv.envs[0]
        if mode == "away_from_plate":
            env.data.qpos[:3] += np.array([1.5, 0., 2.])
        env.data.qvel[:] = 0.
        mujoco.mj_forward(env.model, env.data)
        start = env.angular_rate_body.copy()
        series = []
        for _ in range(20):
            _, info = env._simulate_command(env.propulsion.config.mass * 9.81, np.array([0., 5., 0.]))
            series.append({"q": float(info["angular_rate"][1]), "contact": info["impact"]})
        results[mode] = {"start_rates": start, "after_0p1s_rates": env.angular_rate_body.copy(), "series": series}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/impact_control_analysis"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8-sig")))
    recovery = [r for r in rows if r["controller"] == "PPO"]
    def values(name):
        return np.array([float(r[name]) for r in recovery])
    statistics = {
        "csv": str(args.csv.resolve()), "samples": len(rows), "recovery_samples": len(recovery),
        "contact_fraction": np.mean([r["armor_contact"] == "true" for r in recovery]),
        "action_std": [np.std(values(k)) for k in ["a_T", "a_p", "a_q", "a_r"]],
        "action_saturation_fraction": [np.mean(np.abs(values(k)) >= .99) for k in ["a_T", "a_p", "a_q", "a_r"]],
        "network_clip_fraction": {k: np.mean(np.abs(values("network_" + k)) >= 9.999) for k in ["fx", "fy", "fz", "gx", "gy", "gz"]},
    }
    model, vec = load_policy(DEFAULT_MODEL, DEFAULT_NORMALIZER)
    try:
        baseline = run(model, vec)
        # Verify this reproduction actually corresponds to the supplied CSV.
        reproduction_error = max(abs(frame["tilt"] - float(row["tilt_deg"])) for frame, row in zip(baseline, rows))
        assert len(baseline) == len(rows) and reproduction_error < 1e-6, reproduction_error
        branches = {mode: run(model, vec, mode) for mode in ["zero_thrust_zero_rate", "rate_damping", "attitude_hover", "attitude_70pct_hover"]}
        air_test = isolated_air_test(model, vec, baseline_steps=1000)
        env = vec.venv.envs[0]
        statistics.update({
            "reproduction_max_tilt_error_deg": reproduction_error,
            "motor_arm_positions_body": env.propulsion.arm_positions_body,
            "max_motor_thrust": env.propulsion.config.max_motor_thrust,
            "max_total_thrust": env.propulsion.max_total_thrust,
            "motor_time_constant": env.propulsion.config.motor_time_constant,
            "obs_norm_mean": vec.obs_rms.mean,
            "obs_norm_std": np.sqrt(vec.obs_rms.var + vec.epsilon),
            "air_rate_test": air_test,
        })
        late = [f for f in baseline if f["time"] >= 2]
        statistics["late_2_to_6s"] = {
            "motor_torque_mean": np.mean([f["motor_torque"] for f in late], axis=0),
            "contact_torque_mean": np.mean([f["contact_torque"] for f in late], axis=0),
            "requested_torque_mean": np.mean([f["requested_torque"] for f in late], axis=0),
            "motor_vertical_force_mean": np.mean([f["motor_force_world"][2] for f in late]),
            "contact_vertical_force_mean": np.mean([f["contact_force_world"][2] for f in late]),
            "thrust_dot_plate_normal_mean": np.mean([f["thrust_dot_plate_normal"] for f in late]),
            "infeasible_mix_fraction": np.mean([np.any((f["unclipped_motor_thrust"] < 0) | (f["unclipped_motor_thrust"] > env.propulsion.config.max_motor_thrust)) for f in late]),
        }
        statistics["branches"] = {mode: {
            "steps": len(frames), "end_tilt": frames[-1]["tilt"],
            "end_rates": frames[-1]["rates"], "end_z": frames[-1]["z"],
            "recovered": frames[-1]["recovered"], "crashed": frames[-1]["crashed"],
        } for mode, frames in branches.items()}
        (args.output / "report.json").write_text(json.dumps(json_safe(statistics), ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output / "wrench_trace.json").write_text(json.dumps(json_safe(baseline), ensure_ascii=False), encoding="utf-8")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
        t = np.array([f["time"] for f in baseline])
        axes[0].plot(t, [f["tilt"] for f in baseline], color="#385cda", label="Total tilt (not Euler pitch)")
        axes[0].axhline(117.5, color="gray", ls="--", label="117.5 deg: plate-aligned pose")
        axes[0].set_ylabel("Tilt (deg)")
        for key, label in [("cmd_q_rad_s", "Pitch-rate target"), ("q_rad_s", "Measured q")]:
            axes[1].plot([float(r["response_time_s"]) for r in rows], [float(r[key]) for r in rows], label=label, lw=1)
        axes[1].set_ylabel("Rate (rad/s)")
        def block_mean(values):
            count = len(values) // 10 * 10
            return np.asarray(values)[:count].reshape(-1, 10).mean(axis=1)
        for key, label in [("requested_torque", "Rate controller demand"), ("motor_torque", "Motor torque"), ("contact_torque", "Contact torque")]:
            axes[2].plot(block_mean(t), block_mean([f[key][1] if f[key] is not None else np.nan for f in baseline]), label=label, lw=1)
        axes[2].set_ylabel("Pitch torque (N m)")
        axes[2].set_ylim(-.04, .04)
        axes[2].set_title("50 ms block means of control-step samples (not exact substep integrals)", fontsize=9)
        for key, label in [("motor_force_world", "Motor vertical force"), ("contact_force_world", "Contact vertical force")]:
            axes[3].plot(block_mean(t), block_mean([f[key][2] for f in baseline]), label=label, lw=1)
        axes[3].axhline(.179 * 9.81, color="gray", ls="--", label="Weight")
        axes[3].set_ylabel("World vertical force (N)")
        axes[3].set_ylim(-3, 6)
        axes[3].set_xlabel("Simulation time (s)")
        for ax in axes:
            ax.axvline(1.215, color="#be6470", ls=":")
            ax.legend(loc="upper right", fontsize=8, ncol=3)
            ax.grid(alpha=.2)
        fig.savefig(args.output / "diagnosis.png", dpi=160)
        plt.close(fig)
        print(json.dumps(json_safe(statistics), ensure_ascii=False, indent=2))
        for step in [243,244,250,260,280,300,600,1200]:
            print("FRAME", json.dumps(json_safe(baseline[step - 1])))
    finally:
        vec.close()


if __name__ == "__main__":
    main()
