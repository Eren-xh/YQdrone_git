"""PPO task for recovery immediately after an XH1 armor impact."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from ..controllers.approach_controller import (
        ApproachController,
        ApproachControllerConfig,
    )
    from ..envs.xh1_aviary import XH1Aviary
except ImportError:  # Running train_ppo.py directly from the project directory.
    from controllers.approach_controller import (
        ApproachController,
        ApproachControllerConfig,
    )
    from envs.xh1_aviary import XH1Aviary


class ImpactRecoveryEnv(XH1Aviary):
    """Two-phase task: deterministic approach, then learned recovery.

    PPO actions are ignored during ``APPROACH`` and become active after the
    first MuJoCo contact with ``armor_collision``. This keeps the training
    distribution reproducible and avoids placing MAVLink/APM in every PPO
    sampling step. The armor remains a physical obstacle in both phases,
    including repeated contacts throughout recovery.
    """

    APPROACH = "APPROACH"
    RECOVERY = "RECOVERY"
    RECOVERY_THRESHOLDS = {
        "gravity_z_min": 0.94,
        "rate_norm_max": 1.0,
        "speed_max": 2.5,
        "altitude_min": 0.18,
    }

    def __init__(
        self,
        *args,
        approach_config: Optional[ApproachControllerConfig] = None,
        recovery_hold_steps: int = 20,
        approach_direction_world: Optional[np.ndarray] = None,
        approach_start_distance: float = 0.80,
        impact_clearance: float = 0.075,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.approach_controller = ApproachController(approach_config)
        self.recovery_hold_steps = int(recovery_hold_steps)
        # The body frame is defined with +X pointing to the nose and +Z down.
        # With the CAD-to-world alignment in drone0.xml, body +X maps to world
        # -X at the level pose, so approach from world +X toward -X.
        direction = (
            np.array([-1.0, 0.0, 0.0], dtype=np.float64)
            if approach_direction_world is None
            else np.asarray(approach_direction_world, dtype=np.float64).reshape(3)
        )
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1.0e-8:
            raise ValueError("approach_direction_world must be non-zero")
        self.approach_direction_world = direction / direction_norm
        self.approach_start_distance = float(approach_start_distance)
        self.impact_clearance = float(impact_clearance)
        if self.approach_start_distance <= self.impact_clearance:
            raise ValueError("approach_start_distance must exceed impact_clearance")
        self.phase = self.APPROACH
        self.recovery_steps = 0
        self.success_hold = 0
        self.impact_seen = False
        self.previous_recovery_action = np.zeros(4, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # Keep the same travel distance as the original setup while moving
        # the starting point to the opposite side of the target. A caller can
        # still override this with options={"position": [...]}.
        reset_options = dict(options or {})
        if "position" not in reset_options:
            armor_before_reset = self.armor_position.copy()
            reset_options["position"] = (
                armor_before_reset
                - self.approach_direction_world * self.approach_start_distance
            )
        obs, info = super().reset(seed=seed, options=reset_options)
        armor = self.armor_position.copy()
        # Aim just before the target along the approach direction so the
        # actual collision geometry generates the impact. The target is at
        # world x=0.875 and the XH1 body +X/nose travels toward -X.
        target = armor - self.approach_direction_world * self.impact_clearance
        self.approach_controller.set_target(target)
        self.approach_controller.config.target_velocity = np.zeros(3)
        self.phase = self.APPROACH
        self.recovery_steps = 0
        self.success_hold = 0
        self.impact_seen = False
        self.previous_recovery_action.fill(0.0)
        info.update(
            {
                "phase": self.phase,
                "approach_target": target.copy(),
                "approach_direction_world": self.approach_direction_world.copy(),
            }
        )
        return obs, info

    def _recovery_reward(self, action: np.ndarray, info: dict) -> float:
        # In the requested z-down body frame, upright gravity points along +Z.
        gravity_alignment = float(np.dot(self.gravity_body, np.array([0.0, 0.0, 1.0])))
        rate_norm = float(np.linalg.norm(self.angular_rate_body))
        altitude = float(self.position[2])
        velocity_world = np.asarray(self.velocity_world, dtype=np.float64)
        # A policy that keeps the attitude nearly upright while slowly losing
        # altitude is unsafe.  Make the altitude floor and vertical speed
        # visible in every recovery transition so PPO cannot exploit the old
        # contact-latch equilibrium at z ~= 0.35 m.
        altitude_penalty = max(0.0, 0.25 - altitude)
        vertical_speed_penalty = abs(float(velocity_world[2]))
        horizontal_speed_penalty = float(np.linalg.norm(velocity_world[:2]))
        action = np.asarray(action, dtype=np.float64)
        action_delta = action - self.previous_recovery_action
        hover_action = (
            2.0
            * self.propulsion.config.mass
            * self.propulsion.config.gravity
            / self.propulsion.max_total_thrust
            - 1.0
        )
        thrust_error = float(action[0]) - float(hover_action)
        reward = (
            1.8 * gravity_alignment
            - 0.22 * rate_norm
            - 8.0 * altitude_penalty
            - 0.25 * vertical_speed_penalty
            - 0.05 * horizontal_speed_penalty
            # Keep the policy near the physical hover command while it spends
            # its rate authority removing the impact angular velocity.  The
            # old coefficient was too weak, so low-thrust actions could look
            # attractive while the vehicle was still nearly upright.
            - 0.60 * thrust_error * thrust_error
            - 0.01 * float(np.dot(action[1:], action[1:]))
            - 0.01 * float(np.dot(action_delta, action_delta))
        )
        if info.get("impact", False):
            reward -= 0.05
        self.previous_recovery_action = action.astype(np.float32)
        return float(reward)

    def recovery_status(self) -> dict[str, bool]:
        """Expose the same four checks used by the recovery decision."""
        limits = self.RECOVERY_THRESHOLDS
        return {
            "upright": bool(self.gravity_body[2] > limits["gravity_z_min"]),
            "rates_small": bool(np.linalg.norm(self.angular_rate_body) < limits["rate_norm_max"]),
            "velocity_small": bool(np.linalg.norm(self.velocity_world) < limits["speed_max"]),
            "enough_altitude": bool(self.position[2] > limits["altitude_min"]),
        }

    def _is_recovered(self) -> bool:
        return all(self.recovery_status().values())

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(4)
        action = np.clip(action, -1.0, 1.0)

        if self.phase == self.APPROACH:
            thrust, rates, approach_info = self.approach_controller.command(self)
            obs, info = self._simulate_command(thrust, rates)
            info.update(approach_info)
            if info["ground_contact"]:
                # The vehicle reached the newly added physical ground before
                # making the intended armor impact.
                reward = -10.0
                terminated = True
                truncated = False
                info.update(
                    {
                        "phase": self.phase,
                        "impact_seen": self.impact_seen,
                        "recovered": False,
                        "crashed": True,
                        "low_altitude": bool(self.position[2] < 0.12),
                    }
                )
                return obs, reward, terminated, truncated, info
            if info["impact"]:
                self.phase = self.RECOVERY
                self.impact_seen = True
                self.recovery_steps = 0
                self.success_hold = 0
                reward = 0.0
            else:
                # Approach is deterministic; this small time penalty encourages
                # a short, repeatable collision setup without training on it.
                reward = -0.01
            terminated = False
            truncated = self.step_count >= self.max_episode_steps
            info["phase"] = self.phase
            info["impact_seen"] = self.impact_seen
            return obs, float(reward), terminated, truncated, info

        self.previous_action = action.copy()
        thrust, rates = self.propulsion.decode_action(action)
        obs, info = self._simulate_command(thrust, rates)
        self.recovery_steps += 1
        reward = self._recovery_reward(action, info)

        recovery_criteria = self.recovery_status()
        if all(recovery_criteria.values()):
            self.success_hold += 1
        else:
            self.success_hold = 0
        recovered = self.success_hold >= self.recovery_hold_steps
        low_altitude = bool(self.position[2] < 0.12)
        # Recovery has no artificial time limit. It ends on recovery success
        # or physical contact with the added ground plane. The base Gym
        # horizon may still return ``truncated`` as a global safety cap.
        ground_contact = bool(info.get("ground_contact", False))
        crashed = ground_contact
        terminated = bool(recovered or crashed)
        if recovered:
            reward += 10.0
        elif crashed:
            reward -= 10.0

        truncated = self.step_count >= self.max_episode_steps
        info.update(
            {
                "phase": self.phase,
                "impact_seen": self.impact_seen,
                "recovery_steps": self.recovery_steps,
                "success_hold": self.success_hold,
                "recovery_criteria": recovery_criteria,
                "recovered": recovered,
                "crashed": crashed,
                "low_altitude": low_altitude,
                "ground_contact": ground_contact,
            }
        )
        return obs, float(reward), terminated, truncated, info

