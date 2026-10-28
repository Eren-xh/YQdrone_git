"""Deterministic pre-impact approach controller.

During PPO training this controller owns the vehicle until MuJoCo detects
contact with ``armor_collision``. It creates repeatable, controllable impacts;
the policy is trained only on the post-impact recovery phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ApproachControllerConfig:
    target_position: np.ndarray = field(
        default_factory=lambda: np.array([0.73, 0.0, 0.50], dtype=np.float64)
    )
    target_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    position_kp: np.ndarray = field(
        default_factory=lambda: np.array([1.8, 1.8, 2.2], dtype=np.float64)
    )
    velocity_kd: np.ndarray = field(
        default_factory=lambda: np.array([1.1, 1.1, 1.3], dtype=np.float64)
    )
    max_acceleration: np.ndarray = field(
        default_factory=lambda: np.array([4.0, 4.0, 3.0], dtype=np.float64)
    )
    attitude_kp: np.ndarray = field(
        default_factory=lambda: np.array([5.0, 5.0, 2.0], dtype=np.float64)
    )
    max_rate: np.ndarray = field(
        default_factory=lambda: np.array([6.0, 6.0, 3.0], dtype=np.float64)
    )

    def __post_init__(self) -> None:
        for name in (
            "target_position",
            "target_velocity",
            "position_kp",
            "velocity_kd",
            "max_acceleration",
            "attitude_kp",
            "max_rate",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,)")
            setattr(self, name, value)


class ApproachController:
    """Position-to-attitude controller used before impact."""

    def __init__(self, config: ApproachControllerConfig | None = None) -> None:
        self.config = config or ApproachControllerConfig()

    def set_target(self, target_position: np.ndarray) -> None:
        target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
        self.config.target_position = target_position.copy()

    def command(self, env: Any) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
        position = np.asarray(env.position, dtype=np.float64)
        velocity = np.asarray(env.velocity_world, dtype=np.float64)
        position_error = self.config.target_position - position
        velocity_error = self.config.target_velocity - velocity

        acceleration = (
            self.config.position_kp * position_error
            + self.config.velocity_kd * velocity_error
        )
        acceleration = np.clip(
            acceleration, -self.config.max_acceleration, self.config.max_acceleration
        )
        gravity = np.array([0.0, 0.0, 9.81], dtype=np.float64)
        desired_force_world = env.propulsion.config.mass * (acceleration + gravity)
        total_thrust = float(np.linalg.norm(desired_force_world))
        if total_thrust < 1.0e-8:
            desired_z_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            desired_z_world = desired_force_world / total_thrust

        body_to_world = np.asarray(env._body_to_world(), dtype=np.float64)
        # XH1 uses a z-down body frame. Rotor thrust is along body -Z, so
        # align the current thrust axis rather than the body +Z/down axis.
        current_thrust_axis_world = -body_to_world[:, 2]
        # Cross(current, desired) produces the body rate that tilts the
        # thrust vector toward the desired world force.
        attitude_error_world = np.cross(
            current_thrust_axis_world, desired_z_world
        )
        attitude_error_body = body_to_world.T @ attitude_error_world
        target_rates = self.config.attitude_kp * attitude_error_body
        target_rates -= 0.25 * np.asarray(env.angular_rate_body)
        target_rates = np.clip(target_rates, -self.config.max_rate, self.config.max_rate)

        return total_thrust, target_rates, {
            "approach_position_error": position_error,
            "approach_desired_acceleration": acceleration,
            "approach_desired_force_world": desired_force_world,
        }

