"""Aerodynamic corrections for the XH1 MuJoCo model.

The implementation follows the modular structure of MuJoCo-drones-gym. The
``blade_drag`` term is the same type of rotor-speed-dependent equivalent
blade-drag model used by its ``MJC_DRAG`` mode. It is an engineering model,
not a CFD solver or a per-blade fluid simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import mujoco
import numpy as np


@dataclass
class AerodynamicsConfig:
    # Initial values based on the public Forster-style model. They must be
    # identified for XH1 with motor thrust/drag experiments later.
    blade_drag_coeff: np.ndarray = field(
        default_factory=lambda: np.array([9.1785e-7, 9.1785e-7, 10.311e-7])
    )
    ground_effect_coeff: float = 11.36859
    prop_radius: float = 0.021
    ground_height_clip: float = 0.02
    air_density: float = 1.225
    wind_drag_coefficient: float = 0.001
    wind_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        self.blade_drag_coeff = np.asarray(self.blade_drag_coeff, dtype=np.float64)
        self.wind_world = np.asarray(self.wind_world, dtype=np.float64)
        if self.blade_drag_coeff.shape != (3,):
            raise ValueError("blade_drag_coeff must have three components")
        if self.wind_world.shape != (3,):
            raise ValueError("wind_world must have three components")


class XH1Aerodynamics:
    """Apply optional aerodynamic forces to a MuJoCo body."""

    def __init__(self, config: Optional[AerodynamicsConfig] = None) -> None:
        self.config = config or AerodynamicsConfig()

    def blade_drag_body(
        self, motor_rpm: np.ndarray, velocity_body: np.ndarray
    ) -> np.ndarray:
        """Return the equivalent rotor blade-drag force in body coordinates."""
        rpm = np.asarray(motor_rpm, dtype=np.float64).reshape(-1)
        velocity_body = np.asarray(velocity_body, dtype=np.float64).reshape(3)
        rotor_speed_sum = float(np.sum(2.0 * np.pi * rpm / 60.0))
        return -self.config.blade_drag_coeff * rotor_speed_sum * velocity_body

    def ground_effect_body(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        motor_site_ids: Iterable[int],
        motor_rpm: np.ndarray,
        kf: float,
        gravity_body: np.ndarray,
    ) -> np.ndarray:
        """Return additional upward body force near the ground.

        XH1 uses a z-down body frame, so the upward force is body ``-Z``.
        """
        gravity_body = np.asarray(gravity_body, dtype=np.float64).reshape(3)
        # Disable the approximation when the vehicle is far from level; this
        # prevents a tipped vehicle from receiving an unphysical vertical lift.
        if abs(float(gravity_body[0])) > 0.7 or abs(float(gravity_body[1])) > 0.7:
            return np.zeros(3, dtype=np.float64)
        heights = np.array(
            [data.site_xpos[int(site_id), 2] for site_id in motor_site_ids],
            dtype=np.float64,
        )
        heights = np.clip(heights, self.config.ground_height_clip, np.inf)
        rpm = np.asarray(motor_rpm, dtype=np.float64).reshape(-1)
        # Propulsion kf is expressed per (rad/s)^2, while the actuator state
        # is stored in RPM. Convert before applying the ground-effect model.
        omega = rpm * 2.0 * np.pi / 60.0
        per_rotor = omega * omega * kf * self.config.ground_effect_coeff
        per_rotor *= (self.config.prop_radius / (4.0 * heights)) ** 2
        return np.array([0.0, 0.0, -float(np.sum(per_rotor))], dtype=np.float64)

    def wind_force_world(
        self, velocity_world: np.ndarray, enabled: bool = True
    ) -> np.ndarray:
        if not enabled:
            return np.zeros(3, dtype=np.float64)
        relative = self.config.wind_world - np.asarray(velocity_world, dtype=np.float64)
        speed = float(np.linalg.norm(relative))
        if speed < 1.0e-9:
            return np.zeros(3, dtype=np.float64)
        return (
            0.5
            * self.config.air_density
            * self.config.wind_drag_coefficient
            * speed
            * relative
        )

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_id: int,
        imu_site_id: int,
        motor_site_ids: Iterable[int],
        motor_rpm: np.ndarray,
        velocity_body: np.ndarray,
        velocity_world: np.ndarray,
        gravity_body: np.ndarray,
        kf: float,
        enable_ground_effect: bool = True,
        enable_blade_drag: bool = True,
        enable_wind: bool = False,
    ) -> dict[str, np.ndarray]:
        """Add configured aerodynamic forces to ``data.xfrc_applied``."""
        body_to_world = np.asarray(data.site_xmat[imu_site_id]).reshape(3, 3)
        force_body = np.zeros(3, dtype=np.float64)
        if enable_blade_drag:
            force_body += self.blade_drag_body(motor_rpm, velocity_body)
        if enable_ground_effect:
            force_body += self.ground_effect_body(
                model, data, motor_site_ids, motor_rpm, kf, gravity_body
            )
        drag_world = body_to_world @ force_body
        wind_world = self.wind_force_world(velocity_world, enabled=enable_wind)
        total_world = drag_world + wind_world
        data.xfrc_applied[body_id, :3] += total_world
        return {
            "aero_force_body": force_body,
            "aero_force_world": total_world,
            "blade_drag_body": self.blade_drag_body(motor_rpm, velocity_body)
            if enable_blade_drag
            else np.zeros(3),
            "wind_force_world": wind_world,
        }

    @staticmethod
    def downwash_force_body(
        own_position_world: np.ndarray,
        other_positions_world: np.ndarray,
        own_rpm: np.ndarray,
        prop_radius: float,
        coeff_1: float = 2267.18,
        coeff_2: float = 0.16,
        coeff_3: float = -0.11,
    ) -> np.ndarray:
        """Approximate the downwash force from drones above this drone."""
        own = np.asarray(own_position_world, dtype=np.float64).reshape(3)
        others = np.asarray(other_positions_world, dtype=np.float64)
        if others.size == 0:
            return np.zeros(3, dtype=np.float64)
        total = 0.0
        for other in others.reshape(-1, 3):
            dz = other[2] - own[2]
            dxy = float(np.linalg.norm(other[:2] - own[:2]))
            if dz <= 0.0 or dxy >= 10.0:
                continue
            beta = coeff_2 * dz + coeff_3
            if beta <= 1.0e-6:
                continue
            alpha = coeff_1 * (prop_radius / (4.0 * dz)) ** 2
            total += -alpha * np.exp(-0.5 * (dxy / beta) ** 2)
        return np.array([0.0, 0.0, total], dtype=np.float64)

