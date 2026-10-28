"""XH1 propulsion and four-motor allocation.

The XML contains the rigid geometry and motor sites. This module supplies the
actuator model that is intentionally kept outside MuJoCo's ``<actuator>``
section: PPO commands total thrust and body-rate targets, a rate controller
produces the requested body torques, and a mixer converts the wrench to four
motor thrusts with first-order motor lag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

#Propulsion推进，动力
@dataclass
class PropulsionConfig:
    """Parameters that should eventually be replaced by XH1 bench data."""

    mass: float = 0.179
    gravity: float = 9.81
    motor_count: int = 4

    # N/(rad/s)^2 and Nm/(rad/s)^2. These are conservative initial values for
    # a 5 g motor-plus-propeller unit, not final identified coefficients.
    kf: float = 1.50e-7
    km: float = 3.00e-9
    max_rpm: float = 20_000.0
    min_rpm: float = 0.0
    motor_time_constant: float = 0.025

    rate_limit: np.ndarray = field(
        default_factory=lambda: np.array([10.0, 10.0, 8.0], dtype=np.float64)
    )
    rate_kp: np.ndarray = field(
        default_factory=lambda: np.array([0.0020, 0.0020, 0.0010], dtype=np.float64)
    )
    rate_damping: np.ndarray = field(
        default_factory=lambda: np.array([0.00015, 0.00015, 0.00008], dtype=np.float64)
    )
    torque_limit: np.ndarray = field(
        default_factory=lambda: np.array([0.030, 0.030, 0.012], dtype=np.float64)
    )
    spin_directions: np.ndarray = field(
        # Motor order is (-x,-y), (-x,+y), (+x,-y), (+x,+y). Opposite
        # diagonals rotate in the same direction, as in an X quad.
        default_factory=lambda: np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    )

    def __post_init__(self) -> None:
        self.rate_limit = np.asarray(self.rate_limit, dtype=np.float64)
        self.rate_kp = np.asarray(self.rate_kp, dtype=np.float64)
        self.rate_damping = np.asarray(self.rate_damping, dtype=np.float64)
        self.torque_limit = np.asarray(self.torque_limit, dtype=np.float64)
        self.spin_directions = np.asarray(self.spin_directions, dtype=np.float64)
        if self.spin_directions.shape != (self.motor_count,):
            raise ValueError("spin_directions must have one entry per motor")
        if np.any(self.rate_limit <= 0.0):
            raise ValueError("rate limits must be positive")
        if self.kf <= 0.0 or self.km < 0.0 or self.motor_time_constant <= 0.0:
            raise ValueError("invalid propulsion coefficients")

    @property
    def max_motor_thrust(self) -> float:
        omega = self.max_rpm * 2.0 * np.pi / 60.0
        return float(self.kf * omega * omega)

    @property
    def max_total_thrust(self) -> float:
        return float(self.motor_count * self.max_motor_thrust)


@dataclass
class PropulsionOutput:
    """Result of one control update."""

    total_thrust: float
    target_rates: np.ndarray
    torque_body: np.ndarray
    motor_thrust: np.ndarray
    motor_rpm: np.ndarray
    desired_motor_rpm: np.ndarray


class XH1Propulsion:
    """Rate-controlled X-quad propulsion model for the XH1 XML."""


    '''XH1Propulsion 类的构造函数：创建对象时自动运行，用来设置电机位置、配置和初始转速'''
    def __init__(
        self,
        arm_positions_body: Optional[np.ndarray] = None,
        config: Optional[PropulsionConfig] = None,
    ) -> None:
        '''表示该初始化函数不返回一个值。他的作用是初始化对象内部的属性和状态。'''
        self.config = config or PropulsionConfig()
        if arm_positions_body is None:
            # Fallback geometry only. XH1Aviary replaces this with positions
            # read from motor_*_site in the XML after the model is loaded.
            arm_positions_body = np.array(
                [
                    [0.050, -0.050, 0.0],
                    [0.050, 0.050, 0.0],
                    [-0.050, -0.050, 0.0],
                    [-0.050, 0.050, 0.0],
                ],
                dtype=np.float64,
            )
        self.arm_positions_body = np.asarray(arm_positions_body, dtype=np.float64)
        if self.arm_positions_body.shape != (self.config.motor_count, 3):
            raise ValueError("arm_positions_body must have shape (4, 3)")
        self.motor_rpm = np.zeros(self.config.motor_count, dtype=np.float64)

    @property
    def max_total_thrust(self) -> float:
        return self.config.max_total_thrust

    @property
    def rate_limit(self) -> np.ndarray:
        return self.config.rate_limit.copy()

    '''上面两个@property装饰器的作用是返回当前配置的总最大推力和角速度限制数组
    将类中的方法转换为属性,使得可以像访问属性一样访问方法。这样就可以直接像访问属性一样使用：propulsion.max_total_thrust
    而不是：propulsion.get_max_total_thrust()'''
    

    '''reset 方法：重置电机转速，可选地设置为悬停转速'''
    def reset(self, hover: bool = True) -> None:
        """Reset motor speed, optionally to the nominal hover RPM."""
        if hover:
            hover_thrust = self.config.mass * self.config.gravity / self.config.motor_count
            hover_omega = np.sqrt(max(hover_thrust, 0.0) / self.config.kf)
            hover_rpm = hover_omega * 60.0 / (2.0 * np.pi)#RPM:round per minute，每分钟转数，电机转速单位

            '''将计算出的 RPM 限制在最小和最大转速范围内，然后把所有电机的转速设成这个值。
            .fill(...) 是原地修改数组'''
            self.motor_rpm.fill(np.clip(hover_rpm, self.config.min_rpm, self.config.max_rpm))
        else:
            self.motor_rpm.fill(0.0)


    '''decode_action 把 PPO 输出的归一化动作转换成推进器能使用的实际指令。
    输入是 4 个 [-1, 1] 范围内的数，输出是“总推力”和“三个方向的目标角速度” '''

    '''
    最后返回 (总推力, 目标角速度数组)。例如可以这样调用：
    thrust, rates = propulsion.decode_action(action)
    thrust 是一个数字，rates 是包含三个角速度目标的数组。
    '''
    def decode_action(self, action: np.ndarray) -> tuple[float, np.ndarray]:
        """Map normalized PPO action ``[-1, 1]^4`` to physical commands."""
        action = np.asarray(action, dtype=np.float64).reshape(4)
        action = np.clip(action, -1.0, 1.0)
        total_thrust = 0.5 * (action[0] + 1.0) * self.max_total_thrust
        target_rates = action[1:] * self.config.rate_limit
        return float(total_thrust), target_rates

    def rate_controller(
        self,
        target_rates: np.ndarray,
        measured_rates: np.ndarray,
    ) -> np.ndarray:
        target_rates = np.clip(
            np.asarray(target_rates, dtype=np.float64),
            -self.config.rate_limit,
            self.config.rate_limit,
        )
        measured_rates = np.asarray(measured_rates, dtype=np.float64).reshape(3)
        torque = self.config.rate_kp * (target_rates - measured_rates)
        torque -= self.config.rate_damping * measured_rates
        return np.clip(torque, -self.config.torque_limit, self.config.torque_limit)

    def allocation_matrix(self) -> np.ndarray:
        """Return the motor-thrust to body-wrench allocation matrix."""
        matrix = np.zeros((4, self.config.motor_count), dtype=np.float64)
        yaw_ratio = self.config.km / self.config.kf
        for i, position in enumerate(self.arm_positions_body):
            x, y = position[:2]
            # XH1 uses a z-down body frame, so physical rotor force is
            # F_body=[0, 0, -T]. For r=[x,y,z], r x F gives
            # tau_x=-y*T and tau_y=+x*T.
            matrix[:, i] = [
                1.0,
                -y,
                x,
                self.config.spin_directions[i] * yaw_ratio,
            ]
        return matrix

    def allocate_wrench(self, wrench: np.ndarray) -> np.ndarray:
        """Solve for motor thrusts and clip them to physical limits."""
        wrench = np.asarray(wrench, dtype=np.float64).reshape(4)
        matrix = self.allocation_matrix()
        if abs(float(np.linalg.det(matrix))) > 1.0e-12:
            thrust = np.linalg.solve(matrix, wrench)
        else:
            thrust = np.linalg.pinv(matrix) @ wrench
        return np.clip(thrust, 0.0, self.config.max_motor_thrust)

    def wrench_from_motor_thrust(self, motor_thrust: np.ndarray) -> np.ndarray:
        motor_thrust = np.asarray(motor_thrust, dtype=np.float64).reshape(4)
        return self.allocation_matrix() @ motor_thrust

    def update(
        self,
        action: np.ndarray,
        measured_rates: np.ndarray,
        dt: float,
    ) -> PropulsionOutput:
        """Advance motor dynamics from a normalized PPO action."""
        total_thrust, target_rates = self.decode_action(action)
        return self.update_physical(total_thrust, target_rates, measured_rates, dt)

    def update_physical(
        self,
        total_thrust: float,
        target_rates: np.ndarray,
        measured_rates: np.ndarray,
        dt: float,
    ) -> PropulsionOutput:
        """Advance motor dynamics from physical thrust/rate commands."""
        total_thrust = float(np.clip(total_thrust, 0.0, self.max_total_thrust))
        target_rates = np.clip(
            np.asarray(target_rates, dtype=np.float64).reshape(3),
            -self.config.rate_limit,
            self.config.rate_limit,
        )
        torque_body = self.rate_controller(target_rates, measured_rates)
        desired_thrust = self.allocate_wrench(
            np.concatenate(([total_thrust], torque_body))
        )
        desired_omega = np.sqrt(desired_thrust / self.config.kf)
        desired_rpm = desired_omega * 60.0 / (2.0 * np.pi)
        desired_rpm = np.clip(desired_rpm, self.config.min_rpm, self.config.max_rpm)

        dt = max(float(dt), 1.0e-6)
        alpha = 1.0 - np.exp(-dt / self.config.motor_time_constant)
        self.motor_rpm += alpha * (desired_rpm - self.motor_rpm)
        self.motor_rpm = np.clip(
            self.motor_rpm, self.config.min_rpm, self.config.max_rpm
        )
        omega = self.motor_rpm * 2.0 * np.pi / 60.0
        motor_thrust = self.config.kf * omega * omega
        actual_wrench = self.wrench_from_motor_thrust(motor_thrust)

        return PropulsionOutput(
            total_thrust=float(actual_wrench[0]),
            target_rates=target_rates.copy(),
            torque_body=actual_wrench[1:].copy(),
            motor_thrust=motor_thrust.copy(),
            motor_rpm=self.motor_rpm.copy(),
            desired_motor_rpm=desired_rpm.copy(),
        )

