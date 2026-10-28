"""Gymnasium environment for the XH1 MuJoCo model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

try:
    from ..physics.xh1_aerodynamics import AerodynamicsConfig, XH1Aerodynamics
    from ..physics.xh1_propulsion import PropulsionConfig, PropulsionOutput, XH1Propulsion
except ImportError:  # Running train_ppo.py directly from the project directory.
    from physics.xh1_aerodynamics import AerodynamicsConfig, XH1Aerodynamics
    from physics.xh1_propulsion import PropulsionConfig, PropulsionOutput, XH1Propulsion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = PROJECT_ROOT / "drone_model" / "drone0.xml"


'''_quat_mul 是一个四元数乘法函数，用来把两个旋转组合起来'''
def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for MuJoCo quaternions in ``w, x, y, z`` order."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


'''把roll、pitch、yaw角转换为四元数表示'''
def _rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


class XH1Aviary(gym.Env):
    """Single-XH1 MuJoCo environment with a four-dimensional PPO action.

    The normalized action is ``[total_thrust, p_target, q_target, r_target]``:
    the first component maps to total thrust and the remaining components map
    to body-rate targets. A low-level rate controller and motor mixer remain
    between PPO and MuJoCo, matching the intended deployment architecture.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    '''XH1Aviary 环境的构造函数。
    执行 env = XH1Aviary() 时，
    它会准备好 MuJoCo 模型、推进器和空气动力学模型，以及强化学习需要的动作空间、观测空间和环境状态'''

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_XML,
        sim_freq: int = 1000,
        ctrl_freq: int = 200,
        render_mode: Optional[str] = None,# 渲染模式，None表示不渲染，"rgb_array"表示渲染为RGB数组
        propulsion_config: Optional[PropulsionConfig] = None,
        aerodynamics_config: Optional[AerodynamicsConfig] = None,
        max_episode_steps: int = 1200,
    ) -> None:
        super().__init__()# 调用父类gym.Env的构造函数，初始化环境
        if sim_freq <= 0 or ctrl_freq <= 0 or sim_freq % ctrl_freq != 0:
            raise ValueError("sim_freq must be a positive multiple of ctrl_freq")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")

        self.xml_path = Path(xml_path).resolve()
        self.sim_freq = int(sim_freq)
        self.ctrl_freq = int(ctrl_freq)
        self.sim_timestep = 1.0 / self.sim_freq
        self.control_timestep = 1.0 / self.ctrl_freq
        self.sim_steps_per_ctrl = self.sim_freq // self.ctrl_freq
        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)

        '''加载 MuJoCo 模型和仿真数据'''
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.model.opt.timestep = self.sim_timestep
        self.data = mujoco.MjData(self.model)

        '''MuJoCo 内部使用数字 ID 表示模型中的物体。
        这里按 XML 里的名称找到无人机主体、IMU、四个电机位置、装甲、地面等对象，并将 ID 保存下来，
        供环境后续读取或计算使用'''
        self.root_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "drone0")
        self.imu_site_id = self._required_id(mujoco.mjtObj.mjOBJ_SITE, "imu_site")
        self.motor_site_ids = [#列表推导式，等价于依次查找 motor_0_site 到 motor_3_site，生成一个包含四个 ID 的列表
            self._required_id(mujoco.mjtObj.mjOBJ_SITE, f"motor_{i}_site")
            for i in range(4)
        ]
        self.motor_body_ids = [int(self.model.site_bodyid[site_id]) for site_id in self.motor_site_ids]
        self.armor_geom_id = self._required_id(
            mujoco.mjtObj.mjOBJ_GEOM, "armor_collision"
        )
        self.armor_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY, "armor_target"
        )
        self.ground_geom_id = self._required_id(
            mujoco.mjtObj.mjOBJ_GEOM, "ground_collision"
        )
        self.drone_geom_ids = {
            self._required_id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in (
                "drone0_base_collision",
                "ra_collision",
                "la_collision",
                "rb_collision",
                "lb_collision",
            )
        }

        mujoco.mj_forward(self.model, self.data)
        self.alignment_quat = self.data.qpos[3:7].copy()#保存模型初始四元数作为对齐姿态
        arm_positions = self._motor_positions_body()#从模型中读取电机相对机体的位置
        '''创建推进器和空气动力学对象'''
        self.propulsion = XH1Propulsion(arm_positions, propulsion_config)
        self.aerodynamics = XH1Aerodynamics(aerodynamics_config)

        # 3 specific-force values + 3 angular-rate values + 3 gravity-direction
        # values + 4 previous action values = 13 dimensions.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        observation_limit = np.array(
            [1.0e6] * 6 + [1.01] * 3 + [1.0] * 4, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-observation_limit, high=observation_limit, dtype=np.float32
        )

        self.previous_action = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.last_output: Optional[PropulsionOutput] = None
        self.last_info: dict[str, Any] = {}
        self._renderer: Optional[mujoco.Renderer] = None

    def _required_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"required MuJoCo object not found: {name}")
        return int(obj_id)

    def _sensor(self, name: str) -> np.ndarray:
        sid = self._required_id(mujoco.mjtObj.mjOBJ_SENSOR, name)
        start = int(self.model.sensor_adr[sid])
        dim = int(self.model.sensor_dim[sid])
        return np.asarray(self.data.sensordata[start : start + dim], dtype=np.float64)

    def _body_to_world(self) -> np.ndarray:
        return np.asarray(self.data.site_xmat[self.imu_site_id]).reshape(3, 3)

    def _motor_positions_body(self) -> np.ndarray:
        body_to_world = self._body_to_world()
        # Motor forces create moments about the complete vehicle COM. The IMU
        # site defines the X-quad center, while the CAD root and the complete
        # vehicle COM are offset from that center. Read the force arms after
        # applying the XML installation rotation and IMU frame orientation.
        center = np.asarray(self.data.subtree_com[self.root_body_id])
        return np.array(
            [
                body_to_world.T
                @ (np.asarray(self.data.site_xpos[sid]) - center)
                for sid in self.motor_site_ids
            ],
            dtype=np.float64,
        )

    def _gravity_body(self) -> np.ndarray:
        return self._body_to_world().T @ np.array([0.0, 0.0, -1.0])

    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate(
            [self._sensor("imu_accel"), self._sensor("imu_gyro"), self._gravity_body(), self.previous_action]
        )
        return obs.astype(np.float32)

    def _contact_info(self) -> dict[str, Any]:
        contacts: list[tuple[str, str]] = []
        impact_force = 0.0
        ground_force = 0.0
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            geom_a = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom[0])
            ) or ""
            geom_b = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom[1])
            ) or ""
            contacts.append((geom_a, geom_b))
            if (
                (int(contact.geom[0]) == self.armor_geom_id and int(contact.geom[1]) in self.drone_geom_ids)
                or (int(contact.geom[1]) == self.armor_geom_id and int(contact.geom[0]) in self.drone_geom_ids)
            ):
                contact_force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, index, contact_force)
                impact_force = max(impact_force, float(np.linalg.norm(contact_force[:3])))
            if (
                (int(contact.geom[0]) == self.ground_geom_id and int(contact.geom[1]) in self.drone_geom_ids)
                or (int(contact.geom[1]) == self.ground_geom_id and int(contact.geom[0]) in self.drone_geom_ids)
            ):
                contact_force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, index, contact_force)
                ground_force = max(ground_force, float(np.linalg.norm(contact_force[:3])))
        return {
            "impact": bool(impact_force > 0.0),
            "impact_force": impact_force,
            "impact_impulse": impact_force * self.sim_timestep,
            "ground_contact": bool(ground_force > 0.0),
            "ground_force": ground_force,
            "contacts": contacts,
        }

    def _apply_motor_wrench(self, output: PropulsionOutput) -> None:
        # Apply each rotor force at its XML site. This preserves the actual
        # force arm and avoids collapsing all thrust to the root body's COM.
        body_to_world = self._body_to_world()
        yaw_ratio = self.propulsion.config.km / self.propulsion.config.kf
        for index, site_id in enumerate(self.motor_site_ids):
            thrust = float(output.motor_thrust[index])
            # The requested XH1 body frame is x-forward, y-right, z-down.
            # Rotor thrust therefore points along body -Z (world +Z at level).
            force_world = body_to_world @ np.array([0.0, 0.0, -thrust])
            torque_body = np.array(
                [0.0, 0.0, self.propulsion.config.spin_directions[index] * yaw_ratio * thrust]
            )
            torque_world = body_to_world @ torque_body
            mujoco.mj_applyFT(
                self.model,
                self.data,
                force_world,
                torque_world,
                self.data.site_xpos[site_id],
                self.motor_body_ids[index],
                self.data.qfrc_applied,
            )


    '''_simulate_command 是环境内部用于执行一次物理控制更新的方法：
    它接收总推力和目标角速度，运行推进器与 MuJoCo 仿真，
    最后返回新的观测和诊断信息'''
    def _simulate_command(
        self, total_thrust: float, target_rates: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        measured_rates = self._sensor("imu_gyro")
        self.last_output = self.propulsion.update_physical(
            total_thrust, target_rates, measured_rates, self.control_timestep
        )

        aero_info: dict[str, np.ndarray] = {}
        for _ in range(self.sim_steps_per_ctrl):
            self.data.xfrc_applied[:] = 0.0
            self.data.qfrc_applied[:] = 0.0
            self._apply_motor_wrench(self.last_output)
            # framelinvel is already expressed in world axes. Blade drag
            # requires the same IMU-point velocity resolved in body axes.
            velocity_world = self.velocity_world
            velocity_body = self._body_to_world().T @ velocity_world
            aero_info = self.aerodynamics.apply(
                self.model,
                self.data,
                self.root_body_id,
                self.imu_site_id,
                self.motor_site_ids,
                self.last_output.motor_rpm,
                velocity_body,
                velocity_world,
                self._gravity_body(),
                self.propulsion.config.kf,
            )
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        contact_info = self._contact_info()
        info: dict[str, Any] = {
            **contact_info,
            # Snapshot before DummyVecEnv can automatically reset a terminal
            # episode. Diagnostics must never read that reset as the end state.
            "specific_force": self._sensor("imu_accel").copy(),
            "body_to_world": self._body_to_world().copy(),
            "commanded_total_thrust": float(np.clip(total_thrust, 0.0, self.propulsion.max_total_thrust)),
            "commanded_target_rates": self.last_output.target_rates.copy(),
            "position": self.data.site_xpos[self.imu_site_id].copy(),
            "velocity_world": self.velocity_world,
            "velocity_body": self.velocity_body,
            "angular_rate": self._sensor("imu_gyro").copy(),
            "gravity_body": self._gravity_body().copy(),
            "motor_rpm": self.last_output.motor_rpm.copy(),
            "desired_motor_rpm": self.last_output.desired_motor_rpm.copy(),
            "motor_thrust": self.last_output.motor_thrust.copy(),
            "total_thrust": self.last_output.total_thrust,
            "motor_torque_body": self.last_output.torque_body.copy(),
            "aero_force_body": aero_info.get("aero_force_body", np.zeros(3)),
        }
        self.last_info = info
        return self._get_obs(), info


    '''step 是 Gymnasium 环境的单步交互入口：
    强化学习算法传入一个动作，环境执行一次控制更新，
    然后返回观测、奖励和回合状态。'''

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(4)
        action = np.clip(action, -1.0, 1.0)
        self.previous_action = action.copy()
        total_thrust, target_rates = self.propulsion.decode_action(action)
        obs, info = self._simulate_command(total_thrust, target_rates)
        terminated = False
        truncated = self.step_count >= self.max_episode_steps
        return obs, 0.0, terminated, truncated, info


    '''reset 用来开始一个新的 episode，（开始采集一批新的步）
    并把环境和无人机状态恢复到初始状态。
    Gymnasium 通常在每个 episode 开始时调用它。'''
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}
        mujoco.mj_resetData(self.model, self.data)
        # ``position`` is the requested X-quad/IMU body origin. The XML root
        # body remains at the CAD export origin, so convert the requested body
        # position to the freejoint root position after applying orientation.
        position = np.asarray(options.get("position", [0.0, 0.0, 0.50]), dtype=np.float64)
        rpy = np.asarray(options.get("rpy", [0.0, 0.0, 0.0]), dtype=np.float64)
        desired_body_quat = _rpy_to_quat(rpy)
        root_quat = _quat_mul(desired_body_quat, self.alignment_quat)
        self.data.qpos[:3] = position
        self.data.qpos[3:7] = root_quat / np.linalg.norm(root_quat)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        imu_offset_world = self.data.site_xpos[self.imu_site_id] - self.data.qpos[:3]
        self.data.qpos[:3] = position - imu_offset_world
        mujoco.mj_forward(self.model, self.data)
        self.propulsion.reset(hover=True)
        self.previous_action.fill(0.0)
        self.step_count = 0
        self.last_output = None
        self.last_info = {}
        return self._get_obs(), {"phase": "reset", "position": position.copy()}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="flow_camera")
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.data.site_xpos[self.imu_site_id], dtype=np.float64)

    @property
    def velocity_world(self) -> np.ndarray:
        """IMU-point linear velocity in world axes (MuJoCo framelinvel)."""
        return self._sensor("imu_vel").copy()

    @property
    def velocity_body(self) -> np.ndarray:
        """The same IMU-point velocity in the forward/right/down body frame."""
        return self._body_to_world().T @ self.velocity_world

    @property
    def angular_rate_body(self) -> np.ndarray:
        return self._sensor("imu_gyro")

    @property
    def gravity_body(self) -> np.ndarray:
        return self._gravity_body()

    @property
    def armor_position(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self.armor_body_id], dtype=np.float64)

