# XH1 撞击后稳定恢复仿真工程

本工程用于完成 XH1 四旋翼在撞击装甲板后的快速姿态恢复研究。当前闭环在 MuJoCo 中运行，PPO 只负责撞击后的恢复控制；撞击前由确定性的接近控制器把无人机送到装甲板。这样可以先把“碰撞后的恢复策略”单独训练和验证，再接入 APM、MAVLink 和实机。

本 README 以当前目录中的代码为准。当前可以运行完整的 MuJoCo/Gymnasium/PPO 仿真，但推进系数、叶片阻力系数、碰撞参数和传感器噪声仍需用实测数据标定。

## 1. 工程结构和各模块职责

```text
drone_xh1/
├── drone_model/
│   ├── drone0.xml             # XH1 刚体、质量、惯量、碰撞体、传感器和装甲板
│   └── meshes/                # base_link、ra、la、rb、lb 五个 STL
├── envs/
│   └── xh1_aviary.py          # MuJoCo + Gymnasium 基础环境
├── physics/
│   ├── xh1_propulsion.py      # 总推力/角速度目标、角速度控制、X 型混控、电机滞后
│   └── xh1_aerodynamics.py    # 叶片阻力、地面效应、风力和下洗接口
├── controllers/
│   └── approach_controller.py # 撞击前的位置和速度控制
├── tasks/
│   └── impact_recovery.py      # APPROACH/RECOVERY 状态机和 PPO 奖励
├── train_ppo.py               # Stable-Baselines3 PPO 训练入口
├── evaluate_ppo.py            # 加载模型、统计评估和 MuJoCo 实时可视化
└── main.py                    # XML、质量和观测维度的最小检查
```

运行命令都应在下面的工程目录执行：

```powershell
$Project = 'C:\Users\lenovo\Desktop\bisai\rm\drone\mujoco\drone_xh1'
$Python = 'D:\Miniconda\Miniconda_install\envs\mujoco\python.exe'
Set-Location $Project
```

如果以后把环境安装到了别的 Conda 环境，只需把 `$Python` 改成该环境的 `python.exe`。

## 2. 当前模型的质量和 STL 对应关系

`drone_model/meshes` 中的五个 STL 与当前 XML 的分配如下：

| STL | 对应部件 | XML 质量 | 质量来源 |
|---|---|---:|---|
| `base_link.STL` | 机身主体（电池、飞控、机架等） | 0.159 kg | 用户给定的 159 g |
| `ra.STL` | 电机和桨叶组件 | 0.005 kg | 用户给定的 5 g |
| `la.STL` | 电机和桨叶组件 | 0.005 kg | 用户给定的 5 g |
| `rb.STL` | 电机和桨叶组件 | 0.005 kg | 用户给定的 5 g |
| `lb.STL` | 电机和桨叶组件 | 0.005 kg | 用户给定的 5 g |
| **合计** | XH1 整机 | **0.179 kg** | 159 g + 4 × 5 g |

STL 只描述几何形状，不含材料密度和质量属性。因此它们可以确认“哪个几何对应哪个部件”，不能单独证明 159 g 或 5 g。当前 `fullinertia` 是根据 URDF 的惯量按质量比例缩放得到的工程初值：机身按 `0.159 / 0.09818250698` 缩放，四个电机组件按 `0.005 / 0.09817725153` 缩放。这个假设等价于新增质量与原 URDF link 具有相同的质心和质量分布，最终应使用 CAD 质量属性或实测转动惯量复核。

检查模型总质量：

```powershell
& $Python main.py
```

预期至少看到 `observation shape` 为 `(13,)`，且质量断言通过。若实际整机称重不是 0.179 kg，应同时修改 `drone0.xml` 的质量/惯量和 `PropulsionConfig.mass`，不能只改其中一个。

## 3. 第一步：验证 Python 依赖和 XML

先确认使用的是同一个环境：

```powershell
& $Python -c "import mujoco, gymnasium, stable_baselines3; print('MuJoCo', mujoco.__version__); print('Gymnasium', gymnasium.__version__); print('SB3', stable_baselines3.__version__)"
```

加载 XML，并检查关键对象名称：

```powershell
@'
from pathlib import Path
import mujoco

xml = Path('drone_model/drone0.xml').resolve()
model = mujoco.MjModel.from_xml_path(str(xml))
print('xml:', xml)
print('nbody/ngeom/nsite/nsensor:', model.nbody, model.ngeom, model.nsite, model.nsensor)
print('total body mass:', float(model.body_mass.sum()))
for name in ('motor_0_site', 'motor_1_site', 'motor_2_site', 'motor_3_site',
             'imu_site', 'armor_collision'):
    for obj_type in (mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_GEOM):
        obj_id = mujoco.mj_name2id(model, obj_type, name)
        if obj_id >= 0:
            print(name, obj_type, obj_id)
            break
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
print('motor site world positions:')
for i in range(4):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f'motor_{i}_site')
    print(i, data.site_xpos[site_id].copy())
'@ | & $Python -
```

必须满足：

1. XML 可以加载，没有重复的 body、geom、site、sensor 名称。
2. 总质量为 `0.179 kg`，或与已经确认的实测整机质量一致。
3. 四个 `motor_*_site` 存在并位于四个桨轴中心。
4. `imu_site`、`imu_accel`、`imu_gyro` 存在。
5. `armor_collision` 存在，且装甲板碰撞体使用 `contype="2"`、无人机碰撞体使用 `conaffinity="1"`。
6. 视觉 geom 的 `contype="0"`、`conaffinity="0"`，不会重复参与碰撞。

当前 XML 仍使用 CAD STL 作为碰撞几何，并已加入 MuJoCo Z-up 地面：地面高度为 `z=0`，可视地面为纯白色，无人机碰撞体可以与 `ground_collision` 接触。验证物理参数后，建议将无人机碰撞体替换为低面数 convex mesh 或 box/capsule，保留视觉 STL；这样碰撞更稳定、仿真速度更高。

当前坐标约定为：世界坐标保持 MuJoCo 的 `Z-up`；机体坐标原点位于四旋翼 X 型中心，机体 `+X` 指向机头，机体 `+Z` 垂直向下，机体 `+Y` 按右手系确定。水平初始姿态下，机体 `+X` 映射到世界 `-X`，旋翼推力沿机体 `-Z` 施加。

2026-10-25 修正 CAD 模型的上下安装方向：`drone0` 的 X 轴安装角由 `-pi/2` 改为 `+pi/2`，实体相对原水平姿态绕 roll 轴翻转 `180°`；`imu_site` 的局部四元数同步改为 `0 0 0.707106781186548 -0.707106781186547`。默认 reset 仍使用零姿态偏置，机体坐标仍为前、右、下，推力仍朝世界上方。电机力臂和混控矩阵从修正后的 XML 位置重新计算。旧模型的碰撞分布随实体安装修正发生变化，之前的评估记录仅代表修正前场景。

2026-10-26 修正速度坐标处理：XML 的 `imu_vel` 是 `framelinvel`，直接提供 `imu_site` 测点的世界系线速度。因此 `env.velocity_world` 直接返回该传感器，新增 `env.velocity_body = R_body_to_world.T @ env.velocity_world`。每个物理子步向气动模块分别传入正确的世界系和机体系速度；`info` 同时保存 `velocity_world`、`velocity_body` 两个快照。接近控制器、奖励和恢复条件通过原有属性自动取得修正后的世界速度。`flow_velocity_proxy`（MuJoCo `velocimeter`）作为机体系速度的独立校验。本次没有改变 13 维 PPO 观测、动作接口、控制增益或物理参数；旧策略仍可加载，但运动轨迹和奖励会变化，修正前的评估记录不能作为修正后结果。速度转换回归检查可运行 `python -m unittest discover -s tests -v`。

## 4. 第二步：运行基础环境和碰撞检查

先运行基础环境的 reset 检查：

```powershell
& $Python main.py
```

再运行一次带装甲板接近过程的任务环境。这个检查使用零动作只是为了验证流程；在 `APPROACH` 阶段动作会被接近控制器覆盖：

```powershell
@'
from tasks.impact_recovery import ImpactRecoveryEnv

env = ImpactRecoveryEnv()
obs, info = env.reset(seed=0)
print('reset phase:', info['phase'], 'obs shape:', obs.shape)
for step in range(600):
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0, 0.0, 0.0])
    if info.get('impact_seen'):
        print('impact at control step:', step,
              'force:', info.get('impact_force'),
              'impulse:', info.get('impact_impulse'),
              'phase:', info.get('phase'))
        break
    if terminated or truncated:
        print('ended before impact:', terminated, truncated, info)
        break
else:
    raise RuntimeError('no armor impact detected in 600 control steps')
env.close()
'@ | & $Python -
```

正常情况下会看到阶段从 `APPROACH` 切换为 `RECOVERY`，并输出碰撞力。碰撞没有发生时，优先检查无人机初始位姿、装甲板位置、四个 site 坐标、碰撞 bit 和 STL 单位，不要先修改 PPO 奖励。

## 5. 第三步：理解观测、动作和控制链路

基础环境 `XH1Aviary` 的观测为 13 维：

```text
0:3    IMU 加速度计比力（机体系）
3:6    IMU 角速度（机体系）
6:9    重力方向（机体系）
9:13   上一时刻动作
```

动作是归一化的 4 维向量，范围为 `[-1, 1]`：

```text
action[0]    总推力，映射到 0 ~ max_total_thrust
action[1]    机体系滚转角速度目标
action[2]    机体系俯仰角速度目标
action[3]    机体系偏航角速度目标
```

动作不会直接写入 MuJoCo 的 `<actuator>`。当前链路为：

```text
PPO 动作
  → 总推力/角速度目标
  → 角速度控制器
  → X 型四旋翼混控
  → 电机一阶滞后和 RPM 限制
  → 四个 motor site 的推力与反扭矩
  → MuJoCo 刚体和接触求解
```

`xh1_aerodynamics.py` 在每个控制周期内额外施加：

- 与总旋翼转速相关的等效叶片阻力；
- 靠近地面时的地面效应；
- 可选世界坐标系风力；
- `downwash_force_body()` 提供的多无人机下洗力接口。

叶片阻力是转速相关的工程等效模型，与 MuJoCo-drones-gym 的 `MJC_DRAG` 思路一致；它不是逐片桨叶 CFD。当前单无人机环境默认启用叶片阻力和地面效应，风力默认关闭，下洗函数尚未接入多机循环。

## 6. 第四步：用台架数据标定推进和空气动力学参数

当前 `physics/xh1_propulsion.py` 中的参数是可运行的初值，不应直接用于实机：

```python
kf = 1.50e-7              # N/(rad/s)^2
km = 3.00e-9              # Nm/(rad/s)^2
max_rpm = 20_000
motor_time_constant = 0.025  # s
```

至少需要记录每个电机的 PWM/RPM、推力、反扭矩和阶跃响应：

1. 在推力台上固定一只电机和桨叶，测量稳态 `RPM` 与 `T`。
2. 对每个采样点计算 `kf = T / omega^2`，其中 `omega = 2*pi*RPM/60`。在工作区间拟合 `T = kf * omega^2`。
3. 测量反扭矩 `Q`，拟合 `km = Q / omega^2`。
4. 给电机一个转速阶跃，按 `rpm(t)=rpm_target+(rpm_initial-rpm_target)*exp(-t/tau)` 拟合 `motor_time_constant`。
5. 让整机悬停，验证四电机总推力是否接近 `m*g = 0.179*9.81 ≈ 1.756 N`，即每个电机约 `0.439 N`。
6. 通过水平运动或侧向风实验标定 `blade_drag_coeff`；通过不同离地高度测量悬停推力标定 `ground_effect_coeff`。

标定完成后，修改 `PropulsionConfig` 和 `AerodynamicsConfig` 的默认值，或者在创建环境时显式传入配置。例如：

```python
import numpy as np
from envs.xh1_aviary import XH1Aviary
from physics.xh1_propulsion import PropulsionConfig
from physics.xh1_aerodynamics import AerodynamicsConfig

propulsion = PropulsionConfig(
    mass=0.179,
    kf=YOUR_KF,
    km=YOUR_KM,
    max_rpm=YOUR_MAX_RPM,
    motor_time_constant=YOUR_MOTOR_TAU,
)
aerodynamics = AerodynamicsConfig(
    blade_drag_coeff=np.array([YOUR_DX, YOUR_DY, YOUR_DZ]),
    ground_effect_coeff=YOUR_GROUND_EFFECT,
)
env = XH1Aviary(
    propulsion_config=propulsion,
    aerodynamics_config=aerodynamics,
)
```

如果只修改了 XML 质量而没有同步 `PropulsionConfig.mass`，接近控制器的重力补偿和悬停转速都会错误。

## 7. 第五步：确认撞击前由谁控制无人机

当前训练方案采用两阶段控制：

```text
APPROACH：ApproachController 控制位置、速度和姿态，生成可重复撞击
RECOVERY：检测到 armor_collision 后，PPO 接管并输出四维恢复动作
```

这是训练阶段的推荐做法，因为它不把 APM、串口和 MAVLink 调度延迟混进 PPO 的每一步采样，也能让每次训练都在相近的撞击条件开始。`tasks/impact_recovery.py` 中的碰撞检测基于 MuJoCo 接触对和 `mj_contactForce()`；当前判据是装甲板与无人机碰撞几何之间出现正碰撞力。装甲板的视觉模型和碰撞体在整个 episode 中持续存在；撞击只切换控制权，装甲板的碰撞 bit 始终保持 XML 中的 `contype=2`、`conaffinity=1`，恢复阶段可以继续接触和再次碰撞。之前的碰撞后释放逻辑及其参数已经删除。

当前任务从装甲板的世界 `+X` 侧出发，沿世界 `-X` 飞向目标；这对应机体 `+X` 机头先撞。默认 IMU/四旋翼中心起点约为 `[1.6, 0, 0.35]`，接近目标约为 `[0.875, 0, 0.35]`。装甲板中心高度为 `0.35 m`，绕世界 Y 轴倾斜 `-27.5°`。该符号使装甲板的 `+X` 面朝向来袭方向；从 +X 侧观察，装甲板与地面围成的入射侧为 `117.5°` 钝角（装甲板平面与水平面的锐角为 `62.5°`）。如果修改了机体安装姿态或坐标系，必须重新确认 `approach_direction_world`，否则可能再次变成尾部撞击。

恢复阶段的“坠毁”是任务终止判据，不是 MuJoCo 自动生成的物理标签。当前代码以真实地面接触作为坠毁条件：

- 无人机碰撞几何与 `ground_collision` 接触，字段为 `ground_contact` 和 `crashed`；
- `low_altitude` 只作为高度诊断字段，不单独结束恢复。

恢复成功则要求姿态重力方向对齐、角速度小于 `1 rad/s`、世界速度小于 `2.5 m/s`、高度高于 `0.18 m`，并连续保持 `20` 个控制步。恢复阶段没有人为设置的 `recovery_timeout_steps`；Gym 的 `max_episode_steps` 只是防止仿真无限运行的全局安全上限，达到时返回 `truncated`，不标记为坠毁。

当前默认全局上限仍为 `1200` 个控制步，即总仿真时间 `6 s`。默认撞击约在 `1.2 s` 发生，因此此配置最多观测 `4.8 s` 的恢复过程；达到上限并不代表模型最终能够或不能够恢复。

2026-10-25 持续装甲碰撞复评（以下结果在本次 roll 安装修正之前取得）：`runs/xh1_ppo_bodyframe_fixed.zip` 记录的训练步数为 `1,000,448`。在 seed 100～104 的固定工况下均未恢复、未触地，最终达到全局上限。最终倾角约 `117.45°`、角速度范数 `1.047 rad/s`、高度 `0.251 m`；恢复阶段约 `98.3%` 的采样步仍接触装甲。末步策略要求满总推力并给出 `[1.84, 10, -8] rad/s` 角速度目标，表现为倾翻后持续抵着装甲板，尚未学会完成稳定恢复。完整 seed 100 轨迹抽样、模型 SHA256 和检查点信息见 `runs/xh1_ppo_bodyframe_fixed_persistent_eval.json`。这些数值仅对应报告内校验和标识的模型文件及当时场景。

当前控制链每 `5 ms` 更新一次：先用保存的 `VecNormalize` 统计量归一化 13 维输入，再由两层各 128 单元的 Tanh 策略网络输出 4 维动作。13 维输入是比力、角速度、机体系重力方向和上一动作；策略没有直接接收高度、平移速度或装甲相对位置。重力方向目前由仿真真实姿态计算。动作经过物理量映射、角速度比例控制与阻尼、混控、电机转速饱和和 `25 ms` 电机响应后，才在 MuJoCo 中产生推力与反扭矩。高度和速度虽然参与训练奖励及成功判断，但不直接作为策略输入；低速度或小角速度也不能单独代表恢复成功。

真实系统的控制权建议按同样的逻辑迁移：

1. 撞击前由 APM 的位置/速度控制或伴随计算机的接近控制器完成飞行和撞击。
2. 伴随计算机检测装甲板接触或 IMU 冲击事件。
3. 触发事件后切换到 PPO 推理，将 IMU 和光流观测输入策略。
4. 策略输出经过推力/角速度到电机的安全混控，再发送给飞控或电调接口。
5. PPO 异常、观测失真、通信中断或推力饱和时回退到经过验证的角速度/姿态控制器。

因此，当前仿真中的 `ApproachController` 是 APM/伴随计算机控制链路的替身；它不是最终的实机飞控实现。先用它验证恢复策略，随后再用 SITL/HIL 验证 MAVLink 交接，最后才接实机。

## 8. 第六步：先做 PPO 冒烟训练，再做正式训练

先用很短的训练确认环境、奖励和 SB3 接口：

```powershell
& $Python train_ppo.py --timesteps 1024 --output runs/smoke_xh1_ppo --seed 7
```

正式训练的起点可以使用 100 万步。当前任务保留装甲板持续碰撞；如果某次训练使用过碰撞后释放装甲的版本，它与当前物理条件不同，应先评估，再决定是否重新训练。下面使用新的输出名保存，便于保留原来的模型：

```powershell
& $Python train_ppo.py --timesteps 1000000 --output runs/xh1_ppo_persistent_armor --seed 7
```

训练脚本当前使用：

- `MlpPolicy`，策略网络和价值网络均为 `[128, 128]`；
- `VecNormalize`，对观测和奖励做归一化；
- `n_steps=1024`、`batch_size=256`、`gamma=0.995`；
- 输出 `runs/xh1_ppo_persistent_armor.zip` 和 `runs/xh1_ppo_persistent_armor_vecnormalize.pkl`。

不要在没有保存对应 `VecNormalize` 文件的情况下单独加载模型。推理时必须使用与训练相同的观测归一化统计量，并设置 `training=False`、`norm_reward=False`。可以新建一个临时 `eval_xh1.py`：

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tasks.impact_recovery import ImpactRecoveryEnv

env = DummyVecEnv([lambda: ImpactRecoveryEnv(render_mode="rgb_array")])
env = VecNormalize.load("runs/xh1_ppo_persistent_armor_vecnormalize.pkl", env)
env.training = False
env.norm_reward = False
model = PPO.load("runs/xh1_ppo_persistent_armor.zip", env=env)

obs = env.reset()
for _ in range(1200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done[0]:
        print(info[0])
        break
env.close()
```

评估时至少记录以下指标：是否检测到撞击、撞击力/冲量、恢复耗时、最大角速度、最低高度、恢复保持时间和 episode 是否坠毁。只看平均 reward 不能证明策略已经具备实机安全性。

工程中已经提供了 `evaluate_ppo.py`，可以直接加载本次训练产生的两个文件：

```text
runs/xh1_ppo_persistent_armor.zip
runs/xh1_ppo_persistent_armor_vecnormalize.pkl
```

先做无窗口评估，适合批量跑多个随机种子（`--no-dashboard` 关闭调试网页和逐步记录）：

```powershell
& $Python evaluate_ppo.py `
    --model runs/xh1_ppo_persistent_armor.zip `
    --vecnormalize runs/xh1_ppo_persistent_armor_vecnormalize.pkl `
    --headless --episodes 20 --seed 100 --no-realtime --no-dashboard
```

命令会逐个输出是否检测到撞击、是否恢复、是否坠毁、最大角速度和最低高度，并在最后输出恢复成功率。附加诊断包括结束原因 `end`、最终机体倾角、角速度、速度、推力、是否仍接触装甲板、恢复阶段接触比例和已经历的恢复时间。`end=time_limit` 表示达到全局步数上限，不能解释为已恢复或已坠毁。当前 reset 未使用随机扰动，确定性推理时不同 seed 仍会重复同一初始工况；多次相同结果不等于验证了多种碰撞条件。`--no-realtime` 只适用于无窗口统计评估，会尽可能快地运行。

## 9. 在 MuJoCo 可视化窗口中查看策略

在工程根目录执行：

```powershell
& $Python evaluate_ppo.py `
    --model runs/xh1_ppo_persistent_armor.zip `
    --vecnormalize runs/xh1_ppo_persistent_armor_vecnormalize.pkl `
    --episodes 3 --seed 100
```

脚本会：

1. 加载 PPO 策略和与它匹配的 `VecNormalize` 观测统计量；
2. 打开 MuJoCo 被动 viewer 窗口；
3. 使用外部自由相机显示无人机和装甲板；XML 中的 `flow_camera` 仅保留给未来光流传感器；
4. 显示接触点/接触力（当前 MuJoCo 版本支持时）；
5. 按 200 Hz 控制周期实时播放 `APPROACH → RECOVERY`；
6. 在控制台打印撞击时刻和最终恢复结果；
7. 默认启动本地实时调试网页，自动打开浏览器，并在 `runs/debug/` 保存每个控制步的输入输出。

窗口中可以观察：无人机先由 `ApproachController` 接近装甲板，发生碰撞后进入 `RECOVERY`，此时 PPO 输出才生效。默认相机距离为 1.8 m，鼠标拖动可以旋转/平移，滚轮可以缩放。也可以在启动时调整相机：

```powershell
# 更远的初始视角
& $Python evaluate_ppo.py --camera-distance 3.0 --episodes 1

# 将视线中心移到装甲板附近
& $Python evaluate_ppo.py --lookat-x 0.65 --lookat-y 0.0 --lookat-z 0.5 --episodes 1
```

关闭 viewer 窗口即可结束当前评估。默认情况下，交互式终端中的调试网页服务仍会保留，方便查看回放；在终端按 `Ctrl+C` 退出网页服务。添加 `--no-dashboard-wait` 可以让服务随评估结束而关闭；添加 `--no-dashboard` 可以恢复为只显示 MuJoCo 和终端统计。

如果窗口无法打开，先运行：

```powershell
& $Python -c "import mujoco.viewer; print('MuJoCo viewer import OK')"
```

如果导入正常但当前运行环境没有桌面显示，使用 `--headless`。如果模型能够加载却连续输出 `recovered=False` 或 `crashed=True`，这表示策略在当前碰撞分布下没有恢复成功，需要查看训练步数、奖励、撞击条件和物理参数，而不是重复加载模型。

`xh1_ppo_persistent_armor.zip` 与 `xh1_ppo_persistent_armor_vecnormalize.pkl` 必须来自同一次训练。更换模型时，命令行的两个路径也必须同时更换，例如：

```powershell
& $Python evaluate_ppo.py `
    --model runs/smoke_ppo_applyft.zip `
    --vecnormalize runs/smoke_ppo_applyft_vecnormalize.pkl `
    --headless --episodes 5 --no-realtime
```

### 9.1 实时查看 PPO 输入、动作与姿态响应

不需要安装 Flask、Node.js 或其他网页依赖，使用当前 `mujoco` Conda 环境即可。在工程目录运行你原来的命令：

```powershell
conda activate mujoco
cd C:\Users\lenovo\Desktop\bisai\rm\drone\mujoco\drone_xh1
python evaluate_ppo.py --model runs/xh1_ppo_bodyframe_fixed.zip --vecnormalize runs/xh1_ppo_bodyframe_fixed_vecnormalize.pkl --episodes 5 --seed 100
```

终端会显示 `PPO live dashboard: http://127.0.0.1:8765/`。浏览器会自动打开；如果没有打开，复制终端中的实际地址进入浏览器。8765 端口被占用时会自动选择其他空闲端口，以终端打印的地址为准。也可使用 `--dashboard-port 8766` 指定其他端口。网页与 MuJoCo 窗口可以并排查看。

网页提供：

- **本步实际控制者**：区分接近控制器与 PPO。首次检测到碰撞的那一步仍是接近控制器执行，下一控制步 PPO 才真正生效。
- **13 维模型输入**：比力 3 维、陀螺仪 3 维、机体系单位重力方向 3 维、上一生效 PPO 动作 4 维。左列显示原始观测，右列显示本步 `model.predict()` 真正接收的 `VecNormalize` 归一化输入；达到归一化裁剪界限的值标红。
- **4 维模型动作**：`[a_T, a_p, a_q, a_r]` 为 `model.predict()` 返回的 `[-1,1]` 动作，不是策略网络内部未裁剪的高斯均值。每维显示对应的推力 N 或目标角速度 rad/s，`|a| >= 0.99` 标记“近限幅”。接近阶段这些预测不生效，卡片会显示接近控制器的实际指令。
- **指令与实际响应曲线**：包括 PPO 归一化动作、总推力指令与旋翼实际合推力、三轴目标角速度与陀螺仪角速度、倾角和高度。浅红背景表示控制步末仍接触装甲板。
- **电机与恢复条件**：四个电机的目标 RPM、实际 RPM 和推力，装甲接触力、接触对，以及恢复的四个条件是否满足、连续满足步数、最终结束原因。

动作映射沿用当前训练环境：`T=(a_T+1)/2*T_max`，`[p,q,r]=[a_p,a_q,a_r]*rate_limit`。目前 `T_max` 约 `2.632 N`，角速度上限为 `[10,10,8] rad/s`，网页会读取当前推进参数，而不是固定使用这些数字。实际合推力可能与指令不同，因为中间还有角速度控制器、混控、限幅和电机滞后；它也不包含气动力或装甲接触力。

建议按以下顺序排查一次失败：

1. 点击 **定位 PPO 接管**，查看首次生效的动作，确认推力是否突然降到 0、角速度是否持续顶到上限。
2. 点击 **播放回放**，选择 `0.25×` 或 `0.5×`，对照目标角速度和实际角速度，观察策略有没有逐步消除转动。
3. 拖动时间轴或点击曲线，逐个控制步检查输入值是否因冲击异常变大、归一化输入是否被裁剪，动作是否一直来回振荡。曲线范围可以切到“当前附近 1 s”或“当前附近 0.2 s”，展开撞击瞬间的细节。
4. 查看装甲接触背景、倾角、恢复条件和最终结果，区分“持续贴板”“姿态仍倾翻”“实际角速度跟不上目标”“触地”和“达到全局步数上限”。网页只展示现有判据，尚未额外要求脱离装甲。
5. 用 **轮次** 下拉框回看其他 episode；用 **导出本轮 CSV** 导出本轮全部已接收控制步，或下载完整 JSONL 记录。

输入表对应推理前的 `t`，实际响应和姿态对应本步执行后的 `t+0.005 s`。网页暂停、慢放、拖动不会暂停或改变 MuJoCo，也不会改变策略动作；终端和网页的数据处理在不同线程中运行。页面约 10 Hz 更新，但会接收并记录每个 200 Hz 控制步，不是每秒只存 10 帧。后台文件写入存在短暂缓冲，评估正常结束或在终端按 `Ctrl+C` 时会完成写入。

### 9.2 保存、重新打开与批量评估

每次开启调试都会新建独立目录，不覆盖模型文件：

```text
runs/debug/eval_年月日_时分秒_微秒/
├── session.json       # 模型路径、观测顺序、控制频率、动作上限与恢复阈值
├── telemetry.jsonl    # 每个控制步的输入、输出、响应，以及各轮汇总
└── summary.json       # 各轮评估结果
```

关闭后重新查看任意一次记录：

```powershell
# 将下面路径替换为实际生成的 telemetry.jsonl 路径
python debug_dashboard.py --replay runs/debug/eval_年月日_时分秒_微秒/telemetry.jsonl
```

也可以在已经打开的调试网页中点击 **载入 JSONL 回放** 选择历史记录。载入只在本机网页中处理，不上传到外部服务。服务器用于网页请求的缓存最多保留最近 60000 帧（200 Hz 下约 300 秒），完整磁盘记录保留全部控制步；常用的 5×1200 步评估不触及该上限。网页会保留已经接收的帧；单次超长评估应保留原始 JSONL，网页载入会读取整个文件，需留意浏览器内存。

常用附加参数：

```powershell
# 保留网页，不自动打开浏览器（手动访问终端打印的 URL）
python evaluate_ppo.py --episodes 5 --no-browser

# 不打开 MuJoCo，仍保留网页与完整记录；可在网页中回放加速评估结果
python evaluate_ppo.py --headless --no-realtime --episodes 5 --no-browser

# 纯批量统计：关闭网页和逐步日志，结束即返回终端
python evaluate_ppo.py --headless --no-realtime --episodes 20 --no-dashboard

# 自动化脚本中开启记录，但评估后不等待交互退出
python evaluate_ppo.py --headless --no-realtime --episodes 5 --no-browser --no-dashboard-wait
```

这些命令省略模型路径时使用脚本默认的 `xh1_ppo_bodyframe_fixed` 模型与匹配归一化文件；更换策略时需同时指定 `--model` 和 `--vecnormalize`。非交互式终端不会在结束后等待；日志仍会保存，随后可用回放命令重新打开。

## 10. 第七步：把训练从固定撞击扩展为有效策略

修改撞击方向或装甲接触处理后，应使用当前任务重新评估保存的策略。PPO 的 ZIP 和归一化文件没有保存当时的环境源代码，仅凭文件名不能确定是否使用了碰撞后释放版本。恢复成功还需要检查是否持续依赖装甲板支撑：当前 `_is_recovered()` 尚未要求脱离装甲板，不能把贴板低速姿态直接视为空中稳定恢复。接近过程和装甲板位置基本固定，这适合检查代码链路，不足以证明 PPO 对真实撞击具有鲁棒性。完成冒烟训练后，应按以下顺序改造：

1. 在 `reset()` 中随机化撞击前的初始位置、初始姿态和初始速度。
2. 在接近控制器中随机化目标速度，得到不同撞击速度。
3. 在碰撞前或碰撞瞬间加入随机角速度、碰撞偏心量和冲量。
4. 随机化 `kf`、`km`、电机时间常数、传感器噪声、控制延迟和空气动力学系数。
5. 让训练集和验证集使用不同的随机种子、撞击速度区间和参数区间。
6. 调整奖励，避免策略通过“保持不动”或牺牲高度获得虚假的高分；将恢复时间、角速度、姿态误差和高度安全区间同时纳入指标。

每次只改变一个随机化因素，并保存固定种子的回归测试。否则很难判断性能变化来自策略还是来自任务分布变化。

## 11. 传感器和光流的当前边界

XML 已定义 `imu_accel`、`imu_gyro`、`imu_quat`、`imu_pos`、`imu_vel`、`imu_angvel`，PPO 当前实际使用加速度计、陀螺仪、重力方向和上一动作。`flow_velocity_proxy` 是机体系速度传感器，用于临时测试接口，不是真正的相机光流。

在接入 APM 或进行更真实的视觉仿真前，需要：

1. 确认相机的安装位置、朝向、视场角和畸变参数。
2. 从 `flow_camera` 的渲染图像计算光流，或接入经过验证的光流算法。
3. 将光流质量、有效像素比例和高度信息加入观测或 MAVLink 桥接。
4. 给 IMU 和光流加入与实机频率、噪声、延迟、丢帧一致的模型。

不要把 `flow_velocity_proxy` 的结果直接宣称为实机光流性能。

## 12. APM/MAVLink 接入顺序

当前工程没有 MAVLink 代码。建议按以下顺序接入：

1. **纯 MuJoCo 闭环**：完成 XML、推力、叶片阻力、碰撞检测、PPO 训练和随机化。
2. **离线接口测试**：把环境的 9 维原始传感器量、上一动作、碰撞标志和 PPO 输出记录成文件，单独测试推理延迟和动作范围。
3. **SITL/HIL**：让 APM 负责撞击前飞行；伴随计算机通过 MAVLink 接收 IMU/光流和状态，在碰撞后切换 PPO。先验证时间戳、坐标系、单位和模式切换。
4. **台架测试**：验证总推力、角速度目标、限幅、失联回退和电机方向；不要让无人机自由飞行。
5. **系留低速测试**：从低撞击速度开始，逐步扩大速度、姿态和碰撞冲量范围。

MAVLink 消息的具体选择取决于 APM 固件和控制模式。无论采用标准消息还是自定义桥接，都必须明确：加速度单位、角速度单位、机体系定义、NED/ENU 转换、时间戳、推力归一化范围和 PPO 接管/退出条件。当前 PPO 输出的是“总推力 + 机体系角速度目标”，不能未经混控直接当作四个电机 PWM。

## 13. 常见问题

**XML 找不到 mesh**：必须从 `drone_xh1` 根目录运行，或确保 XML 中的相对路径仍相对于 `drone_model/drone0.xml`。五个文件应位于 `drone_model/meshes/`。

**总质量断言失败**：检查 XML 中 `drone0`、`ra`、`la`、`rb`、`lb` 的质量，并同步修改 `PropulsionConfig.mass`。

**无人机没有推力或翻转**：检查 `motor_*_site` 的局部坐标、`imu_site` 的机体坐标四元数、body `-Z` 推力方向、电机顺序和 `spin_directions`。

**没有撞击事件**：检查装甲板位置、初始高度、碰撞 geom 的 contype/conaffinity、XML 单位和 `armor_collision` 名称。不要把无人机直接 teleport 到装甲板内部。

**训练出现 NaN 或动作爆炸**：先用 1024 步冒烟训练；检查 `kf`、`km`、`max_rpm`、惯量、碰撞刚度和奖励尺度，再逐步增加随机化。

**评估结果与训练结果不同**：检查是否加载了同名的 `*_vecnormalize.pkl`，并确认推理时关闭了 `VecNormalize.training` 和奖励归一化。

**渲染失败**：`render_mode` 只能使用 `None` 或 `"rgb_array"`。无窗口服务器上不要要求 `human` 模式；可先执行 `XH1Aviary(render_mode="rgb_array").render()` 检查离屏渲染。

## 14. 完成标准

在进入 APM 或实机前，至少应满足：

- 固定种子下 XML 加载、质量检查、reset、step 和碰撞检测全部通过；
- 推力台数据已经替换默认 `kf`、`km`、`max_rpm` 和电机时间常数；
- 叶片阻力和地面效应经过独立实验或仿真对比；
- PPO 在未参与训练的撞击速度、姿态、冲量和参数组合上仍能恢复；
- 记录了恢复成功率、恢复时间、最大角速度、最低高度和通信/控制延迟；
- 已有姿态/角速度安全回退控制器，并验证 PPO 越界、超时和失联行为；
- SITL/HIL 的坐标系、单位、时间戳和模式切换已经闭环验证。

只有完成这些检查后，才适合把策略从仿真迁移到系留实机测试。
