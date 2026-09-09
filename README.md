# Unitree G1 29DoF Teaching Toolkit

基于 Python 和 `unitree_sdk2py` 的 Unitree G1 29DoF 通用示教、录制与回放工具包。当前已完成左臂示教录制与真机 replay 闭环，仓库定位为可继续扩展的 general-purpose teaching / recording / playback toolkit。

## 当前能力与状态

长期保留四项能力：

1. Whole-body state recording：`LowState -> 29DoF NPZ`，仅被动录制。
2. Arm teaching recording：`UserCtrl`、低刚度 `q_target` follower、Pinocchio 重力补偿。
3. MuJoCo NPZ playback：可选的离线可视化与检查工具。
4. Real robot arm playback：`NPZ -> MOVE_TO_START -> SETTLE -> PLAYBACK -> HOLD_FINAL -> PASSIVE`。

```text
UPPER_BODY_TEACHING_CORE = PASS

WHOLE_BODY_STATE_RECORDING = PASS
USERCTRL_OWNERSHIP = PASS
SINGLE_MOTOR_CONTROL = PASS
SINGLE_JOINT_TEACH = PASS
LEFT_ARM_MULTI_JOINT_TEACH = PASS
PINOCCHIO_GRAVITY_COMPENSATION = PASS
LEFT_ARM_TEACH_RECORDING = PASS
LEFT_ARM_REAL_ROBOT_PLAYBACK = PASS
FULL_TRAJECTORY_PLAYBACK = PASS
SAFE_EXIT_TO_PASSIVE = PASS

CURRENT_NEXT = DUAL_ARM_GENERALIZATION
```

项目路线：

```text
Phase 1  Motor / UserCtrl validation   PASS
Phase 2  Teaching control              PASS
Phase 3  Teaching recording            PASS
Phase 4  Real robot playback            PASS
Phase 5  Dual-arm generalization        NEXT
Phase 6  Application layer
Phase 7  Piano workflow
```

MuJoCo is optional inspection tooling. Real robot playback does not require simulation validation first. 当前 playback gain 是 `DEFAULT_SAFE_PLAYBACK_GAIN`，属于安全默认值，不是某条轨迹的最佳参数。

## 环境

推荐使用 Python 3.11 Conda 环境，并确保 `unitree_sdk2py`、CycloneDDS、`numpy` 与可选的 `mujoco` 安装在同一环境：

```bash
conda create -n g1_control python=3.11
conda activate g1_control
cd ~/zjy_ws/src/unitree_g1_control
python3 -m pip install -e . --no-deps
```

如果切换 Conda、venv 或 Python 环境，需要在新环境重新执行 editable install。若系统级安装权限受限，可使用 `python3 -m pip install --user -e . --no-deps`。

## 项目结构

```text
src/g1_piano/          核心 Python 模块
scripts/               用户直接运行的命令行入口
assets/                MuJoCo G1 模型资源
data/state/raw/        原始 29DoF 状态录制
data/state/processed/  离线状态处理结果
data/teach/raw/        示教录制 NPZ
data/teach/processed/  示教处理结果
data/test/motor/       电机控制审计数据
tests/                 单元测试
docs/                  核心文档
docs/archive/          阶段性审计归档
archive/               历史参考工具
```

## 使用说明

首先激活环境并进入工程目录：

```bash
conda activate g1_control
cd ~/zjy_ws/src/unitree_g1_control
```

### 1. 29DoF 电机与 UserCtrl 诊断

默认只订阅并打印全部 29 个电机，不发送命令：

```bash
PYTHONPATH=src python3 scripts/test_motor_control.py --interface enp130s0 --dry-run
```

指定关节后，程序仍会先显示实时状态，并要求输入 `YES` 才执行平滑往返测试：

```bash
PYTHONPATH=src python3 scripts/test_motor_control.py \
    --interface enp130s0 --joint left_wrist_yaw
```

当前默认 backend 为 `userctrl`，使用 `rt/user_lowcmd`，并要求 `GetFsmId()` 返回 `1 / PASSIVE` 后才允许 ownership handoff。退出默认调用 `SwitchToInternalCtrl(PASSIVE)`。详见 `docs/MOTOR_CONTROL_AUDIT.md` 与 `docs/USERCTRL_OWNERSHIP_AUDIT.md`。

纯 ownership 验证不会执行位置偏移：

```bash
PYTHONPATH=src python3 scripts/test_motor_control.py \
    --interface enp130s0 --ownership-only --ownership-duration 2
```

### 2. Whole-body 状态录制

```bash
python3 scripts/record_state.py \
    --interface eth0 \
    --duration 20 \
    --output data/state/raw/g1_demo_001.npz

python3 scripts/verify_recording.py \
    data/state/raw/g1_demo_001.npz
```

录制内容包括 29DoF 的 `q`、`dq`、`tau_est`、motor temperature，以及 IMU、`mode_machine`、`tick` 和时间戳。看到 `validation: PASS` 表示 NPZ 结构和基本完整性检查通过。

### 3. 双臂示教录制与真机回放

```bash
python3 scripts/record_left_arm_teach.py --interface eth0 --arm left
# TEACH/record also support --arm right and --arm both
python3 scripts/playback_left_arm.py \
    --input data/teach/raw/20260908_left_arm_take001.npz \
    --interface eth0 --arm left

# 只做 NPZ 离线验证，不连接 DDS 或进入 UserCtrl
python3 scripts/playback_left_arm.py \
    --input data/teach/raw/right_arm_take001.npz \
    --validate-only --recorded-dq-limit 2.0
```

示教录制与真机回放支持 `--arm left|right|both`；回放未指定 `--arm` 时使用 NPZ 内部 `arm_mode`，显式参数与文件 metadata 不一致则在 UserCtrl 前拒绝。示教录制采用 UserCtrl、低刚度 `q_target` follower 和 Pinocchio 重力补偿。真机回放按 `MOVE_TO_START -> SETTLE -> PLAYBACK -> HOLD_FINAL -> PASSIVE` 生命周期执行。命令参数和安全确认以脚本 `--help` 及现场操作规程为准。

### 4. MuJoCo 离线播放

```bash
python3 scripts/play_npz_mujoco.py \
    data/state/raw/g1_demo_001.npz \
    --model assets/robots/unitree_g1/xmls/scene_g1.xml
```

播放器只执行 NPZ 到 MuJoCo 的离线动作复现，不连接真实机器人。常用控制：`Space` 播放/暂停，`Left`/`Right` 前后移动，`J`/`L` 后退/前进 1 秒，`Home`/`End` 跳到首帧/末帧，`R` 回到开头，`[`/`]` 调整速度，`Esc` 退出。

### 5. 状态数据手臂轨迹提取

```bash
python3 scripts/extract_arm_motion.py \
    data/state/raw/g1_demo_001.npz \
    --output data/state/processed/g1_demo_001_arms.npz
```

该步骤从 29DoF 状态数据中提取双臂 14DoF，不修改原始 NPZ。它是既有离线处理能力，不代表当前已完成双臂示教或双臂真机回放。

## 安全说明

真实机器人命令必须在清场、急停可用并由熟悉 Unitree G1 的操作者监督时运行。不要绕过脚本中的确认、ownership、状态机或退出流程。本仓库中的安全默认参数不构成对任意新轨迹的最佳参数保证。

## Known Issues

- editable install 可能受系统 Python 权限限制，可临时使用 `PYTHONPATH=$PWD/src`。
- Matplotlib 在部分环境存在 NumPy ABI 问题，不影响 MuJoCo 播放器核心路径。
- 不建议直接使用系统 Python 3.13，已有 CycloneDDS 二进制可能存在 ABI 不兼容。
