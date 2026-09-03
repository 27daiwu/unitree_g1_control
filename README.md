# Unitree G1 Piano Teaching

基于 Python 和 `unitree_sdk2py` 的 Unitree G1 29DoF 坐姿手臂示教、数据录制与 MuJoCo 离线回放工具。

## 当前功能

- G1 29DoF `LowState` 状态监控
- 29DoF DDS 原始状态录制为 NPZ
- 双臂 14DoF 无损提取
- NPZ 数据验证和质量检查
- MuJoCo 离线动作回放

## 环境

推荐使用 Python 3.11 Conda 环境。请先创建并激活环境：

```bash
conda create -n g1_control python=3.11
conda activate g1_control
```

然后请用户自行按照 Unitree SDK 的安装说明，在该环境中安装 `unitree_sdk2py`（及其 CycloneDDS 依赖），并确保 `numpy`、`mujoco` 已安装。最后在工程根目录执行一次 editable install：
需要 Python 3.10，以及 `unitree_sdk2py`、`numpy`、`mujoco`。

首次进入仓库后，只需执行一次 editable install：

```bash
cd /home/hebe/zjy_ws/src/unitree_g1_control
python3 -m pip install -e . --no-deps
```

安装完成后无需再设置 `PYTHONPATH`，后续可以直接运行 `scripts/` 下的命令。

如果当前 Python 环境没有系统级安装权限，可使用：

```bash
cd ~/zjy_ws/src/unitree_g1_control
python3 -m pip install -e . --no-deps
python3 -m pip install --user -e . --no-deps
```

`unitree_sdk2py` 与 CycloneDDS 必须安装在同一个 Conda 环境中。首次进入该环境后执行一次 editable install，之后可以直接运行 `scripts/` 下的程序，不需要设置 `PYTHONPATH`。如果切换 Conda、venv 或 Python 环境，请在新环境重新执行一次 editable install。

## 项目结构

```text
src/g1_piano/       核心 Python 模块
scripts/             用户直接运行的命令行入口
assets/              MuJoCo G1 模型资源
data/raw/            原始 29DoF 录制
data/processed/      离线处理结果
tests/               单元测试
docs/                审计和开发文档
archive/             历史参考工具
```

## 使用说明

首先激活推荐环境并进入工程目录（项目只需安装一次）：

```bash
conda activate g1_control
cd ~/zjy_ws/src/unitree_g1_control
```

### 1. 录制 G1 状态

机器人与开发机网络连接正常后运行：

```bash
python3 scripts/record_state.py \
    --interface eth0 \
    --duration 20 \
    --output data/raw/g1_demo_001.npz
```

`--interface` 指 DDS 使用的网卡，`--duration` 是录制秒数，`--output` 是 NPZ 输出路径。录制内容包括 29DoF 的 `q`、`dq`、`tau_est`、motor temperature，以及 IMU、`mode_machine`、`tick` 和时间戳。

录制完成后立即验证：

```bash
python3 scripts/verify_recording.py \
    data/raw/g1_demo_001.npz
```

看到 `validation: PASS` 即表示 raw NPZ 的结构和基本完整性检查通过。

### 2. MuJoCo 离线播放

播放器使用录制中的 `q[:, 0:29]` 驱动 MuJoCo G1 29DoF 模型：

```bash
python3 scripts/play_npz_mujoco.py \
    data/raw/g1_demo_001.npz \
    --model assets/robots/unitree_g1/xmls/scene_g1.xml
```

播放器只执行 NPZ 到 MuJoCo 的离线动作复现，不连接真实机器人，也不会发送电机命令。

常用控制：`Space` 播放/暂停，`Left`/`Right` 前后移动，`J`/`L` 后退/前进 1 秒，`Home`/`End` 跳到首帧/末帧，`R` 回到开头，`[`/`]` 调整速度，`Esc` 退出。

可按时间范围和速度播放，也可循环或选择线性采样：

```bash
python3 scripts/play_npz_mujoco.py \
    data/raw/g1_demo_001.npz \
    --model assets/robots/unitree_g1/xmls/scene_g1.xml \
    --start 5 --end 10 --speed 0.5 --sampling linear --loop
```

### 3. 双臂数据提取

```bash
python3 scripts/extract_arm_motion.py \
    data/raw/g1_demo_001.npz \
    --output data/processed/g1_demo_001_arms.npz
```

该步骤从 29DoF raw 数据中提取双臂 14DoF：左臂为 joint `15..21`，右臂为 joint `22..28`，不会修改原始 NPZ。

## 安全说明

当前播放器仅执行 NPZ 到 MuJoCo 的离线回放；真实机器人 Replay 和电机控制尚未开放。

## Known Issues

- editable install 可能受系统 Python 权限限制，推荐使用 `PYTHONPATH=$PWD/src`。
- Matplotlib 在部分本机环境存在 NumPy ABI 问题，不影响 MuJoCo 播放器。
- 不建议直接使用当前系统 Python 3.13，因为已有 CycloneDDS 二进制可能存在 ABI 不兼容。
