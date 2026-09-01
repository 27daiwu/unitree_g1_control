# Unitree G1 Piano Teaching

基于 Python 和 `unitree_sdk2py` 的 Unitree G1 29DoF 坐姿手臂示教、数据录制与 MuJoCo 离线回放工具。

## 当前功能

- G1 29DoF `LowState` 状态监控
- 29DoF DDS 原始状态录制为 NPZ
- 双臂 14DoF 无损提取
- NPZ 数据验证和质量检查
- MuJoCo 离线动作回放

## 环境

需要 Python 3.10，以及 `unitree_sdk2py`、`numpy`、`mujoco`。

首次进入仓库后，只需执行一次 editable install：

```bash
cd /home/hebe/zjy_ws/src/unitree_g1_control
python3 -m pip install -e . --no-deps
```

安装完成后无需再设置 `PYTHONPATH`，后续可以直接运行 `scripts/` 下的命令。

如果当前 Python 环境没有系统级安装权限，可使用：

```bash
python3 -m pip install --user -e . --no-deps
```

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

## 常用命令

```bash
cd /home/hebe/zjy_ws/src/unitree_g1_control
python3 scripts/run_state_monitor.py --interface eth0
python3 scripts/record_state.py --interface eth0 --duration 20 --output data/raw/g1_demo_001.npz
python3 scripts/verify_recording.py data/raw/g1_demo_001.npz
python3 scripts/extract_arm_motion.py data/raw/g1_demo_001.npz --output data/processed/g1_demo_001_arms.npz
python3 scripts/play_npz_mujoco.py data/raw/g1_demo_001.npz --model assets/robots/unitree_g1/xmls/scene_g1.xml
```

## 安全说明

当前播放器仅执行 NPZ 到 MuJoCo 的离线回放；真实机器人 Replay 和电机控制尚未开放。

## Known Issues

- 如果切换了 Python/Conda/venv 环境，需要在新环境中重新执行一次 `python3 -m pip install -e . --no-deps`。
- Matplotlib 在部分本机环境存在 NumPy ABI 问题，不影响 MuJoCo 播放器。
