# Unitree G1 Piano Teaching

基于 Python 和 `unitree_sdk2py` 的 Unitree G1 29DoF 坐姿手臂示教、数据录制与 MuJoCo 离线回放工具。

## 当前功能

- G1 29DoF `LowState` 状态监控
- 29DoF DDS 原始状态录制为 NPZ
- 双臂 14DoF 无损提取
- NPZ 数据验证和质量检查
- MuJoCo 离线动作回放

## 环境

需要 Python 3.10，以及 `unitree_sdk2py`、`numpy`、`mujoco`。从仓库根目录运行时设置：

```bash
export PYTHONPATH=$PWD/src
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
export PYTHONPATH=$PWD/src
python3 scripts/run_state_monitor.py --interface eth0
python3 scripts/record_state.py --interface eth0 --duration 20 --output data/raw/g1_demo_001.npz
python3 scripts/verify_recording.py data/raw/g1_demo_001.npz
python3 scripts/extract_arm_motion.py data/raw/g1_demo_001.npz --output data/processed/g1_demo_001_arms.npz
python3 scripts/play_npz_mujoco.py data/raw/g1_demo_001.npz --model assets/robots/unitree_g1/xmls/scene_g1.xml
```

## 安全说明

当前播放器仅执行 NPZ 到 MuJoCo 的离线回放；真实机器人 Replay 和电机控制尚未开放。

## Known Issues

- editable install 可能受系统 Python 权限限制，推荐使用 `PYTHONPATH=$PWD/src`。
- Matplotlib 在部分本机环境存在 NumPy ABI 问题，不影响 MuJoCo 播放器。
