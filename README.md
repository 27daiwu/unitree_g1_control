# Unitree G1 29DoF 坐姿弹钢琴示教系统

技术路线为 Python + `unitree_sdk2py`。当前阶段只做 Subscriber、原始状态录制和离线分析；不使用 RL，不做 imitation、Replay 或运动控制。

机器人网络：PC1 `192.168.123.161`，PC2/NX `192.168.123.164`，开发端 `192.168.123.222`，接口 `eth0`。

已确认：`rt/lowstate`、29DoF joint mapping、`MotorState.temperature` 为原始 `int16[2]`。BMS/MainBoard topic 仍为 UNKNOWN，禁止猜测。

## 入口

Install the project from the repository root:

```bash
cd /home/hebe/unitree_g1_control
python3 -m pip install -e .
```

Then run:

```bash
python3 scripts/run_state_monitor.py --interface eth0
python3 scripts/record_state.py --interface eth0 --duration 10 --output data/raw/example.npz
python3 scripts/verify_recording.py data/raw/example.npz
python3 -m g1_piano.analysis.audit_recording_quality data/raw/example.npz
```

当前状态：State Monitor PASS，Recorder PASS，Replay 尚未开始。

Known issue: Matplotlib visualization currently unavailable due to local NumPy ABI mismatch.
