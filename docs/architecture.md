# Architecture

```text
rt/lowstate
    ↓
29DoF Raw Recorder
    ↓
Raw NPZ
    ├── Arm Extractor → 14DoF Arm NPZ
    └── MuJoCo Simulator → Offline Playback
```

DDS 回调只负责采集时间戳、复制字段并非阻塞入队；文件写入、统计和异常处理在主线程完成。原始 NPZ 是后续离线处理的唯一数据源，双臂提取保持对应关节数据无损。

后续将增加关键帧、轨迹编辑和安全约束后的真实机器人 Replay。
