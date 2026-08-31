# Architecture

```text
rt/lowstate -> DDS Subscriber -> complete 29DoF raw state
             -> bounded queue -> NPZ recording -> offline verification -> quality audit
```

DDS callback 只做 monotonic/wall timestamp、字段复制和非阻塞入队。文件写入、统计和异常处理在主线程完成。未来可从 29DoF 原始数据选择双臂 14DoF，再做关键帧和轨迹研究；本阶段不实现这些步骤。
