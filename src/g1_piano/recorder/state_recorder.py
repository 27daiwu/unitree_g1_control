"""Record raw, read-only G1 29DoF state from rt/lowstate into NPZ."""

import argparse
import json
import os
import queue
import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from g1_piano.common.joint_map import G1_29DOF_JOINT_NAMES


class RawSampleQueue:
    def __init__(self, maxsize=20000):
        self.samples = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.received = 0
        self.queued = 0
        self.extraction_errors = 0
        self.lock = threading.Lock()

    def callback(self, msg):
        # Keep the DDS callback limited to timestamping, copying, and enqueueing.
        with self.lock:
            self.received += 1
        try:
            sample = (
                time.monotonic(), time.time(),
                tuple(float(msg.motor_state[i].q) for i in range(29)),
                tuple(float(msg.motor_state[i].dq) for i in range(29)),
                tuple(float(msg.motor_state[i].tau_est) for i in range(29)),
                tuple(tuple(int(v) for v in msg.motor_state[i].temperature) for i in range(29)),
                int(msg.mode_machine), tuple(float(v) for v in msg.imu_state.quaternion),
                tuple(float(v) for v in msg.imu_state.gyroscope),
                tuple(float(v) for v in msg.imu_state.accelerometer),
                tuple(float(v) for v in msg.imu_state.rpy),
                int(msg.imu_state.temperature), int(msg.tick), int(msg.mode_pr),
            )
        except Exception:
            with self.lock:
                self.extraction_errors += 1
            return
        try:
            self.samples.put_nowait(sample)
            with self.lock:
                self.queued += 1
        except queue.Full:
            with self.lock:
                self.dropped += 1

    def dropped_count(self):
        with self.lock:
            return self.dropped

    def counts(self):
        with self.lock:
            return self.received, self.queued, self.dropped, self.extraction_errors


def metadata(interface):
    return {
        "robot": "Unitree G1", "dof": 29,
        "joint_names": list(G1_29DOF_JOINT_NAMES),
        "dds_topic": "rt/lowstate", "network_interface": interface,
        "sdk": "unitree_sdk2py", "temperature_dtype": "int16[2]",
        "temperature_semantics": "UNKNOWN",
        "arm_indices_left": [15, 16, 17, 18, 19, 20, 21],
        "arm_indices_right": [22, 23, 24, 25, 26, 27, 28],
    }


def assemble(samples):
    cols = list(zip(*samples)) if samples else [[] for _ in range(15)]
    return {
        "timestamp_monotonic": np.asarray(cols[0], dtype=np.float64),
        "timestamp_wall": np.asarray(cols[1], dtype=np.float64),
        "q": np.asarray(cols[2], dtype=np.float32).reshape((-1, 29)),
        "dq": np.asarray(cols[3], dtype=np.float32).reshape((-1, 29)),
        "tau_est": np.asarray(cols[4], dtype=np.float32).reshape((-1, 29)),
        "temperature": np.asarray(cols[5], dtype=np.int16).reshape((-1, 29, 2)),
        "mode_machine": np.asarray(cols[6], dtype=np.uint8),
        "imu_quaternion": np.asarray(cols[7], dtype=np.float32).reshape((-1, 4)),
        "imu_gyroscope": np.asarray(cols[8], dtype=np.float32).reshape((-1, 3)),
        "imu_accelerometer": np.asarray(cols[9], dtype=np.float32).reshape((-1, 3)),
        "imu_rpy": np.asarray(cols[10], dtype=np.float32).reshape((-1, 3)),
        "imu_temperature": np.asarray(cols[11], dtype=np.int16),
        "tick": np.asarray(cols[12], dtype=np.uint32),
        "mode_pr": np.asarray(cols[13], dtype=np.uint8),
    }


def save_recording(output, data):
    if len(data["timestamp_monotonic"]) == 0:
        raise RuntimeError("recording contained no samples; NPZ was not written")
    try:
        np.savez_compressed(output, **data)
    except OSError as exc:
        raise RuntimeError(f"failed to write NPZ {output}: {exc}") from exc


def print_report(data, output, counts):
    ts = data["timestamp_monotonic"]
    dt = np.diff(ts)
    duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    finite = sum(int(np.isfinite(data[k]).sum()) for k in ("q", "dq", "tau_est"))
    total = sum(data[k].size for k in ("q", "dq", "tau_est"))
    print(f"output file: {output}")
    print(f"samples: {len(ts)}")
    print(f"recording duration: {duration:.6f} s")
    print(f"effective sample rate: {(len(ts)-1)/duration if duration > 0 else 0.0:.3f} Hz")
    for k in ("q", "dq", "tau_est", "temperature"):
        print(f"{k} shape: {data[k].shape}")
    print(f"timestamp monotonic violations: {int(np.count_nonzero(dt <= 0))}")
    print(f"NaN count: {sum(int(np.isnan(data[k]).sum()) for k in data if np.issubdtype(data[k].dtype, np.floating))}")
    print(f"Inf count: {sum(int(np.isinf(data[k]).sum()) for k in data if np.issubdtype(data[k].dtype, np.floating))}")
    print(f"tick first: {int(data['tick'][0]) if len(ts) else 'N/A'}")
    print(f"tick last: {int(data['tick'][-1]) if len(ts) else 'N/A'}")
    print(f"tick non-monotonic count: {int(np.count_nonzero(np.diff(data['tick'].astype(np.int64)) < 0))}")
    received, queued, dropped, extraction_errors = counts
    print(f"DDS callback received/queued/written/dropped/extraction_errors: {received}/{queued}/{len(ts)}/{dropped}/{extraction_errors}")
    print(f"sample count closure: {received == queued + dropped + extraction_errors and queued == len(ts)}")
    if len(dt):
        p = np.percentile(dt, [0, 50, 95, 99, 100])
        print("dt min/mean/median/p95/p99/max: "
              f"{dt.min():.9f} / {dt.mean():.9f} / {p[1]:.9f} / "
              f"{p[2]:.9f} / {p[3]:.9f} / {dt.max():.9f} s")
    else:
        print("dt min/mean/median/p95/p99/max: N/A")
    print(f"finite q/dq/tau_est: {finite}/{total}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--queue-size", type=int, default=20000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.queue_size <= 0:
        parser.error("--queue-size must be positive")

    if os.path.exists(args.output) and not args.overwrite:
        parser.error(f"output already exists: {args.output}; use --overwrite to replace it")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    samples = RawSampleQueue(args.queue_size)
    ChannelFactoryInitialize(0, args.interface)
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(samples.callback, 10)
    print(f"Recording rt/lowstate for {args.duration:.3f}s -> {args.output}")

    deadline = time.monotonic() + args.duration
    drained = []
    try:
        while time.monotonic() < deadline:
            try:
                drained.append(samples.samples.get(timeout=0.05))
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        print("Interrupted; saving samples collected so far.")
    finally:
        subscriber.Close()
    while True:
        try:
            drained.append(samples.samples.get_nowait())
        except queue.Empty:
            break

    data = assemble(drained)
    data["metadata"] = np.asarray(json.dumps(metadata(args.interface), sort_keys=True))
    save_recording(args.output, data)
    print_report(data, args.output, samples.counts())


if __name__ == "__main__":
    main()
