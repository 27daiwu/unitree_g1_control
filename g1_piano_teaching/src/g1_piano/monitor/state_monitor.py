"""Read-only G1 state monitor using unitree_sdk2py DDS subscribers."""

import argparse
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from g1_piano.dds.watchdog import DdsWatchdog
from g1_piano.common.joint_map import G1_29DOF_JOINT_NAMES


class StateStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None
        self.watchdog = DdsWatchdog()

    def update(self, msg):
        with self.lock:
            self.latest = msg
            self.watchdog.received()

    def snapshot(self):
        with self.lock:
            return self.latest, self.watchdog.snapshot()


def format_temperature(value):
    return str(value)


def print_state(msg, dds_info):
    status, age, rate, count = dds_info
    age_text = "-" if age is None else f"{age * 1000.0:.1f} ms"
    print("\033[2J\033[H", end="")
    print("G1 STATE MONITOR (SUBSCRIBER ONLY)")
    print(f"DDS rt/lowstate: {status} | rate={rate:.1f} Hz | window_count={count} | age={age_text}")
    if msg is None:
        print("Waiting for rt/lowstate ...")
        return
    print(f"mode_machine={msg.mode_machine} | mode_pr={msg.mode_pr} | tick={msg.tick}")
    print(f"IMU quaternion={list(msg.imu_state.quaternion)}")
    print(f"IMU gyro={list(msg.imu_state.gyroscope)} accel={list(msg.imu_state.accelerometer)}")
    print(f"IMU rpy={list(msg.imu_state.rpy)} temperature={msg.imu_state.temperature}")
    print("idx joint                    q         dq        tau_est   temperature[2]")
    print("--- ----------------------- -------- --------- --------- ---------------")
    for i, name in enumerate(G1_29DOF_JOINT_NAMES):
        motor = msg.motor_state[i]
        print(f"{i:3d} {name:23s} {motor.q:8.4f} {motor.dq:9.4f} "
              f"{motor.tau_est:9.4f} {list(motor.temperature)}")
    print("BMS: UNKNOWN (no HG BMS DDS topic confirmed in local SDK; not subscribed)")
    print("MainBoardState: UNKNOWN (no DDS topic confirmed in local SDK; not subscribed)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="eth0", help="DDS network interface")
    parser.add_argument("--period", type=float, default=1.0, help="display period in seconds")
    args = parser.parse_args()

    print("Initializing DDS domain 0 on", args.interface)
    ChannelFactoryInitialize(0, args.interface)
    store = StateStore()
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(store.update, 10)
    print("Subscribed to rt/lowstate (LowState_); no write/control interfaces are used.")
    try:
        while True:
            msg, info = store.snapshot()
            print_state(msg, info)
            time.sleep(args.period)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
