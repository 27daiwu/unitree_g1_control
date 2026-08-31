import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelSubscriber,
)

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


received_count = 0
last_received_time = 0.0


def lowstate_handler(msg: LowState_):
    global received_count, last_received_time

    received_count += 1
    last_received_time = time.time()

    # 每 500 帧打印一次，避免刷屏
    if received_count % 500 == 0:
        print("\n========== LOWSTATE ==========")
        print(f"received_count : {received_count}")
        print(f"mode_machine   : {msg.mode_machine}")
        print(f"imu rpy        : {msg.imu_state.rpy}")

        motor0 = msg.motor_state[0]

        print(f"motor[0].q     : {motor0.q}")
        print(f"motor[0].dq    : {motor0.dq}")
        print(f"motor[0].tau   : {motor0.tau_est}")

        print("==============================\n")


def main():
    global last_received_time

    print("======================================")
    print("G1 LowState READ-ONLY DDS test")
    print("network : eth0")
    print("topic   : rt/lowstate")
    print("type    : LowState_")
    print("NO publisher / NO motor control")
    print("======================================")

    # DDS Domain 0，绑定机器人网卡 eth0
    ChannelFactoryInitialize(0, "eth0")

    subscriber = ChannelSubscriber(
        "rt/lowstate",
        LowState_
    )

    subscriber.Init(lowstate_handler, 10)

    print("Waiting for LowState...")

    start_time = time.time()

    try:
        while True:
            time.sleep(1)

            now = time.time()

            if received_count == 0:
                print("DDS WAITING: no LowState received")

            else:
                age = now - last_received_time
                elapsed = now - start_time
                avg_hz = received_count / elapsed

                print(
                    f"DDS OK | "
                    f"count={received_count} | "
                    f"avg_rate={avg_hz:.1f} Hz | "
                    f"last_age={age:.3f}s"
                )

                if age > 1.0:
                    print("WARNING: LowState timeout > 1s")

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()