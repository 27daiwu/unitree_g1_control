#!/usr/bin/env python3
"""G1 29-DoF lowstate diagnostic and one-motor, small-step test.

The default mode is read-only.  A control run requires --motor/--joint and an
interactive ``YES`` confirmation.  LowCmd is a complete-frame interface in
the HG SDK: every frame contains 35 MotorCmd slots.  This tool writes the
selected slot's test command and explicitly holds the other 28 slots at their
latest measured q (kp=kd=tau=0), rather than relying on zero-initialized data.
"""
import argparse
import math
import signal
import sys
import threading
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import traceback

from g1_piano.common.joint_map import (
    G1_29DOF_JOINT_NAMES, INDEX_TO_GROUP, resolve_joint,
)

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, MotorCmd_, LowState_, MotorState_
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.g1.loco.g1_loco_api import InternalFsmMode
except ImportError as exc:  # permit --help and offline tests without SDK
    ChannelFactoryInitialize = ChannelPublisher = ChannelSubscriber = LowCmd_ = MotorCmd_ = LowState_ = MotorState_ = LocoClient = InternalFsmMode = None
    _SDK_ERROR = exc


class State:
    def __init__(self):
        self.lock = threading.Lock(); self.msg = None; self.received_at = 0.0
    def update(self, msg):
        with self.lock: self.msg, self.received_at = msg, time.monotonic()
    def snapshot(self):
        with self.lock: return self.msg, self.received_at


class TestLogger:
    def __init__(self, args, label):
        root = Path(args.log_dir); root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = (root / f"{stamp}_motor_control_{label}.log").resolve()
        self.json_path = self.log_path.with_suffix(".json")
        self.file = self.log_path.open("w", encoding="utf-8")
        self.last_console = None
        self.file.write(f"START timestamp={datetime.now().isoformat()}\nargs={vars(args)}\n")
    def emit(self, level, message, console=False):
        self.file.write(f"{datetime.now().isoformat()} [{level}] {message}\n"); self.file.flush()
        if console and message != self.last_console:
            print(message); self.last_console = message
    def debug(self, message): self.emit("DEBUG", message)
    def info(self, message, console=True): self.emit("INFO", message, console)
    def warning(self, message, console=True): self.emit("WARNING", message, console)
    def error(self, message, console=True): self.emit("ERROR", message, console)
    def result(self, message, console=True): self.emit("RESULT", message, console)
    def close(self): self.file.close()


def finite_motor(m):
    values = (m.q, m.dq, getattr(m, "tau_est", 0.0))
    return all(math.isfinite(float(v)) for v in values)


def print_diagnostic(msg, logger, selected=None, initial=None, target=None, kp=0.0, kd=0.0):
    rows = ["idx joint                    group      q         dq        tau_est temperature"]
    for i, name in enumerate(G1_29DOF_JOINT_NAMES):
        m = msg.motor_state[i]
        rows.append(f"{i:3d} {name:23s} {INDEX_TO_GROUP[i]:9s} {float(m.q):8.4f} {float(m.dq):9.4f} "
                    f"{float(getattr(m, 'tau_est', 0.0)):9.4f} {list(getattr(m, 'temperature', []))}")
    if selected is not None:
        m = msg.motor_state[selected]
        rows.append(f"MOTOR_INDEX={selected} JOINT_NAME={G1_29DOF_JOINT_NAMES[selected]} "
              f"Q_INITIAL={initial:.5f} Q_ACTUAL={float(m.q):.5f} Q_TARGET={target:.5f} "
              f"DQ={float(m.dq):.5f} POSITION_ERROR={target-float(m.q):.5f} KP={kp:.3f} KD={kd:.3f}")
    logger.debug("\n".join(rows))


def set_field(obj, name, value):
    if hasattr(obj, name): setattr(obj, name, value)


def create_motor_cmd():
    """Construct the local HG MotorCmd_ using its generated-IDL signature."""
    return MotorCmd_(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)


def create_lowcmd():
    """Construct a complete local HG LowCmd_ (35 MotorCmd_ slots)."""
    motors = [create_motor_cmd() for _ in range(35)]
    return LowCmd_(0, 0, motors, [0, 0, 0, 0], 0)


def make_command(msg, index, q_target, kp, kd):
    cmd = create_lowcmd()
    # Explicitly fill all slots. HG LowCmd has 35 slots although G1 uses 29.
    slots = getattr(cmd, "motor_cmd", getattr(cmd, "motorCmd", []))
    if callable(slots): slots = slots()
    for i in range(min(35, len(slots))):
        measured = float(msg.motor_state[i].q) if i < 29 else 0.0
        set_field(slots[i], "q", measured); set_field(slots[i], "dq", 0.0)
        set_field(slots[i], "tau", 0.0); set_field(slots[i], "kp", 0.0); set_field(slots[i], "kd", 0.0)
        set_field(slots[i], "mode", 0)
    m = slots[index]
    set_field(m, "q", float(q_target)); set_field(m, "kp", float(kp)); set_field(m, "kd", float(kd))
    set_field(m, "mode", 1)
    return cmd


def make_ownership_command(msg, kd):
    cmd = create_lowcmd()
    slots = getattr(cmd, "motor_cmd", getattr(cmd, "motorCmd", []))
    if callable(slots): slots = slots()
    for i in range(min(29, len(slots))):
        set_field(slots[i], "q", float(msg.motor_state[i].q))
        set_field(slots[i], "dq", 0.0); set_field(slots[i], "kp", 0.0)
        set_field(slots[i], "kd", float(kd)); set_field(slots[i], "tau", 0.0)
    return cmd


def all_q_dq(msg):
    return ([float(msg.motor_state[i].q) for i in range(29)],
            [float(msg.motor_state[i].dq) for i in range(29)])


def fsm_name(fsm_id):
    return {0: "ZERO_TORQUE", 1: "PASSIVE", 3: "SIT", 500: "WALKRUN", 706: "SQUAT"}.get(fsm_id, "UNKNOWN")

ARM_LIMITS = {
    15: (-3.0892, 2.6704), 16: (-1.5882, 2.2515), 17: (-2.618, 2.618), 18: (-1.0472, 2.0944),
    19: (-1.97222, 1.97222), 20: (-1.61443, 1.61443), 21: (-1.61443, 1.61443),
    22: (-3.0892, 2.6704), 23: (-2.2515, 1.5882), 24: (-2.618, 2.618), 25: (-1.0472, 2.0944),
    26: (-1.97222, 1.97222), 27: (-1.61443, 1.61443), 28: (-1.61443, 1.61443),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interface", default="eth0"); p.add_argument("--motor", help="single index 0..28")
    p.add_argument("--joint", help="single exact joint name"); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--delta", type=float, default=0.05); p.add_argument("--kp", type=float, default=60.0)
    p.add_argument("--kd", type=float, default=1.5); p.add_argument("--duration", type=float, default=1.0)
    p.add_argument("--hold", type=float, default=1.0); p.add_argument("--rate", type=float, default=100.0)
    p.add_argument("--max-target-velocity", type=float, default=0.05)
    p.add_argument("--timeout", type=float, default=1.0); p.add_argument("--max-delta", type=float, default=0.05)
    p.add_argument("--backend", choices=("userctrl", "legacy_lowcmd"), default="userctrl",
                   help="userctrl is default; legacy_lowcmd is experimental and explicit")
    p.add_argument("--exit-mode", choices=("passive", "last"), default="passive")
    p.add_argument("--ownership-only", action="store_true",
                   help="handoff to UserCtrl and back without any position displacement")
    p.add_argument("--ownership-duration", type=float, default=2.0,
                   help="seconds to continuously publish safe frames in ownership-only mode")
    p.add_argument("--log-dir", default="logs/test")
    args = p.parse_args()
    if args.motor and args.joint: p.error("use exactly one of --motor or --joint")
    if args.delta <= 0 or args.delta > 0.05: p.error("delta must be >0 and <= 0.05 rad")
    if args.max_target_velocity <= 0: p.error("max-target-velocity must be positive")
    if args.kp < 0 or args.kd < 0 or args.duration <= 0 or args.rate <= 0 or args.ownership_duration <= 0: p.error("invalid control limits")
    if args.ownership_only and args.backend != "userctrl": p.error("--ownership-only requires --backend userctrl")
    if ChannelFactoryInitialize is None:
        p.error(f"unitree_sdk2py unavailable: {_SDK_ERROR}")
    index = resolve_joint(args.motor or args.joint) if (args.motor or args.joint) else None
    if index is not None and index < 15 and not args.ownership_only:
        p.error("single-motor perturbation is restricted to arm joints 15..28; use ownership-only or a future base-hold test")
    label = "ownership_only" if args.ownership_only else (f"motor{index}_{G1_29DOF_JOINT_NAMES[index]}" if index is not None else "monitor")
    logger = TestLogger(args, label)
    print(f"LOG_FILE={logger.log_path}")
    ChannelFactoryInitialize(0, args.interface)
    state = State(); sub = ChannelSubscriber("rt/lowstate", LowState_); sub.Init(state.update, 10)
    command_topic = "rt/user_lowcmd" if args.backend == "userctrl" else "rt/lowcmd"
    logger.info(f"CONTROL_BACKEND={args.backend} COMMAND_TOPIC={command_topic}")
    logger.info("USERCTRL_API_AVAILABLE=" + ("YES" if LocoClient is not None else "NO"))
    logger.info("STATE = WAIT_LOWSTATE")
    if (index is None and not args.ownership_only) or args.dry_run:
        try:
            while True:
                msg, _ = state.snapshot()
                if msg: print_diagnostic(msg, logger)
                time.sleep(1.0)
        except KeyboardInterrupt: logger.info("STATE = SAFE_EXIT")
        logger.close()
        return
    while state.snapshot()[0] is None: time.sleep(0.01)
    msg, received = state.snapshot()
    if time.monotonic() - received > args.timeout:
        logger.error("LOWSTATE_VALID=NO; STATE = SAFE_EXIT"); logger.close(); return
    q_before, dq_before = all_q_dq(msg)
    temperature_available = all(hasattr(msg.motor_state[i], "temperature") and
                                any(math.isfinite(float(v)) for v in getattr(msg.motor_state[i], "temperature", []))
                                for i in range(29))
    logger.info(f"TEMPERATURE_AVAILABLE={'YES' if temperature_available else 'NO'}")
    initial = q_before[index] if index is not None else 0.0
    direction = 1.0
    if index is not None and not args.ownership_only:
        lo, hi = ARM_LIMITS[index]; margin = 0.02
        if initial + args.delta > hi - margin:
            if initial - args.delta >= lo + margin: direction = -1.0
            else: p.error("both perturbation directions violate joint safety margin")
    target = initial + direction * args.delta
    loco = None
    if args.backend == "userctrl":
        loco = LocoClient(); loco.Init()
        code, fsm_id = loco.GetFsmId()
        current_fsm_name = fsm_name(fsm_id)
        logger.info(f"CURRENT_FSM_ID={fsm_id} CURRENT_FSM_NAME={current_fsm_name} LOWSTATE_VALID=YES")
        if code != 0 or fsm_id != 1:
            logger.warning("Robot must enter PASSIVE/DAMPING before user control."); logger.close()
            return
    else:
        print("legacy_lowcmd is EXPERIMENTAL; it is never combined with userctrl.")
    logger.info("STATE = CAPTURE_INITIAL")
    if args.ownership_only:
        logger.info("OWNERSHIP_ONLY: all q_target=q_actual_at_entry, kp=0, kd=damping, tau_ff=0")
    else:
        logger.info(f"TARGET {G1_29DOF_JOINT_NAMES[index]} (motor {index}), q={initial:.5f} -> {target:.5f}, delta={args.delta}, Kp={args.kp}, Kd={args.kd}")
    try:
        confirmed = input("Type YES to enable control: ") == "YES"
    except (EOFError, KeyboardInterrupt):
        confirmed = False
    if not confirmed: logger.info("STATE = SAFE_EXIT"); logger.close(); return
    logger.info("STATE = ENABLE_CONTROL")
    first_frame_sent = False
    ownership_acquired = False
    try:
        pub = ChannelPublisher(command_topic, LowCmd_); pub.Init()
    except Exception as exc:
        logger.error(f"SAFE_EXIT_REASON={type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        logger.error("FIRST_USER_LOWCMD_SENT=NO SWITCH_TO_USER_CTRL_EXECUTED=NO OWNERSHIP_ACQUIRED=NO OWNERSHIP_RELEASE_REQUIRED=NO")
        print(f"TEST_RESULT=FAIL\nSAFE_EXIT_REASON={type(exc).__name__}: {exc}\nLOG_FILE={logger.log_path}"); logger.close()
        return
    switched = False
    if loco is not None:
        print("PREPARE_INITIAL_USER_LOWCMD=YES")
        # First safe frame is prepared and published before ownership handoff.
        try:
            pub.Write(make_ownership_command(msg, args.kd) if args.ownership_only else make_command(msg, index, initial, 0.0, args.kd))
            first_frame_sent = True
            result = loco.SwitchToUserCtrl()
        except Exception as exc:
            logger.error(f"SAFE_EXIT_REASON={type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            logger.error(f"FIRST_USER_LOWCMD_SENT={'YES' if first_frame_sent else 'NO'} SWITCH_TO_USER_CTRL_EXECUTED=UNKNOWN OWNERSHIP_ACQUIRED=NO OWNERSHIP_RELEASE_REQUIRED=NO")
            print(f"TEST_RESULT=FAIL\nSAFE_EXIT_REASON={type(exc).__name__}: {exc}\nLOG_FILE={logger.log_path}"); logger.close()
            return
        logger.info(f"USERCTRL_SWITCH_RESULT={result}")
        switched = result == 0
        ownership_acquired = switched
        if not switched:
            logger.error("STATE = SAFE_EXIT"); logger.close(); return
    else:
        logger.info("USERCTRL_SWITCH_RESULT=NOT_APPLICABLE")
    stop = threading.Event(); signal.signal(signal.SIGINT, lambda *_: stop.set())
    period = 1.0 / args.rate; started = time.monotonic(); phase = "OWNERSHIP_HOLD" if args.ownership_only else "MOVING"
    q_command = initial
    max_delta = [0.0] * 29; max_dq = [0.0] * 29; lowstate_timeout = False; control_error = False
    try:
        while not stop.is_set():
            msg, received = state.snapshot(); now = time.monotonic()
            if msg is None or now - received > args.timeout:
                lowstate_timeout = True; raise RuntimeError("lowstate timeout")
            if args.ownership_only:
                for i in range(29):
                    if not finite_motor(msg.motor_state[i]): raise RuntimeError("invalid motor state")
                    max_delta[i] = max(max_delta[i], abs(float(msg.motor_state[i].q) - q_before[i]))
                    max_dq[i] = max(max_dq[i], abs(float(msg.motor_state[i].dq)))
                pub.Write(make_ownership_command(msg, args.kd))
                logger.info("STATE = OWNERSHIP_HOLD")
                if now - started >= args.ownership_duration: break
                time.sleep(period); continue
            m = msg.motor_state[index]
            if not finite_motor(m) or abs(float(m.q)) > 10 or abs(float(m.dq)) > 100: raise RuntimeError("invalid motor state")
            for i in range(29):
                max_delta[i] = max(max_delta[i], abs(float(msg.motor_state[i].q) - q_before[i]))
                max_dq[i] = max(max_dq[i], abs(float(msg.motor_state[i].dq)))
            elapsed = now - started
            if elapsed < args.duration: desired = initial + (target - initial) * (elapsed / args.duration); phase = "MOVING"
            elif elapsed < args.duration + args.hold: desired = target; phase = "HOLDING"
            elif elapsed < 2 * args.duration + args.hold: desired = target + (initial - target) * ((elapsed - args.duration - args.hold) / args.duration); phase = "RETURNING"
            else: break
            step = args.max_target_velocity * period
            q = q_command + max(-step, min(step, desired - q_command)); q_command = q
            pub.Write(make_command(msg, index, q, args.kp if phase != "RETURNING" else args.kp, args.kd))
            logger.info(f"STATE = {phase}")
            print_diagnostic(msg, logger, index, initial, q, args.kp, args.kd); time.sleep(period)
    except (KeyboardInterrupt, Exception) as exc:
        control_error = not isinstance(exc, KeyboardInterrupt)
        logger.error(f"SAFE_EXIT_REASON={type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        logger.error(f"FIRST_USER_LOWCMD_SENT={'YES' if first_frame_sent else 'NO'} SWITCH_TO_USER_CTRL_EXECUTED={'YES' if loco is not None else 'NO'} OWNERSHIP_ACQUIRED={'YES' if ownership_acquired else 'NO'}")
    finally:
        if args.ownership_only:
            logger.info("STATE = OWNERSHIP_RELEASE")
        if loco is not None and switched:
            restore = loco.SwitchToInternalCtrl(InternalFsmMode.PASSIVE if args.exit_mode == "passive" else InternalFsmMode.LAST)
            logger.info(f"SWITCH_TO_INTERNAL_CTRL_RESULT={restore}")
            logger.info(f"INTERNAL_CTRL_RESTORE_RESULT={restore} EXIT_MODE={args.exit_mode}")
            final_code, final_id = loco.GetFsmId(); logger.info(f"FINAL_FSM_ID={final_id} FINAL_FSM_NAME={fsm_name(final_id)}")
            if args.ownership_only:
                groups = {g: max((max_delta[i] for i in inds), default=0.0) for g, inds in {
                    "LEGS": tuple(range(12)), "WAIST": tuple(range(12,15)),
                    "LEFT_ARM": tuple(range(15,22)), "RIGHT_ARM": tuple(range(22,29))}.items()}
                unexpected = max(max_delta, default=0.0) > 0.02
                passed = (restore == 0 and final_code == 0 and final_id == 1 and not lowstate_timeout and not control_error and not unexpected)
                logger.result(f"MAX_ABS_Q_DELTA_ALL={max(max_delta, default=0.0):.6f} MAX_ABS_DQ_ALL={max(max_dq, default=0.0):.6f}")
                logger.result(" ".join(f"MAX_{k}_Q_DELTA={v:.6f}" for k, v in groups.items()))
                logger.result(f"OWNERSHIP_ONLY_TEST={'PASS' if passed else 'FAIL'} UNEXPECTED_JOINT_MOTION={'YES' if unexpected else 'NO'} LOWSTATE_TIMEOUT={'YES' if lowstate_timeout else 'NO'} CONTROL_ENTRY_SUCCESS={'YES' if switched else 'NO'} CONTROL_RELEASE_SUCCESS={'YES' if restore == 0 and final_id == 1 else 'NO'}")
                result_text = "PASS" if passed else "FAIL"
                summary = {"test_type": "ownership_only", "joint": None, "motor_index": None, "backend": args.backend, "fsm_before": fsm_id, "fsm_after": final_id, "control_entry_success": switched, "control_release_success": restore == 0 and final_id == 1, "unexpected_joint_motion": unexpected, "max_abs_q_delta_all": max(max_delta, default=0.0), "max_abs_dq_all": max(max_dq, default=0.0), "result": result_text}
                logger.json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(f"TEST_RESULT={result_text}\nLOG_FILE={logger.log_path}")
            else:
                actual = max_delta[index]; return_error = abs(float(msg.motor_state[index].q) - initial)
                tracking = actual / abs(args.delta) if args.delta else 0.0
                other_motion = max((max_delta[i] for i in range(29) if i != index), default=0.0) > 0.02
                physically_verified = actual > 0.02 and return_error < 0.02 and not other_motion
                logger.result(f"MOTOR_INDEX={index} JOINT_NAME={G1_29DOF_JOINT_NAMES[index]} Q_INITIAL={initial:.6f} Q_TARGET={target:.6f} COMMANDED_DELTA={target-initial:.6f} MAX_ACTUAL_DELTA={actual:.6f} TRACKING_RATIO={tracking:.4f} RETURN_ERROR={return_error:.6f}")
                logger.result(f"VISIBLE_MOTION_EXPECTED=YES DIRECTION_CORRECT={'YES' if (target-initial) * (float(msg.motor_state[index].q)-initial) >= 0 else 'NO'} RETURN_WORKED={'YES' if return_error < 0.02 else 'NO'} UNEXPECTED_OTHER_JOINT_MOTION={'YES' if other_motion else 'NO'} CONTROL_ENTRY_SUCCESS={'YES' if switched else 'NO'} CONTROL_RELEASE_SUCCESS={'YES' if restore == 0 and final_id == 1 else 'NO'} PHYSICALLY_VERIFIED={'YES' if physically_verified else 'NO'}")
                result_text = "PASS" if physically_verified and restore == 0 and final_id == 1 else "FAIL"
                summary = {"test_type": "single_motor", "joint": G1_29DOF_JOINT_NAMES[index], "motor_index": index, "backend": args.backend, "q_initial": initial, "q_target": target, "commanded_delta": target-initial, "actual_delta": actual, "return_error": return_error, "kp": args.kp, "kd": args.kd, "control_entry_success": switched, "control_release_success": restore == 0 and final_id == 1, "unexpected_other_joint_motion": other_motion, "physically_verified": physically_verified, "result": result_text}
                logger.json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(f"TEST_RESULT={result_text}\nLOG_FILE={logger.log_path}")
        logger.info("STATE = SAFE_EXIT (CONTROLLED_STOP -> OWNERSHIP_RELEASE; no zero-position command sent)", console=False)
        logger.close()


if __name__ == "__main__": main()
