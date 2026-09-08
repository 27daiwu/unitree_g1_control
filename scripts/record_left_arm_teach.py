#!/usr/bin/env python3
"""Run verified left-arm TEACH control and record the actual trajectory to NPZ."""
from __future__ import annotations

import argparse
import json
import math
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import test_motor_control as motor_control
from test_motor_control import (
    ARM_LIMITS, ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
    InternalFsmMode, LocoClient, LowCmd_, LowState_, State, TestLogger,
    fsm_name, make_ownership_command,
)
from test_left_arm_teach import (
    JOINT_CONFIG, LEFT_ARM_INDICES, LEFT_ARM_NAMES, MujocoGravity,
    PinocchioGravity, STOP_DAMPING_KD, full_state, make_arm_command,
    update_teach_state,
)
from g1_piano.common.joint_map import G1_29DOF_JOINT_NAMES, JOINT_TO_INDEX


def default_output(root: Path) -> Path:
    directory = root / "data/teach/raw"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for take in range(1, 1000):
        candidate = directory / f"{stamp}_left_arm_take{take:03d}.npz"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no available take number in data/teach/raw")


def freeze_samples(samples):
    fields = (
        "timestamp_monotonic", "timestamp_wall", "q", "dq", "tau_est",
        "q_target", "kp", "kd", "gravity_enabled", "gravity_scale",
        "tau_gravity_raw", "tau_ff_command", "teach_state",
    )
    if not samples:
        return {
            "timestamp_monotonic": np.empty((0,), np.float64),
            "timestamp_wall": np.empty((0,), np.float64),
            **{name: np.empty((0, 7), np.float32) for name in
               ("q", "dq", "tau_est", "q_target", "kp", "kd",
                "gravity_scale", "tau_gravity_raw", "tau_ff_command")},
            "gravity_enabled": np.empty((0, 7), np.bool_),
            "teach_state": np.empty((0, 7), dtype="U32"),
        }
    cols = {name: [sample[name] for sample in samples] for name in fields}
    data = {
        "timestamp_monotonic": np.asarray(cols["timestamp_monotonic"], dtype=np.float64),
        "timestamp_wall": np.asarray(cols["timestamp_wall"], dtype=np.float64),
        "gravity_enabled": np.asarray(cols["gravity_enabled"], dtype=np.bool_).reshape((-1, 7)),
        "teach_state": np.asarray(cols["teach_state"], dtype="U32").reshape((-1, 7)),
    }
    for name in ("q", "dq", "tau_est", "q_target", "kp", "kd",
                 "gravity_scale", "tau_gravity_raw", "tau_ff_command"):
        data[name] = np.asarray(cols[name], dtype=np.float32).reshape((-1, 7))
    return data


def validate_recording(data):
    count = len(data["timestamp_monotonic"])
    expected = (count, 7)
    shape_ok = all(data[name].shape == expected for name in (
        "q", "dq", "tau_est", "q_target", "kp", "kd", "gravity_enabled",
        "gravity_scale", "tau_gravity_raw", "tau_ff_command", "teach_state"))
    monotonic = count < 2 or bool(np.all(np.diff(data["timestamp_monotonic"]) > 0))
    numeric = ("timestamp_monotonic", "timestamp_wall", "q", "dq", "tau_est",
               "q_target", "kp", "kd", "gravity_scale", "tau_gravity_raw",
               "tau_ff_command")
    finite = all(bool(np.all(np.isfinite(data[name]))) for name in numeric)
    return shape_ok, monotonic, finite


def write_npz_exclusive(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        np.savez_compressed(output, **data)


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interface", default="eth0")
    ap.add_argument("--duration", type=float, help="recording seconds; omit to stop with Ctrl+C")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--timeout", type=float, default=0.5)
    ap.add_argument("--follow-rate", type=float, default=0.25)
    ap.add_argument("--follow-threshold", type=float, default=0.015)
    ap.add_argument("--moving-threshold", type=float, default=0.05)
    ap.add_argument("--release-dq-threshold", type=float, default=0.03)
    ap.add_argument("--release-stable-time", type=float, default=0.5)
    ap.add_argument("--reenter-moving-time", type=float, default=0.15)
    ap.add_argument("--gravity-tau-limit", type=float, default=8.0)
    ap.add_argument("--passive-timeout", type=float, default=2.0)
    ap.add_argument("--gravity-urdf", default="/home/hebe/unitree_workspace/src/unitree_ros/robots/g1_description/g1_29dof_mode_15.urdf")
    ap.add_argument("--gravity-model", default=str(root / "assets/robots/unitree_g1/xmls/g1.xml"))
    ap.add_argument("--log-dir", default="logs/test")
    args = ap.parse_args()
    if args.duration is not None and args.duration <= 0: ap.error("--duration must be positive")
    if min(args.rate, args.timeout, args.follow_rate, args.follow_threshold,
           args.gravity_tau_limit, args.passive_timeout) <= 0:
        ap.error("rate, timeouts, follower settings and torque limit must be positive")
    if motor_control.ChannelFactoryInitialize is None:
        ap.error(f"unitree_sdk2py unavailable: {getattr(motor_control, '_SDK_ERROR', 'unknown')}")
    output_path = (args.output or default_output(root)).expanduser().resolve()
    if output_path.exists(): ap.error(f"output already exists; refusing overwrite: {output_path}")

    logger = TestLogger(args, "left_arm_teach_recorder")
    print(f"LOG_FILE={logger.log_path}")
    logger.info(f"OUTPUT={output_path}")
    logger.info("JOINT_NAMES=" + ",".join(LEFT_ARM_NAMES))
    logger.info("MOTOR_INDICES=15,16,17,18,19,20,21")
    logger.info("GRAVITY_BACKEND=PINOCCHIO_URDF GRAVITY_REFERENCE_BACKEND=MUJOCO BASE_ORIENTATION_ASSUMED_LEVEL=YES")
    logger.info("TEACH_CONFIG=" + json.dumps(JOINT_CONFIG, sort_keys=True))
    try:
        gravity = PinocchioGravity(args.gravity_urdf)
        reference = MujocoGravity(args.gravity_model)
    except Exception as exc:
        logger.error(f"ABORT_BEFORE_USERCTRL=YES MODEL_LOAD_FAILED={exc}"); logger.close(); return 2

    stop = threading.Event(); signal_reason = [None]
    def on_sigint(*_):
        signal_reason[0] = "SIGINT"; stop.set()
    signal.signal(signal.SIGINT, on_sigint)
    state = State(); ChannelFactoryInitialize(0, args.interface)
    sub = ChannelSubscriber("rt/lowstate", LowState_); sub.Init(state.update, 10)
    pub = loco = None; switched = restore_ok = False; final_id = -1
    samples = []; controls = {}; frozen = freeze_samples([])
    recording_started = recording_complete = raw_written = False
    termination_reason = "NOT_STARTED"; start_wall = end_wall = None
    logger.info("STATE = WAIT_LOWSTATE")
    try:
        while state.snapshot()[0] is None and not stop.is_set(): time.sleep(0.01)
        msg, received = state.snapshot()
        if msg is None or time.monotonic() - received > args.timeout:
            raise RuntimeError("invalid or stale LowState")
        q, _, _ = full_state(msg)
        for i in LEFT_ARM_INDICES:
            lo, hi = ARM_LIMITS[i]
            if not lo <= q[i] <= hi: raise RuntimeError(f"initial joint outside limit: {G1_29DOF_JOINT_NAMES[i]}")
            config = JOINT_CONFIG[G1_29DOF_JOINT_NAMES[i]]
            controls[i] = {"q_target": float(q[i]), "state": "TEACH_IDLE",
                           "candidate_since": None, "reentry_since": None,
                           "gravity_enabled": config["gravity_enabled"],
                           "gravity_scale": config["gravity_scale"]}

        pin_tau, mj_tau = gravity.full_torque(q), reference.full_torque(q)
        for name, config in JOINT_CONFIG.items():
            if not config["gravity_enabled"]: continue
            i = JOINT_TO_INDEX[name]
            both_small = max(abs(pin_tau[i]), abs(mj_tau[i])) < 0.1
            sign_ok = both_small or pin_tau[i] * mj_tau[i] > 0
            ratio = 1.0 if both_small else abs(pin_tau[i] / mj_tau[i]) if abs(mj_tau[i]) > 1e-6 else math.inf
            logger.info(f"RUNTIME_SANITY joint={name} sdk_index={i} pinocchio_joint_id={gravity.joint_ids[i]} pinocchio_v_index={gravity.v_indices[i]} pinocchio_tau={pin_tau[i]:.6f} mujoco_dof_index={reference.v_indices[i]} mujoco_tau={mj_tau[i]:.6f} sign_match={'YES' if sign_ok else 'NO'} magnitude_ratio={ratio:.6f}")
            if not sign_ok or not math.isfinite(ratio) or not 0.25 <= ratio <= 4.0:
                raise RuntimeError(f"ABORT_BEFORE_USERCTRL: gravity sanity failed for {name}")
        loco = LocoClient(); loco.Init(); code, fsm_id = loco.GetFsmId()
        if code != 0 or fsm_id != 1:
            raise RuntimeError(f"robot must be PASSIVE/DAMPING (fsm={fsm_name(fsm_id)})")
        try: confirmed = input("Type YES to enable LEFT ARM TEACH RECORDING: ") == "YES"
        except (EOFError, KeyboardInterrupt): confirmed = False
        if not confirmed: raise RuntimeError("operator did not confirm")
        pub = ChannelPublisher("rt/user_lowcmd", LowCmd_); pub.Init()
        pub.Write(make_ownership_command(msg, STOP_DAMPING_KD))
        logger.info("STATE = ENTER_USERCTRL")
        switched = loco.SwitchToUserCtrl() == 0
        if not switched: raise RuntimeError("SwitchToUserCtrl failed")
        recording_started = True; termination_reason = "RUNNING"
        start_wall = time.time(); start_mono = last = time.monotonic()
        period = 1.0 / args.rate; logger.info("LEFT_ARM_TEACH_ACTIVE=YES RECORDING_ACTIVE=YES")
        while not stop.is_set():
            cycle_start = now = time.monotonic()
            if args.duration is not None and now - start_mono >= args.duration:
                recording_complete = True; termination_reason = "NORMAL"; break
            dt = min(now - last, 0.1); last = now
            msg, received = state.snapshot()
            if msg is None or now - received > args.timeout: raise RuntimeError("LowState timeout")
            q, dq, tau_est = full_state(msg)
            tau_full = gravity.full_torque(q)  # exactly one Pinocchio call per cycle
            targets = {}; tau_cmd = {}; raw_values = []
            for i in LEFT_ARM_INDICES:
                name = G1_29DOF_JOINT_NAMES[i]; config = JOINT_CONFIG[name]; control = controls[i]
                previous_target = control["q_target"]
                update_teach_state(control, dq[i], now, args)
                if control["state"] in ("TEACH_MOVING", "TEACH_RELEASE_CANDIDATE"):
                    error = q[i] - control["q_target"]
                    if abs(error) > args.follow_threshold:
                        control["q_target"] += max(-args.follow_rate * dt, min(args.follow_rate * dt, error))
                lo, hi = ARM_LIMITS[i]; control["q_target"] = max(lo, min(hi, control["q_target"]))
                if abs(control["q_target"] - previous_target) > args.follow_rate * dt + 1e-6:
                    raise RuntimeError(f"single-cycle q_target limit violated: {name}")
                raw = float(tau_full[i]) if config["gravity_enabled"] else 0.0
                scaled = config["gravity_scale"] * raw if config["gravity_enabled"] else 0.0
                command = max(-args.gravity_tau_limit, min(args.gravity_tau_limit, scaled)) if config["gravity_enabled"] else 0.0
                if config["gravity_enabled"] and abs(scaled) > args.gravity_tau_limit:
                    logger.warning(f"GRAVITY_TAU_CLAMP joint={name} raw={raw:.6f} scaled={scaled:.6f} limit={args.gravity_tau_limit:.6f}")
                targets[i] = control["q_target"]; tau_cmd[i] = command; raw_values.append(raw)
            pub.Write(make_arm_command(msg, targets, tau_cmd))
            samples.append({
                "timestamp_monotonic": now, "timestamp_wall": time.time(),
                "q": q[15:22].copy(), "dq": dq[15:22].copy(), "tau_est": tau_est[15:22].copy(),
                "q_target": [targets[i] for i in LEFT_ARM_INDICES],
                "kp": [JOINT_CONFIG[name]["kp"] for name in LEFT_ARM_NAMES],
                "kd": [JOINT_CONFIG[name]["kd"] for name in LEFT_ARM_NAMES],
                "gravity_enabled": [JOINT_CONFIG[name]["gravity_enabled"] for name in LEFT_ARM_NAMES],
                "gravity_scale": [JOINT_CONFIG[name]["gravity_scale"] for name in LEFT_ARM_NAMES],
                "tau_gravity_raw": raw_values,
                "tau_ff_command": [tau_cmd[i] for i in LEFT_ARM_INDICES],
                "teach_state": [controls[i]["state"] for i in LEFT_ARM_INDICES],
            })
            time.sleep(max(0.0, period - (time.monotonic() - cycle_start)))
        if signal_reason[0]: termination_reason = signal_reason[0]
    except KeyboardInterrupt:
        termination_reason = "KEYBOARD_INTERRUPT"
    except Exception as exc:
        termination_reason = f"{type(exc).__name__}: {exc}"
        logger.error(f"RECORDING_ERROR={termination_reason}")
    finally:
        end_wall = time.time()
        logger.info("STATE = STOP_RECORDING")
        frozen = freeze_samples(samples)
        shape_ok, timestamp_ok, finite_ok = validate_recording(frozen)
        logger.info(f"STATE = FLUSH_DATA SAMPLES={len(samples)} SHAPE_OK={'YES' if shape_ok else 'NO'} TIMESTAMP_MONOTONIC={'YES' if timestamp_ok else 'NO'} NO_NAN_INF={'YES' if finite_ok else 'NO'}")
        logger.info("STATE = CONTROLLED_STOP ALL_LEFT_ARM_TAU=0")
        try:
            msg = state.snapshot()[0]
            if pub is not None and msg is not None: pub.Write(make_ownership_command(msg, STOP_DAMPING_KD))
        except Exception as exc: logger.error(f"controlled stop publish failed: {exc}")
        if loco is not None and switched:
            try: restore_ok = loco.SwitchToInternalCtrl(InternalFsmMode.PASSIVE) == 0
            except Exception: restore_ok = False
        logger.info("STATE = RETURN_PASSIVE")
        if loco is not None:
            deadline = time.monotonic() + args.passive_timeout
            while time.monotonic() < deadline:
                try:
                    code, final_id = loco.GetFsmId()
                    if code == 0 and final_id == 1: break
                except Exception: pass
                time.sleep(0.05)
        safe_exit = restore_ok and final_id == 1

        count = len(frozen["timestamp_monotonic"])
        duration = (float(frozen["timestamp_monotonic"][-1] - frozen["timestamp_monotonic"][0])
                    if count > 1 else 0.0)
        actual_rate = (count - 1) / duration if duration > 0 else 0.0
        metadata = {
            "format_version": "left-arm-teach-raw-1", "joint_names": list(LEFT_ARM_NAMES),
            "motor_indices": list(LEFT_ARM_INDICES), "control_rate_hz": args.rate,
            "record_rate_hz": args.rate, "actual_record_rate_hz": actual_rate,
            "gravity_backend": "PINOCCHIO_URDF", "gravity_model_path": str(args.gravity_urdf),
            "gravity_reference_model_path": str(args.gravity_model),
            "base_orientation_assumed_level": True, "joint_config": JOINT_CONFIG,
            "follow_rate": args.follow_rate, "follow_threshold": args.follow_threshold,
            "start_timestamp": start_wall, "end_timestamp": end_wall, "duration": duration,
            "sample_count": count, "recording_complete": recording_complete,
            "termination_reason": termination_reason,
        }
        frozen["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
        if recording_started and count > 0 and shape_ok and timestamp_ok and finite_ok:
            try:
                write_npz_exclusive(output_path, frozen); raw_written = True
            except Exception as exc: logger.error(f"RAW_WRITE_FAILED={type(exc).__name__}: {exc}")
        pipeline_pass = (switched and recording_started and count > 0 and shape_ok
                         and timestamp_ok and finite_ok and raw_written and safe_exit)
        logger.result(f"SAMPLE_COUNT={count}\nDURATION={duration:.6f}\nACTUAL_RECORD_RATE={actual_rate:.3f}\nQ_SHAPE={list(frozen['q'].shape)}\nDQ_SHAPE={list(frozen['dq'].shape)}\nTAU_EST_SHAPE={list(frozen['tau_est'].shape)}\nTIMESTAMP_MONOTONIC={'YES' if timestamp_ok else 'NO'}\nNO_NAN_INF={'YES' if finite_ok else 'NO'}\nRECORDING_COMPLETE={'YES' if recording_complete else 'NO'}\nTERMINATION_REASON={termination_reason}\nRAW_FILE_WRITTEN={'YES' if raw_written else 'NO'}\nRAW_FILE={output_path}\nLEFT_ARM_TEACH_ACTIVE={'YES' if switched else 'NO'}\nRECORDING_ACTIVE={'YES' if recording_started else 'NO'}\nSAFE_EXIT_TO_PASSIVE={'YES' if safe_exit else 'NO'}")
        logger.result(f"LEFT_ARM_TEACH_RECORDING_PIPELINE={'PASS' if pipeline_pass else 'FAIL'}")
        logger.close()
    return 0 if pipeline_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
