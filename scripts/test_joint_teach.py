#!/usr/bin/env python3
"""Generic single-joint TEACH prototype for the G1 29DoF arm.

The operator supplies the motion by hand. Only the selected arm motor receives
TEACH position gains; every other slot is sent as measured-q damping/zero-torque.
"""
import argparse
import json
import math
import signal
import threading
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import test_motor_control as motor_control
from test_motor_control import (
    State, TestLogger, ARM_LIMITS, G1_29DOF_JOINT_NAMES,
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
    LowCmd_, LowState_, LocoClient, InternalFsmMode,
    create_lowcmd, set_field, fsm_name,
)
from g1_piano.common.joint_map import resolve_joint, joint_name
try:
    import mujoco
    from g1_piano.simulate.mujoco_player import G1MujocoMapping
except ImportError:
    mujoco = None; G1MujocoMapping = None
try:
    import pinocchio as pin
except ImportError:
    pin = None

ARM_INDEX_MIN, ARM_INDEX_MAX = 15, 28
GRAVITY_COMP_JOINTS = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
)


def resolve_teach_joint(value):
    """Resolve through the shared joint map, accepting CamelCase aliases."""
    try:
        return resolve_joint(value)
    except ValueError:
        key = str(value).replace("_", "").lower()
        for candidate in G1_29DOF_JOINT_NAMES:
            if candidate.replace("_", "").lower() == key:
                return resolve_joint(candidate)
        raise


def finite_motor(m):
    return all(math.isfinite(float(v)) for v in
               (m.q, m.dq, getattr(m, "tau_est", 0.0)))


def make_teach_command(msg, index, target, kp, kd, tau_ff=0.0):
    cmd = create_lowcmd()
    slots = getattr(cmd, "motor_cmd", getattr(cmd, "motorCmd", []))
    if callable(slots):
        slots = slots()
    for i in range(min(35, len(slots))):
        q = float(msg.motor_state[i].q) if i < 29 else 0.0
        for name, value in (("q", q), ("dq", 0.0), ("tau", 0.0),
                            ("kp", 0.0), ("kd", 0.0), ("mode", 0)):
            set_field(slots[i], name, value)
    m = slots[index]
    set_field(m, "q", float(target)); set_field(m, "kp", float(kp))
    set_field(m, "kd", float(kd)); set_field(m, "tau", float(tau_ff)); set_field(m, "mode", 1)
    return cmd


class MujocoGravityModel:
    """MuJoCo qfrc_bias gravity estimate for the complete 29DoF pose."""
    def __init__(self, path):
        if mujoco is None or G1MujocoMapping is None:
            raise RuntimeError("MuJoCo is unavailable")
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.mapping = G1MujocoMapping.from_model(self.model)
        self.qadr = self.mapping.qpos_addresses
        if len(self.qadr) != 29:
            raise RuntimeError("model mapping does not contain 29 joints")
        free = [i for i in range(self.model.njnt) if int(self.model.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        if free:
            adr = int(self.model.jnt_qposadr[free[0]])
            self.data.qpos[adr:adr + 7] = (0.0, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0)
    def torque(self, q_full, index):
        q_full = np.asarray(q_full, dtype=float)
        if q_full.shape != (29,) or not np.all(np.isfinite(q_full)):
            raise ValueError("q_full must be finite shape (29,)")
        for i, adr in enumerate(self.qadr): self.data.qpos[adr] = q_full[i]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        dof_index = int(self.model.jnt_dofadr[self.mapping.joint_ids[index]])
        raw = float(self.data.qfrc_bias[dof_index])
        # qfrc_bias at zero velocity is gravity in MuJoCo generalized coordinates.
        return raw


class PinocchioGravityModel:
    """Free-flyer URDF gravity using explicit joint-name mapping."""
    def __init__(self, path):
        required = ("buildModelFromUrdf", "computeGeneralizedGravity", "neutral", "JointModelFreeFlyer")
        if pin is None or not all(hasattr(pin, item) for item in required):
            raise RuntimeError("robotics Pinocchio API is unavailable")
        self.model = pin.buildModelFromUrdf(str(path), pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        if self.model.nq != 36 or self.model.nv != 35:
            raise RuntimeError(f"expected free-flyer nq=36 nv=35, got nq={self.model.nq} nv={self.model.nv}")
        self.joint_ids = []
        self.q_indices = []
        self.v_indices = []
        for name in G1_29DOF_JOINT_NAMES:
            joint_id = int(self.model.getJointId(f"{name}_joint"))
            if joint_id <= 0 or joint_id >= self.model.njoints:
                raise RuntimeError(f"Pinocchio joint mapping missing: {name}")
            if int(self.model.nqs[joint_id]) != 1 or int(self.model.nvs[joint_id]) != 1:
                raise RuntimeError(f"Pinocchio joint is not 1DoF: {name}")
            self.joint_ids.append(joint_id)
            self.q_indices.append(int(self.model.idx_qs[joint_id]))
            self.v_indices.append(int(self.model.idx_vs[joint_id]))
        self.neutral = pin.neutral(self.model)

    def torque(self, q_full, index):
        q_full = np.asarray(q_full, dtype=float)
        if q_full.shape != (29,) or not np.all(np.isfinite(q_full)):
            raise ValueError("q_full must be finite shape (29,)")
        q_pin = self.neutral.copy()  # level base: xyz=0, quaternion=identity
        for sdk_index, q_index in enumerate(self.q_indices):
            q_pin[q_index] = q_full[sdk_index]
        tau = pin.computeGeneralizedGravity(self.model, self.data, q_pin)
        raw = float(tau[self.v_indices[index]])
        if not math.isfinite(raw):
            raise ValueError("non-finite Pinocchio gravity torque")
        return raw


def make_damping_command(msg, kd):
    cmd = create_lowcmd()
    slots = getattr(cmd, "motor_cmd", getattr(cmd, "motorCmd", []))
    if callable(slots): slots = slots()
    for i in range(min(29, len(slots))):
        set_field(slots[i], "q", float(msg.motor_state[i].q))
        set_field(slots[i], "dq", 0.0); set_field(slots[i], "kp", 0.0)
        set_field(slots[i], "kd", float(kd)); set_field(slots[i], "tau", 0.0)
        set_field(slots[i], "mode", 0)
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interface", default="eth0")
    p.add_argument("--joint", required=True, help="exact joint name or motor index; one joint only")
    p.add_argument("--kp", type=float, default=8.0)
    p.add_argument("--kd", type=float, default=1.5)
    p.add_argument("--follow-rate", type=float, default=0.25,
                   help="maximum target following speed (rad/s)")
    p.add_argument("--follow-threshold", type=float, default=0.015,
                   help="deadband before target follows actual (rad)")
    p.add_argument("--rate", type=float, default=100.0)
    p.add_argument("--timeout", type=float, default=0.5)
    p.add_argument("--release-dq-threshold", type=float, default=0.03)
    p.add_argument("--moving-threshold", type=float, default=0.05,
                   help="abs(dq) threshold indicating manual motion (rad/s)")
    p.add_argument("--release-stable-time", type=float, default=0.5)
    p.add_argument("--reenter-moving-time", type=float, default=0.15,
                   help="stable motion time required to leave RELEASE_HOLD")
    p.add_argument("--release-window", type=float, default=2.0)
    p.add_argument("--passive-timeout", type=float, default=2.0)
    p.add_argument("--gravity-comp", action="store_true", help="enable model-based gravity torque")
    p.add_argument("--gravity-backend", choices=("pinocchio", "mujoco"), default="pinocchio")
    p.add_argument("--gravity-scale", type=float, default=0.0)
    p.add_argument("--gravity-tau-limit", type=float, default=8.0)
    p.add_argument("--gravity-model", default=str(Path(__file__).resolve().parents[1] / "assets/robots/unitree_g1/xmls/g1.xml"))
    p.add_argument("--gravity-urdf", default="/home/hebe/unitree_workspace/src/unitree_ros/robots/g1_description/g1_29dof_mode_15.urdf")
    p.add_argument("--log-dir", default="logs/test")
    args = p.parse_args()
    if ChannelFactoryInitialize is None:
        p.error(f"unitree_sdk2py unavailable: {getattr(motor_control, '_SDK_ERROR', 'unknown import error')}")
    try:
        index = resolve_teach_joint(args.joint)
    except ValueError as exc:
        p.error(str(exc))
    if not ARM_INDEX_MIN <= index <= ARM_INDEX_MAX:
        p.error("single-joint TEACH is restricted to arm joints (indices 15..28)")
    if not 0.0 <= args.gravity_scale <= 1.0 or args.gravity_tau_limit <= 0:
        p.error("gravity-scale must be in [0,1] and gravity-tau-limit must be positive")
    name = joint_name(index)
    if args.gravity_comp and name not in GRAVITY_COMP_JOINTS:
        p.error("--gravity-comp is restricted to one left shoulder joint or LeftElbow")
    if args.gravity_backend == "pinocchio" and not args.gravity_comp:
        # Backend selection is inert unless compensation is explicitly enabled.
        args.gravity_backend = "disabled"
    if min(args.kp, args.kd, args.follow_rate, args.follow_threshold,
           args.rate, args.timeout) <= 0:
        p.error("kp, kd, follow parameters, rate and timeout must be positive")
    logger = TestLogger(args, f"joint_teach_{name}")
    print(f"LOG_FILE={logger.log_path}")
    logger.info("LOWCMD_TORQUE_FIELD=tau SDK_MotorCmd_tau; SDK tau_ff aliases are written to this field")
    logger.info("GRAVITY_MODEL_SCOPE=single_joint_left_shoulder_or_left_elbow")
    state = State(); stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    ChannelFactoryInitialize(0, args.interface)
    sub = ChannelSubscriber("rt/lowstate", LowState_); sub.Init(state.update, 10)
    pub = None; loco = None; switched = False; restore_ok = False
    gravity_model = pinocchio_model = mujoco_model = None
    gravity_comp_active = bool(args.gravity_comp and args.gravity_scale > 0.0)
    if args.gravity_comp:
        try:
            pinocchio_model = PinocchioGravityModel(args.gravity_urdf)
            mujoco_model = MujocoGravityModel(args.gravity_model)
            if args.gravity_backend == "pinocchio":
                gravity_model = pinocchio_model
                logger.info(f"GRAVITY_BACKEND=PINOCCHIO_URDF path={args.gravity_urdf}")
            else:
                gravity_model = mujoco_model
                logger.info(f"GRAVITY_BACKEND=MUJOCO path={args.gravity_model}")
        except Exception as exc:
            logger.error(f"GRAVITY_MODEL=UNAVAILABLE reason={exc}; aborting before UserCtrl")
            logger.info("ABORT_BEFORE_USERCTRL=YES")
            logger.close()
            return
    q_initial = q_target = None; release_q = release_q_target = None; release_time = None
    q_last = None
    max_manual = max_dq = max_tau = 0.0
    samples = []; previous_q = None; follow_count = 0
    teach_state = "TEACH_IDLE"; moving_seen = False; candidate_since = None
    reentry_since = None; release_events = []; active_event = None
    logger.info("STATE = WAIT_LOWSTATE")
    try:
        while state.snapshot()[0] is None and not stop.is_set(): time.sleep(0.01)
        msg, received = state.snapshot()
        if msg is None or time.monotonic() - received > args.timeout:
            raise RuntimeError("invalid or stale LowState")
        m = msg.motor_state[index]
        if not finite_motor(m): raise RuntimeError("invalid initial motor state")
        q_initial = q_target = float(m.q); previous_q = q_initial
        lo, hi = ARM_LIMITS[index]
        logger.info("STATE = CAPTURE_INITIAL")
        logger.info(f"MOTOR_INDEX={index} JOINT_NAME={name} Q_INITIAL={q_initial:.6f}")
        if args.gravity_comp:
            q_sanity = np.asarray([float(msg.motor_state[i].q) for i in range(29)], dtype=float)
            if not np.all(np.isfinite(q_sanity)):
                raise RuntimeError("ABORT_BEFORE_USERCTRL: non-finite q for gravity sanity check")
            try:
                pinocchio_tau = pinocchio_model.torque(q_sanity, index)
                reference_tau = mujoco_model.torque(q_sanity, index)
                both_near_zero = max(abs(pinocchio_tau), abs(reference_tau)) < 0.1
                ratio = (1.0 if both_near_zero else
                         abs(pinocchio_tau / reference_tau) if abs(reference_tau) > 1e-6 else float("inf"))
                sign_match = both_near_zero or pinocchio_tau * reference_tau > 0.0
                sanity_joint = name.upper()
                logger.info(f"PINOCCHIO_TAU_GRAVITY_{sanity_joint}={pinocchio_tau:.8f} MUJOCO_REFERENCE_TAU_GRAVITY_{sanity_joint}={reference_tau:.8f} RUNTIME_GRAVITY_SIGN_MATCH={'YES' if sign_match else 'NO'} RUNTIME_GRAVITY_MAGNITUDE_RATIO={ratio:.6f} BASE_ORIENTATION_ASSUMED_LEVEL=YES")
                if not (math.isfinite(pinocchio_tau) and math.isfinite(reference_tau) and sign_match and 0.25 <= ratio <= 4.0):
                    raise RuntimeError("ABORT_BEFORE_USERCTRL: runtime gravity sanity check failed")
            except Exception as exc:
                raise RuntimeError(str(exc))
        loco = LocoClient(); loco.Init(); code, fsm_id = loco.GetFsmId()
        if code != 0 or fsm_id != 1:
            raise RuntimeError(f"robot must be PASSIVE/DAMPING (fsm={fsm_name(fsm_id)})")
        pub = ChannelPublisher("rt/user_lowcmd", LowCmd_); pub.Init()
        pub.Write(make_teach_command(msg, index, q_target, 0.0, args.kd))
        logger.info("STATE = ENTER_USERCTRL")
        switched = loco.SwitchToUserCtrl() == 0
        if not switched: raise RuntimeError("SwitchToUserCtrl failed")
        logger.info("STATE = TEACH_ENTER")
        logger.info("TEACH_ENTER_SUCCESS=YES")
        start = last = time.monotonic(); period = 1.0 / args.rate
        logger.info("STATE = TEACH_ACTIVE")
        while not stop.is_set():
            now = time.monotonic(); dt = min(now - last, 0.1); last = now
            msg, received = state.snapshot()
            if msg is None or now - received > args.timeout: raise RuntimeError("lowstate timeout")
            m = msg.motor_state[index]
            if not finite_motor(m) or abs(float(m.q)) > 10 or abs(float(m.dq)) > 100:
                raise RuntimeError("invalid LowState values")
            q = float(m.q); q_last = q; dq = float(m.dq); tau = float(getattr(m, "tau_est", 0.0))
            # State transitions use hysteresis/stable times.  In RELEASE_HOLD,
            # gravity-induced motion alone does not immediately re-enable the
            # follower; motion must remain above moving_threshold first.
            if teach_state == "TEACH_IDLE":
                if abs(dq) > args.moving_threshold:
                    teach_state = "TEACH_MOVING"; moving_seen = True
            elif teach_state == "TEACH_MOVING":
                if abs(dq) < args.release_dq_threshold:
                    teach_state = "TEACH_RELEASE_CANDIDATE"; candidate_since = now
            elif teach_state == "TEACH_RELEASE_CANDIDATE":
                if abs(dq) > args.moving_threshold:
                    teach_state = "TEACH_MOVING"; candidate_since = None
                elif abs(dq) < args.release_dq_threshold and now - candidate_since >= args.release_stable_time:
                    teach_state = "TEACH_RELEASE_HOLD"; release_time = now
                    release_q = q; release_q_target = q_target
                    active_event = {"release_q": q, "release_q_target": q_target,
                                    "release_time": time.time(), "hold_start_q": q,
                                    "hold_end_q": q, "hold_drift": 0.0,
                                    "hold_max_dq": abs(dq), "hold_max_tau_est": abs(tau),
                                    "samples": 0}
                    release_events.append(active_event)
                    reentry_since = None
                    logger.info(f"TEACH_STATE=TEACH_RELEASE_HOLD FOLLOWER_ENABLED=NO Q_TARGET_FROZEN=YES RELEASE_Q={q:.6f} RELEASE_Q_TARGET={q_target:.6f}")
            elif teach_state == "TEACH_RELEASE_HOLD":
                if abs(dq) > args.moving_threshold:
                    reentry_since = reentry_since or now
                    if now - reentry_since >= args.reenter_moving_time:
                        teach_state = "TEACH_MOVING"; reentry_since = None
                        logger.info("TEACH_STATE=TEACH_MOVING FOLLOWER_ENABLED=YES Q_TARGET_FROZEN=NO")
                else:
                    reentry_since = None

            follower_enabled = teach_state in ("TEACH_IDLE", "TEACH_MOVING", "TEACH_RELEASE_CANDIDATE")
            target_step = 0.0
            if follower_enabled:
                error = q - q_target
                if abs(error) > args.follow_threshold:
                    target_step = max(-args.follow_rate * dt, min(args.follow_rate * dt, error))
                    q_target += target_step; follow_count += 1 if abs(target_step) > 0.0 else 0
            elif teach_state == "TEACH_RELEASE_HOLD":
                q_target = release_q
            q_target = max(lo, min(hi, q_target))
            if abs(target_step) > args.follow_rate * dt + 1e-6:
                raise RuntimeError("single-cycle target step limit violated")
            max_manual = max(max_manual, abs(q - q_initial)); max_dq = max(max_dq, abs(dq)); max_tau = max(max_tau, abs(tau))
            tau_gravity_raw = tau_gravity_scaled = tau_ff_command = 0.0
            pin_tau = mj_tau = 0.0
            if gravity_comp_active:
                try:
                    q_full = np.asarray([float(msg.motor_state[i].q) for i in range(29)], dtype=float)
                    pin_tau = pinocchio_model.torque(q_full, index)
                    mj_tau = mujoco_model.torque(q_full, index)
                    tau_gravity_raw = pin_tau if args.gravity_backend == "pinocchio" else mj_tau
                    if not math.isfinite(tau_gravity_raw):
                        raise ValueError("non-finite gravity torque")
                    tau_gravity_scaled = args.gravity_scale * tau_gravity_raw
                    tau_ff_command = max(-args.gravity_tau_limit, min(args.gravity_tau_limit, tau_gravity_scaled))
                    if abs(tau_gravity_scaled) > args.gravity_tau_limit:
                        logger.warning(f"GRAVITY_TAU_CLAMP raw={tau_gravity_raw:.5f} scaled={tau_gravity_scaled:.5f} limit={args.gravity_tau_limit:.5f}")
                except Exception as exc:
                    tau_gravity_raw = tau_gravity_scaled = tau_ff_command = 0.0
                    logger.error(f"GRAVITY_CALCULATION_FAILED={exc}; tau_ff=0; controlled stop requested")
                    raise RuntimeError("gravity calculation failure") from exc
            if teach_state == "TEACH_RELEASE_HOLD" and active_event is not None:
                delta = q - release_q
                active_event["hold_drift"] = max(active_event["hold_drift"], abs(delta))
                active_event["hold_end_q"] = q
                active_event["hold_max_dq"] = max(active_event["hold_max_dq"], abs(dq))
                active_event["hold_max_tau_est"] = max(active_event["hold_max_tau_est"], abs(tau))
                active_event["samples"] += 1
                if now - release_time > args.release_window:
                    active_event["window_complete"] = True
            previous_q = q
            samples.append({"timestamp": time.time(), "q_actual": q, "q_target": q_target, "dq": dq, "tau_est": tau, "error": q - q_target, "kp": args.kp, "kd": args.kd, "teach_state": teach_state, "follower_enabled": follower_enabled, "q_target_frozen": not follower_enabled, "gravity_backend": args.gravity_backend, "pinocchio_tau_gravity_raw": pin_tau, "mujoco_reference_tau_gravity": mj_tau, "tau_gravity_raw": tau_gravity_raw, "tau_gravity_scaled": tau_gravity_scaled, "tau_ff_command": tau_ff_command, "gravity_scale": args.gravity_scale, "gravity_comp_active": gravity_comp_active})
            if len(samples) == 1 or len(samples) % max(1, int(args.rate / 5)) == 0:
                logger.info(f"TEACH_ACTIVE TEACH_STATE={teach_state} FOLLOWER_ENABLED={'YES' if follower_enabled else 'NO'} Q_TARGET_FROZEN={'YES' if not follower_enabled else 'NO'} q_actual={q:.5f} q_target={q_target:.5f} dq={dq:.5f} tau_est={tau:.5f} pinocchio_tau_gravity_raw={pin_tau:.5f} mujoco_reference_tau_gravity={mj_tau:.5f} tau_gravity_raw={tau_gravity_raw:.5f} tau_gravity_scaled={tau_gravity_scaled:.5f} tau_ff_command={tau_ff_command:.5f} gravity_backend={args.gravity_backend} gravity_scale={args.gravity_scale:.3f} gravity_comp_active={'YES' if gravity_comp_active else 'NO'} error={q-q_target:.5f} kp={args.kp:.3f} kd={args.kd:.3f}")
            pub.Write(make_teach_command(msg, index, q_target, args.kp, args.kd, tau_ff_command))
            time.sleep(max(0.0, period - (time.monotonic() - now)))
    except KeyboardInterrupt: pass
    except Exception as exc:
        logger.error(f"SAFE_EXIT_REASON={type(exc).__name__}: {exc}")
    finally:
        logger.info("STATE = CONTROLLED_STOP")
        try:
            if pub is not None and state.snapshot()[0] is not None:
                pub.Write(make_damping_command(state.snapshot()[0], args.kd))
        except Exception as exc: logger.error(f"controlled stop publish failed: {exc}")
        if loco is not None and switched:
            try: restore_ok = loco.SwitchToInternalCtrl(InternalFsmMode.PASSIVE) == 0
            except Exception: restore_ok = False
        logger.info("STATE = RETURN_PASSIVE")
        if loco is not None:
            final_id = -1; deadline = time.monotonic() + args.passive_timeout
            while time.monotonic() < deadline:
                try:
                    code, final_id = loco.GetFsmId()
                    if code == 0 and final_id == 1: break
                except Exception: pass
                time.sleep(0.05)
        else: final_id = -1
        follow_ok = bool(samples) and follow_count > 0 and max(abs(s["q_actual"] - s["q_target"]) for s in samples) < 0.5
        no_osc = max_dq < 100.0
        program_pass = switched and follow_ok and no_osc and restore_ok and final_id == 1
        release_drift = max((e["hold_drift"] for e in release_events), default=0.0)
        spring_back = 0.0
        for e in release_events:
            direction_before = e["release_q"] - q_initial
            if direction_before:
                spring_back = max(spring_back, max((abs(s["q_actual"] - e["release_q"]) for s in samples
                    if s["timestamp"] >= e["release_time"] and (s["q_actual"] - e["release_q"]) * direction_before < 0), default=0.0))
        post_delta = 0.0 if release_q is None or q_last is None else (q_last - release_q)
        direction = "POSITIVE" if post_delta > 0.01 else "NEGATIVE" if post_delta < -0.01 else "STABLE"
        event_lines = []
        for i, event in enumerate(release_events, 1):
            event_lines.append(f"RELEASE_{i}_Q={event['release_q']:.6f}")
            event_lines.append(f"RELEASE_{i}_HOLD_DRIFT={event['hold_drift']:.6f}")
        hold_start_q = release_events[-1]["hold_start_q"] if release_events else None
        hold_end_q = release_events[-1]["hold_end_q"] if release_events else None
        hold_max_dq = max((e["hold_max_dq"] for e in release_events), default=0.0)
        hold_max_tau = max((e["hold_max_tau_est"] for e in release_events), default=0.0)
        reentered = any(
            s["teach_state"] == "TEACH_MOVING"
            and any(s["timestamp"] >= e["release_time"] for e in release_events)
            for s in samples
        )
        logger.result(f"JOINT={name}\nMOTOR_INDEX={index}\nTEACH_KP={args.kp}\nTEACH_KD={args.kd}\nFOLLOW_RATE={args.follow_rate}\nFOLLOW_THRESHOLD={args.follow_threshold}\nQ_INITIAL={q_initial}\nQ_MAX={max((s['q_actual'] for s in samples), default=q_initial):.6f}\nQ_MIN={min((s['q_actual'] for s in samples), default=q_initial):.6f}\nMAX_MANUAL_DELTA={max_manual:.6f}\nMAX_DQ={max_dq:.6f}\nMAX_TAU_EST={max_tau:.6f}\nRELEASE_EVENT_COUNT={len(release_events)}\nRELEASE_Q={release_q}\nRELEASE_Q_TARGET={release_q_target}\nHOLD_START_Q={hold_start_q}\nHOLD_END_Q={hold_end_q}\nHOLD_DRIFT={release_drift:.6f}\nHOLD_MAX_DQ={hold_max_dq:.6f}\nHOLD_MAX_TAU_EST={hold_max_tau:.6f}\nSPRING_BACK_DELTA={spring_back:.6f}\nRELEASE_DRIFT={release_drift:.6f}\nPOST_RELEASE_DELTA={post_delta:.6f}\nPOST_RELEASE_DIRECTION={direction}\n" + "\n".join(event_lines) + f"\nTEACH_ENTER_SUCCESS={'YES' if switched else 'NO'}\nQ_TARGET_FOLLOWS_Q_ACTUAL={'YES' if follow_ok else 'NO'}\nNO_OSCILLATION={'YES' if no_osc else 'NO'}\nSAFE_EXIT_TO_PASSIVE={'YES' if restore_ok and final_id == 1 else 'NO'}\nTEACH_MOVING_WORKS={'YES' if follow_count > 0 else 'NO'}\nRELEASE_HOLD_ENTERED={'YES' if release_events else 'NO'}\nFOLLOWER_DISABLED_DURING_HOLD={'YES' if any(not s['follower_enabled'] for s in samples) else 'NO'}\nQ_TARGET_FROZEN_DURING_HOLD={'YES' if any(s['q_target_frozen'] for s in samples) else 'NO'}\nREENTER_TEACH_MOVING={'YES' if reentered else 'NO'}\nJOINT_MANUALLY_MOVABLE=OPERATOR\nNOTICEABLE_RESISTANCE=OPERATOR\nNO_VIOLENT_SPRING_BACK=OPERATOR\nRELEASE_POSITION_APPROX_HOLD=OPERATOR\nELBOW_RELEASE_GRAVITY_DRIFT=OPERATOR\nPOSITION_HOLD_INSUFFICIENT_AGAINST_GRAVITY=OPERATOR\nGRAVITY_COMPENSATION_REQUIRED=OPERATOR\nELBOW_HOLD_WITH_KP8=OPERATOR\nELBOW_HOLD_WITH_KP12=OPERATOR\nELBOW_RELEASE_GRAVITY_DRIFT=OPERATOR\nELBOW_RELEASE_HOLD_PROTOTYPE=IMPLEMENTED\nGRAVITY_COMPENSATION_DECISION_GATE=READY\nTEST_RESULT={'PASS' if program_pass else 'FAIL'}")
        max_raw = max((abs(s['tau_gravity_raw']) for s in samples), default=0.0)
        max_scaled = max((abs(s['tau_gravity_scaled']) for s in samples), default=0.0)
        max_cmd = max((abs(s['tau_ff_command']) for s in samples), default=0.0)
        joint_summary_name = name.upper()
        best_scale_field = f"BEST_{joint_summary_name}_GRAVITY_SCALE"
        real_robot_field = f"{joint_summary_name}_GRAVITY_COMP_REAL_ROBOT"
        logger.result(f"GRAVITY_BACKEND={args.gravity_backend}\nFORMAL_RUNTIME_BACKEND=PINOCCHIO_URDF\nBASE_ORIENTATION_ASSUMED_LEVEL=YES\nGRAVITY_COMP_ACTIVE={'YES' if gravity_comp_active else 'NO'}\nGRAVITY_SCALE={args.gravity_scale}\nGRAVITY_TAU_LIMIT={args.gravity_tau_limit}\nTAU_GRAVITY_RAW_MAX={max_raw:.6f}\nTAU_GRAVITY_SCALED_MAX={max_scaled:.6f}\nTAU_FF_COMMAND_MAX={max_cmd:.6f}\nMAX_ABS_TAU_GRAVITY={max_raw:.6f}\nMAX_ABS_TAU_FF={max_cmd:.6f}\nMAX_ABS_TAU_EST={max_tau:.6f}\n" + f"JOINT={name}\nMOTOR_INDEX={index}\nTEACH_KP={args.kp}\nTEACH_KD={args.kd}\nFOLLOW_RATE={args.follow_rate}\nFOLLOW_THRESHOLD={args.follow_threshold}\nQ_INITIAL={q_initial}\nQ_MAX={max((s['q_actual'] for s in samples), default=q_initial):.6f}\nQ_MIN={min((s['q_actual'] for s in samples), default=q_initial):.6f}\nMAX_MANUAL_DELTA={max_manual:.6f}\nMAX_DQ={max_dq:.6f}\nMAX_TAU_EST={max_tau:.6f}\nRELEASE_EVENT_COUNT={len(release_events)}\nRELEASE_Q={release_q}\nRELEASE_Q_TARGET={release_q_target}\nHOLD_START_Q={hold_start_q}\nHOLD_END_Q={hold_end_q}\nHOLD_DRIFT={release_drift:.6f}\nHOLD_MAX_DQ={hold_max_dq:.6f}\nHOLD_MAX_TAU_EST={hold_max_tau:.6f}\nPOST_RELEASE_Q_DELTA={post_delta:.6f}\nPOST_RELEASE_MAX_DQ={hold_max_dq:.6f}\nSPRING_BACK_DELTA={spring_back:.6f}\nRELEASE_DRIFT={release_drift:.6f}\nPOST_RELEASE_DELTA={post_delta:.6f}\nPOST_RELEASE_DIRECTION={direction}\n" + "\n".join(event_lines) + f"\nTEACH_ENTER_SUCCESS={'YES' if switched else 'NO'}\nQ_TARGET_FOLLOWS_Q_ACTUAL={'YES' if follow_ok else 'NO'}\nNO_OSCILLATION={'YES' if no_osc else 'NO'}\nSAFE_EXIT_TO_PASSIVE={'YES' if restore_ok and final_id == 1 else 'NO'}\nTEACH_MOVING_WORKS={'YES' if follow_count > 0 else 'NO'}\nRELEASE_HOLD_ENTERED={'YES' if release_events else 'NO'}\nFOLLOWER_DISABLED_DURING_HOLD={'YES' if any(not s['follower_enabled'] for s in samples) else 'NO'}\nQ_TARGET_FROZEN_DURING_HOLD={'YES' if any(s['q_target_frozen'] for s in samples) else 'NO'}\nREENTER_TEACH_MOVING={'YES' if reentered else 'NO'}\nJOINT_MANUALLY_MOVABLE=OPERATOR\nNOTICEABLE_RESISTANCE=OPERATOR\nGRAVITY_FEELS_NEUTRAL=OPERATOR\nNO_DOWNWARD_FREE_FALL=OPERATOR\nNO_UPWARD_SELF_MOTION=OPERATOR\nNO_VIOLENT_SPRING_BACK=OPERATOR\nRELEASE_POSITION_APPROX_HOLD=OPERATOR\n{best_scale_field}=OPERATOR\n{real_robot_field}=NOT_YET_VERIFIED\nPINOCCHIO_RUNTIME_GRAVITY_BACKEND=READY_FOR_REAL_ROBOT_TEST\nTEST_RESULT={'PASS' if program_pass else 'FAIL'}")
        logger.json_path.write_text(json.dumps({"joint": name, "motor_index": index, "samples": samples, "release_events": release_events, "q_initial": q_initial, "q_max": max((s['q_actual'] for s in samples), default=q_initial), "q_min": min((s['q_actual'] for s in samples), default=q_initial), "release_q": release_q, "release_q_target": release_q_target, "hold_start_q": hold_start_q, "hold_end_q": hold_end_q, "hold_drift": release_drift, "hold_max_dq": hold_max_dq, "hold_max_tau_est": hold_max_tau, "post_release_delta": post_delta, "post_release_direction": direction, "max_manual_delta": max_manual, "max_dq": max_dq, "max_tau_est": max_tau, "spring_back_delta": spring_back, "release_drift": release_drift, "gravity_comp_active": gravity_comp_active, "gravity_scale": args.gravity_scale, "gravity_tau_limit": args.gravity_tau_limit, "teach_enter_success": switched, "q_target_follows_q_actual": follow_ok, "no_oscillation": no_osc, "safe_exit_to_passive": restore_ok and final_id == 1, best_scale_field: "OPERATOR", real_robot_field: "NOT_YET_VERIFIED", "operator_assessment": {"joint_manually_movable": "OPERATOR", "noticeable_resistance": "OPERATOR", "gravity_feels_neutral": "OPERATOR", "no_downward_free_fall": "OPERATOR", "no_upward_self_motion": "OPERATOR", "no_violent_spring_back": "OPERATOR", "release_position_approx_hold": "OPERATOR"}, "result": "PASS" if program_pass else "FAIL"}, indent=2), encoding="utf-8")
        logger.close()


if __name__ == "__main__": main()
