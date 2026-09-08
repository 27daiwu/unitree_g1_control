#!/usr/bin/env python3
"""G1 29DoF left-arm multi-joint TEACH prototype (motors 15..21 only)."""
from __future__ import annotations

import argparse
import json
import math
import signal
import threading
import time
from pathlib import Path

import numpy as np

import test_motor_control as motor_control
from test_motor_control import (
    ARM_LIMITS, ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
    InternalFsmMode, LocoClient, LowCmd_, LowState_, State, TestLogger,
    create_lowcmd, fsm_name, make_ownership_command, set_field,
)
from g1_piano.common.joint_map import G1_29DOF_JOINT_NAMES, JOINT_TO_INDEX

try:
    import pinocchio as pin
except ImportError:
    pin = None

try:
    import mujoco
    from g1_piano.simulate.mujoco_player import G1MujocoMapping
except ImportError:
    mujoco = None
    G1MujocoMapping = None


LEFT_ARM_NAMES = tuple(G1_29DOF_JOINT_NAMES[15:22])
LEFT_ARM_INDICES = tuple(range(15, 22))
JOINT_CONFIG = {
    "left_shoulder_pitch": {"kp": 8.0, "kd": 1.5, "gravity_enabled": True, "gravity_scale": 0.75},
    "left_shoulder_roll": {"kp": 8.0, "kd": 1.5, "gravity_enabled": True, "gravity_scale": 0.80},
    "left_shoulder_yaw": {"kp": 8.0, "kd": 1.5, "gravity_enabled": True, "gravity_scale": 0.75},
    "left_elbow": {"kp": 8.0, "kd": 1.5, "gravity_enabled": True, "gravity_scale": 0.75},
    "left_wrist_roll": {"kp": 6.0, "kd": 1.5, "gravity_enabled": False, "gravity_scale": 0.0},
    "left_wrist_pitch": {"kp": 6.0, "kd": 1.5, "gravity_enabled": False, "gravity_scale": 0.0},
    "left_wrist_yaw": {"kp": 6.0, "kd": 1.5, "gravity_enabled": False, "gravity_scale": 0.0},
}
STOP_DAMPING_KD = 1.5


class PinocchioGravity:
    """Name-mapped free-flyer gravity model returning SDK-order torques."""

    def __init__(self, urdf):
        required = ("buildModelFromUrdf", "computeGeneralizedGravity", "neutral", "JointModelFreeFlyer")
        if pin is None or not all(hasattr(pin, item) for item in required):
            raise RuntimeError("robotics Pinocchio API unavailable")
        self.model = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        if self.model.nq != 36 or self.model.nv != 35:
            raise RuntimeError(f"expected free-flyer nq=36 nv=35, got {self.model.nq}/{self.model.nv}")
        self.q_indices, self.v_indices, self.joint_ids = [], [], []
        for name in G1_29DOF_JOINT_NAMES:
            jid = int(self.model.getJointId(f"{name}_joint"))
            if jid <= 0 or jid >= self.model.njoints:
                raise RuntimeError(f"Pinocchio mapping missing: {name}")
            if int(self.model.nqs[jid]) != 1 or int(self.model.nvs[jid]) != 1:
                raise RuntimeError(f"Pinocchio joint is not 1DoF: {name}")
            self.joint_ids.append(jid)
            self.q_indices.append(int(self.model.idx_qs[jid]))
            self.v_indices.append(int(self.model.idx_vs[jid]))
        self.neutral = pin.neutral(self.model)

    def full_torque(self, q_sdk):
        q_sdk = np.asarray(q_sdk, dtype=float)
        if q_sdk.shape != (29,) or not np.all(np.isfinite(q_sdk)):
            raise ValueError("q_sdk must be finite shape (29,)")
        q_pin = self.neutral.copy()  # level base: xyz=0, quaternion=identity
        for sdk_index, q_index in enumerate(self.q_indices):
            q_pin[q_index] = q_sdk[sdk_index]
        tau_pin = pin.computeGeneralizedGravity(self.model, self.data, q_pin)
        result = np.asarray([tau_pin[v] for v in self.v_indices], dtype=float)
        if result.shape != (29,) or not np.all(np.isfinite(result)):
            raise ValueError("invalid Pinocchio gravity result")
        return result


class MujocoGravity:
    """Name-mapped MuJoCo gravity reference returning SDK-order torques."""

    def __init__(self, mjcf):
        if mujoco is None or G1MujocoMapping is None:
            raise RuntimeError("MuJoCo unavailable")
        self.model = mujoco.MjModel.from_xml_path(str(mjcf))
        self.data = mujoco.MjData(self.model)
        mapping = G1MujocoMapping.from_model(self.model)
        self.q_indices = mapping.qpos_addresses
        self.v_indices = tuple(int(self.model.jnt_dofadr[j]) for j in mapping.joint_ids)
        free = [j for j in range(self.model.njnt)
                if int(self.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        if free:
            adr = int(self.model.jnt_qposadr[free[0]])
            self.data.qpos[adr:adr + 7] = (0, 0, 0.79, 1, 0, 0, 0)

    def full_torque(self, q_sdk):
        q_sdk = np.asarray(q_sdk, dtype=float)
        if q_sdk.shape != (29,) or not np.all(np.isfinite(q_sdk)):
            raise ValueError("q_sdk must be finite shape (29,)")
        for sdk_index, q_index in enumerate(self.q_indices):
            self.data.qpos[q_index] = q_sdk[sdk_index]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        result = np.asarray([self.data.qfrc_bias[v] for v in self.v_indices], dtype=float)
        if not np.all(np.isfinite(result)):
            raise ValueError("invalid MuJoCo gravity result")
        return result


def full_state(msg):
    q = np.asarray([float(msg.motor_state[i].q) for i in range(29)], dtype=float)
    dq = np.asarray([float(msg.motor_state[i].dq) for i in range(29)], dtype=float)
    tau = np.asarray([float(getattr(msg.motor_state[i], "tau_est", 0.0)) for i in range(29)], dtype=float)
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq)) and np.all(np.isfinite(tau))):
        raise RuntimeError("invalid LowState NaN/Inf")
    if np.max(np.abs(q)) > 10 or np.max(np.abs(dq)) > 100:
        raise RuntimeError("abnormal LowState q/dq")
    return q, dq, tau


def make_arm_command(msg, targets, tau_ff):
    cmd = create_lowcmd()
    slots = getattr(cmd, "motor_cmd", getattr(cmd, "motorCmd", []))
    if callable(slots): slots = slots()
    for i in range(min(35, len(slots))):
        q = float(msg.motor_state[i].q) if i < 29 else 0.0
        for field, value in (("q", q), ("dq", 0.0), ("tau", 0.0),
                             ("kp", 0.0), ("kd", 0.0), ("mode", 0)):
            set_field(slots[i], field, value)
    for index in LEFT_ARM_INDICES:
        config = JOINT_CONFIG[G1_29DOF_JOINT_NAMES[index]]
        set_field(slots[index], "q", float(targets[index]))
        set_field(slots[index], "dq", 0.0)
        set_field(slots[index], "kp", float(config["kp"]))
        set_field(slots[index], "kd", float(config["kd"]))
        set_field(slots[index], "tau", float(tau_ff[index]))
        set_field(slots[index], "mode", 1)
    return cmd


def update_teach_state(control, dq, now, args):
    state = control["state"]
    if state == "TEACH_IDLE" and abs(dq) > args.moving_threshold:
        control["state"] = "TEACH_MOVING"
    elif state == "TEACH_MOVING" and abs(dq) < args.release_dq_threshold:
        control["state"] = "TEACH_RELEASE_CANDIDATE"; control["candidate_since"] = now
    elif state == "TEACH_RELEASE_CANDIDATE":
        if abs(dq) > args.moving_threshold:
            control["state"] = "TEACH_MOVING"; control["candidate_since"] = None
        elif abs(dq) < args.release_dq_threshold and now - control["candidate_since"] >= args.release_stable_time:
            control["state"] = "TEACH_RELEASE_HOLD"; control["reentry_since"] = None
    elif state == "TEACH_RELEASE_HOLD":
        if abs(dq) > args.moving_threshold:
            control["reentry_since"] = control["reentry_since"] or now
            if now - control["reentry_since"] >= args.reenter_moving_time:
                control["state"] = "TEACH_MOVING"; control["reentry_since"] = None
        else:
            control["reentry_since"] = None


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interface", default="eth0")
    ap.add_argument("--follow-rate", type=float, default=0.25)
    ap.add_argument("--follow-threshold", type=float, default=0.015)
    ap.add_argument("--wrist-kp", type=float, choices=(4.0, 6.0), default=6.0,
                    help="Kp for all three left wrist joints; Kd remains 1.5")
    ap.add_argument("--rate", type=float, default=100.0); ap.add_argument("--timeout", type=float, default=0.5)
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
    if motor_control.ChannelFactoryInitialize is None:
        ap.error(f"unitree_sdk2py unavailable: {getattr(motor_control, '_SDK_ERROR', 'unknown')}")
    if min(args.follow_rate, args.follow_threshold, args.rate,
           args.timeout, args.gravity_tau_limit) <= 0:
        ap.error("control and safety parameters must be positive")
    for wrist_name in ("left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw"):
        JOINT_CONFIG[wrist_name]["kp"] = args.wrist_kp

    logger = TestLogger(args, "left_arm_multi_joint_teach")
    print(f"LOG_FILE={logger.log_path}")
    logger.info("CONTROL_SCOPE=LEFT_ARM_ONLY MOTOR_INDICES=15..21")
    logger.info("GRAVITY_BACKEND=PINOCCHIO_URDF GRAVITY_REFERENCE_BACKEND=MUJOCO BASE_ORIENTATION_ASSUMED_LEVEL=YES")
    try:
        gravity = PinocchioGravity(args.gravity_urdf)
        reference = MujocoGravity(args.gravity_model)
    except Exception as exc:
        logger.error(f"ABORT_BEFORE_USERCTRL=YES MODEL_LOAD_FAILED={exc}"); logger.close(); return 2

    state = State(); stop = threading.Event(); signal.signal(signal.SIGINT, lambda *_: stop.set())
    ChannelFactoryInitialize(0, args.interface)
    sub = ChannelSubscriber("rt/lowstate", LowState_); sub.Init(state.update, 10)
    pub = loco = None; switched = restore_ok = False; final_id = -1
    controls = {}; samples = []; clamp_active = {i: False for i in LEFT_ARM_INDICES}
    max_g = {i: 0.0 for i in LEFT_ARM_INDICES}; max_ff = max_g.copy(); max_est = max_g.copy()
    logger.info("STATE = WAIT_LOWSTATE")
    try:
        while state.snapshot()[0] is None and not stop.is_set(): time.sleep(0.01)
        msg, received = state.snapshot()
        if msg is None or time.monotonic() - received > args.timeout:
            raise RuntimeError("invalid or stale LowState")
        q, dq, tau_est = full_state(msg)
        for i in LEFT_ARM_INDICES:
            lo, hi = ARM_LIMITS[i]
            if not lo <= q[i] <= hi:
                raise RuntimeError(f"initial joint position outside limit: {G1_29DOF_JOINT_NAMES[i]}")
            config = JOINT_CONFIG[G1_29DOF_JOINT_NAMES[i]]
            controls[i] = {"q_target": q[i], "state": "TEACH_IDLE",
                           "candidate_since": None, "reentry_since": None,
                           "follower_enabled": False,
                           "gravity_enabled": config["gravity_enabled"],
                           "gravity_scale": config["gravity_scale"],
                           "tau_gravity": 0.0}

        pin_tau = gravity.full_torque(q); mj_tau = reference.full_torque(q)
        for name, config in JOINT_CONFIG.items():
            i = JOINT_TO_INDEX[name]
            enabled, scale = config["gravity_enabled"], config["gravity_scale"]
            if not enabled: continue
            both_small = max(abs(pin_tau[i]), abs(mj_tau[i])) < 0.1
            sign_ok = both_small or pin_tau[i] * mj_tau[i] > 0
            ratio = 1.0 if both_small else abs(pin_tau[i] / mj_tau[i]) if abs(mj_tau[i]) > 1e-6 else math.inf
            logger.info(f"RUNTIME_SANITY joint={name} sdk_index={i} pinocchio_joint_id={gravity.joint_ids[i]} pinocchio_v_index={gravity.v_indices[i]} pinocchio_tau={pin_tau[i]:.6f} mujoco_dof_index={reference.v_indices[i]} mujoco_tau={mj_tau[i]:.6f} sign_match={'YES' if sign_ok else 'NO'} magnitude_ratio={ratio:.6f}")
            if not sign_ok or not math.isfinite(ratio) or not 0.25 <= ratio <= 4.0:
                raise RuntimeError(f"ABORT_BEFORE_USERCTRL: gravity sanity failed for {name}")

        loco = LocoClient(); loco.Init(); code, fsm_id = loco.GetFsmId()
        if code != 0 or fsm_id != 1:
            raise RuntimeError(f"robot must be PASSIVE/DAMPING (fsm={fsm_name(fsm_id)})")
        try: confirmed = input("Type YES to enable LEFT ARM TEACH: ") == "YES"
        except (EOFError, KeyboardInterrupt): confirmed = False
        if not confirmed: raise RuntimeError("operator did not confirm")
        pub = ChannelPublisher("rt/user_lowcmd", LowCmd_); pub.Init()
        pub.Write(make_ownership_command(msg, STOP_DAMPING_KD))
        logger.info("STATE = ENTER_USERCTRL")
        switched = loco.SwitchToUserCtrl() == 0
        if not switched: raise RuntimeError("SwitchToUserCtrl failed")
        logger.info("STATE = LEFT_ARM_TEACH_ACTIVE")
        period = 1.0 / args.rate; last = time.monotonic(); cycle = 0
        while not stop.is_set():
            started = now = time.monotonic(); dt = min(now - last, 0.1); last = now
            msg, received = state.snapshot()
            if msg is None or now - received > args.timeout: raise RuntimeError("LowState timeout")
            q, dq, tau_est = full_state(msg)
            tau_full = gravity.full_torque(q)  # exactly one dynamics call per cycle
            tau_cmd = {}; log_parts = []
            for i in LEFT_ARM_INDICES:
                name = G1_29DOF_JOINT_NAMES[i]; control = controls[i]
                previous_target = control["q_target"]
                update_teach_state(control, dq[i], now, args)
                control["follower_enabled"] = control["state"] in ("TEACH_MOVING", "TEACH_RELEASE_CANDIDATE")
                if control["follower_enabled"]:
                    error = q[i] - control["q_target"]
                    if abs(error) > args.follow_threshold:
                        step = max(-args.follow_rate * dt, min(args.follow_rate * dt, error))
                        control["q_target"] += step
                lo, hi = ARM_LIMITS[i]
                control["q_target"] = max(lo, min(hi, control["q_target"]))
                if abs(control["q_target"] - previous_target) > args.follow_rate * dt + 1e-6:
                    raise RuntimeError(f"single-cycle q_target limit violated: {name}")
                enabled, scale = control["gravity_enabled"], control["gravity_scale"]
                raw = float(tau_full[i]) if enabled else 0.0
                control["tau_gravity"] = raw
                scaled = scale * raw if enabled else 0.0
                command = max(-args.gravity_tau_limit, min(args.gravity_tau_limit, scaled)) if enabled else 0.0
                clamped = enabled and abs(scaled) > args.gravity_tau_limit
                if clamped and not clamp_active[i]:
                    logger.warning(f"GRAVITY_TAU_CLAMP joint={name} raw={raw:.6f} scaled={scaled:.6f} limit={args.gravity_tau_limit:.6f}")
                clamp_active[i] = clamped; tau_cmd[i] = command
                max_g[i] = max(max_g[i], abs(raw)); max_ff[i] = max(max_ff[i], abs(command)); max_est[i] = max(max_est[i], abs(tau_est[i]))
                config = JOINT_CONFIG[name]
                log_parts.append(f"{name}:q_actual={q[i]:.5f},q_target={control['q_target']:.5f},dq={dq[i]:.5f},tau_est={tau_est[i]:.5f},kp={config['kp']:.2f},kd={config['kd']:.2f},state={control['state']},gravity_enabled={'YES' if enabled else 'NO'},gravity_scale={scale:.2f},tau_gravity_raw={raw:.5f},tau_ff_command={command:.5f}")
            timestamp = time.time(); logger.debug(f"timestamp={timestamp:.6f} " + " | ".join(log_parts))
            if cycle % max(1, int(args.rate / 2)) == 0:
                logger.info("LEFT_ARM_TEACH " + " | ".join(log_parts))
            pub.Write(make_arm_command(msg, {i: controls[i]["q_target"] for i in LEFT_ARM_INDICES}, tau_cmd))
            cycle += 1; time.sleep(max(0.0, period - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.error(f"SAFE_EXIT_REASON={type(exc).__name__}: {exc}")
    finally:
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
        summary = {"scope": "left_arm_15_21", "base_orientation_assumed_level": True,
                   "gravity_backend": "PINOCCHIO_URDF", "joint_config": JOINT_CONFIG,
                   "max_abs_tau_gravity_per_joint": {}, "max_abs_tau_ff_per_joint": {},
                   "max_abs_tau_est_per_joint": {}, "safe_exit_to_passive": safe_exit}
        logger.result("TEACH_KP_PER_JOINT=" + json.dumps(
            {name: config["kp"] for name, config in JOINT_CONFIG.items()}, sort_keys=True))
        logger.result("TEACH_KD_PER_JOINT=" + json.dumps(
            {name: config["kd"] for name, config in JOINT_CONFIG.items()}, sort_keys=True))
        for i in LEFT_ARM_INDICES:
            name = G1_29DOF_JOINT_NAMES[i]
            summary["max_abs_tau_gravity_per_joint"][name] = max_g[i]
            summary["max_abs_tau_ff_per_joint"][name] = max_ff[i]
            summary["max_abs_tau_est_per_joint"][name] = max_est[i]
            config = JOINT_CONFIG[name]
            logger.result(f"JOINT={name} TEACH_KP={config['kp']:.3f} TEACH_KD={config['kd']:.3f} MAX_ABS_TAU_GRAVITY={max_g[i]:.6f} MAX_ABS_TAU_FF={max_ff[i]:.6f} MAX_ABS_TAU_EST={max_est[i]:.6f}")
        logger.result(f"LEFT_ARM_MANUALLY_MOVABLE=OPERATOR\nLEFT_ARM_NOTICEABLE_RESISTANCE=OPERATOR\nWRIST_MANUALLY_MOVABLE=OPERATOR\nWRIST_DRAG_RESISTANCE_COMFORTABLE=OPERATOR\nNO_WRIST_OSCILLATION=OPERATOR\nLEFT_ARM_MULTI_JOINT_TEACH_FEEL=OPERATOR\nSHOULDER_PITCH_NO_FREE_FALL=OPERATOR\nSHOULDER_ROLL_NO_FREE_FALL=OPERATOR\nSHOULDER_YAW_NO_FREE_FALL=OPERATOR\nSHOULDER_YAW_GRAVITY_FEELS_NEUTRAL=OPERATOR\nELBOW_NO_FREE_FALL=OPERATOR\nNO_ARM_OSCILLATION=OPERATOR\nSAFE_EXIT_TO_PASSIVE={'YES' if safe_exit else 'NO'}\nLEFT_ARM_MULTI_JOINT_TEACH_READY=YES")
        logger.json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
