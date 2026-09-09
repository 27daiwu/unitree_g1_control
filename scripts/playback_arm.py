#!/usr/bin/env python3
"""Safely replay a recorded left/right/both-arm TEACH trajectory on G1."""
from __future__ import annotations

import argparse
import json
import math
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import test_motor_control as motor_control
from test_motor_control import (
    ARM_LIMITS, ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
    InternalFsmMode, LocoClient, LowCmd_, LowState_, State, TestLogger,
    create_lowcmd, fsm_name, make_ownership_command, set_field,
)
from scripts.test_arm_teach import (
    JOINT_CONFIG, LEFT_ARM_INDICES, LEFT_ARM_NAMES, MujocoGravity,
    PinocchioGravity, STOP_DAMPING_KD, full_state,
)
from g1_piano.common.joint_map import G1_29DOF_JOINT_NAMES, JOINT_TO_INDEX
from g1_piano.common.arm_spec import ArmSpec, get_arm_spec


PLAYBACK_CONFIG = {
    "left_shoulder_pitch": {"kp": 20.0, "kd": 2.0},
    "left_shoulder_roll": {"kp": 20.0, "kd": 2.0},
    "left_shoulder_yaw": {"kp": 20.0, "kd": 2.0},
    "left_elbow": {"kp": 20.0, "kd": 2.0},
    "left_wrist_roll": {"kp": 12.0, "kd": 1.5},
    "left_wrist_pitch": {"kp": 12.0, "kd": 1.5},
    "left_wrist_yaw": {"kp": 12.0, "kd": 1.5},
}


def playback_config(spec: ArmSpec, shoulder_kp=20.0, shoulder_kd=2.0,
                    wrist_kp=12.0, wrist_kd=1.5):
    """Build playback gains without reusing the lower-stiffness TEACH gains."""
    return {
        name: ({"kp": wrist_kp, "kd": wrist_kd}
               if "_wrist_" in name else {"kp": shoulder_kp, "kd": shoulder_kd})
        for name in spec.joint_names
    }


@dataclass(frozen=True)
class PlaybackTrajectory:
    path: Path
    timestamps: np.ndarray
    times: np.ndarray
    q: np.ndarray
    dq: np.ndarray | None
    metadata: dict
    finite_difference_dq: np.ndarray
    arm_spec: ArmSpec

    @property
    def duration(self):
        return float(self.times[-1])


def _identity_field(source, metadata, name):
    """Read identity from top-level NPZ and metadata, rejecting conflicts."""
    top = None
    if name in source.files:
        value = source[name]
        top = str(value.item()) if name == "arm_mode" else value.tolist()
    nested = metadata.get(name)
    if top is not None and nested is not None and top != nested:
        raise ValueError(f"top-level/metadata {name} mismatch: {top!r} != {nested!r}")
    return top if top is not None else nested


def load_trajectory(path: Path, recorded_dq_limit: float,
                    finite_difference_dq_limit: float,
                    dq_sanity_error_limit: float,
                    requested_arm_mode: str | None = None) -> PlaybackTrajectory:
    """Load and fully audit a recorder NPZ without enabling robot I/O."""
    try:
        with np.load(path, allow_pickle=False) as source:
            missing = {"timestamp_monotonic", "q", "metadata"} - set(source.files)
            if missing:
                raise ValueError("missing required fields: " + ", ".join(sorted(missing)))
            timestamps = np.asarray(source["timestamp_monotonic"], dtype=np.float64)
            q = np.asarray(source["q"], dtype=np.float64)
            dq = (np.asarray(source["dq"], dtype=np.float64)
                  if "dq" in source.files else None)
            metadata_value = source["metadata"]
            if metadata_value.shape != ():
                raise ValueError(f"metadata must be a scalar JSON string, got {metadata_value.shape}")
            metadata = json.loads(str(metadata_value.item()))
            if not isinstance(metadata, dict):
                raise ValueError("metadata JSON must be an object")
            arm_mode = _identity_field(source, metadata, "arm_mode")
            joint_names = _identity_field(source, metadata, "joint_names")
            motor_indices = _identity_field(source, metadata, "motor_indices")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load trajectory {path}: {exc}") from exc

    if arm_mode is None:
        # Backward compatibility for verified left-arm files predating arm_mode.
        for candidate in ("left", "right", "both"):
            candidate_spec = get_arm_spec(candidate)
            if (joint_names == list(candidate_spec.joint_names) and
                    motor_indices == list(candidate_spec.motor_indices)):
                arm_mode = candidate
                break
    if arm_mode not in ("left", "right", "both"):
        raise ValueError(f"invalid or missing arm_mode: {arm_mode!r}")
    if requested_arm_mode is not None and requested_arm_mode != arm_mode:
        raise ValueError(
            f"ARM_MODE_MISMATCH FILE_ARM_MODE={arm_mode} "
            f"REQUESTED_ARM_MODE={requested_arm_mode}")
    spec = get_arm_spec(arm_mode)
    joint_count = spec.joint_count
    if timestamps.ndim != 1:
        raise ValueError(f"timestamp_monotonic must have shape [N], got {timestamps.shape}")
    count = len(timestamps)
    if count <= 1:
        raise ValueError("trajectory must contain more than one sample")
    if q.shape != (count, joint_count):
        raise ValueError(f"q must have shape [{count},{joint_count}], got {q.shape}")
    if dq is not None and dq.shape != q.shape:
        raise ValueError(f"dq must have shape {q.shape}, got {dq.shape}")
    finite_values = (timestamps, q) if dq is None else (timestamps, q, dq)
    if not all(np.all(np.isfinite(values)) for values in finite_values):
        raise ValueError("trajectory contains NaN/Inf")
    delta_t = np.diff(timestamps)
    if not np.all(delta_t > 0.0):
        raise ValueError("timestamp_monotonic is not strictly increasing")
    if not isinstance(joint_names, list) or len(joint_names) != joint_count:
        raise ValueError(f"invalid joint_names: {joint_names!r}")
    if not isinstance(motor_indices, list) or len(motor_indices) != joint_count:
        raise ValueError(f"invalid motor_indices: {motor_indices!r}")
    if not all(isinstance(name, str) for name in joint_names):
        raise ValueError("joint_names must contain strings")
    if not all(isinstance(index, int) and not isinstance(index, bool)
               for index in motor_indices):
        raise ValueError("motor_indices must contain integers")
    if len(set(joint_names)) != joint_count:
        raise ValueError("duplicate joint_names")
    if len(set(motor_indices)) != joint_count:
        raise ValueError("duplicate motor_indices")
    if joint_names != list(spec.joint_names):
        raise ValueError(f"joint_names mismatch: {joint_names!r}")
    if motor_indices != list(spec.motor_indices):
        raise ValueError(f"motor_indices mismatch: {motor_indices!r}")
    if metadata.get("recording_complete") is not True:
        raise ValueError("recording_complete is not true; incomplete playback is prohibited")
    if metadata.get("sample_count") not in (None, count):
        raise ValueError(f"metadata sample_count mismatch: {metadata.get('sample_count')} != {count}")
    metadata_duration = metadata.get("duration")
    metadata_rate = metadata.get("actual_record_rate_hz")
    if (not isinstance(metadata_duration, (int, float)) or
            not math.isfinite(metadata_duration) or metadata_duration <= 0.0):
        raise ValueError(f"invalid metadata duration: {metadata_duration!r}")
    if (not isinstance(metadata_rate, (int, float)) or
            not math.isfinite(metadata_rate) or metadata_rate <= 0.0):
        raise ValueError(f"invalid metadata actual_record_rate_hz: {metadata_rate!r}")

    for column, index in enumerate(spec.motor_indices):
        lo, hi = spec.joint_limits[column]
        if np.min(q[:, column]) < lo or np.max(q[:, column]) > hi:
            raise ValueError(f"trajectory joint limit violation: {G1_29DOF_JOINT_NAMES[index]}")

    finite_difference_dq = np.diff(q, axis=0) / delta_t[:, None]
    max_finite_difference = np.max(np.abs(finite_difference_dq), axis=0)
    if dq is not None:
        max_recorded = np.max(np.abs(dq), axis=0)
        max_sanity_error = np.max(np.abs(finite_difference_dq - dq[1:]), axis=0)
        if np.any(max_recorded > recorded_dq_limit):
            raise ValueError(f"recorded dq exceeds {recorded_dq_limit}: {max_recorded.tolist()}")
        if np.any(max_sanity_error > dq_sanity_error_limit):
            raise ValueError(
                f"recorded/finite-difference dq mismatch exceeds {dq_sanity_error_limit}: "
                f"{max_sanity_error.tolist()}")
    if np.any(max_finite_difference > finite_difference_dq_limit):
        raise ValueError(
            f"finite-difference dq exceeds {finite_difference_dq_limit}: "
            f"{max_finite_difference.tolist()}")
    times = timestamps - timestamps[0]
    return PlaybackTrajectory(path, timestamps, times, q, dq, metadata,
                              finite_difference_dq, spec)


def interpolate_q(trajectory: PlaybackTrajectory, trajectory_time: float) -> np.ndarray:
    t = float(np.clip(trajectory_time, 0.0, trajectory.duration))
    return np.asarray([
        np.interp(t, trajectory.times, trajectory.q[:, column])
        for column in range(trajectory.arm_spec.joint_count)
    ], dtype=np.float64)


def make_playback_command(msg, targets, tau_ff, config=PLAYBACK_CONFIG,
                          spec=None):
    """Build a complete frame while enabling only the selected arm joints."""
    spec = spec or get_arm_spec("left")
    cmd = create_lowcmd()
    slots = getattr(cmd, "motor_cmd", getattr(cmd, "motorCmd", []))
    if callable(slots):
        slots = slots()
    for index in range(min(35, len(slots))):
        measured = float(msg.motor_state[index].q) if index < 29 else 0.0
        for field, value in (("q", measured), ("dq", 0.0), ("tau", 0.0),
                             ("kp", 0.0), ("kd", 0.0), ("mode", 0)):
            set_field(slots[index], field, value)
    for column, index in enumerate(spec.motor_indices):
        joint = G1_29DOF_JOINT_NAMES[index]
        set_field(slots[index], "q", float(targets[column]))
        set_field(slots[index], "dq", 0.0)
        set_field(slots[index], "kp", float(config[joint]["kp"]))
        set_field(slots[index], "kd", float(config[joint]["kd"]))
        set_field(slots[index], "tau", float(tau_ff[column]))
        set_field(slots[index], "mode", 1)
    return cmd


def validate_runtime_state(q, dq, tau_est, args, spec=None):
    spec = spec or get_arm_spec("left")
    for column, index in enumerate(spec.motor_indices):
        name = G1_29DOF_JOINT_NAMES[index]
        lo, hi = spec.joint_limits[column]
        if not lo <= q[index] <= hi:
            raise RuntimeError(f"joint limit violation: {name} q={q[index]:.6f}")
        if abs(dq[index]) > args.dq_limit:
            raise RuntimeError(f"dq safety limit: {name} dq={dq[index]:.6f}")
        if abs(tau_est[index]) > args.tau_est_limit:
            raise RuntimeError(f"tau_est safety limit: {name} tau_est={tau_est[index]:.6f}")


def gravity_command(gravity, q_full, args, logger, clamp_active, spec=None):
    spec = spec or get_arm_spec("left")
    tau_full = gravity.full_torque(q_full)  # exactly one Pinocchio call per cycle
    commands = np.zeros(spec.joint_count, dtype=np.float64)
    raw = np.zeros(spec.joint_count, dtype=np.float64)
    gravity_config = spec.config()
    for column, index in enumerate(spec.motor_indices):
        name = G1_29DOF_JOINT_NAMES[index]
        joint_config = gravity_config[name]
        if not joint_config["gravity_enabled"]:
            commands[column] = 0.0
            continue
        raw[column] = float(tau_full[index])
        scaled = joint_config["gravity_scale"] * raw[column]
        commands[column] = float(np.clip(scaled, -args.gravity_tau_limit,
                                         args.gravity_tau_limit))
        clamped = abs(scaled) > args.gravity_tau_limit
        if clamped and not clamp_active[column]:
            logger.warning(
                f"GRAVITY_TAU_CLAMP joint={name} raw={raw[column]:.6f} "
                f"scaled={scaled:.6f} limit={args.gravity_tau_limit:.6f}")
        clamp_active[column] = clamped
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(commands)):
        raise RuntimeError("gravity calculation returned NaN/Inf")
    return raw, commands


def runtime_gravity_sanity(gravity, reference, q, logger, spec=None):
    spec = spec or get_arm_spec("left")
    pin_tau = gravity.full_torque(q)
    mj_tau = reference.full_torque(q)
    for name, gravity_config in spec.config().items():
        if not gravity_config["gravity_enabled"]:
            continue
        index = JOINT_TO_INDEX[name]
        both_small = max(abs(pin_tau[index]), abs(mj_tau[index])) < 0.1
        sign_ok = both_small or pin_tau[index] * mj_tau[index] > 0.0
        ratio = (1.0 if both_small else
                 abs(pin_tau[index] / mj_tau[index]) if abs(mj_tau[index]) > 1e-6
                 else math.inf)
        logger.info(
            f"RUNTIME_SANITY joint={name} sdk_index={index} "
            f"pinocchio_joint_id={gravity.joint_ids[index]} "
            f"pinocchio_v_index={gravity.v_indices[index]} "
            f"pinocchio_tau={pin_tau[index]:.6f} "
            f"mujoco_dof_index={reference.v_indices[index]} "
            f"mujoco_tau={mj_tau[index]:.6f} "
            f"sign_match={'YES' if sign_ok else 'NO'} magnitude_ratio={ratio:.6f}")
        if not sign_ok or not math.isfinite(ratio) or not 0.25 <= ratio <= 4.0:
            raise RuntimeError(f"gravity runtime sanity failed for {name}")


def apply_target_rate_limit(desired, previous, dt, rate_limit):
    max_step = rate_limit * dt
    return previous + np.clip(desired - previous, -max_step, max_step)


def move_to_start_target(current, target, elapsed, duration):
    """Interpolate every selected joint with one shared smoothstep phase."""
    fraction = float(np.clip(elapsed / duration, 0.0, 1.0))
    alpha = 3.0 * fraction * fraction - 2.0 * fraction * fraction * fraction
    return (1.0 - alpha) * np.asarray(current) + alpha * np.asarray(target)


def playback_trajectory_time(elapsed, speed, duration):
    """Map one playback clock to the shared recorded trajectory timeline."""
    return min(float(elapsed) * float(speed), float(duration))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--arm", choices=("left", "right", "both"),
                        help="must match NPZ arm_mode; defaults to file metadata")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--validate-only", action="store_true",
                        help="load and validate NPZ without DDS or robot I/O")
    parser.add_argument("--max-playback-time", type=float,
                        help="play only the first N seconds; omit for the full recording")
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument("--start-error-soft-limit", type=float, default=0.5)
    parser.add_argument("--start-error-hard-limit", type=float, default=2.0)
    parser.add_argument("--move-to-start-duration", type=float, default=3.0)
    parser.add_argument("--move-to-start-max-velocity", type=float, default=0.25)
    parser.add_argument("--move-to-start-timeout", type=float, default=3.0,
                        help="maximum SETTLE_AT_START wait after the planned move")
    parser.add_argument("--start-tracking-tolerance", type=float, default=0.05)
    parser.add_argument("--start-settle-time", type=float, default=0.3)
    parser.add_argument("--hold-final-duration", type=float, default=1.5)
    parser.add_argument("--target-rate-limit", type=float, default=1.5)
    parser.add_argument("--tracking-error-limit", type=float, default=0.25)
    parser.add_argument("--tracking-error-timeout", type=float, default=0.25)
    parser.add_argument("--recorded-dq-limit", type=float, default=1.5)
    parser.add_argument("--finite-difference-dq-limit", type=float, default=1.5)
    parser.add_argument("--dq-sanity-error-limit", type=float, default=0.25)
    parser.add_argument("--dq-limit", type=float, default=2.0)
    parser.add_argument("--tau-est-limit", type=float, default=25.0)
    parser.add_argument("--gravity-tau-limit", type=float, default=8.0)
    parser.add_argument("--passive-timeout", type=float, default=2.0)
    parser.add_argument("--gravity-urdf", default="/home/hebe/unitree_workspace/src/unitree_ros/robots/g1_description/g1_29dof_mode_15.urdf")
    parser.add_argument("--gravity-model", default=str(root / "assets/robots/unitree_g1/xmls/g1.xml"))
    parser.add_argument("--log-dir", default="logs/test")
    parser.add_argument("--shoulder-kp", type=float, default=20.0)
    parser.add_argument("--shoulder-kd", type=float, default=2.0)
    parser.add_argument("--wrist-kp", type=float, default=12.0)
    parser.add_argument("--wrist-kd", type=float, default=1.5)
    args = parser.parse_args()

    positive = ("rate", "timeout", "start_error_soft_limit",
                "start_error_hard_limit", "move_to_start_duration",
                "move_to_start_max_velocity", "move_to_start_timeout",
                "start_tracking_tolerance", "start_settle_time",
                "hold_final_duration", "target_rate_limit",
                "tracking_error_limit", "tracking_error_timeout",
                "recorded_dq_limit", "finite_difference_dq_limit",
                "dq_sanity_error_limit", "dq_limit", "tau_est_limit",
                "gravity_tau_limit", "passive_timeout", "speed")
    if any(getattr(args, name) <= 0.0 for name in positive):
        parser.error("control durations and safety limits must be positive")
    if args.start_error_soft_limit >= args.start_error_hard_limit:
        parser.error("--start-error-soft-limit must be less than --start-error-hard-limit")
    if args.max_playback_time is not None and args.max_playback_time <= 0.0:
        parser.error("--max-playback-time must be positive")
    if min(args.shoulder_kp, args.shoulder_kd, args.wrist_kp, args.wrist_kd) < 0.0:
        parser.error("playback gains must be non-negative")

    input_path = args.input.expanduser().resolve()
    logger = TestLogger(args, "arm_playback")
    print(f"LOG_FILE={logger.log_path}")
    logger.info("STATE = LOAD_TRAJECTORY")
    try:
        trajectory = load_trajectory(
            input_path, args.recorded_dq_limit,
            min(args.finite_difference_dq_limit,
                args.target_rate_limit / args.speed),
            args.dq_sanity_error_limit, args.arm)
    except ValueError as exc:
        logger.error(f"ABORT_BEFORE_USERCTRL=YES OWNERSHIP_ACQUIRED=NO trajectory validation failed: {exc}")
        logger.close()
        return 2
    spec = trajectory.arm_spec
    selected_indices, selected_names = spec.motor_indices, spec.joint_names
    selected_config = playback_config(
        spec, args.shoulder_kp, args.shoulder_kd, args.wrist_kp, args.wrist_kd)
    logger.info("FILE_NAME_ARM_HINT=IGNORED")
    logger.info(f"FILE_ARM_MODE={spec.arm_mode} JOINT_COUNT={spec.joint_count}")
    logger.info("JOINT_NAMES=" + ",".join(selected_names))
    logger.info("MOTOR_INDICES=" + ",".join(map(str, selected_indices)))
    logger.info("STATE = VALIDATE_TRAJECTORY TRAJECTORY_VALID=YES")
    if args.validate_only:
        logger.result("TRAJECTORY_VALIDATION=PASS\nOWNERSHIP_ACQUIRED=NO\nREAL_ROBOT_PLAYBACK_STARTED=NO")
        logger.close()
        return 0
    if motor_control.ChannelFactoryInitialize is None:
        logger.error(f"ABORT_BEFORE_USERCTRL=YES unitree_sdk2py unavailable: {getattr(motor_control, '_SDK_ERROR', 'unknown')}")
        logger.close()
        return 2

    stop = threading.Event()
    signal_reason = [None]

    def on_sigint(*_):
        signal_reason[0] = "SIGINT"
        stop.set()

    signal.signal(signal.SIGINT, on_sigint)
    state = State()
    ChannelFactoryInitialize(0, args.interface)
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(state.update, 10)
    pub = loco = None
    ownership_acquired = restore_ok = False
    final_fsm_id = -1
    confirmed = False
    move_to_start_started = move_to_start_completed = False
    move_to_start_pass = start_settle_pass = playback_timer_started = False
    playback_started = playback_completed = False
    hold_final_pass = False
    safety_abort = False
    termination_reason = "NOT_STARTED"
    max_tracking = np.zeros(spec.joint_count)
    sum_squared_tracking = np.zeros(spec.joint_count)
    tracking_samples = 0
    max_abs_dq = np.zeros(spec.joint_count)
    max_abs_tau_est = np.zeros(spec.joint_count)
    max_abs_tau_ff = np.zeros(spec.joint_count)
    clamp_active = [False] * spec.joint_count
    last_targets = None
    max_start_error = math.nan
    start_pose_soft_warning = False
    start_pose_hard_abort = False
    move_to_start_duration = math.nan
    trajectory_valid = True
    playback_duration = 0.0
    logger.info("STATE = WAIT_LOWSTATE")
    try:
        while state.snapshot()[0] is None and not stop.is_set():
            time.sleep(0.01)
        if stop.is_set():
            raise RuntimeError("interrupted before LowState")
        msg, received = state.snapshot()
        now = time.monotonic()
        if msg is None or now - received > args.timeout:
            raise RuntimeError("invalid or stale LowState")

        playback_duration = min(
            trajectory.duration,
            args.max_playback_time if args.max_playback_time is not None
            else trajectory.duration)
        max_recorded_dq = (np.max(np.abs(trajectory.dq), axis=0)
                           if trajectory.dq is not None else None)
        max_fd_dq = np.max(np.abs(trajectory.finite_difference_dq), axis=0)
        rms_dq_difference = (np.sqrt(np.mean(
            (trajectory.finite_difference_dq - trajectory.dq[1:]) ** 2,
            axis=0)) if trajectory.dq is not None else None)
        logger.info(f"TRAJECTORY_FILE={trajectory.path}")
        logger.info(
            f"SAMPLE_COUNT={len(trajectory.q)} DURATION={trajectory.duration:.6f} "
            f"PLAYBACK_DURATION={playback_duration:.6f} "
            f"ACTUAL_RECORD_RATE={trajectory.metadata.get('actual_record_rate_hz')} "
            f"RECORDING_COMPLETE=YES "
            f"TERMINATION_REASON={trajectory.metadata.get('termination_reason')}")
        logger.info("JOINT_NAMES=" + ",".join(selected_names))
        logger.info("MOTOR_INDICES=" + ",".join(map(str, selected_indices)))
        logger.info("Q_START=" + np.array2string(trajectory.q[0], precision=6))
        logger.info("Q_END=" + np.array2string(
            interpolate_q(trajectory, playback_duration), precision=6))
        for column, name in enumerate(selected_names):
            logger.info(
                f"OFFLINE_VELOCITY_AUDIT joint={name} "
                f"MAX_RECORDED_DQ={'NOT_AVAILABLE' if max_recorded_dq is None else f'{max_recorded_dq[column]:.6f}'} "
                f"MAX_FINITE_DIFFERENCE_DQ={max_fd_dq[column]:.6f} "
                f"RMS_DQ_DIFFERENCE={'NOT_AVAILABLE' if rms_dq_difference is None else f'{rms_dq_difference[column]:.6f}'}")
        logger.info("PLAYBACK_CONFIG=" + json.dumps(
            selected_config, sort_keys=True))
        logger.info("GRAVITY_CONFIG=" + json.dumps({
            name: {"enabled": spec.config()[name]["gravity_enabled"],
                   "scale": spec.config()[name]["gravity_scale"]}
            for name in selected_names
        }, sort_keys=True))
        logger.info("GRAVITY_BACKEND=PINOCCHIO_URDF GRAVITY_REFERENCE_BACKEND=MUJOCO BASE_ORIENTATION_ASSUMED_LEVEL=YES")

        logger.info("STATE = CHECK_FSM")
        logger.info("STATE = REQUIRE_PASSIVE")
        loco = LocoClient()
        loco.Init()
        code, fsm_id = loco.GetFsmId()
        logger.info(f"CURRENT_FSM_ID={fsm_id} CURRENT_FSM_NAME={fsm_name(fsm_id)}")
        if code != 0 or fsm_id != 1:
            raise RuntimeError(
                f"ABORT_BEFORE_USERCTRL: robot must be PASSIVE/DAMPING "
                f"(fsm={fsm_name(fsm_id)})")

        try:
            gravity = PinocchioGravity(args.gravity_urdf)
            reference = MujocoGravity(args.gravity_model)
        except Exception as exc:
            raise RuntimeError(f"MODEL_LOAD_FAILED: {exc}") from exc

        logger.info("STATE = CAPTURE_CURRENT_Q")
        q_full, dq_full, tau_est_full = full_state(msg)
        validate_runtime_state(q_full, dq_full, tau_est_full, args, spec)
        current_q = q_full[list(selected_indices)].copy()
        start_q = trajectory.q[0].copy()
        logger.info("STATE = START_POSE_CHECK")
        start_error = start_q - current_q
        abs_start_error = np.abs(start_error)
        max_start_error = float(np.max(abs_start_error))
        start_pose_soft_warning = bool(
            np.any(abs_start_error > args.start_error_soft_limit))
        start_pose_hard_abort = bool(
            np.any(abs_start_error > args.start_error_hard_limit))
        logger.info("CURRENT_Q=" + np.array2string(current_q, precision=6))
        logger.info("START_Q=" + np.array2string(start_q, precision=6))
        logger.info("START_ERROR=" + np.array2string(start_error, precision=6))
        logger.info("START_ERROR_PER_JOINT=" + json.dumps(
            dict(zip(selected_names, start_error.tolist())), sort_keys=True))
        logger.info(f"MAX_START_ERROR={max_start_error:.6f}")
        max_start_column = int(np.argmax(abs_start_error))
        logger.info(
            f"START_ERROR_CHECK={'FAIL' if start_pose_hard_abort else 'PASS'} "
            f"MAX_START_ERROR_JOINT={selected_names[max_start_column]} "
            f"CURRENT={current_q[max_start_column]:.6f} "
            f"TARGET={start_q[max_start_column]:.6f} "
            f"ERROR={start_error[max_start_column]:.6f} "
            f"LIMIT={args.start_error_hard_limit:.6f}")
        logger.info(
            f"START_POSE_SOFT_WARNING={'YES' if start_pose_soft_warning else 'NO'} "
            f"START_POSE_HARD_ABORT={'YES' if start_pose_hard_abort else 'NO'}")
        if start_pose_hard_abort:
            raise RuntimeError(
                f"ABORT_BEFORE_USERCTRL: start error exceeds hard limit "
                f"{args.start_error_hard_limit}: "
                f"{start_error.tolist()}")
        if start_pose_soft_warning:
            logger.warning(
                "START_POSE_SOFT_LIMIT_EXCEEDED=YES; using automatic "
                "SLOW_MOVE_TO_START")

        # Cubic smoothstep peaks at 1.5 times its average velocity.
        velocity_duration = float(
            np.max(1.5 * abs_start_error / args.move_to_start_max_velocity))
        move_to_start_duration = max(args.move_to_start_duration,
                                     velocity_duration)
        logger.info(
            f"MOVE_TO_START_DURATION={move_to_start_duration:.6f} "
            f"MOVE_TO_START_MAX_VELOCITY={args.move_to_start_max_velocity:.6f}")

        runtime_gravity_sanity(gravity, reference, q_full, logger, spec)
        logger.info("STATE = USER_CONFIRM")
        try:
            confirmed = input(f"Type YES to start {spec.arm_mode.upper()} ARM PLAYBACK: ") == "YES"
        except (EOFError, KeyboardInterrupt):
            confirmed = False
        if not confirmed:
            termination_reason = "OPERATOR_DID_NOT_CONFIRM"
            logger.info("OWNERSHIP_ACQUIRED=NO PLAYBACK_NOT_STARTED")
        else:
            pub = ChannelPublisher("rt/user_lowcmd", LowCmd_)
            pub.Init()
            pub.Write(make_ownership_command(msg, STOP_DAMPING_KD))
            logger.info("STATE = ENTER_USERCTRL")
            ownership_acquired = loco.SwitchToUserCtrl() == 0
            if not ownership_acquired:
                raise RuntimeError("SwitchToUserCtrl failed")

            period = 1.0 / args.rate
            last_cycle = time.monotonic()
            phase_start = last_cycle
            move_start_q = current_q.copy()
            last_targets = current_q.copy()
            excessive_since = [None] * spec.joint_count

            def run_cycle(desired, phase, phase_time, trajectory_time,
                          target_rate_limit=None):
                nonlocal last_cycle, last_targets, tracking_samples
                cycle_start = now_cycle = time.monotonic()
                dt = min(max(now_cycle - last_cycle, 1e-6), 0.1)
                last_cycle = now_cycle
                current_msg, received_at = state.snapshot()
                if current_msg is None or now_cycle - received_at > args.timeout:
                    raise RuntimeError("LowState timeout")
                q_now, dq_now, tau_est_now = full_state(current_msg)
                validate_runtime_state(q_now, dq_now, tau_est_now, args, spec)
                targets = apply_target_rate_limit(
                    np.asarray(desired, dtype=np.float64), last_targets, dt,
                    (args.target_rate_limit if target_rate_limit is None
                     else target_rate_limit))
                for column, index in enumerate(selected_indices):
                    lo, hi = spec.joint_limits[column]
                    if not lo <= targets[column] <= hi:
                        raise RuntimeError(f"q_target joint limit: {selected_names[column]}")
                if not np.all(np.isfinite(targets)):
                    raise RuntimeError("q_target contains NaN/Inf")
                raw_gravity, tau_ff = gravity_command(
                    gravity, q_now, args, logger, clamp_active, spec)
                selected_q = q_now[list(selected_indices)]
                selected_dq = dq_now[list(selected_indices)]
                selected_tau_est = tau_est_now[list(selected_indices)]
                tracking = targets - selected_q
                for column in range(spec.joint_count):
                    if abs(tracking[column]) > args.tracking_error_limit:
                        excessive_since[column] = excessive_since[column] or now_cycle
                        if now_cycle - excessive_since[column] >= args.tracking_error_timeout:
                            raise RuntimeError(
                                f"TRACKING_ERROR_TOO_LARGE joint={selected_names[column]} "
                                f"error={tracking[column]:.6f}")
                    else:
                        excessive_since[column] = None
                max_tracking[:] = np.maximum(max_tracking, np.abs(tracking))
                sum_squared_tracking[:] += tracking ** 2
                max_abs_dq[:] = np.maximum(max_abs_dq, np.abs(selected_dq))
                max_abs_tau_est[:] = np.maximum(max_abs_tau_est, np.abs(selected_tau_est))
                max_abs_tau_ff[:] = np.maximum(max_abs_tau_ff, np.abs(tau_ff))
                tracking_samples += 1
                pub.Write(make_playback_command(current_msg, targets, tau_ff,
                                                selected_config, spec))
                log_parts = []
                for column, name in enumerate(selected_names):
                    gains = selected_config[name]
                    log_parts.append(
                        f"{name}:q_actual={selected_q[column]:.5f},q_target={targets[column]:.5f},"
                        f"tracking_error={tracking[column]:.5f},dq={selected_dq[column]:.5f},"
                        f"tau_est={selected_tau_est[column]:.5f},kp={gains['kp']:.2f},"
                        f"kd={gains['kd']:.2f},tau_gravity={raw_gravity[column]:.5f},"
                        f"tau_ff={tau_ff[column]:.5f}")
                logger.debug(
                    f"timestamp={time.time():.6f} phase={phase} playback_time={phase_time:.6f} "
                    f"trajectory_time={trajectory_time:.6f} " + " | ".join(log_parts))
                last_targets = targets
                time.sleep(max(0.0, period - (time.monotonic() - cycle_start)))
                return q_now, tracking

            logger.info("STATE = MOVE_TO_START PLAYBACK_TIME=NOT_STARTED TRAJECTORY_TIME=0")
            move_to_start_started = True
            while not stop.is_set():
                elapsed = time.monotonic() - phase_start
                fraction = min(elapsed / move_to_start_duration, 1.0)
                desired = move_to_start_target(
                    move_start_q, start_q, elapsed, move_to_start_duration)
                _, tracking = run_cycle(
                    desired, "MOVE_TO_START", elapsed, 0.0,
                    args.move_to_start_max_velocity)
                if fraction >= 1.0:
                    move_to_start_completed = True
                    break
            if stop.is_set():
                raise RuntimeError(signal_reason[0] or "operator stop")

            logger.info("STATE = SETTLE_AT_START PLAYBACK_TIME=NOT_STARTED TRAJECTORY_TIME=0")
            settle_started = time.monotonic()
            stable_since = None
            while not stop.is_set():
                elapsed = time.monotonic() - settle_started
                _, tracking = run_cycle(
                    start_q, "SETTLE_AT_START", elapsed, 0.0,
                    args.move_to_start_max_velocity)
                if np.max(np.abs(tracking)) < args.start_tracking_tolerance:
                    stable_since = stable_since or time.monotonic()
                    if time.monotonic() - stable_since >= args.start_settle_time:
                        start_settle_pass = True
                        move_to_start_pass = True
                        break
                else:
                    stable_since = None
                if elapsed > args.move_to_start_timeout:
                    raise RuntimeError("SETTLE_AT_START tracking timeout")
            if stop.is_set():
                raise RuntimeError(signal_reason[0] or "operator stop")

            logger.info("STATE = PLAYBACK")
            playback_started = True
            playback_start = time.monotonic()
            playback_timer_started = True
            logger.info(f"LEFT_RIGHT_TIME_SYNC=YES SPEED={args.speed:.6f}")
            while not stop.is_set():
                elapsed = time.monotonic() - playback_start
                trajectory_time = playback_trajectory_time(
                    elapsed, args.speed, playback_duration)
                desired = interpolate_q(trajectory, trajectory_time)
                run_cycle(desired, "PLAYBACK", elapsed, trajectory_time)
                if trajectory_time >= playback_duration:
                    playback_completed = True
                    break
            if stop.is_set():
                raise RuntimeError(signal_reason[0] or "operator stop")

            logger.info("STATE = HOLD_FINAL")
            hold_start = time.monotonic()
            final_q = interpolate_q(trajectory, playback_duration)
            while not stop.is_set() and time.monotonic() - hold_start < args.hold_final_duration:
                elapsed = time.monotonic() - hold_start
                run_cycle(final_q, "HOLD_FINAL", elapsed, playback_duration)
            if stop.is_set():
                raise RuntimeError(signal_reason[0] or "operator stop")
            hold_final_pass = True
            termination_reason = "NORMAL"
    except Exception as exc:
        termination_reason = f"{type(exc).__name__}: {exc}"
        before_ownership = not ownership_acquired
        safety_abort = ownership_acquired
        logger.error(
            f"{'ABORT_BEFORE_USERCTRL' if before_ownership else 'SAFETY_ABORT'}=YES "
            f"REASON={termination_reason}")
    finally:
        if ownership_acquired:
            logger.info(f"STATE = CONTROLLED_STOP ALL_{spec.arm_mode.upper()}_ARM_TAU=0")
            stop_deadline = time.monotonic() + 0.2
            while time.monotonic() < stop_deadline:
                try:
                    current_msg = state.snapshot()[0]
                    if pub is not None and current_msg is not None:
                        pub.Write(make_ownership_command(current_msg, STOP_DAMPING_KD))
                except Exception as exc:
                    logger.error(f"controlled stop publish failed: {exc}")
                    break
                time.sleep(0.01)
            try:
                restore_ok = loco.SwitchToInternalCtrl(InternalFsmMode.PASSIVE) == 0
            except Exception as exc:
                logger.error(f"SwitchToInternalCtrl(PASSIVE) failed: {exc}")
                restore_ok = False
            logger.info("STATE = RETURN_PASSIVE")
            deadline = time.monotonic() + args.passive_timeout
            while time.monotonic() < deadline:
                try:
                    code, final_fsm_id = loco.GetFsmId()
                    if code == 0 and final_fsm_id == 1:
                        break
                except Exception:
                    pass
                time.sleep(0.05)

        safe_exit = restore_ok and final_fsm_id == 1 if ownership_acquired else None
        rms_tracking = (np.sqrt(sum_squared_tracking / tracking_samples)
                        if tracking_samples else np.zeros(spec.joint_count))
        logger.result(f"TRAJECTORY_VALID={'YES' if trajectory_valid else 'NO'}")
        logger.result(
            f"{spec.arm_mode.upper()}_ARM_TRAJECTORY_LOAD={'PASS' if trajectory_valid else 'FAIL'}")
        logger.result("START_ERROR_PER_JOINT=" + json.dumps(
            dict(zip(selected_names, [] if math.isnan(max_start_error)
                     else start_error.tolist())), sort_keys=True))
        logger.result(
            f"MAX_START_ERROR={'NOT_AVAILABLE' if math.isnan(max_start_error) else f'{max_start_error:.6f}'}")
        logger.result(
            f"START_POSE_SOFT_WARNING={'YES' if start_pose_soft_warning else 'NO'}")
        logger.result(
            f"START_POSE_HARD_ABORT={'YES' if start_pose_hard_abort else 'NO'}")
        logger.result(
            f"MOVE_TO_START_DURATION={'NOT_AVAILABLE' if math.isnan(move_to_start_duration) else f'{move_to_start_duration:.6f}'}")
        logger.result(
            f"MOVE_TO_START_MAX_VELOCITY={args.move_to_start_max_velocity:.6f}")
        logger.result(
            f"MOVE_TO_START_STARTED={'YES' if move_to_start_started else 'NO'}")
        logger.result(
            f"MOVE_TO_START_COMPLETED={'YES' if move_to_start_completed else 'NO'}")
        logger.result(f"MOVE_TO_START_PASS={'YES' if move_to_start_pass else 'NO'}")
        logger.result(f"MOVE_TO_START={'PASS' if move_to_start_pass else 'FAIL'}")
        logger.result(f"START_SETTLE_PASS={'YES' if start_settle_pass else 'NO'}")
        logger.result(
            f"PLAYBACK_TIMER_STARTED={'YES' if playback_timer_started else 'NO'}")
        logger.result(f"PLAYBACK_STARTED={'YES' if playback_started else 'NO'}")
        logger.result(f"PLAYBACK_COMPLETED={'YES' if playback_completed else 'NO'}")
        for column, name in enumerate(selected_names):
            logger.result(
                f"JOINT={name} MAX_TRACKING_ERROR={max_tracking[column]:.6f} "
                f"RMS_TRACKING_ERROR={rms_tracking[column]:.6f} "
                f"MAX_ABS_DQ={max_abs_dq[column]:.6f} "
                f"MAX_ABS_TAU_EST={max_abs_tau_est[column]:.6f} "
                f"MAX_ABS_TAU_FF={max_abs_tau_ff[column]:.6f}")
        logger.result("MAX_TRACKING_ERROR_PER_JOINT=" + json.dumps(
            dict(zip(selected_names, max_tracking.tolist())), sort_keys=True))
        logger.result("RMS_TRACKING_ERROR_PER_JOINT=" + json.dumps(
            dict(zip(selected_names, rms_tracking.tolist())), sort_keys=True))
        logger.result("MAX_ABS_DQ_PER_JOINT=" + json.dumps(
            dict(zip(selected_names, max_abs_dq.tolist())), sort_keys=True))
        logger.result("MAX_ABS_TAU_EST_PER_JOINT=" + json.dumps(
            dict(zip(selected_names, max_abs_tau_est.tolist())), sort_keys=True))
        logger.result("MAX_ABS_TAU_FF_PER_JOINT=" + json.dumps(
            dict(zip(selected_names, max_abs_tau_ff.tolist())), sort_keys=True))
        logger.result(f"SAFETY_ABORT={'YES' if safety_abort else 'NO'}")
        logger.result(f"HOLD_FINAL_PASS={'YES' if hold_final_pass else 'NO'}")
        logger.result(
            "SAFE_EXIT_TO_PASSIVE=" +
            ("YES" if safe_exit is True else "NO" if safe_exit is False else "NOT_APPLICABLE"))
        logger.result(f"OWNERSHIP_ACQUIRED={'YES' if ownership_acquired else 'NO'}")
        logger.result(f"TERMINATION_REASON={termination_reason}")
        program_pass = (ownership_acquired and move_to_start_pass and playback_started
                        and playback_completed and hold_final_pass and not safety_abort
                        and safe_exit is True)
        logger.result(f"TRACKING_STABLE={'YES' if program_pass else 'NO'}")
        logger.result(f"NO_OSCILLATION=OPERATOR\n{spec.arm_mode.upper()}_ARM_PLAYBACK_REAL_ROBOT=OPERATOR")
        logger.result(f"{spec.arm_mode.upper()}_ARM_TRAJECTORY_PLAYBACK_PIPELINE={'PASS' if program_pass else 'FAIL'}")
        logger.close()
    return 0 if program_pass else (1 if confirmed else 2)


if __name__ == "__main__":
    raise SystemExit(main())
