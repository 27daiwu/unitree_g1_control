import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from g1_piano.common.arm_spec import get_arm_spec
from playback_left_arm import (interpolate_q, load_trajectory,
                               make_playback_command, move_to_start_target,
                               playback_config, playback_trajectory_time)


def write_arm_npz(path, mode="left", *, timestamps=None, q=None, dq=True,
                  names=None, indices=None, top_level=True):
    spec = get_arm_spec(mode)
    timestamps = (np.array([20.0, 20.1, 20.2], dtype=np.float64)
                  if timestamps is None else np.asarray(timestamps, dtype=np.float64))
    if q is None:
        q = np.zeros((len(timestamps), spec.joint_count), dtype=np.float32)
        q[1:] = np.arange(1, len(timestamps))[:, None] * 0.01
    q = np.asarray(q, dtype=np.float32)
    metadata = {
        "arm_mode": mode,
        "joint_names": list(spec.joint_names if names is None else names),
        "motor_indices": list(spec.motor_indices if indices is None else indices),
        "recording_complete": True,
        "sample_count": len(timestamps),
        "duration": float(timestamps[-1] - timestamps[0]),
        "actual_record_rate_hz": 10.0,
        "termination_reason": "NORMAL",
    }
    values = {
        "timestamp_monotonic": timestamps, "q": q,
        "metadata": np.asarray(json.dumps(metadata)),
    }
    if dq:
        values["dq"] = np.full_like(q, 0.1)
    if top_level:
        values.update(arm_mode=np.asarray(mode),
                      joint_names=np.asarray(metadata["joint_names"]),
                      motor_indices=np.asarray(metadata["motor_indices"]))
    np.savez(path, **values)


@pytest.mark.parametrize("mode,count", [("left", 7), ("right", 7), ("both", 14)])
def test_load_arm_modes_and_shapes(tmp_path, mode, count):
    path = tmp_path / "misleading_filename.npz"
    write_arm_npz(path, mode)
    trajectory = load_trajectory(path, 1.0, 1.0, 0.1)
    assert trajectory.arm_spec.arm_mode == mode
    assert trajectory.q.shape == (3, count)


@pytest.mark.parametrize("mode", ["left", "right", "both"])
def test_rejects_wrong_q_shape_for_each_mode(tmp_path, mode):
    path = tmp_path / "wrong_shape.npz"
    spec = get_arm_spec(mode)
    write_arm_npz(path, mode, q=np.zeros((3, spec.joint_count + 1)), dq=False)
    with pytest.raises(ValueError, match="q must have shape"):
        load_trajectory(path, 1.0, 1.0, 0.1)


def test_cli_metadata_match_and_mismatch(tmp_path):
    path = tmp_path / "right_arm_name_but_both_data.npz"
    write_arm_npz(path, "both")
    assert load_trajectory(path, 1.0, 1.0, 0.1, "both").arm_spec.arm_mode == "both"
    with pytest.raises(ValueError, match="ARM_MODE_MISMATCH.*FILE_ARM_MODE=both.*REQUESTED_ARM_MODE=right"):
        load_trajectory(path, 1.0, 1.0, 0.1, "right")


@pytest.mark.parametrize("field,value,match", [
    ("indices", list(range(14, 21)), "motor_indices mismatch"),
    ("indices", [15] * 7, "duplicate motor_indices"),
    ("names", ["invalid"] * 7, "duplicate joint_names"),
])
def test_rejects_invalid_identity(tmp_path, field, value, match):
    path = tmp_path / "invalid.npz"
    kwargs = {field: value}
    write_arm_npz(path, "left", top_level=False, **kwargs)
    with pytest.raises(ValueError, match=match):
        load_trajectory(path, 1.0, 1.0, 0.1)


def test_rejects_nonmonotonic_timestamp(tmp_path):
    path = tmp_path / "bad_time.npz"
    write_arm_npz(path, timestamps=[1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        load_trajectory(path, 1.0, 1.0, 1.0)


def test_rejects_nonfinite_timestamp(tmp_path):
    path = tmp_path / "bad_time.npz"
    write_arm_npz(path, timestamps=[1.0, np.inf, 2.0])
    with pytest.raises(ValueError, match="NaN/Inf"):
        load_trajectory(path, 1.0, 1.0, 1.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_rejects_nonfinite_q(tmp_path, bad):
    path = tmp_path / "bad_q.npz"
    q = np.zeros((3, 7)); q[1, 2] = bad
    write_arm_npz(path, q=q)
    with pytest.raises(ValueError, match="NaN/Inf"):
        load_trajectory(path, 1.0, 1.0, 1.0)


def test_dq_is_optional_but_validated_when_present(tmp_path):
    path = tmp_path / "no_dq.npz"
    write_arm_npz(path, dq=False)
    assert load_trajectory(path, 1.0, 1.0, 0.1).dq is None


def test_both_interpolation_uses_one_time_for_all_joints(tmp_path):
    path = tmp_path / "both.npz"
    spec = get_arm_spec("both")
    q = 0.01 * np.vstack((np.zeros(14), np.arange(14), 2 * np.arange(14))).astype(float)
    write_arm_npz(path, "both", q=q, dq=False)
    trajectory = load_trajectory(path, 100.0, 100.0, 100.0)
    np.testing.assert_allclose(interpolate_q(trajectory, 0.15), 0.015 * np.arange(14))
    assert trajectory.arm_spec.motor_indices == spec.motor_indices


def test_both_move_to_start_uses_one_phase_for_all_joints():
    current = np.arange(14, dtype=float)
    target = current + np.arange(1, 15, dtype=float)
    result = move_to_start_target(current, target, elapsed=1.0, duration=2.0)
    np.testing.assert_allclose((result - current) / (target - current), 0.5)


@pytest.mark.parametrize("speed,expected", [(0.5, 1.0), (1.0, 2.0), (1.5, 3.0)])
def test_playback_speed_uses_shared_recorded_timeline(speed, expected):
    assert playback_trajectory_time(2.0, speed, 10.0) == expected


@pytest.mark.parametrize("mode", ["left", "right", "both"])
def test_command_enables_only_selected_arm(monkeypatch, mode):
    spec = get_arm_spec(mode)
    motors = [SimpleNamespace(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=0)
              for _ in range(35)]
    monkeypatch.setattr("playback_left_arm.create_lowcmd",
                        lambda: SimpleNamespace(motor_cmd=motors))
    msg = SimpleNamespace(motor_state=[SimpleNamespace(q=float(i)) for i in range(29)])
    config = playback_config(spec)
    command = make_playback_command(msg, np.zeros(spec.joint_count),
                                    np.zeros(spec.joint_count), config, spec)
    for index, motor in enumerate(command.motor_cmd[:29]):
        assert motor.mode == (1 if index in spec.motor_indices else 0)
    for name in spec.joint_names:
        expected_kp = 12.0 if "_wrist_" in name else 20.0
        assert config[name]["kp"] == expected_kp
