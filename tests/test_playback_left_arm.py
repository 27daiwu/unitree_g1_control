import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from playback_left_arm import (
    LEFT_ARM_INDICES, LEFT_ARM_NAMES, PLAYBACK_CONFIG, PlaybackTrajectory,
    interpolate_q, load_trajectory, make_playback_command,
)


def write_trajectory(path, *, complete=True, names=None, indices=None):
    timestamps = np.array([10.0, 10.1, 10.2], dtype=np.float64)
    q = np.zeros((3, 7), dtype=np.float32)
    q[1] = 0.01
    q[2] = 0.02
    dq = np.full((3, 7), 0.1, dtype=np.float32)
    metadata = {
        "joint_names": list(LEFT_ARM_NAMES if names is None else names),
        "motor_indices": list(LEFT_ARM_INDICES if indices is None else indices),
        "recording_complete": complete,
        "sample_count": 3,
        "duration": 0.2,
        "actual_record_rate_hz": 10.0,
        "termination_reason": "NORMAL",
    }
    np.savez(path, timestamp_monotonic=timestamps, q=q, dq=dq,
             metadata=np.asarray(json.dumps(metadata)))


def test_load_and_interpolate_valid_trajectory(tmp_path):
    path = tmp_path / "valid.npz"
    write_trajectory(path)
    trajectory = load_trajectory(path, 1.0, 1.0, 0.1)
    np.testing.assert_allclose(interpolate_q(trajectory, 0.15), 0.015)
    assert trajectory.duration == pytest.approx(0.2)


@pytest.mark.parametrize("complete,names,indices", [
    (False, None, None),
    (True, ("wrong",) * 7, None),
    (True, None, range(14, 21)),
])
def test_rejects_incomplete_or_mismapped_trajectory(tmp_path, complete, names, indices):
    path = tmp_path / "invalid.npz"
    write_trajectory(path, complete=complete, names=names, indices=indices)
    with pytest.raises(ValueError):
        load_trajectory(path, 1.0, 1.0, 0.1)


def test_rejects_velocity_excess_before_robot_io(tmp_path):
    path = tmp_path / "fast.npz"
    write_trajectory(path)
    with np.load(path, allow_pickle=False) as source:
        values = {key: source[key] for key in source.files}
    values["q"] = values["q"].copy()
    values["q"][1, 0] = 0.5
    np.savez(path, **values)
    with pytest.raises(ValueError, match="finite-difference dq exceeds"):
        load_trajectory(path, 1.0, 1.0, 10.0)


def test_command_enables_only_left_arm(monkeypatch):
    motors = [SimpleNamespace(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=0)
              for _ in range(35)]
    monkeypatch.setattr("playback_left_arm.create_lowcmd",
                        lambda: SimpleNamespace(motor_cmd=motors))
    msg = SimpleNamespace(motor_state=[SimpleNamespace(q=float(i)) for i in range(29)])
    targets = np.arange(7, dtype=float) / 10.0
    tau_ff = np.arange(7, dtype=float) / 100.0
    cmd = make_playback_command(msg, targets, tau_ff)

    for index, motor in enumerate(cmd.motor_cmd):
        if index in LEFT_ARM_INDICES:
            column = index - 15
            name = LEFT_ARM_NAMES[column]
            assert motor.mode == 1
            assert motor.q == pytest.approx(targets[column])
            assert motor.kp == PLAYBACK_CONFIG[name]["kp"]
            assert motor.kd == PLAYBACK_CONFIG[name]["kd"]
            assert motor.tau == pytest.approx(tau_ff[column])
        else:
            assert motor.mode == 0
            assert motor.kp == 0.0
            assert motor.kd == 0.0
            assert motor.tau == 0.0
