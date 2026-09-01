"""Kinematic playback of the G1 raw recorder NPZ in MuJoCo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import mujoco

from g1_piano.common.joint_map import G1_29DOF_JOINT_NAMES

Sampling = Literal["nearest", "linear"]


def clip_playback_range(duration: float, start: float = 0.0, end: float | None = None) -> tuple[float, float]:
    clipped_start = min(max(0.0, float(start)), duration)
    clipped_end = duration if end is None else min(max(0.0, float(end)), duration)
    if clipped_end < clipped_start:
        raise ValueError("end must be greater than or equal to start")
    return clipped_start, clipped_end


def seek_seconds(current: float, delta: float, start: float, end: float) -> float:
    return float(np.clip(current + delta, start, end))


@dataclass(frozen=True)
class RawNPZ:
    data: dict[str, np.ndarray]
    q: np.ndarray
    timestamps: np.ndarray

    @property
    def times(self) -> np.ndarray:
        return self.timestamps - self.timestamps[0]

    @property
    def duration(self) -> float:
        return float(self.times[-1])


def load_raw_npz(path: str | Path, joint_key: str = "q", time_key: str = "timestamp_monotonic") -> RawNPZ:
    with np.load(path, allow_pickle=True) as src:
        data = {key: src[key] for key in src.files}
    if joint_key not in data:
        raise ValueError(f"missing joint field {joint_key!r}; available={list(data)}")
    if time_key not in data:
        raise ValueError(f"missing time field {time_key!r}; available={list(data)}")
    q = np.asarray(data[joint_key])
    t = np.asarray(data[time_key])
    if q.ndim != 2 or q.shape[1] != 29:
        raise ValueError(f"{joint_key} must have shape (T,29), got {q.shape}")
    if t.ndim != 1 or t.shape[0] != q.shape[0]:
        raise ValueError(f"{time_key} must have shape ({q.shape[0]},), got {t.shape}")
    if q.shape[0] == 0:
        raise ValueError("NPZ contains zero samples")
    if not np.all(np.isfinite(q)):
        raise ValueError(f"{joint_key} contains non-finite values")
    if not np.all(np.isfinite(t)):
        raise ValueError(f"{time_key} contains non-finite values")
    if q.shape[0] > 1 and not np.all(np.diff(t) > 0):
        raise ValueError(f"{time_key} must be strictly increasing")
    return RawNPZ(data=data, q=q.astype(np.float64, copy=False), timestamps=t.astype(np.float64, copy=False))


@dataclass(frozen=True)
class G1MujocoMapping:
    sdk_names: tuple[str, ...]
    mujoco_names: tuple[str, ...]
    joint_ids: tuple[int, ...]
    qpos_addresses: tuple[int, ...]

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "G1MujocoMapping":
        mj_names, ids, addresses = [], [], []
        missing = []
        for sdk_name in G1_29DOF_JOINT_NAMES:
            target = f"{sdk_name}_joint"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target)
            if jid < 0:
                missing.append((sdk_name, target))
                continue
            if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError(f"{target} is not a hinge joint")
            mj_names.append(target); ids.append(int(jid)); addresses.append(int(model.jnt_qposadr[jid]))
        if missing:
            raise ValueError("29DoF joint mapping incomplete: " + ", ".join(f"{s}->{m}" for s, m in missing))
        return cls(tuple(G1_29DOF_JOINT_NAMES), tuple(mj_names), tuple(ids), tuple(addresses))

    def print(self) -> None:
        for i, (sdk, mj, adr) in enumerate(zip(self.sdk_names, self.mujoco_names, self.qpos_addresses)):
            print(f"[{i:02d}] {sdk}\n     -> {mj}\n     -> qpos {adr}")


def nearest_sample_index(times: np.ndarray, playback_time: float) -> int:
    times = np.asarray(times)
    i = int(np.searchsorted(times, playback_time, side="left"))
    if i <= 0: return 0
    if i >= len(times): return len(times) - 1
    return i if playback_time - times[i - 1] >= times[i] - playback_time else i - 1


def interpolate_q(times: np.ndarray, q: np.ndarray, playback_time: float, sampling: Sampling = "nearest") -> tuple[np.ndarray, int]:
    if sampling == "nearest":
        i = nearest_sample_index(times, playback_time)
        return np.asarray(q[i]).copy(), i
    if sampling != "linear": raise ValueError(f"unsupported sampling: {sampling}")
    t = float(np.clip(playback_time, times[0], times[-1]))
    right = int(np.searchsorted(times, t, side="left"))
    if right <= 0: return np.asarray(q[0]).copy(), 0
    if right >= len(times): return np.asarray(q[-1]).copy(), len(times) - 1
    left = right - 1
    alpha = (t - times[left]) / (times[right] - times[left])
    return (1.0 - alpha) * q[left] + alpha * q[right], nearest_sample_index(times, t)


def initialize_data(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    # Prefer a named home keyframe if the selected model provides one.
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    data.qvel[:] = 0.0
    # MuJoCo freejoint quaternion is wxyz; identity is a safe fixed-base pose.
    free = [i for i in range(model.njnt) if int(model.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_FREE)]
    if free and np.linalg.norm(data.qpos[int(model.jnt_qposadr[free[0]]) + 3:int(model.jnt_qposadr[free[0]]) + 7]) < 1e-12:
        adr = int(model.jnt_qposadr[free[0]])
        data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)


def apply_q(model: mujoco.MjModel, data: mujoco.MjData, mapping: G1MujocoMapping, q_render: np.ndarray) -> None:
    for sdk_index, qpos_adr in enumerate(mapping.qpos_addresses):
        data.qpos[qpos_adr] = q_render[sdk_index]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
