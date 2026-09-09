"""Shared left/right/both G1 arm specifications for TEACH and recording."""
from dataclasses import dataclass
from typing import Mapping, Tuple

from .joint_map import G1_29DOF_JOINT_NAMES, JOINT_GROUPS


@dataclass(frozen=True)
class ArmSpec:
    arm_mode: str
    joint_names: Tuple[str, ...]
    motor_indices: Tuple[int, ...]
    kp: Tuple[float, ...]
    kd: Tuple[float, ...]
    gravity_enabled: Tuple[bool, ...]
    gravity_scale: Tuple[float, ...]
    joint_limits: Tuple[Tuple[float, float], ...]

    @property
    def joint_count(self) -> int:
        return len(self.motor_indices)

    def config(self) -> Mapping[str, dict]:
        return {name: {"kp": self.kp[i], "kd": self.kd[i],
                       "gravity_enabled": self.gravity_enabled[i],
                       "gravity_scale": self.gravity_scale[i]}
                for i, name in enumerate(self.joint_names)}


_LEFT = tuple(JOINT_GROUPS["LEFT_ARM"])
_RIGHT = tuple(JOINT_GROUPS["RIGHT_ARM"])

# Baseline is the verified left-arm feel. Right arm uses the mirrored joint
# limits/mapping and the same gains/gravity strategy.
_KP = (8.0, 8.0, 8.0, 8.0, 6.0, 6.0, 6.0)
_KD = (1.5,) * 7
_GRAVITY = (True, True, True, True, False, False, False)
_GRAVITY_SCALE = (0.75, 0.80, 0.75, 0.75, 0.0, 0.0, 0.0)
_LIMITS = ((-3.0892, 2.6704), (-1.5882, 2.2515), (-2.618, 2.618),
           (-1.0472, 2.0944), (-1.97222, 1.97222), (-1.61443, 1.61443),
           (-1.61443, 1.61443))


def get_arm_spec(arm_mode: str) -> ArmSpec:
    mode = str(arm_mode).lower()
    if mode not in {"left", "right", "both"}:
        raise ValueError("arm must be one of: left, right, both")
    indices = _LEFT if mode == "left" else _RIGHT if mode == "right" else _LEFT + _RIGHT
    names = tuple(G1_29DOF_JOINT_NAMES[i] for i in indices)
    repeats = 1 if mode != "both" else 2
    right_limits = (_LIMITS[0], (-2.2515, 1.5882), *_LIMITS[2:])
    limits = _LIMITS if mode == "left" else right_limits if mode == "right" else _LIMITS + right_limits
    # Right shoulder-roll has the mirrored sign convention in the SDK limits.
    return ArmSpec(mode, names, indices, _KP * repeats, _KD * repeats,
                   _GRAVITY * repeats, _GRAVITY_SCALE * repeats, tuple(limits))


LEFT_ARM_SPEC = get_arm_spec("left")
RIGHT_ARM_SPEC = get_arm_spec("right")
BOTH_ARMS_SPEC = get_arm_spec("both")
