"""G1 29-DoF motor-index to joint-name mapping.

Source: unitree_sdk2_python/example/g1/low_level/g1_low_level_example.py
and matching C++ example/g1/low_level/g1_ankle_swing_example.cpp.
"""

G1_29DOF_JOINT_NAMES = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)

assert len(G1_29DOF_JOINT_NAMES) == 29

JOINT_GROUPS = {
    "LEFT_LEG": tuple(range(0, 6)),
    "RIGHT_LEG": tuple(range(6, 12)),
    "WAIST": tuple(range(12, 15)),
    "LEFT_ARM": tuple(range(15, 22)),
    "RIGHT_ARM": tuple(range(22, 29)),
}
LEGS = JOINT_GROUPS["LEFT_LEG"] + JOINT_GROUPS["RIGHT_LEG"]
ARMS = JOINT_GROUPS["LEFT_ARM"] + JOINT_GROUPS["RIGHT_ARM"]
JOINT_TO_INDEX = {name: i for i, name in enumerate(G1_29DOF_JOINT_NAMES)}
INDEX_TO_GROUP = {i: group for group, indices in JOINT_GROUPS.items() for i in indices}


def resolve_joint(value: str) -> int:
    """Resolve a decimal motor index or an exact joint name."""
    try:
        index = int(value)
    except ValueError:
        try:
            return JOINT_TO_INDEX[value]
        except KeyError as exc:
            raise ValueError(f"unknown joint '{value}'") from exc
    if not 0 <= index < len(G1_29DOF_JOINT_NAMES):
        raise ValueError(f"motor index must be in [0, 28], got {index}")
    return index


def joint_name(index: int) -> str:
    return G1_29DOF_JOINT_NAMES[index] if 0 <= index < 29 else "unused"
