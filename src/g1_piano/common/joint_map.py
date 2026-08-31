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


def joint_name(index: int) -> str:
    return G1_29DOF_JOINT_NAMES[index] if 0 <= index < 29 else "unused"
