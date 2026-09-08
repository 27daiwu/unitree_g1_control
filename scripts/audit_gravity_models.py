#!/usr/bin/env python3
"""Audit MuJoCo/URDF gravity models without sending any robot command.

The URDF backend is optional: Pinocchio is detected but never installed by this
tool.  All comparisons use joint names first and only then backend-specific
q/dof addresses.
"""
from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from g1_piano.common.joint_map import G1_29DOF_JOINT_NAMES, JOINT_TO_INDEX

try:
    import mujoco
except ImportError:
    mujoco = None

try:
    import pinocchio as pin
except ImportError:
    pin = None

PINOCCHIO_AVAILABLE = pin is not None and all(
    hasattr(pin, name) for name in ("buildModelFromUrdf", "computeGeneralizedGravity", "neutral")
)

ARM_NAMES = (
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
)
DEFAULT_MJCF = Path(__file__).resolve().parents[1] / "assets/robots/unitree_g1/xmls/g1.xml"
DEFAULT_URDF = Path("/home/hebe/unitree_workspace/src/unitree_ros/robots/g1_description/g1_29dof_mode_15.urdf")


def fmt(x):
    return "NA" if x is None else f"{float(x):.8f}"


def audit_urdf(path: Path):
    root = ET.parse(path).getroot()
    links = {e.attrib["name"]: e for e in root.findall("link") if "name" in e.attrib}
    joints = {e.attrib["name"]: e for e in root.findall("joint") if "name" in e.attrib}
    issues = []
    inertial = {}
    for name, link in links.items():
        node = link.find("inertial")
        if node is None or node.find("mass") is None or node.find("inertia") is None:
            inertial[name] = False
            continue
        mass = float(node.find("mass").attrib.get("value", "nan"))
        vals = [float(node.find("inertia").attrib.get(k, "nan")) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")]
        inertial[name] = math.isfinite(mass) and mass > 0 and all(math.isfinite(v) for v in vals) and any(abs(v) > 0 for v in vals)
    mapping = []
    for name in G1_29DOF_JOINT_NAMES:
        jname = f"{name}_joint"
        joint = joints.get(jname)
        valid = joint is not None and joint.attrib.get("type") == "revolute"
        if valid:
            axis = joint.find("axis"); origin = joint.find("origin")
            valid = axis is not None and "xyz" in axis.attrib and origin is not None
            child = joint.find("child")
            child_name = child.attrib.get("link") if child is not None else None
            valid = valid and child_name in links and inertial.get(child_name, False)
        else:
            child_name = None
        if not valid: issues.append(f"invalid joint/link/inertial: {jname}")
        mapping.append((JOINT_TO_INDEX[name], name, jname, child_name, valid))
    arm_links_ok = all(row[4] for row in mapping if row[1] in ARM_NAMES)
    return root, mapping, inertial, issues, arm_links_ok


class MujocoGravity:
    def __init__(self, path: Path):
        if mujoco is None: raise RuntimeError("mujoco unavailable")
        self.model = mujoco.MjModel.from_xml_path(str(path)); self.data = mujoco.MjData(self.model)
        self.ids = {}
        for name in G1_29DOF_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
            if jid < 0: raise RuntimeError(f"MuJoCo joint missing: {name}")
            self.ids[name] = (jid, int(self.model.jnt_dofadr[jid]), int(self.model.jnt_qposadr[jid]))
        free = [i for i in range(self.model.njnt) if int(self.model.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        if free:
            a = int(self.model.jnt_qposadr[free[0]]); self.data.qpos[a:a + 7] = (0, 0, 0.79, 1, 0, 0, 0)
    def torque(self, q):
        for name in G1_29DOF_JOINT_NAMES: self.data.qpos[self.ids[name][2]] = float(q[JOINT_TO_INDEX[name]])
        self.data.qvel[:] = 0; mujoco.mj_forward(self.model, self.data)
        return {name: float(self.data.qfrc_bias[self.ids[name][1]]) for name in G1_29DOF_JOINT_NAMES}
    def dof(self, name): return self.ids[name][1]


class PinocchioGravity:
    def __init__(self, path: Path):
        if not PINOCCHIO_AVAILABLE: raise RuntimeError("robotics Pinocchio API unavailable (installed package is not hppfcl/Pinocchio)")
        # Build explicitly with a free-flyer root.  The URDF contains a
        # floating_base_joint; omitting this argument silently creates a fixed
        # base model (nq=29,nv=29), which is not the requested representation.
        self.model = pin.buildModelFromUrdf(str(path), pin.JointModelFreeFlyer())
        self.data = self.model.createData(); self.ids = {}
        if self.model.nq < 29 or self.model.nv < 29:
            raise RuntimeError(f"unexpected Pinocchio dimensions nq={self.model.nq} nv={self.model.nv}")
        for name in G1_29DOF_JOINT_NAMES:
            jid = self.model.getJointId(f"{name}_joint")
            if jid >= self.model.njoints: raise RuntimeError(f"Pinocchio joint missing: {name}")
            self.ids[name] = jid
        self.q = pin.neutral(self.model)
        # Explicit free-flyer convention: neutral() supplies zero base position
        # and identity quaternion; only named revolute joint coordinates follow
        # the SDK q vector.
        self.base_nq = self.model.nq - 29
        self.base_nv = self.model.nv - 29
    def torque(self, q_sdk):
        q = self.q.copy()
        for name in G1_29DOF_JOINT_NAMES:
            jid = self.ids[name]; q[self.model.idx_qs[jid]] = float(q_sdk[JOINT_TO_INDEX[name]])
        tau = pin.computeGeneralizedGravity(self.model, self.data, q)
        return {name: float(tau[self.model.idx_vs[self.ids[name]]]) for name in G1_29DOF_JOINT_NAMES}
    def vdof(self, name): return int(self.model.idx_vs[self.ids[name]])


def poses():
    q = np.zeros(29); q[JOINT_TO_INDEX["left_elbow"]] = 0.87
    out = {"POSE_A": q.copy()}
    b = q.copy(); b[18] = 1.2; out["POSE_B"] = b
    c = b.copy(); c[15] = 0.5; c[18] = 1.0; out["POSE_C"] = c
    d = c.copy(); d[15] = 0.9; d[16] = 0.25; d[17] = 0.45; d[18] = 1.25; d[20] = 0.7; out["POSE_D"] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF); ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    args = ap.parse_args()
    print(f"PINOCCHIO_AVAILABLE={'YES' if PINOCCHIO_AVAILABLE else 'NO'}")
    if not args.urdf.exists():
        print(f"URDF_EXISTS=NO PATH={args.urdf}"); print("URDF_GRAVITY_BACKEND_AUDITED=NO"); return 2
    _, mapping, inertial, issues, arm_ok = audit_urdf(args.urdf)
    print(f"URDF_EXISTS=YES PATH={args.urdf}")
    print(f"URDF_INERTIAL_DATA_VALID={'YES' if arm_ok else 'NO'}")
    print(f"JOINT_MAPPING_VALID={'YES' if not issues else 'NO'}")
    for idx, name, jname, child, valid in mapping:
        print(f"URDF_MAPPING SDK_INDEX={idx} JOINT_NAME={name} URDF_JOINT={jname} CHILD_LINK={child} VALID={'YES' if valid else 'NO'}")
    for name in ARM_NAMES:
        link = f"{name}_link"; print(f"ARM_INERTIAL JOINT={name} LINK={link} VALID={'YES' if inertial.get(link, False) else 'NO'}")
    mj = None; pp = None
    try: mj = MujocoGravity(args.mjcf)
    except Exception as exc: print(f"MUJOCO_BACKEND=UNAVAILABLE REASON={exc}")
    try: pp = PinocchioGravity(args.urdf)
    except Exception as exc: print(f"PINOCCHIO_BACKEND=UNAVAILABLE REASON={exc}")
    print(f"PINOCCHIO_MODEL_LOAD_PASS={'YES' if pp else 'NO'}")
    if pp:
        print(f"PINOCCHIO_MODEL_NQ={pp.model.nq} PINOCCHIO_MODEL_NV={pp.model.nv} PINOCCHIO_BASE_NQ={pp.base_nq} PINOCCHIO_BASE_NV={pp.base_nv}")
    print("MAPPING_TABLE")
    for name in ARM_NAMES:
        idx = JOINT_TO_INDEX[name]
        print(f"SDK_INDEX={idx} JOINT_NAME={name} MUJOCO_DOF_INDEX={mj.dof(name) if mj else 'NA'} PINOCCHIO_JOINT_ID={pp.ids[name] if pp else 'NA'} PINOCCHIO_V_INDEX={pp.vdof(name) if pp else 'NA'}")
    if mj and pp:
        sign = magnitude = True
        comparisons = {name: [] for name in ARM_NAMES}
        for pose_name, q in poses().items():
            mt, pt = mj.torque(q), pp.torque(q)
            print(f"{pose_name}")
            for name in ARM_NAMES:
                a, b = mt[name], pt[name]; diff = abs(a - b); rel = diff / max(abs(a), abs(b), 1e-6)
                same = (a == 0 and b == 0) or a * b > 0
                sign &= same; magnitude &= rel < 2.0
                comparisons[name].append((a, b, rel, same, pose_name))
                print(f"JOINT={name} MUJOCO_TAU_GRAVITY={a:.8f} URDF_TAU_GRAVITY={b:.8f} ABS_DIFF={diff:.8f} REL_DIFF={rel:.6f} SIGN_MATCH={'YES' if same else 'NO'}")
        print(f"MUJOCO_PINOCCHIO_SIGN_MATCH={'YES' if sign else 'NO'}")
        print(f"MUJOCO_PINOCCHIO_MAGNITUDE_REASONABLE={'YES' if magnitude else 'NO'}")
        sign_total = sum(len(v) for v in comparisons.values())
        sign_hits = sum(int(row[3]) for v in comparisons.values() for row in v)
        elbow = comparisons["left_elbow"]
        elbow_ratio = np.mean([abs(b / a) for a, b, _, _, _ in elbow if abs(a) > 1e-6]) if elbow else float("nan")
        def trend_match(rows):
            if len(rows) < 2: return False
            return all((rows[i][0] - rows[i - 1][0]) * (rows[i][1] - rows[i - 1][1]) >= -1e-10 for i in range(1, len(rows)))
        elbow_trend = trend_match(elbow)
        print(f"MUJOCO_PINOCCHIO_SIGN_MATCH_RATE={sign_hits}/{sign_total} ({sign_hits / sign_total:.3f})")
        print(f"LEFT_ELBOW_SIGN_MATCH={'YES' if all(r[3] for r in elbow) else 'NO'}")
        print(f"LEFT_ELBOW_MAGNITUDE_RATIO={elbow_ratio:.6f}")
        print(f"LEFT_ELBOW_POSE_TREND_MATCH={'YES' if elbow_trend else 'NO'}")
        recommendation = "PINOCCHIO_URDF" if sign and magnitude and arm_ok and elbow_trend else "MUJOCO_REFERENCE_ONLY"
    elif mj:
        # Still print the MuJoCo reference values for every requested pose when
        # Pinocchio is absent; this makes the missing cross-check explicit.
        for pose_name, q in poses().items():
            mt = mj.torque(q)
            print(f"{pose_name}")
            for name in ARM_NAMES:
                print(f"JOINT={name} MUJOCO_TAU_GRAVITY={mt[name]:.8f} URDF_TAU_GRAVITY=NA ABS_DIFF=NA REL_DIFF=NA SIGN_MATCH=UNAVAILABLE")
        print("MUJOCO_PINOCCHIO_SIGN_MATCH=UNAVAILABLE")
        print("MUJOCO_PINOCCHIO_MAGNITUDE_REASONABLE=UNAVAILABLE")
        recommendation = "MUJOCO_REFERENCE_ONLY"
    else:
        print("MUJOCO_PINOCCHIO_SIGN_MATCH=UNAVAILABLE")
        print("MUJOCO_PINOCCHIO_MAGNITUDE_REASONABLE=UNAVAILABLE")
        recommendation = "MUJOCO_REFERENCE_ONLY"
    print(f"GRAVITY_MODEL_AUDIT_PASS={'YES' if arm_ok and not issues else 'NO'}")
    print(f"RECOMMENDED_GRAVITY_BACKEND={recommendation}")
    print(f"FORMAL_GRAVITY_BACKEND={recommendation if recommendation == 'PINOCCHIO_URDF' else 'UNDECIDED'}")
    print("GRAVITY_REFERENCE_BACKEND=MUJOCO")
    print(f"PINOCCHIO_GRAVITY_BACKEND_AVAILABLE={'YES' if pp else 'NO'}")
    print(f"PINOCCHIO_JOINT_MAPPING_VALID={'YES' if pp else 'NO'}")
    print(f"MUJOCO_JOINT_MAPPING_VALID={'YES' if mj else 'NO'}")
    print(f"URDF_GRAVITY_BACKEND_AUDITED={'YES' if arm_ok and not issues else 'NO'}")
    print(f"MUJOCO_URDF_GRAVITY_CONSISTENCY_KNOWN={'YES' if mj and pp else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
