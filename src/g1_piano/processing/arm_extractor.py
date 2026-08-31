import argparse, json, os
import numpy as np
from g1_piano.common.joint_map import G1_29DOF_JOINT_NAMES

LEFT_INDICES = np.arange(15, 22, dtype=np.int64)
RIGHT_INDICES = np.arange(22, 29, dtype=np.int64)

def extract_arm_motion(source, output=None):
    with np.load(source, allow_pickle=False) as z:
        required = {"timestamp_monotonic", "q", "dq", "tau_est", "temperature", "metadata"}
        missing = required - set(z.files)
        if missing: raise ValueError(f"missing source keys: {sorted(missing)}")
        if z["q"].ndim != 2 or z["q"].shape[1] != 29: raise ValueError("source q must have shape [T,29]")
        T = z["q"].shape[0]
        if z["temperature"].shape != (T, 29, 2): raise ValueError("source temperature must have shape [T,29,2]")
        md = json.loads(str(z["metadata"]))
        data = {"timestamp_monotonic": z["timestamp_monotonic"].copy(), "left_q": z["q"][:, LEFT_INDICES].copy(), "left_dq": z["dq"][:, LEFT_INDICES].copy(), "left_tau_est": z["tau_est"][:, LEFT_INDICES].copy(), "right_q": z["q"][:, RIGHT_INDICES].copy(), "right_dq": z["dq"][:, RIGHT_INDICES].copy(), "right_tau_est": z["tau_est"][:, RIGHT_INDICES].copy(), "left_temperature": z["temperature"][:, LEFT_INDICES].copy(), "right_temperature": z["temperature"][:, RIGHT_INDICES].copy(), "source_joint_indices": np.r_[LEFT_INDICES, RIGHT_INDICES], "source_joint_names": np.asarray([G1_29DOF_JOINT_NAMES[i] for i in np.r_[LEFT_INDICES, RIGHT_INDICES]])}
        data["metadata"] = np.asarray(json.dumps({"format_version":"2B-arm-1", "source_file":os.path.abspath(source), "source_format_version":md.get("format_version","UNKNOWN"), "robot_model":"Unitree G1", "arm_dof":14, "left_indices":LEFT_INDICES.tolist(), "right_indices":RIGHT_INDICES.tolist(), "processing":"index extraction only"}, sort_keys=True))
    if output: np.savez_compressed(output, **data)
    return data

def main():
    p=argparse.ArgumentParser(); p.add_argument("source"); p.add_argument("--output",required=True); a=p.parse_args(); os.makedirs(os.path.dirname(os.path.abspath(a.output)),exist_ok=True); d=extract_arm_motion(a.source,a.output); print(f"source T={len(d['timestamp_monotonic'])}\nprocessed T={len(d['timestamp_monotonic'])}\nleft_q shape={d['left_q'].shape}\nright_q shape={d['right_q'].shape}")
if __name__ == "__main__": main()
