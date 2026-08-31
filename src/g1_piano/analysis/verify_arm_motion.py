import argparse, json
import numpy as np
from g1_piano.processing.arm_extractor import LEFT_INDICES, RIGHT_INDICES
from g1_piano.common.joint_map import G1_29DOF_JOINT_NAMES

def verify(source, arm):
    failures=[]
    with np.load(source,allow_pickle=False) as raw, np.load(arm,allow_pickle=False) as z:
        T=raw["q"].shape[0]
        for side,idx in (("left",LEFT_INDICES),("right",RIGHT_INDICES)):
            for k in ("q","dq","tau_est"):
                key=f"{side}_{k}"; expected=raw[k][:,idx]
                if z[key].shape!=(T,7): failures.append(f"{key} shape")
                elif not np.array_equal(z[key],expected): failures.append(f"{key} not lossless")
            if not np.array_equal(z[f"{side}_temperature"],raw["temperature"][:,idx]): failures.append(f"{side}_temperature not lossless")
        if not np.all(np.diff(z["timestamp_monotonic"])>0): failures.append("timestamp regression")
        for k in z.files:
            if k not in ("metadata","source_joint_names") and np.issubdtype(z[k].dtype,np.number) and not np.all(np.isfinite(z[k])): failures.append(f"{k} NaN/Inf")
        names=list(z["source_joint_names"]); expected=[G1_29DOF_JOINT_NAMES[i] for i in np.r_[LEFT_INDICES,RIGHT_INDICES]]
        if names!=expected: failures.append("joint mapping")
        left_err=float(np.max(np.abs(z["left_q"]-raw["q"][:,LEFT_INDICES])))
        right_err=float(np.max(np.abs(z["right_q"]-raw["q"][:,RIGHT_INDICES])))
        timestamp_regression=int(np.count_nonzero(np.diff(z["timestamp_monotonic"])<=0))
    print(f"source T={T}\nprocessed T={T}\nleft extraction max error={left_err:.9g}\nright extraction max error={right_err:.9g}\ntimestamp regression={timestamp_regression}\nvalidation={'PASS' if not failures else 'FAIL'}")
    if failures:
        print("failures:", failures)
    return not failures
def main():
    p=argparse.ArgumentParser(); p.add_argument("source"); p.add_argument("arm"); a=p.parse_args(); return 0 if verify(a.source,a.arm) else 1
if __name__ == "__main__": raise SystemExit(main())
