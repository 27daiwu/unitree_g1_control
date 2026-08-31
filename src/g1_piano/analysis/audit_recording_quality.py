"""Observation-only quality audit for a raw G1 NPZ recording."""
import argparse
import numpy as np
from g1_piano.common.joint_map import G1_29DOF_JOINT_NAMES

def main():
    p = argparse.ArgumentParser(); p.add_argument("input"); a = p.parse_args()
    with np.load(a.input, allow_pickle=False) as z:
        t, q, dq, tau, temp, tick = (z[k] for k in ("timestamp_monotonic", "q", "dq", "tau_est", "temperature", "tick"))
        dt = np.diff(t); td = np.diff(tick.astype(np.int64))
        print(f"samples={len(t)} duration={t[-1]-t[0]:.6f}s" if len(t) else "samples=0")
        for threshold in (0.0002, 0.0005, 0.002, 0.005, 0.010):
            print(f"dt {'<' if threshold < .001 else '>'} {threshold*1000:.1f}ms ratio={np.mean(dt < threshold) if threshold < .001 else np.mean(dt > threshold):.6f}")
        print(f"timestamp duplicate={np.count_nonzero(dt == 0)} non_monotonic={np.count_nonzero(dt < 0)}")
        values, counts = np.unique(td, return_counts=True)
        histogram = {int(v): int(c) for v, c in zip(values, counts)}
        print(f"tick duplicate={np.count_nonzero(td == 0)} non_monotonic={np.count_nonzero(td < 0)} unique_diffs={values.tolist()} histogram={histogram}")
        if len(dt):
            print(f"callback burst dt<0.5ms ratio={np.mean(dt < 0.0005):.6f} dt>2ms ratio={np.mean(dt > 0.002):.6f}")
        print(f"mode_machine unique={np.unique(z['mode_machine']).tolist()} mode_pr unique={np.unique(z['mode_pr']).tolist()}")
        print(f"temperature min={temp.min()} max={temp.max()} dtype={temp.dtype}")
        for i, name in enumerate(G1_29DOF_JOINT_NAMES):
            qd = np.diff(q[:, i]); dqd = np.diff(dq[:, i])
            print(f"{i:02d} {name} q=[{q[:,i].min():.5f},{q[:,i].max():.5f}] std={q[:,i].std():.5f} "
                  f"dq=[{dq[:,i].min():.5f},{dq[:,i].max():.5f}] std={dq[:,i].std():.5f} "
                  f"tau=[{tau[:,i].min():.5f},{tau[:,i].max():.5f}] std={tau[:,i].std():.5f} "
                  f"max_abs_dq={np.max(np.abs(qd)) if len(qd) else 0:.5f} max_abs_ddq={np.max(np.abs(dqd)) if len(dqd) else 0:.5f}")
        print(f"identical q frame ratio={np.mean(np.all(np.diff(q, axis=0) == 0, axis=1)) if len(q)>1 else 0:.6f}")

if __name__ == "__main__": main()
