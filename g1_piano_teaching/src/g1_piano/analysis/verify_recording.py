"""Independent offline validation for a G1 raw-state NPZ recording."""

import argparse
import json
import sys

import numpy as np

from g1_piano.common.joint_map import G1_29DOF_JOINT_NAMES


REQUIRED = {
    "timestamp_monotonic": (None,), "timestamp_wall": (None,),
    "q": (29,), "dq": (29,), "tau_est": (29,), "temperature": (29, 2),
    "mode_machine": (None,), "imu_quaternion": (4,),
    "imu_gyroscope": (3,), "imu_accelerometer": (3,), "imu_rpy": (3,),
    "imu_temperature": (None,), "tick": (None,), "mode_pr": (None,),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    args = parser.parse_args()
    try:
        with np.load(args.input, allow_pickle=False) as z:
            keys = set(z.files)
            missing = set(REQUIRED) - keys
            if missing:
                raise ValueError(f"missing keys: {sorted(missing)}")
            t = z["timestamp_monotonic"]
            T = len(t)
            failures = []
            for key, tail in REQUIRED.items():
                expected = (T,) if tail == (None,) else (T,) + tail
                if z[key].shape != expected:
                    failures.append(f"{key}.shape={z[key].shape}, expected={expected}")
            if T <= 0:
                failures.append("T must be > 0")
            if len(G1_29DOF_JOINT_NAMES) != 29:
                failures.append("joint_names length is not 29")
            if T > 1 and not np.all(np.diff(t) > 0):
                failures.append("timestamp_monotonic is not strictly increasing")
            for key in ("q", "dq", "tau_est"):
                if not np.all(np.isfinite(z[key])):
                    failures.append(f"{key} contains NaN/Inf")
            metadata = json.loads(str(z["metadata"])) if "metadata" in keys else {}
            if metadata.get("joint_names") != list(G1_29DOF_JOINT_NAMES):
                failures.append("metadata joint_names do not match local 29DoF map")
            if metadata.get("dof") != 29:
                failures.append("metadata dof must be 29")
            if metadata.get("temperature_semantics") != "UNKNOWN":
                failures.append("temperature_semantics must be UNKNOWN")
            print(f"file: {args.input}")
            print(f"keys: {sorted(keys)}")
            print(f"T: {T}")
            for key in REQUIRED:
                print(f"{key} shape: {z[key].shape}")
            dt = np.diff(t)
            if len(dt):
                p = np.percentile(dt, [0, 50, 95, 99, 100])
                duration = t[-1] - t[0]
                print(f"dt min/mean/median/p95/p99/max: {dt.min():.9f} / {dt.mean():.9f} / {p[1]:.9f} / {p[2]:.9f} / {p[3]:.9f} / {dt.max():.9f} s")
                print(f"effective frequency: {(T - 1) / duration:.3f} Hz")
            print(f"timestamp monotonic violations: {int(np.count_nonzero(dt <= 0))}")
            print(f"NaN count: {sum(int(np.isnan(z[k]).sum()) for k in z.files if np.issubdtype(z[k].dtype, np.floating))}")
            print(f"Inf count: {sum(int(np.isinf(z[k]).sum()) for k in z.files if np.issubdtype(z[k].dtype, np.floating))}")
            print(f"temperature shape check: {z['temperature'].shape == (T, 29, 2)}")
            print(f"validation: {'PASS' if not failures else 'FAIL'}")
            if failures:
                for failure in failures:
                    print(f"FAIL: {failure}")
                return 1
            return 0
    except Exception as exc:
        print(f"VALIDATION ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
