import argparse, os, numpy as np
def main():
 p=argparse.ArgumentParser(); p.add_argument("input"); p.add_argument("--output-dir",default="."); a=p.parse_args()
 try: import matplotlib.pyplot as plt
 except Exception as e: print(f"matplotlib unavailable: {e}"); return 2
 os.makedirs(a.output_dir,exist_ok=True)
 with np.load(a.input,allow_pickle=False) as z:
  x=z["timestamp_monotonic"]-z["timestamp_monotonic"][0]
  for side in ("left","right"):
   for signal in ("q","dq"):
    plt.figure(); plt.plot(x,z[f"{side}_{signal}"]); plt.xlabel("time from start (s)"); plt.ylabel(signal); plt.legend([f"{side}_{i}" for i in range(7)]); plt.tight_layout(); plt.savefig(os.path.join(a.output_dir,f"{side}_arm_{signal}.png")); plt.close()
 return 0
if __name__ == "__main__": raise SystemExit(main())
