import argparse
import numpy as np
from g1_piano.common.joint_map import G1_29DOF_JOINT_NAMES
def main():
 p=argparse.ArgumentParser(); p.add_argument("input"); p.add_argument("--stationary-dq-threshold",type=float,required=True); a=p.parse_args()
 with np.load(a.input,allow_pickle=False) as z:
  for side,base in (("left",15),("right",22)):
   q,dq,tau=z[f"{side}_q"],z[f"{side}_dq"],z[f"{side}_tau_est"]
   print(f"{side} stationary sample ratio={np.mean(np.all(np.abs(dq)<a.stationary_dq_threshold,axis=1)):.6f} active sample ratio={np.mean(np.any(np.abs(dq)>=a.stationary_dq_threshold,axis=1)):.6f}")
   for j in range(7): print(f"{G1_29DOF_JOINT_NAMES[base+j]} q min/max/mean/std={q[:,j].min():.6f}/{q[:,j].max():.6f}/{q[:,j].mean():.6f}/{q[:,j].std():.6f} dq min/max/mean/std={dq[:,j].min():.6f}/{dq[:,j].max():.6f}/{dq[:,j].mean():.6f}/{dq[:,j].std():.6f} tau min/max/std={tau[:,j].min():.6f}/{tau[:,j].max():.6f}/{tau[:,j].std():.6f} max_abs_diff_q={np.max(np.abs(np.diff(q[:,j]))):.6f} max_abs_diff_dq={np.max(np.abs(np.diff(dq[:,j]))):.6f}")
if __name__ == "__main__": main()
