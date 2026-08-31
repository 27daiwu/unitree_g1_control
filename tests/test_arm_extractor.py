import json, tempfile, unittest
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from g1_piano.processing.arm_extractor import extract_arm_motion

class ArmTests(unittest.TestCase):
    def test_synthetic_lossless_index_extraction(self):
        T=120; t=np.arange(T,dtype=np.float64)/1000
        q=np.zeros((T,29),np.float32); dq=np.zeros_like(q); tau=np.zeros_like(q); temp=np.zeros((T,29,2),np.int16)
        for i in range(29): q[:,i]=np.sin(t*(i+1)); dq[:,i]=np.cos(t*(i+1)); tau[:,i]=i; temp[:,i]=[i,i+1]
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/"raw.npz"; out=Path(d)/"arm.npz"; np.savez(src,timestamp_monotonic=t,q=q,dq=dq,tau_est=tau,temperature=temp,metadata=np.asarray(json.dumps({"format_version":"raw-1"})))
            a=extract_arm_motion(src,out); self.assertEqual(a["left_q"].shape,(T,7)); self.assertTrue(np.array_equal(a["left_q"],q[:,15:22])); self.assertTrue(np.array_equal(a["right_q"],q[:,22:29]))
    def test_wrong_dof_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.npz"; np.savez(p,timestamp_monotonic=np.arange(2),q=np.zeros((2,28)),dq=np.zeros((2,28)),tau_est=np.zeros((2,28)),temperature=np.zeros((2,28,2)),metadata=np.asarray("{}"))
            with self.assertRaises(ValueError): extract_arm_motion(p)
