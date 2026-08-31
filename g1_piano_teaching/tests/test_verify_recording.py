import unittest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from g1_piano.analysis.verify_recording import REQUIRED


class VerifyTests(unittest.TestCase):
    def test_required_schema(self):
        self.assertEqual(REQUIRED["q"], (29,)); self.assertEqual(REQUIRED["temperature"], (29, 2))

    def test_timestamp_violation_detection(self):
        self.assertFalse(np.all(np.diff(np.array([0.0, 0.1, 0.05])) > 0))

    def test_nan_inf_detection(self):
        x = np.array([1.0, np.nan, np.inf]); self.assertFalse(np.all(np.isfinite(x)))


if __name__ == "__main__": unittest.main()
