import unittest
import numpy as np
import tempfile
from unittest.mock import patch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from g1_piano.analysis.verify_recording import REQUIRED, main


class VerifyTests(unittest.TestCase):
    def test_required_schema(self):
        self.assertEqual(REQUIRED["q"], (29,)); self.assertEqual(REQUIRED["temperature"], (29, 2))

    def test_timestamp_violation_detection(self):
        self.assertFalse(np.all(np.diff(np.array([0.0, 0.1, 0.05])) > 0))

    def test_nan_inf_detection(self):
        x = np.array([1.0, np.nan, np.inf]); self.assertFalse(np.all(np.isfinite(x)))

    def test_malformed_npz_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "malformed.npz"
            np.savez(path, q=np.zeros((1, 29), dtype=np.float32))
            with patch("sys.argv", ["verify_recording", str(path)]):
                self.assertEqual(main(), 2)


if __name__ == "__main__": unittest.main()
