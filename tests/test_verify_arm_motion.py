import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from g1_piano.processing.arm_extractor import LEFT_INDICES, RIGHT_INDICES
class MappingTests(unittest.TestCase):
    def test_index_mapping(self):
        self.assertEqual(LEFT_INDICES.tolist(), list(range(15,22))); self.assertEqual(RIGHT_INDICES.tolist(), list(range(22,29)))
