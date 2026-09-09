import json
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from g1_piano.common.arm_spec import get_arm_spec


class ArmSpecTests(unittest.TestCase):
    def test_mappings(self):
        left, right, both = (get_arm_spec(x) for x in ("left", "right", "both"))
        self.assertEqual(left.motor_indices, (15, 16, 17, 18, 19, 20, 21))
        self.assertEqual(right.motor_indices, (22, 23, 24, 25, 26, 27, 28))
        self.assertEqual(len(both.motor_indices), 14)
        self.assertEqual(len(set(both.motor_indices)), 14)
        self.assertEqual(len(set(both.joint_names)), 14)
        self.assertTrue(set(left.motor_indices).isdisjoint(right.motor_indices))

    def test_config_is_symmetric(self):
        left, right = get_arm_spec("left"), get_arm_spec("right")
        self.assertEqual(tuple(left.kp), tuple(right.kp))
        self.assertEqual(tuple(left.kd), tuple(right.kd))
        self.assertEqual(tuple(left.gravity_enabled), tuple(right.gravity_enabled))


if __name__ == "__main__":
    unittest.main()
