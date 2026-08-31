import tempfile
import unittest
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from g1_piano.recorder.state_recorder import RawSampleQueue, assemble, metadata


class FakeMotor:
    q = 1.0; dq = 2.0; tau_est = 3.0; temperature = (40, 35)


class FakeImu:
    quaternion = (1, 0, 0, 0); gyroscope = (1, 2, 3); accelerometer = (4, 5, 6)
    rpy = (7, 8, 9); temperature = 30


class FakeState:
    motor_state = [FakeMotor() for _ in range(35)]
    mode_machine = 5; tick = 10; mode_pr = 0; imu_state = FakeImu()


class RecorderTests(unittest.TestCase):
    def test_extraction_shapes_and_metadata(self):
        q = RawSampleQueue(); q.callback(FakeState()); data = assemble([q.samples.get_nowait()])
        self.assertEqual(data["q"].shape, (1, 29)); self.assertEqual(data["temperature"].shape, (1, 29, 2))
        self.assertEqual(metadata("eth0")["temperature_semantics"], "UNKNOWN")

    def test_overflow_count(self):
        q = RawSampleQueue(maxsize=1); q.callback(FakeState()); q.callback(FakeState()); self.assertEqual(q.dropped_count(), 1)

    def test_npz_roundtrip(self):
        q = RawSampleQueue(); q.callback(FakeState()); data = assemble([q.samples.get_nowait()])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.npz"; np.savez(path, **data)
            with np.load(path) as z: self.assertEqual(z["q"].shape, (1, 29))


if __name__ == "__main__": unittest.main()
