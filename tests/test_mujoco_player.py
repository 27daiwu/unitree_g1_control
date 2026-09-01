import numpy as np
import pytest
import mujoco

from g1_piano.simulate.mujoco_player import (G1MujocoMapping, clip_playback_range,
    interpolate_q, load_raw_npz, nearest_sample_index, seek_seconds)


def save(tmp_path, q, t):
    p = tmp_path / "x.npz"; np.savez(p, q=q, timestamp_monotonic=t); return p

def test_schema_and_mapping(tmp_path):
    p = save(tmp_path, np.zeros((3, 29)), np.array([1., 1.1, 2.]))
    raw = load_raw_npz(p); assert raw.q.shape == (3, 29); assert raw.duration == pytest.approx(1.0)
    m = mujoco.MjModel.from_xml_path("assets/robots/unitree_g1/xmls/scene_g1.xml")
    mapping = G1MujocoMapping.from_model(m); assert len(mapping.qpos_addresses) == 29

def test_reject_bad_schema(tmp_path):
    with pytest.raises(ValueError): load_raw_npz(save(tmp_path, np.zeros((3, 28)), np.arange(3.)))
    with pytest.raises(ValueError): load_raw_npz(save(tmp_path, np.full((3, 29), np.nan), np.arange(3.)))
    with pytest.raises(ValueError): load_raw_npz(save(tmp_path, np.zeros((3, 29)), np.array([0., 1., 1.])))

def test_sampling():
    t = np.array([0., 1., 3.]); q = np.arange(6., dtype=float).reshape(3, 2)
    assert nearest_sample_index(t, 2.1) == 2
    assert np.allclose(interpolate_q(t, q, 2., "linear")[0], [3., 4.])
    assert np.allclose(interpolate_q(t, q, -1., "linear")[0], q[0])

def test_start_end_clip():
    assert clip_playback_range(20., -1., 25.) == (0., 20.)
    assert clip_playback_range(20., 5., 10.) == (5., 10.)
    with pytest.raises(ValueError): clip_playback_range(20., 10., 5.)
    assert seek_seconds(5., -10., 2., 8.) == 2.
    assert seek_seconds(5., 10., 2., 8.) == 8.
