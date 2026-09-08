# Phase 2A.1 workspace audit and migration record

## Classification before migration

- MOVE: `g1_joint_map.py`, `dds_watchdog.py`, `g1_state_monitor.py`, `g1_state_recorder.py`, `verify_g1_recording.py`, `docs/archive/PHASE_1A_1B_IDL_TOPIC_AUDIT.md`, `data/state/raw/g1_test_001.npz`.
- DELETED: historical read-only LowState smoke test (superseded by `scripts/run_state_monitor.py`).
- KEEP: `.git`, `.gitignore`.
- UNKNOWN: none.
- Cache: `__pycache__` (ignored, not treated as project source).
- Deleted files: the superseded LowState smoke test and the historical Phase 2C0 MuJoCo audit; conclusions are retained in `docs/archive/phase_2c0_mujoco_model_audit.txt`.

## OLD -> NEW

- `/home/hebe/zjy_ws/g1_joint_map.py` -> `/home/hebe/zjy_ws/g1_piano_teaching/src/g1_piano/common/joint_map.py`
- `/home/hebe/zjy_ws/dds_watchdog.py` -> `/home/hebe/zjy_ws/g1_piano_teaching/src/g1_piano/dds/watchdog.py`
- `/home/hebe/zjy_ws/g1_state_monitor.py` -> `/home/hebe/zjy_ws/g1_piano_teaching/src/g1_piano/monitor/state_monitor.py` (root compatibility wrapper retained)
- `/home/hebe/zjy_ws/g1_state_recorder.py` -> `/home/hebe/zjy_ws/g1_piano_teaching/src/g1_piano/recorder/state_recorder.py` (root compatibility wrapper retained)
- `/home/hebe/zjy_ws/verify_g1_recording.py` -> `/home/hebe/zjy_ws/g1_piano_teaching/src/g1_piano/analysis/verify_recording.py` (root compatibility wrapper retained)
- `/home/hebe/zjy_ws/docs/archive/PHASE_1A_1B_IDL_TOPIC_AUDIT.md` -> `/home/hebe/zjy_ws/g1_piano_teaching/docs/archive/PHASE_1A_1B_IDL_TOPIC_AUDIT.md`
- `/home/hebe/zjy_ws/data/state/raw/g1_test_001.npz` -> `/home/hebe/zjy_ws/g1_piano_teaching/data/state/raw/g1_test_001.npz`
Formal implementation exists only under `src/g1_piano`; root scripts are compatibility entry points, not duplicated implementations.
