# G1 29DoF Motor Control Audit

## Scope

This file records physical one-joint tests. It intentionally starts empty:
no joint is considered verified from a software or simulation result.

The verified mapping is the local Unitree G1 low-level definition documented
in `PHASE_1A_1B_IDL_TOPIC_AUDIT.md`: indices 0..28 are left leg, right leg,
waist, left arm, and right arm in that order. Wrist joints should be tested
before elbow/shoulder; legs and waist remain untested until SIT_HOME exists.

## Test record

| index | joint | group | q initial | commanded delta | actual delta | direction | tracking | other joints | service takeover | safe exit | status |
|---:|---|---|---:|---:|---:|---|---|---|---|---|---|
| - | - | - | - | - | - | - | - | - | - | - | NOT TESTED |

## LowCmd semantics used by the diagnostic

The HG `LowCmd_` IDL exposes a complete `motor_cmd` array (35 slots) on
`rt/lowcmd`; G1 29DoF occupies slots 0..28. The diagnostic therefore sends a
complete frame when control is enabled. The selected slot receives the
position/Kp/Kd test command. The other 28 G1 slots are explicitly populated
with their latest measured q and zero gains/torque, and slots 29..34 are also
initialized. This is a deliberate hold-frame policy, not an assumption that
zero-initialized messages mean “do not control”. The exact mode/ownership
behavior must still be confirmed on the target firmware before physical use.

`--dry-run` never creates a publisher and is safe for mapping/state checks.
