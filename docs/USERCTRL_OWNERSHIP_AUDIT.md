# UserCtrl ownership audit (G1)

## SDK result

The local Unitree SDK2 Python source (`unitree_sdk2py/g1/loco`) exposes:

- `LocoClient.GetFsmId()` -> `(code, fsm_id)`
- `LocoClient.SwitchToUserCtrl()` -> integer result code
- `LocoClient.SwitchToInternalCtrl(InternalFsmMode.PASSIVE|LAST|WALKRUN)`
- `InternalFsmMode.PASSIVE == 1`
- `LocoClient.Damp()` (implemented as `SetFsmId(1)`, return value is not propagated)

The matching C++ example confirms the user command topic and message type:
`rt/user_lowcmd` carrying `unitree_hg.msg.dds_.LowCmd_`. The state topic is
`rt/lowstate` carrying `LowState_`. The legacy topic is `rt/lowcmd`.

## Required handoff

```text
WAIT_LOWSTATE
  -> CHECK_FSM (GetFsmId)
  -> REQUIRE_PASSIVE (fsm_id == 1 and valid lowstate)
  -> PREPARE_INITIAL_USER_LOWCMD (publisher + measured-q frame)
  -> SWITCH_TO_USER_CTRL
  -> USER_CONTROL_ACTIVE
```

The first measured-q frame is written before `SwitchToUserCtrl()`, preventing
an ownership-gap target jump. The application never calls `Damp()` implicitly;
an operator must put the robot in PASSIVE/DAMPING first.

## Exit lifecycle

```text
USER_CONTROL_ACTIVE -> CONTROLLED_STOP -> OWNERSHIP_RELEASE
                   -> SwitchToInternalCtrl(PASSIVE)
```

`--exit-mode passive` is the default. `--exit-mode last` is explicit and
restores the SDK's `LAST` internal mode; WALKRUN is not an automatic default.
SAFE_EXIT means ownership release, not merely stopping DDS writes.

## Backend separation

`userctrl` is the default and publishes `rt/user_lowcmd`. The explicit
`legacy_lowcmd` backend publishes `rt/lowcmd`, is experimental, and is never
run concurrently with userctrl. No real-robot command was executed during
this audit.
