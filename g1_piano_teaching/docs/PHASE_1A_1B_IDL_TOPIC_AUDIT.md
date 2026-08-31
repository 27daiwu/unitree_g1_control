# Phase 1A/1B local SDK audit

Audit target: installed `unitree_sdk2py` (`/home/hebe/.local/lib/python3.10/site-packages/unitree_sdk2py`) and matching source tree `/home/hebe/unitree_workspace/unitree_sdk2_python`.

## Confirmed IDL

- `MotorState_`: `mode:uint8`, `q:float32`, `dq:float32`, `ddq:float32`, `tau_est:float32`, `temperature:int16[2]`, `vol:float32`, `sensor:uint32[2]`, `motorstate:uint32`, `reserve:uint32[4]`.
- Motor temperature field: `temperature`, type `int16[2]`. The local generated IDL contains no comments or conversion constants; slot meanings and units are **UNKNOWN** (raw values are displayed).
- `BmsState_`: `version_high:uint8`, `version_low:uint8`, `fn:uint8`, `cell_vol:uint16[40]`, `bmsvoltage:uint32[3]`, `current:int32`, `soc:uint8`, `soh:uint8`, `temperature:int16[12]`, `cycle:uint16`, `manufacturer_date:uint16`, `bmsstate:uint32[5]`, `reserve:uint32[3]`.
- `MainBoardState_`: `fan_state:uint16[6]`, `temperature:int16[6]`, `value:float32[6]`, `state:uint32[6]`.
- `LowState_`: `version:uint32[2]`, `mode_pr:uint8`, `mode_machine:uint8`, `tick:uint32`, `imu_state:IMUState_`, `motor_state:MotorState_[35]`, `wireless_remote:uint8[40]`, `reserve:uint32[4]`, `crc:uint32`.
- `IMUState_`: quaternion `float32[4]`, gyroscope `float32[3]`, accelerometer `float32[3]`, rpy `float32[3]`, temperature `int16`.

## Topics

- Confirmed: `rt/lowstate` -> `unitree_hg.msg.dds_.LowState_`; present in local Python and C++ G1 examples.
- HG `BmsState_` topic: **UNKNOWN**. No local SDK subscriber/example/topic constant confirms one, so monitor does not subscribe.
- HG `MainBoardState_` topic: **UNKNOWN**. No local SDK subscriber/example/topic constant confirms one, so monitor does not subscribe.

## 29DoF mapping

Confirmed from local `example/g1/low_level/g1_low_level_example.py` `G1JointIndex` and matching C++ G1 low-level example. Indices 0..28 are: left leg 0..5, right leg 6..11, waist yaw/roll/pitch 12..14, left arm 15..21, right arm 22..28. The monitor exposes this as `g1_joint_map.py`.

## Required status

`IDL_AUDIT_PASS`  
`MOTOR_TEMP_FIELD_CONFIRMED`  
`BMS_TOPIC_CONFIRMED: UNKNOWN`  
`MAINBOARD_TOPIC_CONFIRMED: UNKNOWN`  
`29DOF_JOINT_MAP_CONFIRMED`  
`SUBSCRIBER_ONLY_CONFIRMED`
