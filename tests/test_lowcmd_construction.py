import pytest


def test_lowcmd_construction():
    module = pytest.importorskip("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    from scripts.test_motor_control import create_lowcmd
    cmd = create_lowcmd()
    assert len(cmd.motor_cmd) == 35
    assert len(cmd.reserve) == 4
    print("LOWCMD_CONSTRUCTION_PASS=YES")
    print(f"MOTOR_CMD_COUNT={len(cmd.motor_cmd)}")
