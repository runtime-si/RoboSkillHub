"""lego_vision: the jaw state from the command log, the wrist camera pose, settings."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from roboskillhub.tasks.assembly_eval import lego_vision as lv


def test_jaw_state_follows_the_last_gripper_command():
    v = lv.LegoVision()
    v.station = SimpleNamespace(command_log=[])
    assert v._jaw() == (False, 0)
    v.station.command_log = [{"grip": 0.042}, {"move": [0, 0, 0, 0]}, {"grip": 0.012}]
    assert v._jaw() == (True, 1)
    v.station.command_log.append({"grip": 0.046})
    assert v._jaw() == (False, 1)


def test_wrist_camera_looks_straight_down_with_the_hand_yaw():
    T = lv._look_down(0.2, -0.1, 0.13, math.radians(30))
    assert T[:3, 2] == pytest.approx([0.0, 0.0, -1.0])  # optical axis down
    assert T[:3, 0] == pytest.approx([math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0])
    assert T[:3, :3] @ T[:3, :3].T == pytest.approx(np.eye(3))
    assert T[:3, 3] == pytest.approx([0.2, -0.1, 0.13])


def test_settings_are_checked():
    with pytest.raises(ValueError, match="cameras"):
        lv.LegoVision(cameras=("left",))
    assert lv.MATCH_GATE_M < 0.03 and lv.DUPLICATE_M <= 0.01 and lv.MIN_IMAGE_INTERVAL_S >= 0.5
