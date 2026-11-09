"""Shared fixtures for the perception_regrasp tests: synthetic renders (the hub's
pose6d ray-caster) and a kinematic fake station. Everything here is synthetic;
the fake proves contract logic, not physics."""

from __future__ import annotations

import math
import time

import numpy as np

from roboskillhub.runtime.contracts import Limits, compose, invert
from roboskillhub.tasks.perception_regrasp import _hub
from roboskillhub.tasks.perception_regrasp.frames import make_T, rot_x, rot_y, rot_z
from roboskillhub.tasks.perception_regrasp.objects import domino, to_contract_frame
from roboskillhub.tasks.perception_regrasp.providers import MaskedObject, RgbdFrame, RgbdProvider

SPEC = domino()
L, W, TH = SPEC.size_m


def synth():
    return _hub.load("pose6d_synth")


def resting(state: str, yaw: float = 0.4, xy=(0.30, 0.02), table_z: float = 0.0) -> np.ndarray:
    R = {"lying": np.eye(3), "standing": rot_y(math.pi / 2), "on_side": rot_x(math.pi / 2)}[state]
    h = {"lying": TH, "standing": L, "on_side": W}[state] / 2
    return make_T(rot_z(yaw) @ R, [xy[0], xy[1], table_z + h])


def render_frame(T_obj, camera: str = "wrist", spec=SPEC, t: float = 10.0, seed: int = 1):
    s = synth()
    c = T_obj[:3, 3]
    if camera == "overhead":
        Tc, K, noise = s.look_at([c[0] - 0.2, c[1] - 0.02, 0.9], c), s.K_ORBBEC, s.ORBBEC
    else:
        Tc, K, noise = s.look_at([c[0], c[1] - 0.12, 0.17], c), s.K_D405, s.D405
    depth, mask = s.render(
        {"box": list(spec.size_m)}, T_obj, Tc, K=K, noise=noise, table_z=0.0, seed=seed
    )
    return RgbdFrame(depth, K, Tc, t, "world", camera), mask


def rgbd_provider(frame, mask, spec=SPEC, oid="d1", **kw) -> RgbdProvider:
    return RgbdProvider(
        lambda: frame, lambda f: [MaskedObject(oid, mask)], {oid: spec}, table_z=0.0, **kw
    )


def pose_dict(T, frame="world"):
    from roboskillhub.tasks.perception_regrasp.frames import T_to_pose

    return T_to_pose(T, frame)


class FakeKinematicStation:
    """Arrives exactly; a closing jaw near the part attaches it rigidly; opening
    wider than the part releases it where it is (no gravity, no contact). It
    reports the part's full 6D pose from a ``fake_tracker`` (not privileged), so
    the skill's logic can be exercised end to end without claiming physics."""

    def __init__(
        self, T_obj: np.ndarray, oid: str = "d1", capabilities=None, slip_after_grasp=False
    ):
        self.capabilities = list(
            capabilities
            or [
                "pose_feedback",
                "cartesian_motion",
                "gripper",
                "stop",
                "object_feedback",
                "object_pose_feedback",
            ]
        )
        self.tcp = pose_dict(make_T(rot_y(math.pi), [0.25, 0.0, 0.25]))
        self.width = 0.08
        self.oid = oid
        self.obj = pose_dict(to_contract_frame(T_obj, "domino"))  # a v1 backend reports the contract frame
        self.held = False
        self.T_TCP_held = None
        self.slip_after_grasp = slip_after_grasp
        self.calls: list[str] = []

    def describe(self):
        return {"frame_id": "world", "capabilities": self.capabilities, "backend": "fake"}

    def get_state(self, *, timeout_s):
        now = time.time() + 1e-4
        if self.held:
            self.obj = compose(self.tcp, self.T_TCP_held)
            if self.slip_after_grasp:
                p = list(self.obj["position_m"])
                p[2] -= 0.03
                self.obj = {**self.obj, "position_m": p}
        objects = {
            self.oid: {
                "position_m": self.obj["position_m"],
                "quaternion_xyzw": self.obj["quaternion_xyzw"],
                "frame_id": "world",
                "timestamp_s": now,
                "source": "fake_tracker",
                "class": "domino",
                "geometry": {"type": "box", "size_m": list(SPEC.size_m)},
            }
        }
        gripper = {
            "width_m": self.width,
            "source": "fake_encoder",
            "timestamp_s": now,
            "holding": self.held,
        }
        return {"pose": self.tcp, "timestamp_s": now, "objects": objects, "gripper": gripper}

    def move_pose(self, target, *, speed_m_s, timeout_s, linear):
        self.calls.append("move")
        self.tcp = target
        return {"ok": True}

    def set_gripper(self, width_m, *, timeout_s):
        self.calls.append(f"gripper:{width_m:.4f}")
        opening = width_m > self.width
        self.width = width_m
        if self.held and opening:
            self.held = False
        elif not self.held and not opening:
            if math.dist(self.obj["position_m"], self.tcp["position_m"]) < 0.03:
                self.held = True
                self.T_TCP_held = compose(
                    invert(self.tcp, frame_id="tcp"), self.obj, frame_id="tcp"
                )
        return {"ok": True}

    def stop(self):
        self.calls.append("stop")


def fake_limits():
    return Limits.from_dict(
        {
            "frame_id": "world",
            "workspace_min_m": [-1, -1, 0.0],
            "workspace_max_m": [1, 1, 1],
            "max_speed_m_s": 0.2,
            "max_contact_distance_m": 0.3,
            "gripper_min_width_m": 0.0,
            "gripper_max_width_m": 0.094,
        }
    )
