"""servoing-by-learned-keypoint-correspondence: the CNS loop with a fake cns tool and a fake arm, and the cns bundle."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "servoing-by-learned-keypoint-correspondence"


@pytest.fixture(scope="module")
def bundle(skills_registry):
    b = skills_registry.get(BUNDLE)
    assert b.kind == "skill"
    return b


@pytest.fixture(scope="module")
def servo(bundle):
    return bundle.canonical_scripts["servo_cns"].module


def rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


DOWN = np.diag([1.0, -1.0, -1.0])  # tool / camera pointing down


def quat(m):
    from scipy.spatial.transform import Rotation

    return dict(zip("xyzw", map(float, Rotation.from_matrix(m).as_quat()), strict=True))


class Arm:
    def __init__(self, x=0.30, y=0.10, z=0.15, yaw=10.0):
        self.p = np.array([x, y, z])
        self.yaw = yaw
        self.q = np.zeros(6)

    def R(self):
        return rz(self.yaw) @ DOWN

    def tools(self, cns):
        return {
            "robot.get_ee_pose": lambda **kw: {"position": dict(zip("xyz", map(float, self.p), strict=True)), "rotation": quat(self.R())},
            "robot.go_to_pose": self.go,
            "robot.move_to_joints": self.joints,
            "robot.get_observation": self.obs,
            **cns,
        }

    def go(self, pose, **kw):
        self.p = np.array([pose["position"][k] for k in "xyz"])
        from scipy.spatial.transform import Rotation

        r = Rotation.from_quat([pose["rotation"][k] for k in "xyzw"]).as_matrix()
        self.yaw = math.degrees(math.atan2(r[1, 0], r[0, 0]))
        return {"outcome": "reached"}

    def joints(self, joint_config, **kw):
        q = np.asarray(joint_config["positions"])
        self.yaw -= math.degrees(q[-1] - self.q[-1])
        self.q = q
        return {"outcome": "reached"}

    def obs(self):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R(), self.p
        depth = np.full((40, 60), 0.15)
        return {"cameras": [{"name": "wrist", "rgb": np.zeros((40, 60, 3), np.uint8), "depth": depth,
                             "intrinsics": [[50, 0, 30], [0, 50, 20], [0, 0, 1]], "pose": T.tolist()}],
                "arms": [{"joint_state": self.q.tolist()}]}


class FakeCNS:
    """Commands the camera velocity that closes the pose error to the goal (camera frame), like a perfect servo."""

    def __init__(self, arm, goal, matches=50, gain=1.0):
        self.arm, self.goal, self.matches, self.gain = arm, goal, matches, gain
        self.started, self.stopped = [], []

    def tools(self):
        return {"cns.start": self.start, "cns.step": self.step, "cns.stop": lambda session_id: self.stopped.append(session_id)}

    def start(self, **kw):
        self.started.append(kw)
        return {"session_id": "s1", "keypoints": 80, "ok": True}

    def step(self, session_id, image):
        R = self.arm.R()
        dp = np.array(self.goal[:2] + (self.arm.p[2],)) - self.arm.p
        dyaw = math.radians((self.goal[2] - self.arm.yaw + 180) % 360 - 180)
        err = float(np.linalg.norm(dp) + abs(dyaw) * 0.05)
        v = R.T @ dp * self.gain / 0.5
        w = R.T @ np.array([0, 0, dyaw]) * self.gain / 0.5
        m = self.matches if not callable(self.matches) else self.matches()
        return {"velocity": [*v, *w], "matches": m, "error": err, "converged": err < 0.0003}


def test_converges_to_the_goal_in_plane(servo):
    arm = Arm(x=0.306, y=0.096, yaw=12.0)
    cns = FakeCNS(arm, (0.30, 0.10, 10.0))
    goal_obs = Arm(x=0.30, y=0.10, yaw=10.0).obs()
    out = servo.run(FakeContext(arm.tools(cns.tools())), goal=goal_obs, camera="wrist")
    assert out["route"] == "converged", out
    assert np.linalg.norm(arm.p[:2] - [0.30, 0.10]) < 0.0005 and abs(arm.yaw - 10.0) < 0.3
    assert cns.started[0]["distance_m"] == pytest.approx(0.15)
    assert cns.stopped == ["s1"]


def test_steps_are_capped(servo):
    arm = Arm(x=0.33)
    xs = []
    cns = FakeCNS(arm, (0.30, 0.10, 10.0))
    t = arm.tools(cns.tools())
    go = t["robot.go_to_pose"]
    t["robot.go_to_pose"] = lambda **kw: (xs.append(kw["pose"]["position"]["x"]), go(**kw))[1]
    servo.run(FakeContext(t), goal=Arm().obs(), camera="wrist", max_frames=3)
    assert np.allclose(np.diff([0.33, *xs]), -0.004, atol=1e-9)


def test_lost_correspondence(servo):
    arm = Arm()
    cns = FakeCNS(arm, (0.31, 0.1, 10.0), matches=2)
    out = servo.run(FakeContext(arm.tools(cns.tools())), goal=Arm().obs(), camera="wrist")
    assert out["route"] == "lost_correspondence" and "3 frames" in out["reason"]
    assert cns.stopped == ["s1"]


def test_timeout_and_refused_step(servo):
    arm = Arm(x=0.33)
    cns = FakeCNS(arm, (0.30, 0.10, 10.0), gain=0.01)
    out = servo.run(FakeContext(arm.tools(cns.tools())), goal=Arm().obs(), camera="wrist", max_frames=4)
    assert out["route"] == "timeout" and out["frames"] == 4
    t = arm.tools(FakeCNS(arm, (0.30, 0.10, 10.0)).tools())
    t["robot.go_to_pose"] = lambda **kw: {"outcome": "unreachable", "reached": False}
    out = servo.run(FakeContext(t), goal=Arm().obs(), camera="wrist")
    assert out["route"] == "timeout" and "refused" in out["reason"]


def test_goal_without_frame_or_distance(servo):
    arm = Arm()
    cns = FakeCNS(arm, (0.3, 0.1, 10.0))
    out = servo.run(FakeContext(arm.tools(cns.tools())), goal={"cameras": []}, camera="wrist")
    assert out["route"] == "lost_correspondence" and not cns.started
    g = Arm().obs()
    del g["cameras"][0]["depth"]
    out = servo.run(FakeContext(arm.tools(cns.tools())), goal=g, camera="wrist")
    assert out["route"] == "lost_correspondence" and "distance" in out["reason"]
    out = servo.run(FakeContext(arm.tools(cns.tools())), goal=g, camera="wrist", distance_m=0.2)
    assert cns.started[-1]["distance_m"] == 0.2


def test_full_mode_uses_go_to_pose_rotation(servo):
    arm = Arm(x=0.302, yaw=14.0)
    cns = FakeCNS(arm, (0.30, 0.10, 10.0))
    out = servo.run(FakeContext(arm.tools(cns.tools())), goal=Arm().obs(), camera="wrist", mode="full")
    assert out["route"] == "converged" and abs(arm.yaw - 10.0) < 0.3
    assert not arm.q.any()  # no joint turns in full mode
    with pytest.raises(ValueError, match="mode"):
        servo.run(FakeContext(arm.tools(cns.tools())), goal=Arm().obs(), camera="wrist", mode="xy")


def test_cns_bundle(skills_registry, tool_registry):
    mod = skills_registry.get("cns").tools_module
    assert len(mod.WEIGHTS_SHA256) == 64 and "6b42138ea8f1" in mod.WEIGHTS_URL
    assert {"goal_image", "intrinsics", "distance_m", "detector"} == set(tool_registry.get("cns.start").schema.inputs)
    assert {"session_id", "image"} == set(tool_registry.get("cns.step").schema.inputs)
    assert "torch" not in vars(mod)
    with pytest.raises(Exception, match="no session"):
        mod.step("nope", np.zeros((4, 4, 3), np.uint8))


def test_example_declares_every_route(bundle):
    ex = json.loads((Path(bundle.bundle_dir) / "examples" / "canonical_subgraph.json").read_text())
    assert set(ex["exit"]["success_values"]) | {ex["on_error"]} == {"converged", "lost_correspondence", "timeout"}
