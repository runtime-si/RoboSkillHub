"""inserting-a-held-peg-top-down: align, wrist-camera refine, descend, search, push, probe the top."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError
from scipy.spatial.transform import Rotation

BUNDLE = "inserting-a-held-peg-top-down"
DOWN = Rotation.from_euler("x", 180, degrees=True)  # tool z pointing at the table
MOUTH = np.array([0.30, -0.10, 0.05])
L, D = 0.10, 0.010
GRIP_DOWN = 0.015  # tool centre under the peg's top
TIP = L - GRIP_DOWN  # tip under the tool centre
FINGER = 0.0102


@pytest.fixture(scope="module")
def mod(skills_registry):
    b = skills_registry.get(BUNDLE)
    assert b.kind == "skill"
    return b.canonical_scripts["insert_held_peg"].module


def _v(a):
    return {"x": float(a[0]), "y": float(a[1]), "z": float(a[2])}


def _q(r: Rotation):
    x, y, z, w = r.as_quat()
    return {"w": float(w), "x": float(x), "y": float(y), "z": float(z)}


def _m(pose):
    pose = pose.get("pose", pose)
    q = pose["rotation"]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    out[:3, 3] = [pose["position"][k] for k in "xyz"]
    return out


def held(sigma_mm=1.0):
    return {
        "arm": "right",
        "T_TCP_object": {
            "position": _v([0.0, 0.0, L / 2 - GRIP_DOWN]),
            "rotation": _q(Rotation.from_euler("x", 180, degrees=True)),
        },
        "tip_offset_m": TIP,
        "axis_tilt_deg": 0.0,
        "sigma_mm": sigma_mm,
    }


def hole(clear=0.0005, chamfer=0.0, depth=0.03, offset=(0.0, 0.0)):
    return {
        "T_world_mouth": {
            "position": _v(MOUTH + np.array([offset[0], offset[1], 0.0])),
            "rotation": _q(Rotation.from_euler("x", 180, degrees=True)),
        },
        "diameter_m": D + clear,
        "depth_m": depth,
        "chamfer_m": chamfer,
    }


PEG = {"length_m": L, "diameter_m": D, "axis_in_object": [0, 0, 1], "tip_end": "-z"}


class World:
    """A right arm holding a peg straight down over a hole at ``true_mouth``.

    Moves land where commanded unless the tip would pass below the mouth plane
    outside the hole's capture radius (stopped at the face), or below the hole's
    floor, or (``jam_at``) deeper than a jam. ``slip`` makes the peg slide up in
    the fingers by that much once the tip is in. The peg stays where it is left
    when the hand opens; a closed-finger push stops on the peg's top (as lag)."""

    def __init__(
        self, true_mouth=MOUTH, clear=0.0005, chamfer=0.0, jam_at=None, slip=0.0, depth=0.03
    ):
        self.H = np.eye(4)
        self.H[:3, :3] = DOWN.as_matrix()
        self.H[:3, 3] = true_mouth + np.array([0.0, 0.0, TIP + 0.06])
        self.mouth = np.asarray(true_mouth, float)
        self.capture = clear / 2 + chamfer + 1e-6
        self.jam_at, self.slip, self.depth = jam_at, slip, depth
        self.grip_shift = 0.0  # peg slid up in the fingers
        self.open = False
        self.peg_tip = None  # world tip once released
        self.cmd = self.H.copy()
        self.calls: list[str] = []
        self.lagging = False

    # where the held tip is for a hand pose
    def tip_of(self, H):
        return H[:3, 3] + H[:3, :3] @ np.array([0.0, 0.0, TIP - self.grip_shift])

    def clamp(self, T):
        if self.open:
            if self.peg_tip is not None:
                top = self.peg_tip + np.array([0.0, 0.0, L])
                lateral = np.linalg.norm((T[:3, 3] - top)[:2])
                floor = top[2] + FINGER
                if lateral < 0.006 and T[2, 3] < floor:
                    T = T.copy()
                    T[2, 3] = floor
                    self.lagging = True
            return T
        tip = self.tip_of(T)
        lateral = float(np.linalg.norm((tip - self.mouth)[:2]))
        below = self.mouth[2] - tip[2]
        stop = None
        if below > 0 and lateral > self.capture:
            stop = 0.0
        elif below > 0:
            stop = self.depth
            if self.jam_at is not None:
                stop = min(stop, self.jam_at)
        if stop is not None and below > stop:
            need = below - stop
            add = min(max(0.0, self.slip - self.grip_shift), need)
            self.grip_shift += add
            need -= add
            if need > 1e-12:
                T = T.copy()
                T[2, 3] += need
                self.lagging = True
        return T

    def go(self, pose=None, **kw):
        T = _m(pose)
        self.cmd = T
        self.lagging = False
        self.H = self.clamp(T)
        return {"outcome": "reached"}

    def ee(self, **kw):
        return {"position": _v(self.H[:3, 3]), "rotation": _q(Rotation.from_matrix(self.H[:3, :3]))}

    def open_g(self, **kw):
        self.open = True
        self.peg_tip = self.tip_of(self.H)
        return {}

    def close_g(self, **kw):
        return {}

    def ctx(self, observation=None, **over):
        tools = {
            "robot.get_ee_pose": self.ee,
            "robot.go_to_pose": self.go,
            "robot.go_to_pose_cartesian": self.go,
            "robot.move_cartesian_until_contact": self.go,
            "robot.solve_ik": {"success": True, "within_servo_tolerance": True},
            "robot.open_gripper": self.open_g,
            "robot.close_gripper": self.close_g,
            "robot.get_observation": observation if observation is not None else {"cameras": {}},
        }
        tools.update(over)
        return FakeContext(tools)


def go(mod, w, **kw):
    kw.setdefault("held", held())
    kw.setdefault("hole", hole())
    kw.setdefault("peg", PEG)
    kw.setdefault("settle_period", 0.0)
    return mod.run(w.ctx(), **kw)


def test_inserts_straight_and_probes_the_top(mod):
    w = World()
    out = go(mod, w)
    assert out["route"] == "inserted", out["detail"]
    assert out["released"] is True
    assert out["target_depth"] == pytest.approx(0.028)
    assert out["tip_depth"] == pytest.approx(0.028, abs=0.002)
    assert out["search"] == "none"
    assert w.peg_tip[2] == pytest.approx(MOUTH[2] - 0.028, abs=0.0015)


def test_json_text_inputs_are_accepted(mod):
    w = World()
    out = go(mod, w, held=json.dumps(held()), hole=json.dumps(hole()), peg=json.dumps(PEG))
    assert out["route"] == "inserted"


def test_tip_from_held_without_length_uses_tip_offset(mod):
    h = held()
    T = mod.tip_in_tcp_from_held(h, {"axis_in_object": [0, 0, 1], "tip_end": "-z"})
    assert T[:3, 3] == pytest.approx([0.0, 0.0, TIP])
    T2 = mod.tip_in_tcp_from_held(h, PEG)
    assert T2[:3, 3] == pytest.approx([0.0, 0.0, TIP])
    assert T2[:3, 2] == pytest.approx([0.0, 0.0, 1.0])


def test_blocked_at_mouth_spiral_finds_the_hole(mod):
    w = World(true_mouth=MOUTH + np.array([0.0015, 0.0, 0.0]), clear=0.0005, chamfer=0.0005)
    out = go(mod, w, hole=hole(chamfer=0.0005))
    assert out["route"] == "inserted", out["detail"]
    assert out["search"].startswith("spiral")
    assert out["probes"] >= 1


def test_missed_hole_when_spiral_and_tilt_fail(mod):
    w = World(true_mouth=MOUTH + np.array([0.012, 0.0, 0.0]))
    out = go(mod, w, hole=hole(chamfer=0.0005))
    assert out["route"] == "missed_hole"
    assert out["search"] == "spiral+tilt"
    assert out["released"] is False


def test_missed_hole_on_contact_above_the_mouth(mod):
    w = World(true_mouth=MOUTH + np.array([0.02, 0.0, 0.01]))
    out = go(mod, w)
    assert out["route"] == "missed_hole"
    assert "over the mouth plane" in out["detail"]


def test_jammed_when_push_stops_short(mod):
    w = World(jam_at=0.010)
    out = go(mod, w)
    assert out["route"] == "jammed", out["detail"]
    assert out["released"] is True
    assert out["hand_depth"] < 0.026


def test_slipped_in_grip_is_reported(mod):
    # the tip meets an obstruction at 12 mm and the peg slides up the fingers
    w2 = World(depth=0.012, slip=0.02)
    out = go(mod, w2, hole=hole(depth=0.030))
    assert out["route"] == "slipped_in_grip", out["detail"]
    assert out["slip"] > 0.002
    assert out["tip_depth"] < out["hand_depth"]


def test_unverified_without_release_or_length(mod):
    out = go(mod, World(), release=False)
    assert out["route"] == "unverified"
    peg = dict(PEG)
    peg.pop("length_m")
    out = go(mod, World(), peg=peg)
    assert out["route"] == "unverified"


def test_unverified_when_probe_meets_nothing(mod):
    w = World()
    orig = w.open_g

    def drop(**kw):
        orig()
        w.peg_tip = w.peg_tip + np.array([0.05, 0.0, 0.0])  # fell over
        return {}

    out = mod.run(
        w.ctx(**{"robot.open_gripper": drop}), held=held(), hole=hole(), peg=PEG, settle_period=0.0
    )
    assert out["route"] == "unverified"


def test_unreachable_when_ik_says_no(mod):
    w = World()
    out = mod.run(
        w.ctx(**{"robot.solve_ik": {"success": False}}),
        held=held(),
        hole=hole(),
        peg=PEG,
        settle_period=0.0,
    )
    assert out["route"] == "unreachable"


def test_unreachable_when_a_move_raises(mod):
    w = World()

    def refuse(**kw):
        raise ToolError("out of workspace")

    out = mod.run(
        w.ctx(**{"robot.go_to_pose": refuse, "robot.go_to_pose_cartesian": refuse}),
        held=held(),
        hole=hole(),
        peg=PEG,
        settle_period=0.0,
    )
    assert out["route"] == "unreachable"


def test_unreachable_for_a_side_hole(mod):
    h = hole()
    h["T_world_mouth"]["rotation"] = _q(Rotation.from_euler("y", 90, degrees=True))
    out = go(mod, World(), hole=h)
    assert out["route"] == "unreachable"
    assert "30 deg" in out["detail"]


def test_fit_circle_recovers_a_partial_rim(mod):
    a = np.linspace(0.3, 4.5, 30)
    pts = np.stack([0.001 + 0.0055 * np.cos(a), -0.002 + 0.0055 * np.sin(a)], axis=1)
    c, r = mod.fit_circle(pts)
    assert c == pytest.approx([0.001, -0.002], abs=1e-6)
    assert r == pytest.approx(0.0055, abs=1e-6)


def _render(mouth, tip, hole_r, peg_r, cam_pos, K, size=(240, 320)):
    """A depth image of a flat face with a round hole at ``mouth`` and a vertical
    shaft ending at ``tip``, seen from ``cam_pos`` looking straight down."""
    h, w = size
    v, u = np.mgrid[0:h, 0:w]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    # camera looking down: camera z = world -z, camera x = world x, camera y = world -y
    world_dir = np.stack([rays[..., 0], -rays[..., 1], -rays[..., 2]], -1)
    depth = np.full((h, w), np.nan)
    # face plane z = mouth z
    t = (cam_pos[2] - mouth[2]) / -world_dir[..., 2] * -1
    t = (mouth[2] - cam_pos[2]) / world_dir[..., 2]
    hit = cam_pos + world_dir * t[..., None]
    in_hole = np.linalg.norm(hit[..., :2] - mouth[:2], axis=-1) < hole_r
    depth[~in_hole] = t[~in_hole]
    depth[in_hole] = t[in_hole] + 0.03
    # shaft: a vertical cylinder from tip up 60 mm; first hit from above is its cap
    # side; approximate by the plane z = tip z within the shaft's radius
    for z in np.linspace(tip[2] + 0.06, tip[2], 61):
        tz = (z - cam_pos[2]) / world_dir[..., 2]
        p = cam_pos + world_dir * tz[..., None]
        on = (np.linalg.norm(p[..., :2] - tip[:2], axis=-1) <= peg_r) & ~(depth < tz)
        on &= np.isnan(depth) | (tz < depth)
        depth[on] = tz[on]
    return depth


def test_read_tip_and_mouth_from_one_depth_view(mod):
    K = np.array([[400.0, 0, 160], [0, 400.0, 120], [0, 0, 1]])
    mouth = MOUTH.copy()
    true_mouth = mouth + np.array([0.002, -0.001, 0.0])
    tip = true_mouth + np.array([-0.001, 0.0005, 0.02])
    cam = true_mouth + np.array([0.03, 0.0, 0.15])
    depth = _render(true_mouth, tip, 0.0055, 0.005, cam, K)
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = cam
    view = {"depth": depth, "K": K, "T": T}
    seen = mod.read_tip_and_mouth(
        view,
        tip + np.array([0.002, 0.0, 0.0]),
        mouth,
        np.array([0.0, 0.0, -1.0]),
        0.005,
        0.0055,
        0.006,
    )
    assert seen is not None and "mouth" in seen and "tip" in seen
    assert seen["mouth"][:2] == pytest.approx(true_mouth[:2], abs=0.0015)
    assert seen["tip"][:2] == pytest.approx(tip[:2], abs=0.0015)


def test_camera_refinement_steps_toward_the_seen_mouth(mod):
    K = np.array([[400.0, 0, 160], [0, 400.0, 120], [0, 0, 1]])
    true_mouth = MOUTH + np.array([0.002, 0.0, 0.0])
    w = World(true_mouth=true_mouth, clear=0.0005)

    def observe(**kw):
        tip = w.tip_of(w.H)
        cam = w.H[:3, 3] + np.array([0.03, 0.0, 0.0])
        depth = _render(true_mouth, tip, (D + 0.0005) / 2, D / 2, cam, K)
        T = np.eye(4)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = cam
        return {
            "cameras": {
                "right_wrist": {
                    "rgb": np.zeros((240, 320, 3), np.uint8),
                    "depth": depth,
                    "intrinsics": K.tolist(),
                    "pose": {"position": _v(cam), "rotation": _q(Rotation.from_matrix(T[:3, :3]))},
                }
            }
        }

    out = mod.run(w.ctx(observation=observe), held=held(), hole=hole(), peg=PEG, settle_period=0.0)
    assert out["refined_by"].startswith("camera"), out
    assert out["camera_correction"] > 0.001
    assert out["route"] == "inserted", out["detail"]
    assert math.isclose(out["hole_position"]["x"], true_mouth[0], abs_tol=0.0008)
