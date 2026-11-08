"""correcting-insertion-offset-from-port-opening: a ray-cast scene (a plate with a dark square
port, a black fingertip disc under the hand) seen by fixed cameras, a stateful fake arm, and the
routing of every exit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from gap.testing import FakeContext
from gap_core.errors import ToolError
from scipy.spatial.transform import Rotation

BUNDLE = "correcting-insertion-offset-from-port-opening"
BUNDLE_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
TOP_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
MOUTH_Z = 0.05
PORT = np.array([0.40, 0.10, MOUTH_Z])
OPENING = 0.012
HOVER = 0.03
W, H, F = 480, 360, 600.0
K = [[F, 0.0, W / 2], [0.0, F, H / 2], [0.0, 0.0, 1.0]]


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["correct_offset"].module


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera-to-world pose (OpenCV axes: z forward, y down) as a hub pose dict."""
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    z = (target - eye) / np.linalg.norm(target - eye)
    if abs(z @ up) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    y = -(up - (up @ z) * z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    q = Rotation.from_matrix(np.stack([x, y, z], axis=1)).as_quat()
    return {
        "position": dict(zip("xyz", map(float, eye), strict=True)),
        "rotation": {"x": q[0], "y": q[1], "z": q[2], "w": q[3]},
    }


CAMERAS = {
    "top": look_at(PORT + [-0.06, 0.0, 0.22], PORT),
    "side": look_at(PORT + [0.04, -0.13, 0.15], PORT),
}


class Scene:
    """A plate at the mouth's height with a square, dark-floored port at ``port``, and the hand:
    a black fingertip disc 10 mm under the tool centre. ``arm.go_to_pose`` moves the hand."""

    def __init__(self, port=PORT, start=None, lag=0.0, cameras=None, refuse=None, raise_on=None):
        self.port = np.asarray(port, dtype=float)
        self.p = np.array(start if start is not None else [*PORT[:2], MOUTH_Z + HOVER], dtype=float)
        self.target = self.p.copy()
        self.lag = lag
        self.cameras = dict(CAMERAS if cameras is None else cameras)
        self.refuse = refuse
        self.raise_on = raise_on or set()
        self.moves: list[tuple[np.ndarray, dict]] = []
        self.opening = OPENING
        self.uniform = False

    # --- robot tools
    def ee(self, **kw):
        if "ee" in self.raise_on:
            raise ToolError("robot.get_ee_pose", "no reading")
        return {
            "pose": {
                "position": dict(zip("xyz", map(float, self.p), strict=True)),
                "rotation": TOP_DOWN,
            }
        }

    def go(self, pose, **kwargs):
        target = np.array([pose["position"][k] for k in "xyz"])
        self.moves.append((target, kwargs))
        if "move" in self.raise_on:
            raise ToolError("robot.go_to_pose", "IK failed")
        if self.refuse:
            return {"reached": False, "outcome": "unreachable"}
        self.p = self.p + (1.0 - self.lag) * (target - self.p)
        return {"reached": True, "outcome": "reached"}

    def observation(self, **kw):
        cams = [self.render(name, pose) for name, pose in self.cameras.items()]
        return {"arms": [], "cameras": cams}

    # --- the renderer
    def _rays(self, pose):
        T = np.eye(4)
        q = pose["rotation"]
        T[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
        T[:3, 3] = [pose["position"][k] for k in "xyz"]
        u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        cam = np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u)], axis=-1)
        return T[:3, 3], cam @ T[:3, :3].T  # z-depth = the ray parameter, since cam z = 1

    def render(self, name, pose):
        origin, rays = self._rays(pose)

        def hit(z):
            s = (z - origin[2]) / rays[..., 2]
            return s, origin[:2] + s[..., None] * rays[..., :2]

        s_plate, xy = hit(MOUTH_Z)
        rel = np.abs(xy - self.port[:2])
        in_hole = (rel[..., 0] < self.opening / 2) & (rel[..., 1] < self.opening / 2)
        rgb = np.full((H, W), 200, dtype=np.uint8)
        depth = s_plate.copy()
        if not self.uniform:
            rgb[in_hole] = 25
            depth[in_hole] = hit(MOUTH_Z - 0.015)[0][in_hole]
        s_tip, xy_tip = hit(self.p[2] - 0.010)
        on_tip = np.linalg.norm(xy_tip - self.p[:2], axis=-1) < 0.005
        rgb[on_tip] = 10
        depth[on_tip] = s_tip[on_tip]
        return {
            "name": name,
            "rgb": np.repeat(rgb[..., None], 3, axis=-1),
            "depth": depth.astype(np.float32),
            "intrinsics": K,
            "pose": pose,
        }

    def ctx(self, extra=None):
        tools = {
            "robot.get_ee_pose": self.ee,
            "robot.go_to_pose": self.go,
            "robot.get_observation": self.observation,
        }
        tools.update(extra or {})
        return FakeContext(tools)


def lateral_error(scene):
    return float(np.linalg.norm(scene.p[:2] - scene.port[:2]))


ESTIMATE = {"x": float(PORT[0]), "y": float(PORT[1]), "z": MOUTH_Z}


def run(script, scene, **kw):
    kw.setdefault("pixel_tol_frac", 0.04)
    return script.run(scene.ctx(), port_position=ESTIMATE, opening_size=OPENING, **kw)


# ----------------------------------------------------------------------------- the loop


def test_a_hand_4_mm_off_is_stepped_onto_the_opening_in_1_mm_moves(script):
    scene = Scene(port=PORT + [0.004, -0.002, 0.0], start=[*PORT[:2], MOUTH_Z + HOVER])
    out = run(script, scene)
    assert out["route"] == "centered", out["detail"]
    assert lateral_error(scene) < 0.0008
    assert out["views"] == 2
    assert (
        np.linalg.norm([out["hole_position"][k] - scene.port[i] for i, k in enumerate("xy")])
        < 0.0006
    )
    assert abs(out["hole_offset"] - np.hypot(0.004, 0.002)) < 0.0006
    assert 4 <= out["steps"] <= 6 and len(scene.moves) == out["steps"]
    prev = np.array([*PORT[:2], MOUTH_Z + HOVER])
    for target, kwargs in scene.moves:
        assert np.linalg.norm(target - prev) <= 0.001 + 1e-9  # capped step
        assert target[2] == pytest.approx(MOUTH_Z + HOVER)  # in the port plane only
        assert kwargs == {}  # no tolerance unless asked
        prev = target


def test_a_centred_hand_is_left_alone(script):
    scene = Scene()
    out = run(script, scene)
    assert out["route"] == "centered"
    assert out["steps"] == 0 and scene.moves == []


def test_the_upstream_tolerance_stops_sooner(script):
    scene = Scene(port=PORT + [0.004, 0.0, 0.0])
    out = run(script, scene, pixel_tol_frac=script.PIXEL_TOL_FRAC)
    assert out["route"] == "centered"
    assert out["steps"] < 4  # 0.12 x the opening is ~1.4 mm here


def test_an_arm_that_lags_its_command_still_converges(script):
    # Each move covers only 60% of the way to its target: stepping from the read-back pose would
    # stall at 0.6 mm a step, stepping from the last target integrates the lag away.
    scene = Scene(port=PORT + [0.003, 0.0, 0.0], lag=0.4)
    out = run(script, scene)
    assert out["route"] == "centered", out["detail"]
    assert lateral_error(scene) < 0.001


def test_the_held_part_tip_offset_is_what_gets_centred(script):
    # The part's tip sits 3 mm along the hand's x: the hand must end 3 mm the other way.
    scene = Scene(port=PORT)
    out = run(script, scene, tip_in_tcp={"x": 0.003, "y": 0.0, "z": 0.0})
    assert out["route"] == "centered", out["detail"]
    assert scene.p[0] == pytest.approx(PORT[0] - 0.003, abs=0.0008)


def test_move_tolerance_is_passed_only_when_set(script):
    scene = Scene(port=PORT + [0.002, 0.0, 0.0])
    run(script, scene, move_tolerance=0.0005)
    assert scene.moves and all(kw == {"tolerance": 0.0005} for _, kw in scene.moves)


# ----------------------------------------------------------------------------- the exits


def test_one_camera_is_not_enough_by_default(script):
    scene = Scene(cameras={"top": CAMERAS["top"]})
    out = run(script, scene)
    assert out["route"] == "opening_not_found"
    assert scene.moves == []
    # ...unless the caller accepts a single view.
    scene = Scene(port=PORT + [0.002, 0.0, 0.0], cameras={"top": CAMERAS["top"]})
    assert run(script, scene, min_views=1)["route"] == "centered"


def test_no_opening_in_view(script):
    scene = Scene()
    scene.uniform = True
    out = run(script, scene)
    assert out["route"] == "opening_not_found"
    assert out["views"] == 0


def test_cameras_selects_by_name_and_ignores_frames_without_calibration(script):
    scene = Scene()
    obs = scene.observation()
    obs["cameras"].append({"name": "webcam", "rgb": obs["cameras"][0]["rgb"]})  # no K, no pose
    ctx = FakeContext(
        {"robot.get_ee_pose": scene.ee, "robot.go_to_pose": scene.go, "robot.get_observation": obs}
    )
    out = script.run(
        ctx, port_position=ESTIMATE, opening_size=OPENING, cameras=["top", "side", "webcam"]
    )
    assert out["route"] == "centered" and out["views"] == 2


def test_a_miscalibrated_camera_makes_the_views_disagree(script):
    bad = json.loads(json.dumps(CAMERAS["side"]))
    bad["position"]["x"] += 0.006  # reported 6 mm off where it renders from
    scene = Scene()
    ctx = scene.ctx({"robot.get_observation": lambda **kw: {
        "cameras": [scene.render("top", CAMERAS["top"]),
                    dict(scene.render("side", CAMERAS["side"]), pose=bad)]}})  # fmt: skip
    out = script.run(ctx, port_position=ESTIMATE, opening_size=OPENING)
    assert out["route"] == "views_disagree"
    assert scene.moves == []


def test_an_opening_far_from_the_estimate_is_not_corrected(script):
    # A 40 mm opening stays inside the search polygon (1.55 x the mouth) 9 mm off; a smaller one
    # would be clipped by it and measured short.
    scene = Scene(port=PORT + [0.0, 0.009, 0.0], start=[*PORT[:2], MOUTH_Z + HOVER])
    scene.opening = 0.040
    out = script.run(scene.ctx(), port_position=ESTIMATE, opening_size=0.040)
    assert out["route"] == "offset_too_large", out["detail"]
    assert out["hole_offset"] > script.MAX_PORT_OFFSET
    assert scene.moves == []


def test_the_total_excursion_cap(script):
    scene = Scene(port=PORT + [0.004, 0.0, 0.0])
    out = run(script, scene, max_total=0.0025)
    assert out["route"] == "offset_too_large"
    assert len(scene.moves) == 2
    assert np.linalg.norm([out["correction"][k] for k in "xyz"]) <= 0.0025


def test_the_step_budget(script):
    scene = Scene(port=PORT + [0.004, 0.0, 0.0])
    out = run(script, scene, max_steps=2)
    assert out["route"] == "no_convergence"
    assert len(scene.moves) == 2 and out["steps"] == 2


@pytest.mark.parametrize("how", ["raises", "routed"])
def test_a_refused_step_is_unreachable(script, how):
    scene = Scene(port=PORT + [0.004, 0.0, 0.0])
    if how == "raises":
        scene.raise_on = {"move"}
    else:
        scene.refuse = True
    out = run(script, scene)
    assert out["route"] == "unreachable"
    assert len(scene.moves) == 1


def test_no_pose_reading_is_unreachable(script):
    scene = Scene()
    scene.raise_on = {"ee"}
    assert run(script, scene)["route"] == "unreachable"


# ----------------------------------------------------------------------------- perception


def test_the_black_fingertips_are_masked_by_depth(script):
    # The fingertip disc, 15 mm over the mouth, is darker than the opening and hides its -x edge
    # from the top camera. The depth mask takes exactly the disc; the opening is found where it
    # is. Without the mask the disc drags its centre 4 px away.
    scene = Scene(port=PORT, start=[PORT[0] - 0.010, PORT[1], MOUTH_Z + 0.025])
    frame = scene.render("top", CAMERAS["top"])
    Kn, T = np.asarray(K), script._pose_matrix(frame["pose"])
    Rp = script.port_frame(np.array([0.0, 0.0, -1.0]))
    quad = np.array([script.project_point(Kn, T, PORT + Rp[:, 0] * sx * 0.006 + Rp[:, 1] * sy * 0.006)
                     for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])  # fmt: skip
    ignored = script.plane_depth_mask(frame["depth"], Kn, T, PORT, Rp[:, 2], 0.008)
    assert np.array_equal(ignored, frame["rgb"][..., 0] == 10)
    true_uv = script.project_point(Kn, T, scene.port)
    found = script.detect_dark_port_opening(frame["rgb"], quad, ignored, center_uv=true_uv)
    assert found["occluded"] > script.OCCLUSION_REFIT
    assert np.linalg.norm(found["center_uv"] - true_uv) < 1.0
    unmasked = script.detect_dark_port_opening(frame["rgb"], quad, None)
    assert np.linalg.norm(unmasked["center_uv"] - true_uv) > 2.0


def test_an_occluded_opening_is_refitted_not_averaged(script):
    img = np.full((120, 160), 200, np.uint8)
    img[40:80, 60:100] = 20  # a 40 px opening centred at (79.5, 59.5)
    ignored = np.zeros_like(img, dtype=bool)
    ignored[30:58, :] = True  # an occluder over its upper 45%
    quad = np.array([[60, 40], [100, 40], [100, 80], [60, 80]], dtype=float) + [3.0, -2.0]
    found = script.detect_dark_port_opening(img, quad, ignored)
    assert found["occluded"] > script.OCCLUSION_REFIT
    assert abs(found["centroid_uv"][1] - 59.5) > 8  # the visible part's centroid leans away
    assert np.allclose(found["center_uv"], [79.5, 59.5], atol=1.0)


def test_an_unoccluded_opening_keeps_the_upstream_centroid(script):
    img = np.full((120, 160), 200, np.uint8)
    img[40:80, 60:100] = 20
    quad = np.array([[60, 40], [100, 40], [100, 80], [60, 80]], dtype=float) + [4.0, 3.0]
    found = script.detect_dark_port_opening(img, quad, np.zeros_like(img, dtype=bool))
    assert "centroid_uv" not in found
    assert np.allclose(found["center_uv"], [79.5, 59.5], atol=0.5)


def test_low_contrast_and_wrong_sized_blobs_are_rejected(script):
    quad = np.array([[60, 40], [100, 40], [100, 80], [60, 80]], dtype=float)
    flat = np.full((120, 160), 120, np.uint8)
    flat[40:80, 60:100] = 110  # spread 10 < 24
    assert script.detect_dark_port_opening(flat, quad) is None
    speck = np.full((120, 160), 200, np.uint8)
    speck[58:62, 78:82] = 10  # 16 px against a 1600 px mouth
    assert script.detect_dark_port_opening(speck, quad) is None


def test_sam3_masks_stand_in_for_the_dark_threshold(script):
    scene = Scene(port=PORT + [0.003, 0.0, 0.0])
    scene.uniform = True  # an opening no darker than its plate: the dark detector cannot see it
    queries = []

    def segment(image, query, **kw):
        queries.append(query)
        name = (
            "top" if np.array_equal(image, scene.render("top", CAMERAS["top"])["rgb"]) else "side"
        )
        saved, scene.uniform = scene.uniform, False
        frame = scene.render(name, CAMERAS[name])
        scene.uniform = saved
        return {"masks": [frame["rgb"][..., 0] == 25], "scores": [0.9]}

    out = script.run(
        scene.ctx({"sam3.segment_text": segment}),
        port_position=ESTIMATE,
        opening_size=OPENING,
        opening_query="the port's opening",
        pixel_tol_frac=0.04,
    )
    assert out["route"] == "centered", out["detail"]
    assert lateral_error(scene) < 0.001
    assert set(queries) == {"the port's opening"}


# ----------------------------------------------------------------------------- geometry


def test_consensus_keeps_the_agreeing_views_and_drops_an_outlier(script):
    Rp = script.port_frame(np.array([0.0, 0.0, -1.0]))
    pts = [np.array([0.0, 0.0, 0.0]), np.array([0.002, 0.0, 0.0]), np.array([0.010, 0.0, 0.0])]
    hole, gap = script.consensus(pts, Rp, 0.003, 2)
    assert gap == pytest.approx(0.002)
    assert np.allclose(hole, [0.001, 0.0, 0.0])
    assert script.consensus(pts[::2], Rp, 0.003, 2)[0] is None


def test_port_frame_and_ray_plane(script):
    Rp = script.port_frame(np.array([0.0, 0.0, -1.0]))
    assert np.allclose(Rp.T @ Rp, np.eye(3)) and np.allclose(Rp[:, 2], [0, 0, -1])
    side = script.port_frame(np.array([1.0, 0.0, 0.0]))  # x hint parallel to the axis
    assert np.allclose(side[:, 2], [1, 0, 0]) and abs(side[:, 0] @ side[:, 2]) < 1e-9
    T = script._pose_matrix(CAMERAS["side"])
    uv = script.project_point(np.asarray(K), T, PORT + [0.002, 0.001, 0.0])
    back = script.ray_to_plane(uv, np.asarray(K), T, PORT, Rp[:, 2])
    assert np.allclose(back, PORT + [0.002, 0.001, 0.0], atol=1e-9)


# ----------------------------------------------------------------------------- the bundle


def test_every_route_is_declared_in_the_example():
    meta = yaml.safe_load((BUNDLE_DIR / "SKILL.md").read_text().split("---")[1])
    routes = set(meta["gap"]["exit_conditions"])
    example = json.loads((BUNDLE_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    assert declared == routes
    src = (BUNDLE_DIR / "scripts" / "correct_offset.py").read_text()
    for route in routes:
        assert f'"{route}"' in src
