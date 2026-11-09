"""calibrating-arm-to-arm-base-transform: the AX = ZB solver, target posing on rendered boards,
and the skill's motion, routing and composition with FakeContext."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from gap.testing import FakeContext
from scipy.spatial.transform import Rotation

BUNDLE = "calibrating-arm-to-arm-base-transform"
K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
PPM = 4000  # rendered board pixels per metre


@pytest.fixture(scope="module")
def bundle(skills_registry):
    b = skills_registry.get(BUNDLE)
    assert b.kind == "skill"
    return b


@pytest.fixture(scope="module")
def script(bundle):
    return bundle.canonical_scripts["calibrate_arms"].module


@pytest.fixture(scope="module")
def he(bundle):
    path = Path(bundle.bundle_dir) / "scripts" / "handeye.py"
    spec = importlib.util.spec_from_file_location("handeye_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def T(euler_deg=(0, 0, 0), t=(0, 0, 0)):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    m[:3, 3] = t
    return m


def inv(m):
    return np.linalg.inv(m)


def err(a, b):
    mm = float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1e3)
    deg = float(np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()))
    return mm, deg


def pose_dict(m):
    q = Rotation.from_matrix(m[:3, :3]).as_quat()
    return {
        "position": dict(zip("xyz", map(float, m[:3, 3]), strict=True)),
        "rotation": dict(zip("xyzw", map(float, q), strict=True)),
    }


_BOARD_IMG = None


def render_board(T_cam_board, script):
    """A 640 x 480 RGB frame of the default ChArUco board at ``T_cam_board`` (pinhole K)."""
    global _BOARD_IMG
    t = script.DEFAULT_TARGET
    if _BOARD_IMG is None:
        board = cv2.aruco.CharucoBoard(
            tuple(t["squares"]),
            t["square_m"],
            t["marker_m"],
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
        )
        w = int(round(t["squares"][0] * t["square_m"] * PPM))
        h = int(round(t["squares"][1] * t["square_m"] * PPM))
        _BOARD_IMG = board.generateImage((w, h), marginSize=0)
    s = np.diag([1.0 / PPM, 1.0 / PPM, 1.0])
    hm = K @ np.column_stack([T_cam_board[:3, 0], T_cam_board[:3, 1], T_cam_board[:3, 3]]) @ s
    out = cv2.warpPerspective(_BOARD_IMG, hm, (640, 480), flags=cv2.INTER_AREA, borderValue=200)
    return np.dstack([out] * 3)


CENTRE = np.array([0.09, 0.12, 0.0])  # the default board's centre in its own frame


def view_of_board(euler, dist=0.42):
    m = T(euler)
    m[:3, 3] = np.array([0.0, 0.0, dist]) - m[:3, :3] @ CENTRE
    return m


VIEW_ANGLES = [
    (15, -10, 30),
    (-20, 15, -60),
    (10, 20, 0),
    (-15, -15, 45),
    (20, 5, -30),
    (-10, 10, 80),
    (5, -20, -10),
]

# Two arms 0.62 m apart, the second turned 1.5 deg, as on the YAM station.
T_L_R = T((0.3, -0.2, 1.5), (0.62, 0.015, -0.004))
BOARD_IN_L = T((180, 0, 92), (0.32, 0.42, -0.03))
HAND_EYE = [T((-45, 1, -90), (-0.085, 0.01, -0.074)), T((-44, -1, -91), (-0.083, 0.007, -0.072))]


def arm_views(arm, noise_mm=0.0, n=7, seed=0):
    rng = np.random.default_rng(seed + arm)
    z = BOARD_IN_L if arm == 0 else inv(T_L_R) @ BOARD_IN_L
    out = []
    for e in VIEW_ANGLES[:n]:
        cam_board = view_of_board(e)
        tool = z @ inv(cam_board) @ inv(HAND_EYE[arm])
        seen = cam_board.copy()
        if noise_mm:
            seen = seen @ T(rng.normal(0, noise_mm / 20, 3), rng.normal(0, noise_mm / 1e3, 3))
        out.append({"T_base_tool": tool.tolist(), "T_cam_target": seen.tolist()})
    return out


# --- solver -----------------------------------------------------------------------


def test_closed_form_and_refinement_recover_x_and_z(he):
    rng = np.random.default_rng(3)
    x = T(rng.normal(0, 40, 3), rng.normal(0, 0.1, 3))
    z = T(rng.normal(0, 40, 3), rng.normal(0, 0.5, 3))
    a = [T(rng.normal(0, 35, 3), rng.normal(0, 0.3, 3)) for _ in range(10)]
    b = [inv(z) @ ai @ x for ai in a]
    for solve in (he.solve_axzb_closed_form, he.solve_axzb):
        xs, zs = solve(a, b)
        assert err(xs, x)[0] < 1e-6 and err(zs, z)[1] < 1e-6
    b_noisy = [bi @ T(rng.normal(0, 0.05, 3), rng.normal(0, 0.001, 3)) for bi in b]  # deg, m
    xs, zs = he.solve_axzb(a, b_noisy)
    assert err(zs, z)[0] < 3.0 and err(xs, x)[1] < 0.5
    mm, deg = he.per_view_errors(a, b_noisy, xs, zs)
    assert len(mm) == len(deg) == 10 and np.median(mm) < 5.0


def test_level_keeps_translation_and_yaw(he):
    m = T((2.0, -1.0, 30.0), (0.6, 0.02, 0.01))
    flat = he.level(m)
    assert np.allclose(flat[:3, 3], m[:3, 3])
    assert he.describe(flat)["tilt_deg"] == pytest.approx(0.0, abs=1e-6)
    assert he.describe(flat)["yaw_deg"] == pytest.approx(he.describe(m)["yaw_deg"], abs=1e-6)


# --- target posing ------------------------------------------------------------------


@pytest.mark.parametrize("euler", VIEW_ANGLES[:4])
def test_charuco_pose_from_a_rendered_frame(script, euler):
    truth = view_of_board(euler)
    got = script.pose_target(render_board(truth, script), K, None, script.DEFAULT_TARGET)
    assert got is not None and got["corners"] >= 30 and got["reproj_px"] < 0.5
    mm, deg = err(got["T_cam_target"], truth)
    assert mm < 1.0 and deg < 0.2


def test_nothing_on_a_blank_frame(script):
    assert (
        script.pose_target(np.full((480, 640, 3), 200, np.uint8), K, None, script.DEFAULT_TARGET)
        is None
    )


def test_single_aruco_marker(script):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side_m, ppm = 0.05, 4000
    marker = cv2.aruco.generateImageMarker(dictionary, 7, int(side_m * ppm))
    pad = int(0.01 * ppm)
    img = np.full((marker.shape[0] + 2 * pad,) * 2, 255, np.uint8)
    img[pad:-pad, pad:-pad] = marker
    truth = T((10, -15, 20))
    truth[:3, 3] = [0.0, 0.0, 0.3]
    # the marker's centre is its frame's origin; the padded image's centre is that point
    off = np.diag([1.0 / ppm, 1.0 / ppm, 1.0])
    off[:2, 2] = -img.shape[0] / 2 / ppm
    hm = K @ np.column_stack([truth[:3, 0], truth[:3, 1], truth[:3, 3]]) @ off
    frame = cv2.warpPerspective(img, hm, (640, 480), flags=cv2.INTER_AREA, borderValue=200)
    target = {"type": "aruco", "dictionary": "DICT_4X4_50", "marker_m": side_m, "id": 7}
    got = script.pose_target(np.dstack([frame] * 3), K, None, target)
    assert got is not None and got["corners"] == 4
    # OpenCV's marker frame: x right, y up, z out of the marker (the image's y down is -y)
    mm, deg = err(got["T_cam_target"], truth @ T((180, 0, 0)))
    assert mm < 2.0 and deg < 2.0  # one 5 cm marker: about a degree of tilt uncertainty
    assert script.pose_target(np.dstack([frame] * 3), K, None, {**target, "id": 8}) is None


# --- the skill from stored views -------------------------------------------------------


def test_views_give_the_base_transform(script):
    out = script.run(FakeContext({}), views={"0": arm_views(0), "1": arm_views(1)})
    assert out["route"] == "calibrated"
    mm, deg = err(np.array(out["T_base0_base1"]), T_L_R)
    assert mm < 0.01 and deg < 0.001
    assert out["translation_m"] == pytest.approx([0.62, 0.015, -0.004], abs=1e-5)
    assert out["yaw_deg"] == pytest.approx(1.5, abs=0.01)
    assert out["jackknife_mm"] < 0.01
    left = out["arms"][0]
    assert left["views"] == 7 and left["rms_mm"] < 0.01
    assert err(np.array(left["T_tool_cam"]), HAND_EYE[0])[0] < 0.01


def test_station_view_keys_are_accepted(script):
    def board_keys(views):
        return [{"T_base_tool": v["T_base_tool"], "T_cam_board": v["T_cam_target"]} for v in views]

    out = script.run(
        FakeContext({}), views={"0": board_keys(arm_views(0)), "1": board_keys(arm_views(1))}
    )
    assert out["route"] == "calibrated"


def test_eye_to_hand_with_one_fixed_camera(script):
    cam_in_l = T((-150, 0, 0), (0.3, 0.1, 0.9))
    tags = [T((0, 0, 0), (0.0, 0.0, 0.03)), T((0, 0, 90), (0.0, 0.01, 0.025))]
    views = {}
    for arm in (0, 1):
        cam = cam_in_l if arm == 0 else inv(T_L_R) @ cam_in_l
        views[str(arm)] = []
        for k, e in enumerate(VIEW_ANGLES):
            cam_tag = view_of_board(e, dist=0.8 + 0.02 * k)
            tool = cam @ cam_tag @ inv(tags[arm])
            views[str(arm)].append({"T_base_tool": tool, "T_cam_target": cam_tag})
    out = script.run(FakeContext({}), views=views, mode="eye_to_hand")
    assert out["route"] == "calibrated"
    assert err(np.array(out["T_base0_base1"]), T_L_R)[0] < 0.01
    assert "T_tool_target" in out["arms"][1] and "T_base_cam" in out["arms"][1]


def test_too_few_views_route_tag_not_seen(script):
    out = script.run(FakeContext({}), views={"0": arm_views(0), "1": arm_views(1, n=4)})
    assert out["route"] == "tag_not_seen" and "[1]" in out["reason"]
    assert out["arms"][1]["views"] == 4 and out["T_base0_base1"] == []


def test_noisy_views_route_residual_too_high(script):
    out = script.run(FakeContext({}), views={"0": arm_views(0, noise_mm=15), "1": arm_views(1)})
    assert out["route"] == "residual_too_high"
    assert "arm 0 RMS" in out["reason"] or "leave-one-out" in out["reason"]
    out = script.run(
        FakeContext({}),
        views={"0": arm_views(0, noise_mm=15), "1": arm_views(1)},
        max_rms_mm=100,
        max_rms_deg=100,
        max_jackknife_mm=1000,
    )
    assert out["route"] == "calibrated"


def test_assume_level_drops_the_tilt(script):
    out = script.run(
        FakeContext({}), views={"0": arm_views(0), "1": arm_views(1)}, assume_level=True
    )
    assert out["tilt_deg"] == pytest.approx(0.0, abs=1e-6)
    assert out["yaw_deg"] == pytest.approx(1.5, abs=0.02)


def test_bad_arguments(script):
    with pytest.raises(ValueError, match="mode"):
        script.run(FakeContext({}), views={}, mode="both")
    with pytest.raises(ValueError, match="two different"):
        script.run(FakeContext({}), views={}, arm_ids=[1, 1])
    with pytest.raises(ValueError, match="joint_configs"):
        script.run(FakeContext({}))


# --- the skill moving the arms -----------------------------------------------------------


class Station:
    """Two arms; joint config [k, ...] puts the arm at its k-th view pose; the wrist camera
    of the arm that moved last renders the board."""

    def __init__(self, script, refuse=()):
        self.script = script
        self.refuse = set(refuse)
        self.arm = 0
        self.k = 0
        self.moves = []

    def tools(self):
        return {
            "robot.move_to_joints": self.move,
            "robot.get_ee_pose": self.ee,
            "robot.get_observation": self.obs,
        }

    def move(self, joint_config, arm_id=0):
        self.moves.append((arm_id, joint_config["positions"][0]))
        if (arm_id, int(joint_config["positions"][0])) in self.refuse:
            return {"outcome": "unreachable", "reached": False, "reason": "IK failed"}
        self.arm, self.k = arm_id, int(joint_config["positions"][0])
        return {"outcome": "reached"}

    def _views(self):
        return arm_views(self.arm)

    def ee(self, arm_id=0):
        assert arm_id == self.arm
        return pose_dict(np.array(self._views()[self.k]["T_base_tool"]))

    def obs(self):
        cam_board = np.array(self._views()[self.k]["T_cam_target"])
        name = ("left_wrist", "right_wrist")[self.arm]
        return {
            "cameras": [
                {"name": name, "rgb": render_board(cam_board, self.script), "intrinsics": K}
            ]
        }


def test_tour_moves_each_arm_and_solves_from_its_frames(script):
    station = Station(script)
    configs = {str(a): [[float(k), 0, 0, 0, 0, 0] for k in range(7)] for a in (0, 1)}
    out = script.run(FakeContext(station.tools()), joint_configs=configs)
    assert [a for a, _ in station.moves] == [0] * 7 + [1] * 7
    assert out["route"] == "calibrated", out["reason"]
    mm, deg = err(np.array(out["T_base0_base1"]), T_L_R)
    assert mm < 2.0 and deg < 0.2
    assert all(o["rms_mm"] < 2.0 for o in out["arms"])


def test_refused_moves_are_skipped_and_counted(script):
    station = Station(script, refuse={(1, 2), (1, 4)})
    configs = {str(a): [[float(k), 0, 0, 0, 0, 0] for k in range(7)] for a in (0, 1)}
    out = script.run(FakeContext(station.tools()), joint_configs=configs, min_views=6)
    assert out["route"] == "tag_not_seen"
    assert out["arms"][1]["views"] == 5 and "move refused" in out["reason"]


def test_example_declares_every_route(bundle):
    ex = json.loads((Path(bundle.bundle_dir) / "examples" / "canonical_subgraph.json").read_text())
    assert set(ex["exit"]["success_values"]) | {ex["on_error"]} == {
        "calibrated",
        "residual_too_high",
        "tag_not_seen",
    }
