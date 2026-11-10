"""measuring-a-held-piece-from-the-wrist-camera: a piece held at a known pose, seen past the blades by a camera on the tool."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from gap.testing import FakeContext

from .jigsaw_cut import make_puzzle

BUNDLE = "measuring-a-held-piece-from-the-wrist-camera"
MPP = 0.00025
K = np.array([[390.0, 0, 160.0], [0, 390.0, 120.0], [0, 0, 1.0]])
W, H = 320, 240
TCP = np.array([0.5, 0.2, 0.30])
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}  # tool z down, tool x = world x, tool y = world -y
HANG = 0.013  # piece bottom under the tool centre
MAT = TCP[2] - 0.03


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["measure_held"].module


@pytest.fixture(scope="module")
def piece():
    pz = make_puzzle(3, 3, cell=(0.0225, 0.0225), seed=2)
    poly = pz.outline(1, 1) - pz.centre(1, 1)
    n = 200
    mask = np.zeros((n, n), np.uint8)
    cv2.fillPoly(
        mask,
        [np.round(np.column_stack([poly[:, 0], -poly[:, 1]]) / MPP + n / 2).astype(np.int32)],
        1,
    )
    return poly, mask


def _frame(poly, x, y, yaw_deg, cam_off=(0.0, 0.012, 0.07)):
    """Depth from a camera on the tool looking straight down: the piece's top, the blades (near), the mat."""
    a = math.radians(yaw_deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    world_poly = poly @ r.T + [x, y]
    c = TCP + np.array(cam_off)
    top = TCP[2] - HANG + 0.002
    vv, uu = np.mgrid[0:H, 0:W]
    dx, dy = (
        (uu - K[0, 2]) / K[0, 0],
        -(vv - K[1, 2]) / K[1, 1],
    )  # camera x = world x, camera y = world -y
    depth = np.full((H, W), c[2] - MAT)
    hx, hy = c[0] + dx * (c[2] - top), c[1] + dy * (c[2] - top)
    pm = np.zeros((H, W), bool)
    path = world_poly.astype(np.float32).reshape(-1, 1, 2)
    for v in range(H):
        for u in range(W):
            if cv2.pointPolygonTest(path, (float(hx[v, u]), float(hy[v, u])), False) >= 0:
                pm[v, u] = True
    depth[pm] = c[2] - top
    # Blades: two 15.5 x 3.9 mm columns either side of a 10 mm grip, seen near (above the piece).
    bx, by = c[0] + dx * 0.05, c[1] + dy * 0.05
    blade = (np.abs(bx - TCP[0]) < 0.00775) & (np.abs(np.abs(by - TCP[1]) - 0.007) < 0.00195)
    depth[blade] = 0.05
    pose = {"position": dict(zip("xyz", (float(v) for v in c), strict=True)), "rotation": DOWN}
    return {
        "name": "wrist_cam",
        "depth": depth.astype(np.float32),
        "intrinsics": K,
        "pose": pose,
        "rgb": np.zeros((H, W, 3), np.uint8),
    }


def _ctx():
    return FakeContext(
        {
            "robot.get_ee_pose": {
                "position": dict(zip("xyz", (float(v) for v in TCP), strict=True)),
                "rotation": DOWN,
            }
        }
    )


@pytest.mark.parametrize(
    ("x", "y", "yaw"), [(0.501, 0.198, 0.0), (0.499, 0.203, 7.0), (0.5025, 0.2, -11.0)]
)
def test_the_piece_pose_is_found_from_a_prior_a_few_mm_and_degrees_off(script, piece, x, y, yaw):
    poly, mask = piece
    cam = _frame(poly, x, y, yaw)
    prior = {"x": x + 0.004, "y": y - 0.003, "yaw_deg": yaw + 9.0}
    out = script.run(_ctx(), mask, MPP, prior, 0.01, camera=cam)
    assert out["route"] == "measured", out["detail"]
    wp = out["world_pose"]
    bc = (script.body_centre_px(mask) - 100) * MPP * np.array([1, -1])
    a = math.radians(yaw)
    want = np.array([x, y]) + [
        bc[0] * math.cos(a) - bc[1] * math.sin(a),
        bc[0] * math.sin(a) + bc[1] * math.cos(a),
    ]
    assert math.hypot(wp["x"] - want[0], wp["y"] - want[1]) < 0.0008
    assert abs(wp["yaw_deg"] - yaw) < 2.0
    assert wp["z"] == pytest.approx(TCP[2] - HANG, abs=0.0006)
    # In the tool frame (tool y = world -y): z is the piece's bottom under the tool centre.
    tp = out["tcp_piece"]
    assert tp["x"] == pytest.approx(wp["x"] - TCP[0], abs=1e-6)
    assert tp["y"] == pytest.approx(-(wp["y"] - TCP[1]), abs=1e-6)
    assert tp["z"] == pytest.approx(HANG, abs=0.0006)
    assert out["sigma"]["x"] < 0.0015 and out["sigma"]["yaw_deg"] < 4.0


def test_nothing_under_the_jaws_is_not_seen(script, piece):
    poly, mask = piece
    cam = _frame(poly, 0.7, 0.2, 0.0)  # the piece is not in the hand
    out = script.run(_ctx(), mask, MPP, {"x": 0.5, "y": 0.2, "yaw_deg": 0.0}, 0.01, camera=cam)
    assert out["route"] == "not_seen"


def test_bad_inputs(script, piece):
    poly, mask = piece
    cam = _frame(poly, 0.5, 0.2, 0.0)
    assert (
        script.run(
            _ctx(), np.zeros((10, 10)), MPP, {"x": 0.5, "y": 0.2, "yaw_deg": 0}, 0.01, camera=cam
        )["route"]
        == "not_seen"
    )
    assert (
        script.run(
            FakeContext(
                {
                    "robot.get_ee_pose": {"position": {"x": 0, "y": 0, "z": 0}, "rotation": DOWN},
                    "robot.get_observation": {"cameras": []},
                }
            ),
            mask,
            MPP,
            {"x": 0, "y": 0, "yaw_deg": 0},
            0.01,
        )["route"]
        == "not_seen"
    )
