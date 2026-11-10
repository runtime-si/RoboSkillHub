"""scanning-the-board-into-a-wrist-mosaic: a fake arm carrying a fake wrist camera over a textured table plane."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "scanning-the-board-into-a-wrist-mosaic"
K = np.array([[400.0, 0.0, 160.0], [0.0, 400.0, 120.0], [0.0, 0.0, 1.0]])
W, H = 320, 240
TEX_RES = 0.0002  # m per texture pixel
TEX_HALF = 0.25  # the texture spans +-25 cm round the world origin
Z0 = 0.01  # the table surface


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["scan_mosaic"].module


@pytest.fixture(scope="module")
def texture():
    rng = np.random.default_rng(0)
    n = int(2 * TEX_HALF / TEX_RES)
    small = rng.integers(0, 255, (n // 16, n // 16, 3), dtype=np.uint8)
    img = cv2.resize(small, (n, n), interpolation=cv2.INTER_CUBIC)
    for _ in range(300):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p = tuple(int(v) for v in rng.integers(0, n, 2))
        cv2.circle(img, p, int(rng.integers(5, 40)), c, -1)
    return img


def _tex_to_world():
    # texture pixel (u, v) -> world (x, y): x right, y up, centred on the origin
    return np.array([[TEX_RES, 0.0, -TEX_HALF], [0.0, -TEX_RES, TEX_HALF], [0.0, 0.0, 1.0]])


def _quat(r):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(r).as_quat()
    return {"w": float(w), "x": float(x), "y": float(y), "z": float(z)}


def _matrix(pose):
    from scipy.spatial.transform import Rotation

    q = pose["rotation"]
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    m[:3, 3] = [pose["position"][k] for k in "xyz"]
    return m


class Station:
    """The tool goes where commanded (plus ``err``); the camera rides on it at TOOL_CAM; the observation
    reports the camera pose with ``calib_err`` added (a hand-eye error)."""

    TOOL_CAM = np.array(
        [[0, -1, 0, 0.03], [1, 0, 0, 0.0], [0, 0, 1, 0.08], [0, 0, 0, 1.0]], dtype=float
    )

    def __init__(
        self, texture, unreachable=None, err=(0.0, 0.0), calib_err=(0.0, 0.0), cameras=True
    ):
        self.tex = texture
        self.tool = np.eye(4)
        self.tool[:3, :3] = np.diag([1.0, -1.0, -1.0])
        self.tool[:3, 3] = [0.0, 0.0, 0.3]
        self.unreachable = unreachable or (lambda p: False)
        self.err = np.array([*err, 0.0])
        self.calib = np.array([*calib_err, 0.0])
        self.cameras = cameras
        self.moves = 0

    def cam_pose(self):
        return self.tool @ self.TOOL_CAM

    def render(self, cam):
        w2c = np.linalg.inv(cam)
        r, t = w2c[:3, :3], w2c[:3, 3]
        h_plane = K @ np.column_stack([r[:, 0], r[:, 1], r[:, 2] * Z0 + t])
        h = h_plane @ _tex_to_world()
        rgb = cv2.warpPerspective(self.tex, h, (W, H), flags=cv2.INTER_LINEAR)
        # Depth of the plane z = Z0 along each pixel's ray.
        vv, uu = np.mgrid[0:H, 0:W]
        rays = np.stack(
            [(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1
        )
        rw = rays @ cam[:3, :3].T
        depth = (Z0 - cam[2, 3]) / rw[..., 2]
        return rgb, depth.astype(np.float32)

    def ctx(self):
        def pose_of(m):
            return {
                "position": dict(zip("xyz", (float(v) for v in m[:3, 3]), strict=True)),
                "rotation": _quat(m[:3, :3]),
            }

        def obs():
            cam = self.cam_pose()
            rgb, depth = self.render(cam)
            reported = cam.copy()
            reported[:3, 3] += self.calib
            cams = [
                {
                    "name": "wrist_cam",
                    "rgb": rgb,
                    "depth": depth,
                    "intrinsics": K,
                    "pose": pose_of(reported),
                }
            ]
            return {"cameras": cams if self.cameras else []}

        def go(pose, **kw):
            m = _matrix(pose)
            if self.unreachable(m):
                return {"reached": False, "outcome": "unreachable"}
            m[:3, 3] += self.err
            self.tool = m
            self.moves += 1
            return {"reached": True}

        def ik(pose, **kw):
            return {"success": not self.unreachable(_matrix(pose))}

        return FakeContext(
            {
                "robot.get_ee_pose": lambda **kw: pose_of(self.tool),
                "robot.get_observation": lambda **kw: obs(),
                "robot.go_to_pose": go,
                "robot.solve_ik": ik,
                "robot.wait_steps": lambda **kw: None,
            }
        )


def _expected(script, texture, out, size, yaw=0.0):
    """The texture resampled into the mosaic's grid."""
    w, h = out["size_px"]
    to_mosaic = script.plane_to_mosaic(np.zeros(3), yaw, (w, h), out["metres_per_pixel"])
    m = to_mosaic @ _tex_to_world()
    return cv2.warpPerspective(texture, m, (w, h), flags=cv2.INTER_LINEAR)


def _ncc(a, b, mask):
    a = a[mask].astype(float).ravel()
    b = b[mask].astype(float).ravel()
    a -= a.mean()
    b -= b.mean()
    return float(a @ b / math.sqrt((a @ a) * (b @ b)))


def test_the_mosaic_reproduces_the_table(script, texture):
    st = Station(texture)
    out = script.run(
        st.ctx(),
        region_center={"x": 0.0, "y": 0.0, "z": Z0},
        region_size=[0.2, 0.14],
        px_per_m=2000.0,
    )
    assert out["route"] == "mosaic_built", out["detail"]
    assert out["camera_height_m"] == pytest.approx(0.2)
    assert out["tiles"] == out["tiles_seen"] > 1
    assert out["covered_fraction"] == pytest.approx(1.0)
    assert out["size_px"] == [400, 280]
    exp = _expected(script, texture, out, (0.2, 0.14))
    gray = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)  # noqa: E731
    assert _ncc(gray(out["mosaic"]), gray(exp), np.ones((280, 400), bool)) > 0.95
    assert out["seam_shift_px"] < 0.5
    hm = out["height_map"]
    assert np.nanmax(np.abs(hm)) < 0.0005 and np.isfinite(hm).mean() > 0.9


def test_a_turned_region_and_a_low_camera(script, texture):
    st = Station(texture)
    out = script.run(
        st.ctx(),
        region_center={"x": 0.02, "y": -0.01, "z": Z0},
        region_size=[0.12, 0.1],
        region_yaw_deg=30.0,
        px_per_m=6000.0,
    )
    assert out["camera_height_m"] == pytest.approx(script.MIN_HEIGHT_M)
    assert out["route"] == "mosaic_built"
    w, h = out["size_px"]
    to_mosaic = script.plane_to_mosaic(
        np.array([0.02, -0.01, 0.0]), 30.0, (w, h), out["metres_per_pixel"]
    )
    exp = cv2.warpPerspective(texture, to_mosaic @ _tex_to_world(), (w, h))
    g = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)  # noqa: E731
    assert _ncc(g(out["mosaic"]), g(exp), np.ones((h, w), bool)) > 0.9


def test_a_pose_error_the_camera_reports_is_harmless_a_calibration_error_shows_in_the_seams(
    script, texture
):
    st = Station(texture, err=(0.003, -0.002))
    out = script.run(
        st.ctx(),
        region_center={"x": 0.0, "y": 0.0, "z": Z0},
        region_size=[0.2, 0.14],
        px_per_m=2000.0,
    )
    assert out["seam_shift_px"] < 0.5
    # A hand-eye error that changes with the frame (here: alternating) is not in the reported pose.
    st = Station(texture)
    flip = {"n": 0}
    base = st.ctx()

    class Ctx:
        def tool(self, name, **kw):
            out = base.tool(name, **kw)
            if name == "robot.get_observation" and out["cameras"]:
                flip["n"] += 1
                out["cameras"][0]["pose"]["position"]["x"] += 0.002 * (1 if flip["n"] % 2 else -1)
            return out

    out = script.run(
        Ctx(), region_center={"x": 0.0, "y": 0.0, "z": Z0}, region_size=[0.2, 0.14], px_per_m=2000.0
    )
    assert out["seam_shift_px"] > 2.0


def test_unreachable_frames_leave_a_partial_mosaic(script, texture):
    # Frames over x > ~0.11 are out of reach whichever way the camera (3 cm off the tool) is rolled.
    st = Station(texture, unreachable=lambda m: m[0, 3] > 0.08)
    out = script.run(
        st.ctx(),
        region_center={"x": 0.0, "y": 0.0, "z": Z0},
        region_size=[0.4, 0.14],
        px_per_m=2000.0,
    )
    assert out["route"] == "partial", out["detail"]
    assert 0.2 < out["covered_fraction"] < 0.98
    assert out["tiles_seen"] < out["tiles"]


def test_failures(script, texture):
    st = Station(texture, cameras=False)
    out = script.run(st.ctx(), region_center={"x": 0.0, "y": 0.0, "z": Z0}, region_size=[0.2, 0.14])
    assert out["route"] == "failed" and "camera" in out["detail"]
    st = Station(texture)
    out = script.run(
        st.ctx(),
        region_center={"x": 0.0, "y": 0.0, "z": Z0},
        region_size=[1.5, 1.5],
        px_per_m=8000.0,
    )
    assert out["route"] == "failed" and "frames needed" in out["detail"]
    assert st.moves == 0
    st = Station(texture, unreachable=lambda m: True)
    out = script.run(st.ctx(), region_center={"x": 0.0, "y": 0.0, "z": Z0}, region_size=[0.2, 0.14])
    assert out["route"] == "failed"
    assert (
        script.run(FakeContext(), region_center={"x": 0, "y": 0, "z": 0}, region_size=[0.1, 0.1])[
            "route"
        ]
        == "failed"
    )


def test_plan_tiles_covers_the_region(script):
    tiles = script.plan_tiles(np.zeros(3), (0.3, 0.2), 0.0, (0.1, 0.08))
    xs = sorted({round(t[0], 6) for t in tiles})
    ys = sorted({round(t[1], 6) for t in tiles})
    assert xs[0] == pytest.approx(-0.1) and xs[-1] == pytest.approx(0.1)
    assert ys[0] == pytest.approx(-0.06) and ys[-1] == pytest.approx(0.06)
    assert max(np.diff(xs)) <= 0.1 * (1 - script.OVERLAP) + 1e-9
    assert len(script.plan_tiles(np.zeros(3), (0.05, 0.05), 0.0, (0.1, 0.08))) == 1
