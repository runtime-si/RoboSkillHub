"""A fake YAM arm over a ray-cast LEGO / DUPLO scene for correcting-brick-offset-from-stud-circles.

The wrist D405 is rendered (tests/bundles/_plate_render.py) from where the hand really is; the arm
reports its pose with a fixed bias, lands each move with a random and a proportional error, and the
reported camera pose uses a nominal hand-eye transform that may differ from the true one.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from gap_core.errors import ToolError
from scipy.spatial.transform import Rotation

from . import _plate_render as R

_s = math.sqrt(0.5)
T_TOOL_CAM = np.eye(4)
T_TOOL_CAM[:3, :3] = np.column_stack([[0, 1, 0], [-_s, 0, _s], [_s, 0, _s]])
T_TOOL_CAM[:3, 3] = [-0.086, 0.0, -0.075]
"""A YAM-like wrist D405: 86 mm behind and 75 mm above the tool centre, looking 45 deg forward."""
TOP_DOWN = np.diag([1.0, -1.0, -1.0])


def se3(m):
    x, y, z, w = Rotation.from_matrix(m[:3, :3]).as_quat()
    return {
        "position": dict(zip("xyz", map(float, m[:3, 3]), strict=True)),
        "rotation": {"w": w, "x": x, "y": y, "z": z},
    }


def mat(pose):
    q = pose["rotation"]
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    m[:3, 3] = [pose["position"][k] for k in "xyz"]
    return m


def small(dx=0.0, dy=0.0, dz=0.0, rx=0.0, ry=0.0, rz=0.0):
    """A small rigid offset (m, deg)."""
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    m[:3, 3] = (dx, dy, dz)
    return m


class Scene:
    """A brick on a 16 x 16 plate (plate frame = R's), the plate placed in the world at tw."""

    def __init__(
        self,
        system="lego",
        brick=(5, 6, 2, 4),
        others=(),
        floaters=(),
        yaw_deg=20.0,
        origin=(0.40, 0.08, 0.0),
        pair=(0, 1),
    ):
        self.system = system
        self.g = R.SYSTEMS[system]
        bx, by, bnx, bny = brick
        bricks = [R.Brick(bx, by, bnx, bny, 1, 0)]
        bricks += [R.Brick(*o) for o in others]
        bricks += [R.Brick(*f) for f in floaters]
        self.scene = R.Scene(system=system, studs=(16, 16), bricks=bricks)
        self.tw = R.plate_pose_world(*origin, yaw_deg)
        p, bh = self.g["pitch"], self.g["brick_h"]
        # The pair: studs (bx, by + j) and (bx + 1, by + j), x along the brick's first axis.
        j = pair[1]
        cx = -8 * p + (bx + 1.0) * p
        cy = -8 * p + (by + j + 0.5) * p
        m = np.eye(4)
        m[:3, 3] = (cx, cy, bh)
        self.pair = self.tw @ m
        """True pair pose in the world: midpoint at the brick top, x along the pair."""


@lru_cache(maxsize=256)
def _render(scene_key, cam_bytes, seed):
    scene = _SCENES[scene_key]
    cam = np.frombuffer(cam_bytes).reshape(4, 4)
    rgb, d = R.render(
        scene.scene,
        R.D405_K,
        np.linalg.inv(scene.tw) @ cam,
        rng=np.random.default_rng(seed),
        supersample=1,
    )
    return rgb, d, R.render.last_ids.copy()


_SCENES: dict[int, Scene] = {}


class Arm:
    def __init__(
        self,
        scene: Scene,
        start: np.ndarray,
        land_sigma: float = 0.0,
        land_frac: float = 0.0,
        land_yaw_deg: float = 0.0,
        read_bias=(0.0, 0.0, 0.0),
        hand_eye_error: np.ndarray | None = None,
        refuse=None,
        seed: int = 0,
    ):
        self.scene = scene
        _SCENES[id(scene)] = scene
        self.true = start.copy()
        self.bias = np.asarray(read_bias, float)
        self.cam_true = T_TOOL_CAM @ (hand_eye_error if hand_eye_error is not None else np.eye(4))
        self.land_sigma, self.land_frac, self.land_yaw = land_sigma, land_frac, land_yaw_deg
        self.refuse = refuse or (lambda pose: None)
        self.rng = np.random.default_rng(seed)
        self.moves: list[np.ndarray] = []
        self.kwargs: list[dict] = []
        self.frames = 0

    def reported(self):
        m = self.true.copy()
        m[:3, 3] += self.bias
        return m

    def go(self, pose, **kwargs):
        cmd = mat(pose)
        self.moves.append(cmd)
        self.kwargs.append(kwargs)
        why = self.refuse(pose)
        if why == "raise":
            raise ToolError("go_to_pose", "joint limit")
        if why:
            return {"reached": False, "outcome": "unreachable", "reason": why}
        goal = cmd.copy()
        goal[:3, 3] -= self.bias
        step = goal[:3, 3] - self.true[:3, 3]
        err = np.zeros(3)
        err[:2] = self.rng.normal(0.0, self.land_sigma, 2)
        err += self.land_frac * step * np.array([1.0, 1.0, 0.0])
        goal[:3, 3] += err
        if self.land_yaw:
            goal[:3, :3] = (
                Rotation.from_euler(
                    "z", self.rng.normal(0.0, self.land_yaw), degrees=True
                ).as_matrix()
                @ goal[:3, :3]
            )
        self.true = goal
        return {"reached": True, "outcome": "reached"}

    def ee(self, **_):
        return {"pose": se3(self.reported())}

    def obs(self, **_):
        cam = self.true @ self.cam_true
        rgb, d, _ = _render(id(self.scene), cam.round(7).tobytes(), self.frames % 7)
        self.frames += 1
        frame = {
            "name": "left_wrist",
            "rgb": rgb,
            "depth": d,
            "intrinsics": R.D405_K,
            "pose": se3(self.reported() @ T_TOOL_CAM),
        }
        return {"cameras": [frame], "arms": [{"ee_pose": se3(self.reported())}]}

    def error(self, yaw_in_pair_deg=None):
        """True lateral distance [mm] of the tool axis from the pair midpoint, and the yaw error [deg]
        of the tool's x from the pair line plus yaw_in_pair (folded to 180)."""
        pair = self.scene.pair
        d = self.true[:3, 3] - pair[:3, 3]
        lat = float(np.hypot(d @ pair[:3, 0], d @ pair[:3, 1])) * 1e3
        yaw = math.degrees(
            math.atan2(self.true[:3, 0] @ pair[:3, 1], self.true[:3, 0] @ pair[:3, 0])
        )
        want = yaw_in_pair_deg if yaw_in_pair_deg is not None else 0.0
        return lat, abs((yaw - want + 90.0) % 180.0 - 90.0)


def start_over(scene: Scene, dx=0.0, dy=0.0, dyaw_deg=0.0, height=0.03) -> np.ndarray:
    """Tool pose top-down over the true pair (+ an offset in the pair frame), x along the pair."""
    pair = scene.pair
    m = np.eye(4)
    m[:3, :3] = (
        pair[:3, :3] @ Rotation.from_euler("z", dyaw_deg, degrees=True).as_matrix() @ TOP_DOWN
    )
    m[:3, 3] = (
        pair[:3, 3]
        + dx * pair[:3, 0]
        + dy * pair[:3, 1]
        + (scene.g["stud_h"] + height) * pair[:3, 2]
    )
    return m


def guess(scene: Scene, dx=0.0, dy=0.0, dyaw_deg=0.0) -> dict:
    """The pair pose off by (dx, dy) [m] and dyaw [deg] in its own frame."""
    return se3(scene.pair @ small(dx, dy, 0.0, 0.0, 0.0, dyaw_deg))
