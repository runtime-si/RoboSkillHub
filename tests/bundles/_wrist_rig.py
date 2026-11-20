"""A one-arm rig with a wrist depth camera, ray-cast: a level support, flat parts lying on it, and optionally a
part held upright in the hand. For the bundles that came from yam_dominos (centring a grasp, setting down by the
measured gap, calibrating a wrist camera)."""

from __future__ import annotations

import math

import numpy as np
from gap.testing import FakeContext
from scipy.spatial.transform import Rotation

K = np.array([[430.0, 0.0, 320.0], [0.0, 430.0, 240.0], [0.0, 0.0, 1.0]])
SIZE = (0.049, 0.0245, 0.0074)  # a domino: length, width, thickness [m]
DOWN = np.diag([1.0, -1.0, -1.0])  # a hand pointing straight down: tool z is world -z


def T(rotation: np.ndarray, position) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rotation, position
    return m


CAMERA_IN_TOOL = T(Rotation.from_euler("x", -12.0, degrees=True).as_matrix(), [0.0, -0.035, -0.15])
"""The camera 15 cm back along the hand and 3.5 cm to one side, tilted 12 deg to look at the tool centre."""


def quat(r: np.ndarray) -> dict[str, float]:
    x, y, z, w = Rotation.from_matrix(r).as_quat()
    return {"w": float(w), "x": float(x), "y": float(y), "z": float(z)}


def pose(m: np.ndarray) -> dict:
    return {
        "position": dict(zip("xyz", map(float, m[:3, 3]), strict=True)),
        "rotation": quat(m[:3, :3]),
    }


def matrix(p: dict) -> np.ndarray:
    q = p["rotation"]
    return T(
        Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix(),
        [p["position"][k] for k in "xyz"],
    )


def obb(centre_xy, yaw: float, size=SIZE) -> dict:
    return {
        "center": {"x": float(centre_xy[0]), "y": float(centre_xy[1]), "z": size[2] / 2},
        "extent": {"x": size[0] / 2, "y": size[1] / 2, "z": size[2] / 2},
        "orientation": quat(Rotation.from_euler("z", yaw).as_matrix()),
    }


class Rig:
    """``parts``: [(x, y, yaw)] lying flat. ``held``: (upper end's offset from the tool centre in the world [3],
    how tall the part is, footprint (a, b)) or None. ``believed``: the camera-in-tool pose the connector poses its
    frames with (default: the true one). ``floor``: the lowest the tool centre goes. ``arm_error``: added to every
    move's x, y (the arm is not where it was sent)."""

    def __init__(
        self,
        tool: np.ndarray,
        parts=(),
        held=None,
        believed: np.ndarray | None = None,
        floor: float = -1.0,
        arm_error=(0.0, 0.0),
        reach=lambda m: True,
    ):
        self.tool, self.parts, self.held = tool.copy(), list(parts), held
        self.true, self.believed = CAMERA_IN_TOOL, CAMERA_IN_TOOL if believed is None else believed
        self.floor, self.arm_error, self.reach = floor, np.asarray(arm_error, float), reach
        self.released, self.moves = False, 0
        self.held_low: float | None = (
            None  # the held part's lower end, once it stands on the support
        )

    # ---- the world, ray-cast
    def depth(self) -> np.ndarray:
        cam = self.tool @ self.true
        h, w = 480, 640
        vs, us = np.mgrid[0:h, 0:w]
        rays = (
            np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
            @ cam[:3, :3].T
        )
        o = cam[:3, 3]

        def plane(z: float):
            t = (z - o[2]) / rays[..., 2]
            return t, o[:2] + t[..., None] * rays[..., :2]

        t0, _ = plane(0.0)
        out = np.where(t0 > 0, t0, 0.0)
        for x, y, yaw in self.parts:
            t, p = plane(SIZE[2])
            d = p - [x, y]
            a = d[..., 0] * math.cos(yaw) + d[..., 1] * math.sin(yaw)
            b = -d[..., 0] * math.sin(yaw) + d[..., 1] * math.cos(yaw)
            out = np.where((np.abs(a) < SIZE[0] / 2) & (np.abs(b) < SIZE[1] / 2) & (t > 0), t, out)
        if self.held is not None and not self.released:
            top = self.held_top()
            t, p = plane(top[2])
            d = p - top[:2]
            out = np.where(
                (np.abs(d[..., 0]) < self.held[2][0] / 2)
                & (np.abs(d[..., 1]) < self.held[2][1] / 2)
                & (t > 0),
                t,
                out,
            )
        return out.astype(np.float32)

    def held_top(self) -> np.ndarray:
        """The held part's upper end: with the hand, until its lower end stands on the support (it then slides in the pads)."""
        top = self.tool[:3, 3] + np.asarray(self.held[0], float)
        return np.array([top[0], top[1], max(top[2], self.held[1])])

    # ---- the connector
    def observation(self, **_kw) -> dict:
        return {
            "arms": [{"ee_pose": pose(self.tool)}],
            "cameras": [
                {
                    "name": "wrist",
                    "pose": pose(self.tool @ self.believed),
                    "intrinsics": K.tolist(),
                    "depth": self.depth(),
                    "rgb": None,
                }
            ],
        }

    def move(self, pose: dict, **_kw) -> dict:  # noqa: A002 -- the connector's argument name
        goal = matrix(pose)
        if not self.reach(goal):
            return {"success": False, "reason": "refused"}
        goal[:2, 3] += self.arm_error
        goal[2, 3] = max(goal[2, 3], self.floor)
        self.tool, self.moves = goal, self.moves + 1
        return {"success": True}

    def open(self, **_kw) -> dict:
        self.released = True
        return {"position": 1.0}

    def ctx(self) -> FakeContext:
        return FakeContext(
            {
                "robot.get_observation": self.observation,
                "robot.get_ee_pose": lambda **kw: {"pose": pose(self.tool)},
                "robot.solve_ik": lambda pose, **kw: {
                    "success": bool(self.reach(matrix(pose))),
                    "within_servo_tolerance": True,
                },
                "robot.go_to_pose": self.move,
                "robot.go_to_pose_cartesian": self.move,
                "robot.open_gripper": self.open,
            }
        )
