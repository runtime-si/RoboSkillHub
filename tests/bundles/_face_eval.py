"""A fake arm with a wrist D405 over loose bricks on a table, for
classifying-brick-resting-face-from-depth.

The table frame sits at TABLE_IN_WORLD; bricks are posed by _face_render.resting_pose. The
wrist camera rides the tool tilted 30 deg (as on the YAM). Moves land where commanded.
Each observation renders the camera's view (cached per pose) with fresh D405 noise.

``python -m tests.bundles._face_eval`` sweeps faces and yaws and prints the SKILL.md table.
"""

from __future__ import annotations

import math

import numpy as np
from gap.testing import FakeContext

from . import _face_render as FR

K = np.array([[394.0, 0.0, 320.0], [0.0, 394.0, 240.0], [0.0, 0.0, 1.0]])
TABLE_IN_WORLD = np.array([0.45, 0.05, 0.02])


def _rot(q):
    w, x, y, z = (float(q[k]) for k in "wxyz")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat(r):
    t = float(np.trace(r))
    if t > 0:
        s = 2 * math.sqrt(t + 1)
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2 * math.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2])
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = 2 * math.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2])
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = 2 * math.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1])
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    return dict(zip("wxyz", map(float, q), strict=True))


def _pose(m):
    return {
        "position": dict(zip("xyz", map(float, m[:3, 3]), strict=True)),
        "rotation": _quat(m[:3, :3]),
    }


def _axis_angle(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * k @ k


class World:
    """One brick (system, studs, face, yaw) at the table origin, plus optional others
    given as (system, nx, ny, face, x, y, yaw) in the table frame."""

    def __init__(
        self,
        system="duplo",
        studs=(2, 2),
        face="studs_up",
        yaw_deg=0.0,
        brick=True,
        others=(),
        brick_height=None,
        camera=True,
        ik_ok=True,
        refuse=None,
        seed=0,
        boxes=(),
    ):
        self.system, self.studs, self.face, self.yaw = system, tuple(studs), face, yaw_deg
        self.boxes = list(boxes)
        self.brick_height = brick_height
        self.bricks = []
        if brick:
            pose = FR.resting_pose(system, *studs, face, 0.0, 0.0, yaw_deg, bh=brick_height)
            self.bricks.append((system, *studs, pose, brick_height))
        for s, nx, ny, f, x, y, yaw in others:
            self.bricks.append((s, nx, ny, FR.resting_pose(s, nx, ny, f, x, y, yaw), None))
        self.camera, self.ik_ok, self.refuse = camera, ik_ok, refuse
        self.rng = np.random.default_rng(seed)
        self.p = np.array([0.35, 0.15, 0.25])
        self.r = np.diag([1.0, -1.0, -1.0])
        self.moves: list[dict] = []
        self.ik_calls: list[dict] = []
        self._cache: dict[bytes, np.ndarray] = {}
        self.table = np.eye(4)
        self.table[:3, 3] = TABLE_IN_WORLD

    def position(self, dx=0.0, dy=0.0):
        """Where perception roughly puts the brick: its xy (plus an error), the table's z."""
        return dict(zip("xyz", map(float, TABLE_IN_WORLD + [dx, dy, 0.0]), strict=True))

    def brick_pose_world(self):
        return self.table @ self.bricks[0][3]

    def tool(self):
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = self.r, self.p
        return m

    def camera_pose(self):
        tc = np.eye(4)
        tc[:3, :3] = _axis_angle([1, 0, 0], math.radians(-30))
        tc[:3, 3] = [0.0, -0.06, -0.07]
        return self.tool() @ tc

    def depth(self, cam_world):
        key = np.round(cam_world, 6).tobytes()
        if key not in self._cache:
            t_tc = np.linalg.inv(self.table) @ cam_world
            self._cache[key] = FR.render_depth(K, t_tc, self.bricks, boxes=self.boxes)
        return FR.add_noise(self._cache[key], K, self.rng)

    def observation(self, **kw):
        cams = []
        if self.camera:
            cam = self.camera_pose()
            cams.append(
                {
                    "name": "left_eye_in_hand",
                    "pose": _pose(cam),
                    "intrinsics": K.tolist(),
                    "depth": self.depth(cam),
                }
            )
        return {"arms": [{"ee_pose": _pose(self.tool())}], "cameras": cams}

    def go_to_pose(self, pose, **kw):
        self.moves.append({"pose": pose, **kw})
        if self.refuse == "raise":
            raise RuntimeError("unreachable: joint limits")
        if self.refuse == "route":
            return {"outcome": "unreachable", "reached": False, "reason": "joint limits"}
        self.p = np.array([pose["position"][k] for k in "xyz"])
        self.r = _rot(pose["rotation"])
        return {"outcome": "reached", "reached": True}

    def solve_ik(self, pose, **kw):
        self.ik_calls.append({"pose": pose, **kw})
        return {"success": self.ik_ok, "within_servo_tolerance": self.ik_ok}

    def ctx(self):
        return FakeContext(
            {
                "robot.get_observation": self.observation,
                "robot.go_to_pose": self.go_to_pose,
                "robot.solve_ik": self.solve_ik,
            }
        )


EXPECTED = {
    "studs_up": "studs_up",
    "upside_down": "upside_down",
    "side_x": "on_side",
    "side_y": "on_side",
}

CASES = [
    ("duplo", (2, 2), ("studs_up", "upside_down", "side_x")),
    ("duplo", (2, 4), ("studs_up", "upside_down", "side_x", "side_y")),
    ("lego", (2, 4), ("studs_up", "upside_down", "side_x", "side_y")),
    ("lego", (2, 2), ("studs_up", "upside_down", "side_x")),
    ("lego", (1, 4), ("studs_up", "upside_down", "side_x", "side_y")),
    ("lego", (1, 2), ("studs_up", "upside_down", "side_x", "side_y")),
]


def true_up(world):
    """The brick's +z (stud direction) in the world."""
    return world.brick_pose_world()[:3, 2]


def main(yaws=(0.0, 17.0, 33.0, 62.0, 105.0)):
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "skills/classifying-brick-resting-face-from-depth/scripts/classify_face.py"
    )
    spec = importlib.util.spec_from_file_location("classify_face", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    print(
        f"| brick | face | routes (of {len(yaws)} yaws) | vertical axis right | studs side right | yaw err deg (max) |"
    )
    print("|---|---|---|---|---|---|")
    for system, studs, fcs in CASES:
        for face in fcs:
            routes: dict[str, int] = {}
            axis_ok = side_ok = side_n = 0
            yaw_err = []
            for i, yaw in enumerate(yaws):
                w = World(system=system, studs=studs, face=face, yaw_deg=yaw, seed=i)
                out = script.run(
                    w.ctx(),
                    brick_position=w.position(0.004, -0.003),
                    system=system,
                    brick_studs=list(studs),
                )
                routes[out["route"]] = routes.get(out["route"], 0) + 1
                want_axis = {"side_x": "x", "side_y": "y"}.get(face, "z")
                if studs[0] == studs[1] and want_axis == "y":
                    want_axis = "x"
                axis_ok += out["vertical_axis"] == want_axis
                up = np.array([out["up_vector"][k] for k in "xyz"])
                if out["route"] == EXPECTED[face]:
                    if face.startswith("side") and out["stud_direction_known"]:
                        side_n += 1
                        side_ok += float(up @ true_up(w)) > 0.9
                    pose = w.brick_pose_world()
                    ref = pose[:3, 0] if want_axis in ("z", "y") else pose[:3, 1]
                    if face.startswith("side") and want_axis == "y":
                        ref = pose[:3, 0]
                    period = 90.0 if (want_axis == "z" and studs[0] == studs[1]) else 180.0
                    tr = math.degrees(math.atan2(ref[1], ref[0]))
                    yaw_err.append(abs((out["yaw_deg"] - tr + period / 2) % period - period / 2))
            side = f"{side_ok}/{side_n}" if face.startswith("side") else "-"
            print(
                f"| {system} {studs[0]}x{studs[1]} | {face} | "
                + ", ".join(f"{k} {v}" for k, v in sorted(routes.items()))
                + f" | {axis_ok}/{len(yaws)} | {side} | "
                + (f"{max(yaw_err):.1f}" if yaw_err else "-")
                + " |"
            )


if __name__ == "__main__":
    main()
