"""A fake arm with a wrist D405 over a rendered baseplate, for verifying-brick-seated-by-height.

The plate sits at PLATE_IN_WORLD; the brick under test is posed by _seat_render.brick_pose.
The wrist camera rides the tool at a fixed offset (tilted 30 deg, as on the YAM). A move
puts the tool where it is sent (or ``short_by`` metres short of it along the move, a servo
that stops early). Each observation renders the camera's view (cached per camera pose)
and adds fresh D405 noise.

``python -m tests.bundles._seat_eval`` sweeps placements and prints the SKILL.md table.
"""

from __future__ import annotations

import math

import numpy as np
from gap.testing import FakeContext

from . import _seat_render as SR
from ._plate_render import SYSTEMS

K = np.array([[394.0, 0.0, 320.0], [0.0, 394.0, 240.0], [0.0, 0.0, 1.0]])
PLATE_IN_WORLD = np.array([0.45, 0.0, 0.03])
PLATE = (12, 12)
CELL = (4, 5)


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
    def __init__(
        self,
        system="duplo",
        studs=(2, 2),
        brick=True,
        neighbours=None,
        layer=1,
        camera=True,
        camera_error=(0.0, 0.0, 0.0),
        ik_ok=True,
        refuse=None,
        short_by=0.0,
        seed=0,
        **placement,
    ):
        self.system, self.studs = system, tuple(studs)
        self.g = SYSTEMS[system]
        self.neighbours = list(neighbours or [])
        self.placement = dict(placement, layer=layer)
        self.has_brick = brick
        self.camera = camera
        self.camera_error = np.asarray(camera_error, float)
        self.ik_ok, self.refuse, self.short_by = ik_ok, refuse, short_by
        self.rng = np.random.default_rng(seed)
        self.p = np.array([0.35, 0.10, 0.25])
        self.r = np.diag([1.0, -1.0, -1.0])  # top-down
        self.moves: list[dict] = []
        self.ik_calls: list[dict] = []
        self._cache: dict[bytes, np.ndarray] = {}
        self.plate_world = np.eye(4)
        self.plate_world[:3, 3] = PLATE_IN_WORLD

    # The seat the skill is given: the seated brick's bottom centre, world frame.
    def seat(self, dx=0.0, dy=0.0):
        m = SR.brick_pose(self.system, *self.studs, *CELL, PLATE, layer=self.placement["layer"])
        w = self.plate_world @ m
        return dict(zip("xyz", map(float, w[:3, 3] + [dx, dy, 0.0]), strict=True))

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
            t_pc = np.linalg.inv(self.plate_world) @ cam_world
            brick = None
            if self.has_brick:
                pose = SR.brick_pose(self.system, *self.studs, *CELL, PLATE, **self.placement)
                brick = (pose, *self.studs)
            self._cache[key] = SR.render_depth(
                self.system, PLATE, K, t_pc, brick, self.neighbours, noise=False
            )
        return SR.add_noise(self._cache[key], K, self.rng)

    def observation(self, **kw):
        cams = []
        if self.camera:
            cam = self.camera_pose()
            reported = cam.copy()
            reported[:3, 3] += self.camera_error
            cams.append(
                {
                    "name": "left_eye_in_hand",
                    "pose": _pose(reported),
                    "intrinsics": K.tolist(),
                    "depth": self.depth(cam),
                    "rgb": np.zeros((480, 640, 3), np.uint8),
                }
            )
        return {"arms": [{"ee_pose": _pose(self.tool())}], "cameras": cams}

    def go_to_pose(self, pose, **kw):
        self.moves.append({"pose": pose, **kw})
        if self.refuse == "raise":
            raise RuntimeError("unreachable: joint limits")
        if self.refuse == "route":
            return {"outcome": "unreachable", "reached": False, "reason": "joint limits"}
        goal = np.array([pose["position"][k] for k in "xyz"])
        if self.short_by:
            d = goal - self.p
            n = float(np.linalg.norm(d)) or 1.0
            goal = goal - d / n * min(self.short_by, n)
        self.p, self.r = goal, _rot(pose["rotation"])
        if self.short_by:
            return {"outcome": "timeout", "reached": False}
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


CASES = [
    ("duplo", (2, 2), "seated", {}),
    ("duplo", (2, 2), "partly pressed, 0.4 stud", {"dz": 0.4 * 0.0045}),
    ("duplo", (2, 2), "resting on the stud tops", {"dz": 0.0045}),
    ("duplo", (2, 2), "one end on the studs (8 deg)", {"tilt_deg": 8.0}),
    ("duplo", (2, 2), "one cell off in x", {"dx": 0.016}),
    ("duplo", (2, 2), "half a cell off, on the studs", {"dx": 0.008, "dz": 0.0045}),
    ("duplo", (2, 2), "seated, camera 4 mm high", {"camera_error": (0.0, 0.0, 0.004)}),
    ("duplo", (2, 2), "seated, flush neighbours", {"neighbours": "ring"}),
    ("duplo", (2, 2), "no brick", {"brick": False}),
    ("lego", (2, 4), "seated", {}),
    ("lego", (2, 4), "partly pressed, 0.45 stud", {"dz": 0.45 * 0.0017}),
    ("lego", (2, 4), "resting on the stud tops", {"dz": 0.0017}),
    ("lego", (2, 4), "one end on the studs", {"tilt_deg": math.degrees(math.atan(0.0017 / 0.032))}),
    ("lego", (2, 4), "one cell off in y", {"dy": 0.008}),
    ("lego", (2, 2), "seated", {}),
    ("lego", (2, 2), "resting on the stud tops", {"dz": 0.0017}),
]


def ring_neighbours(studs):
    """Seated bricks flush against all four sides of the cell (DUPLO 2x2 neighbours)."""
    from ._plate_render import Brick

    x, y = CELL
    nx, ny = studs
    return [
        Brick(x - 2, y, 2, ny, colour=1),
        Brick(x + nx, y, 2, ny, colour=2),
        Brick(x, y - 2, nx, 2, colour=1),
        Brick(x, y + ny, nx, 2, colour=2),
    ]


def run_case(script, system, studs, kw, seed):
    kw = dict(kw)
    if kw.get("neighbours") == "ring":
        kw["neighbours"] = ring_neighbours(studs)
    w = World(system=system, studs=studs, seed=seed, **kw)
    return script.run(w.ctx(), seat_position=w.seat(), system=system, brick_studs=list(studs))


def main(trials=5):
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "skills/verifying-brick-seated-by-height/scripts/verify_seat.py"
    )
    spec = importlib.util.spec_from_file_location("verify_seat", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    print(f"| brick | truth | routes (of {trials}) | height error mm | tilt deg | coverage |")
    print("|---|---|---|---|---|---|")
    for system, studs, label, kw in CASES:
        outs = [run_case(script, system, studs, kw, seed) for seed in range(trials)]
        routes: dict[str, int] = {}
        for o in outs:
            routes[o["route"]] = routes.get(o["route"], 0) + 1
        dh = [o["height_error"] * 1e3 for o in outs]
        tl = [o["tilt_deg"] for o in outs]
        cv = [o["coverage"] for o in outs]
        print(
            f"| {system} {studs[0]}x{studs[1]} | {label} | "
            + ", ".join(f"{k} {v}" for k, v in sorted(routes.items()))
            + f" | {np.mean(dh):+.2f} (sd {np.std(dh):.2f}) | {np.mean(tl):.1f} | {np.mean(cv):.2f} |"
        )


if __name__ == "__main__":
    main()
