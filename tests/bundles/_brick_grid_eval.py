"""computing-brick-pose-from-stud-grid after pose6d: how often a rendered brick
is read back as its cell. Not collected by pytest; run from the repo root:

    .venv/bin/python tests/bundles/_brick_grid_eval.py --trials 50

A brick body (no studs) is placed at a random cell of a randomly yawed plate on
the table, rendered with pose6d's station depth noise (overhead Orbbec at
~0.9 m, or the wrist D405 at 0.15-0.22 m, oblique), its mask dilated a pixel,
registered by pose6d's primitive fast path with this skill's pose6d_mesh and
support_plane - "expected": with the cell's brick_pose (2 mm off) as init,
"blind": no init - and mapped back to a cell. "shifted": the true brick sits
half a pitch off its cell along the plate's x (between two), blind; it should
come back off_grid.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "pose6d"))
_impl = importlib.import_module("_impl")
_synth = importlib.import_module("_synth")
spec = importlib.util.spec_from_file_location(
    "brick_pose",
    ROOT / "skills" / "computing-brick-pose-from-stud-grid" / "scripts" / "brick_pose.py",
)
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)


class _Ctx:
    def tool(self, name, **_):
        raise KeyError(name)


def _pose(m):
    x, y, z, w = Rotation.from_matrix(m[:3, :3]).as_quat()
    return {
        "position": dict(zip("xyz", map(float, m[:3, 3]), strict=True)),
        "rotation": {"w": w, "x": x, "y": y, "z": z},
    }


def _m(p):
    m = np.eye(4)
    q = p["rotation"]
    m[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    m[:3, 3] = [p["position"][k] for k in "xyz"]
    return m


def trial(rng, system, camera, mode):
    ctx = _Ctx()
    plate = np.eye(4)
    plate[:3, :3] = Rotation.from_euler("z", rng.uniform(-180, 180), degrees=True).as_matrix()
    n = [16, 16] if system == "lego" else [10, 10]
    size = [[2, 4], [2, 2], [1, 4], [2, 3], [1, 2]][int(rng.integers(5))]
    rot = int(rng.choice([0, 90]))
    fp = size[::-1] if rot else size
    cell = {
        "x": int(rng.integers(0, n[0] - fp[0] + 1)),
        "y": int(rng.integers(0, n[1] - fp[1] + 1)),
        "layer": 1,
        "rotation": rot,
    }
    fwd = bp.run(ctx, _pose(plate), n, size, cell=cell, system=system)
    t_true = _m(fwd["brick_pose"])
    if mode == "shifted":
        t_true[:3, 3] += plate[:3, 0] * 0.5 * fwd["stud_pitch"]
    if camera == "wrist":
        az, el = rng.uniform(0, 2 * math.pi), math.radians(rng.uniform(50, 70))
        eye = t_true[:3, 3] + rng.uniform(0.15, 0.22) * np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
        )
        cam, k, noise = _synth.look_at(eye, t_true[:3, 3]), _synth.K_D405, _synth.D405
    else:
        cam, k, noise = (
            _synth.look_at([0.0, -0.30, 0.85], [0, 0, 0]),
            _synth.K_ORBBEC,
            _synth.ORBBEC,
        )
    depth, mask = _synth.render(
        fwd["pose6d_mesh"], t_true, cam, K=k, noise=noise, seed=int(rng.integers(1 << 30))
    )
    pts = _synth.backproject(depth, binary_dilation(mask), k) @ cam[:3, :3].T + cam[:3, 3]
    kw = {"viewpoint": cam[:3, 3], "up": [0, 0, 1], "support_plane": fwd["support_plane"]}
    if mode == "expected":
        init = _m(fwd["brick_pose"])
        init[:3, 3] += np.r_[rng.normal(scale=0.002, size=2), 0.0]
        kw["init"] = init
    r = _impl.register(pts, fwd["pose6d_mesh"], **kw)
    back = bp.run(ctx, _pose(plate), n, size, observed_pose=_pose(r["T"]), system=system)
    return {
        "route": back["route"],
        "same_cell": sorted(back["cells"]) == sorted(fwd["cells"]),
        "lateral_mm": 1e3 * back["residual"]["lateral"],
        "yaw_deg": abs(back["residual"]["yaw_deg"]),
        "points": len(pts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    print(
        "| system | camera | mode | pts | right cell & computed | off_grid | other | lateral residual med / p90 (mm) | yaw med / p90 (deg) |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for system in ("lego", "duplo"):
        for camera in ("wrist", "overhead"):
            for mode in ("expected", "blind", "shifted"):
                rows = [trial(rng, system, camera, mode) for _ in range(a.trials)]
                routes = Counter(r["route"] for r in rows)
                ok = sum(r["route"] == "computed" and r["same_cell"] for r in rows)
                lat = np.array([r["lateral_mm"] for r in rows])
                yaw = np.array([r["yaw_deg"] for r in rows])
                other = (
                    ", ".join(
                        f"{k} {v}" for k, v in routes.items() if k not in ("computed", "off_grid")
                    )
                    or "0"
                )
                wrong = sum(r["route"] == "computed" and not r["same_cell"] for r in rows)
                print(
                    f"| {system} | {camera} | {mode} | {int(np.median([r['points'] for r in rows]))} | "
                    f"{ok}/{a.trials}{f' ({wrong} wrong cell)' if wrong else ''} | {routes['off_grid']} | {other} | "
                    f"{np.median(lat):.2f} / {np.percentile(lat, 90):.2f} | {np.median(yaw):.1f} / {np.percentile(yaw, 90):.1f} |"
                )


if __name__ == "__main__":
    main()
