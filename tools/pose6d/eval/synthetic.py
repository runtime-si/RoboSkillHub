# SPDX-License-Identifier: MIT
"""Accuracy and runtime of pose6d on ray-cast depth renders with station noise.

    python tools/pose6d/eval/synthetic.py --trials 20 --out /tmp/pose6d_synth.json

Parts at known poses are rendered from the overhead camera (Orbbec Gemini,
~0.9 m, 2.9 mm std at 1 m growing with depth^2, 1 mm steps) and from the wrist
(RealSense D405, 0.15-0.22 m, oblique, 0.5 mm std at 0.15 m, 0.1 mm steps);
the object mask is dilated by one pixel so table returns bleed in at the edges
as they do from a real segmenter. Each cloud is registered three ways: the
primitive fast path, the fast path with the table as support plane (resting
parts only), and the Open3D FPFH + RANSAC + ICP global path. Rotation error is
taken over the part's symmetries (a cylinder's: axis angle).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _impl  # noqa: E402
import _synth  # noqa: E402

PARTS = {
    "cube_30": ({"box": [0.03, 0.03, 0.03]}, "flat"),
    "brick_2x4": ({"box": [0.0318, 0.0158, 0.0096]}, "flat"),
    "domino_flat": ({"box": [0.048, 0.024, 0.0075]}, "flat"),
    "domino_upright": ({"box": [0.048, 0.024, 0.0075]}, "on_end"),
    "peg_lying": ({"cylinder": [0.005, 0.04]}, "lying"),
    "peg_standing": ({"cylinder": [0.005, 0.04]}, "flat"),
}


def _rot(axis: str, ang: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler(axis, ang).as_matrix()


def resting_pose(spec: dict, rest: str, rng) -> np.ndarray:
    """A stable pose on the table (z = 0) with random yaw and position."""
    R = {"flat": np.eye(3), "on_end": _rot("y", math.pi / 2), "lying": _rot("x", math.pi / 2)}[rest]
    R = _rot("z", rng.uniform(-math.pi, math.pi)) @ R
    if "box" in spec:
        half_up = float(np.abs(R[2]) @ (np.asarray(spec["box"]) / 2))
    else:
        r, h = spec["cylinder"]
        half_up = h / 2 if rest == "flat" else r
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), half_up]
    return T


def free_pose(rng) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=int(rng.integers(1 << 31))).as_matrix()
    T[:3, 3] = [rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), 0.05]
    return T


def camera(kind: str, target: np.ndarray, rng):
    if kind == "overhead":
        return _synth.look_at([0.0, -0.30, 0.85], [0.0, 0.0, 0.0]), _synth.K_ORBBEC, _synth.ORBBEC
    az = rng.uniform(0, 2 * math.pi)
    el = math.radians(rng.uniform(50, 70))
    rng_m = rng.uniform(0.15, 0.22)
    eye = target + rng_m * np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    )
    return _synth.look_at(eye, target), _synth.K_D405, _synth.D405


def rotation_error(spec: dict, R_est: np.ndarray, R_gt: np.ndarray) -> float:
    if "cylinder" in spec:
        c = abs(float(R_est[:, 2] @ R_gt[:, 2]))
        return math.degrees(math.acos(min(1.0, c)))
    group, _ = _impl.symmetry_group(_impl.parse_model(spec), None)
    return min(_impl.rotation_angle_deg(R_est @ S, R_gt) for S in group)


def run(trials: int, seed: int, modes: list[str]) -> list[dict]:
    from scipy.ndimage import binary_dilation

    rng = np.random.default_rng(seed)
    rows = []
    cases = [(p, c, "resting") for p in PARTS for c in ("overhead", "wrist")]
    cases += [(p, "wrist", "free") for p in PARTS if PARTS[p][1] in ("flat", "lying")]
    for part, cam_kind, pose_kind in cases:
        spec, rest = PARTS[part]
        if pose_kind == "free" and part in ("peg_standing",):
            continue
        for _ in range(trials):
            T = resting_pose(spec, rest, rng) if pose_kind == "resting" else free_pose(rng)
            cam, K, noise = camera(cam_kind, T[:3, 3], rng)
            depth, mask = _synth.render(
                spec, T, cam, K=K, noise=noise, seed=int(rng.integers(1 << 31))
            )
            mask = binary_dilation(mask)
            P = _synth.backproject(depth, mask, K) @ cam[:3, :3].T + cam[:3, 3]
            for mode in modes:
                if mode == "support" and pose_kind != "resting":
                    continue
                kw = {"viewpoint": cam[:3, 3], "up": [0.0, 0.0, 1.0]}
                if mode == "support":
                    kw["support_plane"] = [0.0, 0.0, 1.0, 0.0]
                kw["method"] = "global" if mode == "global" else "primitive"
                r = _impl.register(P, spec, **kw)
                rows.append(
                    {
                        "part": part,
                        "camera": cam_kind,
                        "pose": pose_kind,
                        "mode": mode,
                        "n_points": int(len(P)),
                        "rot_err_deg": rotation_error(spec, r["T"][:3, :3], T[:3, :3]),
                        "trans_err_mm": 1000 * float(np.linalg.norm(r["T"][:3, 3] - T[:3, 3])),
                        "fitness": r["fitness"],
                        "coverage": r["coverage"],
                        "rmse_mm": 1000 * r["rmse"],
                        "status": r["status"],
                        "runtime_ms": 1000 * r["runtime_s"],
                    }
                )
    return rows


def summarize(rows: list[dict]) -> str:
    keys = sorted(
        {(r["part"], r["camera"], r["pose"], r["mode"]) for r in rows},
        key=lambda k: (k[1], k[2], k[0], k[3]),
    )
    out = [
        "| camera | pose | part | mode | pts | rot med / p90 (deg) | trans med / p90 (mm) | ok (rot<5, t<3mm) | status=registered | runtime med (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in keys:
        sel = [r for r in rows if (r["part"], r["camera"], r["pose"], r["mode"]) == key]
        rot = np.array([r["rot_err_deg"] for r in sel])
        tr = np.array([r["trans_err_mm"] for r in sel])
        ok = np.mean((rot < 5) & (tr < 3))
        reg = np.mean([r["status"] == "registered" for r in sel])
        out.append(
            f"| {key[1]} | {key[2]} | {key[0]} | {key[3]} | {int(np.median([r['n_points'] for r in sel]))} | "
            f"{np.median(rot):.2f} / {np.percentile(rot, 90):.2f} | {np.median(tr):.2f} / {np.percentile(tr, 90):.2f} | "
            f"{ok:.0%} | {reg:.0%} | {np.median([r['runtime_ms'] for r in sel]):.0f} |"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", default="fast,support,global")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = run(a.trials, a.seed, a.modes.split(","))
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
    print(summarize(rows))


if __name__ == "__main__":
    main()
