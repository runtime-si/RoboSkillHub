"""Measure verifying-piece-flushness-from-wrist-depth on rendered scenes with real D405 noise.

Noise source: wrist D405 frames of a real YAM station (``.npz`` with ``depth_m`` and ``K``; not
committed). Every 256 x 256 px window with at least 95 % valid depth at 8-16 cm that is one
plane (wood table: residual under 4 mm) gives a noise tile: the depth minus its fitted plane
(inverse depth linear in the pixel coordinates),
holes kept. A frame's noise is a mosaic of random tiles, each turned by a random quarter, with
one tile centred on the judged piece (it and its ring then see one continuous real surface;
independent tiles meet with steps that real depth does not have).
``--holey`` takes windows with 40-75 % valid depth instead (the black breadboard).

Each trial renders a 3 x 3 patch of a generated cut on a table (tests/bundles/_flush_scene.py)
under a camera at the window's mean depth, tilted 0-15 degrees, with the centre piece flush,
raised, tilted (one side up), stopped short of a neighbour (open seam) or with its neighbours
raised (sunk); adds the window's noise and holes; and asks the skill, with the neighbours'
outlines, with a ring and the table's height, and with a ring alone.

    python tests/bundles/_flush_eval.py --frames 'DIR/**/*.npz' [--trials 30]
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from tests.bundles import _flush_scene as S  # noqa: E402

SCRIPT = (
    HERE.parents[2]
    / "skills"
    / "verifying-piece-flushness-from-wrist-depth"
    / "scripts"
    / "verify_flush.py"
)
WIN = 256
OPPOSITE = {"west": "east", "east": "west", "south": "north", "north": "south"}
SHIFT = {"west": (1, 0), "east": (-1, 0), "south": (0, 1), "north": (0, -1)}


def _load():
    spec = importlib.util.spec_from_file_location("verify_flush", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def noise_windows(pattern, holey=False, limit=400):
    out = []
    for f in sorted(glob.glob(pattern, recursive=True)):
        d = np.load(f)
        if "depth_m" not in d.files or "K" not in d.files or d["depth_m"].shape != (480, 640):
            continue
        z = d["depth_m"].astype(np.float64)
        k = d["K"]
        for v0 in range(0, 480 - WIN + 1, 56):
            for u0 in range(0, 640 - WIN + 1, 64):
                p = z[v0 : v0 + WIN, u0 : u0 + WIN]
                valid = (p > 0.08) & (p < 0.16)
                frac = valid.mean()
                if holey and not 0.4 <= frac <= 0.75:
                    continue
                if not holey and frac < 0.95:
                    continue
                vv, uu = np.mgrid[v0 : v0 + WIN, u0 : u0 + WIN]
                # A plane's inverse depth is linear in the pixel coordinates.
                a = np.column_stack([uu[valid], vv[valid], np.ones(valid.sum())])
                coef, *_ = np.linalg.lstsq(a, 1.0 / p[valid], rcond=None)
                res = np.full(p.shape, np.nan)
                res[valid] = p[valid] - 1.0 / (a @ coef)
                if np.nanmax(np.abs(res)) > 0.004:
                    continue
                out.append((res, float(np.mean(p[valid])), k))
                if len(out) >= limit:
                    return out
    return out


def trial_scene(rng):
    sides = ["west", "south", "east", "north"]
    n = int(rng.integers(1, 5))
    nb = tuple(sorted(rng.choice(sides, n, replace=False).tolist()))
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--holey", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    mod = _load()
    wins = noise_windows(args.frames, holey=args.holey)
    print(f"{len(wins)} noise windows", file=sys.stderr)
    rng = np.random.default_rng(args.seed)
    cases = [("flush", 0.0)]
    cases += [("raised", m) for m in (0.0003, 0.0005, 0.0008, 0.001, 0.002)]
    cases += [("tilted", m) for m in (0.0003, 0.0006, 0.001, 0.002)]
    cases += [("seam", m) for m in (0.0005, 0.001, 0.0015, 0.002)]
    cases += [("sunk", 0.001)]
    want = {"flush": "flush", "raised": "raised", "tilted": "tilted", "seam": "gap", "sunk": "gap"}
    tally = defaultdict(lambda: defaultdict(int))
    level_err, rise_flush, times = defaultdict(list), [], []
    for t in range(args.trials):
        _, zmean, k = wins[int(rng.integers(len(wins)))]
        pose = S.look_down(
            height=zmean, tilt_deg=float(rng.uniform(0, 15)), yaw_deg=float(rng.uniform(0, 360))
        )
        seed = int(rng.integers(1_000_000))
        for kind, mag in cases:
            nb = trial_scene(rng)
            if kind == "seam":
                free = [d for d in nb if OPPOSITE[d] not in nb]
                if not free:
                    nb = ("west",)
                    free = ["west"]
                d = free[int(rng.integers(len(free)))]
                kw = {"shift_m": (SHIFT[d][0] * mag, SHIFT[d][1] * mag)}
            elif kind == "raised":
                kw = {"raise_m": mag}
            elif kind == "tilted":
                a = rng.uniform(0, 2 * np.pi)
                kw = {"tilt_m": mag, "tilt_dir": (np.cos(a), np.sin(a))}
            elif kind == "sunk":
                kw = {"neighbour_raise_m": mag}
            else:
                kw = {}
            scene = S.Scene(
                seed=seed, neighbours=nb, style="classic" if rng.random() < 0.5 else "wavy"
            )
            depth = S.render(scene, k, pose, (480, 640), **kw)
            noise = np.full(depth.shape, np.nan)
            cv0, cu0 = (480 - WIN) // 2, (640 - WIN) // 2
            for v0 in range(cv0 - WIN, 480, WIN):
                for u0 in range(cu0 - 2 * WIN, 640, WIN):
                    tile = np.rot90(wins[int(rng.integers(len(wins)))][0], int(rng.integers(4)))
                    a0, b0 = max(0, v0), max(0, u0)
                    a1, b1 = min(480, v0 + WIN), min(640, u0 + WIN)
                    if a1 > a0 and b1 > b0:
                        noise[a0:a1, b0:b1] = tile[a0 - v0 : a1 - v0, b0 - u0 : b1 - u0]
            hole = np.isnan(noise)
            depth = depth + np.nan_to_num(noise, nan=0.0)
            depth[hole] = 0.0
            cam = S.camera(depth, k, pose)
            modes = {
                "neighbours": {"neighbour_outlines": scene.neighbours},
                "ring+table_z": {"table_z": 0.0},
                "ring": {},
            }
            for mode, extra in modes.items():
                t0 = time.perf_counter()
                out = mod.run(None, cam, piece_outline=scene.piece, **extra)
                times.append(time.perf_counter() - t0)
                key = f"{kind} {mag * 1e3:.1f}mm"
                tally[mode, key][out["route"]] += 1
                if mode == "neighbours" and "level_m" in out:
                    if kind == "raised":
                        level_err[key].append(out["level_m"] - mag)
                    if kind == "flush":
                        level_err[key].append(out["level_m"])
                        rise_flush.append(out["tilt_rise_m"])
        print(f"trial {t + 1}/{args.trials}", file=sys.stderr)
    summary = {
        "windows": len(wins),
        "trials": args.trials,
        "holey": args.holey,
        "time_median_s": float(np.median(times)),
        "time_max_s": float(np.max(times)),
        "flush_rise_mm": [float(np.median(rise_flush) * 1e3), float(np.max(rise_flush) * 1e3)]
        if rise_flush
        else None,
        "level_error_mm": {
            k: [round(float(np.median(v)) * 1e3, 3), round(float(np.max(np.abs(v))) * 1e3, 3)]
            for k, v in level_err.items()
        },
        "right": {},
        "routes": {},
    }
    for (mode, key), routes in sorted(tally.items()):
        kind = key.split()[0]
        n = sum(routes.values())
        summary["right"].setdefault(mode, {})[key] = f"{routes.get(want[kind], 0)}/{n}"
        summary["routes"].setdefault(mode, {})[key] = dict(routes)
    print(json.dumps(summary, indent=1))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
