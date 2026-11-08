"""Sweep of planning-a-flat-domino-mosaic-from-an-image over synthetic images (CPU).

    .venv/bin/python -m tests.bundles._domino_mosaic_eval [--sets 1 2 4 8 16] [--solver highs]
        [--cost upstream squared] [--image path.jpg]

Prints per image, set count, solver and cost: the grid, the solver status and solve time, the cell
RMSE in pips (and the per-cell rounding floor, what an unlimited supply of every tile would reach),
the pixel RMSE of the mosaic's shading against the image (0..1 grey), the correlation of the placed
pips with the target, and the order checks (steps reaching across a placed tile, steps with no
finger room). The numbers in the skill's SKILL.md come from this script; CBC needs PuLP on the path.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np
from gap.testing import FakeContext

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "skills/planning-a-flat-domino-mosaic-from-an-image/scripts/plan_mosaic.py"


def load():
    spec = importlib.util.spec_from_file_location("plan_mosaic", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic(name: str, h: int = 480, w: int = 400) -> np.ndarray:
    """Grey test pictures, 0..255, a little noise so no two fields tie exactly."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    if name == "blob":
        img = 255 * np.exp(-((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (2 * (0.25 * w) ** 2))
    elif name == "gradient":
        img = 255 * (xx / w * 0.7 + yy / h * 0.3)
    elif name == "rings":
        r = np.hypot(xx - w / 2, yy - h / 2)
        img = 127.5 * (1 + np.cos(r / (0.08 * w) * np.pi))
    elif name == "letter":
        glyph = np.full((h, w), 230, dtype=np.uint8)
        org = (int(0.12 * w), int(0.85 * h))
        cv2.putText(glyph, "R", org, cv2.FONT_HERSHEY_SIMPLEX, w / 45, 20, w // 10)
        img = glyph.astype(np.float64)
    elif name == "checker":
        img = 255.0 * (((xx // (w / 4)) + (yy // (h / 4))) % 2)
    else:
        raise ValueError(name)
    rng = np.random.default_rng(0)
    return np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--images", nargs="*", default=["blob", "gradient", "rings", "letter"])
    ap.add_argument("--image", action="append", default=[], help="an image file to add")
    ap.add_argument("--solver", nargs="+", default=["auto"])
    ap.add_argument("--cost", nargs="+", default=["upstream"])
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--preview-dir", default="")
    ap.add_argument("--max-pips", type=int, default=6)
    ap.add_argument("--contrast", default="dark_on_light")
    ap.add_argument(
        "--fill",
        type=float,
        default=1.0,
        help="under 1: a grid of about this share of the sets' tiles (spare tiles to choose from)",
    )
    args = ap.parse_args()
    mod = load()
    cases = [(n, {"image": synthetic(n)}) for n in args.images]
    cases += [(Path(p).stem, {"image_path": p}) for p in args.image]
    print(
        "image     sets grid    solver cost      status   solve_s  total_s cell_rmse floor "
        "img_rmse corr  reach_over no_finger_room"
    )
    for name, src in cases:
        for sets in args.sets:
            for solver in args.solver:
                for cost in args.cost:
                    t0 = time.perf_counter()
                    preview = (
                        str(Path(args.preview_dir) / f"{name}_{sets}_{solver}_{cost}.png")
                        if args.preview_dir
                        else None
                    )
                    grid = {}
                    if args.fill < 1:
                        shape = (
                            src["image"].shape
                            if "image" in src
                            else cv2.imread(src["image_path"], cv2.IMREAD_GRAYSCALE).shape
                        )
                        per_set = (args.max_pips + 1) * (args.max_pips + 2) // 2
                        fields = 2 * round(args.fill * per_set * sets)
                        r, c = mod.mm.grid_for_sets(fields, shape[1] / shape[0])
                        grid = {"rows": r, "cols": c}
                    out = mod.run(
                        FakeContext(),
                        sets=sets,
                        max_pips=args.max_pips,
                        pip_contrast=args.contrast,
                        **grid,
                        grid_origin=[0.30, 0.20, 0.0],
                        solver=solver,
                        cost=cost,
                        time_limit_s=args.time_limit,
                        preview_path=preview,
                        **src,
                    )
                    total = time.perf_counter() - t0
                    print(
                        f"{name:9s} {sets:4d} {out['rows']:3d}x{out['cols']:<3d} "
                        f"{out['solver']:6s} {cost:9s} {out['solver_status'] or out['route']:8s} "
                        f"{out['solve_time_s']:7.2f} {total:8.2f} {out['cell_rmse']:9.3f} "
                        f"{out['floor_rmse']:5.3f} {out['image_rmse']:8.3f} "
                        f"{out['correlation']:5.3f} {out['reach_over']:10d} "
                        f"{out['steps_without_finger_room']:4d}/{out['n_tiles']}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
