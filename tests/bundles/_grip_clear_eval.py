"""Measure selecting-a-grip-clear-of-placed-neighbours over whole-board fill orders.

A generated cut (tests/bundles/jigsaw_cut.py, classic or random-cut, 22.5 mm cells) is filled
in an order; at every placement the skill is asked for a grip on the next piece with the
pieces already placed round it (within 1.5 cells) given as exact outlines, and again as
occupied cells only. Every grip it returns from the cells is checked against the exact
outlines (fingertip footprints plus release travel against the true neighbours).

Orders: ``rows`` (row by row from a corner: each piece has its west and south neighbours),
``frame`` (the border first, then rows), ``random``.

    python tests/bundles/_grip_clear_eval.py [--size 10] [--orders rows,frame,random]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from tests.bundles.jigsaw_cut import make_puzzle  # noqa: E402

SCRIPT = (
    HERE.parents[2]
    / "skills"
    / "selecting-a-grip-clear-of-placed-neighbours"
    / "scripts"
    / "select_grip.py"
)
CELL = 0.0225


def _load():
    spec = importlib.util.spec_from_file_location("select_grip", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def order_of(kind, cols, rows, rng):
    cells = [(i, j) for j in range(rows) for i in range(cols)]
    if kind == "rows":
        return cells
    if kind == "frame":
        border = [c for c in cells if c[0] in (0, cols - 1) or c[1] in (0, rows - 1)]
        # Round the frame, then the interior row by row.
        ring = sorted(border, key=lambda c: math.atan2(c[1] - rows / 2, c[0] - cols / 2))
        return ring + [c for c in cells if c not in border]
    return [cells[k] for k in rng.permutation(len(cells))]


def true_clearance(mod, grip, neighbours, params):
    """Clearance of a returned grip's fingertips against the exact neighbour outlines."""
    if not neighbours:
        return math.inf
    u = np.array(
        [
            math.cos(math.radians(grip["closing_yaw_deg"])),
            math.sin(math.radians(grip["closing_yaw_deg"])),
        ]
    )
    v = np.array([-u[1], u[0]])
    a, b = (np.asarray(c) for c in grip["contacts"])
    fw, ft, tr = params
    su = np.arange(0.0, ft + tr + 1e-9, 0.0002)
    sv = np.arange(-0.5 * fw, 0.5 * fw + 1e-9, 0.0002)
    gu, gv = (x.ravel() for x in np.meshgrid(su, sv))
    pts = np.vstack([a - np.outer(gu, u) + np.outer(gv, v), b + np.outer(gu, u) + np.outer(gv, v)])
    allp = np.vstack(neighbours)
    lo = np.minimum(pts.min(axis=0), allp.min(axis=0)) - 0.01
    raster = mod._Raster.__new__(mod._Raster)
    hi = np.maximum(pts.max(axis=0), allp.max(axis=0)) + 0.01
    raster.lo = lo
    raster.n = np.ceil((hi - lo) / mod.RES_M).astype(int) + 1
    obs = raster._fill(neighbours)
    import cv2

    raster.dist = cv2.distanceTransform((obs == 0).astype(np.uint8), cv2.DIST_L2, 5) * mod.RES_M
    return raster.clearance(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--orders", default="rows,frame,random")
    ap.add_argument("--styles", default="classic,wavy")
    ap.add_argument("--finger", default="0.008,0.004", help="width,thickness in m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cells-only", action="store_true", help="skip the exact-outline runs")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    mod = _load()
    fw, ft = (float(x) for x in args.finger.split(","))
    params = (fw, ft, mod.RELEASE_TRAVEL_M)
    rng = np.random.default_rng(args.seed)
    summary = {"size": args.size, "finger_m": [fw, ft], "results": {}}
    for style in args.styles.split(","):
        pz = make_puzzle(args.size, args.size, cell=(CELL, CELL), seed=args.seed, style=style)
        for kind in args.orders.split(","):
            order = order_of(kind, args.size, args.size, rng)
            placed: set = set()
            by_n = defaultdict(Counter)
            cells_routes = Counter()
            exact_routes = Counter()
            unsafe, cells_found, times, clear_mm = 0, 0, [], []
            for i, j in order:
                near = [(a, b) for (a, b) in placed if max(abs(a - i), abs(b - j)) <= 1]
                n_sides = sum(
                    (i + di, j + dj) in placed for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                )
                outlines = [pz.outline(a, b) for a, b in near]
                t0 = time.perf_counter()
                out = (
                    {"route": "skipped"}
                    if args.cells_only
                    else mod.run(
                        None,
                        target_outline=pz.outline(i, j),
                        neighbour_outlines=outlines,
                        finger_width=fw,
                        finger_thickness=ft,
                    )
                )
                times.append(time.perf_counter() - t0)
                exact_routes[out["route"]] += 1
                by_n[n_sides][out["route"]] += 1
                if out["route"] == "grip_found" and math.isfinite(out["clearance_m"]):
                    clear_mm.append(out["clearance_m"] * 1e3)
                oc = mod.run(
                    None,
                    target_outline=pz.outline(i, j),
                    occupied_cells=[list(c) for c in near],
                    cell_size=(CELL, CELL),
                    finger_width=fw,
                    finger_thickness=ft,
                )
                cells_routes[oc["route"]] += 1
                if oc["route"] in ("grip_found", "only_offset_place"):
                    cells_found += 1
                    nb = [pz.outline(a, b) for a, b in near]
                    if true_clearance(mod, oc, nb, params) < mod.MIN_CLEARANCE_M - 1e-9:
                        unsafe += 1
                placed.add((i, j))
            summary["results"][f"{style}/{kind}"] = {
                "placements": len(order),
                "exact_outlines": dict(exact_routes),
                "by_placed_sides": {str(k): dict(v) for k, v in sorted(by_n.items())},
                "cells_only": dict(cells_routes),
                "cells_grips_unsafe_against_true_outlines": f"{unsafe}/{cells_found}",
                "clearance_mm_median": float(np.median(clear_mm)) if clear_mm else None,
                "time_median_s": float(np.median(times)),
                "time_max_s": float(np.max(times)),
            }
            print(style, kind, json.dumps(summary["results"][f"{style}/{kind}"]), file=sys.stderr)
    print(json.dumps(summary, indent=1))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
