"""Measure classifying-piece-sides-for-border-first on generated pieces and on real piece photos.

Generated: every piece of 10 x 10 cuts (tests/bundles/jigsaw_cut.py, classic and random-cut), drawn as a mask at
``--px`` pixels per cell, turned at random, edges jittered by up to 1 px.

Real: PuzzleMap (https://huggingface.co/datasets/pablo-moreira/puzzle-map, CC-BY-4.0; not committed): single-piece
photos with per-side labels (OUTER = tab, INNER = blank, SMOOTH = flat). The piece's mask is cut out of its
box by GrabCut (no learned model); many photos have fingers over the piece's outline. Side sequences are
compared up to a quarter turn (the labels name sides in the photo's frame after the piece's annotated
rotation; the classifier names them from its own rectangle).

    python tests/bundles/_classify_sides_eval.py [--puzzlemap DIR] [--list sample_pieces.txt]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import time
from collections import Counter

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from tests.bundles.jigsaw_cut import make_puzzle  # noqa: E402

SCRIPT = (
    HERE.parents[2]
    / "skills"
    / "classifying-piece-sides-for-border-first"
    / "scripts"
    / "classify_sides.py"
)
LABEL = {"OUTER": "tab", "INNER": "blank", "SMOOTH": "flat"}
CLASS = {0: "interior", 1: "edge", 2: "corner"}


def _load():
    spec = importlib.util.spec_from_file_location("classify_sides", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def best_shift(got, want):
    """Most sides agreeing over the 4 cyclic shifts; returns (n_agree, all_agree)."""
    best = max(sum(g == w for g, w in zip(got[k:] + got[:k], want, strict=True)) for k in range(4))
    return best, best == 4


def mask_of(poly, px_per_cell, angle, rng, jitter=1.0):
    c = poly.mean(axis=0)
    a = np.radians(angle)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    p = (poly - c) @ r.T * px_per_cell
    p += rng.uniform(-jitter, jitter, p.shape)
    size = int(2.2 * px_per_cell)
    img = np.zeros((size, size), np.uint8)
    pts = np.column_stack([p[:, 0], -p[:, 1]]) + size / 2
    cv2.fillPoly(img, [np.round(pts).astype(np.int32)], 1)
    return img


def eval_generated(mod, px, rng):
    out = {}
    for style in ("classic", "wavy"):
        conf, sides, seqs, unc = Counter(), [0, 0], [0, 0], 0
        for seed in range(3):
            pz = make_puzzle(10, 10, cell=(1.0, 1.0), seed=seed, style=style)
            for i in range(10):
                for j in range(10):
                    want_types = pz.side_types(i, j)  # south, east, north, west (counter-clockwise)
                    m = mask_of(pz.outline(i, j), px, rng.uniform(0, 360), rng)
                    try:
                        info = mod.classify_mask(m)
                    except ValueError:
                        conf[(pz.piece_class(i, j), "error")] += 1
                        continue
                    conf[(pz.piece_class(i, j), info["piece_class"])] += 1
                    n, ok = best_shift(info["side_types"], list(want_types))
                    sides[0] += n
                    sides[1] += 4
                    seqs[0] += ok
                    seqs[1] += 1
                    unc += info["uncertain_sides"] > 0
        right = sum(v for (t, g), v in conf.items() if t == g)
        out[style] = {
            "pieces": sum(conf.values()),
            "class_right": right,
            "sides_right": f"{sides[0]}/{sides[1]}",
            "sequence_right": f"{seqs[0]}/{seqs[1]}",
            "with_uncertain_side": unc,
            "confusion": {f"{t}->{g}": v for (t, g), v in sorted(conf.items())},
        }
    return out


def grabcut_mask(img, bbox):
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    pad = int(0.15 * max(x1 - x0, y1 - y0))
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
    X1, Y1 = min(img.shape[1], x1 + pad), min(img.shape[0], y1 + pad)
    crop = img[Y0:Y1, X0:X1]
    s = min(1.0, 320.0 / max(crop.shape[:2]))
    small = (
        cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else crop.copy()
    )
    m = np.zeros(small.shape[:2], np.uint8)
    rect = (
        int((x0 - X0) * s),
        int((y0 - Y0) * s),
        max(2, int((x1 - x0) * s)),
        max(2, int((y1 - y0) * s)),
    )
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(small, m, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    return fg


def eval_puzzlemap(mod, root, names):
    ann = json.loads((root / "annotations.json").read_text())
    conf, sides, seqs, unc, times, errors = Counter(), [0, 0], [0, 0], Counter(), [], 0
    for name in names:
        img = cv2.imread(str(root / name))
        if img is None:
            continue
        p = ann[name]["pieces"][0]
        want = [LABEL[p["sides"][k]] for k in ("right", "top", "left", "bottom")]
        truth = CLASS.get(want.count("flat"), "unknown")
        m = grabcut_mask(img, p["bbox"])
        t0 = time.perf_counter()
        try:
            info = mod.classify_mask(m)
        except ValueError:
            errors += 1
            conf[(truth, "error")] += 1
            continue
        times.append(time.perf_counter() - t0)
        conf[(truth, info["piece_class"])] += 1
        n, ok = best_shift(info["side_types"], want)
        sides[0] += n
        sides[1] += 4
        seqs[0] += ok
        seqs[1] += 1
        unc[(info["uncertain_sides"] == 0, truth == info["piece_class"])] += 1
    right = sum(v for (t, g), v in conf.items() if t == g)
    certain = unc[(True, True)] + unc[(True, False)]
    return {
        "pieces": sum(conf.values()),
        "class_right": right,
        "sides_right": f"{sides[0]}/{sides[1]}",
        "sequence_right": f"{seqs[0]}/{seqs[1]}",
        "certain_pieces": certain,
        "certain_class_right": unc[(True, True)],
        "errors": errors,
        "time_median_ms": float(np.median(times) * 1e3) if times else None,
        "confusion": {f"{t}->{g}": v for (t, g), v in sorted(conf.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzlemap", default="")
    ap.add_argument("--list", default="")
    ap.add_argument("--px", type=float, default=64.0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    mod = _load()
    rng = np.random.default_rng(0)
    summary = {"generated": eval_generated(mod, args.px, rng), "px_per_cell": args.px}
    if args.puzzlemap:
        root = pathlib.Path(args.puzzlemap)
        names = (
            pathlib.Path(args.list).read_text().split()
            if args.list
            else sorted(p.name for p in root.glob("*.jpg"))
        )
        summary["puzzlemap"] = eval_puzzlemap(mod, root, names)
    print(json.dumps(summary, indent=1))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
