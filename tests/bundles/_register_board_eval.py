"""Measure registering-assembled-board-to-reference on a real photo against its box picture.

Default data: PuzzleMap (https://huggingface.co/datasets/pablo-moreira/puzzle-map, CC-BY-4.0),
puzzles/100_dc.jpg (the digital picture, 10 x 10) and puzzles/100_dc2.jpg (a photo of the same
print). Not committed; pass the directory holding them.

Each trial puts the photo on a mat under a random perspective (corners moved up to 4 % of the
board), changes the light (gain, a gradient, blur, noise), then swaps pairs of cells, turns
cells 180 degrees and empties cells (mat colour), and hands the skill board corners that are
off by up to ``--corner-error`` cells. Truth per cell is compared with the status.

    python tests/bundles/_register_board_eval.py --data DIR [--trials 40]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import time
from collections import Counter

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve()
SCRIPT = (
    HERE.parents[2]
    / "skills"
    / "registering-assembled-board-to-reference"
    / "scripts"
    / "register_board.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("register_board", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_trial(
    photo,
    cols,
    rows,
    rng,
    n_swap=3,
    n_rot=3,
    n_empty=3,
    corner_error=0.3,
    mat=(70, 96, 84),
    placed_frac=1.0,
):
    h, w = photo.shape[:2]
    cw, ch = w / cols, h / rows
    board = photo.copy()
    cells = [(c, r) for r in range(rows) for c in range(cols)]
    # The placed part grows row by row from the top-left, as an assembly would; the errors
    # are among the placed cells.
    n_placed = max(n_swap * 2 + n_rot + n_empty + 4, round(placed_frac * len(cells)))
    placed = cells[:n_placed]
    cells = list(placed)
    order = rng.permutation(len(cells))
    truth = {cell: "match" for cell in cells}
    k = 0

    def rect(c, r):
        return slice(round(r * ch), round((r + 1) * ch)), slice(round(c * cw), round((c + 1) * cw))

    for _ in range(n_swap):
        a, b = cells[order[k]], cells[order[k + 1]]
        k += 2
        ya, xa = rect(*a)
        yb, xb = rect(*b)
        pa, pb = photo[ya, xa].copy(), photo[yb, xb].copy()
        board[ya, xa] = cv2.resize(pb, (pa.shape[1], pa.shape[0]))
        board[yb, xb] = cv2.resize(pa, (pb.shape[1], pb.shape[0]))
        truth[a] = truth[b] = "mismatch"
    for _ in range(n_rot):
        a = cells[order[k]]
        k += 1
        ya, xa = rect(*a)
        board[ya, xa] = np.rot90(photo[ya, xa], 2)
        truth[a] = "rotated"
    for _ in range(n_empty):
        a = cells[order[k]]
        k += 1
        ya, xa = rect(*a)
        patch = np.empty_like(board[ya, xa])
        patch[:] = mat
        board[ya, xa] = np.clip(patch + rng.normal(0, 4, patch.shape), 0, 255).astype(np.uint8)
        truth[a] = "empty"
    for cell in [(c, r) for r in range(rows) for c in range(cols)]:
        if cell in truth:
            continue
        ya, xa = rect(*cell)
        board[ya, xa] = mat
    # Perspective onto a mat canvas, then light the whole view: gain, a left-right gradient,
    # blur, noise (the mat in an empty cell and round the board see the same light).
    margin = 0.15
    cw_, ch_ = round(w * (1 + 2 * margin)), round(h * (1 + 2 * margin))
    canvas = np.empty((ch_, cw_, 3), np.uint8)
    canvas[:] = mat
    canvas = np.clip(canvas + rng.normal(0, 4, canvas.shape), 0, 255).astype(np.uint8)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-0.04, 0.04, (4, 2)) * [w, h]
    dst = (src + [margin * w, margin * h] + jitter).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(board, m, (cw_, ch_))
    cover = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, (cw_, ch_)) > 0
    canvas[cover] = warped[cover]
    grad = np.linspace(rng.uniform(0.75, 0.95), rng.uniform(1.0, 1.2), cw_)[None, :, None]
    gain = rng.uniform(0.85, 1.1, 3)[None, None, :]
    lit = canvas.astype(np.float32) * grad * gain
    lit = cv2.GaussianBlur(lit, (0, 0), 1.0)
    canvas = np.clip(lit + rng.normal(0, 3, lit.shape), 0, 255).astype(np.uint8)
    given = dst + rng.uniform(-corner_error, corner_error, (4, 2)) * [cw, ch]
    return canvas, dst, given, truth, placed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--reference", default="100_dc.jpg")
    ap.add_argument("--photo", default="100_dc2.jpg")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--corner-error", type=float, default=0.3, help="cells")
    ap.add_argument("--no-corners", action="store_true", help="let SIFT find the board")
    ap.add_argument("--placed", type=float, default=1.0, help="fraction of cells placed")
    ap.add_argument("--no-features", action="store_true", help="no SIFT fallback")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    mod = _load()
    data = pathlib.Path(args.data)
    ref = cv2.cvtColor(cv2.imread(str(data / args.reference)), cv2.COLOR_BGR2RGB)
    photo = cv2.cvtColor(cv2.imread(str(data / args.photo)), cv2.COLOR_BGR2RGB)
    photo = cv2.resize(photo, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(args.seed)
    conf: Counter = Counter()
    routes: Counter = Counter()
    inits: Counter = Counter()
    corner_err, rhos, times, lookups = [], [], [], [0, 0]
    ncc_right, ncc_swapped = [], []
    # 3 of each error on a full 10 x 10 board, fewer on a part-built one (at least 1 of each).
    n_err = max(1, round(3 * args.placed))
    for _ in range(args.trials):
        canvas, true_corners, given, truth, placed = make_trial(
            photo,
            args.cols,
            args.rows,
            rng,
            n_swap=n_err,
            n_rot=n_err,
            n_empty=n_err,
            corner_error=args.corner_error,
            placed_frac=args.placed,
        )
        t0 = time.perf_counter()
        out = mod.run(
            None,
            ref,
            args.cols,
            args.rows,
            image=canvas,
            board_corners_px=None if args.no_corners else given.tolist(),
            use_features=not args.no_features,
            placed_cells=[list(c) for c in placed] if args.placed < 1 else None,
        )
        inits[out.get("init", "")] += 1
        times.append(time.perf_counter() - t0)
        routes[out["route"]] += 1
        if out["route"] != "registered":
            continue
        rhos.append(out["ecc_correlation"])
        # Error over the placed cells' centres: where the photo's cell centre lands in the
        # reference against where it belongs (the photo was resized to the reference's size).
        hm = np.asarray(out["homography"])
        cell_w, cell_h = ref.shape[1] / args.cols, ref.shape[0] / args.rows
        want = np.float64([[(c + 0.5) * cell_w, (r + 0.5) * cell_h] for c, r in placed])
        m_photo = cv2.getPerspectiveTransform(
            np.float32(
                [[0, 0], [ref.shape[1], 0], [ref.shape[1], ref.shape[0]], [0, ref.shape[0]]]
            ),
            true_corners.astype(np.float32),
        )
        in_image = cv2.perspectiveTransform(want.reshape(-1, 1, 2), m_photo)
        got = cv2.perspectiveTransform(in_image, hm).reshape(-1, 2)
        corner_err.append(float(np.max(np.linalg.norm((got - want) / [cell_w, cell_h], axis=1))))
        for e in out["cells"]:
            t = truth[(e["col"], e["row"])]
            conf[(t, e["status"])] += 1
            if "ncc" in e:
                (ncc_right if t == "match" else ncc_swapped if t == "mismatch" else []).append(
                    e["ncc"]
                )
            if t == "mismatch" and "belongs_at" in e:
                lookups[1] += 1
            if t == "mismatch":
                lookups[0] += 1
    summary = {
        "trials": args.trials,
        "routes": dict(routes),
        "inits": dict(inits),
        "cell_error_cells_median": float(np.median(corner_err)) if corner_err else None,
        "cell_error_cells_max": float(np.max(corner_err)) if corner_err else None,
        "ecc_median": float(np.median(rhos)) if rhos else None,
        "ecc_min": float(np.min(rhos)) if rhos else None,
        "time_median_s": float(np.median(times)),
        "time_max_s": float(np.max(times)),
        "ncc_right": [float(np.min(ncc_right)), float(np.median(ncc_right))] if ncc_right else None,
        "ncc_swapped_max": float(np.max(ncc_swapped)) if ncc_swapped else None,
        "swapped_with_belongs_at": lookups,
        "confusion": {f"{t}->{s}": n for (t, s), n in sorted(conf.items())},
    }
    print(json.dumps(summary, indent=1))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
