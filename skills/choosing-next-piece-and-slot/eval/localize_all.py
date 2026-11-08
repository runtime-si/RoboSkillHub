"""Localizer records for every piece of a 20 x 27 procedural puzzle (the target's real grid).

    python skills/choosing-next-piece-and-slot/eval/localize_all.py --seed 2 --out /tmp/np_loc_s2.json

Runs localizing-a-piece-in-a-reference-image on all 540 pieces of the hub's procedural picture
(tests/bundles/_jigsaw_pictures.py: textures, a flat white band over the top 3 rows, a plaid
patch), cut and degraded as a wrist camera would see them, scale known. ``--seed`` is the photo
seed (piece shape, turn, blur, gain, noise); a second seed stands in for photographing the piece
again. The output maps piece id ``p{col}_{row}`` to ``{"piece", "route", "low_texture",
"candidates"}``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

COLS, ROWS = 20, 27
BOARD_M = (0.4572, 0.6096)
PICTURE_PX = (800, 1067)
"""About 40 x 40 px per cell, as the 22 x 25 runs of the localizer (36 x 42)."""

_loc = None
_ref = None


def _init():
    global _loc, _ref
    from tests.bundles._jigsaw_pictures import make_reference

    path = ROOT / "skills/localizing-a-piece-in-a-reference-image/scripts/locate_piece.py"
    spec = importlib.util.spec_from_file_location("lp", path)
    _loc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_loc)
    _ref = make_reference(*PICTURE_PX, seed=0, band_rows=(0.0, 3 / ROWS), plaid=True)


def _one(args):
    from gap.testing import FakeContext

    from tests.bundles._jigsaw_pictures import cut_piece

    col, row, seed, top_k = args
    ppm = 96 / (BOARD_M[0] / COLS)
    img, mask, _, _ = cut_piece(
        _ref, COLS, ROWS, col, row, 96, seed=seed * 100000 + row * 100 + col
    )
    out = _loc.run(
        FakeContext(),
        mask,
        _ref,
        COLS,
        ROWS,
        *BOARD_M,
        piece_rgb=img,
        crop_px_per_m=ppm,
        top_k=top_k,
    )
    cands = [{k: c.get(k) for k in ("col", "row", "score", "angle_deg")} for c in out["candidates"]]
    return {
        "piece": f"p{col}_{row}",
        "route": out["route"],
        "low_texture": bool(out.get("low_texture")),
        "candidates": cands,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = time.time()
    jobs = [(c, r, a.seed, a.top_k) for r in range(ROWS) for c in range(COLS)]
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        recs = {r["piece"]: r for r in ex.map(_one, jobs, chunksize=4)}
    Path(a.out).write_text(json.dumps(recs))
    right = sum(
        bool(r["candidates"])
        and r["candidates"][0]["col"] == int(p[1:].split("_")[0])
        and r["candidates"][0]["row"] == int(p.split("_")[1])
        for p, r in recs.items()
    )
    print(f"{len(recs)} pieces, top-1 cell right {right}, {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
