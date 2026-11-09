"""Scenes of nominally identical dominoes with zero, one or two odd ones, for
flagging-odd-one-out-parts-by-mutual-scoring (rendered by tests/bundles/_domino_render.py).

Layouts (10 tiles, a look-down camera at 0.20 m, a tile 48 px wide):

- upright_line: a line of standing tiles; odd = one fallen on its side, or one standing turned 90 deg.
- face_down_grid: 2 x 5 tiles lying face down; odd = one face up (pips), or one standing up.
- flat_same: 2 x 5 face-up tiles of one value; odd = one of another value, or one face down.

Each scene also comes as a control with no odd tile. Styles, angle jitter, blur, noise and glare are
drawn per scene.

    python -m tests.bundles._odd_one_out_scenes OUT_DIR [--seeds 4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import _domino_render as D

LAYOUTS = ("upright_line", "face_down_grid", "flat_same")
ODD = {
    "upright_line": ("fallen", "turned"),
    "face_down_grid": ("face_up", "standing"),
    "flat_same": ("other_value", "face_down"),
}


def scene(
    layout: str, odd: str | None, seed: int
) -> tuple[dict, list[np.ndarray], list[int], dict]:
    rng = np.random.default_rng(seed)
    style = str(rng.choice(["white", "ivory", "black"]))
    value = (int(rng.integers(1, 7)), int(rng.integers(1, 7)))
    tiles = []
    if layout == "upright_line":
        for i in range(10):
            tiles.append(
                {
                    "value": value,
                    "center_px": (70 + 55 * i + rng.normal(0, 1.5), 240 + rng.normal(0, 1.5)),
                    "angle_deg": 90.0 + rng.normal(0, 3),
                    "state": "upright",
                }
            )
    else:
        state = "face_down" if layout == "face_down_grid" else "flat"
        for r in range(2):
            for c in range(5):
                tiles.append(
                    {
                        "value": value,
                        "center_px": (
                            90 + 115 * c + rng.normal(0, 2),
                            150 + 170 * r + rng.normal(0, 2),
                        ),
                        "angle_deg": 90.0
                        + rng.normal(0, 4)
                        + (180.0 if rng.random() < 0.5 else 0.0),
                        "state": state,
                    }
                )
    odd_idx: list[int] = []
    if odd is not None:
        i = int(rng.integers(0, len(tiles)))
        odd_idx = [i]
        t = tiles[i]
        if odd == "fallen":
            t["state"], t["angle_deg"] = "fallen", float(rng.uniform(0, 180))
        elif odd == "turned":
            t["angle_deg"] = 0.0 + rng.normal(0, 3)
        elif odd == "face_up":
            t["state"] = "flat"
            t["value"] = (int(rng.integers(0, 7)), int(rng.integers(1, 7)))
        elif odd == "standing":
            t["state"] = "upright"
        elif odd == "other_value":
            other = value
            while other == value or tuple(sorted(other)) == tuple(sorted(value)):
                other = (int(rng.integers(0, 7)), int(rng.integers(0, 7)))
            t["value"] = other
        elif odd == "face_down":
            t["state"] = "face_down"
    frame, masks, truth = D.render(
        tiles,
        style=style,
        blur=float(rng.uniform(0.3, 1.0)),
        noise=float(rng.uniform(1.0, 4.0)),
        glare=float(rng.choice([0.0, 0.0, 40.0])),
        seed=seed,
    )
    meta = {"layout": layout, "odd_kind": odd, "odd": odd_idx, "style": style, "seed": seed}
    return frame, masks, odd_idx, meta


def all_scenes(seeds: int = 4):
    for layout in LAYOUTS:
        for kind in (*ODD[layout], None):
            for s in range(seeds):
                yield scene(
                    layout,
                    kind,
                    1000 * LAYOUTS.index(layout)
                    + 10 * s
                    + (0 if kind is None else 1 + ODD[layout].index(kind)),
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for k, (frame, masks, _, meta) in enumerate(all_scenes(a.seeds)):
        name = f"scene_{k:03d}"
        np.savez_compressed(
            out / f"{name}.npz",
            rgb=frame["rgb"],
            depth=frame["depth"],
            intrinsics=frame["intrinsics"],
            masks=np.stack(masks).astype(np.uint8),
        )
        index.append({"name": name, **meta, "pose": frame["pose"]})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)} scenes in {out}")


if __name__ == "__main__":
    main()
