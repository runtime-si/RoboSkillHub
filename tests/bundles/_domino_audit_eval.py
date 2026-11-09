"""Measures auditing-a-domino-line-for-fallen-or-tilted-tiles on random synthetic lines: each tile upright,
displaced (4-8 mm along or across, or turned 12-20 deg), tilted (8-20 deg, either way, its neighbours upright),
leaning on its next neighbour, missing, or the last tile flat. Not collected by pytest:

    .venv/bin/python tests/bundles/_domino_audit_eval.py
"""

from __future__ import annotations

import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _domino_depth_scene as S  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "audit_line",
    Path(__file__).resolve().parents[2]
    / "skills/auditing-a-domino-line-for-fallen-or-tilted-tiles/scripts/audit_line.py",
)
AL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AL)


def scene(rng: random.Random, n: int, pitch: float):
    spec_, truth = {}, ["upright"] * n
    k = 1
    while k < n - 1:
        kind = rng.choice(
            ["upright", "upright", "displaced", "turned", "tilted", "leaning", "missing"]
        )
        if kind == "displaced":
            spec_[k] = {
                rng.choice(["shift", "lateral"]): rng.choice([-1, 1]) * rng.uniform(0.005, 0.008)
            }
            truth[k] = "displaced"
        elif kind == "turned":
            spec_[k] = {"yaw_deg": rng.choice([-1, 1]) * rng.uniform(12, 20)}
            truth[k] = "displaced"
        elif kind == "tilted":
            spec_[k] = {"lean_deg": rng.choice([-1, 1]) * rng.uniform(8, 20)}
            truth[k] = "tilted"
            k += 1  # keep its neighbours clear of a backward lean's room
        elif kind == "leaning":
            spec_[k] = {"lean_deg": S.contact_deg(pitch)}
            truth[k] = "leaning"
            k += 1  # the next stands: it holds the leaner
        elif kind == "missing":
            spec_[k] = None
            truth[k] = "missing"
        k += 1
    if rng.random() < 0.5:
        spec_[n - 1] = {"lean_deg": 90.0}
        truth[n - 1] = "fallen"
    return spec_, truth


def main() -> None:
    for noise in (0.0, 0.001, 0.002):
        rng = random.Random(7)
        conf, lines_right, total_lines, lean_err = Counter(), 0, 0, []
        for _ in range(40):
            pitch = rng.choice([0.030, 0.037, 0.045])
            n = 8
            slots = S.line_slots(n, pitch)
            spec_, truth = scene(rng, n, pitch)
            tiles = [
                S.tile_points(*slots[k], **(spec_.get(k) or {}))
                for k in range(n)
                if spec_.get(k, {}) is not None
            ]
            frame = S.depth_frame(
                tiles,
                centre=(0.0, slots[0][1] + (n - 1) * pitch / 2),
                noise=noise,
                seed=rng.randint(0, 9999),
            )
            res = AL.run(None, slots=slots, table_z=0.0, frame=frame)
            for t, s, k in zip(truth, res["states"], range(n), strict=True):
                conf[(t, s)] += 1
                if t in ("tilted", "leaning") and res["lean_deg"][k] is not None:
                    lean_err.append(abs(res["lean_deg"][k] - (spec_[k]["lean_deg"])))
            lines_right += res["states"] == truth
            total_lines += 1
        right = sum(v for (t, s), v in conf.items() if t == s)
        tot = sum(conf.values())
        per = {
            t: f"{sum(v for (a, b), v in conf.items() if a == t and b == t)}/{sum(v for (a, b), v in conf.items() if a == t)}"
            for t in sorted({a for a, _ in conf})
        }
        wrong = {f"{a}->{b}": v for (a, b), v in conf.items() if a != b}
        lean_err.sort()
        print(
            f"AUDIT noise {noise * 1e3:.0f} mm: tiles {right}/{tot}, lines all right {lines_right}/{total_lines}; per state {per}; "
            f"confusions {wrong}; lean error median {lean_err[len(lean_err) // 2]:.1f} max {lean_err[-1]:.1f} deg"
        )


if __name__ == "__main__":
    main()
