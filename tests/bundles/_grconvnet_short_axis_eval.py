"""proposing-short-axis-grasps on the same station targets as ``_grconvnet_station_eval.py``,
judged by the same criteria. Not collected by pytest; run from the repo root in the hub env:

    .venv/bin/python tests/bundles/_grconvnet_short_axis_eval.py --frames ~/yam_gap

Each target's mask goes through geometry.mask_to_world_points and
geometry.filter_and_compute_obb (the camera-only route the skill's SKILL.md names; their implementations are called directly), then
the skill's script with the YAM's 94 mm span; its stations are ranked centre first.
Needs open3d, scikit-learn and OpenCV (tools/geometry's environment) next to gap-core, e.g.
the grconvnet env plus ``uv pip install open3d scikit-learn opencv-python-headless``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import _grconvnet_criteria as C  # noqa: E402
import _grconvnet_station_eval as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=Path.home() / "yam_gap")
    ap.add_argument("--out", type=Path, default=Path("/tmp/grconv_check"))
    args = ap.parse_args()
    results = {}
    geo = _load("geometry_impl", ROOT / "tools/geometry/_impl.py")
    if "gap" not in sys.modules:  # the script imports NodeContext only for its annotation
        sys.modules["gap"] = types.SimpleNamespace(NodeContext=object)
    spec = importlib.util.spec_from_file_location(
        "short_axis", ROOT / "skills/proposing-short-axis-grasps/scripts/propose_short_axis.py"
    )
    sa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sa)

    fr = E.frames(args.frames)
    ms = E.masks(fr)
    top1 = top3 = 0
    for obj, (scene, m) in ms.items():
        rgb, depth, K, T = fr[scene]
        gt = C.ground_truth(E.kind(obj), depth, K, T, m)
        t0 = time.perf_counter()
        pts = geo.mask_to_world_points(m, depth.astype(np.float32), K.astype(np.float64), T)
        obb = geo.compute_obb(geo.filter_noise(pts, 0.005, 10))
        out = sa.run(None, obb, {"max_grasp_width_m": 0.094})
        dt = time.perf_counter() - t0
        ext = obb["extent"]
        head = f"{obj:12s} obb {2e3 * ext['x']:.0f}x{2e3 * ext['y']:.0f}x{2e3 * ext['z']:.0f}mm {1e3 * dt:.0f}ms"
        results[obj] = {"obb": obb, "proposal": out, "verdicts": []}
        if out["route"] != "proposed":
            print(head, "declined:", out["reason"][:70])
            continue
        verdicts = [
            C.judge(gt, [p["x"], p["y"]], out["close_heading_deg"], out["grip_width_m"])
            for p in out["positions"][:3]
        ]
        results[obj]["verdicts"] = verdicts
        top1 += verdicts[0][0]
        top3 += any(v[0] for v in verdicts)
        print(head, " | ".join(f"{'PASS' if v[0] else 'fail'} {v[1]}" for v in verdicts))
    print(f"short-axis: top1 {top1} top3 {top3} of {len(ms)}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "short_axis_station.json").write_text(json.dumps(results, indent=1, default=float))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    main()
