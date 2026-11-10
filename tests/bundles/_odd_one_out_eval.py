"""Run flagging-odd-one-out-parts-by-mutual-scoring with the real tools/musc on the rendered scenes.

    python -m tests.bundles._odd_one_out_scenes SCENES --seeds 8      # CPU
    python -m tests.bundles._odd_one_out_eval SCENES                  # needs a GPU and the DINOv2 weights

Prints one line per scene and the found / exact / clean counts per layout and odd kind (the SKILL.md
Checks table); writes SCENES/musc_results.json.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
from pathlib import Path

import numpy as np
from gap.testing import FakeContext

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes")
    a = ap.parse_args()
    here = Path(a.scenes)
    musc = _load("musc_tools", ROOT / "tools/musc/tools.py")
    skill = _load(
        "flag",
        ROOT / "skills/flagging-odd-one-out-parts-by-mutual-scoring/scripts/flag_odd_one_out.py",
    )
    rows = []
    for meta in json.loads((here / "index.json").read_text()):
        d = np.load(here / f"{meta['name']}.npz")
        frame = {
            "name": "left_wrist",
            "rgb": d["rgb"],
            "depth": d["depth"],
            "intrinsics": d["intrinsics"],
            "pose": meta["pose"],
        }
        out = skill.run(
            FakeContext({"musc.score": musc.score}), masks=list(d["masks"]), frame=frame
        )
        rows.append(
            {
                **{k: meta[k] for k in ("name", "layout", "odd_kind", "odd")},
                "outliers": out["outliers"],
                "why": [p["why"] for p in out["parts"]],
                "scores": out["scores"],
            }
        )
        print(
            meta["name"], meta["layout"], meta["odd_kind"], meta["odd"], out["outliers"], flush=True
        )
    (here / "musc_results.json").write_text(json.dumps(rows, indent=1))
    tab: dict = collections.defaultdict(collections.Counter)
    for r in rows:
        c = tab[(r["layout"], r["odd_kind"] or "none")]
        c["n"] += 1
        fl, od = set(r["outliers"]), set(r["odd"])
        if od:
            c["found"] += od <= fl
            c["exact"] += od == fl
        else:
            c["clean"] += not fl
    for (layout, kind), c in tab.items():
        print(
            f"{layout} {kind}: n {c['n']} found {c['found']} exact {c['exact']} clean {c['clean']}"
        )


if __name__ == "__main__":
    main()
