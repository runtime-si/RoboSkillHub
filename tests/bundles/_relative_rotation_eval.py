"""Score skills/estimating-relative-rotation-and-symmetry on orientation pairs (CPU). Not collected by pytest.

    python tests/bundles/_relative_rotation_eval.py PAIRS_DIR [outline|catalog]
    MASKSUF=_sam2.npy python ...   # masks from sam2 (files <pair>_<view>_sam2.npy) instead of the rendered ones

catalog: symmetry_order from the part (face-up domino 1, elongated 2, square 4); outline: not given.
Error = |reported - true| modulo the reported period.
"""

import collections
import importlib.util
import json
import os
import sys

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(
    REPO, "skills/estimating-relative-rotation-and-symmetry/scripts/relative_rotation.py"
)
spec = importlib.util.spec_from_file_location("rr", SCRIPT)
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)

D = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "outline"
MASKSUF = os.environ.get("MASKSUF", "_mask.npy")


def view(p):
    img = np.asarray(Image.open(os.path.join(D, p["image"])).convert("RGB"))
    return img, np.load(os.path.join(D, p["mask"].replace("_mask.npy", MASKSUF)))


def main():
    rows = []
    for pr in json.load(open(os.path.join(D, "labels.json")))["pairs"]:
        ri, rm = view(pr["ref"])
        ci, cm = view(pr["tgt"])
        r = rr.run(None, ri, rm, ci, cm, symmetry_order=pr["sym"] if MODE == "catalog" else 0)
        per = r["period_deg"]
        d = (r["rotation_deg"] - pr["rel_deg"]) % per
        rows.append({"sym": pr["sym"], "route": r["route"], "err": min(d, per - d)})
    e = np.array([x["err"] for x in rows])
    print(
        f"{MODE}: n {len(rows)}, median {np.median(e):.2f} deg, <=5 {np.mean(e <= 5):.3f}, <=10 {np.mean(e <= 10):.3f}"
    )
    for sy in (1, 2, 4):
        xs = [x for x in rows if x["sym"] == sy]
        ee = np.array([x["err"] for x in xs])
        routes = dict(collections.Counter(x["route"] for x in xs))
        print(
            f"  symmetry {sy}: n {len(xs)} {routes} median {np.median(ee):.2f} <=5 {np.mean(ee <= 5):.2f}"
        )


if __name__ == "__main__":
    main()
