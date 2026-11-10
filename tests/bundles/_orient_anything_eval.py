"""Orient Anything V2 (the rejected backend) on the orientation pairs. Not collected by pytest; on a GPU host,
with a checkout of SpatialVision/Orient-Anything-V2 @ 73b11c9 and Viglong/OriAnyV2_ckpt
demo_ckpts/rotmod_realrotaug_best.pt (CC-BY-4.0):

    PAIRS_DIR=... python tests/bundles/_orient_anything_eval.py OA_REPO CKPT OUT.json
"""

import json
import math
import os
import sys
import time
import types

import numpy as np
from PIL import Image

REPO, CKPT, OUT = sys.argv[1:4]
PAIRS = os.environ.get("PAIRS_DIR", "orient")
sys.modules.setdefault(
    "rembg", types.ModuleType("rembg")
)  # background removal: not used (masks given)
sys.path.insert(0, REPO)
import torch  # noqa: E402
from utils.app_utils import azi_ele_rot_to_Obj_Rmatrix_batch, inf_single_case  # noqa: E402
from vision_tower import VGGT_OriAny_Ref  # noqa: E402

dtype = torch.bfloat16
t = time.perf_counter()
model = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model = model.eval().cuda()
load_s = time.perf_counter() - t
L = json.load(open(os.path.join(PAIRS, "labels.json")))["pairs"]


def crop(p, masked):
    img = np.asarray(Image.open(os.path.join(PAIRS, p["image"])).convert("RGB")).copy()
    m = np.load(os.path.join(PAIRS, p["mask"]))
    if masked:
        img[~m] = 255
    x1, y1, x2, y2 = p["box"]
    s = max(x2 - x1, y2 - y1) * 1.5
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    b = [
        int(max(0, cx - s / 2)),
        int(max(0, cy - s / 2)),
        int(min(640, cx + s / 2)),
        int(min(480, cy + s / 2)),
    ]
    return Image.fromarray(img[b[1] : b[3], b[0] : b[2]])


def axis_angle(mask):
    ys, xs = np.nonzero(mask)
    c = np.cov(np.stack([xs - xs.mean(), ys - ys.mean()]))
    w, v = np.linalg.eigh(c)
    a = v[:, 1]
    return math.degrees(math.atan2(a[1], a[0])), float(w[1] / max(w[0], 1e-9))


def fold(deg, sym):
    """|deg| after removing multiples of 360/sym, in [0, 180/sym]."""
    p = 360.0 / sym
    d = deg % p
    return min(d, p - d)


rows = []
for i, pr in enumerate(L):
    sym = pr["sym"]
    true = pr["rel_deg"]
    a1, e1 = axis_angle(np.load(os.path.join(PAIRS, pr["ref"]["mask"])))
    a2, _ = axis_angle(np.load(os.path.join(PAIRS, pr["tgt"]["mask"])))
    obb_rel = a2 - a1
    row = {
        "sub": pr["sub"],
        "sym": sym,
        "true": true,
        "true_fold": fold(true, sym),
        "obb_err": abs(fold(obb_rel, sym if sym > 1 else 2) - fold(true, sym if sym > 1 else 2))
        if sym in (1, 2)
        else abs(fold(obb_rel, 4) - fold(true, 4)),
        "elong": e1,
    }
    for masked in (False, True):
        tag = "masked" if masked else "raw"
        t = time.perf_counter()
        with torch.no_grad():
            d = inf_single_case(model, crop(pr["ref"], masked), crop(pr["tgt"], masked))
        dt = time.perf_counter() - t
        Rr = (
            azi_ele_rot_to_Obj_Rmatrix_batch(
                torch.tensor([float(d["rel_az_pred"])]),
                torch.tensor([float(d["rel_el_pred"])]),
                torch.tensor([float(d["rel_ro_pred"])]),
            )[0]
            .double()
            .numpy()
        )
        theta = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rr) - 1) / 2))))
        row[tag] = {
            "theta": theta,
            "err": abs(fold(theta, sym) - fold(abs(true), sym)),
            "alpha": int(d["ref_alpha_pred"]),
            "rel": [float(d["rel_az_pred"]), float(d["rel_el_pred"]), float(d["rel_ro_pred"])],
            "ref_abs": [float(d["ref_az_pred"]), float(d["ref_el_pred"]), float(d["ref_ro_pred"])],
            "s": dt,
        }
    rows.append(row)
    print(
        i,
        pr["sub"],
        round(true, 1),
        "obb",
        round(row["obb_err"], 1),
        "oa raw",
        round(row["raw"]["err"], 1),
        "masked",
        round(row["masked"]["err"], 1),
        "alpha",
        row["raw"]["alpha"],
        row["masked"]["alpha"],
        "sym",
        sym,
        flush=True,
    )
json.dump(
    {"load_s": load_s, "peak_mib": torch.cuda.max_memory_allocated() / 2**20, "rows": rows},
    open(OUT, "w"),
)
print("ORIENTDONE")
