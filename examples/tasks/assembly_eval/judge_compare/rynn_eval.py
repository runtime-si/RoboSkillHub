"""RynnValue on every unit of a manifest (paths relative to --root). Writes one JSON line per unit and model."""

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "stub")
sys.path.insert(0, ".")
import rynn_tools as T

manifest, root, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
units = json.loads(Path(manifest).read_text())
done = set()
if out.exists():
    done = {
        (json.loads(line)["id"], json.loads(line)["model"], json.loads(line)["clip"])
        for line in out.read_text().splitlines()
    }
for key in os.environ.get("RYNN_MODELS", "8b,4b").split(","):
    os.environ["GAP_RYNNVALUE_MODEL"] = key
    T._runtime = None
    import torch

    torch.cuda.empty_cache()
    for u in units:
        imgs = [np.asarray(Image.open(root / f).convert("RGB")) for f in u["frames"]]
        for clip, sel in (("full", imgs), ("first_last", [imgs[0], imgs[-1]])):
            if (u["id"], key, clip) in done:
                continue
            r = T.remaining_time_impl(sel, u["instruction"])
            rec = {
                "id": u["id"],
                "model": key,
                "clip": clip,
                "remaining_s": r["remaining_s"],
                "final_remaining_s": r["final_remaining_s"],
                "success_flag": r["success"],
                "match_flag": r["match"],
                "description": r["description"],
                "timing": r["timing"],
            }
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                key,
                clip,
                u["id"],
                u["success"],
                round(r["final_remaining_s"], 2),
                r["success"],
                r["match"],
                flush=True,
            )
