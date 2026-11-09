"""tools/omdet-turbo's numpy decode vs transformers' post_process_grounded_object_detection on the 7
station frames at 4 floors. Not collected by pytest; on a GPU host from the repository root:

  python tests/bundles/_omdet_turbo_parity.py FRAMES
"""

import importlib.util
import os
import sys

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRAMES = sys.argv[1] if len(sys.argv) > 1 else "frames"
spec = importlib.util.spec_from_file_location(
    "omdet", os.path.join(REPO, "tools/omdet-turbo/tools.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
rt = mod._get_runtime()
PROMPTS = {"cube": ["cube", "box", "pen"], "thread": ["ring", "plug"]}
SETS = {
    "gsota_0511": "cube",
    "gsota_0630": "cube",
    "gsota_1101": "cube",
    "cube_1031": "cube",
    "cable_1421": "thread",
    "setup_1103": "thread",
    "paired_1104": "thread",
}
worst_px, worst_s, n_ref, n_ours, mism = 0.0, 0.0, 0, 0, 0
for name, s in SETS.items():
    img = np.asarray(Image.open(os.path.join(FRAMES, f"{name}.png")).convert("RGB"))
    names = PROMPTS[s]
    for thr in (0.05, 0.1, 0.2, 0.3):
        inputs = rt.processor(
            images=Image.fromarray(img),
            text=names,
            task=f"Detect {', '.join(names)}.",
            return_tensors="pt",
        ).to(rt.device)
        with torch.no_grad():
            out = rt.model(**inputs)
        ref = rt.processor.post_process_grounded_object_detection(
            out, text_labels=[names], threshold=thr, nms_threshold=0.5, target_sizes=[img.shape[:2]]
        )[0]
        rb, rs, rl = (
            ref["boxes"].cpu().numpy(),
            ref["scores"].cpu().numpy(),
            list(ref["labels"].cpu().numpy()),
        )
        ours = mod.detect(img, names, threshold=thr)["detections"]
        ob = np.array([[d["box"][k] for k in ("x1", "y1", "x2", "y2")] for d in ours]).reshape(
            -1, 4
        )
        os_ = np.array([d["score"] for d in ours])
        ol = [names.index(d["label"]) for d in ours]
        n_ref += len(rb)
        n_ours += len(ob)
        order = np.argsort(-rs, kind="stable")
        rb, rs, rl = rb[order], rs[order], [rl[i] for i in order]
        if len(rb) != len(ob) or rl != ol:
            mism += 1
            print("MISMATCH", name, thr, len(rb), len(ob))
            continue
        if len(rb):
            worst_px = max(worst_px, float(np.abs(rb - ob).max()))
            worst_s = max(worst_s, float(np.abs(rs - os_).max()))
print(
    f"PARITY cases=28 mismatched_sets={mism} boxes ref={n_ref} ours={n_ours} max_box_diff_px={worst_px:.4f} max_score_diff={worst_s:.2e}"
)
