"""Score an RF-DETR student (tools/rfdetr) on labelled scenes: found at IoU >= 0.5 per class, FP per frame.
Not collected by pytest; on a GPU host from the repository root, in the rfdetr bundle's environment:

    python tests/bundles/_rfdetr_student_eval.py MODEL_DIR SCENES_DIR [THRESHOLD]

SCENES_DIR holds labels.json ({"scenes": [{"image", "view", "objects": [{"cls", "box"}]}]}) and the images, as
tests/bundles/_target_parts_gen_scenes.py (tools/README.md) writes them; lego and duplo count as "brick".
"""

import importlib.util
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("rf", os.path.join(REPO, "tools/rfdetr/tools.py"))
rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def main():
    model, scenes = sys.argv[1], sys.argv[2]
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    found, total, fp, frames, secs = (
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
        [],
    )
    for s in json.load(open(os.path.join(scenes, "labels.json")))["scenes"]:
        if not s["objects"]:
            continue
        img = np.asarray(Image.open(os.path.join(scenes, s["image"])).convert("RGB"))
        t = time.perf_counter()
        dets = rf.detect(img, model, threshold=thr)["detections"]
        secs.append(time.perf_counter() - t)
        gts = [
            ("domino" if o["cls"] == "domino" else "brick", o["cls"], o["box"])
            for o in s["objects"]
        ]
        used = [False] * len(gts)
        view = s.get("view", "")
        for d in dets:
            b = [d["box"][k] for k in ("x1", "y1", "x2", "y2")]
            best, bi = 0.5, -1
            for i, (c, _, g) in enumerate(gts):
                if not used[i] and c == d["label"] and iou(b, g) >= best:
                    best, bi = iou(b, g), i
            if bi >= 0:
                used[bi] = True
            else:
                fp[view] += 1
        for (_, sub, _), u in zip(gts, used, strict=True):
            found[(view, sub)] += u
            total[(view, sub)] += 1
        frames[view] += 1
    for key in sorted(total):
        print(f"{key[0]:10} {key[1]:8} found {found[key] / total[key]:.2f} ({total[key]})")
    for v in sorted(frames):
        print(f"{v:10} FP per frame {fp[v] / frames[v]:.1f}")
    print(f"median {np.median(secs) * 1000:.1f} ms a frame")


if __name__ == "__main__":
    main()
