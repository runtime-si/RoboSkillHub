"""Run one backend over the target-part eval set (bricks, dominoes, jigsaw pieces) and record raw answers.
Not collected by pytest; on a GPU host from the repository root, one backend per process, in the
backend's environment (mm-grounding-dino: transformers 4.x):

    TARGET_PARTS_DIR=WORK python tests/bundles/_target_parts_eval.py BACKEND OUT.json [MODEL]

WORK holds data/render, data/oddone and data/jigsaw as tools/README.md ("Reproduce") builds them.

BACKEND: owlv2 | owlv2-like | grounding-dino | mm-grounding-dino | omdet-turbo | sam2.
Detectors: every image through tools/sahi's detect_sliced (whole image; overhead renders also 320 px slices), raw
answers kept at a low floor (scored per floor by target_score.py). owlv2-like: the largest labelled instance of each
class is the exemplar, searched in its own image. sam2: every labelled instance's box -> sam2.tracker_init mask,
IoU with the rendered mask.
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.environ.get("TARGET_PARTS_DIR", "target_parts")
BACKEND, OUT = sys.argv[1], sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv) > 3 else None
for env, b in (
    ("GAP_DINO_MODEL", "grounding-dino"),
    ("GAP_MMGDINO_MODEL", "mm-grounding-dino"),
    ("GAP_SAM2_MODEL", "sam2"),
    ("GAP_OWLV2_MODEL", "owlv2"),
):
    if MODEL and BACKEND.startswith(b):
        os.environ[env] = MODEL


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sahi = load("sahi_tools", "tools/sahi/tools.py")
SETS = {
    "render": "data/render",
    "oddone": "data/oddone",
    "jigsaw": "data/jigsaw",
}
PROMPTS = {
    "render": ["lego brick", "domino"],
    "oddone": ["lego brick", "domino"],
    "jigsaw": ["jigsaw puzzle piece"],
}


def scenes():
    for ds, d in SETS.items():
        for s in json.load(open(os.path.join(HERE, d, "labels.json")))["scenes"]:
            if s["objects"]:
                yield ds, d, s


def image(d, s):
    return np.asarray(Image.open(os.path.join(HERE, d, s["image"])).convert("RGB"))


class Ctx:
    def __init__(self, fns, extra):
        self.fns, self.extra, self.log, self.secs = fns, extra, [], []

    def tool(self, name, **kw):
        kw = {**kw, **self.extra.get(name, {})}
        t = time.perf_counter()
        out = self.fns[name](**kw)
        self.secs.append(time.perf_counter() - t)
        self.log.append(out)
        return out


def detector():
    if BACKEND == "owlv2":
        m = load("owl", "tools/owlv2/tools.py")
        return m.detect, "owlv2.detect", {"owlv2.detect": {"threshold": 0.05}}, 0.05
    if BACKEND == "grounding-dino":
        m = load("gd", "tools/grounding-dino/tools.py")
        fn = "grounding-dino.detect"
        return m.detect, fn, {fn: {"box_threshold": 0.1, "text_threshold": 0.2}}, 0.1
    if BACKEND == "mm-grounding-dino":
        m = load("mmg", "tools/mm-grounding-dino/tools.py")
        fn = "mm-grounding-dino.detect"
        return m.detect, fn, {fn: {"box_threshold": 0.1, "text_threshold": 0.2}}, 0.1
    m = load("om", "tools/omdet-turbo/tools.py")
    return m.detect, "omdet-turbo.detect", {"omdet-turbo.detect": {"threshold": 0.05}}, 0.05


def run_detector():
    det, fn, extra, low = detector()
    backend = "grounding-dino" if fn == "grounding-dino.detect" else fn
    t = time.perf_counter()
    det(image=np.zeros((480, 640, 3), np.uint8), query="lego brick. domino")
    load_s = time.perf_counter() - t
    runs = []
    for ds, d, s in scenes():
        img = image(d, s)
        configs = {"full": {"slice_px": 0}}
        if s.get("view") == "overhead":
            configs["s320"] = {"slice_px": 320}
        for cname, cfg in configs.items():
            ctx = Ctx({fn: det}, extra)
            t = time.perf_counter()
            r = sahi.detect_sliced(ctx, img, PROMPTS[ds], backend=backend, min_score=low, **cfg)
            runs.append(
                {
                    "set": ds,
                    "image": s["image"],
                    "view": s.get("view"),
                    "config": cname,
                    "detections": r["detections"],
                    "s": time.perf_counter() - t,
                }
            )
        print("done", ds, s["image"], flush=True)
    return {"runs": runs, "load_s": load_s}


def run_like():
    m = load("owl", "tools/owlv2/tools.py")
    runs = []
    for ds, d, s in scenes():
        img = image(d, s)
        for cls in sorted({o["cls"] for o in s["objects"]}):
            inst = [o for o in s["objects"] if o["cls"] == cls]
            q = max(inst, key=lambda o: (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]))
            t = time.perf_counter()
            r = m.detect_like(img, query_box=q["box"], threshold=0.6, exclude_query=True)
            runs.append(
                {
                    "set": ds,
                    "image": s["image"],
                    "view": s.get("view"),
                    "cls": cls,
                    "query": q["box"],
                    "detections": r["detections"],
                    "s": time.perf_counter() - t,
                }
            )
        print("done", ds, s["image"], flush=True)
    return {"runs": runs}


def run_sam2():
    m = load("s2", "tools/sam2/tools.py")
    rows = []
    for ds, d, s in scenes():
        if not s.get("masks"):
            continue
        img = image(d, s)
        masks = np.load(os.path.join(HERE, d, s["masks"]))["masks"]
        for o, gm in zip(s["objects"], masks, strict=True):
            t = time.perf_counter()
            r = m.tracker_init(img, box=o["box"])
            dt = time.perf_counter() - t
            if r["tracker_id"]:
                m.tracker_close(r["tracker_id"])
            pm = (
                (np.asarray(r["initial_mask"]) > 0)
                if r["initial_mask"] is not None
                else np.zeros_like(gm, bool)
            )
            g = gm > 0
            iou = float((pm & g).sum() / max((pm | g).sum(), 1))
            rows.append(
                {
                    "set": ds,
                    "view": s.get("view"),
                    "cls": o["cls"],
                    "sub": o["sub"],
                    "area": int(g.sum()),
                    "iou": iou,
                    "s": dt,
                }
            )
        print("done", ds, s["image"], flush=True)
    return {"rows": rows}


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    res = (
        run_like()
        if BACKEND == "owlv2-like"
        else run_sam2()
        if BACKEND == "sam2"
        else run_detector()
    )
    res.update({"backend": BACKEND, "model": MODEL})
    try:
        import torch

        res["peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
    except Exception:
        pass
    json.dump(res, open(OUT, "w"))
    print("CHECK done", BACKEND, MODEL)
