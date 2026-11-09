"""Live check of tools/mm-grounding-dino on the 7 labelled YAM overhead frames. Not collected by
pytest; on a GPU host from the repository root, in the bundle's environment (transformers 4.x):

  python tests/bundles/_mm_grounding_dino_station_eval.py OUT/mmg_large.json FRAMES [HF_REPO]
  python tests/bundles/_owlv2_station_score.py text OUT/*.json

FRAMES is a folder of the seven 640 x 480 overhead frames named as in
tests/bundles/data/station_small_parts_labels.json; HF_REPO is one of the bundle's MODELS (default
large). The protocol is _owlv2_station_eval.py's text mode: every frame through tools/sahi's
detect_sliced in six configs, raw answers recorded at box_threshold 0.1 and replayed per floor, plus
5 warm whole-frame calls per frame for latency.
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND = "mm-grounding-dino"
OUT = sys.argv[1]
FRAMES = sys.argv[2] if len(sys.argv) > 2 else "frames"
MODEL = sys.argv[3] if len(sys.argv) > 3 else None
if MODEL:
    os.environ["GAP_MMGDINO_MODEL"] = MODEL


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sahi = load("sahi_tools", "tools/sahi/tools.py")
LABELS = json.load(open(os.path.join(REPO, "tests/bundles/data/station_small_parts_labels.json")))
PROMPTS = {"cube": ["cube", "box", "pen"], "thread": ["ring", "plug"]}
FRAME_SET = {
    "gsota_0511": "cube",
    "gsota_0630": "cube",
    "gsota_1101": "cube",
    "cube_1031": "cube",
    "cable_1421": "thread",
    "setup_1103": "thread",
    "paired_1104": "thread",
}
ROIS = {
    "gsota_0511": [0, 255, 640, 480],
    "gsota_0630": [0, 255, 640, 480],
    "gsota_1101": [0, 250, 640, 480],
    "cube_1031": [0, 70, 640, 480],
    "cable_1421": [0, 255, 640, 480],
    "setup_1103": [0, 255, 640, 480],
    "paired_1104": [0, 260, 640, 480],
}
CONFIGS = {
    "full": {"slice_px": 0},
    "s320": {"slice_px": 320},
    "s256": {"slice_px": 256},
    "full_roi": {"slice_px": 0, "roi": "table"},
    "s320_roi": {"slice_px": 320, "roi": "table"},
    "s256_roi": {"slice_px": 256, "roi": "table"},
}


def frame(name):
    return np.asarray(Image.open(os.path.join(FRAMES, f"{name}.png")).convert("RGB"))


def gpu_mib():
    import torch

    return {
        "max_allocated_mib": int(torch.cuda.max_memory_allocated() // 2**20),
        "max_reserved_mib": int(torch.cuda.max_memory_reserved() // 2**20),
    }


class RecordCtx:
    def __init__(self, fns, extra):
        self.fns, self.extra, self.log, self.secs = fns, extra, [], []

    def tool(self, name, **kw):
        kw = {**kw, **self.extra.get(name, {})}
        t = time.perf_counter()
        out = self.fns[name](**kw)
        self.secs.append(time.perf_counter() - t)
        self.log.append(out)
        return out


class ReplayCtx:
    def __init__(self, log):
        self.log, self.i = log, 0

    def tool(self, name, **kw):
        out = self.log[self.i]
        self.i += 1
        return out


def setup():
    mod = load("mmg", "tools/mm-grounding-dino/tools.py")
    fn = "mm-grounding-dino.detect"
    extra = {fn: {"box_threshold": 0.1, "text_threshold": 0.2}}
    return mod.detect, fn, fn, extra, 0.1, (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)


def main():
    det, fn, sahi_backend, extra, low, floors = setup()
    tag = "tool:" + (MODEL or "openmmlab-community/mm_grounding_dino_large_all").split("/")[-1]
    t = time.perf_counter()
    RecordCtx({fn: det}, extra).tool(fn, image=frame("gsota_0511"), query="cube. box. pen")
    load_s = time.perf_counter() - t
    runs = []
    for fname, fset in FRAME_SET.items():
        img = frame(fname)
        for cname, cfg in CONFIGS.items():
            kw = dict(cfg)
            if kw.get("roi") == "table":
                kw["roi"] = ROIS[fname]
            ctx = RecordCtx({fn: det}, extra)
            t = time.perf_counter()
            sahi.detect_sliced(ctx, img, PROMPTS[fset], backend=sahi_backend, min_score=low, **kw)
            total = time.perf_counter() - t
            for floor in floors:
                r = sahi.detect_sliced(
                    ReplayCtx(ctx.log),
                    img,
                    PROMPTS[fset],
                    backend=sahi_backend,
                    min_score=floor,
                    **kw,
                )
                runs.append(
                    {
                        "frame": fname,
                        "backend": tag,
                        "config": cname,
                        "floor": floor,
                        "detections": r["detections"],
                        "n_backend_calls": r["n_backend_calls"],
                        "timing": {"total_s": total, "per_call_s": ctx.secs},
                    }
                )
        print("done", fname, flush=True)
    # warm whole-frame latency, one direct call per frame x 5
    secs = []
    for fname, fset in FRAME_SET.items():
        img = frame(fname)
        for _ in range(5):
            t = time.perf_counter()
            det(image=img, query=". ".join(PROMPTS[fset]))
            secs.append(time.perf_counter() - t)
    out = {"runs": runs, "load_s": load_s, "gpu": gpu_mib(), "tag": tag, "call_s": secs}
    json.dump(out, open(OUT, "w"))
    print(
        "CHECK",
        tag,
        "load_s",
        round(load_s, 1),
        gpu_mib(),
        "call median",
        round(float(np.median(secs)), 4),
    )


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    main()
