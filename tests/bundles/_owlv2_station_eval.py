"""Live check of tools/owlv2 against grounding-dino on real YAM overhead frames. Not collected by
pytest; run on a GPU host from the repository root, one backend per process (VRAM is the process's own),
with FRAMES a folder of the seven 640 x 480 overhead frames named as in
tests/bundles/data/station_small_parts_labels.json (the sahi check's frames and hand labels):

  python tests/bundles/_owlv2_station_eval.py text owlv2 OUT/text_base.json FRAMES  # GAP_OWLV2_MODEL: base / large
  OWL_TEMPLATE='{}' python tests/bundles/_owlv2_station_eval.py text owlv2 OUT/text_base_bare.json FRAMES
  python tests/bundles/_owlv2_station_eval.py text grounding-dino OUT/text_gdino.json FRAMES
  python tests/bundles/_owlv2_station_eval.py like owlv2 OUT/like_base.json FRAMES
  python tests/bundles/_owlv2_station_score.py text|like OUT/*.json

The interpreter needs torch, transformers, pillow and gap-core (the owlv2 bundle's env).

Text mode: every frame in frames/ with its prompt set, through tools/sahi's detect_sliced (slice_px=0 is
the backend once on the frame / ROI), raw backend answers recorded at a low floor, then replayed per
score floor (sahi's min_score) so each floor is exactly one detect_sliced call. Exemplar mode: a cube /
plug / ring hand label as the exemplar, searched in the other frames (cross-frame), in its own frame
(within-frame, the rings), and in composites with crops of the other frames' cubes pasted on the table.
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRAMES = sys.argv[4] if len(sys.argv) > 4 else "frames"


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
#: The table in each frame, by eye (the same strips as the sahi PR's check).
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
    """ctx.tool -> the backend function in this process; records every answer for replay."""

    def __init__(self, fns, **extra):
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


def text_mode(backend, out_path):
    if backend == "owlv2":
        mod = load("owl", "tools/owlv2/tools.py")
        template = os.environ.get("OWL_TEMPLATE", mod.DEFAULT_TEMPLATE)
        fns = {"owlv2.detect": mod.detect}
        extra = {"owlv2.detect": {"threshold": 0.05, "template": template}}
        sahi_backend, floors = "owlv2.detect", (0.1, 0.15, 0.2, 0.3)
        tag = os.environ.get("GAP_OWLV2_MODEL", mod.DEFAULT_MODEL).split("/")[-1] + ":" + template
    else:
        mod = load("gdino", "tools/grounding-dino/tools.py")
        fns = {"grounding-dino.detect": mod.detect}
        extra = {}
        sahi_backend, floors = "grounding-dino", (0.2, 0.3)
        tag = "grounding-dino-base"
    t = time.perf_counter()
    warm = frame("gsota_0511")
    RecordCtx(fns, **extra).tool(next(iter(fns)), image=warm, query="cube")
    load_s = time.perf_counter() - t
    runs = []
    for fname, fset in FRAME_SET.items():
        img = frame(fname)
        for cname, cfg in CONFIGS.items():
            kw = dict(cfg)
            if kw.get("roi") == "table":
                kw["roi"] = ROIS[fname]
            ctx = RecordCtx(fns, **extra)
            t = time.perf_counter()
            low = 0.2 if backend == "grounding-dino" else 0.05
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
    json.dump({"runs": runs, "load_s": load_s, "gpu": gpu_mib(), "tag": tag}, open(out_path, "w"))
    print("CHECK", tag, "load_s", round(load_s, 1), gpu_mib())


def paste_composite(base_name, donors, seed):
    """``base_name``'s frame with crops (1.6x the label box) of the donors' cubes pasted on its table."""
    rng = np.random.default_rng(seed)
    img = frame(base_name).copy()
    H, W = img.shape[:2]
    roi = ROIS[base_name]
    taken = [np.array(g["box"], float) for g in LABELS[base_name]]
    gts = []
    for d in donors:
        g = next(x for x in LABELS[d] if x["label"] == "cube")
        x1, y1, x2, y2 = g["box"]
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 1.6, (y2 - y1) * 1.6
        sx1, sy1 = int(round(cx - w / 2)), int(round(cy - h / 2))
        crop = frame(d)[sy1 : sy1 + int(h), sx1 : sx1 + int(w)]
        ch, cw = crop.shape[:2]
        for _ in range(200):
            px = int(rng.integers(20, W - cw - 20))
            py = int(rng.integers(roi[1] + 10, H - ch - 60))
            nb = np.array([px, py, px + cw, py + ch], float)
            if all(
                min(nb[2], t[2]) - max(nb[0], t[0]) < -15
                or min(nb[3], t[3]) - max(nb[1], t[1]) < -15
                for t in taken
            ):
                break
        img[py : py + ch, px : px + cw] = crop
        ox, oy = x1 - sx1, y1 - sy1
        box = [px + ox, py + oy, px + ox + (x2 - x1), py + oy + (y2 - y1)]
        taken.append(np.array([px, py, px + cw, py + ch], float))
        gts.append({"label": "cube", "box": box, "from": d})
    return img, gts


def like_mode(out_path):
    mod = load("owl", "tools/owlv2/tools.py")
    tag = os.environ.get("GAP_OWLV2_MODEL", mod.DEFAULT_MODEL).split("/")[-1]
    fns = {"owlv2.detect_like": mod.detect_like}
    floor_low = float(os.environ.get("LIKE_LOW", "0.7"))
    t = time.perf_counter()
    mod.detect_like(frame("gsota_0511"), query_box=LABELS["gsota_0511"][0]["box"])
    load_s = time.perf_counter() - t
    cube_frames = [f for f, s in FRAME_SET.items() if s == "cube"]
    plug_frames = ["cable_1421", "setup_1103", "paired_1104"]
    runs = []

    def search(kind, qframe, qlabel, tframe, img, gts, ignore, query_token, exclude):
        for cname, cfg in CONFIGS.items():
            kw = dict(cfg)
            if kw.get("roi") == "table":
                kw["roi"] = ROIS[tframe]
            ctx = RecordCtx(fns, **{"owlv2.detect_like": {"threshold": floor_low}})
            t = time.perf_counter()
            sahi.detect_sliced(
                ctx, img, [query_token], backend="owlv2.detect_like", min_score=floor_low, **kw
            )
            total = time.perf_counter() - t
            for floor in (0.75, 0.8, 0.85, 0.87, 0.88, 0.9, 0.92):
                r = sahi.detect_sliced(
                    ReplayCtx(ctx.log),
                    img,
                    [query_token],
                    backend="owlv2.detect_like",
                    min_score=floor,
                    **kw,
                )
                dets = [{**d, "label": qlabel} for d in r["detections"]]
                runs.append(
                    {
                        "kind": kind,
                        "query": f"{qframe}:{qlabel}",
                        "frame": tframe,
                        "config": cname,
                        "floor": floor,
                        "detections": dets,
                        "gts": gts,
                        "ignore": ignore,
                        "exclude_box": exclude,
                        "n_backend_calls": r["n_backend_calls"],
                        "timing": {"total_s": total, "per_call_s": ctx.secs},
                    }
                )

    exemplars = {}
    for qf in cube_frames:
        g = next(x for x in LABELS[qf] if x["label"] == "cube")
        e = mod.embed_exemplar(frame(qf), g["box"])
        exemplars[(qf, "cube")] = (e, g["box"])
    for qf in plug_frames:
        g = next(x for x in LABELS[qf] if x["label"] == "plug")
        e = mod.embed_exemplar(frame(qf), g["box"])
        exemplars[(qf, "plug")] = (e, g["box"])
    for qf in ("cable_1421", "setup_1103"):
        rings = [x for x in LABELS[qf] if x["label"] == "ring"]
        for k, g in enumerate(rings):
            e = mod.embed_exemplar(frame(qf), g["box"])
            exemplars[(qf, f"ring{k}")] = (e, g["box"])
    ex_info = {f"{k[0]}:{k[1]}": v[0]["exemplar"] for k, v in exemplars.items()}
    print("exemplars", json.dumps(ex_info), flush=True)

    # cross-frame: an exemplar from one frame, searched in every other frame of its kind
    for label, frames_ in (("cube", cube_frames), ("plug", plug_frames)):
        for qf in frames_:
            e, _ = exemplars[(qf, label)]
            for tf in frames_:
                if tf == qf:
                    continue
                gts = [x for x in LABELS[tf] if x["label"] == label]
                others = [
                    x
                    for x in LABELS[tf]
                    if x["label"] != label and not x["label"].startswith("ignore")
                ]
                search("cross", qf, label, tf, frame(tf), gts, [], e["query"], None)
                runs[-1]["others"] = others
        print("cross done", label, flush=True)
    # within-frame: one ring as the exemplar, the other ring to find (the half-hidden one ignored)
    for qf in ("cable_1421", "setup_1103"):
        rings = [x for x in LABELS[qf] if x["label"] == "ring"]
        ign = [x for x in LABELS[qf] if x["label"] == "ignore:ring"]
        for k in range(len(rings)):
            e, qbox = exemplars[(qf, f"ring{k}")]
            search("within", qf, "ring", qf, frame(qf), [rings[1 - k]], ign, e["query"], qbox)
    print("within done", flush=True)
    # composites: each cube frame with the other three frames' cubes pasted on its table
    for qf in cube_frames:
        donors = [f for f in cube_frames if f != qf]
        img, gts = paste_composite(qf, donors, seed=7)
        Image.fromarray(img).save(
            os.path.join(os.path.dirname(out_path) or ".", f"composite_{qf}.png")
        )
        e, qbox = exemplars[(qf, "cube")]
        search("composite", qf, "cube", qf, img, gts, [], e["query"], qbox)
    print("composite done", flush=True)

    # the one-call path without sahi (query box in the same image) for latency
    img = frame("cable_1421")
    rings = [x for x in LABELS["cable_1421"] if x["label"] == "ring"]
    secs = []
    for _ in range(10):
        t = time.perf_counter()
        mod.detect_like(img, query_box=rings[0]["box"], exclude_query=True, threshold=0.88)
        secs.append(time.perf_counter() - t)
    secs_x = []
    e, _ = exemplars[("cable_1421", "ring0")]
    for _ in range(10):
        t = time.perf_counter()
        mod.detect_like(frame("setup_1103"), query=e["query"], threshold=0.88)
        secs_x.append(time.perf_counter() - t)
    json.dump(
        {
            "runs": runs,
            "load_s": load_s,
            "gpu": gpu_mib(),
            "tag": tag,
            "exemplars": ex_info,
            "same_image_call_s": secs,
            "token_call_s": secs_x,
        },
        open(out_path, "w"),
    )
    print(
        "CHECK like",
        tag,
        "load_s",
        round(load_s, 1),
        gpu_mib(),
        "same-image call median",
        np.median(secs),
    )


if __name__ == "__main__":
    os.makedirs(os.path.dirname(sys.argv[3]) or ".", exist_ok=True)
    mode, backend, out = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == "text":
        text_mode(backend, out)
    else:
        like_mode(out)
