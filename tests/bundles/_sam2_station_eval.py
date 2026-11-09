"""Live check of tools/sam2's streaming tracker on two real YAM clips. Not collected by pytest;
on a GPU host from the repository root, one model per process:

  python tests/bundles/_sam2_station_eval.py extract RECORDINGS CLIPS      # mp4 -> CLIPS/<clip>/NNNN.jpg
  python tests/bundles/_sam2_station_eval.py MODEL OUT.json CLIPS         # MODEL: one of tools/sam2 MODELS

RECORDINGS is the station's recordings folder (the clips' sources are in
tests/bundles/data/sam2_station_clips_labels.json). Each labelled object is seeded with its
hand-labelled box on its prompt frame through sam2.tracker_init, then every following frame goes
through sam2.tracker_update (the live path a closed loop uses). The answer on each labelled keyframe
is scored against the label; latency per update, peak and per-frame GPU memory are recorded.
NOPRUNE=1 disables the session pruning (the memory control); ONLY_CLIP=<clip> runs one clip.
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LABELS = os.path.join(REPO, "tests/bundles/data/sam2_station_clips_labels.json")
TOOLS = os.path.join(REPO, "tools/sam2/tools.py")


def extract(recordings, clips):
    import cv2

    for clip, cfg in json.load(open(LABELS)).items():
        if clip.startswith("_"):
            continue
        src = cfg["source"]
        cap = cv2.VideoCapture(
            os.path.join(recordings, src["recording"].split("recordings/", 1)[1])
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, src["first_frame"])
        os.makedirs(os.path.join(clips, clip), exist_ok=True)
        for i in range(src["n_frames"]):
            ok, im = cap.read()
            if not ok:
                break
            cv2.imwrite(
                os.path.join(clips, clip, f"{i:04d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )


if sys.argv[1] == "extract":
    extract(sys.argv[2], sys.argv[3])
    sys.exit(0)
MODEL, OUT, CLIPS = sys.argv[1:4]
os.environ["GAP_SAM2_MODEL"] = MODEL
spec = importlib.util.spec_from_file_location("sam2_tools", TOOLS)
et = importlib.util.module_from_spec(spec)
spec.loader.exec_module(et)
if os.environ.get("NOPRUNE"):
    et._prune = lambda *a, **k: None  # control: the session keeps every frame
import torch  # noqa: E402

labels = {k: v for k, v in json.load(open(LABELS)).items() if not k.startswith("_")}


def frames(clip):
    d = os.path.join(CLIPS, clip)
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".jpg")]


def load(p):
    return np.asarray(Image.open(p).convert("RGB"))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


t0 = time.perf_counter()
et._get_runtime()
load_s = time.perf_counter() - t0
torch.cuda.reset_peak_memory_stats()
base_alloc = torch.cuda.memory_allocated()
runs = []
ONLY = os.environ.get("ONLY_CLIP")
for clip, cfg in labels.items():
    if ONLY and clip != ONLY:
        continue
    paths = frames(clip)
    for name, obj in cfg["objects"].items():
        pf = obj["prompt_frame"]
        prompt = obj["keys"][str(pf)]
        t = time.perf_counter()
        init = et.tracker_init(load(paths[pf]), box=prompt, object_name=name)
        init_s = time.perf_counter() - t
        tid = init["tracker_id"]
        per = {
            pf: {
                "box": init["initial_box"],
                "present": init["object_present"],
                "conf": init["score"],
            }
        }
        secs, mem = [], []
        for i in range(pf + 1, len(paths)):
            img = load(paths[i])
            torch.cuda.synchronize()
            t = time.perf_counter()
            r = et.tracker_update(tid, img)
            torch.cuda.synchronize()
            secs.append(time.perf_counter() - t)
            mem.append(int(torch.cuda.memory_allocated() - base_alloc))
            per[i] = {"box": r["box"], "present": r["object_present"], "conf": r["confidence"]}
        et.tracker_close(tid)
        keys = []
        for k, v in obj["keys"].items():
            k = int(k)
            if v is None or k == pf:
                continue
            occl = isinstance(v, dict)
            gt = v["occluded"] if occl else v
            p = per[k]
            b = p["box"]
            v_iou = iou([b["x1"], b["y1"], b["x2"], b["y2"]], gt) if b else 0.0
            keys.append(
                {
                    "frame": k,
                    "occluded": occl,
                    "iou": v_iou,
                    "present": p["present"],
                    "conf": p["conf"],
                }
            )
        hit = [x for x in keys if not x["occluded"]]
        n_hit = sum(x["iou"] >= obj["iou_hit"] for x in hit)
        n_occ = sum(x["iou"] >= 0.3 for x in keys if x["occluded"])
        runs.append(
            {
                "clip": clip,
                "object": name,
                "keys": keys,
                "hits": n_hit,
                "n": len(hit),
                "occluded_kept": n_occ,
                "n_occluded": sum(x["occluded"] for x in keys),
                "mean_iou": float(np.mean([x["iou"] for x in hit])),
                "init_s": init_s,
                "update_s_median": float(np.median(secs)),
                "update_s_p90": float(np.percentile(secs, 90)),
                "mem_mib_at": {
                    str(j): mem[j] / 2**20 for j in (0, 99, len(mem) - 1) if j < len(mem)
                },
                "trace": {str(i): per[i] for i in sorted(per)},
            }
        )
        print(
            f"CHECK {MODEL.split('/')[-1]} {clip}:{name} hits {n_hit}/{len(hit)} "
            f"occluded-kept {n_occ} mean IoU {runs[-1]['mean_iou']:.2f} "
            f"update {runs[-1]['update_s_median'] * 1000:.1f} ms mem {runs[-1]['mem_mib_at']}",
            flush=True,
        )
json.dump(
    {
        "model": MODEL,
        "load_s": load_s,
        "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
        "runs": runs,
    },
    open(OUT, "w"),
)
print("CHECK done", MODEL, "peak MiB", round(torch.cuda.max_memory_allocated() / 2**20))
