"""Score target_eval.py outputs: per detector / set / view / class, recall at IoU >= 0.5 (and 0.3) and FPs per image.

TARGET_PARTS_DIR=WORK python tests/bundles/_target_parts_score.py OUT/*.json   # writes WORK/scores.json
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.environ.get("TARGET_PARTS_DIR", "target_parts")
SETS = {"render": "data/render", "oddone": "data/oddone", "jigsaw": "data/jigsaw"}
PROMPT_OF = {
    "lego": "lego brick",
    "duplo": "lego brick",
    "domino": "domino",
    "jigsaw": "jigsaw puzzle piece",
}
GT = {}
for ds, d in SETS.items():
    for s in json.load(open(os.path.join(HERE, d, "labels.json")))["scenes"]:
        GT[(ds, s["image"])] = s


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def box(d):
    b = d["box"]
    return [b["x1"], b["y1"], b["x2"], b["y2"]] if isinstance(b, dict) else list(b)


def snap(label, prompts):
    low = label.lower().strip(" .")
    for p in prompts:
        if p == low or p in low or (low and low in p):
            return p
    return low


def match(dets, gts, thr, label_of, cls_ok):
    """Greedy best-score-first matching; returns per-GT hit flags and the FP count."""
    used = [False] * len(gts)
    fp = 0
    for d in sorted(dets, key=lambda d: -d["score"]):
        b = box(d)
        best, bi = thr, -1
        for i, g in enumerate(gts):
            if used[i] or not cls_ok(d, g):
                continue
            v = iou(b, g["box"])
            if v >= best:
                best, bi = v, i
        if bi >= 0:
            used[bi] = True
        else:
            fp += 1
    return used, fp


def size_bin(g):
    w = min(g["box"][2] - g["box"][0], g["box"][3] - g["box"][1])
    return "<12px" if w < 12 else "12-24px" if w < 24 else ">=24px"


def detectors(paths, floors=(0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)):
    rows = []
    for p in paths:
        data = json.load(open(p))
        if "runs" not in data or data["backend"] == "owlv2-like":
            continue
        name = os.path.basename(p)[:-5]
        for floor in floors:
            agg = defaultdict(
                lambda: [0, 0, 0, 0, 0.0, 0]
            )  # hit50, hit30, n, fp50(image), secs, images
            for r in data["runs"]:
                s = GT[(r["set"], r["image"])]
                prompts = sorted(
                    {PROMPT_OF[o["cls"]] for o in s["objects"]}
                    | ({"lego brick", "domino"} if r["set"] != "jigsaw" else set())
                )
                dets = [d for d in r["detections"] if d["score"] >= floor]
                for d in dets:
                    d["_p"] = snap(d.get("label", ""), prompts)
                for thr, k in ((0.5, 0), (0.3, 1)):
                    used, fp = match(
                        dets, s["objects"], thr, None, lambda d, g: d["_p"] == PROMPT_OF[g["cls"]]
                    )
                    view = r["view"] or r["set"]
                    for g, u in zip(s["objects"], used, strict=True):
                        for key in (
                            (r["set"], view, r["config"], g["cls"]),
                            (r["set"], view, r["config"], g["cls"] + ":" + size_bin(g)),
                        ):
                            agg[key][k] += u
                            if k == 0:
                                agg[key][2] += 1
                    if k == 0:
                        a = agg[(r["set"], view, r["config"], "_images")]
                        a[3] += fp
                        a[4] += r["s"]
                        a[5] += 1
            for key, a in agg.items():
                rows.append(
                    {
                        "det": name,
                        "floor": floor,
                        "set": key[0],
                        "view": key[1],
                        "config": key[2],
                        "cls": key[3],
                        "r50": a[0] / a[2] if a[2] else None,
                        "r30": a[1] / a[2] if a[2] else None,
                        "n": a[2],
                        "fp_per_img": a[3] / a[5] if a[5] else None,
                        "s": a[4] / a[5] if a[5] else None,
                    }
                )
    return rows


def like(paths, floors=(0.8, 0.85, 0.87, 0.9)):
    rows = []
    for p in paths:
        data = json.load(open(p))
        if data.get("backend") != "owlv2-like":
            continue
        for floor in floors:
            agg = defaultdict(
                lambda: [0, 0, 0, 0, 0]
            )  # hit, n, fp, same-class-other-hits, searches
            for r in data["runs"]:
                s = GT[(r["set"], r["image"])]
                others = [g for g in s["objects"] if g["box"] != r["query"]]
                dets = [d for d in r["detections"] if d["score"] >= floor]
                used, fp = match(dets, others, 0.5, None, lambda d, g: True)
                view = r["view"] or r["set"]
                a = agg[(r["set"], view, r["cls"])]
                for g, u in zip(others, used, strict=True):
                    if g["cls"] == r["cls"]:
                        a[0] += u
                        a[1] += 1
                    else:
                        a[3] += u
                a[2] += fp
                a[4] += 1
            for key, a in agg.items():
                rows.append(
                    {
                        "floor": floor,
                        "set": key[0],
                        "view": key[1],
                        "cls": key[2],
                        "recall_same_class": a[0] / a[1] if a[1] else None,
                        "n": a[1],
                        "other_class_hits": a[3],
                        "fp": a[2],
                        "searches": a[4],
                    }
                )
    return rows


def sam2(paths):
    rows = []
    for p in paths:
        data = json.load(open(p))
        if data.get("backend") != "sam2":
            continue
        agg = defaultdict(list)
        for r in data["rows"]:
            agg[(r["set"], r["view"], r["cls"])].append(r["iou"])
            agg[(r["set"], r["view"], r["sub"])].append(r["iou"])
        for k, v in agg.items():
            v = np.array(v)
            rows.append(
                {
                    "set": k[0],
                    "view": k[1],
                    "cls": k[2],
                    "n": len(v),
                    "mean_iou": float(v.mean()),
                    "iou>=0.75": float((v >= 0.75).mean()),
                    "iou<0.5": float((v < 0.5).mean()),
                }
            )
    return rows


if __name__ == "__main__":
    out = {
        "detectors": detectors(sys.argv[1:]),
        "like": like(sys.argv[1:]),
        "sam2": sam2(sys.argv[1:]),
    }
    json.dump(out, open(os.path.join(HERE, "scores.json"), "w"), indent=1)
    print(len(out["detectors"]), len(out["like"]), len(out["sam2"]))
