"""Score _owlv2_station_eval.py's outputs against the hand labels (the SKILL.md tables).

    python tests/bundles/_owlv2_station_score.py text OUT/text_*.json
    python tests/bundles/_owlv2_station_score.py like OUT/like_*.json

A detection is a hit at IoU >= 0.3 with an unmatched label of the same name (best score first);
over an "ignore:" region it counts neither way; the exemplar's own box (exemplar mode in its own
frame) is skipped; every other box is a false positive.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

LABELS = json.load(
    open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "station_small_parts_labels.json"
        )
    )
)
SMALL = {"cube", "pen", "plug", "ring"}
IOU_HIT = 0.3


def xyxy(b):
    return (
        np.array([b["x1"], b["y1"], b["x2"], b["y2"]], float)
        if isinstance(b, dict)
        else np.array(b, float)
    )


def inter(a, b):
    return max(min(a[2], b[2]) - max(a[0], b[0]), 0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0)


def area(a):
    return (a[2] - a[0]) * (a[3] - a[1])


def iou(a, b):
    i = inter(a, b)
    return i / (area(a) + area(b) - i)


def match(dets, gts, ignore, exclude=None):
    used = [False] * len(gts)
    fps = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        b = xyxy(d["box"])
        if exclude is not None and iou(b, xyxy(exclude)) >= 0.5:
            continue
        best, bi = IOU_HIT, -1
        for i, g in enumerate(gts):
            if used[i] or g["label"] != d["label"]:
                continue
            v = iou(b, xyxy(g["box"]))
            if v >= best:
                best, bi = v, i
        if bi >= 0:
            used[bi] = True
            continue
        if any(inter(b, xyxy(g["box"])) / area(b) >= 0.5 for g in ignore):
            continue
        fps.append(d)
    return used, fps


def text(paths):
    agg = defaultdict(lambda: np.zeros(8))
    meta = {}
    for p in paths:
        data = json.load(open(p))
        meta[data["tag"]] = (data["load_s"], data["gpu"])
        for run in data["runs"]:
            gts = [g for g in LABELS[run["frame"]] if not g["label"].startswith("ignore:")]
            ign = [
                {"box": g["box"]} for g in LABELS[run["frame"]] if g["label"] == "ignore:" + "ring"
            ]
            used, fps = match(run["detections"], gts, ign)
            sm = [g["label"] in SMALL for g in gts]
            cubes = [
                g["label"] == "cube" and g.get("box")[3] < 400 and run["frame"].startswith("gsota")
                for g in gts
            ]
            key = (run["backend"], run["config"], run["floor"])
            agg[key] += [
                sum(u and s for u, s in zip(used, sm, strict=True)),
                sum(sm),
                sum(u and not s for u, s in zip(used, sm, strict=True)),
                sum(not s for s in sm),
                len(fps),
                run["timing"]["total_s"],
                sum(u and c for u, c in zip(used, cubes, strict=True)),
                1,
            ]
    for tag, (load_s, gpu) in meta.items():
        print(f"{tag}: load {load_s:.1f} s, torch peak {gpu}")
    print(
        f"{'backend':48} {'config':9} {'floor':>5} {'small':>6} {'cubes@1m':>8} {'boxes':>5} {'FP':>4} {'s/frame':>7}"
    )
    for key in sorted(agg):
        a = agg[key]
        print(
            f"{key[0]:48} {key[1]:9} {key[2]:5.2f} {int(a[0]):2d}/{int(a[1]):<3d} {int(a[6]):4d}/3   "
            f"{int(a[2])}/{int(a[3])}   {int(a[4]):4d} {a[5] / a[7]:7.3f}"
        )


def like(paths):
    for p in paths:
        data = json.load(open(p))
        print(f"== {data['tag']}: load {data['load_s']:.1f} s, torch peak {data['gpu']}")
        print(
            "exemplars (model box IoU with the hand label):",
            {k: v["iou"] for k, v in data["exemplars"].items()},
        )
        print(
            f"one call, query box in the same frame: median {np.median(data['same_image_call_s']):.3f} s; "
            f"with an exemplar string: {np.median(data['token_call_s']):.3f} s"
        )
        agg = defaultdict(lambda: np.zeros(7))
        for run in data["runs"]:
            used, fps = match(run["detections"], run["gts"], run["ignore"], run["exclude_box"])
            others = (
                [g for g in LABELS[run["frame"]] if g["label"] in ("box",)]
                if run["kind"] != "within"
                else []
            )
            fp_box = sum(
                any(iou(xyxy(d["box"]), xyxy(o["box"])) >= 0.3 for o in others) for d in fps
            )
            key = (
                run["kind"],
                run["query"].split(":")[1].rstrip("01"),
                run["config"],
                run["floor"],
            )
            agg[key] += [
                sum(used),
                len(used),
                len(fps),
                fp_box,
                run["timing"]["total_s"],
                run["n_backend_calls"],
                1,
            ]
        print(
            f"{'kind':9} {'obj':5} {'config':9} {'floor':>5} {'found':>7} {'FP':>4} {'FP=box':>6} {'s/frame':>7} {'calls':>5} {'searches':>8}"
        )
        for key in sorted(agg):
            a = agg[key]
            print(
                f"{key[0]:9} {key[1]:5} {key[2]:9} {key[3]:5.2f} {int(a[0]):3d}/{int(a[1]):<3d} {int(a[2]):4d} {int(a[3]):6d} "
                f"{a[4] / a[6]:7.3f} {a[5] / a[6]:5.1f} {int(a[6]):8d}"
            )


if __name__ == "__main__":
    (text if sys.argv[1] == "text" else like)(sys.argv[2:])
