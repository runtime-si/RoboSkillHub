"""Live check of tools/dam4sam against tools/sam2 (same SAM 2.1 base-plus weights).

1. Look-alike "shell games" on a real frame: frame 0 of the overhead station clip
   (tests/bundles/data/sam2_station_clips_labels.json, top148) as the table; a real part cut from it
   (the cube, 22 x 27 px, or the box, 55 x 62 px) pasted three times; the target and one twin trade
   places along one line (they cross mid-clip, the target under the twin in odd seeds), the third copy
   sits beside the path; 60 frames, 2 px jitter. Ground truth: the target's box. Scored: mean box IoU,
   frames after the crossing with IoU < 0.3 ("lost"), and whether the last 10 frames hold IoU >= 0.5.
2. Pen caps among look-alike caps in two real wrist clips of the pens task (21 tracks, 151 frames
   at 7.5 fps; ground truth by homography from one labelled frame, tests/bundles/_pens_lookalike_gt.py):
   mean IoU, IoU by quarter of the clip, identity switches (3+ frames with the tracked box's centre
   on another cap and IoU < 0.3 with the target), lost frames.
3. The labelled real clips (top148 box / cube, wrist574 plug): the hit rate at the keyframes
   (IoU >= iou_hit, as tools/sam2's check).

    python tests/bundles/_dam4sam_eval.py RECORDINGS_DIR PENS_RUNS_DIR OUT.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "tests/bundles/data/sam2_station_clips_labels.json"
PENS = ROOT / "tests/bundles/data/pens_lookalike_tracks.json"


def _stub() -> None:
    try:
        import gap_core.tools  # noqa: F401
    except ImportError:
        gc = types.ModuleType("gap_core")
        err = types.ModuleType("gap_core.errors")
        err.PerceptionFailed = type("PerceptionFailed", (RuntimeError,), {})
        err.ToolError = type("ToolError", (RuntimeError,), {})
        tl = types.ModuleType("gap_core.tools")
        tl.tool = lambda **kw: lambda f: f
        sys.modules.update({"gap_core": gc, "gap_core.errors": err, "gap_core.tools": tl})


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_clip(path: Path, first: int, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out, i = [], 0
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        if i >= first:
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    return out


def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def shell_game(bg: np.ndarray, part_box, seed: int, n: int = 60):
    rng = np.random.default_rng(seed)
    x1, y1, x2, y2 = part_box
    patch = bg[y1:y2, x1:x2].copy()
    ph, pw = patch.shape[:2]
    h, w = bg.shape[:2]
    # a clean table under the moving parts: the background with the part's own spot inpainted
    hole = np.zeros((h, w), np.uint8)
    hole[y1 - 2 : y2 + 2, x1 - 2 : x2 + 2] = 255
    table = cv2.inpaint(bg, hole, 5, cv2.INPAINT_TELEA)
    ang = rng.uniform(0, np.pi)
    d = np.array([np.cos(ang), np.sin(ang)])
    span = rng.uniform(2.5, 4.0) * max(pw, ph)
    c = np.array([rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h])
    a, b = c - d * span / 2, c + d * span / 2
    side = c + np.array([-d[1], d[0]]) * 1.6 * max(pw, ph)
    under = seed % 2 == 1
    frames, gt, twins = [], [], []
    for t in range(n):
        s = np.clip((t - 10) / (n - 20), 0, 1)
        pt = a + (b - a) * s + rng.normal(0, 1.0, 2)
        pd = b + (a - b) * s + rng.normal(0, 1.0, 2)
        img = table.copy()

        def paste(p, img=img):
            px, py = int(round(p[0] - pw / 2)), int(round(p[1] - ph / 2))
            px, py = min(max(px, 0), w - pw), min(max(py, 0), h - ph)
            img[py : py + ph, px : px + pw] = patch
            return [px, py, px + pw, py + ph]

        paste(side)
        if under:
            tb = paste(pt)
            db = paste(pd)
        else:
            db = paste(pd)
            tb = paste(pt)
        frames.append(img)
        gt.append(tb)
        twins.append(db)
    return frames, gt, twins, n // 2


def score_game(boxes, gt, twins, cross):
    ious = [iou(p, g) for p, g in zip(boxes, gt, strict=True)]
    lost = [i < 0.3 for i in ious[cross + 3 :]]
    swapped = any(
        iou(p, d) > 0.5 and iou(p, g) < 0.3
        for p, g, d in zip(boxes[-10:], gt[-10:], twins[-10:], strict=True)
    )
    return {"miou": float(np.mean(ious)), "lost_after": float(np.mean(lost)),
            "kept": bool(np.mean(ious[-10:]) >= 0.5 and min(ious[-10:]) > 0.3), "swapped": swapped}  # fmt: skip


def _centre_in(b, g, shrink=0.15) -> bool:
    if b is None or g is None:
        return False
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    dx, dy = (g[2] - g[0]) * shrink, (g[3] - g[1]) * shrink
    return g[0] + dx <= cx <= g[2] - dx and g[1] + dy <= cy <= g[3] - dy


def run_pens(pens: Path, trackers):
    """Each labelled cap of tests/bundles/data/pens_lookalike_tracks.json as the target in turn."""
    lab = json.loads(PENS.read_text())
    res = {}
    for clip, spec in lab["clips"].items():
        frames = read_clip(pens / spec["video"], lab["first_frame"], 301)[:: lab["stride"]]
        frames = frames[: spec["n_frames"]]
        gts = spec["boxes"]
        for tname, fn in trackers.items():
            rows = []
            for cap, gt in gts.items():
                boxes = fn(frames, gt[0])
                ious, sw, lost = [], [], []
                for i in range(1, len(frames)):
                    if gt[i] is None:
                        continue
                    u = iou(boxes[i], gt[i])
                    other = any(_centre_in(boxes[i], g[i]) for c, g in gts.items() if c != cap)
                    ious.append(u)
                    sw.append(u < 0.3 and other)
                    lost.append(u < 0.3 and not other)
                run, switched = 0, False
                for x in sw:
                    run = run + 1 if x else 0
                    switched |= run >= 3
                rows.append({"cap": cap, "miou": float(np.mean(ious)), "switch_frames": float(np.mean(sw)),
                             "lost_frames": float(np.mean(lost)), "switched": switched,
                             "iou_q": [float(np.mean(q)) for q in np.array_split(ious, 4)]})  # fmt: skip
            agg = {"tracks": len(rows), "miou": float(np.mean([r["miou"] for r in rows])),
                   "id_switch_tracks": int(sum(r["switched"] for r in rows)),
                   "switch_frames": float(np.mean([r["switch_frames"] for r in rows])),
                   "lost_frames": float(np.mean([r["lost_frames"] for r in rows])),
                   "iou_by_quarter": [float(np.mean([r["iou_q"][k] for r in rows])) for k in range(4)]}  # fmt: skip
            res[f"{clip}/{tname}"] = {"agg": agg, "tracks": rows}
            print(f"pens {clip} {tname}: {agg}", flush=True)
    return res


def run_clips(rec: Path, trackers):
    lab = json.loads(LABELS.read_text())
    res = {}
    for clip, spec in lab.items():
        if clip.startswith("_"):
            continue
        src = spec["source"]
        frames = read_clip(rec / Path(src["recording"]).relative_to("yam_gap/real/recordings"),
                           src["first_frame"], src["n_frames"])  # fmt: skip
        for obj, o in spec["objects"].items():
            for tname, fn in trackers.items():
                boxes = fn(frames, o["keys"][str(o["prompt_frame"])])
                hits, n = 0, 0
                for k, v in o["keys"].items():
                    if v is None or int(k) >= len(boxes) or int(k) == o["prompt_frame"]:
                        continue
                    if isinstance(v, dict):
                        g, thr = v["occluded"], 0.3
                    else:
                        g, thr = v, o["iou_hit"]
                    n += 1
                    hits += iou(boxes[int(k)], g) >= thr
                res[f"{clip}/{obj}/{tname}"] = f"{hits}/{n}"
                print(f"clip {clip}/{obj} {tname}: {hits}/{n}", flush=True)
    return res


def main() -> None:
    rec, pens, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    _stub()
    dam = _load("dam4sam_tools", ROOT / "tools/dam4sam/tools.py")
    sam2 = _load("sam2_tools", ROOT / "tools/sam2/tools.py")

    def t_dam(frames, box):
        return dam.track(frames, box=box)["boxes"]

    def t_sam2(frames, box):
        return [None if b is None else [b["x1"], b["y1"], b["x2"], b["y2"]] if isinstance(b, dict) else b
                for b in sam2.track(frames, box=box)["boxes"]]  # fmt: skip

    trackers = {"dam4sam": t_dam, "sam2": t_sam2}
    lab = json.loads(LABELS.read_text())
    bg = read_clip(rec / "20261103_082451/top_148.mp4", 0, 1)[0]
    parts = {"cube": lab["top148"]["objects"]["cube"]["keys"]["0"],
             "box": lab["top148"]["objects"]["box"]["keys"]["0"]}  # fmt: skip
    res: dict = {"games": {}, "clips": {}}
    res["pens"] = run_pens(pens, trackers)
    out.write_text(json.dumps(res, indent=1))
    for pname, pbox in parts.items():
        for tname, fn in trackers.items():
            rows = []
            t0 = time.time()
            for seed in range(10):
                frames, gt, twins, cross = shell_game(bg, pbox, seed)
                rows.append(score_game(fn(frames, gt[0]), gt, twins, cross))
            agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            res["games"][f"{pname}/{tname}"] = agg
            print(f"game {pname} {tname}: {agg} ({time.time() - t0:.0f}s)", flush=True)
            if pname == "cube" and tname == "dam4sam":
                cv2.imwrite(str(out.with_suffix(".game.jpg")), cv2.cvtColor(np.concatenate(
                    [frames[0], frames[cross], frames[-1]], axis=1), cv2.COLOR_RGB2BGR))  # fmt: skip
    res["clips"] = run_clips(rec, trackers)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
