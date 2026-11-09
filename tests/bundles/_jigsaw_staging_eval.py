"""segmenting-and-picking-pieces-classical on synthetic staging scenes: segmentation IoU, split
accuracy of touching pairs, pick points inside the piece, angle and face. Not collected by
pytest; run from the repo root:

    .venv/bin/python tests/bundles/_jigsaw_staging_eval.py --scenes 40 --first-seed 1000

Each scene (``_jigsaw_staging.make_scene``) is 640 x 480 with 6 isolated pieces, 2 touching
pairs, 1 flush (still interlocked) pair and 1 stacked pair, 15 % of the loose pieces face down,
on a mat cycling through green, magenta, blue, grey, black and white, at a given piece size in
pixels. The skill runs with the frame's depth and pose through a stand-in for the geometry
tools (the same back-projection), ``back_rgb`` set, nothing else.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _jigsaw_staging as S  # noqa: E402

SCRIPT = (
    HERE.parents[1] / "skills/segmenting-and-picking-pieces-classical/scripts/segment_pieces.py"
)
MATS = ("green", "magenta", "blue", "grey", "black", "white")


def _module():
    spec = importlib.util.spec_from_file_location("segment_pieces", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pose_matrix(pose):
    from gap_core.types import pose_to_matrix

    return pose_to_matrix(pose)


class GeometryCtx:
    """geometry.mask_to_world_points / pixel_to_world_point, same maths as tools/geometry."""

    def tool(self, name, **kw):
        K, T, depth = np.asarray(kw["intrinsics"]), _pose_matrix(kw["camera_pose"]), kw["depth"]
        if name == "geometry.mask_to_world_points":
            v, u = np.nonzero((kw["mask"] > 0) & np.isfinite(depth) & (depth >= 0.015))
            z = depth[v, u]
            pc = np.column_stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
            return {"points": {"points": (pc @ T[:3, :3].T + T[:3, 3]).tolist()}}
        u, v = int(round(kw["pixel_x"])), int(round(kw["pixel_y"]))
        z = float(depth[v, u])
        p = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
        q = T[:3, :3] @ p + T[:3, 3]
        return {"point": {"x": q[0], "y": q[1], "z": q[2]}}


def one(args):
    seed, px, mat, noise = args
    mod = _module()
    s = S.make_scene(
        seed,
        px_per_piece=px,
        n_single=6,
        n_touch=2,
        n_flush=1,
        n_stacked=1,
        mat=mat,
        depth_noise=max(noise, 0.0),
    )
    cam = S.camera_frame(s)
    kw = {"piece_size_m": s.cell_m}
    if noise < 0:  # no depth at all: the scale comes from the median blob
        cam = {k: v for k, v in cam.items() if k != "depth"}
        kw = {"table_z": 0.0}
    t = time.perf_counter()
    out = mod.run(GeometryCtx(), camera=cam, back_rgb=list(S.BACK), **kw)
    dt = time.perf_counter() - t
    h, w = s.rgb.shape[:2]
    pred = []
    for p in out["pieces"]:
        f = np.zeros((h, w), bool)
        x0, y0, x1, y1 = p["bbox_px"]
        f[y0:y1, x0:x1] = p["mask_crop"] > 0
        pred.append(f)
    pairs = S.match(pred, s.masks)
    by_gt = {b: (a, iou) for a, b, iou in pairs if iou >= 0.5}
    rows = []
    import cv2

    for g, gm in enumerate(s.masks):
        a, iou = by_gt.get(g, (None, 0.0))
        rec = {
            "kind": s.kinds[g],
            "group": s.groups[g],
            "iou": iou,
            "found": a is not None,
            "face_gt": s.faces[g],
        }
        if a is not None:
            p = out["pieces"][a]
            u, v = int(round(p["pick_pixel"]["x"])), int(round(p["pick_pixel"]["y"]))
            dtg = cv2.distanceTransform(np.pad(gm.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            rec["inside"] = bool(gm[v, u])
            rec["angle_err"] = _angle_err(p["angle_deg"], s.angles[g])
            rec["depth_in"] = float(dtg[v, u]) / px  # clearance from the true outline, sides
            rec["touching"] = p["touching"]
            rec["pickable"] = p["pickable"]
            rec["face"] = p["face"]
            rec["angle"] = p["angle_deg"]
            rec["stacked"] = p["stacked"]
            rec["pos"] = p["position"]
        rows.append(rec)
    n_extra = len(pred) - len(by_gt)
    return {
        "rows": rows,
        "extra": n_extra,
        "route": out["route"],
        "time": dt,
        "seed": seed,
        "mat": mat,
    }


def _angle_err(a, b):
    d = (a - b) % 90.0
    return min(d, 90.0 - d)


def report(results, px, noise):
    rows = [r for res in results for r in res["rows"]]
    kinds = ("single", "touch", "flush", "stacked_top", "stacked_bottom")
    cond = "no depth" if noise < 0 else f"depth noise {noise * 1000:.1f} mm"
    print(f"\n## {px:.0f} px per piece side, {cond}, {len(results)} scenes, {len(rows)} pieces")
    print(
        "| kind | n | found (IoU>=0.5) | mean IoU (found) | pick inside | pick >= 0.1 side in | touching flag |"
    )
    print("|---|---|---|---|---|---|---|")
    for k in kinds:
        rr = [r for r in rows if r["kind"] == k]
        if not rr:
            continue
        f = [r for r in rr if r["found"]]
        exp_touch = k != "single"
        print(
            f"| {k} | {len(rr)} | {len(f) / len(rr):.1%} | "
            f"{np.mean([r['iou'] for r in f]) if f else 0:.3f} | "
            f"{np.mean([r['inside'] for r in f]) if f else 0:.1%} | "
            f"{np.mean([r['depth_in'] >= 0.1 for r in f]) if f else 0:.1%} | "
            f"{np.mean([r['touching'] == exp_touch for r in f]) if f else 0:.1%} |"
        )
    # split accuracy: both pieces of a pair found with IoU >= 0.7
    for k in ("touch", "flush"):
        grp = defaultdict(list)
        for res in results:
            for r in res["rows"]:
                if r["kind"] == k:
                    grp[(res["seed"], r["group"])].append(r)
        ok = [all(r["found"] and r["iou"] >= 0.7 for r in g) for g in grp.values() if len(g) == 2]
        if ok:
            print(f"- {k} pairs split (both IoU >= 0.7): {np.mean(ok):.1%} of {len(ok)}")
    st = [r for r in rows if r["kind"] == "stacked_top" and r["found"]]
    if st:
        print(
            f"- top piece of a stack flagged stacked: {np.mean([r['stacked'] for r in st]):.1%}"
            f", pickable {np.mean([r['pickable'] for r in st]):.1%} of {len(st)} found"
        )
    sb = [r for r in rows if r["kind"] == "stacked_bottom" and r["found"]]
    if sb:
        print(
            f"- covered bottom piece not pickable: {np.mean([not r['pickable'] for r in sb]):.1%} of {len(sb)} found"
        )
    sing = [r for r in rows if r["kind"] == "single" and r["found"]]
    gt_up = [r for r in sing if r["face_gt"] == "up"]
    gt_dn = [r for r in sing if r["face_gt"] == "down"]
    print(f"- isolated pieces pickable: {np.mean([r['pickable'] for r in sing]):.1%}")
    ae = [r["angle_err"] for r in sing]
    print(
        f"- side angle error, isolated pieces: median {np.median(ae):.1f} deg, "
        f"within 5 deg {np.mean(np.array(ae) <= 5):.1%}"
    )
    print(
        f"- face: up read up {np.mean([r['face'] == 'up' for r in gt_up]):.1%} of {len(gt_up)}, "
        f"down read down {np.mean([r['face'] == 'down' for r in gt_dn]) if gt_dn else 0:.1%} of {len(gt_dn)}"
    )
    print(f"- extra predicted pieces per scene: {np.mean([r['extra'] for r in results]):.2f}")
    t = [r["time"] for r in results]
    print(f"- time per frame: median {np.median(t):.2f} s, max {np.max(t):.2f} s")
    by_mat = defaultdict(list)
    for res in results:
        for r in res["rows"]:
            if r["kind"] == "single":
                by_mat[res["mat"]].append(r["found"] and r["iou"])
    print(
        "- isolated pieces, mean IoU by mat (0 if missed): "
        + ", ".join(f"{m} {np.mean(v):.3f}" for m, v in by_mat.items())
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--px", type=float, nargs="+", default=[48.0])
    ap.add_argument(
        "--noise",
        type=float,
        nargs="+",
        default=[0.0003],
        help="depth noise [m]; negative: no depth",
    )
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    for noise in a.noise:
        for px in a.px:
            jobs = [(a.first_seed + k, px, MATS[k % len(MATS)], noise) for k in range(a.scenes)]
            with Pool(a.jobs) as pool:
                results = pool.map(one, jobs)
            report(results, px, noise)


if __name__ == "__main__":
    main()
