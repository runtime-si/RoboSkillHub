"""proposing-parallel-grasps-in-tight-clusters on synthetic packed clusters and real station
frames: fraction of parts with a collision-free grasp against the packing gap, timing, and
overlays. Not collected by pytest; run from the repository root:

    .venv/bin/python -m tests.bundles._cluster_grasp_eval --frames ~/yam_gap --out /tmp/cluster_eval

Synthetic: 4 x 4 jigsaw clusters (20 mm cells, 2 mm thick, classic and wavy cuts), 3 x 3
LEGO 2 x 4 bricks (15.8 x 31.8 x 9.6 mm) and 4 x 2 dominoes (24 x 48 x 7.5 mm) lying flat,
pulled apart by a packing gap, seen from 0.18 m by a D405-like camera; with every part's mask
or none (parts told apart by height), clean or with depth noise and edge holes, and three
fingertip thicknesses. Real: the 2026-10-31 overhead snapshot (30 mm cube, open box) and two
2026-11-04 wrist D405 frames over a USB-C plug on wood, each with its stored camera pose.
Writes ``cluster_eval.json`` and ``*.png`` overlays.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from ._cluster_scenes import brick_grid, jigsaw_cluster, render

HERE = Path(__file__).resolve().parent
SCRIPT = (
    HERE.parents[1]
    / "skills/proposing-parallel-grasps-in-tight-clusters/scripts/propose_cluster_grasps.py"
)
GAPS_MM = [0, 2, 4, 6, 8, 10, 12, 14, 16]


def load():
    spec = importlib.util.spec_from_file_location("propose_cluster_grasps", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Ctx:
    def __init__(self, frame=None):
        self.frame = frame

    def tool(self, name, **kw):
        if name == "robot.describe_gripper":
            return {"max_grasp_width_m": 0.086, "span_m": 0.094, "finger": {"reach_m": 0.0102}}
        if name == "robot.grasp_frame":
            return {"rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}}
        raise KeyError(name)


def draw(frame, out, path, labels=None, zoom=1):
    """Grasps on the depth image: jaw line, finger plates at the descent opening."""
    d = np.asarray(frame["depth"], dtype=float)
    ok = d > 0
    lo, hi = np.percentile(d[ok], [1, 99]) if ok.any() else (0, 1)
    g = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    img = (np.stack([g] * 3, -1) * 200 + 30).astype(np.uint8)
    img[~ok] = (90, 0, 90)
    if labels is not None:
        edge = labels != ndi.grey_erosion(labels, size=3)
        img[edge] = (0, 180, 255)
    im = Image.fromarray(img)
    if zoom != 1:
        im = im.resize((im.width * zoom, im.height * zoom), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    k = np.asarray(frame["intrinsics"], float)
    t = out_pose_matrix(frame["pose"])
    tinv = np.linalg.inv(t)

    def px(p):
        q = tinv @ np.append(p, 1.0)
        return (zoom * (k[0, 0] * q[0] / q[2] + k[0, 2]), zoom * (k[1, 1] * q[1] / q[2] + k[1, 2]))

    for rank, gr in reversed(list(enumerate(out["grasps"]))):
        c = np.array([gr["position"]["x"], gr["position"]["y"], gr["position"]["z"]])
        a = np.radians(gr["close_heading_deg"])
        dvec = np.array([np.cos(a), np.sin(a), 0.0])
        nvec = np.array([-np.sin(a), np.cos(a), 0.0])
        w = 0.5 * gr["jaw_width_m"]
        col = (255, 40, 40) if rank == 0 else (255, 200, 0)
        dr.line([px(c - w * dvec), px(c + w * dvec)], fill=col, width=2)
        for s in (1, -1):
            e = c + s * (w + 0.005) * dvec
            dr.line([px(e - 0.01 * nvec), px(e + 0.01 * nvec)], fill=col, width=3)
    im.save(path)


def out_pose_matrix(pose):
    mod = load()
    return mod._matrix(pose)


SCENES = {
    "jigsaw_classic": dict(kind="jigsaw", style="classic", height=0.002),
    "jigsaw_wavy": dict(kind="jigsaw", style="wavy", height=0.002),
    "lego_2x4": dict(kind="bricks", size=(0.0158, 0.0318), height=0.0096, grid=(3, 3)),
    "domino_flat": dict(kind="bricks", size=(0.024, 0.048), height=0.0075, grid=(4, 2)),
}


def scene(name, gap, noise=0.0, holes=0.0, seed=0):
    s = SCENES[name]
    if s["kind"] == "jigsaw":
        parts = jigsaw_cluster(4, 4, gap, seed=seed, style=s["style"])
    else:
        parts = brick_grid(*s["grid"], gap, size=s["size"])
    return render(parts, s["height"], noise=noise, holes=holes, seed=seed)


def sweep(mod, out_dir, results):
    rows = []
    for name in SCENES:
        for variant in ("masks", "no_masks", "masks_noisy"):
            for tip in (0.010, 0.006, 0.004):
                if variant != "masks" and tip != 0.010:
                    continue
                line = []
                for gap_mm in GAPS_MM:
                    noisy = variant == "masks_noisy"
                    sc = scene(
                        name,
                        gap_mm / 1e3,
                        noise=0.0003 if noisy else 0.0,
                        holes=0.5 if noisy else 0.0,
                    )
                    kw = {"piece_masks": sc.masks()} if variant != "no_masks" else {}
                    t0 = time.perf_counter()
                    out = mod.run(Ctx(), camera_frame=sc.frame, tip_thickness=tip, **kw)
                    dt = time.perf_counter() - t0
                    n_parts = len(sc.parts)
                    frac = out["n_graspable"] / n_parts
                    line.append(
                        {
                            "gap_mm": gap_mm,
                            "route": out["route"],
                            "graspable": out["n_graspable"],
                            "parts_found": out["n_pieces"],
                            "parts": n_parts,
                            "fraction": frac,
                            "seconds": dt,
                        }
                    )
                    if variant == "masks" and tip == 0.010 and gap_mm in (0, 6, 12):
                        draw(sc.frame, out, out_dir / f"{name}_gap{gap_mm}.png", sc.labels)
                rows.append(
                    {"scene": name, "variant": variant, "tip_mm": tip * 1e3, "by_gap": line}
                )
                print(
                    f"{name:15s} {variant:12s} tip {tip * 1e3:4.0f} mm  "
                    + " ".join(f"{r['gap_mm']:>2}:{r['graspable']:>2}/{r['parts']}" for r in line)
                    + f"   {np.median([r['seconds'] for r in line]):.2f} s"
                )
    results["sweep"] = rows


def real(mod, root, out_dir, results):
    res = {}

    def overhead():
        d = np.load(root / "real/snapshots/20261031-003520/obs.npz")
        t = np.array(
            json.loads((root / "real/top_camera_pose.before-20261031-004738.json").read_text())[
                "T_base_from_camera"
            ]
        )
        return d, t

    if (root / "real/snapshots/20261031-003520/obs.npz").exists():
        d, t = overhead()
        frame = {"depth": d["depth_m"], "intrinsics": d["K"], "pose": t, "rgb": d["rgb"]}
        dark = ndi.binary_opening(d["rgb"].astype(float).mean(-1) < 75, iterations=1)
        lab, _ = ndi.label(dark)
        cube = ndi.binary_fill_holes(lab == lab[256, 263])
        box = ndi.binary_fill_holes(lab == lab[270, 335])
        for name, target, others in (("cube", cube, [box]), ("box", box, [cube])):
            t0 = time.perf_counter()
            out = mod.run(Ctx(), camera_frame=frame, mask=target, piece_masks=others)
            dt = time.perf_counter() - t0
            res[f"overhead_{name}"] = summary(out, dt)
            draw(
                frame,
                out,
                out_dir / f"real_overhead_{name}.png",
                (target * 1 + sum(others) * 2).astype(int),
            )
        t0 = time.perf_counter()
        roi = {"x_min": -0.25, "x_max": 0.25, "y_min": 0.15, "y_max": 0.55}
        out = mod.run(Ctx(), camera_frame=frame, roi=roi)
        res["overhead_no_mask_roi"] = summary(out, time.perf_counter() - t0)
        draw(frame, out, out_dir / "real_overhead_no_mask.png")
    for run_dir in ("real-tf-1104-142813",):
        for name in ("find_end", "grasp_centre_0"):
            f = root / "threading/runs" / run_dir / f"{name}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            frame = {
                "depth": d["depth_m"],
                "intrinsics": d["K"],
                "pose": d["T_base_camera"],
                "rgb": d["rgb"],
            }
            rgb = d["rgb"].astype(float)
            # the own fingertips: dark and near the bottom corners of the wrist view
            fingers = (rgb.mean(-1) < 60) & (np.arange(480)[:, None] > 300)
            fingers = ndi.binary_dilation(fingers, iterations=6)
            white = (rgb.min(-1) > 150) & ~fingers
            white = ndi.binary_opening(white, iterations=2)
            lab, n = ndi.label(white)
            sizes = ndi.sum(white, lab, range(1, n + 1))
            cable = lab == 1 + int(np.argmax(sizes)) if n else None
            for variant, kw in (("cable_mask", {"mask": cable}), ("no_mask", {})):
                t0 = time.perf_counter()
                out = mod.run(Ctx(), camera_frame=frame, ignore_mask=fingers, **kw)
                res[f"wrist_{name}_{variant}"] = summary(out, time.perf_counter() - t0)
                draw(frame, out, out_dir / f"real_wrist_{name}_{variant}.png")
    results["real"] = res
    for k, v in res.items():
        print(f"{k:32s} {v}")


def summary(out, dt):
    s = {
        "route": out["route"],
        "parts": out["n_pieces"],
        "graspable": out["n_graspable"],
        "support_z": round(out["support_z"], 4),
        "voxel_mm": round(out["voxel_m"] * 1e3, 2),
        "seconds": round(dt, 2),
        "reason": out["reason"],
    }
    if out["grasps"]:
        g = out["grasps"][0]
        s.update(
            {
                "position": {k: round(v, 4) for k, v in g["position"].items()},
                "heading_deg": round(g["close_heading_deg"], 1),
                "jaw_mm": round(g["jaw_width_m"] * 1e3, 1),
                "object_mm": round(g["object_width_m"] * 1e3, 1),
                "height_mm": round(g["part_height_m"] * 1e3, 1),
                "clearance_mm": round(g["clearance_m"] * 1e3, 1),
                "score": round(g["score"], 3),
            }
        )
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=Path.home() / "yam_gap")
    ap.add_argument("--out", type=Path, default=Path("/tmp/cluster_eval"))
    ap.add_argument("--skip-sweep", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    mod = load()
    results: dict = {}
    if not a.skip_sweep:
        sweep(mod, a.out, results)
    real(mod, a.frames, a.out, results)
    (a.out / "cluster_eval.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
