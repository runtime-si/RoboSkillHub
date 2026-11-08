"""grconvnet.propose on real YAM station RGB-D frames: top grasps on a cube, an open box and a
flat cable, drawn, with CPU latency. Not collected by pytest; run in the bundle's env:

    uv run --project tools/grconvnet python tests/bundles/_grconvnet_station_eval.py \
        --frames ~/yam_gap --out /tmp/grconv_check

Frames (640 x 480, depth registered to colour, overhead Orbbec Gemini 335): the 2026-10-31
snapshot (30 mm black cube and an open 100 mm black box on white cloth, table at ~0.57 m) and
three 2026-11-03 threading frames (11:44, 13:58, 14:21: a flat white cable laid three ways on
wood, table at ~1.0 m), each with the
top-camera calibration in force then. Target masks are thresholded by colour here (the
cube and box: dark components; the cable: a thin bright top-hat), standing in for sam3.
Writes ``grconvnet_station.json`` (every run) and ``grconv_<object>_<checkpoint>.png``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1] / "tools" / "grconvnet" / "tools.py"


def load_tools():
    spec = importlib.util.spec_from_file_location("grconvnet_tools", TOOLS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frames(root: Path):
    def one(npz, pose):
        d = np.load(root / npz)
        T = np.array(json.loads((root / pose).read_text())["T_base_from_camera"])
        return d["rgb"], d["depth_m"].astype(np.float32), d["K"], T

    return {
        "cube_scene": one(
            "real/snapshots/20261031-003520/obs.npz",
            "real/top_camera_pose.before-20261031-004738.json",
        ),
        "cable_1421": one(
            "threading/cable_frame_20261103-1421.npz",
            "real/top_camera_pose.before-20261105-023658.json",
        ),
        "cable_1358": one(
            "threading/cable_frame_20261103.npz", "real/top_camera_pose.before-20261105-023658.json"
        ),
        "cable_1144": one(
            "real/snapshots/20261103-114411-threading-setup/top.npz",
            "real/top_camera_pose.before-20261105-023658.json",
        ),
    }


def _component_at(m, u, v):
    lab, _ = ndi.label(m)
    assert lab[v, u], (u, v)
    return lab == lab[v, u]


def masks(fr):
    rgb = fr["cube_scene"][0]
    dark = ndi.binary_opening(rgb.astype(float).mean(-1) < 75, iterations=1)
    out = {
        "cube": ("cube_scene", ndi.binary_fill_holes(_component_at(dark, 263, 256))),
        "box": ("cube_scene", ndi.binary_fill_holes(_component_at(dark, 335, 270))),
    }
    for scene in ("cable_1421", "cable_1358", "cable_1144"):
        f = fr[scene][0].astype(float)
        thin = ndi.white_tophat(f[..., 2] - 0.5 * f[..., 0], size=9) > 22
        thin &= f.min(-1) > 120
        thin[:285] = False  # above the table
        thin[452:] = False
        thin[:, :120] = False  # the left arm
        thin[:, 480:] = False  # the right arm's grey motor housing
        lab, k = ndi.label(ndi.binary_closing(thin, iterations=2))
        sizes = ndi.sum(np.ones_like(lab), lab, range(1, k + 1))
        for name, i in zip("ab", np.argsort(-sizes)[:2], strict=False):
            if sizes[i] >= 150:
                out[f"{scene}{name}"] = (scene, lab == 1 + int(i))
    return out


def draw(rgb, mask, grasps, path, crop, top=3, zoom=4):
    x0, y0 = int(max(crop["x0"], 0)), int(max(crop["y0"], 0))
    x1 = int(min(crop["x0"] + crop["side"], rgb.shape[1]))
    y1 = int(min(crop["y0"] + crop["side"], rgb.shape[0]))
    img = rgb.copy()
    edge = mask ^ ndi.binary_erosion(mask)
    img[edge] = (0, 200, 255)
    im = Image.fromarray(img[y0:y1, x0:x1]).resize(
        ((x1 - x0) * zoom, (y1 - y0) * zoom), Image.NEAREST
    )
    d = ImageDraw.Draw(im)
    colours = [(255, 40, 40), (255, 160, 0), (230, 230, 0)]
    for g in reversed(grasps[:top]):
        c = colours[g["rank"] % 3]
        (au, av), (bu, bv) = g["jaw_px"]
        p = [
            ((au - x0 + 0.5) * zoom, (av - y0 + 0.5) * zoom),
            ((bu - x0 + 0.5) * zoom, (bv - y0 + 0.5) * zoom),
        ]
        d.line(p, fill=c, width=3)
        # Jaw plates: short bars across the closing line at each end.
        ux, uy = p[1][0] - p[0][0], p[1][1] - p[0][1]
        n = max(np.hypot(ux, uy), 1e-6)
        px, py = -uy / n * 4 * zoom, ux / n * 4 * zoom
        for ex, ey in p:
            d.line([(ex - px, ey - py), (ex + px, ey + py)], fill=c, width=5)
        d.text((p[1][0] + 4, p[1][1]), f"{g['rank']}:{g['quality']:.2f}", fill=c)
    im.save(path)


#: Crop modes swept: ("fill", f) sizes the crop so the mask's longer side is f of it;
#: ("scale", s) is a crop of s x the network input centred on the mask (1 = native pixels).
MODES = [("fill", 0.3), ("fill", 0.5), ("scale", 0.75), ("scale", 1.0), ("scale", 1.5)]


DEFAULT_MODE = ("scale", 0.75)


def kind(obj):
    return obj.split("_")[0]  # cube, box, cable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=Path.home() / "yam_gap")
    ap.add_argument("--out", type=Path, default=Path("/tmp/grconv_check"))
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--threads", type=int, nargs="*", default=[1, 4, 8])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    import _grconvnet_criteria as C

    gc = load_tools()
    import torch

    fr = frames(args.frames)
    ms = masks(fr)
    np.savez_compressed(args.out / "grconvnet_masks.npz", **{k: m for k, (_, m) in ms.items()})
    gts = {o: C.ground_truth(kind(o), *fr[sc][1:4], m) for o, (sc, m) in ms.items()}
    runs = []
    for ckpt in gc.CHECKPOINTS:
        size = gc.CHECKPOINTS[ckpt]["input_size"]
        for mode, val in MODES:
            for obj, (scene, m) in ms.items():
                rgb, depth, K, T = fr[scene]
                kw = {"object_fill": val} if mode == "fill" else {"crop_px": val * size}
                out = gc.propose(rgb, depth, K, mask=m, camera_pose=T, n=3, checkpoint=ckpt, **kw)
                verdicts = [
                    C.judge(
                        gts[obj],
                        [g["position"]["x"], g["position"]["y"]],
                        g["close_heading_deg"],
                        g["width_m"],
                    )
                    for g in out["grasps"]
                ]
                runs.append(
                    {
                        "object": obj,
                        "checkpoint": ckpt,
                        "mode": f"{mode}={val}",
                        "result": out,
                        "verdicts": verdicts,
                    }
                )
                if (ckpt, mode, val) == (gc.DEFAULT_CHECKPOINT, *DEFAULT_MODE):
                    draw(rgb, m, out["grasps"], args.out / f"grconv_{obj}.png", out["crop"])
                if (mode, val) == ("scale", 1.0):
                    draw(
                        rgb,
                        m,
                        out["grasps"],
                        args.out / f"grconv_{obj}_{ckpt}_native.png",
                        out["crop"],
                    )

    table = {}
    for r in runs:
        key = (r["checkpoint"], r["mode"])
        t = table.setdefault(key, {"top1": 0, "top3": 0, "n": 0, "none": 0})
        t["n"] += 1
        t["top1"] += bool(r["verdicts"] and r["verdicts"][0][0])
        t["top3"] += any(v[0] for v in r["verdicts"])
        t["none"] += not r["verdicts"]
    print("checkpoint      mode        top1 top3 no-grasp (of", len(ms), "targets)")
    for (ck, mode), t in table.items():
        print(f"{ck:15s} {mode:11s} {t['top1']:4d} {t['top3']:4d} {t['none']:4d}")
    for r in runs:
        if (r["checkpoint"], r["mode"]) != (
            gc.DEFAULT_CHECKPOINT,
            f"{DEFAULT_MODE[0]}={DEFAULT_MODE[1]}",
        ):
            continue
        for g, v in zip(r["result"]["grasps"], r["verdicts"], strict=True):
            print(
                f"  {r['object']:8s} #{g['rank']} q={g['quality']:.2f} {'PASS' if v[0] else 'fail'} {v[1]}"
            )

    # Latency: the default checkpoint and crop on the cube, warm, at several thread counts.
    rgb, depth, K, T = fr["cube_scene"]
    size = gc.CHECKPOINTS[gc.DEFAULT_CHECKPOINT]["input_size"]
    kw = (
        {"object_fill": DEFAULT_MODE[1]}
        if DEFAULT_MODE[0] == "fill"
        else {"crop_px": DEFAULT_MODE[1] * size}
    )
    latency = {}
    for threads in args.threads:
        torch.set_num_threads(threads)
        tot, inf = [], []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            out = gc.propose(rgb, depth, K, mask=ms["cube"][1], camera_pose=T, **kw)
            tot.append(time.perf_counter() - t0)
            inf.append(out["timing"]["infer_s"])
        latency[threads] = {
            "total_ms_median": round(1e3 * statistics.median(tot), 1),
            "total_ms_p90": round(1e3 * sorted(tot)[int(0.9 * len(tot))], 1),
            "infer_ms_median": round(1e3 * statistics.median(inf), 1),
            "pre_ms": round(1e3 * out["timing"]["preprocess_s"], 1),
            "post_ms": round(1e3 * out["timing"]["postprocess_s"], 1),
        }
    summary = {
        "device": os.environ.get("GAP_GRCONVNET_DEVICE", "cpu"),
        "torch": torch.__version__,
        "cpu": open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t"),
        "load_s": {k: getattr(v, "load_s", None) for k, v in gc._runtimes.items()},
        "latency_by_threads": latency,
        "table": {f"{k[0]} {k[1]}": v for k, v in table.items()},
        "runs": runs,
    }
    (args.out / "grconvnet_station.json").write_text(json.dumps(summary, indent=1, default=float))
    print(
        json.dumps(
            {"cpu": summary["cpu"], "device": summary["device"], "latency": latency}, indent=1
        )
    )


if __name__ == "__main__":
    main()
