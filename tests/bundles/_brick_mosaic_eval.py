"""planning-a-brick-mosaic-from-an-image: colour error against part count, stock, timing, and the
downstream brick skills. Not collected by pytest; run from the repo root:

    .venv/bin/python tests/bundles/_brick_mosaic_eval.py [--out /tmp/mosaic_eval]

Images: three synthetic ones made here (palette blocks, a smooth colour sweep, anti-aliased shapes
in off-palette colours) and, when scikit-image is installed, three of its sample photos: astronaut
(NASA, public domain), coffee (Rachel Michetti, CC0) and chelsea (Stefan van der Walt, CC0).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from gap.testing import FakeContext

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "skills" / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pm = _load("plan_mosaic", "planning-a-brick-mosaic-from-an-image/scripts/plan_mosaic.py")
po = _load("plan_order", "planning-brick-assembly-order/scripts/plan_order.py")
cd = _load("check_design", "checking-design-connectivity-and-collisions/scripts/check_design.py")


def blocks(n: int = 256) -> np.ndarray:
    """A 4 x 4 checker of 16 LDraw colours, each block exactly a palette colour."""
    cols = [pm._hex_rgb(h) for _, _, _, h in pm.LEGO[:16]]
    img = np.zeros((n, n, 3), np.uint8)
    s = n // 4
    for i in range(4):
        for j in range(4):
            img[i * s : (i + 1) * s, j * s : (j + 1) * s] = cols[i * 4 + j]
    return img


def sweep(w: int = 384, h: int = 256) -> np.ndarray:
    """Hue across, lightness down (HSV value 1 -> 0.25, saturation 0.85)."""
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = (np.arange(w) * 180 // w)[None, :]
    hsv[..., 1] = 217
    hsv[..., 2] = np.linspace(255, 64, h).astype(np.uint8)[:, None]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def shapes(n: int = 512) -> np.ndarray:
    """Anti-aliased disc, triangle and ring in off-palette colours on a vertical teal-to-sand
    gradient, then blurred and noised a little like a photo."""
    t = np.linspace(0, 1, n)[:, None, None]
    img = ((1 - t) * np.array([40, 120, 130]) + t * np.array([220, 200, 160])).repeat(n, 1)
    img = img.astype(np.float32)
    cv2.circle(img, (n // 3, n // 3), n // 6, (230, 90, 60), -1, cv2.LINE_AA)
    tri = np.array([[n * 0.55, n * 0.85], [n * 0.95, n * 0.85], [n * 0.75, n * 0.45]], np.int32)
    cv2.fillPoly(img, [tri], (120, 60, 160), cv2.LINE_AA)
    cv2.circle(img, (n // 4, int(n * 0.75)), n // 8, (250, 230, 70), n // 30, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    img += np.random.default_rng(0).normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def images() -> dict[str, np.ndarray]:
    out = {"blocks": blocks(), "sweep": sweep(), "shapes": shapes()}
    try:
        import skimage.data as d

        out.update(astronaut=d.astronaut(), coffee=d.coffee(), chelsea=d.chelsea())
    except ImportError:
        print("scikit-image not installed: photos skipped")
    return out


def plan(img: np.ndarray, **kw):
    t = time.perf_counter()
    o = pm.run(FakeContext(), image=img, write_files=False, **kw)
    return o, time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    imgs = images()
    rows = []

    print("\n## Colour error and part count (palette lego, crop, area)\n")
    print("| image | studs | sizes | dither | placements | mean dE | p95 dE | blurred dE | ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    size_sets = {
        "1x1": ["1x1"],
        "+1x2,2x2": ["1x1", "1x2", "2x2"],
        "+..2x4": ["1x1", "1x2", "1x3", "1x4", "2x2", "2x3", "2x4"],
    }
    for name, img in imgs.items():
        for n in (16, 32, 48, 64):
            for sname, sizes in size_sets.items():
                for dither in ("none", "floyd_steinberg"):
                    if dither != "none" and sname != "1x1":
                        continue
                    o, dt = plan(img, plate_studs=[n, n], sizes=sizes, dither=dither)
                    rows.append(
                        {
                            "image": name,
                            "n": n,
                            "sizes": sname,
                            "dither": dither,
                            **{
                                k: o[k]
                                for k in (
                                    "route",
                                    "n_placements",
                                    "mean_delta_e",
                                    "p95_delta_e",
                                    "mean_delta_e_blurred",
                                )
                            },
                            "ms": dt * 1e3,
                        }
                    )
                    print(
                        f"| {name} | {n}x{n} | {sname} | {dither} | {o['n_placements']} | {o['mean_delta_e']:.2f} | {o['p95_delta_e']:.2f} | {o['mean_delta_e_blurred']:.2f} | {dt * 1e3:.0f} |"
                    )

    print("\n## brickmos' pipeline vs this one (32x32, 1x1)\n")
    print(
        "| image | palette | brickmos (stretch, bilinear, RGB euclid) dE | this (crop, area, CIEDE2000) dE | same fit, CIEDE2000 only |"
    )
    print("|---|---|---|---|---|")
    for name, img in imgs.items():
        for palette in ("brickmos", "lego"):
            o, _ = plan(img, plate_studs=[32, 32], palette=palette)
            cells = pm._cells(pm._fit(img / 255.0, 32, 32, "stretch", [255] * 3), 32, 32, "linear")
            pal = pm._palette(palette)
            rgb = np.array([p["rgb"] for p in pal])
            flat = cells.reshape(-1, 3) * 255
            k = np.argmin(((flat[:, None, :] - rgb[None]) ** 2).sum(-1), axis=1)
            lab = pm.srgb_to_lab(cells.reshape(-1, 3))
            pl = np.stack([p["lab"] for p in pal])
            de_bm = pm.deltaE_ciede2000(lab, pl[k]).mean()
            o2, _ = plan(
                img, plate_studs=[32, 32], palette=palette, fit="stretch", resample="linear"
            )
            print(
                f"| {name} | {palette} | {de_bm:.2f} | {o['mean_delta_e']:.2f} | {o2['mean_delta_e']:.2f} |"
            )

    print(
        "\n## Short stock (32x32, lego_basic, 1x1): each colour stocked at a fraction of its need\n"
    )
    print("| image | stock | route | substituted | empty | mean dE | floor dE | ms |")
    print("|---|---|---|---|---|---|---|---|")
    for name in ("shapes", "astronaut", "coffee"):
        if name not in imgs:
            continue
        need, _ = plan(imgs[name], plate_studs=[32, 32], palette="lego_basic")
        counts = {b["colour"]: b["count"] for b in need["bill_of_materials"]}
        for frac, extra in ((1.0, 0), (0.7, 0.35), (0.5, 0.0)):
            # Every used colour at frac of its need; spare stock spread over all colours when extra.
            spare = int(extra * 1024 / 14)
            inv = {c: int(counts.get(c, 0) * frac) + spare for c in pm.LEGO_BASIC_NAMES}
            o, dt = plan(imgs[name], plate_studs=[32, 32], palette="lego_basic", inventory=inv)
            print(
                f"| {name} | {frac:.0%} of need + {spare}/colour | {o['route']} | {o['n_substituted']} | {o['n_empty']} | {o['mean_delta_e']:.2f} | {o['ideal_mean_delta_e']:.2f} | {dt * 1e3:.0f} |"
            )

    print("\n## Timing (astronaut, lego, 1x1 unless noted), ms\n")
    print("| studs | plain | 1x1..2x4 | dither | stock 60% (program) | stock with substitutions |")
    print("|---|---|---|---|---|---|")
    if "astronaut" in imgs:
        img = imgs["astronaut"]
        for n in (32, 48, 64, 96, 128):
            base, t0 = plan(img, plate_studs=[n, n])
            _, t1 = plan(img, plate_studs=[n, n], sizes=size_sets["+..2x4"])
            _, t2 = plan(img, plate_studs=[n, n], dither="floyd_steinberg")
            counts = {b["colour"]: b["count"] for b in base["bill_of_materials"]}
            _, t3 = plan(
                img, plate_studs=[n, n], inventory={c: int(v * 0.6) for c, v in counts.items()}
            )
            big = {c: 2 * v for c, v in counts.items() if v > 0.05 * n * n}
            _, t4 = plan(img, plate_studs=[n, n], inventory=big)
            print(
                f"| {n}x{n} | {t0 * 1e3:.0f} | {t1 * 1e3:.0f} | {t2 * 1e3:.0f} | {t3 * 1e3:.0f} | {t4 * 1e3:.0f} |"
            )

    print("\n## Downstream (32x32 plates on a 32x32 plate)\n")
    print(
        "| image | sizes | placements | checking-design | assembly order, finger check | without |"
    )
    print("|---|---|---|---|---|---|")
    for name in ("shapes", "astronaut"):
        if name not in imgs:
            continue
        for sname, sizes in size_sets.items():
            o, _ = plan(imgs[name], plate_studs=[32, 32], sizes=sizes)
            c = cd.run(
                FakeContext(), bricks=o["placements_json"], plate_size="32x32", stability="off"
            )
            a1 = po.run(FakeContext(), steps_json=o["steps_json"], check_fingers=True)
            a0 = po.run(FakeContext(), steps_json=o["steps_json"], check_fingers=False)
            kept = [e["id"] for e in a0["sequence"]] == list(range(o["n_placements"]))
            print(
                f"| {name} | {sname} | {o['n_placements']} | {c['route']} | {a1['route']} {a1['n_placed']}/{a1['n_bricks']} | {a0['route']} {a0['n_placed']}/{a0['n_bricks']}, order kept {kept} |"
            )

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for name in imgs:
            for tag, kw in (
                ("plain", {}),
                ("dither", {"dither": "floyd_steinberg"}),
                ("merged", {"sizes": size_sets["+1x2,2x2"]}),
            ):
                pm.run(
                    FakeContext(),
                    image=imgs[name],
                    plate_studs=[48, 48],
                    out_dir=str(out / f"{name}_{tag}"),
                    **kw,
                )
        (out / "rows.json").write_text(json.dumps(rows, indent=1))
        print(f"\nwrote previews to {out}")


if __name__ == "__main__":
    main()
