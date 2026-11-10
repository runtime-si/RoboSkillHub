"""Orientation pairs: one part rendered twice in the same wrist view at two yaws, composited on real wrist frames.

    MUJOCO_GL=osmesa TARGET_PARTS_DIR=WORK python tests/bundles/_relative_rotation_gen_pairs.py OUT_DIR N_PAIRS SEED

Uses tests/bundles/_target_parts_gen_scenes.py (tools/README.md's renderer; WORK as described
there). The pairs of the SKILL.md table: 80 pairs, seed 21.

Per pair: ref.png / tgt.png (full frames), the part's box in each, the in-image angle of its long axis in each
(projected through the camera), the true relative in-plane rotation and the part's top-view symmetry order.
"""

import json
import math
import os
import sys

import numpy as np

OUT, N, SEED = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sys.argv = [sys.argv[0], "/tmp/unused", "0", "0", str(SEED)]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _target_parts_gen_scenes as g  # noqa: E402

rng = np.random.default_rng(SEED)


def sym_order(o):
    sub = o["sub"]
    if sub.startswith("domino"):
        return (
            1 if sub == "domino_flat" else 2
        )  # pips break the 2-fold symmetry of a flat face-up tile
    nx, ny = (int(v) for v in sub.rsplit("_", 1)[1].split("x"))
    return 4 if nx == ny else 2


def image_angle(R, t, K, x, y, yaw):
    """Angle (deg) in the image of the part's local x axis, from its centre on the table."""
    c = np.array([x, y, g.TABLE_Z + 0.005])
    a = c + 0.02 * np.array([math.cos(yaw), math.sin(yaw), 0.0])

    def proj(p):
        pc = R.T @ (p - t)
        return K[:2, :2] @ (pc[:2] / pc[2]) + K[:2, 2]

    d = proj(a) - proj(c)
    return math.degrees(math.atan2(d[1], d[0]))


def main():
    from PIL import Image

    os.makedirs(OUT, exist_ok=True)
    K = np.load(
        os.path.join(g.YG, "real/snapshots/20261103-114411-threading-setup/wrist_left.npz")
    )["K"]
    wr = os.path.join(g.HERE, "wrist_bg")
    bgs = [
        np.asarray(Image.open(os.path.join(wr, f)).convert("RGB")) for f in sorted(os.listdir(wr))
    ]
    pairs = []
    k = 0
    while len(pairs) < N:
        k += 1
        h = rng.uniform(0.15, 0.25)
        cx, cy = rng.uniform(0.15, 0.45), rng.uniform(0.25, 0.5)
        R = g.look_down(rng.uniform(0, 2 * math.pi), math.radians(rng.uniform(0, 12)))
        t = np.array([cx, cy, g.TABLE_Z + h])
        o = g.sample_objects(1, rng)[0]
        if o["sub"] == "domino_upright":
            continue
        # the part a little above the image centre (the fingers fill the bottom of the real frames)
        px, py = cx + rng.uniform(-0.02, 0.02), cy + h * 0.12
        yaw1 = rng.uniform(0, 2 * math.pi)
        dyaw = rng.uniform(-math.pi, math.pi)
        bg = bgs[k % len(bgs)]
        rec = {
            "sub": o["sub"],
            "cls": o["cls"],
            "colour": o["colour"],
            "sym": sym_order(o),
            "height_m": round(h, 3),
        }
        ok = True
        for tag, yaw in (("ref", yaw1), ("tgt", yaw1 + dyaw)):
            placed = [{**o, "x": px, "y": py, "yaw": yaw, "rad": 0.0}]
            rgb, inst = g.render(placed, R, t, K, rng)
            m = inst == 0
            if m.sum() < 80:
                ok = False
                break
            ys, xs = np.nonzero(m)
            if ys.max() > 0.62 * 480 or xs.min() < 2 or xs.max() > 637 or ys.min() < 2:
                ok = False
                break
            img = g.composite(bg, rgb, inst, rng)
            Image.fromarray(img).save(os.path.join(OUT, f"p{len(pairs):03d}_{tag}.png"))
            np.save(os.path.join(OUT, f"p{len(pairs):03d}_{tag}_mask.npy"), m)
            rec[tag] = {
                "image": f"p{len(pairs):03d}_{tag}.png",
                "mask": f"p{len(pairs):03d}_{tag}_mask.npy",
                "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                "angle_deg": image_angle(R, t, K, px, py, yaw),
            }
        if not ok:
            continue
        d = rec["tgt"]["angle_deg"] - rec["ref"]["angle_deg"]
        rec["rel_deg"] = (d + 180) % 360 - 180
        pairs.append(rec)
        print("pair", len(pairs), rec["sub"], round(rec["rel_deg"], 1), flush=True)
    json.dump(
        {"about": __doc__, "seed": SEED, "pairs": pairs},
        open(os.path.join(OUT, "labels.json"), "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
