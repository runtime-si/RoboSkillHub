"""Labelled eval scenes of LEGO / DUPLO bricks and dominoes, rendered with MuJoCo through the YAM station's calibrated
cameras and composited onto real station frames.

    MUJOCO_GL=osmesa TARGET_PARTS_DIR=WORK python tests/bundles/_target_parts_gen_scenes.py OUT_DIR N_OVERHEAD N_WRIST SEED

(any Python with mujoco >= 3; CPU rendering). WORK holds frames/ (the 7 overhead frames of
station_small_parts_labels.json), station_small_parts_labels.json and wrist_bg/ (real wrist frames);
the station's camera calibration is read from ~/yam_gap. The eval set is seed 7 with 60 / 40.

Overhead: the real overhead Orbbec intrinsics and its ChArUco-calibrated pose (top_camera_pose_at_capture_20261103-1421),
parts placed on the measured table plane (z = -0.021 m in the left-arm base frame) inside the table region of real
overhead frames, avoiding their labelled objects. Wrist: the left wrist D405 intrinsics, a camera 0.18-0.35 m over the
table looking down (tilt <= 20 deg), real wrist frames as background, parts in the upper 60 % of the frame (the
fingers are at the bottom). Each scene: rgb (composited), per-instance visible masks, class, subclass, colour.
"""

import json
import math
import os
import sys

import mujoco
import numpy as np
from PIL import Image, ImageFilter

OUT = sys.argv[1]
N_TOP, N_WRIST, SEED = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(SEED)
HERE = os.environ.get("TARGET_PARTS_DIR", "target_parts")
YG = os.path.expanduser("~/yam_gap")

TABLE_Z = -0.02102
COLOURS = {  # LDraw values (CCAL 2.0)
    "red": "C91A09",
    "blue": "0055BF",
    "yellow": "F2CD37",
    "green": "237841",
    "black": "05131D",
    "white": "FFFFFF",
    "tan": "E4CD9E",
    "light_grey": "A0A5A9",
    "dark_grey": "6C6E68",
    "orange": "FE8A18",
    "lime": "BBE90B",
    "azure": "36AEBF",
}
DOMINO_COLOURS = {
    "white": ("F4F4F0", "101010"),
    "ivory": ("E8DDB5", "101010"),
    "black": ("141414", "F0F0F0"),
}


def rgba(hexs, a=1.0):
    return f"{int(hexs[0:2], 16) / 255:.3f} {int(hexs[2:4], 16) / 255:.3f} {int(hexs[4:6], 16) / 255:.3f} {a}"


def brick_geoms(kind, nx, ny, colour):
    """Body-frame geoms: a box on z=0 and studs. kind: lego_brick, lego_plate, duplo_brick."""
    pitch, h, sd, sh = {
        "lego_brick": (0.008, 0.0096, 0.0048, 0.0017),
        "lego_plate": (0.008, 0.0032, 0.0048, 0.0017),
        "duplo_brick": (0.016, 0.0192, 0.0094, 0.0045),
    }[kind]
    lx, ly = nx * pitch - 0.0002, ny * pitch - 0.0002
    c = rgba(COLOURS[colour])
    g = [f'<geom type="box" size="{lx / 2} {ly / 2} {h / 2}" pos="0 0 {h / 2}" rgba="{c}"/>']
    for i in range(nx):
        for j in range(ny):
            x, y = (i - (nx - 1) / 2) * pitch, (j - (ny - 1) / 2) * pitch
            g.append(
                f'<geom type="cylinder" size="{sd / 2} {sh / 2}" pos="{x} {y} {h + sh / 2}" rgba="{c}"/>'
            )
    return g, (lx, ly, h + sh)


def domino_geoms(style, pose, rng):
    """A 48 x 24 x 7.5 mm tile. pose: flat (pips up), face_down, upright (standing on a 24 x 7.5 end)."""
    L, W, T = 0.048, 0.024, 0.0075
    body, pip = DOMINO_COLOURS[style]
    cb, cp = rgba(body), rgba(pip)
    if pose == "upright":
        g = [f'<geom type="box" size="{T / 2} {W / 2} {L / 2}" pos="0 0 {L / 2}" rgba="{cb}"/>']
        return g, (T, W, L)
    g = [f'<geom type="box" size="{L / 2} {W / 2} {T / 2}" pos="0 0 {T / 2}" rgba="{cb}"/>']
    if pose == "flat":
        g.append(
            f'<geom type="box" size="0.0003 {W / 2 - 0.003} 0.0002" pos="0 0 {T}" rgba="{cp}"/>'
        )
        layouts = {
            0: [],
            1: [(0, 0)],
            2: [(-1, -1), (1, 1)],
            3: [(-1, -1), (0, 0), (1, 1)],
            4: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
            5: [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)],
            6: [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)],
        }
        for half in (-1, 1):
            for u, v in layouts[int(rng.integers(0, 7))]:
                x = half * L / 4 + u * 0.0055
                g.append(
                    f'<geom type="cylinder" size="0.0018 0.0002" pos="{x} {v * 0.0065} {T}" rgba="{cp}"/>'
                )
    return g, (L, W, T)


def sample_objects(n, rng):
    objs = []
    for _ in range(n):
        r = rng.random()
        if r < 0.35:
            kind, (nx, ny) = (
                "lego_brick",
                [(2, 2), (2, 4), (1, 2), (1, 4), (2, 3), (1, 1)][rng.integers(0, 6)],
            )
        elif r < 0.45:
            kind, (nx, ny) = "lego_plate", [(2, 4), (2, 2), (1, 4)][rng.integers(0, 3)]
        elif r < 0.65:
            kind, (nx, ny) = "duplo_brick", [(2, 2), (2, 4), (2, 3)][rng.integers(0, 3)]
        else:
            kind = "domino"
        if kind == "domino":
            style = ["white", "ivory", "black"][rng.integers(0, 3)]
            pose = rng.choice(["flat", "flat", "face_down", "upright"])
            geoms, size = domino_geoms(style, pose, rng)
            objs.append(
                {
                    "cls": "domino",
                    "sub": f"domino_{pose}",
                    "colour": style,
                    "geoms": geoms,
                    "size": size,
                }
            )
        else:
            colour = list(COLOURS)[rng.integers(0, len(COLOURS))]
            geoms, size = brick_geoms(kind, nx, ny, colour)
            sub = f"{kind}_{nx}x{ny}"
            objs.append(
                {
                    "cls": "duplo" if kind == "duplo_brick" else "lego",
                    "sub": sub,
                    "colour": colour,
                    "geoms": geoms,
                    "size": size,
                }
            )
    return objs


def place(objs, region_xy, rng, min_gap=0.006):
    """Non-overlapping (bounding-circle) placements inside region_xy = (x0, x1, y0, y1); drops what does not fit."""
    placed = []
    for o in objs:
        rad = math.hypot(o["size"][0], o["size"][1]) / 2
        for _ in range(200):
            x, y = rng.uniform(region_xy[0], region_xy[1]), rng.uniform(region_xy[2], region_xy[3])
            if all(math.hypot(x - p["x"], y - p["y"]) > rad + p["rad"] + min_gap for p in placed):
                placed.append({**o, "x": x, "y": y, "yaw": rng.uniform(0, 2 * math.pi), "rad": rad})
                break
    return placed


def scene_xml(placed, cam_pos, xaxis, yaxis, fovy, light_dir):
    bodies = []
    for i, p in enumerate(placed):
        q = f"{math.cos(p['yaw'] / 2)} 0 0 {math.sin(p['yaw'] / 2)}"
        bodies.append(
            f'<body name="o{i}" pos="{p["x"]} {p["y"]} {TABLE_Z}" quat="{q}">'
            + "".join(p["geoms"])
            + "</body>"
        )
    xa, ya = " ".join(f"{v:.6f}" for v in xaxis), " ".join(f"{v:.6f}" for v in yaxis)
    ld = " ".join(f"{v:.3f}" for v in light_dir)
    return f"""<mujoco><visual><headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>
<quality shadowsize="2048"/><global offwidth="640" offheight="480"/></visual>
<worldbody><light directional="true" dir="{ld}" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1" castshadow="false"/>
<camera name="cam" pos="{cam_pos[0]} {cam_pos[1]} {cam_pos[2]}" xyaxes="{xa} {ya}" fovy="{fovy}"/>
{"".join(bodies)}</worldbody></mujoco>"""


def render(placed, R, t, K, rng):
    fovy = math.degrees(2 * math.atan(240 / K[1, 1]))
    ld = rng.normal([0.3, 0.4, -1.0], 0.2)
    m = mujoco.MjModel.from_xml_string(
        scene_xml(placed, t, R[:, 0], -R[:, 1], fovy, ld / np.linalg.norm(ld))
    )
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 480, 640)
    r.update_scene(d, camera="cam")
    rgb = r.render().copy()
    r.enable_segmentation_rendering()
    r.update_scene(d, camera="cam")
    seg = r.render()[:, :, 0].copy()
    r.close()
    body_of = np.full(m.ngeom + 1, -1)
    for gi in range(m.ngeom):
        body_of[gi] = m.geom_bodyid[gi] - 1  # world = 0
    inst = np.where(seg >= 0, body_of[np.clip(seg, 0, m.ngeom)], -1)
    return rgb, inst


def composite(bg, rgb, inst, rng):
    fg = inst >= 0
    out = bg.astype(np.float32).copy()
    # soft contact shadow: the silhouette shifted toward the light's far side, blurred, darkening the table
    sh = np.roll(np.roll(fg, 3, axis=0), 2, axis=1).astype(np.float32)
    sh = (
        np.asarray(
            Image.fromarray((sh * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.5)),
            np.float32,
        )
        / 255
    )
    out *= (1 - 0.45 * sh * (~fg))[:, :, None]
    # match the camera: brightness of the background, slight blur and noise on the rendered parts
    gain = np.clip(bg[fg].mean() / max(rgb[fg].mean(), 1), 0.75, 1.15) if fg.any() else 1.0
    gain *= rng.uniform(0.85, 1.05)
    part = np.asarray(
        Image.fromarray(np.clip(rgb * gain, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(0.6)
        ),
        np.float32,
    )
    part += rng.normal(0, 3.0, part.shape)
    out[fg] = part[fg]
    return np.clip(out, 0, 255).astype(np.uint8)


def look_down(yaw, tilt):
    """OpenCV camera rotation (columns = camera axes in the world) looking down with a tilt about the x axis."""
    cz = np.array([0.0, math.sin(tilt), -math.cos(tilt)])
    cx = np.array([1.0, 0.0, 0.0])
    cy = np.cross(cz, cx)
    Rz = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    )
    return Rz @ np.stack([cx, cy, cz], axis=1)


def save(name, img, inst, placed, meta):
    objs = []
    for i, p in enumerate(placed):
        m = inst == i
        if m.sum() < 15:
            continue
        ys, xs = np.nonzero(m)
        objs.append(
            {
                "cls": p["cls"],
                "sub": p["sub"],
                "colour": p["colour"],
                "area": int(m.sum()),
                "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                "inst": i,
            }
        )
    keep = [o["inst"] for o in objs]
    masks = (
        np.stack([inst == i for i in keep]).astype(np.uint8)
        if keep
        else np.zeros((0, *inst.shape), np.uint8)
    )
    Image.fromarray(img).save(os.path.join(OUT, f"{name}.png"))
    np.savez_compressed(os.path.join(OUT, f"{name}_masks.npz"), masks=masks)
    return {"image": f"{name}.png", "masks": f"{name}_masks.npz", "objects": objs, **meta}


def main():
    os.makedirs(OUT, exist_ok=True)
    pose = json.load(open(os.path.join(YG, "top_camera_pose_at_capture_20261103-1421.json")))
    T = np.array(pose["T_base_from_camera"])
    K_top = np.load(os.path.join(YG, "threading/cable_frame_20261103-1421.npz"))["K"]
    K_wr = np.load(
        os.path.join(YG, "real/snapshots/20261103-114411-threading-setup/wrist_left.npz")
    )["K"]
    labels = json.load(open(os.path.join(HERE, "station_small_parts_labels.json")))
    bgs_top = {
        n: np.asarray(Image.open(os.path.join(HERE, "frames", f"{n}.png")).convert("RGB"))
        for n in labels
    }
    wr_dir = os.path.join(HERE, "wrist_bg")
    bgs_wr = [
        np.asarray(Image.open(os.path.join(wr_dir, f)).convert("RGB"))
        for f in sorted(os.listdir(wr_dir))
    ]
    index = []
    # overhead: table region in base xy that the camera sees (from the calibrated pose; checked by projection)
    for s in range(N_TOP):
        name = list(bgs_top)[s % len(bgs_top)]
        bg = bgs_top[name]
        objs = sample_objects(int(rng.integers(6, 13)), rng)
        for _ in range(30):
            placed = place(objs, (0.0, 0.60, 0.20, 0.62), rng)
            rgb, inst = render(placed, T[:3, :3], T[:3, 3], K_top, rng)
            # reject parts drawn over the frame's labelled objects or above the table strip (rows < 255)
            bad = set()
            for i in range(len(placed)):
                m = inst == i
                if not m.any():
                    continue
                ys, xs = np.nonzero(m)
                if ys.min() < 262:
                    bad.add(i)
                for g in labels[name]:
                    x1, y1, x2, y2 = g["box"]
                    if ((xs >= x1 - 4) & (xs <= x2 + 4) & (ys >= y1 - 4) & (ys <= y2 + 4)).any():
                        bad.add(i)
            if len(bad) <= len(placed) // 3:
                placed = [p for i, p in enumerate(placed) if i not in bad]
                break
            objs = sample_objects(int(rng.integers(6, 13)), rng)
        rgb, inst = render(placed, T[:3, :3], T[:3, 3], K_top, rng)
        img = composite(bg, rgb, inst, rng)
        index.append(
            save(f"top_{s:03d}", img, inst, placed, {"view": "overhead", "background": name})
        )
        print("top", s, len(index[-1]["objects"]), flush=True)
    for s in range(N_WRIST):
        bg = bgs_wr[s % len(bgs_wr)]
        h = rng.uniform(0.18, 0.35)
        cx, cy = rng.uniform(0.15, 0.45), rng.uniform(0.25, 0.5)
        R = look_down(rng.uniform(0, 2 * math.pi), math.radians(rng.uniform(0, 20)))
        t = np.array([cx, cy, TABLE_Z + h]) - R[:, 2] * 0.0
        # the region the camera sees on the table, roughly
        half = h * 240 / K_wr[1, 1]
        objs = sample_objects(int(rng.integers(3, 7)), rng)
        # image rows 0-62 % map to the far part of the view: shift the region toward +y (camera up = -y_cam)
        placed = place(
            objs, (cx - half * 1.1, cx + half * 1.1, cy - half * 0.9, cy + half * 0.5), rng
        )
        rgb, inst = render(placed, R, t, K_wr, rng)
        for i in range(len(placed)):  # fingers occupy the bottom of the real wrist frames
            ys = np.nonzero(inst == i)[0]
            if len(ys) and ys.max() > 0.62 * 480:
                inst[inst == i] = -1
        img = composite(bg, rgb, inst, rng)
        index.append(
            save(f"wrist_{s:03d}", img, inst, placed, {"view": "wrist", "height_m": round(h, 3)})
        )
        print("wrist", s, len(index[-1]["objects"]), flush=True)
    json.dump(
        {"about": __doc__, "seed": SEED, "scenes": index},
        open(os.path.join(OUT, "labels.json"), "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
