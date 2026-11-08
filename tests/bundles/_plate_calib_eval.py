"""Accuracy of calibrating-baseplate-stud-grid on random ray-cast renders (CPU, a few minutes).

    .venv/bin/python -m tests.bundles._plate_calib_eval --trials 30 [--real]

Each calibration trial: a plate at a random yaw and up to 2 deg tilt, 0-14 random bricks on it,
two 40 mm ArUco markers on the table beside it; an overhead frame (0.85-0.95 m, 640 x 480,
fx 461) and a wrist frame (0.18-0.25 m, up to 25 deg off vertical, fx 394, D405 noise), both
2x supersampled. Rows: the coarse pose from the markers, then refined; a coarse pose given with
a random 0-2.5 mm / 0-2 deg error, then refined; the same with the wrist camera's pose off by
0.5 mm and 0.2 deg (hand-eye error, which the fit cannot see). read_studs trials: a random brick
(1x1 to 2x6) at a random yaw seen from 0.15-0.20 m. --real runs read_studs on the station's
D405 frames on disk that show a hole lattice.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from gap.testing import FakeContext

from . import _plate_render as R

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "skills" / "calibrating-baseplate-stud-grid" / "scripts"


def _load(name):
    import importlib.util
    import sys

    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _se3(m):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(m[:3, :3]).as_quat()
    return {
        "position": dict(zip("xyz", map(float, m[:3, 3]), strict=True)),
        "rotation": {"w": w, "x": x, "y": y, "z": z},
    }


def _m(pose):
    from scipy.spatial.transform import Rotation

    q = pose["rotation"]
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    m[:3, 3] = [pose["position"][k] for k in "xyz"]
    return m


def _err(pose, truth):
    e = np.linalg.inv(truth) @ _m(pose)
    return (
        math.hypot(e[0, 3], e[1, 3]) * 1e3,
        abs(e[2, 3]) * 1e3,
        abs(math.degrees(math.atan2(e[1, 0], e[0, 0]))),
        math.degrees(math.acos(min(1.0, e[2, 2]))),
    )


def _perturb(rng, mm, deg):
    from scipy.spatial.transform import Rotation

    m = np.eye(4)
    axis = rng.normal(size=3)
    m[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * math.radians(deg)).as_matrix()
    d = rng.normal(size=3)
    m[:3, 3] = d / np.linalg.norm(d) * mm / 1e3
    return m


def calibration_trials(n, system, seed):
    cal = _load("calibrate_plate")
    rng = np.random.default_rng(seed)
    g = R.SYSTEMS[system]
    p, pt = g["pitch"], g["plate_t"]
    studs = (32, 32) if system == "lego" else (16, 16)
    rows = {
        k: []
        for k in (
            "markers coarse",
            "markers + lattice",
            "given coarse",
            "given + lattice",
            "hand-eye 0.5 mm / 0.2 deg",
        )
    }
    routes = {k: {} for k in rows}
    for _ in range(n):
        bricks = []
        for _b in range(rng.integers(0, 15)):
            nx, ny = (int(v) for v in rng.choice([1, 2, 4], 2))
            bricks.append(
                R.Brick(
                    int(rng.integers(0, studs[0] - nx)),
                    int(rng.integers(0, studs[1] - ny)),
                    nx,
                    ny,
                    1,
                    int(rng.integers(3)),
                )
            )
        markers = [
            R.Marker(3, 0.04, studs[0] * p / 2 + 0.035, 0.0, -pt),
            R.Marker(7, 0.04, -studs[0] * p / 2 - 0.035, 0.05, -pt, 30.0),
        ]
        scene = R.Scene(system=system, studs=studs, bricks=bricks, markers=markers)
        layout = [
            {"id": 3, "size": 0.04, "position": [studs[0] * p / 2 + 0.035, 0.0, -pt]},
            {
                "id": 7,
                "size": 0.04,
                "position": [-studs[0] * p / 2 - 0.035, 0.05, -pt],
                "yaw_deg": 30.0,
            },
        ]
        tw = R.plate_pose_world(
            0.45, -0.05, 0.02, rng.uniform(-180, 180), tuple(rng.uniform(-1.4, 1.4, 2))
        )
        half = studs[0] * p / 2
        target = np.r_[rng.uniform(-0.4, 0.4, 2) * half, 0.0]
        rngm = rng.uniform(0.18, 0.25)
        off = math.radians(rng.uniform(0, 25))
        az = rng.uniform(0, 2 * math.pi)
        eye = target + rngm * np.array(
            [math.sin(off) * math.cos(az), math.sin(off) * math.sin(az), math.cos(off)]
        )
        wrist_p = R.look_at(eye, target, up=(math.cos(az + 1.0), math.sin(az + 1.0), 0.0))
        top_p = R.look_at([rng.uniform(-0.05, 0.05), -0.25, rng.uniform(0.85, 0.95)], [0, 0, 0])
        rgb, d = R.render(scene, R.D405_K, wrist_p, rng=rng)
        wrist = {
            "name": "left_wrist",
            "rgb": rgb,
            "depth": d,
            "intrinsics": R.D405_K,
            "pose": _se3(tw @ wrist_p),
        }
        rgb, d = R.render(scene, R.ORBBEC_K, top_p, rng=rng, baseline=0.05)
        top = {
            "name": "top",
            "rgb": rgb,
            "depth": d,
            "intrinsics": R.ORBBEC_K,
            "pose": _se3(tw @ top_p),
        }
        obs = {"cameras": [top, wrist], "arms": []}
        ctx = FakeContext({"robot.get_observation": obs})
        res = {}
        res["markers coarse"] = cal.run(ctx, list(studs), system, markers=layout, refine=False)
        res["markers + lattice"] = cal.run(ctx, list(studs), system, markers=layout, look=False)
        coarse = _se3(tw @ _perturb(rng, rng.uniform(0, 2.5), rng.uniform(0, 2.0)))
        res["given coarse"] = {"route": "coarse_only", "plate_pose": coarse}
        res["given + lattice"] = cal.run(
            ctx, list(studs), system, coarse_plate_pose=coarse, look=False
        )
        bad = dict(wrist, pose=_se3(tw @ wrist_p @ _perturb(rng, 0.5, 0.2)))
        ctx2 = FakeContext({"robot.get_observation": {"cameras": [top, bad], "arms": []}})
        res["hand-eye 0.5 mm / 0.2 deg"] = cal.run(
            ctx2, list(studs), system, coarse_plate_pose=coarse, look=False
        )
        for k, o in res.items():
            routes[k][o["route"]] = routes[k].get(o["route"], 0) + 1
            if o.get("plate_pose"):
                rows[k].append(
                    (
                        *_err(o["plate_pose"], tw),
                        o.get("studs_used", 0),
                        o.get("residual_mm", 0.0),
                        o["route"],
                    )
                )
    return rows, routes


def studs_trials(n, seed):
    rs = _load("read_studs")
    rng = np.random.default_rng(seed)
    sizes = [(1, 1), (1, 2), (1, 4), (2, 2), (2, 3), (2, 4), (1, 6), (2, 6)]
    out = []
    for _ in range(n):
        a, b = sizes[rng.integers(len(sizes))]
        if rng.random() < 0.5:
            a, b = b, a
        brick = R.Brick(5, 4, a, b, 1, int(rng.integers(3)))
        scene = R.Scene(system="lego", studs=(16, 16), bricks=[brick])
        p, bh = 0.008, 0.0096
        cx, cy = -8 * p + (5 + a / 2) * p, -8 * p + (4 + b / 2) * p
        yaw = rng.uniform(-180, 180)
        tw = R.plate_pose_world(0.4, 0.1, 0.0, yaw)
        rngm, off, az = (
            rng.uniform(0.15, 0.20),
            math.radians(rng.uniform(0, 25)),
            rng.uniform(0, 2 * math.pi),
        )
        target = np.array([cx, cy, bh])
        eye = target + rngm * np.array(
            [math.sin(off) * math.cos(az), math.sin(off) * math.sin(az), math.cos(off)]
        )
        cam_p = R.look_at(eye, target, up=(math.cos(az + 1.0), math.sin(az + 1.0), 0.0))
        rgb, d = R.render(scene, R.D405_K, cam_p, rng=rng)
        mask = R.render.last_ids == 2
        frame = {
            "name": "w",
            "rgb": rgb,
            "depth": d,
            "intrinsics": R.D405_K,
            "pose": _se3(tw @ cam_p),
        }
        o = rs.run(FakeContext({}), mask=mask, frame=frame)
        ok_size = o["brick_size"] == sorted([a, b])
        if o["top_pose"]:
            t = _m(o["top_pose"])
            top = tw @ np.array([cx, cy, bh, 1.0])
            pos = float(np.linalg.norm(t[:3, 3] - top[:3])) * 1e3
            want = yaw + (90.0 if a > b else 0.0)
            fold = 90.0 if a == b else 180.0
            got = math.degrees(math.atan2(t[1, 0], t[0, 0]))
            dyaw = abs((got - want + fold / 2) % fold - fold / 2)
        else:
            pos, dyaw = float("nan"), float("nan")
        out.append(((a, b), o["route"], ok_size, pos, dyaw))
    return out


def real_frames():
    rs = _load("read_studs")
    import cv2

    path = Path.home() / "yam_gap/threading/cable_frame_20261103-1421.npz"
    if not path.is_file():
        print("real: no frames on disk")
        return
    d = np.load(path)
    mask = np.zeros((480, 640), np.uint8)
    cv2.fillConvexPoly(
        mask, np.array([[240, 314], [374, 314], [378, 404], [226, 404]], np.int32), 1
    )
    frame = {"name": "wrist_left", "rgb": d["rgb"], "depth": d["depth_m"], "intrinsics": d["K"]}
    defaults = list(rs.lat.backproject.__defaults__)
    print(
        f"real {path.name}: black perforated board at {np.median(d['depth_m'][mask > 0]):.2f} m, detector=rgb, MAX_DEPTH lifted to 2 m for this check"
    )
    rs.lat.backproject.__defaults__ = (*defaults[:-1], 2.0)
    try:
        for pitch in (0.0125, 0.020, 0.025, 0.0254, 0.050):
            o = rs.run(
                FakeContext({}),
                mask=mask,
                frame=frame,
                stud_pitch=pitch,
                stud_diameter=0.006,
                stud_height=0.001,
                detector="rgb",
            )
            print(
                f"  pitch {pitch * 1e3:5.1f} mm: {o['route']:13s} {o['brick_size']} {o['stud_count']:3d} holes, {o['residual_mm']:.2f} mm RMS"
            )
    finally:
        rs.lat.backproject.__defaults__ = tuple(defaults)


def _stats(v):
    v = np.asarray(v, float)
    return f"{np.median(v):.2f} / {np.percentile(v, 90):.2f} / {v.max():.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--studs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    print(
        "| system | step | routes | lateral mm med / p90 / max | z mm med / p90 / max | yaw deg med / p90 / max | tilt deg med / p90 / max | studs used med | RMS mm med |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for system in ("lego", "duplo"):
        rows, routes = calibration_trials(a.trials, system, a.seed)
        for k, r in rows.items():
            if not r:
                continue
            arr = np.array([x[:6] for x in r], float)
            rt = ", ".join(f"{kk} {vv}" for kk, vv in sorted(routes[k].items()))
            used = f"{np.median(arr[:, 4]):.0f}" if "lattice" in k or "hand" in k else "-"
            rms = f"{np.median(arr[:, 5]):.2f}" if "lattice" in k or "hand" in k else "-"
            print(
                f"| {system} | {k} | {rt} | {_stats(arr[:, 0])} | {_stats(arr[:, 1])} | {_stats(arr[:, 2])} | {_stats(arr[:, 3])} | {used} | {rms} |"
            )
    st = studs_trials(a.studs, a.seed + 1)
    found = [s for s in st if s[1] == "lattice_found"]
    print(
        f"\nread_studs: {len(st)} bricks; routes "
        + ", ".join(
            f"{r} {sum(1 for s in st if s[1] == r)}"
            for r in ("lattice_found", "occluded", "too_few_studs")
        )
    )
    print(
        f"  right size {sum(s[2] for s in st)}/{len(st)}; lattice_found with the wrong size {sum(1 for s in found if not s[2])}"
    )
    if found:
        print(
            f"  top centre error mm med / p90 / max {_stats([s[3] for s in found])}; yaw deg {_stats([s[4] for s in found])}"
        )
    for s in st:
        if s[1] != "lattice_found" or not s[2]:
            print("   ", s)
    if a.real:
        real_frames()
    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
