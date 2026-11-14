# SPDX-License-Identifier: Apache-2.0
"""Write yam_spheres.json/.yml from the fitted spheres and check arm-arm distance (spheres vs meshes) on the station.

    python build_and_validate.py --station-mjcf <twin>/assets/yam_station_arms.xml --n 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from fit_spheres import BODIES, URDF_LINK, body_meshes

HERE = Path(__file__).resolve().parent
STATION_ROOT = [-0.3475, 0.0, 0.023]
"""rss.robot(..., at=(-0.3475, 0, BASE_RAISE)) in the twin's scene.py; world z=0 is the bench top."""


def chain(model: mujoco.MjModel, prefix: str) -> dict:
    out = {}
    for b in BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + b)
        parent = model.body_parentid[bid]
        e = {"parent": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent).removeprefix(prefix),
             "pos": model.body_pos[bid].round(7).tolist(), "quat_wxyz": model.body_quat[bid].round(7).tolist()}
        if model.body_jntnum[bid]:
            j = model.body_jntadr[bid]
            e["joint"] = {"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).removeprefix(prefix),
                          "type": "prismatic" if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE else "revolute",
                          "axis": model.jnt_axis[j].round(6).tolist(), "range": model.jnt_range[j].round(5).tolist()}
        out[b] = e
    return out


def world_points(model, data, prefix, local):
    pts = {}
    for b, p in local.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + b)
        R = data.xmat[bid].reshape(3, 3)
        pts[b] = p @ R.T + data.xpos[bid]
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station-mjcf", required=True)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    model = mujoco.MjModel.from_xml_path(str(Path(a.station_mjcf).expanduser()))
    data = mujoco.MjData(model)
    spheres = json.loads((HERE / "_spheres_raw.json").read_text())
    report = json.loads((HERE / "fit_report.json").read_text())

    bases = {}
    for arm in ("left", "right"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_base")
        bases[arm] = {"frame": "station world (bench top z=0)",
                      "pos": (np.array(STATION_ROOT) + model.body_pos[bid]).round(5).tolist(),
                      "quat_wxyz": model.body_quat[bid].round(6).tolist()}
    doc = {
        "robot": "i2rt YAM, 6-DoF + linear_4310 gripper (robosimstudio yam_i2rt MJCF)",
        "units": "metres; sphere centres in each body's frame (MJCF body frame == URDF link frame)",
        "joint1_note": "model joint1 = hardware joint1 - pi/2 (hardware home (pi/2,0,0,0,0,0) is model zero)",
        "arm_bases": bases,
        "link_names_urdf": URDF_LINK,
        "chain": chain(model, "left_"),
        "spheres": spheres,
        "gripper_note": ("tip_left/tip_right ride prismatic joints joint7/joint8 (0..0.0475 m, 0 = closed) on link_6; "
                         "planners without finger joints attach both tips' spheres to link_6 at the open stroke "
                         "via spheres_link6_fingers_open, or at closed via spheres_link6_fingers_closed"),
        "fit": report,
    }
    for name, q in (("open", 0.0475), ("closed", 0.0)):
        merged = list(spheres["gripper"])
        for tip in ("tip_left", "tip_right"):
            c = doc["chain"][tip]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, np.array(c["quat_wxyz"]))
            R = R.reshape(3, 3)
            off = np.array(c["pos"]) + R @ (np.array(c["joint"]["axis"]) * q)
            for s in spheres[tip]:
                merged.append({"center": (R @ np.array(s["center"]) + off).round(5).tolist(), "radius": s["radius"]})
        doc[f"spheres_link6_fingers_{name}"] = merged

    # validation: arm-arm distance, spheres vs mesh surface samples, on random configs within limits
    local = {}
    for b, m in body_meshes(model, "left_").items():
        s, _ = trimesh.sample.sample_surface_even(m, 8000, seed=2)
        local[b] = np.vstack([s, m.vertices])
    sp_local = {b: (np.array([s["center"] for s in v]), np.array([s["radius"] for s in v])) for b, v in spheres.items()}
    rng = np.random.default_rng(0)
    qadr = {}
    for arm in ("left", "right"):
        for k in range(1, 9):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_joint{k}")
            qadr[(arm, k)] = (model.jnt_qposadr[j], model.jnt_range[j])
    rows = []
    from scipy.spatial import cKDTree
    tries = 0
    while len(rows) < a.n and tries < 5000:
        tries += 1
        for (arm, k), (adr, rg) in qadr.items():
            lo, hi = rg
            data.qpos[adr] = rng.uniform(lo, hi)
        mujoco.mj_kinematics(model, data)
        L = np.vstack(list(world_points(model, data, "left_", local).values()))
        Rp = np.vstack(list(world_points(model, data, "right_", local).values()))
        d_mesh = float(cKDTree(Rp).query(L)[0].min())
        if d_mesh > 0.25 or (len(rows) < a.n // 2 and d_mesh > 0.08):
            continue
        sl, sr = [], []
        for side, acc in (("left_", sl), ("right_", sr)):
            for b, (c, r) in sp_local.items():
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side + b)
                acc.append(np.c_[c @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid], r])
        sl, sr = np.vstack(sl), np.vstack(sr)
        d_sph = float((np.linalg.norm(sl[:, None, :3] - sr[None, :, :3], axis=2) - sl[:, None, 3] - sr[None, :, 3]).min())
        rows.append({"d_mesh_m": round(d_mesh, 4), "d_spheres_m": round(d_sph, 4), "gap_m": round(d_mesh - d_sph, 4)})
        print(rows[-1], flush=True)
    gaps = np.array([r["gap_m"] for r in rows])
    summary = {"n": len(rows), "configs_with_mesh_dist_under_8cm": int(sum(r["d_mesh_m"] < 0.08 for r in rows)),
               "gap_mean_m": round(float(gaps.mean()), 4), "gap_min_m": round(float(gaps.min()), 4),
               "gap_max_m": round(float(gaps.max()), 4),
               "sphere_dist_never_exceeds_mesh_dist": bool((gaps >= 0).all()),
               "note": "uniform in joint limits, rejected unless the arms come within 25 cm (half within 8 cm); mesh distance from dense surface samples "
                       "(~2 mm spacing, penetration reads as ~0)", "rows": rows}
    doc["validation_arm_arm"] = summary
    print({k: v for k, v in summary.items() if k != "rows"})
    (HERE / "yam_spheres.json").write_text(json.dumps(doc, indent=1))
    try:
        import yaml
        (HERE / "yam_spheres.yml").write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=120))
    except ImportError:
        (HERE / "yam_spheres.yml").write_text(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
