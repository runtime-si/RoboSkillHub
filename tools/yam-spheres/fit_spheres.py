# SPDX-License-Identifier: Apache-2.0
"""Fit collision spheres to the i2rt YAM (6-DoF + linear_4310 gripper) meshes and write yam_spheres.json/.yml.

Each MJCF body's mesh is placed in its body frame. Spheres are inscribed in the filled mesh (centre on an
interior voxel, radius = distance to the surface) plus ``--margin``, picked greedily to cover the most
uncovered surface samples until ``--coverage`` of them is covered or ``--max-spheres`` is reached; so a
sphere sticks out of the mesh by at most about the margin.

    python fit_spheres.py --mjcf ~/robosimstudio/assets/embodiments/yam_i2rt/yam_i2rt_bimanual.xml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
BODIES = ["base", "link1", "link2", "link3", "link4", "link5", "link6", "tip_left", "tip_right"]
"""One per moving frame; the jointless ``gripper`` body is fixed on ``link6`` and folded into it."""
URDF_LINK = {
    "base": "base_link",
    "link1": "link_1",
    "link2": "link_2",
    "link3": "link_3",
    "link4": "link_4",
    "link5": "link_5",
    "link6": "link_6",
    "tip_left": "tip_left",
    "tip_right": "tip_right",
}


def fixed_pose(model: mujoco.MjModel, top: int, bid: int) -> np.ndarray | None:
    """``bid``'s pose in ``top``'s frame (4x4) if no joint lies between them, else None."""
    T = np.eye(4)
    while bid != top:
        if model.body_jntnum[bid] or bid == 0:
            return None
        A = np.eye(4)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.body_quat[bid])
        A[:3, :3], A[:3, 3] = R.reshape(3, 3), model.body_pos[bid]
        T = A @ T
        bid = model.body_parentid[bid]
    return T


def body_meshes(model: mujoco.MjModel, prefix: str) -> dict[str, trimesh.Trimesh]:
    """Each ``prefix``-body's mesh geoms, with those of bodies fixed to it, as one trimesh in that body's frame."""
    out = {}
    for b in BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + b)
        parts = []
        for g in range(model.ngeom):
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            T = fixed_pose(model, bid, model.geom_bodyid[g])
            if T is None:
                continue
            m = model.geom_dataid[g]
            v = model.mesh_vert[
                model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]
            ].astype(float)
            f = model.mesh_face[
                model.mesh_faceadr[m] : model.mesh_faceadr[m] + model.mesh_facenum[m]
            ]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, model.geom_quat[g])
            v = v @ R.reshape(3, 3).T + model.geom_pos[g]
            parts.append(trimesh.Trimesh(v @ T[:3, :3].T + T[:3, 3], f, process=True))
        out[b] = trimesh.util.concatenate(parts)
    return out


def fill_voxels(mesh: trimesh.Trimesh, pitch: float):
    vox = mesh.voxelized(pitch).fill()
    return vox, vox.points


def fit(
    mesh: trimesh.Trimesh,
    margin: float,
    coverage: float,
    max_spheres: int,
    pitch: float,
    seed: int = 0,
):
    surf, _ = trimesh.sample.sample_surface_even(mesh, 6000, seed=seed)
    surf = np.vstack([surf, mesh.vertices])
    tree = cKDTree(surf)
    _, interior = fill_voxels(mesh, pitch)
    r_in = tree.query(interior)[0]
    radii = r_in + margin
    centre_tree = cKDTree(surf)
    covers = [
        np.array(centre_tree.query_ball_point(c, r)) for c, r in zip(interior, radii, strict=True)
    ]
    uncovered = np.ones(len(surf), bool)
    chosen = []
    while len(chosen) < max_spheres and uncovered.mean() > 1 - coverage:
        gain = np.array([uncovered[c].sum() if len(c) else 0 for c in covers])
        i = int(gain.argmax())
        if gain[i] == 0:
            break
        chosen.append(i)
        uncovered[covers[i]] = False
    return [(interior[i].tolist(), float(radii[i])) for i in chosen]


def evaluate(mesh: trimesh.Trimesh, spheres, pitch: float):
    """Fraction of mesh vertices inside the sphere union, and the largest protrusion of a sphere outside the mesh."""
    c = np.array([s[0] for s in spheres])
    r = np.array([s[1] for s in spheres])
    v = mesh.vertices
    d = np.linalg.norm(v[:, None] - c[None], axis=2) - r[None]
    inside = (d.min(1) <= 1e-9).mean()
    vox = mesh.voxelized(pitch / 2).fill()
    surf, _ = trimesh.sample.sample_surface_even(mesh, 20000, seed=1)
    tree = cKDTree(np.vstack([surf, v]))
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(400, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts = (c[:, None] + r[:, None, None] * dirs[None]).reshape(-1, 3)
    out = ~vox.is_filled(pts)
    prot = tree.query(pts[out])[0].max() if out.any() else 0.0
    return float(inside), float(prot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--margin", type=float, default=0.015)
    ap.add_argument("--coverage", type=float, default=0.995)
    ap.add_argument("--max-spheres", type=int, default=16)
    ap.add_argument(
        "--link-max",
        action="append",
        default=["link3=32"],
        metavar="BODY=N",
        help="per-body sphere cap overriding --max-spheres",
    )
    ap.add_argument("--pitch", type=float, default=0.004)
    a = ap.parse_args()
    model = mujoco.MjModel.from_xml_path(str(Path(a.mjcf).expanduser()))
    meshes = body_meshes(model, "left_")
    caps = {k: int(v) for k, v in (x.split("=") for x in a.link_max)}
    links, report = {}, {}
    for b, m in meshes.items():
        sp = fit(m, a.margin, a.coverage, caps.get(b, a.max_spheres), a.pitch)
        cov, prot = evaluate(m, sp, a.pitch)
        links[b] = [{"center": [round(x, 5) for x in c], "radius": round(r, 5)} for c, r in sp]
        report[b] = {
            "n_spheres": len(sp),
            "vertex_coverage": round(cov, 4),
            "max_protrusion_m": round(prot, 4),
            "n_vertices": int(len(m.vertices)),
        }
        print(b, report[b], flush=True)
    (HERE / "fit_report.json").write_text(
        json.dumps(
            {
                "margin_m": a.margin,
                "max_spheres": a.max_spheres,
                "link_max": caps,
                "target_coverage": a.coverage,
                "pitch_m": a.pitch,
                "links": report,
            },
            indent=1,
        )
    )
    (HERE / "_spheres_raw.json").write_text(json.dumps(links, indent=1))


if __name__ == "__main__":
    main()
