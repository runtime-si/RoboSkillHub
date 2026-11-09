"""Live check of curobo.plan_with_grasped_object: a YAM arm carries a held 30 mm cube over a wall.

Not collected by pytest. Run on a GPU host from the repository root:

  PYTHONPATH=CUROBO_SRC python tests/bundles/_curobo_attach_live.py YAM_DIR OUT

CUROBO_SRC is a checkout of NVlabs/curobo at 4ea77366ca48ee453e7df139e39fa6532af49f3b (the bundle's pin;
pure Python on v0.8). The interpreter needs torch with CUDA, warp-lang, cuda-core, trimesh, scipy,
pyyaml and gap-core. YAM_DIR is robosimstudio's assets/embodiments/yam_i2rt (yam_i2rt_arm.urdf,
yam_i2rt_arms.xml, assets/*.stl; i2rt's MIT meshes).

The bundle ships no YAM config, so this writes one to OUT/yam.yml: the URDF's chain, collision spheres
from cuRobo's own MORPHIT fit to the MJCF meshes' convex hulls (finger blades posed for a 30 mm grip),
each link's radii grown until they enclose its hull, tool frame ``tcp`` at the MJCF grasp site and
``attached_object`` on the same frame. It registers that file as yam.yml and calls the tool itself
(tools/curobo/tools.py) on: the carry over an 8 cm wall, ten times with a fresh sphere fit and five on
the cached one; the same query without the cube; targets just above the wall top; a 10 cm wall; and
the explicit failures (missing object, target inside the wall). Writes OUT/result.json.
"""

from __future__ import annotations

import importlib
import json
import shutil
import statistics
import sys
import time
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml
from curobo._src.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh

REPO = Path(__file__).resolve().parents[2]
tools = impl = None


def load_bundle(yam_yml: Path):
    """Import tools/curobo as gap_skills.tools.curobo and register yam_yml as yam.yml."""
    for name, path in (
        ("gap_skills", None),
        ("gap_skills.tools", None),
        ("gap_skills.tools.curobo", REPO / "tools" / "curobo"),
    ):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)] if path else []
        sys.modules[name] = mod
    tools_mod = importlib.import_module("gap_skills.tools.curobo.tools")
    impl_mod = importlib.import_module("gap_skills.tools.curobo._curobo_impl")
    impl_mod._PROJECT_ROBOT_CONFIGS["yam.yml"] = yam_yml
    assert impl_mod._V2_AVAILABLE, "cuRobo v0.8 did not import (is CUROBO_SRC on PYTHONPATH?)"
    return tools_mod, impl_mod


# --- the YAM cuRobo config ---------------------------------------------------

FINGER_SLIDE_M = 0.015
TCP = [0.0, 0.0, -0.1347, 0.0, 1.0, 0.0, 0.0]  # x y z qw qx qy qz in link_6
SPHERES = {
    "base_link": 8,
    "link_1": 8,
    "link_2": 12,
    "link_3": 12,
    "link_4": 8,
    "link_5": 8,
    "link_6": 16,
}
BODY_TO_LINK = {
    "right_base": "base_link",
    **{f"right_link{k}": f"link_{k}" for k in range(1, 7)},
}


def tf(el):
    t = np.eye(4)
    if el.get("pos"):
        t[:3, 3] = [float(v) for v in el.get("pos").split()]
    if el.get("quat"):
        q = np.array([float(v) for v in el.get("quat").split()])
        t[:3, :3] = quat_mat(q / np.linalg.norm(q))
    return t


def collect(body, owner, t_owner_body, meshes, mesh_files):
    name = body.get("name")
    if name in BODY_TO_LINK:
        owner, t_owner_body = BODY_TO_LINK[name], np.eye(4)
    for joint in body.findall("joint"):
        if joint.get("type") == "slide":
            axis = np.array([float(v) for v in joint.get("axis").split()])
            slide = np.eye(4)
            slide[:3, 3] = axis * FINGER_SLIDE_M
            t_owner_body = t_owner_body @ slide
    for geom in body.findall("geom"):
        if geom.get("type") == "mesh":
            m = trimesh.load(mesh_files[geom.get("mesh")], force="mesh").convex_hull
            m.apply_transform(t_owner_body @ tf(geom))
            meshes.setdefault(owner, []).append(m)
    for child in body.findall("body"):
        collect(child, owner, t_owner_body @ tf(child), meshes, mesh_files)


def build_yam(yam: Path, out: Path) -> dict:
    """Fit the YAM spheres and write out/yam.yml (with the URDF beside it)."""
    root = ET.parse(yam / "yam_i2rt_arms.xml").getroot()
    mesh_files = {
        m.get("name"): yam / "assets" / m.get("file") for m in root.find("asset").findall("mesh")
    }
    right = next(b for b in root.iter("body") if b.get("name") == "right_base")
    meshes: dict[str, list] = {}
    collect(right, "base_link", np.eye(4), meshes, mesh_files)

    spheres, fit_report = {}, {}
    t0 = time.monotonic()
    for link, n in SPHERES.items():
        mesh = trimesh.util.concatenate(meshes[link])
        fit = fit_spheres_to_mesh(
            mesh, num_spheres=n, fit_type=SphereFitType.MORPHIT, compute_metrics=True
        )
        c = fit.centers.detach().cpu().numpy().reshape(-1, 3)
        r = fit.radii.detach().cpu().numpy().reshape(-1)
        keep = r > 0
        lo, hi = mesh.bounds
        surf = mesh.sample(4000)
        gap = np.min(np.linalg.norm(surf[:, None] - c[keep][None], axis=-1) - r[keep][None], axis=1)
        # MORPHIT under-covers the hull; grow every sphere by the largest gap so it encloses it.
        grow = max(0.0, float(gap.max())) + 0.001
        fit_report[link] = {"n": int(keep.sum()), "grown_mm": round(1000 * grow, 1)}
        spheres[link] = [
            {"center": [round(float(v), 5) for v in cc], "radius": round(float(rr) + grow, 5)}
            for cc, rr in zip(c[keep], r[keep], strict=True)
        ]
        print(
            f"{link}: {int(keep.sum())}/{n} spheres, extent {np.round(hi - lo, 3).tolist()} m, "
            f"surface outside spheres {100 * (gap > 0).mean():.1f}%, max {1000 * gap.max():.1f} mm "
            f"(radii grown {1000 * grow:.1f} mm)"
        )
    print(f"sphere fit {time.monotonic() - t0:.1f} s")

    links = list(SPHERES)
    ignore = {a: [b for b in links[i + 1 : i + 3]] for i, a in enumerate(links)}
    ignore["link_4"] = ["link_5", "link_6", "attached_object"]
    ignore["link_5"] = ["link_6", "attached_object"]
    ignore["link_6"] = ["attached_object"]
    cfg = {
        "robot_cfg": {
            "kinematics": {
                "format_version": 2.0,
                "urdf_path": "yam_i2rt_arm.urdf",
                "asset_root_path": "",
                "base_link": "base_link",
                "tool_frames": ["tcp"],
                "extra_links": {
                    "tcp": {
                        "parent_link_name": "link_6",
                        "link_name": "tcp",
                        "joint_name": "tcp_joint",
                        "joint_type": "FIXED",
                        "fixed_transform": TCP,
                    },
                    "attached_object": {
                        "parent_link_name": "link_6",
                        "link_name": "attached_object",
                        "joint_name": "attach_joint",
                        "joint_type": "FIXED",
                        "fixed_transform": TCP,
                    },
                },
                "collision_link_names": links + ["attached_object"],
                "collision_spheres": spheres,
                "extra_collision_spheres": {"attached_object": 4},
                "collision_sphere_buffer": 0.0,
                "self_collision_ignore": ignore,
                "self_collision_buffer": {k: 0.0 for k in links + ["attached_object"]},
                "use_global_cumul": True,
                "cspace": {
                    "joint_names": [f"joint{k}" for k in range(1, 7)],
                    "default_joint_position": [0.0, 1.4, 1.2, 0.0, 1.1, 0.0],
                    "cspace_distance_weight": [1.0] * 6,
                    "null_space_weight": [1.0] * 6,
                    "max_acceleration": 15.0,
                    "max_jerk": 500.0,
                },
            }
        }
    }
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(yam / "yam_i2rt_arm.urdf", out / "yam_i2rt_arm.urdf")
    (out / "yam.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out / 'yam.yml'}")
    return fit_report


# --- the live check ----------------------------------------------------------

ROBOT = "yam.yml"
CUBE = 0.03
A = np.array([0.30, 0.15, 0.06])
B = np.array([0.30, -0.15, 0.06])
WALL = (np.array([0.15, -0.02, 0.0]), np.array([0.50, 0.02, 0.08]))
HARD_WALL = (np.array([0.15, -0.02, 0.0]), np.array([0.50, 0.02, 0.10]))
TABLE = (np.array([0.10, -0.45, -0.05]), np.array([0.70, 0.45, 0.0]))
DOWN_QUATS = [
    np.array([0.0, 1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0, 0.0]),
    np.array([0.0, 0.7071068, 0.7071068, 0.0]),
    np.array([0.0, 0.7071068, -0.7071068, 0.0]),
]


def box_mesh(name, lo, hi, center=None):
    m = trimesh.creation.box(extents=hi - lo)
    c = (lo + hi) / 2 if center is None else center
    return {
        "name": name,
        "vertices": np.asarray(m.vertices, dtype=np.float32),
        "faces": np.asarray(m.faces, dtype=np.int32),
        "pose": {
            "position": {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
    }


def se3(p, q):
    return {
        "position": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
        "rotation": {"w": float(q[0]), "x": float(q[1]), "y": float(q[2]), "z": float(q[3])},
    }


def quat_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def box_dist(points, box):
    """Signed distance of points to an axis-aligned box (negative inside)."""
    lo, hi = box
    c, h = (lo + hi) / 2, (hi - lo) / 2
    d = np.abs(points - c) - h
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    inside = np.minimum(np.max(d, axis=-1), 0.0)
    return outside + inside


def cube_surface(n=7):
    g = np.linspace(-CUBE / 2, CUBE / 2, n)
    u, v = np.meshgrid(g, g)
    u, v = u.ravel(), v.ravel()
    s = np.full_like(u, CUBE / 2)
    faces = [
        np.c_[s, u, v],
        np.c_[-s, u, v],
        np.c_[u, s, v],
        np.c_[u, -s, v],
        np.c_[u, v, s],
        np.c_[u, v, -s],
    ]
    return np.concatenate(faces)


def planner_for(world_has_obstacles=True):
    return impl._get_attached_planner(
        ROBOT,
        link_name="attached_object",
        with_collision=world_has_obstacles,
        mesh_cache=32,
        position_threshold=0.005,
        rotation_threshold=0.05,
        num_ik_seeds=128,
        use_cuda_graph=False,
        robot_collision_sphere_buffer=0.0,
        collision_activation_distance=0.01,
    )


def fk_tcp(planner, traj):
    js = impl._V2JointState.from_numpy(
        joint_names=planner.joint_names,
        position=np.asarray(traj, dtype=np.float32),
        device_cfg=planner.config.device_cfg,
    )
    kin = planner.compute_kinematics(js)
    pose = kin.tool_poses.get_link_pose("tcp")
    return (
        pose.position.detach().cpu().numpy().reshape(-1, 3),
        pose.quaternion.detach().cpu().numpy().reshape(-1, 4),
    )


def robot_sphere_clearance(planner, traj, boxes):
    js = impl._V2JointState.from_numpy(
        joint_names=planner.joint_names,
        position=np.asarray(traj, dtype=np.float32),
        device_cfg=planner.config.device_cfg,
    )
    sph = (
        planner.compute_kinematics(js)
        .robot_spheres.detach()
        .cpu()
        .numpy()
        .reshape(len(traj), -1, 4)
    )
    out = {}
    for name, box in boxes.items():
        valid = sph[..., 3] > 0
        d = box_dist(sph[..., :3], box) - sph[..., 3]
        out[name] = float(d[valid].min())
    return out


def cube_clearance(planner, traj, q_cube_in_tcp, boxes):
    pos, quat = fk_tcp(planner, traj)
    pts = cube_surface()
    out = {name: np.inf for name in boxes}
    for p, q in zip(pos, quat, strict=True):
        world = (quat_mat(q) @ (q_cube_in_tcp @ pts.T)).T + p
        for name, box in boxes.items():
            out[name] = min(out[name], float(box_dist(world, box).min()))
    return out, pos, quat


def angle_deg(q1, q2):
    d = abs(float(np.dot(q1, q2)))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


def main():
    global tools, impl
    yam_dir, out_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    res: dict = {"gpu": torch.cuda.get_device_name(0), "curobo": "NVlabs/curobo@4ea7736"}
    res["yam_sphere_fit"] = build_yam(yam_dir, out_dir)
    tools, impl = load_bundle(out_dir / "yam.yml")
    events = out_dir / "debug" / "data" / "gap" / "skills" / "curobo" / "events.jsonl"
    t0 = time.monotonic()
    planner = planner_for()
    res["planner_create_s"] = time.monotonic() - t0
    kp = [
        s.kinematics.config.kinematics_config.link_spheres.data_ptr()
        for s in (planner.ik_solver, planner.trajopt_solver)
    ]
    res["ik_and_trajopt_share_link_spheres"] = kp[0] == kp[1]
    res["tool_frames"] = planner.tool_frames
    res["attach_slots"] = impl._attach_ops.sphere_slots(planner, "attached_object")

    # Start configuration: robot-only plan from home to A, gripper down.
    obstacles = {"wall": WALL, "table": TABLE}
    scene = impl._world_to_v2_scene_cfg(
        tools._world_ns({"meshes": [box_mesh("wall", *WALL), box_mesh("table", *TABLE)]}),
        planner.config.device_cfg,
    )
    planner.clear_scene_cache()
    planner.update_world(scene)
    home = np.array([0.0, 1.4, 1.2, 0.0, 1.1, 0.0])
    q_start = None
    for q in DOWN_QUATS:
        r = planner.plan_pose(
            impl._attach_ops.goal(planner, "tcp", A, q),
            impl._attach_ops.start_state(planner, home),
            max_attempts=4,
        )
        ok, traj, _ = impl._attach_ops.outcome(r, 6)
        if ok and traj is not None:
            q_start = traj[-1]
            break
    assert q_start is not None, "no gripper-down configuration reaches A"
    p0, qt0 = fk_tcp(planner, q_start[None])
    res["start"] = {"q": q_start.tolist(), "tcp": p0[0].tolist(), "tcp_quat": qt0[0].tolist()}
    tcp_quat = qt0[0]

    cube = box_mesh("cube", -np.full(3, CUBE / 2), np.full(3, CUBE / 2), center=p0[0])
    world = {"meshes": [box_mesh("table", *TABLE), box_mesh("wall", *WALL), cube]}
    q_cube_in_tcp = quat_mat(tcp_quat).T  # cube axis-aligned in the world at the start

    def carry(target, q=tcp_quat, name="cube", w=world):
        t = time.monotonic()
        out = tools.plan_with_grasped_object(
            world_config=w,
            start_joint_position={"positions": q_start},
            target_pose=se3(target, q),
            object_name=name,
            robot_file=ROBOT,
            position_threshold=0.005,
            position_threshold_z=0.005,
            rotation_threshold=0.05,
            robot_collision_sphere_buffer=0.0,
            debug_out_dir=str(out_dir / "debug"),
        )
        dt = time.monotonic() - t
        last = json.loads(events.read_text().splitlines()[-1]) if events.exists() else {}
        out["timing"] = {k: last.get(k) for k in ("attach_s", "plan_s", "n_spheres")}
        return out, dt

    # 1. The carry over the wall: each run with a fresh MORPHIT fit (the fit is
    #    stochastic and decides the outcome), then repeats on the cached fit.
    def row(out, dt):
        return {
            "success": out["success"],
            "failure_reason": out["failure_reason"],
            "wall_s": dt,
            **out["timing"],
        }

    fresh, best = [], None
    for _ in range(10):
        impl._attach_fit_cache.clear()
        out, dt = carry(B)
        fresh.append(row(out, dt))
        if out["success"] and best is None:
            best = out
            impl._attach_fit_cache_keep = dict(impl._attach_fit_cache)
    res["carry_fresh_fit_runs"] = fresh
    res["carry_fresh_fit_success_rate"] = f"{sum(r['success'] for r in fresh)}/{len(fresh)}"
    if best is not None:
        impl._attach_fit_cache.clear()
        impl._attach_fit_cache.update(impl._attach_fit_cache_keep)
    cached = [row(*carry(B)) for _ in range(5)]
    res["carry_cached_fit_runs"] = cached
    res["carry_cached_fit_success_rate"] = f"{sum(r['success'] for r in cached)}/{len(cached)}"
    for name, rows in (("fresh", fresh), ("cached", cached)):
        for k in ("wall_s", "attach_s", "plan_s"):
            vals = [r[k] for r in rows if r.get(k) is not None]
            res[f"carry_{name}_median_{k}"] = statistics.median(vals) if vals else None

    out = best or {"success": False}
    if out["success"]:
        traj = np.stack([w["positions"] for w in out["trajectory"]["waypoints"]])
        clear, pos, quat = cube_clearance(planner, traj, q_cube_in_tcp, obstacles)
        res["carry"] = {
            "waypoints": len(traj),
            "cube_min_clearance_m": clear,
            "robot_min_clearance_m": robot_sphere_clearance(planner, traj, obstacles),
            "tcp_max_z_m": float(pos[:, 2].max()),
            "tcp_path_len_m": float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()),
            "end_pos_err_mm": float(np.linalg.norm(pos[-1] - B) * 1000),
            "max_tcp_rot_drift_deg": max(angle_deg(q, tcp_quat) for q in quat),
            "start_matches_q_start": bool(np.allclose(traj[0], q_start, atol=1e-3)),
        }
        straight = np.linspace(p0[0], B, 50)
        res["straight_line_cube_wall_clearance_m"] = float(
            min(
                box_dist(
                    (quat_mat(tcp_quat) @ (q_cube_in_tcp @ cube_surface().T)).T + p, WALL
                ).min()
                for p in straight
            )
        )

    # 1b. The same query without the cube: does the robot-only path leave room for it?
    planner.clear_scene_cache()
    planner.update_world(scene)
    r = planner.plan_pose(
        impl._attach_ops.goal(planner, "tcp", B, tcp_quat),
        impl._attach_ops.start_state(planner, q_start),
        max_attempts=8,
    )
    ok, traj, _ = impl._attach_ops.outcome(r, 6)
    res["robot_only_carry_success"] = bool(ok and traj is not None)
    if ok and traj is not None:
        clear, pos, _ = cube_clearance(planner, traj, q_cube_in_tcp, obstacles)
        res["robot_only_path"] = {
            "cube_min_clearance_if_held_m": clear,
            "robot_min_clearance_m": robot_sphere_clearance(planner, traj, obstacles),
            "tcp_max_z_m": float(pos[:, 2].max()),
        }

    # 2. Attachment read-back and sphere coverage of the cube, from the impl's last attach.
    sph = np.array([np.r_[c, r] for c, r in impl._last_post_attach_spheres])
    surf = (quat_mat(tcp_quat) @ (q_cube_in_tcp @ cube_surface(11).T)).T + p0[0]
    gap = np.min(
        np.linalg.norm(surf[:, None, :] - sph[None, :, :3], axis=-1) - sph[None, :, 3], axis=1
    )
    local = np.abs(sph[:, :3] - p0[0])
    res["attached_spheres"] = {
        "max_protrusion_beyond_cube_mm": float(np.max(local + sph[:, 3:4] - CUBE / 2) * 1000),
        "n": len(sph),
        "radius_range_mm": [float(sph[:, 3].min() * 1000), float(sph[:, 3].max() * 1000)],
        "cube_surface_points_outside_spheres_pct": float((gap > 0).mean() * 100),
        "max_uncovered_depth_mm": float(gap.max() * 1000),
    }

    # 3. Targets where only the cube would hit the wall top: robot-only plan vs the held-object plan.
    neg = []
    for dz in (0.005, 0.010, 0.020, 0.040):
        target = np.array([0.30, 0.0, WALL[1][2] + dz])
        held, _ = carry(target)
        planner.clear_scene_cache()
        planner.update_world(scene)
        r = planner.plan_pose(
            impl._attach_ops.goal(planner, "tcp", target, tcp_quat),
            impl._attach_ops.start_state(planner, q_start),
            max_attempts=8,
        )
        robot_only, traj, _ = impl._attach_ops.outcome(r, 6)
        neg.append(
            {
                "tcp_above_wall_top_mm": dz * 1000,
                "cube_penetrates_wall_mm": max(0.0, (CUBE / 2 - dz) * 1000),
                "robot_only_success": bool(robot_only and traj is not None),
                "held_object_success": held["success"],
                "held_object_failure_reason": held["failure_reason"],
            }
        )
    res["wall_top_targets"] = neg

    # 3b. The 10 cm wall: needs a higher lift near YAM's gripper-down reach.
    hard_world = {"meshes": [box_mesh("table", *TABLE), box_mesh("wall", *HARD_WALL), cube]}
    hard = []
    for _ in range(3):
        out, dt = carry(B, w=hard_world)
        hard.append(
            {
                "success": out["success"],
                "failure_reason": out["failure_reason"],
                "wall_s": dt,
                **out["timing"],
            }
        )
    hard_scene = impl._world_to_v2_scene_cfg(
        tools._world_ns({"meshes": [box_mesh("wall", *HARD_WALL), box_mesh("table", *TABLE)]}),
        planner.config.device_cfg,
    )
    planner.clear_scene_cache()
    planner.update_world(hard_scene)
    r = planner.plan_pose(
        impl._attach_ops.goal(planner, "tcp", B, tcp_quat),
        impl._attach_ops.start_state(planner, q_start),
        max_attempts=8,
    )
    ok, traj, _ = impl._attach_ops.outcome(r, 6)
    hard_robot = {"success": bool(ok and traj is not None)}
    if hard_robot["success"]:
        hard_robot["cube_min_clearance_if_held_m"] = cube_clearance(
            planner, traj, q_cube_in_tcp, {"wall": HARD_WALL}
        )[0]
    res["hard_wall_10cm"] = {"held_object_runs": hard, "robot_only": hard_robot}

    # 4. Explicit failures through the tool.
    missing, _ = carry(B, name="brick")
    res["missing_object"] = missing["failure_reason"]
    inside = np.array([0.30, 0.0, 0.09])
    blocked, _ = carry(inside)
    res["target_inside_wall"] = {
        "success": blocked["success"],
        "failure_reason": blocked["failure_reason"],
    }
    after, _ = carry(B)
    res["carry_after_failures_success"] = after["success"]

    print(json.dumps(res, indent=1, default=float))
    (out_dir / "result.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
