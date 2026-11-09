# SPDX-License-Identifier: Apache-2.0
"""Run isolated YAM PhysX acceptance without opening robot sockets.

Run with an existing Isaac Sim 6.2 Python and an installed RoboSkillHub runtime::

    PYTHONPATH=src /path/to/isaac/python isaac_smoke.py \
        --rss-assets /path/to/robosimstudio/assets/embodiments/yam_i2rt \
        --output /new/isolated/run --task push

The installed 6.2 legacy Core APIs and MJCF importer are used. Inputs remain
read-only; generated XML, URDF, Lula YAML, USD and report.json go under --output.
The source model has per-link gravity compensation; this conversion explicitly
disables gravity on robot links. Dynamic objects retain ordinary gravity.
Initial state resets are setup only. Skill actions use joint targets and physics;
no object pose setters during skill execution, robot-object attachments or synthetic
success feedback are used. Drawer fixture parts have their own physical joints.
This is a controlled single-scene test, not original-station or hardware validation.
"""

import argparse
import copy
import hashlib
import json
import math
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--rss-assets",
    type=Path,
    required=True,
    help="Read-only RSS assets/embodiments/yam_i2rt directory",
)
parser.add_argument(
    "--output",
    type=Path,
    required=True,
    help="New isolated output directory; never use a live twin directory",
)
parser.add_argument(
    "--task", choices=("move_pose", "push", "grasp", "pull_drawer"), default="move_pose"
)
parser.add_argument("--grasp-close-width-m", type=float, default=0.03)
parser.add_argument(
    "--drawer-control", choices=("positive", "miss_handle", "locked"), default="positive"
)
args = parser.parse_args()
ROOT = args.output.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
if (ROOT / "report.json").exists():
    raise SystemExit(
        "Output already contains a report; choose a new directory to preserve evidence"
    )
(ROOT / "executed_script.py").write_bytes(Path(__file__).read_bytes())
assets = args.rss_assets.resolve()
# Derive a single arm without modifying the source model or meshes.
tree = ET.parse(assets / "yam_i2rt_arms.xml")
doc = tree.getroot()
doc.set("model", "yam_left")
doc.find("compiler").set("meshdir", str(assets / "assets"))
worldbody = doc.find("worldbody")
for body in list(worldbody):
    if body.get("name") != "left_base":
        worldbody.remove(body)
worldbody[0].set("pos", "0 0 0")
for section in ("contact", "actuator"):
    elem = doc.find(section)
    for child in list(elem):
        if any("right_" in value for value in child.attrib.values()):
            elem.remove(child)
tree.write(ROOT / "yam_left.xml")
u = ET.parse(assets / "yam_i2rt_arm.urdf")
ur = u.getroot()
ur.set("name", "yam_left")
for element in ur.iter():
    for key, value in list(element.attrib.items()):
        if value == "base_link":
            element.set(key, "left_base")
        elif value.startswith("link_"):
            element.set(key, "left_link" + value[5:])
        elif value.startswith("joint"):
            element.set(key, "left_" + value)
ET.SubElement(ur, "link", name="left_grasp_site")
joint = ET.SubElement(ur, "joint", name="grasp_site_fixed", type="fixed")
ET.SubElement(joint, "parent", link="left_link6")
ET.SubElement(joint, "child", link="left_grasp_site")
ET.SubElement(joint, "origin", xyz="0 0 -0.1347", rpy="3.141592653589793 0 0")
u.write(ROOT / "yam_left.urdf")
# Schema verified against Isaac's shipped Denso/cobotta_pro_1300/rmpflow/
# robot_descriptor.yaml. These acceleration/jerk values are test configuration,
# not measured YAM dynamics. This test uses Lula IK, not collision-aware RMPflow.
(ROOT / "yam_left_descriptor.yaml").write_text(
    "api_version: 1.0\ncspace: [left_joint1, left_joint2, left_joint3, left_joint4, left_joint5, left_joint6]\n"
    "root_link: left_base\ndefault_q: [0.0, 1.4, 1.2, 0.0, 1.1, 0.0]\n"
    "acceleration_limits: [10, 10, 10, 10, 10, 10]\njerk_limits: [1000, 1000, 1000, 1000, 1000, 1000]\n"
    "collision_spheres: []\n"
)
REPORT = {
    "schema_version": 1,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "task": "YAM Isaac Sim isolated " + args.task + " smoke",
    "evaluation_role": (
        "negative"
        if args.task == "pull_drawer" and args.drawer_control != "positive"
        else "positive"
    ),
    "started_at_unix_s": time.time(),
    "not_validated": [
        "real robot",
        "full workstation reconstruction",
        "grasp",
        "push",
        "pull_drawer",
    ],
    "assumptions": [
        "Left arm extracted from RSS i2rt MJCF; root moved to world origin",
        "Joint-state resets during scene setup only; no teleport during skill execution",
        "Lula description created from installed NVIDIA descriptor schema; no collision spheres/planning",
        "Grasp TCP uses source MJCF left_grasp_site offset 0,0,-0.1347 and rotation pi about x",
        "Nominal width-to-joint map uses source tip_gap=2*joint_coordinate; gripper geometry and task outcome require their own evaluation",
    ],
}
app = None
try:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": 640,
            "height": 480,
            "anti_aliasing": 0,
            "renderer": "RaytracedLighting",
            "extra_args": [
                "--/app/settings/persistent=false",
                "--/renderer/multiGpu/enabled=false",
            ],
        }
    )
    import numpy as np
    import omni.usd
    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.robot_motion.motion_generation import (
        ArticulationKinematicsSolver,
        LulaKinematicsSolver,
    )
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    from roboskillhub.runtime.contracts import Limits, pose_error
    from roboskillhub.runtime.engine import Runtime
    from roboskillhub.runtime.isaac import IsaacBackend

    print("SMOKE:imports_ready", flush=True)
    REPORT["isaac_version"] = (
        (Path(__import__("isaacsim").__file__).parent / "VERSION").read_text().strip()
    )
    source = ROOT / "yam_left.xml"
    REPORT["source_mjcf"] = str(source)
    REPORT["original_assets"] = str(assets)
    REPORT["original_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            assets / "yam_i2rt_arms.xml",
            assets / "yam_i2rt_arm.urdf",
            *sorted((assets / "assets").glob("*.stl")),
        ]
    }
    REPORT["runtime_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(
            Path(__import__("roboskillhub.runtime.isaac", fromlist=["x"]).__file__).parent.glob(
                "*.py"
            )
        )
    }
    REPORT["source_mjcf_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    config = MJCFImporterConfig(
        mjcf_path=str(source),
        usd_path=str(ROOT / "converted"),
        fix_base=True,
        run_asset_transformer=False,
        run_multi_physics_conversion=True,
        allow_self_collision=False,
    )
    usd = ROOT / "converted_path.txt"
    if usd.exists() and Path(usd.read_text()).exists():
        usd_path = usd.read_text()
    else:
        usd_path = MJCFImporter(config).import_mjcf()
        usd.write_text(usd_path)
    REPORT["usd_path"] = usd_path
    print("SMOKE:converted", usd_path, flush=True)
    # Importing opens its own temporary stage. Start a new independent World stage.
    omni.usd.get_context().new_stage()
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
    add_reference_to_stage(usd_path=usd_path, prim_path="/World/YAM")
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    REPORT["physics_scenes"] = [
        {
            "path": str(prim.GetPath()),
            "solver_type": prim.GetAttribute("physxScene:solverType").Get(),
            "enable_external_forces_every_iteration": prim.GetAttribute(
                "physxScene:enableExternalForcesEveryIteration"
            ).Get(),
        }
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Scene)
    ]
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    REPORT["articulation_roots"] = roots
    REPORT["prim_summary"] = [
        {"path": str(p.GetPath()), "type": p.GetTypeName()}
        for p in stage.Traverse()
        if "joint" in p.GetName().lower()
        or p.GetName() in ["left_base", "left_link6", "left_gripper", "left_grasp_site"]
    ]
    print("SMOKE:roots", roots, flush=True)
    if len(roots) != 1:
        raise RuntimeError(f"Expected one left-arm articulation, got {roots}")
    # Preserve source gravcomp=1 explicitly for the imported arm; importer conversion
    # does not promise MuJoCo per-body gravcomp semantics in PhysX.
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(True)
    REPORT["gravity_compensation"] = (
        "PhysX disableGravity on robot links only, matching source gravcomp=1; dynamically created cube retains gravity"
    )
    # Refine only the concave source fingertip collision representation. Single
    # convex hulls fill their inner recess and touch the cube on a false sloped
    # face. This is an isolated scene variant, not unchanged source physics.
    REPORT["fingertip_collision_variant"] = {
        "description": "source meshes and transforms unchanged; PhysX convex decomposition preserves finger recesses more closely than a single hull",
        "scope": "both fingertip mesh instances, common to move_pose/push/grasp",
        "meshes": [],
    }
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not any(
            f"/{name}/" in str(prim.GetPath()) for name in ("left_tip_left", "left_tip_right")
        ):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        previous = mesh_collision.GetApproximationAttr().Get()
        mesh_collision.GetApproximationAttr().Set("convexDecomposition")
        decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        decomposition.CreateErrorPercentageAttr(1.0)
        decomposition.CreateShrinkWrapAttr(True)
        decomposition.CreateMaxConvexHullsAttr(32)
        REPORT["fingertip_collision_variant"]["meshes"].append(
            {
                "path": str(prim.GetPath()),
                "previous_approximation": previous,
                "approximation": mesh_collision.GetApproximationAttr().Get(),
                "error_percentage": decomposition.GetErrorPercentageAttr().Get(),
                "shrink_wrap": decomposition.GetShrinkWrapAttr().Get(),
                "max_convex_hulls": decomposition.GetMaxConvexHullsAttr().Get(),
                "hull_vertex_limit": decomposition.GetHullVertexLimitAttr().Get(),
                "voxel_resolution": decomposition.GetVoxelResolutionAttr().Get(),
                "min_thickness_m": decomposition.GetMinThicknessAttr().Get(),
            }
        )
    if len(REPORT["fingertip_collision_variant"]["meshes"]) != 2:
        raise RuntimeError("Expected exactly two source fingertip collision mesh instances")
    stage.GetRootLayer().Export(str(ROOT / "scene_collision_override.usda"))
    robot = world.scene.add(SingleArticulation(prim_path=roots[0], name="yam_left"))
    world.reset()
    names = list(robot.dof_names)
    REPORT["joint_names"] = names
    print("SMOKE:joints", names, flush=True)
    home = {
        f"left_joint{i}": q for i, q in enumerate([0.0, 1.4, 1.2, 0.0, 1.1, 0.0, 0.035, 0.035], 1)
    }
    positions = np.array([home[n] for n in names], dtype=np.float64)
    robot.set_joint_positions(positions)
    robot.set_joint_velocities(np.zeros(len(names)))
    robot.apply_action(ArticulationAction(joint_positions=positions))
    for _ in range(120):
        world.step(render=False)
    lula = LulaKinematicsSolver(
        robot_description_path=str(ROOT / "yam_left_descriptor.yaml"),
        urdf_path=str(ROOT / "yam_left.urdf"),
    )
    kin = ArticulationKinematicsSolver(robot, lula, "left_grasp_site")
    limits = Limits(
        frame_id="world",
        workspace_min_m=(-1, -1, -1),
        workspace_max_m=(1, 1, 1),
        max_speed_m_s=0.05,
        max_contact_distance_m=0.2,
        gripper_min_width_m=0,
        gripper_max_width_m=0.095,
        max_timeout_s=30,
        position_tolerance_m=0.003,
        rotation_tolerance_rad=0.03,
    )
    backend = IsaacBackend(
        world=world,
        robot=robot,
        kinematics_solver=kin,
        limits=limits,
        stage_units_in_meters=1.0,
        gripper_joint_indices=[names.index("left_joint7"), names.index("left_joint8")],
        width_to_positions=lambda w: [w / 2, w / 2],
        render=False,
    )
    start = backend.get_state(timeout_s=5)
    REPORT["initial_state"] = start
    REPORT["base_pose"] = [x.tolist() for x in robot.get_world_pose()]
    REPORT["joint_reset_rad_or_m"] = positions.tolist()
    # Validate FK against independently read PhysX rigid-link transform, with the
    # source grasp-site transform. The solver's own FK is not sufficient evidence.
    from isaacsim.core.prims import SingleRigidPrim

    grip_paths = [
        str(p.GetPath())
        for p in stage.Traverse()
        if p.GetName() == "left_gripper" and p.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if not grip_paths:
        grip_paths = [
            str(p.GetPath())
            for p in stage.Traverse()
            if p.GetName() == "left_link6" and p.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
    if len(grip_paths) != 1:
        raise RuntimeError(f"Cannot identify independent physical gripper: {grip_paths}")
    grip = SingleRigidPrim(prim_path=grip_paths[0], name="independent_gripper")
    grip.initialize()
    from scipy.spatial.transform import Rotation

    def physical_tcp():
        p, q = grip.get_world_pose()
        rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        tcp_p = np.array(p) + rot.apply([0, 0, -0.1347])
        tcp_q = (rot * Rotation.from_euler("x", math.pi)).as_quat()
        return {
            "frame_id": "world",
            "position_m": tcp_p.tolist(),
            "quaternion_xyzw": tcp_q.tolist(),
        }

    actual_before = physical_tcp()
    alignment = pose_error(start["pose"], actual_before)
    REPORT["physical_tcp_prim"] = grip_paths[0]
    REPORT["physical_start_pose"] = actual_before
    REPORT["fk_physics_alignment_error"] = {
        "position_m": alignment[0],
        "rotation_rad": alignment[1],
    }
    print("SMOKE:alignment", alignment, flush=True)
    if alignment[0] > 0.001 or alignment[1] > 0.01:
        raise RuntimeError("Imported physics and Lula FK differ beyond 1mm/0.01rad")
    if args.task == "move_pose":
        target = {
            **start["pose"],
            "position_m": [
                start["pose"]["position_m"][0] + 0.01,
                start["pose"]["position_m"][1],
                start["pose"]["position_m"][2] + 0.01,
            ],
        }
        REPORT["request"] = {"target_pose": target, "speed_m_s": 0.025, "timeout_s": 20.0}
    elif args.task == "push":
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid

        from roboskillhub.runtime.isaac import simulation_object_provider

        # Put a rigid cube next to the existing fingertip geometry, supported by a
        # fixed shelf. The source finger colliders are unchanged; contact physics
        # alone must move the cube. Nothing is attached or repositioned during skill.
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        tip_prims = [
            p for p in stage.Traverse() if p.GetName() in ["left_tip_left", "left_tip_right"]
        ]
        ranges = [bounds.ComputeWorldBound(p).ComputeAlignedRange() for p in tip_prims]
        tip_lo = np.min([list(r.GetMin()) for r in ranges], axis=0)
        tip_hi = np.max([list(r.GetMax()) for r in ranges], axis=0)
        cube_size = 0.04
        top_z = float(tip_lo[2] - 0.002)
        vertices = []
        for tip in tip_prims:
            for prim in Usd.PrimRange(tip):
                if prim.IsA(UsdGeom.Mesh):
                    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    )
                    vertices.extend(
                        [
                            list(transform.Transform(Gf.Vec3d(*map(float, p))))
                            for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()
                        ]
                    )
        vertices = np.asarray(vertices)
        # Select the leading source surface within the cube's vertical contact band,
        # rather than the union AABB whose x/y extrema need not belong to one point.
        band = vertices[(vertices[:, 2] > tip_lo[2] + 0.003) & (vertices[:, 2] < tip_lo[2] + 0.035)]
        leading = band[band[:, 0] > np.max(band[:, 0]) - 0.001]
        contact_surface = np.array(
            [np.max(band[:, 0]), np.mean(leading[:, 1]), np.mean(leading[:, 2])]
        )
        center = np.array(
            [
                contact_surface[0] + cube_size / 2 + 0.002,
                contact_surface[1],
                top_z + cube_size / 2 + 0.001,
            ]
        )
        material = PhysicsMaterial(
            "/World/TestMaterial", static_friction=0.6, dynamic_friction=0.5, restitution=0.0
        )
        shelf = world.scene.add(
            FixedCuboid(
                "/World/TestShelf",
                name="shelf",
                position=np.array([center[0] + 0.06, center[1], top_z - 0.015]),
                scale=np.array([0.25, 0.22, 0.03]),
                size=1.0,
                physics_material=material,
            )
        )
        cube = world.scene.add(
            DynamicCuboid(
                "/World/PushCube",
                name="push_cube",
                position=center,
                scale=np.array([cube_size] * 3),
                size=1.0,
                mass=0.05,
                physics_material=material,
            )
        )
        REPORT["push_scene"] = {
            "cube_initial_position_m": center.tolist(),
            "cube_size_m": cube_size,
            "mass_kg": 0.05,
            "static_friction": 0.6,
            "dynamic_friction": 0.5,
            "shelf_top_z_m": top_z,
            "source_fingertip_aabb_m": [tip_lo.tolist(), tip_hi.tolist()],
            "contact_method": "existing finger meshes only",
            "contact_surface_world_m": contact_surface.tolist(),
        }
        world.reset()
        robot.set_joint_positions(positions)
        robot.set_joint_velocities(np.zeros(len(names)))
        robot.apply_action(ArticulationAction(joint_positions=positions))
        for _ in range(120):
            world.step(render=False)
        backend.object_provider = simulation_object_provider(
            {"cube": cube}, frame_id="world", stage_units_in_meters=1.0
        )
        start = backend.get_state(timeout_s=5)
        REPORT["push_initial_state"] = start
        contact = copy.deepcopy(start["pose"])
        approach = copy.deepcopy(contact)
        approach["position_m"][2] += 0.035
        REPORT["request"] = {
            "approach_pose": approach,
            "contact_pose": contact,
            "target_id": "cube",
            "direction_world": [1, 0, 0],
            "distance_m": 0.06,
            "step_m": 0.01,
            "speed_m_s": 0.025,
            "timeout_s": 30.0,
        }
    else:
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid

        from roboskillhub.runtime.isaac import simulation_object_provider

        is_drawer = args.task == "pull_drawer"
        close_width = (
            0.075
            if is_drawer and args.drawer_control == "miss_handle"
            else args.grasp_close_width_m
        )
        # Search a bounded setup grid for a top-down pose reachable by this 6-DOF
        # model. This only plans; it never sends an actuator command. Require all
        # three skill poses to be feasible before constructing the test scene.
        # The source joint4 lower limit makes the old x=.2-.3/z=.2-.3 grid
        # unreachable in a vertical grasp. This lower, further grid was checked
        # independently with bounded URDF FK/IK; Lula must still verify it here.
        backend._sync_base()
        chosen = None
        attempts = 0
        REPORT["setup_ik_attempts"] = []
        for z in (0.14, 0.15):
            for x, y in ((0.40, 0.0), (0.39, 0.0)):
                for yaw in (180,) if is_drawer else (90, 270):
                    attempts += 1
                    q = Rotation.from_euler("xyz", [180, 0, yaw], degrees=True).as_quat()
                    wxyz = np.array([q[3], q[0], q[1], q[2]])
                    init_action, success = kin.compute_inverse_kinematics(
                        target_position=np.array([x, y, z]),
                        target_orientation=wxyz,
                        position_tolerance=0.0005,
                        orientation_tolerance=0.003,
                    )
                    candidate_report = {
                        "pregrasp_position_m": [x, y, z],
                        "yaw_deg": yaw,
                        "pregrasp_success": bool(success),
                        "phase_success": [],
                    }
                    REPORT["setup_ik_attempts"].append(candidate_report)
                    if not success:
                        continue
                    reachable = True
                    phase_positions = (
                        [[x, y, z - 0.09]] + [[x - 0.01 * i, y, z - 0.09] for i in range(1, 7)]
                        if is_drawer
                        else [[x, y, z - 0.09], [x, y, z - 0.04]]
                    )
                    for phase_position in phase_positions:
                        _, reachable = kin.compute_inverse_kinematics(
                            target_position=np.array(phase_position),
                            target_orientation=wxyz,
                            position_tolerance=0.0005,
                            orientation_tolerance=0.003,
                        )
                        candidate_report["phase_success"].append(
                            {"position_m": phase_position, "success": bool(reachable)}
                        )
                        if not reachable:
                            break
                    if reachable:
                        chosen = (
                            {
                                "frame_id": "world",
                                "position_m": [x, y, z],
                                "quaternion_xyzw": q.tolist(),
                            },
                            init_action,
                        )
                        break
                if chosen is not None:
                    break
            if chosen is not None:
                break
        REPORT["setup_ik_candidates_checked"] = attempts
        if chosen is None:
            raise RuntimeError(
                "No pregrasp/grasp/lift combination in the bounded setup grid is reachable"
            )
        pregrasp, init_action = chosen
        positions = np.array([home[n] for n in names], dtype=np.float64)
        positions[np.asarray(init_action.joint_indices, dtype=int)] = init_action.joint_positions
        positions[[names.index("left_joint7"), names.index("left_joint8")]] = 0.04
        material = PhysicsMaterial(
            "/World/TestMaterial", static_friction=0.6, dynamic_friction=0.5, restitution=0.0
        )
        center = np.array(pregrasp["position_m"]) - np.array([0, 0, 0.09])
        cube_size = 0.04
        if is_drawer:
            opening_axis = np.array([-1.0, 0.0, 0.0])
            drawer_origin = center + np.array([0.09, 0.0, -0.0425])
            world.scene.add(
                FixedCuboid(
                    "/World/DrawerHousing",
                    name="drawer_housing",
                    position=drawer_origin + np.array([0, 0, -0.0175]),
                    scale=np.array([0.32, 0.16, 0.02]),
                    size=1.0,
                    physics_material=material,
                )
            )

            def fixture_body(path, name, position, size, mass):
                # Keep each rigid-body root unscaled: joint frames are metric.
                UsdGeom.Xform.Define(stage, path)
                FixedCuboid(
                    path + "/Geometry",
                    name=name + "_geometry",
                    scale=np.asarray(size),
                    size=1.0,
                    physics_material=material,
                )
                return world.scene.add(
                    SingleRigidPrim(prim_path=path, name=name, position=position, mass=mass)
                )

            drawer = fixture_body(
                "/World/DrawerTray", "drawer_tray", drawer_origin, [0.14, 0.10, 0.015], 0.20
            )
            handle = fixture_body(
                "/World/DrawerHandle", "drawer_handle", center, [0.025, 0.04, 0.025], 0.05
            )
            # The handle is a structural part of the drawer; no joint connects
            # either object to the robot. Independently measured body poses must
            # agree with this fixed internal relationship during execution.
            fixed = UsdPhysics.FixedJoint.Define(stage, "/World/DrawerHandleJoint")
            fixed.CreateBody0Rel().SetTargets([drawer.prim_path])
            fixed.CreateBody1Rel().SetTargets([handle.prim_path])
            fixed.CreateLocalPos0Attr(Gf.Vec3f(*map(float, center - drawer_origin)))
            fixed.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            fixed.CreateLocalRot0Attr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
            fixed.CreateLocalRot1Attr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
            slide = UsdPhysics.PrismaticJoint.Define(stage, "/World/DrawerSlide")
            slide.CreateBody1Rel().SetTargets([drawer.prim_path])
            slide.CreateAxisAttr("X")
            slide.CreateLocalPos0Attr(Gf.Vec3f(*map(float, drawer_origin)))
            slide.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            axis_rotation = Gf.Quatf(0, Gf.Vec3f(0, 0, 1))
            slide.CreateLocalRot0Attr(axis_rotation)
            slide.CreateLocalRot1Attr(axis_rotation)
            slide.CreateLowerLimitAttr(0.0)
            slide.CreateUpperLimitAttr(0.0 if args.drawer_control == "locked" else 0.12)
            drive = UsdPhysics.DriveAPI.Apply(slide.GetPrim(), "linear")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(10.0)
            drive.CreateTargetPositionAttr(0.0)
            drive.CreateTargetVelocityAttr(0.0)
            drive.CreateMaxForceAttr(2.0)
            REPORT["drawer_scene"] = {
                "control": args.drawer_control,
                "tray_mass_kg": 0.20,
                "handle_mass_kg": 0.05,
                "tray_size_m": [0.14, 0.10, 0.015],
                "handle_size_m": [0.025, 0.04, 0.025],
                "gravity": "ordinary gravity on tray and handle; only robot links are compensated",
                "surface_material": {"static_friction": 0.6, "dynamic_friction": 0.5},
                "friction_scope": "surface material only; no claim of calibrated Coulomb rail friction",
                "opening_axis_world": opening_axis.tolist(),
                "joint_coordinate_method": "joint_coordinate_from_physics_body_frames",
                "joint_world_anchor_m": drawer_origin.tolist(),
                "joint_local_rot_wxyz": [0, 0, 0, 1],
                "joint_limits_m": [0.0, float(slide.GetUpperLimitAttr().Get())],
                "internal_handle_joint": "physical fixed joint between tray and handle; no robot-object attachment",
                "passive_drive": {
                    "type": "force",
                    "stiffness_N_m": 0.0,
                    "damping_N_s_m": 10.0,
                    "target_position_m": 0.0,
                    "target_velocity_m_s": 0.0,
                    "max_force_N": 2.0,
                    "meaning": "passive viscous resistance; it never commands opening",
                },
            }
            observed_object, target_id, target_path = handle, "handle", "/World/DrawerHandle"
        else:
            shelf = world.scene.add(
                FixedCuboid(
                    "/World/TestShelf",
                    name="shelf",
                    position=center - np.array([0, 0, 0.035]),
                    scale=np.array([0.25, 0.22, 0.03]),
                    size=1.0,
                    physics_material=material,
                )
            )
            cube = world.scene.add(
                DynamicCuboid(
                    "/World/GraspCube",
                    name="grasp_cube",
                    position=center,
                    scale=np.array([cube_size] * 3),
                    size=1.0,
                    mass=0.05,
                    physics_material=material,
                )
            )
            observed_object, target_id, target_path = cube, "cube", "/World/GraspCube"
        from isaacsim.core.prims import RigidPrim

        tip_paths = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.GetName() in ("left_tip_left", "left_tip_right")
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if len(tip_paths) != 2:
            raise RuntimeError(f"Expected two physical fingertip bodies, got {tip_paths}")
        contacts = RigidPrim(
            prim_paths_expr=tip_paths,
            name="target_filtered_finger_contacts",
            reset_xform_properties=False,
            track_contact_forces=True,
            contact_filter_prim_paths_expr=[[target_path], [target_path]],
            max_contact_count=512 if is_drawer else 128,
            disable_stablization=False,
        )
        world.reset()
        contacts.initialize()
        robot.set_joint_positions(positions)
        robot.set_joint_velocities(np.zeros(len(names)))
        robot.apply_action(ArticulationAction(joint_positions=positions))
        for _ in range(120):
            world.step(render=False)
        backend.object_provider = simulation_object_provider(
            {target_id: observed_object}, frame_id="world", stage_units_in_meters=1.0
        )
        REPORT["contact_measurement"] = {
            "finger_paths": tip_paths,
            "filter": target_path,
            "detail_buffer_capacity": 512 if is_drawer else 128,
            "force_units": "normal contact force in N; normal impulses divided by physics_dt; excludes friction",
            "physics_dt_s": float(world.get_physics_dt()),
            "completion_rule": "both target-filtered force norms >0.1N, both joints blocked >1mm before target, measured q(t) difference speed <0.005m/s, per-joint change <50um for 12 consecutive physics steps",
            "holding_semantics": "contact completion does not assert holding; independent object motion determines outcome",
            "velocity_measurement": "successive measured joint positions divided by physics_dt; raw solver qvel is logged separately because prior constrained-contact run reported >0.022m/s despite q(t) changing <0.000026m/s",
            "width_limitation": "2*joint_position is the source nominal aperture only, not measured collision-surface separation; geometry calibration remains required",
            "samples": [],
        }

        def contact_completion(*, width_m, current_positions, target_positions, timeout_s):
            del timeout_s
            # This module-level callback uses an explicit small state record below.
            if not math.isclose(width_m, close_width, abs_tol=1e-12):
                contact_state["stable_steps"] = 0
                contact_state["previous_positions"] = None
                return None
            force = np.asarray(
                contacts.get_contact_force_matrix(dt=world.get_physics_dt()), dtype=float
            )
            velocity = np.asarray(
                robot.get_joint_velocities(joint_indices=backend.gripper_indices), dtype=float
            )
            if force.shape != (2, 1, 3) or not np.isfinite(force).all():
                raise RuntimeError("Physical target-filtered bilateral contact forces unavailable")
            if velocity.shape != (2,) or not np.isfinite(velocity).all():
                raise RuntimeError("Physical gripper joint velocities unavailable")
            step_index = int(world.current_time_step_index)
            if contact_state.get("previous_step") != step_index - 1:
                contact_state["stable_steps"] = 0
                contact_state["previous_positions"] = None
            previous = contact_state["previous_positions"]
            measured_velocity = (
                None
                if previous is None
                else (current_positions - previous) / world.get_physics_dt()
            )
            stable = bool(
                np.all(np.linalg.norm(force[:, 0], axis=1) > 0.1)
                and np.all(current_positions - target_positions > backend.gripper_tolerance)
                and measured_velocity is not None
                and np.max(np.abs(measured_velocity)) < 0.005
                and np.max(np.abs(current_positions - previous)) < 0.00005
            )
            contact_state["stable_steps"] = contact_state["stable_steps"] + 1 if stable else 0
            contact_state["previous_positions"] = current_positions.copy()
            contact_state["previous_step"] = step_index
            sample = {
                "sim_time_s": float(world.current_time),
                "timestamp_s": time.time(),
                "normal_contact_vectors_N": force[:, 0].tolist(),
                "joint_positions_m": current_positions.tolist(),
                "joint_velocities_m_s": velocity.tolist(),
                "measured_position_difference_velocity_m_s": (
                    None if measured_velocity is None else measured_velocity.tolist()
                ),
                "nominal_aperture_m": float(sum(current_positions)),
                "target_positions_m": target_positions.tolist(),
                "stable_steps": contact_state["stable_steps"],
            }
            REPORT["contact_measurement"]["samples"].append(sample)
            if contact_state["stable_steps"] < 12:
                return None
            forces, points, normals, distances, counts, starts = contacts.get_contact_force_data(
                dt=world.get_physics_dt()
            )
            REPORT["contact_measurement"]["detail_count"] = int(np.sum(counts))
            REPORT["contact_measurement"]["detail_buffer_saturated"] = bool(
                np.sum(counts) >= REPORT["contact_measurement"]["detail_buffer_capacity"]
            )
            grip_position, grip_orientation = grip.get_world_pose()
            grip_rotation = Rotation.from_quat(
                [grip_orientation[1], grip_orientation[2], grip_orientation[3], grip_orientation[0]]
            )
            contact_geometry = []
            for finger_index, finger_path in enumerate(tip_paths):
                offset = int(starts[finger_index, 0])
                count = int(counts[finger_index, 0])
                interval = slice(offset, offset + count)
                point_array = np.asarray(points[interval], dtype=float)
                contact_geometry.append(
                    {
                        "finger_path": finger_path,
                        "count": count,
                        "normal_forces_N": np.asarray(forces[interval]).tolist(),
                        "points_world_m": point_array.tolist(),
                        "points_gripper_local_m": grip_rotation.inv()
                        .apply(point_array - np.asarray(grip_position))
                        .tolist(),
                        "normals_world": np.asarray(normals[interval]).tolist(),
                        "separations_m": np.asarray(distances[interval]).tolist(),
                    }
                )
            REPORT["contact_measurement"]["completion_geometry"] = contact_geometry
            REPORT["contact_measurement"]["completion_physical_tcp"] = physical_tcp()
            return {
                "complete": True,
                "requested_width_m": width_m,
                "source": "simulation_truth",
                "timestamp_s": sample["timestamp_s"],
                "reason": "bilateral_target_contact_with_stable_aperture",
                "contact_measurement": sample,
            }

        contact_state = {"stable_steps": 0, "previous_positions": None}
        backend.gripper_completion = contact_completion
        grasp = copy.deepcopy(pregrasp)
        grasp["position_m"][2] = float(center[2])
        if is_drawer:
            drawer_trace = []
            drawer_previous = {"time": None, "q": None}

            def drawer_measurement():
                tray_p, tray_q = drawer.get_world_pose()
                handle_p, handle_q = handle.get_world_pose()
                tray_delta = np.asarray(tray_p) - drawer_origin
                q_measured = float(tray_delta @ opening_axis)
                handle_delta = np.asarray(handle_p) - center
                force = np.asarray(
                    contacts.get_contact_force_matrix(dt=world.get_physics_dt()), dtype=float
                )
                if force.shape != (2, 1, 3) or not np.isfinite(force).all():
                    raise RuntimeError("Drawer handle contact measurement unavailable")
                now = float(world.current_time)
                dt = None if drawer_previous["time"] is None else now - drawer_previous["time"]
                q_speed = (
                    None if dt is None or dt <= 0 else (q_measured - drawer_previous["q"]) / dt
                )
                drawer_previous.update(time=now, q=q_measured)
                damping_force = None if q_speed is None else float(-np.clip(10 * q_speed, -2, 2))
                sample = {
                    "sim_time_s": now,
                    "timestamp_s": time.time(),
                    "source": "simulation_truth",
                    "tray_position_m": np.asarray(tray_p).tolist(),
                    "tray_quaternion_wxyz": np.asarray(tray_q).tolist(),
                    "handle_position_m": np.asarray(handle_p).tolist(),
                    "handle_quaternion_wxyz": np.asarray(handle_q).tolist(),
                    "joint_coordinate_m": q_measured,
                    "joint_velocity_from_position_m_s": q_speed,
                    "raw_tray_linear_velocity_m_s": drawer.get_linear_velocity().tolist(),
                    "off_axis_translation_m": float(
                        np.linalg.norm(tray_delta - q_measured * opening_axis)
                    ),
                    "tray_rotation_rad": float(
                        Rotation.from_quat(np.asarray(tray_q)[[1, 2, 3, 0]]).magnitude()
                    ),
                    "handle_rotation_rad": float(
                        Rotation.from_quat(np.asarray(handle_q)[[1, 2, 3, 0]]).magnitude()
                    ),
                    "handle_joint_coordinate_m": float(handle_delta @ opening_axis),
                    "handle_vs_tray_translation_error_m": float(
                        np.linalg.norm(handle_delta - tray_delta)
                    ),
                    "normal_handle_contact_vectors_N": force[:, 0].tolist(),
                    "passive_drive_model_force_N": damping_force,
                    "passive_drive_model_power_W": None
                    if q_speed is None
                    else damping_force * q_speed,
                    "drive_force_scope": "configured viscous model evaluated at measured velocity, not a force-sensor reading",
                }
                return sample

            passive_samples = []
            for _ in range(240):
                world.step(render=False)
                passive_samples.append(drawer_measurement())
            REPORT["drawer_passive_control"] = {
                "duration_sim_s": 2.0,
                "robot_condition": "stationary pregrasp 90mm above handle; no skill invoked",
                "max_abs_joint_coordinate_m": max(
                    abs(a["joint_coordinate_m"]) for a in passive_samples
                ),
                "samples": passive_samples,
            }
            if REPORT["drawer_passive_control"]["max_abs_joint_coordinate_m"] > 0.0005:
                raise RuntimeError("Drawer drifted beyond 0.5mm before robot contact")
            REPORT["drawer_initial_measurement"] = drawer_measurement()
            REPORT["drawer_initial_state"] = backend.get_state(timeout_s=5)
            REPORT["request"] = {
                "pregrasp_pose": pregrasp,
                "grasp_pose": grasp,
                "target_id": target_id,
                "direction_world": opening_axis.tolist(),
                "distance_m": 0.06,
                "step_m": 0.01,
                "open_width_m": 0.08,
                "close_width_m": close_width,
                "speed_m_s": 0.025,
                "timeout_s": 30.0,
                "release_on_success": True,
            }

            def record_drawer_physics_step(_dt):
                # Isaac calls this before a step; values are the preceding
                # completed physics state, never commanded target positions.
                drawer_trace.append(drawer_measurement())

            world.add_physics_callback("drawer_evidence", record_drawer_physics_step)
            REPORT["drawer_motion_attempts"] = []
            native_move_pose = backend.move_pose

            def record_drawer_motion_attempt(pose, *, speed_m_s, timeout_s, linear):
                is_pull = bool(
                    linear
                    and pose["position_m"][0] < grasp["position_m"][0] - 1e-6
                    and abs(pose["position_m"][2] - grasp["position_m"][2]) < 1e-8
                )
                attempt = {
                    "target_pose": copy.deepcopy(pose),
                    "is_pull_waypoint": is_pull,
                    "started_sim_time_s": float(world.current_time),
                    "before": drawer_measurement(),
                }
                REPORT["drawer_motion_attempts"].append(attempt)
                try:
                    reply = native_move_pose(
                        pose, speed_m_s=speed_m_s, timeout_s=timeout_s, linear=linear
                    )
                    attempt["reply"] = copy.deepcopy(reply)
                    return reply
                except Exception as error:
                    attempt["error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    attempt["finished_sim_time_s"] = float(world.current_time)
                    attempt["after"] = drawer_measurement()

            backend.move_pose = record_drawer_motion_attempt
            stage.GetRootLayer().Export(str(ROOT / "drawer_scene.usda"))
        else:
            lift = copy.deepcopy(grasp)
            lift["position_m"][2] += 0.05
            REPORT["grasp_scene"] = {
                "cube_initial_position_m": center.tolist(),
                "cube_size_m": cube_size,
                "mass_kg": 0.05,
                "static_friction": 0.6,
                "dynamic_friction": 0.5,
                "shelf_top_z_m": float(center[2] - 0.02),
                "initialization": "reset to top-down Lula IK pregrasp once, before skill execution",
                "initial_joint_rad_or_m": positions.tolist(),
                "gripper_calibration": "source nominal tip gap=2*joint7/8; original 1mm joint arrival tolerance or measured stable bilateral cube contact completes closing; neither alone establishes holding",
            }
            REPORT["grasp_initial_state"] = backend.get_state(timeout_s=5)
            REPORT["request"] = {
                "pregrasp_pose": pregrasp,
                "grasp_pose": grasp,
                "lift_pose": lift,
                "target_id": "cube",
                "open_width_m": 0.08,
                "close_width_m": close_width,
                "speed_m_s": 0.025,
                "timeout_s": 30.0,
            }
    before_joints = robot.get_joint_positions().copy()
    sim_t0 = world.current_time
    result = Runtime(backend, limits).invoke(args.task, REPORT["request"])
    physical_after = physical_tcp()
    REPORT["runtime_result"] = result
    REPORT["physical_final_pose"] = physical_after
    REPORT["joint_motion_l2"] = float(np.linalg.norm(robot.get_joint_positions() - before_joints))
    REPORT["simulated_motion_seconds"] = float(world.current_time - sim_t0)
    if args.task == "move_pose":
        err = pose_error(physical_after, target)
        REPORT["physical_final_error"] = {"position_m": err[0], "rotation_rad": err[1]}
        REPORT["passed"] = bool(
            result["status"] == "succeeded"
            and err[0] <= 0.003
            and err[1] <= 0.03
            and REPORT["joint_motion_l2"] > 1e-4
            and REPORT["simulated_motion_seconds"] > 0
        )
    elif args.task == "pull_drawer":
        world.remove_physics_callback("drawer_evidence")
        final_drawer = drawer_measurement()
        REPORT["drawer_execution_trace"] = drawer_trace
        REPORT["drawer_final_measurement"] = final_drawer
        physical_path_executed = bool(
            REPORT["joint_motion_l2"] > 1e-4 and REPORT["simulated_motion_seconds"] > 0
        )
        max_bilateral_contact = max(
            (
                min(float(np.linalg.norm(v)) for v in sample["normal_handle_contact_vectors_N"])
                for sample in drawer_trace
            ),
            default=0.0,
        )
        constraint_samples = drawer_trace + [final_drawer]
        extrema = {
            "max_off_axis_translation_m": max(
                a["off_axis_translation_m"] for a in constraint_samples
            ),
            "max_rotation_rad": max(
                max(a["tray_rotation_rad"], a["handle_rotation_rad"]) for a in constraint_samples
            ),
            "max_handle_vs_tray_translation_error_m": max(
                a["handle_vs_tray_translation_error_m"] for a in constraint_samples
            ),
            "min_joint_coordinate_m": min(a["joint_coordinate_m"] for a in constraint_samples),
            "max_joint_coordinate_m": max(a["joint_coordinate_m"] for a in constraint_samples),
        }
        REPORT["drawer_execution_extrema"] = extrema
        constraints = {
            "off_axis_within_1mm": extrema["max_off_axis_translation_m"] < 0.001,
            "rotation_within_0_01rad": extrema["max_rotation_rad"] < 0.01,
            "independent_handle_vs_joint_within_1mm": extrema[
                "max_handle_vs_tray_translation_error_m"
            ]
            < 0.001,
            "joint_limits_within_1mm": extrema["min_joint_coordinate_m"] >= -0.001
            and extrema["max_joint_coordinate_m"] <= float(slide.GetUpperLimitAttr().Get()) + 0.001,
            "passive_no_self_opening": REPORT["drawer_passive_control"][
                "max_abs_joint_coordinate_m"
            ]
            <= 0.0005,
        }
        REPORT["drawer_constraint_checks"] = constraints
        REPORT["max_simultaneous_bilateral_handle_normal_force_N"] = max_bilateral_contact
        if args.drawer_control == "positive":
            if result["status"] == "succeeded":
                for _ in range(120):
                    world.step(render=False)
                REPORT["drawer_after_release_1s"] = drawer_measurement()
            REPORT["passed"] = bool(
                result["status"] == "succeeded"
                and physical_path_executed
                and all(constraints.values())
                and 0.05 <= final_drawer["joint_coordinate_m"] <= 0.065
                and 0.05 <= REPORT["drawer_after_release_1s"]["joint_coordinate_m"] <= 0.065
                and not REPORT["contact_measurement"].get("detail_buffer_saturated", True)
            )
        else:
            object_failure = "observed object motion" in result.get("error", {}).get("message", "")
            no_release_after_failure = (
                len(
                    [
                        e
                        for e in result.get("trace", [])
                        if e.get("event") == "set_gripper"
                        and math.isclose(e.get("width_m", -1), 0.08)
                    ]
                )
                == 1
            )
            pull_attempts = [a for a in REPORT["drawer_motion_attempts"] if a["is_pull_waypoint"]]
            endpoint = copy.deepcopy(grasp)
            endpoint["position_m"][0] -= 0.06
            endpoint_error = pose_error(physical_after, endpoint)
            full_pull_completed = bool(
                len(pull_attempts) == 6
                and all(a.get("reply", {}).get("ok") is True for a in pull_attempts)
                and endpoint_error[0] <= 0.003
                and endpoint_error[1] <= 0.03
            )
            close_acknowledged = any(
                e.get("event") == "set_gripper"
                and math.isclose(e.get("width_m", -1), close_width)
                and e.get("reply", {}).get("ok") is True
                for e in result.get("trace", [])
            )
            locked_pull_attempted = bool(
                args.drawer_control == "locked"
                and pull_attempts
                and close_acknowledged
                and REPORT["contact_measurement"].get("completion_geometry")
                and max_bilateral_contact > 0.1
                and pull_attempts[0]["finished_sim_time_s"] > pull_attempts[0]["started_sim_time_s"]
            )
            REPORT["control_passed"] = bool(
                result["status"] == "failed"
                and physical_path_executed
                and abs(final_drawer["joint_coordinate_m"]) < 0.002
                and all(constraints.values())
                and no_release_after_failure
                and (
                    locked_pull_attempted
                    or (
                        args.drawer_control == "miss_handle"
                        and object_failure
                        and full_pull_completed
                        and max_bilateral_contact < 0.1
                    )
                )
            )
            REPORT["negative_control_checks"] = {
                "physical_path_executed": physical_path_executed,
                "full_pull_completed": full_pull_completed,
                "pull_attempt_count": len(pull_attempts),
                "endpoint_position_error_m": endpoint_error[0],
                "endpoint_rotation_error_rad": endpoint_error[1],
                "locked_actual_contact_pull_attempted": locked_pull_attempted,
                "runtime_failed_object_motion": object_failure,
                "no_release_after_failure": no_release_after_failure,
                "miss_handle_has_no_bilateral_contact": max_bilateral_contact < 0.1,
            }
            REPORT["passed"] = REPORT["control_passed"]
        REPORT["test_passed"] = REPORT["passed"]
    else:
        REPORT["physical_cube_final_position_m"] = cube.get_world_pose()[0].tolist()
        if args.task == "grasp" and result["status"] == "succeeded":
            for _ in range(120):
                world.step(render=False)
            REPORT["physical_cube_after_hold_1s_m"] = cube.get_world_pose()[0].tolist()
        REPORT["passed"] = bool(
            result["status"] == "succeeded"
            and REPORT["joint_motion_l2"] > 1e-4
            and REPORT["simulated_motion_seconds"] > 0
        )
        if args.task == "grasp":
            REPORT["passed"] = (
                REPORT["passed"]
                and REPORT["physical_cube_after_hold_1s_m"][2] > float(center[2]) + 0.03
            )
    if args.task == "grasp" and REPORT["passed"]:
        # Exercise the adapter's controlled drive hold only after successful grasp
        # and independent retention. Failed skill runs are not retention trials.
        before_stop_targets = np.asarray(robot.get_applied_action().joint_positions).copy()
        before_stop_cube = cube.get_world_pose()[0].copy()
        backend.stop()
        after_stop_targets = np.asarray(robot.get_applied_action().joint_positions).copy()
        preserved = bool(
            np.allclose(
                before_stop_targets[backend.gripper_indices],
                after_stop_targets[backend.gripper_indices],
                rtol=0,
                atol=1e-9,
            )
        )
        for _ in range(120):
            world.step(render=False)
        after_stop_cube = cube.get_world_pose()[0].copy()
        REPORT["stop_retention"] = {
            "kind": "controlled drive hold, not an emergency-stop certificate",
            "source": "simulation_truth",
            "simulated_hold_seconds": 1.0,
            "cube_before_stop_m": before_stop_cube.tolist(),
            "cube_after_stop_hold_m": after_stop_cube.tolist(),
            "gripper_targets_before_stop_m": before_stop_targets[backend.gripper_indices].tolist(),
            "gripper_targets_after_stop_m": after_stop_targets[backend.gripper_indices].tolist(),
            "gripper_measured_positions_after_hold_m": robot.get_joint_positions(
                joint_indices=backend.gripper_indices
            ).tolist(),
            "gripper_target_preserved": preserved,
            "retained_above_30mm": bool(after_stop_cube[2] > float(center[2]) + 0.03),
        }
        REPORT["passed"] = bool(
            REPORT["passed"] and preserved and REPORT["stop_retention"]["retained_above_30mm"]
        )
    print(
        "SMOKE:result",
        json.dumps(
            {
                k: REPORT.get(k)
                for k in [
                    "passed",
                    "physical_final_error",
                    "joint_motion_l2",
                    "simulated_motion_seconds",
                ]
            }
        ),
        flush=True,
    )
except BaseException as e:
    REPORT["passed"] = False
    REPORT["error"] = f"{type(e).__name__}: {e}"
    REPORT["traceback"] = traceback.format_exc()
    print("SMOKE:failed", REPORT["traceback"], flush=True)
finally:
    if (
        REPORT.get("passed")
        and REPORT["evaluation_role"] == "positive"
        and args.task in REPORT["not_validated"]
    ):
        REPORT["not_validated"].remove(args.task)
    REPORT["finished_at_unix_s"] = time.time()
    (ROOT / "report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    if app is not None:
        app.close()

sys.exit(0 if REPORT.get("passed") else 1)
