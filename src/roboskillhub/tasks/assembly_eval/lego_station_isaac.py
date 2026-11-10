# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim 5.1 + BrickSim physics for the LEGO StationBackend (``lego_station.LegoPhysicsStation``).

Runs inside an Isaac Sim Python. One ``SimulationApp`` per process with 4b's physics-only experience
(``domino_isaac.write_physics_kit``: no RTX renderer, no viewport), stepped synchronously with
``World.step(render=False)``; BrickSim's extension is enabled by importing ``bricksim``. A fresh stage per
episode holds:

- a static table (top at z = 0);
- the baseplate (and any pre-built bricks) imported with ``bricksim.core.import_lego``: a free body resting on
  the table, as in RoCo;
- loose bricks allocated with ``bricksim.core.allocate_brick_part`` on the table (the feeder);
- a **floating parallel-jaw hand** built the way 4b's domino hand is (fixed base -> x/y/z prismatic -> yaw
  revolute -> palm -> two prismatic pads, gravity off on the hand links), sized for LEGO bricks and with the
  YAM linear_4310 TCP convention Agent 1's grasp model uses (TCP z out of the fingers, y = closing axis, pad
  tips 10.2 mm past the TCP). It is NOT a YAM arm and not RoCo's DexMate Vega U.

Nothing writes a brick pose, creates a connection or makes a brick kinematic after reset: connections are
BrickSim's contact-triggered welds. Truth (brick poses, engine connections, assembly events, pad contacts)
is read from PhysX/BrickSim and labelled ``simulation_truth`` by the station.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from .domino_geometry import wxyz_to_xyzw

ACTIVE = None
"""The live ``LegoIsaacPhysics`` of this process (one per process), for camera consumers such as the
lego_hub bridge; None until one is built."""

CAMERA_K = ((394.0, 0.0, 320.0), (0.0, 394.0, 240.0), (0.0, 0.0, 1.0))
CAMERA_SIZE = (640, 480)
"""D405-like 640 x 480 pinhole (fx = fy = 394 px; lego_hub.depth.D405_K), used for both cameras."""
OVERHEAD_HEIGHT_M = 0.45
"""Overhead camera height over the plate centre, looking straight down (a D405 / Orbbec on a mast)."""

GRAVITY = 9.81
PHYSICS_DT = 1.0 / 120.0
TOUCH_SEPARATION = 0.0005


class HandSpec:
    """Floating hand for 2xN LEGO bricks (metres, kilograms, newtons)."""

    pad_size = (
        0.008,
        0.004,
        0.014,
    )  # x across the jaw, y along the closing axis (thickness), z along approach
    tip_reach = 0.0102  # pad tips past the TCP along TCP +z (Agent 1's YAM TIP_REACH)
    palm_size = (0.03, 0.06, 0.01)
    max_opening = 0.05
    grip_force = 15.0  # per pad drive force limit
    pad_friction = 1.0
    xyz_stiffness, xyz_damping, xyz_max_force = 4000.0, 120.0, 60.0
    yaw_stiffness, yaw_damping, yaw_max_torque = 20.0, 0.6, 5.0
    pad_stiffness, pad_damping = 1500.0, 15.0
    finger_speed = 0.05


class LegoIsaacPhysics:
    name = "isaac-bricksim"

    def __init__(self, headless: bool = True, render: bool | None = None):
        """``render`` (default: env ``LEGO_RENDER=1``) starts Isaac's RTX experience and adds an overhead
        and a wrist RGB-D camera (``camera_frame``); off, the physics-only kit runs as before."""
        import os

        import isaacsim
        from isaacsim import SimulationApp

        from .domino_isaac import PHYSICS_KIT_NAME, write_physics_kit

        global ACTIVE
        self.render = os.environ.get("LEGO_RENDER") == "1" if render is None else bool(render)
        pkg = Path(isaacsim.__file__).parent
        default = (
            pkg / "apps" / "isaacsim.exp.base.python.kit" if self.render else write_physics_kit(pkg)
        )
        exp = os.environ.get("ISAAC_EXPERIENCE") or str(default)
        config = {"headless": headless, "extra_args": ["--/app/settings/persistent=false"]}
        if exp.endswith(PHYSICS_KIT_NAME):
            config.update(create_new_stage=False, disable_viewport_updates=True)
        if self.render:
            config.update(width=CAMERA_SIZE[0], height=CAMERA_SIZE[1])
        self._app = SimulationApp(config, experience=exp)
        self.experience = exp
        import bricksim  # noqa: F401  (enables the BrickSim extension inside Kit)
        import bricksim.core as bc

        self.bc = bc
        self.hand_spec = HandSpec()
        self._world = None
        self.usd_sha256 = None
        self._sub = 1
        self._dt = PHYSICS_DT
        self.cameras: dict = {}
        self.frame_dir = os.environ.get("LEGO_FRAME_DIR") or None
        self.frames_saved = 0
        ACTIVE = self

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        import importlib.metadata as md

        return {
            "name": self.name,
            "isaacsim": md.version("isaacsim"),
            "bricksim": md.version("bricksim"),
            "physics": f"PhysX via isaacsim.core.api.World (CPU), dt {self._dt:.5f} s, BrickSim clutch",
            "experience": self.experience,
            "renderer": "RTX (overhead + wrist RGB-D, D405 intrinsics)"
            if self.render
            else "none (physics-only kit)",
            "hand": "fixed-base x/y/z prismatic + yaw revolute + 2 prismatic pads (no arm)",
            "usd_sha256": self.usd_sha256,
            "assembly_thresholds": {
                k: getattr(self.bc.get_assembly_thresholds(), k)
                for k in (
                    "distance_tolerance",
                    "max_penetration",
                    "z_angle_tolerance",
                    "required_force",
                    "yaw_tolerance",
                    "position_tolerance",
                )
            },
            "breakage_enabled": bool(self.bc.get_breakage_thresholds().enabled),
        }

    @property
    def control_dt(self) -> float:
        return self._sub * self._dt

    # ------------------------------------------------------------------
    def reset(self, scene: dict, *, breakage: bool = False) -> dict:
        """Build the stage. ``scene``: ``plate_pose`` (pos, rot wxyz of the plate bottom centre),
        ``prebuilt_topology`` (BrickSim topology JSON incl. the plate), ``feeder`` (list of
        ``{id, dims, pos, rot}``), ``hand_start`` (x, y, z, yaw), ``thresholds`` (assembly thresholds dict).
        Returns ``{id: prim path}`` for the plate, pre-built bricks and feeder bricks."""
        import numpy as np
        import omni.physx
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import create_new_stage
        from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

        bc = self.bc
        if self._world is not None:
            self._world.stop()
            World.clear_instance()
        if self.render:
            # create_new_stage() segfaults under the RTX experience (omni.graph.image, Isaac Sim 5.1):
            # reuse the app's stage with everything but Kit's own prims removed.
            import omni.usd

            st = omni.usd.get_context().get_stage()
            for child in list(st.GetPseudoRoot().GetChildren()):
                if not child.GetName().startswith(("OmniverseKit", "Render")):
                    st.RemovePrim(child.GetPath())
        else:
            create_new_stage()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=self._dt,
            rendering_dt=self._dt,
            backend="numpy",
            device="cpu",
        )
        self._world = world
        stage = world.stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        sc = UsdPhysics.Scene.Get(stage, "/physicsScene") or UsdPhysics.Scene.Define(
            stage, "/physicsScene"
        )
        sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        sc.CreateGravityMagnitudeAttr(GRAVITY)
        PhysxSchema.PhysxSceneAPI.Apply(sc.GetPrim()).CreateEnableCCDAttr(True)

        def material(path, friction):
            mat = UsdShade.Material.Define(stage, path)
            api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            api.CreateStaticFrictionAttr(friction)
            api.CreateDynamicFrictionAttr(friction)
            api.CreateRestitutionAttr(0.0)
            return mat

        mats = {
            "table": material("/World/Materials/table", 0.8),
            "pad": material("/World/Materials/pad", 1.0),
        }

        def bind(prim, mat):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                mats[mat], UsdShade.Tokens.weakerThanDescendants, "physics"
            )

        def cube(path, size, pos):
            c = UsdGeom.Cube.Define(stage, path)
            c.CreateSizeAttr(1.0)
            x = UsdGeom.Xformable(c)
            x.ClearXformOpOrder()
            x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
            x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
            x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*size))
            return c.GetPrim()

        table = cube("/World/table", (0.8, 0.8, 0.02), (0.3, 0.0, -0.01))
        UsdPhysics.CollisionAPI.Apply(table)
        bind(table, "table")

        # ---- hand (same structure as domino_isaac.IsaacBackend's) ----
        h = self.hand_spec
        h.grip_force = float(scene.get("grip_force_N", HandSpec.grip_force))
        z0 = scene["hand_start"][2]
        self._z0 = z0
        self._h0 = 0.6 * h.max_opening / 2
        H = "/World/hand"
        UsdGeom.Xform.Define(stage, H)

        def link(path, pos, mass, inertia):
            xf = UsdGeom.Xform.Define(stage, path)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
            p_ = xf.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(p_)
            m = UsdPhysics.MassAPI.Apply(p_)
            m.CreateMassAttr(mass)
            m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
            m.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
            prb = PhysxSchema.PhysxRigidBodyAPI.Apply(p_)
            prb.CreateDisableGravityAttr(True)
            prb.CreateSolverPositionIterationCountAttr(32)
            return p_

        base = link(f"{H}/base", (0, 0, z0), 1.0, (1e-3,) * 3)
        UsdPhysics.ArticulationRootAPI.Apply(base)
        pxa = PhysxSchema.PhysxArticulationAPI.Apply(base)
        pxa.CreateEnabledSelfCollisionsAttr(False)
        pxa.CreateSolverPositionIterationCountAttr(32)
        fj = UsdPhysics.FixedJoint.Define(stage, f"{H}/world_fixed")
        fj.CreateBody1Rel().SetTargets([base.GetPath()])
        fj.CreateLocalPos0Attr(Gf.Vec3f(0, 0, z0))
        lx = link(f"{H}/mount_x", (0, 0, z0), 0.2, (1e-4,) * 3)
        ly = link(f"{H}/mount_y", (0, 0, z0), 0.2, (1e-4,) * 3)
        lz = link(f"{H}/mount_z", (0, 0, z0), 0.2, (1e-4,) * 3)
        palm = link(f"{H}/gripper_palm", (0, 0, z0), 0.2, (1.2e-4, 5e-5, 1.5e-4))
        # TCP is the palm origin; TCP +z points down (world -z). Pads span [tip - pad_z, tip] below the TCP.
        pad_cz = -(h.tip_reach - h.pad_size[2] / 2)
        palm_geom = cube(
            f"{H}/gripper_palm/geom",
            h.palm_size,
            (0, 0, pad_cz + h.pad_size[2] / 2 + h.palm_size[2] / 2),
        )
        UsdPhysics.CollisionAPI.Apply(palm_geom)
        bind(palm_geom, "pad")
        pads = {}
        for name, sign in (("gripper_pad_l", 1), ("gripper_pad_r", -1)):
            pl = link(f"{H}/{name}", (0, sign * self._h0, z0), 0.02, (3e-6,) * 3)
            geom = cube(f"{H}/{name}/geom", h.pad_size, (0, sign * h.pad_size[1] / 2, pad_cz))
            UsdPhysics.CollisionAPI.Apply(geom)
            pc = PhysxSchema.PhysxCollisionAPI.Apply(geom)
            pc.CreateContactOffsetAttr(0.001)
            pc.CreateRestOffsetAttr(0.0)
            bind(geom, "pad")
            cr = PhysxSchema.PhysxContactReportAPI.Apply(pl)
            cr.CreateThresholdAttr(0.0)
            pads[name] = pl

        def drive(joint_prim, kind, stiffness, damping, max_force):
            d = UsdPhysics.DriveAPI.Apply(joint_prim, kind)
            d.CreateTypeAttr("force")
            d.CreateStiffnessAttr(stiffness)
            d.CreateDampingAttr(damping)
            d.CreateMaxForceAttr(max_force)
            d.CreateTargetPositionAttr(0.0)

        def prismatic(name, b0, b1, axis, lo, hi, offset=(0, 0, 0)):
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{H}/{name}")
            j.CreateBody0Rel().SetTargets([b0.GetPath()])
            j.CreateBody1Rel().SetTargets([b1.GetPath()])
            j.CreateAxisAttr(axis)
            j.CreateLowerLimitAttr(lo)
            j.CreateUpperLimitAttr(hi)
            j.CreateLocalPos0Attr(Gf.Vec3f(*offset))
            j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            return j.GetPrim()

        jx = prismatic("hx", base, lx, "X", -1.0, 1.0)
        jy = prismatic("hy", lx, ly, "Y", -1.0, 1.0)
        jz = prismatic("hz", ly, lz, "Z", -1.0, 1.0)
        for j in (jx, jy, jz):
            drive(j, "linear", h.xyz_stiffness, h.xyz_damping, h.xyz_max_force)
        jr = UsdPhysics.RevoluteJoint.Define(stage, f"{H}/hyaw")
        jr.CreateBody0Rel().SetTargets([lz.GetPath()])
        jr.CreateBody1Rel().SetTargets([palm.GetPath()])
        jr.CreateAxisAttr("Z")
        drive(
            jr.GetPrim(),
            "angular",
            h.yaw_stiffness * math.pi / 180,
            h.yaw_damping * math.pi / 180,
            h.yaw_max_torque,
        )
        lim = h.max_opening / 2
        jl = prismatic(
            "fl", palm, pads["gripper_pad_l"], "Y", -self._h0, lim - self._h0, (0, self._h0, 0)
        )
        jrr = prismatic(
            "fr", palm, pads["gripper_pad_r"], "Y", -(lim - self._h0), self._h0, (0, -self._h0, 0)
        )
        for j in (jl, jrr):
            drive(j, "linear", h.pad_stiffness, h.pad_damping, h.grip_force)

        if self.render:
            self._make_cameras(stage, scene)
        world.reset()
        # ---- BrickSim parts (after the physics scene exists) ----
        th = scene.get("thresholds")
        if th:
            t = bc.get_assembly_thresholds()
            for k, v in th.items():
                setattr(t, k, v)
            bc.set_assembly_thresholds(t)
        bt = bc.get_breakage_thresholds()
        bt.enabled = bool(breakage)
        bc.set_breakage_thresholds(bt)
        ppos, prot = scene["plate_pose"]
        parts, _conns = bc.import_lego(
            scene["prebuilt_topology"], -1, ref_rot=tuple(prot), ref_pos=tuple(ppos)
        )
        paths = {scene["prebuilt_ids"][k]: v for k, v in parts.items()}
        for f in scene["feeder"]:
            paths[f["id"]] = bc.allocate_brick_part(
                tuple(f["dims"]),
                tuple(f.get("color", (200, 30, 30))),
                -1,
                tuple(f["rot"]),
                tuple(f["pos"]),
            )
        self.paths = paths
        self.path_to_id = {v: k for k, v in paths.items()}
        if scene.get("fix_baseplate", True):
            # The task's fixed baseplate: a fixed joint from the world to the plate part at its imported pose.
            # (RoCo leaves the plate free on the table; pressing a held brick then drags it.)
            plate = stage.GetPrimAtPath(paths[scene["plate_id"]])
            m = UsdGeom.Xformable(plate).ComputeLocalToWorldTransform(0)
            pj = UsdPhysics.FixedJoint.Define(stage, "/World/baseplate_fixed")
            pj.CreateBody1Rel().SetTargets([plate.GetPath()])
            tr = m.ExtractTranslation()
            pj.CreateLocalPos0Attr(Gf.Vec3f(float(tr[0]), float(tr[1]), float(tr[2])))
            rq = m.ExtractRotationQuat()
            pj.CreateLocalRot0Attr(Gf.Quatf(float(rq.GetReal()), Gf.Vec3f(rq.GetImaginary())))
            pj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            pj.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        out = Path("/tmp") / f"lego_station_{id(self)}.usda"
        stage.GetRootLayer().Export(str(out))
        self.usd_sha256 = hashlib.sha256(out.read_bytes()).hexdigest()
        world.reset()
        self._art = SingleArticulation(f"{H}/base", name="hand")
        self._art.initialize()
        names = list(self._art.dof_names)
        self._idx = [names.index(n) for n in ("hx", "hy", "hz", "hyaw", "fl", "fr")]
        self._np = np
        self._steps = 0
        self._t = 0.0
        self._pairs: dict[tuple[str, str], float] = {}
        self._physx = omni.physx.get_physx_simulation_interface()
        self._int_to_path = PhysicsSchemaTools.intToSdfPath
        from omni.physx.bindings._physx import ContactEventType

        self._lost = ContactEventType.CONTACT_LOST
        sx, sy, sz, syaw = scene["hand_start"]
        self._cmd = (sx, sy, sz, syaw)
        self._open = h.max_opening * 0.8
        for _ in range(int(0.5 / self._dt)):
            self.step(self._cmd, self._open)
        self.bc.get_assembled_connections(True)
        self.bc.get_disassembled_connections(True)
        self.bc.get_assembly_debug_infos()
        return dict(paths)

    # ------------------------------------------------------------------
    def _make_cameras(self, stage, scene: dict) -> None:
        """Overhead (fixed, straight down over the plate) and wrist (posed per frame) USD cameras with the
        D405 pinhole, wrapped as Isaac camera sensors before the world resets."""
        import numpy as np
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, UsdGeom, UsdLux

        np_ = np
        UsdLux.DomeLight.Define(stage, "/World/rsh_lights/dome").CreateIntensityAttr(600.0)
        sun = UsdLux.DistantLight.Define(stage, "/World/rsh_lights/sun")
        sun.CreateIntensityAttr(2500.0)
        sun.CreateAngleAttr(1.0)
        UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(35.0, 20.0, 0.0))
        fx = CAMERA_K[0][0]
        focal = 18.8  # only the focal / aperture ratio matters
        self.cameras = {}
        px, py, pz = (float(v) for v in scene["plate_pose"][0])
        over = np_.eye(4)
        over[:3, :3] = np_.diag([1.0, -1.0, -1.0])  # OpenCV: x right, y down, z (optical axis) down
        over[:3, 3] = (px, py, pz + OVERHEAD_HEIGHT_M)
        for name, pose in (("overhead", over), ("wrist", over)):
            path = f"/World/rsh_cameras/{name}"
            cam = UsdGeom.Camera.Define(stage, path)
            cam.GetFocalLengthAttr().Set(focal)
            cam.GetHorizontalApertureAttr().Set(CAMERA_SIZE[0] * focal / fx)
            cam.GetVerticalApertureAttr().Set(CAMERA_SIZE[1] * focal / CAMERA_K[1][1])
            cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10.0))
            self._set_cv_pose(stage, path, pose)
            sensor = Camera(prim_path=path, name=f"rsh_{name}", resolution=CAMERA_SIZE)
            sensor.initialize()
            sensor.add_distance_to_image_plane_to_frame()
            self.cameras[name] = {"path": path, "sensor": sensor, "pose": pose}

    def _set_cv_pose(self, stage, path: str, pose_cv) -> None:
        """Write an OpenCV camera-to-world pose (z = optical axis) as the USD camera's transform
        (USD cameras look along -z with +y up)."""
        import numpy as np
        from pxr import Gf, UsdGeom

        m = np.array(pose_cv, dtype=float)
        m[:3, :3] = m[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(Gf.Matrix4d(m.T.tolist()))

    def camera_frame(self, name: str, pose_cv=None, settle: int = 8) -> dict:
        """Render one RGB-D frame of camera ``name`` (``overhead`` or ``wrist``), the wrist camera first
        moved to ``pose_cv`` (4x4 OpenCV camera-to-world). Returns ``rgb`` (uint8 HxWx3), ``depth``
        (float32 m along the optical axis, 0 = no return), ``intrinsics`` and ``pose`` (4x4)."""
        import numpy as np

        if not self.render or name not in self.cameras:
            raise RuntimeError("cameras need LegoIsaacPhysics(render=True) (env LEGO_RENDER=1)")
        cam = self.cameras[name]
        if pose_cv is not None:
            cam["pose"] = np.array(pose_cv, dtype=float)
            self._set_cv_pose(self._world.stage, cam["path"], cam["pose"])
        for _ in range(max(1, settle)):
            self._world.render()
        rgba = cam["sensor"].get_rgba()
        depth = cam["sensor"].get_depth()
        rgb = (
            np.zeros((CAMERA_SIZE[1], CAMERA_SIZE[0], 3), np.uint8)
            if rgba is None or not np.size(rgba)
            else np.asarray(rgba)[..., :3].astype(np.uint8)
        )
        d = (
            np.zeros((CAMERA_SIZE[1], CAMERA_SIZE[0]), np.float32)
            if depth is None or not np.size(depth)
            else np.asarray(depth, dtype=np.float32)
        )
        d = np.where(np.isfinite(d) & (d > 0) & (d < 9.0), d, 0.0).astype(np.float32)
        out = {
            "name": name,
            "rgb": rgb,
            "depth": d,
            "intrinsics": np.array(CAMERA_K),
            "pose": cam["pose"].copy(),
            "t": self._t,
        }
        if self.frame_dir:
            Path(self.frame_dir).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(Path(self.frame_dir) / f"{self.frames_saved:05d}_{name}.npz", **out)
            self.frames_saved += 1
        return out

    # ------------------------------------------------------------------
    def _apply(self, pose, opening):
        from isaacsim.core.utils.types import ArticulationAction

        np = self._np
        x, y, z, yaw = pose
        half = max(0.0, opening / 2)
        q = np.zeros(len(self._art.dof_names))
        vals = [x, y, z - self._z0, yaw, half - self._h0, -(half - self._h0)]
        for i, v in zip(self._idx, vals, strict=True):
            q[i] = v
        self._art.apply_action(ArticulationAction(joint_positions=q))

    def _body(self, path: str) -> str:
        if path in self.path_to_id:
            return self.path_to_id[path]
        for p, i in self.path_to_id.items():
            if path.startswith(p + "/"):
                return i
        name = path.rsplit("/", 1)[-1]
        if name == "geom":
            name = path.rsplit("/", 2)[-2]
        return name

    def _read_contacts(self):
        headers, data = self._physx.get_contact_report()
        for hd in headers:
            a = self._body(str(self._int_to_path(hd.actor0)))
            b = self._body(str(self._int_to_path(hd.actor1)))
            key = tuple(sorted((a, b)))
            if hd.type == self._lost:
                self._pairs.pop(key, None)
                continue
            seps = [data[hd.contact_data_offset + i].separation for i in range(hd.num_contact_data)]
            self._pairs[key] = min(seps) if seps else 0.0

    def step(self, hand_pose, opening: float) -> float:
        self._cmd, self._open = tuple(hand_pose), opening
        self._apply(hand_pose, opening)
        for _ in range(self._sub):
            self._world.step(render=False)
            self._read_contacts()
            self._steps += 1
            self._t += self._dt
        return self._t

    def hand(self):
        q = self._art.get_joint_positions()
        x, y, z, yaw, fl, fr = (float(q[i]) for i in self._idx)
        return (x, y, z + self._z0, yaw), (fl + self._h0) + (-fr + self._h0)

    def hand_velocity(self) -> tuple[float, float, float]:
        v = self._art.get_joint_velocities()
        return tuple(float(v[i]) for i in self._idx[:3])

    def z_effort(self) -> float:
        """Measured joint effort of the z prismatic joint (N): the force the z joint transmits."""
        e = self._art.get_measured_joint_efforts()
        return float(e[self._idx[2]])

    def pad_contacts(self) -> dict[str, set]:
        out = {"gripper_pad_l": set(), "gripper_pad_r": set()}
        for (a, b), sep in self._pairs.items():
            if sep > TOUCH_SEPARATION:
                continue
            for pad in out:
                if a == pad:
                    out[pad].add(b)
                elif b == pad:
                    out[pad].add(a)
        return out

    def truth(self) -> dict:
        from pxr import UsdGeom

        stage = self._world.stage
        objs = {}
        for oid, path in self.paths.items():
            m = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)
            t = m.ExtractTranslation()
            q = m.ExtractRotationQuat()
            im = q.GetImaginary()
            objs[oid] = {
                "position": [float(t[0]), float(t[1]), float(t[2])],
                "quat_xyzw": [float(im[0]), float(im[1]), float(im[2]), float(q.GetReal())],
            }
        return {"t": self._t, "objects": objs}

    def connections(self) -> list[dict]:
        out = []
        for s, sp in self.paths.items():
            for hh, hp in self.paths.items():
                if s == hh:
                    continue
                ci = self.bc.lookup_physics_connection(sp, 1, hp, 0)
                if ci is not None:
                    out.append(
                        {"stud": s, "hole": hh, "offset": list(ci.offset), "yaw": int(ci.yaw)}
                    )
        return out

    def drain_events(self) -> list[dict]:
        ev = []
        for kind, fn in (
            ("assembled", self.bc.get_assembled_connections),
            ("disassembled", self.bc.get_disassembled_connections),
        ):
            for ci in fn(True):
                ev.append(
                    {
                        "kind": kind,
                        "stud": self.path_to_id.get(ci.stud_path, ci.stud_path),
                        "hole": self.path_to_id.get(ci.hole_path, ci.hole_path),
                        "offset": list(ci.offset),
                        "yaw": int(ci.yaw),
                    }
                )
        return ev

    def interface_forces(self) -> dict[str, float]:
        """Largest stud/hole contact force pushing each hole brick down this step (BrickSim debug info)."""
        out: dict[str, float] = {}
        for d in self.bc.get_assembly_debug_infos():
            hid = self.path_to_id.get(d.hole_path)
            if hid is not None:
                out[hid] = max(out.get(hid, 0.0), -float(d.projected_force))
        return out

    def apply_up_force(self, oid: str, force_n: float) -> None:
        import carb
        import omni.usd
        from pxr import PhysicsSchemaTools

        pos = self.truth()["objects"][oid]["position"]
        self._physx.apply_force_at_pos(
            omni.usd.get_context().get_stage_id(),
            PhysicsSchemaTools.sdfPathToInt(self.paths[oid]),
            carb.Float3(0.0, 0.0, float(force_n)),
            carb.Float3(*pos),
            "Force",
        )

    def mass(self, oid: str) -> float:
        prim = self._world.stage.GetPrimAtPath(self.paths[oid])
        return float(prim.GetAttribute("physics:mass").Get())

    def structure_transforms(self, topology: dict) -> dict:
        return {int(k): v for k, v in self.bc.compute_structure_transforms(topology, 0).items()}

    def close(self) -> None:
        self._app.close()


__all__ = ["HandSpec", "LegoIsaacPhysics", "wxyz_to_xyzw"]
