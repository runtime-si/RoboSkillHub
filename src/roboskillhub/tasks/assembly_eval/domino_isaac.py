# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim 5.1 (PhysX) backend for the domino evaluation, built from the scene spec.

Must run inside an Isaac Sim Python (``isaacsim`` pip package). Builds a fresh USD stage
per episode: static table and rack colliders, free rigid-body tiles with gravity and
friction, and a hand articulation (fixed base -> x/y/z prismatic -> yaw revolute -> two
prismatic fingers) moved only through joint drives with force limits. After ``reset``
nothing writes a tile pose or velocity, no tile is jointed or made kinematic, and no
collision pair is filtered: the only fixed joint is the hand's base to the world.

Truth comes from the PhysX tensor views (tile poses) and PhysX contact reports (pairs
with separation <= 0.5 mm), both labelled ``simulation_truth``.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path

from .domino_geometry import wxyz_to_xyzw
from .domino_scene import DominoSceneSpec

TOUCH_SEPARATION = 0.0005


class IsaacBackend:
    name = "isaac"

    def __init__(
        self, headless: bool = True, render_dir: Path | None = None, experience: str | None = None
    ):
        import os

        import isaacsim
        from isaacsim import SimulationApp

        # The lean python experience loads far fewer extensions than the full one (RAM on a
        # shared host); override with ISAAC_EXPERIENCE (a .kit path, or "" for the default).
        exp = os.environ.get("ISAAC_EXPERIENCE", experience)
        if exp is None:
            lean = Path(isaacsim.__file__).parent / "apps" / "isaacsim.exp.base.python.kit"
            exp = str(lean) if lean.exists() else ""
        self.experience = exp
        self._app = SimulationApp(
            {
                "headless": headless,
                "width": 640,
                "height": 480,
                "extra_args": ["--/app/settings/persistent=false"],
            },
            experience=exp,
        )

        self._version = (
            (Path(isaacsim.__file__).parent / "VERSION").read_text().strip()
            if (Path(isaacsim.__file__).parent / "VERSION").exists()
            else "unknown"
        )
        self.render_dir = render_dir
        self._steps = 0
        self._world = None
        self.usd_sha256 = None

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        try:
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
        except Exception:  # pragma: no cover - informational only
            gpu = "unknown"
        return {
            "name": "isaac",
            "isaacsim_version": self._version,
            "physics": "PhysX via isaacsim.core.api.World (CPU simulation device, TGS default solver)",
            "gpu": gpu,
            "usd_sha256": self.usd_sha256,
            "experience": self.experience,
            "hand": "fixed-base x/y/z prismatic + yaw revolute + 2 prismatic fingers (no arm)",
        }

    def close(self) -> None:
        self._app.close()

    @property
    def control_dt(self) -> float:
        return self._sub * self._dt

    @property
    def physics_steps(self) -> int:
        return self._steps

    # ------------------------------------------------------------------
    def reset(self, spec: DominoSceneSpec, feeder_poses: list[dict]) -> None:
        import numpy as np
        import omni.physx
        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim, SingleArticulation
        from isaacsim.core.utils.stage import create_new_stage
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

        self.spec = spec
        ph = spec.physics
        self._dt = ph.isaac_dt
        self._sub = max(1, round((1.0 / ph.control_hz) / self._dt))
        if self._world is not None:
            self._world.stop()
            World.clear_instance()
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
        scene_path = "/physicsScene"
        scene = UsdPhysics.Scene.Get(stage, scene_path)
        if not scene:
            scene = UsdPhysics.Scene.Define(stage, scene_path)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(abs(ph.gravity))
        px_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        px_scene.CreateEnableCCDAttr(True)
        px_scene.CreateBounceThresholdAttr(0.05)
        px_scene.CreateFrictionOffsetThresholdAttr(0.001)
        px_scene.CreateFrictionCorrelationDistanceAttr(0.0005)

        def material(path, friction):
            mat = UsdShade.Material.Define(stage, path)
            api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            api.CreateStaticFrictionAttr(friction)
            api.CreateDynamicFrictionAttr(friction)
            api.CreateRestitutionAttr(0.0)
            return mat

        mats = {
            "table": material("/World/Materials/table", spec.table.friction),
            "tile": material("/World/Materials/tile", spec.tile.friction),
            "pad": material("/World/Materials/pad", spec.gripper.pad_friction),
        }

        def bind(prim, mat):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                mats[mat], UsdShade.Tokens.weakerThanDescendants, "physics"
            )

        def collider(prim):
            UsdPhysics.CollisionAPI.Apply(prim)
            pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            pc.CreateContactOffsetAttr(ph.isaac_contact_offset)
            pc.CreateRestOffsetAttr(ph.isaac_rest_offset)

        def cube(path, size, pos, yaw=0.0, quat_xyzw=None):
            c = UsdGeom.Cube.Define(stage, path)
            c.CreateSizeAttr(1.0)
            x = UsdGeom.Xformable(c)
            x.ClearXformOpOrder()
            x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
            if quat_xyzw is None:
                quat_xyzw = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
            qx, qy, qz, qw = quat_xyzw
            x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(qw, qx, qy, qz))
            x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*size))
            return c.GetPrim()

        tb = spec.table
        prim = cube(
            "/World/table",
            (tb.size_xy[0], tb.size_xy[1], tb.thickness),
            (tb.center_xy[0], tb.center_xy[1], tb.top_z - tb.thickness / 2),
        )
        collider(prim)
        bind(prim, "table")
        for k, p in enumerate(spec.separator_poses()):
            prim = cube(
                f"/World/feeder_sep_{k}", spec.feeder.separator_size, p["position"], p["yaw"]
            )
            collider(prim)
            bind(prim, "table")
        t = spec.tile
        self._tile_paths = []
        for k, p in enumerate(feeder_poses):
            path = f"/World/tile_{k}"
            prim = cube(
                path, (t.thickness, t.width, t.height), p["position"], quat_xyzw=p["quat_xyzw"]
            )
            UsdPhysics.RigidBodyAPI.Apply(prim)
            prb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            prb.CreateSolverPositionIterationCountAttr(ph.isaac_solver_position_iterations)
            prb.CreateSolverVelocityIterationCountAttr(ph.isaac_solver_velocity_iterations)
            prb.CreateEnableCCDAttr(True)
            prb.CreateSleepThresholdAttr(0.0)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(t.mass)
            collider(prim)
            bind(prim, "tile")
            cr = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            cr.CreateThresholdAttr(0.0)
            self._tile_paths.append(path)

        # ---- hand articulation ----------------------------------------
        g = spec.gripper
        z0 = spec.transit_tcp_z()
        self._z0 = z0
        self._h0 = 0.6 * g.max_opening / 2  # authored half opening
        pxs, pys, pzs = g.pad_size
        mxs, mys, mzs = g.palm_size

        def link(path, pos, mass, inertia, gravity=False):
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
            prb_ = PhysxSchema.PhysxRigidBodyAPI.Apply(p_)
            prb_.CreateDisableGravityAttr(not gravity)
            prb_.CreateSolverPositionIterationCountAttr(ph.isaac_solver_position_iterations)
            return p_

        H = "/World/hand"
        UsdGeom.Xform.Define(stage, H)
        base = link(f"{H}/base", (0, 0, z0), 1.0, (1e-3, 1e-3, 1e-3))
        art = UsdPhysics.ArticulationRootAPI.Apply(base)
        pxa = PhysxSchema.PhysxArticulationAPI.Apply(base)
        pxa.CreateEnabledSelfCollisionsAttr(False)
        pxa.CreateSolverPositionIterationCountAttr(ph.isaac_solver_position_iterations)
        del art
        fj = UsdPhysics.FixedJoint.Define(stage, f"{H}/world_fixed")
        fj.CreateBody1Rel().SetTargets([base.GetPath()])
        fj.CreateLocalPos0Attr(Gf.Vec3f(0, 0, z0))
        fj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        lx = link(f"{H}/mount_x", (0, 0, z0), 0.2, (1e-4, 1e-4, 1e-4))
        ly = link(f"{H}/mount_y", (0, 0, z0), 0.2, (1e-4, 1e-4, 1e-4))
        lz = link(f"{H}/mount_z", (0, 0, z0), 0.2, (1e-4, 1e-4, 1e-4))
        palm = link(f"{H}/gripper_palm", (0, 0, z0), 0.2, (1.2e-4, 5e-5, 1.5e-4))
        pc = cube(f"{H}/gripper_palm/geom", (mxs, mys, mzs), (0, 0, pzs / 2 + mzs / 2))
        collider(pc)
        bind(pc, "pad")
        pads = {}
        for name, sign in (("gripper_pad_l", 1), ("gripper_pad_r", -1)):
            pl = link(f"{H}/{name}", (0, sign * self._h0, z0), 0.02, (3e-6, 3e-6, 3e-6))
            geom = cube(f"{H}/{name}/geom", (pxs, pys, pzs), (0, sign * pys / 2, 0))
            collider(geom)
            bind(geom, "pad")
            pads[name] = pl

        def drive(joint_prim, kind, stiffness, damping, max_force):
            d = UsdPhysics.DriveAPI.Apply(joint_prim, kind)
            d.CreateTypeAttr("force")
            d.CreateStiffnessAttr(stiffness)
            d.CreateDampingAttr(damping)
            d.CreateMaxForceAttr(max_force)
            d.CreateTargetPositionAttr(0.0)

        def prismatic(name, b0, b1, axis, lo, hi, offset1=(0, 0, 0)):
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{H}/{name}")
            j.CreateBody0Rel().SetTargets([b0.GetPath()])
            j.CreateBody1Rel().SetTargets([b1.GetPath()])
            j.CreateAxisAttr(axis)
            j.CreateLowerLimitAttr(lo)
            j.CreateUpperLimitAttr(hi)
            j.CreateLocalPos0Attr(Gf.Vec3f(*offset1))
            j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            return j.GetPrim()

        jx = prismatic("hx", base, lx, "X", -1.0, 1.0)
        jy = prismatic("hy", lx, ly, "Y", -1.0, 1.0)
        jz = prismatic("hz", ly, lz, "Z", -1.0, 1.0)
        for j in (jx, jy, jz):
            drive(j, "linear", 4000.0, 120.0, 60.0)
        jr = UsdPhysics.RevoluteJoint.Define(stage, f"{H}/hyaw")
        jr.CreateBody0Rel().SetTargets([lz.GetPath()])
        jr.CreateBody1Rel().SetTargets([palm.GetPath()])
        jr.CreateAxisAttr("Z")
        drive(jr.GetPrim(), "angular", 20.0 * math.pi / 180, 0.6 * math.pi / 180, 5.0)
        lim = g.max_opening / 2
        jl = prismatic(
            "fl", palm, pads["gripper_pad_l"], "Y", -self._h0, lim - self._h0, (0, self._h0, 0)
        )
        jrr = prismatic(
            "fr", palm, pads["gripper_pad_r"], "Y", -(lim - self._h0), self._h0, (0, -self._h0, 0)
        )
        for j in (jl, jrr):
            drive(j, "linear", 1500.0, 15.0, g.grip_force)

        out = Path("/tmp") / f"domino_eval_{id(self)}.usda"
        stage.GetRootLayer().Export(str(out))
        self.usd_sha256 = hashlib.sha256(out.read_bytes()).hexdigest()
        self.usd_path = str(out)

        world.reset()
        self._art = SingleArticulation(f"{H}/base", name="hand")
        self._art.initialize()
        self._tiles = RigidPrim(self._tile_paths, name="tiles")
        self._tiles.initialize()
        names = list(self._art.dof_names)
        self._idx = [names.index(n) for n in ("hx", "hy", "hz", "hyaw", "fl", "fr")]
        self._np = np
        self._steps = 0
        self._t = 0.0
        self._pairs: dict[tuple[str, str], float] = {}
        self._physx = omni.physx.get_physx_simulation_interface()
        from pxr import PhysicsSchemaTools

        self._int_to_path = PhysicsSchemaTools.intToSdfPath
        from omni.physx.bindings._physx import ContactEventType

        self._lost = ContactEventType.CONTACT_LOST
        self._cmd = (0.0, 0.0, z0, 0.0, self._h0 * 2)
        self._apply(self._cmd[:4], self._cmd[4])

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
        name = path.rsplit("/", 1)[-1]
        if name == "geom":
            name = path.rsplit("/", 2)[-2]
        return "feeder" if name.startswith("feeder_sep") else name

    def _read_contacts(self):
        headers, data = self._physx.get_contact_report()
        for h in headers:
            a = self._body(str(self._int_to_path(h.actor0)))
            b = self._body(str(self._int_to_path(h.actor1)))
            key = tuple(sorted((a, b)))
            if h.type == self._lost:
                self._pairs.pop(key, None)
                continue
            seps = [data[h.contact_data_offset + i].separation for i in range(h.num_contact_data)]
            self._pairs[key] = min(seps) if seps else 0.0

    def step(self, hand_pose, opening: float) -> float:
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

    def truth(self) -> dict:
        pos, quat = self._tiles.get_world_poses()
        objs = {}
        for k in range(self.spec.n_tiles):
            objs[f"tile_{k}"] = {
                "position": [float(v) for v in pos[k]],
                "quat_xyzw": [float(v) for v in wxyz_to_xyzw(quat[k])],
            }
        contacts: dict[str, set] = {f"tile_{k}": set() for k in range(self.spec.n_tiles)}
        for (a, b), sep in self._pairs.items():
            if sep > TOUCH_SEPARATION:
                continue
            if a in contacts:
                contacts[a].add(b)
            if b in contacts:
                contacts[b].add(a)
        return {
            "t": self._t,
            "objects": objs,
            "contacts": {k: sorted(v) for k, v in contacts.items()},
        }
