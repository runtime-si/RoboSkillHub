# SPDX-License-Identifier: Apache-2.0
"""One i2rt YAM arm (linear_4310 gripper) in MuJoCo over the domino scene, with its own IK.

The arm is the ``left_`` subtree of the i2rt MJCF that RoboSimStudio ships
(``assets/embodiments/yam_i2rt/yam_i2rt_arms.xml``, MIT, Copyright (c) I2RT Robotics; the
licence file sits beside the meshes). The YAM station twin
(``twins_local/yam-station-20261103-gsota/assets/yam_station_arms.xml``) is the same file
with the bases 0.305 m instead of 0.31 m off centre, which does not matter for one arm. Kept
as shipped: link inertias, joint ranges (joint 6 +-2.0944 rad), armature/damping/friction,
gravity compensation, the position servos (arm kp 300 / kv 30, 27 / 10 N.m) and the finger
joints (0..0.0475 m, blade-tip gap = 2 * cmd - 1 mm).

Changed, because the shipped colliders are wrong for small parts:

- The fingertip meshes collide as convex hulls whose inner face stands ~13 mm proud of the
  pad (RoboSimStudio's ``linear_4310`` notes). Here they are visual only; each finger gets a
  pad box (the blade's last 23.7 mm, 26 mm wide, flush with the drawn pad face) and a root box
  (the stepped-back section above it), both measured from the tip mesh.
- The gripper housing mesh does not collide in the shipped file; here it gets a box (the
  mesh's extent, extended down to where the finger yokes start).
- Arm geoms collide with the scene but not with each other (no self-collision, as shipped).
- Finger servos: ``finger_kp`` / ``finger_force_n`` (defaults 600 / 80, as shipped); pads
  use direct contact stiffness like the floating-gripper scene.

Not modelled: the second arm, cameras, cables, joint-3 gravity sag and the station
supervisor's tracking/speed guards (the real arm needs 1.1-1.5x i2rt's gravity model),
motor current limits beyond the actuator force ranges, latency of the real 20 Hz command path.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .domino_mujoco_truth import mujoco_truth
from .domino_scene import DominoSceneSpec

ARM = "left"
JOINTS = tuple(f"{ARM}_joint{i}" for i in range(1, 7))
FINGERS = (f"{ARM}_joint7", f"{ARM}_joint8")
GRIP_OPEN_CMD = 0.0475
GAP_PER_CMD = 2.0
GAP_OFFSET = -0.001
"""Blade-tip gap = 2 * finger command - 1 mm (RoboSimStudio ``linear_4310``: span 0.09396 at 0.0475)."""
J1_HW_OFFSET = math.pi / 2
"""Real station joint 1 reads the model's joint 1 + pi/2 (twin ``station.py``)."""
WORK_POSE_HW = (
    1.5749217975127792,
    0.5956740672922862,
    0.4457541771572444,
    -0.12264438849469705,
    0.03986419470511926,
    0.0005722133211261138,
)
"""Where the real work arm starts a graph (twin.json ``work_arm_start_hw_rad``), hardware angles."""
WORK_POSE = (WORK_POSE_HW[0] - J1_HW_OFFSET, *WORK_POSE_HW[1:])


def default_asset_xml() -> Path:
    env = os.environ.get("ROBOSKILLHUB_YAM_MJCF")
    if env:
        return Path(env)
    return Path.home() / "robosimstudio/assets/embodiments/yam_i2rt/yam_i2rt_arms.xml"


@dataclass(frozen=True)
class YamMount:
    """Where the arm base stands in the scene (world = the domino scene's table frame)."""

    base_xy: tuple[float, float] = (0.0, 0.0)
    base_over_table_m: float = 0.021
    """Twin: bases 23 mm over the bench top (``base_raise_m``), table plane = 2 mm cloth top."""
    yaw: float = 0.0
    source: str = (
        "base height from twin yam-station-20261103-gsota (base_raise 0.023 - cloth 0.002); xy "
        "chosen so the floating-gripper scene's rack and line lie inside the real station's "
        "motion_server tool box (left base frame x -0.15..0.35, y 0.10..0.55 = model x "
        "0.10..0.55, y -0.35..0.15)"
    )


@dataclass(frozen=True)
class YamPhysicsConfig:
    mount: YamMount = field(default_factory=YamMount)
    finger_kp: float = 600.0
    finger_kv: float = 20.0
    finger_force_n: float = 80.0
    pad_friction: float = 1.0
    asset_xml: str | None = None


# ---- geometry probes on the shipped model ---------------------------------------------


def _shipped_xml(path: Path) -> str:
    xml = path.read_text()
    return xml.replace('meshdir="assets"', f'meshdir="{path.parent / "assets"}"')


def _arm_block(xml: str) -> str:
    start = xml.index(f'<body name="{ARM}_base"')
    depth, i = 0, start
    for m in re.finditer(r"<body\b|</body>", xml[start:]):
        depth += 1 if m.group(0) == "<body" else -1
        if depth == 0:
            i = start + m.end()
            break
    return xml[start:i]


def _mat(q_wxyz) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_wxyz(R: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(9))
    return q


def finger_boxes(xml: str) -> dict:
    """Pad and root boxes for each fingertip, in that tip body's frame, measured on the tip
    meshes with the fingers shut (gripper frame: z along the wrist, -z towards the TCP)."""
    import mujoco

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    g = m.body(f"{ARM}_gripper").id
    Rg, pg = d.xmat[g].reshape(3, 3), d.xpos[g]
    out = {}
    for tip in ("tip_left", "tip_right"):
        b = m.body(f"{ARM}_{tip}").id
        gid = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == b)
        mid = m.geom_dataid[gid]
        v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = (d.geom_xmat[gid].reshape(3, 3) @ v.T).T + d.geom_xpos[gid]
        loc = (Rg.T @ (w - pg).T).T  # gripper frame
        zmin = loc[:, 2].min()
        side = 1.0 if loc[:, 1].mean() > 0 else -1.0  # +y: the tip on the gripper's +y side
        pad = loc[loc[:, 2] < zmin + 0.0237]
        face = (pad[:, 1] * -side).max() * -side  # the face nearest the jaw centre
        root = loc[(loc[:, 2] >= zmin + 0.0237) & (loc[:, 2] < zmin + 0.0637)]
        boxes = {
            "pad": (
                [0.0, face + side * 0.003, zmin + 0.0237 / 2],
                [np.abs(pad[:, 0]).max(), 0.003, 0.0237 / 2],
            ),
            "root": (
                [0.0, face + side * 0.012, zmin + 0.0237 + 0.02],
                [np.abs(root[:, 0]).max(), 0.006, 0.02],
            ),
        }
        Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
        Rgb = Rb.T @ Rg  # gripper frame -> tip body frame
        qb = _quat_wxyz(Rgb)
        out[tip] = {
            k: {
                "pos": (Rb.T @ (pg + Rg @ np.array(c) - pb)).tolist(),
                "quat": qb.tolist(),
                "half": [float(h) for h in half],
                "gripper_frame_centre": [float(x) for x in c],
            }
            for k, (c, half) in boxes.items()
        }
    hid = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == g)
    mid = m.geom_dataid[hid]
    v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    w = (d.geom_xmat[hid].reshape(3, 3) @ v.T).T + d.geom_xpos[hid]
    loc = (Rg.T @ (w - pg).T).T
    lo, hi = loc.min(0), loc.max(0)
    lo[2] = min(lo[2], out["tip_left"]["root"]["gripper_frame_centre"][2] + 0.02)
    out["housing"] = {"pos": ((lo + hi) / 2).tolist(), "half": ((hi - lo) / 2).tolist()}
    return out


# ---- the scene ---------------------------------------------------------------------------


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def build_mjcf(
    spec: DominoSceneSpec, cfg: YamPhysicsConfig = YamPhysicsConfig(), base_table_z: float | None = None
) -> tuple[str, dict]:
    """The domino scene (same table, rack and tiles as ``domino_mujoco.build_mjcf``) with one
    YAM arm instead of the floating hand. The base stands ``base_over_table_m`` over
    ``base_table_z`` (the calibrated table; default the scene's), so a table built higher than
    its calibration does not lift the arm. Returns (xml, provenance)."""
    src = Path(cfg.asset_xml) if cfg.asset_xml else default_asset_xml()
    shipped = _shipped_xml(src)
    boxes = finger_boxes(shipped)
    arm = _arm_block(shipped)
    mt = cfg.mount
    base_z = (spec.table.top_z if base_table_z is None else base_table_z) + mt.base_over_table_m
    arm = re.sub(
        rf'<body name="{ARM}_base" gravcomp="1" pos="[^"]*"',
        f'<body name="{ARM}_base" gravcomp="1" pos="{mt.base_xy[0]} {mt.base_xy[1]} {base_z}" '
        f'euler="0 0 {mt.yaw}"',
        arm,
    )
    # every arm geom: collides with the scene (contype 2 & conaffinity 1), never with the arm
    arm = re.sub(r'<geom (?![^>]*contype)', '<geom contype="2" conaffinity="1" ', arm)
    for tip in ("tip_left", "tip_right"):
        # tip meshes: visual only; explicit pad and root boxes instead
        arm = re.sub(
            rf'(<body name="{ARM}_{tip}".*?<geom )contype="2" conaffinity="1" ',
            r'\1contype="0" conaffinity="0" ',
            arm,
            count=1,
            flags=re.S,
        )
        b = boxes[tip]
        pad_name = "gripper_pad_l" if tip == "tip_left" else "gripper_pad_r"
        extra = (
            f'<geom name="{pad_name}" type="box" size="{_fmt(b["pad"]["half"])}" '
            f'pos="{_fmt(b["pad"]["pos"])}" quat="{_fmt(b["pad"]["quat"])}" contype="2" conaffinity="1" '
            f'friction="{cfg.pad_friction} 0.01 0.0001" condim="4" solref="-10000 -20" solmix="100" '
            f'mass="0" rgba="0.2 0.2 0.2 0.6"/>'
            f'<geom name="{ARM}_{tip}_root" type="box" size="{_fmt(b["root"]["half"])}" '
            f'pos="{_fmt(b["root"]["pos"])}" quat="{_fmt(b["root"]["quat"])}" contype="2" conaffinity="1" '
            f'mass="0" rgba="0.2 0.2 0.2 0.3"/>'
        )
        arm = re.sub(
            rf'(<body name="{ARM}_{tip}"[^>]*>\s*<inertial[^>]*/>)', rf"\1{extra}", arm, count=1
        )
    h = boxes["housing"]
    arm = re.sub(
        rf'(<site name="{ARM}_tcp_site")',
        f'<geom name="gripper_palm" type="box" size="{_fmt(h["half"])}" pos="{_fmt(h["pos"])}" '
        f'contype="2" conaffinity="1" mass="0" rgba="0.2 0.2 0.2 0.2"/>\\1',
        arm,
        count=1,
    )
    assets = shipped[shipped.index("<asset>") : shipped.index("</asset>") + len("</asset>")]
    meshdir = re.search(r'meshdir="([^"]*)"', shipped).group(1)

    t, tb, ph = spec.tile, spec.table, spec.physics
    hx, hy, hz = t.half
    sx, sy, sz = (v / 2 for v in spec.feeder.separator_size)
    geoms = [
        f'<geom name="table" type="box" size="{tb.size_xy[0] / 2} {tb.size_xy[1] / 2} {tb.thickness / 2}" '
        f'pos="{tb.center_xy[0]} {tb.center_xy[1]} {tb.top_z - tb.thickness / 2}" friction="{tb.friction} 0.005 0.0001"/>'
    ]
    for k, p in enumerate(spec.separator_poses()):
        x, y, z = p["position"]
        geoms.append(
            f'<geom name="feeder_sep_{k}" type="box" size="{sx} {sy} {sz}" pos="{x} {y} {z}" '
            f'euler="0 0 {p["yaw"]}" friction="{tb.friction} 0.005 0.0001"/>'
        )
    tiles = []
    for k in range(spec.n_tiles):
        p = spec.feeder_pose(k)
        x, y, z = p["position"]
        tiles.append(
            f'<body name="tile_{k}" pos="{x} {y} {z}" euler="0 0 {p["yaw"]}"><freejoint name="tile_{k}_free"/>'
            f'<geom name="tile_{k}" type="box" size="{hx} {hy} {hz}" mass="{t.mass}" '
            f'friction="{t.friction} 0.005 0.0001" condim="4"/></body>'
        )
    acts = "".join(
        f'<position name="act_{j}" joint="{j}" kp="300" kv="30" forcerange="-{27 if i < 3 else 10} {27 if i < 3 else 10}"/>'
        for i, j in enumerate(JOINTS)
    ) + "".join(
        f'<position name="act_{j}" joint="{j}" kp="{cfg.finger_kp}" kv="{cfg.finger_kv}" '
        f'ctrlrange="0 {GRIP_OPEN_CMD}" forcerange="-{cfg.finger_force_n} {cfg.finger_force_n}"/>'
        for j in FINGERS
    )
    xml = f"""<mujoco model="domino_yam_i2rt_v1">
  <compiler angle="radian" meshdir="{meshdir}"/>
  <option timestep="{ph.mujoco_timestep}" gravity="0 0 {ph.gravity}" cone="elliptic" impratio="10" integrator="implicitfast"/>
  {assets}
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1"/>
    {"".join(geoms)}
    {"".join(tiles)}
    {arm}
  </worldbody>
  <contact>
    <exclude body1="{ARM}_gripper" body2="{ARM}_tip_left"/>
    <exclude body1="{ARM}_gripper" body2="{ARM}_tip_right"/>
    <exclude body1="{ARM}_tip_left" body2="{ARM}_tip_right"/>
  </contact>
  <actuator>{acts}</actuator>
</mujoco>"""
    mesh_dir = Path(meshdir)
    provenance = {
        "asset_xml": str(src),
        "asset_xml_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "meshes_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(mesh_dir.glob("*.stl"))
        },
        "licence": "MIT, Copyright (c) I2RT Robotics (LICENSE beside the meshes)",
        "licence_file_present": (src.parent / "LICENSE").exists(),
        "finger_boxes": boxes,
        "mount": {"base_xyz": [*mt.base_xy, base_z], "yaw": mt.yaw, "source": mt.source},
    }
    return xml, provenance


# ---- kinematics ----------------------------------------------------------------------------


def rot_err(R_target: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotation vector taking R to R_target (world frame)."""
    E = R_target @ R.T
    c = max(-1.0, min(1.0, (np.trace(E) - 1) / 2))
    a = math.acos(c)
    v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    if a < 1e-9:
        return 0.5 * v
    if math.pi - a < 1e-6:
        # 180 degrees: axis from the symmetric part
        B = (E + np.eye(3)) / 2
        k = int(np.argmax(np.diag(B)))
        ax = B[:, k] / math.sqrt(max(B[k, k], 1e-12))
        return a * ax
    return a / (2 * math.sin(a)) * v


class YamKinematics:
    """FK / damped-least-squares IK of the grasp site on a private MjData, inside the joint
    ranges. Tolerances follow the real station's ``solve_target`` (1 mm / 0.02 rad)."""

    def __init__(self, model, *, pos_tol=0.001, rot_tol=0.02):
        import mujoco

        self._mj = mujoco
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr = np.array([model.joint(j).qposadr[0] for j in JOINTS])
        self.vadr = np.array([model.joint(j).dofadr[0] for j in JOINTS])
        self.lo = np.array([model.joint(j).range[0] for j in JOINTS])
        self.hi = np.array([model.joint(j).range[1] for j in JOINTS])
        self.site = model.site(f"{ARM}_grasp_site").id
        self.pos_tol, self.rot_tol = pos_tol, rot_tol
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[self.qadr] = q
        self._mj.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, p_t, R_t, q0, *, iters: int = 150, max_travel: float | None = None):
        """(q, pos_err, rot_err, converged). ``max_travel`` bounds |q - q0| per joint."""
        mj = self._mj
        q = np.clip(np.asarray(q0, float), self.lo, self.hi)
        lo, hi = self.lo, self.hi
        if max_travel is not None:
            lo = np.maximum(lo, np.asarray(q0) - max_travel)
            hi = np.minimum(hi, np.asarray(q0) + max_travel)
        lam = 1e-3
        pe = re_ = float("inf")
        for _ in range(iters):
            p, R = self.fk(q)
            ep = np.asarray(p_t) - p
            er = rot_err(np.asarray(R_t), R)
            pe, re_ = float(np.linalg.norm(ep)), float(np.linalg.norm(er))
            if pe < self.pos_tol * 0.2 and re_ < self.rot_tol * 0.2:
                break
            mj.mj_comPos(self.m, self.d)
            mj.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
            J = np.vstack([self._jp[:, self.vadr], self._jr[:, self.vadr]])
            e = np.concatenate([ep, 0.3 * er])
            W = np.diag([1, 1, 1, 0.3, 0.3, 0.3])
            Jw = W @ J
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam * np.eye(6), e)
            n = np.abs(dq).max()
            if n > 0.2:
                dq *= 0.2 / n
            q = np.clip(q + dq, lo, hi)
        p, R = self.fk(q)
        pe = float(np.linalg.norm(np.asarray(p_t) - p))
        re_ = float(np.linalg.norm(rot_err(np.asarray(R_t), R)))
        return q, pe, re_, pe <= self.pos_tol and re_ <= self.rot_tol

    def at_limits(self, q, margin: float = 1e-3) -> list[str]:
        return [
            JOINTS[i]
            for i in range(6)
            if q[i] <= self.lo[i] + margin or q[i] >= self.hi[i] - margin
        ]


# ---- physics wrapper -----------------------------------------------------------------------


class YamDominoPhysics:
    """MuJoCo scene with one YAM arm; same ``truth()`` shape as ``domino_mujoco.MujocoBackend``."""

    name = "mujoco-yam"

    def __init__(self, cfg: YamPhysicsConfig = YamPhysicsConfig(), substeps: int | None = None):
        import mujoco

        self._mj = mujoco
        self.cfg = cfg
        self._substeps = substeps
        self._steps = 0

    def describe(self) -> dict:
        return {
            "name": "mujoco",
            "version": self._mj.__version__,
            "device": "cpu",
            "mjcf_sha256": getattr(self, "mjcf_sha256", None),
            "robot": "i2rt YAM, one arm (left), linear_4310 gripper; position servos as shipped",
            "assets": getattr(self, "provenance", None),
        }

    def reset(
        self, spec: DominoSceneSpec, feeder_poses: list[dict], q0=WORK_POSE, base_table_z=None
    ) -> None:
        mj = self._mj
        self.spec = spec
        xml, self.provenance = build_mjcf(spec, self.cfg, base_table_z)
        self.mjcf = xml
        self.mjcf_sha256 = hashlib.sha256(xml.encode()).hexdigest()
        self.m = mj.MjModel.from_xml_string(xml)
        self.d = mj.MjData(self.m)
        self.kin = YamKinematics(self.m)
        for k, p in enumerate(feeder_poses):
            adr = self.m.jnt_qposadr[self.m.joint(f"tile_{k}_free").id]
            x, y, z, w = p["quat_xyzw"]
            self.d.qpos[adr : adr + 7] = [*p["position"], w, x, y, z]
        self.qadr = self.kin.qadr
        self.fadr = np.array([self.m.joint(j).qposadr[0] for j in FINGERS])
        self.act_arm = np.array([self.m.actuator(f"act_{j}").id for j in JOINTS])
        self.act_fing = np.array([self.m.actuator(f"act_{j}").id for j in FINGERS])
        self.d.qpos[self.qadr] = q0
        self.d.qpos[self.fadr] = GRIP_OPEN_CMD * 0.3
        self.d.ctrl[self.act_arm] = q0
        self.d.ctrl[self.act_fing] = GRIP_OPEN_CMD * 0.3
        mj.mj_forward(self.m, self.d)
        self._dt_ctrl = 1.0 / spec.physics.control_hz
        self._sub = self._substeps or max(1, round(self._dt_ctrl / self.m.opt.timestep))
        self._steps = 0
        self.site = self.kin.site
        self.geom_alias = {
            "gripper_pad_l": "gripper_pad_l",
            "gripper_pad_r": "gripper_pad_r",
            f"{ARM}_tip_left_root": "gripper_palm",
            f"{ARM}_tip_right_root": "gripper_palm",
        }
        arm_bodies = {
            self.m.body(i).name
            for i in range(self.m.nbody)
            if self.m.body(i).name.startswith(f"{ARM}_")
        }
        self.body_alias = {b: "gripper_palm" for b in arm_bodies}

    @property
    def control_dt(self) -> float:
        return self._sub * self.m.opt.timestep

    @property
    def physics_steps(self) -> int:
        return self._steps

    def step(self, q_cmd, finger_cmd: float) -> float:
        self.d.ctrl[self.act_arm] = q_cmd
        self.d.ctrl[self.act_fing] = finger_cmd
        self._mj.mj_step(self.m, self.d, nstep=self._sub)
        self._steps += self._sub
        return float(self.d.time)

    def joints(self) -> np.ndarray:
        return self.d.qpos[self.qadr].copy()

    def finger_cmd_measured(self) -> float:
        return float(self.d.qpos[self.fadr].mean())

    def gap(self) -> float:
        """Blade-tip gap from the finger joint readings."""
        return max(0.0, GAP_PER_CMD * self.finger_cmd_measured() + GAP_OFFSET)

    def tcp(self) -> tuple[np.ndarray, np.ndarray]:
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def hand(self):
        """(x, y, z, yaw) of the measured TCP and the measured gap (``DominoBackend.hand`` shape)."""
        p, R = self.tcp()
        return (float(p[0]), float(p[1]), float(p[2]), math.atan2(R[1, 0], R[0, 0])), self.gap()

    def truth(self) -> dict:
        tiles = {f"tile_{k}": f"tile_{k}" for k in range(self.spec.n_tiles)}
        return mujoco_truth(self.m, self.d, tiles, self.geom_alias, self.body_alias)
