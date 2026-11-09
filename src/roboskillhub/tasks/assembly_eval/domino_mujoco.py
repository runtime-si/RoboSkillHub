# SPDX-License-Identifier: Apache-2.0
"""MuJoCo backend for the domino evaluation (CPU), built from the same scene spec.

For comparison with the Isaac/PhysX backend and with the YAM MuJoCo twin. The hand is a
gravity-compensated x/y/z/yaw mount with two sliding fingers, all driven by position
actuators with force limits; tiles are free bodies. After ``reset`` nothing writes a
tile's state: no welds, no equality constraints, no pose setters.
"""

from __future__ import annotations

from .domino_mujoco_truth import mujoco_truth
from .domino_scene import DominoSceneSpec


def build_mjcf(spec: DominoSceneSpec) -> str:
    t, tb, g, ph = spec.tile, spec.table, spec.gripper, spec.physics
    hx, hy, hz = t.half
    sx, sy, sz = (v / 2 for v in spec.feeder.separator_size)
    px, py, pz = (v / 2 for v in g.pad_size)
    mx, my, mz = (v / 2 for v in g.palm_size)
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
    lim = g.max_opening / 2
    z0 = spec.transit_tcp_z()
    return f"""<mujoco model="domino_eval_v1">
  <option timestep="{ph.mujoco_timestep}" gravity="0 0 {ph.gravity}" cone="elliptic" impratio="10"/>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1"/>
    {"".join(geoms)}
    {"".join(tiles)}
    <body name="mount_x" pos="0 0 0" gravcomp="1">
      <joint name="hx" type="slide" axis="1 0 0" range="-1 1" damping="5"/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="1e-4 1e-4 1e-4"/>
      <body name="mount_y" gravcomp="1">
        <joint name="hy" type="slide" axis="0 1 0" range="-1 1" damping="5"/>
        <inertial pos="0 0 0" mass="0.2" diaginertia="1e-4 1e-4 1e-4"/>
        <body name="mount_z" gravcomp="1">
          <joint name="hz" type="slide" axis="0 0 1" range="-1 1" damping="5" ref="0"/>
          <inertial pos="0 0 0" mass="0.2" diaginertia="1e-4 1e-4 1e-4"/>
          <body name="gripper_palm" gravcomp="1">
            <joint name="hyaw" type="hinge" axis="0 0 1" damping="0.2"/>
            <geom name="gripper_palm" type="box" size="{mx} {my} {mz}" pos="0 0 {pz + mz}" mass="0.2"/>
            <body name="gripper_pad_l" gravcomp="1">
              <joint name="fl" type="slide" axis="0 1 0" range="0 {lim}" damping="2"/>
              <geom name="gripper_pad_l" type="box" size="{px} {py} {pz}" pos="0 {py} 0" mass="0.02"
                    friction="{g.pad_friction} 0.01 0.0001" condim="4"
                    solref="-10000 -20" solmix="100"/>
            </body>
            <body name="gripper_pad_r" gravcomp="1">
              <joint name="fr" type="slide" axis="0 -1 0" range="0 {lim}" damping="2"/>
              <geom name="gripper_pad_r" type="box" size="{px} {py} {pz}" pos="0 -{py} 0" mass="0.02"
                    friction="{g.pad_friction} 0.01 0.0001" condim="4"
                    solref="-10000 -20" solmix="100"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="ax" joint="hx" kp="4000" kv="120" forcerange="-60 60"/>
    <position name="ay" joint="hy" kp="4000" kv="120" forcerange="-60 60"/>
    <position name="az" joint="hz" kp="4000" kv="120" forcerange="-60 60"/>
    <position name="ayaw" joint="hyaw" kp="20" kv="0.6" forcerange="-5 5"/>
    <position name="afl" joint="fl" kp="1500" kv="15" forcerange="-{g.grip_force} {g.grip_force}"/>
    <position name="afr" joint="fr" kp="1500" kv="15" forcerange="-{g.grip_force} {g.grip_force}"/>
  </actuator>
  <keyframe><key name="home" qpos="{" ".join(["0"] * (7 * spec.n_tiles))} 0 0 {z0} 0 {lim * 0.6} {lim * 0.6}"/></keyframe>
</mujoco>"""


class MujocoBackend:
    name = "mujoco"

    def __init__(self, substeps: int | None = None):
        import mujoco  # noqa: F401  (imported lazily; CPU only)

        self._mj = mujoco
        self._substeps = substeps
        self._steps = 0

    def describe(self) -> dict:
        return {
            "name": "mujoco",
            "version": self._mj.__version__,
            "device": "cpu",
            "mjcf_sha256": getattr(self, "mjcf_sha256", None),
            "hand": "gravity-compensated x/y/z/yaw mount + 2 sliding fingers (no arm)",
        }

    def reset(self, spec: DominoSceneSpec, feeder_poses: list[dict]) -> None:
        import hashlib

        mj = self._mj
        self.spec = spec
        xml = build_mjcf(spec)
        self.mjcf = xml
        self.mjcf_sha256 = hashlib.sha256(xml.encode()).hexdigest()
        self.m = mj.MjModel.from_xml_string(xml)
        self.d = mj.MjData(self.m)
        mj.mj_resetDataKeyframe(self.m, self.d, 0)
        # Setup only: place the rack tiles at the seeded poses before time starts.
        for k, p in enumerate(feeder_poses):
            adr = self.m.jnt_qposadr[self.m.joint(f"tile_{k}_free").id]
            x, y, z, w = p["quat_xyzw"]
            self.d.qpos[adr : adr + 7] = [*p["position"], w, x, y, z]
        self.d.ctrl[:] = [0, 0, spec.transit_tcp_z(), 0, self.d.qpos[-2], self.d.qpos[-1]]
        mj.mj_forward(self.m, self.d)
        self._dt_ctrl = 1.0 / spec.physics.control_hz
        self._sub = self._substeps or max(1, round(self._dt_ctrl / self.m.opt.timestep))
        self._steps = 0

    @property
    def control_dt(self) -> float:
        return self._sub * self.m.opt.timestep

    @property
    def physics_steps(self) -> int:
        return self._steps

    def step(self, hand_pose, opening: float) -> float:
        x, y, z, yaw = hand_pose
        half = max(0.0, opening / 2)
        self.d.ctrl[:] = [x, y, z, yaw, half, half]
        self._mj.mj_step(self.m, self.d, nstep=self._sub)
        self._steps += self._sub
        return float(self.d.time)

    def hand(self):
        q = self.d.qpos
        n = 7 * self.spec.n_tiles
        return (float(q[n]), float(q[n + 1]), float(q[n + 2]), float(q[n + 3])), float(
            q[n + 4] + q[n + 5]
        )

    def truth(self) -> dict:
        tiles = {f"tile_{k}": f"tile_{k}" for k in range(self.spec.n_tiles)}
        return mujoco_truth(self.m, self.d, tiles)
