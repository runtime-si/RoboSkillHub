# SPDX-License-Identifier: Apache-2.0
"""Plan both YAM arms at once as one 12-DoF cuRobo robot and time them for the shortest makespan.

cuRobo (tools/curobo's pinned cuRoboV2) finds a collision-free 12-DoF path, inter-arm collision included,
from the merged tools/yam-spheres model (spheres already carry the 15 mm margin). The path is then
re-timed here: a forward/backward pass over the path parameter gives the fastest timing under per-joint
velocity and acceleration limits, and the whole timing is slowed uniformly until the jerk limit holds.
Both arms share one clock, so the makespan is the slower arm's motion along the shared path.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

HUB = Path(__file__).resolve().parents[3]
SPHERES = HUB / "tools" / "yam-spheres" / "yam_spheres.json"
URDF = HUB / "tools" / "curobo" / "robots" / "yam" / "yam_i2rt_arm.urdf"
ARMS = ("left", "right")
LINKS = ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]
VMAX = 0.25
"""rad/s per joint: dual_motion_server's default --stream-max-vel (stream_joints caps it at 1.5)."""
AMAX = 1.0
"""rad/s^2 per joint (not in the station config; a conservative choice for the 4310/4340 motors)."""
JMAX = 10.0
"""rad/s^3 per joint (as AMAX)."""
HZ = 30
"""stream_joints' highest row rate."""
CHUNK = 64
"""stream_joints' most rows per request."""


def load_model(path: Path = SPHERES) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- the 12-DoF robot


def _quat_to_rpy(q) -> tuple[float, float, float]:
    w, x, y, z = q
    r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    p = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return r, p, yw


def dual_urdf(model: dict, urdf: Path = URDF) -> str:
    """Two prefixed copies of the YAM URDF under a ``world`` link, each on its station base pose."""
    import xml.etree.ElementTree as ET

    out = ET.Element("robot", name="yam_dual")
    ET.SubElement(out, "link", name="world")
    src = ET.parse(urdf).getroot()
    for arm in ARMS:
        b = model["arm_bases"][arm]
        j = ET.SubElement(out, "joint", name=f"{arm}_mount", type="fixed")
        ET.SubElement(
            j,
            "origin",
            xyz=" ".join(f"{v:.6f}" for v in b["pos"]),
            rpy=" ".join(f"{v:.8f}" for v in _quat_to_rpy(b["quat_wxyz"])),
        )
        ET.SubElement(j, "parent", link="world")
        ET.SubElement(j, "child", link=f"{arm}_base_link")
        for el in src:
            e = ET.fromstring(ET.tostring(el))
            e.set("name", f"{arm}_{e.get('name')}")
            for tag in ("parent", "child"):
                t = e.find(tag)
                if t is not None:
                    t.set("link", f"{arm}_{t.get('link')}")
            out.append(e)
    return ET.tostring(out, encoding="unicode")


def dual_robot_cfg(model: dict, urdf_path: str) -> dict:
    """cuRobo robot_cfg (format 2.0) for both arms: arm-local self-collision ignores, arm-arm checked."""
    tcp = [float(v) for v in model["tcp"]["pos"] + model["tcp"]["quat_wxyz"]]
    body_of = {v: k for k, v in model["link_names_urdf"].items()}
    spheres, ignore, extra = {}, {}, {}
    for arm in ARMS:
        for i, link in enumerate(LINKS):
            src = (
                model["spheres_link6_fingers_open"]
                if link == "link_6"
                else model["spheres"][body_of[link]]
            )
            spheres[f"{arm}_{link}"] = [{"center": s["center"], "radius": s["radius"]} for s in src]
            nxt = [f"{arm}_{n}" for n in LINKS[i + 1 : i + 3]]
            if nxt:
                ignore[f"{arm}_{link}"] = nxt
        extra[f"{arm}_tcp"] = {
            "parent_link_name": f"{arm}_link_6",
            "link_name": f"{arm}_tcp",
            "joint_name": f"{arm}_tcp_joint",
            "joint_type": "FIXED",
            "fixed_transform": tcp,
        }
    ignore["left_base_link"] = ignore["left_base_link"] + ["right_base_link"]
    names = [f"{arm}_joint{i}" for arm in ARMS for i in range(1, 7)]
    return {
        "robot_cfg": {
            "kinematics": {
                "format_version": 2.0,
                "urdf_path": urdf_path,
                "asset_root_path": "",
                "base_link": "world",
                "tool_frames": [f"{arm}_tcp" for arm in ARMS],
                "extra_links": extra,
                "collision_link_names": list(spheres),
                "collision_spheres": spheres,
                "collision_sphere_buffer": 0.0,
                "self_collision_ignore": ignore,
                "self_collision_buffer": {k: 0.0 for k in spheres},
                "use_global_cumul": True,
                "cspace": {
                    "joint_names": names,
                    "default_joint_position": [0.0, 1.4, 1.2, 0.0, 1.1, 0.0] * 2,
                    "cspace_distance_weight": [1.0] * 12,
                    "null_space_weight": [1.0] * 12,
                    "max_acceleration": AMAX * 4,
                    "max_jerk": JMAX * 20,
                },
            }
        }
    }


# ---------------------------------------------------------------- sphere FK (numpy) for checks


def _mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rot(axis, a) -> np.ndarray:
    k = np.asarray(axis, float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def arm_spheres(model: dict, arm: str, q) -> dict[str, np.ndarray]:
    """World-frame spheres ``(n, 4)`` per URDF link of one arm at joints ``q`` (fingers open)."""
    b = model["arm_bases"][arm]
    R, p = _mat(b["quat_wxyz"]), np.asarray(b["pos"], float)
    body_of = {v: k for k, v in model["link_names_urdf"].items()}
    out = {}
    for i, link in enumerate(LINKS):
        if i:
            e = model["chain"][f"link{i}"]
            p = p + R @ np.asarray(e["pos"], float)
            R = R @ _mat(e["quat_wxyz"]) @ _rot(e["joint"]["axis"], q[i - 1])
        src = (
            model["spheres_link6_fingers_open"]
            if link == "link_6"
            else model["spheres"][body_of[link]]
        )
        c = np.array([s["center"] for s in src]) @ R.T + p
        out[link] = np.c_[c, [s["radius"] for s in src]]
    return out


def _gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        (
            np.linalg.norm(a[:, None, :3] - b[None, :, :3], axis=2) - a[:, None, 3] - b[None, :, 3]
        ).min()
    )


def inter_arm_distance(model: dict, q12) -> float:
    """Smallest sphere-sphere clearance between the two arms [m] (negative = overlap)."""
    L = np.vstack(list(arm_spheres(model, "left", q12[:6]).values()))
    R = np.vstack(list(arm_spheres(model, "right", q12[6:]).values()))
    return _gap(L, R)


def collisions(model: dict, q12) -> list[str]:
    """Overlapping link pairs at ``q12``: arm-arm, and within an arm except links one or two apart."""
    hits = []
    S = {arm: arm_spheres(model, arm, q12[6 * k : 6 * k + 6]) for k, arm in enumerate(ARMS)}
    for arm in ARMS:
        for i, a in enumerate(LINKS):
            for b in LINKS[i + 3 :]:
                if _gap(S[arm][a], S[arm][b]) < 0:
                    hits.append(f"{arm}_{a}/{arm}_{b}")
    if inter_arm_distance(model, q12) < 0:
        hits.append("left/right")
    return hits


# ---------------------------------------------------------------- time-optimal timing


def retime(
    path: np.ndarray, vmax=VMAX, amax=AMAX, jmax=JMAX, ds: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fastest timing of ``path`` (N, dof), at rest at both ends, under per-joint |v|, |a| (forward/backward
    pass on the path parameter) and |jerk| (uniform slow-down). Returns times (M,) and positions (M, dof)."""
    path = np.asarray(path, float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s_knots = np.r_[0.0, np.cumsum(seg)]
    total = s_knots[-1]
    if total < 1e-9:
        return np.array([0.0]), path[:1].copy()
    n = max(50, int(total / (ds or 0.002)))
    s = np.linspace(0, total, n)
    q = np.stack([np.interp(s, s_knots, path[:, j]) for j in range(path.shape[1])], 1)
    h = s[1] - s[0]
    dq = np.gradient(q, h, axis=0)
    ddq = np.gradient(dq, h, axis=0)
    vmax_, amax_ = np.broadcast_to(vmax, dq.shape[1:]), np.broadcast_to(amax, dq.shape[1:])
    with np.errstate(divide="ignore"):
        sd_v = np.min(np.where(np.abs(dq) > 1e-9, vmax_ / np.abs(dq), np.inf), 1)
        sd_c = np.min(np.where(np.abs(ddq) > 1e-9, np.sqrt(amax_ / np.abs(ddq)), np.inf), 1)
    cap = np.minimum(sd_v, sd_c)

    def acc_bounds(i, sd):
        lo, hi = -np.inf, np.inf
        for j in range(dq.shape[1]):
            if abs(dq[i, j]) < 1e-9:
                continue
            a1 = (amax_[j] - ddq[i, j] * sd * sd) / dq[i, j]
            a2 = (-amax_[j] - ddq[i, j] * sd * sd) / dq[i, j]
            lo, hi = max(lo, min(a1, a2)), min(hi, max(a1, a2))
        return lo, hi

    sd = cap.copy()
    sd[0] = sd[-1] = 0.0
    for i in range(n - 1):
        _, hi = acc_bounds(i, sd[i])
        sd[i + 1] = min(sd[i + 1], math.sqrt(max(sd[i] ** 2 + 2 * max(hi, 0.0) * h, 0.0)))
    for i in range(n - 1, 0, -1):
        lo, _ = acc_bounds(i, sd[i])
        sd[i - 1] = min(sd[i - 1], math.sqrt(max(sd[i] ** 2 - 2 * min(lo, 0.0) * h, 0.0)))
    mid = 0.5 * (sd[1:] + sd[:-1])
    dt = np.where(mid > 1e-9, h / np.maximum(mid, 1e-9), 0.0)
    dt[mid <= 1e-9] = np.sqrt(2 * h / np.max(amax_)) * 2
    t = np.r_[0.0, np.cumsum(dt)]
    tt = np.arange(0.0, t[-1] + 1e-9, 1.0 / 240)
    qt = np.stack([np.interp(tt, t, q[:, j]) for j in range(q.shape[1])], 1)
    jerk = np.abs(np.diff(qt, 3, axis=0)).max() * 240**3 if len(qt) > 3 else 0.0
    k = max(1.0, (jerk / jmax) ** (1 / 3))
    return t * k, q


def sample(t: np.ndarray, q: np.ndarray, hz: float = HZ) -> np.ndarray:
    """Rows of ``q`` at ``hz`` over ``t`` (last row = goal)."""
    tt = np.r_[np.arange(0.0, t[-1], 1.0 / hz), t[-1]]
    return np.stack([np.interp(tt, t, q[:, j]) for j in range(q.shape[1])], 1)


def to_tracks(rows12: np.ndarray, grip_open=(1.0, 1.0)) -> list[dict[str, list]]:
    """``stream_joints`` chunks: ``{"left": [[j1..j6, grip_open], ...], "right": [...]}``, <= 64 rows each,
    model joint angles converted to hardware (joint1 + pi/2)."""
    hw = rows12.copy()
    hw[:, 0] += math.pi / 2
    hw[:, 6] += math.pi / 2
    out = []
    for i in range(0, len(hw), CHUNK):
        c = hw[i : i + CHUNK]
        out.append(
            {
                "left": [[*map(float, r[:6]), float(grip_open[0])] for r in c],
                "right": [[*map(float, r[6:]), float(grip_open[1])] for r in c],
            }
        )
    return out


def waypoints(t: np.ndarray, q: np.ndarray, n: int = 8) -> list[dict[str, Any]]:
    """A few timed joint waypoints per arm (model angles) for go_to_pose / move connectors."""
    idx = np.unique(np.linspace(0, len(t) - 1, n).round().astype(int))
    return [{"t": float(t[i]), "left": q[i, :6].tolist(), "right": q[i, 6:].tolist()} for i in idx]


# ---------------------------------------------------------------- planner


class DualArmPlanner:
    """One cuRobo MotionPlanner over both arms (GPU)."""

    def __init__(self, model: dict | None = None, num_trajopt_seeds: int = 4):
        from curobo._src.state.state_joint import JointState
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        self.model = model or load_model()
        self._dir = tempfile.mkdtemp(prefix="yam_dual_")
        urdf = Path(self._dir) / "yam_dual.urdf"
        urdf.write_text(dual_urdf(self.model))
        self.device = DeviceCfg()
        self.JointState = JointState
        cfg = MotionPlannerCfg.create(
            robot=dual_robot_cfg(self.model, str(urdf)),
            self_collision_check=True,
            device_cfg=self.device,
            num_trajopt_seeds=num_trajopt_seeds,
            use_cuda_graph=False,
        )
        self.planner = MotionPlanner(cfg)
        self.joint_names = list(self.planner.joint_names)

    def _js(self, q):
        import torch

        pos = self.device.to_device(torch.tensor(np.asarray(q, float)[None], dtype=torch.float32))
        return self.JointState.from_position(pos, joint_names=self.joint_names)

    def path(self, q_start, q_goal, max_attempts: int = 5) -> tuple[np.ndarray | None, str]:
        res = self.planner.plan_cspace(
            self._js(q_goal), self._js(q_start), max_attempts=max_attempts
        )
        if res is None or not bool(res.success.any()):
            return None, str(getattr(res, "status", None) or "no feasible trajectory")
        traj = res.get_interpolated_plan()
        arr = traj.position.detach().cpu().numpy().reshape(-1, len(self.joint_names))
        return arr.astype(float), "ok"


def plan(
    planner: DualArmPlanner,
    start: dict,
    goal: dict,
    vmax=VMAX,
    amax=AMAX,
    jmax=JMAX,
    hz: float = HZ,
    grip_open=(1.0, 1.0),
) -> dict[str, Any]:
    """``start`` / ``goal``: ``{"left": 6 model angles, "right": ...}``. Exits planned / no_path /
    start_in_collision (also returned when the goal itself collides, with ``detail`` saying so)."""
    m = planner.model
    q0 = np.r_[start["left"], start["right"]].astype(float)
    q1 = np.r_[goal["left"], goal["right"]].astype(float)
    for name, q in (("start", q0), ("goal", q1)):
        hits = collisions(m, q)
        if hits:
            return {
                "exit": "start_in_collision" if name == "start" else "no_path",
                "detail": f"{name} in collision (spheres): {', '.join(hits)}",
            }
    path, why = planner.path(q0, q1)
    if path is None:
        return {"exit": "no_path", "detail": why}
    path[0], path[-1] = q0, q1
    t, q = retime(path, vmax, amax, jmax)
    rows = sample(t, q, hz)
    dmin = min(inter_arm_distance(m, r) for r in rows)
    return {
        "exit": "planned",
        "makespan_s": float(t[-1]),
        "hz": hz,
        "rows": len(rows),
        "tracks": to_tracks(rows, grip_open),
        "waypoints": waypoints(t, q),
        "min_inter_arm_sphere_m": dmin,
        "path_points": len(path),
    }


def run(
    ctx,
    start: dict,
    goal: dict,
    vmax: float = VMAX,
    amax: float = AMAX,
    jmax: float = JMAX,
    hz: float = HZ,
) -> dict[str, Any]:
    global _PLANNER
    if _PLANNER is None:
        _PLANNER = DualArmPlanner()
    return plan(_PLANNER, start, goal, vmax, amax, jmax, hz)


_PLANNER: DualArmPlanner | None = None
