"""kinoweave tool bundle: time-synchronised joint trajectories for the two YAM arms.

Wraps KinoWeave (https://github.com/botrail/kinoweave @ 2da9034697f150c3e0b8a1a69c807d13fca69126,
Apache-2.0, src/kinoweave/planner.py Planner + search/lacam.py) with the YAM pair built from
tools/yam-spheres/yam_spheres.json (link chain, link spheres, station base poses). Changed: problem
construction for YAM, hardware/model joint-1 offset, resampling to the joint-stream ``tracks`` format,
a sequential (one arm, then the other) baseline for comparison. Pure Python, CPU.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

SPHERES_JSON = Path(__file__).resolve().parent.parent / "yam-spheres" / "yam_spheres.json"
SIDES = ("left", "right")
# yam_spheres.json joint1_note: model joint1 = hardware joint1 - pi/2.
J1_OFFSET = math.pi / 2
# link5 -> link6 (joint6) is missing from yam_spheres.json's chain; values from the robosimstudio
# yam_i2rt_bimanual.xml body right_link6 / left_link6 (pos, quat wxyz, axis, range).
LINK6 = {"pos": [2.39858e-07, -0.0419481, 0.0404996], "quat_wxyz": [0.499998, -0.5, -0.5, -0.500002],
         "axis": [0.0, 0.0, -1.0], "range": [-2.0944, 2.0944]}
# Joint-stream limits (yam_gap/real STREAM_JOINTS.md, DUAL_ARM_SERVER.md): rows <= 30 Hz, adjacent rows
# <= 0.35 rad apart, velocity clipped to min(1.5, --stream-max-vel, 0.25*speed) on the station.
STREAM_MAX_HZ = 30.0
STREAM_MAX_ROW_JUMP_RAD = 0.35


def _xyzw(wxyz):
    w, x, y, z = wxyz
    return [x, y, z, w]


def _model(max_vel: float, max_acc: float, fingers: str) -> dict:
    d = json.loads(SPHERES_JSON.read_text())
    chain = d["chain"]
    joints, links = [], ["base"]
    order = ["link1", "link2", "link3", "link4", "link5"]
    parent = "base"
    for i, name in enumerate(order, 1):
        body = chain[name]
        lo, hi = body["joint"]["range"]
        joints.append({"name": f"joint{i}", "kind": "revolute", "parent": parent, "child": name,
                       "axis": body["joint"]["axis"], "mimic": None,
                       "origin": {"position": body["pos"], "quaternion": _xyzw(body["quat_wxyz"])},
                       "limits": {"lower": lo, "upper": hi, "velocity": max_vel, "acceleration": max_acc}})
        links.append(name)
        parent = name
    joints.append({"name": "joint6", "kind": "revolute", "parent": "link5", "child": "link6", "axis": LINK6["axis"],
                   "mimic": None, "origin": {"position": LINK6["pos"], "quaternion": _xyzw(LINK6["quat_wxyz"])},
                   "limits": {"lower": LINK6["range"][0], "upper": LINK6["range"][1], "velocity": max_vel,
                              "acceleration": max_acc}})
    links.append("link6")
    spheres = []
    for name in ["base"] + order:
        for s in d["spheres"][name]:
            spheres.append({"link": name, "center": s["center"], "radius": s["radius"]})
    for s in d[f"spheres_link6_fingers_{fingers}"]:
        spheres.append({"link": "link6", "center": s["center"], "radius": s["radius"]})
    # adjacent links overlap by construction
    allowed = [[a, b] for a, b in zip(links, links[1:])] + [["base", "link2"], ["link4", "link6"], ["link3", "link5"]]
    return {"name": "yam", "revision": "1", "links": links, "joints": joints,
            "joint_order": [f"joint{i}" for i in range(1, 7)], "collision_spheres": spheres,
            "allowed_collisions": allowed}


def _hw_to_model(q) -> list[float]:
    q = [float(v) for v in q][:6]
    q[0] -= J1_OFFSET
    return q


def _model_to_hw(q) -> list[float]:
    q = [float(v) for v in q]
    q[0] += J1_OFFSET
    return q


def _problem(starts: dict, goals: dict, *, max_vel, max_acc, margin, time_limit, time_step, fingers, seed) -> dict:
    d = json.loads(SPHERES_JSON.read_text())
    robots = []
    for side in SIDES:
        base = d["arm_bases"][side]
        robots.append({"name": side, "model": "yam",
                       "base_pose": {"position": base["pos"], "quaternion": _xyzw(base["quat_wxyz"])},
                       "start": _hw_to_model(starts[side]), "goal": _hw_to_model(goals[side])})
    return {"format": "kinoweave.problem", "schema_version": 1, "name": "yam-pair", "scene_revision": "1",
            "conventions": {"angle": "rad", "handedness": "right", "length": "m", "pose_direction": "parent_from_local",
                            "quaternion_order": "xyzw", "time": "s"},
            "config": {"backend": "reference", "device": "cpu", "collision_margin": margin,
                       "collision_tolerance": 1e-9, "joint_tolerance": 1e-6, "seed": seed, "time_limit": time_limit,
                       "time_step": time_step, "validation_time_step": min(0.05, time_step / 2)},
            "models": [_model(max_vel, max_acc, fingers)], "obstacles": [], "robots": robots}


def _solve(problem: dict, samples: int, neighbors: int, algorithm: str):
    from kinoweave import Planner, PlanningProblem, SearchConfig
    from kinoweave.io import loads
    from kinoweave.roadmap import PRMConfig

    prob = loads(json.dumps(problem), expected=PlanningProblem)
    with Planner(prob, search=SearchConfig(algorithm=algorithm),
                 prm=PRMConfig(num_samples=samples, neighbors=neighbors)) as planner:
        return planner.plan()


def _resample(result, hz: float, grips: dict) -> tuple[dict, float]:
    traj = result.trajectory
    duration = float(traj.times[-1])
    n = max(2, int(math.ceil(duration * hz)) + 1)
    ts = np.linspace(0.0, duration, n)
    rows = {s: [] for s in SIDES}
    for t in ts:
        state = traj.evaluate(float(t))
        for s in SIDES:
            rows[s].append(_model_to_hw(state[s].positions) + [float(grips[s])])
    return rows, duration


def _waits(result) -> dict:
    """Seconds each arm stands still between start and its last motion (yields to the other arm)."""
    traj = result.trajectory
    times = list(traj.times)
    out = {}
    for track in traj.tracks:
        pos = [np.asarray(p) for p in track.positions]
        moving = [float(np.max(np.abs(b - a))) > 1e-9 for a, b in zip(pos, pos[1:])]
        last = max((i for i, m in enumerate(moving) if m), default=-1)
        out[track.robot] = round(sum(times[i + 1] - times[i] for i in range(last + 1) if not moving[i]), 3)
    return out


def _row_jump(rows: dict) -> float:
    return max((float(np.max(np.abs(np.diff(np.asarray(r)[:, :6], axis=0)))) if len(r) > 1 else 0.0)
               for r in rows.values())


def plan_two_arms(left_start: list, left_goal: list, right_start: list, right_goal: list, *,
                  hz: float = 30.0, max_vel: float = 0.5, max_acc: float = 1.5, margin: float = 0.01,
                  time_limit: float = 20.0, time_step: float = 0.5, samples: int = 300, neighbors: int = 12,
                  algorithm: str = "lacam", fingers: str = "closed", left_grip: float = 0.0,
                  right_grip: float = 0.0, seed: int = 7, sequential_baseline: bool = True) -> dict[str, Any]:
    """Plan both arms from their starts to their own goals on one clock, with waits/yields where their
    swept volumes would meet; joints in the station's hardware convention (6 per arm, radians).

    Returns ``tracks`` ready for the joint-stream request (``{"left": [[j1..j6, grip_open], ...], "right":
    [...]}`` at ``hz``), ``makespan_s``, ``planning_s``, ``waits_s`` per arm, KinoWeave's sampled
    ``validation`` (YAM sphere model, ``margin`` m clearance), and the sequential baseline's makespan.
    """
    hz = float(min(hz, STREAM_MAX_HZ))
    starts = {"left": left_start, "right": right_start}
    goals = {"left": left_goal, "right": right_goal}
    kw = dict(max_vel=max_vel, max_acc=max_acc, margin=margin, time_limit=time_limit, time_step=time_step,
              fingers=fingers, seed=seed)
    t0 = time.perf_counter()
    result = _solve(_problem(starts, goals, **kw), samples, neighbors, algorithm)
    planning_s = time.perf_counter() - t0
    out: dict[str, Any] = {"found": result.status == "success", "status": result.status,
                           "termination": result.termination_reason, "planning_s": round(planning_s, 3),
                           "hz": hz, "validation": None, "tracks": None, "makespan_s": None}
    if result.validation is not None:
        out["validation"] = {"status": result.validation.status, "checks": list(result.validation.checks),
                             "issues": [str(i) for i in result.validation.issues][:5]}
    if out["found"]:
        rows, duration = _resample(result, hz, {"left": left_grip, "right": right_grip})
        out.update(tracks=rows, makespan_s=round(duration, 3), waits_s=_waits(result),
                   max_row_jump_rad=round(_row_jump(rows), 4),
                   stream_ok=_row_jump(rows) <= STREAM_MAX_ROW_JUMP_RAD)
    if sequential_baseline:
        seq, ok = 0.0, True
        first = _solve(_problem(starts, {"left": left_goal, "right": right_start}, **kw), samples, neighbors, algorithm)
        second = _solve(_problem({"left": left_goal, "right": right_start}, goals, **kw), samples, neighbors, algorithm)
        for r in (first, second):
            ok = ok and r.status == "success"
            if r.status == "success":
                seq += float(r.trajectory.times[-1])
        out["sequential_makespan_s"] = round(seq, 3) if ok else None
    return out


TOOLS = {"kinoweave.plan_two_arms": plan_two_arms}
