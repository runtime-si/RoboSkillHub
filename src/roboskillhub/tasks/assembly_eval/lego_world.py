# SPDX-License-Identifier: Apache-2.0
"""World factory for Agent 5's LEGO graphs over the LEGO physics station, and the independent evaluation.

``physics_lego_world(seed, scene=..., breakage=False)`` returns the keys ``tasks.baselines.StationEnv`` expects
(``backend``, ``limits``, ``inputs``, ``describe``, ``privileged_allowed``, ``interventions``,
``task_success``) plus ``independent()`` (``lego_postconditions`` verdicts on the station's snapshots; no skill
result is read), ``station`` and ``contact_description`` (for the context shim, see
``examples/tasks/assembly_eval/lego_graph_baselines.py``).

Scenes are ``lego_scenes.SCENES`` (``single_on_plate``, ``single_on_brick``, ``seq3_stack``): the baseplate
(32 x 32) rests on the table at (0.30, 0, 0); type-1 scenes import their pre-built bricks with authored
connections; every brick to place starts loose in a feeder row on the table, seeded jitter +-1 mm / +-2 deg.
"""

from __future__ import annotations

import math
import random

from .lego_episode import load_config
from .lego_geometry import compose, q_from_axis_angle, q_mul, wxyz_to_xyzw
from .lego_postconditions import evaluate_episode
from .lego_scenes import SCENES
from .lego_station import BRICK_SIZE_M, PLATE_HEIGHT_M, Q_BRICKSIM_TO_CONTRACT, LegoPhysicsStation

PLATE_ID = "baseplate"
PLATE_POS = (0.30, 0.0, 0.0)
PLATE_ROT_WXYZ = (1.0, 0.0, 0.0, 0.0)
STUD_PITCH_M = 0.008
FEEDER = {"x0": 0.20, "dx": 0.05, "y": -0.20}
SPARES = 1
TRAVEL_MARGIN_M = 0.01
PREASSEMBLY_HEIGHT_M = 0.014
HOVER_BIAS_AWAY_M = 0.0005
HAND_START = (0.30, -0.10, 0.10, 0.0)


def _limits_dict() -> dict:
    return {
        "frame_id": "world",
        "workspace_min_m": [0.0, -0.35, 0.0],
        "workspace_max_m": [0.6, 0.35, 0.3],
        "max_speed_m_s": 0.1,
        "max_contact_distance_m": 0.3,
        "gripper_min_width_m": 0.0,
        "gripper_max_width_m": 0.05,
        "position_tolerance_m": 0.002,
        "max_contact_force_N": 10.0,
    }


def limits():
    from roboskillhub.runtime.contracts import Limits

    return Limits.from_dict(_limits_dict())


_PHYSICS = None


def make_physics():
    global _PHYSICS
    if _PHYSICS is None:  # one SimulationApp per process; fresh stage per episode
        from .lego_station_isaac import LegoIsaacPhysics

        _PHYSICS = LegoIsaacPhysics(headless=True)
    return _PHYSICS


def _topologies(scene_name: str) -> tuple[dict, dict, dict, list[int]]:
    """(goal topology, pre-built topology, {topology part id -> object id}, to-place part ids in BFS order)."""
    from bricksim.topology.legolization import legolization_json_to_topology_json
    from bricksim.topology.ordering import bfs_sort_connections

    sc = SCENES[scene_name]
    kw = {
        "include_base_plate": True,
        "base_plate_size": (32, 32),
        "base_plate_color": (200, 200, 200),
    }
    goal = legolization_json_to_topology_json(sc["goal"], color=(200, 30, 30), **kw)
    start = sc.get("start", {})
    pre = {0}
    for k, node in sc["goal"].items():
        if any(node == s for s in start.values()):
            pre.add(int(k))
    prebuilt = {
        **goal,
        "parts": [p for p in goal["parts"] if p["id"] in pre],
        "connections": [
            c for c in goal["connections"] if c["stud_id"] in pre and c["hole_id"] in pre
        ],
    }
    ids = {0: PLATE_ID, **{p: f"prebuilt_{p}" for p in pre if p != 0}}
    order: list[int] = []
    for c in bfs_sort_connections(goal)["connections"]:
        h = c["hole_id"]
        if h not in pre and h not in order:
            order.append(h)
    return goal, prebuilt, ids, order


def _plate_contract_pose(T_W_plate) -> dict:
    """Baseplate contract pose (top-surface centre) from the plate part's actual pose (bottom centre)."""
    p, q = compose(T_W_plate, ([0.0, 0.0, PLATE_HEIGHT_M], [0.0, 0.0, 0.0, 1.0]))
    return {"frame_id": "world", "position_m": list(p), "quaternion_xyzw": list(q)}


def placement_cell(goal_contract: dict, plate: dict, studs=(2, 4), plate_studs=(32, 32)) -> dict:
    """Grid cell of a seated contract-frame brick pose (Agent 3's Cell convention: lowest-index footprint
    corner, 0-based; contract ``layer`` 0 = on the plate)."""
    from .lego_geometry import relative

    p, q = relative(
        (plate["position_m"], plate["quaternion_xyzw"]),
        (goal_contract["position_m"], goal_contract["quaternion_xyzw"]),
    )
    x_axis = [1 - 2 * (q[1] ** 2 + q[2] ** 2), 2 * (q[0] * q[1] + q[2] * q[3])]
    yaw = math.degrees(math.atan2(x_axis[1], x_axis[0]))
    turns = round(yaw / 90.0) % 4
    rot = (turns * 90) % 180 if studs[0] != studs[1] else (turns * 90) % 360
    fx, fy = studs if rot in (0, 180) else (studs[1], studs[0])
    x = round((p[0] + plate_studs[0] * STUD_PITCH_M / 2) / STUD_PITCH_M - fx / 2)
    y = round((p[1] + plate_studs[1] * STUD_PITCH_M / 2) / STUD_PITCH_M - fy / 2)
    layer = round(p[2] / 0.0096)
    return {"stud_x": int(x), "stud_y": int(y), "layer": int(layer), "yaw_quarter_turns": rot // 90}


GRASP_ACROSS = {
    # close_axis 0: jaws close along the brick's 2-stud side (16 mm), TCP y along brick x.
    0: {
        "quaternion_xyzw": None,
        "open_width_m": 0.026,
        "release_open_width_m": 0.03,
        "carry_speed_m_s": 0.05,
    },
    # close_axis 1: jaws close along the 4-stud side (32 mm), TCP y along brick y; the pads stand on the
    # 2-stud end faces. The hand opens 50 mm (lego_station_isaac.HandSpec.max_opening).
    1: {
        "quaternion_xyzw": [1.0, 0.0, 0.0, 0.0],
        "open_width_m": 0.042,
        "release_open_width_m": 0.046,
        # Pinched by its 16 x 9.6 mm end faces the brick swings about the closing axis in a 20 mm/s carry
        # at 2 N (held_object_slipped in 1-2 of 10); at 4 mm/s it stayed within 0.2 mm.
        "carry_speed_m_s": 0.008,
    },
}


def _cell(m: dict, plate_pose: dict):
    from roboskillhub.tasks.lego.geometry import Cell

    c = placement_cell(m["mating_pose"], plate_pose)
    return Cell(c["stud_x"], c["stud_y"], c["layer"] + 1, c["yaw_quarter_turns"] * 90)


def grasp_close_axes(prebuilt: list[dict], targets: list[dict], plate_pose: dict) -> list[int]:
    """For each target in build order, the closing axis whose finger rows (``lego.geometry.finger_cells``,
    the rows ``lego.align_connection`` checks as ``approach_clear_of_build``) are clear of everything built
    before it - the pre-built bricks and the earlier targets - preferring the 16 mm side. Two bricks placed
    flush along their long sides (seq3_stack) need the second one held by its ends: across its 16 mm side one
    pad would go down between the two bricks."""
    from roboskillhub.tasks.lego.geometry import (
        PlacedBrick,
        brick_type,
        finger_cells,
        occupied_cells,
    )

    bt = brick_type("lego", [2, 4])
    build = [PlacedBrick(m["target_id"], bt, _cell(m, plate_pose)) for m in prebuilt]
    axes = []
    for t in targets:
        cell = _cell(t["mating"], plate_pose)
        occ = occupied_cells(build)
        top = max([c[2] for c in occ] + [cell.layer])
        clear = [a for a in (0, 1) if not any(c in occ for c in finger_cells(bt, cell, a, top))]
        axes.append(clear[0] if clear else 0)
        build.append(PlacedBrick(t["target_id"], bt, cell))
    return axes


def physics_lego_world(
    seed: int,
    *,
    scene: str = "single_on_plate",
    breakage: bool = False,
    grip_force_n: float = 2.0,
    force_limit_n: float = 6.0,
    fix_baseplate: bool = True,
    fault: str | None = None,
    fault_index: int | None = None,
) -> dict:
    physics = make_physics()
    goal_topo, prebuilt, ids, order = _topologies(scene)
    rng = random.Random(seed)
    feeder = []
    # Pieces are named after the targets they are meant for: the graph's verify step observes the target id
    # as the brick's object id (see integration-note).
    names = [f"b{pid}" for pid in order] + [f"spare_{k}" for k in range(SPARES)]
    for k, name in enumerate(names):
        yaw = math.radians(rng.uniform(-2.0, 2.0))
        feeder.append(
            {
                "id": name,
                "dims": (4, 2, 3),
                "pos": (
                    FEEDER["x0"] + FEEDER["dx"] * k + rng.uniform(-0.001, 0.001),
                    FEEDER["y"] + rng.uniform(-0.001, 0.001),
                    0.0005,
                ),
                "rot": (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
            }
        )
    cfg = load_config()
    scene_cfg = {
        "plate_id": PLATE_ID,
        "plate_pose": (PLATE_POS, PLATE_ROT_WXYZ),
        "prebuilt_topology": prebuilt,
        "prebuilt_ids": ids,
        "prebuilt_bricks": [v for k, v in ids.items() if k != 0],
        "feeder": feeder,
        "hand_start": HAND_START,
        "grip_force_N": grip_force_n,
        "fix_baseplate": fix_baseplate,
        "thresholds": {
            "distance_tolerance": cfg["clutch_model"]["assembly"]["distance_tolerance_m"],
            "max_penetration": cfg["clutch_model"]["assembly"]["max_penetration_m"],
            "z_angle_tolerance": math.radians(
                cfg["clutch_model"]["assembly"]["z_angle_tolerance_deg"]
            ),
            "required_force": cfg["clutch_model"]["assembly"]["required_force_n"],
            "yaw_tolerance": math.radians(cfg["clutch_model"]["assembly"]["yaw_tolerance_deg"]),
            "position_tolerance": cfg["clutch_model"]["assembly"]["position_tolerance_m"],
        },
    }
    station = LegoPhysicsStation(
        physics,
        scene_cfg,
        breakage=breakage,
        fault=fault,
        fault_index=len(order) - 1 if fault_index is None else fault_index,
    )
    station._snap("start")

    # Goal poses (BrickSim part frame, bottom centre) from the target topology and the plate's actual pose
    # (import_lego does not put the plate origin at the reference pose).
    rel = physics.structure_transforms(goal_topo)
    pl = physics.truth()["objects"][PLATE_ID]
    T_W_plate = (list(pl["position"]), list(pl["quat_xyzw"]))
    goal_bs = {}
    for pid, (rot, pos) in rel.items():
        goal_bs[pid] = compose(T_W_plate, (list(pos), wxyz_to_xyzw(rot)))
    plate_pose = _plate_contract_pose(T_W_plate)
    targets = []

    def mating(pid: int) -> dict:
        p, q = goal_bs[pid]
        mating_pose = {
            "frame_id": "world",
            "position_m": list(p),
            "quaternion_xyzw": q_mul(q, Q_BRICKSIM_TO_CONTRACT),
        }
        return {
            "kind": "lego_mating",
            "target_id": f"b{pid}" if pid in order else ids[pid],
            "frame_id": "world",
            "baseplate": {
                "object_id": PLATE_ID,
                "T_W_baseplate": plate_pose,
                "stud_pitch_m": STUD_PITCH_M,
                "plate_height_m": PLATE_HEIGHT_M,
                "studs": [32, 32],
                "calibration_id": "sim:bricksim-scene-truth",
            },
            "brick": {"class": "lego_brick", "part": "3001", "studs": [2, 4], "system": "lego"},
            "placement": placement_cell(mating_pose, plate_pose),
            "mating_pose": mating_pose,
            # The graph passes no ``build`` to align/verify, and align rejects a target whose depends_on
            # is not in ``build``: ordering lives in the target's own depends_on (select) only.
            "depends_on": [],
            "requires_support": False,
        }

    prebuilt_matings = [mating(pid) for pid in ids if pid != 0]
    for k, pid in enumerate(order):
        parents = [c["stud_id"] for c in goal_topo["connections"] if c["hole_id"] == pid]
        targets.append(
            {
                "target_id": f"b{pid}",
                "order": k,
                "depends_on": [f"b{s}" for s in parents if s in order],
                "connection_id": f"c{pid}",
                "mating": mating(pid),
                # For the 4a-local graph patch only (align/verify get ``build``): the pre-built bricks and the
                # targets ordered before this one, assumed placed.
                "build": prebuilt_matings + [mating(q) for q in order[:k]],
            }
        )
    for t, axis in zip(
        targets, grasp_close_axes(prebuilt_matings, targets, plate_pose), strict=True
    ):
        t["grasp"] = {
            "close_axis": axis,
            "T_object_TCP": {
                "frame_id": "object",
                "position_m": [0.0, 0.0, 0.013],
                "quaternion_xyzw": GRASP_ACROSS[axis]["quaternion_xyzw"]
                or q_mul(q_from_axis_angle([0, 0, 1], math.pi / 2), [1.0, 0.0, 0.0, 0.0]),
            },
            "open_width_m": GRASP_ACROSS[axis]["open_width_m"],
            # The release must open wider than the brick across the closing axis (32 mm for close_axis 1).
            "release_open_width_m": GRASP_ACROSS[axis]["release_open_width_m"],
            "carry_speed_m_s": GRASP_ACROSS[axis]["carry_speed_m_s"],
        }
    inputs = {
        "targets": targets,
        "feeder": {"pieces": [f["id"] for f in feeder]},
        # Integration (Agent 5): the plate lattice for perception.classify_object_state and the
        # pre-built bricks for core.bind_target (graph builds align/verify ``build`` from them).
        "lattice": {"T_W_baseplate": plate_pose, "plate_studs": [32, 32]},
        "prebuilt": prebuilt_matings,
        "gripper_id": "floating_parallel_gripper",
        "table_z": 0.0,
        # Carried bricks stay above the finished build's stud tops (plus TRAVEL_MARGIN_M) until they are
        # over their seat: core.transport_held's clear_above_z_m.
        "travel_clear_z_m": plate_pose["position_m"][2]
        + max(
            [
                m["placement"]["layer"] + 1
                for m in [*prebuilt_matings, *(t["mating"] for t in targets)]
            ]
            + [1]
        )
        * 0.0096
        + 0.0017
        + TRAVEL_MARGIN_M,
        "grasp": {
            # TCP 13 mm above the brick bottom (pad tips 2.8 mm above it), TCP z down, closing axis (TCP y)
            # along the brick's 16 mm side.
            "T_object_TCP": {
                "frame_id": "object",
                "position_m": [0.0, 0.0, 0.013],
                "quaternion_xyzw": q_mul(
                    q_from_axis_angle([0, 0, 1], math.pi / 2), [1.0, 0.0, 0.0, 0.0]
                ),
            },
            "pregrasp_height_m": 0.04,
            "lift_height_m": 0.04,
            "open_width_m": 0.026,
            "close_width_m": 0.010,
            "uncertainty": {"position_m": 0.001, "rotation_rad": 0.02},
            "collision_geometry": {"type": "box", "size_m": list(BRICK_SIZE_M)},
        },
        "align_limits": {
            "xy_tolerance_m": 0.0005,
            "yaw_tolerance_deg": 2.0,
            # The held brick hovers with its bottom over a same-layer neighbour's stud tops (9.6 + 1.7 mm)
            # plus 2.7 mm: at 10 mm a brick placed flush beside another hung 0.4 mm over its top face and
            # rubbed on it while align corrected (seq3_stack brick 2).
            "preassembly_height_m": PREASSEMBLY_HEIGHT_M,
            # Hover (and press) 0.5 mm off the seat away from a same-layer brick flush beside it: a brick
            # held by its end faces swings toward that neighbour as its walls meet the studs and landed on
            # the neighbour's studs in 8/10 seq3_stack runs (2 N) without it, 1/10 with it.
            "hover_bias_away_m": HOVER_BIAS_AWAY_M,
            "workspace_min_m": [0.0, -0.35, 0.0],
            "workspace_max_m": [0.6, 0.35, 0.3],
            "limits_source": "assembly_eval lego_world v1 (simulated station; not calibrated hardware)",
        },
        "press_limits": {
            "max_travel_m": PREASSEMBLY_HEIGHT_M + 0.0005,
            "speed_m_s": 0.004,
            "timeout_s": 20.0,
            "force_limit_N": force_limit_n,
            "hold_s": 0.2,
            # declared seat band for lego.guarded_seat's post-press check: 0.35 x 1.7 mm stud height
            "seat_depth_tolerance_m": 0.000595,
            "limits_source": "assembly_eval lego_world v1: press stop set per run (default 6 N, 6x BrickSim's "
            "1 N trigger); station max 10 N",
        },
        "release": {"open_width_m": 0.03, "retreat_height_m": 0.03},
        "verify_window": {"window_s": 0.5, "min_samples": 3, "max_obs_age_s": 2.0},
    }
    holder: dict = {}

    def independent() -> dict:
        if "result" not in holder:
            holder["result"] = independent_evaluation(station, goal_topo, ids, order, goal_bs, cfg)
        return holder["result"]

    def interventions() -> int:
        """Bricks a person would have to pick up: off the plate and off the feeder row at the end."""
        objs = physics.truth()["objects"]
        n = 0
        for oid, o in objs.items():
            if oid == PLATE_ID or oid.startswith("prebuilt_"):
                continue
            x, y, z = o["position"]
            on_plate = abs(x - PLATE_POS[0]) < 0.128 and abs(y - PLATE_POS[1]) < 0.128
            in_feeder = abs(y - FEEDER["y"]) < 0.02 and z < 0.002
            n += not (on_plate or in_feeder)
        return n

    describe = {
        **{k: v for k, v in station.describe().items() if k != "capabilities"},
        "seed": seed,
        "scene": scene,
        "breakage": breakage,
        "grip_force_N": grip_force_n,
        "press_force_limit_N": force_limit_n,
        "fault": fault,
        "fault_index": station.fault_index,
        "baseplate": "fixed to the table (fixed joint)"
        if fix_baseplate
        else "free on the table (as RoCo)",
        "targets": len(targets),
    }
    return {
        "backend": station,
        "limits": limits(),
        "inputs": inputs,
        "describe": describe,
        "task_success": lambda: independent()["outcome"]["status"] == "succeeded",
        "independent": independent,
        "privileged_allowed": True,
        "interventions": interventions,
        "station": station,
        "contact_description": station.contact_description(),
    }


def build_record(
    station: LegoPhysicsStation, goal_topo: dict, ids: dict, order: list[int], goal_bs: dict
) -> dict:
    """``lego_postconditions`` record from the station's snapshots and engine events (no skill output)."""
    snaps = station.snapshots
    events = station.engine_events
    releases = [e for e in station.events if e["event"] == "release"]
    closes = [e for e in station.events if e["event"] == "close"]

    def first(label, after, before=float("inf"), brick=None):
        for s in snaps:
            if (
                s["label"] == label
                and after <= s["sim_s"] <= before
                and (brick is None or s["brick"] == brick)
            ):
                return s
        return None

    def last(label, before):
        out = None
        for s in snaps:
            if s["label"] == label and s["sim_s"] <= before:
                out = s
        return out

    final = next((s for s in reversed(snaps) if s["label"] == "final"), snaps[-1])
    assigned: dict[int, str] = {}
    steps = []
    for r in releases:
        b = r["brick"]
        close = next(
            (c for c in reversed(closes) if c["sim_s"] <= r["sim_s"] and c["brick"] == b), None
        )
        t_close = close["sim_s"] if close else 0.0
        pick_start = last("pick_start", t_close) or snaps[0]
        rel_snap = first("released", r["sim_s"] - 1e-9, brick=b)
        nxt_close = next((c["sim_s"] for c in closes if c["sim_s"] > r["sim_s"]), float("inf"))
        settled = first("pick_start", r["sim_s"] + 1e-9, nxt_close + 1e-9) or final
        before = first("before_place", t_close, r["sim_s"], brick=b)
        # Which target this brick was for: the nearest unassigned goal to where it came to rest (a held brick
        # above layer 1 is as close to the layer-2 goal as to its own).
        pos = settled["parts"][b]["p"]
        free = [pid for pid in order if pid not in assigned]
        pid = min(free or order, key=lambda k: math.dist(goal_bs[k][0], pos))
        assigned.setdefault(pid, b)
        snapshots = {
            "pick_start": pick_start,
            "grasped": first("grasped", t_close, r["sim_s"]) or rel_snap,
            "before_place": before or rel_snap,
            "released": rel_snap,
            "retreated": first("retreated", r["sim_s"], settled["sim_s"] + 1e-9) or settled,
            "settled": settled,
        }
        n0, n1 = pick_start["n_engine_events"], settled["n_engine_events"]
        press = next(
            (
                e
                for e in station.events
                if e["event"] == "press" and t_close <= e["sim_s"] <= r["sim_s"]
            ),
            None,
        )
        steps.append(
            {
                "part": b,
                "target_part": pid,
                "pressed": before is not None,
                "press": press,
                "snapshots": snapshots,
                "events": events[n0:n1],
                "max_press_force_n": None,
            }
        )
    # A brick pressed but never released (the graph stopped first) is judged at the end, still in the hand.
    released_bricks = {r["brick"] for r in releases}
    for s in snaps:
        b = s["brick"]
        if s["label"] != "before_place" or b is None or b in released_bricks:
            continue
        released_bricks.add(b)
        close = next(
            (c for c in reversed(closes) if c["sim_s"] <= s["sim_s"] and c["brick"] == b), None
        )
        t_close = close["sim_s"] if close else 0.0
        pick_start = last("pick_start", t_close) or snaps[0]
        free = [pid for pid in order if pid not in assigned]
        pid = min(free or order, key=lambda k: math.dist(goal_bs[k][0], final["parts"][b]["p"]))
        assigned.setdefault(pid, b)
        press = next(
            (e for e in station.events if e["event"] == "press" and e["sim_s"] >= s["sim_s"]), None
        )
        steps.append(
            {
                "part": b,
                "target_part": pid,
                "pressed": True,
                "not_released": True,
                "press": press,
                "snapshots": {
                    "pick_start": pick_start,
                    "grasped": first("grasped", t_close, s["sim_s"]) or s,
                    "before_place": s,
                    "released": final,
                    "retreated": final,
                    "settled": final,
                },
                "events": events[pick_start["n_engine_events"] : final["n_engine_events"]],
                "max_press_force_n": None,
            }
        )
    obj = {0: PLATE_ID, **{k: v for k, v in ids.items() if k != 0}}
    for pid, b in assigned.items():
        obj[pid] = b
    for pid in order:
        obj.setdefault(pid, f"unplaced_b{pid}")
    parts = {PLATE_ID: {"lw": [32, 32], "role": "plate"}}
    for k, v in ids.items():
        if k != 0:
            parts[v] = {"lw": [4, 2], "role": "preplaced"}
    for b in station.brick_ids:
        parts.setdefault(b, {"lw": [4, 2], "role": "to_place"})
    for pid in order:
        parts.setdefault(obj[pid], {"lw": [4, 2], "role": "unplaced"})
    goal_poses = {}
    for pid, oid in obj.items():
        p, q = goal_bs[pid]
        goal_poses[oid] = {"p": list(p), "q": list(q)}
    goal_conns = [
        {
            "stud": obj[c["stud_id"]],
            "hole": obj[c["hole_id"]],
            "offset": list(c["offset"]),
            "yaw": int(c["yaw"]),
        }
        for c in goal_topo["connections"]
    ]
    start = next(s for s in snaps if s["label"] == "start")
    return {
        "parts": parts,
        "goal_poses": goal_poses,
        "start_poses": start["parts"],
        "goal_connections": goal_conns,
        "steps": steps,
        "final_snapshot": final,
        "assignment": {f"b{k}": v for k, v in assigned.items()},
    }


def independent_evaluation(
    station, goal_topo, ids, order, goal_bs, cfg, settle_s: float = 1.0
) -> dict:
    station.settle(settle_s)
    record = build_record(station, goal_topo, ids, order, goal_bs)
    probe = cfg["postconditions"]["probe"]
    for st in record["steps"]:
        st.update(
            station.probe(st["part"], probe["weight_ratio"], probe["steps"], probe["observe_steps"])
        )
    verdict = evaluate_episode(record, cfg)
    return {
        **verdict,
        "record": record,
        "events": station.events,
        "engine_events": station.engine_events,
    }
