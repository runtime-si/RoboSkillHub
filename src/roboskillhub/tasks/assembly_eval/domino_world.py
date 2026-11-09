# SPDX-License-Identifier: Apache-2.0
"""World factory for Agent 5's baselines over the domino physics station, and the
independent evaluation of each episode.

``physics_domino_world(seed, physics="mujoco"|"isaac", slots=3, fault=None)`` returns the
same keys as ``tasks.fake_world.domino_world`` (``backend``, ``limits``, ``inputs``,
``describe``, ``task_success``, ``privileged_allowed``, ``interventions``) plus
``independent()``: after the graph ends it lets the scene settle with the hand parked and
judges every release with ``evaluate_placement`` and every slot with ``evaluate_slots`` on
the simulator-truth sample stream, without reading any skill result. ``task_success`` is
that layout verdict.
"""

from __future__ import annotations

from roboskillhub.runtime.contracts import Limits

from .domino_geometry import dist, tilt_deg
from .domino_postcondition import DEFAULT_THRESHOLDS, evaluate_placement, evaluate_slots
from .domino_runner import feeder_poses_for_seed
from .domino_scene import DominoSceneSpec, FeederSpec, TableSpec, TargetLine
from .domino_station import DominoPhysicsStation

TABLE_Z = 0.0
SPARES = 4


def world_spec(slots: int, *, table_offset_m: float = 0.0) -> DominoSceneSpec:
    """Physics scene: table top at ``TABLE_Z + table_offset_m`` (the offset is the injected
    calibration error), ``slots + 4`` pieces in the rack, slots on a straight line."""
    return DominoSceneSpec(
        n_tiles=slots + SPARES,
        table=TableSpec(top_z=TABLE_Z + table_offset_m, center_xy=(0.35, 0.0), size_xy=(0.6, 0.6)),
        feeder=FeederSpec(origin_xy=(0.20, -0.15), yaw=0.0, pitch=0.040),
        line=TargetLine(start_xy=(0.30, 0.05), yaw=0.0, pitch=0.036),
        version="domino_station_world_v1",
    )


def limits() -> Limits:
    return Limits.from_dict(
        {
            "frame_id": "world",
            "workspace_min_m": [0.0, -0.4, 0.0],
            "workspace_max_m": [0.8, 0.4, 0.4],
            "max_speed_m_s": 0.1,
            "max_contact_distance_m": 0.3,
            "gripper_min_width_m": 0.0,
            "gripper_max_width_m": 0.07,
        }
    )


def make_physics(kind: str):
    if kind == "mujoco":
        from .domino_mujoco import MujocoBackend

        return MujocoBackend()
    if kind == "isaac":
        from .domino_isaac import IsaacBackend

        global _ISAAC
        if _ISAAC is None:  # one SimulationApp per process; fresh stage per episode
            _ISAAC = IsaacBackend(headless=True)
        return _ISAAC
    raise ValueError(kind)


_ISAAC = None


def physics_domino_world(
    seed: int,
    *,
    physics: str = "mujoco",
    slots: int = 3,
    fault: str | None = None,
    fault_index: int | None = None,
    table_offset_m: float = 0.008,
) -> dict:
    spec_true = world_spec(slots, table_offset_m=table_offset_m if fault == "table_offset" else 0.0)
    spec_cal = world_spec(slots)
    feeders = feeder_poses_for_seed(spec_true, seed)
    ids = spec_true.tile_ids()
    station = DominoPhysicsStation(
        make_physics(physics),
        spec_true,
        feeders,
        calibrated_table_z=TABLE_Z,
        piece_ids=ids,
        fault=fault,
        fault_index=slots - 1 if fault_index is None else fault_index,
        table_offset_m=table_offset_m,
    )
    size = [spec_cal.tile.height, spec_cal.tile.width, spec_cal.tile.thickness]
    slot_list = []
    for k in range(slots):
        p = spec_cal.slot_pose(k)
        slot_list.append(
            {
                "slot_id": f"s{k}",
                "order": k,
                "pose": {
                    "frame_id": "world",
                    "position_m": p["position"],
                    "quaternion_xyzw": p["quat_xyzw"],
                },
                "neighbours": [f"s{j}" for j in (k - 1, k + 1) if 0 <= j < slots],
            }
        )
    tb = spec_cal.table
    inputs = {
        "layout": {
            "kind": "domino_layout",
            "layout_id": f"line{slots}-physics",
            "frame_id": "world",
            "piece": {"class": "domino", "size_m": size},
            "slots": slot_list,
        },
        "feeder": {"pieces": ids},
        "gripper_id": "floating_parallel_gripper",
        "table_z": TABLE_Z,
        "grasp": {
            "T_object_TCP": {
                "frame_id": "object",
                "position_m": [0.0, 0.0, 0.004],
                "quaternion_xyzw": [1.0, 0.0, 0.0, 0.0],
            },
            "pregrasp_height_m": 0.05,
            "lift_height_m": 0.06,
            "open_width_m": 0.0195,
            "close_width_m": 0.0055,
            "uncertainty": {"position_m": 0.001, "rotation_rad": 0.02},
            "collision_geometry": {"type": "box", "size_m": size},
        },
        "tolerances": {"max_drift_m": 0.005},
        "place": {
            "geometry": {"size_m": size, "yaw_symmetry_deg": 180.0},
            "support": {
                "frame_id": "world",
                "point_m": [tb.center_xy[0], tb.center_xy[1], TABLE_Z],
                "normal": [0.0, 0.0, 1.0],
                "height_std_m": 0.001,
                "calibration_version": "domino-station-world-v1 (scene ground truth before any injected offset)",
                "region_min_m": [tb.center_xy[0] - 0.28, tb.center_xy[1] - 0.28],
                "region_max_m": [tb.center_xy[0] + 0.28, tb.center_xy[1] + 0.28],
                "observation_id": "table",
            },
            "retreat": {"distance_m": 0.06},
            "hand": {
                "max_open_width_m": spec_cal.gripper.max_opening,
                "tip_reach_m": spec_cal.gripper.pad_size[2] / 2,
            },
            "workspace": {"min_m": [0.0, -0.4, 0.0], "max_m": [0.8, 0.4, 0.4]},
            "timeout_s": 60.0,
            "tolerances_ref": "tasks/domino/config/line_v1.yaml",
        },
    }
    describe = {
        **station.describe(),
        "seed": seed,
        "slots": slots,
        "scene_config_sha256": spec_true.config_hash(),
    }
    describe.pop("capabilities")
    holder: dict = {}

    def independent() -> dict:
        if "result" not in holder:
            holder["result"] = independent_evaluation(station, spec_true, slots)
        return holder["result"]

    def interventions() -> int:
        """Pieces a person would have to pick up: lying anywhere at the end."""
        objs = station.physics.truth()["objects"]
        return sum(1 for o in objs.values() if tilt_deg(o["quaternion_xyzw"]) > 45.0)

    return {
        "backend": station,
        "limits": limits(),
        "inputs": inputs,
        "describe": describe,
        "task_success": lambda: independent()["layout"]["status"] == "succeeded",
        "independent": independent,
        "privileged_allowed": True,
        "interventions": interventions,
        "station": station,
    }


def independent_evaluation(
    station: DominoPhysicsStation, spec: DominoSceneSpec, slots: int, settle_s: float = 1.5
) -> dict:
    """Judge the episode from simulator truth only (the skill results are not read)."""
    station.settle(settle_s)
    samples = station.observer.samples
    t_end = station.sim_t
    th = DEFAULT_THRESHOLDS
    closes = [e for e in station.events if e["event"] == "close"]
    releases = [e for e in station.events if e["event"] == "release"]
    placements = []
    for r in releases:
        t_rel = r["sim_s"]
        before = [c for c in closes if c["sim_s"] <= t_rel]
        t_close = before[-1]["sim_s"] if before else 0.0
        # retreat done: the hand is 20 mm over its release height (or the next close)
        nxt = [c["sim_s"] for c in closes if c["sim_s"] > t_rel]
        t_next = nxt[0] if nxt else t_end
        t_ret = next(
            (t for t, z in station.hand_track if t > t_rel and z > r["hand_z"] + 0.02), t_next
        )
        t_eval = min(t_ret + th.window_s + 0.1, t_end)
        at_rel = _sample_at(samples, t_rel)
        tile = r["tile"]
        pos = at_rel["objects"][tile]["position"] if at_rel else None
        slot = (
            min(range(slots), key=lambda k: dist(spec.slot_pose(k)["position"], pos)) if pos else 0
        )
        at_close = _sample_at(samples, t_close) or at_rel
        neighbours = {}
        for tid, o in (at_close or {}).get("objects", {}).items():
            if tid == tile:
                continue
            if any(dist(o["position"], spec.slot_pose(k)["position"]) < 0.01 for k in range(slots)):
                neighbours[tid] = o
        events = {
            "release_t": t_rel,
            "retreat_done_t": t_ret,
            "eval_t": t_eval,
            "phases": [
                {"name": "carry_and_place", "t0": t_close, "t1": t_rel},
                {"name": "release_and_retreat", "t0": t_rel, "t1": t_ret},
                {"name": "settle", "t0": t_ret, "t1": t_eval},
            ],
        }
        res = evaluate_placement(spec, slot, tile, neighbours, samples, events, th)
        res["events"] = events
        res["release_index"] = r["index"]
        res["release_wall_s"] = r.get("wall_s")
        placements.append(res)
    layout = evaluate_slots(
        spec, samples, t_end - settle_s + 0.25, t_end, [spec.slot_pose(k) for k in range(slots)], th
    )
    return {
        "placements": placements,
        "layout": layout,
        "events": station.events,
        "sim_time_s": t_end,
        "samples": len(samples),
    }


def _sample_at(samples: list[dict], t: float) -> dict | None:
    best = None
    for s in samples:
        if s["stamp"] <= t + 1e-9:
            best = s
        else:
            break
    return best
