"""The two minimal policy graphs G over one fixed K, and their parameter variants.

- ``domino_line``: 3-5 dominoes from an upright feeder onto a straight layout, placed
  and verified by ``domino.place_upright`` (Agent 2).
- ``lego_press``: single-arm vertical press of known bricks onto a fixed baseplate, with
  ``lego.align_connection``, ``lego.guarded_seat``, ``lego.release_and_retreat`` and
  ``lego.verify_connection`` (Agent 3). The seat needs contact capabilities no backend has.

Each builder returns an ``rsh-graph/1`` dict. ``recovery=False`` is the fixed
script (any non-success aborts); ``recovery=True`` re-observes, re-grasps and
retries a slot with a fresh piece within visit budgets. Layouts, mating targets,
feeder contents and grasp offsets are graph *inputs*, never baked into G.
"""

from __future__ import annotations

import itertools
from typing import Any

from roboskillhub.runtime.contracts import CONTRACT_VERSION

TERMINALS = {"done": "succeeded", "abort": "failed", "unsure": "unverified"}
CORE_VERSION = "0.1.0"
PERCEPTION_VERSION = "0.1.0"
PLACE = ("domino.place_upright", "0.1.0")
LEGO_ALIGN = ("lego.align_connection", "0.1.0")
LEGO_SEAT = ("lego.guarded_seat", "0.1.0")
LEGO_VERIFY = ("lego.verify_connection", "0.1.0")
LEGO_RELEASE = ("lego.release_and_retreat", "0.1.0")


def _edges(succeeded: str, failed: str = "abort", **extra: str) -> dict:
    return {
        "succeeded": succeeded,
        "failed": failed,
        "unverified": "unsure",
        "rejected": "abort",
        **extra,
    }


def _pick_chain(
    *,
    recovery: bool,
    attempts: int,
    speed_m_s: float,
    targets: int,
    grasps: str,
    expect_state: str | None,
    classify: dict | None = None,
    grasp: str = "$input.grasp",
) -> dict[str, dict]:
    """observe piece -> (classify) -> grasp candidate -> pick, shared by both graphs.

    ``grasps="perception"`` uses Agent 1's resting-state check and grasp proposals;
    ``"configured"`` derives the grasp from a configured ``T_object_TCP`` instead.
    ``max_visits`` counts visits over the whole episode, so it scales with targets.
    """
    visits = 3 * (attempts + 1) * targets
    regrasp = {"failed:grasp_missed": "observe_piece", "failed:grasp_lost": "observe_piece"}
    # A target the station refuses before moving costs one try of the next candidate.
    fallback = {"failed:unreachable": "choose", "rejected:unreachable": "choose"}
    chain = {
        "observe_piece": {
            "skill": "core.observe",
            "params": {"object_ids": ["$last.select.outputs.piece"], "require_orientation": True},
            "timeout_s": 5,
            "attempts": 2 if recovery else 1,
            "retry_on": ["unverified"] if recovery else [],
            "max_visits": visits,
            "on": _edges("classify" if grasps == "perception" else "candidate"),
        },
        "classify": {
            "skill": "perception.classify_object_state",
            "params": {
                "observation": "$last.observe_piece.outputs.first",
                "table_z": "$input.table_z",
                **(classify or {}),
            },
            "post": [{"path": "$last.classify.outputs.state_label", "equals": expect_state}]
            if expect_state
            else [],
            "on": _edges("candidate", **({"failed": "select"} if recovery else {})),
        },
        "candidate": {
            "skill": "core.grasp_candidate_from_pose",
            "params": {
                "observation": "$last.observe_piece.outputs.first",
                "T_object_TCP": f"{grasp}.T_object_TCP",
                "pregrasp_height_m": "$input.grasp.pregrasp_height_m",
                "lift_height_m": "$input.grasp.lift_height_m",
                "open_width_m": f"{grasp}.open_width_m",
                "close_width_m": "$input.grasp.close_width_m",
            },
            "on": _edges("pick"),
        },
        "pick": {
            "skill": "core.pick",
            "params": {
                "candidate": "$last.candidate.outputs.candidate"
                if grasps == "configured"
                else "$last.choose.outputs.candidate",
                "gripper_id": "$input.gripper_id",
                "speed_m_s": speed_m_s,
                "uncertainty": "$input.grasp.uncertainty",
                "collision_geometry": "$input.grasp.collision_geometry",
            },
            "timeout_s": 30,
            "max_visits": visits,
            "on": _edges(
                "transport",
                **(regrasp if recovery else {}),
                **(fallback if recovery and grasps == "perception" else {}),
            ),
        },
    }
    if grasps == "perception":
        chain["candidate"] = {
            "skill": "perception.propose_grasps",
            "params": {
                "observation": "$last.observe_piece.outputs.first",
                "table_z": "$input.table_z",
                "max_candidates": 4,
            },
            "assign": {"tried": 0},
            "on": _edges("choose", **({"failed": "select"} if recovery else {})),
        }
        chain["choose"] = {
            "skill": "core.select_candidate",
            "params": {
                "candidates": "$last.candidate.outputs.candidates",
                "tried": "$state.tried",
                "max_tries": attempts + 1 if recovery else 1,
            },
            "max_visits": visits,
            "assign": {"tried": "$last.choose.outputs.tried_after"},
            "on": _edges("pick", **({"failed": "select"} if recovery else {})),
        }
    else:
        del chain["classify"]
    return chain


def _grasp_skills(grasps: str) -> dict[str, str]:
    if grasps == "perception":
        return {
            "perception.classify_object_state": PERCEPTION_VERSION,
            "perception.propose_grasps": PERCEPTION_VERSION,
            "core.select_candidate": CORE_VERSION,
        }
    if grasps == "configured":
        return {"core.grasp_candidate_from_pose": CORE_VERSION}
    raise ValueError("grasps must be 'perception' or 'configured'")


def _place_node(*, with_neighbours: bool, recovery: bool, speed_m_s: float) -> dict:
    params = {
        "frame_id": "$input.layout.frame_id",
        "target_id": "$last.select.outputs.piece",
        "target_object_pose": "$last.select.outputs.target.pose",
        "held": "$last.pick.outputs.held",
        "geometry": "$input.place.geometry",
        "support": "$input.place.support",
        "retreat": "$input.place.retreat",
        "hand": "$input.place.hand",
        "workspace": "$input.place.workspace",
        "timeout_s": "$input.place.timeout_s",
        "tolerances_ref": "$input.place.tolerances_ref",
        "speed_m_s": speed_m_s,
    }
    if with_neighbours:
        params["neighbors"] = "$last.observe_neighbours.outputs.observations"
    retry_slot = {
        f"failed:{f}": "select"
        for f in (
            "domino.toppled_after_release",
            "domino.pose_out_of_tolerance",
            "grasp_lost",
            "held_object_slipped",
            "motion_not_converged",
        )
    }
    return {
        "skill": PLACE[0],
        "params": params,
        "timeout_s": 90,
        "assign": {
            "done": "$last.select.outputs.done_after",
            "placed": "$last.select.outputs.placed_after",
        },
        "on": _edges("select", **(retry_slot if recovery else {})),
    }


def domino_line(
    *,
    recovery: bool = True,
    attempts: int = 2,
    speed_m_s: float = 0.03,
    slots: int = 5,
    grasps: str = "perception",
) -> dict[str, Any]:
    """Upright feeder -> straight line; ``domino.place_upright`` (Agent 2) places and verifies."""
    nodes = {
        "select": {
            "skill": "core.select_next_target",
            "params": {
                "targets": "$input.layout.slots",
                "done": "$state.done",
                "pieces": "$input.feeder.pieces",
                "used": "$state.used",
                "placed": "$state.placed",
            },
            "max_visits": (attempts + 1) * slots + 1,
            "branch": [{"path": "$last.select.outputs.complete", "equals": True, "next": "done"}],
            "assign": {"used": "$last.select.outputs.used_after"},
            "on": _edges("observe_piece"),
        },
        **_pick_chain(
            recovery=recovery,
            attempts=attempts,
            speed_m_s=speed_m_s,
            targets=slots,
            grasps=grasps,
            expect_state="standing",
        ),
        "observe_neighbours": {
            "skill": "core.observe",
            "params": {"object_ids": "$last.select.outputs.placed", "require_orientation": True},
            "timeout_s": 5,
            "on": _edges("place"),
        },
        "place_first": _place_node(with_neighbours=False, recovery=recovery, speed_m_s=speed_m_s),
        "place": _place_node(with_neighbours=True, recovery=recovery, speed_m_s=speed_m_s),
    }
    nodes["pick"]["on"]["succeeded"] = "place_first"
    nodes["pick"]["branch"] = [
        {"path": "$last.select.outputs.has_placed", "equals": True, "next": "observe_neighbours"}
    ]
    return {
        "graph_id": f"domino-line-{'replan' if recovery else 'scripted'}",
        "version": "0.4.0",
        "contract_version": CONTRACT_VERSION,
        "params": {
            "recovery": recovery,
            "attempts": attempts,
            "speed_m_s": speed_m_s,
            "grasps": grasps,
        },
        "skills": {
            "core.select_next_target": CORE_VERSION,
            "core.observe": CORE_VERSION,
            **_grasp_skills(grasps),
            "core.pick": CORE_VERSION,
            PLACE[0]: PLACE[1],
        },
        "budgets": {
            "max_steps": 12 * slots * (attempts + 1),
            "max_total_attempts": 10 * slots * (attempts + 1),
            "timeout_s": 120.0 * slots,
        },
        "state": {"done": [], "used": [], "placed": [], "tried": 0},
        "start": "select",
        "nodes": nodes,
        "terminals": TERMINALS,
    }


def lego_press(
    *,
    recovery: bool = True,
    attempts: int = 2,
    clearance_m: float = 0.01,
    speed_m_s: float = 0.02,
    bricks: int = 3,
    grasps: str = "perception",
) -> dict[str, Any]:
    nodes = {
        "select": {
            "skill": "core.select_next_target",
            "params": {
                "targets": "$input.targets",
                "done": "$state.done",
                "pieces": "$input.feeder.pieces",
                "used": "$state.used",
                "key": "target_id",
            },
            "max_visits": (attempts + 1) * bricks + 1,
            "branch": [{"path": "$last.select.outputs.complete", "equals": True, "next": "done"}],
            "assign": {"used": "$last.select.outputs.used_after"},
            "on": _edges("bind"),
        },
        # The target under the id of the piece actually carried, and what is built so far.
        "bind": {
            "skill": "core.bind_target",
            "params": {
                "target": "$last.select.outputs.target",
                "piece": "$last.select.outputs.piece",
                "placed": "$state.placed",
                "prebuilt": "$input.prebuilt",
            },
            "max_visits": (attempts + 1) * bricks,
            "on": _edges("observe_piece"),
        },
        **_pick_chain(
            recovery=recovery,
            attempts=attempts,
            speed_m_s=speed_m_s,
            targets=bricks,
            grasps=grasps,
            expect_state=None,
            classify={"lattice": "$input.lattice", "expected_state": "loose"},
            # The world picks each target's grasp (closing axis clear of what is built before it).
            grasp="$last.select.outputs.target.grasp",
        ),
        "transport": {
            "skill": "core.transport_held",
            "params": {
                "held": "$last.pick.outputs.held",
                "goal": "$last.select.outputs.target.mating.mating_pose",
                "approach_height_m": clearance_m,
                "descend": False,
                "speed_m_s": speed_m_s,
                "clear_above_z_m": "$input.travel_clear_z_m",
            },
            "timeout_s": 30,
            "on": _edges(
                "confirm", **({"failed:held_object_slipped": "select"} if recovery else {})
            ),
        },
        # Holding evidence from the pick is older than align's freshness limit after the carry.
        "confirm": {
            "skill": "core.confirm_hold",
            "params": {"held": "$last.transport.outputs.held"},
            "timeout_s": 5,
            "max_visits": 2 * (attempts + 1) * bricks,
            "on": _edges(
                "align",
                **(
                    {"failed:grasp_lost": "select", "failed:held_object_slipped": "select"}
                    if recovery
                    else {}
                ),
            ),
        },
        "align": {
            "skill": LEGO_ALIGN[0],
            "params": {
                "connection_id": "$last.select.outputs.target.connection_id",
                "target": "$last.bind.outputs.mating",
                "build": "$last.bind.outputs.build",
                "held": "$last.confirm.outputs.held",
                "limits": "$input.align_limits",
                **(
                    {"close_axis": "$last.select.outputs.target.grasp.close_axis"}
                    if grasps == "configured"
                    else {}
                ),
            },
            "timeout_s": 20,
            # A retry needs fresh holding evidence, so it goes back through confirm.
            "on": _edges(
                "seat",
                **(
                    {"failed:held_object_slipped": "select", "unverified": "confirm"}
                    if recovery
                    else {}
                ),
            ),
        },
        "seat": {
            "skill": LEGO_SEAT[0],
            "params": {
                "connection_id": "$last.select.outputs.target.connection_id",
                "alignment": "$last.align.outputs.alignment",
                "press_limits": "$input.press_limits",
            },
            "timeout_s": 20,
            "on": _edges("release", **{"unverified:lego.seat_depth_without_threshold": "release"}),
        },
        "release": {
            "skill": LEGO_RELEASE[0],
            "params": {
                "connection_id": "$last.select.outputs.target.connection_id",
                "object_id": "$last.select.outputs.piece",
                "press": "$last.seat.outputs.press",
                "open_width_m": "$last.select.outputs.target.grasp.release_open_width_m",
                "retreat_height_m": "$input.release.retreat_height_m",
            },
            "timeout_s": 20,
            "on": _edges("verify"),
        },
        "verify": {
            "skill": LEGO_VERIFY[0],
            "params": {
                "connection_id": "$last.select.outputs.target.connection_id",
                "target": "$last.bind.outputs.mating",
                "build": "$last.bind.outputs.build",
                "executor_id": "$last.seat.outputs.executor_id",
                "release": "$last.release.outputs.release",
                "pre_release_observation": "$last.release.outputs.pre_release_observation",
                "window": "$input.verify_window",
            },
            "timeout_s": 10,
            "assign": {
                "done": "$last.select.outputs.done_after",
                "placed": "$last.bind.outputs.placed_after",
            },
            "on": _edges("select"),
        },
    }
    return {
        "graph_id": f"lego-press-{'replan' if recovery else 'scripted'}",
        "version": "0.4.0",
        "contract_version": CONTRACT_VERSION,
        "params": {
            "recovery": recovery,
            "attempts": attempts,
            "clearance_m": clearance_m,
            "speed_m_s": speed_m_s,
            "grasps": grasps,
        },
        "skills": {
            "core.select_next_target": CORE_VERSION,
            "core.observe": CORE_VERSION,
            **_grasp_skills(grasps),
            "core.pick": CORE_VERSION,
            "core.transport_held": CORE_VERSION,
            "core.bind_target": CORE_VERSION,
            "core.confirm_hold": CORE_VERSION,
            LEGO_ALIGN[0]: LEGO_ALIGN[1],
            LEGO_SEAT[0]: LEGO_SEAT[1],
            LEGO_RELEASE[0]: LEGO_RELEASE[1],
            LEGO_VERIFY[0]: LEGO_VERIFY[1],
        },
        "budgets": {
            "max_steps": 12 * bricks * (attempts + 1),
            "max_total_attempts": 10 * bricks * (attempts + 1),
            "timeout_s": 180.0 * bricks,
        },
        "state": {"done": [], "used": [], "placed": [], "tried": 0},
        "start": "select",
        "nodes": nodes,
        "terminals": TERMINALS,
    }


def domino_candidates(slots: int = 5) -> list[dict]:
    """Candidate G for simulation-evaluated search: same K, varied structure and parameters."""
    grid = itertools.product((False, True), (1, 2), (0.02, 0.05))
    return [
        domino_line(recovery=r, attempts=a, speed_m_s=v, slots=slots)
        for r, a, v in grid
        if r or a == 1
    ]


def lego_candidates() -> list[dict]:
    grid = itertools.product((False, True), (1, 2), (0.005, 0.01), (0.01, 0.02))
    return [
        lego_press(recovery=r, attempts=a, clearance_m=c, speed_m_s=v)
        for r, a, c, v in grid
        if r or a == 1
    ]
