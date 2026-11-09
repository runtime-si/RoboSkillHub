"""Bind a held part to a placement target: the TCP poses Agent 2 (domino) and
Agent 3 (LEGO mating) consume, from one held-object convention.

Both use ``T_frame_tcp = T_frame_object_target @ S @ inv(T_TCP_object)`` over the
part's symmetric equivalents ``S``, ordered by how little the hand turns from its
current orientation. An invalid held state is rejected, never bound.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contract import result
from .frames import T_to_pose, inv_T, make_T, rotation_angle
from .held import HeldObject
from .lego import mating_target
from .objects import ObjectSpec, symmetry_rotations


def bind_target(
    held: HeldObject,
    T_frame_target: np.ndarray,
    frame_id: str,
    T_frame_tcp_now: np.ndarray | None = None,
    equivalents: list[np.ndarray] | None = None,
    approach_offset_m: float = 0.0,
    up: np.ndarray | None = None,
) -> dict[str, Any]:
    stage = "bind_target"
    if not held.valid:
        return result(
            "rejected",
            stage,
            failure_type="held_state_invalid",
            reason=held.invalid_reason,
            recovery=["reobserve the held part"],
            verification="none",
        )
    up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=float)
    eqs = equivalents or [T_frame_target @ make_T(S) for S in symmetry_rotations(held.spec)]
    T_obj_tcp = inv_T(held.T_tcp_object)
    opts = []
    for i, Te in enumerate(eqs):
        T = Te @ T_obj_tcp
        T[:3, 3] += approach_offset_m * up
        turn = (
            0.0 if T_frame_tcp_now is None else rotation_angle(T_frame_tcp_now[:3, :3], T[:3, :3])
        )
        opts.append((turn, i, T))
    opts.sort(key=lambda o: o[0])
    return result(
        "succeeded",
        stage,
        outputs={
            "tcp_targets": [
                {
                    "tcp_pose": T_to_pose(T, frame_id),
                    "equivalent_index": i,
                    "hand_turn_rad": round(turn, 4),
                }
                for turn, i, T in opts
            ],
            "held_source": held.source,
            "held_privileged": held.privileged,
            "T_TCP_object": held.to_dict()["T_TCP_object"],
        },
        reason="geometric binding; reachability is the consumer's planner's job",
        verification="geometry",
    )


def bind_domino_target(
    held: HeldObject, T_frame_slot: np.ndarray, frame_id: str, **kw
) -> dict[str, Any]:
    """Agent 2: a standing-domino slot pose (object frame as in :mod:`.objects`)."""
    if held.object_class != "domino":
        return result(
            "rejected",
            "bind_target",
            failure_type="wrong_class",
            reason=f"holding a {held.object_class}",
        )
    return bind_target(held, T_frame_slot, frame_id, **kw)


def bind_lego_mating(
    held: HeldObject,
    T_frame_plate: np.ndarray,
    plate_studs: tuple[int, int],
    cell: dict[str, int],
    frame_id: str,
    hover_m: float = 0.002,
    spec: ObjectSpec | None = None,
    **kw,
) -> dict[str, Any]:
    """Agent 3: a lattice cell on a calibrated plate; the TCP targets sit ``hover_m``
    over the seated pose (the press starts there; seating is Agent 3's)."""
    if held.object_class not in ("lego_brick", "lego_plate"):
        return result(
            "rejected",
            "bind_target",
            failure_type="wrong_class",
            reason=f"holding a {held.object_class}",
        )
    tgt = mating_target(T_frame_plate, spec or held.spec, plate_studs, cell)
    out = bind_target(
        held,
        tgt["T_world_brick"],
        frame_id,
        equivalents=tgt["equivalent_T_world_brick"],
        approach_offset_m=hover_m,
        up=T_frame_plate[:3, 2],
        **kw,
    )
    if out["status"] == "succeeded":
        out["outputs"]["seated_brick_pose"] = T_to_pose(tgt["T_world_brick"], frame_id)
        out["outputs"]["cell"] = dict(cell)
    return out
