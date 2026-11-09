"""Domino resting state from a 6D pose: long axis, thickness axis, bottom face and
lying / standing / on_side / uncertain.

The height-band classifier of ``reading-domino-pips-and-orientation`` (``_state``:
flat / upright / fallen from the top's height over the table) is reused as an
independent cross-check when the table height is known; a pose and a height that
disagree give ``uncertain``, never a confident state.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import _hub
from .objects import ObjectSpec, box_corners

#: Largest tilt of the vertical object axis from the table normal for a resting
#: state [deg]. pose6d's overhead-with-support p90 tilt is 2.4-3.2 deg on a domino
#: (its SKILL.md table); 10 deg leaves room for a real mask and still rejects a
#: tile leaning on a neighbour (~25+ deg for a 48 mm tile on a 7.5 mm one).
TILT_TOL_DEG = 10.0
#: Lowest corner further than this over the table: not resting on it [m] (pose6d's
#: overhead residual is ~1 mm with the support plane, 2.5 mm without).
REST_GAP_M = 0.004

_AXES = ("x", "y", "z")
_STATE_OF_AXIS = {0: "standing", 1: "on_side", 2: "lying"}
_READER_STATE = {"flat": "lying", "upright": "standing", "fallen": "on_side"}


def domino_state(
    T_world_obj: np.ndarray,
    spec: ObjectSpec,
    up: np.ndarray | None = None,
    table_z: float | None = None,
    tilt_tol_deg: float = TILT_TOL_DEG,
    rest_gap_m: float = REST_GAP_M,
    pip_face_known: bool = False,
) -> dict[str, Any]:
    """``state`` is ``lying`` (thickness axis vertical), ``standing`` (long axis
    vertical), ``on_side`` (width axis vertical, the tile fallen onto its long edge)
    or ``uncertain`` with ``reason``. ``bottom_face`` names the object face whose
    outward normal points down (``-z`` etc.); for a box that face and its opposite
    are geometrically the same, so ``bottom_face_resolved`` is True only for z faces
    when the pip face is known (``pip_face_known``)."""
    up = (
        np.array([0.0, 0.0, 1.0])
        if up is None
        else np.asarray(up, dtype=float) / np.linalg.norm(up)
    )
    R = np.asarray(T_world_obj, dtype=float)[:3, :3]
    cos = R.T @ up  # each object axis' alignment with up
    k = int(np.argmax(np.abs(cos)))
    tilt = math.degrees(math.acos(min(1.0, abs(float(cos[k])))))
    down_sign = -1 if cos[k] > 0 else 1  # the face pointing down is -axis when +axis points up
    out: dict[str, Any] = {
        "long_axis": R[:, 0].tolist(),
        "width_axis": R[:, 1].tolist(),
        "thickness_axis": R[:, 2].tolist(),
        "vertical_axis": _AXES[k],
        "tilt_deg": round(tilt, 3),
        "bottom_face": ("-" if down_sign < 0 else "+") + _AXES[k],
        "bottom_face_resolved": bool(k == 2 and pip_face_known),
        "up_end": None,
        "state": "uncertain",
        "reason": "",
        "height_check": None,
    }
    if k == 0:
        out["up_end"] = "+x" if cos[0] > 0 else "-x"
    if tilt > tilt_tol_deg:
        out["reason"] = (
            f"tilted {tilt:.1f} deg from the nearest resting state, over {tilt_tol_deg} deg"
        )
        return out
    state = _STATE_OF_AXIS[k]
    if table_z is not None:
        corners = box_corners(spec) @ R.T + np.asarray(T_world_obj)[:3, 3]
        heights = corners @ up
        gap = float(heights.min() - table_z)
        top = float(heights.max() - table_z)
        reader = _hub.load("domino_reader")._state(top, tuple(spec.size_m))
        out["height_check"] = {
            "bottom_gap_m": round(gap, 5),
            "top_height_m": round(top, 5),
            "reader_state": reader,
            "source": "reading-domino-pips-and-orientation._state",
        }
        if abs(gap) > rest_gap_m:
            out["reason"] = f"lowest corner {gap * 1e3:.1f} mm from the table: not resting on it"
            return out
        if _READER_STATE.get(reader) != state:
            out["reason"] = f"pose says {state}, the top height ({top * 1e3:.1f} mm) says {reader}"
            return out
    out["state"] = state
    return out
