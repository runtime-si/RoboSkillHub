# Adapted from Slip-Aware-Object-Manipulation-with-Parallel-Grippers
# (https://github.com/Gabrieleenx/Slip-Aware-Object-Manipulation-with-Parallel-Grippers, commit 639fe4a4bcf8),
# src/controlled_slippage.py, HoldMode.get_fn (the normal force a pad needs for a tangential force and a torque
# under the ellipsoidal limit surface, hold_margin 2, min_fn 2 N) and the friction / contact-radius values of
# get_friction_param_func (plastic pads: mu_c 0.355-0.365, r 5.8-7.7 mm). MIT License, Copyright (c) 2024
# Gabriel Arslan Waltersson; the license is LICENSE.slip-aware beside this skill.
# Changes: the source runs the model online on two force-torque sensors' readings; here the load is predicted
# before a pick from the part's mass and centre of mass (gravity force and its torque about each candidate grip's
# closing axis, shared by the two pads), candidate grips are ranked by the grip force they need, and the slipping
# controllers and the friction estimation are not ported (the YAM has no force sensors).
"""Rank candidate grips by the grip force they need to hold a part without it slipping or twisting in the jaws."""

from __future__ import annotations

import json
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

G = 9.81

MU_C = 0.36
"""Coulomb friction of pad against part: the source's plastic tests (0.355, 0.365)."""

CONTACT_RADIUS = 0.0067
"""Effective contact radius r [m] of the torsional limit (torque limit = mu_c x fn x r): the mean of the
source's plastic tests (7.7 and 5.8 mm). A 15 x 4 mm rubber patch averages 4.0 mm (SIM_GRIP.md)."""

HOLD_MARGIN = 2.0
MIN_FN = 2.0
"""The source's HoldMode(hold_margin=2, min_fn=2): the force asked for is twice the limit, at least 2 N."""

GRIP_FORCE = 9.5
"""Normal force each pad presses with [N] when the jaws close on a part: a YAM station twin's linear_4310 on a
30 mm cube (kp 600 N/m fingers, 15.9 mm short of the closed stop; SIM_GRIP.md). A real station's force depends on
its gripper torque cap; measure and pass it."""


class Output(TypedDict):
    route: str
    detail: str
    grip_index: int
    grip_position: dict[str, float]
    required_force: float
    available_force: float
    torque: float
    lever: float
    ranked: str


def _arr(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=float).reshape(-1)[:3]


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _closing_axis(c: dict[str, Any]) -> np.ndarray:
    """A candidate's closing axis: ``close_axis`` if given, else the grasp frame's y axis (the station's TCP
    convention: y is the closing axis) from its ``rotation`` (w, x, y, z), else world y."""
    if "close_axis" in c:
        return _arr(c["close_axis"])
    q = c.get("rotation")
    if isinstance(q, dict):
        w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
        return np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
    return np.array([0.0, 1.0, 0.0])


def pad_force_needed(tangential: float, torque: float, mu_c: float, radius: float) -> float:
    """The source's HoldMode limit for one pad: the normal force at which a tangential force and a torque about
    the pad normal sit on the ellipsoidal limit surface (no margin)."""
    r = max(radius, 1e-5)
    norm = math.sqrt(tangential**2 + (torque / r) ** 2) + 1e-5
    gamma_ft = (tangential + 1e-5) / norm
    gamma_tau = (torque / r + 1e-5) / norm
    f_lin = tangential / (mu_c * gamma_ft)
    f_tau = torque / (mu_c * r * gamma_tau)
    return max(f_lin, f_tau)


def run(
    ctx: NodeContext,
    center_of_mass: dict[str, float],
    candidates: Any,
    mass: float = 0.0,
    area: float = 0.0,
    thickness: float = 0.0,
    density: float = 0.0,
    mu_c: float = MU_C,
    contact_radius: float = CONTACT_RADIUS,
    hold_margin: float = HOLD_MARGIN,
    min_fn: float = MIN_FN,
    grip_force: float = GRIP_FORCE,
) -> Output:
    m = float(mass) if mass > 0 else float(area) * float(thickness) * float(density)
    out: Output = {
        "route": "no_grip_holds",
        "detail": "",
        "grip_index": -1,
        "grip_position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "required_force": 0.0,
        "available_force": float(grip_force),
        "torque": 0.0,
        "lever": 0.0,
        "ranked": "[]",
    }
    if isinstance(candidates, dict):
        candidates = candidates.get("poses", [])
    if m <= 0:
        raise ValueError("give mass, or area, thickness and density")
    if not candidates:
        out["detail"] = "no candidate grips"
        return out
    com = _arr(center_of_mass)
    weight = np.array([0.0, 0.0, -m * G])
    ranked = []
    for i, c in enumerate(candidates):
        p = _arr(c["position"])
        axis = _closing_axis(c)
        axis = axis / (float(np.linalg.norm(axis)) or 1.0)
        arm = com - p
        torque = abs(float(np.dot(np.cross(arm, weight), axis)))
        # The lever: the centre of mass's distance from the closing axis, horizontally.
        off = arm - axis * float(np.dot(arm, axis))
        lever = float(math.hypot(off[0], off[1]))
        # Two pads share the weight in their planes (the component along the closing axis is borne by the
        # squeeze, not friction) and the torque.
        tangential = float(np.linalg.norm(weight - axis * float(np.dot(weight, axis)))) / 2
        need = hold_margin * max(
            pad_force_needed(tangential, torque / 2, mu_c, contact_radius), min_fn
        )
        ranked.append(
            {
                "index": i,
                "position": _vec(p),
                "torque": torque,
                "lever": lever,
                "required_force": need,
                "holds": need <= float(grip_force),
            }
        )
    ranked.sort(key=lambda r: r["required_force"])
    out["ranked"] = json.dumps(ranked)
    best = ranked[0]
    out.update(
        {
            "grip_index": best["index"],
            "grip_position": best["position"],
            "required_force": best["required_force"],
            "torque": best["torque"],
            "lever": best["lever"],
        }
    )
    linear_only = hold_margin * max(pad_force_needed(m * G / 2, 0.0, mu_c, contact_radius), min_fn)
    summary = (
        f"best grip {best['index']}: {best['lever'] * 1e3:.1f} mm from the centre of mass, torque "
        f"{best['torque'] * 1e3:.1f} mN.m, needs {best['required_force']:.1f} N per pad "
        f"(x{hold_margin:g} margin) of {grip_force:.1f} N"
    )
    if best["holds"]:
        out["route"] = "grip_selected"
        out["detail"] = summary
    elif linear_only > float(grip_force):
        out["route"] = "needs_suction"
        out["detail"] = (
            f"even a grip through the centre of mass needs {linear_only:.1f} N per pad for the "
            f"{m * 1e3:.0f} g weight alone, over {grip_force:.1f} N; {summary}"
        )
    else:
        out["detail"] = f"no candidate holds the twist: {summary}; grip nearer the centre of mass"
    return out
