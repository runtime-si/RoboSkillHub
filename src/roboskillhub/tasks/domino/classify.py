"""Cause hypotheses for an observed domino placement fault.

The observed fault (``result["failure"]``, from the executor's phase checks or the
independent observer) is kept apart from what probably caused it. Each hypothesis
names its supporting evidence and a confidence (low / medium / high); a fault seen
during the retreat is *not* by itself proof the hand hit the tile.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import geometry as geo


def _h(cause: str, confidence: str, evidence: str) -> dict[str, str]:
    return {"cause": cause, "confidence": confidence, "evidence": evidence}


def _pre_release_offset(observations: list[dict], plan: dict, trace: list[dict]) -> float | None:
    """How far the held tile was observed from where T_tcp_object predicts it, at the
    last observation before the release (support check)."""
    tcp = next(
        (
            e["measured_tcp"]
            for e in reversed(trace)
            if e.get("phase") == "descent" and "measured_tcp" in e
        ),
        None,
    )
    obs = next(
        (
            o
            for o in reversed(observations)
            if any(
                e.get("phase") == "support_check" and e.get("observation") == o.get("sample_id")
                for e in trace
            )
        ),
        None,
    )
    if tcp is None or obs is None:
        return None
    rec = (obs.get("objects") or {}).get(plan["target_id"])
    if not isinstance(rec, dict) or not isinstance(rec.get("pose"), dict):
        return None
    pred = geo.object_from_tcp(tcp, plan["T_tcp_object"])
    return float(np.linalg.norm(np.subtract(rec["pose"]["position_m"], pred["position_m"])))


def classify(result: dict, observations: list[dict], plan: dict | None) -> list[dict[str, Any]]:
    failure = result.get("failure") or {}
    code, phase = failure.get("code"), failure.get("phase")
    trace = result.get("phase_trace") or []
    if result.get("status") == "succeeded" or not code or plan is None:
        return []
    th = plan["thresholds"]
    out: list[dict[str, Any]] = []
    support = next(
        (e for e in trace if e.get("phase") == "support_check" and "support_offset_m" in e), None
    )
    if code in ("support_mismatch", "descent_blocked"):
        if support is not None:
            out.append(
                _h(
                    "table_calibration",
                    "high",
                    f"observed support offset {support['support_offset_m'] * 1e3:.1f} mm",
                )
            )
        else:
            out.append(
                _h(
                    "table_calibration",
                    "medium",
                    f"{code} with no direct support observation; an obstacle under the tile is also possible",
                )
            )
        return out
    if code == "object_dropped":
        out.append(
            _h("slip", "medium", "tile left the hand between the pre-check and the preplace check")
        )
        out.append(_h("grasp_offset", "low", "a marginal grasp also drops tiles in transit"))
        return out
    if code == "placement_error":
        if phase in ("carry", "descent"):
            out.append(_h("tcp_tracking", "high", failure.get("detail", "")))
            return out
        off = _pre_release_offset(observations, plan, trace)
        if off is not None and off > th["position_tolerance_m"] / 2:
            out.append(
                _h(
                    "grasp_offset",
                    "high",
                    f"held tile observed {off * 1e3:.1f} mm from T_tcp_object's prediction before release",
                )
            )
        else:
            out.append(
                _h(
                    "release_drag",
                    "medium",
                    "TCP reached the release pose and the held tile matched T_tcp_object; it moved at or after the release",
                )
            )
            out.append(_h("slip", "low", "or slid after landing"))
        return out
    if code == "unstable_after_release":
        if (
            support is not None
            and abs(support["support_offset_m"]) > th["support_height_tolerance_m"] / 2
        ):
            out.append(
                _h(
                    "table_calibration",
                    "medium",
                    f"support observed {support['support_offset_m'] * 1e3:.1f} mm off",
                )
            )
        if phase == "release":
            out.append(
                _h("release_drag", "medium", "fell between the release and the end of the retreat")
            )
        else:
            out.append(
                _h(
                    "slip",
                    "low",
                    "stood through the retreat, then fell or drifted in the settle window",
                )
            )
            out.append(
                _h(
                    "retreat_contact",
                    "low",
                    "a late topple can also follow a graze during the retreat",
                )
            )
        return out
    if code == "neighbor_disturbed":
        nid = failure.get("neighbor_id")
        if phase == "retreat":
            out.append(
                _h(
                    "retreat_contact",
                    "medium",
                    f"neighbour {nid} undisturbed after the release, disturbed after the retreat",
                )
            )
        elif phase == "release":
            out.append(
                _h("release_drag", "medium", f"neighbour {nid} already moved after the release")
            )
        else:
            out.append(
                _h(
                    "unknown",
                    "low",
                    f"neighbour {nid} moved; no observation separates release from retreat",
                )
            )
        return out
    if code == "release_not_confirmed":
        out.append(_h("gripper_release_fault", "medium", failure.get("detail", "")))
        return out
    if code == "retreat_blocked":
        out.append(_h("retreat_contact", "medium", failure.get("detail", "")))
        return out
    out.append(_h("unknown", "low", failure.get("detail", "")))
    return out
