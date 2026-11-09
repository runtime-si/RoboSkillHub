"""Cause hypotheses for an observed domino placement fault.

The observed fault (``result["failure"]``, from the executor's phase checks or the
independent observer) is kept apart from what probably caused it. Each hypothesis
names its supporting evidence and a confidence (low / medium / high); a fault seen
during the retreat is *not* by itself proof the hand hit the tile.
"""

from __future__ import annotations

from typing import Any


def _h(cause: str, confidence: str, evidence: str) -> dict[str, str]:
    return {"cause": cause, "confidence": confidence, "evidence": evidence}


def _pre_release_offset(trace: list[dict]) -> float | None:
    """How far the held tile was observed from where T_TCP_object predicts it, at the
    support check just before the release (logged by place_upright)."""
    e = next(
        (
            e
            for e in reversed(trace)
            if e.get("phase") == "support_check" and e.get("held_offset_m") is not None
        ),
        None,
    )
    return None if e is None else float(e["held_offset_m"])


def classify(result: dict, observations: list[dict], plan: dict | None) -> list[dict[str, Any]]:
    failure = result.get("failure") or {}
    code, phase = failure.get("code"), failure.get("phase")
    trace = result.get("trace") or []
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
    if code == "tcp_not_converged":
        out.append(_h("tcp_tracking", "high", failure.get("detail", "")))
        return out
    if code == "object_dropped":
        out.append(
            _h("slip", "medium", "tile left the hand between the pre-check and the preplace check")
        )
        out.append(_h("grasp_offset", "low", "a marginal grasp also drops tiles in transit"))
        return out
    if code == "placement_error":
        off = _pre_release_offset(trace)
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
    if code in ("release_failed", "not_resting_on_support"):
        out.append(_h("gripper_release_fault", "medium", failure.get("detail", "")))
        return out
    if code == "retreat_blocked":
        out.append(_h("retreat_contact", "medium", failure.get("detail", "")))
        return out
    out.append(_h("unknown", "low", failure.get("detail", "")))
    return out
