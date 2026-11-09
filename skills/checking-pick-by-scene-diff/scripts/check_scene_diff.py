# Adapted from RoboRSI (https://github.com/nssmd/RoboRSI),
# roborsi/embodied/skills/base/visual_diff/robotwin/policy.py at
# 90c8569705bf4004a4f79e34e650c87ab7f075d4, Apache-2.0.
# Changes: BEFORE is an observation the graph captured before the step and
# AFTER a fresh robot.get_observation (upstream cached a snapshot anchor in the
# process and rendered the simulator camera); the panel goes to the hub's
# vlm.query as an RGB array (upstream wrote a JPEG and called its own VLM
# client), with the system and user text joined into one prompt because the
# tool has no system prompt; the two answer lines become graph exits; an
# optional source region adds a pixel-change share that must agree with the
# VLM's verdict, and disagreement routes uncertain; a missing answer line
# routes no_verdict (upstream read it as NO).
"""Check a pick (or any intended change) from a BEFORE | AFTER | DIFF panel judged by a VLM."""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

DIFF_THRESHOLD = 30
"""Summed |RGB| difference (0-765) above which a pixel counts as changed; upstream's
``n_changed_pixels`` threshold."""

MIN_SOURCE_CHANGE = 0.15
"""Share of source-region pixels that must change for the region to count as changed.
Live check (SKILL.md): 0.43-0.77 where an object left or arrived, 0.0-0.021 where nothing
did (real overhead camera with both arms moving elsewhere: 0.021)."""

VACATED_MAX_RATIO = 0.85
"""Region contrast after / before at or under which the region counts as vacated. Live
check: 0.22 (real plate removed), 0.77 (sim block picked, another dropped beside its
spot), against 1.00 static, 1.65 hand hovering over the object, 2.17 object appeared.
A veto for regions that got busier, not a vacancy test: in the twin pick runs a cube
nudged out of part of a small box scored 0.63-0.68, like a pick (SKILL.md)."""

FILLED_MIN_RATIO = 1.5
"""Region contrast after / before at or over which the region counts as filled (a place).
Live check: 3.74 (sim block put into a box), against 1.00 static."""

REGION_EXPECT = ("vacated", "filled", "changed")

AS_EXPECTED = "changed_as_expected"
OTHERWISE = "changed_otherwise"
UNCHANGED = "unchanged"
UNCERTAIN = "uncertain"
NO_VERDICT = "no_verdict"


class Output(TypedDict):
    route: str
    changed: bool
    matches_expectation: bool
    reason: str
    camera: str
    changed_pixels: int
    changed_fraction: float
    source_changed_fraction: float
    source_contrast_ratio: float
    panel: Any
    raw: str


def prompt_text(expected_change: str) -> str:
    """RoboRSI's system text, then its user text, as one prompt."""
    return (
        "You see a 3-panel image: BEFORE | AFTER | DIFF heatmap (hot = "
        "changed pixels). Compare BEFORE and AFTER. Report on TWO things:\n"
        "  CHANGED: did something visibly change at all? (yes/no)\n"
        f"  MATCH: did the EXPECTED change happen? Expected: {expected_change!r}\n"
        "Reply on TWO lines exactly:\n"
        "  CHANGED: YES or NO\n"
        "  MATCH: YES or NO\n"
        "Then a one-line reason.\n\n"
        "Did the expected change happen? Inspect the 3-panel image."
    )


def diff_map(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Per-pixel summed absolute RGB difference, int32 [H, W] in 0..765."""
    return np.abs(before.astype(np.int16) - after.astype(np.int16)).sum(axis=-1).astype(np.int32)


def build_panel(before: np.ndarray, after: np.ndarray, diff: np.ndarray) -> np.ndarray:
    """RGB panel BEFORE | AFTER | DIFF (hot), labelled as upstream; the heatmap is
    normalised by the frame's largest difference, as upstream did."""
    norm = np.clip(diff * (255.0 / max(int(diff.max()), 1)), 0, 255).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)
    panel = np.ascontiguousarray(np.concatenate([before, after, heat], axis=1))
    w = before.shape[1]
    for i, text in enumerate(("BEFORE", "AFTER", "DIFF (hot)")):
        cv2.putText(panel, text, (i * w + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return panel


def parse_reply(text: str) -> tuple[bool | None, bool | None, str]:
    """(changed, matches, reason) from the two answer lines; None for a missing line."""
    m_chg = re.search(r"CHANGED\s*:\s*\**\s*(YES|NO)", text, re.IGNORECASE)
    m_match = re.search(r"MATCH\s*:\s*\**\s*(YES|NO)", text, re.IGNORECASE)
    changed = None if m_chg is None else m_chg.group(1).upper() == "YES"
    matches = None if m_match is None else m_match.group(1).upper() == "YES"
    rest = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not re.match(r"\W*(CHANGED|MATCH)\s*:", ln.strip(), re.IGNORECASE)
    ]
    return changed, matches, " ".join(rest)[:240]


def _frames(observation: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    cams = (observation or {}).get("cameras") or []
    if isinstance(cams, dict):
        cams = [{"name": k, **v} if "name" not in v else v for k, v in cams.items()]
    return {str(c.get("name") or f"camera{i}"): c for i, c in enumerate(cams)}


def _rgb(frame: dict[str, Any] | None) -> np.ndarray | None:
    rgb = (frame or {}).get("rgb")
    if rgb is None:
        return None
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _is_wrist(name: str) -> bool:
    return any(k in name.lower() for k in ("wrist", "eye_in_hand", "hand"))


def source_mask(shape: tuple[int, int], box_px: Any = None, mask: Any = None) -> np.ndarray | None:
    """Boolean [H, W] mask of the source region from a pixel box [x0, y0, x1, y1] or a mask."""
    h, w = shape
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        if m.shape != (h, w):
            raise ValueError(f"source_mask shape {m.shape} != frame {(h, w)}")
        return m
    if box_px is None:
        return None
    x0, y0, x1, y1 = (int(round(float(v))) for v in box_px)
    x0, x1 = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
    y0, y1 = sorted((max(0, min(h, y0)), max(0, min(h, y1))))
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def region_contrast(rgb: np.ndarray, region: np.ndarray) -> float:
    """Mean RGB distance of the region's pixels from the median colour of a ring around it.

    High while an object sits in the region against its support, low once it is bare. The
    ring is half the region's equivalent side wide (at least 4 px).
    """
    m = region.astype(np.uint8)
    r = max(4, int(0.5 * np.sqrt(float(m.sum()))))
    ring = cv2.dilate(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool) & ~region
    if not ring.any():
        return 0.0
    img = rgb.astype(np.float64)
    med = np.median(img[ring], axis=0)
    return float(np.linalg.norm(img[region] - med, axis=1).mean())


def region_verdict(
    expect: str, changed_share: float, ratio: float, min_change: float
) -> tuple[bool, str]:
    """(agrees with the expected change, why) for the region numbers."""
    if changed_share < min_change:
        return False, f"only {changed_share:.1%} of the region changed (< {min_change:.0%})"
    if expect == "vacated" and ratio > VACATED_MAX_RATIO:
        return False, (
            f"the region did not empty: contrast ratio {ratio:.2f} > {VACATED_MAX_RATIO} "
            "(something still or newly there, e.g. the hand over it)"
        )
    if expect == "filled" and ratio < FILLED_MIN_RATIO:
        return (
            False,
            f"nothing arrived in the region: contrast ratio {ratio:.2f} < {FILLED_MIN_RATIO}",
        )
    return True, f"{changed_share:.1%} of the region changed, contrast ratio {ratio:.2f}"


def _out(route: str, **fields: Any) -> Output:
    out: Output = {
        "route": route,
        "changed": False,
        "matches_expectation": False,
        "reason": "",
        "camera": "",
        "changed_pixels": 0,
        "changed_fraction": 0.0,
        "source_changed_fraction": -1.0,
        "source_contrast_ratio": -1.0,
        "panel": None,
        "raw": "",
    }
    out.update(fields)  # type: ignore[typeddict-item]
    return out


def run(
    ctx: NodeContext,
    before: dict[str, Any],
    expected_change: str,
    now: dict[str, Any] | None = None,
    camera: str | None = None,
    source_box_px: list[float] | None = None,
    source_mask_px: Any = None,
    region_expect: str = "vacated",
    diff_threshold: int = DIFF_THRESHOLD,
    min_source_change: float = MIN_SOURCE_CHANGE,
    model: str | None = None,
) -> Output:
    """Did ``expected_change`` happen between ``before`` and now? One VLM look at BEFORE | AFTER | DIFF.

    ``before`` is a ``robot.get_observation`` taken before the step; ``now`` defaults to a
    fresh one. ``camera`` picks the frame by name (default: the first camera with RGB in
    both, external before wrist). ``source_box_px`` [x0, y0, x1, y1] or ``source_mask_px``
    marks the region the step should change; ``region_expect`` says how (``vacated`` for a
    pick, ``filled`` for a place, ``changed`` for either), and the pixels there must agree
    with the VLM.
    """
    if region_expect not in REGION_EXPECT:
        raise ValueError(f"region_expect {region_expect!r} not in {REGION_EXPECT}")
    if now is None:
        now = ctx.tool("robot.get_observation")
    bf, nf = _frames(before), _frames(now)
    if camera:
        names = [str(camera)]
    else:
        names = sorted((n for n in nf if n in bf), key=_is_wrist)
    name, b, a = "", None, None
    for n in names:
        b, a = _rgb(bf.get(n)), _rgb(nf.get(n))
        if b is not None and a is not None:
            name = n
            break
    if not name:
        return _out(NO_VERDICT, reason=f"no camera with RGB in both observations (asked {names})")
    if b.shape != a.shape:
        return _out(NO_VERDICT, camera=name, reason=f"frame sizes differ: {b.shape} vs {a.shape}")

    diff = diff_map(b, a)
    changed_px = diff > int(diff_threshold)
    region = source_mask(diff.shape, source_box_px, source_mask_px)
    region_ok, region_why = None, ""
    stats: dict[str, Any] = {
        "camera": name,
        "changed_pixels": int(changed_px.sum()),
        "changed_fraction": round(float(changed_px.mean()), 5),
    }
    if region is not None and region.any():
        share = float(changed_px[region].mean())
        c_before = region_contrast(b, region)
        ratio = region_contrast(a, region) / max(c_before, 1e-6)
        region_ok, region_why = region_verdict(
            region_expect, share, ratio, float(min_source_change)
        )
        stats["source_changed_fraction"] = round(share, 4)
        stats["source_contrast_ratio"] = round(ratio, 3)
    panel = build_panel(b, a, diff)
    stats["panel"] = panel

    kwargs: dict[str, Any] = {"prompt": prompt_text(expected_change.strip()), "images": [panel]}
    if model:
        kwargs["model"] = model
    try:
        raw = str(ctx.tool("vlm.query", **kwargs).get("text") or "").strip()
    except Exception as exc:  # noqa: BLE001  (auth / network / unserved tool)
        logger.warning("check_scene_diff: vlm.query failed: %s", exc)
        return _out(NO_VERDICT, reason=f"vlm.query failed: {exc}", **stats)

    changed, matches, reason = parse_reply(raw)
    if changed is None or matches is None:
        return _out(NO_VERDICT, reason="reply lacks a CHANGED or MATCH line", raw=raw, **stats)

    fields: dict[str, Any] = dict(
        changed=changed, matches_expectation=matches, reason=reason, raw=raw, **stats
    )
    if matches and not changed:
        route = UNCERTAIN
        fields["reason"] = f"VLM said MATCH yes but CHANGED no; {reason}"
    elif matches:
        route = AS_EXPECTED
        if region_ok is False:
            route = UNCERTAIN
            fields["reason"] = f"VLM said MATCH yes but {region_why}; {reason}"
    elif changed:
        route = OTHERWISE
    else:
        route = UNCHANGED
        if region_ok:
            route = UNCERTAIN
            fields["reason"] = f"VLM said nothing changed but {region_why}; {reason}"
    logger.info(
        "check_scene_diff: %s (changed=%s match=%s region=%s) %s",
        route,
        changed,
        matches,
        region_why or "-",
        fields["reason"],
    )
    return _out(route, **fields)
