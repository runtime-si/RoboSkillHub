"""Ask a ladder of planners for a visual plan in a declared shape; accept only an answer that has it.

The request says what a valid answer is before anything is asked: which arms, how many
waypoints each, and where each path starts (the arm's tool centre as the camera sees it).
An answer is put in the hub's visual-plan schema (``tools/molmo-motion``) and checked against
that declaration and against what is measured: the image's bounds, the hand's own pixel, the
depth under the endpoint, the metric part's agreement with the measured depth, the two arms'
distance along the plan. The first answer that passes is the plan; one that does not is named
in ``attempts`` and the next planner is asked. An earlier plan passed back in is handed out
again only while it is young and the hands and the surfaces under the endpoints are where
they were. Nothing moves.

The shape follows VFlash's trajectory primitives (arXiv:2611.08449): the request declares the
number of gripper waypoints before anything is predicted, both arms of a bimanual plan are one
request, an answer is accepted on its structure alone, one that is not valid is produced again
by a second decoder, and the plan is renewed on a period. VFlash's own model is not used.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PLANNERS: tuple[str, ...] = ("vlm.query", "robobrain.trace")
"""The default ladder. A tool named ``*.query`` is asked in text for every declared arm at
once and answers JSON; any other tool is taken to return the visual-plan schema with one path
per call (``robobrain.trace``; ``*.forecast`` tools get the start as ``query_points``) and is
asked once per arm. An entry is a tool name or ``{"tool": name, "args": {...}, "name": label}``
(the same tool twice with other ``args`` is two planners; ``name`` is what ``attempts`` and
``planner`` call it, by default the tool and its plain arguments)."""

WAYPOINTS = 10
"""Waypoints per arm: VFlash's robot plans are ten per gripper (over a nominal five seconds)."""

START_TOLERANCE = 0.05
"""How far a path's own start may be from the measured start, as a fraction of the image's
width (32 px at 640: about 6 cm on a table 0.9 m from the camera). An answer that cannot say
where the marked hand is has its coordinates wrong (a swapped axis, pixels for thousandths,
the other gripper)."""

MIN_PLAN_PX = 12.0
"""A path ending closer than this to its start [px] proposes no motion for that arm
(following-a-visual-plan's bound: its propose step refuses such a track as no_plan)."""

DEPTH_WINDOW = 3
"""Half-size [px] of the depth window under a pixel (7 x 7, following-a-visual-plan's)."""

TRAVEL_HEIGHT = 0.12
"""Height over the table [m] at which the two arms' paths are compared: where
following-a-visual-plan carries a path's pixels to fly them."""

MIN_HAND_GAP = 0.15
"""Least distance between the two tool centres along the plan [m]: picking-and-placing-two-
objects' bound (two open YAM hands, fingers about 60 mm either side of the tool centre, plus
30 mm)."""

METRIC_TOLERANCE = 0.05
"""How far a planner's own depth may be from the measured depth at the path's start and under
its end [m], and how far behind a measured surface a waypoint may lie. VFlash reports a mean
waypoint error of 3.6-6.5 cm on its 3D robot benchmarks; RoboBrain 2.5's traces, asked by this
script on real YAM frames, ended 9-29 cm in front of the surface measured under their endpoint
(SKILL.md, Measured), which this refuses."""

REPROJECT_TOLERANCE = 0.03
"""How far a metric waypoint may project from its own pixel, as a fraction of the image's
width (19 px at 640). A 1 cm metric grid (VFlash's) alone moves the projection of a point
0.9 m from a 460 px focal length camera by about 4 px, of one 0.5 m away by about 8 px."""

AGREE_TOLERANCE = 0.04
"""How far two planners' endpoints may be apart and still agree, as a fraction of the image's
width (26 px at 640: about 5 cm on a table 0.9 m away, following-a-visual-plan's bound for its
pointing cross-check). On real YAM frames two capable text planners' answers to the same
request ended within it of each other on 115 of 117 pairs, and a weak planner's accepted
answers 103-225 px from theirs on four of its five tasks (SKILL.md, Measured)."""

HAND_MOVED = 0.02
"""A hand further than this [m] from where it was when the plan was made needs a new plan:
above a real YAM station's 12 mm per-move bound, so settling does not count as moving."""

SCENE_CHANGED = 0.02
"""A depth under an endpoint that changed by more than this [m] since the plan was made means
what the plan pointed at has moved (a few times a calibrated station camera's 5.4 mm RMS)."""

MAX_AGE_S = 30.0
"""An earlier plan older than this [s] is not handed out again. VFlash asks for a new plan
every 60 control steps (three 20-action chunks) while its action model runs; here a plan is
used by whole skill steps and a text planner's call took 3-15 s (SKILL.md, Measured), so the
period is a few calls."""

MARKS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("red", (255, 40, 40)),
    ("blue", (40, 110, 255)),
)
"""The rings drawn on the frame shown to a text planner, one per declared arm, and the names
its answer uses for them."""

ORDERS = ("together", "a_then_b", "b_then_a")


class Output(TypedDict):
    route: str
    detail: str
    plan: dict[str, Any]
    subtask: str
    planner: str
    planner_calls: int
    attempts: list[str]
    reused: bool
    metric: str
    min_arm_separation_m: float
    separation_at_fraction: float
    start_pixels: list[list[float]]
    endpoint_pixels: list[list[float]]


class _NoAnswer(Exception):
    """The planner answered that it has no plan."""


@dataclass
class _Start:
    arm: int
    pixel: np.ndarray  # where the path starts in the image
    hand: np.ndarray  # the arm's tool centre in the world
    at_hand: bool  # the start is the tool centre itself, not a pixel the caller gave


def _xyz(p: dict[str, Any]) -> np.ndarray:
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _pose_matrix(pose: Any) -> np.ndarray:
    """gap's Se3Pose (camera-to-world) or a 4x4."""
    if isinstance(pose, dict):
        p, q = pose["position"], pose["rotation"]
        w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
        n = math.sqrt(w * w + x * x + y * y + z * z)
        w, x, y, z = w / n, x / n, y / n, z / n
        m = np.eye(4)
        m[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
        m[:3, 3] = [float(p[k]) for k in ("x", "y", "z")]
        return m
    m = np.asarray(pose, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError("camera pose is an Se3Pose or a 4x4 camera-to-world matrix")
    return m


def _camera(observation: dict[str, Any], name: str) -> dict[str, Any]:
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    usable = [
        c
        for c in cameras
        if c.get("rgb") is not None
        and c.get("depth") is not None
        and c.get("intrinsics") is not None
        and c.get("pose") is not None
    ]
    if not usable:
        raise ValueError("no camera with rgb, depth, intrinsics and pose in the observation")
    if name:
        for c in usable:
            if c.get("name") == name:
                return c
        raise ValueError(f"no usable camera named {name!r} (has {[c.get('name') for c in usable]})")
    for c in usable:
        if any(k in str(c.get("name", "")).lower() for k in ("top", "overhead")):
            return c
    return usable[0]


class _View:
    """One calibrated RGB-D camera (OpenCV axes, camera-to-world pose)."""

    def __init__(self, camera: dict[str, Any]) -> None:
        self.name = str(camera.get("name", ""))
        self.rgb = np.asarray(camera["rgb"], dtype=np.uint8)
        self.depth = np.asarray(camera["depth"], dtype=np.float64).reshape(self.rgb.shape[:2])
        self.k = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
        self.t = _pose_matrix(camera["pose"])
        self.h, self.w = self.depth.shape

    def inside(self, uv: np.ndarray) -> bool:
        return bool(0.0 <= uv[0] < self.w and 0.0 <= uv[1] < self.h)

    def depth_at(self, uv: np.ndarray) -> float | None:
        """Measured depth under a pixel [m along the optical axis]: a small window's median."""
        u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
        r = DEPTH_WINDOW
        patch = self.depth[max(0, v - r) : v + r + 1, max(0, u - r) : u + r + 1]
        ok = np.isfinite(patch) & (patch > 0.05) & (patch < 5.0)
        return float(np.median(patch[ok])) if ok.any() else None

    def point(self, uv: np.ndarray, z: float) -> np.ndarray:
        """The camera-frame point on a pixel's ray at depth z."""
        return np.array(
            [
                (uv[0] - self.k[0, 2]) * z / self.k[0, 0],
                (uv[1] - self.k[1, 2]) * z / self.k[1, 1],
                z,
            ]
        )

    def to_world(self, cam: np.ndarray) -> np.ndarray:
        return np.asarray(cam) @ self.t[:3, :3].T + self.t[:3, 3]

    def see(self, world: np.ndarray) -> tuple[np.ndarray, float]:
        """The pixel a world point is seen at, and its depth along the optical axis."""
        cam = (np.asarray(world, dtype=np.float64) - self.t[:3, 3]) @ self.t[:3, :3]
        z = float(cam[2])
        if z <= 1e-6:
            return np.array([math.nan, math.nan]), z
        return np.array(
            [
                self.k[0, 0] * cam[0] / z + self.k[0, 2],
                self.k[1, 1] * cam[1] / z + self.k[1, 2],
            ]
        ), z

    def at_height(self, uv: np.ndarray, z: float) -> np.ndarray | None:
        """Where the pixel's ray crosses the plane at world height z."""
        ray = self.t[:3, :3] @ self.point(uv, 1.0)
        if abs(ray[2]) < 1e-6:
            return None
        s = (float(z) - self.t[2, 3]) / ray[2]
        return None if s <= 0 else self.t[:3, 3] + s * ray

    def table_z(self) -> float:
        """The fullest 4 mm height bin of every 2nd depth pixel (following-a-visual-plan's)."""
        v, u = np.mgrid[0 : self.h : 2, 0 : self.w : 2]
        d = self.depth[v, u]
        ok = np.isfinite(d) & (d > 0.05) & (d < 5.0)
        cam = np.stack(
            [
                (u[ok] - self.k[0, 2]) * d[ok] / self.k[0, 0],
                (v[ok] - self.k[1, 2]) * d[ok] / self.k[1, 1],
                d[ok],
            ],
            -1,
        )
        bins, counts = np.unique(np.floor(self.to_world(cam)[:, 2] / 0.004), return_counts=True)
        return float((bins[np.argmax(counts)] + 0.5) * 0.004)


# ---------------------------------------------------------------------------
# The pure parts: the text request and its answer, resampling, the arms' distance
# ---------------------------------------------------------------------------


def mark_image(rgb: np.ndarray, marks: list[tuple[np.ndarray, tuple[int, int, int]]]) -> np.ndarray:
    """The frame with a coloured ring (white-edged) around every start pixel."""
    out = np.ascontiguousarray(np.asarray(rgb)[..., :3]).astype(np.uint8).copy()
    h, w = out.shape[:2]
    radius = max(6.0, 0.014 * w)
    v, u = np.mgrid[0:h, 0:w]
    for uv, colour in marks:
        r = np.hypot(u - float(uv[0]), v - float(uv[1]))
        out[(r >= radius - 4.0) & (r <= radius + 3.0)] = (255, 255, 255)
        out[(r >= radius - 2.5) & (r <= radius + 1.5)] = colour
    return out


def text_request(
    instruction: str,
    width: int,
    height: int,
    marks: list[tuple[int, str, bool]],
    waypoints: int,
    arm_instructions: list[str] | None = None,
) -> str:
    """The typed request for a text planner. ``marks``: (arm id, ring colour, the ring is on
    that arm's tool centre) per declared arm."""
    rings, lines, shape = [], [], []
    for i, (arm, colour, at_hand) in enumerate(marks):
        what = (
            f"the tool centre of arm {arm}'s gripper" if at_hand else f"a point arm {arm} carries"
        )
        rings.append(f"a {colour} ring around {what}")
        if arm_instructions and str(arm_instructions[i]).strip():
            lines.append(
                f"The {colour} ring's part of the task: {str(arm_instructions[i]).strip()}\n"
            )
        shape.append(f'"{colour}": [[x, y], ... exactly {waypoints + 1} pairs]')
    kinds = []
    if any(at_hand for _, _, at_hand in marks):
        kinds.append(
            "For a ring on a gripper's tool centre: the gripper's motion in the next step of the "
            "task, ending on the object or place that step reaches."
        )
    if not all(at_hand for _, _, at_hand in marks):
        kinds.append(
            "For a ring on a carried point: where the task takes that point, ending where the "
            "task leaves it."
        )
    return (
        f"The image ({width} x {height} px) is a fixed camera's view of a robot workcell. Drawn on "
        f"it: {' and '.join(rings)}.\n"
        f"Task: {str(instruction).strip()}\n"
        f"{''.join(lines)}"
        f"Give each ring a path in the image: where its ringed point goes. {' '.join(kinds)} "
        f"A path is {waypoints + 1} points: the ring's centre as you see it, then exactly "
        f"{waypoints} waypoints in order, evenly spaced along the path. A ringed point that should "
        f"not move repeats its ring's centre.\n"
        "Coordinates are [x, y] numbers from 0 to 1000: x runs from the image's left edge (0) to "
        "its right edge (1000), y from its top edge (0) to its bottom edge (1000).\n"
        "Answer with one JSON object and nothing else:\n"
        f'{{"subtask": "<what the paths do, in a few words>", {", ".join(shape)}}}'
    )


def parse_text_answer(
    text: str, colours: list[str], waypoints: int, width: int, height: int
) -> tuple[str, dict[str, dict[str, Any]]]:
    """A text planner's answer as (subtask, {colour: {"start": [u, v], "pixels": [[u, v]]}}) in
    pixels. ValueError names the first rule of the declared shape the answer breaks."""
    body = re.search(r"\{.*\}", text or "", re.S)
    if not body:
        raise ValueError("no JSON object in the answer")
    try:
        data = json.loads(body.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"the answer is not one JSON object ({exc.msg})") from None
    if not isinstance(data, dict):
        raise ValueError("the answer is not one JSON object")
    for colour, _ in MARKS:
        if colour in data and colour not in colours:
            raise ValueError(f"a {colour} path, which was not declared")

    def pair(value: Any, what: str) -> list[float]:
        ok = (
            isinstance(value, list)
            and len(value) == 2
            and all(
                isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c)
                for c in value
            )
        )
        if not ok:
            raise ValueError(f"{what} is not an [x, y] pair of numbers")
        if not all(0.0 <= c <= 1000.0 for c in value):
            raise ValueError(f"{what} {value} is outside 0..1000")
        return [
            min(value[0] * width / 1000.0, width - 1.0),
            min(value[1] * height / 1000.0, height - 1.0),
        ]

    out: dict[str, dict[str, Any]] = {}
    for colour in colours:
        points = data.get(colour)
        if not isinstance(points, list):
            raise ValueError(f"no list of points for the {colour} ring")
        if len(points) != waypoints + 1:
            raise ValueError(
                f"the {colour} path has {len(points) - 1} waypoints after its ring's centre, "
                f"{waypoints} declared"
            )
        pairs = [
            pair(p, f"the {colour} path's " + (f"waypoint {i}" if i else "start"))
            for i, p in enumerate(points)
        ]
        out[colour] = {"start": pairs[0], "pixels": pairs[1:]}
    subtask = data.get("subtask")
    return (subtask.strip() if isinstance(subtask, str) else ""), out


def resample(
    start: np.ndarray, pixels: np.ndarray, n: int, depths: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """``n`` pixels evenly spaced by arc length in the image along start -> pixels (the last
    is the path's own end), and ``depths`` ([1 + len(pixels)], the start's first) interpolated
    at them."""
    pts = np.vstack([np.asarray(start, dtype=np.float64), np.asarray(pixels, dtype=np.float64)])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if s[-1] <= 0.0:
        at = np.zeros(n)
    else:
        at = s[-1] * np.arange(1, n + 1) / n
    out = np.stack([np.interp(at, s, pts[:, 0]), np.interp(at, s, pts[:, 1])], -1)
    out[-1] = pts[-1]
    return out, (None if depths is None else np.interp(at, s, np.asarray(depths, dtype=float)))


def separation(a: np.ndarray, b: np.ndarray, order: str = "together") -> tuple[float, float]:
    """The smallest distance between two arms' routes (each ``[n + 1][x, y]``, the start
    first) and the fraction of the whole plan it is at. ``together``: both arms pass their
    waypoints in step; ``a_then_b`` / ``b_then_a``: one flies its whole route while the other
    waits at its start, then the other flies while the first rests at its end."""
    if order not in ORDERS:
        raise ValueError(f"order is one of {ORDERS}, got {order!r}")
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)

    def along(route: np.ndarray, t: np.ndarray) -> np.ndarray:
        s = t * (len(route) - 1)
        i = np.minimum(s.astype(int), len(route) - 2)
        f = (s - i)[:, None]
        return route[i] * (1.0 - f) + route[i + 1] * f

    t = np.linspace(0.0, 1.0, 20 * (max(len(a), len(b)) - 1) + 1)
    if order == "together":
        pa, pb, at = along(a, t), along(b, t), t
    else:
        first, second = (a, b) if order == "a_then_b" else (b, a)
        p1 = np.vstack([along(first, t), np.repeat(first[-1:], len(t), 0)])
        p2 = np.vstack([np.repeat(second[:1], len(t), 0), along(second, t)])
        pa, pb = (p1, p2) if order == "a_then_b" else (p2, p1)
        at = np.linspace(0.0, 1.0, 2 * len(t))
    d = np.linalg.norm(pa - pb, axis=1)
    i = int(np.argmin(d))
    return float(d[i]), float(at[i])


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------


def _ask_text(
    ctx: NodeContext,
    out: Output,
    tool: str,
    args: dict[str, Any],
    view: _View,
    instruction: str,
    arm_instructions: list[str] | None,
    starts: list[_Start],
    n: int,
) -> dict[str, Any]:
    colours = [MARKS[i][0] for i in range(len(starts))]
    image = mark_image(view.rgb, [(st.pixel, MARKS[i][1]) for i, st in enumerate(starts)])
    prompt = text_request(
        instruction,
        view.w,
        view.h,
        [(st.arm, colours[i], st.at_hand) for i, st in enumerate(starts)],
        n,
        arm_instructions,
    )
    out["planner_calls"] += 1
    reply = ctx.tool(tool, prompt=prompt, image=image, **args)
    text = reply.get("text", "") if isinstance(reply, dict) else str(reply)
    subtask, paths = parse_text_answer(text, colours, n, view.w, view.h)
    tracks = [{"query_pixel": paths[c]["start"], "pixels": paths[c]["pixels"]} for c in colours]
    return {"backend": tool, "subtask": subtask, "tracks": tracks, "typed": True}


def _ask_paths(
    ctx: NodeContext,
    out: Output,
    tool: str,
    args: dict[str, Any],
    view: _View,
    instruction: str,
    arm_instructions: list[str] | None,
    starts: list[_Start],
) -> dict[str, Any]:
    tracks, backend = [], tool
    for i, st in enumerate(starts):
        mine = str(arm_instructions[i]).strip() if arm_instructions else ""
        call: dict[str, Any] = {
            "image": view.rgb,
            "instruction": mine or instruction,
            "depth": view.depth.astype(np.float32),
            "intrinsics": view.k,
            "camera_pose": view.t,
        }
        start = [float(st.pixel[0]), float(st.pixel[1])]
        if tool.endswith(".forecast"):
            call["query_points"] = [start]
        else:
            call["start_point"] = start
        call.update(args)
        out["planner_calls"] += 1
        reply = ctx.tool(tool, **call)
        if not isinstance(reply, dict) or not reply.get("found") or not reply.get("tracks"):
            why = reply.get("error") if isinstance(reply, dict) else None
            raise _NoAnswer(f"arm {st.arm}: {why or 'the planner returned no track'}")
        track = reply["tracks"][0]
        keys = ("query_pixel", "pixels", "start_camera", "points_camera")
        tracks.append({k: track.get(k) for k in keys})
        backend = str(reply.get("backend") or tool)
    return {"backend": backend, "subtask": "", "tracks": tracks, "typed": False}


# ---------------------------------------------------------------------------
# Checking an answer against the declared shape and the measured scene
# ---------------------------------------------------------------------------


def _metric_problems(
    view: _View,
    st: _Start,
    px: np.ndarray,
    cam: np.ndarray,
    start_cam: Any,
    z_end: float,
    tolerance: float,
) -> list[str]:
    """Every way a path's own metric points disagree with what is measured (none: they are
    passed on). Points that are not on their own pixels are compared with nothing else."""
    if not np.isfinite(cam).all() or (cam[:, 2] <= 0).any():
        return ["a metric waypoint is not a point in front of the camera"]
    uv = cam[:, :2] / cam[:, 2:3] * [view.k[0, 0], view.k[1, 1]] + [view.k[0, 2], view.k[1, 2]]
    off = np.linalg.norm(uv - px, axis=1)
    if off.max() > REPROJECT_TOLERANCE * view.w:
        i = int(np.argmax(off))
        return [f"metric waypoint {i + 1} projects {off[i]:.0f} px from its own pixel"]
    problems = []
    # A path from the tool centre starts at the hand's own depth, which the arm knows; the depth
    # image under that pixel shows the table between the fingers or the gripper's back.
    if st.at_hand:
        measured, what = view.see(st.hand)[1], "its hand's depth"
    else:
        measured, what = view.depth_at(st.pixel), "the depth measured at the given start"
    if start_cam is not None and measured is not None:
        gap = float(np.asarray(start_cam, dtype=np.float64).reshape(3)[2]) - measured
        if abs(gap) > tolerance:
            problems.append(f"its start is {abs(gap) * 1e3:.0f} mm off {what}")
    gap = float(cam[-1, 2]) - z_end
    if abs(gap) > tolerance:
        side = "behind" if gap > 0 else "in front of"
        problems.append(
            f"it ends {abs(gap) * 1e3:.0f} mm {side} the surface measured under its endpoint"
        )
    for i in range(len(px) - 1):
        z = view.depth_at(px[i])
        if z is not None and cam[i, 2] > z + tolerance:
            problems.append(
                f"metric waypoint {i + 1} lies {(cam[i, 2] - z) * 1e3:.0f} mm behind a measured "
                "surface"
            )
            break  # the first such waypoint names it
    return problems


def _check(
    ctx: NodeContext,
    view: _View,
    starts: list[_Start],
    answer: dict[str, Any],
    n: int,
    *,
    resample_ok: bool,
    start_tolerance: float,
    metric_tolerance: float,
    require_metric: bool,
    order: str,
    min_hand_gap: float,
    travel_height: float,
    table_z: float | None,
) -> tuple[str, str, dict[str, Any]]:
    """("planned" | "rejected" | "still", why, what a planned answer adds to the output)."""
    tracks: list[dict[str, Any]] = []
    notes: list[str] = []
    metric: list[str] = []
    for st, raw in zip(starts, answer["tracks"], strict=True):
        who = f"arm {st.arm}"
        try:
            px = np.asarray(raw.get("pixels"), dtype=np.float64).reshape(-1, 2)
            given = raw.get("query_pixel")
            own = np.asarray(st.pixel if given is None else given, dtype=np.float64).reshape(2)
        except (TypeError, ValueError):
            return "rejected", f"{who}: the waypoints are not [u, v] pairs", {}
        if not len(px) or not (np.isfinite(px).all() and np.isfinite(own).all()):
            return "rejected", f"{who}: a waypoint is missing or not a finite number", {}
        cam, start_cam = None, raw.get("start_camera")
        try:
            if raw.get("points_camera") is not None:
                cam = np.asarray(raw["points_camera"], dtype=np.float64).reshape(len(px), 3)
        except (TypeError, ValueError):
            cam = None
        if len(px) != n:
            if not resample_ok:
                return "rejected", f"{who}: {len(px)} waypoints, {n} declared", {}
            notes.append(f"{who} resampled from {len(px)}")
            depths = None
            if cam is not None and start_cam is not None:
                depths = np.concatenate([[float(np.asarray(start_cam).reshape(3)[2])], cam[:, 2]])
            px, z = resample(own, px, n, depths)
            cam = (
                None
                if z is None
                else np.stack([view.point(p, d) for p, d in zip(px, z, strict=True)])
            )
        for i, p in enumerate(px):
            if not view.inside(p):
                return (
                    "rejected",
                    f"{who}: waypoint {i + 1} ({p[0]:.0f}, {p[1]:.0f}) px is outside the image",
                    {},
                )
        off, bound = float(np.linalg.norm(own - st.pixel)), start_tolerance * view.w
        if off > bound:
            where = "its hand" if st.at_hand else "the given start"
            return (
                "rejected",
                f"{who}: the path starts {off:.0f} px from {where} (bound {bound:.0f} px)",
                {},
            )
        active = float(np.linalg.norm(px[-1] - st.pixel)) >= MIN_PLAN_PX
        track: dict[str, Any] = {
            "arm_id": st.arm,
            "query_pixel": [float(st.pixel[0]), float(st.pixel[1])],
            "pixels": px.tolist(),
            "predicted": [True] * n,
            "active": active,
            "start_error_px": off,
            "hand_world": st.hand.tolist(),
            "endpoint_depth_m": None,
        }
        if active:
            z_end = view.depth_at(px[-1])
            if z_end is None:
                return (
                    "rejected",
                    f"{who}: no valid depth under the endpoint "
                    f"({px[-1][0]:.0f}, {px[-1][1]:.0f}) px",
                    {},
                )
            track["endpoint_depth_m"] = z_end
            if cam is None:
                metric.append("none")
            else:
                why = _metric_problems(view, st, px, cam, start_cam, z_end, metric_tolerance)
                metric.append(f"{who}: {' and '.join(why)}" if why else "anchored")
                if not why:
                    track["points_camera"] = cam.tolist()
                    track["points_world"] = view.to_world(cam).tolist()
        tracks.append(track)

    if not any(t["active"] for t in tracks):
        return (
            "still",
            f"no motion proposed: every path ends within {MIN_PLAN_PX:.0f} px of its start",
            {},
        )
    dropped = [m for m in metric if m not in ("none", "anchored")]
    if dropped or "none" in metric:
        for t in tracks:  # a metric part is passed on for every moving arm or for none
            t.pop("points_camera", None)
            t.pop("points_world", None)
    if dropped:  # every arm whose metric part disagrees, and every way it does
        verdict = "dropped: " + "; ".join(dropped)
    else:
        verdict = "none" if "none" in metric else "anchored"
    if require_metric and verdict != "anchored":
        return "rejected", f"no metric part that agrees with the measured depth ({verdict})", {}
    extra: dict[str, Any] = {
        "tracks": tracks,
        "metric": verdict,
        "notes": notes,
        "min_arm_separation_m": -1.0,
        "separation_at_fraction": -1.0,
    }
    if len(tracks) == 2:
        tz = table_z
        if tz is None:
            try:
                tz = float(ctx.tool("robot.describe_workspace")["surface_z"])
            except Exception as exc:  # not every connector serves it: fall back to the scene
                logger.info("requesting-a-typed-visual-plan: no workspace surface (%s)", exc)
                tz = view.table_z()
        routes = [
            _route(view, st, t, float(tz) + travel_height)
            for st, t in zip(starts, tracks, strict=True)
        ]
        least, at = separation(routes[0], routes[1], order)
        now = float(np.linalg.norm(routes[0][0] - routes[1][0]))
        extra.update(min_arm_separation_m=least, separation_at_fraction=at)
        # A plan that never brings the hands closer than they already are is let through.
        if least < min(min_hand_gap, now) - 1e-3:
            return (
                "rejected",
                f"arms {starts[0].arm} and {starts[1].arm} come within {least * 1e3:.0f} mm "
                f"(bound {min_hand_gap * 1e3:.0f} mm) at {at * 100:.0f} % of the plan ({order})",
                {},
            )
    return "planned", "", extra


def _route(view: _View, st: _Start, track: dict[str, Any], z_travel: float) -> np.ndarray:
    """An arm's horizontal route ``[n + 1][x, y]``: its start, the path's pixels carried along
    their camera rays to the travel height, the endpoint on the surface measured under it. An
    arm that stays is at its hand throughout."""
    n = len(track["pixels"])
    if not track["active"]:
        return np.repeat(st.hand[None, :2], n + 1, 0)
    first = st.hand[:2]
    if not st.at_hand:
        z = view.depth_at(st.pixel)
        lifted = (
            view.at_height(st.pixel, z_travel)
            if z is None
            else view.to_world(view.point(st.pixel, z))
        )
        first = first if lifted is None else lifted[:2]
    route = [first]
    for p in track["pixels"][:-1]:
        w = view.at_height(np.asarray(p), z_travel)
        route.append(route[-1] if w is None else w[:2])
    end = view.to_world(view.point(np.asarray(track["pixels"][-1]), track["endpoint_depth_m"]))
    route.append(end[:2])
    return np.asarray(route)


def _stale(
    plan: dict[str, Any],
    view: _View,
    starts: list[_Start],
    declared: dict[str, Any],
    instruction: str,
    max_age_s: float,
) -> str:
    """Why an earlier plan cannot be handed out again, or ""."""
    typed = plan.get("typed") or {}
    same = (
        plan.get("found")
        and [int(a) for a in typed.get("arms") or []] == declared["arms"]
        and typed.get("waypoints") == declared["waypoints"]
        and str(plan.get("instruction", "")) == instruction
        and str(plan.get("camera", "")) == view.name
        and list(plan.get("image_size") or []) == [view.w, view.h]
        and len(plan.get("tracks") or []) == len(starts)
    )
    if not same:
        return "it was made for another request"
    if declared["require_metric"] and plan.get("metric") != "anchored":
        return "it has no metric part"
    age = time.time() - float(plan.get("planned_at") or 0.0)
    if max_age_s > 0 and not 0.0 <= age <= max_age_s:
        return f"it is {age:.0f} s old (bound {max_age_s:.0f} s)"
    for st, track in zip(starts, plan["tracks"], strict=True):
        moved = float(np.linalg.norm(st.hand - np.asarray(track.get("hand_world"), dtype=float)))
        if not moved <= HAND_MOVED:
            return f"arm {st.arm}'s hand moved {moved * 1e3:.0f} mm since it was made"
        if track.get("active"):
            z = view.depth_at(np.asarray(track["pixels"][-1], dtype=np.float64))
            was = float(track.get("endpoint_depth_m") or math.nan)
            if z is None or not abs(z - was) <= SCENE_CHANGED:
                change = "has no depth now" if z is None else f"moved {abs(z - was) * 1e3:.0f} mm"
                return f"the surface under arm {st.arm}'s endpoint {change}"
    return ""


def run(
    ctx: NodeContext,
    instruction: str,
    arms: list[int] | None = None,
    waypoints: int = WAYPOINTS,
    planners: list[Any] | None = None,
    plan: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    camera_name: str = "",
    start_pixels: list[Any] | None = None,
    arm_instructions: list[str] | None = None,
    resample_paths: bool = True,
    require_metric: bool = False,
    order: str = "together",
    min_hand_gap: float = MIN_HAND_GAP,
    travel_height: float = TRAVEL_HEIGHT,
    table_z: float | None = None,
    start_tolerance: float = START_TOLERANCE,
    metric_tolerance: float = METRIC_TOLERANCE,
    agree: int = 1,
    agree_tolerance: float = AGREE_TOLERANCE,
    max_age_s: float = MAX_AGE_S,
) -> Output:
    arm_ids = [int(a) for a in (arms if arms is not None else [0])]
    n = int(waypoints)
    if not 1 <= len(arm_ids) <= len(MARKS) or len(set(arm_ids)) != len(arm_ids):
        raise ValueError(f"arms is one or two different arm ids, got {arms!r}")
    if n < 2:
        raise ValueError(f"waypoints is at least 2, got {waypoints!r}")
    if order not in ORDERS:
        raise ValueError(f"order is one of {ORDERS}, got {order!r}")
    for name, per_arm in (("start_pixels", start_pixels), ("arm_instructions", arm_instructions)):
        if per_arm is not None and len(per_arm) != len(arm_ids):
            raise ValueError(f"{name} has one entry per declared arm ({len(arm_ids)})")
    ladder: list[tuple[str, dict[str, Any], str]] = []
    for entry in planners if planners is not None else PLANNERS:
        if isinstance(entry, dict):
            tool, args = str(entry["tool"]), dict(entry.get("args") or {})
            plain = [f"{k}={v}" for k, v in args.items() if isinstance(v, (str, int, float, bool))]
            label = str(entry.get("name") or (f"{tool}({', '.join(plain)})" if plain else tool))
            ladder.append((tool, args, label))
        else:
            ladder.append((str(entry), {}, str(entry)))
    instruction = str(instruction or "").strip()
    out: Output = {
        "route": "no_plan",
        "detail": "",
        "plan": {},
        "subtask": "",
        "planner": "",
        "planner_calls": 0,
        "attempts": [],
        "reused": False,
        "metric": "none",
        "min_arm_separation_m": -1.0,
        "separation_at_fraction": -1.0,
        "start_pixels": [],
        "endpoint_pixels": [],
    }
    if not instruction:
        out["detail"] = "no instruction to ask a planner with"
        return out
    if observation is None:
        observation = ctx.tool("robot.get_observation")
    try:
        view = _View(_camera(observation, camera_name))
    except ValueError as exc:
        out["detail"] = str(exc)
        return out

    # --- where each declared arm's path starts -----------------------------------
    starts: list[_Start] = []
    for i, arm in enumerate(arm_ids):
        reply = ctx.tool("robot.get_ee_pose", **({"arm_id": arm} if arm else {}))
        hand = _xyz(_pose(reply)["position"])
        given = start_pixels[i] if start_pixels is not None else None
        if given is not None:
            pixel = np.array([float(given[0]), float(given[1])])
        else:
            pixel, z = view.see(hand)
            if z <= 0 or not view.inside(pixel):
                out["detail"] = (
                    f"arm {arm}'s tool centre is not in the {view.name or 'camera'} image: no path "
                    "can be asked for it (pass start_pixels to plan from another point)"
                )
                return out
        starts.append(_Start(arm, pixel, hand, given is None))
    out["start_pixels"] = [[float(s.pixel[0]), float(s.pixel[1])] for s in starts]
    declared = {"arms": arm_ids, "waypoints": n, "require_metric": bool(require_metric)}

    # --- an earlier plan: handed out again only while nothing it rests on changed ----
    if plan:
        why = _stale(plan, view, starts, declared, instruction, float(max_age_s))
        if not why:
            age = time.time() - float(plan["planned_at"])
            out["attempts"].append(f"earlier plan: reused ({age:.0f} s old)")
            return _finish(out, dict(plan), reused=True)
        out["attempts"].append(f"earlier plan: not reused: {why}")

    # --- the ladder ----------------------------------------------------------------
    need, refused = max(1, int(agree)), False
    valid: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
    for tool, args, label in ladder:
        try:
            if tool.endswith(".query"):
                answer = _ask_text(
                    ctx, out, tool, args, view, instruction, arm_instructions, starts, n
                )
            else:
                answer = _ask_paths(
                    ctx, out, tool, args, view, instruction, arm_instructions, starts
                )
        except _NoAnswer as exc:
            out["attempts"].append(f"{label}: no answer: {exc}")
            continue
        except ValueError as exc:  # the answer breaks the declared shape
            refused = True
            out["attempts"].append(f"{label}: rejected: {exc}")
            continue
        except Exception as exc:  # a planner that raised gave no answer
            out["attempts"].append(f"{label}: raised: {exc}")
            continue
        verdict, why, extra = _check(
            ctx,
            view,
            starts,
            answer,
            n,
            resample_ok=bool(resample_paths) and not answer["typed"],
            start_tolerance=float(start_tolerance),
            metric_tolerance=float(metric_tolerance),
            require_metric=bool(require_metric),
            order=order,
            min_hand_gap=float(min_hand_gap),
            travel_height=float(travel_height),
            table_z=table_z,
        )
        if verdict != "planned":
            refused = refused or verdict == "rejected"
            out["attempts"].append(
                f"{label}: {verdict}: {why}" if verdict == "rejected" else f"{label}: {why}"
            )
            logger.info("requesting-a-typed-visual-plan: %s", out["attempts"][-1])
            continue
        said = [f"{len(arm_ids)} arm(s) x {n} waypoints", *extra["notes"]]
        if extra["metric"] != "none":
            said.append(f"metric part {extra['metric']}")
        if extra["min_arm_separation_m"] >= 0:
            said.append(f"arms at least {extra['min_arm_separation_m'] * 1e3:.0f} mm apart")
        valid.append((label, answer, extra, ", ".join(said)))
        bound = float(agree_tolerance) * view.w
        agreeing = [v for v in valid if _same_ends(v[2]["tracks"], extra["tracks"], bound)]
        if len(agreeing) < need:
            out["attempts"].append(
                f"{label}: valid ({valid[-1][3]}); {len(agreeing)} of {need} planners agree on "
                "the endpoints"
            )
            continue
        label, answer, extra, text = agreeing[0]  # the ladder's order is the preference
        others = [v[0] for v in agreeing[1:]]
        out["attempts"].append(
            f"{label}: planned: {text}"
            + (f", endpoints within {bound:.0f} px of {' and '.join(others)}'s" if others else "")
        )
        new = {
            "found": True,
            "horizon": n,
            "dt_s": None,
            "frame": "world" if extra["metric"] == "anchored" else "pixel",
            "backend": answer["backend"],
            "planner": label,
            "confirmed_by": others,
            "subtask": answer["subtask"],
            "instruction": instruction,
            "typed": declared,
            "metric": extra["metric"],
            "camera": view.name,
            "image_size": [view.w, view.h],
            "planned_at": time.time(),
            "min_arm_separation_m": extra["min_arm_separation_m"],
            "separation_at_fraction": extra["separation_at_fraction"],
            "tracks": extra["tracks"],
        }
        return _finish(out, new, reused=False)

    if valid:
        out["attempts"].append(f"no {need} planners agree on the endpoints")
    out["route"] = "plan_rejected" if refused or valid else "no_plan"
    out["detail"] = " | ".join(out["attempts"]) or "no planner to ask"
    return out


def _same_ends(a: list[dict[str, Any]], b: list[dict[str, Any]], bound: float) -> bool:
    """Two answers move the same arms and end each within ``bound`` px of the other's."""
    return all(
        ta["active"] == tb["active"]
        and (
            not ta["active"]
            or np.linalg.norm(np.subtract(ta["pixels"][-1], tb["pixels"][-1])) <= bound
        )
        for ta, tb in zip(a, b, strict=True)
    )


def _finish(out: Output, plan: dict[str, Any], *, reused: bool) -> Output:
    out.update(
        route="planned",
        plan=plan,
        subtask=str(plan.get("subtask", "")),
        planner=str(plan.get("planner", "")),
        reused=reused,
        metric=str(plan.get("metric", "none")),
        min_arm_separation_m=float(plan.get("min_arm_separation_m", -1.0)),
        separation_at_fraction=float(plan.get("separation_at_fraction", -1.0)),
        endpoint_pixels=[[float(c) for c in t["pixels"][-1]] for t in plan["tracks"]],
        detail=" | ".join(out["attempts"]),
    )
    return out
