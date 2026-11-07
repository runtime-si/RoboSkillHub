# Adapted from MOKA (https://github.com/moka-manipulation/moka), MIT,
# Copyright (c) 2024 Kuan Fang, Fangchen Liu:
# moka/planners/visual_prompt_utils.py (request_motion, parse_json_string,
# remove_trailing_comments, annotate_visual_prompts, sample_waypoint_in_tile),
# moka/vision/keypoint.py (fps, get_keypoints_from_segmentation),
# moka/planners/planner.py (compute_context_3d) and
# prompts/visual_prompt_planner/{propose_subtasks,select_motion}.txt.
# Changes: the hub's vlm.query in place of GPT-4V; candidate keypoints from a
# given mask, sam3.segment_text or gemini-er.point in place of Grounding DINO +
# SAM; the grid is drawn on a square crop around the objects sized in metres
# (MOKA used a fixed hand-set crop of its table); a waypoint is its tile's
# centre (MOKA sampled it uniformly in the tile); depth is read as a robust
# local percentile (MOKA read one pixel); waypoints are rays cut by the
# contact-height plane; contact heights for a side approach use the target's
# mid height; the grasp heading comes from the object's principal axis (MOKA
# used a learned grasp sampler); the outputs are world-frame hub types.
"""Plan a keypoint motion with MOKA's marked visual prompt: the VLM picks points and tiles, depth makes them 3D."""

from __future__ import annotations

import json
import logging
import math
import re
import string
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

NUM_CANDIDATES = 6
"""Candidate keypoints per object, centre included (MOKA's num_candidate_keypoints)."""

GRID_SIZE = 5
"""Tiles per side of the waypoint grid (MOKA's waypoint_grid_size [5, 5])."""

TILE_SIZE = 0.05
"""Side of one tile at the objects' depth [m]: the crop is at least GRID_SIZE x this
across, so a 'next tile' waypoint is about 5 cm away. MOKA's crop was set by hand
per setup; this is the YAM station's choice for table-top pushes of a few cm."""

CROP_MARGIN = 0.08
"""Free space kept around the candidate keypoints in the crop [m], so the
pre/post-contact tiles have room on every side."""

RENDER_SIZE = 768
"""The marked crop is resized to this many pixels a side for the VLM, so labels
and the 3-5 mm marks stay legible when the crop is ~150 px of a 640 x 480 frame."""

Z_OFFSET = 0.01
"""Height added over a surface point the motion comes down onto [m] (MOKA's
grasp_z_offset). With a YAM's closed fingertips ~10 mm below its tool centre, a
tool centre this high over the point puts the fingertips on it."""

HOVER_CLEARANCE = 0.12
"""Safe height over the support for free moves [m]; MOKA's safe_z 0.25 over its
min_z 0.12 is 13 cm."""

OBJECT_CLEARANCE = 0.05
"""The safe height is also at least this much over the highest keypoint [m]."""

CONTACT_FRACTION = 0.5
"""A side approach ('same' height) meets the target at this fraction of its height
over the support (pushing-objects' choice; MOKA used the target's top + offset,
which a horizontal push passes over)."""

DEPTH_WINDOW = 3
"""Half-size of the pixel window a keypoint's depth is read from [px]."""

DEPTH_PERCENTILE = 30.0
"""Percentile of the window's depths taken (the near side: a point on an object's
edge reads the object, not the table beyond it)."""

SUPPORT_BAND = (0.15, 0.005)
"""The support is looked for this far under / over the lowest keypoint [m]: it
cannot stand above what rests on it, and the floor beyond a table's edge (70 cm
down on the YAM station, inside a crop that reaches past the edge) is ignored."""

SUPPORT_BIN = 0.004
"""Height histogram bin for the support plane [m] (the camera's ~3 mm depth noise
at 1 m); the support is the fullest bin, the table being the broad surface."""

GRASP_PCA_RADIUS = 0.03
"""Object points within this of the grasp point fix its local axis [m]."""

SAM_SCORE_FLOOR = 0.3
"""sam3.segment_text masks scoring under this are ignored (the sam3 bundle's advice)."""

ATTEMPTS = 3
"""VLM replies tried before giving up (MOKA retried until one parsed)."""


SUBTASK_PROMPT = """The input request contains:
    - A string that describes a one-step table-top manipulation task.
    - An image of the current table-top environment captured from a top-down camera.

The output response is one dictionary in JSON form that specifies the task. It contains these fields:
    - 'instruction': A string to describe the task in natural language forms.
    - 'object_grasped': A string to describe the name (within 3 words) of the object that the robot gripper will hold in hand while executing the task (e.g., the object to be picked, turned on, or used as a tool to interact with other objects). This field can be '' if there is no such object involved in this task.
    - 'object_unattached': A string to describe the name (within 3 words) of the object that the robot gripper will interact with directly or via another object without holding it in hand (e.g., the object to be swept, to be pressed, to be pushed, and the target where `object_grasped` will be moved onto). This field can be '' if there is no such object involved in this task.
    - 'motion_direction': A string to describe the direction of the robot gripper motion while performing the task (e.g., 'from left to right', 'backward', 'downward'), in the image's terms (left, right, toward the top or the bottom of the image).

Answer with the dictionary in a ```json block."""


SELECT_MOTION_PROMPT = """Describe the robot gripper's motion to solve the task by selecting pre-defined keypoints and waypoints.

The input request contains:
    - The task information as dictionaries. The dictionary contains these fields:
        * 'instruction': The task in natural language forms.
        * 'object_grasped': The object that the robot gripper will hold in hand while executing the task.
        * 'object_unattached': The object that the robot gripper will interact with either directly or via another object without holding it in hand.
        * 'motion_direction': The motion direction of the robot gripper or the in-hand object while performing the task.
    - An image of the current table-top environment captured from a top-down camera, annotated with a set of visual marks:
        * candidate keypoints on 'object_grasped': Red dots marked as 'P[i]' on the image, where [i] is an integer.
        * candidate keypoints on 'object_unattached': Blue dots marked as 'Q[i]' on the image, where [i] is an integer.
        * grid for waypoints: Grid lines that uniformly divide the images into tiles. The grid equally divides the image into columns marked as {columns} from left to right and rows marked as {rows} from bottom to top. Each tile's name is written in its top-left corner.

The motion consists of an optional grasping phase and a manipulation phase, specified by 'grasp_keypoint', 'function_keypoint', 'target_keypoint', 'pre_contact_waypoint', 'post_contact_waypoint'.

Please note: In the grasping phase, the robot gripper grasps 'object_grasped' at the 'grasp_keypoint'. In the manipulation phase, the robot gripper sequentially moves the 'function_keypoint' to 'pre_contact_waypoint', 'target_keypoint', 'post_contact_waypoint', performing a motion trajectory specified by 'motion_direction' in the input request. If 'object_grasped' is '', the closed gripper itself makes the contact and moves through these points.

More specifically, the definitions of these points are:
    - 'grasp_keypoint': The point on 'object_grasped' indicates the part where the robot gripper should hold.
    - 'function_keypoint': The point on 'object_grasped' indicates the part that will make contact with 'object_unattached.'
    - 'target_keypoint': If the task is pick-and-place, this is the location where object_grasped will be moved to. Otherwise, this is the point on 'object_unattached' indicating the part that will be contacted by 'function_keypoint', or the robot gripper (if 'function_keypoint' is '').
    - 'pre_contact_waypoint': The waypoint in the free space that the functional point moves to before making contact with the 'target_keypoint', if 'object_unattached' is not ''. Otherwise, this will be the waypoint that the functional point moves to after grasping 'object_grasped'.
    - 'post_contact_waypoint': The waypoint in the free space that the functional point moves to after making contact with the 'target_keypoint', if 'object_unattached' is not ''. Otherwise, this will be ''.

The response should be a dictionary in JSON form, which contains:
    - 'grasp_keypoint': Selected from candidate keypoints marked as 'P[i]' on the image. This will be '' if and only if 'object_grasped' is ''.
    - 'function_keypoint': Selected from candidate keypoints marked as 'P[i]' on the image. This will be '' if and only if 'object_grasped' or 'object_unattached' is '' or the task is pick-and-place.
    - 'target_keypoint': Selected from keypoint candidates marked as 'Q[i]' on the image. This will be '' if and only if 'object_unattached' is ''.
    - 'target_tile': The tile where 'target_keypoint' is currently located in. This is selected from candidate tiles '[x][i]' marked on the image, where [x] is the column index as a lower letter and [i] is the row index as an integer.
    - 'pre_contact_tile': The tile where 'pre_contact_waypoint' should be in. This is selected from candidate tiles '[x][i]' marked on the image, where [x] is the column index as a lower letter and [i] is the row index as an integer.
    - 'post_contact_tile': The tile where 'post_contact_waypoint' should be in. This is selected from candidate tiles '[x][i]' marked on the image, where [x] is the column index as a lower letter and [i] is the row index as an integer.
    - 'pre_contact_height': The height of 'pre_contact_waypoint' as one of the two options 'same' (same as the height of making contact with 'target_keypoint') or 'above' (higher than the height of making contact with 'target_keypoint').
    - 'post_contact_height': The height of 'post_contact_waypoint' as one of the two options 'same' (same as the height of making contact with 'target_keypoint') or 'above' (higher than the height of making contact with 'target_keypoint').
    - 'target_angle': Describe how the object should be oriented during this motion in terms of the axis pointing from the grasping point to the function point. Think about this step by step. First analyze whether this axis should be parallel with or perpendicular to the motion direction and the table surface respectively to better perform the task, then choose the axis orientation from one of these strings based on the motion direction: 'forward' (toward the top side of the image), 'backward' (toward the bottom side of the image), 'upward' (perpendicular to and away from the table surface), 'downward' (perpendicular to and towards the table surface), 'left' (towards the left side of the image), 'right' (towards the right side of the image).

Think about this step by step: First, choose 'grasp_keypoint', 'function_keypoint', and 'target_keypoint' on the correct parts of the objects. Next, describe which tile 'target_keypoint' is located in ('target_tile'). Then choose 'pre_contact_tile', 'post_contact_tile', 'pre_contact_height', 'post_contact_height' accordingly such that the resultant motion of 'pre_contact_waypoint'-> 'target_keypoint' -> 'post_contact_waypoint' in 3D follows the input 'motion_direction' and proper contacts will be made. Remember that the columns are marked as {columns} from left to right, and the rows are marked as {rows} from bottom to top. One tile is about {tile_cm:.0f} cm across. Explain the reasoning steps, then give the dictionary in a ```json block.

Task: {task}"""


class Output(TypedDict):
    route: str
    detail: str
    plan: dict[str, Any]
    marked_image: np.ndarray
    motion_image: np.ndarray


# ---------------------------------------------------------------------------
# Camera geometry
# ---------------------------------------------------------------------------


def _camera(observation: dict[str, Any], name: str) -> dict[str, Any]:
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    if not cameras:
        raise ValueError("the observation carries no camera")
    if name:
        for camera in cameras:
            if camera.get("name") == name:
                return camera
        if len(cameras) > 1:
            raise ValueError(f"no camera named {name!r} (has {[c.get('name') for c in cameras]})")
        return cameras[0]
    for camera in cameras:
        if any(k in str(camera.get("name", "")).lower() for k in ("top", "overhead")):
            return camera
    return cameras[0]


def _pose_matrix(pose: Any) -> np.ndarray:
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


class _View:
    """One RGB-D camera: pixel <-> world (OpenCV axes, camera-to-world pose)."""

    def __init__(self, camera: dict[str, Any]) -> None:
        self.rgb = np.asarray(camera["rgb"], dtype=np.uint8)
        self.depth = np.asarray(camera["depth"], dtype=np.float64).reshape(self.rgb.shape[:2])
        self.k = np.asarray(camera["intrinsics"], dtype=np.float64)
        self.t = _pose_matrix(camera["pose"])
        self.h, self.w = self.depth.shape

    def valid(self) -> np.ndarray:
        d = self.depth
        return np.isfinite(d) & (d > 0.05) & (d < 5.0)

    def to_world(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        cam = np.stack(
            [(u - self.k[0, 2]) * z / self.k[0, 0], (v - self.k[1, 2]) * z / self.k[1, 1], z],
            axis=-1,
        )
        return cam @ self.t[:3, :3].T + self.t[:3, 3]

    def point(self, uv: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray | None:
        """World point under a pixel from the near-side percentile of a small window."""
        u, v = int(round(uv[0])), int(round(uv[1]))
        for half in (DEPTH_WINDOW, 2 * DEPTH_WINDOW):
            y0, y1 = max(0, v - half), min(self.h, v + half + 1)
            x0, x1 = max(0, u - half), min(self.w, u + half + 1)
            ok = self.valid()[y0:y1, x0:x1]
            if mask is not None:
                ok = ok & mask[y0:y1, x0:x1]
            if ok.any():
                z = float(np.percentile(self.depth[y0:y1, x0:x1][ok], DEPTH_PERCENTILE))
                return self.to_world(np.array(uv[0], float), np.array(uv[1], float), np.array(z))
        return None

    def ray_at_height(self, uv: np.ndarray, height: float) -> np.ndarray | None:
        """Where the pixel's ray crosses the horizontal plane z = height."""
        ray = self.t[:3, :3] @ np.array(
            [(uv[0] - self.k[0, 2]) / self.k[0, 0], (uv[1] - self.k[1, 2]) / self.k[1, 1], 1.0]
        )
        if abs(ray[2]) < 1e-6:
            return None
        s = (height - self.t[2, 3]) / ray[2]
        return None if s <= 0 else self.t[:3, 3] + s * ray

    def project(self, xyz: np.ndarray) -> np.ndarray:
        cam = (np.asarray(xyz, float) - self.t[:3, 3]) @ self.t[:3, :3]
        return np.array(
            [
                self.k[0, 0] * cam[0] / cam[2] + self.k[0, 2],
                self.k[1, 1] * cam[1] / cam[2] + self.k[1, 2],
            ]
        )


# ---------------------------------------------------------------------------
# Candidate keypoints (MOKA keypoint.py)
# ---------------------------------------------------------------------------


def _fps(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Farthest point sampling from points[0] (MOKA's fps)."""
    points = np.asarray(points, dtype=np.float64)
    n_samples = min(n_samples, len(points))
    left = np.arange(1, len(points))
    chosen = [0]
    dists = np.full(len(points), np.inf)
    for _ in range(1, n_samples):
        last = chosen[-1]
        dists[left] = np.minimum(dists[left], ((points[last] - points[left]) ** 2).sum(-1))
        pick = int(np.argmax(dists[left]))
        chosen.append(int(left[pick]))
        left = np.delete(left, pick)
    return points[chosen]


def _mask_keypoints(mask: np.ndarray, num: int) -> np.ndarray:
    """MOKA's get_keypoints_from_segmentation: the centre first, then farthest
    points of the mask's outline, sorted by image row. The centre is snapped to
    the nearest mask pixel (MOKA kept the outline mean, which can miss a ring)."""
    ys, xs = np.nonzero(mask)
    centre = np.array([xs.mean(), ys.mean()])
    near = int(np.argmin((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2))
    centre = np.array([float(xs[near]), float(ys[near])])
    inner = mask.copy()
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    by, bx = np.nonzero(mask & ~inner)
    outline = np.stack([bx, by], axis=1).astype(np.float64)
    kps = _fps(np.concatenate([centre[None], outline], axis=0), num)
    return np.concatenate([kps[:1], kps[1:][np.argsort(kps[1:, 1], kind="stable")]], axis=0)


def _as_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.size == 0:
        return None
    m = m.reshape(m.shape[-2:]) if m.ndim > 2 else m
    if m.shape != shape:
        raise ValueError(f"mask {m.shape} and image {shape} differ in size")
    m = m > 0
    return m if m.any() else None


def _candidates(
    ctx: NodeContext,
    view: _View,
    name: str,
    mask: Any,
    source: str,
    num: int,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """(pixel keypoints [N, 2], mask or None, which source) for one named object."""
    given = _as_mask(mask, (view.h, view.w))
    if given is not None:
        return _mask_keypoints(given, num), given, "mask"
    if source in ("auto", "sam3"):
        try:
            found = ctx.tool("sam3.segment_text", image=view.rgb, query=name, max_results=3)
            for m, score in zip(found.get("masks") or [], found.get("scores") or [], strict=False):
                if float(score) >= SAM_SCORE_FLOOR:
                    seg = _as_mask(m, (view.h, view.w))
                    if seg is not None:
                        return _mask_keypoints(seg, num), seg, "sam3"
            return None, None, "sam3"
        except Exception as exc:  # sam3 not served: fall through to pointing
            if source == "sam3":
                raise
            logger.info("executing-keypoint-motion: sam3 unavailable (%s), pointing instead", exc)
    reply = ctx.tool(
        "gemini-er.point",
        image=view.rgb,
        query=f"the {name}: points spread over it, on its ends, corners and centre",
        max_points=int(num),
    )
    points = [(float(p["pixel_x"]), float(p["pixel_y"])) for p in (reply.get("points") or [])]
    if not reply.get("found", bool(points)) or not points:
        return None, None, "gemini-er"
    pts = np.array(points[:num], dtype=np.float64)
    centre = pts.mean(axis=0)
    first = int(np.argmin(((pts - centre) ** 2).sum(1)))
    pts = np.concatenate([pts[[first]], np.delete(pts, first, axis=0)], axis=0)
    return (
        np.concatenate([pts[:1], pts[1:][np.argsort(pts[1:, 1], kind="stable")]], axis=0),
        None,
        "gemini-er",
    )


# ---------------------------------------------------------------------------
# VLM reply parsing (MOKA visual_prompt_utils.py)
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> dict[str, Any] | None:
    """MOKA's parse_json_string + remove_trailing_comments, tolerant of fences."""
    blocks = re.findall(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.S)
    candidates = blocks[::-1] + [text]
    for raw in candidates:
        lines = []
        for line in raw.split("\n"):
            cut = line.find("//")
            lines.append((line[:cut] if cut != -1 else line).strip())
        cleaned = "\n".join(lines)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            value = json.loads(cleaned[start : end + 1])
        except ValueError:
            continue
        if isinstance(value, list) and value:
            value = value[0]
        if isinstance(value, dict):
            return value
    return None


def _label_index(value: Any, prefix: str, count: int) -> int | None:
    """'P3' -> 2; '' or None -> None; raises on a label that is not offered."""
    s = str(value or "").strip().strip("'\"").upper()
    if s in ("", "NONE", "NULL"):
        return None
    m = re.fullmatch(rf"{prefix}\s*\[?(\d+)\]?", s)
    if not m or not 1 <= int(m.group(1)) <= count:
        raise ValueError(f"{value!r} is not one of {prefix}1..{prefix}{count}")
    return int(m.group(1)) - 1


def _tile(value: Any, grid: int) -> tuple[int, int] | None:
    """'b4' -> (column 1, row-from-top grid - 4); '' -> None (MOKA's tile maths)."""
    s = str(value or "").strip().strip("'\"").lower()
    if s in ("", "none", "null"):
        return None
    m = re.fullmatch(r"([a-z])\s*\[?(\d+)\]?", s)
    if not m:
        raise ValueError(f"{value!r} is not a tile name")
    col, row = string.ascii_lowercase.index(m.group(1)), int(m.group(2))
    if col >= grid or not 1 <= row <= grid:
        raise ValueError(f"tile {value!r} is outside the {grid} x {grid} grid")
    return col, grid - row


def _height(value: Any, default: str) -> str:
    s = str(value or "").strip().lower()
    return s if s in ("same", "above") else default


# ---------------------------------------------------------------------------
# Drawing (MOKA annotate_visual_prompts / annotate_motion)
# ---------------------------------------------------------------------------


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, size: int, fill) -> None:
    draw.text(xy, text, fill=fill, font=_font(size), stroke_width=2, stroke_fill=(0, 0, 0))


def _marked_crop(
    view: _View,
    box: tuple[int, int, int],
    grid: int,
    marks: list[tuple[str, np.ndarray, tuple[int, int, int]]],
) -> np.ndarray:
    x0, y0, side = box
    crop = Image.fromarray(view.rgb[y0 : y0 + side, x0 : x0 + side]).resize(
        (RENDER_SIZE, RENDER_SIZE), Image.BICUBIC
    )
    draw = ImageDraw.Draw(crop, "RGBA")
    scale = RENDER_SIZE / side
    step = RENDER_SIZE / grid
    for i in range(1, grid):
        draw.line([(i * step, 0), (i * step, RENDER_SIZE)], fill=(0, 0, 0, 110), width=2)
        draw.line([(0, i * step), (RENDER_SIZE, i * step)], fill=(0, 0, 0, 110), width=2)
    for col in range(grid):
        for row in range(grid):
            label = f"{string.ascii_lowercase[col]}{grid - row}"
            _text(draw, (col * step + 4, row * step + 2), label, 18, (255, 255, 255))
    for label, uv, color in marks:
        x, y = (uv[0] - x0) * scale, (uv[1] - y0) * scale
        draw.ellipse(
            [x - 9, y - 9, x + 9, y + 9], fill=(*color, 120), outline=(0, 0, 0, 255), width=2
        )
        tx = min(max(x + 10, 0), RENDER_SIZE - 40)
        ty = min(max(y - 28, 0), RENDER_SIZE - 24)
        _text(draw, (tx, ty), label, 20, (*color, 255))
    return np.asarray(crop.convert("RGB"))


def _motion_image(view: _View, picks: dict[str, np.ndarray | None]) -> np.ndarray:
    img = Image.fromarray(view.rgb.copy())
    draw = ImageDraw.Draw(img, "RGBA")
    colors = {
        "grasp": (255, 0, 0),
        "function": (255, 220, 0),
        "target": (0, 90, 255),
        "pre_contact": (0, 230, 230),
        "post_contact": (0, 230, 230),
    }
    path = [picks[k] for k in ("pre_contact", "target", "post_contact") if picks.get(k) is not None]
    for a, b in zip(path, path[1:], strict=False):
        draw.line([tuple(a), tuple(b)], fill=(0, 230, 230, 200), width=2)
    for name, uv in picks.items():
        if uv is None:
            continue
        c = colors[name]
        draw.ellipse(
            [uv[0] - 5, uv[1] - 5, uv[0] + 5, uv[1] + 5],
            fill=(*c, 150),
            outline=(0, 0, 0, 255),
            width=1,
        )
        _text(draw, (uv[0] + 6, uv[1] - 16), name, 12, (*c, 255))
    return np.asarray(img)


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------


def _vec(v: np.ndarray | None) -> dict[str, float] | None:
    return None if v is None else {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _px(v: np.ndarray | None) -> list[float] | None:
    return None if v is None else [float(v[0]), float(v[1])]


def _query(ctx: NodeContext, tool: str, prompt: str, image: np.ndarray, model: str) -> str:
    kwargs: dict[str, Any] = {"prompt": prompt, "image": image}
    if model:
        kwargs["model"] = model
    reply = ctx.tool(tool, **kwargs)
    return str(reply.get("text", "") if isinstance(reply, dict) else reply)


def _crop_box(
    view: _View, world: np.ndarray, pixels: np.ndarray, grid: int, tile: float, margin: float
):
    """Square crop (x0, y0, side px) centred on the keypoints, >= grid x tile metres across."""
    centre_w = world.mean(axis=0)
    span = float(np.max(np.ptp(world[:, :2], axis=0))) if len(world) > 1 else 0.0
    metres = max(grid * tile, span + 2.0 * margin)
    zc = float((centre_w - view.t[:3, 3]) @ view.t[:3, 2])
    side = int(round(metres * view.k[0, 0] / max(zc, 0.05)))
    side = max(grid * 8, min(side, view.h, view.w))
    centre_px = pixels.mean(axis=0)
    x0 = int(round(min(max(centre_px[0] - side / 2, 0), view.w - side)))
    y0 = int(round(min(max(centre_px[1] - side / 2, 0), view.h - side)))
    return x0, y0, side, metres


def _support_height(heights: np.ndarray, lowest_keypoint: float) -> float:
    """The fullest SUPPORT_BIN of the crop's heights within SUPPORT_BAND of the lowest keypoint."""
    lo, hi = lowest_keypoint - SUPPORT_BAND[0], lowest_keypoint + SUPPORT_BAND[1]
    z = heights[(heights >= lo) & (heights <= hi)]
    if len(z) < 20:
        return lowest_keypoint
    counts, edges = np.histogram(
        z, bins=max(1, int(math.ceil((hi - lo) / SUPPORT_BIN))), range=(lo, hi)
    )
    k = int(np.argmax(counts))
    return float(np.median(z[(z >= edges[k]) & (z <= edges[k + 1])]))


def _principal_heading(points_xy: np.ndarray) -> float | None:
    """Heading [deg] of the minor axis of a set of xy points (across the object)."""
    if len(points_xy) < 3:
        return None
    centred = points_xy - points_xy.mean(axis=0)
    vals, vecs = np.linalg.eigh(centred.T @ centred)
    if vals[1] <= 1e-12 or vals[0] / vals[1] > 0.6:
        return None
    minor = vecs[:, 0]
    return math.degrees(math.atan2(minor[1], minor[0]))


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    instruction: str,
    object_grasped: str | None = None,
    object_unattached: str | None = None,
    motion_direction: str = "",
    camera_name: str = "",
    grasped_mask: Any = None,
    unattached_mask: Any = None,
    candidate_source: str = "auto",
    num_candidates: int = NUM_CANDIDATES,
    grid_size: int = GRID_SIZE,
    tile_size: float = TILE_SIZE,
    crop_margin: float = CROP_MARGIN,
    vlm_tool: str = "vlm.query",
    vlm_model: str = "",
    attempts: int = ATTEMPTS,
    z_offset: float = Z_OFFSET,
    hover_clearance: float = HOVER_CLEARANCE,
    contact_fraction: float = CONTACT_FRACTION,
) -> Output:
    view = _View(_camera(observation, camera_name))
    out: Output = {
        "route": "no_plan",
        "detail": "",
        "plan": {},
        "marked_image": np.zeros((1, 1, 3), np.uint8),
        "motion_image": view.rgb,
    }

    def no_plan(why: str) -> Output:
        out["detail"] = why
        return out

    # 1. The task's objects and direction (MOKA's propose_subtasks, one step).
    task = {
        "instruction": instruction,
        "object_grasped": object_grasped or "",
        "object_unattached": object_unattached or "",
        "motion_direction": motion_direction,
    }
    if object_grasped is None and object_unattached is None:
        parsed = None
        for _ in range(max(1, int(attempts))):
            text = _query(
                ctx, vlm_tool, f"{SUBTASK_PROMPT}\n\nTask: {instruction}", view.rgb, vlm_model
            )
            parsed = _parse_json(text)
            if parsed is not None:
                break
        if parsed is None:
            return no_plan("the VLM gave no parsable task description")
        for key in ("object_grasped", "object_unattached", "motion_direction"):
            task[key] = str(parsed.get(key) or task[key] or "").strip()
    if not task["object_grasped"] and not task["object_unattached"]:
        return no_plan("the task names neither an object to grasp nor one to act on")

    # 2. Candidate keypoints on each object.
    names = {"grasped": task["object_grasped"], "unattached": task["object_unattached"]}
    given = {"grasped": grasped_mask, "unattached": unattached_mask}
    kps: dict[str, np.ndarray | None] = {"grasped": None, "unattached": None}
    masks: dict[str, np.ndarray | None] = {"grasped": None, "unattached": None}
    sources: dict[str, str] = {}
    for role, name in names.items():
        if not name:
            continue
        pts, mask, src = _candidates(
            ctx, view, name, given[role], candidate_source, int(num_candidates)
        )
        sources[role] = src
        if pts is None:
            return no_plan(f"{src} found no {name!r} in the {view.w}x{view.h} frame")
        kps[role], masks[role] = pts, mask
    world_kps: dict[str, list[np.ndarray | None]] = {
        role: [view.point(p, masks[role]) for p in pts] if pts is not None else []
        for role, pts in kps.items()
    }
    all_px = np.concatenate([p for p in kps.values() if p is not None], axis=0)
    all_w = [w for ws in world_kps.values() for w in ws if w is not None]
    if not all_w:
        return no_plan("no candidate keypoint has valid depth")

    # 3. The marked crop: P (red) / Q (blue) marks and the lettered grid.
    grid = int(grid_size)
    x0, y0, side, metres = _crop_box(
        view, np.array(all_w), all_px, grid, float(tile_size), float(crop_margin)
    )
    marks = []
    for role, prefix, color in (
        ("grasped", "P", (230, 30, 30)),
        ("unattached", "Q", (40, 90, 255)),
    ):
        if kps[role] is not None:
            marks += [(f"{prefix}{i + 1}", p, color) for i, p in enumerate(kps[role])]
    marked = _marked_crop(view, (x0, y0, side), grid, marks)
    out["marked_image"] = marked
    letters = string.ascii_lowercase[:grid]
    prompt = SELECT_MOTION_PROMPT.format(
        columns=", ".join(f"'{c}'" for c in letters),
        rows=", ".join(str(i) for i in range(1, grid + 1)),
        tile_cm=100.0 * metres / grid,
        task=json.dumps(task),
    )

    # 4. Ask, parse and check (MOKA's request_motion), a few times at most.
    n_p = 0 if kps["grasped"] is None else len(kps["grasped"])
    n_q = 0 if kps["unattached"] is None else len(kps["unattached"])
    reply, choice, why = "", None, "no reply"
    for _ in range(max(1, int(attempts))):
        reply = _query(ctx, vlm_tool, prompt, marked, vlm_model)
        ctx_json = _parse_json(reply)
        if ctx_json is None:
            why = "the VLM reply held no JSON dictionary"
            continue
        try:
            grasp_i = _label_index(ctx_json.get("grasp_keypoint"), "P", n_p)
            function_i = _label_index(ctx_json.get("function_keypoint"), "P", n_p)
            target_i = _label_index(ctx_json.get("target_keypoint"), "Q", n_q)
            pre_tile = _tile(ctx_json.get("pre_contact_tile"), grid)
            post_tile = _tile(ctx_json.get("post_contact_tile"), grid)
        except ValueError as exc:
            why = f"the VLM picked an unoffered mark: {exc}"
            continue
        if names["grasped"] and grasp_i is None:
            why = "the VLM gave no grasp keypoint for the object to grasp"
            continue
        if target_i is None and pre_tile is None:
            why = "the VLM gave neither a target keypoint nor a waypoint"
            continue
        choice = (ctx_json, grasp_i, function_i, target_i, pre_tile, post_tile)
        break
    out["plan"] = {"task": task, "reply": reply}
    if choice is None:
        return no_plan(why)
    ctx_json, grasp_i, function_i, target_i, pre_tile, post_tile = choice
    if not names["grasped"]:
        grasp_i = function_i = None
    if not names["unattached"]:
        target_i = None
    if grasp_i is None:
        function_i = None

    def pixel_of_tile(t: tuple[int, int] | None) -> np.ndarray | None:
        if t is None:
            return None
        col, row = t
        step = side / grid
        return np.array([x0 + (col + 0.5) * step, y0 + (row + 0.5) * step])

    picks_px = {
        "grasp": None if grasp_i is None else kps["grasped"][grasp_i],
        "function": None if function_i is None else kps["grasped"][function_i],
        "target": None if target_i is None else kps["unattached"][target_i],
        "pre_contact": pixel_of_tile(pre_tile),
        "post_contact": pixel_of_tile(post_tile),
    }
    grasp_w = None if grasp_i is None else world_kps["grasped"][grasp_i]
    function_w = None if function_i is None else world_kps["grasped"][function_i]
    target_w = None if target_i is None else world_kps["unattached"][target_i]
    for label, idx, w in (
        ("grasp", grasp_i, grasp_w),
        ("function", function_i, function_w),
        ("target", target_i, target_w),
    ):
        if idx is not None and w is None:
            return no_plan(f"the {label} keypoint has no valid depth")

    # 5. Heights (MOKA's compute_context_3d and policy height rules).
    crop_valid = view.valid()[y0 : y0 + side, x0 : x0 + side]
    vv, uu = np.nonzero(crop_valid)
    crop_world = view.to_world(
        uu + x0 + 0.0, vv + y0 + 0.0, view.depth[y0 : y0 + side, x0 : x0 + side][crop_valid]
    )
    support_z = _support_height(crop_world[:, 2], min(float(w[2]) for w in all_w))
    pre_height = _height(ctx_json.get("pre_contact_height"), "same")
    post_height = _height(ctx_json.get("post_contact_height"), "same")
    top = max(float(w[2]) for w in all_w)
    safe_z = max(support_z + float(hover_clearance), top + OBJECT_CLEARANCE)
    if target_w is None:
        mode = "move_after_grasp"
        contact_z = float(grasp_w[2]) if grasp_w is not None else support_z
    elif grasp_w is not None and function_w is None:
        mode = "pick_and_place"
        contact_z = float(target_w[2]) + (float(grasp_w[2]) - support_z) + float(z_offset)
    elif pre_height == "above":
        mode = "press" if function_w is None else "tool_press"
        contact_z = float(target_w[2]) + float(z_offset)
    else:
        mode = "push" if function_w is None else "tool_push"
        contact_z = support_z + float(contact_fraction) * (float(target_w[2]) - support_z)
    if target_w is not None:
        target_w = np.array([target_w[0], target_w[1], contact_z])

    def waypoint(uv: np.ndarray | None, height: str) -> np.ndarray | None:
        if uv is None:
            return None
        xy = view.ray_at_height(uv, contact_z)
        if xy is None:
            return None
        return np.array([xy[0], xy[1], safe_z if height == "above" else contact_z])

    pre_w = waypoint(picks_px["pre_contact"], pre_height)
    post_w = waypoint(picks_px["post_contact"], post_height)

    # 6. The hand's heading: across the object at the grasp, across the push otherwise.
    grasp_heading = None
    if grasp_w is not None:
        if masks["grasped"] is not None:
            mv, mu = np.nonzero(masks["grasped"] & view.valid())
            pts = view.to_world(mu + 0.0, mv + 0.0, view.depth[mv, mu])
            near = pts[np.linalg.norm(pts[:, :2] - grasp_w[:2], axis=1) < GRASP_PCA_RADIUS]
            grasp_heading = _principal_heading(near[:, :2]) if len(near) else None
        else:
            pts = np.array([w for w in world_kps["grasped"] if w is not None])
            grasp_heading = _principal_heading(pts[:, :2]) if len(pts) else None
    push_heading = None
    if target_w is not None and pre_w is not None:
        d = target_w[:2] - pre_w[:2]
        if float(np.linalg.norm(d)) > 1e-3:
            push_heading = math.degrees(math.atan2(d[1], d[0]))

    plan = {
        "task": task,
        "mode": mode,
        "labels": {
            k: str(ctx_json.get(k, ""))
            for k in (
                "grasp_keypoint",
                "function_keypoint",
                "target_keypoint",
                "target_tile",
                "pre_contact_tile",
                "post_contact_tile",
                "target_angle",
            )
        },
        "grasp": _vec(grasp_w),
        "function": _vec(function_w),
        "target": _vec(target_w),
        "pre_contact": _vec(pre_w),
        "post_contact": _vec(post_w),
        "pre_contact_height": pre_height,
        "post_contact_height": post_height,
        "contact_z": contact_z,
        "support_z": support_z,
        "safe_z": safe_z,
        "z_offset": float(z_offset),
        "grasp_heading_deg": grasp_heading,
        "push_heading_deg": push_heading,
        "target_angle": str(ctx_json.get("target_angle", "")),
        "pixels": {k: _px(v) for k, v in picks_px.items()},
        "crop": [x0, y0, side],
        "tile_m": metres / grid,
        "candidate_sources": sources,
        "reply": reply,
    }
    out["plan"] = plan
    out["motion_image"] = _motion_image(view, picks_px)
    out["route"] = "planned"
    out["detail"] = (
        f"{mode}: grasp {plan['labels']['grasp_keypoint'] or '-'}, function "
        f"{plan['labels']['function_keypoint'] or '-'}, target {plan['labels']['target_keypoint'] or '-'}, "
        f"tiles {plan['labels']['pre_contact_tile'] or '-'} -> {plan['labels']['post_contact_tile'] or '-'}"
    )
    return out
