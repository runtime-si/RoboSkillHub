"""robobrain tool bundle: BAAI RoboBrain 2.5 keypoint traces, pointing and task progress.

Runs BAAI's RoboBrain 2.5-4B (https://huggingface.co/BAAI/RoboBrain2.5-4B, Apache-2.0,
a Qwen3-VL-4B fine-tune) with plain ``transformers``, the way the reference
``inference.py`` of https://github.com/FlagOpen/RoboBrain2.5 (Apache-2.0) does.
The weights come from the Hugging Face Hub at a pinned commit on first use.

Copied (Apache-2.0, see NOTICE): the pointing and end-effector trace prompts of
RoboBrain2.5 ``inference.py`` @ af98c93, the object-centric trace prompt of
RoboTracer ``Evaluation/test_benchmark.py`` @ 146a39b (whose spatial-trace
training went into RoboBrain 2.5), and the progress prompt and score parsing of
Robo-Dopamine ``examples/inference.py`` @ 2c714ab (RoboBrain 2.5's progress
model). Changed: images are passed as arrays instead of file paths, decoding is
greedy, and the trace can be started at a given pixel by writing the start of
the answer for the model (see :func:`trace`).

The model answers in normalized coordinates, ``(x, y)`` / ``(x, y, d)`` with
x, y in 0-1000 across the image's width and height and d the depth in metres
along the camera's optical axis (RoboTracer's evaluation back-projects it as
Z). They are converted to pixels here and, with the camera intrinsics, to
camera points (and world points with the camera pose).

``robobrain.trace`` returns the hub's visual-plan structure, the one
``molmo-motion.forecast`` documents (``tracks`` of pixels / camera / world
points, ``frame``, ``backend``), so a skill switches trajectory planner by
tool name. ``robobrain.point`` returns ``gemini-er.point``'s shape.

The model loads lazily on first call (module-level singleton); importing this
module never imports torch or transformers. Knobs via env:

- ``GAP_ROBOBRAIN_DEVICE`` -- torch device (default ``cuda``).
- ``GAP_ROBOBRAIN_MODEL`` -- a local model directory instead of the pinned Hub
  snapshot (e.g. a RoboBrain2.5-8B-NV download; only the 4B was checked).
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import tool
from gap_core.types import Se3Pose, pose_to_matrix

logger = logging.getLogger(__name__)

MODEL_REPO = "BAAI/RoboBrain2.5-4B"
#: Hub commit of the checkpoint, checked live on 2026-11-07 (card updated 2026-09-10).
MODEL_REVISION = "ec4b3bf5e969b48ad17df432b479c10402a0079b"

#: Upstream inference.py's generation budget.
_MAX_NEW_TOKENS = 768
#: Robo-Dopamine's budget for the one-line ``<score>`` answer.
_PROGRESS_MAX_NEW_TOKENS = 64

# RoboBrain2.5 inference.py @ af98c93, task="pointing".
_POINT_PROMPT = (
    "{query}. Please provide its 2D coordinates. Your answer should be formatted as a tuple, "
    "i.e. [(x, y)], where the tuple contains the x and y coordinates of a point satisfying the "
    "conditions above."
)
# RoboBrain2.5 inference.py @ af98c93, task="trajectory".
_TRACE_PROMPT_END_EFFECTOR = (
    "Please predict 3D end-effector-centric waypoints to complete the task successfully. "
    'The task is "{instruction}". Your answer should be formatted as a list of tuples, i.e., '
    "[(x1, y1, d1), (x2, y2, d2), ...], where each tuple contains the x and y coordinates and "
    "the depth of the point."
)
# RoboTracer Evaluation/test_benchmark.py @ 146a39b, task "3D".
_TRACE_PROMPT_OBJECT = (
    'Point the 3D object-centric visual trace for the task "{instruction}". Your answer should '
    "be formatted as a list of tuples, i.e., [(x1, y1, d1), (x2, y2, d2), ...], where each tuple "
    "contains the x and y coordinates of the point, and d is the depth of the point."
)
TRACE_MODES = {"end_effector": _TRACE_PROMPT_END_EFFECTOR, "object": _TRACE_PROMPT_OBJECT}

# Robo-Dopamine examples/inference.py @ 2c714ab, SYSTEM_PROMPT (images at the <image> marks).
_PROGRESS_PROMPT = """
You are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.

<Task>
`{task}`

REFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's actual START/END):
- REFERENCE START — Robot Front Image (task just starting): <image>
- REFERENCE END — Robot Front Image (task fully completed): <image>
</Task>

BEFORE Robot Front Image: <image>
BEFORE Robot Left Wrist Image: <image>
BEFORE Robot Right Wrist Image: <image>

AFTER Robot Front Image: <image>
AFTER Robot Left Wrist Image: <image>
AFTER Robot Right Wrist Image: <image>

Goal
Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing the task than BEFORE, using the REFERENCE START/END images as conceptual anchors.

Progress Estimation (no formulas)
1) Calibrate using the references:
   - REFERENCE START = “just beginning”; REFERENCE END = “fully completed.”
   - Visually estimate how far BEFORE and AFTER are along this START→END continuum.
2) Direction:
   - AFTER better than BEFORE → positive score.
   - AFTER worse than BEFORE → negative score.
   - Essentially the same → 0.
3) Normalize to an integer percentage in [-100%, +100%]:
   - For improvements, scale the improvement relative to what remained from BEFORE to END.
   - For regressions, scale the deterioration relative to how far BEFORE had progressed from START.
   - Clip to [-100%, +100%] and round to the nearest integer percent.

Evaluation Criteria (apply across all three views)
1) Task Alignment: Evidence directly tied to `{task}`.
2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.
3) View-Specific Evidence & Consistency:
   - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.
   - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).
   - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.
   - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.
4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.
5) Ambiguity: If evidence is genuinely inconclusive or conflicting without decisive cues, treat progress as unchanged → 0%.

Output Format (STRICT)
Return ONLY one line containing the score wrapped in <score> tags, as an integer percentage with a percent sign:
<score>+NN%</score>  or  <score>-NN%</score>  or  <score>0%</score>
"""
#: Robo-Dopamine's per-image pixel budget for its 8-image prompt (max_pixels=76800).
_PROGRESS_MAX_PIXELS = 76800
#: Robo-Dopamine's examples/blank_goal.png: a uniform (237, 237, 237) 274 x 170 image,
#: its stand-in for "no goal image".
_BLANK_GOAL = np.full((170, 274, 3), 237, dtype=np.uint8)

#: Half-size of the window a start pixel's measured depth is the median of.
_DEPTH_WINDOW_PX = 3

_NUM = r"[-+]?\d+(?:\.\d+)?"
_TUPLE_RE = re.compile(rf"[\(\[]\s*({_NUM})\s*,\s*({_NUM})\s*(?:,\s*({_NUM})\s*)?[\)\]]")
_SCORE_RE = re.compile(r"<score>\s*([-+]?\d+(?:\.\d+)?)\s*%?\s*</score>")

_load_lock = threading.Lock()
_runtime: Any = None


class Point(TypedDict):
    pixel_x: float
    pixel_y: float
    label: str


class PointResult(TypedDict):
    pixel_x: float
    pixel_y: float
    found: bool
    points: list[Point]
    raw: str  # the model's answer text


class PlanTrack(TypedDict):
    query_pixel: list[float]  # [u, v] where the trace starts (given, else the first keypoint)
    pixels: list[list[float]]  # [F][u, v], the keypoints after the start, in order
    start_camera: list[float]  # [x, y, z] m, camera frame; [] without intrinsics
    points_camera: list[list[float]]  # [F][x, y, z] m; [] without intrinsics
    start_world: list[float]  # [x, y, z] m; [] without intrinsics + camera_pose
    points_world: list[list[float]]  # [F][x, y, z] m; [] without intrinsics + camera_pose
    predicted: list[bool]  # [F] the model emitted this keypoint (always true here)
    start_depth_m: float | None  # depth of the start along the optical axis
    depths_m: list[float | None]  # [F] the model's depth per keypoint (None if it gave none)


class PlanTiming(TypedDict):
    load_s: float  # model load paid by this call (0 once resident)
    prepare_s: float  # chat template, image preprocessing, upload
    generate_s: float  # the model's decode
    total_s: float


class PlanResult(TypedDict):
    found: bool  # the trace has at least one keypoint after its start
    tracks: list[PlanTrack]  # one: the trace
    horizon: int  # F
    dt_s: float | None  # None: ordered keypoints, not timed steps
    frame: str  # "world" | "camera" | "pixel": the richest frame the tracks carry
    backend: str  # "robobrain"
    model: str  # repo@revision (or the local directory)
    timing: PlanTiming
    raw: str  # the model's answer text, including any start written for it
    mode: str  # "end_effector" | "object"
    depth_source: str  # "model" | "model, start from depth map" | "none"
    measured_start_depth_m: float | None  # the depth map at the start pixel, when given


class ProgressResult(TypedDict):
    progress: float  # 0..1, the score clipped (forward mode: the score is the progress)
    score: float  # -1..1 as the model answered (+NN% -> NN / 100)
    found: bool  # a <score> tag was parsed
    raw: str
    timing: PlanTiming


# ---------------------------------------------------------------------------
# Geometry and parsing (numpy only)
# ---------------------------------------------------------------------------


def _validate_image(image: Any, what: str = "image") -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"robobrain expects {what} as a uint8 [H, W, 3] RGB array, got "
            f"dtype={arr.dtype} shape={arr.shape}"
        )
    return np.ascontiguousarray(arr)


def _parse_tuples(answer: str) -> list[tuple[float, float, float | None]]:
    """Every ``(x, y)`` / ``(x, y, d)`` (or bracketed) tuple in the answer, in order."""
    out = []
    for x, y, d in _TUPLE_RE.findall(answer or ""):
        out.append((float(x), float(y), float(d) if d else None))
    return out


def _to_pixel(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return x / 1000.0 * width, y / 1000.0 * height


def _to_normalized(u: float, v: float, width: int, height: int) -> tuple[int, int]:
    return int(round(u / width * 1000.0)), int(round(v / height * 1000.0))


def _depth_at(depth: np.ndarray, u: float, v: float) -> float:
    """Median of the valid depths in a small window around (u, v); NaN if none."""
    h, w = depth.shape
    cu, cv = int(round(u)), int(round(v))
    r = _DEPTH_WINDOW_PX
    patch = depth[max(0, cv - r) : min(h, cv + r + 1), max(0, cu - r) : min(w, cu + r + 1)]
    good = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(good)) if good.size else float("nan")


def _deproject(pixels: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    x = (pixels[:, 0] - K[0, 2]) / K[0, 0] * z
    y = (pixels[:, 1] - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=-1)


def _to_world(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    return xyz @ T[:3, :3].T + T[:3, 3]


def _camera_matrix(camera_pose: Se3Pose | np.ndarray) -> np.ndarray:
    """A camera-to-world 4x4 from either an ``Se3Pose`` dict or a 4x4 array."""
    if isinstance(camera_pose, dict):
        return pose_to_matrix(camera_pose)
    return np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)


def _round(rows: Any, nd: int) -> list:
    return np.round(np.asarray(rows, dtype=np.float64), nd).tolist()


def _fit_pixels(image: np.ndarray, max_pixels: int) -> np.ndarray:
    """The image scaled down (aspect kept) to at most ``max_pixels`` pixels."""
    h, w = image.shape[:2]
    if h * w <= max_pixels:
        return image
    from PIL import Image

    s = (max_pixels / float(h * w)) ** 0.5
    size = (max(1, int(w * s)), max(1, int(h * s)))
    return np.asarray(Image.fromarray(image, "RGB").resize(size, Image.BICUBIC))


# ---------------------------------------------------------------------------
# Runtime: transformers over the pinned snapshot
# ---------------------------------------------------------------------------


def _model_id() -> str:
    return os.environ.get("GAP_ROBOBRAIN_MODEL") or f"{MODEL_REPO}@{MODEL_REVISION}"


def _model_dir() -> str:
    override = os.environ.get("GAP_ROBOBRAIN_MODEL")
    if override:
        return override
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION, repo_type="model")


class _Runtime:
    """The processor and the bf16 model, built once and kept resident."""

    def __init__(self, model_dir: str, device: str) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = torch.device(device)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map=str(self.device)
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_dir)

    def generate(
        self,
        segments: list[str | np.ndarray],
        *,
        answer_prefix: str = "",
        max_new_tokens: int = _MAX_NEW_TOKENS,
    ) -> tuple[str, float, float]:
        """``(answer, prepare_s, generate_s)`` for one user turn of text and images.

        ``answer_prefix`` is written as the start of the assistant's answer;
        the returned answer does not repeat it.
        """
        from PIL import Image

        torch = self.torch
        t0 = time.perf_counter()
        content: list[dict] = []
        images = []
        for seg in segments:
            if isinstance(seg, str):
                content.append({"type": "text", "text": seg})
            else:
                content.append({"type": "image"})
                images.append(Image.fromarray(seg, "RGB"))
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt + answer_prefix], images=images, padding=True, return_tensors="pt"
        ).to(self.device)
        prepare_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = ids[:, inputs["input_ids"].shape[1] :]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (text[0] if text else ""), prepare_s, time.perf_counter() - t1


def _build_runtime() -> _Runtime:
    device = os.environ.get("GAP_ROBOBRAIN_DEVICE", "cuda")
    model_dir = _model_dir()
    logger.info("Loading RoboBrain 2.5 (%s) on %s from %s", _model_id(), device, model_dir)
    return _Runtime(model_dir, device)


def _get_runtime() -> tuple[Any, float]:
    """The resident runtime and the seconds this call spent loading it."""
    global _runtime
    with _load_lock:
        if _runtime is None:
            t0 = time.perf_counter()
            _runtime = _build_runtime()
            return _runtime, time.perf_counter() - t0
        return _runtime, 0.0


def _run(
    tool_name: str, segments: list, *, answer_prefix: str = "", max_new_tokens: int
) -> tuple[str, PlanTiming]:
    t_start = time.perf_counter()
    try:
        runtime, load_s = _get_runtime()
    except Exception as exc:
        raise ToolError(tool_name, f"RoboBrain 2.5 failed to load: {exc}") from exc
    try:
        answer, prepare_s, generate_s = runtime.generate(
            segments, answer_prefix=answer_prefix, max_new_tokens=max_new_tokens
        )
    except Exception as exc:
        raise ToolError(tool_name, f"RoboBrain 2.5 failed: {exc}") from exc
    timing: PlanTiming = {
        "load_s": round(load_s, 3),
        "prepare_s": round(prepare_s, 3),
        "generate_s": round(generate_s, 3),
        "total_s": round(time.perf_counter() - t_start, 3),
    }
    return str(answer or ""), timing


def weights_cached() -> bool | None:
    """Filesystem-only probe for ``gap check``: the pinned snapshot is on disk."""
    override = os.environ.get("GAP_ROBOBRAIN_MODEL")
    if override:
        return os.path.isfile(os.path.join(override, "config.json"))
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        hits = [
            try_to_load_from_cache(MODEL_REPO, f, revision=MODEL_REVISION)
            for f in ("config.json", "model.safetensors")
        ]
    except Exception:
        return None
    return all(isinstance(h, str) for h in hits)


def prefetch() -> None:
    """Download the pinned snapshot (``gap skills check --download``)."""
    logger.info("[robobrain] prefetching %s ...", _model_id())
    _model_dir()
    logger.info("[robobrain] prefetch complete")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="robobrain.point",
    summary="Point at what a phrase names via RoboBrain 2.5 (local GPU); returns pixel points.",
    tags=("perception",),
)
def point(image: np.ndarray, query: str, max_points: int = 10) -> PointResult:
    """Point at *query* in *image*; the drop-in for ``gemini-er.point``.

    Args:
        image: uint8 [H, W, 3] RGB array.
        query: What to point at ("the free end of the white cable").
        max_points: Most points returned (the model's prompt asks for one).

    Returns:
        ``{"pixel_x", "pixel_y", "found", "points", "raw"}``: the first point
        in pixels and every point as ``{"pixel_x", "pixel_y", "label"}``.
        ``found=False`` with zeros and no points when nothing parses.
    """
    if not query or not str(query).strip():
        raise ValueError("robobrain.point needs a query")
    arr = _validate_image(image)
    height, width = arr.shape[:2]
    query = str(query).strip()
    answer, _ = _run(
        "robobrain.point",
        [arr, _POINT_PROMPT.format(query=query)],
        max_new_tokens=_MAX_NEW_TOKENS,
    )
    points: list[Point] = []
    for x, y, _d in _parse_tuples(answer)[: max(1, int(max_points))]:
        u, v = _to_pixel(x, y, width, height)
        points.append(
            {
                "pixel_x": float(np.clip(u, 0.0, width - 1.0)),
                "pixel_y": float(np.clip(v, 0.0, height - 1.0)),
                "label": query,
            }
        )
    if not points:
        logger.debug("robobrain.point: nothing for %r (raw=%r)", query, answer[:200])
        return {"pixel_x": 0.0, "pixel_y": 0.0, "found": False, "points": [], "raw": answer}
    return {
        "pixel_x": points[0]["pixel_x"],
        "pixel_y": points[0]["pixel_y"],
        "found": True,
        "points": points,
        "raw": answer,
    }


@tool(
    name="robobrain.trace",
    summary=(
        "An ordered keypoint trajectory for a manipulation from one frame and an instruction "
        "(RoboBrain 2.5); pixels, plus camera/world points with intrinsics, in the visual-plan schema."
    ),
    tags=("perception", "planning"),
)
def trace(
    image: np.ndarray,
    instruction: str,
    start_point: list[float] | None = None,
    depth: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    camera_pose: Se3Pose | np.ndarray | None = None,
    mode: str = "end_effector",
) -> PlanResult:
    """Ask the model for the keypoints of the manipulation *instruction* describes.

    Args:
        image: The current RGB frame ``uint8 [H, W, 3]``.
        instruction: The task ("move the white cable's free end through the
            black ring").
        start_point: ``[u, v]`` pixel the trace must start at (e.g. a
            ``gemini-er.point`` answer for the object to move). It is written
            as the start of the model's answer, so the model continues the
            trace from there; with ``depth`` the start's measured depth is
            written too. Default: the model chooses the start.
        depth: ``float32 [H, W]`` metres, optional. Only the start's depth is
            read from it (the keypoints above surfaces have no measured depth);
            the model's own depths place the rest.
        intrinsics: ``[3, 3]`` pinhole K. With it the keypoints are lifted to
            camera points with the model's depths.
        camera_pose: Camera-to-world ``Se3Pose`` or 4x4, for world points.
        mode: ``end_effector`` (the model card's trajectory prompt: waypoints
            of the gripper) or ``object`` (RoboTracer's object-centric visual
            trace: where the manipulated object goes).

    Returns:
        ``PlanResult``: the visual-plan structure of ``molmo-motion.forecast``
        with one track. ``query_pixel`` is the start, ``pixels`` the keypoints
        after it, ``dt_s`` is ``None`` (keypoints are ordered, not timed).
        ``found=False`` when fewer than two keypoints came back.
    """
    if not instruction or not str(instruction).strip():
        raise ValueError("robobrain.trace needs an instruction")
    if mode not in TRACE_MODES:
        raise ValueError(f"mode must be one of {sorted(TRACE_MODES)}, got {mode!r}")
    arr = _validate_image(image)
    height, width = arr.shape[:2]
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3) if intrinsics is not None else None
    T = _camera_matrix(camera_pose) if camera_pose is not None and K is not None else None
    depth_map = None
    if depth is not None:
        depth_map = np.asarray(depth, dtype=np.float32)
        if depth_map.shape != (height, width):
            raise ValueError(f"depth must be [{height}, {width}] like the image")

    measured = None
    prefix = ""
    if start_point is not None:
        start = np.asarray(start_point, dtype=np.float64).reshape(-1)
        if start.size != 2:
            raise ValueError(f"start_point must be [u, v], got {start_point!r}")
        xn, yn = _to_normalized(float(start[0]), float(start[1]), width, height)
        prefix = f"[({xn}, {yn}, "
        if depth_map is not None:
            measured = _depth_at(depth_map, float(start[0]), float(start[1]))
            if np.isfinite(measured):
                prefix += f"{measured:.2f}), ("
            else:
                measured = None

    prompt = TRACE_MODES[mode].format(instruction=str(instruction).strip())
    answer, timing = _run(
        "robobrain.trace", [arr, prompt], answer_prefix=prefix, max_new_tokens=_MAX_NEW_TOKENS
    )
    raw = prefix + answer
    keypoints = _parse_tuples(raw)
    pixels = np.array([_to_pixel(x, y, width, height) for x, y, _ in keypoints], dtype=np.float64)
    depths = [d for _, _, d in keypoints]
    if start_point is not None:
        start_px = np.asarray(start_point, dtype=np.float64).reshape(2)
        if len(pixels):
            pixels[0] = start_px  # exact, not its 0-1000 rounding
    if depth_map is not None and measured is None and len(pixels):
        m = _depth_at(depth_map, float(pixels[0, 0]), float(pixels[0, 1]))
        measured = m if np.isfinite(m) else None

    if len(pixels) == 0:
        start_px = (
            np.asarray(start_point, dtype=np.float64).reshape(2)
            if start_point is not None
            else np.zeros(2)
        )
        rest, rest_d, start_d = np.zeros((0, 2)), [], None
    else:
        start_px, rest, start_d, rest_d = pixels[0], pixels[1:], depths[0], depths[1:]

    track: PlanTrack = {
        "query_pixel": [float(start_px[0]), float(start_px[1])],
        "pixels": _round(rest, 2) if len(rest) else [],
        "start_camera": [],
        "points_camera": [],
        "start_world": [],
        "points_world": [],
        "predicted": [True] * len(rest),
        "start_depth_m": start_d,
        "depths_m": list(rest_d),
    }
    has_depth = start_d is not None and all(d is not None for d in rest_d)
    if K is not None and has_depth and len(pixels):
        z = np.array([start_d, *rest_d], dtype=np.float64)
        cam = _deproject(np.vstack([start_px[None], rest]), z, K)
        track["start_camera"] = _round(cam[0], 4)
        track["points_camera"] = _round(cam[1:], 4) if len(rest) else []
        if T is not None:
            world = _to_world(cam, T)
            track["start_world"] = _round(world[0], 4)
            track["points_world"] = _round(world[1:], 4) if len(rest) else []
    frame = "pixel"
    if track["start_camera"]:
        frame = "world" if track["start_world"] else "camera"
    if not has_depth:
        depth_source = "none"
    elif start_point is not None and depth_map is not None and prefix.endswith("("):
        depth_source = "model, start from depth map"
    else:
        depth_source = "model"
    return {
        "found": len(rest) >= 1,
        "tracks": [track],
        "horizon": len(rest),
        "dt_s": None,
        "frame": frame,
        "backend": "robobrain",
        "model": _model_id(),
        "timing": timing,
        "raw": raw,
        "mode": mode,
        "depth_source": depth_source,
        "measured_start_depth_m": None if measured is None else round(float(measured), 4),
    }


@tool(
    name="robobrain.progress",
    summary=(
        "Task progress 0..1 of the current frame against the task's start frame "
        "(RoboBrain 2.5 / Robo-Dopamine progress prompt, forward mode)."
    ),
    tags=("perception", "judge"),
)
def progress(
    image: np.ndarray,
    task: str,
    start_image: np.ndarray,
    goal_image: np.ndarray | None = None,
    wrist_images: list[np.ndarray] | None = None,
    start_wrist_images: list[np.ndarray] | None = None,
) -> ProgressResult:
    """How far the scene in *image* has got through *task* since *start_image*.

    Robo-Dopamine's forward mode: BEFORE is the start frame, AFTER the current
    one, the references are the start frame and the goal image, and the
    model's ``<score>+NN%</score>`` is the progress.

    Args:
        image: The current front-camera frame, uint8 [H, W, 3].
        task: The task instruction ("push the small cube into the open box").
        start_image: The same camera's frame when the task began.
        goal_image: A frame of the finished task, if one exists; default the
            uniform grey image Robo-Dopamine uses for "no goal".
        wrist_images: ``[left, right]`` wrist frames now; default the front
            frame repeated (upstream's single-view usage).
        start_wrist_images: ``[left, right]`` wrist frames at the start.

    Returns:
        ``{"progress", "score", "found", "raw", "timing"}``: ``score`` is the
        model's percentage / 100 in -1..1, ``progress`` that clipped to 0..1.
        ``found=False`` (progress 0) when no score parses.
    """
    if not task or not str(task).strip():
        raise ValueError("robobrain.progress needs a task")
    now = _validate_image(image)
    start = _validate_image(start_image, "start_image")
    goal = _validate_image(goal_image, "goal_image") if goal_image is not None else _BLANK_GOAL

    def views(front: np.ndarray, wrists: list | None, what: str) -> list[np.ndarray]:
        if wrists is None:
            return [front, front, front]
        if len(wrists) != 2:
            raise ValueError(f"{what} must be [left, right]")
        return [front] + [_validate_image(w, what) for w in wrists]

    images = [start, goal]
    images += views(start, start_wrist_images, "start_wrist_images")
    images += views(now, wrist_images, "wrist_images")
    images = [_fit_pixels(im, _PROGRESS_MAX_PIXELS) for im in images]
    texts = _PROGRESS_PROMPT.format(task=str(task).strip()).split("<image>")
    segments: list = []
    for text, im in zip(texts[:-1], images, strict=True):
        segments += [text, im]
    segments.append(texts[-1])
    answer, timing = _run("robobrain.progress", segments, max_new_tokens=_PROGRESS_MAX_NEW_TOKENS)
    m = _SCORE_RE.search(answer)
    if m is None:
        logger.debug("robobrain.progress: no score in %r", answer[:200])
        return {"progress": 0.0, "score": 0.0, "found": False, "raw": answer, "timing": timing}
    score = max(-100.0, min(100.0, float(m.group(1)))) / 100.0
    return {
        "progress": float(min(1.0, max(0.0, score))),
        "score": score,
        "found": True,
        "raw": answer,
        "timing": timing,
    }
