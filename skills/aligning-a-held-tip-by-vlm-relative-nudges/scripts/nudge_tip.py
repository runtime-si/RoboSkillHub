"""Put a held part's tip over a hole by asking a VLM only for the direction: each step shows the wrist close-up(s)
with an arrow frame drawn on, asks "+x / -x / +y / -y / centred", moves the hand a step that halves on every
reversal along that axis, and stops when the model says centred twice or both steps are down to the floor."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

START_STEP = 0.004
"""First move along an axis [m]; halved each time the model reverses direction on it."""

MIN_STEP = 0.0005
"""Once both axes' steps are at or under this, the tip is as close as the answers can bring it [m]."""

MAX_REFUSALS = 2
"""Unparseable / refused answers in a row before no_verdict."""

ARROW_PX = 70

PROMPT = (
    "Close-up wrist camera image(s) of a robot gripper holding a pen-like part above a holder with a hole. "
    "Arrows drawn on the image(s) show +x (red) and +y (blue). Look at the bottom end (tip) of the held part "
    "and the hole it should go into. Which single direction should the tip move to get closer to being "
    "centred over the hole: +x, -x, +y or -y? If the tip is already centred over the hole (within about "
    "half a millimetre), answer centred. If you cannot see the tip or the hole, answer unsure. "
    "Answer with exactly one word: +x, -x, +y, -y, centred or unsure."
)

ANSWERS = ("+x", "-x", "+y", "-y", "centred")


class Output(TypedDict):
    route: str
    detail: str
    steps: int
    calls: int
    moved_x: float
    moved_y: float
    answers: list


def parse(text: str) -> str | None:
    """The model's one-word answer, or None (unsure, refusal, prose without one)."""
    t = (text or "").strip().lower().replace("−", "-").replace("centered", "centred")
    m = re.findall(r"(?<![a-z])([+-][xy]|centred|unsure)(?![a-z])", t)
    if not m or m[-1] == "unsure":
        return None
    return m[-1]


def image_axes(k: np.ndarray, t_base_cam: np.ndarray, tip: np.ndarray) -> np.ndarray:
    """2x2 pixel-per-metre Jacobian of the tip's image point w.r.t. a base-frame (x, y) move of the tip."""

    def proj(p):
        c = np.linalg.inv(t_base_cam) @ np.r_[p, 1.0]
        return (k @ c[:3])[:2] / c[2]

    e = 0.005
    p0 = proj(tip)
    return np.stack([(proj(tip + [e, 0, 0]) - p0) / e, (proj(tip + [0, e, 0]) - p0) / e], axis=1)


def overlay(rgb: np.ndarray, origin_px: np.ndarray, jac: np.ndarray) -> np.ndarray:
    """The frame with +x (red) and +y (blue) arrows from ``origin_px`` along the base axes' image directions."""
    im = Image.fromarray(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for col, (name, colour) in enumerate((("+x", (255, 40, 40)), ("+y", (40, 120, 255)))):
        v = jac[:, col] / max(np.linalg.norm(jac[:, col]), 1e-9) * ARROW_PX
        a, b = origin_px, origin_px + v
        d.line([tuple(a), tuple(b)], fill=colour, width=3)
        n = np.array([-v[1], v[0]]) / ARROW_PX * 8
        d.polygon([tuple(b + v / ARROW_PX * 10), tuple(b + n), tuple(b - n)], fill=colour)
        d.text(tuple(b + v / ARROW_PX * 14), name, fill=colour)
    return np.asarray(im)


def servo(
    look: Callable[[], list[np.ndarray] | None],
    ask: Callable[[list[np.ndarray], str], str],
    move: Callable[[float, float], None],
    start_step: float = START_STEP,
    min_step: float = MIN_STEP,
    max_steps: int = 16,
    max_travel: float = 0.015,
) -> Output:
    """The loop without the robot: ``look`` gives the annotated frames, ``ask`` the model's text, ``move`` a base
    (dx, dy) of the hand."""
    out: Output = {"route": "not_converging", "detail": "", "steps": 0, "calls": 0, "moved_x": 0.0,
                   "moved_y": 0.0, "answers": []}  # fmt: skip
    step = {"x": float(start_step), "y": float(start_step)}
    last = {"x": 0, "y": 0}
    centred = refused = 0
    for _ in range(int(max_steps)):
        frames = look()
        if not frames:
            out.update(route="no_verdict", detail="no wrist frame to show the model")
            return out
        out["calls"] += 1
        a = parse(ask(frames, PROMPT))
        out["answers"].append(a)
        if a is None:
            refused += 1
            if refused >= MAX_REFUSALS:
                out.update(route="no_verdict", detail=f"{refused} answers in a row without a direction")
                return out
            continue
        refused = 0
        if a == "centred":
            centred += 1
            if centred >= 2:
                out.update(route="centred", detail=f"centred twice after {out['steps']} move(s)")
                return out
            continue
        centred = 0
        axis, sign = a[1], (1 if a[0] == "+" else -1)
        if last[axis] and last[axis] != sign:
            step[axis] = max(step[axis] / 2, min_step / 2)
        last[axis] = sign
        if step["x"] <= min_step and step["y"] <= min_step and last["x"] and last["y"]:
            out.update(route="centred", detail=f"both steps down to {min_step * 1e3:.1f} mm")
            return out
        dx, dy = (sign * step["x"], 0.0) if axis == "x" else (0.0, sign * step["y"])
        if np.hypot(out["moved_x"] + dx, out["moved_y"] + dy) > max_travel:
            out.update(route="not_converging",
                       detail=f"the answers would take the tip over {max_travel * 1e3:.0f} mm from its start")  # fmt: skip
            return out
        move(dx, dy)
        out["steps"] += 1
        out["moved_x"] += dx
        out["moved_y"] += dy
    out["detail"] = f"no verdict of centred within {max_steps} model calls"
    return out


def azure_ask(usage: dict[str, int] | None = None) -> Callable[[list[np.ndarray], str], str]:
    """Azure OpenAI on its v1 endpoint through the Responses API (``AZURE_OPENAI_ENDPOINT``, ``_API_KEY``,
    ``_DEPLOYMENT``). The hub's vlm.query chat-completions payload (temperature 0, max_tokens) is refused with
    400 by a reasoning deployment whose name does not start with gpt-5, hence this path. Token counts are summed
    into ``usage``."""
    import base64
    import io
    import os

    from openai import OpenAI

    client = OpenAI(base_url=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_API_KEY"])
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    usage = usage if usage is not None else {}

    def ask(frames: list[np.ndarray], prompt: str) -> str:
        content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
        for f in frames:
            buf = io.BytesIO()
            Image.fromarray(np.asarray(f, np.uint8)).save(buf, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "input_image", "image_url": url})
        r = client.responses.create(model=deployment, input=[{"role": "user", "content": content}])
        usage["input"] = usage.get("input", 0) + r.usage.input_tokens
        usage["output"] = usage.get("output", 0) + r.usage.output_tokens
        return r.output_text or ""

    return ask


def _rot(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def _pose(p: dict[str, Any]) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = _rot(p["rotation"])
    t[:3, 3] = [float(p["position"][a]) for a in "xyz"]
    return t


def run(
    ctx: NodeContext,
    tip_offset_m: float,
    cameras: list[str] | None = None,
    ask_tool: str = "vlm.query",
    model: str | None = None,
    backend: str = "tool",
    start_step_m: float = START_STEP,
    min_step_m: float = MIN_STEP,
    max_steps: int = 16,
    arm_id: int = 0,
) -> Output:
    arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    names = cameras or []

    def tip() -> np.ndarray:
        r = ctx.tool("robot.get_ee_pose", **arm)
        t = _pose(r.get("pose", r))
        return t[:3, 3] + t[:3, 2] * float(tip_offset_m)

    def look() -> list[np.ndarray] | None:
        obs = ctx.tool("robot.get_observation")
        cams = [c for c in (obs.get("cameras") or []) if isinstance(c, dict)] if isinstance(obs, dict) else []
        if names:
            cams = [c for c in cams if c.get("name") in names]
        else:
            cams = [c for c in cams if any(s in str(c.get("name", "")) for s in ("wrist", "eye_in_hand"))][:1]
        p, frames = tip(), []
        for c in cams:
            if c.get("rgb") is None or c.get("intrinsics") is None or c.get("pose") is None:
                continue
            k, tbc = np.asarray(c["intrinsics"], float), _pose(c["pose"])
            jac = image_axes(k, tbc, p)
            cc = np.linalg.inv(tbc) @ np.r_[p, 1.0]
            frames.append(overlay(np.asarray(c["rgb"]), (k @ cc[:3])[:2] / cc[2], jac))
        return frames

    def tool_ask(frames: list[np.ndarray], prompt: str) -> str:
        kw: dict[str, Any] = {"prompt": prompt, "images": frames}
        if model:
            kw["model"] = model
        r = ctx.tool(ask_tool, **kw)
        return r.get("text", "") if isinstance(r, dict) else str(r)

    ask = azure_ask() if backend == "azure" else tool_ask

    def move(dx: float, dy: float) -> None:
        r = ctx.tool("robot.get_ee_pose", **arm)
        p = r.get("pose", r)
        goal = {a: float(p["position"][a]) + d for a, d in zip("xyz", (dx, dy, 0.0), strict=True)}
        ctx.tool("robot.go_to_pose", pose={"position": goal, "rotation": p["rotation"]}, tolerance=0.0003, **arm)

    return servo(look, ask, move, start_step_m, min_step_m, max_steps)
