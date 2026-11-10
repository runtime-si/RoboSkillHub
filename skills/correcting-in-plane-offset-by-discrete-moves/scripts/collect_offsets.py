# Adapted from vision-guided-rack-insertion-ot2
# (https://github.com/cooper-group-uol-robotics/vision-guided-rack-insertion-ot2, commit 3bfe3fe87b64),
# data_collection/collect_offsets_single_camera.py (random dx, dy, dtheta jogs from a taught aligned pose,
# one wrist image per jog, the offset as its label), MIT License, Copyright (c) 2026 Hadi Beyzaee
# Juneghani. Changes: robot.* tools instead of panda_py / ROS; the label is the hand's offset read back
# after the move (not the command), so a short or refused move cannot mislabel an image; yaw is turned
# with the last joint; the crop is taken here and saved with an index.jsonl the move-classifier tool reads.
"""Collect self-labelled wrist crops around a taught aligned pose for move-classifier.train."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moves import Hand, wrap  # noqa: E402

logger = logging.getLogger(__name__)

MAX_OFFSET_M = 0.012
"""Upstream's jog range in x and y (+-12 mm)."""
MAX_YAW_DEG = 5.0
"""Upstream's jog range in yaw (+-5 deg)."""
ZERO_FRACTION = 0.35
"""Per axis, the share of jogs drawn inside the label dead-band (0.5 mm / 0.2 deg) instead of the full
range. Uniform jogs over +-12 mm put almost nothing there: in the twin, 0 of 400 images were "000", and a
model that never sees "aligned" never ends the loop. Not upstream."""
ALIGNED_FRACTION = 0.1
"""Share of jogs inside the dead-band on all three axes (the "000" class). Not upstream."""


class Output(TypedDict):
    route: str
    samples: int
    refused: int
    dataset_dir: str
    reason: str


def crop_frame(obs: dict[str, Any], camera: str, crop: list[int] | None) -> np.ndarray | None:
    """The named camera's RGB, cropped ``[top, bottom, left, right]`` pixels off (upstream's margins)."""
    cams = obs.get("cameras") or []
    if isinstance(cams, dict):
        cams = [{"name": k, **v} for k, v in cams.items()]
    for c in cams:
        if str(c.get("name")) == camera and c.get("rgb") is not None:
            img = np.asarray(c["rgb"])[..., :3]
            if img.dtype != np.uint8:
                img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
            if crop:
                t, b, left, r = (int(v) for v in crop)
                h, w = img.shape[:2]
                img = img[t : h - b, left : w - r]
            return np.ascontiguousarray(img)
    return None


def run(
    ctx: NodeContext,
    dataset_dir: str,
    camera: str,
    samples: int = 200,
    crop: list[int] | None = None,
    max_offset_m: float = MAX_OFFSET_M,
    max_yaw_deg: float = MAX_YAW_DEG,
    arm_id: int = 0,
    seed: int = 0,
    max_refused: int = 20,
    zero_fraction: float = ZERO_FRACTION,
    aligned_fraction: float = ALIGNED_FRACTION,
) -> Output:
    """The hand must start at the taught aligned pose (part over its target); it returns there."""
    from PIL import Image

    hand = Hand(ctx, arm_id)
    home, home_yaw, home_rot = hand.state()
    out = Path(dataset_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    index = out / "index.jsonl"
    start = sum(1 for _ in index.open()) if index.exists() else 0
    rng = np.random.default_rng(seed)
    got = refused = 0
    with index.open("a") as f:
        while got < samples and refused <= max_refused:
            all_zero = rng.random() < aligned_fraction
            band = [rng.random() < zero_fraction or all_zero for _ in range(3)]
            dx, dy = (
                rng.uniform(-0.0004, 0.0004) if b else rng.uniform(-max_offset_m, max_offset_m)
                for b in band[:2]
            )
            dyaw = float(
                rng.uniform(-0.15, 0.15) if band[2] else rng.uniform(-max_yaw_deg, max_yaw_deg)
            )
            why = hand.move_xy(home[0] + dx, home[1] + dy, home[2], home_rot)
            if not why:
                _, yaw_now, _ = hand.state()
                why = hand.turn(wrap(home_yaw + dyaw - yaw_now))
            if why:
                refused += 1
                logger.info("collect_offsets: jog refused (%s)", why)
                continue
            pos, yaw, _ = hand.state()
            img = crop_frame(ctx.tool("robot.get_observation"), camera, crop)
            if img is None:
                return {
                    "route": "no_frame",
                    "samples": got,
                    "refused": refused,
                    "dataset_dir": str(out),
                    "reason": f"no RGB from camera {camera!r}",
                }
            name = f"images/{start + got:05d}.png"
            Image.fromarray(img).save(out / name)
            row = {
                "image": name,
                "dx": float(pos[0] - home[0]),
                "dy": float(pos[1] - home[1]),
                "dyaw_deg": float(wrap(yaw - home_yaw)),
                "commanded": [float(dx), float(dy), dyaw],
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            got += 1
    # back to the taught pose
    hand.move_xy(home[0], home[1], home[2], home_rot)
    _, yaw_now, _ = hand.state()
    hand.turn(wrap(home_yaw - yaw_now))
    route = "collected" if got >= samples else "move_refused"
    return {
        "route": route,
        "samples": got,
        "refused": refused,
        "dataset_dir": str(out),
        "reason": f"{got} images, {refused} jogs refused",
    }
