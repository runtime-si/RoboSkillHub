"""The next brick of a planned build: its cell, size and the words a detector and the policy use for it."""

import json
from typing import Any, TypedDict

from gap import NodeContext

COLOUR_WORDS = {"4": "red", "1": "blue", "14": "yellow", "2": "green", "15": "white", "0": "black"}
"""LDraw colour codes -> the word in the detector prompt and the policy instruction."""


class Output(TypedDict):
    cell: dict[str, Any]
    brick_size: str
    brick_studs: list[int]
    part_name: str
    remaining: int


def run(ctx: NodeContext, sequence_json: str, done: int = 0) -> Output:
    """``sequence_json``: planning-brick-assembly-order's order; ``done``: how many are already placed."""
    sequence = json.loads(sequence_json)
    if isinstance(sequence, dict):
        sequence = sequence.get("sequence", [])
    if done >= len(sequence):
        raise ValueError(
            f"the build has {len(sequence)} bricks and {done} are placed: nothing left"
        )
    brick = sequence[done]
    a, b = (int(v) for v in str(brick["brick_size"]).lower().split("x"))
    colour = COLOUR_WORDS.get(str(brick.get("color", brick.get("colour", ""))), "")
    system = str(brick.get("system") or "duplo").upper()
    return {
        "cell": dict(brick["cell"]),
        "brick_size": f"{a}x{b}",
        "brick_studs": [a, b],
        "part_name": " ".join(w for w in (colour, f"{a}x{b}", system, "brick") if w),
        "remaining": len(sequence) - done - 1,
    }
