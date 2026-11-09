"""Known single-object geometry: dominoes and a finite set of LEGO bricks.

Object frames (box frames, as ``tools/pose6d`` fits them, origin at the body centre):

* domino: +x the long axis, +y the width axis, +z the thickness axis; +z is the
  pip face when the pips have been read, else the sign is a symmetry choice.
* LEGO brick: +x along ``studs[0]``, +y along ``studs[1]``, +z out of the stud
  face; the box is the body without studs (the stud/tube relief is ignored by the
  fit, as pose6d's SKILL.md says). This is ``computing-brick-pose-from-stud-grid``'s
  frame moved from the top centre to the body centre.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _hub

DOMINO_SIZE_M = (0.048, 0.024, 0.0075)
"""48 x 24 x 7.5 mm: the tile in pose6d's synthetic evaluation and PR #106's twin runs.
reading-domino-pips-and-orientation defaults to 8 mm thick; pass the set's own size."""


@dataclass(frozen=True)
class ObjectSpec:
    object_class: str
    size_m: tuple[float, float, float]
    symmetry: str = "box"
    """Rotations that leave the shape unchanged: ``box`` (the 4 half-turns of a box
    with distinct edges), ``z2`` (half-turn about z only, e.g. a brick whose studs
    mark +z), ``z4``, ``none``."""
    mass_kg: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def half(self) -> np.ndarray:
        return np.asarray(self.size_m, dtype=float) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_class": self.object_class,
            "size_m": [float(v) for v in self.size_m],
            "symmetry": self.symmetry,
            "mass_kg": self.mass_kg,
            "meta": dict(self.meta),
        }


def domino(
    size_m: tuple[float, float, float] = DOMINO_SIZE_M, mass_kg: float | None = 0.01
) -> ObjectSpec:
    L, W, T = (float(v) for v in size_m)
    if not L > W > T > 0:
        raise ValueError(f"domino size must be length > width > thickness > 0, got {size_m}")
    return ObjectSpec("domino", (L, W, T), "box", mass_kg)


def lego_brick(
    studs: tuple[int, int] = (2, 4), system: str = "lego", plate: bool = False
) -> ObjectSpec:
    """A brick (or a plate, 1/3 height) from the brick-pose skill's presets."""
    bp = _hub.load("brick_pose")
    preset = bp.PRESETS[system]
    a, b = (int(v) for v in studs)
    if a < 1 or b < 1:
        raise ValueError(f"stud counts must be positive, got {studs}")
    pitch = preset["stud_pitch"]
    h = preset["brick_height"] / (3.0 if plate else 1.0)
    play = 2 * bp.FOOTPRINT_PLAY
    sym = "z4" if a == b else "z2"
    return ObjectSpec(
        "lego_plate" if plate else "lego_brick",
        (a * pitch - play, b * pitch - play, h),
        sym,
        None,
        {
            "studs": [a, b],
            "system": system,
            "stud_pitch_m": pitch,
            "stud_height_m": preset["stud_height"],
            "stud_diameter_m": preset["stud_diameter"],
            "body_height_m": h,
        },
    )


def symmetry_rotations(spec: ObjectSpec) -> list[np.ndarray]:
    """Object-frame rotations S with ``T_obj @ S`` the same physical placement."""
    if spec.symmetry == "none":
        return [np.eye(3)]
    if spec.symmetry in ("z2", "z4"):
        n = 2 if spec.symmetry == "z2" else 4
        return [
            np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
            for a in (2 * np.pi * k / n for k in range(n))
        ]
    if spec.symmetry == "box":
        d = np.asarray(spec.size_m)
        out = []
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((1.0, -1.0), repeat=3):
                M = np.zeros((3, 3))
                for row, (col, s) in enumerate(zip(perm, signs, strict=True)):
                    M[row, col] = s
                if np.linalg.det(M) > 0 and np.allclose(np.abs(M) @ d, d, atol=1e-6):
                    out.append(M)
        return out
    raise ValueError(f"unknown symmetry {spec.symmetry!r}")


def box_corners(spec: ObjectSpec) -> np.ndarray:
    h = spec.half
    return np.array(list(itertools.product(*[(-v, v) for v in h])))
