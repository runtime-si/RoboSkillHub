"""Depth frames of a jigsaw piece among placed neighbours, seen by a wrist camera, for
verifying-piece-flushness-from-wrist-depth's tests and evaluation.

A 3 x 3 patch of a generated cut (tests/bundles/jigsaw_cut.py) lies on a table plane; the
centre piece is the one judged. Every pixel's ray is cast onto the height field (table 0,
pieces their thickness plus any defect of the centre piece), twice so the parallax of the
raised surface is right, smoothed by ``blur_px`` as the D405's stereo matching does at a step,
and noise is added: Gaussian, or a real D405 residual (depth minus its fitted plane).
"""

from __future__ import annotations

import cv2
import numpy as np

from .jigsaw_cut import make_puzzle

CELL = 0.0225
THICK = 0.002


def look_down(height=0.12, tilt_deg=8.0, yaw_deg=20.0):
    """Camera-to-world 4x4 for a camera ``height`` over the origin, looking down, tilted."""
    t, y = np.radians(tilt_deg), np.radians(yaw_deg)
    # Camera axes: x right, y down, z forward (down onto the table).
    rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    down = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    pose = np.eye(4)
    pose[:3, :3] = rz @ rx @ down
    pose[:3, 3] = pose[:3, :3] @ np.array([0.0, 0.0, -height])
    return pose


class Scene:
    """Pieces in world x, y round the origin; the centre piece is (1, 1) of a 3 x 3 cut."""

    def __init__(self, seed=0, neighbours=("west", "south"), style="classic"):
        self.puzzle = make_puzzle(3, 3, cell=(CELL, CELL), seed=seed, style=style)
        c = self.puzzle.centre(1, 1)
        self.offset = -c
        nb = {"west": (0, 1), "east": (2, 1), "south": (1, 0), "north": (1, 2)}
        self.neighbour_cells = [nb[d] for d in neighbours]
        self.neighbours = [self.outline(*ij) for ij in self.neighbour_cells]
        self.piece = self.outline(1, 1)

    def outline(self, i, j):
        return self.puzzle.outline(i, j) + self.offset


def render(
    scene,
    k,
    pose,
    shape,
    table_z=0.0,
    thick=THICK,
    raise_m=0.0,
    tilt_m=0.0,
    tilt_dir=(1.0, 0.0),
    shift_m=(0.0, 0.0),
    neighbour_raise_m=0.0,
    blur_px=1.0,
    supersample=2,
):
    """Depth image (metres along the optical axis) of the scene."""
    h, w = shape
    s = supersample
    # Height field on a world grid round the origin.
    res = 0.0001
    half = 2.0 * CELL
    n = int(2 * half / res)
    field = np.zeros((n, n), np.float32)

    def to_grid(poly):
        return np.round((poly + half) / res).astype(np.int32)

    for poly in scene.neighbours:
        cv2.fillPoly(field, [to_grid(poly)], float(thick + neighbour_raise_m))
    piece = scene.piece + np.asarray(shift_m)
    pm = np.zeros((n, n), np.uint8)
    cv2.fillPoly(pm, [to_grid(piece)], 1)
    gy, gx = np.mgrid[0:n, 0:n]
    wx, wy = gx * res - half, gy * res - half
    d = np.asarray(tilt_dir, dtype=np.float64)
    d /= np.linalg.norm(d)
    proj = wx * d[0] + wy * d[1]
    lo, hi = proj[pm > 0].min(), proj[pm > 0].max()
    ramp = (proj - lo) / (hi - lo)
    top = thick + raise_m + tilt_m * ramp
    field = np.where(pm > 0, np.maximum(field, top.astype(np.float32)), field)

    vs, us = np.mgrid[0 : h * s, 0 : w * s] / s
    rays_c = np.stack([(us - k[0, 2]) / k[0, 0], (vs - k[1, 2]) / k[1, 1], np.ones_like(us)], -1)
    r = rays_c @ pose[:3, :3].T
    c = pose[:3, 3]
    hgt = np.zeros(us.shape)
    for _ in range(3):
        t = (table_z + hgt - c[2]) / r[..., 2]
        x, y = c[0] + t * r[..., 0], c[1] + t * r[..., 1]
        gi = np.clip(np.round((x + half) / res).astype(int), 0, n - 1)
        gj = np.clip(np.round((y + half) / res).astype(int), 0, n - 1)
        inside = (np.abs(x) < half) & (np.abs(y) < half)
        hgt = np.where(inside, field[gj, gi], 0.0)
    t = (table_z + hgt - c[2]) / r[..., 2]
    depth = t  # rays have z = 1 in the camera frame, so t is the depth along the axis
    depth = cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    if blur_px > 0:
        depth = cv2.GaussianBlur(depth, (0, 0), blur_px)
    return depth.astype(np.float64)


def quaternion(pose):
    """Se3Pose dict of a 4x4 camera-to-world matrix."""
    m = pose[:3, :3]
    w = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2
    x = np.copysign(x, m[2, 1] - m[1, 2])
    y = np.copysign(y, m[0, 2] - m[2, 0])
    z = np.copysign(z, m[1, 0] - m[0, 1])
    return {
        "position": {"x": float(pose[0, 3]), "y": float(pose[1, 3]), "z": float(pose[2, 3])},
        "rotation": {"w": float(w), "x": float(x), "y": float(y), "z": float(z)},
    }


def camera(depth, k, pose, rgb=None):
    return {
        "name": "wrist",
        "rgb": rgb if rgb is not None else np.zeros((*depth.shape, 3), np.uint8),
        "depth": depth.astype(np.float32),
        "intrinsics": k,
        "pose": quaternion(pose),
    }
