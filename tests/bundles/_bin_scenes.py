"""Synthetic top-down RGB-D frames of boxes on a table, for the bin and container skills' tests.

A scene is a height field and a colour field over world (x, y). The camera looks straight down
(camera x = world x, camera y = world -y, camera z = world -z), so a pixel's ray is solved against
the height field by marching down it in ``step`` (2 mm) slices, so walls show their sides as they
do to a real camera off their axis. Scenes must lie within 1 m of the world origin.
"""

from __future__ import annotations

import math

import numpy as np

DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
"""Camera-to-world rotation of a camera looking straight down (a half turn about x)."""


def yaw_quat(deg: float) -> dict[str, float]:
    h = math.radians(deg) / 2
    return {"w": math.cos(h), "x": 0.0, "y": 0.0, "z": math.sin(h)}


def obb(cx, cy, long_side, short_side, yaw_deg, rim_z, support_z=0.0) -> dict:
    return {
        "center": {"x": cx, "y": cy, "z": (rim_z + support_z) / 2},
        "extent": {"x": long_side / 2, "y": short_side / 2, "z": (rim_z - support_z) / 2},
        "orientation": yaw_quat(yaw_deg),
    }


def local(x, y, cx, cy, yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    dx, dy = x - cx, y - cy
    return c * dx + s * dy, -s * dx + c * dy


class Scene:
    """Boxes and blocks added in order; later ones overwrite where they are higher."""

    def __init__(self, table_rgb=(120, 120, 120)):
        self.items: list = []
        self.table_rgb = table_rgb

    def open_box(
        self,
        cx,
        cy,
        long_side,
        short_side,
        yaw_deg,
        rim_z,
        wall=0.006,
        floor=0.004,
        rgb=(170, 120, 60),
        gaps=(),
    ):
        """An open box; ``gaps``: wall names ("+x", "-x", "+y", "-y") left out."""

        def height(x, y):
            lx, ly = local(x, y, cx, cy, yaw_deg)
            inside = (np.abs(lx) <= long_side / 2) & (np.abs(ly) <= short_side / 2)
            walls = np.zeros_like(inside)
            if "+x" not in gaps:
                walls |= lx > long_side / 2 - wall
            if "-x" not in gaps:
                walls |= lx < -long_side / 2 + wall
            if "+y" not in gaps:
                walls |= ly > short_side / 2 - wall
            if "-y" not in gaps:
                walls |= ly < -short_side / 2 + wall
            return np.where(inside, np.where(walls, rim_z, floor), -1.0)

        self.items.append((height, rgb))
        return self

    def block(self, cx, cy, long_side, short_side, yaw_deg, top_z, rgb=(128, 128, 128)):
        def height(x, y):
            lx, ly = local(x, y, cx, cy, yaw_deg)
            inside = (np.abs(lx) <= long_side / 2) & (np.abs(ly) <= short_side / 2)
            return np.where(inside, top_z, -1.0)

        self.items.append((height, rgb))
        return self

    def sample(self, x, y):
        h = np.zeros_like(x)
        rgb = np.empty(x.shape + (3,), np.uint8)
        rgb[:] = self.table_rgb
        for height, colour in self.items:
            hi = height(x, y)
            over = hi > h
            h = np.where(over, hi, h)
            rgb[over] = colour
        return h, rgb

    def frame(
        self,
        cam_xyz=(0.0, 0.0, 1.0),
        focal=500.0,
        shape=(480, 640),
        name="top",
        rotation=None,
        step=0.002,
    ) -> dict:
        """A CameraFrame from a camera looking straight down from ``cam_xyz``."""
        hgt, wid = shape
        k = np.array([[focal, 0, wid / 2], [0, focal, hgt / 2], [0, 0, 1.0]])
        vv, uu = np.mgrid[0:hgt, 0:wid].astype(float)
        top = max(float(self.sample(*np.meshgrid(*[np.linspace(-1, 1, 201)] * 2))[0].max()), 0.0)
        depth = np.full(shape, float(cam_xyz[2]))
        hit = np.zeros(shape, bool)
        rgb = np.empty(shape + (3,), np.uint8)
        steps = max(2, int(round(top / step)) + 1)
        for z in np.linspace(cam_xyz[2] - top, cam_xyz[2], steps):
            x = cam_xyz[0] + (uu - k[0, 2]) * z / focal
            y = cam_xyz[1] - (vv - k[1, 2]) * z / focal
            h, colour = self.sample(x, y)
            new = ~hit & (h >= cam_xyz[2] - z - 1e-9)
            depth[new] = z
            rgb[new] = colour[new]
            hit |= new
        z = depth
        return {
            "name": name,
            "rgb": rgb,
            "depth": z.astype(np.float32),
            "intrinsics": k,
            "pose": {
                "position": {"x": cam_xyz[0], "y": cam_xyz[1], "z": cam_xyz[2]},
                "rotation": rotation or DOWN,
            },
        }
