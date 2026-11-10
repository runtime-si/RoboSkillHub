# ruff: noqa  (vendored from CNS @ 6b42138, MIT)
# From CNS (https://github.com/hhcaz/CNS @ 6b42138), cns/utils/perception.py, MIT, Copyright (c) 2023 hhcaz.
# Changed: only CameraIntrinsic is kept (the rest needs open3d / pybullet).
import numpy as np


class CameraIntrinsic:
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    def to_dict(self):
        """Serialize intrinsic parameters to a dict object."""
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.flatten().tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Deserialize intrinisic parameters from a dict object."""
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5],
        )
        return intrinsic

    @classmethod
    def default(cls):
        return cls(width=640, height=480, fx=540, fy=540, cx=320, cy=240)

    def pixel_to_norm_camera_plane(self, uv: np.ndarray) -> np.ndarray:
        xy = (uv - np.array([self.cx, self.cy])) / np.array([self.fx, self.fy])
        return xy

    def norm_camera_plane_to_pixel(self, xy: np.ndarray, clip=True, round=False) -> np.ndarray:
        uv = xy * np.array([self.fx, self.fy]) + np.array([self.cx, self.cy])
        if clip:
            uv = np.clip(uv, 0, [self.width - 1, self.height - 1])
        if round:
            uv = np.round(uv).astype(np.int32)
        return uv
