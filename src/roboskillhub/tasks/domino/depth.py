"""Object records from the placing skill's depth upright check (``judge_tile``).

The depth check reads a tile's top height and offset at a slot from calibrated RGB-D
cameras. It does not estimate an orientation or an identity, so its record carries
``pose: None``: verify_stable_placement can use it to see a tile that is *down*
(seen, not standing), but never to declare one upright.
"""

from __future__ import annotations

from typing import Any

from . import geometry as geo
from . import upstream


def depth_upright_record(
    observation: dict,
    slot_pose: dict,
    table_z: float,
    geometry: dict,
    *,
    camera_name: str = "",
    along_room: float = 0.0,
) -> dict[str, Any]:
    """``{"pose": None, "depth_upright": verdict}`` for the tile expected at ``slot_pose``.
    ``observation`` is the robot observation with ``cameras`` (depth, intrinsics, pose)."""
    pd = upstream.place()
    size = {"height": geometry["height_m"], "width": geometry["width_m"], "thickness": geometry["thickness_m"]}
    cams = pd.order_cameras(pd._cameras(observation), camera_name)
    xy, facing = geo.xy_facing(slot_pose)
    verdict = pd.judge_tile(cams, xy, facing, float(table_z), size, float(along_room))
    return {
        "pose": None,
        "identity_confidence": None,
        "depth_upright": {k: (float(v) if isinstance(v, float) else v) for k, v in verdict.items()},
    }
