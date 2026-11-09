"""rsh-task/1 Observations from the placing skill's depth upright check (``judge_tile``).

The depth check reads a tile's top height and offset at a slot from calibrated RGB-D
cameras. It estimates neither an orientation nor an identity, so its Observation has
``quaternion_xyzw: null`` and ``confidence: null``: verify_stable_placement can use its
``state_label`` to see a tile that is down, but never to declare one upright.
"""

from __future__ import annotations

from typing import Any

from . import geometry as geo
from . import upstream


def depth_upright_observation(
    observation: dict,
    object_id: str,
    slot_pose: dict,
    table_z: float,
    geometry: dict,
    *,
    timestamp_s: float,
    camera_name: str = "",
    along_room: float = 0.0,
) -> dict[str, Any]:
    """The tile expected at ``slot_pose`` as seen by depth. ``observation`` is the robot
    observation with ``cameras`` (depth, intrinsics, pose), as the placing skill reads it."""
    pd = upstream.place()
    size = {
        "height": geometry["height_m"],
        "width": geometry["width_m"],
        "thickness": geometry["thickness_m"],
    }
    cams = pd.order_cameras(pd._cameras(observation), camera_name)
    xy, facing = geo.xy_facing(slot_pose)
    v = pd.judge_tile(cams, xy, facing, float(table_z), size, float(along_room))
    if not v["seen"]:
        label, visibility = None, "occluded" if v.get("why") == "occluded" else "unknown"
    elif v["standing"]:
        label, visibility = "standing", "visible"
    elif v.get("top", 0.0) < size["height"] - float(pd.HEIGHT_TOLERANCE):
        label, visibility = "lying", "visible"
    else:
        label, visibility = "uncertain", "visible"
    return {
        "object_id": object_id,
        "class": "domino",
        "frame_id": slot_pose["frame_id"],
        "position_m": None,
        "quaternion_xyzw": None,
        "timestamp_s": float(timestamp_s),
        "source": f"depth_top_check:{v.get('camera') or 'none'}",
        "privileged": False,
        "confidence": None,
        "uncertainty": None,
        "visibility": visibility,
        "state_label": label,
        "reason": v.get("why") or None,
        "depth_check": {k: (float(x) if isinstance(x, float) else x) for k, x in v.items()},
    }
