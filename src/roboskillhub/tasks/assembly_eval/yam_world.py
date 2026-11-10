# SPDX-License-Identifier: Apache-2.0
"""World factory for Agent 5's domino baselines with the YAM arm station.

``yam_domino_world(seed, slots=3, fault=None)`` returns the keys of
``domino_world.physics_domino_world`` (same scene spec, same seeded rack poses, same slots,
same ``independent()`` evaluator) with ``yam_station.YamDominoStation`` as the backend. What
differs is the robot and therefore the station limits: the tool box is the real station's
(``motion_server`` BOUNDS), its floor the 14 mm tool-centre floor, and the jaw is the
linear_4310's 94 mm span with fingertips 10.2 mm past the TCP.
"""

from __future__ import annotations

from roboskillhub.runtime.contracts import Limits

from .domino_geometry import tilt_deg
from .domino_runner import feeder_poses_for_seed
from .domino_world import TABLE_Z, independent_evaluation, world_spec
from .yam_model import YamDominoPhysics, YamPhysicsConfig
from .yam_station import TCP_FLOOR_M, YamDominoStation

TIP_REACH_M = 0.0102
MAX_WIDTH_M = 0.094


def limits(station: YamDominoStation) -> Limits:
    return Limits.from_dict(
        {
            "frame_id": "world",
            "workspace_min_m": [round(v, 4) for v in station.box_min],
            "workspace_max_m": [round(v, 4) for v in station.box_max],
            "max_speed_m_s": 0.1,
            "max_contact_distance_m": 0.3,
            "gripper_min_width_m": 0.0,
            "gripper_max_width_m": MAX_WIDTH_M,
        }
    )


def yam_domino_world(
    seed: int,
    *,
    slots: int = 3,
    fault: str | None = None,
    fault_index: int | None = None,
    table_offset_m: float = 0.008,
    tcp_floor_m: float = TCP_FLOOR_M,
    cfg: YamPhysicsConfig | None = None,
    observations: str = "truth",
    vision=None,
) -> dict:
    """``observations="vision"`` with a ``yam_vision.YamVision`` estimates the tile poses from the
    rendered overhead camera instead of reading the simulator truth (the default)."""
    spec_true = world_spec(slots, table_offset_m=table_offset_m if fault == "table_offset" else 0.0)
    spec_cal = world_spec(slots)
    feeders = feeder_poses_for_seed(spec_true, seed)
    ids = spec_true.tile_ids()
    station = YamDominoStation(
        YamDominoPhysics(cfg or YamPhysicsConfig()),
        spec_true,
        feeders,
        calibrated_table_z=TABLE_Z,
        piece_ids=ids,
        fault=fault,
        fault_index=slots - 1 if fault_index is None else fault_index,
        table_offset_m=table_offset_m,
        tcp_floor_m=tcp_floor_m,
        observations=observations,
        vision=vision,
    )
    size = [spec_cal.tile.height, spec_cal.tile.width, spec_cal.tile.thickness]
    slot_list = []
    for k in range(slots):
        p = spec_cal.slot_pose(k)
        slot_list.append(
            {
                "slot_id": f"s{k}",
                "order": k,
                "pose": {
                    "frame_id": "world",
                    "position_m": p["position"],
                    "quaternion_xyzw": p["quat_xyzw"],
                },
                "neighbours": [f"s{j}" for j in (k - 1, k + 1) if 0 <= j < slots],
            }
        )
    tb = spec_cal.table
    inputs = {
        "layout": {
            "kind": "domino_layout",
            "layout_id": f"line{slots}-physics",
            "frame_id": "world",
            "piece": {"class": "domino", "size_m": size},
            "slots": slot_list,
        },
        "feeder": {"pieces": ids},
        "gripper_id": "yam_left_linear_4310",
        "table_z": TABLE_Z,
        "grasp": {
            "T_object_TCP": {
                "frame_id": "object",
                "position_m": [0.0, 0.0, 0.004],
                "quaternion_xyzw": [1.0, 0.0, 0.0, 0.0],
            },
            "pregrasp_height_m": 0.05,
            "lift_height_m": 0.06,
            "open_width_m": 0.0195,
            "close_width_m": 0.0055,
            "uncertainty": {"position_m": 0.001, "rotation_rad": 0.02},
            "collision_geometry": {"type": "box", "size_m": size},
        },
        "tolerances": {"max_drift_m": 0.005},
        "place": {
            "geometry": {"size_m": size, "yaw_symmetry_deg": 180.0},
            "support": {
                "frame_id": "world",
                "point_m": [tb.center_xy[0], tb.center_xy[1], TABLE_Z],
                "normal": [0.0, 0.0, 1.0],
                "height_std_m": 0.001,
                "calibration_version": "domino-station-world-v1 (scene ground truth before any injected offset)",
                "region_min_m": [tb.center_xy[0] - 0.28, tb.center_xy[1] - 0.28],
                "region_max_m": [tb.center_xy[0] + 0.28, tb.center_xy[1] + 0.28],
                "observation_id": "table",
            },
            "retreat": {"distance_m": 0.06},
            "hand": {"max_open_width_m": MAX_WIDTH_M, "tip_reach_m": TIP_REACH_M},
            "workspace": {
                "min_m": [round(v, 4) for v in station.box_min],
                "max_m": [round(v, 4) for v in station.box_max],
            },
            "timeout_s": 60.0,
            "tolerances_ref": "tasks/domino/config/line_v1.yaml",
        },
    }
    describe = {
        **station.describe(),
        "seed": seed,
        "slots": slots,
        "scene_config_sha256": spec_true.config_hash(),
    }
    describe.pop("capabilities")
    holder: dict = {}

    def independent() -> dict:
        if "result" not in holder:
            holder["result"] = independent_evaluation(station, spec_true, slots)
        return holder["result"]

    def interventions() -> int:
        objs = station.physics.truth()["objects"]
        return sum(1 for o in objs.values() if tilt_deg(o["quat_xyzw"]) > 45.0)

    return {
        "backend": station,
        "limits": limits(station),
        "inputs": inputs,
        "describe": describe,
        "task_success": lambda: independent()["layout"]["status"] == "succeeded",
        "independent": independent,
        "privileged_allowed": observations == "truth",
        "interventions": interventions,
        "station": station,
    }
