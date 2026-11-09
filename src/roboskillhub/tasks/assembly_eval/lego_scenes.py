"""LEGO evaluation scenes: fixed baseplate, known brick types, single-arm vertical assembly.

Each scene is a RoCo task folder (legolization JSON, the format of RoCo_BrickAssembly/tasks): Type 1 has
``structure_start.json`` (already built, imported with authored connections) and ``structure_goal.json``;
Type 2 has only ``structure.json`` and starts from the bare plate. Brick ids index RoCo's
``config/lego_library.json`` (2 = 2x4, 12 = 2x2). Grid x/y are studs, z is brick layers; ``ori`` 1 swaps the
footprint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE_PLATE = {
    "Dimension": [32, 32],
    "Position": [0, -0.2, 0.0],
    "Orientation": [0.7071, 0, 0, 0.7071],
    "Color": "Light Gray",
}

SCENES: dict[str, dict] = {
    # RoCo tasks/type1/example1 verbatim: one red 2x4 pre-built on the plate, put a blue 2x4 on top of it.
    "single_on_brick": {
        "task_type": "1",
        "start": {"1": {"brick_id": 2, "x": 0, "y": 0, "z": 0, "ori": 0, "color": "Red"}},
        "goal": {
            "1": {"brick_id": 2, "x": 0, "y": 0, "z": 0, "ori": 0, "color": "Red"},
            "2": {"brick_id": 2, "x": 0, "y": 0, "z": 1, "ori": 0, "color": "Blue"},
        },
        "note": "one brick-on-brick connection; the unmodified NaivePolicy plans this correctly (offset 0)",
    },
    # One 2x4 on the bare plate.
    "single_on_plate": {
        "task_type": "2",
        "goal": {"1": {"brick_id": 2, "x": 0, "y": 0, "z": 0, "ori": 0, "color": "Red"}},
        "note": "one brick-on-plate connection",
    },
    # Two 2x4 side by side on the plate, a third 2x4 stacked on the first: three placements, one parent each.
    "seq3_stack": {
        "task_type": "2",
        "goal": {
            "1": {"brick_id": 2, "x": 0, "y": 0, "z": 0, "ori": 1, "color": "Red"},
            "2": {"brick_id": 2, "x": 0, "y": 2, "z": 0, "ori": 1, "color": "Green"},
            "3": {"brick_id": 2, "x": 0, "y": 0, "z": 1, "ori": 1, "color": "Blue"},
        },
        "note": "three bricks, three connections; brick 3 needs brick 1",
    },
    # As seq3_stack with one free stud row between bricks 1 and 2 (lego.align_connection refuses a hand
    # approach next to a placed brick: would_damage_build).
    "seq3_spaced": {
        "task_type": "2",
        "goal": {
            "1": {"brick_id": 2, "x": 0, "y": 0, "z": 0, "ori": 1, "color": "Red"},
            "2": {"brick_id": 2, "x": 0, "y": 3, "z": 0, "ori": 1, "color": "Green"},
            "3": {"brick_id": 2, "x": 0, "y": 0, "z": 1, "ori": 1, "color": "Blue"},
        },
        "note": "three bricks, three connections; brick 3 needs brick 1; bricks 1 and 2 one stud apart",
    },
    # Two 2x4 side by side on the plate, a third 2x4 turned 90 degrees bridging both. Orientations chosen so
    # each brick can be picked and placed by one DexMate arm (checked with RoCo's Pinocchio IK; the mirrored
    # layout needs a hand-over for brick 1).
    "seq3_bridge": {
        "task_type": "2",
        "goal": {
            "1": {"brick_id": 2, "x": 0, "y": 0, "z": 0, "ori": 1, "color": "Red"},
            "2": {"brick_id": 2, "x": 0, "y": 2, "z": 0, "ori": 1, "color": "Green"},
            "3": {"brick_id": 2, "x": 1, "y": 0, "z": 1, "ori": 0, "color": "Blue"},
        },
        "note": "three bricks, four connections; brick 3 needs both 1 and 2 for full support",
    },
}


def write_task_dir(name: str, out: Path) -> Path:
    """Write a scene as a RoCo task folder and return it."""
    scene = SCENES[name]
    out.mkdir(parents=True, exist_ok=True)
    if scene["task_type"] == "1":
        (out / "structure_start.json").write_text(json.dumps(scene["start"], indent=2))
        (out / "structure_goal.json").write_text(json.dumps(scene["goal"], indent=2))
    else:
        (out / "structure.json").write_text(json.dumps(scene["goal"], indent=2))
    return out


def scene_hash(name: str) -> str:
    blob = json.dumps({"scene": SCENES[name], "plate": BASE_PLATE}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def user_config(
    task_dir: Path, roco_root: Path, task_type: str, gripper_max_force: float = 0.1
) -> dict:
    """RoCo user_config.json for a scene, with absolute paths and without cameras (the baseline is privileged).

    ``gripper_max_force`` is the gripper joint drive limit; RoCo ships 0.1 (user_config.json is the file RoCo
    lets participants change).
    """
    assets = roco_root / "robot_assets" / "DexMate"
    return {
        "Task_Config": {
            "Task_Path": str(task_dir),
            "Task_Type": task_type,
            "Base_Plate": BASE_PLATE,
        },
        "Robot_Config": {
            "Robot_Package_Dir": str(assets),
            "Robot_URDF_Path": str(assets / "vega_1u_gripper.urdf"),
            "Robot_USD_Path": str(assets / "vega_1u_gripper.usda"),
            "Joint_Home_Position": [
                0.2,
                0.5,
                0,
                1.2,
                1.4,
                -1.57,
                -1.57,
                1,
                -0.35,
                0.2,
                0,
                0,
                -1.2,
                -1.4,
                -1.57,
                1.57,
                -1,
                0.35,
                0.2,
                0,
                0,
                0,
                0,
            ],
            "Robot_Base_Frame": {
                "Position": [0, -0.6, -0.55],
                "Orientation": [0.7071, 0, 0, 0.7071],
            },
            "Joints_Physics": {
                "/World/Robot/R_ee_link/joints/R_gripper_joint": {
                    "Max_Force": gripper_max_force,
                    "Damping": 1.0,
                    "Stiffness": 100.0,
                },
                "/World/Robot/L_ee_link/joints/L_gripper_joint": {
                    "Max_Force": gripper_max_force,
                    "Damping": 1.0,
                    "Stiffness": 100.0,
                },
            },
            "Gripper_Config": {
                "Material": {
                    "Static_Friction": 1.5,
                    "Dynamic_Friction": 1.3,
                    "Restitution": 0.0,
                    "Link_Instance_Names": [
                        "/World/Robot/R_ee_link/gripper_active_link/gripper_active_link",
                        "/World/Robot/R_ee_link/gripper_passive_link/left_gripper_passive_link",
                        "/World/Robot/L_ee_link/gripper_active_link/gripper_active_link",
                        "/World/Robot/L_ee_link/gripper_passive_link/left_gripper_passive_link",
                    ],
                }
            },
        },
        "Env_Config": {
            "Storage_Config": {
                "Size": [0.2, 0.1, 0.1],
                "Position": [0.0, 0, 0.05],
                "Orientation": [1.0, 0.0, 0.0, 0.0],
            }
        },
    }
