"""Build a GaP workflow that places one DUPLO brick with GR00T doing the contact-rich step.

    python make_graph.py OUT_DIR [--steps-json build.json]

The graph decides what and where with existing hub skills, hands the how to GR00T with marks, then checks:

    plate_sg   calibrating-baseplate-stud-grid      the plate's pose from markers + a wrist close-up
    plan_sg    planning-brick-assembly-order        the build as a placement order
    seat_sg    (next_brick.py, here) +              the first brick of the order (cell, size, name),
               computing-brick-pose-from-stud-grid  and where it seats on the plate
    find_sg    perceiving-objects                   where the loose brick lies on the table
    gr00t_sg   gr00t (place_with_gr00t.py, here)    marks: grasp at the brick, release at the seat;
                                                    gr00t.run_skill until one grasp-and-release cycle
    verify_sg  verifying-brick-seated-by-height     seated, or how it is not

OUT_DIR receives workflow.json and scripts/<subgraph>/ - copies of each skill's canonical scripts (gap
resolves a script node's path against the workflow's directory) and this example's two scripts. Run it with
a GR00T server up (GAP_GR00T_URL) and a connector serving the cameras the perception and verify skills read.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
HUB = HERE.parents[3]

DEFAULT_BUILD = [
    {
        "step": 1,
        "bricks": [
            {
                "id": "b1",
                "system": "duplo",
                "cell": {"x": 4, "y": 6, "layer": 1},
                "size": "2x2",
                "rotation": 0,
                "color": "4",
            }
        ],
    }
]
"""One red DUPLO 2x2 on a 16 x 16 DUPLO plate (cell x, y in studs from the plate's corner, layer 1 = on the plate)."""

PLATE = {
    "plate_studs": [16, 16],
    "system": "duplo",
    "markers": [
        {"id": 3, "size": 0.04, "position": [0.163, 0.0, -0.0032]},
        {"id": 7, "size": 0.04, "position": [-0.163, 0.05, -0.0032], "yaw_deg": 30.0},
    ],
    "marker_dictionary": "DICT_4X4_50",
}
"""The plate's markers in the plate frame, as calibrating-baseplate-stud-grid's own example gives them."""


def _sg(
    skill: str, inputs: dict, outputs: dict, nodes: dict, edges: list, exit_: dict, on_error: str
) -> dict:
    return {
        "skill": skill,
        "inputs": inputs,
        "outputs": outputs,
        "nodes": nodes,
        "edges": edges,
        "conditional_edges": {},
        "exit": exit_,
        "on_error": on_error,
    }


def _script(sg: str, name: str, inputs: dict) -> dict:
    return {"type": "script", "script": f"scripts/{sg}/{name}", "inputs": inputs}


def _ref(path: str) -> dict:
    return {"$ref": path}


def workflow(build: list) -> dict:
    single = [["START", "n"], ["n", "END"]]
    # The verify node's footprint: the first brick's (the graph places one; a list type cannot cross subgraphs).
    studs = [int(v) for v in str(build[0]["bricks"][0]["size"]).lower().split("x")]
    subgraphs = {
        "plate_sg": _sg(
            "calibrating-baseplate-stud-grid",
            {},
            {"plate_pose": _ref("n.plate_pose")},
            {"n": _script("plate_sg", "calibrate_plate.py", PLATE)},
            single,
            {
                "router_field": "route",
                "success_values": ["calibrated", "coarse_only", "no_lattice", "no_markers"],
            },
            "no_markers",
        ),
        "plan_sg": _sg(
            "planning-brick-assembly-order",
            {},
            {"sequence_json": _ref("n.sequence_json")},
            {
                "n": _script(
                    "plan_sg",
                    "plan_order.py",
                    {"steps_json": json.dumps(build), "check_fingers": True},
                )
            },
            single,
            {
                "router_field": "route",
                "success_values": [
                    "sequence_ready",
                    "inventory_short",
                    "unreachable_step",
                    "not_buildable",
                    "finger_blocked",
                ],
            },
            "invalid_input",
        ),
        "seat_sg": _sg(
            "computing-brick-pose-from-stud-grid",
            {"plate_pose": "Se3Pose", "sequence_json": "str"},
            {
                "seat_position": _ref("n.seat_position"),
                "seat_yaw_deg": _ref("n.seat_yaw_deg"),
                "part_name": _ref("next.part_name"),
            },
            {
                "next": _script(
                    "seat_sg", "next_brick.py", {"sequence_json": _ref("in.sequence_json")}
                ),
                "n": _script(
                    "seat_sg",
                    "brick_pose.py",
                    {
                        "plate_pose": _ref("in.plate_pose"),
                        "cell": _ref("next.cell"),
                        "brick_size": _ref("next.brick_size"),
                        "plate_studs": PLATE["plate_studs"],
                        "system": "duplo",
                    },
                ),
            },
            [["START", "next"], ["next", "n"], ["n", "END"]],
            {
                "router_field": "route",
                "success_values": [
                    "computed",
                    "off_grid",
                    "out_of_plate",
                    "collides",
                    "unsupported",
                    "ungraspable",
                    "unreachable",
                ],
            },
            "invalid",
        ),
        "find_sg": _sg(
            "perceiving-objects",
            {"part_name": "str"},
            {"target_obb": _ref("obb.obb")},
            {
                "observe": {"type": "tool", "tool": "robot.get_observation"},
                "perceive": _script(
                    "find_sg",
                    "perceive_dino_vlm.py",
                    {"cameras": _ref("observe.cameras"), "object_name": _ref("in.part_name")},
                ),
                "obb": {
                    "type": "tool",
                    "tool": "geometry.filter_and_compute_obb",
                    "inputs": {"points": _ref("perceive.cloud")},
                },
                "found": {"type": "noop"},
            },
            [
                ["START", "observe"],
                ["observe", "perceive"],
                ["perceive", "obb"],
                ["obb", "found"],
                ["found", "END"],
            ],
            {"router_field": None, "success_values": ["found"]},
            "not_found",
        ),
        "gr00t_sg": _sg(
            "gr00t",
            {"target_obb": "OrientedBoundingBox", "seat_position": "Vec3", "part_name": "str"},
            {"rows": _ref("n.rows"), "marks": _ref("n.marks")},
            {
                "n": _script(
                    "gr00t_sg",
                    "place_with_gr00t.py",
                    {
                        "target_obb": _ref("in.target_obb"),
                        "seat_position": _ref("in.seat_position"),
                        "part_name": _ref("in.part_name"),
                        "max_steps": 600,
                    },
                )
            },
            single,
            {
                "router_field": "route",
                "success_values": ["succeeded", "max_steps", "unsafe_action"],
            },
            "failed",
        ),
        "verify_sg": _sg(
            "verifying-brick-seated-by-height",
            {"seat_position": "Vec3", "seat_yaw_deg": "float"},
            {"height_error": _ref("n.height_error")},
            {
                "n": _script(
                    "verify_sg",
                    "verify_seat.py",
                    {
                        "seat_position": _ref("in.seat_position"),
                        "seat_yaw_deg": _ref("in.seat_yaw_deg"),
                        "brick_studs": studs,
                        "system": "duplo",
                    },
                )
            },
            single,
            {
                "router_field": "route",
                "success_values": [
                    "seated",
                    "not_seated",
                    "high_by_stud",
                    "tilted",
                    "shifted_off_grid",
                    "no_brick",
                    "unverified",
                ],
            },
            "unreachable",
        ),
    }
    route = {
        "plate_sg": {
            "calibrated": "plan_sg",
            "coarse_only": "plan_sg",
            "no_lattice": "abort",
            "no_markers": "abort",
        },
        "plan_sg": {
            "sequence_ready": "seat_sg",
            "inventory_short": "seat_sg",
            "unreachable_step": "seat_sg",
            "not_buildable": "abort",
            "finger_blocked": "abort",
            "invalid_input": "abort",
        },
        "seat_sg": {
            "computed": "find_sg",
            **dict.fromkeys(
                [
                    "off_grid",
                    "out_of_plate",
                    "collides",
                    "unsupported",
                    "ungraspable",
                    "unreachable",
                    "invalid",
                ],
                "abort",
            ),
        },
        "find_sg": {"found": "gr00t_sg", "not_found": "abort"},
        "gr00t_sg": {
            "succeeded": "verify_sg",
            "max_steps": "verify_sg",
            "unsafe_action": "abort",
            "failed": "abort",
        },
        "verify_sg": {
            "seated": "done",
            **dict.fromkeys(
                ["not_seated", "high_by_stud", "tilted", "shifted_off_grid", "no_brick"],
                "not_seated",
            ),
            "unverified": "unverified",
            "unreachable": "unverified",
        },
    }
    nodes = {sg: {"type": "subgraph", "ref": sg} for sg in subgraphs}
    nodes.update(
        done={"type": "end", "status": "success"},
        not_seated={"type": "end", "status": "failure"},
        unverified={"type": "end", "status": "failure"},
        abort={"type": "end", "status": "failure"},
    )
    return {
        "version": 3,
        "meta": {
            "name": "lego_place_with_gr00t",
            "description": "Plan and perceive one DUPLO brick and its seat with hub skills; GR00T places and presses it, "
            "steered by grasp/release marks; verifying-brick-seated-by-height checks the result.",
        },
        "nodes": nodes,
        "edges": [["START", "plate_sg"]],
        "conditional_edges": {
            sg: {"router_field": "exit", "mapping": m} for sg, m in route.items()
        },
        "subgraphs": subgraphs,
    }


SKILL_SCRIPTS = {
    "plate_sg": "calibrating-baseplate-stud-grid",
    "plan_sg": "planning-brick-assembly-order",
    "seat_sg": "computing-brick-pose-from-stud-grid",
    "find_sg": "perceiving-objects",
    "verify_sg": "verifying-brick-seated-by-height",
}


def build(out: Path, steps: list | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__")
    for sg, skill in SKILL_SCRIPTS.items():
        shutil.copytree(
            HUB / "skills" / skill / "scripts",
            out / "scripts" / sg,
            dirs_exist_ok=True,
            ignore=ignore,
        )
    shutil.copy2(HERE / "scripts" / "next_brick.py", out / "scripts" / "seat_sg" / "next_brick.py")
    shutil.copytree(
        HERE / "scripts" / "gr00t_sg",
        out / "scripts" / "gr00t_sg",
        dirs_exist_ok=True,
        ignore=ignore,
    )
    (out / "workflow.json").write_text(
        json.dumps(workflow(steps or DEFAULT_BUILD), indent=1) + "\n"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("out", type=Path)
    p.add_argument(
        "--steps-json", type=Path, help="a build (planning-brick-assembly-order's steps_json)"
    )
    a = p.parse_args()
    steps = json.loads(a.steps_json.read_text()) if a.steps_json else None
    print(build(a.out, steps) / "workflow.json")


if __name__ == "__main__":
    main()
