---
name: disassembling-a-brick-by-twist-lever
description: Removes a seated LEGO or DUPLO brick with a parallel gripper and
  no force sensor - ICL's disassembly twist (grip, seat the grip, rotate 14 deg
  about one bottom edge so the stud rows let go one at a time, lift 15 mm with
  the twist kept), the edge being the one under a finger, commanded in small
  steps and read back so tracking lag (or joint effort) tells a brick that holds
  from one that lets go; with a straight pull and a wiggle (alternate small
  levers about both edges) as strategies and fallbacks, a regrasp after a slip,
  and the wrist depth camera reading the seat and its neighbouring cells before
  and after. Use when a brick must come off a baseplate or another brick at a
  cell known from the stud grid (computing-brick-pose-from-stud-grid's seat).
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [lego, duplo, disassembly, lever, contact, wrist-camera]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego]
    provenance:
      authors: [runtime-si]
      code: https://github.com/intelligent-control-lab/Robotic_Lego_Manipulation/blob/9bc91654e708e4d8428ecfa76840103cacb2a19a/src/ros_nodes/lego_task_planning_node.cpp
      upstream_license: MIT
      paper: arXiv:2309.02354
gap:
  requires: {connector: [robot.grasp_frame, robot.describe_gripper, robot.set_grip]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.get_gripper
    - robot.describe_gripper
    - robot.grasp_frame
    - robot.solve_ik
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_to_joints
    - robot.open_gripper
    - robot.close_gripper
    - robot.set_grip
  required_inputs:
    seat_position: Vec3
  produces_outputs:
    strategy: str
    attempts: int
    attempt_log: str
    pivot_side: str
    twist_deg: float
    peak_lag: float
    releases: str
    release_count: int
    seat_lag: float
    holding: bool
    top_height_change: float
    neighbour_change: float
    samples: int
    verified_by: str
  exit_conditions:
    removed: The hand lifted free, the jaws read at the brick's width, and the wrist depth reads the seat more than half a brick height under the brick's top (and less than 1.5) with no neighbouring cell moved; the brick is in the hand at the clearance height.
    stuck: Every strategy tried stalled - for three steps in a row the hand stood lag_threshold further behind its command than in free motion (or a joint effort jumped) and advanced less than half its step, or the lever's tilt stood 4 deg short; the hand went back to the grasp, opened and rose, the brick left on the seat.
    slipped: The last strategy lifted free but the jaws closed past the brick's width, or the wrist depth still reads the brick on its seat (regrasped and the next strategy tried while any were left); the jaws are open over the seat.
    neighbour_lifted: The seat reads more than 1.5 brick heights low (the brick under it came along) or a neighbouring cell's stud top moved more than max(half a stud height, 2 mm) between the views before and after; still holding.
    missed: Before anything was gripped - the wrist depth finds no brick top at the seat, the open jaws stopped 3 mm or more over the grasp, or the jaws closed past the brick's width.
    unverified: Lifted free, but the seat was not read (verify false, or no wrist camera / reachable view / 3 depth readings); detail says whether the jaws hold something.
    unreachable: The grasp, or every strategy's motion, failed the IK preview, or a move was refused.
  canonical_scripts:
    - disassemble_brick: scripts/disassemble_brick.py
  examples:
    - title: Lever a DUPLO 2x2 off a baseplate, straight pull as the fallback
      path: examples/canonical_subgraph.json
---

# disassembling-a-brick-by-twist-lever

(Body written below.)
