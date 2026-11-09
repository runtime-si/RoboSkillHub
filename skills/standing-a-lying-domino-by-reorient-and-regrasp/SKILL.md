---
name: standing-a-lying-domino-by-reorient-and-regrasp
description: Stands a flat-lying domino upright so a top-down pick can take it -
  pinches the lying tile across its width near one end, lifts it, turns it
  about the vertical if the arm needs another facing, pitches the hand 90 deg
  about its closing axis so the tile's long axis stands vertical, reads the
  wrist roll back and finishes it on the last joint, sets the tile on its end
  at a staging spot (by default where it lay), opens a few millimetres, backs
  off and rises; then checks from depth that a tile stands there. Every stage
  is previewed with the IK solver before anything moves, and the facing is
  chosen from those that pass. Use when a domino lies flat (from
  reading-domino-pips-and-orientation or a 6D pose) and must stand before
  placing-an-upright-domino-beside-neighbours picks it into a row.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [domino, reorient, regrasp, upright, pitch, depth-check]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [domino, reorientation]
    provenance:
      authors: [runtime-si]
      code: https://github.com/Zhu-Chenyu/Franka-Domino-Project/blob/main/place_dominoes/place_dominoes/place_dominoes.py
      upstream_license: none (reference only; reimplemented from the behaviour, no code copied)
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
    table_z: float
    tile_center: Vec3
    long_axis: Vec3
  produces_outputs:
    picked: bool
    released: bool
    grasp_height: float
    grip_width: float
    facing_deg: float
    yaw_turn_deg: float
    roll_turn_deg: float
    tilt_at_release_deg: float
    release_position: Vec3
    standing_tile: Se3Pose
    verified: bool
    tile_top: float
    standing_offset: float
  exit_conditions:
    standing: Released at the staging spot and retreated, and the depth check sees a tile standing there (its top within 10 mm of the tile's height and within 8 mm of the spot); with verify=false, what the hand did only. `standing_tile` is the pose to hand to placing-an-upright-domino-beside-neighbours.
    fell: Released, but the depth check sees no tile standing at the spot - its top is low (lying, on its side, leaning hard), more than 8 mm off the spot, or nothing is there.
    unverified: Released and retreated, but no camera saw the spot whole and unoccluded.
    ungraspable: Nothing was lifted - the descent to the tile stopped high (a fingertip on it, or it is not where it was seen), the jaws closed on nothing or not across its width, or the grasp height leaves the fingertips over the tile.
    dropped: The tile was picked, then the jaws read empty or not across it - after the lift, after the pitch or over the staging spot.
    unreachable: No facing passed the IK preview for the grasp, the pitch and the release; a move was refused; the jaws or the held tile stayed turned off their target (no room on the last joint); the descent to the spot stopped high or the held tile leaned past tilt_tolerance_deg there (lifted back); or the retreat was refused after the release. `detail` says whether the tile is still held.
  canonical_scripts:
    - stand_domino: scripts/stand_domino.py
  examples:
    - title: Stand a lying domino where it lies
      path: examples/canonical_subgraph.json
---

# standing-a-lying-domino-by-reorient-and-regrasp

(draft)
