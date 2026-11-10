---
name: picking-a-peg-with-a-side-grasp
description: Picks a peg with the left arm (pin, dowel, pen, rod, plug body) across its
  shaft, at a grip point that leaves at least 25 mm of the free end (the end
  opposite the tip) clear of the pads, and holds it with its axis vertical, free
  end up, at a handover point both arms reach. A standing peg is taken with a
  level side grasp; a lying peg is taken top-down across the shaft (a level grasp
  of a lying peg puts a finger under the table and the tool centre under the
  14 mm floor), lifted, turned about the vertical, and pitched a quarter turn
  about the closing axis. The roll is set on the last joint (+/-2.09 rad) from
  forward kinematics, the hold is confirmed from the jaw opening, and the axis
  tilt is read back. Use when an elongated part (8-12 mm shaft, 60-120 mm long)
  must end upright in the left hand, free end up, for a top-down regrasp by the
  right arm (handing-over-a-peg-to-a-top-down-grasp reads `left_grip`).
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [peg, side-grasp, reorientation, bimanual, handover, insertion]
  hub:
    stage: execute
    embodiments: [any]
    arms: 2
    version: 0.1.0
    tasks: [insertion, reorientation]
    provenance:
      authors: [runtime-si]
      code: https://github.com/runtime-si/RoboSkillHub/tree/main/skills/standing-a-lying-domino-by-reorient-and-regrasp
      upstream_license: Apache-2.0 (this repository; no external code)
gap:
  requires:
    connector: [robot.grasp_frame, robot.describe_gripper, robot.describe_arm, robot.forward_kinematics, sim.get_object_obb]
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.get_gripper
    - robot.describe_gripper
    - robot.describe_arm
    - robot.grasp_frame
    - robot.solve_ik
    - robot.forward_kinematics
    - robot.go_to_pose
    - robot.move_to_joints
    - robot.open_gripper
    - robot.close_gripper
    - sim.get_object_obb
  # Also required: peg (dict {length_m, diameter_m, mass_kg?, axis_in_object, tip_end}; gap's type registry has no
  # dict type). Also produced: left_grip (dict {T_base_tcp: Se3Pose, T_TCP_object: Se3Pose, exposed_len_m: float}).
  required_inputs:
    peg_pose: Se3Pose
    table_z: float
  # Optional inputs: handover_point (Vec3; default midway between the two hands, 0.20 m over the table),
  # arm_id (0), other_arm_id (1; -1 skips the receiver's reach preview), object_name (re-perceive the peg with
  # sim.get_object_obb at the end; empty = judge from the hand), min_exposed_m (0.025), min_tool_height (0.014),
  # tilt_threshold_deg (8), handover_height (0.20), lift_height (0.06), hover (0.05), pad_width (0.0155),
  # joint_limits.
  produces_outputs:
    posture: str
    picked: bool
    axis_tilt_deg: float
    tilt_source: str
    grip_from_free_end_m: float
    grip_width_m: float
    facing_deg: float
    free_end: Vec3
  exit_conditions:
    holding_vertical: The left hand holds the peg at the handover point with its axis within tilt_threshold_deg (8) of vertical, free end up, at least min_exposed_m (less 4 mm) of it beyond the pads; `left_grip` is set. `tilt_source` says whether the tilt is the hand's (no re-perception, so what the hand did, not what the peg did) or perceived.
    grasp_failed: The descent stopped short of the grasp (a fingertip on the peg, or the peg moved), or the jaws closed to a width outside diameter -3 / +7 mm. Opened and backed out; nothing held.
    dropped: The jaws read narrower than the shaft after the lift, at the handover point or at the end, or the re-perceived peg is not between the jaws.
    slipped: Held at the handover point, but the axis is more than tilt_threshold_deg from vertical, or less of the free end is exposed than needed; still holding (`left_grip` is set so a caller can decide).
    ungraspable: Nothing moved. The peg cannot leave min_exposed_m free beyond the pads, is wider than the open hand, leans between standing and lying, or (lying) the fingertips at the tool-centre floor stand over the shaft's equator.
    upside_down: Nothing moved. The peg stands on its free end (tip up); one arm cannot turn it end over end.
    unreachable: No grasp plus held pose passed the IK preview for both arms (nothing moved), or a move, the lift, the turn or the last-joint correction was refused (`detail` says whether the left hand still holds the peg).
  canonical_scripts:
    - pick_peg: scripts/pick_peg.py
  examples:
    - title: Pick a peg and hold it vertical for the right arm
      path: examples/canonical_subgraph.json
---

# picking-a-peg-with-a-side-grasp

Step 1 of the strategy *left side grasp, right top-down regrasp, top-down
insertion* (`recipes/side-grasp-regrasp-top-insert`). It ends with the peg
upright in the left hand, its free end up, so that
handing-over-a-peg-to-a-top-down-grasp can take that end from above and
inserting-a-held-peg-top-down can put the tip into a hole.

## Interfaces (shared with the other two skills; names are exact)

- `peg`: `{length_m, diameter_m, mass_kg?, axis_in_object: [0,0,1], tip_end: "+z" | "-z"}`. The object frame's
  origin is the peg's centre. The **free end** is the one opposite `tip_end`: it is the one that goes up, since the
  tip goes into the hole.
- `peg_pose`: the peg's world pose (from a 6D pose skill, or an OBB whose z is the shaft).
- Output `left_grip`: `{T_base_tcp, T_TCP_object, exposed_len_m}`.
  - `T_base_tcp`: the left tool centre's pose as `robot.get_ee_pose` reports it (`{position, rotation}`, the
    station's world frame, like every other hub pose).
  - `T_TCP_object`: the peg's frame in the tool frame. From the hand: the pose the peg was seen at, read in the tool
    frame at the close, with its offset along the closing axis set to zero (the jaws centre the shaft). With
    `object_name`: re-perceived at the end.
  - `exposed_len_m`: from the free end to the near edge of the pads, along the axis.

## What it does

1. **Posture.** The free end within 20 deg of up: standing. Within 20 deg of
   level: lying. Within 20 deg of down: `upside_down`. Otherwise `ungraspable`.
2. **Grip point.** The middle of the peg (no gravity moment about the jaws,
   which is the axis a parallel pinch on a round shaft turns about: SIM_GRIP.md),
   moved toward the far end when the middle leaves under 25 + 3 mm beyond the
   15.5 mm pads.
3. **Plan before moving.** For each heading of the held approach (the free
   end's heading first, then fanned out by 20-30 deg steps), the IK preview
   checks: the pre-grasp and the grasp (both wrist solutions from
   `robot.grasp_frame`), the hand top-down and pitched (lying) or level
   (standing) at the handover point, and the right arm's top-down reach at the
   free end there. The last joint is set from forward kinematics in every
   preview, so its limit is checked on the roll the hand will really have.
4. **Grasp.**
   - Lying: top-down, jaws across the shaft, tool centre at the 14 mm floor or as
     low as keeps the fingertips 2 mm off the table; refused when the fingertips
     would still be over the shaft's equator (a 10 mm shaft has 1.2 mm under it at
     14 mm; an 8 mm shaft needs the floor lowered).
   - Standing: level approach along the chosen heading, standoff 50 mm, in to the
     shaft's axis.
5. **Close and check.** The jaw gap must read diameter -3 / +7 mm.
6. **Lift, turn, carry.** Lift 60 mm, rechecking the jaws. Lying: turn about the
   vertical so the free end points along the chosen heading, carry top-down to the
   handover point, pitch a quarter turn about the closing axis (10 deg chained IK
   steps, joint jumps over 0.6 rad refused). That turns the free end up and leaves
   the approach level. Standing: carry at the grasp rotation.
7. **Stand it up on the last joint.** The axis is read back from the hand, and
   any lean about the approach over 4 deg is turned out on the last joint. A YAM
   station twin's solver leaves this roll free (PORTING lesson 10).
8. **Report.** Tilt and exposed length come from the hand, or from
   `sim.get_object_obb(object_name)` when a name is given (the twin's stand-in
   for re-perception; on a real station pass a 6D pose instead and leave
   `object_name` empty). The jaws are checked once more.

## Composes with

- proposing-side-grasps: the same `grasp_frame(approach=..., close_heading_deg=...)` contract. This skill fans
  its own headings because the held pose at the handover point, not jaw margin, decides.
- standing-a-lying-domino-by-reorient-and-regrasp: the chained-IK step with the joint-jump guard and the
  roll finished on the last joint come from there. That skill's single-arm pitch failed on a 7.5 mm tile edge
  (a hinge, gripped off its centre of mass). Here a round shaft is gripped at its middle, in the air.
- tracking-gripper-opening-stall: chain it before this skill's close when a stall reading is wanted. This skill
  reads one gap after `close_gripper`.
- checking-dual-arm-clearance: run it before handing over if the right arm is parked near the handover point.
- detecting-contact-from-joint-effort: not used; the only contact here is the grasp.

## Simulation check

Pending (filled in below after the twin runs).

## Limits

- No real-YAM run.
- Without `object_name` (or a re-perceived pose), `axis_tilt_deg` is the hand's: a peg that pivoted in the pads
  reads vertical. The pivot is the known weak mode of a pinch on a round shaft (SIM_GRIP.md).
- `T_base_tcp` is in the world frame the robot tools use, not an arm-base frame.
