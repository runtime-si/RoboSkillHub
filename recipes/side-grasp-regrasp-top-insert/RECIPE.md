---
name: side-grasp-regrasp-top-insert
description: Put an elongated part (pin, dowel, pen, rod, plug body) into a
  hole straight down on a two-arm station - the left arm picks it with a side
  grasp and stands it up free end up, the right arm takes the free end from
  above, and the right arm inserts it top-down with stop-on-contact, a bounded
  search and a probe of where the tip ended. Use when a peg must go into a
  perceived hole along its own axis, the part cannot be picked top-down by the
  end that must stay free, and both arms are addressable by arm_id.
license: Apache-2.0
metadata:
  hub:
    version: 0.1.0
    embodiments: [any]
    tuned_on: [yam]
    tasks: [insertion, bimanual]
    tier: full
recipe:
  inputs:
    peg: object
    peg_pose: Se3Pose
    table_z: float
    hole: object
  start: pick
  nodes:
    pick:
      skill: picking-a-peg-with-a-side-grasp
      note: Left arm, level side grasp of a standing peg (top-down then pitched for a lying one), held vertical, free end up, at a handover point the right arm's top-down grasp reaches (its handover_point is the left tool centre).
    handover:
      skill: handing-over-a-peg-to-a-top-down-grasp
      note: Right arm comes down along -z over the free end; gives `held` for the insertion.
    handover_again:
      skill: handing-over-a-peg-to-a-top-down-grasp
      note: One retry after a receiver miss (the left still holds the peg where it was); a second miss ends the graph.
    insert:
      skill: inserting-a-held-peg-top-down
    done: {end: success}
    abort: {end: failure, recovery: [robot.open_gripper]}
  routes:
    pick:
      holding_vertical: handover
      slipped: abort
      grasp_failed: abort
      dropped: abort
      ungraspable: abort
      upside_down: abort
      unreachable: abort
    handover:
      handed_over: insert
      receiver_missed: handover_again
      collision_risk: abort
      dropped: abort
      unreachable: abort
    handover_again:
      handed_over: insert
      receiver_missed: abort
      collision_risk: abort
      dropped: abort
      unreachable: abort
    insert:
      inserted: done
      unverified: done
      jammed: abort
      missed_hole: abort
      slipped_in_grip: abort
      unreachable: abort
---

# Side grasp, top-down regrasp, top-down insert

The user-specified strategy for elongated parts on the dual-arm YAM station
(`candidates/PEG_SIDE_TOP_INSERT.md`): a parallel gripper cannot push a peg
straight into a hole while it holds the peg from the side (the fingers would
have to go in with it, or the push would turn the peg in the pads), and it
often cannot pick the peg top-down by the end that must stay free. So one arm
picks from the side and stands the peg up, the other takes it from above, and
the second arm pushes along the peg's own axis.

## The interfaces

The three skills share these names (open dicts, wired from node to node):

- `peg`: `{length_m, diameter_m, mass_kg?, axis_in_object: [0,0,1], tip_end}`,
  object origin at the peg's centre.
- `left_grip` (pick to handover): `{T_base_tcp, T_TCP_object, exposed_len_m}`;
  the handover needs `exposed_len_m` >= 3 + 10.2 + 5 mm (bite, fingertip
  reach, margin) and asks the pick for 25 mm.
- `held` (handover to insert): `{arm: "right", T_TCP_object, tip_offset_m,
  axis_tilt_deg, sigma_mm}`.
- `hole`: `{T_world_mouth (z down into the hole), diameter_m, depth_m, chamfer_m?}`.

## Why this order and these routes

1. **The handover point is chosen by the pick.** It must be low enough for
   the right arm's top-down pre-grasp: in the twin a peg centre 0.17-0.18 m
   over the bench put it out of reach, 0.11-0.12 m worked (give the pick a
   `handover_point`; its default is 0.20 m over the table).
2. **One retry of the handover, none of the pick.** A receiver miss leaves the
   peg in the left hand where it was, so trying again costs one approach. The
   other handover exits mean the peg is gone or the geometry is wrong.
3. **`unverified` goes to done.** The insertion's top probe could not tell,
   but the hand reached the depth. Re-perceive the hole before trusting it.

## Twin numbers

See the PR: the end-to-end run in the RoboSimStudio YAM twin (`SIM_GRIP=1`,
10 x 100 mm peg, square holes with 1.0 / 0.5 / 0.2 mm diametral clearance,
5 seeds each).
