---
name: assigning-motion-segments-to-arms-with-handovers
description: Chooses which arm carries an object along each part of its
  path and where to hand it over - a Viterbi DP over the path's waypoints
  whose state is the carrying arm, with robot.solve_ik for each arm at each
  waypoint (unreachable = impossible), the joint motion between waypoints as
  the cost and a handover (only where both arms reach) as a fixed penalty.
  Use when an object must travel across a two-arm workcell farther than one
  arm reaches (left side to right side), before executing the carry with
  handing-over-between-arms at the returned handover poses.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [bimanual, handover, planning, assignment, dynamic-programming, ik]
  hub:
    stage: plan
    embodiments: [any]
    arms: 2
    version: 0.1.0
    tasks: [pick-place, bimanual]
    provenance:
      authors: [runtime-si]
      code: https://github.com/Sukhvansh2004/Bimanual-Manipulation/blob/d7940e7e2819/src/bimanual_manipulation/planning/assignment/solver.py
      upstream_license: MIT
gap:
  allowed_tools:
    - robot.solve_ik
  required_inputs: {}
  produces_outputs:
    n_handovers: int
    total_cost: float
  exit_conditions:
    assignment_found: One arm carries the whole path (cheapest with no handover); assignment lists it per waypoint.
    handover_required: The cheapest path switches arms; handovers lists each (waypoint index, from_arm, to_arm, position), every one at a waypoint both arms reach.
    infeasible: No arm sequence covers every waypoint - a waypoint no arm reaches (unreachable_waypoints), or reachable ones with no shared waypoint to hand over at.
  canonical_scripts:
    - assign_arms: scripts/assign_arms.py
  examples:
    - title: Carry left to right across the station
      path: examples/canonical_subgraph.json
---

# assigning-motion-segments-to-arms-with-handovers

## What it does

`waypoints` (Vec3 or poses; `rotation` for all of them otherwise, e.g. a
top-down grasp frame) are checked with `robot.solve_ik` for every arm in
`arms` (default [0, 1]). DP: the cost of carrying waypoint i with arm k is
the best of staying (the summed joint change from i-1, times
`joint_weight`) or switching from arm j - allowed only if arm k also reaches
waypoint i-1, where the handover happens - for `handover_cost` (2 rad of
joint motion) plus the joint change. Backtracking gives `assignment` (an arm
id per waypoint), `handovers`, `total_cost`. `initial_arm` fixes the arm
that starts (the one already holding).

## Checks

FakeContext unit tests (`tests/bundles/test_assigning_motion_segments_to_arms_with_handovers.py`):
one-side carry, crossing with one handover where both reach, a gap no arm
reaches, raised IK, `initial_arm`, the handover cost against joint motion,
and a left-to-right carry with the CPU YAM model.

**Synthetic carries** (`python -m tests.bundles._assign_arms_eval`, CPU):
30 random straight carries, 12 waypoints each, start and
end uniform in x 0.15-0.42 m, y -0.45-0.45 m, z 0.05-0.20 m (arm 0's base
frame; bases at y = +-0.305 m). One arm alone can carry 13 of 30. The DP
completes 15 of 30: 12 with one arm (`assignment_found`) and 3 with exactly
one handover (`handover_required`; 2 of these 3 have no single-arm solution,
1 hands over because it is cheaper in joint motion). 15 are `infeasible`, 14
of them because some waypoint is out of both arms' reach (the far corners at
x > 0.4 m), 1 because the two reachable stretches share no waypoint. Reachability is a CPU model, not the twin's IK: forward kinematics from
robosimstudio's `yam_bimanual_arm.urdf`, a tool point 0.13 m past link 6
(approximate), damped-least-squares IK with the tool pointing down within
10 deg, two bases 0.61 m apart (the station twin's spacing).

## Limits

- One IK solution per arm and waypoint (upstream keeps several); a path
  that one arm could follow only through a different solution branch is
  judged on the first solution.
- Not ported: grasp feasibility (which side each hand can hold), collision
  between the two arms at the handover (use checking-dual-arm-clearance),
  manipulability, and upstream's leader-follower execution.
- No twin or real run: the output is a plan.

## Provenance

Bimanual-Manipulation (github.com/Sukhvansh2004/Bimanual-Manipulation,
`planning/assignment/solver.py` and `cost.py`, MIT,
LICENSE.bimanual-manipulation): the DP over (segment, arm) states with
segment and switching costs. The header of `scripts/assign_arms.py` lists the
changes.
