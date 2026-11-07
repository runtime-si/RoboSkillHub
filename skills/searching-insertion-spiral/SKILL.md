---
name: searching-insertion-spiral
description: Finds an opening the held part or the closed fingertips can enter
  by probing straight down over a hexagonal grid around the opening's
  estimated mouth, nearest points first, stopping on contact and judging each
  probe by how far below the mouth the hand got - no force sensor. Use when a
  vertical insertion's target is known only to a few millimetres (a camera's
  estimate of a hole, a slot, a ring's opening, a socket), the hand already
  holds the part over it or is empty with the fingers closed, and the arm can
  stop on contact.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [insertion, search, contact, spiral, peg-in-hole]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [insertion]
    provenance:
      authors: [runtime-si]
      paper: arXiv:2206.08800
gap:
  requires: {connector: [robot.move_cartesian_until_contact]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.go_to_pose_cartesian
    - robot.move_cartesian_until_contact
    - robot.solve_ik
  required_inputs:
    mouth_position: Vec3
  produces_outputs:
    entered_at: Vec3
    depth: float
    probes: int
  exit_conditions:
    inserted: A probe's tool centre got at least enter_depth (default 5 mm) below the mouth; the hand stays there unless stay_in is false.
    search_exhausted: Every reachable probe inside search_radius stopped short of enter_depth; `detail` gives the count and the deepest probe.
    unreachable: No probe position could be reached, or the hand's pose could not be read.
  canonical_scripts:
    - spiral_search: scripts/spiral_search.py
  examples:
    - title: Search for a hole whose mouth a perception skill estimated
      path: examples/canonical_subgraph.json
---

# searching-insertion-spiral

A camera puts a hole within a few millimetres; a part with a millimetre of
clearance needs better. Instead of measuring the last millimetres, try them:
probe down at each point of a small hexagonal grid around the estimate, and
the first probe that sinks below the mouth has found the opening.

## The motion

For each grid point, nearest the estimate first:

0. Ask `robot.solve_ik` whether `enter_depth` below the mouth can be reached
   there, and skip the point if not (`check_reach`, default true). That depth
   is what proves entry; the probe itself still heads for `max_depth`. A probe the arm cannot
   finish stalls, and a stall looks exactly like contact: without this check
   an out-of-reach region reads as solid surface. A connector whose IK gives
   no usable answer is not skipped on its account.
1. Move straight to `hover` above the mouth at that point
   (`robot.go_to_pose_cartesian`). A point the hand ends more than 3 mm away
   from is skipped.
2. Probe straight down toward `max_depth` below the mouth with
   `robot.move_cartesian_until_contact`.
3. Read the hand's pose. Whatever ended the probe - arrival, a contact or
   stall outcome, or a connector's "did not reach its target" error - the
   depth reached is what counts.
4. At least `enter_depth` below the mouth: `inserted`. Otherwise lift back to
   the hover and take the next point.

The hand keeps the rotation it starts with. The axis is vertical.

**The fingers must be closed or holding the part.** An open hand straddles a
thin rim or a small object and sinks around it: in a YAM station twin, an
open gripper probing the top of a 30 mm cube "entered" 19.5 mm below its top
without moving it. This skill does not close the hand itself, since that would
drop a held part.

**Depth is the tool centre's.** On a closed YAM linear_4310 the fingertips
reach about 10 mm below it, so a probe that lands on a surface reads about
-10 mm; set `enter_depth` and `max_depth` for the held part's own tip.

## Inputs

- `mouth_position`: the opening's estimated centre at its mouth (the rim's
  height), world frame.
- `search_radius` (6 mm), `pitch` (2 mm), `hover` (10 mm), `enter_depth`
  (5 mm), `max_depth` (15 mm): see the constants.
- `stay_in` (true): leave the hand in the opening on `inserted`, ready to push
  further or release; false lifts it back out.
- `check_reach` (true): the IK preview of step 0.

## Required end states

- `inserted`: `entered_at` is where the tool centre stopped, `depth` how far
  below the mouth, `probes` how many were made.
- `search_exhausted`: the opening is not within `search_radius` of the
  estimate, or the part does not fit. Re-perceive and search again, or widen
  the radius.
- `unreachable`: the region is out of the arm's reach.

## The constants, and why

- **`SEARCH_RADIUS = 6 mm`**: about two standard deviations of a calibrated
  YAM station's hand-camera error (RMS 5.4 mm over the workspace, 2-3 mm near
  the table).
- **`PITCH = 2 mm`**: a hexagonal grid leaves every point within
  `pitch / sqrt(3)` (1.15 mm) of a probe, so a part with that much clearance
  is entered by some probe. 37 probes cover the default radius.
- **`ENTER_DEPTH = 5 mm`**, **`MAX_DEPTH = 15 mm`**: deeper than a rim's
  chamfer and the arm's contact-stop overshoot, shallow enough not to drive a
  part that missed into the surface around the hole.
- **`HOVER = 10 mm`**: clears a rim between probes without long travel.
- **`HOVER_TOLERANCE = 3 mm`**: a probe started further off tests another point.
- **Why the IK preview**: in a YAM station twin, probes over a box between the
  two arms, near the left arm's reach, all stopped 13-26 mm above the rim,
  including probes over the box's open middle and over the bare table beside
  it - stalls at the edge of reach, read as contact. Previewing the full
  `max_depth` instead of `enter_depth` was too strict: beside a cube on the
  table it skipped 13 of 14 probes the arm could in fact have used.

## Checked in simulation

RoboSimStudio, a YAM station twin, left arm, closed empty hand turned top-down
over the estimate, grid radius 48 mm and pitch 24 mm, `enter_depth` 12 mm:

| estimate placed on | probe 0 (on the estimate) | probe 1 (24 mm off) |
|---|---|---|
| a 30 mm cube's top | stopped 9.2 mm above the top | sank 16.9 mm below it: `inserted` |
| a box wall's rim | stopped 9.7 mm above the rim | sank 22.7 mm below it: `inserted` |

## Method

The probe-and-sink search follows Haugaard et al., "Fast robust peg-in-hole
insertion with continuous visual servoing" (arXiv:2206.08800), which reaches
the hole with visual servoing and finishes with a spiral search; their code
carries no licence, so this is a reimplementation from the description, with
contact read from the arm's own stop rather than a force-torque sensor.
