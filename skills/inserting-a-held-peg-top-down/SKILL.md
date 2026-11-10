---
name: inserting-a-held-peg-top-down
description: Inserts a peg that one arm holds top-down (tip below the fingers)
  into a perceived hole straight down - aligns the tip over the mouth from the
  hand read back, refines from the wrist camera with the tip and the mouth in
  the same depth view until the lateral error is under half the clearance,
  descends with stop-on-contact, runs a bounded spiral (and, with a chamfer, a
  tilt-then-upright try) if blocked at the mouth, pushes to the target depth
  with a tracking-lag limit, releases, backs out and probes the peg's top with
  the closed fingers to tell a tip at depth from a peg that slid up in the
  grip. Use after a top-down regrasp (handing-over-a-peg-to-a-top-down-grasp
  gives `held`) when the hole's mouth and depth are perceived (`hole`), on an
  arm with no force sensor (YAM).
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [insertion, peg-in-hole, top-down, contact, search, verification, bimanual]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [insertion]
    provenance:
      authors: [runtime-si]
      code: composed from hub skills inserting-a-peg-with-pose-refinement (ManiSkill, Apache-2.0), searching-insertion-spiral, pressing-in-with-guarded-push (o2ac, BSD-3-Clause) and correcting-insertion-offset-from-port-opening (Apache-2.0); new code is this repository's
      upstream_license: Apache-2.0
gap:
  requires: {connector: [robot.move_cartesian_until_contact]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_cartesian_until_contact
    - robot.solve_ik
    - robot.open_gripper
    - robot.close_gripper
  # Inputs (not listed: gap has no open-dict type here, so they are wired from a script node):
  # held {arm: "right", T_TCP_object: Se3Pose, tip_offset_m, axis_tilt_deg, sigma_mm} (the
  # handover's output), hole {T_world_mouth: Se3Pose (z down into the hole), diameter_m,
  # depth_m, chamfer_m?}, peg {length_m, diameter_m, mass_kg?, axis_in_object, tip_end}.
  produces_outputs:
    stage: string
    refined_by: string
    lateral_error: float
    camera_correction: float
    search: string
    probes: int
    contact_depth: float
    hand_depth: float
    tip_depth: float
    target_depth: float
    slip: float
    peak_lag: float
    released: bool
    tip_in_tcp: Se3Pose
    hole_position: Vec3
    elapsed_s: float
  exit_conditions:
    inserted: The hand pushed the tip, as held, to the target depth, released and backed out, and the closed fingers found the peg's top within slip_tolerance (2 mm) of where a tip at that depth puts it; tip_depth is from the probe.
    jammed: The tip entered the hole (at least enter_depth, 4 mm) but the lag-limited push stopped short of the target depth; the hand released the peg where it stuck and backed out.
    missed_hole: The first contact was above the mouth plane (something over the hole, or the hole estimate is high), or the tip was blocked at the mouth and the spiral (and tilt-then-upright with a chamfer) did not enter; the hand is back over the mouth, still holding.
    slipped_in_grip: The hand reached the target depth but the probe found the peg's top more than slip_tolerance higher than expected - the peg slid up the fingers and the tip is short (tip_depth); the hand is open and backed out.
    unverified: The hand reached the depth but the top probe could not tell - no peg length, release or probe turned off, the probe start out of reach, or no top within 12 mm of where it should be (the peg fell over, or went further than the hand); open and backed out unless release is false.
    unreachable: A move was refused or out of reach by the IK preview, the hand's pose could not be read, the hole's axis is more than 30 deg off straight down, or a sibling skill is not installed (raised; on_error).
  canonical_scripts:
    - insert_held_peg: scripts/insert_held_peg.py
  examples:
    - title: Insert a right-hand top-down-held peg into a perceived hole
      path: examples/canonical_subgraph.json
---

# inserting-a-held-peg-top-down

The third step of the left-side-grasp, right-top-down-regrasp, top-down-insert
strategy for elongated parts (pens, pins, dowels, plugs). The right hand holds
the peg vertically from above, the tip below the fingers; the hole's mouth is
perceived. This skill puts the tip in, pushes it home, and checks where the
tip really ended up after letting go.

It composes four hub skills, loaded from beside it (or a `GAP_SKILLS_PATH`
root), and adds the parts a held, top-down peg needs:

| stage | done by |
|---|---|
| align the tip on the hole's axis, `standoff` (25 mm) over the mouth, refine from the hand read back | inserting-a-peg-with-pose-refinement (`align_pre_insert`) |
| refine from the wrist camera: tip and mouth in the same depth view | **new** (view parsing from correcting-insertion-offset-from-port-opening) |
| descend with stop-on-contact (contact outcome, or the hand shoved off its line) | inserting-a-peg-with-pose-refinement (`push_insert`, no back-off) |
| blocked at the mouth: hexagonal spiral, nearest first | grid from searching-insertion-spiral; probes **new** (`arm_id`, tip depth as held) |
| chamfer: tilt-then-upright, four directions | **new** |
| push to the target depth with a lag limit | pressing-in-with-guarded-push (`target`, `max_lag` 4 mm) |
| release, back out, probe the peg's top | **new** (the probe is a guarded push with closed fingers) |

## Inputs

- `held` (the handover's output): `arm` ("right" -> `arm_id` 1), `T_TCP_object`
  (the peg's frame in the tool frame: origin on the axis at mid-length),
  `tip_offset_m` (used only without a peg length), `sigma_mm` (sets the spiral's
  radius: 3 sigma + 1 mm, inside 2-6 mm; and the camera's search band).
- `hole`: `T_world_mouth` (z into the hole, within 30 deg of straight down),
  `diameter_m`, `depth_m`, `chamfer_m` (optional; the tilt try needs >= 0.3 mm).
- `peg`: `length_m`, `diameter_m`, `axis_in_object` ([0, 0, 1]), `tip_end`
  ("+z" / "-z"). Without `length_m` the top cannot be probed: `unverified`.
- `insert_depth` (default: hole depth less 2 mm, capped so the tool centre
  stays `finger_clear` (15 mm) over the mouth), `enter_depth` (4 mm),
  `mouth_band` (3 mm), `depth_tolerance` (2 mm), `slip_tolerance` (2 mm),
  `max_lag` (4 mm), `camera` (default: a wrist camera of the holding arm),
  `use_camera`, `cam_steps` (6), `max_probes` (37), `tilt_deg` (3),
  `retreat` (40 mm), `top_probe`, `release`, `check_reach`, `arm_id`.

## The wrist-camera refinement

The arm's read-back cannot see a peg that sits tilted or off-centre in the
fingers, nor a hole estimate that is off. One depth frame from the holding
arm's wrist camera sees both:

- **tip**: depth points within the peg radius plus a search band of the
  predicted shaft, above the mouth plane; the lowest 3 mm of them, their
  centroid pushed away from the camera by 2r/pi (only the near half of the
  shaft is visible). Its lateral shift corrects the tip-in-hand transform once.
- **mouth**: points on the fixture's top face (within 1.5 mm of the mouth
  plane) round the estimate; per 24 angular bins the innermost is the rim; a
  Kasa circle through the rims (the peg hides some of them), accepted when its
  radius is within 30 % of the hole's.

Both are in the same camera frame, so the camera's own extrinsic error mostly
cancels in their difference. The hand steps by that difference (at most
1.5 mm a step, 6 mm in all) until it is under half the diametral clearance.
A view that does not show both, or a mouth more than 6 mm from the estimate,
leaves the arm-only alignment (`refined_by: arm`) and the spiral covers the
rest.

## Telling a slipped peg from an inserted one

Without a force sensor the hand cannot feel a peg slide up the fingers: the
push reads as free travel (pressing-in-with-guarded-push: "a part that slides
is not felt"). So after the release the closed fingers probe down onto where
the peg's top should be (a guarded push, resistance from tracking lag). The
top's height minus the peg's length is the tip's real depth: within 2 mm of
the hand's is `inserted`, higher is `slipped_in_grip`.

## Simulation check

Pending (see the PR): YAM station twin with SIM_GRIP=1, right arm.

## Limits

- Vertical holes only (within 30 deg); side insertion is
  inserting-a-peg-with-pose-refinement on its own.
- The spiral pitch is sqrt(3) x (radial clearance + chamfer), at least 0.5 mm,
  37 probes at most: a 0.2 mm hole with no chamfer relies on the camera.
- The camera refinement needs depth on a thin shaft and a flat top face round
  the hole; a dark, unlit or reflective hole may give no reading.
