---
name: grasping-a-cable-by-its-plug
description: Grasps a cable by the rigid plug at its end rather than by its
  thin jacket - finds the plug body behind the cable's tip (curve.endpoints,
  optionally refined by gemini-er.point on the plug and the depth under it),
  comes down top-down with the jaws closing across the plug's short axis and
  the fingertips just over the plug's bottom (never over its mid-height, never
  under the station's tool-centre floor), closes, lifts 4 cm and checks the
  jaw gap still matches the plug's width (and, given a description, the plug
  seen to rise in the hand). Use when a charging, USB or audio cable must be
  picked up to be plugged in or routed and its end has a moulded plug lying
  flat on the table.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [cable, plug, connector, grasp, deformable]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [cable-routing]
    provenance:
      authors: [runtime-si]
      code: https://github.com/zzigak/yam-arcade/blob/2804ddeb49ca1f00cb61b8a73c65b4a58e0190e7/stations/cables/run.py
      upstream_license: MIT
gap:
  requires: {connector: [robot.grasp_frame, robot.set_grip, robot.describe_gripper, robot.describe_workspace]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_to_joints
    - robot.solve_ik
    - robot.grasp_frame
    - robot.set_grip
    - robot.open_gripper
    - robot.close_gripper
    - robot.get_gripper
    - robot.describe_gripper
    - robot.describe_workspace
    - gemini-er.point
    - geometry.pixel_to_world_point
    - sam3.segment_text
    - geometry.mask_to_world_points
  required_inputs: {}
  produces_outputs:
    located_by: string
    grasp_point: Vec3
    close_heading_deg: float
    plug_height: float
    plug_width: float
    planned_tool_z: float
    fingertips_over_table: float
    jaw_width: float
    jaw_width_lifted: float
    plug_rise: float
    lift_checked: bool
  exit_conditions:
    grasped: The jaws closed on the plug (the gap no more than width_slack, 4 mm, under the plug's width) and still read so after the 4 cm test lift; with a plug_description the plug was also seen at least min_rise (2 cm) up in the hand (lift_checked true). The hand is left lifted, closed on the plug.
    missed: The descent stopped on the plug or the cable, the jaws closed to their stop or under the plug's width (the jacket or the plug's edge taken), closed further with the lift (the plug slipped out), or the camera saw the plug stay down; the hand opened and stands over the plug. Re-perceive before another try.
    unreachable: The plug is not resting flat on the table, the fingertips at the tool-centre floor would stand over the plug's mid-height, the plug is wider than the hand opens, no wrist solution reaches it (IK preview), or a rise, the approach, the wrist turn or the descent was refused; `detail` says which.
  canonical_scripts:
    - grasp_plug: scripts/grasp_plug.py
  examples:
    - title: Grasp the plug at a fitted cable's tip
      path: examples/canonical_subgraph.json
---

# grasping-a-cable-by-its-plug

Pick a cable up by the plug at its end. A plug is rigid and several times
wider than the 3.5 mm jacket behind it, so the YAM's tapered fingertips can
take it at the station's tool-centre floor, where they close over a flat
jacket (see grasping-a-cable-lying-flat), and the plug comes up already
held the way it will be plugged in.

## The motion

1. **Where.** The cable's plug end: a `centerline` (`curve.fit_centerline`)
   and `end_node` (0 or -1), the axis taken over its last 2 cm
   (`curve.endpoints`' tangent length); or a `curve.endpoints` `end` (its
   `position` at the tip and outward `direction`); or a given `tip` and
   `direction`. The plug body's centre is `body_offset`
   (half of `plug_length`, 20 mm) back from the tip along the axis. With a
   `plug_description`, `gemini-er.point` on the camera named `camera`
   (default the overhead one) and `geometry.pixel_to_world_point` move the
   centre to the pointed spot (if it is within 3 cm of the end's) and give
   the plug's height from the depth there.
2. **Resting flat.** A plug whose measured top stands more than 3 mm over a
   plug of its height lying on the table, or a cable end higher than that,
   is not resting on the table: `unreachable` (yam-arcade picks only plugs
   whose bottom is within 3 mm of the table).
3. **How deep.** The fingertips go 1.5 mm over the plug's bottom, the tool
   centre `robot.describe_gripper`'s fingertip reach over that, floored at
   `min_tool_height` (14 mm, a real YAM station's floor). If at the floor
   the fingertips would stand over half the plug's height, the pads would
   squeeze its upper edge: `unreachable` (yam-arcade's rule). For an 8 mm
   plug at the 14 mm floor the YAM's fingertips stand 3.8 mm up, just under
   its middle; a 6.5 mm USB-C body needs a station with a lower floor.
4. **Across the plug.** `robot.grasp_frame` with the jaws closing square to
   the plug's axis, both wrist solutions (yam-arcade tries both finger
   directions), each previewed with `robot.solve_ik` at the grasp pose and
   over it.
5. **Approach.** Up first if the hand is low; the jaws set to the plug's
   width plus 16 mm (`robot.set_grip`; yam-arcade's `PLUG_OPEN_MARGIN`);
   over the plug `approach_height` (6 cm) higher. The jaws' closing
   direction is read back and an unturned wrist (a solver that leaves the
   roll free) is turned on the last joint (`robot.move_to_joints`).
6. **Down and close.** Straight down to the grasp height. A stop more than
   6 mm over it is a fingertip on the plug or the cable: `missed`. Then
   `robot.close_gripper`. Jaws at their closed stop, or more than
   `width_slack` (4 mm) under `plug_width`: `missed` (nothing, or the
   jacket).
7. **Lift-check.** Up `lift_height` (4 cm) and the jaws read again: closed
   further than `plug_width - width_slack` is a plug that slipped out
   (`missed`); wider than `plug_width + width_slack` is noted (held askew or
   along its length). With `plug_description`, `sam3.segment_text` on the
   wrist camera and `geometry.mask_to_world_points`: the median height of
   the plug's points within 2 cm of the tool centre, in plan, must stand
   `min_rise` (2 cm) over where the plug lay.

## Inputs

- `centerline` (Centerline) and `end_node` (0, or -1 for the last node's
  end); or `end` (a curve.endpoints end, from a script); or `tip` and
  `direction` (Vec3s).
- `table_z` (else `robot.describe_workspace`'s `surface_z`).
- `plug_length` (0.020 m), `plug_width` (0.012 m), `plug_height` (else
  measured, else 0.008 m), `body_offset` (else half the length; add a metal
  shell's length when the cable's tip is the shell's).
- `plug_description` ("" skips pointing and the camera check), `camera`
  ("overhead"), `wrist_camera` ("wrist").
- `open_margin`, `tip_clearance`, `max_tip_fraction`, `approach_height`,
  `min_tool_height`, `lift_height`, `width_slack`, `min_rise`,
  `reach_tolerance`, `check_reach`, `roll_limits` (the YAM's joint 6),
  `arm_id` (0; passed only when nonzero).

## Required end states

- `grasped`: the hand stands `lift_height` over where it closed, closed on
  the plug. What the jaws read (and the camera saw, when `lift_checked`),
  not where the plug sits in the hand: re-perceive before plugging it in.
- `missed`: the hand is open over the plug. The cable may have moved.
- `unreachable`: nothing grasped; the hand is open (if it moved at all).

## What changed from yam-arcade, and why

yam-arcade's cables station (Genesis + libuipc, the YAM Ultra cart) picks a
cable out of a pile by its plug: `plan_pick_plugs` keeps plugs resting on
the table with their width axis horizontal and nothing within half the
width plus 16 mm, and grasps at the body centre across the width;
`demo_plan_plug` comes down top-down (pitched about the closing axis where
the cart's arm does not reach top-down) to put the pads' bottoms 1.5 mm over
the plug's bottom, never over its mid-height, with the jaws opened to the
width plus 16 mm, and closes with a 15 N force cap per finger (8 N let an
HDMI plug pull out as its stiff lead dragged through the pile). It reads
the plug's pose and size from the simulator and measures the slip in the
flange frame (3.7 mm and 5.9 deg in its demo, over its 2 mm limit).

Here the plug comes from perception (the cable end, optionally a pointed
plug and the depth under it); the depth is the fingertips' and floored at a
real YAM station's tool-centre floor; the pitched approach is dropped (a
YAM station's arm reaches its table top-down; an IK preview stands in for
yam-arcade's planner check); the force cap is whatever the connector's
`close_gripper` applies; and the simulator's slip measurement is a test lift
judged by the jaw gap (and the camera). The pile clearance check is not
ported: give this skill a plug that lies clear (selecting-safe-cable-grasp-
segments or a VLM check first).

## Simulation check

A YAM station twin (left arm, linear_4310 fingertips reaching 10.2 mm, the
jaws' gains 6000 / 120) with the threading twin's 3.5 mm charging cable (a
0.26 m rod of 7.2 mm capsules on VBD beside the MuJoCo arm, 2000 kg/m^3,
bending 0.08 / 0.1, the pads at mu 0.5 on it) laid straight on the
tablecloth. Its first three capsules were made a plug: each carries a welded
box, together 21.6 x 12 x 8 mm (a USB-C overmould), bottom flush with the
cable's, the joints between them 100x stiffer (yam-arcade's moulded-in
section). The simulator's capsule centres stood in for
`curve.fit_centerline` (`end_node` 0); `plug_height` 8 mm and `plug_width`
12 mm were given (as `geometry.obb` would); `min_tool_height` was the real
station's 14 mm. The outcome was measured from the plug's capsules after the
4 cm test lift and again after 2 s held up.

| layout (plug pointing) | route | fingertips over the table at the grasp | jaws closed / lifted | plug axis over the table (before -> lifted -> 2 s) | plug off square in the hand | plug centre from the tool centre, in plan |
|---|---|---|---|---|---|---|
| cable along +y (plug -90 deg) | grasped | 6.1 mm | 13.5 / 13.4 mm | 1.7 -> 39.4 -> 39.7 mm | -0.3 deg (2 s: -0.2) | 0.9 mm |
| cable at 30 deg (plug -150 deg) | grasped | 6.1 mm | 13.5 / 13.4 mm | 1.7 -> 39.3 -> 39.7 mm | -0.2 deg (2 s: -0.3) | 0.6 mm |
| cable at 160 deg (plug -20 deg) | grasped | 6.0 mm | 13.6 / 13.4 mm | 1.7 -> 38.6 -> 38.7 mm | -0.3 deg (2 s: -0.3) | 1.1 mm |
| no plug (bare 3.5 mm cable, same as the first) | missed | 6.1 mm | 0.0 mm | 1.7 -> 1.8 -> 1.7 mm | - | - |

- 3 of 3 plugs came up with the hand and stayed for 2 s, square in the jaws
  to 0.3 deg, centred to 1.1 mm, tilted 6-10 deg about the jaws' axis by the
  cable hanging from the boot. `grasped` was right each time.
- With no plug the jaws shut empty over the bare cable at the same height
  and the skill routed `missed` (closed stop): at the real floor the
  fingertips cannot take a flat 3.5 mm jacket, which is why the plug is the
  place to take it.
- The twin's arm stood 2.1-2.3 mm over the 14 mm command at the grasp
  (fingertips 6.0-6.1 mm over the table by the stated 10.2 mm reach, where
  3.8 mm was planned): the jaws closed on the plug's upper
  quarter and still held it. A real arm stands 1.7-3.2 mm over its command
  near the table, so expect the same.
- Each time the twin's solver left the wrist roll free: the jaws arrived 6,
  64 and 65 deg off the plug's short axis, and one last-joint turn brought
  them within 0.1 deg.
- The twin's jaws read 13.4-13.6 mm on the 12 mm plug, inside the 4 mm slack;
  a real YAM's reading on a plug was not measured.

The camera paths (`gemini-er.point` on the plug, the wrist-camera lift
check) did not run: this twin's observation has no cameras. They are
covered by unit tests only. Nothing here ran on a real YAM, and the twin's
plug is an idealised box with the pads' friction set for the cable's jacket.

## Provenance

Adapted from yam-arcade's `stations/cables/run.py` (zzigak/yam-arcade, MIT,
Copyright (c) 2026 zzigak; `LICENSE.yam-arcade`), with the changes above.
The wrist-roll finish and the move handling follow the hub's
grasping-a-cable-lying-flat.
