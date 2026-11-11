---
name: inspecting-a-held-object-with-the-other-wrist-camera
description: Shows a face of an object one arm holds to the other arm's
  wrist camera - the holding arm carries the object to an inspection point
  (keeping its rotation), the camera arm comes to a stand-off along a viewing
  axis looking back at the object, and the framing is checked from the arms'
  own readings (object centre 0-7 cm ahead of the camera, within 2.5 cm of its
  optical ray). Use when a bimanual station must read a label or code, or look
  at a face the overhead camera cannot see (the underside, the side facing the
  table), with the held object's box from a perception skill and both arms
  addressable by arm_id.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [bimanual, inspection, wrist-camera, active-perception]
  hub:
    stage: perceive
    embodiments: [any]
    arms: 2
    version: 0.1.0
    tasks: [inspection, bimanual]
    provenance:
      authors: [runtime-si]
      code: https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/scan_object.py
      upstream_license: MIT
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_gripper
    - robot.go_to_pose
  required_inputs:
    object_obb: OrientedBoundingBox
    inspection_position: Vec3
    view_axis: Vec3
  produces_outputs:
    range_m: float
    lateral_m: float
    camera_pose: Se3Pose
    object_position: Vec3
  exit_conditions:
    inspected: Both arms arrived; the object's centre (the box centre carried with the holding hand) lies range_m (0 < range_m < max_range) ahead of the camera along its optical axis and lateral_m (< lateral_tolerance) off that ray. Take the wrist frame now; the arms stay put.
    target_not_in_view: The arms arrived but the framing test failed (range or lateral offset out of band), or the holding hand's jaws read at the stop at the start (it holds nothing); `detail` has the numbers.
    arms_too_close: The camera arm's planned tool centre is closer than min_tcp_distance to the holding hand's; the camera arm did not move (the holding arm is at the inspection point). Retry with a longer standoff or another view_axis.
    unreachable: The holding arm could not reach the inspection point, or the camera arm its stand-off (refused or ended over reach_tolerance away); also when holder_arm equals camera_arm or the connector reports no separate second arm.
  canonical_scripts:
    - inspect_held: scripts/inspect_held.py
  examples:
    - title: Look at the underside of an object arm 0 holds with arm 1's wrist camera
      path: examples/canonical_subgraph.json
---

# Inspecting a held object with the other wrist camera

RoboTwin's `scan_object` task, with the hand-held scanner replaced by the second arm's wrist camera.

1. Check both arms: different `arm_id`s, and the second arm's pose must read differently from the first
   (a connector that ignores `arm_id` reports the same arm twice). Check the holding hand's jaws are not
   at the stop.
2. Holding arm: move its tool centre so the object's box centre (held rigidly, at its current offset
   from the hand) lands on `inspection_position`, keeping the hand's rotation.
3. Plan the camera: optical centre `standoff` (5 cm) out along `view_axis` from the object, tool +z (the
   optical axis) along `-view_axis`, tool x kept level. Refuse with `arms_too_close` when that tool
   centre would be within `min_tcp_distance` of the holding hand's tool centre.
4. Camera arm: move there, read its pose back.
5. Framing (`scan_object.check_success`): project the object's centre onto the camera's optical ray;
   `inspected` when it lies `0 < d < 7 cm` ahead and every axis of the offset is `< 2.5 cm`.

## Changes from the source

- In the source the scanner is a grasped tool with a functional point. Here the camera is the second arm's
  wrist camera, and its pose comes from that arm's reading (`camera_offset`: optical centre in the tool
  frame; optical axis = tool +z).
- In the source the object and scanner poses are simulator actor poses. Here the object's position comes
  from its world box (a perception skill, e.g. registering-held-objects) carried with the holding hand.
  Nothing reads simulator truth.
- One arm moves at a time, through `robot.*` calls with an explicit `arm_id`.
- There is a minimum tool-centre distance check between the arms. For mesh-level clearance, run
  checking-dual-arm-clearance before this skill.
- A refused or short move is routed as `unreachable`, not raised.

## Notes

- To see several faces, call this skill again with different `view_axis` values (or roll the holding wrist
  between calls).
- `camera_offset` must be the station's wrist-camera extrinsic for the framing test to mean the camera
  rather than the tool centre. The default (the tool centre) suits stations whose wrist camera looks along
  the jaws, as the YAM D405 does.

## Simulation check (YAM bimanual twin, 35 mm cube held top-down by arm 0, camera arm 1, standoff 5 cm)

There are 16 runs: 4 viewing axes x 4 inspection points from the handover harness, 15 of them finished.
"Framed" is the framing test recomputed from simulator truth (true cube centre against arm 1's reading).

| view_axis | inspected | truth framed | other exits |
|---|---|---|---|
| side (-y, toward the camera arm) | 2/4 | 3/4 | 1x holding arm timed out at point 1; 1x camera move timed out (true framing still in band) |
| back (-x, camera looking away from the bases) | 1/2 | 1/2 | 1x holding arm timed out at point 1 |
| under (-z, camera looking up) | 0/4 | 0/4 | 4x unreachable; in 3/4 runs the arms came 0.2-0.4 mm apart and the cube dropped |
| front (+x, camera looking back at the bases) | 0/4 | 0/4 | 4x unreachable (IK; the arms came 0.1-0.2 mm apart in 2/4) |

- Every `inspected` exit agreed with truth: range 48.2-48.3 mm, lateral 1.2-2.6 mm. The holding arm put the
  cube within 2.0-2.7 mm of the inspection point.
- The tool-centre distance check (`min_tcp_distance`) does not catch link contact. The measured mesh
  clearance reached 0.1-0.4 mm on under / front / back runs while the tool centres stayed over 40 mm apart.
  Run checking-dual-arm-clearance on the camera arm's path first. Prefer side or back views; a YAM wrist
  cannot point straight up at an underside held 15-30 cm over the bench.
