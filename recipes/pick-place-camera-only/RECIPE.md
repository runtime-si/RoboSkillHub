---
name: pick-place-camera-only
description: Put one named rigid object into a named container from the cameras
  alone -- perceive both, grasp across the object's short axis, fall back to a
  planner-chosen grasp if that fails or does not hold, confirm the grasp, then
  carry and release. Use when a single arm must move one object into a
  container and no simulator ground truth is available, as on a real cell.
license: Apache-2.0
metadata:
  hub:
    version: 0.1.0
    embodiments: [any]
    tuned_on: [panda]
    tasks: [pick-place]
    tier: camera_only
recipe:
  inputs:
    object_description: string
  start: perceive_target
  nodes:
    perceive_target: {skill: perceiving-objects, bind: {name: target}}
    perceive_container: {skill: perceiving-objects, bind: {name: container}}
    grasp: {skill: grasping-short-axis}
    confirm: {skill: verifying-grasps}
    regrasp:
      skill: grasping-with-planner
      note: A different grasp generator, so the second attempt is not the first one repeated.
    reconfirm: {skill: verifying-grasps}
    carry: {skill: transporting-objects}
    done: {end: success}
    abort: {end: failure, recovery: [robot.open_gripper]}
  routes:
    perceive_target: {found: perceive_container, not_found: abort}
    perceive_container: {found: grasp, not_found: abort}
    grasp: {grasped: confirm, failed: regrasp}
    confirm:
      {verified: carry, held: carry, reached: carry, unknown: carry, not_held: regrasp, missed: regrasp, empty: regrasp}
    regrasp: {grasped: reconfirm, failed: abort}
    reconfirm:
      {verified: carry, held: carry, reached: carry, unknown: carry, not_held: abort, missed: abort, empty: abort}
    carry: {placed: done, blocked: abort}
---

# Pick and place, cameras only

The shape a coordinator should reach for when the task is "put X in Y" on a
real cell: every skill here reads the scene through the cameras and the robot
through its `describe_*` tools, so the generated graph can leave simulation
(`openskills lock --deploy` passes).

## Why this order

1. **Both perceptions come first.** `transporting-objects` needs the
   container's box and mask as well as the object's; perceiving the container
   before grasping keeps the arm out of the camera's way while it looks.
2. **Two grasp generators, not one tried twice.** `grasping-short-axis` closes
   across the object's narrow side, which suits boxes and bottles;
   `grasping-with-planner` scores a fan of top-down candidates against the
   collision world. A miss with the first is often a geometry the second
   handles.
3. **A confirmation after each grasp, and no loop.** The second confirmation
   routes every failure to `abort`, so the graph ends instead of retrying
   forever.

## Routing the confirmation

`verifying-grasps` has seven exits because its scripts check different
moments. After a close, `verified` and `held` mean the object is in the hand.
`reached` and `unknown` (the gripper could not be read) also go on to `carry`:
a dropped object then shows as `blocked` there, which costs less than
abandoning a grasp that was in fact good.

## Before running on a new arm

The grasp and transport skills' constants were measured on a Franka Panda in
LIBERO (`tuned_on: [panda]`). On another arm, re-measure the gripper settle
times and approach offsets those bundles document in their `references/`, and
check the result in simulation before the real cell.
