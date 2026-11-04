---
name: {{name}}
description: <The task, and the shape of the graph that solves it, in one or two
  sentences.> Use when <the instruction or scene that calls for this shape>.
license: Apache-2.0
metadata:
  hub:
    version: 0.1.0
    embodiments: [yam_i2rt]
    tasks: []
recipe:
  inputs:
    object_description: string
  start: perceive
  nodes:
    perceive: {skill: perceiving-objects, bind: {name: target}}
    grasp: {skill: grasping-direct-ik}
    done: {end: success}
    abort: {end: failure, recovery: [robot.open_gripper]}
  routes:
    perceive: {found: grasp, not_found: abort}
    grasp: {grasped: done, failed: abort}
---

# {{name}}

<Why this order of skills, what each fallback is for, and what to re-measure
before running it on a new arm.>
