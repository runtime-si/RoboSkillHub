---
id: curobo-2023
title: "cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation"
authors:
  - Balakumar Sundaralingam
  - Siva Kumar Sastry Hari
  - Adam Fishman
  - Caelan Garrett
  - Karl Van Wyk
  - Valts Blukis
  - Alexander Millane
  - Helen Oleynikova
  - Ankur Handa
  - Fabio Ramos
  - Nathan Ratliff
  - Dieter Fox
year: 2023
arxiv: "2310.17274"
url: https://arxiv.org/abs/2310.17274
code: https://github.com/NVlabs/curobo
license: Apache-2.0
tasks: [motion-planning]
derived: [tools/curobo]
---

# cuRobo

## The method

Collision-free motion generation for a manipulator posed as one global
optimization and solved with many seeds in parallel on a GPU. Step directions
come from L-BFGS, a parallel noisy line search picks step sizes, and a
particle-based optimizer seeds the trajectory optimizer; a parallel geometric
planner and a collision-free IK solver sit beside it. The robot is modelled as
spheres, checked against the world's cuboids, meshes or depth.

The paper reports difficult problems solved in about 50 ms on average, 60x
faster than the trajectory optimizers it compares against, and over 7000 IK
queries per second.

## What the hub uses

The `curobo` tool bundle (from open-robot-skills) wraps the library as RPC
tools: `curobo.plan_to_pose`, `curobo.plan_to_grasp_poses` (a goal set),
`curobo.plan_grasp_motion` (approach, grasp, lift), `curobo.plan_directed_linear`,
`curobo.plan_with_grasped_object`, `curobo.solve_ik`, and trajectory validation
with and without a held object. Grasp and transport skills call them; a recipe
reaches them through those skills.

## When it is the wrong tool

- **No collision world, no guarantee.** Plans are collision-free against the
  world it is given. On a real cell that world is reconstructed from RGB-D
  (`geometry.build_world_config`); what the cameras miss, the planner misses.
- **A new arm needs a sphere model.** The robot's collision spheres come from
  its cuRobo config file. The bundle plans for cuRobo's built-in Panda config;
  yam, or any other arm, needs its config registered with the bundle first.
- **It needs a CUDA GPU.** For a CPU-only machine, the engine's PyRoKi backend
  is the fallback.
