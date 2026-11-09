---
name: curobo
description: NVIDIA cuRobo motion planning — collision-free trajectories to
  grasp goalsets, transport with an attached object, constrained linear
  moves, single-pose planning, geometric IK, batch grasp feasibility, and
  joint-trajectory collision validation. Use when a workflow needs
  GPU-accelerated, collision-aware arm motion plans.
license: MIT for this bundle's code (as open-robot-skills declared it); the NVIDIA cuRobo it
  installs and calls (NVlabs/curobo@4ea7736, nvidia-curobo) is Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [planning, motion, gpu, cuda]
  hub:
    stage: plan
    embodiments: [any]
    tuned_on: [panda]
    tasks: [motion-planning]
    notes: Plans for curobo's built-in franka.yml. It knows yam.yml is 6-DoF but does not ship that config; plan_with_grasped_object loads one registered in _PROJECT_ROBOT_CONFIGS, the other v0.8 planners only resolve curobo's built-in configs.
    version: 0.2.0
    provenance:
      code: https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/tools/curobo
      upstream_license: MIT (the open-robot-skills bundle); cuRobo itself, https://github.com/NVlabs/curobo/tree/4ea77366ca48ee453e7df139e39fa6532af49f3b, is Apache-2.0
gap:
  requires: {gpu: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "curobo"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - curobo.plan_to_grasp_poses: Collision-free trajectory to one of several grasp poses (goalset).
    - curobo.plan_with_grasped_object: Transport plan with the held object attached to the gripper (failure_reason on failure).
    - curobo.plan_linear: Straight Cartesian trajectory between two EE poses.
    - curobo.plan_directed_linear: Constrained linear motion with axis/orientation holds.
    - curobo.plan_grasp_motion: Approach + grasp + lift sequence (three trajectories).
    - curobo.plan_to_pose: Collision-aware plan to a single TCP pose.
    - curobo.solve_ik: Geometric IK for a single TCP pose (no world collision).
    - curobo.batch_grasp_feasibility: Per-pose grasp/approach IK + corridor feasibility for a grasp batch.
    - curobo.validate_joint_trajectory_robot: Collision-validate joint waypoints (robot vs world + self).
    - curobo.validate_joint_trajectory_grasped: Same, with a grasped object attached at waypoint 0.
    - curobo.cloud_to_attachment: MORPHIT collision spheres for a held object's cloud, in the TCP frame.
---

# curobo

Collision-aware cuRobo motion planning as in-process tools. Trajectories
in/out are gap `Trajectory` dicts (`waypoints: [{positions:
float64[dof]}]`); worlds are gap `WorldConfig` dicts (build one with
`geometry.build_world_config`).

## When to use

- Tabletop pick: `geometry.top_down_grasp_candidates` →
  `curobo.plan_to_grasp_poses` (pass the whole fan as the goalset; check
  `goalset_index` for which one was reached).
- Transport after grasping (carrying a brick or a tile): see "Held-object
  planning" below.
- Drawers/doors: `curobo.plan_grasp_motion` (approach → grasp → lift/pull
  with gripper commands interleaved), or `curobo.plan_directed_linear` for
  a pull along one axis with orientation locked.

## Install

cuRobo JIT-compiles CUDA extensions at install time — build isolation must
be off and `CUDA_HOME` must point at a toolkit matching your torch build:

```bash
export CUDA_HOME=/usr/local/cuda     # toolkit matching torch's CUDA version
uv sync --extra curobo               # (pip: pip install -e ".[curobo]" --no-build-isolation)
```

If the import fails at tool-call time the tools raise a `ToolError` with
this recipe. First planner call per process pays JIT/warmup latency; the
MotionGen/planner instances are cached and reused (HyRL pattern — recreating
them per call corrupts CUDA graph state).

## Gotchas (carried over from the service + curobo_api)

- **Frames**: grasp/target poses are in the robot-base frame (cuRobo treats
  the robot base as world origin). With `grasp_pose_is_fingertip=True`
  (default) grasp positions are fingertip-pad centers and converted to the
  `panda_hand` frame solver-side (offset 0.1029 m along hand Z).
- **Ignore the grasp target**: pass its mesh name in
  `ignore_obstacle_names` for `plan_to_grasp_poses` /
  `batch_grasp_feasibility` — closing on the target is not a collision.
- `robot_collision_sphere_buffer` default −0.01 shrinks robot collision
  spheres 1 cm; reduces IK_FAIL against dense perception meshes. Negative
  is intentional.
- GPU access is serialised by a module lock (cuRobo is not thread-safe);
  CUDA/"graph capture" errors invalidate the cached planners automatically
  before raising `PlanningFailed`.
- `use_cuda_graph` must stay False for the validators
  (`check_start_state` requirement) and for varying world/start setups.
- **curobo version split**: `plan_to_grasp_poses`, `plan_grasp_motion`,
  `plan_directed_linear`, `plan_linear`, `plan_to_pose`,
  `plan_with_grasped_object` target curobo v0.8 (MotionPlanner API;
  the held-object planner uses its AttachmentManager);
  `solve_ik` and `batch_grasp_feasibility` are built on the v0.7
  IKSolver API that v0.8 removed — on a v0.8-only install they raise
  `PlanningFailed` ("not supported on cuRobo v0.8"); plan via the goalset
  tools instead. The validators use the v0.7 MotionGen path too.
- `validate_joint_trajectory_grasped` always invalidates the planner cache
  afterward so the attachment cannot leak; expect the next planning call to
  re-create the planner.
- Planning failures return `success=False` (with `failure_reason` where the
  RPC had one); infrastructure errors raise `PlanningFailed` / `ToolError`.
- Set `debug_out_dir` on the grasp planner to dump world + robot sphere
  OBJ/PLY debug artifacts on failure (default `./curobo_debug*`); the
  transport planner writes only its `events.jsonl` result log there.

## Held-object planning

`curobo.plan_with_grasped_object` plans a carry with the held object as part
of the robot:

- Put the held object in `world_config` as a mesh, posed where it is held at
  `start_joint_position`, and pass its name as `object_name`. It is left out
  of the obstacle set; every other mesh is an obstacle.
- cuRobo's AttachmentManager fits MORPHIT spheres to it (8 asked for, 16
  slots reserved on `link_name`), the radii are grown until the spheres
  enclose the mesh's sampled surface, and they ride on `link_name`, which
  must be rigidly at the robot's first tool frame (`attached_object` on
  `panda_hand` for franka.yml). The spheres are read back through FK; if they
  land more than 1 mm from where they were fitted the call fails
  `attach_frame_mismatch` rather than carry a misplaced object.
- `target_pose` is the tool-frame goal in the robot-base frame. The object is
  detached after every call, whatever happened.
- Failures come back as `success=False` with `failure_reason`:
  `object_name_required`, `object_not_in_world`, `attach_link_missing`,
  `attach_failed`, `attach_frame_mismatch`, `ik_failed` (no collision-free
  IK with the object attached), `trajopt_failed` (with the best pose error),
  `no_trajectory`. Nothing falls back to planning without the object.
- The fit is cached per object geometry (MORPHIT takes about 2 s) and evicted
  when a plan fails. MORPHIT is stochastic and on a tight carry the fit
  decides the outcome, so retry a failed `trajopt_failed` call once or twice.
- `robot_collision_sphere_buffer` resizes the robot's spheres only; the held
  object's are as fitted. `remove_obstacles_from_world` is accepted and
  ignored (the held mesh is always left out). `position_threshold_z` widens
  the single v0.8 position tolerance (the max of the two is used).
- At the pinned commit `MotionPlanner.attachment_manager` points at an
  attribute TrajOptSolver does not have; the bundle reaches the manager on
  the solver's core instead.

Measured on an RTX 4090 D (AutoDL) with a YAM arm (config generated by
`tests/bundles/_curobo_attach_live.py` from robosimstudio's yam_i2rt URDF and
meshes; the bundle does not ship it), carrying a held 30 mm cube from
(0.30, 0.15, 0.06) to (0.30, −0.15, 0.06) m over an 8 cm wall, gripper down.
The script refits the YAM spheres (MORPHIT) on every run, and the results
moved with that fit:

| | run A (the committed script) | run B (earlier YAM fit, GPU shared) |
| --- | --- | --- |
| carries, fresh object fit each | 10/10 | 6/10 (4× `trajopt_failed`) |
| carries on the cached fit | 5/5 | 5/5 |
| median time, fresh / cached | 3.1 s (0.8 s fit + 2.3 s plan) / 1.1 s | 9.4 s (2.0 + 6.8) / 4.2 s |
| over a 10 cm wall instead | 3/3 | 0/3 |

In run A the planned carry keeps the cube 13.8 mm from the wall and 45 mm
from the table (the straight line would put it 19.7 mm into the wall), ends
within 0.001 mm of the target, and the tool turns up to 13° on the way. A
robot-only plan of the same query passes the cube's position 4.0 mm from the
8 cm wall and 2.1 mm *into* the 10 cm one. The cube's 8 spheres (radii
11.5–14.3 mm) leave at most 0.5 mm of its sampled surface outside and
protrude up to 6.7 mm beyond its faces. A missing object returns
`object_not_in_world`, a target inside the wall `ik_failed`; targets 5–20 mm
above the wall top are `ik_failed` for the held plan and robot-only alike
(the fingers), 40 mm above succeeds for both.

## Licence

The bundle's code keeps the MIT license open-robot-skills declared for it.
What it installs and calls, NVIDIA cuRobo at the pinned commit
[4ea7736](https://github.com/NVlabs/curobo/tree/4ea77366ca48ee453e7df139e39fa6532af49f3b)
(the v0.8.0 tag), is Apache-2.0: its `LICENSE` there is the Apache License
2.0 (sha256 `f39b6dc8…7b45c`) and its `pyproject.toml` declares
`Apache-2.0`. The robot assets it ships (franka_description, used through
`franka.yml`) are under its `LICENSE_ASSETS` (franka_ros 0.7.0,
Apache-2.0).
