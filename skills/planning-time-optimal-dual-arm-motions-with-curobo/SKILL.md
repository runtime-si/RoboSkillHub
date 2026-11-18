---
name: planning-time-optimal-dual-arm-motions-with-curobo
description: Plans both YAM arms at once as one 12-DoF cuRobo robot (inter-arm collision checked on the
  tools/yam-spheres model with its 15 mm margin) and times the path for the shortest makespan under joint
  velocity, acceleration and jerk limits, returning per-arm joint tracks for the station's stream_joints
  op and timed waypoints for move connectors. Use when both arms must reach goals together and total task
  time matters more than moving one arm at a time; needs a CUDA GPU with tools/curobo's cuRobo.
license: Apache-2.0
compatibility: requires gap>=0.1, CUDA GPU, nvidia-curobo (tools/curobo pin)
metadata:
  category: motion
  tags: [motion, dual-arm, curobo, time-optimal, yam]
  hub:
    stage: plan
    embodiments: [yam-bimanual]
    version: 0.1.0
    tasks: [bimanual]
    provenance:
      authors: [runtime-si]
gap:
  allowed_tools: []
  required_inputs:
    start: "{left: 6 model joint angles, right: 6}"
    goal: "{left: 6 model joint angles, right: 6}"
  produces_outputs:
    tracks: "list of stream_joints chunks {left: [[j1..j6, grip_open]], right: [...]}, <= 64 rows, hardware angles"
    waypoints: "timed joint waypoints per arm (model angles)"
    makespan_s: float
    min_inter_arm_sphere_m: float
  exit_conditions:
    planned: A collision-free 12-DoF trajectory was found and timed; `tracks` hold it at `hz` (default 30).
    no_path: cuRobo found no feasible trajectory, or the goal itself collides (`detail` says which).
    start_in_collision: The start overlaps on the sphere model (arm-arm or within an arm); `detail` lists the link pairs.
  canonical_scripts:
    - plan_dual_arm: scripts/plan_dual_arm.py
---

# planning-time-optimal-dual-arm-motions-with-curobo

Move both arms at the same time instead of one after the other.

## How

1. Build one robot from two prefixed copies of `tools/curobo/robots/yam/yam_i2rt_arm.urdf` on the station
   bases from `tools/yam-spheres/yam_spheres.json` (left (-0.3475, +0.305, 0.023), right (-0.3475, -0.305,
   0.023)), with the merged spheres (link_6 = gripper + open fingers). Self-collision ignores only links one
   or two apart within an arm, so arm-arm pairs are all checked.
2. `MotionPlanner.plan_cspace` (cuRoboV2 at tools/curobo's pin) finds the 12-DoF path.
3. Re-time it: forward/backward pass on the path parameter for the fastest timing under per-joint
   |v| <= `vmax`, |a| <= `amax`, then a uniform slow-down until |jerk| <= `jmax`. Both arms share the clock.
4. Sample at `hz` and split into `stream_joints` chunks (joint 1 converted to hardware: + pi/2).

Limits: `vmax` 0.25 rad/s (dual_motion_server's default `--stream-max-vel`), `amax` 1.0 rad/s^2, `jmax`
10 rad/s^3 (the station config has no acceleration or jerk limits; these are conservative choices).

## Executing

Send each chunk in one `stream_joints` request with both arms (`tracks`, `hz`, `mode="abs"`, fresh `t_obs`),
per `yam_gap/skill_sources/DUAL_ARM_SERVER.md` section 9B; the server must run with `--allow-stream
--stream-arms both`, and it re-paces rows to its own vmax. `waypoints` serve `move` / `go_to_pose`
connectors when streaming is off (the arms then move one at a time on the current server).

## Caveats

- The sphere model flags the folded home (model zero) as self-colliding (link_2/link_5, a 4.3 mm false
  positive from the margin): start from the working pose.
- Goals are joint angles; get them from IK (`plan` replies' `joints_rad`, or cuRobo IK) first.
- `scripts/bench.py` is the GPU benchmark against planning each arm alone and running them in sequence.
