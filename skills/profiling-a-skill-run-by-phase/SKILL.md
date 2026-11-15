---
name: profiling-a-skill-run-by-phase
description: Runs one hub skill with its ctx wrapped so every robot.* and tool
  call is timed (wall clock, and sim time when a sim clock is given), and splits
  the run into motion, settle/wait, gripper, perception, planning and other
  phases; writes a per-phase table and a timeline JSON. The same script profiles
  a recorded real run offline from the motion server's events.jsonl and the
  client's moves.json, per pen. Use when a skill or task is slower than
  expected and you need to know where the seconds go before optimising it.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: diagnostics
  tags: [profiling, timing, makespan, diagnostics, cpu]
  hub:
    stage: verify
    embodiments: [any]
    arms: 1
    version: 0.1.0
    tasks: [diagnostics, pick-place]
    provenance:
      authors: [runtime-si]
    notes: Phase is by tool name; gaps after a motion count as settle, other gaps as client time (other).
gap:
  allowed_tools:
    - robot.get_observation
    - robot.describe_robot
    - robot.describe_arm
    - robot.forward_kinematics
    - robot.solve_ik
    - robot.get_ee_pose
    - robot.get_gripper
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_to_joints
    - robot.execute_trajectory
    - robot.open_gripper
    - robot.close_gripper
    - robot.grasp_frame
  required_inputs:
    skill: string
  produces_outputs:
    child_route: string
    child_outputs: string
    phase_table: string
    timeline_json: string
  exit_conditions:
    profiled: The skill ran to its own exit (child_route) without raising; phase_table and timeline_json cover every call it made.
    failed: The skill raised; the table and timeline cover the calls up to the error, and child_outputs holds it.
    refused: The skill could not be loaded or inputs_json is not JSON; nothing was run.
  canonical_scripts:
    - profile_skill: scripts/profile_skill.py
---

# profiling-a-skill-run-by-phase

`run(ctx, skill, inputs_json="{}", out_path="")` loads `skill`'s canonical
script (from this hub or `GAP_SKILLS_PATH`), hands it a `Profiler` instead of
the ctx, and records each `ctx.tool` call: name, phase, start and end (wall,
and sim when `Profiler(ctx, sim_clock=...)` is built directly), arm, thread,
and the current segment (`prof.mark("obj3")`). Everything else on the ctx is
forwarded, so the child does not know it is being watched; threads are kept
apart, so it also wraps `running-two-skills-concurrently-on-two-arms`.

## Phases

| phase | what |
| --- | --- |
| motion | `go_to_pose*`, `execute_trajectory`, `move_*`, `stream_*`, `reset`; for a reply carrying `arrived_s` / `motion_s` / `duration_s`, only up to the server-side arrival |
| settle | a motion's time past its arrival, the gap after a motion until the next command, and `wait*` / `sleep*` calls |
| gripper | `open_gripper`, `close_gripper`, `set_gripper*` (grip ops offline) |
| perception | `get_observation`, `get_image*`, detectors, VLMs, SAM, `get_object_obb` |
| planning | `solve_ik*`, cuRobo, `plan*`, `forward_kinematics`, `grasp_frame` |
| other | state reads (`get_ee_pose`, `get_gripper`, `describe_*`) and client time between calls that does not follow a motion |

A tool bundle's call (`<bundle>.<func>`) is classed by its function name, then
its bundle name (`curobo`, `*plan*` planning; `vlm`, `sam*`, detectors
perception).

## Outputs

- `phase_table`: per phase, calls, wall seconds and share (sim seconds when a
  sim clock was given). `phase_table(timeline, by="segment")` splits it per
  object.
- `timeline_json`: the calls and the gaps between them, in order, each with
  `phase`, `start_s`, `end_s` (`kind: gap` / `after_arrival` for derived
  rows).

## Recorded real runs

```bash
python scripts/profile_skill.py --events recordings/motion_X/events.jsonl \
    --moves runs/real-Y/moves.json --json profile.json
```

Server ops `move` / `reset` are motion and `grip` is gripper, from their
`duration_s`. The gap after a move is settle for the `settle_s` moves.json
recorded for it (matched by label prefix within 3 s), the rest is client time:
perception after a `look ...` / `survey ...` move (the client images and
detects there), otherwise other (planning, detection, logging; the server log
cannot tell them apart). Segments are pens (`pen NN` / `pen_NN` in the label;
`setup` before the first).

## Limits

- Phase comes from names. A tool that plans and moves in one call is all
  motion.
- In a simulator whose moves return on arrival, settle is the gap until the
  next command; nothing inside the call is split off.
- The wrapper adds about 2 us per call.
