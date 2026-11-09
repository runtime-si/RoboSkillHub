---
name: workcell
description: Typed JSON boundary for RoboSkillHub workcell capabilities and station descriptions. Use when an agent or graph must call an adapted YAM or Isaac capability through one interface and retain its measured result.
license: Apache-2.0
compatibility: requires the installed RoboSkillHub runtime and configured station
metadata:
  category: integration
  tags: [workcell, yam, isaac, runtime, feedback]
  hub:
    stage: execute
    embodiments: [yam_i2rt]
    arms: 1
    version: 0.1.0
    tasks: [motion, grasp, point-tracking, pushing, drawer]
    provenance:
      authors: [Runtime.si]
    notes: A station-bound adapter boundary; source availability and interface tests do not imply sim/real verification.
gap:
  requires: {}
  tools:
    - workcell.describe: Describe the configured station and capability interface without issuing motion.
    - workcell.invoke: Invoke one named capability with a JSON object request and return its JSON result.
---

# Workcell tools

`workcell.describe()` and `workcell.invoke(capability, request_json)` expose the
same runtime API used by a direct Python agent. The station is selected by
`ROBOSKILLHUB_STATION`; the caller does not supply arbitrary import paths or
replace backend code in the capability request.

The capability names are `move_pose`, `grasp`, `track_points`, `push`,
`pull_drawer` and, from task contract `rsh-task/1`, `set_gripper`,
`move_waypoints` and `observe_objects`. Requests and results cross this graph-tool boundary as JSON strings
so the complete evidence stays intact. The canonical skills expose the terminal
`status` separately for routing: `succeeded`, `failed`, `unverified`, `rejected`.

Install RoboSkillHub in the same environment as the graph runtime. Importing this
module does not initialize a robot, simulator or model; the runtime binds the
configured adapter at invocation. Unsupported requests remain explicit failures
or rejections. A skill's presence in the catalog does not establish that the
current station has its dependencies or that it passed physical validation.

These are local adapter interfaces, not redistributed third-party checkpoints.
