---
name: {{name}}
description: <What it does, in the third person, in one or two sentences.> Use
  when <the situation a coordinator should choose it for, and what it needs
  already in hand>.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: []
  hub:
    stage: execute
    embodiments: [yam_i2rt]
    version: 0.1.0
    tasks: []
    provenance:
      authors: [<your name or org>]
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.go_to_pose
  required_inputs:
    target_obb: OrientedBoundingBox
  produces_outputs:
    reached_pose: Se3Pose
  exit_conditions:
    done: <the state an observer can check that means it worked>
    failed: <the state that means it did not>
  canonical_scripts:
    - main: scripts/main.py
---

# {{name}}

<What a coordinator and a reviewer need: when to use it, when not to, the
inputs it expects and in which frame, what each exit means, and the constants
it relies on with how they were measured.>

## Required end states

- `done`: <...>
- `failed`: <...>
