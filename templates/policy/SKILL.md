---
name: {{name}}
description: <The checkpoint, what it was trained to do, on which robot, from
  which cameras, in which action space.> Use when <the task segment to hand to
  it, and the state the graph must reach first -- e.g. the gripper hovering
  over the target>. Not for <what it was not trained on>.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: policy
  tags: [policy, learned]
  hub:
    stage: execute
    embodiments: [yam_i2rt]
    io:
      action: {space: TBD, dim: TBD, hz: TBD}
      state: {dim: TBD}
      cameras: [TBD]
    version: 0.1.0
    tasks: []
    weights:
      uri: hf://OWNER/REPO@REVISION
      sha256: TBD
      license: <the checkpoint's license>
    provenance:
      authors: [<your name or org>]
      data: <where the demonstrations came from, and how many>
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "server.py", "--port", "{port}"]
    protocol: websocket
    requires_gpu: true
    weights_uri: hf://OWNER/REPO@REVISION
  allowed_tools:
    - robot.get_observation
    - robot.execute_trajectory
  exit_conditions:
    completed: The policy declared the segment done.
    max_windows: The window cap was reached without a completion signal.
    failed: The policy loop errored. This is the subgraph's on_error exit.
  tools:
    - {{name}}.run: Run the checkpoint in closed loop against its own server.
---

# {{name}}

## Publishing the checkpoint

```bash
openskills weights push ./checkpoint --repo <owner>/{{name}} --bundle policies/{{name}}
```

uploads the folder to the Hugging Face Hub as one commit and writes that
commit (`hf://<owner>/{{name}}@<sha>`) and the snapshot's digest into this
file. `server.py` downloads exactly that snapshot and refuses any other.

## What the server speaks

openpi's websocket protocol, which gap's `PolicyExecutor` connects with: on
connect the server sends its metadata, then answers each observation dict with
`{"actions": array[H, D]}`. Fill in `load()` and `infer()` in `server.py`.

## Observation and action conventions

`metadata.hub.io` is the part a check reads; the details go here:
<cameras and their resolution, proprioception, action space and units, control
rate, chunk length. gap's shared policy loop encodes LIBERO observations
today; a checkpoint trained on another layout needs its own encoding in
`tools.py`.>
