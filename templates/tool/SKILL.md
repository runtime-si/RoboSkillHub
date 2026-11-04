---
name: {{name}}
description: <What the tools compute, in the third person.> Use when <the
  situation a skill should call them in>.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: []
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    provenance:
      authors: [<your name or org>]
gap:
  requires: {gpu: false}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "{{name}}"]
    protocol: stdio-msgpack
  tools:
    - {{name}}.example: <One line the coordinator reads.>
---

# {{name}}

The bundle runs in its own environment (`pyproject.toml` here) as a
subprocess gap talks to over msgpack. Load models on the first call, never at
import: gap imports `tools.py` to read the catalog.

If the tools need weights, upload them with
`motif weights push <dir> --repo <owner>/<name> --bundle tools/{{name}}` and
load them from the pinned `hf://` URI it writes under `metadata.hub.weights`.
