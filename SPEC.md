# RoboSkillHub formats, version 1

Every file format the library reads or writes. `roboskillhub check` enforces all
of it.

## Bundles

A bundle is a directory under `tools/`, `skills/` or `policies/` holding a
`SKILL.md` whose frontmatter follows the Agent Skills spec. Runtime details
(exits, tools it may call, how it is served) sit under `gap:`, the block
[Graph-as-Policy](https://github.com/graph-robots/graph-as-policy) reads, so
every bundle also loads there unchanged; when gap is installed,
`roboskillhub check` runs its parser and validator as well.

RoboSkillHub adds one mapping, `metadata.hub`. gap reads only
`metadata.category` and `metadata.tags`, so the addition is invisible to it.

```yaml
metadata:
  category: grasping
  tags: [grasping, cpu]
  hub:
    stage: execute               # required for skills and policies
    embodiments: [yam_i2rt]      # required: rss.robot() names, or [any]
    version: 1.2.0               # required in this repository: semver
    tuned_on: [yam_i2rt]         # the robot its constants were measured on
    arms: 1                      # arms it needs; default 1
    io:                          # required for policies: what the checkpoint was trained with
      action: {space: osc_pose_delta, dim: 7, hz: 20}
      state: {dim: 8, order: [eef_pos, eef_axis_angle, gripper_qpos]}
      cameras: [agentview, wrist]
    tasks: [pick-place]          # task families, for search
    weights:                     # tool and policy bundles with a checkpoint
      uri: hf://<owner>/<repo>@<40-char commit>
      sha256: <64-char digest>
      license: <the checkpoint's license>
    provenance:
      authors: [<names>]
      paper: <papers/ id or arXiv id>
      code: <url of the code it came from>
      upstream_license: <license of that code>
      data: <where training demonstrations came from>
    notes: <one line a user must know>
```

| Key | Rule |
| --- | --- |
| `stage` | one of `perceive`, `plan`, `propose`, `select`, `execute`, `verify`, `recover`, `judge` |
| `embodiments` | the robots it is declared to work on. `any` means it reads the robot only through `describe_*` tools. A declaration, not a result. A policy names the robots its checkpoint was trained on; `any` is an error. |
| `io` | policies only, and required for them: `action` (`space`, `dim`, `hz`), `cameras` (the names the checkpoint reads), optionally `state` (`dim`, `order`). A checkpoint runs only on the layout it learned; a skill reads the robot through `describe_*` instead. |
| `version` | semver; bump MAJOR when an input, output or exit changes |
| `weights.uri` | `hf://` URIs must pin a full commit. Branches and tags move. |
| `weights.sha256` | digest of the snapshot's file manifest (below) |
| `tier` | never needed. If written, it must equal the derived tier. |

### Derived, never declared

- **exits**: the keys of `gap.exit_conditions`.
- **tier**: `full` when `gap.allowed_tools` or `gap.requires.connector` names
  any `sim.*` tool, otherwise `camera_only`. A `full` bundle cannot leave
  simulation.
- **tool bundles called**: every `<bundle>.<func>` in `gap.allowed_tools`
  whose prefix is a tool or policy bundle in the catalog. `robot.*`, `sim.*`,
  `motion.*` and `cable.*` belong to the connector.
- **maturity**: from eval records (below).
- **content_sha256**: one digest over every file in the bundle (relative path
  and content, skipping `.venv`, `__pycache__`, `.git`).

### Rules beyond gap's

Errors for bundles in this repository. For federated bundles the same rule is
a warning, unless gap itself rejects the bundle or the broken part is the
overlay.

- The description contains a "Use when ..." sentence, and the bundle declares
  a `license`.
- A skill or policy declares `exit_conditions`, unless it is `streaming: true`.
- Canonical scripts exist and define `run(ctx, ...)`.
- A tool bundle has `tools.py` and `gap.requires`, namespaces its tools
  `<bundle>.<func>`, and imports no model library (`torch`, `jax`,
  `transformers`, ...) at module level.
- A policy bundle declares `gap.serving` with a `command` list and a
  `protocol`.
- Every `allowed_tools` entry is a connector tool or a tool of a catalog
  bundle.
- Input and output types resolve in gap's type registry.

## Overlays

`overlays/<source>.yaml` maps each bundle of a federated source to its
`metadata.hub` keys. Every bundle the source ships needs an entry, and every
entry needs a bundle, so bumping a pin forces the overlay to keep up.

## Recipes

`recipes/<name>/RECIPE.md`: a task-level template at the coordinator's level of
abstraction, meaning which skill each subgraph uses and where each exit leads.
The subgraphs' insides are left to the skills.

```yaml
---
name: pick-place-camera-only
description: ... Use when ...
license: Apache-2.0
metadata:
  hub: {version: 0.1.0, embodiments: [any], tasks: [pick-place]}
recipe:
  inputs: {object_description: string}       # graph inputs
  start: perceive_target
  nodes:
    perceive_target: {skill: perceiving-objects, bind: {name: target}}
    grasp: {skill: grasping-short-axis, note: optional text for the coordinator}
    done: {end: success}
    abort: {end: failure, recovery: [robot.open_gripper]}
  routes:
    perceive_target: {found: grasp, not_found: abort}
    grasp: {grasped: done, failed: abort}
---
```

- `bind.name` fills the `<name>` in a skill's output names (`<name>_obb`
  becomes `target_obb`).
- Every exit of every skill node is routed, and only its exits. gap requires
  a subgraph to declare exactly its skill's exits.
- Every node is reachable from `start`, and a `success` end is reachable.
- An input no node produces and the recipe does not take is a warning.
- The recipe's tier is derived from its skills.

`roboskillhub view` renders matched recipes into `RECIPES.md`, text a coordinator can
follow when it is appended to the instruction.

## Papers

`papers/<id>/PAPER.md` has frontmatter keys `id` (equal to the directory),
`title`, `year`, `url` or `arxiv`, and optionally `authors`, `venue`, `code`,
`license`, `tasks`, and `derived`: the `tools/`, `skills/`, `policies/` or
`recipes/` entries that implement the method, all of which must exist. The body
covers the method, what the hub uses, and when it is the wrong tool.

## Eval records

`evals/<bundle>/<anything>.json`, merged by maintainers only:

```json
{"bundle": "grasping-short-axis", "version": "0.1.0", "embodiment": "yam_i2rt",
 "setting": "sim", "tier": "camera_only", "suite": "rss/pick_place_camera",
 "episodes": 50, "successes": 41, "passed": true, "date": "2026-11-04",
 "evidence": "https://..."}
```

A bundle with a passed `real` record is `real-verified`; with a passed `sim`
record, `sim-verified`; otherwise `community`.

## Weights

`hf://[datasets/]<owner>/<repo>@<commit>`. The digest is SHA-256 over the
lines `"<relative path>\0<sha256 of file>\n"`, sorted by path, for every file
in the snapshot except `.gitattributes` and anything under `.cache/` or
`.git/`. `roboskillhub weights push` uploads a folder as one commit that mirrors it
exactly, so the digest computed before upload is the digest of what anyone
pulls at that commit.

## `skills.lock`

Written by `roboskillhub lock <graph>` next to `workflow.json`:

```json
{
  "lock_version": 1,
  "graph": {"sha256": "...", "skills": ["perceiving-objects"], "generated_skills": ["generic"]},
  "engine": {"name": "graph-as-policy", "repo": "...", "commit": "..."},
  "hub": {"repo": "...", "commit": "...", "dirty": false},
  "sources": {"open-robot-skills": {"repo": "...", "commit": "..."}},
  "bundles": {
    "perceiving-objects": {"kind": "skill", "source": "open-robot-skills", "path": "skills/perceiving-objects",
                           "content_sha256": "...", "tier": "camera_only"},
    "sam3": {"kind": "tool", "source": "open-robot-skills", "path": "tools/sam3",
             "content_sha256": "...", "tier": "camera_only", "serving": "stdio-msgpack"}
  },
  "connector_tools": ["robot.get_observation"],
  "privileged_tools": [],
  "deployable": true
}
```

- `graph.sha256` covers `workflow.json`, `scripts/` and `checkpoints/`.
- The bundles are the subgraph skills, the tool nodes' bundles, and every
  bundle those skills' `allowed_tools` reach.
- `connector_tools` is what the robot's connector must provide.
- `deployable` means no `sim.*` tool anywhere and every bundle `camera_only`.
  `--deploy` fails otherwise.
- `roboskillhub fetch` checks out each source at its commit, verifies every bundle's
  `content_sha256`, pulls and verifies the weights, and prints the
  `GAP_SKILLS_PATH` to run with.

## `index.json`

`roboskillhub index` writes the whole catalog as one deterministic JSON file: per
bundle every declared and derived field above, then recipes, papers, and
which bundles shadow which. A site, a search API, or a pipeline that does not
want to parse YAML reads this.
