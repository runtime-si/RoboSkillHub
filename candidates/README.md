# Reviewed upstream skill materials

This intake catalog contains concrete implementation material to adapt into
RoboSkillHub. It records source entrypoints, inputs and outputs, robot and sensor
assumptions, simulator privileges, code and weight licenses, and adaptation work.
It is separate from the executable registry in `tools/`, `skills/` and `policies/`.

The initial 2026-11-05 snapshot contains 30 capability records from 21 repositories:
25 records with MIT, Apache-2.0 or BSD-3-Clause code, plus 5 research-only sources.
These are capability records, not 30 independent ready-to-run robot packages.
Every source repository is pinned to a full commit. Weight links are discovery
references; they are not yet the immutable weight manifests used by `skills.lock`.

## Find a source

From this checkout after the normal package setup:

```bash
roboskillhub discover "抓取"
roboskillhub discover "推物"
roboskillhub discover "YAM" --priority P1 --json
roboskillhub discover --destination policy --json
roboskillhub discover --category deformable --json
roboskillhub discover --include-research-only --json
roboskillhub check-candidates
```

Without installing the package, the same commands work from the checkout with:

```bash
PYTHONPATH=src python -m roboskillhub.cli discover "抓取" --json
```

Discovery matches case-insensitive English or Chinese keywords separated by
spaces (OR matching). It searches names, capability descriptions and interfaces.
It is not a task planner and does not prove capability coverage. `search` and
`view` retain their existing executable-registry meaning. No candidate is copied
into a runtime view, downloaded, installed or executed by `discover`.

`--include-research-only` explicitly includes the sources whose code has
noncommercial restrictions. The default code-license grouping is not a blanket
commercial-use determination: weights, datasets, robot assets and dependencies
can have different or unresolved terms, recorded on each entry.

## Start with the current YAM workcell

P1 indicates an adaptation priority, not operational readiness:

- ENPIRE measured pose movement and RGB-D grasp proposals: compare with the
  existing YAM services; preserve measured feedback and coordinate conventions.
- ASPIRE typed API: reuse observation and execution contracts where useful,
  without importing a second task orchestrator.
- GraspGenX: a candidate grasp backend with a serving client. Configure and
  validate the actual YAM gripper; the upstream YAM support is not complete.
- TAPIR: track visible features across images; metric 3D estimates still require
  calibration, valid depth and uncertainty handling.
- ManiSkill pushing and IsaacLab drawer state machines: extract local control
  segments, replace oracle state and robot-specific control, add observable exits.
- LeRobot ACT: a local-data training path, not a selected pretrained YAM skill.

The wider catalog covers tool pulling, object reorientation, peg and charger
insertion, articulated doors, handover, Factory/FORGE assembly, motion planning,
generalist policies, towel manipulation, cloth primitives and DLO shape planning.
DLO Planning 2 is a dual-arm shape planner/MPC source, not a ready-made
cable-threading skill; its Unity/Obi assets have separate rights. Dual-arm and
force-feedback entries retain those requirements; they are not presumed to fit
the currently confirmed single-arm RGB-D workcell.

## Promote material into the executable registry

1. Inspect the pinned source and license records; select only the local ability
   needed. Preserve attribution for any code actually copied.
2. Reuse an existing bundle when the ability already exists. Geometry helpers
   remain tools; task ordering remains a recipe; learned weights use policies.
3. Define input/output types, frames, units, observation freshness, physical
   preconditions, bounded parameters, observable exits and effects on the world.
4. Replace simulator truth with available real observations where required;
   bind robot control to the actual YAM/Isaac connector and controller capabilities.
5. Package the implementation, pin model snapshots, and validate its dependencies.
6. Run independent skill tests in the declared domain, then file eval evidence
   for the exact bundle, workcell and version. Source review is not a sim/real pass.
7. Freeze the accepted skill implementations and allowed edits before graph-search
   comparisons. Record the promoted bundle in the candidate's intake history.

No paper, public checkpoint or successful upstream video on its own proves that
the resulting bundle is callable on this workcell. `check-candidates` checks
metadata integrity only. It does not contact the upstream sites or validate the
implementations, license interpretation, physics or hardware.

## Candidate format

`catalog.json` uses an independent `schema_version: 1` and a `candidates` array.
Each record includes:

- Identity: `id`, `name`, `category`, `capabilities`, `verified_date`.
- Source: `repo`, `source_url`, `source_revision`, `artifact_type`, `entrypoints`.
- Interface: `inputs`, `outputs`, `robots`, `sensors`, `sim_privilege`.
- Rights: `code_license` and a separate `weights` record, including uncertainty.
- Intake: `hub_destination`, `existing_overlap`, `adaptation_steps`, `priority`,
  `limitations`, `evidence_urls`, `intake_group`.
- State: `readiness: source-reviewed`, `invocable_in_hub: false`.

The validator deliberately rejects executable claims inside the intake catalog.
Actual runnable bundles belong to the existing registry and its validation path.
Priorities: P1 is the first adaptation batch; P2 expands task coverage; P3 needs
additional hardware, license resolution, substantial adaptation or has high overlap.

## Excluded and overlapping material

The existing open-robot-skills imports are already in this repository and were
not counted as new acquisitions. FlingBot was not included because an applicable
license was not confirmed in this review. ENPIRE's `place_with_orientation_v1`
was inspected but not promoted as a recommended primitive: in the reviewed source,
release follows the motion attempts even when lowering fails. Any future reuse
needs explicit failure handling and a separately validated contract.

## VLA checkpoints and datasets

`vla-checkpoints.json` and `vla-datasets.json` catalogue fine-tuned robot action
models (VLA checkpoints) and the datasets behind them, with the YAM station as
the reference robot. `vla-catalog.md` is generated from them; `vla-rejected.json`
lists survey rows a merge refused and why. Like the other files here, entries are
not bundles: a checkpoint runs only through a policy bundle's server with a
checked-in profile (`policies/molmoact2-yam/profiles`, `policies/gr00t/profiles`).

```bash
roboskillhub sources "bimanual yam pick" --kind vla      # ranked; any term counts
roboskillhub vla search --embodiment yam_i2rt --compatible as_is
roboskillhub vla datasets yam --json
roboskillhub vla profiles        # as_is + servable by our servers, and which lack a profile
roboskillhub vla merge survey/*.json --write              # dedupe by id, keep the most complete
roboskillhub vla page --write    # regenerate vla-catalog.md (`--check` fails when stale)
roboskillhub check-sources       # validates both collections too
```

Each checkpoint records `id` (Hugging Face repo), `revision` (a full 40-character
commit; branch names are refused), `family`, `base_model`, `format`, `embodiment`,
`arms`, `action{space,dim,horizon,hz}`, `state{dim,order}`, `cameras`,
`training{datasets,episodes,tasks}`, `license{model,inherited,verified}`
(`verified` is an explicit true/false), `gated`, `size_gb`, `servable_by`,
`yam_compatible`, `task_relevance`, `notes`, `source_urls`, `checked_at`, and
optionally `evaluations` (setting, task, episodes, successes, date, evidence).

`yam_compatible`: `as_is` = the checkpoint's inputs and outputs are the station's
(14 absolute joints for two YAM arms, top + two wrist cameras), no retraining;
`adapter` = usable after a mapping of joints, cameras or action space;
`finetune_base` = a base to fine-tune on station data; `no`. `as_is` says nothing
about success: every YAM checkpoint we tried in the station twin scored 0/10 so far
(`evaluations`). `servable_by` names the runtimes; `policies/molmoact2-yam`,
`policies/gr00t:n1.7` and `policies/gr00t:pi05` are this hub's servers, and every
`as_is` checkpoint one of them can run must have a profile there.

## Fine-tuned models outside robot action

`finetuned-models.json` catalogues fine-tuned models with public weights that are
not robot action models: instruction-tuned language models, text classifiers,
vision and multimodal models, speech and audio models, embedders, rerankers and a
reward model. None is a bundle: an entry is a pinned, described starting point for
a tool bundle that would serve the model. Robot action checkpoints stay in
`vla-checkpoints.json`.

```bash
roboskillhub sources "语音识别" --kind models       # ranked; any term counts, English or Chinese
roboskillhub sources "rerank retrieval" --kind models --json
roboskillhub check-sources       # validates this collection too
```

Each entry records `id` (Hugging Face repo), `revision` (a full 40-character
commit), `name`, `domain`, `category`, `task`, `pipeline_tag` (the Hub's task tag,
or null), `base_model`, `method`, `form`, `weight_type`, `params`, `weight_files`
(the `.safetensors` / `.bin` / `.ckpt` files at that revision), `size_gb` (every
file in the repo at that revision, so what a full snapshot downloads), `license`,
`card_license` (the card's licence tag, null when it has none), `gated`, `notes`,
`evidence`, `source_urls` and `checked_at`. The descriptive fields are in Chinese,
as surveyed.

`form`: `full` = the repo holds complete weights; `adapter` = a LoRA or a
conditioning network that does nothing without its base model (`base_model` and
`notes` say which, and how to combine them).

`license` repeats what the card, or the licence it links, states, plus any
base-model terms the survey noted. It is not a clearance: the collection includes
non-commercial weights (`cc-by-nc-sa-4.0`, the NVIDIA SegFormer licence), custom
terms (OpenRAIL variants, the Skywork community licence), models that inherit
Llama terms, and one repo that states no licence at all.
