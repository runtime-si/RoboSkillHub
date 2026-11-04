# Contributing to OpenSkills

Five things can be contributed: a **tool** (a model or algorithm skills call),
a **skill** (a reusable step of a task, with exits a policy branches on), a
**policy** (a trained checkpoint), a **recipe** (a task-level template), and a
**paper card**. Each starts from a template:

```bash
uv sync --extra gap
# Graph-as-Policy's validator, so `openskills check` runs it too:
uv pip install --no-deps "graph-as-policy @ git+https://github.com/graph-robots/graph-as-policy@09e45f8de27bc1a29629382ed7c2b9b4b384b3d6"
uv run --no-sync openskills sync
uv run --no-sync openskills new skill grasping-by-the-rim     # or tool, policy, recipe, paper
# edit skills/grasping-by-the-rim/
uv run --no-sync openskills check                             # 0 errors before you open the PR
uv run --no-sync pytest -q
```

## Which kind is it?

- **It wraps a model or computes something, and is called with arguments?**
  A tool. Perception models, planners, geometry, and yes/no judgments
  (`tools/predicates`) all qualify. Model-backed tools run in their own
  environment (`gap.serving.protocol: stdio-msgpack`), so their dependencies
  never touch anyone else's.
- **It is a step of a task, with outcomes a graph branches on?** A skill. Its
  `exit_conditions` are its contract: name observable end states (`grasped`,
  `not_found`), not internal errors.
- **It is a trained network that closes the loop itself?** A policy. The
  weights go to the Hugging Face Hub (`openskills weights push`), and the bundle
  pins them. It names the robot it was trained on and declares `io`: action
  space and rate, state, cameras. A checkpoint runs only on the layout it
  learned, so `embodiments: [any]` is never honest for one.
- **It is an order of skills that solves a kind of task?** A recipe.
- **It is a method from a paper that a bundle implements?** A paper card,
  listing the bundles under `derived`.

## What review looks for

1. **The description routes.** The coordinator chooses from descriptions
   alone. Say what the bundle does, then "Use when ..." with the situation and
   what must already be in hand.
2. **Exits are observable.** Each exit names a state someone watching could
   check, so a checkpoint can verify it.
3. **Constants carry their measurement.** A number in a script says what it
   was measured on (`tuned_on`) and how. Undocumented magic numbers are the
   main reason a skill fails on a second robot.
4. **No simulator ground truth unless necessary.** Anything calling `sim.*`
   is tier `full` and is never offered for a real robot. If a camera-only
   version is possible, write that one.
5. **Embodiment through `describe_*`.** Read the gripper, arm and workspace
   through the connector's `robot.describe_*` tools rather than hard-coding
   one robot; then `embodiments: [any]` is honest.
6. **Tests for anything pure.** Geometry, predicates and planners' pure parts
   get unit tests.

## Versions and compatibility

Bump `metadata.hub.version` with every change: MAJOR when an input, output
or exit changes (graphs that use it may break), MINOR for new optional
inputs or behaviour, PATCH for fixes. Graphs pin the content, not the
version, so an old graph keeps running the exact bundle it was locked with.

## Eval records and maturity

Maturity (`community`, `sim-verified`, `real-verified`) is set by records
in `evals/`. Only maintainers merge them; `CODEOWNERS` enforces it. To get a
bundle verified, open an issue with the suite, the robot, and the evidence (a
trace or video). See [evals/README.md](evals/README.md).

## Federated sources

Bundles from another registry are not copied here. To add or bump one:
change its pin in `sources.yaml`, run `openskills sync`, update its overlay in
`overlays/` until `openskills check` is clean, and open the PR with the upstream
changelog. A pin bump is its own PR.

## Commit messages

A lowercase `area:` prefix and one declarative sentence:
`skills: grasping-by-the-rim, for bowls and cups too wide to pinch`.
