# How Graph-as-Policy runs a graph, and where OpenSkills plugs in

Read against graph-as-policy at the commit `sources.yaml` pins
(`09e45f8`), which is also the commit RoboSimStudio pins. Paths are in that
repository.

gap has two entry points: **generation** turns an instruction into a graph,
and **execution** runs a graph against a robot or a simulator. Both take the
same `skills=` argument, a list of registry roots. That argument is the whole
interface between gap and this hub.

## 1. Generation: `gap.agent.generate_sync`

```python
import gap

graph = gap.agent.generate_sync(
    "put the red block into the bowl",
    skills=["views/task"],  # what `openskills view` wrote; omitted -> resolved, see 3.
    provider=None,
    model=None,  # None: the pipeline config's provider and its default model
    out_dir="graphs/put_block",
)
graph.path  # the directory holding workflow.json
graph.workflow  # the parsed workflow.json
graph.code  # {"scripts/...py": source, "checkpoints/<subgraph>.py": source}
```

`gap/agent/__init__.py` resolves the registries, then
`gap/agent/multi_agent.py::generate_workflow` runs four stages:

1. **Coordinator.** Sees every skill's `name` and `description` (and the
   tools'), and emits the top-level topology: which subgraphs, which skill
   each one uses, and how each subgraph's exits route. This is the step a
   recipe guides and a registry view narrows. With the whole catalog in front
   of it, the coordinator chooses among hundreds of descriptions; with a view,
   among the dozen that can run on this robot.
2. **Subgraph agents**, one per subgraph, in order. Each reads its skill's
   full `SKILL.md`, its canonical scripts and references, and the schemas of
   its `allowed_tools`, then emits the subgraph's nodes. It must declare
   exactly the skill's `exit_conditions`, which is why a recipe has to route
   every exit.
3. **Checkpoint agent.** One call that writes postcondition checks per
   subgraph (`checkpoints/<subgraph>.py`), evaluated in simulation against
   ground truth.
4. **Assemble and load.** The workflow must parse under
   `gap/runtime/workflow.py`, or generation raises.

## 2. Execution: `gap.execute`

```python
result = gap.execute(
    "graphs/put_block",  # dir, workflow.json path, dict, or builder
    connector,  # robot.* tools, observations, optional ground truth
    skills=registries,  # the synced sources, not the view; see 4.
    inputs={"instruction": "..."},  # read in the graph as {"$ref": "in.instruction"}
    checkpoints="warn",  # "off" | "warn" | "raise"; needs connector.world_snapshot
)
result.success, result.exit_status, result.outputs, result.trace_path, result.error
```

`gap/runtime/execute.py::execute` does, in order:

1. **Materialize.** A dict or builder is written to a temp dir as
   `workflow.json`.
2. **Take the connector's surface**, duck-typed: `connector.tool_registry`
   (its `robot.*`, `sim.*` and other connector tools), `connector.get_observation`
   (which drives a 10 Hz observation stream), and `connector.world_snapshot`
   (optional ground truth; without it, checkpoints are skipped, which is the
   real-robot case).
3. **Load the registries** (see 3) and drain every in-process tool bundle's
   `@tool` registrations into the connector's registry.
4. **Boot policies.** For each `policies/` bundle the graph references,
   `gap/runtime/policy_manager.py` starts
   `uv run --project <bundle> -- python server.py ... --port N` in the
   bundle's own environment and waits until the port accepts connections.
   `PolicyExecutor` then holds one openpi websocket client per policy.
5. **Boot RPC tool bundles.** `gap/runtime/tool_bundle_boot.py` collects the
   bundles the graph needs: those named by `tool` nodes, plus every bundle in
   the `allowed_tools` of each subgraph's skill, since scripts call tools
   through `ctx.tool(...)` rather than as nodes. Each bundle whose `SKILL.md`
   says `serving.protocol: stdio-msgpack` gets a subprocess,
   `uv run --project <bundle> -- python -m gap_core.rpc.server --bundle <name>`
   (`gap-core/src/gap_core/rpc/client.py`). After a catalog handshake, each of
   its tools is registered with `register_rpc`, so the executor cannot tell an
   RPC tool from an in-process one.
6. **Run.** `WorkflowExecutor` schedules nodes in super-steps, running
   independent nodes in parallel threads (`max_node_workers`):
   - a `tool` node calls `registry.invoke(name, ctx, **inputs)`, in-process
     or over RPC;
   - a `script` node imports its file and calls `run(ctx, **inputs)`. When the
     subgraph's skill is a known bundle, the module is loaded as
     `gap_skills.<ns>.<bundle>.scripts.<stem>`, so the bundle's prompts and
     sibling modules resolve (`gap/runtime/nodes.py`). Scripts call tools with
     `ctx.tool("sam3.segment_text", ...)`;
   - a `router` node and the top level's `conditional_edges` branch on a
     field, usually a subgraph's `exit`;
   - an `end` node finishes with `success` or `failure` and runs its
     `recovery` tool calls.
7. **Tear down** every tool server and policy server, whatever happened.
   `execute` never raises for a failed workflow; the exception is in
   `result.error`.

## 3. Which registries

`skills=` given explicitly wins. Otherwise `gap/skills/registries.py` tries
`$GAP_SKILLS_PATH` (an `os.pathsep` list), then `[tool.gap] registries` in the
nearest `pyproject.toml`, then the user config, then an `open-robot-skills`
checkout beside the gap checkout. Earlier registries shadow later ones by
bundle name.

## 4. How RoboSimStudio calls it

`robosimstudio_policy_agent/graph_solution.py::SolutionGraph`:

- resolves the registry list once (`skills.pinned_registries`) and hands the
  same list to graph staging and to `gap.execute`, so the two cannot disagree;
- builds its connector (`connector.py`), a subclass of gap's `Connector` whose
  `robot.*` tools drive the Newton scene through its IK and servo. Robot
  geometry tools (`robot.describe_gripper`, ...) are always registered; `sim.*`
  ground-truth tools only when the run is privileged;
- runs `gap.execute(graph, connector, skills=..., inputs={"instruction", "perception"})`
  on a `gap-graph` thread while the main thread steps the simulation and
  publishes each step's observation.

## 5. Where the hub plugs in

| Step | Hub command | What gap sees |
| --- | --- | --- |
| before generation | `openskills view "<task>" --embodiment yam_i2rt --out views/task` | `skills=["views/task"]` (or `GAP_SKILLS_PATH`): only bundles this robot can run, and the tools they call. `RECIPES.md` is appended to the instruction. |
| simulation | `openskills sync` | `skills=[hub, .openskills/sources/open-robot-skills]`: the real directories |
| after it passes | `openskills lock <graph> --deploy` | nothing; `skills.lock` sits beside `workflow.json` |
| on the robot | `openskills fetch skills.lock --dest ...` | `GAP_SKILLS_PATH` it prints: the same bundles, verified by digest |

Why generation and execution use different registry paths: gap starts an RPC
bundle with `uv run --project <bundle dir>` and does not resolve symlinks. An
upstream bundle's `pyproject.toml` finds the engine at
`../../../graph-as-policy`, a path that only holds from where the bundle
really lives. `openskills sync` and `openskills fetch` both lay sources out side by side
under `sources/` so that relative path holds. A view is symlinks, so it serves
the coordinator, which only reads.
