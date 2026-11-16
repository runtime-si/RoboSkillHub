"""The example subgraphs of the bin and container skills agree with their SKILL.md contracts."""

from __future__ import annotations

import inspect
import json

import pytest

BUNDLES = (
    "finding-boxes-of-known-size-from-overhead-depth",
    "locating-a-container-from-the-wrist-camera",
    "anchoring-a-bin-to-its-walls-from-the-wrist-camera",
    "proposing-open-hand-grasps-in-a-dense-pile",
    "carrying-over-an-obstacle-with-a-leaning-hand",
    "cross-checking-a-placed-count-from-overhead",
)


@pytest.mark.parametrize("name", BUNDLES)
def test_example_matches_the_contract(skills_registry, name):
    info = skills_registry.get(name)
    example = json.loads((info.bundle_dir / "examples" / "canonical_subgraph.json").read_text())
    assert example["skill"] == name
    exits = set(info.meta.exit_conditions)
    assert set(example["exit"]["success_values"]) <= exits and example["on_error"] in exits
    (script,) = info.canonical_scripts.values()
    (node,) = example["nodes"].values()
    assert (info.bundle_dir / node["script"]).is_file()
    params = inspect.signature(script.module.run).parameters
    assert set(node["inputs"]) <= set(params)
    required = {p for p, v in params.items() if p != "ctx" and v.default is inspect.Parameter.empty}
    assert required == set(info.meta.required_inputs) <= set(node["inputs"])
    outputs = set(script.schema.outputs)
    assert {ref["$ref"].split(".", 1)[1] for ref in example["outputs"].values()} <= outputs
    assert set(info.meta.produces_outputs) <= outputs
    # every route the script can return is a declared exit
    source = inspect.getsource(script.module)
    for exit_name in exits:
        assert f'"{exit_name}"' in source
