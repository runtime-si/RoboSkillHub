"""The graph/agent boundary remains discoverable and preserves terminal evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from roboskillhub.catalog import load_catalog
from roboskillhub.checks import GapApi, check_bundle
from roboskillhub.search import infer_families, select

ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES = {
    "moving-to-pose": "move_pose",
    "grasping-with-feedback": "grasp",
    "tracking-points": "track_points",
    "pushing-on-plane": "push",
    "pulling-drawer": "pull_drawer",
}
STATUSES = ("succeeded", "failed", "unverified", "rejected")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("query", "family", "expected"),
    [
        ("位姿运动", "motion", "moving-to-pose"),
        ("move to pose", "motion", "moving-to-pose"),
        ("抓取", "grasp", "grasping-with-feedback"),
        ("grasp with feedback", "grasp", "grasping-with-feedback"),
        ("点跟踪", "point-tracking", "tracking-points"),
        ("point tracking", "point-tracking", "tracking-points"),
        ("推物", "pushing", "pushing-on-plane"),
        ("push on plane", "pushing", "pushing-on-plane"),
        ("拉抽屉", "drawer", "pulling-drawer"),
        ("pull drawer", "drawer", "pulling-drawer"),
    ],
)
def test_workcell_capabilities_are_searchable_in_chinese_and_english(query, family, expected):
    assert family in infer_families(query)
    result = select(load_catalog(ROOT), query, embodiment="yam_i2rt", arms=1)
    assert expected in result.names
    assert "workcell" in result.names
    assert next(p for p in result.picks if p.bundle.name == expected).score > 0


def test_workcell_bundles_declare_complete_routes_and_remain_unverified():
    catalog = load_catalog(ROOT)
    for name in (*CAPABILITIES, "workcell"):
        bundle = catalog.bundles[name]
        issues = check_bundle(bundle, catalog, GapApi())
        assert not [str(issue) for issue in issues if issue.level == "error"]
        assert bundle.maturity == "community"
        if name == "workcell":
            assert set(bundle.declared_tools) == {"workcell.invoke", "workcell.describe"}
        else:
            assert set(bundle.exits) == set(STATUSES)
            assert bundle.required_inputs == {"request_json": "string"}
            assert bundle.produces_outputs == {"result_json": "string", "status": "string"}


@pytest.mark.parametrize("bundle,capability", CAPABILITIES.items())
@pytest.mark.parametrize("status", STATUSES)
def test_canonical_scripts_route_runtime_result_without_inventing_success(
    bundle, capability, status
):
    script = load_module(ROOT / "skills" / bundle / "scripts/main.py", f"test_{bundle}_{status}")
    evidence = {"status": status, "details": {"measured_position_error_m": 0.004}}
    encoded = json.dumps(evidence)
    calls = []

    class Context:
        def tool(self, name, **kwargs):
            calls.append((name, kwargs))
            return encoded

    request = '{"target_pose": {}}'
    assert script.run(Context(), request_json=request) == {"result_json": encoded, "status": status}
    assert calls == [("workcell.invoke", {"capability": capability, "request_json": request})]


@pytest.mark.parametrize("bad_result", [None, {}, "{}", "null", '{"status":"done"}'])
def test_canonical_script_rejects_an_ambiguous_backend_result(bad_result):
    script = load_module(ROOT / "skills/moving-to-pose/scripts/main.py", "test_invalid_motion")

    class Context:
        def tool(self, name, **kwargs):
            return bad_result

    with pytest.raises((TypeError, ValueError)):
        script.run(Context(), request_json="{}")


@pytest.fixture
def workcell_tools(monkeypatch):
    # The direct wrapper can be tested without loading model or graph-engine stacks.
    gap_core = types.ModuleType("gap_core")
    decorators = types.ModuleType("gap_core.tools")
    decorators.tool = lambda **kwargs: lambda fn: fn
    monkeypatch.setitem(sys.modules, "gap_core", gap_core)
    monkeypatch.setitem(sys.modules, "gap_core.tools", decorators)
    runtime_api = types.ModuleType("roboskillhub.runtime.api")
    monkeypatch.setitem(sys.modules, "roboskillhub.runtime.api", runtime_api)
    wrapper = load_module(ROOT / "tools/workcell/tools.py", "test_workcell_tools")
    return wrapper, runtime_api


def test_workcell_tool_passes_json_request_and_preserves_runtime_evidence(workcell_tools):
    wrapper, runtime_api = workcell_tools
    calls = []

    def invoke(**kwargs):
        calls.append(kwargs)
        return {"status": "unverified", "reason": "no independent grasp observation"}

    runtime_api.invoke = invoke
    request = {"object_id": "测试物体"}
    result = json.loads(wrapper.invoke("grasp", json.dumps(request)))
    assert result["status"] == "unverified"
    assert calls == [{"capability": "grasp", "request": request}]


def test_workcell_description_does_not_invoke_motion(workcell_tools):
    wrapper, runtime_api = workcell_tools
    runtime_api.describe = lambda: {"station": "test", "capabilities": ["move_pose"]}
    assert json.loads(wrapper.describe()) == runtime_api.describe()


@pytest.mark.parametrize("payload", ["null", "[]", '"target"', "invalid json"])
def test_workcell_tool_rejects_nonobject_requests_before_dispatch(workcell_tools, payload):
    wrapper, runtime_api = workcell_tools
    calls = []
    runtime_api.invoke = lambda **kwargs: calls.append(kwargs)
    with pytest.raises(ValueError):
        wrapper.invoke("move_pose", payload)
    assert calls == []
