"""Integration of K, the two minimal graphs and the three baselines (Agent 5).

Domino episodes run Agent 2's ``domino.place_upright`` through the real runtime
(task lease, primitives, gate) on the seeded kinematic fake
(``roboskillhub.tasks.fake_world``): they check graph semantics and honesty
rules, not physical behaviour.
"""

from __future__ import annotations

import time

import pytest

from roboskillhub.runtime.engine import Runtime
from roboskillhub.runtime.graph import GraphError, load_graph, run_graph
from roboskillhub.runtime.task import offline_context, primitive_skill_context, run_task_skill
from roboskillhub.tasks import graphs, load_registry
from roboskillhub.tasks.baselines import StationEnv, compare, summarize
from roboskillhub.tasks.fake_world import domino_world

SLOTS = 3


@pytest.fixture(scope="module")
def K():
    registry, _ = load_registry()
    if "domino.place_upright" not in registry:
        pytest.skip("domino module not in this checkout")
    return registry


def world(seed, **kw):
    return domino_world(seed, slots=SLOTS, **kw)


def test_registry_lists_core_and_reports_missing_modules():
    registry, unavailable = load_registry()
    assert {s for s in registry if s.startswith("core.")} == {
        "core.select_next_target",
        "core.observe",
        "core.grasp_candidate_from_pose",
        "core.select_candidate",
        "core.confirm_hold",
        "core.bind_target",
        "core.pick",
        "core.transport_held",
        "core.release_and_retreat",
    }
    for name, reason in unavailable.items():
        assert name.startswith("roboskillhub.tasks.") and reason


def test_graphs_bind_versions_in_K_and_fail_without_their_skills(K):
    for g in [*graphs.domino_candidates(SLOTS), *graphs.lego_candidates()]:
        load_graph(g, K)
    without = {k: v for k, v in K.items() if k != "domino.place_upright"}
    with pytest.raises(GraphError, match="not in K"):
        load_graph(graphs.domino_line(), without)


@pytest.mark.parametrize("recovery", [False, True])
def test_graph_success_is_never_claimed_without_independent_task_success(K, recovery):
    env = StationEnv(world, K)
    graph = graphs.domino_line(recovery=recovery, attempts=2 if recovery else 1, slots=SLOTS)
    episodes = [env(graph, seed) for seed in range(4)]
    for e in episodes:
        if e["status"] == "succeeded":
            assert e["task_success"] is True
        assert e["privileged_inputs"] is True
        assert e["counters"]["steps"] <= load_graph(graph, K)["budgets"]["max_steps"]
        assert not [s for s in e["steps"] if s["failure_type"] == "backend_error"]
    assert summarize(episodes)["graph_successes"] > 0


def test_recovery_graph_beats_the_script_on_paired_seeds(K):
    env = StationEnv(lambda s: world(s, p_topple=0.3), K)
    seeds = range(6)
    script = summarize(
        env(graphs.domino_line(recovery=False, attempts=1, slots=SLOTS), s) for s in seeds
    )
    replan = summarize(env(graphs.domino_line(recovery=True, slots=SLOTS), s) for s in seeds)
    assert replan["graph_successes"] > script["graph_successes"]


def test_missing_orientation_feedback_aborts_with_capability_missing(K):
    def no_pose(seed):
        built = world(seed)
        backend = built["backend"]
        caps = [c for c in backend.describe()["capabilities"] if c != "object_pose_feedback"]
        backend.describe = lambda: {"frame_id": "world", "capabilities": caps}
        return built

    episode = StationEnv(no_pose, K)(graphs.domino_line(slots=SLOTS), 0)
    assert episode["status"] == "failed"
    assert episode["steps"][-1]["failure_type"] == "capability_missing"


def test_stale_observations_end_unverified_never_succeeded(K):
    def stale(seed):
        built = world(seed)
        backend = built["backend"]
        original = backend.get_state

        def get_state(*, timeout_s):
            state = original(timeout_s=timeout_s)
            for record in state["objects"].values():
                record["timestamp_s"] = time.time() - 30
            return state

        backend.get_state = get_state
        return built

    episode = StationEnv(stale, K)(graphs.domino_line(slots=SLOTS), 1)
    assert (episode["status"], episode["terminal"]) == ("unverified", "unsure")
    assert episode["steps"][-1]["failure_type"] == "stale_observation"


def test_lego_graph_stops_at_the_press_without_contact_capability(K):
    from roboskillhub.tasks.lego import SKILLS  # noqa: F401  (module present)

    g = load_graph(graphs.lego_press(bricks=2), K)
    seat = g["nodes"]["seat"]
    assert K[seat["skill"]].requires >= {"contact_force_feedback", "compliant_motion"}
    built = world(0)
    caps = set(built["backend"].describe()["capabilities"])
    assert not K[seat["skill"]].requires <= caps


def test_select_respects_dependencies_and_feeder():
    select = load_registry()[0]["core.select_next_target"]
    targets = [
        {"target_id": "b1", "order": 0, "depends_on": ["b0"]},
        {"target_id": "b0", "order": 1},
    ]
    out = run_task_skill(
        select, {"targets": targets, "done": [], "key": "target_id"}, offline_context()
    )
    assert out["outputs"]["target_id"] == "b0"
    out = run_task_skill(
        select,
        {"targets": targets, "done": [], "key": "target_id", "pieces": ["p0"], "used": ["p0"]},
        offline_context(),
    )
    assert (out["status"], out["failure_type"]) == ("failed", "precondition_unmet")
    out = run_task_skill(
        select, {"targets": targets, "done": ["b0", "b1"], "key": "target_id"}, offline_context()
    )
    assert out["status"] == "succeeded" and out["outputs"]["complete"] is True


def test_grasp_candidate_needs_an_observed_orientation():
    tool = load_registry()[0]["core.grasp_candidate_from_pose"]
    obs = {
        "object_id": "p0",
        "frame_id": "world",
        "position_m": [0.3, 0, 0.02],
        "quaternion_xyzw": None,
        "timestamp_s": time.time(),
        "source": "rgbd_centroid",
        "privileged": False,
        "visibility": "visible",
    }
    request = {
        "observation": obs,
        "T_object_TCP": {
            "frame_id": "object",
            "position_m": [0, 0, 0.01],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        "pregrasp_height_m": 0.05,
        "lift_height_m": 0.05,
        "open_width_m": 0.04,
        "close_width_m": 0.004,
    }
    out = run_task_skill(tool, request, offline_context())
    assert (out["status"], out["failure_type"]) == ("rejected", "evidence_missing")


def test_compare_runs_three_baselines_on_disjoint_paired_seeds(K, monkeypatch):
    few = graphs.domino_candidates(SLOTS)[:2]
    monkeypatch.setattr(graphs, "domino_candidates", lambda slots: few)
    env = StationEnv(world, K)
    report = compare(
        "domino",
        env,
        train_seeds=range(100, 102),
        test_seeds=range(2),
        environment="fake",
        slots=SLOTS,
    )
    assert set(report["baselines"]) == {"fixed_script", "online_replan", "simulation_search"}
    assert report["baselines"]["simulation_search"]["search_rollouts"] == 2 * 2
    for row in report["baselines"].values():
        assert row["episodes"] == 2
    with pytest.raises(ValueError, match="disjoint"):
        compare("domino", env, train_seeds=[1], test_seeds=[1], environment="fake")


def test_task_lease_spans_the_episode_and_skills_reuse_it(K):
    built = world(3)
    backend = built["backend"]
    calls = []
    backend.begin_invocation = lambda *, timeout_s: calls.append("begin") or {"ok": True}
    backend.end_invocation = lambda: calls.append("end") or {"ok": True}
    rt = Runtime(backend, built["limits"])
    with rt.task_lease("ep", 300) as lease:
        episode = run_graph(
            graphs.domino_line(slots=SLOTS),
            K,
            built["inputs"],
            lambda node, rid, t: primitive_skill_context(
                rt, lease, request_id=rid, privileged_allowed=True
            ),
        )
    assert calls == ["begin", "end"]
    assert episode["counters"]["primitive_calls"] > 5


def test_graphs_take_grasps_from_perception_and_refuse_unlabelled_truth(K):
    g = graphs.domino_line(slots=SLOTS)
    assert g["nodes"]["candidate"]["skill"] == "perception.propose_grasps"
    assert g["nodes"]["classify"]["skill"] == "perception.classify_object_state"
    load_graph(graphs.domino_line(slots=SLOTS, grasps="configured"), K)

    def unlabelled(seed):
        built = world(seed)
        built["privileged_allowed"] = False
        return built

    episode = StationEnv(unlabelled, K)(g, 0)
    assert episode["status"] == "failed"
    assert episode["steps"][-1]["failure_type"] == "perception.privileged_input"


def _wrist_limited(seed, *, query: bool):
    """Fake domino station that refuses (before moving) any grasp yawed like candidate 0."""
    built = world(seed)
    station = built["backend"]
    base_move, base_describe = station.move_pose, station.describe

    def refused(pose):
        x, y, z, w = pose["quaternion_xyzw"]
        return pose["position_m"][2] < 0.09 and x * y < 0  # yaw of the first candidate's pair

    def move_pose(pose, **kw):
        if refused(pose):
            return {"ok": False, "error": "wrist limit", "failure_type": "unreachable",
                    "refused_before_motion": True}  # fmt: skip
        return base_move(pose, **kw)

    def describe():
        d = base_describe()
        if query:
            d = {**d, "capabilities": [*d["capabilities"], "reachability_query"]}
        return d

    station.move_pose, station.describe = move_pose, describe
    station.reachable = lambda poses, chain, timeout_s: [
        {"ok": not refused(p), "reason": "wrist limit" if refused(p) else None} for p in poses
    ]
    return built


def test_refused_candidate_falls_back_or_is_filtered(K):
    replan = graphs.domino_line(slots=SLOTS, recovery=True, attempts=2)
    scripted = graphs.domino_line(slots=SLOTS, recovery=False, attempts=1)
    no_query = StationEnv(lambda s: _wrist_limited(s, query=False), K)
    episode = no_query(replan, 1)
    failures = [(s["node"], s["failure_type"]) for s in episode["steps"]]
    assert ("pick", "unreachable") in failures  # failed: pick opened the jaws first
    assert episode["status"] == "succeeded", failures
    assert no_query(scripted, 1)["failure_type"] == "unreachable"
    with_query = StationEnv(lambda s: _wrist_limited(s, query=True), K)
    episode = with_query(scripted, 1)
    assert all(s["failure_type"] != "unreachable" for s in episode["steps"])
    candidate = next(s for s in episode["steps"] if s["node"] == "candidate")
    assert candidate["result"]["outputs"]["unreachable"]


# --- LEGO graph integration (contract 1.4) -----------------------------------


def _ctx(state, observation, **kw):
    from roboskillhub.runtime.task import SkillContext

    def primitive(capability, request, rid):
        assert capability == "observe_objects"
        return {
            "status": "succeeded",
            "capability": capability,
            "evidence": {"observations": [observation]},
        }

    return SkillContext(
        task_id="t",
        request_id="r",
        capabilities=frozenset({"pose_feedback", "gripper_feedback", "object_feedback"}),
        primitive=primitive,
        read_state=lambda: state,
        limits={"held_object_tolerance_m": 0.01},
        **kw,
    )


def _held(stamp):
    return {
        "object_id": "spare_1",
        "gripper_id": "g",
        "T_TCP_object": {
            "frame_id": "tcp",
            "position_m": [0, 0, 0.013],
            "quaternion_xyzw": [0, 0, 0, 1],
        },  # fmt: skip
        "estimate": "privileged_sim",
        "timestamp_s": stamp,
        "valid": True,
        "uncertainty": {"position_m": 0.001, "rotation_rad": 0.02},
    }


def test_confirm_hold_restamps_or_reports_the_loss(K):
    from roboskillhub.runtime.task import run_task_skill

    now = time.time()
    tcp = {"frame_id": "world", "position_m": [0.3, 0.0, 0.1], "quaternion_xyzw": [0, 0, 0, 1]}
    seen = {"object_id": "spare_1", "frame_id": "world", "position_m": [0.3, 0.0, 0.113],
            "timestamp_s": now, "source": "sim_truth", "privileged": True,
            "quaternion_xyzw": None, "visibility": "visible", "confidence": 1.0,
            "uncertainty": None}  # fmt: skip
    gripper = {"holding": True, "width_m": 0.016, "timestamp_s": now, "source": "pads"}
    state = {"tcp_pose": tcp, "timestamp_s": now, "gripper": gripper}
    ok = run_task_skill(
        K["core.confirm_hold"], {"held": _held(now - 3)}, _ctx(state, seen, privileged_allowed=True)
    )
    assert ok["status"] == "succeeded" and ok["outputs"]["held"]["timestamp_s"] == now
    lost = {**state, "gripper": {**gripper, "holding": False}}
    out = run_task_skill(K["core.confirm_hold"], {"held": _held(now - 3)}, _ctx(lost, seen))
    assert (out["status"], out["failure_type"]) == ("failed", "grasp_lost")
    far = {**seen, "position_m": [0.3, 0.03, 0.113]}
    out = run_task_skill(
        K["core.confirm_hold"], {"held": _held(now - 3)}, _ctx(state, far, privileged_allowed=True)
    )
    assert (out["status"], out["failure_type"]) == ("failed", "held_object_slipped")


def test_bind_target_tracks_the_carried_piece_and_the_build(K):
    from roboskillhub.runtime.task import SkillContext, run_task_skill

    prebuilt = [{"target_id": "base_1"}]
    target = {"target_id": "b2", "mating": {"target_id": "b2", "depends_on": ["b1", "base_1"]}}
    placed = [{"target_id": "b1", "object_id": "spare_0", "mating": {"target_id": "spare_0"}}]
    out = run_task_skill(
        K["core.bind_target"],
        {"target": target, "piece": "spare_3", "placed": placed, "prebuilt": prebuilt},
        SkillContext(task_id="t", request_id="r"),
    )["outputs"]
    assert out["mating"]["target_id"] == "spare_3"
    assert out["mating"]["depends_on"] == ["spare_0", "base_1"]
    assert [b["target_id"] for b in out["build"]] == ["base_1", "spare_0"]
    assert out["placed_after"][-1] == {"target_id": "b2", "object_id": "spare_3",
                                       "mating": out["mating"]}  # fmt: skip


def test_lego_graph_wires_build_fresh_hold_lattice_and_piece_ids(K):
    for recovery in (False, True):
        g = graphs.lego_press(recovery=recovery, bricks=3)
        load_graph(g, K)
        n = g["nodes"]
        assert n["align"]["params"]["held"] == "$last.confirm.outputs.held"
        assert n["align"]["params"]["build"] == n["verify"]["params"]["build"]
        assert n["verify"]["params"]["target"] == "$last.bind.outputs.mating"
        assert n["classify"]["params"]["lattice"] == "$input.lattice"
        assert n["classify"]["params"]["expected_state"] == "loose"


def test_classify_decides_a_loose_upright_bin_brick(K):
    from roboskillhub.runtime.task import SkillContext, run_task_skill

    lattice = {
        "T_W_baseplate": {
            "frame_id": "world",
            "position_m": [0.4, 0.0, 0.0032],
            "quaternion_xyzw": [0, 0, 0, 1],
        },  # fmt: skip
        "plate_studs": [32, 32],
    }
    obs = {"object_id": "spare_1", "class": "lego_brick", "frame_id": "world",
           "position_m": [0.1, 0.25, 0.0], "quaternion_xyzw": [0, 0, 0, 1],
           "timestamp_s": time.time(), "source": "sim_truth", "privileged": True,
           "confidence": 1.0, "uncertainty": {"position_m": 0.001, "rotation_rad": 0.02},
           "visibility": "visible"}  # fmt: skip
    ctx = SkillContext(task_id="t", request_id="r", privileged_allowed=True)
    req = {"observation": obs, "table_z": 0.0, "lattice": lattice, "expected_state": "loose"}
    out = run_task_skill(K["perception.classify_object_state"], req, ctx)
    assert out["status"] == "succeeded", out
    tilted = {**obs, "quaternion_xyzw": [0.7071068, 0, 0, 0.7071068]}
    out = run_task_skill(K["perception.classify_object_state"], {**req, "observation": tilted}, ctx)
    assert (out["status"], out["failure_type"]) == ("failed", "perception.unexpected_state")


def test_propose_grasps_uses_the_station_floor_for_a_flat_brick(K):
    from roboskillhub.runtime.task import SkillContext, run_task_skill

    obs = {"object_id": "b1", "class": "lego_brick", "frame_id": "world",
           "position_m": [0.2, -0.2, 0.0], "quaternion_xyzw": [0, 0, 0.7071068, 0.7071068],
           "timestamp_s": time.time(), "source": "sim_truth", "privileged": True,
           "confidence": 1.0, "uncertainty": {"position_m": 0.0001, "rotation_rad": 0.001},
           "visibility": "visible", "geometry": {"type": "box", "size_m": [0.016, 0.032, 0.0096]}}  # fmt: skip
    req = {"observation": obs, "table_z": 0.0, "max_candidates": 4}
    offline = SkillContext(task_id="t", request_id="r", privileged_allowed=True)
    station = SkillContext(
        task_id="t",
        request_id="r",
        privileged_allowed=True,
        primitive=lambda *a: {},
        limits={"workspace_min_m": [0.0, -0.35, 0.0]},
    )
    yam_floor = run_task_skill(K["perception.propose_grasps"], req, offline)
    assert yam_floor["failure_type"] == "perception.no_feasible_grasp"
    out = run_task_skill(K["perception.propose_grasps"], req, station)
    assert out["status"] == "succeeded", out["failure_type"]
    assert out["evidence"]["min_tcp_height_source"] == "station workspace_min_m[2]"


def test_station_env_keeps_the_worlds_independent_verdict(K):
    g = graphs.domino_line(recovery=False, attempts=1, slots=SLOTS)
    order = []

    def judged(seed):
        built = world(seed)
        built["independent"] = lambda: order.append("independent") or {"slots_ok": seed}
        built["interventions"] = lambda: order.append("interventions") or 0
        return built

    episode = StationEnv(judged, K)(g, 0)
    assert episode["independent"] == {"slots_ok": 0}
    assert order == ["independent", "interventions"]

    seen = []

    def hook(ep, verdict, built):
        seen.append((ep["seed"], verdict, "backend" in built, "task_success" in ep))
        return {"paired": len(ep["steps"]), **verdict}

    episode = StationEnv(judged, K, independent=hook)(g, 1)
    assert seen == [(1, {"slots_ok": 1}, True, True)]
    assert episode["independent"] == {"paired": len(episode["steps"]), "slots_ok": 1}

    plain = StationEnv(world, K)(g, 0)
    assert "independent" not in plain
    assert StationEnv(world, K, independent=lambda ep, v, b: v)(g, 0)["independent"] is None
