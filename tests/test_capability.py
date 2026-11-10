"""Capability layer: the pick_object contract, the eval-gated selector, the shared verifier."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from roboskillhub.capability import PickObservation, verify_pick
from roboskillhub.capability.records import Episode, build_record, tally, write_record
from roboskillhub.capability.select import select
from roboskillhub.capability.spec import load_capability, parse, validate
from roboskillhub.capability.verify import classify_trace
from roboskillhub.checks import check_capabilities
from roboskillhub.cli import main

ROOT = Path(__file__).resolve().parents[1]

CLOSE_ON_CUBE = [0.05, 0.04, 0.032, 0.027] + [0.0262] * 9
CLOSE_ON_NOTHING = [0.05, 0.03, 0.012, 0.003, 0.0005]


def lift(follow: float, tail: float | None = None) -> list[dict]:
    """Samples over a 50 mm lift where the object rises *follow* of the TCP; *tail* is the
    share at the last sample (a drop late in the lift)."""
    out = [{"tcp": [0.3, 0.0, 0.10], "object": [0.3, 0.0, 0.02]}]
    for i, dz in enumerate((0.02, 0.035, 0.05)):
        share = tail if tail is not None and i == 2 else follow
        out.append({"tcp": [0.3, 0.0, 0.10 + dz], "object": [0.3, 0.0, 0.02 + share * dz]})
    return out


# ---------------------------------------------------------------- the contract


def test_pick_object_yaml_is_well_formed():
    spec = load_capability("pick_object", ROOT)
    assert set(spec.exits) == {"picked", "missed", "slipped", "unreachable", "refused", "timeout"}
    assert spec.preconditions == ["object_visible", "arm_free", "gripper_open"]
    assert spec.timeout_s == 30
    assert [i.id for i in spec.implementations] == [
        "program-grasp",
        "hub-grasp",
        "policy:molmoact2-yam",
        "policy:pi05-yam",
    ]
    assert spec.implementation("policy:molmoact2-yam").bundle == "molmoact2-yam"
    sim = spec.acceptance["sim"]
    assert (sim.min_episodes, sim.min_success_rate, sim.max_false_picked_rate) == (50, 0.8, 0.01)
    assert check_capabilities(ROOT) == []


def test_validate_reports_bad_fields():
    data = yaml.safe_load((ROOT / "capabilities/pick_object.yaml").read_text())
    data["implementations"].append({"id": "program-grasp", "kind": "magic"})
    data["acceptance"]["twin"] = {}
    data["timeout_s"] = -1
    del data["exits"]
    errors = validate(data)
    text = "\n".join(errors)
    for needle in ("missing exits", "listed twice", "kind must be", "twin", "timeout_s"):
        assert needle in text
    with pytest.raises(ValueError):
        parse(data)


# ---------------------------------------------------------------- selector + records


@pytest.fixture
def hub(tmp_path: Path) -> Path:
    (tmp_path / "capabilities").mkdir()
    shutil.copy(ROOT / "capabilities/pick_object.yaml", tmp_path / "capabilities")
    return tmp_path


def episodes(n: int, picked: int, false_picked: int = 0, labelled: bool = True):
    out = []
    for i in range(n):
        if i < picked:
            truth = False if i < false_picked else True
            out.append(Episode("picked", claim="picked", truth_held=truth if labelled else None))
        else:
            out.append(Episode("missed", claim="picked", truth_held=False if labelled else None))
    return out


def record(hub, impl, *, n, picked, fp=0, setting="sim", embodiment="yam_i2rt", **kw):
    spec = load_capability("pick_object", hub)
    rec = build_record(
        spec,
        impl,
        embodiment=embodiment,
        setting=setting,
        episodes=episodes(n, picked, fp, kw.pop("labelled", True)),
        suite="twin/pick_object",
        evidence="file://run",
        date=kw.pop("date", "2026-11-10"),
        **kw,
    )
    return write_record(hub, rec), rec


def test_tally_counts_false_picked_and_claim_disagreements():
    t = tally(episodes(10, 8, false_picked=1))
    assert (t.episodes, t.successes, t.false_picked) == (10, 7, 1)
    assert t.exits["picked"] == 8 and t.exits["missed"] == 2
    assert t.claim_disagreements == 2  # the implementation claimed picked on the misses


def test_record_matches_spec_eval_keys(hub):
    path, rec = record(hub, "program-grasp", n=50, picked=45)
    assert path == hub / "evals/program-grasp/yam_i2rt-sim-pick_object-2026-11-10.json"
    data = json.loads(path.read_text())
    for key in ("bundle", "version", "embodiment", "setting", "tier", "suite", "episodes"):
        assert key in data
    assert data["passed"] is True and data["gate_failures"] == []
    assert data["implementation"] == "program-grasp" and data["verifier"] == "verify_pick/1"
    path2, _ = record(hub, "program-grasp", n=50, picked=45)
    assert path2.name.endswith("-2.json")


def test_select_only_offers_gate_passers_on_the_asked_setting_and_embodiment(hub):
    record(hub, "program-grasp", n=50, picked=45)
    record(hub, "hub-grasp", n=40, picked=40)
    record(hub, "policy:molmoact2-yam", n=60, picked=50, fp=2)
    record(hub, "policy:pi05-yam", n=50, picked=45, embodiment="franka")
    spec = load_capability("pick_object", hub)
    choices = {c.implementation.id: c for c in select(spec, hub, "yam_i2rt", "sim")}
    assert [k for k, c in choices.items() if c.eligible] == ["program-grasp"]
    assert any("40 episodes < 50" in w for w in choices["hub-grasp"].why)
    assert any("false picked" in w for w in choices["policy:molmoact2-yam"].why)
    assert any("no sim eval record" in w for w in choices["policy:pi05-yam"].why)
    assert not any(c.eligible for c in select(spec, hub, "yam_i2rt", "real"))


def test_select_ignores_the_records_own_passed_and_uses_the_newest(hub):
    path, rec = record(hub, "program-grasp", n=50, picked=45, date="2026-11-01")
    record(hub, "program-grasp", n=50, picked=30, date="2026-11-09")
    rec_unlabelled = record(hub, "hub-grasp", n=50, picked=50, labelled=False)[0]
    data = json.loads(rec_unlabelled.read_text())
    data["passed"] = True
    rec_unlabelled.write_text(json.dumps(data))
    spec = load_capability("pick_object", hub)
    choices = {c.implementation.id: c for c in select(spec, hub, "yam_i2rt")}
    assert not choices["program-grasp"].eligible
    assert "2026-11-09" in choices["program-grasp"].record["_file"]
    assert not choices["hub-grasp"].eligible
    assert any("truth label" in w for w in choices["hub-grasp"].why)


def test_cli_list_show_select(hub, capsys):
    assert main(["--root", str(hub), "capability", "list"]) == 0
    assert "pick_object" in capsys.readouterr().out
    assert main(["--root", str(hub), "capability", "show", "pick_object"]) == 0
    assert json.loads(capsys.readouterr().out)["capability"] == "pick_object"
    args = ["--root", str(hub), "capability", "select", "pick_object", "--embodiment", "yam_i2rt"]
    assert main(args) == 1
    record(hub, "program-grasp", n=50, picked=45)
    assert main(args) == 0
    assert capsys.readouterr().out.split() == ["ok", "program-grasp"]
    assert main([*args, "--why"]) == 0
    out = capsys.readouterr().out
    assert "-- hub-grasp" in out and "passes the gate" in out
    assert main([*args, "--setting", "real", "--json"]) == 1
    assert all(not c["eligible"] for c in json.loads(capsys.readouterr().out))


# ---------------------------------------------------------------- the verifier


def test_trace_classification_reuses_the_hub_tracker():
    assert classify_trace(CLOSE_ON_CUBE, 0.0)["route"] == "stalled_on_object"
    assert classify_trace(CLOSE_ON_CUBE, 0.0, expected_width_m=0.040)["route"] == (
        "unexpected_width"
    )
    assert classify_trace(CLOSE_ON_NOTHING, 0.0)["route"] == "reached"
    assert classify_trace([0.05] * 10, 0.0)["route"] == "stalled_jammed"


def test_picked_needs_two_agreeing_checks_and_ignores_the_claim():
    obs = PickObservation(gripper_trace_m=CLOSE_ON_CUBE, gripper_target_m=0.0, lift=lift(1.0))
    v = verify_pick(obs, claim="missed")
    assert v.exit == "picked" and v.claim_agrees is False and not v.uncertain
    assert v.votes["gripper"] == "held" and v.votes["lift"] == "held"
    alone = verify_pick(PickObservation(gripper={"route": "stalled_on_object"}), claim="picked")
    assert alone.exit == "missed" and alone.uncertain


def test_claim_of_picked_with_empty_jaws_is_missed():
    obs = PickObservation(
        gripper_trace_m=CLOSE_ON_NOTHING,
        gripper_target_m=0.0,
        lift=lift(0.0),
        scene={"route": "unchanged"},
    )
    v = verify_pick(obs, claim="picked")
    assert v.exit == "missed" and v.claim_agrees is False


def test_slip_from_lift_and_from_gap_collapse():
    late_drop = PickObservation(
        gripper={"route": "stalled_on_object", "actual_m": 0.026}, lift=lift(1.0, tail=0.1)
    )
    assert verify_pick(late_drop).exit == "slipped"
    never_rose = PickObservation(
        gripper={"route": "stalled_on_object", "actual_m": 0.026}, lift=lift(0.0)
    )
    assert verify_pick(never_rose).exit == "slipped"
    collapsed = PickObservation(
        gripper={"route": "stalled_on_object", "actual_m": 0.026},
        gripper_after_lift_m=0.002,
        scene={"route": "changed_as_expected"},
    )
    v = verify_pick(collapsed, claim="picked")
    assert v.exit == "slipped" and v.votes["gripper"] == "lost"


def test_wrist_remeasure_and_scene_count_as_checks():
    obs = PickObservation(gripper={"route": "stalled_on_object"}, wrist={"route": "measured"})
    assert verify_pick(obs).exit == "picked"
    obs = PickObservation(gripper={"route": "stalled_on_object"}, wrist={"held": False})
    assert verify_pick(obs).exit == "slipped"
    obs = PickObservation(scene={"route": "changed_as_expected"}, lift=lift(0.95))
    assert verify_pick(obs).exit == "picked"


def test_scene_numbers_from_frames_without_a_vlm():
    before = np.full((64, 64, 3), 200, np.uint8)
    before[24:40, 24:40] = (20, 40, 200)
    after = np.full((64, 64, 3), 200, np.uint8)
    obs = PickObservation(
        before_rgb=before, after_rgb=after, source_box_px=[24, 24, 40, 40], lift=lift(1.0)
    )
    v = verify_pick(obs)
    assert v.evidence["scene"]["route"] == "vacated" and v.exit == "picked"
    same = PickObservation(before_rgb=before, after_rgb=before, source_box_px=[24, 24, 40, 40])
    assert verify_pick(same).votes["scene"] == "empty"


def test_vlm_breaks_ties_but_never_overrides():
    one = PickObservation(gripper={"route": "stalled_on_object"}, vlm={"held": True})
    v = verify_pick(one)
    assert v.exit == "picked" and v.uncertain
    against = PickObservation(
        gripper={"route": "stalled_on_object"},
        scene={"route": "unchanged"},
        vlm={"held": True},
    )
    assert verify_pick(against).exit == "missed"
    two = PickObservation(gripper={"route": "reached"}, lift=lift(0.0), vlm={"held": True})
    assert verify_pick(two).exit == "missed"


def test_unreachable_refused_timeout_and_preconditions():
    assert verify_pick(PickObservation(), claim="unreachable").exit == "unreachable"
    v = verify_pick(PickObservation(gripper={"route": "refused"}), claim="refused")
    assert v.exit == "refused"
    # a visible pick overrides a claim of unreachable
    obs = PickObservation(gripper={"route": "stalled_on_object"}, lift=lift(1.0))
    assert verify_pick(obs, claim="unreachable").exit == "picked"
    pre = PickObservation(preconditions={"object_visible": True, "gripper_open": False})
    v = verify_pick(pre, claim="picked")
    assert v.exit == "refused" and "gripper_open" in v.reason
    late = PickObservation(elapsed_s=31.0, timeout_s=30.0, lift=lift(1.0))
    assert verify_pick(late, claim="picked").exit == "timeout"


def test_sim_truth_lift_is_flagged_privileged():
    obs = PickObservation(
        gripper={"route": "stalled_on_object"}, lift=lift(1.0), lift_source="sim_truth"
    )
    v = verify_pick(obs)
    assert v.exit == "picked" and v.privileged
    assert v.to_dict()["evidence"]["lift"]["source"] == "sim_truth"
    assert Episode.from_verdict(v, truth_held=True).exit == "picked"
