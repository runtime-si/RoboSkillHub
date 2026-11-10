"""correcting-in-plane-offset-by-discrete-moves: collection, the nudge loop and the move-classifier labels."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "correcting-in-plane-offset-by-discrete-moves"


@pytest.fixture(scope="module")
def bundle(skills_registry):
    b = skills_registry.get(BUNDLE)
    assert b.kind == "skill"
    return b


@pytest.fixture(scope="module")
def align(bundle):
    return bundle.canonical_scripts["align"].module


@pytest.fixture(scope="module")
def collect(bundle):
    return bundle.canonical_scripts["collect_offsets"].module


@pytest.fixture(scope="module")
def mc(skills_registry):
    return skills_registry.get("move-classifier").tools_module


def quat_yaw(deg):
    # tool pointing down (x-axis 180 deg flip), turned by yaw about world z
    a = math.radians(deg) / 2
    qz = np.array([0, 0, math.sin(a), math.cos(a)])
    qx = np.array([1.0, 0, 0, 0])  # 180 deg about x

    def mul(p, q):
        x1, y1, z1, w1 = p
        x2, y2, z2, w2 = q
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ]
        )

    q = mul(qz, qx)
    return dict(zip("xyzw", map(float, q), strict=True))


class Arm:
    """A hand that lands where commanded; the last joint turns it about world z (gain -1)."""

    def __init__(self, x=0.3, y=0.1, z=0.15, yaw=30.0, refuse=lambda pose: False):
        self.p = np.array([x, y, z])
        self.yaw = yaw
        self.q = np.zeros(6)
        self.refuse = refuse
        self.moves = 0

    def tools(self, extra=None):
        t = {
            "robot.get_ee_pose": lambda **kw: {
                "position": dict(zip("xyz", map(float, self.p), strict=True)),
                "rotation": quat_yaw(self.yaw),
            },
            "robot.go_to_pose": self.go,
            "robot.go_to_pose_cartesian": self.go,
            "robot.move_to_joints": self.joints,
            "robot.get_observation": self.obs,
        }
        t.update(extra or {})
        return t

    def go(self, pose, **kw):
        if self.refuse(pose):
            return {"outcome": "unreachable", "reached": False}
        self.p = np.array([pose["position"][k] for k in "xyz"])
        self.moves += 1
        return {"outcome": "reached"}

    def joints(self, joint_config, **kw):
        q = np.asarray(joint_config["positions"])
        self.yaw -= math.degrees(q[-1] - self.q[-1])
        self.q = q
        return {"outcome": "reached"}

    def obs(self):
        img = np.full((40, 60, 3), 100, np.uint8)
        return {
            "cameras": [{"name": "wrist", "rgb": img}],
            "arms": [{"joint_state": {"positions": self.q.tolist()}}],
        }


def oracle(arm, target, deadband_m=0.0005, deadband_deg=0.2, mc=None):
    def predict(image, model_path):
        lab = mc.label_for_offset(
            arm.p[0] - target[0],
            arm.p[1] - target[1],
            (arm.yaw - target[2] + 180) % 360 - 180,
            deadband_m,
            deadband_deg,
        )
        return {
            "label": lab,
            "x": lab[0],
            "y": lab[1],
            "yaw": lab[2],
            "confidence": 0.9,
            "aligned": lab == "000",
        }

    return predict


# --- move-classifier ---------------------------------------------------------------


def test_labels_undo_the_offset_with_upstream_deadbands(mc):
    assert len(mc.CLASSES) == 27 and mc.CLASSES.count("000") == 1
    assert mc.label_for_offset(0.003, -0.002, 1.0) == "-+-"
    assert mc.label_for_offset(0.0004, -0.0004, 0.2) == "000"
    assert mc.label_for_offset(0.0006, 0.0, -0.21) == "-0+"


def test_import_does_not_pull_torch(mc):
    assert mc._models == {}
    assert "torch" not in vars(mc)


def test_predict_rejects_bad_images(mc):
    with pytest.raises(ValueError, match="uint8"):
        mc.predict(np.zeros((4, 4, 3), np.float32), "m.pth")


def test_tool_schemas(tool_registry):
    assert set(tool_registry.get("move-classifier.predict").schema.inputs) == {
        "image",
        "model_path",
    }
    assert {"dataset_dir", "model_path", "epochs"} <= set(
        tool_registry.get("move-classifier.train").schema.inputs
    )


# --- align ----------------------------------------------------------------------------


def test_aligns_from_an_offset_with_the_oracle(align, mc):
    arm = Arm(x=0.3 + 0.0042, y=0.1 - 0.0031, yaw=30 + 2.2)
    target = (0.3, 0.1, 30.0)
    ctx = FakeContext(arm.tools({"move-classifier.predict": oracle(arm, target, mc=mc)}))
    out = align.run(ctx, model_path="m.pth", camera="wrist", crop=[0, 10, 5, 5])
    assert out["route"] == "aligned", out
    assert abs(arm.p[0] - 0.3) <= 0.0005 + 1e-9 and abs(arm.p[1] - 0.1) <= 0.0005 + 1e-9
    assert abs(arm.yaw - 30.0) <= 0.45
    assert out["decisions"] >= 3 + 4


def test_crop_is_applied(align, mc):
    seen = []
    arm = Arm()

    def pred(image, model_path):
        seen.append(image.shape)
        return {"label": "000", "aligned": True, "confidence": 1.0}

    align.run(
        FakeContext(arm.tools({"move-classifier.predict": pred})),
        model_path="m",
        camera="wrist",
        crop=[0, 10, 5, 5],
    )
    assert seen[0] == (30, 50, 3) and len(seen) == 3


def test_dithering_hits_max_steps(align):
    arm = Arm()
    labels = iter(["+00", "-00"] * 100)

    def pred(image, model_path):
        lab = next(labels)
        return {"label": lab, "aligned": False, "confidence": 1.0}

    out = align.run(
        FakeContext(arm.tools({"move-classifier.predict": pred})),
        model_path="m",
        camera="wrist",
        max_decisions=10,
    )
    assert out["route"] == "max_steps" and out["decisions"] == 10


def test_runaway_routes_diverging(align):
    arm = Arm()
    pred = lambda image, model_path: {"label": "+0+", "aligned": False, "confidence": 1.0}  # noqa: E731
    out = align.run(
        FakeContext(arm.tools({"move-classifier.predict": pred})),
        model_path="m",
        camera="wrist",
        max_drift_m=0.005,
    )
    assert out["route"] == "diverging" and "from the start" in out["reason"]


def test_refused_step_routes_diverging(align):
    arm = Arm(refuse=lambda pose: True)  # both move tools refuse
    pred = lambda image, model_path: {"label": "+00", "aligned": False, "confidence": 1.0}  # noqa: E731
    out = align.run(
        FakeContext(arm.tools({"move-classifier.predict": pred})), model_path="m", camera="wrist"
    )
    assert out["route"] == "diverging" and "refused" in out["reason"]


def test_no_frame_and_failed_classifier(align):
    arm = Arm()
    t = arm.tools({"move-classifier.predict": lambda **kw: {}})
    t["robot.get_observation"] = {"cameras": []}
    assert align.run(FakeContext(t), model_path="m", camera="wrist")["route"] == "no_frame"

    def boom(**kw):
        raise RuntimeError("no weights")

    out = align.run(
        FakeContext(arm.tools({"move-classifier.predict": boom})), model_path="m", camera="wrist"
    )
    assert out["route"] == "no_frame" and "no weights" in out["reason"]


def test_low_confidence_aligned_does_not_count(align):
    arm = Arm()
    pred = lambda image, model_path: {"label": "000", "aligned": True, "confidence": 0.3}  # noqa: E731
    out = align.run(
        FakeContext(arm.tools({"move-classifier.predict": pred})),
        model_path="m",
        camera="wrist",
        min_confidence=0.5,
        max_decisions=5,
    )
    assert out["route"] == "max_steps"


def test_second_arm_passes_arm_id(align, mc):
    arm = Arm()
    calls = []
    t = arm.tools({"move-classifier.predict": lambda **kw: {"label": "+00", "aligned": False}})
    go = t["robot.go_to_pose"]
    t["robot.go_to_pose"] = lambda **kw: (calls.append(kw.get("arm_id")), go(**kw))[1]
    t["robot.get_observation"] = lambda: {
        "cameras": [{"name": "wrist", "rgb": np.zeros((4, 4, 3), np.uint8)}],
        "arms": [{}, {"joint_state": [0.0] * 6}],
    }
    align.run(FakeContext(t), model_path="m", camera="wrist", arm_id=1, max_decisions=2)
    assert calls and set(calls) == {1}


# --- collect ---------------------------------------------------------------------------


def test_collect_labels_from_the_read_back_and_returns_home(collect, tmp_path, mc):
    arm = Arm()
    home = (arm.p.copy(), arm.yaw)
    out = collect.run(
        FakeContext(arm.tools()),
        dataset_dir=str(tmp_path / "ds"),
        camera="wrist",
        samples=6,
        seed=3,
    )
    assert out["route"] == "collected" and out["samples"] == 6
    rows = [json.loads(line) for line in (tmp_path / "ds" / "index.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    for r in rows:
        assert abs(r["dx"]) <= 0.012 and abs(r["dy"]) <= 0.012 and abs(r["dyaw_deg"]) <= 5.0 + 1e-6
        assert r["dx"] == pytest.approx(r["commanded"][0], abs=1e-9)
        assert r["dyaw_deg"] == pytest.approx(r["commanded"][2], abs=1e-6)
        assert (tmp_path / "ds" / r["image"]).exists()
    assert np.allclose(arm.p, home[0]) and arm.yaw == pytest.approx(home[1], abs=1e-6)
    rows2 = mc.read_index(tmp_path / "ds")
    assert all(Path(r["path"]).exists() for r in rows2)


def test_collect_appends_and_counts_refusals(collect, tmp_path):
    arm = Arm(refuse=lambda pose: pose["position"]["x"] > 0.305)
    out = collect.run(
        FakeContext(arm.tools()),
        dataset_dir=str(tmp_path / "ds"),
        camera="wrist",
        samples=4,
        seed=1,
    )
    assert out["route"] == "collected" and out["refused"] > 0
    out = collect.run(
        FakeContext(arm.tools()),
        dataset_dir=str(tmp_path / "ds"),
        camera="wrist",
        samples=2,
        seed=2,
    )
    assert len((tmp_path / "ds" / "index.jsonl").read_text().splitlines()) == 6
    arm = Arm(refuse=lambda pose: True)
    out = collect.run(
        FakeContext(arm.tools()),
        dataset_dir=str(tmp_path / "ds2"),
        camera="wrist",
        samples=4,
        max_refused=3,
    )
    assert out["route"] == "move_refused" and out["samples"] == 0


def test_examples_declare_every_route(bundle):
    ex = json.loads((Path(bundle.bundle_dir) / "examples" / "canonical_subgraph.json").read_text())
    assert set(ex["exit"]["success_values"]) | {ex["on_error"]} == {
        "aligned",
        "max_steps",
        "diverging",
        "no_frame",
    }
