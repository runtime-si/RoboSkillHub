"""inserting-pauses-to-resolve-two-arm-conflicts on the bundled YAM sphere model."""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext


@pytest.fixture(scope="module")
def ip(skills_registry):
    return (
        skills_registry.get("inserting-pauses-to-resolve-two-arm-conflicts")
        .canonical_scripts["insert_pauses"]
        .module
    )


Z = np.zeros(6)


def _find_clash(ip):
    """Joint goals for both arms whose grippers meet in the middle (searched once, seeded)."""
    m = ip._model()
    ba, bb = ip._base(m, 0, None), ip._base(m, 1, None)
    rng = np.random.default_rng(1)
    lim = np.array([(-2.6, 3.0), (0, 3.6), (0, 3.6), (-1.57, 1.57), (-1.57, 1.57), (-2.0, 2.0)])
    while True:
        qa, qb = rng.uniform(lim[:, 0], lim[:, 1]), rng.uniform(lim[:, 0], lim[:, 1])
        if (
            ip.clearance(qa, qb, ba, bb, m) < -0.03
            and ip.clearance(qa, Z, ba, bb, m) > 0.05
            and ip.clearance(Z, qb, ba, bb, m) > 0.05
        ):
            return qa, qb


@pytest.fixture(scope="module")
def clash(ip):
    return _find_clash(ip)


def _traj(q, t0=0.0, hold=1.0):
    return {"times": [t0, t0 + 1.0, t0 + 1.0 + hold, t0 + 2.0 + hold], "joints": [Z, q, q, Z]}


def test_apart_arms_need_no_pause(ip):
    out = ip.run(FakeContext({}), trajectory_a=_traj(Z), trajectory_b=_traj(Z))
    assert (
        out["route"] == "no_conflict" and out["pauses"] == [] and out["min_clearance_before"] > 0.3
    )


def test_a_clash_is_resolved_by_delaying_the_shorter_arm(ip, clash):
    qa, qb = clash
    a, b = _traj(qa, hold=1.5), _traj(qb, hold=0.5)
    out = ip.run(FakeContext({}), trajectory_a=a, trajectory_b=b)
    assert out["route"] == "pauses_inserted" and out["delayed"] == "b"
    assert out["min_clearance_before"] < 0 and out["min_clearance_after"] >= ip.MARGIN
    assert out["trajectory_a"] == {
        "times": a["times"],
        "joints": [list(map(float, j)) for j in a["joints"]],
    }
    np.asarray(out["trajectory_b"]["joints"])

    # Path unchanged: the same distinct waypoints in order (the holds only repeat one).
    def dedup(js):
        js = np.asarray(js, float)
        return np.asarray(
            [js[0]] + [j for k, j in enumerate(js[1:], 1) if not np.allclose(j, js[k - 1])]
        )

    assert np.allclose(dedup(out["trajectory_b"]["joints"]), dedup(b["joints"]))
    assert out["added_time"] == pytest.approx(sum(p["duration"] for p in out["pauses"]))


def test_delay_can_be_forced_and_unresolved_when_capped(ip, clash):
    qa, qb = clash
    out = ip.run(FakeContext({}), trajectory_a=_traj(qa), trajectory_b=_traj(qb), delay="a")
    assert out["delayed"] == "a"
    out = ip.run(
        FakeContext({}), trajectory_a=_traj(qa, hold=5.0), trajectory_b=_traj(qb), max_pause=0.25
    )
    assert out["route"] == "unresolved"


def test_bad_inputs(ip):
    with pytest.raises(ValueError, match="trajectory_a"):
        ip.run(FakeContext({}), trajectory_a={"times": [0.0], "joints": [Z]}, trajectory_b=_traj(Z))
    with pytest.raises(ValueError, match="increase"):
        ip.run(
            FakeContext({}),
            trajectory_a={"times": [1.0, 0.0], "joints": [Z, Z]},
            trajectory_b=_traj(Z),
        )
    with pytest.raises(ValueError, match="delay"):
        ip.run(FakeContext({}), trajectory_a=_traj(Z), trajectory_b=_traj(Z), delay="c")
