"""lego.verify_connection: executor-independent verdicts on the kit's negatives (lateral offset,
angle error, partial engagement, not pressed fully, insufficient support, lifted on release,
damage to the existing build) and on unusable evidence. Synthetic observations only."""

from __future__ import annotations

import pytest
from lego_fixtures import obs, placed, plate, seat_of, shifted

from roboskillhub.tasks.lego import verify_connection
from roboskillhub.tasks.lego.fakes import FakeClock

CELL = {"x": 4, "y": 4, "layer": 1, "rotation": 0}
T_REL = 1000.0


def request(
    clock,
    brick_pose_fn=lambda seat: seat,
    *,
    system="lego",
    studs="2x4",
    cell=CELL,
    n=3,
    dt=0.3,
    sigma=0.0001,
    build=None,
    neighbours=None,
    **over,
):
    seat = seat_of(system, studs, cell)
    observations = []
    for i in range(n):
        p = brick_pose_fn(seat)
        if isinstance(p, list):  # per-sample poses
            p = p[i]
        observations.append(
            obs(p, T_REL + 0.2 + i * dt, source="overhead-depth", sigma=sigma, obs_id=f"o{i}")
        )
    clock.t = T_REL + 0.2 + (n - 1) * dt + 0.05
    req = {
        "request_id": "req-3",
        "connection_id": "conn-1",
        "brick_id": "b1",
        "brick": {"system": system, "studs": studs},
        "plate": plate(system),
        "target_cell": cell,
        "build": build or [],
        "executor_id": "yam-left-press",
        "released_at_s": T_REL,
        "release": obs(
            {"gripper_open": True, "retreated": True},
            T_REL + 0.1,
            source="gripper-encoder+tcp-fk",
            frame_id="tcp",
        ),
        "observations": observations,
        "neighbours": neighbours or [],
        "window": {"window_s": 0.5, "min_samples": 2, "max_obs_age_s": 2.0},
    }
    req.update(over)
    return req


def run(req, clock):
    return verify_connection(req, clock=clock)


def test_seated_brick_is_satisfied_but_only_geometrically():
    clock = FakeClock()
    out = run(request(clock), clock)
    assert out["status"] == "succeeded", out["message"]
    assert out["evidence"]["verdict"] == "satisfied"
    assert out["evidence"]["claim"] == "geometric_seating_verified"
    assert out["evidence"]["mechanical_interlock_verified"] is False
    assert out["motion_issued"] is False


def test_symmetric_yaw_is_accepted():
    clock = FakeClock()
    out = run(request(clock, lambda s: shifted(s, yaw=180.0)), clock)
    assert out["status"] == "succeeded"


@pytest.mark.parametrize(
    "fn, code",
    [
        (lambda s: shifted(s, dx=0.0008), "lateral_offset"),
        (lambda s: shifted(s, dx=0.008), "wrong_cell"),
        (lambda s: shifted(s, yaw=3.0), "angle_error"),
        (lambda s: shifted(s, dz=0.0009), "not_pressed_fully"),
        (lambda s: shifted(s, dz=0.0017), "resting_on_studs"),
        (lambda s: shifted(s, dz=0.004), "off_studs"),
        (lambda s: shifted(s, dz=0.0003, tilt_x=1.5), "partial_engagement"),
        (lambda s: shifted(s, dz=-0.0096), "below_seat"),
        (lambda s: None, "brick_missing"),
    ],
)
def test_connection_negatives_are_violated(fn, code):
    clock = FakeClock()
    out = run(request(clock, fn), clock)
    assert out["status"] == "failed", out
    assert out["reason_code"] == code, out["evidence"]["findings"]
    assert out["evidence"]["verdict"] == "violated"


def test_partial_engagement_from_corner_heights():
    clock = FakeClock()
    req = request(clock)
    for o in req["observations"]:
        o["value"] = {"pose": o["value"], "corner_bottom_heights_m": [0.0, 0.0, 0.0012, 0.0012]}
    out = run(req, clock)
    assert out["status"] == "failed" and out["reason_code"] == "partial_engagement"


def test_brick_lifted_on_release():
    clock = FakeClock()
    seat = seat_of("lego", "2x4", CELL)
    req = request(clock, lambda s: shifted(s, dz=0.0012))
    req["pre_release_observation"] = obs(seat, T_REL - 0.2, source="overhead-depth")
    out = run(req, clock)
    assert out["status"] == "failed" and out["reason_code"] == "lifted_on_release"
    # Without the pre-release observation the same 1.2 mm reads as resting on the studs.
    req.pop("pre_release_observation")
    assert run(req, clock)["reason_code"] == "resting_on_studs"


def test_brick_moving_during_the_window_is_unstable():
    clock = FakeClock()
    seat = seat_of("lego", "2x4", CELL)
    poses = [seat, shifted(seat, dz=0.0002), shifted(seat, dz=0.0005, dx=0.0001)]
    out = run(request(clock, lambda s: poses), clock)
    assert out["status"] == "failed"
    assert "moved_after_release" in {f["reason"] for f in out["evidence"]["findings"]}


def test_damaging_the_existing_build_is_violated():
    clock = FakeClock()
    nb = placed("n1", 8, 4)
    moved = shifted(seat_of("lego", "2x4", nb["cell"]), dx=0.001, tilt_x=2.0)
    req = request(
        clock,
        build=[nb],
        neighbours=[obs({"brick_id": "n1", "pose": moved}, T_REL + 0.3, source="overhead-depth")],
    )
    out = run(req, clock)
    assert out["status"] == "failed" and out["reason_code"] == "build_disturbed"
    assert out["evidence"]["neighbours"][0]["disturbed"] is True


def test_neighbour_knocked_off_is_violated():
    clock = FakeClock()
    nb = placed("n1", 8, 4)
    req = request(
        clock,
        build=[nb],
        neighbours=[obs({"brick_id": "n1", "pose": None}, T_REL + 0.3, source="overhead-depth")],
    )
    assert run(req, clock)["reason_code"] == "build_disturbed"


def test_insufficient_support_shows_as_brick_below_its_layer():
    clock = FakeClock()
    cell2 = {"x": 4, "y": 4, "layer": 2, "rotation": 0}
    # The supporting brick is not there: the brick sits on the plate, a layer low.
    out = run(
        request(clock, lambda s: shifted(s, dz=-0.0096), cell=cell2, build=[placed("u", 4, 4)]),
        clock,
    )
    assert out["status"] == "failed" and out["reason_code"] == "below_seat"


def test_unobserved_required_neighbour_is_unknown():
    clock = FakeClock()
    req = request(clock, build=[placed("n1", 8, 4)], must_observe_neighbours=["n1"])
    out = run(req, clock)
    assert out["status"] == "unverified" and out["reason_code"] == "neighbour_unobserved"


# ------------------------------------------------------------ evidence problems -> unknown


@pytest.mark.parametrize(
    "mutate, code",
    [
        (
            lambda r: [o.update(source="yam-left-press") for o in r["observations"]],
            "no_usable_observation",
        ),
        (
            lambda r: [o.update(kind="commanded") for o in r["observations"]],
            "no_usable_observation",
        ),
        (
            lambda r: [o.update(timestamp_s=T_REL - 1) for o in r["observations"]],
            "no_usable_observation",
        ),
        (lambda r: [o.update(occluded=True) for o in r["observations"]], "no_usable_observation"),
        (lambda r: r.update(observations=[]), "no_usable_observation"),
        (lambda r: r.update(observations=r["observations"][:1]), "window_incomplete"),
        (lambda r: r["release"].update(source="yam-left-press"), "release_unconfirmed"),
        (lambda r: r["release"].update(value={"gripper_open": True}), "release_unconfirmed"),
    ],
)
def test_unusable_evidence_is_unknown_not_success(mutate, code):
    clock = FakeClock()
    req = request(clock)
    mutate(req)
    out = run(req, clock)
    assert out["status"] == "unverified", out
    assert out["reason_code"] == code
    assert out["evidence"]["claim"] == "none"


def test_stale_observations_are_unknown():
    clock = FakeClock()
    req = request(clock)
    clock.t += 10
    out = run(req, clock)
    assert out["status"] == "unverified" and out["reason_code"] == "no_usable_observation"
    assert "stale_observation" in out["message"]


def test_yam_camera_resolution_cannot_verify_lego_seating():
    clock = FakeClock()
    # 5.4 mm RMS (a calibrated YAM station camera) vs a 0.6 mm LEGO seat band.
    out = run(request(clock, sigma=0.0054), clock)
    assert out["status"] == "unverified" and out["reason_code"] == "resolution_insufficient"


def test_duplo_seat_band_is_wider():
    clock = FakeClock()
    duplo_cell = {"x": 2, "y": 2, "layer": 1, "rotation": 0}
    out = run(
        request(
            clock,
            lambda s: shifted(s, dz=0.0012),
            system="duplo",
            studs="2x2",
            cell=duplo_cell,
            sigma=0.0003,
        ),
        clock,
    )
    assert out["status"] == "succeeded"
    out = run(
        request(
            clock,
            lambda s: shifted(s, dz=0.0045),
            system="duplo",
            studs="2x2",
            cell=duplo_cell,
            sigma=0.0003,
        ),
        clock,
    )
    assert out["reason_code"] == "resting_on_studs"


def test_privileged_truth_is_labelled():
    clock = FakeClock()
    req = request(clock, observer_evidence_level="dynamic_sim_observed")
    for o in req["observations"]:
        o.update(kind="sim_truth", privileged=True)
    out = run(req, clock)
    assert out["status"] == "succeeded"
    assert out["privileged_observation"] is True
    assert out["evidence_level"] == "dynamic_sim_privileged"


def test_unlabelled_sim_truth_is_rejected():
    clock = FakeClock()
    req = request(clock)
    req["observations"][0].update(kind="sim_truth", privileged=False)
    out = run(req, clock)
    assert out["status"] == "rejected" and out["reason_code"] == "unlabelled_privileged"


def test_missing_executor_id_rejected():
    clock = FakeClock()
    req = request(clock)
    req.pop("executor_id")
    assert run(req, clock)["status"] == "rejected"
