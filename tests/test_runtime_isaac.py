"""CPU contract checks against faithful Isaac API doubles, not simulator validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from roboskillhub.runtime import isaac
from roboskillhub.runtime.contracts import EvidenceMissing, ExecutionFailed, Limits, Rejected
from roboskillhub.runtime.engine import Runtime
from roboskillhub.runtime.isaac import IsaacBackend, simulation_object_provider


@dataclass
class Action:
    joint_positions: object = None
    joint_indices: object = None
    joint_velocities: object = None


class Robot:
    def __init__(self):
        # A test encoding of TCP pose plus two finger coordinates, not YAM FK.
        self.q = np.array([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.04, 0.04])
        self.pending = self.q.copy()
        self.actions = []
        self.base = (np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]))

    def get_world_pose(self):
        return self.base

    def get_joint_positions(self, joint_indices=None):
        return self.q.copy() if joint_indices is None else self.q[joint_indices].copy()

    def apply_action(self, action):
        self.actions.append(action)
        indices = np.arange(len(self.q)) if action.joint_indices is None else action.joint_indices
        self.pending[indices] = action.joint_positions
        # Targets are not state; only World.step advances the test articulation.


class World:
    def __init__(self, robot, dt=0.02):
        self.robot, self.dt, self.steps = robot, dt, 0
        self.time_s = 0.0
        self.playing, self.stall = True, False
        self.on_step = lambda: None

    def get_physics_dt(self):
        return self.dt

    @property
    def current_time_step_index(self):
        return self.steps

    @property
    def current_time(self):
        return self.time_s

    def is_playing(self):
        return self.playing

    def step(self, *, render):
        assert isinstance(render, bool)
        self.steps += 1
        self.time_s += self.dt
        if not self.stall:
            self.robot.q[:] = self.robot.pending
        self.on_step()


class Kinematics:
    def __init__(self, robot):
        self.robot, self.calls, self.base_calls = robot, [], []
        self.fail_at = None

    def get_kinematics_solver(self):
        return self

    def set_robot_base_pose(self, position, orientation):
        self.base_calls.append((position.copy(), orientation.copy()))

    def compute_end_effector_pose(self):
        w, x, y, z = self.robot.q[3:7]
        matrix = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        return self.robot.q[:3].copy(), matrix

    def compute_inverse_kinematics(
        self, *, target_position, target_orientation, position_tolerance, orientation_tolerance
    ):
        self.calls.append(
            (
                target_position.copy(),
                target_orientation.copy(),
                position_tolerance,
                orientation_tolerance,
            )
        )
        values = np.concatenate([target_position, target_orientation])
        return Action(joint_positions=values, joint_indices=np.arange(7)), len(
            self.calls
        ) != self.fail_at


@pytest.fixture
def rig():
    robot = Robot()
    world = World(robot)
    solver = Kinematics(robot)
    limits = Limits.from_dict(
        {
            "frame_id": "isaac_world",
            "workspace_min_m": [-1, -1, 0],
            "workspace_max_m": [1, 1, 1],
            "max_speed_m_s": 0.2,
            "max_contact_distance_m": 0.2,
            "gripper_min_width_m": 0,
            "gripper_max_width_m": 0.08,
            "position_tolerance_m": 0.001,
        }
    )
    backend = IsaacBackend(
        world=world,
        robot=robot,
        kinematics_solver=solver,
        limits=limits,
        stage_units_in_meters=1.0,
        gripper_joint_indices=[7, 8],
        width_to_positions=lambda w: [w / 2, w / 2],
        action_factory=Action,
    )
    return backend, world, robot, solver


def pose(x=0.02, q=None):
    return {
        "frame_id": "isaac_world",
        "position_m": [x, 0, 0.2],
        "quaternion_xyzw": q or [0, 0, 0, 1],
    }


def test_motion_steps_targets_then_reports_measured_pose(rig):
    backend, world, robot, solver = rig
    out = backend.move_pose(pose(), speed_m_s=0.1, timeout_s=1, linear=True)
    assert out["ok"] and world.steps == 10
    assert out["pose"]["position_m"] == pytest.approx([0.02, 0, 0.2])
    assert [call[0][0] for call in solver.calls] == pytest.approx(np.linspace(0.002, 0.02, 10))
    assert all(np.array_equal(call[1], [1, 0, 0, 0]) for call in solver.calls)
    assert len(robot.actions) == 10
    assert all(np.array_equal(p, robot.base[0]) for p, _ in solver.base_calls)


def test_xyzw_rotation_interpolates_to_isaac_wxyz_and_reads_rotation_matrix(rig):
    backend, world, _robot, solver = rig
    q = [0, 0, math.sin(0.1), math.cos(0.1)]
    out = backend.move_pose(pose(0, q), speed_m_s=0.1, timeout_s=1, linear=False)
    assert world.steps >= 20
    assert solver.calls[-1][1] == pytest.approx([q[3], *q[:3]])
    assert out["pose"]["quaternion_xyzw"] == pytest.approx(q)


def test_half_turn_fk_conversion_is_stable(rig):
    backend, _world, robot, _solver = rig
    robot.q[3:7] = [0, 1, 0, 0]
    q = backend.get_state(timeout_s=1)["pose"]["quaternion_xyzw"]
    assert abs(q[0]) == pytest.approx(1)
    assert q[1:] == pytest.approx([0, 0, 0])


def test_stage_units_are_explicitly_converted(rig):
    backend, world, robot, solver = rig
    backend.scale = 0.01
    robot.q[:3] = [0, 0, 20]
    robot.pending = robot.q.copy()
    out = backend.move_pose(pose(), speed_m_s=0.1, timeout_s=1, linear=True)
    assert world.steps == 10
    assert solver.calls[-1][0] == pytest.approx([2, 0, 20])
    assert solver.calls[-1][2] == pytest.approx(0.1)
    assert out["pose"]["position_m"] == pytest.approx([0.02, 0, 0.2])


def test_ik_failure_holds_and_never_applies_failed_or_later_targets(rig):
    backend, world, robot, solver = rig
    solver.fail_at = 2
    with pytest.raises(ExecutionFailed, match="IK failed"):
        backend.move_pose(pose(), speed_m_s=0.1, timeout_s=1, linear=True)
    assert len(solver.calls) == 2 and world.steps == 1
    assert len(robot.actions) == 2  # first IK action, then measured joint hold
    assert robot.actions[-1].joint_indices is None
    assert robot.actions[-1].joint_positions == pytest.approx(robot.q)
    assert robot.actions[-1].joint_velocities == pytest.approx(np.zeros(9))


def test_commanded_pose_is_not_fabricated_as_success_when_articulation_stalls(rig):
    backend, world, robot, _solver = rig
    world.stall = True
    with pytest.raises(ExecutionFailed, match="did not converge"):
        backend.move_pose(pose(), speed_m_s=0.1, timeout_s=0.3, linear=True)
    assert world.steps == 15
    assert robot.q[:3] == pytest.approx([0, 0, 0.2])
    assert robot.actions[-1].joint_positions == pytest.approx(robot.q)


def test_wall_deadline_stops_before_next_waypoint(rig, monkeypatch):
    backend, world, robot, solver = rig
    clock = [0.0]
    monkeypatch.setattr(isaac.time, "monotonic", lambda: clock[0])
    world.on_step = lambda: clock.__setitem__(0, 2)
    with pytest.raises(ExecutionFailed, match="timed out"):
        backend.move_pose(pose(), speed_m_s=0.1, timeout_s=1, linear=True)
    assert world.steps == 1 and len(solver.calls) == 1
    assert robot.actions[-1].joint_positions == pytest.approx(robot.q)


def test_invalid_frame_and_paused_world_produce_no_actions(rig):
    backend, world, robot, _solver = rig
    with pytest.raises(Rejected, match="pose frame"):
        backend.move_pose(
            {**pose(), "frame_id": "robot_base"}, speed_m_s=0.1, timeout_s=1, linear=True
        )
    world.playing = False
    with pytest.raises(Rejected, match="already be playing"):
        backend.move_pose(pose(), speed_m_s=0.1, timeout_s=1, linear=True)
    assert not robot.actions and not world.steps


def test_gripper_uses_injected_joint_mapping_without_claiming_grasp(rig):
    backend, world, robot, _solver = rig
    result = backend.set_gripper(0.03, timeout_s=1)
    assert result["ok"] and result["gripper"]["holding"] is None
    assert robot.q[7:] == pytest.approx([0.015, 0.015])
    assert robot.actions[0].joint_indices.tolist() == [7, 8]
    assert world.steps == 1


def test_gripper_stall_times_out_and_holds(rig):
    backend, world, robot, _solver = rig
    world.stall = True
    with pytest.raises(ExecutionFailed, match="gripper did not converge"):
        backend.set_gripper(0.03, timeout_s=0.1)
    assert world.steps == 5
    assert robot.actions[-1].joint_positions == pytest.approx(robot.q)


def test_timeout_shorter_than_physics_step_rejects_without_command(rig):
    backend, world, robot, _solver = rig
    with pytest.raises(Rejected, match="at least one"):
        backend.set_gripper(0.03, timeout_s=0.001)
    assert world.steps == 0 and robot.actions == []


def test_provider_timestamp_source_are_preserved_and_input_is_not_mutated(rig):
    backend, _world, _robot, _solver = rig
    records = {"block": {"position_m": [0.2, 0, 0.1], "timestamp_s": 1234.0, "source": "camera"}}
    backend.object_provider = lambda **_: records
    state = backend.get_state(timeout_s=1)
    assert state["timestamp_s"] > 1234
    assert state["objects"]["block"] == records["block"]
    state["objects"]["block"]["position_m"][0] = 99
    assert records["block"]["position_m"][0] == 0.2


def test_sim_truth_provider_labels_prim_origins_and_converts_units(rig):
    backend, _world, robot, _solver = rig
    backend.object_provider = simulation_object_provider(
        {"handle": robot}, frame_id="isaac_world", stage_units_in_meters=0.01
    )
    record = backend.get_state(timeout_s=1)["objects"]["handle"]
    assert record["source"] == "simulation_truth"
    assert record["position_m"] == pytest.approx([0.001, 0.002, 0.003])
    assert record["timestamp_s"] > 0


def test_describe_does_not_claim_station_validation(rig):
    backend, _world, _robot, _solver = rig
    info = backend.describe()
    assert info["backend"] == "isaac" and info["station_validated"] is False
    assert info["collision_planning"] is False


def test_runtime_can_invoke_isaac_adapter_with_shared_capability_contract(rig):
    backend, world, _robot, _solver = rig
    result = Runtime(backend, backend.limits).invoke(
        "move_pose", {"target_pose": pose(), "speed_m_s": 0.1, "timeout_s": 1}
    )
    assert result["status"] == "succeeded", result
    assert result["backend"]["kind"] == "sim"
    assert result["backend"]["embodiment"] == "yam_i2rt"
    assert world.steps == 10
    assert any(event["event"] == "pose_verification" for event in result["trace"])


def test_contact_completion_retains_evidence_without_claiming_holding(rig):
    backend, world, robot, _solver = rig
    world.stall = True
    seen = []

    def contact(**request):
        seen.append(request)
        return {
            "complete": True,
            "requested_width_m": request["width_m"],
            "timestamp_s": isaac.time.time(),
            "source": "simulation_contact_sensor",
            "reason": "contact_limited",
        }

    backend.gripper_completion = contact
    result = backend.set_gripper(0.0, timeout_s=1)
    assert result["ok"] and result["gripper"]["holding"] is None
    assert result["completion"]["reason"] == "contact_limited"
    assert result["completion"]["source"] == "simulation_contact_sensor"
    assert world.steps == 1 and robot.q[7:].tolist() == [0.04, 0.04]
    assert seen[0]["current_positions"].tolist() == [0.04, 0.04]
    assert seen[0]["target_positions"].tolist() == [0.0, 0.0]


@pytest.mark.parametrize("reply", [False, {"complete": False}])
def test_contact_callback_false_does_not_report_completion(rig, reply):
    backend, world, _robot, _solver = rig
    world.stall = True
    backend.gripper_completion = lambda **_: reply
    with pytest.raises(ExecutionFailed, match="gripper did not converge"):
        backend.set_gripper(0.0, timeout_s=0.1)
    assert world.steps == 5


def test_open_command_cannot_reuse_close_completion_evidence(rig):
    backend, world, robot, _solver = rig
    robot.q[7:] = 0.015
    robot.pending = robot.q.copy()
    world.stall = True
    backend.gripper_completion = lambda **_: {
        "complete": True,
        "requested_width_m": 0.0,
        "timestamp_s": isaac.time.time(),
        "source": "simulation_contact_sensor",
        "reason": "contact_limited",
    }
    with pytest.raises(EvidenceMissing, match="different width"):
        backend.set_gripper(0.08, timeout_s=1)
    assert robot.actions[-1].joint_positions == pytest.approx(robot.q)


def test_previous_completion_evidence_cannot_finish_same_width_new_command(rig):
    backend, world, _robot, _solver = rig
    world.stall = True
    old_stamp = isaac.time.time() - 10
    backend.gripper_completion = lambda **request: {
        "complete": True,
        "requested_width_m": request["width_m"],
        "timestamp_s": old_stamp,
        "source": "simulation_contact_sensor",
        "reason": "contact_limited",
    }
    with pytest.raises(EvidenceMissing, match="stale"):
        backend.set_gripper(0.0, timeout_s=1)


def test_object_capability_is_declared_only_when_provider_exists(rig):
    backend, _world, _robot, _solver = rig
    assert "object_feedback" not in backend.describe()["capabilities"]
    backend.object_provider = lambda **_: {}
    assert "object_feedback" in backend.describe()["capabilities"]


def _complete_contact_close(rig):
    backend, world, robot, _solver = rig
    robot.q[7:] = 0.02
    robot.pending = robot.q.copy()
    world.stall = True
    backend.gripper_completion = lambda **request: {
        "complete": True,
        "requested_width_m": request["width_m"],
        "timestamp_s": isaac.time.time(),
        "source": "test_contact_sensor",
        "reason": "contact_limited",
    }
    result = backend.set_gripper(0.0, timeout_s=1)
    assert result["ok"] and result["gripper"]["holding"] is None
    assert robot.q[7:] == pytest.approx([0.02, 0.02])
    return result


def test_failed_arm_move_preserves_completed_contact_close_target_and_measured_arm(rig):
    backend, _world, robot, solver = rig
    _complete_contact_close(rig)
    solver.fail_at = 1
    with pytest.raises(ExecutionFailed, match="IK failed"):
        backend.move_pose(pose(), speed_m_s=0.1, timeout_s=1, linear=True)
    hold = robot.actions[-1]
    assert hold.joint_positions[:7] == pytest.approx(robot.q[:7])
    assert hold.joint_positions[7:] == pytest.approx([0, 0])
    assert hold.joint_velocities == pytest.approx(np.zeros(9))
    assert backend.get_state(timeout_s=1)["gripper"]["holding"] is None
    backend.stop()  # Runtime may stop again after the adapter aborts.
    assert robot.actions[-1].joint_positions[7:] == pytest.approx([0, 0])


def test_stop_preserves_reached_gripper_target_despite_later_measured_deflection(rig):
    backend, _world, robot, _solver = rig
    backend.set_gripper(0.03, timeout_s=1)
    robot.q[7:] = [0.017, 0.018]
    backend.stop()
    assert robot.actions[-1].joint_positions[:7] == pytest.approx(robot.q[:7])
    assert robot.actions[-1].joint_positions[7:] == pytest.approx([0.015, 0.015])


def test_cancelled_incomplete_close_uses_measured_fingers_and_never_restores_target(rig):
    backend, world, robot, _solver = rig
    world.stall = True
    with pytest.raises(ExecutionFailed, match="gripper did not converge"):
        backend.set_gripper(0.0, timeout_s=0.1)
    assert robot.actions[-1].joint_positions == pytest.approx(robot.q)
    robot.q[7:] = [0.027, 0.028]
    backend.stop()
    assert robot.actions[-1].joint_positions == pytest.approx(robot.q)


def test_cancelled_open_does_not_restore_previous_completed_close(rig):
    backend, _world, robot, _solver = rig
    _complete_contact_close(rig)
    backend.gripper_completion = None
    with pytest.raises(ExecutionFailed, match="gripper did not converge"):
        backend.set_gripper(0.08, timeout_s=0.1)
    # Measured .02 differs from both the old close target 0 and new open .04.
    assert robot.actions[-1].joint_positions[7:] == pytest.approx([0.02, 0.02])
    backend.stop()
    assert robot.actions[-1].joint_positions[7:] == pytest.approx([0.02, 0.02])


def test_completed_open_replaces_old_close_target(rig):
    backend, world, robot, _solver = rig
    _complete_contact_close(rig)
    world.stall = False
    backend.set_gripper(0.08, timeout_s=1)
    robot.q[7:] = [0.036, 0.037]
    backend.stop()
    assert robot.actions[-1].joint_positions[7:] == pytest.approx([0.04, 0.04])


@pytest.mark.parametrize("cancel_at", ["physics_step", "completion_callback"])
def test_cancel_during_gripper_step_cannot_be_accepted_as_completed(rig, cancel_at):
    backend, world, robot, _solver = rig
    if cancel_at == "physics_step":
        world.on_step = backend.stop
    else:
        world.stall = True

        def cancelled_completion(**request):
            backend.stop()
            return {
                "complete": True,
                "requested_width_m": request["width_m"],
                "timestamp_s": isaac.time.time(),
                "source": "test_contact_sensor",
                "reason": "contact_limited",
            }

        backend.gripper_completion = cancelled_completion
    with pytest.raises(ExecutionFailed, match="stopped"):
        backend.set_gripper(0.03, timeout_s=1)
    robot.q[7:] = [0.025, 0.026]
    backend.stop()
    assert robot.actions[-1].joint_positions == pytest.approx(robot.q)


@pytest.mark.parametrize("rewind", ["step", "time", "both"])
def test_world_reset_clock_rewind_discards_completed_gripper_target(rig, rewind):
    backend, world, robot, _solver = rig
    _complete_contact_close(rig)
    if rewind in {"step", "both"}:
        world.steps = 0
    if rewind in {"time", "both"}:
        world.time_s = 0.0
    robot.q[7:] = [0.035, 0.036]
    backend.stop()
    assert robot.actions[-1].joint_positions == pytest.approx(robot.q)
    # Advancing again must not revive the pre-reset target.
    world.steps, world.time_s = 100, 2.0
    backend.stop()
    assert robot.actions[-1].joint_positions == pytest.approx(robot.q)


def test_clock_rewind_after_later_arm_observation_is_detected_above_close_time(rig):
    backend, world, robot, _solver = rig
    _complete_contact_close(rig)
    world.steps, world.time_s = 100, 2.0
    backend.get_state(timeout_s=1)
    world.steps, world.time_s = 50, 1.0
    backend.stop()
    assert robot.actions[-1].joint_positions == pytest.approx(robot.q)


def test_rejected_gripper_request_does_not_invalidate_existing_completed_command(rig):
    backend, _world, robot, _solver = rig
    _complete_contact_close(rig)
    with pytest.raises(Rejected):
        backend.set_gripper(0.09, timeout_s=1)
    backend.stop()
    assert robot.actions[-1].joint_positions[7:] == pytest.approx([0, 0])
