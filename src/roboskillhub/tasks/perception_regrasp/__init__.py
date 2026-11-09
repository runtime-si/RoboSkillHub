"""Perception and regrasp for the LEGO / domino tasks (Agent 1).

Single-object observations from two providers (simulator truth, labelled
privileged; RGB-D via the hub's pose6d), domino resting state, LEGO lattice pose,
parallel-jaw grasp candidates for the YAM linear_4310, held-object pose
``T_TCP_object`` and the reorient-or-regrasp planner with a whole-joint-path
feasibility check. Pure functions and adapters: nothing here moves a robot.

Not registered as a runtime capability; Agent 5 owns that wiring.
"""

from .consumers import bind_domino_target, bind_lego_mating, bind_target
from .contract import SCHEMA_VERSION, ObjectObservation, Uncertainty, result
from .domino import domino_state
from .grasp import (
    LINEAR_4310,
    GraspCandidate,
    GraspLimits,
    GripperSpec,
    propose_grasps,
    to_grasp_request,
)
from .held import (
    HeldObject,
    check_held,
    estimate_from_observation,
    prior_from_grasp,
    register_held_points,
)
from .kinematics import CallableIk, YamIk, evaluate_joint_path
from .lego import brick_pose_in_plate, mating_target, snap_to_lattice
from .objects import ObjectSpec, domino, lego_brick
from .providers import MaskedObject, RgbdFrame, RgbdProvider, SimTruthProvider, pose_error
from .regrasp import Workcell, plan_reorient_or_regrasp, plan_transfer

__all__ = [
    "LINEAR_4310",
    "SCHEMA_VERSION",
    "CallableIk",
    "GraspCandidate",
    "GraspLimits",
    "GripperSpec",
    "HeldObject",
    "MaskedObject",
    "ObjectObservation",
    "ObjectSpec",
    "RgbdFrame",
    "RgbdProvider",
    "SimTruthProvider",
    "Uncertainty",
    "Workcell",
    "YamIk",
    "bind_domino_target",
    "bind_lego_mating",
    "bind_target",
    "brick_pose_in_plate",
    "check_held",
    "domino",
    "domino_state",
    "estimate_from_observation",
    "evaluate_joint_path",
    "lego_brick",
    "mating_target",
    "plan_reorient_or_regrasp",
    "plan_transfer",
    "pose_error",
    "prior_from_grasp",
    "propose_grasps",
    "register_held_points",
    "result",
    "snap_to_lattice",
    "to_grasp_request",
]
