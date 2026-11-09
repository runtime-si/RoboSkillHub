"""``reorient_or_regrasp``: turn the part in the hand when the whole joint path
allows it, else put it down at a staging spot in an intermediate resting pose and
take it again.

A *transfer* is: pregrasp -> grasp -> lift -> turn the part about its centre at
carry height -> carry over the target -> lower -> open -> retreat along the
approach. A transfer is feasible when the grasp is feasible on the part where it
lies, the same ``T_TCP_object`` puts the hand clear of the table at the target
(open jaws included), and IK continued sample by sample along the whole TCP path
(:func:`kinematics.evaluate_joint_path`) stays on one branch, inside the limits,
with the hand (and, while carried, the part) above the table at every sample.

PR #106 found that a 90 deg wrist pitch near the table jumps a YAM joint
mid-turn although both ends are reachable; checking only the end poses would
accept those plans. Tilted grasps (approach turned about the jaw axis) are
searched because a 90 deg turn of the part about the jaw axis is then a
-45..+45 deg pitch of the hand instead of 0..90.

The plan is a list of steps over the kit runtime's primitives; nothing here
executes them. ``release`` is not a public capability yet (integration request),
and ``move_pose`` legs are assumed to be executed as straight TCP legs; the
verified joint path is attached for a joint-space executor.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contract import ObjectObservation, result
from .domino import domino_state
from .frames import T_to_pose, inv_T, make_T, rot_z, rotation_angle
from .grasp import (
    LINEAR_4310,
    GraspCandidate,
    GraspLimits,
    GripperSpec,
    hand_lowest,
    obb_overlap,
    propose_grasps,
    to_grasp_request,
)
from .held import prior_from_grasp
from .kinematics import (
    CART_STEP_M,
    JOINT_JUMP_RAD,
    ROT_STEP_RAD,
    IkSolver,
    YamIk,
    densify,
    evaluate_joint_path,
)
from .objects import ObjectSpec, box_corners, symmetry_rotations, to_contract_frame

#: Part centre over the table while it is turned [m]: a 48 mm tile turned about its
#: centre needs 24 mm plus the hand's reach past it.
CARRY_HEIGHT_M = 0.09
#: Part centre over its target before the final descent [m].
PLACE_APPROACH_M = 0.04
#: Release this far over the resting pose [m] (PR #106 set down 1.5 mm over the table).
RELEASE_GAP_M = 0.0015
#: Part corners must stay this far over the table while carried [m].
CARRY_CLEARANCE_M = 0.005
TILTS_DEG = (0.0, -30.0, 30.0, -45.0, 45.0, -60.0, 60.0)
#: Largest lever of the part's weight about the jaw axis while carried [m]. A
#: two-pad pinch is a hinge about that axis held only by torsional friction; PR #106
#: pinched tiles 14 mm off centre and every one turned out of the hand or dropped
#: in the pitch (twin runs 2-7). 6 mm is a provisional bound, not a measured one:
#: Agent 4's contact sims should replace it.
MAX_PIVOT_ARM_M = 0.006


@dataclass
class Workcell:
    frame_id: str
    table_z: float
    T_frame_base: np.ndarray
    ik: IkSolver
    q_current: np.ndarray | None = None
    gripper: GripperSpec = LINEAR_4310
    limits: GraspLimits = field(default_factory=GraspLimits)
    gripper_id: str = "yam/linear_4310"
    staging_xy: list[tuple[float, float]] = field(default_factory=list)
    obstacles: list[tuple[str, np.ndarray, ObjectSpec]] = field(default_factory=list)
    carry_height_m: float = CARRY_HEIGHT_M
    jump_rad: float = JOINT_JUMP_RAD
    min_limit_margin_rad: float = 0.0
    max_pivot_arm_m: float | None = MAX_PIVOT_ARM_M


@dataclass
class Transfer:
    grasp: GraspCandidate
    T_from: np.ndarray
    T_to: np.ndarray
    waypoints: list[np.ndarray]
    labels: list[str]
    path: Any
    widths: list[float]
    pivot_arm_m: float = 0.0


def _resting_pose(
    spec: ObjectSpec, face_axis: int, face_sign: float, yaw_rad: float, xy, table_z: float
) -> np.ndarray:
    """Part resting on its ``face_sign * face_axis`` face at (x, y), turned ``yaw`` about up."""
    down = np.zeros(3)
    down[face_axis] = face_sign
    # Rows of R0 are the world axes in object coordinates: world +z is -down,
    # world +x the next object axis.
    ez = -down
    ex = np.eye(3)[(face_axis + 1) % 3]
    R0 = np.vstack([ex, np.cross(ez, ex), ez])
    R = rot_z(yaw_rad) @ R0
    h = spec.size_m[face_axis] / 2.0
    return make_T(R, (xy[0], xy[1], table_z + h))


def _goal_equivalents(T_goal: np.ndarray, spec: ObjectSpec) -> list[np.ndarray]:
    out = []
    for S in symmetry_rotations(spec):
        T = T_goal.copy()
        T[:3, :3] = T_goal[:3, :3] @ S
        out.append(T)
    return out


def _lowest_corner(T_obj: np.ndarray, spec: ObjectSpec) -> float:
    return float((box_corners(spec) @ T_obj[:3, :3].T + T_obj[:3, 3])[:, 2].min())


def _object_path(
    T_from: np.ndarray, T_to: np.ndarray, wc: Workcell, spec: ObjectSpec
) -> tuple[list[np.ndarray], list[str]]:
    """Part poses: lift straight up, turn in place at carry height, carry over the
    target, lower, release gap."""
    up = np.array([0.0, 0.0, 1.0])
    lift = T_from.copy()
    lift[:3, 3] = T_from[:3, 3] + up * (wc.limits.lift_m)
    carry_z = max(wc.table_z + wc.carry_height_m, lift[2, 3])
    lift_hi = T_from.copy()
    lift_hi[2, 3] = carry_z
    turned = T_to.copy()
    turned[:3, 3] = [T_from[0, 3], T_from[1, 3], carry_z]
    over = T_to.copy()
    over[2, 3] = max(carry_z, T_to[2, 3] + PLACE_APPROACH_M)
    above = T_to.copy()
    above[2, 3] = T_to[2, 3] + PLACE_APPROACH_M
    rel = T_to.copy()
    rel[2, 3] = T_to[2, 3] + RELEASE_GAP_M
    poses = [T_from, lift, lift_hi, turned, over, above, rel]
    labels = ["grasp", "lift", "carry_up", "turn", "carry_over", "place_above", "place"]
    keep = [0]
    for i in range(1, len(poses)):
        if (
            np.linalg.norm(poses[i][:3, 3] - poses[keep[-1]][:3, 3]) > 1e-4
            or rotation_angle(poses[i], poses[keep[-1]]) > 1e-4
        ):
            keep.append(i)
    return [poses[i] for i in keep], [labels[i] for i in keep]


def plan_transfer(
    T_from: np.ndarray,
    T_to: np.ndarray,
    grasp: GraspCandidate,
    spec: ObjectSpec,
    wc: Workcell,
    q_seed: np.ndarray | None,
    budget: dict[str, int] | None = None,
) -> tuple[Transfer | None, str]:
    """One pick-turn-place with a fixed ``T_object_tcp``; (transfer, "") or (None, reason).
    Only the IK path check counts against ``budget["evals"]``; the geometric
    checks before it are cheap."""
    up = np.array([0.0, 0.0, 1.0])
    T_ot = grasp.T_object_tcp
    gl, gr = wc.limits, wc.gripper
    # Static checks at the target, jaws open (the release).
    T_place_tcp = T_to @ T_ot
    low = hand_lowest(T_place_tcp, gr, grasp.open_width_m, up)
    if min(low.values()) - wc.table_z < gl.table_clearance_m:
        return (
            None,
            f"place_table_clearance ({min(low, key=low.get)} {1e3 * (min(low.values()) - wc.table_z):.1f} mm)",
        )
    if T_place_tcp[2, 3] - wc.table_z < gl.min_tcp_height_m:
        return None, "place_tcp_height"
    if float(T_place_tcp[:3, 2] @ up) > gl.max_approach_up:
        return None, "place_approach_from_below"
    obj_poses, obj_labels = _object_path(T_from, T_to, wc, spec)
    pre = (T_from @ T_ot) @ make_T(position=(0.0, 0.0, -gl.pregrasp_standoff_m))
    retreat = T_place_tcp @ make_T(position=(0.0, 0.0, -gl.pregrasp_standoff_m))
    tcp = [pre] + [T @ T_ot for T in obj_poses] + [retreat]
    labels = ["pregrasp"] + obj_labels + ["retreat"]
    widths = (
        [grasp.open_width_m]
        + [grasp.close_width_m] * (len(obj_poses) - 1)
        + [grasp.open_width_m] * 2
    )
    # Hand / part clearance along the densified TCP path (before the costlier IK).
    samples, s_labels = densify(tcp, labels, CART_STEP_M, ROT_STEP_RAD)
    carried = {"lift", "carry_up", "turn", "carry_over", "place_above"}
    pivot_arm = 0.0
    c_obj = grasp.contact_centre_obj
    for T, lab in zip(samples, s_labels, strict=True):
        if lab.split("->")[-1] in carried | {"place"}:
            R_obj = (T @ inv_T(T_ot))[:3, :3]
            r = R_obj @ (-c_obj)  # contact centre to centre of mass, world
            pivot_arm = max(pivot_arm, abs(float(np.cross(r, [0.0, 0.0, -1.0]) @ T[:3, 1])))
        leg_to = lab.split("->")[-1]
        open_jaws = leg_to in ("pregrasp", "grasp", "retreat") or lab.startswith("place->")
        w = grasp.open_width_m if open_jaws else grasp.close_width_m
        low = hand_lowest(T, gr, w, up)
        if min(low.values()) - wc.table_z < gl.table_clearance_m - 1e-6:
            return None, f"path_table_clearance at {lab} ({min(low, key=low.get)})"
        if leg_to in carried and lab != "grasp->lift":
            T_obj = T @ inv_T(T_ot)
            if _lowest_corner(T_obj, spec) - wc.table_z < CARRY_CLEARANCE_M and leg_to not in (
                "place_above",
            ):
                return None, f"part_hits_table at {lab}"
        for oid, To, ospec in wc.obstacles:
            for name, c, h in gr.hand_boxes(w):
                if obb_overlap(T @ make_T(position=c), h, To, ospec.half, gl.neighbour_margin_m):
                    return None, f"hand_hits_obstacle {oid} at {lab} ({name})"
    if wc.max_pivot_arm_m is not None and pivot_arm > wc.max_pivot_arm_m:
        return (
            None,
            f"pivot_risk: the part's weight acts {pivot_arm * 1e3:.1f} mm off the jaw axis (limit {wc.max_pivot_arm_m * 1e3:.0f} mm)",
        )
    base_inv = inv_T(wc.T_frame_base)
    if budget is not None:
        budget["evals"] = budget.get("evals", 0) + 1
    rep = evaluate_joint_path(
        [base_inv @ T for T in tcp],
        wc.ik,
        q_seed=q_seed,
        labels=labels,
        jump_rad=wc.jump_rad,
        min_limit_margin_rad=wc.min_limit_margin_rad,
    )
    if not rep.feasible:
        return None, rep.reason.split(":")[0] + ": " + rep.reason.split(":", 1)[-1].strip()[:160]
    return Transfer(grasp, T_from, T_to, tcp, labels, rep, widths, pivot_arm), ""


#: Rotation between the carried waypoints handed to ``move_waypoints``: the
#: backend decides the path between waypoints, so the turn is spelled out in steps
#: close to the ones the joint-path check sampled.
WAYPOINT_ROT_STEP_RAD = math.radians(15.0)
WAYPOINT_CART_STEP_M = 0.05
#: Resting-pose check after each transfer (independent observation vs the target).
VERIFY_TRANSLATION_M = 0.005
VERIFY_ROTATION_RAD = math.radians(10.0)


def _transfer_steps(
    tr: Transfer, wc: Workcell, spec: ObjectSpec, idx: int, expect_state: str | None
) -> list[dict[str, Any]]:
    """Contract-v1 primitive requests for one transfer: ``grasp`` (pregrasp, grasp,
    lift), ``move_waypoints`` carrying the part with its ``T_TCP_object`` (turn,
    carry, lower), ``set_gripper`` release, ``move_pose`` retreat, ``observe_objects``."""
    g = tr.grasp
    carried, _ = densify(
        tr.waypoints[2:-1], tr.labels[2:-1], WAYPOINT_CART_STEP_M, WAYPOINT_ROT_STEP_RAD
    )
    carried = carried[1:]
    if len(carried) > 32:
        idxs = np.linspace(0, len(carried) - 1, 32).round().astype(int)
        carried = [carried[i] for i in idxs]
    held_ref = {
        "object_id": g.object_id,
        "T_TCP_object": T_to_pose(to_contract_frame(g.T_tcp_object, spec.object_class), "tcp"),
    }
    return [
        {
            "capability": "grasp",
            "phase": "grasp",
            "transfer": idx,
            "request": to_grasp_request(g, wc.limits),
            "joint_path": tr.path.summary(),
            "q_path": tr.path.q_path,
            "preconditions": ["fresh non-privileged 6D observation of the part", "gripper empty"],
        },
        {
            "capability": "move_waypoints",
            "phase": "transport",
            "transfer": idx,
            "request": {
                "waypoints": [T_to_pose(T, wc.frame_id) for T in carried],
                "linear": False,
                "held": held_ref,
            },
            "note": "segments between waypoints are the backend's; the joint-path check assumed "
            "straight / geodesic TCP legs (a joint-space executor would follow q_path exactly)",
        },
        {
            "capability": "set_gripper",
            "phase": "release",
            "transfer": idx,
            "request": {
                "width_m": round(g.open_width_m, 5),
                "purpose": "release",
                "target_id": g.object_id,
            },
        },
        {
            "capability": "move_pose",
            "phase": "retreat",
            "transfer": idx,
            "request": {"target_pose": T_to_pose(tr.waypoints[-1], wc.frame_id)},
        },
        {
            "capability": "observe_objects",
            "phase": "verify",
            "transfer": idx,
            "request": {"object_ids": [g.object_id], "require_orientation": True},
            "expect": {
                "pose": T_to_pose(to_contract_frame(tr.T_to, spec.object_class), wc.frame_id),
                "state_label": expect_state,
                "tolerance": {
                    "translation_m": VERIFY_TRANSLATION_M,
                    "rotation_rad": VERIFY_ROTATION_RAD,
                },
                "symmetric_equivalents_accepted": True,
            },
        },
    ]


def _state_name(spec: ObjectSpec, T: np.ndarray, table_z: float) -> str | None:
    if spec.object_class == "domino":
        return domino_state(T, spec, table_z=table_z)["state"]
    return None


def plan_reorient_or_regrasp(
    obs: ObjectObservation,
    spec: ObjectSpec,
    T_goal: np.ndarray,
    wc: Workcell,
    now_s: float,
    max_age_s: float = 1.0,
    allow_privileged: bool = False,
    tilts_deg: tuple[float, ...] = TILTS_DEG,
    max_path_evals: int = 120,
    max_grasps: int = 24,
    staging_yaws_deg: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0),
    allow_staging: bool = True,
) -> dict[str, Any]:
    """Plan from an observed resting part to ``T_goal`` (frame ``wc.frame_id``).

    ``succeeded``: a plan passed the kinematic model (not a physical result);
    ``failed``: the search ran and found none within the budget (with the reasons);
    ``rejected``: the observation cannot be acted on (stale, unseen, ambiguous,
    privileged where not allowed, not resting)."""
    t0 = time.perf_counter()
    stage = "plan_reorient_or_regrasp"
    ok, why, ftype = obs.check_usable(now_s, max_age_s, allow_privileged)
    if not ok:
        return result(
            "rejected",
            stage,
            failure_type=ftype,
            reason=why,
            recovery_hints=["reobserve"],
            verification="none",
        )
    if obs.frame_id != wc.frame_id:
        return result(
            "rejected",
            stage,
            failure_type="invalid_request",
            reason=f"observation in {obs.frame_id}, workcell in {wc.frame_id}",
            verification="none",
        )
    assert obs.T_frame_object is not None
    T0 = obs.T_frame_object
    start_state = _state_name(spec, T0, wc.table_z)
    if start_state == "uncertain":
        return result(
            "rejected",
            stage,
            failure_type="perception.uncertain_state",
            reason=obs.attributes.get("domino", {}).get("reason", "resting state uncertain"),
            recovery_hints=["reobserve"],
            verification="none",
        )
    goals = _goal_equivalents(T_goal, spec)
    goal_state = _state_name(spec, T_goal, wc.table_z)
    budget = {"evals": 0}
    reasons: dict[str, int] = {}

    def note(r: str) -> None:
        key = r.split(" ")[0].split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1

    def grasps_at(T: np.ndarray) -> list[GraspCandidate]:
        cs = [
            c
            for c in propose_grasps(
                obs.object_id,
                T,
                spec,
                wc.frame_id,
                wc.table_z,
                wc.gripper,
                wc.limits,
                wc.obstacles,
                tilts_deg,
            )
            if c.feasible
        ]
        return cs[:max_grasps]

    def search(
        T_from: np.ndarray, targets: list[np.ndarray], grasps: list[GraspCandidate], q_seed
    ) -> Transfer | None:
        pairs = sorted(
            itertools.product(grasps, targets),
            key=lambda gt: (_wrist_turn(gt[0], T_from, gt[1]), -gt[0].score),
        )
        for g, Tt in pairs:
            if budget["evals"] >= max_path_evals:
                return None
            tr, r = plan_transfer(T_from, Tt, g, spec, wc, q_seed, budget)
            if tr is not None:
                return tr
            note(r)
        return None

    start_grasps = grasps_at(T0)
    if not start_grasps:
        return result(
            "failed",
            stage,
            failure_type="perception.no_feasible_grasp",
            reason="no grasp candidate passes the jaw, contact and clearance checks where the part lies",
            evidence={"start_state": start_state},
            recovery_hints=["reobserve", "human"],
            verification="geometry",
        )
    direct = search(T0, goals, start_grasps, wc.q_current)
    direct_evals = budget["evals"]
    plan: list[Transfer] = []
    plan_type = None
    if direct is not None:
        plan, plan_type = [direct], "direct_wrist_turn"
    elif allow_staging:
        for Ts in _staging_poses(spec, T0, T_goal, wc, staging_yaws_deg):
            if budget["evals"] >= max_path_evals:
                break
            # Second leg first: if the part cannot go from the staging pose to the goal, skip it.
            t2 = search(Ts, goals, grasps_at(Ts), None)
            if t2 is None:
                continue
            t1 = search(T0, [Ts], start_grasps, wc.q_current)
            if t1 is None:
                continue
            plan, plan_type = [t1, t2], "staging_regrasp"
            break
    evidence = {
        "start_state": start_state,
        "goal_state": goal_state,
        "start_pose": T_to_pose(to_contract_frame(T0, spec.object_class), wc.frame_id),
        "goal_pose": T_to_pose(to_contract_frame(T_goal, spec.object_class), wc.frame_id),
        "observation": {
            "provider": obs.provider,
            "privileged": obs.privileged,
            "timestamp_s": obs.timestamp_s,
        },
        "path_evals": budget["evals"],
        "direct_evals": direct_evals,
        "rejections": reasons,
        "planning_s": round(time.perf_counter() - t0, 3),
        "kinematic_model": "i2rt YAM chain (checking-dual-arm-clearance) + DLS IK"
        if isinstance(wc.ik, YamIk)
        else wc.ik.__class__.__name__,
    }
    if not plan:
        return result(
            "failed",
            stage,
            failure_type="unreachable",
            reason="no direct turn and no staging regrasp passed the whole-joint-path check within the budget "
            "(add staging spots nearer the arm's dexterous region, use the other arm, or raise max_path_evals)",
            evidence=evidence,
            recovery_hints=["replan", "human"],
            verification="kinematic_model",
        )
    steps: list[dict[str, Any]] = []
    held = []
    for i, tr in enumerate(plan):
        steps += _transfer_steps(tr, wc, spec, i, _state_name(spec, tr.T_to, wc.table_z))
        held.append(prior_from_grasp(tr.grasp, spec, wc.gripper_id, obs.timestamp_s, obs).to_dict())
    if plan_type == "staging_regrasp":
        steps[len(steps) // 2 - 1]["note"] = (
            "replan the second transfer from this observation; the planned one is provisional"
        )
    out = {
        "plan_type": plan_type,
        "direct_rejected": None
        if plan_type == "direct_wrist_turn"
        else {"evals": direct_evals, "reasons": dict(reasons)},
        "transfers": [
            {
                "from": T_to_pose(to_contract_frame(tr.T_from, spec.object_class), wc.frame_id),
                "to": T_to_pose(to_contract_frame(tr.T_to, spec.object_class), wc.frame_id),
                "to_state": _state_name(spec, tr.T_to, wc.table_z),
                "grasp": tr.grasp.to_dict(),
                "joint_path": tr.path.summary(),
                "wrist_turn_deg": round(math.degrees(_wrist_turn(tr.grasp, tr.T_from, tr.T_to)), 2),
                "pivot_arm_m": round(tr.pivot_arm_m, 5),
            }
            for tr in plan
        ],
        "steps": steps,
        "held_priors": held,
    }
    return result(
        "succeeded",
        stage,
        reason=f"{plan_type}: kinematic model only, not simulated or executed",
        outputs=out,
        evidence=evidence,
        verification="kinematic_model",
    )


def _wrist_turn(g: GraspCandidate, T_from: np.ndarray, T_to: np.ndarray) -> float:
    """How far the hand turns between pick and place [rad], plus how far it tilts
    from straight-down at either end (the YAM's awkward direction)."""
    Ra = (T_from @ g.T_object_tcp)[:3, :3]
    Rb = (T_to @ g.T_object_tcp)[:3, :3]
    down = np.array([0.0, 0.0, -1.0])
    tilt = max(
        math.acos(np.clip(Ra[:, 2] @ down, -1, 1)), math.acos(np.clip(Rb[:, 2] @ down, -1, 1))
    )
    return rotation_angle(Ra, Rb) + 0.5 * tilt


def _staging_poses(
    spec: ObjectSpec, T0: np.ndarray, T_goal: np.ndarray, wc: Workcell, yaws_deg
) -> list[np.ndarray]:
    """Resting poses at the staging spots (default: where the part lies and where
    it goes), every face down, a few yaws; ordered so that the larger of the two
    turns (start to staging, staging to goal) is smallest."""
    spots = list(wc.staging_xy) or [tuple(T0[:2, 3]), tuple(T_goal[:2, 3])]
    syms = symmetry_rotations(spec)
    cands = []
    for xy in spots:
        for axis, sign in itertools.product(range(3), (1.0, -1.0)):
            for yaw in yaws_deg:
                Ts = _resting_pose(spec, axis, sign, math.radians(yaw), xy, wc.table_z)
                if any(np.linalg.norm(Ts[:3, 3] - o[:3, 3]) < 0.02 for o in (T0,)) and _same(
                    Ts, T0, syms
                ):
                    continue
                a1 = min(rotation_angle(T0[:3, :3], Ts[:3, :3] @ S) for S in syms)
                a2 = min(rotation_angle(Ts[:3, :3], T_goal[:3, :3] @ S) for S in syms)
                cands.append((max(a1, a2), a1 + a2, Ts))
    cands.sort(key=lambda c: (round(c[0], 3), c[1]))
    return [c[2] for c in cands]


def _same(Ta: np.ndarray, Tb: np.ndarray, syms) -> bool:
    return min(rotation_angle(Ta[:3, :3], Tb[:3, :3] @ S) for S in syms) < math.radians(5)
