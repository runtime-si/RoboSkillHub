"""Isaac Sim 5.1 + BrickSim episodes for the LEGO evaluation (privileged waypoint baseline, injected faults).

Runs only in the RoCo/BrickSim venv on a GPU host (examples/tasks/assembly_eval/lego_run_autodl.sh)::

    OMNI_KIT_ACCEPT_EULA=YES python -m roboskillhub.tasks.assembly_eval.lego_isaac_runner \\
        --roco ~/lego_isaac/src/RoCo_BrickAssembly --scene single_on_brick --policy roco_naive \\
        --seeds 0 --out runs/single_naive

The scene, robot and controller are RoCo_BrickAssembly@8e7ad4da's (DexMate Vega U, joint-position FSM); the
policy is either RoCo's NaivePolicy unchanged (``roco_naive``) or ``goal_fsm``, the same FSM aimed at the goal
pose of each brick (computed from the target topology) instead of at the parent brick's origin, one placement
per brick. Both read exact brick poses from the simulator and are reported as privileged. Faults change the
goal_fsm targets of one placement. The runner never creates connections or moves bricks; engagement is
whatever BrickSim's contact trigger produces. Verdicts come from lego_postconditions on recorded snapshots.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

from .lego_episode import (
    G_VERSION,
    K_VERSION,
    OBSERVATION_MODE,
    canonical_hash,
    file_hash,
    load_config,
    summary_row,
)
from .lego_faults import FAULTS, Perturbation, perturbation
from .lego_postconditions import evaluate_episode
from .lego_scenes import SCENES, scene_hash, user_config, write_task_dir

JOINT_ORDER = [
    "Lift",
    "torso_flip",
    "L_arm_j1",
    "L_arm_j2",
    "L_arm_j3",
    "L_arm_j4",
    "L_arm_j5",
    "L_arm_j6",
    "L_arm_j7",
    "L_gripper_joint",
    "L_gripper_joint_01",
    "R_arm_j1",
    "R_arm_j2",
    "R_arm_j3",
    "R_arm_j4",
    "R_arm_j5",
    "R_arm_j6",
    "R_arm_j7",
    "R_gripper_joint",
    "R_gripper_joint_01",
    "head_j1",
    "head_j2",
    "head_j3",
]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--roco", type=Path, required=True, help="RoCo_BrickAssembly checkout at 8e7ad4da"
    )
    ap.add_argument("--scene", choices=sorted(SCENES))
    ap.add_argument("--policy", choices=["roco_naive", "goal_fsm"], default="goal_fsm")
    ap.add_argument("--fault", choices=["none", *sorted(FAULTS)], default="none")
    ap.add_argument(
        "--breakage",
        choices=["roco", "on"],
        default="roco",
        help="roco: as shipped (BrickSim LegoWorld disables breakage); on: BrickSim's default thresholds enabled",
    )
    ap.add_argument(
        "--cases",
        nargs="+",
        default=[],
        help="several scene/policy/fault/breakage cases in one Kit session, e.g. single_on_plate/goal_fsm/none/roco",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--no-probe", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    """Start Kit headless the way the ``bricksim`` launcher does and run ``kit_main`` inside it.

    RoCo's Env initialises World inside a coroutine, which needs the Kit app (``isaacsim.main``); under
    ``SimulationApp`` the same calls crash in World.initialize_simulation_context_async (Isaac Sim 5.1.0).
    """
    import importlib.util
    import os
    import shlex

    argv = sys.argv[1:] if argv is None else argv
    parse_args(argv)
    os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")
    entry = Path(importlib.util.find_spec("bricksim").origin).with_name("_entry.py")
    target = f"{__spec__.name}:kit_main"
    sys.argv = [
        sys.argv[0],
        "isaacsim.exp.full",
        "--no-window",
        "--/app/content/emptyStageOnStart=false",
        "--/crashreporter/enabled=false",
        "--/log/outputStreamLevel=info",
        "--/log/channels/*=warn",
        "--/log/channels/bricksim=info",
        "--exec",
        " ".join(shlex.quote(a) for a in [str(entry), target, *argv]),
    ]
    import isaacsim

    isaacsim.main()


async def kit_main() -> None:
    import os

    rc = 1
    try:
        rc = await run_all(parse_args(sys.argv[1:]))
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        # Kit's shutdown after a BrickSim session can hang for minutes; everything is written by now.
        os._exit(rc)


async def run_all(args) -> int:
    sys.path.insert(0, str(args.roco / "src"))
    cfg = load_config(args.config) if args.config else load_config()
    if args.no_probe:
        cfg["postconditions"]["probe"]["enabled"] = False
    args.out.mkdir(parents=True, exist_ok=True)
    cases = [c.split("/") for c in args.cases] or [
        [args.scene, args.policy, args.fault, args.breakage]
    ]
    for scene, policy, fault, breakage in cases:
        if (
            scene not in SCENES
            or policy not in ("roco_naive", "goal_fsm")
            or breakage not in ("roco", "on")
        ):
            raise SystemExit(f"bad case {scene}/{policy}/{fault}/{breakage}")
        if fault != "none" and scene not in FAULTS[fault]["scenes"]:
            raise SystemExit(f"fault {fault} is not defined for scene {scene}")
    for scene, policy, fault, breakage in cases:
        case = argparse.Namespace(
            **{**vars(args), "scene": scene, "policy": policy, "fault": fault, "breakage": breakage}
        )
        for seed in args.seeds:
            try:
                ep = await run_episode(case, cfg, seed)
            except Exception:
                traceback.print_exc()
                print("EPISODE_ERROR", scene, policy, fault, breakage, seed, flush=True)
                continue
            (args.out / f"{ep['episode_id']}.json").write_text(
                json.dumps(ep, indent=1, default=str)
            )
            row = summary_row(ep)
            print("EPISODE", json.dumps(row), flush=True)
            with open(args.out / "summary.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
    return 0


def _pose_of(T) -> dict:
    from scipy.spatial.transform import Rotation as R

    return {
        "p": [float(v) for v in T[:3, 3]],
        "q": [float(v) for v in R.from_matrix(T[:3, :3]).as_quat()],
    }


def _wxyz_pos_to_T(rot, pos):
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    T = np.eye(4)
    T[:3, :3] = R.from_quat([rot[1], rot[2], rot[3], rot[0]]).as_matrix()
    T[:3, 3] = pos
    return T


def _trans(x, y, z):
    import numpy as np

    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def _joint_obs(env):
    import numpy as np

    q_raw = env.robot.get_joint_positions()
    q = np.zeros(env.robot_pin.nq, dtype=np.float32)
    for i, name in enumerate(JOINT_ORDER):
        if name in env.robot.dof_names:
            q[i] = q_raw[env.robot.dof_names.index(name)]
    return {"joint_positions": q, "images": {}}


def _ik_error(env, q, ee, T_target) -> dict:
    import numpy as np

    T = env.robot_pin.FK(q, [ee])[ee]
    R = T[:3, :3].T @ T_target[:3, :3]
    ang = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    return {"pos_m": float(np.linalg.norm(T[:3, 3] - T_target[:3, 3])), "rot_rad": ang}


def make_policy(NaivePolicy):
    from bricksim.topology.ordering import bfs_sort_connections
    from rocobrick.utils import rot_z, trans_z, world_T_to_robot_T

    class EvalPolicy(NaivePolicy):
        """RoCo NaivePolicy, optionally re-aimed at goal poses (goal_fsm) with one perturbed placement."""

        def __init__(self, env, cfg, mode, goal_T_W, path_to_id, part_lw, pert, fault_index):
            self.cfg, self.mode, self.goal_T_W, self.path_to_id = cfg, mode, goal_T_W, path_to_id
            self.part_lw = part_lw
            self.goal_turns = {}
            self.plan_ctx = {}
            self.reach = {}
            self.infeasible = {}
            self.pert, self.fault_index = pert, fault_index
            self.ik_s = 0.0
            self.ik_report = {}
            super().__init__(env)

        def plan_assembly_sequence(self, topology, pre_placed_parts, to_place_placed):
            if self.mode == "roco_naive":
                plan = super().plan_assembly_sequence(topology, pre_placed_parts, to_place_placed)
                for e in plan:
                    e["part"] = self.path_to_id[e["hole_path"]]
                return plan
            plan, seen = [], set()
            for conn in bfs_sort_connections(topology)["connections"]:
                h = conn["hole_id"]
                if h in pre_placed_parts or h in seen:
                    continue
                seen.add(h)
                plan.append({"part": str(h), "hole_path": to_place_placed[h]})
            return [e for e in plan if e["part"] not in self.pert.skip_parts]

        def plan_goal_sequence(self, plan, task_id):
            t0 = time.perf_counter()
            if self.mode == "roco_naive":
                super().plan_goal_sequence(plan, task_id)
            else:
                self._plan_goal(plan, task_id)
            self.ik_s += time.perf_counter() - t0

        def _place_targets(self, task_id, target_T, T_tcp_brick):
            """TCP targets (preplace, place, retreat) for a brick target pose and the brick pose in the TCP frame."""
            import numpy as np

            b, u = self.cfg["baseline"], self.env.brick_unit_height
            pert = self.pert if task_id == self.fault_index else Perturbation()
            hold_inv = np.linalg.inv(T_tcp_brick)
            preplace = target_T @ trans_z(b["preplace_z_units"] * u) @ hold_inv
            if pert.press:
                place = target_T @ trans_z(-b["press_depth_units"] * u) @ hold_inv
            else:
                place = target_T @ trans_z(pert.hover_above_seat_m) @ hold_inv
            if pert.retreat_sweep_towards is not None:
                # Horizontal sweep in the robot frame towards the neighbour's goal position.
                towards = world_T_to_robot_T(
                    self.goal_T_W[pert.retreat_sweep_towards], self.env.robot_pin.BASE_T
                )
                d = towards[:3, 3] - target_T[:3, 3]
                d[2] = 0.0
                d *= pert.retreat_sweep_distance_m / max(float(np.linalg.norm(d)), 1e-9)
                retreat = target_T @ trans_z(pert.retreat_sweep_height_units * u) @ hold_inv
                retreat[:3, 3] += d
            else:
                retreat = preplace
            return preplace, place, retreat

        def _plan_goal(self, plan, task_id):
            env, b = self.env, self.cfg["baseline"]
            u = env.brick_unit_height
            pert = self.pert if task_id == self.fault_index else Perturbation()
            task = plan[task_id]
            ik, joints, home = (
                env.robot_pin.IK,
                env.robot_pin.controllable_joints,
                env.robot_pin.home_q,
            )
            grab_T = env.get_prim_robot_T(task["hole_path"])
            goal_T = world_T_to_robot_T(self.goal_T_W[task["part"]], env.robot_pin.BASE_T)
            seat = b["grasp_z_units"] * u
            pregrasp_T = grab_T @ trans_z(b["pregrasp_z_units"] * u)
            grasp_T = grab_T @ trans_z(seat)
            nominal_hold = trans_z(-seat)
            # Arm and goal orientation: both the grasp and the place must be reachable. A 2xN brick looks the same
            # turned half a turn (a square one a quarter turn); a turned goal is used only if it is clearly better,
            # since any in-hand offset is doubled by the turn.
            L, W = self.part_lw[task["part"]]
            turns = (0, 1, 2, 3) if L == W else (0, 2)
            best, tried = None, []
            r_m, r_rad = b["reach_tol_m"], b["reach_tol_rad"]
            for ee in ("tip_l", "tip_r"):
                qg, _ = ik({ee: grasp_T}, home.copy(), joints)
                eg = _ik_error(env, qg, ee, grasp_T)
                for k in turns:
                    tgt = (
                        goal_T
                        @ rot_z(k * math.pi / 2)
                        @ _trans(*pert.place_offset_m)
                        @ rot_z(pert.place_yaw_rad)
                    )
                    pre, place, _ = self._place_targets(task_id, tgt, nominal_hold)
                    qp, _ = ik({ee: pre}, qg.copy(), joints)
                    qpl, _ = ik({ee: place}, qp.copy(), joints)
                    errs = [eg, _ik_error(env, qp, ee, pre), _ik_error(env, qpl, ee, place)]
                    ok = all(e["pos_m"] <= r_m and e["rot_rad"] <= r_rad for e in errs)
                    score = sum(e["pos_m"] + 0.05 * e["rot_rad"] for e in errs) + (
                        0.002 if k else 0.0
                    )
                    tried.append(
                        {
                            "ee": ee,
                            "turn": k,
                            "reachable": ok,
                            "pos_m": [e["pos_m"] for e in errs],
                            "rot_rad": [e["rot_rad"] for e in errs],
                        }
                    )
                    if ok and (best is None or score < best[0]):
                        best = (score, ee, k, tgt)
            self.reach[task_id] = tried
            if best is None:
                self.infeasible[task_id] = (
                    "no arm reaches both the grasp and the place pose (IK residual > tolerance)"
                )
                best = (None, "tip_l", 0, goal_T)
            _, ee, k, target_T = best
            self.goal_turns[task_id] = k
            self.plan_ctx[task_id] = {
                "ee": ee,
                "target_T": target_T,
                "hole_path": task["hole_path"],
            }
            sd = {
                "ee_name": ee,
                "to_brick": None,
                "grab_brick": task["hole_path"],
                "home": home.copy(),
                "back_home": home.copy(),
            }
            sd["pregrasp"], _ = ik({ee: pregrasp_T}, sd["home"].copy(), joints)
            sd["grasp"], _ = ik({ee: grasp_T}, sd["pregrasp"].copy(), joints)
            sd["close_gripper"] = self.close_gripper(sd["grasp"], ee)
            sd["retreat_grasp"], _ = ik({ee: pregrasp_T}, sd["close_gripper"].copy(), joints)
            self.state_dicts[task_id] = sd
            self._fill_place(task_id, nominal_hold)
            self.ik_report[task_id].update(
                pregrasp=_ik_error(env, sd["pregrasp"], ee, pregrasp_T),
                grasp=_ik_error(env, sd["grasp"], ee, grasp_T),
            )

        def _fill_place(self, task_id, hold):
            env = self.env
            ik, joints = env.robot_pin.IK, env.robot_pin.controllable_joints
            pert = self.pert if task_id == self.fault_index else Perturbation()
            sd, ctx = self.state_dicts[task_id], self.plan_ctx[task_id]
            ee = ctx["ee"]
            pre, place, retreat = self._place_targets(task_id, ctx["target_T"], hold)
            sd["preplace"], _ = ik({ee: pre}, sd["retreat_grasp"].copy(), joints)
            sd["place"], _ = ik({ee: place}, sd["preplace"].copy(), joints)
            sd["open_gripper"] = (
                self.open_gripper(sd["place"], ee) if pert.open_on_release else sd["place"].copy()
            )
            sd["retreat_place"], _ = ik({ee: retreat}, sd["open_gripper"].copy(), joints)
            rep = self.ik_report.setdefault(task_id, {"ee": ee})
            rep.update(
                preplace=_ik_error(env, sd["preplace"], ee, pre),
                place=_ik_error(env, sd["place"], ee, place),
                retreat_place=_ik_error(env, sd["retreat_place"], ee, retreat),
            )

        def replan_with_held_pose(self, task_id):
            """After lifting: re-aim the place waypoints with the brick's measured pose in the gripper."""
            import numpy as np

            if self.mode != "goal_fsm" or task_id not in self.plan_ctx:
                return None
            t0 = time.perf_counter()
            env, ctx = self.env, self.plan_ctx[task_id]
            T_tcp = env.robot_pin.FK(self.cur_q, [ctx["ee"]])[ctx["ee"]]
            hold = np.linalg.inv(T_tcp) @ env.get_prim_robot_T(ctx["hole_path"])
            self._fill_place(task_id, hold)
            self.ik_s += time.perf_counter() - t0
            seat = self.cfg["baseline"]["grasp_z_units"] * env.brick_unit_height
            d = hold[:3, 3] - np.array([0.0, 0.0, -seat])
            return {
                "held_offset_m": [float(v) for v in d],
                "held_offset_norm_m": float(np.linalg.norm(d)),
            }

    return EvalPolicy


class Recorder:
    def __init__(self, env, ids: dict[str, str]):
        import bricksim.core as bc

        self.env, self.ids, self.bc = env, ids, bc
        self.path_to_id = {v: k for k, v in ids.items()}
        self.step = 0
        self.events: list[dict] = []
        self.traj: list[dict] = []
        self.max_force: dict[str, float] = {}

    def drain(self, phase: str, subtask: int | None):
        for kind, fn in (
            ("assembled", self.bc.get_assembled_connections),
            ("disassembled", self.bc.get_disassembled_connections),
        ):
            for ci in fn(True):
                self.events.append(
                    {
                        "kind": kind,
                        "step": self.step,
                        "phase": phase,
                        "subtask": subtask,
                        "stud": self.path_to_id.get(ci.stud_path, ci.stud_path),
                        "hole": self.path_to_id.get(ci.hole_path, ci.hole_path),
                        "offset": list(ci.offset),
                        "yaw": int(ci.yaw),
                    }
                )
        for d in self.bc.get_assembly_debug_infos():
            hole = self.path_to_id.get(d.hole_path)
            if hole is not None:
                f = -float(d.projected_force)
                if f > self.max_force.get(hole, float("-inf")):
                    self.max_force[hole] = f

    def snap(self) -> dict:
        parts = {pid: _pose_of(self.env.get_prim_world_T(path)) for pid, path in self.ids.items()}
        conns = []
        for s, sp in self.ids.items():
            for h, hp in self.ids.items():
                if s == h:
                    continue
                ci = self.bc.lookup_physics_connection(sp, 1, hp, 0)
                if ci is None:
                    continue
                util = None
                if ci.conn_path:
                    try:
                        util = float(self.bc.get_connection_utilization(ci.conn_path))
                    except Exception:
                        util = None
                conns.append(
                    {
                        "stud": s,
                        "hole": h,
                        "offset": list(ci.offset),
                        "yaw": int(ci.yaw),
                        "utilization": util,
                    }
                )
        q = self.env.robot.get_joint_positions()
        return {"step": self.step, "parts": parts, "conns": conns, "q": [float(v) for v in q]}


async def run_episode(args, cfg, seed: int) -> dict:
    import bricksim.core as bc
    import numpy as np
    import omni.usd
    from rocobrick.env.Env import Env
    from rocobrick.policy.NaivePolicy import Policy as NaivePolicy

    random.seed(seed)
    np.random.seed(seed)
    t_wall = time.perf_counter()
    scene = SCENES[args.scene]
    fault = None if args.fault == "none" else args.fault
    episode_id = f"{args.scene}__{args.policy}__{args.fault}__brk-{args.breakage}__s{seed}"
    work = args.out / "tasks" / episode_id
    task_dir = write_task_dir(args.scene, work / "task")
    ucfg = user_config(task_dir, args.roco.resolve(), scene["task_type"])
    (work / "user_config.json").write_text(json.dumps(ucfg, indent=2))
    sys_cfg = args.roco / "config" / "system_config.json"

    env = Env(
        root_dir=str(args.roco / "run"),
        user_config_path=str((work / "user_config.json").resolve()),
        system_config_path=str(sys_cfg.resolve()),
    )
    await env.reset()
    if args.breakage == "on":
        bt = bc.get_breakage_thresholds()
        bt.enabled = True
        bc.set_breakage_thresholds(bt)
    await env.play()
    await env.get_robot_ready()

    ids: dict[str, str] = {}
    lw: dict[str, list[int]] = {}
    for p in env.topology["parts"]:
        pid = str(p["id"])
        ids[pid] = env.pre_placed_parts.get(p["id"]) or env.to_place_placed.get(p["id"])
        lw[pid] = [int(p["payload"]["L"]), int(p["payload"]["W"])]
    path_to_id = {v: k for k, v in ids.items()}
    roles = {
        pid: (
            "plate"
            if pid == "0"
            else "preplaced"
            if int(pid) in env.pre_placed_parts
            else "to_place"
        )
        for pid in ids
    }

    # Goal poses: plate pose as built times the structure's plate-relative transforms (BrickSim, wxyz).
    T_W_plate = env.get_prim_world_T(ids["0"])
    rel = bc.compute_structure_transforms(env.topology, 0)
    goal_T_W = {str(k): T_W_plate @ _wxyz_pos_to_T(*v) for k, v in rel.items()}
    frame_check = {
        pid: float(np.linalg.norm(goal_T_W[pid][:3, 3] - env.get_prim_world_T(ids[pid])[:3, 3]))
        for pid, r in roles.items()
        if r == "preplaced"
    }

    goal_conns = [
        {
            "stud": str(c["stud_id"]),
            "hole": str(c["hole_id"]),
            "offset": list(c["offset"]),
            "yaw": int(c["yaw"]),
        }
        for c in env.topology["connections"]
    ]
    fault_index = FAULTS[fault]["scenes"].get(args.scene) if fault else None
    if fault and fault_index is None:
        raise SystemExit(f"fault {fault} is not defined for scene {args.scene}")
    pert = Perturbation()
    if fault:
        from bricksim.topology.ordering import bfs_sort_connections

        order = [str(c["hole_id"]) for c in bfs_sort_connections(env.topology)["connections"]]
        to_place = [pid for pid in dict.fromkeys(order) if roles[pid] == "to_place"]
        faulted = to_place[fault_index] if fault_index < len(to_place) else None
        parents = tuple(
            gc["stud"]
            for gc in goal_conns
            if gc["hole"] == faulted and roles[gc["stud"]] == "to_place"
        )
        pert = perturbation(fault, cfg, parents)
        if pert.skip_parts:
            fault_index -= sum(1 for pid in to_place[:fault_index] if pid in pert.skip_parts)

    rec = Recorder(env, ids)
    authored = []
    rec.drain("reset", None)
    authored, rec.events = rec.events, []
    start = rec.snap()

    EvalPolicy = make_policy(NaivePolicy)
    policy = EvalPolicy(env, cfg, args.policy, goal_T_W, path_to_id, lw, pert, fault_index)
    b, pc = cfg["baseline"], cfg["postconditions"]
    probe_cfg = pc["probe"]

    steps_out = []
    hold_q = env.robot_pin.home_q.copy()

    async def advance(phase, subtask, q=None, force_on=None, force_n=0.0):
        if q is not None:
            env.robot_apply_action(q)
        if force_on is not None:
            _apply_up_force(force_on, force_n, env)
        await env.step()
        rec.step += 1
        rec.drain(phase, subtask)
        if rec.step % 10 == 0:
            rec.traj.append({"step": rec.step, "phase": phase, "subtask": subtask})

    for i, entry in enumerate(policy.plan):
        pid = entry["part"]
        st = {
            "part": pid,
            "hole_path": entry["hole_path"],
            "snapshots": {},
            "timed_out": False,
            "faulted": i == fault_index,
        }
        ev0 = len(rec.events)
        rec.max_force.pop(pid, None)
        st["snapshots"]["pick_start"] = rec.snap()
        st["reach"] = policy.reach.get(i)
        if i in policy.infeasible:
            st["skipped"], st["plan_infeasible"] = True, policy.infeasible[i]
            st["events"], st["ee"], st["ik"] = [], None, None
            steps_out.append(st)
            policy.task_id, policy.state = i + 1, "home"
            if policy.task_id < len(policy.plan):
                policy.plan_goal_sequence(policy.plan, policy.task_id)
            continue
        n = 0
        while policy.task_id == i:
            prev = policy.state
            q = policy.get_action(_joint_obs(env))
            await advance(prev, i, q)
            n += 1
            if policy.state != prev:
                if prev == "retreat_grasp":
                    st["snapshots"]["grasped"] = rec.snap()
                    if cfg["baseline"].get("inhand_correction"):
                        st["held"] = policy.replan_with_held_pose(i)
                elif prev == "preplace":
                    st["snapshots"]["before_place"] = rec.snap()
                elif prev == "open_gripper":
                    st["snapshots"]["released"] = rec.snap()
                elif prev == "retreat_place":
                    st["snapshots"]["retreated"] = rec.snap()
            if (
                b.get("stop_on_engagement")
                and policy.mode == "goal_fsm"
                and policy.state == "place"
                and "press_stop_step" not in st
                and any(e["kind"] == "assembled" and e["hole"] == pid for e in rec.events[ev0:])
            ):
                # Guarded press: stop descending once BrickSim reports this brick engaged (privileged signal).
                sd = policy.state_dicts[i]
                sd["place"] = policy.cur_q.copy()
                keep_closed = np.array_equal(sd["open_gripper"], sd["place"])
                sd["open_gripper"] = (
                    sd["place"].copy()
                    if keep_closed
                    else policy.open_gripper(sd["place"], sd["ee_name"])
                )
                st["press_stop_step"] = rec.step
            if n >= b["max_steps_per_brick"]:
                st["timed_out"], st["last_phase"] = True, policy.state
                break
        for _ in range(pc["settle_steps"]):
            await advance("settle", i, hold_q)
        st["snapshots"]["settled"] = rec.snap()
        if probe_cfg["enabled"]:
            try:
                path = ids[pid]
                mass = float(
                    omni.usd.get_context()
                    .get_stage()
                    .GetPrimAtPath(path)
                    .GetAttribute("physics:mass")
                    .Get()
                )
                force = probe_cfg["weight_ratio"] * mass * 9.81
                z0 = env.get_prim_world_T(path)[2, 3]
                rise = 0.0
                for k in range(probe_cfg["steps"] + probe_cfg["observe_steps"]):
                    on = path if k < probe_cfg["steps"] else None
                    await advance("probe", i, hold_q, force_on=on, force_n=force)
                    rise = max(rise, float(env.get_prim_world_T(path)[2, 3] - z0))
                st["probe_max_lift_m"], st["probe_force_n"], st["mass_kg"] = rise, force, mass
                st["snapshots"]["probe_end"] = rec.snap()
            except Exception as e:  # noqa: BLE001
                st["probe_error"] = repr(e)
            for _ in range(pc["settle_steps"]):
                await advance("probe_release", i, hold_q)
            st["snapshots"]["after_probe"] = rec.snap()
        st["events"] = rec.events[ev0:]
        st["max_press_force_n"] = rec.max_force.get(pid)
        st["sim_steps"] = n
        st["ee"] = policy.state_dicts.get(i, {}).get("ee_name")
        st["ik"] = policy.ik_report.get(i)
        st["goal_quarter_turns"] = policy.goal_turns.get(i)
        steps_out.append(st)
        if st["timed_out"]:
            break

    final = rec.snap()
    traj_path = args.out / f"{episode_id}.traj.jsonl"
    with open(traj_path, "w") as f:
        for r in rec.traj:
            f.write(json.dumps(r) + "\n")

    record = {
        "parts": {pid: {"path": ids[pid], "lw": lw[pid], "role": roles[pid]} for pid in ids},
        "goal_poses": {pid: _pose_of(T) for pid, T in goal_T_W.items()},
        "start_poses": start["parts"],
        "goal_connections": goal_conns,
        "steps": steps_out,
        "final_snapshot": final,
    }
    verdict = evaluate_episode(record, cfg)

    assets = args.roco / "robot_assets" / "DexMate"
    thr = bc.get_assembly_thresholds()
    brk = bc.get_breakage_thresholds()
    physics = {
        "assembly": {
            k: getattr(thr, k)
            for k in (
                "enabled",
                "distance_tolerance",
                "max_penetration",
                "z_angle_tolerance",
                "required_force",
                "yaw_tolerance",
                "position_tolerance",
            )
        },
        "breakage": {
            k: getattr(brk, k)
            for k in (
                "enabled",
                "contact_regularization",
                "clutch_axial_compliance",
                "clutch_radial_compliance",
                "clutch_tangential_compliance",
                "friction_coefficient",
                "preloaded_force",
                "breakage_cooldown_time",
            )
        },
        "system_config_sha256": file_hash(sys_cfg),
        "stage": str(omni.usd.get_context().get_stage_url()),
    }
    import importlib.metadata as md

    return {
        "episode_id": episode_id,
        "seed": seed,
        "scene": args.scene,
        "fault": args.fault,
        "breakage": args.breakage,
        "fault_doc": FAULTS[fault]["doc"] if fault else None,
        "expected_failure_types": FAULTS[fault]["expect"] if fault else [None],
        "fault_index": fault_index,
        "perturbation": pert.__dict__,
        "policy": args.policy,
        "privileged": True,
        "observation_mode": OBSERVATION_MODE,
        "K": K_VERSION,
        "G": G_VERSION,
        "versions": {
            p: md.version(p) for p in ("isaacsim", "isaaclab", "bricksim", "torch", "pin", "numpy")
        },
        "hashes": {
            "eval_config": canonical_hash(cfg),
            "scene": scene_hash(args.scene),
            "user_config": canonical_hash(ucfg),
            "physics": canonical_hash(physics),
            "robot_urdf": file_hash(assets / "vega_1u_gripper.urdf"),
            "robot_usd": file_hash(assets / "vega_1u_gripper.usda"),
        },
        "physics": physics,
        "goal": {"scene": scene, "connections": goal_conns},
        "goal_frame_check_m": frame_check,
        "authored_connections_at_reset": authored,
        "record": record,
        "outcome": verdict["outcome"],
        "placements": verdict["placements"],
        "sim_steps": rec.step,
        "planning_s": policy.ik_s,
        "wall_s": time.perf_counter() - t_wall,
        "interventions": 0,
        "recoveries": 0,
        "trajectory_ref": str(traj_path.name),
    }


def _apply_up_force(path: str, force_n: float, env) -> None:
    import carb
    import omni.usd
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools

    T = env.get_prim_world_T(path)
    stage_id = omni.usd.get_context().get_stage_id()
    get_physx_simulation_interface().apply_force_at_pos(
        stage_id,
        PhysicsSchemaTools.sdfPathToInt(path),
        carb.Float3(0.0, 0.0, float(force_n)),
        carb.Float3(*[float(v) for v in T[:3, 3]]),
        "Force",
    )


if __name__ == "__main__":
    main()
