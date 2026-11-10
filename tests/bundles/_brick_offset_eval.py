"""Closed-loop accuracy of correcting-brick-offset-from-stud-circles against a one-shot read.

    python -m tests.bundles._brick_offset_eval --trials 20 [--system lego] [--seed 0]

Per trial: a 2 x 4 brick on a plate at a random yaw, the hand top-down over it 30 mm above the studs
but up to 3 mm / 4 deg off, the expected pair pose up to 1.5 mm per axis / 3 deg off (with the arm's
1.1 mm reporting bias, inside the 0.4-pitch = 3.2 mm prior gate). Arms (tests/bundles/_brick_offset_sim.py):

- ideal: moves land where commanded.
- yam-like: each move lands 10 % long, plus 0.1 mm (sd) and 0.1 deg (sd) of noise; the arm reports
  its pose 1 mm / -0.5 mm off (a kinematic bias common to the tool and the camera).
- hand-eye: yam-like, and the wrist camera is really 0.5 mm / 0.3 deg off its nominal mount.

Methods:

- one-shot (hub): read_studs (calibrating-baseplate-stud-grid) on the peek frame with the brick's
  mask, the pair taken from its top pose (the nearer of the two 180-deg readings), one move.
- one-shot (this detector): the skill with max_iterations 2 (measure, move, measure).
- closed loop: the skill with its defaults.
- closed loop + tool_center_px: the pixel and its range measured once with the hand truly over a
  pair (the paper's calibration; the skill's pair_px and pair_range_m), then the defaults.

Reported: the true lateral distance of the tool axis from the pair midpoint and the yaw error after
the method, median / p90 / max, and the routes and moves.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter

import numpy as np
from gap.testing import FakeContext

from . import _brick_offset_sim as S


def _load(name, script):
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parents[2] / "skills" / name / "scripts" / script
    spec = importlib.util.spec_from_file_location(script[:-3], p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ctx(arm):
    return FakeContext(
        {"robot.get_observation": arm.obs, "robot.get_ee_pose": arm.ee, "robot.go_to_pose": arm.go}
    )


ARMS = {
    "ideal": {},
    "yam-like": {
        "land_frac": 0.1,
        "land_sigma": 0.0001,
        "land_yaw_deg": 0.1,
        "read_bias": (0.001, -0.0005, 0.0),
    },
    "hand-eye": {
        "land_frac": 0.1,
        "land_sigma": 0.0001,
        "land_yaw_deg": 0.1,
        "read_bias": (0.001, -0.0005, 0.0),
        "hand_eye_error": S.small(0.0005, -0.0003, 0.0002, 0.2, -0.2, 0.1),
    },
}


def one_shot_hub(rs, sc, arm, guess):
    """read_studs on the peek frame, then one move over the pair it implies."""
    obs = arm.obs()
    frame = obs["cameras"][0]
    _, _, ids = S._render(
        id(sc), (arm.true @ arm.cam_true).round(7).tobytes(), (arm.frames - 1) % 7
    )
    out = rs.run(FakeContext({}), mask=ids == 2, frame=frame, system=sc.system)
    if out["route"] not in ("lattice_found", "occluded") or not out["top_pose"]:
        return out["route"], 0
    top = S.mat(out["top_pose"])
    p = sc.g["pitch"]
    g = S.mat(guess)
    cands = [top @ S.small(0.0, s * 0.5 * p) for s in (1.0, -1.0)]
    pair = min(cands, key=lambda m: np.linalg.norm(m[:3, 3] - g[:3, 3]))
    tool = S.mat(arm.ee()["pose"])
    yaw_tool = math.atan2(tool[:3, 0] @ pair[:3, 1], tool[:3, 0] @ pair[:3, 0])
    turn = round(yaw_tool / (math.pi / 2)) * (math.pi / 2) - yaw_tool
    want = np.eye(4)
    want[:3, :3] = S.small(rz=math.degrees(turn))[:3, :3] @ tool[:3, :3]
    want[:3, 3] = pair[:3, 3] + (sc.g["stud_h"] + 0.03) * np.array([0.0, 0.0, 1.0])
    arm.go(S.se3(want))
    return "moved", 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--system", default="lego")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="ideal,yam-like,hand-eye")
    a = ap.parse_args()
    co = _load("correcting-brick-offset-from-stud-circles", "correct_offset.py")
    rs = _load("calibrating-baseplate-stud-grid", "read_studs.py")
    p = S.R.SYSTEMS[a.system]["pitch"]
    brick = (5, 6, 2, 4) if a.system == "lego" else (3, 3, 2, 4)
    t0 = time.time()
    for arm_name in a.arms.split(","):
        kw = ARMS[arm_name]
        # tool_center_px: the hand truly over a pair at the peek height, the skill's pair pixel.
        cal_sc = S.Scene(a.system, brick=brick, yaw_deg=0.0)
        cal_arm = S.Arm(
            cal_sc, S.start_over(cal_sc), **{k: v for k, v in kw.items() if k != "land_frac"}
        )
        cal = co.run(
            _ctx(cal_arm), S.guess(cal_sc), system=a.system, max_iterations=1, move_to_peek=False
        )
        px, px_range = cal["pair_px"], cal["pair_range_m"]
        rows: dict[str, list] = {
            "one-shot (hub read_studs)": [],
            "one-shot (this detector)": [],
            "closed loop": [],
            "closed loop + tool_center_px": [],
        }
        rng = np.random.default_rng(a.seed)
        for t in range(a.trials):
            yaw = float(rng.uniform(-40, 40))
            sx, sy = rng.uniform(-0.003, 0.003, 2) * p / 0.008
            syaw = float(rng.uniform(-4, 4))
            gx, gy = rng.uniform(-0.0015, 0.0015, 2) * p / 0.008
            gyaw = float(rng.uniform(-3, 3))
            for meth, res in rows.items():
                sc = S.Scene(a.system, brick=brick, yaw_deg=yaw)
                arm = S.Arm(sc, S.start_over(sc, sx, sy, syaw), seed=a.seed * 1000 + t, **kw)
                guess = S.guess(sc, gx, gy, gyaw)
                if meth.startswith("one-shot (hub"):
                    route, moves = one_shot_hub(rs, sc, arm, guess)
                else:
                    extra = {}
                    if meth == "one-shot (this detector)":
                        extra = {"max_iterations": 2, "move_to_peek": False}
                    elif meth.endswith("tool_center_px"):
                        extra = {"tool_center_px": px, "tool_center_range": px_range}
                    if meth != "one-shot (this detector)":
                        extra["move_to_peek"] = False
                    out = co.run(_ctx(arm), guess, system=a.system, **extra)
                    route, moves = out["route"], out["moves"]
                lat, dyaw = arm.error()
                res.append((lat, dyaw, route, moves))
        print(
            f"\n== {a.system}, arm {arm_name}, {a.trials} trials (tool_center_px {np.round(px, 1).tolist()})"
        )
        print(
            "| method | lateral mm (median / p90 / max) | yaw deg (median / p90 / max) | routes | moves (mean) |"
        )
        print("|---|---|---|---|---|")
        for meth, res in rows.items():
            lat = np.array([r[0] for r in res])
            yw = np.array([r[1] for r in res])
            routes = Counter(r[2] for r in res)

            def q(v):
                return f"{np.median(v):.2f} / {np.percentile(v, 90):.2f} / {v.max():.2f}"

            print(
                f"| {meth} | {q(lat)} | {q(yw)} | "
                f"{', '.join(f'{k} {n}' for k, n in sorted(routes.items()))} | "
                f"{np.mean([r[3] for r in res]):.1f} |"
            )
    print(f"\n({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
