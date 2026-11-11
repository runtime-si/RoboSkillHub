"""One-piece end-to-end jigsaw chain in the YAM twin (harness orchestrator; simulator truth only for scoring,
except where marked TRUTH).

The piece lies on the mat away from its slot, turned; its six neighbours are a fixed board round the slot.
Chain: scan (scanning-the-board-into-a-wrist-mosaic) -> segment from the height map -> side types
(classifying-piece-sides-for-border-first) -> yaw from the side types against the slot's -> grip choice
(selecting-a-grip-clear-of-placed-neighbours, real 15.5 x 3.9 mm blades) -> pick by the harness handle ->
turn (rotating-a-held-piece-within-wrist-limits, drop spot chosen) -> seat (seating-a-piece-by-wiggle-search)
-> flush check (verifying-piece-flushness-from-wrist-depth). Each stage prints a SKILLCHECK e2e line.
"""

import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from gap import NodeContext

HERE = Path(os.environ.get("GAP_SKILLS_PATH", "."))
TIP_REACH = 0.0102
SEAT_VARIANTS = {
    "A": {},
    "B": {"search_radius": 0.004, "yaw_range_deg": 8.0},
    "C": {"approach": "offset_push"},
}
META = json.loads((HERE / os.environ.get("ASSETS", "assets_c03") / "meta.json").read_text())


class Output(TypedDict):
    route: str


def _skill(name, script):
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"), HERE / "skills" / name / "scripts" / script
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _rot(q):
    w, x, y, z = (float(q[k]) for k in "wxyz")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _pose(r):
    return r.get("pose", r)


def _log(stage, **kw):
    print(
        "SKILLCHECK e2e "
        + json.dumps(
            {"stage": stage, **kw},
            default=lambda o: round(float(o), 5) if isinstance(o, (np.floating, float)) else str(o),
        ),
        flush=True,
    )


def _wait(ctx, n=40):
    try:
        ctx.tool("robot.wait_steps", steps=n)
    except Exception:
        pass


def _square_jaws(ctx, phi_deg, above, gain_holder):
    """Turn joint 6 until the jaws' closing axis (tool y) lies along phi (mod 180), then re-centre."""
    gain = gain_holder.get("g", -1.0)
    for _ in range(6):
        r = _rot(_pose(ctx.tool("robot.get_ee_pose"))["rotation"])
        theta = math.degrees(math.atan2(r[1, 1], r[0, 1]))
        residual = ((phi_deg - theta + 90.0) % 180.0) - 90.0
        if abs(residual) < 0.3:
            break
        q = list(ctx.tool("robot.get_observation")["arms"][0]["joint_state"]["positions"])
        cmd = math.radians(residual) / gain
        if abs(q[5] + cmd) > 2.04:
            cmd = math.radians(residual + (180.0 if residual < 0 else -180.0)) / gain
        q[5] += cmd
        ctx.tool("robot.move_to_joints", joint_config={"positions": q})
        r2 = _rot(_pose(ctx.tool("robot.get_ee_pose"))["rotation"])
        moved = _wrap(math.degrees(math.atan2(r2[1, 1], r2[0, 1])) - theta)
        if abs(cmd) > 0.01 and 0.5 < abs(math.radians(moved) / cmd) < 1.5:
            gain = math.radians(moved) / cmd
        rot = _pose(ctx.tool("robot.get_ee_pose"))["rotation"]
        ctx.tool("robot.go_to_pose", pose={"position": above, "rotation": rot}, tolerance=0.001)
    gain_holder["g"] = gain


def run(ctx: NodeContext, index: int, start_dx: float, start_dy: float, start_yaw: float) -> Output:
    t0 = time.time()
    slot = {k: float(os.environ[f"JIGCHECK_SLOT_{k.upper()}"]) for k in "xyz"}
    cw, ch = META["cell"]
    ctx.tool("robot.open_gripper")
    here = _pose(ctx.tool("robot.get_ee_pose"))
    ctx.tool(
        "robot.go_to_pose_cartesian",
        pose={
            "position": {**here["position"], "z": slot["z"] + 0.12},
            "rotation": here["rotation"],
        },
    )
    ctx.tool("check.place_piece", dx=start_dx, dy=start_dy, yaw_deg=start_yaw)
    _wait(ctx, 80)
    truth0 = ctx.tool("check.piece_pose")
    result = {"index": index, "start_yaw": start_yaw}

    def done(stage_failed, **kw):
        final = ctx.tool("check.piece_pose")
        p = final["position"]
        rec = {
            **result,
            "failed_at": stage_failed,
            **kw,
            "final_dxy_mm": [round((p[k] - slot[k]) * 1e3, 2) for k in "xy"],
            "final_dz_mm": round((p["z"] - slot["z"]) * 1e3, 2),
            "final_yaw": round(_wrap(final["yaw_deg"]), 2),
            "final_tilt": round(final["tilt_deg"], 2),
            "seated_truth": bool(
                abs(p["z"] - slot["z"]) < 0.5e-3
                and final["tilt_deg"] < 2.0
                and math.hypot(p["x"] - slot["x"], p["y"] - slot["y"]) < 1.5e-3
            ),
            "elapsed_s": round(time.time() - t0, 1),
        }
        print("SKILLCHECK e2e_result " + json.dumps(rec), flush=True)
        return {"route": "ok"}

    # 1. Scan the staging spot.
    region = {"x": slot["x"] + start_dx, "y": slot["y"] + start_dy, "z": slot["z"]}
    scan = _skill("scanning-the-board-into-a-wrist-mosaic", "scan_mosaic.py").run(
        ctx, region_center=region, region_size=[0.08, 0.08], px_per_m=2000.0
    )
    _log(
        "scan",
        route=scan["route"],
        covered=scan.get("covered_fraction"),
        seams=scan.get("seam_shift_px"),
        tiles=scan.get("tiles_seen"),
    )
    if scan["route"] == "failed":
        return done("scan")
    mpp = scan["metres_per_pixel"]
    org = scan["origin"]
    hm = np.nan_to_num(scan["height_map"], nan=0.0)

    def px_to_world(u, v):
        return org["x"] + u * mpp, org["y"] - v * mpp

    # 2. Segment: the piece stands 2 mm proud, its harness handle 12 mm.
    body = (hm > 0.001).astype(np.uint8)
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(body)
    if n < 2:
        _log("segment", found=False)
        return done("segment")
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lab == k).astype(np.uint8)
    # The slot's reference point is the body's centre (the cell centre), not the mask's centroid, which the
    # knobs pull off by about 2 mm (seen: the seat jammed on a 2.1 mm offset estimate).
    bc = _skill("selecting-a-grip-clear-of-placed-neighbours", "select_grip.py").body_centre_px(
        mask
    )
    piece_xy = px_to_world(float(bc[0]), float(bc[1]))
    tp = truth0["position"]
    _log(
        "segment",
        area_mm2=round(mask.sum() * mpp * mpp * 1e6, 1),
        centre_err_mm=round(1e3 * math.dist(piece_xy, (tp["x"], tp["y"])), 2),
    )

    # 3. Side types and the yaw at which all four read as the slot needs (else re-scan once, closer).
    want = list(META["sides"])
    clsmod = _skill("classifying-piece-sides-for-border-first", "classify_sides.py")
    cls = clsmod.run(ctx, piece_mask=mask * 255, expected_sides=want)
    rescans = 0
    if cls["route"] in ("mismatch", "no_piece"):
        rescans = 1
        scan2 = _skill("scanning-the-board-into-a-wrist-mosaic", "scan_mosaic.py").run(
            ctx, region_center=region, region_size=[0.08, 0.08], px_per_m=3000.0
        )
        hm2 = np.nan_to_num(scan2["height_map"], nan=0.0)
        b2 = cv2.morphologyEx(
            (hm2 > 0.001).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(b2)
        if n2 >= 2:
            m2 = (lab2 == 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
            cls = clsmod.run(ctx, piece_mask=m2 * 255, expected_sides=want)
    yaw_est = float(cls.get("yaw_deg", 0.0))
    _log(
        "classify",
        route=cls["route"],
        rescans=rescans,
        sides=cls.get("side_types"),
        agree=cls.get("sides_agree"),
        yaw_est=round(yaw_est, 2),
        yaw_true=round(_wrap(truth0["yaw_deg"]), 2),
        yaw_err=round(_wrap(yaw_est - truth0["yaw_deg"]), 2),
    )
    if cls["route"] in ("mismatch", "no_piece"):
        return done("classify")

    # 4. Grip choice at the slot with the station blades; the pick uses it (bare piece, no grip block).
    org_board = [slot["x"] - 1.5 * cw, slot["y"] - 1.5 * ch]
    grip = _skill("selecting-a-grip-clear-of-placed-neighbours", "select_grip.py").run(
        ctx,
        piece_mask=mask,
        metres_per_pixel=mpp,
        rotation_deg=-yaw_est,
        slot_center=[slot["x"], slot["y"]],
        occupied_cells=[[0, 0], [1, 0], [2, 0], [0, 2], [1, 2], [2, 2]],
        cell_size=[cw, ch],
        board_origin=org_board,
        tab_reach=float(os.environ.get("TAB_REACH", "0.25")),
    )
    _log(
        "grip",
        route=grip["route"],
        closing_yaw=grip.get("closing_yaw_deg"),
        width_mm=round(1e3 * grip.get("grip_width_m", 0), 1),
        clearance_mm=round(1e3 * min(grip.get("clearance_m", 0), 1.0), 2),
        offset=grip.get("offset_m"),
    )
    if "pick_grip_pixel" not in grip:
        return done("grip")
    gu, gv = grip["pick_grip_pixel"]
    handle_xy = px_to_world(gu, gv)
    long_deg = float(grip["pick_closing_yaw_deg"])
    gw = float(grip["grip_width_m"])

    # 5. Pick the bare piece at that grip, blade tips 0.3 mm over the mat.
    grasp_z = slot["z"] + TIP_REACH + 0.0003
    above = {"x": handle_xy[0], "y": handle_xy[1], "z": grasp_z + 0.05}
    frame = ctx.tool("robot.grasp_frame", close_heading_deg=0.0)
    rot = None
    for key in ("rotation", "rotation_flipped"):
        r = frame.get(key)
        if (
            r
            and ctx.tool(
                "robot.go_to_pose", pose={"position": above, "rotation": r}, tolerance=0.001
            ).get("reached")
            is not False
        ):
            rot = r
            break
    if rot is None:
        _log("pick", ok=False, why="no hover")
        return done("pick")
    gh = {}
    _square_jaws(ctx, long_deg, above, gh)
    rot = _pose(ctx.tool("robot.get_ee_pose"))["rotation"]
    ctx.tool(
        "robot.go_to_pose",
        pose={"position": {**above, "z": grasp_z}, "rotation": rot},
        tolerance=0.0003,
        max_steps=400,
    )
    ctx.tool("robot.set_grip", object_width_m=gw, squeeze_m=0.003)
    _wait(ctx)
    ctx.tool(
        "robot.go_to_pose",
        pose={"position": {**above, "z": grasp_z + 0.03}, "rotation": rot},
        tolerance=0.0005,
    )
    _wait(ctx)
    held = ctx.tool("check.piece_pose")
    lifted = held["position"]["z"] - slot["z"]
    _log(
        "pick",
        ok=bool(lifted > 0.01),
        lifted_mm=round(lifted * 1e3, 1),
        yaw_slip=round(_wrap(held["yaw_deg"] - truth0["yaw_deg"]), 2),
    )
    if lifted < 0.01:
        return done("pick")

    meas = _skill("measuring-a-held-piece-from-the-wrist-camera", "measure_held.py")

    def measure(prior, tag):
        out = meas.run(ctx, piece_mask=mask, metres_per_pixel=mpp, prior=prior, grip_width_m=gw)
        tru = ctx.tool("check.piece_pose")
        if "world_pose" in out:
            wp = out["world_pose"]
            # Truth: the piece origin (cell centre) and its yaw change since the scan (the template's frame).
            _log(
                "measure_" + tag,
                route=out["route"],
                err_mm=[
                    round((wp["x"] - tru["position"]["x"]) * 1e3, 2),
                    round((wp["y"] - tru["position"]["y"]) * 1e3, 2),
                    round((wp["z"] - tru["position"]["z"]) * 1e3, 2),
                ],
                yaw_err=round(_wrap(wp["yaw_deg"] - (tru["yaw_deg"] - truth0["yaw_deg"])), 2),
                sigma=out.get("sigma"),
                pts=out.get("n_piece_points"),
            )
        else:
            _log("measure_" + tag, route=out["route"], detail=out.get("detail"))
        return out

    m1 = measure({"x": piece_xy[0], "y": piece_xy[1], "yaw_deg": 0.0}, "pick")
    slip1 = m1["world_pose"]["yaw_deg"] if m1["route"] == "measured" else 0.0
    ee1 = _pose(ctx.tool("robot.get_ee_pose"))["position"]
    rel1 = (
        np.array([m1["world_pose"]["x"], m1["world_pose"]["y"]])
        if m1["route"] == "measured"
        else np.array(piece_xy)
    ) - [ee1["x"], ee1["y"]]

    # 6. Turn to the slot's yaw (0), drop spot chosen by the skill.
    turn = _wrap(-(yaw_est + slip1))
    mode = os.environ.get("TURN_MODE", "mat")
    if mode == "mat":
        # Turn the pinched piece while the mat still carries it (0.5 mm lift), in 10 degree steps.
        here = _pose(ctx.tool("robot.get_ee_pose"))
        ctx.tool(
            "robot.go_to_pose_cartesian",
            pose={
                "position": {**here["position"], "z": grasp_z + 0.0005},
                "rotation": here["rotation"],
            },
        )
        _wait(ctx)
        kw = {"carry_lift_m": 0.0005, "max_step_deg": 10.0}
    elif mode == "fast":
        kw = {}
    else:
        kw = {"max_step_deg": 5.0}
    rot_out = _skill("rotating-a-held-piece-within-wrist-limits", "rotate_piece.py").run(
        ctx, turn_deg=turn, grasp_z=grasp_z, grip_width_m=gw, **kw
    )
    if mode == "mat":
        tw = ctx.tool("check.piece_pose")
        _log(
            "twist_on_mat",
            route=rot_out["route"],
            yaw_after=round(_wrap(tw["yaw_deg"]), 2),
            yaw_err=round(_wrap(tw["yaw_deg"]), 2),
            tilt=round(tw["tilt_deg"], 2),
            dz_mm=round((tw["position"]["z"] - slot["z"]) * 1e3, 2),
        )
        here = _pose(ctx.tool("robot.get_ee_pose"))
        ctx.tool(
            "robot.go_to_pose_cartesian",
            pose={
                "position": {**here["position"], "z": grasp_z + 0.03},
                "rotation": here["rotation"],
            },
        )
        _wait(ctx)
    held = ctx.tool("check.piece_pose")
    _log(
        "rotate",
        route=rot_out["route"],
        regrasps=rot_out.get("regrasps"),
        turn=round(turn, 1),
        held=bool(held["position"]["z"] - slot["z"] > 0.01),
        piece_yaw_after=round(_wrap(held["yaw_deg"]), 2),
        gripper=ctx.tool("robot.get_gripper"),
    )
    if (
        rot_out["route"] not in ("turned", "turned_with_regrasp")
        or held["position"]["z"] - slot["z"] < 0.01
    ):
        return done(
            "rotate",
            rotate_route=rot_out["route"],
            truly_held=bool(held["position"]["z"] - slot["z"] > 0.01),
        )
    ee2 = _pose(ctx.tool("robot.get_ee_pose"))["position"]
    a = math.radians(rot_out.get("turned_deg", turn))
    rr = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    p2 = rr @ rel1 + [ee2["x"], ee2["y"]]
    m2 = measure(
        {"x": float(p2[0]), "y": float(p2[1]), "yaw_deg": slip1 + rot_out.get("turned_deg", turn)},
        "turned",
    )
    # 7. Seat with the measured offset and the measured remaining yaw error.
    ee = _pose(ctx.tool("robot.get_ee_pose"))["position"]
    if m2["route"] == "measured":
        wp = m2["world_pose"]
        held_offset = {"x": wp["x"] - ee["x"], "y": wp["y"] - ee["y"], "z": wp["z"] - ee["z"]}
        yaw_err = _wrap(-(yaw_est + wp["yaw_deg"]))
    else:
        held_offset = {"x": float(p2[0] - ee["x"]), "y": float(p2[1] - ee["y"]), "z": -0.0132}
        yaw_err = 0.0
    tru = ctx.tool("check.piece_pose")
    true_off = [tru["position"][k] - ee[k] for k in "xy"]
    seat = _skill("seating-a-piece-by-wiggle-search", "seat_piece.py").run(
        ctx,
        slot_position=slot,
        held_offset=held_offset,
        yaw_error_deg=yaw_err,
        max_probes=30,
        max_time_s=300.0,
    )
    _log(
        "seat",
        route=seat["route"],
        probes=seat.get("probes"),
        yaw_err_given=round(yaw_err, 2),
        yaw_err_true=round(_wrap(-tru["yaw_deg"]), 2),
        offset_err_mm=[round((held_offset[k] - true_off[i]) * 1e3, 2) for i, k in enumerate("xy")],
    )
    a = math.radians(turn)

    # 8. Flush check from the wrist camera (the seat skill ends with the camera over the piece).
    obs = ctx.tool("robot.get_observation")
    cam = next((c for c in obs.get("cameras") or [] if "eye_in_hand" in str(c.get("name"))), None)
    if cam is not None:
        cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnt, key=cv2.contourArea)[:, 0, :].astype(float)
        w = np.array([px_to_world(u, vv) for u, vv in c]) - np.array(piece_xy)
        outline = (w @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])) + [
            slot["x"],
            slot["y"],
        ]
        nb = []
        for i, j in [(0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2)]:
            x0, y0 = org_board[0] + (i + 0.15) * cw, org_board[1] + (j + 0.15) * ch
            nb.append(
                [[x0, y0], [x0 + 0.7 * cw, y0], [x0 + 0.7 * cw, y0 + 0.7 * ch], [x0, y0 + 0.7 * ch]]
            )
        fl = _skill("verifying-piece-flushness-from-wrist-depth", "verify_flush.py").run(
            ctx,
            camera=cam,
            piece_outline=outline.tolist(),
            neighbour_outlines=nb,
            table_z=slot["z"],
        )
        _log(
            "flush",
            route=fl["route"],
            level_mm=round(1e3 * fl.get("level_m", float("nan")), 2),
            rise_mm=round(1e3 * fl.get("tilt_rise_m", float("nan")), 2),
        )
    else:
        fl = {"route": "no_camera"}
    return done(
        None if seat["route"] == "seated" else "seat", seat_route=seat["route"], flush=fl["route"]
    )
