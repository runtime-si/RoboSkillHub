"""Run roboskillhub's LEGO Isaac episodes with RoCo's head camera on, saving one RGB frame per recorded snapshot.

    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=~/rynn-eval/rsh/src:~/rynn-eval python -m rynn_capture \
        --roco ~/rynn-eval/src/RoCo_BrickAssembly --out runs/X --cases single_on_brick/goal_fsm/none/roco ...

Nothing in the episode logic changes: the runner's user_config gets RoCo's own Head_Camera block (the runner
strips cameras because its baseline is privileged), and Recorder.snap additionally stores the head camera's RGB
as frames/<n>.png with the path in the snapshot dict ("frame").
"""

import sys
from pathlib import Path

from roboskillhub.tasks.assembly_eval import lego_isaac_runner as R

HEAD = {
    "Head_Camera": {
        "FPS": 30,
        "Resolution": [640, 480],
        "Prim_Path": "/World/Robot/zed_depth_frame/headcam",
    }
}
_state = {"n": 0, "out": None}

_orig_user_config = R.user_config


def user_config(*a, **kw):
    cfg = _orig_user_config(*a, **kw)
    cfg["Robot_Config"]["Camera_Config"] = HEAD
    return cfg


_orig_snap = R.Recorder.snap
_orig_init = R.Recorder.__init__
CAM_PATH = "/World/rynn_eval_cam"
CAM_DIST = float(__import__("os").environ.get("RYNN_CAM_DIST", "0.24"))
CAM_DIR = tuple(
    float(v) for v in __import__("os").environ.get("RYNN_CAM_DIR", "0.35,-0.6,0.72").split(",")
)
# from the target toward the camera; default: the robot's side, raised ~50 deg


def _look_quat(pos, target):
    import numpy as np
    from scipy.spatial.transform import Rotation

    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    # USD camera axes: +X right, +Y up, looking along -Z.
    x, y, z, w = Rotation.from_matrix(np.stack([right, up, -fwd], axis=1)).as_quat()
    return np.array([w, x, y, z])


def init(self, env, ids):
    """Re-aim RoCo's head camera (a sensor RoCo creates before play, which renders) at the base plate."""
    import numpy as np
    import omni.usd
    from pxr import Gf, UsdGeom
    from scipy.spatial.transform import Rotation

    _orig_init(self, env, ids)
    # The goal bricks sit at the base plate's origin in every scene of lego_scenes (final snapshots).
    target = np.asarray(env.get_prim_world_T(ids["0"]))[:3, 3] + np.array([0.0, 0.0, 0.01])
    d = np.asarray(CAM_DIR, float)
    pos = target + CAM_DIST * d / np.linalg.norm(d)
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(HEAD["Head_Camera"]["Prim_Path"])
    w, x, y, z = _look_quat(pos, target)
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    M[:3, 3] = pos
    parent = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())).T
    local = np.linalg.inv(parent) @ M
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(local.T.tolist()))
    UsdGeom.Camera(prim).GetFocalLengthAttr().Set(18.0)
    UsdGeom.Camera(prim).GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    got = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
    print(
        "RYNN_CAM",
        [round(float(v), 4) for v in got[:3, 3]],
        "->",
        [round(float(v), 4) for v in target],
        flush=True,
    )


def snap(self):
    import numpy as np
    from PIL import Image

    d = _orig_snap(self)
    cam = (getattr(self.env, "cameras", {}) or {}).get("Head_Camera")
    if cam is not None and _state["out"] is not None:
        rgb = cam.get_rgb()
        if rgb is not None and np.asarray(rgb).size:
            frames = _state["out"] / "frames"
            frames.mkdir(parents=True, exist_ok=True)
            path = frames / f"{_state['n']:05d}.png"
            Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8)).save(path)
            d["frame"] = str(path.relative_to(_state["out"]))
            _state["n"] += 1
    return d


async def kit_main():
    import os
    import traceback

    R.user_config = user_config
    R.Recorder.__init__ = init
    R.Recorder.snap = snap
    rc = 1
    try:
        args = R.parse_args(sys.argv[1:])
        _state["out"] = args.out.resolve()
        rc = await R.run_all(args)
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)


def main():
    import importlib.util
    import os
    import shlex

    argv = sys.argv[1:]
    R.parse_args(argv)
    os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")
    entry = Path(importlib.util.find_spec("bricksim").origin).with_name("_entry.py")
    sys.argv = [
        sys.argv[0],
        "isaacsim.exp.full",
        "--no-window",
        "--/app/content/emptyStageOnStart=false",
        "--/crashreporter/enabled=false",
        "--/log/outputStreamLevel=info",
        "--/log/channels/*=warn",
        "--/log/channels/bricksim=info",
        f"--/exts/omni.services.transport.server.http/port={os.environ.get('RYNN_KIT_PORT', '8011')}",
        "--exec",
        " ".join(shlex.quote(a) for a in [str(entry), "rynn_capture:kit_main", *argv]),
    ]
    import isaacsim

    isaacsim.main()


if __name__ == "__main__":
    main()
