"""verifying-build-against-model on synthetic depth: detection rate per injected
error and camera. Not collected by pytest (the tests import ``trial``); run
from the repo root:

    .venv/bin/python tests/bundles/_build_verify_eval.py --trials 40

Each trial builds a random supported target of 8 bricks (sizes 1x1-2x6, two
rotations, layers 1-3) on a 12 x 12 baseplate, injects one error into the
built scene, renders it from the overhead camera (~0.9 m, oblique) or the wrist
D405 (0.16-0.24 m, 5-15 deg off vertical) with the station's depth noise, and
runs the skill on the depth frame and the exact plate pose. A trial counts as
detected when the route names the error and the report names the right brick:

- clean: nothing wrong; right = ``matches``.
- missing: a brick with nothing on it is left out; ``missing_bricks`` listing it.
- extra: a brick not in the target is added (on the plate or on a brick);
  ``extra_bricks`` with a brick on its cells.
- misplaced: a brick with nothing on it is moved elsewhere (same size and
  colour); ``misplaced`` pairing it with its target.
- unseated: a brick with nothing on it rests on the stud tops (lifted one stud
  height); ``unseated`` listing it.
- tilted: a brick at least 2 studs long with nothing on it, one end on the
  stud tops and the other seated (turned about the near bottom edge across its
  long side); ``unseated`` listing it.
- wrong_colour: a brick with nothing on it has another colour (RGB rendered
  with LDraw colours, tools/colour's classify); ``wrong_colour`` listing it.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.bundles import _build_scene as S  # noqa: E402

ERRORS = ("clean", "missing", "extra", "misplaced", "unseated", "tilted", "wrong_colour")
EXPECTED = {
    "clean": "matches",
    "missing": "missing_bricks",
    "extra": "extra_bricks",
    "misplaced": "misplaced",
    "unseated": "unseated",
    "tilted": "unseated",
    "wrong_colour": "wrong_colour",
}
PLATE_POSE = {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}


class _Ctx:
    def __init__(self, classify=None):
        self.classify = classify

    def tool(self, name, **kw):
        if name == "colour.classify" and self.classify is not None:
            return self.classify(**kw)
        raise KeyError(name)


def inject(rng, scene: S.Scene, error: str) -> tuple[S.Scene, dict]:
    """The built scene with one error, and what was done (target index etc.)."""
    built = copy.deepcopy(scene)
    info: dict = {"error": error}
    if error == "clean":
        return built, info
    if error == "extra":
        taken = {(x, y) for b in scene.bricks for x, y, _ in b.cells()}
        b = S.random_free_brick(rng, built.bricks, scene.plate, avoid=None)
        if b is None:
            return built, {"error": "skip"}
        del taken
        built.bricks.append(b)
        info["cells"] = {(x, y) for x, y, _ in b.cells()}
        info["layer"] = b.layer
        return built, info
    choices = S.tops(scene.bricks)
    if error == "tilted":
        choices = [i for i in choices if max(scene.bricks[i].footprint) >= 2]
        if not choices:
            return built, {"error": "skip"}
    k = int(rng.choice(choices))
    info["index"] = k
    b = built.bricks[k]
    if error == "missing":
        built.bricks.pop(k)
    elif error == "unseated":
        b.lift = S.SYSTEMS[scene.system]["stud_h"]
    elif error == "tilted":
        b.tilt_lift = S.SYSTEMS[scene.system]["stud_h"]
    elif error == "wrong_colour":
        b.colour = next(c for c in rng.permutation(S.COLOURS) if c != b.colour)
    elif error == "misplaced":
        rest = built.bricks[:k] + built.bricks[k + 1 :]
        old = set(b.cells())
        for _ in range(200):
            nb = S.random_free_brick(rng, rest, scene.plate, size=b.size, colour=b.colour)
            if nb is not None and set(nb.cells()) != old:
                break
        else:
            return built, {"error": "skip"}
        nb.step = b.step
        built.bricks = rest + [nb]
        info["to"] = (nb.x, nb.y, nb.layer, nb.rotation)
    return built, info


def judge(out: dict, info: dict) -> bool:
    err = info["error"]
    if out["route"] != EXPECTED[err]:
        return False
    if err == "clean":
        return True
    if err == "extra":
        return any(
            {
                (e["cell"]["x"] + i, e["cell"]["y"] + j)
                for i in range(e["footprint"][0])
                for j in range(e["footprint"][1])
            }
            <= info["cells"]
            for e in out["extra"]
        )
    k = info["index"]
    key = {
        "missing": "missing",
        "unseated": "unseated",
        "tilted": "unseated",
        "wrong_colour": "wrong_colour",
        "misplaced": "misplaced",
    }[err]
    return any(e["id"] == k for e in out[key])


def _plate_pose(err_mm: float, tilt_deg: float, rng) -> dict:
    """The plate pose the skill is given: the true one (identity) off by
    err_mm in a random direction (x, y and z) and tilted by tilt_deg."""
    if not err_mm and not tilt_deg:
        return PLATE_POSE
    v = rng.normal(size=3)
    v *= err_mm * 1e-3 / np.linalg.norm(v)
    ax = rng.normal(size=2)
    ax /= np.linalg.norm(ax)
    h = np.radians(tilt_deg) / 2
    return {
        "position": {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])},
        "rotation": {
            "w": float(np.cos(h)),
            "x": float(ax[0] * np.sin(h)),
            "y": float(ax[1] * np.sin(h)),
            "z": 0.0,
        },
    }


def trial(
    vb,
    system: str,
    camera: str,
    error: str,
    seed: int,
    classify=None,
    plate_err_mm=0.0,
    plate_tilt_deg=0.0,
) -> tuple[str, bool, dict]:
    rng = np.random.default_rng(seed)
    scene = S.random_build(rng, system)
    target = [b.as_target(i) for i, b in enumerate(scene.bricks)]
    built, info = inject(rng, scene, error)
    if info["error"] == "skip":
        return "skip", False, {}
    cams = {c[0]: c for c in S.cameras(rng, system)}
    _, t_cam, k, noise = cams[camera]
    want_rgb = error == "wrong_colour" or classify is not None
    table = None
    if want_rgb:
        pal = importlib.import_module("gap_skills.tools.colour._impl").PALETTES["lego"]
        table = {e["name"]: tuple(e["rgb"]) for e in pal}
        table["tan"] = table.get("tan", (228, 205, 158))
    depth, rgb = S.render(built, np.eye(4), t_cam, k, noise, seed=seed, rgb_table=table)
    cam = {"depth": depth, "intrinsics": k, "pose": t_cam}
    if want_rgb:
        cam["rgb"] = rgb
    out = vb.run(
        _Ctx(classify),
        target=target,
        camera=cam,
        plate_pose=_plate_pose(plate_err_mm, plate_tilt_deg, np.random.default_rng(seed + 7)),
        plate_studs=list(scene.plate),
        system=system,
        check_colour=want_rgb,
    )
    out["_info"] = info
    return out["route"], judge(out, info), out


def load_skill():
    from gap.skills import load_skills

    registry = load_skills(ROOT)
    for info in registry._skills.values():
        path = info.bundle_dir / "tools.py"
        if info.tools_module is None and path.is_file():
            info.tools_module = registry._import_module(
                path, f"gap_skills.{info.namespace}.{info.name}.tools"
            )
    vb = registry.get("verifying-build-against-model").canonical_scripts["verify_build"].module
    classify = importlib.import_module("gap_skills.tools.colour._impl").classify
    return vb, classify


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--systems", default="lego,duplo")
    ap.add_argument("--cameras", default="overhead,wrist")
    ap.add_argument("--errors", default=",".join(ERRORS))
    ap.add_argument("--plate-err-mm", type=float, default=0.0)
    ap.add_argument("--plate-tilt-deg", type=float, default=0.0)
    args = ap.parse_args()
    vb, classify = load_skill()
    rows = []
    for system in args.systems.split(","):
        for camera in args.cameras.split(","):
            for error in args.errors.split(","):
                hits, n, routes, unres, t0 = 0, 0, Counter(), 0, time.time()
                for seed in range(args.trials):
                    route, ok, out = trial(
                        vb,
                        system,
                        camera,
                        error,
                        1000 * ERRORS.index(error) + seed,
                        classify if error == "wrong_colour" else None,
                        args.plate_err_mm,
                        args.plate_tilt_deg,
                    )
                    if route == "skip":
                        continue
                    n += 1
                    hits += ok
                    routes[route] += 1
                    if error in ("unseated", "tilted") and not ok:
                        k = out["_info"]["index"]
                        unres += any(e["id"] == k for e in out.get("seat_unresolved", []))
                    elif error not in ("unseated", "tilted"):
                        unres += bool(out.get("seat_unresolved"))
                rows.append((system, camera, error, hits, n, dict(routes), unres, time.time() - t0))
                print(
                    f"| {system} | {camera} | {error} | {hits}/{n} | "
                    f"{', '.join(f'{k} {v}' for k, v in routes.most_common())} | {unres} |",
                    flush=True,
                )
    tot = defaultdict(lambda: [0, 0])
    for system, camera, _error, hits, n, *_ in rows:
        tot[(system, camera)][0] += hits
        tot[(system, camera)][1] += n
    for key, (h, n) in tot.items():
        print(key, f"{h}/{n}")


if __name__ == "__main__":
    main()
