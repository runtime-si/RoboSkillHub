"""Measures finding-the-longest-domino-chain: search size on random hands, and the layout on
the YAM station's table box from random start ends (inside the box, no overlaps, the
matching half toward the previous tile). Not collected by pytest; run it directly:

    .venv/bin/python tests/bundles/_domino_chain_eval.py
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import random
import time
from pathlib import Path

import numpy as np
from gap.testing import FakeContext

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills/finding-the-longest-domino-chain/scripts/find_chain.py"
)
spec = importlib.util.spec_from_file_location("find_chain", SCRIPT)
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)

TABLE = {"min_m": [-0.15, 0.10], "max_m": [0.35, 0.55]}  # real/motion_server.py tool box
L, W, MARGIN = 0.048, 0.024, 0.01


def search_size() -> None:
    for maxp, size, n in [
        (12, 15, 2000),
        (9, 15, 500),
        (6, 16, 100),
        (6, 20, 50),
        (6, 24, 20),
        (6, 28, 5),
    ]:
        rng = random.Random(size * 100 + maxp)
        full = [(a, b) for a in range(maxp + 1) for b in range(a, maxp + 1)]
        ms, nodes, exh, proven = [], [], 0, 0
        for _ in range(n):
            hand = rng.sample(full, size)
            t0 = time.perf_counter()
            steps, nd, ex, _, bound = fc.best_chain(hand, rng.randint(0, maxp))
            ms.append((time.perf_counter() - t0) * 1000)
            nodes.append(nd)
            exh += ex
            proven += ex or len(steps) >= bound
        ms.sort()
        nodes.sort()
        print(
            f"SEARCH double-{maxp} hand {size}: {n} hands, exhaustive {exh}, length proven {proven}, "
            f"nodes median {nodes[n // 2]} max {nodes[-1]}, ms median {ms[n // 2]:.1f} max {ms[-1]:.0f}"
        )


def _corners(fp):
    c, u = np.array(fp["center"][:2]), np.array(fp["axis"][:2])
    v = np.array([-u[1], u[0]])
    return np.array([c + su * L / 2 * u + sv * W / 2 * v for su in (-1, 1) for sv in (-1, 1)])


def _overlap(f1, f2, tol=0.0005):
    a, b = _corners(f1), _corners(f2)
    for f in (f1, f2):
        u = np.array(f["axis"][:2])
        for ax in (u, np.array([-u[1], u[0]])):
            pa, pb = a @ ax, b @ ax
            if min(pa.max(), pb.max()) - max(pa.min(), pb.min()) <= tol:
                return False
    return True


def _check(res, start_p) -> None:
    lo, hi = np.array(TABLE["min_m"]), np.array(TABLE["max_m"])
    fps = res["footprints_after"]
    for fp in fps:
        c = _corners(fp)
        assert np.all(c >= lo + MARGIN - 1e-9) and np.all(c <= hi - MARGIN + 1e-9)
    for i, j in itertools.combinations(range(len(fps)), 2):
        assert not _overlap(fps[i], fps[j]), (i, j)
    prev = np.array(start_p[:2])
    for step in res["chain"][: res["placed_count"]]:
        a_end, b_end = (np.array(x[:2]) for x in step["target_ends_world"])
        if step["layout"] != "crosswise":
            near, far = (a_end, b_end) if step["tile"][0] == step["near"] else (b_end, a_end)
            assert np.linalg.norm(near + (far - near) / 4 - prev) < np.linalg.norm(
                near + 3 * (far - near) / 4 - prev
            )
        prev = (a_end + b_end) / 2


def layout(maxp: int, size: int, n: int, seed: int) -> None:
    rng = random.Random(seed)
    full = [(a, b) for a in range(maxp + 1) for b in range(a, maxp + 1)]
    lo, hi = np.array(TABLE["min_m"]), np.array(TABLE["max_m"])
    found = greedy_ok = tiles = blocked = 0
    tries, ms, short = [], [], []
    for _ in range(n):
        hand = rng.sample(full, size)
        p = lo + MARGIN + rng.random() * (hi - lo - 2 * MARGIN) * np.array([1.0, 1.0])
        ang = rng.choice([0.0, 90.0, 180.0, 270.0])
        d = [math.cos(math.radians(ang)), math.sin(math.radians(ang))]
        start = {"value": rng.randint(0, maxp), "position": [p[0], p[1], 0.004], "direction": d}
        t0 = time.perf_counter()
        res = fc.run(FakeContext(), hand=hand, start_end=start, table=TABLE)
        ms.append((time.perf_counter() - t0) * 1000)
        if res["route"] == "no_chain":
            continue
        _check(res, start["position"])
        tiles += res["placed_count"]
        g = fc.run(
            FakeContext(), hand=hand, start_end=start, table=TABLE, layout_tries=res["length"]
        )
        greedy_ok += g["route"] == "chain_found"
        if res["route"] == "chain_found":
            found += 1
            tries.append(res["layout_tries"])
        elif res["placed_count"] == 0:
            blocked += 1
        else:
            short.append(res["length"] - res["placed_count"])
    with_chain = found + len(short) + blocked
    ms.sort()
    tries.sort()
    print(
        f"LAYOUT double-{maxp} hand {size}: {n} starts, {with_chain} with a chain, all placed {found} "
        f"(greedy alone {greedy_ok}), no_room with no first tile (start facing the edge) {blocked}, "
        f"no_room part-way {len(short)} (tiles unplaced {sorted(short)}), "
        f"tiles placed {tiles}, every placement checked; tries median "
        f"{tries[len(tries) // 2] if tries else 0} max {max(tries) if tries else 0}, "
        f"ms median {ms[n // 2]:.1f} max {ms[-1]:.0f}"
    )


if __name__ == "__main__":
    import sys

    if "--layout-only" not in sys.argv:
        search_size()
    layout(12, 15, 500, 1)
    layout(9, 15, 300, 2)
    layout(6, 28, 100, 3)
    full = [(a, b) for a in range(7) for b in range(a, 7)]
    start = {"value": 6, "position": [-0.12, 0.32, 0.004], "direction": [1, 0]}
    res = fc.run(FakeContext(), hand=full, start_end=start, table=TABLE)
    print(f"CHAIN28 route {res['route']} placed {res['placed_count']} tries {res['layout_tries']}")
