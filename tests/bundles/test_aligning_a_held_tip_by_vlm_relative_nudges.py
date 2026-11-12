import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills/aligning-a-held-tip-by-vlm-relative-nudges/scripts/nudge_tip.py"
)
spec = importlib.util.spec_from_file_location("nudge_tip", SCRIPT)
nt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nt)


def _oracle(tip, tol=0.0003):
    def ask(frames, prompt):
        if np.hypot(*tip) < tol:
            return "centred"
        ax = int(abs(tip[1]) > abs(tip[0]))
        return ("-" if tip[ax] > 0 else "+") + "xy"[ax]

    return ask


def test_parse():
    assert nt.parse("+x") == "+x"
    assert nt.parse("Answer: -y.") == "-y"
    assert nt.parse("It is centered.") == "centred"
    assert nt.parse("unsure") is None
    assert nt.parse("I cannot tell") is None


def test_perfect_answers_converge_under_a_millimetre():
    for off in [(0.007, -0.003), (-0.002, 0.009), (0.0045, 0.0045)]:
        tip = np.array(off, float)

        def move(dx, dy, tip=tip):
            tip += [dx, dy]

        r = nt.servo(lambda: [np.zeros((4, 4, 3), np.uint8)], _oracle(tip), move, max_steps=30)
        assert r["route"] == "centred" and np.hypot(*tip) < 0.001, (off, r, tip)


def test_refusals_give_no_verdict():
    r = nt.servo(lambda: [np.zeros((4, 4, 3), np.uint8)], lambda f, p: "unsure", lambda dx, dy: None)
    assert r["route"] == "no_verdict" and r["steps"] == 0


def test_arrows_follow_the_base_axes():
    k = np.array([[400.0, 0, 320], [0, 400, 240], [0, 0, 1]])
    t = np.eye(4)
    t[:3, :3] = np.diag([1.0, -1.0, -1.0])  # looking straight down
    t[:3, 3] = [0, 0, 0.2]
    j = nt.image_axes(k, t, np.zeros(3))
    assert j[0, 0] > 0 and abs(j[1, 0]) < 1e-6 and j[1, 1] < 0
