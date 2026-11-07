"""The three claims, written before any comparison ran.

    python3 beat.py --tier 5m
    python3 beat.py --selftest

For each partner, model and k, the learning rate is chosen on tune episodes (best success, then lowest
partner error), and the claim reads report episodes. Every model plans on the same episodes with the same
sampling seed, so gaps are paired by episode, with a 95% bootstrap interval.

C1 reactive   On partners that react to us (chase, switch), does the factored model beat the joint model
              at the same k = 3? Lower bound above 0 on both partners: modelling the partner as an agent helps.
C2 scripted   On the patrol partner, which ignores us, do they tie? A tie is the expected answer; a factored
              win here would mean the gain is not about reacting.
C3 history    On the switch partner, whose mode is hidden, does the factored model with k = 3 beat k = 1?
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = 1000


def load(path):
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


def tuned(R, tier, partner, model, k, final=False):
    c = [r for r in R if r["tier"] == tier and r["partner"] == partner and r["model"] == model and r["k"] == k and r["final"] == final]
    return max(c, key=lambda r: (np.mean(r["tune"]["success"]), -np.mean(r["tune"].get("other_err", [0])))) if c else None


def gap(a, b, split, seed=0):
    x, y = np.array(a[split]["success"], float), np.array(b[split]["success"], float)
    rng = np.random.default_rng(seed)
    d = [np.mean(x[i] - y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    return float(np.mean(x - y)), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def judge(a, b, split, yes, no):
    if a is None or b is None:
        return {"verdict": "not judged"}
    m, (lo, hi) = gap(a, b, split)
    return {"gap": m, "ci": [lo, hi], "verdict": yes if lo > 0 else no if hi < 0 else "ties"}


def report(R, tier, final=False):
    split = "final" if final else "report"
    t = lambda p, m, k: tuned(R, tier, p, m, k, final)
    out = {"C1": {p: judge(t(p, "factored", 3), t(p, "joint", 3), split, "factored wins", "joint wins") for p in ("chase", "switch")}}
    vs = {v["verdict"] for v in out["C1"].values()}
    out["C1 on both reactive partners"] = vs.pop() if len(vs) == 1 else "partners disagree"
    out["C2"] = judge(t("patrol", "factored", 3), t("patrol", "joint", 3), split, "factored wins", "joint wins")
    out["C3"] = judge(t("switch", "factored", 3), t("switch", "factored", 1), split, "history helps", "history hurts")
    return out


def selftest():
    """Made-up rows: the factored model wins on reacting partners, ties on patrol, and history helps on switch."""
    rng = np.random.default_rng(0)
    n = 300
    u = {"tune": rng.random(n), "report": rng.random(n)}

    def row(p, m, k, q, lr=1e-3):
        return {"tier": "x", "partner": p, "model": m, "k": k, "lr": lr, "final": False,
                **{s: {"success": list((u[s] < q).astype(int)), "other_err": [1 - q]} for s in u}}

    R = [row("chase", "factored", 3, 0.8), row("chase", "joint", 3, 0.6), row("chase", "joint", 3, 0.5, 3e-3),
         row("switch", "factored", 3, 0.75), row("switch", "joint", 3, 0.55), row("switch", "factored", 1, 0.5),
         row("patrol", "factored", 3, 0.9), row("patrol", "joint", 3, 0.9)]
    o = report(R, "x")
    assert o["C1 on both reactive partners"] == "factored wins", o
    assert o["C2"]["verdict"] == "ties" and o["C3"]["verdict"] == "history helps", o
    R2 = R[:3] + [row("switch", "factored", 3, 0.5), row("switch", "joint", 3, 0.7)]
    o = report(R2, "x")
    assert o["C1 on both reactive partners"] == "partners disagree", o
    assert report([], "x")["C2"]["verdict"] == "not judged"
    return "beat selftest passed"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tier", default="5m")
    p.add_argument("--file", default=f"{HERE}/results.jsonl")
    p.add_argument("--final", action="store_true")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    print(selftest() if a.selftest else json.dumps(report(load(a.file), a.tier, a.final), indent=1))
