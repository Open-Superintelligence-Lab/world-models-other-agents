"""The kill test, written before any comparison ran. Prints pass, fail or not judged for each test.

    python3 kill.py --tier 5m

A test with missing rows says not judged, and a missing row never passes. Failure numbers are in EXPERIMENTS.md.
"""
import argparse

import numpy as np

import beat as B
import world as W


def k(name, verdict, note=""):
    print(f"{name:52s} {verdict:11s} {note}")
    return verdict


def s(r, split="report"):
    return float(np.mean(r[split]["success"])) if r else np.nan


def run(R, tier):
    out = {}
    t = lambda p, m, kk=1: B.tuned(R, tier, p, m, kk)
    orc = {p: t(p, "oracle") for p in W.PARTNERS}
    have = [p for p in W.PARTNERS if orc[p]]
    lo = [f"{p} {s(orc[p]):.2f}" for p in have if s(orc[p]) < 0.8]
    out["K0"] = k("K0 the true simulator plans: success >= 0.8", "not judged" if len(have) < len(W.PARTNERS) else ("fail" if lo else "pass"), "; ".join(lo))

    bl = {p: t(p, "blind") for p in W.PARTNERS}
    if not all(bl[p] and orc[p] for p in W.PARTNERS):
        out["K1"] = k("K1 the partner matters only when it moves", "not judged")
    else:
        bad = [p for p in W.REACTIVE if s(orc[p]) - s(bl[p]) < 0.3] + (["still"] if abs(s(orc["still"]) - s(bl["still"])) > 0.05 else [])
        out["K1"] = k("K1 the partner matters only when it moves", "fail" if bad else "pass", "; ".join(bad))

    ls = [t("still", m, 3) for m in ("joint", "factored")]
    if not all(ls) or not orc["still"]:
        out["K2"] = k("K2 learned models match the truth on still", "not judged")
    else:
        bad = [r["model"] for r in ls if s(r) < s(orc["still"]) - 0.1]
        out["K2"] = k("K2 learned models match the truth on still", "fail" if bad else "pass", "; ".join(bad))

    room = [p for p in W.REACTIVE if orc[p] and t(p, "joint", 3) and s(orc[p]) - s(t(p, "joint", 3)) >= 0.1]
    seen = [p for p in W.REACTIVE if orc[p] and t(p, "joint", 3)]
    out["K3"] = k("K3 the joint model leaves 0.1 to gain somewhere", "not judged" if len(seen) < len(W.REACTIVE) else ("pass" if room else "fail"), f"on {room}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="5m")
    ap.add_argument("--file", default="results.jsonl")
    a = ap.parse_args()
    run(B.load(a.file), a.tier)
