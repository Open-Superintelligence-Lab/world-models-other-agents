"""Puts each variant next to the base model with the same partner, model, k and tier, and prints the
difference in success on report episodes with a 95% paired interval, plus the imagination errors.

    python3 compare.py --tier 7s
    python3 compare.py --tier 5m --partner chase

Run the base row and the variant row first (QUESTIONS.md has the commands). For each variant and base, the
learning rate is the one with the best tune success, as in beat.py.
"""
import argparse

import numpy as np

import beat as B


def mean(r, split, key):
    return float(np.mean(r[split][key])) if r and key in r[split] else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="7s")
    ap.add_argument("--partner")
    ap.add_argument("--results", default="results.jsonl")
    a = ap.parse_args(argv)
    R = [r for r in B.load(a.results) if r["tier"] == a.tier and not r["final"] and (not a.partner or r["partner"] == a.partner)]
    keys = sorted({(r["partner"], r["model"], r["k"], r.get("variant", "base")) for r in R})
    print(f"{'partner':8s} {'model':9s} {'k':>2s}  {'variant':34s} {'success':>8s} {'vs base':>8s} {'95% interval':>15s} "
          f"{'me_err':>7s} {'other_err':>9s}")
    for p, m, k, v in keys:
        r = B.tuned(R, a.tier, p, m, k, variant=v)
        base = B.tuned(R, a.tier, p, m, k)
        d, ci = ("", "")
        if v != "base" and base:
            g, (lo, hi) = B.gap(r, base, "report")
            d, ci = f"{g:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]"
        print(f"{p:8s} {m:9s} {k:2d}  {v:34s} {mean(r, 'report', 'success'):8.2f} {d:>8s} {ci:>15s} "
              f"{mean(r, 'report', 'me_err'):7.3f} {mean(r, 'report', 'other_err'):9.3f}")
        if "halves" in r:
            h = r["halves"]
            print(f"{'':22s} halves: our dot from A {h['me_from_A']:.3f}, from B {h['me_from_B']:.3f}; "
                  f"partner from A {h['other_from_A']:.3f}, from B {h['other_from_B']:.3f}")


if __name__ == "__main__":
    main()
