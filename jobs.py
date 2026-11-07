"""Prints one run.py command per row of a grid, skipping rows already in the results file. Pipe it to sh.

    python3 jobs.py 5m | sh
    python3 jobs.py 5m --partners chase --models joint factored --ks 1 3 5
"""
import argparse
import itertools
import json
import os

import model as Mo
import world as W

ap = argparse.ArgumentParser()
ap.add_argument("tier")
ap.add_argument("--partners", nargs="+", default=list(W.PARTNERS))
ap.add_argument("--models", nargs="+", default=list(Mo.MODELS))
ap.add_argument("--ks", nargs="+", type=int, default=[1, 3])
ap.add_argument("--lrs", nargs="+", type=float, default=[3e-4, 1e-3, 3e-3])
ap.add_argument("--worlds", nargs="+", type=int, default=[0])
ap.add_argument("--out", default="results.jsonl")
a = ap.parse_args()
done = set()
if os.path.exists(a.out):
    done = {(r["partner"], r["model"], r["k"], r["lr"], r["world"], r["tier"]) for r in map(json.loads, open(a.out))
            if r.get("variant", "base") == "base"}
for p, m, kk, lr, w in itertools.product(a.partners, a.models, a.ks, a.lrs, a.worlds):
    if m not in Mo.LEARNED and (kk != 1 or lr != a.lrs[0]):
        continue  # the simulators read neither k nor a learning rate
    if (p, m, kk, lr, w, a.tier) not in done:
        print(f"python3 run.py --partner {p} --model {m} --k {kk} --lr {lr} --world {w} --tier {a.tier} --out {a.out}")
