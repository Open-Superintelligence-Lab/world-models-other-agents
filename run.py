"""Trains one world model on one partner, plans with it, and appends one row.

    python3 run.py --partner chase --model factored --k 3 --tier 5m --lr 0.001 --world 0
    python3 run.py --partner chase --model factored --k 3 --sigreg 0       # one question: no SIGReg

Every flag below "the questions" changes one part of the model or the data. A row records the changed flags
as `variant` ("base" when none changed), and compare.py puts a variant next to the base row. QUESTIONS.md
lists which flag answers which question.

A row stores, on tune and report episodes (or the hidden final ones with --final):
  success   reached the goal before touching the partner, per episode
  touched   touched the partner, per episode
and, for learned models, on held-out data trajectories:
  other_err the partner's imagined position after H steps of our recorded actions, distance to the truth
  me_err    the same for our dot
  still_err the partner error of assuming it stays where it is, for scale
  halves    with --halves: how well each latent half alone reads each dot (see model.halves)
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import model as Mo
import world as W

TIERS = {
    "7s": dict(traj=48, T=24, steps=150, h=64, eps=6, ep_len=30, H=6, samples=32, iters=2, seconds=7),
    "1m": dict(traj=300, T=40, steps=1500, h=128, eps=48, ep_len=60, H=8, samples=64, iters=3, seconds=0),
    "5m": dict(traj=1000, T=60, steps=5000, h=256, eps=150, ep_len=60, H=10, samples=128, iters=3, seconds=0),
    "1h": dict(traj=4000, T=80, steps=30000, h=512, eps=500, ep_len=80, H=12, samples=256, iters=4, seconds=0),
}
SPLIT_SEED = {"tune": 0, "report": 0, "final": 0}


def lowest_priority():
    try:
        os.nice(19)
    except OSError:
        pass
    torch.set_num_threads(1)


def cost(P, g):
    """P (B, H, 6) imagined observations, g (B, 2) -> (B,) cost: distance to goal, plus a large penalty
    whenever the imagined dots come within a margin of touching."""
    d_goal = np.linalg.norm(P[:, :, 0:2] - g[:, None], axis=-1)
    d_pair = np.linalg.norm(P[:, :, 0:2] - P[:, :, 2:4], axis=-1)
    return d_goal[:, -1] + 0.5 * d_goal.mean(1) + 5 * np.maximum(W.TOUCH + 0.02 - d_pair, 0).sum(1)


def plan(model, partner, S0, G, k, c, seed):
    """Model-predictive control with the cross-entropy method, replanning every step."""
    rng = np.random.default_rng(seed)
    n, M, H = len(S0), c["samples"], c["H"]
    s, hist = S0.copy(), [S0[:, :W.ODIM].copy()] * k
    done, success, touched = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    mean = np.zeros((n, H, 2))
    for _ in range(c["ep_len"]):
        h = np.stack(hist[-k:], 1)
        enc = hasattr(model, "encode")
        if enc:
            h = model.encode(h)             # learned models: encode the pictures once per step, not per sample
        mu, sd = mean, np.full((n, H, 2), 0.7)
        rep = lambda x: np.repeat(x, M, 0)
        for _ in range(c["iters"]):
            A = np.clip(mu[:, None] + sd[:, None] * rng.standard_normal((n, M, H, 2)), -1, 1)
            P = model.rollout(rep(h), rep(s), rep(G), A.reshape(n * M, H, 2), encoded=enc)
            C = cost(P, rep(G)).reshape(n, M)
            top = np.argsort(C, 1)[:, :max(2, M // 8)]
            E = np.take_along_axis(A, top[:, :, None, None], 1)
            mu, sd = E.mean(1), E.std(1) + 0.05
        s = W.step(partner, s, mu[:, 0], G)
        hist.append(s[:, :W.ODIM].copy())
        t, r = W.touching(s) & ~done, W.reached(s, G) & ~done
        touched |= t
        success |= r & ~t
        done |= t | r
        mean = np.concatenate([mu[:, 1:], np.zeros((n, 1, 2))], 1)
    return success, touched


def pred_errors(model, O, U, k, H):
    """Imagine H steps from step k of held-out data with the recorded actions."""
    win = Mo.windows(O, k)[:, k - 1].reshape(len(O), k, W.ODIM)
    P = model.rollout(win, None, None, U[:, k - 1:k - 1 + H])
    true = O[:, k - 1 + H]
    return {"other_err": np.linalg.norm(P[:, -1, 2:4] - true[:, 2:4], axis=-1).tolist(),
            "me_err": np.linalg.norm(P[:, -1, 0:2] - true[:, 0:2], axis=-1).tolist(),
            "still_err": np.linalg.norm(O[:, k - 1, 2:4] - true[:, 2:4], axis=-1).tolist()}


SEEK = {"mix": 0.5, "seek": 1.0, "random": 0.0}


def variant(a):
    """The changed flags, sorted, as one string; "base" when none changed."""
    ch = [f"{f}={getattr(a, f)}" for f, d in DEFAULTS.items() if getattr(a, f) != d]
    return " ".join(sorted(ch)) or "base"


def training_data(a, c):
    n = max(8, round(c["traj"] * a.data_scale))
    if a.train_partners == "all":    # an equal share from each of the four partners
        parts = [W.collect(p, n // len(W.PARTNERS), c["T"], a.world, SEEK[a.data]) for p in W.PARTNERS]
        return np.concatenate([x[0] for x in parts]), np.concatenate([x[1] for x in parts])
    return W.collect(a.partner, n, c["T"], a.world, SEEK[a.data])


def row(a):
    c = dict(TIERS[a.tier])
    t0 = time.time()
    W.DISTRACTOR = a.distractor
    deadline = t0 + c["seconds"] if c["seconds"] else None
    opt = dict(latent=a.latent, sigreg=a.sigreg, action=not a.no_action, gray=a.gray, per_dot=a.per_dot,
               tell=a.tell, img=a.img)
    model = Mo.make(a.model, a.partner, a.k, c["h"], seed=a.world, match=a.match_size, **opt)
    out = {"partner": a.partner, "model": a.model, "k": a.k, "tier": a.tier, "lr": a.lr, "world": a.world,
           "final": a.final, "variant": variant(a)}
    if a.model in Mo.LEARNED:
        O, U = training_data(a, c)
        Mo.train(model, O, U, c["steps"], a.lr, a.world, deadline)
        out["weights"] = model.test_weights()
        if a.halves:
            out["halves"] = Mo.halves(model, O, W.collect(a.partner, 32, c["T"], a.world + 200)[0])
    for split in (("tune", "final") if a.final else ("tune", "report")):
        S0, G = W.episodes(a.partner, c["eps"], a.world, split)
        sc, tc = plan(model, a.partner, S0, G, a.k, c, a.world * 31 + W.STREAM[split])
        res = {"success": sc.astype(int).tolist(), "touched": tc.astype(int).tolist()}
        if a.model in Mo.LEARNED:
            Oe, Ue = W.collect(a.partner, max(16, c["eps"]), a.k + c["H"], a.world + 100 + W.STREAM[split])
            res.update(pred_errors(model, Oe, Ue, a.k, c["H"]))
        out[split] = res
    out["seconds"] = round(time.time() - t0, 2)
    return out


DEFAULTS = dict(sigreg=0.1, no_action=False, gray=False, match_size=False, per_dot=False, tell=0.0,
                distractor=False, train_partners="one", data="mix", data_scale=1.0, latent=32, img=16)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--partner", default="chase", choices=W.ALL)
    ap.add_argument("--model", default="joint", choices=Mo.MODELS)
    ap.add_argument("--k", type=int, default=1, help="how many past observations a learned model sees")
    ap.add_argument("--tier", default="7s", choices=sorted(TIERS))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--world", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="read the hidden split; use once, at the end")
    ap.add_argument("--out", default="results.jsonl")
    q = ap.add_argument_group("the questions (see QUESTIONS.md); defaults are the base model")
    q.add_argument("--sigreg", type=float, default=0.1, help="SIGReg weight; 0 removes it")
    q.add_argument("--no-action", action="store_true", help="the predictor never sees our action")
    q.add_argument("--gray", action="store_true", help="draw every dot white, so colour can't say which is ours")
    q.add_argument("--match-size", action="store_true", help="joint only: widen it to the factored weight count")
    q.add_argument("--per-dot", action="store_true", help="one encoder per dot colour, each giving one latent half")
    q.add_argument("--tell", type=float, default=0.0, help="weight of a loss that puts us in half A, the partner in half B")
    q.add_argument("--distractor", action="store_true", help="add a harmless third dot on a looping path")
    q.add_argument("--train-partners", default="one", choices=("one", "all"), help="train on this partner or all four")
    q.add_argument("--data", default="mix", choices=sorted(SEEK), help="training moves: goal-seeking, random, or half each")
    q.add_argument("--data-scale", type=float, default=1.0, help="multiply the number of training trajectories")
    q.add_argument("--latent", type=int, default=32, help="latent size")
    q.add_argument("--img", type=int, default=16, help="picture size in pixels, a multiple of 4")
    q.add_argument("--halves", action="store_true", help="also report which latent half holds which dot")
    a = ap.parse_args(argv)
    lowest_priority()
    r = row(a)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    sp = r.get("report", r.get("final"))
    extra = f" partner err {np.mean(sp['other_err']):.3f} (still {np.mean(sp['still_err']):.3f})" if "other_err" in sp else ""
    print(f"{a.partner} {a.model} k={a.k}: success {np.mean(sp['success']):.2f} touched {np.mean(sp['touched']):.2f}{extra} | {r['seconds']} s")
    return r


if __name__ == "__main__":
    main()
