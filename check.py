"""Bug checks with known answers. About 7 seconds on one thread.

    python3 check.py           # known answers
    python3 check.py --slow    # also a 7s row through run.py for every partner and model, and --final

Known answers: the still partner never moves; the patrol partner stays on its circle; the chaser closes in
on a dot that stays put; the switching partner flips mode on its period; the simulator's rollout is the true
step sequence; with a still partner, planning that ignores the partner equals planning with the truth; with
a chaser it loses badly; the factored model's second latent half can't see our current action; a LeWM learns the patrol partner.
"""
import os
import sys
import time

import numpy as np
import torch

try:
    os.nice(19)
except OSError:
    pass
torch.set_num_threads(1)

import beat  # noqa: E402
import model as Mo  # noqa: E402
import run as R  # noqa: E402
import world as W  # noqa: E402

T0 = time.time()


def ok(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)
    if not cond:
        sys.exit(1)


def roll(partner, n=32, T=30, a=None):
    s, g = W.episodes(partner, n, 0, "tune")
    S = [s]
    for _ in range(T):
        s = W.step(partner, s, np.zeros((n, 2)) if a is None else a, g)
        S.append(s)
    return np.stack(S, 1), g


def t_world():
    E = {sp: W.episodes("chase", 32, 0, sp)[0] for sp in ("tune", "report", "final")}
    ok("splits are different episodes", not np.allclose(E["tune"], E["report"]) and not np.allclose(E["report"], E["final"]))
    ok("splits are reproducible", np.allclose(W.episodes("chase", 32, 0, "report")[0], E["report"]))
    S, _ = roll("still")
    ok("the still partner never moves", np.allclose(S[:, :, 2:4], S[:, :1, 2:4]))
    S, _ = roll("patrol")
    r = np.linalg.norm(S[:, 5:, 2:4] - 0.5, axis=-1)
    ok("the patrol partner stays near its circle", np.abs(r - W.RAD).max() < 0.06, f"max off {np.abs(r - W.RAD).max():.3f}")
    S, _ = roll("chase", T=8)
    d = np.linalg.norm(S[:, :, 0:2] - S[:, :, 2:4], axis=-1)
    ok("the chaser closes in on a dot that stays put", np.all(d[:, -1] < d[:, 0] - 0.15) or np.all((d[:, 0] - d[:, -1]) > 0.1))
    S, _ = roll("switch", T=30)
    flips = np.abs(np.diff(S[:, :, 5], axis=1)) > 0.5
    steps = S[:, 1:, 7]
    ok("the switching partner flips exactly on its period", np.array_equal(flips, (steps % S[:, 1:, 6]) == 0))


def t_sim():
    s, g = W.episodes("switch", 16, 0, "tune")
    A = np.random.default_rng(0).uniform(-1, 1, (16, 5, 2))
    P = Mo.Sim("switch").rollout(None, s, g, A)
    x = s
    for t in range(5):
        x = W.step("switch", x, A[:, t], g)
        ok_t = np.allclose(P[:, t], x[:, :4])
        if not ok_t:
            break
    ok("the simulator's rollout is the true step sequence", ok_t)
    P = Mo.Sim("chase", frozen=True).rollout(None, s, g, A)
    ok("blind imagination freezes the partner", np.allclose(P[:, :, 2:4], s[None, :, 2:4].transpose(1, 0, 2).repeat(5, 1)))


def t_plan():
    c = dict(R.TIERS["1m"], eps=24, ep_len=40)
    res = {}
    for p in ("still", "chase"):
        S0, G = W.episodes(p, c["eps"], 0, "tune")
        for m in ("oracle", "blind"):
            res[p, m] = R.plan(Mo.make(m, p), p, S0, G, 1, c, 1)[0]
    ok("with a still partner, ignoring it equals the truth", np.array_equal(res["still", "oracle"], res["still", "blind"]))
    ok("with a chaser, ignoring it loses badly", res["chase", "oracle"].mean() - res["chase", "blind"].mean() > 0.4,
       f"oracle {res['chase', 'oracle'].mean():.2f}, blind {res['chase', 'blind'].mean():.2f}")


def t_models():
    O = np.random.default_rng(0).random((2, 5, 4)).astype(np.float32)
    w = Mo.windows(O, 3)
    ok("windows hold the last k observations, oldest first, padded with the first",
       np.allclose(w[:, 0], np.concatenate([O[:, 0]] * 3, -1)) and np.allclose(w[:, 4], O[:, 2:5].reshape(2, 12)))
    for kind, want in (("factored", False), ("joint", True)):
        m = Mo.make(kind, "chase", k=3, h=32)
        win, a = torch.rand(8, 3 * Mo.LATENT), torch.rand(8, 2, requires_grad=True)
        m.delta(win, a)[:, Mo.LATENT // 2:].sum().backward()
        sees = a.grad.abs().sum().item() > 0
        ok(f"{kind}: the second half of the latent {'depends' if want else 'does not depend'} on our current action", sees == want)
    m = Mo.make("joint", "chase", k=3, h=32)
    h = W.episodes("chase", 4, 0, "tune")[0][:, None, :4].repeat(3, 1)
    A = np.random.default_rng(0).uniform(-1, 1, (4, 5, 2))
    ok("rollout from pictures equals rollout from pre-encoded latents",
       np.allclose(m.rollout(h, None, None, A), m.rollout(m.encode(h), None, None, A), atol=1e-5))
    O, U = W.collect("patrol", 48, 24, 0)
    m = Mo.make("factored", "patrol", k=1, h=32)
    before = [p.clone() for p in m.parameters()]
    Mo.train(m, O, U, 20, 0.0, 0)
    ok("learning rate 0 changes nothing", all(torch.equal(a, b) for a, b in zip(before, m.parameters())))
    Mo.train(m, O, U, 300, 3e-3, 0)
    Oe, Ue = W.collect("patrol", 32, 8, 9)
    e = R.pred_errors(m, Oe, Ue, 1, 6)
    ok("a factored model learns the patrol partner (error under 0.65 x the stay-still guess)", np.mean(e["other_err"]) < 0.65 * np.mean(e["still_err"]),
       f"{np.mean(e['other_err']):.3f} vs still {np.mean(e['still_err']):.3f}")


def t_rows():
    for p in W.PARTNERS:
        for m in Mo.MODELS:
            r = R.main(["--partner", p, "--model", m, "--k", "3", "--tier", "7s", "--out", ""])
            ok(f"run.py 7s row, {p} {m}", len(r["report"]["success"]) == R.TIERS["7s"]["eps"])
    r = R.main(["--tier", "7s", "--final", "--out", ""])
    ok("--final scores the hidden split", "final" in r and "report" not in r)


if __name__ == "__main__":
    t_world()
    t_sim()
    t_models()
    t_plan()
    ok("beat selftest", beat.selftest() == "beat selftest passed")
    if "--slow" in sys.argv:
        t_rows()
    print(f"all checks passed in {time.time() - T0:.1f} s")
