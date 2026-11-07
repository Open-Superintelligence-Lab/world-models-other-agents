"""A table with two dots: ours, which must reach a goal, and a partner that moves by its own rule. Touching
the partner ends the episode as a failure. Batched numpy, deterministic, so a perfect simulator copy can plan.

Partners (none of them is told our plan):
  still    never moves. The control: modelling the partner can't help.
  patrol   circles the table's centre, ignoring us.
  chase    walks towards us.
  switch   alternates between patrol and chase every 6 to 14 steps. The switch time is hidden from the
           observation, so only the partner's recent moves show which mode it is in.
Extra partner, outside the kill test and claims:
  instant  chases where we will be after this step's move, so it reacts to our current action. A world model
           that predicts the partner without our current action (factored) can't represent it exactly.

Distractor (off by default; run.py --distractor turns it on): a third dot that drifts along a fixed looping
path, ignores everyone, and is harmless to touch. It is drawn in green. With it off, its columns stay 0 and
it isn't drawn.

State columns: 0-1 us, 2-3 partner, 4-5 distractor, 6 patrol angle, 7 mode, 8 switch period, 9 step count,
10-11 distractor phase. The observation is columns 0-5.
"""
import numpy as np

PARTNERS = ("still", "patrol", "chase", "switch")    # the kill test and claims use these
EXTRA = ("instant",)
ALL = PARTNERS + EXTRA
REACTIVE = ("chase", "switch")
SPEED, PSPEED, R, THR = 0.04, 0.03, 0.06, 0.06
TOUCH = 2 * R
SPIN, RAD = 0.12, 0.22
SDIM, ODIM = 12, 6
DISTRACTOR = False
STREAM = {"train": 1, "tune": 2, "report": 3, "final": 4}


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _towards(p, t, speed):
    d = t - p
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return p + d / np.maximum(n, 1e-9) * np.minimum(n, speed)


def partner_target(partner, s, g):
    """Where the partner heads this step, for each row."""
    me, ot = s[:, 0:2], s[:, 2:4]
    circle = 0.5 + RAD * np.stack([np.cos(s[:, 6]), np.sin(s[:, 6])], 1)
    if partner == "still":
        return ot
    if partner == "patrol":
        return circle
    if partner in ("chase", "instant"):     # step() passes our new position for instant
        return ot + _unit(me - ot)
    if partner == "switch":
        return np.where(s[:, 7:8] > 0.5, ot + _unit(me - ot), circle)
    raise ValueError(partner)


def step(partner, s, a, g):
    """One step for a batch. Both dots move at once, each from the positions at the start of the step."""
    s = s.copy()
    a = np.clip(a, -1, 1)
    tgt = None if partner == "instant" else partner_target(partner, s, g)
    s[:, 0:2] = np.clip(s[:, 0:2] + SPEED * a, R, 1 - R)
    if tgt is None:
        tgt = partner_target(partner, s, g)
    s[:, 2:4] = np.clip(_towards(s[:, 2:4], tgt, PSPEED), R, 1 - R)
    s[:, 6] = s[:, 6] + SPIN
    s[:, 9] = s[:, 9] + 1
    if partner == "switch":
        flip = (s[:, 9] % s[:, 8]) == 0
        s[:, 7] = np.where(flip, 1 - s[:, 7], s[:, 7])
    if DISTRACTOR:
        s[:, 4:6] = _drift(s)
    return s


def _drift(s):
    t = s[:, 9:10]
    return 0.5 + 0.38 * np.sin(np.array([0.21, 0.17]) * t + s[:, 10:12])


def render(O, img=16, gray=False):
    """O (..., 6) observations -> (..., 3, img, img) float32 pictures, what the learned world models see.
    Our dot is drawn in red (channel 0), the partner in blue (channel 2) and the distractor, when on, in green
    (channel 1), each a soft disc of radius R. With gray=True every dot is drawn white in all three channels,
    so colour no longer says which dot is ours. The goal is not drawn: the planner's cost reads it as a
    position (see model.py)."""
    O = np.asarray(O, np.float32)
    c = (np.arange(img, dtype=np.float32) + 0.5) / img
    X, Y = c[None, :], c[:, None]
    lead = O.shape[:-1]
    o = O.reshape(-1, ODIM)
    out = np.zeros((len(o), 3, img, img), np.float32)
    for ch, j in ((0, 0), (2, 2)) + (((1, 4),) if DISTRACTOR else ()):
        d2 = (X[None] - o[:, j, None, None]) ** 2 + (Y[None] - o[:, j + 1, None, None]) ** 2
        disc = np.exp(-d2 / (2 * (0.7 * R) ** 2))
        if gray:
            out = np.maximum(out, disc[:, None])
        else:
            out[:, ch] = disc
    return out.reshape(*lead, 3, img, img)


def touching(s):
    return np.linalg.norm(s[:, 0:2] - s[:, 2:4], axis=-1) < TOUCH


def reached(s, g):
    return np.linalg.norm(s[:, 0:2] - g, axis=-1) < THR


def init(partner, rng, n):
    """We start near one side and the goal is near the opposite side, so the straight path crosses the middle,
    where every partner starts. Sides are random, so no direction is special."""
    s = np.zeros((n, SDIM))
    side = rng.integers(0, 4, n)
    a, b = rng.uniform(0.08, 0.2, n), rng.uniform(0.15, 0.85, n)
    start = np.stack([a, b], 1)
    goal = np.stack([1 - rng.uniform(0.08, 0.2, n), rng.uniform(0.15, 0.85, n)], 1)
    swap = side % 2 == 1                    # left-right or bottom-top
    start[swap], goal[swap] = start[swap][:, ::-1], goal[swap][:, ::-1]
    flip = side >= 2                        # or the other way round
    start[flip], goal[flip] = 1 - start[flip], 1 - goal[flip]
    s[:, 0:2] = start
    s[:, 6] = rng.uniform(0, 2 * np.pi, n)
    s[:, 8] = rng.integers(6, 15, n)
    s[:, 7] = rng.integers(0, 2, n)
    if partner in ("patrol", "switch"):
        s[:, 2:4] = 0.5 + RAD * np.stack([np.cos(s[:, 6]), np.sin(s[:, 6])], 1)
    else:
        s[:, 2:4] = 0.5 + 0.5 * (goal - start) * rng.uniform(-0.3, 0.3, (n, 1)) + rng.normal(0, 0.06, (n, 2))
    phase = rng.uniform(0, 2 * np.pi, (n, 2))   # drawn always, so the distractor flag leaves other draws unchanged
    if DISTRACTOR:
        s[:, 10:12] = phase
        s[:, 4:6] = _drift(s)
    return s, goal


def episodes(partner, n, seed, split):
    """Start states and goals; each split draws from its own stream."""
    rng = np.random.default_rng([seed, STREAM[split], ALL.index(partner)])
    return init(partner, rng, n)


def collect(partner, n, T, seed, seek_frac=0.5):
    """A seek_frac share of the trajectories head to the goal with noise, the rest use smooth random actions,
    so by default the partner sees both purposeful and aimless behaviour. Returns observations (n, T + 1, 6)
    and actions (n, T, 2). Touching does not end a data trajectory."""
    rng = np.random.default_rng([seed, STREAM["train"], ALL.index(partner), 7])
    s, g = init(partner, rng, n)
    seek = rng.random(n) < seek_frac
    A = rng.uniform(-1, 1, (n, 2))
    O, U = [s[:, :ODIM].copy()], []
    for _ in range(T):
        A = np.clip(0.7 * A + 0.6 * rng.uniform(-1, 1, (n, 2)), -1, 1)
        a = np.where(seek[:, None], _unit(g - s[:, 0:2]) + rng.normal(0, 0.5, (n, 2)), A)
        a = np.clip(a, -1, 1)
        s = step(partner, s, a, g)
        O.append(s[:, :ODIM].copy())
        U.append(a)
    return np.stack(O, 1).astype(np.float32), np.stack(U, 1).astype(np.float32)
