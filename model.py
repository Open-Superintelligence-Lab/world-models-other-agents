"""The world models compared, all with one interface: rollout(hist, state, goal, A) -> imagined
observations (B, H, 4). Learned models read only `hist`, the last k observations, as pictures; the two
simulators read the true `state`.

Both learned models are LeWM (LeWorldModel, arXiv 2603.19312): a convolutional encoder from a picture to a
latent, a predictor from the last k latents and our action to the next latent, trained end to end on
prediction MSE plus 0.1 x SIGReg (the authors' code from lucas-maes/le-wm, 17 knots). They differ only in
the predictor:

joint     one network predicts the whole next latent from the last k latents and our action. The partner is
          part of the environment, which is how LeWM and other world models are built.
factored  the latent is split in two halves. One network predicts the first half from the last k latents and
          our action. A second network predicts the other half from the last k latents only, without our
          current action. Whatever can be predicted without our action (the partner, who can't see our
          next move) can live in the second half; nothing tells the model which half holds what.

Differences from LeWM's paper, all for speed on a laptop: a 2-layer conv encoder in place of ViT-Tiny, an MLP
predictor in place of a ViT-S, 16- or 24-pixel pictures, and 64 SIGReg projections in place of 1024.

The planner needs our position and the partner's to score a plan (goal distance and a touching penalty), so
a small probe reads the 4 positions from a latent. The probe is fitted after training, on the frozen encoder's
latents of the training frames, so it doesn't change the world model, and both learned models get the same
probe. (Fitting it during training, on a moving encoder, left it 3 times less accurate.) LeWM plans towards a goal picture; that
doesn't work here, because a goal picture would also fix where the partner is.

blind     the true simulator with the partner frozen where it is. The floor: planning that ignores the partner.
oracle    the true simulator, hidden mode included. The ceiling.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import world as W

MODELS = ("joint", "factored", "blind", "oracle")
LEARNED = ("joint", "factored")
LATENT = 32


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


def windows(O, k):
    """O (n, T + 1, 4) -> (n, T + 1, k * 4): at each t the last k observations, oldest first, with the first
    observation repeated before the start."""
    pad = np.concatenate([np.repeat(O[:, :1], k - 1, 1), O], 1)
    return np.concatenate([pad[:, i:i + O.shape[1]] for i in range(k)], -1)


class SIGReg(nn.Module):
    """LeWM's regulariser, copied from lucas-maes/le-wm: pushes random 1D projections of the latents towards
    a standard normal, which stops the encoder from collapsing to a constant."""

    def __init__(self, knots=17, num_proj=64):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        w = torch.full((knots,), 2 * dt)
        w[[0, -1]] = dt
        win = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", win)
        self.register_buffer("weights", w * win)

    def forward(self, proj):  # (T, B, D)
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x = (proj @ A).unsqueeze(-1) * self.t
        err = (x.cos().mean(-3) - self.phi).square() + x.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()


class LeWM(nn.Module):
    def __init__(self, kind, k, h, img=16, w=16):
        super().__init__()
        self.kind, self.k, self.img = kind, k, img
        D = LATENT
        self.enc = nn.Sequential(nn.Conv2d(3, w, 4, 2, 1), nn.GELU(), nn.Conv2d(w, 2 * w, 4, 2, 1), nn.GELU(),
                                 nn.Flatten(), nn.Linear(2 * w * (img // 4) ** 2, D))
        if kind == "joint":
            self.pred = mlp(k * D + 2, h, D)
            last = [self.pred[-1]]
        else:
            self.me = mlp(k * D + 2, h // 2, D // 2)
            self.other = mlp(k * D, h, D // 2)
            last = [self.me[-1], self.other[-1]]
        for layer in last:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
        self.probe = mlp(D, 128, 4)
        self.sig = SIGReg()

    def pictures(self, O):
        return torch.from_numpy(W.render(O, self.img))

    def embed(self, x):
        """x (..., 3, img, img) -> (..., D)."""
        lead = x.shape[:-3]
        return self.enc(x.reshape(-1, *x.shape[-3:])).reshape(*lead, LATENT)

    @torch.no_grad()
    def encode(self, hist):
        """hist (B, k, 4) observations -> (B, k, D) latents as numpy, so the planner can encode once per step."""
        return self.embed(self.pictures(hist)).numpy()

    def delta(self, zwin, a):
        """zwin (B, k * D) the last k latents, a (B, 2) -> predicted change of the latest latent, (B, D)."""
        if self.kind == "joint":
            return self.pred(torch.cat([zwin, a], -1))
        return torch.cat([self.me(torch.cat([zwin, a], -1)), self.other(zwin)], -1)

    def test_weights(self):
        """Weights used at test time: encoder, predictor and probe; SIGReg has none."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def rollout(self, hist, state, goal, A):
        """hist is (B, k, 4) observations or (B, k, D) latents from encode()."""
        h = np.asarray(hist, np.float32)
        Z = torch.from_numpy(h) if h.shape[-1] == LATENT else self.embed(self.pictures(h))
        zwin = Z.flatten(1)
        A = torch.as_tensor(A, dtype=torch.float32)
        out = []
        for t in range(A.shape[1]):
            z = zwin[:, -LATENT:] + self.delta(zwin, A[:, t])
            zwin = torch.cat([zwin[:, LATENT:], z], -1)
            out.append(self.probe(z).clamp(W.R, 1 - W.R))
        return torch.stack(out, 1).numpy()


class Sim:
    """The true simulator. With frozen=True the partner never moves in imagination."""

    def __init__(self, partner, frozen=False):
        self.partner, self.frozen = partner, frozen

    def rollout(self, hist, state, goal, A):
        s, out = state.copy(), []
        for t in range(A.shape[1]):
            s = W.step("still" if self.frozen else self.partner, s, A[:, t], goal)
            out.append(s[:, :W.ODIM].copy())
        return np.stack(out, 1)


def make(kind, partner, k=1, h=128, img=16, seed=0):
    if kind in LEARNED:
        torch.manual_seed(seed)
        return LeWM(kind, k, h, img)
    return Sim(partner, frozen=(kind == "blind"))


def losses(model, O, U):
    """O (B, L + 1, 4) observations, U (B, L, 2) actions, with the first k - 1 frames as history only.
    Returns (prediction MSE, SIGReg)."""
    k = model.k
    z = model.embed(model.pictures(O))
    U = torch.from_numpy(U)
    P, Tg = [], []
    for j in range(k - 1, U.shape[1]):
        zw = z[:, j - k + 1:j + 1].flatten(1)
        P.append(zw[:, -LATENT:] + model.delta(zw, U[:, j]))
        Tg.append(z[:, j + 1])
    pred = F.mse_loss(torch.stack(P, 1), torch.stack(Tg, 1))
    return pred, model.sig(z.transpose(0, 1))


def fit_probe(model, O, steps, lr=3e-3, batch=256):
    """Fit the position probe on the frozen encoder's latents of every training frame."""
    with torch.no_grad():
        Z = model.embed(model.pictures(O.reshape(-1, 4)))
    Y = torch.from_numpy(O.reshape(-1, 4))
    opt = torch.optim.Adam(model.probe.parameters(), lr=lr)
    for _ in range(steps):
        j = torch.randint(0, len(Z), (batch,))
        loss = F.mse_loss(model.probe(Z[j]), Y[j])
        opt.zero_grad()
        loss.backward()
        opt.step()


def train(model, O, U, steps, lr, seed, deadline=None, batch=64, ahead=3):
    """LeWM's loss on short clips: k frames of history, then `ahead` predicted steps. The start of each
    trajectory is padded with its first frame, as the planner pads its history. Then the probe is fitted for
    max(1500, steps // 2) steps on precomputed latents, about 1 s per 1500 steps."""
    torch.manual_seed(seed)
    O0, k = O, model.k
    O = np.concatenate([np.repeat(O[:, :1], k - 1, 1), O], 1)
    U = np.concatenate([np.zeros((len(U), k - 1, 2), np.float32), U], 1)
    n, T = U.shape[:2]
    L = min(k - 1 + ahead, T)
    opt = torch.optim.Adam([p for n_, p in model.named_parameters() if not n_.startswith("probe.")], lr=lr)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        if deadline and time.time() > deadline:
            break
        i = rng.integers(0, n, batch)
        t = rng.integers(0, T - L + 1, batch)
        idx = t[:, None] + np.arange(L + 1)
        pred, sig = losses(model, O[i[:, None], idx], U[i[:, None], idx[:, :-1]])
        loss = pred + 0.1 * sig
        opt.zero_grad()
        loss.backward()
        opt.step()
    if lr > 0:
        fit_probe(model, O0, max(1500, steps // 2))
    return model.eval()
