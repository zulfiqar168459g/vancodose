"""GRU sequence models, used only in the architecture comparison.

* ``GRUClearance``: sparse TDM sequence (infusion-rate channel + measured levels) and
  static covariates -> log(CL).
* ``GRUCurve``: infusion-rate sequence + static covariates -> concentration-time curve.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import SEED
from .pk import Regimen

STEP_H = 0.25
GRID = np.arange(0, 72 + 1e-9, STEP_H)


def rate_channel(regimen: Regimen, grid=GRID) -> np.ndarray:
    r = np.zeros_like(grid)
    for t0, amount, dur in regimen.events():
        r[(grid >= t0) & (grid < t0 + dur)] += amount / dur
    return r / 1000.0


def tdm_sequence(regimen: Regimen, levels, grid=GRID) -> np.ndarray:
    x = np.zeros((len(grid), 3), np.float32)
    x[:, 0] = rate_channel(regimen, grid)
    for t, c in levels:
        i = int(np.argmin(np.abs(grid - t)))
        x[i, 1] = 1.0
        x[i, 2] = np.log(max(c, 0.5))
    return x


class _GRU(nn.Module):
    def __init__(self, n_in, n_static, hidden=48, n_out=1, per_step=False):
        super().__init__()
        self.h0 = nn.Linear(n_static, hidden)
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 32), nn.ReLU(), nn.Linear(32, n_out))
        self.per_step = per_step

    def forward(self, seq, static):
        h0 = torch.tanh(self.h0(static)).unsqueeze(0)
        out, h = self.gru(seq, h0)
        if self.per_step:
            s = static.unsqueeze(1).expand(-1, out.shape[1], -1)
            return self.head(torch.cat([out, s], -1)).squeeze(-1)
        return self.head(torch.cat([h[-1], static], -1)).squeeze(-1)


def _train(model, seq, static, y, epochs=150, lr=3e-3, wd=1e-4, val_frac=0.15, patience=20):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    n = len(seq)
    idx = rng.permutation(n)
    nv = max(1, int(n * val_frac))
    va, tr = idx[:nv], idx[nv:]
    S, Z, Y = (torch.tensor(a, dtype=torch.float32) for a in (seq, static, y))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, wait = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 64)):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(S[b], Z[b]), Y[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(model(S[va], Z[va]), Y[va]))
        if v < best - 1e-5:
            best, best_state, wait = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:  # early stopping
                break
    model.load_state_dict(best_state)
    return model


class GRUClearance:
    def fit(self, seq, static, y):
        torch.set_num_threads(2)
        self.mu, self.sd = static.mean(0), static.std(0) + 1e-6
        self.ym = float(y.mean())
        self.model = _train(_GRU(seq.shape[2], static.shape[1]), seq, (static - self.mu) / self.sd, y - self.ym)
        return self

    def predict(self, seq, static):
        with torch.no_grad():
            z = torch.tensor((static - self.mu) / self.sd, dtype=torch.float32)
            return self.model(torch.tensor(seq, dtype=torch.float32), z).numpy() + self.ym


class GRUCurve:
    def fit(self, seq, static, y):
        torch.set_num_threads(2)
        self.mu, self.sd = static.mean(0), static.std(0) + 1e-6
        self.scale = float(y.mean())
        self.model = _train(_GRU(seq.shape[2], static.shape[1], per_step=True), seq, (static - self.mu) / self.sd,
                            y / self.scale, epochs=200)
        return self

    def predict(self, seq, static):
        with torch.no_grad():
            z = torch.tensor((static - self.mu) / self.sd, dtype=torch.float32)
            return np.clip(self.model(torch.tensor(seq, dtype=torch.float32), z).numpy() * self.scale, 0, None)
