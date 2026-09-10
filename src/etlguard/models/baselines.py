"""Baselines, from what a data-quality tool does today up to a supervised ceiling.

Fairness note: the classical detectors are **not** given a single feature vector
and asked to compete with a model that sees 32 timesteps. Each one receives the
current profile *and* its deviation from the window's rolling mean, so it has
access to temporal context too. A baseline that is crippled by construction
proves nothing.

Every baseline exposes the same two outputs as the main model, a run-level
score and a per-node score, so detection and localization are compared on
identical terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def window_features(Xz: np.ndarray, ends: np.ndarray, window: int) -> np.ndarray:
    """``[M, N, 2F]``: the current profile plus its deviation from recent history."""
    cur = Xz[ends]                                            # [M, N, F]
    hist = np.stack([Xz[t - window + 1:t].mean(axis=0) for t in ends])
    return np.concatenate([cur, cur - hist], axis=-1)


class Baseline:
    name = "baseline"

    def fit(self, feats: np.ndarray) -> "Baseline":
        raise NotImplementedError

    def score(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(run_score [M], node_score [M, N])``."""
        raise NotImplementedError


class ZScoreEWMA(Baseline):
    """Per-(node, feature) threshold on deviation from a rolling baseline.

    This is the honest stand-in for what teams actually run today: Great
    Expectations / Deequ style column checks with a rolling reference. It is the
    baseline the whole project has to beat to be worth anything.
    """

    name = "zscore_ewma"

    def fit(self, feats: np.ndarray) -> "ZScoreEWMA":
        self.mu = feats.mean(axis=0)
        self.sd = feats.std(axis=0) + 1e-6
        return self

    def score(self, feats: np.ndarray):
        z = np.abs((feats - self.mu) / self.sd)
        node = z.max(axis=-1)
        return node.max(axis=1), node


class PCAResidual(Baseline):
    """Reconstruction residual of a linear subspace fitted on the training runs."""

    name = "pca_residual"

    def __init__(self, n_components: int = 24):
        self.k = n_components

    def fit(self, feats: np.ndarray) -> "PCAResidual":
        m, n, f = feats.shape
        self.shape = (n, f)
        self.pca = PCA(n_components=min(self.k, m, n * f)).fit(feats.reshape(m, -1))
        return self

    def score(self, feats: np.ndarray):
        m, n, f = feats.shape
        flat = feats.reshape(m, -1)
        rec = self.pca.inverse_transform(self.pca.transform(flat))
        err = ((flat - rec) ** 2).reshape(m, n, f)
        node = err.mean(axis=-1)
        return node.max(axis=1), node


class PerNodeForest(Baseline):
    """One Isolation Forest per pipeline node.

    Fitting per node rather than one detector over the concatenated 342-dim
    vector is what makes a node-level score possible at all, and it is also the
    stronger configuration, since each forest sees a coherent feature space.
    """

    name = "isolation_forest"

    def __init__(self, n_estimators: int = 200, seed: int = 7):
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, feats: np.ndarray) -> "PerNodeForest":
        self.models = [
            IsolationForest(n_estimators=self.n_estimators, random_state=self.seed,
                            n_jobs=-1).fit(feats[:, i])
            for i in range(feats.shape[1])
        ]
        return self

    def score(self, feats: np.ndarray):
        # score_samples is higher for inliers; negate so higher means more anomalous.
        node = np.stack([-self.models[i].score_samples(feats[:, i])
                         for i in range(feats.shape[1])], axis=1)
        return node.max(axis=1), node


class PerNodeLOF(Baseline):
    name = "lof"

    def __init__(self, n_neighbors: int = 30):
        self.k = n_neighbors

    def fit(self, feats: np.ndarray) -> "PerNodeLOF":
        # Subsample: LOF is O(n^2)-ish and the extra points buy nothing here.
        idx = np.linspace(0, len(feats) - 1, min(len(feats), 3000)).astype(int)
        self.models = [
            LocalOutlierFactor(n_neighbors=self.k, novelty=True).fit(feats[idx, i])
            for i in range(feats.shape[1])
        ]
        return self

    def score(self, feats: np.ndarray):
        node = np.stack([-self.models[i].score_samples(feats[:, i])
                         for i in range(feats.shape[1])], axis=1)
        return node.max(axis=1), node


# --------------------------------------------------------------------------- #
# Deep baselines
# --------------------------------------------------------------------------- #

class _MLPAutoEncoder(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, latent: int = 32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_in))

    def forward(self, x):
        return self.dec(self.enc(x))


class DenseAutoEncoder(Baseline):
    """The standard reconstruction baseline: an MLP autoencoder on the profile."""

    name = "dense_ae"

    def __init__(self, epochs: int = 60, lr: float = 1e-3, device: str = "cpu",
                 seed: int = 7):
        self.epochs, self.lr, self.device, self.seed = epochs, lr, device, seed

    def fit(self, feats: np.ndarray) -> "DenseAutoEncoder":
        torch.manual_seed(self.seed)
        m, n, f = feats.shape
        self.shape = (n, f)
        x = torch.tensor(feats.reshape(m, -1), dtype=torch.float32, device=self.device)
        self.model = _MLPAutoEncoder(n * f).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(m, device=self.device)
            for i in range(0, m, 256):
                b = x[perm[i:i + 256]]
                loss = ((self.model(b) - b) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def score(self, feats: np.ndarray):
        self.model.eval()
        m, n, f = feats.shape
        x = torch.tensor(feats.reshape(m, -1), dtype=torch.float32, device=self.device)
        err = ((self.model(x) - x) ** 2).view(m, n, f).cpu().numpy()
        node = err.mean(axis=-1)
        return node.max(axis=1), node


class _LSTMAE(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.enc = nn.LSTM(d_in, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, d_in)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        rep = h[-1].unsqueeze(1).expand(-1, x.shape[1], -1)
        y, _ = self.dec(rep)
        return self.out(y)


class LSTMAutoEncoder(Baseline):
    """Sequence autoencoder over the whole window: the temporal deep baseline."""

    name = "lstm_ae"

    def __init__(self, window: int = 32, epochs: int = 25, lr: float = 1e-3,
                 device: str = "cpu", seed: int = 7):
        self.window, self.epochs, self.lr = window, epochs, lr
        self.device, self.seed = device, seed

    def fit_sequences(self, seqs: np.ndarray) -> "LSTMAutoEncoder":
        """``seqs`` is ``[M, W, N*F]``."""
        torch.manual_seed(self.seed)
        m, w, d = seqs.shape
        x = torch.tensor(seqs, dtype=torch.float32, device=self.device)
        self.model = _LSTMAE(d).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(m, device=self.device)
            for i in range(0, m, 128):
                b = x[perm[i:i + 128]]
                loss = ((self.model(b) - b) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
        return self

    @torch.no_grad()
    def score_sequences(self, seqs: np.ndarray, n_nodes: int):
        self.model.eval()
        m, w, d = seqs.shape
        f = d // n_nodes
        outs = []
        x = torch.tensor(seqs, dtype=torch.float32, device=self.device)
        for i in range(0, m, 256):
            b = x[i:i + 256]
            # Only the final step is scored: that is the run being judged.
            e = ((self.model(b)[:, -1] - b[:, -1]) ** 2).view(-1, n_nodes, f)
            outs.append(e.cpu().numpy())
        err = np.concatenate(outs, axis=0)
        node = err.mean(axis=-1)
        return node.max(axis=1), node


CLASSICAL_BASELINES = [ZScoreEWMA, PCAResidual, PerNodeForest, PerNodeLOF]
