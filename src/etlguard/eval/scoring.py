"""Turn a trained encoder into detection and root-cause scores.

Scoring mirrors training. The model was trained to fill in hidden entries, so at
inference we *ask it the same question* rather than running it unmasked. An
unmasked masked-autoencoder can copy its input straight to its output, and the
reconstruction error then measures nothing.

Two masking configurations at the final timestep, and the difference between
them is the method:

``e_hist``  every node is hidden, so each prediction rests on history alone.
            Large whenever a run departs from its own past: the detection signal.

``e_ctx``   only node *i* is hidden; every other node at that timestep, its
            parents included, stays visible. This is *surprise given inputs*.
            A task downstream of a real failure produces anomalous output, but
            output that follows perfectly from its (also anomalous) inputs, so
            ``e_ctx`` is small and it is correctly read as a victim. The task
            where the incident started has healthy inputs and anomalous output,
            so ``e_ctx`` stays large. That asymmetry is what separates the origin
            from its blast radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class ScoreBundle:
    t: np.ndarray            # [M] index of the scored run
    e_hist: np.ndarray       # [M, N] history-only error per node
    e_ctx: np.ndarray        # [M, N] ancestor-conditioned error per node
    feat_err: np.ndarray     # [M, N, F] per-feature error (history-only pass)
    y_anom: np.ndarray       # [M, N]
    y_root: np.ndarray       # [M, N]
    y_type: np.ndarray       # [M]
    severity: np.ndarray     # [M]

    @property
    def is_incident(self) -> np.ndarray:
        return (self.y_type > 0).astype(np.int64)

    def detection_score(self, rule: str = "max_hist") -> np.ndarray:
        """Run-level anomaly score under one of several pooling rules."""
        if rule == "max_hist":
            return self.e_hist.max(axis=1)
        if rule == "mean_hist":
            return self.e_hist.mean(axis=1)
        if rule == "max_ctx":
            return self.e_ctx.max(axis=1)
        if rule == "top2_hist":
            s = np.sort(self.e_hist, axis=1)[:, -2:]
            return s.mean(axis=1)
        raise ValueError(f"unknown pooling rule {rule!r}")

    DETECTION_RULES = ("max_hist", "mean_hist", "max_ctx", "top2_hist")


@torch.no_grad()
def score_loader(model, loader: DataLoader, device: str,
                 n_nodes: int) -> ScoreBundle:
    model.eval()
    ts, eh, ec, fe, ya, yr, yt, sev = [], [], [], [], [], [], [], []

    for batch in loader:
        x = batch["x"].to(device)
        cal = batch["cal"].to(device)
        b, w, n, f = x.shape

        # ---- history-only: hide the entire final run ----
        m_hist = torch.zeros_like(x)
        m_hist[:, -1] = 1.0
        pred = model(x, m_hist, cal)
        err = (pred[:, -1] - x[:, -1]) ** 2               # [b, n, f]
        eh.append(err.mean(dim=-1).cpu().numpy())
        fe.append(err.cpu().numpy())

        # ---- ancestor-conditioned: hide one node at a time ----
        # Replicating the window N times lets all N queries run as one batch.
        xr = x.unsqueeze(1).expand(b, n, w, n, f).reshape(b * n, w, n, f)
        calr = cal.unsqueeze(1).expand(b, n, w, cal.shape[-1]).reshape(b * n, w, -1)
        m_ctx = torch.zeros_like(xr)
        node_ix = torch.arange(n, device=device).repeat(b)
        m_ctx[torch.arange(b * n, device=device), -1, node_ix] = 1.0
        pred_c = model(xr, m_ctx, calr)
        err_c = ((pred_c[:, -1] - xr[:, -1]) ** 2).mean(dim=-1)   # [b*n, n]
        # Keep only the entry for the node that was actually hidden.
        err_c = err_c[torch.arange(b * n, device=device), node_ix].view(b, n)
        ec.append(err_c.cpu().numpy())

        ts.append(batch["t"].numpy())
        ya.append(batch["y_anom"].numpy())
        yr.append(batch["y_root"].numpy())
        yt.append(batch["y_type"].numpy())
        sev.append(batch["severity"].numpy())

    cat = lambda xs: np.concatenate(xs, axis=0)
    return ScoreBundle(cat(ts), cat(eh), cat(ec), cat(fe),
                       cat(ya), cat(yr), cat(yt), cat(sev))
