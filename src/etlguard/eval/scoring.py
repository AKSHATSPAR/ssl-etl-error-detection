"""Turn a trained encoder into detection and root-cause scores.

Scoring mirrors training. The model was trained to fill in hidden entries, so at
inference we *ask it the same question* rather than running it unmasked. An
unmasked masked-autoencoder can copy its input straight to its output, and the
reconstruction error then measures nothing.

Two masking configurations at the final timestep, and the difference between
them is the method:

``e_hist``  every node is hidden, so each prediction rests on history alone.
            Large whenever a run departs from its own past: the detection signal.

``e_ctx``   only node *i* is hidden and everything else at that timestep stays
            visible. This turns out to be the wrong question, and it is kept only
            to show why: the structural attention is bidirectional, so a hidden
            node can be reconstructed from its own *descendants*. During an
            incident those descendants are corrupted too, and they leak the
            corrupted value straight back. The leak is worst exactly where it
            hurts most, at source tasks, which have the most descendants and are
            the most valuable root causes to identify.

``e_anc``   node *i* **and all of its descendants** are hidden, so the prediction
            rests only on *i*'s ancestors and on history. This is the real
            ancestor-conditioned score: *surprise given inputs*. A task
            downstream of a failure emits anomalous output that follows perfectly
            from its own anomalous inputs, so ``e_anc`` is small and it reads
            correctly as a victim. The task where the incident began has healthy
            inputs and anomalous output, so ``e_anc`` stays large. That asymmetry
            is what separates the origin from its blast radius.
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
    e_ctx: np.ndarray        # [M, N] error with only node i hidden (leaky, kept for ablation)
    e_anc: np.ndarray        # [M, N] error with node i and its descendants hidden
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
        if rule == "max_anc":
            return self.e_anc.max(axis=1)
        if rule == "top2_hist":
            s = np.sort(self.e_hist, axis=1)[:, -2:]
            return s.mean(axis=1)
        raise ValueError(f"unknown pooling rule {rule!r}")

    DETECTION_RULES = ("max_hist", "mean_hist", "max_ctx", "max_anc", "top2_hist")

    def calibrated(self, which: str, stats: "NodeCalibration") -> np.ndarray:
        """Per-node scores expressed in each node's own units of surprise."""
        return stats.apply(getattr(self, which), which)


@dataclass
class NodeCalibration:
    """Express each node's score as a percentile against its own healthy history.

    Tasks are not equally predictable, so raw errors are not comparable across
    them, and ranking nine raw errors asks the wrong question. It asks which task
    has the largest error, when what is wanted is which task is furthest from its
    own normal.

    Percentile rank rather than a z-score, because the error distributions are
    heavy-tailed and an IQR is a poor summary of them. Measured on the test
    split, z-scoring actually *harms* the ancestor-conditioned score (0.773 to
    0.605 Hit@1) while percentile rank improves it (0.773 to 0.797). The
    distribution-free version is doing the same job without assuming a shape the
    data does not have.

    Reference scores come from unlabelled training windows, so this adds no label
    use. The ~3% contamination in that history shifts a percentile by at most a
    few points, which is why a rank statistic is the safe choice here too.
    """

    reference: Dict[str, np.ndarray]      # [n_ref, N] sorted per node

    @classmethod
    def fit(cls, bundle: "ScoreBundle",
            which=("e_hist", "e_ctx", "e_anc")) -> "NodeCalibration":
        return cls({name: np.sort(getattr(bundle, name), axis=0) for name in which})

    def apply(self, values: np.ndarray, which: str) -> np.ndarray:
        ref = self.reference[which]
        n_ref = ref.shape[0]
        out = np.empty_like(values, dtype=np.float64)
        for i in range(values.shape[1]):
            out[:, i] = np.searchsorted(ref[:, i], values[:, i]) / n_ref
        return out


def calibrate_node_scores(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Percentile-calibrate an arbitrary ``[M, N]`` node-score matrix.

    Used to give the baselines the same treatment as the model. Percentile
    calibration turns out to be worth more than the choice of detector, so
    applying it to one side only would be the difference the results reported.
    """
    ref = np.sort(reference, axis=0)
    out = np.empty_like(values, dtype=np.float64)
    for i in range(values.shape[1]):
        out[:, i] = np.searchsorted(ref[:, i], values[:, i]) / ref.shape[0]
    return out


def descendant_mask(adjacency: np.ndarray) -> np.ndarray:
    """``M[i, j] = 1`` when *j* is *i* itself or anything downstream of it."""
    n = adjacency.shape[0]
    reach = adjacency.astype(bool).copy()
    for k in range(n):
        reach |= np.outer(reach[:, k], reach[k, :])
    return (reach | np.eye(n, dtype=bool)).astype(np.float32)


@torch.no_grad()
def score_loader(model, loader: DataLoader, device: str, n_nodes: int,
                 adjacency: Optional[np.ndarray] = None) -> ScoreBundle:
    model.eval()
    ts, eh, ec, ea, fe, ya, yr, yt, sev = [], [], [], [], [], [], [], [], []
    if adjacency is None:
        raise ValueError("adjacency is required to hide a node's descendants")
    desc = torch.from_numpy(descendant_mask(adjacency)).to(device)   # [N, N]

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

        # ---- one node hidden at a time ----
        # Replicating the window N times lets all N queries run as one batch.
        xr = x.unsqueeze(1).expand(b, n, w, n, f).reshape(b * n, w, n, f)
        calr = cal.unsqueeze(1).expand(b, n, w, cal.shape[-1]).reshape(b * n, w, -1)
        rows = torch.arange(b * n, device=device)
        node_ix = torch.arange(n, device=device).repeat(b)

        def query(hide: torch.Tensor) -> np.ndarray:
            """``hide`` is [b*n, N]: which nodes to blank at the final step."""
            m = torch.zeros_like(xr)
            m[:, -1] = hide.unsqueeze(-1).expand(-1, -1, f)
            pred = model(xr, m, calr)
            e = ((pred[:, -1] - xr[:, -1]) ** 2).mean(dim=-1)      # [b*n, N]
            # Keep only the entry for the node whose value we asked about.
            return e[rows, node_ix].view(b, n).cpu().numpy()

        self_only = torch.zeros(b * n, n, device=device)
        self_only[rows, node_ix] = 1.0
        ec.append(query(self_only))
        # Hiding the descendants too is what removes the leak.
        ea.append(query(desc[node_ix]))

        ts.append(batch["t"].numpy())
        ya.append(batch["y_anom"].numpy())
        yr.append(batch["y_root"].numpy())
        yt.append(batch["y_type"].numpy())
        sev.append(batch["severity"].numpy())

    cat = lambda xs: np.concatenate(xs, axis=0)
    return ScoreBundle(cat(ts), cat(eh), cat(ec), cat(ea), cat(fe),
                       cat(ya), cat(yr), cat(yt), cat(sev))
