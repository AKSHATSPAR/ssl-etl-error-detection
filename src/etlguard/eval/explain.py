"""Explainability: what evidence supported the alert, and where it came from.

Three attribution methods, all producing a map over the input window so they can
be compared on the same footing:

1. **Reconstruction error.** Per-feature squared error at the flagged task. The
   direct read of what the model could not predict.
2. **Integrated Gradients.** Attributes the anomaly score back to every
   ``(timestep, node, feature)`` cell in the window, so it can credit evidence in
   *other* tasks and earlier runs, which reconstruction error cannot.
3. **DAG attention rollout.** Which upstream tasks the structural attention
   actually looked at.

The point of scoring these rather than plotting them is that the simulator knows
which symptom family each fault should move. In the published literature
attribution is nearly always displayed and almost never checked, because there is
usually no ground truth to check it against.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


def _anomaly_target(model, x: torch.Tensor, mask: torch.Tensor,
                    cal: torch.Tensor, node: torch.Tensor) -> torch.Tensor:
    """Squared error at the queried node's final timestep, per sample."""
    pred = model(x, mask, cal)
    err = (pred[:, -1] - x[:, -1]) ** 2                     # [B, N, F]
    rows = torch.arange(x.shape[0], device=x.device)
    return err[rows, node].sum(dim=-1)                      # [B]


def integrated_gradients(model, x: torch.Tensor, mask: torch.Tensor,
                         cal: torch.Tensor, node: torch.Tensor,
                         steps: int = 32,
                         baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Integrated Gradients of the anomaly score with respect to the window.

    The baseline is the all-zero window. In the robust-scaled space zero is the
    healthy median of every feature, so the baseline literally means "a
    completely typical run", which is the counterfactual the explanation should
    be measured against. A zero baseline is often arbitrary; here it is not.

    Returns ``[B, W, N, F]`` attributions.
    """
    if baseline is None:
        baseline = torch.zeros_like(x)
    total = torch.zeros_like(x)
    # Riemann midpoint rule: fewer steps for the same accuracy than left edges.
    for k in range(steps):
        alpha = (k + 0.5) / steps
        point = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        target = _anomaly_target(model, point, mask, cal, node).sum()
        grad, = torch.autograd.grad(target, point)
        total = total + grad
    return (x - baseline) * total / steps


@torch.no_grad()
def attention_rollout(model, x: torch.Tensor, mask: torch.Tensor,
                      cal: torch.Tensor) -> Optional[np.ndarray]:
    """Structural attention at the final timestep, averaged over layers and heads.

    Returns ``[B, N, N]`` where entry ``[b, i, j]`` is how much task *i* attended
    to task *j*, or None if the model stored no attention.
    """
    model(x, mask, cal, store_attn=True)
    a = model.attention_over_nodes()          # [B*W, N, N]
    if a is None:
        return None
    b, w, n = x.shape[0], x.shape[1], x.shape[2]
    return a.view(b, w, n, n)[:, -1].cpu().numpy()


def family_scores(attr_node: np.ndarray,
                  group_index: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
    """Collapse a per-feature attribution vector into the eight symptom families."""
    fams = list(group_index)
    vals = np.array([np.abs(attr_node[group_index[f]]).mean() for f in fams])
    return fams, vals


def score_attribution(attr: np.ndarray, y_root: np.ndarray, y_type: np.ndarray,
                      group_index: Dict[str, np.ndarray],
                      expected: Dict[int, set]) -> Dict[str, float]:
    """Fraction of incidents whose top symptom family is the expected one.

    ``attr`` is ``[M, N, F]``: an attribution map over nodes and features, read at
    the true origin so that attribution is judged on its own terms rather than
    being penalised for a localization mistake made earlier in the pipeline.
    """
    fams = list(group_index)
    hits, total, per_type = 0, 0, {}
    for i in range(len(y_type)):
        ft = int(y_type[i])
        want = expected.get(ft)
        roots = np.where(y_root[i] > 0)[0]
        if ft == 0 or want is None or roots.size == 0:
            continue
        _, vals = family_scores(attr[i, int(roots[0])], group_index)
        got = fams[int(np.argmax(vals))]
        ok = int(got in want)
        hits += ok
        total += 1
        per_type.setdefault(ft, []).append(ok)
    return {"accuracy": hits / max(total, 1), "n_scored": total,
            "per_type": {k: float(np.mean(v)) for k, v in per_type.items()}}
