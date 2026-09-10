"""Detection, localization and attribution metrics.

Deliberate omission: **point-adjusted F1 is not the headline metric.** Under
point adjustment, flagging a single timestep inside an incident credits you with
the whole incident, and random noise scores above 0.9 on the standard
benchmarks. It is reported here only alongside the honest number, clearly
labelled, because the literature quotes it and reviewers look for it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

def best_f1(y: np.ndarray, s: np.ndarray) -> Dict[str, float]:
    """Best achievable F1 over all thresholds (an oracle-threshold upper bound)."""
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    fn = y.sum() - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    i = int(np.argmax(f1))
    return {"best_f1": float(f1[i]),
            "precision_at_best_f1": float(tp[i] / max(tp[i] + fp[i], 1)),
            "recall_at_best_f1": float(tp[i] / max(y.sum(), 1)),
            "threshold": float(s[order][i])}


def f1_at_budget(y: np.ndarray, s: np.ndarray, budget: float) -> Dict[str, float]:
    """F1 when only the top ``budget`` fraction of runs may raise an alert.

    This is the number an on-call rotation actually feels: you get so many alerts
    per week and no more, and the question is how many real incidents are in them.
    """
    k = max(1, int(round(budget * len(s))))
    thresh = np.sort(s)[-k]
    pred = (s >= thresh).astype(np.int64)
    tp = int((pred & y).sum())
    prec = tp / max(pred.sum(), 1)
    rec = tp / max(y.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {f"precision@{budget:g}": prec, f"recall@{budget:g}": rec,
            f"f1@{budget:g}": f1}


def point_adjusted_f1(y: np.ndarray, s: np.ndarray) -> float:
    """Point-adjusted best-F1. Reported for comparability only, see module docstring."""
    order = np.argsort(-s)
    best = 0.0
    for k in range(1, len(s) + 1, max(1, len(s) // 200)):
        pred = np.zeros_like(y)
        pred[order[:k]] = 1
        adj = pred.copy()
        # Any hit inside a contiguous anomalous run marks the entire run detected.
        i = 0
        while i < len(y):
            if y[i]:
                j = i
                while j < len(y) and y[j]:
                    j += 1
                if pred[i:j].any():
                    adj[i:j] = 1
                i = j
            else:
                i += 1
        tp = int((adj & y).sum())
        prec = tp / max(adj.sum(), 1)
        rec = tp / max(y.sum(), 1)
        if prec + rec > 0:
            best = max(best, 2 * prec * rec / (prec + rec))
    return float(best)


def detection_metrics(y: np.ndarray, s: np.ndarray, budget: float = 0.01,
                      include_point_adjust: bool = True) -> Dict[str, float]:
    if y.sum() == 0 or y.sum() == len(y):
        return {"auroc": float("nan"), "auprc": float("nan")}
    out = {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "base_rate": float(y.mean()),
    }
    out.update(best_f1(y, s))
    out.update(f1_at_budget(y, s, budget))
    if include_point_adjust:
        out["point_adjusted_f1_NOT_HEADLINE"] = point_adjusted_f1(y, s)
    return out


def detection_delay(y_type: np.ndarray, s: np.ndarray, threshold: float
                    ) -> Dict[str, float]:
    """Runs elapsed between an incident starting and the first alert inside it."""
    fired = s >= threshold
    delays, missed = [], 0
    i = 0
    while i < len(y_type):
        if y_type[i] > 0:
            j = i
            while j < len(y_type) and y_type[j] > 0:
                j += 1
            hit = np.where(fired[i:j])[0]
            if hit.size:
                delays.append(int(hit[0]))
            else:
                missed += 1
            i = j
        else:
            i += 1
    n = len(delays) + missed
    return {"mean_detection_delay": float(np.mean(delays)) if delays else float("nan"),
            "incidents": n,
            "incidents_missed": missed,
            "incident_recall": float(len(delays) / max(n, 1))}


# --------------------------------------------------------------------------- #
# Root-cause localization
# --------------------------------------------------------------------------- #

def localization_metrics(y_root: np.ndarray, r: np.ndarray,
                         ks: Sequence[int] = (1, 3)) -> Dict[str, float]:
    """Hit@k and MRR over the candidate nodes, on incident runs only.

    Ranking is evaluated only where an incident actually occurred; asking "which
    node is at fault" of a healthy run is not a meaningful question.
    """
    m = y_root.sum(axis=1) > 0
    if m.sum() == 0:
        return {f"hit@{k}": float("nan") for k in ks}
    yr, rr = y_root[m], r[m]
    order = np.argsort(-rr, axis=1)
    out: Dict[str, float] = {"n_incidents": int(m.sum())}
    for k in ks:
        topk = order[:, :k]
        hit = np.take_along_axis(yr, topk, axis=1).max(axis=1)
        out[f"hit@{k}"] = float(hit.mean())
    # MRR against the best-ranked true origin (incidents may have >1).
    ranks = []
    for i in range(len(yr)):
        pos = np.where(yr[i][order[i]] > 0)[0]
        ranks.append(1.0 / (pos[0] + 1) if pos.size else 0.0)
    out["mrr"] = float(np.mean(ranks))
    return out


def localization_by_group(y_root: np.ndarray, r: np.ndarray, groups: np.ndarray,
                          labels: List[str], ks=(1, 3)) -> Dict[str, Dict]:
    """Localization broken down by fault type or severity bucket.

    A single mean hides that low-severity incidents are near-impossible and
    high-severity ones are near-trivial; the breakdown is the honest report.
    """
    out = {}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 3:
            continue
        name = labels[int(g)] if int(g) < len(labels) else str(g)
        out[name] = localization_metrics(y_root[m], r[m], ks)
    return out


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #

def attribution_accuracy(feat_err: np.ndarray, y_root: np.ndarray,
                         y_type: np.ndarray, group_index: Dict[str, np.ndarray],
                         expected: Dict[int, set]) -> Dict[str, float]:
    """Does the top-scoring symptom family match the fault's known symptom?

    The simulator knows which family each fault type is supposed to move (a
    unit change moves the metric distribution, a timezone shift moves freshness),
    so attribution can be *scored* rather than merely displayed.
    """
    fams = list(group_index.keys())
    hits, total, per_type = 0, 0, {}
    for i in range(len(y_type)):
        ft = int(y_type[i])
        if ft == 0:
            continue
        roots = np.where(y_root[i] > 0)[0]
        if roots.size == 0:
            continue
        node = int(roots[0])
        want = expected.get(ft)
        if want is None:
            continue
        scores = np.array([feat_err[i, node, group_index[f]].mean() for f in fams])
        got = fams[int(np.argmax(scores))]
        ok = int(got in want)
        hits += ok
        total += 1
        per_type.setdefault(ft, []).append(ok)
    return {"attribution_accuracy": hits / max(total, 1),
            "n_scored": total,
            "per_type": {k: float(np.mean(v)) for k, v in per_type.items()}}
