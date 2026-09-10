"""Derive a fixed-width data-quality feature vector from a table's raw rows.

Every node in the DAG emits a vector of exactly ``N_FEATURES`` numbers, so a
whole pipeline run at time *t* is a matrix of shape ``[n_nodes, N_FEATURES]`` and
a window of runs is a tensor ``[T, n_nodes, N_FEATURES]``.

Design notes
------------
* Statistics are aggregated **by column role**, not by column name.  That is the
  trick that keeps the width constant even though ``customers`` and
  ``daily_metrics`` share no columns.
* Heavy-tailed quantities (row counts, runtimes, memory, staleness) are stored
  as ``log1p`` so that a 10x row explosion is a bounded jump rather than an
  outlier that dominates every loss term.
* Nothing here is normalised against a reference window.  All z-scoring happens
  later, fitted on the *training split only*, to avoid leaking test statistics
  into the features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .schema import Role, TableSpec

# --------------------------------------------------------------------------- #
# Feature vocabulary
# --------------------------------------------------------------------------- #

OPERATIONAL_FEATURES = [
    "log_rows_in",
    "log_rows_out",
    "row_ratio",
    "log_runtime_sec",
    "cpu_util",
    "log_mem_mb",
    "retry_count",
    "log_queue_delay_sec",
]

SCHEMA_FEATURES = [
    "n_columns",
    "schema_change_flag",
    "type_mismatch_rate",
    "missing_col_rate",
]

COMPLETENESS_FEATURES = [
    "null_rate_mean",
    "null_rate_max",
    "pk_null_rate",
    "dup_pk_rate",
    "exact_dup_row_rate",
]

METRIC_FEATURES = [
    "metric_mean",
    "metric_std",
    "metric_min",
    "metric_max",
    "metric_p99",
    "metric_zero_rate",
    "metric_negative_rate",
    "metric_outlier_rate",
]

NUMERIC_FEATURES = [
    "numeric_null_rate",
    "numeric_outlier_rate",
    "numeric_negative_rate",
]

CATEGORICAL_FEATURES = [
    "cat_cardinality_mean",
    "cat_unseen_rate",
    "cat_top1_share",
    "cat_entropy_norm",
]

TEMPORAL_FEATURES = [
    "log_ts_lag_p50",
    "log_ts_lag_max",
    "future_ts_rate",
    "out_of_window_rate",
    "ts_null_rate",
]

INTEGRITY_FEATURES = [
    "orphan_fk_rate",
]

FEATURE_NAMES: List[str] = (
    OPERATIONAL_FEATURES
    + SCHEMA_FEATURES
    + COMPLETENESS_FEATURES
    + METRIC_FEATURES
    + NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + TEMPORAL_FEATURES
    + INTEGRITY_FEATURES
)
N_FEATURES = len(FEATURE_NAMES)
FEATURE_INDEX: Dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}

# Feature-family membership, used by the root-cause explainer to translate a
# feature-attribution vector into a human-readable symptom group.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "operational": OPERATIONAL_FEATURES,
    "schema": SCHEMA_FEATURES,
    "completeness": COMPLETENESS_FEATURES,
    "metric_distribution": METRIC_FEATURES,
    "numeric_distribution": NUMERIC_FEATURES,
    "categorical_distribution": CATEGORICAL_FEATURES,
    "temporal": TEMPORAL_FEATURES,
    "integrity": INTEGRITY_FEATURES,
}


@dataclass
class RunMeta:
    """Operational telemetry for one task execution (the non-data signal)."""

    rows_in: int = 0
    runtime_sec: float = 0.0
    cpu_util: float = 0.0
    mem_mb: float = 0.0
    retry_count: int = 0
    queue_delay_sec: float = 0.0
    # Populated by fault injectors that mutate the schema.
    type_mismatch_rate: float = 0.0
    schema_change_flag: float = 0.0
    missing_col_rate: float = 0.0


def _safe(x: float) -> float:
    """Map NaN/inf to 0.0 — an empty or fully-null column must not poison the tensor."""
    if x is None:
        return 0.0
    x = float(x)
    if not np.isfinite(x):
        return 0.0
    return x


def _outlier_rate(values: np.ndarray) -> float:
    """Fraction of points beyond 5 robust standard deviations (median +/- 5*MAD)."""
    v = values[np.isfinite(values)]
    if v.size < 8:
        return 0.0
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad <= 1e-12:
        # Degenerate spread: anything not equal to the median is an outlier.
        return float(np.mean(np.abs(v - med) > 1e-9))
    return float(np.mean(np.abs(v - med) > 5.0 * 1.4826 * mad))


def _entropy_norm(counts: np.ndarray, n_expected: int) -> float:
    total = counts.sum()
    if total <= 0 or counts.size <= 1:
        return 0.0
    p = counts / total
    ent = float(-(p * np.log(p + 1e-12)).sum())
    denom = np.log(max(n_expected, 2))
    return ent / denom


def profile_table(
    df: pd.DataFrame,
    spec: TableSpec,
    meta: RunMeta,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    fk_reference: Optional[Sequence] = None,
) -> np.ndarray:
    """Compute the ``N_FEATURES``-dim data-quality profile of one table snapshot.

    ``fk_reference`` is the set of parent keys used for the referential-integrity
    check; pass ``None`` for tables with no foreign key.
    """
    feats = np.zeros(N_FEATURES, dtype=np.float64)

    def put(name: str, value: float) -> None:
        feats[FEATURE_INDEX[name]] = _safe(value)

    n_rows = len(df)
    cols = list(df.columns)

    # ---------------- operational ---------------- #
    put("log_rows_in", np.log1p(max(meta.rows_in, 0)))
    put("log_rows_out", np.log1p(max(n_rows, 0)))
    put("row_ratio", n_rows / max(meta.rows_in, 1))
    put("log_runtime_sec", np.log1p(max(meta.runtime_sec, 0.0)))
    put("cpu_util", meta.cpu_util)
    put("log_mem_mb", np.log1p(max(meta.mem_mb, 0.0)))
    put("retry_count", meta.retry_count)
    put("log_queue_delay_sec", np.log1p(max(meta.queue_delay_sec, 0.0)))

    # ---------------- schema ---------------- #
    put("n_columns", len(cols))
    put("schema_change_flag", meta.schema_change_flag)
    put("type_mismatch_rate", meta.type_mismatch_rate)
    put("missing_col_rate", meta.missing_col_rate)

    if n_rows == 0:
        # A wiped batch is itself a strong signal; the zeros above already carry it.
        return feats.astype(np.float32)

    # ---------------- completeness ---------------- #
    null_rates = [float(df[c].isna().mean()) for c in cols] or [0.0]
    put("null_rate_mean", float(np.mean(null_rates)))
    put("null_rate_max", float(np.max(null_rates)))

    pk_name = spec.pk.name
    if pk_name in df.columns:
        pk = df[pk_name]
        put("pk_null_rate", float(pk.isna().mean()))
        put("dup_pk_rate", float(pk.duplicated(keep="first").mean()))
    put("exact_dup_row_rate", float(df.duplicated(keep="first").mean()))

    # ---------------- primary metric distribution ---------------- #
    metric_name = spec.metric.name
    if metric_name in df.columns:
        m = pd.to_numeric(df[metric_name], errors="coerce").to_numpy(dtype=np.float64)
        finite = m[np.isfinite(m)]
        if finite.size:
            put("metric_mean", float(np.mean(finite)))
            put("metric_std", float(np.std(finite)))
            put("metric_min", float(np.min(finite)))
            put("metric_max", float(np.max(finite)))
            put("metric_p99", float(np.percentile(finite, 99)))
            put("metric_zero_rate", float(np.mean(np.abs(finite) < 1e-9)))
            put("metric_negative_rate", float(np.mean(finite < 0)))
            put("metric_outlier_rate", _outlier_rate(finite))

    # ---------------- secondary numeric columns ---------------- #
    num_cols = [c.name for c in spec.by_role(Role.NUMERIC) if c.name in df.columns]
    if num_cols:
        nulls, outs, negs = [], [], []
        for c in num_cols:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
            nulls.append(float(np.mean(~np.isfinite(v))))
            fv = v[np.isfinite(v)]
            outs.append(_outlier_rate(fv))
            negs.append(float(np.mean(fv < 0)) if fv.size else 0.0)
        put("numeric_null_rate", float(np.mean(nulls)))
        put("numeric_outlier_rate", float(np.mean(outs)))
        put("numeric_negative_rate", float(np.mean(negs)))

    # ---------------- categorical distribution ---------------- #
    cat_specs = [c for c in spec.by_role(Role.CATEGORICAL) if c.name in df.columns]
    if cat_specs:
        cards, unseen, top1, ents = [], [], [], []
        for c in cat_specs:
            s = df[c.name].astype("object")
            vc = s.value_counts(dropna=True)
            cards.append(float(vc.size))
            allowed = set(c.categories or [])
            if allowed:
                bad = float(sum(cnt for val, cnt in vc.items() if val not in allowed))
                unseen.append(bad / max(n_rows, 1))
            else:
                unseen.append(0.0)
            top1.append(float(vc.iloc[0] / max(n_rows, 1)) if vc.size else 0.0)
            ents.append(_entropy_norm(vc.to_numpy(dtype=np.float64), len(allowed) or vc.size))
        put("cat_cardinality_mean", float(np.mean(cards)))
        put("cat_unseen_rate", float(np.mean(unseen)))
        put("cat_top1_share", float(np.mean(top1)))
        put("cat_entropy_norm", float(np.mean(ents)))

    # ---------------- temporal freshness ---------------- #
    ts_specs = [c for c in spec.by_role(Role.TIMESTAMP) if c.name in df.columns]
    # Freshness is judged on the *event-time* column, i.e. the last timestamp.
    if ts_specs:
        ts = pd.to_datetime(df[ts_specs[-1].name], errors="coerce")
        put("ts_null_rate", float(ts.isna().mean()))
        valid = ts.dropna()
        if len(valid):
            lag = (window_end - valid).dt.total_seconds().to_numpy(dtype=np.float64)
            put("log_ts_lag_p50", np.log1p(max(float(np.percentile(lag, 50)), 0.0)))
            put("log_ts_lag_max", np.log1p(max(float(np.max(lag)), 0.0)))
            put("future_ts_rate", float(np.mean(valid > window_end)))
            in_window = (valid >= window_start) & (valid <= window_end)
            put("out_of_window_rate", float(1.0 - np.mean(in_window)))

    # ---------------- referential integrity ---------------- #
    fk_specs = [c for c in spec.by_role(Role.FK) if c.name in df.columns]
    if fk_specs and fk_reference is not None:
        ref = pd.Index(pd.Series(list(fk_reference)).dropna().unique())
        keys = df[fk_specs[0].name]
        put("orphan_fk_rate", float((~keys.isin(ref)).mean()))

    return feats.astype(np.float32)
