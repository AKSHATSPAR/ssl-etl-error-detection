"""Fault taxonomy and row-level injectors.

Each injector corrupts the *rows* produced by one task.  Because downstream
tasks then consume those corrupted rows, propagation through the DAG is an
emergent property of the simulation: a null spike at ``clean_orders`` shows up
as a revenue dip at ``aggregate_daily`` without us writing a single line of
propagation logic.  This is what makes root-cause localisation a genuine
problem: at detection time many nodes look wrong, but only one is the origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .profiler import RunMeta
from .schema import Role, TableSpec

MOJIBAKE = ["Ã©tat", "â\x80\x99", "ï¿½", "\\x00", "N/A#", "??", "NULL_STR"]


@dataclass
class FaultEvent:
    """One injected incident: a fault type, at a node, over a span of batches."""

    fault_type: str
    node: str
    start_batch: int
    duration: int
    severity: float                      # 0..1, scaled per fault type
    params: Dict[str, float] = field(default_factory=dict)

    def active_at(self, batch: int) -> bool:
        return self.start_batch <= batch < self.start_batch + self.duration


@dataclass
class FaultSpec:
    """Registry entry: how a fault behaves and where it may be injected."""

    name: str
    fn: Callable
    channel: str                # "data" | "operational" | "both"
    applies_to: str             # "any" | "has_fk" | "is_join" | "is_source"
    description: str


REGISTRY: Dict[str, FaultSpec] = {}


def register(name: str, channel: str, applies_to: str, description: str):
    def deco(fn: Callable):
        REGISTRY[name] = FaultSpec(name, fn, channel, applies_to, description)
        return fn
    return deco


def _pick(rng: np.random.Generator, items: List, k: int = 1) -> List:
    if not items:
        return []
    idx = rng.choice(len(items), size=min(k, len(items)), replace=False)
    return [items[int(i)] for i in np.atleast_1d(idx)]


# --------------------------------------------------------------------------- #
# Schema faults
# --------------------------------------------------------------------------- #

@register("schema_column_drop", "data", "any",
          "An upstream schema migration silently removes a column.")
def schema_column_drop(df, spec, meta, rng, sev, params):
    droppable = [c.name for c in spec.columns
                 if c.role not in (Role.PK, Role.TIMESTAMP) and c.name in df.columns]
    if not droppable:
        return df, meta
    n_drop = 1 if sev < 0.6 else min(2, len(droppable))
    victims = _pick(rng, droppable, n_drop)
    df = df.drop(columns=victims)
    meta.schema_change_flag = 1.0
    meta.missing_col_rate = len(victims) / max(len(spec.columns), 1)
    params["dropped"] = ",".join(victims)
    return df, meta


@register("schema_column_add", "data", "any",
          "A producer adds unannounced columns; contracts and downstream selects drift.")
def schema_column_add(df, spec, meta, rng, sev, params):
    n_add = 1 + int(sev * 2)
    for i in range(n_add):
        df[f"_unexpected_col_{i}"] = rng.normal(0, 1, len(df))
    meta.schema_change_flag = 1.0
    params["added"] = n_add
    return df, meta


@register("type_mismatch", "data", "any",
          "A numeric column arrives as text; a fraction of casts fail and become nulls.")
def type_mismatch(df, spec, meta, rng, sev, params):
    col = spec.metric.name
    if col not in df.columns:
        return df, meta
    frac = 0.05 + 0.55 * sev
    mask = rng.random(len(df)) < frac
    vals = df[col].astype("object").copy()
    vals[mask] = "NaN_str"
    df[col] = pd.to_numeric(vals, errors="coerce")
    meta.type_mismatch_rate = float(mask.mean())
    meta.schema_change_flag = 1.0
    params["frac"] = float(mask.mean())
    return df, meta


# --------------------------------------------------------------------------- #
# Completeness faults
# --------------------------------------------------------------------------- #

@register("null_spike", "data", "any",
          "A source system starts emitting nulls for one or more fields.")
def null_spike(df, spec, meta, rng, sev, params):
    victims = [c.name for c in spec.columns
               if c.role is not Role.PK and c.name in df.columns]
    if not victims:
        return df, meta
    chosen = _pick(rng, victims, 1 + int(sev * 2))
    frac = 0.08 + 0.7 * sev
    for c in chosen:
        mask = rng.random(len(df)) < frac
        # pandas>=3 refuses to place NaN in an integer column, so widen first.
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype("float64")
        df.loc[mask, c] = np.nan
    params["frac"] = frac
    params["columns"] = ",".join(chosen)
    return df, meta


@register("pk_null", "data", "any",
          "Primary keys arrive null, breaking dedup and idempotent upserts.")
def pk_null(df, spec, meta, rng, sev, params):
    col = spec.pk.name
    if col not in df.columns or len(df) == 0:
        return df, meta
    frac = 0.02 + 0.25 * sev
    mask = rng.random(len(df)) < frac
    df[col] = df[col].astype("float64")
    df.loc[mask, col] = np.nan
    params["frac"] = float(mask.mean())
    return df, meta


@register("duplicate_records", "data", "any",
          "A retried or re-delivered batch duplicates rows, inflating every aggregate.")
def duplicate_records(df, spec, meta, rng, sev, params):
    if len(df) == 0:
        return df, meta
    frac = 0.05 + 0.6 * sev
    n = max(1, int(len(df) * frac))
    dupes = df.sample(n=n, replace=True, random_state=int(rng.integers(0, 2**31 - 1)))
    df = pd.concat([df, dupes], ignore_index=True)
    params["frac"] = frac
    return df, meta


@register("row_loss_truncation", "data", "any",
          "A partial upstream load truncates the batch; rows silently go missing.")
def row_loss_truncation(df, spec, meta, rng, sev, params):
    if len(df) == 0:
        return df, meta
    keep = 1.0 - (0.15 + 0.75 * sev)
    n = max(0, int(len(df) * keep))
    df = df.iloc[:n].copy()
    params["keep_frac"] = keep
    return df, meta


# --------------------------------------------------------------------------- #
# Value / distribution faults
# --------------------------------------------------------------------------- #

@register("unit_change", "data", "any",
          "A currency or unit convention changes upstream (e.g. rupees -> paise).")
def unit_change(df, spec, meta, rng, sev, params):
    col = spec.metric.name
    if col not in df.columns:
        return df, meta
    factor = float(rng.choice([100.0, 0.01, 1000.0, 1 / 1000.0]))
    # Low severity => only part of the batch adopts the new convention.
    frac = 0.3 + 0.7 * sev
    mask = rng.random(len(df)) < frac
    df.loc[mask, col] = pd.to_numeric(df.loc[mask, col], errors="coerce") * factor
    params["factor"] = factor
    params["frac"] = float(mask.mean())
    return df, meta


@register("value_corruption", "data", "any",
          "Sensor/ingest corruption injects extreme or sentinel values into a measure.")
def value_corruption(df, spec, meta, rng, sev, params):
    col = spec.metric.name
    if col not in df.columns or len(df) == 0:
        return df, meta
    frac = 0.01 + 0.15 * sev
    mask = rng.random(len(df)) < frac
    sentinel = float(rng.choice([-9999.0, 1e9, -1.0, 0.0]))
    df.loc[mask, col] = sentinel
    params["frac"] = float(mask.mean())
    params["sentinel"] = sentinel
    return df, meta


@register("category_corruption", "data", "any",
          "Encoding/enum drift produces unseen category values downstream systems reject.")
def category_corruption(df, spec, meta, rng, sev, params):
    cats = [c.name for c in spec.by_role(Role.CATEGORICAL) if c.name in df.columns]
    if not cats or len(df) == 0:
        return df, meta
    col = _pick(rng, cats, 1)[0]
    frac = 0.05 + 0.6 * sev
    mask = rng.random(len(df)) < frac
    bad = rng.choice(MOJIBAKE, size=int(mask.sum()))
    df.loc[mask, col] = bad
    params["column"] = col
    params["frac"] = float(mask.mean())
    return df, meta


# --------------------------------------------------------------------------- #
# Temporal faults
# --------------------------------------------------------------------------- #

@register("timezone_shift", "data", "any",
          "A producer switches from UTC to local time; event times jump by hours.")
def timezone_shift(df, spec, meta, rng, sev, params):
    ts_cols = [c.name for c in spec.by_role(Role.TIMESTAMP) if c.name in df.columns]
    if not ts_cols or len(df) == 0:
        return df, meta
    hours = float(rng.choice([-12, -8, -5.5, 5.5, 8, 12])) * (0.4 + 0.6 * sev)
    col = ts_cols[-1]
    df[col] = pd.to_datetime(df[col], errors="coerce") + pd.to_timedelta(hours, unit="h")
    params["hours"] = hours
    return df, meta


@register("late_arrival", "both", "any",
          "The source lags: the batch carries stale events and the task queues behind SLA.")
def late_arrival(df, spec, meta, rng, sev, params):
    ts_cols = [c.name for c in spec.by_role(Role.TIMESTAMP) if c.name in df.columns]
    lag_h = 2.0 + 40.0 * sev
    if ts_cols and len(df):
        col = ts_cols[-1]
        df[col] = pd.to_datetime(df[col], errors="coerce") - pd.to_timedelta(lag_h, unit="h")
    meta.queue_delay_sec += lag_h * 3600.0 * 0.25
    meta.runtime_sec *= 1.0 + 0.5 * sev
    params["lag_hours"] = lag_h
    return df, meta


# --------------------------------------------------------------------------- #
# Join / integrity faults
# --------------------------------------------------------------------------- #

@register("fk_break", "data", "has_fk",
          "Key formatting drifts upstream, so joins orphan rows or drop them entirely.")
def fk_break(df, spec, meta, rng, sev, params):
    fks = [c.name for c in spec.by_role(Role.FK) if c.name in df.columns]
    if not fks or len(df) == 0:
        return df, meta
    col = fks[0]
    frac = 0.1 + 0.7 * sev
    mask = rng.random(len(df)) < frac
    # Offsetting the key by a large constant guarantees it matches nothing.
    df.loc[mask, col] = pd.to_numeric(df.loc[mask, col], errors="coerce") + 10_000_000
    params["frac"] = float(mask.mean())
    return df, meta


@register("join_fanout", "data", "is_join",
          "Duplicate dimension keys turn a 1:1 join into 1:N and explode the fact table.")
def join_fanout(df, spec, meta, rng, sev, params):
    if len(df) == 0:
        return df, meta
    factor = 1.0 + 0.5 + 3.5 * sev
    n_extra = int(len(df) * (factor - 1.0))
    if n_extra <= 0:
        return df, meta
    dupes = df.sample(n=n_extra, replace=True,
                      random_state=int(rng.integers(0, 2**31 - 1)))
    df = pd.concat([df, dupes], ignore_index=True)
    meta.runtime_sec *= factor ** 0.6
    meta.mem_mb *= factor ** 0.7
    params["factor"] = factor
    return df, meta


# --------------------------------------------------------------------------- #
# Operational-only fault (the hard case: the data is perfectly fine)
# --------------------------------------------------------------------------- #

@register("resource_degradation", "operational", "any",
          "Cluster contention: the task retries and blows its SLA while its output stays valid.")
def resource_degradation(df, spec, meta, rng, sev, params):
    meta.runtime_sec *= 1.8 + 6.0 * sev
    meta.queue_delay_sec += 60.0 + 2400.0 * sev
    meta.retry_count += int(1 + 3 * sev)
    meta.cpu_util = min(1.0, meta.cpu_util * (1.3 + 0.6 * sev))
    meta.mem_mb *= 1.2 + 1.5 * sev
    params["slowdown"] = 1.8 + 6.0 * sev
    return df, meta


#: The symptom family each fault is *mechanistically* expected to move, used as
#: ground truth when scoring the explainability output. Sets rather than single
#: values because several faults legitimately present through more than one
#: family: a duplicated batch inflates both the duplicate rate (completeness)
#: and the row ratio (operational), and neither reading is wrong.
SYMPTOM_FAMILY: Dict[str, set] = {
    "schema_column_drop": {"schema"},
    "schema_column_add": {"schema"},
    "type_mismatch": {"schema", "completeness"},
    "null_spike": {"completeness"},
    "pk_null": {"completeness"},
    "duplicate_records": {"completeness", "operational"},
    "row_loss_truncation": {"operational"},
    "unit_change": {"metric_distribution"},
    "value_corruption": {"metric_distribution"},
    "category_corruption": {"categorical_distribution"},
    "timezone_shift": {"temporal"},
    "late_arrival": {"temporal", "operational"},
    "fk_break": {"integrity"},
    "join_fanout": {"completeness", "operational", "metric_distribution"},
    "resource_degradation": {"operational"},
}

FAULT_TYPES: List[str] = sorted(REGISTRY.keys())
FAULT_TYPE_INDEX: Dict[str, int] = {n: i for i, n in enumerate(FAULT_TYPES)}
# Index 0 is reserved for "no fault" in the label tensors.
FAULT_LABELS: List[str] = ["none"] + FAULT_TYPES


def applicable_faults(node_flags: Dict[str, bool]) -> List[str]:
    """Fault types that may legally be injected at a node with the given flags."""
    out = []
    for name, spec in REGISTRY.items():
        if spec.applies_to == "any":
            out.append(name)
        elif node_flags.get(spec.applies_to, False):
            out.append(name)
    return sorted(out)


def apply_fault(df: pd.DataFrame, spec: TableSpec, meta: RunMeta,
                event: FaultEvent, rng: np.random.Generator):
    """Dispatch one fault event onto a task's output."""
    fs = REGISTRY[event.fault_type]
    df = df.copy()
    df, meta = fs.fn(df, spec, meta, rng, event.severity, event.params)
    return df, meta
