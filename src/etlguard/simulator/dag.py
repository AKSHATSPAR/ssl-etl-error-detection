"""The simulated ETL pipeline: nine tasks wired into a DAG with real transforms.

Each node's ``execute`` runs an honest transformation (cast, filter, dedup, join,
group-by) over the rows handed to it by its parents.  Transforms are written
defensively, because a column may be missing after an upstream fault dropped it,
or full of nulls after an upstream cast failed, exactly as a production job
degrades rather than crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import schema as S
from .schema import Role, TableSpec

# Conversion rates used by the "normalise currency" step of clean_orders.
FX_TO_INR = {"INR": 1.0, "USD": 83.0, "GBP": 105.0, "EUR": 90.0}


def _col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    """Fetch a column, or a same-length column of ``default`` if it went missing."""
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index, name=name)


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(_col(df, name), errors="coerce")


def _ensure_key(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Guarantee a join key exists and is numeric.

    When an upstream schema fault drops the key, we materialise it as all-NaN so
    the merge still runs and simply matches nothing, which is the outcome a real job
    produces when it joins on a column that stopped arriving.
    """
    out = df.copy()
    if key not in out.columns:
        out[key] = np.nan
    out[key] = pd.to_numeric(out[key], errors="coerce")
    return out


@dataclass
class Node:
    name: str
    spec: TableSpec
    parents: List[str]
    execute: Callable[[List[pd.DataFrame], dict], pd.DataFrame]
    is_source: bool = False
    is_join: bool = False
    # Baseline operational cost model: seconds per 1000 rows, and memory per 1000 rows.
    base_runtime: float = 4.0
    base_mem: float = 220.0

    @property
    def has_fk(self) -> bool:
        return bool(self.spec.by_role(Role.FK))

    @property
    def flags(self) -> Dict[str, bool]:
        return {"is_source": self.is_source, "is_join": self.is_join, "has_fk": self.has_fk}


# --------------------------------------------------------------------------- #
# Transform implementations
# --------------------------------------------------------------------------- #

def _clean_customers(inputs: List[pd.DataFrame], ctx: dict) -> pd.DataFrame:
    """Drop PII, coerce types, de-duplicate on the natural key."""
    src = inputs[0]
    out = pd.DataFrame(index=src.index)
    out["customer_id"] = _col(src, "customer_id")
    out["lifetime_value"] = _num(src, "lifetime_value")
    out["age"] = _num(src, "age").clip(lower=0, upper=120)
    out["country"] = _col(src, "country", "UNKNOWN")
    out["segment"] = _col(src, "segment", "UNKNOWN")
    out["signup_ts"] = pd.to_datetime(_col(src, "signup_ts"), errors="coerce")
    # A real job dedups on the key; nulls survive as their own group.
    return out.drop_duplicates(subset=["customer_id"], keep="last").reset_index(drop=True)


def _clean_orders(inputs: List[pd.DataFrame], ctx: dict) -> pd.DataFrame:
    """Normalise currency to INR and standardise the status enum."""
    src = inputs[0]
    out = pd.DataFrame(index=src.index)
    out["order_id"] = _col(src, "order_id")
    out["customer_id"] = _col(src, "customer_id")
    fx = _col(src, "currency", "INR").map(FX_TO_INR).astype("float64").fillna(1.0)
    out["amount_inr"] = _num(src, "amount") * fx
    out["quantity"] = _num(src, "quantity")
    out["status"] = _col(src, "status", "placed").astype("object")
    out["channel"] = _col(src, "channel", "web")
    out["order_ts"] = pd.to_datetime(_col(src, "order_ts"), errors="coerce")
    return out.reset_index(drop=True)


def _join_orders_customers(inputs: List[pd.DataFrame], ctx: dict) -> pd.DataFrame:
    """Left-join the customer dimension onto the order fact."""
    orders = _ensure_key(inputs[0], "customer_id")
    customers = _ensure_key(inputs[1], "customer_id")
    dim_cols = [c for c in ["customer_id", "country", "segment", "lifetime_value"]
                if c in customers.columns]
    dim = customers[dim_cols]
    merged = orders.merge(dim, on="customer_id", how="left", suffixes=("", "_dim"))
    out = pd.DataFrame(index=merged.index)
    out["order_id"] = _col(merged, "order_id")
    out["customer_id"] = _col(merged, "customer_id")
    out["amount_inr"] = _num(merged, "amount_inr")
    out["quantity"] = _num(merged, "quantity")
    out["lifetime_value"] = _num(merged, "lifetime_value")
    out["status"] = _col(merged, "status", "placed")
    out["channel"] = _col(merged, "channel", "web")
    out["country"] = _col(merged, "country", "UNKNOWN").fillna("UNKNOWN")
    out["segment"] = _col(merged, "segment", "UNKNOWN").fillna("UNKNOWN")
    out["order_ts"] = pd.to_datetime(_col(merged, "order_ts"), errors="coerce")
    return out.reset_index(drop=True)


def _join_payments(inputs: List[pd.DataFrame], ctx: dict) -> pd.DataFrame:
    """Attach payment facts; unpaid orders are kept with a sentinel method."""
    enriched = _ensure_key(inputs[0], "order_id")
    payments = _ensure_key(inputs[1], "order_id")
    pay_cols = [c for c in ["order_id", "paid_amount", "fee", "method", "paid_ts"]
                if c in payments.columns]
    pay = payments[pay_cols]
    merged = enriched.merge(pay, on="order_id", how="left", suffixes=("", "_pay"))
    out = pd.DataFrame(index=merged.index)
    out["order_id"] = _col(merged, "order_id")
    out["customer_id"] = _col(merged, "customer_id")
    out["amount_inr"] = _num(merged, "amount_inr")
    out["paid_amount"] = _num(merged, "paid_amount")
    out["fee"] = _num(merged, "fee")
    order_ts = pd.to_datetime(_col(merged, "order_ts"), errors="coerce")
    paid_ts = pd.to_datetime(_col(merged, "paid_ts"), errors="coerce")
    out["payment_gap_sec"] = (paid_ts - order_ts).dt.total_seconds()
    out["status"] = _col(merged, "status", "placed")
    out["method"] = _col(merged, "method", "unpaid").fillna("unpaid")
    out["country"] = _col(merged, "country", "UNKNOWN")
    out["order_ts"] = order_ts
    return out.reset_index(drop=True)


def _aggregate_daily(inputs: List[pd.DataFrame], ctx: dict) -> pd.DataFrame:
    """Roll the fact table up to (country, segment) grain for the reporting layer."""
    src = inputs[0].copy()
    if len(src) == 0:
        return pd.DataFrame(columns=S.DAILY_METRICS.names)
    src["country"] = _col(src, "country", "UNKNOWN").fillna("UNKNOWN")
    seg = ctx.get("segment_lookup")
    if seg is not None and "customer_id" in src.columns:
        src["segment_mix"] = src["customer_id"].map(seg).fillna("UNKNOWN")
    else:
        src["segment_mix"] = "UNKNOWN"
    src["amount_inr"] = _num(src, "amount_inr")
    src["is_paid"] = (~_num(src, "paid_amount").isna()).astype("float64")

    grp = src.groupby(["country", "segment_mix"], dropna=False)
    agg = grp.agg(
        revenue=("amount_inr", "sum"),
        orders_count=("order_id", "count"),
        avg_order_value=("amount_inr", "mean"),
        paid_ratio=("is_paid", "mean"),
    ).reset_index()
    agg["agg_id"] = np.arange(len(agg), dtype="int64")
    agg["window_ts"] = ctx["window_end"] - pd.Timedelta(minutes=1)
    return agg[S.DAILY_METRICS.names].reset_index(drop=True)


def _load_warehouse(inputs: List[pd.DataFrame], ctx: dict) -> pd.DataFrame:
    """Final publish step: rename to the warehouse contract and stamp load status."""
    src = inputs[0]
    out = pd.DataFrame(index=src.index)
    out["fact_id"] = np.arange(len(src), dtype="int64")
    out["revenue"] = _num(src, "revenue")
    out["orders_count"] = _num(src, "orders_count")
    out["avg_order_value"] = _num(src, "avg_order_value")
    out["country"] = _col(src, "country", "UNKNOWN")
    # The publish step flags rows it could not fully resolve.
    out["load_status"] = np.where(out["revenue"].isna(), "partial", "ok")
    out["window_ts"] = pd.to_datetime(_col(src, "window_ts"), errors="coerce")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# DAG assembly
# --------------------------------------------------------------------------- #

def _source(_inputs, _ctx):  # replaced at runtime by the batch generator
    raise RuntimeError("source nodes are populated by the generator")


def build_pipeline() -> "Pipeline":
    nodes = [
        Node("extract_customers", S.CUSTOMERS, [], _source,
             is_source=True, base_runtime=3.0, base_mem=140.0),
        Node("extract_orders", S.ORDERS, [], _source,
             is_source=True, base_runtime=5.0, base_mem=200.0),
        Node("extract_payments", S.PAYMENTS, [], _source,
             is_source=True, base_runtime=4.0, base_mem=170.0),
        Node("clean_customers", S.CUSTOMERS_CLEAN, ["extract_customers"], _clean_customers,
             base_runtime=6.0, base_mem=210.0),
        Node("clean_orders", S.ORDERS_CLEAN, ["extract_orders"], _clean_orders,
             base_runtime=7.0, base_mem=260.0),
        Node("join_orders_customers", S.ORDERS_ENRICHED,
             ["clean_orders", "clean_customers"], _join_orders_customers,
             is_join=True, base_runtime=11.0, base_mem=420.0),
        Node("join_payments", S.ORDERS_PAID,
             ["join_orders_customers", "extract_payments"], _join_payments,
             is_join=True, base_runtime=13.0, base_mem=480.0),
        Node("aggregate_daily", S.DAILY_METRICS, ["join_payments"], _aggregate_daily,
             base_runtime=9.0, base_mem=300.0),
        Node("load_warehouse", S.WAREHOUSE, ["aggregate_daily"], _load_warehouse,
             base_runtime=6.0, base_mem=180.0),
    ]
    return Pipeline(nodes)


@dataclass
class Pipeline:
    nodes: List[Node]

    def __post_init__(self):
        self.index: Dict[str, int] = {n.name: i for i, n in enumerate(self.nodes)}
        self._validate_topological_order()

    def _validate_topological_order(self) -> None:
        seen: set = set()
        for n in self.nodes:
            for p in n.parents:
                if p not in seen:
                    raise ValueError(f"{n.name} consumes {p} before it is produced")
            seen.add(n.name)

    @property
    def names(self) -> List[str]:
        return [n.name for n in self.nodes]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def adjacency(self) -> np.ndarray:
        """Directed adjacency ``A[i, j] = 1`` when node *i* feeds node *j*."""
        a = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        for n in self.nodes:
            j = self.index[n.name]
            for p in n.parents:
                a[self.index[p], j] = 1.0
        return a

    def descendants(self, name: str) -> List[str]:
        """All nodes reachable downstream of ``name`` (its blast radius)."""
        out, frontier = set(), [name]
        while frontier:
            cur = frontier.pop()
            for n in self.nodes:
                if cur in n.parents and n.name not in out:
                    out.add(n.name)
                    frontier.append(n.name)
        return sorted(out)
