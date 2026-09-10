"""Table and column specifications for the simulated data-warehouse pipeline.

The simulator produces *row-level* data, not pre-baked feature vectors.  Faults
are injected into the rows and every downstream statistic (null rates, duplicate
rates, distribution shifts) is *derived* from those rows by the profiler.  That
is what makes fault propagation across the DAG an emergent property of the
simulation rather than something we hand-code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Role(str, Enum):
    """Semantic role of a column.

    The profiler aggregates statistics *by role*, which is what lets every node
    in the DAG emit a feature vector of the same fixed width even though the
    underlying tables have different columns.
    """

    PK = "pk"                 # primary key (uniqueness + non-null expected)
    FK = "fk"                 # foreign key (referential integrity expected)
    METRIC = "metric"         # the table's primary numeric measure
    NUMERIC = "numeric"       # secondary numeric column
    CATEGORICAL = "categorical"
    TIMESTAMP = "timestamp"
    TEXT = "text"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    role: Role
    dtype: str
    # Distributional priors used by the healthy generator.
    categories: Optional[List[str]] = None
    low: Optional[float] = None
    high: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    nullable_rate: float = 0.0          # baseline (healthy) null rate
    references: Optional[str] = None    # "table.column" for FKs


@dataclass(frozen=True)
class TableSpec:
    """A table contract.

    Every table is guaranteed to carry a PK, a METRIC column, at least one
    CATEGORICAL column and at least one TIMESTAMP column.  That guarantee is what
    keeps the profiler's output width constant across the whole DAG.
    """

    name: str
    columns: List[ColumnSpec]

    def by_role(self, role: Role) -> List[ColumnSpec]:
        return [c for c in self.columns if c.role is role]

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.columns]

    def get(self, name: str) -> ColumnSpec:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"{self.name} has no column {name!r}")

    @property
    def pk(self) -> ColumnSpec:
        pks = self.by_role(Role.PK)
        if not pks:
            raise ValueError(f"table {self.name} has no primary key")
        return pks[0]

    @property
    def metric(self) -> ColumnSpec:
        ms = self.by_role(Role.METRIC)
        if not ms:
            raise ValueError(f"table {self.name} has no metric column")
        return ms[0]


COUNTRIES = ["IN", "US", "GB", "DE", "AE", "SG", "AU", "CA"]
SEGMENTS = ["retail", "smb", "enterprise"]
ORDER_STATUS = ["placed", "shipped", "delivered", "returned", "cancelled"]
PAY_METHODS = ["card", "upi", "netbank", "wallet", "cod"]
CURRENCIES = ["INR", "USD", "GBP", "EUR"]
CHANNELS = ["web", "ios", "android", "partner"]


# --------------------------------------------------------------------------- #
# Source tables
# --------------------------------------------------------------------------- #

CUSTOMERS = TableSpec(
    name="customers",
    columns=[
        ColumnSpec("customer_id", Role.PK, "int64"),
        ColumnSpec("lifetime_value", Role.METRIC, "float64", mean=48000.0, std=22000.0),
        ColumnSpec("age", Role.NUMERIC, "float64", mean=34.0, std=11.0, nullable_rate=0.004),
        ColumnSpec("country", Role.CATEGORICAL, "object", categories=COUNTRIES),
        ColumnSpec("segment", Role.CATEGORICAL, "object", categories=SEGMENTS),
        ColumnSpec("signup_ts", Role.TIMESTAMP, "datetime64[ns]"),
        ColumnSpec("email", Role.TEXT, "object", nullable_rate=0.01),
    ],
)

ORDERS = TableSpec(
    name="orders",
    columns=[
        ColumnSpec("order_id", Role.PK, "int64"),
        ColumnSpec("customer_id", Role.FK, "int64", references="customers.customer_id"),
        ColumnSpec("amount", Role.METRIC, "float64", mean=2400.0, std=1500.0),
        ColumnSpec("quantity", Role.NUMERIC, "float64", mean=2.6, std=1.8),
        ColumnSpec("status", Role.CATEGORICAL, "object", categories=ORDER_STATUS),
        ColumnSpec("channel", Role.CATEGORICAL, "object", categories=CHANNELS),
        ColumnSpec("currency", Role.CATEGORICAL, "object", categories=CURRENCIES),
        ColumnSpec("order_ts", Role.TIMESTAMP, "datetime64[ns]"),
    ],
)

PAYMENTS = TableSpec(
    name="payments",
    columns=[
        ColumnSpec("payment_id", Role.PK, "int64"),
        ColumnSpec("order_id", Role.FK, "int64", references="orders.order_id"),
        ColumnSpec("paid_amount", Role.METRIC, "float64", mean=2400.0, std=1500.0),
        ColumnSpec("fee", Role.NUMERIC, "float64", mean=28.0, std=14.0),
        ColumnSpec("method", Role.CATEGORICAL, "object", categories=PAY_METHODS),
        ColumnSpec("paid_ts", Role.TIMESTAMP, "datetime64[ns]"),
    ],
)


# --------------------------------------------------------------------------- #
# Derived tables
# --------------------------------------------------------------------------- #

CUSTOMERS_CLEAN = TableSpec(
    name="customers_clean",
    columns=[
        ColumnSpec("customer_id", Role.PK, "int64"),
        ColumnSpec("lifetime_value", Role.METRIC, "float64"),
        ColumnSpec("age", Role.NUMERIC, "float64"),
        ColumnSpec("country", Role.CATEGORICAL, "object", categories=COUNTRIES),
        ColumnSpec("segment", Role.CATEGORICAL, "object", categories=SEGMENTS),
        ColumnSpec("signup_ts", Role.TIMESTAMP, "datetime64[ns]"),
    ],
)

ORDERS_CLEAN = TableSpec(
    name="orders_clean",
    columns=[
        ColumnSpec("order_id", Role.PK, "int64"),
        ColumnSpec("customer_id", Role.FK, "int64", references="customers.customer_id"),
        ColumnSpec("amount_inr", Role.METRIC, "float64"),
        ColumnSpec("quantity", Role.NUMERIC, "float64"),
        ColumnSpec("status", Role.CATEGORICAL, "object", categories=ORDER_STATUS),
        ColumnSpec("channel", Role.CATEGORICAL, "object", categories=CHANNELS),
        ColumnSpec("order_ts", Role.TIMESTAMP, "datetime64[ns]"),
    ],
)

ORDERS_ENRICHED = TableSpec(
    name="orders_enriched",
    columns=[
        ColumnSpec("order_id", Role.PK, "int64"),
        ColumnSpec("customer_id", Role.FK, "int64", references="customers.customer_id"),
        ColumnSpec("amount_inr", Role.METRIC, "float64"),
        ColumnSpec("quantity", Role.NUMERIC, "float64"),
        ColumnSpec("lifetime_value", Role.NUMERIC, "float64"),
        ColumnSpec("status", Role.CATEGORICAL, "object", categories=ORDER_STATUS),
        ColumnSpec("channel", Role.CATEGORICAL, "object", categories=CHANNELS),
        ColumnSpec("country", Role.CATEGORICAL, "object", categories=COUNTRIES),
        ColumnSpec("segment", Role.CATEGORICAL, "object", categories=SEGMENTS),
        ColumnSpec("order_ts", Role.TIMESTAMP, "datetime64[ns]"),
    ],
)

ORDERS_PAID = TableSpec(
    name="orders_paid",
    columns=[
        ColumnSpec("order_id", Role.PK, "int64"),
        ColumnSpec("customer_id", Role.FK, "int64", references="customers.customer_id"),
        ColumnSpec("amount_inr", Role.METRIC, "float64"),
        ColumnSpec("paid_amount", Role.NUMERIC, "float64"),
        ColumnSpec("fee", Role.NUMERIC, "float64"),
        ColumnSpec("payment_gap_sec", Role.NUMERIC, "float64"),
        ColumnSpec("status", Role.CATEGORICAL, "object", categories=ORDER_STATUS),
        ColumnSpec("method", Role.CATEGORICAL, "object", categories=PAY_METHODS + ["unpaid"]),
        ColumnSpec("country", Role.CATEGORICAL, "object", categories=COUNTRIES),
        ColumnSpec("order_ts", Role.TIMESTAMP, "datetime64[ns]"),
    ],
)

DAILY_METRICS = TableSpec(
    name="daily_metrics",
    columns=[
        ColumnSpec("agg_id", Role.PK, "int64"),
        ColumnSpec("revenue", Role.METRIC, "float64"),
        ColumnSpec("orders_count", Role.NUMERIC, "float64"),
        ColumnSpec("avg_order_value", Role.NUMERIC, "float64"),
        ColumnSpec("paid_ratio", Role.NUMERIC, "float64"),
        ColumnSpec("country", Role.CATEGORICAL, "object", categories=COUNTRIES),
        ColumnSpec("segment_mix", Role.CATEGORICAL, "object", categories=SEGMENTS),
        ColumnSpec("window_ts", Role.TIMESTAMP, "datetime64[ns]"),
    ],
)

WAREHOUSE = TableSpec(
    name="warehouse_fact",
    columns=[
        ColumnSpec("fact_id", Role.PK, "int64"),
        ColumnSpec("revenue", Role.METRIC, "float64"),
        ColumnSpec("orders_count", Role.NUMERIC, "float64"),
        ColumnSpec("avg_order_value", Role.NUMERIC, "float64"),
        ColumnSpec("country", Role.CATEGORICAL, "object", categories=COUNTRIES),
        ColumnSpec("load_status", Role.CATEGORICAL, "object", categories=["ok", "partial"]),
        ColumnSpec("window_ts", Role.TIMESTAMP, "datetime64[ns]"),
    ],
)


ALL_TABLES: Dict[str, TableSpec] = {
    t.name: t
    for t in [
        CUSTOMERS, ORDERS, PAYMENTS,
        CUSTOMERS_CLEAN, ORDERS_CLEAN, ORDERS_ENRICHED,
        ORDERS_PAID, DAILY_METRICS, WAREHOUSE,
    ]
}
