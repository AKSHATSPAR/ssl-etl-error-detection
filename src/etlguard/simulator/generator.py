"""Simulation engine: run the pipeline over a timeline and record what happened.

The output of :func:`simulate` is the actual dataset the models train on:

    X          float32 [T, N, F]  data-quality profile of every node at every run
    y_anom     uint8   [T, N]     node is affected (origin *or* downstream of one)
    y_root     uint8   [T, N]     node is the origin of the incident
    y_type     int16   [T]        fault-type id for the batch (0 = healthy)

Labels are recorded but never shown to the self-supervised model; they exist to
evaluate it, and to train the supervised baselines it is compared against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import schema as S
from .dag import Pipeline, build_pipeline
from .faults import (FAULT_LABELS, REGISTRY, FaultEvent, apply_fault,
                     applicable_faults)
from .profiler import FEATURE_NAMES, N_FEATURES, RunMeta, profile_table


@dataclass
class SimConfig:
    """Everything that controls one synthetic pipeline history."""

    n_batches: int = 8000                 # hourly runs (~11 months)
    start: str = "2024-01-01 00:00:00"
    freq_minutes: int = 60
    seed: int = 7

    # Volume model
    base_orders: int = 900
    customer_pool: int = 20000
    payment_coverage: float = 0.86
    growth_rate: float = 0.25             # end-of-timeline multiplier increment
    hour_amplitude: float = 0.45
    weekend_factor: float = 0.72
    volume_noise: float = 0.09

    # Benign distractors (labelled NORMAL on purpose)
    campaigns_per_1000: float = 1.75      # temporary volume surges (rate, not count)
    campaign_gain: float = 1.7
    campaign_len: Tuple[int, int] = (12, 60)
    drift_strength: float = 0.015         # random-walk speed of the country mix

    # Fault schedule
    fault_rate_test: float = 0.14         # fraction of batches carrying an incident
    fault_rate_train: float = 0.03        # historical contamination (SSL must tolerate it)
    duration_lambda: float = 0.45         # geometric decay of incident length
    p_concurrent: float = 0.06            # chance a second, independent incident overlaps
    severity_a: float = 2.0               # Beta(a, b) severity prior
    severity_b: float = 2.2

    # Chronological split
    train_frac: float = 0.60
    val_frac: float = 0.15


# --------------------------------------------------------------------------- #
# Source-data generation
# --------------------------------------------------------------------------- #

class SourceFactory:
    """Generates the three raw source tables for one batch.

    A persistent customer pool gives foreign keys something real to point at, so
    the healthy orphan rate is ~0 and a broken key upstream produces a genuine
    referential-integrity signal downstream.
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        n = cfg.customer_pool
        self.cust_ids = np.arange(100_000, 100_000 + n, dtype=np.int64)
        self.cust_country = rng.choice(len(S.COUNTRIES), size=n,
                                       p=self._country_prior())
        self.cust_segment = rng.choice(len(S.SEGMENTS), size=n, p=[0.7, 0.22, 0.08])
        self.cust_ltv = np.abs(rng.lognormal(10.6, 0.55, size=n))
        self.cust_age = np.clip(rng.normal(34, 11, size=n), 18, 92)
        self.cust_signup = (pd.Timestamp(cfg.start)
                            - pd.to_timedelta(rng.integers(1, 1500, size=n), unit="D"))
        # Country mix drifts slowly over the timeline: a benign distractor.
        self.country_w = np.array(self._country_prior(), dtype=np.float64)

    @staticmethod
    def _country_prior() -> List[float]:
        w = np.array([0.34, 0.22, 0.11, 0.09, 0.07, 0.07, 0.05, 0.05])
        return list(w / w.sum())

    def step_drift(self) -> None:
        """One step of the country-mix random walk (labelled normal)."""
        noise = self.rng.normal(0, self.cfg.drift_strength, size=self.country_w.size)
        w = np.clip(self.country_w + noise, 0.01, None)
        self.country_w = w / w.sum()

    def batch(self, ws: pd.Timestamp, we: pd.Timestamp, n_orders: int,
              rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
        n_cust = int(max(200, 0.4 * n_orders + 250))

        # ---- customers snapshot (a slice of the pool, re-weighted by drift) ----
        p = self.country_w[self.cust_country]
        p = p / p.sum()
        idx = rng.choice(len(self.cust_ids), size=n_cust, replace=False, p=p)
        customers = pd.DataFrame({
            "customer_id": self.cust_ids[idx],
            "lifetime_value": self.cust_ltv[idx] * rng.normal(1.0, 0.02, n_cust),
            "age": self.cust_age[idx].astype("float64"),
            "country": [S.COUNTRIES[i] for i in self.cust_country[idx]],
            "segment": [S.SEGMENTS[i] for i in self.cust_segment[idx]],
            "signup_ts": self.cust_signup[idx],
            "email": [f"user{i}@example.com" for i in self.cust_ids[idx]],
        })
        # A small, stable trickle of missing emails/ages is normal, not a fault.
        customers.loc[rng.random(n_cust) < 0.01, "email"] = None
        customers.loc[rng.random(n_cust) < 0.004, "age"] = np.nan

        # ---- orders (FKs drawn from the snapshot, so joins resolve) ----
        buyer_pos = rng.integers(0, n_cust, size=n_orders)
        buyer_ids = customers["customer_id"].to_numpy()[buyer_pos]
        seg = customers["segment"].to_numpy()[buyer_pos]
        # Enterprise customers place larger orders: a real cross-column dependency.
        seg_mult = np.where(seg == "enterprise", 3.4, np.where(seg == "smb", 1.6, 1.0))
        amount = np.abs(rng.lognormal(7.2, 0.75, n_orders)) * seg_mult
        order_ids = rng.integers(10**9, 2 * 10**9, size=n_orders, dtype=np.int64)
        offsets = rng.random(n_orders) * (we - ws).total_seconds()
        orders = pd.DataFrame({
            "order_id": order_ids,
            "customer_id": buyer_ids,
            "amount": amount,
            "quantity": np.clip(rng.poisson(2.6, n_orders), 1, None).astype("float64"),
            "status": rng.choice(S.ORDER_STATUS, n_orders, p=[.42, .27, .22, .06, .03]),
            "channel": rng.choice(S.CHANNELS, n_orders, p=[.38, .27, .28, .07]),
            "currency": rng.choice(S.CURRENCIES, n_orders, p=[.72, .17, .06, .05]),
            "order_ts": ws + pd.to_timedelta(offsets, unit="s"),
        })

        # ---- payments (subset of this batch's orders) ----
        n_pay = int(n_orders * self.cfg.payment_coverage)
        pay_pos = rng.choice(n_orders, size=n_pay, replace=False)
        gap = np.abs(rng.lognormal(5.2, 1.0, n_pay))
        # A batch may only contain payments that actually settled inside its
        # window; without this clamp a healthy run reports ~10% "future" events
        # and the freshness features lose their meaning.
        order_offsets = offsets[pay_pos]
        window_sec = (we - ws).total_seconds()
        gap = np.minimum(gap, np.maximum(window_sec - order_offsets, 0.0))
        payments = pd.DataFrame({
            "payment_id": rng.integers(3 * 10**9, 4 * 10**9, size=n_pay, dtype=np.int64),
            "order_id": order_ids[pay_pos],
            "paid_amount": amount[pay_pos] * rng.normal(1.0, 0.005, n_pay),
            "fee": amount[pay_pos] * rng.uniform(0.008, 0.022, n_pay),
            "method": rng.choice(S.PAY_METHODS, n_pay, p=[.34, .31, .14, .13, .08]),
            "paid_ts": orders["order_ts"].to_numpy()[pay_pos]
                       + pd.to_timedelta(gap, unit="s"),
        })
        return {"extract_customers": customers,
                "extract_orders": orders,
                "extract_payments": payments}


# --------------------------------------------------------------------------- #
# Volume + operational cost models
# --------------------------------------------------------------------------- #

def _volume_curve(cfg: SimConfig, ts: pd.DatetimeIndex,
                  rng: np.random.Generator) -> np.ndarray:
    """Seasonal, trending, campaign-perturbed order volume, all of it *normal*."""
    t = np.arange(len(ts))
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()

    seasonal = 1.0 + cfg.hour_amplitude * np.sin(2 * np.pi * (hour - 9) / 24.0)
    weekly = np.where(dow >= 5, cfg.weekend_factor, 1.0)
    trend = 1.0 + cfg.growth_rate * (t / max(len(ts) - 1, 1))

    campaign = np.ones(len(ts))
    n_campaigns = max(1, int(round(cfg.campaigns_per_1000 * len(ts) / 1000.0)))
    for _ in range(n_campaigns):
        start = int(rng.integers(0, max(len(ts) - 1, 1)))
        length = int(rng.integers(cfg.campaign_len[0], cfg.campaign_len[1] + 1))
        end = min(len(ts), start + length)
        ramp = np.linspace(0, np.pi, end - start)
        campaign[start:end] *= 1.0 + (cfg.campaign_gain - 1.0) * np.sin(ramp)

    noise = rng.lognormal(0, cfg.volume_noise, len(ts))
    vol = cfg.base_orders * seasonal * weekly * trend * campaign * noise
    return np.clip(vol, 50, None)


def _sample_meta(node, rows_in: int, rows_out: int, contention: float,
                 rng: np.random.Generator) -> RunMeta:
    """Operational telemetry as a noisy function of the work actually done."""
    scale = max(rows_out, rows_in, 1) / 1000.0
    runtime = node.base_runtime * (scale ** 0.85) * contention * rng.lognormal(0, 0.12)
    mem = node.base_mem * (scale ** 0.70) * rng.lognormal(0, 0.08)
    return RunMeta(
        rows_in=rows_in,
        runtime_sec=float(runtime),
        cpu_util=float(np.clip(0.34 + 0.22 * np.tanh(scale) * contention
                               + rng.normal(0, 0.05), 0.02, 1.0)),
        mem_mb=float(mem),
        retry_count=int(rng.poisson(0.02)),
        queue_delay_sec=float(rng.exponential(4.0) * contention),
    )


# --------------------------------------------------------------------------- #
# Fault scheduling
# --------------------------------------------------------------------------- #

def schedule_faults(cfg: SimConfig, pipe: Pipeline, n_train: int,
                    rng: np.random.Generator) -> List[FaultEvent]:
    """Lay incidents over the timeline, denser after the training period.

    Training history is only lightly contaminated (as real history is), which is
    what lets a self-supervised model learn "normal" without labels, while still
    forcing it to be robust to the few bad batches it does see.
    """
    events: List[FaultEvent] = []
    occupied: Dict[str, set] = {n.name: set() for n in pipe.nodes}
    busy_batches: set = set()

    def target_for(split_start: int, split_end: int, rate: float) -> None:
        span = split_end - split_start
        budget = int(span * rate)
        placed, guard = 0, 0
        while placed < budget and guard < budget * 40 + 500:
            guard += 1
            node = pipe.nodes[int(rng.integers(0, pipe.n_nodes))]
            choices = applicable_faults(node.flags)
            ftype = str(rng.choice(choices))
            duration = 1 + int(rng.geometric(cfg.duration_lambda)) - 1
            duration = int(np.clip(duration, 1, 24))
            start = int(rng.integers(split_start, max(split_start + 1, split_end - duration)))
            span_b = set(range(start, start + duration))
            if span_b & occupied[node.name]:
                continue
            # Keep incidents mostly non-overlapping so the root cause is unambiguous.
            if (span_b & busy_batches) and rng.random() > cfg.p_concurrent:
                continue
            sev = float(rng.beta(cfg.severity_a, cfg.severity_b))
            events.append(FaultEvent(ftype, node.name, start, duration, sev))
            occupied[node.name].update(span_b)
            busy_batches.update(span_b)
            placed += duration

    target_for(0, n_train, cfg.fault_rate_train)
    target_for(n_train, cfg.n_batches, cfg.fault_rate_test)
    events.sort(key=lambda e: (e.start_batch, e.node))
    return events


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def _run_pipeline(pipe, sources: Dict[str, pd.DataFrame], ctx: dict,
                  fault_by_node: Dict[str, FaultEvent], contention: float,
                  seed: int, t: int) -> np.ndarray:
    """Execute every task once and return the ``[N, F]`` profile matrix.

    Each node draws its telemetry noise and its fault randomness from its *own*
    seeded generator, keyed on ``(seed, batch, node)``.  That isolation is what
    makes the counterfactual exact: re-running with ``fault_by_node={}`` produces
    a bit-identical result everywhere the fault did not reach, so any difference
    that remains is caused by the fault and nothing else.
    """
    out = np.zeros((pipe.n_nodes, N_FEATURES), dtype=np.float32)
    outputs: Dict[str, pd.DataFrame] = {}
    producer_of = {n.spec.name: n.name for n in pipe.nodes}

    for i, node in enumerate(pipe.nodes):
        crashed = False
        if node.is_source:
            df = sources[node.name]
            rows_in = len(df)
        else:
            parents = [outputs[p] for p in node.parents]
            rows_in = sum(len(p) for p in parents)
            try:
                df = node.execute(parents, ctx)
            except Exception:
                # A task that cannot run at all still produces telemetry: an
                # empty output plus retries. That is a legitimate ETL failure
                # mode, not a simulator bug, so we record it rather than die.
                df = pd.DataFrame(columns=node.spec.names)
                crashed = True

        meta_rng = np.random.default_rng([seed, t, 100 + i])
        meta = _sample_meta(node, rows_in, len(df), contention, meta_rng)
        if crashed:
            meta.retry_count += 3
            meta.schema_change_flag = 1.0

        if node.name in fault_by_node:
            fault_rng = np.random.default_rng([seed, t, 200 + i])
            df, meta = apply_fault(df, node.spec, meta, fault_by_node[node.name],
                                   fault_rng)

        fk_ref = None
        fks = node.spec.by_role(S.Role.FK)
        if fks and fks[0].references:
            ref_table, ref_col = fks[0].references.split(".")
            src_node = producer_of.get(ref_table)
            if src_node in outputs and ref_col in outputs[src_node].columns:
                fk_ref = outputs[src_node][ref_col].to_numpy()
            elif src_node in sources and ref_col in sources[src_node].columns:
                fk_ref = sources[src_node][ref_col].to_numpy()

        out[i] = profile_table(df, node.spec, meta,
                               ctx["window_start"], ctx["window_end"], fk_ref)
        outputs[node.name] = df

    ctx["_last_outputs"] = outputs
    return out


def simulate(cfg: SimConfig, progress: bool = True,
             dump_examples: int = 0) -> Dict[str, object]:
    rng = np.random.default_rng(cfg.seed)
    pipe = build_pipeline()
    N, F, T = pipe.n_nodes, N_FEATURES, cfg.n_batches

    ts = pd.date_range(cfg.start, periods=T, freq=f"{cfg.freq_minutes}min")
    volume = _volume_curve(cfg, ts, rng)
    n_train = int(T * cfg.train_frac)
    n_val = int(T * cfg.val_frac)

    events = schedule_faults(cfg, pipe, n_train, rng)
    by_batch: Dict[int, List[FaultEvent]] = {}
    for e in events:
        for b in range(e.start_batch, e.start_batch + e.duration):
            by_batch.setdefault(b, []).append(e)

    X = np.zeros((T, N, F), dtype=np.float32)
    X_cf = np.zeros((T, N, F), dtype=np.float32)   # counterfactual: same batch, no fault
    y_topo = np.zeros((T, N), dtype=np.uint8)      # topological over-approximation
    y_root = np.zeros((T, N), dtype=np.uint8)
    y_type = np.zeros(T, dtype=np.int16)
    severity = np.zeros(T, dtype=np.float32)

    factory = SourceFactory(cfg, rng)
    segment_lookup = pd.Series(
        [S.SEGMENTS[i] for i in factory.cust_segment], index=factory.cust_ids)

    examples: Dict[str, pd.DataFrame] = {}
    iterator = range(T)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="simulating", unit="batch")

    for t in iterator:
        ws, we = ts[t], ts[t] + pd.Timedelta(minutes=cfg.freq_minutes)
        factory.step_drift()

        contention_rng = np.random.default_rng([cfg.seed, t, 1])
        contention = float(1.0 + 0.22 * np.sin(2 * np.pi * ts[t].hour / 24.0 + 1.1)
                           + contention_rng.normal(0, 0.04))

        src_rng = np.random.default_rng([cfg.seed, t, 0])
        sources = factory.batch(ws, we, int(volume[t]), src_rng)

        active = by_batch.get(t, [])
        by_node: Dict[str, FaultEvent] = {e.node: e for e in active}
        if active:
            worst = max(active, key=lambda e: e.severity)
            y_type[t] = FAULT_LABELS.index(worst.fault_type)
            severity[t] = worst.severity
        for e in active:
            y_root[t, pipe.index[e.node]] = 1
            y_topo[t, pipe.index[e.node]] = 1
            if REGISTRY[e.fault_type].channel != "operational":
                for d in pipe.descendants(e.node):
                    y_topo[t, pipe.index[d]] = 1

        ctx = {"window_start": ws, "window_end": we, "segment_lookup": segment_lookup}
        X[t] = _run_pipeline(pipe, sources, ctx, by_node, contention, cfg.seed, t)
        outputs = ctx.pop("_last_outputs")

        if active:
            # Same batch, same seeds, no fault: the honest baseline for "what
            # this run should have looked like".
            ctx_cf = {"window_start": ws, "window_end": we,
                      "segment_lookup": segment_lookup}
            X_cf[t] = _run_pipeline(pipe, sources, ctx_cf, {}, contention, cfg.seed, t)
            ctx_cf.pop("_last_outputs", None)
        else:
            X_cf[t] = X[t]

        if dump_examples and len(examples) < dump_examples * 2:
            tag = "faulty" if active else "healthy"
            if sum(1 for k in examples if k.startswith(tag)) < dump_examples:
                examples[f"{tag}_t{t}"] = outputs["join_payments"].head(200).copy()

    split = np.zeros(T, dtype=np.uint8)
    split[n_train:n_train + n_val] = 1
    split[n_train + n_val:] = 2

    return {
        "X": X,
        "X_cf": X_cf,
        "y_topo": y_topo,
        "y_root": y_root,
        "y_type": y_type,
        "severity": severity,
        "split": split,
        "timestamps": ts.astype("int64").to_numpy() // 10**9,
        "adjacency": pipe.adjacency(),
        "node_names": pipe.names,
        "feature_names": FEATURE_NAMES,
        "fault_labels": FAULT_LABELS,
        "events": [asdict(e) for e in events],
        "config": asdict(cfg),
        "examples": examples,
    }
