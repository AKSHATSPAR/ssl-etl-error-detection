"""Sanity-check the simulator: does each fault type actually move the features?

For every registered fault we run the *same* batch twice — once healthy, once
with the fault injected at a given node — using an identical RNG seed, so the
only difference between the two runs is the fault itself.  We then report which
profile features moved the most, at the injected node and downstream.

This is the simulator's unit test and, equally, the evidence that the labels are
learnable: a fault that shifts nothing is a mislabelled batch, not a hard one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etlguard.simulator.dag import build_pipeline
from etlguard.simulator.faults import REGISTRY, FaultEvent, applicable_faults
from etlguard.simulator.generator import SimConfig, simulate
from etlguard.data.dataset import RobustScaler
from etlguard.simulator.profiler import FEATURE_NAMES


def run_pair(fault_type: str, node: str, severity: float, seed: int, n_batches: int = 64):
    """Simulate the same short timeline with and without one injected fault."""
    cfg = SimConfig(n_batches=n_batches, seed=seed, base_orders=600,
                    fault_rate_train=0.0, fault_rate_test=0.0,
                    train_frac=1.0, val_frac=0.0)

    import etlguard.simulator.generator as gen
    real_scheduler = gen.schedule_faults

    clean = simulate(cfg, progress=False)

    target = n_batches - 2
    gen.schedule_faults = lambda *a, **k: [FaultEvent(fault_type, node, target, 1, severity)]
    try:
        dirty = simulate(cfg, progress=False)
    finally:
        gen.schedule_faults = real_scheduler

    return clean, dirty, target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()

    pipe = build_pipeline()
    node_by_name = {n.name: n for n in pipe.nodes}
    rows = []

    # Inject each fault at the first node where it is legal, plus a mid-DAG node
    # so we can watch propagation.
    for fault in sorted(REGISTRY):
        candidates = [n.name for n in pipe.nodes if fault in applicable_faults(n.flags)]
        node = "clean_orders" if "clean_orders" in candidates else candidates[0]
        clean, dirty, t = run_pair(fault, node, args.severity, args.seed)

        i = pipe.index[node]
        # Express the change in the same robust units the model will see, so a
        # feature that never moves when healthy is not divided by ~0.
        scaler = RobustScaler.fit(clean["X"], clip=np.inf)
        z = np.abs(dirty["X"][t] - clean["X"][t]) / scaler.scale

        top = np.argsort(-z[i])[: args.top_k]
        n_downstream = int((z.max(axis=1) > 3.0).sum()) - 1
        expected = len(pipe.descendants(node))

        rows.append({
            "fault": fault,
            "node": node,
            "max_z_at_node": round(float(z[i].max()), 1),
            "downstream_moved": f"{max(n_downstream, 0)}/{expected}",
            "top_features": ", ".join(
                f"{FEATURE_NAMES[j]}({z[i, j]:.0f})" for j in top if z[i, j] > 1.0),
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 90)
    print("\nFault -> feature-response matrix "
          f"(severity={args.severity}, z = |delta| / healthy std)\n")
    print(df.to_string(index=False))

    weak = df[df["max_z_at_node"] < 3.0]
    print(f"\n{len(df) - len(weak)}/{len(df)} fault types produce a >3-sigma "
          f"response at the injected node.")
    if len(weak):
        print("Weak responders (worth inspecting):",
              ", ".join(weak["fault"].tolist()))


if __name__ == "__main__":
    main()
