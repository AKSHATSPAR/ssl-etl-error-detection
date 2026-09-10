"""Where localization goes wrong, and what it blames instead.

Aggregate Hit@1 says a fault type is hard. It does not say *why*, and the two
plausible reasons call for different fixes: either the origin is genuinely
invisible, or it is visible but a different task consistently outranks it. This
script separates them by looking at the rank of the true origin and at which
task the model picks when it picks wrong.

    PYTHONPATH=src python scripts/error_analysis.py --ckpt runs/ssl/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etlguard.data.dataset import ETLData, make_loaders
from etlguard.eval.scoring import score_loader
from etlguard.models.st_transformer import ModelConfig, STTransformer
from etlguard.utils.common import pick_device


def node_depths(adjacency: np.ndarray) -> np.ndarray:
    n = adjacency.shape[0]
    depth = np.zeros(n, dtype=int)
    for j in range(n):
        parents = np.where(adjacency[:, j] > 0)[0]
        depth[j] = 0 if parents.size == 0 else 1 + depth[parents].max()
    return depth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/etl_v1"))
    ap.add_argument("--ckpt", type=Path, default=Path("runs/ssl/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/results"))
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--focus", type=str, default="row_loss_truncation,duplicate_records")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = pick_device("auto")
    data = ETLData.load(args.data)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = STTransformer(ModelConfig(**ck["cfg"]), data.adjacency).to(device)
    model.load_state_dict(ck["model"])

    loaders = make_loaders(data, window=args.window, batch_size=64)
    b = score_loader(model, loaders["test"], device, data.n_nodes, data.adjacency)

    depth = node_depths(data.adjacency)
    inc = b.y_type > 0
    root = np.argmax(b.y_root, axis=1)
    order = np.argsort(-b.e_anc, axis=1)
    pred = order[:, 0]
    rank = np.array([int(np.where(order[i] == root[i])[0][0]) + 1
                     for i in range(len(root))])

    # ---- per fault type: is the origin invisible, or just outranked? ---- #
    rows = []
    for ft in np.unique(b.y_type[inc]):
        m = inc & (b.y_type == ft)
        if m.sum() < 3:
            continue
        rows.append({
            "fault": data.fault_labels[int(ft)],
            "n": int(m.sum()),
            "hit@1": float((rank[m] == 1).mean()),
            "mean_rank": float(rank[m].mean()),
            "mean_origin_depth": float(depth[root[m]].mean()),
            "mean_nodes_affected": float(b.y_anom[m].sum(axis=1).mean()),
        })
    df = pd.DataFrame(rows).sort_values("hit@1")
    pd.set_option("display.width", 200)
    print("\nPer fault type\n")
    print(df.round(2).to_string(index=False))
    df.round(3).to_csv(args.out / "error_analysis_by_fault.csv", index=False)

    # ---- does difficulty track how far the fault spreads? ---- #
    c = np.corrcoef(df["mean_nodes_affected"], df["hit@1"])[0, 1]
    print(f"\ncorrelation between blast radius and Hit@1: {c:+.2f}")

    # ---- for the hard faults, what gets blamed instead? ---- #
    for name in args.focus.split(","):
        name = name.strip()
        if name not in data.fault_labels:
            continue
        ft = data.fault_labels.index(name)
        m = inc & (b.y_type == ft)
        if m.sum() == 0:
            continue
        wrong = m & (rank != 1)
        print(f"\n{name}: {int(m.sum())} incidents, "
              f"{int(wrong.sum())} mislocalized")
        print("  origin depth distribution:",
              dict(zip(*np.unique(depth[root[m]], return_counts=True))))
        if wrong.sum():
            tab = pd.crosstab(
                pd.Series([data.node_names[i] for i in root[wrong]], name="true origin"),
                pd.Series([data.node_names[i] for i in pred[wrong]], name="blamed"))
            print(tab.to_string())
            # Is the blamed task downstream of the true one?
            downstream = np.array([
                depth[pred[i]] > depth[root[i]] for i in np.where(wrong)[0]])
            print(f"  blamed a task further downstream: "
                  f"{100 * downstream.mean():.0f}% of errors")

    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
