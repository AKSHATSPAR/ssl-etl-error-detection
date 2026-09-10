"""Compare three attribution methods on ground truth.

    PYTHONPATH=src python scripts/attribution_study.py --ckpt runs/ssl/best.pt

Attribution in this literature is nearly always shown as a picture and almost
never checked, because there is usually nothing to check it against. Here the
simulator knows which symptom family each fault type is mechanistically expected
to move, so each method gets an accuracy rather than a plot.

Scored at the *true* origin rather than the predicted one, so a localization
mistake earlier in the pipeline does not get charged to the explainer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etlguard.data.dataset import TEST, ETLData, ETLWindowDataset
from etlguard.eval.explain import (attention_rollout, integrated_gradients,
                                   score_attribution)
from etlguard.eval.scoring import descendant_mask
from etlguard.models.st_transformer import ModelConfig, STTransformer
from etlguard.simulator.faults import SYMPTOM_FAMILY
from etlguard.simulator.profiler import FEATURE_GROUPS
from etlguard.utils.common import pick_device, save_json

plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/etl_v1"))
    ap.add_argument("--ckpt", type=Path, default=Path("runs/ssl/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/results"))
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = pick_device("auto")
    data = ETLData.load(args.data)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = STTransformer(ModelConfig(**ck["cfg"]), data.adjacency).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    ds = ETLWindowDataset(data, TEST, args.window)
    ends = ds.ends
    inc = np.where(data.y_type[ends] > 0)[0]
    print(f"{len(inc)} incident windows in the test split")

    desc = torch.from_numpy(descendant_mask(data.adjacency)).to(device)
    gidx = {f: np.array([data.feature_names.index(x) for x in feats])
            for f, feats in FEATURE_GROUPS.items()}
    expected = {i: SYMPTOM_FAMILY[lab] for i, lab in enumerate(data.fault_labels)
                if lab in SYMPTOM_FAMILY}

    recon_attr, ig_attr, roll_attr = [], [], []
    y_root, y_type = [], []

    for start in range(0, len(inc), args.batch):
        sel = inc[start:start + args.batch]
        batch = [ds[int(i)] for i in sel]
        x = torch.stack([b["x"] for b in batch]).to(device)
        cal = torch.stack([b["cal"] for b in batch]).to(device)
        yr = np.stack([b["y_root"].numpy() for b in batch])
        root = torch.tensor(yr.argmax(axis=1), device=device)

        # Same masking the localizer uses, so the explanation explains the score
        # that was actually produced rather than some other quantity.
        mask = torch.zeros_like(x)
        mask[:, -1] = desc[root].unsqueeze(-1).expand(-1, -1, x.shape[-1])

        with torch.no_grad():
            pred = model(x, mask, cal)
            recon_attr.append(((pred[:, -1] - x[:, -1]) ** 2).cpu().numpy())

        ig = integrated_gradients(model, x, mask, cal, root, steps=args.ig_steps)
        # Sum over the window: the explanation is per (node, feature).
        ig_attr.append(ig.abs().sum(dim=1).detach().cpu().numpy())

        roll = attention_rollout(model, x, mask, cal)
        if roll is not None:
            roll_attr.append(roll)

        y_root.append(yr)
        y_type.append(np.stack([b["y_type"].numpy() for b in batch]))
        if start % (args.batch * 5) == 0:
            print(f"  {start + len(sel)}/{len(inc)}", flush=True)

    recon = np.concatenate(recon_attr)
    ig = np.concatenate(ig_attr)
    y_root = np.concatenate(y_root)
    y_type = np.concatenate(y_type)

    results = {
        "reconstruction_error": score_attribution(recon, y_root, y_type, gidx, expected),
        "integrated_gradients": score_attribution(ig, y_root, y_type, gidx, expected),
    }

    print("\nAttribution accuracy (top symptom family matches the expected one)\n")
    for name, r in results.items():
        print(f"  {name:<24} {r['accuracy']:.3f}   over {r['n_scored']} incidents")

    # Per fault type, so the failures are visible rather than averaged away.
    rows = []
    for ft in sorted(set(int(t) for t in y_type if t > 0)):
        row = {"fault": data.fault_labels[ft],
               "n": int((y_type == ft).sum())}
        for name, r in results.items():
            row[name] = r["per_type"].get(ft, float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("reconstruction_error")
    print("\nPer fault type\n")
    print(df.round(2).to_string(index=False))
    df.round(3).to_csv(args.out / "attribution_by_fault.csv", index=False)

    if roll_attr:
        roll = np.concatenate(roll_attr)
        mean_roll = roll.mean(axis=0)
        results["attention_rollout_mean"] = mean_roll.tolist()

        # Quantify the "it looks like the DAG" impression. Off-diagonal mass is
        # split between true pipeline edges and everything else, and compared
        # against what uniform attention would put there. Nothing constrains the
        # model to this: the DAG bias is a learned scalar per relation, not a mask.
        n = data.n_nodes
        adj = data.adjacency
        edge = ((adj + adj.T) > 0)
        off = ~np.eye(n, dtype=bool)
        on_edge = mean_roll[edge & off].sum()
        total_off = mean_roll[off].sum()
        share = float(on_edge / max(total_off, 1e-9))
        chance = float(edge[off].sum() / off.sum())
        results["attention_edge_alignment"] = {
            "share_of_off_diagonal_mass_on_true_edges": round(share, 4),
            "share_expected_by_chance": round(chance, 4),
            "lift": round(share / max(chance, 1e-9), 2),
        }
        print(f"\nAttention on true DAG edges: {100 * share:.1f}% of off-diagonal "
              f"mass, against {100 * chance:.1f}% expected by chance "
              f"({share / max(chance, 1e-9):.1f}x)")
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        im = ax.imshow(roll.mean(axis=0), cmap="viridis")
        ax.set_xticks(range(data.n_nodes))
        ax.set_xticklabels(data.node_names, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(data.n_nodes))
        ax.set_yticklabels(data.node_names, fontsize=6.5)
        ax.set_title("Mean structural attention during incidents\n"
                     "(row attends to column)", fontsize=8)
        ax.grid(False)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(args.out / "fig7_attention.png")
        plt.close(fig)

    # Side-by-side accuracy per fault type.
    d = df.set_index("fault")[["reconstruction_error", "integrated_gradients"]]
    d = d.sort_values("reconstruction_error")
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(d) + 1.6))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["reconstruction_error"], height=0.4,
            label="reconstruction error", color="#2b6cb0")
    ax.barh(y + 0.2, d["integrated_gradients"], height=0.4,
            label="integrated gradients", color="#dd6b20")
    ax.set_yticks(y)
    ax.set_yticklabels(d.index, fontsize=7)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("attribution accuracy")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Does the explanation name the right symptom family?")
    fig.tight_layout()
    fig.savefig(args.out / "fig8_attribution.png")
    plt.close(fig)

    save_json(args.out / "attribution_study.json", results)
    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
