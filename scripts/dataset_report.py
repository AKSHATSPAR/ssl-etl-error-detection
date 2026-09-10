"""Produce the descriptive figures and tables for the generated dataset.

Run after `build_dataset`:

    PYTHONPATH=src python scripts/dataset_report.py --data data/etl_v1
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etlguard.data.dataset import ETLData
from etlguard.simulator.profiler import FEATURE_GROUPS

plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def fig_volume_with_incidents(d: ETLData, out: Path, span=(4800, 5400)) -> None:
    """Order volume over time, with incident windows shaded.

    The point of this figure: benign campaigns and seasonality move the row count
    as much as several fault types do, so a threshold on volume cannot work.
    """
    lo, hi = span
    i = d.node_names.index("extract_orders")
    k = d.feature_names.index("log_rows_out")
    t = np.arange(lo, hi)
    rows = np.expm1(d.X[lo:hi, i, k])

    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(t, rows, lw=0.8, color="#2b6cb0", label="rows out of extract_orders")
    faulty = t[d.y_anom[lo:hi].sum(axis=1) > 0]
    for b in faulty:
        ax.axvspan(b - 0.5, b + 0.5, color="#e53e3e", alpha=0.18, lw=0)
    ax.set_xlabel("pipeline run (hourly)")
    ax.set_ylabel("rows")
    ax.set_title("Healthy volume is seasonal and campaign-driven; red spans are labelled incidents")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig1_volume_incidents.png")
    plt.close(fig)


def fig_propagation(d: ETLData, out: Path) -> None:
    """A node x time heat-map around one incident, showing the blast radius."""
    roots = np.where(d.y_root.sum(axis=1) > 0)[0]
    roots = roots[(roots > 100) & (roots < len(d.X) - 100)]
    # Show the incident with the widest measured reach that originates early in
    # the DAG -- the case where localization is hardest and most worth seeing.
    early = [t for t in roots if int(np.argmax(d.y_root[t])) <= 4]
    pool = early or list(roots)
    best_t = int(max(pool, key=lambda t: d.y_anom[t].sum()))
    best_node = int(np.argmax(d.y_root[best_t]))

    lo, hi = best_t - 24, best_t + 24
    # Deviation from the counterfactual, not from the global median: healthy
    # seasonality swings the robust z by several units on its own, which would
    # bury the blast radius in noise. Against the counterfactual, a healthy run
    # is exactly zero by construction.
    dev = (np.abs(d.X[lo:hi] - d.X_cf[lo:hi]) / d.scaler.scale).max(axis=2).T

    fig, ax = plt.subplots(figsize=(9, 3.0))
    im = ax.imshow(dev, aspect="auto", cmap="magma", origin="lower",
                   extent=[lo, hi, -0.5, d.n_nodes - 0.5], vmin=0, vmax=10)
    ax.set_yticks(range(d.n_nodes))
    ax.set_yticklabels(d.node_names, fontsize=7)
    ax.axvline(best_t, color="#63b3ed", ls="--", lw=1.0)
    ax.set_xlabel("pipeline run")
    reached = int(d.y_anom[best_t].sum())
    ax.set_title(f"Blast radius: incident originates at '{d.node_names[best_node]}' "
                 f"(dashed line); {reached} of {d.n_nodes} nodes measurably move")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="max deviation from counterfactual (robust units)")
    fig.tight_layout()
    fig.savefig(out / "fig2_propagation.png")
    plt.close(fig)


def fig_fault_mix(d: ETLData, out: Path) -> None:
    counts = np.bincount(d.y_type, minlength=len(d.fault_labels))[1:]
    labels = d.fault_labels[1:]
    order = np.argsort(counts)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.barh([labels[i] for i in order], counts[order], color="#4a5568")
    ax.set_xlabel("batches affected")
    ax.set_title("Incident mix")
    fig.tight_layout()
    fig.savefig(out / "fig3_fault_mix.png")
    plt.close(fig)


def fig_feature_response(d: ETLData, out: Path) -> None:
    """Mean |robust z| per feature family, healthy runs vs each fault type."""
    fam_idx = {fam: [d.feature_names.index(f) for f in feats]
               for fam, feats in FEATURE_GROUPS.items()}
    rows = []
    for c, label in enumerate(d.fault_labels):
        m = d.y_type == c
        if m.sum() < 5:
            continue
        # Look only at nodes actually implicated, otherwise healthy nodes dilute it.
        sel = d.y_anom[m].astype(bool)
        z = np.abs(d.Xz[m])
        row = {"fault": label, "n": int(m.sum())}
        for fam, idx in fam_idx.items():
            vals = z[:, :, idx].max(axis=2)
            row[fam] = float(vals[sel].mean()) if sel.any() else float(vals.mean())
        rows.append(row)

    df = pd.DataFrame(rows).set_index("fault")
    n_col = df.pop("n")
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(df) + 1.8))
    im = ax.imshow(df.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels([f"{i} (n={n})" for i, n in zip(df.index, n_col)], fontsize=7)
    ax.set_title("Which symptom family each fault type actually moves")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="mean max |robust z| on implicated nodes")
    fig.tight_layout()
    fig.savefig(out / "fig4_fault_signature.png")
    plt.close(fig)
    df.round(2).to_csv(out / "fault_signature.csv")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/etl_v1"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/dataset_report"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = ETLData.load(args.data)
    print(f"loaded X={d.X.shape}  nodes={d.n_nodes}  features={d.n_features}")

    fig_volume_with_incidents(d, args.out)
    fig_propagation(d, args.out)
    fig_fault_mix(d, args.out)
    sig = fig_feature_response(d, args.out)

    print("\nFault signature (mean max |robust z| per symptom family):")
    pd.set_option("display.width", 200)
    print(sig.round(2).to_string())

    # A useful sanity number: how far downstream does the average incident reach?
    reach = d.y_anom.sum(axis=1)[d.y_root.sum(axis=1) > 0]
    print(f"\nmean nodes affected per incident : {reach.mean():.2f} / {d.n_nodes}")
    print(f"incidents where >1 node is affected: "
          f"{100 * (reach > 1).mean():.1f}%")
    print(f"\nfigures written to {args.out}/")


if __name__ == "__main__":
    main()
