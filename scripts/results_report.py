"""Turn the evaluation JSON into tables and figures for the report.

    PYTHONPATH=src python scripts/results_report.py

Reads every ``results_*.json`` under the results directory, so ablation runs
tagged separately are picked up automatically and land in the same comparison.
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

plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

PRETTY = {
    "ssl": "SSL transformer (ours)",
    "zscore_ewma": "Rolling z-score",
    "pca_residual": "PCA residual",
    "isolation_forest": "Isolation Forest",
    "lof": "LOF",
    "dense_ae": "Dense autoencoder",
    "lstm_ae": "LSTM autoencoder",
}


def load_all(root: Path) -> dict:
    merged = {}
    for f in sorted(root.glob("results_*.json")):
        merged.update(json.loads(f.read_text()))
    if not merged:
        raise SystemExit(f"no results_*.json found in {root}")
    return merged


def main_table(res: dict) -> pd.DataFrame:
    rows = []
    for name, r in res.items():
        d, l = r["detection"], r["localization"]
        rows.append({
            "method": PRETTY.get(name, name),
            "AUROC": d.get("auroc"),
            "AUPRC": d.get("auprc"),
            "best F1": d.get("best_f1"),
            "F1 @1%": d.get("f1@0.01"),
            "F1 @5%": d.get("f1@0.05"),
            "F1 @15%": d.get("f1@0.15"),
            "Hit@1": l.get("hit@1"),
            "Hit@3": l.get("hit@3"),
            "MRR": l.get("mrr"),
            "delay": r.get("delay", {}).get("mean_detection_delay"),
        })
    return pd.DataFrame(rows).sort_values("AUPRC", ascending=False)


def fig_comparison(df: pd.DataFrame, out: Path) -> None:
    """Detection and localization side by side.

    Two panels rather than one, because they are different questions and a
    method can be good at one and useless at the other. Putting them on a shared
    axis would hide exactly that.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.32 * len(df) + 1.6))
    for ax, col, title in zip(axes, ["AUPRC", "Hit@1"],
                              ["Detection (AUPRC)", "Localization (Hit@1)"]):
        d = df.sort_values(col)
        colors = ["#2b6cb0" if "ours" in m else "#a0aec0" for m in d["method"]]
        ax.barh(d["method"], d[col], color=colors)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        for y, v in enumerate(d[col]):
            if pd.notna(v):
                ax.text(v + 0.015, y, f"{v:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig5_method_comparison.png")
    plt.close(fig)


def fig_localization_breakdown(res: dict, out: Path, tag: str = "ssl") -> None:
    r = res.get(tag)
    if not r:
        return
    by_fault = r.get("localization_by_fault", {})
    by_sev = r.get("localization_by_severity", {})
    if not by_fault:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 0.3 * len(by_fault) + 1.8),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    items = sorted(by_fault.items(), key=lambda kv: kv[1].get("hit@1", 0))
    axes[0].barh([k for k, _ in items], [v.get("hit@1", 0) for _, v in items],
                 color="#2b6cb0")
    axes[0].set_xlim(0, 1)
    axes[0].set_title("Localization Hit@1 by fault type")
    for y, (_, v) in enumerate(items):
        axes[0].text(0.01, y, f"  n={v.get('n_incidents', 0)}", va="center",
                     fontsize=6.5, color="white")

    if by_sev:
        keys = sorted(by_sev)
        axes[1].bar(range(len(keys)), [by_sev[k].get("hit@1", 0) for k in keys],
                    color="#4a5568")
        axes[1].set_xticks(range(len(keys)))
        axes[1].set_xticklabels([k.replace("sev_", "") for k in keys],
                                rotation=30, ha="right", fontsize=7)
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Hit@1 by severity quartile")
    fig.tight_layout()
    fig.savefig(out / "fig6_localization_breakdown.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("artifacts/results"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    res = load_all(args.results)
    df = main_table(res)
    any_r = next(iter(res.values()))["detection"]
    ceilings = {b: any_r.get(f"f1_ceiling@{b}") for b in ("0.01", "0.05", "0.15")}
    print("\nAlert-budget ceilings (best possible F1 at that budget, "
          f"base rate {any_r.get('base_rate', 0):.3f}):")
    print("  " + "   ".join(f"@{b}: {v:.3f}" for b, v in ceilings.items()
                            if v is not None))
    pd.set_option("display.width", 200)
    print("\nMain comparison (test split)\n")
    print(df.round(3).to_string(index=False))
    df.round(4).to_csv(args.out / "main_table.csv", index=False)
    (args.out / "main_table.md").write_text(df.round(3).to_markdown(index=False))

    fig_comparison(df, args.out)
    fig_localization_breakdown(res, args.out)

    ssl = res.get("ssl")
    if ssl:
        print(f"\ndetection pooling rule chosen on validation: "
              f"{ssl.get('detection_rule')}")
        naive = ssl.get("localization_residual_only", {})
        print(f"\nLocalization, ancestor-conditioned vs deviation-magnitude:")
        print(f"  ancestor-conditioned  Hit@1={ssl['localization'].get('hit@1'):.3f}  "
              f"MRR={ssl['localization'].get('mrr'):.3f}")
        print(f"  deviation magnitude   Hit@1={naive.get('hit@1', float('nan')):.3f}  "
              f"MRR={naive.get('mrr', float('nan')):.3f}")
        att = ssl.get("attribution", {})
        print(f"\nattribution accuracy: {att.get('attribution_accuracy', 0):.3f} "
              f"over {att.get('n_scored', 0)} incidents")
    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
