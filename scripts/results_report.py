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


SEED_TAGS = ("ssl", "seed13", "seed29")


def load_all(root: Path) -> dict:
    merged = {}
    for f in sorted(root.glob("results_*.json")):
        merged.update(json.loads(f.read_text()))
    if not merged:
        raise SystemExit(f"no results_*.json found in {root}")
    return merged


def split_runs(res: dict):
    """Separate the three kinds of run.

    They all land in results_*.json, but they answer different questions and
    must not share a table. An ablation listed next to LOF reads as a competing
    method, which it is not: it is the same method with a component removed.
    """
    ablations = {k: v for k, v in res.items() if k.startswith("abl_")}
    seeds = {k: v for k, v in res.items() if k in SEED_TAGS}
    main = {k: v for k, v in res.items()
            if k not in ablations and k not in seeds or k == "ssl"}
    return main, seeds, ablations


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
    main, seeds, ablations = split_runs(res)
    df = main_table(main)
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

    if seeds:
        sh = np.array([seeds[k]["localization"]["hit@1"] for k in seeds])
        sa = np.array([seeds[k]["detection"]["auprc"] for k in seeds])
        sm = np.array([seeds[k]["localization"]["mrr"] for k in seeds])
        print(f"\nSeed repeats ({len(seeds)} seeds), the noise floor for the "
              f"ablations below")
        print(f"  AUPRC {sa.mean():.3f} +/- {sa.std():.3f}   "
              f"Hit@1 {sh.mean():.3f} +/- {sh.std():.3f}   "
              f"MRR {sm.mean():.3f} +/- {sm.std():.3f}")

    if ablations:
        ref, sd = (sh.mean(), sh.std()) if seeds else (np.nan, np.nan)
        rows = []
        for k, v in sorted(ablations.items()):
            d = v["localization"]["hit@1"] - ref
            rows.append({
                "ablation": k.replace("abl_", ""),
                "AUPRC": v["detection"]["auprc"],
                "Hit@1": v["localization"]["hit@1"],
                "MRR": v["localization"]["mrr"],
                "delta Hit@1": d,
                "verdict": "within seed noise" if abs(d) <= 2 * sd
                           else "OUTSIDE 2 sd",
            })
        adf = pd.DataFrame(rows).sort_values("delta Hit@1", ascending=False)
        print("\nAblations (identical 40-epoch budget)\n")
        print(adf.round(3).to_string(index=False))
        adf.round(4).to_csv(args.out / "ablation_table.csv", index=False)
        (args.out / "ablation_table.md").write_text(
            adf.round(3).to_markdown(index=False))

    ssl = main.get("ssl")
    if ssl:
        print(f"\ndetection pooling rule chosen on validation: "
              f"{ssl.get('detection_rule')}")
        # The headline finding: two independent routes to cross-task
        # comparability, worth the same, and only slightly more together.
        variants = [
            ("deviation magnitude, raw", "localization_residual_only_raw"),
            ("deviation magnitude, calibrated", "localization_residual_only_calibrated"),
            ("ancestor-conditioned, raw", "localization_ancestor_raw"),
            ("ancestor-conditioned, calibrated", "localization"),
        ]
        print("\nLocalization scoring rules, same trained model\n")
        print(f"  {'variant':<38}{'Hit@1':>7}  {'95% CI':<18}{'MRR':>7}")
        for label, key in variants:
            v = ssl.get(key)
            if not v:
                continue
            ci = v.get("hit@1_ci95")
            ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
            print(f"  {label:<38}{v['hit@1']:>7.3f}  {ci_s:<18}{v['mrr']:>7.3f}")
        att = ssl.get("attribution", {})
        print(f"\nattribution accuracy: {att.get('attribution_accuracy', 0):.3f} "
              f"over {att.get('n_scored', 0)} incidents")
    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
