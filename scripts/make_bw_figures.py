"""Regenerate the key figures in the deck's palette: blue, black and white.

    PYTHONPATH=src python scripts/make_bw_figures.py

These are redrawn rather than recoloured. Difference is carried by value,
hatching and line style first, with blue reserved for the one thing each figure
is about, so the figures still read correctly if anyone prints them in
greyscale. The blue was picked to sit at 29/255 in luminance, dark enough to
contrast on white and light enough to stay distinct from black when printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etlguard.data.dataset import ETLData

OUT = Path("artifacts/slides/bw")
BLUE = "#1A5FA8"
BLUE_SOFT = "#7FA8D0"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "font.family": "serif",
    "axes.grid": True, "grid.alpha": 0.25, "grid.color": "0.7",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "0.2", "text.color": "0.1",
    "axes.labelcolor": "0.1", "xtick.color": "0.3", "ytick.color": "0.3",
})


def fig_propagation(d: ETLData) -> None:
    """Blast radius, as a value map rather than a colour map."""
    roots = np.where(d.y_root.sum(axis=1) > 0)[0]
    roots = roots[(roots > 100) & (roots < len(d.X) - 100)]
    early = [t for t in roots if int(np.argmax(d.y_root[t])) <= 4]
    t0 = int(max(early or list(roots), key=lambda t: d.y_anom[t].sum()))
    node = int(np.argmax(d.y_root[t0]))
    lo, hi = t0 - 20, t0 + 20
    dev = (np.abs(d.X[lo:hi] - d.X_cf[lo:hi]) / d.scaler.scale).max(axis=2).T

    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    im = ax.imshow(dev, aspect="auto", cmap="Blues", origin="lower",
                   extent=[lo, hi, -0.5, d.n_nodes - 0.5], vmin=0, vmax=10)
    ax.set_yticks(range(d.n_nodes))
    ax.set_yticklabels([n.replace("_", " ") for n in d.node_names], fontsize=8)
    ax.axvline(t0, color=BLUE, ls="--", lw=1.6)
    ax.set_xlabel("pipeline run")
    ax.set_title(f"One fault at '{d.node_names[node].replace('_',' ')}', "
                 f"{int(d.y_anom[t0].sum())} of {d.n_nodes} tasks move", fontsize=10)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("deviation from the same run without the fault", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "bw_propagation.png"); plt.close(fig)


def fig_methods() -> None:
    """Method comparison. Ours is solid black, the rest are hatched outlines."""
    det = [("LOF", .973), ("Dense AE", .956), ("Ours", .953), ("LSTM AE", .951),
           ("PCA", .942), ("Rolling z", .502), ("Isolation F", .466)]
    loc = [("LSTM AE", .818), ("PCA", .801), ("Ours", .797), ("LOF", .773),
           ("Dense AE", .749), ("Rolling z", .570), ("Isolation F", .340)]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    for ax, data, title in zip(axes, [det, loc],
                               ["Detection (AUPRC)", "Localization (Hit@1)"]):
        data = sorted(data, key=lambda r: r[1])
        names = [n for n, _ in data]; vals = [v for _, v in data]
        for i, (n, v) in enumerate(zip(names, vals)):
            mine = n == "Ours"
            ax.barh(i, v, color=BLUE if mine else "white",
                    edgecolor=BLUE if mine else "black", linewidth=1.1,
                    hatch="" if mine else "///")
            ax.text(v + .015, i, f"{v:.3f}", va="center", fontsize=8,
                    color=BLUE if mine else "0.2",
                    fontweight="bold" if mine else "normal")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8.5,
                           fontweight=["normal", "bold"][0])
        ax.set_xlim(0, 1.08); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "bw_methods.png"); plt.close(fig)


def fig_scoring() -> None:
    """The main finding: four scoring rules, same model."""
    rows = [("Rank by raw error", .588, False),
            ("Compare to own history", .773, False),
            ("Surprise given inputs", .773, False),
            ("Both", .797, True)]
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    for i, (n, v, hl) in enumerate(rows):
        ax.barh(i, v, color=BLUE if hl else "0.80",
                edgecolor=BLUE if hl else "0.45", linewidth=1.1)
        ax.text(v + .012, i, f"{v:.3f}", va="center", fontsize=9,
                color=BLUE if hl else "0.25",
                fontweight="bold" if hl else "normal")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _, _ in rows], fontsize=9.5)
    ax.set_xlim(0, .92); ax.set_xlabel("Hit@1")
    ax.set_title("Same trained model, only the scoring rule changes", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "bw_scoring.png"); plt.close(fig)


def fig_attention(d: ETLData) -> None:
    import json
    f = Path("artifacts/results/attribution_study.json")
    if not f.exists():
        return
    roll = np.array(json.load(f.open())["attention_rollout_mean"])
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(roll, cmap="Blues")
    ax.set_xticks(range(d.n_nodes))
    ax.set_xticklabels([n.replace("_", " ") for n in d.node_names],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(d.n_nodes))
    ax.set_yticklabels([n.replace("_", " ") for n in d.node_names], fontsize=7)
    ax.set_title("It found the pipeline on its own\n"
                 "(dark cells are the real connections)", fontsize=10)
    ax.grid(False); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(OUT / "bw_attention.png"); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = ETLData.load("data/etl_v1")
    fig_propagation(d); fig_methods(); fig_scoring(); fig_attention(d)
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}  {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
