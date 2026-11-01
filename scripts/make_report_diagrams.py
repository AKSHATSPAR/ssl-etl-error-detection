"""Draw the structural diagrams the VIT report template asks for.

    PYTHONPATH=src python scripts/make_report_diagrams.py

System architecture, data flow, use case, class and sequence diagrams. Drawn
rather than screenshotted so they stay legible in print, in one visual language:
plain boxes, a single accent, no gradients or shadows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Ellipse

OUT = Path("artifacts/report_diagrams")
BLUE = "#1A5FA8"
INK = "#16202B"
GREY = "#6B7A8A"
PALE = "#EDF2F7"

plt.rcParams.update({"figure.dpi": 200, "font.family": "serif", "font.size": 9})


def box(ax, x, y, w, h, label, fill="white", edge=INK, fc=INK, fs=9, bold=False,
        rounded=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={rounded}",
                                linewidth=1.1, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            color=fc, fontweight="bold" if bold else "normal", linespacing=1.4)


def arrow(ax, p1, p2, label=None, style="-|>", dashed=False, off=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                                 linewidth=1.0, color=GREY,
                                 linestyle="--" if dashed else "-",
                                 shrinkA=1, shrinkB=1))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + off, label, ha="center", va="bottom", fontsize=7.4,
                color=GREY, backgroundcolor="white")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 10); ax.set_ylim(0, h / w * 10)
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------- #

def system_architecture():
    fig, ax = canvas(9, 5.2)
    H = 5.2 / 9 * 10
    layers = [
        ("Simulation Layer",
         "Pipeline DAG  |  Fault injector  |  Profiler", PALE),
        ("Data Layer",
         "Windowing  |  Robust normalisation  |  Chronological split", "white"),
        ("Model Layer",
         "Masking  |  Spatio-temporal Transformer  |  Reconstruction head", PALE),
        ("Scoring Layer",
         "History-only score  |  Ancestor-conditioned score  |  Calibration", "white"),
        ("Presentation Layer",
         "Metrics and figures  |  Terminal demo  |  Web demo", PALE),
    ]
    y = H - 1.15
    for i, (title, sub, fill) in enumerate(layers):
        box(ax, 0.6, y, 8.8, 0.82, "", fill=fill)
        ax.text(0.95, y + 0.53, title, fontsize=10, fontweight="bold", color=BLUE,
                va="center")
        ax.text(0.95, y + 0.26, sub, fontsize=8.2, color=INK, va="center")
        if i < len(layers) - 1:
            arrow(ax, (5.0, y), (5.0, y - 0.33))
        y -= 1.15
    ax.text(5.0, H - 0.35, "Figure 4.1  System architecture", ha="center",
            fontsize=9, color=GREY)
    fig.tight_layout(); fig.savefig(OUT / "fig_4_1_architecture.png",
                                    bbox_inches="tight"); plt.close(fig)


def data_flow():
    fig, ax = canvas(9, 4.6)
    H = 4.6 / 9 * 10
    box(ax, 0.3, H - 2.2, 1.5, 0.9, "Source\nsystems", fill=PALE)
    for i, (x, lbl) in enumerate([(2.3, "1.0\nSimulate\npipeline"),
                                  (4.2, "2.0\nProfile\nruns"),
                                  (6.1, "3.0\nTrain\nmodel"),
                                  (8.0, "4.0\nScore and\nexplain")]):
        box(ax, x, H - 2.3, 1.55, 1.1, lbl, fill="white", fs=8.2)
        if i:
            arrow(ax, (x - 0.35, H - 1.75), (x, H - 1.75))
    arrow(ax, (1.8, H - 1.75), (2.3, H - 1.75))
    for x, lbl in [(2.3, "D1  Raw rows"), (4.2, "D2  Profiles"),
                   (6.1, "D3  Model"), (8.0, "D4  Results")]:
        box(ax, x, H - 3.5, 1.55, 0.5, lbl, fill=PALE, fs=7.6)
        arrow(ax, (x + 0.78, H - 1.2 - 0.05), (x + 0.78, H - 3.0), dashed=True)
    box(ax, 8.0, H - 4.5, 1.55, 0.6, "Engineer", fill=PALE, fs=8.2)
    arrow(ax, (8.78, H - 3.5), (8.78, H - 3.9))
    ax.text(5.0, H - 0.3, "Figure 4.2  Level-1 data flow diagram", ha="center",
            fontsize=9, color=GREY)
    fig.tight_layout(); fig.savefig(OUT / "fig_4_2_dfd.png",
                                    bbox_inches="tight"); plt.close(fig)


def use_case():
    fig, ax = canvas(8.6, 5.0)
    H = 5.0 / 8.6 * 10
    ax.add_patch(FancyBboxPatch((2.6, 0.5), 5.2, H - 1.4,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                linewidth=1.1, edgecolor=INK, facecolor="white"))
    ax.text(5.2, H - 1.05, "ETL Monitoring System", ha="center", fontsize=9.5,
            fontweight="bold", color=BLUE)
    cases = ["Generate benchmark", "Train model", "Detect broken run",
             "Localize root cause", "Explain symptom", "View diagnosis"]
    for i, c in enumerate(cases):
        cy = H - 1.6 - i * 0.62
        ax.add_patch(Ellipse((5.2, cy), 4.2, 0.46, linewidth=1.0,
                             edgecolor=INK, facecolor=PALE))
        ax.text(5.2, cy, c, ha="center", va="center", fontsize=8.3, color=INK)
    for ay, name, idx in [(H - 2.0, "Data\nEngineer", [2, 3, 4, 5]),
                          (H - 3.6, "Researcher", [0, 1])]:
        ax.add_patch(Circle((1.3, ay + 0.42), 0.17, fill=False, lw=1.1, color=INK))
        ax.plot([1.3, 1.3], [ay + 0.25, ay - 0.1], color=INK, lw=1.1)
        ax.plot([1.1, 1.5], [ay + 0.15, ay + 0.15], color=INK, lw=1.1)
        ax.plot([1.3, 1.12], [ay - 0.1, ay - 0.38], color=INK, lw=1.1)
        ax.plot([1.3, 1.48], [ay - 0.1, ay - 0.38], color=INK, lw=1.1)
        ax.text(1.3, ay - 0.62, name, ha="center", fontsize=8, color=INK)
        for i in idx:
            arrow(ax, (1.65, ay), (3.15, H - 1.6 - i * 0.62))
    ax.text(5.0, H - 0.3, "Figure 4.3  Use case diagram", ha="center",
            fontsize=9, color=GREY)
    fig.tight_layout(); fig.savefig(OUT / "fig_4_3_usecase.png",
                                    bbox_inches="tight"); plt.close(fig)


def class_diagram():
    fig, ax = canvas(9.4, 5.4)
    H = 5.4 / 9.4 * 10
    classes = [
        ("Pipeline", ["nodes: List[Node]", "adjacency()", "descendants()"], 0.3, H - 2.0),
        ("FaultEvent", ["fault_type: str", "node: str", "severity: float"], 3.5, H - 2.0),
        ("Profiler", ["FEATURE_NAMES", "profile_table()"], 6.7, H - 2.0),
        ("ETLData", ["X, y_root, y_anom", "scaler", "load()"], 0.3, H - 4.1),
        ("STTransformer", ["encode()", "forward()", "embed()"], 3.5, H - 4.1),
        ("ScoreBundle", ["e_hist, e_anc", "detection_score()"], 6.7, H - 4.1),
    ]
    for name, attrs, x, y in classes:
        h = 0.42 + 0.26 * len(attrs)
        box(ax, x, y, 2.7, h, "", fill="white")
        ax.text(x + 1.35, y + h - 0.21, name, ha="center", fontsize=9,
                fontweight="bold", color=BLUE)
        ax.plot([x, x + 2.7], [y + h - 0.42, y + h - 0.42], color=INK, lw=0.9)
        for j, a in enumerate(attrs):
            ax.text(x + 0.14, y + h - 0.62 - j * 0.26, a, fontsize=7.5, color=INK)
    arrow(ax, (3.0, H - 1.55), (3.5, H - 1.55), label="injects", off=0.05)
    arrow(ax, (6.2, H - 1.55), (6.7, H - 1.55), label="measures", off=0.05)
    arrow(ax, (8.05, H - 2.0), (8.05, H - 3.4), label="produces", off=0.05)
    arrow(ax, (1.65, H - 2.0), (1.65, H - 3.4), label="builds", off=0.05)
    arrow(ax, (3.0, H - 3.7), (3.5, H - 3.7), label="feeds", off=0.05)
    arrow(ax, (6.2, H - 3.7), (6.7, H - 3.7), label="scored by", off=0.05)
    ax.text(5.0, H - 0.3, "Figure 4.4  Class diagram", ha="center",
            fontsize=9, color=GREY)
    fig.tight_layout(); fig.savefig(OUT / "fig_4_4_class.png",
                                    bbox_inches="tight"); plt.close(fig)


def sequence():
    fig, ax = canvas(9.4, 5.2)
    H = 5.2 / 9.4 * 10
    actors = ["Engineer", "Scorer", "Model", "Calibration", "Explainer"]
    xs = [0.9, 3.0, 5.1, 7.0, 8.9]
    for x, a in zip(xs, actors):
        box(ax, x - 0.75, H - 1.1, 1.5, 0.42, a, fill=PALE, fs=8)
        ax.plot([x, x], [H - 1.15, 0.5], color=GREY, lw=0.8, ls=(0, (3, 3)))
    steps = [
        (0, 1, "submit a pipeline run"),
        (1, 2, "hide final run, ask for prediction"),
        (2, 1, "predicted profile"),
        (1, 2, "hide task + descendants, per task"),
        (2, 1, "per-task surprise"),
        (1, 3, "convert to percentile of own history"),
        (3, 1, "calibrated scores"),
        (1, 4, "per-feature error at top task"),
        (4, 1, "symptom family"),
        (1, 0, "verdict, task, symptom"),
    ]
    y = H - 1.45
    for a, b, lbl in steps:
        back = b < a
        arrow(ax, (xs[a], y), (xs[b], y), dashed=back)
        mx = (xs[a] + xs[b]) / 2
        ax.text(mx, y + 0.07, lbl, ha="center", fontsize=7.2, color=INK,
                backgroundcolor="white")
        y -= 0.37
    ax.text(5.0, H - 0.3, "Figure 4.5  Sequence diagram for one diagnosis",
            ha="center", fontsize=9, color=GREY)
    fig.tight_layout(); fig.savefig(OUT / "fig_4_5_sequence.png",
                                    bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    system_architecture(); data_flow(); use_case(); class_diagram(); sequence()
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name:<32} {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
