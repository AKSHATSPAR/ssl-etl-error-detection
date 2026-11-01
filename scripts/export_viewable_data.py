"""Write spreadsheet-friendly samples of the dataset.

    PYTHONPATH=src python scripts/export_viewable_data.py

The dataset ships as a compressed numpy array because that is what the model
reads, but nobody can open that in a viva. These are small CSVs of the same
thing, openable in Numbers or Excel: the raw rows the pipeline moves, and the
38-number profile the model actually sees.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etlguard.data.dataset import ETLData

OUT = Path("artifacts/data_samples")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = ETLData.load("data/etl_v1")

    # 1. What the pipeline actually moves: ordinary business rows.
    raw = pd.read_parquet("data/etl_v1/examples/healthy_t0.parquet").head(40)
    raw.to_csv(OUT / "1_raw_rows_the_pipeline_moves.csv", index=False)

    # 2. What the model sees: one row per task per hour, 38 columns.
    rows = []
    for t in range(0, 40):
        for i, name in enumerate(d.node_names):
            rows.append({"run": t, "task": name,
                         **{f: float(v) for f, v in zip(d.feature_names, d.X[t, i])}})
    pd.DataFrame(rows).round(4).to_csv(
        OUT / "2_what_the_model_sees_38_features.csv", index=False)

    # 3. The answer key, one row per run.
    lab = pd.DataFrame({
        "run": np.arange(len(d.y_type)),
        "is_broken": (d.y_type > 0).astype(int),
        "fault_type": [d.fault_labels[i] for i in d.y_type],
        "task_that_caused_it": [d.node_names[int(np.argmax(r))] if r.sum() else ""
                                for r in d.y_root],
        "tasks_affected": d.y_anom.sum(axis=1),
        "severity": d.severity.round(2),
        "split": ["train", "validation", "test"][0] if False else
                 [["train", "validation", "test"][s] for s in d.split],
    })
    lab.to_csv(OUT / "3_the_answer_key.csv", index=False)
    lab[lab.is_broken == 1].to_csv(OUT / "4_just_the_broken_runs.csv", index=False)

    for f in sorted(OUT.glob("*.csv")):
        n = sum(1 for _ in f.open()) - 1
        print(f"  {f.name:<42} {n:>6} rows   {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
