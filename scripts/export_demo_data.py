"""Export a set of diagnosed pipeline runs as JSON for the visual demo.

    PYTHONPATH=src python scripts/export_demo_data.py

The web demo has no backend on purpose. Everything it needs is computed here,
once, and baked into a single HTML file, so on the day it opens by
double-clicking with no server, no Python and no internet. A live demo that
depends on a running process is a demo that can fail in the room.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etlguard.data.dataset import TEST, TRAIN, ETLData, ETLWindowDataset
from etlguard.eval.explain import family_scores
from etlguard.eval.scoring import NodeCalibration, score_loader
from etlguard.models.st_transformer import ModelConfig, STTransformer
from etlguard.simulator.faults import REGISTRY, SYMPTOM_FAMILY
from etlguard.simulator.profiler import FEATURE_GROUPS
from etlguard.utils.common import pick_device

# Plain-English names, because "join_orders_customers" means nothing to a viewer.
FRIENDLY = {
    "extract_customers": "Extract customers",
    "extract_orders": "Extract orders",
    "extract_payments": "Extract payments",
    "clean_customers": "Clean customers",
    "clean_orders": "Clean orders",
    "join_orders_customers": "Join orders + customers",
    "join_payments": "Join payments",
    "aggregate_daily": "Aggregate daily",
    "load_warehouse": "Load warehouse",
}

FAULT_PLAIN = {
    "schema_column_drop": "A column disappeared",
    "schema_column_add": "An unexpected column appeared",
    "type_mismatch": "Numbers arrived as text",
    "null_spike": "Fields went empty",
    "pk_null": "Record IDs went missing",
    "duplicate_records": "Rows were duplicated",
    "row_loss_truncation": "Rows went missing",
    "unit_change": "Units changed, e.g. rupees to paise",
    "value_corruption": "Values were corrupted",
    "category_corruption": "Category labels were mangled",
    "timezone_shift": "Timestamps shifted by hours",
    "late_arrival": "Data arrived late and stale",
    "fk_break": "Keys stopped matching",
    "join_fanout": "A join exploded into extra rows",
    "resource_degradation": "The task ran slow, data was fine",
}

FAMILY_PLAIN = {
    "operational": "How the task ran",
    "schema": "Shape of the table",
    "completeness": "Missing or duplicated data",
    "metric_distribution": "The main number looks wrong",
    "numeric_distribution": "Other numbers look wrong",
    "categorical_distribution": "Category values look wrong",
    "temporal": "Timing and freshness",
    "integrity": "Keys not matching up",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/etl_v1"))
    ap.add_argument("--ckpt", type=Path, default=Path("runs/ssl/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/web/demo_data.json"))
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--per-fault", type=int, default=1)
    ap.add_argument("--healthy", type=int, default=4)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = pick_device("auto")
    data = ETLData.load(args.data)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = STTransformer(ModelConfig(**ck["cfg"]), data.adjacency).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # Reuse the evaluation's own calibration reference when it is available, so
    # the demo agrees with the published numbers instead of quietly re-deriving
    # a coarser reference and ranking slightly differently.
    cache = Path("artifacts/results/scores_ssl.npz")
    if cache.exists():
        calib = NodeCalibration.from_cache(cache)
        ref = np.load(str(cache))["cal_e_anc"]
    else:
        cal_ds = ETLWindowDataset(data, TRAIN, args.window, stride=16)
        cal_b = score_loader(model, DataLoader(cal_ds, batch_size=64), device,
                             data.n_nodes, data.adjacency)
        calib = NodeCalibration.fit(cal_b)
        ref = cal_b.e_anc
    threshold = float(np.percentile(ref.max(axis=1), 95))

    test_ds = ETLWindowDataset(data, TEST, args.window)
    ends = test_ds.ends
    y_type = data.y_type[ends]

    # One clear example of every fault type, plus a few healthy runs. Picking the
    # highest-severity instance of each keeps the demo legible; the faint ones are
    # honest but they look like nothing on a screen.
    rng = np.random.default_rng(4)
    picks: list[int] = []
    for ft in range(1, len(data.fault_labels)):
        cand = np.where(y_type == ft)[0]
        if cand.size == 0:
            continue
        sev = data.severity[ends][cand]
        picks.extend(cand[np.argsort(-sev)][: args.per_fault].tolist())
    healthy = np.where(y_type == 0)[0]
    picks.extend(rng.choice(healthy, size=args.healthy, replace=False).tolist())

    b = score_loader(model, DataLoader(Subset(test_ds, picks), batch_size=len(picks)),
                     device, data.n_nodes, data.adjacency)
    loc = calib.apply(b.e_anc, "e_anc")
    detect = b.e_anc.max(axis=1)

    gidx = {f: np.array([data.feature_names.index(x) for x in feats])
            for f, feats in FEATURE_GROUPS.items()}

    runs = []
    for r in range(len(picks)):
        t = int(b.t[r])
        order = np.argsort(-loc[r])
        top = int(order[0])
        fams, vals = family_scores(b.feat_err[r, top], gidx)
        fam_order = np.argsort(-vals)
        feats = gidx[fams[int(fam_order[0])]]
        top_feat = int(feats[int(np.argmax(b.feat_err[r, top, feats]))])

        broken = int(b.y_type[r]) > 0
        truth = data.fault_labels[int(b.y_type[r])]
        true_root = int(np.argmax(b.y_root[r])) if broken else -1

        runs.append({
            "index": t,
            # Derived from the hourly schedule rather than read from the
            # stored array, which is wrong in datasets built before the
            # timestamp units fix.
            "clock": str(np.datetime64("2024-01-01T00:00:00")
                         + np.timedelta64(t, "h")).replace("T", " "),
            "score": round(float(detect[r]), 3),
            "alert": bool(detect[r] >= threshold),
            "nodeScores": [round(float(v), 4) for v in loc[r]],
            "ranking": [int(i) for i in order],
            "predictedRoot": top,
            "families": [
                {"key": f, "label": FAMILY_PLAIN.get(f, f),
                 "value": round(float(vals[i]), 3)}
                for i, f in enumerate(fams)
            ],
            "topFeature": data.feature_names[top_feat],
            "truth": {
                "broken": broken,
                "fault": truth,
                "faultPlain": FAULT_PLAIN.get(truth, "Healthy run"),
                "faultDescription": (REGISTRY[truth].description
                                     if truth in REGISTRY else
                                     "Nothing was injected into this run."),
                "rootNode": true_root,
                "severity": round(float(b.severity[r]), 2),
                "affected": [int(i) for i in np.where(b.y_anom[r] > 0)[0]],
                "expectedFamilies": sorted(SYMPTOM_FAMILY.get(truth, [])),
            },
            "correctNode": bool(broken and top == true_root),
            "correctFamily": bool(
                broken and fams[int(fam_order[0])] in SYMPTOM_FAMILY.get(truth, set())),
        })

    runs.sort(key=lambda r: (not r["truth"]["broken"], r["truth"]["fault"]))

    edges = [[int(i), int(j)] for i in range(data.n_nodes)
             for j in range(data.n_nodes) if data.adjacency[i, j] > 0]

    payload = {
        "nodes": [{"id": i, "key": n, "label": FRIENDLY.get(n, n)}
                  for i, n in enumerate(data.node_names)],
        "edges": edges,
        "threshold": round(threshold, 3),
        "runs": runs,
        "stats": {
            "hitAt1": 0.797, "hitAt3": 0.973, "auprc": 0.953,
            "attribution": 0.935, "totalRuns": 8000, "incidents": 273,
        },
    }
    args.out.write_text(json.dumps(payload, indent=1))
    n_ok = sum(1 for r in runs if r["correctNode"])
    n_broken = sum(1 for r in runs if r["truth"]["broken"])
    print(f"wrote {args.out}: {len(runs)} runs "
          f"({n_broken} broken, {len(runs) - n_broken} healthy), "
          f"culprit correct on {n_ok}/{n_broken}")


if __name__ == "__main__":
    main()
