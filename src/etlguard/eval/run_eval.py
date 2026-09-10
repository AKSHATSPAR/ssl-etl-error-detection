"""Evaluate the pretrained model against every baseline, on identical terms.

    PYTHONPATH=src python -m etlguard.eval.run_eval --ckpt runs/ssl/best.pt

Protocol notes that matter for reading the numbers:

* Everything is fitted on the **training split only**, which is unlabelled and
  ~3% contaminated. No baseline gets a guaranteed-clean training set either.
* The one thing chosen using labels is the detection pooling rule (4 options),
  selected on **validation** AUPRC. It is stated rather than buried, and the
  test split is untouched until the final call.
* Localization is scored only on runs that actually carry an incident.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from ..data.dataset import TEST, TRAIN, VAL, ETLData, ETLWindowDataset, make_loaders
from ..models.baselines import (CLASSICAL_BASELINES, DenseAutoEncoder,
                                LSTMAutoEncoder, window_features)
from ..models.st_transformer import ModelConfig, STTransformer
from ..simulator.faults import SYMPTOM_FAMILY
from ..simulator.profiler import FEATURE_GROUPS
from ..utils.common import fmt, pick_device, save_json, set_seed
from .metrics import (attribution_accuracy, detection_delay, detection_metrics,
                      localization_by_group, localization_metrics)
from .scoring import ScoreBundle, score_loader


def _labels_for(data: ETLData, ends: np.ndarray) -> Dict[str, np.ndarray]:
    return {"y_anom": data.y_anom[ends], "y_root": data.y_root[ends],
            "y_type": data.y_type[ends], "severity": data.severity[ends]}


def evaluate_scores(y_type: np.ndarray, y_root: np.ndarray, run_score: np.ndarray,
                    node_score: np.ndarray, severity: np.ndarray,
                    fault_labels: List[str], budget: float = 0.01) -> Dict:
    y = (y_type > 0).astype(np.int64)
    det = detection_metrics(y, run_score, budget=budget)
    loc = localization_metrics(y_root, node_score)
    delay = detection_delay(y_type, run_score, det["threshold"])

    # Severity quartiles, computed over incidents only.
    sev = severity[y_type > 0]
    out = {"detection": det, "localization": loc, "delay": delay}
    if sev.size > 8:
        edges = np.quantile(sev, [0.25, 0.5, 0.75])
        bucket = np.digitize(severity, edges)
        names = ["sev_Q1_lowest", "sev_Q2", "sev_Q3", "sev_Q4_highest"]
        out["localization_by_severity"] = {
            names[int(b)]: localization_metrics(y_root[(bucket == b) & (y_type > 0)],
                                                node_score[(bucket == b) & (y_type > 0)])
            for b in np.unique(bucket[y_type > 0])
        }
    out["localization_by_fault"] = localization_by_group(
        y_root, node_score, y_type, fault_labels)
    return out


def main(argv=None) -> Dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/etl_v1"))
    ap.add_argument("--ckpt", type=Path, default=Path("runs/ssl/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/results"))
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--budget", type=float, default=0.01)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--tag", type=str, default="ssl")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = pick_device(args.device)
    data = ETLData.load(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict] = {}

    # ------------------------------------------------------------------ #
    # The self-supervised model
    # ------------------------------------------------------------------ #
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["cfg"])
    model = STTransformer(cfg, data.adjacency).to(device)
    model.load_state_dict(ck["model"])

    loaders = make_loaders(data, window=args.window, batch_size=args.batch_size)
    t0 = time.time()
    val_b = score_loader(model, loaders["val"], device, data.n_nodes)
    test_b = score_loader(model, loaders["test"], device, data.n_nodes)
    print(f"scored val+test in {time.time() - t0:.1f}s")

    # Pooling rule chosen on validation, never on test.
    from sklearn.metrics import average_precision_score
    yv = val_b.is_incident
    rule_scores = {r: float(average_precision_score(yv, val_b.detection_score(r)))
                   for r in ScoreBundle.DETECTION_RULES}
    best_rule = max(rule_scores, key=rule_scores.get)
    print(f"detection rule selected on val AUPRC: {best_rule}  "
          f"({ {k: round(v, 3) for k, v in rule_scores.items()} })")

    ssl_res = evaluate_scores(test_b.y_type, test_b.y_root,
                              test_b.detection_score(best_rule),
                              test_b.e_ctx, test_b.severity,
                              data.fault_labels, args.budget)
    ssl_res["detection_rule"] = best_rule
    ssl_res["val_rule_auprc"] = rule_scores

    # The naive localizer: rank nodes by raw deviation, ignoring the DAG. This is
    # the comparison the ancestor-conditioned score has to beat to matter.
    ssl_res["localization_residual_only"] = localization_metrics(
        test_b.y_root, test_b.e_hist)

    gidx = {f: np.array([data.feature_names.index(x) for x in feats])
            for f, feats in FEATURE_GROUPS.items()}
    expected = {i: SYMPTOM_FAMILY[lab] for i, lab in enumerate(data.fault_labels)
                if lab in SYMPTOM_FAMILY}
    ssl_res["attribution"] = attribution_accuracy(
        test_b.feat_err, test_b.y_root, test_b.y_type, gidx, expected)
    results[args.tag] = ssl_res

    # ------------------------------------------------------------------ #
    # Baselines
    # ------------------------------------------------------------------ #
    if not args.skip_baselines:
        tr_ends = ETLWindowDataset(data, TRAIN, args.window).ends
        te_ends = ETLWindowDataset(data, TEST, args.window).ends
        f_tr = window_features(data.Xz, tr_ends, args.window)
        f_te = window_features(data.Xz, te_ends, args.window)
        lab = _labels_for(data, te_ends)

        for cls in CLASSICAL_BASELINES:
            b = cls()
            t1 = time.time()
            b.fit(f_tr)
            run_s, node_s = b.score(f_te)
            results[b.name] = evaluate_scores(
                lab["y_type"], lab["y_root"], run_s, node_s, lab["severity"],
                data.fault_labels, args.budget)
            print(f"  {b.name:<18} fitted+scored in {time.time() - t1:.1f}s")

        ae = DenseAutoEncoder(device=device).fit(f_tr)
        run_s, node_s = ae.score(f_te)
        results[ae.name] = evaluate_scores(
            lab["y_type"], lab["y_root"], run_s, node_s, lab["severity"],
            data.fault_labels, args.budget)
        print(f"  {ae.name:<18} done")

        flat = lambda ends: np.stack(
            [data.Xz[t - args.window + 1:t + 1].reshape(args.window, -1) for t in ends])
        lstm = LSTMAutoEncoder(window=args.window, device=device)
        lstm.fit_sequences(flat(tr_ends))
        run_s, node_s = lstm.score_sequences(flat(te_ends), data.n_nodes)
        results[lstm.name] = evaluate_scores(
            lab["y_type"], lab["y_root"], run_s, node_s, lab["severity"],
            data.fault_labels, args.budget)
        print(f"  {lstm.name:<18} done")

    save_json(args.out / f"results_{args.tag}.json", results)
    print_table(results)
    return results


def print_table(results: Dict[str, Dict]) -> None:
    hdr = f"{'method':<20}{'AUROC':>8}{'AUPRC':>8}{'bestF1':>8}{'F1@1%':>8}" \
          f"{'Hit@1':>8}{'Hit@3':>8}{'MRR':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    order = sorted(results, key=lambda k: -results[k]["detection"].get("auprc", 0))
    for name in order:
        r = results[name]
        d, l = r["detection"], r["localization"]
        print(f"{name:<20}{d.get('auroc', float('nan')):>8.3f}"
              f"{d.get('auprc', float('nan')):>8.3f}"
              f"{d.get('best_f1', float('nan')):>8.3f}"
              f"{d.get('f1@0.01', float('nan')):>8.3f}"
              f"{l.get('hit@1', float('nan')):>8.3f}"
              f"{l.get('hit@3', float('nan')):>8.3f}"
              f"{l.get('mrr', float('nan')):>8.3f}")


if __name__ == "__main__":
    main()
