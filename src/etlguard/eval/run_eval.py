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
from .scoring import (NodeCalibration, ScoreBundle,
                      calibrate_node_scores, score_loader)


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
    ap.add_argument("--baseline-device", type=str, default="cpu",
                    help="deep baselines run on CPU by default: LSTM on MPS is "
                         "unstable and these models are small enough not to care")
    ap.add_argument("--lstm-max-train", type=int, default=2500,
                    help="cap on LSTM training sequences; the full set is a "
                         "209MB tensor and the extra windows buy nothing")
    ap.add_argument("--lstm-epochs", type=int, default=15)
    ap.add_argument("--tag", type=str, default="ssl")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-scores", action="store_true",
                    help="cache the raw score arrays so scoring rules can be "
                         "compared offline without a 5-minute re-score")
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
    # Calibration runs on unlabelled training windows, so it adds no label use.
    cal_b = score_loader(model, loaders["calibration"], device, data.n_nodes,
                         data.adjacency)
    calib = NodeCalibration.fit(cal_b)
    val_b = score_loader(model, loaders["val"], device, data.n_nodes, data.adjacency)
    test_b = score_loader(model, loaders["test"], device, data.n_nodes, data.adjacency)
    print(f"scored calibration+val+test in {time.time() - t0:.1f}s")

    # Pooling rule chosen on validation, never on test.
    from sklearn.metrics import average_precision_score
    yv = val_b.is_incident
    rule_scores = {r: float(average_precision_score(yv, val_b.detection_score(r)))
                   for r in ScoreBundle.DETECTION_RULES}
    best_rule = max(rule_scores, key=rule_scores.get)
    print(f"detection rule selected on val AUPRC: {best_rule}  "
          f"({ {k: round(v, 3) for k, v in rule_scores.items()} })")

    loc_score = test_b.calibrated("e_anc", calib)
    ssl_res = evaluate_scores(test_b.y_type, test_b.y_root,
                              test_b.detection_score(best_rule),
                              loc_score, test_b.severity,
                              data.fault_labels, args.budget)
    ssl_res["detection_rule"] = best_rule
    ssl_res["val_rule_auprc"] = rule_scores

    # Every localizer below uses the same trained model, so the comparison
    # isolates the scoring rule rather than confounding it with a different model.
    variants = {
        "residual_only_raw": test_b.e_hist,
        "self_masked_only_raw": test_b.e_ctx,
        "ancestor_raw": test_b.e_anc,
        "residual_only_calibrated": test_b.calibrated("e_hist", calib),
        "self_masked_only_calibrated": test_b.calibrated("e_ctx", calib),
    }
    for name, sc in variants.items():
        ssl_res[f"localization_{name}"] = localization_metrics(test_b.y_root, sc)

    # Localization split by how far upstream the origin sits. If hiding only the
    # node itself lets its descendants leak the answer, that shows up here as a
    # collapse on shallow (source-side) origins and nowhere else.
    depth = np.zeros(data.n_nodes, dtype=int)
    for j in range(data.n_nodes):
        parents = np.where(data.adjacency[:, j] > 0)[0]
        depth[j] = 0 if parents.size == 0 else 1 + depth[parents].max()
    root_depth = np.where(test_b.y_root.sum(axis=1) > 0,
                          np.argmax(test_b.y_root, axis=1), -1)
    root_depth = np.array([depth[i] if i >= 0 else -1 for i in root_depth])
    ssl_res["localization_by_depth"] = {}
    for d in sorted(set(root_depth[root_depth >= 0])):
        m = root_depth == d
        if m.sum() < 3:
            continue
        ssl_res["localization_by_depth"][f"depth_{d}"] = {
            "ancestor_calibrated": localization_metrics(
                test_b.y_root[m], loc_score[m]),
            "ancestor_uncalibrated": localization_metrics(
                test_b.y_root[m], test_b.e_anc[m]),
            "self_masked_only": localization_metrics(
                test_b.y_root[m], test_b.e_ctx[m]),
            "deviation_magnitude": localization_metrics(
                test_b.y_root[m], test_b.e_hist[m]),
        }
    ssl_res["node_baseline_e_anc"] = {
        n: float(np.median(calib.reference["e_anc"][:, i]))
        for i, n in enumerate(data.node_names)}

    if args.save_scores:
        np.savez_compressed(
            args.out / f"scores_{args.tag}.npz",
            e_hist=test_b.e_hist, e_ctx=test_b.e_ctx, e_anc=test_b.e_anc,
            feat_err=test_b.feat_err, y_root=test_b.y_root, y_anom=test_b.y_anom,
            y_type=test_b.y_type, severity=test_b.severity, t=test_b.t,
            cal_e_hist=cal_b.e_hist, cal_e_ctx=cal_b.e_ctx, cal_e_anc=cal_b.e_anc,
            val_e_hist=val_b.e_hist, val_e_ctx=val_b.e_ctx, val_e_anc=val_b.e_anc,
            val_y_type=val_b.y_type, val_y_root=val_b.y_root)
        print(f"cached score arrays to {args.out}/scores_{args.tag}.npz")

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
            # Same percentile calibration the model gets, fitted the same way on
            # unlabelled training scores. It is worth more than the choice of
            # detector, so applying it to one side only would be the result.
            _, node_tr = b.score(f_tr[::4])
            node_cal = calibrate_node_scores(node_tr, node_s)
            results[b.name] = evaluate_scores(
                lab["y_type"], lab["y_root"], run_s, node_cal, lab["severity"],
                data.fault_labels, args.budget)
            results[b.name]["localization_uncalibrated"] = localization_metrics(
                lab["y_root"], node_s)
            _checkpoint(results, args.out, args.tag)
            print(f"  {b.name:<18} fitted+scored in {time.time() - t1:.1f}s")

        ae = DenseAutoEncoder(device=args.baseline_device).fit(f_tr)
        run_s, node_s = ae.score(f_te)
        _, node_tr = ae.score(f_tr[::4])
        results[ae.name] = evaluate_scores(
            lab["y_type"], lab["y_root"], run_s,
            calibrate_node_scores(node_tr, node_s), lab["severity"],
            data.fault_labels, args.budget)
        results[ae.name]["localization_uncalibrated"] = localization_metrics(
            lab["y_root"], node_s)
        _checkpoint(results, args.out, args.tag)
        print(f"  {ae.name:<18} done")

        def flat(ends):
            return np.stack([data.Xz[t - args.window + 1:t + 1].reshape(args.window, -1)
                             for t in ends]).astype(np.float32)

        # Stride the training windows rather than taking a prefix: consecutive
        # windows overlap by 31 of 32 steps, so a contiguous slice would cover
        # only a few weeks of the year.
        if len(tr_ends) > args.lstm_max_train:
            step = int(np.ceil(len(tr_ends) / args.lstm_max_train))
            lstm_ends = tr_ends[::step]
        else:
            lstm_ends = tr_ends
        t1 = time.time()
        lstm = LSTMAutoEncoder(window=args.window, epochs=args.lstm_epochs,
                               device=args.baseline_device)
        lstm.fit_sequences(flat(lstm_ends))
        run_s, node_s = lstm.score_sequences(flat(te_ends), data.n_nodes)
        _, node_tr = lstm.score_sequences(flat(lstm_ends), data.n_nodes)
        results[lstm.name] = evaluate_scores(
            lab["y_type"], lab["y_root"], run_s,
            calibrate_node_scores(node_tr, node_s), lab["severity"],
            data.fault_labels, args.budget)
        results[lstm.name]["localization_uncalibrated"] = localization_metrics(
            lab["y_root"], node_s)
        _checkpoint(results, args.out, args.tag)
        print(f"  {lstm.name:<18} done in {time.time() - t1:.1f}s "
              f"({len(lstm_ends)} train sequences)")

    save_json(args.out / f"results_{args.tag}.json", results)
    print_table(results)
    return results


def _checkpoint(results: Dict, out: Path, tag: str) -> None:
    """Persist after every method, so one crashing baseline costs one baseline."""
    save_json(out / f"results_{tag}.json", results)


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
