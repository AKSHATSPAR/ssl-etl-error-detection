"""Live demo: take one broken pipeline run and diagnose it end to end.

    PYTHONPATH=src python scripts/demo.py

Picks a real incident from the held-out test split, shows what the pipeline
looked like, and walks through the three questions the system answers: is this
run broken, which task broke it, and what kind of break is it. Ground truth is
printed at the end so the answer can be checked rather than taken on trust.

Runs in about thirty seconds on a laptop and needs only a trained checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etlguard.data.dataset import TEST, TRAIN, ETLData, ETLWindowDataset
from etlguard.eval.explain import family_scores
from etlguard.eval.scoring import NodeCalibration, descendant_mask, score_loader
from etlguard.models.st_transformer import ModelConfig, STTransformer
from etlguard.simulator.faults import SYMPTOM_FAMILY
from etlguard.simulator.profiler import FEATURE_GROUPS
from etlguard.utils.common import pick_device

from torch.utils.data import DataLoader

BOLD, DIM, RED, GRN, YEL, CYN, OFF = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[0m")


def bar(value: float, width: int = 28, cap: float = 1.0) -> str:
    n = int(round(width * min(max(value, 0.0) / cap, 1.0)))
    return "█" * n + "·" * (width - n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/etl_v1"))
    ap.add_argument("--ckpt", type=Path, default=Path("runs/ssl/best.pt"))
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--n", type=int, default=3, help="how many incidents to show")
    ap.add_argument("--fault", type=str, default=None,
                    help="show a specific fault type, e.g. unit_change")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device("auto")
    print(f"{DIM}loading model and data ({device}) ...{OFF}")
    data = ETLData.load(args.data)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = STTransformer(ModelConfig(**ck["cfg"]), data.adjacency).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # Calibration reference from unlabelled training runs, as at evaluation time.
    cal_ds = ETLWindowDataset(data, TRAIN, args.window, stride=16)
    cal_b = score_loader(model, DataLoader(cal_ds, batch_size=64), device,
                         data.n_nodes, data.adjacency)
    calib = NodeCalibration.fit(cal_b)

    test_ds = ETLWindowDataset(data, TEST, args.window)
    ends = test_ds.ends
    candidates = np.where(data.y_type[ends] > 0)[0]
    if args.fault:
        want = data.fault_labels.index(args.fault)
        candidates = candidates[data.y_type[ends][candidates] == want]
        if candidates.size == 0:
            raise SystemExit(f"no test incidents of type {args.fault}")

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(candidates, size=min(args.n, candidates.size), replace=False)

    healthy = np.where(data.y_type[ends] == 0)[0]
    picks = list(picks) + [int(rng.choice(healthy))]     # one healthy run for contrast

    gidx = {f: np.array([data.feature_names.index(x) for x in feats])
            for f, feats in FEATURE_GROUPS.items()}
    desc = torch.from_numpy(descendant_mask(data.adjacency)).to(device)

    sub = torch.utils.data.Subset(test_ds, [int(i) for i in picks])
    b = score_loader(model, DataLoader(sub, batch_size=len(picks)), device,
                     data.n_nodes, data.adjacency)
    loc = calib.apply(b.e_anc, "e_anc")
    detect = b.e_anc.max(axis=1)

    # A threshold at the 95th percentile of calibration runs: about one alert in
    # twenty, which is a plausible on-call budget.
    thresh = float(np.percentile(cal_b.e_anc.max(axis=1), 95))

    for row in range(len(picks)):
        t = int(b.t[row])
        ts = np.datetime64(int(data.timestamps[t]), "s")
        truth_type = data.fault_labels[int(b.y_type[row])]
        broken = b.y_type[row] > 0

        print(f"\n{BOLD}{'=' * 74}{OFF}")
        print(f"{BOLD}PIPELINE RUN {ts}{OFF}   {DIM}(test split, index {t}){OFF}")
        print(f"{BOLD}{'=' * 74}{OFF}")

        # --- Q1: is it broken? --- #
        fired = detect[row] >= thresh
        verdict = f"{RED}ALERT{OFF}" if fired else f"{GRN}healthy{OFF}"
        print(f"\n{BOLD}1. Is this run broken?{OFF}   -> {verdict}")
        print(f"   anomaly score {detect[row]:.3f}   alert threshold {thresh:.3f}")

        if not fired:
            print(f"\n   {DIM}ground truth: {truth_type}{OFF}")
            continue

        # --- Q2: which task? --- #
        order = np.argsort(-loc[row])
        print(f"\n{BOLD}2. Which task caused it?{OFF}")
        for rank, i in enumerate(order[:4], 1):
            mark = f"{CYN}<- most likely origin{OFF}" if rank == 1 else ""
            print(f"   {rank}. {data.node_names[i]:<24} "
                  f"{bar(loc[row, i])} {loc[row, i]:.3f}  {mark}")

        # --- Q3: what kind of break? --- #
        top_node = int(order[0])
        fams, vals = family_scores(b.feat_err[row, top_node], gidx)
        fam_order = np.argsort(-vals)
        feats = gidx[fams[int(fam_order[0])]]
        top_feat = feats[int(np.argmax(b.feat_err[row, top_node, feats]))]
        print(f"\n{BOLD}3. What kind of break?{OFF}")
        for j in fam_order[:3]:
            print(f"   {fams[int(j)]:<26} {bar(vals[int(j)], cap=float(vals.max()))} "
                  f"{vals[int(j)]:.2f}")
        print(f"\n   {YEL}Diagnosis:{OFF} {data.node_names[top_node]} shows an anomaly "
              f"in the {BOLD}{fams[int(fam_order[0])]}{OFF} family,")
        print(f"   driven most by {BOLD}{data.feature_names[top_feat]}{OFF}.")

        # --- did it get it right? --- #
        true_root = int(np.argmax(b.y_root[row]))
        ok_node = top_node == true_root
        want = SYMPTOM_FAMILY.get(truth_type, set())
        ok_fam = fams[int(fam_order[0])] in want
        print(f"\n   {DIM}{'-' * 68}{OFF}")
        print(f"   {DIM}ground truth: fault '{truth_type}' injected at "
              f"'{data.node_names[true_root]}'{OFF}")
        print(f"   {DIM}severity {b.severity[row]:.2f}, "
              f"{int(b.y_anom[row].sum())} of {data.n_nodes} tasks affected{OFF}")
        print(f"   task {'CORRECT' if ok_node else 'wrong'}   "
              f"symptom family {'CORRECT' if ok_fam else 'wrong'}"
              f"{'' if ok_fam else f' (expected one of {sorted(want)})'}")

    print(f"\n{DIM}Localization is right 79.7% of the time on the full test split. "
          f"Re-run with --seed to see different incidents,{OFF}")
    print(f"{DIM}or --fault <name> to pick a type, e.g. "
          f"--fault unit_change or --fault row_loss_truncation (the hard one).{OFF}\n")


if __name__ == "__main__":
    main()
