"""Train and evaluate the seed repeats and the ablations, one after another.

    PYTHONPATH=src python scripts/run_experiments.py --only ablations

Every configuration gets the same training budget. Giving an ablation fewer
epochs would confound "this component helps" with "this run was trained less",
which is the easiest way to manufacture a result that is not there.

Baselines are skipped for these runs since they do not depend on the model, so
each configuration costs one training run plus about ninety seconds of scoring.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (tag, extra training flags, requires). The tag prefix decides which table a run
# lands in; `requires` names a path that must exist for the job to be runnable.
SEEDS = [
    ("seed13", ["--seed", "13"], None),
    ("seed29", ["--seed", "29"], None),
]

ABLATIONS = [
    ("abl_no_dag_bias", ["--no-dag-bias"], None),
    ("abl_no_calendar", ["--no-calendar"], None),
    ("abl_clean_train", ["--healthy-train"], None),
    ("abl_contrastive", ["--contrastive-weight", "0.1"], None),
    # Longer training is the one configuration that changed anything, so it has
    # to be reproducible rather than a number quoted from a run nobody can repeat.
    ("abl_longer_training", ["--epochs", "90", "--patience", "15"], None),
    # The calendar ablation was first measured on timestamps that were 1000x too
    # small, which made the features meaningless and the ablation vacuous. The
    # corrected result needs a dataset built after that fix.
    ("abl_calendar_fixed", ["--data", "data/etl_v2"], Path("data/etl_v2")),
]


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                           env={**__import__("os").environ, "PYTHONPATH": "src"})
    return p.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["all", "seeds", "ablations"], default="all")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--python", type=str, default=".venv/bin/python")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--results", type=Path, default=Path("artifacts/results"))
    ap.add_argument("--logs", type=Path, default=Path("artifacts/logs"))
    args = ap.parse_args()

    jobs = []
    if args.only in ("all", "seeds"):
        jobs += SEEDS
    if args.only in ("all", "ablations"):
        jobs += ABLATIONS

    print(f"{len(jobs)} configurations, {args.epochs} epochs each "
          f"unless a job overrides it\n")
    for i, (tag, flags, requires) in enumerate(jobs, 1):
        if requires is not None and not (ROOT / requires).exists():
            print(f"[{i}/{len(jobs)}] {tag} SKIPPED, needs {requires}.\n"
                  f"    Create it with:  PYTHONPATH=src python -m "
                  f"etlguard.data.build_dataset --n-batches 8000 --out {requires}")
            continue
        out = args.runs / tag
        t0 = time.time()
        # A job may override the shared budget; its own flags win, because
        # argparse takes the last occurrence.
        print(f"[{i}/{len(jobs)}] {tag} training ...", flush=True)
        rc = run([args.python, "-u", "-m", "etlguard.train.pretrain",
                  "--epochs", str(args.epochs), "--out", str(out)] + flags,
                 args.logs / f"train_{tag}.log")
        if rc != 0:
            print(f"    training FAILED (rc={rc}), see {args.logs}/train_{tag}.log")
            continue

        print(f"[{i}/{len(jobs)}] {tag} evaluating ...", flush=True)
        data_flags = []
        if "--data" in flags:
            data_flags = ["--data", flags[flags.index("--data") + 1]]
        rc = run([args.python, "-u", "-m", "etlguard.eval.run_eval",
                  "--ckpt", str(out / "best.pt"), "--tag", tag,
                  "--out", str(args.results), "--skip-baselines"] + data_flags,
                 args.logs / f"eval_{tag}.log")
        status = "ok" if rc == 0 else f"eval FAILED (rc={rc})"
        print(f"    {status}  [{time.time() - t0:.0f}s]", flush=True)

    print("\ndone. build the tables with scripts/results_report.py")


if __name__ == "__main__":
    main()
