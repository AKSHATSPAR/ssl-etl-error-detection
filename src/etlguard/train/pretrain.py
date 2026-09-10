"""Self-supervised pretraining.

    PYTHONPATH=src python -m etlguard.train.pretrain --epochs 40

**No labels are used for training or for model selection.** Checkpoints are
chosen on the validation *masked-reconstruction loss*, which needs no
annotations. Selecting the checkpoint on validation AUPRC would quietly turn
this into a weakly supervised method and inflate every number that follows.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from ..data.dataset import ETLData, make_loaders
from ..models.masking import build_group_index, mixed_mask
from ..models.st_transformer import ModelConfig, STTransformer
from ..simulator.profiler import FEATURE_GROUPS
from ..utils.common import fmt, pick_device, save_json, set_seed


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Normalised temperature-scaled cross entropy over 2B augmented views."""
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
    n = z1.shape[0]
    sim = (z @ z.t()) / temperature
    sim.fill_diagonal_(-1e9)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def augment(x: torch.Tensor, sigma: float = 0.08, scale: float = 0.06
            ) -> torch.Tensor:
    """Jitter and per-feature rescaling.

    Kept deliberately weak: a strong augmentation on this data would *manufacture*
    something indistinguishable from a real fault (amplitude scaling is a unit
    change; heavy dropout is a null spike), and the contrastive loss would then
    be training the model to consider real faults normal.
    """
    s = 1.0 + scale * torch.randn(x.shape[0], 1, 1, x.shape[-1], device=x.device)
    return x * s + sigma * torch.randn_like(x)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Self-supervised pretraining")
    p.add_argument("--data", type=Path, default=Path("data/etl_v1"))
    p.add_argument("--out", type=Path, default=Path("runs/ssl"))
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--group-mask-ratio", type=float, default=0.35)
    p.add_argument("--node-mask-ratio", type=float, default=0.20)
    p.add_argument("--p-node-mask", type=float, default=0.40)
    p.add_argument("--contrastive-weight", type=float, default=0.0,
                   help="ablation: >0 enables the contrastive view objective")
    p.add_argument("--contrastive-subbatch", type=int, default=32,
                   help="contrastive costs 2 extra forward passes; run it on a slice")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--no-dag-bias", action="store_true",
                   help="ablation: remove the DAG attention bias")
    p.add_argument("--no-calendar", action="store_true",
                   help="ablation: hide hour-of-day / day-of-week")
    p.add_argument("--healthy-train", action="store_true",
                   help="ablation: train on clean windows only (the usual shortcut)")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="auto")
    return p


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """MSE over hidden entries only, since predicting the visible ones is free."""
    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def run_epoch(model, loader, group_index, args, device, optim=None, sched=None
              ) -> Dict[str, float]:
    train = optim is not None
    model.train(train)
    totals = {"loss": 0.0, "recon": 0.0, "contrast": 0.0}
    n = 0

    for batch in loader:
        x = batch["x"].to(device)
        cal = batch["cal"].to(device)
        if args.no_calendar:
            cal = torch.zeros_like(cal)

        mask = mixed_mask(x.shape, group_index, args.group_mask_ratio,
                          args.node_mask_ratio, device, args.p_node_mask)

        with torch.set_grad_enabled(train):
            pred = model(x, mask, cal)
            recon = masked_mse(pred, x, mask)
            loss = recon
            contrast = torch.zeros((), device=device)
            if args.contrastive_weight > 0:
                k = min(args.contrastive_subbatch, x.shape[0])
                xs, cs = x[:k], cal[:k]
                zero = torch.zeros_like(xs)
                z1 = model.embed(augment(xs), zero, cs)
                z2 = model.embed(augment(xs), zero, cs)
                contrast = nt_xent(z1, z2, args.temperature)
                loss = loss + args.contrastive_weight * contrast

        if train:
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            if sched is not None:
                sched.step()

        bs = x.shape[0]
        totals["loss"] += loss.item() * bs
        totals["recon"] += recon.item() * bs
        totals["contrast"] += float(contrast) * bs
        n += bs

    return {k: v / max(n, 1) for k, v in totals.items()}


def main(argv=None) -> Path:
    args = build_argparser().parse_args(argv)
    set_seed(args.seed)
    device = pick_device(args.device)

    data = ETLData.load(args.data)
    loaders = make_loaders(data, window=args.window, batch_size=args.batch_size,
                           healthy_train=args.healthy_train)
    group_index = build_group_index(data.feature_names, FEATURE_GROUPS)

    cfg = ModelConfig(
        n_nodes=data.n_nodes, n_features=data.n_features, window=args.window,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, dag_attention_bias=not args.no_dag_bias)
    model = STTransformer(cfg, data.adjacency).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    steps = max(1, len(loaders["train"]) * args.epochs)
    warm = min(args.warmup, max(1, steps // 10))

    def lr_at(step: int) -> float:
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, steps - warm)
        return 0.5 * (1.0 + np.cos(np.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_at)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  params={model.n_parameters():,}  "
          f"train_windows={len(loaders['train'].dataset)}  "
          f"val_windows={len(loaders['val'].dataset)}")

    history, best, best_epoch, bad = [], float("inf"), -1, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, loaders["train"], group_index, args, device, optim, sched)
        va = run_epoch(model, loaders["val"], group_index, args, device)
        row = {"epoch": epoch, "lr": optim.param_groups[0]["lr"],
               **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        marker = ""
        # Selection on reconstruction loss only: no labels anywhere in this loop.
        if va["recon"] < best - 1e-5:
            best, best_epoch, bad = va["recon"], epoch, 0
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg),
                        "args": vars(args)}, args.out / "best.pt")
            marker = "  *"
        else:
            bad += 1
        print(f"epoch {epoch:>3}  train_recon={tr['recon']:.4f}  "
              f"val_recon={va['recon']:.4f}  contrast={tr['contrast']:.3f}{marker}")
        if bad >= args.patience:
            print(f"early stop at epoch {epoch} (no val improvement for {bad})")
            break

    elapsed = time.time() - t0
    save_json(args.out / "history.json",
              {"history": history, "best_epoch": best_epoch,
               "best_val_recon": best, "seconds": round(elapsed, 1),
               "args": vars(args), "model_cfg": asdict(cfg),
               "n_parameters": model.n_parameters()})
    print(f"\nbest val_recon={best:.4f} at epoch {best_epoch}  "
          f"({elapsed:.0f}s)  ->  {args.out/'best.pt'}")
    return args.out / "best.pt"


if __name__ == "__main__":
    main()
