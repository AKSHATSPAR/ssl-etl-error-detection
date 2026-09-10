"""The self-supervised backbone: a DAG-aware spatio-temporal masked autoencoder.

The model is told *which* entries are hidden (the mask is a model input, not
just a loss weight), so filling them in is a well-posed question rather than a
guess about whether a zero means "healthy" or "missing".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .layers import (N_RELATIONS, STBlock, dag_relation_matrix,
                     sinusoidal_encoding)


@dataclass
class ModelConfig:
    n_nodes: int
    n_features: int
    window: int = 32
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    n_calendar: int = 4
    dag_attention_bias: bool = True
    proj_hidden: int = 128           # head for the contrastive objective


class STTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, adjacency: np.ndarray):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # The mask is concatenated to the input, so the model always knows what
        # it is being asked to predict.
        self.input_proj = nn.Linear(2 * cfg.n_features + cfg.n_calendar, d)
        self.node_emb = nn.Parameter(torch.randn(cfg.n_nodes, d) * 0.02)
        self.register_buffer("time_enc", sinusoidal_encoding(cfg.window, d),
                             persistent=False)

        rel = dag_relation_matrix(adjacency)
        self.register_buffer("relation", torch.from_numpy(rel), persistent=False)
        if cfg.dag_attention_bias:
            # One learned scalar per (head, relation): the model decides how much
            # a task should listen to its parents versus unrelated branches.
            self.rel_bias = nn.Parameter(torch.zeros(cfg.n_heads, N_RELATIONS))
        else:
            self.rel_bias = None

        self.blocks = nn.ModuleList([
            STBlock(d, cfg.n_heads, cfg.ff_mult, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.n_features))
        self.proj_head = nn.Sequential(
            nn.Linear(d, cfg.proj_hidden), nn.GELU(),
            nn.Linear(cfg.proj_hidden, cfg.proj_hidden))

    # ------------------------------------------------------------------ #

    def _dag_bias(self) -> Optional[torch.Tensor]:
        if self.rel_bias is None:
            return None
        # [heads, N, N] gathered from the learned per-relation scalars.
        return self.rel_bias[:, self.relation]

    def encode(self, x: torch.Tensor, mask: torch.Tensor, cal: torch.Tensor,
               store_attn: bool = False) -> torch.Tensor:
        """``x, mask -> [B, W, N, d]`` latent states."""
        b, w, n, f = x.shape
        x_in = x * (1.0 - mask)
        cal_b = cal.unsqueeze(2).expand(b, w, n, cal.shape[-1])
        h = self.input_proj(torch.cat([x_in, mask, cal_b], dim=-1))
        h = h + self.node_emb.view(1, 1, n, -1)
        h = h + self.time_enc[:w].view(1, w, 1, -1)

        bias = self._dag_bias()
        for blk in self.blocks:
            h = blk(h, dag_bias=bias, store_attn=store_attn)
        return self.norm(h)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cal: torch.Tensor,
                store_attn: bool = False) -> torch.Tensor:
        return self.head(self.encode(x, mask, cal, store_attn=store_attn))

    def embed(self, x: torch.Tensor, mask: torch.Tensor,
              cal: torch.Tensor) -> torch.Tensor:
        """Window-level embedding for the contrastive objective."""
        h = self.encode(x, mask, cal)
        return self.proj_head(h.mean(dim=(1, 2)))

    # ------------------------------------------------------------------ #

    def attention_over_nodes(self) -> Optional[torch.Tensor]:
        """Structural attention from the last forward pass, averaged over layers.

        Used by the explainability pass to show which upstream path the model
        looked at when it decided a node was surprising.
        """
        mats = [blk.attn_s.last_attention for blk in self.blocks
                if blk.attn_s.last_attention is not None]
        if not mats:
            return None
        return torch.stack(mats).mean(dim=(0, 2))    # [B*W, N, N]

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
