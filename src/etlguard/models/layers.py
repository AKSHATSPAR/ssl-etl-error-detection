"""Attention blocks for the spatio-temporal encoder.

The input tensor is ``[B, W, N, d]``: batch, window, pipeline node, channel.
Attending over ``W*N`` jointly would cost ``O((WN)^2)`` per layer for no good
reason: the node axis is short (9) and the time axis is long (32), and the two
carry different kinds of structure. So each block attends over time and over the
DAG separately, at ``O(W^2 + N^2)``.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Relation of node j to node i, used to bias structural attention.
REL_SELF, REL_PARENT, REL_CHILD, REL_ANCESTOR, REL_DESCENDANT, REL_NONE = range(6)
N_RELATIONS = 6


def dag_relation_matrix(adjacency: np.ndarray) -> np.ndarray:
    """Classify every ordered node pair by its relationship in the DAG.

    ``R[i, j]`` says what *j* is to *i*: its direct parent, a distant ancestor, a
    descendant, or unrelated. The encoder learns one attention bias per relation
    per head, so it can decide for itself how much a task should listen to its
    immediate upstream versus the rest of the pipeline, rather than us hard-coding
    a mask that forbids attention outright.
    """
    n = adjacency.shape[0]
    reach = adjacency.astype(bool).copy()
    # Transitive closure (n is tiny, so the cubic loop is irrelevant).
    for k in range(n):
        reach |= np.outer(reach[:, k], reach[k, :])

    rel = np.full((n, n), REL_NONE, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i == j:
                rel[i, j] = REL_SELF
            elif adjacency[j, i]:
                rel[i, j] = REL_PARENT
            elif adjacency[i, j]:
                rel[i, j] = REL_CHILD
            elif reach[j, i]:
                rel[i, j] = REL_ANCESTOR
            elif reach[i, j]:
                rel[i, j] = REL_DESCENDANT
    return rel


class MultiHeadAttention(nn.Module):
    """Standard MHA with an optional additive bias of shape ``[heads, L, L]``."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self._last_attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None,
                store_attn: bool = False) -> torch.Tensor:
        b, L, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, L, self.h, self.dk).transpose(1, 2)
        k = k.view(b, L, self.h, self.dk).transpose(1, 2)
        v = v.view(b, L, self.h, self.dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if bias is not None:
            scores = scores + bias.unsqueeze(0)
        attn = scores.softmax(dim=-1)
        if store_attn:
            # Detached: this is for the explainability pass, never for the loss.
            self._last_attn = attn.detach()
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(b, L, d)
        return self.proj(out)

    @property
    def last_attention(self) -> Optional[torch.Tensor]:
        return self._last_attn


class STBlock(nn.Module):
    """One encoder layer: attend over time, then over the DAG, then a FFN."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.norm_t = nn.LayerNorm(d_model)
        self.attn_t = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_s = nn.LayerNorm(d_model)
        self.attn_s = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_f = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dag_bias: Optional[torch.Tensor] = None,
                store_attn: bool = False) -> torch.Tensor:
        b, w, n, d = x.shape

        # --- temporal: each node attends along its own history ---
        h = self.norm_t(x).permute(0, 2, 1, 3).reshape(b * n, w, d)
        h = self.attn_t(h)
        x = x + self.drop(h.view(b, n, w, d).permute(0, 2, 1, 3))

        # --- structural: each timestep attends across the DAG ---
        h = self.norm_s(x).reshape(b * w, n, d)
        h = self.attn_s(h, bias=dag_bias, store_attn=store_attn)
        x = x + self.drop(h.view(b, w, n, d))

        x = x + self.drop(self.ff(self.norm_f(x)))
        return x


def sinusoidal_encoding(length: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe
