"""Masking strategies for self-supervised pretraining.

Two strategies, and the distinction between them is the whole design:

**Feature-group masking** hides one symptom family (all five completeness
features, say) at a node and timestep. Masking scattered individual scalars
would be far too easy, because ``null_rate_mean`` is trivially recoverable from
``null_rate_max`` sitting right beside it, so the model could solve the task by
learning within-vector correlations and never look at the pipeline at all.
Hiding a whole family forces it to use other nodes and other timesteps.

**Node masking** hides a node's entire profile at a timestep. To fill it in, the
model has to predict what a task *should* have produced given what its parents
produced, which is exactly the ancestor-consistency objective the root-cause
scorer is built on. A node whose output is predictable from its inputs is
behaving consistently with them; a node whose output is surprising *even given
healthy inputs* is where the incident started.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def build_group_index(feature_names: List[str],
                      groups: Dict[str, List[str]]) -> List[torch.Tensor]:
    """Column indices for each symptom family, as a list of index tensors."""
    return [torch.tensor([feature_names.index(f) for f in feats], dtype=torch.long)
            for feats in groups.values()]


def feature_group_mask(shape: Tuple[int, int, int, int],
                       group_index: List[torch.Tensor],
                       ratio: float, device) -> torch.Tensor:
    """Mask whole symptom families at random (node, timestep) cells.

    Returns a float mask ``[B, W, N, F]`` where 1 means "hidden, predict me".
    """
    b, w, n, f = shape
    g = len(group_index)
    # Choose which (batch, time, node, family) cells to hide.
    hit = torch.rand(b, w, n, g, device=device) < ratio
    mask = torch.zeros(b, w, n, f, device=device)
    for gi, idx in enumerate(group_index):
        mask[..., idx.to(device)] = torch.maximum(
            mask[..., idx.to(device)],
            hit[..., gi].unsqueeze(-1).float().expand(-1, -1, -1, idx.numel()),
        )
    return mask


def node_mask(shape: Tuple[int, int, int, int], ratio: float,
              device, last_step_bias: float = 0.5) -> torch.Tensor:
    """Mask entire node profiles.

    ``last_step_bias`` raises the masking rate on the final timestep, because
    that is the only step scored at inference, so the model should be practised at
    exactly the query it will be asked.
    """
    b, w, n, f = shape
    p = torch.full((b, w, n), ratio, device=device)
    p[:, -1, :] = min(1.0, ratio + last_step_bias)
    hit = torch.rand(b, w, n, device=device) < p
    return hit.unsqueeze(-1).float().expand(-1, -1, -1, f).contiguous()


def mixed_mask(shape, group_index, group_ratio: float, node_ratio: float,
               device, p_node: float = 0.4) -> torch.Tensor:
    """Per-sample choice between the two strategies.

    Choosing per sample rather than blending per cell keeps each training example
    a clean instance of one task; blending them produced a muddier signal in
    early runs because a node was often only partly hidden.
    """
    b = shape[0]
    use_node = (torch.rand(b, device=device) < p_node).view(b, 1, 1, 1)
    m_group = feature_group_mask(shape, group_index, group_ratio, device)
    m_node = node_mask(shape, node_ratio, device)
    return torch.where(use_node, m_node, m_group)
