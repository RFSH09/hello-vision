"""Multiple-instance learning over a session: frame features -> per-SKU presence and count.

Needs torch (``uv sync --extra vision``). Each class has its own attention over the frames,
so the weights show which moments made the model predict a product.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

MAX_COUNT = 3  # counts are 0..3, where 3 means "3 or more"


class SessionMIL(nn.Module):
    """Attention-pooling temporal model.

    Args:
        classes: Number of product classes.
        dim: Backbone token dimension.
        width: Model width.
        layers: Temporal transformer layers.
        dropout: Dropout rate.
    """

    def __init__(
        self, classes: int, dim: int = 384, width: int = 192, layers: int = 2, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU())
        self.token_pos = nn.Parameter(torch.zeros(2 * 17, width))
        self.token_score = nn.Linear(width, 1)
        self.camera = nn.Embedding(2, width)
        self.time = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(
            width, 4, 2 * width, dropout, "gelu", batch_first=True, norm_first=True
        )
        self.temporal = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(classes, width) * 0.02)
        self.presence = nn.Parameter(torch.randn(classes, width) * 0.02)
        self.presence_bias = nn.Parameter(torch.zeros(classes))
        self.count = nn.Linear(2 * width, MAX_COUNT)  # counts 1..MAX_COUNT, given present
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        full: torch.Tensor,
        crop: torch.Tensor,
        cam: torch.Tensor,
        time: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, Any]:
        """Run a padded batch.

        Args:
            full: ``(B, F, 17, dim)`` whole-frame tokens.
            crop: ``(B, F, 17, dim)`` moving-region tokens.
            cam: ``(B, F)`` camera index.
            time: ``(B, F)`` time as a fraction of the session.
            mask: ``(B, F)`` true for real frames.

        Returns:
            ``presence`` logits ``(B, C)``, ``count`` logits ``(B, C, MAX_COUNT)`` and
            ``attention`` ``(B, C, F)``.
        """
        tokens = self.project(torch.cat([full, crop], dim=2)) + self.token_pos
        weights = self.token_score(tokens).softmax(dim=2)
        frames = (weights * tokens).sum(dim=2)
        frames = frames + self.camera(cam) + self.time(time.unsqueeze(-1))
        empty = ~mask.any(dim=1)
        safe = mask.clone()
        safe[empty, 0] = True  # a session without motion still needs one attendable slot
        # Temporal context only decides WHERE to look. What is pooled is each frame's own
        # content, so a high attention weight really points at the evidence.
        context = self.temporal(frames, src_key_padding_mask=~safe)
        scores = torch.einsum("cw,bfw->bcf", self.queries, context) / frames.shape[-1] ** 0.5
        attention = scores.masked_fill(~safe.unsqueeze(1), float("-inf")).softmax(dim=-1)
        pooled = self.drop(torch.einsum("bcf,bfw->bcw", attention, frames))
        presence = (pooled * self.presence).sum(-1) + self.presence_bias
        # counting also needs to see over time (several items may appear one after another)
        over_time = torch.einsum("bcf,bfw->bcw", attention, context)
        count = self.count(torch.cat([pooled, over_time], dim=-1))
        return {"presence": presence, "count": count, "attention": attention}


def mil_loss(
    output: dict[str, Any], counts: torch.Tensor, pos_weight: torch.Tensor
) -> torch.Tensor:
    """Presence BCE (positives up-weighted for rare SKUs) + count cross-entropy when present.

    Args:
        output: Model output.
        counts: ``(B, C)`` true quantities.
        pos_weight: ``(C,)`` weight of positive samples.
    """
    present = (counts > 0).float()
    loss = nn.functional.binary_cross_entropy_with_logits(
        output["presence"], present, pos_weight=pos_weight
    )
    chosen = counts > 0
    if chosen.any():
        target = counts[chosen].clamp(max=MAX_COUNT).long() - 1
        loss = loss + 0.5 * nn.functional.cross_entropy(output["count"][chosen], target)
    return loss


def decode(output: dict[str, Any], threshold: float = 0.5) -> torch.Tensor:
    """Predicted quantities ``(B, C)`` (0 when absent)."""
    present = output["presence"].sigmoid() >= threshold
    return torch.where(present, output["count"].argmax(-1) + 1, 0)
