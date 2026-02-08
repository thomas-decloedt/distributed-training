"""Pydantic models for LoRA data — strict typing, no dict I/O."""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict


class TokenizedExample(BaseModel):
    """Single tokenized example: input_ids, attention_mask, labels."""

    model_config = ConfigDict(frozen=True)

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


class TokenizedBatch(BaseModel):
    """Batched tokenized tensors for model forward pass."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
