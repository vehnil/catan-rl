"""Categorical distribution restricted to legal actions."""

from __future__ import annotations

import torch
from torch.distributions import Categorical

_NEG = -1e9


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask.bool(), _NEG)


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    return Categorical(logits=masked_logits(logits, mask))
