"""
Activation Reconstructor (AR) — the decoder half of the NLA.

Architecture (per paper):
  - Base: the transformer body of T truncated to its first PROBE_LAYER layers.
    The final-layer norm is removed so last_hidden_state = raw x_l (matching
    the hook-captured activation stored in the dataset).
  - Head: affine Linear(d_model, d_model, bias=True), always trainable.
  - Input:  AR_PREFIX + z + AR_SUFFIX  (z = original text for oracle; AV
            description during GRPO). The last-token hidden state at layer l
            is the "value head" position.
  - Output: predicted activation â, shape (batch, d_model).

Truncating to l layers saves ~33% memory and removes the need to map
backwards through layers l+1..24, which a single linear layer cannot do.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from src.config import DTYPE, MODEL_ID, PROBE_LAYER


class Reconstructor(nn.Module):
    def __init__(self, base, d_model: int):
        super().__init__()
        self.base = base
        self.head = nn.Linear(d_model, d_model, bias=True)   # affine, per paper

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:, -1, :]   # (batch, d_model) last-token pool
        return self.head(h)                    # â  (batch, d_model)


def load_ar(device: str, freeze_base: bool = True) -> Reconstructor:
    """
    Load a fresh copy of T, truncate it to PROBE_LAYER, and wrap as AR.

    Truncation: keep only layers[0..PROBE_LAYER], discard the rest and the
    final norm so that last_hidden_state == raw x_l (same quantity the hook
    captures during activation extraction).
    """
    full_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map=device,
    )

    base = full_model.model   # Qwen2Model

    # Truncate to first PROBE_LAYER+1 decoder layers and strip the final norm.
    # The norm was fitted after 24 layers; without it, last_hidden_state is the
    # raw residual stream at layer PROBE_LAYER, matching what the hook captures.
    base.layers = nn.ModuleList(list(base.layers)[: PROBE_LAYER + 1])
    base.norm   = nn.Identity()

    base.train()
    if freeze_base:
        base.requires_grad_(False)

    d  = full_model.config.hidden_size
    ar = Reconstructor(base, d)
    ar.head = ar.head.to(device=device, dtype=DTYPE)
    return ar
