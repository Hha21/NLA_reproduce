"""
Activation Reconstructor (AR) — the decoder half of the NLA.

Architecture:
  - Base: the transformer body of T (Qwen2Model, no LM head), trainable or frozen
  - Head: a single Linear(d_model, d_model) that is always trainable
  - Input:  text tokens (original text for oracle; AV descriptions during GRPO)
  - Output: predicted activation â, shape (batch, d_model)

For the oracle baseline, freeze_base=True so only the head trains.
The residual-stream argument makes this sufficient: the last-layer hidden
state is a downstream linear function of layer 16, so the head can learn
to invert that relationship without touching the transformer weights.

During GRPO (freeze_base=False), the full model fine-tunes as the description
distribution drifts away from natural text toward AV's learned summaries.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from src.config import DTYPE, MODEL_ID


class Reconstructor(nn.Module):
    def __init__(self, base, d_model: int):
        super().__init__()
        self.base = base
        self.head = nn.Linear(d_model, d_model, bias=False)

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
    Load a fresh copy of T and wrap it as AR.

    We load the full CausalLM then discard the LM head — AR only needs the
    transformer body to produce hidden states.
    """
    full_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map=device,
    )
    # LM head is never needed for AR; freeze it to keep it out of the optimizer.
    full_model.lm_head.requires_grad_(False)

    base = full_model.model   # Qwen2Model: embed + 24 decoder layers + final norm
    base.train()

    if freeze_base:
        base.requires_grad_(False)

    d  = full_model.config.hidden_size
    ar = Reconstructor(base, d)
    ar.head = ar.head.to(device=device, dtype=DTYPE)   # match transformer's bf16
    return ar
