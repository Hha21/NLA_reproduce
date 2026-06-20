"""
Activation Verbalizer (AV) — the encoder half of the NLA.

Architecture (per reference design.md):
  - Base: full Qwen2.5-0.5B (all 24 layers + LM head), fully trainable
  - Injection: rare Unicode token ㊗ is reserved as a single-token placeholder.
    At forward time its embedding slot is overwritten with h_l normalised to
    L2 norm = sqrt(d_model)  (injection_scale = sqrt_d_model, reference default).
  - Prompt (masked from loss): "Explain: <concept>㊗</concept>\n<explanation>"
  - SFT target: "{text_truncated}</explanation>"
    Uses the original text — not the LLM summary — for the strongest available signal.

Data split: AV SFT trains on the second half of the dataset (rows n//2 .. n-1).
The first half is reserved for AR SFT so the two warm-starts see disjoint examples.
"""

import math

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from src.config import DTYPE, INJECT_TOKEN, MODEL_ID
from src.model import load_tokenizer


class Verbalizer(nn.Module):
    def __init__(self, base, inject_token_id: int, d_model: int):
        super().__init__()
        self.base            = base
        self.inject_token_id = inject_token_id
        self.injection_scale = math.sqrt(d_model)   # sqrt_d_model per reference

    def _inject_activation(
        self,
        input_ids: torch.Tensor,   # (B, T)
        activation: torch.Tensor,  # (B, d_model) float32
    ) -> torch.Tensor:
        """Replace the ㊗ embedding slot with the L2-normalised activation."""
        embeds = self.base.get_input_embeddings()(input_ids)          # (B, T, d)
        norms      = activation.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        act_normed = (activation * (self.injection_scale / norms)).to(embeds.dtype)  # (B, d)
        # One ㊗ per sequence; argmax gives its position even if it appears only once.
        inject_pos = (input_ids == self.inject_token_id).long().argmax(dim=1)  # (B,)
        batch_idx  = torch.arange(input_ids.shape[0], device=input_ids.device)
        embeds[batch_idx, inject_pos] = act_normed
        return embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        activation: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        embeds = self._inject_activation(input_ids, activation)
        return self.base(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
        )


def load_av(device: str):
    """
    Load full Qwen2.5-0.5B as AV base.

    Guarantees ㊗ is a single token: if the tokenizer already encodes it as one
    token, that ID is used; otherwise it is added as a new special token and the
    embedding table is resized. Returns (av, tokenizer).
    """
    tok = load_tokenizer()

    # Check whether ㊗ is already a single token in the vocabulary.
    existing = tok.encode(INJECT_TOKEN, add_special_tokens=False)
    needs_resize = len(existing) != 1
    if needs_resize:
        tok.add_special_tokens({"additional_special_tokens": [INJECT_TOKEN]})

    inject_id = tok.encode(INJECT_TOKEN, add_special_tokens=False)[0]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map=device,
    )
    if needs_resize:
        model.resize_token_embeddings(len(tok))

    model.train()
    d  = model.config.hidden_size
    av = Verbalizer(model, inject_id, d)
    return av, tok
