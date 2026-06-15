"""
Phase 1 — activation extraction harness.

Design decisions (document these; wrong choices here silently poison later phases):
  - What we capture: x_out from layers[PROBE_LAYER] — the residual-stream vector
    after both the attention and FFN residual adds within that layer. This is the
    standard "residual stream at layer L" used in the NLA paper and mech-interp
    literature generally. Verified below to equal output_hidden_states[PROBE_LAYER+1].
  - Token position: last token (-1). It attends to the full context, so its
    residual stream summarises everything the model has processed so far.
  - Gradient: .detach() in the hook — T is frozen and we never want a graph here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import PROBE_LAYER, DEVICE
from src.model import load_tokenizer, load_target

tok    = load_tokenizer()
target = load_target(DEVICE)

# --- Hook -----------------------------------------------------------------
# Registered directly on layers[PROBE_LAYER]: no index arithmetic, no ambiguity.
# Only this one layer's output is stored; the other 23 are never kept in memory.

_acts = {}

def _hook(module, inp, out):
    _acts["resid"] = out[0].detach()   # out[0]: (batch, seq, hidden_size)

target.model.layers[PROBE_LAYER].register_forward_hook(_hook)

# --------------------------------------------------------------------------

def get_activation(text: str, token_pos: int = -1) -> torch.Tensor:
    """Return the residual-stream vector at PROBE_LAYER for one token position."""
    inputs = tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        target(**inputs)
    return _acts["resid"][0, token_pos]   # (hidden_size,)


if __name__ == "__main__":
    text = "The transformer model processes tokens in parallel."

    act = get_activation(text)
    print(f"Activation shape: {act.shape}")
    print(f"Activation dtype: {act.dtype}")
    print(f"Activation norm:  {act.norm():.4f}")

    # Sanity check: hook must agree with output_hidden_states
    inputs = tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = target(**inputs, output_hidden_states=True)

    via_hs   = out.hidden_states[PROBE_LAYER + 1][0, -1]
    via_hook = get_activation(text)
    print(f"\nMax diff hook vs hidden_states: {(via_hook - via_hs).abs().max():.6f}")
