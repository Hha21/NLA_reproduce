"""Phase 0 — verify model loads and confirm shapes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import PROBE_LAYER, DEVICE
from src.model import load_tokenizer, load_target

tok    = load_tokenizer()
target = load_target(DEVICE)

cfg = target.config
print(f"hidden_size:       {cfg.hidden_size}")
print(f"num_hidden_layers: {cfg.num_hidden_layers}")
print(f"probe layer:       {PROBE_LAYER}")

text   = "The transformer model processes tokens in parallel."
inputs = tok(text, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    out = target(**inputs, output_hidden_states=True)

h = out.hidden_states[PROBE_LAYER + 1]
print(f"\nHidden state at layer {PROBE_LAYER}: {h.shape}")

print("\nSanity generation:")
gen_ids = target.generate(**inputs, max_new_tokens=20)
print(tok.decode(gen_ids[0], skip_special_tokens=True))