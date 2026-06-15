"""
Phase 0 — load the target model and confirm shapes.

Model: Qwen/Qwen2.5-0.5B
  hidden_size        : 896
  num_hidden_layers  : 24
  probe layer (2/3)  : ~16

This copy of the model is "T" in the NLA diagram — it will stay frozen.
We load it in bf16: half the memory of float32, and .backward() still works
through bf16 tensors, which matters when we later train AV and AR.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_LAYER = 16   # approximately 2/3 of 24 layers

# -----------------------------------------------------------------
# 1. Load tokeniser
# -----------------------------------------------------------------
print(f"Loading tokeniser from {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)

# -----------------------------------------------------------------
# 2. Load model weights in bf16
# -----------------------------------------------------------------
print(f"Loading model in bf16 on {DEVICE} ...")
target = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
)

# Freeze every parameter — this is the "T" copy that must never change.
target.eval()
target.requires_grad_(False)

# -----------------------------------------------------------------
# 3. Confirm config shapes
# -----------------------------------------------------------------
cfg = target.config
print(f"\nConfig check:")
print(f"  hidden_size        = {cfg.hidden_size}")          # expect 896
print(f"  num_hidden_layers  = {cfg.num_hidden_layers}")    # expect 24
print(f"  probe layer (2/3)  = {PROBE_LAYER}")

# -----------------------------------------------------------------
# 4. Forward pass — confirm hidden-state tensor shape
# -----------------------------------------------------------------
sample_text = "The transformer model processes tokens in parallel."
inputs = tok(sample_text, return_tensors="pt").to(DEVICE)
#token_view = tok.convert_ids_to_tokens(inputs["input_ids"][0])
#print(f"Actual tokens: {token_view}")

with torch.no_grad():
    out = target(**inputs, output_hidden_states=True)

# hidden_states is a tuple: one tensor per layer + the embedding layer.
# Index 0  = embedding output, indices 1..N = transformer block outputs.
h = out.hidden_states[PROBE_LAYER + 1]   # +1 because index 0 is the embedding
print(f"\nHidden state at layer {PROBE_LAYER}: {h.shape}")
# should print: (batch=1, seq_len, hidden_size=896)

# -----------------------------------------------------------------
# 5. Sanity generation — confirms the tokeniser+model round-trip works
# -----------------------------------------------------------------
print("\nSanity generation:")
gen_ids = target.generate(**inputs, max_new_tokens=20)
print(tok.decode(gen_ids[0], skip_special_tokens=True))

print("\nPhase 0 complete.")
