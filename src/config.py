import torch

MODEL_ID    = "Qwen/Qwen2.5-0.5B"
PROBE_LAYER = 16                # ~2/3 of 24 layers
DTYPE       = torch.bfloat16
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
