import torch

MODEL_ID    = "Qwen/Qwen2.5-0.5B"
PROBE_LAYER = 16                # ~2/3 of 24 layers
DTYPE       = torch.bfloat16
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# AR prompt from the paper (Appendix: Prompting the activation reconstructor).
# AR always receives: AR_PREFIX + z + AR_SUFFIX, and the last-token hidden state
# at layer PROBE_LAYER (the position of the final ">" of <summary>) is fed to the head.
AR_PREFIX = "Summary of the following text: <text>"
AR_SUFFIX = "</text> <summary>"
