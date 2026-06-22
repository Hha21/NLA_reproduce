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

# AV soft-token injection (reference: design.md + stage3_build.py).
# ㊗ is a rare Unicode character used as a single-token placeholder; its embedding
# is overwritten at forward time with the normalised activation vector.
INJECT_TOKEN = "㊗"

# Full user prompt from the paper / reference stage3_build.py (_DEFAULT_ACTOR_TEMPLATE).
# Single user message — no system message. Chat template applied at training time.
# ㊗ inside <concept> tags is replaced with the normalised h_l at embedding time.
AV_USER_PROMPT = (
    "You are a meticulous AI researcher conducting an important investigation into "
    "activation vectors from a language model. Your overall task is to describe the "
    "semantic content of that activation vector.\n\n"
    "We will pass the vector enclosed in <concept> tags into your context. You must "
    "then produce an explanation for the vector, enclosed within <explanation> tags. "
    "The explanation consists of 2-3 text snippets describing that vector.\n\n"
    f"Here is the vector:\n\n<concept>{INJECT_TOKEN}</concept>\n\n"
    "Please provide an explanation."
)
