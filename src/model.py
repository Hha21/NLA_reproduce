from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import MODEL_ID, DTYPE


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


def load_target(device):
    """Load the frozen target model T in bf16. No parameter ever trains through this."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map=device,
    )
    model.eval()
    model.requires_grad_(False)
    return model
