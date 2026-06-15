"""
Activation extraction and dataset utilities.

Design decisions:
  - Corpus: wikitext-103-raw-v1. Small enough to iterate fast, clean English
    prose, enough topical variety for the bottleneck to be meaningful.
  - Token position: we sample a random position per document (at least
    MIN_POSITION tokens in). Sampling rather than always taking the last token
    gives more diverse context lengths and avoids the model always seeing a
    sentence-final token.
  - What we store as text: the text *truncated at the extraction point*.
    The activation only "saw" tokens up to that position, so giving AR the
    full document would hand it information that didn't exist when the
    activation was computed.
  - Activation: stored as float32 numpy array (hidden_size,), unnormalised.
    Normalisation is applied at training time so we can tune it independently.
"""

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset, load_dataset
from tqdm import tqdm

from src.config import DEVICE, PROBE_LAYER

_CORPUS       = ("Salesforce/wikitext", "wikitext-103-raw-v1")
MIN_POSITION  = 32   # minimum tokens of context before the extraction point


def make_extractor(target):
    """
    Register a forward hook on PROBE_LAYER. Return (extract_fn, handle).

    extract_fn(trunc_ids: 1-D Tensor on DEVICE) -> float32 numpy (hidden_size,)
      Runs one forward pass and returns the residual-stream vector at the last
      token of trunc_ids (which is the sampled extraction position).

    Call handle.remove() when done to clean up the hook.
    """
    acts = {}

    def _hook(module, inp, out):
        # Newer transformers returns a plain tensor; older versions return a tuple.
        h = out[0] if isinstance(out, tuple) else out
        acts["resid"] = h.detach()        # (batch, seq, hidden_size)

    handle = target.model.layers[PROBE_LAYER].register_forward_hook(_hook)

    def extract(trunc_ids: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            target(input_ids=trunc_ids.unsqueeze(0))
        return acts["resid"][0, -1].float().cpu().numpy()   # (hidden_size,)

    return extract, handle


def build_dataset(
    target,
    tok,
    n_samples: int = 5_000,
    min_position: int = MIN_POSITION,
    seed: int = 42,
) -> Dataset:
    """
    Stream wikitext-103, extract one activation per document, return a
    HuggingFace Dataset with columns:

      text_truncated  str        — text the model saw up to extraction point
      activation      float32[]  — residual-stream vector, shape (hidden_size,)
    """
    rng    = np.random.default_rng(seed)
    corpus = load_dataset(*_CORPUS, split="train", streaming=True)

    extract, handle = make_extractor(target)
    rows = {"text_truncated": [], "activation": []}

    with tqdm(total=n_samples, desc="Extracting activations") as pbar:
        for doc in corpus:
            text = doc["text"].strip()
            if not text:
                continue

            ids = tok(
                text,
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"][0]

            if len(ids) <= min_position:
                continue

            # Sample extraction point; last token of the truncated sequence.
            pos       = int(rng.integers(min_position, len(ids)))
            trunc_ids = ids[: pos + 1].to(DEVICE)
            trunc_text = tok.decode(trunc_ids.cpu(), skip_special_tokens=True)

            rows["text_truncated"].append(trunc_text)
            rows["activation"].append(extract(trunc_ids))

            pbar.update(1)
            if len(rows["text_truncated"]) >= n_samples:
                break

    handle.remove()
    return Dataset.from_dict(rows)


class ActivationDataset(TorchDataset):
    """
    Wraps a HuggingFace Dataset into a plain PyTorch Dataset.

    Pre-loading texts and activations as Python lists / numpy arrays means
    the DataLoader receives simple (str, ndarray) tuples — avoiding the
    __getitems__ fast-path in newer datasets+PyTorch that mangles shapes.
    """

    def __init__(self, hf_dataset):
        self.texts       = hf_dataset["text_truncated"]
        self.activations = np.stack(hf_dataset["activation"]).astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.activations[idx]   # (str, ndarray 896)
