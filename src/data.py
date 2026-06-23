"""
Activation extraction and dataset utilities.

Design decisions:
  - Corpus: FineWeb sample-10BT (HuggingFaceFW/fineweb). Higher diversity and
    longer documents than wikitext, matching the paper's data source. Streamed
    so the full 10BT corpus never hits disk.
  - Positions per document: up to POSITIONS_PER_DOC extraction points are
    sampled per document (matching the paper's --positions-per-doc 10). This
    is more efficient than 1-per-doc and gives varied context lengths within
    each document.
  - MIN_POSITION: 64 tokens minimum context. FineWeb docs are longer than
    wikitext snippets so a higher floor gives more meaningful truncated texts
    and better summaries downstream.
  - What we store as text: the text truncated at the extraction point.
    The activation only "saw" tokens up to that position.
  - Activation: stored as float32 numpy array (hidden_size,), unnormalised.
"""

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset, load_dataset
from tqdm import tqdm

from src.config import DEVICE, PROBE_LAYER

_CORPUS           = ("HuggingFaceFW/fineweb", "sample-10BT")
MIN_POSITION      = 150   # minimum tokens before the extraction point (~500 chars)
POSITIONS_PER_DOC = 10    # extraction points sampled per document


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
    n_samples: int          = 5_000,
    min_position: int       = MIN_POSITION,
    positions_per_doc: int  = POSITIONS_PER_DOC,
    max_position: int       = 0,       # cap extraction point (tokens); 0 = no cap
    seed: int               = 42,
    shard_size: int         = 0,       # call on_shard every N samples; 0 = disabled
    on_shard                = None,    # callback(shard_index: int, Dataset)
    skip_n: int             = 0,       # replay-skip first N samples (for resume)
) -> Dataset:
    """
    Stream FineWeb sample-10BT, extract up to positions_per_doc activations
    per document, return a HuggingFace Dataset with columns:

      text_truncated  str        — text the model saw up to extraction point
      activation      float32[]  — residual-stream vector, shape (hidden_size,)

    When shard_size > 0, on_shard(idx, ds) is called each time shard_size new
    samples are collected. The returned Dataset contains only the final partial
    shard (may be empty if n_samples is an exact multiple of shard_size).

    When skip_n > 0, the first skip_n positions are counted (rng replayed)
    without running the GPU — use this to resume after completed shards.
    """
    rng    = np.random.default_rng(seed)
    corpus = load_dataset(*_CORPUS, split="train", streaming=True)

    extract, handle = make_extractor(target)
    rows = {"text_truncated": [], "activation": []}

    n_skipped = 0
    n_new     = 0
    n_need    = n_samples - skip_n
    shard_idx = (skip_n // shard_size) if shard_size else 0

    with tqdm(total=n_samples, initial=skip_n, desc="Extracting activations") as pbar:
        for doc in corpus:
            if n_new >= n_need:
                break
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

            # Sample up to positions_per_doc distinct extraction points.
            n_valid = len(ids) - min_position
            n_pos   = min(positions_per_doc, n_valid)
            offsets = rng.choice(n_valid, size=n_pos, replace=False)
            positions = sorted(min_position + offsets)

            # Cap to max_position to avoid OOM on very long documents.
            if max_position > 0:
                positions = [p for p in positions if p <= max_position]

            for pos in positions:
                if n_new >= n_need:
                    break

                # Fast-forward: replay rng without touching the GPU.
                if n_skipped < skip_n:
                    n_skipped += 1
                    continue

                trunc_ids  = ids[:pos + 1].to(DEVICE)
                trunc_text = tok.decode(trunc_ids.cpu(), skip_special_tokens=True)
                rows["text_truncated"].append(trunc_text)
                rows["activation"].append(extract(trunc_ids))
                n_new += 1
                pbar.update(1)

                if shard_size and len(rows["text_truncated"]) >= shard_size:
                    if on_shard is not None:
                        on_shard(shard_idx, Dataset.from_dict(rows))
                    rows = {"text_truncated": [], "activation": []}
                    shard_idx += 1

    handle.remove()
    return Dataset.from_dict(rows)


class ActivationDataset(TorchDataset):
    """
    Wraps a HuggingFace Dataset into a plain PyTorch Dataset.

    Pre-loading texts and activations as Python lists / numpy arrays means
    the DataLoader receives simple (str, ndarray) tuples — avoiding the
    __getitems__ fast-path in newer datasets+PyTorch that mangles shapes.
    """

    def __init__(self, hf_dataset, text_col: str = "text_truncated"):
        self.texts       = hf_dataset[text_col]
        self.activations = np.stack(hf_dataset["activation"]).astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.activations[idx]   # (str, ndarray 896)
