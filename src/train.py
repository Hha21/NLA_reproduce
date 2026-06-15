"""
Training loops and metrics.

  fve()      — Fraction of Variance Explained
  train_ar() — supervised regression loop for the Reconstructor
"""

from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import ActivationDataset


def fve(a: torch.Tensor, a_hat: torch.Tensor) -> float:
    """
    Fraction of Variance Explained.
      FVE = 0  → no better than predicting the corpus mean activation
      FVE = 1  → perfect reconstruction

    Both tensors must be (N, d_model) float32.
    """
    residual_var = ((a - a_hat) ** 2).sum(-1).mean()
    total_var    = ((a - a.mean(0)) ** 2).sum(-1).mean()
    return (1.0 - residual_var / total_var).item()


def _collate(batch, tok, max_length: int, device: str):
    # batch is a list of (text: str, activation: ndarray 896) tuples
    texts, acts_np = zip(*batch)
    acts = torch.tensor(np.stack(acts_np), dtype=torch.float32, device=device)
    enc  = tok(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    return enc["input_ids"], enc["attention_mask"], acts


def train_ar(
    ar,
    dataset,
    tok,
    device: str,
    n_epochs: int   = 5,
    batch_size: int = 16,
    lr: float       = 3e-4,
    max_length: int = 256,
    val_frac: float = 0.1,
):
    """
    Supervised AR training: text → activation regression.

    Splits the dataset 90/10, trains on MSE loss, and logs validation FVE
    after every epoch. Returns the trained AR.
    """
    n_val   = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds = ActivationDataset(dataset.select(range(n_train)))
    val_ds   = ActivationDataset(dataset.select(range(n_train, len(dataset))))

    collate      = partial(_collate, tok=tok, max_length=max_length, device=device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, ar.parameters()),
        lr=lr,
    )

    for epoch in range(n_epochs):
        ar.train()
        total_loss = 0.0

        for input_ids, attn_mask, acts in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False):
            opt.zero_grad()
            a_hat = ar(input_ids, attn_mask).float()   # bf16 → float32
            loss  = F.mse_loss(a_hat, acts)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # Validation FVE
        ar.eval()
        all_acts, all_preds = [], []
        with torch.no_grad():
            for input_ids, attn_mask, acts in val_loader:
                all_acts.append(acts)
                all_preds.append(ar(input_ids, attn_mask).float())

        val_fve = fve(torch.cat(all_acts), torch.cat(all_preds))
        print(f"Epoch {epoch + 1}/{n_epochs}  "
              f"loss: {total_loss / len(train_loader):.4f}  "
              f"val FVE: {val_fve:.4f}")

    return ar
