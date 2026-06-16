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

from src.config import AR_PREFIX, AR_SUFFIX
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
    # Wrap each text in the paper's AR prompt so the model sees the same framing
    # during both oracle training and GRPO (where z will be an AV description).
    prompted = [f"{AR_PREFIX}{t}{AR_SUFFIX}" for t in texts]
    enc = tok(
        prompted,
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
    base_lr: float  = None,
    max_length: int = 256,
    val_frac: float = 0.1,
):
    """
    Supervised AR training: text → activation regression.

    Splits the dataset 90/10, trains on MSE loss, and logs validation FVE
    after every epoch. Returns the trained AR.

    base_lr: if provided, the transformer body uses this LR and the head uses
             `lr`. Useful for fine-tuning (base_lr << lr). If None, a single
             lr is applied to all trainable parameters.
    """
    n_val   = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds = ActivationDataset(dataset.select(range(n_train)))
    val_ds   = ActivationDataset(dataset.select(range(n_train, len(dataset))))

    collate      = partial(_collate, tok=tok, max_length=max_length, device=device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate)

    if base_lr is not None:
        param_groups = [
            {"params": ar.base.parameters(), "lr": base_lr},
            {"params": ar.head.parameters(), "lr": lr},
        ]
    else:
        param_groups = filter(lambda p: p.requires_grad, ar.parameters())

    opt       = torch.optim.AdamW(param_groups, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

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

        scheduler.step()

        # Validation FVE
        ar.eval()
        all_acts, all_preds = [], []
        with torch.no_grad():
            for input_ids, attn_mask, acts in val_loader:
                all_acts.append(acts)
                all_preds.append(ar(input_ids, attn_mask).float())

        val_fve = fve(torch.cat(all_acts), torch.cat(all_preds))
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{n_epochs}  "
              f"loss: {total_loss / len(train_loader):.4f}  "
              f"val FVE: {val_fve:.4f}  "
              f"lr: {current_lr:.2e}")

    return ar
