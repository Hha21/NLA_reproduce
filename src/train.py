"""
Training loops and metrics.

  fve()      — Fraction of Variance Explained
  train_ar() — supervised regression loop for the Reconstructor
  train_av() — SFT warm-start loop for the Verbalizer
"""

import math
import re
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import AR_PREFIX, AR_SUFFIX, AV_USER_PROMPT
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


def _normalize_to_sqrt_d(acts: torch.Tensor) -> torch.Tensor:
    """Scale each activation to L2 norm = sqrt(d_model), matching AV injection_scale."""
    scale = math.sqrt(acts.shape[-1])
    return acts * (scale / acts.norm(dim=-1, keepdim=True).clamp(min=1e-8))


# ---------------------------------------------------------------------------
# AR collate + training loop
# ---------------------------------------------------------------------------

def _ar_collate(batch, tok, max_length: int, device: str):
    texts, acts_np = zip(*batch)
    acts = torch.tensor(np.stack(acts_np), dtype=torch.float32, device=device)
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
    n_epochs: int    = 5,
    batch_size: int  = 16,
    lr: float        = 3e-4,
    base_lr: float   = None,
    max_length: int  = 256,
    val_frac: float  = 0.1,
    text_col: str    = "text_truncated",
    mse_scale: bool  = True,
):
    """
    Supervised AR training: text → activation regression.

    mse_scale: if True, normalise activation targets to L2 norm = sqrt(d_model)
               before computing MSE loss and FVE, matching the reference mse_scale
               = sqrt_d_model config. Keeps the regression target at a stable scale
               independent of raw activation magnitudes.
    """
    n_val   = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds = ActivationDataset(dataset.select(range(n_train)),               text_col=text_col)
    val_ds   = ActivationDataset(dataset.select(range(n_train, len(dataset))), text_col=text_col)

    collate      = partial(_ar_collate, tok=tok, max_length=max_length, device=device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate)

    if base_lr is not None:
        param_groups = [
            {"params": ar.base.parameters(), "lr": base_lr},
            {"params": ar.head.parameters(), "lr": lr},
        ]
    else:
        param_groups = list(filter(lambda p: p.requires_grad, ar.parameters()))

    opt       = torch.optim.AdamW(param_groups, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    for epoch in range(n_epochs):
        ar.train()
        total_loss = 0.0

        for input_ids, attn_mask, acts in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False):
            if mse_scale:
                acts = _normalize_to_sqrt_d(acts)
            opt.zero_grad()
            a_hat = ar(input_ids, attn_mask).float()
            loss  = F.mse_loss(a_hat, acts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        scheduler.step()

        ar.eval()
        all_acts, all_preds = [], []
        with torch.no_grad():
            for input_ids, attn_mask, acts in val_loader:
                if mse_scale:
                    acts = _normalize_to_sqrt_d(acts)
                all_acts.append(acts)
                all_preds.append(ar(input_ids, attn_mask).float())

        val_fve = fve(torch.cat(all_acts), torch.cat(all_preds))
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{n_epochs}  "
              f"loss: {total_loss / len(train_loader):.4f}  "
              f"val FVE: {val_fve:.4f}  "
              f"lr: {current_lr:.2e}")

    return ar


# ---------------------------------------------------------------------------
# AV collate + SFT warm-start loop
# ---------------------------------------------------------------------------

def _av_collate(batch, tok, max_length: int, device: str):
    """
    Build (input_ids, attention_mask, activation, labels) for AV SFT.

    Prompt (masked, labels=-100): chat-template-wrapped user message containing ㊗
    Response (has labels): "<explanation>\\n{summary}\\n</explanation>"

    The ㊗ token in the prompt is located by av.forward() and its embedding is
    replaced with the normalised activation vector at forward time.

    Reference: stage3_build.py _DEFAULT_ACTOR_TEMPLATE + wrap_explanation().
    """
    texts, acts_np = zip(*batch)
    acts = torch.tensor(np.stack(acts_np), dtype=torch.float32, device=device)

    # Tokenize the prompt via chat template (same for every item in the batch).
    prompt_str = tok.apply_chat_template(
        [{"role": "user", "content": AV_USER_PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tok(prompt_str, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    n_prompt = len(prompt_ids)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    all_ids, all_labels = [], []

    for text in texts:
        # Response wrapped in explanation tags, closed with EOS.
        response  = f"<explanation>\n{text}\n</explanation>"
        resp_ids  = tok(response, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        eos       = torch.tensor([tok.eos_token_id], dtype=torch.long)
        full      = torch.cat([prompt_ids, resp_ids, eos])
        if len(full) > max_length:
            full = full[:max_length]
        labels = full.clone()
        labels[: min(n_prompt, len(full))] = -100    # mask the prompt
        all_ids.append(full)
        all_labels.append(labels)

    max_len = max(len(x) for x in all_ids)
    ids_t  = torch.full((len(texts), max_len), pad_id, dtype=torch.long)
    labs_t = torch.full((len(texts), max_len), -100,   dtype=torch.long)
    mask_t = torch.zeros(len(texts), max_len,           dtype=torch.long)

    for i, (ids, labs) in enumerate(zip(all_ids, all_labels)):
        ids_t[i,  : len(ids)] = ids
        labs_t[i, : len(ids)] = labs
        mask_t[i, : len(ids)] = 1

    return ids_t.to(device), mask_t.to(device), acts, labs_t.to(device)


def eval_e2e_fve(av, ar, val_acts: torch.Tensor, tok, device: str,
                 n_eval: int = 100, gen_batch: int = 8, max_new_tokens: int = 120) -> float:
    """
    End-to-end FVE: AV generates description from h_l (greedy), AR reconstructs â.

    Runs on n_eval val samples. Cheap proxy for joint quality during AV warm-start —
    should rise as the AV learns to produce descriptions the AR can decode.
    """
    av.eval()
    ar.eval()

    prompt_str = tok.apply_chat_template(
        [{"role": "user", "content": AV_USER_PROMPT}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_ids_tmpl = tok(
        prompt_str, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0].to(device)

    n = min(n_eval, len(val_acts))
    all_acts_n, all_preds = [], []

    for start in range(0, n, gen_batch):
        acts_b = val_acts[start : start + gen_batch].to(device)
        B      = acts_b.shape[0]

        prompt_ids = prompt_ids_tmpl.unsqueeze(0).expand(B, -1)        # (B, T_p)
        attn_mask  = torch.ones(B, prompt_ids.shape[1], dtype=torch.long, device=device)

        with torch.no_grad():
            gen_ids = av.generate(
                prompt_ids, attn_mask, acts_b,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )   # (B, T_new) — generated tokens only

        descriptions = []
        for ids in gen_ids:
            text = tok.decode(ids, skip_special_tokens=True)
            m    = re.search(r"<explanation>(.*?)(?:</explanation>|$)", text, re.DOTALL)
            descriptions.append(m.group(1).strip() if m else text.strip())

        prompted = [f"{AR_PREFIX}{d}{AR_SUFFIX}" for d in descriptions]
        enc = tok(prompted, return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(device)

        with torch.no_grad():
            a_hat = ar(enc["input_ids"], enc["attention_mask"]).float()

        all_acts_n.append(_normalize_to_sqrt_d(acts_b).cpu())
        all_preds.append(a_hat.cpu())

    return fve(torch.cat(all_acts_n), torch.cat(all_preds))


def train_av(
    av,
    dataset,
    tok,
    device: str,
    n_epochs: int   = 5,
    batch_size: int = 8,
    lr: float       = 2e-5,
    max_length: int = 512,
    val_frac: float = 0.1,
    text_col: str   = "summary",
    ar              = None,   # optional frozen AR for end-to-end FVE after each epoch
    n_fve_eval: int = 100,    # val samples used for e2e FVE (kept small for speed)
):
    """
    SFT warm-start for the Verbalizer: (h_l → summary) pairs.

    If ar is provided (frozen AR checkpoint), computes end-to-end FVE after each
    epoch by generating descriptions with AV and reconstructing with AR.
    """
    n_val   = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    val_ds_raw = dataset.select(range(n_train, len(dataset)))
    train_ds   = ActivationDataset(dataset.select(range(n_train)),   text_col=text_col)
    val_ds     = ActivationDataset(val_ds_raw,                       text_col=text_col)

    collate      = partial(_av_collate, tok=tok, max_length=max_length, device=device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate)

    # Pre-fetch val activations for e2e FVE (only need the vectors, not text).
    if ar is not None:
        n_fve  = min(n_fve_eval, len(val_ds_raw))
        val_acts_fve = torch.tensor(
            np.stack(val_ds_raw.select(range(n_fve))["activation"]),
            dtype=torch.float32,
        )

    opt       = torch.optim.AdamW(av.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    for epoch in range(n_epochs):
        av.train()
        total_loss = 0.0

        for input_ids, attn_mask, acts, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False):
            opt.zero_grad()
            out  = av(input_ids, attn_mask, acts, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(av.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        scheduler.step()

        av.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, attn_mask, acts, labels in val_loader:
                out = av(input_ids, attn_mask, acts, labels=labels)
                val_loss += out.loss.item()

        current_lr = scheduler.get_last_lr()[0]
        log = (f"Epoch {epoch + 1}/{n_epochs}  "
               f"train_loss: {total_loss / len(train_loader):.4f}  "
               f"val_loss: {val_loss / len(val_loader):.4f}  "
               f"lr: {current_lr:.2e}")

        if ar is not None:
            e2e = eval_e2e_fve(av, ar, val_acts_fve, tok, device, n_eval=n_fve)
            log += f"  e2e FVE: {e2e:.4f}"
            av.train()   # restore training mode after eval_e2e_fve sets eval

        print(log)

    return av
