"""
Training loops and metrics.

  fve()      — Fraction of Variance Explained
  train_ar() — supervised regression loop for the Reconstructor
  train_av() — SFT warm-start loop for the Verbalizer
"""

import math
import re
from functools import partial
from pathlib import Path

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
    n_epochs: int    = 5,
    batch_size: int  = 8,
    lr: float        = 2e-5,
    max_length: int  = 512,
    val_frac: float  = 0.1,
    text_col: str    = "summary",
    ar               = None,    # optional frozen AR for end-to-end FVE after each epoch
    n_fve_eval: int  = 100,     # val samples used for e2e FVE (kept small for speed)
    checkpoint_path: str = None, # if set, saves best val_loss checkpoint here
):
    """
    SFT warm-start for the Verbalizer: (h_l → summary) pairs.

    If ar is provided (frozen AR checkpoint), computes end-to-end FVE after each
    epoch by generating descriptions with AV and reconstructing with AR.
    If checkpoint_path is set, saves the best val_loss checkpoint during training.
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

    best_val_loss = float("inf")

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

        val_loss_avg = val_loss / len(val_loader)
        current_lr   = scheduler.get_last_lr()[0]
        log = (f"Epoch {epoch + 1}/{n_epochs}  "
               f"train_loss: {total_loss / len(train_loader):.4f}  "
               f"val_loss: {val_loss_avg:.4f}  "
               f"lr: {current_lr:.2e}")

        if ar is not None:
            e2e = eval_e2e_fve(av, ar, val_acts_fve, tok, device, n_eval=n_fve)
            log += f"  e2e FVE: {e2e:.4f}"
            av.train()   # restore training mode after eval_e2e_fve sets eval

        if checkpoint_path is not None and val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(av.state_dict(), checkpoint_path)
            log += "  [best]"

        print(log)

    return av


# ---------------------------------------------------------------------------
# GRPO joint training: AV (policy gradient) + AR (supervised MSE)
# ---------------------------------------------------------------------------

FAILED_REWARD = -2.0  # penalty for sequences that produce no valid <explanation> tag


def _grpo_rollout(av, ar, tok, prompt_ids, acts, n_samples, max_new_tokens, ar_max_length, device):
    """
    One rollout: for each of N activations generate K descriptions with AV (temp=1),
    then score each with AR. Returns three parallel lists of length N, each containing
    a list of length K.
    """
    N = acts.shape[0]
    K = n_samples
    T_p = len(prompt_ids)
    all_gen, all_desc, all_rew = [], [], []

    for i in range(N):
        act_i = acts[i]
        prompt_K = prompt_ids.unsqueeze(0).expand(K, -1)       # (K, T_p)
        mask_K   = torch.ones(K, T_p, dtype=torch.long, device=device)
        act_K    = act_i.unsqueeze(0).expand(K, -1)            # (K, d)

        with torch.no_grad():
            gen = av.generate(
                prompt_K, mask_K, act_K,
                max_new_tokens=max_new_tokens,
                do_sample=True, temperature=1.0,
                pad_token_id=tok.eos_token_id,
            )  # (K, T_new) — generated tokens only

        # Trim each sequence to include first EOS and strip pad EOS tokens after it
        gen_list = []
        for k in range(K):
            ids = gen[k]
            eos_pos = (ids == tok.eos_token_id).nonzero(as_tuple=True)[0]
            end = eos_pos[0].item() + 1 if len(eos_pos) > 0 else len(ids)
            gen_list.append(ids[:end])

        # Extract descriptions from generated text
        descriptions = []
        for ids in gen_list:
            text = tok.decode(ids, skip_special_tokens=True)
            m = re.search(r"<explanation>(.*?)(?:</explanation>|$)", text, re.DOTALL)
            descriptions.append(m.group(1).strip() if m else None)

        # Score valid descriptions with AR: reward = -MSE(norm(â), norm(h_l))
        valid_ks = [k for k, d in enumerate(descriptions) if d is not None]
        rewards  = [FAILED_REWARD] * K

        if valid_ks:
            texts   = [f"{AR_PREFIX}{descriptions[k]}{AR_SUFFIX}" for k in valid_ks]
            enc     = tok(texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=ar_max_length).to(device)
            act_rep = act_i.unsqueeze(0).expand(len(valid_ks), -1)
            with torch.no_grad():
                a_hat = ar(enc["input_ids"], enc["attention_mask"]).float()
            mse = ((a_hat - _normalize_to_sqrt_d(act_rep)) ** 2).mean(dim=-1)
            for j, k in enumerate(valid_ks):
                rewards[k] = -mse[j].item()

        all_gen.append(gen_list)
        all_desc.append(descriptions)
        all_rew.append(rewards)

    return all_gen, all_desc, all_rew


def _seq_log_probs(av, prompt_ids, gen_list, acts, tok, device, no_grad=False):
    """
    Compute mean per-token log prob for each generated sequence under av.

    prompt_ids : (T_p,)
    gen_list   : list of K tensors (variable-length generated token IDs)
    acts       : (K, d) activations
    Returns    : (K,) tensor — gradient-connected to av when no_grad=False
    """
    from contextlib import nullcontext

    K    = len(gen_list)
    T_p  = len(prompt_ids)
    lens = [len(g) for g in gen_list]
    maxg = max(lens)
    pad  = tok.eos_token_id

    full_ids   = torch.full((K, T_p + maxg), pad, dtype=torch.long, device=device)
    attn       = torch.zeros(K, T_p + maxg,       dtype=torch.long, device=device)
    gen_padded = torch.full((K, maxg),       pad, dtype=torch.long, device=device)
    resp_mask  = torch.zeros(K, maxg,              device=device)

    for k in range(K):
        full_ids[k, :T_p]               = prompt_ids
        full_ids[k, T_p:T_p + lens[k]]  = gen_list[k]
        attn[k, :T_p + lens[k]]         = 1
        gen_padded[k, :lens[k]]          = gen_list[k]
        resp_mask[k, :lens[k]]           = 1.0

    ctx = torch.no_grad() if no_grad else nullcontext()
    with ctx:
        logits = av(full_ids, attn, acts, labels=None).logits  # (K, T_p+maxg, V)

    # logits[:,t] predicts position t+1; generated tokens start at position T_p
    resp_logits = logits[:, T_p - 1:T_p + maxg - 1, :]        # (K, maxg, V)
    log_probs   = torch.log_softmax(resp_logits, dim=-1)
    token_lp    = log_probs.gather(2, gen_padded.unsqueeze(2)).squeeze(2)  # (K, maxg)
    token_lp    = token_lp * resp_mask
    seq_lp      = token_lp.sum(1) / resp_mask.sum(1).clamp(min=1.0)       # (K,)
    return seq_lp


def train_grpo(
    av,
    av_ref,
    ar,
    dataset,
    tok,
    device: str,
    n_steps: int        = 500,
    n_prompts: int      = 16,        # N: activations per rollout (reference: 128 on 8 GPUs)
    n_samples: int      = 8,         # K: descriptions per activation (GRPO group size)
    av_lr: float        = 1.41e-5,   # reference rl.sh line 126
    ar_lr: float        = 1.41e-5,   # reference rl.sh line 102
    kl_coef: float      = 0.01,      # reference rl.sh line 54
    max_new_tokens: int = 150,       # reference rl.sh line 108
    ar_max_length: int  = 256,
    checkpoint_path: str = None,     # base path; saves av+ar every save_interval steps
    save_interval: int   = 100,
    log_interval: int    = 10,
):
    """
    Joint AV GRPO + AR supervised MSE training.

    Each step:
      1. Rollout: N activations → AV generates K descriptions each (on-policy, temp=1).
      2. Reward: AR scores each → r = -MSE(norm(â), norm(h_l)); failed extractions → -2.0.
      3. GRPO advantages: A_k = (r_k - mean) / std within each group of K.
      4. AV loss: -mean(A_k * mean_token_logprob) + kl_coef * KL(π ‖ π_ref).
      5. AR loss: MSE(norm(AR(desc)), norm(h_l)) on all valid descriptions.
      6. Joint backward + constant-LR optimizer step (matches reference rl.sh).

    av_ref: frozen SFT checkpoint of AV used for the KL penalty (pass None to disable KL).
    AR is updated alongside AV — live reward model, not frozen (per reference design).
    """
    av_opt = torch.optim.AdamW(av.parameters(), lr=av_lr, weight_decay=0.0)
    ar_opt = torch.optim.AdamW(ar.parameters(), lr=ar_lr, weight_decay=0.0)

    if av_ref is not None:
        av_ref.eval()
        av_ref.requires_grad_(False)

    # Build fixed AV prompt token IDs once
    prompt_str = tok.apply_chat_template(
        [{"role": "user", "content": AV_USER_PROMPT}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tok(
        prompt_str, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0].to(device)

    # Dataset index shuffling
    idx = torch.randperm(len(dataset)).tolist()
    pos = 0

    for step in range(1, n_steps + 1):

        # Sample N activations
        if pos + n_prompts > len(idx):
            idx = torch.randperm(len(dataset)).tolist()
            pos = 0
        acts = torch.tensor(
            np.stack(dataset.select(idx[pos:pos + n_prompts])["activation"]),
            dtype=torch.float32, device=device,
        )
        pos += n_prompts

        # ── Rollout ──────────────────────────────────────────────────────────
        av.eval()
        ar.eval()
        all_gen, all_desc, all_rew = _grpo_rollout(
            av, ar, tok, prompt_ids, acts, n_samples, max_new_tokens, ar_max_length, device
        )

        # ── GRPO advantages ──────────────────────────────────────────────────
        all_adv = []
        for group_r in all_rew:
            r   = torch.tensor(group_r, dtype=torch.float32)
            adv = (r - r.mean()) / r.std().clamp(min=1e-8)
            all_adv.append(adv.to(device))

        # ── Training ──────────────────────────────────────────────────────────
        av.train()
        ar.train()
        av_opt.zero_grad()
        ar_opt.zero_grad()

        pg_total = kl_total = ar_total = n_valid = 0

        for i in range(n_prompts):
            act_i  = acts[i]
            act_K  = act_i.unsqueeze(0).expand(n_samples, -1)
            adv_i  = all_adv[i]                                  # (K,) constants

            # Policy gradient log probs (gradient flows through AV)
            seq_lp  = _seq_log_probs(av, prompt_ids, all_gen[i], act_K, tok, device)
            pg_loss = -(adv_i * seq_lp).mean()

            # KL against frozen reference AV
            kl_loss = torch.tensor(0.0, device=device)
            if av_ref is not None and kl_coef > 0:
                ref_lp  = _seq_log_probs(av_ref, prompt_ids, all_gen[i], act_K, tok, device, no_grad=True)
                kl_loss = (seq_lp - ref_lp).mean()

            # AR supervised MSE on generated descriptions
            valid_ks = [k for k, d in enumerate(all_desc[i]) if d is not None]
            ar_loss  = torch.tensor(0.0, device=device)
            if valid_ks:
                texts   = [f"{AR_PREFIX}{all_desc[i][k]}{AR_SUFFIX}" for k in valid_ks]
                enc     = tok(texts, return_tensors="pt", padding=True,
                              truncation=True, max_length=ar_max_length).to(device)
                act_rep = act_i.unsqueeze(0).expand(len(valid_ks), -1)
                a_hat   = ar(enc["input_ids"], enc["attention_mask"]).float()
                ar_loss = F.mse_loss(a_hat, _normalize_to_sqrt_d(act_rep))
                n_valid += len(valid_ks)

            loss_i = (pg_loss + kl_coef * kl_loss + ar_loss) / n_prompts
            loss_i.backward()

            pg_total += pg_loss.item()
            kl_total += kl_loss.item()
            ar_total += ar_loss.item() if valid_ks else 0.0

        torch.nn.utils.clip_grad_norm_(av.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
        av_opt.step()
        ar_opt.step()

        if step % log_interval == 0:
            mean_r = np.mean([r for g in all_rew for r in g])
            print(f"Step {step:4d}/{n_steps}  "
                  f"pg: {pg_total/n_prompts:+.4f}  "
                  f"kl: {kl_total/n_prompts:.4f}  "
                  f"ar: {ar_total/n_prompts:.4f}  "
                  f"reward: {mean_r:.4f}  "
                  f"valid: {n_valid}/{n_prompts * n_samples}")

        if checkpoint_path is not None and step % save_interval == 0:
            parent = Path(checkpoint_path).parent
            stem   = Path(checkpoint_path).stem
            torch.save(av.state_dict(), parent / f"{stem}_av_step{step}.pt")
            torch.save(ar.state_dict(), parent / f"{stem}_ar_step{step}.pt")
            print(f"  → checkpoints saved at step {step}")

    return av, ar
