"""
Inference logic for the NLA web demo.

Loads the frozen target model T, the trained Verbalizer AV, and the trained
Reconstructor AR once at startup, then exposes:

  - tokenize(text)              -> list[str]
  - analyze(text, position)     -> dict with explanation + reconstruction cosine

The activation extraction, AV prompt assembly, and AR call all mirror what
`src.train.eval_e2e_fve` does, so the per-token result here is comparable to
the FVE numbers reported in REPRODUCE_LOG.md.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

# Allow `python -m uvicorn server.main:app` from the repo root.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_from_disk

from src.config import AR_PREFIX, AR_SUFFIX, AV_USER_PROMPT, DEVICE, PROBE_LAYER
from src.ar import load_ar
from src.av import load_av
from src.model import load_target


AV_CHECKPOINT     = ROOT / "models" / "av.pt"
AR_CHECKPOINT     = ROOT / "models" / "ar.pt"
ACTIVATIONS_DIR   = ROOT / "activations" / "dataset"
MAX_NEW_TOKENS    = 120                  # AV explanation generation
AR_MAX_LENGTH     = 256
FVE_BASELINE_N    = 5_000                # samples used to estimate the corpus mean


def _load_activations_dataset(path: Path):
    """Load either a flat HuggingFace dataset or a directory of shards."""
    if (path / "dataset_info.json").exists():
        return load_from_disk(str(path))
    shards = path / "shards"
    if shards.exists():
        paths = sorted(s for s in shards.glob("shard_*") if (s / "dataset_info.json").exists())
        if paths:
            return concatenate_datasets([load_from_disk(str(s)) for s in paths])
    raise FileNotFoundError(f"no activations dataset at {path}")


class NLAInference:
    """Holds T, AV, AR, the tokenizer, and the hook on layer PROBE_LAYER."""

    def __init__(self, device: str = DEVICE):
        self.device = device

        # AV (also gives us the tokenizer with ㊗ guaranteed single-token)
        self.av, self.tok = load_av(device)
        self.av.load_state_dict(torch.load(AV_CHECKPOINT, map_location=device))
        self.av.eval()
        for p in self.av.parameters():
            p.requires_grad_(False)

        # AR (truncated to PROBE_LAYER; we freeze everything for inference)
        self.ar = load_ar(device, freeze_base=False)
        self.ar.load_state_dict(torch.load(AR_CHECKPOINT, map_location=device))
        self.ar.eval()
        for p in self.ar.parameters():
            p.requires_grad_(False)

        # Frozen target T + hook on the probe layer
        self.target  = load_target(device)
        self.d_model = self.target.config.hidden_size
        self._act_cache: dict = {}

        def _hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self._act_cache["resid"] = h.detach()

        self._hook_handle = self.target.model.layers[PROBE_LAYER].register_forward_hook(_hook)

        # Pre-tokenise the AV prompt (independent of the activation; same chunk every time)
        prompt_str = self.tok.apply_chat_template(
            [{"role": "user", "content": AV_USER_PROMPT}],
            tokenize=False, add_generation_prompt=True,
        )
        self._av_prompt_ids = self.tok(
            prompt_str, add_special_tokens=False, return_tensors="pt",
        )["input_ids"][0].to(device)

        # End-of-turn token for chat-template generation (Qwen uses <|im_end|>).
        # Fall back gracefully if not present.
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self._chat_eos_ids = [self.tok.eos_token_id]
        if isinstance(im_end, int) and im_end != self.tok.unk_token_id:
            self._chat_eos_ids.append(im_end)

        # Corpus-mean activation in sqrt(d)-normalised space — baseline for per-sample FVE.
        # If the activations dataset isn't present, FVE will be reported as null.
        self.corpus_mean = self._compute_corpus_mean()

    def _compute_corpus_mean(self) -> torch.Tensor | None:
        """Mean of sqrt(d)-normalised activations over a sample of the training corpus.

        Used as the denominator baseline for per-sample FVE = 1 − ‖a − â‖² / ‖a − ā‖².
        Returns None if the activations directory cannot be loaded (the demo still
        works; analyze() just reports fve = None).
        """
        try:
            ds = _load_activations_dataset(ACTIVATIONS_DIR)
        except FileNotFoundError:
            return None

        n   = min(FVE_BASELINE_N, len(ds))
        arr = np.stack(ds.select(range(n))["activation"]).astype(np.float32)
        acts = torch.from_numpy(arr)                                # (n, d)
        scale = math.sqrt(self.d_model)
        norms = acts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (acts * (scale / norms)).mean(dim=0).to(self.device)  # (d,)

    # ------------------------------------------------------------------ helpers
    def _tokenize_ids(self, text: str) -> torch.Tensor:
        return self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)

    def _decode_tokens(self, ids: torch.Tensor) -> list[str]:
        return [self.tok.decode([int(i)]) for i in ids]

    def _extract_activation(self, ids_up_to_pos: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.target(input_ids=ids_up_to_pos.unsqueeze(0))
        return self._act_cache["resid"][0, -1].float()   # (d_model,)

    def _generate_explanation(self, activation: torch.Tensor) -> str:
        act        = activation.unsqueeze(0).to(self.device)
        prompt_ids = self._av_prompt_ids.unsqueeze(0)
        attn_mask  = torch.ones_like(prompt_ids, dtype=torch.long)
        with torch.no_grad():
            gen_ids = self.av.generate(
                prompt_ids, attn_mask, act,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        text = self.tok.decode(gen_ids[0], skip_special_tokens=True)
        m    = re.search(r"<explanation>(.*?)(?:</explanation>|$)", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    def _reconstruct(self, description: str) -> torch.Tensor:
        prompt = f"{AR_PREFIX}{description}{AR_SUFFIX}"
        enc = self.tok(prompt, return_tensors="pt",
                       truncation=True, max_length=AR_MAX_LENGTH).to(self.device)
        with torch.no_grad():
            return self.ar(enc["input_ids"], enc["attention_mask"]).float()[0]   # (d_model,)

    def _analyze_position(self, ids: torch.Tensor, position: int) -> dict:
        """Shared T → AV → AR pipeline for a single token position.

        `ids` is a 1-D LongTensor on the model device. `position` must be
        already resolved (no negative indices).
        """
        act    = self._extract_activation(ids[: position + 1])
        desc   = self._generate_explanation(act)
        a_hat  = self._reconstruct(desc)

        scale  = math.sqrt(self.d_model)
        a_norm = act * (scale / act.norm().clamp(min=1e-8))
        cosine = F.cosine_similarity(a_norm, a_hat, dim=-1).item()

        # Per-sample FVE against the corpus-mean baseline (in normalised space).
        # Definition mirrors src.train.fve so the number is comparable to corpus FVE.
        fve = None
        if self.corpus_mean is not None:
            num = ((a_norm - a_hat).float() ** 2).sum().item()
            den = ((a_norm - self.corpus_mean).float() ** 2).sum().item()
            fve = 1.0 - num / max(den, 1e-8)

        return {
            "position":       position,
            "explanation":    desc,
            "reconstruction": cosine,
            "fve":            fve,
        }

    # ------------------------------------------------------------------ public
    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        return self._decode_tokens(self._tokenize_ids(text))

    def analyze_text(self, text: str, position: int) -> dict:
        """Tokenise `text`, run analyse at `position` (negative = from end)."""
        if not text:
            raise ValueError("text must be non-empty")
        ids = self._tokenize_ids(text)
        n   = ids.shape[0]
        if n == 0:
            raise ValueError("tokenisation produced no tokens")
        if position < 0:
            position += n
        if not (0 <= position < n):
            raise IndexError(f"position {position} out of range for {n} tokens")

        out = self._analyze_position(ids, position)
        out["tokens"] = self._decode_tokens(ids)
        return out

    def analyze_ids(self, token_ids: list[int], position: int) -> dict:
        """Run analyse at `position` over a pre-tokenised sequence.

        Preferred over analyze_text() for the chat flow: avoids any tokenisation
        round-trip ambiguity for special tokens like <|im_start|>.
        """
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        n = len(token_ids)
        if position < 0:
            position += n
        if not (0 <= position < n):
            raise IndexError(f"position {position} out of range for {n} tokens")

        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        return self._analyze_position(ids, position)

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 200,
        temperature:    float = 0.7,
        top_p:          float = 0.9,
    ) -> dict:
        """Apply the chat template, run T.generate, return full token sequence.

        `messages` is a list of {"role": "user"/"assistant"/"system", "content": str}.
        Returns the same messages with the new assistant turn appended, plus the
        raw token IDs / strings for the *entire* templated context so the frontend
        can render and click any token (including special chat-template ones).

        Marks where the assistant turn starts so the UI can colour it differently.
        """
        if not messages:
            raise ValueError("messages must be non-empty")

        prompt_str = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = self.tok(
            prompt_str, return_tensors="pt", add_special_tokens=False,
        )["input_ids"].to(self.device)

        prompt_len = input_ids.shape[1]
        do_sample  = temperature > 0
        attn_mask  = torch.ones_like(input_ids, dtype=torch.long)

        gen_kwargs = dict(
            max_new_tokens = max_new_tokens,
            do_sample      = do_sample,
            pad_token_id   = self.tok.eos_token_id,
            eos_token_id   = self._chat_eos_ids,
            attention_mask = attn_mask,
        )
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
            gen_kwargs["top_p"]       = top_p

        with torch.no_grad():
            out = self.target.generate(input_ids, **gen_kwargs)

        full_ids = out[0].tolist()
        gen_ids  = full_ids[prompt_len:]
        # Strip a trailing chat-EOS so the assistant text we return is clean,
        # but leave full_ids/tokens as-is so the user can inspect those tokens.
        assistant_text = self.tok.decode(gen_ids, skip_special_tokens=True).strip()

        new_messages = list(messages) + [{"role": "assistant", "content": assistant_text}]

        return {
            "messages":              new_messages,
            "token_ids":             full_ids,
            "tokens":                [self.tok.decode([i]) for i in full_ids],
            "is_special":            [bool(i in self.tok.all_special_ids) for i in full_ids],
            "assistant_token_start": prompt_len,
            "assistant_text":        assistant_text,
        }
