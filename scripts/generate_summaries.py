"""
Stage 2: generate linguistic-feature explanations for each text_truncated.

Uses the reference codebase prompt (stage2_api_explain.py): asks the LLM to
identify 2-3 features a language model would use to predict the next token at
the truncation point (~80-100 words). Responses are validated for
<analysis>...</analysis> tags; malformed responses are retried.

Adds a 'summary' column to the dataset (name kept for compatibility with
--text-col summary in training scripts). Safe to interrupt and resume —
completed entries are checkpointed to activations/summaries_checkpoint.json.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python scripts/generate_summaries.py

    # Test run
    python scripts/generate_summaries.py --limit 50
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import json
import os
import shutil

from datasets import load_from_disk
from openai import AsyncOpenAI, APIStatusError
from tqdm.asyncio import tqdm as atqdm

DATA_DIR   = Path("activations/dataset")
CHECKPOINT = Path("activations/summaries_checkpoint.json")

# Prompt from the reference codebase (stage2_api_explain.py).
# 2-3 features, ~80-100 words — the version actually used in the published experiments.
_PROMPT = """\
A language model needs to predict what text comes next after a snippet which \
will be presented to you shortly. Identify the 2-3 most important features it \
would use for this prediction.

Focus on what the language model must be "thinking about" at the point where \
the provided text ends. You should not need to reference the fact that the text \
is truncated/incomplete/a prefix: the language model is causal, so only sees \
the prefix to what it predicts and this is implicit.

Order features by what is most important for predicting the next tokens. \
Each feature should consist of a concise ~10-20 word description. Feel free \
to include specific textual examples inline.

Feature types to consider (as inspiration, not a rigid checklist):
- Syntactic/structural constraints: "unclosed parenthesis requires matching close"
- Immediate semantic expectations: "list promised three items but only two given"
- Stylistic/register patterns: "formal academic tone maintained throughout"
- Narrative/argumentative momentum: "thesis stated, supporting evidence now expected"
- Domain/genre signals: "medical case history following SOAP format"
- Repetition/continuation patterns: "same phrase structure repeating with variations"

The final feature must describe the very end of the presented sequence: its \
role, what it's part of, and immediate constraints on what follows.

Format — IMPORTANT: keep to ~80-100 words total and ALWAYS close the tag:
<analysis>
[first feature — include specific examples when relevant]
[second feature]
[final feature: the last token, its role, immediate constraints]
</analysis>

Text to analyze:

<begin_text>{text}<end_text>"""

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir",    default=str(DATA_DIR))
parser.add_argument("--model",       default="deepseek-v4-flash",
                    help="DeepSeek model ID (check api-docs.deepseek.com for exact IDs)")
parser.add_argument("--concurrency", type=int, default=50,
                    help="Max concurrent requests (V4-Flash supports up to 2500)")
parser.add_argument("--limit",       type=int, default=None,
                    help="Only process this many samples (for testing)")
args = parser.parse_args()

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    sys.exit("Set DEEPSEEK_API_KEY environment variable before running.")

client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

checkpoint: dict[int, str] = {}


def save_checkpoint():
    CHECKPOINT.write_text(json.dumps(checkpoint))


import re as _re

_LIST_PREFIX_RE = _re.compile(r"^\s*(?:[-*•+]|\d+[.)])\s+", _re.MULTILINE)
_BOLD_RE        = _re.compile(r"\*\*(.+?)\*\*")
_ANALYSIS_RE    = _re.compile(r"<analysis>(.*?)</analysis>", _re.DOTALL)


def _extract_and_clean(raw: str) -> str | None:
    """Mirror the reference _extract_and_clean():

    1. Extract content inside <analysis>...</analysis> (tags are NOT stored).
    2. Strip list-prefix markers (bullets, numbers).
    3. Strip **bold** markers.
    4. Strip stray * _ chars from line edges.
    5. Drop empty lines; rejoin with double newline.

    Returns None if tags are missing or fewer than 2 features remain.
    """
    m = _ANALYSIS_RE.search(raw)
    if m is None:
        return None
    content = m.group(1)
    cleaned = []
    for line in content.split("\n"):
        line = _LIST_PREFIX_RE.sub("", line)
        line = _BOLD_RE.sub(r"\1 ", line)
        line = line.strip().strip("*_")
        if line:
            cleaned.append(line)
    if len(cleaned) < 2:
        return None
    return "\n\n".join(cleaned)


MIN_TEXT_CHARS = 400  # skip texts too short for meaningful analysis


async def summarise(sem: asyncio.Semaphore, idx: int, text: str) -> tuple[int, str | None]:
    if len(text) < MIN_TEXT_CHARS:
        return idx, None
    async with sem:
        for attempt in range(4):
            try:
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": _PROMPT.format(text=text)}],
                    temperature=0.3,
                )
                raw = resp.choices[0].message.content.strip()
                result = _extract_and_clean(raw)
                if result:
                    return idx, result
                # Malformed or too few features — retry without sleeping
            except APIStatusError as e:
                if e.status_code == 429 or e.status_code >= 500:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return idx, None
        return idx, None


async def main():
    global checkpoint

    print(f"Loading dataset from {args.data_dir}...")
    ds = load_from_disk(args.data_dir)
    print(f"  {len(ds)} samples total")

    if CHECKPOINT.exists():
        raw = json.loads(CHECKPOINT.read_text())
        checkpoint = {int(k): v for k, v in raw.items() if v and v.strip()}
        print(f"  Resuming — {len(checkpoint)} explanations already done")

    # Build todo from ALL indices that pass the length filter and aren't done yet.
    # --limit caps the number of items to process, not the index range scanned.
    passable = [
        i for i in range(len(ds))
        if len(ds[i]["text_truncated"]) >= MIN_TEXT_CHARS and i not in checkpoint
    ]
    n_short = sum(1 for i in range(len(ds)) if len(ds[i]["text_truncated"]) < MIN_TEXT_CHARS)
    todo = passable if args.limit is None else passable[:args.limit]
    print(f"  {n_short} samples too short (< {MIN_TEXT_CHARS} chars), skipped")
    print(f"  {len(todo)} to process\n")

    if todo:
        sem   = asyncio.Semaphore(args.concurrency)
        tasks = [summarise(sem, i, ds[i]["text_truncated"]) for i in todo]

        try:
            async for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks),
                                    desc="Generating explanations"):
                idx, summary = await coro
                if summary:
                    checkpoint[idx] = summary
                if len(checkpoint) % 500 == 0:
                    save_checkpoint()
        except (KeyboardInterrupt, Exception) as exc:
            print(f"\nInterrupted ({exc.__class__.__name__}). Saving checkpoint...")
            save_checkpoint()
            print(f"  {len(checkpoint)} explanations saved to {CHECKPOINT}")
            print("  Re-run to resume.")
            sys.exit(1)

        save_checkpoint()
        print(f"\nCheckpoint saved to {CHECKPOINT}")

    summaries = [checkpoint.get(i, "") for i in range(len(ds))]

    if "summary" in ds.column_names:
        ds = ds.remove_columns(["summary"])
    ds = ds.add_column("summary", summaries)

    tmp = Path(args.data_dir + "_tmp")
    ds.save_to_disk(str(tmp))
    shutil.rmtree(args.data_dir)
    tmp.rename(args.data_dir)
    print(f"Saved dataset with 'summary' column to {args.data_dir}")

    # Show a processed example (first entry in checkpoint, not necessarily ds[0])
    ex_idx = int(next(iter(checkpoint))) if checkpoint else 0
    sample = ds[ex_idx]
    print(f"\nExample (idx={ex_idx}):")
    print(f"  text_truncated: {sample['text_truncated'][:120]}...")
    print(f"  summary:        {sample['summary'][:300]}")


if __name__ == "__main__":
    asyncio.run(main())
