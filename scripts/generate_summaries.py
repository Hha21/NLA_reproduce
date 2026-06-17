"""
Stage 2: generate DeepSeek summaries for each text_truncated in the dataset.

Adds a 'summary' column to the dataset on disk. Safe to interrupt and resume —
completed summaries are checkpointed to activations/summaries_checkpoint.json
and skipped on rerun.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python scripts/generate_summaries.py

    # Smaller test run
    python scripts/generate_summaries.py --limit 500
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import json
import os

import numpy as np
from datasets import load_from_disk
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

DATA_DIR   = Path("activations/dataset")
CHECKPOINT = Path("activations/summaries_checkpoint.json")

# Paper's summarization prompt (elicits a clean summary with no preamble).
_PROMPT = (
    "Summarise the following text in 1-3 sentences, "
    "focusing on the main topic and key information. "
    "Reply with only the summary, no preamble.\n\n"
    "{text}"
)

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir",    default=str(DATA_DIR))
parser.add_argument("--model",       default="deepseek/deepseek-chat",
                    help="OpenRouter model ID")
parser.add_argument("--concurrency", type=int, default=32,
                    help="Max concurrent API requests")
parser.add_argument("--limit",       type=int, default=None,
                    help="Only process this many samples (for testing)")
args = parser.parse_args()

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("Set OPENROUTER_API_KEY environment variable before running.")

client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)


async def summarise(sem: asyncio.Semaphore, idx: int, text: str) -> tuple[int, str]:
    async with sem:
        resp = await client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": _PROMPT.format(text=text)}],
            max_tokens=256,
            temperature=0.3,
        )
        return idx, resp.choices[0].message.content.strip()


async def main():
    print(f"Loading dataset from {args.data_dir}...")
    ds = load_from_disk(args.data_dir)
    n  = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"  {len(ds)} samples total, processing {n}")

    # Load checkpoint
    checkpoint: dict[int, str] = {}
    if CHECKPOINT.exists():
        checkpoint = {int(k): v for k, v in json.loads(CHECKPOINT.read_text()).items()}
        print(f"  Resuming — {len(checkpoint)} summaries already done")

    # Indices still needed
    todo = [i for i in range(n) if i not in checkpoint]
    print(f"  {len(todo)} remaining\n")

    if todo:
        sem   = asyncio.Semaphore(args.concurrency)
        tasks = [summarise(sem, i, ds[i]["text_truncated"]) for i in todo]

        async for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks),
                                desc="Generating summaries"):
            idx, summary = await coro
            checkpoint[idx] = summary

            # Checkpoint every 500 completions
            if len(checkpoint) % 500 == 0:
                CHECKPOINT.write_text(json.dumps(checkpoint))

        CHECKPOINT.write_text(json.dumps(checkpoint))
        print(f"\nCheckpoint saved to {CHECKPOINT}")

    # Build ordered summary list and add column
    summaries = [checkpoint[i] for i in range(n)]

    if n < len(ds):
        # Partial run — only update the processed slice
        existing = ds["summary"] if "summary" in ds.column_names else [""] * len(ds)
        for i, s in enumerate(summaries):
            existing[i] = s
        summaries = existing

    ds = ds.add_column("summary", summaries)
    ds.save_to_disk(args.data_dir)
    print(f"Saved dataset with 'summary' column to {args.data_dir}")

    # Quick sanity check
    sample = ds[0]
    print(f"\nExample:")
    print(f"  text_truncated: {sample['text_truncated'][:120]}...")
    print(f"  summary:        {sample['summary']}")


if __name__ == "__main__":
    asyncio.run(main())
