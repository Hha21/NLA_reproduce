"""
Stage 2: generate summaries for each text_truncated in the dataset.

Adds a 'summary' column to the dataset on disk. Safe to interrupt and resume —
completed summaries are checkpointed to activations/summaries_checkpoint.json
and skipped on rerun.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python scripts/generate_summaries.py

    # Test run
    python scripts/generate_summaries.py --limit 50

    # Use V4-Pro if higher quality needed (~$12 vs ~$4 for 100K)
    python scripts/generate_summaries.py --model deepseek-v4-pro
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

_PROMPT = (
    "Summarise the following text in 1-3 sentences, "
    "focusing on the main topic and key information. "
    "Reply with only the summary, no preamble.\n\n"
    "{text}"
)

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


async def summarise(sem: asyncio.Semaphore, idx: int, text: str) -> tuple[int, str | None]:
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": _PROMPT.format(text=text)}],
                    max_tokens=256,
                    temperature=0.3,
                )
                return idx, resp.choices[0].message.content.strip()
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
    n  = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"  {len(ds)} samples total, processing {n}")

    if CHECKPOINT.exists():
        raw = json.loads(CHECKPOINT.read_text())
        checkpoint = {int(k): v for k, v in raw.items() if v and v.strip()}
        print(f"  Resuming — {len(checkpoint)} summaries already done")

    todo = [i for i in range(n) if i not in checkpoint]
    print(f"  {len(todo)} remaining\n")

    if todo:
        sem   = asyncio.Semaphore(args.concurrency)
        tasks = [summarise(sem, i, ds[i]["text_truncated"]) for i in todo]

        try:
            async for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks),
                                    desc="Generating summaries"):
                idx, summary = await coro
                if summary:
                    checkpoint[idx] = summary
                if len(checkpoint) % 500 == 0:
                    save_checkpoint()
        except (KeyboardInterrupt, Exception) as exc:
            print(f"\nInterrupted ({exc.__class__.__name__}). Saving checkpoint...")
            save_checkpoint()
            print(f"  {len(checkpoint)} summaries saved to {CHECKPOINT}")
            print("  Re-run to resume.")
            sys.exit(1)

        save_checkpoint()
        print(f"\nCheckpoint saved to {CHECKPOINT}")

    summaries = [checkpoint.get(i, "") for i in range(n)]

    if n < len(ds):
        existing = list(ds["summary"]) if "summary" in ds.column_names else [""] * len(ds)
        for i, s in enumerate(summaries):
            existing[i] = s
        summaries = existing

    if "summary" in ds.column_names:
        ds = ds.remove_columns(["summary"])
    ds = ds.add_column("summary", summaries)

    tmp = Path(args.data_dir + "_tmp")
    ds.save_to_disk(str(tmp))
    shutil.rmtree(args.data_dir)
    tmp.rename(args.data_dir)
    print(f"Saved dataset with 'summary' column to {args.data_dir}")

    sample = ds[0]
    print(f"\nExample:")
    print(f"  text_truncated: {sample['text_truncated'][:120]}...")
    print(f"  summary:        {sample['summary']}")


if __name__ == "__main__":
    asyncio.run(main())
