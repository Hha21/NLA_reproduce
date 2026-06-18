"""
Stage 2: generate linguistic-feature explanations for each text_truncated.

Uses the NLA paper's analysis prompt (from the paper appendix, warmstart-data-
generation section): asks the LLM to identify 4-5 features a language model
would use to predict the next token at the truncation point. Responses are
validated for <analysis>...</analysis> tags; malformed responses are retried.

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

# Prompt from the NLA paper appendix (warmstart-data-generation section).
# Asks for 4-5 features about what the LM is "thinking about" at the
# truncation point — directly targeting the information x_l encodes.
# NOTE: the reference GitHub repo (stage2_api_explain.py) uses a shorter
# 2-3 feature version; the paper itself specifies 4-5 features / 150-200 words.
_PROMPT = """\
A language model needs to predict what text comes next after a snippet which \
will be presented to you shortly. Identify the 4-5 most important features it \
would use for this prediction.

Focus on what the language model must be "thinking about" at the point where \
the provided text ends. You should not need reference the fact that the text is \
truncated/incomplete/a prefix: the language model is causal, so only sees the \
prefix to what it predicts and this is implicit.

Order features by what is most important for predicting the next tokens.
Each feature should consist of a ~5-15 word description. When describing \
patterns, feel free to:

Note when patterns repeat, mirror, or continue from earlier in the text
Show how the same pattern manifests in multiple forms
Use probabilistic language ("often", "typically", etc.) when the model faces \
uncertainty
Feel free to include specific - and relevant - textual examples inline.
Track both what tokens are expected AND what contextual patterns determine \
those expectations

Feature types to consider (as inspiration, not a rigid checklist):

Syntactic/structural constraints: "unclosed parenthesis from line 3 requires \
matching closing parenthesis before statement ends"
Immediate semantic expectations: "list promised three items but only two given, \
third item now required"
Stylistic/register patterns: "formal academic tone maintained throughout using \
passive voice and latinate vocabulary"
Narrative/argumentative momentum: "thesis statement just completed, supporting \
evidence or first counterargument now expected"
Logical/causal dependencies: "causal premise about market conditions established, \
economic consequence must now follow"
Domain/genre signals: "medical case history following standard SOAP format, now \
in Assessment section"
Discourse/dialogue context: "speaker interrupted mid-sentence during heated \
argument, continuation of same thought expected"
Repetition/continuation patterns: "same prepositional phrase structure repeating \
with variations like 'from a X perspective'"
Distributional expectations: "verb ending in -ing typically followed by noun \
phrase in this technical documentation style"
Epistemic/meta-textual stance: "hedging language with 'may' and 'possibly' \
showing continued uncertainty about empirical claims"

Additionally:
If the text contains H: and A: markers, these indicate dialogue turns between \
a human and an assistant in a chat transcript.
The final feature in your explanation must describe the very end of the \
presented sequence: its role, what it's part of, what it implies, and how it \
relates to patterns established earlier, as appropriate.
Be specific and precise. Consider how the model tracks patterns across multiple \
levels simultaneously-surface forms, semantic content, genre conventions, and \
sequential dependencies. Features can describe both immediate next-token \
constraints and longer-range structural expectations.

Format (use up to 150-200 words total):
<analysis>
[first feature—include specific examples from text when relevant]
[second feature]
...
[final feature: analysis of last token, its role, and immediate constraints]
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


def _valid(text: str) -> bool:
    """Response must contain opening and closing analysis tags with content."""
    return "<analysis>" in text and "</analysis>" in text


async def summarise(sem: asyncio.Semaphore, idx: int, text: str) -> tuple[int, str | None]:
    async with sem:
        for attempt in range(4):
            try:
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": _PROMPT.format(text=text)}],
                    max_tokens=350,
                    temperature=0.3,
                )
                result = resp.choices[0].message.content.strip()
                if _valid(result):
                    return idx, result
                # Malformed (no closing tag) — retry without sleeping
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
        print(f"  Resuming — {len(checkpoint)} explanations already done")

    todo = [i for i in range(n) if i not in checkpoint]
    print(f"  {len(todo)} remaining\n")

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
    print(f"  summary:        {sample['summary'][:200]}")


if __name__ == "__main__":
    asyncio.run(main())
