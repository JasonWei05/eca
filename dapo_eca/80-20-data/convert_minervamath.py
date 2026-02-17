#!/usr/bin/env python3
"""Convert math-ai/minervamath to verl format."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def convert_minervamath(output_path: str) -> None:
    """Download and convert MinervaMAth to verl format.

    Uses the 'question' and 'answer' fields from the dataset.
    """
    print("Loading math-ai/minervamath from HuggingFace...")
    ds = load_dataset("math-ai/minervamath", split="test")
    print(f"Loaded {len(ds)} problems")

    records = []
    skipped_no_answer = 0

    for idx, item in enumerate(ds):
        answer = item.get("answer", "")
        if not answer:
            skipped_no_answer += 1
            continue

        ground_truth = str(answer).strip()

        # Format the problem with instruction to output boxed answer
        problem_text = item["question"]
        if not problem_text.endswith("Please output the final answer within \\boxed{}."):
            problem_text += " Please output the final answer within \\boxed{}."

        record = {
            "data_source": "minervamath",
            "prompt": [
                {
                    "content": problem_text,
                    "role": "user",
                }
            ],
            "reward_model": {
                "ground_truth": ground_truth,
                "style": "rule",
            },
            "extra_info": {
                "index": idx,
                "split": "test",
            },
        }
        records.append(record)

    print(f"Kept {len(records)} problems")
    print(f"Skipped {skipped_no_answer} problems with no answer")

    df = pd.DataFrame(records)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, index=False)
    print("Done!")

    print(f"\n=== Sample converted record ===")
    print(f"data_source: {df.iloc[0]['data_source']}")
    print(f"prompt: {df.iloc[0]['prompt']}")
    print(f"reward_model: {df.iloc[0]['reward_model']}")
    print(f"extra_info: {df.iloc[0]['extra_info']}")


def main():
    parser = ArgumentParser(description="Convert math-ai/minervamath to verl format")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save converted parquet (default: $HOME/verl/data/math__minervamath.parquet)",
    )
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = str(Path.home() / "verl" / "data" / "math__minervamath.parquet")

    convert_minervamath(args.output_path)


if __name__ == "__main__":
    main()
