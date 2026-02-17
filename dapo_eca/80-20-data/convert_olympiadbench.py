#!/usr/bin/env python3
"""Convert math-ai/OlympiadBench to verl format (text-only problems)."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def convert_olympiadbench(output_path: str) -> None:
    """Download and convert OlympiadBench to verl format.

    Filters to text-only problems (no images) and uses the first answer
    from the final_answer list as ground truth.
    """
    print("Loading math-ai/OlympiadBench from HuggingFace...")
    ds = load_dataset("math-ai/OlympiadBench", split="test")
    print(f"Loaded {len(ds)} problems total")

    records = []
    skipped_image = 0
    skipped_no_answer = 0

    for idx, item in enumerate(ds):
        # Skip problems that require images
        has_image = any(item.get(f"image_{i}") is not None for i in range(1, 6))
        if has_image:
            skipped_image += 1
            continue

        # Get ground truth from final_answer list
        final_answer = item.get("final_answer", [])
        if not final_answer or not final_answer[0]:
            skipped_no_answer += 1
            continue

        # Use first answer, strip surrounding $ signs from LaTeX
        ground_truth = str(final_answer[0]).strip()
        if ground_truth.startswith("$") and ground_truth.endswith("$"):
            ground_truth = ground_truth[1:-1].strip()

        # Format the problem with instruction to output boxed answer
        problem_text = item["question"]
        if not problem_text.endswith("Please output the final answer within \\boxed{}."):
            problem_text += " Please output the final answer within \\boxed{}."

        record = {
            "data_source": "olympiadbench",
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
                "subfield": item.get("subfield", "unknown"),
                "difficulty": item.get("difficulty", "unknown"),
                "answer_type": item.get("answer_type", "unknown"),
                "is_multiple_answer": item.get("is_multiple_answer", False),
                "split": "test",
            },
        }
        records.append(record)

    print(f"Kept {len(records)} text-only problems")
    print(f"Skipped {skipped_image} image-dependent problems")
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
    parser = ArgumentParser(description="Convert math-ai/OlympiadBench to verl format")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save converted parquet (default: $HOME/verl/data/math__olympiadbench.parquet)",
    )
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = str(Path.home() / "verl" / "data" / "math__olympiadbench.parquet")

    convert_olympiadbench(args.output_path)


if __name__ == "__main__":
    main()
