#!/usr/bin/env python3
"""Convert MathArena/aime_2025 to verl format matching AIME 2024."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def convert_aime2025(output_path: str) -> None:
    """Download and convert AIME 2025 dataset to verl format.

    Args:
        output_path: Path to save the converted parquet file
    """
    print("Loading MathArena/aime_2025 from HuggingFace...")
    ds = load_dataset("MathArena/aime_2025", split="train")

    print(f"Loaded {len(ds)} problems")

    # Convert to the verl format matching AIME 2024
    records = []
    for item in ds:
        # Format the problem with instruction to output boxed answer
        problem_text = item["problem"]
        if not problem_text.endswith("Please output the final answer within \\boxed{}."):
            problem_text += " Please output the final answer within \\boxed{}."

        record = {
            "data_source": "aime2025",  # Starts with "aime" so routes to math_dapo scorer
            "prompt": [
                {
                    "content": problem_text,
                    "role": "user"
                }
            ],
            "reward_model": {
                "ground_truth": str(item["answer"]),  # Ensure string type
                "style": "rule"
            },
            "extra_info": {
                "problem_idx": item["problem_idx"],
                "problem_type": item["problem_type"],
                "year": 2025,
                "split": "test"
            }
        }
        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    print("\nConverted dataset:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  data_source: {df['data_source'].unique().tolist()}")

    # Save to parquet
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, index=False)
    print("Done!")

    # Show a sample
    print("\n=== Sample converted record ===")
    print(f"data_source: {df.iloc[0]['data_source']}")
    print(f"prompt: {df.iloc[0]['prompt']}")
    print(f"reward_model: {df.iloc[0]['reward_model']}")
    print(f"extra_info: {df.iloc[0]['extra_info']}")


def main():
    parser = ArgumentParser(description="Convert AIME 2025 dataset to verl format")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save converted parquet (default: $HOME/verl/data/math__aime2025_30.parquet)"
    )
    args = parser.parse_args()

    if args.output_path is None:
        from pathlib import Path
        home = Path.home()
        args.output_path = str(home / "verl" / "data" / "math__aime2025_30.parquet")

    convert_aime2025(args.output_path)


if __name__ == "__main__":
    main()
