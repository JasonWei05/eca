#!/usr/bin/env python3
"""Convert MathArena AIME datasets to verl format."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def convert_aime(dataset_name: str, year: int, output_path: str) -> None:
    """Download and convert a MathArena AIME dataset to verl format.

    Args:
        dataset_name: HuggingFace dataset name (e.g., MathArena/aime_2025)
        year: Year for extra_info and data_source label
        output_path: Path to save the converted parquet file
    """
    print(f"Loading {dataset_name} from HuggingFace...")
    ds = load_dataset(dataset_name, split="train")

    print(f"Loaded {len(ds)} problems")

    data_source = f"aime{year}"

    # Convert to the verl format matching AIME 2024
    records = []
    for idx, item in enumerate(ds):
        # Format the problem with instruction to output boxed answer
        problem_text = item["problem"]
        if not problem_text.endswith("Please output the final answer within \\boxed{}."):
            problem_text += " Please output the final answer within \\boxed{}."

        record = {
            "data_source": data_source,  # Starts with "aime" so routes to math_verify scorer
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
                "problem_idx": item.get("problem_idx", idx),
                "problem_type": ", ".join(item["problem_type"]) if isinstance(item.get("problem_type"), list) else str(item.get("problem_type", "unknown")),
                "year": year,
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
    parser = ArgumentParser(description="Convert MathArena AIME dataset to verl format")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MathArena/aime_2025",
        help="HuggingFace dataset name (default: MathArena/aime_2025)"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="AIME year (default: 2025)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save converted parquet (default: $HOME/verl/data/math__aime{year}_{n}.parquet)"
    )
    args = parser.parse_args()

    if args.output_path is None:
        home = Path.home()
        args.output_path = str(home / "verl" / "data" / f"math__aime{args.year}_30.parquet")

    convert_aime(args.dataset, args.year, args.output_path)


if __name__ == "__main__":
    main()
