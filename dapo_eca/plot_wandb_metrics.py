#!/usr/bin/env python3
"""Plot gradient-step metrics from a W&B run."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import wandb


STEP_SUFFIX_RE = re.compile(r"^(?P<base>.+)_step_(?P<grad_step>\d+)$")
DEFAULT_METRIC_QUERIES = [
    # Fixed probe KL vs 3 baselines (substring matches all 3 baseline variants)
    "kl_k3_first_mb_vs_",
    "kl_k3_quarter_mb_vs_",
    "kl_k3_middle_mb_vs_",
    "kl_k3_three_quarter_mb_vs_",
    "kl_k3_last_mb_vs_",
    # Current minibatch KL vs 3 baselines
    "kl_k3_current_mb_vs_",
    # Current minibatch IS-ratio vs 3 baselines
    "is_ratio_mse_current_mb_vs_",
    "is_ratio_rmse_current_mb_vs_",
    # Per-step clip fraction, TVD, grad norm
    "tvd_sampled_token",
    "pg_dualclipfrac",
    "pg_clipfrac_low",
    "pg_clipfrac_high",
    "grad_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull W&B run history and export gradient-step plots for every "
            "training step in the run. By default this saves fixed-minibatch "
            "probe KL families plus current-minibatch KL families under "
            "per-step folders."
        )
    )
    parser.add_argument(
        "--run-path",
        default="/rllm-swe/DAPO/runs/i4g9zwxl",
        help="W&B run path (entity/project/runs/run_id).",
    )
    parser.add_argument(
        "--output-dir",
        default="wandb_metric_plots_all_steps-3-12-fixed",
        help="Root directory to save per-step subfolders and plots.",
    )
    parser.add_argument(
        "--metric-types",
        nargs="+",
        default=DEFAULT_METRIC_QUERIES,
        help=(
            "Metric family queries to match for every training step. "
            "Substring match by default. Use 'all' to include all families. "
            "Default: first/quarter/middle/three-quarter/last/current minibatch KL."
        ),
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Match each --metric-types entry as an exact family name.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of training steps to export. Default: all available steps.",
    )
    parser.add_argument(
        "--step-key",
        default="_step",
        help="History column used to select training step row (default: _step).",
    )
    parser.add_argument(
        "--grad-start",
        type=int,
        default=0,
        help="Inclusive gradient-step start (default: 0).",
    )
    parser.add_argument(
        "--grad-end",
        type=int,
        default=15,
        help="Inclusive gradient-step end (default: 15).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200000,
        help="Maximum W&B history samples to fetch (default: 200000).",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List matching metric families and exit.",
    )
    return parser.parse_args()


def get_plt():
    import matplotlib.pyplot as plt

    return plt


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def choose_step_key(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    for key in ["_step", "global_step", "trainer/global_step", "step"]:
        if key in df.columns:
            print(
                f"[warn] step key '{requested}' not found; using '{key}' instead.",
                file=sys.stderr,
            )
            return key
    df["row_index"] = range(len(df))
    print(
        "[warn] no recognized step key found; using synthetic 'row_index'.",
        file=sys.stderr,
    )
    return "row_index"


def extract_metric_families(columns: list[str]) -> dict[str, dict[int, str]]:
    families: dict[str, dict[int, str]] = {}
    for col in columns:
        match = STEP_SUFFIX_RE.match(col)
        if not match:
            continue
        base = match.group("base")
        grad_step = int(match.group("grad_step"))
        families.setdefault(base, {})[grad_step] = col
    return families


def match_families(
    family_names: list[str], query: str, exact: bool
) -> list[str]:
    if query.strip().lower() in {"", "all", "*"}:
        return family_names
    if exact:
        return [query] if query in family_names else []
    lowered = query.lower()
    return [name for name in family_names if lowered in name.lower()]


def select_rows(
    df: pd.DataFrame, step_key: str, train_steps: list[int] | None
) -> list[tuple[int, pd.Series]]:
    ordered = df.sort_values(step_key).dropna(subset=[step_key])
    if ordered.empty:
        raise RuntimeError(f"No non-null rows found for step key '{step_key}'.")

    step_rows = ordered.copy()
    step_rows["_normalized_step"] = step_rows[step_key].apply(
        lambda value: int(round(float(value)))
    )
    step_rows = (
        step_rows.groupby("_normalized_step", sort=True, as_index=False)
        .tail(1)
        .sort_values("_normalized_step")
    )

    if train_steps is not None:
        requested_steps = sorted(set(train_steps))
        step_rows = step_rows[step_rows["_normalized_step"].isin(requested_steps)]
        missing_steps = sorted(set(requested_steps) - set(step_rows["_normalized_step"].tolist()))
        for missing_step in missing_steps:
            print(
                f"[warn] training step {missing_step} not found in run history; skipping.",
                file=sys.stderr,
            )

    return [
        (int(row["_normalized_step"]), row)
        for _, row in step_rows.iterrows()
    ]


def build_plot_df(
    row: pd.Series,
    families: dict[str, dict[int, str]],
    selected_families: list[str],
    grad_start: int,
    grad_end: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for family in selected_families:
        step_to_col = families[family]
        for grad_step, col in sorted(step_to_col.items()):
            if grad_step < grad_start or grad_step > grad_end:
                continue
            value = row.get(col, pd.NA)
            if pd.isna(value):
                continue
            rows.append(
                {
                    "metric_family": family,
                    "gradient_step": grad_step,
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows)


def resolve_metric_queries(
    family_names: list[str],
    metric_queries: list[str],
    exact: bool,
) -> dict[str, list[str]]:
    matched_by_query: dict[str, list[str]] = {}
    for query in metric_queries:
        matched_families = match_families(family_names, query, exact)
        if not matched_families:
            print(f"[warn] no metric families matched query '{query}'.", file=sys.stderr)
            continue
        matched_by_query[query] = matched_families
    return matched_by_query


def plot_individual(plot_df: pd.DataFrame, family: str, out_path: Path) -> bool:
    plt = get_plt()
    subset = plot_df[plot_df["metric_family"] == family].sort_values("gradient_step")
    if subset.empty:
        return False
    plt.figure(figsize=(7.8, 4.5))
    plt.plot(
        subset["gradient_step"],
        subset["value"],
        marker="o",
        linewidth=1.2,
        markersize=3.0,
    )
    plt.xlabel("gradient_step")
    plt.ylabel("value")
    plt.title(family)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def plot_combined(plot_df: pd.DataFrame, out_path: Path) -> bool:
    plt = get_plt()
    if plot_df.empty:
        return False
    plt.figure(figsize=(9, 5))
    plotted = False
    for family in sorted(plot_df["metric_family"].unique()):
        subset = plot_df[plot_df["metric_family"] == family].sort_values("gradient_step")
        if subset.empty:
            continue
        plt.plot(
            subset["gradient_step"],
            subset["value"],
            marker="o",
            linewidth=1.2,
            markersize=2.8,
            label=family,
        )
        plotted = True
    if not plotted:
        plt.close()
        return False
    plt.xlabel("gradient_step")
    plt.ylabel("value")
    plt.title("Matched metric families")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def save_query_outputs(
    row: pd.Series,
    families: dict[str, dict[int, str]],
    matched_families: list[str],
    output_dir: Path,
    run_id: str,
    train_step: int,
    metric_query: str,
    grad_start: int,
    grad_end: int,
) -> int:
    plot_df = build_plot_df(
        row=row,
        families=families,
        selected_families=matched_families,
        grad_start=grad_start,
        grad_end=grad_end,
    )
    if plot_df.empty:
        print(
            f"Skipped '{metric_query}' for training step {train_step}: "
            f"no values in gradient-step range [{grad_start}, {grad_end}]."
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    query_slug = sanitize_filename(metric_query)
    run_slug = sanitize_filename(run_id)

    pivot_df = (
        plot_df.pivot(index="gradient_step", columns="metric_family", values="value")
        .sort_index()
        .reset_index()
    )
    csv_path = output_dir / (
        f"run_{run_slug}_trainstep_{train_step}_"
        f"grad_{grad_start}_{grad_end}_{query_slug}.csv"
    )
    pivot_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    saved = 0
    for family in matched_families:
        png_path = output_dir / (
            f"run_{run_slug}_trainstep_{train_step}_"
            f"{sanitize_filename(family)}_g{grad_start}-{grad_end}.png"
        )
        if plot_individual(plot_df, family, png_path):
            print(f"Saved plot: {png_path}")
            saved += 1
        else:
            print(f"Skipped empty family: {family}")

    combined_path = output_dir / (
        f"run_{run_slug}_trainstep_{train_step}_"
        f"combined_{query_slug}_g{grad_start}-{grad_end}.png"
    )
    if plot_combined(plot_df, combined_path):
        print(f"Saved combined plot: {combined_path}")
    else:
        print(f"No combined plot generated for '{metric_query}'.")

    return saved


def main() -> int:
    args = parse_args()
    if args.grad_end < args.grad_start:
        raise ValueError("--grad-end must be >= --grad-start")

    run_path = args.run_path.lstrip("/")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading run: {run_path}")
    run = wandb.Api().run(run_path)
    history_df = run.history(samples=args.samples)
    if history_df.empty:
        print("Run history is empty.")
        return 1

    step_key = choose_step_key(history_df, args.step_key)
    families = extract_metric_families(list(history_df.columns))
    if not families:
        print(
            "No metrics with '_step_<k>' suffix were found. "
            "This script expects gradient-step metrics in that format."
        )
        return 1

    all_families = sorted(families.keys())
    matched_by_query = resolve_metric_queries(all_families, args.metric_types, args.exact)
    if not matched_by_query:
        print("No metric families matched any requested query.")
        print("Available metric families:")
        for name in all_families:
            print(f"  - {name}")
        return 1

    print(f"Matched metric queries ({len(matched_by_query)}):")
    for query, matched_families in matched_by_query.items():
        print(f"  - {query}")
        for name in matched_families:
            print(f"    * {name}")

    if args.list_only:
        return 0

    selected_rows = select_rows(history_df, step_key, args.train_steps)
    if not selected_rows:
        print("No training steps were selected for export.")
        return 1

    total_saved = 0
    for train_step, row in selected_rows:
        step_dir = output_dir / f"trainstep_{train_step}"
        print(f"Exporting training step {train_step} -> {step_dir}")
        for metric_query, matched_families in matched_by_query.items():
            metric_dir = step_dir / sanitize_filename(metric_query)
            total_saved += save_query_outputs(
                row=row,
                families=families,
                matched_families=matched_families,
                output_dir=metric_dir,
                run_id=run.id,
                train_step=train_step,
                metric_query=metric_query,
                grad_start=args.grad_start,
                grad_end=args.grad_end,
            )

    print(
        f"Done. Exported {len(selected_rows)} training step(s) "
        f"into {output_dir} and generated {total_saved} individual plot(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
