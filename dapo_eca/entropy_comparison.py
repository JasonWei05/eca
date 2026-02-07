"""
Entropy comparison script for analyzing Shannon and Von Neumann entropy
across inference runs on AIME2025 dataset using Qwen3-4B-Thinking model.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class RunResult:
    question_idx: int
    run_idx: int
    output_text: str
    extracted_answer: Optional[str]
    correct_answer: str
    is_correct: bool
    shannon_entropies: list[float]
    vn_entropies: dict[str, list[float]]
    num_tokens: int


@dataclass
class QuestionSummary:
    question_idx: int
    question: str
    correct_answer: str
    num_correct: int
    num_runs: int
    avg_shannon_entropy: float
    avg_vn_entropies: dict[str, float]
    avg_shannon_correct: Optional[float]
    avg_shannon_incorrect: Optional[float]


class VNEntropyCalculator:
    """Computes Von Neumann entropy using low-rank eigenvalue optimization."""

    def __init__(self, embedding_matrix: torch.Tensor, pca_dim: int = 512):
        self.device = embedding_matrix.device
        self.embeddings = F.normalize(embedding_matrix.float(), dim=-1)
        self.pca_matrix = self._compute_pca(embedding_matrix.float(), pca_dim)
        self.embeddings_proj = self.embeddings @ self.pca_matrix

    def _compute_pca(self, embeddings: torch.Tensor, n_components: int) -> torch.Tensor:
        mean = embeddings.mean(dim=0, keepdim=True)
        centered = embeddings - mean
        U, S, Vh = torch.svd_lowrank(centered.T, q=n_components)
        return U

    def compute_vn_entropy(self, logits: torch.Tensor, top_p_percent: float) -> float:
        probs = F.softmax(logits.float(), dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        # Apply top-p filtering (nucleus sampling)
        cutoff_mask = cumsum > (top_p_percent / 100.0)
        if cutoff_mask.any():
            cutoff = cutoff_mask.nonzero()[0].item() + 1
        else:
            cutoff = len(sorted_probs)

        top_p_probs = sorted_probs[:cutoff]
        top_p_indices = sorted_indices[:cutoff]

        # Renormalize to sum to 1
        top_p_probs = top_p_probs / top_p_probs.sum()

        selected_embeddings = self.embeddings_proj[top_p_indices]

        # Low-rank eigenvalue computation
        sqrt_probs = torch.sqrt(top_p_probs).unsqueeze(-1)
        weighted_embeddings = selected_embeddings * sqrt_probs

        gram_matrix = weighted_embeddings @ weighted_embeddings.T

        eigenvalues = torch.linalg.eigvalsh(gram_matrix)
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)

        vn_entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))
        return vn_entropy.item()


def compute_shannon_entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-10)
    return -torch.sum(probs * log_probs).item()


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} in generated text."""
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def check_answer(extracted: Optional[str], correct: str) -> bool:
    """Check if extracted answer matches correct answer (AIME answers are integers)."""
    if extracted is None:
        return False

    # Clean both answers
    extracted_clean = re.sub(r"[^\d.-]", "", extracted)
    correct_clean = re.sub(r"[^\d.-]", "", str(correct))

    if not extracted_clean or not correct_clean:
        return extracted.strip().lower() == str(correct).strip().lower()

    try:
        return abs(float(extracted_clean) - float(correct_clean)) < 1e-6
    except ValueError:
        return extracted.strip().lower() == str(correct).strip().lower()


def build_chat_messages(question: str) -> list[dict]:
    """Build chat messages for the model."""
    system_prompt = (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{question} Please put your final answer in \\boxed{{}}."},
    ]


def build_prompts(tokenizer, questions: list[str]) -> list[str]:
    prompts = []
    for question in questions:
        messages = build_chat_messages(question)
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            system = messages[0]["content"]
            user = messages[1]["content"]
            prompt = f"{system}\n{user}"
        prompts.append(prompt)
    return prompts


def run_transformers_inference(
    model,
    tokenizer,
    questions: list[str],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.99,
) -> list[tuple[str, list[torch.Tensor]]]:
    prompts = build_prompts(tokenizer, questions)
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    input_lengths = encoded["attention_mask"].sum(dim=1)

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )

    sequences = outputs.sequences
    scores = outputs.scores or []
    results = []

    for i, input_len in enumerate(input_lengths.tolist()):
        generated_ids = sequences[i, input_len:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        logits_list = [score[i].float().detach() for score in scores]
        results.append((output_text, logits_list))

    return results


def init_distributed():
    if "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank


def save_result_to_jsonl(result: RunResult, filepath: Path):
    """Append a single result to JSONL file."""
    record = {
        "question_idx": result.question_idx,
        "run_idx": result.run_idx,
        "output_text": result.output_text,
        "extracted_answer": result.extracted_answer,
        "correct_answer": result.correct_answer,
        "is_correct": result.is_correct,
        "shannon_entropies": result.shannon_entropies,
        "vn_entropies": result.vn_entropies,
        "num_tokens": result.num_tokens,
    }
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_completed_runs(filepath: Path) -> set[tuple[int, int]]:
    """Load set of (question_idx, run_idx) pairs already completed."""
    completed = set()
    if filepath.exists():
        with open(filepath) as f:
            for line in f:
                record = json.loads(line)
                completed.add((record["question_idx"], record["run_idx"]))
    return completed


def compute_analysis_summary(results: list[RunResult], top_p_values: list[float]) -> dict:
    """Compute summary statistics from all results."""
    by_question = {}
    for r in results:
        if r.question_idx not in by_question:
            by_question[r.question_idx] = []
        by_question[r.question_idx].append(r)

    question_summaries = []
    all_correct_shannon = []
    all_incorrect_shannon = []
    all_correct_vn = {f"p{p}": [] for p in top_p_values}
    all_incorrect_vn = {f"p{p}": [] for p in top_p_values}

    for q_idx, runs in sorted(by_question.items()):
        num_correct = sum(1 for r in runs if r.is_correct)
        num_runs = len(runs)

        all_shannon = [e for r in runs for e in r.shannon_entropies]
        avg_shannon = np.mean(all_shannon) if all_shannon else 0.0

        avg_vn = {}
        for key in all_correct_vn.keys():
            all_vn = [e for r in runs for e in r.vn_entropies.get(key, [])]
            avg_vn[key] = float(np.mean(all_vn)) if all_vn else 0.0

        correct_runs = [r for r in runs if r.is_correct]
        incorrect_runs = [r for r in runs if not r.is_correct]

        correct_shannon = [e for r in correct_runs for e in r.shannon_entropies]
        incorrect_shannon = [e for r in incorrect_runs for e in r.shannon_entropies]

        avg_shannon_correct = float(np.mean(correct_shannon)) if correct_shannon else None
        avg_shannon_incorrect = float(np.mean(incorrect_shannon)) if incorrect_shannon else None

        all_correct_shannon.extend(correct_shannon)
        all_incorrect_shannon.extend(incorrect_shannon)

        for key in all_correct_vn.keys():
            correct_vn = [e for r in correct_runs for e in r.vn_entropies.get(key, [])]
            incorrect_vn = [e for r in incorrect_runs for e in r.vn_entropies.get(key, [])]
            all_correct_vn[key].extend(correct_vn)
            all_incorrect_vn[key].extend(incorrect_vn)

        question_summaries.append(
            {
                "question_idx": q_idx,
                "num_correct": num_correct,
                "num_runs": num_runs,
                "accuracy": num_correct / num_runs if num_runs > 0 else 0.0,
                "avg_shannon_entropy": float(avg_shannon),
                "avg_vn_entropies": avg_vn,
                "avg_shannon_correct": avg_shannon_correct,
                "avg_shannon_incorrect": avg_shannon_incorrect,
            }
        )

    global_summary = {
        "total_questions": len(by_question),
        "total_runs": len(results),
        "overall_accuracy": sum(1 for r in results if r.is_correct) / len(results)
        if results
        else 0.0,
        "avg_shannon_correct": float(np.mean(all_correct_shannon))
        if all_correct_shannon
        else None,
        "avg_shannon_incorrect": float(np.mean(all_incorrect_shannon))
        if all_incorrect_shannon
        else None,
        "avg_vn_correct": {
            k: float(np.mean(v)) if v else None for k, v in all_correct_vn.items()
        },
        "avg_vn_incorrect": {
            k: float(np.mean(v)) if v else None for k, v in all_incorrect_vn.items()
        },
    }

    return {"global_summary": global_summary, "question_summaries": question_summaries}


def create_plots(results: list[RunResult], output_dir: Path, top_p_values: list[float]):
    """Generate analysis plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    correct_shannon = [e for r in results if r.is_correct for e in r.shannon_entropies]
    incorrect_shannon = [e for r in results if not r.is_correct for e in r.shannon_entropies]

    # 1. Correct vs Incorrect Shannon Entropy Box Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    data_to_plot = []
    labels = []
    if correct_shannon:
        data_to_plot.append(correct_shannon)
        labels.append(f"Correct (n={len(correct_shannon)})")
    if incorrect_shannon:
        data_to_plot.append(incorrect_shannon)
        labels.append(f"Incorrect (n={len(incorrect_shannon)})")
    if data_to_plot:
        ax.boxplot(data_to_plot, tick_labels=labels)
        ax.set_ylabel("Shannon Entropy")
        ax.set_title("Shannon Entropy: Correct vs Incorrect Responses")
        plt.tight_layout()
        plt.savefig(plots_dir / "shannon_correct_vs_incorrect.png", dpi=150)
    plt.close()

    # 2. Shannon vs VN Entropy Correlation (for default top_p=99)
    default_key = "p99"
    shannon_vals = []
    vn_vals = []
    for r in results:
        if default_key in r.vn_entropies:
            for s, v in zip(r.shannon_entropies, r.vn_entropies[default_key]):
                shannon_vals.append(s)
                vn_vals.append(v)

    if shannon_vals and vn_vals:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(shannon_vals, vn_vals, alpha=0.1, s=1)
        ax.set_xlabel("Shannon Entropy")
        ax.set_ylabel(f"Von Neumann Entropy ({default_key})")
        ax.set_title("Shannon vs Von Neumann Entropy Correlation")
        plt.tight_layout()
        plt.savefig(plots_dir / "shannon_vs_vn_correlation.png", dpi=150)
        plt.close()

    # 3. VN Entropy by top_p value
    fig, ax = plt.subplots(figsize=(10, 6))
    for p in top_p_values:
        key = f"p{p}"
        correct_vn = [e for r in results if r.is_correct for e in r.vn_entropies.get(key, [])]
        incorrect_vn = [e for r in results if not r.is_correct for e in r.vn_entropies.get(key, [])]
        if correct_vn:
            ax.scatter(
                [p - 0.2],
                [np.mean(correct_vn)],
                color="green",
                s=100,
                marker="o",
                label="Correct" if p == top_p_values[0] else "",
            )
        if incorrect_vn:
            ax.scatter(
                [p + 0.2],
                [np.mean(incorrect_vn)],
                color="red",
                s=100,
                marker="x",
                label="Incorrect" if p == top_p_values[0] else "",
            )
    ax.set_xlabel("top_p (%)")
    ax.set_ylabel("Mean Von Neumann Entropy")
    ax.set_title("Von Neumann Entropy by top_p Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "vn_entropy_by_top_p.png", dpi=150)
    plt.close()

    print(f"Plots saved to {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Entropy comparison analysis")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="Model name or path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="entropy_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-runs", type=int, default=16, help="Number of inference runs per question"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to process (for testing)",
    )
    parser.add_argument(
        "--parallel-questions",
        type=int,
        default=64,
        help="Number of questi;ons to process in parallel (batched inference)",
    )
    parser.add_argument(
        "--pca-dim", type=int, default=512, help="PCA dimension for VN entropy"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16384,
        help="Maximum new tokens to generate",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_p_values = [95.0, 99.0, 99.5]

    rank, world_size, local_rank = init_distributed()
    device = (
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"Loading model with Transformers: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model.to(device)
    model.eval()

    if rank == 0:
        print("Initializing VN entropy calculator...")
    embedding_matrix = model.get_input_embeddings().weight.detach()
    vn_calculator = VNEntropyCalculator(embedding_matrix, pca_dim=args.pca_dim)

    if rank == 0:
        print("Loading AIME2025 dataset...")
    dataset = load_dataset("MathArena/aime_2025", split="train")

    if args.max_questions is not None:
        dataset = dataset.select(range(min(args.max_questions, len(dataset))))

    if rank == 0:
        print(f"Processing {len(dataset)} questions with {args.num_runs} runs each")

    results_file = (
        output_dir / f"entropy_results_rank{rank}.jsonl"
        if world_size > 1
        else output_dir / "entropy_results.jsonl"
    )
    completed = load_completed_runs(results_file)
    if rank == 0:
        print(f"Found {len(completed)} completed runs")

    # Collect all tasks (question_idx, run_idx, question, correct_answer) to process
    tasks = []
    for q_idx, item in enumerate(dataset):
        question = item["problem"]
        correct_answer = item["answer"]
        for run_idx in range(args.num_runs):
            if (q_idx, run_idx) not in completed:
                tasks.append((q_idx, run_idx, question, correct_answer))

    tasks = [t for i, t in enumerate(tasks) if i % world_size == rank]
    if rank == 0:
        print(f"Total tasks to process: {len(tasks)}")
        print(f"Processing with parallelism: {args.parallel_questions}")

    all_results = []

    # Process tasks in batches
    batch_iter = range(0, len(tasks), args.parallel_questions)
    if rank == 0:
        batch_iter = tqdm(batch_iter, desc="Batches")
    for batch_start in batch_iter:
        batch_tasks = tasks[batch_start : batch_start + args.parallel_questions]
        batch_questions = [t[2] for t in batch_tasks]

        batch_outputs = run_transformers_inference(
            model,
            tokenizer,
            batch_questions,
            max_new_tokens=args.max_new_tokens,
        )

        # Process each result in the batch
        for (q_idx, run_idx, question, correct_answer), (
            output_text,
            logits_list,
        ) in zip(batch_tasks, batch_outputs):
            extracted = extract_boxed_answer(output_text)
            is_correct = check_answer(extracted, correct_answer)

            shannon_entropies = [compute_shannon_entropy(lg) for lg in logits_list]

            vn_entropies = {}
            for top_p in top_p_values:
                key = f"p{top_p}"
                vn_entropies[key] = [
                    vn_calculator.compute_vn_entropy(lg, top_p) for lg in logits_list
                ]

            result = RunResult(
                question_idx=q_idx,
                run_idx=run_idx,
                output_text=output_text,
                extracted_answer=extracted,
                correct_answer=correct_answer,
                is_correct=is_correct,
                shannon_entropies=shannon_entropies,
                vn_entropies=vn_entropies,
                num_tokens=len(logits_list),
            )

            save_result_to_jsonl(result, results_file)
            all_results.append(result)

            del logits_list

        torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        print("\nLoading all results for analysis...")
        all_results = []
        result_files = (
            [output_dir / f"entropy_results_rank{r}.jsonl" for r in range(world_size)]
            if world_size > 1
            else [output_dir / "entropy_results.jsonl"]
        )
        for result_file in result_files:
            if not result_file.exists():
                continue
            with open(result_file) as f:
                for line in f:
                    record = json.loads(line)
                    all_results.append(
                        RunResult(
                            question_idx=record["question_idx"],
                            run_idx=record["run_idx"],
                            output_text=record["output_text"],
                            extracted_answer=record["extracted_answer"],
                            correct_answer=record["correct_answer"],
                            is_correct=record["is_correct"],
                            shannon_entropies=record["shannon_entropies"],
                            vn_entropies=record["vn_entropies"],
                            num_tokens=record["num_tokens"],
                        )
                    )

        print(f"Analyzing {len(all_results)} total runs...")
        summary = compute_analysis_summary(all_results, top_p_values)

        summary_file = output_dir / "analysis_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")

        print("\nGenerating plots...")
        create_plots(all_results, output_dir, top_p_values)

        print("\n=== Analysis Complete ===")
        gs = summary["global_summary"]
        print(f"Total questions: {gs['total_questions']}")
        print(f"Total runs: {gs['total_runs']}")
        print(f"Overall accuracy: {gs['overall_accuracy']:.2%}")
        if gs["avg_shannon_correct"] is not None:
            print(f"Avg Shannon entropy (correct): {gs['avg_shannon_correct']:.4f}")
        if gs["avg_shannon_incorrect"] is not None:
            print(f"Avg Shannon entropy (incorrect): {gs['avg_shannon_incorrect']:.4f}")
        if gs.get("avg_vn_correct"):
            for key, value in gs["avg_vn_correct"].items():
                if value is not None:
                    print(f"Avg VN entropy {key} (correct): {value:.4f}")
        if gs.get("avg_vn_incorrect"):
            for key, value in gs["avg_vn_incorrect"].items():
                if value is not None:
                    print(f"Avg VN entropy {key} (incorrect): {value:.4f}")


if __name__ == "__main__":
    main()
