# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Entropy comparison script for analyzing Shannon and Von Neumann entropy
across inference runs on AIME2025 dataset using Qwen3-4B-Thinking model.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams


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


def run_vllm_inference(
    llm: LLM, questions: list[str], max_new_tokens: int = 16384
) -> list[tuple[str, list[torch.Tensor]]]:
    """
    Run vLLM inference for multiple questions in parallel.

    Returns:
        List of (output_text, logits_list) tuples, one per question.
    """
    # Build chat prompts - vLLM handles chat templates automatically
    chat_inputs = [build_chat_messages(q) for q in questions]

    # Configure sampling parameters
    # Note: logprobs parameter controls how many top tokens to return logprobs for
    # Request as many as possible to get good entropy estimates
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.99,
        max_tokens=max_new_tokens,
        logprobs=10000,  # Request logprobs for top tokens (must match max_logprobs in LLM)
    )

    # Run generation - vLLM handles batching automatically
    outputs = llm.chat(chat_inputs, sampling_params=sampling_params)

    results = []
    vocab_size = llm.llm_engine.model_config.hf_config.vocab_size

    for output in outputs:
        output_text = output.outputs[0].text

        # Convert vLLM logprobs to full vocab logits tensors
        logits_list = []
        for token_logprobs in output.outputs[0].logprobs:
            if token_logprobs is None:
                continue

            # token_logprobs is a dict {token_id: Logprob object}
            # Create logits tensor filled with very negative values (-100 in log space)
            logits = torch.full((vocab_size,), -100.0, dtype=torch.float32)

            # Fill in the logprobs we have from vLLM
            for token_id, logprob_obj in token_logprobs.items():
                logits[token_id] = logprob_obj.logprob

            # Move to GPU if available
            if torch.cuda.is_available():
                logits = logits.cuda()

            logits_list.append(logits)

        results.append((output_text, logits_list))

    return results


def process_question(
    model,
    tokenizer,
    vn_calculator: VNEntropyCalculator,
    question: str,
    correct_answer: str,
    question_idx: int,
    num_runs: int,
    top_p_values: list[float],
) -> list[RunResult]:
    """Process a single question with multiple inference runs."""
    results = []

    for run_idx in range(num_runs):
        output_text, logits_list = run_inference(model, tokenizer, question)

        extracted = extract_boxed_answer(output_text)
        is_correct = check_answer(extracted, correct_answer)

        shannon_entropies = [compute_shannon_entropy(lg) for lg in logits_list]

        vn_entropies = {}
        for top_p in top_p_values:
            key = f"p{top_p}"
            vn_entropies[key] = [
                vn_calculator.compute_vn_entropy(lg, top_p) for lg in logits_list
            ]

        results.append(
            RunResult(
                question_idx=question_idx,
                run_idx=run_idx,
                output_text=output_text,
                extracted_answer=extracted,
                correct_answer=correct_answer,
                is_correct=is_correct,
                shannon_entropies=shannon_entropies,
                vn_entropies=vn_entropies,
                num_tokens=len(logits_list),
            )
        )

        del logits_list
        torch.cuda.empty_cache()

    return results


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
        "--num-runs", type=int, default=4, help="Number of inference runs per question"
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
        default=32,
        help="Number of questions to process in parallel (batched inference)",
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

    print(f"Loading model with vLLM: {args.model}")

    # Set GCC paths (PyTorch requires GCC 9+)
    gcc_home = "/opt/packages/gcc/v13.3.1-p20240614/b2gpu"
    os.environ["CC"] = f"{gcc_home}/bin/gcc"
    os.environ["CXX"] = f"{gcc_home}/bin/g++"
    os.environ["PATH"] = f"{gcc_home}/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        f"{gcc_home}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    )

    # Set CUDA paths for vLLM (required for custom kernels)
    cuda_home = "/opt/packages/cuda/v12.6.1"
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["CUDA_PATH"] = cuda_home
    os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        f"{cuda_home}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    )

    # vLLM configuration
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,  # Reduced to avoid OOM
        enforce_eager=False,  # Use CUDA graph for better performance
        max_logprobs=10000,  # Increase max logprobs limit for entropy calculation
    )

    print("Initializing VN entropy calculator...")
    # Get embedding matrix from vLLM model
    try:
        # Try to access through model executor (vLLM internal structure)
        model_runner = llm.llm_engine.model_executor.driver_worker
        embedding_matrix = model_runner.model_runner.model.model.embed_tokens.weight.detach()
    except (AttributeError, RuntimeError) as e:
        print(f"Failed to extract embeddings from vLLM: {e}")
        print("Loading model separately with transformers to get embeddings...")
        from transformers import AutoModel
        temp_model = AutoModel.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        embedding_matrix = temp_model.embed_tokens.weight.detach().cpu()
        del temp_model
        torch.cuda.empty_cache()

    vn_calculator = VNEntropyCalculator(embedding_matrix, pca_dim=args.pca_dim)

    print("Loading AIME2025 dataset...")
    dataset = load_dataset("MathArena/aime_2025", split="train")

    if args.max_questions is not None:
        dataset = dataset.select(range(min(args.max_questions, len(dataset))))

    print(f"Processing {len(dataset)} questions with {args.num_runs} runs each")

    results_file = output_dir / "entropy_results.jsonl"
    completed = load_completed_runs(results_file)
    print(f"Found {len(completed)} completed runs")

    # Collect all tasks (question_idx, run_idx, question, correct_answer) to process
    tasks = []
    for q_idx, item in enumerate(dataset):
        question = item["problem"]
        correct_answer = item["answer"]
        for run_idx in range(args.num_runs):
            if (q_idx, run_idx) not in completed:
                tasks.append((q_idx, run_idx, question, correct_answer))

    print(f"Total tasks to process: {len(tasks)}")
    print(f"Processing with parallelism: {args.parallel_questions}")

    all_results = []

    # Process tasks in batches
    for batch_start in tqdm(
        range(0, len(tasks), args.parallel_questions), desc="Batches"
    ):
        batch_tasks = tasks[batch_start : batch_start + args.parallel_questions]
        batch_questions = [t[2] for t in batch_tasks]

        # Run vLLM inference (handles batching automatically)
        batch_outputs = run_vllm_inference(
            llm, batch_questions, max_new_tokens=args.max_new_tokens
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

    # Load all results for analysis
    print("\nLoading all results for analysis...")
    all_results = []
    with open(results_file) as f:
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


if __name__ == "__main__":
    main()
