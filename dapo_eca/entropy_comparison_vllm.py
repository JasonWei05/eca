import argparse
import json
import os
from pathlib import Path
from typing import Optional
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from entropy_comparison import (
    VNEntropyCalculator,
    RunResult,
    compute_shannon_entropy,
    extract_boxed_answer,
    check_answer,
    compute_analysis_summary,
    create_plots,
    save_result_to_jsonl,
    load_completed_runs,
    build_chat_messages,
)


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


def _token_to_id(tokenizer, token: str) -> Optional[int]:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) == 1:
            return encoded[0]
        return None
    return token_id


def call_vllm_chat(
    base_url: str,
    model: str,
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_logprobs: int,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    url = base_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_vllm_inference(
    tokenizer,
    questions: list[str],
    base_url: str,
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_logprobs: int,
) -> list[tuple[str, list[torch.Tensor]]]:
    results = []
    vocab_size = tokenizer.vocab_size
    for question in questions:
        messages = build_chat_messages(question)
        response = call_vllm_chat(
            base_url=base_url,
            model=model,
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_logprobs=top_logprobs,
        )
        choice = response["choices"][0]
        output_text = choice["message"]["content"]
        logprobs = choice.get("logprobs", {})
        token_logprobs = logprobs.get("content") or []
        logits_list = []
        for token_info in token_logprobs:
            top_list = token_info.get("top_logprobs") or []
            logits = torch.full((vocab_size,), -100.0, dtype=torch.float32)
            for item in top_list:
                token = item.get("token")
                logprob = item.get("logprob")
                if token is None or logprob is None:
                    continue
                token_id = _token_to_id(tokenizer, token)
                if token_id is None:
                    continue
                logits[token_id] = logprob
            logits_list.append(logits)
        results.append((output_text, logits_list))
    return results


def parse_top_p_values(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def main():
    parser = argparse.ArgumentParser(description="Entropy comparison via vLLM server")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="Model name on the vLLM server",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:30000",
        help="vLLM OpenAI server base URL",
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
        default=1,
        help="Number of questions to process sequentially per batch",
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
    parser.add_argument(
        "--top-p-values",
        type=str,
        default="95,99,99.5",
        help="Comma-separated top-p percentages for VN entropy",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.99, help="Sampling top-p"
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=10000,
        help="Top logprobs to request from vLLM",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_p_values = parse_top_p_values(args.top_p_values)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model.eval()
    embedding_matrix = model.get_input_embeddings().weight.detach()
    vn_calculator = VNEntropyCalculator(embedding_matrix, pca_dim=args.pca_dim)

    dataset = load_dataset("MathArena/aime_2025", split="train")
    if args.max_questions is not None:
        dataset = dataset.select(range(min(args.max_questions, len(dataset))))

    results_file = output_dir / "entropy_results.jsonl"
    completed = load_completed_runs(results_file)

    tasks = []
    for q_idx, item in enumerate(dataset):
        question = item["problem"]
        correct_answer = item["answer"]
        for run_idx in range(args.num_runs):
            if (q_idx, run_idx) not in completed:
                tasks.append((q_idx, run_idx, question, correct_answer))

    all_results = []
    for batch_start in range(0, len(tasks), args.parallel_questions):
        batch_tasks = tasks[batch_start : batch_start + args.parallel_questions]
        batch_questions = [t[2] for t in batch_tasks]
        batch_outputs = run_vllm_inference(
            tokenizer=tokenizer,
            questions=batch_questions,
            base_url=args.base_url,
            model=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_logprobs=args.top_logprobs,
        )
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

    summary = compute_analysis_summary(all_results, top_p_values)
    summary_file = output_dir / "analysis_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    create_plots(all_results, output_dir, top_p_values)


if __name__ == "__main__":
    main()
