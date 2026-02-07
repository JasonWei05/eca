"""
Check if ordering is preserved between pairs of entropy measurements.
For N random token pairs, check if measurement_a[i] > measurement_a[j]
implies measurement_b[i] > measurement_b[j].
"""

import json
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL_PATH = "/home/tiger/jason/eca/Qwen3-4b-Instruct"
CACHE_FILE = "/home/tiger/jason/eca/aime_outputs.json"
TOP_K = 64
PCA_DIMS = [32, 64, 128]
N_PAIRS = 5000


def compute_pca_matrix(embeddings, pca_dim):
    centered = embeddings - embeddings.mean(dim=0)
    _, _, V = torch.pca_lowrank(centered, q=pca_dim)
    return V


def compute_vn_entropy(logits_seq, emb_normalized, pca_matrices, top_k, pca_dim):
    V_pca = pca_matrices[pca_dim]
    proj_emb = F.normalize(emb_normalized @ V_pca, dim=-1)
    top_logits, top_indices = torch.topk(logits_seq, top_k, dim=-1)
    probs = F.softmax(top_logits.float(), dim=-1)
    sel_emb = proj_emb[top_indices]
    w = torch.sqrt(probs).unsqueeze(-1) * sel_emb
    K, d = w.shape[-2], w.shape[-1]
    if d < K:
        gram = w.transpose(-1, -2) @ w
    else:
        gram = w @ w.transpose(-1, -2)
    gram = gram + 1e-6 * torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).unsqueeze(0)
    eigs = torch.clamp(torch.linalg.eigvalsh(gram), min=1e-10)
    return -torch.sum(eigs * torch.log(eigs), dim=-1)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(CACHE_FILE) as f:
        cache_data = json.load(f)
    print(f"Loaded {len(cache_data)} problems from cache")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Embedding matrix
    with safe_open(f"{MODEL_PATH}/model-00001-of-00003.safetensors", framework="pt") as f:
        raw_embeddings = f.get_tensor("model.embed_tokens.weight").float().cuda()
    centered = raw_embeddings - raw_embeddings.mean(dim=0)
    emb_normalized = F.normalize(centered, dim=-1)

    pca_matrices = {}
    for pd in PCA_DIMS:
        pca_matrices[pd] = compute_pca_matrix(raw_embeddings, pd)

    # Collect per-token entropies across all problems
    measurement_names = ["Shannon"] + [f"VN(k={TOP_K},pca={pd})" for pd in PCA_DIMS]
    all_values = {name: [] for name in measurement_names}

    for i, item in enumerate(cache_data):
        full_text = item["prompt"] + item["generated_text"]
        input_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(model.device)
        prompt_ids = tokenizer(item["prompt"], return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]
        gen_len = input_ids.shape[1] - prompt_len

        print(f"Problem {i+1}/{len(cache_data)}: {gen_len} tokens")

        with torch.no_grad():
            outputs = model(input_ids)
        logits_seq = outputs.logits[0, prompt_len - 1:-1].float()

        # Shannon
        probs_full = F.softmax(logits_seq, dim=-1)
        shannon = -torch.sum(probs_full * torch.log(probs_full + 1e-10), dim=-1)
        all_values["Shannon"].append(shannon.cpu().numpy())

        # VN entropy
        for pd in PCA_DIMS:
            chunk_size = 2048
            vn_chunks = []
            for start in range(0, gen_len, chunk_size):
                end = min(start + chunk_size, gen_len)
                vn = compute_vn_entropy(logits_seq[start:end], emb_normalized, pca_matrices, TOP_K, pd)
                vn_chunks.append(vn.cpu().numpy())
            vn_all = np.concatenate(vn_chunks)
            all_values[f"VN(k={TOP_K},pca={pd})"].append(vn_all)

    # Flatten all tokens
    for name in measurement_names:
        all_values[name] = np.concatenate(all_values[name])

    total_tokens = len(all_values["Shannon"])
    print(f"\nTotal tokens: {total_tokens}")

    # Sample N random pairs of token indices
    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, total_tokens, size=N_PAIRS)
    idx_b = rng.integers(0, total_tokens, size=N_PAIRS)
    # Avoid identical pairs
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % total_tokens

    # Check ordering preservation for all pairs of measurements
    pairs = list(combinations(measurement_names, 2))

    print(f"\nOrdering preservation ({N_PAIRS} random token pairs):")
    print(f"{'Measurement A':<25} {'Measurement B':<25} {'Preserved %':>12}")
    print("-" * 65)

    for name_a, name_b in pairs:
        vals_a = all_values[name_a]
        vals_b = all_values[name_b]

        diff_a = vals_a[idx_a] - vals_a[idx_b]
        diff_b = vals_b[idx_a] - vals_b[idx_b]

        # Ordering preserved: same sign (both positive or both negative)
        # Exclude ties (diff == 0)
        non_tie = (diff_a != 0) & (diff_b != 0)
        if non_tie.sum() == 0:
            pct = float('nan')
        else:
            preserved = ((diff_a > 0) == (diff_b > 0))[non_tie].mean() * 100
            pct = preserved

        print(f"{name_a:<25} {name_b:<25} {pct:>11.1f}%")


if __name__ == "__main__":
    main()
