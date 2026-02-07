"""
AIME 2024 inference + entropy analysis for Qwen3-4B-Base.
Computes Shannon entropy and VN entropy across generated tokens.
Caches logits to disk so decoding only happens once.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from safetensors import safe_open

MODEL_PATH = "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"
MAX_NEW_TOKENS = 2048
CACHE_FILE = "/home/tiger/jason/eca/aime_outputs.json"

# VN entropy configs
TOP_KS = [64]
PCA_DIMS = [32, 64, 128]


def compute_pca_matrix(embeddings, pca_dim):
    """Compute PCA projection matrix from embedding matrix."""
    centered = embeddings - embeddings.mean(dim=0)
    _, _, V = torch.pca_lowrank(centered, q=pca_dim)
    return V  # (D, pca_dim)


def compute_vn_entropy(logits_seq, embedding_matrix, pca_matrices, top_k, pca_dim):
    """
    Compute VN entropy for a sequence of logits.
    logits_seq: (seq_len, vocab_size)
    Returns: (seq_len,) tensor of VN entropy values.
    """
    V_pca = pca_matrices[pca_dim]
    proj_emb = F.normalize(embedding_matrix @ V_pca, dim=-1)  # (vocab, pca_dim)

    top_logits, top_indices = torch.topk(logits_seq, top_k, dim=-1)  # (seq_len, top_k)
    probs = F.softmax(top_logits.float(), dim=-1)

    sel_emb = proj_emb[top_indices]  # (seq_len, top_k, pca_dim)
    w = torch.sqrt(probs).unsqueeze(-1) * sel_emb  # (seq_len, top_k, pca_dim)

    # Duality trick
    K, d = w.shape[-2], w.shape[-1]
    if d < K:
        gram = w.transpose(-1, -2) @ w
    else:
        gram = w @ w.transpose(-1, -2)

    eigs = torch.clamp(torch.linalg.eigvalsh(gram), min=1e-10)
    return -torch.sum(eigs * torch.log(eigs), dim=-1)


def generate_and_cache():
    """Run inference and save logits + metadata to disk."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print("Loading dataset...")
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    problems = ds["Problem"]
    print(f"Found {len(problems)} problems")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float8_e4m3fn, device_map="auto", low_cpu_mem_usage=True   
    )
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = []
    for problem in problems:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"{problem} Put your final answer in \\boxed{{}}"},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    print(f"Batched input shape: {inputs['input_ids'].shape}")

    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=1.0,
            top_k=0,
            temperature=1.0,
            return_dict_in_generate=True,
            output_logits=True,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[:, input_len:]

    eos_token_id = tokenizer.eos_token_id
    cache_data = []
    for i in range(generated_ids.shape[0]):
        eos_positions = (generated_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]
        actual_len = eos_positions[0].item() + 1 if len(eos_positions) > 0 else generated_ids.shape[1]
        gen_text = tokenizer.decode(generated_ids[i, :actual_len], skip_special_tokens=True)
        cache_data.append({
            "problem": problems[i],
            "prompt": texts[i],
            "generated_text": gen_text,
        })

    import json
    print(f"Saving cache to {CACHE_FILE}...")
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=2)
    print("Cache saved.")


def main():
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load or generate
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached outputs from {CACHE_FILE}...")
        with open(CACHE_FILE) as f:
            cache_data = json.load(f)
        print(f"Loaded {len(cache_data)} problems from cache")
    else:
        generate_and_cache()
        with open(CACHE_FILE) as f:
            cache_data = json.load(f)

    # Load model for forward pass to get logits
    print("Loading model for logit computation...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    )
    model.eval()

    # Get embedding matrix for VN entropy
    print("Loading embedding matrix for VN entropy...")
    with safe_open(f"{MODEL_PATH}/model-00001-of-00003.safetensors", framework="pt") as f:
        raw_embeddings = f.get_tensor("model.embed_tokens.weight").float().cuda()

    centered = raw_embeddings - raw_embeddings.mean(dim=0)

    print("Computing PCA matrices...")
    pca_matrices = {}
    for pca_dim in PCA_DIMS:
        pca_matrices[pca_dim] = compute_pca_matrix(raw_embeddings, pca_dim)
        print(f"  PCA dim={pca_dim} done")

    emb_normalized = F.normalize(centered, dim=-1)

    # Storage
    all_shannon = []
    all_vn = {}
    for tk in TOP_KS:
        for pd in PCA_DIMS:
            all_vn[(tk, pd)] = []

    B = len(cache_data)
    for i, item in enumerate(cache_data):
        # Reconstruct full sequence: prompt + generated text
        full_text = item["prompt"] + item["generated_text"]
        input_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(model.device)
        prompt_ids = tokenizer(item["prompt"], return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]
        gen_len = input_ids.shape[1] - prompt_len

        print(f"\n--- Problem {i+1}/{B}: {gen_len} tokens ---")

        # Forward pass to get logits
        with torch.no_grad():
            outputs = model(input_ids)
        # Logits for generated tokens: logits at positions [prompt_len-1 : -1] predict tokens [prompt_len:]
        logits_seq = outputs.logits[0, prompt_len - 1:-1].float()  # (gen_len, vocab_size)

        # 1. Shannon entropy
        probs_full = F.softmax(logits_seq, dim=-1)
        shannon = -torch.sum(probs_full * torch.log(probs_full + 1e-10), dim=-1)
        all_shannon.append(shannon.cpu().numpy())
        print(f"  Shannon: mean={shannon.mean().item():.4f}, std={shannon.std().item():.4f}")

        # 2. VN entropy
        for tk in TOP_KS:
            for pd in PCA_DIMS:
                chunk_size = 2048
                vn_chunks = []
                for start in range(0, gen_len, chunk_size):
                    end = min(start + chunk_size, gen_len)
                    chunk = logits_seq[start:end]
                    vn = compute_vn_entropy(chunk, emb_normalized, pca_matrices, tk, pd)
                    vn_chunks.append(vn.cpu().numpy())
                vn_all = np.concatenate(vn_chunks)
                all_vn[(tk, pd)].append(vn_all)
                print(f"  VN(k={tk}, pca={pd}): mean={vn_all.mean():.4f}, std={vn_all.std():.4f}")

    # Aggregate all tokens across all problems
    all_shannon_flat = np.concatenate(all_shannon)
    all_vn_flat = {}
    for key in all_vn:
        all_vn_flat[key] = np.concatenate(all_vn[key])

    # Print summary stats
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS (across all tokens, all problems)")
    print(f"{'='*60}")
    print(f"Shannon entropy: mean={all_shannon_flat.mean():.4f}, std={all_shannon_flat.std():.4f}")
    for (tk, pd), vals in all_vn_flat.items():
        print(f"VN(k={tk}, pca={pd}): mean={vals.mean():.4f}, std={vals.std():.4f}")

    # Plot overlapping distributions
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    bins = np.linspace(0, 4, 20)

    from scipy.stats import gaussian_kde

    # Shannon
    ax.hist(all_shannon_flat, bins=bins, alpha=0.3, density=True, label="Shannon entropy")
    kde = gaussian_kde(all_shannon_flat.clip(0, 4))
    x = np.linspace(0, 4, 500)
    ax.plot(x, kde(x), linewidth=2, label="Shannon (kde)")

    # VN entropy for each config
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_vn_flat)))
    for idx, ((tk, pd), vals) in enumerate(all_vn_flat.items()):
        clipped = vals.clip(0, 4)
        ax.hist(clipped, bins=bins, alpha=0.15, density=True, color=colors[idx])
        kde = gaussian_kde(clipped)
        ax.plot(x, kde(x), linewidth=2, label=f"VN(k={tk}, pca={pd})", color=colors[idx])

    ax.set_xlim(0, 4)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e2)
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("Entropy Distributions: Shannon vs VN (AIME 2024, Qwen3-4B-Base)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/home/tiger/jason/eca/entropy_distributions.png", dpi=150)
    print("\nPlot saved to entropy_distributions.png")


if __name__ == "__main__":
    main()
