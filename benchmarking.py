import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

model_path = "Qwen3-8B-Base"

# Load only the embedding layer
with safe_open(f"{model_path}/model-00001-of-00005.safetensors", framework="pt") as f:
    raw_embeddings = f.get_tensor("model.embed_tokens.weight").float().cuda()

tokenizer = AutoTokenizer.from_pretrained("/home/tiger/jason/eca/Qwen3-8B-Base")

print(f"Embedding shape: {raw_embeddings.shape}")  # (vocab_size, hidden_dim)
V, D = raw_embeddings.shape

# --- PCA benchmark ---
centered = raw_embeddings - raw_embeddings.mean(dim=0)
for dim in reversed([16, 32, 64, 128, 256, 512, 1024, 2048]):
    torch.cuda.synchronize()
    start = time.perf_counter()
    U, S, Vh = torch.pca_lowrank(centered, q=dim)
    torch.cuda.synchronize()
    print(f"PCA dim={dim:4d}  time={time.perf_counter() - start:.4f}s")

# --- VN entropy benchmark ---
# Generate 1024 random logit distributions, top-k 512, softmax, top-p 0.99
N_TOKENS = 8192
TOP_K = 32

logits = torch.randn(N_TOKENS, V, device="cuda") * 5
top_logits, top_indices = torch.topk(logits, TOP_K, dim=-1)
probs = F.softmax(top_logits, dim=-1)

def vn_entropy_fn(w):
    # Duality trick: use smaller Gram matrix
    K, d = w.shape[-2], w.shape[-1]
    if d < K:
        gram = w.transpose(-1, -2) @ w  # (N, d, d) instead of (N, K, K)
    else:
        gram = w @ w.transpose(-1, -2)
    eigs = torch.clamp(torch.linalg.eigvalsh(gram), min=1e-10)
    return -torch.sum(eigs * torch.log(eigs), dim=-1)

print(f"\nVN entropy benchmark: {N_TOKENS} tokens, top_k={TOP_K}")

for pca_dim in reversed([16, 32, 64, 128, 256, 512, 1024, 2048]):
    # PCA project centered data, then normalize for VN entropy (Tr(rho)=1)
    _, _, V_pca = torch.pca_lowrank(centered, q=pca_dim)  # V_pca: (D, pca_dim)
    proj_embeddings = F.normalize(centered @ V_pca, dim=-1)  # (vocab, pca_dim)

    sel_emb = proj_embeddings[top_indices]  # (N_TOKENS, TOP_K, pca_dim)
    w = torch.sqrt(probs).unsqueeze(-1) * sel_emb

    # Warmup
    vn_entropy_fn(w)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    vn_entropy = vn_entropy_fn(w)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"VN entropy pca_dim={pca_dim:4d}  time={elapsed:.4f}s  mean_entropy={vn_entropy.mean().item():.4f}")
