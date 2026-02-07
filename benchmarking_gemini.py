import time

import torch
import torch.nn.functional as F

# --- Configuration ---
N_TOKENS = 4096
TOP_K = 64         # Standard setting
# TOP_K = 512      # Uncomment to see where SLQ truly shines (massive speedup)
VOCAB_SIZE = 32000
DEVICE = "cuda"

# Force high precision for accumulation, but bfloat16 for storage/matmul
torch.set_float32_matmul_precision('high')

def get_mock_data():
    """Generates mock embeddings and probabilities to simulate the model output."""
    print(f"Generating mock data ({N_TOKENS} tokens, Top-K {TOP_K})...")
    
    # Pre-generate embeddings for different PCA dimensions to simulate the projection step
    dims = [16, 32, 64, 128, 256, 512, 1024, 2048]
    embeddings_map = {}
    
    for d in dims:
        # Random projected embeddings (simulating centered @ V_pca)
        # using bfloat16 for H100 optimization
        embeddings_map[d] = torch.randn(VOCAB_SIZE, d, device=DEVICE, dtype=torch.bfloat16)
        
    # Random logits -> topk -> softmax
    logits = torch.randn(N_TOKENS, VOCAB_SIZE, device=DEVICE)
    top_logits, top_indices = torch.topk(logits, TOP_K, dim=-1)
    probs = F.softmax(top_logits, dim=-1).to(torch.bfloat16) # Scale in bf16
    
    return embeddings_map, probs, top_indices

# -----------------------------------------------------------------------------
# 1. Exact Implementation (Optimized with Duality + Compilation)
# -----------------------------------------------------------------------------

def exact_vn_entropy(emb, probs, indices):
    N, K = indices.shape
    _, D = emb.shape
    
    # Gather
    sel_emb = emb[indices] # (N, K, D)
    
    # Scale: w = sqrt(p) * v
    w = torch.sqrt(probs).unsqueeze(-1) * sel_emb
    
    # Duality Trick:
    # If the embedding dim (D) is smaller than the context size (K),
    # it's faster to compute eigenvalues of the DxD matrix (W.T @ W)
    # than the KxK matrix (W @ W.T). The non-zero eigenvalues are identical.
    if D < K:
        # (N, K, D) -> (N, D, K) @ (N, K, D) -> (N, D, D)
        gram = torch.matmul(w.transpose(-1, -2), w)
    else:
        # (N, K, D) @ (N, D, K) -> (N, K, K)
        gram = torch.matmul(w, w.transpose(-1, -2))
        
    # Eigendecomposition (must be float32 for stability)
    gram = gram.float()
    eigs = torch.linalg.eigvalsh(gram)
    
    # Entropy
    eigs = torch.clamp(eigs, min=1e-10)
    return -torch.sum(eigs * torch.log(eigs), dim=-1)

# -----------------------------------------------------------------------------
# 2. Stochastic Lanczos Quadrature (Approximation)
# -----------------------------------------------------------------------------

def slq_vn_entropy(emb, probs, indices, n_probes=10, steps=15):
    """
    Approximates VN entropy without full eigendecomposition.
    Best for when K is large (>128) or D is very large.
    """
    # 1. Form Gram Matrix (Bottleneck 1)
    sel_emb = emb[indices]
    w = torch.sqrt(probs).unsqueeze(-1) * sel_emb
    
    # SLQ operates on the KxK Gram matrix
    gram = torch.matmul(w, w.transpose(-1, -2)) 
    
    B, N, _ = gram.shape
    dtype = gram.dtype
    
    # 2. Lanczos Iteration (Bottleneck 2 - Matrix-Vector heavy)
    # Initialize probes
    v = torch.randint(0, 2, (B, N, n_probes), device=DEVICE).to(dtype) * 2 - 1
    v = F.normalize(v, dim=1)
    
    alphas = torch.zeros(B, n_probes, steps, device=DEVICE, dtype=dtype)
    betas = torch.zeros(B, n_probes, steps - 1, device=DEVICE, dtype=dtype)
    v_prev = torch.zeros_like(v)
    
    for j in range(steps):
        w_vec = torch.bmm(gram, v)
        if j > 0:
            beta = betas[:, :, j-1].unsqueeze(1)
            w_vec = w_vec - beta * v_prev
        
        alpha = (w_vec * v).sum(dim=1, keepdim=True)
        alphas[:, :, j] = alpha.squeeze(1)
        w_vec = w_vec - alpha * v
        
        if j < steps - 1:
            beta = torch.norm(w_vec, dim=1, keepdim=True)
            betas[:, :, j] = beta.squeeze(1)
            v_prev = v
            v = w_vec / (beta + 1e-6)

    # 3. Solve Tiny Tridiagonal Systems (Fast)
    T_size = B * n_probes
    T = torch.zeros(T_size, steps, steps, device=DEVICE, dtype=torch.float32)
    idx = torch.arange(steps, device=DEVICE)
    T[:, idx, idx] = alphas.reshape(T_size, steps).float()
    idx_off = torch.arange(steps - 1, device=DEVICE)
    b_flat = betas.reshape(T_size, steps - 1).float()
    T[:, idx_off, idx_off + 1] = b_flat
    T[:, idx_off + 1, idx_off] = b_flat
    
    eigvals, eigvecs = torch.linalg.eigh(T)
    weights = eigvecs[:, 0, :] ** 2
    
    eigvals = torch.clamp(eigvals, min=1e-10)
    func_vals = -eigvals * torch.log(eigvals)
    
    return (weights * func_vals).sum(dim=1).view(B, n_probes).mean(dim=1)

# -----------------------------------------------------------------------------
# Compilation
# -----------------------------------------------------------------------------
print("Compiling functions (this may take a minute)...")
# max-autotune creates optimized kernels for the specific H100 hardware
fast_exact = torch.compile(exact_vn_entropy, mode="max-autotune")
fast_slq = torch.compile(slq_vn_entropy, mode="max-autotune")

# -----------------------------------------------------------------------------
# Main Benchmark Loop
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    embeddings_map, probs, top_indices = get_mock_data()
    
    print(f"\n{'PCA Dim':<8} | {'Exact (ms)':<10} | {'SLQ (ms)':<10} | {'Speedup':<8} | {'Error %':<8}")
    print("-" * 65)
    
    for pca_dim in [16, 64, 256, 1024, 2048]:
        emb = embeddings_map[pca_dim]
        
        # --- Exact Benchmark ---
        # Warmup
        _ = fast_exact(emb, probs, top_indices) 
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        exact_val = fast_exact(emb, probs, top_indices)
        torch.cuda.synchronize()
        t_exact = (time.perf_counter() - start) * 1000
        
        # --- SLQ Benchmark ---
        # Warmup
        _ = fast_slq(emb, probs, top_indices)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        slq_val = fast_slq(emb, probs, top_indices)
        torch.cuda.synchronize()
        t_slq = (time.perf_counter() - start) * 1000
        
        # --- Stats ---
        speedup = t_exact / t_slq
        # Calculate mean relative error
        error = torch.mean(torch.abs(exact_val - slq_val) / (torch.abs(exact_val) + 1e-6)) * 100
        
        print(f"{pca_dim:<8d} | {t_exact:<10.2f} | {t_slq:<10.2f} | {speedup:<8.2f} | {error:<8.2f}")
    
    print("\nNote: SLQ speedup increases significantly if TOP_K > 64.")