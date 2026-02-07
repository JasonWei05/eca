"""
Fixed von Neumann entropy methods.
Bug fix: torch.diagonal needs dim1=-2, dim2=-1 (not both -1)
"""

import torch
import torch.nn.functional as F
import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def vn_entropy_exact(gram: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Baseline - exact eigvalsh."""
    eigs = torch.clamp(torch.linalg.eigvalsh(gram), min=eps)
    return -torch.sum(eigs * torch.log(eigs), dim=-1)


def vn_entropy_qr(gram: torch.Tensor, n_iter: int = 50, eps: float = 1e-10) -> torch.Tensor:
    """
    QR iteration for eigenvalues.
    A_{k+1} = R_k @ Q_k where A_k = Q_k @ R_k
    """
    A = gram.clone()
    
    for _ in range(n_iter):
        Q, R = torch.linalg.qr(A)
        A = torch.bmm(R, Q)
    
    # Get diagonal - eigenvalues (FIXED: dim1=-2, dim2=-1)
    eigs = torch.diagonal(A, dim1=-2, dim2=-1)
    eigs = torch.clamp(eigs, min=eps)
    return -torch.sum(eigs * torch.log(eigs), dim=-1)


def vn_entropy_qr_shifted(gram: torch.Tensor, n_iter: int = 30, eps: float = 1e-10) -> torch.Tensor:
    """QR with Rayleigh shift for faster convergence."""
    batch, n, _ = gram.shape
    device, dtype = gram.device, gram.dtype
    
    A = gram.clone()
    I = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)
    
    for _ in range(n_iter):
        # Rayleigh shift
        shift = A[:, -1, -1].view(-1, 1, 1)
        Q, R = torch.linalg.qr(A - shift * I)
        A = torch.bmm(R, Q) + shift * I
    
    eigs = torch.diagonal(A, dim1=-2, dim2=-1)
    eigs = torch.clamp(eigs, min=eps)
    return -torch.sum(eigs * torch.log(eigs), dim=-1)


def vn_entropy_qr_wilkinson(gram: torch.Tensor, n_iter: int = 30, eps: float = 1e-10) -> torch.Tensor:
    """QR with Wilkinson shift - better convergence than Rayleigh."""
    batch, n, _ = gram.shape
    device, dtype = gram.device, gram.dtype
    
    A = gram.clone()
    I = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)
    
    for _ in range(n_iter):
        # Wilkinson shift from bottom 2x2 block
        a = A[:, -2, -2]
        b = A[:, -2, -1]  
        c = A[:, -1, -1]
        
        delta = (a - c) / 2
        sign_delta = torch.sign(delta)
        sign_delta = torch.where(sign_delta == 0, torch.ones_like(sign_delta), sign_delta)
        
        shift = c - sign_delta * b * b / (torch.abs(delta) + torch.sqrt(delta*delta + b*b) + eps)
        shift = shift.view(-1, 1, 1)
        
        Q, R = torch.linalg.qr(A - shift * I)
        A = torch.bmm(R, Q) + shift * I
    
    eigs = torch.diagonal(A, dim1=-2, dim2=-1)
    eigs = torch.clamp(eigs, min=eps)
    return -torch.sum(eigs * torch.log(eigs), dim=-1)


def vn_entropy_subspace(gram: torch.Tensor, n_iter: int = 50, 
                        reorth_freq: int = 5, eps: float = 1e-10) -> torch.Tensor:
    """Simultaneous iteration (subspace iteration for all eigenvalues)."""
    batch, n, _ = gram.shape
    device, dtype = gram.device, gram.dtype
    
    Q = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1).contiguous()
    
    for i in range(n_iter):
        Q = torch.bmm(gram, Q)
        if (i + 1) % reorth_freq == 0:
            Q, _ = torch.linalg.qr(Q)
    
    Q, _ = torch.linalg.qr(Q)
    
    # Rayleigh quotient: H = Q^T A Q
    AQ = torch.bmm(gram, Q)
    H = torch.bmm(Q.transpose(-1, -2), AQ)
    
    # Eigenvalues are on diagonal of H
    eigs = torch.diagonal(H, dim1=-2, dim2=-1)
    eigs = torch.clamp(eigs, min=eps)
    return -torch.sum(eigs * torch.log(eigs), dim=-1)


def vn_entropy_chunked(gram: torch.Tensor, chunk_size: int = 256, eps: float = 1e-10) -> torch.Tensor:
    """Process eigvalsh in chunks - may hit better cuSOLVER paths."""
    batch = gram.shape[0]
    results = []
    
    for start in range(0, batch, chunk_size):
        end = min(start + chunk_size, batch)
        eigs = torch.linalg.eigvalsh(gram[start:end])
        eigs = torch.clamp(eigs, min=eps)
        results.append(-torch.sum(eigs * torch.log(eigs), dim=-1))
    
    return torch.cat(results)


def vn_entropy_fp16(gram: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Half precision."""
    eigs = torch.linalg.eigvalsh(gram.half()).float()
    eigs = torch.clamp(eigs, min=eps)
    return -torch.sum(eigs * torch.log(eigs), dim=-1)


def vn_entropy_bf16(gram: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Bfloat16."""
    eigs = torch.linalg.eigvalsh(gram.bfloat16()).float()
    eigs = torch.clamp(eigs, min=eps)
    return -torch.sum(eigs * torch.log(eigs), dim=-1)


def benchmark(n_tokens=4096, top_k=64, pca_dim=256):
    print(f"\n{'='*80}")
    print(f"Benchmark: {n_tokens} tokens, top_k={top_k}, pca_dim={pca_dim}")
    print(f"{'='*80}")
    
    torch.manual_seed(42)
    V = 100000
    
    proj_embeddings = F.normalize(torch.randn(V, pca_dim, device="cuda"), dim=-1)
    logits = torch.randn(n_tokens, V, device="cuda") * 5
    top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
    probs = F.softmax(top_logits, dim=-1)
    
    sel_emb = proj_embeddings[top_indices]
    w = torch.sqrt(probs).unsqueeze(-1) * sel_emb
    gram = torch.bmm(w, w.transpose(-1, -2))
    
    print(f"gram shape: {gram.shape}")
    
    # Exact baseline
    print("\nComputing exact baseline...")
    for _ in range(2):  # warmup
        _ = vn_entropy_exact(gram)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(3):
        exact = vn_entropy_exact(gram)
    torch.cuda.synchronize()
    exact_time = (time.perf_counter() - start) / 3 * 1000
    print(f"Exact eigvalsh: {exact_time:.2f} ms, mean: {exact.mean().item():.4f}")
    
    best_chunk = 2048
    
    methods = [
        ("exact (eigvalsh)", lambda: vn_entropy_exact(gram)),
        (f"chunked ({best_chunk})", lambda: vn_entropy_chunked(gram, best_chunk)),
        ("QR (30)", lambda: vn_entropy_qr(gram, 30)),
        ("QR shifted (30)", lambda: vn_entropy_qr_shifted(gram, 30)),
        ("QR shifted (50)", lambda: vn_entropy_qr_shifted(gram, 50)),
        ("QR Wilkinson (30)", lambda: vn_entropy_qr_wilkinson(gram, 30)),
        ("QR Wilkinson (50)", lambda: vn_entropy_qr_wilkinson(gram, 50)),
        ("subspace (50)", lambda: vn_entropy_subspace(gram, 50)),
        ("subspace (100)", lambda: vn_entropy_subspace(gram, 100)),
        ("fp16", lambda: vn_entropy_fp16(gram)),
        ("bf16", lambda: vn_entropy_bf16(gram)),
    ]
    
    print(f"\n{'Method':<22} | {'Time (ms)':>10} | {'Mean':>10} | {'Rel Err':>12} | {'Speedup':>8}")
    print("-" * 72)
    
    for name, fn in methods:
        try:
            for _ in range(2):
                _ = fn()
            torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(3):
                result = fn()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / 3 * 1000
            
            rel_err = (result - exact).abs().mean() / exact.abs().mean()
            speedup = exact_time / elapsed
            
            print(f"{name:<22} | {elapsed:>10.2f} | {result.mean().item():>10.4f} | {rel_err.item():>12.2e} | {speedup:>7.2f}x")
        except Exception as e:
            print(f"{name:<22} | ERROR: {e}")


if __name__ == "__main__":
    benchmark()