# Low-Rank Eigenvalue Computation: Mathematical Details

**Supplement to Von Neumann Entropy Implementation**

This document provides detailed mathematical derivations for the low-rank eigenvalue computation trick.

---

## The Problem Setup

We want to compute eigenvalues of the density matrix:

```
ρ = Σ(i=1 to k) p_i · e_i · e_i^T
```

Where:
- `p_i` are probabilities (sum to 1)
- `e_i` are normalized embedding vectors (d-dimensional, ||e_i|| = 1)
- Each `e_i · e_i^T` is a rank-1 matrix (d×d)
- `ρ` is d×d, symmetric, positive semi-definite

**Challenge:**
- Eigendecomposition of d×d matrix costs O(d³) FLOPs
- For d=5120: ~134 billion FLOPs per position
- We need a better way!

---

## Key Mathematical Insight

### The Fundamental Theorem

**Theorem:** For any matrix E with shape (d×k) where k ≤ d:

The **non-zero eigenvalues** of `E @ E^T` (d×d) are **exactly the same** as the non-zero eigenvalues of `E^T @ E` (k×k).

The matrix `E @ E^T` has (d-k) additional zero eigenvalues.

**Why this matters:**
- `E @ E^T` is d×d (expensive to eigendecompose)
- `E^T @ E` is k×k (cheap to eigendecompose)
- When k << d (e.g., k=50, d=5120), this is **much** faster!

---

## Proof of the Fundamental Theorem

Let me prove why the non-zero eigenvalues are the same.

### Forward Direction: E@E^T → E^T@E

Suppose λ ≠ 0 is an eigenvalue of `E @ E^T` with eigenvector v:

```
(E @ E^T) v = λ v    ... (1)
```

Now multiply both sides on the left by `E^T`:

```
E^T @ (E @ E^T) @ v = λ E^T @ v
(E^T @ E) @ (E^T @ v) = λ (E^T @ v)    ... (2)
```

Define `w = E^T @ v`. Then equation (2) becomes:

```
(E^T @ E) w = λ w    ... (3)
```

So w is an eigenvector of `E^T @ E` with the same eigenvalue λ!

**But wait:** We need to verify that w ≠ 0.

From equation (1): `E @ E^T @ v = λ v`

Take inner product with v:
```
v^T @ E @ E^T @ v = λ v^T @ v
||E^T @ v||² = λ ||v||²
```

Since λ ≠ 0 and ||v|| ≠ 0 (v is eigenvector), we have ||E^T @ v|| ≠ 0, so w ≠ 0. ✓

### Backward Direction: E^T@E → E@E^T

Similarly, if λ ≠ 0 is an eigenvalue of `E^T @ E` with eigenvector w:

```
(E^T @ E) w = λ w    ... (4)
```

Multiply both sides on the left by E:

```
E @ (E^T @ E) @ w = λ E @ w
(E @ E^T) @ (E @ w) = λ (E @ w)    ... (5)
```

Define `v = E @ w`. Then:

```
(E @ E^T) v = λ v    ... (6)
```

Again, we can verify v ≠ 0 by taking norms.

**Conclusion:** The non-zero eigenvalues of `E @ E^T` and `E^T @ E` are identical!

---

## Application to Density Matrix ρ

Now let's apply this to our density matrix.

### Step 1: Reformulate ρ as E @ E^T

Recall:
```
ρ = Σ(i=1 to k) p_i · e_i · e_i^T
```

We can write this in matrix form. Define:
```
E = [√p_1·e_1, √p_2·e_2, ..., √p_k·e_k]
```

Where E is a (d×k) matrix whose columns are the scaled embeddings.

**Then:**
```
E @ E^T = [√p_1·e_1, √p_2·e_2, ..., √p_k·e_k] @ [√p_1·e_1, √p_2·e_2, ..., √p_k·e_k]^T

        = Σ(i=1 to k) (√p_i·e_i) @ (√p_i·e_i)^T

        = Σ(i=1 to k) p_i · e_i · e_i^T

        = ρ
```

Perfect! We've expressed ρ in the form `E @ E^T`.

### Step 2: Compute Gram Matrix G = E^T @ E

Instead of eigendecomposing ρ (d×d), we eigendecompose:

```
G = E^T @ E    (k×k matrix)
```

Let's expand this:

```
G[i,j] = (√p_i·e_i)^T @ (√p_j·e_j)
       = √p_i · √p_j · (e_i^T @ e_j)
       = √p_i · √p_j · <e_i, e_j>
```

So G is the **Gram matrix** of the scaled embeddings.

**In code:**
```python
# E is (k, d) - each row is √p_i · e_i
E = sqrt(p)[:, None] * embeddings  # Broadcasting: (k, 1) * (k, d) = (k, d)

# Gram matrix G = E @ E^T (but E is stored with embeddings as rows, not columns)
# So we actually compute G = E @ E^T
G = E @ E.T  # (k, d) @ (d, k) = (k, k)
```

### Step 3: Eigendecompose the Gram Matrix

Now we eigendecompose the small (k×k) matrix G:

```python
eigenvalues = torch.linalg.eigvalsh(G)  # Returns k eigenvalues
```

By our theorem, these are **exactly the non-zero eigenvalues of ρ**!

The remaining (d-k) eigenvalues of ρ are zero (but they don't contribute to the entropy anyway, since 0·log(0) = 0).

### Step 4: Compute von Neumann Entropy

```python
# Eigenvalues of G are eigenvalues of ρ
# VN entropy: H = -Σ λ log λ

# Handle λ=0 case: 0·log(0) → 0 by convention
log_eigenvalues = torch.log(eigenvalues + eps)
vn_entropy = -torch.sum(eigenvalues * log_eigenvalues)
```

---

## Why This Is Much Faster

### Complexity Comparison

**Naive approach (eigendecompose ρ directly):**
- Form ρ: O(k·d²) to compute Σ p_i · e_i · e_i^T
- Eigendecompose ρ (d×d): O(d³)
- **Total: O(d³)** dominates

For d=5120: ~134 billion FLOPs

**Low-rank approach (eigendecompose G = E^T @ E):**
- Form E: O(k·d) to scale embeddings by √p
- Compute G = E^T @ E: O(k²·d)
- Eigendecompose G (k×k): O(k³)
- **Total: O(k²·d + k³)**

For k=50, d=5120:
- k²·d = 50² × 5120 = 12.8 million FLOPs
- k³ = 50³ = 125 thousand FLOPs
- **Total: ~13 million FLOPs**

**Speedup: 134 billion / 13 million ≈ 10,000×** 🎉

---

## Concrete Example: Worked Through

Let's do a small concrete example to see this in action.

### Setup

Suppose:
- Vocabulary: 5 tokens
- Embedding dimension: 3
- After top-k selection with k=3, we have:
  - Token 1: p₁=0.5, e₁=[1, 0, 0]
  - Token 2: p₂=0.3, e₂=[0, 1, 0]
  - Token 3: p₃=0.2, e₃=[0, 0, 1]

(Using orthonormal embeddings for simplicity)

### Method 1: Direct Computation of ρ (Naive)

```
ρ = p₁·e₁·e₁^T + p₂·e₂·e₂^T + p₃·e₃·e₃^T

  = 0.5·[1,0,0]^T·[1,0,0] + 0.3·[0,1,0]^T·[0,1,0] + 0.2·[0,0,1]^T·[0,0,1]

  = 0.5·[1 0 0]   + 0.3·[0 0 0]   + 0.2·[0 0 0]
       [0 0 0]         [0 1 0]         [0 0 0]
       [0 0 0]         [0 0 0]         [0 0 1]

  = [0.5  0   0 ]
    [0   0.3  0 ]
    [0    0  0.2]
```

Eigenvalues of ρ: **λ = [0.5, 0.3, 0.2]**

VN entropy:
```
H = -(0.5·log(0.5) + 0.3·log(0.3) + 0.2·log(0.2))
  = -(0.5·(-0.693) + 0.3·(-1.204) + 0.2·(-1.609))
  = -(-0.347 - 0.361 - 0.322)
  = 1.030
```

### Method 2: Gram Matrix (Efficient)

Form scaled embedding matrix E:
```
E = [√p₁·e₁]   = [√0.5 · [1,0,0]]   = [0.707  0     0   ]
    [√p₂·e₂]     [√0.3 · [0,1,0]]     [0      0.548  0   ]
    [√p₃·e₃]     [√0.2 · [0,0,1]]     [0      0     0.447]
```

So E is a (3×3) matrix in this case (k=d=3).

Compute Gram matrix G = E @ E^T:
```
G = [0.707  0     0   ] @ [0.707  0      0    ]
    [0      0.548  0   ]   [0      0.548  0    ]
    [0      0     0.447]   [0      0      0.447]

  = [0.5  0    0  ]
    [0    0.3  0  ]
    [0    0    0.2]
```

Eigenvalues of G: **λ = [0.5, 0.3, 0.2]** ✓

**Same eigenvalues!** And we only had to eigendecompose a (3×3) matrix instead of... well, in this case also (3×3), but imagine if the full embedding space was d=5120!

### More Realistic Example: k << d

Now suppose:
- Full embedding dimension: d=5120
- After top-k: k=50 tokens
- Embeddings are arbitrary vectors in ℝ^5120

**Naive method:**
- Form ρ (5120×5120): ~130 MB memory
- Eigendecompose (5120×5120): ~134 billion FLOPs

**Gram matrix method:**
- Form E (50×5120): ~1 MB memory
- Compute G = E @ E^T (50×50): 50² × 5120 = 12.8M FLOPs
- Eigendecompose (50×50): 50³ = 125k FLOPs
- **Total: ~13M FLOPs, ~1 MB memory**

The eigenvalues of G give us the 50 non-zero eigenvalues of ρ. The other 5070 eigenvalues of ρ are zero (but contribute nothing to entropy).

---

## Intuition: Why Does This Work?

### Geometric Intuition

Think of ρ as describing an **ellipsoid in d-dimensional space**:
- The eigenvalues are the squared lengths of the principal axes
- The eigenvectors are the directions of these axes

When ρ = Σ p_i · e_i · e_i^T:
- We're adding up k rank-1 projections
- The resulting ellipsoid lives in a k-dimensional subspace (spanned by {e₁, ..., eₖ})
- Outside this subspace, all eigenvalues are zero

So we only care about eigenvalues in the k-dimensional subspace where the action is happening!

The Gram matrix G = E^T @ E computes these eigenvalues **in the coordinates of the embeddings** rather than the full d-dimensional space.

### Linear Algebra Intuition

Another way to think about it:

The Gram matrix G tells us about relationships between the embeddings:
```
G[i,j] = √p_i · √p_j · cos(θ_ij)
```

Where θ_ij is the angle between embeddings e_i and e_j.

- If embeddings are **orthogonal** (semantically diverse), G is nearly diagonal, eigenvalues spread out → **high VN entropy**
- If embeddings are **aligned** (semantically similar), G has correlations, eigenvalues concentrate → **low VN entropy**

This is exactly what we want to measure!

---

## Implementation in PyTorch

Here's the clean implementation using the low-rank trick:

```python
def compute_vn_entropy_lowrank(
    top_k_probs: torch.Tensor,         # (batch, seq_len, k)
    selected_embeddings: torch.Tensor, # (batch, seq_len, k, d)
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute VN entropy using low-rank eigenvalue trick.

    Args:
        top_k_probs: Probabilities of top-k tokens (sum to ~1)
        selected_embeddings: Embedding vectors of top-k tokens (normalized)
        eps: Small constant for numerical stability

    Returns:
        vn_entropy: (batch, seq_len) VN entropy per position
    """
    # Step 1: Form scaled embedding matrix E
    # E[i] = √p_i · e_i for each token i
    sqrt_probs = torch.sqrt(top_k_probs).unsqueeze(-1)  # (batch, seq_len, k, 1)
    E = selected_embeddings * sqrt_probs                 # (batch, seq_len, k, d)

    # Step 2: Compute Gram matrix G = E @ E^T
    # For efficiency, use einsum to batch over (batch, seq_len) dimensions
    G = torch.einsum('...ki,...kj->...ij', E, E)  # (batch, seq_len, k, k)

    # Step 3: Eigendecompose G (small k×k matrices)
    eigenvalues = torch.linalg.eigvalsh(G)  # (batch, seq_len, k)

    # Step 4: Compute VN entropy H = -Σ λ log λ
    eigenvalues = torch.clamp(eigenvalues, min=eps)  # Numerical stability
    log_eigenvalues = torch.log(eigenvalues)
    vn_entropy = -torch.sum(eigenvalues * log_eigenvalues, dim=-1)

    return vn_entropy  # (batch, seq_len)
```

### Why einsum?

The line `torch.einsum('...ki,...kj->...ij', E, E)` computes:
- For each batch and sequence position (represented by `...`)
- Take the k embeddings (each d-dimensional)
- Compute the k×k Gram matrix: G[i,j] = Σ_d E[i,d] * E[j,d]

This is equivalent to:
```python
G = E @ E.transpose(-2, -1)  # Also works, but einsum is more explicit
```

---

## Connection to SVD

There's another way to see this using the **Singular Value Decomposition (SVD)**.

### SVD Perspective

For matrix E (d×k):
```
E = U @ Σ @ V^T
```

Where:
- U is (d×k) with orthonormal columns
- Σ is (k×k) diagonal with singular values σ₁, ..., σₖ
- V is (k×k) orthogonal

Then:
```
E @ E^T = (U @ Σ @ V^T) @ (V @ Σ @ U^T)
        = U @ Σ @ (V^T @ V) @ Σ @ U^T
        = U @ Σ² @ U^T
```

So the eigenvalues of E @ E^T are σ₁², σ₂², ..., σₖ² (plus d-k zeros).

Similarly:
```
E^T @ E = (V @ Σ @ U^T) @ (U @ Σ @ V^T)
        = V @ Σ @ (U^T @ U) @ Σ @ V^T
        = V @ Σ² @ V^T
```

So the eigenvalues of E^T @ E are also σ₁², σ₂², ..., σₖ².

**Same eigenvalues!** (Via a different route)

---

## Numerical Considerations

### Stability Issues

When k is large or embeddings are nearly parallel, the Gram matrix G can be:
- **Ill-conditioned** (large condition number)
- Have **very small eigenvalues** (near machine precision)

**Solutions:**

1. **Regularization:**
   ```python
   G = G + eps * torch.eye(k)  # Add small ridge
   ```

2. **Clipping:**
   ```python
   eigenvalues = torch.clamp(eigenvalues, min=eps)
   ```

3. **Use double precision for critical step:**
   ```python
   G_fp64 = G.double()
   eigenvalues = torch.linalg.eigvalsh(G_fp64).float()
   ```

### Negative Eigenvalues

Due to numerical errors, G might have tiny negative eigenvalues (should be ≥0).

**Solution:** Clip them to zero or small epsilon:
```python
eigenvalues = torch.clamp(eigenvalues, min=eps)
```

This is safe because they're essentially zero and contribute 0·log(0) ≈ 0 to entropy.

---

## Summary

### The Low-Rank Trick in One Picture

```
ρ = Σ p_i · e_i · e_i^T
    ↓ (reformulate)
  = E @ E^T          where E = [√p_1·e_1, ..., √p_k·e_k]
    ↓ (size)
  (d×d matrix)      E is (d×k)

Instead of eigendecomposing ρ (expensive):
  eigenvalues(ρ) = eigenvalues(E @ E^T) ← O(d³) cost

Use theorem to compute:
  eigenvalues(ρ) = eigenvalues(E^T @ E) ← O(k³) cost
                            ↑
                          = G (Gram matrix)
                          (k×k matrix)

When k << d: Massive speedup! 🚀
```

### Key Takeaways

1. **ρ has rank ≤ k** (sum of k rank-1 matrices)
2. **Non-zero eigenvalues of AB and BA are the same** (fundamental theorem)
3. **Gram matrix G = E^T @ E** captures all information about ρ's eigenvalues
4. **Complexity reduction:** O(d³) → O(k²d + k³)
5. **Speedup for k=50, d=5120:** ~10,000×

This is why the low-rank approach makes von Neumann entropy computationally feasible!
