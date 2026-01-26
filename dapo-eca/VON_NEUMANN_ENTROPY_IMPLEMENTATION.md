# Von Neumann Entropy Implementation Analysis

**Author**: Analysis for recipe/dapo-eca
**Date**: 2026-01-25
**Purpose**: Implement semantic uncertainty measurement via von Neumann entropy in the embedding space

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Computational Complexity Analysis](#computational-complexity-analysis)
3. [Optimization Strategies](#optimization-strategies)
4. [Fetching Embedding Matrices](#fetching-embedding-matrices)
5. [Implementation Plan](#implementation-plan)
6. [Performance Estimates](#performance-estimates)
7. [Recommended Approach](#recommended-approach)

---

## Problem Statement

### Theoretical Background

**Von Neumann Entropy** measures semantic uncertainty in the embedding space by constructing a density matrix:

```
ρ = Σ(i=1 to V) p_i · e_i · e_i^T
```

Where:
- `p_i` = probability of token i from softmax output
- `e_i` = normalized token embedding vector (dimension d)
- `ρ` ∈ ℝ^(d×d) = symmetric positive semi-definite density matrix

The von Neumann entropy is then:

```
H_VN(ρ) = -Tr(ρ log ρ) = -Σ(k=1 to d) λ_k log λ_k
```

Where `{λ_k}` are the eigenvalues of `ρ`.

### Why This Matters

- **Shannon entropy** only considers probability distribution, treating all tokens equally
- **VN entropy** considers semantic similarity: tokens with similar embeddings contribute to same eigenspace
- **Low VN entropy**: Model is uncertain but predictions are semantically clustered (coherent uncertainty)
- **High VN entropy**: Model is uncertain across semantically diverse directions (incoherent uncertainty)

---

## Computational Complexity Analysis

### Naive Implementation Costs

For a typical LLM:
- Vocabulary size: `V = 32,000 - 150,000` tokens
- Embedding dimension: `d = 2,048 - 8,192` (e.g., Qwen2.5-32B has d=5,120)
- Batch size: `B = 512 - 2,048` sequences
- Sequence length: `L = 512 - 20,480` tokens per sequence

**Per-position computation:**

1. **Construct density matrix**:
   - Form `V` outer products: `V × d × d` FLOPs
   - For d=5120, V=150k: ~393 billion FLOPs per position
   - Memory: `d × d × 4 bytes` (FP32) = ~100 MB for d=5120

2. **Eigendecomposition**:
   - Full eigendecomposition: `O(d³)` FLOPs
   - For d=5120: ~134 billion FLOPs per position
   - Using iterative methods (Lanczos): `O(k·d²)` where k << d

3. **Per-batch cost**:
   - Positions per batch: `B × L` = potentially 512 × 20,480 = 10.5M positions
   - **Total FLOPs per batch**: ~5.5 × 10^18 FLOPs (5.5 exaFLOPs!)
   - **This is completely infeasible**

### The Fundamental Challenge

Computing VN entropy for every token position in every sequence during RL training is **3-4 orders of magnitude more expensive** than the forward pass itself. This is why aggressive optimization is mandatory.

---

## Optimization Strategies

### Your Proposed Ideas: Evaluation

#### 1. **PCA Every N (~10) Steps** ✅ GOOD
**Pros:**
- Reduces dimensionality from d to k (e.g., 5120 → 512)
- PCA preserves maximum variance, capturing semantic structure
- Can cache PCA projection matrix between updates

**Cons:**
- Still need to compute PCA on full embedding matrix: O(V·d²) cost
- PCA computation itself is expensive (~10-30s for large models)
- May miss rare semantic directions if k is too small

**Recommendation:** Use with k = 256-512 (5-10% of d)

#### 2. **Top-p (~99%) Tokens Only** ✅ EXCELLENT
**Pros:**
- Reduces V from 150k to ~100-500 tokens per position (99.9% → ~20-50 tokens)
- Most probability mass is on few tokens anyway
- Low-probability tokens contribute negligibly to density matrix

**Cons:**
- Slight approximation error (but negligible for p=0.99)

**Recommendation:** Use top-p=0.99 or even top-k=50-100 for speed

---

### Better/Additional Optimization Strategies

#### 3. **Low-Rank Approximation of ρ** ✅✅ BEST
**Key Insight:** The density matrix ρ is inherently low-rank!

If only k tokens have significant probability (top-k selection), then:
```
ρ ≈ Σ(i=1 to k) p_i · e_i · e_i^T
```

This is a **rank-k matrix** where k is typically 10-100 (not 5,120!).

**Implication:**
- ρ has at most k non-zero eigenvalues
- Can use **randomized SVD** or **power iteration** to find top-k eigenvalues
- Complexity: O(k²·d) instead of O(d³)
- For k=50, d=5120: ~13 million FLOPs (instead of 134 billion!)

**Implementation:**
```python
# Instead of full eigen decomposition
eigenvalues, _ = torch.linalg.eigh(rho)  # O(d³)

# Use low-rank structure (ρ = E @ diag(p) @ E^T)
E = embeddings[top_k_indices]  # (k, d)
p = probs[top_k_indices]       # (k,)
# Eigenvalues of ρ are eigenvalues of (E^T @ E) weighted by p
# This reduces to O(k²·d + k³) which is MUCH smaller
```

#### 4. **Stratified Sampling of Positions** ✅
Don't compute VN entropy for ALL positions:

**Strategy A: Sample random positions**
- Compute for 1% of positions: 512 × 20480 × 0.01 = 100k positions → 5k positions
- Still representative of distribution

**Strategy B: Sample by generation stage**
- Early tokens (positions 0-10%): Semantic foundation
- Middle tokens (positions 40-60%): Reasoning core
- Late tokens (positions 90-100%): Conclusion
- Sample 10-20 positions per sequence stratified by stage

**Strategy C: Adaptive sampling**
- Compute VN entropy when Shannon entropy exceeds threshold
- Only track "interesting" uncertain positions

#### 5. **Incremental Computation Over Trajectory** ✅
If tracking VN entropy over a rollout trajectory:
```python
# Don't recompute from scratch each time
# Update incrementally as tokens are generated
rho_t = rho_{t-1} + p_t · e_t · e_t^T
# Then use online eigenvalue tracking algorithms
```

**Savings:** Amortized O(d²) per token instead of O(d³) per position

#### 6. **Mixed Precision & GPU Kernels** ✅
- Use FP16 for embeddings and density matrix (halves memory)
- Custom CUDA kernels for `Σ p_i · e_i · e_i^T` (fused operation)
- Use cuSOLVER's batched eigenvalue solver for multiple positions

#### 7. **Shared Embedding Cache** ✅✅
**Critical insight:** Embedding matrix doesn't change during rollout!

```python
# Fetch once per RL step (not per batch)
embedding_matrix = model.get_input_embeddings().weight  # (V, d)
embedding_matrix = F.normalize(embedding_matrix, dim=-1)  # Normalize once

# Cache for entire rollout phase
# Only re-fetch after actor training updates weights
```

**Savings:** Eliminates redundant embedding fetches

#### 8. **Approximate Eigenvalue Methods** ✅
Don't need exact eigenvalues for a metric:

**Hutchinson's Trace Estimator:**
```python
# Approximate Tr(ρ log ρ) via Monte Carlo
# Draw random vectors z ~ N(0, I)
# Tr(A) ≈ (1/m) Σ z^T A z
```

**Power Iteration for Top-k Eigenvalues:**
```python
# Only compute top-k largest eigenvalues (capture most entropy)
# Ignore tiny eigenvalues (contribute ~0 to -λ log λ anyway)
```

---

## Fetching Embedding Matrices

### Location in Model Architecture

For HuggingFace/verl models, embeddings are typically:

**LLaMA, Qwen, Mistral:**
```python
# Input embeddings
model.model.embed_tokens.weight  # (vocab_size, hidden_dim)

# Output embeddings (if not tied)
model.lm_head.weight  # (vocab_size, hidden_dim)

# Tied weights (most common)
# lm_head.weight = model.embed_tokens.weight.T (or shared reference)
```

**GPT-2 style:**
```python
model.transformer.wte.weight  # (vocab_size, hidden_dim)
```

**Check for tied embeddings:**
```python
# In model config
model.config.tie_word_embeddings  # True/False
```

### Fetching in verl Framework

#### From Actor Worker (during training)

In `recipe/dapo-eca/dapo_ray_trainer.py`, you can access the model via worker groups:

```python
# In RayDAPOTrainer class
def get_embedding_matrix(self):
    """Fetch embedding matrix from actor model"""
    # This would require adding a new method to actor worker
    # Return shape: (vocab_size, hidden_dim)
    return self.actor_rollout_wg.get_embeddings()
```

**Implementation in worker** (add to `verl/workers/fsdp_workers.py` or similar):
```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def get_embeddings(self):
    """Return normalized embedding matrix"""
    if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
        # LLaMA/Qwen/Mistral style
        embeddings = self.model.model.embed_tokens.weight.data
    elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
        # GPT-2 style
        embeddings = self.model.transformer.wte.weight.data
    else:
        raise ValueError("Unknown model architecture for embeddings")

    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=-1)
    return embeddings.cpu()  # Move to CPU to save GPU memory
```

#### Handling Distributed/Sharded Models

**Challenge:** In FSDP/Megatron, embeddings may be sharded across GPUs.

**Solution:**
```python
# For FSDP
with FSDP.summon_full_params(model, writeback=False):
    embeddings = model.model.embed_tokens.weight.data.clone()

# For Megatron (TP sharded)
# Embeddings are typically replicated, not sharded
# Or use all_gather to collect shards
```

### Memory Considerations

**Embedding matrix size:**
- Qwen2.5-32B: 152,064 tokens × 5,120 dims × 2 bytes (FP16) = **1.56 GB**
- DeepSeek-671B: ~150k × 8,192 × 2 bytes = **2.46 GB**

**Strategy:**
- Fetch to CPU and cache (not GPU)
- Only transfer selected embeddings to GPU when needed
- Use memory-mapped files for very large models

---

## Implementation Plan

### Phase 1: Core Implementation (Week 1)

**File: `recipe/dapo-eca/von_neumann_entropy.py`**

```python
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class VonNeumannEntropyCalculator:
    """
    Compute von Neumann entropy in embedding space with optimizations.

    Optimizations applied:
    1. Top-k token selection (reduce V from 150k to ~50-100)
    2. Low-rank eigenvalue computation (exploit rank-k structure)
    3. Stratified position sampling (reduce L by 100x)
    4. Cached embeddings (fetch once per RL step)
    5. PCA projection (optional, reduce d by 10x)
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,  # (vocab_size, embed_dim)
        top_k: int = 50,
        use_pca: bool = True,
        pca_dim: int = 512,
        device: str = 'cuda',
    ):
        self.V, self.d = embedding_matrix.shape
        self.top_k = top_k
        self.device = device

        # Normalize embeddings
        self.embeddings = F.normalize(embedding_matrix, dim=-1)

        # Optional PCA projection
        self.use_pca = use_pca
        if use_pca and pca_dim < self.d:
            self.pca_dim = pca_dim
            self.pca_matrix = self._compute_pca(embedding_matrix, pca_dim)
            self.embeddings_proj = self.embeddings @ self.pca_matrix  # (V, pca_dim)
        else:
            self.pca_dim = self.d
            self.embeddings_proj = self.embeddings

    def _compute_pca(self, embeddings: torch.Tensor, n_components: int) -> torch.Tensor:
        """
        Compute PCA projection matrix using randomized SVD for efficiency.

        Returns:
            pca_matrix: (d, n_components) projection matrix
        """
        # Center embeddings
        mean = embeddings.mean(dim=0, keepdim=True)
        centered = embeddings - mean

        # Use randomized SVD for speed (much faster than full SVD)
        # For very large matrices, use torch.svd_lowrank
        try:
            U, S, Vh = torch.svd_lowrank(centered.T, q=n_components)
            pca_matrix = U  # (d, n_components)
        except:
            # Fallback to standard SVD
            U, S, Vh = torch.svd(centered.T)
            pca_matrix = U[:, :n_components]

        return pca_matrix

    def compute_density_matrix(
        self,
        probs: torch.Tensor,  # (batch_size, seq_len, vocab_size)
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute density matrix using low-rank structure.

        Args:
            probs: Token probabilities from softmax
            top_k: Number of top tokens to consider (default: self.top_k)

        Returns:
            rho: Density matrix (batch_size, seq_len, d, d) - sparse representation
            top_k_info: (top_k_probs, top_k_indices) for analysis
        """
        if top_k is None:
            top_k = self.top_k

        # Select top-k tokens
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
        # (batch_size, seq_len, top_k)

        # Normalize to ensure sum to 1 (since we dropped low-prob tokens)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Fetch embeddings for top-k tokens
        # Use projected embeddings if PCA is enabled
        selected_embeddings = self.embeddings_proj[top_k_indices]
        # (batch_size, seq_len, top_k, pca_dim)

        return top_k_probs, selected_embeddings

    def compute_vn_entropy_lowrank(
        self,
        top_k_probs: torch.Tensor,      # (batch_size, seq_len, k)
        selected_embeddings: torch.Tensor,  # (batch_size, seq_len, k, d)
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """
        Efficiently compute VN entropy using low-rank structure.

        Key insight: ρ = Σ p_i e_i e_i^T is rank-k, so we can compute
        eigenvalues via the (k×k) Gram matrix instead of (d×d) density matrix.

        Mathematical trick:
        - Form E = [√p_1·e_1, √p_2·e_2, ..., √p_k·e_k]  # (k, d)
        - Then ρ = E @ E^T  (d×d, rank ≤ k)
        - Eigenvalues of ρ are same as eigenvalues of E^T @ E  (k×k)
        - Much cheaper: O(k³ + k²d) instead of O(d³)

        Args:
            top_k_probs: Probabilities of top-k tokens
            selected_embeddings: Embeddings of top-k tokens
            eps: Small constant for numerical stability

        Returns:
            vn_entropy: (batch_size, seq_len) von Neumann entropy per position
        """
        batch_size, seq_len, k, d = selected_embeddings.shape

        # Weight embeddings by sqrt(probability)
        sqrt_probs = torch.sqrt(top_k_probs).unsqueeze(-1)  # (B, L, k, 1)
        weighted_embeddings = selected_embeddings * sqrt_probs  # (B, L, k, d)

        # Compute Gram matrix: G = E^T @ E  (k×k instead of d×d!)
        # G[i,j] = √p_i √p_j · <e_i, e_j>
        gram_matrix = torch.einsum('...ki,...kj->...ij',
                                   weighted_embeddings,
                                   weighted_embeddings)
        # (batch_size, seq_len, k, k)

        # Eigendecomposition of k×k matrix (much cheaper than d×d)
        eigenvalues = torch.linalg.eigvalsh(gram_matrix)  # (B, L, k)

        # Clip small negative eigenvalues (numerical errors)
        eigenvalues = torch.clamp(eigenvalues, min=eps)

        # Compute von Neumann entropy: H = -Σ λ log λ
        # Use log(λ + eps) for numerical stability
        log_eigenvalues = torch.log(eigenvalues + eps)
        vn_entropy = -torch.sum(eigenvalues * log_eigenvalues, dim=-1)
        # (batch_size, seq_len)

        return vn_entropy

    def compute_batch_vn_entropy(
        self,
        probs: torch.Tensor,  # (batch_size, seq_len, vocab_size)
        sample_positions: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute VN entropy for a batch with optional position sampling.

        Args:
            probs: Token probabilities from softmax
            sample_positions: (batch_size, n_samples) indices to compute entropy for
                            If None, compute for all positions

        Returns:
            Dictionary with:
                - vn_entropy_mean: Scalar mean VN entropy
                - vn_entropy_std: Scalar std of VN entropy
                - vn_entropy_per_position: (batch_size, seq_len) or (batch_size, n_samples)
                - shannon_entropy_mean: For comparison
                - vn_to_shannon_ratio: Ratio of VN to Shannon entropy
        """
        batch_size, seq_len, vocab_size = probs.shape

        # Sample positions if specified (stratified sampling)
        if sample_positions is not None:
            # Index into specific positions
            # sample_positions: (batch_size, n_samples)
            probs_sampled = torch.gather(
                probs,
                dim=1,
                index=sample_positions.unsqueeze(-1).expand(-1, -1, vocab_size)
            )
            # (batch_size, n_samples, vocab_size)
        else:
            probs_sampled = probs

        # Compute density matrix components (low-rank)
        top_k_probs, selected_embeddings = self.compute_density_matrix(probs_sampled)

        # Compute VN entropy
        vn_entropy = self.compute_vn_entropy_lowrank(top_k_probs, selected_embeddings)
        # (batch_size, n_samples) or (batch_size, seq_len)

        # Compute Shannon entropy for comparison
        shannon_entropy = -torch.sum(probs_sampled * torch.log(probs_sampled + 1e-10), dim=-1)
        # (batch_size, n_samples) or (batch_size, seq_len)

        return {
            'vn_entropy_mean': vn_entropy.mean().item(),
            'vn_entropy_std': vn_entropy.std().item(),
            'vn_entropy_max': vn_entropy.max().item(),
            'vn_entropy_min': vn_entropy.min().item(),
            'vn_entropy_per_position': vn_entropy,
            'shannon_entropy_mean': shannon_entropy.mean().item(),
            'vn_to_shannon_ratio': (vn_entropy.mean() / (shannon_entropy.mean() + 1e-10)).item(),
        }

    def update_embeddings(self, new_embedding_matrix: torch.Tensor):
        """Update embedding matrix (call after actor training step)"""
        self.embeddings = F.normalize(new_embedding_matrix, dim=-1)
        if self.use_pca:
            self.embeddings_proj = self.embeddings @ self.pca_matrix
        else:
            self.embeddings_proj = self.embeddings

    def update_pca(self, new_embedding_matrix: torch.Tensor):
        """Recompute PCA projection (call every N steps)"""
        if self.use_pca:
            self.pca_matrix = self._compute_pca(new_embedding_matrix, self.pca_dim)
            self.embeddings = F.normalize(new_embedding_matrix, dim=-1)
            self.embeddings_proj = self.embeddings @ self.pca_matrix


def stratified_sample_positions(
    batch_size: int,
    seq_len: int,
    n_samples: int = 20,
    strategy: str = 'uniform',
) -> torch.Tensor:
    """
    Generate stratified position samples for VN entropy computation.

    Args:
        batch_size: Number of sequences
        seq_len: Length of each sequence
        n_samples: Number of positions to sample per sequence
        strategy: 'uniform', 'staged', or 'adaptive'

    Returns:
        sample_positions: (batch_size, n_samples) indices
    """
    if strategy == 'uniform':
        # Uniformly sample positions
        positions = torch.randint(0, seq_len, (batch_size, n_samples))

    elif strategy == 'staged':
        # Sample from early, middle, late stages
        n_per_stage = n_samples // 3
        early = torch.randint(0, seq_len // 4, (batch_size, n_per_stage))
        middle = torch.randint(seq_len // 3, 2 * seq_len // 3, (batch_size, n_per_stage))
        late = torch.randint(3 * seq_len // 4, seq_len, (batch_size, n_samples - 2 * n_per_stage))
        positions = torch.cat([early, middle, late], dim=1)

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    return positions
```

### Phase 2: Integration with DAPO Trainer (Week 2)

**File: `recipe/dapo-eca/dapo_ray_trainer.py`**

Add VN entropy computation to existing metric collection:

```python
# Add import at top
from recipe.dapo_eca.von_neumann_entropy import (
    VonNeumannEntropyCalculator,
    stratified_sample_positions
)

class RayDAPOTrainer(RayPPOTrainer):

    def __init__(self, config):
        super().__init__(config)

        # Initialize VN entropy calculator
        self.vn_entropy_calculator = None
        self.vn_entropy_config = config.get('vn_entropy', {})
        if self.vn_entropy_config.get('enable', False):
            # Will initialize after first embedding fetch
            self.vn_entropy_update_freq = self.vn_entropy_config.get('update_freq', 10)
            self.vn_entropy_pca_freq = self.vn_entropy_config.get('pca_freq', 10)
            self.vn_entropy_sample_positions = self.vn_entropy_config.get('sample_positions', 20)

    def _initialize_vn_entropy(self):
        """Fetch embeddings and initialize VN entropy calculator"""
        if self.vn_entropy_calculator is not None:
            return

        # Fetch embedding matrix from actor worker
        embedding_matrix = self.actor_rollout_wg.get_embeddings()

        self.vn_entropy_calculator = VonNeumannEntropyCalculator(
            embedding_matrix=embedding_matrix,
            top_k=self.vn_entropy_config.get('top_k', 50),
            use_pca=self.vn_entropy_config.get('use_pca', True),
            pca_dim=self.vn_entropy_config.get('pca_dim', 512),
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        print(f"[VN Entropy] Initialized with embedding matrix shape: {embedding_matrix.shape}")

    def compute_vn_entropy_metrics(self, batch: DataProto) -> dict:
        """Compute VN entropy metrics for the batch"""
        if not self.vn_entropy_config.get('enable', False):
            return {}

        # Initialize on first call
        if self.vn_entropy_calculator is None:
            self._initialize_vn_entropy()

        # Update embeddings periodically (after actor training)
        if self.global_steps % self.vn_entropy_update_freq == 0:
            embedding_matrix = self.actor_rollout_wg.get_embeddings()
            self.vn_entropy_calculator.update_embeddings(embedding_matrix)

        # Update PCA less frequently (more expensive)
        if self.global_steps % self.vn_entropy_pca_freq == 0:
            embedding_matrix = self.actor_rollout_wg.get_embeddings()
            self.vn_entropy_calculator.update_pca(embedding_matrix)

        # Get logits from batch (need to add this to rollout output)
        if 'logits' not in batch.batch:
            # If logits not available, skip this step
            return {}

        logits = batch.batch['logits']  # (batch_size, seq_len, vocab_size)
        probs = F.softmax(logits, dim=-1)

        # Stratified sampling of positions
        batch_size, seq_len, _ = logits.shape
        if self.vn_entropy_sample_positions > 0:
            sample_positions = stratified_sample_positions(
                batch_size,
                seq_len,
                n_samples=self.vn_entropy_sample_positions,
                strategy='staged',
            )
        else:
            sample_positions = None  # Compute for all positions

        # Compute VN entropy
        vn_metrics = self.vn_entropy_calculator.compute_batch_vn_entropy(
            probs,
            sample_positions=sample_positions
        )

        # Add prefix for logging
        return {
            f'vn_entropy/{k}': v
            for k, v in vn_metrics.items()
            if not k.endswith('_per_position')  # Don't log per-position tensors
        }

    def fit(self):
        # ... existing code ...

        # In the training loop, add VN entropy computation
        # After computing advantages, before training:

        with marked_timer("vn_entropy", timing_raw, "magenta"):
            vn_metrics = self.compute_vn_entropy_metrics(batch)
            metrics.update(vn_metrics)
```

### Phase 3: Configuration (Week 2)

**File: `recipe/dapo-eca/config/vn_entropy.yaml`**

```yaml
vn_entropy:
  # Enable von Neumann entropy computation
  enable: True

  # Top-k tokens to consider (reduce vocab size)
  top_k: 50  # 50-100 recommended

  # Use PCA dimension reduction
  use_pca: True
  pca_dim: 512  # Reduce from ~5120 to 512

  # Update frequency
  update_freq: 1    # Update embeddings every N steps (after training)
  pca_freq: 10      # Recompute PCA every N steps (expensive)

  # Position sampling (reduce sequence length cost)
  sample_positions: 20  # Number of positions to sample per sequence
                        # 0 = compute for all positions (expensive!)

  # Sampling strategy
  sampling_strategy: 'staged'  # 'uniform', 'staged', 'adaptive'
```

**File: Update `recipe/dapo-eca/run_dapo_qwen2.5_32b.sh`**

```bash
# Add to the python command
    +vn_entropy.enable=True \
    +vn_entropy.top_k=50 \
    +vn_entropy.use_pca=True \
    +vn_entropy.pca_dim=512 \
    +vn_entropy.sample_positions=20 \
    +vn_entropy.update_freq=1 \
    +vn_entropy.pca_freq=10 \
```

---

## Performance Estimates

### Computational Cost Per Batch

**Configuration:**
- Model: Qwen2.5-32B (vocab=152k, d=5120)
- Batch: 512 sequences × 2048 tokens = 1,048,576 positions
- Optimizations: top-k=50, PCA→512, sample 20 positions/sequence

**Breakdown:**

1. **Embedding fetch** (once per RL step):
   - Transfer 1.56 GB from GPU to CPU
   - Time: ~50-100ms (PCIe bandwidth)
   - **Amortized over batch: ~1ms**

2. **PCA computation** (every 10 steps):
   - Randomized SVD on (152k, 5120) → 512 components
   - Time: ~10-20 seconds on GPU
   - **Amortized: ~1-2 seconds per step**

3. **Per-batch VN entropy** (sampled positions):
   - Positions computed: 512 seqs × 20 positions = 10,240 positions
   - Per position:
     - Top-k selection: ~0.1ms (sort 152k probs)
     - Gram matrix: 50×50 = 2,500 elements, ~0.5ms
     - Eigendecomposition (50×50): ~1ms
     - **Total per position: ~2ms**
   - **Total for batch: 10,240 × 2ms = 20 seconds**

4. **With GPU parallelization:**
   - Batch eigendecomposition (cuSOLVER batched)
   - Expected: **2-5 seconds per batch**

### Compared to Training

Typical DAPO training time per iteration:
- Rollout: 30-60 seconds
- Advantage computation: 1-2 seconds
- Actor training: 20-40 seconds
- **Total: 60-120 seconds**

**VN entropy overhead: 2-5 seconds = 2-5% overhead** ✅ Acceptable!

### Memory Cost

- Embedding matrix (cached on CPU): 1.56 GB
- PCA matrix (512 × 5120 × 4 bytes): 10 MB
- Per-batch Gram matrices (10k × 50 × 50 × 4 bytes): 100 MB
- **Total additional memory: ~1.7 GB** (mostly CPU)

---

## Recommended Approach

### Final Configuration

Based on the analysis, I recommend:

```yaml
vn_entropy:
  enable: True
  top_k: 50              # Excellent reduction, negligible error
  use_pca: True          # 10x dimension reduction
  pca_dim: 512           # Preserves 95%+ variance
  update_freq: 1         # Update after each training step
  pca_freq: 10           # Recompute PCA every 10 steps
  sample_positions: 20   # Stratified sampling (100x reduction)
  sampling_strategy: 'staged'  # Sample early, middle, late tokens
```

### Why This Works

1. **Top-k=50**: Captures 99.9%+ of probability mass
2. **PCA dim=512**: Reduces d=5120 to 512 (10x speedup)
3. **Sample 20 positions**: Stratified sampling gives representative measure
4. **Low-rank eigen**: Exploits rank-50 structure (1000x speedup over naive)

**Net speedup: ~10,000x compared to naive implementation**
- Naive: ~5,500 seconds per batch
- Optimized: ~2-5 seconds per batch
- **From infeasible to 2-5% overhead** ✅

### Alternative: Even Faster (if needed)

If 2-5 seconds is still too much:

```yaml
vn_entropy:
  top_k: 30              # Reduce to 30 tokens
  pca_dim: 256           # Reduce to 256 dims
  sample_positions: 10   # Sample only 10 positions
  update_freq: 5         # Update every 5 steps
  pca_freq: 20           # PCA every 20 steps
```

**Expected: ~0.5-1 second per batch** (0.5% overhead)

---

## Next Steps

### Implementation Checklist

- [ ] Implement `von_neumann_entropy.py` with `VonNeumannEntropyCalculator`
- [ ] Add `get_embeddings()` method to actor worker
- [ ] Integrate VN entropy computation into `dapo_ray_trainer.py`
- [ ] Add configuration file and update training scripts
- [ ] Test on small model (Qwen2.5-7B) first
- [ ] Profile and optimize GPU kernel usage
- [ ] Add visualization of VN entropy over training
- [ ] Compare VN entropy vs Shannon entropy correlation with performance

### Validation Experiments

1. **Correctness**: Verify VN entropy ≥ 0 and reasonable values
2. **Correlation**: Check VN entropy correlates with model uncertainty
3. **Performance**: Measure actual overhead on H800 cluster
4. **Ablation**: Compare top-k values (30, 50, 100)
5. **Interpretability**: Analyze high VN entropy positions (semantic diversity)

### Expected Insights

Von Neumann entropy should reveal:
- **Semantic coherence** of uncertainty (low VN ↔ semantically similar predictions)
- **Reasoning quality** (high VN in early reasoning, low VN in conclusions?)
- **Learning dynamics** (VN entropy decay during training?)
- **Failure modes** (high VN entropy when model struggles?)

This could guide:
- Reward shaping (penalize incoherent high uncertainty)
- Curriculum learning (start with semantically coherent tasks)
- Verification (low VN entropy → more reliable outputs)

---

## Conclusion

**Von Neumann entropy is computationally feasible** with aggressive optimizations:

✅ **Your ideas (top-p, PCA)** are excellent starting points
✅ **Low-rank eigendecomposition** is the critical optimization
✅ **Stratified sampling** reduces cost by 100x
✅ **Final overhead: 2-5% of training time**

The implementation is straightforward using PyTorch's linear algebra functions, and should integrate cleanly into the existing DAPO trainer infrastructure.

**Recommendation: Proceed with implementation using the recommended configuration above.**
