# Adaptive Token Weighting for GRPO: Theory & Implementation Spec

## 1. The Problem

Standard GRPO applies sequence-level advantages uniformly to all tokens. This is suboptimal:

- **On-policy (few gradient steps per rollout):** Low-entropy tokens get tiny gradients due to softmax geometry, wasting the natural gradient's power. Upweighting them by inverse-entropy fixes this.
- **Off-policy (many gradient steps per rollout):** Low-entropy tokens contribute near-zero signal but substantial noise from the uniform sequence advantage. They become dead weight. Downweighting them by entropy fixes this.

Four recent papers confirm this empirically but use opposite heuristics:
- EMPG (arXiv:2509.09265), ResT (arXiv:2509.21826): inverse-entropy helps at k≤4
- Beyond 80/20 (arXiv:2506.01939), GTPO (arXiv:2508.04349): entropy-proportional helps at k≥16

**Our contribution:** A single adaptive weight formula that automatically transitions between these regimes based on measured off-policyness. No manual switching, one hyperparameter.

---

## 2. Key Definitions

**Two policies to keep straight:**

| Name | Symbol | What it is | When computed |
|------|--------|-----------|---------------|
| **Rollout policy** | μ = π^(0) | The policy that generated the data (inference) | Once, before inner loop. Frozen. |
| **Training policy** | π^(k) | The policy being optimized, at inner step k | Every inner step. Changes each step. |

**Per-token quantities (all computed under the training policy π^(k)):**

| Symbol | Definition | Meaning |
|--------|-----------|---------|
| f_t | 1 - Σ_j π^(k)_j(s_t)² | Fisher trace at token t. Measures entropy. f_t ≈ 0 means deterministic, f_t ≈ 1 means uniform. |
| ρ_t | π^(k)(a_t \| s_t) / μ(a_t \| s_t) | IS ratio. How much π^(k) has drifted from the rollout at this token. |
| A_seq | sequence-level advantage | Same scalar applied to every token in the sequence. |

---

## 3. Intuition

At a near-deterministic token (e.g., π("the") = 0.99, f_t ≈ 0.02):
- The **gradient signal** is tiny: the advantage zero-sum forces the greedy action's advantage to be Θ(ε), and the softmax gradient multiplies by π(a), giving signal ∝ ε².
- The **gradient noise** from the uniform sequence advantage decays slower: noise ∝ ε (from the score function variance).
- **Signal-to-noise ratio** ∝ ε²/ε = ε → 0. The token is drowned in noise.

At a high-entropy token (e.g., π("therefore") = 0.3, f_t ≈ 0.7):
- Signal is Θ(1), noise is Θ(σ²), SNR is Θ(1/σ²). Useful.

**On-policy**, this SNR problem is masked because the natural gradient preconditioner F⁻¹ amplifies low-entropy gradients by 1/ε, compensating exactly. But **off-policy**, F⁻¹ is unreliable (amplifies estimation error from stale data), so the raw SNR collapse dominates and low-entropy tokens must be suppressed.

---

## 4. The Formula

### Per-token weight at inner step k:

```
w_t = [1 / (f_t + λ_k)] × [f_t / (f_t + c)]
       \_________________/   \_______________/
        Geometry term          SNR filter
```

**Inputs (all computed at each inner step k):**
- **f_t = 1 - Σ_j π^(k)_j²** — Fisher trace under the **training policy** π^(k). NOT the rollout policy.
- **λ_k = γ √(mean((ρ_t - 1)²))** — scalar measuring how far π^(k) has drifted from rollout μ
- **c = Var(A_seq) / B** — noise floor: advantage variance ÷ number of sequences in batch
- **γ = 1.0** — the only hyperparameter (trust region scale)

Normalize w_t per-sequence to mean 1 (preserves effective learning rate).

### What each term does:

**Geometry term 1/(f_t + λ_k):**
- On-policy (λ_k ≈ 0): ≈ 1/f_t, inverse-entropy. Matches the natural gradient preconditioner.
- Off-policy (λ_k >> f_t): ≈ 1/λ_k, a flat constant. Geometry correction disabled.
- Low-entropy tokens (small f_t) lose their geometry correction first as λ_k grows.

**SNR filter f_t/(f_t + c):**
- High entropy (f_t >> c): ≈ 1. Full signal passes.
- Low entropy (f_t << c): ≈ f_t/c → 0. Token suppressed (SNR collapsed).
- Active in both regimes. On-policy, it caps the 1/f_t weight at 1/c = B/Var(A_seq).

---

## 5. Theoretical Justification

### The Wiener filter

Choose scalar w to minimize MSE against an ideal update d*:
```
w* = argmin_w E[||wĝ - d*||²]  =  ⟨E[ĝ], d*⟩ / (|E[ĝ]|² + Var(ĝ)/B)
```
This factors exactly as **w* = [Geometry Scaler] × [SNR Filter]**.

### Three scaling facts as token certainty ε = 1 - π(greedy) → 0:

**Fact 1 — Signal ∝ ε² (quadratic).** The policy gradient at token t has component g*_a = π(a)A*(a). By the advantage zero-sum (Σ_a π(a)A*(s_t, a) = 0, which sums over vocabulary actions a for a fixed token t — a definition property of any proper Q* - V*), the greedy advantage is A*(greedy) = Θ(ε). So |g*|² = Σ[π(a)A*(a)]² = Θ(ε²).

**Fact 2 — Noise ∝ ε (linear).** By EMPG Proposition 1: E[||∇log π||²] = 1 - exp(-H_2) ≈ 2ε. Multiplied by σ²_seq: Var(ĝ) = Θ(σ²_seq × ε).

**Fact 3 — Natural gradient target is Θ(1).** By Kakade (2001): F⁻¹g* = A* for softmax. So |F⁻¹g*|² = Σ A*(a)² = Θ(1), dominated by rare actions with Θ(1) advantages.

### On-policy result (Theorem 1)

Target d* = F⁻¹g* (natural gradient). Geometry scaler = ⟨g*, F⁻¹g*⟩/|g*|² = Var_π(A*)/|g*|² = Θ(ε)/Θ(ε²) = **Θ(1/ε)**. The weight is inverse-entropy. Including the SNR filter: **w = 1/(f_t + c)**.

### Off-policy result (Theorem 2)

Target d* = g* (Euclidean gradient — natural gradient is biased off-policy, Thomas 2014). Geometry scaler = 1. Weight is purely SNR filter: **w = f_t/(f_t + c)** ∝ entropy.

### The target transition

Tikhonov regularization (Martens 2020): d*(λ) = (F + λI)⁻¹g. For softmax, F has eigenvalues ~ε, so the preconditioner 1/(ε + λ) smoothly transitions from 1/ε (on-policy) to 1/λ (off-policy). λ_k is set by the Fisher estimation error, bounded by √(E[(ρ-1)²]).

---

## 6. Implementation

### Core function

```python
def compute_token_weights(
    logits_current,     # [batch, seq_len, vocab] from training policy π^(k)
    probs_old_chosen,   # [batch, seq_len] from rollout policy μ (cached, frozen)
    action_ids,         # [batch, seq_len] sampled token ids (from rollout)
    advantages,         # [batch] sequence-level advantages
    attention_mask,     # [batch, seq_len] padding mask
    gamma=1.0,          # trust region scale (only hyperparameter)
):
    """
    Adaptive per-token weights for GRPO.
    Automatically transitions from inverse-entropy (on-policy)
    to entropy-proportional (off-policy) based on measured IS drift.

    IMPORTANT: logits_current must be from the TRAINING policy π^(k),
    not the rollout policy. This changes at every inner gradient step.
    probs_old_chosen is from the ROLLOUT policy μ=π^(0), computed once and cached.
    """
    # --- Training policy distribution (recomputed every inner step) ---
    probs_current = torch.softmax(logits_current, dim=-1)

    # --- Fisher trace under TRAINING policy π^(k) ---
    f_t = 1.0 - (probs_current ** 2).sum(dim=-1)       # [batch, seq_len]
    f_t = f_t.clamp(min=1e-8)

    # --- IS ratio: how far π^(k) has drifted from rollout μ ---
    prob_current_chosen = probs_current.gather(
        -1, action_ids.unsqueeze(-1)
    ).squeeze(-1)
    rho_t = prob_current_chosen / probs_old_chosen.clamp(min=1e-8)

    # --- Lambda: off-policyness (one scalar for the whole batch) ---
    # Only computed over non-padding tokens
    valid_rho = (rho_t - 1.0) ** 2
    valid_rho = (valid_rho * attention_mask).sum() / attention_mask.sum()
    lambda_k = gamma * valid_rho.sqrt()

    # --- Noise floor: c = Var(A_seq) / B ---
    c = advantages.var() / advantages.shape[0]
    c = c.clamp(min=1e-8)

    # --- Unified weight ---
    geometry = 1.0 / (f_t + lambda_k)
    snr = f_t / (f_t + c)
    w_t = geometry * snr

    # --- Normalize per-sequence (only over non-padding tokens) ---
    seq_lengths = attention_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    w_mean = (w_t * attention_mask).sum(dim=-1, keepdim=True) / seq_lengths
    w_t = w_t / w_mean.clamp(min=1e-8)

    # Zero out padding
    w_t = w_t * attention_mask

    return w_t
```

### Integration into GRPO training loop

```python
for rl_step in range(num_rl_steps):

    # ========== ROLLOUT PHASE (data collection, π^(0)) ==========
    with torch.no_grad():
        sequences, action_ids, rewards = generate_rollouts(policy)  # π^(0)
        logits_old = policy(sequences)
        probs_old = torch.softmax(logits_old, dim=-1)
        # Cache rollout probs for chosen actions — this is FROZEN
        probs_old_chosen = probs_old.gather(
            -1, action_ids.unsqueeze(-1)
        ).squeeze(-1)                                               # [batch, seq_len]
        advantages = compute_advantages(rewards)                    # [batch]

    # ========== INNER LOOP (k gradient steps on same rollout) ==========
    for k in range(num_inner_steps):

        # Forward pass: TRAINING policy π^(k) — changes every step
        logits_current = policy(sequences)

        # Adaptive weights (uses π^(k) for f_t, π^(0) for ρ_t)
        w_t = compute_token_weights(
            logits_current=logits_current,       # from π^(k)
            probs_old_chosen=probs_old_chosen,   # from μ=π^(0), cached
            action_ids=action_ids,
            advantages=advantages,
            attention_mask=attention_mask,
            gamma=1.0,
        )

        # Standard GRPO clipped surrogate loss
        probs_current = torch.softmax(logits_current, dim=-1)
        prob_current_chosen = probs_current.gather(
            -1, action_ids.unsqueeze(-1)
        ).squeeze(-1)
        rho_t = prob_current_chosen / probs_old_chosen.clamp(min=1e-8)

        clipped_rho = torch.clamp(rho_t, 1.0 - eps_clip, 1.0 + eps_clip)
        adv = advantages.unsqueeze(-1)          # [batch, 1]
        surrogate = torch.min(rho_t * adv, clipped_rho * adv)

        # Apply per-token weights to loss
        loss = -(w_t * surrogate * attention_mask).sum() / attention_mask.sum()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Ablation baselines

```python
def compute_ablation_weights(f_t, lambda_k, c, attention_mask, mode="unified"):
    """
    Ablation weight modes. All normalized per-sequence to mean 1.

    Modes:
        "uniform"     — standard GRPO (no reweighting)
        "inverse"     — on-policy optimal: 1/(f_t + c)
        "entropy"     — off-policy optimal: f_t/(f_t + c)
        "binary_80_20" — Beyond 80/20 style: mask bottom 80% by entropy
        "unified"     — our method: adapts automatically
    """
    if mode == "uniform":
        w = torch.ones_like(f_t)
    elif mode == "inverse":
        w = 1.0 / (f_t + c)
    elif mode == "entropy":
        w = f_t / (f_t + c)
    elif mode == "binary_80_20":
        threshold = torch.quantile(f_t[attention_mask.bool()], 0.8)
        w = (f_t >= threshold).float().clamp(min=0.01)
    elif mode == "unified":
        w = (1.0 / (f_t + lambda_k)) * (f_t / (f_t + c))

    # Per-sequence normalization over non-padding tokens
    seq_lengths = attention_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    w_mean = (w * attention_mask).sum(dim=-1, keepdim=True) / seq_lengths
    w = w / w_mean.clamp(min=1e-8)
    w = w * attention_mask
    return w
```

### Diagnostic logging

```python
def log_diagnostics(f_t, rho_t, w_t, lambda_k, c, attention_mask, step_k):
    """Add to any run. Tracks theory predictions at each inner step."""
    mask = attention_mask.bool()
    f_valid = f_t[mask]
    w_valid = w_t[mask]
    median_f = f_valid.median()
    return {
        'inner_step': step_k,
        'lambda_k': lambda_k.item(),
        'noise_floor_c': c.item(),
        'crossover_epsilon': (lambda_k * c).sqrt().item(),
        'mean_fisher_trace': f_valid.mean().item(),
        'mean_rho_deviation': ((rho_t[mask] - 1.0)**2).mean().sqrt().item(),
        'weight_std': w_valid.std().item(),
        'frac_weight_below_half': (w_valid < 0.5).float().mean().item(),
        'mean_w_low_entropy': w_valid[f_valid < median_f].mean().item(),
        'mean_w_high_entropy': w_valid[f_valid >= median_f].mean().item(),
    }
```

---

## 7. Automatic Behavior at Each k

| Inner step k | λ_k | Geometry 1/(f_t + λ_k) | SNR f_t/(f_t + c) | Net weight |
|---|---|---|---|---|
| 0-2 | ≈ 0 | ≈ 1/f_t (inverse entropy) | caps extremes | **Inverse entropy, capped** |
| 4-8 | growing | flattening for low-f_t tokens | unchanged | **Transition** |
| 16-32 | large | ≈ 1/λ (constant) | dominates | **Entropy-proportional** |

No code changes between regimes. λ_k is measured from IS ratios at each step.

---

## 8. Experiments (Priority Order)

### Experiment 1: Validate the Flip (6 runs)

| Run | k | Weight | Expected result |
|-----|---|--------|----------------|
| 1 | 2 | uniform | Baseline |
| 2 | 2 | inverse | **Best at k=2** |
| 3 | 2 | entropy | Worst at k=2 |
| 4 | 32 | uniform | Baseline |
| 5 | 32 | inverse | Worst at k=32 |
| 6 | 32 | entropy | **Best at k=32** |

### Experiment 2: Test the Unified Weight (4 runs)

| Run | k | Weight | Expected result |
|-----|---|--------|----------------|
| 7 | 2 | unified | ≈ Run 2 (matches inverse) |
| 8 | 32 | unified | ≈ Run 6 (matches entropy) |
| 9 | 8 | unified | Better than both inverse and entropy |
| 10 | 8 | uniform | Baseline for k=8 |

### Experiment 3: Diagnostic (no training cost)

Add `log_diagnostics()` to any run. Verify:
- λ_k increases with k
- Signal² decays as ~f_t² per entropy bin
- Noise decays as ~f_t per entropy bin
- Weight ratio high/low entropy grows with k

---

## 9. Quick Reference Card

```
FORMULA:  w = (1 / (f_t + λ)) × (f_t / (f_t + c))    normalized per-sequence

f_t     = 1 - Σπ²          Fisher trace from training policy π^(k) (each inner step)
λ_k     = γ√(mean((ρ-1)²)) Off-policyness from IS ratios
c       = Var(A_seq) / B    Noise floor from batch statistics
γ       = 1.0               Only hyperparameter

ON-POLICY  (λ≈0):  w ≈ 1/(f_t + c)           Inverse entropy with cap
OFF-POLICY (λ>>f):  w ≈ f_t/((f_t + c) × λ)   Entropy-proportional
TRANSITION:         Automatic via λ_k growth
```
