# Triple-Baseline KL Divergence & IS-Ratio Metrics

## Problem

During PPO training with 16 gradient steps per batch, we track KL divergence between the current policy and its state before training to measure off-policyness. The original implementation used a single baseline: `old_log_probs` computed via a forward pass through the FSDP actor model in **eval mode** (via `compute_log_prob()` in `ray_trainer.py`).

We observed non-zero KL divergence at step 0 (before any gradient update), even though the weights are identical. This happens because the per-step metric forward pass runs in **train mode**, while the baseline was computed in **eval mode**. With `attn_implementation=sdpa`, PyTorch's SDPA dispatches to different attention kernels depending on the model mode (e.g., Flash Attention vs Memory-Efficient Attention), producing slightly different floating-point results from the same weights.

To properly diagnose and separate the sources of discrepancy, we now compare against **three baselines**.

## Design

The current policy (after k gradient steps) is always evaluated in **train mode**, since that's the mode used during actual gradient updates. We compare it against three different snapshots of the policy *before* training:

| Baseline | Key in batch | Source | What it isolates |
|----------|-------------|--------|-----------------|
| **eval_old_lp** | `old_log_probs` | FSDP actor model, eval mode, computed once before training | Standard PPO baseline (existing behavior) |
| **train_old_lp** | `old_log_probs_train` | FSDP actor model, train mode, computed once before training | Isolates SDPA eval/train kernel difference |
| **vllm_rollout** | `rollout_log_probs` | vLLM inference engine, computed during generation | Isolates FSDP/vLLM engine gap |

### Expected step-0 values

- `vs_train_old_lp` at step 0: **~0** (same mode, same weights, same data = sanity check)
- `vs_eval_old_lp` at step 0: **small but non-zero** (SDPA kernel difference only)
- `vs_vllm_rollout` at step 0: **non-zero** (full train/inference engine gap)

After gradient updates, all three diverge further as the policy drifts.

## Metric Naming Convention

```
actor/{metric_type}_{scope}_vs_{baseline}_step_{step_idx}
```

Where:
- `metric_type`: `kl_k3`, `is_ratio_mse`, `is_ratio_rmse`
- `scope`: `current_mb` (current minibatch) or `{probe_name}_mb` (fixed probe: first, quarter, middle, three_quarter, last)
- `baseline`: `eval_old_lp`, `train_old_lp`, `vllm_rollout`
- `step_idx`: 0-15 (gradient step index within the batch)

With entropy splits: `..._low_entropy_step_{idx}`, `..._high_entropy_step_{idx}`

### Example metric names

```
actor/kl_k3_current_mb_vs_eval_old_lp_step_0
actor/kl_k3_current_mb_vs_train_old_lp_step_0
actor/kl_k3_current_mb_vs_vllm_rollout_step_0
actor/kl_k3_middle_mb_vs_eval_old_lp_step_5
actor/is_ratio_rmse_current_mb_vs_vllm_rollout_step_3
actor/kl_k3_current_mb_vs_eval_old_lp_low_entropy_step_10
```

## Implementation Details

### Files modified

#### `verl/workers/actor/dp_actor.py`

**Change 1: Compute train-mode baseline before mini-batch split**

Added a full-shard train-mode forward pass before `data.split()` to compute `old_log_probs_train`. This runs once per training batch (not per gradient step), using the same weights as `old_log_probs` but with `self.actor_module.train()` instead of `.eval()`. The result is injected into `data.batch["old_log_probs_train"]` before the split, so each mini-batch automatically gets its correct slice.

**Change 2: Refactored metric functions into forward-pass + computation layers**

The old functions `_compute_prestep_k3_metrics()` and `_compute_prestep_is_ratio_mse_metrics()` each did their own forward pass AND metric computation. This was wasteful when comparing the same forward pass against multiple baselines. Replaced with:

- `_run_metric_forward_pass(target_mini_batch, mode)` — runs one forward pass, returns `list[(log_prob, inputs_dict)]`
- `_kl_k3_from_forward(forward_results, baseline_key, entropy_threshold)` — pure KL computation against a named baseline
- `_is_ratio_mse_from_forward(forward_results, baseline_key, entropy_threshold)` — pure IS-ratio MSE computation

**Change 3: Updated call site — train-mode only, loop over 3 baselines**

All metric forward passes use train mode (matching the actual gradient update). Each forward pass result is compared against all 3 baselines via cheap arithmetic. This gives 6 forward passes per step (5 fixed probes + 1 current MB), each producing 3x baseline comparisons.

#### `dapo_eca/eca_test.sh`

Added `actor_rollout_ref.rollout.calculate_log_probs=True` to enable vLLM to return per-token log probabilities during training generation. This flows through `AgentLoopManager` → vLLM's `SamplingParams(logprobs=True)` → `rollout_log_probs` field in the batch.

### Performance impact

- **One new full-shard train-mode forward pass** before the mini-batch loop (same cost as the existing eval-mode `compute_log_prob` call)
- **No additional per-step forward passes** — metric forward passes are reused; comparing against 3 baselines instead of 1 is just cheap arithmetic
- **vLLM log probs** have a small cost during generation but no training-time cost
- **Forward passes per step**: 6 (5 fixed probes + 1 current MB, all train mode)

## Why This Matters

1. **Debugging step-0 KL**: If `vs_train_old_lp` is ~0 at step 0 but `vs_eval_old_lp` is not, the discrepancy is purely from SDPA kernel differences (not a bug).

2. **Measuring true off-policyness**: The `vs_vllm_rollout` metric shows the actual gap between the training policy and the inference policy that generated the data. This is the "real" off-policyness that PPO's clipped objective is trying to handle.

3. **Separating concerns**: By having three baselines, you can independently measure:
   - SDPA kernel effects (eval vs train baselines at step 0)
   - Training drift (any baseline across steps 0-15)
   - Train/inference engine gap (vLLM baseline vs FSDP baselines)
