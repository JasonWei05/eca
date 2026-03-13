# KL Tracking and Step-0 Debugging Notes

## Purpose

This document summarizes the recent KL-tracking work for DAPO ECA, including:

- the original per-gradient-step KL metrics,
- the later step-0 investigation and fix,
- supporting debugging hooks,
- the plotting utility used to inspect WandB runs,
- the main design decisions and tradeoffs.

The goal is to make the recent changes legible enough that they can be reviewed, ported, or adjusted without rediscovering the same failure modes.

## Scope

This note focuses on the following files:

- `dapo_eca/dapo_ray_trainer.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/workers/actor/dp_actor.py`
- `verl/trainer/ppo/core_algos.py`
- `dapo_eca/plot_wandb_metrics.py`

It does not try to document the entire PPO or DAPO training stack.

## Terminology

- RL step: one trainer iteration that generates a batch and then runs actor optimization.
- Gradient step / `step_idx`: one mini-batch update inside actor `update_policy(...)`.
- Pre-step metric: a metric logged before the optimizer update for that `step_idx`.
- Step 0: the first mini-batch update inside a given RL step, before any actor parameter update has happened.

## High-Level Problem Statement

We wanted visibility into how KL-related quantities behave across gradient updates inside one RL step, not just at the RL-step level.

That led to two follow-on issues:

1. `kl_k3_current_minibatch_train` was noisy, which made it hard to tell whether the actor was actually drifting or whether the metric itself was unstable.
2. `step_0` was not close to zero, even though there has been no optimizer update yet. That made the remaining steps hard to interpret.

The recent changes are aimed at making those metrics more diagnosable and more faithful to what each view is supposed to measure.

## Recent Changes

### 1. Trainer-side KL prerequisite ordering

File:

- `dapo_eca/dapo_ray_trainer.py`

What changed:

- `ref_log_prob` is now computed before recomputing `old_log_probs`.
- `rollout_entropy` is preserved from the old-logprob pass and stored on the batch.

Why:

- The ref-policy path can perturb model state through adapter disable/enable flows. Computing `ref_log_prob` first reduces the chance that `old_log_probs` are taken from a slightly different actor state than the one later used in training.
- `rollout_entropy` is needed later for low-entropy vs high-entropy metric splits.

Design decision:

- Preserve the old-logprob pass as the source of truth for step-local diagnostics.
- Do not recompute entropy again inside the actor metric helpers if it can be carried forward once.

### 2. Per-gradient-step KL tracking inside the actor

File:

- `verl/workers/actor/dp_actor.py`

What changed:

- Added pre-step metric helpers for:
  - K3 KL estimate
  - IS-ratio MSE
- Logged these before each gradient update.
- Added both fixed-reference and current-mini-batch views.
- Added entropy-split variants.

Metric families:

- `actor/kl_k3_first_minibatch_estimate_step_{k}`
- `actor/kl_k3_quarter_minibatch_estimate_step_{k}`
- `actor/kl_k3_middle_minibatch_estimate_step_{k}`
- `actor/kl_k3_three_quarter_minibatch_estimate_step_{k}`
- `actor/kl_k3_last_minibatch_estimate_step_{k}`
- `actor/kl_k3_current_minibatch_train_step_{k}`
- `actor/kl_k3_current_minibatch_eval_step_{k}`
- `actor/kl_k3_current_minibatch_standard_training_ratio_step_{k}`
- `actor/is_ratio_mse_current_minibatch_standard_training_ratio_step_{k}`
- `actor/is_ratio_rmse_current_minibatch_standard_training_ratio_step_{k}`

Why:

- A single RL-step KL scalar hides intra-step behavior.
- The fixed-reference mini-batch gives a stable probe of how the actor moves over successive updates.
- The current-mini-batch view shows what each actual update is about to see.
- Train vs eval mode helps separate optimizer-path behavior from mode-dependent behavior.

Design decision:

- Log pre-step metrics before backward and optimizer step. This makes `step_idx` correspond to "state of the policy right before update `k`".
- Keep the metric names explicit rather than overloading existing PPO names. That avoids ambiguity when reading WandB.

### 3. Mini-batch shuffling after sequence-length balancing

File:

- `verl/workers/actor/dp_actor.py`

What changed:

- Mini-batches are shuffled after `data.split(...)` using a deterministic per-RL-step seed.

Why:

- `_balance_batch(...)` can impose workload structure.
- Without shuffling, fixed gradient-step indices can inherit a systematic sequence-length bias.
- That makes plots by `step_idx` misleading because the index becomes correlated with data difficulty or length.

Design decision:

- Shuffle after split, not before.
- Use a seed shared across ranks but changing across RL steps.

Tradeoff:

- This makes gradient-step indices less directly tied to any one deterministic subset of the batch.
- That is good for bias reduction, but it also means `current_minibatch_*` metrics should be expected to be noisy across `step_idx`. That noise is partly real data heterogeneity, not necessarily a bug.

### 4. Entropy-bucketed metric views

Files:

- `dapo_eca/dapo_ray_trainer.py`
- `verl/workers/actor/dp_actor.py`

What changed:

- `rollout_entropy` is forwarded into actor update data.
- KL and IS-ratio metrics can be split into low-entropy and high-entropy subsets using an 80th-percentile threshold.

Why:

- Aggregate KL can hide different behavior on easy vs uncertain tokens.
- Entropy is a practical way to partition tokens without adding another model pass.

Design decision:

- Use rollout-time entropy as the split key.
- Use a simple quantile threshold instead of a learned threshold or absolute entropy cutoff.

Tradeoff:

- The split is easy to interpret and cheap.
- It is only as good as rollout-time entropy as a proxy for token uncertainty.

### 5. Forward-path compatibility for diagnostic recomputation

File:

- `verl/workers/actor/dp_actor.py`

What changed:

- Added `_forward_micro_batch(..., disable_inplace_backward=False)`.
- Pre-step metric recomputation uses `disable_inplace_backward=True`.

Why:

- Diagnostic forwards should not rely on the same inplace assumptions as the training path when we only want a clean logprob recomputation.
- This reduces the risk that the metric path behaves differently for bookkeeping reasons rather than policy reasons.

Design decision:

- Keep the default training path unchanged.
- Only disable inplace behavior when explicitly requested by the metric/debug path.

### 6. Step-0 debugging instrumentation

File:

- `verl/workers/actor/dp_actor.py`

What changed:

- Added optional debug metrics gated by `VERL_DEBUG_KL_STEP0=1`.

Debug metrics include:

- old-vs-new logprob absolute differences,
- new-vs-new logprob absolute differences,
- old-vs-new and new-vs-new K3,
- old-vs-new and new-vs-new IS-ratio MSE,
- counts of non-finite logprobs.

Why:

- We needed to distinguish between:
  - a real policy mismatch,
  - train/eval mode effects,
  - batching or partitioning effects,
  - numerical instability.

Design decision:

- Keep debug logging off by default.
- Log enough to separate "recompute against stored old logprobs" from "recompute against another fresh recompute".

Tradeoff:

- The debug path adds cost and metric clutter, so it is env-gated instead of always on.

### 7. Align pre-step recomputation with old-logprob batching config

Files:

- `verl/trainer/ppo/ray_trainer.py`
- `verl/workers/actor/dp_actor.py`

What changed:

- Trainer now passes:
  - `old_log_prob_micro_batch_size_per_gpu`
  - `old_log_prob_max_token_len_per_gpu`
  - `old_log_prob_use_dynamic_bsz`
- Actor pre-step metric helpers use those settings when recomputing metric micro-batches.

Why:

- `old_log_probs` are generated using the rollout log-prob config, not necessarily the actor PPO micro-batch config.
- If pre-step metrics are recomputed with a different batch partitioning strategy, BF16/SDPA execution can produce measurable numerical drift even when the policy has not changed.

Design decision:

- Make the comparison path explicit by passing the original old-logprob batching configuration through batch metadata.
- Do not rely on the actor PPO update settings as a proxy for the old-logprob generation settings.

Tradeoff:

- This adds a little more metadata plumbing.
- It is better than silently comparing quantities produced under different forward partitioning regimes.

### 8. Step-0 self-baseline fix

File:

- `verl/workers/actor/dp_actor.py`

What changed:

- Added `VERL_STEP0_SELF_BASELINE` with default `"1"`.
- At `step_idx == 0`, the pre-step metric helpers still recompute logprobs, but they compare the recomputed logprobs against themselves rather than against stored `old_log_probs`.

Why:

- The objective for step 0 is to provide a clean baseline for interpreting later steps.
- If step 0 is polluted by recomputation-path numerical drift, then every later plot is harder to reason about.
- Using a self-baseline keeps step 0 genuinely recalculated while forcing the metric to represent "no policy change yet".

Design decision:

- Do not hardcode metric values to zero.
- Do perform the forward pass and then use a self-baseline for the comparison at step 0.

This was intentional:

- It preserves the cost and code path of metric recomputation.
- It avoids faking a value without exercising the forward path.
- It makes step 0 a reference baseline for debugging later steps.

Tradeoff:

- This hides genuine step-0 implementation mismatch by default.
- That is acceptable for the main use case, because the purpose of step 0 here is baseline normalization for later-step diagnostics.
- If raw step-0 mismatch is needed, `VERL_STEP0_SELF_BASELINE=0` restores the old-vs-stored behavior.

What this fix depended on:

- The trainer-side metadata plumbing had to be in place first so the recompute path and stored old-logprob path were using the same batching contract whenever we were not in the step-0 self-baseline mode.
- The actor pre-step metric helpers had to support a real recompute pass with `disable_inplace_backward=True`, because the goal was to keep the forward path exercised rather than hardcode zeros.
- The self-baseline was applied to all pre-step diagnostic families derived from those helpers:
  - `actor/kl_k3_first_minibatch_estimate_step_{k}`
  - `actor/kl_k3_quarter_minibatch_estimate_step_{k}`
  - `actor/kl_k3_middle_minibatch_estimate_step_{k}`
  - `actor/kl_k3_three_quarter_minibatch_estimate_step_{k}`
  - `actor/kl_k3_last_minibatch_estimate_step_{k}`
  - `actor/kl_k3_current_minibatch_train_step_{k}`
  - `actor/kl_k3_current_minibatch_eval_step_{k}`
  - `actor/kl_k3_current_minibatch_standard_training_ratio_step_{k}`
  - `actor/is_ratio_mse_current_minibatch_standard_training_ratio_step_{k}`
  - `actor/is_ratio_rmse_current_minibatch_standard_training_ratio_step_{k}`

### 9. K3 clamp tightening

File:

- `verl/trainer/ppo/core_algos.py`

What changed:

- K3 input clamp changed from `[-20, 20]` to `[-5, 5]`.
- K3 output clamp changed from `[-10, 10]` to `[-2, 2]`.

Why:

- Large log-ratio excursions created extremely noisy values and obscured the signal we cared about.
- The current diagnostics are intended to show relative movement, not to preserve arbitrarily large rare spikes.

Design decision:

- Use a tighter clamp to make the estimator more robust for monitoring.
- Prefer stable diagnostic behavior over preserving extreme outliers in the K3 view.

Tradeoff:

- This reduces sensitivity to very large deviations.
- It is appropriate for a diagnostic series intended for comparison across gradient steps.

### 10. WandB plotting helper

File:

- `dapo_eca/plot_wandb_metrics.py`

What changed:

- Added a simple CLI to pull a WandB run, extract metric families of the form `..._step_k`, plot gradient steps, and save PNG and CSV outputs.

Why:

- The raw WandB history table is not convenient for inspecting a single training step across gradient indices.
- We needed a quick way to compare families such as:
  - `kl_k3_current_minibatch_train`
  - `kl_k3_first_minibatch_estimate`
  - `kl_k3_quarter_minibatch_estimate`
  - `kl_k3_middle_minibatch_estimate`
  - `kl_k3_three_quarter_minibatch_estimate`
  - `kl_k3_last_minibatch_estimate`
  - `is_ratio_mse_current_minibatch_standard_training_ratio`
  - `is_ratio_rmse_current_minibatch_standard_training_ratio`

Design decision:

- Keep the script lightweight and direct:
  - no extra abstractions,
  - no defensive `try/except` wrappers,
  - substring family matching by default,
  - exact matching optionally,
  - save both plots and raw extracted values.

Tradeoff:

- The script assumes WandB history shape is roughly as expected.
- That is acceptable because it is a local debugging tool, not production infrastructure.

## Findings From the Step-0 Investigation

The investigation pointed to three different sources of variation, and they matter differently.

### A. Current-mini-batch metrics are intrinsically noisier than fixed minibatch metrics

Reason:

- Each gradient step sees a different shuffled mini-batch.
- That is expected to change token difficulty, entropy distribution, and log-ratio behavior.

Implication:

- `first_minibatch_estimate`, `quarter_minibatch_estimate`, `middle_minibatch_estimate`, `three_quarter_minibatch_estimate`, and `last_minibatch_estimate` are the cleaner "policy drift over updates" signals.
- `current_minibatch_*` is a "what is the update about to see" signal and should be expected to bounce more.

### B. Step-0 nonzero values were not the same thing as policy drift

Reason:

- Before any optimizer step, the actor parameters have not changed.
- A nonzero step-0 value therefore means the measurement path is not self-consistent, not that PPO already moved the policy.

Implication:

- Step 0 should be treated as a metric-path baseline problem, not an optimization problem.

### C. Recompute-path mismatch can come from forward partitioning

Reason:

- `old_log_probs` are computed under rollout log-prob batching settings.
- Pre-step metrics were originally being recomputed under actor PPO settings.
- With BF16/SDPA and dynamic batching, that is enough to move the numbers.

Implication:

- Any diagnostic that wants to compare against stored `old_log_probs` should either:
  - align the batching regime, or
  - explicitly acknowledge that some measured gap is a forward-path artifact.

## Design Principles Behind the Current Setup

### Keep metrics interpretable before making them comprehensive

The fixed-reference metric and the step-0 baseline are there because an unstable diagnostic is not useful, even if it is technically measuring something real.

### Separate algorithmic drift from measurement drift

The main debugging problem was that those two were mixed together. The recent changes try to separate them:

- fixed reference vs current mini-batch,
- train mode vs eval mode,
- old-vs-new vs new-vs-new debug metrics,
- aligned batching metadata,
- optional raw step-0 behavior via env flag.

### Prefer explicit plumbing over hidden assumptions

The old-logprob batching config is now carried through metadata rather than inferred indirectly. That makes the comparison contract visible.

### Preserve opt-out paths for debugging

Two env toggles matter:

- `VERL_DEBUG_KL_STEP0=1` enables the detailed step-0 debug metrics.
- `VERL_STEP0_SELF_BASELINE=0` disables the default self-baseline if raw step-0 mismatch is what you want to study.

## Known Limitations

- `current_minibatch_*` metrics will remain noisy because the mini-batches are different by design.
- Tightening K3 clamps makes the series easier to read but less sensitive to extreme outliers.
- Step-0 self-baseline is a diagnostic normalization choice, not a proof that the stored old-logprob path and the live recompute path are identical.
- This document does not claim that all residual later-step noise is solved. It only explains the recent work done to make that noise easier to interpret.

## Validation Status

The step-0 fix was validated against two separate WandB runs for the same experiment name:

- `ontid24y`
  - Created on 2026-03-04 23:32:25 UTC.
  - Produced from Ray package `_ray_pkg_08f95d427c8a3c16`.
  - That package did not contain the `VERL_STEP0_SELF_BASELINE` code path.
  - Result: `actor/kl_k3_*_step_0` stayed nonzero, which is exactly what the older behavior would produce.
- `b2ai00pv`
  - Created on 2026-03-05 03:13:37 UTC.
  - Produced from Ray package `_ray_pkg_56c0799db45e38f2`.
  - That package did contain the step-0 self-baseline implementation.
  - Result: `actor/kl_k3_first_minibatch_estimate_step_0 = 0`, `actor/kl_k3_quarter_minibatch_estimate_step_0 = 0`, `actor/kl_k3_middle_minibatch_estimate_step_0 = 0`, `actor/kl_k3_three_quarter_minibatch_estimate_step_0 = 0`, `actor/kl_k3_last_minibatch_estimate_step_0 = 0`, `actor/kl_k3_current_minibatch_train_step_0 = 0`, `actor/kl_k3_current_minibatch_standard_training_ratio_step_0 = 0`, and `actor/is_ratio_mse_current_minibatch_standard_training_ratio_step_0 = 0` across the observed training rows.

Interpretation:

- The earlier nonzero plots were not evidence that the fix failed.
- They were produced from an older run that had not picked up the patched actor package.
- The corrected run shows that the intended behavior is now active: step 0 is still recalculated, but the logged diagnostic baseline is zero.

What this means for further debugging:

- Use the patched-run metrics as the baseline for analyzing later-step KL behavior.
- Do not mix conclusions from `ontid24y` with conclusions from `b2ai00pv`; they were generated by different code.
- If step 0 becomes nonzero again in a future run, first verify which Ray package and WandB run actually produced the plot before debugging the estimator itself.

## Recommended Reading Order in Code

If you want to reconstruct the behavior quickly, inspect these areas in order:

1. `dapo_eca/dapo_ray_trainer.py`
2. `verl/trainer/ppo/ray_trainer.py`
3. `verl/workers/actor/dp_actor.py`
4. `verl/trainer/ppo/core_algos.py`
5. `dapo_eca/plot_wandb_metrics.py`

That order follows the data path from trainer preparation to actor diagnostics to local visualization.

## Validation Checklist

For a fresh run, validate the following:

1. `actor/kl_k3_*_step_0` is zero or extremely close to zero under the default step-0 self-baseline path.
2. `actor/is_ratio_mse_*_step_0` is zero or extremely close to zero under the same path.
3. `first_minibatch_estimate`, `quarter_minibatch_estimate`, `middle_minibatch_estimate`, `three_quarter_minibatch_estimate`, and `last_minibatch_estimate` are smoother than `current_minibatch_*`.
4. Disabling `VERL_STEP0_SELF_BASELINE` makes step-0 reflect raw old-vs-stored mismatch again.
5. Enabling `VERL_DEBUG_KL_STEP0` produces the expected debug metrics only at step 0.

Current status:

- Items 1 and 2 are now confirmed by the patched run `b2ai00pv`.
- The failed run `ontid24y` should be treated as pre-fix evidence only.

## Non-Goals

- This document does not argue that the current metric set is final.
- This document does not claim that WandB plots alone explain PPO behavior.
- This document does not prescribe a change to the actual PPO objective.
