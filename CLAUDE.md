# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **research fork of verl** (Volcano Engine Reinforcement Learning for LLMs) focused on **ECA (Entropy-Controlled Advantage)** — adaptive token-level advantage reweighting for GRPO training. The base verl library implements the HybridFlow architecture (EuroSys 2025) for distributed RL training of LLMs.

The core research contribution lives in `dapo_eca/` and modifies the DAPO (Dynamic Advantage Policy Optimization) training loop to reweight per-token advantages based on entropy/gradient information.

## ECA Algorithm

Three mutually exclusive ECA modes (set in `algorithm.*` config):

- **`eca_linear`**: Multiplies each token's advantage by `grad_sq / sum_G` where `grad_sq = 1 - Σπ²`. Best for on-policy with few inner steps.
- **`eca_softmax`**: Defers reweighting to actor training loop, computes `weights = softmax(gamma * grad_sq)`. Supports `scheduled_eca` list for step-dependent gamma.
- **`eca_on_policy`** (primary): Token weight `w_t = 1 / (f_t + c)` normalized per-sequence, where `f_t = 1 - Σπ²` (Fisher trace) and `c = Var(A_seq) / B_prompts` (noise floor). Upweights low-entropy tokens that get tiny gradients from softmax geometry.

Key config parameters:
```
algorithm.eca_on_policy=True
algorithm.eca_on_policy_c_min=1e-8
actor_rollout_ref.actor.calculate_sum_pi_squared=True
actor_rollout_ref.actor.entropy_checkpointing=True
```

Theory documentation: `eca_docs/IMPLEMENTATION_DETAILS.md`

## Key Files (ECA-specific)

| File | Purpose |
|------|---------|
| `dapo_eca/main_dapo.py` | Entry point: `python -m dapo_eca.main_dapo` |
| `dapo_eca/dapo_ray_trainer.py` | `RayDAPOTrainer` — extends `RayPPOTrainer` with ECA modes, Rényi-2 entropy, grad_sq computation |
| `dapo_eca/eca_qwen3_4b_base.sh` | Reference training script (single node, 8 GPUs) |
| `dapo_eca/eca_qwen3_4b_base_multinode.sh` | Multi-node variant |
| `dapo_eca/config/dapo_trainer.yaml` | Hydra config for DAPO/ECA training |
| `dapo_eca/runtime_env.yaml` | Ray runtime env (includes LD_PRELOAD for GLIBC shim) |
| `verl/workers/actor/dp_actor.py` | Actor with ECA softmax reweighting in `update_policy()` (~line 761) |
| `verl/trainer/config/algorithm.py` | ECA config fields (~line 606) |
| `verl/utils/torch_functional.py` | `calculate_sum_pi_squared_from_logits()`, `entropy_from_logits()` |
| `verl/utils/vn_entropy.py` | Von Neumann entropy calculator (low-rank PCA optimization) |

## Running Training

```bash
# Single node (8 GPUs)
bash dapo_eca/eca_qwen3_4b_base.sh

# Multi-node
bash dapo_eca/eca_qwen3_4b_base_multinode.sh
```

The script starts Ray, then submits a job via `ray job submit` that runs `python3 -m dapo_eca.main_dapo` with Hydra config overrides. Key environment variables:
- `MODEL_PATH` — path to base model (default: `Qwen3-4B-Base`)
- `TRAIN_FILE` — training parquet file
- `CKPTS_DIR` — checkpoint output directory
- `NNODES` — number of nodes
- `RAY_DATA_HOME` — base directory for data files (default: `$HOME/verl`)

GLIBC compatibility: The system requires `LD_PRELOAD=/home/tiger/miniconda3/envs/eca/lib/glibc_compat.so` for flash_attn on GLIBC 2.28.

## Development Commands

### Linting & Formatting
```bash
pre-commit run --all-files        # All hooks
pre-commit run --all-files ruff   # Just ruff
```

Ruff config: line length 120, rules E/F/UP/B/I/G. Config in `pyproject.toml`.

mypy enforced on only 4 modules: `verl.trainer.config.algorithm`, `verl.trainer.ppo.core_algos`, `verl.trainer.ppo.reward`, `verl.workers.reward_manager`.

### Testing
```bash
pytest -s -x tests/path/to/test_file.py              # Single file
pytest -s -x tests/path/to/test_file.py::test_name   # Single test
pytest -s -x --asyncio-mode=auto tests/               # All CPU tests
```

## verl Base Architecture

### Hybrid-Controller Model

Single-controller + multi-worker architecture via Ray:

```
Controller (Driver Process)
  └── RayPPOTrainer / RayDAPOTrainer
      └── Ray Resource Pool
          ├── ActorRolloutRefWorker (generation + ref log probs)
          ├── TrainingWorker (actor update)
          ├── TrainingWorker (critic update)
          └── Reward computation
```

### Data Protocol

`DataProto` in `verl/protocol.py` — central data structure with `batch` (TensorDict), `non_tensor_batch` (dict), `meta_info` (dict). Supports `.chunk(n)`, `.concat(list)`, GPU/CPU transfers. `DataProtoFuture` wraps `list[ray.ObjectRef]` for async inter-worker data flow.

### Dispatch Modes

`@register(dispatch_mode=...)` in `verl/single_controller/base/decorator.py`:
- `ONE_TO_ALL` — broadcast, `DP_COMPUTE_PROTO` — data-parallel with DataProto, `RANK_ZERO` — only rank 0

### Training Loop (DAPO/ECA)

1. Sample prompts → 2. Rollout (vLLM generates sequences + log probs) → 3. Reward (DAPO reward manager with overlong buffer penalty) → 4. Compute `sum_pi_squared` and `grad_sq` from rollout logits → 5. Advantage computation (GRPO) → 6. ECA reweighting of advantages → 7. Actor training (PPO clipped loss with LoRA) → 8. Checkpoint & WandB logging

### Key verl Modules

- `verl/trainer/ppo/core_algos.py` — All RL algorithms via `@register_adv_est()` and `@register_policy_loss()` decorators
- `verl/trainer/ppo/ray_trainer.py` — Base training loop orchestrator
- `verl/workers/fsdp_workers.py` — Legacy FSDP workers (default)
- `verl/workers/engine_workers.py` — New unified workers (when `trainer.use_legacy_worker_impl="disable"`)
- `verl/trainer/config/` — Hydra config dataclasses (algorithm, actor, rollout, etc.)

### Configuration

Hydra-based. Override via CLI: `algorithm.eca_on_policy=True actor_rollout_ref.actor.strategy=fsdp2`. Config dataclasses in `verl/trainer/config/` and `verl/workers/config/`. Auto-generated YAML via `scripts/generate_trainer_config.sh` (pre-commit hook — CI fails if stale).

## Extension Points

- **New RL algorithm**: `@register_adv_est()` in `verl/trainer/ppo/core_algos.py`, update enum in `verl/trainer/config/algorithm.py`
- **New policy loss**: `@register_policy_loss()` in `core_algos.py`
- **New training backend**: Subclass `BaseEngine` in `verl/workers/engine/`
- **New rollout engine**: Subclass `BaseRollout` in `verl/workers/rollout/`

## ECA Metrics & Analysis

Training logs these ECA-specific metrics:
- `actor/renyi2_entropy` — Rényi-2 entropy: `-log(Σπ²)`
- `actor/vn_entropy` — Von Neumann entropy (if enabled)
- `train/acc/mean`, `train/acc/pass@n` — accuracy metrics
- `val-aux/{source}/acc/pass@k` — unbiased pass@k estimator: `1 - C(n-c,k)/C(n,k)`

Analysis scripts in project root:
- `entropy_heatmap.py` — visualize token-level entropy as colored boxes
- `entropy_ordering.py` — test rank ordering across entropy measures
- `aime_analysis.py` — analyze AIME benchmark outputs
- `benchmarking.py` / `benchmarking_gemini.py` / `benchmarking_claud.py` — benchmark comparisons

## Documentation

- `eca_docs/IMPLEMENTATION_DETAILS.md` — Full ECA theory (adaptive token weighting, Wiener filter justification)
- `dapo_eca/VON_NEUMANN_ENTROPY_IMPLEMENTATION.md` — VN entropy implementation (low-rank eigenvalue trick, PCA)
- `dapo_eca/LOW_RANK_MATH_DETAILS.md` — Mathematical foundations for low-rank optimization
- `dapo_eca/README.md` — DAPO reproduction results (52% on AIME 2024 with Qwen2.5-32B)
- `dapo_eca/CHANGES.md` — Changelog of all ECA-specific modifications
