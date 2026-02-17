# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**verl** (Volcano Engine Reinforcement Learning for LLMs) is a flexible, efficient, and production-ready RL training library for large language models. It implements the **HybridFlow** architecture (EuroSys 2025 paper) with a hybrid-controller programming model that enables:

- Multiple RL algorithms: PPO, GRPO, REINFORCE++, RLOO, REMAX, DAPO, etc.
- Training backends: FSDP, FSDP2, Megatron-LM, VEOmni, MindSpeed
- Rollout/inference engines: vLLM, SGLang, TRT-LLM, HF Transformers
- Model scales: 7B to 671B+ with expert parallelism
- Hardware: NVIDIA, AMD (ROCm), Ascend NPU

## Installation & Setup

**Basic installation:**
```bash
# Python-only development (quickest iteration)
pip install -e .[test,vllm]   # For vLLM backend
# OR
pip install -e .[test,sglang]  # For SGLang backend
```

**Full installation with dependencies:**
See https://verl.readthedocs.io/en/latest/start/install.html

**Important version notes:**
- Use vLLM >= 0.8.2 (avoid 0.7.x - has OOM bugs)
- FSDP2 is recommended over FSDP for better performance
- Recipe directory is a git submodule: `git submodule update --init --recursive recipe`

## Development Commands

### Code Linting & Formatting

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run on staged changes
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hooks
pre-commit run --all-files ruff
pre-commit run --all-files autogen-trainer-cfg
```

The pre-commit hooks include:
- `ruff` - Linting (rules: E, F, UP, B, I, G) and formatting. Line length 120. Config in `pyproject.toml`. Ignored rules: F405, F403, E731, B007, UP032, G004, UP045, UP035.
- `mypy` - Type checking. Only 4 modules enforced (`ignore_errors=false`): `verl.trainer.config.algorithm`, `verl.trainer.ppo.core_algos`, `verl.trainer.ppo.reward`, `verl.workers.reward_manager`. All other modules have `ignore_errors=true`.
- `autogen-trainer-cfg` - Generates `_generated_ppo_trainer.yaml` and `_generated_ppo_megatron_trainer.yaml` from Hydra configs via `scripts/generate_trainer_config.sh`. CI fails if generated files are stale.
- `check-docstrings` - Doc string coverage check
- `check-license` - License header verification
- `compileall` - Python compilation check

### Testing

**CPU unit tests:**
```bash
# All CPU tests (test files ending with *_on_cpu.py)
pytest -s -x --asyncio-mode=auto tests/
```

**Single test file or test function:**
```bash
pytest -s -x tests/path/to/test_file.py
pytest -s -x tests/path/to/test_file.py::test_name
```

**GPU unit tests:**
```bash
# Standard GPU unit tests
pytest -s -x --ignore-glob="*test_special_*.py" --ignore-glob='*on_cpu.py' \
  --ignore-glob="*test_vllm*" --ignore-glob="*_sglang*" \
  --ignore-glob="tests/models/" --ignore-glob='tests/special*' tests/

# Distributed tests (require multiple GPUs)
torchrun --standalone --nnodes=1 --nproc-per-node=8 tests/utils/test_special_linear_cross_entropy_tp.py
torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/workers/actor/test_special_dp_actor.py
```

**Test directory structure:**
- `tests/<module>/` - Unit tests for each verl module
- `tests/special_distributed/` - Multi-GPU tests
- `tests/special_e2e/` - End-to-end tests
- `tests/special_npu/` - NPU-specific tests
- `tests/special_sanity/` - Quick sanity checks
- `tests/special_standalone/` - Tests requiring dedicated environments
- Tests ending with `_on_cpu.py` run on CPU only

### Entry Points & Examples

**Training invocation:** `python -m verl.trainer.main_ppo` (Hydra-configured)

**Runtime call chain:**
```
main_ppo.py:main()  →  run_ppo(config)  →  TaskRunner.run()
    ├── create_rl_dataset() + create_rl_sampler()
    ├── RayPPOTrainer.__init__()
    ├── RayPPOTrainer.init_workers()
    └── RayPPOTrainer.fit()   # main training loop
```

Other entry points: `main_eval.py` (offline evaluation), `main_generation.py` (batch generation), `main_generation_server.py` (generation server).

**Shell script examples:**
```bash
bash examples/ppo_trainer/run_deepseek7b_llm.sh
bash examples/grpo_trainer/run_qwen3-8b.sh
```

## Code Architecture

### Core Architectural Pattern: Hybrid-Controller Model

verl uses a **single-controller + multi-worker** architecture orchestrated by Ray:

```
Controller (Driver Process)
  └── RayPPOTrainer - orchestrates training loop
      └── Ray Resource Pool - manages GPU allocation
          ├── ActorRolloutRefWorker (generation phase)
          │   ├── Actor model (policy)
          │   ├── Reference policy (KL constraint)
          │   └── Rollout engine (vLLM/SGLang)
          ├── TrainingWorker (actor training phase)
          ├── TrainingWorker (critic training phase)
          └── Optional: Reward computation worker
```

**Key Insight:** The controller coordinates workers via Ray, but data distribution is explicit through `@register` decorators that specify dispatch modes.

### Data Protocol: DataProto

Central data structure in `verl/protocol.py`:

```python
@dataclass
class DataProto:
    batch: TensorDict              # Tensors with same batch dimension
    non_tensor_batch: dict         # Non-tensor metadata (UIDs, strings)
    meta_info: dict                # Metadata about the batch
```

- TensorDict allows treating dict of tensors as a single tensor for vectorized ops
- DataProto handles batching, chunking, concatenation, GPU/CPU transfers
- Methods: `.chunk(n)`, `.concat(list)`, serialization for Ray transport
- **DataProtoFuture**: Wraps `list[ray.ObjectRef]` to enable async data flow between workers without resolving futures on the driver. Supports `.get()` to materialize.

### Dispatch Modes (Data Distribution)

The `@register` decorator in `verl/single_controller/base/decorator.py` controls data distribution:

- `ONE_TO_ALL` - Broadcast to all workers (config, resets)
- `ALL_TO_ALL` - Each worker processes independently (parallel batches)
- `DP_COMPUTE` - Distribute across data-parallel ranks (training)
- `DP_COMPUTE_PROTO` - Like DP_COMPUTE for DataProto objects (rollouts)
- `DP_COMPUTE_PROTO_WITH_FUNC` - DP_COMPUTE_PROTO with custom dispatch function
- `DP_COMPUTE_METRIC` - Distribute and aggregate metrics across DP ranks
- `DIRECT_ROLLOUT_METHOD` - Direct method call on rollout engine
- `RANK_ZERO` - Only rank 0 executes (logging, collection)

### PPO Training Loop Flow

1. **Init Phase**: Parse config, create Ray cluster, initialize worker groups
2. **Sample prompts** from train_dataset
3. **Rollout Phase**: ActorRolloutRefWorker generates sequences with log probs
4. **Reward Computation**: User-defined reward function (model-based or function-based)
5. **Critic/Value Computation**: Critic model estimates values
6. **Advantage Computation**: GAE/GRPO/REINFORCE++/etc. computes advantages
7. **Actor Training**: Mini-batch training with PPO clipped loss, gradient accumulation
8. **Critic Training**: Optional MSE loss on value estimates
9. **Checkpoint & Logging**: Save weights, log to WandB/MLflow/TensorBoard

### Key Modules

**Single Controller** (`verl/single_controller/`):
- `base/worker.py` - Base Worker class managing distributed ranks
- `base/decorator.py` - `@register()` decorator for automatic data distribution
- `ray/base.py` - RayWorkerGroup manages Ray actors

**Workers** (`verl/workers/`):
- `engine_workers.py` (608 lines) - **New unified** TrainingWorker + ActorRolloutRefWorker. Used when `trainer.use_legacy_worker_impl="disable"`.
- `fsdp_workers.py` (2067 lines) - **Legacy** FSDP worker implementations. Used when `trainer.use_legacy_worker_impl="auto"` (default) or `"enable"` with FSDP backend.
- `megatron_workers.py` (1498 lines) - **Legacy** Megatron worker implementations. Used with Megatron backend.
- `engine/base.py` - BaseEngine abstract interface
- `rollout/base.py` - BaseRollout abstract interface

**Trainer** (`verl/trainer/ppo/`):
- `ray_trainer.py` - Main training loop orchestrator
- `core_algos.py` - All RL algorithms and advantage estimation (~2200 lines). New algorithms are added via `@register_adv_est()` decorator.
- Algorithm selection via config: `algorithm.adv_estimator=gae|grpo|reinforce_plus_plus|reinforce_plus_plus_baseline|remax|rloo|opo|grpo_passk|gpg|rloo_vectorized|grpo_vectorized|optimal_token_baseline` (or any custom string registered via `register_adv_est`)

**Experimental** (`verl/experimental/`): Features under development including `fully_async_policy`, `transfer_queue`, `one_step_off_policy`, `vla`, `agent_loop`, `reward_loop`, `dynamic_dataset`, `dataset`.

## Extension Points

1. **New Training Backend**: Subclass `BaseEngine` in `verl/workers/engine/`
2. **New Rollout Engine**: Subclass `BaseRollout` in `verl/workers/rollout/`
3. **New RL Algorithm**: Register with `@register_adv_est()` in `trainer/ppo/core_algos.py`
4. **New Loss Function**: Register with `@register_policy_loss()` in `core_algos.py`
5. **Custom Reward**: Implement callable with signature `(input) -> reward_dict`

## Common Patterns

**Method Registration:**
```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def train_batch(self, data: DataProto) -> DataProto:
    # Automatically dispatches to all DP ranks
```

**Context Managers:**
```python
with engine.train_mode():
    output = engine.train_batch(data, loss_fn)
with engine.eval_mode():
    output = engine.infer_batch(data)
```

**Algorithm Registration:**
```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(token_level_rewards, ...):
    # GRPO-specific advantage computation
```

## Configuration System

verl uses Hydra for configuration management:

- Config templates in `verl/trainer/config/*.yaml`
- Override via command line: `trainer.n_gpus_per_node=8 data.train_batch_size=1024`
- Config dataclasses in `verl/workers/config.py` and `verl/trainer/config/`
- Auto-generated configs via `scripts/generate_trainer_config.sh` (pre-commit hook)
- Component subdirectories: `actor/`, `critic/`, `data/`, `rollout/`, `algorithm/`, `engine/`, `model/`, `optim/`, `profiler/`, `ref/`, `reward_model/`

## Important Notes

- **FSDP2 is recommended**: Set `actor_rollout_ref.actor.strategy=fsdp2` for better performance
- **vLLM 0.8+ features**: Enable cuda graph with `actor_rollout_ref.rollout.enforce_eager=False`
- **Recipe submodule**: Run `git submodule update --init --recursive recipe` to get training recipes
- **3D-HybridEngine**: VEOmni engine reshards actor between generation (TP=1) and training (full TP) for efficiency
- **GPU colocating**: `max_colocate_count` in `RayResourcePool` controls how many WorkerGroups share one GPU. Use 1 for FSDP (all roles in one process), 3+ for Megatron (separate actor/critic/rollout processes).
- **Docs**: Build with `cd docs && pip install -r requirements-docs.txt && make html`

## Quick Start for Common Tasks

**Adding new RL algorithm:**
1. Edit `verl/trainer/ppo/core_algos.py`
2. Add function with `@register_adv_est(AdvantageEstimator.YOUR_ALGO)`
3. Update config enum in `verl/trainer/config/algorithm.py`

**Debugging data flow:**
1. Check `verl/protocol.py` for DataProto methods
2. Check `verl/workers/config.py` for data structure definitions
3. Add logging in `verl/trainer/ppo/ray_trainer.py`

**Performance tuning:**
1. See https://verl.readthedocs.io/en/latest/perf/perf_tuning.html
2. Check `verl/trainer/ppo/metric_utils.py` for metrics
3. Tune batch sizes, gradient accumulation, offloading in config YAML

**Utility scripts:** `scripts/diagnose.py` (environment diagnostics), `scripts/converter_hf_to_mcore.py` (HF→Megatron checkpoint conversion), `scripts/rollout_viewer.py` (TUI for viewing rollout data).

## Contributing

See CONTRIBUTING.md for code style, testing requirements, PR guidelines, and CI workflow details.
