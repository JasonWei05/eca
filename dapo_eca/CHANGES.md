# ECA Fork Changes

This document tracks all modifications made to the verl codebase and environment
for the ECA (Entropy-Controlled Advantage) DAPO training pipeline.

## Environment Fixes

### 1. GLIBC Compatibility Shim

**Problem:** flash_attn 2.8.1 (built against torch 2.8+cu128) references
`__libc_single_threaded` which requires GLIBC 2.32, but this system has GLIBC 2.28.

**Fix:** Created a GLIBC compat shim and binary-patched the flash_attn `.so` file.

- **`/tmp/glibc_compat.c`** — Source for the shim:
  ```c
  char __libc_single_threaded = 1;
  ```
- **`/home/tiger/miniconda3/envs/eca/lib/glibc_compat.so`** — Compiled shim,
  loaded via `LD_PRELOAD` in both `eca_qwen3_4b_base.sh` and `runtime_env.yaml`.
- **`flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so`** — Binary-patched ELF
  `.gnu.version` and `.gnu.version_r` sections to replace GLIBC_2.32 dependency
  with GLIBC_2.14. Backup at same path with `.bak` extension.

**IMPORTANT:** If you reinstall flash-attn (e.g. via `pip install flash_attn-*.whl`),
the binary patch will be overwritten. You must re-apply the patch afterward.
The install script `scripts/install_vllm_sglang_mcore.sh` has been updated to
handle this automatically (see below).

### 2. NumPy / Numba Compatibility

**Problem:** vLLM's ngram_proposer imports numba, which requires `numpy<=2.2`.
The default numpy 2.4.x caused `ImportError: Numba needs NumPy 2.2 or less`.

**Fix:** Pin numpy to `>=2.0.0,<2.3.0` in setup.py, install script, and
pyproject.toml dependencies. The original `numpy<2.0.0` constraint was too
restrictive and conflicted with other packages.

## Code Changes

### 3. Training Accuracy Metrics (`verl/trainer/ppo/ray_trainer.py`)

Added training-time accuracy logging after reward computation (~line 1590):

- **`train/acc/mean`** — Average 0/1 accuracy across all responses in the batch
  (raw correct/incorrect before overlong buffer penalty).
- **`train/acc/pass@{n}`** — Fraction of prompts where at least 1 of n responses
  is correct (n = `rollout.n`, e.g. 16). Only logged when `len(acc) % n == 0`.

### 4. Validation pass@k Metrics (`verl/trainer/ppo/metric_utils.py`)

Added unbiased pass@k estimator to `process_validation_metrics()` (~line 644):

- **`val-aux/{data_source}/acc/pass@2`**
- **`val-aux/{data_source}/acc/pass@8`**
- **`val-aux/{data_source}/acc/pass@32`**

Uses the standard unbiased estimator: `pass@k = 1 - C(n-c, k) / C(n, k)`
where n = total samples per prompt, c = number correct. Only computed when
`k <= n_resps` (e.g. pass@32 requires at least 32 samples per prompt).

### 5. Separate Eval Sampling Config

Added `max_new_tokens` field to `SamplingConfig` and plumbed it through the
validation override path so eval can use different generation length than training.

**Files changed:**

- **`verl/workers/config/rollout.py`** — Added `max_new_tokens: Optional[int] = None`
  to `SamplingConfig` dataclass.
- **`verl/experimental/agent_loop/agent_loop.py`** — Pass `max_new_tokens` to
  sampling_params during validation when `val_kwargs.max_new_tokens` is set.
- **`verl/experimental/fully_async_policy/agent_loop/agent_loop.py`** — Same change.
- **`verl/trainer/config/rollout/rollout.yaml`** — Added `max_new_tokens: null`
  to val_kwargs section.
- **`verl/trainer/config/_generated_ppo_trainer.yaml`** — Regenerated.
- **`verl/trainer/config/_generated_ppo_megatron_trainer.yaml`** — Regenerated.

### 6. ECA Algorithm Config Fields (`verl/trainer/config/ppo_trainer.yaml`)

Added three fields to the `algorithm` section (already existed in the Python
`AlgoConfig` dataclass but were missing from the YAML):

- `eca_linear: False`
- `eca_softmax: False`
- `eca_gamma: 1.0`

## Script Changes

### 7. Training Script (`dapo_eca/eca_qwen3_4b_base.sh`)

- Fixed `WORKING_DIR` and `MODEL_PATH` to point to `/home/tiger/jason_wei/eca`
- Added `export LD_PRELOAD=.../glibc_compat.so` before `ray start`
- Set `val_kwargs.temperature=0.7` (was `${temperature}` = 1.0)
- Added `val_kwargs.max_new_tokens=$((1024 * 10))` for longer eval outputs

### 8. Ray Runtime Environment (`dapo_eca/runtime_env.yaml`)

- Added `LD_PRELOAD` env var so Ray workers inherit the GLIBC compat shim.

## Reinstall Procedure

After a fresh `git clone` or `git pull`:

```bash
# 1. Install dependencies (handles numpy pin, flash_attn patch, glibc shim)
bash scripts/install_vllm_sglang_mcore.sh

# 2. Install verl in editable mode
pip install --no-deps -e .

# 3. Verify
python -c "import verl; print('verl OK')"
python -c "import flash_attn; print('flash_attn OK')"
python -c "import numpy; print(f'numpy {numpy.__version__}')"
python -c "import numba; print(f'numba {numba.__version__}')"
```
