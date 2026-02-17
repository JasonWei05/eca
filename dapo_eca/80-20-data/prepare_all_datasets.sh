#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Output directory for prepared data
DATA_DIR="${DATA_DIR:-${HOME}/verl/data}"
OVERWRITE="${OVERWRITE:-0}"
mkdir -p "${DATA_DIR}"

# Source files (download if not present)
TRAIN_SOURCE="${DATA_DIR}/math__combined_54.4k.parquet"
export TRAIN_SOURCE
if [[ ! -f "${TRAIN_SOURCE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Downloading training source dataset..."
  wget -O "${TRAIN_SOURCE}" \
    "https://huggingface.co/datasets/LLM360/guru-RL-92k/resolve/main/train/math__combined_54.4k.parquet"
else
  echo "Training source already exists: ${TRAIN_SOURCE}"
fi

# Output files
TRAIN_FILE="${DATA_DIR}/math__combined_54.4k_filtered.parquet"
export TRAIN_FILE
AIME2024_8X_FILE="${DATA_DIR}/math__aime2024_repeated_8x_240.parquet"
AIME2024_32X_FILE="${DATA_DIR}/math__aime2024_repeated_32x_960.parquet"
AIME2025_FILE="${DATA_DIR}/math__aime2025_30.parquet"
AIME2025_32X_FILE="${DATA_DIR}/math__aime2025_repeated_32x_960.parquet"
AIME2026_FILE="${DATA_DIR}/math__aime2026_30.parquet"
AIME2026_32X_FILE="${DATA_DIR}/math__aime2026_repeated_32x_960.parquet"
MATH500_FILE="${DATA_DIR}/math__math_500.parquet"
MATH500_2X_FILE="${DATA_DIR}/math__math_500_repeated_2x_1000.parquet"
OLYMPIADBENCH_FILE="${DATA_DIR}/math__olympiadbench.parquet"
OLYMPIADBENCH_2X_FILE="${DATA_DIR}/math__olympiadbench_repeated_2x.parquet"
MINERVAMATH_FILE="${DATA_DIR}/math__minervamath.parquet"
MINERVAMATH_4X_FILE="${DATA_DIR}/math__minervamath_repeated_4x.parquet"

echo "=== Dataset Preparation ==="
echo "Data directory: ${DATA_DIR}"
echo ""

# Step 1: Download AIME 2024 8x (base for duplication)
if [[ ! -f "${AIME2024_8X_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Downloading AIME 2024 8x dataset..."
  wget -O "${AIME2024_8X_FILE}" \
    "https://huggingface.co/datasets/LLM360/guru-RL-92k/resolve/main/offline_eval/math__aime_repeated_8x_240.parquet"
else
  echo "AIME 2024 8x already exists: ${AIME2024_8X_FILE}"
fi

# Step 2: Download MATH500
if [[ ! -f "${MATH500_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Downloading MATH500 dataset..."
  wget -O "${MATH500_FILE}" \
    "https://huggingface.co/datasets/LLM360/guru-RL-92k/resolve/main/offline_eval/math__math_500.parquet"
else
  echo "MATH500 already exists: ${MATH500_FILE}"
fi

# Step 3: Create AIME 2024 32x (4x duplication of 8x = 32x total)
if [[ ! -f "${AIME2024_32X_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Creating AIME 2024 32x dataset (for avg@32 scoring)..."
  python3 "${SCRIPT_DIR}/duplicate_aime.py" \
    --input_path "${AIME2024_8X_FILE}" \
    --save_path "${AIME2024_32X_FILE}" \
    --repeat_times 4
else
  echo "AIME 2024 32x already exists: ${AIME2024_32X_FILE}"
fi

# Step 3b: Convert AIME 2025 from HuggingFace
if [[ ! -f "${AIME2025_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Converting AIME 2025 from MathArena/aime_2025..."
  python3 "${SCRIPT_DIR}/convert_aime2025.py" --dataset MathArena/aime_2025 --year 2025 --output_path "${AIME2025_FILE}"
else
  echo "AIME 2025 already exists: ${AIME2025_FILE}"
fi

# Step 3c: Create AIME 2025 32x (32x duplication for avg@32)
if [[ ! -f "${AIME2025_32X_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Creating AIME 2025 32x dataset (for avg@32 scoring)..."
  python3 "${SCRIPT_DIR}/duplicate_aime.py" \
    --input_path "${AIME2025_FILE}" \
    --save_path "${AIME2025_32X_FILE}" \
    --repeat_times 32
else
  echo "AIME 2025 32x already exists: ${AIME2025_32X_FILE}"
fi

# Step 3d: Convert AIME 2026 from HuggingFace
if [[ ! -f "${AIME2026_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Converting AIME 2026 from MathArena/aime_2026..."
  python3 "${SCRIPT_DIR}/convert_aime2025.py" --dataset MathArena/aime_2026 --year 2026 --output_path "${AIME2026_FILE}"
else
  echo "AIME 2026 already exists: ${AIME2026_FILE}"
fi

# Step 3e: Create AIME 2026 32x (32x duplication for avg@32)
if [[ ! -f "${AIME2026_32X_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Creating AIME 2026 32x dataset (for avg@32 scoring)..."
  python3 "${SCRIPT_DIR}/duplicate_aime.py" \
    --input_path "${AIME2026_FILE}" \
    --save_path "${AIME2026_32X_FILE}" \
    --repeat_times 32
else
  echo "AIME 2026 32x already exists: ${AIME2026_32X_FILE}"
fi

# Step 3f: Create MATH500 2x (500 * 2 = 1000)
if [[ ! -f "${MATH500_2X_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Creating MATH500 2x dataset (~1000 instances)..."
  python3 "${SCRIPT_DIR}/duplicate_aime.py" \
    --input_path "${MATH500_FILE}" \
    --save_path "${MATH500_2X_FILE}" \
    --repeat_times 2
else
  echo "MATH500 2x already exists: ${MATH500_2X_FILE}"
fi

# Step 3g: Convert OlympiadBench from HuggingFace
if [[ ! -f "${OLYMPIADBENCH_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Converting OlympiadBench from math-ai/OlympiadBench..."
  python3 "${SCRIPT_DIR}/convert_olympiadbench.py" --output_path "${OLYMPIADBENCH_FILE}"
else
  echo "OlympiadBench already exists: ${OLYMPIADBENCH_FILE}"
fi

# Step 3h: Create OlympiadBench 2x (~1000 instances)
if [[ ! -f "${OLYMPIADBENCH_2X_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Creating OlympiadBench 2x dataset (~1000 instances)..."
  python3 "${SCRIPT_DIR}/duplicate_aime.py" \
    --input_path "${OLYMPIADBENCH_FILE}" \
    --save_path "${OLYMPIADBENCH_2X_FILE}" \
    --repeat_times 2
else
  echo "OlympiadBench 2x already exists: ${OLYMPIADBENCH_2X_FILE}"
fi

# Step 3i: Convert MinervaMAth from HuggingFace
if [[ ! -f "${MINERVAMATH_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Converting MinervaMAth from math-ai/minervamath..."
  python3 "${SCRIPT_DIR}/convert_minervamath.py" --output_path "${MINERVAMATH_FILE}"
else
  echo "MinervaMAth already exists: ${MINERVAMATH_FILE}"
fi

# Step 3j: Create MinervaMAth 4x (~1000 instances)
if [[ ! -f "${MINERVAMATH_4X_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Creating MinervaMAth 4x dataset (~1000 instances)..."
  python3 "${SCRIPT_DIR}/duplicate_aime.py" \
    --input_path "${MINERVAMATH_FILE}" \
    --save_path "${MINERVAMATH_4X_FILE}" \
    --repeat_times 4
else
  echo "MinervaMAth 4x already exists: ${MINERVAMATH_4X_FILE}"
fi

# Step 4: Filter and normalize training dataset
if [[ ! -f "${TRAIN_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Filtering and normalizing training dataset..."
  python3 - <<'PYTHON_SCRIPT'
import pandas as pd
import os

source_file = os.environ.get('TRAIN_SOURCE')
output_file = os.environ.get('TRAIN_FILE')

print(f"Reading {source_file}...")
df = pd.read_parquet(source_file)
print(f"Original shape: {df.shape}")
print(f"Original columns: {list(df.columns)}")
print(f"Original data_source values:\n{df['data_source'].value_counts()}")

# Keep only required columns
required_cols = ['data_source', 'prompt', 'reward_model', 'extra_info']
available_cols = [c for c in required_cols if c in df.columns]
df_filtered = df[available_cols].copy()

# Normalize data_source to "math" for compatibility with reward scoring
# The reward scorer expects "math", "math_dapo", or strings starting with "aime"
df_filtered['data_source'] = 'math'

print(f"\nFiltered shape: {df_filtered.shape}")
print(f"Filtered columns: {list(df_filtered.columns)}")
print(f"Normalized data_source: {df_filtered['data_source'].unique()}")

print(f"\nSaving to {output_file}...")
df_filtered.to_parquet(output_file, index=False)
print("Done!")
PYTHON_SCRIPT
else
  echo "Filtered training file already exists: ${TRAIN_FILE}"
fi

# Step 5: Filter test datasets to keep only required keys
echo ""
echo "Filtering test datasets..."
python3 "${SCRIPT_DIR}/filter_test_dataset_keys.py" --input_file "${AIME2024_8X_FILE}"
python3 "${SCRIPT_DIR}/filter_test_dataset_keys.py" --input_file "${AIME2024_32X_FILE}"
python3 "${SCRIPT_DIR}/filter_test_dataset_keys.py" --input_file "${MATH500_FILE}"
# AIME 2025 is already in correct format from convert_aime2025.py

# Step 6: Set distinct data_source values for log/metric distinguishability
# Reward scorer recognizes: "math", "math500", and anything starting with "aime"
echo ""
echo "Setting data_source labels..."
python3 -c "
import pandas as pd, sys, os
updates = [
    ('${AIME2024_8X_FILE}', 'aime2024'),
    ('${AIME2024_32X_FILE}', 'aime2024'),
    ('${AIME2025_32X_FILE}', 'aime2025'),
    ('${AIME2026_32X_FILE}', 'aime2026'),
    ('${MATH500_FILE}', 'math500'),
    ('${MATH500_2X_FILE}', 'math500'),
    ('${OLYMPIADBENCH_FILE}', 'olympiadbench'),
    ('${OLYMPIADBENCH_2X_FILE}', 'olympiadbench'),
    ('${MINERVAMATH_FILE}', 'minervamath'),
    ('${MINERVAMATH_4X_FILE}', 'minervamath'),
]
for path, ds in updates:
    if not os.path.exists(path):
        print(f'SKIP (not found): {path}')
        continue
    df = pd.read_parquet(path)
    df['data_source'] = ds
    df.to_parquet(path, index=False)
    print(f'  {os.path.basename(path)}: data_source -> {ds}')
"

echo ""
echo "=== Summary ==="
echo "Training file:     ${TRAIN_FILE}"
echo "AIME 2024 32x:     ${AIME2024_32X_FILE}"
echo "AIME 2025 32x:     ${AIME2025_32X_FILE}"
echo "AIME 2026 32x:     ${AIME2026_32X_FILE}"
echo "MATH500 2x:        ${MATH500_2X_FILE}"
echo "OlympiadBench 2x:  ${OLYMPIADBENCH_2X_FILE}"
echo "MinervaMAth 4x:    ${MINERVAMATH_4X_FILE}"
echo ""
echo "Update your training script with:"
echo "  TRAIN_FILE=\"${TRAIN_FILE}\""
echo "  data.val_files=\"['${AIME2024_32X_FILE}','${AIME2025_32X_FILE}','${AIME2026_32X_FILE}','${MATH500_2X_FILE}','${OLYMPIADBENCH_2X_FILE}','${MINERVAMATH_4X_FILE}']\""
