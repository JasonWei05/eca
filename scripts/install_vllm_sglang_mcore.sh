#!/bin/bash

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}

export MAX_JOBS=32

echo "1. install inference frameworks and pytorch they need"
if [ $USE_SGLANG -eq 1 ]; then
    pip install "sglang[all]==0.5.2" --no-cache-dir && pip install torch-memory-saver --no-cache-dir
fi
pip install --no-cache-dir "vllm==0.11.0"

echo "2. install basic packages"
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy>=2.0.0,<2.3.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard

echo "pyext is lack of maintainace and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
echo "pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
# Install flash-attn-2.8.1 (cxx11abi=False)
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && \
    pip install --no-cache-dir flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

pip install --no-cache-dir flashinfer-python==0.3.1

echo "3b. GLIBC compat: build shim and patch flash_attn .so for GLIBC < 2.32"
CONDA_PREFIX="${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}"
GLIBC_COMPAT_SO="${CONDA_PREFIX}/lib/glibc_compat.so"
if ! [ -f "${GLIBC_COMPAT_SO}" ]; then
    echo "  Building glibc_compat.so ..."
    TMPFILE=$(mktemp /tmp/glibc_compat_XXXXXX.c)
    echo 'char __libc_single_threaded = 1;' > "${TMPFILE}"
    gcc -shared -o "${GLIBC_COMPAT_SO}" "${TMPFILE}" -fPIC
    rm -f "${TMPFILE}"
    echo "  Created ${GLIBC_COMPAT_SO}"
else
    echo "  ${GLIBC_COMPAT_SO} already exists, skipping build"
fi

# Patch flash_attn .so to remove GLIBC_2.32 requirement
# NOTE: We cannot `import flash_attn` to find the .so path because the import
# itself fails on GLIBC < 2.32 systems (the very issue we are patching).
# Instead, search site-packages directly.
FLASH_SO=$(python -c "
import sysconfig, glob, os
for key in ('platlib', 'purelib'):
    sp = sysconfig.get_path(key)
    if not sp:
        continue
    for pattern in ('flash_attn_2_cuda*.so', 'flash_attn/flash_attn_2_cuda*.so'):
        matches = glob.glob(os.path.join(sp, pattern))
        if matches:
            print(matches[0])
            raise SystemExit(0)
" 2>/dev/null)
if [ -n "${FLASH_SO}" ] && [ -f "${FLASH_SO}" ]; then
    if readelf -V "${FLASH_SO}" 2>/dev/null | grep -q "GLIBC_2.32"; then
        echo "  Patching ${FLASH_SO} to remove GLIBC_2.32 dependency ..."
        cp "${FLASH_SO}" "${FLASH_SO}.bak"
        python3 scripts/patch_flash_attn_glibc.py "${FLASH_SO}"
        echo "  Backup saved as ${FLASH_SO}.bak"
    else
        echo "  flash_attn .so does not reference GLIBC_2.32, no patch needed"
    fi
else
    echo "  WARNING: flash_attn .so not found, skipping patch"
fi


if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    pip install "onnxscript==0.3.1"
    NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.6
    pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1
fi


echo "5. May need to fix opencv"
pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"


if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overridden)"
    pip install nvidia-cudnn-cu12==9.10.2.21
fi

echo "Successfully installed all packages"
