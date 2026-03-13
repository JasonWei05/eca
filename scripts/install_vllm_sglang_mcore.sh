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
# Build flash-attn from source to match local CUDA arch and GLIBC.
# Prebuilt wheels often have GLIBC_2.32 deps and wrong GPU arch.
CONDA_PREFIX="${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}"

# Install GCC 12 via conda if system GCC is too old (<9) or missing.
# nvcc 12.x supports GCC up to 12; system GCC 8.x is too old for torch headers.
SYSTEM_GCC_VER=$(gcc -dumpversion 2>/dev/null | cut -d. -f1)
if [ -z "${SYSTEM_GCC_VER}" ] || [ "${SYSTEM_GCC_VER}" -lt 9 ] 2>/dev/null; then
    echo "  System GCC ${SYSTEM_GCC_VER:-not found} too old (need 9-12). Installing GCC 12 via conda..."
    conda install -y -c conda-forge gcc_linux-64=12 gxx_linux-64=12
fi

# Use conda GCC if available
if [ -f "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc" ]; then
    export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
echo "  Building flash-attn from source (CC=${CC:-gcc}, CUDA_HOME=${CUDA_HOME})..."
FLASH_ATTENTION_FORCE_BUILD=TRUE \
    pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1

pip install --no-cache-dir flashinfer-python==0.3.1

# Build glibc_compat.so (still needed by other libs via LD_PRELOAD in runtime_env.yaml)
GLIBC_COMPAT_SO="${CONDA_PREFIX}/lib/glibc_compat.so"
if ! [ -f "${GLIBC_COMPAT_SO}" ]; then
    echo "  Building glibc_compat.so ..."
    TMPFILE=$(mktemp /tmp/glibc_compat_XXXXXX.c)
    echo 'char __libc_single_threaded = 1;' > "${TMPFILE}"
    ${CC:-gcc} -shared -o "${GLIBC_COMPAT_SO}" "${TMPFILE}" -fPIC
    rm -f "${TMPFILE}"
    echo "  Created ${GLIBC_COMPAT_SO}"
else
    echo "  ${GLIBC_COMPAT_SO} already exists, skipping build"
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
