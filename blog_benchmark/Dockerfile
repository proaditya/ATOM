# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Qwen3.8-27B single-request benchmark (blog_benchmark/build_and_repro.sh), ROCm 10.0 (gfx950): the PyTorch ROCm 10
# image (torch 2.13.0+rocm10.0.0), ROCm's devel wheels, aiter at the commit of the measured runs, ATOM from this
# checkout, the pinned FlyDSL (frontend 19418894 over the flydsl 0.4.0.dev963 wheel) and the megakernel and drafter
# code objects, prebuilt (compile-only: no GPU at build time). The environment is scripts/qwen38_flash_mono/repro.sh's
# setup, installed into the image's own Python. No weights: the model and the draft come in at run time.
# From the checkout:
#
#   docker build -f blog_benchmark/Dockerfile -t atom-qwen38-27b .
#   docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g \
#       -v /path/to/models:/models -e MODEL=/models/Qwen3.8-27B -e DRAFT=/models/Qwen3.8-27B-DFlash2 \
#       -v /path/to/work:/work atom-qwen38-27b
#
# The default command is blog_benchmark/build_and_repro.sh: both paths next to the published numbers, results in /work.

# rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.13.0
FROM rocm/pytorch@sha256:bbdaba66029f905321be3bbc95206a2d9a56a2bc5300877d146ad24d40db9a12

ARG AITER_COMMIT=c53e9b3f4285c36ea785e614ce5cf8b04c78b3bd
ENV PYTHONDONTWRITEBYTECODE=1 AITER_LOG_LEVEL=WARNING HSA_COREDUMP_PATTERN=/dev/null
# ROCm 10: four HIP hardware queues slow every kernel of a megakernel round; one removes that (see the recipe)
ENV GPU_MAX_HW_QUEUES=1
COPY scripts/qwen38_flash_mono/repro-constraints.txt /opt/constraints.txt

# ROCm's compilers and headers (aiter builds its kernels at run time) as TheRock wheels
RUN PIP_CONSTRAINT=/opt/constraints.txt python3 -m pip install --no-cache-dir "rocm[devel]==10.0.0" \
        --index-url https://stable.repo.amd.com/rocm/whl-next \
    && python3 -m rocm_sdk init
ENV ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
ENV HIP_PATH=$ROCM_PATH CMAKE_PREFIX_PATH=$ROCM_PATH PATH=$ROCM_PATH/bin:$PATH \
    LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib/host-math/lib

# AITER_USE_SYSTEM_TRITON=1: aiter's setup.py would otherwise replace the image's Triton (the build torch pins)
# with a ROCm 7.2 wheel, which runs ATOM's Triton kernels slower (Qwen3.8-Flash-Next's regular decode: 17.3
# instead of 13.9 ms per round on an MI350X)
RUN git clone --filter=blob:none https://github.com/ROCm/aiter.git /opt/aiter \
    && cd /opt/aiter && git checkout -q "$AITER_COMMIT" && git submodule update --init --recursive --depth 1 \
    && PIP_CONSTRAINT=/opt/constraints.txt python3 -m pip install --no-cache-dir -r requirements.txt \
    && AITER_USE_SYSTEM_TRITON=1 MAX_JOBS=64 PREBUILD_KERNELS=0 GPU_ARCHS=gfx950 python3 setup.py develop

COPY . /opt/atom
WORKDIR /opt/atom
# (no .git in the image: setuptools-scm falls back to the package's version; safe.directory serves a checkout
# bind-mounted over /opt/atom, owned by another user than the container's root)
RUN git config --system --add safe.directory '*' \
    && PIP_CONSTRAINT=/opt/constraints.txt python3 -m pip install --no-cache-dir setuptools-scm \
    && PIP_CONSTRAINT=/opt/constraints.txt python3 -m pip install --no-cache-dir --no-build-isolation -e . msgpack

# steps 3 and 4 of the script, here: FlyDSL for this Python, the megakernel and the drafter code objects
ENV FLYDSL_DIR=/opt/qwen38/flydsl \
    DRAFTER_OBJECTS=/opt/qwen38/drafter-objects \
    ATOM_QWEN38_MK_CACHE=/opt/qwen38/kernel-cache
RUN SETUP_ONLY=1 WORK=/tmp/setup PY=python3 bash blog_benchmark/build_and_repro.sh \
    && rm -rf /tmp/setup "$FLYDSL_DIR/wheel/x"

# the Triton torch pins, and a consistent environment
RUN python3 -m pip check && python3 -c "import importlib.metadata as m; \
need = [r.split('==')[1] for r in m.requires('torch') if r.startswith('triton==')][0]; \
have = m.version('triton'); \
assert have == need, f'triton {have}, torch needs {need}'; print('triton', have)"

# checkpoints downloaded by id, prompt sets, results: on the /work volume
ENV WORK=/work HF_HOME=/work/hf-cache
VOLUME /work
CMD ["bash", "blog_benchmark/build_and_repro.sh"]
