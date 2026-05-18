FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    git \
    build-essential \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN python -m pip install \
    numpy \
    scipy \
    opencv-python \
    plyfile \
    pillow \
    matplotlib \
    rich \
    tqdm \
    ninja

ARG VBOGS_RENDER_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ARG VBOGS_RENDER_MAX_JOBS=1

ENV TORCH_CUDA_ARCH_LIST="${VBOGS_RENDER_CUDA_ARCH_LIST}" \
    MAX_JOBS="${VBOGS_RENDER_MAX_JOBS}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${VBOGS_RENDER_MAX_JOBS}" \
    NINJAFLAGS="-j${VBOGS_RENDER_MAX_JOBS}"

ARG VBOGS_GIT_URL=https://github.com/oakley-Thomas/VBOGS.git
ARG VBOGS_GIT_REF=main

RUN git clone "${VBOGS_GIT_URL}" /workspace/VBOGS && \
    cd /workspace/VBOGS && \
    git fetch --tags origin && \
    (git checkout "${VBOGS_GIT_REF}" || git checkout -B "${VBOGS_GIT_REF}" "origin/${VBOGS_GIT_REF}") && \
    git submodule update --init --recursive

RUN git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git /workspace/gaussian-splatting && \
    cd /workspace/gaussian-splatting && \
    git checkout 2eee0e26d2d5fd00ec462df47752223952f6bf4e && \
    git submodule update --init --recursive && \
    cd submodules/simple-knn && \
    python setup.py install && \
    cd ../diff-gaussian-rasterization && \
    python setup.py install

WORKDIR /workspace/VBOGS

RUN python -m pip install -e /workspace/VBOGS/vbgs

ENV PYTHONPATH=/workspace/VBOGS:/workspace/VBOGS/vbgs:/workspace/gaussian-splatting

CMD ["sleep", "infinity"]
