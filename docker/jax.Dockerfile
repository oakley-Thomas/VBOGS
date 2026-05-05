FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install numpy pyyaml scipy

ARG VBOGS_GIT_URL=https://github.com/oakley-Thomas/VBOGS.git
ARG VBOGS_GIT_REF=main

RUN git clone "${VBOGS_GIT_URL}" /workspace/VBOGS && \
    cd /workspace/VBOGS && \
    git fetch --tags origin && \
    (git checkout "${VBOGS_GIT_REF}" || git checkout -B "${VBOGS_GIT_REF}" "origin/${VBOGS_GIT_REF}") && \
    git submodule update --init --recursive

WORKDIR /workspace/VBOGS

RUN python -m pip install -e /workspace/VBOGS/vbgs[gpu]

ENV PYTHONPATH=/workspace/VBOGS:/workspace/VBOGS/vbgs
ENV JAX_PLATFORMS=cuda

CMD ["sleep", "infinity"]
