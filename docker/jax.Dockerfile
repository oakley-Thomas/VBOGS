FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    ca-certificates \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    numpy \
    pyyaml \
    scipy \
    datasets \
    equinox \
    hydra-core \
    "jax[cuda12]>=0.3.4" \
    "jaxlib[cuda12]>=0.3.4" \
    jaxtyping \
    matplotlib \
    multimethod \
    opencv-python \
    pillow \
    rich \
    ruff \
    tqdm

WORKDIR /workspace/VBOGS

ENV PYTHONPATH=/workspace/VBOGS:/workspace/VBOGS/vbgs
ENV JAX_PLATFORMS=cuda

CMD ["sleep", "infinity"]
