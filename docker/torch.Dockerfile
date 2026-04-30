FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    unzip \
    build-essential \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN python -m pip install torch_scatter \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

RUN python -m pip install \
    plyfile \
    tensorboard \
    tqdm \
    einops \
    wandb \
    lpips \
    laspy \
    jaxtyping \
    colorama \
    opencv-python \
    scikit-learn \
    matplotlib \
    kornia \
    pyyaml \
    ninja

RUN python -m pip install rich && \
    python -m pip install gsplat \
    --index-url https://docs.gsplat.studio/whl/pt24cu124

RUN python -c "import gsplat, torch; assert torch.version.cuda == '12.4', torch.version.cuda; assert '+pt24cu124' in getattr(gsplat, '__version__', ''), getattr(gsplat, '__version__', 'unknown')"

ARG VBOGS_GIT_URL=https://github.com/oakley-Thomas/VBOGS.git
ARG VBOGS_GIT_REF=main

RUN git clone "${VBOGS_GIT_URL}" /workspace/VBOGS && \
    cd /workspace/VBOGS && \
    git fetch --tags origin && \
    (git checkout "${VBOGS_GIT_REF}" || git checkout -B "${VBOGS_GIT_REF}" "origin/${VBOGS_GIT_REF}") && \
    git submodule update --init --recursive

WORKDIR /workspace/VBOGS

ENV PYTHONPATH=/workspace/VBOGS:/workspace/VBOGS/Octree-AnyGS

CMD ["sleep", "infinity"]
