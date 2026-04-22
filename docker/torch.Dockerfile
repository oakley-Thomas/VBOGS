FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
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

WORKDIR /workspace/VBOGS

COPY . /workspace/VBOGS

RUN python -m pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

RUN python -m pip install torch_scatter \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

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

RUN python -m pip install rich gsplat \
    --extra-index-url https://docs.gsplat.studio/whl/pt27cu128

ENV PYTHONPATH=/workspace/VBOGS:/workspace/VBOGS/Octree-AnyGS

CMD ["sleep", "infinity"]
