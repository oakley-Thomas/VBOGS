FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

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
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

RUN python -m pip install torch_scatter \
    -f https://data.pyg.org/whl/torch-2.7.1+cu128.html

RUN python -m pip install \
    numpy \
    scipy \
    datasets \
    equinox \
    hydra-core \
    jax \
    jaxtyping \
    multimethod \
    opencv-python \
    plyfile \
    tensorboard \
    pillow \
    matplotlib \
    einops \
    wandb \
    lpips \
    laspy \
    colorama \
    scikit-learn \
    kornia \
    pyyaml \
    huggingface_hub \
    nvidia-ncore \
    rich \
    tqdm \
    ninja \
    "fastapi==0.115.14" \
    "uvicorn[standard]==0.34.3"

ARG VBOGS_RENDER_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0+PTX;12.0+PTX"
ARG VBOGS_RENDER_MAX_JOBS=1

ENV TORCH_CUDA_ARCH_LIST="${VBOGS_RENDER_CUDA_ARCH_LIST}" \
    MAX_JOBS="${VBOGS_RENDER_MAX_JOBS}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${VBOGS_RENDER_MAX_JOBS}" \
    NINJAFLAGS="-j${VBOGS_RENDER_MAX_JOBS}"

RUN python -m pip install --no-build-isolation gsplat==1.5.3 && \
    python -c "import gsplat, torch; assert torch.version.cuda == '12.8', torch.version.cuda; print('gsplat', getattr(gsplat, '__version__', 'unknown'))"

RUN git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git /workspace/gaussian-splatting && \
    cd /workspace/gaussian-splatting && \
    git checkout 2eee0e26d2d5fd00ec462df47752223952f6bf4e && \
    git submodule update --init --recursive && \
    sed -i '1i #include <cfloat>' submodules/simple-knn/simple_knn.cu && \
    cd submodules/simple-knn && \
    python setup.py install && \
    cd ../diff-gaussian-rasterization && \
    python setup.py install

RUN python -m pip install "kornia==0.7.4"

WORKDIR /workspace/VBOGS

ENV PYTHONPATH=/workspace/VBOGS:/workspace/VBOGS/Octree-AnyGS:/workspace/VBOGS/vbgs:/workspace/gaussian-splatting

EXPOSE 8070

CMD ["sleep", "infinity"]
