FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    colmap \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    python3 \
    python3-pip \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    numpy \
    opencv-python \
    Pillow \
    plyfile

WORKDIR /workspace/VBOGS

ENV PYTHONPATH=/workspace/VBOGS

CMD ["sleep", "infinity"]
