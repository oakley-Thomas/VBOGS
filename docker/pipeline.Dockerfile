FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    docker.io \
    ffmpeg \
    git \
    python3 \
    python3-yaml \
    rclone \
    unzip \
    zip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

COPY scripts/bootstrap_stack_repo.py /usr/local/bin/vbogs-bootstrap-repo
RUN chmod +x /usr/local/bin/vbogs-bootstrap-repo

WORKDIR /workspace/VBOGS

CMD ["sleep", "infinity"]
