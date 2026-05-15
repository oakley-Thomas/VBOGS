FROM docker:27-cli AS docker-cli

FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    ffmpeg \
    git \
    python3 \
    python3-yaml \
    rclone \
    unzip \
    zip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

ARG VBOGS_GIT_URL=https://github.com/oakley-Thomas/VBOGS.git
ARG VBOGS_GIT_REF=main

RUN git clone "${VBOGS_GIT_URL}" /workspace/VBOGS && \
    cd /workspace/VBOGS && \
    git fetch --tags origin && \
    (git checkout "${VBOGS_GIT_REF}" || git checkout -B "${VBOGS_GIT_REF}" "origin/${VBOGS_GIT_REF}") && \
    git submodule update --init --recursive

WORKDIR /workspace/VBOGS

CMD ["sleep", "infinity"]
