# Build the browser bundle separately so the control image stays small and has
# the same Docker CLI/runtime boundary as vbogs-pipeline.
FROM node:22-alpine AS frontend
WORKDIR /src/web
COPY web/package.json ./
RUN npm install --ignore-scripts
COPY web/ ./
RUN npm run build

FROM nvidia/cuda:12.8.0-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io python3 python3-numpy python3-pip python3-yaml && \
    rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m pip install --no-cache-dir "fastapi==0.115.14" "uvicorn[standard]==0.34.3" "httpx==0.28.1" "websockets==15.0.1"
WORKDIR /workspace/VBOGS
COPY --from=frontend /src/web/dist /opt/vbogs-web-dist
ENV PYTHONPATH=/workspace/VBOGS
EXPOSE 8090
ENV VBOGS_GUI_STATIC_DIR=/opt/vbogs-web-dist
CMD ["python", "scripts/serve_vbogs_web.py"]
