# COLMAP publishes dated CI tags rather than release-number tags. This official
# CUDA image was verified on Docker Hub; keeping the dated tag is reproducible.
FROM colmap/colmap:20260727.7626

USER root
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg python3 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/VBOGS
ENV PYTHONPATH=/workspace/VBOGS
CMD ["sleep", "infinity"]
