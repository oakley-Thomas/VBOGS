FROM docker:27-cli

RUN apk add --no-cache git python3 py3-yaml

ARG VBOGS_GIT_URL=https://github.com/oakley-Thomas/VBOGS.git
ARG VBOGS_GIT_REF=main

RUN git clone "${VBOGS_GIT_URL}" /workspace/VBOGS && \
    cd /workspace/VBOGS && \
    git fetch --tags origin && \
    (git checkout "${VBOGS_GIT_REF}" || git checkout -B "${VBOGS_GIT_REF}" "origin/${VBOGS_GIT_REF}") && \
    git submodule update --init --recursive

WORKDIR /workspace/VBOGS

CMD ["sleep", "infinity"]
