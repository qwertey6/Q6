# Dockerfile — authoritative reproduction environment.
#
# Near-threshold PSE cases are pixel-sensitive; pin everything by digest.

# TODO(M0->M1): pin debian:bookworm-slim by digest in environment.lock after first build.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# System deps:
#   - ffmpeg: ffmpeg_photosensitivity adapter + codec round-trip fixture build
#   - cmake/clang/git: build EA IRIS from source at pinned commit
#   - python3 + venv: harness, detector, report
#   - octave: best-effort runtime for Apple VFR MATLAB reference impl
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget git make \
        ffmpeg \
        cmake clang lld \
        python3 python3-pip python3-venv \
        octave \
        jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python deps — exact pins from environment.lock.
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

# Source is mounted at /workspace by `make reproduce`. The build proceeds via
# `make all` inside the container so determinism is preserved end-to-end.
CMD ["make", "all"]
