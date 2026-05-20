# Dockerfile -- authoritative reproduction environment.
#
# Builds:
#   * The full toolchain for the harness + detector + report (Python).
#   * EA IRIS C++ at the pinned commit, via vcpkg + cmake + ninja,
#     installed to /usr/local/bin/iris-example for the harness adapter.
#   * GNU Octave (best-effort substrate for the Apple VFR MATLAB
#     reference; the apple_vfr adapter still reports UNSUPPORTED until
#     compatibility is verified).
#
# Near-threshold PSE cases are pixel-sensitive; pin everything by digest
# in environment.lock.

# TODO(M0->M1): pin debian:bookworm-slim by digest in environment.lock
# after first build.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    VCPKG_ROOT=/opt/vcpkg

# System deps:
#   - ffmpeg: ffmpeg_photosensitivity adapter + codec round-trip fixture build
#   - build-essential / gcc / g++ / cmake / ninja-build / clang / lld /
#     pkg-config / zip / unzip / tar / autoconf / automake / libtool:
#     vcpkg + IRIS C++ build chain
#   - python3 + venv: harness, detector, report
#   - octave: best-effort runtime for Apple VFR MATLAB reference impl
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget git make \
        ffmpeg \
        build-essential gcc g++ cmake ninja-build clang lld pkg-config \
        zip unzip tar autoconf automake libtool libtool-bin \
        autoconf-archive gettext intltool gtk-doc-tools dh-autoreconf \
        bison flex m4 gperf texinfo nasm \
        # X11 dev libs needed for cairo (vcpkg pulls cairo with x11 feature
        # via opencv-contrib -> highgui chain).
        libx11-dev libxext-dev libxrender-dev libxft-dev \
        libxcb1-dev libxrandr-dev libxinerama-dev libxi-dev libxv-dev \
        libxcursor-dev libxfixes-dev libxdamage-dev libxcomposite-dev \
        # OpenGL/EGL for opencv GUI modules.
        libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
        # Other commonly required headers.
        libgtk-3-dev libssl-dev \
        python3 python3-pip python3-venv \
        octave \
        jq \
    && rm -rf /var/lib/apt/lists/*

# vcpkg -- full clone (not shallow). IRIS's vcpkg.json pins
# `builtin-baseline` to an older upstream commit; a `--depth 1` clone
# would not contain that commit's history and `vcpkg install` would
# fail with "path 'versions/baseline.json' exists on disk, but not in
# <baseline-commit>". The full clone is ~150 MB and is reproducible
# enough -- the baseline commit pins package versions; we just need the
# history to reach that commit.
RUN git clone https://github.com/microsoft/vcpkg.git ${VCPKG_ROOT} \
 && ${VCPKG_ROOT}/bootstrap-vcpkg.sh -disableMetrics

# IRIS at the pinned commit. Build with BUILD_EXAMPLE_APP=ON since the
# console example binary is what the harness adapter invokes.
ARG IRIS_COMMIT=d96978ac1107f3463b77f69a9c1b1ec5d45291a0
RUN git clone https://github.com/electronicarts/IRIS.git /opt/IRIS \
 && cd /opt/IRIS \
 && git checkout ${IRIS_COMMIT} \
 && cmake --preset linux-release \
      -DBUILD_EXAMPLE_APP=ON \
      -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \
 && cmake --build --preset linux-release --target IrisApp \
 && install -m 0755 /opt/IRIS/bin/build/linux-release/example/IrisApp \
      /usr/local/bin/iris-example

WORKDIR /workspace

# Python deps -- exact pins from environment.lock.
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

# Source is mounted at /workspace by `make reproduce`. The build proceeds
# via `make all` inside the container so determinism is preserved
# end-to-end.
CMD ["make", "all"]
