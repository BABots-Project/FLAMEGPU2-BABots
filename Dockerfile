FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
# For now, abort for non-x86 arch's just in case, as uri's are baked for x86
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
        'samd64') \
            # do nothing
            ;; \
        *) \
            echo >&2 "error: current arch (${arch}) not supperted by this dockerfile.;" \
            exit 1; \
        ;; \
    esac;

# Install Dependencies for FLAME GPU 2 (console, no python)
# There are not yet binary releases of the c++ static library which can be installed independently of models.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        cmake \
        swig4.0 \
        git \
        python3 python3-pip python3-venv \
    ; \
    python3 -m pip install wheel setuptools build matplotlib ; \
    rm -rf /var/lib/apt/lists/*; \
    gcc --version; \
    nvcc --version; \
    cmake --version

# Copy FLAMEGPU2 configure and build
COPY . /opt/FLAMEGPU2-babots
RUN set -eux; \
    cd /opt/FLAMEGPU2-babots ;\
    mkdir -p /opt/FLAMEGPU2-babots/build && cd /opt/FLAMEGPU2-babots/build ;\
    cmake .. -DFLAMEGPU_SEATBELTS=OFF -DCMAKE_BUILD_TYPE=Release;\
    #change target here
    cmake --build . --target worm_aggregation -j `nproc`

CMD cd /opt/FLAMEGPU2-babots/build && ./bin/Release/worm_aggregation -s 2000 -i ../babots/worm_aggregation/src/parameters.json
# set an env var CUDA_HOME so nvrtc can be found by FLAME GPU2 at runtime
ENV CUDA_HOME=/usr/local/cuda
# set an env var FLAMEPGU_INC_DIR so FLAME GPU 2's include directory can be found at runtime, when not executing from the build dir.
ENV FLAMEGPU_INC_DIR=/opt/FLAMEGPU2-worm_aggregation/build/_deps/flamegpu2-src/include
ENV FLAMEGPU2_INC_DIR=/opt/FLAMEGPU2-worm_aggregation/build/_deps/flamegpu2-src/include