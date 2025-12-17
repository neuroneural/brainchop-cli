# Builder stage
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    git \
    cmake \
    libomp-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and build niimath
WORKDIR /build
RUN git clone https://github.com/rordenlab/niimath.git && \
    cd niimath && \
    cd src && \
    make

# Main image
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    libomp5 \
    zlib1g \
    pigz \
    clang \
    llvm \
    gcc-aarch64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /opt/niimath/linux

# Copy the binary - CMake typically builds it without extension
COPY --from=builder /build/niimath/src/niimath /opt/niimath/linux/niimath

# Rest of your Dockerfile
WORKDIR /app
COPY . .
ENV NIIMATH_PATH=/opt/niimath
ENV NIIMATH_TEMP=/tmp
ENV AFNI_COMPRESSOR=PIGZ

# Set compiler flags for ARM compatibility
ENV CFLAGS="-O2 -fPIC -ffreestanding -fno-math-errno"
ENV LDFLAGS="-fuse-ld=lld"

# Install Python dependencies
RUN pip install --no-cache-dir .

# Set the entrypoint
ENTRYPOINT ["brainchop"]
