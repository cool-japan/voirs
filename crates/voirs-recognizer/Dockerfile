# Multi-stage Dockerfile for VoiRS Recognizer
# Supports both development and production builds

# ============================================================================
# Stage 1: Base dependencies and tools
# ============================================================================
FROM rust:1.78-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    pkg-config \
    cmake \
    git \
    curl \
    wget \
    # Audio libraries
    libasound2-dev \
    libpulse-dev \
    libjack-dev \
    portaudio19-dev \
    # SSL and crypto
    libssl-dev \
    ca-certificates \
    # Python for tooling
    python3 \
    python3-pip \
    # Additional utilities
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Rust tools
RUN rustup component add rustfmt clippy
RUN cargo install cargo-audit cargo-deny wasm-pack

# Create app directory
WORKDIR /app

# ============================================================================
# Stage 2: Development environment
# ============================================================================
FROM base as development

# Install additional development tools
RUN cargo install \
    cargo-watch \
    cargo-tarpaulin \
    cargo-outdated \
    cargo-tree \
    cargo-expand \
    cargo-bloat

# Install Node.js for WASM development
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Install wasm-pack
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Copy source code
COPY . .

# Expose ports for development
EXPOSE 8080 3000 8000

# Development command
CMD ["cargo", "watch", "-x", "test"]

# ============================================================================
# Stage 3: Build environment
# ============================================================================
FROM base as builder

# Set build environment variables
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true
ENV RUST_BACKTRACE=1

# Copy dependency manifests
COPY Cargo.toml Cargo.lock ./
COPY crates/voirs-recognizer/Cargo.toml ./crates/voirs-recognizer/

# Create dummy source files to cache dependencies
RUN mkdir -p src crates/voirs-recognizer/src \
    && echo "fn main() {}" > src/main.rs \
    && echo "fn main() {}" > crates/voirs-recognizer/src/lib.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release --all-features
RUN rm -rf src crates/

# Copy actual source code
COPY . .

# Build the actual application
RUN touch src/lib.rs crates/voirs-recognizer/src/lib.rs \
    && cargo build --release --all-features \
    && cargo test --release --all-features

# Generate C header
RUN python3 generate-header.py

# Build WASM packages
RUN ./build-wasm.sh

# Strip binaries
RUN strip target/release/libvoirs_recognizer.so 2>/dev/null || true

# ============================================================================
# Stage 4: Production runtime
# ============================================================================
FROM debian:bookworm-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    # Audio runtime libraries
    libasound2 \
    libpulse0 \
    libjack0 \
    portaudio19-dev \
    # SSL runtime
    libssl3 \
    ca-certificates \
    # Utilities
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r voirs && useradd -r -g voirs -s /bin/false voirs

# Create application directories
RUN mkdir -p /app/lib /app/bin /app/include /app/models /app/data \
    && chown -R voirs:voirs /app

# Copy built artifacts from builder stage
COPY --from=builder --chown=voirs:voirs /app/target/release/libvoirs_recognizer.* /app/lib/
COPY --from=builder --chown=voirs:voirs /app/include/voirs_recognizer.h /app/include/
COPY --from=builder --chown=voirs:voirs /app/pkg /app/pkg/
COPY --from=builder --chown=voirs:voirs /app/examples /app/examples/
COPY --chown=voirs:voirs README.md LICENSE-* /app/

# Set up environment
ENV LD_LIBRARY_PATH=/app/lib:$LD_LIBRARY_PATH
ENV PATH=/app/bin:$PATH

# Create health check script
RUN echo '#!/bin/bash\n\
# Health check for VoiRS Recognizer\n\
if [ -f /app/lib/libvoirs_recognizer.so ]; then\n\
    echo "VoiRS Recognizer library found"\n\
    exit 0\n\
else\n\
    echo "VoiRS Recognizer library not found"\n\
    exit 1\n\
fi' > /app/healthcheck.sh \
    && chmod +x /app/healthcheck.sh \
    && chown voirs:voirs /app/healthcheck.sh

# Switch to non-root user
USER voirs
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Default command
CMD ["echo", "VoiRS Recognizer container ready. Library available at /app/lib/libvoirs_recognizer.so"]

# ============================================================================
# Stage 5: REST API server
# ============================================================================
FROM production as rest-api

# Copy REST API binary (if built separately)
# COPY --from=builder --chown=voirs:voirs /app/target/release/voirs-recognizer-server /app/bin/

# Expose REST API port
EXPOSE 8080

# Environment variables for REST API
ENV RUST_LOG=info
ENV VOIRS_BIND_ADDRESS=0.0.0.0:8080
ENV VOIRS_MODEL_PATH=/app/models
ENV VOIRS_DATA_PATH=/app/data

# Create startup script for REST API
RUN echo '#!/bin/bash\n\
echo "Starting VoiRS Recognizer REST API..."\n\
echo "Bind address: $VOIRS_BIND_ADDRESS"\n\
echo "Model path: $VOIRS_MODEL_PATH"\n\
echo "Data path: $VOIRS_DATA_PATH"\n\
# Add actual REST API startup command here\n\
echo "REST API server would start here"\n\
sleep infinity' > /app/start-api.sh \
    && chmod +x /app/start-api.sh

# Health check for REST API
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start REST API server
CMD ["/app/start-api.sh"]

# ============================================================================
# Stage 6: WASM development server
# ============================================================================
FROM node:18-slim as wasm-dev

# Install serve for development
RUN npm install -g serve http-server

# Copy WASM packages
COPY --from=builder /app/pkg /app/pkg/
COPY --from=builder /app/examples/wasm /app/examples/
COPY --from=builder /app/package.json /app/

WORKDIR /app

# Install npm dependencies
RUN npm install

# Expose development server port
EXPOSE 8080

# Start development server
CMD ["serve", "-s", ".", "-l", "8080"]

# ============================================================================
# Build arguments and labels
# ============================================================================

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Labels
LABEL maintainer="VoiRS Team" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="voirs-recognizer" \
      org.label-schema.description="VoiRS Speech Recognition and Analysis Library" \
      org.label-schema.url="https://github.com/cool-japan/voirs" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/cool-japan/voirs" \
      org.label-schema.vendor="Cool Japan" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"