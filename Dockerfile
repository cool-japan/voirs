# Multi-stage build for VoiRS - Enhanced Version
FROM rust:1.78-bookworm as builder

# Install comprehensive system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libasound2-dev \
    libpulse-dev \
    libjack-dev \
    portaudio19-dev \
    build-essential \
    cmake \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY .cargo/ .cargo/
COPY crates/ crates/

# Build the application with optimizations for production
RUN cargo build --release --workspace --no-default-features --features "cli"

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libasound2 \
    libpulse0 \
    libjack0 \
    portaudio19-dev \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false voirs

# Create directories
RUN mkdir -p /app/models /app/audio /app/output /app/config && \
    chown -R voirs:voirs /app

# Copy built binaries from builder stage
COPY --from=builder /app/target/release/voirs-cli /usr/local/bin/voirs-cli
COPY --from=builder /app/target/release/libvoirs_ffi.so /usr/local/lib/ 2>/dev/null || true

# Set library path for FFI
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Copy models directory structure
COPY --chown=voirs:voirs models/ /app/models/

# Switch to non-root user
USER voirs

# Set working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD voirs-cli --version || exit 1

# Default command
CMD ["voirs-cli", "--help"]

# Expose common ports for API access (if enabled)
EXPOSE 8080 8443

# Labels
LABEL org.opencontainers.image.title="VoiRS"
LABEL org.opencontainers.image.description="Pure-Rust Neural Speech Synthesis Framework"
LABEL org.opencontainers.image.vendor="cool-japan"
LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/cool-japan/voirs"