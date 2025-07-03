# Multi-stage build for VoiRS
FROM rust:1.70-bookworm as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY .cargo/ .cargo/
COPY crates/ crates/

# Build the application
RUN cargo build --release --bin voirs

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libasound2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false voirs

# Create directories
RUN mkdir -p /app/models /app/data && \
    chown -R voirs:voirs /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/voirs /usr/local/bin/voirs

# Copy models directory structure
COPY --chown=voirs:voirs models/ /app/models/

# Switch to non-root user
USER voirs

# Set working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD voirs --version || exit 1

# Default command
CMD ["voirs", "--help"]

# Labels
LABEL org.opencontainers.image.title="VoiRS"
LABEL org.opencontainers.image.description="Pure-Rust Neural Speech Synthesis Framework"
LABEL org.opencontainers.image.vendor="cool-japan"
LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/cool-japan/voirs"