# VoiRS Evaluation Docker Image
# Multi-stage build for minimal production image

# Build stage
FROM rust:1.85-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 -U -s /bin/sh -d /app app

# Set working directory
WORKDIR /app

# Copy workspace Cargo files
COPY Cargo.toml Cargo.lock ./
COPY crates/voirs-evaluation/Cargo.toml ./crates/voirs-evaluation/

# Create dummy source files to cache dependencies
RUN mkdir -p crates/voirs-evaluation/src && \
    echo "fn main() {}" > crates/voirs-evaluation/src/main.rs && \
    echo "fn main() {}" > crates/voirs-evaluation/src/bin/voirs-eval.rs

# Build dependencies (cached layer)
RUN cargo build --release --bin voirs-eval && \
    rm -rf crates/voirs-evaluation/src

# Copy actual source code
COPY crates/voirs-evaluation/src ./crates/voirs-evaluation/src
COPY crates/voirs-evaluation/benches ./crates/voirs-evaluation/benches
COPY crates/voirs-evaluation/examples ./crates/voirs-evaluation/examples

# Copy other required crates (adjust as needed)
COPY crates/voirs-sdk ./crates/voirs-sdk
COPY crates/voirs-recognizer ./crates/voirs-recognizer
COPY crates/voirs-g2p ./crates/voirs-g2p

# Build the application
RUN cargo build --release --bin voirs-eval && \
    strip /app/target/release/voirs-eval

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 -U -s /bin/sh -d /app app

# Create necessary directories
RUN mkdir -p /data /cache /tmp /state /etc/voirs && \
    chown -R app:app /data /cache /tmp /state /etc/voirs

# Copy binary from builder
COPY --from=builder /app/target/release/voirs-eval /usr/local/bin/voirs-eval
RUN chmod +x /usr/local/bin/voirs-eval

# Switch to app user
USER app

# Set working directory
WORKDIR /app

# Expose ports
EXPOSE 8080 9000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/usr/local/bin/voirs-eval", "health"]

# Default command
ENTRYPOINT ["/usr/local/bin/voirs-eval"]
CMD ["worker"]

# Metadata
LABEL org.opencontainers.image.title="VoiRS Evaluation" \
      org.opencontainers.image.description="Distributed evaluation system for VoiRS speech synthesis" \
      org.opencontainers.image.version="0.1.0-alpha.3" \
      org.opencontainers.image.authors="VoiRS Team" \
      org.opencontainers.image.url="https://github.com/cool-japan/voirs" \
      org.opencontainers.image.source="https://github.com/cool-japan/voirs" \
      org.opencontainers.image.licenses="Apache-2.0"
