# Multi-stage Dockerfile for VoiRS Feedback System
# This provides containerized deployment with optimized production builds

# Stage 1: Build environment
FROM rust:1.75-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    musl-dev \
    pkgconfig \
    openssl-dev \
    sqlite-dev \
    postgresql-dev \
    gcc \
    g++ \
    make \
    libc-dev

WORKDIR /usr/src/app

# Copy Cargo files for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY crates/voirs-feedback/Cargo.toml ./crates/voirs-feedback/

# Create dummy source files to cache dependencies
RUN mkdir -p crates/voirs-feedback/src && \
    echo "fn main() {}" > crates/voirs-feedback/src/main.rs && \
    echo "// dummy" > crates/voirs-feedback/src/lib.rs

# Build dependencies (this will be cached if Cargo files don't change)
RUN cargo build --release --manifest-path crates/voirs-feedback/Cargo.toml

# Remove dummy files and copy real source
RUN rm -rf crates/voirs-feedback/src
COPY crates/voirs-feedback/src ./crates/voirs-feedback/src
COPY crates/voirs-feedback/tests ./crates/voirs-feedback/tests
COPY crates/voirs-feedback/examples ./crates/voirs-feedback/examples

# Copy other required crates (adjust paths as needed)
COPY crates/voirs-sdk ./crates/voirs-sdk
COPY crates/voirs-recognizer ./crates/voirs-recognizer
COPY crates/voirs-evaluation ./crates/voirs-evaluation

# Build the application
RUN cargo build --release --manifest-path crates/voirs-feedback/Cargo.toml

# Stage 2: Runtime environment
FROM alpine:3.18

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    libgcc \
    openssl \
    sqlite \
    postgresql-client

# Create app user for security
RUN addgroup -g 1000 voirs && \
    adduser -D -s /bin/sh -u 1000 -G voirs voirs

WORKDIR /app

# Copy the compiled binary
COPY --from=builder /usr/src/app/target/release/voirs-feedback /app/
COPY --from=builder /usr/src/app/crates/voirs-feedback/examples /app/examples

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config && \
    chown -R voirs:voirs /app

# Copy configuration files
COPY crates/voirs-feedback/config /app/config

USER voirs

# Expose ports
EXPOSE 8080 8081 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Environment variables
ENV RUST_LOG=info
ENV VOIRS_CONFIG_PATH=/app/config
ENV VOIRS_DATA_PATH=/app/data
ENV VOIRS_LOG_PATH=/app/logs

# Default command
CMD ["./voirs-feedback"]