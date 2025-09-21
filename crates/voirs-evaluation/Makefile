# VoiRS Evaluation - Cross-platform Makefile
#
# This Makefile provides convenient build targets for the VoiRS evaluation crate
# Compatible with GNU Make and BSD Make

# Project configuration
PROJECT_NAME := voirs-evaluation
CARGO_TARGET_DIR ?= target
CARGO_FLAGS := 
FEATURES := 
RELEASE_FLAGS := 

# Detect platform
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
endif
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
endif
ifeq ($(UNAME_S),FreeBSD)
    PLATFORM := freebsd
endif
ifdef OS
    ifeq ($(OS),Windows_NT)
        PLATFORM := windows
    endif
endif

# Detect architecture
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
    ARCH := x86_64
endif
ifeq ($(UNAME_M),arm64)
    ARCH := aarch64
endif
ifeq ($(UNAME_M),aarch64)
    ARCH := aarch64
endif

# Colors for output (if terminal supports it)
ifneq (,$(shell tput setaf 1 2>/dev/null))
    RED := $(shell tput setaf 1)
    GREEN := $(shell tput setaf 2)
    YELLOW := $(shell tput setaf 3)
    BLUE := $(shell tput setaf 4)
    RESET := $(shell tput sgr0)
else
    RED := 
    GREEN := 
    YELLOW := 
    BLUE := 
    RESET := 
endif

# Default target
.DEFAULT_GOAL := help

# Export environment variables
export CARGO_TARGET_DIR
export RUST_BACKTRACE=1

# Phony targets
.PHONY: help info clean build build-release build-all test test-release bench docs check \
        install python ci format clippy examples integration-tests doc-tests \
        dev-setup platform-info

# Help target
help:
	@echo "$(BLUE)VoiRS Evaluation Build System$(RESET)"
	@echo ""
	@echo "$(GREEN)Available targets:$(RESET)"
	@echo "  help                    Show this help message"
	@echo "  info                    Show build environment information"
	@echo "  clean                   Clean build artifacts"
	@echo "  build                   Build the project (debug mode)"
	@echo "  build-release           Build the project (release mode)"
	@echo "  build-all               Build with all features enabled"
	@echo "  test                    Run all tests"
	@echo "  test-release            Run tests in release mode"
	@echo "  bench                   Run benchmarks"
	@echo "  docs                    Generate documentation"
	@echo "  check                   Run code quality checks"
	@echo "  format                  Format code"
	@echo "  clippy                  Run Clippy lints"
	@echo "  install                 Install the package"
	@echo "  python                  Build Python bindings"
	@echo "  ci                      Run full CI pipeline"
	@echo "  dev-setup               Setup development environment"
	@echo "  platform-info           Show platform information"
	@echo ""
	@echo "$(GREEN)Variables:$(RESET)"
	@echo "  FEATURES=<features>     Specify features to enable"
	@echo "  CARGO_FLAGS=<flags>     Additional cargo flags"
	@echo ""
	@echo "$(GREEN)Examples:$(RESET)"
	@echo "  make build FEATURES=python"
	@echo "  make test-release"
	@echo "  make ci"

# Info target
info: platform-info
	@echo "$(BLUE)[INFO]$(RESET) Build environment information"
	@echo "Platform: $(PLATFORM)"
	@echo "Architecture: $(ARCH)"
	@echo "Cargo target directory: $(CARGO_TARGET_DIR)"
	@echo "Rust version: $$(rustc --version 2>/dev/null || echo 'Not found')"
	@echo "Cargo version: $$(cargo --version 2>/dev/null || echo 'Not found')"

# Platform info
platform-info:
	@echo "$(BLUE)[INFO]$(RESET) Platform: $(PLATFORM) ($(ARCH))"

# Clean target
clean:
	@echo "$(BLUE)[INFO]$(RESET) Cleaning build artifacts..."
	cargo clean
	@if [ -d "$(CARGO_TARGET_DIR)/doc" ]; then rm -rf "$(CARGO_TARGET_DIR)/doc"; fi
	@echo "$(GREEN)[SUCCESS]$(RESET) Clean completed"

# Build targets
build:
	@echo "$(BLUE)[INFO]$(RESET) Building $(PROJECT_NAME) (debug mode)..."
	cargo build $(CARGO_FLAGS) $(if $(FEATURES),--features $(FEATURES))
	@echo "$(GREEN)[SUCCESS]$(RESET) Build completed"

build-release:
	@echo "$(BLUE)[INFO]$(RESET) Building $(PROJECT_NAME) (release mode)..."
	cargo build --release $(CARGO_FLAGS) $(if $(FEATURES),--features $(FEATURES))
	@echo "$(GREEN)[SUCCESS]$(RESET) Release build completed"

build-all:
	@echo "$(BLUE)[INFO]$(RESET) Building $(PROJECT_NAME) with all features..."
	cargo build --all-features $(CARGO_FLAGS)
	@echo "$(GREEN)[SUCCESS]$(RESET) Build with all features completed"

# Examples target
examples:
	@echo "$(BLUE)[INFO]$(RESET) Building examples..."
	cargo build --examples $(CARGO_FLAGS) $(if $(FEATURES),--features $(FEATURES))
	@echo "$(GREEN)[SUCCESS]$(RESET) Examples built"

# Test targets
test:
	@echo "$(BLUE)[INFO]$(RESET) Running tests..."
	cargo test $(CARGO_FLAGS) $(if $(FEATURES),--features $(FEATURES))
	@echo "$(GREEN)[SUCCESS]$(RESET) All tests passed"

test-release:
	@echo "$(BLUE)[INFO]$(RESET) Running tests (release mode)..."
	cargo test --release $(CARGO_FLAGS) $(if $(FEATURES),--features $(FEATURES))
	@echo "$(GREEN)[SUCCESS]$(RESET) All release tests passed"

integration-tests:
	@echo "$(BLUE)[INFO]$(RESET) Running integration tests..."
	cargo test --test integration_tests $(CARGO_FLAGS) $(if $(FEATURES),--features $(FEATURES))
	@echo "$(GREEN)[SUCCESS]$(RESET) Integration tests passed"

doc-tests:
	@echo "$(BLUE)[INFO]$(RESET) Running documentation tests..."
	cargo test --doc $(CARGO_FLAGS) $(if $(FEATURES),--features $(FEATURES))
	@echo "$(GREEN)[SUCCESS]$(RESET) Documentation tests passed"

# Benchmark target
bench:
	@echo "$(BLUE)[INFO]$(RESET) Running benchmarks..."
	cargo bench --all
	@echo "$(GREEN)[SUCCESS]$(RESET) Benchmarks completed"

# Documentation target
docs:
	@echo "$(BLUE)[INFO]$(RESET) Generating documentation..."
	cargo doc --all-features --no-deps
	@echo "$(GREEN)[SUCCESS]$(RESET) Documentation generated"

# Code quality targets
check: format clippy
	@echo "$(GREEN)[SUCCESS]$(RESET) All code quality checks passed"

format:
	@echo "$(BLUE)[INFO]$(RESET) Formatting code..."
	cargo fmt
	@echo "$(GREEN)[SUCCESS]$(RESET) Code formatted"

clippy:
	@echo "$(BLUE)[INFO]$(RESET) Running Clippy lints..."
	cargo clippy --all-features -- -D warnings
	@echo "$(GREEN)[SUCCESS]$(RESET) Clippy checks passed"

# Install target
install:
	@echo "$(BLUE)[INFO]$(RESET) Installing $(PROJECT_NAME)..."
	cargo install --path . --all-features
	@echo "$(GREEN)[SUCCESS]$(RESET) Installation completed"

# Python bindings target
python:
	@echo "$(BLUE)[INFO]$(RESET) Building Python bindings..."
	@if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then \
		echo "$(RED)[ERROR]$(RESET) Python is required for building Python bindings"; \
		exit 1; \
	fi
	cargo build --release --features python
	@if command -v maturin >/dev/null 2>&1; then \
		maturin build --release --features python; \
	else \
		echo "$(YELLOW)[WARNING]$(RESET) maturin not found. Install with: pip install maturin"; \
	fi
	@echo "$(GREEN)[SUCCESS]$(RESET) Python bindings built"

# Development setup
dev-setup:
	@echo "$(BLUE)[INFO]$(RESET) Setting up development environment..."
	@# Install rustfmt and clippy if not already installed
	@if ! command -v rustfmt >/dev/null 2>&1; then \
		echo "Installing rustfmt..."; \
		rustup component add rustfmt; \
	fi
	@if ! command -v cargo-clippy >/dev/null 2>&1; then \
		echo "Installing clippy..."; \
		rustup component add clippy; \
	fi
	@# Install additional development tools
	@echo "Consider installing additional development tools:"
	@echo "  cargo install cargo-udeps  # Find unused dependencies"
	@echo "  cargo install cargo-audit  # Security audit"
	@echo "  cargo install cargo-watch  # File watching"
	@echo "$(GREEN)[SUCCESS]$(RESET) Development environment setup completed"

# CI pipeline
ci: clean check build-all test docs
	@echo "$(GREEN)[SUCCESS]$(RESET) CI pipeline completed successfully"

# Quick development cycle
dev: format clippy test
	@echo "$(GREEN)[SUCCESS]$(RESET) Development cycle completed"

# Watch target (requires cargo-watch)
watch:
	@if command -v cargo-watch >/dev/null 2>&1; then \
		echo "$(BLUE)[INFO]$(RESET) Starting file watcher..."; \
		cargo watch -x "build" -x "test"; \
	else \
		echo "$(RED)[ERROR]$(RESET) cargo-watch not installed. Install with: cargo install cargo-watch"; \
		exit 1; \
	fi

# Security audit (requires cargo-audit)
audit:
	@if command -v cargo-audit >/dev/null 2>&1; then \
		echo "$(BLUE)[INFO]$(RESET) Running security audit..."; \
		cargo audit; \
		echo "$(GREEN)[SUCCESS]$(RESET) Security audit completed"; \
	else \
		echo "$(YELLOW)[WARNING]$(RESET) cargo-audit not installed. Install with: cargo install cargo-audit"; \
	fi

# Check for unused dependencies (requires cargo-udeps)
unused-deps:
	@if command -v cargo-udeps >/dev/null 2>&1; then \
		echo "$(BLUE)[INFO]$(RESET) Checking for unused dependencies..."; \
		cargo +nightly udeps --all-features; \
		echo "$(GREEN)[SUCCESS]$(RESET) Unused dependency check completed"; \
	else \
		echo "$(YELLOW)[WARNING]$(RESET) cargo-udeps not installed. Install with: cargo install cargo-udeps"; \
	fi

# Profile build
profile:
	@echo "$(BLUE)[INFO]$(RESET) Building with profiling enabled..."
	RUSTFLAGS="-C force-frame-pointers=yes" cargo build --release
	@echo "$(GREEN)[SUCCESS]$(RESET) Profile build completed"

# Size optimization build
size-opt:
	@echo "$(BLUE)[INFO]$(RESET) Building for size optimization..."
	RUSTFLAGS="-C opt-level=z -C target-cpu=native" cargo build --release
	@echo "$(GREEN)[SUCCESS]$(RESET) Size-optimized build completed"

# Performance optimization build
perf-opt:
	@echo "$(BLUE)[INFO]$(RESET) Building for performance optimization..."
	RUSTFLAGS="-C opt-level=3 -C target-cpu=native" cargo build --release
	@echo "$(GREEN)[SUCCESS]$(RESET) Performance-optimized build completed"