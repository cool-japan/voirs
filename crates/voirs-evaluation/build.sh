#!/bin/bash
# VoiRS Evaluation - Cross-platform Build Script (Unix)
# 
# This script provides automated building and testing for Unix-like systems
# (Linux, macOS, FreeBSD, etc.)

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="voirs-evaluation"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to detect platform
detect_platform() {
    case "$(uname -s)" in
        Darwin*)    echo "macos" ;;
        Linux*)     echo "linux" ;;
        FreeBSD*)   echo "freebsd" ;;
        *)          echo "unknown" ;;
    esac
}

# Function to detect architecture
detect_arch() {
    case "$(uname -m)" in
        x86_64)     echo "x86_64" ;;
        arm64)      echo "aarch64" ;;
        aarch64)    echo "aarch64" ;;
        armv7*)     echo "armv7" ;;
        *)          echo "$(uname -m)" ;;
    esac
}

# Function to check dependencies
check_dependencies() {
    log_info "Checking build dependencies..."
    
    # Check Rust
    if ! command -v rustc &> /dev/null; then
        log_error "Rust is not installed. Please install Rust from https://rustup.rs/"
        exit 1
    fi
    
    # Check Cargo
    if ! command -v cargo &> /dev/null; then
        log_error "Cargo is not installed. Please install Rust toolchain."
        exit 1
    fi
    
    # Check Python (for Python bindings)
    if command -v python3 &> /dev/null; then
        log_info "Python 3 found: $(python3 --version)"
    else
        log_warning "Python 3 not found. Python bindings will be disabled."
    fi
    
    # Platform-specific dependencies
    local platform=$(detect_platform)
    case "$platform" in
        "macos")
            # Check for Xcode Command Line Tools
            if ! xcode-select -p &> /dev/null; then
                log_warning "Xcode Command Line Tools not found. Some features may not work."
            fi
            ;;
        "linux")
            # Check for common build tools
            if ! command -v pkg-config &> /dev/null; then
                log_warning "pkg-config not found. Install it with your package manager."
            fi
            ;;
    esac
    
    log_success "Dependency check completed"
}

# Function to setup environment
setup_environment() {
    log_info "Setting up build environment..."
    
    # Set environment variables
    export CARGO_TARGET_DIR="$CARGO_TARGET_DIR"
    export RUST_BACKTRACE=1
    
    # Platform-specific setup
    local platform=$(detect_platform)
    case "$platform" in
        "macos")
            # Set SDK path for macOS
            export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
            ;;
        "linux")
            # Set library path for Linux
            export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
            ;;
    esac
    
    log_info "Platform: $platform"
    log_info "Architecture: $(detect_arch)"
    log_info "Rust version: $(rustc --version)"
    log_info "Cargo version: $(cargo --version)"
}

# Function to clean build artifacts
clean() {
    log_info "Cleaning build artifacts..."
    cargo clean
    rm -rf target/doc
    log_success "Clean completed"
}

# Function to build the project
build() {
    local release_flag=""
    local features=""
    
    # Parse build options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --release)
                release_flag="--release"
                shift
                ;;
            --features)
                features="--features $2"
                shift 2
                ;;
            --all-features)
                features="--all-features"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    log_info "Building $PROJECT_NAME..."
    log_info "Build flags: $release_flag $features"
    
    # Build the main library
    cargo build $release_flag $features
    
    # Build examples
    log_info "Building examples..."
    cargo build --examples $release_flag $features
    
    # Build benchmarks
    log_info "Building benchmarks..."
    cargo build --benches $release_flag $features
    
    log_success "Build completed successfully"
}

# Function to run tests
test() {
    local test_flags=""
    
    # Parse test options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --release)
                test_flags="--release"
                shift
                ;;
            --features)
                test_flags="$test_flags --features $2"
                shift 2
                ;;
            --all-features)
                test_flags="$test_flags --all-features"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    log_info "Running tests..."
    
    # Run unit tests
    cargo test $test_flags
    
    # Run integration tests
    log_info "Running integration tests..."
    cargo test --test integration_tests $test_flags
    
    # Run doc tests
    log_info "Running documentation tests..."
    cargo test --doc $test_flags
    
    log_success "All tests passed"
}

# Function to run benchmarks
bench() {
    log_info "Running benchmarks..."
    
    # Run all benchmarks
    cargo bench --all
    
    log_success "Benchmarks completed"
}

# Function to generate documentation
docs() {
    log_info "Generating documentation..."
    
    # Generate docs
    cargo doc --all-features --no-deps
    
    # Open docs if requested
    if [[ "$1" == "--open" ]]; then
        cargo doc --all-features --no-deps --open
    fi
    
    log_success "Documentation generated"
}

# Function to check code quality
check() {
    log_info "Running code quality checks..."
    
    # Check formatting
    log_info "Checking code formatting..."
    cargo fmt -- --check
    
    # Run clippy
    log_info "Running Clippy lints..."
    cargo clippy --all-features -- -D warnings
    
    # Check for unused dependencies
    if command -v cargo-udeps &> /dev/null; then
        log_info "Checking for unused dependencies..."
        cargo +nightly udeps --all-features
    else
        log_warning "cargo-udeps not installed. Skipping unused dependency check."
    fi
    
    log_success "Code quality checks passed"
}

# Function to install the package
install() {
    log_info "Installing $PROJECT_NAME..."
    
    cargo install --path . --all-features
    
    log_success "Installation completed"
}

# Function to build Python bindings
build_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required for building Python bindings"
        exit 1
    fi
    
    log_info "Building Python bindings..."
    
    # Build with Python feature
    cargo build --release --features python
    
    # Build Python wheel
    if command -v maturin &> /dev/null; then
        maturin build --release --features python
    else
        log_warning "maturin not found. Install with: pip install maturin"
    fi
    
    log_success "Python bindings built"
}

# Function to run full CI pipeline
ci() {
    log_info "Running full CI pipeline..."
    
    check_dependencies
    setup_environment
    clean
    check
    build --all-features
    test --all-features
    docs
    
    log_success "CI pipeline completed successfully"
}

# Function to show help
show_help() {
    cat << EOF
VoiRS Evaluation Build Script

USAGE:
    ./build.sh [COMMAND] [OPTIONS]

COMMANDS:
    build           Build the project
    test            Run tests
    bench           Run benchmarks
    docs            Generate documentation
    check           Run code quality checks
    clean           Clean build artifacts
    install         Install the package
    python          Build Python bindings
    ci              Run full CI pipeline
    help            Show this help message

BUILD OPTIONS:
    --release       Build in release mode
    --features      Specify features to enable
    --all-features  Enable all features

EXAMPLES:
    ./build.sh build --release --all-features
    ./build.sh test --features python
    ./build.sh docs --open
    ./build.sh ci

EOF
}

# Main script logic
main() {
    cd "$SCRIPT_DIR"
    
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    local command="$1"
    shift
    
    case "$command" in
        "build")
            check_dependencies
            setup_environment
            build "$@"
            ;;
        "test")
            check_dependencies
            setup_environment
            test "$@"
            ;;
        "bench")
            check_dependencies
            setup_environment
            bench
            ;;
        "docs")
            check_dependencies
            setup_environment
            docs "$@"
            ;;
        "check")
            check_dependencies
            setup_environment
            check
            ;;
        "clean")
            clean
            ;;
        "install")
            check_dependencies
            setup_environment
            install
            ;;
        "python")
            check_dependencies
            setup_environment
            build_python
            ;;
        "ci")
            ci
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"