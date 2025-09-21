#!/bin/bash

# VoiRS FFI Test Runner
# Optimized test execution with environment-aware synthesis test handling

set -e

# Default values
RUN_SYNTHESIS_TESTS=false
VERBOSE=false
PACKAGE="voirs-ffi"
TEST_PROFILE="default"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -s, --synthesis     Enable synthesis tests (slow, requires models)
    -v, --verbose       Enable verbose output
    -p, --package       Package to test (default: voirs-ffi)
    --profile          Test profile (default: default)
    -h, --help         Show this help message

Examples:
    $0                  # Run fast tests only (recommended for CI)
    $0 -s               # Run all tests including synthesis
    $0 -v               # Run with verbose output
    $0 -p voirs-sdk     # Test different package

Environment Variables:
    VOIRS_SKIP_SYNTHESIS_TESTS  Set to skip synthesis tests
    CI                          Auto-detected CI environment
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--synthesis)
            RUN_SYNTHESIS_TESTS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -p|--package)
            PACKAGE="$2"
            shift 2
            ;;
        --profile)
            TEST_PROFILE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Detect CI environment
if [ -n "$CI" ] || [ -n "$GITHUB_ACTIONS" ] || [ -n "$GITLAB_CI" ] || [ -n "$TRAVIS" ]; then
    log_info "CI environment detected, using optimized test configuration"
    RUN_SYNTHESIS_TESTS=false
fi

# Set synthesis test environment variable
if [ "$RUN_SYNTHESIS_TESTS" = false ]; then
    export VOIRS_SKIP_SYNTHESIS_TESTS=1
    log_info "Synthesis tests will be skipped for faster execution"
else
    unset VOIRS_SKIP_SYNTHESIS_TESTS
    log_warn "Synthesis tests enabled - this will be slow and requires models"
fi

# Build the cargo command
CARGO_CMD="cargo nextest run --no-fail-fast"
if [ "$PACKAGE" != "all" ]; then
    CARGO_CMD="$CARGO_CMD -p $PACKAGE"
fi

if [ "$TEST_PROFILE" != "default" ]; then
    CARGO_CMD="$CARGO_CMD --profile $TEST_PROFILE"
fi

if [ "$VERBOSE" = true ]; then
    CARGO_CMD="$CARGO_CMD --verbose"
fi

# Display configuration
log_info "Test Configuration:"
log_info "  Package: $PACKAGE"
log_info "  Profile: $TEST_PROFILE"
log_info "  Synthesis Tests: $RUN_SYNTHESIS_TESTS"
log_info "  Verbose: $VERBOSE"
log_info "  Command: $CARGO_CMD"
echo

# Run tests
log_info "Starting test execution..."
start_time=$(date +%s)

if eval "$CARGO_CMD"; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    log_info "Tests completed successfully in ${duration}s"
    
    # Show test summary
    if [ "$RUN_SYNTHESIS_TESTS" = false ]; then
        log_info "Fast test mode: Synthesis tests were skipped"
        log_info "To run synthesis tests: $0 --synthesis"
    else
        log_info "Full test mode: All tests including synthesis were executed"
    fi
else
    log_error "Tests failed!"
    exit 1
fi