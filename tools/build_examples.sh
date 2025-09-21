#!/bin/bash
#
# VoiRS Examples Build Script
# ===========================
#
# This script provides a comprehensive build and test system for VoiRS examples.
# It can be used locally for development or in CI/CD pipelines for validation.
#
# Usage:
#   ./build_examples.sh [OPTIONS]
#
# Options:
#   --help, -h              Show this help message
#   --build-only            Only build examples, don't test
#   --test-only             Only test examples, don't build
#   --clean                 Clean build artifacts first
#   --parallel N            Number of parallel jobs (default: number of CPUs)
#   --timeout SECONDS       Test timeout in seconds (default: 60)
#   --verbose               Enable verbose output
#   --report PATH           Generate detailed report
#   --ci                    CI mode (non-interactive, fail fast)
#   --examples PATTERN      Only build/test examples matching pattern
#   --category CATEGORY     Only build/test examples in category
#   --quick                 Quick mode (fast builds, basic tests)
#   --full                  Full mode (all optimizations, thorough tests)
#
# Examples:
#   ./build_examples.sh                    # Build and test all examples
#   ./build_examples.sh --clean --full     # Clean build with full testing
#   ./build_examples.sh --examples "*benchmark*"  # Only benchmark examples
#   ./build_examples.sh --ci --report ci_report.json  # CI mode with report
#
# Exit codes:
#   0 - Success
#   1 - Build failures
#   2 - Test failures  
#   3 - Setup/configuration errors

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="${EXAMPLES_DIR:-$(pwd)}"
CARGO_TOML="${EXAMPLES_DIR}/Cargo.toml"
TARGET_DIR="${EXAMPLES_DIR}/target"
REPORTS_DIR="${EXAMPLES_DIR}/build_reports"

# Default options
BUILD_ONLY=false
TEST_ONLY=false
CLEAN=false
PARALLEL_JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
TIMEOUT=60
VERBOSE=false
REPORT_PATH=""
CI_MODE=false
EXAMPLES_PATTERN="*"
CATEGORY=""
QUICK_MODE=false
FULL_MODE=false

# Colors for output (if terminal supports it)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    PURPLE='\033[0;35m'
    CYAN='\033[0;36m'
    WHITE='\033[1;37m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    PURPLE=''
    CYAN=''
    WHITE=''
    NC=''
fi

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $*"
    fi
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

# Show help
show_help() {
    cat << EOF
VoiRS Examples Build Script

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    --build-only            Only build examples, don't test
    --test-only             Only test examples, don't build  
    --clean                 Clean build artifacts first
    --parallel N            Number of parallel jobs (default: $PARALLEL_JOBS)
    --timeout SECONDS       Test timeout in seconds (default: $TIMEOUT)
    --verbose               Enable verbose output
    --report PATH           Generate detailed report
    --ci                    CI mode (non-interactive, fail fast)
    --examples PATTERN      Only build/test examples matching pattern
    --category CATEGORY     Only build/test examples in category
    --quick                 Quick mode (fast builds, basic tests)
    --full                  Full mode (all optimizations, thorough tests)

EXAMPLES:
    $0                                    # Build and test all examples
    $0 --clean --full                     # Clean build with full testing
    $0 --examples "*benchmark*"           # Only benchmark examples
    $0 --ci --report ci_report.json       # CI mode with report

CATEGORIES:
    performance             Performance and benchmarking examples
    testing                 Testing and validation examples
    realtime                Real-time and streaming examples
    spatial                 Spatial audio examples
    voice                   Voice cloning and customization examples
    production              Production deployment examples

EXIT CODES:
    0 - Success
    1 - Build failures
    2 - Test failures
    3 - Setup/configuration errors
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            --build-only)
                BUILD_ONLY=true
                shift
                ;;
            --test-only)
                TEST_ONLY=true
                shift
                ;;
            --clean)
                CLEAN=true
                shift
                ;;
            --parallel)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --report)
                REPORT_PATH="$2"
                shift 2
                ;;
            --ci)
                CI_MODE=true
                shift
                ;;
            --examples)
                EXAMPLES_PATTERN="$2"
                shift 2
                ;;
            --category)
                CATEGORY="$2"
                shift 2
                ;;
            --quick)
                QUICK_MODE=true
                shift
                ;;
            --full)
                FULL_MODE=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 3
                ;;
        esac
    done

    # Validate arguments
    if [[ "$BUILD_ONLY" == "true" && "$TEST_ONLY" == "true" ]]; then
        log_error "Cannot specify both --build-only and --test-only"
        exit 3
    fi

    if [[ "$QUICK_MODE" == "true" && "$FULL_MODE" == "true" ]]; then
        log_error "Cannot specify both --quick and --full"
        exit 3
    fi

    # Set up CI mode defaults
    if [[ "$CI_MODE" == "true" ]]; then
        set -e  # Exit on any error
        VERBOSE=true
        if [[ -z "$REPORT_PATH" ]]; then
            REPORT_PATH="ci_build_report_$(date +%Y%m%d_%H%M%S).json"
        fi
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if we're in the right directory
    if [[ ! -f "$CARGO_TOML" ]]; then
        log_error "Cargo.toml not found at $CARGO_TOML"
        log_error "Please run this script from the VoiRS examples directory"
        exit 3
    fi

    # Check Rust toolchain
    if ! command -v cargo >/dev/null 2>&1; then
        log_error "Cargo not found. Please install Rust toolchain"
        exit 3
    fi

    # Check Rust version
    local rust_version
    rust_version=$(rustc --version | cut -d' ' -f2)
    log_debug "Rust version: $rust_version"

    # Create reports directory
    mkdir -p "$REPORTS_DIR"

    # Check disk space (warn if less than 1GB)
    local available_space
    if command -v df >/dev/null 2>&1; then
        available_space=$(df "$EXAMPLES_DIR" | tail -1 | awk '{print $4}')
        if [[ "$available_space" -lt 1048576 ]]; then  # 1GB in KB
            log_warn "Low disk space: $(( available_space / 1024 ))MB available"
        fi
    fi

    log_success "Prerequisites check passed"
}

# Discover examples from Cargo.toml
discover_examples() {
    log_info "Discovering examples..."

    # Extract example names from Cargo.toml
    local examples=()
    while IFS= read -r line; do
        if [[ "$line" =~ name\ =\ \"([^\"]+)\" ]]; then
            local example_name="${BASH_REMATCH[1]}"
            
            # Apply pattern filter
            if [[ "$example_name" == $EXAMPLES_PATTERN ]]; then
                examples+=("$example_name")
            fi
        fi
    done < <(grep -A1 '\[\[example\]\]' "$CARGO_TOML")

    # Apply category filter if specified
    if [[ -n "$CATEGORY" ]]; then
        local filtered_examples=()
        for example in "${examples[@]}"; do
            case "$example" in
                *benchmark*|*performance*)
                    [[ "$CATEGORY" == "performance" ]] && filtered_examples+=("$example")
                    ;;
                *test*|*validation*)
                    [[ "$CATEGORY" == "testing" ]] && filtered_examples+=("$example")
                    ;;
                *streaming*|*realtime*)
                    [[ "$CATEGORY" == "realtime" ]] && filtered_examples+=("$example")
                    ;;
                *spatial*|*3d*)
                    [[ "$CATEGORY" == "spatial" ]] && filtered_examples+=("$example")
                    ;;
                *cloning*|*voice*)
                    [[ "$CATEGORY" == "voice" ]] && filtered_examples+=("$example")
                    ;;
                *production*|*monitoring*)
                    [[ "$CATEGORY" == "production" ]] && filtered_examples+=("$example")
                    ;;
            esac
        done
        examples=("${filtered_examples[@]}")
    fi

    if [[ ${#examples[@]} -eq 0 ]]; then
        log_error "No examples found matching criteria"
        exit 3
    fi

    log_info "Found ${#examples[@]} examples: ${examples[*]}"
    echo "${examples[@]}"
}

# Clean build artifacts
clean_artifacts() {
    log_info "Cleaning build artifacts..."
    
    cargo clean --manifest-path "$CARGO_TOML"
    
    # Clean generated files
    find "$EXAMPLES_DIR" -name "*.wav" -type f -delete 2>/dev/null || true
    find "$EXAMPLES_DIR" -name "*_report.json" -type f -delete 2>/dev/null || true
    find "$EXAMPLES_DIR" -name "*_metrics.txt" -type f -delete 2>/dev/null || true
    
    log_success "Artifacts cleaned"
}

# Build examples
build_examples() {
    local examples=("$@")
    log_info "Building ${#examples[@]} examples with $PARALLEL_JOBS parallel jobs..."

    local build_start_time
    build_start_time=$(date +%s)
    local failed_builds=()
    local successful_builds=()

    # Configure build options based on mode
    local build_args=("--release")
    if [[ "$QUICK_MODE" == "true" ]]; then
        build_args+=("--profile" "dev")  # Use dev profile for speed
    elif [[ "$FULL_MODE" == "true" ]]; then
        build_args+=("--profile" "release")
        export RUSTFLAGS="-C target-cpu=native"  # Optimize for current CPU
    fi

    # Build examples in parallel using xargs
    printf '%s\n' "${examples[@]}" | \
    xargs -n 1 -P "$PARALLEL_JOBS" -I {} bash -c "
        echo 'Building example: {}'
        start_time=\$(date +%s)
        if cargo build --example {} --manifest-path '$CARGO_TOML' ${build_args[*]} 2>&1; then
            end_time=\$(date +%s)
            duration=\$((end_time - start_time))
            echo \"✅ Built {} (\${duration}s)\"
        else
            echo \"❌ Failed to build {}\"
            exit 1
        fi
    " || {
        # Collect failed builds
        for example in "${examples[@]}"; do
            local binary_path="$TARGET_DIR/release/examples/$example"
            if [[ ! -f "$binary_path" ]]; then
                failed_builds+=("$example")
            else
                successful_builds+=("$example")
            fi
        done
    }

    # Re-check what actually built successfully
    successful_builds=()
    failed_builds=()
    for example in "${examples[@]}"; do
        local binary_path="$TARGET_DIR/release/examples/$example"
        if [[ -f "$binary_path" ]]; then
            successful_builds+=("$example")
        else
            failed_builds+=("$example")
        fi
    done

    local build_end_time
    build_end_time=$(date +%s)
    local build_duration=$((build_end_time - build_start_time))

    log_info "Build completed in ${build_duration}s"
    log_success "${#successful_builds[@]}/${#examples[@]} examples built successfully"

    if [[ ${#failed_builds[@]} -gt 0 ]]; then
        log_error "Failed builds: ${failed_builds[*]}"
        if [[ "$CI_MODE" == "true" ]]; then
            exit 1
        fi
    fi

    echo "${successful_builds[@]}"
}

# Test examples
test_examples() {
    local examples=("$@")
    log_info "Testing ${#examples[@]} examples..."

    local test_start_time
    test_start_time=$(date +%s)
    local failed_tests=()
    local successful_tests=()

    # Determine timeout based on mode
    local test_timeout="$TIMEOUT"
    if [[ "$QUICK_MODE" == "true" ]]; then
        test_timeout=$((TIMEOUT / 2))  # Shorter timeout for quick mode
    elif [[ "$FULL_MODE" == "true" ]]; then
        test_timeout=$((TIMEOUT * 2))  # Longer timeout for full mode
    fi

    # Test examples with limited parallelism to avoid resource contention
    local test_parallel=$((PARALLEL_JOBS / 2))
    [[ $test_parallel -lt 1 ]] && test_parallel=1

    printf '%s\n' "${examples[@]}" | \
    xargs -n 1 -P "$test_parallel" -I {} bash -c "
        echo 'Testing example: {}'
        start_time=\$(date +%s)
        
        # Run with timeout
        if timeout $test_timeout cargo run --example {} --release --manifest-path '$CARGO_TOML' >/dev/null 2>&1; then
            end_time=\$(date +%s)
            duration=\$((end_time - start_time))
            echo \"✅ Tested {} (\${duration}s)\"
        else
            echo \"❌ Test failed for {}\"
            exit 1
        fi
    " || {
        # This will capture failures, but we need to recheck individually
        true
    }

    # Check which tests actually passed by looking for generated files or using exit codes
    for example in "${examples[@]}"; do
        log_debug "Checking test result for $example"
        
        # Run individual test to check result
        if timeout "$test_timeout" cargo run --example "$example" --release --manifest-path "$CARGO_TOML" >/dev/null 2>&1; then
            successful_tests+=("$example")
        else
            failed_tests+=("$example")
        fi
    done

    local test_end_time
    test_end_time=$(date +%s)
    local test_duration=$((test_end_time - test_start_time))

    log_info "Testing completed in ${test_duration}s"
    log_success "${#successful_tests[@]}/${#examples[@]} examples tested successfully"

    if [[ ${#failed_tests[@]} -gt 0 ]]; then
        log_error "Failed tests: ${failed_tests[*]}"
        if [[ "$CI_MODE" == "true" ]]; then
            exit 2
        fi
    fi

    echo "${successful_tests[@]}"
}

# Generate simple report
generate_report() {
    local examples=("$@")
    local report_file="${REPORT_PATH:-$REPORTS_DIR/build_report_$(date +%Y%m%d_%H%M%S).txt}"
    
    log_info "Generating report: $report_file"

    cat > "$report_file" << EOF
VoiRS Examples Build Report
==========================
Generated: $(date)
Platform: $(uname -a)
Rust Version: $(rustc --version)

Configuration:
- Examples Directory: $EXAMPLES_DIR
- Parallel Jobs: $PARALLEL_JOBS
- Test Timeout: $TIMEOUT seconds
- Quick Mode: $QUICK_MODE
- Full Mode: $FULL_MODE
- CI Mode: $CI_MODE

Examples Processed: ${#examples[@]}
- Pattern: $EXAMPLES_PATTERN
- Category: ${CATEGORY:-"all"}

Results:
$(for example in "${examples[@]}"; do
    local binary_path="$TARGET_DIR/release/examples/$example"
    if [[ -f "$binary_path" ]]; then
        local size
        size=$(ls -lh "$binary_path" | awk '{print $5}')
        echo "✅ $example (binary: $size)"
    else
        echo "❌ $example (build failed)"
    fi
done)

Generated Files:
$(find "$EXAMPLES_DIR" -name "*.wav" -o -name "*_report.json" -o -name "*_metrics.txt" | head -20)

EOF

    log_success "Report saved to: $report_file"
}

# Main execution
main() {
    parse_args "$@"

    # Print banner
    if [[ "$CI_MODE" != "true" ]]; then
        echo -e "${CYAN}"
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║                    VoiRS Examples Build                     ║"
        echo "║              Comprehensive Build & Test System              ║"
        echo "╚══════════════════════════════════════════════════════════════╝"
        echo -e "${NC}"
    fi

    check_prerequisites

    # Clean if requested
    if [[ "$CLEAN" == "true" ]]; then
        clean_artifacts
    fi

    # Discover examples
    local examples
    examples=($(discover_examples))

    local final_examples=("${examples[@]}")

    # Build phase
    if [[ "$TEST_ONLY" != "true" ]]; then
        local built_examples
        built_examples=($(build_examples "${examples[@]}"))
        final_examples=("${built_examples[@]}")
    fi

    # Test phase
    if [[ "$BUILD_ONLY" != "true" ]] && [[ ${#final_examples[@]} -gt 0 ]]; then
        local tested_examples
        tested_examples=($(test_examples "${final_examples[@]}"))
        final_examples=("${tested_examples[@]}")
    fi

    # Generate report if requested
    if [[ -n "$REPORT_PATH" ]] || [[ "$CI_MODE" == "true" ]]; then
        generate_report "${final_examples[@]}"
    fi

    # Final summary
    log_info "Build script completed successfully"
    log_success "Processed ${#final_examples[@]} examples"
    
    if [[ "$CI_MODE" != "true" ]]; then
        echo -e "${GREEN}All done! 🎉${NC}"
    fi

    exit 0
}

# Run main function with all arguments
main "$@"