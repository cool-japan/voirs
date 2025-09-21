#!/bin/bash
# VoiRS Evaluation - Comprehensive Test Runner
#
# This script runs the complete test suite including unit tests, integration tests,
# benchmarks, and quality checks. Designed for local development and CI environments.

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default options
RUN_BENCHMARKS=false
RUN_FUZZING=false
RUN_MEMORY_TESTS=false
VERBOSE=false
FEATURES="all-features"
PROFILE="debug"
PARALLEL_JOBS=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions with timestamps
log_info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] [INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] [SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] [WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] [ERROR]${NC} $1"
}

log_section() {
    echo -e "${PURPLE}[$(date +'%H:%M:%S')] [SECTION]${NC} $1"
}

# Function to show help
show_help() {
    cat << EOF
VoiRS Evaluation Test Runner

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --benchmarks        Run performance benchmarks
    --fuzzing          Run fuzzing tests  
    --memory-tests     Run memory leak detection tests
    --verbose          Enable verbose output
    --features <feat>  Specify features to test (default: all-features)
    --profile <prof>   Build profile: debug or release (default: debug)
    --jobs <n>         Number of parallel test jobs
    --help             Show this help message

EXAMPLES:
    $0                                    # Run basic test suite
    $0 --benchmarks --verbose            # Run with benchmarks and verbose output
    $0 --features python --profile release  # Test Python features in release mode
    $0 --fuzzing --memory-tests          # Run extended testing

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmarks)
            RUN_BENCHMARKS=true
            shift
            ;;
        --fuzzing)
            RUN_FUZZING=true
            shift
            ;;
        --memory-tests)
            RUN_MEMORY_TESTS=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --features)
            FEATURES="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --jobs)
            PARALLEL_JOBS="-j $2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate profile
if [[ "$PROFILE" != "debug" && "$PROFILE" != "release" ]]; then
    log_error "Invalid profile: $PROFILE. Use 'debug' or 'release'"
    exit 1
fi

# Setup cargo flags
CARGO_FLAGS=""
if [[ "$PROFILE" == "release" ]]; then
    CARGO_FLAGS="--release"
fi

if [[ "$FEATURES" == "all-features" ]]; then
    FEATURE_FLAGS="--all-features"
elif [[ -n "$FEATURES" ]]; then
    FEATURE_FLAGS="--features $FEATURES"
else
    FEATURE_FLAGS=""
fi

VERBOSE_FLAGS=""
if [[ "$VERBOSE" == "true" ]]; then
    VERBOSE_FLAGS="--verbose"
fi

# Test results tracking
PASSED_TESTS=()
FAILED_TESTS=()
SKIPPED_TESTS=()
START_TIME=$(date +%s)

# Function to run a test section
run_test_section() {
    local section_name="$1"
    local test_command="$2"
    local is_optional="${3:-false}"
    
    log_section "Running $section_name"
    
    local section_start=$(date +%s)
    
    if eval "$test_command"; then
        local section_end=$(date +%s)
        local section_duration=$((section_end - section_start))
        log_success "$section_name completed in ${section_duration}s"
        PASSED_TESTS+=("$section_name")
        return 0
    else
        local section_end=$(date +%s)
        local section_duration=$((section_end - section_start))
        if [[ "$is_optional" == "true" ]]; then
            log_warning "$section_name failed (optional) in ${section_duration}s"
            SKIPPED_TESTS+=("$section_name")
            return 0
        else
            log_error "$section_name failed in ${section_duration}s"
            FAILED_TESTS+=("$section_name")
            return 1
        fi
    fi
}

# Change to project directory
cd "$PROJECT_DIR"

echo "🚀 VoiRS Evaluation Comprehensive Test Suite"
echo "============================================="
echo "Configuration:"
echo "  Features: $FEATURES"
echo "  Profile: $PROFILE"
echo "  Benchmarks: $RUN_BENCHMARKS"
echo "  Fuzzing: $RUN_FUZZING"
echo "  Memory Tests: $RUN_MEMORY_TESTS"
echo "  Verbose: $VERBOSE"
echo "  Parallel Jobs: ${PARALLEL_JOBS:-auto}"
echo ""

# Environment information
log_info "Environment Information:"
echo "  Rust Version: $(rustc --version)"
echo "  Cargo Version: $(cargo --version)"
echo "  Platform: $(uname -s) $(uname -m)"
echo "  CPU Cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'unknown')"
echo ""

# ============================================================================
# Phase 1: Code Quality Checks
# ============================================================================

log_section "Phase 1: Code Quality Checks"

run_test_section "Code Formatting Check" \
    "cargo fmt --all -- --check $VERBOSE_FLAGS"

run_test_section "Clippy Lints" \
    "cargo clippy --all-targets $FEATURE_FLAGS $VERBOSE_FLAGS -- -D warnings"

# ============================================================================
# Phase 2: Build Verification
# ============================================================================

log_section "Phase 2: Build Verification"

run_test_section "Main Library Build" \
    "cargo build $CARGO_FLAGS $FEATURE_FLAGS $VERBOSE_FLAGS $PARALLEL_JOBS"

run_test_section "Examples Build" \
    "cargo build --examples $CARGO_FLAGS $FEATURE_FLAGS $VERBOSE_FLAGS $PARALLEL_JOBS"

run_test_section "Benchmarks Build" \
    "cargo build --benches $CARGO_FLAGS $FEATURE_FLAGS $VERBOSE_FLAGS $PARALLEL_JOBS"

run_test_section "Documentation Build" \
    "cargo doc $FEATURE_FLAGS --no-deps $VERBOSE_FLAGS"

# ============================================================================
# Phase 3: Core Testing 
# ============================================================================

log_section "Phase 3: Core Testing"

run_test_section "Unit Tests" \
    "cargo test --lib $CARGO_FLAGS $FEATURE_FLAGS $VERBOSE_FLAGS $PARALLEL_JOBS"

run_test_section "Integration Tests" \
    "cargo test --test integration_tests $CARGO_FLAGS $FEATURE_FLAGS $VERBOSE_FLAGS $PARALLEL_JOBS"

run_test_section "Documentation Tests" \
    "cargo test --doc $CARGO_FLAGS $FEATURE_FLAGS $VERBOSE_FLAGS $PARALLEL_JOBS"

# ============================================================================
# Phase 4: Specialized Testing
# ============================================================================

log_section "Phase 4: Specialized Testing"

# Regression tests
run_test_section "Regression Tests" \
    "cargo test --test regression_tests $CARGO_FLAGS $FEATURE_FLAGS $VERBOSE_FLAGS"

# Human perception validation
run_test_section "Human Perception Validation" \
    "cargo test --test human_perception_validation $CARGO_FLAGS $FEATURE_FLAGS $VERBOSE_FLAGS"

# Metric validation
run_test_section "Metric Validation" \
    "cargo test --test metric_validation $CARGO_FLAGS $FEATURE_FLAGS $VERBOSE_FLAGS"

# Statistical significance validation
run_test_section "Statistical Significance Validation" \
    "cargo test --test statistical_significance_validation $CARGO_FLAGS $FEATURE_FLAGS $VERBOSE_FLAGS"

# Performance monitoring tests
run_test_section "Performance Monitoring Tests" \
    "cargo test --test performance_regression_monitoring $CARGO_FLAGS $FEATURE_FLAGS $VERBOSE_FLAGS"

# ============================================================================
# Phase 5: Optional Extended Testing
# ============================================================================

if [[ "$RUN_BENCHMARKS" == "true" ]]; then
    log_section "Phase 5a: Performance Benchmarks"
    
    run_test_section "Evaluation Metrics Benchmark" \
        "cargo bench --bench evaluation_metrics $FEATURE_FLAGS $VERBOSE_FLAGS" \
        "true"
    
    run_test_section "Memory Benchmark" \
        "cargo bench --bench memory_benchmark $FEATURE_FLAGS $VERBOSE_FLAGS" \
        "true"
    
    run_test_section "Optimization Validation Benchmark" \
        "cargo bench --bench optimization_validation $FEATURE_FLAGS $VERBOSE_FLAGS" \
        "true"
fi

if [[ "$RUN_FUZZING" == "true" ]]; then
    log_section "Phase 5b: Fuzzing Tests"
    
    run_test_section "Fuzzing Tests" \
        "cargo test --test fuzzing $CARGO_FLAGS $FEATURE_FLAGS $VERBOSE_FLAGS -- --ignored" \
        "true"
fi

if [[ "$RUN_MEMORY_TESTS" == "true" ]]; then
    log_section "Phase 5c: Memory Tests"
    
    run_test_section "Memory Leak Tests" \
        "cargo test --test memory_tests $CARGO_FLAGS $FEATURE_FLAGS $VERBOSE_FLAGS" \
        "true"
    
    # Check for valgrind
    if command -v valgrind &> /dev/null; then
        run_test_section "Valgrind Memory Check" \
            "valgrind --tool=memcheck --leak-check=full --error-exitcode=1 cargo test --test simple_performance_monitoring $CARGO_FLAGS $FEATURE_FLAGS -- --test-threads=1" \
            "true"
    else
        log_warning "Valgrind not available, skipping memory leak detection"
        SKIPPED_TESTS+=("Valgrind Memory Check")
    fi
fi

# ============================================================================
# Phase 6: Additional Quality Checks (Optional)
# ============================================================================

log_section "Phase 6: Additional Quality Checks (Optional)"

# Security audit
if command -v cargo-audit &> /dev/null; then
    run_test_section "Security Audit" \
        "cargo audit" \
        "true"
else
    log_warning "cargo-audit not installed, skipping security audit"
    SKIPPED_TESTS+=("Security Audit")
fi

# Unused dependencies check
if command -v cargo-udeps &> /dev/null; then
    run_test_section "Unused Dependencies Check" \
        "cargo +nightly udeps $FEATURE_FLAGS" \
        "true"
else
    log_warning "cargo-udeps not installed, skipping unused dependencies check"
    SKIPPED_TESTS+=("Unused Dependencies Check")
fi

# Check for outdated dependencies
if command -v cargo-outdated &> /dev/null; then
    run_test_section "Outdated Dependencies Check" \
        "cargo outdated --exit-code 1" \
        "true"
else
    log_warning "cargo-outdated not installed, skipping outdated dependencies check"
    SKIPPED_TESTS+=("Outdated Dependencies Check")
fi

# ============================================================================
# Results Summary
# ============================================================================

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "============================================="
log_section "Test Results Summary"
echo ""

echo -e "${CYAN}Total Duration:${NC} ${TOTAL_DURATION}s"
echo ""

if [[ ${#PASSED_TESTS[@]} -gt 0 ]]; then
    echo -e "${GREEN}✅ Passed Tests (${#PASSED_TESTS[@]}):${NC}"
    for test in "${PASSED_TESTS[@]}"; do
        echo "  ✅ $test"
    done
    echo ""
fi

if [[ ${#SKIPPED_TESTS[@]} -gt 0 ]]; then
    echo -e "${YELLOW}⚠️  Skipped Tests (${#SKIPPED_TESTS[@]}):${NC}"
    for test in "${SKIPPED_TESTS[@]}"; do
        echo "  ⚠️  $test"
    done  
    echo ""
fi

if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
    echo -e "${RED}❌ Failed Tests (${#FAILED_TESTS[@]}):${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  ❌ $test"
    done
    echo ""
fi

# Final status
echo "============================================="
if [[ ${#FAILED_TESTS[@]} -eq 0 ]]; then
    log_success "All tests completed successfully! 🎉"
    echo ""
    echo "📊 Summary Statistics:"
    echo "  ✅ Passed: ${#PASSED_TESTS[@]}"
    echo "  ⚠️  Skipped: ${#SKIPPED_TESTS[@]}"
    echo "  ❌ Failed: 0"
    echo "  ⏱️  Duration: ${TOTAL_DURATION}s"
    echo ""
    echo "🚀 Ready for commit/deployment!"
    exit 0
else
    log_error "Some tests failed! Please fix the issues before proceeding."
    echo ""
    echo "📊 Summary Statistics:"
    echo "  ✅ Passed: ${#PASSED_TESTS[@]}"
    echo "  ⚠️  Skipped: ${#SKIPPED_TESTS[@]}"
    echo "  ❌ Failed: ${#FAILED_TESTS[@]}"
    echo "  ⏱️  Duration: ${TOTAL_DURATION}s"
    echo ""
    echo "🔧 Next Steps:"
    echo "  1. Review the failed tests above"
    echo "  2. Fix the issues and re-run the tests"
    echo "  3. Consider running individual test sections for faster iteration"
    exit 1
fi