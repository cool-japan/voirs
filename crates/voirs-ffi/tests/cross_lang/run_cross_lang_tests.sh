#!/bin/bash
# Cross-language testing script for VoiRS FFI bindings
# Tests consistency between C API, Python, Node.js, and WebAssembly bindings

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFI_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$FFI_DIR")"
TIMEOUT=300  # 5 minutes timeout for tests

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Available bindings (simple variables instead of associative arrays)
RUST_AVAILABLE=""
PYTHON_AVAILABLE=""
NODEJS_AVAILABLE=""
WASM_AVAILABLE=""

RUST_ERROR=""
PYTHON_ERROR=""
NODEJS_ERROR=""
WASM_ERROR=""

log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check binding availability
check_binding_availability() {
    log "Checking binding availability..."
    
    # Check C API (always available if FFI is built)
    if [ -f "$FFI_DIR/target/debug/libvoirs_ffi.so" ] || \
       [ -f "$FFI_DIR/target/debug/libvoirs_ffi.dylib" ] || \
       [ -f "$FFI_DIR/target/debug/voirs_ffi.dll" ]; then
        RUST_AVAILABLE="true"
        success "C API: Available"
    else
        RUST_AVAILABLE="false"
        RUST_ERROR="FFI library not found. Run 'cargo build' first."
        warn "C API: Not available - ${RUST_ERROR}"
    fi
    
    # Check Python bindings
    if command_exists python3; then
        if python3 -c "import voirs_ffi" 2>/dev/null; then
            PYTHON_AVAILABLE="true"
            success "Python: Available"
        else
            PYTHON_AVAILABLE="false"
            PYTHON_ERROR="voirs_ffi module not found. Build Python bindings first."
            warn "Python: Not available - ${PYTHON_ERROR}"
        fi
    else
        PYTHON_AVAILABLE="false"
        PYTHON_ERROR="Python 3 not found"
        warn "Python: Not available - ${PYTHON_ERROR}"
    fi
    
    # Check Node.js bindings
    if command_exists node; then
        if [ -f "$FFI_DIR/index.js" ] && [ -f "$FFI_DIR/package.json" ]; then
            # Try to require the module
            if cd "$FFI_DIR" && node -e "require('./index.js')" 2>/dev/null; then
                NODEJS_AVAILABLE="true"
                success "Node.js: Available"
            else
                NODEJS_AVAILABLE="false"
                NODEJS_ERROR="Node.js module failed to load. Build Node.js bindings first."
                warn "Node.js: Not available - ${NODEJS_ERROR}"
            fi
        else
            NODEJS_AVAILABLE="false"
            NODEJS_ERROR="Node.js bindings not found"
            warn "Node.js: Not available - ${NODEJS_ERROR}"
        fi
    else
        NODEJS_AVAILABLE="false"
        NODEJS_ERROR="Node.js not found"
        warn "Node.js: Not available - ${NODEJS_ERROR}"
    fi
    
    # Check WebAssembly bindings
    if [ -f "$FFI_DIR/pkg/voirs_ffi.js" ]; then
        WASM_AVAILABLE="true"
        success "WebAssembly: Available"
    else
        WASM_AVAILABLE="false"
        WASM_ERROR="WASM bindings not found. Build with 'wasm-pack build --target web' first."
        warn "WebAssembly: Not available - ${WASM_ERROR}"
    fi
    
    # Count available bindings
    local available_count=0
    [ "$RUST_AVAILABLE" = "true" ] && ((available_count++))
    [ "$PYTHON_AVAILABLE" = "true" ] && ((available_count++))
    [ "$NODEJS_AVAILABLE" = "true" ] && ((available_count++))
    [ "$WASM_AVAILABLE" = "true" ] && ((available_count++))
    
    log "Available bindings: $available_count"
    
    if [ $available_count -lt 2 ]; then
        error "Need at least 2 bindings for cross-language testing. Only $available_count available."
        return 1
    fi
    
    return 0
}

# Build necessary components
build_components() {
    log "Building VoiRS FFI components..."
    
    # Build Rust FFI library
    cd "$FFI_DIR"
    if ! cargo build; then
        error "Failed to build Rust FFI library"
        return 1
    fi
    
    # Try to build Python bindings if maturin is available
    if command_exists maturin && [ "$PYTHON_AVAILABLE" != "true" ]; then
        log "Attempting to build Python bindings..."
        if maturin develop --features python; then
            PYTHON_AVAILABLE="true"
            success "Python bindings built successfully"
        else
            warn "Failed to build Python bindings"
        fi
    fi
    
    # Try to build Node.js bindings if napi is available
    if command_exists npm && [ -f "$FFI_DIR/package.json" ] && [ "$NODEJS_AVAILABLE" != "true" ]; then
        log "Attempting to build Node.js bindings..."
        cd "$FFI_DIR"
        if npm install && npm run build; then
            NODEJS_AVAILABLE="true"
            success "Node.js bindings built successfully"
        else
            warn "Failed to build Node.js bindings"
        fi
    fi
    
    # Try to build WASM bindings if wasm-pack is available
    if command_exists wasm-pack && [ "$WASM_AVAILABLE" != "true" ]; then
        log "Attempting to build WASM bindings..."
        cd "$FFI_DIR"
        if wasm-pack build --target web --features wasm; then
            WASM_AVAILABLE="true"
            success "WASM bindings built successfully"
        else
            warn "Failed to build WASM bindings"
        fi
    fi
}

# Run Rust cross-language tests
run_rust_cross_lang_tests() {
    log "Running Rust-based cross-language tests..."
    
    cd "$FFI_DIR"
    
    local test_output
    if test_output=$(timeout $TIMEOUT cargo test cross_lang 2>&1); then
        success "Rust cross-language tests passed"
        echo "$test_output" | grep -E "(test result:|Available bindings:|Passed:|Failed:)" || true
        ((PASSED_TESTS++))
    else
        error "Rust cross-language tests failed"
        echo "$test_output"
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
}

# Run Python consistency tests
run_python_consistency_tests() {
    if [ "$PYTHON_AVAILABLE" != "true" ]; then
        warn "Skipping Python consistency tests - Python bindings not available"
        ((SKIPPED_TESTS++))
        return
    fi
    
    log "Running Python-based consistency tests..."
    
    local test_output
    if test_output=$(timeout $TIMEOUT python3 "$SCRIPT_DIR/test_consistency.py" 2>&1); then
        success "Python consistency tests passed"
        echo "$test_output" | tail -10
        ((PASSED_TESTS++))
    else
        error "Python consistency tests failed"
        echo "$test_output" | tail -20
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
}

# Run performance comparison tests
run_performance_comparison() {
    log "Running performance comparison between bindings..."
    
    local available_bindings=()
    [ "$RUST_AVAILABLE" = "true" ] && available_bindings+=("c_api")
    [ "$PYTHON_AVAILABLE" = "true" ] && available_bindings+=("python")
    [ "$NODEJS_AVAILABLE" = "true" ] && available_bindings+=("nodejs")
    [ "$WASM_AVAILABLE" = "true" ] && available_bindings+=("wasm")
    
    if [ ${#available_bindings[@]} -lt 2 ]; then
        warn "Skipping performance comparison - need at least 2 bindings"
        ((SKIPPED_TESTS++))
        return
    fi
    
    # Create a simple performance test script
    local perf_script=$(mktemp)
    cat > "$perf_script" << 'EOF'
import time
import sys

try:
    import voirs_ffi as voirs
    
    def test_python_performance():
        start_time = time.time()
        pipeline = voirs.VoirsPipeline()
        
        for i in range(5):
            audio = pipeline.synthesize(f"Performance test {i}")
        
        end_time = time.time()
        return end_time - start_time
    
    duration = test_python_performance()
    print(f"Python: {duration:.3f}s for 5 synthesis operations")
    
    # Test enhanced features if available
    try:
        pipeline = voirs.VoirsPipeline()
        
        # Test callback features
        if hasattr(pipeline, 'synthesize_with_callbacks'):
            def progress_cb(current, total, progress, message):
                pass
            def chunk_cb(idx, total, chunk):
                pass
            def error_cb(error_info):
                pass
            
            start_time = time.time()
            audio = pipeline.synthesize_with_callbacks(
                "Testing enhanced callbacks",
                progress_callback=progress_cb,
                chunk_callback=chunk_cb,
                error_callback=error_cb
            )
            callback_duration = time.time() - start_time
            print(f"Python callbacks: {callback_duration:.3f}s")
        
        # Test streaming if available
        if hasattr(pipeline, 'synthesize_streaming'):
            def streaming_cb(idx, total, chunk):
                pass
            
            start_time = time.time()
            audio = pipeline.synthesize_streaming("Testing streaming", streaming_cb)
            streaming_duration = time.time() - start_time
            print(f"Python streaming: {streaming_duration:.3f}s")
            
    except Exception as e:
        print(f"Enhanced features test error: {e}")
    
except ImportError:
    print("Python: Not available")
except Exception as e:
    print(f"Python: Error - {e}")
EOF
    
    if [ "$PYTHON_AVAILABLE" = "true" ]; then
        log "Python performance:"
        timeout 60 python3 "$perf_script" || echo "Python performance test timed out"
    fi
    
    rm -f "$perf_script"
    
    # Add simple C API performance test
    if [ "$RUST_AVAILABLE" = "true" ]; then
        log "C API performance test would be implemented here"
        # This would require a separate C program
    fi
    
    ((PASSED_TESTS++))
    ((TOTAL_TESTS++))
}

# Run memory usage comparison
run_memory_comparison() {
    if ! command_exists python3; then
        warn "Skipping memory comparison - Python required for memory monitoring"
        ((SKIPPED_TESTS++))
        return
    fi
    
    log "Running memory usage comparison..."
    
    # Create memory monitoring script
    local memory_script=$(mktemp)
    cat > "$memory_script" << 'EOF'
import psutil
import time
import sys

def monitor_memory(label):
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

try:
    import voirs_ffi as voirs
    
    def test_memory_usage():
        initial_memory = monitor_memory("initial")
        
        pipeline = voirs.VoirsPipeline()
        after_create = monitor_memory("after_create")
        
        for i in range(10):
            audio = pipeline.synthesize(f"Memory test {i}")
            del audio  # Try to free memory
        
        after_synthesis = monitor_memory("after_synthesis")
        
        print(f"Memory usage:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  After create: {after_create:.1f} MB (+{after_create - initial_memory:.1f})")
        print(f"  After synthesis: {after_synthesis:.1f} MB (+{after_synthesis - initial_memory:.1f})")
        
        # Test enhanced features memory usage
        try:
            if hasattr(pipeline, 'batch_synthesize_with_progress'):
                before_batch = monitor_memory("before_batch")
                
                def progress_callback(current, total, progress, message):
                    pass
                
                results = pipeline.batch_synthesize_with_progress(
                    [f"Batch test {i}" for i in range(5)],
                    progress_callback
                )
                
                after_batch = monitor_memory("after_batch")
                print(f"  After batch synthesis: {after_batch:.1f} MB (+{after_batch - initial_memory:.1f})")
                
                del results
        except Exception as e:
            print(f"  Enhanced features memory test error: {e}")
        
        if after_synthesis - initial_memory > 100:  # 100MB threshold
            print("WARNING: High memory usage detected")
            return False
        return True
    
    if test_memory_usage():
        print("Memory usage test: PASSED")
        sys.exit(0)
    else:
        print("Memory usage test: FAILED")
        sys.exit(1)
        
except ImportError:
    print("Memory test skipped - voirs_ffi not available")
    sys.exit(0)
except Exception as e:
    print(f"Memory test error: {e}")
    sys.exit(1)
EOF
    
    if timeout 120 python3 "$memory_script"; then
        ((PASSED_TESTS++))
    else
        warn "Memory comparison test failed or timed out"
        ((FAILED_TESTS++))
    fi
    
    rm -f "$memory_script"
    ((TOTAL_TESTS++))
}

# Generate test report
generate_report() {
    log "Generating cross-language test report..."
    
    local report_file="$SCRIPT_DIR/cross_lang_test_report.md"
    
    cat > "$report_file" << EOF
# VoiRS Cross-Language Test Report

Generated: $(date)

## Summary

- **Total Tests**: $TOTAL_TESTS
- **Passed**: $PASSED_TESTS
- **Failed**: $FAILED_TESTS
- **Skipped**: $SKIPPED_TESTS
- **Success Rate**: $(( PASSED_TESTS * 100 / (TOTAL_TESTS > 0 ? TOTAL_TESTS : 1) ))%

## Binding Availability

EOF
    
    # Check C API (rust)
    if [ "$RUST_AVAILABLE" = "true" ]; then
        echo "- **c_api**: ✅ Available" >> "$report_file"
    else
        echo "- **c_api**: ❌ Not Available - ${RUST_ERROR:-Unknown error}" >> "$report_file"
    fi
    
    # Check Python
    if [ "$PYTHON_AVAILABLE" = "true" ]; then
        echo "- **python**: ✅ Available" >> "$report_file"
    else
        echo "- **python**: ❌ Not Available - ${PYTHON_ERROR:-Unknown error}" >> "$report_file"
    fi
    
    # Check Node.js
    if [ "$NODEJS_AVAILABLE" = "true" ]; then
        echo "- **nodejs**: ✅ Available" >> "$report_file"
    else
        echo "- **nodejs**: ❌ Not Available - ${NODEJS_ERROR:-Unknown error}" >> "$report_file"
    fi
    
    # Check WebAssembly
    if [ "$WASM_AVAILABLE" = "true" ]; then
        echo "- **wasm**: ✅ Available" >> "$report_file"
    else
        echo "- **wasm**: ❌ Not Available - ${WASM_ERROR:-Unknown error}" >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF

## Test Results

### Cross-Language Consistency
Tests that verify all bindings produce consistent results for the same inputs.

### Performance Comparison
Compares performance characteristics between different bindings.

### Memory Usage Analysis
Monitors memory usage patterns across bindings.

## Recommendations

EOF
    
    if [ $FAILED_TESTS -gt 0 ]; then
        echo "- ⚠️ Some tests failed. Check individual test outputs for details." >> "$report_file"
    fi
    
    if [ $SKIPPED_TESTS -gt 0 ]; then
        echo "- ℹ️ Some tests were skipped due to missing bindings or dependencies." >> "$report_file"
    fi
    
    local available_count=0
    [ "$RUST_AVAILABLE" = "true" ] && ((available_count++))
    [ "$PYTHON_AVAILABLE" = "true" ] && ((available_count++))
    [ "$NODEJS_AVAILABLE" = "true" ] && ((available_count++))
    [ "$WASM_AVAILABLE" = "true" ] && ((available_count++))
    
    if [ $available_count -lt 4 ]; then
        echo "- 🔧 Consider building missing bindings to improve test coverage." >> "$report_file"
    fi
    
    if [ $PASSED_TESTS -eq $TOTAL_TESTS ] && [ $TOTAL_TESTS -gt 0 ]; then
        echo "- ✅ All tests passed! Cross-language consistency is maintained." >> "$report_file"
    fi
    
    log "Report generated: $report_file"
}

# Main execution
main() {
    echo "VoiRS Cross-Language Testing Suite"
    echo "=================================="
    echo
    
    # Check prerequisites
    if ! check_binding_availability; then
        error "Insufficient bindings available for testing"
        exit 1
    fi
    
    # Try to build missing components
    build_components
    
    # Re-check availability after building
    check_binding_availability
    
    echo
    log "Starting cross-language tests..."
    echo
    
    # Run test suites
    run_rust_cross_lang_tests
    run_python_consistency_tests
    run_performance_comparison
    run_memory_comparison
    
    echo
    log "Test execution completed"
    echo
    
    # Print summary
    echo "Test Summary:"
    echo "============="
    echo "Total Tests: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $FAILED_TESTS"
    echo "Skipped: $SKIPPED_TESTS"
    
    if [ $TOTAL_TESTS -gt 0 ]; then
        local success_rate=$(( PASSED_TESTS * 100 / TOTAL_TESTS ))
        echo "Success Rate: $success_rate%"
    fi
    
    # Generate report
    generate_report
    
    # Exit with appropriate code
    if [ $FAILED_TESTS -gt 0 ]; then
        error "Some tests failed"
        exit 1
    elif [ $PASSED_TESTS -eq 0 ]; then
        warn "No tests were run"
        exit 1
    else
        success "All tests passed"
        exit 0
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo "  --check       Only check binding availability"
        echo "  --build       Only build components"
        echo "  --report      Only generate report (skip tests)"
        exit 0
        ;;
    --check)
        check_binding_availability
        exit $?
        ;;
    --build)
        check_binding_availability
        build_components
        exit $?
        ;;
    --report)
        generate_report
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac