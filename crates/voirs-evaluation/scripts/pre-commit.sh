#!/bin/bash
# VoiRS Evaluation - Pre-commit Hook Script
#
# This script runs automated quality checks before commits to ensure
# code quality and prevent CI failures.
#
# Install as a git hook with:
#   ln -s ../../scripts/pre-commit.sh .git/hooks/pre-commit

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

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

# Track overall success
OVERALL_SUCCESS=true

# Function to run a check and track results
run_check() {
    local check_name="$1"
    local check_command="$2"
    
    log_info "Running $check_name..."
    
    if eval "$check_command"; then
        log_success "$check_name passed"
        return 0
    else
        log_error "$check_name failed"
        OVERALL_SUCCESS=false
        return 1
    fi
}

# Change to project directory
cd "$PROJECT_DIR"

echo "🚀 VoiRS Evaluation Pre-commit Checks"
echo "======================================"

# Check 1: Code formatting
run_check "Code formatting" "cargo fmt --all -- --check"

# Check 2: Clippy lints
run_check "Clippy lints" "cargo clippy --all-targets --all-features -- -D warnings"

# Check 3: Build check
run_check "Build check" "cargo build --all-features"

# Check 4: Unit tests
run_check "Unit tests" "cargo test --lib --all-features"

# Check 5: Integration tests
run_check "Integration tests" "cargo test --test integration_tests --all-features"

# Check 6: Documentation tests
run_check "Documentation tests" "cargo test --doc --all-features"

# Check 7: Documentation generation
run_check "Documentation generation" "cargo doc --all-features --no-deps"

# Optional checks (warnings only)
echo ""
log_info "Running optional checks (warnings only)..."

# Check for unused dependencies (if cargo-udeps is installed)
if command -v cargo-udeps &> /dev/null; then
    log_info "Checking for unused dependencies..."
    if ! cargo +nightly udeps --all-features &> /dev/null; then
        log_warning "Unused dependencies detected (run: cargo +nightly udeps --all-features)"
    else
        log_success "No unused dependencies found"
    fi
else
    log_warning "cargo-udeps not installed (install with: cargo install cargo-udeps)"
fi

# Security audit (if cargo-audit is installed)
if command -v cargo-audit &> /dev/null; then
    log_info "Running security audit..."
    if ! cargo audit &> /dev/null; then
        log_warning "Security vulnerabilities detected (run: cargo audit)"
    else
        log_success "No security vulnerabilities found"
    fi
else
    log_warning "cargo-audit not installed (install with: cargo install cargo-audit)"
fi

# Check for TODO/FIXME comments in staged files
if command -v git &> /dev/null && git rev-parse --git-dir &> /dev/null; then
    log_info "Checking for TODO/FIXME comments in staged files..."
    
    # Get staged files
    staged_files=$(git diff --cached --name-only --diff-filter=AM | grep -E '\.(rs|toml|md)$' || true)
    
    if [ -n "$staged_files" ]; then
        todo_count=0
        for file in $staged_files; do
            if [ -f "$file" ]; then
                todos=$(grep -n -E "(TODO|FIXME|XXX|HACK)" "$file" || true)
                if [ -n "$todos" ]; then
                    if [ $todo_count -eq 0 ]; then
                        log_warning "Found TODO/FIXME comments in staged files:"
                    fi
                    echo "$file:"
                    echo "$todos" | sed 's/^/  /'
                    todo_count=$((todo_count + 1))
                fi
            fi
        done
        
        if [ $todo_count -eq 0 ]; then
            log_success "No TODO/FIXME comments in staged files"
        fi
    fi
fi

# Performance regression check (quick benchmark)
log_info "Running quick performance check..."
if cargo bench --all-features --bench evaluation_metrics -- --test &> /dev/null; then
    log_success "Quick performance check passed"
else
    log_warning "Quick performance check failed or skipped"
fi

# Final result
echo ""
echo "======================================"
if [ "$OVERALL_SUCCESS" = true ]; then
    log_success "All pre-commit checks passed! ✨"
    echo ""
    echo "💡 Tips for maintaining code quality:"
    echo "  - Run 'cargo test --all-features' regularly during development"
    echo "  - Use 'cargo clippy --fix' to automatically fix lint issues"
    echo "  - Keep dependencies up to date with 'cargo update'"
    echo "  - Consider running benchmarks with 'cargo bench' for performance-critical changes"
    exit 0
else
    log_error "Some pre-commit checks failed! ❌"
    echo ""
    echo "🔧 To fix issues:"
    echo "  - Run 'cargo fmt' to fix formatting"
    echo "  - Run 'cargo clippy --fix --all-targets --all-features' to fix lint issues"
    echo "  - Fix failing tests and ensure all tests pass"
    echo "  - Check documentation and fix any broken doc links"
    echo ""
    echo "⚡ Quick fix command:"
    echo "  cargo fmt && cargo clippy --fix --all-targets --all-features --allow-dirty"
    exit 1
fi