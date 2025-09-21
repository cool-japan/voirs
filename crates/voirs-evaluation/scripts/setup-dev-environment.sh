#!/bin/bash
# VoiRS Evaluation - Development Environment Setup
#
# This script sets up a complete development environment for VoiRS evaluation
# including all necessary tools, hooks, and dependencies.

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Options
INSTALL_OPTIONAL_TOOLS=false
SETUP_GIT_HOOKS=true
INSTALL_PYTHON_DEPS=false
INSTALL_R_DEPS=false
VERBOSE=false

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

log_section() {
    echo -e "${PURPLE}[SECTION]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to show help
show_help() {
    cat << EOF
VoiRS Evaluation Development Environment Setup

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --optional-tools    Install optional development tools
    --no-git-hooks     Skip git hooks setup
    --python-deps      Install Python dependencies
    --r-deps           Install R dependencies  
    --verbose          Enable verbose output
    --help             Show this help message

DESCRIPTION:
    This script sets up a complete development environment including:
    - Rust toolchain components
    - Development tools (clippy, rustfmt, etc.)
    - Git hooks for pre-commit checks
    - Optional: Additional quality tools
    - Optional: Python/R integration dependencies

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --optional-tools)
            INSTALL_OPTIONAL_TOOLS=true
            shift
            ;;
        --no-git-hooks)
            SETUP_GIT_HOOKS=false
            shift
            ;;
        --python-deps)
            INSTALL_PYTHON_DEPS=true
            shift
            ;;
        --r-deps)
            INSTALL_R_DEPS=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
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

# Change to project directory
cd "$PROJECT_DIR"

echo "🚀 VoiRS Evaluation Development Environment Setup"
echo "================================================="
echo ""

# ============================================================================
# Phase 1: System Information and Prerequisites
# ============================================================================

log_section "Phase 1: System Information and Prerequisites"

log_info "Detecting system information..."
echo "  OS: $(uname -s)"
echo "  Architecture: $(uname -m)"
echo "  Shell: $SHELL"

if command_exists git; then
    echo "  Git: $(git --version)"
else
    log_error "Git is not installed. Please install Git first."
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    log_warning "Not in a git repository. Some features may not work."
fi

echo ""

# ============================================================================
# Phase 2: Rust Toolchain Setup
# ============================================================================

log_section "Phase 2: Rust Toolchain Setup"

# Check if Rust is installed
if command_exists rustc && command_exists cargo; then
    log_success "Rust toolchain found:"
    echo "  Rust: $(rustc --version)"
    echo "  Cargo: $(cargo --version)"
else
    log_error "Rust toolchain not found. Please install Rust from https://rustup.rs/"
    exit 1
fi

# Install/update required components
log_info "Installing required Rust components..."

components=(
    "rustfmt"
    "clippy"
)

for component in "${components[@]}"; do
    if rustup component add "$component" 2>/dev/null; then
        log_success "Installed/updated $component"
    else
        log_warning "Failed to install $component (may already be installed)"
    fi
done

# Install nightly toolchain for advanced features
log_info "Installing nightly toolchain for advanced features..."
if rustup toolchain install nightly 2>/dev/null; then
    log_success "Nightly toolchain installed"
else
    log_warning "Failed to install nightly toolchain"
fi

echo ""

# ============================================================================
# Phase 3: Development Tools Installation
# ============================================================================

log_section "Phase 3: Development Tools Installation"

# Core tools (always install)
core_tools=(
    "cargo-audit:Security audit tool"
    "cargo-outdated:Dependency update checker"
    "cargo-tree:Dependency tree viewer"
)

log_info "Installing core development tools..."
for tool_info in "${core_tools[@]}"; do
    IFS=':' read -r tool description <<< "$tool_info"
    
    if command_exists "$tool"; then
        log_success "$tool already installed ($description)"
    else
        log_info "Installing $tool ($description)..."
        if cargo install "$tool" >/dev/null 2>&1; then
            log_success "Installed $tool"
        else
            log_warning "Failed to install $tool"
        fi
    fi
done

# Optional tools
if [[ "$INSTALL_OPTIONAL_TOOLS" == "true" ]]; then
    log_info "Installing optional development tools..."
    
    optional_tools=(
        "cargo-udeps:Unused dependency checker (requires nightly)"
        "cargo-watch:File watcher for automatic rebuilds"
        "cargo-expand:Macro expansion viewer"
        "cargo-tarpaulin:Code coverage tool"
        "cargo-criterion:Benchmark result analysis"
        "cargo-fuzz:Fuzzing framework"
    )
    
    for tool_info in "${optional_tools[@]}"; do
        IFS=':' read -r tool description <<< "$tool_info"
        
        if command_exists "$tool"; then
            log_success "$tool already installed ($description)"
        else
            log_info "Installing $tool ($description)..."
            if [[ "$tool" == "cargo-udeps" ]]; then
                # cargo-udeps requires nightly
                if cargo +nightly install "$tool" >/dev/null 2>&1; then
                    log_success "Installed $tool"
                else
                    log_warning "Failed to install $tool (requires nightly Rust)"
                fi
            else
                if cargo install "$tool" >/dev/null 2>&1; then
                    log_success "Installed $tool"
                else
                    log_warning "Failed to install $tool"
                fi
            fi
        fi
    done
fi

echo ""

# ============================================================================
# Phase 4: Git Hooks Setup
# ============================================================================

if [[ "$SETUP_GIT_HOOKS" == "true" ]] && git rev-parse --git-dir >/dev/null 2>&1; then
    log_section "Phase 4: Git Hooks Setup"
    
    git_dir=$(git rev-parse --git-dir)
    hooks_dir="$git_dir/hooks"
    
    # Create hooks directory if it doesn't exist
    mkdir -p "$hooks_dir"
    
    # Install pre-commit hook
    pre_commit_hook="$hooks_dir/pre-commit"
    pre_commit_script="$SCRIPT_DIR/pre-commit.sh"
    
    if [[ -f "$pre_commit_script" ]]; then
        log_info "Installing pre-commit hook..."
        
        # Create symlink to pre-commit script
        if ln -sf "../../scripts/pre-commit.sh" "$pre_commit_hook" 2>/dev/null; then
            chmod +x "$pre_commit_hook"
            log_success "Pre-commit hook installed"
        else
            log_warning "Failed to install pre-commit hook (using copy instead)"
            cp "$pre_commit_script" "$pre_commit_hook"
            chmod +x "$pre_commit_hook"
        fi
    else
        log_warning "Pre-commit script not found at $pre_commit_script"
    fi
    
    # Install commit-msg hook for conventional commits (if desired)
    log_info "Setting up commit message template..."
    cat > "$PROJECT_DIR/.gitmessage" << 'EOF'
# <type>(<scope>): <subject>
#
# <body>
#
# <footer>
#
# Type: feat, fix, docs, style, refactor, perf, test, chore
# Scope: component or file affected
# Subject: imperative, lowercase, no period
# Body: what and why (optional)
# Footer: breaking changes, issue references (optional)
EOF
    
    git config commit.template .gitmessage 2>/dev/null || log_warning "Failed to set commit template"
    
    echo ""
fi

# ============================================================================
# Phase 5: Python Dependencies (Optional)
# ============================================================================

if [[ "$INSTALL_PYTHON_DEPS" == "true" ]]; then
    log_section "Phase 5: Python Dependencies Setup"
    
    if command_exists python3; then
        log_info "Python 3 found: $(python3 --version)"
        
        # Check if pip is available
        if python3 -m pip --version >/dev/null 2>&1; then
            log_info "Installing Python dependencies for integration..."
            
            python_deps=(
                "numpy"
                "scipy"
                "pandas"
                "matplotlib"
                "maturin"
            )
            
            for dep in "${python_deps[@]}"; do
                log_info "Installing $dep..."
                if python3 -m pip install --user "$dep" >/dev/null 2>&1; then
                    log_success "Installed $dep"
                else
                    log_warning "Failed to install $dep"
                fi
            done
        else
            log_warning "pip not available for Python 3"
        fi
    else
        log_warning "Python 3 not found. Install Python 3 for Python integration features."
    fi
    
    echo ""
fi

# ============================================================================
# Phase 6: R Dependencies (Optional)
# ============================================================================

if [[ "$INSTALL_R_DEPS" == "true" ]]; then
    log_section "Phase 6: R Dependencies Setup"
    
    if command_exists R; then
        log_info "R found: $(R --version | head -n1)"
        
        log_info "Installing essential R packages..."
        r_packages=(
            "ggplot2"
            "dplyr"
            "tidyr"
            "readr"
            "devtools"
        )
        
        for package in "${r_packages[@]}"; do
            log_info "Installing R package: $package"
            if R --slave -e "if (!require('$package', quietly=TRUE)) install.packages('$package', repos='https://cran.r-project.org/', quiet=TRUE)" >/dev/null 2>&1; then
                log_success "Installed R package: $package"
            else
                log_warning "Failed to install R package: $package"
            fi
        done
    else
        log_warning "R not found. Install R for statistical analysis features."
    fi
    
    echo ""
fi

# ============================================================================
# Phase 7: Project-Specific Setup
# ============================================================================

log_section "Phase 7: Project-specific Setup"

# Create useful development scripts
log_info "Creating development convenience scripts..."

# Create a quick test script
cat > "$PROJECT_DIR/quick-test.sh" << 'EOF'
#!/bin/bash
# Quick test runner for development
set -e
echo "🧪 Running quick tests..."
cargo test --lib --all-features
echo "✅ Quick tests passed!"
EOF
chmod +x "$PROJECT_DIR/quick-test.sh"
log_success "Created quick-test.sh"

# Create a benchmark runner script
cat > "$PROJECT_DIR/run-benchmarks.sh" << 'EOF'
#!/bin/bash
# Run all benchmarks
set -e
echo "🏃 Running benchmarks..."
cargo bench --all-features
echo "✅ Benchmarks completed!"
EOF
chmod +x "$PROJECT_DIR/run-benchmarks.sh"
log_success "Created run-benchmarks.sh"

# Set up local configuration
log_info "Setting up local configuration..."

# Create .cargo/config.toml for local settings
mkdir -p .cargo
cat > .cargo/config.toml << 'EOF'
# Local cargo configuration
[build]
# Uncomment to use more CPU cores for compilation
# jobs = 8

[target.x86_64-unknown-linux-gnu]
# Uncomment for faster linking on Linux
# linker = "clang"
# rustflags = ["-C", "link-arg=-fuse-ld=lld"]

# Enable faster builds in development
[profile.dev]
# Uncomment for faster builds (less optimization)
# opt-level = 1

[profile.dev.package."*"]
# Compile dependencies with some optimization
opt-level = 2
EOF
log_success "Created local cargo configuration"

echo ""

# ============================================================================
# Phase 8: Verification and Testing
# ============================================================================

log_section "Phase 8: Environment Verification"

log_info "Verifying development environment..."

# Test basic build
log_info "Testing basic build..."
if cargo check --all-features >/dev/null 2>&1; then
    log_success "Basic build test passed"
else
    log_error "Basic build test failed"
fi

# Test formatting
log_info "Testing code formatting..."
if cargo fmt --all -- --check >/dev/null 2>&1; then
    log_success "Code formatting test passed"
else
    log_warning "Code formatting test failed (run 'cargo fmt' to fix)"
fi

# Test clippy
log_info "Testing clippy lints..."
if cargo clippy --all-features -- -D warnings >/dev/null 2>&1; then
    log_success "Clippy test passed"
else
    log_warning "Clippy test failed (run 'cargo clippy --fix' to fix some issues)"
fi

echo ""

# ============================================================================
# Results Summary
# ============================================================================

log_section "Setup Complete! 🎉"

echo "Your VoiRS Evaluation development environment is ready!"
echo ""
echo "📋 What was set up:"
echo "  ✅ Rust toolchain components (rustfmt, clippy)"
if [[ "$SETUP_GIT_HOOKS" == "true" ]]; then
    echo "  ✅ Git pre-commit hooks"
fi
echo "  ✅ Core development tools"
if [[ "$INSTALL_OPTIONAL_TOOLS" == "true" ]]; then
    echo "  ✅ Optional development tools"
fi
if [[ "$INSTALL_PYTHON_DEPS" == "true" ]]; then
    echo "  ✅ Python integration dependencies"
fi
if [[ "$INSTALL_R_DEPS" == "true" ]]; then
    echo "  ✅ R integration dependencies"
fi
echo "  ✅ Development convenience scripts"
echo "  ✅ Local cargo configuration"
echo ""

echo "🚀 Next steps:"
echo "  1. Run './quick-test.sh' to verify everything works"
echo "  2. Run './scripts/run-all-tests.sh' for comprehensive testing"
echo "  3. Start developing with confidence!"
echo ""

echo "💡 Development tips:"
echo "  • Use 'cargo watch -x test' for continuous testing during development"
echo "  • Run 'cargo clippy --fix' to automatically fix lint issues"
echo "  • Use 'cargo expand' to see macro expansions"
echo "  • Run './run-benchmarks.sh' after performance-related changes"
echo ""

echo "📚 Documentation:"
echo "  • Build docs: cargo doc --all-features --open"
echo "  • View README: less README.md"
echo "  • Check BUILD.md for build system details"
echo ""

log_success "Development environment setup complete! Happy coding! 🦀"