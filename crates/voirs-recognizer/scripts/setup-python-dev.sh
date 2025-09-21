#!/bin/bash
# Python Development Setup Script for VoiRS Recognizer
# This script helps resolve Python linking issues and sets up the Python development environment

set -e

echo "🐍 VoiRS Recognizer Python Development Setup"
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Python is installed
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_status "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    if command -v pip3 &> /dev/null; then
        print_status "pip3 found"
    else
        print_error "pip3 is not installed"
        exit 1
    fi
}

# Install Python development headers
install_python_dev() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        print_status "Setting up Python development environment for macOS"
        
        # Check if Homebrew is installed
        if command -v brew &> /dev/null; then
            print_status "Homebrew found, ensuring python3-dev is available"
            brew install python3 || true
        else
            print_warning "Homebrew not found, consider installing it for easier Python management"
        fi
        
        # For macOS, we need to ensure the Python framework is properly linked
        if [[ -n "$CONDA_PREFIX" ]]; then
            print_status "Conda environment detected: $CONDA_PREFIX"
            export PYTHONPATH="$CONDA_PREFIX/lib/python3.*/site-packages"
        fi
        
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        print_status "Setting up Python development environment for Linux"
        
        if command -v apt-get &> /dev/null; then
            # Ubuntu/Debian
            print_status "Installing Python development headers (Ubuntu/Debian)"
            sudo apt-get update
            sudo apt-get install -y python3-dev python3-pip
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL
            print_status "Installing Python development headers (CentOS/RHEL)"
            sudo yum install -y python3-devel python3-pip
        elif command -v dnf &> /dev/null; then
            # Fedora
            print_status "Installing Python development headers (Fedora)"
            sudo dnf install -y python3-devel python3-pip
        else
            print_warning "Unknown Linux distribution, please install python3-dev manually"
        fi
    else
        print_warning "Unknown operating system: $OSTYPE"
    fi
}

# Set up Python environment variables
setup_python_env() {
    print_status "Setting up Python environment variables"
    
    # Get Python library path
    PYTHON_LIB_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('stdlib'))")
    PYTHON_INCLUDE_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
    
    # Set environment variables for PyO3
    export PYTHONPATH="$PYTHON_LIB_PATH:$PYTHONPATH"
    export PYTHON_SYS_EXECUTABLE=$(which python3)
    
    print_status "Python library path: $PYTHON_LIB_PATH"
    print_status "Python include path: $PYTHON_INCLUDE_PATH"
    print_status "Python executable: $PYTHON_SYS_EXECUTABLE"
    
    # Create a .env file for persistent environment variables
    cat > .env << EOF
# Python development environment variables for VoiRS Recognizer
export PYTHONPATH="$PYTHON_LIB_PATH:\$PYTHONPATH"
export PYTHON_SYS_EXECUTABLE=$(which python3)
export PYO3_PYTHON=$(which python3)
EOF
    
    print_status "Environment variables saved to .env file"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies"
    
    # Create requirements.txt for development
    cat > requirements-dev.txt << EOF
# Development dependencies for VoiRS Recognizer Python bindings
numpy>=1.21.0
pytest>=7.0.0
pytest-asyncio>=0.21.0
black>=22.0.0
mypy>=0.991
maturin>=1.0.0
EOF
    
    pip3 install -r requirements-dev.txt
    print_status "Python dependencies installed"
}

# Test Python bindings compilation
test_python_bindings() {
    print_status "Testing Python bindings compilation"
    
    # Source the environment variables
    source .env
    
    # Try to build with Python feature
    if cargo build --features python --no-default-features; then
        print_status "Python bindings compiled successfully"
    else
        print_error "Python bindings compilation failed"
        print_warning "Please check the error messages above and ensure all dependencies are installed"
        exit 1
    fi
}

# Create Python development README
create_python_readme() {
    cat > PYTHON_DEVELOPMENT.md << EOF
# Python Development for VoiRS Recognizer

This guide helps you set up and develop the Python bindings for VoiRS Recognizer.

## Prerequisites

- Python 3.8+ with development headers
- Rust toolchain
- PyO3 dependencies

## Setup

1. Run the setup script:
   \`\`\`bash
   ./scripts/setup-python-dev.sh
   \`\`\`

2. Source the environment variables:
   \`\`\`bash
   source .env
   \`\`\`

## Building Python Bindings

\`\`\`bash
# Build with Python features
cargo build --features python

# Run tests (without Python bindings due to linking issues)
cargo test --no-default-features --features whisper,analysis

# Build Python wheel
maturin develop
\`\`\`

## Common Issues

### Linking Errors on macOS

If you encounter linking errors like "symbol not found", try:

1. Ensure Xcode Command Line Tools are installed:
   \`\`\`bash
   xcode-select --install
   \`\`\`

2. Set up the Python environment:
   \`\`\`bash
   export PYTHONPATH=\$(python3 -c "import sysconfig; print(sysconfig.get_path('stdlib'))")
   export PYO3_PYTHON=\$(which python3)
   \`\`\`

3. Use the correct Python version:
   \`\`\`bash
   # For Homebrew Python
   export PYTHONPATH="/usr/local/lib/python3.x/site-packages"
   
   # For system Python
   export PYTHONPATH="/System/Library/Frameworks/Python.framework/Versions/3.x/lib/python3.x"
   \`\`\`

### Linux Linking Issues

Ensure development headers are installed:
\`\`\`bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# CentOS/RHEL
sudo yum install python3-devel

# Fedora
sudo dnf install python3-devel
\`\`\`

## Testing

\`\`\`bash
# Run Rust tests (core functionality)
cargo test --no-default-features --features whisper,analysis

# Run Python tests (after maturin develop)
pytest python/tests/
\`\`\`

## Development Workflow

1. Make changes to Rust code
2. Build with \`maturin develop\`
3. Test Python bindings
4. Run integration tests

## Troubleshooting

If you encounter issues:

1. Check Python version compatibility (3.8+)
2. Verify development headers are installed
3. Ensure environment variables are set correctly
4. Check the GitHub Issues for known problems

For more help, see the [Community Support](#community-support) section in the main README.
EOF
    
    print_status "Python development documentation created"
}

# Main execution
main() {
    echo "Starting Python development setup..."
    
    check_python
    check_pip
    install_python_dev
    setup_python_env
    install_python_deps
    create_python_readme
    
    print_status "Python development environment setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Source the environment variables: source .env"
    echo "2. Build with Python features: cargo build --features python"
    echo "3. See PYTHON_DEVELOPMENT.md for detailed development guide"
    echo ""
    print_warning "Note: If you encounter linking issues, see the troubleshooting section in PYTHON_DEVELOPMENT.md"
}

# Run main function
main "$@"