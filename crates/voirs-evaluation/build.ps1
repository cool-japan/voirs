# VoiRS Evaluation - Cross-platform Build Script (Windows PowerShell)
# 
# This script provides automated building and testing for Windows systems

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [switch]$Release,
    [string]$Features = "",
    [switch]$AllFeatures,
    [switch]$Open
)

# Script configuration
$ProjectName = "voirs-evaluation"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CargoTargetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { "target" }

# Set error action preference
$ErrorActionPreference = "Stop"

# Logging functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to detect architecture
function Get-Architecture {
    switch ($env:PROCESSOR_ARCHITECTURE) {
        "AMD64" { return "x86_64" }
        "ARM64" { return "aarch64" }
        "x86" { return "i686" }
        default { return $env:PROCESSOR_ARCHITECTURE }
    }
}

# Function to check dependencies
function Test-Dependencies {
    Write-Info "Checking build dependencies..."
    
    # Check Rust
    try {
        $rustVersion = & rustc --version 2>$null
        Write-Info "Rust found: $rustVersion"
    }
    catch {
        Write-Error "Rust is not installed. Please install Rust from https://rustup.rs/"
        exit 1
    }
    
    # Check Cargo
    try {
        $cargoVersion = & cargo --version 2>$null
        Write-Info "Cargo found: $cargoVersion"
    }
    catch {
        Write-Error "Cargo is not installed. Please install Rust toolchain."
        exit 1
    }
    
    # Check Python (for Python bindings)
    try {
        $pythonVersion = & python --version 2>$null
        Write-Info "Python found: $pythonVersion"
    }
    catch {
        try {
            $pythonVersion = & python3 --version 2>$null
            Write-Info "Python 3 found: $pythonVersion"
        }
        catch {
            Write-Warning "Python not found. Python bindings will be disabled."
        }
    }
    
    # Check Visual Studio Build Tools
    $vsInstallPath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsInstallPath) {
        $vsInfo = & $vsInstallPath -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vsInfo) {
            Write-Info "Visual Studio Build Tools found: $vsInfo"
        }
    } else {
        Write-Warning "Visual Studio Build Tools not found. Some features may not work."
    }
    
    Write-Success "Dependency check completed"
}

# Function to setup environment
function Initialize-Environment {
    Write-Info "Setting up build environment..."
    
    # Set environment variables
    $env:CARGO_TARGET_DIR = $CargoTargetDir
    $env:RUST_BACKTRACE = "1"
    
    # Windows-specific setup
    if ($env:VCINSTALLDIR) {
        Write-Info "Visual C++ environment detected"
    }
    
    Write-Info "Platform: Windows"
    Write-Info "Architecture: $(Get-Architecture)"
    Write-Info "Rust version: $(& rustc --version)"
    Write-Info "Cargo version: $(& cargo --version)"
}

# Function to clean build artifacts
function Invoke-Clean {
    Write-Info "Cleaning build artifacts..."
    & cargo clean
    if (Test-Path "target\doc") {
        Remove-Item -Recurse -Force "target\doc"
    }
    Write-Success "Clean completed"
}

# Function to build the project
function Invoke-Build {
    $buildArgs = @()
    
    if ($Release) {
        $buildArgs += "--release"
    }
    
    if ($AllFeatures) {
        $buildArgs += "--all-features"
    }
    elseif ($Features) {
        $buildArgs += "--features", $Features
    }
    
    Write-Info "Building $ProjectName..."
    Write-Info "Build arguments: $($buildArgs -join ' ')"
    
    # Build the main library
    & cargo build @buildArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    # Build examples
    Write-Info "Building examples..."
    & cargo build --examples @buildArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    # Build benchmarks
    Write-Info "Building benchmarks..."
    & cargo build --benches @buildArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    Write-Success "Build completed successfully"
}

# Function to run tests
function Invoke-Test {
    $testArgs = @()
    
    if ($Release) {
        $testArgs += "--release"
    }
    
    if ($AllFeatures) {
        $testArgs += "--all-features"
    }
    elseif ($Features) {
        $testArgs += "--features", $Features
    }
    
    Write-Info "Running tests..."
    
    # Run unit tests
    & cargo test @testArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    # Run integration tests
    Write-Info "Running integration tests..."
    & cargo test --test integration_tests @testArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    # Run doc tests
    Write-Info "Running documentation tests..."
    & cargo test --doc @testArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    Write-Success "All tests passed"
}

# Function to run benchmarks
function Invoke-Bench {
    Write-Info "Running benchmarks..."
    
    # Run all benchmarks
    & cargo bench --all
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    Write-Success "Benchmarks completed"
}

# Function to generate documentation
function Invoke-Docs {
    Write-Info "Generating documentation..."
    
    # Generate docs
    & cargo doc --all-features --no-deps
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    # Open docs if requested
    if ($Open) {
        & cargo doc --all-features --no-deps --open
    }
    
    Write-Success "Documentation generated"
}

# Function to check code quality
function Invoke-Check {
    Write-Info "Running code quality checks..."
    
    # Check formatting
    Write-Info "Checking code formatting..."
    & cargo fmt -- --check
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    # Run clippy
    Write-Info "Running Clippy lints..."
    & cargo clippy --all-features -- -D warnings
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    # Check for unused dependencies
    try {
        & cargo udeps --version 2>$null | Out-Null
        Write-Info "Checking for unused dependencies..."
        & cargo +nightly udeps --all-features
    }
    catch {
        Write-Warning "cargo-udeps not installed. Skipping unused dependency check."
    }
    
    Write-Success "Code quality checks passed"
}

# Function to install the package
function Invoke-Install {
    Write-Info "Installing $ProjectName..."
    
    & cargo install --path . --all-features
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    Write-Success "Installation completed"
}

# Function to build Python bindings
function Invoke-PythonBuild {
    # Check Python availability
    $pythonCmd = $null
    try {
        & python --version 2>$null | Out-Null
        $pythonCmd = "python"
    }
    catch {
        try {
            & python3 --version 2>$null | Out-Null
            $pythonCmd = "python3"
        }
        catch {
            Write-Error "Python is required for building Python bindings"
            exit 1
        }
    }
    
    Write-Info "Building Python bindings..."
    
    # Build with Python feature
    & cargo build --release --features python
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    # Build Python wheel
    try {
        & maturin --version 2>$null | Out-Null
        & maturin build --release --features python
    }
    catch {
        Write-Warning "maturin not found. Install with: pip install maturin"
    }
    
    Write-Success "Python bindings built"
}

# Function to run full CI pipeline
function Invoke-CI {
    Write-Info "Running full CI pipeline..."
    
    Test-Dependencies
    Initialize-Environment
    Invoke-Clean
    Invoke-Check
    Invoke-Build -AllFeatures
    Invoke-Test -AllFeatures
    Invoke-Docs
    
    Write-Success "CI pipeline completed successfully"
}

# Function to show help
function Show-Help {
    Write-Host @"
VoiRS Evaluation Build Script

USAGE:
    .\build.ps1 [COMMAND] [OPTIONS]

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

OPTIONS:
    -Release        Build in release mode
    -Features       Specify features to enable
    -AllFeatures    Enable all features
    -Open           Open documentation after generation

EXAMPLES:
    .\build.ps1 build -Release -AllFeatures
    .\build.ps1 test -Features python
    .\build.ps1 docs -Open
    .\build.ps1 ci

"@
}

# Main script logic
function Main {
    Set-Location $ScriptDir
    
    switch ($Command.ToLower()) {
        "build" {
            Test-Dependencies
            Initialize-Environment
            Invoke-Build
        }
        "test" {
            Test-Dependencies
            Initialize-Environment
            Invoke-Test
        }
        "bench" {
            Test-Dependencies
            Initialize-Environment
            Invoke-Bench
        }
        "docs" {
            Test-Dependencies
            Initialize-Environment
            Invoke-Docs
        }
        "check" {
            Test-Dependencies
            Initialize-Environment
            Invoke-Check
        }
        "clean" {
            Invoke-Clean
        }
        "install" {
            Test-Dependencies
            Initialize-Environment
            Invoke-Install
        }
        "python" {
            Test-Dependencies
            Initialize-Environment
            Invoke-PythonBuild
        }
        "ci" {
            Invoke-CI
        }
        "help" {
            Show-Help
        }
        default {
            Write-Error "Unknown command: $Command"
            Show-Help
            exit 1
        }
    }
}

# Run main function
try {
    Main
}
catch {
    Write-Error "Build script failed: $_"
    exit 1
}