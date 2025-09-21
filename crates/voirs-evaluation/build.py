#!/usr/bin/env python3
"""
VoiRS Evaluation - Cross-platform Build Script (Python)

This script provides automated building and testing across all platforms
using Python for maximum compatibility.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color
    
    @classmethod
    def disable_colors(cls):
        """Disable colors for non-terminal output."""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = cls.NC = ''


class Logger:
    """Simple logging utility."""
    
    @staticmethod
    def info(message: str):
        print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")
    
    @staticmethod
    def success(message: str):
        print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")
    
    @staticmethod
    def warning(message: str):
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")
    
    @staticmethod
    def error(message: str):
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")


class PlatformInfo:
    """Platform detection and information."""
    
    @staticmethod
    def get_platform() -> str:
        """Get the current platform name."""
        system = platform.system().lower()
        if system == "darwin":
            return "macos"
        elif system == "windows":
            return "windows"
        elif system == "linux":
            return "linux"
        elif system == "freebsd":
            return "freebsd"
        else:
            return "unknown"
    
    @staticmethod
    def get_architecture() -> str:
        """Get the current architecture."""
        machine = platform.machine().lower()
        if machine in ["x86_64", "amd64"]:
            return "x86_64"
        elif machine in ["arm64", "aarch64"]:
            return "aarch64"
        elif machine.startswith("armv7"):
            return "armv7"
        else:
            return machine


class BuildScript:
    """Main build script class."""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.project_name = "voirs-evaluation"
        self.cargo_target_dir = os.environ.get("CARGO_TARGET_DIR", "target")
        
        # Disable colors if not in terminal
        if not sys.stdout.isatty():
            Colors.disable_colors()
    
    def run_command(self, command: List[str], cwd: Optional[Path] = None) -> int:
        """Run a command and return the exit code."""
        if cwd is None:
            cwd = self.script_dir
        
        Logger.info(f"Running: {' '.join(command)}")
        try:
            result = subprocess.run(command, cwd=cwd, check=False)
            return result.returncode
        except FileNotFoundError:
            Logger.error(f"Command not found: {command[0]}")
            return 1
    
    def check_command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH."""
        return shutil.which(command) is not None
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        Logger.info("Checking build dependencies...")
        
        # Check Rust
        if not self.check_command_exists("rustc"):
            Logger.error("Rust is not installed. Please install Rust from https://rustup.rs/")
            return False
        
        # Check Cargo
        if not self.check_command_exists("cargo"):
            Logger.error("Cargo is not installed. Please install Rust toolchain.")
            return False
        
        # Get versions
        try:
            rust_version = subprocess.check_output(["rustc", "--version"], text=True).strip()
            cargo_version = subprocess.check_output(["cargo", "--version"], text=True).strip()
            Logger.info(f"Rust found: {rust_version}")
            Logger.info(f"Cargo found: {cargo_version}")
        except subprocess.CalledProcessError:
            Logger.error("Failed to get Rust/Cargo versions")
            return False
        
        # Check Python (for Python bindings)
        python_cmd = None
        for cmd in ["python3", "python"]:
            if self.check_command_exists(cmd):
                python_cmd = cmd
                break
        
        if python_cmd:
            try:
                python_version = subprocess.check_output([python_cmd, "--version"], text=True).strip()
                Logger.info(f"Python found: {python_version}")
            except subprocess.CalledProcessError:
                Logger.warning("Python found but version check failed")
        else:
            Logger.warning("Python not found. Python bindings will be disabled.")
        
        # Platform-specific checks
        platform_name = PlatformInfo.get_platform()
        if platform_name == "windows":
            # Check for Visual Studio Build Tools (simplified check)
            if "VCINSTALLDIR" in os.environ:
                Logger.info("Visual C++ environment detected")
            else:
                Logger.warning("Visual Studio Build Tools may not be properly configured")
        elif platform_name == "macos":
            # Check for Xcode Command Line Tools
            if self.run_command(["xcode-select", "-p"]) != 0:
                Logger.warning("Xcode Command Line Tools not found. Some features may not work.")
        elif platform_name == "linux":
            # Check for common build tools
            if not self.check_command_exists("pkg-config"):
                Logger.warning("pkg-config not found. Install it with your package manager.")
        
        Logger.success("Dependency check completed")
        return True
    
    def setup_environment(self):
        """Setup build environment."""
        Logger.info("Setting up build environment...")
        
        # Set environment variables
        os.environ["CARGO_TARGET_DIR"] = self.cargo_target_dir
        os.environ["RUST_BACKTRACE"] = "1"
        
        # Platform-specific setup
        platform_name = PlatformInfo.get_platform()
        if platform_name == "macos":
            # Set SDK path for macOS
            try:
                sdk_path = subprocess.check_output(
                    ["xcrun", "--sdk", "macosx", "--show-sdk-path"], 
                    text=True
                ).strip()
                os.environ["SDKROOT"] = sdk_path
            except subprocess.CalledProcessError:
                Logger.warning("Failed to set macOS SDK path")
        elif platform_name == "linux":
            # Set library path for Linux
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"/usr/local/lib:{current_ld_path}"
        
        Logger.info(f"Platform: {platform_name}")
        Logger.info(f"Architecture: {PlatformInfo.get_architecture()}")
    
    def clean(self) -> int:
        """Clean build artifacts."""
        Logger.info("Cleaning build artifacts...")
        
        exit_code = self.run_command(["cargo", "clean"])
        if exit_code != 0:
            return exit_code
        
        # Remove documentation directory
        doc_dir = self.script_dir / "target" / "doc"
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
        
        Logger.success("Clean completed")
        return 0
    
    def build(self, release: bool = False, features: Optional[str] = None, all_features: bool = False) -> int:
        """Build the project."""
        build_args = ["cargo", "build"]
        
        if release:
            build_args.append("--release")
        
        if all_features:
            build_args.append("--all-features")
        elif features:
            build_args.extend(["--features", features])
        
        Logger.info(f"Building {self.project_name}...")
        Logger.info(f"Build arguments: {' '.join(build_args[2:])}")
        
        # Build the main library
        exit_code = self.run_command(build_args)
        if exit_code != 0:
            return exit_code
        
        # Build examples
        Logger.info("Building examples...")
        example_args = build_args + ["--examples"]
        exit_code = self.run_command(example_args)
        if exit_code != 0:
            return exit_code
        
        # Build benchmarks
        Logger.info("Building benchmarks...")
        bench_args = build_args + ["--benches"]
        exit_code = self.run_command(bench_args)
        if exit_code != 0:
            return exit_code
        
        Logger.success("Build completed successfully")
        return 0
    
    def test(self, release: bool = False, features: Optional[str] = None, all_features: bool = False) -> int:
        """Run tests."""
        test_args = ["cargo", "test"]
        
        if release:
            test_args.append("--release")
        
        if all_features:
            test_args.append("--all-features")
        elif features:
            test_args.extend(["--features", features])
        
        Logger.info("Running tests...")
        
        # Run unit tests
        exit_code = self.run_command(test_args)
        if exit_code != 0:
            return exit_code
        
        # Run integration tests
        Logger.info("Running integration tests...")
        integration_args = test_args + ["--test", "integration_tests"]
        exit_code = self.run_command(integration_args)
        if exit_code != 0:
            return exit_code
        
        # Run doc tests
        Logger.info("Running documentation tests...")
        doc_args = test_args + ["--doc"]
        exit_code = self.run_command(doc_args)
        if exit_code != 0:
            return exit_code
        
        Logger.success("All tests passed")
        return 0
    
    def bench(self) -> int:
        """Run benchmarks."""
        Logger.info("Running benchmarks...")
        
        exit_code = self.run_command(["cargo", "bench", "--all"])
        if exit_code != 0:
            return exit_code
        
        Logger.success("Benchmarks completed")
        return 0
    
    def docs(self, open_docs: bool = False) -> int:
        """Generate documentation."""
        Logger.info("Generating documentation...")
        
        docs_args = ["cargo", "doc", "--all-features", "--no-deps"]
        if open_docs:
            docs_args.append("--open")
        
        exit_code = self.run_command(docs_args)
        if exit_code != 0:
            return exit_code
        
        Logger.success("Documentation generated")
        return 0
    
    def check(self) -> int:
        """Run code quality checks."""
        Logger.info("Running code quality checks...")
        
        # Check formatting
        Logger.info("Checking code formatting...")
        exit_code = self.run_command(["cargo", "fmt", "--", "--check"])
        if exit_code != 0:
            return exit_code
        
        # Run clippy
        Logger.info("Running Clippy lints...")
        exit_code = self.run_command(["cargo", "clippy", "--all-features", "--", "-D", "warnings"])
        if exit_code != 0:
            return exit_code
        
        # Check for unused dependencies
        if self.check_command_exists("cargo-udeps"):
            Logger.info("Checking for unused dependencies...")
            exit_code = self.run_command(["cargo", "+nightly", "udeps", "--all-features"])
            if exit_code != 0:
                Logger.warning("Unused dependency check failed, but continuing...")
        else:
            Logger.warning("cargo-udeps not installed. Skipping unused dependency check.")
        
        Logger.success("Code quality checks passed")
        return 0
    
    def install(self) -> int:
        """Install the package."""
        Logger.info(f"Installing {self.project_name}...")
        
        exit_code = self.run_command(["cargo", "install", "--path", ".", "--all-features"])
        if exit_code != 0:
            return exit_code
        
        Logger.success("Installation completed")
        return 0
    
    def build_python(self) -> int:
        """Build Python bindings."""
        # Check Python availability
        python_cmd = None
        for cmd in ["python3", "python"]:
            if self.check_command_exists(cmd):
                python_cmd = cmd
                break
        
        if not python_cmd:
            Logger.error("Python is required for building Python bindings")
            return 1
        
        Logger.info("Building Python bindings...")
        
        # Build with Python feature
        exit_code = self.run_command(["cargo", "build", "--release", "--features", "python"])
        if exit_code != 0:
            return exit_code
        
        # Build Python wheel
        if self.check_command_exists("maturin"):
            exit_code = self.run_command(["maturin", "build", "--release", "--features", "python"])
            if exit_code != 0:
                Logger.warning("maturin build failed, but continuing...")
        else:
            Logger.warning("maturin not found. Install with: pip install maturin")
        
        Logger.success("Python bindings built")
        return 0
    
    def ci(self) -> int:
        """Run full CI pipeline."""
        Logger.info("Running full CI pipeline...")
        
        if not self.check_dependencies():
            return 1
        
        self.setup_environment()
        
        # Run all CI steps
        steps = [
            ("clean", lambda: self.clean()),
            ("check", lambda: self.check()),
            ("build", lambda: self.build(all_features=True)),
            ("test", lambda: self.test(all_features=True)),
            ("docs", lambda: self.docs()),
        ]
        
        for step_name, step_func in steps:
            Logger.info(f"CI Step: {step_name}")
            exit_code = step_func()
            if exit_code != 0:
                Logger.error(f"CI pipeline failed at step: {step_name}")
                return exit_code
        
        Logger.success("CI pipeline completed successfully")
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VoiRS Evaluation Cross-platform Build Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build.py build --release --all-features
  python build.py test --features python
  python build.py docs --open
  python build.py ci
        """
    )
    
    parser.add_argument(
        "command",
        choices=["build", "test", "bench", "docs", "check", "clean", "install", "python", "ci"],
        help="Command to execute"
    )
    
    parser.add_argument("--release", action="store_true", help="Build in release mode")
    parser.add_argument("--features", help="Specify features to enable")
    parser.add_argument("--all-features", action="store_true", help="Enable all features")
    parser.add_argument("--open", action="store_true", help="Open documentation after generation")
    
    args = parser.parse_args()
    
    build_script = BuildScript()
    
    # Change to script directory
    os.chdir(build_script.script_dir)
    
    # Execute command
    try:
        if args.command == "build":
            exit_code = build_script.build(args.release, args.features, args.all_features)
        elif args.command == "test":
            exit_code = build_script.test(args.release, args.features, args.all_features)
        elif args.command == "bench":
            exit_code = build_script.bench()
        elif args.command == "docs":
            exit_code = build_script.docs(args.open)
        elif args.command == "check":
            exit_code = build_script.check()
        elif args.command == "clean":
            exit_code = build_script.clean()
        elif args.command == "install":
            exit_code = build_script.install()
        elif args.command == "python":
            exit_code = build_script.build_python()
        elif args.command == "ci":
            exit_code = build_script.ci()
        else:
            Logger.error(f"Unknown command: {args.command}")
            exit_code = 1
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        Logger.error("Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        Logger.error(f"Build script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()