#!/usr/bin/env python3
"""
Test runner for VoiRS Python bindings.

This script provides comprehensive testing capabilities including:
- Unit tests
- Integration tests  
- Performance tests
- Stress tests
- Memory leak detection
- Coverage reporting
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the parent directory to the path so we can import voirs_ffi
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

try:
    import voirs_ffi as voirs
    VOIRS_AVAILABLE = True
except ImportError:
    VOIRS_AVAILABLE = False


class TestRunner:
    """Comprehensive test runner for VoiRS Python bindings."""
    
    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.results = {}
        
    def run_unit_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run unit tests."""
        print("=" * 60)
        print("RUNNING UNIT TESTS")
        print("=" * 60)
        
        unit_dir = self.test_dir / "unit"
        if not unit_dir.exists():
            return {"status": "skipped", "reason": "Unit test directory not found"}
        
        cmd = [
            sys.executable, "-m", "pytest", 
            str(unit_dir),
            "-v" if verbose else "-q",
            "--tb=short",
            "-x",  # Stop on first failure
            "--durations=10"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run integration tests."""
        print("\n" + "=" * 60)
        print("RUNNING INTEGRATION TESTS")
        print("=" * 60)
        
        integration_dir = self.test_dir / "integration"
        if not integration_dir.exists():
            return {"status": "skipped", "reason": "Integration test directory not found"}
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(integration_dir),
            "-v" if verbose else "-q",
            "--tb=short",
            "--durations=10"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run performance tests."""
        print("\n" + "=" * 60)
        print("RUNNING PERFORMANCE TESTS")
        print("=" * 60)
        
        performance_dir = self.test_dir / "performance"
        if not performance_dir.exists():
            return {"status": "skipped", "reason": "Performance test directory not found"}
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(performance_dir),
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "performance",
            "--durations=0"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def run_stress_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run stress tests."""
        print("\n" + "=" * 60)
        print("RUNNING STRESS TESTS")
        print("=" * 60)
        
        performance_dir = self.test_dir / "performance"
        if not performance_dir.exists():
            return {"status": "skipped", "reason": "Performance test directory not found"}
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(performance_dir / "test_stress.py"),
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "stress",
            "--durations=0"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def run_memory_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run memory-focused tests."""
        print("\n" + "=" * 60)
        print("RUNNING MEMORY TESTS")
        print("=" * 60)
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "memory",
            "--durations=0"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def run_all_tests(self, verbose: bool = False, include_stress: bool = False) -> Dict[str, Any]:
        """Run all tests."""
        print("=" * 60)
        print("RUNNING ALL TESTS")
        print("=" * 60)
        
        # Determine which tests to run
        test_dirs = []
        if (self.test_dir / "unit").exists():
            test_dirs.append(str(self.test_dir / "unit"))
        if (self.test_dir / "integration").exists():
            test_dirs.append(str(self.test_dir / "integration"))
        if include_stress and (self.test_dir / "performance").exists():
            test_dirs.append(str(self.test_dir / "performance"))
        
        if not test_dirs:
            return {"status": "skipped", "reason": "No test directories found"}
        
        cmd = [
            sys.executable, "-m", "pytest",
            *test_dirs,
            "-v" if verbose else "-q",
            "--tb=short",
            "--durations=10"
        ]
        
        if include_stress:
            cmd.extend(["-m", "not stress"])  # Exclude stress tests by default
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def run_with_coverage(self, verbose: bool = False) -> Dict[str, Any]:
        """Run tests with coverage reporting."""
        print("\n" + "=" * 60)
        print("RUNNING TESTS WITH COVERAGE")
        print("=" * 60)
        
        try:
            import coverage
        except ImportError:
            return {"status": "skipped", "reason": "Coverage package not available"}
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "--cov=voirs_ffi",
            "--cov-report=term-missing",
            "--cov-report=html",
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "not stress"  # Exclude stress tests for coverage
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check if all prerequisites are available."""
        return {
            "pytest": PYTEST_AVAILABLE,
            "voirs_ffi": VOIRS_AVAILABLE,
            "test_dir": self.test_dir.exists(),
            "python_version": sys.version_info >= (3, 7)
        }
    
    def print_summary(self, results: Dict[str, Any]):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total_duration = 0
        passed_count = 0
        failed_count = 0
        skipped_count = 0
        
        for test_type, result in results.items():
            if test_type == "prerequisites":
                continue
                
            status = result.get("status", "unknown")
            duration = result.get("duration", 0)
            total_duration += duration
            
            if status == "passed":
                passed_count += 1
                status_icon = "✓"
            elif status == "failed":
                failed_count += 1
                status_icon = "✗"
            else:
                skipped_count += 1
                status_icon = "○"
            
            print(f"{status_icon} {test_type.upper()}: {status} ({duration:.2f}s)")
        
        print(f"\nTotal: {passed_count} passed, {failed_count} failed, {skipped_count} skipped")
        print(f"Duration: {total_duration:.2f}s")
        
        if failed_count > 0:
            print("\nFAILED TESTS:")
            for test_type, result in results.items():
                if result.get("status") == "failed":
                    print(f"\n{test_type.upper()} STDERR:")
                    print(result.get("stderr", "No error output"))
        
        return failed_count == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run VoiRS Python binding tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test-type", "-t", 
                       choices=["unit", "integration", "performance", "stress", "memory", "all", "coverage"],
                       default="all", help="Type of tests to run")
    parser.add_argument("--include-stress", action="store_true", help="Include stress tests")
    parser.add_argument("--test-dir", type=Path, help="Test directory path")
    
    args = parser.parse_args()
    
    # Determine test directory
    if args.test_dir:
        test_dir = args.test_dir
    else:
        test_dir = Path(__file__).parent
    
    if not test_dir.exists():
        print(f"Error: Test directory {test_dir} does not exist")
        return 1
    
    runner = TestRunner(test_dir)
    
    # Check prerequisites
    prereqs = runner.check_prerequisites()
    print("Prerequisites check:")
    for name, available in prereqs.items():
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
    
    if not all(prereqs.values()):
        print("\nSome prerequisites are missing. Please install required packages.")
        return 1
    
    # Run tests based on type
    results = {"prerequisites": prereqs}
    
    if args.test_type == "unit":
        results["unit"] = runner.run_unit_tests(args.verbose)
    elif args.test_type == "integration":
        results["integration"] = runner.run_integration_tests(args.verbose)
    elif args.test_type == "performance":
        results["performance"] = runner.run_performance_tests(args.verbose)
    elif args.test_type == "stress":
        results["stress"] = runner.run_stress_tests(args.verbose)
    elif args.test_type == "memory":
        results["memory"] = runner.run_memory_tests(args.verbose)
    elif args.test_type == "coverage":
        results["coverage"] = runner.run_with_coverage(args.verbose)
    else:  # all
        results["unit"] = runner.run_unit_tests(args.verbose)
        results["integration"] = runner.run_integration_tests(args.verbose)
        results["performance"] = runner.run_performance_tests(args.verbose)
        if args.include_stress:
            results["stress"] = runner.run_stress_tests(args.verbose)
    
    # Print summary
    success = runner.print_summary(results)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())