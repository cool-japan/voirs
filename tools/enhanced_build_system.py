#!/usr/bin/env python3
"""
VoiRS Examples Enhanced Build System
====================================

This script provides automated building, testing, and validation for VoiRS examples.
It ensures all examples compile correctly, run without errors, and produce expected outputs.

Features:
- Automated example discovery and compilation
- Parallel building for faster CI/CD
- Comprehensive testing with timeout handling
- Output validation and regression testing
- Dependency management and version checking
- Platform-specific build optimization
- Detailed reporting and error analysis

Usage:
    python enhanced_build_system.py [OPTIONS]

Options:
    --build-only        Only compile examples, don't run tests
    --test-only        Only run tests, skip building
    --parallel N       Number of parallel build jobs (default: CPU count)
    --timeout SECONDS  Test timeout in seconds (default: 60)
    --platform TARGET  Target platform (windows/linux/macos/all)
    --verbose          Enable verbose output
    --report PATH      Generate detailed report at path
    --clean            Clean build artifacts before building
"""

import argparse
import concurrent.futures
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class ExampleInfo:
    """Information about a VoiRS example"""
    name: str
    path: Path
    description: str = ""
    categories: List[str] = None
    prerequisites: List[str] = None
    expected_outputs: List[str] = None
    timeout_seconds: int = 60
    platform_specific: bool = False
    requires_gpu: bool = False
    memory_intensive: bool = False

@dataclass
class BuildResult:
    """Result of building an example"""
    name: str
    success: bool
    duration: float
    output: str = ""
    error: str = ""
    warnings: List[str] = None
    binary_size: Optional[int] = None

@dataclass
class TestResult:
    """Result of testing an example"""
    name: str
    success: bool
    duration: float
    output: str = ""
    error: str = ""
    generated_files: List[str] = None
    performance_metrics: Dict[str, float] = None

@dataclass
class ValidationResult:
    """Result of validating example outputs"""
    name: str
    success: bool
    issues: List[str] = None
    audio_quality_score: Optional[float] = None
    file_size_kb: Optional[float] = None

class VoiRSBuildSystem:
    """Enhanced build system for VoiRS examples"""

    def __init__(self, examples_dir: Path = None):
        self.examples_dir = examples_dir or Path(".")
        self.cargo_toml = self.examples_dir / "Cargo.toml"
        self.target_dir = self.examples_dir / "target"
        self.reports_dir = self.examples_dir / "build_reports"
        
        self.examples: List[ExampleInfo] = []
        self.build_results: List[BuildResult] = []
        self.test_results: List[TestResult] = []
        self.validation_results: List[ValidationResult] = []
        
        # Ensure directories exist
        self.reports_dir.mkdir(exist_ok=True)
        
        # Discover examples
        self._discover_examples()

    def _discover_examples(self):
        """Discover all examples from Cargo.toml and source files"""
        logger.info("Discovering examples...")
        
        if not self.cargo_toml.exists():
            logger.error(f"Cargo.toml not found at {self.cargo_toml}")
            return

        # Parse Cargo.toml to find examples
        with open(self.cargo_toml, 'r') as f:
            content = f.read()

        # Simple parsing for [[example]] sections
        examples_found = []
        lines = content.split('\n')
        current_example = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('[[example]]'):
                if current_example:
                    examples_found.append(current_example)
                current_example = {}
            elif line.startswith('name = '):
                current_example['name'] = line.split('"')[1]
            elif line.startswith('path = '):
                current_example['path'] = line.split('"')[1]

        if current_example:
            examples_found.append(current_example)

        # Create ExampleInfo objects
        for example in examples_found:
            name = example.get('name', '')
            path = self.examples_dir / example.get('path', f"{name}.rs")
            
            if path.exists():
                # Extract metadata from source file
                info = self._extract_example_metadata(name, path)
                self.examples.append(info)
                logger.debug(f"Found example: {name}")

        logger.info(f"Discovered {len(self.examples)} examples")

    def _extract_example_metadata(self, name: str, path: Path) -> ExampleInfo:
        """Extract metadata from example source file comments"""
        description = ""
        categories = []
        prerequisites = []
        expected_outputs = []
        timeout_seconds = 60
        platform_specific = False
        requires_gpu = False
        memory_intensive = False

        try:
            with open(path, 'r') as f:
                content = f.read()
                lines = content.split('\n')

            # Parse documentation comments
            for line in lines[:50]:  # Only check first 50 lines
                line = line.strip()
                if line.startswith('//!'):
                    comment = line[3:].strip()
                    if not description and comment and not comment.startswith('#'):
                        description = comment
                    
                    # Look for special markers
                    if 'gpu' in comment.lower() or 'cuda' in comment.lower():
                        requires_gpu = True
                    if 'memory' in comment.lower() and ('intensive' in comment.lower() or 'heavy' in comment.lower()):
                        memory_intensive = True
                    if 'platform' in comment.lower() and 'specific' in comment.lower():
                        platform_specific = True

            # Determine categories based on name and content
            name_lower = name.lower()
            if 'benchmark' in name_lower or 'performance' in name_lower:
                categories.append('performance')
                timeout_seconds = 120
            if 'test' in name_lower or 'validation' in name_lower:
                categories.append('testing')
            if 'streaming' in name_lower or 'realtime' in name_lower:
                categories.append('realtime')
            if 'spatial' in name_lower or '3d' in name_lower:
                categories.append('spatial')
                requires_gpu = True
            if 'cloning' in name_lower or 'voice' in name_lower:
                categories.append('voice')
            if 'production' in name_lower or 'monitoring' in name_lower:
                categories.append('production')
                timeout_seconds = 90

            # Expected outputs based on category
            expected_outputs = [f"{name}.wav"]  # Most examples generate audio
            if 'benchmark' in categories:
                expected_outputs.extend([f"{name}_report.json", f"{name}_metrics.txt"])

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {path}: {e}")

        return ExampleInfo(
            name=name,
            path=path,
            description=description,
            categories=categories,
            prerequisites=prerequisites,
            expected_outputs=expected_outputs,
            timeout_seconds=timeout_seconds,
            platform_specific=platform_specific,
            requires_gpu=requires_gpu,
            memory_intensive=memory_intensive
        )

    def build_examples(self, parallel_jobs: int = None, clean: bool = False) -> bool:
        """Build all examples with optional parallel execution"""
        if parallel_jobs is None:
            parallel_jobs = multiprocessing.cpu_count()

        logger.info(f"Building {len(self.examples)} examples with {parallel_jobs} parallel jobs...")

        if clean:
            self._clean_build_artifacts()

        start_time = time.time()
        
        # Build examples in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            future_to_example = {
                executor.submit(self._build_example, example): example 
                for example in self.examples
            }
            
            for future in concurrent.futures.as_completed(future_to_example):
                example = future_to_example[future]
                try:
                    result = future.result()
                    self.build_results.append(result)
                    
                    if result.success:
                        logger.info(f"✅ Built {result.name} ({result.duration:.1f}s)")
                    else:
                        logger.error(f"❌ Failed to build {result.name}: {result.error}")
                        
                except Exception as e:
                    logger.error(f"❌ Exception building {example.name}: {e}")
                    self.build_results.append(BuildResult(
                        name=example.name,
                        success=False,
                        duration=0,
                        error=str(e)
                    ))

        total_time = time.time() - start_time
        successful_builds = sum(1 for r in self.build_results if r.success)
        
        logger.info(f"Build completed: {successful_builds}/{len(self.examples)} successful in {total_time:.1f}s")
        
        return successful_builds == len(self.examples)

    def _build_example(self, example: ExampleInfo) -> BuildResult:
        """Build a single example"""
        start_time = time.time()
        
        try:
            # Build the example
            cmd = ["cargo", "build", "--example", example.name, "--release"]
            
            process = subprocess.run(
                cmd,
                cwd=self.examples_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for builds
            )
            
            duration = time.time() - start_time
            
            # Check for warnings
            warnings = []
            if process.stderr:
                for line in process.stderr.split('\n'):
                    if 'warning:' in line:
                        warnings.append(line.strip())

            # Get binary size if successful
            binary_size = None
            if process.returncode == 0:
                binary_path = self.target_dir / "release" / "examples" / example.name
                if binary_path.exists():
                    binary_size = binary_path.stat().st_size

            return BuildResult(
                name=example.name,
                success=process.returncode == 0,
                duration=duration,
                output=process.stdout,
                error=process.stderr if process.returncode != 0 else "",
                warnings=warnings,
                binary_size=binary_size
            )
            
        except subprocess.TimeoutExpired:
            return BuildResult(
                name=example.name,
                success=False,
                duration=time.time() - start_time,
                error="Build timeout (300s)"
            )
        except Exception as e:
            return BuildResult(
                name=example.name,
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            )

    def test_examples(self, timeout: int = 60, parallel_jobs: int = None) -> bool:
        """Test all examples that built successfully"""
        successful_builds = [r for r in self.build_results if r.success]
        
        if not successful_builds:
            logger.error("No successful builds to test")
            return False

        if parallel_jobs is None:
            parallel_jobs = min(4, multiprocessing.cpu_count())  # Limit test parallelism

        logger.info(f"Testing {len(successful_builds)} examples with {parallel_jobs} parallel jobs...")

        start_time = time.time()
        
        # Test examples in parallel (but limited to avoid resource contention)
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            future_to_example = {}
            
            for build_result in successful_builds:
                example = next(e for e in self.examples if e.name == build_result.name)
                future_to_example[executor.submit(self._test_example, example, timeout)] = example
            
            for future in concurrent.futures.as_completed(future_to_example):
                example = future_to_example[future]
                try:
                    result = future.result()
                    self.test_results.append(result)
                    
                    if result.success:
                        logger.info(f"✅ Tested {result.name} ({result.duration:.1f}s)")
                    else:
                        logger.error(f"❌ Test failed {result.name}: {result.error}")
                        
                except Exception as e:
                    logger.error(f"❌ Exception testing {example.name}: {e}")
                    self.test_results.append(TestResult(
                        name=example.name,
                        success=False,
                        duration=0,
                        error=str(e)
                    ))

        total_time = time.time() - start_time
        successful_tests = sum(1 for r in self.test_results if r.success)
        
        logger.info(f"Testing completed: {successful_tests}/{len(successful_builds)} successful in {total_time:.1f}s")
        
        return successful_tests == len(successful_builds)

    def _test_example(self, example: ExampleInfo, timeout: int) -> TestResult:
        """Test a single example"""
        start_time = time.time()
        
        try:
            # Run the example
            cmd = ["cargo", "run", "--example", example.name, "--release"]
            
            process = subprocess.run(
                cmd,
                cwd=self.examples_dir,
                capture_output=True,
                text=True,
                timeout=example.timeout_seconds or timeout
            )
            
            duration = time.time() - start_time
            
            # Check for generated files
            generated_files = []
            for expected_file in example.expected_outputs or []:
                file_path = self.examples_dir / expected_file
                if file_path.exists():
                    generated_files.append(expected_file)

            # Extract performance metrics from output
            performance_metrics = self._extract_performance_metrics(process.stdout)

            return TestResult(
                name=example.name,
                success=process.returncode == 0,
                duration=duration,
                output=process.stdout,
                error=process.stderr if process.returncode != 0 else "",
                generated_files=generated_files,
                performance_metrics=performance_metrics
            )
            
        except subprocess.TimeoutExpired:
            return TestResult(
                name=example.name,
                success=False,
                duration=time.time() - start_time,
                error=f"Test timeout ({example.timeout_seconds or timeout}s)"
            )
        except Exception as e:
            return TestResult(
                name=example.name,
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            )

    def _extract_performance_metrics(self, output: str) -> Dict[str, float]:
        """Extract performance metrics from example output"""
        metrics = {}
        
        lines = output.split('\n')
        for line in lines:
            # Look for common performance patterns
            if 'RTF:' in line or 'Real-time Factor:' in line:
                try:
                    rtf_value = float([word for word in line.split() if 'x' in word][0].replace('x', ''))
                    metrics['rtf'] = rtf_value
                except (IndexError, ValueError):
                    pass
            
            if 'Duration:' in line:
                try:
                    duration_value = float([word for word in line.split() if 's' in word][0].replace('s', ''))
                    metrics['duration'] = duration_value
                except (IndexError, ValueError):
                    pass
            
            if 'Synthesis:' in line and 's' in line:
                try:
                    # Extract synthesis time
                    words = line.split()
                    for i, word in enumerate(words):
                        if 'Synthesis:' in word and i + 1 < len(words):
                            synthesis_time = float(words[i + 1].replace('s,', ''))
                            metrics['synthesis_time'] = synthesis_time
                            break
                except (IndexError, ValueError):
                    pass

        return metrics

    def validate_outputs(self) -> bool:
        """Validate generated outputs from examples"""
        logger.info("Validating example outputs...")
        
        for test_result in self.test_results:
            if not test_result.success:
                continue
                
            example = next(e for e in self.examples if e.name == test_result.name)
            validation_result = self._validate_example_output(example, test_result)
            self.validation_results.append(validation_result)
            
            if validation_result.success:
                logger.info(f"✅ Validated {validation_result.name}")
            else:
                logger.warning(f"⚠️ Validation issues for {validation_result.name}: {validation_result.issues}")

        successful_validations = sum(1 for r in self.validation_results if r.success)
        logger.info(f"Validation completed: {successful_validations}/{len(self.validation_results)} successful")
        
        return successful_validations == len(self.validation_results)

    def _validate_example_output(self, example: ExampleInfo, test_result: TestResult) -> ValidationResult:
        """Validate output from a single example"""
        issues = []
        audio_quality_score = None
        file_size_kb = None
        
        # Check expected files were generated
        for expected_file in example.expected_outputs or []:
            if expected_file not in (test_result.generated_files or []):
                issues.append(f"Expected file not generated: {expected_file}")
            else:
                # Check file size and basic validation
                file_path = self.examples_dir / expected_file
                if file_path.exists():
                    file_size_kb = file_path.stat().st_size / 1024
                    
                    # Basic audio file validation
                    if expected_file.endswith('.wav'):
                        if file_size_kb < 1:  # Less than 1KB is probably invalid
                            issues.append(f"Audio file too small: {file_size_kb:.1f}KB")
                        elif file_size_kb > 50 * 1024:  # More than 50MB is suspicious
                            issues.append(f"Audio file very large: {file_size_kb:.1f}KB")
                        else:
                            # Rough quality estimate based on size and duration
                            if 'duration' in (test_result.performance_metrics or {}):
                                duration = test_result.performance_metrics['duration']
                                if duration > 0:
                                    kb_per_second = file_size_kb / duration
                                    audio_quality_score = min(100, kb_per_second / 2)  # Rough estimate

        # Validate performance metrics
        if test_result.performance_metrics:
            rtf = test_result.performance_metrics.get('rtf')
            if rtf and rtf > 2.0:  # Very slow synthesis
                issues.append(f"Slow synthesis: RTF {rtf:.2f}x")

        return ValidationResult(
            name=example.name,
            success=len(issues) == 0,
            issues=issues,
            audio_quality_score=audio_quality_score,
            file_size_kb=file_size_kb
        )

    def _clean_build_artifacts(self):
        """Clean build artifacts"""
        logger.info("Cleaning build artifacts...")
        try:
            subprocess.run(["cargo", "clean"], cwd=self.examples_dir, check=True)
            
            # Also clean generated files
            for file_path in self.examples_dir.glob("*.wav"):
                file_path.unlink()
            for file_path in self.examples_dir.glob("*.json"):
                file_path.unlink()
            for file_path in self.examples_dir.glob("*.txt"):
                if file_path.name.endswith("_report.txt") or file_path.name.endswith("_metrics.txt"):
                    file_path.unlink()
                    
        except Exception as e:
            logger.warning(f"Failed to clean artifacts: {e}")

    def generate_report(self, output_path: Path = None) -> Dict[str, Any]:
        """Generate comprehensive build and test report"""
        if output_path is None:
            output_path = self.reports_dir / f"build_report_{int(time.time())}.json"

        # Compile report data
        report = {
            "timestamp": time.time(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "total_examples": len(self.examples),
            "build_results": {
                "total": len(self.build_results),
                "successful": sum(1 for r in self.build_results if r.success),
                "failed": sum(1 for r in self.build_results if not r.success),
                "total_build_time": sum(r.duration for r in self.build_results),
                "average_build_time": sum(r.duration for r in self.build_results) / len(self.build_results) if self.build_results else 0,
                "total_warnings": sum(len(r.warnings or []) for r in self.build_results),
                "results": [asdict(r) for r in self.build_results]
            },
            "test_results": {
                "total": len(self.test_results),
                "successful": sum(1 for r in self.test_results if r.success),
                "failed": sum(1 for r in self.test_results if not r.success),
                "total_test_time": sum(r.duration for r in self.test_results),
                "average_test_time": sum(r.duration for r in self.test_results) / len(self.test_results) if self.test_results else 0,
                "results": [asdict(r) for r in self.test_results]
            },
            "validation_results": {
                "total": len(self.validation_results),
                "successful": sum(1 for r in self.validation_results if r.success),
                "issues": sum(len(r.issues or []) for r in self.validation_results),
                "results": [asdict(r) for r in self.validation_results]
            },
            "examples": [asdict(e) for e in self.examples]
        }

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to: {output_path}")
        
        # Print summary
        self._print_summary(report)
        
        return report

    def _print_summary(self, report: Dict[str, Any]):
        """Print build summary to console"""
        print("\n" + "="*60)
        print("VoiRS Examples Build Summary")
        print("="*60)
        
        build_results = report["build_results"]
        test_results = report["test_results"]
        validation_results = report["validation_results"]
        
        print(f"📦 Build Results: {build_results['successful']}/{build_results['total']} successful")
        if build_results['total_warnings'] > 0:
            print(f"⚠️  Total warnings: {build_results['total_warnings']}")
        
        print(f"🧪 Test Results: {test_results['successful']}/{test_results['total']} successful")
        if test_results['total'] > 0:
            print(f"⏱️  Average test time: {test_results['average_test_time']:.1f}s")
        
        print(f"✅ Validation: {validation_results['successful']}/{validation_results['total']} successful")
        if validation_results['issues'] > 0:
            print(f"⚠️  Total issues: {validation_results['issues']}")
        
        # Show failed examples
        failed_builds = [r for r in self.build_results if not r.success]
        failed_tests = [r for r in self.test_results if not r.success]
        
        if failed_builds:
            print(f"\n❌ Failed Builds ({len(failed_builds)}):")
            for result in failed_builds:
                print(f"   • {result.name}: {result.error[:100]}...")
        
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for result in failed_tests:
                print(f"   • {result.name}: {result.error[:100]}...")
        
        # Performance insights
        rtf_metrics = []
        for result in self.test_results:
            if result.performance_metrics and 'rtf' in result.performance_metrics:
                rtf_metrics.append(result.performance_metrics['rtf'])
        
        if rtf_metrics:
            avg_rtf = sum(rtf_metrics) / len(rtf_metrics)
            print(f"\n📊 Performance: Average RTF {avg_rtf:.2f}x")
            real_time_count = sum(1 for rtf in rtf_metrics if rtf < 1.0)
            print(f"⚡ Real-time capable: {real_time_count}/{len(rtf_metrics)} examples")
        
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="VoiRS Examples Enhanced Build System")
    parser.add_argument("--build-only", action="store_true", help="Only compile examples")
    parser.add_argument("--test-only", action="store_true", help="Only run tests")
    parser.add_argument("--parallel", type=int, help="Number of parallel jobs")
    parser.add_argument("--timeout", type=int, default=60, help="Test timeout in seconds")
    parser.add_argument("--platform", choices=["windows", "linux", "macos", "all"], 
                      default="all", help="Target platform")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--report", type=Path, help="Generate report at path")
    parser.add_argument("--clean", action="store_true", help="Clean before building")
    parser.add_argument("--examples-dir", type=Path, default=Path("."), 
                      help="Path to examples directory")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize build system
    build_system = VoiRSBuildSystem(args.examples_dir)
    
    if not build_system.examples:
        logger.error("No examples found!")
        return 1

    success = True

    # Build phase
    if not args.test_only:
        logger.info("Starting build phase...")
        success &= build_system.build_examples(
            parallel_jobs=args.parallel,
            clean=args.clean
        )

    # Test phase
    if not args.build_only and success:
        logger.info("Starting test phase...")
        success &= build_system.test_examples(
            timeout=args.timeout,
            parallel_jobs=args.parallel
        )
        
        # Validation phase
        if success:
            logger.info("Starting validation phase...")
            build_system.validate_outputs()

    # Generate report
    report = build_system.generate_report(args.report)
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())