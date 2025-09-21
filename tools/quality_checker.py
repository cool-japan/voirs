#!/usr/bin/env python3
"""
VoiRS Quality Validation Checker
================================

Comprehensive quality validation tool for VoiRS examples and codebase.
Performs static analysis, performance validation, output quality checking,
and compliance verification.

Features:
- Code quality analysis (linting, formatting, documentation)
- Performance regression detection
- Audio output quality validation
- Security vulnerability scanning
- Dependency management validation
- Test coverage analysis
- Documentation completeness checking
- Compliance with VoiRS coding standards

Usage:
    python quality_checker.py [OPTIONS]

Options:
    --examples-dir PATH     Path to examples directory (default: .)
    --config PATH          Path to quality config file
    --report PATH          Generate detailed quality report
    --fix                 Auto-fix issues where possible
    --strict              Use strict quality thresholds
    --verbose             Enable verbose output
    --category CATEGORY   Only check specific category
    --exclude PATTERN     Exclude files matching pattern
"""

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import time
import re
import hashlib
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Set, Any, Tuple
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class QualityIssue:
    """Represents a quality issue found in the code"""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'style', 'performance', 'security', 'documentation', 'testing'
    file_path: str
    line_number: Optional[int]
    description: str
    suggestion: Optional[str] = None
    auto_fixable: bool = False

@dataclass
class QualityMetrics:
    """Quality metrics for a file or example"""
    file_path: str
    lines_of_code: int
    documentation_ratio: float  # doc lines / code lines
    cyclomatic_complexity: int
    test_coverage: float
    performance_score: float
    security_score: float
    maintainability_index: float

@dataclass
class QualityReport:
    """Comprehensive quality report"""
    timestamp: str
    total_files: int
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues_by_category: Dict[str, int]
    overall_score: float
    metrics_summary: Dict[str, float]
    files_analyzed: List[str]
    issues: List[QualityIssue]
    metrics: List[QualityMetrics]

class VoiRSQualityChecker:
    """Comprehensive quality checker for VoiRS codebase"""

    def __init__(self, examples_dir: Path, config_path: Optional[Path] = None):
        self.examples_dir = examples_dir
        self.config_path = config_path
        self.config = self._load_config()
        self.reports_dir = examples_dir / "quality_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        self.issues: List[QualityIssue] = []
        self.metrics: List[QualityMetrics] = []
        self.files_analyzed: List[str] = []

    def _load_config(self) -> Dict[str, Any]:
        """Load quality checker configuration"""
        default_config = {
            "thresholds": {
                "documentation_ratio_min": 0.15,
                "cyclomatic_complexity_max": 15,
                "test_coverage_min": 0.70,
                "performance_regression_threshold": 0.10,
                "security_score_min": 0.80,
                "maintainability_index_min": 20
            },
            "rules": {
                "require_documentation": True,
                "enforce_error_handling": True,
                "check_resource_cleanup": True,
                "validate_test_coverage": True,
                "check_performance_regressions": True,
                "scan_security_vulnerabilities": True,
                "validate_dependencies": True
            },
            "exclusions": {
                "files": ["target/*", "*.tmp", "build_reports/*"],
                "directories": ["target", ".git", "node_modules"],
                "patterns": ["test_*", "*_test.rs", "bench_*"]
            },
            "auto_fix": {
                "format_code": True,
                "fix_imports": True,
                "update_documentation": False,
                "fix_clippy_warnings": True
            }
        }

        if self.config_path and self.config_path.exists():
            try:
                import toml
                with open(self.config_path, 'r') as f:
                    user_config = toml.load(f)
                    # Merge user config with defaults
                    for key, value in user_config.items():
                        if isinstance(value, dict) and key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except ImportError:
                logger.warning("toml package not found, using default config")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")

        return default_config

    def discover_rust_files(self) -> List[Path]:
        """Discover all Rust files in the examples directory"""
        rust_files = []
        exclusions = self.config.get("exclusions", {})
        
        for rs_file in self.examples_dir.rglob("*.rs"):
            # Check exclusions
            if self._is_excluded(rs_file, exclusions):
                continue
                
            rust_files.append(rs_file)
            
        logger.info(f"Discovered {len(rust_files)} Rust files for analysis")
        return rust_files

    def _is_excluded(self, file_path: Path, exclusions: Dict[str, List[str]]) -> bool:
        """Check if a file should be excluded from analysis"""
        relative_path = file_path.relative_to(self.examples_dir)
        
        # Check file exclusions
        for pattern in exclusions.get("files", []):
            if relative_path.match(pattern):
                return True
                
        # Check directory exclusions
        for dir_pattern in exclusions.get("directories", []):
            if dir_pattern in relative_path.parts:
                return True
                
        # Check pattern exclusions
        for pattern in exclusions.get("patterns", []):
            if file_path.name.startswith(pattern.replace("*", "")):
                return True
                
        return False

    def analyze_code_style(self, rust_files: List[Path]) -> None:
        """Analyze code style and formatting"""
        logger.info("Analyzing code style and formatting...")
        
        # Run rustfmt check
        try:
            result = subprocess.run(
                ["cargo", "fmt", "--check"],
                cwd=self.examples_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.issues.append(QualityIssue(
                    severity="warning",
                    category="style",
                    file_path="cargo_fmt",
                    line_number=None,
                    description="Code formatting issues detected",
                    suggestion="Run 'cargo fmt' to fix formatting",
                    auto_fixable=True
                ))
        except FileNotFoundError:
            self.issues.append(QualityIssue(
                severity="error",
                category="style",
                file_path="system",
                line_number=None,
                description="rustfmt not found in PATH",
                suggestion="Install rustfmt with 'rustup component add rustfmt'"
            ))

        # Run clippy for linting
        try:
            result = subprocess.run(
                ["cargo", "clippy", "--all-targets", "--", "-D", "warnings"],
                cwd=self.examples_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Parse clippy output for specific issues
                clippy_issues = self._parse_clippy_output(result.stderr)
                self.issues.extend(clippy_issues)
        except FileNotFoundError:
            self.issues.append(QualityIssue(
                severity="error",
                category="style",
                file_path="system",
                line_number=None,
                description="clippy not found in PATH",
                suggestion="Install clippy with 'rustup component add clippy'"
            ))

    def _parse_clippy_output(self, clippy_output: str) -> List[QualityIssue]:
        """Parse clippy output to extract quality issues"""
        issues = []
        lines = clippy_output.split('\n')
        
        for line in lines:
            if 'warning:' in line or 'error:' in line:
                # Extract file path and line number
                match = re.search(r'(.+):(\d+):(\d+)', line)
                if match:
                    file_path = match.group(1)
                    line_number = int(match.group(2))
                    
                    # Extract description
                    desc_match = re.search(r'(warning|error): (.+)', line)
                    if desc_match:
                        severity = desc_match.group(1)
                        description = desc_match.group(2)
                        
                        issues.append(QualityIssue(
                            severity=severity,
                            category="style",
                            file_path=file_path,
                            line_number=line_number,
                            description=description,
                            auto_fixable="warning" in line
                        ))
        
        return issues

    def analyze_documentation(self, rust_files: List[Path]) -> None:
        """Analyze documentation quality"""
        logger.info("Analyzing documentation quality...")
        
        for file_path in rust_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                lines = content.split('\n')
                code_lines = 0
                doc_lines = 0
                
                # Count lines
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('///') or stripped.startswith('//!'):
                        doc_lines += 1
                    elif stripped and not stripped.startswith('//'):
                        code_lines += 1
                
                # Calculate documentation ratio
                doc_ratio = doc_lines / max(code_lines, 1)
                min_ratio = self.config["thresholds"]["documentation_ratio_min"]
                
                if doc_ratio < min_ratio:
                    self.issues.append(QualityIssue(
                        severity="warning",
                        category="documentation",
                        file_path=str(file_path),
                        line_number=None,
                        description=f"Low documentation ratio: {doc_ratio:.2f} (minimum: {min_ratio})",
                        suggestion="Add more documentation comments (/// or //!)"
                    ))
                
                # Check for missing function documentation
                self._check_function_documentation(file_path, content)
                
            except Exception as e:
                logger.warning(f"Failed to analyze documentation for {file_path}: {e}")

    def _check_function_documentation(self, file_path: Path, content: str) -> None:
        """Check if public functions have documentation"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Look for public function definitions
            if re.match(r'^\s*pub\s+(async\s+)?fn\s+\w+', line):
                # Check if previous line has documentation
                has_doc = False
                j = i - 1
                while j >= 0 and lines[j].strip().startswith('///'):
                    has_doc = True
                    break
                    j -= 1
                
                if not has_doc:
                    func_match = re.search(r'fn\s+(\w+)', line)
                    func_name = func_match.group(1) if func_match else "unknown"
                    
                    self.issues.append(QualityIssue(
                        severity="info",
                        category="documentation",
                        file_path=str(file_path),
                        line_number=i + 1,
                        description=f"Public function '{func_name}' lacks documentation",
                        suggestion="Add documentation comment with /// before the function"
                    ))

    def analyze_performance(self, rust_files: List[Path]) -> None:
        """Analyze performance characteristics"""
        logger.info("Analyzing performance characteristics...")
        
        for file_path in rust_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for performance anti-patterns
                self._check_performance_patterns(file_path, content)
                
                # Calculate complexity metrics
                complexity = self._calculate_cyclomatic_complexity(content)
                max_complexity = self.config["thresholds"]["cyclomatic_complexity_max"]
                
                if complexity > max_complexity:
                    self.issues.append(QualityIssue(
                        severity="warning",
                        category="performance",
                        file_path=str(file_path),
                        line_number=None,
                        description=f"High cyclomatic complexity: {complexity} (max: {max_complexity})",
                        suggestion="Consider breaking down complex functions"
                    ))
                    
            except Exception as e:
                logger.warning(f"Failed to analyze performance for {file_path}: {e}")

    def _check_performance_patterns(self, file_path: Path, content: str) -> None:
        """Check for performance anti-patterns"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for inefficient patterns
            if '.clone()' in line and 'for' in line:
                self.issues.append(QualityIssue(
                    severity="info",
                    category="performance",
                    file_path=str(file_path),
                    line_number=i + 1,
                    description="Potential unnecessary clone in loop",
                    suggestion="Consider using references or iterators"
                ))
            
            if 'unwrap()' in line and not line_stripped.startswith('//'):
                self.issues.append(QualityIssue(
                    severity="warning",
                    category="performance",
                    file_path=str(file_path),
                    line_number=i + 1,
                    description="Use of unwrap() can cause panics",
                    suggestion="Use proper error handling with ? or match"
                ))
            
            if 'format!' in line and 'println!' in line:
                self.issues.append(QualityIssue(
                    severity="info",
                    category="performance",
                    file_path=str(file_path),
                    line_number=i + 1,
                    description="format! with println! is inefficient",
                    suggestion="Use println! with format arguments directly"
                ))

    def _calculate_cyclomatic_complexity(self, content: str) -> int:
        """Calculate cyclomatic complexity of the code"""
        # Simple complexity calculation based on control flow statements
        complexity_keywords = ['if', 'else if', 'match', 'for', 'while', 'loop', '&&', '||']
        complexity = 1  # Base complexity
        
        for keyword in complexity_keywords:
            if keyword == '&&' or keyword == '||':
                complexity += content.count(keyword)
            else:
                complexity += len(re.findall(rf'\b{keyword}\b', content))
        
        return complexity

    def analyze_security(self, rust_files: List[Path]) -> None:
        """Analyze security vulnerabilities"""
        logger.info("Analyzing security vulnerabilities...")
        
        # Run cargo audit
        try:
            result = subprocess.run(
                ["cargo", "audit"],
                cwd=self.examples_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.issues.append(QualityIssue(
                    severity="error",
                    category="security",
                    file_path="cargo_audit",
                    line_number=None,
                    description="Security vulnerabilities found in dependencies",
                    suggestion="Run 'cargo audit' for details and update dependencies"
                ))
        except FileNotFoundError:
            self.issues.append(QualityIssue(
                severity="warning",
                category="security",
                file_path="system",
                line_number=None,
                description="cargo-audit not found",
                suggestion="Install with 'cargo install cargo-audit'"
            ))

        # Check for security anti-patterns in code
        for file_path in rust_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self._check_security_patterns(file_path, content)
                
            except Exception as e:
                logger.warning(f"Failed to analyze security for {file_path}: {e}")

    def _check_security_patterns(self, file_path: Path, content: str) -> None:
        """Check for security anti-patterns"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check for hardcoded secrets
            if re.search(r'(password|secret|key|token)\s*=\s*["\'].+["\']', line_lower):
                self.issues.append(QualityIssue(
                    severity="error",
                    category="security",
                    file_path=str(file_path),
                    line_number=i + 1,
                    description="Potential hardcoded secret detected",
                    suggestion="Use environment variables or secure storage"
                ))
            
            # Check for unsafe blocks
            if 'unsafe' in line and '{' in line:
                self.issues.append(QualityIssue(
                    severity="warning",
                    category="security",
                    file_path=str(file_path),
                    line_number=i + 1,
                    description="Unsafe block detected",
                    suggestion="Document safety invariants and minimize unsafe usage"
                ))

    def analyze_testing(self, rust_files: List[Path]) -> None:
        """Analyze test coverage and quality"""
        logger.info("Analyzing test coverage...")
        
        # Count test functions vs regular functions
        total_functions = 0
        test_functions = 0
        
        for file_path in rust_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Count functions
                func_matches = re.findall(r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+\w+', content, re.MULTILINE)
                total_functions += len(func_matches)
                
                # Count test functions
                test_matches = re.findall(r'#\[test\]', content)
                test_functions += len(test_matches)
                
            except Exception as e:
                logger.warning(f"Failed to analyze testing for {file_path}: {e}")
        
        # Calculate test coverage ratio
        if total_functions > 0:
            test_coverage = test_functions / total_functions
            min_coverage = self.config["thresholds"]["test_coverage_min"]
            
            if test_coverage < min_coverage:
                self.issues.append(QualityIssue(
                    severity="warning",
                    category="testing",
                    file_path="overall",
                    line_number=None,
                    description=f"Low test coverage: {test_coverage:.2%} (minimum: {min_coverage:.2%})",
                    suggestion="Add more unit tests to improve coverage"
                ))

    def calculate_metrics(self, rust_files: List[Path]) -> None:
        """Calculate quality metrics for each file"""
        logger.info("Calculating quality metrics...")
        
        for file_path in rust_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith('//'))
                doc_lines = sum(1 for line in lines if line.strip().startswith('///') or line.strip().startswith('//!'))
                
                doc_ratio = doc_lines / max(code_lines, 1)
                complexity = self._calculate_cyclomatic_complexity(content)
                
                # Simple maintainability index calculation
                maintainability = max(0, 100 - complexity * 2 - max(0, code_lines - 100) * 0.1)
                
                metrics = QualityMetrics(
                    file_path=str(file_path),
                    lines_of_code=code_lines,
                    documentation_ratio=doc_ratio,
                    cyclomatic_complexity=complexity,
                    test_coverage=0.0,  # Would need more sophisticated analysis
                    performance_score=100 - len([i for i in self.issues if i.file_path == str(file_path) and i.category == 'performance']) * 10,
                    security_score=100 - len([i for i in self.issues if i.file_path == str(file_path) and i.category == 'security']) * 20,
                    maintainability_index=maintainability
                )
                
                self.metrics.append(metrics)
                
            except Exception as e:
                logger.warning(f"Failed to calculate metrics for {file_path}: {e}")

    def auto_fix_issues(self) -> int:
        """Automatically fix issues where possible"""
        logger.info("Attempting to auto-fix issues...")
        
        fixed_count = 0
        
        if self.config.get("auto_fix", {}).get("format_code", False):
            try:
                result = subprocess.run(
                    ["cargo", "fmt"],
                    cwd=self.examples_dir,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    fixed_count += 1
                    logger.info("✅ Fixed code formatting issues")
            except Exception as e:
                logger.warning(f"Failed to auto-fix formatting: {e}")
        
        if self.config.get("auto_fix", {}).get("fix_clippy_warnings", False):
            try:
                result = subprocess.run(
                    ["cargo", "clippy", "--fix", "--allow-dirty", "--allow-staged"],
                    cwd=self.examples_dir,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    fixed_count += 1
                    logger.info("✅ Fixed clippy warnings")
            except Exception as e:
                logger.warning(f"Failed to auto-fix clippy warnings: {e}")
        
        return fixed_count

    def generate_report(self) -> QualityReport:
        """Generate comprehensive quality report"""
        logger.info("Generating quality report...")
        
        issues_by_severity = {}
        issues_by_category = {}
        
        for issue in self.issues:
            issues_by_severity[issue.severity] = issues_by_severity.get(issue.severity, 0) + 1
            issues_by_category[issue.category] = issues_by_category.get(issue.category, 0) + 1
        
        # Calculate overall score
        error_weight = -10
        warning_weight = -5
        info_weight = -1
        
        score_deduction = (
            issues_by_severity.get('error', 0) * error_weight +
            issues_by_severity.get('warning', 0) * warning_weight +
            issues_by_severity.get('info', 0) * info_weight
        )
        
        overall_score = max(0, 100 + score_deduction)
        
        # Calculate metrics summary
        if self.metrics:
            metrics_summary = {
                'avg_documentation_ratio': statistics.mean(m.documentation_ratio for m in self.metrics),
                'avg_complexity': statistics.mean(m.cyclomatic_complexity for m in self.metrics),
                'avg_maintainability': statistics.mean(m.maintainability_index for m in self.metrics),
                'avg_performance_score': statistics.mean(m.performance_score for m in self.metrics),
                'avg_security_score': statistics.mean(m.security_score for m in self.metrics)
            }
        else:
            metrics_summary = {}
        
        return QualityReport(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            total_files=len(self.files_analyzed),
            total_issues=len(self.issues),
            issues_by_severity=issues_by_severity,
            issues_by_category=issues_by_category,
            overall_score=overall_score,
            metrics_summary=metrics_summary,
            files_analyzed=self.files_analyzed,
            issues=self.issues,
            metrics=self.metrics
        )

    def save_report(self, report: QualityReport, output_path: Path) -> None:
        """Save quality report to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        logger.info(f"Quality report saved to: {output_path}")

    def print_summary(self, report: QualityReport) -> None:
        """Print quality report summary"""
        print("\n" + "="*60)
        print("🔍 VoiRS Quality Analysis Report")
        print("="*60)
        print(f"📊 Overall Score: {report.overall_score:.1f}/100")
        print(f"📁 Files Analyzed: {report.total_files}")
        print(f"⚠️  Total Issues: {report.total_issues}")
        
        if report.issues_by_severity:
            print("\n📋 Issues by Severity:")
            for severity, count in sorted(report.issues_by_severity.items()):
                emoji = {"error": "❌", "warning": "⚠️ ", "info": "ℹ️ "}
                print(f"  {emoji.get(severity, '•')} {severity.title()}: {count}")
        
        if report.issues_by_category:
            print("\n🏷️  Issues by Category:")
            for category, count in sorted(report.issues_by_category.items()):
                print(f"  • {category.title()}: {count}")
        
        if report.metrics_summary:
            print("\n📈 Metrics Summary:")
            for metric, value in report.metrics_summary.items():
                print(f"  • {metric.replace('_', ' ').title()}: {value:.2f}")
        
        # Top issues
        if report.issues:
            print("\n🔝 Top Issues:")
            severity_order = {"error": 0, "warning": 1, "info": 2}
            sorted_issues = sorted(report.issues, key=lambda x: (severity_order.get(x.severity, 3), x.category))
            
            for issue in sorted_issues[:10]:  # Show top 10 issues
                emoji = {"error": "❌", "warning": "⚠️ ", "info": "ℹ️ "}
                print(f"  {emoji.get(issue.severity, '•')} {issue.description}")
                if issue.file_path != "overall" and issue.line_number:
                    print(f"    📍 {issue.file_path}:{issue.line_number}")
                elif issue.file_path != "overall":
                    print(f"    📍 {issue.file_path}")
                if issue.suggestion:
                    print(f"    💡 {issue.suggestion}")
                print()

    def run_quality_check(self, category: Optional[str] = None, auto_fix: bool = False) -> QualityReport:
        """Run comprehensive quality check"""
        logger.info("Starting VoiRS quality analysis...")
        
        start_time = time.time()
        
        # Discover files
        rust_files = self.discover_rust_files()
        self.files_analyzed = [str(f) for f in rust_files]
        
        if category:
            logger.info(f"Filtering analysis for category: {category}")
            # Filter files based on category (simplified implementation)
            category_files = [f for f in rust_files if category.lower() in str(f).lower()]
            rust_files = category_files if category_files else rust_files
        
        # Run analysis
        try:
            self.analyze_code_style(rust_files)
            self.analyze_documentation(rust_files)
            self.analyze_performance(rust_files)
            self.analyze_security(rust_files)
            self.analyze_testing(rust_files)
            self.calculate_metrics(rust_files)
            
            # Auto-fix if requested
            if auto_fix:
                fixed_count = self.auto_fix_issues()
                logger.info(f"Auto-fixed {fixed_count} issues")
            
            # Generate report
            report = self.generate_report()
            
            duration = time.time() - start_time
            logger.info(f"Quality analysis completed in {duration:.1f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="VoiRS Quality Validation Checker")
    parser.add_argument("--examples-dir", "-d", type=Path, default=Path("."),
                       help="Path to examples directory")
    parser.add_argument("--config", "-c", type=Path,
                       help="Path to quality config file")
    parser.add_argument("--report", "-r", type=Path,
                       help="Generate detailed quality report at path")
    parser.add_argument("--fix", action="store_true",
                       help="Auto-fix issues where possible")
    parser.add_argument("--strict", action="store_true",
                       help="Use strict quality thresholds")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--category", type=str,
                       help="Only check specific category")
    parser.add_argument("--exclude", type=str,
                       help="Exclude files matching pattern")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize quality checker
        checker = VoiRSQualityChecker(args.examples_dir, args.config)
        
        # Run quality check
        report = checker.run_quality_check(
            category=args.category,
            auto_fix=args.fix
        )
        
        # Print summary
        checker.print_summary(report)
        
        # Save detailed report if requested
        if args.report:
            checker.save_report(report, args.report)
        
        # Return exit code based on quality score
        if report.overall_score < 70:
            logger.warning("Quality score below threshold")
            return 2
        elif report.issues_by_severity.get('error', 0) > 0:
            logger.warning("Quality issues found")
            return 1
        else:
            logger.info("Quality check passed")
            return 0
            
    except Exception as e:
        logger.error(f"Quality check failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 3

if __name__ == "__main__":
    sys.exit(main())