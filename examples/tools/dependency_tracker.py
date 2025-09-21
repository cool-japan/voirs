#!/usr/bin/env python3
"""
VoiRS Examples Dependency Management Tool

This tool manages and tracks dependencies across all VoiRS examples,
ensuring compatibility and handling version updates.
"""

import json
import toml
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
import argparse
from datetime import datetime


@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    version: str
    features: List[str]
    optional: bool
    used_in_examples: List[str]
    last_updated: str
    is_workspace_dep: bool


@dataclass
class DependencyReport:
    """Complete dependency analysis report."""
    total_dependencies: int
    unique_dependencies: int
    workspace_dependencies: int
    outdated_dependencies: List[str]
    conflicting_versions: Dict[str, List[str]]
    unused_dependencies: List[str]
    security_issues: List[str]
    compatibility_matrix: Dict[str, Dict[str, str]]
    recommendations: List[str]


class DependencyTracker:
    """Manages dependencies across VoiRS examples."""
    
    def __init__(self, examples_dir: Path):
        self.examples_dir = examples_dir
        self.cargo_toml_path = examples_dir / "Cargo.toml"
        self.dependencies: Dict[str, DependencyInfo] = {}
        self.examples: List[Path] = []
        
    def discover_examples(self) -> None:
        """Discover all Rust example files."""
        self.examples = list(self.examples_dir.glob("*.rs"))
        print(f"📁 Discovered {len(self.examples)} example files")
        
    def parse_cargo_toml(self) -> Dict:
        """Parse the main Cargo.toml file."""
        if not self.cargo_toml_path.exists():
            raise FileNotFoundError(f"Cargo.toml not found at {self.cargo_toml_path}")
            
        with open(self.cargo_toml_path, 'r') as f:
            return toml.load(f)
            
    def analyze_dependencies(self) -> None:
        """Analyze all dependencies used in examples."""
        print("🔍 Analyzing dependencies...")
        
        cargo_data = self.parse_cargo_toml()
        deps = cargo_data.get('dependencies', {})
        workspace_deps = cargo_data.get('workspace', {}).get('dependencies', {})
        
        for name, info in deps.items():
            dep_info = self._parse_dependency_info(name, info, workspace_deps)
            dep_info.used_in_examples = self._find_examples_using_dependency(name)
            self.dependencies[name] = dep_info
            
    def _parse_dependency_info(self, name: str, info, workspace_deps: Dict) -> DependencyInfo:
        """Parse dependency information from Cargo.toml."""
        if isinstance(info, str):
            version = info
            features = []
            optional = False
        elif isinstance(info, dict):
            version = info.get('version', '')
            features = info.get('features', [])
            optional = info.get('optional', False)
        else:
            version = ''
            features = []
            optional = False
            
        is_workspace_dep = name in workspace_deps
        
        return DependencyInfo(
            name=name,
            version=version,
            features=features,
            optional=optional,
            used_in_examples=[],
            last_updated=datetime.now().isoformat(),
            is_workspace_dep=is_workspace_dep
        )
        
    def _find_examples_using_dependency(self, dep_name: str) -> List[str]:
        """Find which examples use a specific dependency."""
        using_examples = []
        
        for example_path in self.examples:
            try:
                with open(example_path, 'r') as f:
                    content = f.read()
                    # Look for use statements and crate references
                    if (f"use {dep_name}" in content or 
                        f"extern crate {dep_name}" in content or
                        f"{dep_name}::" in content):
                        using_examples.append(example_path.name)
            except Exception as e:
                print(f"⚠️  Warning: Could not read {example_path}: {e}")
                
        return using_examples
        
    def check_outdated_dependencies(self) -> List[str]:
        """Check for outdated dependencies using cargo outdated."""
        try:
            result = subprocess.run(
                ['cargo', 'outdated', '--format', 'json'],
                cwd=self.examples_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                outdated_data = json.loads(result.stdout)
                return [dep['name'] for dep in outdated_data.get('dependencies', [])]
            else:
                print("⚠️  Could not check outdated dependencies. Install with: cargo install cargo-outdated")
                return []
                
        except (subprocess.SubprocessError, json.JSONDecodeError):
            return []
            
    def check_security_issues(self) -> List[str]:
        """Check for security vulnerabilities using cargo audit."""
        try:
            result = subprocess.run(
                ['cargo', 'audit', '--format', 'json'],
                cwd=self.examples_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                audit_data = json.loads(result.stdout)
                vulnerabilities = audit_data.get('vulnerabilities', {}).get('list', [])
                return [vuln['package']['name'] for vuln in vulnerabilities]
            else:
                print("⚠️  Could not check security issues. Install with: cargo install cargo-audit")
                return []
                
        except (subprocess.SubprocessError, json.JSONDecodeError):
            return []
            
    def find_version_conflicts(self) -> Dict[str, List[str]]:
        """Find dependencies with conflicting versions."""
        conflicts = {}
        
        # This would require more sophisticated analysis
        # For now, we'll check for common patterns
        for name, dep in self.dependencies.items():
            if name.startswith("voirs-"):
                # Check if all VoiRS crates use the same version
                version = dep.version
                if version not in conflicts.get("voirs_versions", []):
                    conflicts.setdefault("voirs_versions", []).append(version)
                    
        return {k: v for k, v in conflicts.items() if len(v) > 1}
        
    def find_unused_dependencies(self) -> List[str]:
        """Find dependencies that aren't used in any examples."""
        unused = []
        for name, dep in self.dependencies.items():
            if not dep.used_in_examples and not dep.optional:
                unused.append(name)
        return unused
        
    def generate_compatibility_matrix(self) -> Dict[str, Dict[str, str]]:
        """Generate a compatibility matrix showing which examples work with which dependencies."""
        matrix = {}
        
        for example_path in self.examples:
            example_name = example_path.stem
            matrix[example_name] = {}
            
            for dep_name, dep in self.dependencies.items():
                if example_name + ".rs" in dep.used_in_examples:
                    matrix[example_name][dep_name] = dep.version
                    
        return matrix
        
    def generate_recommendations(self, report: DependencyReport) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if report.outdated_dependencies:
            recommendations.append(
                f"📈 Update {len(report.outdated_dependencies)} outdated dependencies: "
                f"{', '.join(report.outdated_dependencies[:3])}{'...' if len(report.outdated_dependencies) > 3 else ''}"
            )
            
        if report.unused_dependencies:
            recommendations.append(
                f"🧹 Remove {len(report.unused_dependencies)} unused dependencies to reduce build time"
            )
            
        if report.security_issues:
            recommendations.append(
                f"🔒 Address {len(report.security_issues)} security vulnerabilities immediately"
            )
            
        if report.conflicting_versions:
            recommendations.append(
                "⚖️  Resolve version conflicts to ensure consistent behavior"
            )
            
        if report.workspace_dependencies < report.unique_dependencies * 0.8:
            recommendations.append(
                "🏗️  Consider moving more dependencies to workspace configuration"
            )
            
        return recommendations
        
    def generate_report(self) -> DependencyReport:
        """Generate a comprehensive dependency report."""
        print("📊 Generating dependency report...")
        
        outdated = self.check_outdated_dependencies()
        security_issues = self.check_security_issues()
        conflicts = self.find_version_conflicts()
        unused = self.find_unused_dependencies()
        compat_matrix = self.generate_compatibility_matrix()
        
        report = DependencyReport(
            total_dependencies=len(self.dependencies),
            unique_dependencies=len(self.dependencies),
            workspace_dependencies=sum(1 for d in self.dependencies.values() if d.is_workspace_dep),
            outdated_dependencies=outdated,
            conflicting_versions=conflicts,
            unused_dependencies=unused,
            security_issues=security_issues,
            compatibility_matrix=compat_matrix,
            recommendations=[]
        )
        
        report.recommendations = self.generate_recommendations(report)
        return report
        
    def save_report(self, report: DependencyReport, output_path: Path) -> None:
        """Save the dependency report to a file."""
        report_data = {
            'report': asdict(report),
            'dependencies': {name: asdict(dep) for name, dep in self.dependencies.items()},
            'generated_at': datetime.now().isoformat(),
            'examples_analyzed': len(self.examples)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        print(f"📄 Report saved to {output_path}")
        
    def print_summary(self, report: DependencyReport) -> None:
        """Print a summary of the dependency analysis."""
        print("\n" + "="*60)
        print("🎯 VOIRS DEPENDENCY ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"📦 Total Dependencies: {report.total_dependencies}")
        print(f"🏗️  Workspace Dependencies: {report.workspace_dependencies}")
        print(f"📁 Examples Analyzed: {len(self.examples)}")
        
        if report.outdated_dependencies:
            print(f"📈 Outdated: {len(report.outdated_dependencies)}")
            
        if report.security_issues:
            print(f"🔒 Security Issues: {len(report.security_issues)}")
            
        if report.unused_dependencies:
            print(f"🧹 Unused: {len(report.unused_dependencies)}")
            
        print("\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")
            
        if not report.recommendations:
            print("✅ All dependencies look good!")


def main():
    parser = argparse.ArgumentParser(description="VoiRS Examples Dependency Tracker")
    parser.add_argument("--examples-dir", type=Path, default=Path.cwd(),
                       help="Path to examples directory")
    parser.add_argument("--output", type=Path, default=Path("dependency_report.json"),
                       help="Output file for detailed report")
    parser.add_argument("--update", action="store_true",
                       help="Attempt to update outdated dependencies")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check dependencies, don't generate full report")
    
    args = parser.parse_args()
    
    try:
        tracker = DependencyTracker(args.examples_dir)
        tracker.discover_examples()
        tracker.analyze_dependencies()
        
        if args.check_only:
            outdated = tracker.check_outdated_dependencies()
            security = tracker.check_security_issues()
            
            if outdated:
                print(f"📈 Outdated dependencies: {', '.join(outdated)}")
            if security:
                print(f"🔒 Security issues: {', '.join(security)}")
            if not outdated and not security:
                print("✅ All dependencies are up to date and secure!")
            return
            
        report = tracker.generate_report()
        tracker.print_summary(report)
        tracker.save_report(report, args.output)
        
        if args.update and report.outdated_dependencies:
            print("\n🔄 Updating outdated dependencies...")
            subprocess.run(['cargo', 'update'], cwd=args.examples_dir)
            print("✅ Dependencies updated!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()