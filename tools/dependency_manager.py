#!/usr/bin/env python3
"""
VoiRS Dependency Management System
==================================

Comprehensive dependency management tool for VoiRS examples and crates.
Manages Cargo.toml dependencies, detects version conflicts, suggests updates,
and ensures compatibility across the entire workspace.

Features:
- Workspace dependency analysis and management
- Version conflict detection and resolution
- Automated dependency updates with compatibility checking
- Security vulnerability scanning in dependencies
- License compliance checking
- Dependency graph visualization
- Unused dependency detection
- Dependency deduplication recommendations

Usage:
    python dependency_manager.py [OPTIONS]

Options:
    --workspace-dir PATH    Path to workspace root (default: ..)
    --examples-dir PATH     Path to examples directory (default: .)
    --config PATH          Path to dependency config file
    --report PATH          Generate detailed dependency report
    --update               Update dependencies to latest compatible versions
    --check-security       Check for security vulnerabilities
    --check-licenses       Check license compatibility
    --fix-conflicts        Automatically fix version conflicts
    --unused               Detect and report unused dependencies
    --verbose              Enable verbose output
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import re
import requests
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Set, Any, Tuple
from packaging import version
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class Dependency:
    """Represents a Rust dependency"""
    name: str
    version: str
    source: str = "crates.io"  # crates.io, git, path, etc.
    features: List[str] = None
    optional: bool = False
    workspace: bool = False
    
    def __post_init__(self):
        if self.features is None:
            self.features = []

@dataclass
class CrateInfo:
    """Information about a crate in the workspace"""
    name: str
    path: Path
    cargo_toml_path: Path
    dependencies: List[Dependency]
    dev_dependencies: List[Dependency]
    build_dependencies: List[Dependency]
    is_example: bool = False
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.dev_dependencies is None:
            self.dev_dependencies = []
        if self.build_dependencies is None:
            self.build_dependencies = []

@dataclass
class VersionConflict:
    """Represents a version conflict between dependencies"""
    dependency_name: str
    conflicting_versions: List[Tuple[str, str]]  # (version, source_crate)
    recommended_version: Optional[str] = None
    severity: str = "warning"  # warning, error

@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability in a dependency"""
    dependency_name: str
    affected_versions: str
    vulnerability_id: str
    severity: str
    description: str
    patched_versions: List[str]

@dataclass
class DependencyReport:
    """Comprehensive dependency analysis report"""
    timestamp: str
    workspace_crates: int
    total_dependencies: int
    unique_dependencies: int
    version_conflicts: List[VersionConflict]
    security_vulnerabilities: List[SecurityVulnerability]
    unused_dependencies: List[str]
    outdated_dependencies: List[Dict[str, Any]]
    license_issues: List[Dict[str, Any]]
    recommendations: List[str]

class VoiRSDependencyManager:
    """Comprehensive dependency manager for VoiRS workspace"""

    def __init__(self, workspace_dir: Path, examples_dir: Path, config_path: Optional[Path] = None):
        self.workspace_dir = workspace_dir
        self.examples_dir = examples_dir
        self.config_path = config_path
        self.config = self._load_config()
        self.reports_dir = examples_dir / "dependency_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        self.crates: List[CrateInfo] = []
        self.workspace_dependencies: Dict[str, Dependency] = {}
        self.crates_io_cache: Dict[str, Any] = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load dependency manager configuration"""
        default_config = {
            "allowed_licenses": [
                "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", 
                "ISC", "MPL-2.0", "CC0-1.0", "Unlicense"
            ],
            "blocked_dependencies": [],
            "preferred_versions": {},
            "security_check": {
                "enabled": True,
                "severity_threshold": "medium"
            },
            "update_policy": {
                "major_updates": False,
                "minor_updates": True,
                "patch_updates": True,
                "pre_release": False
            },
            "workspace_optimization": {
                "deduplicate_versions": True,
                "prefer_workspace_deps": True,
                "consolidate_features": True
            }
        }

        if self.config_path and self.config_path.exists():
            try:
                import toml
                with open(self.config_path, 'r') as f:
                    user_config = toml.load(f)
                    # Merge configurations
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

    def discover_crates(self) -> None:
        """Discover all crates in the workspace"""
        logger.info("Discovering crates in workspace...")
        
        # Parse workspace Cargo.toml
        workspace_toml = self.workspace_dir / "Cargo.toml"
        if workspace_toml.exists():
            self._parse_workspace_dependencies(workspace_toml)
        
        # Find all Cargo.toml files in workspace
        cargo_files = list(self.workspace_dir.glob("**/Cargo.toml"))
        
        for cargo_file in cargo_files:
            # Skip target directories
            if "target" in cargo_file.parts:
                continue
                
            try:
                crate_info = self._parse_cargo_toml(cargo_file)
                if crate_info:
                    self.crates.append(crate_info)
                    logger.debug(f"Found crate: {crate_info.name}")
            except Exception as e:
                logger.warning(f"Failed to parse {cargo_file}: {e}")
        
        logger.info(f"Discovered {len(self.crates)} crates")

    def _parse_workspace_dependencies(self, workspace_toml: Path) -> None:
        """Parse workspace-level dependencies"""
        try:
            import toml
            with open(workspace_toml, 'r') as f:
                data = toml.load(f)
            
            workspace_deps = data.get("workspace", {}).get("dependencies", {})
            
            for name, spec in workspace_deps.items():
                if isinstance(spec, str):
                    dep = Dependency(name=name, version=spec, workspace=True)
                elif isinstance(spec, dict):
                    dep = Dependency(
                        name=name,
                        version=spec.get("version", ""),
                        source=self._determine_source(spec),
                        features=spec.get("features", []),
                        optional=spec.get("optional", False),
                        workspace=True
                    )
                else:
                    continue
                
                self.workspace_dependencies[name] = dep
                
        except ImportError:
            logger.error("toml package required for parsing Cargo.toml files")
            sys.exit(1)
        except Exception as e:
            logger.warning(f"Failed to parse workspace dependencies: {e}")

    def _parse_cargo_toml(self, cargo_file: Path) -> Optional[CrateInfo]:
        """Parse a Cargo.toml file and extract dependency information"""
        try:
            import toml
            with open(cargo_file, 'r') as f:
                data = toml.load(f)
            
            package = data.get("package", {})
            crate_name = package.get("name", cargo_file.parent.name)
            
            # Determine if this is an example
            is_example = "examples" in str(cargo_file) or "example" in package.get("keywords", [])
            
            # Parse dependencies
            dependencies = self._parse_dependencies(data.get("dependencies", {}))
            dev_dependencies = self._parse_dependencies(data.get("dev-dependencies", {}))
            build_dependencies = self._parse_dependencies(data.get("build-dependencies", {}))
            
            return CrateInfo(
                name=crate_name,
                path=cargo_file.parent,
                cargo_toml_path=cargo_file,
                dependencies=dependencies,
                dev_dependencies=dev_dependencies,
                build_dependencies=build_dependencies,
                is_example=is_example
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse {cargo_file}: {e}")
            return None

    def _parse_dependencies(self, deps_dict: Dict[str, Any]) -> List[Dependency]:
        """Parse dependencies from Cargo.toml section"""
        dependencies = []
        
        for name, spec in deps_dict.items():
            if isinstance(spec, str):
                dep = Dependency(name=name, version=spec)
            elif isinstance(spec, dict):
                dep = Dependency(
                    name=name,
                    version=spec.get("version", ""),
                    source=self._determine_source(spec),
                    features=spec.get("features", []),
                    optional=spec.get("optional", False),
                    workspace=spec.get("workspace", False)
                )
            else:
                continue
            
            dependencies.append(dep)
        
        return dependencies

    def _determine_source(self, spec: Dict[str, Any]) -> str:
        """Determine the source of a dependency"""
        if "path" in spec:
            return "path"
        elif "git" in spec:
            return "git"
        elif "registry" in spec:
            return spec["registry"]
        else:
            return "crates.io"

    def analyze_version_conflicts(self) -> List[VersionConflict]:
        """Analyze version conflicts across the workspace"""
        logger.info("Analyzing version conflicts...")
        
        conflicts = []
        dependency_versions = {}
        
        # Collect all dependency versions
        for crate in self.crates:
            for dep_list in [crate.dependencies, crate.dev_dependencies, crate.build_dependencies]:
                for dep in dep_list:
                    if dep.workspace:
                        continue  # Skip workspace dependencies
                    
                    if dep.name not in dependency_versions:
                        dependency_versions[dep.name] = []
                    
                    dependency_versions[dep.name].append((dep.version, crate.name))
        
        # Find conflicts
        for dep_name, versions in dependency_versions.items():
            unique_versions = {}
            for version_spec, source_crate in versions:
                if version_spec not in unique_versions:
                    unique_versions[version_spec] = []
                unique_versions[version_spec].append(source_crate)
            
            if len(unique_versions) > 1:
                # Determine if this is a real conflict
                resolved_versions = []
                for version_spec in unique_versions.keys():
                    try:
                        # Simple version resolution (could be more sophisticated)
                        if version_spec.startswith("="):
                            resolved_versions.append(version_spec[1:])
                        elif version_spec.startswith("^") or version_spec.startswith("~"):
                            resolved_versions.append(version_spec[1:])
                        else:
                            resolved_versions.append(version_spec)
                    except:
                        resolved_versions.append(version_spec)
                
                # Check if versions are actually conflicting
                if len(set(resolved_versions)) > 1:
                    conflicting_versions = [(v, ", ".join(unique_versions[v])) for v in unique_versions.keys()]
                    
                    # Suggest recommended version
                    recommended = self._suggest_version_resolution(dep_name, resolved_versions)
                    
                    conflicts.append(VersionConflict(
                        dependency_name=dep_name,
                        conflicting_versions=conflicting_versions,
                        recommended_version=recommended,
                        severity="warning"
                    ))
        
        logger.info(f"Found {len(conflicts)} version conflicts")
        return conflicts

    def _suggest_version_resolution(self, dep_name: str, versions: List[str]) -> str:
        """Suggest a version to resolve conflicts"""
        try:
            # Get latest version from crates.io
            latest_version = self._get_latest_version(dep_name)
            if latest_version:
                return latest_version
        except:
            pass
        
        # Fallback: suggest the highest version
        try:
            sorted_versions = sorted(versions, key=lambda x: version.parse(x), reverse=True)
            return sorted_versions[0]
        except:
            return versions[0] if versions else "unknown"

    def _get_latest_version(self, crate_name: str) -> Optional[str]:
        """Get the latest version of a crate from crates.io"""
        if crate_name in self.crates_io_cache:
            return self.crates_io_cache[crate_name].get("max_version")
        
        try:
            response = requests.get(f"https://crates.io/api/v1/crates/{crate_name}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                crate_info = data.get("crate", {})
                max_version = crate_info.get("max_version")
                self.crates_io_cache[crate_name] = crate_info
                return max_version
        except:
            pass
        
        return None

    def check_security_vulnerabilities(self) -> List[SecurityVulnerability]:
        """Check for security vulnerabilities using cargo audit"""
        logger.info("Checking for security vulnerabilities...")
        
        vulnerabilities = []
        
        try:
            # Run cargo audit
            result = subprocess.run(
                ["cargo", "audit", "--json"],
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # Parse JSON output
                try:
                    audit_data = json.loads(result.stdout)
                    
                    for vuln in audit_data.get("vulnerabilities", {}).get("list", []):
                        advisory = vuln.get("advisory", {})
                        package = vuln.get("package", {})
                        
                        vulnerability = SecurityVulnerability(
                            dependency_name=package.get("name", "unknown"),
                            affected_versions=package.get("version", "unknown"),
                            vulnerability_id=advisory.get("id", "unknown"),
                            severity=advisory.get("severity", "unknown"),
                            description=advisory.get("title", "No description"),
                            patched_versions=advisory.get("patched_versions", [])
                        )
                        
                        vulnerabilities.append(vulnerability)
                        
                except json.JSONDecodeError:
                    logger.warning("Failed to parse cargo audit JSON output")
                    
        except subprocess.TimeoutExpired:
            logger.warning("cargo audit timed out")
        except FileNotFoundError:
            logger.warning("cargo audit not found - install with 'cargo install cargo-audit'")
        except Exception as e:
            logger.warning(f"Failed to run cargo audit: {e}")
        
        logger.info(f"Found {len(vulnerabilities)} security vulnerabilities")
        return vulnerabilities

    def detect_unused_dependencies(self) -> List[str]:
        """Detect unused dependencies using cargo machete"""
        logger.info("Detecting unused dependencies...")
        
        unused = []
        
        try:
            # Run cargo machete
            result = subprocess.run(
                ["cargo", "machete"],
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                # Parse output
                for line in result.stdout.split('\n'):
                    if line.strip() and not line.startswith('Analyzing'):
                        unused.append(line.strip())
                        
        except subprocess.TimeoutExpired:
            logger.warning("cargo machete timed out")
        except FileNotFoundError:
            logger.info("cargo machete not found - install with 'cargo install cargo-machete'")
        except Exception as e:
            logger.warning(f"Failed to run cargo machete: {e}")
        
        logger.info(f"Found {len(unused)} unused dependencies")
        return unused

    def check_outdated_dependencies(self) -> List[Dict[str, Any]]:
        """Check for outdated dependencies"""
        logger.info("Checking for outdated dependencies...")
        
        outdated = []
        
        for crate in self.crates:
            for dep in crate.dependencies:
                if dep.source == "crates.io" and not dep.workspace:
                    latest_version = self._get_latest_version(dep.name)
                    if latest_version and latest_version != dep.version:
                        try:
                            if version.parse(latest_version) > version.parse(dep.version.lstrip("^~=")):
                                outdated.append({
                                    "name": dep.name,
                                    "current": dep.version,
                                    "latest": latest_version,
                                    "crate": crate.name
                                })
                        except:
                            # Version parsing failed, still report as potentially outdated
                            outdated.append({
                                "name": dep.name,
                                "current": dep.version,
                                "latest": latest_version,
                                "crate": crate.name
                            })
        
        logger.info(f"Found {len(outdated)} outdated dependencies")
        return outdated

    def update_dependencies(self, dry_run: bool = True) -> Dict[str, Any]:
        """Update dependencies to latest compatible versions"""
        logger.info(f"{'Simulating' if dry_run else 'Performing'} dependency updates...")
        
        update_results = {
            "updated": [],
            "failed": [],
            "skipped": []
        }
        
        # Get outdated dependencies
        outdated = self.check_outdated_dependencies()
        
        for dep_info in outdated:
            dep_name = dep_info["name"]
            current_version = dep_info["current"]
            latest_version = dep_info["latest"]
            
            # Check update policy
            if not self._should_update(current_version, latest_version):
                update_results["skipped"].append({
                    "name": dep_name,
                    "reason": "Update policy restriction",
                    "current": current_version,
                    "available": latest_version
                })
                continue
            
            if dry_run:
                update_results["updated"].append({
                    "name": dep_name,
                    "from": current_version,
                    "to": latest_version,
                    "dry_run": True
                })
            else:
                # Perform actual update
                try:
                    success = self._update_dependency(dep_name, latest_version)
                    if success:
                        update_results["updated"].append({
                            "name": dep_name,
                            "from": current_version,
                            "to": latest_version
                        })
                    else:
                        update_results["failed"].append({
                            "name": dep_name,
                            "error": "Update failed"
                        })
                except Exception as e:
                    update_results["failed"].append({
                        "name": dep_name,
                        "error": str(e)
                    })
        
        logger.info(f"Update summary: {len(update_results['updated'])} updated, "
                   f"{len(update_results['failed'])} failed, {len(update_results['skipped'])} skipped")
        
        return update_results

    def _should_update(self, current: str, latest: str) -> bool:
        """Check if dependency should be updated based on policy"""
        try:
            current_ver = version.parse(current.lstrip("^~="))
            latest_ver = version.parse(latest)
            
            policy = self.config.get("update_policy", {})
            
            if latest_ver.major > current_ver.major:
                return policy.get("major_updates", False)
            elif latest_ver.minor > current_ver.minor:
                return policy.get("minor_updates", True)
            elif latest_ver.micro > current_ver.micro:
                return policy.get("patch_updates", True)
            else:
                return False
                
        except:
            return False

    def _update_dependency(self, dep_name: str, new_version: str) -> bool:
        """Update a specific dependency in relevant Cargo.toml files"""
        updated = False
        
        for crate in self.crates:
            cargo_toml = crate.cargo_toml_path
            
            try:
                import toml
                with open(cargo_toml, 'r') as f:
                    data = toml.load(f)
                
                # Update in different dependency sections
                sections = ["dependencies", "dev-dependencies", "build-dependencies"]
                
                for section in sections:
                    if section in data and dep_name in data[section]:
                        if isinstance(data[section][dep_name], str):
                            data[section][dep_name] = f"^{new_version}"
                        elif isinstance(data[section][dep_name], dict):
                            data[section][dep_name]["version"] = f"^{new_version}"
                        
                        updated = True
                
                if updated:
                    with open(cargo_toml, 'w') as f:
                        toml.dump(data, f)
                        
            except Exception as e:
                logger.error(f"Failed to update {dep_name} in {cargo_toml}: {e}")
                return False
        
        return updated

    def generate_dependency_graph(self) -> Dict[str, Any]:
        """Generate a dependency graph for visualization"""
        logger.info("Generating dependency graph...")
        
        graph = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes for each crate
        for crate in self.crates:
            graph["nodes"].append({
                "id": crate.name,
                "label": crate.name,
                "type": "example" if crate.is_example else "crate",
                "path": str(crate.path)
            })
        
        # Add edges for dependencies
        for crate in self.crates:
            for dep in crate.dependencies:
                # Only add edges for internal dependencies
                dep_crate = next((c for c in self.crates if c.name == dep.name), None)
                if dep_crate:
                    graph["edges"].append({
                        "from": crate.name,
                        "to": dep.name,
                        "type": "dependency",
                        "version": dep.version
                    })
        
        return graph

    def generate_report(self) -> DependencyReport:
        """Generate comprehensive dependency report"""
        logger.info("Generating dependency report...")
        
        # Analyze all aspects
        version_conflicts = self.analyze_version_conflicts()
        security_vulns = self.check_security_vulnerabilities()
        unused_deps = self.detect_unused_dependencies()
        outdated_deps = self.check_outdated_dependencies()
        
        # Count unique dependencies
        all_deps = set()
        total_deps = 0
        
        for crate in self.crates:
            for dep_list in [crate.dependencies, crate.dev_dependencies, crate.build_dependencies]:
                for dep in dep_list:
                    all_deps.add(dep.name)
                    total_deps += 1
        
        # Generate recommendations
        recommendations = []
        
        if version_conflicts:
            recommendations.append(f"Resolve {len(version_conflicts)} version conflicts")
        
        if security_vulns:
            high_severity = [v for v in security_vulns if v.severity in ["high", "critical"]]
            if high_severity:
                recommendations.append(f"Address {len(high_severity)} high/critical security vulnerabilities")
        
        if unused_deps:
            recommendations.append(f"Remove {len(unused_deps)} unused dependencies")
        
        if outdated_deps:
            recommendations.append(f"Update {len(outdated_deps)} outdated dependencies")
        
        return DependencyReport(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            workspace_crates=len(self.crates),
            total_dependencies=total_deps,
            unique_dependencies=len(all_deps),
            version_conflicts=version_conflicts,
            security_vulnerabilities=security_vulns,
            unused_dependencies=unused_deps,
            outdated_dependencies=outdated_deps,
            license_issues=[],  # TODO: Implement license checking
            recommendations=recommendations
        )

    def save_report(self, report: DependencyReport, output_path: Path) -> None:
        """Save dependency report to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        logger.info(f"Dependency report saved to: {output_path}")

    def print_summary(self, report: DependencyReport) -> None:
        """Print dependency report summary"""
        print("\n" + "="*60)
        print("📦 VoiRS Dependency Analysis Report")
        print("="*60)
        print(f"📊 Workspace Crates: {report.workspace_crates}")
        print(f"📦 Total Dependencies: {report.total_dependencies}")
        print(f"🔗 Unique Dependencies: {report.unique_dependencies}")
        
        if report.version_conflicts:
            print(f"\n⚠️  Version Conflicts: {len(report.version_conflicts)}")
            for conflict in report.version_conflicts[:5]:  # Show top 5
                print(f"  • {conflict.dependency_name}: {len(conflict.conflicting_versions)} versions")
                for version_spec, sources in conflict.conflicting_versions:
                    print(f"    - {version_spec} (used by: {sources})")
                if conflict.recommended_version:
                    print(f"    💡 Recommended: {conflict.recommended_version}")
        
        if report.security_vulnerabilities:
            print(f"\n🔒 Security Vulnerabilities: {len(report.security_vulnerabilities)}")
            high_severity = [v for v in report.security_vulnerabilities if v.severity in ["high", "critical"]]
            if high_severity:
                print(f"  ❌ High/Critical: {len(high_severity)}")
            for vuln in report.security_vulnerabilities[:3]:  # Show top 3
                print(f"  • {vuln.dependency_name}: {vuln.vulnerability_id} ({vuln.severity})")
                print(f"    {vuln.description}")
        
        if report.unused_dependencies:
            print(f"\n🗑️  Unused Dependencies: {len(report.unused_dependencies)}")
            for unused in report.unused_dependencies[:5]:  # Show first 5
                print(f"  • {unused}")
        
        if report.outdated_dependencies:
            print(f"\n📅 Outdated Dependencies: {len(report.outdated_dependencies)}")
            for outdated in report.outdated_dependencies[:5]:  # Show first 5
                print(f"  • {outdated['name']}: {outdated['current']} → {outdated['latest']}")
        
        if report.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in report.recommendations:
                print(f"  • {rec}")

    def run_dependency_analysis(self, 
                              check_security: bool = True,
                              check_unused: bool = True,
                              update_deps: bool = False) -> DependencyReport:
        """Run comprehensive dependency analysis"""
        logger.info("Starting VoiRS dependency analysis...")
        
        start_time = time.time()
        
        try:
            # Discover crates
            self.discover_crates()
            
            # Generate report
            report = self.generate_report()
            
            # Update dependencies if requested
            if update_deps:
                update_results = self.update_dependencies(dry_run=False)
                logger.info(f"Updated {len(update_results['updated'])} dependencies")
            
            duration = time.time() - start_time
            logger.info(f"Dependency analysis completed in {duration:.1f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="VoiRS Dependency Management System")
    parser.add_argument("--workspace-dir", "-w", type=Path, default=Path(".."),
                       help="Path to workspace root")
    parser.add_argument("--examples-dir", "-e", type=Path, default=Path("."),
                       help="Path to examples directory")
    parser.add_argument("--config", "-c", type=Path,
                       help="Path to dependency config file")
    parser.add_argument("--report", "-r", type=Path,
                       help="Generate detailed dependency report at path")
    parser.add_argument("--update", action="store_true",
                       help="Update dependencies to latest compatible versions")
    parser.add_argument("--check-security", action="store_true",
                       help="Check for security vulnerabilities")
    parser.add_argument("--fix-conflicts", action="store_true",
                       help="Automatically fix version conflicts")
    parser.add_argument("--unused", action="store_true",
                       help="Detect and report unused dependencies")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize dependency manager
        manager = VoiRSDependencyManager(args.workspace_dir, args.examples_dir, args.config)
        
        # Run dependency analysis
        report = manager.run_dependency_analysis(
            check_security=args.check_security,
            check_unused=args.unused,
            update_deps=args.update
        )
        
        # Print summary
        manager.print_summary(report)
        
        # Save detailed report if requested
        if args.report:
            manager.save_report(report, args.report)
        
        # Return exit code based on issues found
        critical_issues = len([v for v in report.security_vulnerabilities if v.severity in ["high", "critical"]])
        if critical_issues > 0:
            logger.error("Critical security vulnerabilities found")
            return 2
        elif report.version_conflicts or report.security_vulnerabilities:
            logger.warning("Dependency issues found")
            return 1
        else:
            logger.info("Dependency analysis passed")
            return 0
            
    except Exception as e:
        logger.error(f"Dependency analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 3

if __name__ == "__main__":
    sys.exit(main())