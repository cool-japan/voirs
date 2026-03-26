#!/usr/bin/env python3
"""
VoiRS Version Management System
===============================

Comprehensive version management tool for VoiRS workspace and examples.
Manages version consistency, compatibility testing, and automated version updates
across the entire VoiRS ecosystem.

Features:
- Workspace version synchronization
- Semantic version management and validation
- Cross-version compatibility testing
- Automated version bumping with dependency updates
- Release preparation and validation
- Version conflict detection and resolution
- Multi-version example testing
- Compatibility matrix generation

Usage:
    python version_manager.py [OPTIONS]

Options:
    --workspace-dir PATH    Path to workspace root (default: ..)
    --config PATH          Path to version config file
    --report PATH          Generate detailed version report
    --bump TYPE            Bump version (major/minor/patch)
    --set-version VER      Set specific version across workspace
    --check-compatibility  Check version compatibility
    --update-examples      Update examples to match workspace versions
    --validate-release     Validate versions for release
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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Set, Any, Tuple
from packaging import version
import tempfile

# Bug 1 fix: Use tomllib (stdlib 3.11+) with tomli fallback instead of toml
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            tomllib = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class VersionInfo:
    """Version information for a crate"""
    name: str
    current_version: str
    path: Path
    cargo_toml_path: Path
    is_workspace_member: bool = False
    is_example: bool = False
    dependencies: Dict[str, str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = {}

@dataclass
class VersionConflict:
    """Represents a version mismatch"""
    crate_name: str
    expected_version: str
    actual_version: str
    source_path: str
    conflict_type: str  # 'workspace', 'dependency', 'example'

@dataclass
class CompatibilityIssue:
    """Represents a compatibility issue between versions"""
    issue_type: str
    description: str
    affected_crates: List[str]
    severity: str  # 'error', 'warning', 'info'
    suggestion: Optional[str] = None

@dataclass
class VersionReport:
    """Comprehensive version analysis report"""
    timestamp: str
    workspace_version: Optional[str]
    total_crates: int
    workspace_members: int
    examples: int
    version_conflicts: List[VersionConflict]
    compatibility_issues: List[CompatibilityIssue]
    outdated_examples: List[str]
    release_readiness: Dict[str, Any]
    recommendations: List[str]

class VoiRSVersionManager:
    """Comprehensive version manager for VoiRS workspace"""

    def __init__(self, workspace_dir: Path, config_path: Optional[Path] = None):
        self.workspace_dir = workspace_dir
        self.config_path = config_path
        self.config = self._load_config()
        self.reports_dir = workspace_dir / "version_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        self.workspace_toml = workspace_dir / "Cargo.toml"
        self.workspace_version: Optional[str] = None
        self.crates: List[VersionInfo] = []
        self.workspace_members: Set[str] = set()

    def _load_config(self) -> Dict[str, Any]:
        """Load version manager configuration"""
        default_config = {
            "versioning": {
                "strategy": "workspace",  # workspace, independent, mixed
                "pre_release_suffix": "",
                "build_metadata": ""
            },
            "compatibility": {
                "check_breaking_changes": True,
                "semver_strict": True,
                "allow_pre_release": False
            },
            "release": {
                "require_clean_git": True,
                "run_tests_before_release": True,
                "update_changelog": True,
                "create_git_tag": True
            },
            "examples": {
                "sync_with_workspace": True,
                "allow_independent_versions": False,
                "validate_compatibility": True
            },
            "validation": {
                "check_dependency_versions": True,
                "validate_example_compatibility": True,
                "check_feature_flags": True
            }
        }

        if self.config_path and self.config_path.exists():
            if tomllib is None:
                logger.warning("No TOML parser available (install tomli for Python <3.11), using default config")
            else:
                try:
                    with open(self.config_path, 'rb') as f:
                        user_config = tomllib.load(f)
                        # Merge configurations
                        for key, value in user_config.items():
                            if isinstance(value, dict) and key in default_config:
                                default_config[key].update(value)
                            else:
                                default_config[key] = value
                except Exception as e:
                    logger.warning(f"Failed to load config: {e}, using defaults")

        return default_config

    def discover_crates(self) -> None:
        """Discover all crates in the workspace"""
        logger.info("Discovering crates and versions...")
        
        # Parse workspace Cargo.toml
        if self.workspace_toml.exists():
            self._parse_workspace_toml()
        
        # Find all Cargo.toml files
        cargo_files = list(self.workspace_dir.glob("**/Cargo.toml"))
        
        for cargo_file in cargo_files:
            if "target" in cargo_file.parts:
                continue
                
            try:
                version_info = self._parse_cargo_version(cargo_file)
                if version_info:
                    self.crates.append(version_info)
                    logger.debug(f"Found crate: {version_info.name} v{version_info.current_version}")
            except Exception as e:
                logger.warning(f"Failed to parse {cargo_file}: {e}")
        
        logger.info(f"Discovered {len(self.crates)} crates")

    def _parse_workspace_toml(self) -> None:
        """Parse workspace Cargo.toml to extract version and members"""
        if tomllib is None:
            raise RuntimeError(
                "No TOML parser available. Install tomli (pip install tomli) "
                "or use Python 3.11+ which includes tomllib in the standard library."
            )
        try:
            with open(self.workspace_toml, 'rb') as f:
                data = tomllib.load(f)

            # Extract workspace version if present
            package = data.get("package", {})
            self.workspace_version = package.get("version")

            # Also check workspace.package for workspace-level version
            workspace = data.get("workspace", {})
            ws_package = workspace.get("package", {})
            if not self.workspace_version and ws_package.get("version"):
                self.workspace_version = ws_package.get("version")

            # Extract workspace members
            members = workspace.get("members", [])

            for member in members:
                # Handle glob patterns
                if "*" in member:
                    member_paths = list(self.workspace_dir.glob(member))
                    for path in member_paths:
                        if path.is_dir():
                            self.workspace_members.add(path.name)
                else:
                    self.workspace_members.add(Path(member).name)

        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(f"Failed to parse workspace Cargo.toml: {e}")

    def _parse_cargo_version(self, cargo_file: Path) -> Optional[VersionInfo]:
        """Parse a Cargo.toml file and extract version information"""
        try:
            if tomllib is None:
                logger.warning("No TOML parser available, skipping %s", cargo_file)
                return None
            with open(cargo_file, 'rb') as f:
                data = tomllib.load(f)
            
            package = data.get("package", {})
            if not package:
                return None
            
            crate_name = package.get("name", cargo_file.parent.name)
            version_spec = package.get("version", "0.0.0")
            
            # Handle workspace versions
            if isinstance(version_spec, dict) and version_spec.get("workspace"):
                current_version = self.workspace_version or "0.0.0"
            else:
                current_version = str(version_spec)
            
            # Check if this is a workspace member
            is_workspace_member = crate_name in self.workspace_members or cargo_file.parent.name in self.workspace_members
            
            # Check if this is an example
            is_example = "examples" in str(cargo_file) or "example" in package.get("keywords", [])
            
            # Extract VoiRS dependencies
            dependencies = {}
            deps_section = data.get("dependencies", {})
            
            for dep_name, dep_spec in deps_section.items():
                if dep_name.startswith("voirs"):
                    if isinstance(dep_spec, str):
                        dependencies[dep_name] = dep_spec
                    elif isinstance(dep_spec, dict):
                        dependencies[dep_name] = dep_spec.get("version", "")
            
            return VersionInfo(
                name=crate_name,
                current_version=current_version,
                path=cargo_file.parent,
                cargo_toml_path=cargo_file,
                is_workspace_member=is_workspace_member,
                is_example=is_example,
                dependencies=dependencies
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse {cargo_file}: {e}")
            return None

    def analyze_version_conflicts(self) -> List[VersionConflict]:
        """Analyze version conflicts across the workspace"""
        logger.info("Analyzing version conflicts...")
        
        conflicts = []
        
        # Check workspace version consistency
        if self.workspace_version:
            for crate in self.crates:
                if crate.is_workspace_member and crate.current_version != self.workspace_version:
                    conflicts.append(VersionConflict(
                        crate_name=crate.name,
                        expected_version=self.workspace_version,
                        actual_version=crate.current_version,
                        source_path=str(crate.path),
                        conflict_type="workspace"
                    ))
        
        # Check dependency version consistency
        for crate in self.crates:
            for dep_name, dep_version in crate.dependencies.items():
                # Find the actual version of this dependency in workspace
                dep_crate = next((c for c in self.crates if c.name == dep_name), None)
                if dep_crate:
                    # Clean version strings for comparison
                    clean_dep_version = self._clean_version(dep_version)
                    clean_actual_version = self._clean_version(dep_crate.current_version)
                    
                    if clean_dep_version and clean_actual_version and clean_dep_version != clean_actual_version:
                        conflicts.append(VersionConflict(
                            crate_name=f"{crate.name} -> {dep_name}",
                            expected_version=clean_actual_version,
                            actual_version=clean_dep_version,
                            source_path=str(crate.path),
                            conflict_type="dependency"
                        ))
        
        logger.info(f"Found {len(conflicts)} version conflicts")
        return conflicts

    def _clean_version(self, version_str: Any) -> str:
        """Clean version string for comparison"""
        if not version_str:
            return ""
        
        # Handle dictionary versions (workspace = true)
        if isinstance(version_str, dict):
            if version_str.get("workspace"):
                return self.workspace_version or ""
            return version_str.get("version", "")
        
        # Convert to string if not already
        version_str = str(version_str)
        
        # Remove version prefixes (^, ~, =, etc.)
        cleaned = re.sub(r'^[^\d]*', '', version_str)
        
        # Extract just the version number
        match = re.match(r'(\d+\.\d+\.\d+)', cleaned)
        if match:
            return match.group(1)
        
        return cleaned

    def check_compatibility(self) -> List[CompatibilityIssue]:
        """Check compatibility issues between versions"""
        logger.info("Checking version compatibility...")
        
        issues = []
        
        # Check for breaking changes between versions
        for crate in self.crates:
            if crate.is_example:
                for dep_name, dep_version in crate.dependencies.items():
                    if dep_name.startswith("voirs"):
                        dep_crate = next((c for c in self.crates if c.name == dep_name), None)
                        if dep_crate:
                            compatibility_issue = self._check_version_compatibility(
                                dep_name, dep_version, dep_crate.current_version
                            )
                            if compatibility_issue:
                                issues.append(compatibility_issue)
        
        # Check for pre-release versions in stable dependencies
        if not self.config["compatibility"]["allow_pre_release"]:
            for crate in self.crates:
                if self._is_pre_release(crate.current_version):
                    issues.append(CompatibilityIssue(
                        issue_type="pre_release",
                        description=f"Crate {crate.name} uses pre-release version {crate.current_version}",
                        affected_crates=[crate.name],
                        severity="warning",
                        suggestion="Consider using stable version for production"
                    ))
        
        logger.info(f"Found {len(issues)} compatibility issues")
        return issues

    def _check_version_compatibility(self, dep_name: str, required_version: str, actual_version: str) -> Optional[CompatibilityIssue]:
        """Check if two versions are compatible"""
        try:
            req_ver = version.parse(self._clean_version(required_version))
            act_ver = version.parse(self._clean_version(actual_version))
            
            # Check for major version differences (breaking changes)
            if req_ver.major != act_ver.major:
                return CompatibilityIssue(
                    issue_type="major_version_mismatch",
                    description=f"Major version mismatch for {dep_name}: required {required_version}, actual {actual_version}",
                    affected_crates=[dep_name],
                    severity="error",
                    suggestion="Update dependency to match major version"
                )
            
            # Check for minor version compatibility
            if self.config["compatibility"]["semver_strict"] and req_ver.minor > act_ver.minor:
                return CompatibilityIssue(
                    issue_type="minor_version_downgrade",
                    description=f"Minor version downgrade for {dep_name}: required {required_version}, actual {actual_version}",
                    affected_crates=[dep_name],
                    severity="warning",
                    suggestion="Consider updating to newer minor version"
                )
                
        except Exception as e:
            logger.debug(f"Failed to parse versions for {dep_name}: {e}")
        
        return None

    def _is_pre_release(self, version_str: str) -> bool:
        """Check if version is a pre-release"""
        try:
            ver = version.parse(self._clean_version(version_str))
            return ver.is_prerelease
        except Exception:
            return False

    def bump_version(self, bump_type: str, target_crate: Optional[str] = None, dry_run: bool = False) -> Dict[str, str]:
        """Bump version across workspace or specific crate"""
        logger.info(f"Bumping {bump_type} version...{' (dry-run)' if dry_run else ''}")

        if bump_type not in ["major", "minor", "patch"]:
            raise ValueError("Bump type must be 'major', 'minor', or 'patch'")

        updated_versions = {}

        if target_crate:
            # Bump specific crate
            crate = next((c for c in self.crates if c.name == target_crate), None)
            if not crate:
                raise ValueError(f"Crate {target_crate} not found")

            new_version = self._calculate_new_version(crate.current_version, bump_type)
            success = self._update_crate_version(crate, new_version, dry_run=dry_run)
            if success:
                updated_versions[target_crate] = new_version
        else:
            # Bump workspace version
            if self.workspace_version:
                new_workspace_version = self._calculate_new_version(self.workspace_version, bump_type)

                # Update workspace Cargo.toml
                if self._update_workspace_version(new_workspace_version, dry_run=dry_run):
                    updated_versions["workspace"] = new_workspace_version

                # Update all workspace members
                for crate in self.crates:
                    if crate.is_workspace_member:
                        success = self._update_crate_version(crate, new_workspace_version, dry_run=dry_run)
                        if success:
                            updated_versions[crate.name] = new_workspace_version

        logger.info(f"Updated {len(updated_versions)} versions")
        return updated_versions

    def set_version(self, target_version: str, target_crate: Optional[str] = None, dry_run: bool = False) -> Dict[str, str]:
        """Set an explicit version across workspace or specific crate"""
        # Validate semver format
        try:
            parsed = version.parse(target_version)
            # Ensure it looks like a real semver (not just a number)
            if not re.match(r'^\d+\.\d+\.\d+', target_version):
                raise ValueError(f"Version must be in semver format (e.g., 1.2.3), got: {target_version}")
        except Exception as e:
            raise ValueError(f"Invalid version '{target_version}': {e}")

        logger.info(f"Setting version to {target_version}...{' (dry-run)' if dry_run else ''}")

        updated_versions = {}

        if target_crate:
            crate = next((c for c in self.crates if c.name == target_crate), None)
            if not crate:
                raise ValueError(f"Crate {target_crate} not found")

            success = self._update_crate_version(crate, target_version, dry_run=dry_run)
            if success:
                updated_versions[target_crate] = target_version
        else:
            # Set workspace version
            if self._update_workspace_version(target_version, dry_run=dry_run):
                updated_versions["workspace"] = target_version

            # Update all workspace members
            for crate in self.crates:
                if crate.is_workspace_member:
                    success = self._update_crate_version(crate, target_version, dry_run=dry_run)
                    if success:
                        updated_versions[crate.name] = target_version

        logger.info(f"Set version for {len(updated_versions)} crates")
        return updated_versions

    def _calculate_new_version(self, current_version: str, bump_type: str) -> str:
        """Calculate new version based on bump type"""
        try:
            ver = version.parse(self._clean_version(current_version))
            
            if bump_type == "major":
                new_ver = f"{ver.major + 1}.0.0"
            elif bump_type == "minor":
                new_ver = f"{ver.major}.{ver.minor + 1}.0"
            elif bump_type == "patch":
                new_ver = f"{ver.major}.{ver.minor}.{ver.micro + 1}"
            else:
                raise ValueError(f"Invalid bump type: {bump_type}")
            
            # Add pre-release suffix if configured
            pre_release = self.config["versioning"]["pre_release_suffix"]
            if pre_release:
                new_ver = f"{new_ver}-{pre_release}"
            
            # Add build metadata if configured
            build_metadata = self.config["versioning"]["build_metadata"]
            if build_metadata:
                new_ver = f"{new_ver}+{build_metadata}"
            
            return new_ver
            
        except Exception as e:
            logger.error(f"Failed to calculate new version: {e}")
            raise

    def _update_crate_version(self, crate: VersionInfo, new_version: str, dry_run: bool = False) -> bool:
        """Update version in a crate's Cargo.toml"""
        try:
            with open(crate.cargo_toml_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Replace version in [package] section using regex
            new_content = re.sub(
                r'(^\s*version\s*=\s*)"[^"]*"',
                rf'\g<1>"{new_version}"',
                content,
                count=1,
                flags=re.MULTILINE
            )

            if new_content == content:
                logger.debug(f"No version field found in {crate.cargo_toml_path}")
                return False

            if dry_run:
                logger.info(f"[DRY-RUN] Would update {crate.name} to version {new_version}")
                return True

            with open(crate.cargo_toml_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            logger.info(f"Updated {crate.name} to version {new_version}")
            return True

        except Exception as e:
            logger.error(f"Failed to update version for {crate.name}: {e}")

        return False

    def _update_workspace_version(self, new_version: str, dry_run: bool = False) -> bool:
        """Update workspace version"""
        try:
            with open(self.workspace_toml, 'r', encoding='utf-8') as f:
                content = f.read()

            # Replace version in [package] or [workspace.package] section
            new_content = re.sub(
                r'(^\s*version\s*=\s*)"[^"]*"',
                rf'\g<1>"{new_version}"',
                content,
                count=1,
                flags=re.MULTILINE
            )

            if new_content == content:
                logger.debug("No version field found in workspace Cargo.toml")
                return False

            if dry_run:
                logger.info(f"[DRY-RUN] Would update workspace to version {new_version}")
                return True

            with open(self.workspace_toml, 'w', encoding='utf-8') as f:
                f.write(new_content)

            logger.info(f"Updated workspace to version {new_version}")
            return True

        except Exception as e:
            logger.error(f"Failed to update workspace version: {e}")

        return False

    def update_examples_to_workspace_versions(self, dry_run: bool = False) -> int:
        """Update example dependencies to match workspace versions"""
        logger.info("Updating examples to match workspace versions...")

        updated_count = 0

        for crate in self.crates:
            if crate.is_example:
                try:
                    with open(crate.cargo_toml_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    new_content = content
                    changed = False

                    for dep_name in crate.dependencies:
                        if dep_name.startswith("voirs"):
                            dep_crate = next((c for c in self.crates if c.name == dep_name), None)
                            if dep_crate:
                                target_version = dep_crate.current_version
                                # Replace version strings like "^0.2.0" or "0.2.0" for this dep
                                pattern = rf'({re.escape(dep_name)}\s*=\s*(?:{{[^}}]*version\s*=\s*)?")[^"]*(")'
                                replacement = rf'\g<1>^{target_version}\2'
                                replaced = re.sub(pattern, replacement, new_content)
                                if replaced != new_content:
                                    new_content = replaced
                                    changed = True

                    if changed:
                        if dry_run:
                            logger.info(f"[DRY-RUN] Would update example {crate.name} dependencies")
                        else:
                            with open(crate.cargo_toml_path, 'w', encoding='utf-8') as f:
                                f.write(new_content)
                            logger.info(f"Updated example {crate.name} dependencies")
                        updated_count += 1

                except Exception as e:
                    logger.error(f"Failed to update example {crate.name}: {e}")

        logger.info(f"Updated {updated_count} examples")
        return updated_count

    def validate_release_readiness(self, run_tests: bool = False) -> Dict[str, Any]:
        """Validate if the workspace is ready for release.

        Args:
            run_tests: If True, run cargo test. Should only be True when
                       --validate-release is explicitly passed.
        """
        logger.info("Validating release readiness...")
        
        validation_results = {
            "ready": True,
            "issues": [],
            "warnings": [],
            "checks": {
                "version_consistency": False,
                "clean_git": False,
                "tests_pass": False,
                "no_pre_release": False
            }
        }
        
        # Check version consistency
        conflicts = self.analyze_version_conflicts()
        if conflicts:
            validation_results["ready"] = False
            validation_results["issues"].append(f"Found {len(conflicts)} version conflicts")
        else:
            validation_results["checks"]["version_consistency"] = True
        
        # Check git status
        if self.config["release"]["require_clean_git"]:
            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=self.workspace_dir,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0 and not result.stdout.strip():
                    validation_results["checks"]["clean_git"] = True
                else:
                    validation_results["ready"] = False
                    validation_results["issues"].append("Git repository has uncommitted changes")
                    
            except Exception as e:
                validation_results["warnings"].append(f"Failed to check git status: {e}")
        
        # Check for pre-release versions
        pre_release_crates = [c.name for c in self.crates if self._is_pre_release(c.current_version)]
        if pre_release_crates:
            validation_results["warnings"].append(f"Pre-release versions found: {', '.join(pre_release_crates)}")
        else:
            validation_results["checks"]["no_pre_release"] = True
        
        # Run tests only when explicitly requested (opt-in)
        if run_tests and self.config["release"]["run_tests_before_release"]:
            try:
                result = subprocess.run(
                    ["cargo", "test", "--workspace"],
                    cwd=self.workspace_dir,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    validation_results["checks"]["tests_pass"] = True
                else:
                    validation_results["ready"] = False
                    validation_results["issues"].append("Tests are failing")
                    
            except subprocess.TimeoutExpired:
                validation_results["ready"] = False
                validation_results["issues"].append("Tests timed out")
            except Exception as e:
                validation_results["warnings"].append(f"Failed to run tests: {e}")
        
        return validation_results

    def generate_version_report(self) -> VersionReport:
        """Generate comprehensive version report"""
        logger.info("Generating version report...")
        
        conflicts = self.analyze_version_conflicts()
        compatibility_issues = self.check_compatibility()
        release_readiness = self.validate_release_readiness(run_tests=False)
        
        # Find outdated examples
        outdated_examples = []
        for crate in self.crates:
            if crate.is_example:
                for dep_name, dep_version in crate.dependencies.items():
                    if dep_name.startswith("voirs"):
                        dep_crate = next((c for c in self.crates if c.name == dep_name), None)
                        if dep_crate:
                            clean_dep_version = self._clean_version(dep_version)
                            clean_actual_version = self._clean_version(dep_crate.current_version)
                            
                            if clean_dep_version and clean_actual_version and clean_dep_version != clean_actual_version:
                                outdated_examples.append(crate.name)
                                break
        
        # Generate recommendations
        recommendations = []
        
        if conflicts:
            recommendations.append(f"Resolve {len(conflicts)} version conflicts")
        
        if compatibility_issues:
            error_issues = [i for i in compatibility_issues if i.severity == "error"]
            if error_issues:
                recommendations.append(f"Fix {len(error_issues)} critical compatibility issues")
        
        if outdated_examples:
            recommendations.append(f"Update {len(outdated_examples)} outdated examples")
        
        if not release_readiness["ready"]:
            recommendations.append("Address release readiness issues before releasing")
        
        return VersionReport(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            workspace_version=self.workspace_version,
            total_crates=len(self.crates),
            workspace_members=len([c for c in self.crates if c.is_workspace_member]),
            examples=len([c for c in self.crates if c.is_example]),
            version_conflicts=conflicts,
            compatibility_issues=compatibility_issues,
            outdated_examples=outdated_examples,
            release_readiness=release_readiness,
            recommendations=recommendations
        )

    def save_report(self, report: VersionReport, output_path: Path) -> None:
        """Save version report to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        logger.info(f"Version report saved to: {output_path}")

    def print_summary(self, report: VersionReport) -> None:
        """Print version report summary"""
        print("\n" + "="*60)
        print("🏷️  VoiRS Version Management Report")
        print("="*60)
        print(f"📦 Workspace Version: {report.workspace_version or 'Not set'}")
        print(f"📊 Total Crates: {report.total_crates}")
        print(f"🏗️  Workspace Members: {report.workspace_members}")
        print(f"📋 Examples: {report.examples}")
        
        if report.version_conflicts:
            print(f"\n⚠️  Version Conflicts: {len(report.version_conflicts)}")
            for conflict in report.version_conflicts[:5]:  # Show top 5
                print(f"  • {conflict.crate_name}")
                print(f"    Expected: {conflict.expected_version}, Actual: {conflict.actual_version}")
                print(f"    Type: {conflict.conflict_type}, Path: {conflict.source_path}")
        
        if report.compatibility_issues:
            print(f"\n🔧 Compatibility Issues: {len(report.compatibility_issues)}")
            for issue in report.compatibility_issues[:3]:  # Show top 3
                severity_emoji = {"error": "❌", "warning": "⚠️ ", "info": "ℹ️ "}
                print(f"  {severity_emoji.get(issue.severity, '•')} {issue.description}")
                if issue.suggestion:
                    print(f"    💡 {issue.suggestion}")
        
        if report.outdated_examples:
            print(f"\n📅 Outdated Examples: {len(report.outdated_examples)}")
            for example in report.outdated_examples[:5]:  # Show first 5
                print(f"  • {example}")
        
        # Release readiness
        readiness = report.release_readiness
        status_emoji = "✅" if readiness["ready"] else "❌"
        print(f"\n🚀 Release Readiness: {status_emoji} {'Ready' if readiness['ready'] else 'Not Ready'}")
        
        checks = readiness["checks"]
        for check_name, passed in checks.items():
            check_emoji = "✅" if passed else "❌"
            check_label = check_name.replace("_", " ").title()
            print(f"  {check_emoji} {check_label}")
        
        if readiness["issues"]:
            print("  Issues:")
            for issue in readiness["issues"]:
                print(f"    • {issue}")
        
        if readiness["warnings"]:
            print("  Warnings:")
            for warning in readiness["warnings"]:
                print(f"    • {warning}")
        
        if report.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in report.recommendations:
                print(f"  • {rec}")

    def run_version_management(self,
                              bump_type: Optional[str] = None,
                              set_version_str: Optional[str] = None,
                              update_examples: bool = False,
                              dry_run: bool = False,
                              validate_release: bool = False) -> VersionReport:
        """Run comprehensive version management"""
        logger.info("Starting VoiRS version management...")

        start_time = time.time()

        try:
            # Discover crates
            self.discover_crates()

            # Perform version operations
            if bump_type:
                self.bump_version(bump_type, dry_run=dry_run)

            if set_version_str:
                self.set_version(set_version_str, dry_run=dry_run)

            if update_examples:
                self.update_examples_to_workspace_versions(dry_run=dry_run)

            # Generate report (does not run tests by default)
            report = self.generate_version_report()

            # If --validate-release was explicitly passed, run full validation with tests
            if validate_release:
                logger.info("Running full release validation (with tests)...")
                report.release_readiness = self.validate_release_readiness(run_tests=True)

            duration = time.time() - start_time
            logger.info(f"Version management completed in {duration:.1f}s")

            return report

        except Exception as e:
            logger.error(f"Version management failed: {e}")
            raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="VoiRS Version Management System")
    parser.add_argument("--workspace-dir", "-w", type=Path, default=Path("."),
                       help="Path to workspace root (default: current directory)")
    parser.add_argument("--config", "-c", type=Path,
                       help="Path to version config file")
    parser.add_argument("--report", "-r", type=Path,
                       help="Generate detailed version report at path")
    parser.add_argument("--bump", choices=["major", "minor", "patch"],
                       help="Bump version (major/minor/patch)")
    parser.add_argument("--set-version", type=str,
                       help="Set specific version across workspace")
    parser.add_argument("--check-compatibility", action="store_true",
                       help="Check version compatibility")
    parser.add_argument("--update-examples", action="store_true",
                       help="Update examples to match workspace versions")
    parser.add_argument("--validate-release", action="store_true",
                       help="Validate versions for release (runs cargo test)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize version manager
        manager = VoiRSVersionManager(args.workspace_dir, args.config)

        # Run version management
        report = manager.run_version_management(
            bump_type=args.bump,
            set_version_str=args.set_version,
            update_examples=args.update_examples,
            dry_run=args.dry_run,
            validate_release=args.validate_release,
        )

        # Print summary
        manager.print_summary(report)

        # Save detailed report if requested
        if args.report:
            manager.save_report(report, args.report)

        # Consistent exit codes: 0=success, 1=error, 2=argparse error (handled by argparse)
        if report.version_conflicts or any(i.severity == "error" for i in report.compatibility_issues):
            logger.info("Version issues found")
            return 1
        else:
            logger.info("Version management completed successfully")
            return 0

    except Exception as e:
        logger.error(f"Version management failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())