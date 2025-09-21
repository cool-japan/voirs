#!/usr/bin/env python3
"""
VoiRS Version Management Tool

Handles multiple VoiRS versions, compatibility checking, and migration assistance.
"""

import json
import toml
import subprocess
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import argparse
from datetime import datetime
import semver


class CompatibilityLevel(Enum):
    COMPATIBLE = "compatible"
    DEPRECATED = "deprecated"
    BREAKING = "breaking"
    UNKNOWN = "unknown"


@dataclass
class VersionInfo:
    """Information about a VoiRS version."""
    version: str
    release_date: str
    major_features: List[str]
    breaking_changes: List[str]
    deprecated_features: List[str]
    compatibility_level: CompatibilityLevel
    examples_compatible: List[str]
    migration_notes: str


@dataclass
class ExampleCompatibility:
    """Compatibility information for an example."""
    example_name: str
    current_version: str
    compatible_versions: List[str]
    required_features: List[str]
    breaking_in_versions: List[str]
    migration_required: bool
    migration_guide: str


class VersionManager:
    """Manages VoiRS versions and compatibility."""
    
    def __init__(self, examples_dir: Path):
        self.examples_dir = examples_dir
        self.versions_file = examples_dir / "tools" / "versions.json"
        self.versions: Dict[str, VersionInfo] = {}
        self.examples: List[Path] = []
        self._load_version_data()
        
    def _load_version_data(self) -> None:
        """Load version data from configuration file."""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                data = json.load(f)
                for version, info in data.items():
                    self.versions[version] = VersionInfo(**info)
        else:
            # Initialize with known VoiRS versions
            self._initialize_default_versions()
            
    def _initialize_default_versions(self) -> None:
        """Initialize with default VoiRS version information."""
        default_versions = {
            "0.1.0": VersionInfo(
                version="0.1.0",
                release_date="2024-01-01",
                major_features=["Basic TTS", "Voice Cloning", "Emotion Control"],
                breaking_changes=[],
                deprecated_features=[],
                compatibility_level=CompatibilityLevel.COMPATIBLE,
                examples_compatible=[],
                migration_notes=""
            ),
            "0.2.0": VersionInfo(
                version="0.2.0",
                release_date="2024-06-01",
                major_features=["Streaming Synthesis", "GPU Acceleration", "SSML Support"],
                breaking_changes=["Config API restructure"],
                deprecated_features=["Old config format"],
                compatibility_level=CompatibilityLevel.BREAKING,
                examples_compatible=[],
                migration_notes="Update configuration objects to use new builder pattern"
            ),
            "0.3.0": VersionInfo(
                version="0.3.0",
                release_date="2024-12-01",
                major_features=["Multi-modal Synthesis", "Advanced Emotions", "Real-time Processing"],
                breaking_changes=["Emotion API changes"],
                deprecated_features=["Simple emotion enum"],
                compatibility_level=CompatibilityLevel.BREAKING,
                examples_compatible=[],
                migration_notes="Migrate from EmotionType enum to EmotionVector struct"
            )
        }
        self.versions = default_versions
        self._save_version_data()
        
    def _save_version_data(self) -> None:
        """Save version data to configuration file."""
        self.versions_file.parent.mkdir(exist_ok=True)
        
        data = {}
        for version, info in self.versions.items():
            data[version] = {
                "version": info.version,
                "release_date": info.release_date,
                "major_features": info.major_features,
                "breaking_changes": info.breaking_changes,
                "deprecated_features": info.deprecated_features,
                "compatibility_level": info.compatibility_level.value,
                "examples_compatible": info.examples_compatible,
                "migration_notes": info.migration_notes
            }
            
        with open(self.versions_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def discover_examples(self) -> None:
        """Discover all example files."""
        self.examples = list(self.examples_dir.glob("*.rs"))
        print(f"📁 Discovered {len(self.examples)} examples")
        
    def get_current_version(self) -> str:
        """Get the current VoiRS version from Cargo.toml."""
        cargo_path = self.examples_dir / "Cargo.toml"
        if not cargo_path.exists():
            return "unknown"
            
        with open(cargo_path, 'r') as f:
            cargo_data = toml.load(f)
            
        # Look for VoiRS dependencies
        deps = cargo_data.get('dependencies', {})
        for dep_name, dep_info in deps.items():
            if dep_name.startswith('voirs-'):
                if isinstance(dep_info, dict):
                    return dep_info.get('version', 'unknown')
                elif isinstance(dep_info, str):
                    return dep_info
                    
        return "unknown"
        
    def analyze_example_compatibility(self, example_path: Path) -> ExampleCompatibility:
        """Analyze compatibility of a specific example."""
        example_name = example_path.stem
        
        try:
            with open(example_path, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️  Could not read {example_path}: {e}")
            return ExampleCompatibility(
                example_name=example_name,
                current_version="unknown",
                compatible_versions=[],
                required_features=[],
                breaking_in_versions=[],
                migration_required=False,
                migration_guide=""
            )
            
        # Analyze code patterns to determine compatibility
        compatible_versions = []
        breaking_in_versions = []
        required_features = []
        migration_required = False
        migration_guide = ""
        
        # Check for version-specific patterns
        if "EmotionType::" in content:
            # Old emotion API - breaking in 0.3.0
            breaking_in_versions.append("0.3.0")
            migration_required = True
            migration_guide += "Update EmotionType enum usage to EmotionVector struct. "
            
        if "SynthesisConfig::new(" in content:
            # Old config API - breaking in 0.2.0
            breaking_in_versions.append("0.2.0")
            migration_required = True
            migration_guide += "Use SynthesisConfig::builder() instead of new(). "
            
        if "streaming" in content.lower():
            required_features.append("streaming")
            # Streaming available from 0.2.0+
            compatible_versions.extend(["0.2.0", "0.3.0"])
        else:
            # Basic synthesis available in all versions
            compatible_versions.extend(["0.1.0", "0.2.0", "0.3.0"])
            
        if "gpu" in content.lower() or "cuda" in content.lower():
            required_features.append("gpu-acceleration")
            # GPU acceleration available from 0.2.0+
            compatible_versions = [v for v in compatible_versions if semver.compare(v, "0.2.0") >= 0]
            
        return ExampleCompatibility(
            example_name=example_name,
            current_version=self.get_current_version(),
            compatible_versions=list(set(compatible_versions)),
            required_features=required_features,
            breaking_in_versions=list(set(breaking_in_versions)),
            migration_required=migration_required,
            migration_guide=migration_guide.strip()
        )
        
    def check_version_compatibility(self, target_version: str) -> Dict[str, ExampleCompatibility]:
        """Check compatibility of all examples with a target version."""
        print(f"🔍 Checking compatibility with VoiRS {target_version}...")
        
        compatibility_report = {}
        
        for example_path in self.examples:
            compat = self.analyze_example_compatibility(example_path)
            
            # Check specific compatibility with target version
            if target_version in compat.breaking_in_versions:
                compat.migration_required = True
            elif target_version not in compat.compatible_versions:
                # Unknown compatibility
                print(f"⚠️  {compat.example_name}: Unknown compatibility with {target_version}")
                
            compatibility_report[compat.example_name] = compat
            
        return compatibility_report
        
    def generate_migration_plan(self, from_version: str, to_version: str) -> Dict:
        """Generate a migration plan between versions."""
        print(f"📋 Generating migration plan: {from_version} → {to_version}")
        
        migration_steps = []
        affected_examples = []
        
        # Check what changed between versions
        if from_version in self.versions and to_version in self.versions:
            from_info = self.versions[from_version]
            to_info = self.versions[to_version]
            
            if to_info.breaking_changes:
                migration_steps.append({
                    "type": "breaking_changes",
                    "description": "Handle breaking changes",
                    "changes": to_info.breaking_changes,
                    "action": "required"
                })
                
            if to_info.deprecated_features:
                migration_steps.append({
                    "type": "deprecations",
                    "description": "Update deprecated features",
                    "features": to_info.deprecated_features,
                    "action": "recommended"
                })
                
            if to_info.major_features:
                migration_steps.append({
                    "type": "new_features",
                    "description": "Consider adopting new features",
                    "features": to_info.major_features,
                    "action": "optional"
                })
                
        # Find affected examples
        compatibility_report = self.check_version_compatibility(to_version)
        for example_name, compat in compatibility_report.items():
            if compat.migration_required:
                affected_examples.append({
                    "example": example_name,
                    "issues": compat.breaking_in_versions,
                    "guide": compat.migration_guide
                })
                
        return {
            "from_version": from_version,
            "to_version": to_version,
            "migration_steps": migration_steps,
            "affected_examples": affected_examples,
            "estimated_effort": self._estimate_migration_effort(migration_steps, affected_examples)
        }
        
    def _estimate_migration_effort(self, steps: List[Dict], examples: List[Dict]) -> str:
        """Estimate the effort required for migration."""
        breaking_changes = sum(1 for step in steps if step.get("action") == "required")
        affected_count = len(examples)
        
        if breaking_changes == 0 and affected_count == 0:
            return "minimal"
        elif breaking_changes <= 2 and affected_count <= 5:
            return "low"
        elif breaking_changes <= 5 and affected_count <= 15:
            return "medium"
        else:
            return "high"
            
    def update_version_tags(self) -> None:
        """Update version compatibility tags in example files."""
        print("🏷️  Updating version tags in examples...")
        
        for example_path in self.examples:
            compat = self.analyze_example_compatibility(example_path)
            
            try:
                with open(example_path, 'r') as f:
                    content = f.read()
                    
                # Look for existing version tag
                version_pattern = r'//\s*@voirs-version:\s*([^\n]+)'
                new_tag = f"// @voirs-version: {', '.join(compat.compatible_versions)}"
                
                if "// @voirs-version:" in content:
                    # Update existing tag
                    content = re.sub(version_pattern, new_tag, content)
                else:
                    # Add new tag at the top
                    lines = content.split('\n')
                    if lines and lines[0].startswith('//'):
                        # Insert after existing header comments
                        insert_idx = 0
                        for i, line in enumerate(lines):
                            if not line.startswith('//') and line.strip():
                                insert_idx = i
                                break
                        lines.insert(insert_idx, new_tag)
                    else:
                        lines.insert(0, new_tag)
                    content = '\n'.join(lines)
                    
                with open(example_path, 'w') as f:
                    f.write(content)
                    
            except Exception as e:
                print(f"⚠️  Could not update {example_path}: {e}")
                
    def print_compatibility_summary(self, compatibility_report: Dict[str, ExampleCompatibility]) -> None:
        """Print a summary of compatibility analysis."""
        print("\n" + "="*60)
        print("🎯 VOIRS VERSION COMPATIBILITY SUMMARY")
        print("="*60)
        
        total_examples = len(compatibility_report)
        migration_needed = sum(1 for c in compatibility_report.values() if c.migration_required)
        
        print(f"📁 Total Examples: {total_examples}")
        print(f"⚠️  Migration Required: {migration_needed}")
        print(f"✅ Compatible: {total_examples - migration_needed}")
        
        if migration_needed > 0:
            print(f"\n📋 EXAMPLES REQUIRING MIGRATION:")
            for name, compat in compatibility_report.items():
                if compat.migration_required:
                    print(f"  • {name}: {compat.migration_guide}")
                    
    def save_compatibility_report(self, report: Dict[str, ExampleCompatibility], output_path: Path) -> None:
        """Save compatibility report to file."""
        report_data = {
            'compatibility_report': {
                name: asdict(compat) for name, compat in report.items()
            },
            'generated_at': datetime.now().isoformat(),
            'current_version': self.get_current_version(),
            'total_examples': len(report),
            'migration_needed': sum(1 for c in report.values() if c.migration_required)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        print(f"📄 Compatibility report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VoiRS Version Manager")
    parser.add_argument("--examples-dir", type=Path, default=Path.cwd(),
                       help="Path to examples directory")
    parser.add_argument("--target-version", type=str,
                       help="Target VoiRS version to check compatibility")
    parser.add_argument("--migration-plan", type=str, nargs=2, metavar=("FROM", "TO"),
                       help="Generate migration plan between versions")
    parser.add_argument("--update-tags", action="store_true",
                       help="Update version compatibility tags in examples")
    parser.add_argument("--output", type=Path, default=Path("compatibility_report.json"),
                       help="Output file for compatibility report")
    
    args = parser.parse_args()
    
    try:
        manager = VersionManager(args.examples_dir)
        manager.discover_examples()
        
        if args.update_tags:
            manager.update_version_tags()
            return
            
        if args.migration_plan:
            from_version, to_version = args.migration_plan
            plan = manager.generate_migration_plan(from_version, to_version)
            print(json.dumps(plan, indent=2))
            return
            
        target_version = args.target_version or manager.get_current_version()
        if target_version == "unknown":
            print("❌ Could not determine VoiRS version. Please specify --target-version")
            sys.exit(1)
            
        compatibility_report = manager.check_version_compatibility(target_version)
        manager.print_compatibility_summary(compatibility_report)
        manager.save_compatibility_report(compatibility_report, args.output)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()