#!/usr/bin/env python3
"""
VoiRS Coding Standards Enforcer
===============================

Comprehensive coding standards enforcement tool for VoiRS codebase.
Ensures consistent code style, documentation standards, naming conventions,
and project structure across all examples and crates.

Features:
- Rust code formatting and linting enforcement
- Documentation standards validation
- Naming convention enforcement
- Project structure validation
- Automated code fixes where possible
- Custom VoiRS-specific rules
- Integration with existing quality tools
- Configurable rule sets and exceptions

Usage:
    python standards_enforcer.py [OPTIONS]

Options:
    --workspace-dir PATH    Path to workspace root (default: ..)
    --examples-dir PATH     Path to examples directory (default: .)
    --config PATH          Path to standards config file
    --report PATH          Generate detailed standards report
    --fix                 Auto-fix violations where possible
    --check-only          Only check standards, don't fix
    --strict              Use strict enforcement (fail on warnings)
    --verbose             Enable verbose output
    --rules RULES         Comma-separated list of rule categories to check
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
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class StandardsViolation:
    """Represents a coding standards violation"""
    rule_id: str
    category: str  # formatting, documentation, naming, structure, etc.
    severity: str  # error, warning, info
    file_path: str
    line_number: Optional[int]
    column_number: Optional[int]
    description: str
    current_code: Optional[str]
    suggested_fix: Optional[str]
    auto_fixable: bool = False

@dataclass 
class StandardsConfig:
    """Configuration for coding standards"""
    formatting: Dict[str, Any]
    documentation: Dict[str, Any]
    naming: Dict[str, Any]
    structure: Dict[str, Any]
    custom_rules: Dict[str, Any]

@dataclass
class StandardsReport:
    """Comprehensive standards enforcement report"""
    timestamp: str
    files_checked: int
    total_violations: int
    violations_by_category: Dict[str, int]
    violations_by_severity: Dict[str, int]
    auto_fixed: int
    manual_fixes_needed: int
    violations: List[StandardsViolation]
    overall_score: float
    recommendations: List[str]

class VoiRSStandardsEnforcer:
    """Comprehensive coding standards enforcer for VoiRS"""

    def __init__(self, workspace_dir: Path, examples_dir: Path, config_path: Optional[Path] = None):
        self.workspace_dir = workspace_dir
        self.examples_dir = examples_dir
        self.config_path = config_path
        self.config = self._load_config()
        self.reports_dir = examples_dir / "standards_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        self.violations: List[StandardsViolation] = []
        self.files_checked: List[str] = []
        self.auto_fixed_count = 0

    def _load_config(self) -> StandardsConfig:
        """Load coding standards configuration"""
        default_config = {
            "formatting": {
                "rust_fmt": {
                    "max_width": 100,
                    "tab_spaces": 4,
                    "newline_style": "Unix",
                    "use_small_heuristics": "Default",
                    "fn_args_layout": "Tall",
                    "brace_style": "SameLineWhere",
                    "control_brace_style": "AlwaysSameLine",
                    "trailing_comma": "Vertical",
                    "match_arm_trailing_comma": True,
                    "match_block_trailing_comma": False,
                    "blank_lines_upper_bound": 1,
                    "blank_lines_lower_bound": 0
                },
                "line_length": 100,
                "indent_size": 4,
                "trailing_whitespace": False,
                "final_newline": True
            },
            "documentation": {
                "require_module_docs": True,
                "require_public_fn_docs": True,
                "require_public_struct_docs": True,
                "require_public_enum_docs": True,
                "require_public_trait_docs": True,
                "min_doc_length": 10,
                "doc_style": "rust_doc",  # /// or //!
                "require_examples_in_docs": False,
                "allow_missing_docs_attrs": False
            },
            "naming": {
                "snake_case_functions": True,
                "snake_case_variables": True,
                "pascal_case_types": True,
                "screaming_snake_case_constants": True,
                "kebab_case_crates": True,
                "max_identifier_length": 50,
                "min_identifier_length": 2,
                "forbidden_names": ["temp", "tmp", "foo", "bar", "baz"],
                "required_prefixes": {
                    "test_functions": "test_",
                    "benchmark_functions": "bench_"
                }
            },
            "structure": {
                "max_file_length": 2000,
                "max_function_length": 100,
                "max_struct_fields": 20,
                "max_enum_variants": 50,
                "require_mod_rs": False,
                "organize_imports": True,
                "group_imports": True,
                "sort_imports": True,
                "separate_std_imports": True
            },
            "custom_rules": {
                "voirs_specific": {
                    "require_error_handling": True,
                    "forbid_unwrap_in_examples": True,
                    "require_async_for_io": True,
                    "prefer_voirs_types": True,
                    "require_performance_comments": False
                },
                "quality": {
                    "max_cognitive_complexity": 15,
                    "max_cyclomatic_complexity": 10,
                    "require_unit_tests": False,
                    "min_test_coverage": 0.0
                }
            }
        }

        if self.config_path and self.config_path.exists():
            try:
                import toml
                with open(self.config_path, 'r') as f:
                    user_config = toml.load(f)
                    # Deep merge configurations
                    self._deep_merge(default_config, user_config)
            except ImportError:
                logger.warning("toml package not found, using default config")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")

        return StandardsConfig(
            formatting=default_config["formatting"],
            documentation=default_config["documentation"],
            naming=default_config["naming"],
            structure=default_config["structure"],
            custom_rules=default_config["custom_rules"]
        )

    def _deep_merge(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
        """Deep merge two dictionaries"""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def discover_rust_files(self) -> List[Path]:
        """Discover all Rust files to check"""
        rust_files = []
        
        # Check workspace root
        for pattern in ["**/*.rs"]:
            for file_path in self.workspace_dir.glob(pattern):
                if self._should_check_file(file_path):
                    rust_files.append(file_path)
        
        logger.info(f"Discovered {len(rust_files)} Rust files for standards checking")
        return rust_files

    def _should_check_file(self, file_path: Path) -> bool:
        """Check if a file should be included in standards checking"""
        # Skip target directories
        if "target" in file_path.parts:
            return False
        
        # Skip hidden files
        if any(part.startswith('.') for part in file_path.parts):
            return False
        
        # Skip generated files
        if "generated" in str(file_path).lower():
            return False
        
        return True

    def check_formatting_standards(self, rust_files: List[Path]) -> None:
        """Check Rust formatting standards"""
        logger.info("Checking formatting standards...")
        
        # Check rustfmt configuration
        rustfmt_config = self.workspace_dir / "rustfmt.toml"
        if not rustfmt_config.exists():
            self.violations.append(StandardsViolation(
                rule_id="FMT001",
                category="formatting",
                severity="warning",
                file_path=str(self.workspace_dir),
                line_number=None,
                column_number=None,
                description="rustfmt.toml configuration file missing",
                current_code=None,
                suggested_fix="Create rustfmt.toml with project formatting rules",
                auto_fixable=True
            ))
        
        # Run rustfmt check
        try:
            result = subprocess.run(
                ["cargo", "fmt", "--check"],
                cwd=self.workspace_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Parse rustfmt output for specific violations
                violations = self._parse_rustfmt_output(result.stdout)
                self.violations.extend(violations)
        except FileNotFoundError:
            self.violations.append(StandardsViolation(
                rule_id="FMT002",
                category="formatting",
                severity="error",
                file_path="system",
                line_number=None,
                column_number=None,
                description="rustfmt not found in PATH",
                current_code=None,
                suggested_fix="Install rustfmt with 'rustup component add rustfmt'",
                auto_fixable=False
            ))
        
        # Check line length and other formatting rules
        for file_path in rust_files:
            self._check_file_formatting(file_path)

    def _parse_rustfmt_output(self, output: str) -> List[StandardsViolation]:
        """Parse rustfmt output to extract violations"""
        violations = []
        
        for line in output.split('\n'):
            if line.strip() and not line.startswith('Diff in'):
                violations.append(StandardsViolation(
                    rule_id="FMT003",
                    category="formatting",
                    severity="warning",
                    file_path=line.strip(),
                    line_number=None,
                    column_number=None,
                    description="File not formatted according to rustfmt rules",
                    current_code=None,
                    suggested_fix="Run 'cargo fmt' to fix formatting",
                    auto_fixable=True
                ))
        
        return violations

    def _check_file_formatting(self, file_path: Path) -> None:
        """Check formatting standards for a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            self.files_checked.append(str(file_path))
            
            for i, line in enumerate(lines, 1):
                # Check line length
                if len(line.rstrip()) > self.config.formatting["line_length"]:
                    self.violations.append(StandardsViolation(
                        rule_id="FMT004",
                        category="formatting",
                        severity="warning",
                        file_path=str(file_path),
                        line_number=i,
                        column_number=len(line.rstrip()),
                        description=f"Line exceeds maximum length of {self.config.formatting['line_length']} characters",
                        current_code=line.rstrip(),
                        suggested_fix="Break line into multiple lines",
                        auto_fixable=False
                    ))
                
                # Check trailing whitespace
                if not self.config.formatting["trailing_whitespace"] and line.rstrip() != line.rstrip('\n'):
                    self.violations.append(StandardsViolation(
                        rule_id="FMT005",
                        category="formatting",
                        severity="info",
                        file_path=str(file_path),
                        line_number=i,
                        column_number=len(line.rstrip()),
                        description="Line has trailing whitespace",
                        current_code=line.rstrip('\n'),
                        suggested_fix="Remove trailing whitespace",
                        auto_fixable=True
                    ))
            
            # Check final newline
            if self.config.formatting["final_newline"] and lines and not lines[-1].endswith('\n'):
                self.violations.append(StandardsViolation(
                    rule_id="FMT006",
                    category="formatting",
                    severity="info",
                    file_path=str(file_path),
                    line_number=len(lines),
                    column_number=len(lines[-1]),
                    description="File should end with a newline",
                    current_code=lines[-1] if lines else "",
                    suggested_fix="Add final newline",
                    auto_fixable=True
                ))
                
        except Exception as e:
            logger.warning(f"Failed to check formatting for {file_path}: {e}")

    def check_documentation_standards(self, rust_files: List[Path]) -> None:
        """Check documentation standards"""
        logger.info("Checking documentation standards...")
        
        for file_path in rust_files:
            self._check_file_documentation(file_path)

    def _check_file_documentation(self, file_path: Path) -> None:
        """Check documentation standards for a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Check module-level documentation
            if self.config.documentation["require_module_docs"]:
                has_module_doc = any(line.strip().startswith('//!') for line in lines[:20])
                if not has_module_doc:
                    self.violations.append(StandardsViolation(
                        rule_id="DOC001",
                        category="documentation",
                        severity="warning",
                        file_path=str(file_path),
                        line_number=1,
                        column_number=1,
                        description="Module lacks top-level documentation",
                        current_code=None,
                        suggested_fix="Add //! module documentation at the top of the file",
                        auto_fixable=False
                    ))
            
            # Check public function documentation
            if self.config.documentation["require_public_fn_docs"]:
                self._check_function_documentation(file_path, content)
            
            # Check public struct documentation
            if self.config.documentation["require_public_struct_docs"]:
                self._check_struct_documentation(file_path, content)
            
        except Exception as e:
            logger.warning(f"Failed to check documentation for {file_path}: {e}")

    def _check_function_documentation(self, file_path: Path, content: str) -> None:
        """Check function documentation"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Look for public function definitions
            if re.match(r'^\s*pub\s+(async\s+)?fn\s+\w+', line):
                func_match = re.search(r'fn\s+(\w+)', line)
                if func_match:
                    func_name = func_match.group(1)
                    
                    # Check if previous lines have documentation
                    has_doc = False
                    j = i - 1
                    while j >= 0 and (lines[j].strip().startswith('///') or lines[j].strip() == ''):
                        if lines[j].strip().startswith('///'):
                            has_doc = True
                            break
                        j -= 1
                    
                    if not has_doc:
                        self.violations.append(StandardsViolation(
                            rule_id="DOC002",
                            category="documentation",
                            severity="warning",
                            file_path=str(file_path),
                            line_number=i + 1,
                            column_number=1,
                            description=f"Public function '{func_name}' lacks documentation",
                            current_code=line.strip(),
                            suggested_fix=f"Add /// documentation comment before function {func_name}",
                            auto_fixable=False
                        ))

    def _check_struct_documentation(self, file_path: Path, content: str) -> None:
        """Check struct documentation"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Look for public struct definitions
            if re.match(r'^\s*pub\s+struct\s+\w+', line):
                struct_match = re.search(r'struct\s+(\w+)', line)
                if struct_match:
                    struct_name = struct_match.group(1)
                    
                    # Check if previous lines have documentation
                    has_doc = False
                    j = i - 1
                    while j >= 0 and (lines[j].strip().startswith('///') or lines[j].strip() == ''):
                        if lines[j].strip().startswith('///'):
                            has_doc = True
                            break
                        j -= 1
                    
                    if not has_doc:
                        self.violations.append(StandardsViolation(
                            rule_id="DOC003",
                            category="documentation",
                            severity="warning",
                            file_path=str(file_path),
                            line_number=i + 1,
                            column_number=1,
                            description=f"Public struct '{struct_name}' lacks documentation",
                            current_code=line.strip(),
                            suggested_fix=f"Add /// documentation comment before struct {struct_name}",
                            auto_fixable=False
                        ))

    def check_naming_standards(self, rust_files: List[Path]) -> None:
        """Check naming convention standards"""
        logger.info("Checking naming standards...")
        
        for file_path in rust_files:
            self._check_file_naming(file_path)

    def _check_file_naming(self, file_path: Path) -> None:
        """Check naming standards for a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check function naming
            self._check_function_naming(file_path, content)
            
            # Check variable naming
            self._check_variable_naming(file_path, content)
            
            # Check type naming
            self._check_type_naming(file_path, content)
            
            # Check constant naming
            self._check_constant_naming(file_path, content)
            
        except Exception as e:
            logger.warning(f"Failed to check naming for {file_path}: {e}")

    def _check_function_naming(self, file_path: Path, content: str) -> None:
        """Check function naming conventions"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            func_match = re.search(r'fn\s+(\w+)', line)
            if func_match:
                func_name = func_match.group(1)
                
                # Check snake_case for functions
                if self.config.naming["snake_case_functions"]:
                    if not re.match(r'^[a-z][a-z0-9_]*$', func_name):
                        self.violations.append(StandardsViolation(
                            rule_id="NAM001",
                            category="naming",
                            severity="warning",
                            file_path=str(file_path),
                            line_number=i + 1,
                            column_number=line.find(func_name),
                            description=f"Function '{func_name}' should use snake_case naming",
                            current_code=line.strip(),
                            suggested_fix=f"Rename to {self._to_snake_case(func_name)}",
                            auto_fixable=False
                        ))
                
                # Check forbidden names
                if func_name in self.config.naming["forbidden_names"]:
                    self.violations.append(StandardsViolation(
                        rule_id="NAM002",
                        category="naming",
                        severity="error",
                        file_path=str(file_path),
                        line_number=i + 1,
                        column_number=line.find(func_name),
                        description=f"Function name '{func_name}' is forbidden",
                        current_code=line.strip(),
                        suggested_fix="Use a more descriptive name",
                        auto_fixable=False
                    ))
                
                # Check identifier length
                if len(func_name) < self.config.naming["min_identifier_length"]:
                    self.violations.append(StandardsViolation(
                        rule_id="NAM003",
                        category="naming",
                        severity="warning",
                        file_path=str(file_path),
                        line_number=i + 1,
                        column_number=line.find(func_name),
                        description=f"Function name '{func_name}' is too short (minimum {self.config.naming['min_identifier_length']} characters)",
                        current_code=line.strip(),
                        suggested_fix="Use a more descriptive name",
                        auto_fixable=False
                    ))
                
                if len(func_name) > self.config.naming["max_identifier_length"]:
                    self.violations.append(StandardsViolation(
                        rule_id="NAM004",
                        category="naming",
                        severity="warning",
                        file_path=str(file_path),
                        line_number=i + 1,
                        column_number=line.find(func_name),
                        description=f"Function name '{func_name}' is too long (maximum {self.config.naming['max_identifier_length']} characters)",
                        current_code=line.strip(),
                        suggested_fix="Use a shorter, more concise name",
                        auto_fixable=False
                    ))

    def _check_variable_naming(self, file_path: Path, content: str) -> None:
        """Check variable naming conventions"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Look for variable declarations
            var_matches = re.findall(r'let\s+(\w+)', line)
            for var_name in var_matches:
                if self.config.naming["snake_case_variables"]:
                    if not re.match(r'^[a-z][a-z0-9_]*$', var_name):
                        self.violations.append(StandardsViolation(
                            rule_id="NAM005",
                            category="naming",
                            severity="warning",
                            file_path=str(file_path),
                            line_number=i + 1,
                            column_number=line.find(var_name),
                            description=f"Variable '{var_name}' should use snake_case naming",
                            current_code=line.strip(),
                            suggested_fix=f"Rename to {self._to_snake_case(var_name)}",
                            auto_fixable=False
                        ))

    def _check_type_naming(self, file_path: Path, content: str) -> None:
        """Check type naming conventions"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Check struct names
            struct_match = re.search(r'struct\s+(\w+)', line)
            if struct_match:
                type_name = struct_match.group(1)
                if self.config.naming["pascal_case_types"]:
                    if not re.match(r'^[A-Z][A-Za-z0-9]*$', type_name):
                        self.violations.append(StandardsViolation(
                            rule_id="NAM006",
                            category="naming",
                            severity="warning",
                            file_path=str(file_path),
                            line_number=i + 1,
                            column_number=line.find(type_name),
                            description=f"Struct '{type_name}' should use PascalCase naming",
                            current_code=line.strip(),
                            suggested_fix=f"Rename to {self._to_pascal_case(type_name)}",
                            auto_fixable=False
                        ))
            
            # Check enum names
            enum_match = re.search(r'enum\s+(\w+)', line)
            if enum_match:
                type_name = enum_match.group(1)
                if self.config.naming["pascal_case_types"]:
                    if not re.match(r'^[A-Z][A-Za-z0-9]*$', type_name):
                        self.violations.append(StandardsViolation(
                            rule_id="NAM007",
                            category="naming",
                            severity="warning",
                            file_path=str(file_path),
                            line_number=i + 1,
                            column_number=line.find(type_name),
                            description=f"Enum '{type_name}' should use PascalCase naming",
                            current_code=line.strip(),
                            suggested_fix=f"Rename to {self._to_pascal_case(type_name)}",
                            auto_fixable=False
                        ))

    def _check_constant_naming(self, file_path: Path, content: str) -> None:
        """Check constant naming conventions"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Look for constant declarations
            const_match = re.search(r'const\s+(\w+)', line)
            if const_match:
                const_name = const_match.group(1)
                if self.config.naming["screaming_snake_case_constants"]:
                    if not re.match(r'^[A-Z][A-Z0-9_]*$', const_name):
                        self.violations.append(StandardsViolation(
                            rule_id="NAM008",
                            category="naming",
                            severity="warning",
                            file_path=str(file_path),
                            line_number=i + 1,
                            column_number=line.find(const_name),
                            description=f"Constant '{const_name}' should use SCREAMING_SNAKE_CASE naming",
                            current_code=line.strip(),
                            suggested_fix=f"Rename to {self._to_screaming_snake_case(const_name)}",
                            auto_fixable=False
                        ))

    def _to_snake_case(self, name: str) -> str:
        """Convert name to snake_case"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _to_pascal_case(self, name: str) -> str:
        """Convert name to PascalCase"""
        return ''.join(word.capitalize() for word in name.split('_'))

    def _to_screaming_snake_case(self, name: str) -> str:
        """Convert name to SCREAMING_SNAKE_CASE"""
        return self._to_snake_case(name).upper()

    def check_structure_standards(self, rust_files: List[Path]) -> None:
        """Check project structure standards"""
        logger.info("Checking structure standards...")
        
        for file_path in rust_files:
            self._check_file_structure(file_path)

    def _check_file_structure(self, file_path: Path) -> None:
        """Check structure standards for a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Check file length
            if len(lines) > self.config.structure["max_file_length"]:
                self.violations.append(StandardsViolation(
                    rule_id="STR001",
                    category="structure",
                    severity="warning",
                    file_path=str(file_path),
                    line_number=len(lines),
                    column_number=1,
                    description=f"File exceeds maximum length of {self.config.structure['max_file_length']} lines",
                    current_code=None,
                    suggested_fix="Consider breaking file into smaller modules",
                    auto_fixable=False
                ))
            
            # Check function length
            self._check_function_length(file_path, content)
            
            # Check import organization
            if self.config.structure["organize_imports"]:
                self._check_import_organization(file_path, content)
                
        except Exception as e:
            logger.warning(f"Failed to check structure for {file_path}: {e}")

    def _check_function_length(self, file_path: Path, content: str) -> None:
        """Check function length"""
        lines = content.split('\n')
        in_function = False
        function_start = 0
        function_name = ""
        brace_count = 0
        
        for i, line in enumerate(lines):
            if re.search(r'fn\s+(\w+)', line):
                func_match = re.search(r'fn\s+(\w+)', line)
                if func_match:
                    function_name = func_match.group(1)
                    function_start = i
                    in_function = True
                    brace_count = line.count('{') - line.count('}')
            elif in_function:
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    function_length = i - function_start + 1
                    if function_length > self.config.structure["max_function_length"]:
                        self.violations.append(StandardsViolation(
                            rule_id="STR002",
                            category="structure",
                            severity="warning",
                            file_path=str(file_path),
                            line_number=function_start + 1,
                            column_number=1,
                            description=f"Function '{function_name}' exceeds maximum length of {self.config.structure['max_function_length']} lines",
                            current_code=None,
                            suggested_fix="Consider breaking function into smaller functions",
                            auto_fixable=False
                        ))
                    in_function = False

    def _check_import_organization(self, file_path: Path, content: str) -> None:
        """Check import organization"""
        lines = content.split('\n')
        import_lines = []
        
        for i, line in enumerate(lines):
            if line.strip().startswith('use '):
                import_lines.append((i, line))
        
        if not import_lines:
            return
        
        # Check if imports are grouped
        if self.config.structure["group_imports"]:
            std_imports = []
            external_imports = []
            local_imports = []
            
            for i, line in import_lines:
                if 'std::' in line or 'core::' in line:
                    std_imports.append((i, line))
                elif line.strip().startswith('use crate::') or line.strip().startswith('use super::'):
                    local_imports.append((i, line))
                else:
                    external_imports.append((i, line))
            
            # Check grouping order
            expected_order = std_imports + external_imports + local_imports
            actual_order = sorted(import_lines, key=lambda x: x[0])
            
            if [line for _, line in expected_order] != [line for _, line in actual_order]:
                self.violations.append(StandardsViolation(
                    rule_id="STR003",
                    category="structure",
                    severity="info",
                    file_path=str(file_path),
                    line_number=import_lines[0][0] + 1,
                    column_number=1,
                    description="Imports should be grouped: std, external crates, local",
                    current_code=None,
                    suggested_fix="Reorganize imports by group",
                    auto_fixable=True
                ))

    def check_custom_voirs_rules(self, rust_files: List[Path]) -> None:
        """Check VoiRS-specific custom rules"""
        logger.info("Checking VoiRS-specific rules...")
        
        for file_path in rust_files:
            self._check_voirs_rules(file_path)

    def _check_voirs_rules(self, file_path: Path) -> None:
        """Check VoiRS-specific rules for a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for unwrap() in examples
            if self.config.custom_rules["voirs_specific"]["forbid_unwrap_in_examples"] and "examples" in str(file_path):
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if '.unwrap()' in line and not line.strip().startswith('//'):
                        self.violations.append(StandardsViolation(
                            rule_id="VRS001",
                            category="custom",
                            severity="error",
                            file_path=str(file_path),
                            line_number=i + 1,
                            column_number=line.find('.unwrap()'),
                            description="unwrap() is forbidden in examples - use proper error handling",
                            current_code=line.strip(),
                            suggested_fix="Use ? operator or match statement for error handling",
                            auto_fixable=False
                        ))
            
            # Check for proper error handling
            if self.config.custom_rules["voirs_specific"]["require_error_handling"]:
                self._check_error_handling(file_path, content)
                
        except Exception as e:
            logger.warning(f"Failed to check VoiRS rules for {file_path}: {e}")

    def _check_error_handling(self, file_path: Path, content: str) -> None:
        """Check for proper error handling patterns"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Look for functions that should have error handling
            if re.search(r'fn\s+\w+.*->.*Result', line):
                continue  # Function returns Result, good
            
            if re.search(r'fn\s+\w+', line) and ('io::' in content or 'File::' in content):
                # Function does I/O but doesn't return Result
                func_match = re.search(r'fn\s+(\w+)', line)
                if func_match:
                    func_name = func_match.group(1)
                    if not func_name.startswith('test_'):  # Skip test functions
                        self.violations.append(StandardsViolation(
                            rule_id="VRS002",
                            category="custom",
                            severity="warning",
                            file_path=str(file_path),
                            line_number=i + 1,
                            column_number=1,
                            description=f"Function '{func_name}' performs I/O but doesn't return Result",
                            current_code=line.strip(),
                            suggested_fix="Consider returning Result<T, Error> for proper error handling",
                            auto_fixable=False
                        ))

    def auto_fix_violations(self) -> int:
        """Automatically fix violations where possible"""
        logger.info("Attempting to auto-fix violations...")
        
        fixed_count = 0
        
        # Group violations by file for efficient fixing
        violations_by_file = {}
        for violation in self.violations:
            if violation.auto_fixable:
                if violation.file_path not in violations_by_file:
                    violations_by_file[violation.file_path] = []
                violations_by_file[violation.file_path].append(violation)
        
        for file_path, file_violations in violations_by_file.items():
            try:
                fixed_count += self._fix_file_violations(Path(file_path), file_violations)
            except Exception as e:
                logger.warning(f"Failed to auto-fix violations in {file_path}: {e}")
        
        # Run formatting fixes
        try:
            # Run rustfmt
            result = subprocess.run(
                ["cargo", "fmt"],
                cwd=self.workspace_dir,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                fixed_count += 1
                logger.info("✅ Applied rustfmt formatting fixes")
        except Exception as e:
            logger.warning(f"Failed to run rustfmt: {e}")
        
        self.auto_fixed_count = fixed_count
        logger.info(f"Auto-fixed {fixed_count} violations")
        return fixed_count

    def _fix_file_violations(self, file_path: Path, violations: List[StandardsViolation]) -> int:
        """Fix violations in a single file"""
        if not file_path.exists():
            return 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            modified = False
            
            for violation in violations:
                if violation.rule_id == "FMT005":  # Trailing whitespace
                    if violation.line_number and violation.line_number <= len(lines):
                        line_idx = violation.line_number - 1
                        lines[line_idx] = lines[line_idx].rstrip()
                        modified = True
                
                elif violation.rule_id == "FMT006":  # Final newline
                    if lines and not content.endswith('\n'):
                        lines.append('')
                        modified = True
            
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                return 1
            
        except Exception as e:
            logger.warning(f"Failed to fix violations in {file_path}: {e}")
        
        return 0

    def generate_report(self) -> StandardsReport:
        """Generate comprehensive standards report"""
        logger.info("Generating standards report...")
        
        violations_by_category = {}
        violations_by_severity = {}
        
        for violation in self.violations:
            violations_by_category[violation.category] = violations_by_category.get(violation.category, 0) + 1
            violations_by_severity[violation.severity] = violations_by_severity.get(violation.severity, 0) + 1
        
        # Calculate overall score
        error_weight = -10
        warning_weight = -5
        info_weight = -1
        
        score_deduction = (
            violations_by_severity.get('error', 0) * error_weight +
            violations_by_severity.get('warning', 0) * warning_weight +
            violations_by_severity.get('info', 0) * info_weight
        )
        
        overall_score = max(0, 100 + score_deduction)
        
        # Generate recommendations
        recommendations = []
        
        if violations_by_severity.get('error', 0) > 0:
            recommendations.append(f"Fix {violations_by_severity['error']} critical errors")
        
        if violations_by_category.get('formatting', 0) > 0:
            recommendations.append("Run 'cargo fmt' to fix formatting issues")
        
        if violations_by_category.get('documentation', 0) > 0:
            recommendations.append("Add documentation to public APIs")
        
        if violations_by_category.get('naming', 0) > 0:
            recommendations.append("Follow Rust naming conventions")
        
        auto_fixable_count = sum(1 for v in self.violations if v.auto_fixable)
        manual_fixes_needed = len(self.violations) - auto_fixable_count
        
        return StandardsReport(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            files_checked=len(self.files_checked),
            total_violations=len(self.violations),
            violations_by_category=violations_by_category,
            violations_by_severity=violations_by_severity,
            auto_fixed=self.auto_fixed_count,
            manual_fixes_needed=manual_fixes_needed,
            violations=self.violations,
            overall_score=overall_score,
            recommendations=recommendations
        )

    def save_report(self, report: StandardsReport, output_path: Path) -> None:
        """Save standards report to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        logger.info(f"Standards report saved to: {output_path}")

    def print_summary(self, report: StandardsReport) -> None:
        """Print standards report summary"""
        print("\n" + "="*60)
        print("📏 VoiRS Coding Standards Report")
        print("="*60)
        print(f"📊 Overall Score: {report.overall_score:.1f}/100")
        print(f"📁 Files Checked: {report.files_checked}")
        print(f"⚠️  Total Violations: {report.total_violations}")
        print(f"🔧 Auto-Fixed: {report.auto_fixed}")
        print(f"✋ Manual Fixes Needed: {report.manual_fixes_needed}")
        
        if report.violations_by_severity:
            print("\n📋 Violations by Severity:")
            for severity, count in sorted(report.violations_by_severity.items()):
                emoji = {"error": "❌", "warning": "⚠️ ", "info": "ℹ️ "}
                print(f"  {emoji.get(severity, '•')} {severity.title()}: {count}")
        
        if report.violations_by_category:
            print("\n🏷️  Violations by Category:")
            for category, count in sorted(report.violations_by_category.items()):
                print(f"  • {category.title()}: {count}")
        
        # Top violations
        if report.violations:
            print("\n🔝 Top Violations:")
            severity_order = {"error": 0, "warning": 1, "info": 2}
            sorted_violations = sorted(report.violations, key=lambda x: (severity_order.get(x.severity, 3), x.category))
            
            for violation in sorted_violations[:10]:  # Show top 10
                emoji = {"error": "❌", "warning": "⚠️ ", "info": "ℹ️ "}
                print(f"  {emoji.get(violation.severity, '•')} {violation.description}")
                if violation.line_number:
                    print(f"    📍 {violation.file_path}:{violation.line_number}")
                else:
                    print(f"    📍 {violation.file_path}")
                if violation.suggested_fix:
                    print(f"    💡 {violation.suggested_fix}")
                print()
        
        if report.recommendations:
            print("💡 Recommendations:")
            for rec in report.recommendations:
                print(f"  • {rec}")

    def run_standards_check(self, rule_categories: Optional[List[str]] = None, auto_fix: bool = False) -> StandardsReport:
        """Run comprehensive standards check"""
        logger.info("Starting VoiRS coding standards enforcement...")
        
        start_time = time.time()
        
        try:
            # Discover files
            rust_files = self.discover_rust_files()
            
            # Run checks based on categories
            all_categories = ["formatting", "documentation", "naming", "structure", "custom"]
            categories_to_check = rule_categories if rule_categories else all_categories
            
            if "formatting" in categories_to_check:
                self.check_formatting_standards(rust_files)
            
            if "documentation" in categories_to_check:
                self.check_documentation_standards(rust_files)
            
            if "naming" in categories_to_check:
                self.check_naming_standards(rust_files)
            
            if "structure" in categories_to_check:
                self.check_structure_standards(rust_files)
            
            if "custom" in categories_to_check:
                self.check_custom_voirs_rules(rust_files)
            
            # Auto-fix if requested
            if auto_fix:
                self.auto_fix_violations()
            
            # Generate report
            report = self.generate_report()
            
            duration = time.time() - start_time
            logger.info(f"Standards check completed in {duration:.1f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"Standards check failed: {e}")
            raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="VoiRS Coding Standards Enforcer")
    parser.add_argument("--workspace-dir", "-w", type=Path, default=Path(".."),
                       help="Path to workspace root")
    parser.add_argument("--examples-dir", "-e", type=Path, default=Path("."),
                       help="Path to examples directory")
    parser.add_argument("--config", "-c", type=Path,
                       help="Path to standards config file")
    parser.add_argument("--report", "-r", type=Path,
                       help="Generate detailed standards report at path")
    parser.add_argument("--fix", action="store_true",
                       help="Auto-fix violations where possible")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check standards, don't fix")
    parser.add_argument("--strict", action="store_true",
                       help="Use strict enforcement (fail on warnings)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--rules", type=str,
                       help="Comma-separated list of rule categories to check")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Parse rule categories
        rule_categories = None
        if args.rules:
            rule_categories = [cat.strip() for cat in args.rules.split(',')]
        
        # Initialize standards enforcer
        enforcer = VoiRSStandardsEnforcer(args.workspace_dir, args.examples_dir, args.config)
        
        # Run standards check
        report = enforcer.run_standards_check(
            rule_categories=rule_categories,
            auto_fix=args.fix and not args.check_only
        )
        
        # Print summary
        enforcer.print_summary(report)
        
        # Save detailed report if requested
        if args.report:
            enforcer.save_report(report, args.report)
        
        # Return exit code based on violations found
        if report.violations_by_severity.get('error', 0) > 0:
            logger.error("Critical standards violations found")
            return 2
        elif args.strict and report.total_violations > 0:
            logger.warning("Standards violations found (strict mode)")
            return 1
        elif report.violations_by_severity.get('warning', 0) > 0:
            logger.warning("Standards warnings found")
            return 1
        else:
            logger.info("Standards check passed")
            return 0
            
    except Exception as e:
        logger.error(f"Standards check failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 3

if __name__ == "__main__":
    sys.exit(main())