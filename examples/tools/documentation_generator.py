#!/usr/bin/env python3
"""
VoiRS Documentation Generator

Comprehensive documentation generation tool for VoiRS examples and API.
Supports multiple output formats and automated documentation workflows.
"""

import os
import sys
import json
import re
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import shutil

@dataclass
class ExampleInfo:
    """Information about a VoiRS example."""
    name: str
    path: str
    description: str
    category: str
    complexity: str
    dependencies: List[str]
    code_lines: int
    documentation_lines: int
    functions: List[str]
    structs: List[str]
    tests: List[str]
    
@dataclass
class APIInfo:
    """Information about VoiRS API elements."""
    name: str
    type: str  # function, struct, enum, trait
    signature: str
    description: str
    module: str
    visibility: str
    examples: List[str]
    
@dataclass
class DocumentationConfig:
    """Configuration for documentation generation."""
    input_paths: List[str]
    output_path: str
    formats: List[str]  # markdown, html, json, pdf
    include_examples: bool
    include_api: bool
    include_tutorials: bool
    include_reference: bool
    template_path: Optional[str]
    style_path: Optional[str]
    
class DocumentationGenerator:
    """Main documentation generator class."""
    
    def __init__(self, config: DocumentationConfig):
        self.config = config
        self.examples: List[ExampleInfo] = []
        self.api_elements: List[APIInfo] = []
        self.categories = set()
        
    def analyze_rust_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a Rust file and extract documentation information."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        info = {
            'functions': [],
            'structs': [],
            'enums': [],
            'traits': [],
            'tests': [],
            'doc_comments': [],
            'dependencies': [],
            'total_lines': len(content.splitlines()),
            'code_lines': 0,
            'doc_lines': 0,
            'comment_lines': 0
        }
        
        lines = content.splitlines()
        in_doc_comment = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Count line types
            if stripped.startswith('///') or stripped.startswith('//!'):
                info['doc_lines'] += 1
                info['doc_comments'].append({
                    'line': i + 1,
                    'content': stripped[3:].strip()
                })
            elif stripped.startswith('//'):
                info['comment_lines'] += 1
            elif stripped and not stripped.startswith('//'):
                info['code_lines'] += 1
                
            # Extract functions
            if re.match(r'^\s*(pub\s+)?(async\s+)?fn\s+(\w+)', line):
                match = re.search(r'fn\s+(\w+)', line)
                if match:
                    func_name = match.group(1)
                    # Get function signature
                    signature = line.strip()
                    # Look for closing brace or opening brace
                    j = i + 1
                    while j < len(lines) and '{' not in signature:
                        signature += ' ' + lines[j].strip()
                        j += 1
                        if j - i > 10:  # Prevent infinite loop
                            break
                    
                    info['functions'].append({
                        'name': func_name,
                        'signature': signature,
                        'line': i + 1,
                        'visibility': 'pub' if 'pub' in line else 'private'
                    })
                    
            # Extract structs
            if re.match(r'^\s*(pub\s+)?struct\s+(\w+)', line):
                match = re.search(r'struct\s+(\w+)', line)
                if match:
                    struct_name = match.group(1)
                    info['structs'].append({
                        'name': struct_name,
                        'line': i + 1,
                        'visibility': 'pub' if 'pub' in line else 'private'
                    })
                    
            # Extract enums
            if re.match(r'^\s*(pub\s+)?enum\s+(\w+)', line):
                match = re.search(r'enum\s+(\w+)', line)
                if match:
                    enum_name = match.group(1)
                    info['enums'].append({
                        'name': enum_name,
                        'line': i + 1,
                        'visibility': 'pub' if 'pub' in line else 'private'
                    })
                    
            # Extract traits
            if re.match(r'^\s*(pub\s+)?trait\s+(\w+)', line):
                match = re.search(r'trait\s+(\w+)', line)
                if match:
                    trait_name = match.group(1)
                    info['traits'].append({
                        'name': trait_name,
                        'line': i + 1,
                        'visibility': 'pub' if 'pub' in line else 'private'
                    })
                    
            # Extract tests
            if re.match(r'^\s*#\[test\]', line):
                if i + 1 < len(lines):
                    test_line = lines[i + 1]
                    match = re.search(r'fn\s+(\w+)', test_line)
                    if match:
                        test_name = match.group(1)
                        info['tests'].append({
                            'name': test_name,
                            'line': i + 2
                        })
                        
            # Extract use statements (dependencies)
            if re.match(r'^\s*use\s+', line):
                info['dependencies'].append(line.strip())
                
        return info
    
    def discover_examples(self) -> None:
        """Discover all VoiRS examples in the specified paths."""
        for input_path in self.config.input_paths:
            path = Path(input_path)
            if not path.exists():
                print(f"Warning: Path {input_path} does not exist")
                continue
                
            # Find all Rust files
            rust_files = list(path.glob("**/*.rs"))
            
            for rust_file in rust_files:
                if rust_file.name.startswith('.'):
                    continue
                    
                # Skip if it's a test file in tests directory
                if 'tests' in rust_file.parts and rust_file.parent.name == 'tests':
                    continue
                    
                analysis = self.analyze_rust_file(rust_file)
                
                # Determine category based on file name and path
                category = self.determine_category(rust_file)
                self.categories.add(category)
                
                # Determine complexity
                complexity = self.determine_complexity(analysis)
                
                # Extract description from doc comments
                description = self.extract_description(analysis['doc_comments'])
                
                example_info = ExampleInfo(
                    name=rust_file.stem,
                    path=str(rust_file),
                    description=description,
                    category=category,
                    complexity=complexity,
                    dependencies=[dep for dep in analysis['dependencies']],
                    code_lines=analysis['code_lines'],
                    documentation_lines=analysis['doc_lines'],
                    functions=[f['name'] for f in analysis['functions']],
                    structs=[s['name'] for s in analysis['structs']],
                    tests=[t['name'] for t in analysis['tests']]
                )
                
                self.examples.append(example_info)
                
                # Add API elements for public items
                for func in analysis['functions']:
                    if func['visibility'] == 'pub':
                        api_element = APIInfo(
                            name=func['name'],
                            type='function',
                            signature=func['signature'],
                            description=f"Function from {rust_file.name}",
                            module=rust_file.stem,
                            visibility='public',
                            examples=[rust_file.stem]
                        )
                        self.api_elements.append(api_element)
                        
    def determine_category(self, file_path: Path) -> str:
        """Determine the category of an example based on file path and name."""
        name = file_path.stem.lower()
        
        categories = {
            'basic': ['hello_world', 'basic_configuration', 'simple_synthesis'],
            'advanced': ['production_pipeline', 'performance_benchmarking', 'comprehensive_benchmark'],
            'integration': ['python_integration', 'cpp_integration', 'wasm_integration', 'mobile_integration', 'desktop_integration'],
            'audio': ['voice_cloning', 'voice_conversion', 'singing_synthesis', 'spatial_audio', 'emotion_control'],
            'testing': ['testing_framework', 'ab_testing', 'memory_profiling', 'debug_troubleshooting'],
            'ai': ['ai_integration', 'multimodal_integration', 'creative_applications'],
            'deployment': ['cloud_deployment', 'iot_edge_synthesis', 'game_integration', 'vr_ar_immersive'],
            'community': ['community_contributions', 'use_case_gallery', 'best_practices', 'faq_examples'],
            'tools': ['documentation_testing', 'streaming_synthesis_optimization', 'robust_error_handling']
        }
        
        for category, keywords in categories.items():
            if any(keyword in name for keyword in keywords):
                return category
                
        return 'miscellaneous'
    
    def determine_complexity(self, analysis: Dict[str, Any]) -> str:
        """Determine the complexity level of an example."""
        code_lines = analysis['code_lines']
        functions = len(analysis['functions'])
        structs = len(analysis['structs'])
        
        complexity_score = code_lines / 100 + functions * 2 + structs * 3
        
        if complexity_score < 5:
            return 'beginner'
        elif complexity_score < 15:
            return 'intermediate'
        else:
            return 'advanced'
    
    def extract_description(self, doc_comments: List[Dict[str, Any]]) -> str:
        """Extract a description from documentation comments."""
        if not doc_comments:
            return "No description available."
            
        # Take the first few doc comments as description
        description_parts = []
        for comment in doc_comments[:3]:  # First 3 comments
            content = comment['content'].strip()
            if content and not content.startswith('# ') and not content.startswith('## '):
                description_parts.append(content)
                
        if description_parts:
            return ' '.join(description_parts)
        else:
            return "No description available."
    
    def generate_markdown_documentation(self) -> str:
        """Generate comprehensive Markdown documentation."""
        md_content = []
        
        # Header
        md_content.append("# VoiRS Examples Documentation")
        md_content.append(f"\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        # Table of contents
        md_content.append("## Table of Contents")
        md_content.append("- [Overview](#overview)")
        md_content.append("- [Examples by Category](#examples-by-category)")
        md_content.append("- [Examples by Complexity](#examples-by-complexity)")
        md_content.append("- [API Reference](#api-reference)")
        md_content.append("- [Quick Reference](#quick-reference)")
        md_content.append("- [Statistics](#statistics)")
        md_content.append()
        
        # Overview
        md_content.append("## Overview")
        md_content.append(f"This documentation covers {len(self.examples)} VoiRS examples across {len(self.categories)} categories.")
        md_content.append(f"Total lines of code: {sum(ex.code_lines for ex in self.examples):,}")
        md_content.append(f"Total documentation lines: {sum(ex.documentation_lines for ex in self.examples):,}")
        md_content.append()
        
        # Examples by category
        md_content.append("## Examples by Category")
        md_content.append()
        
        for category in sorted(self.categories):
            category_examples = [ex for ex in self.examples if ex.category == category]
            if not category_examples:
                continue
                
            md_content.append(f"### {category.title()}")
            md_content.append()
            
            for example in sorted(category_examples, key=lambda x: x.name):
                md_content.append(f"#### {example.name}")
                md_content.append(f"**Path:** `{example.path}`")
                md_content.append(f"**Complexity:** {example.complexity}")
                md_content.append(f"**Description:** {example.description}")
                md_content.append(f"**Code Lines:** {example.code_lines}")
                md_content.append(f"**Functions:** {len(example.functions)}")
                md_content.append(f"**Tests:** {len(example.tests)}")
                
                if example.functions:
                    md_content.append(f"**Key Functions:** {', '.join(example.functions[:5])}")
                    if len(example.functions) > 5:
                        md_content.append(f" *(and {len(example.functions) - 5} more)*")
                        
                md_content.append()
        
        # Examples by complexity
        md_content.append("## Examples by Complexity")
        md_content.append()
        
        complexity_levels = ['beginner', 'intermediate', 'advanced']
        for complexity in complexity_levels:
            complexity_examples = [ex for ex in self.examples if ex.complexity == complexity]
            if not complexity_examples:
                continue
                
            md_content.append(f"### {complexity.title()}")
            md_content.append()
            
            for example in sorted(complexity_examples, key=lambda x: x.name):
                md_content.append(f"- **{example.name}** ({example.category}) - {example.description[:100]}...")
                
            md_content.append()
        
        # API Reference
        if self.config.include_api and self.api_elements:
            md_content.append("## API Reference")
            md_content.append()
            
            api_by_type = {}
            for api in self.api_elements:
                if api.type not in api_by_type:
                    api_by_type[api.type] = []
                api_by_type[api.type].append(api)
                
            for api_type, apis in api_by_type.items():
                md_content.append(f"### {api_type.title()}s")
                md_content.append()
                
                for api in sorted(apis, key=lambda x: x.name):
                    md_content.append(f"#### {api.name}")
                    md_content.append(f"**Module:** {api.module}")
                    md_content.append(f"**Signature:** `{api.signature}`")
                    md_content.append(f"**Description:** {api.description}")
                    if api.examples:
                        md_content.append(f"**Used in:** {', '.join(api.examples)}")
                    md_content.append()
        
        # Quick Reference
        md_content.append("## Quick Reference")
        md_content.append()
        md_content.append("### Getting Started")
        md_content.append("1. Start with `hello_world.rs` for basic synthesis")
        md_content.append("2. Review `basic_configuration.rs` for configuration patterns")
        md_content.append("3. Explore category-specific examples based on your use case")
        md_content.append()
        
        md_content.append("### Common Patterns")
        md_content.append("- **Error Handling:** See `robust_error_handling_patterns.rs`")
        md_content.append("- **Performance:** See `performance_benchmarking.rs`")
        md_content.append("- **Testing:** See `testing_framework_example.rs`")
        md_content.append("- **Production:** See `production_pipeline_example.rs`")
        md_content.append()
        
        # Statistics
        md_content.append("## Statistics")
        md_content.append()
        md_content.append("### Code Metrics")
        
        total_code_lines = sum(ex.code_lines for ex in self.examples)
        total_doc_lines = sum(ex.documentation_lines for ex in self.examples)
        total_functions = sum(len(ex.functions) for ex in self.examples)
        total_tests = sum(len(ex.tests) for ex in self.examples)
        
        md_content.append(f"- **Total Examples:** {len(self.examples)}")
        md_content.append(f"- **Total Code Lines:** {total_code_lines:,}")
        md_content.append(f"- **Total Documentation Lines:** {total_doc_lines:,}")
        md_content.append(f"- **Total Functions:** {total_functions}")
        md_content.append(f"- **Total Tests:** {total_tests}")
        md_content.append(f"- **Documentation Ratio:** {(total_doc_lines / max(total_code_lines, 1) * 100):.1f}%")
        md_content.append()
        
        # Category statistics
        md_content.append("### By Category")
        for category in sorted(self.categories):
            category_examples = [ex for ex in self.examples if ex.category == category]
            category_lines = sum(ex.code_lines for ex in category_examples)
            md_content.append(f"- **{category.title()}:** {len(category_examples)} examples, {category_lines:,} lines")
            
        md_content.append()
        
        # Complexity statistics
        md_content.append("### By Complexity")
        for complexity in complexity_levels:
            complexity_examples = [ex for ex in self.examples if ex.complexity == complexity]
            if complexity_examples:
                avg_lines = sum(ex.code_lines for ex in complexity_examples) // len(complexity_examples)
                md_content.append(f"- **{complexity.title()}:** {len(complexity_examples)} examples, avg {avg_lines} lines")
                
        return '\n'.join(md_content)
    
    def generate_json_documentation(self) -> str:
        """Generate JSON documentation for programmatic access."""
        json_doc = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator_version': '1.0.0',
                'total_examples': len(self.examples),
                'total_categories': len(self.categories)
            },
            'examples': [asdict(ex) for ex in self.examples],
            'api_elements': [asdict(api) for api in self.api_elements],
            'categories': list(sorted(self.categories)),
            'statistics': {
                'total_code_lines': sum(ex.code_lines for ex in self.examples),
                'total_doc_lines': sum(ex.documentation_lines for ex in self.examples),
                'total_functions': sum(len(ex.functions) for ex in self.examples),
                'total_tests': sum(len(ex.tests) for ex in self.examples),
                'by_category': {
                    category: {
                        'count': len([ex for ex in self.examples if ex.category == category]),
                        'code_lines': sum(ex.code_lines for ex in self.examples if ex.category == category)
                    }
                    for category in self.categories
                },
                'by_complexity': {
                    complexity: len([ex for ex in self.examples if ex.complexity == complexity])
                    for complexity in ['beginner', 'intermediate', 'advanced']
                }
            }
        }
        
        return json.dumps(json_doc, indent=2, default=str)
    
    def generate_quick_reference(self) -> str:
        """Generate a quick reference guide."""
        ref_content = []
        
        ref_content.append("# VoiRS Quick Reference")
        ref_content.append()
        
        # Getting started section
        ref_content.append("## Getting Started")
        beginner_examples = [ex for ex in self.examples if ex.complexity == 'beginner']
        for example in sorted(beginner_examples, key=lambda x: x.code_lines)[:5]:
            ref_content.append(f"- **{example.name}** - {example.description[:80]}...")
            
        ref_content.append()
        
        # Common patterns
        ref_content.append("## Common Patterns")
        ref_content.append()
        
        pattern_examples = {
            'Configuration': [ex for ex in self.examples if 'config' in ex.name.lower()],
            'Error Handling': [ex for ex in self.examples if 'error' in ex.name.lower()],
            'Performance': [ex for ex in self.examples if 'performance' in ex.name.lower() or 'benchmark' in ex.name.lower()],
            'Testing': [ex for ex in self.examples if 'test' in ex.name.lower()],
            'Integration': [ex for ex in self.examples if 'integration' in ex.name.lower()]
        }
        
        for pattern, examples in pattern_examples.items():
            if examples:
                ref_content.append(f"### {pattern}")
                for example in examples[:3]:  # Top 3 examples
                    ref_content.append(f"- `{example.name}` - {example.description[:60]}...")
                ref_content.append()
        
        # Cheat sheet
        ref_content.append("## Cheat Sheet")
        ref_content.append()
        ref_content.append("### Basic Synthesis")
        ref_content.append("```rust")
        ref_content.append("// Basic synthesis pattern")
        ref_content.append("let config = VoiRSConfig::default();")
        ref_content.append("let synthesizer = VoiRSSynthesizer::new(config)?;")
        ref_content.append("let audio = synthesizer.synthesize(\"Hello, world!\")?;")
        ref_content.append("```")
        ref_content.append()
        
        ref_content.append("### Configuration")
        ref_content.append("```rust")
        ref_content.append("// Configuration pattern")
        ref_content.append("let config = VoiRSConfig::builder()")
        ref_content.append("    .sample_rate(22050)")
        ref_content.append("    .quality(QualityLevel::High)")
        ref_content.append("    .build()?;")
        ref_content.append("```")
        ref_content.append()
        
        ref_content.append("### Error Handling")
        ref_content.append("```rust")
        ref_content.append("// Robust error handling")
        ref_content.append("match synthesizer.synthesize(text) {")
        ref_content.append("    Ok(audio) => process_audio(audio),")
        ref_content.append("    Err(VoiRSError::InvalidInput(msg)) => handle_input_error(msg),")
        ref_content.append("    Err(e) => handle_other_error(e),")
        ref_content.append("}")
        ref_content.append("```")
        ref_content.append()
        
        return '\n'.join(ref_content)
    
    def generate_html_documentation(self, markdown_content: str) -> str:
        """Generate HTML documentation from Markdown (basic implementation)."""
        # This is a basic HTML wrapper - in production, you'd use a proper Markdown to HTML converter
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoiRS Documentation</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3, h4 {{ color: #2c3e50; }}
        h1 {{ border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
        code {{ background: #f8f9fa; padding: 2px 4px; border-radius: 3px; font-family: 'Monaco', 'Menlo', monospace; }}
        pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        pre code {{ background: none; padding: 0; }}
        .toc {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #fff; border: 1px solid #ddd; border-radius: 5px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .category {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-left: 4px solid #3498db; }}
    </style>
</head>
<body>
    <div id="content">
        {self.markdown_to_basic_html(markdown_content)}
    </div>
</body>
</html>
"""
        return html_template
    
    def markdown_to_basic_html(self, markdown: str) -> str:
        """Convert basic Markdown to HTML (simplified implementation)."""
        html = markdown
        
        # Headers
        html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^#### (.*?)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
        
        # Code blocks
        html = re.sub(r'```(\w+)?\n(.*?)\n```', r'<pre><code>\2</code></pre>', html, flags=re.DOTALL)
        
        # Inline code
        html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
        
        # Bold
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        
        # Links
        html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)
        
        # Lists
        html = re.sub(r'^- (.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        
        # Paragraphs
        html = re.sub(r'\n\n', r'</p><p>', html)
        html = '<p>' + html + '</p>'
        
        # Clean up empty paragraphs
        html = re.sub(r'<p>\s*</p>', '', html)
        
        return html
    
    def generate_all_formats(self) -> Dict[str, str]:
        """Generate documentation in all requested formats."""
        results = {}
        
        if 'markdown' in self.config.formats:
            print("Generating Markdown documentation...")
            results['markdown'] = self.generate_markdown_documentation()
            
        if 'json' in self.config.formats:
            print("Generating JSON documentation...")
            results['json'] = self.generate_json_documentation()
            
        if 'reference' in self.config.formats:
            print("Generating Quick Reference...")
            results['reference'] = self.generate_quick_reference()
            
        if 'html' in self.config.formats and 'markdown' in results:
            print("Generating HTML documentation...")
            results['html'] = self.generate_html_documentation(results['markdown'])
            
        return results
    
    def save_documentation(self, docs: Dict[str, str]) -> None:
        """Save generated documentation to files."""
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_extensions = {
            'markdown': '.md',
            'json': '.json',
            'reference': '_reference.md',
            'html': '.html'
        }
        
        for format_name, content in docs.items():
            if format_name in file_extensions:
                filename = f"voirs_documentation{file_extensions[format_name]}"
                file_path = output_path / filename
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                print(f"Saved {format_name} documentation to: {file_path}")
    
    def run(self) -> None:
        """Run the complete documentation generation process."""
        print("VoiRS Documentation Generator")
        print("=" * 40)
        
        print(f"Discovering examples in: {', '.join(self.config.input_paths)}")
        self.discover_examples()
        print(f"Found {len(self.examples)} examples in {len(self.categories)} categories")
        
        print(f"Generating documentation in formats: {', '.join(self.config.formats)}")
        docs = self.generate_all_formats()
        
        print(f"Saving documentation to: {self.config.output_path}")
        self.save_documentation(docs)
        
        print("\nGeneration complete!")
        print(f"Examples analyzed: {len(self.examples)}")
        print(f"API elements found: {len(self.api_elements)}")
        print(f"Categories: {', '.join(sorted(self.categories))}")

def main():
    """Main entry point for the documentation generator."""
    parser = argparse.ArgumentParser(description="VoiRS Documentation Generator")
    parser.add_argument("--input", "-i", nargs="+", default=["."],
                       help="Input paths to analyze (default: current directory)")
    parser.add_argument("--output", "-o", default="./docs",
                       help="Output directory for documentation (default: ./docs)")
    parser.add_argument("--formats", "-f", nargs="+", 
                       choices=["markdown", "html", "json", "reference"],
                       default=["markdown", "json", "reference"],
                       help="Output formats to generate")
    parser.add_argument("--no-examples", action="store_true",
                       help="Skip example analysis")
    parser.add_argument("--no-api", action="store_true",
                       help="Skip API documentation")
    parser.add_argument("--template", "-t", 
                       help="Path to custom template directory")
    parser.add_argument("--style", "-s",
                       help="Path to custom style file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    config = DocumentationConfig(
        input_paths=args.input,
        output_path=args.output,
        formats=args.formats,
        include_examples=not args.no_examples,
        include_api=not args.no_api,
        include_tutorials=True,
        include_reference="reference" in args.formats,
        template_path=args.template,
        style_path=args.style
    )
    
    try:
        generator = DocumentationGenerator(config)
        generator.run()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())