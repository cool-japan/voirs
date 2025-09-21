#!/usr/bin/env python3
"""
VoiRS Example Generator

This tool generates new VoiRS examples following established patterns and best practices.
It provides templates and automatic generation for various types of examples.

Usage:
    python3 example_generator.py --type <example_type> --name <example_name> [options]

Example Types:
    - basic: Simple TTS example
    - advanced: Complex feature demonstration
    - integration: Platform or library integration
    - performance: Benchmarking or optimization
    - testing: Test framework or validation
    - quality: Quality assessment or comparison
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Example templates
EXAMPLE_TEMPLATES = {
    "basic": {
        "description": "Basic text-to-speech example",
        "dependencies": ["voirs", "tokio", "anyhow", "tracing", "tracing-subscriber"],
        "features": ["simple_synthesis"],
        "template": """//! {title} - VoiRS {description}
//!
//! {detailed_description}
//!
//! ## What this example does:
//! {what_it_does}
//!
//! ## Prerequisites:
//! - Rust 1.70+ installed
//! - VoiRS dependencies configured
//!
//! ## Running this example:
//! ```bash
//! cargo run --example {name}
//! ```
//!
//! ## Expected output:
//! {expected_output}

use anyhow::Result;
use tracing::{{info, warn}};
use voirs::{{
    create_acoustic, create_g2p, create_vocoder,
    AcousticBackend, G2pBackend, VocoderBackend,
    VoirsPipelineBuilder,
}};

#[tokio::main]
async fn main() -> Result<()> {{
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("🎤 VoiRS {title}");
    info!("========================{{}}=", "=".repeat({title}.len()));

    // TODO: Implement example logic here
    {implementation_skeleton}

    info!("✅ Example completed successfully!");
    Ok(())
}}

{additional_functions}
"""
    },
    "advanced": {
        "description": "Advanced feature demonstration",
        "dependencies": ["voirs", "voirs-sdk", "tokio", "anyhow", "tracing", "serde"],
        "features": ["advanced_synthesis", "real_time"],
        "template": """//! {title} - Advanced VoiRS {description}
//!
//! {detailed_description}
//!
//! ## Features Demonstrated:
//! {features_demonstrated}
//!
//! ## Requirements:
//! - VoiRS SDK with advanced features
//! - Sufficient system resources
//!
//! ## Usage:
//! ```bash
//! cargo run --example {name} -- [options]
//! ```

use anyhow::{{Context, Result}};
use serde::{{Deserialize, Serialize}};
use std::{{sync::Arc, time::Instant}};
use tokio::{{sync::RwLock, time::sleep}};
use tracing::{{debug, error, info, warn}};
use voirs_sdk::{{
    builder::VoirsPipelineBuilder,
    config::{{PipelineConfig, SynthesisConfig}},
    pipeline::Pipeline,
}};

/// Configuration for this example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleConfig {{
    /// Enable debug output
    pub debug: bool,
    /// Processing timeout in seconds
    pub timeout_secs: u64,
    /// Quality level (1-5)
    pub quality: u8,
}}

impl Default for ExampleConfig {{
    fn default() -> Self {{
        Self {{
            debug: false,
            timeout_secs: 30,
            quality: 3,
        }}
    }}
}}

#[tokio::main]
async fn main() -> Result<()> {{
    // Initialize enhanced logging
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(true)
        .with_thread_ids(true)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("🚀 VoiRS {title}");
    info!("Advanced Feature Demonstration");
    info!("=".repeat(50));

    // Load configuration
    let config = load_config().await?;
    info!("📋 Configuration: {{:?}}", config);

    // TODO: Implement advanced example logic
    {implementation_skeleton}

    info!("🎉 Advanced example completed successfully!");
    Ok(())
}}

async fn load_config() -> Result<ExampleConfig> {{
    // Try to load from file, fall back to default
    let config_path = "example_config.json";
    if Path::new(config_path).exists() {{
        let content = tokio::fs::read_to_string(config_path).await?;
        serde_json::from_str(&content).context("Failed to parse config")
    }} else {{
        Ok(ExampleConfig::default())
    }}
}}

{additional_functions}
"""
    },
    "integration": {
        "description": "Platform or library integration",
        "dependencies": ["voirs", "voirs-sdk", "tokio", "anyhow", "serde"],
        "features": ["integration", "cross_platform"],
        "template": """//! {title} - VoiRS {integration_type} Integration
//!
//! {detailed_description}
//!
//! ## Integration Features:
//! {integration_features}
//!
//! ## Platform Support:
//! {platform_support}
//!
//! ## Setup Instructions:
//! {setup_instructions}

use anyhow::{{Context, Result}};
use std::{{collections::HashMap, sync::Arc}};
use tokio::sync::RwLock;
use tracing::{{debug, error, info, instrument, warn}};
use voirs_sdk::{{
    builder::VoirsPipelineBuilder,
    config::PipelineConfig,
    pipeline::Pipeline,
}};

/// Integration manager for {integration_type}
pub struct {integration_class}Manager {{
    pipeline: Arc<RwLock<Option<Pipeline>>>,
    config: IntegrationConfig,
    is_initialized: bool,
}}

#[derive(Debug, Clone)]
pub struct IntegrationConfig {{
    /// Platform-specific settings
    pub platform_settings: HashMap<String, String>,
    /// Performance optimization level
    pub optimization_level: u8,
    /// Enable compatibility mode
    pub compatibility_mode: bool,
}}

impl {integration_class}Manager {{
    pub fn new(config: IntegrationConfig) -> Self {{
        Self {{
            pipeline: Arc::new(RwLock::new(None)),
            config,
            is_initialized: false,
        }}
    }}

    #[instrument(skip(self))]
    pub async fn initialize(&mut self) -> Result<()> {{
        info!("Initializing {integration_type} integration...");
        
        // TODO: Implement integration initialization
        {integration_init}

        self.is_initialized = true;
        info!("✅ {integration_type} integration initialized successfully");
        Ok(())
    }}

    #[instrument(skip(self))]
    pub async fn process(&self, input: &str) -> Result<Vec<u8>> {{
        if !self.is_initialized {{
            return Err(anyhow::anyhow!("Integration not initialized"));
        }}

        // TODO: Implement integration processing
        {integration_process}

        Ok(vec![])
    }}
}}

#[tokio::main]
async fn main() -> Result<()> {{
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("🔌 VoiRS {title}");
    info!("{integration_type} Integration Example");
    info!("=".repeat(40));

    // Initialize integration
    let config = IntegrationConfig {{
        platform_settings: HashMap::new(),
        optimization_level: 3,
        compatibility_mode: false,
    }};
    
    let mut manager = {integration_class}Manager::new(config);
    manager.initialize().await?;

    // TODO: Demonstrate integration usage
    {usage_demo}

    info!("🎉 Integration example completed!");
    Ok(())
}}

{additional_functions}
"""
    },
    "performance": {
        "description": "Performance benchmarking and optimization",
        "dependencies": ["voirs", "voirs-sdk", "tokio", "anyhow", "sysinfo", "serde"],
        "features": ["benchmarking", "performance_monitoring"],
        "template": """//! {title} - VoiRS Performance {benchmark_type}
//!
//! {detailed_description}
//!
//! ## Benchmarking Capabilities:
//! {benchmarking_capabilities}
//!
//! ## Metrics Collected:
//! {metrics_collected}
//!
//! ## Usage:
//! ```bash
//! cargo run --example {name} -- --iterations 100 --profile performance
//! ```

use anyhow::{{Context, Result}};
use serde::{{Deserialize, Serialize}};
use std::{{
    collections::HashMap,
    sync::{{Arc, atomic::{{AtomicU64, Ordering}}}},
    time::{{Duration, Instant}},
}};
use sysinfo::{{System, SystemExt, ProcessExt}};
use tokio::{{sync::RwLock, time::sleep}};
use tracing::{{debug, error, info, instrument, warn}};
use voirs_sdk::{{
    builder::VoirsPipelineBuilder,
    config::{{PipelineConfig, SynthesisConfig}},
    pipeline::Pipeline,
}};

/// Performance metrics collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {{
    /// Total processing time
    pub total_time_ms: u64,
    /// Average processing time per operation
    pub avg_time_ms: f64,
    /// Peak memory usage in MB
    pub peak_memory_mb: u64,
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    /// Real-time factor (processing_time / audio_duration)
    pub real_time_factor: f64,
    /// Throughput (operations per second)
    pub throughput: f64,
    /// Success rate percentage
    pub success_rate: f64,
}}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {{
    /// Number of iterations to run
    pub iterations: usize,
    /// Warmup iterations before measurement
    pub warmup_iterations: usize,
    /// Enable detailed profiling
    pub detailed_profiling: bool,
    /// Target quality level
    pub quality_level: u8,
}}

pub struct PerformanceBenchmark {{
    config: BenchmarkConfig,
    pipeline: Arc<RwLock<Option<Pipeline>>>,
    metrics: Arc<RwLock<PerformanceMetrics>>,
    system: System,
    operation_count: AtomicU64,
}}

impl PerformanceBenchmark {{
    pub fn new(config: BenchmarkConfig) -> Self {{
        Self {{
            config,
            pipeline: Arc::new(RwLock::new(None)),
            metrics: Arc::new(RwLock::new(PerformanceMetrics {{
                total_time_ms: 0,
                avg_time_ms: 0.0,
                peak_memory_mb: 0,
                cpu_utilization: 0.0,
                real_time_factor: 0.0,
                throughput: 0.0,
                success_rate: 0.0,
            }})),
            system: System::new_all(),
            operation_count: AtomicU64::new(0),
        }}
    }}

    #[instrument(skip(self))]
    pub async fn initialize(&mut self) -> Result<()> {{
        info!("Initializing performance benchmark...");
        
        // TODO: Initialize pipeline for benchmarking
        {benchmark_init}

        info!("✅ Benchmark initialized");
        Ok(())
    }}

    #[instrument(skip(self))]
    pub async fn run_benchmark(&mut self) -> Result<PerformanceMetrics> {{
        info!("🚀 Starting performance benchmark");
        info!("Iterations: {{}}, Warmup: {{}}", self.config.iterations, self.config.warmup_iterations);

        // Warmup phase
        info!("🔥 Running warmup iterations...");
        for i in 0..self.config.warmup_iterations {{
            self.run_single_operation(true).await?;
            if i % 10 == 0 {{
                debug!("Warmup iteration {{}}/{{}}", i, self.config.warmup_iterations);
            }}
        }}

        // Measurement phase
        info!("📊 Running measurement iterations...");
        let start_time = Instant::now();
        let mut successful_operations = 0;

        for i in 0..self.config.iterations {{
            match self.run_single_operation(false).await {{
                Ok(_) => successful_operations += 1,
                Err(e) => warn!("Operation {{}} failed: {{}}", i, e),
            }}

            if i % 50 == 0 && i > 0 {{
                info!("Progress: {{}}/{{}}", i, self.config.iterations);
            }}
        }}

        let total_time = start_time.elapsed();
        
        // Calculate final metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_time_ms = total_time.as_millis() as u64;
        metrics.avg_time_ms = total_time.as_millis() as f64 / self.config.iterations as f64;
        metrics.throughput = self.config.iterations as f64 / total_time.as_secs_f64();
        metrics.success_rate = (successful_operations as f64 / self.config.iterations as f64) * 100.0;

        info!("📈 Benchmark completed!");
        info!("Total time: {{:.2}}s", total_time.as_secs_f64());
        info!("Average time: {{:.2}}ms", metrics.avg_time_ms);
        info!("Throughput: {{:.2}} ops/sec", metrics.throughput);
        info!("Success rate: {{:.1}}%", metrics.success_rate);

        Ok(metrics.clone())
    }}

    async fn run_single_operation(&mut self, is_warmup: bool) -> Result<()> {{
        let operation_start = Instant::now();
        
        // TODO: Implement single benchmark operation
        {single_operation}

        let operation_time = operation_start.elapsed();
        self.operation_count.fetch_add(1, Ordering::Relaxed);

        if !is_warmup && self.config.detailed_profiling {{
            debug!("Operation completed in {{:.2}}ms", operation_time.as_millis());
        }}

        Ok(())
    }}

    pub async fn generate_report(&self, output_path: &str) -> Result<()> {{
        let metrics = self.metrics.read().await;
        let report = serde_json::to_string_pretty(&*metrics)?;
        
        tokio::fs::write(output_path, report).await?;
        info!("📄 Performance report saved to: {{}}", output_path);
        
        Ok(())
    }}
}}

#[tokio::main]
async fn main() -> Result<()> {{
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("⚡ VoiRS {title}");
    info!("Performance Benchmarking Suite");
    info!("=".repeat(40));

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let iterations = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let config = BenchmarkConfig {{
        iterations,
        warmup_iterations: iterations / 10,
        detailed_profiling: true,
        quality_level: 3,
    }};

    // Run benchmark
    let mut benchmark = PerformanceBenchmark::new(config);
    benchmark.initialize().await?;
    
    let metrics = benchmark.run_benchmark().await?;
    benchmark.generate_report("performance_report.json").await?;

    info!("🎉 Performance benchmark completed!");
    Ok(())
}}

{additional_functions}
"""
    }
}

# Additional function templates
FUNCTION_TEMPLATES = {
    "error_handling": """
/// Enhanced error handling with context
fn handle_synthesis_error(error: anyhow::Error) -> Result<()> {
    error!("Synthesis failed: {:?}", error);
    
    // Attempt recovery or provide helpful error message
    if let Some(io_error) = error.downcast_ref::<std::io::Error>() {
        warn!("I/O Error detected: {}", io_error);
        return Err(anyhow::anyhow!("File system error - check permissions and disk space"));
    }
    
    Err(error)
}
""",
    "configuration": """
/// Load configuration from file or environment
async fn load_configuration() -> Result<HashMap<String, String>> {
    let mut config = HashMap::new();
    
    // Try config file first
    if let Ok(content) = tokio::fs::read_to_string("config.toml").await {
        // Parse TOML config
        debug!("Loaded configuration from file");
    }
    
    // Override with environment variables
    for (key, value) in std::env::vars() {
        if key.starts_with("VOIRS_") {
            config.insert(key, value);
        }
    }
    
    Ok(config)
}
""",
    "monitoring": """
/// System resource monitoring
struct ResourceMonitor {
    start_time: Instant,
    initial_memory: u64,
}

impl ResourceMonitor {
    fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        
        Self {
            start_time: Instant::now(),
            initial_memory: system.used_memory(),
        }
    }
    
    fn report(&self) -> String {
        let elapsed = self.start_time.elapsed();
        let mut system = System::new_all();
        system.refresh_all();
        
        format!(
            "Elapsed: {:.2}s, Memory: {:.1}MB", 
            elapsed.as_secs_f64(),
            (system.used_memory() - self.initial_memory) as f64 / 1024.0 / 1024.0
        )
    }
}
"""
}

class ExampleGenerator:
    """Generate VoiRS examples from templates"""
    
    def __init__(self, examples_dir: str):
        self.examples_dir = Path(examples_dir)
        self.cargo_toml_path = self.examples_dir / "Cargo.toml"
        
    def generate_example(self, 
                        example_type: str,
                        name: str,
                        title: Optional[str] = None,
                        description: Optional[str] = None,
                        **kwargs) -> Tuple[str, str]:
        """Generate a new example file and return (content, cargo_entry)"""
        
        if example_type not in EXAMPLE_TEMPLATES:
            raise ValueError(f"Unknown example type: {example_type}. Available: {list(EXAMPLE_TEMPLATES.keys())}")
            
        template_config = EXAMPLE_TEMPLATES[example_type]
        
        # Generate title and description
        if not title:
            title = name.replace('_', ' ').title()
        if not description:
            description = template_config["description"]
            
        # Prepare template variables
        template_vars = {
            "title": title,
            "name": name,
            "description": description,
            "detailed_description": self._generate_detailed_description(example_type, **kwargs),
            "what_it_does": self._generate_what_it_does(example_type, **kwargs),
            "expected_output": self._generate_expected_output(example_type, **kwargs),
            "implementation_skeleton": self._generate_implementation_skeleton(example_type, **kwargs),
            "additional_functions": self._generate_additional_functions(example_type, **kwargs),
            **kwargs
        }
        
        # Generate content from template
        content = template_config["template"].format(**template_vars)
        
        # Generate Cargo.toml entry
        cargo_entry = f'''
[[example]]
name = "{name}"
path = "{name}.rs"
'''
        
        return content, cargo_entry
        
    def _generate_detailed_description(self, example_type: str, **kwargs) -> str:
        """Generate detailed description based on example type"""
        descriptions = {
            "basic": "This example demonstrates the fundamental concepts of VoiRS text-to-speech synthesis.\nIt shows how to set up a basic pipeline and generate audio from text.",
            "advanced": "This advanced example showcases sophisticated VoiRS features and optimization techniques.\nIt demonstrates best practices for production-ready applications.",
            "integration": f"This example shows how to integrate VoiRS with {kwargs.get('integration_type', 'external systems')}.\nIt provides practical patterns for real-world deployment scenarios.",
            "performance": "This performance-focused example demonstrates benchmarking and optimization techniques.\nIt shows how to measure and improve synthesis performance."
        }
        return descriptions.get(example_type, "A comprehensive VoiRS example.")
        
    def _generate_what_it_does(self, example_type: str, **kwargs) -> str:
        """Generate what the example does section"""
        what_it_does = {
            "basic": "//! 1. Initialize a basic TTS pipeline\n//! 2. Synthesize sample text\n//! 3. Save audio output\n//! 4. Display basic statistics",
            "advanced": "//! 1. Configure advanced synthesis parameters\n//! 2. Demonstrate multiple synthesis modes\n//! 3. Show error handling and recovery\n//! 4. Provide performance monitoring",
            "integration": f"//! 1. Initialize {kwargs.get('integration_type', 'platform')} integration\n//! 2. Configure platform-specific settings\n//! 3. Demonstrate seamless operation\n//! 4. Handle platform-specific requirements",
            "performance": "//! 1. Set up performance benchmarking\n//! 2. Run multiple synthesis iterations\n//! 3. Collect detailed metrics\n//! 4. Generate performance reports"
        }
        return what_it_does.get(example_type, "//! 1. Demonstrate VoiRS functionality\n//! 2. Show best practices\n//! 3. Provide practical examples")
        
    def _generate_expected_output(self, example_type: str, **kwargs) -> str:
        """Generate expected output section"""
        outputs = {
            "basic": "//! - Progress logs in console\n//! - Generated audio file\n//! - Basic performance statistics",
            "advanced": "//! - Detailed synthesis logs\n//! - Multiple output files\n//! - Advanced metrics and statistics\n//! - Performance analysis",
            "integration": f"//! - {kwargs.get('integration_type', 'Platform')} integration logs\n//! - Successful platform communication\n//! - Integration status reports",
            "performance": "//! - Benchmark progress logs\n//! - Performance metrics display\n//! - JSON performance report\n//! - Resource utilization statistics"
        }
        return outputs.get(example_type, "//! - Console output showing example progress\n//! - Generated files or results")
        
    def _generate_implementation_skeleton(self, example_type: str, **kwargs) -> str:
        """Generate implementation skeleton based on type"""
        skeletons = {
            "basic": '''
    // Create TTS pipeline
    let g2p = create_g2p(G2pBackend::English)?;
    let acoustic = create_acoustic(AcousticBackend::Vits)?;
    let vocoder = create_vocoder(VocoderBackend::HiFiGan)?;
    
    let pipeline = VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic(acoustic)
        .with_vocoder(vocoder)
        .build()?;
    
    // Synthesize text
    let audio = pipeline.synthesize(text).await?;
    
    // Save result
    std::fs::write("output.wav", audio)?;
    info!("Audio saved to output.wav");
''',
            "advanced": '''
    // Create advanced pipeline configuration
    let config = PipelineConfig::default()
        .with_quality(voirs_sdk::Quality::High)
        .with_optimization(true);
    
    let pipeline = VoirsPipelineBuilder::new()
        .with_config(config)
        .build()
        .await?;
    
    // Advanced synthesis with monitoring
    let start_time = Instant::now();
    let result = pipeline.synthesize_advanced("Advanced synthesis example").await?;
    let duration = start_time.elapsed();
    
    info!("Synthesis completed in {:.2}ms", duration.as_millis());
''',
            "integration": '''
    // Initialize platform integration
    let integration_config = {kwargs.get("integration_class", "Platform")}Config::default();
    let integration = {kwargs.get("integration_class", "Platform")}Integration::new(integration_config)?;
    
    // Connect to platform
    integration.connect().await?;
    
    // Perform integration-specific operations
    let result = integration.process_with_voirs("Integration test").await?;
    
    info!("Integration completed successfully");
''',
            "performance": '''
    // Initialize performance monitoring
    let monitor = ResourceMonitor::new();
    
    // Run performance test
    for i in 0..self.config.iterations {
        let start = Instant::now();
        
        // Perform synthesis operation
        let result = pipeline.synthesize(&format!("Performance test iteration {}", i)).await?;
        
        let duration = start.elapsed();
        debug!("Iteration {} completed in {:.2}ms", i, duration.as_millis());
    }
    
    info!("Performance test completed: {}", monitor.report());
'''
        }
        return skeletons.get(example_type, "    // TODO: Implement example logic\n    info!(\"Example implementation needed\");")
        
    def _generate_additional_functions(self, example_type: str, **kwargs) -> str:
        """Generate additional helper functions"""
        functions = []
        
        if example_type in ["advanced", "performance"]:
            functions.append(FUNCTION_TEMPLATES["error_handling"])
            functions.append(FUNCTION_TEMPLATES["configuration"])
            
        if example_type == "performance":
            functions.append(FUNCTION_TEMPLATES["monitoring"])
            
        return "\n".join(functions)
        
    def add_to_cargo_toml(self, name: str, cargo_entry: str) -> bool:
        """Add example entry to Cargo.toml"""
        try:
            with open(self.cargo_toml_path, 'r') as f:
                content = f.read()
                
            # Check if example already exists
            if f'name = "{name}"' in content:
                return False
                
            # Find the last [[example]] section and add after it
            lines = content.split('\n')
            insert_index = -1
            
            for i, line in enumerate(lines):
                if line.startswith('[[example]]'):
                    # Find the end of this example block
                    j = i + 1
                    while j < len(lines) and not lines[j].startswith('[') and lines[j].strip():
                        j += 1
                    insert_index = j
                    
            if insert_index == -1:
                # Add before [dependencies] section
                for i, line in enumerate(lines):
                    if line.startswith('[dependencies]'):
                        insert_index = i
                        break
                        
            if insert_index != -1:
                lines.insert(insert_index, cargo_entry.strip())
                
                with open(self.cargo_toml_path, 'w') as f:
                    f.write('\n'.join(lines))
                return True
                
        except Exception as e:
            print(f"Error updating Cargo.toml: {e}")
            
        return False
        
    def create_example_file(self, name: str, content: str) -> bool:
        """Create the example file"""
        file_path = self.examples_dir / f"{name}.rs"
        
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error creating example file: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Generate VoiRS examples")
    parser.add_argument("--type", choices=list(EXAMPLE_TEMPLATES.keys()), required=True,
                       help="Type of example to generate")
    parser.add_argument("--name", required=True,
                       help="Name of the example (snake_case)")
    parser.add_argument("--title", help="Human-readable title")
    parser.add_argument("--description", help="Brief description")
    parser.add_argument("--examples-dir", default=".",
                       help="Directory containing examples (default: current)")
    
    # Type-specific arguments
    parser.add_argument("--integration-type", help="Type of integration (e.g., 'Web', 'Mobile')")
    parser.add_argument("--integration-class", help="Class name for integration")
    parser.add_argument("--benchmark-type", help="Type of benchmark (e.g., 'Latency', 'Throughput')")
    
    parser.add_argument("--dry-run", action="store_true",
                       help="Print generated content without creating files")
    
    args = parser.parse_args()
    
    # Validate name format
    if not re.match(r'^[a-z][a-z0-9_]*$', args.name):
        print("Error: Example name must be snake_case")
        sys.exit(1)
        
    # Prepare kwargs for template generation
    kwargs = {}
    if args.integration_type:
        kwargs["integration_type"] = args.integration_type
        kwargs["integration_class"] = args.integration_class or args.integration_type.replace(" ", "")
        kwargs["integration_features"] = f"- {args.integration_type} API integration\n//! - Cross-platform compatibility\n//! - Error handling and recovery"
        kwargs["platform_support"] = f"- {args.integration_type} platform\n//! - Multiple API versions\n//! - Fallback mechanisms"
        kwargs["setup_instructions"] = f"1. Install {args.integration_type} SDK\n//! 2. Configure API credentials\n//! 3. Set up platform-specific dependencies"
        kwargs["integration_init"] = "// Initialize platform SDK\n        // Configure authentication\n        // Verify platform compatibility"
        kwargs["integration_process"] = "// Process input through platform\n        // Handle platform-specific formatting\n        // Return processed result"
        kwargs["usage_demo"] = "// Demonstrate platform integration\n    // Show error handling\n    // Display integration statistics"
        
    if args.benchmark_type:
        kwargs["benchmark_type"] = args.benchmark_type
        kwargs["benchmarking_capabilities"] = f"- {args.benchmark_type} measurement\n//! - Statistical analysis\n//! - Performance regression detection"
        kwargs["metrics_collected"] = f"- {args.benchmark_type} metrics\n//! - Resource utilization\n//! - Success rates"
        kwargs["benchmark_init"] = "// Initialize benchmark pipeline\n        // Configure measurement parameters\n        // Set up monitoring"
        kwargs["single_operation"] = f"// Perform single {args.benchmark_type.lower()} test\n        // Measure relevant metrics\n        // Record results"
        
    # Generate example
    generator = ExampleGenerator(args.examples_dir)
    
    try:
        content, cargo_entry = generator.generate_example(
            args.type, args.name, args.title, args.description, **kwargs
        )
        
        if args.dry_run:
            print("Generated example content:")
            print("=" * 50)
            print(content)
            print("\nCargo.toml entry:")
            print("=" * 20)
            print(cargo_entry)
        else:
            # Create files
            success = True
            
            if not generator.create_example_file(args.name, content):
                print(f"Failed to create example file: {args.name}.rs")
                success = False
                
            if not generator.add_to_cargo_toml(args.name, cargo_entry):
                print(f"Failed to update Cargo.toml (example may already exist)")
                success = False
                
            if success:
                print(f"✅ Successfully created example: {args.name}")
                print(f"📝 File: {args.name}.rs")
                print(f"📦 Updated: Cargo.toml")
                print(f"\n🚀 Run with: cargo run --example {args.name}")
            else:
                print("❌ Failed to create example")
                sys.exit(1)
                
    except Exception as e:
        print(f"Error generating example: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()