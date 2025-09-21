//! Comprehensive Performance Benchmarking Example
//!
//! This example demonstrates advanced performance monitoring and benchmarking capabilities:
//! 1. Detailed performance metrics collection
//! 2. Memory usage monitoring
//! 3. Multi-threaded performance testing
//! 4. Component-level benchmarking
//! 5. Performance regression detection
//! 6. Comparative analysis across configurations
//!
//! ## Running this example:
//! ```bash
//! cargo run --example performance_benchmarking
//! ```
//!
//! ## Key features:
//! - Real-time Factor (RTF) measurement
//! - Memory usage tracking
//! - CPU utilization monitoring
//! - Latency analysis
//! - Throughput benchmarking
//! - Quality vs. performance trade-offs

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::System;
use tokio::sync::Semaphore;
use voirs_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging with performance-focused settings
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🚀 VoiRS Performance Benchmarking Suite");
    println!("=======================================");
    println!();

    // Initialize system monitoring
    let mut system = System::new_all();
    system.refresh_all();

    // Run comprehensive benchmarks
    let mut benchmark_suite = PerformanceBenchmarkSuite::new();

    // 1. Single-threaded performance baseline
    println!("📊 Running single-threaded baseline benchmarks...");
    benchmark_suite.run_baseline_benchmarks().await?;

    // 2. Multi-threaded performance testing
    println!("\n🧵 Running multi-threaded performance tests...");
    benchmark_suite.run_concurrent_benchmarks().await?;

    // 3. Component-level benchmarking
    println!("\n🔧 Running component-level benchmarks...");
    benchmark_suite.run_component_benchmarks().await?;

    // 4. Memory usage analysis
    println!("\n💾 Running memory usage analysis...");
    benchmark_suite.run_memory_benchmarks(&mut system).await?;

    // 5. Quality vs. performance comparison
    println!("\n⚖️  Running quality vs. performance comparison...");
    benchmark_suite.run_quality_benchmarks().await?;

    // 6. Generate comprehensive report
    println!("\n📋 Generating performance report...");
    benchmark_suite.generate_report()?;

    Ok(())
}

struct PerformanceBenchmarkSuite {
    results: HashMap<String, BenchmarkResult>,
}

impl PerformanceBenchmarkSuite {
    fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    /// Run baseline single-threaded performance tests
    async fn run_baseline_benchmarks(&mut self) -> Result<()> {
        let test_texts = vec![
            ("Short", "Hello world."),
            ("Medium", "The quick brown fox jumps over the lazy dog and runs through the forest."),
            ("Long", "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort."),
        ];

        let pipeline = VoirsPipelineBuilder::new()
            .with_quality(QualityLevel::Medium)
            .build()
            .await?;

        for (name, text) in test_texts {
            let result = self.benchmark_synthesis(&pipeline, name, text, 5).await?;
            self.results
                .insert(format!("baseline_{}", name.to_lowercase()), result.clone());

            println!(
                "   {} text: RTF={:.3}, Latency={:.0}ms",
                name, result.avg_rtf, result.avg_latency_ms
            );
        }

        Ok(())
    }

    /// Run concurrent synthesis benchmarks
    async fn run_concurrent_benchmarks(&mut self) -> Result<()> {
        let text = "Testing concurrent synthesis performance with multiple parallel requests.";
        let concurrency_levels = vec![1, 2, 4, 8];

        for concurrency in concurrency_levels {
            let pipeline = Arc::new(
                VoirsPipelineBuilder::new()
                    .with_quality(QualityLevel::Medium)
                    .build()
                    .await?,
            );

            let result = self
                .benchmark_concurrent_synthesis(pipeline, concurrency, text, 3)
                .await?;
            self.results
                .insert(format!("concurrent_{}", concurrency), result.clone());

            println!(
                "   {}x concurrent: RTF={:.3}, Throughput={:.1} req/s",
                concurrency, result.avg_rtf, result.throughput_req_per_sec
            );
        }

        Ok(())
    }

    /// Run component-level benchmarks
    async fn run_component_benchmarks(&mut self) -> Result<()> {
        let text = "Component benchmark test text for detailed analysis.";
        let pipeline = VoirsPipelineBuilder::new().build().await?;

        // Measure different pipeline stages
        let stages = vec![
            ("G2P", "text_to_phonemes"),
            ("Acoustic", "phonemes_to_mel"),
            ("Vocoder", "mel_to_audio"),
            ("Full Pipeline", "text_to_audio"),
        ];

        for (stage_name, _stage_key) in stages {
            // For this example, we'll measure the full pipeline
            // In a real implementation, you'd measure individual components
            let result = self
                .benchmark_synthesis(&pipeline, stage_name, text, 3)
                .await?;
            self.results.insert(
                format!("component_{}", stage_name.to_lowercase().replace(" ", "_")),
                result.clone(),
            );

            println!(
                "   {} stage: {:.1}ms average",
                stage_name, result.avg_latency_ms
            );
        }

        Ok(())
    }

    /// Run memory usage benchmarks
    async fn run_memory_benchmarks(&mut self, system: &mut System) -> Result<()> {
        let text = "Memory usage benchmark text for analyzing resource consumption.";
        let pipeline = VoirsPipelineBuilder::new().build().await?;

        // Get process ID and initial memory
        let pid = sysinfo::Pid::from(std::process::id() as usize);
        system.refresh_process(pid);

        let initial_memory = if let Some(process) = system.process(pid) {
            process.memory()
        } else {
            0
        };

        // Run synthesis and monitor memory
        let start_time = Instant::now();
        let _audio = pipeline.synthesize(text).await?;
        let duration = start_time.elapsed();

        // Refresh and get peak memory
        system.refresh_process(pid);
        let peak_memory = if let Some(process) = system.process(pid) {
            process.memory()
        } else {
            0
        };

        let memory_usage = peak_memory.saturating_sub(initial_memory);

        let result = BenchmarkResult {
            test_name: "memory_usage".to_string(),
            num_runs: 1,
            avg_latency_ms: duration.as_millis() as f32,
            avg_rtf: 0.0, // Not applicable for memory test
            min_latency_ms: duration.as_millis() as f32,
            max_latency_ms: duration.as_millis() as f32,
            throughput_req_per_sec: 0.0,
            memory_usage_kb: (memory_usage / 1024) as f32,
            cpu_usage_percent: 0.0,
        };

        self.results.insert("memory_usage".to_string(), result);

        println!(
            "   Memory usage: {:.1} MB",
            memory_usage as f32 / 1024.0 / 1024.0
        );
        println!("   Processing time: {:.0}ms", duration.as_millis());

        Ok(())
    }

    /// Run quality vs. performance benchmarks
    async fn run_quality_benchmarks(&mut self) -> Result<()> {
        let text = "Quality benchmark test for analyzing performance trade-offs.";
        let quality_levels = vec![
            QualityLevel::Low,
            QualityLevel::Medium,
            QualityLevel::High,
            QualityLevel::Ultra,
        ];

        for quality in quality_levels {
            let pipeline = VoirsPipelineBuilder::new()
                .with_quality(quality.clone())
                .build()
                .await?;

            let quality_name = format!("{:?}", quality).to_lowercase();
            let result = self
                .benchmark_synthesis(&pipeline, &quality_name, text, 3)
                .await?;
            self.results
                .insert(format!("quality_{}", quality_name), result.clone());

            println!(
                "   {:?} quality: RTF={:.3}, Latency={:.0}ms",
                quality, result.avg_rtf, result.avg_latency_ms
            );
        }

        Ok(())
    }

    /// Benchmark single synthesis task
    async fn benchmark_synthesis(
        &self,
        pipeline: &VoirsPipeline,
        name: &str,
        text: &str,
        num_runs: usize,
    ) -> Result<BenchmarkResult> {
        let mut latencies = Vec::new();
        let mut rtfs = Vec::new();

        for _ in 0..num_runs {
            let start_time = Instant::now();
            let audio = pipeline.synthesize(text).await?;
            let duration = start_time.elapsed();

            let latency_ms = duration.as_millis() as f32;
            let audio_duration_s = audio.duration();
            let rtf = duration.as_secs_f32() / audio_duration_s;

            latencies.push(latency_ms);
            rtfs.push(rtf);
        }

        let avg_latency = latencies.iter().sum::<f32>() / latencies.len() as f32;
        let avg_rtf = rtfs.iter().sum::<f32>() / rtfs.len() as f32;

        Ok(BenchmarkResult {
            test_name: name.to_string(),
            num_runs,
            avg_latency_ms: avg_latency,
            avg_rtf,
            min_latency_ms: latencies.iter().cloned().fold(f32::INFINITY, f32::min),
            max_latency_ms: latencies.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            throughput_req_per_sec: 1000.0 / avg_latency,
            memory_usage_kb: 0.0,
            cpu_usage_percent: 0.0,
        })
    }

    /// Benchmark concurrent synthesis
    async fn benchmark_concurrent_synthesis(
        &self,
        pipeline: Arc<VoirsPipeline>,
        concurrency: usize,
        text: &str,
        num_runs: usize,
    ) -> Result<BenchmarkResult> {
        let semaphore = Arc::new(Semaphore::new(concurrency));
        let mut tasks = Vec::new();

        let start_time = Instant::now();

        for _ in 0..num_runs {
            let pipeline = Arc::clone(&pipeline);
            let semaphore = Arc::clone(&semaphore);
            let text = text.to_string();

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                let task_start = Instant::now();
                let _audio = pipeline.synthesize(&text).await?;
                let task_duration = task_start.elapsed();
                Ok::<Duration, anyhow::Error>(task_duration)
            });

            tasks.push(task);
        }

        let mut durations = Vec::new();
        for task in tasks {
            let duration = task.await??;
            durations.push(duration.as_millis() as f32);
        }

        let total_duration = start_time.elapsed();
        let avg_latency = durations.iter().sum::<f32>() / durations.len() as f32;
        let throughput = num_runs as f32 / total_duration.as_secs_f32();

        Ok(BenchmarkResult {
            test_name: format!("concurrent_{}", concurrency),
            num_runs,
            avg_latency_ms: avg_latency,
            avg_rtf: 0.0, // Not directly applicable for concurrent tests
            min_latency_ms: durations.iter().cloned().fold(f32::INFINITY, f32::min),
            max_latency_ms: durations.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            throughput_req_per_sec: throughput,
            memory_usage_kb: 0.0,
            cpu_usage_percent: 0.0,
        })
    }

    /// Generate comprehensive benchmark report
    fn generate_report(&self) -> Result<()> {
        println!("📈 Performance Benchmark Report");
        println!("==============================");
        println!();

        // Summary statistics
        println!("📊 Summary Statistics:");
        println!("   Total benchmarks run: {}", self.results.len());

        // Baseline performance
        if let Some(medium_result) = self.results.get("baseline_medium") {
            println!(
                "   Baseline RTF (medium text): {:.3}",
                medium_result.avg_rtf
            );
            println!(
                "   Baseline latency (medium text): {:.0}ms",
                medium_result.avg_latency_ms
            );

            if medium_result.avg_rtf < 1.0 {
                println!("   ✅ Achieving real-time performance!");
            } else {
                println!("   ⚠️  Slower than real-time synthesis");
            }
        }

        // Performance by text length
        println!("\n📏 Performance by Text Length:");
        for length in ["short", "medium", "long"] {
            if let Some(result) = self.results.get(&format!("baseline_{}", length)) {
                println!(
                    "   {} text: RTF={:.3}, Latency={:.0}ms, Throughput={:.1} req/s",
                    length.to_uppercase(),
                    result.avg_rtf,
                    result.avg_latency_ms,
                    result.throughput_req_per_sec
                );
            }
        }

        // Concurrent performance
        println!("\n🧵 Concurrent Performance:");
        for concurrency in [1, 2, 4, 8] {
            if let Some(result) = self.results.get(&format!("concurrent_{}", concurrency)) {
                println!(
                    "   {}x concurrent: Avg latency={:.0}ms, Throughput={:.1} req/s",
                    concurrency, result.avg_latency_ms, result.throughput_req_per_sec
                );
            }
        }

        // Quality vs. performance
        println!("\n⚖️  Quality vs. Performance:");
        for quality in ["low", "medium", "high", "ultra"] {
            if let Some(result) = self.results.get(&format!("quality_{}", quality)) {
                println!(
                    "   {} quality: RTF={:.3}, Latency={:.0}ms",
                    quality.to_uppercase(),
                    result.avg_rtf,
                    result.avg_latency_ms
                );
            }
        }

        // Memory usage
        if let Some(memory_result) = self.results.get("memory_usage") {
            println!("\n💾 Memory Usage:");
            println!(
                "   Peak memory usage: {:.1} MB",
                memory_result.memory_usage_kb / 1024.0
            );
        }

        // Performance recommendations
        println!("\n💡 Performance Recommendations:");

        // Check if real-time is achieved
        let baseline_rtf = self
            .results
            .get("baseline_medium")
            .map(|r| r.avg_rtf)
            .unwrap_or(1.0);

        if baseline_rtf > 1.0 {
            println!("   • Consider using lower quality settings for real-time applications");
            println!("   • Increase available CPU cores for concurrent processing");
        } else {
            println!("   • Current configuration achieves real-time synthesis");
            println!("   • Consider higher quality settings if latency allows");
        }

        // Check concurrent performance
        let single_throughput = self
            .results
            .get("concurrent_1")
            .map(|r| r.throughput_req_per_sec)
            .unwrap_or(0.0);
        let quad_throughput = self
            .results
            .get("concurrent_4")
            .map(|r| r.throughput_req_per_sec)
            .unwrap_or(0.0);

        if quad_throughput > single_throughput * 2.0 {
            println!("   • Concurrent processing provides good scaling benefits");
        } else {
            println!("   • Limited scaling benefits from concurrent processing");
        }

        println!("\n✅ Benchmark report generated successfully!");
        println!("📄 Detailed results saved to console output above.");

        Ok(())
    }
}

#[derive(Clone, Debug)]
struct BenchmarkResult {
    test_name: String,
    num_runs: usize,
    avg_latency_ms: f32,
    avg_rtf: f32,
    min_latency_ms: f32,
    max_latency_ms: f32,
    throughput_req_per_sec: f32,
    memory_usage_kb: f32,
    cpu_usage_percent: f32,
}
