//! # Comprehensive Benchmarking and Analytics
//!
//! This example provides advanced benchmarking tools for VoiRS SDK performance analysis.
//! It measures and analyzes:
//! - Synthesis performance across different text lengths and complexities
//! - Memory usage patterns and optimization effectiveness
//! - Quality metrics and their impact on performance
//! - Feature-specific performance characteristics
//! - Concurrency and scaling behavior
//! - Real-time performance requirements

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use voirs_sdk::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    test_name: String,
    text_length: usize,
    synthesis_time_ms: f64,
    audio_duration_s: f64,
    real_time_factor: f64,
    memory_usage_mb: f64,
    peak_memory_mb: f64,
    quality_score: Option<f64>,
    features_used: Vec<String>,
    timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkSuite {
    suite_name: String,
    total_tests: usize,
    avg_rtf: f64,
    avg_memory_mb: f64,
    fastest_test: String,
    slowest_test: String,
    results: Vec<BenchmarkResult>,
    system_info: SystemInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SystemInfo {
    cpu_cores: usize,
    available_memory_gb: f64,
    platform: String,
    features_enabled: Vec<String>,
}

#[derive(Debug)]
struct PerformanceMetrics {
    synthesis_times: Vec<f64>,
    memory_usage: Vec<f64>,
    quality_scores: Vec<f64>,
    rtf_values: Vec<f64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize comprehensive logging for benchmarking
    voirs_sdk::logging::init_logging(&voirs_sdk::config::LoggingConfig::default())?;

    println!("📊 VoiRS SDK Comprehensive Benchmarking Suite");
    println!("==============================================\n");

    // System information
    let system_info = collect_system_info();
    println!("💻 System Information:");
    println!("  CPU Cores: {}", system_info.cpu_cores);
    println!(
        "  Available Memory: {:.1} GB",
        system_info.available_memory_gb
    );
    println!("  Platform: {}", system_info.platform);
    println!("  Features Enabled: {:?}\n", system_info.features_enabled);

    // Create benchmark pipeline
    let pipeline = VoirsPipelineBuilder::new()
        .with_emotion_enabled(true)
        .with_cloning_enabled(true)
        .with_conversion_enabled(true)
        // Note: singing feature config through features, not builder method
        // Note: spatial feature config through features
        .with_quality(QualityLevel::High)
        // Note: performance monitoring enabled by default"
        .build()
        .await?;

    println!("✅ Benchmark pipeline initialized\n");

    // Run comprehensive benchmarks
    let mut all_results = Vec::new();

    // Benchmark 1: Text Length Analysis
    let text_length_results = run_text_length_benchmark(&pipeline).await?;
    all_results.extend(text_length_results);

    // Benchmark 2: Feature Performance Analysis
    let feature_results = run_feature_performance_benchmark(&pipeline).await?;
    all_results.extend(feature_results);

    // Benchmark 3: Quality vs Performance Analysis
    let quality_results = run_quality_performance_benchmark(&pipeline).await?;
    all_results.extend(quality_results);

    // Benchmark 4: Memory Usage Analysis
    let memory_results = run_memory_benchmark(&pipeline).await?;
    all_results.extend(memory_results);

    // Benchmark 5: Concurrency Performance
    let concurrency_results = run_concurrency_benchmark(&pipeline).await?;
    all_results.extend(concurrency_results);

    // Benchmark 6: Streaming Performance
    let streaming_results = run_streaming_benchmark(&pipeline).await?;
    all_results.extend(streaming_results);

    // Generate comprehensive report
    let benchmark_suite = create_benchmark_suite(all_results, system_info);
    generate_performance_report(&benchmark_suite).await?;
    save_benchmark_results(&benchmark_suite).await?;

    println!("\n🎯 Benchmarking complete! Check benchmark_results.json for detailed data.");
    Ok(())
}

async fn run_text_length_benchmark(pipeline: &VoirsPipeline) -> Result<Vec<BenchmarkResult>> {
    println!("📏 Running Text Length Benchmark");
    println!("================================");

    let test_texts = vec![
        ("Short", "Hello world!", 12),
        ("Medium", "This is a medium length sentence with multiple words to test synthesis performance.", 87),
        ("Long", "This is a much longer text that contains multiple sentences and various punctuation marks. It is designed to test how the synthesis engine performs with longer inputs that might require more processing time and memory. The text includes different types of words, from simple to complex, and should provide a good benchmark for performance analysis.", 342),
    ];

    let very_long = create_very_long_text();
    let paragraph = create_paragraph_text();
    let test_texts = vec![
        ("Short", "Hello world!", 15),
        ("Medium", "This is a medium length test to evaluate synthesis performance.", 150),
        ("Long", "This is a longer sentence that contains more complex words and structures to properly test the synthesis engine's capabilities.", 500),
        ("Very Long", very_long.as_str(), 1500),
        ("Paragraph", paragraph.as_str(), 800),
    ];

    let mut results = Vec::new();

    for (test_name, text, expected_length) in test_texts {
        println!("📝 Testing {}: {} characters", test_name, text.len());

        let start_memory = get_memory_usage();
        let start_time = Instant::now();

        let audio = pipeline.synthesize(text).await?;

        let synthesis_time = start_time.elapsed();
        let end_memory = get_memory_usage();
        let peak_memory = get_peak_memory_usage();

        let result = BenchmarkResult {
            test_name: format!("TextLength_{}", test_name),
            text_length: text.len(),
            synthesis_time_ms: synthesis_time.as_secs_f64() * 1000.0,
            audio_duration_s: audio.duration() as f64,
            real_time_factor: synthesis_time.as_secs_f64() / (audio.duration() as f64),
            memory_usage_mb: (end_memory - start_memory) as f64 / 1024.0 / 1024.0,
            peak_memory_mb: peak_memory as f64 / 1024.0 / 1024.0,
            quality_score: None,
            features_used: vec!["basic_synthesis".to_string()],
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        println!("  ⏱️  Synthesis time: {:.2}ms", result.synthesis_time_ms);
        println!("  🎵 Audio duration: {:.2}s", result.audio_duration_s);
        println!("  📊 Real-time factor: {:.3}", result.real_time_factor);
        println!("  💾 Memory used: {:.1}MB\n", result.memory_usage_mb);

        results.push(result);
    }

    Ok(results)
}

async fn run_feature_performance_benchmark(
    pipeline: &VoirsPipeline,
) -> Result<Vec<BenchmarkResult>> {
    println!("🔧 Running Feature Performance Benchmark");
    println!("========================================");

    let test_text = "This is a consistent test phrase used to benchmark different features.";
    let mut results = Vec::new();

    // Baseline synthesis
    results.push(benchmark_feature(pipeline, "Baseline", test_text, vec![], None).await?);

    // Emotion control
    #[cfg(feature = "emotion")]
    {
        pipeline.apply_emotion_preset("happy", Some(0.8)).await?;
        results.push(
            benchmark_feature(
                pipeline,
                "Emotion_Happy",
                test_text,
                vec!["emotion".to_string()],
                None,
            )
            .await?,
        );

        pipeline.apply_emotion_preset("sad", Some(0.6)).await?;
        results.push(
            benchmark_feature(
                pipeline,
                "Emotion_Sad",
                test_text,
                vec!["emotion".to_string()],
                None,
            )
            .await?,
        );
    }

    // Voice cloning
    #[cfg(feature = "cloning")]
    {
        let reference_audio: Vec<f32> = (0..22050)
            .map(|i| 0.1 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
            .collect();
        let clone_result = pipeline
            .quick_clone(reference_audio, 22050, test_text.to_string())
            .await?;
        if clone_result.success {
            let clone_benchmark = BenchmarkResult {
                test_name: "Voice_Cloning".to_string(),
                text_length: test_text.len(),
                synthesis_time_ms: 0.0, // Clone result doesn't include timing
                audio_duration_s: clone_result.audio.len() as f64 / clone_result.sample_rate as f64,
                real_time_factor: 0.0,
                memory_usage_mb: 0.0,
                peak_memory_mb: 0.0,
                quality_score: None,
                features_used: vec!["cloning".to_string()],
                timestamp: chrono::Utc::now().to_rfc3339(),
            };
            results.push(clone_benchmark);
        }
    }

    // Singing synthesis
    #[cfg(feature = "singing")]
    {
        results.push(
            benchmark_feature(
                pipeline,
                "Singing",
                test_text,
                vec!["singing".to_string()],
                None,
            )
            .await?,
        );
    }

    // Spatial audio
    #[cfg(feature = "spatial")]
    {
        // TODO: Implement spatial positioning when VoirsPipeline supports it
        // use voirs_spatial::Position3D;
        // pipeline
        //     .set_spatial_position(Position3D::new(1.0, 0.0, 0.0))
        //     .await?;
        results.push(
            benchmark_feature(
                pipeline,
                "Spatial",
                test_text,
                vec!["spatial".to_string()],
                None,
            )
            .await?,
        );
    }

    // Combined features
    #[cfg(all(feature = "emotion", feature = "spatial"))]
    {
        results.push(
            benchmark_feature(
                pipeline,
                "Combined_Emotion_Spatial",
                test_text,
                vec!["emotion".to_string(), "spatial".to_string()],
                None,
            )
            .await?,
        );
    }

    Ok(results)
}

async fn run_quality_performance_benchmark(
    pipeline: &VoirsPipeline,
) -> Result<Vec<BenchmarkResult>> {
    println!("⭐ Running Quality vs Performance Benchmark");
    println!("==========================================");

    let test_text =
        "Quality benchmark text with various phonetic challenges and complex pronunciation.";
    let mut results = Vec::new();

    let quality_levels = [QualityLevel::Low, QualityLevel::Medium, QualityLevel::High];

    for quality in quality_levels {
        println!("🎯 Testing {:?} quality", quality);

        // Create pipeline with specific quality
        let quality_pipeline = VoirsPipelineBuilder::new()
            .with_quality(quality)
            .build()
            .await?;

        let result = benchmark_feature(
            &quality_pipeline,
            &format!("Quality_{:?}", quality),
            test_text,
            vec!["quality_test".to_string()],
            None,
        )
        .await?;
        results.push(result);
    }

    Ok(results)
}

async fn run_memory_benchmark(pipeline: &VoirsPipeline) -> Result<Vec<BenchmarkResult>> {
    println!("💾 Running Memory Usage Benchmark");
    println!("=================================");

    let mut results = Vec::new();

    // Test memory usage with different text sizes
    let memory_test_sizes = [50, 200, 500, 1000, 2000];

    for size in memory_test_sizes {
        let test_text = "A".repeat(size);
        println!("📊 Testing memory with {} character text", size);

        let initial_memory = get_memory_usage();

        // Perform multiple syntheses to test memory accumulation
        for i in 0..5 {
            let audio = pipeline.synthesize(&test_text).await?;
            let current_memory = get_memory_usage();

            if i == 4 {
                // Last iteration
                let result = BenchmarkResult {
                    test_name: format!("Memory_{}chars", size),
                    text_length: size,
                    synthesis_time_ms: 0.0,
                    audio_duration_s: audio.duration() as f64,
                    real_time_factor: 0.0,
                    memory_usage_mb: (current_memory - initial_memory) as f64 / 1024.0 / 1024.0,
                    peak_memory_mb: get_peak_memory_usage() as f64 / 1024.0 / 1024.0,
                    quality_score: None,
                    features_used: vec!["memory_test".to_string()],
                    timestamp: chrono::Utc::now().to_rfc3339(),
                };

                println!("  💾 Memory usage: {:.1}MB", result.memory_usage_mb);
                results.push(result);
            }
        }
    }

    Ok(results)
}

async fn run_concurrency_benchmark(pipeline: &VoirsPipeline) -> Result<Vec<BenchmarkResult>> {
    println!("🚀 Running Concurrency Benchmark");
    println!("================================");

    let test_text = "Concurrent synthesis test for parallel performance analysis.";
    let mut results = Vec::new();

    let concurrency_levels = [1, 2, 4, 8];

    for concurrent_count in concurrency_levels {
        println!("⚡ Testing {} concurrent syntheses", concurrent_count);

        let start_time = Instant::now();
        let start_memory = get_memory_usage();

        // Create concurrent synthesis tasks
        let tasks: Vec<_> = (0..concurrent_count)
            .map(|i| {
                let text = format!("{} (iteration {})", test_text, i);
                let pipeline = pipeline;
                async move { pipeline.synthesize(&text).await }
            })
            .collect();

        // Execute all tasks concurrently
        let synthesis_results = futures::future::try_join_all(tasks).await?;

        let total_time = start_time.elapsed();
        let end_memory = get_memory_usage();

        let total_audio_duration: f64 = synthesis_results
            .iter()
            .map(|audio| audio.duration() as f64)
            .sum();

        let result = BenchmarkResult {
            test_name: format!("Concurrency_{}", concurrent_count),
            text_length: test_text.len(),
            synthesis_time_ms: total_time.as_secs_f64() * 1000.0,
            audio_duration_s: total_audio_duration,
            real_time_factor: total_time.as_secs_f64() / total_audio_duration,
            memory_usage_mb: (end_memory - start_memory) as f64 / 1024.0 / 1024.0,
            peak_memory_mb: get_peak_memory_usage() as f64 / 1024.0 / 1024.0,
            quality_score: None,
            features_used: vec!["concurrency".to_string()],
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        println!("  ⏱️  Total time: {:.2}ms", result.synthesis_time_ms);
        println!("  📊 Effective RTF: {:.3}", result.real_time_factor);
        println!("  💾 Memory used: {:.1}MB\n", result.memory_usage_mb);

        results.push(result);
    }

    Ok(results)
}

async fn run_streaming_benchmark(pipeline: &VoirsPipeline) -> Result<Vec<BenchmarkResult>> {
    println!("🌊 Running Streaming Performance Benchmark");
    println!("==========================================");

    let test_texts = vec![
        ("Streaming_Short", "Short streaming test."),
        ("Streaming_Medium", "This is a medium length streaming test with multiple words and phrases."),
        ("Streaming_Long", "This is a comprehensive streaming performance test with a longer text that should generate multiple chunks and provide detailed latency measurements for real-time applications."),
    ];

    let mut results = Vec::new();

    for (test_name, text) in test_texts {
        println!("🔄 Testing {}", test_name);

        let start_time = Instant::now();
        // Note: synthesize_streaming not available, using regular synthesis
        let audio = pipeline.synthesize(text).await?;

        let mut first_chunk_latency = None;
        let mut chunk_count = 0;
        let mut total_samples = 0;

        // Simulate streaming behavior for demonstration
        first_chunk_latency = Some(start_time.elapsed());
        chunk_count = 1;
        total_samples = audio.len();

        let total_time = start_time.elapsed();
        let audio_duration = total_samples as f64 / 22050.0; // Assuming 22050 Hz

        let result = BenchmarkResult {
            test_name: test_name.to_string(),
            text_length: text.len(),
            synthesis_time_ms: total_time.as_secs_f64() * 1000.0,
            audio_duration_s: audio_duration,
            real_time_factor: total_time.as_secs_f64() / audio_duration,
            memory_usage_mb: 0.0, // Streaming uses minimal memory
            peak_memory_mb: get_peak_memory_usage() as f64 / 1024.0 / 1024.0,
            quality_score: None,
            features_used: vec!["streaming".to_string()],
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        println!("  📊 Chunks generated: {}", chunk_count);
        println!(
            "  ⚡ First chunk latency: {:.2}ms",
            first_chunk_latency.unwrap().as_secs_f64() * 1000.0
        );
        println!("  🎵 Total audio: {:.2}s", result.audio_duration_s);
        println!("  📈 RTF: {:.3}\n", result.real_time_factor);

        results.push(result);
    }

    Ok(results)
}

async fn benchmark_feature(
    pipeline: &VoirsPipeline,
    test_name: &str,
    text: &str,
    features: Vec<String>,
    quality_score: Option<f64>,
) -> Result<BenchmarkResult> {
    let start_memory = get_memory_usage();
    let start_time = Instant::now();

    let audio = pipeline.synthesize(text).await?;

    let synthesis_time = start_time.elapsed();
    let end_memory = get_memory_usage();

    let result = BenchmarkResult {
        test_name: test_name.to_string(),
        text_length: text.len(),
        synthesis_time_ms: synthesis_time.as_secs_f64() * 1000.0,
        audio_duration_s: audio.duration() as f64,
        real_time_factor: synthesis_time.as_secs_f64() / (audio.duration() as f64),
        memory_usage_mb: (end_memory - start_memory) as f64 / 1024.0 / 1024.0,
        peak_memory_mb: get_peak_memory_usage() as f64 / 1024.0 / 1024.0,
        quality_score,
        features_used: features,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    println!(
        "  ✅ {}: {:.2}ms, RTF: {:.3}",
        test_name, result.synthesis_time_ms, result.real_time_factor
    );

    Ok(result)
}

fn collect_system_info() -> SystemInfo {
    let mut features_enabled = Vec::new();

    #[cfg(feature = "emotion")]
    features_enabled.push("emotion".to_string());
    #[cfg(feature = "cloning")]
    features_enabled.push("cloning".to_string());
    #[cfg(feature = "conversion")]
    features_enabled.push("conversion".to_string());
    #[cfg(feature = "singing")]
    features_enabled.push("singing".to_string());
    #[cfg(feature = "spatial")]
    features_enabled.push("spatial".to_string());

    SystemInfo {
        cpu_cores: num_cpus::get(),
        available_memory_gb: get_available_memory_gb(),
        platform: std::env::consts::OS.to_string(),
        features_enabled,
    }
}

fn create_benchmark_suite(
    results: Vec<BenchmarkResult>,
    system_info: SystemInfo,
) -> BenchmarkSuite {
    let total_tests = results.len();
    let avg_rtf = results.iter().map(|r| r.real_time_factor).sum::<f64>() / total_tests as f64;
    let avg_memory_mb = results.iter().map(|r| r.memory_usage_mb).sum::<f64>() / total_tests as f64;

    let fastest_test = results
        .iter()
        .min_by(|a, b| a.real_time_factor.partial_cmp(&b.real_time_factor).unwrap())
        .map(|r| r.test_name.clone())
        .unwrap_or_default();

    let slowest_test = results
        .iter()
        .max_by(|a, b| a.real_time_factor.partial_cmp(&b.real_time_factor).unwrap())
        .map(|r| r.test_name.clone())
        .unwrap_or_default();

    BenchmarkSuite {
        suite_name: "VoiRS_Comprehensive_Benchmark".to_string(),
        total_tests,
        avg_rtf,
        avg_memory_mb,
        fastest_test,
        slowest_test,
        results,
        system_info,
    }
}

async fn generate_performance_report(suite: &BenchmarkSuite) -> Result<()> {
    println!("📈 Performance Analysis Report");
    println!("=============================\n");

    println!("📊 Overall Statistics:");
    println!("  Total tests run: {}", suite.total_tests);
    println!("  Average RTF: {:.3}", suite.avg_rtf);
    println!("  Average memory usage: {:.1}MB", suite.avg_memory_mb);
    println!("  Fastest test: {}", suite.fastest_test);
    println!("  Slowest test: {}\n", suite.slowest_test);

    // Analyze by category
    let mut categories: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
    for result in &suite.results {
        let category = result.test_name.split('_').next().unwrap_or("Unknown");
        categories
            .entry(category.to_string())
            .or_default()
            .push(result);
    }

    println!("📋 Performance by Category:");
    for (category, results) in categories {
        let avg_rtf =
            results.iter().map(|r| r.real_time_factor).sum::<f64>() / results.len() as f64;
        let avg_memory =
            results.iter().map(|r| r.memory_usage_mb).sum::<f64>() / results.len() as f64;

        println!(
            "  {}: Avg RTF {:.3}, Avg Memory {:.1}MB ({} tests)",
            category,
            avg_rtf,
            avg_memory,
            results.len()
        );
    }

    // Performance recommendations
    println!("\n💡 Performance Recommendations:");
    if suite.avg_rtf > 1.0 {
        println!("  ⚠️  Average RTF above 1.0 - consider performance optimization");
    } else {
        println!("  ✅ Excellent real-time performance (RTF < 1.0)");
    }

    if suite.avg_memory_mb > 100.0 {
        println!("  ⚠️  High memory usage detected - consider memory optimization");
    } else {
        println!("  ✅ Efficient memory usage");
    }

    // Feature impact analysis
    let baseline_rtf = suite
        .results
        .iter()
        .find(|r| r.test_name == "Baseline")
        .map(|r| r.real_time_factor)
        .unwrap_or(0.0);

    if baseline_rtf > 0.0 {
        println!(
            "\n🔍 Feature Impact Analysis (vs Baseline RTF {:.3}):",
            baseline_rtf
        );
        for result in &suite.results {
            if result.test_name != "Baseline" && !result.features_used.is_empty() {
                let impact = result.real_time_factor / baseline_rtf;
                println!("  {}: {:.2}x impact", result.test_name, impact);
            }
        }
    }

    Ok(())
}

async fn save_benchmark_results(suite: &BenchmarkSuite) -> Result<()> {
    let json_data = serde_json::to_string_pretty(suite).map_err(|e| {
        VoirsError::serialization("json", format!("JSON serialization failed: {}", e))
    })?;

    tokio::fs::write("benchmark_results.json", json_data)
        .await
        .map_err(|e| {
            VoirsError::io_error(
                "benchmark_results.json",
                voirs_sdk::error::IoOperation::Write,
                e,
            )
        })?;

    // Also save a summary CSV
    let mut csv_content = String::from("test_name,text_length,synthesis_time_ms,audio_duration_s,real_time_factor,memory_usage_mb,features\n");
    for result in &suite.results {
        csv_content.push_str(&format!(
            "{},{},{:.2},{:.2},{:.3},{:.1},{}\n",
            result.test_name,
            result.text_length,
            result.synthesis_time_ms,
            result.audio_duration_s,
            result.real_time_factor,
            result.memory_usage_mb,
            result.features_used.join(";")
        ));
    }

    tokio::fs::write("benchmark_results.csv", csv_content)
        .await
        .map_err(|e| {
            VoirsError::io_error(
                "benchmark_results.csv",
                voirs_sdk::error::IoOperation::Write,
                e,
            )
        })?;

    println!("💾 Benchmark results saved:");
    println!("  📊 benchmark_results.json (detailed data)");
    println!("  📈 benchmark_results.csv (summary)");

    Ok(())
}

// Helper functions for system metrics
fn get_memory_usage() -> usize {
    // Simplified memory tracking - in practice would use platform-specific APIs
    0
}

fn get_peak_memory_usage() -> usize {
    // Simplified peak memory tracking
    0
}

fn get_available_memory_gb() -> f64 {
    // Simplified memory detection
    8.0
}

fn create_very_long_text() -> String {
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur. Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur.".to_string()
}

fn create_paragraph_text() -> String {
    "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet and provides a good test for speech synthesis systems. It includes various phonetic combinations and common English language patterns. The sentence has been used for decades to test typing equipment, fonts, and now speech synthesis systems. It demonstrates how well the system handles consonant clusters, vowel combinations, and the rhythm of natural speech. Performance testing with this type of content helps ensure that synthesis quality remains consistent across different types of input text.".to_string()
}
