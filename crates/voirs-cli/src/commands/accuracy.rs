//! Accuracy benchmarking command for VoiRS CLI.
//!
//! This module provides CLI commands for running comprehensive accuracy benchmarks
//! including CMU English phoneme tests, JVS Japanese mora tests, and Common Voice
//! multilingual evaluations.

use clap::{Args, Subcommand};
use std::path::PathBuf;
use tokio::time::Instant;

#[cfg(not(doctest))]
use async_trait::async_trait;

#[cfg(not(doctest))]
use voirs_evaluation::accuracy_benchmarks::{
    AccuracyBenchmarkConfig, AccuracyBenchmarkRunner, DatasetConfig, DatasetType, LanguageCode,
};

/// Accuracy benchmarking commands
#[derive(Debug, Clone, Args)]
pub struct AccuracyCommand {
    #[command(subcommand)]
    pub command: AccuracySubcommand,
}

/// Accuracy benchmarking subcommands
#[derive(Debug, Clone, Subcommand)]
pub enum AccuracySubcommand {
    /// Run comprehensive accuracy benchmarks
    Run(RunAccuracyArgs),
    /// Run specific dataset benchmark
    Dataset(DatasetAccuracyArgs),
    /// List available test datasets
    List(ListDatasetsArgs),
    /// Generate accuracy benchmark report
    Report(ReportArgs),
}

/// Arguments for running comprehensive accuracy benchmarks
#[derive(Debug, Clone, Args)]
pub struct RunAccuracyArgs {
    /// Output directory for benchmark results
    #[arg(short, long, default_value = "/tmp/voirs_accuracy_benchmarks")]
    pub output_dir: PathBuf,

    /// Enable detailed per-case reporting
    #[arg(long, default_value = "true")]
    pub detailed: bool,

    /// Maximum processing time per sample (seconds)
    #[arg(long, default_value = "10.0")]
    pub max_time: f64,

    /// Include only specific languages (comma-separated)
    #[arg(long)]
    pub languages: Option<String>,

    /// Custom dataset file path
    #[arg(long)]
    pub custom_dataset: Option<PathBuf>,

    /// Maximum samples per dataset (for faster testing)
    #[arg(long)]
    pub max_samples: Option<usize>,
}

/// Arguments for running specific dataset benchmarks
#[derive(Debug, Clone, Args)]
pub struct DatasetAccuracyArgs {
    /// Dataset type to benchmark
    #[arg(value_enum)]
    pub dataset: DatasetTypeArg,

    /// Language for the dataset
    #[arg(short, long, value_enum)]
    pub language: LanguageCodeArg,

    /// Custom data file path
    #[arg(short, long)]
    pub data_path: Option<PathBuf>,

    /// Target accuracy threshold
    #[arg(short, long)]
    pub target_accuracy: Option<f64>,

    /// Maximum number of test samples
    #[arg(short, long)]
    pub max_samples: Option<usize>,

    /// Output file for results
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Arguments for listing available datasets
#[derive(Debug, Clone, Args)]
pub struct ListDatasetsArgs {
    /// Show detailed information about each dataset
    #[arg(long)]
    pub detailed: bool,

    /// Filter by language
    #[arg(short, long, value_enum)]
    pub language: Option<LanguageCodeArg>,
}

/// Arguments for generating accuracy reports
#[derive(Debug, Clone, Args)]
pub struct ReportArgs {
    /// Input benchmark results file
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output format (json, txt, html)
    #[arg(short, long, default_value = "txt")]
    pub format: String,

    /// Output file path
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Dataset type argument for CLI
#[derive(Debug, Clone, clap::ValueEnum)]
pub enum DatasetTypeArg {
    Cmu,
    Jvs,
    CommonVoice,
    Custom,
}

#[cfg(not(doctest))]
impl From<DatasetTypeArg> for DatasetType {
    fn from(arg: DatasetTypeArg) -> Self {
        match arg {
            DatasetTypeArg::Cmu => DatasetType::CMU,
            DatasetTypeArg::Jvs => DatasetType::JVS,
            DatasetTypeArg::CommonVoice => DatasetType::CommonVoice,
            DatasetTypeArg::Custom => DatasetType::Custom,
        }
    }
}

/// Language code argument for CLI
#[derive(Debug, Clone, clap::ValueEnum)]
pub enum LanguageCodeArg {
    EnUs,
    Ja,
    Es,
    Fr,
    De,
    ZhCn,
}

#[cfg(not(doctest))]
impl From<LanguageCodeArg> for LanguageCode {
    fn from(arg: LanguageCodeArg) -> Self {
        match arg {
            LanguageCodeArg::EnUs => LanguageCode::EnUs,
            LanguageCodeArg::Ja => LanguageCode::Ja,
            LanguageCodeArg::Es => LanguageCode::Es,
            LanguageCodeArg::Fr => LanguageCode::Fr,
            LanguageCodeArg::De => LanguageCode::De,
            LanguageCodeArg::ZhCn => LanguageCode::ZhCn,
        }
    }
}

/// Execute accuracy benchmarking commands
#[cfg(not(doctest))]
pub async fn execute_accuracy_command(
    args: AccuracyCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        AccuracySubcommand::Run(run_args) => run_comprehensive_benchmarks(run_args).await,
        AccuracySubcommand::Dataset(dataset_args) => run_dataset_benchmark(dataset_args).await,
        AccuracySubcommand::List(list_args) => list_available_datasets(list_args).await,
        AccuracySubcommand::Report(report_args) => generate_accuracy_report(report_args).await,
    }
}

/// Stub implementation for doctests
#[cfg(doctest)]
pub async fn execute_accuracy_command(
    _args: AccuracyCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

/// Run comprehensive accuracy benchmarks
#[cfg(not(doctest))]
async fn run_comprehensive_benchmarks(
    args: RunAccuracyArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 VoiRS Comprehensive Accuracy Benchmarks");
    println!("==========================================\n");

    let start_time = Instant::now();

    // Configure benchmark
    let mut config = AccuracyBenchmarkConfig::default();
    config.output_dir = args.output_dir.to_string_lossy().to_string();
    config.detailed_reporting = args.detailed;
    config.max_processing_time = args.max_time;

    // Filter datasets by language if specified
    if let Some(languages_str) = &args.languages {
        let requested_languages: Vec<LanguageCode> = languages_str
            .split(',')
            .filter_map(|lang| match lang.trim() {
                "en-US" | "en" => Some(LanguageCode::EnUs),
                "ja" => Some(LanguageCode::Ja),
                "es" => Some(LanguageCode::Es),
                "fr" => Some(LanguageCode::Fr),
                "de" => Some(LanguageCode::De),
                "zh-CN" | "zh" => Some(LanguageCode::ZhCn),
                _ => None,
            })
            .collect();

        config
            .datasets
            .retain(|dataset| requested_languages.contains(&dataset.language));

        println!(
            "📊 Running benchmarks for languages: {:?}",
            requested_languages
        );
    }

    // Add custom dataset if specified
    if let Some(custom_path) = &args.custom_dataset {
        let custom_config = DatasetConfig {
            name: "Custom_Dataset".to_string(),
            dataset_type: DatasetType::Custom,
            language: LanguageCode::EnUs, // Default, will be parsed from file
            data_path: custom_path.to_string_lossy().to_string(),
            target_accuracy: 0.90,
            max_samples: args.max_samples,
        };
        config.datasets.push(custom_config);
        println!("📁 Added custom dataset: {}", custom_path.display());
    }

    // Override max samples if specified
    if let Some(max_samples) = args.max_samples {
        for dataset in &mut config.datasets {
            dataset.max_samples = Some(max_samples);
        }
        println!("📏 Limited to {} samples per dataset", max_samples);
    }

    // Create and run benchmark runner
    let mut runner = AccuracyBenchmarkRunner::new(config);

    println!("🔄 Loading test cases...");
    runner.load_test_cases().await?;

    println!("🚀 Running accuracy benchmarks...");

    // Note: In a real implementation, you would pass actual G2P/TTS/ASR systems here
    // For now, we'll use the simulation mode built into the benchmark runner
    let results = runner
        .run_benchmarks(
            None::<&DummyG2pSystem>,
            None::<&DummyTtsSystem>,
            None::<&DummyAsrSystem>,
        )
        .await?;

    let total_time = start_time.elapsed();

    // Display results summary
    println!(
        "\n✅ Benchmark completed in {:.2} seconds",
        total_time.as_secs_f64()
    );
    println!("\n📊 ACCURACY BENCHMARK RESULTS");
    println!("{}", "=".repeat(50));

    println!("\nOverall Metrics:");
    println!(
        "  • Total test cases: {}",
        results.overall_metrics.total_cases
    );
    println!(
        "  • Overall phoneme accuracy: {:.2}%",
        results.overall_metrics.overall_phoneme_accuracy * 100.0
    );
    println!(
        "  • Overall word accuracy: {:.2}%",
        results.overall_metrics.overall_word_accuracy * 100.0
    );
    println!(
        "  • Targets met: {}/{} ({:.1}%)",
        results.overall_metrics.targets_met,
        results.overall_metrics.total_targets,
        results.overall_metrics.pass_rate
    );

    println!("\nLanguage-Specific Results:");
    for (language, accuracy) in &results.overall_metrics.language_accuracies {
        println!("  • {:?}: {:.2}%", language, accuracy * 100.0);
    }

    println!("\nDataset Results:");
    for (dataset_name, dataset_result) in &results.dataset_results {
        let status = if dataset_result.target_met {
            "✅"
        } else {
            "❌"
        };
        println!(
            "  {} {}: {:.2}% ({:.1}% target)",
            status,
            dataset_name,
            dataset_result.phoneme_accuracy * 100.0,
            dataset_result.target_accuracy * 100.0
        );
    }

    println!("\nPerformance Statistics:");
    println!(
        "  • Average processing time: {:.2} ms",
        results.performance_stats.avg_processing_time_ms
    );
    println!(
        "  • Throughput: {:.1} cases/sec",
        results.performance_stats.throughput_cases_per_sec
    );
    println!(
        "  • Peak memory usage: {:.1} MB",
        results.performance_stats.peak_memory_mb
    );

    println!("\nResults saved to: {}", args.output_dir.display());

    // Exit with appropriate code
    if results.overall_metrics.pass_rate >= 80.0 {
        println!("\n🎉 All accuracy targets achieved!");
        std::process::exit(0);
    } else {
        println!("\n⚠️  Some accuracy targets not met. See detailed report for recommendations.");
        std::process::exit(1);
    }
}

/// Run benchmark for specific dataset
#[cfg(not(doctest))]
async fn run_dataset_benchmark(
    args: DatasetAccuracyArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "🎯 Running {} Dataset Benchmark",
        format!("{:?}", args.dataset)
    );
    println!("{}", "=".repeat(40));

    let dataset_config = DatasetConfig {
        name: format!("{:?}_Benchmark", args.dataset),
        dataset_type: args.dataset.into(),
        language: args.language.into(),
        data_path: args
            .data_path
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "tests/datasets/default.txt".to_string()),
        target_accuracy: args.target_accuracy.unwrap_or(0.90),
        max_samples: args.max_samples,
    };

    let config = AccuracyBenchmarkConfig {
        datasets: vec![dataset_config],
        detailed_reporting: true,
        ..Default::default()
    };

    let mut runner = AccuracyBenchmarkRunner::new(config);
    runner.load_test_cases().await?;

    let results = runner
        .run_benchmarks(
            None::<&DummyG2pSystem>,
            None::<&DummyTtsSystem>,
            None::<&DummyAsrSystem>,
        )
        .await?;

    // Display results
    for (dataset_name, dataset_result) in &results.dataset_results {
        println!("\nDataset: {}", dataset_name);
        println!("Language: {:?}", dataset_result.language);
        println!("Test cases: {}", dataset_result.total_cases);
        println!(
            "Phoneme accuracy: {:.2}%",
            dataset_result.phoneme_accuracy * 100.0
        );
        println!(
            "Word accuracy: {:.2}%",
            dataset_result.word_accuracy * 100.0
        );
        println!("Target: {:.1}%", dataset_result.target_accuracy * 100.0);
        println!(
            "Result: {}",
            if dataset_result.target_met {
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
        );
    }

    Ok(())
}

/// List available datasets
#[cfg(not(doctest))]
async fn list_available_datasets(args: ListDatasetsArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Available Accuracy Test Datasets");
    println!("{}", "=".repeat(40));

    let datasets = vec![
        (
            "CMU English Phoneme Test",
            DatasetType::CMU,
            LanguageCode::EnUs,
            0.95,
            "English phoneme accuracy using CMU pronunciation dictionary",
        ),
        (
            "JVS Japanese Mora Test",
            DatasetType::JVS,
            LanguageCode::Ja,
            0.90,
            "Japanese mora accuracy using JVS speech corpus",
        ),
        (
            "Common Voice Spanish",
            DatasetType::CommonVoice,
            LanguageCode::Es,
            0.88,
            "Spanish pronunciation from Mozilla Common Voice",
        ),
        (
            "Common Voice French",
            DatasetType::CommonVoice,
            LanguageCode::Fr,
            0.88,
            "French pronunciation from Mozilla Common Voice",
        ),
        (
            "Common Voice German",
            DatasetType::CommonVoice,
            LanguageCode::De,
            0.88,
            "German pronunciation from Mozilla Common Voice",
        ),
        (
            "Common Voice Chinese",
            DatasetType::CommonVoice,
            LanguageCode::ZhCn,
            0.85,
            "Mandarin Chinese from Mozilla Common Voice",
        ),
    ];

    for (name, dataset_type, language, target, description) in datasets {
        // Filter by language if specified
        if let Some(filter_lang) = &args.language {
            let filter_lang_code: LanguageCode = filter_lang.clone().into();
            if language != filter_lang_code {
                continue;
            }
        }

        println!("\n📊 {}", name);
        println!("   Type: {:?}", dataset_type);
        println!("   Language: {:?}", language);
        println!("   Target accuracy: {:.1}%", target * 100.0);

        if args.detailed {
            println!("   Description: {}", description);
            println!("   Status: Available");
        }
    }

    println!("\nTo run a specific dataset:");
    println!("  voirs-cli accuracy dataset <dataset_type> --language <lang>");
    println!("\nTo run all datasets:");
    println!("  voirs-cli accuracy run");

    Ok(())
}

/// Generate accuracy report from results file
#[cfg(not(doctest))]
async fn generate_accuracy_report(args: ReportArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "📄 Generating accuracy report from: {}",
        args.input.display()
    );

    // Read and parse the JSON results file
    let contents = std::fs::read_to_string(&args.input)
        .map_err(|e| format!("Failed to read results file: {}", e))?;

    let results: voirs_evaluation::accuracy_benchmarks::AccuracyBenchmarkResults =
        serde_json::from_str(&contents)
            .map_err(|e| format!("Failed to parse results JSON: {}", e))?;

    // Generate report in requested format
    let report_content = match args.format.to_lowercase().as_str() {
        "json" => generate_json_report(&results)?,
        "txt" => generate_text_report(&results),
        "html" => generate_html_report(&results),
        _ => {
            return Err(format!(
                "Unsupported format: {}. Supported formats: json, txt, html",
                args.format
            )
            .into())
        }
    };

    // Write to output file or stdout
    match args.output {
        Some(output_path) => {
            std::fs::write(&output_path, report_content)
                .map_err(|e| format!("Failed to write report: {}", e))?;
            println!("✅ Report generated: {}", output_path.display());
        }
        None => {
            println!("\n{}", report_content);
        }
    }

    Ok(())
}

/// Generate JSON format report (pretty-printed)
#[cfg(not(doctest))]
fn generate_json_report(
    results: &voirs_evaluation::accuracy_benchmarks::AccuracyBenchmarkResults,
) -> Result<String, Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(results)?;
    Ok(json)
}

/// Generate text format report
#[cfg(not(doctest))]
fn generate_text_report(
    results: &voirs_evaluation::accuracy_benchmarks::AccuracyBenchmarkResults,
) -> String {
    let mut report = String::new();

    // Header
    report.push_str("VoiRS Accuracy Benchmark Report\n");
    report.push_str(&"=".repeat(50));
    report.push_str("\n\n");

    // Timestamp and execution info
    report.push_str(&format!("Generated: {}\n", results.timestamp));
    report.push_str(&format!(
        "Total execution time: {:.2} seconds\n\n",
        results.total_time_seconds
    ));

    // Overall metrics
    report.push_str("OVERALL METRICS\n");
    report.push_str(&"-".repeat(30));
    report.push_str("\n");
    report.push_str(&format!(
        "Total test cases: {}\n",
        results.overall_metrics.total_cases
    ));
    report.push_str(&format!(
        "Overall phoneme accuracy: {:.2}%\n",
        results.overall_metrics.overall_phoneme_accuracy * 100.0
    ));
    report.push_str(&format!(
        "Overall word accuracy: {:.2}%\n",
        results.overall_metrics.overall_word_accuracy * 100.0
    ));
    report.push_str(&format!(
        "Targets met: {}/{} ({:.1}%)\n\n",
        results.overall_metrics.targets_met,
        results.overall_metrics.total_targets,
        results.overall_metrics.pass_rate
    ));

    // Language-specific results
    report.push_str("LANGUAGE-SPECIFIC RESULTS\n");
    report.push_str(&"-".repeat(30));
    report.push_str("\n");
    for (language, accuracy) in &results.overall_metrics.language_accuracies {
        report.push_str(&format!("{:?}: {:.2}%\n", language, accuracy * 100.0));
    }
    report.push_str("\n");

    // Dataset results
    report.push_str("DATASET RESULTS\n");
    report.push_str(&"-".repeat(30));
    report.push_str("\n");
    for (dataset_name, dataset_result) in &results.dataset_results {
        let status = if dataset_result.target_met {
            "✅ PASS"
        } else {
            "❌ FAIL"
        };
        report.push_str(&format!("Dataset: {}\n", dataset_name));
        report.push_str(&format!("  Status: {}\n", status));
        report.push_str(&format!("  Language: {:?}\n", dataset_result.language));
        report.push_str(&format!(
            "  Test cases: {} (Success: {}, Failed: {})\n",
            dataset_result.total_cases,
            dataset_result.successful_cases,
            dataset_result.failed_cases
        ));
        report.push_str(&format!(
            "  Phoneme accuracy: {:.2}%\n",
            dataset_result.phoneme_accuracy * 100.0
        ));
        report.push_str(&format!(
            "  Word accuracy: {:.2}%\n",
            dataset_result.word_accuracy * 100.0
        ));
        report.push_str(&format!(
            "  Target: {:.1}%\n",
            dataset_result.target_accuracy * 100.0
        ));
        report.push_str(&format!(
            "  Average edit distance: {:.2}\n",
            dataset_result.average_edit_distance
        ));
        report.push_str(&format!(
            "  Processing time: {:.2} ± {:.2} ms\n\n",
            dataset_result.processing_time_ms.mean_ms, dataset_result.processing_time_ms.std_dev_ms
        ));
    }

    // Performance statistics
    report.push_str("PERFORMANCE STATISTICS\n");
    report.push_str(&"-".repeat(30));
    report.push_str("\n");
    report.push_str(&format!(
        "Average processing time: {:.2} ms\n",
        results.performance_stats.avg_processing_time_ms
    ));
    report.push_str(&format!(
        "Median processing time: {:.2} ms\n",
        results.performance_stats.median_processing_time_ms
    ));
    report.push_str(&format!(
        "95th percentile: {:.2} ms\n",
        results.performance_stats.p95_processing_time_ms
    ));
    report.push_str(&format!(
        "Throughput: {:.1} cases/sec\n",
        results.performance_stats.throughput_cases_per_sec
    ));
    report.push_str(&format!(
        "Peak memory usage: {:.1} MB\n",
        results.performance_stats.peak_memory_mb
    ));

    report
}

/// Generate HTML format report
#[cfg(not(doctest))]
fn generate_html_report(
    results: &voirs_evaluation::accuracy_benchmarks::AccuracyBenchmarkResults,
) -> String {
    let mut html = String::new();

    // HTML header
    html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
    html.push_str("    <meta charset=\"UTF-8\">\n");
    html.push_str(
        "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
    );
    html.push_str("    <title>VoiRS Accuracy Benchmark Report</title>\n");
    html.push_str("    <style>\n");
    html.push_str("        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }\n");
    html.push_str("        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }\n");
    html.push_str(
        "        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }\n",
    );
    html.push_str("        h2 { color: #34495e; margin-top: 30px; }\n");
    html.push_str("        .metric { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }\n");
    html.push_str("        .pass { color: #27ae60; font-weight: bold; }\n");
    html.push_str("        .fail { color: #e74c3c; font-weight: bold; }\n");
    html.push_str("        table { width: 100%; border-collapse: collapse; margin: 20px 0; }\n");
    html.push_str(
        "        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }\n",
    );
    html.push_str("        th { background-color: #3498db; color: white; }\n");
    html.push_str("        .timestamp { color: #7f8c8d; font-style: italic; }\n");
    html.push_str("    </style>\n");
    html.push_str("</head>\n<body>\n");
    html.push_str("    <div class=\"container\">\n");

    // Header
    html.push_str("        <h1>🎯 VoiRS Accuracy Benchmark Report</h1>\n");
    html.push_str(&format!(
        "        <p class=\"timestamp\">Generated: {}</p>\n",
        results.timestamp
    ));
    html.push_str(&format!(
        "        <p class=\"timestamp\">Execution time: {:.2} seconds</p>\n",
        results.total_time_seconds
    ));

    // Overall metrics
    html.push_str("        <h2>📊 Overall Metrics</h2>\n");
    html.push_str("        <div class=\"metric\">\n");
    html.push_str(&format!(
        "            <strong>Total test cases:</strong> {}<br>\n",
        results.overall_metrics.total_cases
    ));
    html.push_str(&format!(
        "            <strong>Overall phoneme accuracy:</strong> {:.2}%<br>\n",
        results.overall_metrics.overall_phoneme_accuracy * 100.0
    ));
    html.push_str(&format!(
        "            <strong>Overall word accuracy:</strong> {:.2}%<br>\n",
        results.overall_metrics.overall_word_accuracy * 100.0
    ));
    html.push_str(&format!(
        "            <strong>Targets met:</strong> {}/{} ({:.1}%)\n",
        results.overall_metrics.targets_met,
        results.overall_metrics.total_targets,
        results.overall_metrics.pass_rate
    ));
    html.push_str("        </div>\n");

    // Language results table
    html.push_str("        <h2>🌍 Language-Specific Results</h2>\n");
    html.push_str("        <table>\n");
    html.push_str("            <tr><th>Language</th><th>Accuracy</th></tr>\n");
    for (language, accuracy) in &results.overall_metrics.language_accuracies {
        html.push_str(&format!(
            "            <tr><td>{:?}</td><td>{:.2}%</td></tr>\n",
            language,
            accuracy * 100.0
        ));
    }
    html.push_str("        </table>\n");

    // Dataset results table
    html.push_str("        <h2>📚 Dataset Results</h2>\n");
    html.push_str("        <table>\n");
    html.push_str("            <tr><th>Dataset</th><th>Status</th><th>Language</th><th>Cases</th><th>Phoneme Acc.</th><th>Word Acc.</th><th>Target</th></tr>\n");
    for (dataset_name, dataset_result) in &results.dataset_results {
        let status_class = if dataset_result.target_met {
            "pass"
        } else {
            "fail"
        };
        let status_text = if dataset_result.target_met {
            "✅ PASS"
        } else {
            "❌ FAIL"
        };
        html.push_str(&format!(
            "            <tr><td>{}</td><td class=\"{}\"> {}</td><td>{:?}</td><td>{}</td><td>{:.2}%</td><td>{:.2}%</td><td>{:.1}%</td></tr>\n",
            dataset_name, status_class, status_text, dataset_result.language,
            dataset_result.total_cases, dataset_result.phoneme_accuracy * 100.0,
            dataset_result.word_accuracy * 100.0, dataset_result.target_accuracy * 100.0
        ));
    }
    html.push_str("        </table>\n");

    // Performance statistics
    html.push_str("        <h2>⚡ Performance Statistics</h2>\n");
    html.push_str("        <div class=\"metric\">\n");
    html.push_str(&format!(
        "            <strong>Average processing time:</strong> {:.2} ms<br>\n",
        results.performance_stats.avg_processing_time_ms
    ));
    html.push_str(&format!(
        "            <strong>Median processing time:</strong> {:.2} ms<br>\n",
        results.performance_stats.median_processing_time_ms
    ));
    html.push_str(&format!(
        "            <strong>95th percentile:</strong> {:.2} ms<br>\n",
        results.performance_stats.p95_processing_time_ms
    ));
    html.push_str(&format!(
        "            <strong>Throughput:</strong> {:.1} cases/sec<br>\n",
        results.performance_stats.throughput_cases_per_sec
    ));
    html.push_str(&format!(
        "            <strong>Peak memory usage:</strong> {:.1} MB\n",
        results.performance_stats.peak_memory_mb
    ));
    html.push_str("        </div>\n");

    // HTML footer
    html.push_str("    </div>\n</body>\n</html>");

    html
}

// Dummy system implementations for demonstration
// In a real implementation, these would be replaced with actual system interfaces

#[cfg(not(doctest))]
struct DummyG2pSystem;
#[cfg(not(doctest))]
#[async_trait]
impl voirs_evaluation::accuracy_benchmarks::G2pSystem for DummyG2pSystem {
    async fn convert_to_phonemes(
        &self,
        text: &str,
        _language: LanguageCode,
    ) -> Result<Vec<String>, voirs_evaluation::EvaluationError> {
        Ok(text.chars().map(|c| c.to_string()).collect())
    }
}

#[cfg(not(doctest))]
struct DummyTtsSystem;
#[cfg(not(doctest))]
#[async_trait]
impl voirs_evaluation::accuracy_benchmarks::TtsSystem for DummyTtsSystem {
    async fn synthesize(
        &self,
        _text: &str,
        _language: LanguageCode,
    ) -> Result<voirs_sdk::AudioBuffer, voirs_evaluation::EvaluationError> {
        Ok(voirs_sdk::AudioBuffer::mono(vec![0.1; 16000], 16000))
    }
}

#[cfg(not(doctest))]
struct DummyAsrSystem;
#[cfg(not(doctest))]
#[async_trait]
impl voirs_evaluation::accuracy_benchmarks::AsrSystem for DummyAsrSystem {
    async fn transcribe(
        &self,
        _audio: &voirs_sdk::AudioBuffer,
        _language: LanguageCode,
    ) -> Result<String, voirs_evaluation::EvaluationError> {
        Ok("dummy transcription".to_string())
    }
}
