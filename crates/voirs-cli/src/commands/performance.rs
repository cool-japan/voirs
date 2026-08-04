//! Performance targets testing and monitoring commands for VoiRS CLI.

use clap::{Args, Subcommand};
use std::path::PathBuf;
use std::time::Duration;
use voirs_acoustic::performance_targets::{PerformanceTargets, PerformanceTargetsMonitor};

/// Performance targets commands
#[derive(Debug, Clone, Args)]
pub struct PerformanceCommand {
    #[command(subcommand)]
    pub command: PerformanceSubcommand,
}

/// Performance subcommands
#[derive(Debug, Clone, Subcommand)]
pub enum PerformanceSubcommand {
    /// Run comprehensive performance targets test
    Test(TestPerformanceArgs),
    /// Monitor performance in real-time
    Monitor(MonitorPerformanceArgs),
    /// Show current performance status
    Status(StatusArgs),
    /// Generate performance report
    Report(ReportArgs),
    /// Profile synthesis performance with detailed breakdown
    Profile(ProfileArgs),
}

/// Arguments for performance testing
#[derive(Debug, Clone, Args)]
pub struct TestPerformanceArgs {
    /// Test name for identification
    #[arg(short, long, default_value = "comprehensive_performance_test")]
    pub test_name: String,

    /// Target latency in milliseconds
    #[arg(long, default_value = "1.0")]
    pub max_latency_ms: f32,

    /// Target memory per model in MB
    #[arg(long, default_value = "100.0")]
    pub max_memory_mb: f32,

    /// Target batch throughput in sentences/second
    #[arg(long, default_value = "1000.0")]
    pub min_throughput_sps: f32,

    /// Output directory for test results (if omitted, results are printed
    /// to stdout only and not saved to disk)
    #[arg(short, long)]
    pub output_dir: Option<PathBuf>,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

/// Arguments for performance monitoring
#[derive(Debug, Clone, Args)]
pub struct MonitorPerformanceArgs {
    /// Monitoring interval in seconds
    #[arg(short, long, default_value = "5")]
    pub interval_seconds: u64,

    /// Duration to monitor in seconds (0 = indefinite)
    #[arg(short, long, default_value = "60")]
    pub duration_seconds: u64,

    /// Output file for monitoring log
    #[arg(short, long)]
    pub output_file: Option<PathBuf>,

    /// Enable real-time display
    #[arg(long)]
    pub live_display: bool,
}

/// Arguments for status check
#[derive(Debug, Clone, Args)]
pub struct StatusArgs {
    /// Show detailed status information
    #[arg(long)]
    pub detailed: bool,

    /// Output format (text, json)
    #[arg(long, default_value = "text")]
    pub format: String,
}

/// Arguments for performance report
#[derive(Debug, Clone, Args)]
pub struct ReportArgs {
    /// Duration for report in minutes
    #[arg(short, long, default_value = "10")]
    pub duration_minutes: u64,

    /// Output file for report
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Report format (text, html, json)
    #[arg(long, default_value = "text")]
    pub format: String,
}

/// Arguments for performance profiling
#[derive(Debug, Clone, Args)]
pub struct ProfileArgs {
    /// Text to synthesize for profiling
    #[arg(
        short,
        long,
        default_value = "The quick brown fox jumps over the lazy dog."
    )]
    pub text: String,

    /// Voice to use for profiling
    #[arg(short, long)]
    pub voice: Option<String>,

    /// Number of iterations to run
    #[arg(short = 'n', long, default_value = "10")]
    pub iterations: usize,

    /// Output file for profiling results (JSON format)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Show detailed component breakdown
    #[arg(long)]
    pub detailed: bool,

    /// Generate flamegraph (requires cargo-flamegraph)
    #[arg(long)]
    pub flamegraph: bool,

    /// Include memory profiling
    #[arg(long)]
    pub memory: bool,

    /// Include I/O profiling
    #[arg(long)]
    pub io: bool,
}

/// Execute performance commands
pub async fn execute_performance_command(
    args: PerformanceCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        PerformanceSubcommand::Test(test_args) => run_performance_test(test_args).await,
        PerformanceSubcommand::Monitor(monitor_args) => run_performance_monitor(monitor_args).await,
        PerformanceSubcommand::Status(status_args) => show_performance_status(status_args).await,
        PerformanceSubcommand::Report(report_args) => {
            generate_performance_report(report_args).await
        }
        PerformanceSubcommand::Profile(profile_args) => run_performance_profile(profile_args).await,
    }
}

/// Run comprehensive performance targets test
async fn run_performance_test(args: TestPerformanceArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 VoiRS Performance Targets Test");
    println!("=================================");

    if args.verbose {
        println!("Test configuration:");
        println!("  • Max latency: {:.1}ms", args.max_latency_ms);
        println!("  • Max memory per model: {:.0}MB", args.max_memory_mb);
        println!(
            "  • Min batch throughput: {:.0} sentences/sec",
            args.min_throughput_sps
        );
        println!(
            "  • Output directory: {}",
            args.output_dir
                .as_deref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "(not saved to disk)".to_string())
        );
        println!();
    }

    // Create performance targets
    let targets = PerformanceTargets {
        max_latency_ms: args.max_latency_ms,
        max_memory_per_model_mb: args.max_memory_mb,
        min_batch_throughput_sps: args.min_throughput_sps,
        max_cpu_usage_percent: 80.0,
        max_memory_alloc_rate: 500.0,
        min_cache_hit_rate: 85.0,
    };

    // Create and run performance monitor
    let mut monitor = PerformanceTargetsMonitor::new(targets);

    println!("🚀 Running performance test: {}", args.test_name);
    let start_time = std::time::Instant::now();

    match monitor.run_performance_test(&args.test_name).await {
        Ok(test_result) => {
            let elapsed = start_time.elapsed();

            println!("✅ Performance test completed in {:?}", elapsed);
            println!();

            // Display results summary
            println!("📊 Performance Test Results");
            println!("==========================");
            println!("Test Name: {}", test_result.test_name);
            println!("Duration: {:?}", test_result.duration);
            println!(
                "Targets Met: {}",
                if test_result.meets_targets {
                    "✅ YES"
                } else {
                    "❌ NO"
                }
            );
            println!("Total Measurements: {}", test_result.measurements.len());
            println!();

            // Summary statistics
            let summary = &test_result.summary;
            println!("Performance Summary:");
            println!(
                "  • Average Latency: {:.2}ms (target: <{:.1}ms)",
                summary.avg_latency_ms, args.max_latency_ms
            );
            println!("  • P95 Latency: {:.2}ms", summary.p95_latency_ms);
            println!(
                "  • Peak Memory: {:.1}MB (target: <{:.0}MB)",
                summary.peak_memory_mb, args.max_memory_mb
            );
            println!(
                "  • Average Throughput: {:.1} ops/sec (target: >{:.0} ops/sec)",
                summary.avg_throughput_ops, args.min_throughput_sps
            );
            println!("  • Success Rate: {:.1}%", summary.success_rate);
            println!();

            // Show violations if any
            if !test_result.violations.is_empty() {
                println!("⚠️  Target Violations:");
                for violation in &test_result.violations {
                    println!(
                        "  • {}: {} (severity: {}/10)",
                        violation.target_type, violation.description, violation.severity
                    );
                    if args.verbose {
                        println!("    Remediation: {}", violation.remediation);
                    }
                }
                println!();
            }

            // Show recommendations
            if !test_result.recommendations.is_empty() {
                println!("💡 Optimization Recommendations:");
                for (i, recommendation) in test_result.recommendations.iter().enumerate() {
                    println!("  {}. {}", i + 1, recommendation);
                }
                println!();
            }

            // Save results only if the user explicitly requested an output directory.
            if let Some(output_dir) = &args.output_dir {
                std::fs::create_dir_all(output_dir)?;
                let results_file = output_dir.join("performance_test_results.json");
                let json_content = serde_json::to_string_pretty(&test_result)?;
                std::fs::write(&results_file, json_content)?;
                println!("📁 Results saved to: {}", results_file.display());
            }

            if test_result.meets_targets {
                println!("🎉 All performance targets achieved!");
                std::process::exit(0);
            } else {
                println!("⚠️  Some performance targets not met. See recommendations above.");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("❌ Performance test failed: {}", e);
            std::process::exit(1);
        }
    }
}

/// Run real-time performance monitoring
async fn run_performance_monitor(
    args: MonitorPerformanceArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 VoiRS Performance Monitor");
    println!("============================");
    println!("Monitoring interval: {}s", args.interval_seconds);

    if args.duration_seconds > 0 {
        println!("Duration: {}s", args.duration_seconds);
    } else {
        println!("Duration: Indefinite (Ctrl+C to stop)");
    }
    println!();

    let targets = PerformanceTargets::default();
    let mut monitor = PerformanceTargetsMonitor::new(targets);

    let monitoring_interval = Duration::from_secs(args.interval_seconds);
    monitor.start_monitoring(monitoring_interval).await?;

    println!("🔄 Performance monitoring started...");

    let start_time = std::time::Instant::now();
    let max_duration = if args.duration_seconds > 0 {
        Some(Duration::from_secs(args.duration_seconds))
    } else {
        None
    };

    loop {
        // Check if we should stop monitoring
        if let Some(max_dur) = max_duration {
            if start_time.elapsed() >= max_dur {
                break;
            }
        }

        // Display current status if live display is enabled
        if args.live_display {
            let status = monitor.get_performance_status();

            // Clear screen (simple approach)
            print!("\x1b[2J\x1b[H");

            println!("📈 VoiRS Performance Monitor - Live View");
            println!("========================================");
            println!("Monitoring time: {:?}", start_time.elapsed());
            println!(
                "Targets met: {}",
                if status.targets_met {
                    "✅ YES"
                } else {
                    "❌ NO"
                }
            );
            println!(
                "Active monitoring: {}",
                if status.monitoring_active {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!("Measurements collected: {}", status.measurement_count);
            println!();

            let summary = &status.current_summary;
            println!("Current Performance:");
            println!(
                "  • Latency: avg {:.2}ms, p95 {:.2}ms",
                summary.avg_latency_ms, summary.p95_latency_ms
            );
            println!(
                "  • Memory: avg {:.1}MB, peak {:.1}MB",
                summary.avg_memory_mb, summary.peak_memory_mb
            );
            println!("  • Throughput: {:.1} ops/sec", summary.avg_throughput_ops);
            println!(
                "  • CPU: avg {:.1}%, peak {:.1}%",
                summary.avg_cpu_usage, summary.peak_cpu_usage
            );

            if !status.active_violations.is_empty() {
                println!();
                println!("⚠️  Active Violations:");
                for violation in &status.active_violations {
                    println!("  • {}: {}", violation.target_type, violation.description);
                }
            }

            println!();
            println!("Press Ctrl+C to stop monitoring...");
        }

        // Wait for next monitoring interval
        tokio::time::sleep(monitoring_interval).await;
    }

    monitor.stop_monitoring();
    println!("\n📊 Performance monitoring completed.");

    // Generate final report
    let report = monitor.generate_performance_report(start_time.elapsed());
    println!("\n📋 Final Performance Report:");
    println!("Target Compliance: {:.1}%", report.target_compliance);

    if let Some(output_file) = args.output_file {
        let report_content = format!(
            "VoiRS Performance Monitoring Report\n\
                                     ===================================\n\
                                     Duration: {:?}\n\
                                     Target Compliance: {:.1}%\n\
                                     Targets Met: {}\n\
                                     Measurements: {}\n",
            start_time.elapsed(),
            report.target_compliance,
            report.performance_status.targets_met,
            report.performance_status.measurement_count
        );

        std::fs::write(&output_file, report_content)?;
        println!("📁 Monitoring log saved to: {}", output_file.display());
    }

    Ok(())
}

/// Show current performance status
async fn show_performance_status(args: StatusArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 VoiRS Performance Status");
    println!("===========================");

    let targets = PerformanceTargets::default();
    let monitor = PerformanceTargetsMonitor::new(targets);
    let status = monitor.get_performance_status();

    match args.format.as_str() {
        "json" => {
            let json_output = serde_json::to_string_pretty(&status)?;
            println!("{}", json_output);
        }
        _ => {
            println!(
                "Targets Met: {}",
                if status.targets_met {
                    "✅ YES"
                } else {
                    "❌ NO"
                }
            );
            println!(
                "Monitoring Active: {}",
                if status.monitoring_active {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!("Measurements Collected: {}", status.measurement_count);
            println!();

            if args.detailed {
                let summary = &status.current_summary;
                println!("Performance Summary:");
                println!("  • Total Operations: {}", summary.total_operations);
                println!("  • Success Rate: {:.1}%", summary.success_rate);
                println!("  • Average Latency: {:.2}ms", summary.avg_latency_ms);
                println!("  • P95 Latency: {:.2}ms", summary.p95_latency_ms);
                println!("  • Max Latency: {:.2}ms", summary.max_latency_ms);
                println!("  • Average Memory: {:.1}MB", summary.avg_memory_mb);
                println!("  • Peak Memory: {:.1}MB", summary.peak_memory_mb);
                println!(
                    "  • Average Throughput: {:.1} ops/sec",
                    summary.avg_throughput_ops
                );
                println!(
                    "  • Min Throughput: {:.1} ops/sec",
                    summary.min_throughput_ops
                );
                println!("  • Average CPU: {:.1}%", summary.avg_cpu_usage);
                println!("  • Peak CPU: {:.1}%", summary.peak_cpu_usage);
                println!();

                let latency_stats = &status.latency_stats;
                println!("Latency Optimizer:");
                println!("  • Average Latency: {:.2}ms", latency_stats.avg_latency_ms);
                println!(
                    "  • Target Latency: {:.2}ms",
                    latency_stats.target_latency_ms
                );
                println!("  • Meeting Target: {}", latency_stats.is_meeting_target);
                println!(
                    "  • Optimal Chunk Size: {}",
                    latency_stats.optimal_chunk_size
                );
                println!("  • Measurements: {}", latency_stats.measurements_count);
                println!();

                let pool_stats = &status.memory_pool_stats;
                println!("Memory Pool:");
                println!("  • Cache Hits: {}", pool_stats.hits);
                println!("  • Cache Misses: {}", pool_stats.misses);
                println!("  • Returns: {}", pool_stats.returns);
                println!("  • Total Pooled: {}", pool_stats.total_pooled);
                if pool_stats.hits + pool_stats.misses > 0 {
                    let hit_rate = pool_stats.hits as f64
                        / (pool_stats.hits + pool_stats.misses) as f64
                        * 100.0;
                    println!("  • Hit Rate: {:.1}%", hit_rate);
                }
            }

            if !status.active_violations.is_empty() {
                println!("⚠️  Active Violations:");
                for violation in &status.active_violations {
                    println!("  • {}: {}", violation.target_type, violation.description);
                    if args.detailed {
                        println!(
                            "    Expected: {:.2}, Actual: {:.2}, Severity: {}/10",
                            violation.expected, violation.actual, violation.severity
                        );
                        println!("    Remediation: {}", violation.remediation);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Generate performance report
async fn generate_performance_report(args: ReportArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Generating VoiRS Performance Report");
    println!("======================================");

    let targets = PerformanceTargets::default();
    let monitor = PerformanceTargetsMonitor::new(targets);

    let report_duration = Duration::from_secs(args.duration_minutes * 60);
    let report = monitor.generate_performance_report(report_duration);

    let report_content = match args.format.as_str() {
        "json" => serde_json::to_string_pretty(&report)?,
        "html" => generate_html_report(&report),
        _ => generate_text_report(&report),
    };

    match args.output {
        Some(output_file) => {
            std::fs::write(&output_file, &report_content)?;
            println!("📁 Report saved to: {}", output_file.display());
        }
        None => {
            println!("{}", report_content);
        }
    }

    Ok(())
}

/// Generate text format performance report
fn generate_text_report(report: &voirs_acoustic::performance_targets::PerformanceReport) -> String {
    format!(
        "VoiRS Performance Report\n\
         ========================\n\
         \n\
         Target Compliance: {:.1}%\n\
         Targets Met: {}\n\
         \n\
         Current Performance:\n\
         • Latency: avg {:.2}ms, p95 {:.2}ms, max {:.2}ms\n\
         • Memory: avg {:.1}MB, peak {:.1}MB\n\
         • Throughput: avg {:.1} ops/s, min {:.1} ops/s\n\
         • CPU Usage: avg {:.1}%, peak {:.1}%\n\
         • Success Rate: {:.1}%\n\
         \n\
         Active Violations: {}\n\
         \n\
         Optimization Suggestions:\n\
         {}\n",
        report.target_compliance,
        report.performance_status.targets_met,
        report.performance_status.current_summary.avg_latency_ms,
        report.performance_status.current_summary.p95_latency_ms,
        report.performance_status.current_summary.max_latency_ms,
        report.performance_status.current_summary.avg_memory_mb,
        report.performance_status.current_summary.peak_memory_mb,
        report.performance_status.current_summary.avg_throughput_ops,
        report.performance_status.current_summary.min_throughput_ops,
        report.performance_status.current_summary.avg_cpu_usage,
        report.performance_status.current_summary.peak_cpu_usage,
        report.performance_status.current_summary.success_rate,
        report.performance_status.active_violations.len(),
        report.optimization_suggestions.join("\n• ")
    )
}

/// Generate HTML format performance report
fn generate_html_report(report: &voirs_acoustic::performance_targets::PerformanceReport) -> String {
    format!(
        "<!DOCTYPE html>\n\
         <html>\n\
         <head>\n\
         <title>VoiRS Performance Report</title>\n\
         <style>\n\
         body {{ font-family: Arial, sans-serif; margin: 40px; }}\n\
         .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}\n\
         .metric {{ margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 3px; }}\n\
         .violation {{ color: #d32f2f; font-weight: bold; }}\n\
         .success {{ color: #388e3c; font-weight: bold; }}\n\
         </style>\n\
         </head>\n\
         <body>\n\
         <div class=\"header\">\n\
         <h1>🎯 VoiRS Performance Report</h1>\n\
         <p>Target Compliance: <span class=\"{}\">{:.1}%</span></p>\n\
         </div>\n\
         \n\
         <h2>Current Performance</h2>\n\
         <div class=\"metric\">Average Latency: {:.2}ms</div>\n\
         <div class=\"metric\">Peak Memory: {:.1}MB</div>\n\
         <div class=\"metric\">Average Throughput: {:.1} ops/s</div>\n\
         <div class=\"metric\">Success Rate: {:.1}%</div>\n\
         \n\
         <h2>Optimization Suggestions</h2>\n\
         <ul>\n\
         {}\n\
         </ul>\n\
         \n\
         </body>\n\
         </html>",
        if report.target_compliance >= 80.0 {
            "success"
        } else {
            "violation"
        },
        report.target_compliance,
        report.performance_status.current_summary.avg_latency_ms,
        report.performance_status.current_summary.peak_memory_mb,
        report.performance_status.current_summary.avg_throughput_ops,
        report.performance_status.current_summary.success_rate,
        report
            .optimization_suggestions
            .iter()
            .map(|s| format!("<li>{}</li>", s))
            .collect::<Vec<_>>()
            .join("\n")
    )
}

/// Run detailed performance profiling
async fn run_performance_profile(args: ProfileArgs) -> Result<(), Box<dyn std::error::Error>> {
    use serde::{Deserialize, Serialize};
    use std::time::Instant;
    use voirs_g2p::G2p;

    println!("🔍 VoiRS Performance Profiler");
    println!("============================");
    println!();
    println!("Configuration:");
    println!("  • Text: \"{}\"", args.text);
    println!("  • Voice: {}", args.voice.as_deref().unwrap_or("default"));
    println!("  • Iterations: {}", args.iterations);
    println!("  • Detailed: {}", if args.detailed { "yes" } else { "no" });
    println!(
        "  • Memory profiling: {}",
        if args.memory { "yes" } else { "no" }
    );
    println!("  • I/O profiling: {}", if args.io { "yes" } else { "no" });
    println!();

    if args.flamegraph {
        println!("⚠️  Flamegraph generation requires cargo-flamegraph to be installed.");
        println!("    Install with: cargo install flamegraph");
        println!("    Run with: cargo flamegraph --bin voirs -- performance profile");
        println!();
    }

    /// Real per-iteration timing. `full_pipeline_ms`/`audio_duration_s` are
    /// `None` whenever no production `VoirsPipeline` could be built in this
    /// environment (e.g. no cached/downloadable voice model) -- never a
    /// fabricated stand-in value.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct ComponentTiming {
        /// Real standalone G2P conversion time (`voirs_g2p`'s rule-based
        /// backend on English text -- always measurable, no model weights
        /// needed).
        g2p_ms: f64,
        /// Real end-to-end `VoirsPipeline::synthesize` wall-clock time (its
        /// own G2P + acoustic model + vocoder combined). The SDK does not
        /// expose a public per-stage hook, so acoustic and vocoder cannot be
        /// measured separately without fabricating a split.
        full_pipeline_ms: Option<f64>,
        /// Real duration (seconds) of the audio `full_pipeline_ms` produced,
        /// read directly from the returned `AudioBuffer` -- used for a real
        /// (not assumed-1-second) real-time factor.
        audio_duration_s: Option<f64>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    struct ComponentTimingSummary {
        g2p_ms: f64,
        full_pipeline_ms: Option<f64>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct ProfileResult {
        iterations: usize,
        timings: Vec<ComponentTiming>,
        average: ComponentTimingSummary,
        min: ComponentTimingSummary,
        max: ComponentTimingSummary,
        std_dev: ComponentTimingSummary,
        /// `None` if a full pipeline was never available, explaining why
        /// `full_pipeline_ms` is `None` throughout `timings`.
        pipeline_unavailable_reason: Option<String>,
        memory_usage_mb: Option<f64>,
        /// Real per-iteration peak-RSS deltas are not implemented; honestly
        /// `None` rather than a formula-derived count. See `--memory` for a
        /// real (if coarser) memory measurement.
        io_operations: Option<u64>,
    }

    // Real standalone G2P timing needs no model weights, so it always runs.
    let g2p = voirs_g2p::backends::RuleBasedG2p::new(voirs_g2p::LanguageCode::EnUs);

    // Real full-pipeline attempt, bounded so a missing network connection
    // (needed to fetch a default voice's model weights) fails fast and
    // honestly instead of hanging this command for minutes.
    println!("🚀 Preparing synthesis pipeline for profiling...");
    let mut pipeline_builder = voirs_sdk::VoirsPipeline::builder();
    if let Some(voice) = args.voice.as_deref() {
        pipeline_builder = pipeline_builder.with_voice(voice);
    }
    let (pipeline, pipeline_unavailable_reason) =
        match tokio::time::timeout(Duration::from_secs(15), pipeline_builder.build()).await {
            Ok(Ok(pipeline)) => (Some(pipeline), None),
            Ok(Err(e)) => (
                None,
                Some(format!("could not build a synthesis pipeline: {e}")),
            ),
            Err(_) => (
                None,
                Some("building the synthesis pipeline timed out after 15s".to_string()),
            ),
        };
    if let Some(reason) = &pipeline_unavailable_reason {
        println!("⚠️  Full-pipeline timing unavailable: {reason}");
        println!("    Only G2P timing will be measured for this run.");
    }
    println!();

    let mut timings = Vec::new();
    let mut memory_samples = Vec::new();

    println!("🚀 Running profiling iterations...");
    let overall_start = Instant::now();

    for i in 0..args.iterations {
        // Real G2P timing: actually converts `args.text`, not a sleep.
        let g2p_start = Instant::now();
        let phonemes = g2p
            .to_phonemes(&args.text, Some(voirs_g2p::LanguageCode::EnUs))
            .await;
        let g2p_ms = g2p_start.elapsed().as_secs_f64() * 1000.0;
        if let Err(e) = phonemes {
            eprintln!("\n⚠️  G2P conversion failed for this iteration: {e}");
        }

        // Real full-pipeline timing, when a pipeline was actually built.
        let (full_pipeline_ms, audio_duration_s) = if let Some(pipeline) = &pipeline {
            let synth_start = Instant::now();
            match pipeline.synthesize(&args.text).await {
                Ok(audio) => (
                    Some(synth_start.elapsed().as_secs_f64() * 1000.0),
                    Some(audio.duration() as f64),
                ),
                Err(e) => {
                    eprintln!("\n⚠️  Synthesis failed for this iteration: {e}");
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        timings.push(ComponentTiming {
            g2p_ms,
            full_pipeline_ms,
            audio_duration_s,
        });

        // Real memory profiling: actual process RSS, not a growth formula.
        if args.memory {
            if let Some(mb) = measure_current_memory_mb() {
                memory_samples.push(mb);
            }
        }

        if (i + 1) % 10 == 0 || i == args.iterations - 1 {
            print!("\r  Progress: {}/{} iterations", i + 1, args.iterations);
            std::io::Write::flush(&mut std::io::stdout())?;
        }
    }

    println!();
    let overall_duration = overall_start.elapsed();
    println!("✅ Profiling completed in {:?}", overall_duration);
    println!();

    // Calculate statistics
    let count = timings.len() as f64;
    let g2p_values: Vec<f64> = timings.iter().map(|t| t.g2p_ms).collect();
    let full_pipeline_values: Vec<f64> =
        timings.iter().filter_map(|t| t.full_pipeline_ms).collect();

    let mean = |values: &[f64]| -> f64 { values.iter().sum::<f64>() / values.len() as f64 };
    let mean_opt = |values: &[f64]| -> Option<f64> {
        if values.is_empty() {
            None
        } else {
            Some(mean(values))
        }
    };
    let min_opt = |values: &[f64]| -> Option<f64> { values.iter().copied().reduce(f64::min) };
    let max_opt = |values: &[f64]| -> Option<f64> { values.iter().copied().reduce(f64::max) };
    let std_dev_opt = |values: &[f64], avg: f64| -> Option<f64> {
        if values.is_empty() {
            None
        } else {
            Some(
                (values.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / values.len() as f64)
                    .sqrt(),
            )
        }
    };

    let g2p_avg = mean(&g2p_values);
    let full_pipeline_avg = mean_opt(&full_pipeline_values);

    let average = ComponentTimingSummary {
        g2p_ms: g2p_avg,
        full_pipeline_ms: full_pipeline_avg,
    };
    let min = ComponentTimingSummary {
        g2p_ms: min_opt(&g2p_values).unwrap_or(0.0),
        full_pipeline_ms: min_opt(&full_pipeline_values),
    };
    let max = ComponentTimingSummary {
        g2p_ms: max_opt(&g2p_values).unwrap_or(0.0),
        full_pipeline_ms: max_opt(&full_pipeline_values),
    };
    let std_dev = ComponentTimingSummary {
        g2p_ms: std_dev_opt(&g2p_values, g2p_avg).unwrap_or(0.0),
        full_pipeline_ms: full_pipeline_avg.and_then(|avg| std_dev_opt(&full_pipeline_values, avg)),
    };

    let memory_usage_mb = if args.memory {
        mean_opt(&memory_samples)
    } else {
        None
    };

    // Real per-file I/O instrumentation is not implemented; report that
    // honestly instead of a formula-derived count.
    let io_operations = None;
    if args.io {
        println!(
            "ℹ️  I/O profiling requested, but no real per-syscall I/O counter is implemented \
             in this build; reporting none rather than a fabricated count."
        );
    }

    let result = ProfileResult {
        iterations: args.iterations,
        timings: timings.clone(),
        average,
        min,
        max,
        std_dev,
        pipeline_unavailable_reason,
        memory_usage_mb,
        io_operations,
    };

    // Display results
    println!("📊 Profile Results");
    println!("==================");
    println!();
    println!("Component Breakdown (Average):");
    println!("  • G2P (standalone):     {:>8.2}ms", result.average.g2p_ms);
    match result.average.full_pipeline_ms {
        Some(full_pipeline_ms) => {
            println!("  • Full pipeline:        {:>8.2}ms", full_pipeline_ms);
        }
        None => {
            println!(
                "  • Full pipeline:        unavailable ({})",
                result
                    .pipeline_unavailable_reason
                    .as_deref()
                    .unwrap_or("unknown reason")
            );
        }
    }
    println!();

    if args.detailed {
        println!("Detailed Statistics:");
        println!("  Component      │  Min (ms) │  Max (ms) │  Avg (ms) │ StdDev (ms)");
        println!("  ───────────────┼───────────┼───────────┼───────────┼────────────");
        println!(
            "  G2P            │ {:>9.2} │ {:>9.2} │ {:>9.2} │ {:>11.2}",
            result.min.g2p_ms, result.max.g2p_ms, result.average.g2p_ms, result.std_dev.g2p_ms
        );
        match (
            result.min.full_pipeline_ms,
            result.max.full_pipeline_ms,
            result.average.full_pipeline_ms,
            result.std_dev.full_pipeline_ms,
        ) {
            (Some(min), Some(max), Some(avg), Some(std_dev)) => {
                println!(
                    "  Full pipeline  │ {:>9.2} │ {:>9.2} │ {:>9.2} │ {:>11.2}",
                    min, max, avg, std_dev
                );
            }
            _ => {
                println!("  Full pipeline  │       n/a │       n/a │       n/a │         n/a");
            }
        }
        println!();
    }

    if let Some(memory) = result.memory_usage_mb {
        println!("Memory Usage (real process RSS):");
        println!("  • Average: {:.1} MB", memory);
        println!();
    } else if args.memory {
        println!("Memory Usage: could not be measured on this platform");
        println!();
    }

    if args.io {
        println!("I/O Operations: not measured (see note above)");
        println!();
    }

    // Performance insights: only computed from real measurements.
    println!("💡 Performance Insights:");
    match result.average.full_pipeline_ms {
        Some(full_pipeline_ms) => {
            let bottleneck = if full_pipeline_ms > result.average.g2p_ms {
                "Acoustic model + vocoder (full pipeline dominates standalone G2P)"
            } else {
                "G2P conversion"
            };
            println!("  • Bottleneck: {}", bottleneck);

            // Real RTF: real synthesis wall time over the real audio
            // duration that was actually produced (never an assumed 1s).
            let real_audio_seconds: f64 = timings.iter().filter_map(|t| t.audio_duration_s).sum();
            if real_audio_seconds > 0.0 {
                let real_synthesis_seconds: f64 = timings
                    .iter()
                    .filter_map(|t| t.full_pipeline_ms)
                    .sum::<f64>()
                    / 1000.0;
                let rtf = real_synthesis_seconds / real_audio_seconds;
                println!("  • Real-Time Factor: {:.2}x (measured against real synthesized audio duration)", rtf);

                if rtf < 0.1 {
                    println!("  • ✅ Excellent performance (RTF < 0.1)");
                } else if rtf < 0.5 {
                    println!("  • ✅ Good performance (RTF < 0.5)");
                } else if rtf < 1.0 {
                    println!("  • ⚠️  Acceptable performance (RTF < 1.0)");
                } else {
                    println!("  • ❌ Poor performance (RTF >= 1.0) - optimization needed");
                }
            } else {
                println!("  • Real-Time Factor: not computed (no audio duration recorded)");
            }
        }
        None => {
            println!(
                "  • Bottleneck / Real-Time Factor: not computed (full pipeline unavailable: {})",
                result
                    .pipeline_unavailable_reason
                    .as_deref()
                    .unwrap_or("unknown reason")
            );
        }
    }
    println!();

    // Save results to file if requested
    if let Some(output_path) = args.output {
        let json_output = serde_json::to_string_pretty(&result)?;
        std::fs::write(&output_path, json_output)?;
        println!("✅ Profile results saved to: {}", output_path.display());
    }

    Ok(())
}

/// Measure this process's real current memory usage in MB.
///
/// Prefers `/proc/self/status` `VmRSS` on Linux (current resident set size).
/// Falls back to `getrusage`'s `ru_maxrss` on other Unix platforms, which is
/// *peak* RSS rather than current usage (KB on Linux, bytes on macOS) --
/// still a real measurement, just a different real quantity. Returns `None`
/// (never a fabricated constant) if no real measurement could be obtained.
fn measure_current_memory_mb() -> Option<f64> {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                if let Some(kb) = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<f64>().ok())
                {
                    return Some(kb / 1024.0);
                }
            }
        }
    }

    #[cfg(unix)]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
        if result == 0 {
            let usage = unsafe { usage.assume_init() };
            #[cfg(target_os = "linux")]
            {
                return Some(usage.ru_maxrss as f64 / 1024.0);
            }
            #[cfg(target_os = "macos")]
            {
                return Some(usage.ru_maxrss as f64 / (1024.0 * 1024.0));
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos")))]
            {
                let _ = usage;
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_args_defaults() {
        // Test that ProfileArgs has sensible defaults
        let args = ProfileArgs {
            text: "test".to_string(),
            voice: None,
            iterations: 10,
            output: None,
            detailed: false,
            flamegraph: false,
            memory: false,
            io: false,
        };

        assert_eq!(args.text, "test");
        assert!(args.voice.is_none());
        assert_eq!(args.iterations, 10);
        assert!(!args.detailed);
        assert!(!args.flamegraph);
        assert!(!args.memory);
        assert!(!args.io);
    }

    #[tokio::test]
    async fn test_profile_execution() {
        // Test basic profile execution
        let args = ProfileArgs {
            text: "Hello world".to_string(),
            voice: Some("test-voice".to_string()),
            iterations: 5,
            output: None,
            detailed: true,
            flamegraph: false,
            memory: true,
            io: true,
        };

        let result = run_performance_profile(args).await;
        assert!(result.is_ok(), "Profile execution should succeed");
    }

    #[tokio::test]
    async fn test_profile_with_output_file() {
        use std::env;

        let temp_dir = env::temp_dir();
        let output_file = temp_dir.join("profile_test_output.json");

        let args = ProfileArgs {
            text: "Test profiling".to_string(),
            voice: None,
            iterations: 3,
            output: Some(output_file.clone()),
            detailed: false,
            flamegraph: false,
            memory: false,
            io: false,
        };

        let result = run_performance_profile(args).await;
        assert!(result.is_ok(), "Profile with output file should succeed");

        // Check that output file was created
        assert!(output_file.exists(), "Output file should be created");

        // Verify JSON content
        let content = std::fs::read_to_string(&output_file).unwrap();
        assert!(
            content.contains("iterations"),
            "Output should contain iterations field"
        );
        assert!(
            content.contains("average"),
            "Output should contain average field"
        );

        // Cleanup
        let _ = std::fs::remove_file(output_file);
    }

    #[test]
    fn test_profile_args_validation() {
        // Test that ProfileArgs accepts valid configurations
        let args = ProfileArgs {
            text: "The quick brown fox".to_string(),
            voice: Some("kokoro-en".to_string()),
            iterations: 100,
            output: Some(PathBuf::from("/tmp/profile.json")),
            detailed: true,
            flamegraph: true,
            memory: true,
            io: true,
        };

        assert_eq!(args.iterations, 100);
        assert!(args.detailed);
        assert!(args.flamegraph);
        assert!(args.memory);
        assert!(args.io);
    }

    #[test]
    fn test_component_timing_calculation() {
        // Test that component timing percentages are calculated correctly
        let total_ms = 10.0;
        let g2p_ms = 2.0;
        let acoustic_ms = 5.0;
        let vocoder_ms = 3.0;

        let g2p_percent = (g2p_ms / total_ms) * 100.0;
        let acoustic_percent = (acoustic_ms / total_ms) * 100.0;
        let vocoder_percent = (vocoder_ms / total_ms) * 100.0;

        assert_eq!(g2p_percent, 20.0);
        assert_eq!(acoustic_percent, 50.0);
        assert_eq!(vocoder_percent, 30.0);

        // Total should be 100%
        let total_percent = g2p_percent + acoustic_percent + vocoder_percent;
        assert!((total_percent - 100.0_f64).abs() < 0.001);
    }

    #[test]
    fn test_measure_current_memory_mb_returns_real_positive_value() {
        // Regression test for the `50.0 + i * 0.1` fabricated memory-growth
        // formula: the real measurement must return a plausible positive
        // number for this actually-running process (never a formula output
        // masquerading as a measurement).
        match measure_current_memory_mb() {
            Some(mb) => assert!(mb > 0.0, "measured RSS must be positive, got {mb}"),
            None => {
                // Acceptable on platforms with neither /proc/self/status nor
                // getrusage, but not on the Unix CI/dev machines this crate
                // targets.
            }
        }
    }

    #[tokio::test]
    async fn test_profile_io_and_memory_are_never_fabricated_formulas() {
        let temp_dir = std::env::temp_dir();
        let output_file = temp_dir.join(format!(
            "voirs_profile_honesty_test_{}.json",
            std::process::id()
        ));

        let args = ProfileArgs {
            text: "Regression test for fabricated performance numbers.".to_string(),
            voice: None,
            iterations: 4,
            output: Some(output_file.clone()),
            detailed: true,
            flamegraph: false,
            memory: true,
            io: true,
        };

        let result = run_performance_profile(args).await;
        assert!(
            result.is_ok(),
            "profiling should complete even when no production pipeline is available"
        );

        let content = std::fs::read_to_string(&output_file).expect("output file should exist");
        let json: serde_json::Value =
            serde_json::from_str(&content).expect("output should be valid JSON");

        // Regression: `io_operations` must never be the old `iterations * 3`
        // formula (which for 4 iterations would be exactly 12) -- it must be
        // absent/null since no real per-syscall I/O counter is implemented.
        assert!(
            json["io_operations"].is_null(),
            "io_operations must be honestly null, not a fabricated iterations*3 count: {json}"
        );

        // Regression: each per-iteration timing must carry a real,
        // non-negative, finite G2P measurement (not a JSON-unrepresentable
        // NaN/Infinity, and not literally negative).
        let timings = json["timings"]
            .as_array()
            .expect("timings should be an array");
        assert_eq!(timings.len(), 4);
        let g2p_values: Vec<f64> = timings
            .iter()
            .map(|t| t["g2p_ms"].as_f64().expect("g2p_ms should be a number"))
            .collect();
        assert!(
            g2p_values.iter().all(|&ms| ms.is_finite() && ms >= 0.0),
            "g2p_ms values must be real finite non-negative measurements: {g2p_values:?}"
        );

        // Whenever a full pipeline was not available, `full_pipeline_ms`
        // must be `null`, never a fabricated number.
        if json["pipeline_unavailable_reason"].is_string() {
            for timing in timings {
                assert!(
                    timing["full_pipeline_ms"].is_null(),
                    "full_pipeline_ms must be null when the pipeline is unavailable, got: {timing}"
                );
            }
        }

        let _ = std::fs::remove_file(&output_file);
    }

    #[tokio::test]
    async fn test_g2p_timing_scales_with_real_input_length_direct() {
        // Directly exercises the same `voirs_g2p::backends::RuleBasedG2p`
        // call `run_performance_profile` uses, without paying the
        // pipeline-build timeout cost. Regression test for the old
        // "Simulate G2P work" `tokio::time::sleep(Duration::from_millis(2))`,
        // which never read `args.text` at all: real conversion work must
        // scale with input length, a fixed sleep cannot.
        use voirs_g2p::G2p;
        let g2p = voirs_g2p::backends::RuleBasedG2p::new(voirs_g2p::LanguageCode::EnUs);

        let short = "hi";
        let long = "the quick brown fox jumps over the lazy dog ".repeat(500);

        // Average several repetitions to smooth out scheduler jitter under
        // a parallel test run.
        let mut short_total = std::time::Duration::ZERO;
        let mut long_total = std::time::Duration::ZERO;
        const REPS: u32 = 20;
        for _ in 0..REPS {
            let start = std::time::Instant::now();
            g2p.to_phonemes(short, Some(voirs_g2p::LanguageCode::EnUs))
                .await
                .expect("real G2P conversion should succeed for plain ASCII text");
            short_total += start.elapsed();

            let start = std::time::Instant::now();
            g2p.to_phonemes(&long, Some(voirs_g2p::LanguageCode::EnUs))
                .await
                .expect("real G2P conversion should succeed for plain ASCII text");
            long_total += start.elapsed();
        }

        assert!(
            long_total > short_total,
            "G2P conversion of ~23,000 characters must take longer than 2 characters \
             (short={short_total:?}, long={long_total:?}); a hardcoded sleep would show \
             no such scaling with input"
        );
    }
}
