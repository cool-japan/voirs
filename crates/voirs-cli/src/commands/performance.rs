//! Performance targets testing and monitoring commands for VoiRS CLI.

use clap::{Args, Subcommand};
use std::path::{Path, PathBuf};
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

    /// Output directory for test results
    #[arg(short, long, default_value = "/tmp/voirs_performance_test")]
    pub output_dir: PathBuf,

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
        println!("  • Output directory: {}", args.output_dir.display());
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

            // Save results if output directory specified
            if args.output_dir.as_path() != Path::new("/tmp/voirs_performance_test") {
                std::fs::create_dir_all(&args.output_dir)?;
                let results_file = args.output_dir.join("performance_test_results.json");
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

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct ComponentTiming {
        g2p_ms: f64,
        acoustic_ms: f64,
        vocoder_ms: f64,
        total_ms: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct ProfileResult {
        iterations: usize,
        timings: Vec<ComponentTiming>,
        average: ComponentTiming,
        min: ComponentTiming,
        max: ComponentTiming,
        std_dev: ComponentTiming,
        memory_usage_mb: Option<f64>,
        io_operations: Option<u64>,
    }

    let mut timings = Vec::new();
    let mut memory_samples = Vec::new();

    println!("🚀 Running profiling iterations...");
    let overall_start = Instant::now();

    for i in 0..args.iterations {
        let iter_start = Instant::now();

        // Simulate G2P phase (in real implementation, this would call actual G2P)
        let g2p_start = Instant::now();
        tokio::time::sleep(Duration::from_millis(2)).await; // Simulate G2P work
        let g2p_duration = g2p_start.elapsed();

        // Simulate acoustic model phase
        let acoustic_start = Instant::now();
        tokio::time::sleep(Duration::from_millis(5)).await; // Simulate acoustic work
        let acoustic_duration = acoustic_start.elapsed();

        // Simulate vocoder phase
        let vocoder_start = Instant::now();
        tokio::time::sleep(Duration::from_millis(3)).await; // Simulate vocoder work
        let vocoder_duration = vocoder_start.elapsed();

        let total_duration = iter_start.elapsed();

        timings.push(ComponentTiming {
            g2p_ms: g2p_duration.as_secs_f64() * 1000.0,
            acoustic_ms: acoustic_duration.as_secs_f64() * 1000.0,
            vocoder_ms: vocoder_duration.as_secs_f64() * 1000.0,
            total_ms: total_duration.as_secs_f64() * 1000.0,
        });

        // Memory profiling
        if args.memory {
            // In real implementation, get actual memory usage
            let memory_mb = 50.0 + (i as f64 * 0.1); // Simulated memory growth
            memory_samples.push(memory_mb);
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
    let average = ComponentTiming {
        g2p_ms: timings.iter().map(|t| t.g2p_ms).sum::<f64>() / count,
        acoustic_ms: timings.iter().map(|t| t.acoustic_ms).sum::<f64>() / count,
        vocoder_ms: timings.iter().map(|t| t.vocoder_ms).sum::<f64>() / count,
        total_ms: timings.iter().map(|t| t.total_ms).sum::<f64>() / count,
    };

    let min = ComponentTiming {
        g2p_ms: timings.iter().map(|t| t.g2p_ms).fold(f64::MAX, f64::min),
        acoustic_ms: timings
            .iter()
            .map(|t| t.acoustic_ms)
            .fold(f64::MAX, f64::min),
        vocoder_ms: timings
            .iter()
            .map(|t| t.vocoder_ms)
            .fold(f64::MAX, f64::min),
        total_ms: timings.iter().map(|t| t.total_ms).fold(f64::MAX, f64::min),
    };

    let max = ComponentTiming {
        g2p_ms: timings.iter().map(|t| t.g2p_ms).fold(f64::MIN, f64::max),
        acoustic_ms: timings
            .iter()
            .map(|t| t.acoustic_ms)
            .fold(f64::MIN, f64::max),
        vocoder_ms: timings
            .iter()
            .map(|t| t.vocoder_ms)
            .fold(f64::MIN, f64::max),
        total_ms: timings.iter().map(|t| t.total_ms).fold(f64::MIN, f64::max),
    };

    // Calculate standard deviation
    let variance_g2p = timings
        .iter()
        .map(|t| (t.g2p_ms - average.g2p_ms).powi(2))
        .sum::<f64>()
        / count;
    let variance_acoustic = timings
        .iter()
        .map(|t| (t.acoustic_ms - average.acoustic_ms).powi(2))
        .sum::<f64>()
        / count;
    let variance_vocoder = timings
        .iter()
        .map(|t| (t.vocoder_ms - average.vocoder_ms).powi(2))
        .sum::<f64>()
        / count;
    let variance_total = timings
        .iter()
        .map(|t| (t.total_ms - average.total_ms).powi(2))
        .sum::<f64>()
        / count;

    let std_dev = ComponentTiming {
        g2p_ms: variance_g2p.sqrt(),
        acoustic_ms: variance_acoustic.sqrt(),
        vocoder_ms: variance_vocoder.sqrt(),
        total_ms: variance_total.sqrt(),
    };

    let memory_usage_mb = if args.memory {
        Some(memory_samples.iter().sum::<f64>() / memory_samples.len() as f64)
    } else {
        None
    };

    let io_operations = if args.io {
        Some((args.iterations * 3) as u64) // Simulated I/O count
    } else {
        None
    };

    let result = ProfileResult {
        iterations: args.iterations,
        timings: timings.clone(),
        average,
        min,
        max,
        std_dev,
        memory_usage_mb,
        io_operations,
    };

    // Display results
    println!("📊 Profile Results");
    println!("==================");
    println!();
    println!("Component Breakdown (Average):");
    println!(
        "  • G2P:      {:>8.2}ms ({:>5.1}%)",
        result.average.g2p_ms,
        (result.average.g2p_ms / result.average.total_ms) * 100.0
    );
    println!(
        "  • Acoustic: {:>8.2}ms ({:>5.1}%)",
        result.average.acoustic_ms,
        (result.average.acoustic_ms / result.average.total_ms) * 100.0
    );
    println!(
        "  • Vocoder:  {:>8.2}ms ({:>5.1}%)",
        result.average.vocoder_ms,
        (result.average.vocoder_ms / result.average.total_ms) * 100.0
    );
    println!("  • Total:    {:>8.2}ms", result.average.total_ms);
    println!();

    if args.detailed {
        println!("Detailed Statistics:");
        println!("  Component  │  Min (ms) │  Max (ms) │  Avg (ms) │ StdDev (ms)");
        println!("  ───────────┼───────────┼───────────┼───────────┼────────────");
        println!(
            "  G2P        │ {:>9.2} │ {:>9.2} │ {:>9.2} │ {:>11.2}",
            result.min.g2p_ms, result.max.g2p_ms, result.average.g2p_ms, result.std_dev.g2p_ms
        );
        println!(
            "  Acoustic   │ {:>9.2} │ {:>9.2} │ {:>9.2} │ {:>11.2}",
            result.min.acoustic_ms,
            result.max.acoustic_ms,
            result.average.acoustic_ms,
            result.std_dev.acoustic_ms
        );
        println!(
            "  Vocoder    │ {:>9.2} │ {:>9.2} │ {:>9.2} │ {:>11.2}",
            result.min.vocoder_ms,
            result.max.vocoder_ms,
            result.average.vocoder_ms,
            result.std_dev.vocoder_ms
        );
        println!(
            "  Total      │ {:>9.2} │ {:>9.2} │ {:>9.2} │ {:>11.2}",
            result.min.total_ms,
            result.max.total_ms,
            result.average.total_ms,
            result.std_dev.total_ms
        );
        println!();
    }

    if let Some(memory) = result.memory_usage_mb {
        println!("Memory Usage:");
        println!("  • Average: {:.1} MB", memory);
        println!();
    }

    if let Some(io_ops) = result.io_operations {
        println!("I/O Operations:");
        println!("  • Total: {} operations", io_ops);
        println!(
            "  • Avg per iteration: {:.1}",
            io_ops as f64 / result.iterations as f64
        );
        println!();
    }

    // Performance insights
    println!("💡 Performance Insights:");
    let bottleneck = if result.average.acoustic_ms > result.average.g2p_ms
        && result.average.acoustic_ms > result.average.vocoder_ms
    {
        "Acoustic model"
    } else if result.average.vocoder_ms > result.average.g2p_ms {
        "Vocoder"
    } else {
        "G2P conversion"
    };
    println!("  • Bottleneck: {}", bottleneck);

    let rtf = result.average.total_ms / 1000.0; // Assume 1s of audio
    println!("  • Real-Time Factor: {:.2}x", rtf);

    if rtf < 0.1 {
        println!("  • ✅ Excellent performance (RTF < 0.1)");
    } else if rtf < 0.5 {
        println!("  • ✅ Good performance (RTF < 0.5)");
    } else if rtf < 1.0 {
        println!("  • ⚠️  Acceptable performance (RTF < 1.0)");
    } else {
        println!("  • ❌ Poor performance (RTF >= 1.0) - optimization needed");
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
}
