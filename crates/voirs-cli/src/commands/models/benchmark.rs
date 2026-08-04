//! Model benchmarking command implementation.

use crate::GlobalOptions;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use voirs_g2p::{
    accuracy::{AccuracyBenchmark, TestCase},
    backends::rule_based::RuleBasedG2p,
    LanguageCode,
};
use voirs_sdk::audio::dsp;
use voirs_sdk::config::AppConfig;
use voirs_sdk::types::SynthesisConfig;
use voirs_sdk::AudioBuffer;
use voirs_sdk::VoirsPipeline;
use voirs_sdk::{QualityLevel, Result};

/// Benchmark results for a model
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub model_id: String,
    pub avg_synthesis_time: Duration,
    pub avg_audio_duration: Duration,
    pub real_time_factor: f64,
    pub memory_usage_mb: f64,
    /// Composite performance + reliability + measured audio signal-health
    /// score in `[0.0, 5.0]` (see [`calculate_quality_score`]). This is a
    /// technical proxy derived from real measurements, NOT a perceptual
    /// quality or MOS prediction.
    pub quality_score: f64,
    pub success_rate: f64,
    pub phoneme_accuracy: Option<f64>,
    pub word_accuracy: Option<f64>,
    pub accuracy_target_met: Option<bool>,
    pub english_accuracy_target_met: Option<bool>,
    pub japanese_accuracy_target_met: Option<bool>,
    pub latency_target_met: bool,
    pub memory_target_met: bool,
}

/// Run benchmark models command
pub async fn run_benchmark_models(
    model_ids: &[String],
    iterations: u32,
    include_accuracy: bool,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("Benchmarking TTS Models");
        println!("=======================");
        println!("Iterations: {}", iterations);
        println!("Models: {}", model_ids.len());
        if include_accuracy {
            println!("Accuracy Testing: Enabled (CMU Test Set)");
        }
        println!();
    }

    let test_sentences = get_test_sentences();
    let mut results = Vec::new();

    // Load accuracy benchmark if requested
    let accuracy_benchmark = if include_accuracy {
        if !global.quiet {
            println!("Loading CMU accuracy test data...");
        }
        Some(load_cmu_accuracy_benchmark()?)
    } else {
        None
    };

    for model_id in model_ids {
        if !global.quiet {
            println!("Benchmarking model: {}", model_id);
        }

        let result = benchmark_model(
            model_id,
            &test_sentences,
            iterations,
            accuracy_benchmark.as_ref(),
            config,
            global,
        )
        .await?;
        results.push(result);

        if !global.quiet {
            println!("  ✓ Completed\n");
        }
    }

    // Display results
    display_benchmark_results(&results, global);

    // Generate comparison report
    if results.len() > 1 {
        generate_comparison_report(&results, global);
    }

    Ok(())
}

/// Benchmark a single model
async fn benchmark_model(
    model_id: &str,
    test_sentences: &[String],
    iterations: u32,
    accuracy_benchmark: Option<&AccuracyBenchmark>,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<BenchmarkResult> {
    // Load specific model by ID
    if !global.quiet {
        println!("    Loading model: {}", model_id);
    }

    let pipeline = load_model_pipeline(model_id, config, global).await?;

    let synth_config = SynthesisConfig {
        quality: QualityLevel::High,
        ..Default::default()
    };

    let mut total_synthesis_time = Duration::from_secs(0);
    let mut total_audio_duration = 0.0f32;
    let mut successful_runs = 0usize;
    let mut memory_samples = Vec::new();
    let mut audio_health_samples: Vec<AudioHealthSample> = Vec::new();

    // Run benchmark iterations
    for i in 0..iterations {
        if !global.quiet && iterations > 1 {
            print!("  Progress: {}/{}\r", i + 1, iterations);
        }

        for sentence in test_sentences {
            let start_time = Instant::now();

            // Measure memory before synthesis
            let memory_before = get_memory_usage();

            // Attempt synthesis
            match pipeline
                .synthesize_with_config(sentence, &synth_config)
                .await
            {
                Ok(audio) => {
                    let synthesis_time = start_time.elapsed();
                    total_synthesis_time += synthesis_time;
                    total_audio_duration += audio.duration();
                    successful_runs += 1;

                    // Measure memory after synthesis
                    let memory_after = get_memory_usage();
                    memory_samples.push(memory_after - memory_before);

                    // Real, measured DSP signal-health stats from the actual
                    // synthesized audio (see `calculate_quality_score`).
                    audio_health_samples.push(measure_audio_health(&audio));
                }
                Err(e) => {
                    tracing::warn!("Synthesis failed for '{}': {}", sentence, e);
                }
            }
        }
    }

    // Calculate metrics. A model with no real/loadable weights can produce
    // zero successful syntheses -- guard every division so that case reports
    // an honest all-zero/worst-case result instead of panicking (dividing a
    // Duration by zero, or calling `Duration::from_secs_f64` with NaN, both
    // panic) or silently propagating NaN into the displayed report.
    let total_runs = iterations as usize * test_sentences.len();
    let avg_synthesis_time = if total_runs > 0 {
        total_synthesis_time / total_runs as u32
    } else {
        Duration::from_secs(0)
    };
    let avg_audio_duration = if successful_runs > 0 {
        Duration::from_secs_f64(total_audio_duration as f64 / successful_runs as f64)
    } else {
        Duration::from_secs(0)
    };
    let real_time_factor = if avg_audio_duration.as_secs_f64() > 0.0 {
        avg_synthesis_time.as_secs_f64() / avg_audio_duration.as_secs_f64()
    } else {
        // No audio was ever produced: worst-case RTF, not a fabricated "ok"
        // number. `calculate_quality_score`'s performance ladder maps this
        // to its lowest score.
        f64::INFINITY
    };
    let success_rate = if total_runs > 0 {
        successful_runs as f64 / total_runs as f64
    } else {
        0.0
    };
    let avg_memory_usage = if !memory_samples.is_empty() {
        memory_samples.iter().sum::<f64>() / memory_samples.len() as f64
    } else {
        0.0
    };

    // Quality score: a real weighted combination of measured performance
    // (RTF), measured reliability (success rate), and -- when at least one
    // synthesis succeeded -- a real DSP signal-health proxy computed from
    // the actual synthesized audio. No per-architecture guessing by
    // substring-matching `model_id`.
    let audio_health = aggregate_audio_health(&audio_health_samples);
    let quality_score = calculate_quality_score(&real_time_factor, &success_rate, audio_health);

    // Check performance targets
    let latency_target_met = check_latency_target(&avg_synthesis_time);
    let memory_target_met = check_memory_target(avg_memory_usage);

    if !global.quiet {
        println!("    Performance Targets:");
        println!(
            "      Latency (<1ms): {} - {:.2}ms",
            if latency_target_met {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            },
            avg_synthesis_time.as_millis()
        );
        println!(
            "      Memory (<100MB): {} - {:.1}MB",
            if memory_target_met {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            },
            avg_memory_usage
        );
    }

    // Run accuracy benchmark if requested (`accuracy_benchmark.is_some()`
    // just gates whether the user asked for `--accuracy`; the actual test
    // cases used are resolved fresh inside `run_accuracy_test` against
    // whatever language THIS pipeline was really built for -- see its doc
    // comment for why the multi-language `benchmark` value isn't used
    // directly here).
    let (
        phoneme_accuracy,
        word_accuracy,
        accuracy_target_met,
        english_accuracy_target_met,
        japanese_accuracy_target_met,
    ) = if accuracy_benchmark.is_some() {
        if !global.quiet {
            println!("    Running accuracy tests...");
        }

        match run_accuracy_test(&pipeline, global).await {
            Ok(outcome) => {
                if !global.quiet {
                    println!(
                        "    Phoneme Accuracy: {:.2}% (language: {})",
                        outcome.phoneme_accuracy * 100.0,
                        outcome.language.as_str()
                    );
                    println!("    Word Accuracy: {:.2}%", outcome.word_accuracy * 100.0);
                    println!(
                        "    Overall Target Met: {}",
                        if outcome.overall_target_met {
                            "✅"
                        } else {
                            "❌"
                        }
                    );
                    match outcome.english_target_met {
                        Some(en) => println!(
                            "    English Target (>95%): {}",
                            if en { "✅" } else { "❌" }
                        ),
                        None => println!("    English Target: not evaluated (pipeline is not configured for English)"),
                    }
                    match outcome.japanese_target_met {
                        Some(ja) => println!(
                            "    Japanese Target (>90%): {}",
                            if ja { "✅" } else { "❌" }
                        ),
                        None => println!("    Japanese Target: not evaluated (pipeline is not configured for Japanese)"),
                    }
                    println!(
                        "    Synthesis probe: {}/{} probe word(s) synthesized successfully",
                        outcome.synth_successes, outcome.synth_probe_count
                    );
                }
                (
                    Some(outcome.phoneme_accuracy),
                    Some(outcome.word_accuracy),
                    Some(outcome.overall_target_met),
                    outcome.english_target_met,
                    outcome.japanese_target_met,
                )
            }
            Err(e) => {
                if !global.quiet {
                    println!("    Accuracy test failed: {}", e);
                }
                (None, None, None, None, None)
            }
        }
    } else {
        (None, None, None, None, None)
    };

    Ok(BenchmarkResult {
        model_id: model_id.to_string(),
        avg_synthesis_time,
        avg_audio_duration,
        real_time_factor,
        memory_usage_mb: avg_memory_usage,
        quality_score,
        success_rate,
        phoneme_accuracy,
        word_accuracy,
        accuracy_target_met,
        english_accuracy_target_met,
        japanese_accuracy_target_met,
        latency_target_met,
        memory_target_met,
    })
}

/// Get test sentences for benchmarking
fn get_test_sentences() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog.".to_string(),
        "Hello, this is a test of the text-to-speech system.".to_string(),
        "Artificial intelligence is transforming the way we communicate.".to_string(),
        "The weather today is absolutely beautiful with clear skies.".to_string(),
        "Machine learning models require careful tuning and validation.".to_string(),
    ]
}

/// Get current memory usage in MB
fn get_memory_usage() -> f64 {
    // Try multiple methods to get memory usage

    // Method 1: Try reading /proc/self/status on Linux
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<f64>() {
                        return kb / 1024.0; // Convert KB to MB
                    }
                }
            }
        }
    }

    // Method 2: Use rusage on Unix systems
    #[cfg(unix)]
    {
        unsafe {
            let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
            if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) == 0 {
                let usage = usage.assume_init();
                // On Linux, ru_maxrss is in KB; on macOS, it's in bytes
                #[cfg(target_os = "linux")]
                return usage.ru_maxrss as f64 / 1024.0; // KB to MB

                #[cfg(target_os = "macos")]
                return usage.ru_maxrss as f64 / (1024.0 * 1024.0); // bytes to MB
            }
        }
    }

    // Method 3: Try reading /proc/meminfo for available memory on Linux
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        let mut total_kb = None;
        let mut available_kb = None;

        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<f64>() {
                        total_kb = Some(kb);
                    }
                }
            } else if line.starts_with("MemAvailable:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<f64>() {
                        available_kb = Some(kb);
                    }
                }
            }
        }

        if let (Some(total), Some(available)) = (total_kb, available_kb) {
            let used_mb = (total - available) / 1024.0; // Convert KB to MB
            return used_mb;
        }
    }

    // Fallback: Return a placeholder value if all methods fail
    // This ensures the benchmark can still run even if memory monitoring fails
    50.0 // Default 50MB estimate
}

/// Real, measured DSP signal-health signals for one synthesized audio
/// buffer. Never fabricated: every field is computed directly from the
/// actual samples produced by the pipeline under test.
#[derive(Debug, Clone, Copy)]
struct AudioHealthSample {
    /// Whether the signal clips (any sample at or above the clip threshold).
    clipped: bool,
    /// Whether the signal is effectively silent (RMS below an audible
    /// threshold), which for a non-empty synthesis request indicates broken
    /// output rather than genuine silence.
    near_silent: bool,
    /// Spectral flatness (0 = tonal/structured, 1 = white-noise-like),
    /// `None` when the buffer is too short for the FFT window used.
    spectral_flatness: Option<f32>,
}

/// FFT window used for the spectral-flatness signal-health component. Short
/// synthesized clips (e.g. very short probe words) may fall below this,
/// which is handled by omitting the spectral component rather than
/// substituting a value (see `aggregate_audio_health`).
const AUDIO_HEALTH_FFT_SIZE: usize = 1024;

/// Measure real, objective DSP signal-health stats from actual synthesized
/// audio. This is a technical signal proxy (clipping / silence / spectral
/// flatness), NOT a perceptual quality or MOS prediction.
fn measure_audio_health(audio: &AudioBuffer) -> AudioHealthSample {
    let clipped = audio.is_clipped(0.999);
    let rms = audio.rms();
    // Roughly -60 dBFS: for a non-empty synthesis request this indicates
    // dropped/empty output rather than intentional quiet audio.
    let near_silent = rms < 0.001;

    let spectral_flatness = if audio.len() >= AUDIO_HEALTH_FFT_SIZE {
        dsp::spectral_statistics(audio, AUDIO_HEALTH_FFT_SIZE)
            .ok()
            .map(|stats| stats.flatness)
    } else {
        None
    };

    AudioHealthSample {
        clipped,
        near_silent,
        spectral_flatness,
    }
}

/// Aggregate per-run audio-health samples into a single `[0.0, 1.0]` signal
/// (higher = healthier), or `None` if there is nothing to measure (zero
/// successful syntheses). When the spectral-flatness component could not be
/// computed for ANY sample (e.g. every synthesized clip was shorter than the
/// FFT window), that component is excluded from the average entirely rather
/// than substituted with a constant -- a missing measurement must not be
/// disguised as a measured one.
fn aggregate_audio_health(samples: &[AudioHealthSample]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }

    let clip_rate = samples.iter().filter(|s| s.clipped).count() as f64 / samples.len() as f64;
    let silence_rate =
        samples.iter().filter(|s| s.near_silent).count() as f64 / samples.len() as f64;

    let mut components = vec![1.0 - clip_rate, 1.0 - silence_rate];

    let flatness_values: Vec<f64> = samples
        .iter()
        .filter_map(|s| s.spectral_flatness)
        .map(f64::from)
        .collect();
    if !flatness_values.is_empty() {
        let avg_flatness = flatness_values.iter().sum::<f64>() / flatness_values.len() as f64;
        // Lower flatness (more tonal/structured, as real speech is) scores
        // higher; flatness near 1.0 (white-noise-like) scores near zero.
        components.push((1.0 - avg_flatness).clamp(0.0, 1.0));
    }

    Some(components.iter().sum::<f64>() / components.len() as f64)
}

/// Calculate quality score based on real, measured metrics.
///
/// `real_time_factor` and `success_rate` are real measurements from actually
/// running the model under test. `audio_health` (see
/// [`aggregate_audio_health`]) is a real, measured DSP signal-health proxy
/// derived from the model's ACTUAL synthesized audio -- never a
/// per-architecture guess keyed off the model's name string. When no audio
/// was ever successfully synthesized, the health component is dropped
/// entirely (not substituted with a constant) and the score reduces to the
/// average of the performance and reliability components.
///
/// This is a composite performance/signal-health indicator, not a
/// perceptual quality or MOS prediction.
fn calculate_quality_score(
    real_time_factor: &f64,
    success_rate: &f64,
    audio_health: Option<f64>,
) -> f64 {
    // Performance score: Logarithmic scale for better granularity
    // RTF < 0.05 is exceptional, 0.05-0.1 is excellent, 0.1-0.5 is good, 0.5-1.0 is acceptable
    let performance_score = if *real_time_factor < 0.05 {
        5.0
    } else if *real_time_factor < 0.1 {
        4.5 + 0.5 * (0.1 - real_time_factor) / 0.05 // 4.5-5.0
    } else if *real_time_factor < 0.25 {
        3.5 + 1.0 * (0.25 - real_time_factor) / 0.15 // 3.5-4.5
    } else if *real_time_factor < 0.5 {
        2.5 + 1.0 * (0.5 - real_time_factor) / 0.25 // 2.5-3.5
    } else if *real_time_factor < 1.0 {
        1.5 + 1.0 * (1.0 - real_time_factor) / 0.5 // 1.5-2.5
    } else if *real_time_factor < 2.0 {
        0.5 + 1.0 * (2.0 - real_time_factor) / 1.0 // 0.5-1.5
    } else {
        (5.0 / real_time_factor).min(0.5) // Decreasing score for very slow models
    };

    // Reliability score: Non-linear scaling emphasizing high success rates
    let reliability_score = if *success_rate >= 0.99 {
        5.0
    } else if *success_rate >= 0.95 {
        4.0 + 1.0 * (success_rate - 0.95) / 0.04 // 4.0-5.0
    } else if *success_rate >= 0.90 {
        3.0 + 1.0 * (success_rate - 0.90) / 0.05 // 3.0-4.0
    } else if *success_rate >= 0.75 {
        1.5 + 1.5 * (success_rate - 0.75) / 0.15 // 1.5-3.0
    } else {
        success_rate * 2.0 // 0.0-1.5
    };

    match audio_health {
        Some(health) => {
            let health_score = (health * 5.0).clamp(0.0, 5.0);
            // Weighted average: 40% performance, 40% reliability, 20% real
            // measured audio signal-health.
            (performance_score * 0.4 + reliability_score * 0.4 + health_score * 0.2).clamp(0.0, 5.0)
        }
        // No successful synthesis to measure audio health from: do not
        // substitute a fabricated value, just drop that component.
        None => ((performance_score + reliability_score) / 2.0).clamp(0.0, 5.0),
    }
}

/// Display benchmark results
fn display_benchmark_results(results: &[BenchmarkResult], global: &GlobalOptions) {
    if global.quiet {
        return;
    }

    println!("Benchmark Results:");
    println!("==================");

    for result in results {
        println!("\nModel: {}", result.model_id);
        println!(
            "  Avg Synthesis Time: {:.2}ms",
            result.avg_synthesis_time.as_millis()
        );
        println!(
            "  Avg Audio Duration: {:.2}ms",
            result.avg_audio_duration.as_millis()
        );
        println!("  Real-time Factor: {:.2}x", result.real_time_factor);
        println!("  Memory Usage: {:.1} MB", result.memory_usage_mb);
        println!(
            "  Quality Score: {:.1}/5.0 (performance + reliability + measured audio signal-health; NOT a perceptual/MOS quality prediction)",
            result.quality_score
        );
        println!("  Success Rate: {:.1}%", result.success_rate * 100.0);

        // Display accuracy metrics if available
        if let Some(phoneme_acc) = result.phoneme_accuracy {
            println!("  Phoneme Accuracy: {:.2}%", phoneme_acc * 100.0);
        }
        if let Some(word_acc) = result.word_accuracy {
            println!("  Word Accuracy: {:.2}%", word_acc * 100.0);
        }
        if let Some(target_met) = result.accuracy_target_met {
            println!("  Accuracy Targets:");
            if let Some(en_target) = result.english_accuracy_target_met {
                println!(
                    "    English (>95%): {}",
                    if en_target {
                        "✅ PASSED"
                    } else {
                        "❌ FAILED"
                    }
                );
            }
            if let Some(ja_target) = result.japanese_accuracy_target_met {
                println!(
                    "    Japanese (>90%): {}",
                    if ja_target {
                        "✅ PASSED"
                    } else {
                        "❌ FAILED"
                    }
                );
            }
            println!(
                "    Overall: {}",
                if target_met {
                    "✅ PASSED"
                } else {
                    "❌ FAILED"
                }
            );
        }

        // Display performance targets
        println!(
            "  Latency Target (<1ms): {}",
            if result.latency_target_met {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            }
        );
        println!(
            "  Memory Target (<100MB): {}",
            if result.memory_target_met {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            }
        );
    }
}

/// Generate comparison report
fn generate_comparison_report(results: &[BenchmarkResult], global: &GlobalOptions) {
    if global.quiet {
        return;
    }

    println!("\n\nComparison Report:");
    println!("==================");

    // Find best performers
    let fastest = results.iter().min_by(|a, b| {
        a.real_time_factor
            .partial_cmp(&b.real_time_factor)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let most_reliable = results.iter().max_by(|a, b| {
        a.success_rate
            .partial_cmp(&b.success_rate)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let highest_quality = results.iter().max_by(|a, b| {
        a.quality_score
            .partial_cmp(&b.quality_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let most_efficient = results.iter().min_by(|a, b| {
        a.memory_usage_mb
            .partial_cmp(&b.memory_usage_mb)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(model) = fastest {
        println!(
            "🏃 Fastest Model: {} ({:.2}x real-time)",
            model.model_id, model.real_time_factor
        );
    }

    if let Some(model) = most_reliable {
        println!(
            "🎯 Most Reliable: {} ({:.1}% success rate)",
            model.model_id,
            model.success_rate * 100.0
        );
    }

    if let Some(model) = highest_quality {
        println!(
            "⭐ Highest Quality Score: {} ({:.1}/5.0, performance+reliability+signal-health)",
            model.model_id, model.quality_score
        );
    }

    if let Some(model) = most_efficient {
        println!(
            "💾 Most Memory Efficient: {} ({:.1} MB)",
            model.model_id, model.memory_usage_mb
        );
    }

    // Performance target analysis
    println!("\n📊 Performance Target Analysis:");
    let models_meeting_latency = results.iter().filter(|r| r.latency_target_met).count();
    let models_meeting_memory = results.iter().filter(|r| r.memory_target_met).count();
    let models_meeting_all_targets = results
        .iter()
        .filter(|r| {
            r.latency_target_met && r.memory_target_met && r.accuracy_target_met.unwrap_or(false)
        })
        .count();

    println!(
        "  🚀 Models meeting latency target (<1ms): {}/{}",
        models_meeting_latency,
        results.len()
    );
    println!(
        "  🧠 Models meeting memory target (<100MB): {}/{}",
        models_meeting_memory,
        results.len()
    );

    if results.iter().any(|r| r.accuracy_target_met.is_some()) {
        println!(
            "  🎯 Models meeting all targets: {}/{}",
            models_meeting_all_targets,
            results.len()
        );

        if models_meeting_all_targets > 0 {
            println!("  ✅ Production-ready models found!");
        } else {
            println!("  ⚠️  No models currently meet all production targets");
        }
    }
}

/// Load a pipeline configured for a specific model
async fn load_model_pipeline(
    model_id: &str,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<VoirsPipeline> {
    // Check if model exists in cache
    let cache_dir = config.pipeline.effective_cache_dir();
    let model_path = cache_dir.join("models").join(model_id);

    if !model_path.exists() {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Model '{}' not found in cache. Please download it first using 'voirs download-model {}'",
            model_id, model_id
        )));
    }

    // Load model configuration
    let model_config_path = model_path.join("config.json");
    let model_config = if model_config_path.exists() {
        let config_content = std::fs::read_to_string(&model_config_path).map_err(|e| {
            voirs_sdk::VoirsError::IoError {
                path: model_config_path.clone(),
                operation: voirs_sdk::error::IoOperation::Read,
                source: e,
            }
        })?;

        serde_json::from_str::<ModelMetadata>(&config_content).map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!(
                "Invalid model config for '{}': {}",
                model_id, e
            ))
        })?
    } else {
        // Create default model metadata if config doesn't exist
        ModelMetadata {
            id: model_id.to_string(),
            name: model_id.to_string(),
            description: format!("Model {}", model_id),
            model_type: "neural".to_string(),
            quality: QualityLevel::High,
            requires_gpu: false,
            memory_requirements_mb: 512,
            acoustic_model: "model.safetensors".to_string(),
            vocoder_model: "vocoder.safetensors".to_string(),
            g2p_model: None,
        }
    };

    if !global.quiet {
        println!(
            "      Model: {} ({})",
            model_config.name, model_config.description
        );
        println!(
            "      Type: {}, Quality: {:?}",
            model_config.model_type, model_config.quality
        );
        println!(
            "      Memory required: {} MB",
            model_config.memory_requirements_mb
        );
    }

    // Check memory requirements
    let available_memory = get_memory_usage(); // This gets current usage, we need available
    if model_config.memory_requirements_mb as f64 > available_memory {
        tracing::warn!(
            "Model '{}' requires {} MB but only {:.1} MB may be available",
            model_id,
            model_config.memory_requirements_mb,
            available_memory
        );
    }

    // Verify model files exist
    let acoustic_path = model_path.join(&model_config.acoustic_model);
    let vocoder_path = model_path.join(&model_config.vocoder_model);

    if !acoustic_path.exists() {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Acoustic model file not found: {}",
            acoustic_path.display()
        )));
    }

    if !vocoder_path.exists() {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Vocoder model file not found: {}",
            vocoder_path.display()
        )));
    }

    // Build pipeline with model-specific configuration
    let mut builder = VoirsPipeline::builder().with_quality(model_config.quality);

    // Configure GPU usage based on model requirements and system capabilities
    if model_config.requires_gpu && (config.pipeline.use_gpu || global.gpu) {
        builder = builder.with_gpu_acceleration(true);
        if !global.quiet {
            println!("      GPU acceleration: enabled (required by model)");
        }
    } else if config.pipeline.use_gpu || global.gpu {
        builder = builder.with_gpu_acceleration(true);
        if !global.quiet {
            println!("      GPU acceleration: enabled");
        }
    } else if !global.quiet {
        println!("      GPU acceleration: disabled");
    }

    // Set thread count based on configuration
    if let Some(threads) = config.pipeline.num_threads {
        builder = builder.with_threads(threads);
        if !global.quiet {
            println!("      Threads: {}", threads);
        }
    } else {
        let default_threads = config.pipeline.effective_thread_count();
        builder = builder.with_threads(default_threads);
        if !global.quiet {
            println!("      Threads: {} (auto)", default_threads);
        }
    }

    // Build the pipeline
    let pipeline = builder.build().await.map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to load model '{}': {}", model_id, e))
    })?;

    if !global.quiet {
        println!("      ✓ Model loaded successfully");
    }

    Ok(pipeline)
}

/// Canonical CMU-style G2P test set: `(word, expected IPA phonemes,
/// language)`. This is the single source of truth for accuracy test data --
/// shared by [`load_cmu_accuracy_benchmark`] (used for an upfront sanity
/// check that the test corpus loads) and [`run_accuracy_test`] (which builds
/// its own language-filtered subset from it, see that function's doc
/// comment for why).
fn cmu_test_words() -> Vec<(&'static str, Vec<&'static str>, LanguageCode)> {
    vec![
        // Basic phoneme coverage
        ("hello", vec!["h", "ə", "ˈl", "oʊ"], LanguageCode::EnUs),
        ("world", vec!["w", "ɜːr", "l", "d"], LanguageCode::EnUs),
        ("cat", vec!["k", "æ", "t"], LanguageCode::EnUs),
        ("dog", vec!["d", "ɔː", "ɡ"], LanguageCode::EnUs),
        ("house", vec!["h", "aʊ", "s"], LanguageCode::EnUs),
        ("tree", vec!["t", "r", "iː"], LanguageCode::EnUs),
        ("water", vec!["ˈw", "ɔː", "t", "ər"], LanguageCode::EnUs),
        ("phone", vec!["f", "oʊ", "n"], LanguageCode::EnUs),
        // Vowel patterns
        ("beat", vec!["b", "iː", "t"], LanguageCode::EnUs),
        ("bit", vec!["b", "ɪ", "t"], LanguageCode::EnUs),
        ("bet", vec!["b", "ɛ", "t"], LanguageCode::EnUs),
        ("bat", vec!["b", "æ", "t"], LanguageCode::EnUs),
        ("bot", vec!["b", "ɑː", "t"], LanguageCode::EnUs),
        ("boat", vec!["b", "oʊ", "t"], LanguageCode::EnUs),
        ("boot", vec!["b", "uː", "t"], LanguageCode::EnUs),
        ("but", vec!["b", "ʌ", "t"], LanguageCode::EnUs),
        // Consonant clusters
        ("street", vec!["s", "t", "r", "iː", "t"], LanguageCode::EnUs),
        ("spring", vec!["s", "p", "r", "ɪ", "ŋ"], LanguageCode::EnUs),
        ("school", vec!["s", "k", "uː", "l"], LanguageCode::EnUs),
        ("throw", vec!["θ", "r", "oʊ"], LanguageCode::EnUs),
        ("three", vec!["θ", "r", "iː"], LanguageCode::EnUs),
        // Irregular words
        ("one", vec!["w", "ʌ", "n"], LanguageCode::EnUs),
        ("two", vec!["t", "uː"], LanguageCode::EnUs),
        ("eight", vec!["eɪ", "t"], LanguageCode::EnUs),
        ("through", vec!["θ", "r", "uː"], LanguageCode::EnUs),
        ("though", vec!["ð", "oʊ"], LanguageCode::EnUs),
        ("rough", vec!["r", "ʌ", "f"], LanguageCode::EnUs),
        // Complex multisyllabic words
        (
            "computer",
            vec!["k", "ə", "m", "ˈp", "j", "uː", "t", "ər"],
            LanguageCode::EnUs,
        ),
        (
            "beautiful",
            vec!["ˈb", "j", "uː", "t", "ɪ", "f", "ə", "l"],
            LanguageCode::EnUs,
        ),
        (
            "restaurant",
            vec!["ˈr", "ɛ", "s", "t", "ər", "ɑː", "n", "t"],
            LanguageCode::EnUs,
        ),
        (
            "university",
            vec!["j", "uː", "n", "ɪ", "ˈv", "ɜːr", "s", "ə", "t", "i"],
            LanguageCode::EnUs,
        ),
        (
            "pronunciation",
            vec!["p", "r", "ə", "n", "ʌ", "n", "s", "i", "ˈeɪ", "ʃ", "ə", "n"],
            LanguageCode::EnUs,
        ),
        // Names and proper nouns
        (
            "california",
            vec!["k", "æ", "l", "ɪ", "ˈf", "ɔːr", "n", "j", "ə"],
            LanguageCode::EnUs,
        ),
        (
            "washington",
            vec!["ˈw", "ɑː", "ʃ", "ɪ", "ŋ", "t", "ə", "n"],
            LanguageCode::EnUs,
        ),
        (
            "america",
            vec!["ə", "ˈm", "ɛr", "ɪ", "k", "ə"],
            LanguageCode::EnUs,
        ),
        // Technical/scientific terms
        (
            "technology",
            vec!["t", "ɛ", "k", "ˈn", "ɑː", "l", "ə", "dʒ", "i"],
            LanguageCode::EnUs,
        ),
        (
            "artificial",
            vec!["ɑːr", "t", "ɪ", "ˈf", "ɪ", "ʃ", "ə", "l"],
            LanguageCode::EnUs,
        ),
        (
            "intelligence",
            vec!["ɪ", "n", "ˈt", "ɛ", "l", "ɪ", "dʒ", "ə", "n", "s"],
            LanguageCode::EnUs,
        ),
        (
            "synthesis",
            vec!["ˈs", "ɪ", "n", "θ", "ə", "s", "ɪ", "s"],
            LanguageCode::EnUs,
        ),
        // Japanese test cases (using romaji for simplicity)
        (
            "こんにちは",
            vec!["k", "o", "n", "n", "i", "ch", "i", "w", "a"],
            LanguageCode::Ja,
        ),
        (
            "ありがとう",
            vec!["a", "r", "i", "g", "a", "t", "o", "u"],
            LanguageCode::Ja,
        ),
        (
            "おはよう",
            vec!["o", "h", "a", "y", "o", "u"],
            LanguageCode::Ja,
        ),
        (
            "さよなら",
            vec!["s", "a", "y", "o", "n", "a", "r", "a"],
            LanguageCode::Ja,
        ),
        (
            "コンピュータ",
            vec!["k", "o", "n", "p", "y", "u", "u", "t", "a"],
            LanguageCode::Ja,
        ),
        (
            "テクノロジー",
            vec!["t", "e", "k", "u", "n", "o", "r", "o", "j", "i", "i"],
            LanguageCode::Ja,
        ),
        (
            "アニメーション",
            vec!["a", "n", "i", "m", "e", "e", "sh", "o", "n"],
            LanguageCode::Ja,
        ),
        (
            "大学",
            vec!["d", "a", "i", "g", "a", "k", "u"],
            LanguageCode::Ja,
        ),
        (
            "東京",
            vec!["t", "o", "u", "k", "y", "o", "u"],
            LanguageCode::Ja,
        ),
        (
            "日本語",
            vec!["n", "i", "h", "o", "n", "g", "o"],
            LanguageCode::Ja,
        ),
    ]
}

/// Load the full (all-languages) CMU accuracy benchmark, used only as an
/// upfront sanity check that the test corpus is well-formed before running
/// potentially slow model benchmarks. [`run_accuracy_test`] does NOT
/// evaluate against this multi-language set directly -- it builds its own
/// language-filtered subset from [`cmu_test_words`].
fn load_cmu_accuracy_benchmark() -> Result<AccuracyBenchmark> {
    let mut benchmark = AccuracyBenchmark::new();
    for (word, phonemes, lang) in cmu_test_words() {
        benchmark.add_test_case(TestCase {
            word: word.to_string(),
            expected_phonemes: phonemes.into_iter().map(|p| p.to_string()).collect(),
            language: lang,
        });
    }
    Ok(benchmark)
}

/// Map an SDK language code onto the `voirs-g2p` language code used by the
/// real G2P backends.
///
/// This mirrors `voirs_sdk::pipeline::init::PipelineInitializer::g2p_language`,
/// which is `pub(crate)` inside voirs-sdk and therefore not reachable from
/// this crate. Keep in sync if that mapping changes.
fn sdk_language_to_g2p_language(language: voirs_sdk::types::LanguageCode) -> Option<LanguageCode> {
    use voirs_sdk::types::LanguageCode as Sdk;

    Some(match language {
        Sdk::EnUs => LanguageCode::EnUs,
        Sdk::EnGb => LanguageCode::EnGb,
        Sdk::JaJp | Sdk::Ja => LanguageCode::Ja,
        Sdk::DeDe | Sdk::De => LanguageCode::De,
        Sdk::FrFr | Sdk::Fr => LanguageCode::Fr,
        Sdk::EsEs | Sdk::EsMx | Sdk::Es => LanguageCode::Es,
        Sdk::ItIt | Sdk::It => LanguageCode::It,
        Sdk::PtBr | Sdk::Pt => LanguageCode::Pt,
        Sdk::ZhCn => LanguageCode::ZhCn,
        Sdk::KoKr | Sdk::Ko => LanguageCode::Ko,
        Sdk::RuRu | Sdk::Ru => LanguageCode::Ru,
        Sdk::Ar => LanguageCode::Ar,
        _ => return None,
    })
}

/// Outcome of a real accuracy test run against one model's pipeline.
struct AccuracyTestOutcome {
    /// Resolved G2P language the pipeline was actually evaluated against.
    language: LanguageCode,
    phoneme_accuracy: f64,
    word_accuracy: f64,
    overall_target_met: bool,
    /// `None` when the pipeline was not configured for English (the target
    /// was simply not evaluated -- distinct from evaluated-and-failed).
    english_target_met: Option<bool>,
    /// `None` when the pipeline was not configured for Japanese.
    japanese_target_met: Option<bool>,
    synth_successes: usize,
    synth_probe_count: usize,
}

/// Maximum number of CMU words actually synthesized through the real
/// pipeline as a robustness probe (see doc comment on [`run_accuracy_test`]).
/// Kept small: this exercises real synthesis, which is not free.
const ACCURACY_SYNTH_PROBE_LIMIT: usize = 8;

/// Run a real accuracy test against the model's pipeline.
///
/// Two things are measured, both grounded in real behavior for the specific
/// model under test:
///
/// 1. **Phoneme/word accuracy**, evaluated against the REAL production
///    rule-based G2P backend (`voirs_g2p::backends::rule_based::RuleBasedG2p`)
///    for whichever language `pipeline` was actually built with (queried via
///    the public `pipeline.get_config()`, then mapped the same way
///    `voirs-sdk`'s own `PipelineInitializer::load_g2p` resolves it --
///    `rule_based` is currently the only implemented G2P backend, and the
///    SDK itself refuses to build a pipeline naming any other backend). We
///    cannot literally borrow the pipeline's own `Arc<dyn G2p>`
///    (`VoirsPipeline::g2p()` is `pub(crate)` in voirs-sdk), so this
///    constructs an equivalent instance through the public `voirs_g2p` API --
///    this is the real production phoneme-rule engine, not a placeholder.
///    Only the test cases for the RESOLVED language are evaluated: mixing in
///    another language's words would silently run e.g. Japanese text through
///    English phonological rules and report the resulting near-zero accuracy
///    as if it reflected the model, which is exactly the kind of
///    model-independent, misleading number this rewrite exists to remove.
/// 2. **Pipeline synthesis health**: a bounded probe (see
///    [`ACCURACY_SYNTH_PROBE_LIMIT`]) of real words in the resolved language
///    are actually run through `pipeline.synthesize()`. A model that cannot
///    even produce audio for these simple inputs fails the overall target
///    regardless of its G2P accuracy.
///
/// Returns `Err` if the pipeline's resolved language has no G2P ruleset or
/// no matching test cases, rather than fabricating a pass/fail verdict for a
/// language nothing here actually covers.
async fn run_accuracy_test(
    pipeline: &VoirsPipeline,
    global: &GlobalOptions,
) -> Result<AccuracyTestOutcome> {
    // 1. Resolve the language this SPECIFIC pipeline's G2P was actually built
    //    for (mirrors voirs-sdk's own internal resolution).
    let pipeline_config = pipeline.get_config().await;
    let sdk_language = pipeline_config
        .language_code
        .unwrap_or(pipeline_config.default_synthesis.language);
    let g2p_language = sdk_language_to_g2p_language(sdk_language).ok_or_else(|| {
        voirs_sdk::VoirsError::config_error(format!(
            "Accuracy benchmarking is not available: pipeline is configured for language \
             {sdk_language:?}, which has no voirs-g2p ruleset."
        ))
    })?;

    // 2. Build the real production G2P backend for that language and
    //    evaluate ONLY the test cases that apply to it.
    let g2p = RuleBasedG2p::new(g2p_language);

    let mut filtered_benchmark = AccuracyBenchmark::new();
    let mut probe_words: Vec<String> = Vec::new();
    for (word, phonemes, lang) in cmu_test_words() {
        if lang == g2p_language {
            if probe_words.len() < ACCURACY_SYNTH_PROBE_LIMIT {
                probe_words.push(word.to_string());
            }
            filtered_benchmark.add_test_case(TestCase {
                word: word.to_string(),
                expected_phonemes: phonemes.into_iter().map(|p| p.to_string()).collect(),
                language: lang,
            });
        }
    }

    if filtered_benchmark.test_case_count() == 0 {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "No accuracy test cases available for resolved language {g2p_language:?}"
        )));
    }

    let metrics = filtered_benchmark
        .evaluate(&g2p)
        .await
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Accuracy test failed: {}", e)))?;

    // 3. Exercise the REAL pipeline: actually synthesize the probe words.
    if !global.quiet {
        println!(
            "    Probing real synthesis for {} word(s) in {}...",
            probe_words.len(),
            g2p_language.as_str()
        );
    }
    let mut synth_successes = 0usize;
    for word in &probe_words {
        if pipeline.synthesize(word).await.is_ok() {
            synth_successes += 1;
        }
    }
    let synth_success_rate = if probe_words.is_empty() {
        0.0
    } else {
        synth_successes as f64 / probe_words.len() as f64
    };

    // Accuracy targets: English >95%, Japanese >90% (project requirements),
    // combined with a real synthesis-health gate -- a model whose G2P scores
    // well on paper but cannot actually synthesize the same words has not
    // met the target.
    let accuracy_threshold = match g2p_language {
        LanguageCode::EnUs | LanguageCode::EnGb => 0.95,
        LanguageCode::Ja => 0.90,
        _ => 0.80,
    };
    let language_target_met =
        metrics.phoneme_accuracy >= accuracy_threshold && synth_success_rate >= 0.8;

    let (english_target_met, japanese_target_met) = match g2p_language {
        LanguageCode::EnUs | LanguageCode::EnGb => (Some(language_target_met), None),
        LanguageCode::Ja => (None, Some(language_target_met)),
        _ => (None, None),
    };

    Ok(AccuracyTestOutcome {
        language: g2p_language,
        phoneme_accuracy: metrics.phoneme_accuracy,
        word_accuracy: metrics.word_accuracy,
        overall_target_met: language_target_met,
        english_target_met,
        japanese_target_met,
        synth_successes,
        synth_probe_count: probe_words.len(),
    })
}

/// Check if latency target is met (<1ms for typical sentences)
fn check_latency_target(avg_synthesis_time: &Duration) -> bool {
    avg_synthesis_time.as_millis() < 1
}

/// Check if memory target is met (<100MB)
fn check_memory_target(memory_usage_mb: f64) -> bool {
    memory_usage_mb < 100.0
}

/// Model metadata structure
#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub description: String,
    pub model_type: String,
    pub quality: QualityLevel,
    pub requires_gpu: bool,
    pub memory_requirements_mb: u32,
    pub acoustic_model: String,
    pub vocoder_model: String,
    pub g2p_model: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_global() -> GlobalOptions {
        GlobalOptions {
            config: None,
            verbose: 0,
            quiet: true,
            format: None,
            voice: None,
            gpu: false,
            threads: None,
        }
    }

    #[test]
    fn test_get_test_sentences() {
        let sentences = get_test_sentences();
        assert!(!sentences.is_empty());
        assert!(sentences.len() >= 3);
    }

    #[test]
    fn test_calculate_quality_score_in_range() {
        let score = calculate_quality_score(&0.5, &1.0, Some(0.9));
        assert!((0.0..=5.0).contains(&score));

        let score_no_health = calculate_quality_score(&0.5, &1.0, None);
        assert!((0.0..=5.0).contains(&score_no_health));
    }

    /// Regression test: the quality score must be driven by REAL measured
    /// inputs, not a per-architecture guess keyed off a model name string
    /// (there is no `model_id` parameter any more). Verify the score
    /// actually varies when the real inputs vary.
    #[test]
    fn test_calculate_quality_score_varies_with_real_inputs() {
        let fast_reliable_healthy = calculate_quality_score(&0.02, &1.0, Some(1.0));
        let slow_unreliable_unhealthy = calculate_quality_score(&5.0, &0.1, Some(0.0));
        assert!(
            fast_reliable_healthy > slow_unreliable_unhealthy,
            "a fast, reliable, healthy result must score higher than a slow, unreliable, \
             unhealthy one: {fast_reliable_healthy} vs {slow_unreliable_unhealthy}"
        );

        // Audio health specifically must move the score for otherwise
        // identical performance/reliability inputs.
        let healthy = calculate_quality_score(&0.2, &0.98, Some(1.0));
        let unhealthy = calculate_quality_score(&0.2, &0.98, Some(0.0));
        assert!(
            healthy > unhealthy,
            "healthier real audio signal must score higher: {healthy} vs {unhealthy}"
        );
    }

    #[test]
    fn test_aggregate_audio_health_no_samples_is_none() {
        assert!(aggregate_audio_health(&[]).is_none());
    }

    #[test]
    fn test_aggregate_audio_health_clean_signal_scores_high() {
        let samples = vec![
            AudioHealthSample {
                clipped: false,
                near_silent: false,
                spectral_flatness: Some(0.1),
            };
            5
        ];
        let health = aggregate_audio_health(&samples).expect("must have a value");
        assert!(
            health > 0.8,
            "clean, non-clipping, non-silent audio should score high: {health}"
        );
    }

    #[test]
    fn test_aggregate_audio_health_clipping_and_silence_score_low() {
        let samples = vec![
            AudioHealthSample {
                clipped: true,
                near_silent: true,
                spectral_flatness: Some(0.9),
            };
            5
        ];
        let health = aggregate_audio_health(&samples).expect("must have a value");
        assert!(
            health < 0.3,
            "clipping, silent, noisy audio should score low: {health}"
        );
    }

    /// Regression test: when NO sample has a spectral-flatness measurement
    /// (e.g. every synthesized clip was too short for the FFT window), that
    /// component must be excluded from the average, not silently replaced
    /// with a constant. Compare against an otherwise-identical set of
    /// samples that DO have flatness data to make sure the two are computed
    /// via genuinely different code paths (different component counts).
    #[test]
    fn test_aggregate_audio_health_excludes_missing_spectral_component() {
        let without_flatness = vec![
            AudioHealthSample {
                clipped: false,
                near_silent: false,
                spectral_flatness: None,
            };
            3
        ];
        let with_perfect_flatness = vec![
            AudioHealthSample {
                clipped: false,
                near_silent: false,
                spectral_flatness: Some(0.0),
            };
            3
        ];

        let health_without = aggregate_audio_health(&without_flatness).unwrap();
        let health_with = aggregate_audio_health(&with_perfect_flatness).unwrap();

        // Both must be valid real [0,1] numbers, but they must not be
        // silently forced to the exact same value by a hardcoded fallback:
        // omitting the spectral component (2-way average of clip/silence,
        // both perfect => 1.0) differs from including a perfect (flatness=0
        // => component=1.0) spectral component (3-way average, also 1.0 in
        // this specific case) -- so instead we check a case where they
        // WOULD differ if a wrong constant were substituted.
        assert!((health_without - 1.0).abs() < 1e-9);
        assert!((health_with - 1.0).abs() < 1e-9);

        // Now use a non-trivial flatness value to prove the component is
        // really being averaged in, not ignored or replaced by a constant.
        let with_bad_flatness = vec![
            AudioHealthSample {
                clipped: false,
                near_silent: false,
                spectral_flatness: Some(1.0), // worst-case: white-noise-like
            };
            3
        ];
        let health_bad_flatness = aggregate_audio_health(&with_bad_flatness).unwrap();
        assert!(
            health_bad_flatness < health_without,
            "a real bad spectral-flatness measurement must pull the score down when present \
             ({health_bad_flatness}), unlike when it's absent entirely ({health_without})"
        );
    }

    #[test]
    fn test_measure_audio_health_detects_clipping() {
        let clipped_audio =
            AudioBuffer::mono(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], 16000);
        let sample = measure_audio_health(&clipped_audio);
        assert!(sample.clipped);
    }

    #[test]
    fn test_measure_audio_health_detects_silence() {
        let silent_audio = AudioBuffer::mono(vec![0.0; 4000], 16000);
        let sample = measure_audio_health(&silent_audio);
        assert!(sample.near_silent);
        assert!(!sample.clipped);
    }

    #[test]
    fn test_measure_audio_health_short_buffer_has_no_flatness() {
        // Shorter than AUDIO_HEALTH_FFT_SIZE: the spectral component must be
        // genuinely omitted (`None`), not defaulted.
        let short_audio = AudioBuffer::mono(vec![0.5; 16], 16000);
        let sample = measure_audio_health(&short_audio);
        assert!(sample.spectral_flatness.is_none());
    }

    #[test]
    fn test_get_memory_usage() {
        let usage = get_memory_usage();
        assert!(usage >= 0.0);
    }

    #[test]
    fn test_check_latency_target() {
        // Test passing latency (under 1ms)
        let fast_duration = Duration::from_micros(500);
        assert!(check_latency_target(&fast_duration));

        // Test failing latency (over 1ms)
        let slow_duration = Duration::from_millis(2);
        assert!(!check_latency_target(&slow_duration));

        // Test edge case (exactly 1ms)
        let edge_duration = Duration::from_millis(1);
        assert!(!check_latency_target(&edge_duration));
    }

    #[test]
    fn test_check_memory_target() {
        // Test passing memory (under 100MB)
        assert!(check_memory_target(50.0));

        // Test failing memory (over 100MB)
        assert!(!check_memory_target(150.0));

        // Test edge case (exactly 100MB)
        assert!(!check_memory_target(100.0));
    }

    #[test]
    fn test_sdk_language_to_g2p_language_mapping() {
        assert_eq!(
            sdk_language_to_g2p_language(voirs_sdk::types::LanguageCode::EnUs),
            Some(LanguageCode::EnUs)
        );
        assert_eq!(
            sdk_language_to_g2p_language(voirs_sdk::types::LanguageCode::JaJp),
            Some(LanguageCode::Ja)
        );
        assert_eq!(
            sdk_language_to_g2p_language(voirs_sdk::types::LanguageCode::Ja),
            Some(LanguageCode::Ja)
        );
        // A language voirs-g2p genuinely has no ruleset for must map to
        // `None` (fail-closed upstream), not silently alias to English.
        assert_eq!(
            sdk_language_to_g2p_language(voirs_sdk::types::LanguageCode::Th),
            None
        );
    }

    #[test]
    fn test_cmu_test_words_language_filtering() {
        let words = cmu_test_words();
        let english_count = words
            .iter()
            .filter(|(_, _, l)| *l == LanguageCode::EnUs)
            .count();
        let japanese_count = words
            .iter()
            .filter(|(_, _, l)| *l == LanguageCode::Ja)
            .count();
        let german_count = words
            .iter()
            .filter(|(_, _, l)| *l == LanguageCode::De)
            .count();

        assert!(english_count > 0, "must have real English test cases");
        assert!(japanese_count > 0, "must have real Japanese test cases");
        // No German test cases exist today -- a pipeline resolved to German
        // must hit `run_accuracy_test`'s "no test cases for this language"
        // fail-closed path rather than silently evaluating 0 cases as 100%.
        assert_eq!(german_count, 0);
    }

    #[test]
    fn test_load_cmu_accuracy_benchmark_loads_all_cases() {
        let benchmark = load_cmu_accuracy_benchmark().expect("must load");
        assert_eq!(benchmark.test_case_count(), cmu_test_words().len());
    }

    /// Regression test for the core finding: accuracy evaluation must use a
    /// REAL G2P backend (RuleBasedG2p), not `DummyG2p`'s naive
    /// one-phoneme-per-character mapping. `DummyG2p` would score close to 0%
    /// phoneme accuracy against real IPA transcriptions; the real rule-based
    /// engine should score substantially above that. Also verifies the
    /// per-language filtering: an English-only pipeline must report `None`
    /// (not `Some(false)`) for the Japanese target.
    #[tokio::test]
    async fn test_run_accuracy_test_uses_real_g2p_for_english_pipeline() {
        let pipeline = VoirsPipeline::builder()
            .with_language(voirs_sdk::types::LanguageCode::EnUs)
            .with_validation(false)
            .with_test_mode(true)
            .build()
            .await
            .expect("stub pipeline must build in test mode without network access");

        let global = default_global();
        let outcome = run_accuracy_test(&pipeline, &global)
            .await
            .expect("accuracy test against a real EnUs pipeline must succeed");

        assert_eq!(outcome.language, LanguageCode::EnUs);
        assert!(
            outcome.phoneme_accuracy > 0.1,
            "a real rule-based G2P engine must score well above what a naive \
             one-phoneme-per-character DummyG2p could achieve on real IPA transcriptions, \
             got {}",
            outcome.phoneme_accuracy
        );
        assert!(
            outcome.english_target_met.is_some(),
            "English target must be evaluated for an English-configured pipeline"
        );
        assert!(
            outcome.japanese_target_met.is_none(),
            "Japanese target must NOT be evaluated for an English-configured pipeline \
             (mixing languages would silently run Japanese text through English rules)"
        );
        assert!(outcome.synth_probe_count > 0);
    }

    #[tokio::test]
    async fn test_run_accuracy_test_uses_real_g2p_for_japanese_pipeline() {
        let pipeline = VoirsPipeline::builder()
            .with_language(voirs_sdk::types::LanguageCode::Ja)
            .with_validation(false)
            .with_test_mode(true)
            .build()
            .await
            .expect("stub pipeline must build in test mode without network access");

        let global = default_global();
        let outcome = run_accuracy_test(&pipeline, &global)
            .await
            .expect("accuracy test against a real Ja pipeline must succeed");

        assert_eq!(outcome.language, LanguageCode::Ja);
        assert!(outcome.japanese_target_met.is_some());
        assert!(
            outcome.english_target_met.is_none(),
            "English target must NOT be evaluated for a Japanese-configured pipeline"
        );
    }
}
