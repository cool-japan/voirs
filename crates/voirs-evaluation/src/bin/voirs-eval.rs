//! VoiRS Evaluation CLI Tool
//!
//! Command-line interface for speech quality evaluation, pronunciation assessment,
//! and comparative analysis.
//!
//! # Usage
//!
//! ```bash
//! # Quality evaluation
//! voirs-eval quality --generated audio.wav --reference reference.wav
//!
//! # Pronunciation assessment
//! voirs-eval pronunciation --audio speech.wav --text "Hello world" --language en-US
//!
//! # Batch evaluation
//! voirs-eval batch --input batch.json --output results.json
//!
//! # POLQA evaluation
//! voirs-eval polqa --generated audio.wav --reference ref.wav --bandwidth wideband
//! ```

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use voirs_evaluation::pronunciation::PronunciationEvaluatorImpl;
use voirs_evaluation::quality::{PolqaBandwidth, PolqaEvaluator, QualityEvaluator};
use voirs_evaluation::traits::{
    PronunciationEvaluator as PronunciationEvaluatorTrait, QualityEvaluationConfig,
    QualityEvaluator as QualityEvaluatorTrait,
};
use voirs_sdk::{AudioBuffer, LanguageCode};

#[derive(Parser)]
#[command(name = "voirs-eval")]
#[command(about = "VoiRS Speech Quality Evaluation CLI", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output format (json, csv, text)
    #[arg(short, long, default_value = "text")]
    format: OutputFormat,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Evaluate speech quality
    Quality {
        /// Generated audio file
        #[arg(short, long)]
        generated: PathBuf,

        /// Reference audio file (optional)
        #[arg(short, long)]
        reference: Option<PathBuf>,

        /// Enable objective metrics (PESQ, STOI, MCD)
        #[arg(long, default_value = "true")]
        objective: bool,

        /// Enable subjective quality prediction (MOS)
        #[arg(long, default_value = "true")]
        subjective: bool,
    },

    /// Evaluate pronunciation
    Pronunciation {
        /// Audio file to evaluate
        #[arg(short, long)]
        audio: PathBuf,

        /// Expected text
        #[arg(short, long)]
        text: String,

        /// Language code (e.g., en-US, ja-JP)
        #[arg(short, long)]
        language: String,

        /// Enable phoneme-level scoring
        #[arg(long, default_value = "true")]
        phoneme_level: bool,
    },

    /// POLQA evaluation (ITU-T P.863)
    Polqa {
        /// Generated audio file
        #[arg(short, long)]
        generated: PathBuf,

        /// Reference audio file
        #[arg(short, long)]
        reference: PathBuf,

        /// Bandwidth mode (narrowband, wideband, superwideband, fullband)
        #[arg(short, long, default_value = "wideband")]
        bandwidth: BandwidthMode,
    },

    /// Batch evaluation from JSON file
    Batch {
        /// Input JSON file with evaluation tasks
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for results
        #[arg(short, long)]
        output: PathBuf,

        /// Number of parallel workers
        #[arg(short = 'j', long, default_value = "4")]
        workers: usize,
    },

    /// Compare multiple audio files
    Compare {
        /// Audio files to compare
        #[arg(required = true)]
        files: Vec<PathBuf>,

        /// Reference audio file (optional)
        #[arg(short, long)]
        reference: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum OutputFormat {
    Json,
    Csv,
    Text,
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum BandwidthMode {
    Narrowband,
    Wideband,
    Superwideband,
    Fullband,
}

impl From<BandwidthMode> for PolqaBandwidth {
    fn from(mode: BandwidthMode) -> Self {
        match mode {
            BandwidthMode::Narrowband => PolqaBandwidth::NarrowBand,
            BandwidthMode::Wideband => PolqaBandwidth::WideBand,
            BandwidthMode::Superwideband => PolqaBandwidth::SuperWideBand,
            BandwidthMode::Fullband => PolqaBandwidth::FullBand,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Initialize tracing if verbose
    if cli.verbose {
        eprintln!("Verbose logging enabled");
    }

    match cli.command {
        Commands::Quality {
            generated,
            reference,
            objective,
            subjective,
        } => {
            evaluate_quality(generated, reference, objective, subjective, cli.format).await?;
        }
        Commands::Pronunciation {
            audio,
            text,
            language,
            phoneme_level,
        } => {
            evaluate_pronunciation(audio, text, language, phoneme_level, cli.format).await?;
        }
        Commands::Polqa {
            generated,
            reference,
            bandwidth,
        } => {
            evaluate_polqa(generated, reference, bandwidth, cli.format).await?;
        }
        Commands::Batch {
            input,
            output,
            workers,
        } => {
            batch_evaluate(input, output, workers, cli.format).await?;
        }
        Commands::Compare { files, reference } => {
            compare_files(files, reference, cli.format).await?;
        }
    }

    Ok(())
}

async fn evaluate_quality(
    generated: PathBuf,
    reference: Option<PathBuf>,
    objective: bool,
    subjective: bool,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load audio files
    let generated_audio = load_audio_file(&generated)?;
    let reference_audio = if let Some(ref_path) = reference.as_ref() {
        Some(load_audio_file(ref_path)?)
    } else {
        None
    };

    // Create evaluator
    let evaluator = QualityEvaluator::new().await?;

    // Configure evaluation
    let _config = QualityEvaluationConfig {
        objective_metrics: objective,
        ..Default::default()
    };
    let _subjective = subjective; // For future use

    // Evaluate
    let result = evaluator
        .evaluate_quality(&generated_audio, reference_audio.as_ref(), None)
        .await?;

    // Output results
    match format {
        OutputFormat::Json => {
            let output = serde_json::json!({
                "overall_score": result.overall_score,
                "component_scores": result.component_scores,
                "confidence": result.confidence,
                "recommendations": result.recommendations,
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        OutputFormat::Csv => {
            println!("metric,value");
            println!("overall_score,{}", result.overall_score);
            println!("confidence,{}", result.confidence);
            for (metric, value) in &result.component_scores {
                println!("{},{}", metric, value);
            }
        }
        OutputFormat::Text => {
            println!("=== Quality Evaluation Results ===");
            println!("Generated: {}", generated.display());
            if let Some(ref_path) = reference.as_ref() {
                println!("Reference: {}", ref_path.display());
            }
            println!("\nOverall Score: {:.3}", result.overall_score);
            println!("Confidence: {:.3}", result.confidence);
            println!("\nComponent Scores:");
            for (metric, value) in &result.component_scores {
                println!("  {}: {:.3}", metric, value);
            }
            if !result.recommendations.is_empty() {
                println!("\nRecommendations:");
                for rec in &result.recommendations {
                    println!("  - {}", rec);
                }
            }
        }
    }

    Ok(())
}

async fn evaluate_pronunciation(
    audio: PathBuf,
    text: String,
    language: String,
    _phoneme_level: bool,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load audio
    let audio_buffer = load_audio_file(&audio)?;

    // Create evaluator
    let evaluator = PronunciationEvaluatorImpl::new().await?;

    // Evaluate
    let result = evaluator
        .evaluate_pronunciation(&audio_buffer, &text, None)
        .await?;

    // Output results
    match format {
        OutputFormat::Json => {
            let output = serde_json::json!({
                "overall_score": result.overall_score,
                "fluency_score": result.fluency_score,
                "rhythm_score": result.rhythm_score,
                "stress_accuracy": result.stress_accuracy,
                "intonation_accuracy": result.intonation_accuracy,
                "phoneme_count": result.phoneme_scores.len(),
                "word_count": result.word_scores.len(),
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        OutputFormat::Csv => {
            println!("metric,value");
            println!("overall_score,{}", result.overall_score);
            println!("fluency_score,{}", result.fluency_score);
            println!("rhythm_score,{}", result.rhythm_score);
            println!("stress_accuracy,{}", result.stress_accuracy);
            println!("intonation_accuracy,{}", result.intonation_accuracy);
        }
        OutputFormat::Text => {
            println!("=== Pronunciation Evaluation Results ===");
            println!("Audio: {}", audio.display());
            println!("Text: \"{}\"", text);
            println!("Language: {}", language);
            println!("\nOverall Score: {:.3}", result.overall_score);
            println!("Fluency: {:.3}", result.fluency_score);
            println!("Rhythm: {:.3}", result.rhythm_score);
            println!("Stress Accuracy: {:.3}", result.stress_accuracy);
            println!("Intonation Accuracy: {:.3}", result.intonation_accuracy);
            println!(
                "\nPhoneme Scores: {} phonemes evaluated",
                result.phoneme_scores.len()
            );
            println!("Word Scores: {} words evaluated", result.word_scores.len());
        }
    }

    Ok(())
}

async fn evaluate_polqa(
    generated: PathBuf,
    reference: PathBuf,
    bandwidth: BandwidthMode,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load audio files
    let generated_audio = load_audio_file(&generated)?;
    let reference_audio = load_audio_file(&reference)?;

    // Create POLQA evaluator
    let evaluator = match bandwidth {
        BandwidthMode::Narrowband => PolqaEvaluator::new_narrowband()?,
        BandwidthMode::Wideband => PolqaEvaluator::new_wideband()?,
        BandwidthMode::Superwideband => PolqaEvaluator::new_superwideband()?,
        BandwidthMode::Fullband => PolqaEvaluator::new_fullband()?,
    };

    // Evaluate
    let score = evaluator
        .calculate_polqa(&reference_audio, &generated_audio)
        .await?;

    // Output results
    match format {
        OutputFormat::Json => {
            let result = serde_json::json!({
                "polqa_score": score,
                "bandwidth": format!("{:?}", bandwidth),
                "quality": interpret_polqa_score(score),
            });
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        OutputFormat::Csv => {
            println!("metric,value");
            println!("polqa_score,{}", score);
            println!("bandwidth,{:?}", bandwidth);
            println!("quality,{}", interpret_polqa_score(score));
        }
        OutputFormat::Text => {
            println!("=== POLQA Evaluation Results ===");
            println!("Generated: {}", generated.display());
            println!("Reference: {}", reference.display());
            println!("Bandwidth: {:?}", bandwidth);
            println!("\nPOLQA Score: {:.3}", score);
            println!("Quality: {}", interpret_polqa_score(score));
        }
    }

    Ok(())
}

async fn batch_evaluate(
    input: PathBuf,
    output: PathBuf,
    workers: usize,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;
    use tokio::sync::Semaphore;
    use tokio::task::JoinHandle;

    // Define batch task structure
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct BatchTask {
        id: String,
        generated: String,
        reference: Option<String>,
        #[serde(default)]
        language: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct BatchConfig {
        tasks: Vec<BatchTask>,
        #[serde(default)]
        output_format: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct BatchResult {
        id: String,
        overall_score: f32,
        confidence: f32,
        pesq: Option<f32>,
        stoi: Option<f32>,
        mcd: Option<f32>,
        processing_time_ms: f64,
        error: Option<String>,
    }

    // Load batch configuration
    let batch_json = std::fs::read_to_string(&input)?;
    let batch_config: BatchConfig = serde_json::from_str(&batch_json)?;

    println!("Processing batch evaluation from: {}", input.display());
    println!("Total tasks: {}", batch_config.tasks.len());
    println!("Parallel workers: {}", workers);
    println!("Output will be written to: {}", output.display());
    println!();

    // Create semaphore for controlling parallelism
    let semaphore = Arc::new(Semaphore::new(workers));

    // Create quality evaluator (shared across tasks)
    let evaluator = Arc::new(QualityEvaluator::new().await?);

    // Process tasks in parallel
    let mut handles: Vec<JoinHandle<BatchResult>> = Vec::new();
    let total_tasks = batch_config.tasks.len();

    for (idx, task) in batch_config.tasks.into_iter().enumerate() {
        let semaphore = Arc::clone(&semaphore);
        let evaluator = Arc::clone(&evaluator);
        let task_num = idx + 1;

        let handle = tokio::spawn(async move {
            // Acquire semaphore permit
            let _permit = semaphore.acquire().await.expect("Semaphore closed");

            println!(
                "[{}/{}] Processing task: {}",
                task_num, total_tasks, task.id
            );

            let start_time = std::time::Instant::now();

            // Execute evaluation
            let result = async {
                // Load generated audio
                let generated_path = PathBuf::from(&task.generated);
                let generated_audio =
                    load_audio_file(&generated_path).map_err(|e| format!("{}", e))?;

                // Load reference audio if provided
                let reference_audio = if let Some(ref ref_path_str) = task.reference {
                    let ref_path = PathBuf::from(ref_path_str);
                    Some(load_audio_file(&ref_path).map_err(|e| format!("{}", e))?)
                } else {
                    None
                };

                // Perform evaluation
                let eval_result = evaluator
                    .evaluate_quality(&generated_audio, reference_audio.as_ref(), None)
                    .await
                    .map_err(|e| format!("{}", e))?;

                Ok::<_, String>(eval_result)
            }
            .await;

            let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

            match result {
                Ok(eval_result) => {
                    println!(
                        "[{}/{}] ✓ Task {} completed: score={:.3}, time={:.1}ms",
                        task_num, total_tasks, task.id, eval_result.overall_score, processing_time
                    );

                    BatchResult {
                        id: task.id,
                        overall_score: eval_result.overall_score,
                        confidence: eval_result.confidence,
                        pesq: eval_result.component_scores.get("pesq").copied(),
                        stoi: eval_result.component_scores.get("stoi").copied(),
                        mcd: eval_result.component_scores.get("mcd").copied(),
                        processing_time_ms: processing_time,
                        error: None,
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[{}/{}] ✗ Task {} failed: {}",
                        task_num, total_tasks, task.id, e
                    );

                    BatchResult {
                        id: task.id,
                        overall_score: 0.0,
                        confidence: 0.0,
                        pesq: None,
                        stoi: None,
                        mcd: None,
                        processing_time_ms: processing_time,
                        error: Some(e.to_string()),
                    }
                }
            }
        });

        handles.push(handle);
    }

    // Collect all results
    println!("\nWaiting for all tasks to complete...");
    let mut results = Vec::new();
    for handle in handles {
        let result = handle.await?;
        results.push(result);
    }

    // Calculate summary statistics
    let successful_results: Vec<_> = results.iter().filter(|r| r.error.is_none()).collect();
    let failed_count = results.len() - successful_results.len();
    let avg_score = if !successful_results.is_empty() {
        successful_results
            .iter()
            .map(|r| r.overall_score)
            .sum::<f32>()
            / successful_results.len() as f32
    } else {
        0.0
    };
    let avg_time = if !results.is_empty() {
        results.iter().map(|r| r.processing_time_ms).sum::<f64>() / results.len() as f64
    } else {
        0.0
    };

    println!("\n=== Batch Evaluation Summary ===");
    println!("Total tasks: {}", results.len());
    println!("Successful: {}", successful_results.len());
    println!("Failed: {}", failed_count);
    println!("Average score: {:.3}", avg_score);
    println!("Average processing time: {:.1}ms", avg_time);
    println!();

    // Write results to output file
    match format {
        OutputFormat::Json => {
            let output_data = serde_json::json!({
                "summary": {
                    "total_tasks": results.len(),
                    "successful": successful_results.len(),
                    "failed": failed_count,
                    "average_score": avg_score,
                    "average_time_ms": avg_time,
                },
                "results": results,
            });
            let json_str = serde_json::to_string_pretty(&output_data)?;
            std::fs::write(&output, json_str)?;
        }
        OutputFormat::Csv => {
            let mut csv_content = String::from(
                "id,overall_score,confidence,pesq,stoi,mcd,processing_time_ms,error\n",
            );
            for result in &results {
                csv_content.push_str(&format!(
                    "{},{},{},{},{},{},{},{}\n",
                    result.id,
                    result.overall_score,
                    result.confidence,
                    result.pesq.map(|v| v.to_string()).unwrap_or_default(),
                    result.stoi.map(|v| v.to_string()).unwrap_or_default(),
                    result.mcd.map(|v| v.to_string()).unwrap_or_default(),
                    result.processing_time_ms,
                    result.error.as_deref().unwrap_or("")
                ));
            }
            std::fs::write(&output, csv_content)?;
        }
        OutputFormat::Text => {
            let mut text_content = String::from("=== Batch Evaluation Results ===\n\n");
            text_content.push_str(&format!("Total tasks: {}\n", results.len()));
            text_content.push_str(&format!("Successful: {}\n", successful_results.len()));
            text_content.push_str(&format!("Failed: {}\n", failed_count));
            text_content.push_str(&format!("Average score: {:.3}\n", avg_score));
            text_content.push_str(&format!("Average time: {:.1}ms\n\n", avg_time));

            for result in &results {
                text_content.push_str(&format!("Task: {}\n", result.id));
                text_content.push_str(&format!("  Overall Score: {:.3}\n", result.overall_score));
                text_content.push_str(&format!("  Confidence: {:.3}\n", result.confidence));
                if let Some(pesq) = result.pesq {
                    text_content.push_str(&format!("  PESQ: {:.3}\n", pesq));
                }
                if let Some(stoi) = result.stoi {
                    text_content.push_str(&format!("  STOI: {:.3}\n", stoi));
                }
                if let Some(mcd) = result.mcd {
                    text_content.push_str(&format!("  MCD: {:.3}\n", mcd));
                }
                text_content.push_str(&format!("  Time: {:.1}ms\n", result.processing_time_ms));
                if let Some(error) = &result.error {
                    text_content.push_str(&format!("  Error: {}\n", error));
                }
                text_content.push('\n');
            }
            std::fs::write(&output, text_content)?;
        }
    }

    println!("Results written to: {}", output.display());

    Ok(())
}

async fn compare_files(
    files: Vec<PathBuf>,
    reference: Option<PathBuf>,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing {} audio files", files.len());

    // Load reference if provided
    let reference_audio = if let Some(ref_path) = reference.as_ref() {
        Some(load_audio_file(ref_path)?)
    } else {
        None
    };

    // Create evaluator
    let evaluator = QualityEvaluator::new().await?;

    // Evaluate each file
    let mut results = Vec::new();
    for file in &files {
        let audio = load_audio_file(file)?;
        let result = evaluator
            .evaluate_quality(&audio, reference_audio.as_ref(), None)
            .await?;
        results.push((file.display().to_string(), result));
    }

    // Output comparison
    match format {
        OutputFormat::Json => {
            let comparison = serde_json::json!({
                "files": results.iter().map(|(name, result)| {
                    serde_json::json!({
                        "file": name,
                        "overall_score": result.overall_score,
                        "confidence": result.confidence,
                    })
                }).collect::<Vec<_>>(),
            });
            println!("{}", serde_json::to_string_pretty(&comparison)?);
        }
        OutputFormat::Csv => {
            println!("file,overall_score,confidence");
            for (name, result) in &results {
                println!("{},{},{}", name, result.overall_score, result.confidence);
            }
        }
        OutputFormat::Text => {
            println!("=== File Comparison Results ===");
            if let Some(ref_path) = reference {
                println!("Reference: {}", ref_path.display());
            }
            println!();
            for (name, result) in &results {
                println!("File: {}", name);
                println!("  Overall Score: {:.3}", result.overall_score);
                println!("  Confidence: {:.3}", result.confidence);
                println!();
            }
        }
    }

    Ok(())
}

fn load_audio_file(path: &PathBuf) -> Result<AudioBuffer, Box<dyn std::error::Error>> {
    // Use hound to load WAV files
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .into_samples::<i16>()
            .map(|s| s.map(|sample| sample as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };

    Ok(AudioBuffer::new(
        samples,
        spec.sample_rate,
        spec.channels as u32,
    ))
}

fn interpret_polqa_score(score: f32) -> &'static str {
    match score {
        s if s >= 4.0 => "Excellent",
        s if s >= 3.5 => "Good",
        s if s >= 2.5 => "Fair",
        s if s >= 1.5 => "Poor",
        _ => "Bad",
    }
}
