//! Synthesis command implementations.

use crate::cli_types::CliAudioFormat;
use crate::ssml;
use crate::{utils, GlobalOptions};
use hound::WavWriter;
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use std::collections::VecDeque;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{Duration, Instant};
use voirs_sdk::config::AppConfig;
use voirs_sdk::error::IoOperation;
use voirs_sdk::types::SynthesisConfig;
use voirs_sdk::{AudioFormat, QualityLevel, Result, VoirsError, VoirsPipeline};

/// Enhanced synthesis options with validation
#[derive(Debug, Clone)]
pub struct EnhancedSynthesisOptions {
    /// Text content to synthesize
    pub text: String,
    /// Output file path
    pub output: Option<std::path::PathBuf>,
    /// Speaking rate (0.25 - 4.0)
    pub rate: f32,
    /// Pitch shift in semitones (-24.0 - 24.0)
    pub pitch: f32,
    /// Volume gain in dB (-40.0 - 20.0)
    pub volume: f32,
    /// Quality level
    pub quality: QualityLevel,
    /// Enable audio enhancement
    pub enhance: bool,
    /// Voice ID (optional, uses default if None)
    pub voice_id: Option<String>,
    /// Enable SSML processing
    pub enable_ssml: bool,
    /// Enable real-time streaming output
    pub enable_realtime: bool,
    /// Target latency in milliseconds for real-time mode
    pub target_latency_ms: Option<f32>,
    /// Maximum retries for failed operations
    pub max_retries: usize,
    /// Enable fallback to default voice on error
    pub enable_fallback: bool,
}

impl Default for EnhancedSynthesisOptions {
    fn default() -> Self {
        Self {
            text: String::new(),
            output: None,
            rate: 1.0,
            pitch: 0.0,
            volume: 0.0,
            quality: QualityLevel::High,
            enhance: false,
            voice_id: None,
            enable_ssml: false,
            enable_realtime: false,
            target_latency_ms: None,
            max_retries: 3,
            enable_fallback: true,
        }
    }
}

/// Configuration for streaming synthesis
#[derive(Debug, Clone)]
struct StreamingConfig {
    /// Maximum chunk size in characters (default: 1000)
    max_chunk_size: usize,
    /// Maximum words per chunk (default: 200)
    max_words_per_chunk: usize,
    /// Overlap between chunks in characters (default: 50)
    chunk_overlap: usize,
    /// Enable streaming output (default: false)
    enable_streaming: bool,
    /// Memory limit in MB (default: 500)
    memory_limit_mb: usize,
    /// Maximum concurrent chunks (default: 4)
    max_concurrent_chunks: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 1000,
            max_words_per_chunk: 200,
            chunk_overlap: 50,
            enable_streaming: false,
            memory_limit_mb: 500,
            max_concurrent_chunks: 4,
        }
    }
}

/// Validate synthesis parameters
fn validate_synthesis_options(options: &EnhancedSynthesisOptions) -> Result<()> {
    // Validate text content
    if options.text.trim().is_empty() {
        return Err(VoirsError::InvalidConfiguration {
            field: "text".to_string(),
            value: "empty".to_string(),
            reason: "Text content cannot be empty".to_string(),
            valid_values: None,
        });
    }

    if options.text.len() > 1_000_000 {
        return Err(VoirsError::InvalidConfiguration {
            field: "text".to_string(),
            value: format!("{} chars", options.text.len()),
            reason: "Text content too long (max 1MB)".to_string(),
            valid_values: Some(vec!["Less than 1,000,000 characters".to_string()]),
        });
    }

    // Validate speaking rate
    if !(0.25..=4.0).contains(&options.rate) {
        return Err(VoirsError::InvalidConfiguration {
            field: "rate".to_string(),
            value: options.rate.to_string(),
            reason: "Speaking rate must be between 0.25 and 4.0".to_string(),
            valid_values: Some(vec!["0.25".to_string(), "4.0".to_string()]),
        });
    }

    // Validate pitch shift
    if !(-24.0..=24.0).contains(&options.pitch) {
        return Err(VoirsError::InvalidConfiguration {
            field: "pitch".to_string(),
            value: options.pitch.to_string(),
            reason: "Pitch shift must be between -24.0 and 24.0 semitones".to_string(),
            valid_values: Some(vec!["-24.0".to_string(), "24.0".to_string()]),
        });
    }

    // Validate volume gain
    if !(-40.0..=20.0).contains(&options.volume) {
        return Err(VoirsError::InvalidConfiguration {
            field: "volume".to_string(),
            value: options.volume.to_string(),
            reason: "Volume gain must be between -40.0 and 20.0 dB".to_string(),
            valid_values: Some(vec!["-40.0".to_string(), "20.0".to_string()]),
        });
    }

    // Validate target latency if specified
    if let Some(latency) = options.target_latency_ms {
        if !(10.0..=5000.0).contains(&latency) {
            return Err(VoirsError::InvalidConfiguration {
                field: "target_latency_ms".to_string(),
                value: latency.to_string(),
                reason: "Target latency must be between 10ms and 5000ms".to_string(),
                valid_values: Some(vec!["10.0".to_string(), "5000.0".to_string()]),
            });
        }
    }

    // Validate output path if specified
    if let Some(output_path) = &options.output {
        if let Some(parent) = output_path.parent() {
            // Only check if parent is not empty (empty means current directory)
            if !parent.as_os_str().is_empty() && !parent.exists() {
                return Err(VoirsError::IoError {
                    path: parent.to_path_buf(),
                    operation: IoOperation::Metadata,
                    source: std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "Output directory does not exist",
                    ),
                });
            }
        }

        // Check if file extension is supported
        if let Some(ext) = output_path.extension().and_then(|e| e.to_str()) {
            match ext.to_lowercase().as_str() {
                "wav" | "flac" | "mp3" | "opus" | "ogg" => {} // Supported formats
                _ => {
                    return Err(VoirsError::UnsupportedFileFormat {
                        path: output_path.clone(),
                        format: ext.to_string(),
                    })
                }
            }
        }
    }

    Ok(())
}

/// Enhanced synthesis with retry logic and better error handling
pub async fn run_enhanced_synthesize(
    options: EnhancedSynthesisOptions,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    // Validate input parameters
    validate_synthesis_options(&options)?;

    if !global.quiet {
        tracing::info!("Starting enhanced synthesis with options: {:?}", options);
    }

    let mut last_error = None;
    let mut retry_count = 0;

    while retry_count <= options.max_retries {
        match try_synthesize(&options, config, global, retry_count).await {
            Ok(result) => return Ok(result),
            Err(e) => {
                last_error = Some(e.clone());
                retry_count += 1;

                if retry_count <= options.max_retries {
                    let backoff_ms = 1000 * (2_u64.pow(retry_count.saturating_sub(1) as u32));
                    if !global.quiet {
                        tracing::warn!(
                            "Synthesis attempt {} failed: {}. Retrying in {}ms...",
                            retry_count,
                            e,
                            backoff_ms
                        );
                    }
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                }
            }
        }
    }

    // All retries exhausted
    let final_error = last_error.unwrap_or_else(|| VoirsError::SynthesisFailed {
        text: options.text.clone(),
        text_length: options.text.len(),
        stage: voirs_sdk::error::SynthesisStage::AudioFinalization,
        cause: "Unknown error after all retries".into(),
    });

    Err(VoirsError::SynthesisFailed {
        text: options.text.clone(),
        text_length: options.text.len(),
        stage: voirs_sdk::error::SynthesisStage::AudioFinalization,
        cause: format!(
            "Failed after {} attempts. Last error: {}",
            options.max_retries + 1,
            final_error
        )
        .into(),
    })
}

/// Attempt synthesis with enhanced options
async fn try_synthesize(
    options: &EnhancedSynthesisOptions,
    config: &AppConfig,
    global: &GlobalOptions,
    attempt: usize,
) -> Result<()> {
    // Process SSML if enabled
    let processed_text = if options.enable_ssml {
        ssml::process_ssml(&options.text)?
    } else {
        options.text.clone()
    };

    // Determine if we need streaming synthesis
    let should_stream = processed_text.len() > 500
        || processed_text.split_whitespace().count() > 100
        || options.enable_realtime;

    if should_stream {
        return run_enhanced_streaming_synthesis(options, &processed_text, config, global, attempt)
            .await;
    }

    // Build pipeline with enhanced options
    let mut pipeline_builder = VoirsPipeline::builder()
        .with_quality(options.quality)
        .with_gpu_acceleration(config.pipeline.use_gpu || global.gpu);

    // Apply voice selection if specified
    if let Some(voice_id) = &options.voice_id {
        pipeline_builder = pipeline_builder.with_voice(voice_id);
    }

    let pipeline = pipeline_builder.build().await.map_err(|e| {
        if options.enable_fallback && attempt == 0 {
            VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::Acoustic,
                message: format!(
                    "Pipeline build failed: {}. Will retry with fallback options.",
                    e
                ),
                source: Some(e.into()),
            }
        } else {
            e
        }
    })?;

    // Create enhanced synthesis config
    let mut synth_config = SynthesisConfig {
        speaking_rate: options.rate,
        pitch_shift: options.pitch,
        volume_gain: options.volume,
        enable_enhancement: options.enhance,
        quality: options.quality,
        ..Default::default()
    };

    // Configure real-time streaming if enabled
    if options.enable_realtime {
        // For real-time mode, use streaming chunks
        if options.enable_realtime {
            synth_config.streaming_chunk_size = Some(8); // Process in 8-word chunks
        }
    }

    // Synthesize audio
    let audio = pipeline
        .synthesize_with_config(&processed_text, &synth_config)
        .await?;

    // Determine output path
    let output_path = if let Some(path) = &options.output {
        path.clone()
    } else {
        let format: AudioFormat = global.format.map(|f| f.into()).unwrap_or_default();
        let filename = utils::generate_output_filename(&processed_text, format);
        std::env::current_dir()?.join(filename)
    };

    // Check if output is stdout
    let output_is_stdout = output_path.to_str() == Some("-");

    // Save audio with enhanced error handling
    if output_is_stdout {
        // Write to stdout
        save_wav_to_stdout(&audio)?;
    } else {
        // Write to file
        let format = utils::format_from_extension(&output_path)
            .or(global.format.map(|f| f.into()))
            .unwrap_or_default();

        audio
            .save(&output_path, format)
            .map_err(|e| VoirsError::IoError {
                path: output_path.clone(),
                operation: IoOperation::Write,
                source: std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to save audio: {}", e),
                ),
            })?;

        if !global.quiet {
            println!("✓ Synthesis complete: {}", output_path.display());
            println!("  Duration: {:.2}s", audio.duration());
            println!("  Quality: {:?}", options.quality);
            if attempt > 0 {
                println!("  Completed after {} retries", attempt);
            }
        }
    }

    Ok(())
}

/// Enhanced streaming synthesis with better error handling
async fn run_enhanced_streaming_synthesis(
    options: &EnhancedSynthesisOptions,
    processed_text: &str,
    config: &AppConfig,
    global: &GlobalOptions,
    attempt: usize,
) -> Result<()> {
    tracing::info!(
        "Running enhanced streaming synthesis for text of length: {}",
        processed_text.len()
    );

    let mut streaming_config = StreamingConfig::default();

    // Adjust streaming config based on options
    if options.enable_realtime {
        streaming_config.enable_streaming = true;
        streaming_config.max_concurrent_chunks = 2; // Lower for real-time
        if let Some(latency_ms) = options.target_latency_ms {
            // Adjust chunk size based on target latency
            streaming_config.max_chunk_size = ((latency_ms / 1000.0) * 150.0) as usize; // ~150 chars per second
            streaming_config.max_chunk_size = streaming_config.max_chunk_size.max(100).min(2000);
        }
    }

    if !global.quiet {
        println!(
            "🔄 Processing text ({} characters) with enhanced streaming synthesis...",
            processed_text.len()
        );
    }

    // Split text into chunks with enhanced error handling
    let chunks = split_text_into_chunks(processed_text, &streaming_config).map_err(|e| {
        VoirsError::TextPreprocessingError {
            message: format!("Failed to split text into chunks: {}", e),
            text_sample: processed_text.chars().take(100).collect(),
        }
    })?;

    if !global.quiet {
        println!("📝 Split into {} chunks for processing", chunks.len());
    }

    // Build pipeline with fallback handling
    let pipeline = Arc::new(build_enhanced_pipeline(options, config, global, attempt).await?);

    // Create enhanced synthesis config
    let synth_config = SynthesisConfig {
        speaking_rate: options.rate,
        pitch_shift: options.pitch,
        volume_gain: options.volume,
        enable_enhancement: options.enhance,
        streaming_chunk_size: if options.enable_realtime {
            Some(8)
        } else {
            None
        },
        ..Default::default()
    };

    // Determine output path
    let output_path = if let Some(path) = &options.output {
        path.clone()
    } else {
        let format: AudioFormat = global.format.map(|f| f.into()).unwrap_or_default();
        let filename = utils::generate_output_filename(processed_text, format);
        std::env::current_dir()?.join(filename)
    };

    // Process chunks with enhanced progress tracking
    let format = utils::format_from_extension(&output_path)
        .or(global.format.map(|f| f.into()))
        .unwrap_or_default();

    let audio_segments = process_chunks_with_enhanced_progress(
        chunks.clone(),
        &pipeline,
        &synth_config,
        &streaming_config,
        global,
        attempt,
    )
    .await?;

    // Combine audio segments and save
    let combined_audio =
        combine_audio_segments(audio_segments).map_err(|e| VoirsError::AudioError {
            message: format!("Failed to combine audio segments: {}", e),
            buffer_info: None,
        })?;

    combined_audio
        .save(&output_path, format)
        .map_err(|e| VoirsError::IoError {
            path: output_path.clone(),
            operation: IoOperation::Write,
            source: std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to save combined audio: {}", e),
            ),
        })?;

    if !global.quiet {
        println!(
            "✅ Enhanced streaming synthesis complete: {}",
            output_path.display()
        );
        println!("  Duration: {:.2}s", combined_audio.duration());
        println!("  Processed {} chunks", chunks.len());
        println!("  Quality: {:?}", options.quality);
        if attempt > 0 {
            println!("  Completed after {} retries", attempt);
        }
    }

    Ok(())
}

/// Build pipeline with enhanced fallback options
async fn build_enhanced_pipeline(
    options: &EnhancedSynthesisOptions,
    config: &AppConfig,
    global: &GlobalOptions,
    attempt: usize,
) -> Result<VoirsPipeline> {
    let mut pipeline_builder = VoirsPipeline::builder()
        .with_quality(options.quality)
        .with_gpu_acceleration(config.pipeline.use_gpu || global.gpu);

    // Voice selection with fallback logic
    if let Some(voice_id) = &options.voice_id {
        pipeline_builder = pipeline_builder.with_voice(voice_id);
    } else if attempt > 0 && options.enable_fallback {
        // Use fallback voice on retry
        tracing::info!("Using fallback voice for retry attempt {}", attempt);
        pipeline_builder = pipeline_builder.with_voice("default");
    }

    // Quality fallback on retry
    let effective_quality = if attempt > 1 && options.enable_fallback {
        match options.quality {
            QualityLevel::Ultra => QualityLevel::High,
            QualityLevel::High => QualityLevel::Medium,
            _ => options.quality,
        }
    } else {
        options.quality
    };

    pipeline_builder = pipeline_builder.with_quality(effective_quality);

    pipeline_builder.build().await
}

/// Run text synthesis command (legacy interface for backwards compatibility)
pub async fn run_synthesize(
    text: &str,
    output: Option<&Path>,
    rate: f32,
    pitch: f32,
    volume: f32,
    quality: QualityLevel,
    enhance: bool,
    play: bool,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    let output_path = output.map(|p| p.to_path_buf());

    let options = EnhancedSynthesisOptions {
        text: text.to_string(),
        output: output_path.clone(),
        rate,
        pitch,
        volume,
        quality,
        enhance,
        ..Default::default()
    };

    run_enhanced_synthesize(options, config, global).await?;

    // Play audio if requested and we have an output path that's not stdout
    if play {
        if let Some(ref path) = output_path {
            if path.to_str() != Some("-") {
                use crate::audio::playback::play_audio_file_simple;
                play_audio_file_simple(path)?;
            }
        }
    }

    Ok(())
}

/// Run streaming synthesis for long texts
pub async fn run_streaming_synthesis(
    text: &str,
    output: Option<&Path>,
    rate: f32,
    pitch: f32,
    volume: f32,
    quality: QualityLevel,
    enhance: bool,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    tracing::info!(
        "Running streaming synthesis for text of length: {}",
        text.len()
    );

    let streaming_config = StreamingConfig::default();

    if !global.quiet {
        println!(
            "Processing long text ({} characters) with streaming synthesis...",
            text.len()
        );
    }

    // Split text into chunks
    let chunks = split_text_into_chunks(text, &streaming_config)?;

    if !global.quiet {
        println!("Split into {} chunks for processing", chunks.len());
    }

    // Build pipeline
    let pipeline = Arc::new(
        VoirsPipeline::builder()
            .with_quality(quality)
            .with_gpu_acceleration(config.pipeline.use_gpu || global.gpu)
            .build()
            .await?,
    );

    // Create synthesis config
    let synth_config = SynthesisConfig {
        speaking_rate: rate,
        pitch_shift: pitch,
        volume_gain: volume,
        enable_enhancement: enhance,
        quality,
        ..Default::default()
    };

    // Determine output path
    let output_path = if let Some(path) = output {
        path.to_path_buf()
    } else {
        let format: AudioFormat = global.format.map(|f| f.into()).unwrap_or_default();
        let filename = utils::generate_output_filename(text, format);
        std::env::current_dir()?.join(filename)
    };

    // Process chunks with progress tracking
    let audio_segments =
        process_chunks_with_progress(&chunks, &pipeline, &synth_config, &streaming_config, global)
            .await?;

    // Combine audio segments
    let combined_audio = combine_audio_segments(audio_segments)?;

    // Check if output is stdout
    let output_is_stdout = output_path.to_str() == Some("-");

    // Save audio
    if output_is_stdout {
        // Write to stdout
        save_wav_to_stdout(&combined_audio)?;
    } else {
        // Write to file
        let format = utils::format_from_extension(&output_path)
            .or(global.format.map(|f| f.into()))
            .unwrap_or_default();

        combined_audio.save(&output_path, format)?;

        if !global.quiet {
            println!("Streaming synthesis complete: {}", output_path.display());
            println!("Duration: {:.2}s", combined_audio.duration());
            println!("Processed {} chunks", chunks.len());
        }
    }

    Ok(())
}

/// Run file synthesis command
pub async fn run_synthesize_file(
    input: &Path,
    output_dir: Option<&Path>,
    rate: f32,
    quality: QualityLevel,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    tracing::info!("Synthesizing file: {}", input.display());

    // Read input file
    let content = std::fs::read_to_string(input).map_err(|e| voirs_sdk::VoirsError::from(e))?;

    // Determine output directory
    let output_dir = if let Some(dir) = output_dir {
        dir.to_path_buf()
    } else {
        input
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .to_path_buf()
    };

    // Ensure output directory exists
    std::fs::create_dir_all(&output_dir).map_err(|e| voirs_sdk::VoirsError::from(e))?;

    // Process file content
    // If file has multiple lines, treat each line as separate synthesis
    let lines: Vec<&str> = content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty() && !line.starts_with('#')) // Skip empty lines and comments
        .collect();

    if lines.is_empty() {
        if !global.quiet {
            println!("No content to synthesize in file: {}", input.display());
        }
        return Ok(());
    }

    // Build pipeline
    let pipeline = VoirsPipeline::builder()
        .with_quality(quality)
        .with_gpu_acceleration(config.pipeline.use_gpu || global.gpu)
        .build()
        .await?;

    // Create synthesis config
    let synth_config = SynthesisConfig {
        speaking_rate: rate,
        quality,
        ..Default::default()
    };

    if !global.quiet {
        println!("Processing {} lines from {}", lines.len(), input.display());
    }

    let mut total_duration = 0.0;
    let base_name = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    // Process each line
    for (i, line) in lines.iter().enumerate() {
        if !global.quiet {
            let display_line = if line.len() > 50 {
                format!("{}...", &line[..47])
            } else {
                line.to_string()
            };
            println!(
                "Synthesizing line {}/{}: {}",
                i + 1,
                lines.len(),
                display_line
            );
        }

        // Synthesize audio
        let audio = pipeline.synthesize_with_config(line, &synth_config).await?;

        // Generate output filename
        let format: AudioFormat = global.format.map(|f| f.into()).unwrap_or_default();
        let output_filename = if lines.len() == 1 {
            format!("{}.{}", base_name, format.extension())
        } else {
            format!("{}_{:03}.{}", base_name, i + 1, format.extension())
        };
        let output_path = output_dir.join(output_filename);

        // Save audio
        audio.save(&output_path, format)?;

        total_duration += audio.duration();

        if !global.quiet {
            println!(
                "  Saved: {} ({:.2}s)",
                output_path.display(),
                audio.duration()
            );
        }
    }

    if !global.quiet {
        println!("File synthesis complete!");
        println!("  Processed: {} lines", lines.len());
        println!("  Total duration: {:.2}s", total_duration);
        println!("  Output directory: {}", output_dir.display());
    }

    Ok(())
}

/// Split text into chunks for processing
fn split_text_into_chunks(text: &str, config: &StreamingConfig) -> Result<Vec<String>> {
    let mut chunks = Vec::new();

    // Use sentence boundaries for cleaner splitting
    let sentence_regex = Regex::new(r"[.!?]\s+").unwrap();
    let sentences: Vec<&str> = sentence_regex.split(text).collect();

    let mut current_chunk = String::new();
    let mut current_word_count = 0;

    for sentence in sentences {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }

        let sentence_word_count = sentence.split_whitespace().count();

        // Check if adding this sentence would exceed limits
        if !current_chunk.is_empty()
            && (current_chunk.len() + sentence.len() > config.max_chunk_size
                || current_word_count + sentence_word_count > config.max_words_per_chunk)
        {
            // Start new chunk
            if !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
                current_chunk.clear();
                current_word_count = 0;
            }
        }

        if !current_chunk.is_empty() {
            current_chunk.push(' ');
        }
        current_chunk.push_str(sentence);
        current_word_count += sentence_word_count;
    }

    // Add final chunk if not empty
    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    // If no chunks were created, use the original text
    if chunks.is_empty() {
        chunks.push(text.to_string());
    }

    Ok(chunks)
}

/// Process chunks with progress tracking and concurrent execution
async fn process_chunks_with_progress(
    chunks: &[String],
    pipeline: &Arc<VoirsPipeline>,
    synth_config: &SynthesisConfig,
    streaming_config: &StreamingConfig,
    global: &GlobalOptions,
) -> Result<Vec<voirs_sdk::AudioBuffer>> {
    let progress_bar = if !global.quiet {
        let pb = ProgressBar::new(chunks.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message("Processing chunks");
        Some(pb)
    } else {
        None
    };

    let semaphore = Arc::new(Semaphore::new(streaming_config.max_concurrent_chunks));
    let mut tasks = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        let pipeline_clone = pipeline.clone();
        let synth_config_clone = synth_config.clone();
        let chunk_clone = chunk.clone();
        let semaphore_clone = semaphore.clone();
        let pb_clone = progress_bar.clone();

        let task = tokio::spawn(async move {
            let _permit = semaphore_clone.acquire().await.unwrap();

            let start_time = Instant::now();
            let result = pipeline_clone
                .synthesize_with_config(&chunk_clone, &synth_config_clone)
                .await;
            let elapsed = start_time.elapsed();

            if let Some(pb) = pb_clone {
                pb.inc(1);
                pb.set_message(format!("Chunk {}: {:.2}s", i + 1, elapsed.as_secs_f64()));
            }

            result.map(|audio| (i, audio))
        });

        tasks.push(task);
    }

    // Wait for all tasks to complete
    let mut results = Vec::new();
    for task in tasks {
        match task.await {
            Ok(Ok((index, audio))) => {
                results.push((index, audio));
            }
            Ok(Err(e)) => {
                if let Some(pb) = &progress_bar {
                    pb.finish_with_message("Processing failed");
                }
                return Err(e);
            }
            Err(e) => {
                if let Some(pb) = &progress_bar {
                    pb.finish_with_message("Processing failed");
                }
                return Err(voirs_sdk::VoirsError::model_error(format!(
                    "Task failed: {}",
                    e
                )));
            }
        }
    }

    if let Some(pb) = &progress_bar {
        pb.finish_with_message("Processing complete");
    }

    // Sort results by index to maintain order
    results.sort_by_key(|(index, _)| *index);
    let audio_segments: Vec<voirs_sdk::AudioBuffer> =
        results.into_iter().map(|(_, audio)| audio).collect();

    Ok(audio_segments)
}

/// Process chunks with enhanced progress tracking and error recovery
async fn process_chunks_with_enhanced_progress(
    chunks: Vec<String>,
    pipeline: &Arc<VoirsPipeline>,
    synth_config: &SynthesisConfig,
    streaming_config: &StreamingConfig,
    global: &GlobalOptions,
    attempt: usize,
) -> Result<Vec<voirs_sdk::AudioBuffer>> {
    let progress_bar = if !global.quiet {
        let pb = ProgressBar::new(chunks.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-")
        );

        let msg = if attempt > 0 {
            format!("Processing chunks (retry {})", attempt)
        } else {
            "Processing chunks".to_string()
        };
        pb.set_message(msg);
        Some(pb)
    } else {
        None
    };

    let semaphore = Arc::new(Semaphore::new(streaming_config.max_concurrent_chunks));
    let mut tasks = Vec::new();
    let total_chunks = chunks.len();
    let total_text_length: usize = chunks.iter().map(|c| c.len()).sum();

    for (i, chunk) in chunks.into_iter().enumerate() {
        let pipeline_clone = pipeline.clone();
        let synth_config_clone = synth_config.clone();
        let semaphore_clone = semaphore.clone();
        let pb_clone = progress_bar.clone();
        let total_chunks_clone = total_chunks;

        let task = tokio::spawn(async move {
            let _permit = semaphore_clone.acquire().await.unwrap();

            let start_time = Instant::now();
            let mut last_error = None;

            // Enhanced retry logic for individual chunks
            for retry in 0..3 {
                match pipeline_clone
                    .synthesize_with_config(&chunk, &synth_config_clone)
                    .await
                {
                    Ok(audio) => {
                        let elapsed = start_time.elapsed();

                        if let Some(pb) = pb_clone {
                            pb.inc(1);
                            pb.set_message(format!(
                                "Chunk {}/{} completed ({:.2}s){}",
                                i + 1,
                                total_chunks_clone,
                                elapsed.as_secs_f64(),
                                if retry > 0 {
                                    format!(" after {} retries", retry)
                                } else {
                                    String::new()
                                }
                            ));
                        }

                        return Ok((i, audio));
                    }
                    Err(e) => {
                        last_error = Some(e);
                        if retry < 2 {
                            // Brief backoff before retry
                            tokio::time::sleep(Duration::from_millis(500 * (retry + 1) as u64))
                                .await;
                        }
                    }
                }
            }

            // All retries failed for this chunk
            Err(last_error.unwrap_or_else(|| VoirsError::SynthesisFailed {
                text: chunk.clone(),
                text_length: chunk.len(),
                stage: voirs_sdk::error::SynthesisStage::AcousticModeling,
                cause: "Chunk processing failed after retries".into(),
            }))
        });

        tasks.push(task);
    }

    // Wait for all tasks to complete with enhanced error reporting
    let mut results = Vec::new();
    let mut failed_chunks = Vec::new();

    for (task_idx, task) in tasks.into_iter().enumerate() {
        match task.await {
            Ok(Ok((index, audio))) => {
                results.push((index, audio));
            }
            Ok(Err(e)) => {
                failed_chunks.push((task_idx, e));
            }
            Err(e) => {
                failed_chunks.push((
                    task_idx,
                    VoirsError::InternalError {
                        component: "synthesis_task".to_string(),
                        message: format!("Task failed: {}", e),
                    },
                ));
            }
        }
    }

    // Handle failures
    if !failed_chunks.is_empty() {
        if let Some(pb) = &progress_bar {
            pb.finish_with_message(format!(
                "Processing failed ({} chunks failed)",
                failed_chunks.len()
            ));
        }

        let error_details = failed_chunks
            .iter()
            .map(|(idx, err)| format!("Chunk {}: {}", idx + 1, err))
            .collect::<Vec<_>>()
            .join("; ");

        return Err(VoirsError::SynthesisFailed {
            text: format!("{} chunks", total_chunks),
            text_length: total_text_length,
            stage: voirs_sdk::error::SynthesisStage::AcousticModeling,
            cause: format!(
                "{} out of {} chunks failed: {}",
                failed_chunks.len(),
                total_chunks,
                error_details
            )
            .into(),
        });
    }

    if let Some(pb) = &progress_bar {
        pb.finish_with_message("✅ All chunks processed successfully");
    }

    // Sort results by index to maintain order
    results.sort_by_key(|(index, _)| *index);
    let audio_segments: Vec<voirs_sdk::AudioBuffer> =
        results.into_iter().map(|(_, audio)| audio).collect();

    Ok(audio_segments)
}

/// Combine multiple audio segments into a single audio buffer
fn combine_audio_segments(segments: Vec<voirs_sdk::AudioBuffer>) -> Result<voirs_sdk::AudioBuffer> {
    if segments.is_empty() {
        return Err(VoirsError::AudioError {
            message: "No audio segments to combine".to_string(),
            buffer_info: None,
        });
    }

    if segments.len() == 1 {
        return Ok(segments.into_iter().next().unwrap());
    }

    // For now, use the first segment's sample rate and channels
    let first_segment = &segments[0];
    let sample_rate = first_segment.sample_rate();
    let channels = first_segment.channels();

    // Combine all samples
    let mut combined_samples = Vec::new();
    for segment in segments {
        combined_samples.extend(segment.samples());
    }

    // Create new audio buffer
    let buffer = voirs_sdk::AudioBuffer::new(combined_samples, sample_rate, channels);
    Ok(buffer)
}

/// Save audio buffer as WAV to stdout
fn save_wav_to_stdout(audio: &voirs_sdk::AudioBuffer) -> Result<()> {
    use std::io::Write;

    let stdout = std::io::stdout();
    let mut handle = stdout.lock();

    let samples = audio.samples();
    let sample_rate = audio.sample_rate();
    let channels = audio.channels() as u16;
    let bits_per_sample = 16u16;

    // Calculate sizes
    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample) / 8;
    let block_align = channels * bits_per_sample / 8;
    let data_size = num_samples * u32::from(bits_per_sample) / 8;
    let data_size_plus_36 = data_size + 36;

    // Write RIFF header
    handle.write_all(b"RIFF")?;
    handle.write_all(&data_size_plus_36.to_le_bytes())?;
    handle.write_all(b"WAVE")?;

    // Write fmt chunk
    handle.write_all(b"fmt ")?;
    handle.write_all(&16u32.to_le_bytes())?; // Subchunk1Size (16 for PCM)
    handle.write_all(&1u16.to_le_bytes())?; // AudioFormat (1 = PCM)
    handle.write_all(&channels.to_le_bytes())?;
    handle.write_all(&sample_rate.to_le_bytes())?;
    handle.write_all(&byte_rate.to_le_bytes())?;
    handle.write_all(&block_align.to_le_bytes())?;
    handle.write_all(&bits_per_sample.to_le_bytes())?;

    // Write data chunk
    handle.write_all(b"data")?;
    handle.write_all(&data_size.to_le_bytes())?;

    // Write audio samples (convert f32 [-1.0, 1.0] to i16)
    for &sample in samples {
        let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        handle.write_all(&sample_i16.to_le_bytes())?;
    }

    handle.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.max_chunk_size, 1000);
        assert_eq!(config.max_words_per_chunk, 200);
        assert_eq!(config.chunk_overlap, 50);
        assert_eq!(config.max_concurrent_chunks, 4);
    }

    #[test]
    fn test_split_text_into_chunks() {
        let text = "This is the first sentence. This is the second sentence! And this is the third sentence?";
        let config = StreamingConfig {
            max_chunk_size: 50,
            max_words_per_chunk: 10,
            ..Default::default()
        };

        let chunks = split_text_into_chunks(text, &config).unwrap();
        assert!(chunks.len() > 1);
        assert!(chunks
            .iter()
            .all(|chunk| chunk.len() <= config.max_chunk_size));
    }

    #[test]
    fn test_split_text_single_chunk() {
        let text = "Short text.";
        let config = StreamingConfig::default();

        let chunks = split_text_into_chunks(text, &config).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_split_text_empty() {
        let text = "";
        let config = StreamingConfig::default();

        let chunks = split_text_into_chunks(text, &config).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }
}
