//! Voice cloning commands for the VoiRS CLI

use crate::{error::CliError, output::OutputFormatter};
use clap::{Args, Subcommand};
use dasp::{Frame, Sample};
use hound;
use std::path::PathBuf;
#[cfg(feature = "cloning")]
use voirs_cloning::{
    CloningConfig, CloningMethod, SpeakerProfile, VoiceCloneRequest, VoiceCloner, VoiceSample,
};

/// Voice cloning commands
#[cfg(feature = "cloning")]
#[derive(Debug, Clone, Subcommand)]
pub enum CloningCommand {
    /// Clone voice from reference samples
    Clone(CloneArgs),
    /// Quick clone from single audio file
    Quick(QuickCloneArgs),
    /// List cached speaker profiles
    ListProfiles(ListProfilesArgs),
    /// Validate reference audio for cloning
    Validate(ValidateArgs),
    /// Clear speaker cache
    ClearCache(ClearCacheArgs),
}

#[derive(Debug, Clone, Args)]
pub struct CloneArgs {
    /// Reference audio files (multiple samples for better quality)
    #[arg(long, required = true)]
    pub reference_files: Vec<PathBuf>,
    /// Text to synthesize with cloned voice
    pub text: String,
    /// Output audio file path
    pub output: PathBuf,
    /// Cloning method (few-shot, one-shot, zero-shot, fine-tuning)
    #[arg(long, default_value = "few-shot")]
    pub method: String,
    /// Speaker name/ID for caching
    #[arg(long)]
    pub speaker_id: Option<String>,
    /// Quality threshold (0.0 to 1.0)
    #[arg(long, default_value = "0.7")]
    pub quality_threshold: f32,
    /// Sample rate for output audio
    #[arg(long, default_value = "22050")]
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Args)]
pub struct QuickCloneArgs {
    /// Single reference audio file
    pub reference_file: PathBuf,
    /// Text to synthesize with cloned voice
    pub text: String,
    /// Output audio file path
    pub output: PathBuf,
    /// Sample rate for output audio
    #[arg(long, default_value = "22050")]
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Args)]
pub struct ListProfilesArgs {
    /// Output format for the profile list
    #[arg(long, default_value = "table")]
    pub format: String,
    /// Show detailed profile information
    #[arg(long)]
    pub detailed: bool,
}

#[derive(Debug, Clone, Args)]
pub struct ValidateArgs {
    /// Reference audio files to validate
    #[arg(required = true)]
    pub audio_files: Vec<PathBuf>,
    /// Output validation report format
    #[arg(long, default_value = "table")]
    pub format: String,
    /// Minimum quality threshold
    #[arg(long, default_value = "0.6")]
    pub min_quality: f32,
}

#[derive(Debug, Clone, Args)]
pub struct ClearCacheArgs {
    /// Confirm cache clearing without prompt
    #[arg(long)]
    pub yes: bool,
}

/// Execute cloning commands
#[cfg(feature = "cloning")]
pub async fn execute_cloning_command(
    command: CloningCommand,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    match command {
        CloningCommand::Clone(args) => execute_clone(args, output_formatter).await,
        CloningCommand::Quick(args) => execute_quick_clone(args, output_formatter).await,
        CloningCommand::ListProfiles(args) => execute_list_profiles(args, output_formatter).await,
        CloningCommand::Validate(args) => execute_validate(args, output_formatter).await,
        CloningCommand::ClearCache(args) => execute_clear_cache(args, output_formatter).await,
    }
}

#[cfg(feature = "cloning")]
async fn execute_clone(
    args: CloneArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    // Validate cloning method
    let method = match args.method.to_lowercase().as_str() {
        "few-shot" | "few_shot" => CloningMethod::FewShot,
        "one-shot" | "one_shot" => CloningMethod::OneShot,
        "zero-shot" | "zero_shot" => CloningMethod::ZeroShot,
        "fine-tuning" | "fine_tuning" => CloningMethod::FineTuning,
        "voice-conversion" | "voice_conversion" => CloningMethod::VoiceConversion,
        "hybrid" => CloningMethod::Hybrid,
        _ => return Err(CliError::invalid_parameter("method", "Invalid cloning method. Use: few-shot, one-shot, zero-shot, fine-tuning, voice-conversion, or hybrid")),
    };

    if args.quality_threshold < 0.0 || args.quality_threshold > 1.0 {
        return Err(CliError::invalid_parameter(
            "quality_threshold",
            "Quality threshold must be between 0.0 and 1.0",
        ));
    }

    // Load reference audio files
    println!(
        "Loading {} reference audio files...",
        args.reference_files.len()
    );
    let mut voice_samples = Vec::new();

    for (i, file_path) in args.reference_files.iter().enumerate() {
        if !file_path.exists() {
            return Err(CliError::config(format!(
                "Reference file not found: {}",
                file_path.display()
            )));
        }

        println!("  Loading sample {}: {}", i + 1, file_path.display());

        // Load actual audio file
        let audio_data = load_audio_file(file_path).map_err(|e| {
            CliError::config(format!(
                "Failed to load audio file {}: {}",
                file_path.display(),
                e
            ))
        })?;

        let sample = VoiceSample::new(
            format!("sample_{}", i),
            audio_data.samples,
            audio_data.sample_rate,
        );
        voice_samples.push(sample);
    }

    // Create voice cloner
    let cloner = VoiceCloner::new()
        .map_err(|e| CliError::config(format!("Failed to create voice cloner: {}", e)))?;

    // Create speaker profile
    let speaker_id = args
        .speaker_id
        .unwrap_or_else(|| format!("speaker_{}", fastrand::u64(..)));

    println!("Cloning voice with method: {:?}", method);
    println!("Speaker ID: {}", speaker_id);
    println!("Target text: '{}'", args.text);

    // Perform cloning
    println!("Processing voice cloning...");
    let mut speaker_data = voirs_cloning::SpeakerData::new(SpeakerProfile::new(
        speaker_id.clone(),
        speaker_id.clone(),
    ));

    // Add voice samples to speaker data
    speaker_data.reference_samples = voice_samples;

    let request = VoiceCloneRequest::new(
        format!("clone_{}", fastrand::u64(..)),
        speaker_data,
        method,
        args.text.clone(),
    );

    let result = cloner
        .clone_voice(request)
        .await
        .map_err(|e| CliError::config(format!("Voice cloning failed: {}", e)))?;

    if result.success {
        // Save audio to output file
        save_audio_file(&result.audio, args.sample_rate, &args.output)
            .map_err(|e| CliError::AudioError(format!("Failed to save audio: {}", e)))?;

        output_formatter.success(&format!(
            "Voice cloning completed! Quality score: {:.2}, Output saved to: {}",
            result.similarity_score,
            args.output.display()
        ));
    } else {
        let error_msg = result.error_message.unwrap_or("Unknown error".to_string());
        return Err(CliError::AudioError(format!(
            "Voice cloning failed: {}",
            error_msg
        )));
    }

    Ok(())
}

#[cfg(feature = "cloning")]
async fn execute_quick_clone(
    args: QuickCloneArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    if !args.reference_file.exists() {
        return Err(CliError::config(format!(
            "Reference file not found: {}",
            args.reference_file.display()
        )));
    }

    println!("Quick cloning from: {}", args.reference_file.display());
    println!("Target text: '{}'", args.text);

    // Load reference audio
    let audio_data = load_audio_file(&args.reference_file)
        .map_err(|e| CliError::AudioError(format!("Failed to load reference audio: {}", e)))?;

    // Create voice cloner
    let cloner = VoiceCloner::new()
        .map_err(|e| CliError::config(format!("Failed to create voice cloner: {}", e)))?;

    // Create voice sample and speaker data
    let voice_sample = VoiceSample::new(
        "quick_clone_sample".to_string(),
        audio_data.samples,
        audio_data.sample_rate,
    );

    let mut speaker_data = voirs_cloning::SpeakerData::new(SpeakerProfile::new(
        "quick_clone_speaker".to_string(),
        "Quick Clone".to_string(),
    ));
    speaker_data.reference_samples = vec![voice_sample];

    let request = VoiceCloneRequest::new(
        format!("quick_clone_{}", fastrand::u64(..)),
        speaker_data,
        CloningMethod::OneShot,
        args.text.clone(),
    );

    // Perform quick cloning
    println!("Processing quick voice cloning...");
    let result = cloner
        .clone_voice(request)
        .await
        .map_err(|e| CliError::AudioError(format!("Quick cloning failed: {}", e)))?;

    if result.success {
        // Save audio to output file
        save_audio_file(&result.audio, args.sample_rate, &args.output)
            .map_err(|e| CliError::AudioError(format!("Failed to save audio: {}", e)))?;

        output_formatter.success(&format!(
            "Quick cloning completed! Quality score: {:.2}, Output saved to: {}",
            result.similarity_score,
            args.output.display()
        ));
    } else {
        let error_msg = result.error_message.unwrap_or("Unknown error".to_string());
        return Err(CliError::AudioError(format!(
            "Quick cloning failed: {}",
            error_msg
        )));
    }

    Ok(())
}

#[cfg(feature = "cloning")]
async fn execute_list_profiles(
    args: ListProfilesArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    // Create voice cloner to access cached profiles
    let cloner = VoiceCloner::new()
        .map_err(|e| CliError::config(format!("Failed to create voice cloner: {}", e)))?;

    let profiles = cloner.list_cached_speakers().await;

    if profiles.is_empty() {
        println!("No cached speaker profiles found.");
        return Ok(());
    }

    match args.format.as_str() {
        "table" => {
            println!("{:<20} {:<30} {}", "Speaker ID", "Description", "Samples");
            println!("{}", "-".repeat(60));
            for profile in profiles {
                println!(
                    "{:<20} {:<30} {}",
                    profile.id,
                    profile.name,
                    profile.samples.len()
                );
            }
        }
        "json" => {
            let json_profiles: Vec<_> = profiles
                .iter()
                .map(|id| {
                    serde_json::json!({
                        "speaker_id": id,
                        "description": "Cached speaker",
                        "sample_count": "N/A",
                        "details": if args.detailed {
                            Some(serde_json::json!({"cached": true}))
                        } else {
                            None
                        }
                    })
                })
                .collect();

            println!(
                "{}",
                serde_json::to_string_pretty(&json_profiles).map_err(CliError::Serialization)?
            );
        }
        _ => {
            for profile in profiles {
                println!("{}: {}", profile.id, profile.name);
            }
        }
    }

    Ok(())
}

#[cfg(feature = "cloning")]
async fn execute_validate(
    args: ValidateArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    if args.min_quality < 0.0 || args.min_quality > 1.0 {
        return Err(CliError::invalid_parameter(
            "min_quality",
            "Minimum quality must be between 0.0 and 1.0",
        ));
    }

    println!(
        "Validating {} audio files for cloning...",
        args.audio_files.len()
    );

    let mut validation_results = Vec::new();
    let mut all_valid = true;

    for (i, file_path) in args.audio_files.iter().enumerate() {
        if !file_path.exists() {
            validation_results.push((
                format!("File {}", i + 1),
                file_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
                "NOT_FOUND".to_string(),
                0.0,
            ));
            all_valid = false;
            continue;
        }

        println!("  Validating: {}", file_path.display());

        // Load and validate audio file
        let (quality_score, status) = match load_audio_file(file_path) {
            Ok(audio_data) => {
                let quality = validate_audio_quality(&audio_data);
                let status = if quality >= args.min_quality {
                    "VALID"
                } else {
                    all_valid = false;
                    "LOW_QUALITY"
                };
                (quality, status)
            }
            Err(_) => {
                all_valid = false;
                (0.0, "LOAD_ERROR")
            }
        };

        validation_results.push((
            format!("File {}", i + 1),
            file_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            status.to_string(),
            quality_score,
        ));
    }

    // Display results
    match args.format.as_str() {
        "table" => {
            println!(
                "{:<10} {:<30} {:<12} {}",
                "File", "Name", "Status", "Quality"
            );
            println!("{}", "-".repeat(70));
            for (file_num, name, status, quality) in validation_results {
                println!(
                    "{:<10} {:<30} {:<12} {:.2}",
                    file_num, name, status, quality
                );
            }
        }
        "json" => {
            let json_results: Vec<_> = validation_results
                .into_iter()
                .map(|(file_num, name, status, quality)| {
                    serde_json::json!({
                        "file": file_num,
                        "filename": name,
                        "status": status,
                        "quality_score": quality
                    })
                })
                .collect();

            println!(
                "{}",
                serde_json::to_string_pretty(&json_results).map_err(CliError::Serialization)?
            );
        }
        _ => {
            for (file_num, name, status, quality) in validation_results {
                println!(
                    "{} ({}): {} - Quality: {:.2}",
                    file_num, name, status, quality
                );
            }
        }
    }

    if all_valid {
        output_formatter.success("All audio files are valid for voice cloning!");
    } else {
        output_formatter
            .warning("Some audio files may not be suitable for high-quality voice cloning.");
    }

    Ok(())
}

#[cfg(feature = "cloning")]
async fn execute_clear_cache(
    args: ClearCacheArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    if !args.yes {
        println!("This will clear all cached speaker profiles. Continue? (y/N)");
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(CliError::Io)?;

        if !input.trim().to_lowercase().starts_with('y') {
            println!("Cache clearing cancelled.");
            return Ok(());
        }
    }

    // Create voice cloner to clear cache
    let cloner = VoiceCloner::new()
        .map_err(|e| CliError::config(format!("Failed to create voice cloner: {}", e)))?;

    cloner
        .clear_cache()
        .await
        .map_err(|e| CliError::config(format!("Failed to clear cache: {}", e)))?;

    output_formatter.success("Speaker profile cache cleared successfully!");
    Ok(())
}

/// Audio data structure
#[derive(Debug)]
struct AudioData {
    samples: Vec<f32>,
    sample_rate: u32,
}

/// Load audio file and convert to f32 samples
fn load_audio_file(path: &PathBuf) -> Result<AudioData, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    // Convert samples to f32
    let samples: Result<Vec<f32>, _> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect(),
        hound::SampleFormat::Int => match spec.bits_per_sample {
            8 => reader
                .samples::<i8>()
                .map(|s| s.map(|sample| sample as f32 / i8::MAX as f32))
                .collect(),
            16 => reader
                .samples::<i16>()
                .map(|s| s.map(|sample| sample as f32 / i16::MAX as f32))
                .collect(),
            24 => reader
                .samples::<i32>()
                .map(|s| s.map(|sample| (sample >> 8) as f32 / (i32::MAX >> 8) as f32))
                .collect(),
            32 => reader
                .samples::<i32>()
                .map(|s| s.map(|sample| sample as f32 / i32::MAX as f32))
                .collect(),
            _ => {
                return Err(format!("Unsupported bit depth: {}", spec.bits_per_sample).into());
            }
        },
    };

    let samples = samples?;

    // Convert to mono if stereo
    let mono_samples = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|frame| (frame[0] + frame[1]) / 2.0)
            .collect()
    } else {
        samples
    };

    Ok(AudioData {
        samples: mono_samples,
        sample_rate: spec.sample_rate,
    })
}

/// Save audio data to WAV file
fn save_audio_file(
    audio_data: &[f32],
    sample_rate: u32,
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;

    for &sample in audio_data {
        let sample_i16 = (sample * i16::MAX as f32) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;
    Ok(())
}

/// Validate audio quality for voice cloning
fn validate_audio_quality(audio_data: &AudioData) -> f32 {
    let samples = &audio_data.samples;

    if samples.is_empty() {
        return 0.0;
    }

    // Calculate quality metrics
    let mut quality_score: f32 = 1.0;

    // 1. Check duration (should be at least 1 second, ideally 3-10 seconds)
    let duration_seconds = samples.len() as f32 / audio_data.sample_rate as f32;
    if duration_seconds < 1.0 {
        quality_score *= 0.3; // Very short audio
    } else if duration_seconds < 3.0 {
        quality_score *= 0.7; // Short but usable
    } else if duration_seconds > 30.0 {
        quality_score *= 0.8; // Very long, might have issues
    }

    // 2. Check for silence or very low volume
    let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
    if rms < 0.01 {
        quality_score *= 0.2; // Too quiet
    } else if rms < 0.05 {
        quality_score *= 0.6; // Quite quiet
    }

    // 3. Check for clipping
    let clipped_samples = samples.iter().filter(|&&x| x.abs() > 0.95).count();
    let clipping_ratio = clipped_samples as f32 / samples.len() as f32;
    if clipping_ratio > 0.1 {
        quality_score *= 0.4; // High clipping
    } else if clipping_ratio > 0.01 {
        quality_score *= 0.7; // Some clipping
    }

    // 4. Check dynamic range
    let max_val = samples.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let min_non_zero = samples
        .iter()
        .filter(|&&x| x.abs() > 0.001)
        .map(|x| x.abs())
        .fold(1.0f32, f32::min);

    let dynamic_range = if min_non_zero > 0.0 {
        20.0 * (max_val / min_non_zero).log10()
    } else {
        0.0
    };

    if dynamic_range < 20.0 {
        quality_score *= 0.5; // Poor dynamic range
    } else if dynamic_range < 40.0 {
        quality_score *= 0.8; // Okay dynamic range
    }

    // 5. Check sample rate appropriateness
    if audio_data.sample_rate < 16000 {
        quality_score *= 0.6; // Low sample rate
    } else if audio_data.sample_rate < 22050 {
        quality_score *= 0.9; // Acceptable sample rate
    }

    quality_score.clamp(0.0, 1.0)
}
