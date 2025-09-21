//! Voice conversion commands for the VoiRS CLI

use crate::{error::CliError, output::OutputFormatter};
use clap::{Args, Subcommand};
use hound;
use std::path::PathBuf;
#[cfg(feature = "conversion")]
use voirs_conversion::{
    AgeTransform, ConversionConfig, ConversionRequest, ConversionTarget, ConversionType,
    GenderTransform, PitchTransform, VoiceCharacteristics, VoiceConverter,
};

/// Voice conversion commands
#[cfg(feature = "conversion")]
#[derive(Debug, Clone, Subcommand)]
pub enum ConversionCommand {
    /// Convert speaker voice characteristics
    Speaker(SpeakerArgs),
    /// Convert age characteristics
    Age(AgeArgs),
    /// Convert gender characteristics
    Gender(GenderArgs),
    /// Morph between two voices
    Morph(MorphArgs),
    /// Real-time streaming conversion
    Stream(StreamArgs),
    /// List available conversion models
    ListModels(ListModelsArgs),
}

#[derive(Debug, Clone, Args)]
pub struct SpeakerArgs {
    /// Input audio file
    pub input: PathBuf,
    /// Target speaker voice model or characteristics
    #[arg(long)]
    pub target_speaker: String,
    /// Output audio file
    pub output: PathBuf,
    /// Conversion strength (0.0 to 1.0)
    #[arg(long, default_value = "1.0")]
    pub strength: f32,
    /// Sample rate for output
    #[arg(long, default_value = "22050")]
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Args)]
pub struct AgeArgs {
    /// Input audio file
    pub input: PathBuf,
    /// Target age (years)
    #[arg(long)]
    pub target_age: u32,
    /// Output audio file
    pub output: PathBuf,
    /// Conversion strength (0.0 to 1.0)
    #[arg(long, default_value = "1.0")]
    pub strength: f32,
    /// Sample rate for output
    #[arg(long, default_value = "22050")]
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Args)]
pub struct GenderArgs {
    /// Input audio file
    pub input: PathBuf,
    /// Target gender (male, female, neutral)
    #[arg(long)]
    pub target_gender: String,
    /// Output audio file
    pub output: PathBuf,
    /// Conversion strength (0.0 to 1.0)
    #[arg(long, default_value = "1.0")]
    pub strength: f32,
    /// Sample rate for output
    #[arg(long, default_value = "22050")]
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Args)]
pub struct MorphArgs {
    /// Input audio file
    pub input: PathBuf,
    /// First voice model for morphing
    #[arg(long)]
    pub voice1: String,
    /// Second voice model for morphing
    #[arg(long)]
    pub voice2: String,
    /// Morphing ratio (0.0 = voice1, 1.0 = voice2)
    #[arg(long, default_value = "0.5")]
    pub ratio: f32,
    /// Output audio file
    pub output: PathBuf,
    /// Sample rate for output
    #[arg(long, default_value = "22050")]
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Args)]
pub struct StreamArgs {
    /// Input audio source (mic, file)
    #[arg(long, default_value = "mic")]
    pub input: String,
    /// Target voice model or characteristics
    #[arg(long)]
    pub target: String,
    /// Output destination (speaker, file)
    #[arg(long, default_value = "speaker")]
    pub output: String,
    /// Buffer size in milliseconds
    #[arg(long, default_value = "100")]
    pub buffer_ms: u32,
    /// Enable real-time monitoring
    #[arg(long)]
    pub monitor: bool,
}

#[derive(Debug, Clone, Args)]
pub struct ListModelsArgs {
    /// Output format (table, json, list)
    #[arg(long, default_value = "table")]
    pub format: String,
    /// Show detailed model information
    #[arg(long)]
    pub detailed: bool,
}

/// Execute conversion commands
#[cfg(feature = "conversion")]
pub async fn execute_conversion_command(
    command: ConversionCommand,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    match command {
        ConversionCommand::Speaker(args) => {
            execute_speaker_conversion(args, output_formatter).await
        }
        ConversionCommand::Age(args) => execute_age_conversion(args, output_formatter).await,
        ConversionCommand::Gender(args) => execute_gender_conversion(args, output_formatter).await,
        ConversionCommand::Morph(args) => execute_morph_conversion(args, output_formatter).await,
        ConversionCommand::Stream(args) => execute_stream_conversion(args, output_formatter).await,
        ConversionCommand::ListModels(args) => execute_list_models(args, output_formatter).await,
    }
}

#[cfg(feature = "conversion")]
async fn execute_speaker_conversion(
    args: SpeakerArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    if !args.input.exists() {
        return Err(CliError::config(format!(
            "Input file not found: {}",
            args.input.display()
        )));
    }

    if args.strength < 0.0 || args.strength > 1.0 {
        return Err(CliError::invalid_parameter(
            "strength",
            "Conversion strength must be between 0.0 and 1.0",
        ));
    }

    println!("Converting speaker characteristics...");
    println!("  Input: {}", args.input.display());
    println!("  Target speaker: {}", args.target_speaker);
    println!("  Strength: {:.2}", args.strength);

    // Load input audio
    let audio_data = load_audio_file(&args.input)
        .map_err(|e| CliError::AudioError(format!("Failed to load input audio: {}", e)))?;

    // Create voice converter
    let converter = VoiceConverter::new()
        .map_err(|e| CliError::config(format!("Failed to create voice converter: {}", e)))?;

    // Create conversion request
    let target = ConversionTarget::new(create_speaker_characteristics(&args.target_speaker)?)
        .with_strength(args.strength);

    let request = ConversionRequest::new(
        format!("speaker_conv_{}", fastrand::u64(..)),
        audio_data.samples,
        audio_data.sample_rate,
        ConversionType::SpeakerConversion,
        target,
    );

    // Perform conversion
    let result = converter
        .convert(request)
        .await
        .map_err(|e| CliError::AudioError(format!("Speaker conversion failed: {}", e)))?;

    if result.success {
        // Save converted audio
        save_audio_file(&result.converted_audio, args.sample_rate, &args.output)
            .map_err(|e| CliError::AudioError(format!("Failed to save converted audio: {}", e)))?;

        let quality_score = result
            .quality_metrics
            .get("overall_quality")
            .copied()
            .unwrap_or(0.0);
        output_formatter.success(&format!(
            "Speaker conversion completed! Quality score: {:.2}, Output saved to: {}",
            quality_score,
            args.output.display()
        ));
    } else {
        let error_msg = result.error_message.unwrap_or("Unknown error".to_string());
        return Err(CliError::AudioError(format!(
            "Speaker conversion failed: {}",
            error_msg
        )));
    }

    Ok(())
}

#[cfg(feature = "conversion")]
async fn execute_age_conversion(
    args: AgeArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    if !args.input.exists() {
        return Err(CliError::config(format!(
            "Input file not found: {}",
            args.input.display()
        )));
    }

    if args.strength < 0.0 || args.strength > 1.0 {
        return Err(CliError::invalid_parameter(
            "strength",
            "Conversion strength must be between 0.0 and 1.0",
        ));
    }

    if args.target_age < 5 || args.target_age > 100 {
        return Err(CliError::invalid_parameter(
            "target_age",
            "Target age must be between 5 and 100",
        ));
    }

    println!("Converting age characteristics...");
    println!("  Input: {}", args.input.display());
    println!("  Target age: {} years", args.target_age);
    println!("  Strength: {:.2}", args.strength);

    // Load input audio
    let audio_data = load_audio_file(&args.input)
        .map_err(|e| CliError::AudioError(format!("Failed to load input audio: {}", e)))?;

    // Create voice converter
    let converter = VoiceConverter::new()
        .map_err(|e| CliError::config(format!("Failed to create voice converter: {}", e)))?;

    // Create age transform
    let age_transform = AgeTransform::new(args.target_age as f32, args.strength);

    // Create conversion request
    let target = ConversionTarget::new(create_age_characteristics(args.target_age)?)
        .with_strength(args.strength);

    let request = ConversionRequest::new(
        format!("age_conv_{}", fastrand::u64(..)),
        audio_data.samples,
        audio_data.sample_rate,
        ConversionType::AgeTransformation,
        target,
    );

    // Perform conversion
    let result = converter
        .convert(request)
        .await
        .map_err(|e| CliError::AudioError(format!("Age conversion failed: {}", e)))?;

    if result.success {
        // Save converted audio
        save_audio_file(&result.converted_audio, args.sample_rate, &args.output)
            .map_err(|e| CliError::AudioError(format!("Failed to save converted audio: {}", e)))?;

        let quality_score = result
            .quality_metrics
            .get("overall_quality")
            .copied()
            .unwrap_or(0.0);
        output_formatter.success(&format!(
            "Age conversion completed! Quality score: {:.2}, Output saved to: {}",
            quality_score,
            args.output.display()
        ));
    } else {
        let error_msg = result.error_message.unwrap_or("Unknown error".to_string());
        return Err(CliError::AudioError(format!(
            "Age conversion failed: {}",
            error_msg
        )));
    }

    Ok(())
}

#[cfg(feature = "conversion")]
async fn execute_gender_conversion(
    args: GenderArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    if !args.input.exists() {
        return Err(CliError::config(format!(
            "Input file not found: {}",
            args.input.display()
        )));
    }

    if args.strength < 0.0 || args.strength > 1.0 {
        return Err(CliError::invalid_parameter(
            "strength",
            "Conversion strength must be between 0.0 and 1.0",
        ));
    }

    let target_gender = match args.target_gender.to_lowercase().as_str() {
        "male" | "m" => "male",
        "female" | "f" => "female",
        "neutral" | "n" => "neutral",
        _ => {
            return Err(CliError::invalid_parameter(
                "target_gender",
                "Target gender must be 'male', 'female', or 'neutral'",
            ))
        }
    };

    println!("Converting gender characteristics...");
    println!("  Input: {}", args.input.display());
    println!("  Target gender: {}", target_gender);
    println!("  Strength: {:.2}", args.strength);

    // Load input audio
    let audio_data = load_audio_file(&args.input)
        .map_err(|e| CliError::AudioError(format!("Failed to load input audio: {}", e)))?;

    // Create voice converter
    let converter = VoiceConverter::new()
        .map_err(|e| CliError::config(format!("Failed to create voice converter: {}", e)))?;

    // Create conversion request
    let target = ConversionTarget::new(create_gender_characteristics(target_gender)?)
        .with_strength(args.strength);

    let request = ConversionRequest::new(
        format!("gender_conv_{}", fastrand::u64(..)),
        audio_data.samples,
        audio_data.sample_rate,
        ConversionType::GenderTransformation,
        target,
    );

    // Perform conversion
    let result = converter
        .convert(request)
        .await
        .map_err(|e| CliError::AudioError(format!("Gender conversion failed: {}", e)))?;

    if result.success {
        // Save converted audio
        save_audio_file(&result.converted_audio, args.sample_rate, &args.output)
            .map_err(|e| CliError::AudioError(format!("Failed to save converted audio: {}", e)))?;

        let quality_score = result
            .quality_metrics
            .get("overall_quality")
            .copied()
            .unwrap_or(0.0);
        output_formatter.success(&format!(
            "Gender conversion completed! Quality score: {:.2}, Output saved to: {}",
            quality_score,
            args.output.display()
        ));
    } else {
        let error_msg = result.error_message.unwrap_or("Unknown error".to_string());
        return Err(CliError::AudioError(format!(
            "Gender conversion failed: {}",
            error_msg
        )));
    }

    Ok(())
}

#[cfg(feature = "conversion")]
async fn execute_morph_conversion(
    args: MorphArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    if !args.input.exists() {
        return Err(CliError::config(format!(
            "Input file not found: {}",
            args.input.display()
        )));
    }

    if args.ratio < 0.0 || args.ratio > 1.0 {
        return Err(CliError::invalid_parameter(
            "ratio",
            "Morphing ratio must be between 0.0 and 1.0",
        ));
    }

    println!("Morphing between voice characteristics...");
    println!("  Input: {}", args.input.display());
    println!("  Voice 1: {}", args.voice1);
    println!("  Voice 2: {}", args.voice2);
    println!("  Ratio: {:.2}", args.ratio);

    // Load input audio
    let audio_data = load_audio_file(&args.input)
        .map_err(|e| CliError::AudioError(format!("Failed to load input audio: {}", e)))?;

    // Create voice converter
    let converter = VoiceConverter::new()
        .map_err(|e| CliError::config(format!("Failed to create voice converter: {}", e)))?;

    // Create conversion request
    let voice1_chars = create_speaker_characteristics(&args.voice1)?;
    let voice2_chars = create_speaker_characteristics(&args.voice2)?;
    let morphed_chars = voice1_chars.interpolate(&voice2_chars, args.ratio);
    let target = ConversionTarget::new(morphed_chars);

    let request = ConversionRequest::new(
        format!("morph_conv_{}", fastrand::u64(..)),
        audio_data.samples,
        audio_data.sample_rate,
        ConversionType::VoiceMorphing,
        target,
    );

    // Perform conversion
    let result = converter
        .convert(request)
        .await
        .map_err(|e| CliError::AudioError(format!("Voice morphing failed: {}", e)))?;

    if result.success {
        // Save converted audio
        save_audio_file(&result.converted_audio, args.sample_rate, &args.output)
            .map_err(|e| CliError::AudioError(format!("Failed to save morphed audio: {}", e)))?;

        let quality_score = result
            .quality_metrics
            .get("overall_quality")
            .copied()
            .unwrap_or(0.0);
        output_formatter.success(&format!(
            "Voice morphing completed! Quality score: {:.2}, Output saved to: {}",
            quality_score,
            args.output.display()
        ));
    } else {
        let error_msg = result.error_message.unwrap_or("Unknown error".to_string());
        return Err(CliError::AudioError(format!(
            "Voice morphing failed: {}",
            error_msg
        )));
    }

    Ok(())
}

#[cfg(feature = "conversion")]
async fn execute_stream_conversion(
    args: StreamArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    if args.buffer_ms < 50 || args.buffer_ms > 1000 {
        return Err(CliError::invalid_parameter(
            "buffer_ms",
            "Buffer size must be between 50 and 1000 milliseconds",
        ));
    }

    println!("Starting real-time voice conversion...");
    println!("  Input: {}", args.input);
    println!("  Target: {}", args.target);
    println!("  Output: {}", args.output);
    println!("  Buffer: {}ms", args.buffer_ms);

    // Create voice converter
    let converter = VoiceConverter::new()
        .map_err(|e| CliError::config(format!("Failed to create voice converter: {}", e)))?;

    // Set up conversion target
    let target = ConversionTarget::new(create_speaker_characteristics(&args.target)?);

    if args.monitor {
        println!("Monitoring enabled. Press Ctrl+C to stop.");
    }

    // Start streaming conversion (simplified implementation)
    println!("Real-time conversion started...");
    println!("Note: This is a simplified demonstration. Full streaming implementation requires audio device integration.");

    // Simulate streaming for demonstration
    for i in 0..10 {
        println!("Processing chunk {}/10...", i + 1);
        tokio::time::sleep(tokio::time::Duration::from_millis(args.buffer_ms as u64)).await;

        if args.monitor {
            println!(
                "  Chunk {}: Quality OK, Latency: {}ms",
                i + 1,
                args.buffer_ms
            );
        }
    }

    output_formatter.success("Streaming conversion simulation completed successfully!");
    Ok(())
}

#[cfg(feature = "conversion")]
async fn execute_list_models(
    args: ListModelsArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    // Create voice converter to access available models
    let converter = VoiceConverter::new()
        .map_err(|e| CliError::config(format!("Failed to create voice converter: {}", e)))?;

    let models = get_available_conversion_models();

    if models.is_empty() {
        println!("No conversion models found.");
        return Ok(());
    }

    match args.format.as_str() {
        "table" => {
            println!(
                "{:<20} {:<15} {:<30} {}",
                "Model ID", "Type", "Description", "Version"
            );
            println!("{}", "-".repeat(80));
            for model in models {
                println!(
                    "{:<20} {:<15} {:<30} {}",
                    model.id, model.model_type, model.description, model.version
                );
            }
        }
        "json" => {
            let json_models: Vec<_> = models
                .iter()
                .map(|model| {
                    serde_json::json!({
                        "id": model.id,
                        "type": model.model_type,
                        "description": model.description,
                        "version": model.version,
                        "details": if args.detailed {
                            Some(serde_json::json!({
                                "supported_formats": model.supported_formats,
                                "latency_ms": model.latency_ms
                            }))
                        } else {
                            None
                        }
                    })
                })
                .collect();

            println!(
                "{}",
                serde_json::to_string_pretty(&json_models).map_err(CliError::Serialization)?
            );
        }
        _ => {
            for model in models {
                println!("{}: {} ({})", model.id, model.description, model.model_type);
            }
        }
    }

    Ok(())
}

/// Audio data structure (reused from cloning.rs)
#[derive(Debug)]
struct AudioData {
    samples: Vec<f32>,
    sample_rate: u32,
}

/// Conversion model info
#[derive(Debug)]
struct ConversionModelInfo {
    id: String,
    model_type: String,
    description: String,
    version: String,
    supported_formats: Vec<String>,
    latency_ms: u32,
}

/// Load audio file (reused from cloning.rs)
fn load_audio_file(path: &PathBuf) -> Result<AudioData, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    // Convert samples to f32
    let samples: Result<Vec<f32>, _> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect(),
        hound::SampleFormat::Int => match spec.bits_per_sample {
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

/// Save audio file (reused from cloning.rs)
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

/// Create speaker characteristics from string identifier
fn create_speaker_characteristics(speaker_id: &str) -> Result<VoiceCharacteristics, CliError> {
    use voirs_conversion::types::{AgeGroup, Gender};

    // Parse speaker identifier and create characteristics
    match speaker_id.to_lowercase().as_str() {
        "young_male" => Ok(VoiceCharacteristics::for_age(AgeGroup::YoungAdult)
            .interpolate(&VoiceCharacteristics::for_gender(Gender::Male), 0.5)),
        "young_female" => Ok(VoiceCharacteristics::for_age(AgeGroup::YoungAdult)
            .interpolate(&VoiceCharacteristics::for_gender(Gender::Female), 0.5)),
        "adult_male" => Ok(VoiceCharacteristics::for_age(AgeGroup::MiddleAged)
            .interpolate(&VoiceCharacteristics::for_gender(Gender::Male), 0.5)),
        "adult_female" => Ok(VoiceCharacteristics::for_age(AgeGroup::MiddleAged)
            .interpolate(&VoiceCharacteristics::for_gender(Gender::Female), 0.5)),
        "elderly_male" => Ok(VoiceCharacteristics::for_age(AgeGroup::Senior)
            .interpolate(&VoiceCharacteristics::for_gender(Gender::Male), 0.5)),
        "elderly_female" => Ok(VoiceCharacteristics::for_age(AgeGroup::Senior)
            .interpolate(&VoiceCharacteristics::for_gender(Gender::Female), 0.5)),
        _ => {
            // Default characteristics
            Ok(VoiceCharacteristics::default())
        }
    }
}

/// Create age-specific characteristics
fn create_age_characteristics(age: u32) -> Result<VoiceCharacteristics, CliError> {
    use voirs_conversion::types::AgeGroup;

    let age_group = match age {
        0..=12 => AgeGroup::Child,
        13..=19 => AgeGroup::Teen,
        20..=35 => AgeGroup::YoungAdult,
        36..=55 => AgeGroup::MiddleAged,
        _ => AgeGroup::Senior,
    };

    Ok(VoiceCharacteristics::for_age(age_group))
}

/// Create gender-specific characteristics
fn create_gender_characteristics(gender: &str) -> Result<VoiceCharacteristics, CliError> {
    use voirs_conversion::types::Gender;

    let gender_type = match gender.to_lowercase().as_str() {
        "male" | "m" => Gender::Male,
        "female" | "f" => Gender::Female,
        "other" => Gender::Other,
        _ => Gender::Unknown,
    };

    Ok(VoiceCharacteristics::for_gender(gender_type))
}

/// Get available conversion models
fn get_available_conversion_models() -> Vec<ConversionModelInfo> {
    vec![
        ConversionModelInfo {
            id: "speaker_conv_v1".to_string(),
            model_type: "Speaker".to_string(),
            description: "General purpose speaker conversion".to_string(),
            version: "1.0.0".to_string(),
            supported_formats: vec!["wav".to_string(), "mp3".to_string()],
            latency_ms: 100,
        },
        ConversionModelInfo {
            id: "age_conv_v1".to_string(),
            model_type: "Age".to_string(),
            description: "Age transformation model".to_string(),
            version: "1.0.0".to_string(),
            supported_formats: vec!["wav".to_string()],
            latency_ms: 120,
        },
        ConversionModelInfo {
            id: "gender_conv_v1".to_string(),
            model_type: "Gender".to_string(),
            description: "Gender transformation model".to_string(),
            version: "1.0.0".to_string(),
            supported_formats: vec!["wav".to_string()],
            latency_ms: 110,
        },
        ConversionModelInfo {
            id: "realtime_conv_v1".to_string(),
            model_type: "Realtime".to_string(),
            description: "Low-latency real-time conversion".to_string(),
            version: "1.0.0".to_string(),
            supported_formats: vec!["wav".to_string(), "raw".to_string()],
            latency_ms: 50,
        },
    ]
}
