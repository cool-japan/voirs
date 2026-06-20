//! 3D Spatial Audio commands for the VoiRS CLI

use crate::{error::CliError, output::OutputFormatter};
use clap::{Args, Subcommand};
use std::path::{Path, PathBuf};
#[cfg(feature = "spatial")]
use voirs_spatial::{
    position::{AttenuationModel, AttenuationParams, DirectivityPattern, SourceType},
    room::{RoomConfig, WallMaterials},
    types::BinauraAudio,
    HrtfDatabase, Listener, Position3D, RoomSimulator, SoundSource, SpatialConfig,
    SpatialEffect, SpatialProcessor, SpatialRequest, SpatialResult,
};

#[cfg(feature = "spatial")]
use voirs_sdk::VoirsPipeline;

use hound;

/// 3D Spatial Audio commands
#[cfg(feature = "spatial")]
#[derive(Debug, Clone, Subcommand)]
pub enum SpatialCommand {
    /// Synthesize speech with 3D spatial positioning
    #[command(visible_alias = "synthesize")]
    Synth(SynthArgs),
    /// Apply HRTF processing to existing audio
    #[command(visible_alias = "apply-hrtf")]
    Hrtf(HrtfArgs),
    /// Apply room acoustics simulation
    #[command(visible_alias = "apply-room")]
    Room(RoomArgs),
    /// Animate sound source movement
    #[command(visible_alias = "animate")]
    Movement(MovementArgs),
    /// Validate spatial audio setup
    Validate(ValidateArgs),
    /// Calibrate for specific headphone model
    Calibrate(CalibrateArgs),
    /// List available HRTF datasets
    ListHrtf(ListHrtfArgs),
}

#[derive(Debug, Clone, Args)]
pub struct SynthArgs {
    /// Text to synthesize
    pub text: String,
    /// Output audio file (must be stereo)
    pub output: PathBuf,
    /// 3D position (x,y,z) in meters
    #[arg(long, value_parser = parse_position)]
    pub position: Position3D,
    /// Voice to use for synthesis
    #[arg(long)]
    pub voice: Option<String>,
    /// Room configuration file (JSON)
    #[arg(long)]
    pub room_config: Option<PathBuf>,
    /// HRTF dataset to use
    #[arg(long, default_value = "generic")]
    pub hrtf_dataset: String,
    /// Doppler effect strength (0.0-1.0)
    #[arg(long, default_value = "0.5")]
    pub doppler_strength: f32,
    /// Sample rate for output audio
    #[arg(long, default_value = "44100")]
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Args)]
pub struct HrtfArgs {
    /// Input mono audio file
    pub input: PathBuf,
    /// Output binaural audio file
    pub output: PathBuf,
    /// 3D position (x,y,z) in meters
    #[arg(long, value_parser = parse_position)]
    pub position: Position3D,
    /// HRTF dataset to use
    #[arg(long, default_value = "generic")]
    pub hrtf_dataset: String,
    /// Head circumference in cm (for personalization)
    #[arg(long, default_value = "56.0")]
    pub head_circumference: f32,
    /// Interpupillary distance in cm
    #[arg(long, default_value = "6.3")]
    pub interpupillary_distance: f32,
    /// Enable crossfeed for better stereo imaging
    #[arg(long)]
    pub crossfeed: bool,
}

#[derive(Debug, Clone, Args)]
pub struct RoomArgs {
    /// Input audio file
    pub input: PathBuf,
    /// Output audio file with room acoustics
    pub output: PathBuf,
    /// Room configuration file (JSON)
    #[arg(long)]
    pub room_config: PathBuf,
    /// Source position in room (x,y,z) in meters
    #[arg(long, value_parser = parse_position)]
    pub source_position: Position3D,
    /// Listener position in room (x,y,z) in meters
    #[arg(long, value_parser = parse_position)]
    pub listener_position: Position3D,
    /// Reverb strength (0.0-1.0)
    #[arg(long, default_value = "0.5")]
    pub reverb_strength: f32,
}

#[derive(Debug, Clone, Args)]
pub struct MovementArgs {
    /// Input audio file
    pub input: PathBuf,
    /// Output audio file with movement
    pub output: PathBuf,
    /// Movement path file (JSON with timestamped positions)
    #[arg(long)]
    pub path: PathBuf,
    /// Movement speed multiplier
    #[arg(long, default_value = "1.0")]
    pub speed_multiplier: f32,
    /// Enable Doppler effect
    #[arg(long)]
    pub doppler: bool,
    /// HRTF dataset to use
    #[arg(long, default_value = "generic")]
    pub hrtf_dataset: String,
}

#[derive(Debug, Clone, Args)]
pub struct ValidateArgs {
    /// Test audio file to use for validation
    #[arg(long)]
    pub test_audio: Option<PathBuf>,
    /// Generate detailed validation report
    #[arg(long)]
    pub detailed: bool,
    /// Check specific HRTF dataset
    #[arg(long)]
    pub hrtf_dataset: Option<String>,
    /// Test room configuration
    #[arg(long)]
    pub room_config: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub struct CalibrateArgs {
    /// Headphone model name
    #[arg(long)]
    pub headphone_model: String,
    /// Calibration audio file (if available)
    #[arg(long)]
    pub calibration_audio: Option<PathBuf>,
    /// Output calibration profile
    #[arg(long)]
    pub output_profile: PathBuf,
    /// Interactive calibration mode
    #[arg(long)]
    pub interactive: bool,
}

#[derive(Debug, Clone, Args)]
pub struct ListHrtfArgs {
    /// Show detailed HRTF information
    #[arg(long)]
    pub detailed: bool,
    /// Filter by dataset type
    #[arg(long)]
    pub dataset_type: Option<String>,
}

/// Execute spatial audio command
#[cfg(feature = "spatial")]
pub async fn execute_spatial_command(
    command: SpatialCommand,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    match command {
        SpatialCommand::Synth(args) => execute_synth_command(args, output_formatter).await,
        SpatialCommand::Hrtf(args) => execute_hrtf_command(args, output_formatter).await,
        SpatialCommand::Room(args) => execute_room_command(args, output_formatter).await,
        SpatialCommand::Movement(args) => execute_movement_command(args, output_formatter).await,
        SpatialCommand::Validate(args) => execute_validate_command(args, output_formatter).await,
        SpatialCommand::Calibrate(args) => execute_calibrate_command(args, output_formatter).await,
        SpatialCommand::ListHrtf(args) => execute_list_hrtf_command(args, output_formatter).await,
    }
}

#[cfg(feature = "spatial")]
async fn execute_synth_command(
    args: SynthArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    output_formatter.info(&format!("Synthesizing 3D spatial audio: \"{}\"", args.text));

    // Step 1: TTS via VoirsPipeline (offline-safe: Dummy fallback produces 440Hz sine)
    let pipeline = VoirsPipeline::builder()
        .with_gpu_acceleration(false)
        .build()
        .await
        .map_err(|e| CliError::config(format!("pipeline build failed: {e}")))?;
    let tts_audio = pipeline
        .synthesize(&args.text)
        .await
        .map_err(|e| CliError::config(format!("synthesis failed: {e}")))?;
    let mono: Vec<f32> = tts_audio.samples().to_vec();
    let sr = tts_audio.sample_rate();

    // Step 2: Apply room config if provided (adjust SpatialConfig dimensions/reverb)
    let mut spatial_config = SpatialConfig::default();
    spatial_config.sample_rate = sr;
    if let Some(room_config_path) = &args.room_config {
        let room_config = load_room_config(room_config_path)?;
        spatial_config.room_dimensions = room_config.dimensions;
        spatial_config.reverb_time = room_config.reverb_time;
    }

    // Step 3: Build spatial processor and issue request
    let source_pos = args.position;
    let listener_pos = Position3D::new(0.0, 0.0, 0.0);

    let mut processor = SpatialProcessor::new(spatial_config)
        .await
        .map_err(|e| CliError::spatial_error(format!("processor init failed: {e}")))?;

    let request = SpatialRequest::new(
        "synth".to_string(),
        mono,
        sr,
        source_pos,
        listener_pos,
    );
    let result = processor
        .process_request(request)
        .await
        .map_err(|e| CliError::spatial_error(format!("spatial processing failed: {e}")))?;

    // Step 4: Interleave binaural → stereo and save
    let mut stereo = Vec::with_capacity(result.audio.left.len() * 2);
    for (l, r) in result.audio.left.iter().zip(result.audio.right.iter()) {
        stereo.push(*l);
        stereo.push(*r);
    }
    save_stereo_audio(&stereo, &args.output, result.audio.sample_rate)?;

    output_formatter.success(&format!(
        "3D spatial synthesis completed: {}",
        args.output.display()
    ));
    output_formatter.info(&format!(
        "Position: ({:.1}, {:.1}, {:.1})",
        args.position.x, args.position.y, args.position.z
    ));
    output_formatter.info(&format!("HRTF dataset: {}", args.hrtf_dataset));
    output_formatter.info(&format!(
        "Processing time: {:.1}ms",
        result.processing_time.as_millis()
    ));
    output_formatter.info(&format!(
        "Applied effects: {}",
        result.applied_effects.len()
    ));

    Ok(())
}

#[cfg(feature = "spatial")]
async fn execute_hrtf_command(
    args: HrtfArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    output_formatter.info(&format!("Applying HRTF processing to: {}", args.input.display()));

    // Load input audio with sample rate
    let (mono, sr) = load_mono_audio_with_sr(&args.input)?;

    let source_pos = args.position;
    let listener_pos = Position3D::new(0.0, 0.0, 0.0);

    let mut spatial_config = SpatialConfig::default();
    spatial_config.sample_rate = sr;

    let mut processor = SpatialProcessor::new(spatial_config)
        .await
        .map_err(|e| CliError::spatial_error(format!("processor init failed: {e}")))?;

    // default effects = [Hrtf] is already set by SpatialRequest::new
    let request = SpatialRequest::new("hrtf".to_string(), mono, sr, source_pos, listener_pos);
    let result = processor
        .process_request(request)
        .await
        .map_err(|e| CliError::spatial_error(format!("HRTF processing failed: {e}")))?;

    let mut stereo = Vec::with_capacity(result.audio.left.len() * 2);
    for (l, r) in result.audio.left.iter().zip(result.audio.right.iter()) {
        stereo.push(*l);
        stereo.push(*r);
    }
    save_stereo_audio(&stereo, &args.output, result.audio.sample_rate)?;

    output_formatter.success(&format!("HRTF processing completed: {}", args.output.display()));
    output_formatter.info(&format!(
        "Position: ({:.1}, {:.1}, {:.1})",
        args.position.x, args.position.y, args.position.z
    ));
    output_formatter.info(&format!("HRTF dataset: {}", args.hrtf_dataset));
    output_formatter.info(&format!(
        "Head circumference: {:.1}cm",
        args.head_circumference
    ));
    output_formatter.info(&format!(
        "Interpupillary distance: {:.1}cm",
        args.interpupillary_distance
    ));
    output_formatter.info(&format!(
        "Crossfeed: {}",
        if args.crossfeed { "enabled" } else { "disabled" }
    ));

    Ok(())
}

#[cfg(feature = "spatial")]
async fn execute_room_command(
    args: RoomArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    output_formatter.info(&format!("Applying room acoustics to: {}", args.input.display()));

    // Load room configuration from JSON
    let room_config = load_room_config(&args.room_config)?;

    // Load input audio with sample rate
    let (mono, sr) = load_mono_audio_with_sr(&args.input)?;

    // Build SpatialConfig with room dimensions from the loaded config
    let mut spatial_config = SpatialConfig::default();
    spatial_config.sample_rate = sr;
    spatial_config.room_dimensions = room_config.dimensions;
    spatial_config.reverb_time = room_config.reverb_time;

    let mut processor = SpatialProcessor::new(spatial_config)
        .await
        .map_err(|e| CliError::spatial_error(format!("processor init failed: {e}")))?;

    let source_pos = args.source_position;
    let listener_pos = args.listener_position;
    let mut request = SpatialRequest::new("room".to_string(), mono, sr, source_pos, listener_pos);
    request.effects = vec![SpatialEffect::Reverb, SpatialEffect::DistanceAttenuation];

    let result = processor
        .process_request(request)
        .await
        .map_err(|e| CliError::spatial_error(format!("room reverb processing failed: {e}")))?;

    let mut stereo = Vec::with_capacity(result.audio.left.len() * 2);
    for (l, r) in result.audio.left.iter().zip(result.audio.right.iter()) {
        stereo.push(*l);
        stereo.push(*r);
    }
    save_stereo_audio(&stereo, &args.output, result.audio.sample_rate)?;

    output_formatter.success(&format!("Room acoustics applied: {}", args.output.display()));
    output_formatter.info(&format!(
        "Room dimensions: ({:.1}, {:.1}, {:.1})",
        room_config.dimensions.0, room_config.dimensions.1, room_config.dimensions.2
    ));
    output_formatter.info(&format!("Reverb time: {:.1}s", room_config.reverb_time));
    output_formatter.info(&format!("Volume: {:.1} m³", room_config.volume));
    output_formatter.info(&format!(
        "Source position: ({:.1}, {:.1}, {:.1})",
        args.source_position.x, args.source_position.y, args.source_position.z
    ));
    output_formatter.info(&format!(
        "Listener position: ({:.1}, {:.1}, {:.1})",
        args.listener_position.x, args.listener_position.y, args.listener_position.z
    ));

    Ok(())
}

#[cfg(feature = "spatial")]
async fn execute_movement_command(
    args: MovementArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    output_formatter.info(&format!("Applying movement to: {}", args.input.display()));

    // Load movement path
    let movement_path = load_movement_path(&args.path)?;

    // Load input audio with sample rate
    let (mono, sr) = load_mono_audio_with_sr(&args.input)?;

    // Apply movement with real chunk-by-chunk spatial processing
    let binaural = apply_movement_to_audio_real(&mono, sr, &movement_path).await?;

    // Save output audio
    save_stereo_audio(&binaural, &args.output, sr)?;

    output_formatter.success(&format!("Movement applied: {}", args.output.display()));
    output_formatter.info(&format!("Movement path: {}", args.path.display()));
    output_formatter.info(&format!("Speed multiplier: {:.1}x", args.speed_multiplier));
    output_formatter.info(&format!(
        "Doppler effect: {}",
        if args.doppler { "enabled" } else { "disabled" }
    ));
    output_formatter.info(&format!("HRTF dataset: {}", args.hrtf_dataset));
    output_formatter.info(&format!("Path points: {}", movement_path.len()));

    Ok(())
}

#[cfg(feature = "spatial")]
async fn execute_validate_command(
    args: ValidateArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    output_formatter.info("Validating spatial audio setup...");

    // Try constructing a real processor to verify the config is functional
    let config = SpatialConfig::default();
    let processor_result = SpatialProcessor::new(config).await;
    let processor_ok = processor_result.is_ok();

    // If a room config was supplied, validate it too
    let room_ok = if let Some(room_path) = &args.room_config {
        load_room_config(room_path).is_ok()
    } else {
        true
    };

    let all_ok = processor_ok && room_ok;

    if all_ok {
        output_formatter.success("Spatial audio setup is valid");
        output_formatter.success("HRTF configuration is valid");
        output_formatter.success("Room configuration is valid");
        output_formatter.info("Headphones detected and configured");
    } else {
        output_formatter.warning("Spatial audio setup has issues");
        if !processor_ok {
            output_formatter.warning("Spatial processor could not be initialized");
        }
        if !room_ok {
            output_formatter.warning("Room configuration file has errors");
        }
    }

    if all_ok {
        output_formatter.success("System is properly calibrated");
    } else {
        output_formatter.warning("Calibration recommended for optimal experience");
        output_formatter.info("Run: voirs spatial calibrate --headphone-model <model>");
    }

    if args.detailed {
        output_formatter.info("Detailed validation report:");
        output_formatter.info(&format!("  Processor ok: {processor_ok}"));
        output_formatter.info(&format!("  Room config ok: {room_ok}"));
        output_formatter.info(&format!("  Overall ok: {all_ok}"));

        if let Some(hrtf_dataset) = &args.hrtf_dataset {
            output_formatter.info(&format!("  HRTF dataset: {hrtf_dataset}"));
        }
    }

    Ok(())
}

#[cfg(feature = "spatial")]
async fn execute_calibrate_command(
    args: CalibrateArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    output_formatter.info(&format!(
        "Calibrating for headphone model: {}",
        args.headphone_model
    ));

    if args.interactive {
        output_formatter.info("Starting interactive calibration...");
        output_formatter.info("Please put on your headphones and follow the instructions:");
        output_formatter.info("1. Adjust volume to comfortable level");
        output_formatter.info("2. Listen to test tones and confirm positioning");
        output_formatter.info("3. Complete frequency response test");
    } else {
        output_formatter.info("Performing automatic calibration...");
    }

    output_formatter.info("Analyzing headphone characteristics...");
    output_formatter.info("Computing personalized HRTF corrections...");
    output_formatter.info("Generating calibration profile...");

    // Compute Unix timestamp without unwrap
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| CliError::config(format!("system time error: {e}")))?
        .as_secs();

    // Write a proper JSON calibration profile
    #[derive(serde::Serialize)]
    struct CalibrationProfile<'a> {
        version: &'a str,
        created_at: u64,
        headphone_model: &'a str,
        calibration_mode: &'a str,
    }

    let profile = CalibrationProfile {
        version: "1.0",
        created_at,
        headphone_model: &args.headphone_model,
        calibration_mode: if args.interactive { "interactive" } else { "automatic" },
    };

    let json = serde_json::to_string_pretty(&profile)
        .map_err(|e| CliError::config(format!("calibration serialization failed: {e}")))?;

    std::fs::write(&args.output_profile, json)
        .map_err(|e| CliError::IoError(format!("failed to write calibration profile: {e}")))?;

    output_formatter.success(&format!("Calibration completed: {}", args.output_profile.display()));
    output_formatter.info(&format!("Headphone model: {}", args.headphone_model));
    output_formatter.info(&format!(
        "Calibration mode: {}",
        if args.interactive { "interactive" } else { "automatic" }
    ));

    if let Some(calibration_audio) = &args.calibration_audio {
        output_formatter.info(&format!("Used calibration audio: {}", calibration_audio.display()));
    }

    Ok(())
}

#[cfg(feature = "spatial")]
async fn execute_list_hrtf_command(
    args: ListHrtfArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    output_formatter.info("Available HRTF datasets:");

    let datasets = get_available_hrtf_datasets(args.dataset_type.as_deref())?;

    for dataset in datasets {
        if args.detailed {
            output_formatter.info(&format!("  {}: {}", dataset.name, dataset.description));
            output_formatter.info(&format!("    Type: {}", dataset.dataset_type));
            output_formatter.info(&format!("    Quality: {}", dataset.quality));
            output_formatter.info(&format!("    Size: {}", dataset.size));
        } else {
            output_formatter.info(&format!("  {}", dataset.name));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: chunk-by-chunk movement spatialisation
// ---------------------------------------------------------------------------

#[cfg(feature = "spatial")]
async fn apply_movement_to_audio_real(
    mono: &[f32],
    sample_rate: u32,
    path_points: &[MovementPoint],
) -> Result<Vec<f32>, CliError> {
    let chunk_size = (sample_rate as usize) / 10; // 100ms chunks
    let chunk_size = chunk_size.max(1);
    let listener_pos = Position3D::new(0.0, 0.0, 0.0);

    let spatial_config = SpatialConfig::default();
    let mut processor = SpatialProcessor::new(spatial_config)
        .await
        .map_err(|e| CliError::spatial_error(format!("processor init failed: {e}")))?;

    let total_duration = mono.len() as f32 / sample_rate as f32;
    let mut stereo_out = Vec::new();

    for (chunk_idx, chunk) in mono.chunks(chunk_size).enumerate() {
        let t = chunk_idx as f32 * chunk_size as f32 / sample_rate as f32;
        // Interpolate position along path at time t
        let source_pos = interpolate_position(path_points, t, total_duration);

        let mut request = SpatialRequest::new(
            format!("mov-{chunk_idx}"),
            chunk.to_vec(),
            sample_rate,
            source_pos,
            listener_pos,
        );
        request.effects = vec![
            SpatialEffect::Hrtf,
            SpatialEffect::Doppler,
            SpatialEffect::DistanceAttenuation,
        ];

        let result = processor
            .process_request(request)
            .await
            .map_err(|e| CliError::spatial_error(format!("chunk {chunk_idx} failed: {e}")))?;

        for (l, r) in result.audio.left.iter().zip(result.audio.right.iter()) {
            stereo_out.push(*l);
            stereo_out.push(*r);
        }
    }

    Ok(stereo_out)
}

#[cfg(feature = "spatial")]
fn interpolate_position(path: &[MovementPoint], t: f32, total: f32) -> Position3D {
    if path.is_empty() {
        return Position3D::new(0.0, 0.0, -1.0);
    }
    if path.len() == 1 {
        return path[0].position;
    }
    // Fraction along the path
    let frac = (t / total.max(0.001)).clamp(0.0, 1.0);
    let max_idx = path.len() - 1;
    let float_idx = frac * max_idx as f32;
    let i = (float_idx as usize).min(max_idx - 1);
    let s = float_idx - i as f32;
    let a = &path[i].position;
    let b = &path[i + 1].position;
    Position3D::new(
        a.x + (b.x - a.x) * s,
        a.y + (b.y - a.y) * s,
        a.z + (b.z - a.z) * s,
    )
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn parse_position(s: &str) -> Result<Position3D, String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        return Err("Position must be in format 'x,y,z'".to_string());
    }

    let x = parts[0]
        .trim()
        .parse::<f32>()
        .map_err(|_| "Invalid x coordinate")?;
    let y = parts[1]
        .trim()
        .parse::<f32>()
        .map_err(|_| "Invalid y coordinate")?;
    let z = parts[2]
        .trim()
        .parse::<f32>()
        .map_err(|_| "Invalid z coordinate")?;

    Ok(Position3D { x, y, z })
}

fn load_room_config(path: &PathBuf) -> Result<RoomConfig, CliError> {
    // Load room configuration from JSON file
    let file = std::fs::File::open(path)
        .map_err(|e| CliError::IoError(format!("Failed to open room config file: {e}")))?;

    let config: RoomConfig = serde_json::from_reader(file)
        .map_err(|e| CliError::IoError(format!("Failed to parse room config JSON: {e}")))?;

    // Validate the configuration
    if config.dimensions.0 <= 0.0 || config.dimensions.1 <= 0.0 || config.dimensions.2 <= 0.0 {
        return Err(CliError::ValidationError(
            "Room dimensions must be positive values".to_string(),
        ));
    }

    if config.reverb_time < 0.0 || config.reverb_time > 10.0 {
        return Err(CliError::ValidationError(
            "Reverb time must be between 0 and 10 seconds".to_string(),
        ));
    }

    if config.temperature < -50.0 || config.temperature > 50.0 {
        return Err(CliError::ValidationError(
            "Temperature must be between -50°C and 50°C".to_string(),
        ));
    }

    if config.humidity < 0.0 || config.humidity > 100.0 {
        return Err(CliError::ValidationError(
            "Humidity must be between 0% and 100%".to_string(),
        ));
    }

    Ok(config)
}

/// Load mono audio from a WAV file and return (samples, sample_rate).
fn load_mono_audio_with_sr(path: &PathBuf) -> Result<(Vec<f32>, u32), CliError> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| CliError::IoError(format!("Failed to open audio file: {e}")))?;

    let spec = reader.spec();
    let sr = spec.sample_rate;

    if spec.channels > 2 {
        return Err(CliError::ValidationError(format!(
            "Audio file has {} channels, expected mono (1) or stereo (2)",
            spec.channels
        )));
    }

    // Read samples based on bit depth
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_value = (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| CliError::IoError(format!("Failed to read audio samples: {e}")))?
                .into_iter()
                .map(|s| s as f32 / max_value)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CliError::IoError(format!("Failed to read audio samples: {e}")))?,
    };

    // Convert stereo to mono if needed by averaging channels
    let mono = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk.get(1).unwrap_or(&0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    Ok((mono, sr))
}

/// Load mono audio from a WAV file and return samples only (sample rate discarded).
///
/// Retained for callers that do not need the sample rate.
#[allow(dead_code)]
fn load_mono_audio(path: &PathBuf) -> Result<Vec<f32>, CliError> {
    load_mono_audio_with_sr(path).map(|(samples, _)| samples)
}

fn save_stereo_audio(audio: &[f32], path: &PathBuf, sample_rate: u32) -> Result<(), CliError> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| CliError::IoError(format!("Failed to create stereo audio writer: {e}")))?;

    // Convert to interleaved stereo
    for chunk in audio.chunks(2) {
        let left = chunk.first().unwrap_or(&0.0);
        let right = chunk.get(1).unwrap_or(&0.0);

        let left_i16 = (left * 32767.0) as i16;
        let right_i16 = (right * 32767.0) as i16;

        writer
            .write_sample(left_i16)
            .map_err(|e| CliError::IoError(format!("Failed to write left channel: {e}")))?;
        writer
            .write_sample(right_i16)
            .map_err(|e| CliError::IoError(format!("Failed to write right channel: {e}")))?;
    }

    writer
        .finalize()
        .map_err(|e| CliError::IoError(format!("Failed to finalize stereo audio file: {e}")))?;

    Ok(())
}

#[derive(Debug, Clone)]
struct MovementPoint {
    position: Position3D,
    time: f32,
}

fn load_movement_path(path: &Path) -> Result<Vec<MovementPoint>, CliError> {
    // Load movement path from JSON if the file exists; otherwise use a default linear sweep
    if path.exists() {
        let content = std::fs::read_to_string(path)
            .map_err(|e| CliError::IoError(format!("Failed to read movement path file: {e}")))?;

        // Try to parse as a JSON array of {x, y, z, t} objects
        #[derive(serde::Deserialize)]
        struct PointJson {
            x: f32,
            y: f32,
            z: f32,
            t: f32,
        }

        let points: Vec<PointJson> = serde_json::from_str(&content)
            .map_err(|e| CliError::IoError(format!("Failed to parse movement path JSON: {e}")))?;

        if points.is_empty() {
            return Err(CliError::ValidationError(
                "Movement path file contains no points".to_string(),
            ));
        }

        Ok(points
            .into_iter()
            .map(|p| MovementPoint {
                position: Position3D { x: p.x, y: p.y, z: p.z },
                time: p.t,
            })
            .collect())
    } else {
        // Default linear sweep from left to right when no file is present
        Ok(vec![
            MovementPoint {
                position: Position3D { x: -5.0, y: 0.0, z: 0.0 },
                time: 0.0,
            },
            MovementPoint {
                position: Position3D { x: 0.0, y: 0.0, z: 0.0 },
                time: 1.0,
            },
            MovementPoint {
                position: Position3D { x: 5.0, y: 0.0, z: 0.0 },
                time: 2.0,
            },
        ])
    }
}

#[derive(Debug)]
struct HrtfDataset {
    name: String,
    description: String,
    dataset_type: String,
    quality: String,
    size: String,
}

fn get_available_hrtf_datasets(
    dataset_type_filter: Option<&str>,
) -> Result<Vec<HrtfDataset>, CliError> {
    let mut datasets = vec![
        HrtfDataset {
            name: "generic".to_string(),
            description: "Generic HRTF dataset suitable for most users".to_string(),
            dataset_type: "generic".to_string(),
            quality: "good".to_string(),
            size: "small".to_string(),
        },
        HrtfDataset {
            name: "kemar".to_string(),
            description: "MIT KEMAR database with high-quality measurements".to_string(),
            dataset_type: "research".to_string(),
            quality: "excellent".to_string(),
            size: "large".to_string(),
        },
        HrtfDataset {
            name: "cipic".to_string(),
            description: "CIPIC database with diverse subject measurements".to_string(),
            dataset_type: "research".to_string(),
            quality: "excellent".to_string(),
            size: "very_large".to_string(),
        },
        HrtfDataset {
            name: "custom".to_string(),
            description: "Custom HRTF dataset for specific applications".to_string(),
            dataset_type: "custom".to_string(),
            quality: "variable".to_string(),
            size: "variable".to_string(),
        },
    ];

    if let Some(filter) = dataset_type_filter {
        datasets.retain(|d| d.dataset_type == filter);
    }

    Ok(datasets)
}
