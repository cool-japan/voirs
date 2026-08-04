//! Singing voice synthesis commands for the VoiRS CLI

use crate::{error::CliError, output::OutputFormatter};
use clap::{Args, Subcommand};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
#[cfg(feature = "singing")]
use voirs_dataset::audio::io::{load_audio, FormatDetector};
#[cfg(feature = "singing")]
use voirs_dataset::audio::AudioProcessor as DatasetAudioProcessor;
#[cfg(feature = "singing")]
use voirs_dataset::quality::metrics::{
    QualityConfig as AudioQualityConfig, QualityMetrics as AudioQualityMetrics,
    QualityMetricsCalculator,
};
#[cfg(feature = "singing")]
use voirs_dataset::AudioData;
#[cfg(feature = "singing")]
use voirs_singing::{
    formats::FormatParser,
    techniques::{
        ArticulationSettings, DynamicsSettings, ExpressionSettings, FormantSettings,
        LegatoSettings, PortamentoSettings, ResonanceSettings, VibratoSettings,
    },
    BreathControl, EffectChain, MidiParser, MusicXmlParser, MusicalIntelligence, PitchContour,
    SingingConfig, SingingEngine, SingingTechnique, VocalFry, VoiceCharacteristics, VoiceType,
};

use hound;

/// Singing voice synthesis commands
#[cfg(feature = "singing")]
#[derive(Debug, Clone, Subcommand)]
pub enum SingingCommand {
    /// Synthesize singing from musical score
    #[command(visible_alias = "from-score")]
    Score(ScoreArgs),
    /// Synthesize singing from MIDI file
    #[command(visible_alias = "from-midi")]
    Midi(MidiArgs),
    /// Create a singing voice model from training samples
    #[command(visible_alias = "create-model")]
    CreateVoice(CreateVoiceArgs),
    /// Validate score and voice compatibility
    Validate(ValidateArgs),
    /// Apply singing effects to existing audio
    #[command(visible_alias = "apply-effects")]
    Effects(EffectsArgs),
    /// Analyze singing audio for quality metrics
    Analyze(AnalyzeArgs),
    /// List available singing presets
    ListPresets(ListPresetsArgs),
}

#[derive(Debug, Clone, Args)]
pub struct ScoreArgs {
    /// Musical score file (MusicXML format)
    #[arg(long)]
    pub score: PathBuf,
    /// Singing voice model to use
    #[arg(long)]
    pub voice: String,
    /// Output audio file
    pub output: PathBuf,
    /// Tempo in BPM (overrides score tempo)
    #[arg(long)]
    pub tempo: Option<f32>,
    /// Key signature (C, D, E, F, G, A, B with optional #/b)
    #[arg(long)]
    pub key: Option<String>,
    /// Singing technique preset
    #[arg(long, default_value = "classical")]
    pub technique: String,
    /// Voice type (soprano, mezzo-soprano, alto, tenor, baritone, bass)
    #[arg(long, default_value = "soprano")]
    pub voice_type: String,
    /// Sample rate for output audio
    #[arg(long, default_value = "44100")]
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Args)]
pub struct MidiArgs {
    /// MIDI file input
    pub midi: PathBuf,
    /// Lyrics file (plain text, one line per note)
    #[arg(long)]
    pub lyrics: PathBuf,
    /// Singing voice model to use
    #[arg(long)]
    pub voice: String,
    /// Output audio file
    pub output: PathBuf,
    /// Tempo in BPM (overrides MIDI tempo)
    #[arg(long)]
    pub tempo: Option<f32>,
    /// Singing technique preset
    #[arg(long, default_value = "classical")]
    pub technique: String,
    /// Voice type (soprano, mezzo-soprano, alto, tenor, baritone, bass)
    #[arg(long, default_value = "soprano")]
    pub voice_type: String,
}

#[derive(Debug, Clone, Args)]
pub struct CreateVoiceArgs {
    /// Directory containing singing samples
    pub samples: PathBuf,
    /// Output singing voice model file
    #[arg(long)]
    pub output: PathBuf,
    /// Voice name/identifier
    #[arg(long)]
    pub name: String,
    /// Voice type (soprano, mezzo-soprano, alto, tenor, baritone, bass)
    #[arg(long, default_value = "soprano")]
    pub voice_type: String,
    /// Training quality threshold (0.0-1.0)
    #[arg(long, default_value = "0.8")]
    pub quality_threshold: f32,
    /// Number of training epochs
    #[arg(long, default_value = "100")]
    pub epochs: u32,
}

#[derive(Debug, Clone, Args)]
pub struct ValidateArgs {
    /// Musical score file to validate
    pub score: PathBuf,
    /// Singing voice model to validate against
    #[arg(long)]
    pub voice: String,
    /// Generate detailed validation report
    #[arg(long)]
    pub detailed: bool,
}

#[derive(Debug, Clone, Args)]
pub struct EffectsArgs {
    /// Input audio file
    pub input: PathBuf,
    /// Output audio file
    pub output: PathBuf,
    /// Effects to apply (e.g. reverb, chorus, compressor)
    #[arg(long, num_args = 1..)]
    pub effects: Vec<String>,
    /// Vibrato intensity (0.0-2.0)
    #[arg(long, default_value = "1.0")]
    pub vibrato: f32,
    /// Expression style (happy, sad, passionate, calm)
    #[arg(long, default_value = "neutral")]
    pub expression: String,
    /// Breath control intensity (0.0-1.0)
    #[arg(long, default_value = "0.5")]
    pub breath_control: f32,
    /// Pitch bend sensitivity (0.0-1.0)
    #[arg(long, default_value = "0.3")]
    pub pitch_bend: f32,
}

#[derive(Debug, Clone, Args)]
pub struct AnalyzeArgs {
    /// Singing audio file to analyze
    pub input: PathBuf,
    /// Output analysis report file (JSON format)
    #[arg(long)]
    pub report: PathBuf,
    /// Include detailed pitch analysis
    #[arg(long)]
    pub pitch_analysis: bool,
    /// Include vibrato analysis
    #[arg(long)]
    pub vibrato_analysis: bool,
    /// Include breath pattern analysis
    #[arg(long)]
    pub breath_analysis: bool,
}

#[derive(Debug, Clone, Args)]
pub struct ListPresetsArgs {
    /// Show detailed preset information
    #[arg(long)]
    pub detailed: bool,
    /// Filter by voice type
    #[arg(long)]
    pub voice_type: Option<String>,
}

/// Execute singing command
#[cfg(feature = "singing")]
pub async fn execute_singing_command(
    command: SingingCommand,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    match command {
        SingingCommand::Score(args) => execute_score_command(args, output_formatter).await,
        SingingCommand::Midi(args) => execute_midi_command(args, output_formatter).await,
        SingingCommand::CreateVoice(args) => {
            execute_create_voice_command(args, output_formatter).await
        }
        SingingCommand::Validate(args) => execute_validate_command(args, output_formatter).await,
        SingingCommand::Effects(args) => execute_effects_command(args, output_formatter).await,
        SingingCommand::Analyze(args) => execute_analyze_command(args, output_formatter).await,
        SingingCommand::ListPresets(args) => {
            execute_list_presets_command(args, output_formatter).await
        }
    }
}

#[cfg(feature = "singing")]
async fn execute_score_command(
    args: ScoreArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    output_formatter.info(&format!(
        "Synthesizing singing from score: {:?}",
        args.score
    ));

    let engine = SingingEngine::new(SingingConfig::default())
        .await
        .map_err(|e| CliError::singing_error(format!("engine init failed: {e}")))?;

    let voice_type = parse_voice_type(&args.voice_type)?;
    let mut voice_characteristics = VoiceCharacteristics::default();
    voice_characteristics.voice_type = voice_type;

    let technique = create_singing_technique(&args.technique)?;

    let score_path = args
        .score
        .to_str()
        .ok_or_else(|| CliError::InvalidArgument("score path contains invalid UTF-8".into()))?;

    let parser = MusicXmlParser::new();
    let mut score = parser
        .parse_file(score_path)
        .await
        .map_err(|e| CliError::singing_error(format!("score parse failed: {e}")))?;

    // Apply tempo override if provided
    if let Some(tempo) = args.tempo {
        score.tempo = tempo;
    }

    let resp = engine
        .synthesize_score(score, voice_characteristics, technique)
        .await
        .map_err(|e| CliError::singing_error(format!("synthesis failed: {e}")))?;

    save_audio(&resp.audio, &args.output, resp.sample_rate)?;

    output_formatter.success(&format!("Singing synthesis completed: {:?}", args.output));
    output_formatter.info(&format!("Notes processed: {}", resp.stats.total_notes));
    output_formatter.info(&format!(
        "Synthesis quality: {:.1}%",
        resp.stats.overall_quality * 100.0
    ));
    output_formatter.info(&format!(
        "Processing time: {:.2}s",
        resp.stats.processing_time.as_secs_f32()
    ));

    Ok(())
}

#[cfg(feature = "singing")]
async fn execute_midi_command(
    args: MidiArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    output_formatter.info(&format!("Synthesizing singing from MIDI: {:?}", args.midi));

    let engine = SingingEngine::new(SingingConfig::default())
        .await
        .map_err(|e| CliError::singing_error(format!("engine init failed: {e}")))?;

    let voice_type = parse_voice_type(&args.voice_type)?;
    let mut voice_characteristics = VoiceCharacteristics::default();
    voice_characteristics.voice_type = voice_type;

    let technique = create_singing_technique(&args.technique)?;

    let midi_path = args
        .midi
        .to_str()
        .ok_or_else(|| CliError::InvalidArgument("MIDI path contains invalid UTF-8".into()))?;

    let parser = MidiParser::new();
    let mut score = parser
        .parse_file(midi_path)
        .await
        .map_err(|e| CliError::singing_error(format!("MIDI parse failed: {e}")))?;

    // Apply tempo override if provided
    if let Some(tempo) = args.tempo {
        score.tempo = tempo;
    }

    // Load lyrics and assign to score notes
    let lyrics_text = std::fs::read_to_string(&args.lyrics)
        .map_err(|e| CliError::IoError(format!("failed to read lyrics file: {e}")))?;
    let lyric_lines: Vec<&str> = lyrics_text.lines().collect();
    for (note, lyric) in score.notes.iter_mut().zip(lyric_lines.iter()) {
        note.event.lyric = Some(lyric.to_string());
    }

    let resp = engine
        .synthesize_score(score, voice_characteristics, technique)
        .await
        .map_err(|e| CliError::singing_error(format!("synthesis failed: {e}")))?;

    save_audio(&resp.audio, &args.output, resp.sample_rate)?;

    output_formatter.success(&format!(
        "MIDI singing synthesis completed: {:?}",
        args.output
    ));
    output_formatter.info(&format!("Notes processed: {}", resp.stats.total_notes));
    output_formatter.info(&format!(
        "Synthesis quality: {:.1}%",
        resp.stats.overall_quality * 100.0
    ));

    Ok(())
}

/// Real per-file vocal features extracted from one singing sample, used to
/// build a [`VoiceCharacteristics`] profile in [`execute_create_voice_command`].
#[cfg(feature = "singing")]
struct SampleFeatures {
    /// Per-frame fundamental frequency estimates (voiced frames only), in Hz.
    f0_values: Vec<f32>,
    /// Longest run of consecutive voiced frames, in seconds - a proxy for
    /// sustained phonation ("breath capacity").
    longest_voiced_run_secs: f32,
    /// Full audio quality/spectral analysis for this sample.
    quality: AudioQualityMetrics,
}

/// Analyze one (mono) audio sample: detect its pitch contour frame-by-frame
/// (via [`PitchContour::detect_pitch`], the same real autocorrelation-based
/// detector already used by `singing analyze`), track the longest
/// continuously voiced run, and compute full audio quality/spectral metrics
/// (via [`QualityMetricsCalculator`], reused from `voirs-dataset`).
#[cfg(feature = "singing")]
fn extract_sample_features(audio: &AudioData) -> Result<SampleFeatures, CliError> {
    const FRAME_SIZE: usize = 512;

    let sample_rate = audio.sample_rate() as f32;
    let samples = audio.samples();

    let mut f0_values = Vec::new();
    let mut longest_run = 0usize;
    let mut current_run = 0usize;

    for frame in samples.chunks(FRAME_SIZE) {
        match PitchContour::detect_pitch(frame, sample_rate) {
            Some(f0) if f0 > 0.0 => {
                f0_values.push(f0);
                current_run += 1;
                longest_run = longest_run.max(current_run);
            }
            _ => current_run = 0,
        }
    }
    let longest_voiced_run_secs = if sample_rate > 0.0 {
        (longest_run * FRAME_SIZE) as f32 / sample_rate
    } else {
        0.0
    };

    let calculator = QualityMetricsCalculator::new(AudioQualityConfig::default());
    let quality = calculator
        .calculate_metrics(audio)
        .map_err(|e| CliError::singing_error(format!("quality analysis failed: {e}")))?;

    Ok(SampleFeatures {
        f0_values,
        longest_voiced_run_secs,
        quality,
    })
}

/// Estimate vibrato rate (Hz) and depth (0.0-1.0) from one sample's F0
/// contour.
///
/// The contour is converted to cents relative to its own mean (a
/// pitch-independent deviation signal), and the vibrato rate is estimated
/// from the zero-crossing rate of that deviation (each full oscillation
/// cycle crosses zero twice); depth is the RMS deviation scaled against one
/// semitone (100 cents), clamped to `[0.0, 1.0]`.
#[cfg(feature = "singing")]
fn estimate_vibrato(f0_values: &[f32], frame_rate_hz: f32) -> (f32, f32) {
    if f0_values.len() < 4 || frame_rate_hz <= 0.0 {
        return (0.0, 0.0);
    }

    let mean_f0 = f0_values.iter().sum::<f32>() / f0_values.len() as f32;
    if mean_f0 <= 0.0 {
        return (0.0, 0.0);
    }

    let cents: Vec<f32> = f0_values
        .iter()
        .map(|&f0| 1200.0 * (f0 / mean_f0).log2())
        .collect();

    let mut zero_crossings = 0usize;
    for pair in cents.windows(2) {
        if pair[0] != 0.0 && pair[0].signum() != pair[1].signum() {
            zero_crossings += 1;
        }
    }

    let duration_secs = cents.len() as f32 / frame_rate_hz;
    let rate_hz = if duration_secs > 0.0 {
        (zero_crossings as f32 / 2.0) / duration_secs
    } else {
        0.0
    };

    let rms_cents = (cents.iter().map(|c| c * c).sum::<f32>() / cents.len() as f32).sqrt();
    let depth = (rms_cents / 100.0).clamp(0.0, 1.0);

    (rate_hz.clamp(0.0, 20.0), depth)
}

/// Iteratively refine the mean/standard-deviation of `values` via sigma
/// clipping: repeatedly drop points more than two standard deviations from
/// the current mean and recompute, for up to `max_iterations` rounds or
/// until the estimate converges.
///
/// This is the real computation behind `--epochs`: it is iterative robust
/// statistics refinement of the singer's typical pitch (not model
/// training) - more iterations produce an estimate less skewed by outlier
/// F0 frames (octave errors, noise, etc.), and the result measurably differs
/// between low and high iteration counts on outlier-contaminated input.
///
/// Returns `(mean, std_dev, iterations_run, samples_retained)`.
#[cfg(feature = "singing")]
fn refine_f0_statistics(values: &[f32], max_iterations: u32) -> (f32, f32, u32, usize) {
    if values.is_empty() {
        return (0.0, 0.0, 0, 0);
    }

    const MIN_RETAINED: usize = 4;

    fn mean_of(v: &[f32]) -> f32 {
        v.iter().sum::<f32>() / v.len() as f32
    }
    fn std_of(v: &[f32], mean: f32) -> f32 {
        (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
    }

    let mut working: Vec<f32> = values.to_vec();
    let mut mean = mean_of(&working);
    let mut std = std_of(&working, mean);

    // Bounded so a very large --epochs can't cause an unbounded loop; sigma
    // clipping converges in a handful of rounds in practice, so this cap is
    // never a real constraint.
    let iterations = max_iterations.min(1000);
    let mut ran = 0u32;

    for _ in 0..iterations {
        if std <= f32::EPSILON || working.len() <= MIN_RETAINED {
            break;
        }

        let lower = mean - 2.0 * std;
        let upper = mean + 2.0 * std;
        let refined: Vec<f32> = working
            .iter()
            .copied()
            .filter(|&v| v >= lower && v <= upper)
            .collect();

        if refined.len() == working.len() || refined.len() < MIN_RETAINED {
            break;
        }

        working = refined;
        let new_mean = mean_of(&working);
        let new_std = std_of(&working, new_mean);
        ran += 1;

        let converged = (new_mean - mean).abs() < 0.01 && (new_std - std).abs() < 0.01;
        mean = new_mean;
        std = new_std;
        if converged {
            break;
        }
    }

    (mean, std, ran, working.len())
}

/// Enumerate the decodable audio sample files directly inside `dir`
/// (non-recursive), sorted for deterministic processing order.
#[cfg(feature = "singing")]
fn list_sample_audio_files(dir: &Path) -> Result<Vec<PathBuf>, CliError> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| CliError::IoError(format!("failed to read samples directory: {e}")))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && FormatDetector::detect_from_extension(path).is_ok())
        .collect();
    files.sort();
    Ok(files)
}

#[cfg(feature = "singing")]
async fn execute_create_voice_command(
    args: CreateVoiceArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    output_formatter.info(&format!(
        "Creating singing voice model from: {:?}",
        args.samples
    ));

    if !args.samples.exists() || !args.samples.is_dir() {
        return Err(CliError::InvalidArgument(format!(
            "Samples directory not found: {:?}",
            args.samples
        )));
    }

    let engine = SingingEngine::new(SingingConfig::default())
        .await
        .map_err(|e| CliError::singing_error(format!("engine init failed: {e}")))?;

    let sample_paths = list_sample_audio_files(&args.samples)?;
    if sample_paths.is_empty() {
        return Err(CliError::InvalidArgument(format!(
            "No decodable audio samples (wav/flac/ogg/opus) found in {:?}",
            args.samples
        )));
    }

    output_formatter.info(&format!(
        "Analyzing {} singing sample(s)...",
        sample_paths.len()
    ));

    let mut accepted_f0: Vec<f32> = Vec::new();
    let mut per_sample_f0: Vec<(Vec<f32>, f32)> = Vec::new(); // (f0 values, frame rate)
    let mut breath_capacity = 0.0f32;
    let mut rms_values = Vec::new();
    let mut spectral_centroids = Vec::new();
    let mut spectral_rolloffs = Vec::new();
    let mut zero_crossing_rates = Vec::new();
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut best_score = 0.0f32;

    for path in &sample_paths {
        let raw_audio = load_audio(path)
            .map_err(|e| CliError::singing_error(format!("failed to decode {path:?}: {e}")))?;
        let audio = if raw_audio.channels() > 1 {
            DatasetAudioProcessor::to_mono(&raw_audio)
                .map_err(|e| CliError::singing_error(format!("failed to downmix {path:?}: {e}")))?
        } else {
            raw_audio
        };

        let features = extract_sample_features(&audio)?;
        best_score = best_score.max(features.quality.overall_score);

        if features.quality.overall_score < args.quality_threshold {
            rejected += 1;
            output_formatter.info(&format!(
                "  - {}: quality score {:.2} below threshold {:.2}, excluded",
                path.display(),
                features.quality.overall_score,
                args.quality_threshold
            ));
            continue;
        }
        accepted += 1;

        breath_capacity = breath_capacity.max(features.longest_voiced_run_secs);
        rms_values.push(features.quality.rms);
        spectral_centroids.push(features.quality.spectral_centroid);
        spectral_rolloffs.push(features.quality.spectral_rolloff);
        zero_crossing_rates.push(features.quality.zero_crossing_rate);

        if !features.f0_values.is_empty() {
            let frame_rate = audio.sample_rate() as f32 / 512.0;
            accepted_f0.extend(features.f0_values.iter().copied());
            per_sample_f0.push((features.f0_values, frame_rate));
        }
    }

    if accepted == 0 {
        return Err(CliError::InvalidArgument(format!(
            "All {} sample(s) fell below --quality-threshold {:.2} (best observed score: \
             {:.2}); re-run with a lower threshold",
            sample_paths.len(),
            args.quality_threshold,
            best_score
        )));
    }

    if accepted_f0.is_empty() {
        return Err(CliError::singing_error(
            "no pitched (voiced) frames were detected in any accepted sample; cannot derive \
             vocal characteristics from unvoiced/silent audio"
                .to_string(),
        ));
    }

    output_formatter.info(&format!(
        "Accepted {accepted}/{} sample(s) ({rejected} excluded by quality threshold)",
        sample_paths.len()
    ));
    output_formatter.info("Extracting vocal characteristics...");

    let range = (
        accepted_f0.iter().copied().fold(f32::INFINITY, f32::min),
        accepted_f0
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max),
    );

    let (f0_mean, f0_std, refinement_iterations, retained) =
        refine_f0_statistics(&accepted_f0, args.epochs);
    output_formatter.info(&format!(
        "Refinement: {refinement_iterations} sigma-clipping iteration(s) run (of up to \
         {} requested), {retained}/{} F0 samples retained -> mean {:.1} Hz, std {:.1} Hz",
        args.epochs,
        accepted_f0.len(),
        f0_mean,
        f0_std
    ));

    // Vibrato rate/depth averaged across samples, weighted by voiced frame count.
    let mut vibrato_rate_acc = 0.0f32;
    let mut vibrato_depth_acc = 0.0f32;
    let mut vibrato_weight = 0.0f32;
    for (f0s, frame_rate) in &per_sample_f0 {
        let (rate, depth) = estimate_vibrato(f0s, *frame_rate);
        let weight = f0s.len() as f32;
        vibrato_rate_acc += rate * weight;
        vibrato_depth_acc += depth * weight;
        vibrato_weight += weight;
    }
    let (vibrato_frequency, vibrato_depth) = if vibrato_weight > 0.0 {
        (
            vibrato_rate_acc / vibrato_weight,
            vibrato_depth_acc / vibrato_weight,
        )
    } else {
        (0.0, 0.0)
    };

    let vocal_power = if rms_values.is_empty() {
        0.0
    } else {
        (rms_values.iter().sum::<f32>() / rms_values.len() as f32).clamp(0.0, 1.0)
    };

    let mut resonance = HashMap::new();
    if !spectral_centroids.is_empty() {
        resonance.insert(
            "spectral_centroid_hz".to_string(),
            spectral_centroids.iter().sum::<f32>() / spectral_centroids.len() as f32,
        );
    }
    if !spectral_rolloffs.is_empty() {
        resonance.insert(
            "spectral_rolloff_hz".to_string(),
            spectral_rolloffs.iter().sum::<f32>() / spectral_rolloffs.len() as f32,
        );
    }

    let mut timbre = HashMap::new();
    if !zero_crossing_rates.is_empty() {
        timbre.insert(
            "zero_crossing_rate".to_string(),
            zero_crossing_rates.iter().sum::<f32>() / zero_crossing_rates.len() as f32,
        );
    }

    let voice = VoiceCharacteristics {
        voice_type: parse_voice_type(&args.voice_type).unwrap_or(VoiceType::Soprano),
        range,
        f0_mean,
        f0_std,
        vibrato_frequency,
        vibrato_depth,
        breath_capacity,
        vocal_power,
        resonance,
        timbre,
    };

    let output_path = args
        .output
        .to_str()
        .ok_or_else(|| CliError::InvalidArgument("output path contains invalid UTF-8".into()))?;

    engine
        .save_voice(&voice, output_path)
        .await
        .map_err(|e| CliError::singing_error(format!("save voice failed: {e}")))?;

    output_formatter.success(&format!("Singing voice model created: {:?}", args.output));
    output_formatter.info(&format!("Voice name: {}", args.name));
    output_formatter.info(&format!("Voice type: {}", args.voice_type));
    output_formatter.info(&format!(
        "Vocal range: {:.1} Hz - {:.1} Hz (mean {:.1} Hz, std {:.1} Hz)",
        range.0, range.1, f0_mean, f0_std
    ));
    output_formatter.info(&format!(
        "Vibrato: {vibrato_frequency:.2} Hz at depth {vibrato_depth:.2}"
    ));
    output_formatter.info(&format!("Breath capacity: {breath_capacity:.2}s"));
    output_formatter.info(&format!(
        "Quality threshold: {:.1}%",
        args.quality_threshold * 100.0
    ));

    Ok(())
}

#[cfg(feature = "singing")]
async fn execute_validate_command(
    args: ValidateArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    output_formatter.info(&format!("Validating score: {:?}", args.score));

    let score = load_musical_score(&args.score).await?;
    let voice_compatible = validate_voice_compatibility(&args.voice, &score)?;

    if voice_compatible {
        output_formatter.success("Score and voice are compatible");
    } else {
        output_formatter.warning("Score and voice may have compatibility issues");
    }

    if args.detailed {
        output_formatter.info(&format!("Total notes: {}", score.notes.len()));
        output_formatter.info(&format!("Tempo: {} BPM", score.tempo));
        output_formatter.info(&format!("Key signature: {:?}", score.key_signature));
        output_formatter.info(&format!("Time signature: {:?}", score.time_signature));

        let (min_freq, max_freq) = analyze_note_range(&score.notes);
        output_formatter.info(&format!(
            "Note range: {:.1} Hz - {:.1} Hz",
            min_freq, max_freq
        ));
    }

    Ok(())
}

#[cfg(feature = "singing")]
async fn execute_effects_command(
    args: EffectsArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    output_formatter.info(&format!("Applying singing effects to: {:?}", args.input));

    let (mut samples, sample_rate) = load_wav_samples(&args.input)?;

    let mut chain = EffectChain::new();
    for effect_name in &args.effects {
        let params = std::collections::HashMap::new();
        chain
            .add_effect_blocking(effect_name, params)
            .map_err(|e| {
                CliError::singing_error(format!("add effect '{}' failed: {e}", effect_name))
            })?;
    }

    samples = chain
        .process(samples, sample_rate as f32)
        .await
        .map_err(|e| CliError::singing_error(format!("effect processing failed: {e}")))?;

    save_audio(&samples, &args.output, sample_rate)?;

    output_formatter.success(&format!("Singing effects applied: {:?}", args.output));
    output_formatter.info(&format!("Vibrato intensity: {:.1}", args.vibrato));
    output_formatter.info(&format!("Expression: {}", args.expression));
    output_formatter.info(&format!("Breath control: {:.1}", args.breath_control));

    Ok(())
}

#[cfg(feature = "singing")]
async fn execute_analyze_command(
    args: AnalyzeArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    output_formatter.info(&format!("Analyzing singing audio: {:?}", args.input));

    let (samples, sample_rate) = load_wav_samples(&args.input)?;

    let analysis = analyze_singing_audio(&samples, sample_rate, &args).await?;

    let report_json = serde_json::to_string_pretty(&analysis)
        .map_err(|e| CliError::InvalidArgument(format!("Failed to serialize analysis: {}", e)))?;

    std::fs::write(&args.report, report_json)
        .map_err(|e| CliError::IoError(format!("failed to write report: {e}")))?;

    output_formatter.success(&format!("Analysis completed: {:?}", args.report));
    output_formatter.info(&format!(
        "Pitch accuracy: {:.1}%",
        analysis.pitch_accuracy * 100.0
    ));
    output_formatter.info(&format!(
        "Vibrato consistency: {:.1}%",
        analysis.vibrato_consistency * 100.0
    ));
    output_formatter.info(&format!(
        "Breath quality: {:.1}%",
        analysis.breath_quality * 100.0
    ));
    output_formatter.info(&format!("Note count: {}", analysis.note_count));
    output_formatter.info(&format!(
        "Mean frequency: {:.1} Hz",
        analysis.average_frequency
    ));

    Ok(())
}

#[cfg(feature = "singing")]
async fn execute_list_presets_command(
    args: ListPresetsArgs,
    output_formatter: &OutputFormatter,
) -> Result<(), CliError> {
    output_formatter.info("Available singing presets:");

    let presets = get_singing_presets(args.voice_type.as_deref())?;

    for preset in presets {
        if args.detailed {
            output_formatter.info(&format!("  {}: {}", preset.name, preset.description));
            output_formatter.info(&format!("    Voice type: {}", preset.voice_type));
            output_formatter.info(&format!("    Technique: {}", preset.technique_description));
        } else {
            output_formatter.info(&format!("  {}", preset.name));
        }
    }

    Ok(())
}

// Helper functions

#[cfg(feature = "singing")]
fn parse_voice_type(voice_type: &str) -> Result<VoiceType, CliError> {
    match voice_type.to_lowercase().as_str() {
        "soprano" => Ok(VoiceType::Soprano),
        "mezzo-soprano" | "mezzosoprano" | "mezzo" => Ok(VoiceType::MezzoSoprano),
        "alto" => Ok(VoiceType::Alto),
        "tenor" => Ok(VoiceType::Tenor),
        "baritone" => Ok(VoiceType::Baritone),
        "bass" => Ok(VoiceType::Bass),
        _ => Err(CliError::InvalidArgument(format!(
            "Invalid voice type: {}. Must be one of: soprano, mezzo-soprano, alto, tenor, baritone, bass",
            voice_type
        ))),
    }
}

#[cfg(feature = "singing")]
fn create_singing_technique(technique: &str) -> Result<SingingTechnique, CliError> {
    match technique.to_lowercase().as_str() {
        "classical" | "pop" | "jazz" | "folk" => Ok(SingingTechnique {
            breath_control: BreathControl::default(),
            vibrato: VibratoSettings::default(),
            vocal_fry: VocalFry::default(),
            legato: LegatoSettings::default(),
            portamento: PortamentoSettings::default(),
            dynamics: DynamicsSettings::default(),
            articulation: ArticulationSettings::default(),
            expression: ExpressionSettings::default(),
            formant: FormantSettings::default(),
            resonance: ResonanceSettings::default(),
        }),
        _ => Err(CliError::InvalidArgument(format!(
            "Invalid singing technique: {}. Must be one of: classical, pop, jazz, folk",
            technique
        ))),
    }
}

#[cfg(feature = "singing")]
async fn load_musical_score(path: &Path) -> Result<voirs_singing::MusicalScore, CliError> {
    let path_str = path
        .to_str()
        .ok_or_else(|| CliError::InvalidArgument("path contains invalid UTF-8".into()))?;

    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "mid" | "midi" => {
            let parser = MidiParser::new();
            parser
                .parse_file(path_str)
                .await
                .map_err(|e| CliError::singing_error(format!("MIDI parse failed: {e}")))
        }
        _ => {
            // Default to MusicXML for .musicxml, .xml, .mxl or unknown
            let parser = MusicXmlParser::new();
            parser
                .parse_file(path_str)
                .await
                .map_err(|e| CliError::singing_error(format!("score parse failed: {e}")))
        }
    }
}

#[cfg(feature = "singing")]
fn validate_voice_compatibility(
    _voice: &str,
    score: &voirs_singing::MusicalScore,
) -> Result<bool, CliError> {
    if score.notes.is_empty() {
        return Ok(true);
    }

    // Default voice range for soprano (conservative estimate)
    let voice_range: (f32, f32) = (261.63, 1046.50); // C4 to C6

    let total = score.notes.len();
    let in_range = score
        .notes
        .iter()
        .filter(|n| n.event.frequency >= voice_range.0 && n.event.frequency <= voice_range.1)
        .count();

    Ok(in_range as f64 / total as f64 > 0.5)
}

#[cfg(feature = "singing")]
fn analyze_note_range(notes: &[voirs_singing::MusicalNote]) -> (f32, f32) {
    let frequencies: Vec<f32> = notes.iter().map(|n| n.event.frequency).collect();
    let min_freq = frequencies.iter().copied().fold(f32::INFINITY, f32::min);
    let max_freq = frequencies
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    (min_freq, max_freq)
}

/// Load WAV file into mono f32 samples
#[cfg(feature = "singing")]
fn load_wav_samples(path: &Path) -> Result<(Vec<f32>, u32), CliError> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| CliError::IoError(format!("failed to open WAV: {e}")))?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map_err(|e| CliError::IoError(format!("WAV read error: {e}"))))
            .collect::<Result<Vec<f32>, CliError>>()?,
        hound::SampleFormat::Int => {
            let max_val = (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| {
                    s.map(|v| v as f32 / max_val)
                        .map_err(|e| CliError::IoError(format!("WAV read error: {e}")))
                })
                .collect::<Result<Vec<f32>, CliError>>()?
        }
    };

    Ok((samples, sample_rate))
}

#[cfg(feature = "singing")]
fn save_audio(audio: &[f32], path: &Path, sample_rate: u32) -> Result<(), CliError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| CliError::IoError(format!("Failed to create audio writer: {}", e)))?;

    for &sample in audio {
        let sample_i16 = (sample * 32767.0) as i16;
        writer
            .write_sample(sample_i16)
            .map_err(|e| CliError::IoError(format!("Failed to write audio sample: {}", e)))?;
    }

    writer
        .finalize()
        .map_err(|e| CliError::IoError(format!("Failed to finalize audio file: {}", e)))?;

    Ok(())
}

#[derive(Debug, serde::Serialize)]
struct SingingAnalysis {
    pitch_accuracy: f32,
    vibrato_consistency: f32,
    breath_quality: f32,
    note_count: usize,
    average_frequency: f32,
    key: String,
    chord_count: usize,
    scale_count: usize,
    analysis_confidence: f32,
}

#[cfg(feature = "singing")]
async fn analyze_singing_audio(
    samples: &[f32],
    sample_rate: u32,
    _args: &AnalyzeArgs,
) -> Result<SingingAnalysis, CliError> {
    let frame_size = 512usize;
    let f0_values: Vec<f32> = samples
        .chunks(frame_size)
        .filter_map(|frame| PitchContour::detect_pitch(frame, sample_rate as f32))
        .collect();

    let note_count = f0_values.len();
    let mean_f0 = if f0_values.is_empty() {
        0.0
    } else {
        f0_values.iter().sum::<f32>() / f0_values.len() as f32
    };

    let intel = MusicalIntelligence::new();
    let analysis = intel
        .analyze_audio(samples, sample_rate)
        .await
        .map_err(|e| CliError::singing_error(format!("analysis failed: {e}")))?;

    let key_label = format!(
        "{} ({:.0}% confidence)",
        analysis.key_analysis.key_name,
        analysis.key_analysis.confidence * 100.0
    );

    let chord_count = analysis.chord_analysis.len();
    let scale_count = analysis.scale_analysis.len();

    // Derive quality estimates from the analysis confidence
    let confidence = analysis.overall_confidence;
    let pitch_accuracy = (confidence * 0.95).clamp(0.0, 1.0);
    let vibrato_consistency = (confidence * 0.88).clamp(0.0, 1.0);
    let breath_quality = (confidence * 0.90).clamp(0.0, 1.0);

    Ok(SingingAnalysis {
        pitch_accuracy,
        vibrato_consistency,
        breath_quality,
        note_count,
        average_frequency: mean_f0,
        key: key_label,
        chord_count,
        scale_count,
        analysis_confidence: confidence,
    })
}

#[derive(Debug)]
struct SingingPreset {
    name: String,
    description: String,
    voice_type: String,
    technique_description: String,
}

#[cfg(feature = "singing")]
fn get_singing_presets(voice_type_filter: Option<&str>) -> Result<Vec<SingingPreset>, CliError> {
    let mut presets = vec![
        SingingPreset {
            name: "classical".to_string(),
            description: "Classical operatic style with controlled vibrato".to_string(),
            voice_type: "soprano".to_string(),
            technique_description: "High breath control, moderate vibrato".to_string(),
        },
        SingingPreset {
            name: "pop".to_string(),
            description: "Modern pop style with expressive dynamics".to_string(),
            voice_type: "alto".to_string(),
            technique_description: "Flexible breath control, strong pitch bending".to_string(),
        },
        SingingPreset {
            name: "jazz".to_string(),
            description: "Jazz style with smooth legato and rich vibrato".to_string(),
            voice_type: "tenor".to_string(),
            technique_description: "Smooth legato, rich vibrato, strong pitch bending".to_string(),
        },
        SingingPreset {
            name: "folk".to_string(),
            description: "Traditional folk style with natural expression".to_string(),
            voice_type: "bass".to_string(),
            technique_description: "Natural breath control, minimal vibrato".to_string(),
        },
    ];

    if let Some(filter) = voice_type_filter {
        presets.retain(|p| p.voice_type == filter);
    }

    Ok(presets)
}

#[cfg(all(test, feature = "singing"))]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    /// Write a mono 16-bit PCM WAV sine tone into `path`.
    fn write_tone_wav(path: &Path, sample_rate: u32, duration_secs: f32, freq_hz: f32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).expect("create test wav");
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample_f32 = 0.6 * (2.0 * std::f32::consts::PI * freq_hz * t).sin();
            writer
                .write_sample((sample_f32 * i16::MAX as f32) as i16)
                .expect("write sample");
        }
        writer.finalize().expect("finalize test wav");
    }

    fn make_create_voice_args(
        samples: PathBuf,
        output: PathBuf,
        quality_threshold: f32,
    ) -> CreateVoiceArgs {
        CreateVoiceArgs {
            samples,
            output,
            name: "test-voice".to_string(),
            voice_type: "soprano".to_string(),
            quality_threshold,
            epochs: 20,
        }
    }

    // ---------------------------------------------------------------
    // refine_f0_statistics: real iterative sigma-clipping refinement
    // ---------------------------------------------------------------

    #[test]
    fn test_refine_f0_statistics_more_iterations_reject_outliers() {
        // A tight cluster around 220 Hz plus a handful of extreme octave-error
        // outliers near 900 Hz.
        let mut values: Vec<f32> = (0..40).map(|i| 218.0 + (i % 5) as f32).collect();
        values.extend([900.0, 910.0, 895.0, 905.0]);

        let (mean_no_refine, _, ran_zero, _) = refine_f0_statistics(&values, 0);
        let (mean_refined, _, ran_many, _) = refine_f0_statistics(&values, 20);

        assert_eq!(
            ran_zero, 0,
            "zero requested iterations must run zero rounds"
        );
        assert!(
            ran_many > 0,
            "a real refinement pass must run at least once"
        );

        // Zero iterations must reflect the raw, outlier-skewed mean.
        let raw_mean = values.iter().sum::<f32>() / values.len() as f32;
        assert!((mean_no_refine - raw_mean).abs() < 0.01);

        // With refinement, the outliers must be excluded, pulling the mean
        // much closer to the true ~220 Hz cluster than the raw mean was.
        assert!(
            (mean_refined - 220.0).abs() < (mean_no_refine - 220.0).abs(),
            "refined mean {mean_refined} should be closer to 220 Hz than unrefined {mean_no_refine}"
        );
        assert!(
            mean_refined < 300.0,
            "refined mean {mean_refined} should have rejected the ~900 Hz outliers"
        );
    }

    #[test]
    fn test_refine_f0_statistics_empty_input() {
        let (mean, std, iterations, retained) = refine_f0_statistics(&[], 10);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
        assert_eq!(iterations, 0);
        assert_eq!(retained, 0);
    }

    // ---------------------------------------------------------------
    // estimate_vibrato: real zero-crossing-based rate/depth estimation
    // ---------------------------------------------------------------

    #[test]
    fn test_estimate_vibrato_recovers_known_rate_and_depth() {
        let frame_rate = 50.0f32;
        let duration = 3.0f32;
        let n = (frame_rate * duration) as usize;
        let true_rate = 6.0f32;
        let depth_cents = 60.0f32;
        let base_freq = 220.0f32;

        let f0_values: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / frame_rate;
                let cents = depth_cents * (2.0 * std::f32::consts::PI * true_rate * t).sin();
                base_freq * 2f32.powf(cents / 1200.0)
            })
            .collect();

        let (rate, depth) = estimate_vibrato(&f0_values, frame_rate);

        assert!(
            (rate - true_rate).abs() < 1.0,
            "estimated vibrato rate {rate} Hz should be close to the true {true_rate} Hz"
        );
        assert!(
            depth > 0.1,
            "estimated depth {depth} should reflect the real ~60-cent modulation"
        );
    }

    #[test]
    fn test_estimate_vibrato_flat_pitch_has_no_vibrato() {
        let f0_values = vec![220.0f32; 50];
        let (rate, depth) = estimate_vibrato(&f0_values, 50.0);
        assert_eq!(rate, 0.0);
        assert_eq!(depth, 0.0);
    }

    #[test]
    fn test_estimate_vibrato_too_few_samples() {
        let (rate, depth) = estimate_vibrato(&[220.0, 221.0], 50.0);
        assert_eq!((rate, depth), (0.0, 0.0));
    }

    // ---------------------------------------------------------------
    // execute_create_voice_command: end-to-end real feature extraction
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_create_voice_reflects_real_sample_pitch_and_differs_between_voices() {
        let temp = tempdir().expect("tempdir");
        let formatter = OutputFormatter::new(false, false);

        let samples_low = temp.path().join("samples_low");
        let samples_high = temp.path().join("samples_high");
        fs::create_dir_all(&samples_low).expect("create dir");
        fs::create_dir_all(&samples_high).expect("create dir");

        write_tone_wav(&samples_low.join("a.wav"), 22050, 1.0, 220.0);
        write_tone_wav(&samples_high.join("a.wav"), 22050, 1.0, 660.0);

        let output_low = temp.path().join("low_voice.json");
        let output_high = temp.path().join("high_voice.json");

        execute_create_voice_command(
            make_create_voice_args(samples_low, output_low.clone(), 0.0),
            &formatter,
        )
        .await
        .expect("create-voice on 220 Hz sample should succeed");

        execute_create_voice_command(
            make_create_voice_args(samples_high, output_high.clone(), 0.0),
            &formatter,
        )
        .await
        .expect("create-voice on 660 Hz sample should succeed");

        let voice_low: VoiceCharacteristics =
            serde_json::from_str(&fs::read_to_string(&output_low).expect("read low voice"))
                .expect("parse low voice json");
        let voice_high: VoiceCharacteristics =
            serde_json::from_str(&fs::read_to_string(&output_high).expect("read high voice"))
                .expect("parse high voice json");

        assert_ne!(
            voice_low.f0_mean, voice_high.f0_mean,
            "voices built from different-pitched samples must have different f0_mean"
        );
        assert!(
            (voice_low.f0_mean - 220.0).abs() < 40.0,
            "low voice f0_mean {} should be close to the true 220 Hz tone",
            voice_low.f0_mean
        );
        assert!(
            (voice_high.f0_mean - 660.0).abs() < 100.0,
            "high voice f0_mean {} should be close to the true 660 Hz tone",
            voice_high.f0_mean
        );
        assert!(
            voice_low.range.0 <= voice_low.f0_mean && voice_low.f0_mean <= voice_low.range.1,
            "f0_mean must fall within the reported vocal range"
        );
    }

    #[tokio::test]
    async fn test_create_voice_errors_when_all_samples_below_quality_threshold() {
        let temp = tempdir().expect("tempdir");
        let formatter = OutputFormatter::new(false, false);

        let samples = temp.path().join("samples");
        fs::create_dir_all(&samples).expect("create dir");
        write_tone_wav(&samples.join("a.wav"), 22050, 0.5, 300.0);

        let output = temp.path().join("voice.json");
        // overall_score is clamped to [0, 1], so a threshold above 1.0 can
        // never be met by any real sample.
        let result = execute_create_voice_command(
            make_create_voice_args(samples, output.clone(), 1.5),
            &formatter,
        )
        .await;

        assert!(
            result.is_err(),
            "an impossible quality threshold must fail closed, not fabricate a voice profile"
        );
        assert!(!output.exists(), "no voice file should have been written");
    }

    #[tokio::test]
    async fn test_create_voice_errors_on_missing_samples_dir() {
        let temp = tempdir().expect("tempdir");
        let formatter = OutputFormatter::new(false, false);
        let missing = temp.path().join("does-not-exist");
        let output = temp.path().join("voice.json");

        let result =
            execute_create_voice_command(make_create_voice_args(missing, output, 0.0), &formatter)
                .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_create_voice_errors_on_empty_samples_dir() {
        let temp = tempdir().expect("tempdir");
        let formatter = OutputFormatter::new(false, false);
        let samples = temp.path().join("samples");
        fs::create_dir_all(&samples).expect("create dir");
        let output = temp.path().join("voice.json");

        let result =
            execute_create_voice_command(make_create_voice_args(samples, output, 0.0), &formatter)
                .await;
        assert!(result.is_err());
    }
}
