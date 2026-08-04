//! Dataset management and validation commands.
//!
//! This module provides functionality for dataset validation, conversion,
//! splitting, preprocessing, and analysis for speech synthesis datasets.

use crate::progress::create_file_progress;
use crate::{DatasetCommands, GlobalOptions};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use voirs_dataset::audio::io::{load_audio, save_audio};
use voirs_dataset::formats::ManifestEntry;
use voirs_dataset::preprocessors::{AudioPreprocessingConfig, AudioPreprocessor};
use voirs_dataset::quality::metrics::{
    QualityConfig as AudioQualityConfig, QualityMetrics as AudioQualityMetrics,
    QualityMetricsCalculator,
};
use voirs_dataset::{
    AudioData, AudioFormat, Dataset, DatasetSample, DatasetSplit, DatasetSplits, LanguageCode,
    SplitConfig, SplitStrategy,
};
use voirs_sdk::config::AppConfig;
use voirs_sdk::error::IoOperation;
use voirs_sdk::{Result, VoirsError};

/// Build a [`VoirsError::IoError`] for a failed filesystem operation.
fn io_err(path: &Path, operation: IoOperation, source: std::io::Error) -> VoirsError {
    VoirsError::IoError {
        path: path.to_path_buf(),
        operation,
        source,
    }
}

/// Wrap a [`voirs_dataset::DatasetError`] as a [`VoirsError`].
fn dataset_err(e: voirs_dataset::DatasetError) -> VoirsError {
    VoirsError::audio_error(e.to_string())
}

/// Audio file validation result
#[derive(Debug, Clone)]
struct AudioFileInfo {
    path: PathBuf,
    sample_rate: u32,
    channels: u16,
    duration: f32,
    samples: usize,
    peak_level: f32,
    rms_level: Option<f32>,
    has_clipping: bool,
}

/// Dataset validation statistics
#[derive(Debug, Clone, Default)]
struct ValidationStatistics {
    total_files: usize,
    valid_files: usize,
    invalid_files: usize,
    total_duration: f32,
    sample_rates: HashMap<u32, usize>,
    min_duration: f32,
    max_duration: f32,
    avg_duration: f32,
    clipped_files: usize,
    avg_peak_level: f32,
    avg_rms_level: f32,
}

/// Execute dataset command
pub async fn execute_dataset_command(
    command: &DatasetCommands,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    match command {
        DatasetCommands::Validate {
            path,
            dataset_type,
            detailed,
        } => validate_dataset(path, dataset_type.as_deref(), *detailed, global).await,
        DatasetCommands::Convert {
            input,
            output,
            from,
            to,
        } => convert_dataset(input, output, from, to, global).await,
        DatasetCommands::Split {
            path,
            train_ratio,
            val_ratio,
            test_ratio,
            seed,
        } => split_dataset(path, *train_ratio, *val_ratio, *test_ratio, *seed, global).await,
        DatasetCommands::Preprocess {
            input,
            output,
            sample_rate,
            normalize,
            filter,
        } => preprocess_dataset(input, output, *sample_rate, *normalize, *filter, global).await,
        DatasetCommands::Analyze {
            path,
            output,
            detailed,
        } => analyze_dataset(path, output.as_deref(), *detailed, global).await,
    }
}

/// Validate dataset structure and quality
async fn validate_dataset(
    path: &Path,
    dataset_type: Option<&str>,
    detailed: bool,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("🔍 Validating dataset: {}", path.display());
        if let Some(dt) = dataset_type {
            println!("   Dataset type: {}", dt);
        } else {
            println!("   Dataset type: auto-detect");
        }
        println!();
    }

    // Check if path exists
    if !path.exists() {
        return Err(VoirsError::config_error(format!(
            "Dataset path does not exist: {}",
            path.display()
        )));
    }

    if !path.is_dir() {
        return Err(VoirsError::config_error(format!(
            "Dataset path is not a directory: {}",
            path.display()
        )));
    }

    if !global.quiet {
        println!("📊 Scanning and validating audio files...");
    }

    // Actually validate audio files (not just count)
    let audio_files = validate_audio_files(path, global).await?;
    let text_files = scan_text_files(path)?;

    // Calculate statistics
    let stats = calculate_validation_stats(&audio_files);

    if !global.quiet {
        println!(
            "✅ Found {} audio files ({} valid, {} invalid)",
            stats.total_files, stats.valid_files, stats.invalid_files
        );
        println!("✅ Found {} text files", text_files);

        if stats.valid_files > 0 {
            println!("\n📊 Audio Statistics:");
            println!(
                "   - Total duration: {:.1} hours",
                stats.total_duration / 3600.0
            );
            println!("   - Average duration: {:.2}s", stats.avg_duration);
            println!(
                "   - Duration range: {:.2}s - {:.2}s",
                stats.min_duration, stats.max_duration
            );

            // Sample rate distribution
            if stats.sample_rates.len() == 1 {
                let (sr, _) = stats
                    .sample_rates
                    .iter()
                    .next()
                    .expect("Sample rates map should have exactly one entry");
                println!("   - Sample rate: {} Hz (consistent)", sr);
            } else {
                println!("   - Sample rates (inconsistent):");
                for (sr, count) in &stats.sample_rates {
                    println!("     * {} Hz: {} files", sr, count);
                }
            }

            println!(
                "   - Average peak level: {:.1} dB",
                20.0 * stats.avg_peak_level.log10()
            );
            println!(
                "   - Average RMS level: {:.1} dB",
                20.0 * stats.avg_rms_level.log10()
            );

            if stats.clipped_files > 0 {
                println!("   ⚠️  Clipping detected: {} files", stats.clipped_files);
            } else {
                println!("   - Clipping: ✅ None detected");
            }
        }

        if detailed && stats.valid_files > 0 {
            println!("\n📋 Detailed Analysis:");

            // Quality checks
            if stats.sample_rates.len() > 1 {
                println!("   ⚠️  Sample rate inconsistency detected");
                println!("      Recommend resampling all files to a common sample rate");
            } else {
                println!("   ✅ Sample rate consistency: All files match");
            }

            if stats.clipped_files > 0 {
                println!(
                    "   ⚠️  Audio clipping: {} files affected ({:.1}%)",
                    stats.clipped_files,
                    (stats.clipped_files as f32 / stats.valid_files as f32) * 100.0
                );
            } else {
                println!("   ✅ Audio quality: No clipping detected");
            }

            // Duration analysis
            if stats.min_duration < 0.5 {
                println!(
                    "   ⚠️  Very short files detected (min: {:.2}s)",
                    stats.min_duration
                );
            }
            if stats.max_duration > 20.0 {
                println!(
                    "   ⚠️  Very long files detected (max: {:.2}s)",
                    stats.max_duration
                );
            }

            // Text-audio pairing
            if text_files > 0 {
                if text_files == stats.valid_files {
                    println!("   ✅ Text-audio pairing: Complete ({} pairs)", text_files);
                } else {
                    println!(
                        "   ⚠️  Text-audio mismatch: {} audio, {} text files",
                        stats.valid_files, text_files
                    );
                }
            }
        }

        if stats.invalid_files > 0 {
            println!(
                "\n⚠️  {} invalid/unreadable audio files found",
                stats.invalid_files
            );
        }

        if stats.valid_files == 0 {
            println!("\n❌ No valid audio files found in dataset");
            return Err(VoirsError::config_error("Empty or invalid dataset"));
        }

        println!("\n🎉 Dataset validation completed!");
    }

    Ok(())
}

/// Convert a flat directory of audio files (plus optional same-stem `.txt`
/// transcripts) from one audio codec to another.
///
/// Every source file matching `from`'s extension is actually decoded and
/// re-encoded to `to`; nothing is copied blindly and nothing is reported as
/// converted unless the destination file genuinely exists afterwards (see
/// [`save_audio_verified`], which guards against `voirs_dataset` encoders
/// that silently fall back to WAV when an external encoder is unavailable).
async fn convert_dataset(
    input: &Path,
    output: &Path,
    from: &str,
    to: &str,
    global: &GlobalOptions,
) -> Result<()> {
    if !input.exists() || !input.is_dir() {
        return Err(VoirsError::config_error(format!(
            "Input dataset path does not exist or is not a directory: {}",
            input.display()
        )));
    }

    let from_format = audio_format_from_name(from)?;
    let to_format = audio_format_from_name(to)?;
    let from_ext = audio_format_extension(from_format);
    let to_ext = audio_format_extension(to_format);

    if !global.quiet {
        println!("🔄 Converting dataset format");
        println!("   From: {} ({})", from, input.display());
        println!("   To: {} ({})", to, output.display());
        println!();
    }

    std::fs::create_dir_all(output).map_err(|e| io_err(output, IoOperation::Create, e))?;

    let source_files = list_files_with_extension(input, from_ext)?;
    if source_files.is_empty() {
        return Err(VoirsError::config_error(format!(
            "No '{from_ext}' audio files found in {}",
            input.display()
        )));
    }

    let progress = if global.quiet {
        None
    } else {
        Some(create_file_progress(source_files.len(), "Converting"))
    };

    let mut converted = 0usize;
    let mut transcripts_copied = 0usize;

    for src in &source_files {
        let stem = src
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("sample")
            .to_string();

        if let Some(p) = &progress {
            p.set_message(stem.clone());
        }

        let audio = load_audio(src).map_err(|e| {
            VoirsError::audio_error(format!("Failed to decode {}: {e}", src.display()))
        })?;

        let dest = output.join(format!("{stem}.{to_ext}"));
        save_audio_verified(&audio, &dest, to_format)?;
        converted += 1;

        // Copy a same-stem transcript through unchanged, if present.
        let src_txt = src.with_extension("txt");
        if src_txt.exists() {
            let dest_txt = output.join(format!("{stem}.txt"));
            std::fs::copy(&src_txt, &dest_txt)
                .map_err(|e| io_err(&src_txt, IoOperation::Copy, e))?;
            transcripts_copied += 1;
        }

        if let Some(p) = &progress {
            p.inc(1);
        }
    }

    if let Some(p) = progress {
        p.finish_with_message(format!("Converted {converted} files"));
    }

    if !global.quiet {
        println!("\n✅ Dataset conversion completed!");
        println!("   Files converted: {converted}");
        println!("   Transcripts copied: {transcripts_copied}");
        println!("   Output saved to: {}", output.display());
    }

    Ok(())
}

/// Parse a CLI-supplied dataset audio format name into an [`AudioFormat`].
fn audio_format_from_name(name: &str) -> Result<AudioFormat> {
    match name.trim().to_lowercase().as_str() {
        "wav" | "wave" => Ok(AudioFormat::Wav),
        "flac" => Ok(AudioFormat::Flac),
        "mp3" => Ok(AudioFormat::Mp3),
        "ogg" | "vorbis" => Ok(AudioFormat::Ogg),
        "opus" => Ok(AudioFormat::Opus),
        other => Err(VoirsError::config_error(format!(
            "Unsupported dataset audio format '{other}'. Supported formats: wav, flac, mp3, ogg, opus"
        ))),
    }
}

/// Canonical file extension for an [`AudioFormat`].
fn audio_format_extension(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::Wav => "wav",
        AudioFormat::Flac => "flac",
        AudioFormat::Mp3 => "mp3",
        AudioFormat::Ogg => "ogg",
        AudioFormat::Opus => "opus",
    }
}

/// Save `audio` to `path` and verify the encoder actually produced a file at
/// that exact path.
///
/// Some `voirs_dataset` codec encoders silently fall back to writing a
/// sibling `.wav` file instead of the requested format when an external
/// dependency is unavailable (FLAC without the `ffi-codecs` build feature;
/// MP3/OGG/OPUS without an `ffmpeg` binary on `PATH`) - and still return
/// `Ok(())`. Without this check the caller would report a successful
/// conversion for a file that was never created.
fn save_audio_verified(audio: &AudioData, path: &Path, format: AudioFormat) -> Result<()> {
    save_audio(audio, path).map_err(dataset_err)?;

    if !path.exists() {
        return Err(VoirsError::audio_error(format!(
            "Encoding to '{}' failed: no output file was produced at {} (the encoder in this \
             build likely requires the 'ffi-codecs' feature or an 'ffmpeg' binary on PATH); \
             use --to wav for a format that always works in this build",
            audio_format_extension(format),
            path.display(),
        )));
    }

    Ok(())
}

/// List files directly inside `dir` (non-recursive) whose extension matches
/// `ext` case-insensitively. Results are sorted for deterministic ordering.
fn list_files_with_extension(dir: &Path, ext: &str) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if dir.is_dir() {
        for entry in std::fs::read_dir(dir).map_err(|e| io_err(dir, IoOperation::Read, e))? {
            let entry = entry.map_err(|e| io_err(dir, IoOperation::Read, e))?;
            let path = entry.path();
            if path.is_file() {
                if let Some(file_ext) = path.extension().and_then(|e| e.to_str()) {
                    if file_ext.eq_ignore_ascii_case(ext) {
                        files.push(path);
                    }
                }
            }
        }
    }

    files.sort();
    Ok(files)
}

/// List every audio file (any [`AudioFormat`] extension) directly inside
/// `dir`, sorted for deterministic ordering.
fn list_audio_file_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for ext in ["wav", "flac", "mp3", "ogg", "opus"] {
        files.extend(list_files_with_extension(dir, ext)?);
    }
    files.sort();
    Ok(files)
}

/// Find a same-stem `.txt` transcript for an audio file, if one exists.
fn find_paired_transcript(audio_path: &Path) -> Option<String> {
    let txt_path = audio_path.with_extension("txt");
    std::fs::read_to_string(txt_path)
        .ok()
        .map(|s| s.trim().to_string())
}

/// Split a flat directory of audio files (plus optional same-stem `.txt`
/// transcripts) into `train`/`val`/`test` subdirectories.
///
/// Each source file is actually decoded so the split can be driven by
/// [`voirs_dataset::DatasetSplits::create_splits`] (real shuffling, seeded by
/// `seed` when provided); membership is then materialized as copied audio
/// files plus a JSON manifest per split, and the raw split indices are saved
/// via [`DatasetSplits::save_indices`] for reproducibility.
async fn split_dataset(
    path: &Path,
    train_ratio: f32,
    val_ratio: f32,
    test_ratio: Option<f32>,
    seed: Option<u64>,
    global: &GlobalOptions,
) -> Result<()> {
    if !path.exists() || !path.is_dir() {
        return Err(VoirsError::config_error(format!(
            "Dataset path does not exist or is not a directory: {}",
            path.display()
        )));
    }

    // Calculate test ratio if not provided
    let test_ratio = test_ratio.unwrap_or(1.0 - train_ratio - val_ratio);

    if !global.quiet {
        println!("✂️  Splitting dataset: {}", path.display());
        println!("   Train: {:.1}%", train_ratio * 100.0);
        println!("   Validation: {:.1}%", val_ratio * 100.0);
        println!("   Test: {:.1}%", test_ratio * 100.0);
        if let Some(s) = seed {
            println!("   Seed: {}", s);
        }
        println!();
    }

    // Validate the ratios up front (via the real crate validation, so the
    // error message is specific and accurate) before doing any expensive
    // file I/O or audio decoding.
    let mut split_config =
        SplitConfig::new(train_ratio, val_ratio, test_ratio, SplitStrategy::Random)
            .map_err(dataset_err)?;
    if let Some(s) = seed {
        split_config = split_config.with_seed(s);
    }

    // Refuse to clobber a previous split silently.
    for split_name in ["train", "val", "test"] {
        let split_dir = path.join(split_name);
        if split_dir.is_dir() {
            let has_entries = std::fs::read_dir(&split_dir)
                .map_err(|e| io_err(&split_dir, IoOperation::Read, e))?
                .next()
                .is_some();
            if has_entries {
                return Err(VoirsError::config_error(format!(
                    "Split output directory {} already exists and is not empty; remove it \
                     before re-running split",
                    split_dir.display()
                )));
            }
        }
    }

    let audio_paths = list_audio_file_paths(path)?;
    if audio_paths.is_empty() {
        return Err(VoirsError::config_error(
            "No audio files found in dataset".to_string(),
        ));
    }

    let progress = if global.quiet {
        None
    } else {
        Some(create_file_progress(audio_paths.len(), "Loading"))
    };

    let mut samples = Vec::with_capacity(audio_paths.len());
    for audio_path in &audio_paths {
        let stem = audio_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("sample")
            .to_string();

        if let Some(p) = &progress {
            p.set_message(stem.clone());
        }

        let audio = load_audio(audio_path).map_err(|e| {
            VoirsError::audio_error(format!("Failed to decode {}: {e}", audio_path.display()))
        })?;
        let text = find_paired_transcript(audio_path).unwrap_or_default();
        samples.push(DatasetSample::new(stem, text, audio, LanguageCode::EnUs));

        if let Some(p) = &progress {
            p.inc(1);
        }
    }
    if let Some(p) = progress {
        p.finish_with_message("Loaded".to_string());
    }

    let splits = DatasetSplits::create_splits(samples, split_config).map_err(dataset_err)?;

    if !global.quiet {
        println!("📊 Split summary:");
        println!("   Total files: {}", audio_paths.len());
        println!("   Train: {} files", splits.train.len());
        println!("   Validation: {} files", splits.validation.len());
        println!("   Test: {} files", splits.test.len());
        println!("\n📝 Writing split files and manifests...");
    }

    write_dataset_split(path, "train", &splits.train, &audio_paths)?;
    write_dataset_split(path, "val", &splits.validation, &audio_paths)?;
    write_dataset_split(path, "test", &splits.test, &audio_paths)?;

    splits.save_indices(path).map_err(dataset_err)?;

    if !global.quiet {
        println!("✅ Dataset split completed!");
    }

    Ok(())
}

/// Materialize one dataset split: copy each member's audio file (and
/// same-stem transcript, if any) into `base/<split_name>/`, and write a JSON
/// manifest (using [`ManifestEntry`]) listing every member with its real
/// decoded duration.
fn write_dataset_split(
    base: &Path,
    split_name: &str,
    split: &DatasetSplit,
    source_paths: &[PathBuf],
) -> Result<()> {
    if split.is_empty() {
        return Ok(());
    }

    let split_dir = base.join(split_name);
    std::fs::create_dir_all(&split_dir).map_err(|e| io_err(&split_dir, IoOperation::Create, e))?;

    let mut manifest: Vec<ManifestEntry> = Vec::with_capacity(split.len());

    for (member_idx, &original_idx) in split.indices.iter().enumerate() {
        let sample = &split.samples[member_idx];
        let source_path = &source_paths[original_idx];
        let file_name = source_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("sample")
            .to_string();

        let dest_audio = split_dir.join(&file_name);
        std::fs::copy(source_path, &dest_audio)
            .map_err(|e| io_err(source_path, IoOperation::Copy, e))?;

        let source_txt = source_path.with_extension("txt");
        if source_txt.exists() {
            let dest_txt = split_dir.join(format!("{}.txt", sample.id));
            std::fs::copy(&source_txt, &dest_txt)
                .map_err(|e| io_err(&source_txt, IoOperation::Copy, e))?;
        }

        let mut entry =
            ManifestEntry::new(sample.id.clone(), sample.text.clone(), file_name.clone());
        entry.duration = Some(sample.audio.duration());
        manifest.push(entry);
    }

    let manifest_path = split_dir.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| VoirsError::serialization("json", e.to_string()))?;
    std::fs::write(&manifest_path, manifest_json)
        .map_err(|e| io_err(&manifest_path, IoOperation::Write, e))?;

    Ok(())
}

/// Preprocess a flat directory of audio files for training: real resampling
/// (via [`AudioPreprocessor`]/[`AudioPreprocessingConfig`]), real peak
/// normalization, and real silence trimming, writing genuinely processed
/// audio (and any same-stem transcript, copied unchanged) into `output`.
async fn preprocess_dataset(
    input: &Path,
    output: &Path,
    sample_rate: u32,
    normalize: bool,
    filter: bool,
    global: &GlobalOptions,
) -> Result<()> {
    if !input.exists() || !input.is_dir() {
        return Err(VoirsError::config_error(format!(
            "Input dataset path does not exist or is not a directory: {}",
            input.display()
        )));
    }

    if !global.quiet {
        println!("⚙️  Preprocessing dataset");
        println!("   Input: {}", input.display());
        println!("   Output: {}", output.display());
        println!("   Target sample rate: {} Hz", sample_rate);
        println!(
            "   Normalize audio: {}",
            if normalize { "Yes" } else { "No" }
        );
        println!(
            "   Apply filters (trim silence): {}",
            if filter { "Yes" } else { "No" }
        );
        println!();
    }

    std::fs::create_dir_all(output).map_err(|e| io_err(output, IoOperation::Create, e))?;

    let audio_paths = list_audio_file_paths(input)?;
    if audio_paths.is_empty() {
        return Err(VoirsError::config_error(format!(
            "No audio files found in {}",
            input.display()
        )));
    }

    // `filter` maps onto real silence trimming (a genuine content-dependent
    // DSP filter step); duration bounds are left open since the CLI does not
    // expose flags for them, so no file is ever silently dropped here.
    let preprocessor = AudioPreprocessor::with_config(AudioPreprocessingConfig {
        target_sample_rate: Some(sample_rate),
        normalize,
        trim_silence: filter,
        silence_threshold: 0.01,
        to_mono: false,
        apply_fade: false,
        min_duration: None,
        max_duration: None,
    });

    let progress = if global.quiet {
        None
    } else {
        Some(create_file_progress(audio_paths.len(), "Preprocessing"))
    };

    let mut processed = 0usize;
    for audio_path in &audio_paths {
        let stem = audio_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("sample")
            .to_string();

        if let Some(p) = &progress {
            p.set_message(stem.clone());
        }

        let audio = load_audio(audio_path).map_err(|e| {
            VoirsError::audio_error(format!("Failed to decode {}: {e}", audio_path.display()))
        })?;
        let text = find_paired_transcript(audio_path).unwrap_or_default();
        let mut sample = DatasetSample::new(stem.clone(), text, audio, LanguageCode::EnUs);

        preprocessor.preprocess(&mut sample).map_err(dataset_err)?;

        let ext = audio_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("wav");
        let format = audio_format_from_name(ext).unwrap_or(AudioFormat::Wav);
        let dest = output.join(format!("{stem}.{}", audio_format_extension(format)));
        save_audio_verified(&sample.audio, &dest, format)?;

        let src_txt = audio_path.with_extension("txt");
        if src_txt.exists() {
            let dest_txt = output.join(format!("{stem}.txt"));
            std::fs::copy(&src_txt, &dest_txt)
                .map_err(|e| io_err(&src_txt, IoOperation::Copy, e))?;
        }

        processed += 1;
        if let Some(p) = &progress {
            p.inc(1);
        }
    }

    if let Some(p) = progress {
        p.finish_with_message(format!("Preprocessed {processed} files"));
    }

    if !global.quiet {
        println!("✅ Preprocessing completed!");
        println!("   Files processed: {processed}");
        println!("   Processed files saved to: {}", output.display());
    }

    Ok(())
}

/// Aggregated, real per-dataset quality/duration statistics computed from
/// [`AudioQualityMetrics`] measured on every successfully decoded file.
#[derive(Debug, Clone, Default)]
struct AggregatedQuality {
    total_duration: f32,
    min_duration: f32,
    max_duration: f32,
    mean_duration: f32,
    std_duration: f32,
    avg_snr: f32,
    avg_dynamic_range: f32,
    avg_peak_db: Option<f32>,
    uniform_sample_rate: Option<u32>,
}

/// Aggregate per-file quality metrics into dataset-wide statistics. Every
/// field is derived from `metrics`, so the result always varies with the
/// dataset actually analyzed (an empty slice yields all-zero/`None` fields).
fn aggregate_quality_metrics(metrics: &[AudioQualityMetrics]) -> AggregatedQuality {
    if metrics.is_empty() {
        return AggregatedQuality::default();
    }

    let n = metrics.len() as f32;
    let durations: Vec<f32> = metrics.iter().map(|m| m.duration).collect();
    let total_duration: f32 = durations.iter().sum();
    let mean_duration = total_duration / n;
    let variance = durations
        .iter()
        .map(|d| (d - mean_duration).powi(2))
        .sum::<f32>()
        / n;
    let std_duration = variance.sqrt();
    let min_duration = durations.iter().copied().fold(f32::MAX, f32::min);
    let max_duration = durations.iter().copied().fold(f32::MIN, f32::max);

    let avg_snr = metrics.iter().map(|m| m.snr).sum::<f32>() / n;
    let avg_dynamic_range = metrics.iter().map(|m| m.dynamic_range).sum::<f32>() / n;
    let avg_peak = metrics.iter().map(|m| m.peak).sum::<f32>() / n;
    let avg_peak_db = if avg_peak > 0.0 {
        Some(20.0 * avg_peak.log10())
    } else {
        None
    };

    let first_sr = metrics[0].sample_rate;
    let uniform_sample_rate = metrics
        .iter()
        .all(|m| m.sample_rate == first_sr)
        .then_some(first_sr);

    AggregatedQuality {
        total_duration,
        min_duration,
        max_duration,
        mean_duration,
        std_duration,
        avg_snr,
        avg_dynamic_range,
        avg_peak_db,
        uniform_sample_rate,
    }
}

/// Generate real dataset statistics and analysis.
///
/// Every audio file is actually decoded and measured via
/// [`QualityMetricsCalculator`] (duration, SNR, dynamic range, peak level);
/// every text statistic is computed from the real content of `.txt` files
/// (never guessed from a file count, and never from raw `.csv`/`.json`
/// markup). Files that fail to decode are counted and reported, not hidden.
async fn analyze_dataset(
    path: &Path,
    output: Option<&Path>,
    detailed: bool,
    global: &GlobalOptions,
) -> Result<()> {
    if !path.exists() || !path.is_dir() {
        return Err(VoirsError::config_error(format!(
            "Dataset path does not exist or is not a directory: {}",
            path.display()
        )));
    }

    if !global.quiet {
        println!("📊 Analyzing dataset: {}", path.display());
        println!();
    }

    let audio_paths = list_audio_file_paths(path)?;
    let text_file_count = scan_text_files(path)?;
    let txt_paths = list_files_with_extension(path, "txt")?;

    let calculator = QualityMetricsCalculator::new(AudioQualityConfig::default());
    let progress = if global.quiet || audio_paths.is_empty() {
        None
    } else {
        Some(create_file_progress(audio_paths.len(), "Analyzing"))
    };

    let mut file_metrics: Vec<AudioQualityMetrics> = Vec::with_capacity(audio_paths.len());
    let mut unreadable = 0usize;

    for audio_path in &audio_paths {
        if let Some(p) = &progress {
            p.set_message(
                audio_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default()
                    .to_string(),
            );
        }

        let decoded = load_audio(audio_path)
            .map_err(dataset_err)
            .and_then(|audio| calculator.calculate_metrics(&audio).map_err(dataset_err));

        match decoded {
            Ok(metrics) => file_metrics.push(metrics),
            Err(e) => {
                unreadable += 1;
                if !global.quiet {
                    eprintln!("⚠️  Failed to analyze {}: {e}", audio_path.display());
                }
            }
        }

        if let Some(p) = &progress {
            p.inc(1);
        }
    }
    if let Some(p) = progress {
        p.finish_with_message("Analysis complete".to_string());
    }

    // Text statistics come only from real `.txt` content, never from raw
    // `.csv`/`.json` markup (which would inflate character/vocabulary counts
    // with formatting noise rather than actual words).
    let mut total_chars = 0usize;
    let mut vocabulary: HashSet<String> = HashSet::new();
    for txt_path in &txt_paths {
        if let Ok(content) = std::fs::read_to_string(txt_path) {
            total_chars += content.chars().count();
            for word in content.split_whitespace() {
                let normalized: String = word
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect();
                if !normalized.is_empty() {
                    vocabulary.insert(normalized);
                }
            }
        }
    }
    let avg_chars_per_file = if txt_paths.is_empty() {
        0.0
    } else {
        total_chars as f32 / txt_paths.len() as f32
    };

    let stats = aggregate_quality_metrics(&file_metrics);
    let peak_db_str = match stats.avg_peak_db {
        Some(db) => format!("{db:.1} dB"),
        None => "N/A".to_string(),
    };
    let sample_rate_str = match stats.uniform_sample_rate {
        Some(sr) => format!("{sr} Hz (consistent)"),
        None => "mixed across files".to_string(),
    };

    if !global.quiet {
        println!("📈 Dataset Statistics:");
        println!("   Total audio files: {}", audio_paths.len());
        println!("   Total text files: {}", text_file_count);
        if unreadable > 0 {
            println!("   ⚠️  Audio files that failed to decode: {unreadable}");
        }

        if !file_metrics.is_empty() {
            println!(
                "   Total duration: {:.2} hours",
                stats.total_duration / 3600.0
            );
            println!("   Average file duration: {:.2}s", stats.mean_duration);
            println!("   Sample rate: {sample_rate_str}");
        }

        if detailed && !file_metrics.is_empty() {
            println!("\n🔍 Detailed Analysis:");
            println!("   Duration distribution:");
            println!("     - Min: {:.2}s", stats.min_duration);
            println!("     - Max: {:.2}s", stats.max_duration);
            println!("     - Mean: {:.2}s", stats.mean_duration);
            println!("     - Std dev: {:.2}s", stats.std_duration);

            println!("   Audio quality metrics (average over decoded files):");
            println!("     - SNR: {:.1} dB", stats.avg_snr);
            println!("     - Dynamic range: {:.1} dB", stats.avg_dynamic_range);
            println!("     - Peak level: {peak_db_str}");

            println!("   Text analysis (from {} .txt files):", txt_paths.len());
            println!("     - Total characters: {total_chars}");
            println!("     - Average characters per file: {avg_chars_per_file:.1}");
            println!("     - Vocabulary size: {} unique words", vocabulary.len());
        }
    }

    if let Some(output_path) = output {
        if !global.quiet {
            println!("\n💾 Saving analysis report to: {}", output_path.display());
        }

        let report = format!(
            "# Dataset Analysis Report\n\n\
            ## Summary\n\
            - Audio files: {audio_count}\n\
            - Audio files that failed to decode: {unreadable}\n\
            - Text files: {text_count}\n\
            - Total duration: {total_hours:.2} hours\n\
            - Average duration: {mean_duration:.2} seconds\n\
            - Duration std dev: {std_duration:.2} seconds\n\n\
            ## Quality Metrics (averaged over {analyzed_count} decoded files)\n\
            - Sample rate: {sample_rate_str}\n\
            - SNR: {avg_snr:.1} dB\n\
            - Dynamic range: {avg_dynamic_range:.1} dB\n\
            - Peak level: {peak_db_str}\n\n\
            ## Text Statistics (from {txt_count} .txt files)\n\
            - Total characters: {total_chars}\n\
            - Vocabulary size: {vocab_size} unique words\n\n\
            Generated by VoiRS CLI\n",
            audio_count = audio_paths.len(),
            unreadable = unreadable,
            text_count = text_file_count,
            total_hours = stats.total_duration / 3600.0,
            mean_duration = stats.mean_duration,
            std_duration = stats.std_duration,
            analyzed_count = file_metrics.len(),
            sample_rate_str = sample_rate_str,
            avg_snr = stats.avg_snr,
            avg_dynamic_range = stats.avg_dynamic_range,
            peak_db_str = peak_db_str,
            txt_count = txt_paths.len(),
            total_chars = total_chars,
            vocab_size = vocabulary.len(),
        );

        std::fs::write(output_path, report)
            .map_err(|e| io_err(output_path, IoOperation::Write, e))?;
    }

    if !global.quiet {
        println!("\n✅ Dataset analysis completed!");
    }

    Ok(())
}

/// Validate audio files and return detailed information
async fn validate_audio_files(path: &Path, global: &GlobalOptions) -> Result<Vec<AudioFileInfo>> {
    let mut audio_files = Vec::new();

    if path.is_dir() {
        for entry in std::fs::read_dir(path).map_err(|e| VoirsError::IoError {
            path: path.to_path_buf(),
            operation: voirs_sdk::error::IoOperation::Read,
            source: e,
        })? {
            let entry = entry.map_err(|e| VoirsError::IoError {
                path: path.to_path_buf(),
                operation: voirs_sdk::error::IoOperation::Read,
                source: e,
            })?;

            let file_path = entry.path();
            if let Some(ext) = file_path.extension() {
                if ext == "wav" {
                    if let Some(info) = validate_wav_file(&file_path, global).await {
                        audio_files.push(info);
                    }
                } else if ext == "flac" || ext == "mp3" {
                    // For now, count but don't validate non-WAV files
                    // Full implementation would use claxon/minimp3
                    if !global.quiet {
                        eprintln!(
                            "⚠️  Skipping {}: {} format not yet supported for validation",
                            file_path.display(),
                            ext.to_str().unwrap_or("unknown")
                        );
                    }
                }
            }
        }
    }

    Ok(audio_files)
}

/// Validate a single WAV file
async fn validate_wav_file(path: &PathBuf, global: &GlobalOptions) -> Option<AudioFileInfo> {
    use hound::WavReader;

    match WavReader::open(path) {
        Ok(reader) => {
            let spec = reader.spec();
            let sample_rate = spec.sample_rate;
            let channels = spec.channels;
            let bits_per_sample = spec.bits_per_sample;
            let sample_format = spec.sample_format;

            // Read all samples to calculate duration and quality metrics
            let samples: Vec<f32> = match (sample_format, bits_per_sample) {
                (hound::SampleFormat::Int, 16) => reader
                    .into_samples::<i16>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / i16::MAX as f32)
                    .collect(),
                (hound::SampleFormat::Int, 24) => {
                    reader
                        .into_samples::<i32>()
                        .filter_map(|s| s.ok())
                        .map(|s| s as f32 / 8388608.0) // 2^23
                        .collect()
                }
                (hound::SampleFormat::Int, 32) => reader
                    .into_samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / i32::MAX as f32)
                    .collect(),
                (hound::SampleFormat::Float, 32) => reader
                    .into_samples::<f32>()
                    .filter_map(|s| s.ok())
                    .collect(),
                _ => {
                    if !global.quiet {
                        eprintln!(
                            "⚠️  Unsupported format: {} ({} bit, {:?})",
                            path.display(),
                            bits_per_sample,
                            sample_format
                        );
                    }
                    return None;
                }
            };

            if samples.is_empty() {
                if !global.quiet {
                    eprintln!("⚠️  Empty audio file: {}", path.display());
                }
                return None;
            }

            let sample_count = samples.len();
            let duration = sample_count as f32 / (sample_rate * channels as u32) as f32;

            // Create AudioData to use voirs-dataset's quality metrics
            let audio_data = AudioData::new(samples, sample_rate, channels as u32);

            // Calculate peak level
            let peak_level = audio_data.peak().unwrap_or(0.0);

            // Calculate RMS level
            let rms_level = audio_data.rms();

            // Detect clipping (samples at or near maximum amplitude)
            let has_clipping = peak_level >= 0.99;

            Some(AudioFileInfo {
                path: path.clone(),
                sample_rate,
                channels,
                duration,
                samples: sample_count,
                peak_level,
                rms_level,
                has_clipping,
            })
        }
        Err(e) => {
            if !global.quiet {
                eprintln!("⚠️  Failed to read {}: {}", path.display(), e);
            }
            None
        }
    }
}

/// Calculate validation statistics from audio file info
fn calculate_validation_stats(files: &[AudioFileInfo]) -> ValidationStatistics {
    if files.is_empty() {
        return ValidationStatistics::default();
    }

    let valid_files = files.len();
    let total_files = valid_files; // Invalid files already filtered out

    let mut sample_rates = HashMap::new();
    let mut total_duration = 0.0;
    let mut min_duration = f32::MAX;
    let mut max_duration = f32::MIN;
    let mut clipped_files = 0;
    let mut total_peak = 0.0;
    let mut total_rms = 0.0;
    let mut rms_count = 0;

    for file in files {
        // Sample rate distribution
        *sample_rates.entry(file.sample_rate).or_insert(0) += 1;

        // Duration statistics
        total_duration += file.duration;
        min_duration = min_duration.min(file.duration);
        max_duration = max_duration.max(file.duration);

        // Clipping detection
        if file.has_clipping {
            clipped_files += 1;
        }

        // Peak level average
        total_peak += file.peak_level;

        // RMS level average
        if let Some(rms) = file.rms_level {
            total_rms += rms;
            rms_count += 1;
        }
    }

    let avg_duration = total_duration / valid_files as f32;
    let avg_peak_level = total_peak / valid_files as f32;
    let avg_rms_level = if rms_count > 0 {
        total_rms / rms_count as f32
    } else {
        0.0
    };

    ValidationStatistics {
        total_files,
        valid_files,
        invalid_files: 0, // Already filtered during validation
        total_duration,
        sample_rates,
        min_duration,
        max_duration,
        avg_duration,
        clipped_files,
        avg_peak_level,
        avg_rms_level,
    }
}

/// Scan for audio files in directory
fn scan_audio_files(path: &Path) -> Result<usize> {
    let mut count = 0;

    if path.is_dir() {
        for entry in std::fs::read_dir(path).map_err(|e| VoirsError::IoError {
            path: path.to_path_buf(),
            operation: voirs_sdk::error::IoOperation::Read,
            source: e,
        })? {
            let entry = entry.map_err(|e| VoirsError::IoError {
                path: path.to_path_buf(),
                operation: voirs_sdk::error::IoOperation::Read,
                source: e,
            })?;

            if let Some(ext) = entry.path().extension() {
                if matches!(ext.to_str(), Some("wav") | Some("flac") | Some("mp3")) {
                    count += 1;
                }
            }
        }
    }

    Ok(count)
}

/// Scan for text files in directory
fn scan_text_files(path: &Path) -> Result<usize> {
    let mut count = 0;

    if path.is_dir() {
        for entry in std::fs::read_dir(path).map_err(|e| VoirsError::IoError {
            path: path.to_path_buf(),
            operation: voirs_sdk::error::IoOperation::Read,
            source: e,
        })? {
            let entry = entry.map_err(|e| VoirsError::IoError {
                path: path.to_path_buf(),
                operation: voirs_sdk::error::IoOperation::Read,
                source: e,
            })?;

            if let Some(ext) = entry.path().extension() {
                if matches!(ext.to_str(), Some("txt") | Some("csv") | Some("json")) {
                    count += 1;
                }
            }
        }
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    /// Default [`GlobalOptions`] for tests (quiet, to keep test output clean).
    fn quiet_global() -> GlobalOptions {
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

    /// Write a mono 16-bit PCM WAV sine tone, for tests that need a real,
    /// distinguishable, decodable audio file.
    fn write_test_wav(path: &Path, sample_rate: u32, duration_secs: f32, freq_hz: f32) {
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
            let sample_f32 = 0.5 * (2.0 * std::f32::consts::PI * freq_hz * t).sin();
            writer
                .write_sample((sample_f32 * i16::MAX as f32) as i16)
                .expect("write sample");
        }
        writer.finalize().expect("finalize test wav");
    }

    // ---------------------------------------------------------------
    // convert_dataset
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_convert_dataset_wav_round_trip_produces_real_decodable_output() {
        let temp = tempdir().expect("tempdir");
        let input = temp.path().join("in");
        let output = temp.path().join("out");
        fs::create_dir_all(&input).expect("create input dir");

        write_test_wav(&input.join("a.wav"), 22050, 0.5, 440.0);
        write_test_wav(&input.join("b.wav"), 22050, 0.3, 220.0);
        fs::write(input.join("a.txt"), "hello world").expect("write transcript");

        let global = quiet_global();
        convert_dataset(&input, &output, "wav", "wav", &global)
            .await
            .expect("wav -> wav conversion should succeed");

        assert!(output.join("a.wav").exists(), "a.wav must exist in output");
        assert!(output.join("b.wav").exists(), "b.wav must exist in output");
        assert!(
            output.join("a.txt").exists(),
            "paired transcript must be copied"
        );

        // The output must be genuinely decodable and non-empty, not a stub file.
        let reader_a = hound::WavReader::open(output.join("a.wav")).expect("decode a.wav");
        let reader_b = hound::WavReader::open(output.join("b.wav")).expect("decode b.wav");
        assert!(reader_a.len() > 0);
        assert!(reader_b.len() > 0);
        // Different source durations must yield different real sample counts.
        assert_ne!(reader_a.len(), reader_b.len());
    }

    #[tokio::test]
    async fn test_convert_dataset_rejects_unsupported_format_name() {
        let temp = tempdir().expect("tempdir");
        let input = temp.path().join("in");
        let output = temp.path().join("out");
        fs::create_dir_all(&input).expect("create input dir");
        write_test_wav(&input.join("a.wav"), 16000, 0.2, 440.0);

        let global = quiet_global();
        let result = convert_dataset(&input, &output, "wav", "not-a-real-format", &global).await;
        assert!(result.is_err(), "unsupported target format must error");
        assert!(
            !output.exists(),
            "no output directory should be created before format validation"
        );
    }

    // These two tests are the complementary halves of the same contract,
    // selected by voirs-cli's own `ffi-codecs` feature (which forwards to
    // voirs-dataset's `ffi-codecs`, the feature that actually gates
    // `voirs_dataset::audio::io::save_flac`'s real encoder vs. its
    // WAV-fallback path). Exactly one of the two compiles in any given
    // build, so the observable behavior is pinned in both configurations.
    #[cfg(not(feature = "ffi-codecs"))]
    #[tokio::test]
    async fn test_convert_dataset_flac_target_fails_closed_when_encoder_unavailable() {
        // Without voirs-cli's `ffi-codecs` feature (the default build),
        // voirs-dataset's `save_flac` deterministically falls back to
        // writing a sibling `.wav` file. The command must therefore return a
        // real error naming the missing output, never a fabricated
        // "conversion completed" success.
        let temp = tempdir().expect("tempdir");
        let input = temp.path().join("in");
        let output = temp.path().join("out");
        fs::create_dir_all(&input).expect("create input dir");
        write_test_wav(&input.join("a.wav"), 16000, 0.2, 440.0);

        let global = quiet_global();
        let result = convert_dataset(&input, &output, "wav", "flac", &global).await;

        assert!(
            result.is_err(),
            "flac target must fail closed in this build instead of reporting fake success"
        );
        assert!(
            !output.join("a.flac").exists(),
            "no .flac file should have been produced"
        );
    }

    #[cfg(feature = "ffi-codecs")]
    #[tokio::test]
    async fn test_convert_dataset_flac_target_succeeds_when_encoder_available() {
        // With voirs-cli's `ffi-codecs` feature enabled, voirs-dataset's
        // `save_flac` uses the real Pure-Rust OxiAudio FLAC encoder, so the
        // conversion must genuinely succeed and produce a decodable FLAC
        // file - not silently fall back to WAV.
        let temp = tempdir().expect("tempdir");
        let input = temp.path().join("in");
        let output = temp.path().join("out");
        fs::create_dir_all(&input).expect("create input dir");
        write_test_wav(&input.join("a.wav"), 16000, 0.2, 440.0);

        let global = quiet_global();
        let result = convert_dataset(&input, &output, "wav", "flac", &global).await;

        assert!(
            result.is_ok(),
            "flac target must succeed when the ffi-codecs encoder is available: {:?}",
            result.err()
        );
        let flac_path = output.join("a.flac");
        assert!(flac_path.exists(), "a.flac must exist in output");

        // The output must be genuinely decodable and non-empty, not a stub file.
        let decoded = load_audio(&flac_path).expect("decode a.flac as real FLAC");
        assert!(
            !decoded.samples().is_empty(),
            "decoded FLAC must have samples"
        );
    }

    #[tokio::test]
    async fn test_convert_dataset_errors_on_missing_input_dir() {
        let temp = tempdir().expect("tempdir");
        let input = temp.path().join("does-not-exist");
        let output = temp.path().join("out");

        let global = quiet_global();
        let result = convert_dataset(&input, &output, "wav", "wav", &global).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_convert_dataset_errors_when_no_source_files_match() {
        let temp = tempdir().expect("tempdir");
        let input = temp.path().join("in");
        let output = temp.path().join("out");
        fs::create_dir_all(&input).expect("create input dir");
        // Only a flac-named placeholder exists; requesting `--from wav` must
        // find nothing and error rather than silently "succeed" with zero
        // files processed.
        fs::write(input.join("a.flac"), b"not decoded, just present").unwrap();

        let global = quiet_global();
        let result = convert_dataset(&input, &output, "wav", "wav", &global).await;
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // split_dataset
    // ---------------------------------------------------------------

    fn populate_split_source(dir: &Path, count: usize) {
        fs::create_dir_all(dir).expect("create dir");
        for i in 0..count {
            let freq = 100.0 + i as f32 * 10.0;
            write_test_wav(&dir.join(format!("sample_{i:03}.wav")), 16000, 0.05, freq);
            fs::write(
                dir.join(format!("sample_{i:03}.txt")),
                format!("utterance number {i}"),
            )
            .expect("write transcript");
        }
    }

    #[tokio::test]
    async fn test_split_dataset_writes_manifests_covering_every_file_without_overlap() {
        let temp = tempdir().expect("tempdir");
        let dataset_dir = temp.path().join("dataset");
        populate_split_source(&dataset_dir, 20);

        let global = quiet_global();
        split_dataset(&dataset_dir, 0.7, 0.15, Some(0.15), Some(42), &global)
            .await
            .expect("split should succeed");

        let mut all_members = HashSet::new();
        let mut total = 0usize;
        for split_name in ["train", "val", "test"] {
            let manifest_path = dataset_dir.join(split_name).join("manifest.json");
            assert!(manifest_path.exists(), "{split_name} manifest must exist");
            let content = fs::read_to_string(&manifest_path).expect("read manifest");
            let entries: Vec<ManifestEntry> =
                serde_json::from_str(&content).expect("parse manifest");
            for entry in &entries {
                assert!(
                    dataset_dir
                        .join(split_name)
                        .join(&entry.audio_path)
                        .exists(),
                    "manifest-referenced audio file must actually exist on disk"
                );
                assert!(
                    all_members.insert(entry.id.clone()),
                    "sample {} must not appear in more than one split",
                    entry.id
                );
            }
            total += entries.len();
        }
        assert_eq!(
            total, 20,
            "every source file must land in exactly one split"
        );

        assert!(dataset_dir.join("train_indices.json").exists());
        assert!(dataset_dir.join("split_config.json").exists());
    }

    #[tokio::test]
    async fn test_split_dataset_same_seed_is_reproducible_different_seed_differs() {
        let temp = tempdir().expect("tempdir");
        let dir_seed_a1 = temp.path().join("seed_a1");
        let dir_seed_a2 = temp.path().join("seed_a2");
        let dir_seed_b = temp.path().join("seed_b");
        populate_split_source(&dir_seed_a1, 30);
        populate_split_source(&dir_seed_a2, 30);
        populate_split_source(&dir_seed_b, 30);

        let global = quiet_global();
        split_dataset(&dir_seed_a1, 0.7, 0.15, Some(0.15), Some(7), &global)
            .await
            .expect("split a1 should succeed");
        split_dataset(&dir_seed_a2, 0.7, 0.15, Some(0.15), Some(7), &global)
            .await
            .expect("split a2 should succeed");
        split_dataset(&dir_seed_b, 0.7, 0.15, Some(0.15), Some(99), &global)
            .await
            .expect("split b should succeed");

        let train_ids = |dir: &Path| -> Vec<String> {
            let content =
                fs::read_to_string(dir.join("train_indices.json")).expect("read train indices");
            serde_json::from_str::<Vec<usize>>(&content)
                .expect("parse indices")
                .into_iter()
                .map(|i| i.to_string())
                .collect()
        };

        let indices_a1 = train_ids(&dir_seed_a1);
        let indices_a2 = train_ids(&dir_seed_a2);
        let indices_b = train_ids(&dir_seed_b);

        assert_eq!(
            indices_a1, indices_a2,
            "identical seed on identical input must reproduce the same split membership"
        );
        assert_ne!(
            indices_a1, indices_b,
            "a different seed must (with overwhelming probability) change split membership"
        );
    }

    #[tokio::test]
    async fn test_split_dataset_rejects_ratios_that_do_not_sum_to_one() {
        let temp = tempdir().expect("tempdir");
        let dataset_dir = temp.path().join("dataset");
        populate_split_source(&dataset_dir, 5);

        let global = quiet_global();
        let result = split_dataset(&dataset_dir, 0.9, 0.5, Some(0.5), None, &global).await;
        assert!(result.is_err());
        // No split directories should have been created for an invalid request.
        assert!(!dataset_dir.join("train").exists());
    }

    #[tokio::test]
    async fn test_split_dataset_refuses_to_clobber_existing_split() {
        let temp = tempdir().expect("tempdir");
        let dataset_dir = temp.path().join("dataset");
        populate_split_source(&dataset_dir, 10);

        let global = quiet_global();
        split_dataset(&dataset_dir, 0.6, 0.2, Some(0.2), Some(1), &global)
            .await
            .expect("first split should succeed");

        let second = split_dataset(&dataset_dir, 0.6, 0.2, Some(0.2), Some(2), &global).await;
        assert!(
            second.is_err(),
            "re-running split over an existing non-empty split dir must error, not silently mix results"
        );
    }

    // ---------------------------------------------------------------
    // preprocess_dataset
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_preprocess_dataset_resamples_to_requested_rate() {
        let temp = tempdir().expect("tempdir");
        let input = temp.path().join("in");
        let output = temp.path().join("out");
        fs::create_dir_all(&input).expect("create input dir");
        write_test_wav(&input.join("a.wav"), 44100, 0.5, 440.0);

        let global = quiet_global();
        preprocess_dataset(&input, &output, 22050, false, false, &global)
            .await
            .expect("preprocess should succeed");

        let reader = hound::WavReader::open(output.join("a.wav")).expect("decode output");
        assert_eq!(reader.spec().sample_rate, 22050);
        // 44100 Hz * 0.5s = 22050 input samples; resampling 44100 -> 22050
        // (ratio 2.0) halves the sample count to ~11025.
        let expected = 11_025i64;
        let actual = reader.len() as i64;
        assert!(
            (actual - expected).abs() < 200,
            "expected roughly {expected} samples after halving the rate, got {actual}"
        );
    }

    #[tokio::test]
    async fn test_preprocess_dataset_normalize_increases_peak_toward_full_scale() {
        let temp = tempdir().expect("tempdir");
        let input = temp.path().join("in");
        let output = temp.path().join("out");
        fs::create_dir_all(&input).expect("create input dir");

        // A quiet tone (well below full scale).
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        {
            let mut writer =
                hound::WavWriter::create(input.join("a.wav"), spec).expect("create wav");
            for i in 0..8000 {
                let t = i as f32 / 16000.0;
                let s = 0.05 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
                writer
                    .write_sample((s * i16::MAX as f32) as i16)
                    .expect("write sample");
            }
            writer.finalize().expect("finalize");
        }

        let global = quiet_global();
        preprocess_dataset(&input, &output, 16000, true, false, &global)
            .await
            .expect("preprocess should succeed");

        let mut reader = hound::WavReader::open(output.join("a.wav")).expect("decode output");
        let peak = reader
            .samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s| s.unsigned_abs())
            .max()
            .unwrap_or(0);
        assert!(
            peak > 20000,
            "normalized peak should approach full scale, got {peak}"
        );
    }

    #[tokio::test]
    async fn test_preprocess_dataset_filter_trims_leading_silence() {
        let temp = tempdir().expect("tempdir");
        let input = temp.path().join("in");
        let output_filtered = temp.path().join("out_filtered");
        let output_unfiltered = temp.path().join("out_unfiltered");
        fs::create_dir_all(&input).expect("create input dir");

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        {
            let mut writer =
                hound::WavWriter::create(input.join("a.wav"), spec).expect("create wav");
            // 0.3s of silence, then 0.3s of tone.
            for _ in 0..4800 {
                writer.write_sample(0i16).expect("write silence");
            }
            for i in 0..4800 {
                let t = i as f32 / 16000.0;
                let s = 0.8 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
                writer
                    .write_sample((s * i16::MAX as f32) as i16)
                    .expect("write sample");
            }
            writer.finalize().expect("finalize");
        }

        let global = quiet_global();
        preprocess_dataset(&input, &output_filtered, 16000, false, true, &global)
            .await
            .expect("filtered preprocess should succeed");
        preprocess_dataset(&input, &output_unfiltered, 16000, false, false, &global)
            .await
            .expect("unfiltered preprocess should succeed");

        let filtered_len = hound::WavReader::open(output_filtered.join("a.wav"))
            .expect("decode filtered")
            .len();
        let unfiltered_len = hound::WavReader::open(output_unfiltered.join("a.wav"))
            .expect("decode unfiltered")
            .len();

        assert!(
            filtered_len < unfiltered_len,
            "trimming silence must shorten the output ({filtered_len} vs {unfiltered_len})"
        );
    }

    #[tokio::test]
    async fn test_preprocess_dataset_errors_on_missing_input_dir() {
        let temp = tempdir().expect("tempdir");
        let input = temp.path().join("does-not-exist");
        let output = temp.path().join("out");

        let global = quiet_global();
        let result = preprocess_dataset(&input, &output, 22050, false, false, &global).await;
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // analyze_dataset
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_analyze_dataset_stats_vary_with_dataset_content() {
        let temp = tempdir().expect("tempdir");
        let short_dir = temp.path().join("short");
        let long_dir = temp.path().join("long");
        fs::create_dir_all(&short_dir).expect("create dir");
        fs::create_dir_all(&long_dir).expect("create dir");

        write_test_wav(&short_dir.join("a.wav"), 16000, 0.2, 440.0);
        fs::write(short_dir.join("a.txt"), "hi").unwrap();

        write_test_wav(&long_dir.join("a.wav"), 16000, 3.0, 440.0);
        write_test_wav(&long_dir.join("b.wav"), 16000, 4.0, 220.0);
        fs::write(
            long_dir.join("a.txt"),
            "this is a substantially longer transcript with many more words in it",
        )
        .unwrap();

        let global = quiet_global();
        let short_report = temp.path().join("short_report.md");
        let long_report = temp.path().join("long_report.md");

        analyze_dataset(&short_dir, Some(&short_report), true, &global)
            .await
            .expect("analyze short dataset should succeed");
        analyze_dataset(&long_dir, Some(&long_report), true, &global)
            .await
            .expect("analyze long dataset should succeed");

        let short_text = fs::read_to_string(&short_report).expect("read short report");
        let long_text = fs::read_to_string(&long_report).expect("read long report");

        assert_ne!(
            short_text, long_text,
            "reports for datasets with different content must differ"
        );

        // Extract the "Average duration" (in seconds) line from each and
        // confirm the long dataset (0.2s vs mean of 3.0s/4.0s = 3.5s) reports
        // a materially larger duration - fabricated code always printed the
        // same "4.2 seconds" constant regardless of input.
        let extract_avg_duration_secs = |report: &str| -> f32 {
            report
                .lines()
                .find(|l| l.starts_with("- Average duration"))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|v| v.trim().split(' ').next())
                .and_then(|v| v.parse::<f32>().ok())
                .expect("average duration line present")
        };
        let short_avg = extract_avg_duration_secs(&short_text);
        let long_avg = extract_avg_duration_secs(&long_text);
        assert!(
            long_avg > short_avg * 5.0,
            "long dataset ({long_avg}s avg) should report much more duration than short ({short_avg}s avg)"
        );
    }

    #[tokio::test]
    async fn test_analyze_dataset_vocabulary_reflects_actual_text_content() {
        let temp = tempdir().expect("tempdir");
        let dataset_dir = temp.path().join("dataset");
        fs::create_dir_all(&dataset_dir).expect("create dir");
        write_test_wav(&dataset_dir.join("a.wav"), 16000, 0.1, 440.0);
        fs::write(dataset_dir.join("a.txt"), "alpha beta gamma alpha").unwrap();

        let global = quiet_global();
        let report_path = temp.path().join("report.md");
        analyze_dataset(&dataset_dir, Some(&report_path), true, &global)
            .await
            .expect("analyze should succeed");

        let report = fs::read_to_string(&report_path).expect("read report");
        // 3 unique words (alpha, beta, gamma) despite 4 tokens.
        assert!(
            report.contains("Vocabulary size: 3 unique words"),
            "report should reflect the real 3-word vocabulary, got:\n{report}"
        );
    }

    #[tokio::test]
    async fn test_analyze_dataset_errors_on_missing_path() {
        let temp = tempdir().expect("tempdir");
        let missing = temp.path().join("does-not-exist");
        let global = quiet_global();
        let result = analyze_dataset(&missing, None, false, &global).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_scan_audio_files_empty_dir() {
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let count = scan_audio_files(temp_dir.path()).expect("Failed to scan audio files");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_scan_audio_files_with_files() {
        let temp_dir = tempdir().expect("Failed to create temp directory");

        // Create some test files
        fs::write(temp_dir.path().join("test1.wav"), b"test").expect("Failed to write test1.wav");
        fs::write(temp_dir.path().join("test2.flac"), b"test").expect("Failed to write test2.flac");
        fs::write(temp_dir.path().join("test3.mp3"), b"test").expect("Failed to write test3.mp3");
        fs::write(temp_dir.path().join("test4.txt"), b"test").expect("Failed to write test4.txt"); // Should be ignored

        let count = scan_audio_files(temp_dir.path()).expect("Failed to scan audio files");
        assert_eq!(count, 3);
    }

    #[test]
    fn test_scan_text_files() {
        let temp_dir = tempdir().expect("Failed to create temp directory");

        // Create some test files
        fs::write(temp_dir.path().join("test1.txt"), b"test").expect("Failed to write test1.txt");
        fs::write(temp_dir.path().join("test2.csv"), b"test").expect("Failed to write test2.csv");
        fs::write(temp_dir.path().join("test3.json"), b"test").expect("Failed to write test3.json");
        fs::write(temp_dir.path().join("test4.wav"), b"test").expect("Failed to write test4.wav"); // Should be ignored

        let count = scan_text_files(temp_dir.path()).expect("Failed to scan text files");
        assert_eq!(count, 3);
    }
}
