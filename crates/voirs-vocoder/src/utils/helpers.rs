//! Helper utilities for common vocoder operations.
//!
//! This module provides ergonomic helper functions for common tasks:
//! - Audio buffer manipulation and conversion
//! - Mel spectrogram validation and preprocessing
//! - Batch processing utilities
//! - Quality assessment shortcuts
//!
//! # Examples
//!
//! ```rust,ignore
//! use voirs_vocoder::utils::helpers::*;
//!
//! // Validate mel spectrogram before processing
//! if let Err(e) = validate_mel_spectrogram(&mel) {
//!     eprintln!("Invalid mel: {}", e);
//!     return;
//! }
//!
//! // Process with automatic quality checks
//! let audio = process_with_quality_check(vocoder, mel, config).await?;
//! ```

use crate::{AudioBuffer, MelSpectrogram, Result, SynthesisConfig, Vocoder, VocoderError};
use std::time::{Duration, Instant};

/// Validation result for mel spectrograms
#[derive(Debug, Clone)]
pub struct MelValidation {
    /// Whether the mel spectrogram is valid
    pub is_valid: bool,
    /// List of validation warnings
    pub warnings: Vec<String>,
    /// List of validation errors
    pub errors: Vec<String>,
}

impl MelValidation {
    /// Create a new validation result
    pub fn new() -> Self {
        Self {
            is_valid: true,
            warnings: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Add a warning message
    pub fn warn(&mut self, message: impl Into<String>) {
        self.warnings.push(message.into());
    }

    /// Add an error message and mark as invalid
    pub fn error(&mut self, message: impl Into<String>) {
        self.errors.push(message.into());
        self.is_valid = false;
    }

    /// Check if there are any issues (warnings or errors)
    pub fn has_issues(&self) -> bool {
        !self.warnings.is_empty() || !self.errors.is_empty()
    }
}

impl Default for MelValidation {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate a mel spectrogram for common issues
///
/// Checks for:
/// - Empty data
/// - Unusual dimensions
/// - NaN or infinite values
/// - Excessive dynamic range
/// - Silent frames
///
/// # Arguments
/// * `mel` - The mel spectrogram to validate
///
/// # Returns
/// A validation result with any warnings or errors found
pub fn validate_mel_spectrogram(mel: &MelSpectrogram) -> MelValidation {
    let mut validation = MelValidation::new();

    // Check for empty data
    if mel.data.is_empty() {
        validation.error("Mel spectrogram has no data");
        return validation;
    }

    let num_mels = mel.data.len();
    let num_frames = mel.data[0].len();

    // Validate dimensions
    if num_mels < 40 || num_mels > 160 {
        validation.warn(format!(
            "Unusual number of mel bands: {} (typical range: 40-160)",
            num_mels
        ));
    }

    if num_frames == 0 {
        validation.error("Mel spectrogram has zero frames");
        return validation;
    }

    if num_frames < 10 {
        validation.warn(format!(
            "Very short mel spectrogram: {} frames (may cause artifacts)",
            num_frames
        ));
    }

    // Check for consistent frame lengths
    for (i, mel_band) in mel.data.iter().enumerate() {
        if mel_band.len() != num_frames {
            validation.error(format!(
                "Inconsistent frame length at mel band {}: expected {}, got {}",
                i,
                num_frames,
                mel_band.len()
            ));
        }
    }

    // Statistical validation
    let mut has_nan = false;
    let mut has_inf = false;
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    let mut silent_frames = 0;

    for mel_band in mel.data.iter() {
        for &value in mel_band.iter() {
            if value.is_nan() {
                has_nan = true;
            } else if value.is_infinite() {
                has_inf = true;
            } else {
                min_val = min_val.min(value);
                max_val = max_val.max(value);
            }
        }
    }

    // Check each frame for silence
    for frame_idx in 0..num_frames {
        let mut frame_energy = 0.0;
        for mel_band in mel.data.iter() {
            frame_energy += mel_band[frame_idx].abs();
        }
        if frame_energy < 1e-6 {
            silent_frames += 1;
        }
    }

    if has_nan {
        validation.error("Mel spectrogram contains NaN values");
    }

    if has_inf {
        validation.error("Mel spectrogram contains infinite values");
    }

    // Dynamic range check
    if !has_nan && !has_inf && min_val.is_finite() && max_val.is_finite() {
        let dynamic_range = max_val - min_val;
        if dynamic_range > 200.0 {
            validation.warn(format!(
                "Very large dynamic range: {:.1} dB (may cause clipping)",
                dynamic_range
            ));
        } else if dynamic_range < 20.0 {
            validation.warn(format!(
                "Very small dynamic range: {:.1} dB (may sound flat)",
                dynamic_range
            ));
        }
    }

    // Silent frame check
    let silent_ratio = silent_frames as f32 / num_frames as f32;
    if silent_ratio > 0.5 {
        validation.warn(format!(
            "{:.1}% of frames are silent (may indicate preprocessing issues)",
            silent_ratio * 100.0
        ));
    }

    validation
}

/// Process timing information for performance analysis
#[derive(Debug, Clone)]
pub struct ProcessingTiming {
    /// Total processing duration
    pub total_duration: Duration,
    /// Audio duration that was generated
    pub audio_duration: Duration,
    /// Real-time factor (processing time / audio duration)
    pub real_time_factor: f64,
    /// Processing throughput in samples per second
    pub throughput_samples_per_sec: f64,
}

impl ProcessingTiming {
    /// Create timing information from measurements
    pub fn new(processing_duration: Duration, audio: &AudioBuffer) -> Self {
        let audio_duration = Duration::from_secs_f64(audio.duration() as f64);
        let real_time_factor = if audio_duration.as_secs_f64() > 0.0 {
            processing_duration.as_secs_f64() / audio_duration.as_secs_f64()
        } else {
            f64::INFINITY
        };

        let throughput_samples_per_sec = if processing_duration.as_secs_f64() > 0.0 {
            audio.len() as f64 / processing_duration.as_secs_f64()
        } else {
            0.0
        };

        Self {
            total_duration: processing_duration,
            audio_duration,
            real_time_factor,
            throughput_samples_per_sec,
        }
    }

    /// Check if processing is faster than real-time
    pub fn is_realtime(&self) -> bool {
        self.real_time_factor < 1.0
    }

    /// Get a human-readable summary of the timing
    pub fn summary(&self) -> String {
        format!(
            "Processed {:.2}s audio in {:.2}s (RTF: {:.3}x, {} real-time, {:.0} samples/sec)",
            self.audio_duration.as_secs_f64(),
            self.total_duration.as_secs_f64(),
            self.real_time_factor,
            if self.is_realtime() {
                "faster than"
            } else {
                "slower than"
            },
            self.throughput_samples_per_sec
        )
    }
}

/// Process a mel spectrogram with automatic timing measurement
///
/// This is a convenience wrapper that times the vocoding operation
/// and returns both the result and timing information.
///
/// # Arguments
/// * `vocoder` - The vocoder to use
/// * `mel` - The mel spectrogram to process
/// * `config` - Optional synthesis configuration
///
/// # Returns
/// A tuple of (audio_buffer, timing_info) or an error
pub async fn vocode_with_timing(
    vocoder: &dyn Vocoder,
    mel: &MelSpectrogram,
    config: Option<&SynthesisConfig>,
) -> Result<(AudioBuffer, ProcessingTiming)> {
    let start = Instant::now();
    let audio = vocoder.vocode(mel, config).await?;
    let duration = start.elapsed();
    let timing = ProcessingTiming::new(duration, &audio);
    Ok((audio, timing))
}

/// Batch process multiple mel spectrograms with progress tracking
///
/// # Arguments
/// * `vocoder` - The vocoder to use
/// * `mels` - Iterator of mel spectrograms to process
/// * `config` - Optional synthesis configuration
/// * `progress_callback` - Optional callback called after each mel is processed
///
/// # Returns
/// A vector of audio buffers or an error
pub async fn batch_vocode<I, F>(
    vocoder: &dyn Vocoder,
    mels: I,
    config: Option<&SynthesisConfig>,
    mut progress_callback: Option<F>,
) -> Result<Vec<AudioBuffer>>
where
    I: IntoIterator<Item = MelSpectrogram>,
    F: FnMut(usize, usize),
{
    let mels: Vec<_> = mels.into_iter().collect();
    let total = mels.len();
    let mut results = Vec::with_capacity(total);

    for (i, mel) in mels.into_iter().enumerate() {
        let audio = vocoder.vocode(&mel, config).await?;
        results.push(audio);

        if let Some(ref mut callback) = progress_callback {
            callback(i + 1, total);
        }
    }

    Ok(results)
}

/// Create a mel spectrogram with validation checks
///
/// This is a convenience wrapper around MelSpectrogram::new that performs
/// validation and returns helpful error messages if issues are found.
///
/// # Arguments
/// * `data` - The mel spectrogram data
/// * `sample_rate` - Sample rate of the original audio
/// * `hop_length` - Hop length used in STFT
///
/// # Returns
/// A validated mel spectrogram or an error
pub fn create_mel_spectrogram_validated(
    data: Vec<Vec<f32>>,
    sample_rate: u32,
    hop_length: u32,
) -> Result<MelSpectrogram> {
    // Basic validation
    if data.is_empty() {
        return Err(VocoderError::InvalidMelSpectrogram(
            "Mel spectrogram data is empty".to_string(),
        ));
    }

    if sample_rate == 0 {
        return Err(VocoderError::InvalidMelSpectrogram(
            "Sample rate cannot be zero".to_string(),
        ));
    }

    if hop_length == 0 {
        return Err(VocoderError::InvalidMelSpectrogram(
            "Hop length cannot be zero".to_string(),
        ));
    }

    let mel = MelSpectrogram::new(data, sample_rate, hop_length);
    let validation = validate_mel_spectrogram(&mel);

    if !validation.is_valid {
        return Err(VocoderError::InvalidMelSpectrogram(format!(
            "Validation failed: {}",
            validation.errors.join(", ")
        )));
    }

    // Log warnings if any
    if !validation.warnings.is_empty() {
        tracing::warn!(
            "Mel spectrogram validation warnings: {}",
            validation.warnings.join(", ")
        );
    }

    Ok(mel)
}

/// Normalize mel spectrogram to a target range
///
/// Useful for ensuring consistent input ranges to vocoders
///
/// # Arguments
/// * `mel` - The mel spectrogram to normalize
/// * `target_min` - Target minimum value
/// * `target_max` - Target maximum value
///
/// # Returns
/// A new normalized mel spectrogram
pub fn normalize_mel_spectrogram(
    mel: &MelSpectrogram,
    target_min: f32,
    target_max: f32,
) -> MelSpectrogram {
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;

    // Find current range
    for mel_band in mel.data.iter() {
        for &value in mel_band.iter() {
            if value.is_finite() {
                min_val = min_val.min(value);
                max_val = max_val.max(value);
            }
        }
    }

    // Avoid division by zero
    if (max_val - min_val).abs() < 1e-8 {
        return MelSpectrogram::new(mel.data.clone(), mel.sample_rate, mel.hop_length);
    }

    // Normalize data
    let scale = (target_max - target_min) / (max_val - min_val);
    let normalized_data: Vec<Vec<f32>> = mel
        .data
        .iter()
        .map(|mel_band| {
            mel_band
                .iter()
                .map(|&value| {
                    if value.is_finite() {
                        (value - min_val) * scale + target_min
                    } else {
                        target_min
                    }
                })
                .collect()
        })
        .collect();

    MelSpectrogram::new(normalized_data, mel.sample_rate, mel.hop_length)
}

/// Concatenate multiple audio buffers with optional crossfade
///
/// # Arguments
/// * `buffers` - Audio buffers to concatenate
/// * `crossfade_samples` - Number of samples to crossfade between buffers (0 for no crossfade)
///
/// # Returns
/// A single concatenated audio buffer
pub fn concatenate_audio_buffers(
    buffers: &[AudioBuffer],
    crossfade_samples: usize,
) -> Result<AudioBuffer> {
    if buffers.is_empty() {
        return Err(VocoderError::InputError(
            "Cannot concatenate empty buffer list".to_string(),
        ));
    }

    if buffers.len() == 1 {
        return Ok(buffers[0].clone());
    }

    // Validate all buffers have same sample rate and channels
    let sample_rate = buffers[0].sample_rate();
    let channels = buffers[0].channels();

    for (i, buffer) in buffers.iter().enumerate().skip(1) {
        if buffer.sample_rate() != sample_rate {
            return Err(VocoderError::InputError(format!(
                "Sample rate mismatch at buffer {}: expected {}, got {}",
                i,
                sample_rate,
                buffer.sample_rate()
            )));
        }
        if buffer.channels() != channels {
            return Err(VocoderError::InputError(format!(
                "Channel count mismatch at buffer {}: expected {}, got {}",
                i,
                channels,
                buffer.channels()
            )));
        }
    }

    let mut result = Vec::new();

    for (i, buffer) in buffers.iter().enumerate() {
        let samples = buffer.samples();

        if i == 0 {
            // First buffer: add everything except crossfade region at end
            if crossfade_samples > 0 && samples.len() > crossfade_samples {
                result.extend_from_slice(&samples[..samples.len() - crossfade_samples]);
            } else {
                result.extend_from_slice(samples);
            }
        } else if i == buffers.len() - 1 {
            // Last buffer: apply crossfade at start, then add rest
            if crossfade_samples > 0 && samples.len() > crossfade_samples {
                let prev_tail =
                    &buffers[i - 1].samples()[buffers[i - 1].len() - crossfade_samples..];
                let curr_head = &samples[..crossfade_samples];

                // Apply linear crossfade with center-point sampling
                for j in 0..crossfade_samples {
                    let t = (j as f32 + 0.5) / crossfade_samples as f32;
                    let fade_out = 1.0 - t;
                    let fade_in = t;
                    result.push(prev_tail[j] * fade_out + curr_head[j] * fade_in);
                }

                // Add remaining samples
                result.extend_from_slice(&samples[crossfade_samples..]);
            } else {
                result.extend_from_slice(samples);
            }
        } else {
            // Middle buffers: crossfade at start, keep middle, prepare for crossfade at end
            if crossfade_samples > 0 && samples.len() > 2 * crossfade_samples {
                let prev_tail =
                    &buffers[i - 1].samples()[buffers[i - 1].len() - crossfade_samples..];
                let curr_head = &samples[..crossfade_samples];

                // Apply linear crossfade at start with center-point sampling
                for j in 0..crossfade_samples {
                    let t = (j as f32 + 0.5) / crossfade_samples as f32;
                    let fade_out = 1.0 - t;
                    let fade_in = t;
                    result.push(prev_tail[j] * fade_out + curr_head[j] * fade_in);
                }

                // Add middle section
                result.extend_from_slice(
                    &samples[crossfade_samples..samples.len() - crossfade_samples],
                );
            } else {
                result.extend_from_slice(samples);
            }
        }
    }

    Ok(AudioBuffer::new(result, sample_rate, channels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_validation_empty() {
        let mel = MelSpectrogram::new(vec![], 22050, 256);
        let validation = validate_mel_spectrogram(&mel);
        assert!(!validation.is_valid);
        assert!(validation.errors.iter().any(|e| e.contains("no data")));
    }

    #[test]
    fn test_mel_validation_valid() {
        let data = vec![vec![1.0; 100]; 80];
        let mel = MelSpectrogram::new(data, 22050, 256);
        let validation = validate_mel_spectrogram(&mel);
        assert!(validation.is_valid);
    }

    #[test]
    fn test_mel_validation_nan() {
        let mut data = vec![vec![1.0; 100]; 80];
        data[0][0] = f32::NAN;
        let mel = MelSpectrogram::new(data, 22050, 256);
        let validation = validate_mel_spectrogram(&mel);
        assert!(!validation.is_valid);
        assert!(validation.errors.iter().any(|e| e.contains("NaN")));
    }

    #[test]
    fn test_normalize_mel_spectrogram() {
        let data = vec![vec![0.0, 1.0, 2.0, 3.0]; 80];
        let mel = MelSpectrogram::new(data, 22050, 256);
        let normalized = normalize_mel_spectrogram(&mel, 0.0, 1.0);

        for mel_band in normalized.data.iter() {
            assert_eq!(mel_band[0], 0.0);
            assert_eq!(mel_band[3], 1.0);
        }
    }

    #[test]
    fn test_processing_timing() {
        let audio = AudioBuffer::new(vec![0.0; 22050], 22050, 1); // 1 second of audio
        let duration = Duration::from_millis(100); // Processed in 100ms
        let timing = ProcessingTiming::new(duration, &audio);

        assert!(timing.is_realtime());
        assert!(timing.real_time_factor < 0.2); // Much faster than real-time
    }

    #[test]
    fn test_concatenate_audio_buffers_no_crossfade() {
        let buf1 = AudioBuffer::new(vec![1.0, 2.0, 3.0], 22050, 1);
        let buf2 = AudioBuffer::new(vec![4.0, 5.0, 6.0], 22050, 1);
        let result = concatenate_audio_buffers(&[buf1, buf2], 0).unwrap();

        assert_eq!(result.len(), 6);
        assert_eq!(result.samples(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_concatenate_audio_buffers_with_crossfade() {
        let buf1 = AudioBuffer::new(vec![1.0, 1.0, 1.0, 1.0], 22050, 1);
        let buf2 = AudioBuffer::new(vec![2.0, 2.0, 2.0, 2.0], 22050, 1);
        let result = concatenate_audio_buffers(&[buf1, buf2], 2).unwrap();

        // Should have: [1.0, 1.0] + crossfade + [2.0, 2.0]
        assert_eq!(result.len(), 6);
        // First part from buf1
        assert_eq!(result.samples()[0], 1.0);
        assert_eq!(result.samples()[1], 1.0);
        // Crossfaded part should be between 1.0 and 2.0
        assert!(
            result.samples()[2] > 1.0 && result.samples()[2] < 2.0,
            "Expected value between 1.0 and 2.0, got {}",
            result.samples()[2]
        );
        assert!(
            result.samples()[3] > 1.0 && result.samples()[3] < 2.0,
            "Expected value between 1.0 and 2.0, got {}",
            result.samples()[3]
        );
        // Last part from buf2
        assert_eq!(result.samples()[4], 2.0);
        assert_eq!(result.samples()[5], 2.0);
    }

    #[test]
    fn test_concatenate_audio_buffers_sample_rate_mismatch() {
        let buf1 = AudioBuffer::new(vec![1.0, 2.0], 22050, 1);
        let buf2 = AudioBuffer::new(vec![3.0, 4.0], 44100, 1);
        let result = concatenate_audio_buffers(&[buf1, buf2], 0);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Sample rate mismatch"));
    }
}
