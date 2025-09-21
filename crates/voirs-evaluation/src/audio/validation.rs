//! Audio file validation and corruption detection utilities
//!
//! This module provides comprehensive validation for audio files to detect
//! and gracefully handle corrupted or invalid audio data.

use crate::{error_enhancement, EvaluationError};
use std::f32;
use voirs_sdk::AudioBuffer;

/// Audio validation configuration
#[derive(Debug, Clone)]
pub struct AudioValidationConfig {
    /// Minimum audio duration in seconds
    pub min_duration_seconds: f32,
    /// Maximum audio duration in seconds  
    pub max_duration_seconds: f32,
    /// Minimum sample rate
    pub min_sample_rate: u32,
    /// Maximum sample rate
    pub max_sample_rate: u32,
    /// Maximum allowed consecutive silent samples
    pub max_silence_samples: usize,
    /// Maximum allowed clipping percentage (0.0-1.0)
    pub max_clipping_percentage: f32,
    /// Minimum dynamic range in dB
    pub min_dynamic_range_db: f32,
    /// Check for DC offset
    pub check_dc_offset: bool,
    /// Maximum allowed DC offset
    pub max_dc_offset: f32,
}

impl Default for AudioValidationConfig {
    fn default() -> Self {
        Self {
            min_duration_seconds: 0.1,     // 100ms minimum
            max_duration_seconds: 3600.0,  // 1 hour maximum
            min_sample_rate: 8000,         // 8kHz minimum
            max_sample_rate: 192_000,      // 192kHz maximum
            max_silence_samples: 8000,     // 1 second at 8kHz
            max_clipping_percentage: 0.05, // 5% clipping allowed
            min_dynamic_range_db: 6.0,     // 6dB minimum dynamic range
            check_dc_offset: true,
            max_dc_offset: 0.1, // 10% of full scale
        }
    }
}

/// Audio validation result
#[derive(Debug, Clone)]
pub struct AudioValidationResult {
    /// Whether the audio is valid
    pub is_valid: bool,
    /// List of validation warnings (non-fatal issues)
    pub warnings: Vec<String>,
    /// List of validation errors (fatal issues)
    pub errors: Vec<String>,
    /// Audio quality metrics
    pub quality_metrics: AudioQualityMetrics,
}

/// Audio quality metrics from validation
#[derive(Debug, Clone)]
pub struct AudioQualityMetrics {
    /// Audio duration in seconds
    pub duration_seconds: f32,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
    /// RMS level
    pub rms_level: f32,
    /// Peak level
    pub peak_level: f32,
    /// Dynamic range in dB
    pub dynamic_range_db: f32,
    /// Clipping percentage (0.0-1.0)
    pub clipping_percentage: f32,
    /// DC offset
    pub dc_offset: f32,
    /// Silent samples count
    pub silent_samples: usize,
    /// Signal-to-noise ratio estimate
    pub snr_estimate_db: Option<f32>,
}

/// Audio file validator
pub struct AudioValidator {
    config: AudioValidationConfig,
}

impl AudioValidator {
    /// Create a new audio validator with default configuration
    pub fn new() -> Self {
        Self {
            config: AudioValidationConfig::default(),
        }
    }

    /// Create a new audio validator with custom configuration
    pub fn with_config(config: AudioValidationConfig) -> Self {
        Self { config }
    }

    /// Validate an audio buffer
    pub fn validate(&self, audio: &AudioBuffer) -> AudioValidationResult {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();

        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(audio);

        // Validate duration
        if quality_metrics.duration_seconds < self.config.min_duration_seconds {
            errors.push(format!(
                "Audio duration {:.3}s is below minimum {:.3}s",
                quality_metrics.duration_seconds, self.config.min_duration_seconds
            ));
        }

        if quality_metrics.duration_seconds > self.config.max_duration_seconds {
            warnings.push(format!(
                "Audio duration {:.1}s exceeds recommended maximum {:.1}s",
                quality_metrics.duration_seconds, self.config.max_duration_seconds
            ));
        }

        // Validate sample rate
        if quality_metrics.sample_rate < self.config.min_sample_rate {
            errors.push(format!(
                "Sample rate {}Hz is below minimum {}Hz",
                quality_metrics.sample_rate, self.config.min_sample_rate
            ));
        }

        if quality_metrics.sample_rate > self.config.max_sample_rate {
            warnings.push(format!(
                "Sample rate {}Hz exceeds recommended maximum {}Hz",
                quality_metrics.sample_rate, self.config.max_sample_rate
            ));
        }

        // Check for excessive silence
        if quality_metrics.silent_samples > self.config.max_silence_samples {
            warnings.push(format!(
                "Audio contains {} consecutive silent samples (max {})",
                quality_metrics.silent_samples, self.config.max_silence_samples
            ));
        }

        // Check for clipping
        if quality_metrics.clipping_percentage > self.config.max_clipping_percentage {
            warnings.push(format!(
                "Audio has {:.1}% clipping (max {:.1}%)",
                quality_metrics.clipping_percentage * 100.0,
                self.config.max_clipping_percentage * 100.0
            ));
        }

        // Check dynamic range
        if quality_metrics.dynamic_range_db < self.config.min_dynamic_range_db {
            warnings.push(format!(
                "Dynamic range {:.1}dB is below recommended minimum {:.1}dB",
                quality_metrics.dynamic_range_db, self.config.min_dynamic_range_db
            ));
        }

        // Check DC offset
        if self.config.check_dc_offset
            && quality_metrics.dc_offset.abs() > self.config.max_dc_offset
        {
            warnings.push(format!(
                "DC offset {:.3} exceeds threshold {:.3}",
                quality_metrics.dc_offset, self.config.max_dc_offset
            ));
        }

        // Check for NaN or infinite values
        let samples = audio.samples();
        if samples.iter().any(|&x| !x.is_finite()) {
            errors.push("Audio contains NaN or infinite values".to_string());
        }

        // Check for completely silent audio
        if quality_metrics.peak_level == 0.0 {
            errors.push("Audio appears to be completely silent".to_string());
        }

        AudioValidationResult {
            is_valid: errors.is_empty(),
            warnings,
            errors,
            quality_metrics,
        }
    }

    /// Calculate audio quality metrics
    fn calculate_quality_metrics(&self, audio: &AudioBuffer) -> AudioQualityMetrics {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();
        let channels = audio.channels();

        let duration_seconds = samples.len() as f32 / (sample_rate * channels) as f32;

        // Calculate RMS and peak levels
        let sum_squares: f64 = samples.iter().map(|&x| (x as f64).powi(2)).sum();
        let rms_level = (sum_squares / samples.len() as f64).sqrt() as f32;
        let peak_level = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

        // Calculate dynamic range
        let dynamic_range_db = if rms_level > 0.0 && peak_level > 0.0 {
            20.0 * (peak_level / rms_level).log10()
        } else {
            0.0
        };

        // Calculate clipping percentage
        let clipping_threshold = 0.99;
        let clipped_samples = samples
            .iter()
            .filter(|&&x| x.abs() > clipping_threshold)
            .count();
        let clipping_percentage = clipped_samples as f32 / samples.len() as f32;

        // Calculate DC offset
        let dc_offset = samples.iter().sum::<f32>() / samples.len() as f32;

        // Count consecutive silent samples
        let silent_threshold = 0.001; // -60dB
        let mut max_silent_samples = 0;
        let mut current_silent_samples = 0;

        for &sample in samples {
            if sample.abs() < silent_threshold {
                current_silent_samples += 1;
                max_silent_samples = max_silent_samples.max(current_silent_samples);
            } else {
                current_silent_samples = 0;
            }
        }

        // Estimate SNR (simplified calculation)
        let snr_estimate_db = if rms_level > 0.0 {
            // Estimate noise floor as bottom 10% of energy
            let mut sorted_squares: Vec<f32> = samples.iter().map(|&x| x * x).collect();
            sorted_squares.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let noise_floor_idx = (sorted_squares.len() as f32 * 0.1) as usize;
            let noise_rms = (sorted_squares[..noise_floor_idx].iter().sum::<f32>()
                / noise_floor_idx as f32)
                .sqrt();

            if noise_rms > 0.0 {
                Some(20.0 * (rms_level / noise_rms).log10())
            } else {
                None
            }
        } else {
            None
        };

        AudioQualityMetrics {
            duration_seconds,
            sample_rate,
            channels,
            rms_level,
            peak_level,
            dynamic_range_db,
            clipping_percentage,
            dc_offset,
            silent_samples: max_silent_samples,
            snr_estimate_db,
        }
    }

    /// Attempt to repair common audio issues
    pub fn attempt_repair(&self, audio: &AudioBuffer) -> Result<AudioBuffer, EvaluationError> {
        let samples = audio.samples();
        let mut repaired_samples = samples.to_vec();

        // Remove NaN and infinite values
        for sample in &mut repaired_samples {
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }

        // Clamp extreme values
        for sample in &mut repaired_samples {
            *sample = sample.clamp(-1.0, 1.0);
        }

        // Remove DC offset if significant
        let dc_offset = repaired_samples.iter().sum::<f32>() / repaired_samples.len() as f32;
        if dc_offset.abs() > self.config.max_dc_offset {
            for sample in &mut repaired_samples {
                *sample -= dc_offset;
            }
        }

        // Create repaired audio buffer based on channel configuration
        let repaired_audio = if audio.channels() == 1 {
            AudioBuffer::mono(repaired_samples, audio.sample_rate())
        } else {
            AudioBuffer::stereo(repaired_samples, audio.sample_rate())
        };

        Ok(repaired_audio)
    }
}

impl Default for AudioValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate audio buffer and return detailed error on failure
pub fn validate_audio_buffer(
    audio: &AudioBuffer,
    component: &str,
) -> Result<AudioValidationResult, EvaluationError> {
    let validator = AudioValidator::new();
    let result = validator.validate(audio);

    if !result.is_valid {
        let error_message =
            error_enhancement::ErrorMessageBuilder::new(&format!("{} audio validation", component))
                .actual(&format!(
                    "audio failed validation with {} errors",
                    result.errors.len()
                ))
                .suggestions(&[
                    "Check if the audio file is corrupted or truncated",
                    "Verify the audio format is supported",
                    "Try using the audio repair functionality",
                    "Check audio loading pipeline for issues",
                ])
                .build();

        return Err(EvaluationError::InvalidInput {
            message: format!(
                "{}\n\nValidation errors:\n{}",
                error_message,
                result.errors.join("\n")
            ),
        });
    }

    Ok(result)
}

/// Attempt to load and validate audio with automatic repair
pub fn load_and_validate_audio(
    audio: AudioBuffer,
    component: &str,
) -> Result<AudioBuffer, EvaluationError> {
    let validator = AudioValidator::new();
    let validation_result = validator.validate(&audio);

    if validation_result.is_valid {
        return Ok(audio);
    }

    // If validation failed, attempt repair
    match validator.attempt_repair(&audio) {
        Ok(repaired_audio) => {
            let repair_validation = validator.validate(&repaired_audio);
            if repair_validation.is_valid {
                Ok(repaired_audio)
            } else {
                Err(EvaluationError::AudioProcessingError {
                    message: error_enhancement::ErrorMessageBuilder::new(&format!(
                        "{} audio repair",
                        component
                    ))
                    .actual("audio repair failed to resolve all issues")
                    .suggestions(&[
                        "Audio file may be severely corrupted",
                        "Try re-encoding the audio file",
                        "Check if the source audio is valid",
                        "Use a different audio file for evaluation",
                    ])
                    .build(),
                    source: None,
                })
            }
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_validator_valid_audio() {
        let samples = (0..16000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let audio = AudioBuffer::mono(samples, 16000);

        let validator = AudioValidator::new();
        let result = validator.validate(&audio);

        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_audio_validator_silent_audio() {
        let samples = vec![0.0; 16000];
        let audio = AudioBuffer::mono(samples, 16000);

        let validator = AudioValidator::new();
        let result = validator.validate(&audio);

        assert!(!result.is_valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("completely silent")));
    }

    #[test]
    fn test_audio_validator_clipping() {
        let mut samples = vec![0.5; 16000];
        // Add some clipped samples
        for i in 0..1000 {
            samples[i] = 1.0; // Clipped
        }
        let audio = AudioBuffer::mono(samples, 16000);

        let validator = AudioValidator::new();
        let result = validator.validate(&audio);

        assert!(result.warnings.iter().any(|w| w.contains("clipping")));
    }

    #[test]
    fn test_audio_repair_nan_values() {
        let mut samples = vec![0.5; 16000];
        samples[100] = f32::NAN;
        samples[200] = f32::INFINITY;
        let audio = AudioBuffer::mono(samples, 16000);

        let validator = AudioValidator::new();
        let repaired = validator.attempt_repair(&audio).unwrap();

        assert!(repaired.samples().iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_load_and_validate_audio() {
        let samples = (0..16000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let audio = AudioBuffer::mono(samples, 16000);

        let result = load_and_validate_audio(audio, "test");
        assert!(result.is_ok());
    }
}
