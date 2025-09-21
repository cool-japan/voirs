//! Automatic audio format conversion utilities
//!
//! This module provides automatic conversion between different audio formats,
//! sample rates, and channel configurations to ensure compatibility between
//! reference and test audio files.

use super::{conversion::*, validation::*, AudioIoResult};
use crate::{error_enhancement, EvaluationError};
use voirs_sdk::AudioBuffer;

/// Automatic conversion configuration
#[derive(Debug, Clone)]
pub struct AutoConversionConfig {
    /// Target sample rate (None = use highest available)
    pub target_sample_rate: Option<u32>,
    /// Target channel count (None = use mono if available)
    pub target_channels: Option<u32>,
    /// Whether to normalize audio levels
    pub normalize_levels: bool,
    /// Whether to match audio lengths
    pub match_lengths: bool,
    /// Length matching strategy
    pub length_strategy: LengthMatchingStrategy,
    /// Whether to apply DC offset removal
    pub remove_dc_offset: bool,
    /// Quality level for resampling (0-10)
    pub resample_quality: u8,
}

impl Default for AutoConversionConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: None,
            target_channels: Some(1), // Default to mono
            normalize_levels: true,
            match_lengths: true,
            length_strategy: LengthMatchingStrategy::TrimToShortest,
            remove_dc_offset: true,
            resample_quality: 7, // High quality
        }
    }
}

/// Strategy for matching audio lengths
#[derive(Debug, Clone, Copy)]
pub enum LengthMatchingStrategy {
    /// Trim both to the shortest length
    TrimToShortest,
    /// Pad shorter with silence to match longer
    PadToLongest,
    /// Repeat shorter until it matches longer
    RepeatToLongest,
    /// Keep original lengths (no modification)
    KeepOriginal,
}

/// Result of automatic conversion
#[derive(Debug)]
pub struct ConversionResult {
    /// Converted reference audio
    pub reference: AudioBuffer,
    /// Converted test/degraded audio  
    pub test: AudioBuffer,
    /// Applied conversions summary
    pub conversions_applied: Vec<String>,
    /// Validation results for converted audio
    pub validation_results: (AudioValidationResult, AudioValidationResult),
}

/// Automatic audio converter
pub struct AutoConverter {
    config: AutoConversionConfig,
    validator: AudioValidator,
}

impl AutoConverter {
    /// Create a new auto converter with default configuration
    pub fn new() -> Self {
        Self {
            config: AutoConversionConfig::default(),
            validator: AudioValidator::new(),
        }
    }

    /// Create a new auto converter with custom configuration
    pub fn with_config(config: AutoConversionConfig) -> Self {
        Self {
            config,
            validator: AudioValidator::new(),
        }
    }

    /// Automatically convert two audio buffers to be compatible
    pub fn convert_for_compatibility(
        &self,
        reference: AudioBuffer,
        test: AudioBuffer,
        component: &str,
    ) -> Result<ConversionResult, EvaluationError> {
        let mut conversions_applied = Vec::new();
        let mut ref_audio = reference;
        let mut test_audio = test;

        // Step 1: Validate and repair if needed
        ref_audio = load_and_validate_audio(ref_audio, &format!("{} reference", component))?;
        test_audio = load_and_validate_audio(test_audio, &format!("{} test", component))?;

        // Step 2: Determine target sample rate
        let target_sample_rate = self.determine_target_sample_rate(&ref_audio, &test_audio);

        // Step 3: Determine target channels
        let target_channels = self.determine_target_channels(&ref_audio, &test_audio);

        // Step 4: Apply sample rate conversion if needed
        if ref_audio.sample_rate() != target_sample_rate {
            ref_audio = self.convert_sample_rate(ref_audio, target_sample_rate)?;
            conversions_applied.push(format!(
                "Reference sample rate converted from {} Hz to {} Hz",
                ref_audio.sample_rate(),
                target_sample_rate
            ));
        }

        if test_audio.sample_rate() != target_sample_rate {
            test_audio = self.convert_sample_rate(test_audio, target_sample_rate)?;
            conversions_applied.push(format!(
                "Test sample rate converted from {} Hz to {} Hz",
                test_audio.sample_rate(),
                target_sample_rate
            ));
        }

        // Step 5: Apply channel conversion if needed
        if ref_audio.channels() != target_channels {
            ref_audio = self.convert_channels(ref_audio, target_channels)?;
            conversions_applied.push(format!(
                "Reference channels converted from {} to {}",
                ref_audio.channels(),
                target_channels
            ));
        }

        if test_audio.channels() != target_channels {
            test_audio = self.convert_channels(test_audio, target_channels)?;
            conversions_applied.push(format!(
                "Test channels converted from {} to {}",
                test_audio.channels(),
                target_channels
            ));
        }

        // Step 6: Apply length matching if needed
        if self.config.match_lengths {
            let (new_ref, new_test) = self.match_audio_lengths(ref_audio, test_audio)?;
            ref_audio = new_ref;
            test_audio = new_test;
            conversions_applied.push(format!(
                "Audio lengths matched using {:?} strategy",
                self.config.length_strategy
            ));
        }

        // Step 7: Apply post-processing
        if self.config.remove_dc_offset {
            ref_audio = self.remove_dc_offset(ref_audio)?;
            test_audio = self.remove_dc_offset(test_audio)?;
            conversions_applied.push("DC offset removed".to_string());
        }

        if self.config.normalize_levels {
            ref_audio = self.normalize_audio(ref_audio)?;
            test_audio = self.normalize_audio(test_audio)?;
            conversions_applied.push("Audio levels normalized".to_string());
        }

        // Step 8: Final validation
        let ref_validation = self.validator.validate(&ref_audio);
        let test_validation = self.validator.validate(&test_audio);

        if !ref_validation.is_valid || !test_validation.is_valid {
            return Err(EvaluationError::AudioProcessingError {
                message: error_enhancement::ErrorMessageBuilder::new(&format!(
                    "{} auto conversion",
                    component
                ))
                .actual("converted audio failed final validation")
                .suggestions(&[
                    "Check if source audio files are severely corrupted",
                    "Try different conversion parameters",
                    "Verify audio compatibility manually",
                ])
                .build(),
                source: None,
            });
        }

        Ok(ConversionResult {
            reference: ref_audio,
            test: test_audio,
            conversions_applied,
            validation_results: (ref_validation, test_validation),
        })
    }

    /// Determine the target sample rate for conversion
    fn determine_target_sample_rate(
        &self,
        ref_audio: &AudioBuffer,
        test_audio: &AudioBuffer,
    ) -> u32 {
        if let Some(target_rate) = self.config.target_sample_rate {
            target_rate
        } else {
            // Use the higher sample rate for better quality
            ref_audio.sample_rate().max(test_audio.sample_rate())
        }
    }

    /// Determine the target channel count for conversion
    fn determine_target_channels(&self, ref_audio: &AudioBuffer, test_audio: &AudioBuffer) -> u32 {
        if let Some(target_channels) = self.config.target_channels {
            target_channels
        } else {
            // Prefer mono if either audio is mono, otherwise use minimum channels
            if ref_audio.channels() == 1 || test_audio.channels() == 1 {
                1
            } else {
                ref_audio.channels().min(test_audio.channels())
            }
        }
    }

    /// Convert sample rate of audio buffer
    fn convert_sample_rate(
        &self,
        audio: AudioBuffer,
        target_rate: u32,
    ) -> Result<AudioBuffer, EvaluationError> {
        convert_sample_rate(audio, target_rate, self.config.resample_quality).map_err(|e| {
            EvaluationError::AudioProcessingError {
                message: format!("Sample rate conversion failed: {}", e),
                source: None,
            }
        })
    }

    /// Convert channel count of audio buffer
    fn convert_channels(
        &self,
        audio: AudioBuffer,
        target_channels: u32,
    ) -> Result<AudioBuffer, EvaluationError> {
        convert_channels(audio, target_channels).map_err(|e| {
            EvaluationError::AudioProcessingError {
                message: format!("Channel conversion failed: {}", e),
                source: None,
            }
        })
    }

    /// Match audio lengths according to the configured strategy
    fn match_audio_lengths(
        &self,
        ref_audio: AudioBuffer,
        test_audio: AudioBuffer,
    ) -> Result<(AudioBuffer, AudioBuffer), EvaluationError> {
        let ref_samples = ref_audio.samples();
        let test_samples = test_audio.samples();

        let (new_ref_samples, new_test_samples) = match self.config.length_strategy {
            LengthMatchingStrategy::TrimToShortest => {
                let min_len = ref_samples.len().min(test_samples.len());
                (
                    ref_samples[..min_len].to_vec(),
                    test_samples[..min_len].to_vec(),
                )
            }
            LengthMatchingStrategy::PadToLongest => {
                let max_len = ref_samples.len().max(test_samples.len());
                let mut padded_ref = ref_samples.to_vec();
                let mut padded_test = test_samples.to_vec();

                padded_ref.resize(max_len, 0.0);
                padded_test.resize(max_len, 0.0);

                (padded_ref, padded_test)
            }
            LengthMatchingStrategy::RepeatToLongest => {
                let max_len = ref_samples.len().max(test_samples.len());

                let repeated_ref = if ref_samples.len() < max_len {
                    let mut result = Vec::with_capacity(max_len);
                    while result.len() < max_len {
                        let remaining = max_len - result.len();
                        let to_add = remaining.min(ref_samples.len());
                        result.extend_from_slice(&ref_samples[..to_add]);
                    }
                    result
                } else {
                    ref_samples.to_vec()
                };

                let repeated_test = if test_samples.len() < max_len {
                    let mut result = Vec::with_capacity(max_len);
                    while result.len() < max_len {
                        let remaining = max_len - result.len();
                        let to_add = remaining.min(test_samples.len());
                        result.extend_from_slice(&test_samples[..to_add]);
                    }
                    result
                } else {
                    test_samples.to_vec()
                };

                (repeated_ref, repeated_test)
            }
            LengthMatchingStrategy::KeepOriginal => (ref_samples.to_vec(), test_samples.to_vec()),
        };

        let new_ref_audio = if ref_audio.channels() == 1 {
            AudioBuffer::mono(new_ref_samples, ref_audio.sample_rate())
        } else {
            AudioBuffer::stereo(new_ref_samples, ref_audio.sample_rate())
        };

        let new_test_audio = if test_audio.channels() == 1 {
            AudioBuffer::mono(new_test_samples, test_audio.sample_rate())
        } else {
            AudioBuffer::stereo(new_test_samples, test_audio.sample_rate())
        };

        Ok((new_ref_audio, new_test_audio))
    }

    /// Remove DC offset from audio
    fn remove_dc_offset(&self, audio: AudioBuffer) -> Result<AudioBuffer, EvaluationError> {
        remove_dc_offset(audio).map_err(|e| EvaluationError::AudioProcessingError {
            message: format!("DC offset removal failed: {}", e),
            source: None,
        })
    }

    /// Normalize audio levels
    fn normalize_audio(&self, audio: AudioBuffer) -> Result<AudioBuffer, EvaluationError> {
        normalize_audio(audio).map_err(|e| EvaluationError::AudioProcessingError {
            message: format!("Audio normalization failed: {}", e),
            source: None,
        })
    }
}

impl Default for AutoConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for automatic conversion with default settings
pub fn auto_convert_audio_pair(
    reference: AudioBuffer,
    test: AudioBuffer,
    component: &str,
) -> Result<(AudioBuffer, AudioBuffer), EvaluationError> {
    let converter = AutoConverter::new();
    let result = converter.convert_for_compatibility(reference, test, component)?;
    Ok((result.reference, result.test))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_converter_same_format() {
        let ref_samples = (0..16000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let test_samples = (0..16000).map(|i| (i as f32 * 0.02).sin() * 0.3).collect();

        let ref_audio = AudioBuffer::mono(ref_samples, 16000);
        let test_audio = AudioBuffer::mono(test_samples, 16000);

        let converter = AutoConverter::new();
        let result = converter
            .convert_for_compatibility(ref_audio, test_audio, "test")
            .unwrap();

        assert_eq!(result.reference.sample_rate(), result.test.sample_rate());
        assert_eq!(result.reference.channels(), result.test.channels());
    }

    #[test]
    fn test_auto_converter_different_sample_rates() {
        let ref_samples = (0..16000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let test_samples = (0..44100).map(|i| (i as f32 * 0.02).sin() * 0.3).collect();

        let ref_audio = AudioBuffer::mono(ref_samples, 16000);
        let test_audio = AudioBuffer::mono(test_samples, 44100);

        let converter = AutoConverter::new();
        let result = converter
            .convert_for_compatibility(ref_audio, test_audio, "test")
            .unwrap();

        // Should convert to the higher sample rate (44100)
        assert_eq!(result.reference.sample_rate(), 44100);
        assert_eq!(result.test.sample_rate(), 44100);
        assert!(result
            .conversions_applied
            .iter()
            .any(|s| s.contains("sample rate converted")));
    }

    #[test]
    fn test_length_matching_trim_to_shortest() {
        // Create realistic audio samples with dynamic range
        let ref_samples: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let test_samples: Vec<f32> = (0..32000).map(|i| (i as f32 * 0.02).sin() * 0.3).collect();

        let ref_audio = AudioBuffer::mono(ref_samples, 16000);
        let test_audio = AudioBuffer::mono(test_samples, 16000);

        let config = AutoConversionConfig {
            length_strategy: LengthMatchingStrategy::TrimToShortest,
            ..Default::default()
        };

        let converter = AutoConverter::with_config(config);
        let result = converter
            .convert_for_compatibility(ref_audio, test_audio, "test")
            .unwrap();

        assert_eq!(result.reference.samples().len(), 16000);
        assert_eq!(result.test.samples().len(), 16000);
    }

    #[test]
    fn test_convenience_function() {
        let ref_samples = (0..16000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let test_samples = (0..16000).map(|i| (i as f32 * 0.02).sin() * 0.3).collect();

        let ref_audio = AudioBuffer::mono(ref_samples, 16000);
        let test_audio = AudioBuffer::mono(test_samples, 16000);

        let (converted_ref, converted_test) =
            auto_convert_audio_pair(ref_audio, test_audio, "test").unwrap();

        assert_eq!(converted_ref.sample_rate(), converted_test.sample_rate());
        assert_eq!(converted_ref.channels(), converted_test.channels());
    }
}
