//! Post-processing pipeline for audio enhancement and quality improvement
//!
//! This module provides a comprehensive post-processing pipeline for audio output
//! from vocoders, including dynamic range processing, noise reduction, and
//! frequency enhancement capabilities.

use crate::{AudioBuffer, Result};
use serde::{Deserialize, Serialize};

pub mod compression;
pub mod enhancement;
pub mod format_conversion;
pub mod noise_gate;

/// Configuration for the complete post-processing pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostProcessingConfig {
    /// Dynamic range compression settings
    pub compression: CompressionConfig,
    /// Noise gate configuration
    pub noise_gate: NoiseGateConfig,
    /// High-frequency enhancement settings
    pub enhancement: EnhancementConfig,
    /// Output format conversion settings
    pub format_conversion: FormatConversionConfig,
    /// Whether to enable the entire pipeline
    pub enabled: bool,
}

impl Default for PostProcessingConfig {
    fn default() -> Self {
        Self {
            compression: CompressionConfig::default(),
            noise_gate: NoiseGateConfig::default(),
            enhancement: EnhancementConfig::default(),
            format_conversion: FormatConversionConfig::default(),
            enabled: true,
        }
    }
}

/// Dynamic range compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Compression threshold in dB
    pub threshold: f32,
    /// Compression ratio (e.g., 4.0 = 4:1)
    pub ratio: f32,
    /// Attack time in milliseconds
    pub attack_ms: f32,
    /// Release time in milliseconds
    pub release_ms: f32,
    /// Makeup gain in dB
    pub makeup_gain: f32,
    /// Whether to enable soft knee
    pub soft_knee: bool,
    /// Whether compression is enabled
    pub enabled: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            threshold: -12.0, // dB
            ratio: 3.0,       // 3:1 compression
            attack_ms: 5.0,   // Fast attack
            release_ms: 50.0, // Medium release
            makeup_gain: 2.0, // dB
            soft_knee: true,
            enabled: true,
        }
    }
}

/// Noise gate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseGateConfig {
    /// Gate threshold in dB
    pub threshold: f32,
    /// Attack time in milliseconds
    pub attack_ms: f32,
    /// Release time in milliseconds
    pub release_ms: f32,
    /// Hold time in milliseconds
    pub hold_ms: f32,
    /// Whether spectral subtraction is enabled
    pub spectral_subtraction: bool,
    /// Spectral subtraction factor
    pub subtraction_factor: f32,
    /// Whether noise gate is enabled
    pub enabled: bool,
}

impl Default for NoiseGateConfig {
    fn default() -> Self {
        Self {
            threshold: -40.0,  // dB
            attack_ms: 1.0,    // Very fast attack
            release_ms: 100.0, // Smooth release
            hold_ms: 10.0,     // Brief hold
            spectral_subtraction: true,
            subtraction_factor: 0.5,
            enabled: true,
        }
    }
}

/// High-frequency enhancement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancementConfig {
    /// Brightness enhancement gain in dB
    pub brightness_gain: f32,
    /// High-frequency boost cutoff in Hz
    pub high_freq_cutoff: f32,
    /// Enhancement Q factor
    pub q_factor: f32,
    /// Presence enhancement gain in dB
    pub presence_gain: f32,
    /// Presence frequency center in Hz
    pub presence_freq: f32,
    /// Whether enhancement is enabled
    pub enabled: bool,
}

impl Default for EnhancementConfig {
    fn default() -> Self {
        Self {
            brightness_gain: 1.5,     // dB
            high_freq_cutoff: 8000.0, // Hz
            q_factor: 0.7,            // Smooth boost
            presence_gain: 0.8,       // dB
            presence_freq: 3000.0,    // Hz
            enabled: true,
        }
    }
}

/// Output format conversion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatConversionConfig {
    /// Target sample rate (0 = no resampling)
    pub target_sample_rate: u32,
    /// Target bit depth
    pub target_bit_depth: u16,
    /// Target number of channels
    pub target_channels: u16,
    /// Dithering method
    pub dithering: DitheringMethod,
    /// Whether format conversion is enabled
    pub enabled: bool,
}

impl Default for FormatConversionConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 0, // No resampling by default
            target_bit_depth: 16,
            target_channels: 1,
            dithering: DitheringMethod::TriangularPdf,
            enabled: false,
        }
    }
}

/// Dithering methods for bit depth conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DitheringMethod {
    /// No dithering
    None,
    /// Rectangular PDF dithering
    RectangularPdf,
    /// Triangular PDF dithering (recommended)
    TriangularPdf,
    /// Shaped dithering
    Shaped,
}

/// Main post-processing pipeline processor
#[derive(Debug)]
pub struct PostProcessor {
    config: PostProcessingConfig,
    compressor: compression::DynamicCompressor,
    noise_gate: noise_gate::NoiseGate,
    enhancer: enhancement::FrequencyEnhancer,
    format_converter: format_conversion::FormatConverter,
    sample_rate: f32,
}

impl PostProcessor {
    /// Create a new post-processor with the given configuration
    pub fn new(config: PostProcessingConfig, sample_rate: f32) -> Result<Self> {
        let compressor = compression::DynamicCompressor::new(&config.compression, sample_rate)?;
        let noise_gate = noise_gate::NoiseGate::new(&config.noise_gate, sample_rate)?;
        let enhancer = enhancement::FrequencyEnhancer::new(&config.enhancement, sample_rate)?;
        let format_converter = format_conversion::FormatConverter::new(&config.format_conversion)?;

        Ok(Self {
            config,
            compressor,
            noise_gate,
            enhancer,
            format_converter,
            sample_rate,
        })
    }

    /// Process audio through the complete pipeline
    pub fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Step 1: Noise gate and spectral subtraction
        if self.config.noise_gate.enabled {
            self.noise_gate.process(audio)?;
        }

        // Step 2: Dynamic range compression
        if self.config.compression.enabled {
            self.compressor.process(audio)?;
        }

        // Step 3: High-frequency enhancement
        if self.config.enhancement.enabled {
            self.enhancer.process(audio)?;
        }

        // Step 4: Format conversion (if needed)
        if self.config.format_conversion.enabled {
            self.format_converter.process(audio)?;
        }

        Ok(())
    }

    /// Update the configuration and reinitialize processors
    pub fn update_config(&mut self, config: PostProcessingConfig) -> Result<()> {
        self.config = config.clone();

        self.compressor =
            compression::DynamicCompressor::new(&config.compression, self.sample_rate)?;
        self.noise_gate = noise_gate::NoiseGate::new(&config.noise_gate, self.sample_rate)?;
        self.enhancer = enhancement::FrequencyEnhancer::new(&config.enhancement, self.sample_rate)?;
        self.format_converter = format_conversion::FormatConverter::new(&config.format_conversion)?;

        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> &PostProcessingConfig {
        &self.config
    }

    /// Get processing statistics
    pub fn statistics(&self) -> ProcessingStats {
        ProcessingStats {
            compression_gain_reduction: self.compressor.gain_reduction(),
            noise_gate_attenuation: self.noise_gate.attenuation(),
            enhancement_gain: self.enhancer.total_gain(),
            processed_samples: self.compressor.processed_samples(),
        }
    }
}

/// Statistics from the post-processing pipeline
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    /// Current compression gain reduction in dB
    pub compression_gain_reduction: f32,
    /// Current noise gate attenuation in dB
    pub noise_gate_attenuation: f32,
    /// Total enhancement gain applied in dB
    pub enhancement_gain: f32,
    /// Total number of samples processed
    pub processed_samples: u64,
}

/// Utility function to convert dB to linear scale
pub fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Utility function to convert linear scale to dB
pub fn linear_to_db(linear: f32) -> f32 {
    20.0 * linear.abs().max(1e-10).log10()
}

/// Calculate RMS level of audio samples
pub fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let sum_squares: f32 = samples.iter().map(|x| x * x).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

/// Apply soft knee to compression curve
pub fn soft_knee_compression(input_db: f32, threshold: f32, ratio: f32, knee_width: f32) -> f32 {
    if input_db <= threshold - knee_width / 2.0 {
        // Below knee, no compression
        input_db
    } else if input_db >= threshold + knee_width / 2.0 {
        // Above knee, full compression
        threshold + (input_db - threshold) / ratio
    } else {
        // In knee region, smooth transition
        let knee_ratio = (input_db - threshold + knee_width / 2.0) / knee_width;
        let transition_ratio = 1.0 + (ratio - 1.0) * knee_ratio;
        threshold + (input_db - threshold) / transition_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_conversion() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-6);
        assert!((db_to_linear(20.0) - 10.0).abs() < 1e-6);
        assert!((linear_to_db(1.0) - 0.0).abs() < 1e-6);
        assert!((linear_to_db(10.0) - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_rms_calculation() {
        let samples = vec![0.0, 1.0, 0.0, -1.0];
        let rms = calculate_rms(&samples);
        assert!((rms - (0.5_f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_soft_knee_compression() {
        let threshold = -10.0;
        let ratio = 4.0;
        let knee_width = 4.0;

        // Below knee should pass through
        assert!(
            (soft_knee_compression(-15.0, threshold, ratio, knee_width) - (-15.0)).abs() < 1e-6
        );

        // Above knee should compress
        let compressed = soft_knee_compression(-5.0, threshold, ratio, knee_width);
        assert!(compressed < -5.0); // Should be reduced (more negative)
        assert!(compressed > threshold); // But not below threshold
    }

    #[test]
    fn test_post_processing_config_default() {
        let config = PostProcessingConfig::default();
        assert!(config.enabled);
        assert!(config.compression.enabled);
        assert!(config.noise_gate.enabled);
        assert!(config.enhancement.enabled);
    }
}
