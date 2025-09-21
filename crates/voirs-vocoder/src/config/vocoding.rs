//! Vocoding configuration module.
//!
//! This module provides configuration options for vocoding quality,
//! performance modes, and processing parameters.

use super::{PerformanceProfile, ValidationResult};
use serde::{Deserialize, Serialize};

/// Vocoding quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityLevel {
    /// Low quality (fastest processing)
    Low,
    /// Medium quality (balanced)
    Medium,
    /// High quality (slower processing)
    High,
    /// Ultra quality (highest quality, slowest)
    Ultra,
}

impl Default for QualityLevel {
    fn default() -> Self {
        Self::Medium
    }
}

/// Vocoding performance modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VocodingMode {
    /// Optimized for quality
    Quality,
    /// Balanced quality and speed
    Balanced,
    /// Optimized for speed
    Speed,
    /// Real-time processing mode
    Realtime,
}

impl Default for VocodingMode {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Vocoding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocodingConfig {
    /// Quality level for vocoding
    pub quality: QualityLevel,
    /// Performance mode
    pub mode: VocodingMode,
    /// Performance profile
    pub profile: PerformanceProfile,
    /// Sample rate for vocoding (Hz)
    pub sample_rate: u32,
    /// Bit depth for output
    pub bit_depth: u16,
    /// Number of mel frequency channels
    pub mel_channels: u32,
    /// FFT size for spectral processing
    pub fft_size: usize,
    /// Hop length for frame processing
    pub hop_length: u32,
    /// Window size for spectral analysis
    pub window_size: usize,
    /// Enable high-frequency enhancement
    pub enable_hf_enhancement: bool,
    /// Enable spectral enhancement
    pub enable_spectral_enhancement: bool,
    /// Enable temporal smoothing
    pub enable_temporal_smoothing: bool,
    /// Temperature for stochastic vocoding (0.0-2.0)
    pub temperature: f32,
    /// Guidance scale for quality control (0.0-2.0)
    pub guidance_scale: f32,
    /// Number of inference steps for iterative vocoders
    pub inference_steps: u32,
    /// Use mixed precision (FP16) if supported
    pub use_mixed_precision: bool,
    /// Enable post-processing
    pub enable_post_processing: bool,
}

impl Default for VocodingConfig {
    fn default() -> Self {
        Self {
            quality: QualityLevel::Medium,
            mode: VocodingMode::Balanced,
            profile: PerformanceProfile::Balanced,
            sample_rate: 22050,
            bit_depth: 16,
            mel_channels: 80,
            fft_size: 1024,
            hop_length: 256,
            window_size: 1024,
            enable_hf_enhancement: true,
            enable_spectral_enhancement: true,
            enable_temporal_smoothing: true,
            temperature: 1.0,
            guidance_scale: 1.0,
            inference_steps: 50,
            use_mixed_precision: false,
            enable_post_processing: true,
        }
    }
}

impl VocodingConfig {
    /// Create configuration for high-quality vocoding
    pub fn high_quality() -> Self {
        Self {
            quality: QualityLevel::High,
            mode: VocodingMode::Quality,
            profile: PerformanceProfile::Quality,
            sample_rate: 44100,
            bit_depth: 24,
            mel_channels: 128,
            fft_size: 2048,
            hop_length: 512,
            window_size: 2048,
            enable_hf_enhancement: true,
            enable_spectral_enhancement: true,
            enable_temporal_smoothing: true,
            temperature: 0.8,
            guidance_scale: 1.2,
            inference_steps: 100,
            use_mixed_precision: false,
            enable_post_processing: true,
        }
    }

    /// Create configuration for real-time vocoding
    pub fn realtime() -> Self {
        Self {
            quality: QualityLevel::Low,
            mode: VocodingMode::Realtime,
            profile: PerformanceProfile::Realtime,
            sample_rate: 22050,
            bit_depth: 16,
            mel_channels: 80,
            fft_size: 512,
            hop_length: 128,
            window_size: 512,
            enable_hf_enhancement: false,
            enable_spectral_enhancement: false,
            enable_temporal_smoothing: false,
            temperature: 1.0,
            guidance_scale: 1.0,
            inference_steps: 10,
            use_mixed_precision: true,
            enable_post_processing: false,
        }
    }

    /// Create configuration for low-resource environments
    pub fn low_resource() -> Self {
        Self {
            quality: QualityLevel::Low,
            mode: VocodingMode::Speed,
            profile: PerformanceProfile::Speed,
            sample_rate: 16000,
            bit_depth: 16,
            mel_channels: 64,
            fft_size: 512,
            hop_length: 128,
            window_size: 512,
            enable_hf_enhancement: false,
            enable_spectral_enhancement: false,
            enable_temporal_smoothing: false,
            temperature: 1.0,
            guidance_scale: 1.0,
            inference_steps: 5,
            use_mixed_precision: true,
            enable_post_processing: false,
        }
    }

    /// Create configuration for ultra-high quality
    pub fn ultra_quality() -> Self {
        Self {
            quality: QualityLevel::Ultra,
            mode: VocodingMode::Quality,
            profile: PerformanceProfile::Quality,
            sample_rate: 48000,
            bit_depth: 32,
            mel_channels: 256,
            fft_size: 4096,
            hop_length: 1024,
            window_size: 4096,
            enable_hf_enhancement: true,
            enable_spectral_enhancement: true,
            enable_temporal_smoothing: true,
            temperature: 0.6,
            guidance_scale: 1.5,
            inference_steps: 200,
            use_mixed_precision: false,
            enable_post_processing: true,
        }
    }

    /// Validate the vocoding configuration
    pub fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate sample rate
        if self.sample_rate < 8000 {
            errors.push("Sample rate must be at least 8000 Hz".to_string());
        } else if self.sample_rate > 96000 {
            warnings.push("Sample rate above 96000 Hz may cause performance issues".to_string());
        }

        // Validate bit depth
        match self.bit_depth {
            16 | 24 | 32 => {}
            _ => errors.push("Bit depth must be 16, 24, or 32".to_string()),
        }

        // Validate mel channels
        if self.mel_channels < 32 {
            warnings.push("Mel channels below 32 may reduce quality".to_string());
        } else if self.mel_channels > 512 {
            warnings.push("Mel channels above 512 may cause excessive memory usage".to_string());
        }

        // Validate FFT size (must be power of 2)
        if !self.fft_size.is_power_of_two() {
            errors.push("FFT size must be a power of 2".to_string());
        }
        if self.fft_size < 256 {
            warnings.push("FFT size below 256 may reduce quality".to_string());
        } else if self.fft_size > 8192 {
            warnings.push("FFT size above 8192 may cause performance issues".to_string());
        }

        // Validate hop length
        if self.hop_length == 0 {
            errors.push("Hop length must be greater than 0".to_string());
        } else if self.hop_length > self.fft_size as u32 {
            errors.push("Hop length should not exceed FFT size".to_string());
        }

        // Validate window size
        if !self.window_size.is_power_of_two() {
            warnings.push("Window size should be a power of 2 for optimal performance".to_string());
        }
        if self.window_size != self.fft_size {
            warnings.push("Window size should typically match FFT size".to_string());
        }

        // Validate temperature
        if self.temperature < 0.0 || self.temperature > 2.0 {
            warnings.push("Temperature should be between 0.0 and 2.0".to_string());
        }

        // Validate guidance scale
        if self.guidance_scale < 0.0 || self.guidance_scale > 3.0 {
            warnings.push("Guidance scale should be between 0.0 and 3.0".to_string());
        }

        // Validate inference steps
        if self.inference_steps == 0 {
            errors.push("Inference steps must be greater than 0".to_string());
        } else if self.inference_steps > 1000 {
            warnings.push("High inference steps may cause very slow processing".to_string());
        }

        // Check for mode-specific warnings
        match self.mode {
            VocodingMode::Realtime => {
                if self.inference_steps > 20 {
                    warnings.push(
                        "High inference steps may not be suitable for real-time mode".to_string(),
                    );
                }
                if self.enable_post_processing {
                    warnings.push("Post-processing may add latency in real-time mode".to_string());
                }
            }
            VocodingMode::Quality => {
                if self.inference_steps < 50 {
                    warnings
                        .push("Low inference steps may reduce quality in quality mode".to_string());
                }
            }
            _ => {}
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        }
    }

    /// Get estimated memory usage in MB
    pub fn estimated_memory_mb(&self) -> f32 {
        let fft_memory = (self.fft_size * 8) as f32; // Complex numbers
        let mel_memory = (self.mel_channels * 1024 * 4) as f32; // Float values
        let buffer_memory = (self.sample_rate * 4) as f32; // 1 second buffer

        (fft_memory + mel_memory + buffer_memory) / (1024.0 * 1024.0)
    }

    /// Get estimated processing time factor (relative to real-time)
    pub fn estimated_rtf(&self) -> f32 {
        let mut rtf = 1.0;

        // Adjust based on quality
        rtf *= match self.quality {
            QualityLevel::Low => 0.1,
            QualityLevel::Medium => 0.3,
            QualityLevel::High => 0.8,
            QualityLevel::Ultra => 2.0,
        };

        // Adjust based on inference steps
        rtf *= (self.inference_steps as f32 / 50.0).min(10.0);

        // Adjust based on enhancements
        if self.enable_hf_enhancement {
            rtf *= 1.2;
        }
        if self.enable_spectral_enhancement {
            rtf *= 1.3;
        }
        if self.enable_temporal_smoothing {
            rtf *= 1.1;
        }
        if self.enable_post_processing {
            rtf *= 1.2;
        }

        rtf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_vocoding_config() {
        let config = VocodingConfig::default();
        assert_eq!(config.quality, QualityLevel::Medium);
        assert_eq!(config.sample_rate, 22050);
        assert_eq!(config.mel_channels, 80);
    }

    #[test]
    fn test_high_quality_config() {
        let config = VocodingConfig::high_quality();
        assert_eq!(config.quality, QualityLevel::High);
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.bit_depth, 24);
    }

    #[test]
    fn test_realtime_config() {
        let config = VocodingConfig::realtime();
        assert_eq!(config.mode, VocodingMode::Realtime);
        assert!(!config.enable_post_processing);
        assert!(config.use_mixed_precision);
    }

    #[test]
    fn test_config_validation() {
        let mut config = VocodingConfig::default();

        // Valid configuration
        let result = config.validate();
        assert!(result.is_valid);

        // Invalid FFT size (not power of 2)
        config.fft_size = 1000;
        let result = config.validate();
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.contains("power of 2")));

        // Invalid sample rate
        config.fft_size = 1024; // Reset to valid
        config.sample_rate = 1000;
        let result = config.validate();
        assert!(!result.is_valid);
    }

    #[test]
    fn test_memory_estimation() {
        let config = VocodingConfig::default();
        let memory_mb = config.estimated_memory_mb();
        assert!(memory_mb > 0.0);
        assert!(memory_mb < 1000.0); // Reasonable upper bound
    }

    #[test]
    fn test_rtf_estimation() {
        let low_config = VocodingConfig::low_resource();
        let high_config = VocodingConfig::high_quality();

        let low_rtf = low_config.estimated_rtf();
        let high_rtf = high_config.estimated_rtf();

        assert!(low_rtf > 0.0);
        assert!(high_rtf > low_rtf); // High quality should take longer
    }
}
