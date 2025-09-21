//! Configuration for singing voice vocoder.

use serde::{Deserialize, Serialize};

/// Configuration for singing voice vocoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingingVocoderConfig {
    /// Sample rate for audio processing
    pub sample_rate: u32,
    /// Hop size for frame processing
    pub hop_size: usize,
    /// Pitch stability configuration
    pub pitch_stability: PitchStabilityConfig,
    /// Vibrato processing configuration
    pub vibrato: VibratoConfig,
    /// Harmonic enhancement configuration
    pub harmonic_enhancement: HarmonicEnhancementConfig,
    /// Breath sound processing configuration
    pub breath_sound: BreathSoundConfig,
    /// Artifact reduction configuration
    pub artifact_reduction: ArtifactReductionConfig,
    /// Quality metrics configuration
    pub quality_metrics: QualityMetricsConfig,
}

/// Configuration for pitch stability processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchStabilityConfig {
    /// Threshold for pitch stability detection
    pub stability_threshold: f32,
    /// Smoothing factor for pitch correction
    pub smoothing_factor: f32,
    /// Maximum pitch deviation allowed
    pub max_pitch_deviation: f32,
    /// Enable pitch correction
    pub enable_correction: bool,
    /// Correction strength (0.0-1.0)
    pub correction_strength: f32,
}

/// Configuration for vibrato processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibratoConfig {
    /// Enable vibrato enhancement
    pub enable_enhancement: bool,
    /// Vibrato detection threshold
    pub detection_threshold: f32,
    /// Enhancement strength (0.0-1.0)
    pub enhancement_strength: f32,
    /// Vibrato frequency range (Hz)
    pub frequency_range: (f32, f32),
    /// Vibrato depth range (cents)
    pub depth_range: (f32, f32),
}

/// Configuration for harmonic enhancement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicEnhancementConfig {
    /// Enable harmonic enhancement
    pub enable_enhancement: bool,
    /// Number of harmonics to enhance
    pub harmonic_count: u32,
    /// Enhancement strength per harmonic
    pub enhancement_strengths: Vec<f32>,
    /// Frequency range for enhancement
    pub frequency_range: (f32, f32),
    /// Adaptive enhancement based on voice characteristics
    pub adaptive_enhancement: bool,
}

/// Configuration for breath sound processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathSoundConfig {
    /// Enable breath sound processing
    pub enable_processing: bool,
    /// Breath detection threshold
    pub detection_threshold: f32,
    /// Breath enhancement strength
    pub enhancement_strength: f32,
    /// Frequency range for breath sounds
    pub frequency_range: (f32, f32),
    /// Breath sound reduction strength
    pub reduction_strength: f32,
}

/// Configuration for artifact reduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactReductionConfig {
    /// Enable artifact reduction
    pub enable_reduction: bool,
    /// Spectral noise reduction strength
    pub noise_reduction_strength: f32,
    /// Harmonic artifact reduction strength
    pub harmonic_artifact_reduction: f32,
    /// Temporal artifact reduction strength
    pub temporal_artifact_reduction: f32,
    /// Frequency range for artifact detection
    pub artifact_frequency_range: (f32, f32),
}

/// Configuration for quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetricsConfig {
    /// Enable quality metrics calculation
    pub enable_metrics: bool,
    /// Calculate pitch accuracy
    pub calculate_pitch_accuracy: bool,
    /// Calculate harmonic clarity
    pub calculate_harmonic_clarity: bool,
    /// Calculate spectral stability
    pub calculate_spectral_stability: bool,
    /// Calculate overall singing quality
    pub calculate_singing_quality: bool,
}

impl Default for SingingVocoderConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            hop_size: 256,
            pitch_stability: PitchStabilityConfig::default(),
            vibrato: VibratoConfig::default(),
            harmonic_enhancement: HarmonicEnhancementConfig::default(),
            breath_sound: BreathSoundConfig::default(),
            artifact_reduction: ArtifactReductionConfig::default(),
            quality_metrics: QualityMetricsConfig::default(),
        }
    }
}

impl Default for PitchStabilityConfig {
    fn default() -> Self {
        Self {
            stability_threshold: 0.05,
            smoothing_factor: 0.8,
            max_pitch_deviation: 0.2,
            enable_correction: true,
            correction_strength: 0.7,
        }
    }
}

impl Default for VibratoConfig {
    fn default() -> Self {
        Self {
            enable_enhancement: true,
            detection_threshold: 0.1,
            enhancement_strength: 0.5,
            frequency_range: (4.0, 8.0),
            depth_range: (10.0, 100.0),
        }
    }
}

impl Default for HarmonicEnhancementConfig {
    fn default() -> Self {
        Self {
            enable_enhancement: true,
            harmonic_count: 5,
            enhancement_strengths: vec![1.0, 0.8, 0.6, 0.4, 0.2],
            frequency_range: (80.0, 8000.0),
            adaptive_enhancement: true,
        }
    }
}

impl Default for BreathSoundConfig {
    fn default() -> Self {
        Self {
            enable_processing: true,
            detection_threshold: 0.05,
            enhancement_strength: 0.3,
            frequency_range: (1000.0, 8000.0),
            reduction_strength: 0.5,
        }
    }
}

impl Default for ArtifactReductionConfig {
    fn default() -> Self {
        Self {
            enable_reduction: true,
            noise_reduction_strength: 0.6,
            harmonic_artifact_reduction: 0.7,
            temporal_artifact_reduction: 0.5,
            artifact_frequency_range: (20.0, 20000.0),
        }
    }
}

impl Default for QualityMetricsConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            calculate_pitch_accuracy: true,
            calculate_harmonic_clarity: true,
            calculate_spectral_stability: true,
            calculate_singing_quality: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_singing_vocoder_config_default() {
        let config = SingingVocoderConfig::default();
        assert_eq!(config.sample_rate, 22050);
        assert_eq!(config.hop_size, 256);
        assert!(config.pitch_stability.enable_correction);
        assert!(config.vibrato.enable_enhancement);
        assert!(config.harmonic_enhancement.enable_enhancement);
        assert!(config.breath_sound.enable_processing);
        assert!(config.artifact_reduction.enable_reduction);
        assert!(config.quality_metrics.enable_metrics);
    }

    #[test]
    fn test_pitch_stability_config() {
        let config = PitchStabilityConfig::default();
        assert_eq!(config.stability_threshold, 0.05);
        assert_eq!(config.smoothing_factor, 0.8);
        assert_eq!(config.max_pitch_deviation, 0.2);
        assert!(config.enable_correction);
        assert_eq!(config.correction_strength, 0.7);
    }

    #[test]
    fn test_vibrato_config() {
        let config = VibratoConfig::default();
        assert!(config.enable_enhancement);
        assert_eq!(config.detection_threshold, 0.1);
        assert_eq!(config.enhancement_strength, 0.5);
        assert_eq!(config.frequency_range, (4.0, 8.0));
        assert_eq!(config.depth_range, (10.0, 100.0));
    }

    #[test]
    fn test_harmonic_enhancement_config() {
        let config = HarmonicEnhancementConfig::default();
        assert!(config.enable_enhancement);
        assert_eq!(config.harmonic_count, 5);
        assert_eq!(config.enhancement_strengths.len(), 5);
        assert_eq!(config.frequency_range, (80.0, 8000.0));
        assert!(config.adaptive_enhancement);
    }

    #[test]
    fn test_breath_sound_config() {
        let config = BreathSoundConfig::default();
        assert!(config.enable_processing);
        assert_eq!(config.detection_threshold, 0.05);
        assert_eq!(config.enhancement_strength, 0.3);
        assert_eq!(config.frequency_range, (1000.0, 8000.0));
        assert_eq!(config.reduction_strength, 0.5);
    }

    #[test]
    fn test_artifact_reduction_config() {
        let config = ArtifactReductionConfig::default();
        assert!(config.enable_reduction);
        assert_eq!(config.noise_reduction_strength, 0.6);
        assert_eq!(config.harmonic_artifact_reduction, 0.7);
        assert_eq!(config.temporal_artifact_reduction, 0.5);
        assert_eq!(config.artifact_frequency_range, (20.0, 20000.0));
    }

    #[test]
    fn test_quality_metrics_config() {
        let config = QualityMetricsConfig::default();
        assert!(config.enable_metrics);
        assert!(config.calculate_pitch_accuracy);
        assert!(config.calculate_harmonic_clarity);
        assert!(config.calculate_spectral_stability);
        assert!(config.calculate_singing_quality);
    }
}
