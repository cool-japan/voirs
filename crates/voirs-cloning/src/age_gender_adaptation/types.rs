use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Age categories for voice adaptation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgeCategory {
    /// Child voice (5-12 years)
    Child,
    /// Teenager voice (13-18 years)
    Teenager,
    /// Young adult voice (19-30 years)
    YoungAdult,
    /// Adult voice (31-50 years)
    Adult,
    /// Middle-aged voice (51-65 years)
    MiddleAged,
    /// Senior voice (65+ years)
    Senior,
}

/// Gender categories for voice adaptation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenderCategory {
    /// Masculine voice characteristics
    Masculine,
    /// Feminine voice characteristics
    Feminine,
    /// Neutral/androgynous voice characteristics
    Neutral,
}

/// Voice adaptation target combining age and gender
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceAdaptationTarget {
    /// Target age category
    pub age: AgeCategory,
    /// Target gender category
    pub gender: GenderCategory,
    /// Age intensity (0.0 = minimal change, 1.0 = maximum change)
    pub age_intensity: f32,
    /// Gender intensity (0.0 = minimal change, 1.0 = maximum change)
    pub gender_intensity: f32,
    /// Preserve speaker identity (0.0 = no preservation, 1.0 = maximum preservation)
    pub identity_preservation: f32,
}

/// Age and Gender adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgeGenderAdaptationConfig {
    /// Fundamental frequency (F0) modification parameters
    pub f0_adaptation: F0AdaptationConfig,
    /// Formant frequency modification parameters
    pub formant_adaptation: FormantAdaptationConfig,
    /// Voice quality modification parameters
    pub quality_adaptation: QualityAdaptationConfig,
    /// Spectral adaptation parameters
    pub spectral_adaptation: SpectralAdaptationConfig,
    /// Temporal adaptation parameters
    pub temporal_adaptation: TemporalAdaptationConfig,
    /// Enable real-time adaptation
    pub real_time_enabled: bool,
    /// Adaptation smoothness factor (0.0-1.0)
    pub smoothness_factor: f32,
}

/// Fundamental frequency adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F0AdaptationConfig {
    /// Base F0 shift in semitones for age adaptation
    pub age_f0_shift_range: (f32, f32), // (min, max) semitones
    /// Base F0 shift in semitones for gender adaptation
    pub gender_f0_shift_range: (f32, f32), // (min, max) semitones
    /// F0 variation adaptation (affects prosody)
    pub f0_variation_factor: f32,
    /// Jitter adaptation for voice quality
    pub jitter_adaptation: f32,
}

/// Formant frequency adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantAdaptationConfig {
    /// Formant frequency shifts for age (F1, F2, F3, F4)
    pub age_formant_shifts: [f32; 4],
    /// Formant frequency shifts for gender (F1, F2, F3, F4)
    pub gender_formant_shifts: [f32; 4],
    /// Formant bandwidth adaptation factors
    pub bandwidth_factors: [f32; 4],
    /// Vocal tract length simulation factor
    pub vocal_tract_length_factor: f32,
}

/// Voice quality adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAdaptationConfig {
    /// Breathiness adaptation (0.0-1.0)
    pub breathiness_range: (f32, f32),
    /// Roughness adaptation (0.0-1.0)
    pub roughness_range: (f32, f32),
    /// Harmonics-to-noise ratio adaptation
    pub hnr_adaptation: f32,
    /// Spectral tilt adaptation (dB/octave)
    pub spectral_tilt_range: (f32, f32),
}

/// Spectral adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAdaptationConfig {
    /// Spectral envelope warping factor
    pub envelope_warping: f32,
    /// High frequency emphasis/de-emphasis (dB)
    pub high_freq_emphasis: f32,
    /// Spectral smoothing factor
    pub smoothing_factor: f32,
    /// Noise floor adaptation (dB)
    pub noise_floor_adaptation: f32,
}

/// Temporal adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAdaptationConfig {
    /// Speech rate adaptation factor
    pub speech_rate_factor: f32,
    /// Pause duration adaptation factor
    pub pause_duration_factor: f32,
    /// Articulation precision adaptation
    pub articulation_precision: f32,
    /// Rhythm adaptation intensity
    pub rhythm_adaptation: f32,
}

/// Age and Gender adaptation result
#[derive(Debug, Clone)]
pub struct AgeGenderAdaptationResult {
    /// Adaptation success status
    pub success: bool,
    /// Adapted voice characteristics
    pub adapted_characteristics: VoiceCharacteristics,
    /// Adaptation confidence score (0.0-1.0)
    pub confidence: f32,
    /// Quality metrics of adapted voice
    pub quality_metrics: AdaptationQualityMetrics,
    /// Processing statistics
    pub processing_stats: AdaptationProcessingStats,
}

/// Adapted voice characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCharacteristics {
    /// Estimated apparent age
    pub apparent_age: f32,
    /// Estimated gender score (-1.0 = masculine, +1.0 = feminine)
    pub gender_score: f32,
    /// Fundamental frequency statistics
    pub f0_statistics: F0Statistics,
    /// Formant frequencies (F1, F2, F3, F4)
    pub formant_frequencies: [f32; 4],
    /// Voice quality metrics
    pub voice_quality: VoiceQualityMetrics,
    /// Spectral characteristics
    pub spectral_characteristics: SpectralCharacteristics,
}

/// F0 statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F0Statistics {
    /// Mean F0 in Hz
    pub mean_f0: f32,
    /// F0 standard deviation
    pub f0_std: f32,
    /// F0 range (max - min)
    pub f0_range: f32,
    /// Jitter percentage
    pub jitter: f32,
}

/// Voice quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQualityMetrics {
    /// Breathiness level (0.0-1.0)
    pub breathiness: f32,
    /// Roughness level (0.0-1.0)
    pub roughness: f32,
    /// Harmonics-to-noise ratio (dB)
    pub hnr: f32,
    /// Spectral tilt (dB/octave)
    pub spectral_tilt: f32,
}

/// Spectral characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralCharacteristics {
    /// Spectral centroid (Hz)
    pub spectral_centroid: f32,
    /// Spectral rolloff (Hz)
    pub spectral_rolloff: f32,
    /// Spectral flux
    pub spectral_flux: f32,
    /// High frequency energy ratio
    pub high_freq_ratio: f32,
}

/// Adaptation quality metrics
#[derive(Debug, Clone)]
pub struct AdaptationQualityMetrics {
    /// Naturalness score (0.0-1.0)
    pub naturalness: f32,
    /// Identity preservation score (0.0-1.0)
    pub identity_preservation: f32,
    /// Target achievement score (0.0-1.0)
    pub target_achievement: f32,
    /// Audio quality score (0.0-1.0)
    pub audio_quality: f32,
}

/// Adaptation processing statistics
#[derive(Debug, Clone)]
pub struct AdaptationProcessingStats {
    /// Processing time
    pub processing_time: std::time::Duration,
    /// Number of frames processed
    pub frames_processed: usize,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Adaptation convergence achieved
    pub converged: bool,
}

/// Main Age/Gender adaptation processor
#[derive(Debug)]
pub struct AgeGenderAdapter {
    /// Adaptation configuration
    pub(super) config: AgeGenderAdaptationConfig,
    /// Adaptation models cache
    pub(super) model_cache: HashMap<String, AgeGenderModel>,
    /// Voice analysis cache
    pub(super) analysis_cache: HashMap<String, VoiceCharacteristics>,
}

/// Age/Gender adaptation model
#[derive(Debug, Clone)]
pub struct AgeGenderModel {
    /// Source voice characteristics
    pub source_characteristics: VoiceCharacteristics,
    /// Target adaptation parameters
    pub target: VoiceAdaptationTarget,
    /// Adaptation transformation matrices
    pub transformation_matrices: TransformationMatrices,
    /// Model training statistics
    pub training_stats: ModelTrainingStats,
}

/// Transformation matrices for adaptation
#[derive(Debug, Clone)]
pub struct TransformationMatrices {
    /// F0 transformation curve
    pub f0_transform: Array1<f32>,
    /// Formant transformation matrix (4x4 for F1-F4)
    pub formant_transform: Array2<f32>,
    /// Spectral envelope transformation
    pub spectral_transform: Array1<f32>,
    /// Quality parameters transformation
    pub quality_transform: Array1<f32>,
}

/// Model training statistics
#[derive(Debug, Clone)]
pub struct ModelTrainingStats {
    /// Training samples used
    pub training_samples: usize,
    /// Training accuracy achieved
    pub training_accuracy: f32,
    /// Cross-validation score
    pub cv_score: f32,
    /// Model complexity score
    pub complexity_score: f32,
}

impl Default for AgeGenderAdaptationConfig {
    fn default() -> Self {
        Self {
            f0_adaptation: F0AdaptationConfig::default(),
            formant_adaptation: FormantAdaptationConfig::default(),
            quality_adaptation: QualityAdaptationConfig::default(),
            spectral_adaptation: SpectralAdaptationConfig::default(),
            temporal_adaptation: TemporalAdaptationConfig::default(),
            real_time_enabled: false,
            smoothness_factor: 0.3,
        }
    }
}

impl Default for F0AdaptationConfig {
    fn default() -> Self {
        Self {
            age_f0_shift_range: (-24.0, 24.0),
            gender_f0_shift_range: (-12.0, 12.0),
            f0_variation_factor: 1.0,
            jitter_adaptation: 0.1,
        }
    }
}

impl Default for FormantAdaptationConfig {
    fn default() -> Self {
        Self {
            age_formant_shifts: [0.0, 0.0, 0.0, 0.0],
            gender_formant_shifts: [0.0, 0.0, 0.0, 0.0],
            bandwidth_factors: [1.0, 1.0, 1.0, 1.0],
            vocal_tract_length_factor: 1.0,
        }
    }
}

impl Default for QualityAdaptationConfig {
    fn default() -> Self {
        Self {
            breathiness_range: (0.0, 0.5),
            roughness_range: (0.0, 0.3),
            hnr_adaptation: 0.0,
            spectral_tilt_range: (-5.0, 5.0),
        }
    }
}

impl Default for SpectralAdaptationConfig {
    fn default() -> Self {
        Self {
            envelope_warping: 0.0,
            high_freq_emphasis: 0.0,
            smoothing_factor: 0.1,
            noise_floor_adaptation: 0.0,
        }
    }
}

impl Default for TemporalAdaptationConfig {
    fn default() -> Self {
        Self {
            speech_rate_factor: 1.0,
            pause_duration_factor: 1.0,
            articulation_precision: 1.0,
            rhythm_adaptation: 0.0,
        }
    }
}

impl Default for VoiceAdaptationTarget {
    fn default() -> Self {
        Self {
            age: AgeCategory::Adult,
            gender: GenderCategory::Neutral,
            age_intensity: 0.5,
            gender_intensity: 0.5,
            identity_preservation: 0.7,
        }
    }
}
