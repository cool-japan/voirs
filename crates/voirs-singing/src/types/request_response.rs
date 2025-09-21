//! Request and response types for singing synthesis operations

use super::voice_types::VoiceCharacteristics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Quality settings for singing synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySettings {
    /// Synthesis quality level (0-10)
    pub quality_level: u8,
    /// Enable high-quality pitch processing
    pub high_quality_pitch: bool,
    /// Enable advanced vibrato modeling
    pub advanced_vibrato: bool,
    /// Enable breath modeling
    pub breath_modeling: bool,
    /// Enable formant modeling
    pub formant_modeling: bool,
    /// FFT size for spectral processing
    pub fft_size: usize,
    /// Hop size for spectral processing
    pub hop_size: usize,
}

/// Singing request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingingRequest {
    /// Musical score to sing
    pub score: crate::score::MusicalScore,
    /// Voice characteristics to use
    pub voice: VoiceCharacteristics,
    /// Singing technique parameters
    pub technique: crate::techniques::SingingTechnique,
    /// Effects to apply
    pub effects: Vec<String>,
    /// Output sample rate
    pub sample_rate: u32,
    /// Target duration (optional)
    pub target_duration: Option<Duration>,
    /// Quality settings
    pub quality: QualitySettings,
}

/// Singing response
#[derive(Debug, Clone)]
pub struct SingingResponse {
    /// Synthesized audio samples
    pub audio: Vec<f32>,
    /// Sample rate of the audio
    pub sample_rate: u32,
    /// Duration of the audio
    pub duration: Duration,
    /// Applied voice characteristics
    pub voice: VoiceCharacteristics,
    /// Applied technique
    pub technique: crate::techniques::SingingTechnique,
    /// Performance statistics
    pub stats: SingingStats,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Performance statistics for singing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingingStats {
    /// Total notes processed
    pub total_notes: usize,
    /// Total processing time
    pub processing_time: Duration,
    /// Average pitch accuracy
    pub pitch_accuracy: f32,
    /// Vibrato consistency
    pub vibrato_consistency: f32,
    /// Breath quality
    pub breath_quality: f32,
    /// Timing accuracy
    pub timing_accuracy: f32,
    /// Expression consistency
    pub expression_consistency: f32,
    /// Formant quality
    pub formant_quality: f32,
    /// Spectral quality
    pub spectral_quality: f32,
    /// Overall quality score
    pub overall_quality: f32,
}

impl Default for QualitySettings {
    fn default() -> Self {
        Self {
            quality_level: 7,
            high_quality_pitch: true,
            advanced_vibrato: true,
            breath_modeling: true,
            formant_modeling: true,
            fft_size: 2048,
            hop_size: 512,
        }
    }
}

impl Default for SingingStats {
    fn default() -> Self {
        Self {
            total_notes: 0,
            processing_time: Duration::from_secs(0),
            pitch_accuracy: 0.0,
            vibrato_consistency: 0.0,
            breath_quality: 0.0,
            timing_accuracy: 0.0,
            expression_consistency: 0.0,
            formant_quality: 0.0,
            spectral_quality: 0.0,
            overall_quality: 0.0,
        }
    }
}
