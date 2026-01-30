//! Type definitions for audio quality analysis results

use serde::{Deserialize, Serialize};

/// Comprehensive audio quality analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveQualityAnalysis {
    /// Overall perceptual quality score (0.0-1.0)
    pub perceptual_quality: f32,
    /// Neural network quality prediction (0.0-1.0)
    pub neural_prediction: f32,
    /// PESQ-style score (1.0-5.0)
    pub pesq_score: f32,
    /// STOI-style intelligibility score (0.0-1.0)
    pub stoi_score: f32,
    /// PEMO-Q perceptual score (0.0-1.0)
    pub pemo_q_score: f32,
    /// Advanced spectral analysis
    pub spectral_analysis: SpectralQualityAnalysis,
    /// Temporal quality analysis
    pub temporal_analysis: TemporalQualityAnalysis,
    /// Psychoacoustic analysis
    pub psychoacoustic_analysis: PsychoacousticAnalysis,
    /// Multi-dimensional quality metrics
    pub multidimensional_quality: MultidimensionalQuality,
    /// Analysis statistics
    pub analysis_stats: AnalysisStatistics,
}

/// Advanced spectral quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralQualityAnalysis {
    /// Spectral distortion measure
    pub spectral_distortion: f32,
    /// Cepstral distance
    pub cepstral_distance: f32,
    /// Log spectral distance
    pub log_spectral_distance: f32,
    /// Itakura-Saito distortion
    pub itakura_saito_distortion: f32,
    /// Spectral correlation
    pub spectral_correlation: f32,
    /// Spectral flatness deviation
    pub spectral_flatness_deviation: f32,
    /// Harmonic distortion analysis
    pub harmonic_distortion: HarmonicDistortionAnalysis,
}

/// Harmonic distortion analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicDistortionAnalysis {
    /// Total harmonic distortion
    pub thd: f32,
    /// Individual harmonic ratios (up to 10th harmonic)
    pub harmonic_ratios: Vec<f32>,
    /// Intermodulation distortion
    pub intermodulation_distortion: f32,
    /// Harmonic-to-noise ratio
    pub harmonic_to_noise_ratio: f32,
}

/// Temporal quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalQualityAnalysis {
    /// Temporal coherence score
    pub temporal_coherence: f32,
    /// Envelope correlation
    pub envelope_correlation: f32,
    /// Zero-crossing rate deviation
    pub zcr_deviation: f32,
    /// Temporal smoothness
    pub temporal_smoothness: f32,
    /// Phase coherence
    pub phase_coherence: f32,
    /// Rhythm preservation
    pub rhythm_preservation: f32,
}

/// Advanced psychoacoustic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsychoacousticAnalysis {
    /// Perceptual loudness deviation
    pub loudness_deviation: f32,
    /// Critical band analysis
    pub critical_band_analysis: ResearchCriticalBandAnalysis,
    /// Masking threshold deviation
    pub masking_threshold_deviation: f32,
    /// Sharpness deviation
    pub sharpness_deviation: f32,
    /// Roughness measure
    pub roughness: f32,
    /// Fluctuation strength
    pub fluctuation_strength: f32,
    /// Tonality analysis
    pub tonality: TonalityAnalysis,
}

/// Critical band analysis for psychoacoustic assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchCriticalBandAnalysis {
    /// Per-band energy deviations (24 Bark bands)
    pub band_deviations: Vec<f32>,
    /// Overall band distortion
    pub overall_distortion: f32,
    /// High-frequency content preservation
    pub hf_preservation: f32,
    /// Low-frequency content preservation
    pub lf_preservation: f32,
}

/// Tonality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TonalityAnalysis {
    /// Tonal vs noise component ratio
    pub tonal_noise_ratio: f32,
    /// Tonal component preservation
    pub tonal_preservation: f32,
    /// Noise component deviation
    pub noise_deviation: f32,
    /// Spectral peaks preservation
    pub spectral_peaks_preservation: f32,
}

/// Multi-dimensional quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultidimensionalQuality {
    /// Naturalness dimension (0.0-1.0)
    pub naturalness: f32,
    /// Clarity dimension (0.0-1.0)
    pub clarity: f32,
    /// Pleasantness dimension (0.0-1.0)
    pub pleasantness: f32,
    /// Intelligibility dimension (0.0-1.0)
    pub intelligibility: f32,
    /// Spaciousness dimension (0.0-1.0)
    pub spaciousness: f32,
    /// Warmth dimension (0.0-1.0)
    pub warmth: f32,
    /// Brightness dimension (0.0-1.0)
    pub brightness: f32,
    /// Presence dimension (0.0-1.0)
    pub presence: f32,
}

/// Analysis processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisStatistics {
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Number of frames analyzed
    pub frames_analyzed: usize,
    /// Sample rate used
    pub sample_rate: u32,
    /// Audio duration in seconds
    pub duration_seconds: f32,
    /// Analysis algorithms used
    pub algorithms_used: Vec<String>,
}
