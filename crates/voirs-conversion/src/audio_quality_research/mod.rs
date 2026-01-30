//! Audio Quality Research - Advanced perceptual audio quality algorithms
//!
//! This module provides state-of-the-art perceptual audio quality algorithms
//! for research and development in voice conversion quality assessment.
//!
//! ## Features
//!
//! - **Perceptual Audio Codecs Metrics**: PEMO-Q, PESQ, STOI-based quality assessment
//! - **Psychoacoustic Modeling**: Advanced human auditory system modeling
//! - **Neural Quality Metrics**: AI-based quality prediction models
//! - **Spectral Quality Analysis**: Advanced spectral distortion measurements
//! - **Temporal Quality Assessment**: Time-domain quality analysis
//! - **Multi-dimensional Quality Spaces**: Quality assessment in multiple perceptual dimensions
//!
//! ## Example
//!
//! ```rust
//! use voirs_conversion::audio_quality_research::{AudioQualityResearcher, ResearchConfig};
//!
//! let config = ResearchConfig::default()
//!     .with_neural_models(true)
//!     .with_psychoacoustic_depth(5);
//!
//! let mut researcher = AudioQualityResearcher::new(config)?;
//!
//! let original = vec![0.1, 0.2, -0.1, 0.05]; // Original audio
//! let processed = vec![0.09, 0.19, -0.11, 0.04]; // Processed audio
//!
//! let quality_analysis = researcher.comprehensive_analysis(&original, &processed, 16000)?;
//!
//! println!("Perceptual Quality Score: {:.3}", quality_analysis.perceptual_quality);
//! println!("Neural Prediction: {:.3}", quality_analysis.neural_prediction);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Module declarations
mod config;
mod neural_model;
mod researcher;
mod types;

// Implementation modules (not publicly exposed)
mod analysis_methods;
mod helper_methods;
mod multidimensional_methods;
mod psychoacoustic_methods;
mod spectral_methods;
mod temporal_methods;

#[cfg(test)]
mod tests;

// Public re-exports
pub use config::ResearchConfig;
pub use neural_model::NeuralQualityModel;
pub use researcher::AudioQualityResearcher;
pub use types::{
    AnalysisStatistics, ComprehensiveQualityAnalysis, HarmonicDistortionAnalysis,
    MultidimensionalQuality, PsychoacousticAnalysis, ResearchCriticalBandAnalysis,
    SpectralQualityAnalysis, TemporalQualityAnalysis, TonalityAnalysis,
};
