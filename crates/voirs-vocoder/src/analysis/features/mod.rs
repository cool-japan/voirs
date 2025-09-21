//! Feature extraction for machine learning and audio analysis
//!
//! Provides comprehensive feature extraction including:
//! - MFCC (Mel-frequency cepstral coefficients)
//! - Mel-scale features
//! - Chroma features
//! - Spectral features
//! - Temporal features
//! - Perceptual features

use crate::{Result, VocoderError};
use ndarray::{Array1, Array2};

// Module declarations
pub mod spectral;
pub mod temporal;
pub mod perceptual;
pub mod rhythm;
pub mod timbral;
pub mod harmonic;
pub mod mfcc;
pub mod chroma;
pub mod filterbanks;

// Re-exports
pub use spectral::*;
pub use temporal::*;
pub use perceptual::*;
pub use rhythm::*;
pub use timbral::*;
pub use harmonic::*;
pub use mfcc::*;
pub use chroma::*;
pub use filterbanks::*;

/// Comprehensive feature set for machine learning
#[derive(Debug, Clone)]
pub struct FeatureSet {
    /// MFCC coefficients
    pub mfcc: Array2<f32>,
    
    /// Mel-scale spectrogram
    pub mel_spectrogram: Array2<f32>,
    
    /// Chroma features (12-dimensional)
    pub chroma: Array2<f32>,
    
    /// Spectral features over time
    pub spectral_features: SpectralFeatureMatrix,
    
    /// Temporal features
    pub temporal_features: TemporalFeatureVector,
    
    /// Perceptual features
    pub perceptual_features: PerceptualFeatureVector,
    
    /// Rhythm and beat features
    pub rhythm_features: RhythmFeatureVector,
    
    /// Timbral features
    pub timbral_features: TimbralFeatureVector,
    
    /// Harmonic features
    pub harmonic_features: HarmonicFeatureVector,
}

impl Default for FeatureSet {
    fn default() -> Self {
        Self {
            mfcc: Array2::zeros((0, 0)),
            mel_spectrogram: Array2::zeros((0, 0)),
            chroma: Array2::zeros((0, 0)),
            spectral_features: SpectralFeatureMatrix::default(),
            temporal_features: TemporalFeatureVector::default(),
            perceptual_features: PerceptualFeatureVector::default(),
            rhythm_features: RhythmFeatureVector::default(),
            timbral_features: TimbralFeatureVector::default(),
            harmonic_features: HarmonicFeatureVector::default(),
        }
    }
}

/// Feature extraction configuration
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Number of MFCC coefficients
    pub n_mfcc: usize,
    
    /// Number of mel filter banks
    pub n_mels: usize,
    
    /// FFT window size
    pub n_fft: usize,
    
    /// Hop size for analysis
    pub hop_length: usize,
    
    /// Sample rate
    pub sample_rate: f32,
    
    /// Minimum frequency for mel filters
    pub fmin: f32,
    
    /// Maximum frequency for mel filters
    pub fmax: Option<f32>,
    
    /// Power for mel-scale computation
    pub power: f32,
    
    /// Pre-emphasis coefficient
    pub preemphasis: f32,
    
    /// Enable delta features
    pub delta: bool,
    
    /// Enable delta-delta features
    pub delta_delta: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            n_mfcc: 13,
            n_mels: 80,
            n_fft: 2048,
            hop_length: 512,
            sample_rate: 22050.0,
            fmin: 80.0,
            fmax: None,
            power: 2.0,
            preemphasis: 0.97,
            delta: false,
            delta_delta: false,
        }
    }
}

/// Main feature extractor
#[derive(Debug)]
pub struct FeatureExtractor {
    config: FeatureConfig,
    mel_filterbank: MelFilterbank,
    chroma_filterbank: ChromaFilterbank,
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new(config: FeatureConfig) -> Result<Self> {
        let mel_filterbank = MelFilterbank::new(
            config.n_fft / 2 + 1,
            config.n_mels,
            config.sample_rate,
            config.fmin,
            config.fmax.unwrap_or(config.sample_rate / 2.0),
        )?;
        
        let chroma_filterbank = ChromaFilterbank::new(
            config.n_fft / 2 + 1,
            config.sample_rate,
            config.fmin,
        )?;
        
        Ok(Self {
            config,
            mel_filterbank,
            chroma_filterbank,
        })
    }
    
    /// Extract all features from audio
    pub fn extract_features(&self, audio: &[f32]) -> Result<FeatureSet> {
        // Extract MFCC and mel spectrogram
        let (mfcc, mel_spectrogram) = self.compute_mfcc_and_mel(audio)?;
        
        // Extract chroma features
        let chroma = self.compute_chroma(audio)?;
        
        // Extract spectral features
        let spectral_features = self.compute_spectral_features(audio)?;
        
        // Extract temporal features
        let temporal_features = self.compute_temporal_features(audio)?;
        
        // Extract perceptual features
        let perceptual_features = self.compute_perceptual_features(audio)?;
        
        // Extract rhythm features
        let rhythm_features = self.compute_rhythm_features(audio)?;
        
        // Extract timbral features
        let timbral_features = self.compute_timbral_features(audio)?;
        
        // Extract harmonic features
        let harmonic_features = self.compute_harmonic_features(audio)?;
        
        Ok(FeatureSet {
            mfcc,
            mel_spectrogram,
            chroma,
            spectral_features,
            temporal_features,
            perceptual_features,
            rhythm_features,
            timbral_features,
            harmonic_features,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_config_default() {
        let config = FeatureConfig::default();
        assert_eq!(config.n_mfcc, 13);
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.n_fft, 2048);
        assert_eq!(config.hop_length, 512);
    }

    #[test]
    fn test_feature_set_default() {
        let features = FeatureSet::default();
        assert_eq!(features.mfcc.shape(), &[0, 0]);
        assert_eq!(features.mel_spectrogram.shape(), &[0, 0]);
        assert_eq!(features.chroma.shape(), &[0, 0]);
    }

    #[test]
    fn test_feature_extractor_creation() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);
        assert!(extractor.is_ok());
    }
}