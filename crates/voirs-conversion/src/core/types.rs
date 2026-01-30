//! Supporting types for voice conversion

use crate::{config::ConversionConfig, Result};

use super::converter::VoiceConverter;

/// Builder for VoiceConverter
#[derive(Debug)]
pub struct VoiceConverterBuilder {
    config: ConversionConfig,
}

impl VoiceConverterBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: ConversionConfig::default(),
        }
    }

    /// Set config
    pub fn config(mut self, config: ConversionConfig) -> Self {
        self.config = config;
        self
    }

    /// Build converter
    pub fn build(self) -> Result<VoiceConverter> {
        VoiceConverter::with_config(self.config)
    }
}

impl Default for VoiceConverterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Audio features extracted for processing
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    /// Spectral features (MFCCs, spectral centroid, etc.)
    pub spectral: Vec<f32>,
    /// Temporal features (energy, ZCR, etc.)
    pub temporal: Vec<f32>,
    /// Prosodic features (F0, intensity, timing)
    pub prosodic: Vec<f32>,
    /// Speaker embedding (if available)
    pub speaker_embedding: Option<Vec<f32>>,
    /// Voice quality features (breathiness, roughness, etc.)
    pub quality: Vec<f32>,
    /// Formant frequencies
    pub formants: Vec<f32>,
    /// Harmonic features
    pub harmonics: Vec<f32>,
}

impl AudioFeatures {
    /// Create new audio features
    pub fn new(spectral: Vec<f32>, temporal: Vec<f32>, prosodic: Vec<f32>) -> Self {
        Self {
            spectral,
            temporal,
            prosodic,
            speaker_embedding: None,
            quality: Vec::new(),
            formants: Vec::new(),
            harmonics: Vec::new(),
        }
    }

    /// Add speaker embedding
    pub fn with_speaker_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.speaker_embedding = Some(embedding);
        self
    }

    /// Add voice quality features
    pub fn with_quality_features(mut self, quality: Vec<f32>) -> Self {
        self.quality = quality;
        self
    }

    /// Add formant frequencies
    pub fn with_formants(mut self, formants: Vec<f32>) -> Self {
        self.formants = formants;
        self
    }

    /// Add harmonic features
    pub fn with_harmonics(mut self, harmonics: Vec<f32>) -> Self {
        self.harmonics = harmonics;
        self
    }

    /// Calculate similarity to another feature set
    pub fn similarity(&self, other: &AudioFeatures) -> f32 {
        let mut similarity_scores = Vec::new();

        // Spectral similarity
        if !self.spectral.is_empty() && !other.spectral.is_empty() {
            let spectral_sim = self.cosine_similarity(&self.spectral, &other.spectral);
            similarity_scores.push(spectral_sim * 0.3); // Weight: 30%
        }

        // Prosodic similarity
        if !self.prosodic.is_empty() && !other.prosodic.is_empty() {
            let prosodic_sim = self.cosine_similarity(&self.prosodic, &other.prosodic);
            similarity_scores.push(prosodic_sim * 0.3); // Weight: 30%
        }

        // Speaker embedding similarity
        if let (Some(emb1), Some(emb2)) = (&self.speaker_embedding, &other.speaker_embedding) {
            let speaker_sim = self.cosine_similarity(emb1, emb2);
            similarity_scores.push(speaker_sim * 0.4); // Weight: 40%
        }

        // Quality similarity
        if !self.quality.is_empty() && !other.quality.is_empty() {
            let quality_sim = self.cosine_similarity(&self.quality, &other.quality);
            similarity_scores.push(quality_sim * 0.1); // Weight: 10%
        }

        if similarity_scores.is_empty() {
            0.0
        } else {
            similarity_scores.iter().sum::<f32>() / similarity_scores.len() as f32
        }
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let min_len = a.len().min(b.len());
        if min_len == 0 {
            return 0.0;
        }

        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for i in 0..min_len {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a.sqrt() * norm_b.sqrt())
    }

    /// Extract speaker identity features
    pub fn extract_speaker_identity(&self) -> Vec<f32> {
        let mut identity_features = Vec::new();

        // Use speaker embedding if available
        if let Some(embedding) = &self.speaker_embedding {
            identity_features.extend_from_slice(embedding);
        } else {
            // Combine other features for speaker identity
            if !self.spectral.is_empty() {
                identity_features.extend_from_slice(&self.spectral[0..self.spectral.len().min(13)]);
                // First 13 MFCCs
            }
            if !self.prosodic.is_empty() {
                identity_features.extend_from_slice(&self.prosodic[0..self.prosodic.len().min(4)]);
                // F0 stats
            }
            if !self.formants.is_empty() {
                identity_features.extend_from_slice(&self.formants); // Formant frequencies
            }
        }

        identity_features
    }
}

/// Quality metrics for conversion results
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Similarity to source (0.0 to 1.0)
    pub similarity: f32,
    /// Naturalness of converted audio (0.0 to 1.0)
    pub naturalness: f32,
    /// Strength of conversion applied (0.0 to 1.0)
    pub conversion_strength: f32,
}

/// Statistics about the voice converter
#[derive(Debug, Clone)]
pub struct ConversionStats {
    /// Number of loaded models
    pub loaded_models: usize,
    /// Number of cached voice characteristics
    pub cached_voices: usize,
    /// Device being used for processing
    pub device: String,
    /// Current configuration
    pub config: ConversionConfig,
}
