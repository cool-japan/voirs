//! Machine Learning Enhancement Module
//!
//! This module provides advanced ML-based audio enhancement capabilities
//! including neural enhancement models, artifact removal, bandwidth extension,
//! and quality upsampling.

#[allow(unused_imports)]
use crate::VocoderError;
use crate::{AudioBuffer, Result};
use async_trait::async_trait;
use std::collections::HashMap;

pub mod artifact_removal;
pub mod bandwidth_extension;
pub mod neural_enhancement;
pub mod quality_upsampling;

/// Configuration for ML enhancement operations
#[derive(Debug, Clone)]
pub struct MLEnhancementConfig {
    /// Enhancement strength (0.0 to 1.0)
    pub strength: f32,
    /// Target quality level
    pub target_quality: QualityLevel,
    /// Whether to preserve original dynamics
    pub preserve_dynamics: bool,
    /// Processing mode
    pub mode: ProcessingMode,
    /// Additional model-specific parameters
    pub model_params: HashMap<String, f32>,
}

impl Default for MLEnhancementConfig {
    fn default() -> Self {
        let mut model_params = HashMap::new();
        model_params.insert("noise_reduction".to_string(), 0.3);
        model_params.insert("harmonic_enhancement".to_string(), 0.2);
        model_params.insert("transient_preservation".to_string(), 0.8);

        Self {
            strength: 0.5,
            target_quality: QualityLevel::High,
            preserve_dynamics: true,
            mode: ProcessingMode::Balanced,
            model_params,
        }
    }
}

/// Quality levels for enhancement
#[derive(Debug, Clone, PartialEq)]
pub enum QualityLevel {
    Low,
    Medium,
    High,
    Ultra,
}

/// Processing modes for different use cases
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingMode {
    /// Balance between quality and performance
    Balanced,
    /// Prioritize audio quality
    Quality,
    /// Prioritize processing speed
    Speed,
    /// Preserve original characteristics
    Conservative,
    /// Aggressive enhancement
    Aggressive,
}

/// ML enhancement statistics
#[derive(Debug, Clone, Default)]
pub struct EnhancementStats {
    /// Samples processed
    pub samples_processed: u64,
    /// Average enhancement applied (0.0 to 1.0)
    pub avg_enhancement: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Model confidence score
    pub confidence_score: f32,
    /// Artifacts detected and removed
    pub artifacts_removed: u32,
    /// Quality improvement estimate
    pub quality_improvement: f32,
}

/// Trait for ML-based audio enhancement
#[async_trait]
pub trait MLEnhancer: Send + Sync {
    /// Enhance audio using ML models
    async fn enhance(
        &self,
        audio: &AudioBuffer,
        config: &MLEnhancementConfig,
    ) -> Result<AudioBuffer>;

    /// Enhance audio in place for memory efficiency
    async fn enhance_inplace(
        &self,
        audio: &mut AudioBuffer,
        config: &MLEnhancementConfig,
    ) -> Result<()>;

    /// Batch enhancement for multiple audio buffers
    async fn enhance_batch(
        &self,
        audios: &[AudioBuffer],
        configs: Option<&[MLEnhancementConfig]>,
    ) -> Result<Vec<AudioBuffer>>;

    /// Get enhancement statistics from last operation
    fn get_stats(&self) -> EnhancementStats;

    /// Check if the enhancer is ready (model loaded, etc.)
    fn is_ready(&self) -> bool;

    /// Get supported quality levels
    fn supported_quality_levels(&self) -> Vec<QualityLevel>;

    /// Get enhancer metadata
    fn metadata(&self) -> EnhancerMetadata;
}

/// Metadata about an ML enhancer
#[derive(Debug, Clone)]
pub struct EnhancerMetadata {
    /// Enhancer name
    pub name: String,
    /// Version
    pub version: String,
    /// Supported sample rates
    pub supported_sample_rates: Vec<u32>,
    /// Maximum input duration in seconds
    pub max_duration: Option<f32>,
    /// Memory requirements in MB
    pub memory_requirements: u32,
    /// Real-time factor
    pub rtf: f32,
    /// Model size in MB
    pub model_size: f32,
}

/// Factory for creating ML enhancers
pub struct MLEnhancerFactory;

impl MLEnhancerFactory {
    /// Create a neural enhancement model
    pub fn create_neural_enhancer() -> Result<Box<dyn MLEnhancer>> {
        Ok(Box::new(neural_enhancement::NeuralEnhancer::new()?))
    }

    /// Create an artifact removal model
    pub fn create_artifact_remover() -> Result<Box<dyn MLEnhancer>> {
        Ok(Box::new(artifact_removal::ArtifactRemover::new()?))
    }

    /// Create a bandwidth extension model
    pub fn create_bandwidth_extender() -> Result<Box<dyn MLEnhancer>> {
        Ok(Box::new(bandwidth_extension::BandwidthExtender::new()?))
    }

    /// Create a quality upsampling model
    pub fn create_quality_upsampler() -> Result<Box<dyn MLEnhancer>> {
        Ok(Box::new(quality_upsampling::QualityUpsampler::new()?))
    }

    /// Create a composite enhancer that combines multiple models
    pub fn create_composite_enhancer(enhancers: Vec<Box<dyn MLEnhancer>>) -> CompositeEnhancer {
        CompositeEnhancer::new(enhancers)
    }
}

/// Composite enhancer that chains multiple enhancement models
pub struct CompositeEnhancer {
    enhancers: Vec<Box<dyn MLEnhancer>>,
    stats: EnhancementStats,
}

impl CompositeEnhancer {
    /// Create a new composite enhancer
    pub fn new(enhancers: Vec<Box<dyn MLEnhancer>>) -> Self {
        Self {
            enhancers,
            stats: EnhancementStats::default(),
        }
    }

    /// Add an enhancer to the chain
    pub fn add_enhancer(&mut self, enhancer: Box<dyn MLEnhancer>) {
        self.enhancers.push(enhancer);
    }

    /// Get the number of enhancers in the chain
    pub fn len(&self) -> usize {
        self.enhancers.len()
    }

    /// Check if the chain is empty
    pub fn is_empty(&self) -> bool {
        self.enhancers.is_empty()
    }
}

#[async_trait]
impl MLEnhancer for CompositeEnhancer {
    async fn enhance(
        &self,
        audio: &AudioBuffer,
        config: &MLEnhancementConfig,
    ) -> Result<AudioBuffer> {
        let mut enhanced = audio.clone();

        for enhancer in &self.enhancers {
            enhanced = enhancer.enhance(&enhanced, config).await?;
        }

        Ok(enhanced)
    }

    async fn enhance_inplace(
        &self,
        audio: &mut AudioBuffer,
        config: &MLEnhancementConfig,
    ) -> Result<()> {
        for enhancer in &self.enhancers {
            enhancer.enhance_inplace(audio, config).await?;
        }
        Ok(())
    }

    async fn enhance_batch(
        &self,
        audios: &[AudioBuffer],
        configs: Option<&[MLEnhancementConfig]>,
    ) -> Result<Vec<AudioBuffer>> {
        let mut results = audios.to_vec();

        for enhancer in &self.enhancers {
            results = enhancer.enhance_batch(&results, configs).await?;
        }

        Ok(results)
    }

    fn get_stats(&self) -> EnhancementStats {
        self.stats.clone()
    }

    fn is_ready(&self) -> bool {
        self.enhancers.iter().all(|e| e.is_ready())
    }

    fn supported_quality_levels(&self) -> Vec<QualityLevel> {
        // Return the intersection of all supported quality levels
        if self.enhancers.is_empty() {
            return vec![];
        }

        let mut supported = self.enhancers[0].supported_quality_levels();
        for enhancer in &self.enhancers[1..] {
            let other_supported = enhancer.supported_quality_levels();
            supported.retain(|level| other_supported.contains(level));
        }

        supported
    }

    fn metadata(&self) -> EnhancerMetadata {
        EnhancerMetadata {
            name: "Composite Enhancer".to_string(),
            version: "1.0.0".to_string(),
            supported_sample_rates: vec![16000, 22050, 44100, 48000],
            max_duration: None,
            memory_requirements: self
                .enhancers
                .iter()
                .map(|e| e.metadata().memory_requirements)
                .sum(),
            rtf: self.enhancers.iter().map(|e| e.metadata().rtf).sum::<f32>(),
            model_size: self.enhancers.iter().map(|e| e.metadata().model_size).sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_enhancement_config_default() {
        let config = MLEnhancementConfig::default();
        assert_eq!(config.strength, 0.5);
        assert_eq!(config.target_quality, QualityLevel::High);
        assert!(config.preserve_dynamics);
        assert_eq!(config.mode, ProcessingMode::Balanced);
        assert!(!config.model_params.is_empty());
    }

    #[test]
    fn test_quality_levels() {
        let levels = vec![
            QualityLevel::Low,
            QualityLevel::Medium,
            QualityLevel::High,
            QualityLevel::Ultra,
        ];

        for level in levels {
            // Test that quality levels are properly comparable
            assert_eq!(level.clone(), level);
        }
    }

    #[test]
    fn test_processing_modes() {
        let modes = vec![
            ProcessingMode::Balanced,
            ProcessingMode::Quality,
            ProcessingMode::Speed,
            ProcessingMode::Conservative,
            ProcessingMode::Aggressive,
        ];

        for mode in modes {
            // Test that processing modes are properly comparable
            assert_eq!(mode.clone(), mode);
        }
    }

    #[test]
    fn test_enhancement_stats_default() {
        let stats = EnhancementStats::default();
        assert_eq!(stats.samples_processed, 0);
        assert_eq!(stats.avg_enhancement, 0.0);
        assert_eq!(stats.processing_time_ms, 0.0);
        assert_eq!(stats.confidence_score, 0.0);
        assert_eq!(stats.artifacts_removed, 0);
        assert_eq!(stats.quality_improvement, 0.0);
    }

    #[test]
    fn test_composite_enhancer_creation() {
        let enhancers: Vec<Box<dyn MLEnhancer>> = vec![];
        let composite = CompositeEnhancer::new(enhancers);

        assert_eq!(composite.len(), 0);
        assert!(composite.is_empty());
        assert!(composite.is_ready());
    }
}
