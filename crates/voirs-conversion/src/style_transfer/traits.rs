//! Traits for style transfer components

use crate::Result;

/// Content encoder trait
pub trait ContentEncoder: Send + Sync {
    /// Encode content from audio
    fn encode_content(&self, audio: &[f32], sample_rate: u32) -> Result<ContentRepresentation>;

    /// Get content dimension
    fn content_dim(&self) -> usize;
}

/// Style encoder trait
pub trait StyleEncoderTrait: Send + Sync {
    /// Encode style from audio
    fn encode_style(&self, audio: &[f32], sample_rate: u32) -> Result<StyleRepresentation>;

    /// Get style dimension
    fn style_dim(&self) -> usize;
}

/// Style extractor trait
pub trait StyleExtractorTrait: Send + Sync {
    /// Extract style features
    fn extract_features(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>>;

    /// Get feature dimension
    fn feature_dim(&self) -> usize;
}

/// Embedding network trait
pub trait EmbeddingNetwork: Send + Sync {
    /// Compute embedding from features
    fn compute_embedding(&self, features: &[f32]) -> Result<Vec<f32>>;

    /// Get embedding dimension
    fn embedding_dim(&self) -> usize;
}

/// Style decoder trait
pub trait StyleDecoderTrait: Send + Sync {
    /// Decode style into audio
    fn decode(
        &self,
        content: &ContentRepresentation,
        style: &StyleRepresentation,
        sample_rate: u32,
    ) -> Result<Vec<f32>>;

    /// Get supported synthesis method
    fn method(&self) -> SynthesisMethod;
}

/// Synthesis network trait
pub trait SynthesisNetwork: Send + Sync {
    /// Synthesize audio from content and style
    fn synthesize(
        &self,
        content: &ContentRepresentation,
        style: &StyleRepresentation,
        sample_rate: u32,
    ) -> Result<Vec<f32>>;

    /// Get synthesis method
    fn method(&self) -> SynthesisMethod;
}

/// Style quality metric trait
pub trait StyleQualityMetric: Send + Sync {
    /// Assess style transfer quality
    fn assess(
        &self,
        source_audio: &[f32],
        transferred_audio: &[f32],
        target_style: &StyleRepresentation,
        sample_rate: u32,
    ) -> Result<f32>;

    /// Get metric name
    fn name(&self) -> &str;
}

// Forward declarations for types used in traits
// These are defined in components.rs
use super::components::{ContentRepresentation, StyleRepresentation};
use super::config::SynthesisMethod;
