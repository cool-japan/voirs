//! Speaker embedding extraction and management
//!
//! This module provides comprehensive speaker embedding functionality for voice cloning,
//! including extraction, adaptation, and analysis of speaker characteristics.
//!
//! ## Module Structure
//!
//! - `types`: Type definitions for embeddings, configurations, and results
//! - `speaker_embedding`: Core SpeakerEmbedding implementation
//! - `impls`: SpeakerEmbeddingExtractor and FeatureExtractor implementations
//! - `network`: Neural network implementation
//! - `defaults`: Default trait implementations for configuration types

mod defaults;
mod impls;
mod network;
pub mod simd_ops; // Public module for testing and external use
mod speaker_embedding;
mod types;

// Re-export all public types
pub use types::*;

// Re-export SIMD operations for high-performance embedding operations
pub use simd_ops::{
    simd_cosine_similarity, simd_dot_product, simd_euclidean_distance, simd_l2_norm,
    simd_normalize_inplace, simd_weighted_average,
};
