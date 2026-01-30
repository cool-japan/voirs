//! SpeakerEmbedding implementation with optimized operations
//!
//! This module provides high-performance speaker embedding operations using
//! optimized ndarray operations from scirs2-core for critical computational paths.

use super::types::*;
use scirs2_core::ndarray::{Array1, ArrayView1};

impl SpeakerEmbedding {
    /// Create new embedding
    pub fn new(vector: Vec<f32>) -> Self {
        let dimension = vector.len();
        Self {
            vector,
            dimension,
            confidence: 1.0,
            metadata: EmbeddingMetadata::default(),
        }
    }

    /// Create embedding with metadata
    pub fn with_metadata(vector: Vec<f32>, metadata: EmbeddingMetadata) -> Self {
        let dimension = vector.len();
        Self {
            vector,
            dimension,
            confidence: 1.0,
            metadata,
        }
    }

    /// Calculate cosine similarity to another embedding (Optimized with SIMD)
    ///
    /// Uses highly optimized SIMD operations from scirs2-core for maximum performance.
    /// This is a critical path that's called frequently during voice cloning.
    ///
    /// # Performance
    /// This method uses direct SIMD operations for up to 8x speedup on modern CPUs
    /// with AVX2/AVX512 (x86_64) or NEON (ARM) support.
    pub fn similarity(&self, other: &Self) -> f32 {
        // Use direct SIMD operations for maximum performance
        super::simd_ops::simd_cosine_similarity(&self.vector, &other.vector)
    }

    /// Calculate Euclidean distance to another embedding (Optimized with SIMD)
    ///
    /// Uses highly optimized SIMD operations from scirs2-core for maximum performance.
    ///
    /// # Performance
    /// This method uses direct SIMD operations for up to 8x speedup on modern CPUs.
    pub fn distance(&self, other: &Self) -> f32 {
        // Use direct SIMD operations for maximum performance
        super::simd_ops::simd_euclidean_distance(&self.vector, &other.vector)
    }

    /// L2 normalize the embedding (Optimized with SIMD)
    ///
    /// Uses highly optimized SIMD operations from scirs2-core for maximum performance.
    ///
    /// # Performance
    /// This method uses direct SIMD operations for up to 8x speedup on modern CPUs.
    pub fn normalize(&mut self) {
        // Use direct SIMD operations for maximum performance
        super::simd_ops::simd_normalize_inplace(&mut self.vector);
    }

    /// Check if embedding is valid
    pub fn is_valid(&self) -> bool {
        !self.vector.is_empty()
            && self.vector.iter().all(|x| x.is_finite())
            && self.confidence >= 0.0
            && self.confidence <= 1.0
    }

    /// Get embedding quality score based on confidence and voice quality
    pub fn quality_score(&self) -> f32 {
        let voice_quality_score = self.metadata.voice_quality.overall_quality();
        (self.confidence + voice_quality_score) / 2.0
    }
}

impl VoiceQuality {
    /// Compute overall quality score
    pub fn overall_quality(&self) -> f32 {
        let f0_stability = 1.0 / (1.0 + self.jitter * 1000.0);
        let amplitude_stability = 1.0 / (1.0 + self.shimmer * 100.0);
        let energy_consistency = if self.energy_std > 0.0 {
            self.energy_mean / (self.energy_mean + self.energy_std)
        } else {
            1.0
        };

        (f0_stability + amplitude_stability + energy_consistency) / 3.0
    }
}
