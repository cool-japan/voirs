//! # QualityConfig - Trait Implementations
//!
//! This module contains trait implementations for `QualityConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `StandardConfig`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{MethodWeights, QualityConfig, RealtimeAssessmentConfig};

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            perceptual_assessment: true,
            spectral_analysis: true,
            temporal_analysis: true,
            artifact_detection: true,
            embedding_similarity: true,
            quality_threshold: 0.7,
            similarity_threshold: 0.8,
            analysis_window_size: 1024,
            analysis_hop_size: 512,
            enable_caching: true,
            method_weights: MethodWeights::default(),
            realtime_settings: RealtimeAssessmentConfig::default(),
        }
    }
}

impl crate::api_standards::StandardConfig for QualityConfig {
    fn validate(&self) -> crate::Result<()> {
        use crate::api_standards::error_patterns::*;
        validate_range("quality_threshold", self.quality_threshold, 0.0, 1.0)?;
        validate_range("similarity_threshold", self.similarity_threshold, 0.0, 1.0)?;
        validate_positive("analysis_window_size", self.analysis_window_size)?;
        validate_positive("analysis_hop_size", self.analysis_hop_size)?;
        if self.analysis_hop_size > self.analysis_window_size {
            return Err(crate::Error::Validation(
                "analysis_hop_size cannot be larger than analysis_window_size".to_string(),
            ));
        }
        Ok(())
    }
    fn name(&self) -> &'static str {
        "QualityConfig"
    }
    fn version(&self) -> &'static str {
        "1.2.0"
    }
    fn merge_with(&mut self, other: &Self) -> crate::Result<()> {
        if other.quality_threshold != 0.7 {
            self.quality_threshold = other.quality_threshold;
        }
        if other.similarity_threshold != 0.8 {
            self.similarity_threshold = other.similarity_threshold;
        }
        if other.analysis_window_size != 1024 {
            self.analysis_window_size = other.analysis_window_size;
        }
        if other.analysis_hop_size != 512 {
            self.analysis_hop_size = other.analysis_hop_size;
        }
        self.validate()
    }
}
