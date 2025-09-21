//! Machine Learning integration for voirs-dataset
//!
//! This module provides advanced ML features including feature learning,
//! active learning, and domain adaptation for enhanced dataset processing.

pub mod active;
pub mod domain;
pub mod features;

// Re-export commonly used types
pub use active::{ActiveLearner, SamplingStrategy, UncertaintyMetric};
pub use domain::{AdaptationStrategy, DomainAdapter, DomainShift};
pub use features::{FeatureConfig, FeatureLearner, LearnedFeatures};
