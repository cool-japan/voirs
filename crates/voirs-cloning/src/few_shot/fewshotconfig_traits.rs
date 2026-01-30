//! # FewShotConfig - Trait Implementations
//!
//! This module contains trait implementations for `FewShotConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for FewShotConfig {
    fn default() -> Self {
        Self {
            num_shots: 5,
            num_queries: 3,
            embedding_dim: 256,
            meta_hidden_dims: vec![512, 256, 128],
            meta_learning_rate: 0.001,
            adaptation_learning_rate: 0.01,
            meta_episodes: 1000,
            adaptation_steps: 5,
            prototype_temperature: 0.1,
            quality_threshold: 0.7,
            use_quality_weighting: true,
            enable_cross_lingual: false,
            distance_metric: DistanceMetric::Cosine,
            meta_algorithm: MetaLearningAlgorithm::MAML,
        }
    }
}
