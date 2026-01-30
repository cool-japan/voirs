//! # CloningQualityAssessor - Trait Implementations
//!
//! This module contains trait implementations for `CloningQualityAssessor`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//! - `Default`
//! - `StandardApiPattern`
//! - `StandardAsyncOperations`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{
    api_standards::StandardConfig,
    embedding::{SpeakerEmbedding, SpeakerEmbeddingExtractor},
    types::VoiceSample,
    Error, Result,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

use super::types::{AssessmentStats, CloningQualityAssessor, QualityConfig};

impl std::fmt::Debug for CloningQualityAssessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CloningQualityAssessor")
            .field("config", &self.config)
            .field("embedding_extractor", &self.embedding_extractor)
            .field("metrics_cache", &"<Arc<RwLock<HashMap>>>")
            .field("performance_stats", &"<Arc<RwLock<AssessmentStats>>>")
            .finish()
    }
}

impl Default for CloningQualityAssessor {
    fn default() -> Self {
        Self::new().expect("Failed to create default CloningQualityAssessor")
    }
}

impl crate::api_standards::StandardApiPattern for CloningQualityAssessor {
    type Config = QualityConfig;
    type Builder = ();
    fn new() -> Result<Self> {
        Self::with_config(QualityConfig::default())
    }
    fn with_config(config: Self::Config) -> Result<Self> {
        config.validate()?;
        let embedding_extractor = if config.embedding_similarity {
            Some(SpeakerEmbeddingExtractor::default())
        } else {
            None
        };
        Ok(Self {
            config,
            embedding_extractor,
            metrics_cache: Arc::new(RwLock::new(HashMap::new())),
            performance_stats: Arc::new(RwLock::new(AssessmentStats::new())),
        })
    }
    fn builder() -> Self::Builder {}
    fn get_config(&self) -> &Self::Config {
        &self.config
    }
    fn update_config(&mut self, config: Self::Config) -> Result<()> {
        config.validate()?;
        if config.embedding_similarity != self.config.embedding_similarity {
            self.embedding_extractor = if config.embedding_similarity {
                Some(SpeakerEmbeddingExtractor::default())
            } else {
                None
            };
        }
        self.config = config;
        Ok(())
    }
}

#[async_trait::async_trait]
impl crate::api_standards::StandardAsyncOperations for CloningQualityAssessor {
    async fn initialize(&mut self) -> Result<()> {
        if let Some(ref mut extractor) = self.embedding_extractor {
            extractor.initialize_network().await?;
        }
        info!("CloningQualityAssessor initialized successfully");
        Ok(())
    }
    async fn cleanup(&mut self) -> Result<()> {
        self.metrics_cache.write().await.clear();
        info!("CloningQualityAssessor cleaned up successfully");
        Ok(())
    }
    async fn health_check(&self) -> Result<crate::api_standards::ComponentHealth> {
        use std::collections::HashMap;
        let stats = self.performance_stats.read().await;
        let cache = self.metrics_cache.read().await;
        let mut metrics = HashMap::new();
        metrics.insert("cache_entries".to_string(), cache.len() as f64);
        metrics.insert(
            "total_assessments".to_string(),
            stats.total_assessments as f64,
        );
        metrics.insert("cache_hits".to_string(), stats.cache_hits as f64);
        metrics.insert("cache_misses".to_string(), stats.cache_misses as f64);
        let cache_hit_rate = if stats.total_assessments > 0 {
            stats.cache_hits as f64 / stats.total_assessments as f64
        } else {
            0.0
        };
        metrics.insert("cache_hit_rate".to_string(), cache_hit_rate);
        let is_healthy = self.embedding_extractor.is_some() || !self.config.embedding_similarity;
        let status_message = if is_healthy {
            format!(
                "Quality assessor healthy - {} assessments completed",
                stats.total_assessments
            )
        } else {
            "Quality assessor misconfigured - embedding similarity enabled but no extractor"
                .to_string()
        };
        Ok(crate::api_standards::ComponentHealth::healthy(&status_message).with_metrics(metrics))
    }
}
