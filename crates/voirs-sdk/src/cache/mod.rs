//! Comprehensive caching system for VoiRS SDK.
//!
//! This module provides a multi-layered caching system that includes:
//! - Model caching for efficient model loading and reuse
//! - Synthesis result caching for faster repeated synthesis
//! - Intelligent cache management and coordination
//! - Performance monitoring and health checking
//! - Automatic maintenance and optimization

pub mod distributed;
pub mod encryption;
pub mod management;
pub mod models;
pub mod results;

// Re-export main types for convenience
pub use distributed::{
    CacheNode, ConsistencyLevel, DistributedCacheClient, DistributedCacheCoordinator,
    DistributedCacheStats, InMemoryDistributedCache, NodeStatus,
};
pub use encryption::{CacheEncryption, EncryptedData, EncryptionConfig, EncryptionMetadata};
pub use management::{CacheManager, CacheManagerConfig, CombinedCacheStats};
pub use models::{
    AdvancedModelCache, CacheUsageSummary as ModelCacheUsageSummary, ModelCacheConfig,
    ModelMetadata, ModelPriority, ModelType,
};
pub use results::{
    CacheUsageSummary as ResultCacheUsageSummary, CachedSynthesisResult, QualityMetrics,
    ResultCacheConfig, SynthesisMetadata, SynthesisResultCache,
};

// Legacy compatibility exports
pub use management::CacheManager as CacheCoordinator;
pub use models::AdvancedModelCache as ModelCache;
pub use results::SynthesisResultCache as ResultCache;

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, time::SystemTime};

/// Convenience function to create a default cache manager
pub async fn create_default_cache_manager(cache_dir: impl Into<PathBuf>) -> Result<CacheManager> {
    let config = CacheManagerConfig::default();
    CacheManager::new(config, cache_dir.into()).await
}

/// Convenience function to create a cache manager with custom settings
pub async fn create_cache_manager(
    cache_dir: impl Into<PathBuf>,
    memory_limit_mb: usize,
    disk_limit_mb: usize,
) -> Result<CacheManager> {
    let mut config = CacheManagerConfig::default();

    // Update memory limits
    config.model_cache.memory_cache_size_mb = memory_limit_mb / 2;
    config.result_cache.memory_cache_size_mb = memory_limit_mb / 2;

    // Update disk limits
    config.model_cache.disk_cache_size_mb = disk_limit_mb / 2;
    config.result_cache.disk_cache_size_mb = disk_limit_mb / 2;

    // Update global limit
    config.global_settings.max_total_memory_mb = memory_limit_mb;

    CacheManager::new(config, cache_dir.into()).await
}

/// Cache configuration builder for easy setup
#[derive(Debug, Clone)]
pub struct CacheConfigBuilder {
    config: CacheManagerConfig,
}

impl CacheConfigBuilder {
    /// Create new config builder
    pub fn new() -> Self {
        Self {
            config: CacheManagerConfig::default(),
        }
    }

    /// Set memory limits
    pub fn memory_limits(mut self, model_cache_mb: usize, result_cache_mb: usize) -> Self {
        self.config.model_cache.memory_cache_size_mb = model_cache_mb;
        self.config.result_cache.memory_cache_size_mb = result_cache_mb;
        self.config.global_settings.max_total_memory_mb = model_cache_mb + result_cache_mb;
        self
    }

    /// Set disk limits
    pub fn disk_limits(mut self, model_cache_mb: usize, result_cache_mb: usize) -> Self {
        self.config.model_cache.disk_cache_size_mb = model_cache_mb;
        self.config.result_cache.disk_cache_size_mb = result_cache_mb;
        self
    }

    /// Enable/disable monitoring
    pub fn monitoring(mut self, enabled: bool) -> Self {
        self.config.monitoring.enable_health_monitoring = enabled;
        self.config.monitoring.enable_performance_monitoring = enabled;
        self
    }

    /// Enable/disable automatic maintenance
    pub fn auto_maintenance(mut self, enabled: bool) -> Self {
        self.config.maintenance.enable_auto_maintenance = enabled;
        self
    }

    /// Set cache warming
    pub fn cache_warming(mut self, enabled: bool) -> Self {
        self.config.global_settings.enable_startup_warming = enabled;
        self.config.model_cache.enable_cache_warming = enabled;
        self
    }

    /// Enable compression
    pub fn compression(mut self, enabled: bool) -> Self {
        self.config.global_settings.enable_compression = enabled;
        self.config.model_cache.enable_compression = enabled;
        self.config.result_cache.enable_compression = enabled;
        self
    }

    /// Set TTL for results
    pub fn result_ttl(mut self, seconds: u64) -> Self {
        self.config.result_cache.default_ttl_seconds = seconds;
        self
    }

    /// Set model TTL
    pub fn model_ttl(mut self, seconds: u64) -> Self {
        self.config.model_cache.model_ttl_seconds = seconds;
        self
    }

    /// Build the configuration
    pub fn build(self) -> CacheManagerConfig {
        self.config
    }

    /// Build and create cache manager
    pub async fn create_manager(self, cache_dir: impl Into<PathBuf>) -> Result<CacheManager> {
        CacheManager::new(self.config, cache_dir.into()).await
    }
}

impl Default for CacheConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSystemInfo {
    /// Cache system version
    pub version: String,

    /// Supported cache types
    pub supported_cache_types: Vec<String>,

    /// Available features
    pub features: Vec<String>,

    /// System capabilities
    pub capabilities: SystemCapabilities,

    /// Default configuration
    pub default_config: CacheManagerConfig,
}

/// System capabilities for caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemCapabilities {
    /// Maximum memory available for caching (MB)
    pub max_memory_mb: usize,

    /// Maximum disk space available for caching (MB)
    pub max_disk_mb: usize,

    /// Supports compression
    pub supports_compression: bool,

    /// Supports encryption
    pub supports_encryption: bool,

    /// Supports background tasks
    pub supports_background_tasks: bool,

    /// Supports distributed caching
    pub supports_distributed_caching: bool,
}

/// Get cache system information
pub fn get_system_info() -> CacheSystemInfo {
    CacheSystemInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        supported_cache_types: vec![
            "model".to_string(),
            "result".to_string(),
            "metadata".to_string(),
        ],
        features: vec![
            "lru_eviction".to_string(),
            "ttl_expiration".to_string(),
            "compression".to_string(),
            "health_monitoring".to_string(),
            "performance_metrics".to_string(),
            "background_maintenance".to_string(),
            "similarity_matching".to_string(),
            "cache_warming".to_string(),
        ],
        capabilities: SystemCapabilities {
            max_memory_mb: 8192,  // 8GB - could be detected dynamically
            max_disk_mb: 100_000, // 100GB - could be detected dynamically
            supports_compression: true,
            supports_encryption: true, // Cache encryption implemented
            supports_background_tasks: true,
            supports_distributed_caching: true, // Distributed caching support implemented
        },
        default_config: CacheManagerConfig::default(),
    }
}

/// Cache statistics summary for easy monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatsSummary {
    /// Model cache stats
    pub model_cache: SimpleCacheStats,

    /// Result cache stats
    pub result_cache: SimpleCacheStats,

    /// Global stats
    pub global: GlobalCacheStatsSummary,

    /// Health score (0.0-1.0)
    pub health_score: f64,

    /// Last updated
    pub last_updated: SystemTime,
}

/// Simplified cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleCacheStats {
    /// Number of entries
    pub entries: usize,

    /// Memory usage in MB
    pub memory_usage_mb: usize,

    /// Hit rate percentage
    pub hit_rate_percent: f64,

    /// Miss rate percentage
    pub miss_rate_percent: f64,
}

/// Global cache statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalCacheStatsSummary {
    /// Total memory usage in MB
    pub total_memory_mb: usize,

    /// Total entries across all caches
    pub total_entries: usize,

    /// Overall hit rate percentage
    pub overall_hit_rate_percent: f64,

    /// Memory utilization percentage
    pub memory_utilization_percent: f64,
}

/// Convert detailed stats to summary
impl From<CombinedCacheStats> for CacheStatsSummary {
    fn from(stats: CombinedCacheStats) -> Self {
        Self {
            model_cache: SimpleCacheStats {
                entries: stats.model_stats.basic_stats.total_entries,
                memory_usage_mb: stats.model_stats.basic_stats.memory_usage_bytes / (1024 * 1024),
                hit_rate_percent: stats.model_stats.basic_stats.hit_rate as f64,
                miss_rate_percent: stats.model_stats.basic_stats.miss_rate as f64,
            },
            result_cache: SimpleCacheStats {
                entries: stats.result_stats.basic_stats.total_entries,
                memory_usage_mb: stats.result_stats.basic_stats.memory_usage_bytes / (1024 * 1024),
                hit_rate_percent: stats.result_stats.basic_stats.hit_rate as f64,
                miss_rate_percent: stats.result_stats.basic_stats.miss_rate as f64,
            },
            global: GlobalCacheStatsSummary {
                total_memory_mb: stats.global_stats.total_memory_usage_mb,
                total_entries: stats.global_stats.total_entries,
                overall_hit_rate_percent: stats.global_stats.overall_hit_rate,
                memory_utilization_percent: if stats.global_stats.total_memory_usage_mb > 0 {
                    (stats.global_stats.total_memory_usage_mb as f64 / 2048.0) * 100.0
                // Assume 2GB limit
                } else {
                    0.0
                },
            },
            health_score: stats.health_metrics.overall_health_score,
            last_updated: stats.last_updated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_create_default_cache_manager() {
        let temp_dir = TempDir::new().unwrap();
        let manager = create_default_cache_manager(temp_dir.path()).await.unwrap();

        let stats = manager.get_combined_stats().await;
        assert_eq!(stats.global_stats.total_entries, 0);
    }

    #[tokio::test]
    async fn test_create_cache_manager_with_limits() {
        let temp_dir = TempDir::new().unwrap();
        let manager = create_cache_manager(temp_dir.path(), 1024, 2048)
            .await
            .unwrap();

        let stats = manager.get_combined_stats().await;
        assert_eq!(stats.global_stats.total_entries, 0);
    }

    #[tokio::test]
    async fn test_cache_config_builder() {
        let temp_dir = TempDir::new().unwrap();

        let manager = CacheConfigBuilder::new()
            .memory_limits(512, 512)
            .disk_limits(1024, 1024)
            .monitoring(false) // Disable for testing
            .auto_maintenance(false) // Disable for testing
            .cache_warming(false)
            .compression(true)
            .result_ttl(3600)
            .model_ttl(86400)
            .create_manager(temp_dir.path())
            .await
            .unwrap();

        let stats = manager.get_combined_stats().await;
        assert_eq!(stats.global_stats.total_entries, 0);
        manager.shutdown().await.unwrap();
    }

    #[test]
    fn test_get_system_info() {
        let info = get_system_info();
        assert!(!info.version.is_empty());
        assert!(!info.supported_cache_types.is_empty());
        assert!(!info.features.is_empty());
    }

    #[test]
    fn test_cache_stats_summary_conversion() {
        let detailed_stats = CombinedCacheStats::default();
        let summary: CacheStatsSummary = detailed_stats.into();

        assert_eq!(summary.model_cache.entries, 0);
        assert_eq!(summary.result_cache.entries, 0);
        assert_eq!(summary.global.total_entries, 0);
    }
}
