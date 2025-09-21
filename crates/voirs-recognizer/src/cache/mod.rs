//! Advanced caching system with persistence and compression
//!
//! This module provides intelligent caching capabilities including:
//! - Persistent model weight caching with LRU eviction
//! - Feature cache for repeated inputs  
//! - Result caching with configurable invalidation
//! - Compression support for cache storage
//! - Multi-level cache hierarchy

use crate::RecognitionError;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;

pub mod persistent;
pub mod compression;
pub mod feature_cache;
pub mod result_cache;

/// Advanced cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Enable persistent caching to disk
    pub enable_persistence: bool,
    /// Cache directory path
    pub cache_dir: PathBuf,
    /// Maximum cache size in MB
    pub max_cache_size_mb: usize,
    /// Cache compression level (0-9)
    pub compression_level: u8,
    /// Cache entry TTL
    pub default_ttl: Duration,
    /// Enable LRU eviction
    pub enable_lru_eviction: bool,
    /// Maximum number of cached items
    pub max_items: usize,
    /// Enable feature caching
    pub enable_feature_cache: bool,
    /// Enable result caching
    pub enable_result_cache: bool,
    /// Background cleanup interval
    pub cleanup_interval: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_persistence: true,
            cache_dir: PathBuf::from("./cache"),
            max_cache_size_mb: 1024, // 1GB default
            compression_level: 6,
            default_ttl: Duration::from_secs(3600), // 1 hour
            enable_lru_eviction: true,
            max_items: 1000,
            enable_feature_cache: true,
            enable_result_cache: true,
            cleanup_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Multi-level cache manager
pub struct AdvancedCacheManager {
    /// Configuration
    config: CacheConfig,
    /// Persistent cache layer
    persistent_cache: Arc<RwLock<persistent::PersistentCache>>,
    /// Feature cache for audio processing
    feature_cache: Arc<RwLock<feature_cache::FeatureCache>>,
    /// Result cache for transcription results
    result_cache: Arc<RwLock<result_cache::ResultCache>>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
    /// Background cleanup task handle
    cleanup_handle: Option<tokio::task::JoinHandle<()>>,
}

/// Cache statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total cache size in bytes
    pub total_size_bytes: u64,
    /// Number of cached items
    pub item_count: usize,
    /// Cache hit rate
    pub hit_rate: f64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Last cleanup time
    pub last_cleanup: Option<Instant>,
    /// Eviction count
    pub eviction_count: u64,
}

impl AdvancedCacheManager {
    /// Create new advanced cache manager
    pub async fn new(config: CacheConfig) -> Result<Self, RecognitionError> {
        // Create cache directory if it doesn't exist
        if config.enable_persistence {
            tokio::fs::create_dir_all(&config.cache_dir).await
                .map_err(|e| RecognitionError::ResourceError {
                    message: format!("Failed to create cache directory: {}", e),
                    source: Some(Box::new(e)),
                })?;
        }

        let persistent_cache = Arc::new(RwLock::new(
            persistent::PersistentCache::new(config.clone()).await?
        ));
        
        let feature_cache = Arc::new(RwLock::new(
            feature_cache::FeatureCache::new(config.clone())
        ));
        
        let result_cache = Arc::new(RwLock::new(
            result_cache::ResultCache::new(config.clone())
        ));
        
        let stats = Arc::new(RwLock::new(CacheStats::default()));

        let mut manager = Self {
            config,
            persistent_cache,
            feature_cache,
            result_cache,
            stats,
            cleanup_handle: None,
        };

        // Start background cleanup task
        manager.start_cleanup_task().await;

        Ok(manager)
    }

    /// Cache model weights with automatic compression
    pub async fn cache_model_weights<T: serde::Serialize + serde::de::DeserializeOwned>(
        &self,
        model_key: &str,
        weights: &T,
        priority: CachePriority,
    ) -> Result<(), RecognitionError> {
        if !self.config.enable_persistence {
            return Ok(());
        }

        let mut cache = self.persistent_cache.write().await;
        cache.store_with_priority(model_key, weights, priority).await?;
        
        // Update stats
        let mut stats = self.stats.write().await;
        stats.item_count += 1;
        
        Ok(())
    }

    /// Retrieve cached model weights
    pub async fn get_model_weights<T: serde::de::DeserializeOwned>(
        &self,
        model_key: &str,
    ) -> Option<T> {
        if !self.config.enable_persistence {
            return None;
        }

        let mut cache = self.persistent_cache.write().await;
        match cache.retrieve::<T>(model_key).await {
            Ok(Some(weights)) => {
                // Update stats
                let mut stats = self.stats.write().await;
                stats.hits += 1;
                Some(weights)
            }
            _ => {
                // Update stats
                let mut stats = self.stats.write().await;
                stats.misses += 1;
                None
            }
        }
    }

    /// Cache audio features for repeated processing
    pub async fn cache_audio_features(
        &self,
        audio_hash: &str,
        features: &[f32],
        sample_rate: u32,
    ) -> Result<(), RecognitionError> {
        if !self.config.enable_feature_cache {
            return Ok(());
        }

        let mut cache = self.feature_cache.write().await;
        cache.store_features(audio_hash, features, sample_rate).await
    }

    /// Retrieve cached audio features
    pub async fn get_audio_features(
        &self,
        audio_hash: &str,
    ) -> Option<(Vec<f32>, u32)> {
        if !self.config.enable_feature_cache {
            return None;
        }

        let mut cache = self.feature_cache.write().await;
        match cache.get_features(audio_hash).await {
            Some(features) => {
                // Update stats
                let mut stats = self.stats.write().await;
                stats.hits += 1;
                Some(features)
            }
            None => {
                // Update stats
                let mut stats = self.stats.write().await;
                stats.misses += 1;
                None
            }
        }
    }

    /// Cache transcription results
    pub async fn cache_transcription_result(
        &self,
        input_hash: &str,
        result: &TranscriptionResult,
        model_id: &str,
    ) -> Result<(), RecognitionError> {
        if !self.config.enable_result_cache {
            return Ok(());
        }

        let mut cache = self.result_cache.write().await;
        cache.store_result(input_hash, result, model_id).await
    }

    /// Retrieve cached transcription result
    pub async fn get_transcription_result(
        &self,
        input_hash: &str,
        model_id: &str,
    ) -> Option<TranscriptionResult> {
        if !self.config.enable_result_cache {
            return None;
        }

        let mut cache = self.result_cache.write().await;
        match cache.get_result(input_hash, model_id).await {
            Some(result) => {
                // Update stats
                let mut stats = self.stats.write().await;
                stats.hits += 1;
                Some(result)
            }
            None => {
                // Update stats
                let mut stats = self.stats.write().await;
                stats.misses += 1;
                None
            }
        }
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let mut stats = self.stats.read().await.clone();
        
        // Calculate hit rate
        let total_requests = stats.hits + stats.misses;
        if total_requests > 0 {
            stats.hit_rate = stats.hits as f64 / total_requests as f64;
        }

        // Get compression ratio from persistent cache
        if self.config.enable_persistence {
            let cache = self.persistent_cache.read().await;
            stats.compression_ratio = cache.get_compression_ratio().await;
        }

        stats
    }

    /// Perform manual cache cleanup
    pub async fn cleanup(&self) -> Result<CleanupResult, RecognitionError> {
        let mut result = CleanupResult::default();

        // Cleanup persistent cache
        if self.config.enable_persistence {
            let mut cache = self.persistent_cache.write().await;
            let persistent_result = cache.cleanup().await?;
            result.items_removed += persistent_result.items_removed;
            result.bytes_freed += persistent_result.bytes_freed;
        }

        // Cleanup feature cache
        if self.config.enable_feature_cache {
            let mut cache = self.feature_cache.write().await;
            let feature_result = cache.cleanup().await?;
            result.items_removed += feature_result.items_removed;
            result.bytes_freed += feature_result.bytes_freed;
        }

        // Cleanup result cache
        if self.config.enable_result_cache {
            let mut cache = self.result_cache.write().await;
            let result_cleanup = cache.cleanup().await?;
            result.items_removed += result_cleanup.items_removed;
            result.bytes_freed += result_cleanup.bytes_freed;
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.last_cleanup = Some(Instant::now());
        stats.eviction_count += result.items_removed as u64;

        Ok(result)
    }

    /// Clear all caches
    pub async fn clear_all(&self) -> Result<(), RecognitionError> {
        if self.config.enable_persistence {
            let mut cache = self.persistent_cache.write().await;
            cache.clear().await?;
        }

        if self.config.enable_feature_cache {
            let mut cache = self.feature_cache.write().await;
            cache.clear().await?;
        }

        if self.config.enable_result_cache {
            let mut cache = self.result_cache.write().await;
            cache.clear().await?;
        }

        // Reset stats
        let mut stats = self.stats.write().await;
        *stats = CacheStats::default();

        Ok(())
    }

    /// Start background cleanup task
    async fn start_cleanup_task(&mut self) {
        let persistent_cache = self.persistent_cache.clone();
        let feature_cache = self.feature_cache.clone();
        let result_cache = self.result_cache.clone();
        let stats = self.stats.clone();
        let cleanup_interval = self.config.cleanup_interval;
        let enable_persistence = self.config.enable_persistence;
        let enable_feature_cache = self.config.enable_feature_cache;
        let enable_result_cache = self.config.enable_result_cache;

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            
            loop {
                interval.tick().await;
                
                // Perform background cleanup
                let mut total_removed = 0u64;
                
                if enable_persistence {
                    if let Ok(mut cache) = persistent_cache.try_write() {
                        if let Ok(result) = cache.cleanup().await {
                            total_removed += result.items_removed as u64;
                        }
                    }
                }

                if enable_feature_cache {
                    if let Ok(mut cache) = feature_cache.try_write() {
                        if let Ok(result) = cache.cleanup().await {
                            total_removed += result.items_removed as u64;
                        }
                    }
                }

                if enable_result_cache {
                    if let Ok(mut cache) = result_cache.try_write() {
                        if let Ok(result) = cache.cleanup().await {
                            total_removed += result.items_removed as u64;
                        }
                    }
                }

                // Update stats
                if let Ok(mut stats) = stats.try_write() {
                    stats.last_cleanup = Some(Instant::now());
                    stats.eviction_count += total_removed;
                }
            }
        });

        self.cleanup_handle = Some(handle);
    }
}

impl Drop for AdvancedCacheManager {
    fn drop(&mut self) {
        if let Some(handle) = self.cleanup_handle.take() {
            handle.abort();
        }
    }
}

/// Cache priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CachePriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Transcription result for caching
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub confidence: f32,
    pub timestamps: Vec<(f32, f32)>,
    pub language: Option<String>,
    pub model_used: String,
    pub processing_time_ms: u64,
    pub created_at: SystemTime,
}

/// Cache cleanup result
#[derive(Debug, Default)]
pub struct CleanupResult {
    pub items_removed: usize,
    pub bytes_freed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_cache_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = AdvancedCacheManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = AdvancedCacheManager::new(config).await.unwrap();
        let stats = manager.get_stats().await;
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[tokio::test]
    async fn test_cache_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = AdvancedCacheManager::new(config).await.unwrap();
        let result = manager.cleanup().await;
        assert!(result.is_ok());
    }
}