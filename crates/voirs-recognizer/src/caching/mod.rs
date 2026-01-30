//! Advanced caching strategies for model inference
//!
//! This module provides comprehensive caching capabilities to improve performance
//! of speech recognition by caching model outputs, embeddings, and intermediate results.
//!
//! # Features
//!
//! - **LRU Cache**: Least Recently Used eviction policy
//! - **TTL Support**: Time-based expiration for cache entries
//! - **Memory-Aware**: Automatic eviction based on memory pressure
//! - **Cache Warming**: Preload frequently used items
//! - **Compression**: Optional compression for large cached items
//! - **Distributed**: Support for distributed caching systems
//! - **Metrics**: Comprehensive cache hit/miss statistics
//!
//! # Example
//!
//! ```rust,no_run
//! use voirs_recognizer::caching::{ModelCache, CacheConfig, EvictionPolicy};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create cache with LRU policy and 1000 item capacity
//!     let config = CacheConfig {
//!         max_entries: 1000,
//!         max_memory_mb: 512,
//!         eviction_policy: EvictionPolicy::LRU,
//!         ttl_seconds: Some(3600), // 1 hour TTL
//!         enable_compression: true,
//!         ..Default::default()
//!     };
//!
//!     let cache = ModelCache::new(config).await?;
//!
//!     // Cache a model output
//!     let audio_hash = "audio_12345".to_string();
//!     let transcription = "Hello world".to_string();
//!     cache.put(audio_hash.clone(), transcription.clone()).await?;
//!
//!     // Retrieve from cache
//!     if let Some(cached) = cache.get(&audio_hash).await? {
//!         println!("Cache hit: {}", cached);
//!     }
//!
//!     // Get cache statistics
//!     let stats = cache.stats().await;
//!     println!("Hit rate: {:.2}%", stats.hit_rate() * 100.0);
//!
//!     Ok(())
//! }
//! ```

use crate::RecognitionError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

pub mod distributed;
pub mod lru;
pub mod warming;

/// Main model cache interface
pub struct ModelCache<T> {
    /// Cache configuration
    config: CacheConfig,
    /// Internal cache storage
    storage: Arc<RwLock<CacheStorage<T>>>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
    /// Cache warming manager
    warming_manager: Option<Arc<warming::CacheWarmingManager<T>>>,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of entries
    pub max_entries: usize,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Time-to-live in seconds (None = no expiration)
    pub ttl_seconds: Option<u64>,
    /// Enable compression for cached items
    pub enable_compression: bool,
    /// Enable cache warming
    pub enable_warming: bool,
    /// Distributed cache settings
    pub distributed: Option<DistributedCacheConfig>,
}

/// Cache eviction policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In First Out
    FIFO,
    /// Time-To-Live based
    TTL,
    /// Memory pressure based
    MemoryPressure,
}

/// Distributed cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedCacheConfig {
    /// Redis connection URL (optional)
    pub redis_url: Option<String>,
    /// Memcached servers (optional)
    pub memcached_servers: Vec<String>,
    /// Enable local cache as L1
    pub enable_local_cache: bool,
}

/// Internal cache storage
pub(crate) struct CacheStorage<T> {
    /// Cache entries
    entries: HashMap<String, CacheEntry<T>>,
    /// LRU access order
    lru_list: lru::LRUList,
    /// Current memory usage estimate in bytes
    current_memory_bytes: usize,
}

/// Cache entry with metadata
#[derive(Clone)]
struct CacheEntry<T> {
    /// Cached value
    value: T,
    /// When the entry was created
    created_at: Instant,
    /// When the entry was last accessed
    last_accessed: Instant,
    /// Access count
    access_count: u64,
    /// Size in bytes (estimate)
    size_bytes: usize,
    /// TTL for this entry
    ttl: Option<Duration>,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total insertions
    pub insertions: u64,
    /// Total evictions
    pub evictions: u64,
    /// Current entry count
    pub current_entries: usize,
    /// Current memory usage in bytes
    pub current_memory_bytes: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            max_memory_mb: 512,
            eviction_policy: EvictionPolicy::LRU,
            ttl_seconds: Some(3600), // 1 hour default
            enable_compression: false,
            enable_warming: false,
            distributed: None,
        }
    }
}

impl<T: Clone + Send + Sync + 'static> ModelCache<T> {
    /// Create a new model cache
    ///
    /// # Errors
    ///
    /// Returns an error if cache initialization fails
    pub async fn new(config: CacheConfig) -> Result<Self, RecognitionError> {
        let storage = Arc::new(RwLock::new(CacheStorage {
            entries: HashMap::new(),
            lru_list: lru::LRUList::new(),
            current_memory_bytes: 0,
        }));

        let stats = Arc::new(RwLock::new(CacheStats::default()));

        let warming_manager = if config.enable_warming {
            Some(Arc::new(warming::CacheWarmingManager::new(storage.clone())))
        } else {
            None
        };

        Ok(Self {
            config,
            storage,
            stats,
            warming_manager,
        })
    }

    /// Get a value from the cache
    ///
    /// # Errors
    ///
    /// Returns an error if cache access fails
    pub async fn get(&self, key: &str) -> Result<Option<T>, RecognitionError> {
        let mut storage = self.storage.write().await;
        let mut stats = self.stats.write().await;

        // Check if entry exists and not expired
        let value_to_return = if let Some(entry) = storage.entries.get(key) {
            // Check if entry has expired
            if let Some(ttl) = entry.ttl {
                if entry.created_at.elapsed() > ttl {
                    // Entry expired, remove it
                    storage.entries.remove(key);
                    storage.lru_list.remove(key);
                    stats.misses += 1;
                    stats.evictions += 1;
                    stats.current_entries = storage.entries.len();
                    return Ok(None);
                }
            }
            Some(entry.value.clone())
        } else {
            None
        };

        if let Some(value) = value_to_return {
            // Update access metadata (now we can get mutable reference)
            if let Some(entry) = storage.entries.get_mut(key) {
                entry.last_accessed = Instant::now();
                entry.access_count += 1;
            }

            // Update LRU list
            storage.lru_list.access(key);

            stats.hits += 1;
            Ok(Some(value))
        } else {
            stats.misses += 1;
            Ok(None)
        }
    }

    /// Put a value into the cache
    ///
    /// # Errors
    ///
    /// Returns an error if cache insertion fails
    pub async fn put(&self, key: String, value: T) -> Result<(), RecognitionError> {
        let mut storage = self.storage.write().await;
        let mut stats = self.stats.write().await;

        // Estimate size (simplified - in production would use accurate size calculation)
        let size_bytes = std::mem::size_of::<T>() + key.len();

        // Check if we need to evict entries
        while storage.entries.len() >= self.config.max_entries
            || storage.current_memory_bytes + size_bytes > self.config.max_memory_mb * 1024 * 1024
        {
            self.evict_one(&mut storage, &mut stats).await?;
        }

        let ttl = self
            .config
            .ttl_seconds
            .map(|secs| Duration::from_secs(secs));

        let entry = CacheEntry {
            value,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 0,
            size_bytes,
            ttl,
        };

        storage.current_memory_bytes += size_bytes;
        storage.entries.insert(key.clone(), entry);
        storage.lru_list.insert(key);

        stats.insertions += 1;
        stats.current_entries = storage.entries.len();
        stats.current_memory_bytes = storage.current_memory_bytes;

        Ok(())
    }

    /// Evict one entry based on eviction policy
    async fn evict_one(
        &self,
        storage: &mut CacheStorage<T>,
        stats: &mut CacheStats,
    ) -> Result<(), RecognitionError> {
        let key_to_evict = match self.config.eviction_policy {
            EvictionPolicy::LRU => storage.lru_list.get_lru(),
            EvictionPolicy::LFU => self.find_lfu_key(storage),
            EvictionPolicy::FIFO => self.find_fifo_key(storage),
            EvictionPolicy::TTL => self.find_expired_key(storage),
            EvictionPolicy::MemoryPressure => self.find_largest_key(storage),
        };

        if let Some(key) = key_to_evict {
            if let Some(entry) = storage.entries.remove(&key) {
                storage.current_memory_bytes = storage
                    .current_memory_bytes
                    .saturating_sub(entry.size_bytes);
                storage.lru_list.remove(&key);
                stats.evictions += 1;
            }
        }

        Ok(())
    }

    /// Find least frequently used key
    fn find_lfu_key(&self, storage: &CacheStorage<T>) -> Option<String> {
        storage
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
            .map(|(key, _)| key.clone())
    }

    /// Find oldest key (FIFO)
    fn find_fifo_key(&self, storage: &CacheStorage<T>) -> Option<String> {
        storage
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.created_at)
            .map(|(key, _)| key.clone())
    }

    /// Find expired key
    fn find_expired_key(&self, storage: &CacheStorage<T>) -> Option<String> {
        let now = Instant::now();
        storage
            .entries
            .iter()
            .find(|(_, entry)| {
                entry
                    .ttl
                    .map(|ttl| entry.created_at + ttl < now)
                    .unwrap_or(false)
            })
            .map(|(key, _)| key.clone())
    }

    /// Find largest key by size
    fn find_largest_key(&self, storage: &CacheStorage<T>) -> Option<String> {
        storage
            .entries
            .iter()
            .max_by_key(|(_, entry)| entry.size_bytes)
            .map(|(key, _)| key.clone())
    }

    /// Clear all cache entries
    ///
    /// # Errors
    ///
    /// Returns an error if cache clearing fails
    pub async fn clear(&self) -> Result<(), RecognitionError> {
        let mut storage = self.storage.write().await;
        storage.entries.clear();
        storage.lru_list.clear();
        storage.current_memory_bytes = 0;
        Ok(())
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }

    /// Warm the cache with frequently used items
    ///
    /// # Errors
    ///
    /// Returns an error if cache warming fails
    pub async fn warm(&self, items: Vec<(String, T)>) -> Result<(), RecognitionError> {
        if let Some(warming_mgr) = &self.warming_manager {
            warming_mgr.warm(items).await?;
        }
        Ok(())
    }

    /// Get cache size
    pub async fn size(&self) -> usize {
        self.storage.read().await.entries.len()
    }

    /// Check if cache contains a key
    pub async fn contains_key(&self, key: &str) -> bool {
        self.storage.read().await.entries.contains_key(key)
    }
}

impl CacheStats {
    /// Calculate cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Calculate memory usage percentage
    pub fn memory_usage_percent(&self, max_memory_bytes: usize) -> f64 {
        if max_memory_bytes == 0 {
            0.0
        } else {
            (self.current_memory_bytes as f64 / max_memory_bytes as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_creation() {
        let config = CacheConfig::default();
        let cache: ModelCache<String> = ModelCache::new(config).await.unwrap();
        assert_eq!(cache.size().await, 0);
    }

    #[tokio::test]
    async fn test_cache_put_get() {
        let config = CacheConfig::default();
        let cache: ModelCache<String> = ModelCache::new(config).await.unwrap();

        cache
            .put("key1".to_string(), "value1".to_string())
            .await
            .unwrap();
        let value = cache.get("key1").await.unwrap();
        assert_eq!(value, Some("value1".to_string()));
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let config = CacheConfig::default();
        let cache: ModelCache<String> = ModelCache::new(config).await.unwrap();

        let value = cache.get("nonexistent").await.unwrap();
        assert_eq!(value, None);
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let config = CacheConfig {
            max_entries: 2,
            ..Default::default()
        };
        let cache: ModelCache<String> = ModelCache::new(config).await.unwrap();

        cache
            .put("key1".to_string(), "value1".to_string())
            .await
            .unwrap();
        cache
            .put("key2".to_string(), "value2".to_string())
            .await
            .unwrap();
        cache
            .put("key3".to_string(), "value3".to_string())
            .await
            .unwrap();

        assert_eq!(cache.size().await, 2);
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let config = CacheConfig::default();
        let cache: ModelCache<String> = ModelCache::new(config).await.unwrap();

        cache
            .put("key1".to_string(), "value1".to_string())
            .await
            .unwrap();
        let _ = cache.get("key1").await.unwrap();
        let _ = cache.get("key2").await.unwrap();

        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let config = CacheConfig::default();
        let cache: ModelCache<String> = ModelCache::new(config).await.unwrap();

        cache
            .put("key1".to_string(), "value1".to_string())
            .await
            .unwrap();
        cache.clear().await.unwrap();
        assert_eq!(cache.size().await, 0);
    }

    #[tokio::test]
    async fn test_cache_contains_key() {
        let config = CacheConfig::default();
        let cache: ModelCache<String> = ModelCache::new(config).await.unwrap();

        cache
            .put("key1".to_string(), "value1".to_string())
            .await
            .unwrap();
        assert!(cache.contains_key("key1").await);
        assert!(!cache.contains_key("key2").await);
    }
}
