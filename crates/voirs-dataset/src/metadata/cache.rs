//! Metadata caching layer for performance optimization
//!
//! Provides LRU and disk-based caching with invalidation strategies
//! and memory usage optimization.

use crate::{DatasetError, Result};
use bincode;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use tokio::fs;

/// Metadata cache for efficient data access
pub struct MetadataCache {
    /// In-memory LRU cache
    memory_cache: Arc<RwLock<LruCache>>,
    /// Disk cache directory
    disk_cache_dir: Option<PathBuf>,
    /// Cache configuration
    config: CacheConfig,
    /// Cache statistics
    stats: Arc<RwLock<CacheStatistics>>,
    /// Background cleanup task handle
    cleanup_handle: Option<tokio::task::JoinHandle<()>>,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum memory cache entries
    pub max_memory_entries: usize,
    /// Enable disk cache
    pub enable_disk_cache: bool,
    /// Disk cache size limit in bytes
    pub max_disk_size: u64,
    /// Cache entry TTL
    pub ttl: Duration,
    /// Cache invalidation strategy
    pub invalidation_strategy: InvalidationStrategy,
    /// Compression level (0-9, 0 = no compression)
    pub compression_level: u8,
    /// Background cleanup interval
    pub cleanup_interval: Duration,
}

/// Cache invalidation strategies
#[derive(Debug, Clone)]
pub enum InvalidationStrategy {
    /// Time-based TTL
    TimeToLive,
    /// Access-based LRU
    LeastRecentlyUsed,
    /// Size-based eviction
    SizeBased,
    /// Combined strategy
    Combined,
}

/// Cache strategy for different use cases
#[derive(Debug, Clone)]
pub enum CacheStrategy {
    /// Aggressive caching for read-heavy workloads
    Aggressive,
    /// Conservative caching for memory-constrained environments
    Conservative,
    /// Balanced caching for general use
    Balanced,
    /// Custom strategy
    Custom(CacheConfig),
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_memory_entries: 10000,
            enable_disk_cache: true,
            max_disk_size: 1024 * 1024 * 1024,      // 1GB
            ttl: Duration::from_secs(24 * 60 * 60), // 24 hours
            invalidation_strategy: InvalidationStrategy::Combined,
            compression_level: 6,
            cleanup_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// LRU cache implementation
struct LruCache {
    /// Cache entries
    entries: HashMap<CacheKey, CacheEntry>,
    /// Access order for LRU
    access_order: VecDeque<CacheKey>,
    /// Maximum capacity
    max_capacity: usize,
    /// Current size in bytes
    current_size: u64,
}

/// Cache entry
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Cached data
    data: CachedData,
    /// Creation timestamp
    created_at: Instant,
    /// Last access timestamp
    last_accessed: Instant,
    /// Access count
    access_count: u64,
    /// Entry size in bytes
    size: u64,
}

/// Cached data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CachedData {
    /// Manifest data
    Manifest(Vec<u8>),
    /// Index data
    Index(Vec<u8>),
    /// Query results
    QueryResults(Vec<u8>),
    /// Sample metadata
    SampleMetadata(Vec<u8>),
    /// Statistics data
    Statistics(Vec<u8>),
}

/// Cache key for addressing entries
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Key type
    key_type: CacheKeyType,
    /// Unique identifier
    identifier: String,
    /// Optional sub-key
    sub_key: Option<String>,
}

/// Cache key types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CacheKeyType {
    Manifest,
    Index,
    Query,
    Sample,
    Statistics,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total evictions
    pub evictions: u64,
    /// Current memory usage
    pub memory_usage: u64,
    /// Current disk usage
    pub disk_usage: u64,
    /// Average access time
    pub avg_access_time: Duration,
    /// Hit rate (0.0 - 1.0)
    pub hit_rate: f64,
}

impl MetadataCache {
    /// Create a new metadata cache with default configuration
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create a new metadata cache with custom configuration
    pub fn with_config(config: CacheConfig) -> Self {
        let memory_cache = Arc::new(RwLock::new(LruCache::new(config.max_memory_entries)));
        let stats = Arc::new(RwLock::new(CacheStatistics::default()));

        let mut cache = Self {
            memory_cache,
            disk_cache_dir: None,
            config,
            stats,
            cleanup_handle: None,
        };

        cache.start_background_cleanup();
        cache
    }

    /// Create cache with strategy
    pub fn with_strategy(strategy: CacheStrategy) -> Self {
        let config = match strategy {
            CacheStrategy::Aggressive => CacheConfig {
                max_memory_entries: 50000,
                max_disk_size: 5 * 1024 * 1024 * 1024,  // 5GB
                ttl: Duration::from_secs(48 * 60 * 60), // 48 hours
                compression_level: 3,
                ..Default::default()
            },
            CacheStrategy::Conservative => CacheConfig {
                max_memory_entries: 1000,
                max_disk_size: 100 * 1024 * 1024,      // 100MB
                ttl: Duration::from_secs(2 * 60 * 60), // 2 hours
                compression_level: 9,
                ..Default::default()
            },
            CacheStrategy::Balanced => CacheConfig::default(),
            CacheStrategy::Custom(config) => config,
        };

        Self::with_config(config)
    }

    /// Set disk cache directory
    pub async fn set_disk_cache_dir(&mut self, dir: PathBuf) -> Result<()> {
        if self.config.enable_disk_cache {
            fs::create_dir_all(&dir)
                .await
                .map_err(DatasetError::IoError)?;
            self.disk_cache_dir = Some(dir);
        }
        Ok(())
    }

    /// Get data from cache
    pub async fn get(&self, key: &CacheKey) -> Result<Option<CachedData>> {
        let start_time = Instant::now();

        // Try memory cache first
        if let Some(data) = self.get_from_memory(key).await? {
            self.update_stats(true, start_time.elapsed()).await;
            return Ok(Some(data));
        }

        // Try disk cache if enabled
        if self.config.enable_disk_cache {
            if let Some(data) = self.get_from_disk(key).await? {
                // Store in memory cache for future access
                self.put_in_memory(key.clone(), data.clone()).await?;
                self.update_stats(true, start_time.elapsed()).await;
                return Ok(Some(data));
            }
        }

        self.update_stats(false, start_time.elapsed()).await;
        Ok(None)
    }

    /// Put data into cache
    pub async fn put(&self, key: CacheKey, data: CachedData) -> Result<()> {
        // Store in memory cache
        self.put_in_memory(key.clone(), data.clone()).await?;

        // Store in disk cache if enabled
        if self.config.enable_disk_cache {
            self.put_in_disk(key, data).await?;
        }

        Ok(())
    }

    /// Remove data from cache
    pub async fn remove(&self, key: &CacheKey) -> Result<bool> {
        let memory_removed = self.remove_from_memory(key).await?;
        let disk_removed = if self.config.enable_disk_cache {
            self.remove_from_disk(key).await?
        } else {
            false
        };

        Ok(memory_removed || disk_removed)
    }

    /// Clear all cache entries
    pub async fn clear(&self) -> Result<()> {
        // Clear memory cache
        {
            let mut cache = self
                .memory_cache
                .write()
                .map_err(|_| DatasetError::ConfigError("Cache lock poisoned".to_string()))?;
            cache.clear();
        }

        // Clear disk cache
        if let Some(disk_dir) = &self.disk_cache_dir {
            if disk_dir.exists() {
                fs::remove_dir_all(disk_dir)
                    .await
                    .map_err(DatasetError::IoError)?;
                fs::create_dir_all(disk_dir)
                    .await
                    .map_err(DatasetError::IoError)?;
            }
        }

        Ok(())
    }

    /// Get cache statistics
    pub async fn get_statistics(&self) -> Result<CacheStatistics> {
        let stats = self
            .stats
            .read()
            .map_err(|_| DatasetError::ConfigError("Stats lock poisoned".to_string()))?;
        Ok(stats.clone())
    }

    /// Cleanup expired entries
    pub async fn cleanup_expired(&self) -> Result<u64> {
        let mut removed_count = 0;

        // Cleanup memory cache
        {
            let mut cache = self
                .memory_cache
                .write()
                .map_err(|_| DatasetError::ConfigError("Cache lock poisoned".to_string()))?;

            let now = Instant::now();
            let mut keys_to_remove = Vec::new();

            for (key, entry) in &cache.entries {
                if now.duration_since(entry.created_at) > self.config.ttl {
                    keys_to_remove.push(key.clone());
                }
            }

            for key in keys_to_remove {
                cache.remove_entry(&key);
                removed_count += 1;
            }
        }

        // Cleanup disk cache
        if let Some(disk_dir) = &self.disk_cache_dir {
            removed_count += self.cleanup_disk_expired(disk_dir).await?;
        }

        Ok(removed_count)
    }

    /// Optimize cache structure
    pub async fn optimize(&self) -> Result<()> {
        // Trigger cleanup
        self.cleanup_expired().await?;

        // Optimize memory cache structure
        {
            let mut cache = self
                .memory_cache
                .write()
                .map_err(|_| DatasetError::ConfigError("Cache lock poisoned".to_string()))?;
            cache.optimize();
        }

        Ok(())
    }

    // Internal methods

    async fn get_from_memory(&self, key: &CacheKey) -> Result<Option<CachedData>> {
        let mut cache = self
            .memory_cache
            .write()
            .map_err(|_| DatasetError::ConfigError("Cache lock poisoned".to_string()))?;

        if let Some(entry) = cache.entries.get_mut(key) {
            // Check TTL
            if Instant::now().duration_since(entry.created_at) > self.config.ttl {
                cache.remove_entry(key);
                return Ok(None);
            }

            let data = entry.data.clone();

            // Update access information
            entry.last_accessed = Instant::now();
            entry.access_count += 1;

            // Update LRU order
            cache.update_access_order(key);

            Ok(Some(data))
        } else {
            Ok(None)
        }
    }

    async fn put_in_memory(&self, key: CacheKey, data: CachedData) -> Result<()> {
        let mut cache = self
            .memory_cache
            .write()
            .map_err(|_| DatasetError::ConfigError("Cache lock poisoned".to_string()))?;

        let data_size = self.estimate_size(&data);
        let now = Instant::now();

        let entry = CacheEntry {
            data,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            size: data_size,
        };

        cache.put(key, entry);
        Ok(())
    }

    async fn remove_from_memory(&self, key: &CacheKey) -> Result<bool> {
        let mut cache = self
            .memory_cache
            .write()
            .map_err(|_| DatasetError::ConfigError("Cache lock poisoned".to_string()))?;
        Ok(cache.remove_entry(key))
    }

    async fn get_from_disk(&self, key: &CacheKey) -> Result<Option<CachedData>> {
        if let Some(disk_dir) = &self.disk_cache_dir {
            let file_path = self.get_disk_path(disk_dir, key);

            if file_path.exists() {
                // Check file modification time for TTL
                let metadata = fs::metadata(&file_path)
                    .await
                    .map_err(DatasetError::IoError)?;

                if let Ok(modified) = metadata.modified() {
                    let age = SystemTime::now()
                        .duration_since(modified)
                        .unwrap_or(Duration::ZERO);

                    if age > self.config.ttl {
                        let _ = fs::remove_file(&file_path).await;
                        return Ok(None);
                    }
                }

                let compressed_data = fs::read(&file_path).await.map_err(DatasetError::IoError)?;

                let data = if self.config.compression_level > 0 {
                    self.decompress_data(&compressed_data)?
                } else {
                    compressed_data
                };

                let cached_data: CachedData = bincode::deserialize(&data).map_err(|e| {
                    DatasetError::FormatError(format!("Cache deserialization failed: {e}"))
                })?;

                return Ok(Some(cached_data));
            }
        }

        Ok(None)
    }

    async fn put_in_disk(&self, key: CacheKey, data: CachedData) -> Result<()> {
        if let Some(disk_dir) = &self.disk_cache_dir {
            let file_path = self.get_disk_path(disk_dir, &key);

            // Create parent directories if needed
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent)
                    .await
                    .map_err(DatasetError::IoError)?;
            }

            let serialized = bincode::serialize(&data).map_err(|e| {
                DatasetError::FormatError(format!("Cache serialization failed: {e}"))
            })?;

            let final_data = if self.config.compression_level > 0 {
                self.compress_data(&serialized)?
            } else {
                serialized
            };

            fs::write(&file_path, final_data)
                .await
                .map_err(DatasetError::IoError)?;
        }

        Ok(())
    }

    async fn remove_from_disk(&self, key: &CacheKey) -> Result<bool> {
        if let Some(disk_dir) = &self.disk_cache_dir {
            let file_path = self.get_disk_path(disk_dir, key);

            if file_path.exists() {
                fs::remove_file(&file_path)
                    .await
                    .map_err(DatasetError::IoError)?;
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn get_disk_path(&self, disk_dir: &Path, key: &CacheKey) -> PathBuf {
        let key_hash = self.hash_key(key);
        let type_dir = format!("{:?}", key.key_type).to_lowercase();

        disk_dir.join(type_dir).join(format!("{key_hash}.cache"))
    }

    fn hash_key(&self, key: &CacheKey) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    fn estimate_size(&self, data: &CachedData) -> u64 {
        match data {
            CachedData::Manifest(data) => data.len() as u64,
            CachedData::Index(data) => data.len() as u64,
            CachedData::QueryResults(data) => data.len() as u64,
            CachedData::SampleMetadata(data) => data.len() as u64,
            CachedData::Statistics(data) => data.len() as u64,
        }
    }

    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = GzEncoder::new(
            Vec::new(),
            Compression::new(self.config.compression_level as u32),
        );
        encoder
            .write_all(data)
            .map_err(|e| DatasetError::ConfigError(format!("Compression failed: {e}")))?;
        encoder
            .finish()
            .map_err(|e| DatasetError::ConfigError(format!("Compression finish failed: {e}")))
    }

    fn decompress_data(&self, compressed: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(compressed);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .map_err(|e| DatasetError::ConfigError(format!("Decompression failed: {e}")))?;
        Ok(decompressed)
    }

    async fn cleanup_disk_expired(&self, disk_dir: &Path) -> Result<u64> {
        let mut removed_count = 0;
        let now = SystemTime::now();

        let mut dir_entries = fs::read_dir(disk_dir)
            .await
            .map_err(DatasetError::IoError)?;

        while let Some(entry) = dir_entries
            .next_entry()
            .await
            .map_err(DatasetError::IoError)?
        {
            let path = entry.path();

            if path.is_file() && path.extension().is_some_and(|ext| ext == "cache") {
                let metadata = entry.metadata().await.map_err(DatasetError::IoError)?;

                if let Ok(modified) = metadata.modified() {
                    let age = now.duration_since(modified).unwrap_or(Duration::ZERO);

                    if age > self.config.ttl {
                        let _ = fs::remove_file(&path).await;
                        removed_count += 1;
                    }
                }
            }
        }

        Ok(removed_count)
    }

    async fn update_stats(&self, hit: bool, access_time: Duration) {
        if let Ok(mut stats) = self.stats.write() {
            if hit {
                stats.hits += 1;
            } else {
                stats.misses += 1;
            }

            let total_requests = stats.hits + stats.misses;
            stats.hit_rate = if total_requests > 0 {
                stats.hits as f64 / total_requests as f64
            } else {
                0.0
            };

            // Update average access time
            let current_avg_nanos = stats.avg_access_time.as_nanos() as u64;
            let new_avg_nanos = (current_avg_nanos * (total_requests - 1)
                + access_time.as_nanos() as u64)
                / total_requests;
            stats.avg_access_time = Duration::from_nanos(new_avg_nanos);
        }
    }

    fn start_background_cleanup(&mut self) {
        let cache = Arc::clone(&self.memory_cache);
        let config = self.config.clone();
        let _disk_dir = self.disk_cache_dir.clone();

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.cleanup_interval);

            loop {
                interval.tick().await;

                // Cleanup memory cache
                if let Ok(mut cache_guard) = cache.write() {
                    let now = Instant::now();
                    let mut keys_to_remove = Vec::new();

                    for (key, entry) in &cache_guard.entries {
                        if now.duration_since(entry.created_at) > config.ttl {
                            keys_to_remove.push(key.clone());
                        }
                    }

                    for key in keys_to_remove {
                        cache_guard.remove_entry(&key);
                    }
                }
            }
        });

        self.cleanup_handle = Some(handle);
    }
}

impl LruCache {
    fn new(max_capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            max_capacity,
            current_size: 0,
        }
    }

    fn put(&mut self, key: CacheKey, entry: CacheEntry) {
        // Remove existing entry if present
        if let Some(old_entry) = self.entries.remove(&key) {
            self.current_size -= old_entry.size;
            self.remove_from_access_order(&key);
        }

        // Evict entries if necessary
        while self.entries.len() >= self.max_capacity
            || (self.current_size + entry.size) > (self.max_capacity as u64 * 1024 * 1024)
        {
            self.evict_lru();
        }

        // Add new entry
        self.current_size += entry.size;
        self.entries.insert(key.clone(), entry);
        self.access_order.push_back(key);
    }

    fn remove_entry(&mut self, key: &CacheKey) -> bool {
        if let Some(entry) = self.entries.remove(key) {
            self.current_size -= entry.size;
            self.remove_from_access_order(key);
            true
        } else {
            false
        }
    }

    fn update_access_order(&mut self, key: &CacheKey) {
        self.remove_from_access_order(key);
        self.access_order.push_back(key.clone());
    }

    fn remove_from_access_order(&mut self, key: &CacheKey) {
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
        }
    }

    fn evict_lru(&mut self) {
        if let Some(key) = self.access_order.pop_front() {
            if let Some(entry) = self.entries.remove(&key) {
                self.current_size -= entry.size;
            }
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
        self.current_size = 0;
    }

    fn optimize(&mut self) {
        // Remove any orphaned entries from access_order
        self.access_order
            .retain(|key| self.entries.contains_key(key));
    }
}

impl CacheKey {
    /// Create a new cache key
    pub fn new(key_type: CacheKeyType, identifier: String) -> Self {
        Self {
            key_type,
            identifier,
            sub_key: None,
        }
    }

    /// Create a cache key with sub-key
    pub fn with_sub_key(key_type: CacheKeyType, identifier: String, sub_key: String) -> Self {
        Self {
            key_type,
            identifier,
            sub_key: Some(sub_key),
        }
    }

    /// Get the key type
    pub fn key_type(&self) -> &CacheKeyType {
        &self.key_type
    }

    /// Get the identifier
    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    /// Get the sub-key
    pub fn sub_key(&self) -> Option<&str> {
        self.sub_key.as_deref()
    }
}

impl Default for MetadataCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_cache_basic_operations() {
        let cache = MetadataCache::new();

        let key = CacheKey::new(CacheKeyType::Sample, "test-sample".to_string());
        let data = CachedData::SampleMetadata(b"test data".to_vec());

        // Test put and get
        cache.put(key.clone(), data.clone()).await.unwrap();
        let retrieved = cache.get(&key).await.unwrap();
        assert!(retrieved.is_some());

        // Test remove
        let removed = cache.remove(&key).await.unwrap();
        assert!(removed);

        let after_remove = cache.get(&key).await.unwrap();
        assert!(after_remove.is_none());
    }

    #[test]
    async fn test_cache_statistics() {
        let cache = MetadataCache::new();

        let key = CacheKey::new(CacheKeyType::Sample, "stats-test".to_string());
        let data = CachedData::SampleMetadata(b"stats data".to_vec());

        // Generate some cache activity
        cache.put(key.clone(), data).await.unwrap();
        let _ = cache.get(&key).await.unwrap(); // Hit
        let _ = cache
            .get(&CacheKey::new(CacheKeyType::Sample, "missing".to_string()))
            .await
            .unwrap(); // Miss

        let stats = cache.get_statistics().await.unwrap();
        assert!(stats.hits > 0);
        assert!(stats.misses > 0);
        assert!(stats.hit_rate > 0.0 && stats.hit_rate < 1.0);
    }

    #[test]
    async fn test_cache_strategies() {
        let aggressive_cache = MetadataCache::with_strategy(CacheStrategy::Aggressive);
        let conservative_cache = MetadataCache::with_strategy(CacheStrategy::Conservative);

        // Both caches should work with their respective configurations
        let key = CacheKey::new(CacheKeyType::Index, "strategy-test".to_string());
        let data = CachedData::Index(b"strategy data".to_vec());

        aggressive_cache
            .put(key.clone(), data.clone())
            .await
            .unwrap();
        conservative_cache.put(key.clone(), data).await.unwrap();

        assert!(aggressive_cache.get(&key).await.unwrap().is_some());
        assert!(conservative_cache.get(&key).await.unwrap().is_some());
    }

    #[test]
    async fn test_cache_cleanup() {
        let config = CacheConfig {
            ttl: Duration::from_millis(50),
            ..Default::default()
        };

        let cache = MetadataCache::with_config(config);

        let key = CacheKey::new(CacheKeyType::Query, "cleanup-test".to_string());
        let data = CachedData::QueryResults(b"cleanup data".to_vec());

        cache.put(key.clone(), data).await.unwrap();

        // Wait for TTL to expire
        tokio::time::sleep(Duration::from_millis(100)).await;

        let removed_count = cache.cleanup_expired().await.unwrap();
        assert!(removed_count > 0);

        let after_cleanup = cache.get(&key).await.unwrap();
        assert!(after_cleanup.is_none());
    }
}
