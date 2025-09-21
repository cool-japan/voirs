//! Advanced Caching System for VoiRS Voice Conversion
//!
//! This module provides sophisticated caching mechanisms to optimize performance
//! by reducing redundant computations and improving memory access patterns.

use crate::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Multi-level cache system for voice conversion operations
#[derive(Debug)]
pub struct ConversionCacheSystem {
    /// L1 cache for frequently accessed small items
    l1_cache: Arc<RwLock<LruCache<String, CachedItem>>>,
    /// L2 cache for larger or less frequently accessed items
    l2_cache: Arc<RwLock<LruCache<String, CachedItem>>>,
    /// Persistent cache for expensive computations
    persistent_cache: Arc<RwLock<HashMap<String, PersistentCacheItem>>>,
    /// Cache configuration
    config: CacheConfig,
    /// Cache statistics and metrics
    stats: Arc<Mutex<CacheStatistics>>,
    /// Cache policies for different data types
    policies: HashMap<CacheItemType, CachePolicy>,
}

/// Configuration for the cache system
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum size of L1 cache in bytes
    pub l1_max_size: usize,
    /// Maximum size of L2 cache in bytes
    pub l2_max_size: usize,
    /// Maximum number of items in L1 cache
    pub l1_max_items: usize,
    /// Maximum number of items in L2 cache
    pub l2_max_items: usize,
    /// TTL for cached items
    pub default_ttl: Duration,
    /// Enable compression for large items
    pub enable_compression: bool,
    /// Compression threshold in bytes
    pub compression_threshold: usize,
    /// Enable cache persistence
    pub enable_persistence: bool,
    /// Maximum size of persistent cache
    pub persistent_max_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_max_size: 50 * 1024 * 1024,  // 50MB
            l2_max_size: 200 * 1024 * 1024, // 200MB
            l1_max_items: 1000,
            l2_max_items: 5000,
            default_ttl: Duration::from_secs(300), // 5 minutes
            enable_compression: true,
            compression_threshold: 1024 * 1024, // 1MB
            enable_persistence: true,
            persistent_max_size: 1024 * 1024 * 1024, // 1GB
        }
    }
}

/// Cache policy for different types of data
#[derive(Debug, Clone)]
pub struct CachePolicy {
    /// Priority level (higher = more important to keep)
    pub priority: CachePriority,
    /// Time-to-live for this type of data
    pub ttl: Duration,
    /// Whether this type should be compressed
    pub compress: bool,
    /// Whether this type should persist across sessions
    pub persist: bool,
    /// Maximum size for individual items of this type
    pub max_item_size: usize,
}

/// Priority levels for cache items
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CachePriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Types of cacheable items
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheItemType {
    /// Model parameters and weights
    ModelData,
    /// Preprocessed audio features
    AudioFeatures,
    /// Conversion results for small audio clips
    ConversionResults,
    /// HRTF data and spatial processing
    SpatialData,
    /// Quality assessment results
    QualityMetrics,
    /// Configuration and metadata
    Metadata,
    /// Temporary intermediate results
    Intermediate,
}

/// Cached item with metadata
#[derive(Debug, Clone)]
pub struct CachedItem {
    /// The cached data
    pub data: CachedData,
    /// When the item was created
    pub created_at: Instant,
    /// When the item was last accessed
    pub last_accessed: Instant,
    /// Number of times accessed
    pub access_count: usize,
    /// Time-to-live for this item
    pub ttl: Duration,
    /// Priority of this item
    pub priority: CachePriority,
    /// Type of cached item
    pub item_type: CacheItemType,
    /// Size in bytes
    pub size: usize,
    /// Whether the item is compressed
    pub compressed: bool,
}

/// Different types of cached data
#[derive(Debug, Clone)]
pub enum CachedData {
    /// Raw binary data
    Binary(Vec<u8>),
    /// Audio samples
    Audio(Vec<f32>),
    /// Model parameters
    ModelParams(Vec<f32>),
    /// Text data
    Text(String),
    /// Structured data (JSON)
    Structured(serde_json::Value),
    /// Compressed data
    Compressed(Vec<u8>),
}

/// Persistent cache item with extended metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentCacheItem {
    /// The cached data (serialized)
    pub data: Vec<u8>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last access timestamp
    pub last_accessed: SystemTime,
    /// Access count
    pub access_count: usize,
    /// Item type
    pub item_type: CacheItemType,
    /// Original size before compression
    pub original_size: usize,
    /// Compressed size
    pub compressed_size: usize,
    /// Hash of the original data for integrity checking
    pub data_hash: u64,
}

/// Cache statistics and metrics
#[derive(Debug, Default, Clone)]
pub struct CacheStatistics {
    /// L1 cache statistics
    pub l1_stats: CacheLevelStats,
    /// L2 cache statistics
    pub l2_stats: CacheLevelStats,
    /// Persistent cache statistics
    pub persistent_stats: CacheLevelStats,
    /// Global statistics
    pub total_requests: usize,
    /// Global hit rate
    pub global_hit_rate: f64,
    /// Memory usage by cache level
    pub memory_usage: HashMap<String, usize>,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Statistics for a single cache level
#[derive(Debug, Default, Clone)]
pub struct CacheLevelStats {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Number of evictions
    pub evictions: usize,
    /// Current number of items
    pub current_items: usize,
    /// Current size in bytes
    pub current_size: usize,
    /// Number of compressed items
    pub compressed_items: usize,
    /// Bytes saved through compression
    pub bytes_saved: usize,
    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Average access time
    pub avg_access_time: Duration,
}

/// Performance metrics for cache operations
#[derive(Debug, Default, Clone)]
pub struct PerformanceMetrics {
    /// Average time to store an item
    pub avg_store_time: Duration,
    /// Average time to retrieve an item
    pub avg_retrieve_time: Duration,
    /// Average compression time
    pub avg_compression_time: Duration,
    /// Average decompression time
    pub avg_decompression_time: Duration,
    /// Compression ratio achieved
    pub compression_ratio: f64,
}

/// LRU (Least Recently Used) cache implementation
#[derive(Debug)]
pub struct LruCache<K, V> {
    /// Maximum capacity in number of items
    max_items: usize,
    /// Maximum size in bytes
    max_size: usize,
    /// Current size in bytes
    current_size: usize,
    /// Cache entries
    entries: HashMap<K, V>,
    /// Access order (most recent first)
    access_order: VecDeque<K>,
}

impl<K: Clone + Eq + Hash, V> LruCache<K, V> {
    /// Create a new LRU cache
    pub fn new(max_items: usize, max_size: usize) -> Self {
        Self {
            max_items,
            max_size,
            current_size: 0,
            entries: HashMap::new(),
            access_order: VecDeque::new(),
        }
    }

    /// Get an item from the cache
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.entries.contains_key(key) {
            self.move_to_front(key.clone());
            self.entries.get(key)
        } else {
            None
        }
    }

    /// Insert an item into the cache
    pub fn insert(&mut self, key: K, value: V, size: usize) -> bool {
        // Check if item is too large for cache
        if size > self.max_size {
            return false;
        }

        // If key already exists, update it
        if self.entries.contains_key(&key) {
            self.entries.insert(key.clone(), value);
            self.move_to_front(key);
            return true;
        }

        // Ensure we have space
        while (self.entries.len() >= self.max_items || self.current_size + size > self.max_size)
            && !self.entries.is_empty()
        {
            self.evict_lru();
        }

        // Insert new item
        self.entries.insert(key.clone(), value);
        self.access_order.push_front(key);
        self.current_size += size;
        true
    }

    /// Remove an item from the cache
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(value) = self.entries.remove(key) {
            self.access_order.retain(|k| k != key);
            Some(value)
        } else {
            None
        }
    }

    /// Get the current number of items
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get current size in bytes
    pub fn current_size(&self) -> usize {
        self.current_size
    }

    /// Clear all items from the cache
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
        self.current_size = 0;
    }

    fn move_to_front(&mut self, key: K) {
        self.access_order.retain(|k| k != &key);
        self.access_order.push_front(key);
    }

    fn evict_lru(&mut self) {
        if let Some(key) = self.access_order.pop_back() {
            self.entries.remove(&key);
            // Note: We can't easily track individual item sizes here,
            // so we approximate. In a real implementation, we'd store size with each item.
            self.current_size = self
                .current_size
                .saturating_sub(self.current_size / self.entries.len().max(1));
        }
    }
}

impl Default for ConversionCacheSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversionCacheSystem {
    /// Create a new cache system with default configuration
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create a new cache system with custom configuration
    pub fn with_config(config: CacheConfig) -> Self {
        let mut policies = HashMap::new();

        // Set up default policies for different item types
        policies.insert(
            CacheItemType::ModelData,
            CachePolicy {
                priority: CachePriority::Critical,
                ttl: Duration::from_secs(3600), // 1 hour
                compress: true,
                persist: true,
                max_item_size: 50 * 1024 * 1024, // 50MB
            },
        );

        policies.insert(
            CacheItemType::AudioFeatures,
            CachePolicy {
                priority: CachePriority::High,
                ttl: Duration::from_secs(600), // 10 minutes
                compress: true,
                persist: false,
                max_item_size: 10 * 1024 * 1024, // 10MB
            },
        );

        policies.insert(
            CacheItemType::ConversionResults,
            CachePolicy {
                priority: CachePriority::Medium,
                ttl: Duration::from_secs(300), // 5 minutes
                compress: false,
                persist: false,
                max_item_size: 5 * 1024 * 1024, // 5MB
            },
        );

        policies.insert(
            CacheItemType::Intermediate,
            CachePolicy {
                priority: CachePriority::Low,
                ttl: Duration::from_secs(60), // 1 minute
                compress: false,
                persist: false,
                max_item_size: 1024 * 1024, // 1MB
            },
        );

        Self {
            l1_cache: Arc::new(RwLock::new(LruCache::new(
                config.l1_max_items,
                config.l1_max_size,
            ))),
            l2_cache: Arc::new(RwLock::new(LruCache::new(
                config.l2_max_items,
                config.l2_max_size,
            ))),
            persistent_cache: Arc::new(RwLock::new(HashMap::new())),
            config: config.clone(),
            stats: Arc::new(Mutex::new(CacheStatistics::default())),
            policies,
        }
    }

    /// Store an item in the cache
    pub fn store(&self, key: String, data: CachedData, item_type: CacheItemType) -> Result<()> {
        let start_time = Instant::now();

        let default_policy = CachePolicy {
            priority: CachePriority::Medium,
            ttl: self.config.default_ttl,
            compress: false,
            persist: false,
            max_item_size: 10 * 1024 * 1024,
        };
        let policy = self.policies.get(&item_type).unwrap_or(&default_policy);

        let (processed_data, size, compressed) = self.process_data_for_storage(data, policy)?;

        let item = CachedItem {
            data: processed_data,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 0,
            ttl: policy.ttl,
            priority: policy.priority,
            item_type,
            size,
            compressed,
        };

        // Determine which cache level to use
        let stored =
            if size <= self.config.l1_max_size / 10 && policy.priority >= CachePriority::High {
                // Store in L1 cache for small, important items
                let mut l1 = self.l1_cache.write().unwrap();
                l1.insert(key.clone(), item.clone(), size)
            } else if size <= self.config.l2_max_size / 10 {
                // Store in L2 cache for larger items
                let mut l2 = self.l2_cache.write().unwrap();
                l2.insert(key.clone(), item.clone(), size)
            } else {
                false
            };

        if !stored && policy.persist && self.config.enable_persistence {
            // Store in persistent cache
            self.store_persistent(key, item)?;
        }

        // Update statistics
        let store_time = start_time.elapsed();
        self.update_store_stats(store_time);

        Ok(())
    }

    /// Retrieve an item from the cache
    pub fn retrieve(&self, key: &str) -> Option<CachedData> {
        let start_time = Instant::now();

        // Try L1 cache first
        {
            let mut l1 = self.l1_cache.write().unwrap();
            if let Some(item) = l1.get(&key.to_string()) {
                if !self.is_expired(item) {
                    let result = self.process_data_for_retrieval(&item.data, item.compressed);
                    self.update_retrieve_stats(start_time.elapsed(), true, "L1");
                    return result;
                }
            }
        }

        // Try L2 cache
        {
            let mut l2 = self.l2_cache.write().unwrap();
            if let Some(item) = l2.get(&key.to_string()) {
                if !self.is_expired(item) {
                    let result = self.process_data_for_retrieval(&item.data, item.compressed);
                    self.update_retrieve_stats(start_time.elapsed(), true, "L2");
                    return result;
                }
            }
        }

        // Try persistent cache
        if self.config.enable_persistence {
            let persistent = self.persistent_cache.read().unwrap();
            if let Some(persistent_item) = persistent.get(key) {
                if !self.is_persistent_expired(persistent_item) {
                    // Deserialize and decompress if needed
                    if let Ok(cached_data) = self.deserialize_persistent_data(persistent_item) {
                        self.update_retrieve_stats(start_time.elapsed(), true, "Persistent");
                        return Some(cached_data);
                    }
                }
            }
        }

        // Cache miss
        self.update_retrieve_stats(start_time.elapsed(), false, "None");
        None
    }

    /// Remove an item from all cache levels
    pub fn remove(&self, key: &str) {
        {
            let mut l1 = self.l1_cache.write().unwrap();
            l1.remove(&key.to_string());
        }
        {
            let mut l2 = self.l2_cache.write().unwrap();
            l2.remove(&key.to_string());
        }
        if self.config.enable_persistence {
            let mut persistent = self.persistent_cache.write().unwrap();
            persistent.remove(key);
        }
    }

    /// Clear all caches
    pub fn clear_all(&self) {
        {
            let mut l1 = self.l1_cache.write().unwrap();
            l1.clear();
        }
        {
            let mut l2 = self.l2_cache.write().unwrap();
            l2.clear();
        }
        if self.config.enable_persistence {
            let mut persistent = self.persistent_cache.write().unwrap();
            persistent.clear();
        }
    }

    /// Get cache statistics
    pub fn get_statistics(&self) -> CacheStatistics {
        let mut stats = self.stats.lock().unwrap();

        // Update current cache level stats
        {
            let l1 = self.l1_cache.read().unwrap();
            stats.l1_stats.current_items = l1.len();
            stats.l1_stats.current_size = l1.current_size();
        }
        {
            let l2 = self.l2_cache.read().unwrap();
            stats.l2_stats.current_items = l2.len();
            stats.l2_stats.current_size = l2.current_size();
        }
        if self.config.enable_persistence {
            let persistent = self.persistent_cache.read().unwrap();
            stats.persistent_stats.current_items = persistent.len();
            stats.persistent_stats.current_size =
                persistent.values().map(|item| item.compressed_size).sum();
        }

        // Calculate global hit rate
        let total_hits = stats.l1_stats.hits + stats.l2_stats.hits + stats.persistent_stats.hits;
        let total_misses =
            stats.l1_stats.misses + stats.l2_stats.misses + stats.persistent_stats.misses;
        let total_requests = total_hits + total_misses;

        if total_requests > 0 {
            stats.global_hit_rate = total_hits as f64 / total_requests as f64;
        }

        stats.clone()
    }

    /// Optimize cache by removing expired items and reorganizing
    pub fn optimize(&self) {
        self.cleanup_expired_items();
        self.rebalance_caches();
        self.compress_underutilized_items();
    }

    /// Create a cache key from conversion parameters
    pub fn create_cache_key(
        &self,
        conversion_type: &ConversionType,
        audio_hash: u64,
        target_hash: u64,
        quality_level: u8,
    ) -> String {
        format!(
            "conv_{}_{:016x}_{:016x}_q{}",
            self.conversion_type_to_string(conversion_type),
            audio_hash,
            target_hash,
            quality_level
        )
    }

    /// Generate hash for audio data
    pub fn hash_audio_data(&self, audio: &[f32]) -> u64 {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();

        // Hash length and some sample points for efficiency
        audio.len().hash(&mut hasher);

        if audio.len() <= 1000 {
            // Hash all samples for small audio
            for &sample in audio {
                ((sample * 10000.0) as i32).hash(&mut hasher);
            }
        } else {
            // Hash samples at regular intervals for large audio
            let step = audio.len() / 100;
            for i in (0..audio.len()).step_by(step) {
                ((audio[i] * 10000.0) as i32).hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    // Private helper methods

    fn process_data_for_storage(
        &self,
        data: CachedData,
        policy: &CachePolicy,
    ) -> Result<(CachedData, usize, bool)> {
        let size = self.estimate_data_size(&data);

        if policy.compress
            && self.config.enable_compression
            && size > self.config.compression_threshold
        {
            // Compress the data
            let compressed_data = self.compress_data(&data)?;
            let compressed_size = compressed_data.len();
            Ok((
                CachedData::Compressed(compressed_data),
                compressed_size,
                true,
            ))
        } else {
            Ok((data, size, false))
        }
    }

    fn process_data_for_retrieval(
        &self,
        data: &CachedData,
        compressed: bool,
    ) -> Option<CachedData> {
        if compressed {
            if let CachedData::Compressed(compressed_data) = data {
                self.decompress_data(compressed_data).ok()
            } else {
                None
            }
        } else {
            Some(data.clone())
        }
    }

    fn compress_data(&self, data: &CachedData) -> Result<Vec<u8>> {
        let serialized = serde_json::to_vec(data).map_err(|e| {
            Error::processing(format!("Failed to serialize data for compression: {e}"))
        })?;

        // Use flate2 for real compression with high compression level for cache efficiency
        use flate2::{write::GzEncoder, Compression};
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
        encoder
            .write_all(&serialized)
            .map_err(|e| Error::processing(format!("Failed to compress data: {e}")))?;
        encoder
            .finish()
            .map_err(|e| Error::processing(format!("Failed to finalize compression: {e}")))
    }

    fn decompress_data(&self, compressed: &[u8]) -> Result<CachedData> {
        // Use flate2 for real decompression
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(compressed);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .map_err(|e| Error::processing(format!("Failed to decompress data: {e}")))?;

        serde_json::from_slice(&decompressed)
            .map_err(|e| Error::processing(format!("Failed to deserialize decompressed data: {e}")))
    }

    fn compress_data_max(&self, data: &CachedData) -> Result<Vec<u8>> {
        let serialized = serde_json::to_vec(data).map_err(|e| {
            Error::processing(format!("Failed to serialize data for max compression: {e}"))
        })?;

        // Use maximum compression level for long-term storage
        use flate2::{write::GzEncoder, Compression};
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(&serialized).map_err(|e| {
            Error::processing(format!("Failed to compress data with max compression: {e}"))
        })?;
        encoder
            .finish()
            .map_err(|e| Error::processing(format!("Failed to finalize max compression: {e}")))
    }

    fn estimate_data_size(&self, data: &CachedData) -> usize {
        match data {
            CachedData::Binary(bytes) => bytes.len(),
            CachedData::Audio(samples) => samples.len() * std::mem::size_of::<f32>(),
            CachedData::ModelParams(params) => params.len() * std::mem::size_of::<f32>(),
            CachedData::Text(text) => text.len(),
            CachedData::Structured(value) => serde_json::to_string(value).unwrap_or_default().len(),
            CachedData::Compressed(bytes) => bytes.len(),
        }
    }

    fn is_expired(&self, item: &CachedItem) -> bool {
        item.created_at.elapsed() > item.ttl
    }

    fn is_persistent_expired(&self, item: &PersistentCacheItem) -> bool {
        let age = SystemTime::now()
            .duration_since(item.created_at)
            .unwrap_or_default();
        age > Duration::from_secs(86400) // 24 hours for persistent items
    }

    fn store_persistent(&self, key: String, item: CachedItem) -> Result<()> {
        let serialized_data = serde_json::to_vec(&item.data)
            .map_err(|e| Error::processing(format!("Failed to serialize persistent data: {e}")))?;

        let compressed_data = self.compress_data(&item.data)?;

        let persistent_item = PersistentCacheItem {
            data: compressed_data.clone(),
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 0,
            item_type: item.item_type,
            original_size: serialized_data.len(),
            compressed_size: compressed_data.len(),
            data_hash: self.hash_data(&serialized_data),
        };

        let mut persistent = self.persistent_cache.write().unwrap();
        persistent.insert(key, persistent_item);

        Ok(())
    }

    fn deserialize_persistent_data(&self, item: &PersistentCacheItem) -> Result<CachedData> {
        self.decompress_data(&item.data)
    }

    fn hash_data(&self, data: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish()
    }

    fn cleanup_expired_items(&self) {
        // L1 cache cleanup
        {
            let mut l1 = self.l1_cache.write().unwrap();
            let keys_to_remove: Vec<String> = l1
                .entries
                .iter()
                .filter(|(_, item)| self.is_expired(item))
                .map(|(k, _)| k.clone())
                .collect();

            for key in keys_to_remove {
                l1.remove(&key);
            }
        }

        // L2 cache cleanup
        {
            let mut l2 = self.l2_cache.write().unwrap();
            let keys_to_remove: Vec<String> = l2
                .entries
                .iter()
                .filter(|(_, item)| self.is_expired(item))
                .map(|(k, _)| k.clone())
                .collect();

            for key in keys_to_remove {
                l2.remove(&key);
            }
        }

        // Persistent cache cleanup
        if self.config.enable_persistence {
            let mut persistent = self.persistent_cache.write().unwrap();
            let keys_to_remove: Vec<String> = persistent
                .iter()
                .filter(|(_, item)| self.is_persistent_expired(item))
                .map(|(k, _)| k.clone())
                .collect();

            for key in keys_to_remove {
                persistent.remove(&key);
            }
        }
    }

    fn rebalance_caches(&self) {
        let now = Instant::now();
        let l1_available_space = {
            let l1 = self.l1_cache.read().unwrap();
            self.config.l1_max_size.saturating_sub(l1.current_size())
        };

        // Only rebalance if we have significant space available
        if l1_available_space > 1024 * 1024 {
            // 1MB threshold
            let mut candidates_for_promotion = Vec::new();

            // Collect high-access items from L2 for potential promotion to L1
            {
                let l2 = self.l2_cache.read().unwrap();
                for (key, item) in l2.entries.iter() {
                    // Calculate access frequency (accesses per minute)
                    let age_minutes = now.duration_since(item.created_at).as_secs() / 60;
                    let access_frequency = if age_minutes > 0 {
                        item.access_count as f64 / age_minutes as f64
                    } else {
                        item.access_count as f64
                    };

                    // Consider items with high access frequency and recent access
                    let recently_accessed =
                        now.duration_since(item.last_accessed) < Duration::from_secs(300); // 5 minutes
                    let high_priority = item.priority >= CachePriority::High;
                    let frequently_accessed = access_frequency > 2.0; // More than 2 accesses per minute

                    if (frequently_accessed || high_priority)
                        && recently_accessed
                        && item.size <= l1_available_space
                    {
                        candidates_for_promotion.push((
                            key.clone(),
                            item.clone(),
                            access_frequency,
                        ));
                    }
                }
            }

            // Sort candidates by access frequency (highest first)
            candidates_for_promotion
                .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

            // Promote candidates to L1, respecting space constraints
            let mut space_used = 0;
            for (key, item, _) in candidates_for_promotion {
                let item_size = item.size;
                if space_used + item_size > l1_available_space {
                    break;
                }

                // Move item from L2 to L1
                {
                    let mut l2 = self.l2_cache.write().unwrap();
                    if l2.remove(&key).is_some() {
                        space_used += item_size;

                        let mut l1 = self.l1_cache.write().unwrap();
                        l1.insert(key, item, item_size);
                    }
                }
            }
        }

        // Also demote least accessed L1 items to L2 if L1 is near capacity
        let l1_utilization = {
            let l1 = self.l1_cache.read().unwrap();
            l1.current_size() as f64 / self.config.l1_max_size as f64
        };

        if l1_utilization > 0.9 {
            // If L1 is more than 90% full
            let mut candidates_for_demotion = Vec::new();

            {
                let l1 = self.l1_cache.read().unwrap();
                for (key, item) in l1.entries.iter() {
                    let age_minutes = now.duration_since(item.created_at).as_secs() / 60;
                    let access_frequency = if age_minutes > 0 {
                        item.access_count as f64 / age_minutes as f64
                    } else {
                        item.access_count as f64
                    };

                    let not_recently_accessed =
                        now.duration_since(item.last_accessed) > Duration::from_secs(600); // 10 minutes
                    let low_priority = item.priority <= CachePriority::Medium;
                    let infrequently_accessed = access_frequency < 0.5; // Less than 0.5 accesses per minute

                    if (infrequently_accessed || low_priority) && not_recently_accessed {
                        candidates_for_demotion.push((key.clone(), item.clone(), access_frequency));
                    }
                }
            }

            // Sort candidates by access frequency (lowest first)
            candidates_for_demotion
                .sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            // Demote candidates from L1 to L2
            let l2_available_space = {
                let l2 = self.l2_cache.read().unwrap();
                self.config.l2_max_size.saturating_sub(l2.current_size())
            };

            let mut space_used = 0;
            for (key, item, _) in candidates_for_demotion {
                let item_size = item.size;
                if space_used + item_size > l2_available_space {
                    break;
                }

                // Move item from L1 to L2
                {
                    let mut l1 = self.l1_cache.write().unwrap();
                    if l1.remove(&key).is_some() {
                        space_used += item_size;

                        let mut l2 = self.l2_cache.write().unwrap();
                        l2.insert(key, item, item_size);
                    }
                }
            }
        }
    }

    fn compress_underutilized_items(&self) {
        let now = Instant::now();
        let compression_threshold = Duration::from_secs(1800); // 30 minutes without access
        let min_size_for_compression = self.config.compression_threshold;

        // Compress underutilized items in L2 cache
        {
            let mut l2 = self.l2_cache.write().unwrap();
            let mut items_to_compress = Vec::new();

            for (key, item) in l2.entries.iter() {
                let time_since_access = now.duration_since(item.last_accessed);
                let should_compress = !item.compressed
                    && time_since_access > compression_threshold
                    && item.size >= min_size_for_compression
                    && item.access_count < 5; // Don't compress frequently used items

                if should_compress {
                    items_to_compress.push(key.clone());
                }
            }

            // Compress the identified items
            for key in items_to_compress {
                if let Some(mut item) = l2.entries.get_mut(&key) {
                    if let Ok(compressed_data) = self.compress_data(&item.data) {
                        let original_size = item.size;
                        item.data = CachedData::Compressed(compressed_data);
                        item.size = match &item.data {
                            CachedData::Compressed(data) => data.len(),
                            _ => item.size,
                        };
                        item.compressed = true;

                        // Update statistics
                        let mut stats = self.stats.lock().unwrap();
                        stats.l2_stats.compressed_items += 1;
                        stats.l2_stats.bytes_saved += original_size.saturating_sub(item.size);
                    }
                }
            }
        }

        // Also compress underutilized items in persistent cache
        if self.config.enable_persistence {
            let mut persistent = self.persistent_cache.write().unwrap();
            let mut items_to_recompress = Vec::new();

            for (key, item) in persistent.iter() {
                let time_since_access = SystemTime::now()
                    .duration_since(item.last_accessed)
                    .unwrap_or(Duration::from_secs(0));

                // Recompress with better compression for very old items
                let should_recompress = time_since_access > Duration::from_secs(7200) // 2 hours
                    && item.access_count < 3
                    && item.compressed_size > 1024; // Only recompress larger items

                if should_recompress {
                    items_to_recompress.push(key.clone());
                }
            }

            for key in items_to_recompress {
                if let Some(item) = persistent.get_mut(&key) {
                    // Decompress, then recompress with maximum compression
                    if let Ok(decompressed_data) = self.decompress_data(&item.data) {
                        // Use maximum compression for long-term storage
                        if let Ok(recompressed_data) = self.compress_data_max(&decompressed_data) {
                            let old_size = item.compressed_size;
                            item.data = recompressed_data;
                            item.compressed_size = item.data.len();

                            // Update statistics
                            let mut stats = self.stats.lock().unwrap();
                            stats.persistent_stats.bytes_saved +=
                                old_size.saturating_sub(item.compressed_size);
                        }
                    }
                }
            }
        }
    }

    fn conversion_type_to_string(&self, conversion_type: &ConversionType) -> &'static str {
        match conversion_type {
            ConversionType::PitchShift => "pitch",
            ConversionType::SpeedTransformation => "speed",
            ConversionType::SpeakerConversion => "speaker",
            ConversionType::AgeTransformation => "age",
            ConversionType::GenderTransformation => "gender",
            ConversionType::VoiceMorphing => "morphing",
            ConversionType::EmotionalTransformation => "emotion",
            ConversionType::ZeroShotConversion => "zero_shot",
            ConversionType::Custom(_) => "custom",
            ConversionType::PassThrough => "passthrough",
        }
    }

    fn update_store_stats(&self, duration: Duration) {
        let mut stats = self.stats.lock().unwrap();
        // Update average store time using exponential moving average
        let alpha = 0.1;
        let current_avg = stats.performance_metrics.avg_store_time.as_nanos() as f64;
        let new_avg = current_avg * (1.0 - alpha) + duration.as_nanos() as f64 * alpha;
        stats.performance_metrics.avg_store_time = Duration::from_nanos(new_avg as u64);
    }

    fn update_retrieve_stats(&self, duration: Duration, hit: bool, cache_level: &str) {
        let mut stats = self.stats.lock().unwrap();

        // Update retrieve time
        let alpha = 0.1;
        let current_avg = stats.performance_metrics.avg_retrieve_time.as_nanos() as f64;
        let new_avg = current_avg * (1.0 - alpha) + duration.as_nanos() as f64 * alpha;
        stats.performance_metrics.avg_retrieve_time = Duration::from_nanos(new_avg as u64);

        // Update hit/miss stats for the appropriate cache level
        match cache_level {
            "L1" => {
                if hit {
                    stats.l1_stats.hits += 1;
                } else {
                    stats.l1_stats.misses += 1;
                }
                let total = stats.l1_stats.hits + stats.l1_stats.misses;
                stats.l1_stats.hit_rate = stats.l1_stats.hits as f64 / total as f64;
            }
            "L2" => {
                if hit {
                    stats.l2_stats.hits += 1;
                } else {
                    stats.l2_stats.misses += 1;
                }
                let total = stats.l2_stats.hits + stats.l2_stats.misses;
                stats.l2_stats.hit_rate = stats.l2_stats.hits as f64 / total as f64;
            }
            "Persistent" => {
                if hit {
                    stats.persistent_stats.hits += 1;
                } else {
                    stats.persistent_stats.misses += 1;
                }
                let total = stats.persistent_stats.hits + stats.persistent_stats.misses;
                stats.persistent_stats.hit_rate = stats.persistent_stats.hits as f64 / total as f64;
            }
            _ => {}
        }

        stats.total_requests += 1;
    }
}

// Implement serde for CachedData
impl Serialize for CachedData {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        match self {
            CachedData::Binary(data) => {
                let mut state = serializer.serialize_struct("CachedData", 2)?;
                state.serialize_field("type", "Binary")?;
                state.serialize_field("data", data)?;
                state.end()
            }
            CachedData::Audio(data) => {
                let mut state = serializer.serialize_struct("CachedData", 2)?;
                state.serialize_field("type", "Audio")?;
                state.serialize_field("data", data)?;
                state.end()
            }
            CachedData::ModelParams(data) => {
                let mut state = serializer.serialize_struct("CachedData", 2)?;
                state.serialize_field("type", "ModelParams")?;
                state.serialize_field("data", data)?;
                state.end()
            }
            CachedData::Text(data) => {
                let mut state = serializer.serialize_struct("CachedData", 2)?;
                state.serialize_field("type", "Text")?;
                state.serialize_field("data", data)?;
                state.end()
            }
            CachedData::Structured(data) => {
                let mut state = serializer.serialize_struct("CachedData", 2)?;
                state.serialize_field("type", "Structured")?;
                state.serialize_field("data", data)?;
                state.end()
            }
            CachedData::Compressed(data) => {
                let mut state = serializer.serialize_struct("CachedData", 2)?;
                state.serialize_field("type", "Compressed")?;
                state.serialize_field("data", data)?;
                state.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for CachedData {
    fn deserialize<D>(deserializer: D) -> std::result::Result<CachedData, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct CachedDataVisitor;

        impl<'de> Visitor<'de> for CachedDataVisitor {
            type Value = CachedData;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a CachedData struct")
            }

            fn visit_map<V>(self, mut map: V) -> std::result::Result<CachedData, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut data_type: Option<String> = None;
                let mut data: Option<serde_json::Value> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "type" => {
                            if data_type.is_some() {
                                return Err(de::Error::duplicate_field("type"));
                            }
                            data_type = Some(map.next_value()?);
                        }
                        "data" => {
                            if data.is_some() {
                                return Err(de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        }
                        _ => {
                            let _: serde_json::Value = map.next_value()?;
                        }
                    }
                }

                let data_type = data_type.ok_or_else(|| de::Error::missing_field("type"))?;
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;

                match data_type.as_str() {
                    "Binary" => Ok(CachedData::Binary(
                        serde_json::from_value(data).map_err(de::Error::custom)?,
                    )),
                    "Audio" => Ok(CachedData::Audio(
                        serde_json::from_value(data).map_err(de::Error::custom)?,
                    )),
                    "ModelParams" => Ok(CachedData::ModelParams(
                        serde_json::from_value(data).map_err(de::Error::custom)?,
                    )),
                    "Text" => Ok(CachedData::Text(
                        serde_json::from_value(data).map_err(de::Error::custom)?,
                    )),
                    "Structured" => Ok(CachedData::Structured(data)),
                    "Compressed" => Ok(CachedData::Compressed(
                        serde_json::from_value(data).map_err(de::Error::custom)?,
                    )),
                    _ => Err(de::Error::unknown_variant(
                        &data_type,
                        &[
                            "Binary",
                            "Audio",
                            "ModelParams",
                            "Text",
                            "Structured",
                            "Compressed",
                        ],
                    )),
                }
            }
        }

        deserializer.deserialize_struct("CachedData", &["type", "data"], CachedDataVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_cache_basic_operations() {
        let mut cache = LruCache::new(3, 1000);

        assert!(cache.insert("key1".to_string(), "value1".to_string(), 10));
        assert!(cache.insert("key2".to_string(), "value2".to_string(), 10));
        assert!(cache.insert("key3".to_string(), "value3".to_string(), 10));

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&"key1".to_string()), Some(&"value1".to_string()));

        // This should evict key2 (least recently used)
        assert!(cache.insert("key4".to_string(), "value4".to_string(), 10));
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&"key2".to_string()), None);
    }

    #[test]
    fn test_conversion_cache_system() {
        let cache = ConversionCacheSystem::new();

        let audio_data = CachedData::Audio(vec![0.1, 0.2, 0.3, 0.4]);

        // Store data
        cache
            .store(
                "test_key".to_string(),
                audio_data.clone(),
                CacheItemType::AudioFeatures,
            )
            .unwrap();

        // Retrieve data
        let retrieved = cache.retrieve("test_key");
        assert!(retrieved.is_some());

        // Test cache miss
        let missing = cache.retrieve("nonexistent_key");
        assert!(missing.is_none());
    }

    #[test]
    fn test_cache_key_generation() {
        let cache = ConversionCacheSystem::new();

        let key1 = cache.create_cache_key(
            &ConversionType::PitchShift,
            0x1234567890abcdef,
            0xfedcba0987654321,
            5,
        );

        let key2 = cache.create_cache_key(
            &ConversionType::PitchShift,
            0x1234567890abcdef,
            0xfedcba0987654321,
            5,
        );

        assert_eq!(key1, key2); // Same parameters should generate same key

        let key3 = cache.create_cache_key(
            &ConversionType::SpeedTransformation,
            0x1234567890abcdef,
            0xfedcba0987654321,
            5,
        );

        assert_ne!(key1, key3); // Different conversion type should generate different key
    }

    #[test]
    fn test_audio_hashing() {
        let cache = ConversionCacheSystem::new();

        let audio1 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let audio2 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let audio3 = vec![0.1, 0.2, 0.3, 0.4, 0.6]; // Different last sample

        let hash1 = cache.hash_audio_data(&audio1);
        let hash2 = cache.hash_audio_data(&audio2);
        let hash3 = cache.hash_audio_data(&audio3);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_cache_statistics() {
        let cache = ConversionCacheSystem::new();

        // Perform some operations
        let audio_data = CachedData::Audio(vec![0.1, 0.2, 0.3]);
        cache
            .store("key1".to_string(), audio_data, CacheItemType::AudioFeatures)
            .unwrap();

        let _retrieved = cache.retrieve("key1"); // Hit
        let _missing = cache.retrieve("key2"); // Miss

        let stats = cache.get_statistics();
        assert!(stats.total_requests > 0);
        assert!(stats.l1_stats.hits > 0 || stats.l2_stats.hits > 0);
    }

    #[test]
    fn test_cache_optimization() {
        let cache = ConversionCacheSystem::new();

        // Store some data
        let audio_data = CachedData::Audio(vec![0.1; 1000]);
        cache
            .store("key1".to_string(), audio_data, CacheItemType::AudioFeatures)
            .unwrap();

        // Run optimization
        cache.optimize();

        // Cache should still function normally after optimization
        let retrieved = cache.retrieve("key1");
        assert!(retrieved.is_some());
    }
}
