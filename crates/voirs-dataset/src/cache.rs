//! LRU cache for dataset samples
//!
//! This module provides a Least Recently Used (LRU) cache for storing frequently
//! accessed dataset samples to improve performance.

use crate::{DatasetSample, Result};
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, Mutex};

/// LRU cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total number of cache hits
    pub hits: usize,
    /// Total number of cache misses
    pub misses: usize,
    /// Total number of evictions
    pub evictions: usize,
    /// Current cache size
    pub current_size: usize,
    /// Maximum cache capacity
    pub capacity: usize,
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

    /// Calculate cache miss rate
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
    }
}

/// LRU cache for dataset samples
pub struct LruCache<K: Hash + Eq + Clone, V: Clone> {
    cache: HashMap<K, V>,
    order: VecDeque<K>,
    capacity: usize,
    stats: CacheStats,
}

impl<K: Hash + Eq + Clone, V: Clone> LruCache<K, V> {
    /// Create a new LRU cache with the specified capacity
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Cache capacity must be greater than 0");

        Self {
            cache: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            capacity,
            stats: CacheStats {
                capacity,
                ..Default::default()
            },
        }
    }

    /// Get a value from the cache
    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some(value) = self.cache.get(key) {
            // Clone the value first to avoid borrow conflicts
            let value = value.clone();
            // Update access order
            self.move_to_front(key);
            self.stats.hits += 1;
            Some(value)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert a key-value pair into the cache
    pub fn insert(&mut self, key: K, value: V) {
        if self.cache.contains_key(&key) {
            // Update existing entry
            self.cache.insert(key.clone(), value);
            self.move_to_front(&key);
        } else {
            // Insert new entry
            if self.cache.len() >= self.capacity {
                self.evict_lru();
            }

            self.cache.insert(key.clone(), value);
            self.order.push_front(key);
        }

        self.stats.current_size = self.cache.len();
    }

    /// Remove a key from the cache
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(value) = self.cache.remove(key) {
            self.order.retain(|k| k != key);
            self.stats.current_size = self.cache.len();
            Some(value)
        } else {
            None
        }
    }

    /// Clear all entries from the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.order.clear();
        self.stats.current_size = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get mutable cache statistics
    pub fn stats_mut(&mut self) -> &mut CacheStats {
        &mut self.stats
    }

    /// Get cache capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get current cache size
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Move a key to the front of the access order
    fn move_to_front(&mut self, key: &K) {
        self.order.retain(|k| k != key);
        self.order.push_front(key.clone());
    }

    /// Evict the least recently used entry
    fn evict_lru(&mut self) {
        if let Some(old_key) = self.order.pop_back() {
            self.cache.remove(&old_key);
            self.stats.evictions += 1;
        }
    }
}

/// Thread-safe LRU cache for dataset samples
#[derive(Clone)]
pub struct SharedLruCache<K: Hash + Eq + Clone, V: Clone> {
    inner: Arc<Mutex<LruCache<K, V>>>,
}

impl<K: Hash + Eq + Clone, V: Clone> SharedLruCache<K, V> {
    /// Create a new shared LRU cache
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LruCache::new(capacity))),
        }
    }

    /// Get a value from the cache
    pub fn get(&self, key: &K) -> Option<V> {
        self.inner.lock().ok()?.get(key)
    }

    /// Insert a key-value pair into the cache
    pub fn insert(&self, key: K, value: V) {
        if let Ok(mut cache) = self.inner.lock() {
            cache.insert(key, value);
        }
    }

    /// Remove a key from the cache
    pub fn remove(&self, key: &K) -> Option<V> {
        self.inner.lock().ok()?.remove(key)
    }

    /// Clear all entries from the cache
    pub fn clear(&self) {
        if let Ok(mut cache) = self.inner.lock() {
            cache.clear();
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.inner
            .lock()
            .ok()
            .map(|cache| cache.stats().clone())
            .unwrap_or_default()
    }

    /// Get cache capacity
    pub fn capacity(&self) -> usize {
        self.inner
            .lock()
            .ok()
            .map(|cache| cache.capacity())
            .unwrap_or(0)
    }

    /// Get current cache size
    pub fn len(&self) -> usize {
        self.inner.lock().ok().map(|cache| cache.len()).unwrap_or(0)
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.inner
            .lock()
            .ok()
            .map(|cache| cache.is_empty())
            .unwrap_or(true)
    }
}

/// Specialized cache for dataset samples
pub type SampleCache = LruCache<String, DatasetSample>;

/// Thread-safe specialized cache for dataset samples
pub type SharedSampleCache = SharedLruCache<String, DatasetSample>;

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache
    pub capacity: usize,
    /// Whether to enable caching
    pub enabled: bool,
    /// Whether to use thread-safe shared cache
    pub thread_safe: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            capacity: 1000,
            enabled: true,
            thread_safe: false,
        }
    }
}

impl CacheConfig {
    /// Create a new cache configuration
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            enabled: true,
            thread_safe: false,
        }
    }

    /// Enable thread-safe caching
    pub fn with_thread_safe(mut self, thread_safe: bool) -> Self {
        self.thread_safe = thread_safe;
        self
    }

    /// Enable or disable caching
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioData, LanguageCode, QualityMetrics};

    fn create_test_sample(id: &str) -> DatasetSample {
        DatasetSample {
            id: id.to_string(),
            audio: AudioData::new(vec![0.0; 100], 16000, 1),
            text: format!("Test sample {}", id),
            speaker: None,
            language: LanguageCode::EnUs,
            quality: QualityMetrics::default(),
            phonemes: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_lru_cache_creation() {
        let cache: LruCache<String, i32> = LruCache::new(10);
        assert_eq!(cache.capacity(), 10);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_lru_cache_insert_and_get() {
        let mut cache: LruCache<String, i32> = LruCache::new(3);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3);

        assert_eq!(cache.get(&"a".to_string()), Some(1));
        assert_eq!(cache.get(&"b".to_string()), Some(2));
        assert_eq!(cache.get(&"c".to_string()), Some(3));
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_lru_cache_eviction() {
        let mut cache: LruCache<String, i32> = LruCache::new(3);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3);
        cache.insert("d".to_string(), 4); // Should evict "a"

        assert_eq!(cache.get(&"a".to_string()), None);
        assert_eq!(cache.get(&"b".to_string()), Some(2));
        assert_eq!(cache.get(&"c".to_string()), Some(3));
        assert_eq!(cache.get(&"d".to_string()), Some(4));
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_lru_cache_access_order() {
        let mut cache: LruCache<String, i32> = LruCache::new(3);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3);

        // Access "a" to move it to front
        cache.get(&"a".to_string());

        // Insert "d", should evict "b" (least recently used)
        cache.insert("d".to_string(), 4);

        assert_eq!(cache.get(&"a".to_string()), Some(1));
        assert_eq!(cache.get(&"b".to_string()), None);
        assert_eq!(cache.get(&"c".to_string()), Some(3));
        assert_eq!(cache.get(&"d".to_string()), Some(4));
    }

    #[test]
    fn test_lru_cache_update() {
        let mut cache: LruCache<String, i32> = LruCache::new(3);

        cache.insert("a".to_string(), 1);
        cache.insert("a".to_string(), 10); // Update value

        assert_eq!(cache.get(&"a".to_string()), Some(10));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_cache_remove() {
        let mut cache: LruCache<String, i32> = LruCache::new(3);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);

        assert_eq!(cache.remove(&"a".to_string()), Some(1));
        assert_eq!(cache.get(&"a".to_string()), None);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_cache_clear() {
        let mut cache: LruCache<String, i32> = LruCache::new(3);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3);

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.get(&"a".to_string()), None);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache: LruCache<String, i32> = LruCache::new(2);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);

        // Hits
        cache.get(&"a".to_string());
        cache.get(&"b".to_string());

        // Misses
        cache.get(&"c".to_string());
        cache.get(&"d".to_string());

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hit_rate(), 0.5);
        assert_eq!(stats.miss_rate(), 0.5);

        // Eviction
        cache.insert("c".to_string(), 3);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_sample_cache() {
        let mut cache: SampleCache = LruCache::new(10);

        let sample1 = create_test_sample("001");
        let sample2 = create_test_sample("002");

        cache.insert(sample1.id.clone(), sample1.clone());
        cache.insert(sample2.id.clone(), sample2.clone());

        let retrieved = cache.get(&"001".to_string()).unwrap();
        assert_eq!(retrieved.id, "001");
        assert_eq!(retrieved.text, "Test sample 001");
    }

    #[test]
    fn test_shared_lru_cache() {
        let cache: SharedLruCache<String, i32> = SharedLruCache::new(10);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);

        assert_eq!(cache.get(&"a".to_string()), Some(1));
        assert_eq!(cache.get(&"b".to_string()), Some(2));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_shared_cache_clone() {
        let cache1: SharedLruCache<String, i32> = SharedLruCache::new(10);
        cache1.insert("a".to_string(), 1);

        let cache2 = cache1.clone();
        assert_eq!(cache2.get(&"a".to_string()), Some(1));

        // Modifications in cache2 affect cache1
        cache2.insert("b".to_string(), 2);
        assert_eq!(cache1.get(&"b".to_string()), Some(2));
    }

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.capacity, 1000);
        assert!(config.enabled);
        assert!(!config.thread_safe);
    }

    #[test]
    fn test_cache_config_builder() {
        let config = CacheConfig::new(500)
            .with_thread_safe(true)
            .with_enabled(true);

        assert_eq!(config.capacity, 500);
        assert!(config.enabled);
        assert!(config.thread_safe);
    }

    #[test]
    #[should_panic(expected = "Cache capacity must be greater than 0")]
    fn test_lru_cache_zero_capacity() {
        let _cache: LruCache<String, i32> = LruCache::new(0);
    }

    #[test]
    fn test_stats_reset() {
        let mut cache: LruCache<String, i32> = LruCache::new(10);

        cache.insert("a".to_string(), 1);
        cache.get(&"a".to_string());
        cache.get(&"b".to_string());

        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);

        cache.stats_mut().reset();

        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
    }
}
