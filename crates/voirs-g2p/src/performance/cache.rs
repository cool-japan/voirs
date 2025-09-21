//! Cache functionality for G2P performance optimization

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::{G2p, G2pMetadata, LanguageCode, Phoneme, Result};
use async_trait::async_trait;
use serde::Serialize;

/// Cache statistics for monitoring performance
#[derive(Debug, Clone, Default, Serialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_size: usize,
}

impl CacheStats {
    /// Get cache hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Get cache miss rate (0.0 to 1.0)
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }
}

/// Cache entry with timestamp for LRU eviction
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    value: T,
    timestamp: Instant,
    access_count: u64,
}

/// Advanced cache eviction strategies
#[derive(Debug, Clone, Copy)]
pub enum EvictionStrategy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time-based eviction (TTL)
    TTL,
    /// Adaptive cache based on access patterns
    Adaptive,
}

/// Advanced cache configuration
#[derive(Debug, Clone)]
pub struct AdvancedCacheConfig {
    pub max_size: usize,
    pub ttl: Option<Duration>,
    pub eviction_strategy: EvictionStrategy,
    pub preload_popular: bool,
    pub adaptive_sizing: bool,
    pub stats_collection: bool,
}

impl Default for AdvancedCacheConfig {
    fn default() -> Self {
        Self {
            max_size: 10000,
            ttl: Some(Duration::from_secs(3600)), // 1 hour
            eviction_strategy: EvictionStrategy::Adaptive,
            preload_popular: true,
            adaptive_sizing: true,
            stats_collection: true,
        }
    }
}

/// Enhanced cache entry with access frequency tracking
#[derive(Debug, Clone)]
struct AdvancedCacheEntry<T> {
    value: T,
    timestamp: Instant,
    last_access: Instant,
    access_count: u64,
    access_frequency: f64,
    size_estimate: usize,
}

impl<T> AdvancedCacheEntry<T> {
    fn new(value: T, size_estimate: usize) -> Self {
        let now = Instant::now();
        Self {
            value,
            timestamp: now,
            last_access: now,
            access_count: 1,
            access_frequency: 1.0,
            size_estimate,
        }
    }

    fn access(&mut self) {
        let now = Instant::now();
        let time_since_last = now.duration_since(self.last_access).as_secs_f64();

        // Update access frequency using exponential moving average
        self.access_frequency = 0.8 * self.access_frequency + 0.2 * (1.0 / (time_since_last + 1.0));
        self.access_count += 1;
        self.last_access = now;
    }

    fn score(&self, strategy: EvictionStrategy) -> f64 {
        match strategy {
            EvictionStrategy::LRU => self.last_access.elapsed().as_secs_f64(),
            EvictionStrategy::LFU => -(self.access_count as f64),
            EvictionStrategy::TTL => self.timestamp.elapsed().as_secs_f64(),
            EvictionStrategy::Adaptive => {
                // Combine recency, frequency, and size
                let recency_score = self.last_access.elapsed().as_secs_f64();
                let frequency_score = -(self.access_frequency);
                let size_penalty = (self.size_estimate as f64).sqrt();

                recency_score + frequency_score + size_penalty * 0.1
            }
        }
    }
}

/// High-performance advanced cache for G2P pronunciations
pub struct AdvancedG2pCache<K, V> {
    cache: Arc<Mutex<HashMap<K, AdvancedCacheEntry<V>>>>,
    config: AdvancedCacheConfig,
    stats: Arc<Mutex<CacheStats>>,
    popular_keys: Arc<Mutex<Vec<K>>>,
}

impl<K, V> AdvancedG2pCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Create a new advanced cache with configuration
    pub fn new(config: AdvancedCacheConfig) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            config,
            stats: Arc::new(Mutex::new(CacheStats::default())),
            popular_keys: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get value from cache with advanced access tracking
    pub fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        if let Some(entry) = cache.get_mut(key) {
            entry.access();
            stats.hits += 1;
            Some(entry.value.clone())
        } else {
            stats.misses += 1;
            None
        }
    }

    /// Insert value into cache with intelligent eviction
    pub fn insert(&self, key: K, value: V, size_estimate: usize) {
        let mut cache = self.cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        // Check if we need to evict
        if cache.len() >= self.config.max_size {
            self.evict_intelligently(&mut cache, &mut stats);
        }

        let entry = AdvancedCacheEntry::new(value, size_estimate);
        cache.insert(key.clone(), entry);

        // Update popular keys if enabled
        if self.config.preload_popular {
            let mut popular = self.popular_keys.lock().unwrap();
            if !popular.contains(&key) {
                popular.push(key);
                if popular.len() > 100 {
                    popular.remove(0); // Keep only recent popular keys
                }
            }
        }

        stats.total_size = cache.len();
    }

    /// Intelligent eviction based on configured strategy
    fn evict_intelligently(
        &self,
        cache: &mut HashMap<K, AdvancedCacheEntry<V>>,
        stats: &mut CacheStats,
    ) {
        if cache.is_empty() {
            return;
        }

        // Find the best candidate for eviction
        let mut best_key = None;
        let mut best_score = f64::NEG_INFINITY;

        for (key, entry) in cache.iter() {
            let score = entry.score(self.config.eviction_strategy);
            if score > best_score {
                best_score = score;
                best_key = Some(key.clone());
            }
        }

        if let Some(key) = best_key {
            cache.remove(&key);
            stats.evictions += 1;
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear cache
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();

        let mut stats = self.stats.lock().unwrap();
        *stats = CacheStats::default();
    }

    /// Get current cache size
    pub fn size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Preload popular pronunciations
    pub fn preload(&self, popular_items: Vec<(K, V, usize)>) {
        for (key, value, size) in popular_items {
            self.insert(key, value, size);
        }
    }

    /// Optimize cache based on access patterns
    pub fn optimize(&self) {
        if !self.config.adaptive_sizing {
            return;
        }

        let cache = self.cache.lock().unwrap();
        let stats = self.stats.lock().unwrap();

        // If hit rate is low, consider adjusting strategy or size
        if stats.hit_rate() < 0.5 && cache.len() > 1000 {
            // Could implement dynamic strategy switching here
            // For now, just log the optimization opportunity
            tracing::info!(
                "Cache optimization opportunity detected: hit_rate={:.2}, size={}",
                stats.hit_rate(),
                cache.len()
            );
        }
    }
}

/// High-performance LRU cache for G2P pronunciations (legacy interface)
pub struct G2pCache<K, V> {
    cache: Arc<Mutex<HashMap<K, CacheEntry<V>>>>,
    max_size: usize,
    ttl: Option<Duration>,
    stats: Arc<Mutex<CacheStats>>,
}

impl<K, V> G2pCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Create new cache with specified capacity
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::with_capacity(max_size))),
            max_size,
            ttl: None,
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }

    /// Create cache with TTL (time-to-live)
    pub fn with_ttl(max_size: usize, ttl: Duration) -> Self {
        let mut cache = Self::new(max_size);
        cache.ttl = Some(ttl);
        cache
    }

    /// Get value from cache
    pub fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        if let Some(entry) = cache.get_mut(key) {
            // Check TTL expiration
            if let Some(ttl) = self.ttl {
                if entry.timestamp.elapsed() > ttl {
                    cache.remove(key);
                    stats.misses += 1;
                    return None;
                }
            }

            // Update access timestamp and count
            entry.timestamp = Instant::now();
            entry.access_count += 1;

            stats.hits += 1;
            Some(entry.value.clone())
        } else {
            stats.misses += 1;
            None
        }
    }

    /// Insert value into cache
    pub fn insert(&self, key: K, value: V) {
        let mut cache = self.cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        // Evict oldest entries if cache is full
        if cache.len() >= self.max_size {
            self.evict_lru(&mut cache, &mut stats);
        }

        let entry = CacheEntry {
            value,
            timestamp: Instant::now(),
            access_count: 1,
        };

        cache.insert(key, entry);
        stats.total_size = cache.len();
    }

    /// Evict least recently used entries
    fn evict_lru(&self, cache: &mut HashMap<K, CacheEntry<V>>, stats: &mut CacheStats) {
        // Find the least recently used entry
        let mut oldest_key = None;
        let mut oldest_time = Instant::now();

        for (key, entry) in cache.iter() {
            if entry.timestamp < oldest_time {
                oldest_time = entry.timestamp;
                oldest_key = Some(key.clone());
            }
        }

        if let Some(key) = oldest_key {
            cache.remove(&key);
            stats.evictions += 1;
        }
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        cache.clear();
        stats.total_size = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get current cache size
    pub fn size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Get cache capacity
    pub fn capacity(&self) -> usize {
        self.max_size
    }

    /// Batch insert multiple key-value pairs efficiently
    pub fn batch_insert(&self, items: Vec<(K, V)>) {
        let mut cache = self.cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        for (key, value) in items {
            // Evict oldest entries if cache is full
            if cache.len() >= self.max_size {
                self.evict_lru(&mut cache, &mut stats);
            }

            let entry = CacheEntry {
                value,
                timestamp: Instant::now(),
                access_count: 1,
            };

            cache.insert(key, entry);
        }

        stats.total_size = cache.len();
    }
}

/// Cached G2P wrapper for any G2P backend
pub struct CachedG2p<T> {
    backend: T,
    cache: G2pCache<String, Vec<Phoneme>>,
    cache_by_language: bool,
}

impl<T> CachedG2p<T> {
    /// Create new cached G2P wrapper
    pub fn new(backend: T, cache_size: usize) -> Self {
        Self {
            backend,
            cache: G2pCache::new(cache_size),
            cache_by_language: true,
        }
    }

    /// Create cached G2P with TTL
    pub fn with_ttl(backend: T, cache_size: usize, ttl: Duration) -> Self {
        Self {
            backend,
            cache: G2pCache::with_ttl(cache_size, ttl),
            cache_by_language: true,
        }
    }

    /// Enable/disable language-specific caching
    pub fn set_cache_by_language(&mut self, enabled: bool) {
        self.cache_by_language = enabled;
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Generate cache key for text and language
    fn make_cache_key(&self, text: &str, lang: Option<LanguageCode>) -> String {
        if self.cache_by_language {
            if let Some(lang) = lang {
                format!("{}:{text}", lang.as_str())
            } else {
                format!("default:{text}")
            }
        } else {
            text.to_string()
        }
    }
}

#[async_trait]
impl<T: G2p + Send + Sync> G2p for CachedG2p<T> {
    async fn to_phonemes(&self, text: &str, lang: Option<LanguageCode>) -> Result<Vec<Phoneme>> {
        let cache_key = self.make_cache_key(text, lang);

        // Check cache first
        if let Some(phonemes) = self.cache.get(&cache_key) {
            return Ok(phonemes);
        }

        // Cache miss - get from backend
        let phonemes = self.backend.to_phonemes(text, lang).await?;

        // Store in cache
        self.cache.insert(cache_key, phonemes.clone());

        Ok(phonemes)
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        self.backend.supported_languages()
    }

    fn metadata(&self) -> G2pMetadata {
        let mut metadata = self.backend.metadata();
        metadata.name = format!("Cached {}", metadata.name);
        metadata.description = format!("Cached wrapper for {}", metadata.description);
        metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_cache_basic_operations() {
        let cache = G2pCache::new(10);

        // Test insertion and retrieval
        cache.insert("key1".to_string(), "value1".to_string());
        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));

        // Test miss
        assert_eq!(cache.get(&"nonexistent".to_string()), None);

        // Test stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = G2pCache::new(2);

        // Fill cache
        cache.insert("key1".to_string(), "value1".to_string());
        cache.insert("key2".to_string(), "value2".to_string());
        assert_eq!(cache.size(), 2);

        // Trigger eviction
        cache.insert("key3".to_string(), "value3".to_string());
        assert_eq!(cache.size(), 2);

        // Check eviction stats
        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn test_cache_ttl() {
        let cache = G2pCache::with_ttl(10, Duration::from_millis(1));

        cache.insert("key1".to_string(), "value1".to_string());
        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));

        // Wait for TTL expiration
        std::thread::sleep(Duration::from_millis(2));

        // Should be expired
        assert_eq!(cache.get(&"key1".to_string()), None);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }
}
