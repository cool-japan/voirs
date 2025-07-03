//! Performance optimizations for G2P conversion

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::{Phoneme, LanguageCode, Result, G2p, G2pMetadata};
use async_trait::async_trait;

/// Cache statistics for monitoring performance
#[derive(Debug, Clone, Default)]
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

/// High-performance LRU cache for G2P pronunciations
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
                format!("{}:{}", lang.as_str(), text)
            } else {
                format!("default:{}", text)
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

/// Batch processing utilities for high-throughput G2P conversion
pub struct BatchProcessor;

impl BatchProcessor {
    /// Process multiple texts in parallel
    pub async fn process_batch<T: crate::G2p + Send + Sync + 'static>(
        backend: Arc<T>,
        texts: Vec<String>,
        language: Option<LanguageCode>,
        max_concurrent: usize,
    ) -> Result<Vec<Result<Vec<Phoneme>>>> {
        use tokio::task;
        
        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        let mut handles = Vec::new();
        
        for text in texts {
            let backend = backend.clone();
            let semaphore = semaphore.clone();
            
            let handle = task::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                backend.to_phonemes(&text, language).await
            });
            
            handles.push(handle);
        }
        
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap();
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// Process texts with different languages in parallel
    pub async fn process_multilingual_batch<T: crate::G2p + Send + Sync + 'static>(
        backend: Arc<T>,
        texts_with_langs: Vec<(String, Option<LanguageCode>)>,
        max_concurrent: usize,
    ) -> Result<Vec<Result<Vec<Phoneme>>>> {
        use tokio::task;
        
        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        let mut handles = Vec::new();
        
        for (text, lang) in texts_with_langs {
            let backend = backend.clone();
            let semaphore = semaphore.clone();
            
            let handle = task::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                backend.to_phonemes(&text, lang).await
            });
            
            handles.push(handle);
        }
        
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap();
            results.push(result);
        }
        
        Ok(results)
    }
}

/// SIMD-accelerated text processing utilities
pub mod simd {
    /// Check if SIMD is available on this platform
    pub fn is_simd_available() -> bool {
        cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64")
    }
    
    /// Fast character filtering using SIMD (when available)
    pub fn filter_alphabetic(text: &str) -> String {
        if !is_simd_available() {
            return text.chars().filter(|c| c.is_alphabetic()).collect();
        }
        
        // For now, fall back to standard implementation
        // TODO: Add actual SIMD implementation using portable_simd when stable
        text.chars().filter(|c| c.is_alphabetic()).collect()
    }
    
    /// Fast whitespace normalization using SIMD (when available)
    pub fn normalize_whitespace(text: &str) -> String {
        if !is_simd_available() {
            return text.split_whitespace().collect::<Vec<_>>().join(" ");
        }
        
        // For now, fall back to standard implementation
        // TODO: Add actual SIMD implementation using portable_simd when stable
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DummyG2p, G2p};
    use std::time::Duration;
    
    #[test]
    fn test_cache_basic_operations() {
        let cache: G2pCache<String, Vec<Phoneme>> = G2pCache::new(100);
        
        // Test miss
        assert!(cache.get(&"hello".to_string()).is_none());
        
        // Test insert and hit
        let phonemes = vec![Phoneme::new("h"), Phoneme::new("e")];
        cache.insert("hello".to_string(), phonemes.clone());
        assert_eq!(cache.get(&"hello".to_string()), Some(phonemes));
        
        // Test stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.total_size, 1);
    }
    
    #[test]
    fn test_cache_eviction() {
        let cache: G2pCache<String, Vec<Phoneme>> = G2pCache::new(2);
        
        // Fill cache to capacity
        cache.insert("a".to_string(), vec![Phoneme::new("a")]);
        cache.insert("b".to_string(), vec![Phoneme::new("b")]);
        assert_eq!(cache.size(), 2);
        
        // Insert one more to trigger eviction
        cache.insert("c".to_string(), vec![Phoneme::new("c")]);
        assert_eq!(cache.size(), 2);
        
        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
    }
    
    #[test]
    fn test_cache_ttl() {
        let cache: G2pCache<String, Vec<Phoneme>> = G2pCache::with_ttl(100, Duration::from_millis(1));
        
        // Insert and immediately get
        cache.insert("test".to_string(), vec![Phoneme::new("t")]);
        assert!(cache.get(&"test".to_string()).is_some());
        
        // Wait for TTL to expire
        std::thread::sleep(Duration::from_millis(2));
        assert!(cache.get(&"test".to_string()).is_none());
    }
    
    #[tokio::test]
    async fn test_batch_processing() {
        let backend = Arc::new(DummyG2p::new());
        let texts = vec!["hello".to_string(), "world".to_string()];
        
        let results = BatchProcessor::process_batch(
            backend, 
            texts, 
            Some(LanguageCode::EnUs), 
            2
        ).await.unwrap();
        
        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
    }
    
    #[test]
    fn test_simd_availability() {
        // Just test that the function doesn't panic
        let _ = simd::is_simd_available();
    }
    
    #[test]
    fn test_simd_filter_alphabetic() {
        let text = "hello123world!";
        let filtered = simd::filter_alphabetic(text);
        assert_eq!(filtered, "helloworld");
    }
    
    #[test]
    fn test_simd_normalize_whitespace() {
        let text = "hello   world  \t  foo";
        let normalized = simd::normalize_whitespace(text);
        assert_eq!(normalized, "hello world foo");
    }
}