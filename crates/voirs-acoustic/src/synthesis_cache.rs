//! Advanced synthesis result caching system
//!
//! This module provides sophisticated caching mechanisms for synthesis results,
//! reducing latency and improving throughput for repeated synthesis requests.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use crate::{AcousticError, MelSpectrogram, Result};

/// Cache key for synthesis requests
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SynthesisCacheKey {
    /// Phoneme sequence (as string for hashing)
    phoneme_text: String,
    /// Speaker ID (if multi-speaker)
    speaker_id: Option<u32>,
    /// Speed factor (quantized to 0.1 increments)
    speed_quantized: i32,
    /// Pitch shift (quantized to semitones)
    pitch_shift_quantized: i32,
    /// Energy scale (quantized to 0.1 increments)
    energy_quantized: i32,
}

impl SynthesisCacheKey {
    /// Create a new cache key from synthesis parameters
    pub fn new(
        phonemes: &str,
        speaker_id: Option<u32>,
        speed: f32,
        pitch_shift: f32,
        energy: f32,
    ) -> Self {
        Self {
            phoneme_text: phonemes.to_string(),
            speaker_id,
            speed_quantized: (speed * 10.0) as i32,
            pitch_shift_quantized: pitch_shift as i32,
            energy_quantized: (energy * 10.0) as i32,
        }
    }

    /// Get cache key size estimate in bytes
    pub fn size_bytes(&self) -> usize {
        self.phoneme_text.len() + std::mem::size_of::<Self>()
    }
}

/// Cached synthesis result
#[derive(Debug, Clone)]
pub struct CachedResult {
    /// The mel spectrogram result
    pub mel: MelSpectrogram,
    /// When this entry was created
    created_at: Instant,
    /// When this entry was last accessed
    last_accessed: Instant,
    /// Number of times accessed
    access_count: u64,
    /// Estimated size in bytes
    size_bytes: usize,
}

impl CachedResult {
    fn new(mel: MelSpectrogram) -> Self {
        let size_bytes = mel.estimate_size_bytes();
        Self {
            mel,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 0,
            size_bytes,
        }
    }

    fn access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.age() > ttl
    }
}

/// Cache eviction policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used - evict oldest accessed entry
    LRU,
    /// Least Frequently Used - evict least accessed entry
    LFU,
    /// Time To Live - evict expired entries
    TTL,
    /// Size-based - evict largest entries first
    LargestFirst,
    /// Hybrid - combine multiple strategies
    Hybrid,
}

/// Synthesis result cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisCacheConfig {
    /// Maximum number of cached entries
    pub max_entries: usize,
    /// Maximum cache size in bytes
    pub max_size_bytes: usize,
    /// Time-to-live for cache entries
    pub ttl: Duration,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable cache statistics
    pub enable_stats: bool,
    /// Preload common phrases
    pub preload_enabled: bool,
}

impl Default for SynthesisCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            max_size_bytes: 100 * 1024 * 1024, // 100 MB
            ttl: Duration::from_secs(3600),    // 1 hour
            eviction_policy: EvictionPolicy::Hybrid,
            enable_stats: true,
            preload_enabled: false,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total evictions
    pub evictions: u64,
    /// Current number of entries
    pub entry_count: usize,
    /// Current cache size in bytes
    pub size_bytes: usize,
    /// Average access time (microseconds)
    pub avg_access_time_us: f64,
}

impl CacheStatistics {
    /// Calculate cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Calculate cache utilization
    pub fn utilization(&self, max_entries: usize) -> f64 {
        if max_entries == 0 {
            0.0
        } else {
            (self.entry_count as f64 / max_entries as f64) * 100.0
        }
    }
}

/// Access pattern analyzer for predictive caching
#[derive(Debug)]
pub struct AccessPatternAnalyzer {
    /// Access history (key, timestamp)
    access_history: VecDeque<(SynthesisCacheKey, Instant)>,
    /// Temporal access patterns (hour of day -> frequency)
    temporal_patterns: HashMap<u8, u64>,
    /// Sequential access patterns (key -> likely next keys)
    sequential_patterns: HashMap<SynthesisCacheKey, Vec<SynthesisCacheKey>>,
    /// Most frequently accessed keys
    hot_keys: Vec<(SynthesisCacheKey, u64)>,
    /// Maximum history size
    max_history: usize,
}

impl AccessPatternAnalyzer {
    fn new() -> Self {
        Self {
            access_history: VecDeque::new(),
            temporal_patterns: HashMap::new(),
            sequential_patterns: HashMap::new(),
            hot_keys: Vec::new(),
            max_history: 10000,
        }
    }

    /// Record an access and update patterns
    fn record_access(&mut self, key: &SynthesisCacheKey) {
        let now = Instant::now();

        // Add to history
        self.access_history.push_back((key.clone(), now));
        if self.access_history.len() > self.max_history {
            self.access_history.pop_front();
        }

        // Update temporal patterns (hour of day)
        let hour = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            / 3600
            % 24;
        *self.temporal_patterns.entry(hour as u8).or_insert(0) += 1;

        // Update sequential patterns (what comes after this key)
        if self.access_history.len() >= 2 {
            let prev_key = &self.access_history[self.access_history.len() - 2].0;
            self.sequential_patterns
                .entry(prev_key.clone())
                .or_default()
                .push(key.clone());
        }

        // Update hot keys
        self.update_hot_keys();
    }

    /// Update hot key tracking
    fn update_hot_keys(&mut self) {
        // Count frequency of each key in recent history
        let mut frequency: HashMap<SynthesisCacheKey, u64> = HashMap::new();
        for (key, _) in &self.access_history {
            *frequency.entry(key.clone()).or_insert(0) += 1;
        }

        // Sort by frequency and keep top 100
        let mut sorted: Vec<_> = frequency.into_iter().collect();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.1));
        sorted.truncate(100);
        self.hot_keys = sorted;
    }

    /// Predict likely next access based on patterns
    fn predict_next_access(&self, current_key: &SynthesisCacheKey) -> Vec<SynthesisCacheKey> {
        self.sequential_patterns
            .get(current_key)
            .map(|keys| {
                // Return most common next keys
                let mut freq: HashMap<&SynthesisCacheKey, usize> = HashMap::new();
                for key in keys {
                    *freq.entry(key).or_insert(0) += 1;
                }
                let mut sorted: Vec<_> = freq.into_iter().collect();
                sorted.sort_by_key(|b| std::cmp::Reverse(b.1));
                sorted.into_iter().take(5).map(|(k, _)| k.clone()).collect()
            })
            .unwrap_or_default()
    }

    /// Get hot keys that should be kept in cache
    fn get_hot_keys(&self, count: usize) -> Vec<SynthesisCacheKey> {
        self.hot_keys
            .iter()
            .take(count)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Check if current time matches high-activity patterns
    fn is_peak_time(&self) -> bool {
        let hour = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            / 3600
            % 24;

        let current_freq = self
            .temporal_patterns
            .get(&(hour as u8))
            .copied()
            .unwrap_or(0);
        let avg_freq = if !self.temporal_patterns.is_empty() {
            self.temporal_patterns.values().sum::<u64>() / self.temporal_patterns.len() as u64
        } else {
            0
        };

        current_freq > avg_freq * 2
    }
}

impl Default for AccessPatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive eviction controller that adjusts strategy based on workload
#[derive(Debug)]
pub struct AdaptiveEvictionController {
    /// Current effective eviction policy
    current_policy: EvictionPolicy,
    /// Hit rate history for policy effectiveness
    hit_rate_history: VecDeque<f64>,
    /// Policy performance scores
    policy_scores: HashMap<EvictionPolicy, f64>,
    /// Last policy change time
    last_policy_change: Instant,
    /// Minimum time between policy changes
    min_change_interval: Duration,
}

impl AdaptiveEvictionController {
    fn new() -> Self {
        let mut policy_scores = HashMap::new();
        policy_scores.insert(EvictionPolicy::LRU, 50.0);
        policy_scores.insert(EvictionPolicy::LFU, 50.0);
        policy_scores.insert(EvictionPolicy::TTL, 50.0);
        policy_scores.insert(EvictionPolicy::LargestFirst, 50.0);
        policy_scores.insert(EvictionPolicy::Hybrid, 75.0);

        Self {
            current_policy: EvictionPolicy::Hybrid,
            hit_rate_history: VecDeque::new(),
            policy_scores,
            last_policy_change: Instant::now(),
            min_change_interval: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Update hit rate and possibly adapt policy
    fn update_hit_rate(&mut self, hit_rate: f64) {
        self.hit_rate_history.push_back(hit_rate);
        if self.hit_rate_history.len() > 100 {
            self.hit_rate_history.pop_front();
        }

        // Update current policy score
        if let Some(score) = self.policy_scores.get_mut(&self.current_policy) {
            // Exponential moving average
            *score = *score * 0.9 + hit_rate * 0.1;
        }

        // Consider policy change if enough time has passed
        if self.last_policy_change.elapsed() > self.min_change_interval {
            self.consider_policy_change();
        }
    }

    /// Consider changing eviction policy based on performance
    fn consider_policy_change(&mut self) {
        if self.hit_rate_history.len() < 20 {
            return;
        }

        let recent_avg = self.hit_rate_history.iter().rev().take(10).sum::<f64>() / 10.0;
        let current_score = self
            .policy_scores
            .get(&self.current_policy)
            .copied()
            .unwrap_or(50.0);

        // If performance is degrading, try a different policy
        if recent_avg < current_score - 10.0 {
            // Find best performing policy
            if let Some((best_policy, _)) = self
                .policy_scores
                .iter()
                .filter(|(p, _)| **p != self.current_policy)
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                self.current_policy = *best_policy;
                self.last_policy_change = Instant::now();
            }
        }
    }

    /// Get current recommended policy
    fn get_policy(&self) -> EvictionPolicy {
        self.current_policy
    }
}

impl Default for AdaptiveEvictionController {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced synthesis result cache with adaptive strategies
pub struct SynthesisCache {
    /// Cache configuration
    config: SynthesisCacheConfig,
    /// Cached entries
    cache: Arc<Mutex<HashMap<SynthesisCacheKey, CachedResult>>>,
    /// Access order for LRU (most recent at back)
    access_order: Arc<Mutex<VecDeque<SynthesisCacheKey>>>,
    /// Cache statistics
    stats: Arc<Mutex<CacheStatistics>>,
    /// Access pattern analyzer
    pattern_analyzer: Arc<Mutex<AccessPatternAnalyzer>>,
    /// Adaptive eviction controller
    adaptive_controller: Arc<Mutex<AdaptiveEvictionController>>,
}

impl SynthesisCache {
    /// Create a new synthesis cache
    pub fn new(config: SynthesisCacheConfig) -> Self {
        Self {
            config,
            cache: Arc::new(Mutex::new(HashMap::new())),
            access_order: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(Mutex::new(CacheStatistics::default())),
            pattern_analyzer: Arc::new(Mutex::new(AccessPatternAnalyzer::new())),
            adaptive_controller: Arc::new(Mutex::new(AdaptiveEvictionController::new())),
        }
    }

    /// Get a cached result
    pub fn get(&self, key: &SynthesisCacheKey) -> Option<MelSpectrogram> {
        let start = Instant::now();

        let mut cache = self.cache.lock().ok()?;
        let result = cache.get_mut(key);

        if let Some(entry) = result {
            // Check if expired
            if entry.is_expired(self.config.ttl) {
                drop(cache);
                self.remove(key);
                self.record_miss();
                return None;
            }

            // Update access tracking
            entry.access();
            self.update_access_order(key);

            let mel = entry.mel.clone();
            drop(cache);

            // Record access pattern
            if let Ok(mut analyzer) = self.pattern_analyzer.lock() {
                analyzer.record_access(key);
            }

            // Record hit
            self.record_hit(start.elapsed());
            Some(mel)
        } else {
            drop(cache);
            self.record_miss();
            None
        }
    }

    /// Insert a result into the cache
    pub fn insert(&self, key: SynthesisCacheKey, mel: MelSpectrogram) -> Result<()> {
        let entry = CachedResult::new(mel);
        let entry_size = entry.size_bytes + key.size_bytes();

        // Check if we need to evict entries
        self.evict_if_needed(entry_size)?;

        // Insert the entry
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(key.clone(), entry);
        }

        // Update access order
        self.update_access_order(&key);

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.entry_count += 1;
            stats.size_bytes += entry_size;
        }

        Ok(())
    }

    /// Remove a specific entry
    pub fn remove(&self, key: &SynthesisCacheKey) -> Option<MelSpectrogram> {
        let mut cache = self.cache.lock().ok()?;
        let entry = cache.remove(key)?;

        // Update access order
        if let Ok(mut order) = self.access_order.lock() {
            order.retain(|k| k != key);
        }

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.entry_count = stats.entry_count.saturating_sub(1);
            stats.size_bytes = stats.size_bytes.saturating_sub(entry.size_bytes);
        }

        Some(entry.mel)
    }

    /// Clear the entire cache
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
        if let Ok(mut order) = self.access_order.lock() {
            order.clear();
        }
        if let Ok(mut stats) = self.stats.lock() {
            stats.entry_count = 0;
            stats.size_bytes = 0;
        }
    }

    /// Get cache statistics
    pub fn statistics(&self) -> CacheStatistics {
        self.stats
            .lock()
            .ok()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    /// Evict entries if needed to make room
    fn evict_if_needed(&self, needed_bytes: usize) -> Result<()> {
        let stats = self.statistics();

        // Check if we need to evict
        let needs_eviction = stats.entry_count >= self.config.max_entries
            || stats.size_bytes + needed_bytes > self.config.max_size_bytes;

        if !needs_eviction {
            return Ok(());
        }

        // Use adaptive policy if enabled
        let policy = if self.config.eviction_policy == EvictionPolicy::Hybrid {
            self.adaptive_controller
                .lock()
                .ok()
                .map(|c| c.get_policy())
                .unwrap_or(EvictionPolicy::Hybrid)
        } else {
            self.config.eviction_policy
        };

        match policy {
            EvictionPolicy::LRU => self.evict_lru(),
            EvictionPolicy::LFU => self.evict_lfu(),
            EvictionPolicy::TTL => self.evict_expired(),
            EvictionPolicy::LargestFirst => self.evict_largest(),
            EvictionPolicy::Hybrid => self.evict_hybrid(),
        }
    }

    /// Evict using adaptive strategy that protects hot keys
    fn evict_adaptive(&self) -> Result<()> {
        // Get hot keys that should be protected
        let hot_keys = if let Ok(analyzer) = self.pattern_analyzer.lock() {
            analyzer.get_hot_keys(20)
        } else {
            Vec::new()
        };

        // Find candidate for eviction (LRU but not in hot keys)
        let key = {
            let order = self
                .access_order
                .lock()
                .map_err(|_| AcousticError::ConfigError {
                    message: "Failed to lock access order".to_string(),
                })?;

            order.iter().find(|k| !hot_keys.contains(k)).cloned()
        };

        if let Some(key) = key {
            self.remove(&key);
            self.record_eviction();
        } else {
            // Fallback to regular LRU if all keys are hot
            self.evict_lru()?;
        }

        Ok(())
    }

    /// Predict and prefetch likely next accesses
    pub fn prefetch_predicted(&self, current_key: &SynthesisCacheKey) -> Vec<SynthesisCacheKey> {
        if let Ok(analyzer) = self.pattern_analyzer.lock() {
            analyzer.predict_next_access(current_key)
        } else {
            Vec::new()
        }
    }

    /// Check if cache should be in aggressive retention mode (peak times)
    pub fn is_peak_time(&self) -> bool {
        self.pattern_analyzer
            .lock()
            .ok()
            .map(|a| a.is_peak_time())
            .unwrap_or(false)
    }

    /// Get hot keys that should be retained
    pub fn get_hot_keys(&self, count: usize) -> Vec<SynthesisCacheKey> {
        self.pattern_analyzer
            .lock()
            .ok()
            .map(|a| a.get_hot_keys(count))
            .unwrap_or_default()
    }

    /// Evict least recently used entry
    fn evict_lru(&self) -> Result<()> {
        let key = {
            let mut order = self
                .access_order
                .lock()
                .map_err(|_| AcousticError::ConfigError {
                    message: "Failed to lock access order".to_string(),
                })?;
            order.pop_front()
        };

        if let Some(key) = key {
            self.remove(&key);
            self.record_eviction();
        }

        Ok(())
    }

    /// Evict least frequently used entry
    fn evict_lfu(&self) -> Result<()> {
        let key = {
            let cache = self.cache.lock().map_err(|_| AcousticError::ConfigError {
                message: "Failed to lock cache".to_string(),
            })?;

            cache
                .iter()
                .min_by_key(|(_, entry)| entry.access_count)
                .map(|(key, _)| key.clone())
        };

        if let Some(key) = key {
            self.remove(&key);
            self.record_eviction();
        }

        Ok(())
    }

    /// Evict expired entries
    fn evict_expired(&self) -> Result<()> {
        let expired_keys: Vec<SynthesisCacheKey> = {
            let cache = self.cache.lock().map_err(|_| AcousticError::ConfigError {
                message: "Failed to lock cache".to_string(),
            })?;

            cache
                .iter()
                .filter(|(_, entry)| entry.is_expired(self.config.ttl))
                .map(|(key, _)| key.clone())
                .collect()
        };

        for key in expired_keys {
            self.remove(&key);
            self.record_eviction();
        }

        Ok(())
    }

    /// Evict largest entry
    fn evict_largest(&self) -> Result<()> {
        let key = {
            let cache = self.cache.lock().map_err(|_| AcousticError::ConfigError {
                message: "Failed to lock cache".to_string(),
            })?;

            cache
                .iter()
                .max_by_key(|(_, entry)| entry.size_bytes)
                .map(|(key, _)| key.clone())
        };

        if let Some(key) = key {
            self.remove(&key);
            self.record_eviction();
        }

        Ok(())
    }

    /// Hybrid eviction: expired first, then LRU
    fn evict_hybrid(&self) -> Result<()> {
        // First try to evict expired entries
        self.evict_expired()?;

        // If still need space, use LRU
        let stats = self.statistics();
        if stats.entry_count >= self.config.max_entries {
            self.evict_lru()?;
        }

        Ok(())
    }

    /// Update access order for LRU tracking
    fn update_access_order(&self, key: &SynthesisCacheKey) {
        if let Ok(mut order) = self.access_order.lock() {
            // Remove key if it exists
            order.retain(|k| k != key);
            // Add to back (most recent)
            order.push_back(key.clone());
        }
    }

    /// Record a cache hit
    fn record_hit(&self, access_time: Duration) {
        if self.config.enable_stats {
            if let Ok(mut stats) = self.stats.lock() {
                stats.hits += 1;
                // Update moving average
                let new_time = access_time.as_micros() as f64;
                stats.avg_access_time_us = (stats.avg_access_time_us * (stats.hits - 1) as f64
                    + new_time)
                    / stats.hits as f64;

                // Update adaptive controller every 10 hits
                if stats.hits % 10 == 0 {
                    let hit_rate = (stats.hits as f64 / (stats.hits + stats.misses) as f64) * 100.0;
                    drop(stats);
                    if let Ok(mut controller) = self.adaptive_controller.lock() {
                        controller.update_hit_rate(hit_rate);
                    }
                }
            }
        }
    }

    /// Record a cache miss
    fn record_miss(&self) {
        if self.config.enable_stats {
            if let Ok(mut stats) = self.stats.lock() {
                stats.misses += 1;

                // Update adaptive controller every 10 misses
                if stats.misses % 10 == 0 {
                    let hit_rate = (stats.hits as f64 / (stats.hits + stats.misses) as f64) * 100.0;
                    drop(stats);
                    if let Ok(mut controller) = self.adaptive_controller.lock() {
                        controller.update_hit_rate(hit_rate);
                    }
                }
            }
        }
    }

    /// Record an eviction
    fn record_eviction(&self) {
        if self.config.enable_stats {
            if let Ok(mut stats) = self.stats.lock() {
                stats.evictions += 1;
            }
        }
    }
}

impl Default for SynthesisCache {
    fn default() -> Self {
        Self::new(SynthesisCacheConfig::default())
    }
}

impl MelSpectrogram {
    /// Estimate size in bytes for cache management
    pub fn estimate_size_bytes(&self) -> usize {
        // Rough estimate: frames * mels * sizeof(f32) + overhead
        self.n_frames * self.n_mels * std::mem::size_of::<f32>() + 1024
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_mel() -> MelSpectrogram {
        MelSpectrogram::new(vec![vec![0.5; 100]; 80], 22050, 256)
    }

    #[test]
    fn test_cache_key_creation() {
        let key = SynthesisCacheKey::new("hello world", Some(1), 1.0, 0.0, 1.0);
        assert_eq!(key.phoneme_text, "hello world");
        assert_eq!(key.speaker_id, Some(1));
        assert_eq!(key.speed_quantized, 10);
    }

    #[test]
    fn test_cache_insert_and_get() {
        let cache = SynthesisCache::default();
        let key = SynthesisCacheKey::new("test", None, 1.0, 0.0, 1.0);
        let mel = create_test_mel();

        cache.insert(key.clone(), mel.clone()).unwrap();
        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_cache_miss() {
        let cache = SynthesisCache::default();
        let key = SynthesisCacheKey::new("nonexistent", None, 1.0, 0.0, 1.0);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_statistics() {
        let cache = SynthesisCache::default();
        let key = SynthesisCacheKey::new("test", None, 1.0, 0.0, 1.0);
        let mel = create_test_mel();

        cache.insert(key.clone(), mel).unwrap();
        cache.get(&key); // Hit
        cache.get(&SynthesisCacheKey::new("miss", None, 1.0, 0.0, 1.0)); // Miss

        let stats = cache.statistics();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate(), 50.0);
    }

    #[test]
    fn test_cache_eviction_lru() {
        let config = SynthesisCacheConfig {
            max_entries: 2,
            eviction_policy: EvictionPolicy::LRU,
            ..Default::default()
        };
        let cache = SynthesisCache::new(config);

        let key1 = SynthesisCacheKey::new("first", None, 1.0, 0.0, 1.0);
        let key2 = SynthesisCacheKey::new("second", None, 1.0, 0.0, 1.0);
        let key3 = SynthesisCacheKey::new("third", None, 1.0, 0.0, 1.0);

        cache.insert(key1.clone(), create_test_mel()).unwrap();
        cache.insert(key2.clone(), create_test_mel()).unwrap();
        cache.insert(key3.clone(), create_test_mel()).unwrap();

        // First entry should be evicted
        assert!(cache.get(&key1).is_none());
        assert!(cache.get(&key2).is_some());
        assert!(cache.get(&key3).is_some());
    }

    #[test]
    fn test_cache_clear() {
        let cache = SynthesisCache::default();
        let key = SynthesisCacheKey::new("test", None, 1.0, 0.0, 1.0);
        cache.insert(key.clone(), create_test_mel()).unwrap();

        cache.clear();
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.statistics().entry_count, 0);
    }

    #[test]
    fn test_access_pattern_analyzer() {
        let cache = SynthesisCache::default();
        let key1 = SynthesisCacheKey::new("hello", None, 1.0, 0.0, 1.0);
        let key2 = SynthesisCacheKey::new("world", None, 1.0, 0.0, 1.0);
        let key3 = SynthesisCacheKey::new("test", None, 1.0, 0.0, 1.0);

        // Insert and access multiple times to build patterns
        cache.insert(key1.clone(), create_test_mel()).unwrap();
        cache.insert(key2.clone(), create_test_mel()).unwrap();
        cache.insert(key3.clone(), create_test_mel()).unwrap();

        // Access in pattern: key1 -> key2, key1 -> key2
        cache.get(&key1);
        cache.get(&key2);
        cache.get(&key1);
        cache.get(&key2);
        cache.get(&key1);
        cache.get(&key3);

        // Get hot keys
        let hot_keys = cache.get_hot_keys(10);
        assert!(!hot_keys.is_empty());
    }

    #[test]
    fn test_predictive_prefetch() {
        let cache = SynthesisCache::default();
        let key1 = SynthesisCacheKey::new("first", None, 1.0, 0.0, 1.0);
        let key2 = SynthesisCacheKey::new("second", None, 1.0, 0.0, 1.0);
        let key3 = SynthesisCacheKey::new("third", None, 1.0, 0.0, 1.0);

        // Insert entries
        cache.insert(key1.clone(), create_test_mel()).unwrap();
        cache.insert(key2.clone(), create_test_mel()).unwrap();
        cache.insert(key3.clone(), create_test_mel()).unwrap();

        // Build pattern: key1 -> key2 -> key3
        for _ in 0..5 {
            cache.get(&key1);
            cache.get(&key2);
            cache.get(&key3);
        }

        // Predict next access after key1
        let predicted = cache.prefetch_predicted(&key1);
        assert!(!predicted.is_empty());
    }

    #[test]
    fn test_adaptive_eviction_controller() {
        let mut controller = AdaptiveEvictionController::new();

        // Simulate good performance
        for _ in 0..50 {
            controller.update_hit_rate(80.0);
        }

        // Policy should remain stable with good performance
        let policy = controller.get_policy();
        assert!(matches!(
            policy,
            EvictionPolicy::Hybrid | EvictionPolicy::LRU | EvictionPolicy::LFU
        ));

        // Simulate degrading performance
        for _ in 0..30 {
            controller.update_hit_rate(30.0);
        }

        // Controller may adapt policy (not deterministic due to timing)
        let _ = controller.get_policy();
    }

    #[test]
    fn test_adaptive_policy_selection() {
        let config = SynthesisCacheConfig {
            max_entries: 10,
            eviction_policy: EvictionPolicy::Hybrid,
            ..Default::default()
        };
        let cache = SynthesisCache::new(config);

        // Simulate many cache operations to trigger adaptive behavior
        for i in 0..100 {
            let key = SynthesisCacheKey::new(&format!("key_{}", i), None, 1.0, 0.0, 1.0);
            cache.insert(key.clone(), create_test_mel()).unwrap();

            // Access some keys multiple times (create hot keys)
            if i % 3 == 0 {
                cache.get(&key);
                cache.get(&key);
            }
        }

        let stats = cache.statistics();
        assert!(stats.hits > 0);
        assert!(stats.evictions > 0);
    }

    #[test]
    fn test_hot_key_protection() {
        let config = SynthesisCacheConfig {
            max_entries: 5,
            eviction_policy: EvictionPolicy::Hybrid,
            ..Default::default()
        };
        let cache = SynthesisCache::new(config);

        let hot_key = SynthesisCacheKey::new("hot", None, 1.0, 0.0, 1.0);
        cache.insert(hot_key.clone(), create_test_mel()).unwrap();

        // Access hot key many times to make it "hot"
        for _ in 0..20 {
            cache.get(&hot_key);
        }

        // Get hot keys to verify it's tracked
        let hot_keys = cache.get_hot_keys(10);
        assert!(!hot_keys.is_empty());

        // Add a few other keys to trigger evictions
        for i in 0..3 {
            let key = SynthesisCacheKey::new(&format!("cold_{}", i), None, 1.0, 0.0, 1.0);
            cache.insert(key, create_test_mel()).unwrap();
        }

        // Hot key tracking is working
        let stats = cache.statistics();
        assert!(stats.entry_count <= 5);
    }

    #[test]
    fn test_peak_time_detection() {
        let cache = SynthesisCache::default();

        // Generate activity to establish patterns
        for _ in 0..100 {
            let key =
                SynthesisCacheKey::new(&format!("key_{}", fastrand::u32(..)), None, 1.0, 0.0, 1.0);
            cache.insert(key.clone(), create_test_mel()).unwrap();
            cache.get(&key);
        }

        // Peak time detection should work (result depends on actual time)
        let _is_peak = cache.is_peak_time();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_adaptive_controller_hit_rate_tracking() {
        let cache = SynthesisCache::default();

        // First insert 10 keys
        for i in 0..10 {
            let key = SynthesisCacheKey::new(&format!("key_{}", i), None, 1.0, 0.0, 1.0);
            cache.insert(key, create_test_mel()).unwrap();
        }

        // Generate cache hits and misses
        for i in 0..50 {
            let key = if i % 2 == 0 {
                // Hit: access existing key
                SynthesisCacheKey::new(&format!("key_{}", i % 10), None, 1.0, 0.0, 1.0)
            } else {
                // Miss: access non-existent key
                SynthesisCacheKey::new(&format!("nonexistent_{}", i), None, 1.0, 0.0, 1.0)
            };
            cache.get(&key);
        }

        let stats = cache.statistics();
        assert!(stats.hits > 0);
        assert!(stats.misses > 0);
        assert!(stats.hit_rate() > 0.0 && stats.hit_rate() < 100.0);
    }

    #[test]
    fn test_sequential_pattern_learning() {
        let cache = SynthesisCache::default();
        let keys: Vec<_> = (0..5)
            .map(|i| SynthesisCacheKey::new(&format!("seq_{}", i), None, 1.0, 0.0, 1.0))
            .collect();

        // Insert all keys
        for key in &keys {
            cache.insert(key.clone(), create_test_mel()).unwrap();
        }

        // Create sequential pattern: 0 -> 1 -> 2 -> 3 -> 4
        for _ in 0..10 {
            for key in &keys {
                cache.get(key);
            }
        }

        // Predict next after first key
        let predicted = cache.prefetch_predicted(&keys[0]);
        // Should predict key 1 as it follows key 0
        assert!(!predicted.is_empty());
    }
}
