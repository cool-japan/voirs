//! Feature-specific caching optimizations for vocoder operations
//!
//! Provides specialized caching strategies for different vocoder features including:
//! - Mel spectrogram caching
//! - Model weight caching
//! - Audio buffer caching
//! - Result caching with quality-aware eviction

use super::{CacheConfig, CacheOptimizer};
use crate::{AudioBuffer, MelSpectrogram, Result};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Cache entry metadata
#[derive(Debug, Clone)]
pub struct CacheEntryMetadata {
    /// When the entry was created
    pub created_at: Instant,
    /// When the entry was last accessed
    pub last_accessed: Instant,
    /// How many times the entry has been accessed
    pub access_count: u64,
    /// Quality score of the cached result (0.0-1.0)
    pub quality_score: f32,
    /// Size of the cached data in bytes
    pub size_bytes: usize,
    /// Processing time saved by caching (estimated)
    pub time_saved_ms: f32,
}

impl CacheEntryMetadata {
    /// Create new metadata
    pub fn new(quality_score: f32, size_bytes: usize, time_saved_ms: f32) -> Self {
        let now = Instant::now();
        Self {
            created_at: now,
            last_accessed: now,
            access_count: 0,
            quality_score,
            size_bytes,
            time_saved_ms,
        }
    }

    /// Update access information
    pub fn record_access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    /// Calculate cache value score for eviction decisions
    pub fn cache_value_score(&self) -> f32 {
        let age_penalty = self.created_at.elapsed().as_secs_f32() / 3600.0; // Hours
        let access_bonus = (self.access_count as f32).ln_1p();
        let quality_bonus = self.quality_score;
        let size_penalty = (self.size_bytes as f32 / 1024.0).ln_1p(); // KB

        (quality_bonus + access_bonus * 0.5 - age_penalty * 0.2 - size_penalty * 0.1).max(0.0)
    }
}

/// Mel spectrogram cache key
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MelCacheKey {
    /// Checksum of mel data
    pub data_hash: u64,
    /// Number of mel channels
    pub n_mels: usize,
    /// Number of time frames
    pub n_frames: usize,
    /// Sample rate
    pub sample_rate: u32,
    /// Hop length
    pub hop_length: u32,
}

impl Hash for MelCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data_hash.hash(state);
        self.n_mels.hash(state);
        self.n_frames.hash(state);
        self.sample_rate.hash(state);
        self.hop_length.hash(state);
    }
}

impl MelCacheKey {
    /// Create cache key from mel spectrogram
    pub fn from_mel(mel: &MelSpectrogram) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash mel data
        for row in &mel.data {
            for &value in row {
                value.to_bits().hash(&mut hasher);
            }
        }

        let data_hash = hasher.finish();

        Self {
            data_hash,
            n_mels: mel.n_mels,
            n_frames: mel.n_frames,
            sample_rate: mel.sample_rate,
            hop_length: mel.hop_length,
        }
    }
}

/// Audio buffer cache key
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioCacheKey {
    /// Checksum of audio samples
    pub data_hash: u64,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
    /// Number of samples
    pub num_samples: usize,
}

impl Hash for AudioCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data_hash.hash(state);
        self.sample_rate.hash(state);
        self.channels.hash(state);
        self.num_samples.hash(state);
    }
}

impl AudioCacheKey {
    /// Create cache key from audio buffer
    pub fn from_audio(audio: &AudioBuffer) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash audio samples (sample every Nth sample for performance)
        let step = (audio.samples().len() / 1000).max(1);
        for (i, &sample) in audio.samples().iter().enumerate() {
            if i % step == 0 {
                sample.to_bits().hash(&mut hasher);
            }
        }

        let data_hash = hasher.finish();

        Self {
            data_hash,
            sample_rate: audio.sample_rate(),
            channels: audio.channels(),
            num_samples: audio.samples().len(),
        }
    }
}

/// Generic cache entry
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    /// Cached data
    pub data: T,
    /// Entry metadata
    pub metadata: CacheEntryMetadata,
}

impl<T> CacheEntry<T> {
    /// Create new cache entry
    pub fn new(data: T, quality_score: f32, size_bytes: usize, time_saved_ms: f32) -> Self {
        Self {
            data,
            metadata: CacheEntryMetadata::new(quality_score, size_bytes, time_saved_ms),
        }
    }

    /// Access the cached data and update metadata
    pub fn access(&mut self) -> &T {
        self.metadata.record_access();
        &self.data
    }
}

/// Feature-specific cache configuration
#[derive(Debug, Clone)]
pub struct FeatureCacheConfig {
    /// Maximum cache size in bytes
    pub max_size_bytes: usize,
    /// Maximum number of entries
    pub max_entries: usize,
    /// Entry TTL (time to live)
    pub entry_ttl: Duration,
    /// Minimum quality score for caching
    pub min_quality_score: f32,
    /// Enable compression for cached data
    pub enable_compression: bool,
    /// Prefetch related entries
    pub enable_prefetch: bool,
    /// Cache hit rate target (for adaptive sizing)
    pub target_hit_rate: f32,
}

impl Default for FeatureCacheConfig {
    fn default() -> Self {
        Self {
            max_size_bytes: 512 * 1024 * 1024, // 512MB
            max_entries: 10000,
            entry_ttl: Duration::from_secs(3600), // 1 hour
            min_quality_score: 0.7,
            enable_compression: true,
            enable_prefetch: true,
            target_hit_rate: 0.8,
        }
    }
}

/// Audio processing result cache
pub struct AudioResultCache {
    /// Cache storage
    cache: Arc<RwLock<HashMap<AudioCacheKey, CacheEntry<AudioBuffer>>>>,
    /// Cache configuration
    config: FeatureCacheConfig,
    /// Cache optimizer
    #[allow(dead_code)] // Reserved for future cache optimization algorithms
    optimizer: CacheOptimizer,
    /// Current cache size in bytes
    current_size: Arc<RwLock<usize>>,
    /// Cache statistics
    hits: Arc<RwLock<u64>>,
    misses: Arc<RwLock<u64>>,
}

impl AudioResultCache {
    /// Create new audio result cache
    pub fn new(config: FeatureCacheConfig) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            optimizer: CacheOptimizer::new(CacheConfig::default()),
            config,
            current_size: Arc::new(RwLock::new(0)),
            hits: Arc::new(RwLock::new(0)),
            misses: Arc::new(RwLock::new(0)),
        }
    }

    /// Get cached audio result
    pub fn get(&self, key: &AudioCacheKey) -> Option<AudioBuffer> {
        let mut cache = self.cache.write().unwrap();

        if let Some(entry) = cache.get_mut(key) {
            // Check if entry is still valid
            if entry.metadata.created_at.elapsed() <= self.config.entry_ttl {
                *self.hits.write().unwrap() += 1;
                Some(entry.access().clone())
            } else {
                // Entry expired, remove it
                let size = entry.metadata.size_bytes;
                cache.remove(key);
                *self.current_size.write().unwrap() -= size;
                *self.misses.write().unwrap() += 1;
                None
            }
        } else {
            *self.misses.write().unwrap() += 1;
            None
        }
    }

    /// Store audio result in cache
    pub fn put(
        &self,
        key: AudioCacheKey,
        audio: AudioBuffer,
        quality_score: f32,
        processing_time_ms: f32,
    ) -> Result<()> {
        if quality_score < self.config.min_quality_score {
            return Ok(()); // Don't cache low quality results
        }

        let audio_size = std::mem::size_of_val(audio.samples());
        let entry = CacheEntry::new(audio, quality_score, audio_size, processing_time_ms);

        let mut cache = self.cache.write().unwrap();
        let mut current_size = self.current_size.write().unwrap();

        // Evict entries if necessary
        while (*current_size + audio_size > self.config.max_size_bytes
            || cache.len() >= self.config.max_entries)
            && !cache.is_empty()
        {
            self.evict_least_valuable(&mut cache, &mut current_size);
        }

        // Add new entry
        if let Some(old_entry) = cache.insert(key, entry) {
            *current_size -= old_entry.metadata.size_bytes;
        }
        *current_size += audio_size;

        Ok(())
    }

    /// Evict the least valuable cache entry
    fn evict_least_valuable(
        &self,
        cache: &mut HashMap<AudioCacheKey, CacheEntry<AudioBuffer>>,
        current_size: &mut usize,
    ) {
        let mut worst_key = None;
        let mut worst_score = f32::INFINITY;

        for (key, entry) in cache.iter() {
            let score = entry.metadata.cache_value_score();
            if score < worst_score {
                worst_score = score;
                worst_key = Some(key.clone());
            }
        }

        if let Some(key) = worst_key {
            if let Some(entry) = cache.remove(&key) {
                *current_size -= entry.metadata.size_bytes;
            }
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let hits = *self.hits.read().unwrap();
        let misses = *self.misses.read().unwrap();
        let hit_rate = if hits + misses > 0 {
            hits as f32 / (hits + misses) as f32
        } else {
            0.0
        };

        CacheStats {
            hits,
            misses,
            hit_rate,
            entries: self.cache.read().unwrap().len(),
            size_bytes: *self.current_size.read().unwrap(),
            max_size_bytes: self.config.max_size_bytes,
        }
    }

    /// Clear cache
    pub fn clear(&self) {
        self.cache.write().unwrap().clear();
        *self.current_size.write().unwrap() = 0;
    }
}

/// Mel spectrogram cache for preprocessing results
pub struct MelCache {
    /// Cache storage
    cache: Arc<RwLock<HashMap<MelCacheKey, CacheEntry<MelSpectrogram>>>>,
    /// Cache configuration
    config: FeatureCacheConfig,
    /// Current cache size
    current_size: Arc<RwLock<usize>>,
    /// Statistics
    hits: Arc<RwLock<u64>>,
    misses: Arc<RwLock<u64>>,
}

impl MelCache {
    /// Create new mel spectrogram cache
    pub fn new(config: FeatureCacheConfig) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            current_size: Arc::new(RwLock::new(0)),
            hits: Arc::new(RwLock::new(0)),
            misses: Arc::new(RwLock::new(0)),
        }
    }

    /// Get cached mel spectrogram
    pub fn get(&self, key: &MelCacheKey) -> Option<MelSpectrogram> {
        let mut cache = self.cache.write().unwrap();

        if let Some(entry) = cache.get_mut(key) {
            if entry.metadata.created_at.elapsed() <= self.config.entry_ttl {
                *self.hits.write().unwrap() += 1;
                Some(entry.access().clone())
            } else {
                let size = entry.metadata.size_bytes;
                cache.remove(key);
                *self.current_size.write().unwrap() -= size;
                *self.misses.write().unwrap() += 1;
                None
            }
        } else {
            *self.misses.write().unwrap() += 1;
            None
        }
    }

    /// Store mel spectrogram in cache
    pub fn put(
        &self,
        key: MelCacheKey,
        mel: MelSpectrogram,
        processing_time_ms: f32,
    ) -> Result<()> {
        let mel_size = mel.n_mels * mel.n_frames * std::mem::size_of::<f32>();
        let entry = CacheEntry::new(mel, 1.0, mel_size, processing_time_ms); // Assume mel quality is always good

        let mut cache = self.cache.write().unwrap();
        let mut current_size = self.current_size.write().unwrap();

        // Evict if necessary
        while (*current_size + mel_size > self.config.max_size_bytes
            || cache.len() >= self.config.max_entries)
            && !cache.is_empty()
        {
            self.evict_least_valuable(&mut cache, &mut current_size);
        }

        // Add new entry
        if let Some(old_entry) = cache.insert(key, entry) {
            *current_size -= old_entry.metadata.size_bytes;
        }
        *current_size += mel_size;

        Ok(())
    }

    /// Evict least valuable entry
    fn evict_least_valuable(
        &self,
        cache: &mut HashMap<MelCacheKey, CacheEntry<MelSpectrogram>>,
        current_size: &mut usize,
    ) {
        let mut worst_key = None;
        let mut worst_score = f32::INFINITY;

        for (key, entry) in cache.iter() {
            let score = entry.metadata.cache_value_score();
            if score < worst_score {
                worst_score = score;
                worst_key = Some(key.clone());
            }
        }

        if let Some(key) = worst_key {
            if let Some(entry) = cache.remove(&key) {
                *current_size -= entry.metadata.size_bytes;
            }
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let hits = *self.hits.read().unwrap();
        let misses = *self.misses.read().unwrap();
        let hit_rate = if hits + misses > 0 {
            hits as f32 / (hits + misses) as f32
        } else {
            0.0
        };

        CacheStats {
            hits,
            misses,
            hit_rate,
            entries: self.cache.read().unwrap().len(),
            size_bytes: *self.current_size.read().unwrap(),
            max_size_bytes: self.config.max_size_bytes,
        }
    }

    /// Clear cache
    pub fn clear(&self) {
        self.cache.write().unwrap().clear();
        *self.current_size.write().unwrap() = 0;
        *self.hits.write().unwrap() = 0;
        *self.misses.write().unwrap() = 0;
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Hit rate (0.0-1.0)
    pub hit_rate: f32,
    /// Current number of entries
    pub entries: usize,
    /// Current cache size in bytes
    pub size_bytes: usize,
    /// Maximum cache size in bytes
    pub max_size_bytes: usize,
}

impl CacheStats {
    /// Check if cache performance is good
    pub fn is_performing_well(&self, target_hit_rate: f32) -> bool {
        self.hit_rate >= target_hit_rate
    }

    /// Get cache utilization (0.0-1.0)
    pub fn utilization(&self) -> f32 {
        if self.max_size_bytes > 0 {
            self.size_bytes as f32 / self.max_size_bytes as f32
        } else {
            0.0
        }
    }
}

/// Feature-aware cache manager
pub struct FeatureCacheManager {
    /// Audio result cache
    audio_cache: AudioResultCache,
    /// Mel spectrogram cache
    mel_cache: MelCache,
    /// Cache configuration
    config: FeatureCacheConfig,
}

impl FeatureCacheManager {
    /// Create new feature cache manager
    pub fn new(config: FeatureCacheConfig) -> Self {
        let audio_config = FeatureCacheConfig {
            max_size_bytes: config.max_size_bytes / 2, // Split between audio and mel caches
            ..config.clone()
        };

        let mel_config = FeatureCacheConfig {
            max_size_bytes: config.max_size_bytes / 2,
            ..config.clone()
        };

        Self {
            audio_cache: AudioResultCache::new(audio_config),
            mel_cache: MelCache::new(mel_config),
            config,
        }
    }

    /// Get cached audio result
    pub fn get_audio(&self, input_audio: &AudioBuffer) -> Option<AudioBuffer> {
        let key = AudioCacheKey::from_audio(input_audio);
        self.audio_cache.get(&key)
    }

    /// Cache audio processing result
    pub fn cache_audio(
        &self,
        input: &AudioBuffer,
        output: AudioBuffer,
        quality_score: f32,
        processing_time_ms: f32,
    ) -> Result<()> {
        let key = AudioCacheKey::from_audio(input);
        self.audio_cache
            .put(key, output, quality_score, processing_time_ms)
    }

    /// Get cached mel spectrogram
    pub fn get_mel(&self, mel: &MelSpectrogram) -> Option<MelSpectrogram> {
        let key = MelCacheKey::from_mel(mel);
        self.mel_cache.get(&key)
    }

    /// Cache mel spectrogram
    pub fn cache_mel(&self, mel: MelSpectrogram, processing_time_ms: f32) -> Result<()> {
        let key = MelCacheKey::from_mel(&mel);
        self.mel_cache.put(key, mel, processing_time_ms)
    }

    /// Get comprehensive cache statistics
    pub fn get_comprehensive_stats(&self) -> ComprehensiveCacheStats {
        let audio_stats = self.audio_cache.get_stats();
        let mel_stats = self.mel_cache.get_stats();

        ComprehensiveCacheStats {
            audio_cache: audio_stats,
            mel_cache: mel_stats,
            total_size_bytes: self.config.max_size_bytes,
        }
    }

    /// Optimize cache based on usage patterns
    pub fn optimize(&self) {
        let audio_stats = self.audio_cache.get_stats();
        let mel_stats = self.mel_cache.get_stats();

        // If one cache is underperforming, we could dynamically adjust sizes
        // This is a simplified optimization - in a real system, this would be more sophisticated
        if audio_stats.hit_rate < self.config.target_hit_rate - 0.1 {
            // Consider increasing audio cache size or adjusting eviction policy
            tracing::warn!(
                "Audio cache performance below target: {:.2} < {:.2}",
                audio_stats.hit_rate,
                self.config.target_hit_rate
            );
        }

        if mel_stats.hit_rate < self.config.target_hit_rate - 0.1 {
            tracing::warn!(
                "Mel cache performance below target: {:.2} < {:.2}",
                mel_stats.hit_rate,
                self.config.target_hit_rate
            );
        }
    }

    /// Clear all caches
    pub fn clear_all(&self) {
        self.audio_cache.clear();
        self.mel_cache.clear();
    }
}

/// Comprehensive cache statistics
#[derive(Debug, Clone)]
pub struct ComprehensiveCacheStats {
    /// Audio cache statistics
    pub audio_cache: CacheStats,
    /// Mel cache statistics
    pub mel_cache: CacheStats,
    /// Total allocated cache size
    pub total_size_bytes: usize,
}

impl ComprehensiveCacheStats {
    /// Get overall cache hit rate
    pub fn overall_hit_rate(&self) -> f32 {
        let total_hits = self.audio_cache.hits + self.mel_cache.hits;
        let total_requests = total_hits + self.audio_cache.misses + self.mel_cache.misses;

        if total_requests > 0 {
            total_hits as f32 / total_requests as f32
        } else {
            0.0
        }
    }

    /// Get total cache utilization
    pub fn total_utilization(&self) -> f32 {
        let used_bytes = self.audio_cache.size_bytes + self.mel_cache.size_bytes;
        if self.total_size_bytes > 0 {
            used_bytes as f32 / self.total_size_bytes as f32
        } else {
            0.0
        }
    }

    /// Get estimated time saved by caching (rough estimate)
    pub fn estimated_time_saved_ms(&self) -> f32 {
        // This is a rough estimate - in practice, we'd track actual time savings
        let audio_time_saved = self.audio_cache.hits as f32 * 50.0; // Assume 50ms per hit
        let mel_time_saved = self.mel_cache.hits as f32 * 20.0; // Assume 20ms per hit
        audio_time_saved + mel_time_saved
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioBuffer;

    #[test]
    fn test_audio_cache_key_creation() {
        let audio = AudioBuffer::new(vec![1.0, 2.0, 3.0, 4.0], 22050, 1);
        let key1 = AudioCacheKey::from_audio(&audio);
        let key2 = AudioCacheKey::from_audio(&audio);

        assert_eq!(key1, key2);
        assert_eq!(key1.sample_rate, 22050);
        assert_eq!(key1.channels, 1);
        assert_eq!(key1.num_samples, 4);
    }

    #[test]
    fn test_mel_cache_key_creation() {
        let mel_data = vec![vec![0.5; 100]; 80];
        let mel = MelSpectrogram::new(mel_data, 22050, 256);
        let key1 = MelCacheKey::from_mel(&mel);
        let key2 = MelCacheKey::from_mel(&mel);

        assert_eq!(key1, key2);
        assert_eq!(key1.n_mels, 80);
        assert_eq!(key1.n_frames, 100);
        assert_eq!(key1.sample_rate, 22050);
    }

    #[test]
    fn test_audio_result_cache() {
        let config = FeatureCacheConfig::default();
        let cache = AudioResultCache::new(config);

        let audio = AudioBuffer::new(vec![1.0, 2.0, 3.0], 22050, 1);
        let key = AudioCacheKey::from_audio(&audio);

        // Cache miss initially
        assert!(cache.get(&key).is_none());

        // Store audio
        cache.put(key.clone(), audio.clone(), 0.8, 10.0).unwrap();

        // Cache hit
        let cached = cache.get(&key).unwrap();
        assert_eq!(cached.samples(), audio.samples());

        let stats = cache.get_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate, 0.5);
    }

    #[test]
    fn test_feature_cache_manager() {
        let config = FeatureCacheConfig::default();
        let manager = FeatureCacheManager::new(config);

        let audio = AudioBuffer::new(vec![1.0, 2.0, 3.0], 22050, 1);
        let output = AudioBuffer::new(vec![2.0, 4.0, 6.0], 22050, 1);

        // Cache miss initially
        assert!(manager.get_audio(&audio).is_none());

        // Cache result
        manager
            .cache_audio(&audio, output.clone(), 0.9, 15.0)
            .unwrap();

        // Cache hit
        let cached = manager.get_audio(&audio).unwrap();
        assert_eq!(cached.samples(), output.samples());

        let stats = manager.get_comprehensive_stats();
        assert!(stats.overall_hit_rate() > 0.0);
    }

    #[test]
    fn test_cache_entry_metadata() {
        let mut metadata = CacheEntryMetadata::new(0.8, 1024, 50.0);

        assert_eq!(metadata.access_count, 0);
        assert_eq!(metadata.quality_score, 0.8);
        assert_eq!(metadata.size_bytes, 1024);

        metadata.record_access();
        assert_eq!(metadata.access_count, 1);

        let score = metadata.cache_value_score();
        assert!(score > 0.0);
    }
}
