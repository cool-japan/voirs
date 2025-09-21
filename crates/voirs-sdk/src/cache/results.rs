//! Result caching system for synthesis results and intermediate outputs.

use crate::{
    audio::AudioBuffer,
    error::{Result, VoirsError},
    traits::CacheStats,
    types::{MelSpectrogram, Phoneme},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    hash::{Hash, Hasher},
    path::PathBuf,
    sync::{Arc, RwLock},
    time::{Duration, SystemTime},
};
use tokio::{fs, io::AsyncWriteExt};
use tracing::{debug, info};

/// Advanced synthesis result cache with intelligent storage and retrieval
pub struct SynthesisResultCache {
    /// In-memory cache storage
    memory_cache: Arc<RwLock<HashMap<String, CachedSynthesisResult>>>,

    /// LRU tracking queue
    lru_queue: Arc<RwLock<VecDeque<String>>>,

    /// Disk cache directory
    disk_cache_dir: Option<PathBuf>,

    /// Cache configuration
    config: ResultCacheConfig,

    /// Current memory usage
    current_memory_usage: Arc<RwLock<usize>>,

    /// Cache statistics
    stats: Arc<RwLock<ResultCacheStats>>,

    /// Access frequency tracking
    access_frequency: Arc<RwLock<HashMap<String, AccessInfo>>>,

    /// Text similarity index for related results
    similarity_index: Arc<RwLock<HashMap<String, Vec<String>>>>,

    /// Quality metrics cache
    quality_cache: Arc<RwLock<HashMap<String, QualityMetrics>>>,
}

/// Configuration for result caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultCacheConfig {
    /// Memory cache size in MB
    pub memory_cache_size_mb: usize,

    /// Disk cache size in MB
    pub disk_cache_size_mb: usize,

    /// Default TTL for results
    pub default_ttl_seconds: u64,

    /// TTL for high-quality results
    pub high_quality_ttl_seconds: u64,

    /// TTL for low-quality results
    pub low_quality_ttl_seconds: u64,

    /// Enable disk caching
    pub enable_disk_cache: bool,

    /// Enable compression for cached results
    pub enable_compression: bool,

    /// Enable similarity-based caching
    pub enable_similarity_cache: bool,

    /// Text similarity threshold (0.0-1.0)
    pub similarity_threshold: f64,

    /// Maximum similar results to track
    pub max_similar_results: usize,

    /// Enable background cleanup
    pub enable_background_cleanup: bool,

    /// Cleanup interval in seconds
    pub cleanup_interval_seconds: u64,

    /// Quality threshold for retention
    pub quality_retention_threshold: f64,

    /// Enable result validation
    pub enable_validation: bool,

    /// Cache partitioning by language
    pub partition_by_language: bool,

    /// Maximum text length for caching
    pub max_text_length: usize,
}

impl Default for ResultCacheConfig {
    fn default() -> Self {
        Self {
            memory_cache_size_mb: 512,
            disk_cache_size_mb: 2048,
            default_ttl_seconds: 86400,       // 24 hours
            high_quality_ttl_seconds: 604800, // 7 days
            low_quality_ttl_seconds: 3600,    // 1 hour
            enable_disk_cache: true,
            enable_compression: true,
            enable_similarity_cache: true,
            similarity_threshold: 0.85,
            max_similar_results: 10,
            enable_background_cleanup: true,
            cleanup_interval_seconds: 3600, // 1 hour
            quality_retention_threshold: 0.7,
            enable_validation: true,
            partition_by_language: true,
            max_text_length: 10000,
        }
    }
}

/// Cached synthesis result with metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CachedSynthesisResult {
    /// Original text
    pub text: String,

    /// Text language
    pub language: String,

    /// Configuration hash used for synthesis
    pub config_hash: u64,

    /// Generated phonemes
    pub phonemes: Vec<Phoneme>,

    /// Generated mel spectrogram
    pub mel_spectrogram: MelSpectrogram,

    /// Generated audio buffer
    pub audio_buffer: AudioBuffer,

    /// Synthesis metadata
    pub metadata: SynthesisMetadata,

    /// Cache metadata
    pub cache_metadata: CacheMetadata,

    /// Quality metrics
    pub quality_metrics: QualityMetrics,

    /// Size in bytes (estimated)
    pub size_bytes: usize,
}

/// Synthesis metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisMetadata {
    /// Voice used for synthesis
    pub voice_id: String,

    /// Model versions used
    pub model_versions: HashMap<String, String>,

    /// Synthesis duration in milliseconds
    pub synthesis_duration_ms: u64,

    /// Audio duration in seconds
    pub audio_duration_seconds: f64,

    /// Sample rate
    pub sample_rate: u32,

    /// Bit depth
    pub bit_depth: u16,

    /// Number of channels
    pub channels: u16,

    /// Synthesis parameters
    pub parameters: HashMap<String, f64>,

    /// Processing stages timing
    pub stage_timings: HashMap<String, u64>,
}

/// Cache metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    /// When the result was cached
    pub cached_at: SystemTime,

    /// When the result expires
    pub expires_at: SystemTime,

    /// Last access time
    pub last_accessed: SystemTime,

    /// Access count
    pub access_count: u64,

    /// Cache hit count
    pub hit_count: u64,

    /// Result priority
    pub priority: ResultPriority,

    /// Whether result is pinned
    pub pinned: bool,

    /// Cache source
    pub source: CacheSource,

    /// Checksum for integrity
    pub checksum: String,
}

/// Result priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResultPriority {
    /// Critical results (never evict)
    Critical,
    /// High priority
    High,
    /// Normal priority
    Normal,
    /// Low priority (evict first)
    Low,
}

impl Default for ResultPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Cache source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheSource {
    /// Generated fresh
    Fresh,
    /// Loaded from memory cache
    Memory,
    /// Loaded from disk cache
    Disk,
    /// Similarity match
    Similar(String),
}

/// Quality metrics for synthesis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Overall quality score (0.0-1.0)
    pub overall_score: f64,

    /// Audio quality metrics
    pub audio_quality: AudioQualityMetrics,

    /// Pronunciation accuracy
    pub pronunciation_accuracy: f64,

    /// Naturalness score
    pub naturalness_score: f64,

    /// Intelligibility score
    pub intelligibility_score: f64,

    /// Prosody quality
    pub prosody_quality: f64,

    /// Synthesis confidence
    pub confidence_score: f64,

    /// Error metrics
    pub error_metrics: ErrorMetrics,
}

/// Audio quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityMetrics {
    /// Signal-to-noise ratio
    pub snr_db: f64,

    /// Total harmonic distortion
    pub thd_percent: f64,

    /// Frequency response quality
    pub frequency_response_score: f64,

    /// Dynamic range
    pub dynamic_range_db: f64,

    /// Spectral quality
    pub spectral_quality_score: f64,
}

/// Error metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Pronunciation errors
    pub pronunciation_errors: u32,

    /// Timing errors
    pub timing_errors: u32,

    /// Spectral artifacts
    pub spectral_artifacts: u32,

    /// Discontinuities
    pub discontinuities: u32,
}

/// Parameters for caching synthesis results
#[derive(Debug)]
pub struct SynthesisCacheParams {
    pub text: String,
    pub language: String,
    pub config_hash: u64,
    pub phonemes: Vec<Phoneme>,
    pub mel_spectrogram: MelSpectrogram,
    pub audio_buffer: AudioBuffer,
    pub metadata: SynthesisMetadata,
    pub quality_metrics: QualityMetrics,
}

/// Access information for frequency tracking
#[derive(Debug, Clone)]
pub struct AccessInfo {
    /// Total access count
    pub access_count: u64,

    /// Last access time
    pub last_access: SystemTime,

    /// Access frequency (accesses per hour)
    pub frequency: f64,

    /// Access history
    pub access_history: VecDeque<SystemTime>,
}

/// Result cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResultCacheStats {
    /// Basic cache stats
    pub basic_stats: CacheStats,

    /// Results generated
    pub results_generated: u64,

    /// Results served from cache
    pub results_served_from_cache: u64,

    /// Cache misses
    pub cache_misses: u64,

    /// Similarity matches
    pub similarity_matches: u64,

    /// Average synthesis time (ms)
    pub avg_synthesis_time_ms: f64,

    /// Average quality score
    pub avg_quality_score: f64,

    /// Quality distribution
    pub quality_distribution: QualityDistribution,

    /// Language distribution
    pub language_distribution: HashMap<String, u64>,

    /// Text length distribution
    pub text_length_distribution: HashMap<String, u64>,

    /// Disk cache usage
    pub disk_usage_bytes: u64,

    /// Compression ratio
    pub compression_ratio: f64,

    /// Results expired
    pub results_expired: u64,

    /// Results evicted
    pub results_evicted: u64,
}

/// Quality score distribution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityDistribution {
    /// Excellent quality (0.9-1.0)
    pub excellent: u64,

    /// Good quality (0.7-0.9)
    pub good: u64,

    /// Fair quality (0.5-0.7)
    pub fair: u64,

    /// Poor quality (0.0-0.5)
    pub poor: u64,
}

impl SynthesisResultCache {
    /// Create new synthesis result cache
    pub fn new(config: ResultCacheConfig, disk_cache_dir: Option<PathBuf>) -> Result<Self> {
        // Create disk cache directory if specified
        if let Some(ref dir) = disk_cache_dir {
            if config.enable_disk_cache {
                std::fs::create_dir_all(dir).map_err(|e| {
                    VoirsError::cache_error(format!("Failed to create cache directory: {e}"))
                })?;
            }
        }

        Ok(Self {
            memory_cache: Arc::new(RwLock::new(HashMap::new())),
            lru_queue: Arc::new(RwLock::new(VecDeque::new())),
            disk_cache_dir,
            config,
            current_memory_usage: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(ResultCacheStats::default())),
            access_frequency: Arc::new(RwLock::new(HashMap::new())),
            similarity_index: Arc::new(RwLock::new(HashMap::new())),
            quality_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Get cached synthesis result
    pub async fn get_synthesis_result(
        &self,
        text: &str,
        language: &str,
        config_hash: u64,
    ) -> Option<CachedSynthesisResult> {
        let cache_key = self.make_cache_key(text, language, config_hash);

        // Try exact match first
        if let Some(result) = self.get_exact_match(&cache_key).await {
            self.update_access_info(&cache_key).await;
            self.update_hit_stats().await;
            return Some(result);
        }

        // Try similarity match if enabled
        if self.config.enable_similarity_cache {
            if let Some(result) = self.get_similarity_match(text, language, config_hash).await {
                self.update_hit_stats().await;
                return Some(result);
            }
        }

        // Update miss stats
        self.update_miss_stats().await;
        None
    }

    /// Get exact cache match
    async fn get_exact_match(&self, cache_key: &str) -> Option<CachedSynthesisResult> {
        let cache = self.memory_cache.read().unwrap();

        if let Some(result) = cache.get(cache_key) {
            // Check expiration
            if result.cache_metadata.expires_at > SystemTime::now() {
                return Some(result.clone());
            }
        }

        None
    }

    /// Get similarity-based match
    async fn get_similarity_match(
        &self,
        text: &str,
        _language: &str,
        _config_hash: u64,
    ) -> Option<CachedSynthesisResult> {
        // Collect candidate keys without holding the lock
        let candidate_keys = {
            let similarity_index = self.similarity_index.read().unwrap();
            let mut candidates = Vec::new();

            for (cached_text, similar_keys) in similarity_index.iter() {
                let similarity = self.calculate_text_similarity(text, cached_text);

                if similarity >= self.config.similarity_threshold {
                    candidates.extend(similar_keys.iter().cloned());
                }
            }

            candidates
        };

        // Now search for valid cached results without holding locks during await
        for similar_key in candidate_keys {
            if let Some(result) = self.get_exact_match(&similar_key).await {
                // Update stats for similarity match
                {
                    let mut stats = self.stats.write().unwrap();
                    stats.similarity_matches += 1;
                }

                return Some(result);
            }
        }

        None
    }

    /// Calculate text similarity (simple implementation)
    fn calculate_text_similarity(&self, text1: &str, text2: &str) -> f64 {
        // Simple Levenshtein distance-based similarity
        let distance = self.levenshtein_distance(text1, text2);
        let max_len = text1.len().max(text2.len());

        if max_len == 0 {
            1.0
        } else {
            1.0 - (distance as f64 / max_len as f64)
        }
    }

    /// Calculate Levenshtein distance
    #[allow(clippy::needless_range_loop)]
    fn levenshtein_distance(&self, text1: &str, text2: &str) -> usize {
        let chars1: Vec<char> = text1.chars().collect();
        let chars2: Vec<char> = text2.chars().collect();
        let len1 = chars1.len();
        let len2 = chars2.len();

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[len1][len2]
    }

    /// Cache synthesis result
    pub async fn put_synthesis_result(&self, params: SynthesisCacheParams) -> Result<()> {
        // Check text length limit
        if params.text.len() > self.config.max_text_length {
            return Err(VoirsError::cache_error(format!(
                "Text too long for caching: {} > {}",
                params.text.len(),
                self.config.max_text_length
            )));
        }

        let cache_key = self.make_cache_key(&params.text, &params.language, params.config_hash);

        // Calculate TTL based on quality
        let ttl = self.calculate_ttl(&params.quality_metrics);

        // Calculate size before moving values
        let estimated_size =
            self.estimate_result_size(&params.text, &params.phonemes, &params.audio_buffer);

        // Create cached result
        let cached_result = CachedSynthesisResult {
            text: params.text.clone(),
            language: params.language.clone(),
            config_hash: params.config_hash,
            phonemes: params.phonemes,
            mel_spectrogram: params.mel_spectrogram,
            audio_buffer: params.audio_buffer,
            metadata: params.metadata,
            cache_metadata: CacheMetadata {
                cached_at: SystemTime::now(),
                expires_at: SystemTime::now() + Duration::from_secs(ttl),
                last_accessed: SystemTime::now(),
                access_count: 0,
                hit_count: 0,
                priority: self.determine_priority(&params.quality_metrics),
                pinned: false,
                source: CacheSource::Fresh,
                checksum: self.calculate_checksum(&params.text, params.config_hash),
            },
            quality_metrics: params.quality_metrics.clone(),
            size_bytes: estimated_size,
        };

        // Ensure capacity
        self.ensure_capacity(cached_result.size_bytes).await?;

        // Store in memory cache
        {
            let mut cache = self.memory_cache.write().unwrap();
            let mut current_usage = self.current_memory_usage.write().unwrap();
            let mut lru_queue = self.lru_queue.write().unwrap();

            let result_size = cached_result.size_bytes;
            cache.insert(cache_key.clone(), cached_result.clone());
            *current_usage += result_size;
            lru_queue.push_front(cache_key.clone());
        }

        // Update similarity index
        if self.config.enable_similarity_cache {
            self.update_similarity_index(&params.text, &cache_key).await;
        }

        // Store quality metrics
        {
            let mut quality_cache = self.quality_cache.write().unwrap();
            quality_cache.insert(cache_key.clone(), params.quality_metrics);
        }

        // Update statistics
        self.update_cache_stats(&cached_result).await;

        // Persist to disk if enabled
        if self.config.enable_disk_cache {
            self.persist_to_disk(&cache_key, &cached_result).await?;
        }

        info!(
            "Cached synthesis result for '{}' ({})",
            params.text, cache_key
        );
        Ok(())
    }

    /// Calculate TTL based on quality metrics
    fn calculate_ttl(&self, quality_metrics: &QualityMetrics) -> u64 {
        if quality_metrics.overall_score >= 0.9 {
            self.config.high_quality_ttl_seconds
        } else if quality_metrics.overall_score >= 0.5 {
            self.config.default_ttl_seconds
        } else {
            self.config.low_quality_ttl_seconds
        }
    }

    /// Determine result priority based on quality
    fn determine_priority(&self, quality_metrics: &QualityMetrics) -> ResultPriority {
        if quality_metrics.overall_score >= 0.95 {
            ResultPriority::Critical
        } else if quality_metrics.overall_score >= 0.8 {
            ResultPriority::High
        } else if quality_metrics.overall_score >= 0.6 {
            ResultPriority::Normal
        } else {
            ResultPriority::Low
        }
    }

    /// Ensure sufficient cache capacity
    async fn ensure_capacity(&self, required_bytes: usize) -> Result<()> {
        let current_usage = *self.current_memory_usage.read().unwrap();
        let max_bytes = self.config.memory_cache_size_mb * 1024 * 1024;

        if current_usage + required_bytes > max_bytes {
            self.evict_lru_results(required_bytes).await?;
        }

        Ok(())
    }

    /// Evict LRU results to free space
    async fn evict_lru_results(&self, required_bytes: usize) -> Result<()> {
        let mut freed_bytes = 0;
        let mut evicted_count = 0;

        loop {
            let key_to_evict = {
                let mut lru_queue = self.lru_queue.write().unwrap();
                lru_queue.pop_back()
            };

            if let Some(key) = key_to_evict {
                // Check if result is pinned
                let can_evict = {
                    let cache = self.memory_cache.read().unwrap();
                    cache
                        .get(&key)
                        .map(|result| !result.cache_metadata.pinned)
                        .unwrap_or(false)
                };

                if can_evict {
                    if let Some(result) = self.remove_result(&key).await? {
                        freed_bytes += result.size_bytes;
                        evicted_count += 1;

                        if freed_bytes >= required_bytes {
                            break;
                        }
                    }
                }
            } else {
                // No more results to evict
                break;
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.results_evicted += evicted_count;
        }

        info!(
            "Evicted {} results, freed {} bytes",
            evicted_count, freed_bytes
        );
        Ok(())
    }

    /// Remove a result from cache
    async fn remove_result(&self, key: &str) -> Result<Option<CachedSynthesisResult>> {
        let removed_result = {
            let mut cache = self.memory_cache.write().unwrap();
            cache.remove(key)
        };

        if let Some(ref result) = removed_result {
            // Update memory usage
            {
                let mut current_usage = self.current_memory_usage.write().unwrap();
                *current_usage = current_usage.saturating_sub(result.size_bytes);
            }

            // Remove from quality cache
            {
                let mut quality_cache = self.quality_cache.write().unwrap();
                quality_cache.remove(key);
            }

            // Update statistics
            {
                let mut stats = self.stats.write().unwrap();
                stats.basic_stats.total_entries = stats.basic_stats.total_entries.saturating_sub(1);
                stats.basic_stats.memory_usage_bytes = *self.current_memory_usage.read().unwrap();
            }
        }

        Ok(removed_result)
    }

    /// Update similarity index
    async fn update_similarity_index(&self, text: &str, cache_key: &str) {
        let mut similarity_index = self.similarity_index.write().unwrap();

        similarity_index
            .entry(text.to_string())
            .or_default()
            .push(cache_key.to_string());

        // Limit the number of similar results
        for similar_keys in similarity_index.values_mut() {
            if similar_keys.len() > self.config.max_similar_results {
                similar_keys.truncate(self.config.max_similar_results);
            }
        }
    }

    /// Update access information
    async fn update_access_info(&self, cache_key: &str) {
        let mut access_frequency = self.access_frequency.write().unwrap();
        let now = SystemTime::now();

        let access_info = access_frequency
            .entry(cache_key.to_string())
            .or_insert_with(|| AccessInfo {
                access_count: 0,
                last_access: now,
                frequency: 0.0,
                access_history: VecDeque::new(),
            });

        access_info.access_count += 1;
        access_info.last_access = now;
        access_info.access_history.push_front(now);

        // Keep only recent history
        while access_info.access_history.len() > 100 {
            access_info.access_history.pop_back();
        }

        // Calculate frequency (accesses per hour)
        let hour_ago = now - Duration::from_secs(3600);
        let recent_accesses = access_info
            .access_history
            .iter()
            .filter(|&&access_time| access_time >= hour_ago)
            .count();

        access_info.frequency = recent_accesses as f64;
    }

    /// Update hit statistics
    async fn update_hit_stats(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.results_served_from_cache += 1;

        // Update hit rate
        let total_requests = stats.results_served_from_cache + stats.cache_misses;
        stats.basic_stats.hit_rate =
            ((stats.results_served_from_cache as f64 / total_requests as f64) * 100.0) as f32;
        stats.basic_stats.miss_rate = 100.0 - stats.basic_stats.hit_rate;
    }

    /// Update miss statistics
    async fn update_miss_stats(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.cache_misses += 1;

        // Update miss rate
        let total_requests = stats.results_served_from_cache + stats.cache_misses;
        stats.basic_stats.miss_rate =
            ((stats.cache_misses as f64 / total_requests as f64) * 100.0) as f32;
        stats.basic_stats.hit_rate = 100.0 - stats.basic_stats.miss_rate;
    }

    /// Update cache statistics
    async fn update_cache_stats(&self, result: &CachedSynthesisResult) {
        let mut stats = self.stats.write().unwrap();

        stats.results_generated += 1;
        stats.basic_stats.total_entries += 1;
        stats.basic_stats.memory_usage_bytes = *self.current_memory_usage.read().unwrap();

        // Update quality distribution
        let quality = result.quality_metrics.overall_score;
        if quality >= 0.9 {
            stats.quality_distribution.excellent += 1;
        } else if quality >= 0.7 {
            stats.quality_distribution.good += 1;
        } else if quality >= 0.5 {
            stats.quality_distribution.fair += 1;
        } else {
            stats.quality_distribution.poor += 1;
        }

        // Update language distribution
        *stats
            .language_distribution
            .entry(result.language.clone())
            .or_insert(0) += 1;

        // Update text length distribution
        let length_category = match result.text.len() {
            0..=100 => "short",
            101..=500 => "medium",
            501..=2000 => "long",
            _ => "very_long",
        };
        *stats
            .text_length_distribution
            .entry(length_category.to_string())
            .or_insert(0) += 1;

        // Update average quality
        let total_quality =
            stats.avg_quality_score * (stats.results_generated - 1) as f64 + quality;
        stats.avg_quality_score = total_quality / stats.results_generated as f64;

        // Update average synthesis time
        let synthesis_time = result.metadata.synthesis_duration_ms as f64;
        let total_time =
            stats.avg_synthesis_time_ms * (stats.results_generated - 1) as f64 + synthesis_time;
        stats.avg_synthesis_time_ms = total_time / stats.results_generated as f64;
    }

    /// Persist result to disk
    async fn persist_to_disk(&self, cache_key: &str, result: &CachedSynthesisResult) -> Result<()> {
        if let Some(ref cache_dir) = self.disk_cache_dir {
            let file_path = cache_dir.join(format!("{cache_key}.cache"));

            // Serialize result (simplified - would use proper serialization in practice)
            let serialized = serde_json::to_vec(result)
                .map_err(|e| VoirsError::cache_error(format!("Failed to serialize result: {e}")))?;

            // Write to disk
            let mut file = fs::File::create(&file_path).await.map_err(|e| {
                VoirsError::cache_error(format!("Failed to create cache file: {e}"))
            })?;

            file.write_all(&serialized)
                .await
                .map_err(|e| VoirsError::cache_error(format!("Failed to write cache file: {e}")))?;

            debug!("Persisted result to disk: {}", file_path.display());
        }

        Ok(())
    }

    /// Make cache key
    fn make_cache_key(&self, text: &str, language: &str, config_hash: u64) -> String {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        language.hash(&mut hasher);
        config_hash.hash(&mut hasher);

        format!("synthesis_{}_{:x}", language, hasher.finish())
    }

    /// Calculate checksum
    fn calculate_checksum(&self, text: &str, config_hash: u64) -> String {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        config_hash.hash(&mut hasher);

        format!("{:x}", hasher.finish())
    }

    /// Estimate result size
    fn estimate_result_size(&self, text: &str, phonemes: &[Phoneme], audio: &AudioBuffer) -> usize {
        text.len() +
        phonemes.len() * 32 + // Rough estimate for phoneme data
        audio.samples().len() * 4 + // Assuming 32-bit samples
        1024 // Metadata overhead
    }

    /// Clear expired results
    pub async fn cleanup_expired(&self) -> Result<usize> {
        let now = SystemTime::now();
        let mut expired_keys = Vec::new();

        // Find expired results
        {
            let cache = self.memory_cache.read().unwrap();
            for (key, result) in cache.iter() {
                if result.cache_metadata.expires_at <= now && !result.cache_metadata.pinned {
                    expired_keys.push(key.clone());
                }
            }
        }

        // Remove expired results
        let mut removed_count = 0;
        for key in expired_keys {
            if self.remove_result(&key).await?.is_some() {
                removed_count += 1;
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.results_expired += removed_count;
        }

        Ok(removed_count as usize)
    }

    /// Clear all cached results
    pub async fn clear(&self) -> Result<()> {
        {
            let mut cache = self.memory_cache.write().unwrap();
            let mut current_usage = self.current_memory_usage.write().unwrap();
            let mut lru_queue = self.lru_queue.write().unwrap();
            let mut access_frequency = self.access_frequency.write().unwrap();
            let mut similarity_index = self.similarity_index.write().unwrap();
            let mut quality_cache = self.quality_cache.write().unwrap();

            cache.clear();
            *current_usage = 0;
            lru_queue.clear();
            access_frequency.clear();
            similarity_index.clear();
            quality_cache.clear();
        }

        // Reset statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.basic_stats.total_entries = 0;
            stats.basic_stats.memory_usage_bytes = 0;
        }

        info!("Cleared all synthesis results from cache");
        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> ResultCacheStats {
        self.stats.read().unwrap().clone()
    }

    /// Get cache usage summary
    pub async fn get_usage_summary(&self) -> CacheUsageSummary {
        let cache = self.memory_cache.read().unwrap();
        let current_usage = *self.current_memory_usage.read().unwrap();
        let max_bytes = self.config.memory_cache_size_mb * 1024 * 1024;

        CacheUsageSummary {
            total_results: cache.len(),
            memory_usage_bytes: current_usage,
            memory_usage_mb: current_usage / (1024 * 1024),
            memory_utilization: (current_usage as f64 / max_bytes as f64) * 100.0,
            avg_result_size: if !cache.is_empty() {
                current_usage / cache.len()
            } else {
                0
            },
        }
    }
}

/// Cache usage summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheUsageSummary {
    /// Total number of cached results
    pub total_results: usize,

    /// Memory usage in bytes
    pub memory_usage_bytes: usize,

    /// Memory usage in MB
    pub memory_usage_mb: usize,

    /// Memory utilization percentage
    pub memory_utilization: f64,

    /// Average result size in bytes
    pub avg_result_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        audio::AudioBuffer,
        types::{MelSpectrogram, Phoneme},
    };

    #[tokio::test]
    async fn test_synthesis_result_cache_creation() {
        let config = ResultCacheConfig::default();
        let cache = SynthesisResultCache::new(config, None).unwrap();

        let summary = cache.get_usage_summary().await;
        assert_eq!(summary.total_results, 0);
    }

    #[tokio::test]
    async fn test_result_caching_and_retrieval() {
        let config = ResultCacheConfig::default();
        let cache = SynthesisResultCache::new(config, None).unwrap();

        let text = "Hello, world!";
        let language = "en";
        let config_hash = 12345u64;
        let phonemes = vec![Phoneme::new("h"), Phoneme::new("ɛ")];
        let mel = MelSpectrogram::new(vec![vec![0.5; 100]; 80], 22050, 256);
        let audio = AudioBuffer::sine_wave(440.0, 1.0, 22050, 0.5);

        let metadata = SynthesisMetadata {
            voice_id: "test_voice".to_string(),
            model_versions: HashMap::new(),
            synthesis_duration_ms: 100,
            audio_duration_seconds: 1.0,
            sample_rate: 22050,
            bit_depth: 16,
            channels: 1,
            parameters: HashMap::new(),
            stage_timings: HashMap::new(),
        };

        let quality_metrics = QualityMetrics {
            overall_score: 0.85,
            audio_quality: AudioQualityMetrics {
                snr_db: 30.0,
                thd_percent: 0.1,
                frequency_response_score: 0.9,
                dynamic_range_db: 60.0,
                spectral_quality_score: 0.85,
            },
            pronunciation_accuracy: 0.9,
            naturalness_score: 0.8,
            intelligibility_score: 0.95,
            prosody_quality: 0.8,
            confidence_score: 0.85,
            error_metrics: ErrorMetrics {
                pronunciation_errors: 0,
                timing_errors: 0,
                spectral_artifacts: 0,
                discontinuities: 0,
            },
        };

        // Cache the result
        let cache_params = SynthesisCacheParams {
            text: text.to_string(),
            language: language.to_string(),
            config_hash,
            phonemes,
            mel_spectrogram: mel,
            audio_buffer: audio,
            metadata,
            quality_metrics,
        };

        cache.put_synthesis_result(cache_params).await.unwrap();

        // Retrieve the result
        let result = cache
            .get_synthesis_result(text, language, config_hash)
            .await;
        assert!(result.is_some());

        let cached = result.unwrap();
        assert_eq!(cached.text, text);
        assert_eq!(cached.language, language);
    }

    #[tokio::test]
    async fn test_result_expiration() {
        let config = ResultCacheConfig {
            default_ttl_seconds: 1, // Very short TTL
            ..Default::default()
        };

        let cache = SynthesisResultCache::new(config, None).unwrap();

        let text = "Test expiration";
        let language = "en";
        let config_hash = 67890u64;
        let phonemes = vec![Phoneme::new("t")];
        let mel = MelSpectrogram::new(vec![vec![0.5; 10]; 10], 22050, 256);
        let audio = AudioBuffer::sine_wave(440.0, 0.1, 22050, 0.5);

        let metadata = SynthesisMetadata {
            voice_id: "test_voice".to_string(),
            model_versions: HashMap::new(),
            synthesis_duration_ms: 50,
            audio_duration_seconds: 0.1,
            sample_rate: 22050,
            bit_depth: 16,
            channels: 1,
            parameters: HashMap::new(),
            stage_timings: HashMap::new(),
        };

        let quality_metrics = QualityMetrics {
            overall_score: 0.7,
            audio_quality: AudioQualityMetrics {
                snr_db: 25.0,
                thd_percent: 0.2,
                frequency_response_score: 0.8,
                dynamic_range_db: 50.0,
                spectral_quality_score: 0.7,
            },
            pronunciation_accuracy: 0.8,
            naturalness_score: 0.7,
            intelligibility_score: 0.9,
            prosody_quality: 0.7,
            confidence_score: 0.7,
            error_metrics: ErrorMetrics {
                pronunciation_errors: 1,
                timing_errors: 0,
                spectral_artifacts: 0,
                discontinuities: 0,
            },
        };

        let cache_params = SynthesisCacheParams {
            text: text.to_string(),
            language: language.to_string(),
            config_hash,
            phonemes,
            mel_spectrogram: mel,
            audio_buffer: audio,
            metadata,
            quality_metrics,
        };

        cache.put_synthesis_result(cache_params).await.unwrap();

        // Should find it immediately
        assert!(cache
            .get_synthesis_result(text, language, config_hash)
            .await
            .is_some());

        // Wait for expiration
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        // Should not find it after expiration
        assert!(cache
            .get_synthesis_result(text, language, config_hash)
            .await
            .is_none());
    }

    #[tokio::test]
    async fn test_cache_cleanup() {
        let config = ResultCacheConfig::default();
        let cache = SynthesisResultCache::new(config, None).unwrap();

        // Add a test result
        let text = "Cleanup test";
        let language = "en";
        let config_hash = 99999u64;
        let phonemes = vec![Phoneme::new("k")];
        let mel = MelSpectrogram::new(vec![vec![0.5; 10]; 10], 22050, 256);
        let audio = AudioBuffer::sine_wave(440.0, 0.1, 22050, 0.5);

        let metadata = SynthesisMetadata {
            voice_id: "test_voice".to_string(),
            model_versions: HashMap::new(),
            synthesis_duration_ms: 30,
            audio_duration_seconds: 0.1,
            sample_rate: 22050,
            bit_depth: 16,
            channels: 1,
            parameters: HashMap::new(),
            stage_timings: HashMap::new(),
        };

        let quality_metrics = QualityMetrics {
            overall_score: 0.6,
            audio_quality: AudioQualityMetrics {
                snr_db: 20.0,
                thd_percent: 0.3,
                frequency_response_score: 0.7,
                dynamic_range_db: 40.0,
                spectral_quality_score: 0.6,
            },
            pronunciation_accuracy: 0.7,
            naturalness_score: 0.6,
            intelligibility_score: 0.8,
            prosody_quality: 0.6,
            confidence_score: 0.6,
            error_metrics: ErrorMetrics {
                pronunciation_errors: 2,
                timing_errors: 1,
                spectral_artifacts: 0,
                discontinuities: 0,
            },
        };

        let cache_params = SynthesisCacheParams {
            text: text.to_string(),
            language: language.to_string(),
            config_hash,
            phonemes,
            mel_spectrogram: mel,
            audio_buffer: audio,
            metadata,
            quality_metrics,
        };

        cache.put_synthesis_result(cache_params).await.unwrap();

        // Cleanup (should find 0 expired since we just added it)
        let expired_count = cache.cleanup_expired().await.unwrap();
        assert_eq!(expired_count, 0);

        // Clear all
        cache.clear().await.unwrap();
        let summary = cache.get_usage_summary().await;
        assert_eq!(summary.total_results, 0);
    }
}
