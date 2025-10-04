//! Performance optimization for singing synthesis
//!
//! This module provides advanced performance optimizations including voice caching,
//! precomputation, streaming, and data compression for high-performance singing synthesis.

#![allow(dead_code)]

use crate::models::VoiceModel;
use crate::score::{MusicalNote, MusicalScore};
use crate::types::VoiceCharacteristics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Optimized LRU cache implementation with proper access order tracking
#[derive(Debug)]
struct Lru<K, V> {
    map: HashMap<K, (V, usize)>, // (value, access_order)
    access_counter: usize,
    capacity: usize,
}

impl<K: Clone + std::hash::Hash + Eq, V> Lru<K, V> {
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            access_counter: 0,
            capacity,
        }
    }

    fn get(&mut self, key: &K) -> Option<&V> {
        if let Some((value, access_order)) = self.map.get_mut(key) {
            self.access_counter = self.access_counter.wrapping_add(1);
            *access_order = self.access_counter;
            Some(value)
        } else {
            None
        }
    }

    fn insert(&mut self, key: K, value: V) {
        // Remove least recently used items if at capacity
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            self.evict_lru();
        }

        self.access_counter = self.access_counter.wrapping_add(1);
        self.map.insert(key, (value, self.access_counter));
    }

    /// Evict the least recently used item
    fn evict_lru(&mut self) {
        if let Some((lru_key, _)) = self
            .map
            .iter()
            .min_by_key(|(_, (_, access_order))| *access_order)
            .map(|(k, _)| (k.clone(), ()))
        {
            self.map.remove(&lru_key);
        }
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn clear(&mut self) {
        self.map.clear();
        self.access_counter = 0;
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        self.map.remove(key).map(|(value, _)| value)
    }

    /// Get keys sorted by access order (most recent first)
    fn keys_by_access_order(&self) -> Vec<K> {
        let mut items: Vec<_> = self
            .map
            .iter()
            .map(|(k, (_, access_order))| (k.clone(), *access_order))
            .collect();
        items.sort_by_key(|(_, access_order)| std::cmp::Reverse(*access_order));
        items.into_iter().map(|(k, _)| k).collect()
    }

    /// Remove old entries based on access time threshold
    fn remove_old_entries(&mut self, threshold_age: usize) {
        let current_counter = self.access_counter;
        let keys_to_remove: Vec<K> = self
            .map
            .iter()
            .filter(|(_, (_, access_order))| {
                current_counter.saturating_sub(*access_order) > threshold_age
            })
            .map(|(k, _)| k.clone())
            .collect();

        for key in keys_to_remove {
            self.map.remove(&key);
        }
    }
}

/// Voice caching system for fast voice model switching
#[derive(Debug)]
pub struct VoiceCache {
    /// LRU cache for voice models
    cache: Arc<RwLock<Lru<String, CachedVoice>>>,
    /// Cache configuration
    config: VoiceCacheConfig,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
    /// Background preloader
    preloader: Arc<Mutex<VoicePreloader>>,
}

/// Cached voice data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedVoice {
    /// Voice model data
    pub model: VoiceModel,
    /// Voice characteristics
    pub characteristics: VoiceCharacteristics,
    /// Precomputed synthesis parameters
    pub synthesis_params: PrecomputedSynthesisParams,
    /// Compression metadata
    pub compression_info: CompressionInfo,
    /// Last access time
    pub last_accessed: std::time::SystemTime,
    /// Usage frequency
    pub usage_count: u32,
    /// Memory footprint (bytes)
    pub memory_footprint: usize,
}

/// Voice cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCacheConfig {
    /// Maximum number of cached voices
    pub max_voices: usize,
    /// Maximum memory usage (bytes)
    pub max_memory: usize,
    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Preload popular voices
    pub preload_popular: bool,
    /// Background preloading
    pub background_preload: bool,
    /// Compression settings
    pub compression: CompressionConfig,
    /// Cache persistence
    pub persistent_cache: bool,
    /// Cache directory path
    pub cache_directory: Option<String>,
}

/// Cache eviction policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time-based expiration
    TTL,
    /// Size-based eviction
    SizeBased,
    /// Hybrid policy
    Hybrid,
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Total requests
    pub requests: u64,
    /// Current memory usage
    pub memory_usage: usize,
    /// Average load time (ms)
    pub avg_load_time: f32,
    /// Eviction count
    pub evictions: u64,
    /// Preload successes
    pub preload_successes: u64,
    /// Preload failures
    pub preload_failures: u64,
}

/// Voice preloader for background loading
#[derive(Debug)]
pub struct VoicePreloader {
    /// Voices to preload
    preload_queue: Vec<String>,
    /// Preload priority scores
    priority_scores: HashMap<String, f32>,
    /// Currently preloading
    currently_preloading: Option<String>,
    /// Preload statistics
    preload_stats: PreloadStats,
}

/// Preload statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreloadStats {
    /// Voices preloaded
    pub voices_preloaded: u32,
    /// Total preload time
    pub total_preload_time: Duration,
    /// Average preload time per voice
    pub avg_preload_time: Duration,
    /// Preload success rate
    pub success_rate: f32,
}

/// Precomputation system for expensive calculations
#[derive(Debug)]
pub struct PrecomputationEngine {
    /// Precomputed data cache
    cache: Arc<RwLock<HashMap<String, PrecomputedData>>>,
    /// Computation configuration
    config: PrecomputationConfig,
    /// Computation statistics
    stats: Arc<RwLock<ComputationStats>>,
    /// Background computation thread pool
    thread_pool: Option<ThreadPool>,
}

/// Precomputed data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrecomputedData {
    /// Synthesis parameters
    SynthesisParams(PrecomputedSynthesisParams),
    /// Pitch contours
    PitchContours(Vec<f32>),
    /// Harmonic series
    HarmonicSeries(HarmonicSeriesData),
    /// Formant frequencies
    FormantFrequencies(Vec<f32>),
    /// Spectral envelopes
    SpectralEnvelopes(SpectralData),
    /// Phase relationships
    PhaseRelationships(PhaseData),
}

/// Precomputed synthesis parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecomputedSynthesisParams {
    /// Voice-specific parameters
    pub voice_params: HashMap<String, f32>,
    /// Frequency-dependent coefficients
    pub frequency_coefficients: Vec<f32>,
    /// Dynamic range mappings
    pub dynamic_mappings: Vec<(f32, f32)>,
    /// Articulation lookup tables
    pub articulation_luts: HashMap<String, Vec<f32>>,
    /// Vibrato wave tables
    pub vibrato_tables: Vec<f32>,
    /// Breath pattern templates
    pub breath_templates: Vec<Vec<f32>>,
}

/// Harmonic series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicSeriesData {
    /// Fundamental frequency
    pub fundamental: f32,
    /// Harmonic frequencies
    pub harmonics: Vec<f32>,
    /// Harmonic amplitudes
    pub amplitudes: Vec<f32>,
    /// Harmonic phases
    pub phases: Vec<f32>,
    /// Inharmonicity factors
    pub inharmonicity: Vec<f32>,
}

/// Spectral data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralData {
    /// Frequency bins
    pub frequencies: Vec<f32>,
    /// Magnitude spectrum
    pub magnitudes: Vec<f32>,
    /// Phase spectrum
    pub phases: Vec<f32>,
    /// Spectral centroids
    pub centroids: Vec<f32>,
}

/// Phase relationship data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseData {
    /// Phase offsets between harmonics
    pub harmonic_phases: Vec<f32>,
    /// Cross-channel phase relationships
    pub channel_phases: HashMap<String, f32>,
    /// Time-varying phase evolution
    pub phase_evolution: Vec<f32>,
}

/// Precomputation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecomputationConfig {
    /// Enable precomputation
    pub enabled: bool,
    /// Precomputation strategies
    pub strategies: Vec<PrecomputationStrategy>,
    /// Background computation
    pub background_compute: bool,
    /// Memory limit for precomputed data
    pub memory_limit: usize,
    /// Computation quality level
    pub quality_level: ComputationQuality,
    /// Cache persistence
    pub persistent_cache: bool,
    /// Adaptive precomputation
    pub adaptive: bool,
}

/// Precomputation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecomputationStrategy {
    /// Precompute at voice load time
    VoiceLoad,
    /// Precompute at score analysis time
    ScoreAnalysis,
    /// Precompute during idle time
    IdleTime,
    /// Predictive precomputation
    Predictive,
    /// Just-in-time computation
    JustInTime,
}

/// Computation quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputationQuality {
    /// Fast computation, lower accuracy
    Fast,
    /// Balanced computation
    Balanced,
    /// High quality computation
    High,
    /// Maximum quality computation
    Maximum,
}

/// Computation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComputationStats {
    /// Total computations performed
    pub computations: u64,
    /// Cache hits for precomputed data
    pub precompute_hits: u64,
    /// Cache misses requiring computation
    pub precompute_misses: u64,
    /// Average computation time (ms)
    pub avg_compute_time: f32,
    /// Memory used by precomputed data
    pub memory_usage: usize,
    /// Background computations completed
    pub background_computations: u64,
}

/// Streaming system for large musical scores
#[derive(Debug)]
pub struct StreamingEngine {
    /// Streaming configuration
    config: StreamingConfig,
    /// Current stream state
    state: Arc<RwLock<StreamState>>,
    /// Buffer management
    buffer_manager: Arc<Mutex<BufferManager>>,
    /// Streaming statistics
    stats: Arc<RwLock<StreamingStats>>,
}

/// Streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Buffer size in samples
    pub buffer_size: usize,
    /// Number of buffers
    pub buffer_count: usize,
    /// Lookahead time (seconds)
    pub lookahead_time: f32,
    /// Streaming quality
    pub quality: StreamingQuality,
    /// Adaptive buffering
    pub adaptive_buffering: bool,
    /// Background loading
    pub background_loading: bool,
    /// Memory limit for streaming buffers
    pub memory_limit: usize,
}

/// Streaming quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamingQuality {
    /// Low latency, basic quality
    LowLatency,
    /// Balanced latency and quality
    Balanced,
    /// High quality, higher latency
    HighQuality,
    /// Maximum quality streaming
    Maximum,
}

/// Current streaming state
#[derive(Debug, Clone)]
pub struct StreamState {
    /// Currently playing position
    pub playback_position: f64,
    /// Currently loaded score section
    pub loaded_section: Option<ScoreSection>,
    /// Next sections to load
    pub pending_sections: Vec<ScoreSection>,
    /// Streaming status
    pub status: StreamingStatus,
    /// Buffer fill levels
    pub buffer_levels: Vec<f32>,
}

/// Score section for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreSection {
    /// Start time (seconds)
    pub start_time: f64,
    /// End time (seconds)
    pub end_time: f64,
    /// Musical notes in this section
    pub notes: Vec<MusicalNote>,
    /// Voice assignments
    pub voice_assignments: HashMap<String, String>,
    /// Section priority
    pub priority: u8,
}

/// Streaming status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamingStatus {
    /// Ready to stream
    Ready,
    /// Currently streaming
    Streaming,
    /// Buffering
    Buffering,
    /// Paused
    Paused,
    /// Error state
    Error,
}

/// Buffer manager for streaming
#[derive(Debug)]
pub struct BufferManager {
    /// Audio buffers
    audio_buffers: Vec<AudioBuffer>,
    /// Buffer allocation strategy
    allocation_strategy: BufferAllocationStrategy,
    /// Memory pool
    memory_pool: MemoryPool,
    /// Buffer statistics
    buffer_stats: BufferStats,
}

/// Audio buffer for streaming
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Buffer ID
    pub id: String,
    /// Audio data
    pub data: Vec<f32>,
    /// Buffer timestamp
    pub timestamp: f64,
    /// Buffer duration
    pub duration: f64,
    /// Buffer status
    pub status: BufferStatus,
    /// Buffer priority
    pub priority: u8,
}

/// Buffer status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BufferStatus {
    /// Empty buffer
    Empty,
    /// Being filled
    Filling,
    /// Ready for playback
    Ready,
    /// Currently playing
    Playing,
    /// Played and can be recycled
    Played,
}

/// Buffer allocation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BufferAllocationStrategy {
    /// Static allocation
    Static,
    /// Dynamic allocation
    Dynamic,
    /// Pool-based allocation
    Pooled,
    /// Circular buffer
    Circular,
}

/// Memory pool for efficient allocation
#[derive(Debug)]
pub struct MemoryPool {
    /// Available memory blocks
    available_blocks: Vec<MemoryBlock>,
    /// Allocated blocks
    allocated_blocks: HashMap<String, MemoryBlock>,
    /// Pool configuration
    config: MemoryPoolConfig,
}

/// Memory block
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Block ID
    pub id: String,
    /// Block size
    pub size: usize,
    /// Block data
    pub data: Vec<u8>,
    /// Allocation timestamp
    pub allocated_at: Instant,
}

/// Memory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Initial pool size
    pub initial_size: usize,
    /// Maximum pool size
    pub max_size: usize,
    /// Block size
    pub block_size: usize,
    /// Growth strategy
    pub growth_strategy: PoolGrowthStrategy,
}

/// Pool growth strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolGrowthStrategy {
    /// Fixed size pool
    Fixed,
    /// Linear growth
    Linear,
    /// Exponential growth
    Exponential,
    /// Adaptive growth
    Adaptive,
}

/// Buffer statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BufferStats {
    /// Total buffers allocated
    pub buffers_allocated: u32,
    /// Current buffer usage
    pub current_usage: usize,
    /// Peak buffer usage
    pub peak_usage: usize,
    /// Buffer underruns
    pub underruns: u32,
    /// Buffer overruns
    pub overruns: u32,
    /// Average buffer fill time (ms)
    pub avg_fill_time: f32,
}

/// Streaming statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamingStats {
    /// Total streaming time
    pub total_streaming_time: Duration,
    /// Data streamed (bytes)
    pub data_streamed: u64,
    /// Streaming interruptions
    pub interruptions: u32,
    /// Average streaming bitrate
    pub avg_bitrate: f32,
    /// Buffer health score (0.0-1.0)
    pub buffer_health: f32,
}

/// Data compression system
pub struct CompressionEngine {
    /// Compression configuration
    config: CompressionConfig,
    /// Compression statistics
    stats: Arc<RwLock<CompressionStats>>,
    /// Compression algorithms
    algorithms: HashMap<CompressionAlgorithm, Box<dyn CompressionCodec>>,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Default compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (0-9)
    pub level: u8,
    /// Quality preservation
    pub preserve_quality: bool,
    /// Lossy compression allowed
    pub allow_lossy: bool,
    /// Target compression ratio
    pub target_ratio: f32,
    /// Adaptive compression
    pub adaptive: bool,
}

/// Compression algorithms
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// FLAC lossless compression
    FLAC,
    /// General purpose compression (zlib)
    Deflate,
    /// High compression ratio (LZMA)
    LZMA,
    /// Fast compression (LZ4)
    LZ4,
    /// Voice-optimized compression
    VoiceOptimized,
    /// Custom compression algorithm
    Custom(String),
}

/// Compression codec trait
pub trait CompressionCodec: Send + Sync {
    /// Compress data
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError>;

    /// Decompress data
    fn decompress(&self, compressed_data: &[u8]) -> Result<Vec<u8>, CompressionError>;

    /// Get compression ratio estimate
    fn estimate_ratio(&self, data: &[u8]) -> f32;

    /// Get codec information
    fn info(&self) -> CompressionCodecInfo;
}

/// Compression codec information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionCodecInfo {
    /// Codec name
    pub name: String,
    /// Compression speed (relative)
    pub speed: CompressionSpeed,
    /// Typical compression ratio
    pub typical_ratio: f32,
    /// Is lossless
    pub lossless: bool,
}

/// Compression speed levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionSpeed {
    /// Very fast compression
    VeryFast,
    /// Fast compression
    Fast,
    /// Balanced speed/ratio
    Balanced,
    /// Slow but high ratio
    Slow,
    /// Very slow, maximum ratio
    VerySlow,
}

/// Compression error types
#[derive(Debug, thiserror::Error)]
pub enum CompressionError {
    /// Algorithm not supported
    #[error("Compression algorithm not supported: {0:?}")]
    UnsupportedAlgorithm(CompressionAlgorithm),

    /// Compression failed
    #[error("Compression failed: {0}")]
    CompressionFailed(String),

    /// Decompression failed
    #[error("Decompression failed: {0}")]
    DecompressionFailed(String),

    /// Invalid data
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

/// Compression metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionInfo {
    /// Algorithm used
    pub algorithm: CompressionAlgorithm,
    /// Original size
    pub original_size: usize,
    /// Compressed size
    pub compressed_size: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Compression time
    pub compression_time: Duration,
    /// Checksum for verification
    pub checksum: u32,
}

/// Compression statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Total compressions
    pub compressions: u64,
    /// Total decompressions
    pub decompressions: u64,
    /// Bytes compressed
    pub bytes_compressed: u64,
    /// Bytes decompressed
    pub bytes_decompressed: u64,
    /// Average compression ratio
    pub avg_compression_ratio: f32,
    /// Average compression time (ms)
    pub avg_compression_time: f32,
    /// Average decompression time (ms)
    pub avg_decompression_time: f32,
}

/// Thread pool for background operations
#[derive(Debug)]
struct ThreadPool {
    // Placeholder for thread pool implementation
    // In practice, you'd use a crate like `rayon` or `tokio`
    _phantom: std::marker::PhantomData<()>,
}

// === Implementation ===

impl VoiceCache {
    /// Create a new voice cache
    pub fn new(config: VoiceCacheConfig) -> Self {
        Self {
            cache: Arc::new(RwLock::new(Lru::new(config.max_voices))),
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
            preloader: Arc::new(Mutex::new(VoicePreloader::new())),
        }
    }

    /// Get a voice from cache
    pub fn get_voice(&self, voice_id: &str) -> Option<CachedVoice> {
        let mut stats = self.stats.write().unwrap();
        stats.requests += 1;

        if let Ok(mut cache) = self.cache.write() {
            if let Some(cached_voice) = cache.get(&voice_id.to_string()) {
                stats.hits += 1;
                let mut voice = cached_voice.clone();
                voice.last_accessed = std::time::SystemTime::now();
                voice.usage_count += 1;

                // Update the cache entry with new access info
                cache.insert(voice_id.to_string(), voice.clone());
                Some(voice)
            } else {
                stats.misses += 1;
                None
            }
        } else {
            None
        }
    }

    /// Cache a voice
    pub fn cache_voice(&self, voice_id: String, voice: CachedVoice) {
        if let Ok(mut cache) = self.cache.write() {
            // Check memory limits
            if self.check_memory_limits(&voice) {
                cache.insert(voice_id, voice);
            } else {
                self.evict_voices_for_space(voice.memory_footprint);
                cache.insert(voice_id, voice);
            }
        }
    }

    /// Check if memory limits allow caching this voice
    fn check_memory_limits(&self, voice: &CachedVoice) -> bool {
        let stats = self.stats.read().unwrap();
        stats.memory_usage + voice.memory_footprint <= self.config.max_memory
    }

    /// Evict voices to make space
    fn evict_voices_for_space(&self, needed_space: usize) {
        match self.config.eviction_policy {
            EvictionPolicy::LRU => self.evict_lru(needed_space),
            EvictionPolicy::LFU => self.evict_lfu(needed_space),
            _ => self.evict_lru(needed_space), // Default to LRU
        }
    }

    /// Evict least recently used voices
    fn evict_lru(&self, _needed_space: usize) {
        // Implementation would evict oldest voices based on last_accessed
        // For brevity, simplified here
        if let Ok(mut stats) = self.stats.write() {
            stats.evictions += 1;
        }
    }

    /// Evict least frequently used voices
    fn evict_lfu(&self, _needed_space: usize) {
        // Implementation would evict voices with lowest usage_count
        if let Ok(mut stats) = self.stats.write() {
            stats.evictions += 1;
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        self.stats.read().unwrap().clone()
    }

    /// Clear the cache
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
        if let Ok(mut stats) = self.stats.write() {
            stats.memory_usage = 0;
            *stats = CacheStats::default();
        }
    }

    /// Perform memory leak prevention cleanup
    pub fn cleanup_memory_leaks(&self) {
        // Force cleanup of expired cache entries
        if let Ok(mut cache) = self.cache.write() {
            let now = std::time::SystemTime::now();
            let mut expired_keys = Vec::new();

            // Collect expired entries (older than 1 hour)
            for (key, (voice, _)) in cache.map.iter() {
                if let Ok(elapsed) = now.duration_since(voice.last_accessed) {
                    if elapsed > std::time::Duration::from_secs(3600) {
                        expired_keys.push(key.clone());
                    }
                }
            }

            // Remove expired entries to prevent memory leaks
            for key in expired_keys {
                cache.remove(&key);
            }

            // Additional cleanup: remove old entries based on access patterns
            cache.remove_old_entries(10000); // Remove entries not accessed in last 10000 operations
        }

        // Update memory usage statistics
        self.update_memory_stats();
    }

    /// Update memory usage statistics to prevent accounting leaks
    fn update_memory_stats(&self) {
        if let Ok(cache) = self.cache.read() {
            if let Ok(mut stats) = self.stats.write() {
                stats.memory_usage = cache
                    .map
                    .values()
                    .map(|(voice, _)| voice.memory_footprint)
                    .sum();
            }
        }
    }

    /// Get cache efficiency metrics for optimization
    pub fn get_cache_efficiency(&self) -> f32 {
        if let Ok(stats) = self.stats.read() {
            if stats.requests > 0 {
                stats.hits as f32 / stats.requests as f32
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Optimize cache by removing low-priority items
    pub fn optimize_cache(&self) {
        if let Ok(mut cache) = self.cache.write() {
            // If cache is near capacity, proactively remove less accessed items
            if cache.len() > (self.config.max_voices * 3 / 4) {
                let keys_by_access = cache.keys_by_access_order();
                let to_remove = cache.len() - (self.config.max_voices / 2);

                // Remove least recently used items
                for key in keys_by_access.iter().rev().take(to_remove) {
                    cache.remove(key);
                }
            }
        }

        self.update_memory_stats();
    }
}

impl Default for VoicePreloader {
    fn default() -> Self {
        Self::new()
    }
}

impl VoicePreloader {
    /// Create a new voice preloader
    pub fn new() -> Self {
        Self {
            preload_queue: Vec::new(),
            priority_scores: HashMap::new(),
            currently_preloading: None,
            preload_stats: PreloadStats::default(),
        }
    }

    /// Add voice to preload queue
    pub fn queue_voice(&mut self, voice_id: String, priority: f32) {
        if !self.preload_queue.contains(&voice_id) {
            self.preload_queue.push(voice_id.clone());
            self.priority_scores.insert(voice_id, priority);
            self.sort_by_priority();
        }
    }

    /// Sort preload queue by priority
    fn sort_by_priority(&mut self) {
        self.preload_queue.sort_by(|a, b| {
            let priority_a = self.priority_scores.get(a).unwrap_or(&0.0);
            let priority_b = self.priority_scores.get(b).unwrap_or(&0.0);
            priority_a
                .partial_cmp(priority_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get next voice to preload
    pub fn next_voice(&mut self) -> Option<String> {
        self.preload_queue.pop()
    }
}

impl PrecomputationEngine {
    /// Create a new precomputation engine
    pub fn new(config: PrecomputationConfig) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(ComputationStats::default())),
            thread_pool: None, // Would initialize actual thread pool
        }
    }

    /// Precompute synthesis parameters for a voice
    pub fn precompute_voice_params(
        &self,
        voice_id: &str,
        voice: &VoiceModel,
    ) -> PrecomputedSynthesisParams {
        let start_time = Instant::now();

        // Simulate precomputation
        let params = PrecomputedSynthesisParams {
            voice_params: self.compute_voice_parameters(voice),
            frequency_coefficients: self.compute_frequency_coefficients(voice),
            dynamic_mappings: self.compute_dynamic_mappings(voice),
            articulation_luts: self.compute_articulation_luts(voice),
            vibrato_tables: self.compute_vibrato_tables(voice),
            breath_templates: self.compute_breath_templates(voice),
        };

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.computations += 1;
            let compute_time = start_time.elapsed().as_millis() as f32;
            stats.avg_compute_time = (stats.avg_compute_time * (stats.computations - 1) as f32
                + compute_time)
                / stats.computations as f32;
        }

        // Cache the result
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(
                format!("voice_params_{voice_id}"),
                PrecomputedData::SynthesisParams(params.clone()),
            );
        }

        params
    }

    /// Compute voice-specific parameters
    fn compute_voice_parameters(&self, _voice: &VoiceModel) -> HashMap<String, f32> {
        // Simplified implementation
        let mut params = HashMap::new();
        params.insert("brightness".to_string(), 0.7);
        params.insert("warmth".to_string(), 0.8);
        params.insert("breathiness".to_string(), 0.3);
        params
    }

    /// Compute frequency-dependent coefficients
    fn compute_frequency_coefficients(&self, _voice: &VoiceModel) -> Vec<f32> {
        // Generate frequency response coefficients
        (0..1024)
            .map(|i| {
                let freq = i as f32 / 1024.0;
                (freq * std::f32::consts::PI).sin() * 0.8 + 0.2
            })
            .collect()
    }

    /// Compute dynamic range mappings
    fn compute_dynamic_mappings(&self, _voice: &VoiceModel) -> Vec<(f32, f32)> {
        // Create input/output mappings for dynamics
        (0..64)
            .map(|i| {
                let input = i as f32 / 64.0;
                let output = input.powf(1.5); // Gentle compression curve
                (input, output)
            })
            .collect()
    }

    /// Compute articulation lookup tables
    fn compute_articulation_luts(&self, _voice: &VoiceModel) -> HashMap<String, Vec<f32>> {
        let mut luts = HashMap::new();

        // Legato transition table
        let legato_lut: Vec<f32> = (0..256)
            .map(|i| {
                let x = i as f32 / 255.0;
                (x * std::f32::consts::PI * 0.5).sin()
            })
            .collect();
        luts.insert("legato".to_string(), legato_lut);

        // Staccato envelope table
        let staccato_lut: Vec<f32> = (0..256)
            .map(|i| {
                let x = i as f32 / 255.0;
                if x < 0.1 {
                    x * 10.0
                } else {
                    ((1.0 - x) / 0.9).max(0.0)
                }
            })
            .collect();
        luts.insert("staccato".to_string(), staccato_lut);

        luts
    }

    /// Compute vibrato wave tables
    fn compute_vibrato_tables(&self, _voice: &VoiceModel) -> Vec<f32> {
        // Generate sine wave table for vibrato
        (0..1024)
            .map(|i| {
                let phase = i as f32 / 1024.0 * 2.0 * std::f32::consts::PI;
                phase.sin()
            })
            .collect()
    }

    /// Compute breath pattern templates
    fn compute_breath_templates(&self, _voice: &VoiceModel) -> Vec<Vec<f32>> {
        vec![
            // Natural breath pattern
            (0..512)
                .map(|i| {
                    let x = i as f32 / 512.0;
                    (x * std::f32::consts::PI).sin().max(0.0) * 0.3
                })
                .collect(),
            // Deep breath pattern
            (0..1024)
                .map(|i| {
                    let x = i as f32 / 1024.0;
                    (x * std::f32::consts::PI * 0.5).sin().max(0.0) * 0.5
                })
                .collect(),
        ]
    }

    /// Get precomputed data
    pub fn get_precomputed(&self, key: &str) -> Option<PrecomputedData> {
        self.cache.read().unwrap().get(key).cloned()
    }

    /// Clear precomputed cache to prevent memory leaks
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
        if let Ok(mut stats) = self.stats.write() {
            stats.memory_usage = 0;
        }
    }

    /// Perform memory leak prevention cleanup
    pub fn cleanup_memory_leaks(&self) {
        if let Ok(mut cache) = self.cache.write() {
            let current_size = cache.len();

            // If cache exceeds memory limit, remove oldest entries
            if current_size * 1024 > self.config.memory_limit {
                let entries_to_remove = current_size / 4; // Remove 25% of entries
                let mut keys_to_remove: Vec<String> =
                    cache.keys().take(entries_to_remove).cloned().collect();

                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
        }

        // Update memory usage statistics
        self.update_memory_stats();
    }

    /// Update memory usage statistics to prevent accounting leaks
    fn update_memory_stats(&self) {
        if let Ok(cache) = self.cache.read() {
            if let Ok(mut stats) = self.stats.write() {
                // Estimate memory usage based on cache size
                stats.memory_usage = cache.len() * 1024; // Rough estimate
            }
        }
    }
}

impl StreamingEngine {
    /// Create a new streaming engine
    pub fn new(config: StreamingConfig) -> Self {
        let buffer_count = config.buffer_count;
        let buffer_manager = Arc::new(Mutex::new(BufferManager::new(&config)));
        Self {
            config,
            state: Arc::new(RwLock::new(StreamState {
                playback_position: 0.0,
                loaded_section: None,
                pending_sections: Vec::new(),
                status: StreamingStatus::Ready,
                buffer_levels: vec![0.0; buffer_count],
            })),
            buffer_manager,
            stats: Arc::new(RwLock::new(StreamingStats::default())),
        }
    }

    /// Start streaming a musical score
    pub fn start_streaming(&self, score: &MusicalScore) -> Result<(), String> {
        // Analyze score and create sections
        let sections = self.create_score_sections(score);

        // Update state
        if let Ok(mut state) = self.state.write() {
            state.pending_sections = sections;
            state.status = StreamingStatus::Buffering;
            state.playback_position = 0.0;
        }

        // Start buffer loading
        self.load_initial_buffers()?;

        Ok(())
    }

    /// Create score sections for streaming
    fn create_score_sections(&self, score: &MusicalScore) -> Vec<ScoreSection> {
        let mut sections = Vec::new();
        let section_duration = self.config.lookahead_time;
        let total_duration = score
            .notes
            .iter()
            .map(|note| note.start_time + note.duration)
            .fold(0.0, f32::max);

        let mut current_time = 0.0_f32;
        while current_time < total_duration {
            let end_time = (current_time + section_duration).min(total_duration);

            // Find notes in this time range
            let section_notes: Vec<MusicalNote> = score
                .notes
                .iter()
                .filter(|note| note.start_time >= current_time && note.start_time < end_time)
                .cloned()
                .collect();

            if !section_notes.is_empty() {
                sections.push(ScoreSection {
                    start_time: current_time as f64,
                    end_time: end_time as f64,
                    notes: section_notes,
                    voice_assignments: HashMap::new(),
                    priority: 1,
                });
            }

            current_time = end_time;
        }

        sections
    }

    /// Load initial buffers
    fn load_initial_buffers(&self) -> Result<(), String> {
        if let Ok(mut buffer_manager) = self.buffer_manager.lock() {
            buffer_manager.prepare_buffers()?;
        }

        if let Ok(mut state) = self.state.write() {
            state.status = StreamingStatus::Ready;
        }

        Ok(())
    }

    /// Update streaming state
    pub fn update(&self, current_time: f64) -> Result<(), String> {
        if let Ok(mut state) = self.state.write() {
            state.playback_position = current_time;

            // Check if we need to load more sections
            self.check_buffer_health(&mut state)?;
        }

        Ok(())
    }

    /// Check buffer health and load more data if needed
    fn check_buffer_health(&self, state: &mut StreamState) -> Result<(), String> {
        let lookahead_time = self.config.lookahead_time as f64;
        let critical_time = state.playback_position + lookahead_time;

        // Check if we have enough buffered content
        let needs_more_data = state
            .pending_sections
            .iter()
            .any(|section| section.start_time <= critical_time);

        if needs_more_data {
            state.status = StreamingStatus::Buffering;
            // Load more sections (implementation would load in background)
        }

        Ok(())
    }

    /// Get streaming statistics
    pub fn get_stats(&self) -> StreamingStats {
        self.stats.read().unwrap().clone()
    }
}

impl BufferManager {
    /// Create a new buffer manager
    pub fn new(config: &StreamingConfig) -> Self {
        Self {
            audio_buffers: Vec::with_capacity(config.buffer_count),
            allocation_strategy: BufferAllocationStrategy::Pooled,
            memory_pool: MemoryPool::new(MemoryPoolConfig {
                initial_size: config.buffer_size * config.buffer_count,
                max_size: config.memory_limit,
                block_size: config.buffer_size,
                growth_strategy: PoolGrowthStrategy::Adaptive,
            }),
            buffer_stats: BufferStats::default(),
        }
    }

    /// Prepare buffers for streaming
    pub fn prepare_buffers(&mut self) -> Result<(), String> {
        // Initialize audio buffers
        for i in 0..self.audio_buffers.capacity() {
            let buffer = AudioBuffer {
                id: format!("buffer_{i}"),
                data: Vec::new(),
                timestamp: 0.0,
                duration: 0.0,
                status: BufferStatus::Empty,
                priority: 0,
            };
            self.audio_buffers.push(buffer);
        }

        self.buffer_stats.buffers_allocated = self.audio_buffers.len() as u32;
        Ok(())
    }

    /// Get next available buffer
    pub fn get_available_buffer(&mut self) -> Option<&mut AudioBuffer> {
        self.audio_buffers
            .iter_mut()
            .find(|buffer| matches!(buffer.status, BufferStatus::Empty | BufferStatus::Played))
    }

    /// Update buffer statistics
    pub fn update_stats(&mut self) {
        let filled_buffers = self
            .audio_buffers
            .iter()
            .filter(|buffer| matches!(buffer.status, BufferStatus::Ready | BufferStatus::Playing))
            .count();

        self.buffer_stats.current_usage = filled_buffers;
        if filled_buffers > self.buffer_stats.peak_usage {
            self.buffer_stats.peak_usage = filled_buffers;
        }
    }
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(config: MemoryPoolConfig) -> Self {
        let mut available_blocks = Vec::new();

        // Pre-allocate initial blocks
        for i in 0..(config.initial_size / config.block_size) {
            available_blocks.push(MemoryBlock {
                id: format!("block_{i}"),
                size: config.block_size,
                data: vec![0u8; config.block_size],
                allocated_at: Instant::now(),
            });
        }

        Self {
            available_blocks,
            allocated_blocks: HashMap::new(),
            config,
        }
    }

    /// Allocate a memory block
    pub fn allocate(&mut self, id: String) -> Option<MemoryBlock> {
        if let Some(block) = self.available_blocks.pop() {
            self.allocated_blocks.insert(id.clone(), block.clone());
            Some(block)
        } else {
            // Try to grow the pool
            self.try_grow_pool()?;
            self.available_blocks.pop()
        }
    }

    /// Deallocate a memory block
    pub fn deallocate(&mut self, id: &str) {
        if let Some(block) = self.allocated_blocks.remove(id) {
            self.available_blocks.push(block);
        }
    }

    /// Try to grow the memory pool
    fn try_grow_pool(&mut self) -> Option<()> {
        match self.config.growth_strategy {
            PoolGrowthStrategy::Fixed => None,
            _ => {
                // Simplified growth - just add one block
                if self.total_size() < self.config.max_size {
                    let new_block = MemoryBlock {
                        id: format!("block_{}", self.total_blocks()),
                        size: self.config.block_size,
                        data: vec![0u8; self.config.block_size],
                        allocated_at: Instant::now(),
                    };
                    self.available_blocks.push(new_block);
                    Some(())
                } else {
                    None
                }
            }
        }
    }

    /// Get total memory pool size
    fn total_size(&self) -> usize {
        (self.available_blocks.len() + self.allocated_blocks.len()) * self.config.block_size
    }

    /// Get total number of blocks
    fn total_blocks(&self) -> usize {
        self.available_blocks.len() + self.allocated_blocks.len()
    }

    /// Cleanup all allocated blocks to prevent memory leaks
    pub fn cleanup(&mut self) {
        // Move all allocated blocks back to available pool for proper cleanup
        let allocated_keys: Vec<String> = self.allocated_blocks.keys().cloned().collect();
        for key in allocated_keys {
            self.deallocate(&key);
        }
    }
}

/// Implement Drop to ensure proper cleanup of memory pool
impl Drop for MemoryPool {
    fn drop(&mut self) {
        // Ensure all allocated blocks are cleaned up
        self.cleanup();

        // Clear all blocks to free memory
        self.available_blocks.clear();
        self.allocated_blocks.clear();
    }
}

impl CompressionEngine {
    /// Create a new compression engine
    pub fn new(config: CompressionConfig) -> Self {
        let mut algorithms: HashMap<CompressionAlgorithm, Box<dyn CompressionCodec>> =
            HashMap::new();

        // Add built-in codecs (simplified for example)
        algorithms.insert(CompressionAlgorithm::None, Box::new(NoCompressionCodec));
        algorithms.insert(CompressionAlgorithm::LZ4, Box::new(LZ4Codec));

        Self {
            config,
            stats: Arc::new(RwLock::new(CompressionStats::default())),
            algorithms,
        }
    }

    /// Compress voice data
    pub fn compress_voice_data(
        &self,
        data: &[u8],
    ) -> Result<(Vec<u8>, CompressionInfo), CompressionError> {
        let start_time = Instant::now();

        let codec = self.algorithms.get(&self.config.algorithm).ok_or(
            CompressionError::UnsupportedAlgorithm(self.config.algorithm.clone()),
        )?;

        let compressed = codec.compress(data)?;
        let compression_time = start_time.elapsed();

        let info = CompressionInfo {
            algorithm: self.config.algorithm.clone(),
            original_size: data.len(),
            compressed_size: compressed.len(),
            compression_ratio: compressed.len() as f32 / data.len() as f32,
            compression_time,
            checksum: self.calculate_checksum(data),
        };

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.compressions += 1;
            stats.bytes_compressed += data.len() as u64;
            let time_ms = compression_time.as_millis() as f32;
            stats.avg_compression_time =
                (stats.avg_compression_time * (stats.compressions - 1) as f32 + time_ms)
                    / stats.compressions as f32;
            stats.avg_compression_ratio = (stats.avg_compression_ratio
                * (stats.compressions - 1) as f32
                + info.compression_ratio)
                / stats.compressions as f32;
        }

        Ok((compressed, info))
    }

    /// Decompress voice data
    pub fn decompress_voice_data(
        &self,
        compressed_data: &[u8],
        info: &CompressionInfo,
    ) -> Result<Vec<u8>, CompressionError> {
        let start_time = Instant::now();

        let codec =
            self.algorithms
                .get(&info.algorithm)
                .ok_or(CompressionError::UnsupportedAlgorithm(
                    info.algorithm.clone(),
                ))?;

        let decompressed = codec.decompress(compressed_data)?;
        let decompression_time = start_time.elapsed();

        // Verify checksum
        if self.calculate_checksum(&decompressed) != info.checksum {
            return Err(CompressionError::InvalidData(
                "Checksum mismatch".to_string(),
            ));
        }

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.decompressions += 1;
            stats.bytes_decompressed += decompressed.len() as u64;
            let time_ms = decompression_time.as_millis() as f32;
            stats.avg_decompression_time =
                (stats.avg_decompression_time * (stats.decompressions - 1) as f32 + time_ms)
                    / stats.decompressions as f32;
        }

        Ok(decompressed)
    }

    /// Calculate simple checksum
    fn calculate_checksum(&self, data: &[u8]) -> u32 {
        data.iter()
            .fold(0u32, |acc, &byte| acc.wrapping_add(byte as u32))
    }

    /// Get compression statistics
    pub fn get_stats(&self) -> CompressionStats {
        self.stats.read().unwrap().clone()
    }
}

// Simple codec implementations for demonstration

/// No compression codec
struct NoCompressionCodec;

impl CompressionCodec for NoCompressionCodec {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        Ok(data.to_vec())
    }

    fn decompress(&self, compressed_data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        Ok(compressed_data.to_vec())
    }

    fn estimate_ratio(&self, _data: &[u8]) -> f32 {
        1.0
    }

    fn info(&self) -> CompressionCodecInfo {
        CompressionCodecInfo {
            name: "None".to_string(),
            speed: CompressionSpeed::VeryFast,
            typical_ratio: 1.0,
            lossless: true,
        }
    }
}

/// Simple LZ4-style codec (simplified implementation)
struct LZ4Codec;

impl CompressionCodec for LZ4Codec {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        // Simplified compression (just demonstrate interface)
        // Real implementation would use actual LZ4 algorithm
        let mut compressed = Vec::new();
        compressed.extend_from_slice(&(data.len() as u32).to_le_bytes());

        // Simple run-length encoding as placeholder
        let mut i = 0;
        while i < data.len() {
            let byte = data[i];
            let mut count = 1;

            while i + count < data.len() && data[i + count] == byte && count < 255 {
                count += 1;
            }

            compressed.push(count as u8);
            compressed.push(byte);
            i += count;
        }

        Ok(compressed)
    }

    fn decompress(&self, compressed_data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if compressed_data.len() < 4 {
            return Err(CompressionError::InvalidData("Too short".to_string()));
        }

        let original_size = u32::from_le_bytes([
            compressed_data[0],
            compressed_data[1],
            compressed_data[2],
            compressed_data[3],
        ]) as usize;

        let mut decompressed = Vec::with_capacity(original_size);
        let mut i = 4;

        while i + 1 < compressed_data.len() {
            let count = compressed_data[i] as usize;
            let byte = compressed_data[i + 1];

            decompressed.extend(vec![byte; count]);
            i += 2;
        }

        Ok(decompressed)
    }

    fn estimate_ratio(&self, _data: &[u8]) -> f32 {
        0.7 // Estimated 30% compression
    }

    fn info(&self) -> CompressionCodecInfo {
        CompressionCodecInfo {
            name: "LZ4".to_string(),
            speed: CompressionSpeed::Fast,
            typical_ratio: 0.7,
            lossless: true,
        }
    }
}

// Default implementations
impl Default for VoiceCacheConfig {
    fn default() -> Self {
        Self {
            max_voices: 32,
            max_memory: 512 * 1024 * 1024, // 512 MB
            eviction_policy: EvictionPolicy::LRU,
            preload_popular: true,
            background_preload: true,
            compression: CompressionConfig::default(),
            persistent_cache: false,
            cache_directory: None,
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::LZ4,
            level: 6,
            preserve_quality: true,
            allow_lossy: false,
            target_ratio: 0.5,
            adaptive: true,
        }
    }
}

impl Default for PrecomputationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategies: vec![
                PrecomputationStrategy::VoiceLoad,
                PrecomputationStrategy::IdleTime,
            ],
            background_compute: true,
            memory_limit: 256 * 1024 * 1024, // 256 MB
            quality_level: ComputationQuality::Balanced,
            persistent_cache: true,
            adaptive: true,
        }
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 4096,
            buffer_count: 8,
            lookahead_time: 2.0,
            quality: StreamingQuality::Balanced,
            adaptive_buffering: true,
            background_loading: true,
            memory_limit: 64 * 1024 * 1024, // 64 MB
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_cache_creation() {
        let config = VoiceCacheConfig::default();
        let cache = VoiceCache::new(config);

        let stats = cache.get_stats();
        assert_eq!(stats.requests, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_voice_cache_miss() {
        let config = VoiceCacheConfig::default();
        let cache = VoiceCache::new(config);

        let result = cache.get_voice("nonexistent_voice");
        assert!(result.is_none());

        let stats = cache.get_stats();
        assert_eq!(stats.requests, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_precomputation_engine() {
        let config = PrecomputationConfig::default();
        let engine = PrecomputationEngine::new(config);

        let voice = VoiceModel::new("test".to_string(), VoiceCharacteristics::default());
        let params = engine.precompute_voice_params("test_voice", &voice);

        assert!(!params.voice_params.is_empty());
        assert!(!params.frequency_coefficients.is_empty());
    }

    #[test]
    fn test_streaming_engine_creation() {
        let config = StreamingConfig::default();
        let engine = StreamingEngine::new(config);

        let stats = engine.get_stats();
        assert_eq!(stats.interruptions, 0);
        assert_eq!(stats.data_streamed, 0);
    }

    #[test]
    fn test_compression_engine() {
        let config = CompressionConfig::default();
        let engine = CompressionEngine::new(config);

        let test_data = b"Hello, this is test data for compression!";
        let result = engine.compress_voice_data(test_data);

        assert!(result.is_ok());
        let (compressed, info) = result.unwrap();
        assert!(compressed.len() > 0);
        assert_eq!(info.original_size, test_data.len());
    }

    #[test]
    fn test_compression_roundtrip() {
        let config = CompressionConfig::default();
        let engine = CompressionEngine::new(config);

        let original_data = b"This is test data for compression roundtrip test!";

        let (compressed, info) = engine.compress_voice_data(original_data).unwrap();
        let decompressed = engine.decompress_voice_data(&compressed, &info).unwrap();

        assert_eq!(original_data.to_vec(), decompressed);
    }

    #[test]
    fn test_memory_pool() {
        let config = MemoryPoolConfig {
            initial_size: 1024,
            max_size: 2048,
            block_size: 256,
            growth_strategy: PoolGrowthStrategy::Linear,
        };

        let mut pool = MemoryPool::new(config);

        let block1 = pool.allocate("test_block_1".to_string());
        assert!(block1.is_some());

        let block2 = pool.allocate("test_block_2".to_string());
        assert!(block2.is_some());

        pool.deallocate("test_block_1");

        let block3 = pool.allocate("test_block_3".to_string());
        assert!(block3.is_some());
    }

    #[test]
    fn test_buffer_manager() {
        let config = StreamingConfig::default();
        let mut buffer_manager = BufferManager::new(&config);

        assert!(buffer_manager.prepare_buffers().is_ok());

        let available = buffer_manager.get_available_buffer();
        assert!(available.is_some());

        if let Some(buffer) = available {
            assert_eq!(buffer.status, BufferStatus::Empty);
        }
    }

    #[test]
    fn test_voice_preloader() {
        let mut preloader = VoicePreloader::new();

        preloader.queue_voice("voice1".to_string(), 0.8);
        preloader.queue_voice("voice2".to_string(), 0.9);
        preloader.queue_voice("voice3".to_string(), 0.7);

        // Should return voice2 first (highest priority)
        let next = preloader.next_voice();
        assert_eq!(next, Some("voice2".to_string()));

        // Then voice1
        let next = preloader.next_voice();
        assert_eq!(next, Some("voice1".to_string()));
    }
}
