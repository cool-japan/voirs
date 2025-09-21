//! Performance and Memory Optimizations for VoiRS Conversion
//!
//! This module provides optimizations for memory usage and performance,
//! particularly for handling small audio samples and reducing memory overhead.

use crate::prelude::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, RwLock,
};
use std::thread;
use std::time::{Duration, Instant};

/// Memory-efficient audio buffer pool to reduce allocations
#[derive(Debug)]
pub struct AudioBufferPool {
    small_buffers: Arc<Mutex<Vec<Vec<f32>>>>, // For audio < 1 second
    medium_buffers: Arc<Mutex<Vec<Vec<f32>>>>, // For audio 1-5 seconds
    large_buffers: Arc<Mutex<Vec<Vec<f32>>>>, // For audio > 5 seconds
    stats: Arc<Mutex<PoolStats>>,
}

#[derive(Debug, Default, Clone)]
pub struct PoolStats {
    small_hits: usize,
    small_misses: usize,
    medium_hits: usize,
    medium_misses: usize,
    large_hits: usize,
    large_misses: usize,
}

impl Default for AudioBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioBufferPool {
    pub fn new() -> Self {
        Self {
            small_buffers: Arc::new(Mutex::new(Vec::with_capacity(10))),
            medium_buffers: Arc::new(Mutex::new(Vec::with_capacity(5))),
            large_buffers: Arc::new(Mutex::new(Vec::with_capacity(2))),
            stats: Arc::new(Mutex::new(PoolStats::default())),
        }
    }

    /// Get a buffer from the pool, creating one if necessary
    pub fn get_buffer(&self, min_size: usize) -> Vec<f32> {
        let category = self.categorize_size(min_size);

        match category {
            BufferCategory::Small => {
                let mut buffers = self.small_buffers.lock().unwrap();
                if let Some(mut buffer) = buffers.pop() {
                    if buffer.capacity() >= min_size {
                        buffer.clear();
                        buffer.resize(min_size, 0.0);
                        self.stats.lock().unwrap().small_hits += 1;
                        return buffer;
                    } else {
                        // Buffer too small, return it and create new one
                        buffers.push(buffer);
                    }
                }
                self.stats.lock().unwrap().small_misses += 1;
                let mut buffer = Vec::with_capacity(std::cmp::max(min_size, 22050)); // 1 second at 22kHz
                buffer.resize(min_size, 0.0);
                buffer
            }
            BufferCategory::Medium => {
                let mut buffers = self.medium_buffers.lock().unwrap();
                if let Some(mut buffer) = buffers.pop() {
                    if buffer.capacity() >= min_size {
                        buffer.clear();
                        buffer.resize(min_size, 0.0);
                        self.stats.lock().unwrap().medium_hits += 1;
                        return buffer;
                    } else {
                        buffers.push(buffer);
                    }
                }
                self.stats.lock().unwrap().medium_misses += 1;
                Vec::with_capacity(std::cmp::max(min_size, 110250)) // 5 seconds at 22kHz
            }
            BufferCategory::Large => {
                let mut buffers = self.large_buffers.lock().unwrap();
                if let Some(mut buffer) = buffers.pop() {
                    if buffer.capacity() >= min_size {
                        buffer.clear();
                        buffer.resize(min_size, 0.0);
                        self.stats.lock().unwrap().large_hits += 1;
                        return buffer;
                    } else {
                        buffers.push(buffer);
                    }
                }
                self.stats.lock().unwrap().large_misses += 1;
                Vec::with_capacity(min_size)
            }
        }
    }

    /// Return a buffer to the pool for reuse
    pub fn return_buffer(&self, buffer: Vec<f32>) {
        let category = self.categorize_size(buffer.capacity());

        match category {
            BufferCategory::Small => {
                let mut buffers = self.small_buffers.lock().unwrap();
                if buffers.len() < 10 {
                    // Limit pool size
                    buffers.push(buffer);
                }
            }
            BufferCategory::Medium => {
                let mut buffers = self.medium_buffers.lock().unwrap();
                if buffers.len() < 5 {
                    buffers.push(buffer);
                }
            }
            BufferCategory::Large => {
                let mut buffers = self.large_buffers.lock().unwrap();
                if buffers.len() < 2 {
                    buffers.push(buffer);
                }
            }
        }
    }

    fn categorize_size(&self, size: usize) -> BufferCategory {
        if size <= 22050 {
            // <= 1 second at 22kHz
            BufferCategory::Small
        } else if size <= 110250 {
            // <= 5 seconds at 22kHz
            BufferCategory::Medium
        } else {
            BufferCategory::Large
        }
    }

    pub fn get_stats(&self) -> PoolStats {
        self.stats.lock().unwrap().clone()
    }
}

impl PoolStats {
    fn hit_rate(&self) -> f64 {
        let total_hits = self.small_hits + self.medium_hits + self.large_hits;
        let total_requests =
            total_hits + self.small_misses + self.medium_misses + self.large_misses;

        if total_requests == 0 {
            0.0
        } else {
            total_hits as f64 / total_requests as f64
        }
    }
}

#[derive(Debug)]
enum BufferCategory {
    Small,
    Medium,
    Large,
}

/// Fast-path optimization for small audio samples
#[derive(Debug)]
pub struct SmallAudioOptimizer {
    buffer_pool: AudioBufferPool,
    small_sample_cache: Arc<Mutex<HashMap<String, CachedResult>>>,
}

#[derive(Debug, Clone)]
struct CachedResult {
    result: Vec<f32>,
    timestamp: Instant,
    access_count: usize,
}

impl Default for SmallAudioOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SmallAudioOptimizer {
    pub fn new() -> Self {
        Self {
            buffer_pool: AudioBufferPool::new(),
            small_sample_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Optimize conversion for very small audio samples (< 0.5 seconds)
    pub fn optimize_small_conversion(
        &self,
        audio: &[f32],
        conversion_type: &ConversionType,
        target: &ConversionTarget,
    ) -> Option<Vec<f32>> {
        // Only optimize for very small samples to reduce memory overhead
        if audio.len() > 11025 {
            // 0.5 seconds at 22kHz
            return None;
        }

        // Create cache key
        let cache_key = self.create_cache_key(audio, conversion_type, target);

        // Check cache first
        if let Some(cached) = self.get_cached_result(&cache_key) {
            return Some(cached);
        }

        // For very small audio, use simplified processing
        let optimized_result = match conversion_type {
            ConversionType::PitchShift => self.fast_pitch_shift(audio, target),
            ConversionType::SpeedTransformation => self.fast_speed_transform(audio, target),
            ConversionType::PassThrough => {
                // Ultra-fast passthrough - just copy
                Some(audio.to_vec())
            }
            _ => None, // Use full processing for complex conversions
        };

        // Cache result if successful
        if let Some(ref result) = optimized_result {
            self.cache_result(cache_key, result.clone());
        }

        optimized_result
    }

    fn create_cache_key(
        &self,
        audio: &[f32],
        conversion_type: &ConversionType,
        target: &ConversionTarget,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash audio characteristics instead of full audio to reduce memory
        let audio_hash = if audio.len() <= 100 {
            // For very small audio, hash the whole thing
            audio
                .iter()
                .fold(0u64, |acc, &x| acc.wrapping_add((x * 1000.0) as u64))
        } else {
            // For larger small audio, hash samples at regular intervals
            audio
                .iter()
                .step_by(audio.len() / 20)
                .fold(0u64, |acc, &x| acc.wrapping_add((x * 1000.0) as u64))
        };

        audio_hash.hash(&mut hasher);
        conversion_type.hash(&mut hasher);

        // Hash key target characteristics
        match conversion_type {
            ConversionType::PitchShift => {
                (target.characteristics.pitch.mean_f0 as u32).hash(&mut hasher);
            }
            ConversionType::SpeedTransformation => {
                ((target.characteristics.timing.speaking_rate * 1000.0) as u32).hash(&mut hasher);
            }
            _ => {}
        }

        format!("small_audio_{}", hasher.finish())
    }

    fn get_cached_result(&self, key: &str) -> Option<Vec<f32>> {
        let mut cache = self.small_sample_cache.lock().unwrap();

        if let Some(cached) = cache.get_mut(key) {
            // Check if cache entry is still valid (5 minutes)
            if cached.timestamp.elapsed() < Duration::from_secs(300) {
                cached.access_count += 1;
                return Some(cached.result.clone());
            } else {
                // Remove expired entry
                cache.remove(key);
            }
        }
        None
    }

    fn cache_result(&self, key: String, result: Vec<f32>) {
        let mut cache = self.small_sample_cache.lock().unwrap();

        // Limit cache size to prevent memory bloat
        if cache.len() >= 100 {
            // Remove least recently used entries
            let mut entries: Vec<_> = cache
                .iter()
                .map(|(k, v)| (k.clone(), v.timestamp))
                .collect();
            entries.sort_by_key(|(_, timestamp)| *timestamp);

            // Remove oldest 10 entries
            for (k, _) in entries.iter().take(10) {
                cache.remove(k);
            }
        }

        cache.insert(
            key,
            CachedResult {
                result,
                timestamp: Instant::now(),
                access_count: 1,
            },
        );
    }

    fn fast_pitch_shift(&self, audio: &[f32], target: &ConversionTarget) -> Option<Vec<f32>> {
        let target_f0 = target.characteristics.pitch.mean_f0;
        let base_f0 = 220.0; // Assume base frequency
        let ratio = target_f0 / base_f0;

        // Simple pitch shifting for small samples - linear interpolation
        let output_len = audio.len();
        let mut result = self.buffer_pool.get_buffer(output_len);

        for (i, result_item) in result.iter_mut().enumerate().take(output_len) {
            let src_index = (i as f32 * ratio) as usize;
            if src_index < audio.len() {
                *result_item = audio[src_index];
            } else {
                *result_item = 0.0;
            }
        }

        Some(result)
    }

    fn fast_speed_transform(&self, audio: &[f32], target: &ConversionTarget) -> Option<Vec<f32>> {
        let speed_ratio = target.characteristics.timing.speaking_rate;
        let output_len = (audio.len() as f32 / speed_ratio) as usize;

        let mut result = self.buffer_pool.get_buffer(output_len);

        for (i, result_item) in result.iter_mut().enumerate().take(output_len) {
            let src_index = (i as f32 * speed_ratio) as usize;
            if src_index < audio.len() {
                *result_item = audio[src_index];
            } else {
                *result_item = 0.0;
            }
        }

        Some(result)
    }

    /// Get buffer pool statistics
    pub fn get_pool_stats(&self) -> PoolStats {
        self.buffer_pool.get_stats()
    }

    /// Clear cache to free memory
    pub fn clear_cache(&self) {
        self.small_sample_cache.lock().unwrap().clear();
    }
}

/// Performance monitor for conversion operations
#[derive(Debug)]
pub struct ConversionPerformanceMonitor {
    conversion_times: Arc<Mutex<Vec<Duration>>>,
    memory_usage: Arc<Mutex<Vec<usize>>>,
    start_time: Option<Instant>,
}

impl Default for ConversionPerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversionPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            conversion_times: Arc::new(Mutex::new(Vec::new())),
            memory_usage: Arc::new(Mutex::new(Vec::new())),
            start_time: None,
        }
    }

    pub fn start_timing(&mut self) {
        self.start_time = Some(Instant::now());
    }

    pub fn end_timing(&mut self) {
        if let Some(start) = self.start_time.take() {
            let duration = start.elapsed();
            self.conversion_times.lock().unwrap().push(duration);
        }
    }

    pub fn record_memory_usage(&self, bytes: usize) {
        self.memory_usage.lock().unwrap().push(bytes);
    }

    pub fn get_average_conversion_time(&self) -> Duration {
        let times = self.conversion_times.lock().unwrap();
        if times.is_empty() {
            Duration::from_millis(0)
        } else {
            let total_nanos: u64 = times.iter().map(|d| d.as_nanos() as u64).sum();
            Duration::from_nanos(total_nanos / times.len() as u64)
        }
    }

    pub fn get_memory_stats(&self) -> (usize, usize, f64) {
        let usage = self.memory_usage.lock().unwrap();
        if usage.is_empty() {
            (0, 0, 0.0)
        } else {
            let min = *usage.iter().min().unwrap();
            let max = *usage.iter().max().unwrap();
            let avg = usage.iter().sum::<usize>() as f64 / usage.len() as f64;
            (min, max, avg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_buffer_pool() {
        let pool = AudioBufferPool::new();

        // Test small buffer
        let buffer1 = pool.get_buffer(1000);
        assert!(buffer1.capacity() >= 1000);

        pool.return_buffer(buffer1);

        // Should reuse the buffer
        let buffer2 = pool.get_buffer(500);
        assert!(buffer2.capacity() >= 500);

        let stats = pool.get_stats();
        assert!(stats.hit_rate() > 0.0);
    }

    #[test]
    fn test_small_audio_optimizer() {
        let optimizer = SmallAudioOptimizer::new();

        // Create small test audio
        let audio: Vec<f32> = (0..1000)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 22050.0).sin() * 0.1)
            .collect();

        let mut target_chars = VoiceCharacteristics::default();
        target_chars.pitch.mean_f0 = 880.0;
        let target = ConversionTarget::new(target_chars);

        // Test optimization
        let result =
            optimizer.optimize_small_conversion(&audio, &ConversionType::PitchShift, &target);
        assert!(result.is_some());

        // Test caching - second call should be faster
        let result2 =
            optimizer.optimize_small_conversion(&audio, &ConversionType::PitchShift, &target);
        assert!(result2.is_some());
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = ConversionPerformanceMonitor::new();

        monitor.start_timing();
        std::thread::sleep(Duration::from_millis(10));
        monitor.end_timing();

        monitor.record_memory_usage(1024);
        monitor.record_memory_usage(2048);

        let avg_time = monitor.get_average_conversion_time();
        assert!(avg_time.as_millis() >= 10);

        let (min_mem, max_mem, avg_mem) = monitor.get_memory_stats();
        assert_eq!(min_mem, 1024);
        assert_eq!(max_mem, 2048);
        assert_eq!(avg_mem, 1536.0);
    }
}
