//! Memory optimization utilities for VoiRS Recognizer
//!
//! This module provides comprehensive memory optimization functionality including:
//! - Memory pool management for frequent allocations
//! - Smart buffer reuse and recycling
//! - Memory pressure detection and management
//! - Zero-copy operations where possible
//! - Memory-efficient data structures

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Memory pool for efficient buffer reuse
pub struct MemoryPool<T> {
    /// Available buffers organized by size
    available: HashMap<usize, Vec<T>>,
    /// Maximum number of buffers to keep per size
    max_buffers_per_size: usize,
    /// Statistics for monitoring
    stats: MemoryPoolStats,
    /// Creation function for new buffers
    create_fn: Box<dyn Fn(usize) -> T + Send + Sync>,
}

/// Memory pool statistics
#[derive(Debug, Default, Clone)]
/// Memory Pool Stats
pub struct MemoryPoolStats {
    /// Total allocations requested
    pub total_allocations: usize,
    /// Cache hits (reused buffers)
    pub cache_hits: usize,
    /// Cache misses (new allocations)
    pub cache_misses: usize,
    /// Current number of pooled buffers
    pub pooled_buffers: usize,
    /// Peak number of pooled buffers
    pub peak_pooled_buffers: usize,
    /// Memory saved through reuse (estimated bytes)
    pub memory_saved_bytes: usize,
}

/// Audio buffer pool specialized for f32 audio data
pub type AudioBufferPool = MemoryPool<Vec<f32>>;

/// Feature buffer pool for ML features
pub type FeatureBufferPool = MemoryPool<Vec<f32>>;

/// Memory pressure monitor
pub struct MemoryPressureMonitor {
    /// Memory pressure thresholds
    thresholds: MemoryThresholds,
    /// Current memory usage tracking
    current_usage: Arc<Mutex<MemoryUsage>>,
    /// Pressure callbacks
    callbacks: Vec<Box<dyn Fn(MemoryPressureLevel) + Send + Sync>>,
    /// Last check time
    last_check: Instant,
    /// Check interval
    check_interval: Duration,
}

/// Memory pressure levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
/// Memory Pressure Level
pub enum MemoryPressureLevel {
    /// Normal memory usage
    Normal,
    /// Moderate pressure - start releasing non-essential caches
    Moderate,
    /// High pressure - aggressively free memory
    High,
    /// Critical pressure - emergency cleanup
    Critical,
}

/// Memory thresholds configuration
#[derive(Debug, Clone)]
/// Memory Thresholds
pub struct MemoryThresholds {
    /// Moderate pressure threshold (percentage of total memory)
    pub moderate_percent: f32,
    /// High pressure threshold (percentage of total memory)
    pub high_percent: f32,
    /// Critical pressure threshold (percentage of total memory)
    pub critical_percent: f32,
    /// Memory check interval
    pub check_interval_seconds: u64,
}

/// Current memory usage information
#[derive(Debug, Clone)]
/// Memory Usage
pub struct MemoryUsage {
    /// Total system memory in bytes
    pub total_memory_bytes: u64,
    /// Available memory in bytes
    pub available_memory_bytes: u64,
    /// Used memory in bytes
    pub used_memory_bytes: u64,
    /// Process RSS memory in bytes
    pub process_rss_bytes: u64,
    /// Last update timestamp
    pub last_updated: Instant,
}

impl Default for MemoryUsage {
    fn default() -> Self {
        Self {
            total_memory_bytes: 0,
            available_memory_bytes: 0,
            used_memory_bytes: 0,
            process_rss_bytes: 0,
            last_updated: Instant::now(),
        }
    }
}

/// Memory-efficient audio chunk iterator
pub struct AudioChunkIterator<'a> {
    /// Audio data reference
    data: &'a [f32],
    /// Chunk size
    chunk_size: usize,
    /// Current position
    position: usize,
    /// Overlap size between chunks
    overlap: usize,
}

/// Memory-efficient circular buffer for streaming audio
#[derive(Debug)]
/// Circular Audio Buffer
pub struct CircularAudioBuffer {
    /// Internal buffer
    buffer: Vec<f32>,
    /// Write position
    write_pos: usize,
    /// Read position
    read_pos: usize,
    /// Number of valid samples
    valid_samples: usize,
    /// Buffer capacity
    capacity: usize,
}

/// Zero-copy audio slice wrapper
#[derive(Debug)]
/// Audio Slice
pub struct AudioSlice<'a> {
    /// Audio data reference
    data: &'a [f32],
    /// Sample rate
    sample_rate: u32,
    /// Channel count
    channels: u32,
}

impl<T> MemoryPool<T> {
    /// Create a new memory pool
    pub fn new<F>(create_fn: F) -> Self
    where
        F: Fn(usize) -> T + Send + Sync + 'static,
    {
        Self {
            available: HashMap::new(),
            max_buffers_per_size: 10,
            stats: MemoryPoolStats::default(),
            create_fn: Box::new(create_fn),
        }
    }

    /// Acquire a buffer of the specified size
    pub fn acquire(&mut self, size: usize) -> T {
        self.stats.total_allocations += 1;

        if let Some(buffers) = self.available.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                self.stats.cache_hits += 1;
                self.stats.pooled_buffers -= 1;
                return buffer;
            }
        }

        self.stats.cache_misses += 1;
        (self.create_fn)(size)
    }

    /// Return a buffer to the pool
    pub fn release(&mut self, size: usize, buffer: T) {
        let buffers = self.available.entry(size).or_insert_with(Vec::new);

        if buffers.len() < self.max_buffers_per_size {
            buffers.push(buffer);
            self.stats.pooled_buffers += 1;
            self.stats.peak_pooled_buffers = self
                .stats
                .peak_pooled_buffers
                .max(self.stats.pooled_buffers);

            // Estimate memory saved
            self.stats.memory_saved_bytes += size * std::mem::size_of::<T>();
        }
        // If pool is full, just drop the buffer
    }

    /// Get pool statistics
    pub fn stats(&self) -> &MemoryPoolStats {
        &self.stats
    }

    /// Clear the pool and free all cached buffers
    pub fn clear(&mut self) {
        self.available.clear();
        self.stats.pooled_buffers = 0;
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f32 {
        if self.stats.total_allocations > 0 {
            self.stats.cache_hits as f32 / self.stats.total_allocations as f32
        } else {
            0.0
        }
    }
}

impl AudioBufferPool {
    /// Create a new audio buffer pool  
    pub fn new_audio_pool() -> Self {
        MemoryPool::new(|size| Vec::with_capacity(size))
    }

    /// Acquire an audio buffer with the specified capacity
    pub fn acquire_audio_buffer(&mut self, capacity: usize) -> Vec<f32> {
        let mut buffer = self.acquire(capacity);
        buffer.clear(); // Ensure it's empty
        buffer
    }

    /// Release an audio buffer back to the pool
    pub fn release_audio_buffer(&mut self, mut buffer: Vec<f32>) {
        let capacity = buffer.capacity();
        buffer.clear(); // Clear data but keep capacity
        self.release(capacity, buffer);
    }
}

impl MemoryPressureMonitor {
    /// Create a new memory pressure monitor
    pub fn new(thresholds: MemoryThresholds) -> Self {
        let check_interval = Duration::from_secs(thresholds.check_interval_seconds);
        Self {
            thresholds,
            current_usage: Arc::new(Mutex::new(MemoryUsage::default())),
            callbacks: Vec::new(),
            last_check: Instant::now(),
            check_interval,
        }
    }

    /// Add a callback for memory pressure events
    pub fn add_callback<F>(&mut self, callback: F)
    where
        F: Fn(MemoryPressureLevel) + Send + Sync + 'static,
    {
        self.callbacks.push(Box::new(callback));
    }

    /// Update memory usage and check pressure
    pub fn check_pressure(&mut self) -> MemoryPressureLevel {
        let now = Instant::now();
        if now.duration_since(self.last_check) < self.check_interval {
            return self.get_current_pressure_level();
        }

        self.last_check = now;

        // Update memory usage
        let usage = self.get_system_memory_usage();
        let pressure_level = self.calculate_pressure_level(&usage);
        {
            let mut current = self.current_usage.lock().unwrap();
            *current = usage;
        }

        // Notify callbacks if pressure changed
        if pressure_level != MemoryPressureLevel::Normal {
            for callback in &self.callbacks {
                callback(pressure_level.clone());
            }
        }

        pressure_level
    }

    /// Get current pressure level without updating
    pub fn get_current_pressure_level(&self) -> MemoryPressureLevel {
        let usage = self.current_usage.lock().unwrap();
        self.calculate_pressure_level(&usage)
    }

    /// Calculate pressure level from memory usage
    fn calculate_pressure_level(&self, usage: &MemoryUsage) -> MemoryPressureLevel {
        if usage.total_memory_bytes == 0 {
            return MemoryPressureLevel::Normal;
        }

        let usage_percent =
            (usage.used_memory_bytes as f32 / usage.total_memory_bytes as f32) * 100.0;

        if usage_percent >= self.thresholds.critical_percent {
            MemoryPressureLevel::Critical
        } else if usage_percent >= self.thresholds.high_percent {
            MemoryPressureLevel::High
        } else if usage_percent >= self.thresholds.moderate_percent {
            MemoryPressureLevel::Moderate
        } else {
            MemoryPressureLevel::Normal
        }
    }

    /// Get system memory usage (platform-specific implementation)
    fn get_system_memory_usage(&self) -> MemoryUsage {
        #[cfg(target_os = "linux")]
        {
            self.get_linux_memory_usage()
        }
        #[cfg(target_os = "macos")]
        {
            self.get_macos_memory_usage()
        }
        #[cfg(target_os = "windows")]
        {
            self.get_windows_memory_usage()
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            // Fallback for other platforms
            MemoryUsage {
                total_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB default
                available_memory_bytes: 4 * 1024 * 1024 * 1024, // 4GB default
                used_memory_bytes: 4 * 1024 * 1024 * 1024,
                process_rss_bytes: 512 * 1024 * 1024, // 512MB default
                last_updated: Instant::now(),
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn get_linux_memory_usage(&self) -> MemoryUsage {
        // Read /proc/meminfo for system memory
        let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let mut total_kb = 0;
        let mut available_kb = 0;

        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                total_kb = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
            } else if line.starts_with("MemAvailable:") {
                available_kb = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
            }
        }

        // Read /proc/self/status for process memory
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let mut rss_kb = 0;

        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                rss_kb = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
                break;
            }
        }

        let total_bytes = total_kb * 1024;
        let available_bytes = available_kb * 1024;
        let used_bytes = total_bytes - available_bytes;
        let rss_bytes = rss_kb * 1024;

        MemoryUsage {
            total_memory_bytes: total_bytes,
            available_memory_bytes: available_bytes,
            used_memory_bytes: used_bytes,
            process_rss_bytes: rss_bytes,
            last_updated: Instant::now(),
        }
    }

    #[cfg(target_os = "macos")]
    fn get_macos_memory_usage(&self) -> MemoryUsage {
        // For macOS, we would use system calls or estimate
        // This is a simplified implementation
        MemoryUsage {
            total_memory_bytes: 16 * 1024 * 1024 * 1024, // 16GB estimate
            available_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB estimate
            used_memory_bytes: 8 * 1024 * 1024 * 1024,
            process_rss_bytes: 512 * 1024 * 1024, // 512MB estimate
            last_updated: Instant::now(),
        }
    }

    #[cfg(target_os = "windows")]
    fn get_windows_memory_usage(&self) -> MemoryUsage {
        // For Windows, we would use Windows API or estimate
        // This is a simplified implementation
        MemoryUsage {
            total_memory_bytes: 16 * 1024 * 1024 * 1024, // 16GB estimate
            available_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB estimate
            used_memory_bytes: 8 * 1024 * 1024 * 1024,
            process_rss_bytes: 512 * 1024 * 1024, // 512MB estimate
            last_updated: Instant::now(),
        }
    }
}

impl<'a> AudioChunkIterator<'a> {
    /// Create a new audio chunk iterator
    pub fn new(data: &'a [f32], chunk_size: usize, overlap: usize) -> Self {
        Self {
            data,
            chunk_size,
            position: 0,
            overlap,
        }
    }

    /// Create iterator without overlap
    pub fn without_overlap(data: &'a [f32], chunk_size: usize) -> Self {
        Self::new(data, chunk_size, 0)
    }
}

impl<'a> Iterator for AudioChunkIterator<'a> {
    type Item = &'a [f32];

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.data.len() {
            return None;
        }

        let end = (self.position + self.chunk_size).min(self.data.len());
        let chunk = &self.data[self.position..end];

        // Move position forward, accounting for overlap
        self.position += self.chunk_size - self.overlap;

        Some(chunk)
    }
}

impl CircularAudioBuffer {
    /// Create a new circular buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity],
            write_pos: 0,
            read_pos: 0,
            valid_samples: 0,
            capacity,
        }
    }

    /// Write samples to the buffer
    pub fn write(&mut self, samples: &[f32]) -> usize {
        let mut written = 0;

        for &sample in samples {
            if self.valid_samples < self.capacity {
                self.buffer[self.write_pos] = sample;
                self.write_pos = (self.write_pos + 1) % self.capacity;
                self.valid_samples += 1;
                written += 1;
            } else {
                // Buffer is full, overwrite oldest data
                self.buffer[self.write_pos] = sample;
                self.write_pos = (self.write_pos + 1) % self.capacity;
                self.read_pos = (self.read_pos + 1) % self.capacity;
                written += 1;
            }
        }

        written
    }

    /// Read samples from the buffer
    pub fn read(&mut self, output: &mut [f32]) -> usize {
        let mut read_count = 0;

        for i in 0..output.len() {
            if self.valid_samples > 0 {
                output[i] = self.buffer[self.read_pos];
                self.read_pos = (self.read_pos + 1) % self.capacity;
                self.valid_samples -= 1;
                read_count += 1;
            } else {
                break;
            }
        }

        read_count
    }

    /// Peek at samples without consuming them
    pub fn peek(&self, output: &mut [f32]) -> usize {
        let mut peek_count = 0;
        let mut peek_pos = self.read_pos;
        let mut remaining_samples = self.valid_samples;

        for i in 0..output.len() {
            if remaining_samples > 0 {
                output[i] = self.buffer[peek_pos];
                peek_pos = (peek_pos + 1) % self.capacity;
                remaining_samples -= 1;
                peek_count += 1;
            } else {
                break;
            }
        }

        peek_count
    }

    /// Get number of available samples
    pub fn available(&self) -> usize {
        self.valid_samples
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.valid_samples == self.capacity
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.valid_samples == 0
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.read_pos = 0;
        self.write_pos = 0;
        self.valid_samples = 0;
    }
}

impl<'a> AudioSlice<'a> {
    /// Create a new audio slice
    pub fn new(data: &'a [f32], sample_rate: u32, channels: u32) -> Self {
        Self {
            data,
            sample_rate,
            channels,
        }
    }

    /// Get the audio data
    pub fn data(&self) -> &[f32] {
        self.data
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get channel count
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        let samples_per_channel = self.data.len() / self.channels as usize;
        samples_per_channel as f32 / self.sample_rate as f32
    }

    /// Create an iterator over audio chunks
    pub fn chunks(&self, chunk_size: usize) -> AudioChunkIterator<'a> {
        AudioChunkIterator::without_overlap(self.data, chunk_size)
    }

    /// Create an iterator with overlap
    pub fn overlapping_chunks(&self, chunk_size: usize, overlap: usize) -> AudioChunkIterator<'a> {
        AudioChunkIterator::new(self.data, chunk_size, overlap)
    }
}

impl Default for MemoryThresholds {
    fn default() -> Self {
        Self {
            moderate_percent: 70.0,
            high_percent: 85.0,
            critical_percent: 95.0,
            check_interval_seconds: 5,
        }
    }
}

/// Memory optimization utilities
pub struct MemoryOptimizer {
    /// Audio buffer pool
    audio_pool: Arc<Mutex<AudioBufferPool>>,
    /// Feature buffer pool
    feature_pool: Arc<Mutex<FeatureBufferPool>>,
    /// Memory pressure monitor
    pressure_monitor: Arc<Mutex<MemoryPressureMonitor>>,
    /// Cleanup callbacks
    cleanup_callbacks: Vec<Box<dyn Fn() + Send + Sync>>,
}

impl MemoryOptimizer {
    /// Create a new memory optimizer
    pub fn new() -> Self {
        let mut monitor = MemoryPressureMonitor::new(MemoryThresholds::default());

        // Add default pressure handling
        let audio_pool_weak =
            Arc::downgrade(&Arc::new(Mutex::new(AudioBufferPool::new_audio_pool())));
        monitor.add_callback(move |level| {
            if let Some(pool) = audio_pool_weak.upgrade() {
                match level {
                    MemoryPressureLevel::High | MemoryPressureLevel::Critical => {
                        if let Ok(mut pool) = pool.lock() {
                            pool.clear();
                            tracing::warn!(
                                "Cleared audio buffer pool due to memory pressure: {:?}",
                                level
                            );
                        }
                    }
                    _ => {}
                }
            }
        });

        Self {
            audio_pool: Arc::new(Mutex::new(AudioBufferPool::new_audio_pool())),
            feature_pool: Arc::new(Mutex::new(MemoryPool::new(|size| Vec::with_capacity(size)))),
            pressure_monitor: Arc::new(Mutex::new(monitor)),
            cleanup_callbacks: Vec::new(),
        }
    }

    /// Acquire an audio buffer
    pub fn acquire_audio_buffer(&self, capacity: usize) -> Vec<f32> {
        if let Ok(mut pool) = self.audio_pool.lock() {
            pool.acquire_audio_buffer(capacity)
        } else {
            Vec::with_capacity(capacity)
        }
    }

    /// Release an audio buffer
    pub fn release_audio_buffer(&self, buffer: Vec<f32>) {
        if let Ok(mut pool) = self.audio_pool.lock() {
            pool.release_audio_buffer(buffer);
        }
    }

    /// Check memory pressure and trigger cleanup if needed
    pub fn check_memory_pressure(&self) -> MemoryPressureLevel {
        if let Ok(mut monitor) = self.pressure_monitor.lock() {
            let level = monitor.check_pressure();

            // Trigger cleanup callbacks for high pressure
            if level >= MemoryPressureLevel::High {
                for callback in &self.cleanup_callbacks {
                    callback();
                }
            }

            level
        } else {
            MemoryPressureLevel::Normal
        }
    }

    /// Add a cleanup callback
    pub fn add_cleanup_callback<F>(&mut self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.cleanup_callbacks.push(Box::new(callback));
    }

    /// Get memory statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let audio_stats = if let Ok(pool) = self.audio_pool.lock() {
            pool.stats().clone()
        } else {
            MemoryPoolStats::default()
        };

        let feature_stats = if let Ok(pool) = self.feature_pool.lock() {
            pool.stats().clone()
        } else {
            MemoryPoolStats::default()
        };

        let current_usage = if let Ok(monitor) = self.pressure_monitor.lock() {
            monitor.current_usage.lock().unwrap().clone()
        } else {
            MemoryUsage::default()
        };

        MemoryStats {
            audio_pool_stats: audio_stats,
            feature_pool_stats: feature_stats,
            current_usage,
            pressure_level: self.check_memory_pressure(),
        }
    }
}

/// Combined memory statistics
#[derive(Debug, Clone)]
/// Memory Stats
pub struct MemoryStats {
    /// audio pool stats
    pub audio_pool_stats: MemoryPoolStats,
    /// feature pool stats
    pub feature_pool_stats: MemoryPoolStats,
    /// current usage
    pub current_usage: MemoryUsage,
    /// pressure level
    pub pressure_level: MemoryPressureLevel,
}

/// Global memory optimizer instance
static GLOBAL_MEMORY_OPTIMIZER: Lazy<Arc<Mutex<MemoryOptimizer>>> =
    Lazy::new(|| Arc::new(Mutex::new(MemoryOptimizer::new())));

/// Get the global memory optimizer
pub fn global_memory_optimizer() -> Arc<Mutex<MemoryOptimizer>> {
    GLOBAL_MEMORY_OPTIMIZER.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let mut pool = AudioBufferPool::new_audio_pool();

        // Acquire and release buffers
        let buffer1 = pool.acquire_audio_buffer(1024);
        assert_eq!(buffer1.capacity(), 1024);

        pool.release_audio_buffer(buffer1);
        assert_eq!(pool.stats().pooled_buffers, 1);

        // Acquire again - should reuse
        let buffer2 = pool.acquire_audio_buffer(1024);
        assert_eq!(pool.stats().cache_hits, 1);
        assert_eq!(pool.stats().pooled_buffers, 0);

        pool.release_audio_buffer(buffer2);
    }

    #[test]
    fn test_circular_buffer() {
        let mut buffer = CircularAudioBuffer::new(4);

        // Write samples
        let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
        let written = buffer.write(&samples);
        assert_eq!(written, 5);
        assert_eq!(buffer.available(), 4); // Circular buffer overflow

        // Read samples
        let mut output = [0.0; 3];
        let read = buffer.read(&mut output);
        assert_eq!(read, 3);
        assert_eq!(output, [2.0, 3.0, 4.0]); // First sample was overwritten
    }

    #[test]
    fn test_audio_chunk_iterator() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut iter = AudioChunkIterator::new(&data, 3, 1);

        assert_eq!(iter.next(), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(iter.next(), Some(&[3.0, 4.0, 5.0][..]));
        assert_eq!(iter.next(), Some(&[5.0, 6.0][..]));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_audio_slice() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let slice = AudioSlice::new(&data, 16000, 2);

        assert_eq!(slice.duration(), 2.0 / 16000.0); // 2 samples per channel
        assert_eq!(slice.channels(), 2);
        assert_eq!(slice.sample_rate(), 16000);
    }

    #[test]
    fn test_memory_pressure_calculation() {
        let thresholds = MemoryThresholds::default();
        let monitor = MemoryPressureMonitor::new(thresholds);

        let usage = MemoryUsage {
            total_memory_bytes: 1000,
            used_memory_bytes: 800, // 80% usage
            available_memory_bytes: 200,
            process_rss_bytes: 100,
            last_updated: Instant::now(),
        };

        let level = monitor.calculate_pressure_level(&usage);
        assert_eq!(level, MemoryPressureLevel::Moderate);
    }
}
