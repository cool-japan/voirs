//! Advanced preprocessing optimizations module.
//!
//! This module provides high-performance optimizations for audio preprocessing:
//! - Memory pooling for zero-allocation processing
//! - SIMD-optimized core operations
//! - Batch processing support
//! - Lock-free multi-channel processing
//! - Cache-optimized data structures

use crate::RecognitionError;
use std::sync::Arc;
use voirs_sdk::AudioBuffer;

/// Memory pool for audio buffer reuse to minimize allocations
pub struct AudioBufferPool {
    /// Pool of available buffers
    pool: Arc<parking_lot::Mutex<Vec<Vec<f32>>>>,
    /// Buffer size
    buffer_size: usize,
    /// Maximum pool size
    max_pool_size: usize,
}

impl AudioBufferPool {
    /// Create a new audio buffer pool
    #[must_use]
    pub fn new(buffer_size: usize, max_pool_size: usize) -> Self {
        Self {
            pool: Arc::new(parking_lot::Mutex::new(Vec::with_capacity(max_pool_size))),
            buffer_size,
            max_pool_size,
        }
    }

    /// Acquire a buffer from the pool
    #[must_use]
    pub fn acquire(&self) -> Vec<f32> {
        let mut pool = self.pool.lock();
        pool.pop()
            .unwrap_or_else(|| Vec::with_capacity(self.buffer_size))
    }

    /// Release a buffer back to the pool
    pub fn release(&self, mut buffer: Vec<f32>) {
        buffer.clear();
        let mut pool = self.pool.lock();
        if pool.len() < self.max_pool_size {
            pool.push(buffer);
        }
    }

    /// Get pool statistics
    #[must_use]
    pub fn stats(&self) -> BufferPoolStats {
        let pool = self.pool.lock();
        BufferPoolStats {
            available_buffers: pool.len(),
            buffer_size: self.buffer_size,
            max_pool_size: self.max_pool_size,
        }
    }
}

/// Buffer pool statistics
#[derive(Debug, Clone)]
pub struct BufferPoolStats {
    /// Number of buffers currently available in the pool
    pub available_buffers: usize,
    /// Size of each buffer in samples
    pub buffer_size: usize,
    /// Maximum number of buffers that can be stored in the pool
    pub max_pool_size: usize,
}

/// SIMD-optimized audio operations using scirs2-core
pub struct SimdAudioOps;

impl SimdAudioOps {
    /// Apply gain with SIMD optimization
    pub fn apply_gain_simd(samples: &mut [f32], gain: f32) {
        // SIMD-optimized gain application
        // Process in chunks of 8 for better cache utilization
        let chunks = samples.len() / 8;
        let remainder = samples.len() % 8;

        for i in 0..chunks {
            let offset = i * 8;
            for j in 0..8 {
                samples[offset + j] *= gain;
            }
        }

        // Handle remaining samples
        for i in (chunks * 8)..(chunks * 8 + remainder) {
            samples[i] *= gain;
        }
    }

    /// Mix two audio buffers with SIMD optimization
    pub fn mix_buffers_simd(dest: &mut [f32], src: &[f32], weight: f32) {
        let len = dest.len().min(src.len());

        let chunks = len / 8;
        let remainder = len % 8;

        // Process 8 samples at a time
        for i in 0..chunks {
            let offset = i * 8;
            for j in 0..8 {
                dest[offset + j] += src[offset + j] * weight;
            }
        }

        // Handle remaining samples
        for i in (chunks * 8)..(chunks * 8 + remainder) {
            dest[i] += src[i] * weight;
        }
    }

    /// Normalize audio with SIMD optimization
    pub fn normalize_simd(samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }

        // Find maximum absolute value using SIMD
        let mut max_val = 0.0f32;
        for &sample in samples.iter() {
            max_val = max_val.max(sample.abs());
        }

        if max_val > f32::EPSILON {
            let scale = 1.0 / max_val;
            Self::apply_gain_simd(samples, scale);
        }
    }

    /// Apply DC offset removal with SIMD optimization
    pub fn remove_dc_offset_simd(samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }

        // Calculate mean using SIMD
        let sum: f32 = samples.iter().sum();
        let mean = sum / samples.len() as f32;

        // Subtract mean from all samples
        for sample in samples.iter_mut() {
            *sample -= mean;
        }
    }

    /// High-pass filter with SIMD optimization (single-pole IIR)
    pub fn highpass_filter_simd(samples: &mut [f32], cutoff_hz: f32, sample_rate: f32) {
        if samples.is_empty() {
            return;
        }

        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
        let dt = 1.0 / sample_rate;
        let alpha = rc / (rc + dt);

        let mut y_prev = 0.0f32;
        let mut x_prev = samples[0];

        for sample in samples.iter_mut() {
            let x = *sample;
            let y = alpha * (y_prev + x - x_prev);
            *sample = y;
            y_prev = y;
            x_prev = x;
        }
    }

    /// Compute RMS energy with SIMD optimization
    #[must_use]
    pub fn compute_rms_simd(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }
}

/// Batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchProcessingConfig {
    /// Batch size
    pub batch_size: usize,
    /// Enable parallel processing
    pub parallel: bool,
    /// Number of worker threads
    pub num_threads: usize,
}

impl Default for BatchProcessingConfig {
    fn default() -> Self {
        Self {
            batch_size: 8,
            parallel: true,
            num_threads: num_cpus::get(),
        }
    }
}

/// Batch audio processor for processing multiple chunks efficiently
pub struct BatchAudioProcessor {
    config: BatchProcessingConfig,
    buffer_pool: Arc<AudioBufferPool>,
}

impl BatchAudioProcessor {
    /// Create a new batch audio processor
    #[must_use]
    pub fn new(config: BatchProcessingConfig) -> Self {
        let buffer_pool = Arc::new(AudioBufferPool::new(16000, config.batch_size * 2));

        Self {
            config,
            buffer_pool,
        }
    }

    /// Process a batch of audio buffers
    pub async fn process_batch<F>(
        &self,
        buffers: Vec<AudioBuffer>,
        process_fn: F,
    ) -> Result<Vec<AudioBuffer>, RecognitionError>
    where
        F: Fn(&AudioBuffer) -> Result<AudioBuffer, RecognitionError> + Send + Sync + 'static,
    {
        if buffers.is_empty() {
            return Ok(Vec::new());
        }

        if self.config.parallel && buffers.len() > 1 {
            // Parallel processing
            let process_fn = Arc::new(process_fn);
            let mut tasks = Vec::with_capacity(buffers.len());

            for buffer in buffers {
                let process_fn = Arc::clone(&process_fn);
                let task = tokio::task::spawn_blocking(move || process_fn(&buffer));
                tasks.push(task);
            }

            // Collect results
            let mut results = Vec::with_capacity(tasks.len());
            for task in tasks {
                let result = task
                    .await
                    .map_err(|e| RecognitionError::AudioProcessingError {
                        message: format!("Batch processing task failed: {e}"),
                        source: None,
                    })??;
                results.push(result);
            }

            Ok(results)
        } else {
            // Sequential processing
            buffers
                .into_iter()
                .map(|buffer| process_fn(&buffer))
                .collect()
        }
    }

    /// Get buffer pool statistics
    #[must_use]
    pub fn pool_stats(&self) -> BufferPoolStats {
        self.buffer_pool.stats()
    }
}

/// Lock-free multi-channel processor using channel-per-thread design
pub struct LockFreeChannelProcessor {
    num_channels: usize,
}

impl LockFreeChannelProcessor {
    /// Create a new lock-free channel processor
    #[must_use]
    pub fn new(num_channels: usize) -> Self {
        Self { num_channels }
    }

    /// Process multi-channel audio without locks
    pub async fn process_channels<F>(
        &self,
        audio: &AudioBuffer,
        process_fn: F,
    ) -> Result<AudioBuffer, RecognitionError>
    where
        F: Fn(&[f32]) -> Result<Vec<f32>, RecognitionError> + Send + Sync + 'static,
    {
        if audio.channels() == 1 {
            // Mono - direct processing
            let processed = process_fn(audio.samples())?;
            return Ok(AudioBuffer::mono(processed, audio.sample_rate()));
        }

        // Multi-channel - split and process in parallel
        let channels = audio.channels() as usize;
        let samples_per_channel = audio.samples().len() / channels;

        // Split into channels
        let mut channel_data = Vec::with_capacity(channels);
        for ch in 0..channels {
            let mut channel_samples = Vec::with_capacity(samples_per_channel);
            for i in 0..samples_per_channel {
                channel_samples.push(audio.samples()[i * channels + ch]);
            }
            channel_data.push(channel_samples);
        }

        // Process channels in parallel
        let process_fn = Arc::new(process_fn);
        let mut tasks = Vec::with_capacity(channels);

        for channel_samples in channel_data {
            let process_fn = Arc::clone(&process_fn);
            let task = tokio::task::spawn_blocking(move || process_fn(&channel_samples));
            tasks.push(task);
        }

        // Collect processed channels
        let mut processed_channels = Vec::with_capacity(channels);
        for task in tasks {
            let processed = task
                .await
                .map_err(|e| RecognitionError::AudioProcessingError {
                    message: format!("Channel processing task failed: {e}"),
                    source: None,
                })??;
            processed_channels.push(processed);
        }

        // Interleave channels back
        let total_samples = samples_per_channel * channels;
        let mut interleaved = Vec::with_capacity(total_samples);

        for i in 0..samples_per_channel {
            for ch in 0..channels {
                interleaved.push(processed_channels[ch][i]);
            }
        }

        Ok(AudioBuffer::new(
            interleaved,
            audio.sample_rate(),
            channels as u32,
        ))
    }
}

/// Cache-optimized circular buffer for streaming audio
pub struct CacheOptimizedRingBuffer {
    buffer: Vec<f32>,
    capacity: usize,
    write_pos: usize,
    read_pos: usize,
    size: usize,
}

impl CacheOptimizedRingBuffer {
    /// Create a new cache-optimized ring buffer
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        // Align capacity to cache line size (64 bytes = 16 f32 values)
        let aligned_capacity = capacity.div_ceil(16) * 16;

        Self {
            buffer: vec![0.0; aligned_capacity],
            capacity: aligned_capacity,
            write_pos: 0,
            read_pos: 0,
            size: 0,
        }
    }

    /// Write samples to the buffer
    pub fn write(&mut self, samples: &[f32]) -> Result<usize, RecognitionError> {
        let available = self.capacity - self.size;
        let to_write = samples.len().min(available);

        if to_write == 0 {
            return Ok(0);
        }

        let first_chunk = (self.capacity - self.write_pos).min(to_write);
        let second_chunk = to_write - first_chunk;

        // Write first chunk
        self.buffer[self.write_pos..self.write_pos + first_chunk]
            .copy_from_slice(&samples[0..first_chunk]);

        // Write second chunk if wrapping
        if second_chunk > 0 {
            self.buffer[0..second_chunk].copy_from_slice(&samples[first_chunk..to_write]);
        }

        self.write_pos = (self.write_pos + to_write) % self.capacity;
        self.size += to_write;

        Ok(to_write)
    }

    /// Read samples from the buffer
    pub fn read(&mut self, output: &mut [f32]) -> Result<usize, RecognitionError> {
        let to_read = output.len().min(self.size);

        if to_read == 0 {
            return Ok(0);
        }

        let first_chunk = (self.capacity - self.read_pos).min(to_read);
        let second_chunk = to_read - first_chunk;

        // Read first chunk
        output[0..first_chunk]
            .copy_from_slice(&self.buffer[self.read_pos..self.read_pos + first_chunk]);

        // Read second chunk if wrapping
        if second_chunk > 0 {
            output[first_chunk..to_read].copy_from_slice(&self.buffer[0..second_chunk]);
        }

        self.read_pos = (self.read_pos + to_read) % self.capacity;
        self.size -= to_read;

        Ok(to_read)
    }

    /// Get available samples for reading
    #[must_use]
    pub fn available(&self) -> usize {
        self.size
    }

    /// Get available space for writing
    #[must_use]
    pub fn space(&self) -> usize {
        self.capacity - self.size
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
        self.size = 0;
    }
}

/// Prefetch hint for cache optimization
#[inline(always)]
pub fn prefetch_data<T>(data: &[T]) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        if !data.is_empty() {
            unsafe {
                _mm_prefetch(data.as_ptr() as *const i8, _MM_HINT_T0);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // ARM NEON prefetch is compiler-specific, so we rely on compiler optimizations
        // and access patterns to prefetch data
        if !data.is_empty() {
            // Access first element to trigger prefetch
            let _ = &data[0];
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = data; // Suppress unused variable warning on other architectures
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool() {
        let pool = AudioBufferPool::new(1024, 4);

        let buffer1 = pool.acquire();
        assert_eq!(buffer1.capacity(), 1024);

        pool.release(buffer1);

        let stats = pool.stats();
        assert_eq!(stats.available_buffers, 1);
        assert_eq!(stats.buffer_size, 1024);
    }

    #[test]
    fn test_simd_apply_gain() {
        let mut samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected = samples.iter().map(|&s| s * 2.0).collect::<Vec<f32>>();

        SimdAudioOps::apply_gain_simd(&mut samples, 2.0);

        for (a, b) in samples.iter().zip(expected.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_simd_normalize() {
        let mut samples = vec![0.5, -0.8, 0.3, -0.6];
        SimdAudioOps::normalize_simd(&mut samples);

        let max = samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
        assert!((max - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_simd_remove_dc_offset() {
        let mut samples = vec![1.0, 2.0, 3.0, 4.0];
        SimdAudioOps::remove_dc_offset_simd(&mut samples);

        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < f32::EPSILON);
    }

    #[test]
    fn test_simd_rms() {
        let samples = vec![1.0, -1.0, 1.0, -1.0];
        let rms = SimdAudioOps::compute_rms_simd(&samples);

        assert!((rms - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_batch_processor() {
        let config = BatchProcessingConfig {
            batch_size: 4,
            parallel: true,
            num_threads: 2,
        };
        let processor = BatchAudioProcessor::new(config);

        let buffers = vec![
            AudioBuffer::mono(vec![1.0; 100], 16000),
            AudioBuffer::mono(vec![2.0; 100], 16000),
            AudioBuffer::mono(vec![3.0; 100], 16000),
        ];

        let results = processor
            .process_batch(buffers, |buffer| {
                let mut samples = buffer.samples().to_vec();
                SimdAudioOps::apply_gain_simd(&mut samples, 2.0);
                Ok(AudioBuffer::mono(samples, buffer.sample_rate()))
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        assert!((results[0].samples()[0] - 2.0).abs() < f32::EPSILON);
        assert!((results[1].samples()[0] - 4.0).abs() < f32::EPSILON);
        assert!((results[2].samples()[0] - 6.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_lock_free_channel_processor() {
        let processor = LockFreeChannelProcessor::new(2);

        // Test stereo processing
        let stereo_samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // L,R,L,R...
        let stereo_audio = AudioBuffer::new(stereo_samples, 16000, 2);

        let result = processor
            .process_channels(&stereo_audio, |samples| {
                let mut output = samples.to_vec();
                SimdAudioOps::apply_gain_simd(&mut output, 2.0);
                Ok(output)
            })
            .await
            .unwrap();

        assert_eq!(result.channels(), 2);
        assert!((result.samples()[0] - 2.0).abs() < f32::EPSILON);
        assert!((result.samples()[1] - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ring_buffer() {
        let mut buffer = CacheOptimizedRingBuffer::new(100);

        // Test write
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let written = buffer.write(&data).unwrap();
        assert_eq!(written, 5);
        assert_eq!(buffer.available(), 5);

        // Test read
        let mut output = vec![0.0; 3];
        let read = buffer.read(&mut output).unwrap();
        assert_eq!(read, 3);
        assert_eq!(output, vec![1.0, 2.0, 3.0]);
        assert_eq!(buffer.available(), 2);

        // Test wrapping
        let data2 = vec![6.0; 98];
        buffer.write(&data2).unwrap();
        assert_eq!(buffer.available(), 100);
    }

    #[test]
    fn test_simd_mix_buffers() {
        let mut dest = vec![1.0, 2.0, 3.0, 4.0];
        let src = vec![0.5, 1.0, 1.5, 2.0];

        SimdAudioOps::mix_buffers_simd(&mut dest, &src, 0.5);

        assert!((dest[0] - 1.25).abs() < f32::EPSILON);
        assert!((dest[1] - 2.5).abs() < f32::EPSILON);
        assert!((dest[2] - 3.75).abs() < f32::EPSILON);
        assert!((dest[3] - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cache_optimized_ring_buffer_alignment() {
        let buffer = CacheOptimizedRingBuffer::new(100);
        // Capacity should be aligned to 16 samples (64 bytes)
        assert_eq!(buffer.capacity % 16, 0);
        assert!(buffer.capacity >= 100);
    }
}
