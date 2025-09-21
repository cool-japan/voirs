//! Advanced chunk-based processing for neural vocoders
//!
//! This module provides sophisticated chunk processing strategies including:
//! - Overlap-add windowing to reduce boundary artifacts
//! - Adaptive chunk sizing based on content complexity
//! - Memory-efficient ring buffer processing
//! - Latency-optimized processing pipelines

use crate::{AudioBuffer, MelSpectrogram, Result, Vocoder, VocoderError};
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex as AsyncMutex;

/// Advanced chunk processor with overlap-add windowing
pub struct AdvancedChunkProcessor {
    /// Underlying vocoder
    vocoder: Arc<dyn Vocoder>,

    /// Configuration
    config: AdvancedChunkConfig,

    /// Overlap buffer for seamless processing
    overlap_buffer: Arc<AsyncMutex<OverlapBuffer>>,

    /// Processing statistics
    stats: Arc<RwLock<AdvancedChunkStats>>,

    /// Content analyzer for adaptive sizing
    content_analyzer: Arc<ContentAnalyzer>,

    /// Memory pool for efficient allocation
    memory_pool: Arc<MemoryPool>,
}

/// Configuration for advanced chunk processing
#[derive(Debug, Clone)]
pub struct AdvancedChunkConfig {
    /// Base chunk size in frames
    pub base_chunk_size: usize,

    /// Overlap ratio (0.0 to 0.5)
    pub overlap_ratio: f32,

    /// Enable adaptive chunk sizing
    pub adaptive_sizing: bool,

    /// Maximum chunk size for adaptive sizing
    pub max_chunk_size: usize,

    /// Minimum chunk size for adaptive sizing
    pub min_chunk_size: usize,

    /// Window function type
    pub window_type: WindowType,

    /// Enable content-aware processing
    pub content_aware: bool,

    /// Memory budget in MB
    pub memory_budget_mb: f32,

    /// Lookahead frames for better boundary handling
    pub lookahead_frames: usize,
}

impl Default for AdvancedChunkConfig {
    fn default() -> Self {
        Self {
            base_chunk_size: 256,
            overlap_ratio: 0.25,
            adaptive_sizing: true,
            max_chunk_size: 1024,
            min_chunk_size: 128,
            window_type: WindowType::Hann,
            content_aware: true,
            memory_budget_mb: 64.0,
            lookahead_frames: 32,
        }
    }
}

/// Window function types for overlap-add processing
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    /// Hann window (default)
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Kaiser window
    Kaiser { beta: f32 },
    /// Rectangular window (no windowing)
    Rectangular,
}

/// Overlap buffer for seamless chunk processing
struct OverlapBuffer {
    /// Previous chunk overlap data
    overlap_data: VecDeque<Vec<f32>>,

    /// Overlap size in frames
    overlap_size: usize,

    /// Window function coefficients
    #[allow(dead_code)]
    window_coeffs: Vec<f32>,

    /// Fade-in coefficients
    fade_in: Vec<f32>,

    /// Fade-out coefficients
    fade_out: Vec<f32>,
}

impl OverlapBuffer {
    fn new(overlap_size: usize, window_type: WindowType) -> Self {
        let window_coeffs = generate_window(overlap_size * 2, window_type);
        let fade_in = generate_fade(overlap_size, true);
        let fade_out = generate_fade(overlap_size, false);

        Self {
            overlap_data: VecDeque::new(),
            overlap_size,
            window_coeffs,
            fade_in,
            fade_out,
        }
    }

    /// Apply overlap-add to audio buffer
    fn apply_overlap_add(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        let channels = audio.channels() as usize;
        let samples = audio.samples_mut();
        if samples.is_empty() {
            return Ok(());
        }

        let frames_per_channel = samples.len() / channels;

        // Process each channel (interleaved format)
        for ch in 0..channels {
            if let Some(prev_overlap) = self.overlap_data.get(ch) {
                let overlap_len = prev_overlap
                    .len()
                    .min(frames_per_channel)
                    .min(self.overlap_size);

                // Apply overlap-add with windowing to interleaved samples
                for i in 0..overlap_len {
                    let sample_idx = i * channels + ch;
                    if sample_idx < samples.len() && i < prev_overlap.len() {
                        let fade_out_coeff = self.fade_out.get(i).copied().unwrap_or(0.0);
                        let fade_in_coeff = self.fade_in.get(i).copied().unwrap_or(1.0);

                        samples[sample_idx] =
                            samples[sample_idx] * fade_in_coeff + prev_overlap[i] * fade_out_coeff;
                    }
                }
            }
        }

        // Store overlap for next chunk (extract from interleaved format)
        self.overlap_data.clear();
        for ch in 0..channels {
            let mut channel_overlap = Vec::new();
            let overlap_start = frames_per_channel.saturating_sub(self.overlap_size);

            for i in overlap_start..frames_per_channel {
                let sample_idx = i * channels + ch;
                if sample_idx < samples.len() {
                    channel_overlap.push(samples[sample_idx]);
                }
            }
            self.overlap_data.push_back(channel_overlap);
        }

        Ok(())
    }
}

/// Content analyzer for adaptive chunk sizing
struct ContentAnalyzer {
    /// Complexity threshold for adaptive sizing
    complexity_threshold: f32,

    /// Analysis window size
    #[allow(dead_code)]
    analysis_window: usize,
}

impl ContentAnalyzer {
    fn new() -> Self {
        Self {
            complexity_threshold: 0.1,
            analysis_window: 64,
        }
    }

    /// Analyze content complexity to determine optimal chunk size
    fn analyze_complexity(&self, mel: &MelSpectrogram) -> f32 {
        if mel.data.is_empty() {
            return 0.0;
        }

        let mut total_variance = 0.0;
        let mut count = 0;

        // Calculate variance across mel bins
        for frame in &mel.data {
            if frame.len() < 2 {
                continue;
            }

            let mean: f32 = frame.iter().sum::<f32>() / frame.len() as f32;
            let variance: f32 =
                frame.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / frame.len() as f32;

            total_variance += variance;
            count += 1;
        }

        if count > 0 {
            total_variance / count as f32
        } else {
            0.0
        }
    }

    /// Determine optimal chunk size based on content
    fn get_optimal_chunk_size(&self, mel: &MelSpectrogram, config: &AdvancedChunkConfig) -> usize {
        if !config.adaptive_sizing {
            return config.base_chunk_size;
        }

        let complexity = self.analyze_complexity(mel);

        // More complex content gets smaller chunks for better quality
        // Less complex content gets larger chunks for efficiency
        let size_factor = if complexity > self.complexity_threshold {
            0.7 // Smaller chunks for complex content
        } else {
            1.3 // Larger chunks for simple content
        };

        let optimal_size = (config.base_chunk_size as f32 * size_factor) as usize;
        optimal_size.clamp(config.min_chunk_size, config.max_chunk_size)
    }
}

/// Memory pool for efficient buffer allocation
struct MemoryPool {
    /// Pre-allocated audio buffers
    audio_pool: Arc<AsyncMutex<Vec<AudioBuffer>>>,

    /// Pre-allocated mel buffers
    #[allow(dead_code)]
    mel_pool: Arc<AsyncMutex<Vec<MelSpectrogram>>>,

    /// Pool configuration
    #[allow(dead_code)]
    max_pool_size: usize,

    /// Memory usage tracking
    current_usage_mb: Arc<RwLock<f32>>,
}

impl MemoryPool {
    fn new(max_pool_size: usize) -> Self {
        Self {
            audio_pool: Arc::new(AsyncMutex::new(Vec::new())),
            mel_pool: Arc::new(AsyncMutex::new(Vec::new())),
            max_pool_size,
            current_usage_mb: Arc::new(RwLock::new(0.0)),
        }
    }

    /// Get or create an audio buffer
    async fn get_audio_buffer(&self, channels: u32, total_samples: usize) -> AudioBuffer {
        let mut pool = self.audio_pool.lock().await;

        // Try to reuse an existing buffer
        if let Some(buffer) = pool.pop() {
            // For simplicity, just create a new buffer with the right size
            // In a real implementation, you'd resize the existing buffer
            drop(buffer); // Return the buffer to avoid memory leak
        }

        // Create new buffer with interleaved samples
        AudioBuffer::new(vec![0.0; total_samples], 22050, channels)
    }

    /// Return an audio buffer to the pool
    #[allow(dead_code)]
    async fn return_audio_buffer(&self, buffer: AudioBuffer) {
        let mut pool = self.audio_pool.lock().await;

        if pool.len() < self.max_pool_size {
            pool.push(buffer);
        }
    }

    /// Get current memory usage
    fn get_memory_usage_mb(&self) -> f32 {
        *self.current_usage_mb.read().unwrap()
    }
}

/// Advanced chunk processing statistics
#[derive(Debug, Clone, Default)]
pub struct AdvancedChunkStats {
    /// Total chunks processed
    pub chunks_processed: u64,

    /// Average chunk size used
    pub avg_chunk_size: f32,

    /// Overlap-add applications
    pub overlap_operations: u64,

    /// Adaptive sizing decisions
    pub adaptive_sizing_decisions: u64,

    /// Memory pool hits
    pub memory_pool_hits: u64,

    /// Memory pool misses
    pub memory_pool_misses: u64,

    /// Content complexity histogram
    pub complexity_histogram: [u64; 10],

    /// Processing time breakdown
    pub windowing_time_ms: f32,
    pub vocoding_time_ms: f32,
    pub overlap_time_ms: f32,

    /// Memory efficiency metrics
    pub memory_efficiency: f32,
    pub peak_memory_usage_mb: f32,
}

impl AdvancedChunkProcessor {
    /// Create new advanced chunk processor
    pub fn new(vocoder: Arc<dyn Vocoder>, config: AdvancedChunkConfig) -> Self {
        let overlap_size = (config.base_chunk_size as f32 * config.overlap_ratio) as usize;
        let overlap_buffer = Arc::new(AsyncMutex::new(OverlapBuffer::new(
            overlap_size,
            config.window_type,
        )));

        Self {
            vocoder,
            config: config.clone(),
            overlap_buffer,
            stats: Arc::new(RwLock::new(AdvancedChunkStats::default())),
            content_analyzer: Arc::new(ContentAnalyzer::new()),
            memory_pool: Arc::new(MemoryPool::new(16)),
        }
    }

    /// Process mel spectrogram with advanced chunking
    pub async fn process_advanced(&self, mel: MelSpectrogram) -> Result<AudioBuffer> {
        let start_time = std::time::Instant::now();

        // Analyze content for adaptive chunk sizing
        let optimal_chunk_size = self
            .content_analyzer
            .get_optimal_chunk_size(&mel, &self.config);

        // Record complexity for statistics
        let complexity = self.content_analyzer.analyze_complexity(&mel);
        self.update_complexity_histogram(complexity);

        // Split into overlapping chunks if needed
        let chunks = self.split_into_chunks(&mel, optimal_chunk_size).await?;

        let mut audio_results = Vec::new();

        // Process each chunk
        for chunk in chunks {
            let chunk_start = std::time::Instant::now();

            // Perform vocoding
            let mut audio = self.vocoder.vocode(&chunk, None).await?;

            let vocoding_time = chunk_start.elapsed().as_secs_f32() * 1000.0;

            // Apply overlap-add windowing
            let overlap_start = std::time::Instant::now();
            self.overlap_buffer
                .lock()
                .await
                .apply_overlap_add(&mut audio)?;
            let overlap_time = overlap_start.elapsed().as_secs_f32() * 1000.0;

            audio_results.push(audio);

            // Update timing statistics
            if let Ok(mut stats) = self.stats.write() {
                stats.vocoding_time_ms += vocoding_time;
                stats.overlap_time_ms += overlap_time;
                stats.overlap_operations += 1;
            }
        }

        // Concatenate results
        let final_audio = self.concatenate_audio_buffers(audio_results).await?;

        // Update statistics
        let total_time = start_time.elapsed().as_secs_f32() * 1000.0;
        self.update_stats(optimal_chunk_size, total_time);

        Ok(final_audio)
    }

    /// Split mel spectrogram into overlapping chunks
    async fn split_into_chunks(
        &self,
        mel: &MelSpectrogram,
        chunk_size: usize,
    ) -> Result<Vec<MelSpectrogram>> {
        let overlap_size = (chunk_size as f32 * self.config.overlap_ratio) as usize;
        let step_size = chunk_size - overlap_size;

        let mut chunks = Vec::new();
        let mut start_frame = 0;

        while start_frame < mel.n_frames {
            let end_frame = (start_frame + chunk_size).min(mel.n_frames);

            // Extract chunk data
            let chunk_data: Vec<Vec<f32>> = mel.data[start_frame..end_frame].to_vec();

            if !chunk_data.is_empty() {
                let chunk = MelSpectrogram::new(chunk_data, mel.sample_rate, mel.hop_length);
                chunks.push(chunk);
            }

            start_frame += step_size;

            // Avoid infinite loop for very small step sizes
            if step_size == 0 {
                break;
            }
        }

        Ok(chunks)
    }

    /// Concatenate audio buffers with proper alignment
    async fn concatenate_audio_buffers(&self, buffers: Vec<AudioBuffer>) -> Result<AudioBuffer> {
        if buffers.is_empty() {
            return Err(VocoderError::VocodingError(
                "No audio buffers to concatenate".to_string(),
            ));
        }

        if buffers.len() == 1 {
            return Ok(buffers.into_iter().next().unwrap());
        }

        let first_buffer = &buffers[0];
        let channels = first_buffer.channels();
        let _sample_rate = first_buffer.sample_rate();

        // Calculate total samples (interleaved)
        let total_samples: usize = buffers.iter().map(|b| b.len()).sum();

        // Get buffer from pool
        let mut result = self
            .memory_pool
            .get_audio_buffer(channels, total_samples)
            .await;

        // Concatenate all samples (already interleaved)
        let result_samples = result.samples_mut();
        let mut current_pos = 0;

        for buffer in &buffers {
            let buffer_samples = buffer.samples();
            let end_pos = current_pos + buffer_samples.len();

            if end_pos <= result_samples.len() {
                result_samples[current_pos..end_pos].copy_from_slice(buffer_samples);
                current_pos = end_pos;
            }
        }

        Ok(result)
    }

    /// Update complexity histogram for statistics
    fn update_complexity_histogram(&self, complexity: f32) {
        if let Ok(mut stats) = self.stats.write() {
            let bin = ((complexity * 10.0) as usize).min(9);
            stats.complexity_histogram[bin] += 1;
        }
    }

    /// Update processing statistics
    fn update_stats(&self, chunk_size: usize, _processing_time: f32) {
        if let Ok(mut stats) = self.stats.write() {
            stats.chunks_processed += 1;
            stats.avg_chunk_size = (stats.avg_chunk_size * (stats.chunks_processed - 1) as f32
                + chunk_size as f32)
                / stats.chunks_processed as f32;

            if self.config.adaptive_sizing {
                stats.adaptive_sizing_decisions += 1;
            }

            // Update memory efficiency
            let current_memory = self.memory_pool.get_memory_usage_mb();
            stats.memory_efficiency = if current_memory > 0.0 {
                self.config.memory_budget_mb / current_memory
            } else {
                1.0
            };

            if current_memory > stats.peak_memory_usage_mb {
                stats.peak_memory_usage_mb = current_memory;
            }
        }
    }

    /// Get advanced processing statistics
    pub fn get_stats(&self) -> AdvancedChunkStats {
        self.stats.read().unwrap().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        if let Ok(mut stats) = self.stats.write() {
            *stats = AdvancedChunkStats::default();
        }
    }

    /// Update configuration
    pub async fn update_config(&mut self, new_config: AdvancedChunkConfig) {
        self.config = new_config.clone();

        // Update overlap buffer if window type changed
        let overlap_size = (new_config.base_chunk_size as f32 * new_config.overlap_ratio) as usize;
        *self.overlap_buffer.lock().await =
            OverlapBuffer::new(overlap_size, new_config.window_type);
    }
}

/// Generate window function coefficients
fn generate_window(size: usize, window_type: WindowType) -> Vec<f32> {
    let mut window = vec![0.0; size];

    match window_type {
        WindowType::Hann =>
        {
            #[allow(clippy::needless_range_loop)]
            for i in 0..size {
                window[i] =
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos());
            }
        }
        WindowType::Hamming =>
        {
            #[allow(clippy::needless_range_loop)]
            for i in 0..size {
                window[i] =
                    0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos();
            }
        }
        WindowType::Blackman => {
            for (i, item) in window.iter_mut().enumerate().take(size) {
                let n = i as f32;
                let n_max = (size - 1) as f32;
                *item = 0.42 - 0.5 * (2.0 * std::f32::consts::PI * n / n_max).cos()
                    + 0.08 * (4.0 * std::f32::consts::PI * n / n_max).cos();
            }
        }
        WindowType::Kaiser { beta } => {
            let i0_beta = modified_bessel_i0(beta);
            for (i, item) in window.iter_mut().enumerate().take(size) {
                let n = i as f32;
                let n_max = (size - 1) as f32;
                let arg = beta * (1.0 - ((2.0 * n / n_max) - 1.0).powi(2)).sqrt();
                *item = modified_bessel_i0(arg) / i0_beta;
            }
        }
        WindowType::Rectangular => {
            window.fill(1.0);
        }
    }

    window
}

/// Generate fade in/out coefficients
fn generate_fade(size: usize, fade_in: bool) -> Vec<f32> {
    let mut fade = vec![0.0; size];

    #[allow(clippy::needless_range_loop)]
    for i in 0..size {
        let t = i as f32 / size as f32;
        fade[i] = if fade_in {
            t // Linear fade in
        } else {
            1.0 - t // Linear fade out
        };
    }

    fade
}

/// Modified Bessel function of the first kind (order 0)
fn modified_bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_half_sq = (x / 2.0).powi(2);

    for k in 1..=20 {
        term *= x_half_sq / (k as f32).powi(2);
        sum += term;

        // Early termination for convergence
        if term < 1e-8 {
            break;
        }
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DummyVocoder;

    #[tokio::test]
    async fn test_advanced_chunk_processor_creation() {
        let vocoder = Arc::new(DummyVocoder::new());
        let config = AdvancedChunkConfig::default();
        let processor = AdvancedChunkProcessor::new(vocoder, config);

        let stats = processor.get_stats();
        assert_eq!(stats.chunks_processed, 0);
    }

    #[tokio::test]
    async fn test_window_generation() {
        let window = generate_window(64, WindowType::Hann);
        assert_eq!(window.len(), 64);
        assert!(window[0] < 0.1); // Should be near 0 at edges
        assert!(window[32] > 0.9); // Should be near 1 in middle
    }

    #[tokio::test]
    async fn test_content_analyzer() {
        let analyzer = ContentAnalyzer::new();
        let mel_data = vec![vec![0.5; 80]; 100];
        let mel = MelSpectrogram::new(mel_data, 22050, 256);

        let complexity = analyzer.analyze_complexity(&mel);
        assert!(complexity >= 0.0);

        let config = AdvancedChunkConfig::default();
        let chunk_size = analyzer.get_optimal_chunk_size(&mel, &config);
        assert!(chunk_size >= config.min_chunk_size);
        assert!(chunk_size <= config.max_chunk_size);
    }

    #[test]
    fn test_overlap_buffer() {
        let mut buffer = OverlapBuffer::new(32, WindowType::Hann);
        let mut audio = AudioBuffer::new(vec![1.0; 256], 22050, 2); // 128 frames * 2 channels

        // First call should work without overlap
        let result = buffer.apply_overlap_add(&mut audio);
        assert!(result.is_ok());

        // Second call should apply overlap
        let mut audio2 = AudioBuffer::new(vec![0.5; 256], 22050, 2); // 128 frames * 2 channels
        let result2 = buffer.apply_overlap_add(&mut audio2);
        assert!(result2.is_ok());
    }

    #[test]
    fn test_modified_bessel_i0() {
        let result = modified_bessel_i0(0.0);
        assert!((result - 1.0).abs() < 1e-6);

        let result = modified_bessel_i0(1.0);
        assert!(result > 1.0);
    }
}
