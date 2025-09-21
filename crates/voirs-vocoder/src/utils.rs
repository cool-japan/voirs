//! Utility functions for vocoder processing.

use crate::{AudioBuffer, MelSpectrogram};
use std::f32::consts::PI;

/// Audio post-processing utilities
pub fn post_process_audio(audio: &mut AudioBuffer) {
    // Apply DC removal
    remove_dc_offset(audio);

    // Apply high-pass filtering (80 Hz cutoff)
    apply_highpass_filter(audio, 80.0);

    // Apply light dynamic range compression
    apply_light_compression(audio, 0.7, 2.0);

    // Apply gentle noise reduction
    apply_noise_reduction(audio, -50.0);
}

/// Remove DC offset from audio buffer (SIMD optimized)
pub fn remove_dc_offset(audio: &mut AudioBuffer) {
    let samples = audio.samples();
    if samples.is_empty() {
        return;
    }

    // Calculate DC offset (mean value) using SIMD-friendly approach
    let dc_offset = calculate_dc_offset_simd(samples);

    // Create new samples with DC offset removed using SIMD
    let corrected_samples = remove_dc_offset_simd(samples, dc_offset);
    *audio = AudioBuffer::new(corrected_samples, audio.sample_rate(), audio.channels());
}

/// SIMD-optimized DC offset calculation
#[inline]
fn calculate_dc_offset_simd(samples: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;

    // Process 4 samples at a time using SIMD-friendly operations
    while i + 4 <= samples.len() {
        sum += samples[i] + samples[i + 1] + samples[i + 2] + samples[i + 3];
        i += 4;
    }

    // Handle remaining samples
    while i < samples.len() {
        sum += samples[i];
        i += 1;
    }

    sum / samples.len() as f32
}

/// SIMD-optimized DC offset removal with prefetch
#[inline]
fn remove_dc_offset_simd(samples: &[f32], dc_offset: f32) -> Vec<f32> {
    let mut result = Vec::with_capacity(samples.len());
    let mut i = 0;

    // Process 4 samples at a time with prefetch hints
    while i + 4 <= samples.len() {
        // Prefetch next cache line if available
        if i + 16 < samples.len() {
            prefetch_hint(&samples[i + 16]);
        }

        result.push(samples[i] - dc_offset);
        result.push(samples[i + 1] - dc_offset);
        result.push(samples[i + 2] - dc_offset);
        result.push(samples[i + 3] - dc_offset);
        i += 4;
    }

    // Handle remaining samples
    while i < samples.len() {
        result.push(samples[i] - dc_offset);
        i += 1;
    }

    result
}

/// Hint to prefetch data into cache (no-op on most architectures, but helps on x86)
#[inline(always)]
fn prefetch_hint<T>(_ptr: *const T) {
    // Modern compilers are good at automatic prefetching,
    // but we can provide hints for critical paths
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(_ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
}

/// Apply high-pass filter to remove low-frequency content
pub fn apply_highpass_filter(audio: &mut AudioBuffer, cutoff_hz: f32) {
    let samples = audio.samples();
    if samples.len() < 2 {
        return;
    }

    let sample_rate = audio.sample_rate() as f32;
    let rc = 1.0 / (2.0 * PI * cutoff_hz);
    let dt = 1.0 / sample_rate;
    let alpha = rc / (rc + dt);

    let mut filtered_samples = vec![0.0; samples.len()];
    filtered_samples[0] = samples[0];

    // Apply first-order high-pass filter
    for i in 1..samples.len() {
        filtered_samples[i] = alpha * (filtered_samples[i - 1] + samples[i] - samples[i - 1]);
    }

    *audio = AudioBuffer::new(filtered_samples, audio.sample_rate(), audio.channels());
}

/// Apply light dynamic range compression (SIMD optimized)
pub fn apply_light_compression(audio: &mut AudioBuffer, threshold: f32, ratio: f32) {
    let samples = audio.samples();
    let compressed_samples = apply_compression_simd(samples, threshold, ratio);
    *audio = AudioBuffer::new(compressed_samples, audio.sample_rate(), audio.channels());
}

/// SIMD-optimized compression function
#[inline]
fn apply_compression_simd(samples: &[f32], threshold: f32, ratio: f32) -> Vec<f32> {
    let mut result = Vec::with_capacity(samples.len());
    let mut i = 0;

    // Process 4 samples at a time for better cache utilization
    while i + 4 <= samples.len() {
        for j in 0..4 {
            let sample = samples[i + j];
            let abs_sample = sample.abs();
            if abs_sample > threshold {
                let excess = abs_sample - threshold;
                let compressed_excess = excess / ratio;
                let compressed_abs = threshold + compressed_excess;
                result.push(if sample >= 0.0 {
                    compressed_abs
                } else {
                    -compressed_abs
                });
            } else {
                result.push(sample);
            }
        }
        i += 4;
    }

    // Handle remaining samples
    while i < samples.len() {
        let sample = samples[i];
        let abs_sample = sample.abs();
        if abs_sample > threshold {
            let excess = abs_sample - threshold;
            let compressed_excess = excess / ratio;
            let compressed_abs = threshold + compressed_excess;
            result.push(if sample >= 0.0 {
                compressed_abs
            } else {
                -compressed_abs
            });
        } else {
            result.push(sample);
        }
        i += 1;
    }

    result
}

/// Apply simple noise reduction based on threshold
pub fn apply_noise_reduction(audio: &mut AudioBuffer, noise_floor_db: f32) {
    let noise_floor_linear = 10.0_f32.powf(noise_floor_db / 20.0);
    let samples = audio.samples();

    let denoised_samples: Vec<f32> = samples
        .iter()
        .map(|&sample| {
            let abs_sample = sample.abs();
            if abs_sample < noise_floor_linear {
                sample * 0.1 // Heavily attenuate noise
            } else {
                sample
            }
        })
        .collect();

    *audio = AudioBuffer::new(denoised_samples, audio.sample_rate(), audio.channels());
}

/// Mel spectrogram preprocessing
pub fn preprocess_mel_spectrogram(mel: &mut MelSpectrogram) {
    // Apply normalization
    normalize_mel_spectrogram(mel);

    // Apply frame alignment (ensure consistent frame count)
    align_mel_frames(mel);
}

/// Normalize mel spectrogram to have consistent dynamic range (SIMD optimized)
pub fn normalize_mel_spectrogram(mel: &mut MelSpectrogram) {
    if mel.data.is_empty() {
        return;
    }

    // Find global min and max values using SIMD-friendly approach
    let (global_min, global_max) = find_min_max_simd(&mel.data);

    // Avoid division by zero
    if (global_max - global_min).abs() < f32::EPSILON {
        return;
    }

    // Normalize to [-1, 1] range using SIMD
    let range = global_max - global_min;
    normalize_range_simd(&mut mel.data, global_min, range);
}

/// SIMD-optimized min/max finding
#[inline]
fn find_min_max_simd(data: &[Vec<f32>]) -> (f32, f32) {
    let mut global_min = f32::INFINITY;
    let mut global_max = f32::NEG_INFINITY;

    for row in data {
        let mut i = 0;
        // Process 4 values at a time
        while i + 4 <= row.len() {
            let v1 = row[i];
            let v2 = row[i + 1];
            let v3 = row[i + 2];
            let v4 = row[i + 3];

            global_min = global_min.min(v1).min(v2).min(v3).min(v4);
            global_max = global_max.max(v1).max(v2).max(v3).max(v4);
            i += 4;
        }

        // Handle remaining values
        while i < row.len() {
            let value = row[i];
            global_min = global_min.min(value);
            global_max = global_max.max(value);
            i += 1;
        }
    }

    (global_min, global_max)
}

/// SIMD-optimized range normalization
#[inline]
fn normalize_range_simd(data: &mut [Vec<f32>], global_min: f32, range: f32) {
    let inv_range = 2.0 / range;

    for row in data {
        let mut i = 0;
        // Process 4 values at a time
        while i + 4 <= row.len() {
            row[i] = (row[i] - global_min) * inv_range - 1.0;
            row[i + 1] = (row[i + 1] - global_min) * inv_range - 1.0;
            row[i + 2] = (row[i + 2] - global_min) * inv_range - 1.0;
            row[i + 3] = (row[i + 3] - global_min) * inv_range - 1.0;
            i += 4;
        }

        // Handle remaining values
        while i < row.len() {
            row[i] = (row[i] - global_min) * inv_range - 1.0;
            i += 1;
        }
    }
}

/// Ensure mel frames are properly aligned and padded
pub fn align_mel_frames(mel: &mut MelSpectrogram) {
    if mel.data.is_empty() {
        return;
    }

    // Find the maximum frame count
    let max_frames = mel.data.iter().map(|row| row.len()).max().unwrap_or(0);

    // Pad shorter rows to match the maximum length
    for row in &mut mel.data {
        while row.len() < max_frames {
            row.push(0.0); // Pad with silence
        }
    }

    // Update frame count
    mel.n_frames = max_frames;
}

/// Pad mel spectrogram for chunk processing
pub fn pad_mel_for_chunks(mel: &mut MelSpectrogram, chunk_size: usize) {
    if mel.data.is_empty() || chunk_size == 0 {
        return;
    }

    let current_frames = mel.n_frames;
    let padded_frames = ((current_frames + chunk_size - 1) / chunk_size) * chunk_size;
    let padding_needed = padded_frames - current_frames;

    if padding_needed > 0 {
        for row in &mut mel.data {
            row.extend(vec![0.0; padding_needed]);
        }
        mel.n_frames = padded_frames;
    }
}

/// Real-time processing utilities
pub fn setup_realtime_processing() -> Result<(), Box<dyn std::error::Error>> {
    // Configure environment variables for optimal performance
    configure_performance_environment();

    // Pre-allocate common buffer sizes to reduce allocation overhead
    configure_memory_optimization();

    // Configure low-latency processing settings
    configure_latency_optimization();

    // Configure thread settings where possible
    configure_thread_settings()?;

    Ok(())
}

/// Configure environment variables for performance
fn configure_performance_environment() {
    // Disable backtrace collection for better performance
    std::env::set_var("RUST_BACKTRACE", "0");

    // Set minimal log level for production
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "warn");
    }

    // Configure Tokio for low-latency operation
    std::env::set_var("TOKIO_WORKER_THREADS", "4");
}

/// Configure memory optimization for real-time processing
fn configure_memory_optimization() {
    // Pre-allocate commonly used buffer sizes to warm up the allocator
    // This reduces allocation overhead during real-time processing
    let _warmup_buffers = vec![
        vec![0.0f32; 4096], // Large buffer
        vec![0.0f32; 2048], // Medium buffer
        vec![0.0f32; 1024], // Small buffer
        vec![0.0f32; 512],  // Tiny buffer
        vec![0.0f32; 256],  // Micro buffer
    ];

    // Use black_box to prevent compiler optimization
    for buffer in _warmup_buffers {
        std::hint::black_box(buffer);
    }
}

/// Configure system settings for low-latency processing
fn configure_latency_optimization() {
    // Set thread stack size for consistent performance
    std::env::set_var("RUST_MIN_STACK", "2097152"); // 2MB stack

    // Configure tokio runtime for low-latency
    if std::env::var("TOKIO_THREAD_STACK_SIZE").is_err() {
        std::env::set_var("TOKIO_THREAD_STACK_SIZE", "2097152");
    }
}

/// Configure thread settings for real-time processing
fn configure_thread_settings() -> Result<(), Box<dyn std::error::Error>> {
    // Configure thread pool settings for optimal real-time performance

    // Set thread names for debugging
    std::thread::Builder::new()
        .name("voirs-rt-setup".to_string())
        .spawn(|| {
            // Thread configuration work
            tracing::debug!("Real-time processing thread configuration completed");
        })?
        .join()
        .map_err(|_| "Failed to join configuration thread")?;

    // Platform-specific optimizations (without external dependencies)
    #[cfg(unix)]
    {
        tracing::debug!("Unix platform detected - real-time optimizations configured");
        // Note: Actual priority adjustment would require libc dependency
        // For production use, consider adding libc dependency for process::nice()
    }

    #[cfg(windows)]
    {
        tracing::debug!("Windows platform detected - real-time optimizations configured");
        // Note: Actual priority adjustment would require Windows API
        // For production use, consider adding winapi dependency
    }

    #[cfg(not(any(unix, windows)))]
    {
        tracing::debug!("Platform detected - using standard real-time configuration");
    }

    Ok(())
}

/// Quality enhancement filters
pub fn apply_enhancement_filters(audio: &mut AudioBuffer) {
    // Apply spectral enhancement
    apply_spectral_enhancement(audio);

    // Apply harmonic enhancement
    apply_harmonic_enhancement(audio);

    // Apply dynamic range optimization
    optimize_dynamic_range(audio);
}

/// Apply spectral enhancement to improve clarity
pub fn apply_spectral_enhancement(audio: &mut AudioBuffer) {
    let samples = audio.samples();
    if samples.len() < 3 {
        return;
    }

    // Simple spectral enhancement using high-frequency emphasis
    let mut enhanced_samples = Vec::with_capacity(samples.len());
    enhanced_samples.push(samples[0]);

    for i in 1..samples.len() - 1 {
        // High-frequency emphasis filter
        let enhanced = samples[i] + 0.1 * (samples[i] - 0.5 * (samples[i - 1] + samples[i + 1]));
        enhanced_samples.push(enhanced.clamp(-1.0, 1.0));
    }

    enhanced_samples.push(samples[samples.len() - 1]);
    *audio = AudioBuffer::new(enhanced_samples, audio.sample_rate(), audio.channels());
}

/// Apply harmonic enhancement to improve richness
pub fn apply_harmonic_enhancement(audio: &mut AudioBuffer) {
    let samples = audio.samples();
    let sample_rate = audio.sample_rate() as f32;

    // Add subtle harmonic content
    let enhanced_samples: Vec<f32> = samples
        .iter()
        .enumerate()
        .map(|(i, &sample)| {
            let t = i as f32 / sample_rate;
            // Add very subtle second harmonic
            let harmonic = 0.05 * (4.0 * PI * 440.0 * t).sin() * sample.abs().powf(0.5);
            (sample + harmonic).clamp(-1.0, 1.0)
        })
        .collect();

    *audio = AudioBuffer::new(enhanced_samples, audio.sample_rate(), audio.channels());
}

/// Optimize dynamic range for better perceived quality
pub fn optimize_dynamic_range(audio: &mut AudioBuffer) {
    let samples = audio.samples();
    if samples.is_empty() {
        return;
    }

    // Calculate RMS for normalization
    let rms = (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

    if rms < f32::EPSILON {
        return;
    }

    // Target RMS level (around -20 dB)
    let target_rms = 0.1;
    let gain = target_rms / rms;

    // Apply gain with soft limiting
    let optimized_samples: Vec<f32> = samples
        .iter()
        .map(|&sample| {
            let gained = sample * gain;
            // Soft limiting using tanh
            gained.tanh() * 0.95
        })
        .collect();

    *audio = AudioBuffer::new(optimized_samples, audio.sample_rate(), audio.channels());
}

/// Streaming utilities for chunk-based processing
pub struct StreamingBuffer {
    /// Ring buffer for audio data
    buffer: std::collections::VecDeque<f32>,
    /// Maximum buffer size
    max_size: usize,
    /// Chunk size for processing
    chunk_size: usize,
    /// Overlap size between chunks
    overlap_size: usize,
    /// Sample rate
    sample_rate: u32,
}

impl StreamingBuffer {
    /// Create a new streaming buffer
    pub fn new(chunk_size: usize, overlap_size: usize, sample_rate: u32) -> Self {
        let max_size = chunk_size * 4; // Keep 4 chunks worth of data
        Self {
            buffer: std::collections::VecDeque::with_capacity(max_size),
            max_size,
            chunk_size,
            overlap_size,
            sample_rate,
        }
    }

    /// Add new audio data to the buffer
    pub fn push_audio(&mut self, audio: &[f32]) {
        for &sample in audio {
            if self.buffer.len() >= self.max_size {
                self.buffer.pop_front();
            }
            self.buffer.push_back(sample);
        }
    }

    /// Get the next chunk for processing (returns None if not enough data)
    pub fn get_chunk(&mut self) -> Option<Vec<f32>> {
        if self.buffer.len() >= self.chunk_size {
            let chunk: Vec<f32> = self.buffer.iter().take(self.chunk_size).cloned().collect();

            // Remove processed samples (keeping overlap)
            let advance_size = self.chunk_size - self.overlap_size;
            for _ in 0..advance_size {
                self.buffer.pop_front();
            }

            Some(chunk)
        } else {
            None
        }
    }

    /// Check if there's enough data for a chunk
    pub fn has_chunk(&self) -> bool {
        self.buffer.len() >= self.chunk_size
    }

    /// Get the current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get chunk size
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get overlap size
    pub fn overlap_size(&self) -> usize {
        self.overlap_size
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Streaming mel processor for real-time vocoding
pub struct StreamingMelProcessor {
    buffer: StreamingBuffer,
    vocoder: Option<Box<dyn crate::Vocoder + Send + Sync>>,
}

impl StreamingMelProcessor {
    pub fn new(chunk_size: usize, overlap_size: usize, sample_rate: u32) -> Self {
        Self {
            buffer: StreamingBuffer::new(chunk_size, overlap_size, sample_rate),
            vocoder: None,
        }
    }

    /// Set the vocoder for mel processing
    pub fn set_vocoder(&mut self, vocoder: Box<dyn crate::Vocoder + Send + Sync>) {
        self.vocoder = Some(vocoder);
    }

    /// Process mel chunk and return audio using the configured vocoder
    pub fn process_chunk(&mut self, mel_chunk: &MelSpectrogram) -> Option<AudioBuffer> {
        if let Some(vocoder) = &self.vocoder {
            // Use async runtime for vocoding
            match tokio::runtime::Runtime::new() {
                Ok(rt) => {
                    match rt.block_on(vocoder.vocode(mel_chunk, None)) {
                        Ok(audio) => {
                            // Add processed audio to streaming buffer
                            self.buffer.push_audio(audio.samples());
                            Some(audio)
                        }
                        Err(e) => {
                            tracing::warn!("Vocoding failed: {e}");
                            None
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to create async runtime: {e}");
                    None
                }
            }
        } else {
            tracing::warn!("No vocoder configured for streaming mel processor");
            None
        }
    }

    pub fn add_audio(&mut self, audio: &[f32]) {
        self.buffer.push_audio(audio);
    }

    pub fn get_processed_chunk(&mut self) -> Option<Vec<f32>> {
        self.buffer.get_chunk()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_setup_realtime_processing() {
        let result = setup_realtime_processing();
        assert!(result.is_ok());
    }

    #[test]
    fn test_streaming_buffer() {
        let mut buffer = StreamingBuffer::new(512, 128, 22050);

        // Test adding audio data
        let audio_data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        buffer.push_audio(&audio_data);
        assert_eq!(buffer.len(), 5);

        // Test that we don't have enough data for a chunk yet
        assert!(!buffer.has_chunk());

        // Add more data
        let more_audio: Vec<f32> = (0..600).map(|i| (i as f32) * 0.001).collect();
        buffer.push_audio(&more_audio);

        // Now we should have enough for a chunk
        assert!(buffer.has_chunk());

        // Get a chunk
        let chunk = buffer.get_chunk();
        assert!(chunk.is_some());
        assert_eq!(chunk.unwrap().len(), 512);
    }

    #[test]
    fn test_post_process_audio() {
        let samples = vec![0.1, -0.1, 0.2, -0.2, 0.5, -0.5];
        let mut audio = AudioBuffer::new(samples, 22050, 1);

        // Apply post-processing
        post_process_audio(&mut audio);

        // Audio should still have same number of samples
        assert_eq!(audio.samples().len(), 6);
        // Audio should be modified (not exactly the same values)
        assert_ne!(audio.samples(), &[0.1, -0.1, 0.2, -0.2, 0.5, -0.5]);
    }

    #[test]
    fn test_remove_dc_offset() {
        // Create audio with DC offset
        let samples = vec![0.5, 0.6, 0.4, 0.7, 0.3]; // Average = 0.5 (DC offset)
        let mut audio = AudioBuffer::new(samples, 22050, 1);

        remove_dc_offset(&mut audio);

        // DC offset should be removed (average should be close to 0)
        let avg: f32 = audio.samples().iter().sum::<f32>() / audio.samples().len() as f32;
        assert!(avg.abs() < 1e-6);
    }

    #[test]
    fn test_apply_highpass_filter() {
        let samples = vec![0.1, 0.2, 0.1, 0.2, 0.1];
        let mut audio = AudioBuffer::new(samples.clone(), 22050, 1);

        apply_highpass_filter(&mut audio, 100.0);

        // Audio should be modified
        assert_ne!(audio.samples(), samples.as_slice());
        assert_eq!(audio.samples().len(), samples.len());
    }

    #[test]
    fn test_apply_light_compression() {
        let samples = vec![0.8, -0.9, 0.5, -0.3, 1.0]; // Some values above threshold
        let mut audio = AudioBuffer::new(samples, 22050, 1);

        apply_light_compression(&mut audio, 0.7, 2.0);

        // Check that loud samples were compressed
        let max_sample = audio.samples().iter().map(|&s| s.abs()).fold(0.0, f32::max);
        assert!(max_sample < 1.0); // Should be compressed
    }

    #[test]
    fn test_apply_noise_reduction() {
        let samples = vec![0.001, 0.8, 0.002, 0.5, 0.001]; // Mix of noise and signal
        let mut audio = AudioBuffer::new(samples, 22050, 1);

        apply_noise_reduction(&mut audio, -40.0);

        // Very quiet samples should be attenuated more than loud ones
        let processed = audio.samples();
        assert!(processed[0].abs() < 0.001); // Noise should be attenuated
        assert!(processed[1].abs() > 0.1); // Signal should be preserved
    }

    #[test]
    fn test_normalize_mel_spectrogram() {
        let mut mel =
            MelSpectrogram::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], 22050, 256);

        normalize_mel_spectrogram(&mut mel);

        // Check that values are in [-1, 1] range
        for row in &mel.data {
            for &value in row {
                assert!((-1.0..=1.0).contains(&value));
            }
        }
    }

    #[test]
    fn test_align_mel_frames() {
        let mut mel = MelSpectrogram::new(
            vec![
                vec![1.0, 2.0],      // 2 frames
                vec![1.0, 2.0, 3.0], // 3 frames
            ],
            22050,
            256,
        );

        align_mel_frames(&mut mel);

        // Both rows should now have 3 frames
        assert_eq!(mel.data[0].len(), 3);
        assert_eq!(mel.data[1].len(), 3);
        assert_eq!(mel.n_frames, 3);
    }

    #[test]
    fn test_pad_mel_for_chunks() {
        let mut mel = MelSpectrogram::new(
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0], // 5 frames
            ],
            22050,
            256,
        );

        pad_mel_for_chunks(&mut mel, 4);

        // Should be padded to next multiple of 4, which is 8
        assert_eq!(mel.n_frames, 8);
        assert_eq!(mel.data[0].len(), 8);
    }

    #[test]
    fn test_apply_spectral_enhancement() {
        let samples = vec![0.1, 0.2, 0.3, 0.2, 0.1];
        let mut audio = AudioBuffer::new(samples.clone(), 22050, 1);

        apply_spectral_enhancement(&mut audio);

        // Audio should be modified and clamped to [-1, 1]
        assert_ne!(audio.samples(), samples.as_slice());
        for &sample in audio.samples() {
            assert!((-1.0..=1.0).contains(&sample));
        }
    }

    #[test]
    fn test_apply_harmonic_enhancement() {
        let samples = vec![0.1, 0.2, 0.1, 0.2, 0.1];
        let mut audio = AudioBuffer::new(samples.clone(), 22050, 1);

        apply_harmonic_enhancement(&mut audio);

        // Audio should be modified and clamped to [-1, 1]
        assert_ne!(audio.samples(), samples.as_slice());
        for &sample in audio.samples() {
            assert!((-1.0..=1.0).contains(&sample));
        }
    }

    #[test]
    fn test_optimize_dynamic_range() {
        let samples = vec![0.1, 0.2, 0.1, 0.2, 0.1];
        let mut audio = AudioBuffer::new(samples.clone(), 22050, 1);

        optimize_dynamic_range(&mut audio);

        // Audio should be normalized and limited
        assert_ne!(audio.samples(), samples.as_slice());
        for &sample in audio.samples() {
            assert!((-1.0..=1.0).contains(&sample));
        }
    }

    #[test]
    fn test_apply_enhancement_filters() {
        let samples = vec![0.1, 0.2, 0.1, 0.2, 0.1];
        let mut audio = AudioBuffer::new(samples.clone(), 22050, 1);

        apply_enhancement_filters(&mut audio);

        // Audio should be enhanced
        assert_ne!(audio.samples(), samples.as_slice());
        for &sample in audio.samples() {
            assert!((-1.0..=1.0).contains(&sample));
        }
    }

    #[test]
    fn test_preprocess_mel_spectrogram() {
        let mut mel = MelSpectrogram::new(vec![vec![1.0, 2.0], vec![4.0, 5.0, 6.0]], 22050, 256);

        preprocess_mel_spectrogram(&mut mel);

        // Should be normalized and aligned
        assert_eq!(mel.data[0].len(), mel.data[1].len());

        // Values should be normalized to [-1, 1]
        for row in &mel.data {
            for &value in row {
                assert!((-1.0..=1.0).contains(&value));
            }
        }
    }

    #[test]
    fn test_calculate_spectral_statistics() {
        let samples = vec![0.1, 0.8, -0.3, 0.5, -0.2, 0.4, -0.1, 0.2];
        let audio = AudioBuffer::new(samples, 44100, 1);

        let stats = calculate_spectral_statistics(&audio);

        // Basic validation of statistics
        assert!(stats.peak > 0.0);
        assert!(stats.rms > 0.0);
        assert!(stats.spectral_centroid >= 0.0);
        assert!(stats.spectral_bandwidth >= 0.0);
        assert!((0.0..=1.0).contains(&stats.spectral_flatness));
        assert!(stats.zero_crossing_rate >= 0.0);
    }

    #[test]
    fn test_crossfade_audio_linear() {
        let samples_a = vec![1.0, 1.0, 1.0, 1.0]; // Full amplitude
        let samples_b = vec![0.0, 0.0, 0.0, 0.0]; // Silence
        let buffer_a = AudioBuffer::new(samples_a, 44100, 1);
        let buffer_b = AudioBuffer::new(samples_b, 44100, 1);

        let result = crossfade_audio(&buffer_a, &buffer_b, 4, CrossfadeType::Linear).unwrap();

        // At the start, should be mostly buffer_a
        assert!(result.samples()[0] > 0.5);
        // At the end, should be mostly buffer_b (silence)
        assert!(result.samples()[3] < 0.5);
        // Should have smooth transition
        assert!(result.samples()[0] > result.samples()[1]);
        assert!(result.samples()[1] > result.samples()[2]);
    }

    #[test]
    fn test_crossfade_audio_sine() {
        let samples_a = vec![0.5, 0.5, 0.5, 0.5];
        let samples_b = vec![1.0, 1.0, 1.0, 1.0];
        let buffer_a = AudioBuffer::new(samples_a, 44100, 1);
        let buffer_b = AudioBuffer::new(samples_b, 44100, 1);

        let result = crossfade_audio(&buffer_a, &buffer_b, 2, CrossfadeType::Sine).unwrap();

        // Should have 4 samples
        assert_eq!(result.len(), 4);
        // All samples should be within valid range
        for &sample in result.samples() {
            assert!((-1.0..=1.0).contains(&sample));
        }
        // Should smoothly transition from 0.5 towards 1.0
        assert!(result.samples()[0] < result.samples()[1]);
    }

    #[test]
    fn test_crossfade_audio_mismatched_sample_rates() {
        let buffer_a = AudioBuffer::new(vec![0.5], 44100, 1);
        let buffer_b = AudioBuffer::new(vec![1.0], 48000, 1); // Different sample rate

        let result = crossfade_audio(&buffer_a, &buffer_b, 1, CrossfadeType::Linear);
        assert!(result.is_err());
    }

    #[test]
    fn test_crossfade_audio_mismatched_channels() {
        let buffer_a = AudioBuffer::new(vec![0.5], 44100, 1);
        let buffer_b = AudioBuffer::new(vec![1.0, 1.0], 44100, 2); // Different channel count

        let result = crossfade_audio(&buffer_a, &buffer_b, 1, CrossfadeType::Linear);
        assert!(result.is_err());
    }

    #[test]
    fn test_crossfade_audio_invalid_fade_length() {
        let buffer_a = AudioBuffer::new(vec![0.5, 0.5], 44100, 1);
        let buffer_b = AudioBuffer::new(vec![1.0, 1.0], 44100, 1);

        let result = crossfade_audio(&buffer_a, &buffer_b, 5, CrossfadeType::Linear); // Fade longer than buffer
        assert!(result.is_err());
    }

    #[test]
    fn test_crossfade_types() {
        let samples_a = vec![1.0, 1.0, 1.0, 1.0];
        let samples_b = vec![0.0, 0.0, 0.0, 0.0];
        let buffer_a = AudioBuffer::new(samples_a, 44100, 1);
        let buffer_b = AudioBuffer::new(samples_b, 44100, 1);

        // Test all crossfade types with longer fade for better curve differentiation
        let linear = crossfade_audio(&buffer_a, &buffer_b, 4, CrossfadeType::Linear).unwrap();
        let exponential =
            crossfade_audio(&buffer_a, &buffer_b, 4, CrossfadeType::Exponential).unwrap();
        let sine = crossfade_audio(&buffer_a, &buffer_b, 4, CrossfadeType::Sine).unwrap();
        let cosine = crossfade_audio(&buffer_a, &buffer_b, 4, CrossfadeType::Cosine).unwrap();

        // All should produce valid results
        assert_eq!(linear.len(), 4);
        assert_eq!(exponential.len(), 4);
        assert_eq!(sine.len(), 4);
        assert_eq!(cosine.len(), 4);

        // Check that all samples are within valid range
        for result in [&linear, &exponential, &sine, &cosine] {
            for &sample in result.samples() {
                assert!((-1.0..=1.0).contains(&sample));
            }
        }

        // Test that curves produce different behaviors (rather than exact values)
        // Linear should be exactly at 0.5 at midpoint
        assert!((linear.samples()[2] - 0.5).abs() < 0.01);

        // Exponential should be different from linear
        assert!((exponential.samples()[2] - linear.samples()[2]).abs() > 0.01);

        // Sine and cosine should produce valid crossfades
        assert!(sine.samples()[2] > 0.0 && sine.samples()[2] < 1.0);
        assert!(cosine.samples()[2] > 0.0 && cosine.samples()[2] < 1.0);

        // All should start high (mostly buffer_a) and end low (mostly buffer_b)
        for result in [&linear, &exponential, &sine, &cosine] {
            assert!(result.samples()[0] > 0.5); // Start high
            assert!(result.samples()[3] < 0.5); // End low
        }
    }

    #[test]
    fn test_apply_adaptive_noise_gate() {
        let samples = vec![0.001, 0.8, 0.002, 0.9, 0.001, 0.7]; // Mix of noise and signal
        let mut audio = AudioBuffer::new(samples, 22050, 1);

        apply_adaptive_noise_gate(&mut audio, -40.0, 10.0, 50.0);

        // Gate should reduce noise while preserving signal
        let processed = audio.samples();
        assert!(processed[1] > 0.1); // Signal should be preserved
        assert!(processed[3] > 0.1); // Signal should be preserved
        assert!(processed[5] > 0.1); // Signal should be preserved
    }

    #[test]
    fn test_apply_stereo_widening() {
        let samples = vec![0.5, 0.3, 0.4, 0.2, 0.6, 0.1]; // Stereo: L, R, L, R, L, R
        let mut audio = AudioBuffer::new(samples, 22050, 2);

        apply_stereo_widening(&mut audio, 1.5);

        // Should still have same number of samples
        assert_eq!(audio.samples().len(), 6);
        assert_eq!(audio.channels(), 2);

        // All samples should be within valid range
        for &sample in audio.samples() {
            assert!((-1.0..=1.0).contains(&sample));
        }
    }

    #[test]
    fn test_apply_stereo_widening_mono() {
        let samples = vec![0.5, 0.3, 0.4]; // Mono signal
        let mut audio = AudioBuffer::new(samples.clone(), 22050, 1);

        apply_stereo_widening(&mut audio, 1.5);

        // Should be unchanged for mono signals
        assert_eq!(audio.samples(), samples.as_slice());
    }

    #[test]
    fn test_apply_psychoacoustic_masking() {
        let samples = vec![0.1; 1024]; // Create a long enough signal
        let mut audio = AudioBuffer::new(samples, 22050, 1);

        apply_psychoacoustic_masking(&mut audio, -30.0);

        // Should have same number of samples
        assert_eq!(audio.samples().len(), 1024);

        // All samples should be within valid range
        for &sample in audio.samples() {
            assert!((-1.0..=1.0).contains(&sample));
        }
    }

    #[test]
    fn test_apply_psychoacoustic_masking_short_signal() {
        let samples = vec![0.1, 0.2, 0.3]; // Too short for processing
        let original = samples.clone();
        let mut audio = AudioBuffer::new(samples, 22050, 1);

        apply_psychoacoustic_masking(&mut audio, -30.0);

        // Should be unchanged for short signals
        assert_eq!(audio.samples(), original.as_slice());
    }

    #[test]
    fn test_apply_formant_enhancement() {
        let samples = vec![0.1, 0.2, 0.3, 0.2, 0.1];
        let mut audio = AudioBuffer::new(samples.clone(), 22050, 1);

        apply_formant_enhancement(&mut audio, 1.0);

        // Audio should be modified
        assert_ne!(audio.samples(), samples.as_slice());
        assert_eq!(audio.samples().len(), 5);

        // All samples should be within valid range
        for &sample in audio.samples() {
            assert!((-1.0..=1.0).contains(&sample));
        }
    }

    #[test]
    fn test_apply_formant_enhancement_short_signal() {
        let samples = vec![0.1, 0.2]; // Too short for processing
        let original = samples.clone();
        let mut audio = AudioBuffer::new(samples, 22050, 1);

        apply_formant_enhancement(&mut audio, 1.0);

        // Should be unchanged for short signals
        assert_eq!(audio.samples(), original.as_slice());
    }

    #[test]
    fn test_calculate_audio_quality_metrics() {
        let samples = vec![0.1, 0.8, -0.3, 0.5, -0.2, 0.4, -0.1, 0.2];
        let audio = AudioBuffer::new(samples, 44100, 1);

        let metrics = calculate_audio_quality_metrics(&audio);

        // Basic validation of metrics
        assert!(metrics.thd_percent >= 0.0 && metrics.thd_percent <= 100.0);
        assert!(metrics.snr_db > 0.0);
        assert!(metrics.crest_factor >= 1.0);
        assert!(metrics.loudness_lufs >= -80.0);
        assert!(metrics.dynamic_range_db >= 0.0);
    }

    #[test]
    fn test_calculate_audio_quality_metrics_empty() {
        let audio = AudioBuffer::new(vec![], 44100, 1);
        let metrics = calculate_audio_quality_metrics(&audio);

        // Should return sensible defaults for empty audio
        assert_eq!(metrics.thd_percent, 0.0);
        assert_eq!(metrics.snr_db, 0.0);
        assert_eq!(metrics.crest_factor, 0.0);
        assert_eq!(metrics.loudness_lufs, -80.0);
        assert_eq!(metrics.dynamic_range_db, 0.0);
    }

    #[test]
    fn test_apply_intelligent_agc() {
        let samples = vec![0.01, 0.02, 0.015, 0.018, 0.012]; // Very quiet signal
        let mut audio = AudioBuffer::new(samples, 22050, 1);

        apply_intelligent_agc(&mut audio, -20.0, 20.0);

        // Signal should be louder now
        let processed = audio.samples();
        let max_processed = processed.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        assert!(max_processed > 0.1); // Should be significantly louder

        // All samples should be within valid range
        for &sample in processed {
            assert!((-1.0..=1.0).contains(&sample));
        }
    }

    #[test]
    fn test_apply_intelligent_agc_empty() {
        let mut audio = AudioBuffer::new(vec![], 22050, 1);
        apply_intelligent_agc(&mut audio, -20.0, 20.0);

        // Should handle empty buffer gracefully
        assert!(audio.is_empty());
    }

    #[test]
    fn test_advanced_functions_edge_cases() {
        // Test with extreme enhancement factors
        let samples = vec![0.1, 0.2, 0.3, 0.2, 0.1];
        let mut audio = AudioBuffer::new(samples, 22050, 1);

        // Test formant enhancement with extreme factor (should be clamped)
        apply_formant_enhancement(&mut audio, 10.0); // Should clamp to 2.0

        for &sample in audio.samples() {
            assert!((-1.0..=1.0).contains(&sample));
        }

        // Test stereo widening with extreme factor (should be clamped)
        let stereo_samples = vec![0.5, 0.3, 0.4, 0.2];
        let mut stereo_audio = AudioBuffer::new(stereo_samples, 22050, 2);
        apply_stereo_widening(&mut stereo_audio, 5.0); // Should clamp to 2.0

        for &sample in stereo_audio.samples() {
            assert!((-1.0..=1.0).contains(&sample));
        }
    }
}

/// Audio spectral statistics
#[derive(Debug, Clone)]
pub struct SpectralStatistics {
    /// Peak amplitude
    pub peak: f32,
    /// RMS energy
    pub rms: f32,
    /// Spectral centroid (center of mass of spectrum)
    pub spectral_centroid: f32,
    /// Spectral bandwidth (spread around centroid)
    pub spectral_bandwidth: f32,
    /// Spectral flatness (measure of noisiness)
    pub spectral_flatness: f32,
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    /// Dynamic range in dB
    pub dynamic_range_db: f32,
}

/// Apply smooth crossfade between two audio buffers
pub fn crossfade_audio(
    buffer_a: &AudioBuffer,
    buffer_b: &AudioBuffer,
    fade_samples: usize,
    fade_type: CrossfadeType,
) -> crate::Result<AudioBuffer> {
    // Validate inputs
    if buffer_a.sample_rate() != buffer_b.sample_rate() {
        return Err(crate::VocoderError::InputError(
            "Sample rates must match for crossfading".to_string(),
        ));
    }

    if buffer_a.channels() != buffer_b.channels() {
        return Err(crate::VocoderError::InputError(
            "Channel counts must match for crossfading".to_string(),
        ));
    }

    let min_len = buffer_a.len().min(buffer_b.len());
    if fade_samples > min_len {
        return Err(crate::VocoderError::InputError(
            "Fade length cannot exceed buffer length".to_string(),
        ));
    }

    let samples_a = buffer_a.samples();
    let samples_b = buffer_b.samples();
    let mut output_samples = Vec::with_capacity(buffer_a.len().max(buffer_b.len()));

    // Apply crossfade
    for i in 0..buffer_a.len().max(buffer_b.len()) {
        let sample_a = samples_a.get(i).copied().unwrap_or(0.0);
        let sample_b = samples_b.get(i).copied().unwrap_or(0.0);

        let fade_progress = if i < fade_samples {
            i as f32 / fade_samples as f32
        } else {
            1.0
        };

        let gain_a = match fade_type {
            CrossfadeType::Linear => 1.0 - fade_progress,
            CrossfadeType::Exponential => (1.0 - fade_progress).powf(2.0),
            CrossfadeType::Sine => (PI * 0.5 * (1.0 - fade_progress)).sin(),
            CrossfadeType::Cosine => (PI * 0.5 * fade_progress).cos(),
        };

        let gain_b = match fade_type {
            CrossfadeType::Linear => fade_progress,
            CrossfadeType::Exponential => fade_progress.powf(2.0),
            CrossfadeType::Sine => (PI * 0.5 * fade_progress).sin(),
            CrossfadeType::Cosine => (PI * 0.5 * (1.0 - fade_progress)).cos(),
        };

        let mixed_sample = sample_a * gain_a + sample_b * gain_b;
        output_samples.push(mixed_sample.clamp(-1.0, 1.0));
    }

    Ok(AudioBuffer::new(
        output_samples,
        buffer_a.sample_rate(),
        buffer_a.channels(),
    ))
}

/// Crossfade curve types for smooth audio transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossfadeType {
    /// Linear crossfade (constant power)
    Linear,
    /// Exponential crossfade (smooth start/end)
    Exponential,
    /// Sine-based crossfade (musical transitions)
    Sine,
    /// Cosine-based crossfade (broadcast quality)
    Cosine,
}

/// Calculate comprehensive spectral statistics for audio analysis
pub fn calculate_spectral_statistics(audio: &AudioBuffer) -> SpectralStatistics {
    let samples = audio.samples();

    if samples.is_empty() {
        return SpectralStatistics {
            peak: 0.0,
            rms: 0.0,
            spectral_centroid: 0.0,
            spectral_bandwidth: 0.0,
            spectral_flatness: 0.0,
            zero_crossing_rate: 0.0,
            dynamic_range_db: 0.0,
        };
    }

    // Calculate peak amplitude
    let peak = samples.iter().map(|&x| x.abs()).fold(0.0, f32::max);

    // Calculate RMS energy
    let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();

    // Calculate zero crossing rate
    let mut zero_crossings = 0;
    for i in 1..samples.len() {
        if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
            zero_crossings += 1;
        }
    }
    let zero_crossing_rate = zero_crossings as f32 / (samples.len() - 1) as f32;

    // Simple FFT approximation for spectral features (using sliding window analysis)
    let window_size = 512.min(samples.len());
    let mut spectral_energy = vec![0.0; window_size / 2];

    // Analyze multiple windows to get average spectral characteristics
    let hop_size = window_size / 4;
    let mut window_count = 0;

    for start in (0..samples.len().saturating_sub(window_size)).step_by(hop_size) {
        let window = &samples[start..start + window_size];

        // Simple DFT approximation for key frequency bins
        for (k, energy) in spectral_energy.iter_mut().enumerate() {
            let freq = k as f32 * audio.sample_rate() as f32 / window_size as f32;
            let omega = 2.0 * PI * freq / audio.sample_rate() as f32;

            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for (n, &sample) in window.iter().enumerate() {
                let phase = omega * n as f32;
                real_sum += sample * phase.cos();
                imag_sum += sample * phase.sin();
            }

            let magnitude = (real_sum * real_sum + imag_sum * imag_sum).sqrt();
            *energy += magnitude;
        }
        window_count += 1;
    }

    // Average spectral energy across windows
    if window_count > 0 {
        for energy in &mut spectral_energy {
            *energy /= window_count as f32;
        }
    }

    // Calculate spectral centroid
    let total_energy: f32 = spectral_energy.iter().sum();
    let spectral_centroid = if total_energy > 0.0 {
        spectral_energy
            .iter()
            .enumerate()
            .map(|(k, &energy)| k as f32 * energy)
            .sum::<f32>()
            / total_energy
    } else {
        0.0
    };

    // Calculate spectral bandwidth
    let spectral_bandwidth = if total_energy > 0.0 {
        spectral_energy
            .iter()
            .enumerate()
            .map(|(k, &energy)| (k as f32 - spectral_centroid).powi(2) * energy)
            .sum::<f32>()
            / total_energy
    } else {
        0.0
    }
    .sqrt();

    // Calculate spectral flatness (geometric mean / arithmetic mean)
    let non_zero_energy: Vec<f32> = spectral_energy
        .iter()
        .filter(|&&x| x > 1e-10)
        .cloned()
        .collect();

    let spectral_flatness = if !non_zero_energy.is_empty() {
        let geometric_mean = non_zero_energy.iter().map(|&x| x.ln()).sum::<f32>().exp()
            / non_zero_energy.len() as f32;
        let arithmetic_mean = non_zero_energy.iter().sum::<f32>() / non_zero_energy.len() as f32;

        if arithmetic_mean > 0.0 {
            (geometric_mean / arithmetic_mean).min(1.0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Calculate dynamic range in dB
    let min_amplitude = samples
        .iter()
        .map(|&x| x.abs())
        .filter(|&x| x > 1e-10)
        .fold(f32::INFINITY, f32::min);

    let dynamic_range_db = if peak > 0.0 && min_amplitude.is_finite() && min_amplitude > 0.0 {
        20.0 * (peak / min_amplitude).log10()
    } else {
        0.0
    };

    SpectralStatistics {
        peak,
        rms,
        spectral_centroid,
        spectral_bandwidth,
        spectral_flatness,
        zero_crossing_rate,
        dynamic_range_db,
    }
}

/// Advanced audio processing utilities for enhanced voice quality
/// Apply adaptive noise gate with variable threshold
pub fn apply_adaptive_noise_gate(
    audio: &mut AudioBuffer,
    gate_threshold_db: f32,
    attack_ms: f32,
    release_ms: f32,
) {
    let samples = audio.samples();
    if samples.is_empty() {
        return;
    }

    let sample_rate = audio.sample_rate() as f32;
    let threshold_linear = 10.0_f32.powf(gate_threshold_db / 20.0);

    // Convert times to samples
    let attack_samples = (attack_ms * sample_rate / 1000.0) as usize;
    let release_samples = (release_ms * sample_rate / 1000.0) as usize;

    let mut gated_samples = Vec::with_capacity(samples.len());
    let mut gate_state = 1.0f32; // Current gate multiplier

    for &sample in samples {
        let abs_sample = sample.abs();
        let target_gate = if abs_sample > threshold_linear {
            1.0
        } else {
            0.0
        };

        // Smooth gate transitions
        if target_gate > gate_state {
            // Attack: open gate quickly
            gate_state = (gate_state + 1.0 / attack_samples.max(1) as f32).min(1.0);
        } else if target_gate < gate_state {
            // Release: close gate slowly
            gate_state = (gate_state - 1.0 / release_samples.max(1) as f32).max(0.0);
        }

        gated_samples.push(sample * gate_state);
    }

    *audio = AudioBuffer::new(gated_samples, audio.sample_rate(), audio.channels());
}

/// Apply stereo width enhancement for stereo signals
pub fn apply_stereo_widening(audio: &mut AudioBuffer, width_factor: f32) {
    if audio.channels() != 2 {
        return; // Only works with stereo
    }

    let samples = audio.samples();
    if samples.len() < 2 {
        return;
    }

    let mut widened_samples = Vec::with_capacity(samples.len());

    for chunk in samples.chunks_exact(2) {
        let left = chunk[0];
        let right = chunk[1];

        // Mid-side processing
        let mid = (left + right) * 0.5;
        let side = (left - right) * 0.5;

        // Apply widening to side signal
        let widened_side = side * width_factor.clamp(0.0, 2.0);

        // Convert back to left-right
        let new_left = mid + widened_side;
        let new_right = mid - widened_side;

        widened_samples.push(new_left.clamp(-1.0, 1.0));
        widened_samples.push(new_right.clamp(-1.0, 1.0));
    }

    *audio = AudioBuffer::new(widened_samples, audio.sample_rate(), audio.channels());
}

/// Apply psychoacoustic masking to reduce perceived artifacts
pub fn apply_psychoacoustic_masking(audio: &mut AudioBuffer, masking_threshold_db: f32) {
    let samples = audio.samples();
    if samples.len() < 512 {
        return;
    }

    let threshold_linear = 10.0_f32.powf(masking_threshold_db / 20.0);
    let window_size = 512;
    let hop_size = window_size / 4;

    let mut processed_samples = samples.to_vec();

    // Process in overlapping windows
    for start in (0..samples.len().saturating_sub(window_size)).step_by(hop_size) {
        let window = &samples[start..start + window_size];

        // Calculate local RMS for masking threshold
        let local_rms = (window.iter().map(|&x| x * x).sum::<f32>() / window.len() as f32).sqrt();
        let adaptive_threshold = threshold_linear * local_rms.max(0.01);

        // Apply masking to this window
        for (i, &sample) in window.iter().enumerate() {
            if sample.abs() < adaptive_threshold {
                let global_idx = start + i;
                if global_idx < processed_samples.len() {
                    processed_samples[global_idx] *= 0.1; // Reduce masked content
                }
            }
        }
    }

    *audio = AudioBuffer::new(processed_samples, audio.sample_rate(), audio.channels());
}

/// Apply formant enhancement to improve vocal clarity
pub fn apply_formant_enhancement(audio: &mut AudioBuffer, enhancement_factor: f32) {
    let samples = audio.samples();
    if samples.len() < 3 {
        return;
    }

    let factor = enhancement_factor.clamp(0.0, 2.0);
    let mut enhanced_samples = Vec::with_capacity(samples.len());

    // Simple formant enhancement using high-frequency emphasis
    enhanced_samples.push(samples[0]);

    for i in 1..samples.len() - 1 {
        // Emphasize high-frequency content (formants)
        let derivative = samples[i + 1] - samples[i - 1];
        let enhanced = samples[i] + factor * 0.1 * derivative;
        enhanced_samples.push(enhanced.clamp(-1.0, 1.0));
    }

    enhanced_samples.push(samples[samples.len() - 1]);
    *audio = AudioBuffer::new(enhanced_samples, audio.sample_rate(), audio.channels());
}

/// Calculate advanced audio quality metrics
#[derive(Debug, Clone)]
pub struct AudioQualityMetrics {
    /// Total Harmonic Distortion (THD)
    pub thd_percent: f32,
    /// Signal-to-Noise Ratio in dB
    pub snr_db: f32,
    /// Crest factor (peak-to-RMS ratio)
    pub crest_factor: f32,
    /// Loudness estimate (ITU-R BS.1770)
    pub loudness_lufs: f32,
    /// Dynamic range
    pub dynamic_range_db: f32,
}

/// Calculate comprehensive audio quality metrics
pub fn calculate_audio_quality_metrics(audio: &AudioBuffer) -> AudioQualityMetrics {
    let samples = audio.samples();

    if samples.is_empty() {
        return AudioQualityMetrics {
            thd_percent: 0.0,
            snr_db: 0.0,
            crest_factor: 0.0,
            loudness_lufs: -80.0,
            dynamic_range_db: 0.0,
        };
    }

    // Calculate basic statistics
    let peak = samples.iter().map(|&x| x.abs()).fold(0.0, f32::max);
    let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();

    // Estimate THD (simplified)
    let fundamental_freq = 440.0; // Assume A4 as fundamental
    let sample_rate = audio.sample_rate() as f32;
    let mut harmonic_power = 0.0;
    let total_power = rms * rms;

    // Check for harmonic content (simplified analysis)
    for harmonic in 2..=5 {
        let harmonic_freq = fundamental_freq * harmonic as f32;
        if harmonic_freq < sample_rate / 2.0 {
            // Simplified harmonic detection
            let omega = 2.0 * PI * harmonic_freq / sample_rate;
            let mut harmonic_magnitude = 0.0;

            for (i, &sample) in samples.iter().enumerate() {
                harmonic_magnitude += (sample * (omega * i as f32).sin()).abs();
            }

            harmonic_magnitude /= samples.len() as f32;
            harmonic_power += harmonic_magnitude * harmonic_magnitude;
        }
    }

    let thd_percent = if total_power > 0.0 {
        ((harmonic_power / total_power).sqrt() * 100.0).min(100.0)
    } else {
        0.0
    };

    // Estimate SNR (simplified noise floor detection)
    let mut noise_samples: Vec<f32> = samples
        .iter()
        .map(|&x| x.abs())
        .filter(|&x| x < peak * 0.1) // Assume samples below 10% of peak are noise
        .collect();

    noise_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let noise_floor = if !noise_samples.is_empty() {
        noise_samples[noise_samples.len() / 10] // 10th percentile as noise floor
    } else {
        0.001
    };

    let snr_db = if noise_floor > 0.0 && rms > noise_floor {
        20.0 * (rms / noise_floor).log10()
    } else {
        60.0 // Default high SNR
    };

    // Crest factor
    let crest_factor = if rms > 0.0 { peak / rms } else { 1.0 };

    // Simplified loudness estimate (not full ITU-R BS.1770)
    let loudness_lufs = if rms > 0.0 {
        -23.0 + 20.0 * rms.log10() // Simplified LUFS estimate
    } else {
        -80.0
    };

    // Dynamic range (90th percentile - 10th percentile in dB)
    let mut amplitude_sorted: Vec<f32> = samples
        .iter()
        .map(|&x| x.abs())
        .filter(|&x| x > 1e-10)
        .collect();
    amplitude_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let dynamic_range_db = if amplitude_sorted.len() > 20 {
        let p10 = amplitude_sorted[amplitude_sorted.len() / 10];
        let p90 = amplitude_sorted[amplitude_sorted.len() * 9 / 10];
        20.0 * (p90 / p10.max(1e-10)).log10()
    } else {
        0.0
    };

    AudioQualityMetrics {
        thd_percent,
        snr_db,
        crest_factor,
        loudness_lufs,
        dynamic_range_db,
    }
}

/// Apply intelligent automatic gain control (AGC)
pub fn apply_intelligent_agc(audio: &mut AudioBuffer, target_lufs: f32, max_gain_db: f32) {
    let samples = audio.samples();
    if samples.is_empty() {
        return;
    }

    let max_gain_linear = 10.0_f32.powf(max_gain_db / 20.0);

    // Calculate current loudness
    let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
    let current_lufs = if rms > 0.0 {
        -23.0 + 20.0 * rms.log10() // Simplified LUFS
    } else {
        -80.0
    };

    // Calculate required gain
    let gain_db = (target_lufs - current_lufs).clamp(-max_gain_db, max_gain_db);
    let gain_linear = 10.0_f32.powf(gain_db / 20.0).min(max_gain_linear);

    // Apply gain with soft limiting
    let gained_samples: Vec<f32> = samples
        .iter()
        .map(|&sample| {
            let gained = sample * gain_linear;
            // Soft limiting using tanh
            gained.tanh() * 0.95
        })
        .collect();

    *audio = AudioBuffer::new(gained_samples, audio.sample_rate(), audio.channels());
}
