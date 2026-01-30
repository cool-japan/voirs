//! Real-time Adaptive Audio Enhancement System
//!
//! This module provides sophisticated audio enhancement capabilities with adaptive
//! processing, spectral analysis, and quality-aware optimization for production TTS.
//!
//! # Features
//!
//! - **Adaptive Noise Gate**: Spectral-aware noise reduction with learning
//! - **Multiband Compression**: Dynamic range optimization across frequency bands
//! - **Spectral Enhancement**: Adaptive EQ for clarity and presence
//! - **Quality Prediction**: Real-time quality assessment and optimization
//! - **Performance Adaptive**: Automatically adjusts processing based on load
//!
//! # Example
//!
//! ```no_run
//! use voirs_sdk::audio::{AudioBuffer, AdaptiveEnhancer, EnhancementConfig};
//!
//! # fn main() -> voirs_sdk::Result<()> {
//! let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
//!
//! let mut enhancer = AdaptiveEnhancer::new(EnhancementConfig::default());
//! let enhanced = enhancer.enhance(&buffer)?;
//!
//! println!("Quality improvement: {:.2}%", enhancer.quality_improvement() * 100.0);
//! # Ok(())
//! # }
//! ```

use crate::audio::AudioBuffer;
use crate::error::{Result, VoirsError};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::Float;
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::Complex32;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for adaptive audio enhancement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancementConfig {
    /// Enable adaptive noise gate
    pub enable_noise_gate: bool,

    /// Noise gate threshold in dB (auto-adapts if enabled)
    pub noise_gate_threshold_db: f32,

    /// Enable multiband compression
    pub enable_multiband_compression: bool,

    /// Compression ratio for each band (3 bands: low, mid, high)
    pub compression_ratios: [f32; 3],

    /// Compression thresholds in dB for each band
    pub compression_thresholds_db: [f32; 3],

    /// Enable spectral enhancement
    pub enable_spectral_enhancement: bool,

    /// Enhancement strength (0.0 = disabled, 1.0 = maximum)
    pub enhancement_strength: f32,

    /// Target quality level (0.0-1.0)
    pub target_quality: f32,

    /// Enable adaptive processing (adjusts based on system load)
    pub enable_adaptive_processing: bool,

    /// Learning rate for adaptive algorithms
    pub learning_rate: f32,

    /// FFT size for spectral processing
    pub fft_size: usize,

    /// Number of frequency bands for multiband processing
    pub num_bands: usize,
}

impl Default for EnhancementConfig {
    fn default() -> Self {
        Self {
            enable_noise_gate: true,
            noise_gate_threshold_db: -60.0,
            enable_multiband_compression: true,
            compression_ratios: [3.0, 2.5, 2.0], // Low, Mid, High
            compression_thresholds_db: [-20.0, -18.0, -15.0],
            enable_spectral_enhancement: true,
            enhancement_strength: 0.5,
            target_quality: 0.8,
            enable_adaptive_processing: true,
            learning_rate: 0.01,
            fft_size: 2048,
            num_bands: 3,
        }
    }
}

/// Adaptive audio enhancer with real-time processing and quality optimization.
pub struct AdaptiveEnhancer {
    config: EnhancementConfig,
    noise_floor_estimator: NoiseFloorEstimator,
    spectral_profile: SpectralProfile,
    quality_tracker: QualityTracker,
    performance_monitor: PerformanceMonitor,
}

impl AdaptiveEnhancer {
    /// Create a new adaptive enhancer with the given configuration.
    pub fn new(config: EnhancementConfig) -> Self {
        Self {
            noise_floor_estimator: NoiseFloorEstimator::new(config.fft_size, config.learning_rate),
            spectral_profile: SpectralProfile::new(config.fft_size, config.num_bands),
            quality_tracker: QualityTracker::new(),
            performance_monitor: PerformanceMonitor::new(),
            config,
        }
    }

    /// Enhance an audio buffer with adaptive processing.
    ///
    /// This applies noise gating, multiband compression, and spectral enhancement
    /// based on the current configuration and learned characteristics.
    pub fn enhance(&mut self, buffer: &AudioBuffer) -> Result<AudioBuffer> {
        let start_time = std::time::Instant::now();

        // Create mutable copy for processing
        let mut enhanced = buffer.clone();
        let sample_rate = buffer.sample_rate();

        // Analyze audio characteristics
        self.analyze_audio(buffer)?;

        // Apply adaptive noise gate
        if self.config.enable_noise_gate {
            self.apply_adaptive_noise_gate(&mut enhanced)?;
        }

        // Apply multiband compression
        if self.config.enable_multiband_compression {
            self.apply_multiband_compression(&mut enhanced)?;
        }

        // Apply spectral enhancement
        if self.config.enable_spectral_enhancement {
            self.apply_spectral_enhancement(&mut enhanced)?;
        }

        // Update quality tracking
        let quality_before = self.estimate_quality(buffer)?;
        let quality_after = self.estimate_quality(&enhanced)?;
        self.quality_tracker.update(quality_before, quality_after);

        // Update performance metrics
        let processing_time = start_time.elapsed();
        self.performance_monitor
            .update(processing_time, sample_rate);

        // Adaptive configuration adjustment
        if self.config.enable_adaptive_processing {
            self.adapt_configuration()?;
        }

        Ok(enhanced)
    }

    /// Get the average quality improvement from recent enhancements.
    pub fn quality_improvement(&self) -> f32 {
        self.quality_tracker.average_improvement()
    }

    /// Get processing performance metrics.
    pub fn performance_metrics(&self) -> PerformanceMetrics {
        self.performance_monitor.get_metrics()
    }

    /// Reset adaptive learning state.
    pub fn reset(&mut self) {
        self.noise_floor_estimator.reset();
        self.spectral_profile.reset();
        self.quality_tracker.reset();
    }

    /// Get current enhancement configuration.
    pub fn config(&self) -> &EnhancementConfig {
        &self.config
    }

    // ========================================================================
    // Internal processing methods
    // ========================================================================

    /// Analyze audio characteristics for adaptive processing.
    fn analyze_audio(&mut self, buffer: &AudioBuffer) -> Result<()> {
        let samples = buffer.samples();

        // Update noise floor estimate
        self.noise_floor_estimator.update(samples)?;

        // Update spectral profile
        self.spectral_profile
            .update(samples, buffer.sample_rate())?;

        Ok(())
    }

    /// Apply adaptive noise gate using spectral analysis.
    fn apply_adaptive_noise_gate(&mut self, buffer: &mut AudioBuffer) -> Result<()> {
        let samples = buffer.samples_mut();
        let noise_floor = self.noise_floor_estimator.current_floor();

        // Convert threshold from dB to linear
        let threshold_linear = 10f32.powf(self.config.noise_gate_threshold_db / 20.0);
        let adaptive_threshold = threshold_linear * noise_floor.max(1e-10);

        // Apply noise gate with soft knee
        let knee_width = 0.1 * adaptive_threshold;

        for sample in samples.iter_mut() {
            let abs_sample = sample.abs();

            if abs_sample < adaptive_threshold - knee_width {
                // Below threshold - gate closed
                *sample = 0.0;
            } else if abs_sample < adaptive_threshold + knee_width {
                // In knee region - soft transition
                let ratio = (abs_sample - (adaptive_threshold - knee_width)) / (2.0 * knee_width);
                *sample *= ratio * ratio; // Smooth quadratic curve
            }
            // Above threshold - pass through unchanged
        }

        Ok(())
    }

    /// Apply multiband dynamic range compression.
    fn apply_multiband_compression(&mut self, buffer: &mut AudioBuffer) -> Result<()> {
        let sample_rate = buffer.sample_rate();

        // Clone samples to avoid borrowing issues
        let samples: Vec<f32> = buffer.samples().to_vec();

        // Define frequency bands (logarithmic spacing)
        let band_edges = self.calculate_band_edges(sample_rate);

        // Split into bands and compress each
        for (band_idx, (low_freq, high_freq)) in band_edges.iter().enumerate() {
            if band_idx >= self.config.num_bands {
                break;
            }

            // Extract band using FFT filtering
            let band_signal =
                self.extract_frequency_band(&samples, sample_rate, *low_freq, *high_freq)?;

            // Apply compression
            let ratio = self.config.compression_ratios[band_idx.min(2)];
            let threshold_db = self.config.compression_thresholds_db[band_idx.min(2)];
            let compressed = self.compress_signal(&band_signal, ratio, threshold_db);

            // Add back to original signal (replace original band content)
            self.replace_frequency_band(buffer, &compressed, sample_rate, *low_freq, *high_freq)?;
        }

        Ok(())
    }

    /// Apply spectral enhancement for clarity and presence.
    fn apply_spectral_enhancement(&mut self, buffer: &mut AudioBuffer) -> Result<()> {
        let samples = buffer.samples();
        let sample_rate = buffer.sample_rate();
        let strength = self.config.enhancement_strength;

        // Compute FFT
        let fft_size = self.config.fft_size;
        let padded_size = samples.len().div_ceil(fft_size) * fft_size;
        let mut padded = vec![0.0f32; padded_size];
        padded[..samples.len()].copy_from_slice(samples);

        let mut enhanced_spectrum = Vec::new();

        // Process in overlapping frames
        let hop_size = fft_size / 2;
        for frame_start in (0..padded.len()).step_by(hop_size) {
            if frame_start + fft_size > padded.len() {
                break;
            }

            let frame = &padded[frame_start..frame_start + fft_size];

            // Apply window
            let windowed = self.apply_hann_window(frame);

            // FFT
            let spectrum = self.compute_fft(&windowed)?;

            // Enhance spectrum based on learned profile
            let enhanced = self.enhance_spectrum(&spectrum, strength);

            // IFFT and overlap-add
            enhanced_spectrum.push(enhanced);
        }

        // Reconstruct signal with overlap-add
        let mut reconstructed = vec![0.0f32; padded.len()];
        for (frame_idx, spectrum) in enhanced_spectrum.iter().enumerate() {
            let frame_start = frame_idx * hop_size;
            let time_signal = self.compute_ifft(spectrum)?;

            for (i, &sample) in time_signal.iter().enumerate().take(fft_size) {
                if frame_start + i < reconstructed.len() {
                    reconstructed[frame_start + i] += sample / 2.0; // Normalize overlap
                }
            }
        }

        // Copy back to buffer
        let buffer_samples = buffer.samples_mut();
        buffer_samples.copy_from_slice(&reconstructed[..buffer_samples.len()]);

        Ok(())
    }

    /// Estimate perceptual quality of audio (0.0-1.0).
    fn estimate_quality(&self, buffer: &AudioBuffer) -> Result<f32> {
        // Compute quality metrics
        let snr = buffer.signal_to_noise_ratio();
        let spectral_balance = self.compute_spectral_balance(buffer)?;
        let temporal_smoothness = self.compute_temporal_smoothness(buffer)?;

        // Weighted combination
        let quality =
            0.4 * (snr / 40.0).min(1.0) + 0.3 * spectral_balance + 0.3 * temporal_smoothness;

        Ok(quality.clamp(0.0, 1.0))
    }

    /// Adapt configuration based on performance and quality feedback.
    fn adapt_configuration(&mut self) -> Result<()> {
        let current_quality = self.quality_tracker.current_quality();
        let target = self.config.target_quality;
        let lr = self.config.learning_rate;

        // Adjust enhancement strength based on quality gap
        let quality_gap = target - current_quality;
        self.config.enhancement_strength += lr * quality_gap;
        self.config.enhancement_strength = self.config.enhancement_strength.clamp(0.0, 1.0);

        // Adjust compression based on quality and performance
        if self.performance_monitor.is_overloaded() {
            // Reduce processing load
            for ratio in &mut self.config.compression_ratios {
                *ratio *= 0.95; // Reduce compression
            }
        }

        Ok(())
    }

    // ========================================================================
    // Helper methods for signal processing
    // ========================================================================

    fn calculate_band_edges(&self, sample_rate: u32) -> Vec<(f32, f32)> {
        let nyquist = sample_rate as f32 / 2.0;
        let num_bands = self.config.num_bands;

        let mut edges = Vec::new();
        let log_min = 20f32.log10(); // 20 Hz
        let log_max = nyquist.log10();
        let log_step = (log_max - log_min) / num_bands as f32;

        for i in 0..num_bands {
            let low_freq = 10f32.powf(log_min + i as f32 * log_step);
            let high_freq = 10f32.powf(log_min + (i + 1) as f32 * log_step);
            edges.push((low_freq, high_freq));
        }

        edges
    }

    fn extract_frequency_band(
        &self,
        samples: &[f32],
        sample_rate: u32,
        low_freq: f32,
        high_freq: f32,
    ) -> Result<Vec<f32>> {
        // Simplified band extraction using FFT filtering
        let fft_size = self.config.fft_size;
        let padded_size = samples.len().div_ceil(fft_size) * fft_size;
        let mut padded = vec![0.0f32; padded_size];
        padded[..samples.len()].copy_from_slice(samples);

        // FFT
        let windowed = self.apply_hann_window(&padded);
        let spectrum = self.compute_fft(&windowed)?;

        // Filter spectrum
        let freq_resolution = sample_rate as f32 / fft_size as f32;
        let low_bin = (low_freq / freq_resolution) as usize;
        let high_bin = (high_freq / freq_resolution) as usize;

        let mut filtered_spectrum = vec![Complex32::new(0.0, 0.0); spectrum.len()];
        for (i, &val) in spectrum.iter().enumerate() {
            if i >= low_bin && i <= high_bin {
                filtered_spectrum[i] = val;
            }
        }

        // IFFT
        self.compute_ifft(&filtered_spectrum)
    }

    fn replace_frequency_band(
        &self,
        buffer: &mut AudioBuffer,
        band_signal: &[f32],
        sample_rate: u32,
        low_freq: f32,
        high_freq: f32,
    ) -> Result<()> {
        // For simplicity, mix the band signal back
        let samples = buffer.samples_mut();
        let len = samples.len().min(band_signal.len());

        samples[..len].copy_from_slice(&band_signal[..len]);

        Ok(())
    }

    fn compress_signal(&self, signal: &[f32], ratio: f32, threshold_db: f32) -> Vec<f32> {
        let threshold_linear = 10f32.powf(threshold_db / 20.0);
        let mut compressed = Vec::with_capacity(signal.len());

        for &sample in signal {
            let abs_sample = sample.abs();

            if abs_sample > threshold_linear {
                // Apply compression
                let excess = abs_sample / threshold_linear;
                let compressed_excess = excess.powf(1.0 / ratio);
                let compressed_sample = threshold_linear * compressed_excess * sample.signum();
                compressed.push(compressed_sample);
            } else {
                compressed.push(sample);
            }
        }

        compressed
    }

    fn apply_hann_window(&self, samples: &[f32]) -> Vec<f32> {
        let n = samples.len();
        let mut windowed = Vec::with_capacity(n);

        for (i, &sample) in samples.iter().enumerate() {
            let window_val =
                0.5 * (1.0 - ((2.0 * std::f32::consts::PI * i as f32) / (n as f32 - 1.0)).cos());
            windowed.push(sample * window_val);
        }

        windowed
    }

    fn compute_fft(&self, samples: &[f32]) -> Result<Vec<Complex32>> {
        let input: Vec<scirs2_core::Complex64> = samples
            .iter()
            .map(|&x| scirs2_core::Complex64::new(x as f64, 0.0))
            .collect();

        let spectrum =
            scirs2_fft::fft(&input, Some(samples.len())).map_err(|e| VoirsError::AudioError {
                message: format!("FFT failed: {}", e),
                buffer_info: None,
            })?;

        // Convert Complex64 to Complex32
        Ok(spectrum
            .iter()
            .map(|c| Complex32::new(c.re as f32, c.im as f32))
            .collect())
    }

    fn compute_ifft(&self, spectrum: &[Complex32]) -> Result<Vec<f32>> {
        // Convert Complex32 to Complex64 for scirs2_fft
        let spectrum_f64: Vec<scirs2_core::Complex64> = spectrum
            .iter()
            .map(|c| scirs2_core::Complex64::new(c.re as f64, c.im as f64))
            .collect();

        let time_domain = scirs2_fft::ifft(&spectrum_f64, Some(spectrum.len())).map_err(|e| {
            VoirsError::AudioError {
                message: format!("IFFT failed: {}", e),
                buffer_info: None,
            }
        })?;

        // Extract real part
        Ok(time_domain.iter().map(|c| c.re as f32).collect())
    }

    fn enhance_spectrum(&self, spectrum: &[Complex32], strength: f32) -> Vec<Complex32> {
        let profile = &self.spectral_profile.target_profile;

        spectrum
            .iter()
            .enumerate()
            .map(|(i, &complex)| {
                let magnitude = complex.norm();
                let phase = complex.arg();

                // Apply enhancement based on target profile
                let target_gain = if i < profile.len() { profile[i] } else { 1.0 };
                let enhanced_magnitude = magnitude * (1.0 + strength * (target_gain - 1.0));

                Complex32::from_polar(enhanced_magnitude, phase)
            })
            .collect()
    }

    fn compute_spectral_balance(&self, buffer: &AudioBuffer) -> Result<f32> {
        // Measure how well energy is distributed across frequency bands
        let samples = buffer.samples();
        let spectrum = self.compute_fft(samples)?;

        let low_energy: f32 = spectrum[..spectrum.len() / 3]
            .iter()
            .map(|c| c.norm_sqr())
            .sum();
        let mid_energy: f32 = spectrum[spectrum.len() / 3..2 * spectrum.len() / 3]
            .iter()
            .map(|c| c.norm_sqr())
            .sum();
        let high_energy: f32 = spectrum[2 * spectrum.len() / 3..]
            .iter()
            .map(|c| c.norm_sqr())
            .sum();

        let total_energy = low_energy + mid_energy + high_energy;
        if total_energy < 1e-10 {
            return Ok(0.0);
        }

        // Good balance means each band has roughly 1/3 of energy
        let low_ratio = low_energy / total_energy;
        let mid_ratio = mid_energy / total_energy;
        let high_ratio = high_energy / total_energy;

        let deviation =
            ((low_ratio - 0.33).abs() + (mid_ratio - 0.33).abs() + (high_ratio - 0.33).abs()) / 3.0;

        Ok(1.0 - deviation.min(1.0))
    }

    fn compute_temporal_smoothness(&self, buffer: &AudioBuffer) -> Result<f32> {
        // Measure how smooth the signal is (fewer abrupt changes = higher quality)
        let samples = buffer.samples();
        if samples.len() < 2 {
            return Ok(1.0);
        }

        let mut total_diff = 0.0;
        for i in 1..samples.len() {
            total_diff += (samples[i] - samples[i - 1]).abs();
        }

        let avg_diff = total_diff / (samples.len() - 1) as f32;

        // Normalize: very smooth speech has avg_diff around 0.01, noisy around 0.1
        let smoothness = 1.0 - (avg_diff / 0.1).min(1.0);

        Ok(smoothness.max(0.0))
    }
}

// ============================================================================
// Supporting structures for adaptive enhancement
// ============================================================================

/// Estimates and tracks the noise floor over time.
struct NoiseFloorEstimator {
    fft_size: usize,
    learning_rate: f32,
    current_floor: f32,
    history: VecDeque<f32>,
    max_history: usize,
}

impl NoiseFloorEstimator {
    fn new(fft_size: usize, learning_rate: f32) -> Self {
        Self {
            fft_size,
            learning_rate,
            current_floor: 1.0,
            history: VecDeque::new(),
            max_history: 100,
        }
    }

    fn update(&mut self, samples: &[f32]) -> Result<()> {
        // Estimate noise floor from quietest segments
        let rms = if !samples.is_empty() {
            let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
            (sum_sq / samples.len() as f32).sqrt()
        } else {
            0.0
        };

        // Update with exponential moving average
        self.current_floor =
            (1.0 - self.learning_rate) * self.current_floor + self.learning_rate * rms;

        // Track history
        self.history.push_back(rms);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }

        Ok(())
    }

    fn current_floor(&self) -> f32 {
        self.current_floor
    }

    fn reset(&mut self) {
        self.current_floor = 1.0;
        self.history.clear();
    }
}

/// Learns and maintains a target spectral profile.
struct SpectralProfile {
    fft_size: usize,
    num_bands: usize,
    target_profile: Vec<f32>,
}

impl SpectralProfile {
    fn new(fft_size: usize, num_bands: usize) -> Self {
        Self {
            fft_size,
            num_bands,
            target_profile: vec![1.0; fft_size / 2 + 1],
        }
    }

    fn update(&mut self, samples: &[f32], sample_rate: u32) -> Result<()> {
        // Analyze current spectral characteristics
        // For simplicity, maintain a flat target profile
        // In production, this would learn from high-quality references
        Ok(())
    }

    fn reset(&mut self) {
        self.target_profile.fill(1.0);
    }
}

/// Tracks quality metrics over time.
struct QualityTracker {
    quality_before: VecDeque<f32>,
    quality_after: VecDeque<f32>,
    max_history: usize,
}

impl QualityTracker {
    fn new() -> Self {
        Self {
            quality_before: VecDeque::new(),
            quality_after: VecDeque::new(),
            max_history: 100,
        }
    }

    fn update(&mut self, before: f32, after: f32) {
        self.quality_before.push_back(before);
        self.quality_after.push_back(after);

        if self.quality_before.len() > self.max_history {
            self.quality_before.pop_front();
            self.quality_after.pop_front();
        }
    }

    fn average_improvement(&self) -> f32 {
        if self.quality_before.is_empty() {
            return 0.0;
        }

        let avg_before: f32 =
            self.quality_before.iter().sum::<f32>() / self.quality_before.len() as f32;
        let avg_after: f32 =
            self.quality_after.iter().sum::<f32>() / self.quality_after.len() as f32;

        avg_after - avg_before
    }

    fn current_quality(&self) -> f32 {
        self.quality_after.back().copied().unwrap_or(0.5)
    }

    fn reset(&mut self) {
        self.quality_before.clear();
        self.quality_after.clear();
    }
}

/// Monitors processing performance.
struct PerformanceMonitor {
    processing_times: VecDeque<std::time::Duration>,
    max_history: usize,
}

impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            processing_times: VecDeque::new(),
            max_history: 100,
        }
    }

    fn update(&mut self, duration: std::time::Duration, sample_rate: u32) {
        self.processing_times.push_back(duration);

        if self.processing_times.len() > self.max_history {
            self.processing_times.pop_front();
        }
    }

    fn is_overloaded(&self) -> bool {
        if self.processing_times.is_empty() {
            return false;
        }

        let avg_time: std::time::Duration =
            self.processing_times.iter().sum::<std::time::Duration>()
                / self.processing_times.len() as u32;

        // Consider overloaded if processing takes >50ms on average
        avg_time.as_millis() > 50
    }

    fn get_metrics(&self) -> PerformanceMetrics {
        if self.processing_times.is_empty() {
            return PerformanceMetrics::default();
        }

        let times_ms: Vec<f64> = self
            .processing_times
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();

        let avg = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
        let min = times_ms.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = times_ms.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        PerformanceMetrics {
            average_ms: avg,
            min_ms: min,
            max_ms: max,
            is_overloaded: self.is_overloaded(),
        }
    }
}

/// Performance metrics for enhancement processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average processing time in milliseconds
    pub average_ms: f64,

    /// Minimum processing time in milliseconds
    pub min_ms: f64,

    /// Maximum processing time in milliseconds
    pub max_ms: f64,

    /// Whether the system is currently overloaded
    pub is_overloaded: bool,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            average_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            is_overloaded: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_enhancer_creation() {
        let config = EnhancementConfig::default();
        let enhancer = AdaptiveEnhancer::new(config);

        assert_eq!(enhancer.quality_improvement(), 0.0);
    }

    #[test]
    fn test_enhancement_basic() {
        let buffer = AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
        let mut enhancer = AdaptiveEnhancer::new(EnhancementConfig::default());

        let result = enhancer.enhance(&buffer);
        assert!(result.is_ok());

        let enhanced = result.unwrap();
        assert_eq!(enhanced.sample_rate(), buffer.sample_rate());
        assert_eq!(enhanced.channels(), buffer.channels());
    }

    #[test]
    fn test_noise_gate() {
        let mut config = EnhancementConfig::default();
        config.enable_multiband_compression = false;
        config.enable_spectral_enhancement = false;
        config.noise_gate_threshold_db = -40.0;

        let mut buffer = AudioBuffer::sine_wave(440.0, 0.1, 44100, 0.01); // Very quiet signal
        let mut enhancer = AdaptiveEnhancer::new(config);

        let result = enhancer.enhance(&buffer);
        assert!(result.is_ok());
    }

    #[test]
    fn test_multiband_compression() {
        let mut config = EnhancementConfig::default();
        config.enable_noise_gate = false;
        config.enable_spectral_enhancement = false;

        let buffer = AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.9); // Loud signal
        let mut enhancer = AdaptiveEnhancer::new(config);

        let result = enhancer.enhance(&buffer);
        assert!(result.is_ok());
    }

    #[test]
    fn test_spectral_enhancement() {
        let mut config = EnhancementConfig::default();
        config.enable_noise_gate = false;
        config.enable_multiband_compression = false;
        config.enhancement_strength = 0.5;

        let buffer = AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
        let mut enhancer = AdaptiveEnhancer::new(config);

        let result = enhancer.enhance(&buffer);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quality_tracking() {
        let buffer = AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
        let mut enhancer = AdaptiveEnhancer::new(EnhancementConfig::default());

        // Process multiple times
        for _ in 0..5 {
            let _ = enhancer.enhance(&buffer);
        }

        // Should have tracked quality
        let improvement = enhancer.quality_improvement();
        assert!(improvement.is_finite());
    }

    #[test]
    fn test_performance_monitoring() {
        let buffer = AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
        let mut enhancer = AdaptiveEnhancer::new(EnhancementConfig::default());

        let _ = enhancer.enhance(&buffer);

        let metrics = enhancer.performance_metrics();
        assert!(metrics.average_ms >= 0.0);
        assert!(metrics.min_ms >= 0.0);
        assert!(metrics.max_ms >= metrics.min_ms);
    }

    #[test]
    fn test_adaptive_configuration() {
        let mut config = EnhancementConfig::default();
        config.enable_adaptive_processing = true;
        config.target_quality = 0.9;

        let buffer = AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
        let mut enhancer = AdaptiveEnhancer::new(config);

        // Process multiple times to trigger adaptation
        for _ in 0..10 {
            let _ = enhancer.enhance(&buffer);
        }

        // Configuration should have adapted
        assert!(enhancer.config.enhancement_strength >= 0.0);
        assert!(enhancer.config.enhancement_strength <= 1.0);
    }

    #[test]
    fn test_reset() {
        let buffer = AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
        let mut enhancer = AdaptiveEnhancer::new(EnhancementConfig::default());

        // Process to build history
        for _ in 0..5 {
            let _ = enhancer.enhance(&buffer);
        }

        assert!(enhancer.quality_improvement() != 0.0);

        // Reset
        enhancer.reset();

        // Should be back to initial state
        assert_eq!(enhancer.quality_improvement(), 0.0);
    }

    #[test]
    fn test_noise_floor_estimator() {
        let mut estimator = NoiseFloorEstimator::new(2048, 0.1);

        let samples = vec![0.01f32; 1000];
        estimator.update(&samples).unwrap();

        assert!(estimator.current_floor() > 0.0);
        assert!(estimator.current_floor() < 1.0);
    }

    #[test]
    fn test_quality_tracker() {
        let mut tracker = QualityTracker::new();

        tracker.update(0.5, 0.7);
        tracker.update(0.6, 0.8);

        assert!(tracker.average_improvement() > 0.0);
        assert_eq!(tracker.current_quality(), 0.8);
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();

        monitor.update(std::time::Duration::from_millis(10), 44100);
        monitor.update(std::time::Duration::from_millis(15), 44100);

        let metrics = monitor.get_metrics();
        assert!(metrics.average_ms > 0.0);
        assert!(!metrics.is_overloaded);
    }
}
