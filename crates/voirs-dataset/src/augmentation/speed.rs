//! Speed perturbation augmentation
//!
//! This module provides high-quality speed perturbation for audio augmentation.
//! It uses time-stretching algorithms to change the playback speed while preserving
//! pitch characteristics.

use crate::{AudioData, Result};
use std::f32::consts::PI;

/// Speed perturbation configuration
#[derive(Debug, Clone)]
pub struct SpeedConfig {
    /// Speed factors to apply
    pub speed_factors: Vec<f32>,
    /// Preserve pitch during speed changes
    pub preserve_pitch: bool,
    /// Window size for time-stretching (samples)
    pub window_size: usize,
    /// Overlap ratio for time-stretching
    pub overlap_ratio: f32,
    /// Use high-quality algorithm (slower but better)
    pub high_quality: bool,
}

impl Default for SpeedConfig {
    fn default() -> Self {
        Self {
            speed_factors: vec![0.9, 1.0, 1.1],
            preserve_pitch: true,
            window_size: 1024,
            overlap_ratio: 0.5,
            high_quality: true,
        }
    }
}

/// Speed perturbation augmentor
pub struct SpeedAugmentor {
    config: SpeedConfig,
}

impl SpeedAugmentor {
    /// Create new speed augmentor with configuration
    pub fn new(config: SpeedConfig) -> Self {
        Self { config }
    }

    /// Create speed augmentor with default configuration
    pub fn with_default_config() -> Self {
        Self::new(SpeedConfig::default())
    }

    /// Apply speed perturbation to audio
    pub fn apply_speed_perturbation(
        &self,
        audio: &AudioData,
        speed_factor: f32,
    ) -> Result<AudioData> {
        if (speed_factor - 1.0).abs() < f32::EPSILON {
            return Ok(audio.clone());
        }

        if self.config.high_quality {
            self.apply_high_quality_stretch(audio, speed_factor)
        } else {
            self.apply_simple_stretch(audio, speed_factor)
        }
    }

    /// Generate all speed variants for given audio
    pub fn generate_variants(&self, audio: &AudioData) -> Result<Vec<AudioData>> {
        let mut variants = Vec::new();

        for &factor in &self.config.speed_factors {
            let augmented = self.apply_speed_perturbation(audio, factor)?;
            variants.push(augmented);
        }

        Ok(variants)
    }

    /// Apply high-quality time-stretching using WSOLA (Waveform Similarity Overlap-Add)
    fn apply_high_quality_stretch(
        &self,
        audio: &AudioData,
        speed_factor: f32,
    ) -> Result<AudioData> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();
        let channels = audio.channels() as usize;

        // Calculate output length
        let output_length = (samples.len() as f32 / speed_factor) as usize;
        let mut output_samples = vec![0.0; output_length];

        // Process each channel separately
        for ch in 0..channels {
            let channel_samples = extract_channel(samples, ch, channels);
            let stretched = self.wsola_stretch(&channel_samples, speed_factor)?;
            interleave_channel(&mut output_samples, &stretched, ch, channels);
        }

        Ok(AudioData::new(output_samples, sample_rate, channels as u32))
    }

    /// Apply simple time-stretching (faster but lower quality)
    fn apply_simple_stretch(&self, audio: &AudioData, speed_factor: f32) -> Result<AudioData> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();
        let channels = audio.channels();

        // Simple linear interpolation
        let output_length = (samples.len() as f32 / speed_factor) as usize;
        let mut output_samples = Vec::with_capacity(output_length);

        for i in 0..output_length {
            let source_pos = i as f32 * speed_factor;
            let source_idx = source_pos as usize;
            let frac = source_pos - source_idx as f32;

            if source_idx + 1 < samples.len() {
                let sample1 = samples[source_idx];
                let sample2 = samples[source_idx + 1];
                let interpolated = sample1 * (1.0 - frac) + sample2 * frac;
                output_samples.push(interpolated);
            } else if source_idx < samples.len() {
                output_samples.push(samples[source_idx]);
            } else {
                output_samples.push(0.0);
            }
        }

        Ok(AudioData::new(output_samples, sample_rate, channels))
    }

    /// WSOLA (Waveform Similarity Overlap-Add) time-stretching algorithm
    fn wsola_stretch(&self, samples: &[f32], speed_factor: f32) -> Result<Vec<f32>> {
        let window_size = self.config.window_size;
        let hop_size = (window_size as f32 * (1.0 - self.config.overlap_ratio)) as usize;
        let output_hop_size = (hop_size as f32 / speed_factor) as usize;

        let num_frames = samples.len() / hop_size;
        let output_length = num_frames * output_hop_size;
        let mut output = vec![0.0; output_length];

        // Create Hann window
        let window = create_hann_window(window_size);

        let mut output_pos = 0;
        let mut input_pos = 0;

        while input_pos + window_size < samples.len() && output_pos + window_size < output.len() {
            // Extract windowed frame
            let mut frame_samples = vec![0.0; window_size];
            for i in 0..window_size {
                if input_pos + i < samples.len() {
                    frame_samples[i] = samples[input_pos + i] * window[i];
                }
            }

            // Apply window and overlap-add
            for i in 0..window_size {
                if output_pos + i < output.len() {
                    output[output_pos + i] += frame_samples[i];
                }
            }

            // Find best match for next frame (simplified WSOLA)
            let search_start = input_pos + hop_size;
            let search_end = (search_start + hop_size / 2).min(samples.len() - window_size);
            let mut best_pos = search_start;
            let mut best_correlation = 0.0;

            for search_pos in search_start..search_end {
                let correlation =
                    calculate_correlation(samples, search_pos, &output, output_pos, window_size);
                if correlation > best_correlation {
                    best_correlation = correlation;
                    best_pos = search_pos;
                }
            }

            input_pos = best_pos;
            output_pos += output_hop_size;
        }

        Ok(output)
    }
}

/// Create Hann window
fn create_hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos()))
        .collect()
}

/// Extract single channel from interleaved audio
fn extract_channel(samples: &[f32], channel: usize, total_channels: usize) -> Vec<f32> {
    samples
        .iter()
        .skip(channel)
        .step_by(total_channels)
        .cloned()
        .collect()
}

/// Interleave channel back into output
fn interleave_channel(
    output: &mut [f32],
    channel_samples: &[f32],
    channel: usize,
    total_channels: usize,
) {
    for (i, &sample) in channel_samples.iter().enumerate() {
        let output_idx = i * total_channels + channel;
        if output_idx < output.len() {
            output[output_idx] = sample;
        }
    }
}

/// Calculate correlation between two audio segments
fn calculate_correlation(
    samples1: &[f32],
    pos1: usize,
    samples2: &[f32],
    pos2: usize,
    window_size: usize,
) -> f32 {
    let mut correlation = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    for i in 0..window_size {
        if pos1 + i < samples1.len() && pos2 + i < samples2.len() {
            let s1 = samples1[pos1 + i];
            let s2 = samples2[pos2 + i];
            correlation += s1 * s2;
            norm1 += s1 * s1;
            norm2 += s2 * s2;
        }
    }

    let norm = (norm1 * norm2).sqrt();
    if norm > 0.0 {
        correlation / norm
    } else {
        0.0
    }
}

/// Speed perturbation statistics
#[derive(Debug, Clone)]
pub struct SpeedStats {
    /// Number of variants generated
    pub variants_generated: usize,
    /// Speed factors applied
    pub speed_factors: Vec<f32>,
    /// Processing time
    pub processing_time: std::time::Duration,
    /// Quality metrics
    pub quality_metrics: Vec<f32>,
}

impl Default for SpeedStats {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeedStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self {
            variants_generated: 0,
            speed_factors: Vec::new(),
            processing_time: std::time::Duration::from_secs(0),
            quality_metrics: Vec::new(),
        }
    }

    /// Add variant statistics
    pub fn add_variant(&mut self, speed_factor: f32, quality: f32) {
        self.variants_generated += 1;
        self.speed_factors.push(speed_factor);
        self.quality_metrics.push(quality);
    }

    /// Set processing time
    pub fn set_processing_time(&mut self, duration: std::time::Duration) {
        self.processing_time = duration;
    }

    /// Get average quality
    pub fn average_quality(&self) -> f32 {
        if self.quality_metrics.is_empty() {
            0.0
        } else {
            self.quality_metrics.iter().sum::<f32>() / self.quality_metrics.len() as f32
        }
    }
}

/// Batch speed perturbation processor
pub struct BatchSpeedProcessor {
    augmentor: SpeedAugmentor,
}

impl BatchSpeedProcessor {
    /// Create new batch processor
    pub fn new(config: SpeedConfig) -> Self {
        Self {
            augmentor: SpeedAugmentor::new(config),
        }
    }

    /// Process multiple audio files with speed perturbation
    pub fn process_batch(
        &self,
        audio_files: &[AudioData],
    ) -> Result<(Vec<Vec<AudioData>>, SpeedStats)> {
        let start_time = std::time::Instant::now();
        let mut all_variants = Vec::new();
        let mut stats = SpeedStats::new();

        for audio in audio_files {
            let variants = self.augmentor.generate_variants(audio)?;

            // Calculate quality metrics for each variant
            for (i, variant) in variants.iter().enumerate() {
                let speed_factor = self.augmentor.config.speed_factors[i];
                let quality = calculate_audio_quality(variant);
                stats.add_variant(speed_factor, quality);
            }

            all_variants.push(variants);
        }

        let processing_time = start_time.elapsed();
        stats.set_processing_time(processing_time);

        Ok((all_variants, stats))
    }
}

/// Calculate basic audio quality metric
fn calculate_audio_quality(audio: &AudioData) -> f32 {
    let samples = audio.samples();
    if samples.is_empty() {
        return 0.0;
    }

    // Calculate signal-to-noise ratio approximation
    let energy = samples.iter().map(|&x| x * x).sum::<f32>();
    let rms = (energy / samples.len() as f32).sqrt();

    // Simple quality metric based on RMS
    (rms * 100.0).min(100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioData;

    fn create_test_audio() -> AudioData {
        // Create simple sine wave test data
        let sample_rate = 16000;
        let duration_secs = 1.0;
        let frequency = 440.0; // A4 note
        let num_samples = (sample_rate as f32 * duration_secs) as usize;

        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
            })
            .collect();

        AudioData::new(samples, sample_rate, 1)
    }

    #[test]
    fn test_speed_config_default() {
        let config = SpeedConfig::default();
        assert_eq!(config.speed_factors, vec![0.9, 1.0, 1.1]);
        assert!(config.preserve_pitch);
        assert_eq!(config.window_size, 1024);
        assert_eq!(config.overlap_ratio, 0.5);
        assert!(config.high_quality);
    }

    #[test]
    fn test_speed_augmentor_creation() {
        let config = SpeedConfig::default();
        let augmentor = SpeedAugmentor::new(config);
        assert_eq!(augmentor.config.speed_factors.len(), 3);
    }

    #[test]
    fn test_speed_augmentor_with_default_config() {
        let augmentor = SpeedAugmentor::with_default_config();
        assert_eq!(augmentor.config.speed_factors, vec![0.9, 1.0, 1.1]);
    }

    #[test]
    fn test_speed_perturbation_no_change() {
        let augmentor = SpeedAugmentor::with_default_config();
        let audio = create_test_audio();
        let result = augmentor.apply_speed_perturbation(&audio, 1.0).unwrap();

        // Should return identical audio for speed factor 1.0
        assert_eq!(result.samples().len(), audio.samples().len());
        assert_eq!(result.sample_rate(), audio.sample_rate());
        assert_eq!(result.channels(), audio.channels());
    }

    #[test]
    fn test_speed_perturbation_slowdown() {
        let augmentor = SpeedAugmentor::with_default_config();
        let audio = create_test_audio();
        let result = augmentor.apply_speed_perturbation(&audio, 0.5).unwrap();

        // Slower speed should result in longer audio
        assert!(result.samples().len() > audio.samples().len());
        assert_eq!(result.sample_rate(), audio.sample_rate());
        assert_eq!(result.channels(), audio.channels());
    }

    #[test]
    fn test_speed_perturbation_speedup() {
        let augmentor = SpeedAugmentor::with_default_config();
        let audio = create_test_audio();
        let result = augmentor.apply_speed_perturbation(&audio, 2.0).unwrap();

        // Faster speed should result in shorter audio
        assert!(result.samples().len() < audio.samples().len());
        assert_eq!(result.sample_rate(), audio.sample_rate());
        assert_eq!(result.channels(), audio.channels());
    }

    #[test]
    fn test_generate_variants() {
        let config = SpeedConfig {
            speed_factors: vec![0.8, 1.0, 1.2],
            ..Default::default()
        };
        let augmentor = SpeedAugmentor::new(config);
        let audio = create_test_audio();
        let variants = augmentor.generate_variants(&audio).unwrap();

        assert_eq!(variants.len(), 3);
        // Check that lengths are different for different speed factors
        assert_ne!(variants[0].samples().len(), variants[2].samples().len());
    }

    #[test]
    fn test_simple_vs_high_quality() {
        let config_simple = SpeedConfig {
            high_quality: false,
            ..Default::default()
        };
        let config_hq = SpeedConfig {
            high_quality: true,
            ..Default::default()
        };

        let augmentor_simple = SpeedAugmentor::new(config_simple);
        let augmentor_hq = SpeedAugmentor::new(config_hq);
        let audio = create_test_audio();

        let result_simple = augmentor_simple
            .apply_speed_perturbation(&audio, 1.5)
            .unwrap();
        let result_hq = augmentor_hq.apply_speed_perturbation(&audio, 1.5).unwrap();

        // Both should change the length, but potentially with different quality
        assert!(result_simple.samples().len() < audio.samples().len());
        assert!(result_hq.samples().len() < audio.samples().len());
    }

    #[test]
    fn test_hann_window_creation() {
        let window = create_hann_window(512);
        assert_eq!(window.len(), 512);
        // Hann window should start and end near zero
        assert!(window[0] < 0.1);
        assert!(window[511] < 0.1);
        // Maximum should be around the middle
        assert!(window[256] > 0.9);
    }

    #[test]
    fn test_extract_channel() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 samples, 2 channels
        let channel_0 = extract_channel(&samples, 0, 2);
        let channel_1 = extract_channel(&samples, 1, 2);

        assert_eq!(channel_0, vec![1.0, 3.0, 5.0]);
        assert_eq!(channel_1, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_interleave_channel() {
        let mut output = vec![0.0; 6];
        let channel_data = vec![1.0, 3.0, 5.0];
        interleave_channel(&mut output, &channel_data, 0, 2);

        assert_eq!(output[0], 1.0);
        assert_eq!(output[2], 3.0);
        assert_eq!(output[4], 5.0);
        assert_eq!(output[1], 0.0); // Other channel remains unchanged
    }

    #[test]
    fn test_correlation_calculation() {
        let samples1 = vec![1.0, 0.5, -0.5, -1.0];
        let samples2 = vec![1.0, 0.5, -0.5, -1.0];

        let correlation = calculate_correlation(&samples1, 0, &samples2, 0, 4);
        // Perfect correlation should be close to 1.0
        assert!((correlation - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_speed_stats_creation() {
        let stats = SpeedStats::new();
        assert_eq!(stats.variants_generated, 0);
        assert_eq!(stats.speed_factors.len(), 0);
        assert_eq!(stats.quality_metrics.len(), 0);
    }

    #[test]
    fn test_speed_stats_default() {
        let stats = SpeedStats::default();
        assert_eq!(stats.variants_generated, 0);
    }
}
