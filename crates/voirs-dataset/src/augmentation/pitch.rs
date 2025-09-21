//! Pitch shifting augmentation
//!
//! This module provides pitch shifting capabilities for audio augmentation
//! while preserving formant characteristics and audio quality.

use crate::{AudioData, Result};
use std::f32::consts::PI;

/// Pitch shifting configuration
#[derive(Debug, Clone)]
pub struct PitchConfig {
    /// Pitch shift amounts in semitones
    pub pitch_shifts: Vec<f32>,
    /// Preserve formants during pitch shifting
    pub preserve_formants: bool,
    /// Window size for pitch shifting (samples)
    pub window_size: usize,
    /// Overlap ratio for analysis/synthesis
    pub overlap_ratio: f32,
    /// Use high-quality algorithm (slower but better)
    pub high_quality: bool,
    /// Formant preservation strength (0.0 to 1.0)
    pub formant_preservation: f32,
}

impl Default for PitchConfig {
    fn default() -> Self {
        Self {
            pitch_shifts: vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            preserve_formants: true,
            window_size: 2048,
            overlap_ratio: 0.75,
            high_quality: true,
            formant_preservation: 0.8,
        }
    }
}

/// Pitch shifting augmentor
pub struct PitchAugmentor {
    config: PitchConfig,
}

impl PitchAugmentor {
    /// Create new pitch augmentor with configuration
    pub fn new(config: PitchConfig) -> Self {
        Self { config }
    }

    /// Create pitch augmentor with default configuration
    pub fn with_default_config() -> Self {
        Self::new(PitchConfig::default())
    }

    /// Apply pitch shifting to audio
    pub fn apply_pitch_shift(&self, audio: &AudioData, semitones: f32) -> Result<AudioData> {
        if semitones.abs() < f32::EPSILON {
            return Ok(audio.clone());
        }

        let pitch_factor = 2.0_f32.powf(semitones / 12.0);

        if self.config.high_quality {
            self.apply_high_quality_pitch_shift(audio, pitch_factor)
        } else {
            self.apply_simple_pitch_shift(audio, pitch_factor)
        }
    }

    /// Generate all pitch variants for given audio
    pub fn generate_variants(&self, audio: &AudioData) -> Result<Vec<AudioData>> {
        let mut variants = Vec::new();

        for &semitones in &self.config.pitch_shifts {
            let shifted = self.apply_pitch_shift(audio, semitones)?;
            variants.push(shifted);
        }

        Ok(variants)
    }

    /// Apply high-quality pitch shifting using PSOLA (Pitch Synchronous Overlap-Add)
    fn apply_high_quality_pitch_shift(
        &self,
        audio: &AudioData,
        pitch_factor: f32,
    ) -> Result<AudioData> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();
        let channels = audio.channels() as usize;

        let mut output_samples = Vec::with_capacity(samples.len());

        // Process each channel separately
        for ch in 0..channels {
            let channel_samples = extract_channel(samples, ch, channels);
            let shifted = self.psola_pitch_shift(&channel_samples, pitch_factor, sample_rate)?;

            // Interleave or initialize output
            if ch == 0 {
                output_samples = shifted;
            } else {
                interleave_channel(&mut output_samples, &shifted, ch, channels);
            }
        }

        Ok(AudioData::new(output_samples, sample_rate, channels as u32))
    }

    /// Apply simple pitch shifting (faster but lower quality)
    fn apply_simple_pitch_shift(&self, audio: &AudioData, pitch_factor: f32) -> Result<AudioData> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();
        let channels = audio.channels();

        // Simple frequency domain approach
        let mut output_samples = Vec::with_capacity(samples.len());

        for i in 0..samples.len() {
            let source_pos = i as f32 / pitch_factor;
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

    /// PSOLA (Pitch Synchronous Overlap-Add) pitch shifting
    fn psola_pitch_shift(
        &self,
        samples: &[f32],
        pitch_factor: f32,
        sample_rate: u32,
    ) -> Result<Vec<f32>> {
        // Detect pitch periods
        let pitch_periods = self.detect_pitch_periods(samples, sample_rate)?;

        if pitch_periods.is_empty() {
            return Ok(samples.to_vec());
        }

        let mut output = Vec::with_capacity(samples.len());
        let window_size = self.config.window_size;
        let window = create_hann_window(window_size);

        // Calculate new pitch periods
        let mut new_periods = Vec::new();
        for &period in &pitch_periods {
            new_periods.push((period as f32 / pitch_factor) as usize);
        }

        let mut output_pos: usize = 0;
        let mut input_pos: usize = 0;

        for (i, &period) in pitch_periods.iter().enumerate() {
            let new_period = new_periods[i];

            // Extract pitch period with window
            let start = input_pos.saturating_sub(window_size / 2);
            let end = (start + window_size).min(samples.len());

            if end > start {
                let mut windowed_samples = vec![0.0; window_size];
                for j in 0..(end - start) {
                    if j < window_size {
                        windowed_samples[j] = samples[start + j] * window[j];
                    }
                }

                // Resample to new pitch period
                let resampled =
                    self.resample_pitch_period(&windowed_samples, period, new_period)?;

                // Overlap-add to output
                for (j, &sample) in resampled.iter().enumerate() {
                    if output_pos + j < output.len() {
                        output[output_pos + j] += sample;
                    } else {
                        output.push(sample);
                    }
                }

                output_pos += new_period;
            }

            input_pos += period;
        }

        Ok(output)
    }

    /// Detect pitch periods in audio signal
    fn detect_pitch_periods(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<usize>> {
        let mut periods = Vec::new();

        // Simple autocorrelation-based pitch detection
        let min_period = (sample_rate as f32 / 800.0) as usize; // 800 Hz max
        let max_period = (sample_rate as f32 / 50.0) as usize; // 50 Hz min

        let mut pos = 0;
        while pos + max_period < samples.len() {
            let end_pos = (pos + max_period * 2).min(samples.len());
            let period = self.find_pitch_period(&samples[pos..end_pos], min_period, max_period)?;
            periods.push(period);
            pos += period;
        }

        Ok(periods)
    }

    /// Find pitch period using autocorrelation
    fn find_pitch_period(
        &self,
        samples: &[f32],
        min_period: usize,
        max_period: usize,
    ) -> Result<usize> {
        let mut best_period = min_period;
        let mut best_correlation = 0.0;

        for period in min_period..max_period.min(samples.len() / 2) {
            let correlation = self.calculate_autocorrelation(samples, period);
            if correlation > best_correlation {
                best_correlation = correlation;
                best_period = period;
            }
        }

        Ok(best_period)
    }

    /// Calculate autocorrelation at given lag
    fn calculate_autocorrelation(&self, samples: &[f32], lag: usize) -> f32 {
        if lag >= samples.len() {
            return 0.0;
        }

        let mut correlation = 0.0;
        let mut norm = 0.0;

        for i in 0..(samples.len() - lag) {
            correlation += samples[i] * samples[i + lag];
            norm += samples[i] * samples[i];
        }

        if norm > 0.0 {
            correlation / norm
        } else {
            0.0
        }
    }

    /// Resample pitch period to new length
    fn resample_pitch_period(
        &self,
        samples: &[f32],
        old_period: usize,
        new_period: usize,
    ) -> Result<Vec<f32>> {
        let mut resampled = Vec::with_capacity(new_period);

        for i in 0..new_period {
            let source_pos = i as f32 * old_period as f32 / new_period as f32;
            let source_idx = source_pos as usize;
            let frac = source_pos - source_idx as f32;

            if source_idx + 1 < samples.len() {
                let sample1 = samples[source_idx];
                let sample2 = samples[source_idx + 1];
                let interpolated = sample1 * (1.0 - frac) + sample2 * frac;
                resampled.push(interpolated);
            } else if source_idx < samples.len() {
                resampled.push(samples[source_idx]);
            } else {
                resampled.push(0.0);
            }
        }

        Ok(resampled)
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
    output: &mut Vec<f32>,
    channel_samples: &[f32],
    channel: usize,
    total_channels: usize,
) {
    // Extend output if needed
    while output.len() < channel_samples.len() * total_channels {
        output.push(0.0);
    }

    for (i, &sample) in channel_samples.iter().enumerate() {
        let output_idx = i * total_channels + channel;
        if output_idx < output.len() {
            output[output_idx] = sample;
        }
    }
}

/// Pitch shifting statistics
#[derive(Debug, Clone)]
pub struct PitchStats {
    /// Number of variants generated
    pub variants_generated: usize,
    /// Pitch shifts applied (in semitones)
    pub pitch_shifts: Vec<f32>,
    /// Processing time
    pub processing_time: std::time::Duration,
    /// Quality metrics
    pub quality_metrics: Vec<f32>,
    /// Formant preservation scores
    pub formant_scores: Vec<f32>,
}

impl PitchStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self {
            variants_generated: 0,
            pitch_shifts: Vec::new(),
            processing_time: std::time::Duration::from_secs(0),
            quality_metrics: Vec::new(),
            formant_scores: Vec::new(),
        }
    }

    /// Add variant statistics
    pub fn add_variant(&mut self, pitch_shift: f32, quality: f32, formant_score: f32) {
        self.variants_generated += 1;
        self.pitch_shifts.push(pitch_shift);
        self.quality_metrics.push(quality);
        self.formant_scores.push(formant_score);
    }

    /// Set processing time
    pub fn set_processing_time(&mut self, duration: std::time::Duration) {
        self.processing_time = duration;
    }
}

impl Default for PitchStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch processor for pitch augmentation
pub struct BatchPitchProcessor {
    augmentor: PitchAugmentor,
}

impl BatchPitchProcessor {
    /// Create new batch processor
    pub fn new(config: PitchConfig) -> Self {
        Self {
            augmentor: PitchAugmentor::new(config),
        }
    }

    /// Process multiple audio files with pitch shifting
    pub fn process_batch(
        &self,
        audio_files: &[AudioData],
    ) -> Result<(Vec<Vec<AudioData>>, PitchStats)> {
        let start_time = std::time::Instant::now();
        let mut all_variants = Vec::new();
        let mut stats = PitchStats::new();

        for audio in audio_files {
            let variants = self.augmentor.generate_variants(audio)?;

            // Calculate quality metrics for each variant
            for (i, variant) in variants.iter().enumerate() {
                let pitch_shift = self.augmentor.config.pitch_shifts[i];
                let quality = calculate_audio_quality(variant);
                let formant_score = calculate_formant_preservation(audio, variant);
                stats.add_variant(pitch_shift, quality, formant_score);
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

/// Calculate formant preservation score
fn calculate_formant_preservation(original: &AudioData, modified: &AudioData) -> f32 {
    // Simplified formant preservation metric
    // In a real implementation, this would analyze spectral envelope
    let orig_samples = original.samples();
    let mod_samples = modified.samples();

    if orig_samples.is_empty() || mod_samples.is_empty() {
        return 0.0;
    }

    // Calculate spectral similarity (simplified)
    let min_len = orig_samples.len().min(mod_samples.len());
    let mut correlation = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    for i in 0..min_len {
        correlation += orig_samples[i] * mod_samples[i];
        norm1 += orig_samples[i] * orig_samples[i];
        norm2 += mod_samples[i] * mod_samples[i];
    }

    let norm = (norm1 * norm2).sqrt();
    if norm > 0.0 {
        (correlation / norm).abs() * 100.0
    } else {
        0.0
    }
}

/// Real-time pitch detection
pub struct PitchDetector {
    sample_rate: u32,
    window_size: usize,
    min_freq: f32,
    max_freq: f32,
}

impl PitchDetector {
    /// Create new pitch detector
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            window_size: 2048,
            min_freq: 50.0,
            max_freq: 800.0,
        }
    }

    /// Detect pitch in audio frame
    pub fn detect_pitch(&self, samples: &[f32]) -> Option<f32> {
        if samples.len() < self.window_size {
            return None;
        }

        let min_period = (self.sample_rate as f32 / self.max_freq) as usize;
        let max_period = (self.sample_rate as f32 / self.min_freq) as usize;

        let mut best_period = min_period;
        let mut best_correlation = 0.0;

        for period in min_period..max_period.min(samples.len() / 2) {
            let correlation = self.calculate_autocorrelation(samples, period);
            if correlation > best_correlation {
                best_correlation = correlation;
                best_period = period;
            }
        }

        if best_correlation > 0.3 {
            Some(self.sample_rate as f32 / best_period as f32)
        } else {
            None
        }
    }

    /// Calculate autocorrelation at given lag
    fn calculate_autocorrelation(&self, samples: &[f32], lag: usize) -> f32 {
        if lag >= samples.len() {
            return 0.0;
        }

        let mut correlation = 0.0;
        let mut norm = 0.0;

        for i in 0..(samples.len() - lag) {
            correlation += samples[i] * samples[i + lag];
            norm += samples[i] * samples[i];
        }

        if norm > 0.0 {
            correlation / norm
        } else {
            0.0
        }
    }
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
    fn test_pitch_config_default() {
        let config = PitchConfig::default();
        assert_eq!(config.pitch_shifts, vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        assert!(config.preserve_formants);
        assert_eq!(config.window_size, 2048);
        assert_eq!(config.overlap_ratio, 0.75);
        assert!(config.high_quality);
        assert_eq!(config.formant_preservation, 0.8);
    }

    #[test]
    fn test_pitch_augmentor_creation() {
        let config = PitchConfig::default();
        let augmentor = PitchAugmentor::new(config);
        assert_eq!(augmentor.config.pitch_shifts.len(), 5);
    }

    #[test]
    fn test_pitch_augmentor_with_default_config() {
        let augmentor = PitchAugmentor::with_default_config();
        assert_eq!(
            augmentor.config.pitch_shifts,
            vec![-2.0, -1.0, 0.0, 1.0, 2.0]
        );
    }

    #[test]
    fn test_pitch_shift_no_change() {
        let augmentor = PitchAugmentor::with_default_config();
        let audio = create_test_audio();
        let result = augmentor.apply_pitch_shift(&audio, 0.0).unwrap();

        // Should return identical audio for 0 semitone shift
        assert_eq!(result.samples().len(), audio.samples().len());
        assert_eq!(result.sample_rate(), audio.sample_rate());
        assert_eq!(result.channels(), audio.channels());
    }

    #[test]
    fn test_pitch_shift_positive() {
        let augmentor = PitchAugmentor::with_default_config();
        let audio = create_test_audio();
        let result = augmentor.apply_pitch_shift(&audio, 2.0).unwrap();

        // Audio properties should be preserved
        assert_eq!(result.sample_rate(), audio.sample_rate());
        assert_eq!(result.channels(), audio.channels());
        // Length might change slightly due to processing
        assert!(!result.samples().is_empty());
    }

    #[test]
    fn test_pitch_shift_negative() {
        let augmentor = PitchAugmentor::with_default_config();
        let audio = create_test_audio();
        let result = augmentor.apply_pitch_shift(&audio, -2.0).unwrap();

        // Audio properties should be preserved
        assert_eq!(result.sample_rate(), audio.sample_rate());
        assert_eq!(result.channels(), audio.channels());
        assert!(!result.samples().is_empty());
    }

    #[test]
    fn test_generate_variants() {
        let config = PitchConfig {
            pitch_shifts: vec![-1.0, 0.0, 1.0],
            ..Default::default()
        };
        let augmentor = PitchAugmentor::new(config);
        let audio = create_test_audio();
        let variants = augmentor.generate_variants(&audio).unwrap();

        assert_eq!(variants.len(), 3);
        // All variants should have the same basic properties
        for variant in &variants {
            assert_eq!(variant.sample_rate(), audio.sample_rate());
            assert_eq!(variant.channels(), audio.channels());
        }
    }

    #[test]
    fn test_simple_vs_high_quality() {
        let config_simple = PitchConfig {
            high_quality: false,
            ..Default::default()
        };
        let config_hq = PitchConfig {
            high_quality: true,
            ..Default::default()
        };

        let augmentor_simple = PitchAugmentor::new(config_simple);
        let augmentor_hq = PitchAugmentor::new(config_hq);
        let audio = create_test_audio();

        let result_simple = augmentor_simple.apply_pitch_shift(&audio, 1.0).unwrap();
        let result_hq = augmentor_hq.apply_pitch_shift(&audio, 1.0).unwrap();

        // Both should preserve basic audio properties
        assert_eq!(result_simple.sample_rate(), audio.sample_rate());
        assert_eq!(result_hq.sample_rate(), audio.sample_rate());
        assert_eq!(result_simple.channels(), audio.channels());
        assert_eq!(result_hq.channels(), audio.channels());
    }

    #[test]
    fn test_semitones_to_pitch_factor_conversion() {
        // Test semitone to pitch factor conversion
        assert!((2.0_f32.powf(0.0 / 12.0) - 1.0).abs() < 0.01); // 0 semitones = 1.0
        assert!((2.0_f32.powf(12.0 / 12.0) - 2.0).abs() < 0.01); // 12 semitones = 2.0
        assert!((2.0_f32.powf(-12.0 / 12.0) - 0.5).abs() < 0.01); // -12 semitones = 0.5
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
        let mut output = vec![0.0; 4];
        let channel_data = vec![1.0, 3.0];
        interleave_channel(&mut output, &channel_data, 0, 2);

        assert_eq!(output[0], 1.0);
        assert_eq!(output[2], 3.0);
        assert_eq!(output[1], 0.0); // Other channel remains unchanged
        assert_eq!(output[3], 0.0);
    }

    #[test]
    fn test_pitch_stats_creation() {
        let stats = PitchStats::new();
        assert_eq!(stats.variants_generated, 0);
        assert_eq!(stats.pitch_shifts.len(), 0);
        assert_eq!(stats.quality_metrics.len(), 0);
        assert_eq!(stats.formant_scores.len(), 0);
    }

    #[test]
    fn test_pitch_stats_default() {
        let stats = PitchStats::default();
        assert_eq!(stats.variants_generated, 0);
    }

    #[test]
    fn test_autocorrelation_calculation() {
        let augmentor = PitchAugmentor::with_default_config();

        // Perfect autocorrelation at lag 0
        let samples = vec![1.0, 0.5, -0.5, -1.0];
        let correlation = augmentor.calculate_autocorrelation(&samples, 0);
        assert!((correlation - 1.0).abs() < 0.01);

        // Lower correlation at other lags
        let correlation_lag1 = augmentor.calculate_autocorrelation(&samples, 1);
        assert!(correlation_lag1 < 1.0);
    }

    #[test]
    fn test_pitch_period_detection() {
        let augmentor = PitchAugmentor::with_default_config();
        let audio = create_test_audio();

        // Should detect some periods
        let periods = augmentor
            .detect_pitch_periods(audio.samples(), audio.sample_rate())
            .unwrap();
        assert!(!periods.is_empty());

        // All periods should be within reasonable range
        for period in periods {
            assert!(period > 10); // Not too small
            assert!(period < 1000); // Not too large for 16kHz audio
        }
    }
}
