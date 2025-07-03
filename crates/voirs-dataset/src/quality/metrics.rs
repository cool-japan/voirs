//! Quality metrics implementation
//!
//! This module provides comprehensive quality assessment for audio data
//! including SNR, clipping detection, dynamic range analysis, and spectral quality.

use crate::{AudioData, DatasetSample, Result};
use std::collections::HashMap;

/// Comprehensive quality metrics for audio data
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Signal-to-noise ratio in dB
    pub snr: f32,
    /// Root mean square level
    pub rms: f32,
    /// Peak amplitude level
    pub peak: f32,
    /// Dynamic range in dB
    pub dynamic_range: f32,
    /// Clipping percentage (0.0 to 1.0)
    pub clipping_percentage: f32,
    /// Total harmonic distortion + noise
    pub thd_n: f32,
    /// Spectral centroid in Hz
    pub spectral_centroid: f32,
    /// Spectral rolloff in Hz
    pub spectral_rolloff: f32,
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    /// Speech activity score (0.0 to 1.0)
    pub speech_activity: f32,
    /// Audio duration in seconds
    pub duration: f32,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f32,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            snr: 0.0,
            rms: 0.0,
            peak: 0.0,
            dynamic_range: 0.0,
            clipping_percentage: 0.0,
            thd_n: 0.0,
            spectral_centroid: 0.0,
            spectral_rolloff: 0.0,
            zero_crossing_rate: 0.0,
            speech_activity: 0.0,
            duration: 0.0,
            sample_rate: 22050,
            channels: 1,
            overall_score: 0.0,
        }
    }
}

/// Quality metrics configuration
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// Clipping threshold (0.0 to 1.0)
    pub clipping_threshold: f32,
    /// SNR estimation window size
    pub snr_window_size: usize,
    /// Minimum speech activity threshold
    pub speech_threshold: f32,
    /// Spectral rolloff percentage (0.0 to 1.0)
    pub rolloff_percentage: f32,
    /// Enable detailed spectral analysis
    pub detailed_spectral: bool,
    /// Frame size for spectral analysis
    pub frame_size: usize,
    /// Hop size for spectral analysis
    pub hop_size: usize,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            clipping_threshold: 0.99,
            snr_window_size: 2048,
            speech_threshold: 0.01,
            rolloff_percentage: 0.85,
            detailed_spectral: true,
            frame_size: 2048,
            hop_size: 512,
        }
    }
}

/// Quality metrics calculator
pub struct QualityMetricsCalculator {
    config: QualityConfig,
}

impl QualityMetricsCalculator {
    /// Create new quality metrics calculator
    pub fn new(config: QualityConfig) -> Self {
        Self { config }
    }
    
    /// Create calculator with default configuration
    pub fn default() -> Self {
        Self::new(QualityConfig::default())
    }
    
    /// Calculate comprehensive quality metrics for audio data
    pub fn calculate_metrics(&self, audio: &AudioData) -> Result<QualityMetrics> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();
        let channels = audio.channels();
        
        if samples.is_empty() {
            return Ok(QualityMetrics::default());
        }
        
        let mut metrics = QualityMetrics {
            duration: samples.len() as f32 / sample_rate as f32,
            sample_rate,
            channels,
            ..Default::default()
        };
        
        // Calculate basic amplitude metrics
        self.calculate_amplitude_metrics(&mut metrics, samples)?;
        
        // Calculate noise and distortion metrics
        self.calculate_noise_metrics(&mut metrics, samples)?;
        
        // Calculate spectral metrics
        if self.config.detailed_spectral {
            self.calculate_spectral_metrics(&mut metrics, samples, sample_rate)?;
        }
        
        // Calculate speech activity
        self.calculate_speech_activity(&mut metrics, samples)?;
        
        // Calculate overall quality score
        self.calculate_overall_score(&mut metrics)?;
        
        Ok(metrics)
    }
    
    /// Calculate quality metrics for a dataset sample
    pub fn calculate_sample_metrics(&self, sample: &DatasetSample) -> Result<QualityMetrics> {
        self.calculate_metrics(&sample.audio)
    }
    
    /// Calculate amplitude-based metrics
    fn calculate_amplitude_metrics(&self, metrics: &mut QualityMetrics, samples: &[f32]) -> Result<()> {
        // RMS calculation
        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        metrics.rms = (sum_squares / samples.len() as f32).sqrt();
        
        // Peak calculation
        metrics.peak = samples.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        
        // Dynamic range calculation
        let min_amplitude = samples.iter().map(|&x| x.abs()).filter(|&x| x > 0.0).fold(1.0, f32::min);
        if min_amplitude > 0.0 && metrics.peak > 0.0 {
            metrics.dynamic_range = 20.0 * (metrics.peak / min_amplitude).log10();
        }
        
        // Clipping detection
        let clipping_count = samples.iter()
            .filter(|&&x| x.abs() >= self.config.clipping_threshold)
            .count();
        metrics.clipping_percentage = clipping_count as f32 / samples.len() as f32;
        
        Ok(())
    }
    
    /// Calculate noise and distortion metrics
    fn calculate_noise_metrics(&self, metrics: &mut QualityMetrics, samples: &[f32]) -> Result<()> {
        // SNR estimation using windowed approach
        metrics.snr = self.estimate_snr(samples)?;
        
        // THD+N estimation (simplified)
        metrics.thd_n = self.estimate_thd_n(samples)?;
        
        Ok(())
    }
    
    /// Calculate spectral metrics
    fn calculate_spectral_metrics(&self, metrics: &mut QualityMetrics, samples: &[f32], sample_rate: u32) -> Result<()> {
        // Zero crossing rate
        metrics.zero_crossing_rate = self.calculate_zcr(samples);
        
        // Spectral centroid and rolloff (simplified without FFT)
        let (centroid, rolloff) = self.calculate_spectral_features(samples, sample_rate)?;
        metrics.spectral_centroid = centroid;
        metrics.spectral_rolloff = rolloff;
        
        Ok(())
    }
    
    /// Calculate speech activity detection
    fn calculate_speech_activity(&self, metrics: &mut QualityMetrics, samples: &[f32]) -> Result<()> {
        let window_size = 1024;
        let hop_size = 512;
        let threshold = self.config.speech_threshold;
        
        let mut active_frames = 0;
        let mut total_frames = 0;
        
        for i in (0..samples.len()).step_by(hop_size) {
            let end = (i + window_size).min(samples.len());
            let window = &samples[i..end];
            
            // Calculate frame energy
            let energy: f32 = window.iter().map(|&x| x * x).sum();
            let frame_energy = energy / window.len() as f32;
            
            if frame_energy > threshold {
                active_frames += 1;
            }
            total_frames += 1;
        }
        
        metrics.speech_activity = if total_frames > 0 {
            active_frames as f32 / total_frames as f32
        } else {
            0.0
        };
        
        Ok(())
    }
    
    /// Calculate overall quality score
    fn calculate_overall_score(&self, metrics: &mut QualityMetrics) -> Result<()> {
        let mut score = 1.0;
        
        // Penalize clipping
        score *= (1.0 - metrics.clipping_percentage).max(0.0);
        
        // Penalize low SNR
        if metrics.snr < 20.0 {
            score *= (metrics.snr / 20.0).max(0.0);
        }
        
        // Penalize low speech activity
        if metrics.speech_activity < 0.5 {
            score *= metrics.speech_activity * 2.0;
        }
        
        // Penalize high THD+N
        if metrics.thd_n > 0.01 {
            score *= (0.01 / metrics.thd_n).min(1.0);
        }
        
        // Penalize very low or very high dynamic range
        if metrics.dynamic_range < 20.0 || metrics.dynamic_range > 120.0 {
            let optimal_range = 60.0;
            let deviation = (metrics.dynamic_range - optimal_range).abs();
            score *= (1.0 - deviation / optimal_range).max(0.1);
        }
        
        metrics.overall_score = score.max(0.0).min(1.0);
        
        Ok(())
    }
    
    /// Estimate signal-to-noise ratio
    fn estimate_snr(&self, samples: &[f32]) -> Result<f32> {
        if samples.len() < self.config.snr_window_size * 2 {
            return Ok(0.0);
        }
        
        // Simple SNR estimation using energy difference between loud and quiet sections
        let window_size = self.config.snr_window_size;
        let mut energy_values = Vec::new();
        
        for i in (0..samples.len()).step_by(window_size / 2) {
            let end = (i + window_size).min(samples.len());
            let window = &samples[i..end];
            let energy: f32 = window.iter().map(|&x| x * x).sum();
            energy_values.push(energy / window.len() as f32);
        }
        
        if energy_values.is_empty() {
            return Ok(0.0);
        }
        
        // Sort energies and use top 25% as signal, bottom 25% as noise
        energy_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let quarter = energy_values.len() / 4;
        
        let noise_energy: f32 = energy_values[..quarter].iter().sum::<f32>() / quarter as f32;
        let signal_energy: f32 = energy_values[energy_values.len() - quarter..].iter().sum::<f32>() / quarter as f32;
        
        if noise_energy > 0.0 {
            Ok(10.0 * (signal_energy / noise_energy).log10())
        } else {
            Ok(100.0) // Very high SNR if no noise detected
        }
    }
    
    /// Estimate total harmonic distortion + noise (simplified)
    fn estimate_thd_n(&self, samples: &[f32]) -> Result<f32> {
        // Simplified THD+N estimation using high-frequency content analysis
        let mut total_power = 0.0;
        let mut harmonic_power = 0.0;
        
        for i in 0..samples.len() {
            let power = samples[i] * samples[i];
            total_power += power;
            
            // Simple harmonic detection (very simplified)
            if i > 0 && samples[i].signum() != samples[i - 1].signum() {
                harmonic_power += power * 0.1; // Rough approximation
            }
        }
        
        if total_power > 0.0 {
            Ok(harmonic_power / total_power)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate zero crossing rate
    fn calculate_zcr(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }
        
        let mut crossings = 0;
        for i in 1..samples.len() {
            if samples[i].signum() != samples[i - 1].signum() {
                crossings += 1;
            }
        }
        
        crossings as f32 / (samples.len() - 1) as f32
    }
    
    /// Calculate spectral features (simplified without FFT)
    fn calculate_spectral_features(&self, samples: &[f32], sample_rate: u32) -> Result<(f32, f32)> {
        // Simplified spectral analysis without FFT
        // This is a basic approximation for demonstration
        
        let mut weighted_freq_sum = 0.0;
        let mut magnitude_sum = 0.0;
        let mut energy_accumulator = 0.0;
        let rolloff_threshold = self.config.rolloff_percentage;
        
        // Simplified frequency analysis using local features
        for i in 1..samples.len() {
            let magnitude = samples[i].abs();
            let frequency = (i as f32 / samples.len() as f32) * (sample_rate as f32 / 2.0);
            
            weighted_freq_sum += frequency * magnitude;
            magnitude_sum += magnitude;
            energy_accumulator += magnitude * magnitude;
        }
        
        let centroid = if magnitude_sum > 0.0 {
            weighted_freq_sum / magnitude_sum
        } else {
            0.0
        };
        
        // Simplified rolloff calculation
        let target_energy = energy_accumulator * rolloff_threshold;
        let mut accumulated_energy = 0.0;
        let mut rolloff = 0.0;
        
        for i in 1..samples.len() {
            accumulated_energy += samples[i].abs() * samples[i].abs();
            if accumulated_energy >= target_energy {
                rolloff = (i as f32 / samples.len() as f32) * (sample_rate as f32 / 2.0);
                break;
            }
        }
        
        Ok((centroid, rolloff))
    }
}

/// Batch quality metrics processor
pub struct BatchQualityProcessor {
    calculator: QualityMetricsCalculator,
}

impl BatchQualityProcessor {
    /// Create new batch processor
    pub fn new(config: QualityConfig) -> Self {
        Self {
            calculator: QualityMetricsCalculator::new(config),
        }
    }
    
    /// Process multiple audio files and calculate quality metrics
    pub fn process_batch(&self, audio_files: &[AudioData]) -> Result<Vec<QualityMetrics>> {
        let mut all_metrics = Vec::with_capacity(audio_files.len());
        
        for audio in audio_files {
            let metrics = self.calculator.calculate_metrics(audio)?;
            all_metrics.push(metrics);
        }
        
        Ok(all_metrics)
    }
    
    /// Process multiple dataset samples
    pub fn process_samples(&self, samples: &[DatasetSample]) -> Result<Vec<QualityMetrics>> {
        let mut all_metrics = Vec::with_capacity(samples.len());
        
        for sample in samples {
            let metrics = self.calculator.calculate_sample_metrics(sample)?;
            all_metrics.push(metrics);
        }
        
        Ok(all_metrics)
    }
    
    /// Calculate summary statistics for batch
    pub fn calculate_summary(&self, metrics: &[QualityMetrics]) -> QualitySummary {
        if metrics.is_empty() {
            return QualitySummary::default();
        }
        
        let count = metrics.len() as f32;
        
        QualitySummary {
            total_samples: metrics.len(),
            average_snr: metrics.iter().map(|m| m.snr).sum::<f32>() / count,
            average_rms: metrics.iter().map(|m| m.rms).sum::<f32>() / count,
            average_dynamic_range: metrics.iter().map(|m| m.dynamic_range).sum::<f32>() / count,
            average_clipping: metrics.iter().map(|m| m.clipping_percentage).sum::<f32>() / count,
            average_speech_activity: metrics.iter().map(|m| m.speech_activity).sum::<f32>() / count,
            average_overall_score: metrics.iter().map(|m| m.overall_score).sum::<f32>() / count,
            min_score: metrics.iter().map(|m| m.overall_score).fold(f32::INFINITY, f32::min),
            max_score: metrics.iter().map(|m| m.overall_score).fold(f32::NEG_INFINITY, f32::max),
            low_quality_count: metrics.iter().filter(|m| m.overall_score < 0.5).count(),
            high_quality_count: metrics.iter().filter(|m| m.overall_score > 0.8).count(),
        }
    }
}

/// Summary statistics for quality metrics
#[derive(Debug, Clone)]
pub struct QualitySummary {
    /// Total number of samples analyzed
    pub total_samples: usize,
    /// Average SNR across all samples
    pub average_snr: f32,
    /// Average RMS level
    pub average_rms: f32,
    /// Average dynamic range
    pub average_dynamic_range: f32,
    /// Average clipping percentage
    pub average_clipping: f32,
    /// Average speech activity
    pub average_speech_activity: f32,
    /// Average overall quality score
    pub average_overall_score: f32,
    /// Minimum quality score
    pub min_score: f32,
    /// Maximum quality score
    pub max_score: f32,
    /// Number of low-quality samples (score < 0.5)
    pub low_quality_count: usize,
    /// Number of high-quality samples (score > 0.8)
    pub high_quality_count: usize,
}

impl Default for QualitySummary {
    fn default() -> Self {
        Self {
            total_samples: 0,
            average_snr: 0.0,
            average_rms: 0.0,
            average_dynamic_range: 0.0,
            average_clipping: 0.0,
            average_speech_activity: 0.0,
            average_overall_score: 0.0,
            min_score: 0.0,
            max_score: 0.0,
            low_quality_count: 0,
            high_quality_count: 0,
        }
    }
}

impl QualitySummary {
    /// Get quality distribution as percentages
    pub fn quality_distribution(&self) -> (f32, f32, f32) {
        if self.total_samples == 0 {
            return (0.0, 0.0, 0.0);
        }
        
        let low_percentage = (self.low_quality_count as f32 / self.total_samples as f32) * 100.0;
        let high_percentage = (self.high_quality_count as f32 / self.total_samples as f32) * 100.0;
        let medium_percentage = 100.0 - low_percentage - high_percentage;
        
        (low_percentage, medium_percentage, high_percentage)
    }
    
    /// Check if the dataset meets quality standards
    pub fn meets_quality_standards(&self, min_average_score: f32, max_low_quality_percentage: f32) -> bool {
        let (low_percentage, _, _) = self.quality_distribution();
        
        self.average_overall_score >= min_average_score && low_percentage <= max_low_quality_percentage
    }
}

/// Quality trend analyzer for tracking quality changes over time
pub struct QualityTrendAnalyzer {
    history: Vec<QualitySummary>,
    window_size: usize,
}

impl QualityTrendAnalyzer {
    /// Create new trend analyzer
    pub fn new(window_size: usize) -> Self {
        Self {
            history: Vec::new(),
            window_size,
        }
    }
    
    /// Add new quality summary to history
    pub fn add_summary(&mut self, summary: QualitySummary) {
        self.history.push(summary);
        
        // Keep only the last window_size entries
        if self.history.len() > self.window_size {
            self.history.remove(0);
        }
    }
    
    /// Calculate quality trend (positive = improving, negative = degrading)
    pub fn calculate_trend(&self) -> f32 {
        if self.history.len() < 2 {
            return 0.0;
        }
        
        let recent_avg = self.history.iter()
            .rev()
            .take(self.history.len() / 2)
            .map(|s| s.average_overall_score)
            .sum::<f32>() / (self.history.len() / 2) as f32;
            
        let older_avg = self.history.iter()
            .take(self.history.len() / 2)
            .map(|s| s.average_overall_score)
            .sum::<f32>() / (self.history.len() / 2) as f32;
        
        recent_avg - older_avg
    }
    
    /// Get quality stability score (0.0 = very unstable, 1.0 = very stable)
    pub fn calculate_stability(&self) -> f32 {
        if self.history.len() < 3 {
            return 1.0;
        }
        
        let scores: Vec<f32> = self.history.iter().map(|s| s.average_overall_score).collect();
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        
        let variance = scores.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / scores.len() as f32;
            
        let std_dev = variance.sqrt();
        
        // Higher stability for lower standard deviation
        (1.0 - std_dev).max(0.0)
    }
}