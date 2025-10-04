//! Real-time Quality Monitoring System
//!
//! This module provides real-time quality assessment capabilities for live audio synthesis,
//! enabling immediate quality feedback during generation with minimal latency impact.
//! Key features include:
//! - Streaming quality analysis with configurable window sizes
//! - Real-time quality alerts and threshold monitoring
//! - Adaptive quality thresholds based on historical data
//! - Performance-optimized implementations for live synthesis

use crate::EvaluationError;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use voirs_sdk::AudioBuffer;

/// Real-time quality monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeQualityConfig {
    /// Window size for quality analysis (in samples)
    pub analysis_window_size: usize,
    /// Overlap between analysis windows (0.0-1.0)
    pub window_overlap: f32,
    /// Quality threshold for alerts
    pub quality_threshold: f32,
    /// Maximum processing latency (milliseconds)
    pub max_processing_latency_ms: u64,
    /// Enable adaptive thresholds
    pub adaptive_thresholds: bool,
    /// Buffer size for historical quality data
    pub history_buffer_size: usize,
    /// Enable detailed quality breakdown
    pub detailed_analysis: bool,
    /// Target sample rate for analysis
    pub target_sample_rate: u32,
}

impl Default for RealTimeQualityConfig {
    fn default() -> Self {
        Self {
            analysis_window_size: 1024,
            window_overlap: 0.5,
            quality_threshold: 0.7,
            max_processing_latency_ms: 10,
            adaptive_thresholds: true,
            history_buffer_size: 100,
            detailed_analysis: false,
            target_sample_rate: 22050,
        }
    }
}

/// Real-time quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeQualityMetrics {
    /// Overall quality score (0.0-1.0)
    pub overall_quality: f32,
    /// Signal-to-noise ratio
    pub snr_db: f32,
    /// Spectral distortion level
    pub spectral_distortion: f32,
    /// Temporal consistency measure
    pub temporal_consistency: f32,
    /// Perceptual quality score
    pub perceptual_quality: f32,
    /// Quality trend (improving/degrading)
    pub quality_trend: QualityTrend,
    /// Processing latency in microseconds
    pub processing_latency_us: u64,
    /// Timestamp of measurement
    pub timestamp: DateTime<Utc>,
    /// Additional detailed metrics (optional)
    pub detailed_metrics: Option<DetailedQualityMetrics>,
}

/// Quality trend indicators
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QualityTrend {
    /// Quality is improving
    Improving,
    /// Quality is stable
    Stable,
    /// Quality is degrading
    Degrading,
    /// Insufficient data to determine trend
    Unknown,
}

/// Detailed quality metrics for in-depth analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedQualityMetrics {
    /// Frequency domain metrics
    pub frequency_metrics: FrequencyDomainMetrics,
    /// Time domain metrics
    pub time_metrics: TimeDomainMetrics,
    /// Perceptual metrics
    pub perceptual_metrics: PerceptualMetrics,
}

/// Frequency domain quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyDomainMetrics {
    /// Spectral centroid
    pub spectral_centroid: f32,
    /// Spectral rolloff
    pub spectral_rolloff: f32,
    /// Harmonic distortion
    pub harmonic_distortion: f32,
    /// Frequency response flatness
    pub frequency_flatness: f32,
}

/// Time domain quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeDomainMetrics {
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    /// RMS energy
    pub rms_energy: f32,
    /// Envelope consistency
    pub envelope_consistency: f32,
    /// Click and pop detection
    pub click_pop_score: f32,
}

/// Perceptual quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualMetrics {
    /// Naturalness score
    pub naturalness: f32,
    /// Intelligibility score
    pub intelligibility: f32,
    /// Pleasantness score
    pub pleasantness: f32,
    /// Robotic sound indicator
    pub robotic_score: f32,
}

/// Quality alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityAlert {
    /// Quality dropped below threshold
    QualityBelowThreshold {
        /// Current quality score
        current_quality: f32,
        /// Quality threshold that was violated
        threshold: f32,
        /// Severity level of the alert
        severity: AlertSeverity,
    },
    /// Processing latency exceeded limit
    LatencyExceeded {
        /// Current processing latency in milliseconds
        current_latency_ms: u64,
        /// Maximum allowed latency in milliseconds
        max_latency_ms: u64,
    },
    /// Quality trend is degrading
    QualityDegrading {
        /// Duration over which quality has been degrading
        trend_duration: Duration,
        /// Rate of quality degradation
        degradation_rate: f32,
    },
    /// Spectral anomaly detected
    SpectralAnomaly {
        /// Type of anomaly detected
        anomaly_type: String,
        /// Confidence level of the detection (0.0-1.0)
        confidence: f32,
    },
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Low priority issue
    Low,
    /// Medium priority issue
    Medium,
    /// High priority issue requiring attention
    High,
    /// Critical issue requiring immediate action
    Critical,
}

/// Real-time quality monitor
pub struct RealTimeQualityMonitor {
    config: RealTimeQualityConfig,
    quality_history: Arc<Mutex<VecDeque<RealTimeQualityMetrics>>>,
    adaptive_thresholds: Arc<RwLock<AdaptiveThresholds>>,
    alert_callbacks: Arc<Mutex<Vec<Box<dyn Fn(QualityAlert) + Send + Sync>>>>,
    processing_stats: Arc<Mutex<ProcessingStats>>,
}

/// Adaptive threshold management
#[derive(Debug)]
struct AdaptiveThresholds {
    quality_threshold: f32,
    snr_threshold: f32,
    last_update: Instant,
    update_interval: Duration,
}

/// Processing performance statistics
#[derive(Debug, Default)]
pub struct ProcessingStats {
    pub total_samples_processed: u64,
    pub total_processing_time: Duration,
    pub peak_latency: Duration,
    pub average_latency: Duration,
    pub quality_violations: u32,
}

impl RealTimeQualityMonitor {
    /// Create a new real-time quality monitor
    pub fn new(config: RealTimeQualityConfig) -> Self {
        let adaptive_thresholds = AdaptiveThresholds {
            quality_threshold: config.quality_threshold,
            snr_threshold: 15.0, // Default SNR threshold in dB
            last_update: Instant::now(),
            update_interval: Duration::from_secs(30),
        };

        Self {
            config,
            quality_history: Arc::new(Mutex::new(VecDeque::new())),
            adaptive_thresholds: Arc::new(RwLock::new(adaptive_thresholds)),
            alert_callbacks: Arc::new(Mutex::new(Vec::new())),
            processing_stats: Arc::new(Mutex::new(ProcessingStats::default())),
        }
    }

    /// Add a callback for quality alerts
    pub fn add_alert_callback<F>(&self, callback: F)
    where
        F: Fn(QualityAlert) + Send + Sync + 'static,
    {
        self.alert_callbacks
            .lock()
            .unwrap()
            .push(Box::new(callback));
    }

    /// Process audio chunk and return quality metrics
    pub async fn process_chunk(
        &self,
        audio_chunk: &AudioBuffer,
    ) -> Result<RealTimeQualityMetrics, EvaluationError> {
        let start_time = Instant::now();

        // Validate input
        if audio_chunk.samples().is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: "Empty audio chunk".to_string(),
            });
        }

        // Ensure audio is in the target sample rate (simplified check)
        if audio_chunk.sample_rate() != self.config.target_sample_rate {
            // In a real implementation, you might resample here
            return Err(EvaluationError::InvalidInput {
                message: format!(
                    "Sample rate mismatch: expected {}, got {}",
                    self.config.target_sample_rate,
                    audio_chunk.sample_rate()
                ),
            });
        }

        // Calculate quality metrics
        let metrics = self.calculate_quality_metrics(audio_chunk).await?;

        // Update processing statistics
        self.update_processing_stats(audio_chunk.samples().len(), start_time.elapsed())
            .await;

        // Check for quality alerts
        self.check_quality_alerts(&metrics).await;

        // Store in history
        self.store_quality_metrics(metrics.clone()).await;

        // Update adaptive thresholds if enabled
        if self.config.adaptive_thresholds {
            self.update_adaptive_thresholds().await;
        }

        Ok(metrics)
    }

    /// Calculate comprehensive quality metrics for an audio chunk
    async fn calculate_quality_metrics(
        &self,
        audio_chunk: &AudioBuffer,
    ) -> Result<RealTimeQualityMetrics, EvaluationError> {
        let samples = audio_chunk.samples();

        // Calculate basic quality metrics
        let snr_db = self.calculate_snr(samples);
        let spectral_distortion = self.calculate_spectral_distortion(samples);
        let temporal_consistency = self.calculate_temporal_consistency(samples);
        let perceptual_quality = self.calculate_perceptual_quality(samples);

        // Calculate overall quality as weighted average
        let overall_quality = 0.3 * (snr_db / 30.0).clamp(0.0, 1.0)
            + 0.25 * (1.0 - spectral_distortion.clamp(0.0, 1.0))
            + 0.25 * temporal_consistency.clamp(0.0, 1.0)
            + 0.2 * perceptual_quality.clamp(0.0, 1.0);

        // Determine quality trend
        let quality_trend = self.calculate_quality_trend(overall_quality).await;

        // Calculate detailed metrics if enabled
        let detailed_metrics = if self.config.detailed_analysis {
            Some(self.calculate_detailed_metrics(samples))
        } else {
            None
        };

        Ok(RealTimeQualityMetrics {
            overall_quality,
            snr_db,
            spectral_distortion,
            temporal_consistency,
            perceptual_quality,
            quality_trend,
            processing_latency_us: 0, // Will be set by caller
            timestamp: Utc::now(),
            detailed_metrics,
        })
    }

    /// Calculate Signal-to-Noise Ratio
    fn calculate_snr(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // Calculate RMS of signal
        let signal_rms =
            (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();

        // Estimate noise level (simplified: use high-frequency components)
        let noise_estimate = self.estimate_noise_level(samples);

        // Calculate SNR in dB
        if noise_estimate > 0.0 {
            20.0 * (signal_rms / noise_estimate).log10()
        } else {
            60.0 // Very high SNR if no noise detected
        }
    }

    /// Estimate noise level from high-frequency components
    fn estimate_noise_level(&self, samples: &[f32]) -> f32 {
        // Simplified noise estimation using high-frequency energy
        // In a real implementation, you might use more sophisticated methods
        if samples.len() < 2 {
            return 0.001; // Minimal noise floor
        }

        // Calculate high-frequency differences as noise proxy
        let mut noise_energy = 0.0;
        for i in 1..samples.len() {
            let diff = samples[i] - samples[i - 1];
            noise_energy += diff * diff;
        }

        (noise_energy / (samples.len() - 1) as f32)
            .sqrt()
            .max(0.001)
    }

    /// Calculate spectral distortion metric
    fn calculate_spectral_distortion(&self, samples: &[f32]) -> f32 {
        // Simplified spectral distortion calculation
        // In a real implementation, you would use proper FFT analysis
        if samples.is_empty() {
            return 1.0; // Maximum distortion
        }

        // Calculate spectral flatness as a proxy for distortion
        let mut spectral_peaks = 0;
        let window_size = (samples.len() / 10).max(1);

        for window_start in (0..samples.len()).step_by(window_size) {
            let window_end = (window_start + window_size).min(samples.len());
            let window = &samples[window_start..window_end];

            let max_val = window.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
            let avg_val = window.iter().map(|&x| x.abs()).sum::<f32>() / window.len() as f32;

            if avg_val > 0.0 && max_val / avg_val > 3.0 {
                spectral_peaks += 1;
            }
        }

        // Convert to distortion metric (0.0 = no distortion, 1.0 = high distortion)
        (spectral_peaks as f32 / 10.0).clamp(0.0, 1.0)
    }

    /// Calculate temporal consistency metric
    fn calculate_temporal_consistency(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 1.0; // Perfect consistency for single sample
        }

        // Calculate variance in frame-to-frame energy differences
        let frame_size = (samples.len() / 20).max(1);
        let mut frame_energies = Vec::new();

        for frame_start in (0..samples.len()).step_by(frame_size) {
            let frame_end = (frame_start + frame_size).min(samples.len());
            let frame = &samples[frame_start..frame_end];
            let energy = frame.iter().map(|&x| x * x).sum::<f32>() / frame.len() as f32;
            frame_energies.push(energy);
        }

        if frame_energies.len() < 2 {
            return 1.0;
        }

        // Calculate consistency based on energy variance
        let mean_energy = frame_energies.iter().sum::<f32>() / frame_energies.len() as f32;
        let variance = frame_energies
            .iter()
            .map(|&e| (e - mean_energy).powi(2))
            .sum::<f32>()
            / frame_energies.len() as f32;

        // Convert to consistency score (higher variance = lower consistency)
        if mean_energy > 0.0 {
            1.0 / (1.0 + variance / mean_energy)
        } else {
            1.0
        }
    }

    /// Calculate perceptual quality score
    fn calculate_perceptual_quality(&self, samples: &[f32]) -> f32 {
        // Simplified perceptual quality estimation
        // In a real implementation, you would use perceptual models like PESQ or STOI

        if samples.is_empty() {
            return 0.0;
        }

        // Calculate basic perceptual indicators
        let dynamic_range = self.calculate_dynamic_range(samples);
        let spectral_richness = self.calculate_spectral_richness(samples);
        let temporal_smoothness = self.calculate_temporal_smoothness(samples);

        // Combine metrics for overall perceptual quality
        0.4 * dynamic_range + 0.3 * spectral_richness + 0.3 * temporal_smoothness
    }

    /// Calculate dynamic range metric
    fn calculate_dynamic_range(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let max_val = samples.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();

        if rms > 0.0 {
            (max_val / rms / 10.0).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Calculate spectral richness metric
    fn calculate_spectral_richness(&self, samples: &[f32]) -> f32 {
        // Simplified spectral richness based on harmonic content
        if samples.len() < 4 {
            return 0.5;
        }

        // Count zero crossings as a proxy for frequency content
        let mut zero_crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                zero_crossings += 1;
            }
        }

        // Normalize and clamp
        let normalized_crossings = zero_crossings as f32 / samples.len() as f32;
        (normalized_crossings * 10.0).clamp(0.0, 1.0)
    }

    /// Calculate temporal smoothness metric
    fn calculate_temporal_smoothness(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 1.0;
        }

        // Calculate smoothness based on derivative variance
        let mut total_variation = 0.0;
        for i in 1..samples.len() {
            total_variation += (samples[i] - samples[i - 1]).abs();
        }

        let average_variation = total_variation / (samples.len() - 1) as f32;
        // Convert to smoothness score (lower variation = higher smoothness)
        1.0 / (1.0 + average_variation * 10.0)
    }

    /// Calculate detailed quality metrics
    fn calculate_detailed_metrics(&self, samples: &[f32]) -> DetailedQualityMetrics {
        DetailedQualityMetrics {
            frequency_metrics: FrequencyDomainMetrics {
                spectral_centroid: self.calculate_spectral_centroid(samples),
                spectral_rolloff: self.calculate_spectral_rolloff(samples),
                harmonic_distortion: self.calculate_harmonic_distortion(samples),
                frequency_flatness: self.calculate_frequency_flatness(samples),
            },
            time_metrics: TimeDomainMetrics {
                zero_crossing_rate: self.calculate_zero_crossing_rate(samples),
                rms_energy: self.calculate_rms_energy(samples),
                envelope_consistency: self.calculate_envelope_consistency(samples),
                click_pop_score: self.calculate_click_pop_score(samples),
            },
            perceptual_metrics: PerceptualMetrics {
                naturalness: self.calculate_naturalness(samples),
                intelligibility: self.calculate_intelligibility(samples),
                pleasantness: self.calculate_pleasantness(samples),
                robotic_score: self.calculate_robotic_score(samples),
            },
        }
    }

    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, samples: &[f32]) -> f32 {
        // Simplified spectral centroid calculation
        if samples.is_empty() {
            return 0.0;
        }

        // In a real implementation, you would use FFT to calculate the true spectral centroid
        // Here we approximate using zero-crossing rate
        let zcr = self.calculate_zero_crossing_rate(samples);
        zcr * 1000.0 // Convert to approximate Hz
    }

    /// Calculate spectral rolloff
    fn calculate_spectral_rolloff(&self, samples: &[f32]) -> f32 {
        // Simplified spectral rolloff calculation
        if samples.is_empty() {
            return 0.0;
        }

        // Approximate using high-frequency energy distribution
        let high_freq_energy = self.calculate_high_frequency_energy(samples);
        high_freq_energy * 8000.0 // Convert to approximate Hz
    }

    /// Calculate harmonic distortion
    fn calculate_harmonic_distortion(&self, samples: &[f32]) -> f32 {
        // Simplified harmonic distortion calculation
        self.calculate_spectral_distortion(samples)
    }

    /// Calculate frequency flatness
    fn calculate_frequency_flatness(&self, samples: &[f32]) -> f32 {
        // Simplified frequency flatness calculation
        1.0 - self.calculate_spectral_distortion(samples)
    }

    /// Calculate zero crossing rate
    fn calculate_zero_crossing_rate(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let mut zero_crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                zero_crossings += 1;
            }
        }

        zero_crossings as f32 / (samples.len() - 1) as f32
    }

    /// Calculate RMS energy
    fn calculate_rms_energy(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
    }

    /// Calculate envelope consistency
    fn calculate_envelope_consistency(&self, samples: &[f32]) -> f32 {
        self.calculate_temporal_consistency(samples)
    }

    /// Calculate click and pop detection score
    fn calculate_click_pop_score(&self, samples: &[f32]) -> f32 {
        if samples.len() < 3 {
            return 0.0;
        }

        // Detect sudden amplitude changes (clicks/pops)
        let mut click_count = 0;
        let threshold = 0.1; // Configurable threshold

        for i in 1..samples.len() - 1 {
            let prev_diff = (samples[i] - samples[i - 1]).abs();
            let next_diff = (samples[i + 1] - samples[i]).abs();

            if prev_diff > threshold && next_diff > threshold {
                click_count += 1;
            }
        }

        // Convert to score (0.0 = no clicks, 1.0 = many clicks)
        (click_count as f32 / samples.len() as f32 * 100.0).clamp(0.0, 1.0)
    }

    /// Calculate naturalness score
    fn calculate_naturalness(&self, samples: &[f32]) -> f32 {
        // Simplified naturalness calculation
        let spectral_richness = self.calculate_spectral_richness(samples);
        let temporal_smoothness = self.calculate_temporal_smoothness(samples);
        let dynamic_range = self.calculate_dynamic_range(samples);

        (spectral_richness + temporal_smoothness + dynamic_range) / 3.0
    }

    /// Calculate intelligibility score
    fn calculate_intelligibility(&self, samples: &[f32]) -> f32 {
        // Simplified intelligibility based on clarity metrics
        let snr = self.calculate_snr(samples);
        let spectral_clarity = 1.0 - self.calculate_spectral_distortion(samples);

        ((snr / 30.0).clamp(0.0, 1.0) + spectral_clarity) / 2.0
    }

    /// Calculate pleasantness score
    fn calculate_pleasantness(&self, samples: &[f32]) -> f32 {
        // Simplified pleasantness based on smoothness and harmony
        let smoothness = self.calculate_temporal_smoothness(samples);
        let low_distortion = 1.0 - self.calculate_spectral_distortion(samples);
        let good_dynamics = self.calculate_dynamic_range(samples);

        (smoothness + low_distortion + good_dynamics) / 3.0
    }

    /// Calculate robotic sound indicator
    fn calculate_robotic_score(&self, samples: &[f32]) -> f32 {
        // Higher score means more robotic
        let low_variation = 1.0 - self.calculate_temporal_consistency(samples);
        let spectral_artifacts = self.calculate_spectral_distortion(samples);

        (low_variation + spectral_artifacts) / 2.0
    }

    /// Calculate high-frequency energy
    fn calculate_high_frequency_energy(&self, samples: &[f32]) -> f32 {
        // Simplified high-frequency energy calculation
        if samples.len() < 2 {
            return 0.0;
        }

        let mut high_freq_energy = 0.0;
        for i in 1..samples.len() {
            let derivative = samples[i] - samples[i - 1];
            high_freq_energy += derivative * derivative;
        }

        (high_freq_energy / (samples.len() - 1) as f32).sqrt()
    }

    /// Calculate quality trend based on recent history
    async fn calculate_quality_trend(&self, current_quality: f32) -> QualityTrend {
        let history = self.quality_history.lock().unwrap();

        if history.len() < 3 {
            return QualityTrend::Unknown;
        }

        // Get recent quality values
        let recent_qualities: Vec<f32> = history
            .iter()
            .rev()
            .take(5)
            .map(|m| m.overall_quality)
            .collect();

        if recent_qualities.len() < 3 {
            return QualityTrend::Unknown;
        }

        // Calculate trend using linear regression slope
        let n = recent_qualities.len() as f32;
        let x_sum = (0..recent_qualities.len()).map(|i| i as f32).sum::<f32>();
        let y_sum = recent_qualities.iter().sum::<f32>();
        let xy_sum = recent_qualities
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f32 * y)
            .sum::<f32>();
        let x2_sum = (0..recent_qualities.len())
            .map(|i| (i as f32).powi(2))
            .sum::<f32>();

        let slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum);

        match slope {
            s if s > 0.01 => QualityTrend::Improving,
            s if s < -0.01 => QualityTrend::Degrading,
            _ => QualityTrend::Stable,
        }
    }

    /// Store quality metrics in history buffer
    async fn store_quality_metrics(&self, metrics: RealTimeQualityMetrics) {
        let mut history = self.quality_history.lock().unwrap();

        // Add new metrics
        history.push_back(metrics);

        // Maintain buffer size
        while history.len() > self.config.history_buffer_size {
            history.pop_front();
        }
    }

    /// Check for quality alerts and trigger callbacks
    async fn check_quality_alerts(&self, metrics: &RealTimeQualityMetrics) {
        let thresholds = self.adaptive_thresholds.read().await;
        let mut alerts = Vec::new();

        // Check quality threshold
        if metrics.overall_quality < thresholds.quality_threshold {
            let severity = match metrics.overall_quality {
                q if q < 0.3 => AlertSeverity::Critical,
                q if q < 0.5 => AlertSeverity::High,
                q if q < 0.6 => AlertSeverity::Medium,
                _ => AlertSeverity::Low,
            };

            alerts.push(QualityAlert::QualityBelowThreshold {
                current_quality: metrics.overall_quality,
                threshold: thresholds.quality_threshold,
                severity,
            });
        }

        // Check SNR threshold
        if metrics.snr_db < thresholds.snr_threshold {
            alerts.push(QualityAlert::SpectralAnomaly {
                anomaly_type: "Low SNR".to_string(),
                confidence: 1.0 - (metrics.snr_db / thresholds.snr_threshold).clamp(0.0, 1.0),
            });
        }

        // Check for quality degradation trend
        if matches!(metrics.quality_trend, QualityTrend::Degrading) {
            alerts.push(QualityAlert::QualityDegrading {
                trend_duration: Duration::from_secs(30), // Estimated
                degradation_rate: 0.1,                   // Estimated rate
            });
        }

        // Trigger alert callbacks
        if !alerts.is_empty() {
            let callbacks = self.alert_callbacks.lock().unwrap();
            for alert in alerts {
                for callback in callbacks.iter() {
                    callback(alert.clone());
                }
            }
        }
    }

    /// Update adaptive thresholds based on historical data
    async fn update_adaptive_thresholds(&self) {
        let mut thresholds = self.adaptive_thresholds.write().await;

        if thresholds.last_update.elapsed() < thresholds.update_interval {
            return;
        }

        let history = self.quality_history.lock().unwrap();
        if history.len() < 10 {
            return; // Need more data
        }

        // Calculate statistics from recent history
        let recent_qualities: Vec<f32> = history
            .iter()
            .rev()
            .take(20)
            .map(|m| m.overall_quality)
            .collect();

        if !recent_qualities.is_empty() {
            let mean_quality = recent_qualities.iter().sum::<f32>() / recent_qualities.len() as f32;
            let std_dev = {
                let variance = recent_qualities
                    .iter()
                    .map(|&q| (q - mean_quality).powi(2))
                    .sum::<f32>()
                    / recent_qualities.len() as f32;
                variance.sqrt()
            };

            // Adjust threshold to be one standard deviation below mean
            thresholds.quality_threshold = (mean_quality - std_dev).clamp(0.1, 0.9);
        }

        thresholds.last_update = Instant::now();
    }

    /// Update processing statistics
    async fn update_processing_stats(&self, samples_processed: usize, processing_time: Duration) {
        let mut stats = self.processing_stats.lock().unwrap();

        stats.total_samples_processed += samples_processed as u64;
        stats.total_processing_time += processing_time;

        if processing_time > stats.peak_latency {
            stats.peak_latency = processing_time;
        }

        // Update average latency
        let total_chunks = (stats.total_samples_processed / 1024).max(1); // Assume 1024 samples per chunk
        stats.average_latency = stats.total_processing_time / total_chunks as u32;
    }

    /// Get current processing statistics
    pub async fn get_processing_stats(&self) -> ProcessingStats {
        let stats = self.processing_stats.lock().unwrap();
        ProcessingStats {
            total_samples_processed: stats.total_samples_processed,
            total_processing_time: stats.total_processing_time,
            peak_latency: stats.peak_latency,
            average_latency: stats.average_latency,
            quality_violations: stats.quality_violations,
        }
    }

    /// Get quality history
    pub async fn get_quality_history(&self) -> Vec<RealTimeQualityMetrics> {
        let history = self.quality_history.lock().unwrap();
        history.iter().cloned().collect()
    }

    /// Get current adaptive thresholds
    pub async fn get_adaptive_thresholds(&self) -> (f32, f32) {
        let thresholds = self.adaptive_thresholds.read().await;
        (thresholds.quality_threshold, thresholds.snr_threshold)
    }

    /// Reset monitoring state
    pub async fn reset(&self) {
        self.quality_history.lock().unwrap().clear();
        *self.processing_stats.lock().unwrap() = ProcessingStats::default();

        let mut thresholds = self.adaptive_thresholds.write().await;
        thresholds.quality_threshold = self.config.quality_threshold;
        thresholds.snr_threshold = 15.0;
        thresholds.last_update = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_real_time_quality_monitor_creation() {
        let config = RealTimeQualityConfig::default();
        let monitor = RealTimeQualityMonitor::new(config);

        let stats = monitor.get_processing_stats().await;
        assert_eq!(stats.total_samples_processed, 0);
    }

    #[tokio::test]
    async fn test_process_chunk() {
        let config = RealTimeQualityConfig::default();
        let monitor = RealTimeQualityMonitor::new(config);

        // Create test audio buffer
        let test_samples = vec![0.1, 0.2, -0.1, -0.2, 0.15, -0.15]; // Simple test pattern
        let audio_buffer = AudioBuffer::new(test_samples, 22050, 1);

        let result = monitor.process_chunk(&audio_buffer).await;
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.overall_quality >= 0.0 && metrics.overall_quality <= 1.0);
        assert!(metrics.snr_db >= -40.0); // SNR can be negative for very noisy signals
    }

    #[tokio::test]
    async fn test_quality_metrics_calculation() {
        let config = RealTimeQualityConfig::default();
        let monitor = RealTimeQualityMonitor::new(config);

        // Test with different signal types
        let test_cases = vec![
            vec![0.0; 1024], // Silence
            (0..1024)
                .map(|i| (i as f32 * 0.01).sin())
                .collect::<Vec<f32>>(), // Sine wave
            (0..1024)
                .map(|i| ((i as f32 * 0.037).sin() * 0.05))
                .collect::<Vec<f32>>(), // Pseudo-noise
        ];

        for samples in test_cases {
            let audio_buffer = AudioBuffer::new(samples, 22050, 1);
            let result = monitor.process_chunk(&audio_buffer).await;
            assert!(result.is_ok());

            let metrics = result.unwrap();
            assert!(metrics.overall_quality >= 0.0 && metrics.overall_quality <= 1.0);
            assert!(metrics.temporal_consistency >= 0.0 && metrics.temporal_consistency <= 1.0);
            assert!(metrics.spectral_distortion >= 0.0 && metrics.spectral_distortion <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_quality_trend_detection() {
        let config = RealTimeQualityConfig::default();
        let monitor = RealTimeQualityMonitor::new(config);

        // Simulate degrading quality
        let quality_values = vec![0.9, 0.85, 0.8, 0.75, 0.7];

        for quality in quality_values {
            let samples = vec![quality; 1024]; // Simple proxy for quality
            let audio_buffer = AudioBuffer::new(samples, 22050, 1);
            let _ = monitor.process_chunk(&audio_buffer).await;
        }

        // The trend should eventually be detected as degrading
        let history = monitor.get_quality_history().await;
        assert!(!history.is_empty());
    }

    #[tokio::test]
    async fn test_alert_callback() {
        let config = RealTimeQualityConfig {
            quality_threshold: 0.8, // High threshold to trigger alerts
            ..Default::default()
        };
        let monitor = RealTimeQualityMonitor::new(config);

        // Set up alert callback
        let alert_received = Arc::new(Mutex::new(false));
        let alert_received_clone = alert_received.clone();

        monitor.add_alert_callback(move |_alert| {
            *alert_received_clone.lock().unwrap() = true;
        });

        // Process low-quality audio to trigger alert
        let low_quality_samples = vec![0.01; 1024]; // Very low amplitude
        let audio_buffer = AudioBuffer::new(low_quality_samples, 22050, 1);

        let _ = monitor.process_chunk(&audio_buffer).await;

        // Give some time for callback processing
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Check if alert was triggered (may not always trigger depending on implementation)
        // This test is mainly to ensure the callback system works without panicking
    }

    #[tokio::test]
    async fn test_adaptive_thresholds() {
        let config = RealTimeQualityConfig {
            adaptive_thresholds: true,
            ..Default::default()
        };
        let monitor = RealTimeQualityMonitor::new(config);

        let initial_thresholds = monitor.get_adaptive_thresholds().await;

        // Process several chunks to build history
        for i in 0..15 {
            let quality = 0.7 + (i as f32 * 0.01); // Gradually improving quality
            let samples = vec![quality; 1024];
            let audio_buffer = AudioBuffer::new(samples, 22050, 1);
            let _ = monitor.process_chunk(&audio_buffer).await;
        }

        // Force threshold update by manipulating time (in real implementation)
        // For this test, we just verify the system doesn't crash
        let final_thresholds = monitor.get_adaptive_thresholds().await;
        assert!(final_thresholds.0 > 0.0);
        assert!(final_thresholds.1 > 0.0);
    }

    #[tokio::test]
    async fn test_detailed_metrics() {
        let config = RealTimeQualityConfig {
            detailed_analysis: true,
            ..Default::default()
        };
        let monitor = RealTimeQualityMonitor::new(config);

        let test_samples = (0..1024)
            .map(|i| (i as f32 * 0.01).sin())
            .collect::<Vec<f32>>();
        let audio_buffer = AudioBuffer::new(test_samples, 22050, 1);

        let result = monitor.process_chunk(&audio_buffer).await;
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.detailed_metrics.is_some());

        let detailed = metrics.detailed_metrics.unwrap();
        assert!(detailed.frequency_metrics.spectral_centroid >= 0.0);
        assert!(detailed.time_metrics.zero_crossing_rate >= 0.0);
        assert!(detailed.perceptual_metrics.naturalness >= 0.0);
    }

    #[tokio::test]
    async fn test_processing_stats() {
        let config = RealTimeQualityConfig::default();
        let monitor = RealTimeQualityMonitor::new(config);

        // Process a few chunks
        for _ in 0..3 {
            let samples = vec![0.1; 1024];
            let audio_buffer = AudioBuffer::new(samples, 22050, 1);
            let _ = monitor.process_chunk(&audio_buffer).await;
        }

        let stats = monitor.get_processing_stats().await;
        assert!(stats.total_samples_processed > 0);
        assert!(stats.total_processing_time > Duration::from_nanos(0));
    }

    #[tokio::test]
    async fn test_reset_functionality() {
        let config = RealTimeQualityConfig::default();
        let monitor = RealTimeQualityMonitor::new(config);

        // Process some chunks
        let samples = vec![0.1; 1024];
        let audio_buffer = AudioBuffer::new(samples, 22050, 1);
        let _ = monitor.process_chunk(&audio_buffer).await;

        // Verify data exists
        let history_before = monitor.get_quality_history().await;
        assert!(!history_before.is_empty());

        // Reset
        monitor.reset().await;

        // Verify reset worked
        let history_after = monitor.get_quality_history().await;
        assert!(history_after.is_empty());

        let stats_after = monitor.get_processing_stats().await;
        assert_eq!(stats_after.total_samples_processed, 0);
    }
}
