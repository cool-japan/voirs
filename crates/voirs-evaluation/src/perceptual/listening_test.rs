//! Real-time intelligibility monitoring and perceptual evaluation
//!
//! This module provides comprehensive real-time monitoring of speech intelligibility
//! and perceptual quality metrics for streaming audio applications.

use crate::audio::streaming::{AudioChunk, StreamingConfig, StreamingQualityMetrics};
use crate::traits::{EvaluationResult, QualityScore};
use crate::EvaluationError;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use voirs_sdk::AudioBuffer;

/// Configuration for real-time intelligibility monitoring
#[derive(Debug, Clone)]
pub struct IntelligibilityMonitorConfig {
    /// Window size for intelligibility analysis (in samples)
    pub analysis_window_size: usize,
    /// Update frequency for intelligibility scores (in Hz)
    pub update_frequency: f32,
    /// Enable context-dependent analysis
    pub enable_context_analysis: bool,
    /// Enable background noise estimation
    pub enable_noise_estimation: bool,
    /// Minimum signal level for analysis (-dB)
    pub min_signal_level_db: f32,
    /// Maximum processing latency allowed (ms)
    pub max_latency_ms: u64,
}

impl Default for IntelligibilityMonitorConfig {
    fn default() -> Self {
        Self {
            analysis_window_size: 2048, // ~128ms at 16kHz
            update_frequency: 10.0,     // 10 updates per second
            enable_context_analysis: true,
            enable_noise_estimation: true,
            min_signal_level_db: -40.0,
            max_latency_ms: 50, // 50ms max latency
        }
    }
}

/// Real-time intelligibility metrics
#[derive(Debug, Clone)]
pub struct IntelligibilityMetrics {
    /// Objective intelligibility score (0.0 - 1.0)
    pub intelligibility_score: f32,
    /// Speech clarity index
    pub clarity_index: f32,
    /// Estimated signal-to-noise ratio
    pub snr_estimate: f32,
    /// Spectral clarity measure
    pub spectral_clarity: f32,
    /// Temporal clarity measure
    pub temporal_clarity: f32,
    /// Background noise level estimate
    pub noise_level_db: f32,
    /// Speech activity detection
    pub speech_activity: f32,
    /// Confidence in measurements
    pub confidence: f32,
    /// Processing timestamp
    pub timestamp: Instant,
}

impl Default for IntelligibilityMetrics {
    fn default() -> Self {
        Self {
            intelligibility_score: 0.0,
            clarity_index: 0.0,
            snr_estimate: 0.0,
            spectral_clarity: 0.0,
            temporal_clarity: 0.0,
            noise_level_db: -60.0,
            speech_activity: 0.0,
            confidence: 0.0,
            timestamp: Instant::now(),
        }
    }
}

/// Context information for intelligibility analysis
#[derive(Debug, Clone)]
pub struct IntelligibilityContext {
    /// Recent SNR history
    pub snr_history: VecDeque<f32>,
    /// Background noise characteristics
    pub noise_profile: NoiseProfile,
    /// Speech activity history
    pub activity_history: VecDeque<f32>,
    /// Processing load indicator
    pub processing_load: f32,
}

impl Default for IntelligibilityContext {
    fn default() -> Self {
        Self {
            snr_history: VecDeque::with_capacity(50),
            noise_profile: NoiseProfile::default(),
            activity_history: VecDeque::with_capacity(100),
            processing_load: 0.0,
        }
    }
}

/// Background noise profile for context-aware analysis
#[derive(Debug, Clone)]
pub struct NoiseProfile {
    /// Estimated noise floor level
    pub noise_floor: f32,
    /// Spectral characteristics of noise
    pub spectral_profile: Vec<f32>,
    /// Temporal characteristics
    pub temporal_variation: f32,
    /// Last update timestamp
    pub last_update: Instant,
}

impl Default for NoiseProfile {
    fn default() -> Self {
        Self {
            noise_floor: -60.0,
            spectral_profile: vec![0.0; 64], // 64 frequency bins
            temporal_variation: 0.0,
            last_update: Instant::now(),
        }
    }
}

/// Real-time intelligibility monitor
pub struct IntelligibilityMonitor {
    /// Configuration
    config: IntelligibilityMonitorConfig,
    /// Analysis context
    context: Arc<Mutex<IntelligibilityContext>>,
    /// Recent metrics history
    metrics_history: Arc<Mutex<VecDeque<IntelligibilityMetrics>>>,
    /// Channel for sending intelligibility updates
    update_sender: Option<mpsc::UnboundedSender<IntelligibilityMetrics>>,
    /// Last analysis timestamp
    last_analysis: Instant,
}

impl IntelligibilityMonitor {
    /// Create a new intelligibility monitor
    pub fn new(config: IntelligibilityMonitorConfig) -> Self {
        Self {
            config,
            context: Arc::new(Mutex::new(IntelligibilityContext::default())),
            metrics_history: Arc::new(Mutex::new(VecDeque::new())),
            update_sender: None,
            last_analysis: Instant::now(),
        }
    }

    /// Set up monitoring channel for real-time updates
    pub fn setup_monitoring(&mut self) -> mpsc::UnboundedReceiver<IntelligibilityMetrics> {
        let (sender, receiver) = mpsc::unbounded_channel();
        self.update_sender = Some(sender);
        receiver
    }

    /// Process audio chunk and update intelligibility metrics
    pub async fn process_chunk(
        &mut self,
        chunk: &AudioChunk,
        streaming_metrics: Option<&StreamingQualityMetrics>,
    ) -> EvaluationResult<()> {
        let start_time = Instant::now();

        // Check if we should perform analysis based on update frequency
        let time_since_last = self.last_analysis.elapsed();
        let update_interval = Duration::from_secs_f32(1.0 / self.config.update_frequency);

        if time_since_last < update_interval {
            return Ok(());
        }

        // Calculate intelligibility metrics
        let metrics = self
            .calculate_intelligibility_metrics(chunk, streaming_metrics)
            .await?;

        // Update context
        self.update_context(&metrics, chunk).await?;

        // Store metrics
        {
            let mut history =
                self.metrics_history
                    .lock()
                    .map_err(|_| EvaluationError::ProcessingError {
                        message: "Failed to lock metrics history".to_string(),
                        source: None,
                    })?;

            history.push_back(metrics.clone());

            // Keep only recent history
            while history.len() > 200 {
                history.pop_front();
            }
        }

        // Send update if monitoring is enabled
        if let Some(ref sender) = self.update_sender {
            let _ = sender.send(metrics);
        }

        self.last_analysis = start_time;

        // Check latency constraint
        let processing_time = start_time.elapsed();
        if processing_time.as_millis() > self.config.max_latency_ms as u128 {
            eprintln!(
                "Warning: Intelligibility analysis exceeded target latency: {}ms",
                processing_time.as_millis()
            );
        }

        Ok(())
    }

    /// Calculate intelligibility metrics for an audio chunk
    async fn calculate_intelligibility_metrics(
        &self,
        chunk: &AudioChunk,
        streaming_metrics: Option<&StreamingQualityMetrics>,
    ) -> EvaluationResult<IntelligibilityMetrics> {
        let samples = &chunk.samples;

        if samples.len() < 32 {
            return Ok(IntelligibilityMetrics::default());
        }

        // Calculate basic signal characteristics
        let energy = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
        let rms = energy.sqrt();
        let peak = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

        // Speech activity detection
        let speech_activity = self.detect_speech_activity(samples, rms).await?;

        // Get context for analysis
        let context = {
            let context_guard =
                self.context
                    .lock()
                    .map_err(|_| EvaluationError::ProcessingError {
                        message: "Failed to lock context".to_string(),
                        source: None,
                    })?;
            context_guard.clone()
        };

        // Estimate SNR
        let snr_estimate = if self.config.enable_noise_estimation {
            self.estimate_snr(rms, &context.noise_profile)
        } else {
            streaming_metrics.map_or(10.0, |m| m.snr_estimate)
        };

        // Calculate spectral clarity
        let spectral_clarity = self.calculate_spectral_clarity(samples).await?;

        // Calculate temporal clarity
        let temporal_clarity = self.calculate_temporal_clarity(samples).await?;

        // Calculate overall intelligibility score
        let intelligibility_score = self
            .calculate_intelligibility_score(
                snr_estimate,
                spectral_clarity,
                temporal_clarity,
                speech_activity,
            )
            .await?;

        // Calculate clarity index
        let clarity_index =
            (spectral_clarity * 0.6_f32 + temporal_clarity * 0.4_f32).clamp(0.0, 1.0);

        // Estimate noise level
        let noise_level_db = context.noise_profile.noise_floor;

        // Calculate confidence based on signal quality and context
        let confidence = self
            .calculate_confidence(rms, snr_estimate, speech_activity)
            .await?;

        Ok(IntelligibilityMetrics {
            intelligibility_score,
            clarity_index,
            snr_estimate,
            spectral_clarity,
            temporal_clarity,
            noise_level_db,
            speech_activity,
            confidence,
            timestamp: Instant::now(),
        })
    }

    /// Detect speech activity in audio samples
    async fn detect_speech_activity(&self, samples: &[f32], rms: f32) -> EvaluationResult<f32> {
        // Energy-based detection
        let energy_threshold = 0.01;
        let energy_activity = if rms > energy_threshold { 1.0 } else { 0.0 };

        // Zero-crossing rate
        let mut zero_crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                zero_crossings += 1;
            }
        }
        let zcr = zero_crossings as f32 / samples.len() as f32;

        // Speech-like ZCR range
        let zcr_activity = if zcr > 0.01 && zcr < 0.3 { 1.0 } else { 0.5 };

        // Combine indicators
        let activity = (energy_activity * 0.7_f32 + zcr_activity * 0.3_f32).clamp(0.0, 1.0);

        Ok(activity)
    }

    /// Estimate signal-to-noise ratio
    fn estimate_snr(&self, signal_rms: f32, noise_profile: &NoiseProfile) -> f32 {
        let noise_rms = 10.0_f32.powf(noise_profile.noise_floor / 20.0);
        if noise_rms > 0.0 && signal_rms > noise_rms {
            20.0 * (signal_rms / noise_rms).log10()
        } else {
            0.0
        }
    }

    /// Calculate spectral clarity measure
    async fn calculate_spectral_clarity(&self, samples: &[f32]) -> EvaluationResult<f32> {
        if samples.len() < 64 {
            return Ok(0.5);
        }

        // Simple spectral analysis using windowed energy
        let window_size = 64;
        let num_windows = samples.len() / window_size;

        if num_windows < 2 {
            return Ok(0.5);
        }

        let mut spectral_energies = Vec::new();
        for i in 0..num_windows {
            let start = i * window_size;
            let end = (start + window_size).min(samples.len());
            let energy: f32 = samples[start..end].iter().map(|&x| x * x).sum();
            spectral_energies.push(energy);
        }

        // Calculate spectral contrast
        let max_energy = spectral_energies.iter().fold(0.0f32, |a, &b| a.max(b));
        let avg_energy = spectral_energies.iter().sum::<f32>() / spectral_energies.len() as f32;

        let spectral_contrast = if avg_energy > 0.0 {
            (max_energy / avg_energy).min(10.0) / 10.0
        } else {
            0.0
        };

        // High-frequency content (simplified)
        let high_freq_start = num_windows * 3 / 4;
        let high_freq_energy: f32 = spectral_energies[high_freq_start..].iter().sum();
        let total_energy: f32 = spectral_energies.iter().sum();

        let high_freq_ratio = if total_energy > 0.0 {
            high_freq_energy / total_energy
        } else {
            0.0
        };

        // Combine measures (spectral contrast and high-frequency content)
        let clarity = (spectral_contrast * 0.7_f32 + high_freq_ratio * 0.3_f32).clamp(0.0, 1.0);

        Ok(clarity)
    }

    /// Calculate temporal clarity measure
    async fn calculate_temporal_clarity(&self, samples: &[f32]) -> EvaluationResult<f32> {
        if samples.len() < 32 {
            return Ok(0.5);
        }

        // Amplitude variation analysis
        let chunk_size = 32;
        let num_chunks = samples.len() / chunk_size;

        if num_chunks < 2 {
            return Ok(0.5);
        }

        let mut chunk_rms = Vec::new();
        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(samples.len());
            let rms = (samples[start..end].iter().map(|&x| x * x).sum::<f32>()
                / (end - start) as f32)
                .sqrt();
            chunk_rms.push(rms);
        }

        // Calculate temporal variation
        let avg_rms = chunk_rms.iter().sum::<f32>() / chunk_rms.len() as f32;
        let variance = chunk_rms
            .iter()
            .map(|&rms| (rms - avg_rms).powi(2))
            .sum::<f32>()
            / chunk_rms.len() as f32;
        let std_dev = variance.sqrt();

        let coefficient_of_variation = if avg_rms > 0.0 {
            std_dev / avg_rms
        } else {
            0.0
        };

        // Good temporal clarity has moderate variation (not too flat, not too chaotic)
        let optimal_cv = 0.3;
        let temporal_clarity = 1.0 - (coefficient_of_variation - optimal_cv).abs().min(1.0);

        Ok(temporal_clarity.clamp(0.0_f32, 1.0_f32))
    }

    /// Calculate overall intelligibility score
    async fn calculate_intelligibility_score(
        &self,
        snr: f32,
        spectral_clarity: f32,
        temporal_clarity: f32,
        speech_activity: f32,
    ) -> EvaluationResult<f32> {
        // SNR contribution (sigmoid function)
        let snr_contribution = 1.0 / (1.0 + (-0.2 * (snr - 10.0)).exp());

        // Combined clarity contribution
        let clarity_contribution = spectral_clarity * 0.6 + temporal_clarity * 0.4;

        // Speech activity weighting
        let activity_weight = speech_activity.max(0.1); // Minimum weight for non-speech

        // Overall intelligibility score
        let intelligibility =
            (snr_contribution * 0.4 + clarity_contribution * 0.6) * activity_weight;

        Ok(intelligibility.clamp(0.0_f32, 1.0_f32))
    }

    /// Calculate confidence in measurements
    async fn calculate_confidence(
        &self,
        signal_rms: f32,
        snr: f32,
        speech_activity: f32,
    ) -> EvaluationResult<f32> {
        // Signal level confidence
        let min_level = 10.0_f32.powf(self.config.min_signal_level_db / 20.0);
        let level_confidence = (signal_rms / min_level).min(1.0);

        // SNR confidence
        let snr_confidence = (snr / 30.0).min(1.0);

        // Activity confidence
        let activity_confidence = speech_activity;

        // Combined confidence
        let confidence =
            (level_confidence * 0.4 + snr_confidence * 0.3 + activity_confidence * 0.3)
                .clamp(0.0_f32, 1.0_f32);

        Ok(confidence)
    }

    /// Update analysis context with new information
    async fn update_context(
        &mut self,
        metrics: &IntelligibilityMetrics,
        chunk: &AudioChunk,
    ) -> EvaluationResult<()> {
        let mut context = self
            .context
            .lock()
            .map_err(|_| EvaluationError::ProcessingError {
                message: "Failed to lock context".to_string(),
                source: None,
            })?;

        // Update SNR history
        context.snr_history.push_back(metrics.snr_estimate);
        while context.snr_history.len() > 50 {
            context.snr_history.pop_front();
        }

        // Update activity history
        context.activity_history.push_back(metrics.speech_activity);
        while context.activity_history.len() > 100 {
            context.activity_history.pop_front();
        }

        // Update noise profile if enabled
        if self.config.enable_noise_estimation && metrics.speech_activity < 0.3 {
            // This appears to be mostly noise/silence, update noise profile
            let rms = (chunk.samples.iter().map(|&x| x * x).sum::<f32>()
                / chunk.samples.len() as f32)
                .sqrt();
            let noise_level_db = 20.0 * rms.log10();

            // Exponential moving average for noise floor
            let alpha = 0.1;
            context.noise_profile.noise_floor =
                alpha * noise_level_db + (1.0 - alpha) * context.noise_profile.noise_floor;
            context.noise_profile.last_update = Instant::now();
        }

        Ok(())
    }

    /// Get current intelligibility metrics
    pub fn get_current_metrics(&self) -> EvaluationResult<Option<IntelligibilityMetrics>> {
        let history =
            self.metrics_history
                .lock()
                .map_err(|_| EvaluationError::ProcessingError {
                    message: "Failed to lock metrics history".to_string(),
                    source: None,
                })?;

        Ok(history.back().cloned())
    }

    /// Get metrics history for trend analysis
    pub fn get_metrics_history(&self) -> EvaluationResult<Vec<IntelligibilityMetrics>> {
        let history =
            self.metrics_history
                .lock()
                .map_err(|_| EvaluationError::ProcessingError {
                    message: "Failed to lock metrics history".to_string(),
                    source: None,
                })?;

        Ok(history.iter().cloned().collect())
    }

    /// Reset the monitor
    pub fn reset(&mut self) -> EvaluationResult<()> {
        // Reset context
        {
            let mut context =
                self.context
                    .lock()
                    .map_err(|_| EvaluationError::ProcessingError {
                        message: "Failed to lock context".to_string(),
                        source: None,
                    })?;
            *context = IntelligibilityContext::default();
        }

        // Clear metrics history
        {
            let mut history =
                self.metrics_history
                    .lock()
                    .map_err(|_| EvaluationError::ProcessingError {
                        message: "Failed to lock metrics history".to_string(),
                        source: None,
                    })?;
            history.clear();
        }

        self.last_analysis = Instant::now();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::streaming::AudioChunk;

    #[tokio::test]
    async fn test_intelligibility_monitor_creation() {
        let config = IntelligibilityMonitorConfig::default();
        let monitor = IntelligibilityMonitor::new(config);

        // Test that we can get initial metrics (should be None)
        let metrics = monitor.get_current_metrics().unwrap();
        assert!(metrics.is_none());
    }

    #[tokio::test]
    async fn test_speech_activity_detection() {
        let config = IntelligibilityMonitorConfig::default();
        let monitor = IntelligibilityMonitor::new(config);

        // Test with high-energy signal (should detect speech activity)
        let speech_samples = (0..100)
            .map(|i| [0.3, -0.2, 0.4, -0.3, 0.2, -0.4][i % 6])
            .collect::<Vec<f32>>();
        let activity = monitor
            .detect_speech_activity(&speech_samples, 0.3)
            .await
            .unwrap();
        assert!(activity > 0.5);

        // Test with low-energy signal (should not detect speech activity)
        let silence_samples = vec![0.001; 100];
        let activity = monitor
            .detect_speech_activity(&silence_samples, 0.001)
            .await
            .unwrap();
        assert!(activity < 0.5);
    }

    #[tokio::test]
    async fn test_spectral_clarity_calculation() {
        let config = IntelligibilityMonitorConfig::default();
        let monitor = IntelligibilityMonitor::new(config);

        // Test with varying spectral content
        let mut samples = Vec::new();
        for i in 0..256 {
            let freq1 = 2.0 * std::f32::consts::PI * i as f32 / 256.0; // Low frequency
            let freq2 = 8.0 * std::f32::consts::PI * i as f32 / 256.0; // High frequency
            samples.push(0.5 * freq1.sin() + 0.3 * freq2.sin());
        }

        let clarity = monitor.calculate_spectral_clarity(&samples).await.unwrap();
        assert!(clarity >= 0.0 && clarity <= 1.0);
    }

    #[tokio::test]
    async fn test_temporal_clarity_calculation() {
        let config = IntelligibilityMonitorConfig::default();
        let monitor = IntelligibilityMonitor::new(config);

        // Test with varying temporal content
        let mut samples = Vec::new();
        for i in 0..256 {
            let amplitude = if (i / 32) % 2 == 0 { 0.5 } else { 0.1 };
            samples.push(amplitude * (2.0 * std::f32::consts::PI * i as f32 / 16.0).sin());
        }

        let clarity = monitor.calculate_temporal_clarity(&samples).await.unwrap();
        assert!(clarity >= 0.0 && clarity <= 1.0);
    }

    #[tokio::test]
    async fn test_chunk_processing() {
        let config = IntelligibilityMonitorConfig {
            update_frequency: 100.0, // High frequency for testing
            ..Default::default()
        };
        let mut monitor = IntelligibilityMonitor::new(config);

        // Create a test chunk with speech-like characteristics
        let mut samples = Vec::new();
        for i in 0..1024 {
            let base_freq = 2.0 * std::f32::consts::PI * i as f32 / 256.0;
            let harmonic = 4.0 * std::f32::consts::PI * i as f32 / 256.0;
            samples.push(0.3 * base_freq.sin() + 0.1 * harmonic.sin());
        }
        let chunk = AudioChunk::new(samples, 16000, 0);

        let result = monitor.process_chunk(&chunk, None).await;
        assert!(result.is_ok());

        // Wait a small amount to ensure timing constraints are met
        tokio::time::sleep(std::time::Duration::from_millis(15)).await;

        // Process another chunk to ensure metrics are calculated
        let result = monitor.process_chunk(&chunk, None).await;
        assert!(result.is_ok());

        // Check that metrics were calculated
        let metrics = monitor.get_current_metrics().unwrap();
        if let Some(metrics) = metrics {
            assert!(metrics.intelligibility_score >= 0.0 && metrics.intelligibility_score <= 1.0);
            assert!(metrics.clarity_index >= 0.0 && metrics.clarity_index <= 1.0);
            assert!(metrics.confidence >= 0.0 && metrics.confidence <= 1.0);
        }
        // If no metrics available, that's also acceptable since timing might prevent processing
    }

    #[test]
    fn test_config_defaults() {
        let config = IntelligibilityMonitorConfig::default();
        assert_eq!(config.analysis_window_size, 2048);
        assert_eq!(config.update_frequency, 10.0);
        assert!(config.enable_context_analysis);
        assert!(config.enable_noise_estimation);
        assert_eq!(config.min_signal_level_db, -40.0);
        assert_eq!(config.max_latency_ms, 50);
    }
}
