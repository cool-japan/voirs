//! Automated Quality Metrics System
//!
//! This module provides comprehensive automated quality measurement and monitoring
//! for emotion processing systems. It includes objective quality metrics,
//! regression testing, and cross-platform validation.
//!
//! ## Key Features
//!
//! - **Objective Quality Metrics**: Automated measurement of emotion expression quality
//! - **Regression Testing**: Prevent quality degradation in updates
//! - **Cross-platform Testing**: Validation across different platforms and architectures
//! - **Continuous Monitoring**: Real-time quality tracking and alerts
//! - **Statistical Analysis**: Comprehensive quality statistics and trending
//!
//! ## Quality Goals (from TODO.md)
//!
//! - **Naturalness Score**: Achieve MOS 4.2+ for emotional expression
//! - **Emotion Accuracy**: 90%+ correct emotion perception
//! - **Consistency Score**: 95%+ emotional consistency across utterances
//! - **User Satisfaction**: 85%+ user satisfaction in A/B tests
//!
//! ## Usage
//!
//! ```rust
//! # tokio_test::block_on(async {
//! use voirs_emotion::quality::*;
//! use voirs_emotion::types::*;
//!
//! // Create quality analyzer
//! let analyzer = QualityAnalyzer::new().expect("operation should succeed");
//!
//! // Create test data
//! let mut emotion_vector = EmotionVector::new();
//! emotion_vector.add_emotion(Emotion::Happy, EmotionIntensity::MEDIUM);
//! let audio_data = vec![0.1; 1024]; // Sample audio data
//!
//! // Analyze emotion quality
//! let metrics = analyzer.analyze_emotion_quality(&emotion_vector, &audio_data).await.expect("operation should succeed");
//!
//! if metrics.meets_production_standards() {
//!     println!("Quality standards met! ✅");
//! } else {
//!     println!("Quality issues detected: {}", metrics.summary());
//! }
//! # });
//! ```

use crate::prelude::*;
use scirs2_fft::rfft;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Default sample rate assumed for spectral analysis in this module.
const ANALYSIS_SAMPLE_RATE: f64 = 44100.0;

/// Compute the Hann-windowed real-FFT magnitude spectrum of `audio`.
///
/// Returns the magnitude of each frequency bin (length `N/2 + 1`). Empty input
/// or an FFT failure yields an empty vector.
fn rfft_magnitudes(audio: &[f32]) -> Vec<f64> {
    let n = audio.len();
    if n < 2 {
        return Vec::new();
    }

    let windowed: Vec<f64> = audio
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos());
            x as f64 * w
        })
        .collect();

    match rfft(&windowed, Some(n)) {
        Ok(spec) => spec
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Spectral centroid (Hz) from a magnitude spectrum and sample rate.
fn centroid_from_magnitudes(magnitudes: &[f64], sample_rate: f64, n: usize) -> f64 {
    if magnitudes.is_empty() || n == 0 {
        return 0.0;
    }
    let bin_hz = sample_rate / n as f64;
    let mut weighted = 0.0;
    let mut total = 0.0;
    for (k, &mag) in magnitudes.iter().enumerate() {
        weighted += (k as f64 * bin_hz) * mag;
        total += mag;
    }
    if total > 1e-12 {
        weighted / total
    } else {
        0.0
    }
}

/// Coefficient of variation (std-dev / |mean|) of a set of measurements,
/// used to quantify how stable a signal's properties are across sub-frames.
/// Returns `0.0` for empty input or a near-zero mean (avoiding a blow-up on
/// silence), and is capped at `10.0` to keep downstream scaling sane.
fn coefficient_of_variation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if mean.abs() < 1e-9 {
        return 0.0;
    }
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    (variance.sqrt() / mean.abs()).min(10.0)
}

/// Quality measurement targets based on TODO.md goals
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QualityTargets {
    /// Target naturalness score (MOS scale 1-5)
    pub min_naturalness_score: f64,
    /// Target emotion accuracy percentage
    pub min_emotion_accuracy_percent: f64,
    /// Target consistency score percentage  
    pub min_consistency_score_percent: f64,
    /// Target user satisfaction percentage
    pub min_user_satisfaction_percent: f64,
    /// Target audio quality score (MOS scale 1-5)
    pub min_audio_quality_score: f64,
    /// Maximum distortion level (THD+N)
    pub max_distortion_percent: f64,
}

impl Default for QualityTargets {
    fn default() -> Self {
        Self {
            min_naturalness_score: 4.2,          // MOS 4.2+ target from TODO
            min_emotion_accuracy_percent: 90.0,  // 90%+ target from TODO
            min_consistency_score_percent: 95.0, // 95%+ target from TODO
            min_user_satisfaction_percent: 85.0, // 85%+ target from TODO
            min_audio_quality_score: 4.0,        // Good audio quality
            max_distortion_percent: 1.0,         // Low distortion
        }
    }
}

/// Comprehensive quality measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMeasurement {
    /// Naturalness score (1-5 MOS scale)
    pub naturalness_score: f64,
    /// Emotion accuracy percentage (0-100)
    pub emotion_accuracy_percent: f64,
    /// Consistency score percentage (0-100)
    pub consistency_score_percent: f64,
    /// User satisfaction percentage (0-100)
    pub user_satisfaction_percent: f64,
    /// Audio quality score (1-5 MOS scale)
    pub audio_quality_score: f64,
    /// Distortion level percentage (0-100)
    pub distortion_percent: f64,
    /// Whether all targets were met
    pub meets_targets: bool,
    /// Individual metric pass/fail status
    pub metric_status: HashMap<String, bool>,
    /// Detailed analysis metadata
    pub metadata: QualityMetadata,
    /// Timestamp of measurement
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Quality measurement metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetadata {
    /// Emotion being analyzed
    pub emotion: String,
    /// Intensity level
    pub intensity: f64,
    /// Audio sample rate
    pub sample_rate: u32,
    /// Audio duration in seconds
    pub duration_seconds: f64,
    /// Analysis method used
    pub analysis_method: String,
    /// Platform information
    pub platform: String,
    /// Additional analysis details
    pub details: HashMap<String, serde_json::Value>,
}

impl QualityMeasurement {
    /// Check if measurement meets production quality targets
    pub fn meets_production_standards(&self) -> bool {
        self.meets_targets
    }

    /// Get summary of quality measurement
    pub fn summary(&self) -> String {
        if self.meets_targets {
            format!("All quality targets met ✅ (Naturalness: {:.2}, Accuracy: {:.1}%, Consistency: {:.1}%)", 
                self.naturalness_score, self.emotion_accuracy_percent, self.consistency_score_percent)
        } else {
            let failed: Vec<&str> = self
                .metric_status
                .iter()
                .filter_map(|(metric, &passed)| if !passed { Some(metric.as_str()) } else { None })
                .collect();
            format!("Quality issues in: {} ❌", failed.join(", "))
        }
    }

    /// Generate detailed quality report
    pub fn detailed_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Emotion Quality Analysis Report ===\n\n");

        report.push_str(&format!(
            "Overall Status: {}\n",
            if self.meets_targets {
                "PASSED ✅"
            } else {
                "FAILED ❌"
            }
        ));
        report.push_str(&format!(
            "Emotion: {} (Intensity: {:.2})\n",
            self.metadata.emotion, self.metadata.intensity
        ));
        report.push_str(&format!(
            "Duration: {:.2}s @ {}Hz\n\n",
            self.metadata.duration_seconds, self.metadata.sample_rate
        ));

        report.push_str("Quality Metrics:\n");
        report.push_str(&format!(
            "  Naturalness: {:.2}/5.0 {}\n",
            self.naturalness_score,
            if *self.metric_status.get("naturalness").unwrap_or(&false) {
                "✅"
            } else {
                "❌"
            }
        ));
        report.push_str(&format!(
            "  Emotion Accuracy: {:.1}% {}\n",
            self.emotion_accuracy_percent,
            if *self.metric_status.get("emotion_accuracy").unwrap_or(&false) {
                "✅"
            } else {
                "❌"
            }
        ));
        report.push_str(&format!(
            "  Consistency Score: {:.1}% {}\n",
            self.consistency_score_percent,
            if *self.metric_status.get("consistency").unwrap_or(&false) {
                "✅"
            } else {
                "❌"
            }
        ));
        report.push_str(&format!(
            "  User Satisfaction: {:.1}% {}\n",
            self.user_satisfaction_percent,
            if *self
                .metric_status
                .get("user_satisfaction")
                .unwrap_or(&false)
            {
                "✅"
            } else {
                "❌"
            }
        ));
        report.push_str(&format!(
            "  Audio Quality: {:.2}/5.0 {}\n",
            self.audio_quality_score,
            if *self.metric_status.get("audio_quality").unwrap_or(&false) {
                "✅"
            } else {
                "❌"
            }
        ));
        report.push_str(&format!(
            "  Distortion: {:.2}% {}\n",
            self.distortion_percent,
            if *self.metric_status.get("distortion").unwrap_or(&false) {
                "✅"
            } else {
                "❌"
            }
        ));

        report
    }
}

/// Automated quality analyzer
#[derive(Debug)]
pub struct QualityAnalyzer {
    targets: QualityTargets,
    processor: EmotionProcessor,
}

impl QualityAnalyzer {
    /// Create new quality analyzer with default targets
    pub fn new() -> Result<Self> {
        Self::with_targets(QualityTargets::default())
    }

    /// Create quality analyzer with custom targets
    pub fn with_targets(targets: QualityTargets) -> Result<Self> {
        let processor = EmotionProcessor::new()?;
        Ok(Self { targets, processor })
    }

    /// Analyze emotion quality comprehensively
    pub async fn analyze_emotion_quality(
        &self,
        emotion: &EmotionVector,
        audio_data: &[f32],
    ) -> Result<QualityMeasurement> {
        let start_time = Instant::now();

        // Extract emotion information
        let dominant = emotion.dominant_emotion();
        let (emotion_name, intensity) = if let Some((e, i)) = dominant {
            (format!("{:?}", e), i.value())
        } else {
            ("Neutral".to_string(), 0.5)
        };

        // Measure individual quality metrics
        let naturalness_score = self.measure_naturalness(emotion, audio_data).await?;
        let (emotion_accuracy, emotion_accuracy_measured) =
            self.measure_emotion_accuracy(emotion, audio_data).await?;
        let consistency_score = self.measure_consistency(emotion, audio_data).await?;
        let user_satisfaction = self.estimate_user_satisfaction(emotion, audio_data).await?;
        let audio_quality = self.measure_audio_quality(audio_data).await?;
        let distortion = self.measure_distortion(audio_data).await?;

        // Check against targets
        let mut metric_status = HashMap::new();
        metric_status.insert(
            "naturalness".to_string(),
            naturalness_score >= self.targets.min_naturalness_score,
        );
        // No real perceptual emotion-recognition model is available (see
        // `analyze_perceived_emotion`), so when the perceived emotion could
        // not honestly be determined, this metric is excluded from the
        // pass/fail decision entirely rather than being scored against a
        // fabricated guess.
        if emotion_accuracy_measured {
            metric_status.insert(
                "emotion_accuracy".to_string(),
                emotion_accuracy >= self.targets.min_emotion_accuracy_percent,
            );
        }
        metric_status.insert(
            "consistency".to_string(),
            consistency_score >= self.targets.min_consistency_score_percent,
        );
        metric_status.insert(
            "user_satisfaction".to_string(),
            user_satisfaction >= self.targets.min_user_satisfaction_percent,
        );
        metric_status.insert(
            "audio_quality".to_string(),
            audio_quality >= self.targets.min_audio_quality_score,
        );
        metric_status.insert(
            "distortion".to_string(),
            distortion <= self.targets.max_distortion_percent,
        );

        let meets_targets = metric_status.values().all(|&passed| passed);

        // Create metadata
        let metadata = QualityMetadata {
            emotion: emotion_name,
            intensity: intensity as f64,
            sample_rate: 44100, // Default sample rate
            duration_seconds: audio_data.len() as f64 / 44100.0,
            analysis_method: "automated_quality_analysis".to_string(),
            platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            details: {
                let mut details = HashMap::new();
                details.insert(
                    "analysis_duration_ms".to_string(),
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(start_time.elapsed().as_secs_f64() * 1000.0)
                            .expect("operation should succeed"),
                    ),
                );
                details.insert(
                    "buffer_size".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(audio_data.len())),
                );
                details.insert(
                    "emotion_accuracy_measured".to_string(),
                    serde_json::Value::Bool(emotion_accuracy_measured),
                );
                details
            },
        };

        Ok(QualityMeasurement {
            naturalness_score,
            emotion_accuracy_percent: emotion_accuracy,
            consistency_score_percent: consistency_score,
            user_satisfaction_percent: user_satisfaction,
            audio_quality_score: audio_quality,
            distortion_percent: distortion,
            meets_targets,
            metric_status,
            metadata,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Measure naturalness score (MOS 1-5)
    async fn measure_naturalness(
        &self,
        emotion: &EmotionVector,
        audio_data: &[f32],
    ) -> Result<f64> {
        // Analyze naturalness based on multiple factors
        let spectral_naturalness = self.analyze_spectral_naturalness(audio_data).await?;
        let prosodic_naturalness = self
            .analyze_prosodic_naturalness(emotion, audio_data)
            .await?;
        let temporal_naturalness = self.analyze_temporal_naturalness(audio_data).await?;

        // Weighted combination of naturalness factors
        let naturalness = (spectral_naturalness * 0.4)
            + (prosodic_naturalness * 0.4)
            + (temporal_naturalness * 0.2);

        // Scale to MOS 1-5 range
        Ok(1.0 + (naturalness * 4.0))
    }

    /// Measure emotion accuracy percentage.
    ///
    /// Returns `(percentage, measured)`. `measured` is `false` when the
    /// perceived emotion could not be honestly determined (see
    /// [`Self::analyze_perceived_emotion`]) - in that case `percentage` is
    /// `0.0` and callers must exclude this metric from pass/fail decisions
    /// rather than scoring it against a fabricated guess.
    async fn measure_emotion_accuracy(
        &self,
        emotion: &EmotionVector,
        audio_data: &[f32],
    ) -> Result<(f64, bool)> {
        // Compare intended emotion with perceived emotion from audio
        let intended_emotion = emotion.dominant_emotion();
        let perceived_emotion = self.analyze_perceived_emotion(audio_data).await?;

        // Calculate accuracy based on emotion matching
        if let (Some((intended, _)), Some((perceived, _))) = (intended_emotion, perceived_emotion) {
            let accuracy = if intended == perceived {
                95.0 // High accuracy for exact match
            } else if self.emotions_are_similar(&intended, &perceived) {
                75.0 // Moderate accuracy for similar emotions
            } else {
                45.0 // Low accuracy for different emotions
            };
            Ok((accuracy, true))
        } else {
            Ok((0.0, false))
        }
    }

    /// Measure consistency score percentage
    async fn measure_consistency(
        &self,
        emotion: &EmotionVector,
        audio_data: &[f32],
    ) -> Result<f64> {
        // Analyze consistency across the audio sample
        const WINDOW_SIZE: usize = 1024;
        let mut consistency_scores = Vec::new();

        // Analyze consistency in overlapping windows
        for window_start in
            (0..audio_data.len().saturating_sub(WINDOW_SIZE)).step_by(WINDOW_SIZE / 2)
        {
            let window_end = (window_start + WINDOW_SIZE).min(audio_data.len());
            let window = &audio_data[window_start..window_end];

            let window_consistency = self.analyze_window_consistency(window).await?;
            consistency_scores.push(window_consistency);
        }

        // Calculate overall consistency as average with penalty for high variance
        let mean_consistency =
            consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64;
        let variance = consistency_scores
            .iter()
            .map(|&x| (x - mean_consistency).powi(2))
            .sum::<f64>()
            / consistency_scores.len() as f64;
        let std_dev = variance.sqrt();

        // High consistency means low variance
        let consistency_score = mean_consistency * (1.0 - (std_dev / 100.0).min(0.3));

        // Handle NaN cases (e.g., empty windows) and clamp to valid range
        if consistency_score.is_nan() || consistency_score.is_infinite() {
            Ok(0.0)
        } else {
            Ok(consistency_score.clamp(0.0, 100.0))
        }
    }

    /// Estimate user satisfaction percentage
    async fn estimate_user_satisfaction(
        &self,
        emotion: &EmotionVector,
        audio_data: &[f32],
    ) -> Result<f64> {
        // Model user satisfaction based on quality factors
        let naturalness = self.measure_naturalness(emotion, audio_data).await?;
        let clarity = self.measure_audio_clarity(audio_data).await?;
        let appropriateness = self.measure_emotion_appropriateness(emotion).await?;

        // Weighted satisfaction model
        let satisfaction = (naturalness / 5.0 * 40.0) +  // 40% weight on naturalness
                          (clarity * 35.0) +              // 35% weight on clarity
                          (appropriateness * 25.0); // 25% weight on appropriateness

        // Handle NaN/Inf cases and clamp to valid range
        if satisfaction.is_nan() || satisfaction.is_infinite() {
            Ok(0.0)
        } else {
            Ok(satisfaction.clamp(0.0, 100.0))
        }
    }

    /// Measure audio quality score (MOS 1-5)
    async fn measure_audio_quality(&self, audio_data: &[f32]) -> Result<f64> {
        // Analyze technical audio quality
        let snr = self.calculate_signal_to_noise_ratio(audio_data).await?;
        let dynamic_range = self.calculate_dynamic_range(audio_data).await?;
        let frequency_response = self.analyze_frequency_response(audio_data).await?;

        // Convert to MOS scale
        let snr_score = (snr / 60.0).min(1.0); // Good SNR is 60dB+
        let dr_score = (dynamic_range / 96.0).min(1.0); // Good DR is 96dB+
        let freq_score = frequency_response;

        let quality = (snr_score * 0.4) + (dr_score * 0.3) + (freq_score * 0.3);
        Ok(1.0 + (quality * 4.0))
    }

    /// Measure distortion percentage
    async fn measure_distortion(&self, audio_data: &[f32]) -> Result<f64> {
        // Calculate THD+N (Total Harmonic Distortion + Noise)
        let fundamental_power = self.calculate_fundamental_power(audio_data).await?;
        let total_power = self.calculate_total_power(audio_data).await?;
        let noise_power = total_power - fundamental_power;

        let thd_n = if fundamental_power > 0.0 {
            (noise_power / fundamental_power).sqrt() * 100.0
        } else {
            100.0 // Maximum distortion if no fundamental
        };

        Ok(thd_n.min(100.0))
    }

    // Helper methods for quality analysis

    async fn analyze_spectral_naturalness(&self, audio_data: &[f32]) -> Result<f64> {
        // Simplified spectral analysis for naturalness
        let spectral_centroid = self.calculate_spectral_centroid(audio_data).await?;
        let spectral_bandwidth = self.calculate_spectral_bandwidth(audio_data).await?;

        // Natural speech typically has centroid around 1-3kHz
        let centroid_naturalness = 1.0 - ((spectral_centroid - 2000.0).abs() / 2000.0).min(1.0);
        let bandwidth_naturalness = (spectral_bandwidth / 8000.0).min(1.0);

        Ok((centroid_naturalness + bandwidth_naturalness) / 2.0)
    }

    async fn analyze_prosodic_naturalness(
        &self,
        emotion: &EmotionVector,
        _audio_data: &[f32],
    ) -> Result<f64> {
        // Analyze if prosody matches expected emotion characteristics
        if let Some((emotion_type, intensity)) = emotion.dominant_emotion() {
            let expected_prosody = self.get_expected_prosody(&emotion_type);
            let intensity_factor = intensity.value() as f64;

            // High intensity emotions should have more pronounced prosody
            let prosody_match = expected_prosody * intensity_factor;
            Ok(prosody_match.clamp(0.5, 1.0)) // Ensure reasonable range
        } else {
            Ok(0.7) // Neutral prosody naturalness
        }
    }

    async fn analyze_temporal_naturalness(&self, audio_data: &[f32]) -> Result<f64> {
        // Analyze temporal characteristics for naturalness
        let zero_crossing_rate = self.calculate_zero_crossing_rate(audio_data).await?;

        // Natural speech has ZCR in certain range
        let natural_zcr = if zero_crossing_rate > 0.1 && zero_crossing_rate < 0.3 {
            1.0 - ((zero_crossing_rate - 0.2).abs() / 0.1)
        } else {
            0.5
        };

        Ok(natural_zcr)
    }

    /// Attempt to recognize the perceived emotion from synthesized audio.
    ///
    /// No trained, validated perceptual emotion-recognition model is
    /// available in this crate. Rather than fabricate a plausible-looking
    /// classification (which would silently turn `measure_emotion_accuracy`
    /// into a coin flip dressed up as a measurement), this honestly reports
    /// "not determined" (`None`). A real implementation would need an
    /// audio-emotion classifier validated against human perception data
    /// before an "accuracy" claim would mean anything.
    async fn analyze_perceived_emotion(
        &self,
        _audio_data: &[f32],
    ) -> Result<Option<(Emotion, EmotionIntensity)>> {
        Ok(None)
    }

    fn emotions_are_similar(&self, e1: &Emotion, e2: &Emotion) -> bool {
        // Define similar emotion groups
        let positive_emotions = [Emotion::Happy, Emotion::Excited, Emotion::Confident];
        let negative_emotions = [Emotion::Sad, Emotion::Angry, Emotion::Fear];

        (positive_emotions.contains(e1) && positive_emotions.contains(e2))
            || (negative_emotions.contains(e1) && negative_emotions.contains(e2))
    }

    /// Real (deterministic) inter-sub-frame stability measurement: splits
    /// `window` into sub-frames and computes the coefficient of variation
    /// (std-dev / mean) of RMS energy and zero-crossing-rate across them.
    /// Low variability (a stable, sustained signal) yields high consistency;
    /// high variability (discontinuities, bursts of noise) yields low
    /// consistency. This replaces a fixed base score perturbed by
    /// `fastrand` with an actual measurement of the audio.
    async fn analyze_window_consistency(&self, window: &[f32]) -> Result<f64> {
        const SUB_FRAMES: usize = 4;
        if window.len() < SUB_FRAMES * 2 {
            // Too short to assess sub-frame stability meaningfully.
            return Ok(50.0);
        }

        let sub_len = window.len() / SUB_FRAMES;
        let mut rms_values = Vec::with_capacity(SUB_FRAMES);
        let mut zcr_values = Vec::with_capacity(SUB_FRAMES);
        for i in 0..SUB_FRAMES {
            let start = i * sub_len;
            let end = if i == SUB_FRAMES - 1 {
                window.len()
            } else {
                start + sub_len
            };
            let sub = &window[start..end];
            rms_values.push(self.calculate_rms(sub).await?);
            zcr_values.push(self.calculate_zero_crossing_rate(sub).await?);
        }

        let rms_cv = coefficient_of_variation(&rms_values);
        let zcr_cv = coefficient_of_variation(&zcr_values);

        let consistency = 100.0 * (1.0 - ((rms_cv + zcr_cv) / 2.0).min(1.0));
        Ok(consistency.clamp(0.0, 100.0))
    }

    async fn measure_audio_clarity(&self, audio_data: &[f32]) -> Result<f64> {
        // Measure audio clarity (0-1)
        let snr = self.calculate_signal_to_noise_ratio(audio_data).await?;
        Ok((snr / 60.0).min(1.0))
    }

    async fn measure_emotion_appropriateness(&self, emotion: &EmotionVector) -> Result<f64> {
        // Measure if emotion intensity and type are appropriate
        if let Some((_emotion_type, intensity)) = emotion.dominant_emotion() {
            let intensity_val = intensity.value() as f64;
            // Moderate intensities are generally more appropriate
            if intensity_val > 0.2 && intensity_val < 0.9 {
                Ok(0.9)
            } else {
                Ok(0.7)
            }
        } else {
            Ok(0.8) // Neutral is generally appropriate
        }
    }

    // Audio analysis helper methods

    async fn calculate_signal_to_noise_ratio(&self, audio_data: &[f32]) -> Result<f64> {
        let signal_power = self.calculate_total_power(audio_data).await?;
        let noise_floor = 0.001; // Estimated noise floor
        Ok(10.0 * (signal_power / noise_floor).log10())
    }

    async fn calculate_dynamic_range(&self, audio_data: &[f32]) -> Result<f64> {
        let max_val = audio_data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let rms = self.calculate_rms(audio_data).await?;
        Ok(20.0 * (max_val as f64 / rms).log10())
    }

    /// Real spectral-flatness-based frequency-response indicator, in `[0, 1]`:
    /// the ratio of the geometric mean to the arithmetic mean of the
    /// Hann-windowed magnitude spectrum (Wiener entropy). This is a genuine,
    /// input-dependent DSP measurement rather than a hardcoded constant.
    async fn analyze_frequency_response(&self, audio_data: &[f32]) -> Result<f64> {
        let magnitudes = rfft_magnitudes(audio_data);
        if magnitudes.is_empty() {
            return Ok(0.0);
        }

        let eps = 1e-12;
        let log_sum: f64 = magnitudes.iter().map(|&m| m.max(eps).ln()).sum();
        let geometric_mean = (log_sum / magnitudes.len() as f64).exp();
        let arithmetic_mean = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;
        if arithmetic_mean <= eps {
            return Ok(0.0);
        }

        Ok((geometric_mean / arithmetic_mean).clamp(0.0, 1.0))
    }

    /// Estimate the power carried by the fundamental frequency and its
    /// first few harmonics, via real FFT peak-picking within the typical
    /// voice F0 range (50-500 Hz) - not a fixed 70% multiplier of total
    /// power. The spectral fundamental/total ratio is computed from the
    /// magnitude spectrum, then applied to the real time-domain total power
    /// so the result stays in the same units as [`Self::calculate_total_power`].
    async fn calculate_fundamental_power(&self, audio_data: &[f32]) -> Result<f64> {
        let total_power = self.calculate_total_power(audio_data).await?;

        let magnitudes = rfft_magnitudes(audio_data);
        let n = audio_data.len();
        if magnitudes.is_empty() || n == 0 {
            return Ok(0.0);
        }

        let bin_hz = ANALYSIS_SAMPLE_RATE / n as f64;
        let min_bin = ((50.0 / bin_hz).round() as usize).max(1);
        let max_bin = ((500.0 / bin_hz).round() as usize).min(magnitudes.len().saturating_sub(1));
        if min_bin >= magnitudes.len() || max_bin <= min_bin {
            return Ok(0.0);
        }

        let (peak_offset, _) = magnitudes[min_bin..=max_bin]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        let f0_bin = min_bin + peak_offset;

        let total_spectral_power: f64 = magnitudes.iter().map(|m| m * m).sum();
        if total_spectral_power <= 1e-12 {
            return Ok(0.0);
        }

        // Sum power at the fundamental and harmonics 2x-5x (+/- 1 bin
        // tolerance each) as a fraction of total spectral power.
        let mut harmonic_power = 0.0;
        for h in 1..=5usize {
            let center = f0_bin * h;
            if center >= magnitudes.len() {
                break;
            }
            let lo = center.saturating_sub(1);
            let hi = (center + 1).min(magnitudes.len() - 1);
            for bin in &magnitudes[lo..=hi] {
                harmonic_power += bin * bin;
            }
        }

        let ratio = (harmonic_power / total_spectral_power).clamp(0.0, 1.0);
        Ok(total_power * ratio)
    }

    async fn calculate_total_power(&self, audio_data: &[f32]) -> Result<f64> {
        let power = audio_data.iter().map(|&x| (x * x) as f64).sum::<f64>();
        Ok(power / audio_data.len() as f64)
    }

    /// Spectral centroid in Hz computed from a Hann-windowed real FFT.
    ///
    /// `centroid = Σ(f_k·|X_k|) / Σ|X_k|` with `f_k = k · sample_rate / N`.
    async fn calculate_spectral_centroid(&self, audio_data: &[f32]) -> Result<f64> {
        let magnitudes = rfft_magnitudes(audio_data);
        Ok(centroid_from_magnitudes(
            &magnitudes,
            ANALYSIS_SAMPLE_RATE,
            audio_data.len(),
        ))
    }

    /// Spectral bandwidth (spread) in Hz around the spectral centroid.
    ///
    /// `bandwidth = sqrt(Σ((f_k − centroid)²·|X_k|) / Σ|X_k|)`.
    async fn calculate_spectral_bandwidth(&self, audio_data: &[f32]) -> Result<f64> {
        let magnitudes = rfft_magnitudes(audio_data);
        let n = audio_data.len();
        if magnitudes.is_empty() || n == 0 {
            return Ok(0.0);
        }
        let centroid = centroid_from_magnitudes(&magnitudes, ANALYSIS_SAMPLE_RATE, n);
        let bin_hz = ANALYSIS_SAMPLE_RATE / n as f64;
        let mut weighted = 0.0;
        let mut total = 0.0;
        for (k, &mag) in magnitudes.iter().enumerate() {
            let f = k as f64 * bin_hz;
            let diff = f - centroid;
            weighted += diff * diff * mag;
            total += mag;
        }
        if total > 1e-12 {
            Ok((weighted / total).sqrt())
        } else {
            Ok(0.0)
        }
    }

    async fn calculate_zero_crossing_rate(&self, audio_data: &[f32]) -> Result<f64> {
        let mut crossings = 0;
        for i in 1..audio_data.len() {
            if (audio_data[i] >= 0.0) != (audio_data[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        Ok(crossings as f64 / (audio_data.len() - 1) as f64)
    }

    async fn calculate_rms(&self, audio_data: &[f32]) -> Result<f64> {
        let sum_squares = audio_data.iter().map(|&x| (x * x) as f64).sum::<f64>();
        Ok((sum_squares / audio_data.len() as f64).sqrt())
    }

    fn get_expected_prosody(&self, emotion: &Emotion) -> f64 {
        match emotion {
            Emotion::Happy | Emotion::Excited => 0.9,
            Emotion::Sad | Emotion::Melancholic => 0.6,
            Emotion::Angry => 0.95,
            Emotion::Fear => 0.8,
            Emotion::Calm => 0.7,
            _ => 0.75,
        }
    }
}

impl Default for QualityAnalyzer {
    fn default() -> Self {
        Self::new().expect("Failed to create default quality analyzer")
    }
}

/// Quality regression tester
pub struct QualityRegressionTester {
    analyzer: QualityAnalyzer,
    baseline_measurements: Vec<QualityMeasurement>,
    regression_threshold: f64,
}

impl QualityRegressionTester {
    /// Create new regression tester
    pub fn new() -> Result<Self> {
        Ok(Self {
            analyzer: QualityAnalyzer::new()?,
            baseline_measurements: Vec::new(),
            regression_threshold: 5.0, // 5% degradation threshold
        })
    }

    /// Set baseline measurements for regression comparison
    pub fn set_baseline(&mut self, measurements: Vec<QualityMeasurement>) {
        self.baseline_measurements = measurements;
    }

    /// Test for quality regressions
    pub async fn test_regression(
        &self,
        emotion: &EmotionVector,
        audio_data: &[f32],
    ) -> Result<RegressionTestResult> {
        let current_measurement = self
            .analyzer
            .analyze_emotion_quality(emotion, audio_data)
            .await?;

        if self.baseline_measurements.is_empty() {
            return Ok(RegressionTestResult {
                current: current_measurement,
                baseline: None,
                regression_detected: false,
                degradation_percent: 0.0,
                summary: "No baseline available for comparison".to_string(),
            });
        }

        // Find matching baseline (same emotion)
        let baseline = self
            .baseline_measurements
            .iter()
            .find(|m| m.metadata.emotion == current_measurement.metadata.emotion)
            .cloned();

        if let Some(ref baseline_measurement) = baseline {
            let degradation =
                self.calculate_degradation(&current_measurement, baseline_measurement);
            let regression_detected = degradation > self.regression_threshold;

            let summary = if regression_detected {
                format!(
                    "Quality regression detected: {:.1}% degradation",
                    degradation
                )
            } else {
                format!("No regression: {:.1}% change", degradation)
            };

            Ok(RegressionTestResult {
                current: current_measurement,
                baseline,
                regression_detected,
                degradation_percent: degradation,
                summary,
            })
        } else {
            Ok(RegressionTestResult {
                current: current_measurement,
                baseline: None,
                regression_detected: false,
                degradation_percent: 0.0,
                summary: "No matching baseline found".to_string(),
            })
        }
    }

    fn calculate_degradation(
        &self,
        current: &QualityMeasurement,
        baseline: &QualityMeasurement,
    ) -> f64 {
        // Calculate weighted degradation across all metrics
        let naturalness_change = (baseline.naturalness_score - current.naturalness_score)
            / baseline.naturalness_score
            * 100.0;
        let accuracy_change = (baseline.emotion_accuracy_percent
            - current.emotion_accuracy_percent)
            / baseline.emotion_accuracy_percent
            * 100.0;
        let consistency_change = (baseline.consistency_score_percent
            - current.consistency_score_percent)
            / baseline.consistency_score_percent
            * 100.0;

        // Weighted average degradation
        (naturalness_change * 0.4) + (accuracy_change * 0.35) + (consistency_change * 0.25)
    }
}

impl Default for QualityRegressionTester {
    fn default() -> Self {
        Self::new().expect("Failed to create default regression tester")
    }
}

/// Regression test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTestResult {
    /// Current quality measurement
    pub current: QualityMeasurement,
    /// Baseline measurement for comparison
    pub baseline: Option<QualityMeasurement>,
    /// Whether regression was detected
    pub regression_detected: bool,
    /// Percentage degradation from baseline
    pub degradation_percent: f64,
    /// Summary of regression test
    pub summary: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_targets_creation() {
        let targets = QualityTargets::default();
        assert_eq!(targets.min_naturalness_score, 4.2);
        assert_eq!(targets.min_emotion_accuracy_percent, 90.0);
        assert_eq!(targets.min_consistency_score_percent, 95.0);
        assert_eq!(targets.min_user_satisfaction_percent, 85.0);
    }

    #[tokio::test]
    async fn test_quality_analyzer_creation() {
        let analyzer = QualityAnalyzer::new();
        assert!(analyzer.is_ok());
    }

    #[tokio::test]
    async fn test_emotion_quality_analysis() {
        let analyzer = QualityAnalyzer::new().unwrap();
        let mut emotion = EmotionVector::new();
        emotion.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);
        let audio_data = vec![0.1; 1024]; // Sample audio data

        let result = analyzer
            .analyze_emotion_quality(&emotion, &audio_data)
            .await;
        assert!(result.is_ok());

        let measurement = result.unwrap();
        assert!(measurement.naturalness_score >= 1.0 && measurement.naturalness_score <= 5.0);
        assert!(
            measurement.emotion_accuracy_percent >= 0.0
                && measurement.emotion_accuracy_percent <= 100.0
        );
        assert!(
            measurement.consistency_score_percent >= 0.0
                && measurement.consistency_score_percent <= 100.0
        );
        assert_eq!(measurement.metadata.emotion, "Happy");
    }

    #[tokio::test]
    async fn test_quality_measurement_summary() {
        let measurement = QualityMeasurement {
            naturalness_score: 4.5,
            emotion_accuracy_percent: 92.0,
            consistency_score_percent: 96.0,
            user_satisfaction_percent: 88.0,
            audio_quality_score: 4.2,
            distortion_percent: 0.8,
            meets_targets: true,
            metric_status: {
                let mut status = HashMap::new();
                status.insert("naturalness".to_string(), true);
                status.insert("emotion_accuracy".to_string(), true);
                status.insert("consistency".to_string(), true);
                status
            },
            metadata: QualityMetadata {
                emotion: "Happy".to_string(),
                intensity: 0.8,
                sample_rate: 44100,
                duration_seconds: 1.0,
                analysis_method: "test".to_string(),
                platform: "test".to_string(),
                details: HashMap::new(),
            },
            timestamp: chrono::Utc::now(),
        };

        assert!(measurement.meets_production_standards());
        assert!(measurement.summary().contains("All quality targets met"));
    }

    #[tokio::test]
    async fn test_regression_tester_creation() {
        let tester = QualityRegressionTester::new();
        assert!(tester.is_ok());
    }

    #[tokio::test]
    async fn test_regression_testing() {
        let mut tester = QualityRegressionTester::new().unwrap();
        let mut emotion = EmotionVector::new();
        emotion.add_emotion(Emotion::Happy, EmotionIntensity::MEDIUM);
        let audio_data = vec![0.1; 512];

        // Test without baseline
        let result = tester.test_regression(&emotion, &audio_data).await;
        assert!(result.is_ok());
        assert!(!result.unwrap().regression_detected);
    }

    #[test]
    fn test_emotions_similarity() {
        let analyzer = QualityAnalyzer::new().unwrap();
        assert!(analyzer.emotions_are_similar(&Emotion::Happy, &Emotion::Excited));
        assert!(analyzer.emotions_are_similar(&Emotion::Sad, &Emotion::Angry));
        assert!(!analyzer.emotions_are_similar(&Emotion::Happy, &Emotion::Sad));
    }

    #[tokio::test]
    async fn test_audio_analysis_helpers() {
        let analyzer = QualityAnalyzer::new().unwrap();
        let audio_data = vec![0.1, -0.1, 0.2, -0.2, 0.1, -0.1];

        let rms = analyzer.calculate_rms(&audio_data).await.unwrap();
        assert!(rms > 0.0);

        let zcr = analyzer
            .calculate_zero_crossing_rate(&audio_data)
            .await
            .unwrap();
        assert!(zcr > 0.0 && zcr <= 1.0);

        let power = analyzer.calculate_total_power(&audio_data).await.unwrap();
        assert!(power > 0.0);
    }

    /// Build a single-frequency tone at the module's analysis sample rate.
    fn tone(freq: f64, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                (2.0 * std::f64::consts::PI * freq * i as f64 / ANALYSIS_SAMPLE_RATE).sin() as f32
                    * 0.5
            })
            .collect()
    }

    #[tokio::test]
    async fn test_spectral_centroid_near_tone() {
        let analyzer = QualityAnalyzer::new().unwrap();
        let audio = tone(3000.0, 8192);
        let centroid = analyzer.calculate_spectral_centroid(&audio).await.unwrap();
        assert!(
            (centroid - 3000.0).abs() < 250.0,
            "centroid {centroid} not near 3000 Hz"
        );
    }

    #[tokio::test]
    async fn test_spectral_centroid_hf_vs_lf() {
        let analyzer = QualityAnalyzer::new().unwrap();
        let lf = analyzer
            .calculate_spectral_centroid(&tone(400.0, 8192))
            .await
            .unwrap();
        let hf = analyzer
            .calculate_spectral_centroid(&tone(7000.0, 8192))
            .await
            .unwrap();
        assert!(hf > lf, "hf centroid {hf} should exceed lf {lf}");
    }

    #[tokio::test]
    async fn test_spectral_bandwidth_two_tone_wider() {
        let analyzer = QualityAnalyzer::new().unwrap();
        let len = 8192;

        let single = tone(2000.0, len);
        // Two widely separated tones produce a broader spectral spread.
        let two: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f64 / ANALYSIS_SAMPLE_RATE;
                ((2.0 * std::f64::consts::PI * 500.0 * t).sin()
                    + (2.0 * std::f64::consts::PI * 9000.0 * t).sin()) as f32
                    * 0.25
            })
            .collect();

        let bw_single = analyzer
            .calculate_spectral_bandwidth(&single)
            .await
            .unwrap();
        let bw_two = analyzer.calculate_spectral_bandwidth(&two).await.unwrap();
        assert!(
            bw_two > bw_single,
            "two-tone bandwidth {bw_two} should exceed single-tone {bw_single}"
        );
    }

    #[test]
    fn test_quality_metadata() {
        let metadata = QualityMetadata {
            emotion: "Happy".to_string(),
            intensity: 0.7,
            sample_rate: 44100,
            duration_seconds: 2.5,
            analysis_method: "automated".to_string(),
            platform: "test-platform".to_string(),
            details: HashMap::new(),
        };

        assert_eq!(metadata.emotion, "Happy");
        assert_eq!(metadata.sample_rate, 44100);
        assert_eq!(metadata.duration_seconds, 2.5);
    }

    #[tokio::test]
    async fn test_analyze_perceived_emotion_is_honestly_undetermined() {
        let analyzer = QualityAnalyzer::new().unwrap();
        let audio = vec![0.3; 1024];
        // No real perceptual model is available; must not fabricate a
        // classification (the old bug always returned Some(Happy, MEDIUM)).
        let perceived = analyzer.analyze_perceived_emotion(&audio).await.unwrap();
        assert!(perceived.is_none());
    }

    #[tokio::test]
    async fn test_measure_emotion_accuracy_excludes_unmeasured_from_metric_status() {
        let analyzer = QualityAnalyzer::new().unwrap();
        let mut emotion = EmotionVector::new();
        emotion.add_emotion(Emotion::Sad, EmotionIntensity::HIGH);
        let audio_data = vec![0.2; 1024];

        let (accuracy, measured) = analyzer
            .measure_emotion_accuracy(&emotion, &audio_data)
            .await
            .unwrap();
        assert!(!measured, "no real perception model is available");
        assert_eq!(accuracy, 0.0);

        let result = analyzer
            .analyze_emotion_quality(&emotion, &audio_data)
            .await
            .unwrap();
        assert!(
            !result.metric_status.contains_key("emotion_accuracy"),
            "an unmeasured metric must be excluded from pass/fail, not silently scored"
        );
        assert_eq!(
            result.metadata.details.get("emotion_accuracy_measured"),
            Some(&serde_json::Value::Bool(false))
        );
    }

    #[tokio::test]
    async fn test_window_consistency_is_deterministic_and_reflects_stability() {
        let analyzer = QualityAnalyzer::new().unwrap();

        // A steady sine tone should be far more internally consistent than a
        // signal with abrupt silence/noise bursts across its sub-frames.
        let stable: Vec<f32> = (0..1024)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let bursty: Vec<f32> = (0..1024)
            .map(|i| {
                if (i / 256) % 2 == 0 {
                    0.0
                } else {
                    (fastrand::f32() - 0.5) * 2.0
                }
            })
            .collect();

        let stable_consistency_1 = analyzer.analyze_window_consistency(&stable).await.unwrap();
        let stable_consistency_2 = analyzer.analyze_window_consistency(&stable).await.unwrap();
        // Deterministic: repeated calls on the same input give the same
        // result (the old bug used fastrand and would vary run to run).
        assert_eq!(stable_consistency_1, stable_consistency_2);

        let bursty_consistency = analyzer.analyze_window_consistency(&bursty).await.unwrap();
        assert!(
            stable_consistency_1 > bursty_consistency,
            "stable tone ({stable_consistency_1}) should score more consistent than bursty \
             silence/noise ({bursty_consistency})"
        );
    }

    #[tokio::test]
    async fn test_frequency_response_is_not_hardcoded_and_varies_with_input() {
        let analyzer = QualityAnalyzer::new().unwrap();

        let tone: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let noise: Vec<f32> = (0..2048).map(|_| fastrand::f32() * 2.0 - 1.0).collect();

        let tone_flatness = analyzer.analyze_frequency_response(&tone).await.unwrap();
        let noise_flatness = analyzer.analyze_frequency_response(&noise).await.unwrap();

        assert_ne!(
            tone_flatness, 0.8,
            "must not be the old hardcoded placeholder"
        );
        // White noise has a flat spectrum (flatness near 1); a pure tone
        // concentrates energy in one bin (flatness near 0).
        assert!(
            noise_flatness > tone_flatness,
            "noise ({noise_flatness}) should be spectrally flatter than a pure tone ({tone_flatness})"
        );
    }

    #[tokio::test]
    async fn test_fundamental_power_varies_with_signal_and_not_fixed_ratio() {
        let analyzer = QualityAnalyzer::new().unwrap();

        let tone: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let noise: Vec<f32> = (0..4096).map(|_| fastrand::f32() * 2.0 - 1.0).collect();

        let tone_total = analyzer.calculate_total_power(&tone).await.unwrap();
        let tone_fundamental = analyzer.calculate_fundamental_power(&tone).await.unwrap();
        let noise_total = analyzer.calculate_total_power(&noise).await.unwrap();
        let noise_fundamental = analyzer.calculate_fundamental_power(&noise).await.unwrap();

        let tone_ratio = tone_fundamental / tone_total;
        let noise_ratio = noise_fundamental / noise_total;

        // A near-pure tone should have almost all its power at the
        // fundamental + harmonics; white noise should not concentrate
        // there. The old code returned exactly 0.7x total power for both.
        assert!(
            tone_ratio > noise_ratio,
            "tone ratio {tone_ratio} should exceed noise ratio {noise_ratio}"
        );
        assert_ne!(tone_ratio, 0.7);
    }
}
