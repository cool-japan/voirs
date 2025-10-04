//! Listening test simulation for subjective quality evaluation
//!
//! This module provides comprehensive simulation of human listening tests including:
//! - Virtual listener simulation with bias modeling
//! - Listening test result generation with statistical variation
//! - Quality scale transformation between different evaluation schemes
//! - Reliability assessment of simulated results

use crate::traits::{EvaluationResult, QualityScore};
use crate::EvaluationError;
use scirs2_core::random::prelude::*;
use scirs2_core::random::thread_rng;
use std::collections::HashMap;
use voirs_sdk::AudioBuffer;

/// Listening test simulator for subjective quality evaluation
pub struct ListeningTestSimulator {
    /// Virtual listeners
    listeners: Vec<VirtualListener>,
    /// Bias model for simulation
    bias_model: BiasModel,
    /// Quality scale transformer
    scale_transformer: QualityScaleTransformer,
    /// Random number generator
    rng: ThreadRng,
}

impl ListeningTestSimulator {
    /// Create a new listening test simulator
    pub fn new() -> Self {
        Self {
            listeners: Vec::new(),
            bias_model: BiasModel::default(),
            scale_transformer: QualityScaleTransformer::default(),
            rng: thread_rng(),
        }
    }

    /// Add a virtual listener to the simulation
    pub fn add_listener(&mut self, listener: VirtualListener) {
        self.listeners.push(listener);
    }

    /// Configure bias model
    pub fn set_bias_model(&mut self, bias_model: BiasModel) {
        self.bias_model = bias_model;
    }

    /// Configure quality scale transformer
    pub fn set_scale_transformer(&mut self, transformer: QualityScaleTransformer) {
        self.scale_transformer = transformer;
    }

    /// Simulate a listening test
    pub async fn simulate_test(
        &mut self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> EvaluationResult<ListeningTestResult> {
        if self.listeners.is_empty() {
            return Err(EvaluationError::ConfigurationError {
                message: "No virtual listeners configured".to_string(),
            }
            .into());
        }

        let mut individual_scores = Vec::new();
        let mut response_patterns = Vec::new();

        // Simulate each listener's response
        for listener in &self.listeners {
            let base_score = self.calculate_base_score(audio, reference).await?;
            let biased_score = self.apply_listener_bias(base_score, listener);
            let transformed_score = self.scale_transformer.transform(biased_score);

            individual_scores.push(transformed_score);
            response_patterns.push(listener.generate_response_pattern(&mut self.rng));
        }

        // Calculate aggregated statistics
        let mean_score = individual_scores.iter().sum::<f32>() / individual_scores.len() as f32;
        let variance = individual_scores
            .iter()
            .map(|score| (score - mean_score).powi(2))
            .sum::<f32>()
            / individual_scores.len() as f32;
        let std_dev = variance.sqrt();

        // Calculate confidence interval
        let confidence_95 = 1.96 * std_dev / (individual_scores.len() as f32).sqrt();

        // Assess reliability
        let reliability = self.assess_reliability(&individual_scores, &response_patterns);

        Ok(ListeningTestResult {
            mean_score,
            std_dev,
            confidence_interval_95: (mean_score - confidence_95, mean_score + confidence_95),
            individual_scores,
            response_patterns,
            reliability,
            num_listeners: self.listeners.len(),
            test_type: "Subjective Quality Assessment".to_string(),
        })
    }

    /// Calculate base objective quality score
    async fn calculate_base_score(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> EvaluationResult<f32> {
        let sample_rate = audio.sample_rate() as f32;
        let duration = audio.len() as f32 / sample_rate;

        // Enhanced quality indicators
        let signal_power = audio.samples().iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32;
        let dynamic_range = self.calculate_dynamic_range(audio);
        let spectral_flatness = self.calculate_spectral_flatness(audio);
        let snr_estimate = self.estimate_snr(audio);
        let thd_estimate = self.estimate_thd(audio);
        let spectral_centroid = self.calculate_spectral_centroid(audio);
        let zero_crossing_rate = self.calculate_zero_crossing_rate(audio);

        // Duration-based quality assessment
        let duration_score = match duration {
            d if d >= 1.0 && d <= 8.0 => 1.0,
            d if d >= 0.5 && d < 1.0 => 0.9,
            d if d > 8.0 && d <= 15.0 => 0.85,
            _ => 0.7,
        };

        // Enhanced scoring with multiple dimensions
        let power_score = (signal_power * 10000.0).clamp(0.0, 1.0);
        let dynamic_score = (dynamic_range / 60.0).clamp(0.0, 1.0);
        let spectral_score = spectral_flatness;
        let snr_score = (snr_estimate / 40.0).clamp(0.0, 1.0);
        let thd_score = 1.0 - (thd_estimate / 0.1).clamp(0.0, 1.0);
        let centroid_score = (1.0 - (spectral_centroid - 2000.0).abs() / 8000.0).clamp(0.0, 1.0);
        let zcr_score = (1.0 - (zero_crossing_rate - 0.1).abs() / 0.3).clamp(0.0, 1.0);

        // Reference-based scoring if available
        let reference_score = if let Some(ref_audio) = reference {
            self.calculate_reference_similarity(audio, ref_audio)
        } else {
            0.8 // Neutral score when no reference
        };

        // Weighted combination of all factors
        let base_score = (power_score * 0.15
            + duration_score * 0.12
            + dynamic_score * 0.18
            + spectral_score * 0.15
            + snr_score * 0.12
            + thd_score * 0.08
            + centroid_score * 0.08
            + zcr_score * 0.05
            + reference_score * 0.07)
            .clamp(0.05, 0.98);

        Ok(base_score)
    }

    /// Calculate dynamic range of audio
    fn calculate_dynamic_range(&self, audio: &AudioBuffer) -> f32 {
        let data = audio.samples();
        if data.is_empty() {
            return 0.0;
        }

        let max_val = data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let rms = (data.iter().map(|&x| x * x).sum::<f32>() / data.len() as f32).sqrt();

        if rms > 0.0 {
            20.0 * (max_val / rms).log10()
        } else {
            0.0
        }
    }

    /// Calculate spectral flatness using FFT-based analysis
    fn calculate_spectral_flatness(&self, audio: &AudioBuffer) -> f32 {
        let data = audio.samples();
        if data.len() < 512 {
            return 0.5;
        }

        // Use a frame size that's a power of 2
        let frame_size = 512;
        let hop_size = frame_size / 2;
        let mut flatness_values = Vec::new();

        for i in (0..data.len().saturating_sub(frame_size)).step_by(hop_size) {
            let frame = &data[i..i + frame_size];

            // Apply Hann window
            let windowed: Vec<f32> = frame
                .iter()
                .enumerate()
                .map(|(j, &x)| {
                    let window = 0.5
                        * (1.0
                            - (2.0 * std::f32::consts::PI * j as f32 / (frame_size - 1) as f32)
                                .cos());
                    x * window
                })
                .collect();

            // Simple magnitude spectrum calculation (approximation)
            let mut magnitude_spectrum = Vec::with_capacity(frame_size / 2);
            for k in 0..frame_size / 2 {
                let mut real = 0.0;
                let mut imag = 0.0;
                for (n, &sample) in windowed.iter().enumerate() {
                    let phase =
                        -2.0 * std::f32::consts::PI * k as f32 * n as f32 / frame_size as f32;
                    real += sample * phase.cos();
                    imag += sample * phase.sin();
                }
                magnitude_spectrum.push((real * real + imag * imag).sqrt());
            }

            // Calculate spectral flatness for this frame
            let geometric_mean = magnitude_spectrum
                .iter()
                .filter(|&&x| x > 1e-10)
                .map(|&x| x.ln())
                .sum::<f32>()
                / magnitude_spectrum.len() as f32;
            let geometric_mean = geometric_mean.exp();

            let arithmetic_mean =
                magnitude_spectrum.iter().sum::<f32>() / magnitude_spectrum.len() as f32;

            if arithmetic_mean > 1e-10 {
                flatness_values.push(geometric_mean / arithmetic_mean);
            }
        }

        if flatness_values.is_empty() {
            0.5
        } else {
            flatness_values.iter().sum::<f32>() / flatness_values.len() as f32
        }
    }

    /// Estimate Signal-to-Noise Ratio
    fn estimate_snr(&self, audio: &AudioBuffer) -> f32 {
        let data = audio.samples();
        if data.len() < 1024 {
            return 20.0; // Default reasonable SNR
        }

        // Split into signal and noise estimation
        let mid_point = data.len() / 2;
        let signal_part = &data[mid_point / 2..mid_point + mid_point / 2];
        let noise_part = &data[0..mid_point / 4];

        let signal_power =
            signal_part.iter().map(|&x| x * x).sum::<f32>() / signal_part.len() as f32;
        let noise_power = noise_part.iter().map(|&x| x * x).sum::<f32>() / noise_part.len() as f32;

        if noise_power > 1e-10 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            40.0 // High SNR when no detectable noise
        }
    }

    /// Estimate Total Harmonic Distortion
    fn estimate_thd(&self, audio: &AudioBuffer) -> f32 {
        let data = audio.samples();
        if data.len() < 1024 {
            return 0.01; // Low distortion default
        }

        // Simplified THD estimation using harmonic analysis
        let fundamental_freq = self.estimate_fundamental_frequency(audio);
        let sample_rate = audio.sample_rate() as f32;

        if fundamental_freq < 50.0 || fundamental_freq > sample_rate / 4.0 {
            return 0.05; // Moderate distortion for non-tonal signals
        }

        // Calculate power at fundamental and harmonics
        let mut harmonic_powers = Vec::new();
        for harmonic in 1..=5 {
            let freq = fundamental_freq * harmonic as f32;
            let power = self.calculate_power_at_frequency(audio, freq);
            harmonic_powers.push(power);
        }

        if harmonic_powers[0] > 1e-10 {
            let fundamental_power = harmonic_powers[0];
            let harmonic_sum: f32 = harmonic_powers.iter().skip(1).sum();
            (harmonic_sum / fundamental_power).sqrt().min(1.0)
        } else {
            0.05
        }
    }

    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, audio: &AudioBuffer) -> f32 {
        let data = audio.samples();
        if data.len() < 512 {
            return 2000.0; // Default centroid
        }

        let frame_size = 512;
        let sample_rate = audio.sample_rate() as f32;
        let mut centroid_values = Vec::new();

        for i in (0..data.len().saturating_sub(frame_size)).step_by(frame_size / 2) {
            let frame = &data[i..i + frame_size];

            // Calculate magnitude spectrum
            let mut magnitude_spectrum = Vec::with_capacity(frame_size / 2);
            for k in 0..frame_size / 2 {
                let mut real = 0.0;
                let mut imag = 0.0;
                for (n, &sample) in frame.iter().enumerate() {
                    let phase =
                        -2.0 * std::f32::consts::PI * k as f32 * n as f32 / frame_size as f32;
                    real += sample * phase.cos();
                    imag += sample * phase.sin();
                }
                magnitude_spectrum.push((real * real + imag * imag).sqrt());
            }

            // Calculate spectral centroid
            let mut weighted_sum = 0.0;
            let mut magnitude_sum = 0.0;

            for (k, &magnitude) in magnitude_spectrum.iter().enumerate() {
                let frequency = k as f32 * sample_rate / frame_size as f32;
                weighted_sum += frequency * magnitude;
                magnitude_sum += magnitude;
            }

            if magnitude_sum > 1e-10 {
                centroid_values.push(weighted_sum / magnitude_sum);
            }
        }

        if centroid_values.is_empty() {
            2000.0
        } else {
            centroid_values.iter().sum::<f32>() / centroid_values.len() as f32
        }
    }

    /// Calculate zero crossing rate
    fn calculate_zero_crossing_rate(&self, audio: &AudioBuffer) -> f32 {
        let data = audio.samples();
        if data.len() < 2 {
            return 0.0;
        }

        let mut zero_crossings = 0;
        for i in 1..data.len() {
            if (data[i] >= 0.0) != (data[i - 1] >= 0.0) {
                zero_crossings += 1;
            }
        }

        zero_crossings as f32 / (data.len() - 1) as f32
    }

    /// Calculate similarity with reference audio
    fn calculate_reference_similarity(&self, audio: &AudioBuffer, reference: &AudioBuffer) -> f32 {
        let audio_data = audio.samples();
        let ref_data = reference.samples();

        if audio_data.len() != ref_data.len() {
            return 0.5; // Neutral score for length mismatch
        }

        // Calculate normalized cross-correlation
        let audio_mean = audio_data.iter().sum::<f32>() / audio_data.len() as f32;
        let ref_mean = ref_data.iter().sum::<f32>() / ref_data.len() as f32;

        let mut numerator = 0.0;
        let mut audio_sq_sum = 0.0;
        let mut ref_sq_sum = 0.0;

        for i in 0..audio_data.len() {
            let audio_centered = audio_data[i] - audio_mean;
            let ref_centered = ref_data[i] - ref_mean;

            numerator += audio_centered * ref_centered;
            audio_sq_sum += audio_centered * audio_centered;
            ref_sq_sum += ref_centered * ref_centered;
        }

        let denominator = (audio_sq_sum * ref_sq_sum).sqrt();
        if denominator > 1e-10 {
            ((numerator / denominator).abs()).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Estimate fundamental frequency of audio signal
    fn estimate_fundamental_frequency(&self, audio: &AudioBuffer) -> f32 {
        let data = audio.samples();
        let sample_rate = audio.sample_rate() as f32;

        if data.len() < 1024 {
            return 440.0; // Default A4
        }

        // Simplified autocorrelation-based F0 estimation
        let max_lag = (sample_rate / 50.0) as usize; // Minimum 50 Hz
        let min_lag = (sample_rate / 1000.0) as usize; // Maximum 1000 Hz

        let mut max_correlation = 0.0;
        let mut best_lag = min_lag;

        for lag in min_lag..=max_lag.min(data.len() / 2) {
            let mut correlation = 0.0;
            let mut count = 0;

            for i in 0..(data.len() - lag) {
                correlation += data[i] * data[i + lag];
                count += 1;
            }

            if count > 0 {
                correlation /= count as f32;
                if correlation > max_correlation {
                    max_correlation = correlation;
                    best_lag = lag;
                }
            }
        }

        sample_rate / best_lag as f32
    }

    /// Calculate power at specific frequency
    fn calculate_power_at_frequency(&self, audio: &AudioBuffer, frequency: f32) -> f32 {
        let data = audio.samples();
        let sample_rate = audio.sample_rate() as f32;

        if data.len() < 256 || frequency <= 0.0 {
            return 0.0;
        }

        // Calculate power using Goertzel algorithm approximation
        let k = (frequency * data.len() as f32 / sample_rate).round() as usize;
        let mut real = 0.0;
        let mut imag = 0.0;

        for (n, &sample) in data.iter().enumerate() {
            let phase = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / data.len() as f32;
            real += sample * phase.cos();
            imag += sample * phase.sin();
        }

        (real * real + imag * imag) / (data.len() as f32 * data.len() as f32)
    }

    /// Apply listener-specific bias to the score
    fn apply_listener_bias(&self, base_score: f32, listener: &VirtualListener) -> f32 {
        let mut biased_score = base_score;

        // Apply demographic bias
        biased_score += listener.age_bias * self.bias_model.age_factor;
        biased_score += listener.experience_bias * self.bias_model.experience_factor;
        biased_score += listener.cultural_bias * self.bias_model.cultural_factor;

        // Apply expertise-based bias
        let expertise_bias = match listener.expertise_level {
            ExpertiseLevel::Expert => 0.02, // Experts are slightly more critical
            ExpertiseLevel::Average => 0.0,
            ExpertiseLevel::Novice => 0.08, // Novices tend to be more lenient
        };
        biased_score += expertise_bias;

        // Apply response tendency with expertise modulation
        let tendency_strength = match listener.expertise_level {
            ExpertiseLevel::Expert => 0.05, // Experts have less bias
            ExpertiseLevel::Average => 0.1,
            ExpertiseLevel::Novice => 0.15, // Novices have stronger bias
        };

        match listener.response_tendency {
            ResponseTendency::Optimistic => biased_score += tendency_strength,
            ResponseTendency::Pessimistic => biased_score -= tendency_strength,
            ResponseTendency::Neutral => {}
        }

        // Apply session fatigue (simulates listening fatigue over time)
        let fatigue_bias = self.bias_model.fatigue_factor * thread_rng().gen_range(0.0..1.0);
        biased_score -= fatigue_bias;

        // Apply context bias (order effects, previous ratings influence)
        let context_bias = self.bias_model.context_factor * thread_rng().gen_range(-0.5..0.5);
        biased_score += context_bias;

        // Apply consistency-based random noise
        let noise_factor = 1.0 - listener.consistency;
        let noise = thread_rng().gen_range(-noise_factor..noise_factor) * 0.1;
        biased_score += noise;

        // Apply listening environment bias
        let environment_bias = self.simulate_environment_effects();
        biased_score += environment_bias;

        biased_score.clamp(0.0, 1.0)
    }

    /// Simulate environmental effects on listening
    fn simulate_environment_effects(&self) -> f32 {
        let mut rng = thread_rng();

        // Background noise effect
        let noise_level = rng.gen_range(0.0..0.3);
        let noise_bias = -noise_level * 0.1;

        // Room acoustics effect
        let reverb_level = rng.gen_range(0.0..0.5);
        let reverb_bias = -reverb_level * 0.05;

        // Equipment quality effect
        let equipment_quality = rng.gen_range(0.5..1.0);
        let equipment_bias = (equipment_quality - 0.75) * 0.2;

        // Attention level effect
        let attention = rng.gen_range(0.7..1.0);
        let attention_bias = (attention - 0.85) * 0.3;

        ((noise_bias + reverb_bias + equipment_bias + attention_bias) as f32).clamp(-0.2, 0.1)
    }

    /// Assess reliability of the test results
    fn assess_reliability(
        &self,
        scores: &[f32],
        patterns: &[ResponsePattern],
    ) -> ReliabilityAssessment {
        let mean_score = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores
            .iter()
            .map(|score| (score - mean_score).powi(2))
            .sum::<f32>()
            / scores.len() as f32;

        // Calculate Cronbach's alpha (simplified)
        let alpha = if variance > 0.0 {
            let item_variance: f32 =
                scores.iter().map(|&s| s.powi(2)).sum::<f32>() / scores.len() as f32;
            let k = scores.len() as f32;
            (k / (k - 1.0)) * (1.0 - item_variance / variance)
        } else {
            1.0
        };

        // Assess consistency
        let consistency_score = 1.0 - (variance / 0.25); // Normalize variance

        // Check for outliers
        let outlier_count = scores
            .iter()
            .filter(|&&score| (score - mean_score).abs() > 2.0 * variance.sqrt())
            .count();

        let outlier_ratio = outlier_count as f32 / scores.len() as f32;

        ReliabilityAssessment {
            cronbach_alpha: alpha.max(0.0).min(1.0),
            consistency_score: consistency_score.max(0.0).min(1.0),
            outlier_ratio,
            is_reliable: alpha > 0.7 && consistency_score > 0.6 && outlier_ratio < 0.1,
            confidence_level: ((alpha + consistency_score) / 2.0 * (1.0 - outlier_ratio))
                .max(0.0)
                .min(1.0),
        }
    }
}

impl Default for ListeningTestSimulator {
    fn default() -> Self {
        let mut simulator = Self::new();

        // Add default virtual listeners
        simulator.add_listener(VirtualListener::new_expert());
        simulator.add_listener(VirtualListener::new_average());
        simulator.add_listener(VirtualListener::new_novice());

        simulator
    }
}

/// Virtual listener with specific characteristics
#[derive(Debug, Clone)]
pub struct VirtualListener {
    /// Listener ID
    pub id: String,
    /// Age-related bias (-1.0 to 1.0)
    pub age_bias: f32,
    /// Experience-related bias (-1.0 to 1.0)
    pub experience_bias: f32,
    /// Cultural bias (-1.0 to 1.0)
    pub cultural_bias: f32,
    /// Response consistency (0.0 to 1.0)
    pub consistency: f32,
    /// Response tendency
    pub response_tendency: ResponseTendency,
    /// Listening expertise level
    pub expertise_level: ExpertiseLevel,
}

impl VirtualListener {
    /// Create a new virtual listener
    pub fn new(id: String) -> Self {
        Self {
            id,
            age_bias: 0.0,
            experience_bias: 0.0,
            cultural_bias: 0.0,
            consistency: 0.8,
            response_tendency: ResponseTendency::Neutral,
            expertise_level: ExpertiseLevel::Average,
        }
    }

    /// Create an expert listener
    pub fn new_expert() -> Self {
        Self {
            id: "expert_listener".to_string(),
            age_bias: 0.02,
            experience_bias: 0.1,
            cultural_bias: 0.0,
            consistency: 0.95,
            response_tendency: ResponseTendency::Neutral,
            expertise_level: ExpertiseLevel::Expert,
        }
    }

    /// Create an average listener
    pub fn new_average() -> Self {
        Self {
            id: "average_listener".to_string(),
            age_bias: 0.0,
            experience_bias: 0.0,
            cultural_bias: 0.05,
            consistency: 0.75,
            response_tendency: ResponseTendency::Neutral,
            expertise_level: ExpertiseLevel::Average,
        }
    }

    /// Create a novice listener
    pub fn new_novice() -> Self {
        Self {
            id: "novice_listener".to_string(),
            age_bias: -0.05,
            experience_bias: -0.1,
            cultural_bias: 0.1,
            consistency: 0.6,
            response_tendency: ResponseTendency::Optimistic,
            expertise_level: ExpertiseLevel::Novice,
        }
    }

    /// Generate response pattern for this listener
    pub fn generate_response_pattern(&self, rng: &mut ThreadRng) -> ResponsePattern {
        let base_response_time = match self.expertise_level {
            ExpertiseLevel::Expert => 2.5,
            ExpertiseLevel::Average => 4.0,
            ExpertiseLevel::Novice => 6.0,
        };

        let response_time: f32 = base_response_time + rng.gen_range(-1.0..1.0);
        let hesitation_count = rng.gen_range(0..=2);
        let revision_count = if self.consistency < 0.7 {
            rng.gen_range(0..=1)
        } else {
            0
        };

        ResponsePattern {
            response_time: response_time.max(1.0),
            hesitation_count,
            revision_count,
            confidence_level: self.consistency,
        }
    }
}

/// Response tendency of virtual listeners
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseTendency {
    /// Tendency to rate quality higher than actual
    Optimistic,
    /// Tendency to rate quality lower than actual
    Pessimistic,
    /// No particular bias in quality rating
    Neutral,
}

/// Expertise level of virtual listeners
#[derive(Debug, Clone, PartialEq)]
pub enum ExpertiseLevel {
    /// Expert listener with trained audio perception
    Expert,
    /// Average listener with normal audio perception
    Average,
    /// Novice listener with limited audio experience
    Novice,
}

/// Bias model for listening test simulation
#[derive(Debug, Clone)]
pub struct BiasModel {
    /// Age bias factor
    pub age_factor: f32,
    /// Experience bias factor
    pub experience_factor: f32,
    /// Cultural bias factor
    pub cultural_factor: f32,
    /// Session fatigue factor
    pub fatigue_factor: f32,
    /// Context bias factor
    pub context_factor: f32,
}

impl Default for BiasModel {
    fn default() -> Self {
        Self {
            age_factor: 0.1,
            experience_factor: 0.15,
            cultural_factor: 0.05,
            fatigue_factor: 0.02,
            context_factor: 0.03,
        }
    }
}

/// Response pattern analysis
#[derive(Debug, Clone)]
pub struct ResponsePattern {
    /// Response time in seconds
    pub response_time: f32,
    /// Number of hesitations
    pub hesitation_count: usize,
    /// Number of revisions
    pub revision_count: usize,
    /// Confidence level (0.0 to 1.0)
    pub confidence_level: f32,
}

/// Quality scale transformer between different evaluation schemes
#[derive(Debug, Clone)]
pub struct QualityScaleTransformer {
    /// Source scale range
    pub source_range: (f32, f32),
    /// Target scale range
    pub target_range: (f32, f32),
    /// Transformation function type
    pub transform_type: TransformType,
}

impl QualityScaleTransformer {
    /// Create a new scale transformer
    pub fn new(
        source_range: (f32, f32),
        target_range: (f32, f32),
        transform_type: TransformType,
    ) -> Self {
        Self {
            source_range,
            target_range,
            transform_type,
        }
    }

    /// Transform a score from source to target scale
    pub fn transform(&self, score: f32) -> f32 {
        // Normalize to [0, 1]
        let normalized =
            (score - self.source_range.0) / (self.source_range.1 - self.source_range.0);
        let normalized = normalized.clamp(0.0, 1.0);

        // Apply transformation function
        let transformed = match self.transform_type {
            TransformType::Linear => normalized,
            TransformType::Logarithmic => normalized.ln() / 1.0_f32.ln(),
            TransformType::Exponential => normalized.exp() / 1.0_f32.exp(),
            TransformType::Sigmoid => 1.0 / (1.0 + (-5.0 * (normalized - 0.5)).exp()),
        };

        // Scale to target range
        self.target_range.0 + transformed * (self.target_range.1 - self.target_range.0)
    }
}

impl Default for QualityScaleTransformer {
    fn default() -> Self {
        Self::new((0.0, 1.0), (0.0, 1.0), TransformType::Linear)
    }
}

/// Transformation function types
#[derive(Debug, Clone, PartialEq)]
pub enum TransformType {
    /// Linear transformation (1:1 mapping)
    Linear,
    /// Logarithmic transformation (compressed at high values)
    Logarithmic,
    /// Exponential transformation (expanded at high values)
    Exponential,
    /// Sigmoid transformation (S-curve mapping)
    Sigmoid,
}

/// Result of a simulated listening test
#[derive(Debug, Clone)]
pub struct ListeningTestResult {
    /// Mean opinion score
    pub mean_score: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// 95% confidence interval
    pub confidence_interval_95: (f32, f32),
    /// Individual listener scores
    pub individual_scores: Vec<f32>,
    /// Response patterns
    pub response_patterns: Vec<ResponsePattern>,
    /// Reliability assessment
    pub reliability: ReliabilityAssessment,
    /// Number of listeners
    pub num_listeners: usize,
    /// Test type
    pub test_type: String,
}

impl ListeningTestResult {
    /// Convert to quality score
    pub fn to_quality_score(&self) -> QualityScore {
        let mut component_scores = HashMap::new();
        component_scores.insert("mean_opinion_score".to_string(), self.mean_score);
        component_scores.insert(
            "consistency".to_string(),
            self.reliability.consistency_score,
        );
        component_scores.insert("reliability".to_string(), self.reliability.cronbach_alpha);

        let mut recommendations = Vec::new();
        if self.reliability.outlier_ratio > 0.1 {
            recommendations.push("Consider removing outlier responses".to_string());
        }
        if self.std_dev > 0.5 {
            recommendations.push("High variance detected - consider more listeners".to_string());
        }
        if self.num_listeners < 5 {
            recommendations
                .push("Consider increasing number of listeners for better reliability".to_string());
        }

        QualityScore {
            overall_score: self.mean_score,
            component_scores,
            recommendations,
            confidence: self.reliability.confidence_level,
            processing_time: None,
        }
    }
}

/// Reliability assessment of listening test results
#[derive(Debug, Clone)]
pub struct ReliabilityAssessment {
    /// Cronbach's alpha coefficient
    pub cronbach_alpha: f32,
    /// Consistency score
    pub consistency_score: f32,
    /// Ratio of outlier responses
    pub outlier_ratio: f32,
    /// Whether the test is considered reliable
    pub is_reliable: bool,
    /// Overall confidence level
    pub confidence_level: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_sdk::AudioBuffer;

    #[tokio::test]
    async fn test_listening_test_simulator() {
        let mut simulator = ListeningTestSimulator::default();

        // Create test audio
        let sample_rate = 22050;
        let duration = 2.0;
        let samples = (sample_rate as f32 * duration) as usize;
        let data: Vec<f32> = (0..samples)
            .map(|i| {
                (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1
            })
            .collect();
        let audio = AudioBuffer::new(data, sample_rate, 1);

        let result = simulator.simulate_test(&audio, None).await.unwrap();

        assert!(result.mean_score >= 0.0 && result.mean_score <= 1.0);
        assert!(result.std_dev >= 0.0);
        assert_eq!(result.individual_scores.len(), 3); // Default has 3 listeners
        assert!(
            result.reliability.cronbach_alpha >= 0.0 && result.reliability.cronbach_alpha <= 1.0
        );
    }

    #[test]
    fn test_virtual_listener_creation() {
        let expert = VirtualListener::new_expert();
        let average = VirtualListener::new_average();
        let novice = VirtualListener::new_novice();

        assert_eq!(expert.expertise_level, ExpertiseLevel::Expert);
        assert_eq!(average.expertise_level, ExpertiseLevel::Average);
        assert_eq!(novice.expertise_level, ExpertiseLevel::Novice);

        assert!(expert.consistency > average.consistency);
        assert!(average.consistency > novice.consistency);
    }

    #[test]
    fn test_quality_scale_transformer() {
        let transformer =
            QualityScaleTransformer::new((0.0, 1.0), (1.0, 5.0), TransformType::Linear);

        assert_eq!(transformer.transform(0.0), 1.0);
        assert_eq!(transformer.transform(1.0), 5.0);
        assert_eq!(transformer.transform(0.5), 3.0);
    }

    #[test]
    fn test_response_pattern_generation() {
        let expert = VirtualListener::new_expert();
        let mut rng = thread_rng();

        let pattern = expert.generate_response_pattern(&mut rng);

        assert!(pattern.response_time > 0.0);
        assert_eq!(pattern.confidence_level, expert.consistency);
    }

    #[test]
    fn test_bias_model() {
        let bias_model = BiasModel::default();

        assert!(bias_model.age_factor > 0.0);
        assert!(bias_model.experience_factor > 0.0);
        assert!(bias_model.cultural_factor > 0.0);
    }
}
