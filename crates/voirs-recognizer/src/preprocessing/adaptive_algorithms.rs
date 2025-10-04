//! Adaptive algorithms for dynamic audio processing parameter adjustment
//!
//! This module provides intelligent, adaptive processing that automatically
//! adjusts parameters based on real-time audio characteristics including:
//! - Adaptive noise suppression based on SNR estimation
//! - Dynamic gain control with speech/music detection
//! - Intelligent echo cancellation parameter adjustment
//! - Adaptive filtering based on audio content analysis

use crate::RecognitionError;
use std::collections::VecDeque;
use voirs_sdk::AudioBuffer;

/// Audio content type detected by the adaptive system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Audio Content Type
pub enum AudioContentType {
    /// Speech
    Speech,
    /// Music
    Music,
    /// Noise
    Noise,
    /// Mixed
    Mixed,
    /// Silence
    Silence,
}

/// Adaptive algorithm configuration
#[derive(Debug, Clone)]
/// Adaptive Config
pub struct AdaptiveConfig {
    /// Window size for analysis (samples)
    pub analysis_window_size: usize,
    /// Overlap between analysis windows
    pub window_overlap: f32,
    /// History length for adaptive decisions
    pub history_length: usize,
    /// Adaptation rate (0.0 - 1.0)
    pub adaptation_rate: f32,
    /// SNR threshold for noise suppression adaptation
    pub snr_threshold: f32,
    /// Speech detection sensitivity
    pub speech_detection_sensitivity: f32,
    /// Music detection sensitivity
    pub music_detection_sensitivity: f32,
    /// Sample rate
    pub sample_rate: u32,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            analysis_window_size: 2048,
            window_overlap: 0.5,
            history_length: 10,
            adaptation_rate: 0.1,
            snr_threshold: 10.0,
            speech_detection_sensitivity: 0.7,
            music_detection_sensitivity: 0.6,
            sample_rate: 16000,
        }
    }
}

/// Audio analysis features used for adaptive processing
#[derive(Debug, Clone)]
/// Audio Features
pub struct AudioFeatures {
    /// Signal-to-noise ratio (dB)
    pub snr_db: f32,
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    /// Spectral centroid (Hz)
    pub spectral_centroid: f32,
    /// Spectral rolloff (Hz)
    pub spectral_rolloff: f32,
    /// Spectral flux
    pub spectral_flux: f32,
    /// Harmonic-to-noise ratio
    pub harmonic_noise_ratio: f32,
    /// Energy (RMS)
    pub energy: f32,
    /// Pitch confidence
    pub pitch_confidence: f32,
    /// Spectral flatness
    pub spectral_flatness: f32,
    /// Temporal features
    pub temporal_stability: f32,
}

/// Adaptive processing parameters determined by the algorithm
#[derive(Debug, Clone)]
/// Adaptive Parameters
pub struct AdaptiveParameters {
    /// Noise suppression strength (0.0 - 1.0)
    pub noise_suppression_strength: f32,
    /// AGC target level (dB)
    pub agc_target_level: f32,
    /// AGC attack time (seconds)
    pub agc_attack_time: f32,
    /// AGC release time (seconds)
    pub agc_release_time: f32,
    /// Echo cancellation filter length
    pub echo_filter_length: usize,
    /// Echo cancellation adaptation rate
    pub echo_adaptation_rate: f32,
    /// Bandwidth extension strength
    pub bandwidth_extension_strength: f32,
    /// Content type classification
    pub content_type: AudioContentType,
    /// Confidence in content classification
    pub classification_confidence: f32,
}

impl Default for AdaptiveParameters {
    fn default() -> Self {
        Self {
            noise_suppression_strength: 0.5,
            agc_target_level: -20.0,
            agc_attack_time: 0.001,
            agc_release_time: 0.1,
            echo_filter_length: 1024,
            echo_adaptation_rate: 0.01,
            bandwidth_extension_strength: 0.3,
            content_type: AudioContentType::Mixed,
            classification_confidence: 0.5,
        }
    }
}

/// Adaptive processing statistics
#[derive(Debug, Clone)]
/// Adaptive Stats
pub struct AdaptiveStats {
    /// Number of adaptations performed
    pub adaptations_count: u32,
    /// Average adaptation rate
    pub avg_adaptation_rate: f32,
    /// Content type detection accuracy
    pub detection_accuracy: f32,
    /// Processing overhead (ms)
    pub processing_overhead_ms: f64,
    /// Current parameters
    pub current_parameters: AdaptiveParameters,
}

/// Result of adaptive processing
#[derive(Debug, Clone)]
/// Adaptive Result
pub struct AdaptiveResult {
    /// Adaptive parameters
    pub parameters: AdaptiveParameters,
    /// Audio features extracted
    pub features: AudioFeatures,
    /// Processing statistics
    pub stats: AdaptiveStats,
}

/// Adaptive algorithm processor
#[derive(Debug)]
/// Adaptive Processor
pub struct AdaptiveProcessor {
    config: AdaptiveConfig,
    features_history: VecDeque<AudioFeatures>,
    parameters_history: VecDeque<AdaptiveParameters>,
    content_type_history: VecDeque<AudioContentType>,
    adaptation_count: u32,
    last_parameters: AdaptiveParameters,
    noise_estimator: NoiseEstimator,
    pitch_tracker: PitchTracker,
    spectral_analyzer: SpectralAnalyzer,
}

impl AdaptiveProcessor {
    /// Create a new adaptive processor
    pub fn new(config: AdaptiveConfig) -> Result<Self, RecognitionError> {
        let features_history = VecDeque::with_capacity(config.history_length);
        let parameters_history = VecDeque::with_capacity(config.history_length);
        let content_type_history = VecDeque::with_capacity(config.history_length);

        let noise_estimator = NoiseEstimator::new(config.sample_rate)?;
        let pitch_tracker = PitchTracker::new(config.sample_rate)?;
        let spectral_analyzer =
            SpectralAnalyzer::new(config.analysis_window_size, config.sample_rate)?;

        Ok(Self {
            config,
            features_history,
            parameters_history,
            content_type_history,
            adaptation_count: 0,
            last_parameters: AdaptiveParameters::default(),
            noise_estimator,
            pitch_tracker,
            spectral_analyzer,
        })
    }

    /// Analyze audio and adapt processing parameters
    pub fn analyze_and_adapt(
        &mut self,
        audio: &AudioBuffer,
    ) -> Result<AdaptiveResult, RecognitionError> {
        let start_time = std::time::Instant::now();

        // Extract audio features
        let features = self.extract_features(audio)?;

        // Classify content type
        let content_type = self.classify_content(&features)?;

        // Determine adaptive parameters
        let mut parameters = self.determine_parameters(&features, content_type)?;

        // Apply temporal smoothing
        parameters = self.apply_temporal_smoothing(parameters)?;

        // Update history
        self.update_history(features.clone(), parameters.clone(), content_type);

        // Calculate statistics
        let stats = self.calculate_stats(&parameters);

        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(AdaptiveResult {
            parameters,
            features,
            stats: AdaptiveStats {
                adaptations_count: self.adaptation_count,
                avg_adaptation_rate: self.calculate_avg_adaptation_rate(),
                detection_accuracy: self.calculate_detection_accuracy(),
                processing_overhead_ms: processing_time,
                current_parameters: self.last_parameters.clone(),
            },
        })
    }

    /// Extract comprehensive audio features
    fn extract_features(&mut self, audio: &AudioBuffer) -> Result<AudioFeatures, RecognitionError> {
        let samples = audio.samples();

        // Basic energy and RMS
        let energy = self.calculate_energy(samples);

        // Zero crossing rate
        let zcr = self.calculate_zero_crossing_rate(samples);

        // Noise estimation and SNR
        let noise_level = self.noise_estimator.estimate(samples)?;
        let signal_level = energy;
        let snr_db = if noise_level > 0.0 {
            20.0 * (signal_level / noise_level).log10()
        } else {
            60.0 // Very high SNR if no noise detected
        };

        // Spectral features
        let spectral_features = self.spectral_analyzer.analyze(samples)?;

        // Pitch tracking
        let pitch_features = self.pitch_tracker.track(samples)?;

        // Temporal stability
        let temporal_stability = self.calculate_temporal_stability(samples);

        Ok(AudioFeatures {
            snr_db,
            zero_crossing_rate: zcr,
            spectral_centroid: spectral_features.centroid,
            spectral_rolloff: spectral_features.rolloff,
            spectral_flux: spectral_features.flux,
            harmonic_noise_ratio: pitch_features.harmonic_noise_ratio,
            energy,
            pitch_confidence: pitch_features.confidence,
            spectral_flatness: spectral_features.flatness,
            temporal_stability,
        })
    }

    /// Classify audio content type
    fn classify_content(
        &self,
        features: &AudioFeatures,
    ) -> Result<AudioContentType, RecognitionError> {
        // Multi-feature classifier for content type detection

        // Silence detection
        if features.energy < 0.001 {
            return Ok(AudioContentType::Silence);
        }

        // Speech indicators
        let speech_score = (features.pitch_confidence * 0.3)
            + ((features.harmonic_noise_ratio.min(10.0) / 10.0) * 0.2)
            + ((features.zero_crossing_rate.min(0.3) / 0.3) * 0.2)
            + ((1.0 - features.spectral_flatness) * 0.3);

        // Music indicators
        let music_score = (features.harmonic_noise_ratio.min(20.0) / 20.0 * 0.4)
            + ((1.0 - features.spectral_flatness) * 0.3)
            + (features.temporal_stability * 0.3);

        // Noise indicators
        let noise_score = (features.spectral_flatness * 0.5)
            + ((1.0 - features.pitch_confidence) * 0.3)
            + ((1.0 - features.temporal_stability) * 0.2);

        // Classification based on scores
        let max_score = speech_score.max(music_score).max(noise_score);

        if max_score < 0.4 {
            Ok(AudioContentType::Mixed)
        } else if speech_score == max_score
            && speech_score > self.config.speech_detection_sensitivity
        {
            Ok(AudioContentType::Speech)
        } else if music_score == max_score && music_score > self.config.music_detection_sensitivity
        {
            Ok(AudioContentType::Music)
        } else if noise_score == max_score {
            Ok(AudioContentType::Noise)
        } else {
            Ok(AudioContentType::Mixed)
        }
    }

    /// Determine optimal parameters based on features and content type
    fn determine_parameters(
        &self,
        features: &AudioFeatures,
        content_type: AudioContentType,
    ) -> Result<AdaptiveParameters, RecognitionError> {
        let mut params = AdaptiveParameters::default();
        params.content_type = content_type;

        match content_type {
            AudioContentType::Speech => {
                // Optimize for speech clarity
                params.noise_suppression_strength = if features.snr_db < self.config.snr_threshold {
                    0.8 // Aggressive noise suppression for noisy speech
                } else {
                    0.3 // Light noise suppression for clean speech
                };

                params.agc_target_level = -18.0; // Optimize for speech intelligibility
                params.agc_attack_time = 0.002; // Fast attack for speech transients
                params.agc_release_time = 0.05; // Medium release
                params.echo_filter_length = 2048; // Longer filter for speech
                params.echo_adaptation_rate = 0.02; // Moderate adaptation
                params.bandwidth_extension_strength = 0.5; // Enhance speech bandwidth
                params.classification_confidence = 0.9;
            }

            AudioContentType::Music => {
                // Optimize for musical quality
                params.noise_suppression_strength = 0.2; // Gentle noise suppression
                params.agc_target_level = -23.0; // Preserve dynamics
                params.agc_attack_time = 0.005; // Slower attack to preserve transients
                params.agc_release_time = 0.2; // Longer release for music
                params.echo_filter_length = 1024; // Shorter filter for music
                params.echo_adaptation_rate = 0.005; // Slow adaptation
                params.bandwidth_extension_strength = 0.7; // Enhance full bandwidth
                params.classification_confidence = 0.8;
            }

            AudioContentType::Noise => {
                // Aggressive noise reduction
                params.noise_suppression_strength = 0.9;
                params.agc_target_level = -25.0; // Lower target level
                params.agc_attack_time = 0.001; // Fast attack
                params.agc_release_time = 0.3; // Slow release
                params.echo_filter_length = 512; // Short filter
                params.echo_adaptation_rate = 0.001; // Very slow adaptation
                params.bandwidth_extension_strength = 0.1; // Minimal enhancement
                params.classification_confidence = 0.7;
            }

            AudioContentType::Silence => {
                // Minimal processing for silence
                params.noise_suppression_strength = 0.1;
                params.agc_target_level = -30.0;
                params.agc_attack_time = 0.01;
                params.agc_release_time = 0.5;
                params.echo_filter_length = 256;
                params.echo_adaptation_rate = 0.0;
                params.bandwidth_extension_strength = 0.0;
                params.classification_confidence = 0.95;
            }

            AudioContentType::Mixed => {
                // Balanced parameters
                params.noise_suppression_strength = 0.5;
                params.agc_target_level = -20.0;
                params.agc_attack_time = 0.003;
                params.agc_release_time = 0.1;
                params.echo_filter_length = 1024;
                params.echo_adaptation_rate = 0.01;
                params.bandwidth_extension_strength = 0.4;
                params.classification_confidence = 0.5;
            }
        }

        // Fine-tune based on SNR
        if features.snr_db < 5.0 {
            params.noise_suppression_strength = (params.noise_suppression_strength + 0.3).min(1.0);
        } else if features.snr_db > 20.0 {
            params.noise_suppression_strength = (params.noise_suppression_strength - 0.2).max(0.0);
        }

        Ok(params)
    }

    /// Apply temporal smoothing to prevent parameter oscillation
    fn apply_temporal_smoothing(
        &mut self,
        mut params: AdaptiveParameters,
    ) -> Result<AdaptiveParameters, RecognitionError> {
        if self.parameters_history.is_empty() {
            self.last_parameters = params.clone();
            return Ok(params);
        }

        let alpha = self.config.adaptation_rate;

        // Smooth critical parameters
        params.noise_suppression_strength = alpha * params.noise_suppression_strength
            + (1.0 - alpha) * self.last_parameters.noise_suppression_strength;

        params.agc_target_level =
            alpha * params.agc_target_level + (1.0 - alpha) * self.last_parameters.agc_target_level;

        params.bandwidth_extension_strength = alpha * params.bandwidth_extension_strength
            + (1.0 - alpha) * self.last_parameters.bandwidth_extension_strength;

        self.last_parameters = params.clone();
        self.adaptation_count += 1;

        Ok(params)
    }

    /// Update processing history
    fn update_history(
        &mut self,
        features: AudioFeatures,
        parameters: AdaptiveParameters,
        content_type: AudioContentType,
    ) {
        if self.features_history.len() >= self.config.history_length {
            self.features_history.pop_front();
        }
        if self.parameters_history.len() >= self.config.history_length {
            self.parameters_history.pop_front();
        }
        if self.content_type_history.len() >= self.config.history_length {
            self.content_type_history.pop_front();
        }

        self.features_history.push_back(features);
        self.parameters_history.push_back(parameters);
        self.content_type_history.push_back(content_type);
    }

    /// Calculate processing statistics
    fn calculate_stats(&self, parameters: &AdaptiveParameters) -> AdaptiveStats {
        AdaptiveStats {
            adaptations_count: self.adaptation_count,
            avg_adaptation_rate: self.calculate_avg_adaptation_rate(),
            detection_accuracy: self.calculate_detection_accuracy(),
            processing_overhead_ms: 0.0, // Will be filled by caller
            current_parameters: parameters.clone(),
        }
    }

    /// Calculate average adaptation rate
    fn calculate_avg_adaptation_rate(&self) -> f32 {
        if self.parameters_history.len() < 2 {
            return 0.0;
        }

        let mut total_change = 0.0;
        for i in 1..self.parameters_history.len() {
            let prev = &self.parameters_history[i - 1];
            let curr = &self.parameters_history[i];

            let change = (curr.noise_suppression_strength - prev.noise_suppression_strength).abs() +
                        (curr.agc_target_level - prev.agc_target_level).abs() / 10.0 + // Normalize dB scale
                        (curr.bandwidth_extension_strength - prev.bandwidth_extension_strength).abs();

            total_change += change;
        }

        total_change / (self.parameters_history.len() - 1) as f32
    }

    /// Calculate content detection accuracy estimate
    fn calculate_detection_accuracy(&self) -> f32 {
        if self.content_type_history.len() < 3 {
            return 0.5;
        }

        // Estimate accuracy based on consistency of classification
        let mut consistent_count = 0;
        let window_size = 3;

        for i in window_size..self.content_type_history.len() {
            let recent: Vec<AudioContentType> = self
                .content_type_history
                .range((i - window_size)..i)
                .cloned()
                .collect();
            let most_common = Self::most_common_content_type(&recent);

            if recent.iter().all(|&ct| ct == most_common) {
                consistent_count += 1;
            }
        }

        consistent_count as f32 / (self.content_type_history.len() - window_size) as f32
    }

    /// Find most common content type in slice
    fn most_common_content_type(types: &[AudioContentType]) -> AudioContentType {
        let mut counts = [0; 5]; // For 5 content types

        for &content_type in types {
            let index = match content_type {
                AudioContentType::Speech => 0,
                AudioContentType::Music => 1,
                AudioContentType::Noise => 2,
                AudioContentType::Mixed => 3,
                AudioContentType::Silence => 4,
            };
            counts[index] += 1;
        }

        let max_index = counts
            .iter()
            .position(|&x| x == *counts.iter().max().unwrap())
            .unwrap();
        match max_index {
            0 => AudioContentType::Speech,
            1 => AudioContentType::Music,
            2 => AudioContentType::Noise,
            3 => AudioContentType::Mixed,
            4 => AudioContentType::Silence,
            _ => AudioContentType::Mixed,
        }
    }

    /// Calculate basic audio energy
    fn calculate_energy(&self, samples: &[f32]) -> f32 {
        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Calculate zero crossing rate
    fn calculate_zero_crossing_rate(&self, samples: &[f32]) -> f32 {
        let mut crossings = 0;

        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / (samples.len() - 1) as f32
    }

    /// Calculate temporal stability
    fn calculate_temporal_stability(&self, samples: &[f32]) -> f32 {
        if samples.len() < 100 {
            return 0.5;
        }

        let chunk_size = samples.len() / 10;
        let mut chunk_energies = Vec::new();

        for i in 0..10 {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(samples.len());
            let chunk = &samples[start..end];
            let energy = self.calculate_energy(chunk);
            chunk_energies.push(energy);
        }

        // Calculate coefficient of variation
        let mean = chunk_energies.iter().sum::<f32>() / chunk_energies.len() as f32;
        let variance = chunk_energies
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / chunk_energies.len() as f32;
        let std_dev = variance.sqrt();

        if mean > 0.0 {
            1.0 - (std_dev / mean).min(1.0) // Higher stability = lower coefficient of variation
        } else {
            0.0
        }
    }

    /// Reset processor state
    pub fn reset(&mut self) -> Result<(), RecognitionError> {
        self.features_history.clear();
        self.parameters_history.clear();
        self.content_type_history.clear();
        self.adaptation_count = 0;
        self.last_parameters = AdaptiveParameters::default();
        self.noise_estimator.reset()?;
        self.pitch_tracker.reset()?;
        self.spectral_analyzer.reset()?;
        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> &AdaptiveConfig {
        &self.config
    }

    /// Get current parameters
    pub fn current_parameters(&self) -> &AdaptiveParameters {
        &self.last_parameters
    }
}

/// Noise level estimator
#[derive(Debug)]
struct NoiseEstimator {
    sample_rate: u32,
    noise_floor: f32,
    adaptation_rate: f32,
}

impl NoiseEstimator {
    fn new(sample_rate: u32) -> Result<Self, RecognitionError> {
        Ok(Self {
            sample_rate,
            noise_floor: 0.001,
            adaptation_rate: 0.01,
        })
    }

    fn estimate(&mut self, samples: &[f32]) -> Result<f32, RecognitionError> {
        let energy = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();

        // Simple noise floor estimation using minimum statistics
        if energy < self.noise_floor * 2.0 {
            self.noise_floor =
                self.adaptation_rate * energy + (1.0 - self.adaptation_rate) * self.noise_floor;
        }

        Ok(self.noise_floor)
    }

    fn reset(&mut self) -> Result<(), RecognitionError> {
        self.noise_floor = 0.001;
        Ok(())
    }
}

/// Simple pitch tracker
#[derive(Debug)]
struct PitchTracker {
    sample_rate: u32,
    min_pitch: f32,
    max_pitch: f32,
}

impl PitchTracker {
    fn new(sample_rate: u32) -> Result<Self, RecognitionError> {
        Ok(Self {
            sample_rate,
            min_pitch: 50.0,  // 50 Hz
            max_pitch: 800.0, // 800 Hz
        })
    }

    fn track(&self, samples: &[f32]) -> Result<PitchFeatures, RecognitionError> {
        // Simplified autocorrelation-based pitch detection
        let min_period = (self.sample_rate as f32 / self.max_pitch) as usize;
        let max_period = (self.sample_rate as f32 / self.min_pitch) as usize;

        let mut max_correlation = 0.0;
        let mut best_period = min_period;

        for period in min_period..=max_period.min(samples.len() / 2) {
            let mut correlation = 0.0;
            let mut norm1 = 0.0;
            let mut norm2 = 0.0;

            for i in 0..(samples.len() - period) {
                correlation += samples[i] * samples[i + period];
                norm1 += samples[i] * samples[i];
                norm2 += samples[i + period] * samples[i + period];
            }

            if norm1 > 0.0 && norm2 > 0.0 {
                correlation /= (norm1 * norm2).sqrt();
                if correlation > max_correlation {
                    max_correlation = correlation;
                    best_period = period;
                }
            }
        }

        let pitch_confidence = max_correlation.max(0.0);
        let fundamental_freq = self.sample_rate as f32 / best_period as f32;

        // Simple harmonic-to-noise ratio estimation
        let harmonic_noise_ratio = if pitch_confidence > 0.3 {
            pitch_confidence * 10.0 // Convert to rough HNR in dB
        } else {
            0.0
        };

        Ok(PitchFeatures {
            confidence: pitch_confidence,
            harmonic_noise_ratio,
        })
    }

    fn reset(&mut self) -> Result<(), RecognitionError> {
        // No state to reset in this simple implementation
        Ok(())
    }
}

/// Pitch tracking features
#[derive(Debug)]
struct PitchFeatures {
    confidence: f32,
    harmonic_noise_ratio: f32,
}

/// Spectral analyzer
#[derive(Debug)]
struct SpectralAnalyzer {
    fft_size: usize,
    sample_rate: u32,
    prev_spectrum: Vec<f32>,
}

impl SpectralAnalyzer {
    fn new(fft_size: usize, sample_rate: u32) -> Result<Self, RecognitionError> {
        Ok(Self {
            fft_size,
            sample_rate,
            prev_spectrum: Vec::new(),
        })
    }

    fn analyze(&mut self, samples: &[f32]) -> Result<SpectralFeatures, RecognitionError> {
        // Simplified spectral analysis
        let window_size = self.fft_size.min(samples.len());
        let windowed: Vec<f32> = samples[..window_size]
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let window_val = 0.5
                    * (1.0
                        - (2.0 * std::f32::consts::PI * i as f32 / (window_size - 1) as f32).cos());
                x * window_val
            })
            .collect();

        // Simple magnitude spectrum computation (placeholder for proper FFT)
        let mut spectrum = vec![0.0; window_size / 2];
        for i in 0..spectrum.len() {
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for n in 0..window_size {
                let angle = -2.0 * std::f32::consts::PI * i as f32 * n as f32 / window_size as f32;
                real_sum += windowed[n] * angle.cos();
                imag_sum += windowed[n] * angle.sin();
            }

            spectrum[i] = (real_sum * real_sum + imag_sum * imag_sum).sqrt();
        }

        // Calculate spectral features
        let centroid = self.calculate_spectral_centroid(&spectrum);
        let rolloff = self.calculate_spectral_rolloff(&spectrum);
        let flatness = self.calculate_spectral_flatness(&spectrum);
        let flux = self.calculate_spectral_flux(&spectrum);

        self.prev_spectrum = spectrum;

        Ok(SpectralFeatures {
            centroid,
            rolloff,
            flatness,
            flux,
        })
    }

    fn calculate_spectral_centroid(&self, spectrum: &[f32]) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total_magnitude = 0.0;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let freq = i as f32 * self.sample_rate as f32 / (2.0 * spectrum.len() as f32);
            weighted_sum += freq * magnitude;
            total_magnitude += magnitude;
        }

        if total_magnitude > 0.0 {
            weighted_sum / total_magnitude
        } else {
            0.0
        }
    }

    fn calculate_spectral_rolloff(&self, spectrum: &[f32]) -> f32 {
        let total_energy: f32 = spectrum.iter().map(|&x| x * x).sum();
        let threshold = 0.85 * total_energy;

        let mut cumulative_energy = 0.0;
        for (i, &magnitude) in spectrum.iter().enumerate() {
            cumulative_energy += magnitude * magnitude;
            if cumulative_energy >= threshold {
                return i as f32 * self.sample_rate as f32 / (2.0 * spectrum.len() as f32);
            }
        }

        self.sample_rate as f32 / 2.0
    }

    fn calculate_spectral_flatness(&self, spectrum: &[f32]) -> f32 {
        let geometric_mean = spectrum
            .iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| x.ln())
            .sum::<f32>()
            / spectrum.len() as f32;

        let arithmetic_mean = spectrum.iter().sum::<f32>() / spectrum.len() as f32;

        if arithmetic_mean > 0.0 {
            geometric_mean.exp() / arithmetic_mean
        } else {
            0.0
        }
    }

    fn calculate_spectral_flux(&self, spectrum: &[f32]) -> f32 {
        if self.prev_spectrum.is_empty() || self.prev_spectrum.len() != spectrum.len() {
            return 0.0;
        }

        let mut flux = 0.0;
        for (curr, prev) in spectrum.iter().zip(self.prev_spectrum.iter()) {
            let diff = curr - prev;
            if diff > 0.0 {
                flux += diff * diff;
            }
        }

        flux.sqrt()
    }

    fn reset(&mut self) -> Result<(), RecognitionError> {
        self.prev_spectrum.clear();
        Ok(())
    }
}

/// Spectral features
#[derive(Debug)]
struct SpectralFeatures {
    centroid: f32,
    rolloff: f32,
    flatness: f32,
    flux: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_processor_creation() {
        let config = AdaptiveConfig::default();
        let processor = AdaptiveProcessor::new(config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_content_classification() {
        let config = AdaptiveConfig::default();
        let processor = AdaptiveProcessor::new(config).unwrap();

        // Test silence classification
        let silent_features = AudioFeatures {
            snr_db: 40.0,
            zero_crossing_rate: 0.1,
            spectral_centroid: 1000.0,
            spectral_rolloff: 2000.0,
            spectral_flux: 0.1,
            harmonic_noise_ratio: 5.0,
            energy: 0.0001, // Very low energy
            pitch_confidence: 0.2,
            spectral_flatness: 0.8,
            temporal_stability: 0.9,
        };

        let content_type = processor.classify_content(&silent_features).unwrap();
        assert_eq!(content_type, AudioContentType::Silence);
    }

    #[test]
    fn test_parameter_adaptation() {
        let config = AdaptiveConfig::default();
        let mut processor = AdaptiveProcessor::new(config).unwrap();

        let samples = vec![0.1; 4096];
        let audio = AudioBuffer::mono(samples, 16000);

        let result = processor.analyze_and_adapt(&audio);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.stats.processing_overhead_ms > 0.0);
        assert!(result.parameters.classification_confidence >= 0.0);
        assert!(result.parameters.classification_confidence <= 1.0);
    }

    #[test]
    fn test_noise_estimator() {
        let mut estimator = NoiseEstimator::new(16000).unwrap();

        let noisy_samples = vec![0.01; 1000];
        let noise_level = estimator.estimate(&noisy_samples).unwrap();
        assert!(noise_level > 0.0);
        assert!(noise_level < 1.0);
    }

    #[test]
    fn test_pitch_tracker() {
        let tracker = PitchTracker::new(16000).unwrap();

        // Generate a simple sine wave at 440 Hz
        let mut samples = Vec::new();
        for i in 0..1000 {
            let t = i as f32 / 16000.0;
            samples.push((2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5);
        }

        let features = tracker.track(&samples).unwrap();
        assert!(features.confidence > 0.0);
        assert!(features.harmonic_noise_ratio >= 0.0);
    }

    #[test]
    fn test_spectral_analyzer() {
        let mut analyzer = SpectralAnalyzer::new(2048, 16000).unwrap();

        let samples = vec![0.1; 2048];
        let features = analyzer.analyze(&samples).unwrap();

        assert!(features.centroid >= 0.0);
        assert!(features.rolloff >= 0.0);
        assert!(features.flatness >= 0.0);
        assert!(features.flux >= 0.0);
    }
}
