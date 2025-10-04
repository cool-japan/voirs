//! Audio processing and feature extraction
//!
//! This module provides comprehensive audio processing capabilities including
//! voice activity detection, feature extraction, and audio quality analysis.

use super::types::*;
use crate::FeedbackError;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Voice Activity Detection (VAD) system
pub struct VoiceActivityDetector {
    config: VadConfig,
    energy_buffer: VecDeque<f32>,
    zcr_buffer: VecDeque<f32>,
    spectral_buffer: VecDeque<f32>,
    frame_count: usize,
    noise_floor: f32,
    adaptive_threshold: f32,
}

impl VoiceActivityDetector {
    /// Create a new voice activity detector
    pub fn new(config: VadConfig) -> Self {
        Self {
            energy_buffer: VecDeque::with_capacity(config.history_frames),
            zcr_buffer: VecDeque::with_capacity(config.history_frames),
            spectral_buffer: VecDeque::with_capacity(config.history_frames),
            frame_count: 0,
            noise_floor: config.initial_noise_floor,
            adaptive_threshold: config.energy_threshold,
            config,
        }
    }

    /// Process audio frame and detect voice activity
    pub fn process_frame(&mut self, audio_frame: &[f32]) -> VadResult {
        self.frame_count += 1;

        // Calculate frame features
        let energy = self.calculate_energy(audio_frame);
        let zcr = self.calculate_zero_crossing_rate(audio_frame);
        let spectral_centroid = self.calculate_spectral_centroid(audio_frame);

        // Update buffers
        Self::update_buffer(&mut self.energy_buffer, energy);
        Self::update_buffer(&mut self.zcr_buffer, zcr);
        Self::update_buffer(&mut self.spectral_buffer, spectral_centroid);

        // Update adaptive parameters
        self.update_adaptive_parameters();

        // Apply VAD algorithms
        let energy_vad = self.energy_based_vad(energy);
        let zcr_vad = self.zcr_based_vad(zcr);
        let spectral_vad = self.spectral_based_vad(spectral_centroid);
        let ml_vad = self.ml_based_vad(energy, zcr, spectral_centroid);

        // Combine results using majority voting
        let voice_detected = self.combine_vad_results(energy_vad, zcr_vad, spectral_vad, ml_vad);

        // Calculate confidence
        let confidence = self.calculate_confidence(energy_vad, zcr_vad, spectral_vad, ml_vad);

        VadResult {
            voice_detected,
            confidence,
            energy,
            zero_crossing_rate: zcr,
            spectral_centroid,
            frame_number: self.frame_count,
            energy_vad,
            zcr_vad,
            spectral_vad,
            ml_vad,
        }
    }

    /// Calculate RMS energy of audio frame
    fn calculate_energy(&self, frame: &[f32]) -> f32 {
        if frame.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = frame.iter().map(|&x| x * x).sum();
        (sum_squares / frame.len() as f32).sqrt()
    }

    /// Calculate zero crossing rate
    fn calculate_zero_crossing_rate(&self, frame: &[f32]) -> f32 {
        if frame.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 1..frame.len() {
            if (frame[i] >= 0.0 && frame[i - 1] < 0.0) || (frame[i] < 0.0 && frame[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / (frame.len() - 1) as f32
    }

    /// Calculate spectral centroid (simplified)
    fn calculate_spectral_centroid(&self, frame: &[f32]) -> f32 {
        if frame.is_empty() {
            return 0.0;
        }

        // Simple spectral centroid approximation using high-frequency energy
        let mid_point = frame.len() / 2;
        let low_energy: f32 = frame[0..mid_point].iter().map(|&x| x.abs()).sum();
        let high_energy: f32 = frame[mid_point..].iter().map(|&x| x.abs()).sum();

        let total_energy = low_energy + high_energy;
        if total_energy > 0.0 {
            high_energy / total_energy
        } else {
            0.0
        }
    }

    /// Update circular buffer
    fn update_buffer(buffer: &mut VecDeque<f32>, value: f32) {
        if buffer.len() >= buffer.capacity() {
            buffer.pop_front();
        }
        buffer.push_back(value);
    }

    /// Update adaptive parameters based on recent history
    fn update_adaptive_parameters(&mut self) {
        if self.energy_buffer.len() < 10 {
            return; // Need sufficient history
        }

        // Update noise floor as minimum energy in recent history
        let min_energy = self
            .energy_buffer
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        self.noise_floor = self.noise_floor * 0.95 + min_energy * 0.05;

        // Adaptive threshold based on noise floor
        self.adaptive_threshold = self.noise_floor * self.config.energy_threshold_multiplier;
    }

    /// Energy-based VAD
    fn energy_based_vad(&self, energy: f32) -> bool {
        energy > self.adaptive_threshold
    }

    /// Zero crossing rate based VAD
    fn zcr_based_vad(&self, zcr: f32) -> bool {
        zcr > self.config.zcr_threshold && zcr < self.config.max_zcr_threshold
    }

    /// Spectral-based VAD
    fn spectral_based_vad(&self, spectral_centroid: f32) -> bool {
        spectral_centroid > self.config.spectral_threshold
            && spectral_centroid < self.config.max_spectral_threshold
    }

    /// Machine learning-based VAD (simple feature-based classifier)
    fn ml_based_vad(&self, energy: f32, zcr: f32, spectral_centroid: f32) -> bool {
        // Simple linear classifier weights (could be trained on data)
        let weights = [2.0, 1.5, 1.2]; // [energy, zcr, spectral]
        let features = [(energy - self.noise_floor).max(0.0), zcr, spectral_centroid];

        let score: f32 = weights
            .iter()
            .zip(features.iter())
            .map(|(w, f)| w * f)
            .sum();

        score > self.config.ml_threshold
    }

    /// Combine VAD results using majority voting with weights
    fn combine_vad_results(&self, energy: bool, zcr: bool, spectral: bool, ml: bool) -> bool {
        let weights = self.config.algorithm_weights;
        let weighted_score = (energy as u8 as f32) * weights[0]
            + (zcr as u8 as f32) * weights[1]
            + (spectral as u8 as f32) * weights[2]
            + (ml as u8 as f32) * weights[3];

        let total_weight: f32 = weights.iter().sum();
        weighted_score / total_weight > self.config.decision_threshold
    }

    /// Calculate confidence score based on algorithm agreement
    fn calculate_confidence(&self, energy: bool, zcr: bool, spectral: bool, ml: bool) -> f32 {
        let results = [energy, zcr, spectral, ml];
        let agreement_count = results.iter().filter(|&&x| x).count();
        let total_count = results.len();

        // Base confidence on agreement level
        let base_confidence = agreement_count as f32 / total_count as f32;

        // Adjust confidence based on recent stability
        let stability_factor = self.calculate_stability_factor();

        (base_confidence * 0.7 + stability_factor * 0.3).min(1.0)
    }

    /// Calculate stability factor based on recent decisions
    fn calculate_stability_factor(&self) -> f32 {
        if self.energy_buffer.len() < 5 {
            return 0.5; // Default stability
        }

        // Simple stability measure: low variance in recent energy
        let recent_energies: Vec<f32> = self.energy_buffer.iter().rev().take(5).cloned().collect();
        let mean: f32 = recent_energies.iter().sum::<f32>() / recent_energies.len() as f32;
        let variance: f32 = recent_energies
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / recent_energies.len() as f32;

        let stability = 1.0 / (1.0 + variance);
        stability.min(1.0)
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.energy_buffer.clear();
        self.zcr_buffer.clear();
        self.spectral_buffer.clear();
        self.frame_count = 0;
        self.noise_floor = self.config.initial_noise_floor;
        self.adaptive_threshold = self.config.energy_threshold;
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> VadStatistics {
        VadStatistics {
            frames_processed: self.frame_count,
            current_noise_floor: self.noise_floor,
            adaptive_threshold: self.adaptive_threshold,
            buffer_size: self.energy_buffer.len(),
            average_energy: if self.energy_buffer.is_empty() {
                0.0
            } else {
                self.energy_buffer.iter().sum::<f32>() / self.energy_buffer.len() as f32
            },
            average_zcr: if self.zcr_buffer.is_empty() {
                0.0
            } else {
                self.zcr_buffer.iter().sum::<f32>() / self.zcr_buffer.len() as f32
            },
        }
    }
}

/// VAD configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadConfig {
    /// Energy threshold for voice detection
    pub energy_threshold: f32,
    /// Energy threshold multiplier for adaptive threshold
    pub energy_threshold_multiplier: f32,
    /// Zero crossing rate threshold
    pub zcr_threshold: f32,
    /// Maximum zero crossing rate threshold (to filter noise)
    pub max_zcr_threshold: f32,
    /// Spectral centroid threshold
    pub spectral_threshold: f32,
    /// Maximum spectral centroid threshold
    pub max_spectral_threshold: f32,
    /// Machine learning classifier threshold
    pub ml_threshold: f32,
    /// Decision threshold for combining algorithms
    pub decision_threshold: f32,
    /// Algorithm weights [energy, zcr, spectral, ml]
    pub algorithm_weights: [f32; 4],
    /// Number of frames to keep in history
    pub history_frames: usize,
    /// Initial noise floor estimate
    pub initial_noise_floor: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            energy_threshold: 0.01,
            energy_threshold_multiplier: 3.0,
            zcr_threshold: 0.1,
            max_zcr_threshold: 0.8,
            spectral_threshold: 0.3,
            max_spectral_threshold: 0.9,
            ml_threshold: 1.0,
            decision_threshold: 0.5,
            algorithm_weights: [2.0, 1.0, 1.0, 1.5], // Favor energy and ML
            history_frames: 50,
            initial_noise_floor: 0.001,
        }
    }
}

/// VAD result for a single frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadResult {
    /// Whether voice was detected
    pub voice_detected: bool,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Frame energy
    pub energy: f32,
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    /// Spectral centroid
    pub spectral_centroid: f32,
    /// Frame number
    pub frame_number: usize,
    /// Individual algorithm results
    pub energy_vad: bool,
    /// Description
    pub zcr_vad: bool,
    /// Description
    pub spectral_vad: bool,
    /// Description
    pub ml_vad: bool,
}

/// VAD statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadStatistics {
    /// Total frames processed
    pub frames_processed: usize,
    /// Current noise floor estimate
    pub current_noise_floor: f32,
    /// Current adaptive threshold
    pub adaptive_threshold: f32,
    /// Current buffer size
    pub buffer_size: usize,
    /// Average energy over recent frames
    pub average_energy: f32,
    /// Average zero crossing rate over recent frames
    pub average_zcr: f32,
}

/// Advanced audio processor with VAD and feature extraction
pub struct AudioProcessor {
    vad: VoiceActivityDetector,
    feature_extractor: FeatureExtractor,
    config: AudioProcessorConfig,
}

impl AudioProcessor {
    /// Create a new audio processor
    pub fn new(config: AudioProcessorConfig) -> Self {
        Self {
            vad: VoiceActivityDetector::new(config.vad_config.clone()),
            feature_extractor: FeatureExtractor::new(config.feature_config.clone()),
            config,
        }
    }

    /// Process audio chunk with VAD and feature extraction
    pub fn process_chunk(
        &mut self,
        audio_data: &[f32],
    ) -> Result<AudioProcessingResult, FeedbackError> {
        let start_time = std::time::Instant::now();
        let frame_size = self.config.frame_size;
        let mut results = Vec::new();

        // Process audio in frames
        for chunk in audio_data.chunks(frame_size) {
            // Pad frame if necessary
            let mut frame = chunk.to_vec();
            if frame.len() < frame_size {
                frame.resize(frame_size, 0.0);
            }

            // Voice activity detection
            let vad_result = self.vad.process_frame(&frame);

            // Feature extraction (only if voice detected or forced)
            let features = if vad_result.voice_detected || self.config.always_extract_features {
                Some(self.feature_extractor.extract_features(&frame)?)
            } else {
                None
            };

            results.push(FrameProcessingResult {
                vad_result,
                features,
                frame_index: results.len(),
            });
        }

        let voice_activity_ratio = self.calculate_voice_activity_ratio(&results);
        let quality_score = self.calculate_quality_score(&results);

        // Calculate actual processing time
        let processing_time_ms = (start_time.elapsed().as_secs_f64() * 1000.0) as f32;

        Ok(AudioProcessingResult {
            frames: results,
            processing_time_ms,
            voice_activity_ratio,
            quality_score,
        })
    }

    /// Calculate voice activity ratio
    fn calculate_voice_activity_ratio(&self, results: &[FrameProcessingResult]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }

        let voice_frames = results
            .iter()
            .filter(|r| r.vad_result.voice_detected)
            .count();

        voice_frames as f32 / results.len() as f32
    }

    /// Calculate overall quality score
    fn calculate_quality_score(&self, results: &[FrameProcessingResult]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }

        let avg_confidence: f32 =
            results.iter().map(|r| r.vad_result.confidence).sum::<f32>() / results.len() as f32;

        let avg_energy: f32 =
            results.iter().map(|r| r.vad_result.energy).sum::<f32>() / results.len() as f32;

        // Combine confidence and energy for quality score
        (avg_confidence * 0.7 + avg_energy.min(1.0) * 0.3).min(1.0)
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.vad.reset();
        self.feature_extractor.reset();
    }

    /// Get VAD statistics
    pub fn get_vad_statistics(&self) -> VadStatistics {
        self.vad.get_statistics()
    }
}

/// Feature extractor for audio analysis
pub struct FeatureExtractor {
    config: FeatureExtractionConfig,
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new(config: FeatureExtractionConfig) -> Self {
        Self { config }
    }

    /// Extract features from audio frame
    pub fn extract_features(&self, frame: &[f32]) -> Result<AudioFeatures, FeedbackError> {
        Ok(AudioFeatures {
            energy: self.calculate_energy(frame),
            zero_crossing_rate: self.calculate_zcr(frame),
            spectral_centroid: self.calculate_spectral_centroid(frame),
            spectral_rolloff: self.calculate_spectral_rolloff(frame),
            mfcc: if self.config.extract_mfcc {
                Some(self.calculate_mfcc(frame)?)
            } else {
                None
            },
            pitch: if self.config.extract_pitch {
                Some(self.estimate_pitch(frame))
            } else {
                None
            },
        })
    }

    /// Calculate RMS energy
    fn calculate_energy(&self, frame: &[f32]) -> f32 {
        if frame.is_empty() {
            return 0.0;
        }
        let sum_squares: f32 = frame.iter().map(|&x| x * x).sum();
        (sum_squares / frame.len() as f32).sqrt()
    }

    /// Calculate zero crossing rate
    fn calculate_zcr(&self, frame: &[f32]) -> f32 {
        if frame.len() < 2 {
            return 0.0;
        }
        let mut crossings = 0;
        for i in 1..frame.len() {
            if (frame[i] >= 0.0 && frame[i - 1] < 0.0) || (frame[i] < 0.0 && frame[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        crossings as f32 / (frame.len() - 1) as f32
    }

    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, frame: &[f32]) -> f32 {
        // Simplified spectral centroid calculation
        let mid_point = frame.len() / 2;
        let low_energy: f32 = frame[0..mid_point].iter().map(|&x| x.abs()).sum();
        let high_energy: f32 = frame[mid_point..].iter().map(|&x| x.abs()).sum();

        let total_energy = low_energy + high_energy;
        if total_energy > 0.0 {
            high_energy / total_energy
        } else {
            0.0
        }
    }

    /// Calculate spectral rolloff
    fn calculate_spectral_rolloff(&self, frame: &[f32]) -> f32 {
        // Simplified spectral rolloff at 85%
        let total_energy: f32 = frame.iter().map(|&x| x.abs()).sum();
        let threshold = total_energy * 0.85;

        let mut cumulative_energy = 0.0;
        for (i, &sample) in frame.iter().enumerate() {
            cumulative_energy += sample.abs();
            if cumulative_energy >= threshold {
                return i as f32 / frame.len() as f32;
            }
        }
        1.0
    }

    /// Calculate MFCC features (Mel-frequency cepstral coefficients)
    fn calculate_mfcc(&self, frame: &[f32]) -> Result<Vec<f32>, FeedbackError> {
        if frame.is_empty() {
            return Ok(vec![0.0; self.config.mfcc_coefficients]);
        }

        // Step 1: Apply windowing (Hamming window)
        let windowed_frame = self.apply_hamming_window(frame);

        // Step 2: Compute FFT magnitude spectrum
        let spectrum = self.compute_fft_magnitude(&windowed_frame)?;

        // Step 3: Apply mel filter bank
        let mel_energies = self.apply_mel_filter_bank(&spectrum)?;

        // Step 4: Take logarithm
        let log_mel_energies: Vec<f32> = mel_energies
            .iter()
            .map(|&energy| (energy.max(1e-10)).ln())
            .collect();

        // Step 5: Apply DCT (Discrete Cosine Transform)
        let mfcc_coefficients = self.apply_dct(&log_mel_energies)?;

        // Return the requested number of coefficients
        Ok(mfcc_coefficients
            .into_iter()
            .take(self.config.mfcc_coefficients)
            .collect())
    }

    /// Apply Hamming window to the frame
    fn apply_hamming_window(&self, frame: &[f32]) -> Vec<f32> {
        let n = frame.len();
        frame
            .iter()
            .enumerate()
            .map(|(i, &sample)| {
                let window_value =
                    0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos();
                sample * window_value
            })
            .collect()
    }

    /// Compute FFT magnitude spectrum (simplified implementation)
    fn compute_fft_magnitude(&self, frame: &[f32]) -> Result<Vec<f32>, FeedbackError> {
        let n = frame.len();
        let mut spectrum = vec![0.0; n / 2];

        // Simplified FFT magnitude calculation
        // In a real implementation, this would use an FFT library like rustfft
        for k in 0..n / 2 {
            let mut real = 0.0;
            let mut imag = 0.0;

            for (i, &sample) in frame.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * i as f32 / n as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            spectrum[k] = (real * real + imag * imag).sqrt();
        }

        Ok(spectrum)
    }

    /// Apply mel filter bank to the spectrum
    fn apply_mel_filter_bank(&self, spectrum: &[f32]) -> Result<Vec<f32>, FeedbackError> {
        let num_filters = 26; // Standard number of mel filters
        let sample_rate = 16000.0; // Default sample rate for speech processing
        let nyquist = sample_rate / 2.0;

        // Convert frequency to mel scale
        let freq_to_mel = |freq: f32| 2595.0 * (1.0 + freq / 700.0).log10();
        let mel_to_freq = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

        // Create mel filter bank
        let low_freq_mel = freq_to_mel(0.0);
        let high_freq_mel = freq_to_mel(nyquist);
        let mel_points: Vec<f32> = (0..=num_filters + 1)
            .map(|i| {
                low_freq_mel + (high_freq_mel - low_freq_mel) * i as f32 / (num_filters + 1) as f32
            })
            .collect();

        let freq_points: Vec<f32> = mel_points.iter().map(|&mel| mel_to_freq(mel)).collect();

        // Convert frequency points to FFT bin indices
        let bin_points: Vec<usize> = freq_points
            .iter()
            .map(|&freq| ((freq * spectrum.len() as f32 * 2.0) / sample_rate).floor() as usize)
            .collect();

        // Apply triangular filters
        let mut mel_energies = vec![0.0; num_filters];

        for m in 1..=num_filters {
            let left = bin_points[m - 1];
            let center = bin_points[m];
            let right = bin_points[m + 1];

            for k in left..=right {
                if k < spectrum.len() {
                    let weight = if k <= center {
                        (k - left) as f32 / (center - left) as f32
                    } else {
                        (right - k) as f32 / (right - center) as f32
                    };

                    mel_energies[m - 1] += spectrum[k] * weight;
                }
            }
        }

        Ok(mel_energies)
    }

    /// Apply Discrete Cosine Transform (DCT)
    fn apply_dct(&self, log_mel_energies: &[f32]) -> Result<Vec<f32>, FeedbackError> {
        let n = log_mel_energies.len();
        let mut dct_coefficients = vec![0.0; self.config.mfcc_coefficients];

        for k in 0..self.config.mfcc_coefficients {
            let mut sum = 0.0;
            for i in 0..n {
                let cos_term = (std::f32::consts::PI * k as f32 * (i as f32 + 0.5)) / n as f32;
                sum += log_mel_energies[i] * cos_term.cos();
            }

            let normalization = if k == 0 {
                (1.0 / n as f32).sqrt()
            } else {
                (2.0 / n as f32).sqrt()
            };

            dct_coefficients[k] = sum * normalization;
        }

        Ok(dct_coefficients)
    }

    /// Estimate pitch using autocorrelation
    fn estimate_pitch(&self, frame: &[f32]) -> f32 {
        if frame.len() < 2 {
            return 0.0;
        }

        // Simple autocorrelation-based pitch estimation
        let mut max_correlation = 0.0;
        let mut best_lag = 0;

        let min_lag = 20; // Minimum lag for pitch detection
        let max_lag = frame.len() / 4; // Maximum lag

        for lag in min_lag..max_lag {
            let mut correlation = 0.0;
            let mut norm1 = 0.0;
            let mut norm2 = 0.0;

            for i in 0..(frame.len() - lag) {
                correlation += frame[i] * frame[i + lag];
                norm1 += frame[i] * frame[i];
                norm2 += frame[i + lag] * frame[i + lag];
            }

            if norm1 > 0.0 && norm2 > 0.0 {
                correlation /= (norm1 * norm2).sqrt();
                if correlation > max_correlation {
                    max_correlation = correlation;
                    best_lag = lag;
                }
            }
        }

        if max_correlation > 0.3 && best_lag > 0 {
            // Convert lag to frequency (assuming 16kHz sample rate)
            16000.0 / best_lag as f32
        } else {
            0.0 // No clear pitch detected
        }
    }

    /// Reset feature extractor
    pub fn reset(&mut self) {
        // Reset any internal state if needed
    }
}

/// Audio processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioProcessorConfig {
    /// VAD configuration
    pub vad_config: VadConfig,
    /// Feature extraction configuration
    pub feature_config: FeatureExtractionConfig,
    /// Frame size for processing
    pub frame_size: usize,
    /// Whether to always extract features (even without voice)
    pub always_extract_features: bool,
}

impl Default for AudioProcessorConfig {
    fn default() -> Self {
        Self {
            vad_config: VadConfig::default(),
            feature_config: FeatureExtractionConfig::default(),
            frame_size: 512,
            always_extract_features: false,
        }
    }
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionConfig {
    /// Whether to extract MFCC features
    pub extract_mfcc: bool,
    /// Number of MFCC coefficients
    pub mfcc_coefficients: usize,
    /// Whether to extract pitch
    pub extract_pitch: bool,
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            extract_mfcc: true,
            mfcc_coefficients: 13,
            extract_pitch: true,
        }
    }
}

/// Audio features extracted from a frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeatures {
    /// RMS energy
    pub energy: f32,
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    /// Spectral centroid
    pub spectral_centroid: f32,
    /// Spectral rolloff
    pub spectral_rolloff: f32,
    /// MFCC coefficients (optional)
    pub mfcc: Option<Vec<f32>>,
    /// Estimated pitch in Hz (optional)
    pub pitch: Option<f32>,
}

/// Result of processing a single frame
#[derive(Debug, Clone)]
pub struct FrameProcessingResult {
    /// VAD result for this frame
    pub vad_result: VadResult,
    /// Extracted features (if voice detected)
    pub features: Option<AudioFeatures>,
    /// Frame index in the chunk
    pub frame_index: usize,
}

/// Result of processing an audio chunk
#[derive(Debug, Clone)]
pub struct AudioProcessingResult {
    /// Results for each frame
    pub frames: Vec<FrameProcessingResult>,
    /// Total processing time in milliseconds
    pub processing_time_ms: f32,
    /// Ratio of frames with voice activity (0.0 - 1.0)
    pub voice_activity_ratio: f32,
    /// Overall quality score (0.0 - 1.0)
    pub quality_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_creation() {
        let config = VadConfig::default();
        let vad = VoiceActivityDetector::new(config);
        let stats = vad.get_statistics();

        assert_eq!(stats.frames_processed, 0);
        assert_eq!(stats.buffer_size, 0);
    }

    #[test]
    fn test_vad_voice_detection() {
        let mut vad = VoiceActivityDetector::new(VadConfig::default());

        // Test with silence (low energy)
        let silence = vec![0.001f32; 512];
        let result = vad.process_frame(&silence);

        assert!(!result.voice_detected);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);

        // Test with voice-like signal (higher energy)
        let voice = vec![0.1f32; 512];
        let result = vad.process_frame(&voice);

        // Should be more likely to detect voice
        assert!(result.energy > 0.0);
    }

    #[test]
    fn test_energy_calculation() {
        let vad = VoiceActivityDetector::new(VadConfig::default());

        // Test with known values
        let frame = vec![0.5, -0.5, 0.5, -0.5];
        let energy = vad.calculate_energy(&frame);

        assert!((energy - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let vad = VoiceActivityDetector::new(VadConfig::default());

        // Alternating signal should have high ZCR
        let frame = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let zcr = vad.calculate_zero_crossing_rate(&frame);

        assert!(zcr > 0.8); // Should be close to 1.0
    }

    #[test]
    fn test_audio_processor() {
        let config = AudioProcessorConfig::default();
        let mut processor = AudioProcessor::new(config);

        let audio_data = vec![0.1f32; 1024]; // Some audio data
        let result = processor.process_chunk(&audio_data).unwrap();

        assert!(!result.frames.is_empty());
        assert!(result.voice_activity_ratio >= 0.0 && result.voice_activity_ratio <= 1.0);
        assert!(result.quality_score >= 0.0 && result.quality_score <= 1.0);
    }

    #[test]
    fn test_feature_extraction() {
        let config = FeatureExtractionConfig::default();
        let extractor = FeatureExtractor::new(config);

        let frame = vec![0.1f32; 512];
        let features = extractor.extract_features(&frame).unwrap();

        assert!(features.energy > 0.0);
        assert!(features.zero_crossing_rate >= 0.0);
        assert!(features.spectral_centroid >= 0.0 && features.spectral_centroid <= 1.0);
        assert!(features.mfcc.is_some());
        assert!(features.pitch.is_some());
    }

    #[test]
    fn test_vad_reset() {
        let mut vad = VoiceActivityDetector::new(VadConfig::default());

        // Process some frames
        let frame = vec![0.1f32; 512];
        vad.process_frame(&frame);
        vad.process_frame(&frame);

        let stats_before = vad.get_statistics();
        assert!(stats_before.frames_processed > 0);

        // Reset and check
        vad.reset();
        let stats_after = vad.get_statistics();
        assert_eq!(stats_after.frames_processed, 0);
        assert_eq!(stats_after.buffer_size, 0);
    }

    #[test]
    fn test_adaptive_threshold() {
        let config = VadConfig {
            history_frames: 20,
            initial_noise_floor: 0.001,
            ..Default::default()
        };
        let mut vad = VoiceActivityDetector::new(config);

        // Process multiple quiet frames to establish noise floor
        let quiet_frame = vec![0.002f32; 512];
        for _ in 0..15 {
            vad.process_frame(&quiet_frame);
        }

        let stats = vad.get_statistics();
        assert!(stats.current_noise_floor > 0.0);
        assert!(stats.adaptive_threshold > stats.current_noise_floor);
    }
}
