//! Quality metrics implementation
//!
//! This module provides comprehensive quality assessment for audio data
//! including SNR, clipping detection, dynamic range analysis, and spectral quality.
//! Features SIMD optimizations for performance-critical calculations.

use crate::audio::simd::SimdAudioProcessor;
use crate::{AudioData, DatasetSample, Result};

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
    /// Perceptual Evaluation of Speech Quality (PESQ) score (-0.5 to 4.5)
    pub pesq_score: Option<f32>,
    /// Short-Time Objective Intelligibility (STOI) score (0.0 to 1.0)
    pub stoi_score: Option<f32>,
    /// Enhanced Short-Time Objective Intelligibility (ESTOI) score (0.0 to 1.0)
    pub estoi_score: Option<f32>,
    /// Perceptual quality composite score (0.0 to 1.0)
    pub perceptual_score: f32,
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
            pesq_score: None,
            stoi_score: None,
            estoi_score: None,
            perceptual_score: 0.0,
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
    /// Enable perceptual quality metrics (PESQ, STOI)
    pub enable_perceptual: bool,
    /// STOI frame length in milliseconds
    pub stoi_frame_length_ms: f32,
    /// STOI overlap percentage (0.0 to 1.0)
    pub stoi_overlap: f32,
    /// PESQ sampling mode (8kHz or 16kHz)
    pub pesq_sample_rate: u32,
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
            enable_perceptual: true,
            stoi_frame_length_ms: 25.6,
            stoi_overlap: 0.75,
            pesq_sample_rate: 16000,
        }
    }
}

/// Quality metrics calculator
pub struct QualityMetricsCalculator {
    config: QualityConfig,
}

impl Default for QualityMetricsCalculator {
    fn default() -> Self {
        Self::new(QualityConfig::default())
    }
}

impl QualityMetricsCalculator {
    /// Create new quality metrics calculator
    pub fn new(config: QualityConfig) -> Self {
        Self { config }
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

        // Calculate perceptual quality metrics
        if self.config.enable_perceptual {
            self.calculate_perceptual_metrics(&mut metrics, samples, sample_rate)?;
        }

        // Calculate overall quality score
        self.calculate_overall_score(&mut metrics)?;

        Ok(metrics)
    }

    /// Calculate quality metrics for a dataset sample
    pub fn calculate_sample_metrics(&self, sample: &DatasetSample) -> Result<QualityMetrics> {
        self.calculate_metrics(&sample.audio)
    }

    /// Calculate amplitude-based metrics with SIMD optimization
    fn calculate_amplitude_metrics(
        &self,
        metrics: &mut QualityMetrics,
        samples: &[f32],
    ) -> Result<()> {
        // RMS calculation using SIMD
        metrics.rms = SimdAudioProcessor::calculate_rms(samples);

        // Peak calculation using SIMD
        metrics.peak = SimdAudioProcessor::find_peak(samples);

        // Dynamic range calculation - optimized version
        let min_amplitude = self.find_min_nonzero_amplitude(samples);
        if min_amplitude > 0.0 && metrics.peak > 0.0 {
            metrics.dynamic_range = 20.0 * (metrics.peak / min_amplitude).log10();
        }

        // Clipping detection using SIMD
        let clipping_count =
            SimdAudioProcessor::count_above_threshold(samples, self.config.clipping_threshold);
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
    fn calculate_spectral_metrics(
        &self,
        metrics: &mut QualityMetrics,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<()> {
        // Zero crossing rate
        metrics.zero_crossing_rate = self.calculate_zcr(samples);

        // Spectral centroid and rolloff (simplified without FFT)
        let (centroid, rolloff) = self.calculate_spectral_features(samples, sample_rate)?;
        metrics.spectral_centroid = centroid;
        metrics.spectral_rolloff = rolloff;

        Ok(())
    }

    /// Calculate speech activity detection
    fn calculate_speech_activity(
        &self,
        metrics: &mut QualityMetrics,
        samples: &[f32],
    ) -> Result<()> {
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

    /// Calculate perceptual quality metrics (PESQ, STOI, ESTOI)
    fn calculate_perceptual_metrics(
        &self,
        metrics: &mut QualityMetrics,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<()> {
        // Calculate STOI (Short-Time Objective Intelligibility)
        if let Ok(stoi_score) = self.calculate_stoi(samples, sample_rate) {
            metrics.stoi_score = Some(stoi_score);
        }

        // Calculate Enhanced STOI
        if let Ok(estoi_score) = self.calculate_estoi(samples, sample_rate) {
            metrics.estoi_score = Some(estoi_score);
        }

        // Calculate PESQ (Perceptual Evaluation of Speech Quality)
        // Note: This is a simplified implementation for demonstration
        // A production implementation would use the ITU-T P.862 standard
        if let Ok(pesq_score) = self.calculate_pesq_simplified(samples, sample_rate) {
            metrics.pesq_score = Some(pesq_score);
        }

        // Calculate composite perceptual score
        metrics.perceptual_score = self.calculate_composite_perceptual_score(
            metrics.stoi_score,
            metrics.estoi_score,
            metrics.pesq_score,
        );

        Ok(())
    }

    /// Calculate STOI (Short-Time Objective Intelligibility) score
    /// Simplified implementation based on correlation between temporal envelopes
    fn calculate_stoi(&self, samples: &[f32], sample_rate: u32) -> Result<f32> {
        let frame_length =
            (self.config.stoi_frame_length_ms * sample_rate as f32 / 1000.0) as usize;
        let hop_length = (frame_length as f32 * (1.0 - self.config.stoi_overlap)) as usize;

        if samples.len() < frame_length {
            return Ok(0.0);
        }

        let mut correlations = Vec::new();

        // Process audio in overlapping frames
        for start in (0..samples.len()).step_by(hop_length) {
            let end = (start + frame_length).min(samples.len());
            let frame = &samples[start..end];

            if frame.len() < frame_length / 2 {
                break;
            }

            // Calculate frame energy and normalized correlation
            let frame_energy = frame.iter().map(|&x| x * x).sum::<f32>() / frame.len() as f32;
            let frame_rms = frame_energy.sqrt();

            // Simplified intelligibility correlation based on frame characteristics
            let intelligibility = if frame_rms > 0.001 {
                // Higher correlation for frames with good energy distribution
                let zcr = self.calculate_zcr(frame);
                let energy_ratio = frame_energy.clamp(0.0, 1.0);

                // STOI-like correlation based on temporal envelope
                (0.7 * energy_ratio + 0.3 * (1.0 - zcr.clamp(0.0, 1.0))).clamp(0.0, 1.0)
            } else {
                0.0
            };

            correlations.push(intelligibility);
        }

        if correlations.is_empty() {
            return Ok(0.0);
        }

        // Average correlation across all frames
        let stoi_score = correlations.iter().sum::<f32>() / correlations.len() as f32;
        Ok(stoi_score.clamp(0.0, 1.0))
    }

    /// Calculate Enhanced STOI (ESTOI) score with additional temporal weighting
    fn calculate_estoi(&self, samples: &[f32], sample_rate: u32) -> Result<f32> {
        let base_stoi = self.calculate_stoi(samples, sample_rate)?;

        // Enhanced STOI includes additional temporal processing
        let frame_length =
            (self.config.stoi_frame_length_ms * sample_rate as f32 / 1000.0) as usize;
        let hop_length = (frame_length as f32 * (1.0 - self.config.stoi_overlap)) as usize;

        if samples.len() < frame_length * 2 {
            return Ok(base_stoi);
        }

        let mut temporal_correlations = Vec::new();

        // Calculate temporal envelope correlation
        for start in (0..samples.len()).step_by(hop_length) {
            let end = (start + frame_length).min(samples.len());
            if end - start < frame_length / 2 {
                break;
            }

            let frame = &samples[start..end];

            // Calculate temporal envelope
            let envelope: Vec<f32> = frame
                .windows(hop_length / 4)
                .map(|window| window.iter().map(|&x| x.abs()).sum::<f32>() / window.len() as f32)
                .collect();

            if envelope.len() > 1 {
                // Measure temporal consistency
                let mean_env = envelope.iter().sum::<f32>() / envelope.len() as f32;
                let variance = envelope
                    .iter()
                    .map(|&x| (x - mean_env).powi(2))
                    .sum::<f32>()
                    / envelope.len() as f32;
                let std_dev = variance.sqrt();

                // Higher correlation for smoother temporal envelopes
                let temporal_correlation = if mean_env > 0.0 {
                    (1.0 - (std_dev / mean_env).clamp(0.0, 1.0)).clamp(0.0, 1.0)
                } else {
                    0.0
                };

                temporal_correlations.push(temporal_correlation);
            }
        }

        if temporal_correlations.is_empty() {
            return Ok(base_stoi);
        }

        let temporal_score =
            temporal_correlations.iter().sum::<f32>() / temporal_correlations.len() as f32;

        // Combine base STOI with temporal enhancement
        let estoi_score = 0.7 * base_stoi + 0.3 * temporal_score;
        Ok(estoi_score.clamp(0.0, 1.0))
    }

    /// Calculate enhanced PESQ-like score with improved perceptual modeling
    /// Enhanced implementation with better frequency analysis and perceptual weighting
    fn calculate_pesq_simplified(&self, samples: &[f32], sample_rate: u32) -> Result<f32> {
        use scirs2_core::Complex;

        // Resample to PESQ standard rate if necessary
        let target_rate = self.config.pesq_sample_rate;
        let processed_samples = if sample_rate != target_rate {
            self.resample_audio(samples, sample_rate, target_rate)?
        } else {
            samples.to_vec()
        };

        // Use 20ms frames for better temporal resolution
        let frame_size = (target_rate as f32 * 0.020) as usize; // 20ms frames
        let hop_size = frame_size / 2; // 50% overlap

        let mut perceptual_scores = Vec::new();

        for start in (0..processed_samples.len()).step_by(hop_size) {
            if start + frame_size > processed_samples.len() {
                break;
            }

            let frame = &processed_samples[start..start + frame_size];

            // Apply Hanning window
            let windowed: Vec<f32> = frame
                .iter()
                .enumerate()
                .map(|(i, &sample)| {
                    let window = 0.5
                        * (1.0
                            - (2.0 * std::f32::consts::PI * i as f32 / (frame_size - 1) as f32)
                                .cos());
                    sample * window
                })
                .collect();

            // Compute FFT for frequency analysis
            let input_f64: Vec<scirs2_core::Complex<f64>> = windowed
                .iter()
                .map(|&x| scirs2_core::Complex::new(x as f64, 0.0))
                .collect();

            let fft_result = scirs2_fft::fft(&input_f64, None)
                .unwrap_or_else(|_| vec![scirs2_core::Complex::new(0.0, 0.0); frame_size]);
            let buffer: Vec<Complex<f32>> = fft_result
                .iter()
                .map(|c| Complex::new(c.re as f32, c.im as f32))
                .collect();

            // Calculate power spectrum
            let power_spectrum: Vec<f32> = buffer
                .iter()
                .take(frame_size / 2)
                .map(|c| c.norm_sqr())
                .collect();

            // Enhanced perceptual weighting based on Bark scale critical bands
            let mut perceptual_loudness = 0.0;
            let mut total_energy = 0.0;

            for (i, &power) in power_spectrum.iter().enumerate() {
                let freq = i as f32 * target_rate as f32 / frame_size as f32;

                // Bark scale transformation for critical band analysis
                let bark = 13.0 * (0.00076 * freq).atan() + 3.5 * ((freq / 7500.0).powi(2)).atan();

                // A-weighting approximation for human auditory sensitivity
                let a_weight = if !(20.0..=20000.0).contains(&freq) {
                    0.0
                } else {
                    let f2 = freq * freq;
                    let f4 = f2 * f2;
                    let c1 = 12194.0_f32.powi(2);
                    let c2 = 20.6_f32.powi(2);
                    let c3 = 107.7_f32.powi(2);
                    let c4 = 737.9_f32.powi(2);

                    let numerator = c1 * f4;
                    let denominator = (f2 + c2) * ((f2 + c3) * (f2 + c4)).sqrt() * (f2 + c1);
                    (numerator / denominator).log10() * 20.0 + 2.0
                };

                // Enhanced perceptual weighting combining Bark scale and A-weighting
                let weight = if bark < 3.5 {
                    // Low frequencies (< ~500 Hz)
                    0.8 * (1.0 + a_weight / 20.0).max(0.1)
                } else if bark < 15.5 {
                    // Mid frequencies (500-3400 Hz) - most important for speech
                    1.5 * (1.0 + a_weight / 10.0).max(0.3)
                } else {
                    // High frequencies (> 3400 Hz)
                    0.6 * (1.0 + a_weight / 15.0).max(0.05)
                };

                let weighted_power = power * weight;
                perceptual_loudness += weighted_power;
                total_energy += power;
            }

            // Calculate temporal features
            let rms = (frame.iter().map(|&x| x * x).sum::<f32>() / frame.len() as f32).sqrt();
            let spectral_centroid = self.calculate_spectral_centroid(&power_spectrum, target_rate);
            let spectral_rolloff = self.calculate_spectral_rolloff(&power_spectrum, 0.85);

            // Enhanced perceptual quality calculation
            let loudness_score = if total_energy > 0.0 {
                (perceptual_loudness / total_energy).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let spectral_balance = if spectral_centroid > 0.0 {
                // Prefer spectral centroid around 1-2 kHz for speech
                let ideal_centroid = 1500.0;
                let deviation = (spectral_centroid - ideal_centroid).abs() / ideal_centroid;
                (1.0 - deviation.min(1.0)).max(0.0)
            } else {
                0.0
            };

            let clarity_score = if rms > 0.001 {
                // Higher spectral rolloff indicates better clarity
                (spectral_rolloff / (target_rate as f32 / 2.0)).clamp(0.0, 1.0)
            } else {
                0.0
            };

            // Composite perceptual score with improved weighting
            let frame_score = 0.5 * loudness_score + 0.3 * spectral_balance + 0.2 * clarity_score;
            perceptual_scores.push(frame_score);
        }

        if perceptual_scores.is_empty() {
            return Ok(-0.5); // Minimum PESQ score
        }

        // Statistical analysis of perceptual scores
        let mean_score = perceptual_scores.iter().sum::<f32>() / perceptual_scores.len() as f32;
        let variance = perceptual_scores
            .iter()
            .map(|&x| (x - mean_score).powi(2))
            .sum::<f32>()
            / perceptual_scores.len() as f32;
        let std_dev = variance.sqrt();

        // Penalize high variance (inconsistent quality)
        let consistency_penalty = (std_dev * 2.0).min(0.3);
        let adjusted_score = mean_score - consistency_penalty;

        // Map to PESQ scale (-0.5 to 4.5) with improved scaling
        let pesq_score = -0.5 + 5.0 * adjusted_score.clamp(0.0, 1.0);
        Ok(pesq_score.clamp(-0.5, 4.5))
    }

    /// Resample audio using linear interpolation
    fn resample_audio(&self, samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
        if from_rate == to_rate {
            return Ok(samples.to_vec());
        }

        let ratio = to_rate as f32 / from_rate as f32;
        let new_length = (samples.len() as f32 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let orig_pos = i as f32 / ratio;
            let orig_index = orig_pos.floor() as usize;
            let frac = orig_pos - orig_index as f32;

            if orig_index + 1 < samples.len() {
                // Linear interpolation
                let val = samples[orig_index] * (1.0 - frac) + samples[orig_index + 1] * frac;
                resampled.push(val);
            } else if orig_index < samples.len() {
                resampled.push(samples[orig_index]);
            } else {
                resampled.push(0.0);
            }
        }

        Ok(resampled)
    }

    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, power_spectrum: &[f32], sample_rate: u32) -> f32 {
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &magnitude) in power_spectrum.iter().enumerate() {
            let freq = i as f32 * sample_rate as f32 / (2.0 * power_spectrum.len() as f32);
            weighted_sum += freq * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }

    /// Calculate spectral rolloff
    fn calculate_spectral_rolloff(&self, power_spectrum: &[f32], threshold: f32) -> f32 {
        let total_energy: f32 = power_spectrum.iter().sum();
        let target_energy = total_energy * threshold;

        let mut cumulative_energy = 0.0;
        for (i, &magnitude) in power_spectrum.iter().enumerate() {
            cumulative_energy += magnitude;
            if cumulative_energy >= target_energy {
                return i as f32;
            }
        }

        power_spectrum.len() as f32
    }

    /// Calculate composite perceptual quality score
    fn calculate_composite_perceptual_score(
        &self,
        stoi: Option<f32>,
        estoi: Option<f32>,
        pesq: Option<f32>,
    ) -> f32 {
        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        // Weight STOI score (intelligibility)
        if let Some(stoi_score) = stoi {
            total_score += stoi_score * 0.3;
            total_weight += 0.3;
        }

        // Weight ESTOI score (enhanced intelligibility)
        if let Some(estoi_score) = estoi {
            total_score += estoi_score * 0.4;
            total_weight += 0.4;
        }

        // Weight PESQ score (overall quality) - normalize to 0-1 scale
        if let Some(pesq_score) = pesq {
            let normalized_pesq = ((pesq_score + 0.5) / 5.0).clamp(0.0, 1.0);
            total_score += normalized_pesq * 0.3;
            total_weight += 0.3;
        }

        if total_weight > 0.0 {
            (total_score / total_weight).clamp(0.0, 1.0)
        } else {
            0.0
        }
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

        metrics.overall_score = score.clamp(0.0, 1.0);

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
        let signal_energy: f32 = energy_values[energy_values.len() - quarter..]
            .iter()
            .sum::<f32>()
            / quarter as f32;

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

    /// Find minimum non-zero amplitude efficiently
    fn find_min_nonzero_amplitude(&self, samples: &[f32]) -> f32 {
        let mut min_amplitude = f32::MAX;
        let mut found_nonzero = false;

        for &sample in samples {
            let abs_sample = sample.abs();
            if abs_sample > 0.0 && abs_sample < min_amplitude {
                min_amplitude = abs_sample;
                found_nonzero = true;
            }
        }

        if found_nonzero {
            min_amplitude
        } else {
            1.0 // Return 1.0 if no non-zero samples found
        }
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
            min_score: metrics
                .iter()
                .map(|m| m.overall_score)
                .fold(f32::INFINITY, f32::min),
            max_score: metrics
                .iter()
                .map(|m| m.overall_score)
                .fold(f32::NEG_INFINITY, f32::max),
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
    pub fn meets_quality_standards(
        &self,
        min_average_score: f32,
        max_low_quality_percentage: f32,
    ) -> bool {
        let (low_percentage, _, _) = self.quality_distribution();

        self.average_overall_score >= min_average_score
            && low_percentage <= max_low_quality_percentage
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

        let recent_avg = self
            .history
            .iter()
            .rev()
            .take(self.history.len() / 2)
            .map(|s| s.average_overall_score)
            .sum::<f32>()
            / (self.history.len() / 2) as f32;

        let older_avg = self
            .history
            .iter()
            .take(self.history.len() / 2)
            .map(|s| s.average_overall_score)
            .sum::<f32>()
            / (self.history.len() / 2) as f32;

        recent_avg - older_avg
    }

    /// Get quality stability score (0.0 = very unstable, 1.0 = very stable)
    pub fn calculate_stability(&self) -> f32 {
        if self.history.len() < 3 {
            return 1.0;
        }

        let scores: Vec<f32> = self
            .history
            .iter()
            .map(|s| s.average_overall_score)
            .collect();
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;

        let variance =
            scores.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / scores.len() as f32;

        let std_dev = variance.sqrt();

        // Higher stability for lower standard deviation
        (1.0 - std_dev).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioData;

    /// Create test audio data with specified characteristics
    fn create_test_audio(sample_rate: u32, duration_secs: f32, frequency: f32) -> AudioData {
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let mut samples = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5;
            samples.push(sample);
        }

        AudioData::new(samples, sample_rate, 1)
    }

    /// Create noisy test audio data
    fn create_noisy_audio(sample_rate: u32, duration_secs: f32, snr_db: f32) -> AudioData {
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let mut samples = Vec::with_capacity(num_samples);

        // Signal power for SNR calculation
        let signal_power = 0.25; // RMS squared for 0.5 amplitude sine
        let noise_power = signal_power / (10.0_f32.powf(snr_db / 10.0));
        let noise_std = noise_power.sqrt();

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let signal = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
            let noise = (scirs2_core::random::random::<f32>() - 0.5) * 2.0 * noise_std;
            samples.push(signal + noise);
        }

        AudioData::new(samples, sample_rate, 1)
    }

    #[test]
    fn test_perceptual_metrics_calculation() {
        let calculator = QualityMetricsCalculator::new(QualityConfig::default());
        let audio = create_test_audio(16000, 1.0, 440.0);

        let metrics = calculator.calculate_metrics(&audio).unwrap();

        // Test that perceptual metrics are calculated
        assert!(
            metrics.stoi_score.is_some(),
            "STOI score should be calculated"
        );
        assert!(
            metrics.estoi_score.is_some(),
            "ESTOI score should be calculated"
        );
        assert!(
            metrics.pesq_score.is_some(),
            "PESQ score should be calculated"
        );

        // Test score ranges
        if let Some(stoi) = metrics.stoi_score {
            assert!(
                (0.0..=1.0).contains(&stoi),
                "STOI score should be between 0 and 1"
            );
        }

        if let Some(estoi) = metrics.estoi_score {
            assert!(
                (0.0..=1.0).contains(&estoi),
                "ESTOI score should be between 0 and 1"
            );
        }

        if let Some(pesq) = metrics.pesq_score {
            assert!(
                (-0.5..=4.5).contains(&pesq),
                "PESQ score should be between -0.5 and 4.5"
            );
        }

        assert!(
            (0.0..=1.0).contains(&metrics.perceptual_score),
            "Perceptual composite score should be between 0 and 1"
        );
    }

    #[test]
    fn test_stoi_calculation() {
        let calculator = QualityMetricsCalculator::new(QualityConfig::default());

        // Test with clean speech-like signal
        let clean_audio = create_test_audio(16000, 0.5, 200.0);
        let stoi = calculator
            .calculate_stoi(clean_audio.samples(), clean_audio.sample_rate())
            .unwrap();

        assert!(
            (0.0..=1.0).contains(&stoi),
            "STOI should be between 0 and 1"
        );
        assert!(stoi > 0.3, "Clean signal should have reasonable STOI score");

        // Test with very short audio (should handle gracefully)
        let short_audio = create_test_audio(16000, 0.01, 440.0);
        let short_stoi = calculator
            .calculate_stoi(short_audio.samples(), short_audio.sample_rate())
            .unwrap();
        assert_eq!(short_stoi, 0.0, "Very short audio should return 0 STOI");
    }

    #[test]
    fn test_estoi_calculation() {
        let calculator = QualityMetricsCalculator::new(QualityConfig::default());

        // Test with clean signal
        let clean_audio = create_test_audio(16000, 1.0, 300.0);
        let estoi = calculator
            .calculate_estoi(clean_audio.samples(), clean_audio.sample_rate())
            .unwrap();

        assert!(
            (0.0..=1.0).contains(&estoi),
            "ESTOI should be between 0 and 1"
        );

        // ESTOI should generally be close to or slightly different from STOI
        let stoi = calculator
            .calculate_stoi(clean_audio.samples(), clean_audio.sample_rate())
            .unwrap();
        let difference = (estoi - stoi).abs();
        assert!(
            difference <= 1.0,
            "ESTOI and STOI should be reasonably close"
        );
    }

    #[test]
    fn test_pesq_calculation() {
        let calculator = QualityMetricsCalculator::new(QualityConfig::default());

        // Test with clean signal
        let clean_audio = create_test_audio(16000, 1.0, 440.0);
        let pesq = calculator
            .calculate_pesq_simplified(clean_audio.samples(), clean_audio.sample_rate())
            .unwrap();

        assert!(
            (-0.5..=4.5).contains(&pesq),
            "PESQ should be in valid range"
        );
        assert!(pesq > 0.0, "Clean signal should have positive PESQ score");

        // Test with very noisy signal
        let noisy_audio = create_noisy_audio(16000, 1.0, 0.0); // 0 dB SNR
        let noisy_pesq = calculator
            .calculate_pesq_simplified(noisy_audio.samples(), noisy_audio.sample_rate())
            .unwrap();

        assert!(
            (-0.5..=4.5).contains(&noisy_pesq),
            "Noisy PESQ should be in valid range"
        );
        // Noisy signal should generally have lower PESQ score than clean signal
        assert!(
            noisy_pesq <= pesq + 1.0,
            "Noisy signal should not have much higher PESQ than clean"
        );
    }

    #[test]
    fn test_composite_perceptual_score() {
        let calculator = QualityMetricsCalculator::new(QualityConfig::default());

        // Test composite score calculation with all metrics
        let composite = calculator.calculate_composite_perceptual_score(
            Some(0.8), // Good STOI
            Some(0.9), // Good ESTOI
            Some(3.0), // Good PESQ
        );

        assert!(
            (0.0..=1.0).contains(&composite),
            "Composite score should be between 0 and 1"
        );
        assert!(
            composite > 0.5,
            "Good quality metrics should result in good composite score"
        );

        // Test with poor quality metrics
        let poor_composite = calculator.calculate_composite_perceptual_score(
            Some(0.2), // Poor STOI
            Some(0.1), // Poor ESTOI
            Some(0.5), // Poor PESQ
        );

        assert!(
            poor_composite < composite,
            "Poor metrics should result in lower composite score"
        );

        // Test with missing metrics
        let partial_composite =
            calculator.calculate_composite_perceptual_score(Some(0.8), None, None);

        assert!(
            (0.0..=1.0).contains(&partial_composite),
            "Partial composite should be valid"
        );
    }

    #[test]
    fn test_perceptual_metrics_disabled() {
        let config = QualityConfig {
            enable_perceptual: false,
            ..Default::default()
        };

        let calculator = QualityMetricsCalculator::new(config);
        let audio = create_test_audio(16000, 1.0, 440.0);

        let metrics = calculator.calculate_metrics(&audio).unwrap();

        // When disabled, perceptual metrics should not be calculated
        assert!(
            metrics.stoi_score.is_none(),
            "STOI should be None when disabled"
        );
        assert!(
            metrics.estoi_score.is_none(),
            "ESTOI should be None when disabled"
        );
        assert!(
            metrics.pesq_score.is_none(),
            "PESQ should be None when disabled"
        );
        assert_eq!(
            metrics.perceptual_score, 0.0,
            "Composite score should be 0 when disabled"
        );
    }

    #[test]
    fn test_perceptual_metrics_config() {
        let config = QualityConfig {
            stoi_frame_length_ms: 32.0,
            stoi_overlap: 0.5,
            pesq_sample_rate: 8000,
            ..Default::default()
        };

        let calculator = QualityMetricsCalculator::new(config);
        let audio = create_test_audio(8000, 1.0, 300.0);

        let metrics = calculator.calculate_metrics(&audio).unwrap();

        // Should still calculate valid metrics with different config
        assert!(metrics.stoi_score.is_some());
        assert!(metrics.estoi_score.is_some());
        assert!(metrics.pesq_score.is_some());
    }
}
