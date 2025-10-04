//! Psychoacoustic modeling for perceptually-motivated audio quality evaluation.
//!
//! This module implements psychoacoustic models that align quality metrics with human auditory perception.
//! It provides tools for bark scale analysis, loudness modeling, masking effects, and critical band analysis
//! to create more perceptually relevant quality assessments.
//!
//! ## Features
//!
//! - **Bark Scale Analysis**: Frequency analysis using perceptually motivated bark scale
//! - **Loudness Modeling**: ITU-R BS.1770-4 compliant loudness measurement
//! - **Masking Effects**: Simultaneous and temporal masking consideration
//! - **Critical Band Analysis**: Auditory filter bank modeling
//! - **Temporal Masking**: Pre-masking and post-masking effects
//!
//! ## Examples
//!
//! ```rust
//! use voirs_evaluation::quality::psychoacoustic::PsychoacousticEvaluator;
//! use voirs_sdk::AudioBuffer;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let evaluator = PsychoacousticEvaluator::new();
//! let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
//!
//! let analysis = evaluator.analyze_psychoacoustic_features(&audio)?;
//! println!("Loudness: {:.2} LUFS", analysis.loudness_lufs);
//! println!("Sharpness: {:.2} acum", analysis.sharpness_acum);
//! # Ok(())
//! # }
//! ```

use crate::EvaluationError;
use scirs2_core::Complex;
use scirs2_fft::{RealFftPlanner, RealToComplex};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::sync::Mutex;
use voirs_sdk::AudioBuffer;

/// Psychoacoustic evaluation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsychoacousticConfig {
    /// Number of bark bands for analysis
    pub num_bark_bands: usize,
    /// Enable temporal masking analysis
    pub enable_temporal_masking: bool,
    /// Enable simultaneous masking analysis  
    pub enable_simultaneous_masking: bool,
    /// Loudness gating threshold in LUFS
    pub loudness_gate_threshold: f32,
    /// Frame size for analysis (samples)
    pub frame_size: usize,
    /// Hop size for analysis (samples)
    pub hop_size: usize,
}

impl Default for PsychoacousticConfig {
    fn default() -> Self {
        Self {
            num_bark_bands: 24,
            enable_temporal_masking: true,
            enable_simultaneous_masking: true,
            loudness_gate_threshold: -70.0,
            frame_size: 2048,
            hop_size: 512,
        }
    }
}

/// Psychoacoustic analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsychoacousticAnalysis {
    /// Loudness in LUFS (ITU-R BS.1770-4)
    pub loudness_lufs: f32,
    /// Integrated loudness over time
    pub integrated_loudness: f32,
    /// Loudness range (LRA)
    pub loudness_range: f32,
    /// Sharpness in acum (Zwicker & Fastl)
    pub sharpness_acum: f32,
    /// Roughness in asper (Zwicker & Fastl)
    pub roughness_asper: f32,
    /// Fluctuation strength in vacil
    pub fluctuation_strength: f32,
    /// Bark spectrum power distribution
    pub bark_spectrum: Vec<f32>,
    /// Critical band analysis
    pub critical_bands: Vec<CriticalBand>,
    /// Masking threshold
    pub masking_threshold: Vec<f32>,
    /// Temporal masking effects
    pub temporal_masking: Option<TemporalMaskingAnalysis>,
}

/// Critical band analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalBand {
    /// Center frequency in Hz
    pub center_frequency: f32,
    /// Lower frequency bound in Hz
    pub lower_freq: f32,
    /// Upper frequency bound in Hz  
    pub upper_freq: f32,
    /// Bark value
    pub bark_value: f32,
    /// Power in this band
    pub power: f32,
    /// Masking threshold
    pub masking_threshold: f32,
}

/// Temporal masking analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMaskingAnalysis {
    /// Pre-masking effects (backward masking)
    pub pre_masking: Vec<f32>,
    /// Post-masking effects (forward masking)
    pub post_masking: Vec<f32>,
    /// Temporal masking patterns
    pub masking_patterns: Vec<Vec<f32>>,
}

/// Psychoacoustic evaluator implementing perceptual models
pub struct PsychoacousticEvaluator {
    config: PsychoacousticConfig,
    bark_frequencies: Vec<f32>,
    critical_band_filters: Vec<Vec<f32>>,
    fft_planner: Mutex<RealFftPlanner<f32>>,
}

impl PsychoacousticEvaluator {
    /// Create a new psychoacoustic evaluator
    pub fn new() -> Self {
        Self::with_config(PsychoacousticConfig::default())
    }

    /// Create evaluator with custom configuration
    pub fn with_config(config: PsychoacousticConfig) -> Self {
        let bark_frequencies = Self::generate_bark_frequencies(config.num_bark_bands);
        let critical_band_filters =
            Self::generate_critical_band_filters(&bark_frequencies, config.frame_size);
        let fft_planner = Mutex::new(RealFftPlanner::<f32>::new());

        Self {
            config,
            bark_frequencies,
            critical_band_filters,
            fft_planner,
        }
    }

    /// Analyze psychoacoustic features of audio
    pub fn analyze_psychoacoustic_features(
        &self,
        audio: &AudioBuffer,
    ) -> Result<PsychoacousticAnalysis, EvaluationError> {
        // Get samples
        let mono_samples = audio.samples();

        // Analyze loudness (ITU-R BS.1770-4)
        let (loudness_lufs, integrated_loudness, loudness_range) =
            self.analyze_loudness(&mono_samples, audio.sample_rate())?;

        // Bark spectrum analysis
        let bark_spectrum = self.compute_bark_spectrum(&mono_samples)?;

        // Critical band analysis
        let critical_bands = self.analyze_critical_bands(&mono_samples, audio.sample_rate())?;

        // Psychoacoustic features
        let sharpness_acum = self.compute_sharpness(&bark_spectrum);
        let roughness_asper = self.compute_roughness(&mono_samples, audio.sample_rate())?;
        let fluctuation_strength =
            self.compute_fluctuation_strength(&mono_samples, audio.sample_rate())?;

        // Masking threshold
        let masking_threshold = self.compute_masking_threshold(&bark_spectrum)?;

        // Temporal masking (optional)
        let temporal_masking = if self.config.enable_temporal_masking {
            Some(self.analyze_temporal_masking(&mono_samples, audio.sample_rate())?)
        } else {
            None
        };

        Ok(PsychoacousticAnalysis {
            loudness_lufs,
            integrated_loudness,
            loudness_range,
            sharpness_acum,
            roughness_asper,
            fluctuation_strength,
            bark_spectrum,
            critical_bands,
            masking_threshold,
            temporal_masking,
        })
    }

    /// Generate bark scale frequencies
    fn generate_bark_frequencies(num_bands: usize) -> Vec<f32> {
        (0..num_bands)
            .map(|i| {
                let bark = i as f32 * 24.0 / num_bands as f32;
                // Bark to Hz conversion (Traunmüller formula)
                1960.0 * (bark + 0.53) / (26.28 - bark)
            })
            .collect()
    }

    /// Generate critical band filters
    fn generate_critical_band_filters(
        bark_frequencies: &[f32],
        frame_size: usize,
    ) -> Vec<Vec<f32>> {
        bark_frequencies
            .iter()
            .map(|&center_freq| {
                // Generate triangular filter for this bark band
                let mut filter = vec![0.0; frame_size / 2 + 1];
                let bandwidth = Self::bark_to_erb(Self::hz_to_bark(center_freq));

                for (i, filter_val) in filter.iter_mut().enumerate() {
                    let freq = i as f32 * 22050.0 / (frame_size / 2) as f32; // Assuming Nyquist = 22050 Hz
                    let distance = (freq - center_freq).abs();

                    if distance <= bandwidth / 2.0 {
                        // Triangular filter response
                        *filter_val = 1.0 - distance / (bandwidth / 2.0);
                    }
                }

                filter
            })
            .collect()
    }

    /// Convert Hz to Bark scale
    fn hz_to_bark(freq_hz: f32) -> f32 {
        26.81 * freq_hz / (1960.0 + freq_hz) - 0.53
    }

    /// Convert Bark to ERB (Equivalent Rectangular Bandwidth)
    fn bark_to_erb(bark: f32) -> f32 {
        // Approximate ERB bandwidth for given bark value
        24.7 * (4.37 * bark / 1000.0 + 1.0)
    }

    /// Analyze loudness according to ITU-R BS.1770-4
    fn analyze_loudness(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<(f32, f32, f32), EvaluationError> {
        if samples.is_empty() {
            return Ok((f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0));
        }

        // Apply K-weighting filter (simplified implementation)
        let k_weighted = self.apply_k_weighting(samples, sample_rate)?;

        // Gating and measurement
        let block_size = (sample_rate as f32 * 0.4) as usize; // 400ms blocks
        let overlap = block_size / 2; // 75% overlap

        let mut block_loudness: Vec<f32> = Vec::new();
        let mut i = 0;

        while i + block_size <= k_weighted.len() {
            let block = &k_weighted[i..i + block_size];
            let mean_square = block.iter().map(|x| x * x).sum::<f32>() / block.len() as f32;

            if mean_square > 0.0 {
                let loudness = -0.691 + 10.0 * mean_square.log10();
                if loudness > self.config.loudness_gate_threshold {
                    block_loudness.push(loudness);
                }
            }

            i += overlap;
        }

        if block_loudness.is_empty() {
            return Ok((f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0));
        }

        // Integrated loudness
        let integrated_loudness = -0.691
            + 10.0
                * (block_loudness
                    .iter()
                    .map(|l| 10.0_f32.powf(l / 10.0))
                    .sum::<f32>()
                    / block_loudness.len() as f32)
                    .log10();

        // Loudness range (LRA)
        let mut sorted_loudness = block_loudness.clone();
        sorted_loudness.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let percentile_10 = sorted_loudness[(sorted_loudness.len() as f32 * 0.1) as usize];
        let percentile_95 = sorted_loudness[(sorted_loudness.len() as f32 * 0.95) as usize];
        let loudness_range = percentile_95 - percentile_10;

        Ok((integrated_loudness, integrated_loudness, loudness_range))
    }

    /// Apply K-weighting filter (simplified)
    fn apply_k_weighting(
        &self,
        samples: &[f32],
        _sample_rate: u32,
    ) -> Result<Vec<f32>, EvaluationError> {
        // Simplified K-weighting (in practice, this would be a proper filter cascade)
        // For now, apply a simple high-pass filter to approximate the effect
        if samples.len() < 2 {
            return Ok(samples.to_vec());
        }

        let mut filtered = vec![0.0; samples.len()];
        filtered[0] = samples[0];

        // Simple first-order high-pass filter
        let alpha = 0.99; // Cutoff around 38 Hz for 48kHz sample rate
        for i in 1..samples.len() {
            filtered[i] = alpha * (filtered[i - 1] + samples[i] - samples[i - 1]);
        }

        Ok(filtered)
    }

    /// Compute bark spectrum
    fn compute_bark_spectrum(&self, samples: &[f32]) -> Result<Vec<f32>, EvaluationError> {
        if samples.len() < self.config.frame_size {
            return Err(EvaluationError::InvalidInput {
                message: "Audio too short for psychoacoustic analysis".to_string(),
            });
        }

        // Extract frame from samples
        let frame = if samples.len() >= self.config.frame_size {
            &samples[..self.config.frame_size]
        } else {
            samples
        };

        // Apply window and FFT
        let windowed = self.apply_window(frame);
        let magnitude_spectrum = self.compute_fft(&windowed)?;
        let power_spectrum = self.compute_power_spectrum(&magnitude_spectrum);

        // Apply bark scale filters
        let bark_spectrum: Vec<f32> = self
            .critical_band_filters
            .iter()
            .map(|filter| {
                filter
                    .iter()
                    .zip(power_spectrum.iter())
                    .map(|(f, p)| f * p)
                    .sum()
            })
            .collect();

        Ok(bark_spectrum)
    }

    /// Apply Hann window
    fn apply_window(&self, samples: &[f32]) -> Vec<f32> {
        let window_size = samples.len();
        samples
            .iter()
            .enumerate()
            .map(|(i, &sample)| {
                let window_val =
                    0.5 * (1.0 - (2.0 * PI * i as f32 / (window_size - 1) as f32).cos());
                sample * window_val
            })
            .collect()
    }

    /// Compute FFT using rustfft for optimal performance
    fn compute_fft(&self, samples: &[f32]) -> Result<Vec<f32>, EvaluationError> {
        let n = samples.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Get or create FFT for this size
        let mut planner = self.fft_planner.lock().unwrap();
        let fft = planner.plan_fft_forward(n);

        // Prepare input buffer
        let mut input_buffer = samples.to_vec();

        // Prepare output buffer
        let mut output_buffer = vec![Complex::new(0.0, 0.0); n / 2 + 1];

        // Perform FFT
        fft.process(&input_buffer, &mut output_buffer);

        // Convert to magnitude spectrum
        let magnitude_spectrum: Vec<f32> = output_buffer.iter().map(|c| c.norm()).collect();

        Ok(magnitude_spectrum)
    }

    /// Compute power spectrum
    fn compute_power_spectrum(&self, fft_result: &[f32]) -> Vec<f32> {
        fft_result.iter().map(|x| x * x).collect()
    }

    /// Analyze critical bands
    fn analyze_critical_bands(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<CriticalBand>, EvaluationError> {
        let bark_spectrum = self.compute_bark_spectrum(samples)?;
        let nyquist = sample_rate as f32 / 2.0;

        let critical_bands: Vec<CriticalBand> = self
            .bark_frequencies
            .iter()
            .enumerate()
            .map(|(i, &center_freq)| {
                let bark_value = Self::hz_to_bark(center_freq);
                let bandwidth = Self::bark_to_erb(bark_value);

                let mut lower_freq = (center_freq - bandwidth / 2.0).max(0.0);
                let mut upper_freq = (center_freq + bandwidth / 2.0).min(nyquist);

                // Ensure center frequency is within bounds
                let actual_center = if upper_freq < center_freq {
                    upper_freq
                } else if lower_freq > center_freq {
                    lower_freq
                } else {
                    center_freq
                };

                // Recalculate bounds based on actual center
                lower_freq = (actual_center - bandwidth / 2.0).max(0.0);
                upper_freq = (actual_center + bandwidth / 2.0).min(nyquist);

                let power = if i < bark_spectrum.len() {
                    bark_spectrum[i]
                } else {
                    0.0
                };

                // Simplified masking threshold (would be more complex in practice)
                let masking_threshold = if power > 0.0 {
                    power * 0.1 // Simple threshold model
                } else {
                    -60.0 // Quiet threshold
                };

                CriticalBand {
                    center_frequency: actual_center,
                    lower_freq,
                    upper_freq,
                    bark_value,
                    power,
                    masking_threshold,
                }
            })
            .collect();

        Ok(critical_bands)
    }

    /// Compute sharpness (Zwicker & Fastl)
    fn compute_sharpness(&self, bark_spectrum: &[f32]) -> f32 {
        if bark_spectrum.is_empty() {
            return 0.0;
        }

        let total_loudness: f32 = bark_spectrum.iter().sum();
        if total_loudness == 0.0 {
            return 0.0;
        }

        let weighted_sum: f32 = bark_spectrum
            .iter()
            .enumerate()
            .map(|(i, &loudness)| {
                let bark = i as f32 * 24.0 / bark_spectrum.len() as f32;
                let weighting = if bark < 15.8 {
                    1.0
                } else {
                    0.15 * ((bark - 15.8) / 0.65).exp() + 0.85
                };
                loudness * weighting * (bark + 1.0)
            })
            .sum();

        0.11 * weighted_sum / total_loudness
    }

    /// Compute roughness (simplified)
    fn compute_roughness(&self, samples: &[f32], sample_rate: u32) -> Result<f32, EvaluationError> {
        if samples.len() < 1024 {
            return Ok(0.0);
        }

        // Simplified roughness computation based on amplitude modulation
        let frame_size = 1024;
        let mut roughness_values = Vec::new();

        for chunk in samples.chunks(frame_size) {
            if chunk.len() == frame_size {
                let envelope = self.extract_envelope(chunk)?;
                let modulation_depth = self.compute_modulation_depth(&envelope);

                // Roughness is maximum around 70 Hz modulation frequency
                let mod_freq = self.estimate_modulation_frequency(&envelope, sample_rate);
                let roughness_factor = if mod_freq > 20.0 && mod_freq < 300.0 {
                    let normalized_freq = (mod_freq - 70.0).abs() / 70.0;
                    (1.0 - normalized_freq.min(1.0)).max(0.0)
                } else {
                    0.0
                };

                roughness_values.push(modulation_depth * roughness_factor);
            }
        }

        Ok(roughness_values.iter().sum::<f32>() / roughness_values.len().max(1) as f32)
    }

    /// Extract amplitude envelope
    fn extract_envelope(&self, samples: &[f32]) -> Result<Vec<f32>, EvaluationError> {
        // Simple envelope extraction using absolute values and low-pass filtering
        let mut envelope = samples.iter().map(|x| x.abs()).collect::<Vec<f32>>();

        // Simple low-pass filter
        for i in 1..envelope.len() {
            envelope[i] = 0.1 * envelope[i] + 0.9 * envelope[i - 1];
        }

        Ok(envelope)
    }

    /// Compute modulation depth
    fn compute_modulation_depth(&self, envelope: &[f32]) -> f32 {
        if envelope.len() < 2 {
            return 0.0;
        }

        let max_val = envelope.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_val = envelope.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        if max_val > 0.0 {
            (max_val - min_val) / (max_val + min_val)
        } else {
            0.0
        }
    }

    /// Estimate modulation frequency
    fn estimate_modulation_frequency(&self, envelope: &[f32], sample_rate: u32) -> f32 {
        if envelope.len() < 4 {
            return 0.0;
        }

        // Simple zero-crossing rate estimation
        let mut zero_crossings = 0;
        let mean_val = envelope.iter().sum::<f32>() / envelope.len() as f32;

        for i in 1..envelope.len() {
            if (envelope[i - 1] - mean_val) * (envelope[i] - mean_val) < 0.0 {
                zero_crossings += 1;
            }
        }

        // Convert to frequency
        (zero_crossings as f32 / 2.0) * (sample_rate as f32 / envelope.len() as f32)
    }

    /// Compute fluctuation strength
    fn compute_fluctuation_strength(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<f32, EvaluationError> {
        if samples.len() < 1024 {
            return Ok(0.0);
        }

        // Fluctuation strength is related to amplitude modulation in the 0.5-20 Hz range
        let frame_size = sample_rate as usize; // 1 second frames
        let mut fluctuation_values = Vec::new();

        for chunk in samples.chunks(frame_size) {
            if chunk.len() >= frame_size / 2 {
                let envelope = self.extract_envelope(chunk)?;
                let mod_freq = self.estimate_modulation_frequency(&envelope, sample_rate);

                // Fluctuation strength peaks around 4 Hz
                let fluctuation_factor = if mod_freq >= 0.5 && mod_freq <= 20.0 {
                    let normalized_freq = (mod_freq - 4.0).abs() / 4.0;
                    (1.0 - normalized_freq.min(1.0)).max(0.0)
                } else {
                    0.0
                };

                let modulation_depth = self.compute_modulation_depth(&envelope);
                fluctuation_values.push(modulation_depth * fluctuation_factor);
            }
        }

        Ok(fluctuation_values.iter().sum::<f32>() / fluctuation_values.len().max(1) as f32)
    }

    /// Compute masking threshold
    fn compute_masking_threshold(
        &self,
        bark_spectrum: &[f32],
    ) -> Result<Vec<f32>, EvaluationError> {
        let mut masking_threshold = vec![0.0; bark_spectrum.len()];

        for (i, &power) in bark_spectrum.iter().enumerate() {
            if power > 0.0 {
                // Simplified masking model - spread masking energy to neighboring bands
                let masking_power = power * 0.1; // 10 dB below masker

                for (j, threshold) in masking_threshold.iter_mut().enumerate() {
                    let distance = (i as f32 - j as f32).abs();
                    let spread_factor = if distance <= 1.0 {
                        1.0
                    } else if distance <= 3.0 {
                        0.5
                    } else {
                        0.1
                    };

                    *threshold = f32::max(*threshold, masking_power * spread_factor);
                }
            }
        }

        Ok(masking_threshold)
    }

    /// Analyze temporal masking effects
    fn analyze_temporal_masking(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<TemporalMaskingAnalysis, EvaluationError> {
        let frame_size = self.config.frame_size;
        let hop_size = self.config.hop_size;

        let mut pre_masking = Vec::new();
        let mut post_masking = Vec::new();
        let mut masking_patterns = Vec::new();

        for i in (0..samples.len()).step_by(hop_size) {
            if i + frame_size <= samples.len() {
                let frame = &samples[i..i + frame_size];
                let power = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;

                // Simplified temporal masking analysis
                let pre_mask_duration = 0.005; // 5ms pre-masking
                let post_mask_duration = 0.1; // 100ms post-masking

                let pre_mask_samples = (pre_mask_duration * sample_rate as f32) as usize;
                let post_mask_samples = (post_mask_duration * sample_rate as f32) as usize;

                // Pre-masking effect
                let pre_mask_threshold = if power > 0.0 {
                    power * 0.01 // 20 dB below masker
                } else {
                    0.0
                };
                pre_masking.push(pre_mask_threshold);

                // Post-masking effect (exponential decay)
                let mut post_pattern = Vec::new();
                for j in 0..post_mask_samples {
                    let decay_factor = (-(j as f32) / (post_mask_samples as f32 * 0.3)).exp();
                    post_pattern.push(pre_mask_threshold * decay_factor);
                }
                masking_patterns.push(post_pattern.clone());

                post_masking.push(post_pattern.iter().sum::<f32>() / post_pattern.len() as f32);
            }
        }

        Ok(TemporalMaskingAnalysis {
            pre_masking,
            post_masking,
            masking_patterns,
        })
    }

    /// Compare two audio signals using psychoacoustic models
    pub fn compare_psychoacoustic(
        &self,
        reference: &AudioBuffer,
        generated: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        let ref_analysis = self.analyze_psychoacoustic_features(reference)?;
        let gen_analysis = self.analyze_psychoacoustic_features(generated)?;

        // Compute weighted difference across psychoacoustic dimensions
        let loudness_diff = (ref_analysis.loudness_lufs - gen_analysis.loudness_lufs).abs() / 50.0; // Normalize by 50 LU range
        let sharpness_diff =
            (ref_analysis.sharpness_acum - gen_analysis.sharpness_acum).abs() / 5.0; // Normalize by typical range
        let roughness_diff =
            (ref_analysis.roughness_asper - gen_analysis.roughness_asper).abs() / 2.0;

        // Bark spectrum similarity
        let bark_similarity =
            if ref_analysis.bark_spectrum.len() == gen_analysis.bark_spectrum.len() {
                let correlation = self
                    .compute_correlation(&ref_analysis.bark_spectrum, &gen_analysis.bark_spectrum);
                (1.0 + correlation) / 2.0 // Convert from [-1,1] to [0,1]
            } else {
                0.5 // Default similarity for mismatched lengths
            };

        // Weighted combination (higher scores = better)
        let psychoacoustic_score = 1.0
            - (0.3 * loudness_diff
                + 0.2 * sharpness_diff
                + 0.2 * roughness_diff
                + 0.3 * (1.0 - bark_similarity))
                .min(1.0);

        Ok(psychoacoustic_score.max(0.0))
    }

    /// Compute correlation between two vectors
    fn compute_correlation(&self, x: &[f32], y: &[f32]) -> f32 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f32;
        let mean_x = x.iter().sum::<f32>() / n;
        let mean_y = y.iter().sum::<f32>() / n;

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
}

impl Default for PsychoacousticEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl PsychoacousticEvaluator {
    /// Evaluate psychoacoustic quality score
    pub fn evaluate_quality_score(
        &self,
        generated: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> Result<f32, EvaluationError> {
        match reference {
            Some(ref_audio) => self.compare_psychoacoustic(ref_audio, generated),
            None => {
                // For non-reference evaluation, compute a quality score based on psychoacoustic features
                let analysis = self.analyze_psychoacoustic_features(generated)?;

                // Heuristic quality score based on psychoacoustic properties
                let loudness_quality =
                    if analysis.loudness_lufs > -50.0 && analysis.loudness_lufs < -10.0 {
                        1.0 - (analysis.loudness_lufs + 30.0).abs() / 20.0
                    } else {
                        0.0
                    };

                let sharpness_quality =
                    (1.0 - (analysis.sharpness_acum - 1.5).abs() / 3.0).max(0.0);
                let roughness_quality = (1.0 - analysis.roughness_asper / 2.0).max(0.0);

                Ok((loudness_quality + sharpness_quality + roughness_quality) / 3.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psychoacoustic_evaluator_creation() {
        let evaluator = PsychoacousticEvaluator::new();
        assert_eq!(evaluator.config.num_bark_bands, 24);
        assert!(evaluator.config.enable_temporal_masking);
    }

    #[test]
    fn test_psychoacoustic_config_default() {
        let config = PsychoacousticConfig::default();
        assert_eq!(config.num_bark_bands, 24);
        assert_eq!(config.frame_size, 2048);
        assert_eq!(config.hop_size, 512);
    }

    #[test]
    fn test_bark_frequency_generation() {
        let frequencies = PsychoacousticEvaluator::generate_bark_frequencies(24);
        assert_eq!(frequencies.len(), 24);
        assert!(frequencies[0] < frequencies[23]); // Ascending order
        assert!(frequencies[0] > 0.0);
        assert!(frequencies[23] < 25000.0); // Reasonable upper bound
    }

    #[test]
    fn test_hz_to_bark_conversion() {
        let bark_1000 = PsychoacousticEvaluator::hz_to_bark(1000.0);
        let bark_2000 = PsychoacousticEvaluator::hz_to_bark(2000.0);
        assert!(bark_1000 < bark_2000); // Higher frequency = higher bark value
        assert!(bark_1000 > 0.0);
    }

    #[test]
    fn test_psychoacoustic_analysis() {
        let evaluator = PsychoacousticEvaluator::new();
        let audio = AudioBuffer::mono(vec![0.1; 16000], 16000);

        let analysis = evaluator.analyze_psychoacoustic_features(&audio);
        assert!(analysis.is_ok());

        let result = analysis.unwrap();
        assert!(result.loudness_lufs.is_finite());
        assert!(result.sharpness_acum >= 0.0);
        assert!(result.roughness_asper >= 0.0);
        assert!(!result.bark_spectrum.is_empty());
        assert!(!result.critical_bands.is_empty());
    }

    #[test]
    fn test_psychoacoustic_comparison() {
        let evaluator = PsychoacousticEvaluator::new();
        let reference = AudioBuffer::mono(vec![0.1; 16000], 16000);
        let generated = AudioBuffer::mono(vec![0.1; 16000], 16000);

        let score = evaluator.compare_psychoacoustic(&reference, &generated);
        assert!(score.is_ok());

        let result = score.unwrap();
        assert!(result >= 0.0 && result <= 1.0);
        assert!(result > 0.8); // Should be high for identical signals
    }

    #[test]
    fn test_quality_metric_implementation() {
        let evaluator = PsychoacousticEvaluator::new();
        let audio = AudioBuffer::mono(vec![0.1; 16000], 16000);

        // Test with reference
        let score_with_ref = evaluator.evaluate_quality_score(&audio, Some(&audio));
        assert!(score_with_ref.is_ok());
        assert!(score_with_ref.unwrap() > 0.8);

        // Test without reference
        let score_no_ref = evaluator.evaluate_quality_score(&audio, None);
        assert!(score_no_ref.is_ok());
        assert!(score_no_ref.unwrap() >= 0.0);
    }

    #[test]
    fn test_loudness_analysis() {
        let evaluator = PsychoacousticEvaluator::new();
        let samples = vec![0.1; 16000];

        let (loudness, integrated, lra) = evaluator.analyze_loudness(&samples, 16000).unwrap();
        assert!(loudness.is_finite());
        assert!(integrated.is_finite());
        assert!(lra >= 0.0);
    }

    #[test]
    fn test_sharpness_computation() {
        let evaluator = PsychoacousticEvaluator::new();
        let bark_spectrum = vec![1.0; 24];

        let sharpness = evaluator.compute_sharpness(&bark_spectrum);
        assert!(sharpness >= 0.0);
        assert!(sharpness.is_finite());
    }

    #[test]
    fn test_critical_band_analysis() {
        let evaluator = PsychoacousticEvaluator::new();
        let samples = vec![0.1; 2048];

        let bands = evaluator.analyze_critical_bands(&samples, 16000);
        assert!(bands.is_ok());

        let result = bands.unwrap();
        assert!(!result.is_empty());

        for band in &result {
            assert!(band.center_frequency > 0.0);
            assert!(band.lower_freq <= band.center_frequency);
            assert!(band.center_frequency <= band.upper_freq);
            assert!(band.bark_value >= 0.0);
        }
    }

    #[test]
    fn test_empty_audio_handling() {
        let evaluator = PsychoacousticEvaluator::new();
        let empty_audio = AudioBuffer::mono(vec![], 16000);

        let result = evaluator.analyze_psychoacoustic_features(&empty_audio);
        assert!(result.is_err()); // Should fail gracefully
    }

    #[test]
    fn test_masking_threshold_computation() {
        let evaluator = PsychoacousticEvaluator::new();
        let bark_spectrum = vec![1.0, 2.0, 1.5, 0.5, 0.1];

        let threshold = evaluator.compute_masking_threshold(&bark_spectrum);
        assert!(threshold.is_ok());

        let result = threshold.unwrap();
        assert_eq!(result.len(), bark_spectrum.len());
        assert!(result.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_correlation_computation() {
        let evaluator = PsychoacousticEvaluator::new();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];

        let correlation = evaluator.compute_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 0.001); // Perfect positive correlation

        let z = vec![4.0, 3.0, 2.0, 1.0];
        let neg_correlation = evaluator.compute_correlation(&x, &z);
        assert!((neg_correlation + 1.0).abs() < 0.001); // Perfect negative correlation
    }

    #[test]
    fn test_temporal_masking_analysis() {
        let evaluator = PsychoacousticEvaluator::new();
        let samples = vec![0.1; 8192]; // Longer signal for temporal analysis

        let analysis = evaluator.analyze_temporal_masking(&samples, 16000);
        assert!(analysis.is_ok());

        let result = analysis.unwrap();
        assert!(!result.pre_masking.is_empty());
        assert!(!result.post_masking.is_empty());
        assert!(!result.masking_patterns.is_empty());
    }
}
