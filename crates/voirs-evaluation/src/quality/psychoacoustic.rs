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

        // Apply the ITU-R BS.1770-4 K-weighting filter chain.
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
        sorted_loudness.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let percentile_10 = sorted_loudness[(sorted_loudness.len() as f32 * 0.1) as usize];
        let percentile_95 = sorted_loudness[(sorted_loudness.len() as f32 * 0.95) as usize];
        let loudness_range = percentile_95 - percentile_10;

        Ok((integrated_loudness, integrated_loudness, loudness_range))
    }

    /// Apply the ITU-R BS.1770-4 K-weighting filter to `samples`.
    ///
    /// K-weighting is the two-stage IIR pre-filter that precedes the gated
    /// loudness measurement. It consists of:
    ///
    /// * **Stage 1 - "pre-filter" (high-shelf)** modelling the acoustic effect
    ///   of the head: f0 ~ 1681.97 Hz, Q ~ 0.7071, gain ~ +3.999 dB.
    /// * **Stage 2 - RLB high-pass** (revised low-frequency B-weighting):
    ///   f0 ~ 38.135 Hz, Q ~ 0.5.
    ///
    /// The biquad coefficients are derived for the supplied `sample_rate` via the
    /// bilinear transform (RBJ-cookbook high-shelf / high-pass forms), so the
    /// filter is correct at any sample rate rather than only at the 48 kHz
    /// reference. The two sections are applied in series using a transposed
    /// direct-form II structure with `f64` state for numerical accuracy at the
    /// very-low RLB corner frequency.
    fn apply_k_weighting(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<f32>, EvaluationError> {
        if samples.len() < 2 {
            return Ok(samples.to_vec());
        }

        let fs = f64::from(sample_rate);
        let pre_filter = Biquad::k_weighting_pre_filter(fs);
        let rlb_high_pass = Biquad::k_weighting_rlb(fs);

        // Stage 1 (high-shelf) then stage 2 (RLB high-pass) in series.
        let stage1 = pre_filter.filter(samples);
        let stage2 = rlb_high_pass.filter(&stage1);

        Ok(stage2)
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
        let mut planner = self
            .fft_planner
            .lock()
            .expect("lock should not be poisoned");
        let fft = planner.plan_fft_forward(n);

        // Prepare input buffer
        let mut input_buffer = samples.to_vec();

        // Prepare output buffer
        let mut output_buffer = vec![Complex::new(0.0, 0.0); n / 2 + 1];

        // Perform FFT
        fft.process(&input_buffer, &mut output_buffer)
            .map_err(|e| EvaluationError::AudioProcessingError {
                message: e.to_string(),
                source: None,
            })?;

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

/// A second-order IIR section (biquad) used to build the ITU-R BS.1770-4
/// K-weighting cascade.
///
/// Coefficients are stored already normalised by `a0`, so the difference
/// equation is `y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]`.
#[derive(Debug, Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl Biquad {
    /// Stage 1 of the K-weighting filter: the "pre-filter" high-shelf that
    /// models the acoustic effect of the head (ITU-R BS.1770-4 Annex 1).
    ///
    /// Coefficients are derived for `sample_rate` (Hz) from the analogue
    /// prototype (f0 = 1681.9744509555319 Hz, Q = 0.7071752369554193,
    /// gain = 3.999843853973347 dB) via the bilinear transform. At 48 kHz this
    /// reproduces the BS.1770-4 reference coefficients
    /// `b = [1.53512485958697, -2.69169618940638, 1.19839281085285]` and
    /// `a = [1, -1.69065929318241, 0.73248077421585]`.
    fn k_weighting_pre_filter(sample_rate: f64) -> Self {
        let f0 = 1_681.974_450_955_532_f64;
        let q = 0.707_175_236_955_419_3_f64;
        let gain_db = 3.999_843_853_973_347_f64;

        let k = (std::f64::consts::PI * f0 / sample_rate).tan();
        let k2 = k * k;
        // High-frequency shelf gain as a (power) ratio and its companion term.
        let vh = 10.0_f64.powf(gain_db / 20.0);
        let vb = vh.powf(0.499_666_774_154_541_6_f64);

        let denom = 1.0 + k / q + k2;
        Self {
            b0: (vh + vb * k / q + k2) / denom,
            b1: 2.0 * (k2 - vh) / denom,
            b2: (vh - vb * k / q + k2) / denom,
            a1: 2.0 * (k2 - 1.0) / denom,
            a2: (1.0 - k / q + k2) / denom,
        }
    }

    /// Stage 2 of the K-weighting filter: the RLB (revised low-frequency
    /// B-weighting) high-pass (ITU-R BS.1770-4 Annex 1).
    ///
    /// Coefficients are derived for `sample_rate` (Hz) from the analogue
    /// prototype (f0 = 38.13547087613982 Hz, Q = 0.5003270373253953) via the
    /// bilinear transform. At 48 kHz this reproduces the BS.1770-4 reference
    /// coefficients `b = [1, -2, 1]` and
    /// `a = [1, -1.99004745483398, 0.99007225036616]`.
    fn k_weighting_rlb(sample_rate: f64) -> Self {
        let f0 = 38.135_470_876_139_82_f64;
        let q = 0.500_327_037_325_395_3_f64;

        let k = (std::f64::consts::PI * f0 / sample_rate).tan();
        let k2 = k * k;
        let denom = 1.0 + k / q + k2;

        Self {
            b0: 1.0,
            b1: -2.0,
            b2: 1.0,
            a1: 2.0 * (k2 - 1.0) / denom,
            a2: (1.0 - k / q + k2) / denom,
        }
    }

    /// Filter `samples` through this biquad using a transposed direct-form II
    /// structure with `f64` state, returning the `f32` output (same length).
    fn filter(&self, samples: &[f32]) -> Vec<f32> {
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        samples
            .iter()
            .map(|&sample| {
                let x = sample as f64;
                let y = self.b0 * x + s1;
                s1 = self.b1 * x - self.a1 * y + s2;
                s2 = self.b2 * x - self.a2 * y;
                y as f32
            })
            .collect()
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

    // ------- ITU-R BS.1770-4 K-weighting filter tests -------

    /// Steady-state magnitude response (output RMS / input RMS) of the
    /// K-weighting filter for a pure sine at `freq_hz`, measured over the
    /// settled second half of a 1 s signal so the filter transient is ignored.
    fn k_weighting_gain(
        evaluator: &PsychoacousticEvaluator,
        freq_hz: f64,
        sample_rate: u32,
    ) -> f64 {
        let n = sample_rate as usize; // 1 second
        let input: Vec<f32> = (0..n)
            .map(|i| {
                (0.5 * (2.0 * std::f64::consts::PI * freq_hz * i as f64 / f64::from(sample_rate))
                    .sin()) as f32
            })
            .collect();
        let output = evaluator
            .apply_k_weighting(&input, sample_rate)
            .expect("k-weighting must succeed");

        let start = n / 2;
        let rms = |signal: &[f32]| -> f64 {
            let tail = &signal[start..];
            (tail.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / tail.len() as f64).sqrt()
        };
        rms(&output) / rms(&input)
    }

    /// The derived biquad coefficients at 48 kHz must match the published
    /// ITU-R BS.1770-4 reference values within a tight tolerance.
    #[test]
    fn test_k_weighting_coefficients_match_bs1770_reference_48k() {
        let tol = 1e-4;

        let pre = Biquad::k_weighting_pre_filter(48_000.0);
        assert!(
            (pre.b0 - 1.535_124_859_586_97).abs() < tol,
            "pre.b0 = {}",
            pre.b0
        );
        assert!(
            (pre.b1 - (-2.691_696_189_406_38)).abs() < tol,
            "pre.b1 = {}",
            pre.b1
        );
        assert!(
            (pre.b2 - 1.198_392_810_852_85).abs() < tol,
            "pre.b2 = {}",
            pre.b2
        );
        assert!(
            (pre.a1 - (-1.690_659_293_182_41)).abs() < tol,
            "pre.a1 = {}",
            pre.a1
        );
        assert!(
            (pre.a2 - 0.732_480_774_215_85).abs() < tol,
            "pre.a2 = {}",
            pre.a2
        );

        let rlb = Biquad::k_weighting_rlb(48_000.0);
        assert!((rlb.b0 - 1.0).abs() < 1e-12, "rlb.b0 = {}", rlb.b0);
        assert!((rlb.b1 - (-2.0)).abs() < 1e-12, "rlb.b1 = {}", rlb.b1);
        assert!((rlb.b2 - 1.0).abs() < 1e-12, "rlb.b2 = {}", rlb.b2);
        assert!(
            (rlb.a1 - (-1.990_047_454_833_98)).abs() < tol,
            "rlb.a1 = {}",
            rlb.a1
        );
        assert!(
            (rlb.a2 - 0.990_072_250_366_16).abs() < tol,
            "rlb.a2 = {}",
            rlb.a2
        );
    }

    /// The RLB high-pass has a transmission zero at DC, so a constant input must
    /// be reduced to essentially zero once the transient settles.
    #[test]
    fn test_k_weighting_strongly_attenuates_dc() {
        let evaluator = PsychoacousticEvaluator::new();
        let fs = 48_000u32;
        let dc = vec![0.5f32; fs as usize]; // 1 second of DC
        let filtered = evaluator
            .apply_k_weighting(&dc, fs)
            .expect("k-weighting must succeed");

        let tail = &filtered[filtered.len() / 2..];
        let residual_rms =
            (tail.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / tail.len() as f64).sqrt();
        assert!(
            residual_rms < 1e-4,
            "DC must be strongly attenuated, residual rms = {residual_rms}"
        );
    }

    /// A ~2 kHz tone passes with the high-shelf boost (gain > 1), while low
    /// frequencies are attenuated by the RLB high-pass: 100 Hz sits below the
    /// 2 kHz level (and below unity) and 20 Hz (below the 38 Hz corner) is
    /// attenuated even more strongly.
    #[test]
    fn test_k_weighting_low_freq_attenuated_relative_to_2khz() {
        let evaluator = PsychoacousticEvaluator::new();
        let fs = 48_000u32;

        let gain_20 = k_weighting_gain(&evaluator, 20.0, fs);
        let gain_100 = k_weighting_gain(&evaluator, 100.0, fs);
        let gain_2k = k_weighting_gain(&evaluator, 2_000.0, fs);

        assert!(
            gain_2k > 1.0,
            "2 kHz gain {gain_2k} should exceed unity (shelf boost)"
        );
        assert!(
            gain_100 < 1.0,
            "100 Hz gain {gain_100} should be < 1 (attenuated)"
        );
        assert!(
            gain_100 < gain_2k,
            "100 Hz gain {gain_100} should be below 2 kHz gain {gain_2k}"
        );
        assert!(
            gain_20 < 0.3,
            "20 Hz gain {gain_20} should be strongly attenuated"
        );
        assert!(
            gain_20 < gain_100,
            "20 Hz gain {gain_20} should be attenuated more than 100 Hz gain {gain_100}"
        );
    }

    /// The coefficient derivation must adapt to the sample rate: the same
    /// physical corner frequencies imply different normalised tap values at
    /// 44.1 kHz than at 48 kHz, yet a 2 kHz tone is still boosted and DC removed.
    #[test]
    fn test_k_weighting_adapts_to_sample_rate() {
        let pre_48 = Biquad::k_weighting_pre_filter(48_000.0);
        let pre_44 = Biquad::k_weighting_pre_filter(44_100.0);
        assert!(
            (pre_48.a1 - pre_44.a1).abs() > 1e-4,
            "coefficients must differ across sample rates"
        );

        let evaluator = PsychoacousticEvaluator::new();
        let gain_2k = k_weighting_gain(&evaluator, 2_000.0, 44_100);
        let gain_dc_like = k_weighting_gain(&evaluator, 5.0, 44_100);
        assert!(
            gain_2k > 1.0,
            "2 kHz gain {gain_2k} should exceed unity at 44.1 kHz"
        );
        assert!(
            gain_dc_like < gain_2k,
            "near-DC gain {gain_dc_like} should be far below 2 kHz gain {gain_2k}"
        );
    }
}
