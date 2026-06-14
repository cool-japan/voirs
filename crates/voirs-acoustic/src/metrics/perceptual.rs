//! Perceptual audio quality metrics
//!
//! This module provides perceptual metrics for evaluating TTS synthesis quality,
//! including PESQ, STOI, SI-SDR, and other metrics that correlate with human
//! perception of audio quality.

use crate::{AcousticError, Result};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Perceptual quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualMetrics {
    /// PESQ score (1.0-4.5, higher is better)
    pub pesq_score: f32,
    /// STOI score (0.0-1.0, higher is better)
    pub stoi_score: f32,
    /// SI-SDR score in dB (higher is better)
    pub si_sdr: Option<f32>,
    /// Overall perceptual score (0-100)
    pub overall_score: f32,
}

impl Default for PerceptualMetrics {
    fn default() -> Self {
        Self {
            pesq_score: 0.0,
            stoi_score: 0.0,
            si_sdr: None,
            overall_score: 0.0,
        }
    }
}

/// Perceptual quality evaluator
pub struct PerceptualEvaluator {
    /// Sample rate for audio processing
    sample_rate: u32,
    /// Frame size for STOI computation
    stoi_frame_size: usize,
    /// Overlap for STOI frames
    stoi_overlap: usize,
    /// Third-octave band filters for STOI
    stoi_bands: Vec<StoiBand>,
}

/// STOI frequency band information
#[derive(Debug, Clone)]
struct StoiBand {
    /// Center frequency of the band
    #[allow(dead_code)]
    center_freq: f32,
    /// Lower cutoff frequency
    low_freq: f32,
    /// Upper cutoff frequency
    high_freq: f32,
    /// Band weight
    #[allow(dead_code)]
    weight: f32,
}

impl Default for PerceptualEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl PerceptualEvaluator {
    /// Create new perceptual evaluator with default parameters
    pub fn new() -> Self {
        let sample_rate = 16000; // Standard for PESQ/STOI
        let stoi_frame_size = 256;
        let stoi_overlap = 128;
        let stoi_bands = Self::create_stoi_bands(sample_rate);

        Self {
            sample_rate,
            stoi_frame_size,
            stoi_overlap,
            stoi_bands,
        }
    }

    /// Create evaluator with custom sample rate
    pub fn with_sample_rate(sample_rate: u32) -> Self {
        let stoi_frame_size = 256;
        let stoi_overlap = 128;
        let stoi_bands = Self::create_stoi_bands(sample_rate);

        Self {
            sample_rate,
            stoi_frame_size,
            stoi_overlap,
            stoi_bands,
        }
    }

    /// Compute PESQ score (Perceptual Evaluation of Speech Quality)
    pub fn compute_pesq(&self, degraded: &[f32], reference: &[f32]) -> Result<f32> {
        if degraded.is_empty() || reference.is_empty() {
            return Err(AcousticError::InputError {
                message: "Empty audio samples".to_string(),
            });
        }

        // Simplified PESQ implementation
        // Real PESQ requires complex psychoacoustic modeling
        let pesq_score = self.compute_simplified_pesq(degraded, reference)?;

        // PESQ score range is 1.0 to 4.5
        Ok(pesq_score.clamp(1.0, 4.5))
    }

    /// Compute STOI score (Short-Time Objective Intelligibility)
    pub fn compute_stoi(&self, degraded: &[f32], reference: &[f32]) -> Result<f32> {
        if degraded.is_empty() || reference.is_empty() {
            return Err(AcousticError::InputError {
                message: "Empty audio samples".to_string(),
            });
        }

        // Align signals by length
        let min_len = degraded.len().min(reference.len());
        let degraded = &degraded[..min_len];
        let reference = &reference[..min_len];

        let stoi_score = self.compute_stoi_core(degraded, reference)?;

        // STOI score range is 0.0 to 1.0
        Ok(stoi_score.clamp(0.0, 1.0))
    }

    /// Compute SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
    pub fn compute_si_sdr(&self, estimated: &[f32], target: &[f32]) -> Result<f32> {
        if estimated.is_empty() || target.is_empty() {
            return Err(AcousticError::InputError {
                message: "Empty audio samples".to_string(),
            });
        }

        let min_len = estimated.len().min(target.len());
        let estimated = &estimated[..min_len];
        let target = &target[..min_len];

        // Compute the optimal scaling factor
        let alpha =
            self.compute_dot_product(estimated, target) / self.compute_dot_product(target, target);

        // Compute scaled target
        let scaled_target: Vec<f32> = target.iter().map(|&x| alpha * x).collect();

        // Compute signal and noise powers
        let signal_power = self.compute_signal_power(&scaled_target);
        let noise_power = self.compute_noise_power(estimated, &scaled_target);

        if noise_power <= 0.0 {
            return Ok(60.0); // Very high SDR for perfect match
        }

        let si_sdr = 10.0 * (signal_power / noise_power).log10();

        // Reasonable range for SI-SDR
        Ok(si_sdr.clamp(-20.0, 60.0))
    }

    /// Compute intrinsic quality score without reference
    pub fn compute_intrinsic_quality(&self, audio: &[f32]) -> Result<f32> {
        if audio.is_empty() {
            return Err(AcousticError::InputError {
                message: "Empty audio samples".to_string(),
            });
        }

        // Compute various intrinsic quality indicators
        let snr = self.compute_intrinsic_snr(audio)?;
        let spectral_quality = self.compute_spectral_quality(audio)?;
        let temporal_quality = self.compute_temporal_quality(audio)?;

        // Combine into overall quality score (PESQ-like scale)
        let quality = (snr * 0.4 + spectral_quality * 0.4 + temporal_quality * 0.2).clamp(1.0, 4.5);

        Ok(quality)
    }

    /// Compute perceptual loudness
    pub fn compute_loudness(&self, audio: &[f32]) -> Result<f32> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        // Apply A-weighting filter (simplified)
        let weighted_audio = self.apply_a_weighting(audio)?;

        // Compute RMS with perceptual weighting
        let rms = self.compute_rms(&weighted_audio);

        // Convert to loudness units (simplified)
        let loudness = 20.0 * (rms + 1e-10).log10();

        Ok(loudness)
    }

    /// Compute bark-scale spectral distortion
    pub fn compute_bark_spectral_distortion(
        &self,
        degraded: &[f32],
        reference: &[f32],
    ) -> Result<f32> {
        let min_len = degraded.len().min(reference.len());
        let degraded = &degraded[..min_len];
        let reference = &reference[..min_len];

        // Convert to bark scale representation
        let degraded_bark = self.convert_to_bark_scale(degraded)?;
        let reference_bark = self.convert_to_bark_scale(reference)?;

        // Compute distortion in bark domain
        let mut total_distortion = 0.0f32;
        let bark_bands = degraded_bark.len().min(reference_bark.len());

        for i in 0..bark_bands {
            let diff = degraded_bark[i] - reference_bark[i];
            total_distortion += diff * diff;
        }

        Ok((total_distortion / bark_bands as f32).sqrt())
    }

    // Private helper methods

    /// Estimate a PESQ-like MOS-LQO score from a reference / degraded pair.
    ///
    /// This is **not** a bit-exact ITU-T P.862 implementation (that standard is
    /// large and licensed); it is a defensible perceptual approximation built
    /// around the same core idea as PESQ: a per-frame, per-critical-band
    /// comparison of the reference and degraded loudness spectra.
    ///
    /// Pipeline:
    /// 1. Level-align the degraded signal to the reference RMS (PESQ normalizes
    ///    to a fixed listening level, so the score is largely gain-invariant).
    /// 2. Compute a frame-wise Bark-band log-spectral disturbance
    ///    ([`Self::compute_bark_log_spectral_disturbance`]): an
    ///    articulation-weighted mean of `|ΔdB|` per critical band, aggregated
    ///    across frames with an `L2` (frame-RMS) norm so the worst frames
    ///    dominate.
    /// 3. Map the disturbance (dB) through a monotonic sigmoid to a MOS-LQO in
    ///    `[1.0, 4.5]`.
    /// 4. Lightly modulate the score with the waveform (temporal) correlation
    ///    and the loudness match, which catch gross misalignment / dropouts that
    ///    a magnitude-only spectral measure cannot see.
    fn compute_simplified_pesq(&self, degraded: &[f32], reference: &[f32]) -> Result<f32> {
        let min_len = degraded.len().min(reference.len());
        if min_len == 0 {
            return Ok(1.0);
        }
        let degraded = &degraded[..min_len];
        let reference = &reference[..min_len];

        // (1) Level alignment: scale the degraded signal so its overall RMS
        // matches the reference, making the spectral comparison gain-invariant.
        let ref_rms = self.compute_rms(reference);
        let deg_rms = self.compute_rms(degraded);
        let aligned: Vec<f32> = if deg_rms > 1e-8 {
            let gain = ref_rms / deg_rms;
            degraded.iter().map(|&x| x * gain).collect()
        } else {
            degraded.to_vec()
        };

        // (2)+(3) Core perceptual term: Bark-band log-spectral disturbance
        // mapped to a MOS-LQO-like value.
        let disturbance = self.compute_bark_log_spectral_disturbance(reference, &aligned)?;
        const D_HALF: f32 = 6.0; // dB of disturbance at which the term halves
        const SLOPE: f32 = 1.6;
        let mos_spectral = 1.0 + 3.5 / (1.0 + (disturbance / D_HALF).powf(SLOPE));

        // (4) Secondary cues: temporal correlation and loudness match. Both lie
        // in [0, 1] and together scale the head-room above the 1.0 floor by
        // 0.7..=1.0, so a spectrally-plausible but time-warped or wrongly-loud
        // signal is still penalized.
        let temporal_sim = self.compute_temporal_similarity(degraded, reference)?;
        let loudness_sim = self.compute_loudness_similarity(degraded, reference)?;
        let modulation = 0.7 + 0.2 * temporal_sim + 0.1 * loudness_sim;

        let pesq = 1.0 + (mos_spectral - 1.0) * modulation;
        Ok(pesq)
    }

    /// Frame-wise Bark-band log-spectral disturbance between a reference and a
    /// degraded signal, in decibels (`0.0` for identical inputs).
    ///
    /// For each overlapping, Hann-windowed frame the magnitude spectrum is
    /// obtained with [`scirs2_fft::rfft`], its power is integrated into Bark
    /// critical bands (Traunmüller's Hz→Bark map), converted to dB, and the
    /// absolute reference/degraded dB difference per band is combined with an
    /// articulation-index-inspired log-normal frequency weighting centered near
    /// 1.8 kHz. Frame disturbances are aggregated with an `L2` norm so that
    /// loud, badly-distorted frames dominate the result.
    fn compute_bark_log_spectral_disturbance(
        &self,
        reference: &[f32],
        degraded: &[f32],
    ) -> Result<f32> {
        const N_FFT: usize = 512;
        const HOP: usize = 256;
        const POWER_FLOOR: f64 = 1e-7;

        let len = reference.len().min(degraded.len());
        if len == 0 {
            return Ok(0.0);
        }

        let nyquist = self.sample_rate as f32 / 2.0;
        let n_bark = (hz_to_bark(nyquist as f64).ceil() as usize).max(1);

        // Articulation-style log-normal weight per Bark band (center 1.8 kHz).
        let band_weights: Vec<f32> = (0..n_bark)
            .map(|b| {
                let center_hz = bark_to_hz(b as f64 + 0.5).max(1.0);
                let z = (center_hz.ln() - 1800.0_f64.ln()) / 1.2;
                (-0.5 * z * z).exp() as f32
            })
            .collect();
        let weight_sum: f32 = band_weights.iter().sum::<f32>().max(f32::EPSILON);

        // Periodic Hann window for the analysis frames.
        let window: Vec<f64> = (0..N_FFT)
            .map(|i| {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64;
                0.5 * (1.0 - phase.cos())
            })
            .collect();

        let n_frames = if len >= N_FFT {
            (len - N_FFT) / HOP + 1
        } else {
            1
        };
        let n_freqs = N_FFT / 2 + 1;

        let mut frame_disturbances: Vec<f32> = Vec::with_capacity(n_frames);
        let mut ref_buf = vec![0.0_f64; N_FFT];
        let mut deg_buf = vec![0.0_f64; N_FFT];

        for frame_idx in 0..n_frames {
            let start = frame_idx * HOP;

            // Window the frame (zero-padding past the end of the signal).
            for (i, ((rb, db), &win)) in ref_buf
                .iter_mut()
                .zip(deg_buf.iter_mut())
                .zip(window.iter())
                .enumerate()
            {
                let idx = start + i;
                let (r, d) = if idx < len {
                    (reference[idx] as f64, degraded[idx] as f64)
                } else {
                    (0.0, 0.0)
                };
                *rb = r * win;
                *db = d * win;
            }

            let ref_spec = scirs2_fft::rfft(&ref_buf, Some(N_FFT)).map_err(|e| {
                AcousticError::ProcessingError {
                    message: format!("rfft failed on reference frame {frame_idx}: {e:?}"),
                }
            })?;
            let deg_spec = scirs2_fft::rfft(&deg_buf, Some(N_FFT)).map_err(|e| {
                AcousticError::ProcessingError {
                    message: format!("rfft failed on degraded frame {frame_idx}: {e:?}"),
                }
            })?;

            // Integrate per-bin power into Bark critical bands.
            let mut ref_bands = vec![0.0_f64; n_bark];
            let mut deg_bands = vec![0.0_f64; n_bark];
            for (k, (rc, dc)) in ref_spec
                .iter()
                .zip(deg_spec.iter())
                .enumerate()
                .take(n_freqs)
            {
                let f_hz = k as f64 * self.sample_rate as f64 / N_FFT as f64;
                let band = (hz_to_bark(f_hz).floor().max(0.0) as usize).min(n_bark - 1);
                ref_bands[band] += rc.re * rc.re + rc.im * rc.im;
                deg_bands[band] += dc.re * dc.re + dc.im * dc.im;
            }

            // Weighted mean |ΔdB| across critical bands.
            let mut weighted = 0.0f32;
            for ((&rp, &dp), &w) in ref_bands
                .iter()
                .zip(deg_bands.iter())
                .zip(band_weights.iter())
            {
                let l_ref = 10.0 * (rp + POWER_FLOOR).log10();
                let l_deg = 10.0 * (dp + POWER_FLOOR).log10();
                weighted += w * (l_ref - l_deg).abs() as f32;
            }
            frame_disturbances.push(weighted / weight_sum);
        }

        if frame_disturbances.is_empty() {
            return Ok(0.0);
        }

        // L2 (frame-RMS) aggregation emphasizes the worst frames.
        let sum_sq: f32 = frame_disturbances.iter().map(|&d| d * d).sum();
        Ok((sum_sq / frame_disturbances.len() as f32).sqrt())
    }

    fn compute_stoi_core(&self, degraded: &[f32], reference: &[f32]) -> Result<f32> {
        let mut correlations = Vec::new();

        // Process in overlapping frames
        let mut frame_start = 0;
        while frame_start + self.stoi_frame_size <= degraded.len() {
            let deg_frame = &degraded[frame_start..frame_start + self.stoi_frame_size];
            let ref_frame = &reference[frame_start..frame_start + self.stoi_frame_size];

            // Apply filterbank to both frames
            let deg_bands = self.apply_stoi_filterbank(deg_frame)?;
            let ref_bands = self.apply_stoi_filterbank(ref_frame)?;

            // Compute correlation for each band
            for (deg_band, ref_band) in deg_bands.iter().zip(ref_bands.iter()) {
                let correlation = self.compute_correlation(deg_band, ref_band);
                correlations.push(correlation);
            }

            frame_start += self.stoi_frame_size - self.stoi_overlap;
        }

        // Average correlations
        if correlations.is_empty() {
            Ok(0.0)
        } else {
            Ok(correlations.iter().sum::<f32>() / correlations.len() as f32)
        }
    }

    fn apply_stoi_filterbank(&self, signal: &[f32]) -> Result<Vec<Vec<f32>>> {
        let mut band_outputs = Vec::new();

        for band in &self.stoi_bands {
            let filtered = self.apply_bandpass_filter(signal, band.low_freq, band.high_freq)?;
            band_outputs.push(filtered);
        }

        Ok(band_outputs)
    }

    /// Apply a 2nd-order (biquad) band-pass filter to `signal`.
    ///
    /// # Filter design
    ///
    /// The coefficients are derived with the RBJ *Audio-EQ-Cookbook* band-pass
    /// formulas (the **constant 0 dB peak gain** variant). The band is given by
    /// its lower and upper -3 dB cutoff frequencies; from those we recover:
    ///
    /// * center frequency `f0 = sqrt(low * high)` — the geometric mean is the
    ///   natural center of a constant-`Q` band-pass: the point of unity gain and
    ///   the axis of symmetry of the magnitude response on a log-frequency axis;
    /// * bandwidth `bw = high - low` (Hz), giving the quality factor
    ///   `Q = f0 / bw`.
    ///
    /// The cookbook coefficients (before normalization by `a0`) are
    ///
    /// ```text
    /// w0    = 2*pi*f0 / sample_rate
    /// alpha = sin(w0) / (2*Q)
    /// b0 =  alpha       b1 = 0            b2 = -alpha
    /// a0 =  1 + alpha   a1 = -2*cos(w0)   a2 =  1 - alpha
    /// ```
    ///
    /// They are normalized by `a0` and the difference equation is evaluated with
    /// a **direct-form-II transposed** structure, which has good round-off
    /// behavior for audio-rate IIR filtering. The resulting response has exactly
    /// unity gain at `f0`, zero gain at DC and Nyquist, and a 6 dB/octave
    /// roll-off on either side of the pass-band.
    fn apply_bandpass_filter(
        &self,
        signal: &[f32],
        low_freq: f32,
        high_freq: f32,
    ) -> Result<Vec<f32>> {
        let n = signal.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let nyquist = self.sample_rate as f32 / 2.0;
        let low = low_freq.clamp(0.0, nyquist);
        let high = high_freq.clamp(0.0, nyquist);

        // Degenerate band (empty or inverted): nothing can pass through it.
        if high <= low {
            return Ok(vec![0.0; n]);
        }

        let bandwidth = high - low;
        // Geometric mean for the center; fall back to the arithmetic mean only
        // when the lower edge collapses to DC (where the geometric mean is 0).
        let f_center = if low > 0.0 {
            (low * high).sqrt()
        } else {
            0.5 * (low + high)
        };

        // A center at DC or Nyquist has no valid band-pass; emit a silent band
        // rather than producing NaNs from the coefficient formulas.
        if f_center <= 0.0 || f_center >= nyquist {
            return Ok(vec![0.0; n]);
        }

        // RBJ Audio-EQ-Cookbook band-pass (constant 0 dB peak gain).
        let w0 = 2.0 * PI * f_center / self.sample_rate as f32;
        let q = f_center / bandwidth;
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q);

        let a0 = 1.0 + alpha;
        let b0 = alpha / a0;
        // b1 is exactly 0 for the band-pass and is omitted from the recurrence.
        let b2 = -alpha / a0;
        let a1 = (-2.0 * cos_w0) / a0;
        let a2 = (1.0 - alpha) / a0;

        // Direct-form-II transposed evaluation of the biquad.
        let mut filtered = Vec::with_capacity(n);
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        for &x in signal {
            let y = b0 * x + s1;
            s1 = s2 - a1 * y; // the (b1 * x) term is zero for a band-pass
            s2 = b2 * x - a2 * y;
            filtered.push(y);
        }

        Ok(filtered)
    }

    fn compute_correlation(&self, signal1: &[f32], signal2: &[f32]) -> f32 {
        if signal1.len() != signal2.len() || signal1.is_empty() {
            return 0.0;
        }

        let mean1 = signal1.iter().sum::<f32>() / signal1.len() as f32;
        let mean2 = signal2.iter().sum::<f32>() / signal2.len() as f32;

        let mut numerator = 0.0f32;
        let mut denom1 = 0.0f32;
        let mut denom2 = 0.0f32;

        for (&x, &y) in signal1.iter().zip(signal2.iter()) {
            let x_centered = x - mean1;
            let y_centered = y - mean2;

            numerator += x_centered * y_centered;
            denom1 += x_centered * x_centered;
            denom2 += y_centered * y_centered;
        }

        let denominator = (denom1 * denom2).sqrt();
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    fn compute_dot_product(&self, signal1: &[f32], signal2: &[f32]) -> f32 {
        signal1
            .iter()
            .zip(signal2.iter())
            .map(|(&x, &y)| x * y)
            .sum()
    }

    fn compute_signal_power(&self, signal: &[f32]) -> f32 {
        signal.iter().map(|&x| x * x).sum::<f32>() / signal.len() as f32
    }

    fn compute_noise_power(&self, estimated: &[f32], target: &[f32]) -> f32 {
        let noise: Vec<f32> = estimated
            .iter()
            .zip(target.iter())
            .map(|(&e, &t)| e - t)
            .collect();
        self.compute_signal_power(&noise)
    }

    fn compute_intrinsic_snr(&self, audio: &[f32]) -> Result<f32> {
        // Estimate noise level from quiet segments
        let rms = self.compute_rms(audio);
        let peak = audio.iter().fold(0.0f32, |max, &val| max.max(val.abs()));

        if rms > 0.0 {
            let snr = 20.0 * (peak / rms).log10();
            Ok(snr.clamp(0.0, 4.5))
        } else {
            Ok(2.5)
        }
    }

    fn compute_spectral_quality(&self, audio: &[f32]) -> Result<f32> {
        // Analyze spectral characteristics
        let spectral_centroid = self.compute_spectral_centroid_simple(audio);
        let spectral_spread = self.compute_spectral_spread_simple(audio);

        // Quality based on spectral characteristics
        let centroid_quality = (spectral_centroid / (self.sample_rate as f32 / 4.0)).min(1.0);
        let spread_quality = (1.0 - spectral_spread / (self.sample_rate as f32 / 2.0)).max(0.0);

        Ok((centroid_quality + spread_quality) * 2.25 + 1.0) // Scale to 1-4.5
    }

    fn compute_temporal_quality(&self, audio: &[f32]) -> Result<f32> {
        // Analyze temporal characteristics
        let zero_crossing_rate = self.compute_zero_crossing_rate(audio);
        let short_time_energy_var = self.compute_short_time_energy_variance(audio);

        // Quality based on temporal stability
        let zcr_quality = (1.0 - zero_crossing_rate / 0.5).clamp(0.0, 1.0);
        let energy_quality = (1.0 - short_time_energy_var / 10.0).clamp(0.0, 1.0);

        Ok((zcr_quality + energy_quality) * 1.75 + 1.0) // Scale to 1-4.5
    }

    fn compute_rms(&self, signal: &[f32]) -> f32 {
        if signal.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = signal.iter().map(|&x| x * x).sum();
        (sum_squares / signal.len() as f32).sqrt()
    }

    /// Apply the IEC 61672-1 (IEC 61672:2013) **A-weighting** frequency curve.
    ///
    /// # Standard
    ///
    /// A-weighting models the relative loudness perceived by the human ear and
    /// is defined in IEC 61672-1 by an analog transfer function with four real
    /// pole pairs and a double zero at the origin (12 dB/octave high-pass at the
    /// low end). The pole frequencies are the standardized values
    ///
    /// ```text
    /// f1 =    20.598997 Hz   (double pole)
    /// f2 =   107.65265  Hz
    /// f3 =   737.86223  Hz
    /// f4 = 12194.217    Hz   (double pole)
    /// ```
    ///
    /// and the response is normalized so that the gain is exactly 0 dB at
    /// 1 kHz (`A1000 ≈ +2.0 dB`, applied here as a linear scaling). The
    /// continuous-time transfer function is
    ///
    /// ```text
    ///           K * s^4
    /// H(s) = ------------------------------------------------
    ///        (s + w1)^2 (s + w2)(s + w3)(s + w4)^2
    /// ```
    ///
    /// with `wi = 2*pi*fi` and `K` chosen so `|H(j*2*pi*1000)| = 1`.
    ///
    /// # Digital realization
    ///
    /// `H(s)` is converted to a digital filter at the evaluator's sample rate
    /// with the **bilinear transform** (`s -> (2/T) (1 - z^-1)/(1 + z^-1)`,
    /// `T = 1/sample_rate`). Because the analog prototype is a product of three
    /// second-order sections — `s^2 / (s + w1)^2`, `s / ((s+w2)(s+w3))` scaled
    /// appropriately, and `s / (s + w4)^2` — the digital result is a cascade of
    /// three biquads. Each biquad is applied with the same direct-form-II
    /// transposed structure used by [`Self::apply_bandpass_filter`].
    fn apply_a_weighting(&self, audio: &[f32]) -> Result<Vec<f32>> {
        if audio.len() < 2 {
            return Ok(audio.to_vec());
        }

        let biquads = self.a_weighting_biquads();
        let mut signal: Vec<f64> = audio.iter().map(|&x| x as f64).collect();

        // Apply each biquad section in cascade (direct-form-II transposed).
        for bq in &biquads {
            let mut s1 = 0.0f64;
            let mut s2 = 0.0f64;
            for sample in signal.iter_mut() {
                let x = *sample;
                let y = bq.b0 * x + s1;
                s1 = bq.b1 * x - bq.a1 * y + s2;
                s2 = bq.b2 * x - bq.a2 * y;
                *sample = y;
            }
        }

        Ok(signal.into_iter().map(|x| x as f32).collect())
    }

    /// Build the three-biquad cascade implementing the IEC 61672-1 A-weighting
    /// curve at the evaluator's current sample rate via the bilinear transform.
    ///
    /// Coefficients follow the canonical biquad convention `b0 + b1 z^-1 +
    /// b2 z^-2` over `1 + a1 z^-1 + a2 z^-2` (i.e. `a0` normalized to 1).
    fn a_weighting_biquads(&self) -> [Biquad; 3] {
        // Standardized A-weighting pole frequencies (Hz).
        const F1: f64 = 20.598997;
        const F2: f64 = 107.65265;
        const F3: f64 = 737.86223;
        const F4: f64 = 12194.217;
        // A1000: gain (dB) of the *un-normalized* analog response at 1 kHz,
        // used to renormalize the curve to 0 dB at 1 kHz.
        const A1000_DB: f64 = 1.9997;

        let fs = self.sample_rate as f64;
        let t = 1.0 / fs;
        // Bilinear pre-factor c = 2/T = 2 * fs.
        let c = 2.0 * fs;

        let w1 = 2.0 * std::f64::consts::PI * F1;
        let w2 = 2.0 * std::f64::consts::PI * F2;
        let w3 = 2.0 * std::f64::consts::PI * F3;
        let w4 = 2.0 * std::f64::consts::PI * F4;

        // --- Section A: s^2 / (s + w1)^2 (double pole at w1, double zero at 0).
        // Bilinear: s = c (1 - z^-1)/(1 + z^-1).
        //   s^2            -> c^2 (1 - z^-1)^2
        //   (s + w1)^2     -> (c + w1)^2 + 2(w1^2 - c^2) z^-1 + (c - w1)^2 z^-2
        let a0_1 = (c + w1) * (c + w1);
        let biquad1 = Biquad::normalized(
            c * c,
            -2.0 * c * c,
            c * c,
            a0_1,
            2.0 * (w1 * w1 - c * c),
            (c - w1) * (c - w1),
        );

        // --- Section B: s^2 / ((s + w2)(s + w3)) — combine the two simple poles
        // (w2, w3) with two of the four zeros at the origin.
        //   s^2          -> c^2 (1 - z^-1)^2
        //   (s+w2)(s+w3) -> (c+w2)(c+w3)
        //                   + [2(w2 w3 - c^2)] z^-1
        //                   + (c-w2)(c-w3) z^-2
        let a0_2 = (c + w2) * (c + w3);
        let biquad2 = Biquad::normalized(
            c * c,
            -2.0 * c * c,
            c * c,
            a0_2,
            2.0 * (w2 * w3 - c * c),
            (c - w2) * (c - w3),
        );

        // --- Section C: 1 / (s + w4)^2 (double pole at w4, no further zeros).
        //   numerator constant 1 -> (1 + z^-1)^2 after clearing the bilinear
        //   denominators, i.e. b = {1, 2, 1}.
        //   (s + w4)^2 -> (c + w4)^2 + 2(w4^2 - c^2) z^-1 + (c - w4)^2 z^-2
        let a0_3 = (c + w4) * (c + w4);
        let mut biquad3 = Biquad::normalized(
            1.0,
            2.0,
            1.0,
            a0_3,
            2.0 * (w4 * w4 - c * c),
            (c - w4) * (c - w4),
        );

        // Normalize the cascade to 0 dB at 1 kHz. The IEC 61672-1 A1000 (≈ +2 dB
        // analog gain) plus the residual bilinear numerator scale fold into a
        // single linear factor. Computing it from the *realized* digital cascade
        // response — rather than algebraically pre-folding `c^2` and
        // `10^(A1000/20)` — is exact at every sample rate and immune to the
        // numerator-scaling subtleties of the bilinear transform.
        let mag_1k = biquad1.magnitude_at(1000.0, fs)
            * biquad2.magnitude_at(1000.0, fs)
            * biquad3.magnitude_at(1000.0, fs);
        debug_assert!((A1000_DB - 1.9997).abs() < 1e-6); // documents the target curve
        let gain = if mag_1k > f64::EPSILON {
            1.0 / mag_1k
        } else {
            1.0
        };
        biquad3.b0 *= gain;
        biquad3.b1 *= gain;
        biquad3.b2 *= gain;

        let _ = t; // T is captured via c = 2/T; kept for documentation clarity.
        [biquad1, biquad2, biquad3]
    }

    /// Project the magnitude spectrum of `signal` onto the **Bark critical-band
    /// scale** (24 bands spanning 0–24 Bark).
    ///
    /// # Standard
    ///
    /// Frequencies are warped to Bark with Traunmüller's (1990) analytic
    /// approximation including the standard low/high-frequency corrections (see
    /// [`hz_to_bark`]). Each of the 24 critical bands is realized as a
    /// **triangular weighting** on the Bark axis: band `b` is centered on Bark
    /// `b + 0.5` with unity weight at the center, falling linearly to zero at the
    /// neighbouring band centers (Bark `b - 0.5` and `b + 1.5`). The power
    /// spectrum is integrated through these overlapping triangles — a standard
    /// critical-band / auditory-filterbank construction — and the per-band RMS
    /// (`sqrt` of weighted mean power) is returned. The output length (24) and
    /// element type (`f32`) match the historical interface.
    fn convert_to_bark_scale(&self, signal: &[f32]) -> Result<Vec<f32>> {
        const BARK_BANDS: usize = 24; // Standard number of critical bands.
        let mut bark_spectrum = vec![0.0f32; BARK_BANDS];

        if signal.is_empty() {
            return Ok(bark_spectrum);
        }

        // One-sided magnitude spectrum (n/2+1 bins) via the real FFT.
        let magnitudes = self.compute_magnitude_spectrum(signal)?;
        if magnitudes.is_empty() {
            return Ok(bark_spectrum);
        }

        // The magnitude spectrum is one-sided over [0, Nyquist]; bin k maps to
        // frequency f_k = k * sample_rate / n, where n is the FFT length used by
        // compute_magnitude_spectrum (n = signal.len(), giving n/2+1 bins).
        let n_fft = signal.len();
        let bin_hz = self.sample_rate as f64 / n_fft as f64;

        // Accumulate triangular-weighted power and the weight mass per band so
        // each band reports a properly normalized (weighted-mean) power.
        let mut weighted_power = vec![0.0f64; BARK_BANDS];
        let mut weight_mass = vec![0.0f64; BARK_BANDS];

        for (k, &mag) in magnitudes.iter().enumerate() {
            let f_hz = k as f64 * bin_hz;
            let bark = hz_to_bark(f_hz);
            let power = (mag as f64) * (mag as f64);

            // A frequency at Bark `z` contributes to the two bands whose
            // triangular skirts overlap it: the band centered just below and the
            // one just above. Band b is centered at z_c = b + 0.5; the triangle
            // is 1 at z_c and 0 at z_c ± 1.
            let lower = (bark - 0.5).floor(); // candidate band center index (b)
            for band_center in [lower, lower + 1.0] {
                let b = band_center as isize;
                if b < 0 || b as usize >= BARK_BANDS {
                    continue;
                }
                let z_c = b as f64 + 0.5;
                let w = 1.0 - (bark - z_c).abs();
                if w > 0.0 {
                    let bi = b as usize;
                    weighted_power[bi] += w * power;
                    weight_mass[bi] += w;
                }
            }
        }

        for (out, (&wp, &wm)) in bark_spectrum
            .iter_mut()
            .zip(weighted_power.iter().zip(weight_mass.iter()))
        {
            if wm > 0.0 {
                *out = (wp / wm).sqrt() as f32;
            }
        }

        Ok(bark_spectrum)
    }

    fn compute_temporal_similarity(&self, signal1: &[f32], signal2: &[f32]) -> Result<f32> {
        let correlation = self.compute_correlation(signal1, signal2);
        Ok(correlation.abs())
    }

    fn compute_loudness_similarity(&self, signal1: &[f32], signal2: &[f32]) -> Result<f32> {
        let loud1 = self.compute_loudness(signal1)?;
        let loud2 = self.compute_loudness(signal2)?;

        let diff = (loud1 - loud2).abs();
        let similarity = (-diff / 10.0).exp(); // Exponential decay with difference

        Ok(similarity)
    }

    /// One-sided magnitude spectrum (`n/2 + 1` bins, DC … Nyquist) of `signal`
    /// computed with the real FFT [`scirs2_fft::rfft`].
    ///
    /// This replaces a naive O(n²) DFT with the SciRS2 real-FFT (O(n log n)),
    /// using the same call convention as the rest of this module (the bin
    /// magnitudes are the complex norms of the returned spectrum).
    fn compute_magnitude_spectrum(&self, signal: &[f32]) -> Result<Vec<f32>> {
        let n = signal.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let input_f64: Vec<f64> = signal.iter().map(|&x| x as f64).collect();
        let spectrum =
            scirs2_fft::rfft(&input_f64, Some(n)).map_err(|e| AcousticError::ProcessingError {
                message: format!("rfft failed in compute_magnitude_spectrum: {e:?}"),
            })?;

        // rfft returns the one-sided spectrum (n/2 + 1 complex bins); the
        // magnitude of each bin is its complex norm.
        Ok(spectrum.iter().map(|c| c.norm() as f32).collect())
    }

    fn compute_spectral_centroid_simple(&self, signal: &[f32]) -> f32 {
        let spectrum = match self.compute_magnitude_spectrum(signal) {
            Ok(spec) => spec,
            Err(_) => return 0.0,
        };

        let mut weighted_sum = 0.0f32;
        let mut magnitude_sum = 0.0f32;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            weighted_sum += i as f32 * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }

    fn compute_spectral_spread_simple(&self, signal: &[f32]) -> f32 {
        let spectrum = match self.compute_magnitude_spectrum(signal) {
            Ok(spec) => spec,
            Err(_) => return 0.0,
        };

        let centroid = self.compute_spectral_centroid_simple(signal);
        let mut weighted_sum = 0.0f32;
        let mut magnitude_sum = 0.0f32;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let diff = i as f32 - centroid;
            weighted_sum += diff * diff * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            (weighted_sum / magnitude_sum).sqrt()
        } else {
            0.0
        }
    }

    fn compute_zero_crossing_rate(&self, signal: &[f32]) -> f32 {
        if signal.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 1..signal.len() {
            if (signal[i] >= 0.0) != (signal[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / (signal.len() - 1) as f32
    }

    fn compute_short_time_energy_variance(&self, signal: &[f32]) -> f32 {
        let frame_size = 256;
        let hop_size = 128;

        let mut energies = Vec::new();
        let mut frame_start = 0;

        while frame_start + frame_size <= signal.len() {
            let frame = &signal[frame_start..frame_start + frame_size];
            let energy = frame.iter().map(|&x| x * x).sum::<f32>() / frame_size as f32;
            energies.push(energy);
            frame_start += hop_size;
        }

        if energies.len() < 2 {
            return 0.0;
        }

        let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;
        let variance = energies
            .iter()
            .map(|&energy| (energy - mean_energy).powi(2))
            .sum::<f32>()
            / energies.len() as f32;

        variance
    }

    fn create_stoi_bands(sample_rate: u32) -> Vec<StoiBand> {
        // Create third-octave bands for STOI
        let mut bands = Vec::new();
        let nyquist = sample_rate as f32 / 2.0;

        // Standard third-octave center frequencies
        let center_freqs = vec![
            125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
            2000.0, 2500.0, 3150.0, 4000.0,
        ];

        for &center_freq in &center_freqs {
            if center_freq < nyquist {
                let bandwidth = center_freq * 0.23; // Approximate third-octave bandwidth
                let low_freq = center_freq - bandwidth / 2.0;
                let high_freq = center_freq + bandwidth / 2.0;

                bands.push(StoiBand {
                    center_freq,
                    low_freq: low_freq.max(0.0),
                    high_freq: high_freq.min(nyquist),
                    weight: 1.0,
                });
            }
        }

        bands
    }
}

/// Convert a frequency in Hz to the Bark scale using Traunmüller's (1990)
/// analytic approximation with the standard low/high-frequency corrections.
fn hz_to_bark(f_hz: f64) -> f64 {
    let f = f_hz.max(0.0);
    let mut z = 26.81 * f / (1960.0 + f) - 0.53;
    if z < 2.0 {
        z += 0.15 * (2.0 - z);
    } else if z > 20.1 {
        z += 0.22 * (z - 20.1);
    }
    z
}

/// Approximate inverse of [`hz_to_bark`] (ignoring the small edge corrections),
/// giving the center frequency in Hz of a Bark value. Used only for the
/// frequency-importance weighting, where the correction terms are negligible.
fn bark_to_hz(bark: f64) -> f64 {
    let z = bark + 0.53;
    let denom = (26.81 - z).max(1e-6);
    1960.0 * z / denom
}

/// A normalized second-order IIR section (biquad) with transfer function
///
/// ```text
///        b0 + b1 z^-1 + b2 z^-2
/// H(z) = ----------------------
///         1 + a1 z^-1 + a2 z^-2
/// ```
///
/// Used to build the IEC 61672-1 A-weighting cascade (one biquad per
/// bilinear-transformed second-order analog section).
#[derive(Debug, Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl Biquad {
    /// Construct a biquad from raw (un-normalized) numerator/denominator
    /// coefficients, dividing through by `a0` so the leading denominator
    /// coefficient is 1.
    fn normalized(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Magnitude `|H(e^{jw})|` of the section at frequency `f_hz` for sample
    /// rate `fs`, evaluated directly from the coefficients (no external
    /// dependency). `w = 2*pi*f/fs`.
    fn magnitude_at(&self, f_hz: f64, fs: f64) -> f64 {
        let w = 2.0 * std::f64::consts::PI * f_hz / fs;
        let (s1, c1) = w.sin_cos();
        let (s2, c2) = (2.0 * w).sin_cos();
        // Numerator and denominator as complex numbers (real/imag parts).
        let num_re = self.b0 + self.b1 * c1 + self.b2 * c2;
        let num_im = -(self.b1 * s1 + self.b2 * s2);
        let den_re = 1.0 + self.a1 * c1 + self.a2 * c2;
        let den_im = -(self.a1 * s1 + self.a2 * s2);
        let num_mag = (num_re * num_re + num_im * num_im).sqrt();
        let den_mag = (den_re * den_re + den_im * den_im).sqrt();
        if den_mag > f64::EPSILON {
            num_mag / den_mag
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_audio(length: usize, frequency: f32, sample_rate: f32) -> Vec<f32> {
        (0..length)
            .map(|i| (2.0 * PI * frequency * i as f32 / sample_rate).sin() * 0.5)
            .collect()
    }

    #[test]
    fn test_perceptual_evaluator_creation() {
        let evaluator = PerceptualEvaluator::new();
        assert_eq!(evaluator.sample_rate, 16000);
        assert_eq!(evaluator.stoi_frame_size, 256);
    }

    #[test]
    fn test_pesq_computation() {
        let evaluator = PerceptualEvaluator::new();
        let audio1 = create_test_audio(1000, 440.0, 16000.0);
        let audio2 = create_test_audio(1000, 440.0, 16000.0);

        let pesq = evaluator.compute_pesq(&audio1, &audio2).unwrap();
        assert!(pesq >= 1.0);
        assert!(pesq <= 4.5);
    }

    #[test]
    fn test_stoi_computation() {
        let evaluator = PerceptualEvaluator::new();
        let audio1 = create_test_audio(1000, 440.0, 16000.0);
        let audio2 = create_test_audio(1000, 440.0, 16000.0);

        let stoi = evaluator.compute_stoi(&audio1, &audio2).unwrap();
        assert!(stoi >= 0.0);
        assert!(stoi <= 1.0);
    }

    #[test]
    fn test_si_sdr_computation() {
        let evaluator = PerceptualEvaluator::new();
        let audio1 = create_test_audio(1000, 440.0, 16000.0);
        let audio2 = create_test_audio(1000, 440.0, 16000.0);

        let si_sdr = evaluator.compute_si_sdr(&audio1, &audio2).unwrap();
        assert!(si_sdr >= -20.0);
        assert!(si_sdr <= 60.0);
    }

    #[test]
    fn test_intrinsic_quality() {
        let evaluator = PerceptualEvaluator::new();
        let audio = create_test_audio(1000, 440.0, 16000.0);

        let quality = evaluator.compute_intrinsic_quality(&audio).unwrap();
        assert!(quality >= 1.0);
        assert!(quality <= 4.5);
    }

    #[test]
    fn test_loudness_computation() {
        let evaluator = PerceptualEvaluator::new();
        let audio = create_test_audio(1000, 440.0, 16000.0);

        let loudness = evaluator.compute_loudness(&audio).unwrap();
        assert!(loudness.is_finite());
    }

    #[test]
    fn test_bark_spectral_distortion() {
        let evaluator = PerceptualEvaluator::new();
        let audio1 = create_test_audio(1000, 440.0, 16000.0);
        let audio2 = create_test_audio(1000, 440.0, 16000.0);

        let distortion = evaluator
            .compute_bark_spectral_distortion(&audio1, &audio2)
            .unwrap();
        assert!(distortion >= 0.0);
        assert_eq!(distortion, 0.0); // Same audio should have 0 distortion
    }

    #[test]
    fn test_correlation_computation() {
        let evaluator = PerceptualEvaluator::new();
        let signal1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let signal2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let correlation = evaluator.compute_correlation(&signal1, &signal2);
        assert!((correlation - 1.0).abs() < 0.001); // Perfect correlation
    }

    #[test]
    fn test_empty_input_error() {
        let evaluator = PerceptualEvaluator::new();
        let empty_audio: Vec<f32> = vec![];
        let audio = create_test_audio(100, 440.0, 16000.0);

        assert!(evaluator.compute_pesq(&empty_audio, &audio).is_err());
        assert!(evaluator.compute_stoi(&empty_audio, &audio).is_err());
        assert!(evaluator.compute_si_sdr(&empty_audio, &audio).is_err());
        assert!(evaluator.compute_intrinsic_quality(&empty_audio).is_err());
    }

    #[test]
    fn test_bandpass_passes_center_frequency() {
        let evaluator = PerceptualEvaluator::with_sample_rate(16000);
        let sr = 16000.0_f32;
        let low = 900.0_f32;
        let high = 1100.0_f32;
        // The RBJ band-pass peaks at the geometric mean of the band edges.
        let f_center = (low * high).sqrt();

        let signal = create_test_audio(8000, f_center, sr);
        let filtered = evaluator
            .apply_bandpass_filter(&signal, low, high)
            .expect("bandpass filter should succeed");

        // Compare steady-state RMS (skip the IIR start-up transient).
        let skip = 2000;
        let in_rms = evaluator.compute_rms(&signal[skip..]);
        let out_rms = evaluator.compute_rms(&filtered[skip..]);
        let ratio = out_rms / in_rms;
        assert!(
            ratio > 0.9 && ratio < 1.1,
            "center-frequency gain should be ~1.0 (got {ratio})"
        );
    }

    #[test]
    fn test_bandpass_attenuates_octave_outside_band() {
        let evaluator = PerceptualEvaluator::with_sample_rate(16000);
        let sr = 16000.0_f32;
        let low = 900.0_f32;
        let high = 1100.0_f32;
        let f_center = (low * high).sqrt();

        // One octave above and below the center are well outside the band.
        for tone in [f_center * 2.0, f_center / 2.0] {
            let signal = create_test_audio(8000, tone, sr);
            let filtered = evaluator
                .apply_bandpass_filter(&signal, low, high)
                .expect("bandpass filter should succeed");

            let skip = 2000;
            let in_rms = evaluator.compute_rms(&signal[skip..]);
            let out_rms = evaluator.compute_rms(&filtered[skip..]);
            assert!(
                out_rms < 0.35 * in_rms,
                "tone an octave outside the band should be strongly attenuated \
                 (tone {tone} Hz, out/in = {})",
                out_rms / in_rms
            );
        }
    }

    #[test]
    fn test_bandpass_attenuates_dc() {
        let evaluator = PerceptualEvaluator::with_sample_rate(16000);
        let low = 900.0_f32;
        let high = 1100.0_f32;

        // A flat/DC signal: a band-pass has zero gain at DC.
        let signal = vec![0.5_f32; 8000];
        let filtered = evaluator
            .apply_bandpass_filter(&signal, low, high)
            .expect("bandpass filter should succeed");

        let skip = 2000;
        let in_rms = evaluator.compute_rms(&signal[skip..]);
        let out_rms = evaluator.compute_rms(&filtered[skip..]);
        assert!(
            out_rms < 0.02 * in_rms,
            "DC should be strongly attenuated by the band-pass (out/in = {})",
            out_rms / in_rms
        );
    }

    /// Deterministic cosine of a single full-spectrum bin (no RNG).
    fn cosine_at_bin(n: usize, bin: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * bin as f32 * i as f32 / n as f32).cos())
            .collect()
    }

    #[test]
    fn test_a_weighting_unity_at_1khz() {
        // IEC 61672-1 A-weighting is 0 dB (±~1 dB) at 1 kHz.
        let sr = 48000.0_f32;
        let evaluator = PerceptualEvaluator::with_sample_rate(sr as u32);

        let len = 48000; // 1 s
        let tone_1k = create_test_audio(len, 1000.0, sr);
        let weighted = evaluator.apply_a_weighting(&tone_1k).unwrap();

        // Skip the IIR start-up transient before measuring steady-state RMS.
        let skip = 8000;
        let in_rms = evaluator.compute_rms(&tone_1k[skip..]);
        let out_rms = evaluator.compute_rms(&weighted[skip..]);
        let gain_db = 20.0 * (out_rms / in_rms).log10();
        assert!(
            gain_db.abs() < 1.0,
            "A-weighting at 1 kHz should be ~0 dB (got {gain_db} dB)"
        );
    }

    #[test]
    fn test_a_weighting_attenuates_100hz_relative_to_1khz() {
        // A-weighting strongly attenuates low frequencies: ~-19 dB at 100 Hz
        // relative to the 0 dB reference at 1 kHz.
        let sr = 48000.0_f32;
        let evaluator = PerceptualEvaluator::with_sample_rate(sr as u32);
        let len = 48000;
        let skip = 8000;

        let tone_100 = create_test_audio(len, 100.0, sr);
        let tone_1k = create_test_audio(len, 1000.0, sr);

        let w_100 = evaluator.apply_a_weighting(&tone_100).unwrap();
        let w_1k = evaluator.apply_a_weighting(&tone_1k).unwrap();

        let gain_100_db = 20.0
            * (evaluator.compute_rms(&w_100[skip..]) / evaluator.compute_rms(&tone_100[skip..]))
                .log10();
        let gain_1k_db = 20.0
            * (evaluator.compute_rms(&w_1k[skip..]) / evaluator.compute_rms(&tone_1k[skip..]))
                .log10();

        // 100 Hz must be substantially attenuated relative to 1 kHz. The
        // standard tabulated value is about -19.1 dB; allow generous slack for
        // finite-length RMS estimation while still proving real weighting.
        let relative = gain_100_db - gain_1k_db;
        assert!(
            relative < -10.0,
            "100 Hz should be well below 1 kHz under A-weighting \
             (100 Hz = {gain_100_db} dB, 1 kHz = {gain_1k_db} dB, rel = {relative} dB)"
        );
    }

    #[test]
    fn test_bark_scale_1khz_lands_near_band_8() {
        // A 1 kHz tone sits at ~8.5 Bark, so the peak critical band should be
        // band 8 or 9 (24-band layout).
        let sr = 16000.0_f32;
        let evaluator = PerceptualEvaluator::with_sample_rate(sr as u32);

        // 1024-point cosine whose bin lands on (near) 1 kHz: bin = f*n/sr.
        let n = 1024;
        let bin = (1000.0_f32 * n as f32 / sr).round() as usize; // ~64
        let signal = cosine_at_bin(n, bin);

        let bark = evaluator.convert_to_bark_scale(&signal).unwrap();
        assert_eq!(bark.len(), 24, "Bark spectrum must have 24 bands");

        let peak_band = bark
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            (8..=9).contains(&peak_band),
            "1 kHz tone should peak in Bark band 8-9 (got {peak_band})"
        );
    }

    #[test]
    fn test_bark_scale_monotonic_band_centers() {
        // The underlying Hz->Bark map must be monotonically increasing, and a
        // rising sweep of single tones must move the peak band upward.
        let sr = 16000.0_f32;
        let evaluator = PerceptualEvaluator::with_sample_rate(sr as u32);
        let n = 1024;

        // hz_to_bark monotonicity (direct check on the standard mapping).
        let mut prev = hz_to_bark(0.0);
        for f in [50.0, 200.0, 500.0, 1000.0, 2000.0, 4000.0, 7000.0] {
            let z = hz_to_bark(f);
            assert!(z > prev, "hz_to_bark must increase ({f} Hz -> {z})");
            prev = z;
        }

        // Peak Bark band increases as the input tone frequency increases.
        let mut last_peak = 0usize;
        for &f in &[300.0_f32, 1000.0, 3000.0] {
            let bin = (f * n as f32 / sr).round() as usize;
            let signal = cosine_at_bin(n, bin);
            let bark = evaluator.convert_to_bark_scale(&signal).unwrap();
            let peak = bark
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            assert!(
                peak >= last_peak,
                "peak Bark band should not decrease with rising frequency \
                 ({f} Hz -> band {peak}, previous {last_peak})"
            );
            last_peak = peak;
        }
    }

    #[test]
    fn test_magnitude_spectrum_peaks_at_expected_bin() {
        // rfft magnitude of a single-bin cosine must peak at that bin.
        let evaluator = PerceptualEvaluator::with_sample_rate(16000);
        let n = 512;
        let target_bin = 40;
        let signal = cosine_at_bin(n, target_bin);

        let mag = evaluator.compute_magnitude_spectrum(&signal).unwrap();
        assert_eq!(mag.len(), n / 2 + 1, "one-sided spectrum is n/2+1 bins");

        let peak_bin = mag
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            peak_bin, target_bin,
            "single-bin cosine should peak at its own bin"
        );

        // Energy is concentrated: the peak dwarfs the average of the rest.
        let others_mean: f32 = mag
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != target_bin)
            .map(|(_, &m)| m)
            .sum::<f32>()
            / (mag.len() - 1) as f32;
        assert!(
            mag[target_bin] > 50.0 * others_mean.max(1e-6),
            "peak bin should dominate (peak {}, others_mean {others_mean})",
            mag[target_bin]
        );
    }
}
