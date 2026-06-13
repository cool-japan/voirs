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

    fn apply_a_weighting(&self, audio: &[f32]) -> Result<Vec<f32>> {
        // Simplified A-weighting filter
        // In practice, this would be a proper IIR filter implementation
        let mut weighted = audio.to_vec();

        if weighted.len() > 1 {
            // Simple high-pass characteristic of A-weighting
            for i in 1..weighted.len() {
                weighted[i] = 0.7 * weighted[i] + 0.3 * (audio[i] - audio[i - 1]);
            }
        }

        Ok(weighted)
    }

    fn convert_to_bark_scale(&self, signal: &[f32]) -> Result<Vec<f32>> {
        // Convert signal to bark scale representation
        // This is a simplified implementation
        let bark_bands = 24; // Standard number of bark bands
        let mut bark_spectrum = vec![0.0f32; bark_bands];

        let band_size = signal.len() / bark_bands;

        for (band_idx, bark_value) in bark_spectrum.iter_mut().enumerate() {
            let start_idx = band_idx * band_size;
            let end_idx = ((band_idx + 1) * band_size).min(signal.len());

            if start_idx < end_idx {
                let band_energy: f32 = signal[start_idx..end_idx].iter().map(|&x| x * x).sum();
                *bark_value = (band_energy / (end_idx - start_idx) as f32).sqrt();
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

    fn compute_magnitude_spectrum(&self, signal: &[f32]) -> Result<Vec<f32>> {
        // Simplified magnitude spectrum computation
        let n = signal.len();
        let mut spectrum = vec![0.0f32; n / 2];

        for (i, spec_val) in spectrum.iter_mut().enumerate() {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for (j, &sample) in signal.iter().enumerate() {
                let angle = -2.0 * PI * i as f32 * j as f32 / n as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            *spec_val = (real * real + imag * imag).sqrt();
        }

        Ok(spectrum)
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
}
