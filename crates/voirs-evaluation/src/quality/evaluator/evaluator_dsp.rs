//! Self-contained DSP feature extractors for [`super::QualityEvaluator`].
//!
//! These helpers implement the real signal-processing algorithms backing the
//! perceptual / spectral feature methods of `QualityEvaluator`. They previously
//! returned hardcoded constants; this module provides honest implementations.
//!
//! All spectral analysis uses the SciRS2 abstractions (`scirs2_fft::rfft` for
//! the real-input FFT and `scirs2_core::numeric::Complex64` for the complex
//! spectrum). No external `rand` / `ndarray` / `rayon` / `num_complex` /
//! `nalgebra` crates are used, in accordance with the SciRS2 policy.
//!
//! The functions are free-standing (`pub(super)`) and operate on raw `&[f32]`
//! sample slices plus a `sample_rate`, so that the (already large)
//! `evaluator.rs` does not grow past the 2000-line limit.

use crate::EvaluationError;
use scirs2_core::numeric::Complex64;

/// Floor used to avoid `log(0)` / division by zero in spectral ratios.
const EPS: f64 = 1e-12;

/// Compute a Hann window of length `n`.
///
/// Uses the periodic Hann definition `0.5 - 0.5·cos(2πi/n)`, matching the rest
/// of the crate's framing code.
fn hann_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n.max(1)];
    }
    let denom = n as f64;
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / denom).cos())
        .collect()
}

/// Round `n` up to the next power of two (with a sane minimum).
fn next_pow2(n: usize) -> usize {
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p.max(2)
}

/// Real-FFT magnitude spectrum of a whole signal.
///
/// The signal is Hann-windowed and transformed with `scirs2_fft::rfft` to a
/// power-of-two length (`>= signal length`, capped to keep the transform cheap
/// for very long buffers). Returns the magnitude `|X[k]|` for the non-negative
/// frequency bins (`fft_len / 2 + 1` values).
pub(super) fn magnitude_spectrum(samples: &[f32]) -> Result<Vec<f32>, EvaluationError> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    // Cap the analysis length so a single FFT stays affordable for long audio;
    // 32768 covers ~2 s at 16 kHz which is plenty for global spectral shape.
    let analysis_len = samples.len().min(32768);
    let fft_len = next_pow2(analysis_len);
    let window = hann_window(analysis_len);
    let mut buffer: Vec<f64> = vec![0.0; fft_len];
    for (i, (&s, &w)) in samples[..analysis_len]
        .iter()
        .zip(window.iter())
        .enumerate()
    {
        buffer[i] = s as f64 * w;
    }
    let spectrum = scirs2_fft::rfft(&buffer, Some(fft_len))?;
    Ok(spectrum
        .iter()
        .map(|c: &Complex64| ((c.re * c.re + c.im * c.im).sqrt()) as f32)
        .collect())
}

/// Complex real-FFT spectrum of a whole signal at an explicit `fft_len`.
///
/// Used by phase-coherence analysis which needs both magnitude and phase. The
/// signal is Hann-windowed (over `min(len, fft_len)` samples) and zero-padded
/// to `fft_len`.
fn complex_spectrum(samples: &[f32], fft_len: usize) -> Result<Vec<Complex64>, EvaluationError> {
    let used = samples.len().min(fft_len);
    let window = hann_window(used);
    let mut buffer: Vec<f64> = vec![0.0; fft_len];
    for (i, (&s, &w)) in samples[..used].iter().zip(window.iter()).enumerate() {
        buffer[i] = s as f64 * w;
    }
    Ok(scirs2_fft::rfft(&buffer, Some(fft_len))?)
}

/// Per-frame Hann-windowed magnitude spectra (STFT magnitudes).
///
/// Returns one magnitude vector per frame of length `window_size`, advancing by
/// `hop_size`. Each frame is transformed with `scirs2_fft::rfft`, so every
/// magnitude vector has `window_size / 2 + 1` bins. Frames shorter than
/// `window_size` (the trailing remainder) are skipped.
fn frame_magnitude_spectra(
    samples: &[f32],
    window_size: usize,
    hop_size: usize,
) -> Result<Vec<Vec<f32>>, EvaluationError> {
    if samples.len() < window_size || window_size == 0 || hop_size == 0 {
        return Ok(Vec::new());
    }
    let window = hann_window(window_size);
    let mut frames = Vec::new();
    let mut start = 0;
    while start + window_size <= samples.len() {
        let mut buffer: Vec<f64> = Vec::with_capacity(window_size);
        for i in 0..window_size {
            buffer.push(samples[start + i] as f64 * window[i]);
        }
        let spectrum = scirs2_fft::rfft(&buffer, Some(window_size))?;
        let mags: Vec<f32> = spectrum
            .iter()
            .map(|c: &Complex64| ((c.re * c.re + c.im * c.im).sqrt()) as f32)
            .collect();
        frames.push(mags);
        start += hop_size;
    }
    Ok(frames)
}

/// Short-time RMS energy envelope of a signal.
///
/// Computes the root-mean-square amplitude over consecutive (non-overlapping)
/// frames of `frame` samples, giving a coarse amplitude envelope used by the
/// attack / decay estimators.
fn rms_envelope(samples: &[f32], frame: usize) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let frame = frame.max(1);
    let mut env = Vec::with_capacity(samples.len() / frame + 1);
    let mut i = 0;
    while i < samples.len() {
        let end = (i + frame).min(samples.len());
        let slice = &samples[i..end];
        let energy: f32 = slice.iter().map(|&x| x * x).sum::<f32>() / slice.len() as f32;
        env.push(energy.sqrt());
        i += frame;
    }
    env
}

/// Spectral rolloff: the frequency below which `roll_percent` (e.g. 0.85) of the
/// total spectral energy is contained.
///
/// Averaged over STFT frames. For each frame we accumulate magnitude energy
/// across bins and locate the bin where the running sum first reaches
/// `roll_percent` of the frame total, converting that bin index to Hz via
/// `bin · sample_rate / window_size`. The per-frame rolloffs are then averaged.
pub(super) fn spectral_rolloff(
    samples: &[f32],
    sample_rate: f32,
    window_size: usize,
    hop_size: usize,
    roll_percent: f32,
) -> Result<f32, EvaluationError> {
    let frames = frame_magnitude_spectra(samples, window_size, hop_size)?;
    if frames.is_empty() {
        // Fall back to a single whole-signal spectrum if the buffer is short.
        let spectrum = magnitude_spectrum(samples)?;
        return Ok(rolloff_from_spectrum(&spectrum, sample_rate, roll_percent));
    }
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for frame in &frames {
        sum += rolloff_from_spectrum(frame, sample_rate, roll_percent) as f64;
        count += 1;
    }
    Ok((sum / count.max(1) as f64) as f32)
}

/// Locate the rolloff frequency within a single magnitude spectrum.
///
/// The number of bins implies the FFT length (`2·(bins-1)`), which together
/// with `sample_rate` maps a bin index to its centre frequency.
fn rolloff_from_spectrum(spectrum: &[f32], sample_rate: f32, roll_percent: f32) -> f32 {
    if spectrum.len() < 2 {
        return sample_rate / 2.0;
    }
    let fft_len = (spectrum.len() - 1) * 2;
    let total: f32 = spectrum.iter().map(|&m| m * m).sum();
    if total <= 0.0 {
        return sample_rate / 2.0;
    }
    let threshold = total * roll_percent;
    let mut running = 0.0f32;
    for (k, &m) in spectrum.iter().enumerate() {
        running += m * m;
        if running >= threshold {
            return k as f32 * sample_rate / fft_len as f32;
        }
    }
    sample_rate / 2.0
}

/// Spectral flux: mean over frames of the half-wave-rectified spectral
/// difference `Σ_k max(0, |X_t[k]| − |X_{t−1}[k]|)`, normalized per frame by the
/// number of bins.
///
/// Captures how rapidly the magnitude spectrum changes between adjacent frames
/// — a low value for stationary tones, higher for transient / noisy signals.
pub(super) fn spectral_flux(
    samples: &[f32],
    window_size: usize,
    hop_size: usize,
) -> Result<f32, EvaluationError> {
    let frames = frame_magnitude_spectra(samples, window_size, hop_size)?;
    if frames.len() < 2 {
        return Ok(0.0);
    }
    let mut total_flux = 0.0f64;
    for pair in frames.windows(2) {
        let prev = &pair[0];
        let cur = &pair[1];
        let bins = prev.len().min(cur.len());
        if bins == 0 {
            continue;
        }
        let mut frame_flux = 0.0f64;
        for k in 0..bins {
            let diff = cur[k] - prev[k];
            if diff > 0.0 {
                frame_flux += diff as f64;
            }
        }
        total_flux += frame_flux / bins as f64;
    }
    Ok((total_flux / (frames.len() - 1) as f64) as f32)
}

/// Attack time: the time (seconds) for the rising amplitude envelope to climb
/// from 10% to 90% of its peak.
///
/// The RMS envelope is computed over short frames; we find the global peak, then
/// the first frame (scanning forward from the start, up to the peak) that
/// crosses 10% and the first that crosses 90% of the peak. The difference in
/// frame indices is scaled by the per-frame duration.
pub(super) fn attack_time(samples: &[f32], sample_rate: f32) -> Result<f32, EvaluationError> {
    if samples.is_empty() || sample_rate <= 0.0 {
        return Ok(0.0);
    }
    let frame = ((sample_rate * 0.005).round() as usize).max(1); // ~5 ms frames
    let env = rms_envelope(samples, frame);
    if env.len() < 2 {
        return Ok(0.0);
    }
    let (peak_idx, peak) =
        env.iter().enumerate().fold(
            (0usize, 0.0f32),
            |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            },
        );
    if peak <= 0.0 {
        return Ok(0.0);
    }
    let low_th = 0.1 * peak;
    let high_th = 0.9 * peak;
    let mut t_low: Option<usize> = None;
    let mut t_high: Option<usize> = None;
    for (i, &e) in env.iter().enumerate().take(peak_idx + 1) {
        if t_low.is_none() && e >= low_th {
            t_low = Some(i);
        }
        if t_high.is_none() && e >= high_th {
            t_high = Some(i);
            break;
        }
    }
    let frame_dur = frame as f32 / sample_rate;
    match (t_low, t_high) {
        (Some(lo), Some(hi)) if hi >= lo => Ok((hi - lo) as f32 * frame_dur),
        _ => Ok(0.0),
    }
}

/// Decay time: the time (seconds) for the energy envelope to fall from its peak
/// to ~10% of the peak.
///
/// The RMS envelope is scanned forward from the peak frame to the first frame
/// that drops to or below 10% of the peak; the index difference is scaled by the
/// per-frame duration. If the envelope never decays that far, the time to the
/// end of the signal is returned.
pub(super) fn decay_time(samples: &[f32], sample_rate: f32) -> Result<f32, EvaluationError> {
    if samples.is_empty() || sample_rate <= 0.0 {
        return Ok(0.0);
    }
    let frame = ((sample_rate * 0.005).round() as usize).max(1);
    let env = rms_envelope(samples, frame);
    if env.len() < 2 {
        return Ok(0.0);
    }
    let (peak_idx, peak) =
        env.iter().enumerate().fold(
            (0usize, 0.0f32),
            |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            },
        );
    if peak <= 0.0 {
        return Ok(0.0);
    }
    let decay_th = 0.1 * peak;
    let frame_dur = frame as f32 / sample_rate;
    for (i, &e) in env.iter().enumerate().skip(peak_idx) {
        if e <= decay_th {
            return Ok((i - peak_idx) as f32 * frame_dur);
        }
    }
    Ok((env.len() - 1 - peak_idx) as f32 * frame_dur)
}

/// Locate local maxima ("peaks") of a magnitude spectrum.
///
/// A bin is a peak when it strictly exceeds its immediate neighbours and is at
/// least `rel_floor` times the global maximum (suppressing noise ripples).
/// Returns `(frequency_hz, magnitude)` pairs.
fn spectral_peaks(spectrum: &[f32], sample_rate: f32, rel_floor: f32) -> Vec<(f32, f32)> {
    if spectrum.len() < 3 {
        return Vec::new();
    }
    let fft_len = (spectrum.len() - 1) * 2;
    let global_max = spectrum.iter().cloned().fold(0.0f32, f32::max);
    let floor = global_max * rel_floor;
    let mut peaks = Vec::new();
    for k in 1..spectrum.len() - 1 {
        let m = spectrum[k];
        if m > spectrum[k - 1] && m >= spectrum[k + 1] && m >= floor {
            let freq = k as f32 * sample_rate / fft_len as f32;
            peaks.push((freq, m));
        }
    }
    peaks
}

/// Sensory roughness from spectral peak pairs (Vassilakis-style).
///
/// For every pair of prominent spectral peaks we evaluate an asymmetric beating
/// weight that peaks near a frequency difference of ~70/critical-bandwidth Hz
/// and contributes proportionally to the product of the two amplitudes. The
/// modulation model follows Sethares/Vassilakis:
///
/// `R = Σ_{i<j} (a_i·a_j)^0.1 · ((2·min)/(a_i+a_j))^3.11 · g(Δf)`
///
/// with `g(Δf) = exp(-3.5·s·Δf) − exp(-5.75·s·Δf)` and
/// `s = 0.24 / (0.0207·f_min + 18.96)`. The result is normalized to roughly
/// `[0, 1]` by the total peak energy so that a pure tone (one peak) yields ~0.
pub(super) fn roughness(samples: &[f32], sample_rate: f32) -> Result<f32, EvaluationError> {
    let spectrum = magnitude_spectrum(samples)?;
    let peaks = spectral_peaks(&spectrum, sample_rate, 0.1);
    if peaks.len() < 2 {
        return Ok(0.0);
    }
    let mut roughness_sum = 0.0f64;
    for i in 0..peaks.len() {
        for j in (i + 1)..peaks.len() {
            let (f1, a1) = peaks[i];
            let (f2, a2) = peaks[j];
            let df = (f2 - f1).abs() as f64;
            let f_min = f1.min(f2).max(1.0) as f64;
            let a1 = a1 as f64;
            let a2 = a2 as f64;
            let amin = a1.min(a2);
            let s = 0.24 / (0.0207 * f_min + 18.96);
            let g = (-3.5 * s * df).exp() - (-5.75 * s * df).exp();
            if g <= 0.0 {
                continue;
            }
            let amp_term = (a1 * a2).powf(0.1) * (2.0 * amin / (a1 + a2)).powf(3.11);
            roughness_sum += amp_term * g;
        }
    }
    // Normalize by total peak amplitude so the value is scale-independent.
    let total_amp: f64 = peaks.iter().map(|&(_, a)| a as f64).sum::<f64>().max(EPS);
    Ok((roughness_sum / total_amp) as f32)
}

/// Convert a frequency in Hz to its critical-band rate (Bark), Zwicker/Terhardt.
fn hz_to_bark(hz: f32) -> f32 {
    let f = hz.max(0.0);
    13.0 * (0.00076 * f).atan() + 3.5 * ((f / 7500.0).powi(2)).atan()
}

/// Zwicker sharpness (acum), specific-loudness weighted.
///
/// The magnitude spectrum is aggregated into 24 Bark bands to approximate the
/// specific-loudness pattern `N'(z)` (using band energy as a loudness proxy).
/// Sharpness is the weighted first moment over Bark:
///
/// `S = c · Σ_z z · g(z) · N'(z) / Σ_z N'(z)`
///
/// with the standard upper-band weighting `g(z) = 1` for `z ≤ 16` and
/// `g(z) = 0.066·exp(0.171·z)` above, and `c ≈ 0.11`. Higher values indicate a
/// brighter, more high-frequency-dominated timbre.
pub(super) fn sharpness(samples: &[f32], sample_rate: f32) -> Result<f32, EvaluationError> {
    let spectrum = magnitude_spectrum(samples)?;
    if spectrum.len() < 2 {
        return Ok(0.0);
    }
    let fft_len = (spectrum.len() - 1) * 2;
    let n_bark = 24usize;
    let mut band_loudness = vec![0.0f64; n_bark];
    for (k, &m) in spectrum.iter().enumerate() {
        let freq = k as f32 * sample_rate / fft_len as f32;
        let z = hz_to_bark(freq);
        let band = (z.floor() as usize).min(n_bark - 1);
        // Specific loudness proxy: energy^0.23 (loudness exponent ~0.23).
        band_loudness[band] += (m as f64 * m as f64).powf(0.23 / 2.0);
    }
    let total: f64 = band_loudness.iter().sum::<f64>().max(EPS);
    let mut weighted = 0.0f64;
    for (z_idx, &nz) in band_loudness.iter().enumerate() {
        let z = z_idx as f64 + 0.5;
        let g = if z <= 16.0 {
            1.0
        } else {
            0.066 * (0.171 * z).exp()
        };
        weighted += z * g * nz;
    }
    Ok((0.11 * weighted / total) as f32)
}

/// Spectral-flatness-based tonality: `1 − SFM`.
///
/// The spectral flatness measure (SFM) is the ratio of the geometric mean to the
/// arithmetic mean of the power spectrum. SFM ≈ 1 for white noise (flat
/// spectrum, low tonality) and ≈ 0 for a pure tone (peaky spectrum, high
/// tonality). Tonality is therefore `1 − SFM`.
pub(super) fn tonality(samples: &[f32], _sample_rate: f32) -> Result<f32, EvaluationError> {
    let spectrum = magnitude_spectrum(samples)?;
    // Skip the DC bin which biases flatness.
    if spectrum.len() < 3 {
        return Ok(0.0);
    }
    let power: Vec<f64> = spectrum[1..]
        .iter()
        .map(|&m| (m as f64 * m as f64) + EPS)
        .collect();
    let n = power.len() as f64;
    let log_sum: f64 = power.iter().map(|&p| p.ln()).sum();
    let geo_mean = (log_sum / n).exp();
    let arith_mean = power.iter().sum::<f64>() / n;
    let sfm = (geo_mean / arith_mean.max(EPS)).clamp(0.0, 1.0);
    Ok((1.0 - sfm) as f32)
}

/// Estimate the fundamental period (in samples) of a frame via normalized
/// autocorrelation, searching the range implied by `[f_min, f_max]`.
///
/// Returns `Some(period)` when a sufficiently strong periodicity is found
/// (normalized correlation above 0.3), else `None` (unvoiced / aperiodic).
fn estimate_period(frame: &[f32], sample_rate: f32, f_min: f32, f_max: f32) -> Option<usize> {
    if frame.len() < 4 || sample_rate <= 0.0 {
        return None;
    }
    let min_period = ((sample_rate / f_max).floor() as usize).max(2);
    let max_period = ((sample_rate / f_min).ceil() as usize).min(frame.len() / 2);
    if max_period <= min_period {
        return None;
    }
    let mut best_corr = 0.0f32;
    let mut best_period = 0usize;
    for period in min_period..=max_period {
        let mut corr = 0.0f32;
        let mut norm1 = 0.0f32;
        let mut norm2 = 0.0f32;
        for i in 0..(frame.len() - period) {
            corr += frame[i] * frame[i + period];
            norm1 += frame[i] * frame[i];
            norm2 += frame[i + period] * frame[i + period];
        }
        let denom = (norm1 * norm2).sqrt();
        if denom > 0.0 {
            let nc = corr / denom;
            if nc > best_corr {
                best_corr = nc;
                best_period = period;
            }
        }
    }
    if best_period > 0 && best_corr > 0.3 {
        Some(best_period)
    } else {
        None
    }
}

/// Harmonicity: ratio of energy located at harmonic multiples of an estimated
/// f0 to the total spectral energy.
///
/// The f0 is estimated by autocorrelation over the (cropped) signal; its
/// frequency yields harmonic bin positions `n·f0`. We sum magnitude energy in a
/// small tolerance window around each harmonic and divide by the total spectral
/// energy. A clean harmonic tone scores near 1; noise scores low.
pub(super) fn harmonicity(samples: &[f32], sample_rate: f32) -> Result<f32, EvaluationError> {
    if samples.is_empty() || sample_rate <= 0.0 {
        return Ok(0.0);
    }
    let analysis_len = samples.len().min(4096);
    let frame = &samples[..analysis_len];
    let period = match estimate_period(frame, sample_rate, 60.0, 1000.0) {
        Some(p) => p,
        None => return Ok(0.0),
    };
    let f0 = sample_rate / period as f32;
    let spectrum = magnitude_spectrum(samples)?;
    if spectrum.len() < 2 {
        return Ok(0.0);
    }
    let fft_len = (spectrum.len() - 1) * 2;
    let bin_hz = sample_rate / fft_len as f32;
    let total_energy: f64 = spectrum.iter().map(|&m| m as f64 * m as f64).sum::<f64>() + EPS;
    let nyquist = sample_rate / 2.0;
    let tolerance_bins = ((f0 / bin_hz) * 0.05).ceil().max(1.0) as i64; // ±5% of f0
    let mut harmonic_energy = 0.0f64;
    let mut h = 1;
    loop {
        let hf = f0 * h as f32;
        if hf >= nyquist {
            break;
        }
        let center = (hf / bin_hz).round() as i64;
        let lo = (center - tolerance_bins).max(0) as usize;
        let hi = ((center + tolerance_bins) as usize).min(spectrum.len() - 1);
        let mut peak = 0.0f64;
        for &m in &spectrum[lo..=hi] {
            let e = m as f64 * m as f64;
            if e > peak {
                peak = e;
            }
        }
        harmonic_energy += peak;
        h += 1;
    }
    Ok((harmonic_energy / total_energy).clamp(0.0, 1.0) as f32)
}

/// Pad / truncate two signals to a common power-of-two length for paired FFT
/// analysis, returning the chosen length.
fn common_fft_len(a: &[f32], b: &[f32]) -> usize {
    let n = a.len().min(b.len()).clamp(1, 32768);
    next_pow2(n)
}

/// Spectral convergence between a signal and reference:
/// `‖|X| − |Y|‖_F / ‖|Y|‖_F`.
///
/// Both magnitude spectra are computed at the same FFT length; the Frobenius
/// norm of their difference is normalized by the reference norm. Identical
/// signals yield ~0.
pub(super) fn spectral_convergence(
    signal: &[f32],
    reference: &[f32],
) -> Result<f32, EvaluationError> {
    if signal.is_empty() || reference.is_empty() {
        return Ok(0.0);
    }
    let fft_len = common_fft_len(signal, reference);
    let x = paired_magnitude(signal, fft_len)?;
    let y = paired_magnitude(reference, fft_len)?;
    let bins = x.len().min(y.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for k in 0..bins {
        let d = x[k] as f64 - y[k] as f64;
        num += d * d;
        den += (y[k] as f64) * (y[k] as f64);
    }
    if den <= EPS {
        return Ok(0.0);
    }
    Ok((num.sqrt() / den.sqrt()) as f32)
}

/// Hann-windowed magnitude spectrum at an explicit FFT length (for paired
/// signal/reference comparisons).
fn paired_magnitude(samples: &[f32], fft_len: usize) -> Result<Vec<f32>, EvaluationError> {
    let spectrum = complex_spectrum(samples, fft_len)?;
    Ok(spectrum
        .iter()
        .map(|c: &Complex64| ((c.re * c.re + c.im * c.im).sqrt()) as f32)
        .collect())
}

/// Log-spectral distance between a signal and reference:
/// `sqrt(mean_k (10·log10(Px/Py))²)`.
///
/// Power spectra `Px`, `Py` are computed at a common FFT length; the RMS of the
/// per-bin log-power ratio (in dB) is returned. Identical signals yield ~0.
pub(super) fn log_spectral_distance(
    signal: &[f32],
    reference: &[f32],
) -> Result<f32, EvaluationError> {
    if signal.is_empty() || reference.is_empty() {
        return Ok(0.0);
    }
    let fft_len = common_fft_len(signal, reference);
    let x = paired_magnitude(signal, fft_len)?;
    let y = paired_magnitude(reference, fft_len)?;
    let bins = x.len().min(y.len());
    if bins == 0 {
        return Ok(0.0);
    }
    let mut sum_sq = 0.0f64;
    for k in 0..bins {
        let px = (x[k] as f64).powi(2) + EPS;
        let py = (y[k] as f64).powi(2) + EPS;
        let d = 10.0 * (px / py).log10();
        sum_sq += d * d;
    }
    Ok((sum_sq / bins as f64).sqrt() as f32)
}

/// Spectral similarity: cosine similarity of the two magnitude spectra, in
/// `[0, 1]`.
///
/// `cos = <|X|, |Y|> / (‖|X|‖·‖|Y|‖)`. Identical signals yield 1.
pub(super) fn spectral_similarity(
    signal: &[f32],
    reference: &[f32],
) -> Result<f32, EvaluationError> {
    if signal.is_empty() || reference.is_empty() {
        return Ok(0.0);
    }
    let fft_len = common_fft_len(signal, reference);
    let x = paired_magnitude(signal, fft_len)?;
    let y = paired_magnitude(reference, fft_len)?;
    let bins = x.len().min(y.len());
    let mut dot = 0.0f64;
    let mut nx = 0.0f64;
    let mut ny = 0.0f64;
    for k in 0..bins {
        let a = x[k] as f64;
        let b = y[k] as f64;
        dot += a * b;
        nx += a * a;
        ny += b * b;
    }
    let denom = (nx.sqrt() * ny.sqrt()).max(EPS);
    Ok((dot / denom).clamp(0.0, 1.0) as f32)
}

/// Temporal similarity: Pearson correlation of the two short-time RMS energy
/// envelopes, mapped to `[0, 1]`.
///
/// Envelopes are computed over ~10 ms frames and resampled (by index clamping)
/// to a common length; their Pearson correlation `r` is rescaled as
/// `(r + 1) / 2`. Identical signals yield 1.
pub(super) fn temporal_similarity(
    signal: &[f32],
    reference: &[f32],
    sample_rate: f32,
) -> Result<f32, EvaluationError> {
    if signal.is_empty() || reference.is_empty() || sample_rate <= 0.0 {
        return Ok(0.0);
    }
    let frame = ((sample_rate * 0.01).round() as usize).max(1); // ~10 ms
    let env_a = rms_envelope(signal, frame);
    let env_b = rms_envelope(reference, frame);
    let n = env_a.len().min(env_b.len());
    if n < 2 {
        return Ok(0.0);
    }
    let a = &env_a[..n];
    let b = &env_b[..n];
    let mean_a: f64 = a.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let mean_b: f64 = b.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let mut cov = 0.0f64;
    let mut va = 0.0f64;
    let mut vb = 0.0f64;
    for i in 0..n {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    let denom = (va.sqrt() * vb.sqrt()).max(EPS);
    let r = (cov / denom).clamp(-1.0, 1.0);
    Ok(((r + 1.0) / 2.0) as f32)
}

/// Per-frame f0 contour estimated by autocorrelation.
///
/// Returns one f0 value (Hz) per frame; unvoiced frames are reported as `0.0`.
fn f0_contour(samples: &[f32], sample_rate: f32) -> Vec<f32> {
    let frame_size = 1024usize;
    let hop = 512usize;
    if samples.len() < frame_size || sample_rate <= 0.0 {
        return Vec::new();
    }
    let mut contour = Vec::new();
    let mut start = 0;
    while start + frame_size <= samples.len() {
        let frame = &samples[start..start + frame_size];
        let f0 = match estimate_period(frame, sample_rate, 60.0, 500.0) {
            Some(p) => sample_rate / p as f32,
            None => 0.0,
        };
        contour.push(f0);
        start += hop;
    }
    contour
}

/// F0-contour similarity between a signal and reference.
///
/// Both f0 contours are estimated per frame (autocorrelation). Over frames that
/// are voiced in both signals, we compute a similarity from the relative pitch
/// error `1 − |f0_a − f0_b| / max(f0_a, f0_b)`, averaged. If there is no common
/// voiced region the similarity is 0. Identical signals yield ~1.
pub(super) fn f0_similarity(
    signal: &[f32],
    reference: &[f32],
    sample_rate: f32,
) -> Result<f32, EvaluationError> {
    let ca = f0_contour(signal, sample_rate);
    let cb = f0_contour(reference, sample_rate);
    let n = ca.len().min(cb.len());
    if n == 0 {
        return Ok(0.0);
    }
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for i in 0..n {
        if ca[i] > 0.0 && cb[i] > 0.0 {
            let hi = ca[i].max(cb[i]);
            let rel_err = (ca[i] - cb[i]).abs() / hi;
            sum += (1.0 - rel_err).max(0.0) as f64;
            count += 1;
        }
    }
    if count == 0 {
        return Ok(0.0);
    }
    Ok((sum / count as f64) as f32)
}

/// Phase coherence between a signal and reference.
///
/// Both signals are transformed at a common FFT length; per bin we form the
/// cross-spectral phase difference `arg(X·conj(Y))` and measure its consistency
/// as the magnitude of the mean phasor `|(1/K)·Σ_k exp(j·Δφ_k)|`, weighted by
/// the geometric mean of the two bin magnitudes so that silent bins do not
/// dominate. Identical signals (`Δφ = 0` everywhere) yield ~1; random relative
/// phase yields ~0.
pub(super) fn phase_coherence(signal: &[f32], reference: &[f32]) -> Result<f32, EvaluationError> {
    if signal.is_empty() || reference.is_empty() {
        return Ok(0.0);
    }
    let fft_len = common_fft_len(signal, reference);
    let xs = complex_spectrum(signal, fft_len)?;
    let ys = complex_spectrum(reference, fft_len)?;
    let bins = xs.len().min(ys.len());
    let mut acc_re = 0.0f64;
    let mut acc_im = 0.0f64;
    let mut weight = 0.0f64;
    for k in 0..bins {
        let x = xs[k];
        let y = ys[k];
        // Cross-spectrum X · conj(Y).
        let cross_re = x.re * y.re + x.im * y.im;
        let cross_im = x.im * y.re - x.re * y.im;
        let cross_mag = (cross_re * cross_re + cross_im * cross_im).sqrt();
        if cross_mag <= EPS {
            continue;
        }
        let w = ((x.re * x.re + x.im * x.im) * (y.re * y.re + y.im * y.im)).sqrt();
        // Unit phasor of the cross-spectrum, weighted by bin energy.
        acc_re += w * cross_re / cross_mag;
        acc_im += w * cross_im / cross_mag;
        weight += w;
    }
    if weight <= EPS {
        return Ok(0.0);
    }
    let coherence = ((acc_re / weight).powi(2) + (acc_im / weight).powi(2)).sqrt();
    Ok(coherence.clamp(0.0, 1.0) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Deterministic, approximately white pseudo-noise via a fixed
    /// linear-congruential generator (no RNG crate). The top 32 bits of the LCG
    /// state are mapped to a zero-mean value in `[-1, 1)`.
    fn lcg_noise(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            // Numerical Recipes LCG constants.
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Top 32 bits -> [0, 1) -> [-1, 1).
            let u = (state >> 32) as f32 / (1u64 << 32) as f32;
            out.push(2.0 * u - 1.0);
        }
        out
    }

    /// Pure sine tone at `freq` Hz.
    fn sine(n: usize, freq: f32, sample_rate: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    const SR: f32 = 16000.0;

    #[test]
    fn test_tonality_tone_vs_noise() {
        let tone = sine(16000, 440.0, SR);
        let noise = lcg_noise(16000, 0x1234_5678);
        let tone_tonality = tonality(&tone, SR).unwrap();
        let noise_tonality = tonality(&noise, SR).unwrap();
        // A pure tone has a peaky spectrum (high tonality, low flatness);
        // white noise is flat (low tonality).
        assert!(
            tone_tonality > noise_tonality,
            "tone {tone_tonality} should exceed noise {noise_tonality}"
        );
        assert!(
            tone_tonality > 0.5,
            "tone tonality {tone_tonality} should be high"
        );
        assert!(
            noise_tonality < 0.5,
            "noise tonality {noise_tonality} should be low"
        );
    }

    #[test]
    fn test_harmonicity_tone_high() {
        let tone = sine(16000, 220.0, SR);
        let noise = lcg_noise(16000, 0xDEAD_BEEF);
        let tone_h = harmonicity(&tone, SR).unwrap();
        let noise_h = harmonicity(&noise, SR).unwrap();
        assert!(tone_h > 0.5, "tone harmonicity {tone_h} should be high");
        assert!(
            tone_h > noise_h,
            "tone {tone_h} should be more harmonic than noise {noise_h}"
        );
    }

    #[test]
    fn test_rolloff_below_nyquist_for_lowpass() {
        // Low-frequency tone => spectral energy concentrated low => rolloff well
        // below Nyquist (8000 Hz for 16 kHz sample rate).
        let tone = sine(16000, 300.0, SR);
        let rolloff = spectral_rolloff(&tone, SR, 1024, 512, 0.85).unwrap();
        assert!(
            rolloff < SR / 2.0,
            "rolloff {rolloff} should be below Nyquist {}",
            SR / 2.0
        );
        assert!(
            rolloff < 2000.0,
            "rolloff {rolloff} should be low for a 300 Hz tone"
        );
    }

    #[test]
    fn test_identical_signal_convergence_and_similarity() {
        let tone = sine(8192, 500.0, SR);
        let sc = spectral_convergence(&tone, &tone).unwrap();
        let sim = spectral_similarity(&tone, &tone).unwrap();
        let lsd = log_spectral_distance(&tone, &tone).unwrap();
        let pc = phase_coherence(&tone, &tone).unwrap();
        let tsim = temporal_similarity(&tone, &tone, SR).unwrap();
        let f0sim = f0_similarity(&tone, &tone, SR).unwrap();
        assert!(
            sc < 1e-3,
            "spectral convergence of identical signal {sc} should be ~0"
        );
        assert!(
            sim > 0.999,
            "spectral similarity of identical signal {sim} should be ~1"
        );
        assert!(
            lsd < 1e-2,
            "log-spectral distance of identical signal {lsd} should be ~0"
        );
        assert!(
            pc > 0.999,
            "phase coherence of identical signal {pc} should be ~1"
        );
        assert!(
            tsim > 0.999,
            "temporal similarity of identical signal {tsim} should be ~1"
        );
        assert!(
            f0sim > 0.99,
            "f0 similarity of identical signal {f0sim} should be ~1"
        );
    }

    #[test]
    fn test_attack_time_ramp() {
        // Linear amplitude ramp over 0.5 s then steady => measurable attack.
        let n = 8000;
        let ramp_len = 4000;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let env = if i < ramp_len {
                i as f32 / ramp_len as f32
            } else {
                1.0
            };
            samples.push(env * (2.0 * PI * 440.0 * i as f32 / SR).sin());
        }
        let at = attack_time(&samples, SR).unwrap();
        assert!(
            at > 0.0,
            "attack time {at} should be measurable and positive"
        );
        // The 10%->90% span of a linear ramp of 0.25 s is ~0.2 s; allow margin.
        assert!(
            at < 0.5,
            "attack time {at} should be within the ramp duration"
        );
    }

    #[test]
    fn test_decay_time_exponential() {
        // Exponentially decaying tone => positive decay time.
        let n = 8000;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let env = (-(i as f32) / 1000.0).exp();
                env * (2.0 * PI * 440.0 * i as f32 / SR).sin()
            })
            .collect();
        let dt = decay_time(&samples, SR).unwrap();
        assert!(dt > 0.0, "decay time {dt} should be positive");
    }

    #[test]
    fn test_spectral_flux_noise_exceeds_tone() {
        let tone = sine(16000, 440.0, SR);
        let noise = lcg_noise(16000, 0xABCD);
        let tone_flux = spectral_flux(&tone, 1024, 512).unwrap();
        let noise_flux = spectral_flux(&noise, 1024, 512).unwrap();
        assert!(tone_flux >= 0.0 && noise_flux >= 0.0);
        assert!(
            noise_flux > tone_flux,
            "noise flux {noise_flux} should exceed stationary-tone flux {tone_flux}"
        );
    }

    #[test]
    fn test_roughness_two_close_tones() {
        // Two tones ~70 Hz apart fall in the beating range => non-zero roughness;
        // a single pure tone => ~0.
        let n = 16000;
        let two: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR;
                0.5 * (2.0 * PI * 440.0 * t).sin() + 0.5 * (2.0 * PI * 510.0 * t).sin()
            })
            .collect();
        let single = sine(n, 440.0, SR);
        let rough_two = roughness(&two, SR).unwrap();
        let rough_single = roughness(&single, SR).unwrap();
        assert!(
            rough_two > rough_single,
            "beating pair {rough_two} > single {rough_single}"
        );
        assert!(rough_two > 0.0);
    }

    #[test]
    fn test_sharpness_high_vs_low() {
        // A high-frequency tone should be sharper than a low-frequency tone.
        let low = sine(16000, 200.0, SR);
        let high = sine(16000, 6000.0, SR);
        let s_low = sharpness(&low, SR).unwrap();
        let s_high = sharpness(&high, SR).unwrap();
        assert!(
            s_high > s_low,
            "high-freq sharpness {s_high} > low-freq {s_low}"
        );
        assert!(s_low >= 0.0);
    }

    #[test]
    fn test_flatness_tone_vs_noise_inverse() {
        // Sanity: tonality(noise) low, tonality(tone) high already tested;
        // here verify spectral convergence between tone and noise is large.
        let tone = sine(8192, 440.0, SR);
        let noise = lcg_noise(8192, 0x55AA);
        let sc = spectral_convergence(&tone, &noise).unwrap();
        let sim = spectral_similarity(&tone, &noise).unwrap();
        assert!(
            sc > 0.1,
            "tone vs noise convergence {sc} should be substantial"
        );
        assert!(
            sim < 0.99,
            "tone vs noise similarity {sim} should be well below 1"
        );
    }
}
