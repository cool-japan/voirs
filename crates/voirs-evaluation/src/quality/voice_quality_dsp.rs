//! Voice-quality DSP primitives shared by the cross-language intelligibility
//! analysis.
//!
//! These free functions implement the low-level signal analysis used to derive
//! voice-quality metrics (jitter, shimmer, harmonic-to-noise ratio, spectral
//! tilt) and formant descriptors (bandwidths, clarity) directly from a `&[f32]`
//! signal plus its sample rate. They operate on raw slices so they can be
//! unit-tested in isolation, and use the SciRS2 abstractions
//! (`scirs2_fft::rfft`, `scirs2_core::Complex`).

use scirs2_core::Complex;

/// Lowest fundamental frequency considered voiced (Hz).
const MIN_F0_HZ: f64 = 50.0;
/// Highest fundamental frequency considered voiced (Hz).
const MAX_F0_HZ: f64 = 500.0;
/// Minimum normalized autocorrelation for a signal to be treated as voiced.
pub(crate) const VOICING_THRESHOLD: f64 = 0.45;

/// Estimate the dominant pitch period (in samples) using a normalized
/// cross-correlation search over the plausible speech F0 range.
///
/// Returns `(period_in_samples, normalized_correlation)` where the correlation
/// lies in `[0, 1]` and doubles as a voicing strength. Returns `None` when the
/// signal is too short or no periodicity is found.
pub(crate) fn estimate_pitch_period(samples: &[f32], sample_rate: u32) -> Option<(f64, f64)> {
    if sample_rate == 0 {
        return None;
    }
    let min_lag = (sample_rate as f64 / MAX_F0_HZ).floor() as usize;
    let max_lag_raw = (sample_rate as f64 / MIN_F0_HZ).ceil() as usize;
    let n = samples.len();
    if min_lag < 1 || n <= 2 * min_lag {
        return None;
    }
    let max_lag = max_lag_raw.min(n - 1);
    if max_lag < min_lag {
        return None;
    }

    let signal: Vec<f64> = samples.iter().map(|&x| x as f64).collect();
    let mut correlations = vec![0.0f64; max_lag + 1];
    let mut best_lag = 0usize;
    let mut best_correlation = 0.0f64;

    for lag in min_lag..=max_lag {
        let mut cross = 0.0;
        let mut energy_lead = 0.0;
        let mut energy_lag = 0.0;
        for i in lag..n {
            let lead = signal[i];
            let lagged = signal[i - lag];
            cross += lead * lagged;
            energy_lead += lead * lead;
            energy_lag += lagged * lagged;
        }
        let denom = (energy_lead * energy_lag).sqrt();
        let correlation = if denom > 0.0 { cross / denom } else { 0.0 };
        correlations[lag] = correlation;
        if correlation > best_correlation {
            best_correlation = correlation;
            best_lag = lag;
        }
    }

    if best_lag == 0 {
        return None;
    }

    // Octave correction: prefer the smallest sub-multiple that is still strongly
    // periodic, guarding against locking onto a multiple of the true period.
    let mut fundamental = best_lag;
    for divisor in 2..=4 {
        let candidate = best_lag / divisor;
        if candidate >= min_lag && correlations[candidate] >= 0.85 * best_correlation {
            fundamental = candidate;
        }
    }

    Some((fundamental as f64, best_correlation.clamp(0.0, 1.0)))
}

/// Locate approximate pitch marks (per-cycle peak positions) guided by an
/// average period. Returns the sample indices of successive cycle peaks.
fn detect_pitch_marks(samples: &[f32], period: f64) -> Vec<usize> {
    let n = samples.len();
    let period_i = period.round() as usize;
    if period_i < 2 || n < 2 * period_i {
        return Vec::new();
    }
    let tolerance = ((period * 0.35).round() as usize).max(1);

    // First mark: largest-magnitude sample within the first ~1.5 periods.
    let first_window = ((1.5 * period) as usize).min(n);
    let mut current = argmax_abs(&samples[..first_window]);
    let mut marks = vec![current];

    loop {
        let predicted = current + period_i;
        if predicted >= n {
            break;
        }
        let lo = predicted.saturating_sub(tolerance);
        let hi = (predicted + tolerance + 1).min(n);
        if lo >= hi {
            break;
        }
        let next = lo + argmax_abs(&samples[lo..hi]);
        if next <= current {
            break;
        }
        marks.push(next);
        current = next;
    }

    marks
}

/// Index of the sample with the largest absolute value within `slice`.
fn argmax_abs(slice: &[f32]) -> usize {
    let mut index = 0;
    let mut best = f32::NEG_INFINITY;
    for (i, &value) in slice.iter().enumerate() {
        let magnitude = value.abs();
        if magnitude > best {
            best = magnitude;
            index = i;
        }
    }
    index
}

/// Compute `(jitter, shimmer)` from the cycle structure implied by `period`.
///
/// Returns `(0.0, 0.0)` when fewer than three pitch marks are found.
pub(crate) fn compute_jitter_shimmer(samples: &[f32], period: f64) -> (f32, f32) {
    let marks = detect_pitch_marks(samples, period);
    if marks.len() < 3 {
        return (0.0, 0.0);
    }

    // Inter-mark intervals (cycle periods, in samples).
    let periods: Vec<f64> = marks.windows(2).map(|w| (w[1] - w[0]) as f64).collect();

    // Per-cycle peak amplitude (max absolute sample within each cycle).
    let amplitudes: Vec<f64> = marks
        .windows(2)
        .map(|w| {
            samples[w[0]..w[1]]
                .iter()
                .map(|&x| x.abs() as f64)
                .fold(0.0, f64::max)
        })
        .collect();

    let jitter = relative_mean_abs_difference(&periods) as f32;
    let shimmer = relative_mean_abs_difference(&amplitudes) as f32;

    (jitter.clamp(0.0, 1.0), shimmer.clamp(0.0, 2.0))
}

/// `mean(|xᵢ₊₁ − xᵢ|) / mean(x)`; `0.0` when there are fewer than two values or
/// the mean is non-positive.
fn relative_mean_abs_difference(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if mean <= 0.0 {
        return 0.0;
    }
    let diff_sum: f64 = values.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    let mean_diff = diff_sum / (values.len() - 1) as f64;
    mean_diff / mean
}

/// Harmonic-to-noise ratio (dB) from the normalized autocorrelation `r` at the
/// pitch period: `10·log10(r / (1 − r))`.
pub(crate) fn hnr_from_correlation(correlation: f64) -> f32 {
    let r = correlation.clamp(1e-6, 1.0 - 1e-6);
    (10.0 * (r / (1.0 - r)).log10()) as f32
}

/// Spectral tilt in dB/octave: the slope of a least-squares fit of the
/// log-magnitude spectrum (dB) against log2(frequency). Negative for speech and
/// low-pass spectra. Returns `0.0` for degenerate inputs.
pub(crate) fn compute_spectral_tilt(samples: &[f32], sample_rate: u32) -> f32 {
    if sample_rate == 0 || samples.len() < 64 {
        return 0.0;
    }
    let n = fft_window_size(samples.len());
    if n < 16 {
        return 0.0;
    }
    let start = (samples.len() - n) / 2;
    let segment = &samples[start..start + n];

    // Hann-windowed f64 analysis buffer.
    let denom = (n as f64 - 1.0).max(1.0);
    let buffer: Vec<f64> = segment
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let window = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / denom).cos();
            s as f64 * window
        })
        .collect();

    let spectrum = match scirs2_fft::rfft(&buffer, Some(n)) {
        Ok(spectrum) => spectrum,
        Err(_) => return 0.0,
    };

    let bin_hz = sample_rate as f64 / n as f64;
    let f_min = 100.0_f64.max(bin_hz);
    let f_max = (sample_rate as f64 / 2.0) * 0.95;

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    let mut count = 0.0;
    for (k, value) in spectrum.iter().enumerate() {
        let frequency = k as f64 * bin_hz;
        if frequency < f_min || frequency > f_max {
            continue;
        }
        let magnitude = (value.re * value.re + value.im * value.im).sqrt();
        let db = 20.0 * (magnitude + 1e-12).log10();
        let x = frequency.log2();
        sum_x += x;
        sum_y += db;
        sum_xx += x * x;
        sum_xy += x * db;
        count += 1.0;
    }
    if count < 2.0 {
        return 0.0;
    }
    let regression_denom = count * sum_xx - sum_x * sum_x;
    if regression_denom.abs() < 1e-12 {
        return 0.0;
    }
    ((count * sum_xy - sum_x * sum_y) / regression_denom) as f32
}

/// Largest power of two not exceeding `len`, capped at 8192 and floored at 256
/// (or `len` itself when the signal is shorter than 256 samples).
fn fft_window_size(len: usize) -> usize {
    if len < 256 {
        return len;
    }
    let mut n = 1usize;
    while n * 2 <= len && n * 2 <= 8192 {
        n *= 2;
    }
    n
}

/// LPC analysis order for a sample rate: roughly `2 + sample_rate/1000` (one
/// pole pair per kHz plus a few), clamped to `[8, 50]`.
fn lpc_order(sample_rate: u32) -> usize {
    (2 + sample_rate as usize / 1000).clamp(8, 50)
}

/// Compute LPC coefficients via autocorrelation + Levinson-Durbin.
///
/// Returns the polynomial coefficients `a[1..=order]` (length `order`) for
/// `A(z) = 1 + Σ aₖ z⁻ᵏ`, or `None` when the signal is too short / degenerate.
fn lpc_coefficients(samples: &[f32], order: usize) -> Option<Vec<f64>> {
    let n = samples.len();
    if order == 0 || n <= order + 1 {
        return None;
    }

    // Hann window to reduce edge effects before autocorrelation.
    let denom = (n as f64 - 1.0).max(1.0);
    let windowed: Vec<f64> = samples
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let window = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / denom).cos();
            s as f64 * window
        })
        .collect();

    let mut autocorr = vec![0.0f64; order + 1];
    for (lag, slot) in autocorr.iter_mut().enumerate() {
        let mut sum = 0.0;
        for i in lag..n {
            sum += windowed[i] * windowed[i - lag];
        }
        *slot = sum;
    }
    if autocorr[0] <= 0.0 {
        return None;
    }
    // Tiny white-noise floor for numerical stability of the recursion.
    autocorr[0] *= 1.0 + 1e-9;

    levinson_durbin(&autocorr, order)
}

/// Levinson-Durbin recursion over the autocorrelation `r[0..=order]`. Returns
/// LPC coefficients `a[1..=order]` (length `order`) or `None`.
fn levinson_durbin(r: &[f64], order: usize) -> Option<Vec<f64>> {
    if r.len() <= order || r[0] <= 0.0 {
        return None;
    }
    let mut a = vec![0.0f64; order + 1];
    a[0] = 1.0;
    let mut error = r[0];

    for i in 1..=order {
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let reflection = -acc / error;
        if !reflection.is_finite() {
            return None;
        }
        let previous = a.clone();
        for j in 1..i {
            a[j] = previous[j] + reflection * previous[i - j];
        }
        a[i] = reflection;
        error *= 1.0 - reflection * reflection;
        if error <= 0.0 {
            error = 1e-9;
        }
    }

    Some(a[1..=order].to_vec())
}

/// Evaluate the LPC all-pole spectral envelope (dB, ignoring the constant gain)
/// over a linear frequency grid `[0, Nyquist]`. Returns `(frequencies, dB)`.
fn lpc_envelope_db(coeffs: &[f64], sample_rate: u32, n_points: usize) -> (Vec<f32>, Vec<f32>) {
    let mut frequencies = Vec::with_capacity(n_points);
    let mut magnitudes_db = Vec::with_capacity(n_points);
    if n_points < 2 || sample_rate == 0 {
        return (frequencies, magnitudes_db);
    }
    let nyquist = sample_rate as f64 / 2.0;
    let step = (n_points - 1) as f64;
    for m in 0..n_points {
        let ratio = m as f64 / step; // 0..1
        let omega = std::f64::consts::PI * ratio;
        // A(e^{jω}) = 1 + Σ aₖ e^{-jωk}.
        let mut a_eval = Complex::new(1.0_f64, 0.0_f64);
        for (k0, &ak) in coeffs.iter().enumerate() {
            let k = (k0 + 1) as f64;
            let phase = -omega * k;
            a_eval += Complex::new(ak, 0.0) * Complex::new(phase.cos(), phase.sin());
        }
        let mag_a = a_eval.norm().max(1e-12);
        // Envelope magnitude is 1/|A|, i.e. -20·log10|A| in dB.
        frequencies.push((nyquist * ratio) as f32);
        magnitudes_db.push((-20.0 * mag_a.log10()) as f32);
    }
    (frequencies, magnitudes_db)
}

/// Indices of local maxima in `values` (strict left, non-strict right to be
/// robust to plateaus).
fn local_maxima(values: &[f32]) -> Vec<usize> {
    let mut peaks = Vec::new();
    if values.len() < 3 {
        return peaks;
    }
    for i in 1..values.len() - 1 {
        if values[i] > values[i - 1] && values[i] >= values[i + 1] {
            peaks.push(i);
        }
    }
    peaks
}

/// Indices of local minima in `values`.
fn local_minima(values: &[f32]) -> Vec<usize> {
    let mut valleys = Vec::new();
    if values.len() < 3 {
        return valleys;
    }
    for i in 1..values.len() - 1 {
        if values[i] < values[i - 1] && values[i] <= values[i + 1] {
            valleys.push(i);
        }
    }
    valleys
}

/// Estimate the −3 dB bandwidth (Hz) of the spectral-envelope peak at `peak_idx`
/// with linear interpolation of the crossing frequencies.
fn peak_bandwidth_hz(frequencies: &[f32], magnitudes_db: &[f32], peak_idx: usize) -> f32 {
    let threshold = magnitudes_db[peak_idx] - 3.0;

    // Walk left to the first bin at or below the threshold.
    let mut left = peak_idx;
    while left > 0 && magnitudes_db[left] > threshold {
        left -= 1;
    }
    let left_freq = if magnitudes_db[left] <= threshold && left < peak_idx {
        interpolate_crossing(
            frequencies[left],
            magnitudes_db[left],
            frequencies[left + 1],
            magnitudes_db[left + 1],
            threshold,
        )
    } else {
        frequencies[left]
    };

    // Walk right to the first bin at or below the threshold.
    let mut right = peak_idx;
    while right + 1 < magnitudes_db.len() && magnitudes_db[right] > threshold {
        right += 1;
    }
    let right_freq = if magnitudes_db[right] <= threshold && right > peak_idx {
        interpolate_crossing(
            frequencies[right - 1],
            magnitudes_db[right - 1],
            frequencies[right],
            magnitudes_db[right],
            threshold,
        )
    } else {
        frequencies[right]
    };

    (right_freq - left_freq).max(0.0)
}

/// Linear interpolation of the frequency at which the line through `(f0, m0)`
/// and `(f1, m1)` reaches `target`.
fn interpolate_crossing(f0: f32, m0: f32, f1: f32, m1: f32, target: f32) -> f32 {
    let span = m1 - m0;
    if span.abs() < f32::EPSILON {
        return f0;
    }
    f0 + (target - m0) / span * (f1 - f0)
}

/// Compute the −3 dB bandwidths (Hz) of the first three LPC spectral-envelope
/// peaks (F1/F2/F3). Always returns exactly three finite, positive values,
/// falling back to `100.0` Hz when peaks cannot be resolved.
pub(crate) fn compute_formant_bandwidths(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    let fallback = vec![100.0_f32, 100.0, 100.0];
    if sample_rate == 0 {
        return fallback;
    }
    let order = lpc_order(sample_rate);
    let coeffs = match lpc_coefficients(samples, order) {
        Some(coeffs) => coeffs,
        None => return fallback,
    };
    let (frequencies, magnitudes_db) = lpc_envelope_db(&coeffs, sample_rate, 512);
    if frequencies.len() < 3 {
        return fallback;
    }

    let peaks = local_maxima(&magnitudes_db);
    let mut bandwidths: Vec<f32> = peaks
        .iter()
        .take(3)
        .map(|&p| peak_bandwidth_hz(&frequencies, &magnitudes_db, p))
        .filter(|bw| bw.is_finite() && *bw > 0.0)
        .collect();

    // Guarantee exactly three finite, positive bandwidths.
    while bandwidths.len() < 3 {
        let fill = bandwidths.last().copied().unwrap_or(100.0);
        bandwidths.push(fill);
    }
    bandwidths.truncate(3);
    bandwidths
}

/// Formant clarity in `[0, 1]`: how prominently the (up to three) lowest
/// spectral-envelope peaks stand above their flanking valleys, normalized by a
/// reference prominence of ~20 dB.
pub(crate) fn compute_formant_clarity(samples: &[f32], sample_rate: u32) -> f32 {
    if sample_rate == 0 {
        return 0.0;
    }
    let order = lpc_order(sample_rate);
    let coeffs = match lpc_coefficients(samples, order) {
        Some(coeffs) => coeffs,
        None => return 0.0,
    };
    let (_, magnitudes_db) = lpc_envelope_db(&coeffs, sample_rate, 512);
    let peaks = local_maxima(&magnitudes_db);
    if peaks.is_empty() {
        return 0.0;
    }
    let valleys = local_minima(&magnitudes_db);

    let mut prominence_sum = 0.0f32;
    let mut counted = 0u32;
    for &peak in peaks.iter().take(3) {
        let left_valley = valleys
            .iter()
            .rev()
            .find(|&&v| v < peak)
            .map(|&v| magnitudes_db[v]);
        let right_valley = valleys
            .iter()
            .find(|&&v| v > peak)
            .map(|&v| magnitudes_db[v]);
        // Prominence above the higher (more constraining) flanking valley.
        let reference = match (left_valley, right_valley) {
            (Some(l), Some(r)) => l.max(r),
            (Some(l), None) => l,
            (None, Some(r)) => r,
            (None, None) => continue,
        };
        prominence_sum += (magnitudes_db[peak] - reference).max(0.0);
        counted += 1;
    }
    if counted == 0 {
        return 0.0;
    }
    (prominence_sum / counted as f32 / 20.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SAMPLE_RATE: u32 = 16_000;

    /// Deterministic pseudo-random noise in `[-1, 1)` (small LCG, no `rand`).
    fn lcg_noise(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = (state >> 33) as f32 / (1u64 << 31) as f32; // [0, 1)
                unit * 2.0 - 1.0
            })
            .collect()
    }

    /// A clean sine with an integer period (`freq` divides `sample_rate`).
    fn pure_sine(freq: f64, len: usize, amplitude: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let phase = 2.0 * std::f64::consts::PI * freq * i as f64 / TEST_SAMPLE_RATE as f64;
                (amplitude as f64 * phase.sin()) as f32
            })
            .collect()
    }

    /// A voiced signal built cycle-by-cycle with per-cycle period and amplitude
    /// perturbation (deterministic), so it stays voiced but exhibits jitter and
    /// shimmer.
    fn perturbed_voiced_signal(num_cycles: usize, base_period: f64) -> Vec<f32> {
        let noise = lcg_noise(0x5151_2323, num_cycles * 2);
        let mut out = Vec::new();
        for cycle in 0..num_cycles {
            let period_jitter = 1.0 + 0.08 * noise[cycle * 2] as f64;
            let amplitude = 0.5 * (1.0 + 0.25 * noise[cycle * 2 + 1] as f64);
            let period = (base_period * period_jitter).max(4.0);
            let cycle_len = period.round() as usize;
            for i in 0..cycle_len {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / period;
                out.push((amplitude * phase.sin()) as f32);
            }
        }
        out
    }

    #[test]
    fn test_clean_sine_low_jitter_shimmer() {
        // 200 Hz at 16 kHz → exactly 80 samples per period.
        let signal = pure_sine(200.0, TEST_SAMPLE_RATE as usize, 0.5);
        let (period, correlation) =
            estimate_pitch_period(&signal, TEST_SAMPLE_RATE).expect("sine is voiced");
        assert!(correlation >= VOICING_THRESHOLD);
        assert!((period - 80.0).abs() <= 2.0, "period was {period}");

        let (jitter, shimmer) = compute_jitter_shimmer(&signal, period);
        assert!(jitter < 0.05, "clean jitter should be ~0, got {jitter}");
        assert!(shimmer < 0.05, "clean shimmer should be ~0, got {shimmer}");
    }

    #[test]
    fn test_perturbed_signal_higher_jitter_shimmer() {
        let clean = pure_sine(200.0, TEST_SAMPLE_RATE as usize, 0.5);
        let perturbed = perturbed_voiced_signal(180, 80.0);

        let (clean_period, _) =
            estimate_pitch_period(&clean, TEST_SAMPLE_RATE).expect("clean voiced");
        let (clean_jitter, clean_shimmer) = compute_jitter_shimmer(&clean, clean_period);

        let (pert_period, pert_corr) =
            estimate_pitch_period(&perturbed, TEST_SAMPLE_RATE).expect("perturbed voiced");
        assert!(
            pert_corr >= VOICING_THRESHOLD,
            "perturbed should stay voiced"
        );
        let (pert_jitter, pert_shimmer) = compute_jitter_shimmer(&perturbed, pert_period);

        assert!(
            pert_jitter > clean_jitter,
            "perturbed jitter {pert_jitter} should exceed clean jitter {clean_jitter}"
        );
        assert!(
            pert_shimmer > clean_shimmer,
            "perturbed shimmer {pert_shimmer} should exceed clean shimmer {clean_shimmer}"
        );
    }

    #[test]
    fn test_lowpass_more_negative_tilt_than_white_noise() {
        let white = lcg_noise(0x1234_5678, TEST_SAMPLE_RATE as usize);

        // Strong one-pole low-pass (very low cutoff) → steep high-frequency rolloff.
        let alpha = 0.95f32;
        let mut low_pass = vec![0.0f32; white.len()];
        let mut previous = 0.0f32;
        for (i, &x) in white.iter().enumerate() {
            previous = alpha * previous + (1.0 - alpha) * x;
            low_pass[i] = previous;
        }

        let tilt_white = compute_spectral_tilt(&white, TEST_SAMPLE_RATE);
        let tilt_low_pass = compute_spectral_tilt(&low_pass, TEST_SAMPLE_RATE);

        assert!(tilt_white.is_finite() && tilt_low_pass.is_finite());
        assert!(
            tilt_low_pass < tilt_white,
            "low-pass tilt {tilt_low_pass} should be more negative than white tilt {tilt_white}"
        );
        assert!(tilt_low_pass < 0.0, "low-pass tilt should be negative");
    }

    #[test]
    fn test_formant_bandwidth_three_positive_values() {
        // Vowel-like signal: sum of three formant sinusoids (~/a/).
        let formants = [700.0_f64, 1220.0, 2600.0];
        let signal: Vec<f32> = (0..TEST_SAMPLE_RATE as usize)
            .map(|i| {
                let t = i as f64 / TEST_SAMPLE_RATE as f64;
                let value: f64 = formants
                    .iter()
                    .map(|&f| (2.0 * std::f64::consts::PI * f * t).sin())
                    .sum();
                (value / formants.len() as f64) as f32
            })
            .collect();

        let bandwidths = compute_formant_bandwidths(&signal, TEST_SAMPLE_RATE);
        assert_eq!(bandwidths.len(), 3);
        for bandwidth in bandwidths {
            assert!(
                bandwidth.is_finite() && bandwidth > 0.0,
                "bandwidth {bandwidth} must be finite and positive"
            );
        }

        // Clarity for a clearly-resonant signal should be meaningfully high.
        let clarity = compute_formant_clarity(&signal, TEST_SAMPLE_RATE);
        assert!(
            (0.0..=1.0).contains(&clarity) && clarity > 0.1,
            "formant clarity {clarity} should be a prominent value in [0, 1]"
        );
    }
}
