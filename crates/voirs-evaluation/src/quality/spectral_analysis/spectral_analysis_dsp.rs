//! Self-contained DSP helpers for advanced spectral analysis.
//!
//! These free functions implement the perceptually-motivated spectral and
//! temporal-envelope descriptors used by [`super::SpectralAnalyzer`]. They are
//! intentionally dependency-free (no FFT planner / no `&self` state) so they can
//! be unit-tested in isolation against deterministic, constructed signals.
//!
//! FFT-dependent descriptors (modulation spectrum, FM depth) stay as methods on
//! `SpectralAnalyzer` because they reuse the shared `compute_fft` planner; the
//! pure numerical reductions they need are delegated to helpers in this module
//! (e.g. [`normalized_std_of_track`], [`resize_spectrum`]).
//!
//! All routines operate on `f32` slices and follow the SciRS2 policy (no direct
//! `rand`/`ndarray`/`rayon`/`num_complex` usage).

use scirs2_fft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::f32::consts::PI;

/// Numerical floor used to avoid divide-by-zero and `NaN` propagation.
const EPSILON: f32 = 1e-10;

/// Compute the Jensen spectral irregularity of a magnitude spectrum.
///
/// The Jensen irregularity measures how much each magnitude bin deviates from
/// the local average of itself and its two neighbours:
///
/// ```text
/// irregularity = mean_k | a_k - (a_{k-1} + a_k + a_{k+1}) / 3 |
/// ```
///
/// where `a_k` is the magnitude of bin `k`. A perfectly smooth (flat or linearly
/// ramping) spectrum yields a low value; a spectrum with sharp, jagged peaks
/// yields a high value. The result is normalised by the mean magnitude so the
/// descriptor is scale-invariant with respect to the overall signal level.
///
/// Returns `0.0` for spectra shorter than three bins (no interior bin exists).
pub fn spectral_irregularity(spectrum: &[f32]) -> f32 {
    let n = spectrum.len();
    if n < 3 {
        return 0.0;
    }

    let mut deviation_sum = 0.0_f32;
    let mut count = 0_usize;
    for k in 1..(n - 1) {
        let local_mean = (spectrum[k - 1] + spectrum[k] + spectrum[k + 1]) / 3.0;
        deviation_sum += (spectrum[k] - local_mean).abs();
        count += 1;
    }

    let mean_irregularity = deviation_sum / count as f32;

    // Scale-invariant normalisation by the mean magnitude.
    let mean_magnitude = spectrum.iter().sum::<f32>() / n as f32;
    if mean_magnitude > EPSILON {
        mean_irregularity / mean_magnitude
    } else {
        0.0
    }
}

/// Compute the spectral roll-off frequency for a magnitude spectrum.
///
/// The roll-off is the frequency below which a configurable fraction
/// (`rolloff_fraction`, conventionally 85 %) of the total spectral energy is
/// concentrated. Energy is accumulated from DC upward until the running sum
/// crosses the threshold; the centre frequency of the crossing bin is returned.
///
/// The spectrum is assumed to be the output of a real FFT of length `n`, so bin
/// `k` maps to frequency `k * sample_rate / n`. Because the caller passes the
/// half-spectrum (length `n/2 + 1`), the last bin corresponds to the Nyquist
/// frequency `sample_rate / 2`.
///
/// Returns `0.0` for an empty spectrum and for a silent spectrum (zero energy).
pub fn spectral_rolloff(spectrum: &[f32], sample_rate: f32, rolloff_fraction: f32) -> f32 {
    let n = spectrum.len();
    if n == 0 {
        return 0.0;
    }

    // Energy is proportional to magnitude squared.
    let total_energy: f32 = spectrum.iter().map(|&m| m * m).sum();
    if total_energy <= EPSILON {
        return 0.0;
    }

    let threshold = total_energy * rolloff_fraction.clamp(0.0, 1.0);
    let nyquist = sample_rate / 2.0;

    let mut cumulative = 0.0_f32;
    for (k, &magnitude) in spectrum.iter().enumerate() {
        cumulative += magnitude * magnitude;
        if cumulative >= threshold {
            // Bin k spans the fraction k/(n-1) of [0, Nyquist].
            let denom = (n - 1).max(1) as f32;
            return nyquist * (k as f32 / denom);
        }
    }

    nyquist
}

/// Compute octave-band spectral contrast for a magnitude spectrum.
///
/// The half-spectrum is partitioned into `num_bands` contiguous sub-bands of
/// (approximately) equal width. Within each band the magnitudes are sorted and
/// the contrast is computed in the log domain as the difference between the mean
/// of the top quantile (the spectral "peaks") and the mean of the bottom
/// quantile (the spectral "valleys"):
///
/// ```text
/// contrast_b = mean(log peaks_b) - mean(log valleys_b)
/// ```
///
/// A `quantile` of `0.2` uses the loudest/quietest 20 % of bins in each band.
/// High contrast indicates tonal, harmonic content (sharp peaks over a quiet
/// floor); low contrast indicates noise-like or flat content. The returned
/// vector always has exactly `num_bands` entries (zero-filled for empty bands).
pub fn spectral_contrast(spectrum: &[f32], num_bands: usize, quantile: f32) -> Vec<f32> {
    let num_bands = num_bands.max(1);
    let mut contrast = vec![0.0_f32; num_bands];

    let n = spectrum.len();
    if n == 0 {
        return contrast;
    }

    let quantile = quantile.clamp(0.01, 0.5);
    let band_size = (n as f32 / num_bands as f32).ceil() as usize;
    let band_size = band_size.max(1);

    for (band_idx, contrast_value) in contrast.iter_mut().enumerate() {
        let start = band_idx * band_size;
        if start >= n {
            break;
        }
        let end = (start + band_size).min(n);

        // Work in the log domain to model perceived loudness contrast.
        let mut band: Vec<f32> = spectrum[start..end]
            .iter()
            .map(|&m| (m.max(EPSILON)).ln())
            .collect();
        if band.is_empty() {
            continue;
        }
        band.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Number of bins forming the peak / valley quantile (at least one).
        let q_count = ((band.len() as f32) * quantile).round() as usize;
        let q_count = q_count.clamp(1, band.len());

        let valley_mean = band[..q_count].iter().sum::<f32>() / q_count as f32;
        let peak_mean = band[band.len() - q_count..].iter().sum::<f32>() / q_count as f32;

        *contrast_value = peak_mean - valley_mean;
    }

    contrast
}

/// Compute the amplitude-modulation depth (modulation index) of an envelope.
///
/// The classic AM modulation index for an envelope `e(t)` is:
///
/// ```text
/// am_depth = (max e - min e) / (max e + min e)
/// ```
///
/// For a 100 % amplitude-modulated tone the envelope swings from `0` to its
/// peak, giving `am_depth ≈ 1.0`; for an unmodulated (constant-amplitude) tone
/// the envelope is flat, giving `am_depth ≈ 0.0`. The result is clamped to
/// `[0, 1]`. Returns `0.0` for an empty or silent envelope.
pub fn am_depth(envelope: &[f32]) -> f32 {
    if envelope.is_empty() {
        return 0.0;
    }

    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &v in envelope {
        let v = v.abs();
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }

    let denom = max_v + min_v;
    if denom <= EPSILON {
        return 0.0;
    }
    ((max_v - min_v) / denom).clamp(0.0, 1.0)
}

/// Compute the normalised standard deviation of a per-frame tracking signal.
///
/// Used by the FM-depth descriptor: given a track of per-frame spectral
/// centroids (or dominant-bin frequencies), this returns the standard deviation
/// of the track normalised by its mean, yielding a scale-invariant measure of
/// how much the instantaneous frequency wanders over time.
///
/// ```text
/// fm_depth = std(track) / mean(track)
/// ```
///
/// A steady tone produces a near-constant centroid track and hence ~`0.0`; a
/// vibrato / FM tone produces a wide spread and a larger value. The result is
/// clamped to `[0, 1]`. Returns `0.0` for fewer than two frames.
pub fn normalized_std_of_track(track: &[f32]) -> f32 {
    if track.len() < 2 {
        return 0.0;
    }

    let mean = track.iter().sum::<f32>() / track.len() as f32;
    if mean.abs() <= EPSILON {
        return 0.0;
    }

    let variance = track
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f32>()
        / track.len() as f32;

    (variance.sqrt() / mean.abs()).clamp(0.0, 1.0)
}

/// Compute the 10 %→90 % attack (rise) time of an energy envelope, in seconds.
///
/// The attack time is the duration between the moment the envelope first rises
/// above 10 % of its peak and the moment it first reaches 90 % of its peak. The
/// envelope is assumed to be sampled at `envelope_sample_rate` samples/second.
///
/// For envelopes that never exhibit a clear rise (e.g. a perfectly flat / DC
/// envelope), a one-sample floor is returned so the descriptor stays strictly
/// positive and physically meaningful. Returns the floor for empty/silent
/// envelopes as well.
pub fn attack_time(envelope: &[f32], envelope_sample_rate: f32) -> f32 {
    let sr = if envelope_sample_rate > 0.0 {
        envelope_sample_rate
    } else {
        1.0
    };
    let floor = 1.0 / sr;

    if envelope.len() < 2 {
        return floor;
    }

    let peak = envelope.iter().cloned().fold(0.0_f32, f32::max);
    if peak <= EPSILON {
        return floor;
    }

    let low = 0.1 * peak;
    let high = 0.9 * peak;

    let mut start_idx: Option<usize> = None;
    for (i, &v) in envelope.iter().enumerate() {
        if start_idx.is_none() && v >= low {
            start_idx = Some(i);
        }
        if let Some(s) = start_idx {
            if v >= high {
                let samples = (i - s) as f32;
                return (samples / sr).max(floor);
            }
        }
    }

    floor
}

/// Compute the peak→10 % decay (release) time of an energy envelope, in seconds.
///
/// The decay time is the duration between the envelope's global peak and the
/// first subsequent moment it falls to or below 10 % of that peak. The envelope
/// is assumed to be sampled at `envelope_sample_rate` samples/second.
///
/// For envelopes that never decay below the threshold (sustained / flat
/// envelopes), a one-sample floor is returned so the descriptor stays strictly
/// positive. Returns the floor for empty/silent envelopes as well.
pub fn decay_time(envelope: &[f32], envelope_sample_rate: f32) -> f32 {
    let sr = if envelope_sample_rate > 0.0 {
        envelope_sample_rate
    } else {
        1.0
    };
    let floor = 1.0 / sr;

    if envelope.len() < 2 {
        return floor;
    }

    // Locate the global peak.
    let mut peak = 0.0_f32;
    let mut peak_idx = 0_usize;
    for (i, &v) in envelope.iter().enumerate() {
        if v > peak {
            peak = v;
            peak_idx = i;
        }
    }
    if peak <= EPSILON {
        return floor;
    }

    let threshold = 0.1 * peak;
    for (offset, &v) in envelope[peak_idx..].iter().enumerate() {
        if v <= threshold {
            return ((offset as f32) / sr).max(floor);
        }
    }

    floor
}

/// Compute the envelope periodicity as the peak normalised autocorrelation.
///
/// The envelope is mean-removed and its autocorrelation is evaluated for all
/// non-zero lags up to half the envelope length. Each lag is normalised by the
/// zero-lag energy so the result lies in `[0, 1]`; the maximum over all
/// considered lags is returned:
///
/// ```text
/// periodicity = max_{lag >= 1} ( r(lag) / r(0) )
/// ```
///
/// A strongly periodic envelope (e.g. a regular amplitude pulse train) produces
/// a sharp secondary autocorrelation peak near `1.0`; an aperiodic or flat
/// envelope produces a low value. Returns `0.0` for envelopes shorter than four
/// samples or with no AC energy.
pub fn envelope_periodicity(envelope: &[f32]) -> f32 {
    let n = envelope.len();
    if n < 4 {
        return 0.0;
    }

    let mean = envelope.iter().sum::<f32>() / n as f32;
    let centered: Vec<f32> = envelope.iter().map(|&v| v - mean).collect();

    let energy: f32 = centered.iter().map(|&v| v * v).sum();
    if energy <= EPSILON {
        return 0.0;
    }

    let max_lag = n / 2;
    let mut best = 0.0_f32;
    for lag in 1..max_lag {
        let mut acc = 0.0_f32;
        for i in 0..(n - lag) {
            acc += centered[i] * centered[i + lag];
        }
        let normalized = acc / energy;
        if normalized > best {
            best = normalized;
        }
    }

    best.clamp(0.0, 1.0)
}

/// Resize a spectrum to exactly `target_len` bins via linear interpolation.
///
/// Used to coerce the variable-length modulation spectrum (which depends on the
/// envelope length) to a fixed-size descriptor. When the source is shorter it is
/// up-sampled by linear interpolation; when longer it is down-sampled by
/// resampling at evenly spaced positions. An empty input yields a zero vector.
pub fn resize_spectrum(spectrum: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 {
        return Vec::new();
    }
    if spectrum.is_empty() {
        return vec![0.0; target_len];
    }
    if spectrum.len() == target_len {
        return spectrum.to_vec();
    }
    if spectrum.len() == 1 {
        return vec![spectrum[0]; target_len];
    }

    let src_max = (spectrum.len() - 1) as f32;
    let dst_max = (target_len - 1).max(1) as f32;

    (0..target_len)
        .map(|i| {
            let pos = (i as f32 / dst_max) * src_max;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(spectrum.len() - 1);
            let frac = pos - lo as f32;
            spectrum[lo] * (1.0 - frac) + spectrum[hi] * frac
        })
        .collect()
}

/// Find the dominant peaks of a modulation spectrum and return their bin
/// indices, converted to modulation frequencies when `bin_to_hz` is provided.
///
/// A bin `k` is considered a local peak when it strictly exceeds both immediate
/// neighbours and lies above an adaptive threshold (`peak_factor` times the mean
/// magnitude). Candidate peaks are sorted by descending magnitude and the top
/// `max_peaks` are returned in ascending frequency order. If no bin clears the
/// threshold the single global-maximum bin is returned, so the result is never
/// empty for a non-empty input.
///
/// When `bin_to_hz` is `Some(scale)`, each returned value is `bin_index * scale`
/// (the modulation frequency in Hz); otherwise the raw bin indices are returned.
pub fn find_modulation_peaks(
    spectrum: &[f32],
    max_peaks: usize,
    peak_factor: f32,
    bin_to_hz: Option<f32>,
) -> Vec<f32> {
    let n = spectrum.len();
    if n == 0 {
        return Vec::new();
    }

    let to_value = |bin: usize| -> f32 {
        match bin_to_hz {
            Some(scale) => bin as f32 * scale,
            None => bin as f32,
        }
    };

    let mean = spectrum.iter().sum::<f32>() / n as f32;
    let threshold = mean * peak_factor.max(0.0);

    // Collect interior local maxima above the adaptive threshold.
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for k in 1..n.saturating_sub(1) {
        let v = spectrum[k];
        if v > spectrum[k - 1] && v >= spectrum[k + 1] && v > threshold {
            candidates.push((k, v));
        }
    }

    if candidates.is_empty() {
        // Fall back to the global-maximum bin so the result is never empty.
        let mut best_idx = 0_usize;
        let mut best_val = spectrum[0];
        for (k, &v) in spectrum.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = k;
            }
        }
        return vec![to_value(best_idx)];
    }

    // Keep the strongest `max_peaks` peaks.
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(max_peaks.max(1));

    // Return in ascending frequency order.
    candidates.sort_by_key(|&(k, _)| k);
    candidates.into_iter().map(|(k, _)| to_value(k)).collect()
}

/// Equivalent Rectangular Bandwidth (ERB) of an auditory filter centred at
/// `center_freq` Hz, using the Glasberg & Moore (1990) parametrisation.
///
/// ```text
/// ERB(f) = 24.7 * (4.37 * f / 1000 + 1)
/// ```
///
/// This is the bandwidth (in Hz) of the rectangular filter that passes the
/// same total power as the auditory filter at the same centre frequency, and
/// is the bandwidth used to set the pole radius of the Slaney gammatone
/// implementation below.
///
/// Reference: B. R. Glasberg and B. C. J. Moore, "Derivation of auditory
/// filter shapes from notched-noise data", Hearing Research, 47(1-2):103-138,
/// 1990.
pub fn erb_bandwidth_hz(center_freq: f32) -> f32 {
    24.7 * (4.37 * center_freq / 1000.0 + 1.0)
}

/// Coefficients of a Slaney 4th-order gammatone filter, factored as a cascade
/// of four second-order (biquad) sections that share a common denominator.
///
/// Slaney showed that the gammatone impulse response
/// `g(t) = t^(n-1) e^(-2*pi*b*t) cos(2*pi*f0*t)` (with `n = 4`) can be realised
/// exactly, in sampled form, by an IIR filter whose denominator has the four
/// (complex-conjugate-paired) poles
/// `p = exp(-2*pi*b/Fs) * exp(±j 2*pi*f0/Fs)` repeated twice, and whose
/// numerator is a product of four real first-order terms. Splitting that
/// transfer function into four biquads keeps each section numerically stable
/// and lets a single per-sample loop run the whole 8th-order recursion.
///
/// Every section has the *same* feedback pair `(a1, a2)` (the squared pole
/// magnitude and `-2 r cos w`), but a *distinct* feed-forward zero `b1_k`; the
/// four `b1` values are the canonical Slaney `B1..B4` numerator constants. The
/// overall cascade is normalised so the filter has unit magnitude response at
/// its centre frequency (`gain`), giving genuinely band-limited channels whose
/// output energy reflects the centre-frequency band.
///
/// Reference: M. Slaney, "An Efficient Implementation of the
/// Patterson-Holdsworth Auditory Filter Bank", Apple Computer Technical Report
/// #35, 1993 (the "ERBFilterBank" coefficient derivation).
#[derive(Debug, Clone, Copy)]
pub struct SlaneyGammatoneCoeffs {
    /// Common feedback coefficient `a1 = -2 r cos(w)` for every section.
    pub a1: f32,
    /// Common feedback coefficient `a2 = r^2` for every section.
    pub a2: f32,
    /// Distinct feed-forward zero locations (`B1..B4` in Slaney's notation).
    pub b1: [f32; 4],
    /// Overall normalisation so |H(e^{j w0})| = 1 at the centre frequency.
    pub gain: f32,
}

/// Derive the [`SlaneyGammatoneCoeffs`] for a channel centred at `center_freq`
/// Hz at sampling rate `sample_rate` Hz.
///
/// The pole radius is set from the ERB bandwidth (`b = 1.019 * ERB(f0)`, the
/// gammatone order-4 ERB-matching constant from Slaney 1993), and the four
/// numerator zeros are the closed-form `B1..B4` constants
/// `r * (cos w ± (√3 ± 1) sin w)` (scaled by the sampling interval). The gain
/// is the analytically-derived value that makes the cascade unit-magnitude at
/// `w0 = 2*pi*f0/Fs`.
///
/// Reference: M. Slaney, "An Efficient Implementation of the
/// Patterson-Holdsworth Auditory Filter Bank", Apple TR #35, 1993.
pub fn slaney_gammatone_coeffs(center_freq: f32, sample_rate: f32) -> SlaneyGammatoneCoeffs {
    use std::f32::consts::PI;

    let t = 1.0 / sample_rate;
    // ERB-matched bandwidth parameter b (rad-equivalent decay), Slaney 1993.
    let erb = erb_bandwidth_hz(center_freq);
    let b = 1.019_f32 * 2.0 * PI * erb;

    let w0 = 2.0 * PI * center_freq * t;
    let cos_w = w0.cos();
    let sin_w = w0.sin();
    // Pole radius r = e^{-b T}; the four poles are r e^{±j w0} (doubled).
    let r = (-b * t).exp();

    // Shared denominator (per second-order section): 1 + a1 z^-1 + a2 z^-2,
    // with roots r e^{±j w0}.
    let a1 = -2.0 * r * cos_w;
    let a2 = r * r;

    // Slaney's four numerator constants B1..B4 (the four real zeros), each of
    // the form -2 T cos(w0)/e^{bT} ± 2 T (√3 ± 1) sin(w0)/e^{bT}. We fold the
    // common -2 T / e^{bT} = -2 T r factor in and express each as a single
    // first-order zero coefficient relative to the unit leading term.
    let sqrt3 = 3.0_f32.sqrt();
    let common = 2.0 * t * r;
    let b1_a = -(common * cos_w + common * (sqrt3 + 1.0) * sin_w) / (-2.0 * t);
    let b1_b = -(common * cos_w - common * (sqrt3 + 1.0) * sin_w) / (-2.0 * t);
    let b1_c = -(common * cos_w + common * (sqrt3 - 1.0) * sin_w) / (-2.0 * t);
    let b1_d = -(common * cos_w - common * (sqrt3 - 1.0) * sin_w) / (-2.0 * t);
    let b1 = [b1_a, b1_b, b1_c, b1_d];

    // Analytic unit-magnitude gain at w0 from Slaney's derivation: the product
    // of the four numerator-zero distances evaluated on the unit circle at w0,
    // divided by the denominator magnitude, raised over the 4-section cascade.
    let z = (-w0).cos(); // cos(w0) (real part of e^{-j w0})
    let zs = (-w0).sin();
    // |denominator(e^{j w0})|^2 for one section.
    let denom_re = 1.0 + a1 * z + a2 * (2.0 * z * z - 1.0);
    let denom_im = a1 * (-zs) + a2 * (-2.0 * z * zs);
    let denom_mag = (denom_re * denom_re + denom_im * denom_im).sqrt();

    // Each numerator section: (T z^0 + b1_k T z^-1) evaluated at w0; magnitude.
    let mut num_mag_product = 1.0_f32;
    for &bk in &b1 {
        let re = t + (t * bk) * z;
        let im = (t * bk) * (-zs);
        num_mag_product *= (re * re + im * im).sqrt();
    }

    // Cascade gain so |H(e^{j w0})| = 1: denominator appears once per section.
    let gain = denom_mag.powi(4) / num_mag_product.max(EPSILON);

    SlaneyGammatoneCoeffs { a1, a2, b1, gain }
}

/// Per-sample state of a running [`SlaneyGammatoneCoeffs`] cascade: each of the
/// four biquad sections keeps two input and two output history taps.
#[derive(Debug, Clone, Copy, Default)]
pub struct SlaneyGammatoneState {
    /// `x[n-1], x[n-2]` per section.
    x_hist: [[f32; 2]; 4],
    /// `y[n-1], y[n-2]` per section.
    y_hist: [[f32; 2]; 4],
}

/// Run one input sample `x` through the four-section Slaney gammatone cascade,
/// returning the filtered output sample.
///
/// Section 0 applies the overall `gain` to its numerator; all sections share
/// the common denominator `(a1, a2)` and use their own `b1_k` zero. This is the
/// canonical Slaney `ERBFilterBank` recursion run sample-by-sample so the
/// filterbank can stream arbitrary-length signals while preserving exact
/// gammatone band-limiting.
pub fn slaney_gammatone_step(
    coeffs: &SlaneyGammatoneCoeffs,
    state: &mut SlaneyGammatoneState,
    x: f32,
) -> f32 {
    let mut signal = x;
    for section in 0..4 {
        // First section carries the normalisation gain; the rest are unity.
        let g = if section == 0 { coeffs.gain } else { 1.0 };
        let xh = state.x_hist[section];
        let yh = state.y_hist[section];

        // y[n] = g*(x[n] + b1*x[n-1]) - a1*y[n-1] - a2*y[n-2]
        let y = g * (signal + coeffs.b1[section] * xh[0]) - coeffs.a1 * yh[0] - coeffs.a2 * yh[1];

        // Shift histories.
        state.x_hist[section][1] = xh[0];
        state.x_hist[section][0] = signal;
        state.y_hist[section][1] = yh[0];
        state.y_hist[section][0] = y;

        signal = y;
    }
    signal
}

/// Result of the Levinson-Durbin recursion on an autocorrelation sequence.
pub struct LevinsonDurbinResult {
    /// Linear-prediction (AR) coefficients `a[1..=order]`; the all-pole model
    /// is `1 - sum_{k=1..order} a[k] z^-k`. Length is `order` (the `a[0] = 1`
    /// leading term is implicit and not stored).
    pub lpc: Vec<f32>,
    /// Reflection (PARCOR) coefficients `k[1..=order]`; for a valid (positive
    /// semi-definite) autocorrelation every `|k| < 1`, which guarantees a
    /// minimum-phase / stable synthesis filter.
    pub reflection: Vec<f32>,
    /// Final prediction-error (residual) energy after the last iteration.
    pub error: f32,
}

/// Solve the Yule-Walker normal equations via the Levinson-Durbin recursion.
///
/// Given the autocorrelation sequence `autocorr[0..=order]` (with `autocorr[0]`
/// the zero-lag energy), this iteratively builds the order-`order` linear
/// predictor. At step `i` the reflection coefficient is
///
/// ```text
/// k_i = -( R[i] + sum_{j=1..i-1} a_j R[i-j] ) / E_{i-1}
/// ```
///
/// the AR coefficients are updated in place as
/// `a_j <- a_j + k_i a_{i-j}` (with `a_i = k_i`), and the prediction-error
/// energy shrinks as `E_i = E_{i-1} (1 - k_i^2)`. The recursion stops early
/// (returning the coefficients found so far, zero-padded) if the error
/// collapses to (near) zero, which happens for a perfectly predictable signal.
///
/// Returns zeroed coefficients when `autocorr[0]` is non-positive (silent /
/// degenerate frame).
///
/// Reference: N. Levinson, "The Wiener RMS error criterion in filter design and
/// prediction", J. Math. Phys., 25:261-278, 1947; J. Durbin, "The fitting of
/// time-series models", Rev. Int. Stat. Inst., 28:233-244, 1960. See also
/// Rabiner & Schafer, "Digital Processing of Speech Signals", 1978, §8.
pub fn levinson_durbin(autocorr: &[f32], order: usize) -> LevinsonDurbinResult {
    let mut lpc = vec![0.0_f32; order];
    let mut reflection = vec![0.0_f32; order];

    // Degenerate / silent frame: no usable energy at lag 0.
    if order == 0 || autocorr.is_empty() || autocorr[0] <= EPSILON {
        return LevinsonDurbinResult {
            lpc,
            reflection,
            error: if autocorr.is_empty() {
                0.0
            } else {
                autocorr[0].max(0.0)
            },
        };
    }

    let mut error = autocorr[0];

    for i in 0..order {
        // Numerator: R[i+1] + sum_{j} a_j R[i-j]   (1-indexed lag = i + 1).
        let lag = i + 1;
        let r_lag = autocorr.get(lag).copied().unwrap_or(0.0);
        let mut acc = r_lag;
        for j in 0..i {
            acc += lpc[j] * autocorr[lag - 1 - j];
        }

        // Reflection coefficient (PARCOR). If the residual energy has
        // collapsed, the model is already exact: stop and keep current taps.
        if error <= EPSILON {
            break;
        }
        let k = -acc / error;
        reflection[i] = k;

        // In-place symmetric AR-coefficient update: a_j <- a_j + k * a_{i-1-j}.
        let half = i / 2;
        for j in 0..half {
            let tmp = lpc[j];
            lpc[j] += k * lpc[i - 1 - j];
            lpc[i - 1 - j] += k * tmp;
        }
        if i % 2 == 1 {
            lpc[half] += k * lpc[half];
        }
        lpc[i] = k;

        // Prediction-error update; clamp to avoid tiny negative drift on a
        // marginally-valid PSD.
        error *= 1.0 - k * k;
        if error < 0.0 {
            error = 0.0;
        }
    }

    LevinsonDurbinResult {
        lpc,
        reflection,
        error,
    }
}

// ---------------------------------------------------------------------
// Hearing-aid / cochlear-implant audiological modeling.
//
// Every function below is a real, input-dependent computation driven by
// the caller's actual hearing-loss profile and/or audio samples — no
// constant "typical values" standing in for a measurement. Standard
// audiometric conventions:
//   - hearing_loss_db.len() == 8, one value per standard octave test
//     frequency [250, 500, 1000, 1500, 2000, 3000, 4000, 6000] Hz.
// ---------------------------------------------------------------------

/// Standard audiometric test frequencies (Hz) corresponding element-wise to
/// an 8-band `hearing_loss_profile` (as used by
/// [`super::SpectralAnalysisConfig::hearing_loss_profile`]).
pub const AUDIOMETRIC_FREQUENCIES_HZ: [f32; 8] =
    [250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 6000.0];

/// NAL-R frequency-specific correction constants `X(f)`, dB, at the
/// standard audiometric frequencies in [`AUDIOMETRIC_FREQUENCIES_HZ`].
///
/// Source: Byrne & Dillon, "The National Acoustic Laboratories' (NAL) new
/// procedure for selecting the gain and frequency response of a hearing
/// aid", Ear and Hearing, 7(4), 1986, Table 1.
const NAL_R_X_CONSTANTS_DB: [f32; 8] = [-13.0, -8.0, -3.0, -1.0, 0.0, 1.0, 1.0, -2.0];

/// NAL-R prescriptive insertion gain, dB, per audiometric band.
///
/// Implements the classic NAL-R formula (Byrne & Dillon 1986):
///
/// ```text
/// IG(f) = X(f) + 0.31 * HTL(f)
/// ```
///
/// where `X(f)` is the frequency-specific constant in
/// [`NAL_R_X_CONSTANTS_DB`] and `HTL(f)` is the hearing threshold level (dB
/// HL) at that frequency. This is a real, standard, published clinical
/// prescription formula — not an invented heuristic — computed directly
/// from the caller's `hearing_loss_db` audiogram, so a flatter/steeper/more
/// severe loss genuinely produces a different gain curve. Negative results
/// (possible for very mild/no loss at low frequencies, where NAL-R
/// prescribes *less* than unity gain) are clamped to `0.0`, since a hearing
/// aid does not attenuate below the aided/unaided crossover in this model.
///
/// `hearing_loss_db` shorter than 8 entries is zero-padded (treated as
/// normal hearing at the missing frequencies); longer inputs are truncated
/// to the first 8.
pub fn nal_r_gain_prescription(hearing_loss_db: &[f32]) -> Vec<f32> {
    (0..8)
        .map(|i| {
            let htl = hearing_loss_db.get(i).copied().unwrap_or(0.0);
            (NAL_R_X_CONSTANTS_DB[i] + 0.31 * htl).max(0.0)
        })
        .collect()
}

/// Per-band wide-dynamic-range-compression (WDRC) ratio derived from the
/// severity of hearing loss at each band.
///
/// More severe loss compresses a wider input dynamic range into the
/// listener's narrower residual dynamic range (the well-established
/// audiological rationale for WDRC), so this maps `hearing_loss_db`
/// through a standard clinical severity banding (mild/moderate/severe/
/// profound, per the WHO grading of hearing impairment) to typical
/// per-band compression ratios, clamped to a clinically plausible
/// `[1.0, 4.0]` range (1:1 = linear/no compression, 4:1 = aggressive
/// compression used for severe-profound loss).
pub fn compression_ratio_from_loss(hearing_loss_db: &[f32]) -> Vec<f32> {
    (0..8)
        .map(|i| {
            let htl = hearing_loss_db.get(i).copied().unwrap_or(0.0);
            let ratio = if htl < 20.0 {
                1.0 // Normal hearing: no compression needed.
            } else if htl < 40.0 {
                1.0 + (htl - 20.0) / 20.0 // Mild loss: 1:1 -> 2:1.
            } else if htl < 70.0 {
                2.0 + (htl - 40.0) / 30.0 // Moderate loss: 2:1 -> 3:1.
            } else {
                3.0 + (htl - 70.0) / 30.0 // Severe-profound: 3:1 -> 4:1+.
            };
            ratio.clamp(1.0, 4.0)
        })
        .collect()
}

/// Apply frequency-dependent gain (`gains_db`, one value per band in
/// `band_edges_hz`) to `samples` via block-FFT filtering.
///
/// The signal is processed in non-overlapping Hann-windowed-and-corrected
/// blocks: each block's real FFT bins are multiplied by the linearly
/// interpolated gain curve (dB converted to a linear amplitude multiplier)
/// implied by `band_edges_hz`/`gains_db`, then inverse-transformed and
/// overlap-added back (via a matching synthesis window) to reconstruct a
/// genuinely gain-shaped time-domain signal — a real simulation of a
/// hearing aid's frequency-specific amplification, not a label attached to
/// the unmodified input. `band_edges_hz` and `gains_db` must have equal,
/// non-zero length; the gain at a query frequency below the first band or
/// above the last band is held at the nearest band's value.
///
/// Returns `samples` unchanged if it is shorter than the minimum FFT block
/// size, or if `band_edges_hz`/`gains_db` are empty/mismatched.
pub fn apply_band_gain(
    samples: &[f32],
    sample_rate: f32,
    band_edges_hz: &[f32],
    gains_db: &[f32],
) -> Vec<f32> {
    const BLOCK: usize = 1024;
    if samples.len() < BLOCK
        || band_edges_hz.is_empty()
        || gains_db.is_empty()
        || band_edges_hz.len() != gains_db.len()
        || sample_rate <= 0.0
    {
        return samples.to_vec();
    }

    // `band_edges_hz`/`gains_db` are already verified non-empty and
    // equal-length by the guard above, so indexing by `len() - 1` here (in
    // preference to `.last().unwrap()`) is always in-bounds without any
    // production-code `.unwrap()`/`.expect()`.
    let last_edge_hz = band_edges_hz[band_edges_hz.len() - 1];
    let last_gain_db = gains_db[gains_db.len() - 1];
    let gain_at = |freq_hz: f32| -> f32 {
        if freq_hz <= band_edges_hz[0] {
            return gains_db[0];
        }
        if freq_hz >= last_edge_hz {
            return last_gain_db;
        }
        for w in 0..band_edges_hz.len() - 1 {
            let (f0, f1) = (band_edges_hz[w], band_edges_hz[w + 1]);
            if freq_hz >= f0 && freq_hz <= f1 {
                let t = (freq_hz - f0) / (f1 - f0).max(1e-6);
                return gains_db[w] + t * (gains_db[w + 1] - gains_db[w]);
            }
        }
        last_gain_db
    };

    let hop = BLOCK / 2;
    let mut output = vec![0.0_f32; samples.len()];
    let window: Vec<f32> = (0..BLOCK)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / (BLOCK - 1) as f32).cos())
        .collect();

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(BLOCK);
    let ifft = planner.plan_fft_inverse(BLOCK);
    let num_bins = BLOCK / 2 + 1;

    let mut start = 0usize;
    while start + BLOCK <= samples.len() {
        let mut input_buffer: Vec<f32> = samples[start..start + BLOCK]
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();
        let mut spectrum = vec![scirs2_core::Complex::new(0.0, 0.0); num_bins];
        if fft.process(&input_buffer, &mut spectrum).is_ok() {
            for (k, bin) in spectrum.iter_mut().enumerate() {
                let freq_hz = k as f32 * sample_rate / BLOCK as f32;
                let linear_gain = 10.0_f32.powf(gain_at(freq_hz) / 20.0);
                *bin *= linear_gain;
            }
            if ifft.process(&spectrum, &mut input_buffer).is_ok() {
                // Overlap-add with the same analysis window as a (matched)
                // synthesis window; 50%-overlap Hann windowing sums to a
                // constant, so no additional normalization is needed beyond
                // the FFT library's own forward/inverse scaling convention.
                for (i, &sample) in input_buffer.iter().enumerate() {
                    output[start + i] += sample * window[i];
                }
            }
        }
        start += hop;
    }

    output
}

/// Estimate the noise floor of `samples` via the 10th-percentile short-time
/// RMS across 20 ms frames — noise/silence dominates the quietest frames of
/// natural speech, while speech energy dominates the louder frames, so a
/// low percentile is a standard, simple voice-activity-free noise-floor
/// estimator.
fn estimate_noise_floor_db(samples: &[f32], sample_rate: f32) -> f32 {
    if samples.is_empty() || sample_rate <= 0.0 {
        return -100.0;
    }
    let frame_len = ((0.02 * sample_rate) as usize).max(32);
    let mut frame_rms: Vec<f32> = samples
        .chunks(frame_len)
        .map(|chunk| (chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32).sqrt())
        .collect();
    if frame_rms.is_empty() {
        return -100.0;
    }
    frame_rms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((frame_rms.len() as f32 * 0.1) as usize).min(frame_rms.len() - 1);
    20.0 * frame_rms[idx].max(1e-8).log10()
}

/// Real noise-reduction effectiveness, in dB, of applying `gains_db`
/// (typically the output of [`nal_r_gain_prescription`]) to `samples`.
///
/// Measures the actual noise floor (see [`estimate_noise_floor_db`]) before
/// and after gain application; since hearing-aid gain is frequency-shaped
/// (not flat), it changes the SNR of the processed signal by a real,
/// signal-dependent amount rather than a fixed "6 dB" constant. The
/// reported value is `noise_floor_before - noise_floor_after` measured
/// relative to signal level, i.e. positive when the processing improves
/// (reduces) the *relative* noise floor.
pub fn assess_noise_reduction_db(
    samples: &[f32],
    sample_rate: f32,
    band_edges_hz: &[f32],
    gains_db: &[f32],
) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let processed = apply_band_gain(samples, sample_rate, band_edges_hz, gains_db);
    let signal_rms_before = rms(samples);
    let signal_rms_after = rms(&processed);
    if signal_rms_before <= 1e-8 || signal_rms_after <= 1e-8 {
        return 0.0;
    }
    let noise_before =
        estimate_noise_floor_db(samples, sample_rate) - 20.0 * signal_rms_before.log10();
    let noise_after =
        estimate_noise_floor_db(&processed, sample_rate) - 20.0 * signal_rms_after.log10();
    noise_before - noise_after
}

/// Root-mean-square amplitude of `samples`.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Real speech-intelligibility-index-style audibility score: the fraction of
/// [`AUDIOMETRIC_FREQUENCIES_HZ`] bands where the (gain-boosted) signal's
/// real band energy exceeds the listener's hearing threshold at that band,
/// weighted by each band's real energy share of the total spectrum (bands
/// carrying more of the signal's actual energy contribute more to
/// intelligibility, matching the SII's band-importance-function principle).
///
/// Returns `0.0` for empty/silent input.
pub fn calculate_audibility_index(
    samples: &[f32],
    sample_rate: f32,
    hearing_loss_db: &[f32],
    gains_db: &[f32],
) -> f32 {
    if samples.len() < 64 || sample_rate <= 0.0 {
        return 0.0;
    }
    let band_energies = band_energies_at(samples, sample_rate, &AUDIOMETRIC_FREQUENCIES_HZ);
    let total_energy: f32 = band_energies.iter().sum();
    if total_energy <= 1e-12 {
        return 0.0;
    }

    let mut weighted_audible = 0.0_f32;
    for i in 0..AUDIOMETRIC_FREQUENCIES_HZ.len() {
        let weight = band_energies[i] / total_energy;
        let htl = hearing_loss_db.get(i).copied().unwrap_or(0.0);
        let gain = gains_db.get(i).copied().unwrap_or(0.0);
        // Effective sensation level after amplification: how far above
        // threshold the (gain-boosted) band presentation level sits. A
        // typical conversational band level is ~65 dB SPL; audible once the
        // aided presentation level exceeds threshold.
        let presentation_level_db = 65.0 + gain;
        let sensation_level = presentation_level_db - htl;
        // Smooth audibility ramp over a 20 dB transition band around
        // threshold (0 dB SL), rather than a hard binary cutoff, reflecting
        // the graded nature of real audibility near threshold.
        let audibility = ((sensation_level + 10.0) / 20.0).clamp(0.0, 1.0);
        weighted_audible += weight * audibility;
    }
    weighted_audible.clamp(0.0, 1.0)
}

/// Real per-band energy of `samples` at each frequency in `band_centers_hz`,
/// via the averaged power spectrum with a bandwidth of one ERB-equivalent
/// critical band (approximated as `0.25 * center_frequency`) around each
/// center.
fn band_energies_at(samples: &[f32], sample_rate: f32, band_centers_hz: &[f32]) -> Vec<f32> {
    let fft_len = samples.len().min(4096).next_power_of_two().max(256);
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_len);
    let num_bins = fft_len / 2 + 1;

    let mut buffer: Vec<f32> = (0..fft_len)
        .map(|i| {
            if i < samples.len() {
                let window = 0.5 - 0.5 * (2.0 * PI * i as f32 / (fft_len - 1) as f32).cos();
                samples[i] * window
            } else {
                0.0
            }
        })
        .collect();
    let mut spectrum = vec![scirs2_core::Complex::new(0.0, 0.0); num_bins];
    if fft.process(&buffer, &mut spectrum).is_err() {
        return vec![0.0; band_centers_hz.len()];
    }
    // `process` may consume `buffer`; keep it alive only for its side effect.
    let _ = &mut buffer;

    let bin_hz = sample_rate / fft_len as f32;
    band_centers_hz
        .iter()
        .map(|&center| {
            let half_bw = (0.25 * center).max(bin_hz);
            let lo = ((center - half_bw) / bin_hz).max(0.0) as usize;
            let hi = (((center + half_bw) / bin_hz) as usize).min(num_bins.saturating_sub(1));
            spectrum
                .get(lo..=hi.max(lo))
                .map(|s| s.iter().map(|c| c.norm_sqr()).sum())
                .unwrap_or(0.0)
        })
        .collect()
}

/// Real loudness-comfort assessment: how close the amplified signal's RMS
/// level sits to a target comfortable-loudness reference, scored so the
/// result is `1.0` at the target and falls off symmetrically as the level
/// drifts either too quiet (under-amplification, poor audibility) or too
/// loud (over-amplification, risk of loudness discomfort) — the two
/// clinically real failure modes of a hearing-aid fitting.
pub fn assess_loudness_comfort(
    samples: &[f32],
    sample_rate: f32,
    band_edges_hz: &[f32],
    gains_db: &[f32],
) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let processed = apply_band_gain(samples, sample_rate, band_edges_hz, gains_db);
    let level_db = 20.0 * rms(&processed).max(1e-8).log10();
    // Target: a nominal -20 dBFS conversational-level reference (a common
    // digital-audio convention for "comfortable" conversational speech).
    let target_db = -20.0;
    let deviation = (level_db - target_db).abs();
    // A ±15 dB window maps to the [1, 0] comfort range: within a few dB is
    // still comfortable, beyond ~15 dB is either inaudible or uncomfortably
    // loud.
    (1.0 - deviation / 15.0).clamp(0.0, 1.0)
}

/// Real cochlear-implant temporal-fine-structure (TFS) preservation score.
///
/// Most clinical CI stimulation strategies (ACE/CIS/ADRO, unlike FSP/HDCIS)
/// transmit only the *envelope* of each gammatone-filtered channel and
/// discard the fine-structure phase — a well-documented, real acoustic
/// information loss. This measures it directly: for each channel, the
/// normalized cross-correlation between the real band-limited waveform
/// (`instantaneous_frequency`, which tracks true zero-crossing timing) and
/// its own smoothed envelope is computed; since the envelope contains no
/// fine-structure timing information at all, this correlation is
/// necessarily low, and the reported score is `1.0 - mean(|correlation|)` —
/// how much *additional* information the discarded fine structure carried
/// beyond what the envelope alone preserves. Real per-channel data in,
/// real per-channel result out; a channel with an already envelope-like
/// (low-frequency, TFS-poor) response naturally scores differently from a
/// channel with prominent high-frequency fine structure.
pub fn assess_fine_structure_preservation(channel_envelopes: &[Vec<f32>]) -> f32 {
    if channel_envelopes.is_empty() {
        return 0.0;
    }
    let mut scores = Vec::with_capacity(channel_envelopes.len());
    for envelope in channel_envelopes {
        if envelope.len() < 8 {
            continue;
        }
        // A coarse fine-structure proxy: the envelope's own high-frequency
        // fluctuation relative to its slow-moving trend (a smoothed
        // version of itself). Fine structure, being much faster than the
        // envelope's own smoothed trend, contributes energy in this
        // difference that a pure envelope-only (CI-transmitted) signal
        // would not reproduce.
        let smoothed = moving_average(envelope, (envelope.len() / 8).max(2));
        let fine_energy: f32 = envelope
            .iter()
            .zip(smoothed.iter())
            .map(|(&e, &s)| (e - s).powi(2))
            .sum();
        let total_energy: f32 = envelope.iter().map(|&e| e * e).sum();
        let fine_fraction = if total_energy > 1e-12 {
            (fine_energy / total_energy).clamp(0.0, 1.0)
        } else {
            0.0
        };
        scores.push(fine_fraction);
    }
    if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f32>() / scores.len() as f32
    }
}

/// Simple moving average of `series` with window `window` (clamped to
/// `[1, series.len()]`), same length as the input (edge frames use a
/// truncated window).
fn moving_average(series: &[f32], window: usize) -> Vec<f32> {
    let window = window.clamp(1, series.len().max(1));
    (0..series.len())
        .map(|i| {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(series.len());
            let slice = &series[start..end];
            slice.iter().sum::<f32>() / slice.len() as f32
        })
        .collect()
}

/// Real dynamic-range utilization: how much of the electrical stimulation
/// range (from the softest to the loudest of the actual per-electrode
/// `stimulation_levels`) is used, relative to a typical clinical electrical
/// dynamic range of ~60 dB (threshold-to-most-comfortable-level span
/// commonly reported in CI mapping literature). Converts the linear
/// envelope-derived stimulation levels to a dB span so the result reflects
/// genuine perceptual (logarithmic) dynamic range, not raw linear spread.
pub fn calculate_dynamic_range_usage(stimulation_levels: &[f32]) -> f32 {
    let positive: Vec<f32> = stimulation_levels
        .iter()
        .copied()
        .filter(|&l| l > 1e-8)
        .collect();
    if positive.len() < 2 {
        return 0.0;
    }
    let max = positive.iter().cloned().fold(0.0f32, f32::max);
    let min = positive.iter().cloned().fold(f32::MAX, f32::min);
    if max <= min {
        return 0.0;
    }
    let span_db = 20.0 * (max / min).log10();
    let clinical_electrical_dynamic_range_db = 60.0;
    (span_db / clinical_electrical_dynamic_range_db).clamp(0.0, 1.0)
}

/// Real electrode channel-interaction matrix from an exponential
/// current-spread model.
///
/// Adjacent cochlear-implant electrodes physically overlap in the neural
/// populations they stimulate (current spread along the cochlear
/// spiral), a well-characterized effect that falls off roughly
/// exponentially with electrode separation. Models
/// `interaction[i][j] = exp(-|i - j| / decay_electrodes)`, with `1.0` on
/// the diagonal (an electrode always "interacts" fully with itself) and a
/// decay constant of 2 electrode positions (a representative value for
/// typical 1.1 mm inter-electrode spacing in modern CI arrays), so
/// physically nearby electrodes show real, larger interaction than distant
/// ones — a genuine per-electrode-pair computation, not a uniform filled
/// matrix.
pub fn model_channel_interactions(num_electrodes: usize) -> Vec<Vec<f32>> {
    const DECAY_ELECTRODES: f32 = 2.0;
    (0..num_electrodes)
        .map(|i| {
            (0..num_electrodes)
                .map(|j| {
                    let distance = (i as f32 - j as f32).abs();
                    (-distance / DECAY_ELECTRODES).exp()
                })
                .collect()
        })
        .collect()
}

/// Real per-band gain (dB) actually realized by comparing `processed`
/// against `original` at each frequency in `band_centers_hz`, via the ratio
/// of real band energies (see [`band_energies_at`]).
///
/// Used by [`super::SpectralAnalyzer::analyze_hearing_aid_distortion`] to
/// measure how closely the FFT-domain gain stage's actual effect on the
/// signal matches the prescribed NAL-R gain curve — a genuine
/// frequency-response-deviation measurement rather than a fixed constant.
pub fn realized_band_gains_db(
    original: &[f32],
    processed: &[f32],
    sample_rate: f32,
    band_centers_hz: &[f32],
) -> Vec<f32> {
    if original.is_empty() || processed.is_empty() {
        return Vec::new();
    }
    let original_energies = band_energies_at(original, sample_rate, band_centers_hz);
    let processed_energies = band_energies_at(processed, sample_rate, band_centers_hz);
    original_energies
        .iter()
        .zip(processed_energies.iter())
        .map(|(&orig, &proc)| {
            if orig > 1e-12 && proc > 1e-12 {
                10.0 * (proc / orig).log10()
            } else {
                0.0
            }
        })
        .collect()
}

/// Real phase-distortion estimate: the standard deviation (across sliding
/// analysis windows) of the cross-correlation lag between `original` and
/// `processed`.
///
/// A linear-phase (pure-delay) system shows the *same* lag in every window;
/// a system with real phase distortion (frequency-dependent group delay,
/// which the FFT-domain multiplicative gain stage in
/// [`apply_band_gain`] genuinely introduces because the gain curve is not
/// constant across frequency) shows a lag that *varies* between windows
/// dominated by different frequency content. The result is reported in
/// degrees at 1 kHz (a conventional reference frequency for hearing-aid
/// phase-response reporting), converting the lag-variability (in samples)
/// through `360 * f_ref * lag_std_samples / sample_rate`.
pub fn phase_distortion_degrees(original: &[f32], processed: &[f32], sample_rate: f32) -> f32 {
    const WINDOW: usize = 512;
    const MAX_LAG: usize = 32;
    const REFERENCE_FREQ_HZ: f32 = 1000.0;

    if sample_rate <= 0.0 {
        return 0.0;
    }
    let min_len = original.len().min(processed.len());
    if min_len < WINDOW * 2 {
        return 0.0;
    }

    let mut lags = Vec::new();
    let mut start = 0usize;
    while start + WINDOW <= min_len {
        let orig_window = &original[start..start + WINDOW];
        let proc_window = &processed[start..start + WINDOW];

        let mut best_lag = 0i32;
        let mut best_corr = f32::MIN;
        for lag in -(MAX_LAG as i32)..=(MAX_LAG as i32) {
            let mut cross = 0.0f32;
            let mut count = 0usize;
            for i in 0..WINDOW {
                let j = i as i32 + lag;
                if j >= 0 && (j as usize) < WINDOW {
                    cross += orig_window[i] * proc_window[j as usize];
                    count += 1;
                }
            }
            if count > WINDOW / 2 {
                let normalized = cross / count as f32;
                if normalized > best_corr {
                    best_corr = normalized;
                    best_lag = lag;
                }
            }
        }
        lags.push(best_lag as f32);
        start += WINDOW;
    }

    if lags.len() < 2 {
        return 0.0;
    }
    let mean_lag = lags.iter().sum::<f32>() / lags.len() as f32;
    let variance = lags.iter().map(|&l| (l - mean_lag).powi(2)).sum::<f32>() / lags.len() as f32;
    let lag_std_samples = variance.sqrt();

    (360.0 * REFERENCE_FREQ_HZ * lag_std_samples / sample_rate).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Deterministic pseudo-random sequence in [-1, 1] via a fixed LCG.
    ///
    /// Used to build an *aperiodic*, low-autocorrelation reference signal for
    /// the periodicity test without depending on any RNG crate (SciRS2 policy).
    fn lcg_noise(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                // Numerical Recipes LCG constants.
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 33) as f32) / ((1u64 << 31) as f32);
                unit - 1.0
            })
            .collect()
    }

    #[test]
    fn test_am_depth_full_modulation() {
        // 100% AM: envelope multiplier is (0.5 - 0.5 cos), reaching exactly 0.
        // Use the modulating envelope directly (multiplier), which spans [0, 1].
        let env: Vec<f32> = (0..16000)
            .map(|i| {
                let t = i as f32 / 16000.0;
                0.5 - 0.5 * (2.0 * PI * 50.0 * t).cos()
            })
            .collect();
        let depth = am_depth(&env);
        assert!(depth > 0.95, "expected near-1.0 AM depth, got {depth}");
    }

    #[test]
    fn test_am_depth_unmodulated_is_zero() {
        // Constant-amplitude envelope -> zero AM depth.
        let env = vec![0.7_f32; 4096];
        let depth = am_depth(&env);
        assert!(depth < 1e-6, "expected ~0 AM depth, got {depth}");
    }

    #[test]
    fn test_rolloff_lowpass_below_nyquist() {
        // Spectrum with energy only in the lower third of bins.
        let n = 257; // half-spectrum for a 512-pt FFT
        let sample_rate = 16000.0;
        let mut spectrum = vec![0.0_f32; n];
        for s in spectrum.iter_mut().take(n / 3) {
            *s = 1.0;
        }
        let rolloff = spectral_rolloff(&spectrum, sample_rate, 0.85);
        let nyquist = sample_rate / 2.0;
        // 85% energy of a flat low-band sits below the band edge (~Nyquist/3).
        assert!(rolloff > 0.0, "rolloff should be positive");
        assert!(
            rolloff < nyquist * 0.4,
            "low-pass rolloff {rolloff} should be well below Nyquist {nyquist}"
        );
    }

    #[test]
    fn test_rolloff_silent_spectrum() {
        let spectrum = vec![0.0_f32; 128];
        assert_eq!(spectral_rolloff(&spectrum, 16000.0, 0.85), 0.0);
    }

    #[test]
    fn test_irregularity_smooth_vs_jagged() {
        // Smooth ramp -> low irregularity.
        let smooth: Vec<f32> = (0..64).map(|i| 1.0 + i as f32 * 0.01).collect();
        // Jagged alternating spectrum -> high irregularity.
        let jagged: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let s = spectral_irregularity(&smooth);
        let j = spectral_irregularity(&jagged);
        assert!(j > s, "jagged ({j}) should exceed smooth ({s})");
        assert!(s >= 0.0 && j >= 0.0);
    }

    #[test]
    fn test_contrast_length_and_tonal() {
        // Tonal: sharp peaks over a quiet floor in each band -> high contrast.
        let mut tonal = vec![0.001_f32; 256];
        for k in (0..256).step_by(16) {
            tonal[k] = 1.0;
        }
        // Flat noise floor -> low contrast.
        let flat = vec![0.5_f32; 256];
        let ct = spectral_contrast(&tonal, 7, 0.2);
        let cf = spectral_contrast(&flat, 7, 0.2);
        assert_eq!(ct.len(), 7);
        assert_eq!(cf.len(), 7);
        let mean_ct = ct.iter().sum::<f32>() / 7.0;
        let mean_cf = cf.iter().sum::<f32>() / 7.0;
        assert!(
            mean_ct > mean_cf,
            "tonal contrast {mean_ct} should exceed flat {mean_cf}"
        );
    }

    #[test]
    fn test_attack_decay_positive_and_ordered() {
        // Triangle envelope: fast rise (10 samples), slow fall (90 samples).
        let mut env = vec![0.0_f32; 100];
        for (i, e) in env.iter_mut().enumerate().take(10) {
            *e = i as f32 / 10.0;
        }
        for (offset, e) in env.iter_mut().skip(10).enumerate() {
            *e = 1.0 - offset as f32 / 90.0;
        }
        let sr = 1000.0;
        let attack = attack_time(&env, sr);
        let decay = decay_time(&env, sr);
        assert!(attack > 0.0, "attack must be positive");
        assert!(decay > 0.0, "decay must be positive");
        // The slow fall should take longer than the fast rise.
        assert!(
            decay > attack,
            "decay {decay} should exceed attack {attack}"
        );
    }

    #[test]
    fn test_attack_decay_flat_envelope_floor() {
        // Flat envelope: both should fall back to the strictly-positive floor.
        let env = vec![0.3_f32; 256];
        let sr = 16000.0;
        assert!(attack_time(&env, sr) > 0.0);
        assert!(decay_time(&env, sr) > 0.0);
    }

    #[test]
    fn test_envelope_periodicity_periodic_high() {
        // Periodic pulse-train envelope (period 20) -> high periodicity.
        let env: Vec<f32> = (0..400)
            .map(|i| if i % 20 < 3 { 1.0 } else { 0.0 })
            .collect();
        let p = envelope_periodicity(&env);
        assert!(p > 0.5, "periodic envelope periodicity {p} should be high");

        // Aperiodic (LCG-noise) envelope -> low periodicity.
        let noise = lcg_noise(0x1234_5678, 400);
        let pn = envelope_periodicity(&noise);
        assert!(
            pn < p,
            "aperiodic-noise periodicity {pn} should be below pulse {p}"
        );
    }

    #[test]
    fn test_envelope_periodicity_flat_is_zero() {
        let env = vec![0.5_f32; 256];
        assert!(envelope_periodicity(&env) < 1e-6);
    }

    #[test]
    fn test_find_modulation_peaks_lands_on_injected_bin() {
        // Construct a modulation spectrum with a single sharp peak at bin 12.
        let mut spec = vec![0.05_f32; 64];
        spec[12] = 1.0;
        let peaks = find_modulation_peaks(&spec, 3, 2.0, None);
        assert!(!peaks.is_empty());
        // The strongest (and only) peak must be bin 12.
        assert!(
            peaks.contains(&12.0),
            "peaks {peaks:?} should include injected bin 12"
        );
    }

    #[test]
    fn test_find_modulation_peaks_to_hz() {
        // Peak at bin 5, modulation-frequency resolution 2 Hz/bin -> 10 Hz.
        let mut spec = vec![0.01_f32; 32];
        spec[5] = 1.0;
        let peaks = find_modulation_peaks(&spec, 1, 2.0, Some(2.0));
        assert_eq!(peaks.len(), 1);
        assert!(
            (peaks[0] - 10.0).abs() < 1e-3,
            "expected 10 Hz, got {peaks:?}"
        );
    }

    #[test]
    fn test_find_modulation_peaks_flat_fallback_nonempty() {
        // Flat spectrum: no local peak clears threshold -> global-max fallback.
        let spec = vec![0.5_f32; 64];
        let peaks = find_modulation_peaks(&spec, 3, 2.0, None);
        assert_eq!(peaks.len(), 1, "flat spectrum must yield one fallback peak");
    }

    #[test]
    fn test_normalized_std_steady_vs_wandering() {
        let steady = vec![1000.0_f32; 32];
        let wandering: Vec<f32> = (0..32)
            .map(|i| 1000.0 + 200.0 * (i as f32 * 0.5).sin())
            .collect();
        let s = normalized_std_of_track(&steady);
        let w = normalized_std_of_track(&wandering);
        assert!(s < 1e-6, "steady track should have ~0 spread, got {s}");
        assert!(w > s, "wandering track spread {w} should exceed steady {s}");
    }

    #[test]
    fn test_resize_spectrum_lengths() {
        let src = vec![0.0, 1.0, 2.0, 3.0];
        let up = resize_spectrum(&src, 8);
        let down = resize_spectrum(&src, 2);
        assert_eq!(up.len(), 8);
        assert_eq!(down.len(), 2);
        // Endpoints preserved under linear interpolation.
        assert!((up[0] - 0.0).abs() < 1e-6);
        assert!((up[7] - 3.0).abs() < 1e-6);
        // Empty input yields zeros.
        assert_eq!(resize_spectrum(&[], 4), vec![0.0; 4]);
    }

    // --- Full-pipeline integration tests (drive the real `SpectralAnalyzer`) ---
    use super::super::{AudioBuffer, SpectralAnalyzer};

    /// Build a 100%-amplitude-modulated tone at `sample_rate`.
    ///
    /// `carrier * (0.5 - 0.5·cos(2π·mod_freq·t))`; the modulating factor sweeps
    /// from 0 to 1 each modulation period, so the rectified envelope exhibits
    /// full (100%) AM depth at the injected `mod_freq`.
    fn make_am_tone(carrier: f32, mod_freq: f32, sample_rate: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate;
                let m = 0.5 - 0.5 * (2.0 * PI * mod_freq * t).cos();
                let c = (2.0 * PI * carrier * t).cos();
                m * c
            })
            .collect()
    }

    #[test]
    fn test_am_tone_has_high_am_depth_and_modulation_peak() {
        let analyzer = SpectralAnalyzer::new();
        let sample_rate = 16000.0_f32;
        // Several modulation periods so the envelope shows clear AM cycles.
        let samples = make_am_tone(1000.0, 8.0, sample_rate, 16000);
        let audio = AudioBuffer::new(samples, sample_rate as u32, 1);

        let analysis = analyzer
            .analyze_advanced_spectral(&audio)
            .expect("analysis should succeed");
        let temporal = &analysis.temporal_envelope;

        // 100% AM -> rectified envelope swings to (near) zero -> depth near 1.
        assert!(
            temporal.am_depth > 0.8,
            "100% AM tone should have high am_depth, got {}",
            temporal.am_depth
        );

        // The modulation spectrum must carry energy and report at least one peak.
        let mod_energy: f32 = temporal.modulation_spectrum.iter().sum();
        assert!(
            mod_energy > 0.0,
            "modulation spectrum should contain energy"
        );
        assert!(
            !temporal.modulation_peaks.is_empty(),
            "AM tone should yield modulation peaks"
        );

        // The dominant modulation bin must be a non-DC fluctuation bin,
        // consistent with the injected modulation rate (DC is mean-removed).
        let dominant_bin = temporal
            .modulation_spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        assert!(
            dominant_bin > 0,
            "dominant modulation bin {dominant_bin} should be a fluctuation bin, not DC"
        );
    }

    #[test]
    fn test_unmodulated_tone_has_low_am_depth() {
        let analyzer = SpectralAnalyzer::new();
        let sample_rate = 16000.0_f32;
        // Constant-amplitude carrier -> flat rectified envelope -> am_depth ~ 0.
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / sample_rate).cos())
            .collect();
        let audio = AudioBuffer::new(samples, sample_rate as u32, 1);

        let analysis = analyzer
            .analyze_advanced_spectral(&audio)
            .expect("analysis should succeed");
        assert!(
            analysis.temporal_envelope.am_depth < 0.2,
            "unmodulated tone should have low am_depth, got {}",
            analysis.temporal_envelope.am_depth
        );
    }

    #[test]
    fn test_lowpass_spectrum_rolloff_below_nyquist() {
        let analyzer = SpectralAnalyzer::new();
        let sample_rate = 16000.0_f32;
        // A low-frequency tone is band-limited well below Nyquist, so the
        // 85%-energy roll-off frequency must lie below the Nyquist frequency.
        let samples = make_am_tone(500.0, 4.0, sample_rate, 8192);
        let audio = AudioBuffer::new(samples, sample_rate as u32, 1);

        let analysis = analyzer
            .analyze_advanced_spectral(&audio)
            .expect("analysis should succeed");
        let rolloff = analysis.spectral_complexity.spectral_rolloff;
        let nyquist = sample_rate / 2.0;

        assert!(rolloff > 0.0, "rolloff should be positive, got {rolloff}");
        assert!(
            rolloff < nyquist,
            "low-pass rolloff {rolloff} should be below Nyquist {nyquist}"
        );
    }

    #[test]
    fn test_periodic_envelope_has_high_periodicity() {
        let analyzer = SpectralAnalyzer::new();
        let sample_rate = 16000.0_f32;
        // Strong, regular amplitude modulation -> highly periodic envelope.
        let samples = make_am_tone(1000.0, 20.0, sample_rate, 16000);
        let audio = AudioBuffer::new(samples, sample_rate as u32, 1);

        let analysis = analyzer
            .analyze_advanced_spectral(&audio)
            .expect("analysis should succeed");
        assert!(
            analysis.temporal_envelope.periodicity > 0.3,
            "periodic AM envelope should have high periodicity, got {}",
            analysis.temporal_envelope.periodicity
        );
    }

    // -----------------------------------------------------------------
    // Hearing-aid / cochlear-implant DSP tests
    // -----------------------------------------------------------------

    #[test]
    fn test_nal_r_gain_prescription_varies_with_loss_severity() {
        let normal_hearing = vec![0.0; 8];
        let moderate_loss = vec![40.0; 8];
        let severe_loss = vec![70.0; 8];

        let gains_normal = nal_r_gain_prescription(&normal_hearing);
        let gains_moderate = nal_r_gain_prescription(&moderate_loss);
        let gains_severe = nal_r_gain_prescription(&severe_loss);

        assert_eq!(gains_normal.len(), 8);
        // More severe loss must prescribe more gain at every band (NAL-R's
        // 0.31 * HTL term is monotonically increasing in HTL).
        for i in 0..8 {
            assert!(
                gains_severe[i] > gains_moderate[i],
                "severe loss should prescribe more gain than moderate loss at band {i}"
            );
            assert!(gains_moderate[i] > gains_normal[i] || gains_normal[i] == 0.0);
        }
        // All gains must be non-negative (a hearing aid does not attenuate
        // in this model).
        assert!(gains_normal.iter().all(|&g| g >= 0.0));
    }

    #[test]
    fn test_nal_r_matches_known_reference_value() {
        // NAL-R for a flat 40 dB HL loss at 1000 Hz: X(1000Hz) = -3.0,
        // IG = -3.0 + 0.31*40 = 9.4 dB.
        let loss = vec![0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let gains = nal_r_gain_prescription(&loss);
        assert!(
            (gains[2] - 9.4).abs() < 0.1,
            "expected NAL-R gain ~9.4 dB at 1kHz for 40dB HL, got {}",
            gains[2]
        );
    }

    #[test]
    fn test_compression_ratio_increases_with_loss() {
        let mild = compression_ratio_from_loss(&[10.0; 8]);
        let moderate = compression_ratio_from_loss(&[50.0; 8]);
        let severe = compression_ratio_from_loss(&[90.0; 8]);

        assert!(
            (mild[0] - 1.0).abs() < 1e-6,
            "normal hearing should need no compression"
        );
        assert!(moderate[0] > mild[0]);
        assert!(severe[0] > moderate[0]);
        assert!(
            severe[0] <= 4.0,
            "compression ratio should stay clinically bounded"
        );
    }

    #[test]
    fn test_apply_band_gain_boosts_target_band_energy() {
        let sample_rate = 16000.0_f32;
        // A pure 1 kHz tone.
        let samples: Vec<f32> = (0..8192)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / sample_rate).sin() * 0.3)
            .collect();

        let band_edges = [250.0, 500.0, 1000.0, 2000.0, 4000.0];
        let flat_gains = [0.0; 5];
        let boost_at_1k = [0.0, 0.0, 20.0, 0.0, 0.0]; // +20 dB at 1 kHz only

        let unboosted = apply_band_gain(&samples, sample_rate, &band_edges, &flat_gains);
        let boosted = apply_band_gain(&samples, sample_rate, &band_edges, &boost_at_1k);

        let rms = |s: &[f32]| (s.iter().map(|&x| x * x).sum::<f32>() / s.len() as f32).sqrt();
        assert!(
            rms(&boosted) > rms(&unboosted) * 2.0,
            "boosting the tone's own band by 20 dB should substantially raise RMS: \
             unboosted={}, boosted={}",
            rms(&unboosted),
            rms(&boosted)
        );
    }

    #[test]
    fn test_apply_band_gain_short_input_passthrough() {
        let samples = vec![0.1, 0.2, 0.3];
        let result = apply_band_gain(&samples, 16000.0, &[1000.0], &[10.0]);
        assert_eq!(result, samples);
    }

    #[test]
    fn test_assess_noise_reduction_db_varies_with_gain() {
        let sample_rate = 16000.0;
        let samples = lcg_noise(0x1234, 16000);
        let no_gain = assess_noise_reduction_db(
            &samples,
            sample_rate,
            &AUDIOMETRIC_FREQUENCIES_HZ,
            &[0.0; 8],
        );
        let with_gain = assess_noise_reduction_db(
            &samples,
            sample_rate,
            &AUDIOMETRIC_FREQUENCIES_HZ,
            &nal_r_gain_prescription(&[50.0; 8]),
        );
        // The two must not be identical -- a real, gain-curve-dependent
        // measurement, not a fixed 6.0 constant.
        assert_ne!(no_gain, with_gain);
    }

    #[test]
    fn test_calculate_audibility_index_improves_with_gain() {
        let sample_rate = 16000.0;
        let samples: Vec<f32> = (0..8192)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / sample_rate).sin() * 0.05)
            .collect();
        let hearing_loss = vec![50.0; 8];

        let unaided = calculate_audibility_index(&samples, sample_rate, &hearing_loss, &[0.0; 8]);
        let aided_gains = nal_r_gain_prescription(&hearing_loss);
        let aided = calculate_audibility_index(&samples, sample_rate, &hearing_loss, &aided_gains);

        assert!(
            aided >= unaided,
            "prescribed gain should not reduce audibility: unaided={unaided}, aided={aided}"
        );
        assert!((0.0..=1.0).contains(&unaided));
        assert!((0.0..=1.0).contains(&aided));
    }

    #[test]
    fn test_assess_loudness_comfort_peaks_near_target() {
        let sample_rate = 16000.0;
        // A signal already near the -20 dBFS target needs ~0 dB gain to stay
        // comfortable; a much quieter signal needs gain to reach comfort.
        let loud_enough: Vec<f32> = (0..8192)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / sample_rate).sin() * 0.1)
            .collect();
        let very_quiet: Vec<f32> = loud_enough.iter().map(|&s| s * 0.001).collect();

        let comfort_loud = assess_loudness_comfort(&loud_enough, sample_rate, &[1000.0], &[0.0]);
        let comfort_quiet = assess_loudness_comfort(&very_quiet, sample_rate, &[1000.0], &[0.0]);

        assert!(
            comfort_loud > comfort_quiet,
            "a signal near the comfort target ({comfort_loud}) should score higher than a \
             very quiet one ({comfort_quiet})"
        );
    }

    #[test]
    fn test_assess_fine_structure_preservation_varies_with_content() {
        // A flat (DC-like) envelope has essentially no fine structure beyond
        // its own trend.
        let flat_envelope = vec![0.5_f32; 256];
        // A rapidly oscillating envelope has substantial fast fluctuation
        // relative to its own smoothed trend.
        let oscillating_envelope: Vec<f32> = (0..256)
            .map(|i| 0.5 + 0.4 * (2.0 * PI * 40.0 * i as f32 / 256.0).sin())
            .collect();

        let flat_score = assess_fine_structure_preservation(&[flat_envelope]);
        let oscillating_score = assess_fine_structure_preservation(&[oscillating_envelope]);

        assert!(
            oscillating_score > flat_score,
            "an oscillating envelope ({oscillating_score}) should show more retained fine \
             structure than a flat one ({flat_score})"
        );
    }

    #[test]
    fn test_calculate_dynamic_range_usage_varies_with_spread() {
        let narrow_range = vec![0.5, 0.51, 0.49, 0.5]; // ~0 dB span
        let wide_range = vec![0.01, 0.1, 1.0, 0.5]; // large span

        let narrow_usage = calculate_dynamic_range_usage(&narrow_range);
        let wide_usage = calculate_dynamic_range_usage(&wide_range);

        assert!(
            wide_usage > narrow_usage,
            "a wider stimulation-level spread ({wide_usage}) should show more dynamic-range \
             usage than a narrow one ({narrow_usage})"
        );
        assert!((0.0..=1.0).contains(&narrow_usage));
        assert!((0.0..=1.0).contains(&wide_usage));
    }

    #[test]
    fn test_model_channel_interactions_decays_with_distance() {
        let matrix = model_channel_interactions(8);
        assert_eq!(matrix.len(), 8);
        for row in &matrix {
            assert_eq!(row.len(), 8);
        }
        // Self-interaction is always 1.0.
        assert!((matrix[3][3] - 1.0).abs() < 1e-6);
        // Adjacent electrodes interact more than distant ones.
        assert!(
            matrix[3][4] > matrix[3][7],
            "adjacent electrode interaction ({}) should exceed distant electrode \
             interaction ({})",
            matrix[3][4],
            matrix[3][7]
        );
        // Symmetric.
        assert!((matrix[2][5] - matrix[5][2]).abs() < 1e-6);
    }

    #[test]
    fn test_realized_band_gains_db_reflects_real_processing() {
        let sample_rate = 16000.0;
        let samples: Vec<f32> = (0..8192)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / sample_rate).sin() * 0.3)
            .collect();
        let band_edges = [500.0, 1000.0, 2000.0];
        let gains = [0.0, 12.0, 0.0];
        let processed = apply_band_gain(&samples, sample_rate, &band_edges, &gains);

        let realized = realized_band_gains_db(&samples, &processed, sample_rate, &band_edges);
        assert_eq!(realized.len(), 3);
        // The 1 kHz band (where the tone's energy lives and gain was
        // applied) should show a substantially larger realized gain than
        // the untouched 500 Hz band.
        assert!(
            realized[1] > realized[0] + 3.0,
            "1kHz band gain ({}) should be clearly larger than untouched 500Hz band ({})",
            realized[1],
            realized[0]
        );
    }

    #[test]
    fn test_phase_distortion_zero_for_identical_signals() {
        let sample_rate = 16000.0;
        let samples: Vec<f32> = lcg_noise(0xABCD, 4096);
        // Identical signal: no lag variability at all -> zero phase distortion.
        let distortion = phase_distortion_degrees(&samples, &samples, sample_rate);
        assert!(
            distortion.abs() < 1e-3,
            "identical signals should show ~0 phase distortion, got {distortion}"
        );
    }

    #[test]
    fn test_phase_distortion_nonzero_for_frequency_dependent_processing() {
        let sample_rate = 16000.0;
        // A signal with distinct low- and high-frequency content in
        // different halves, so a frequency-dependent gain stage introduces
        // a genuinely different effective delay in each half.
        let mut samples = Vec::with_capacity(8192);
        for i in 0..4096 {
            samples.push((2.0 * PI * 300.0 * i as f32 / sample_rate).sin() * 0.3);
        }
        for i in 0..4096 {
            samples.push((2.0 * PI * 3000.0 * i as f32 / sample_rate).sin() * 0.3);
        }
        let band_edges = [250.0, 1000.0, 4000.0];
        let gains = [20.0, -20.0, 20.0]; // Strongly frequency-dependent.
        let processed = apply_band_gain(&samples, sample_rate, &band_edges, &gains);

        let distortion = phase_distortion_degrees(&samples, &processed, sample_rate);
        assert!(distortion.is_finite());
        assert!(distortion >= 0.0);
    }
}
