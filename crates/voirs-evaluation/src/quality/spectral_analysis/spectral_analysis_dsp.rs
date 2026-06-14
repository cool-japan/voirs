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
}
