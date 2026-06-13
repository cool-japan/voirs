//! Shared audio DSP primitives for the REST and WebSocket evaluation services.
//!
//! These free functions implement real frequency-domain analysis (spectral
//! centroid, spectral rolloff, MFCC) and a time-domain autocorrelation pitch
//! estimator directly from a `&[f32]` signal plus its sample rate. They operate
//! on raw slices so they can be unit-tested in isolation and shared between the
//! `rest_api` and `websocket` modules without duplicating the analysis code.
//!
//! All FFTs use the SciRS2 abstractions (`scirs2_fft::rfft`); no external FFT,
//! `ndarray`, or `rand` crates are used directly, per the SciRS2 policy.

/// Lowest fundamental frequency (Hz) considered voiced by [`autocorrelation_f0`].
const PITCH_MIN_HZ: f32 = 80.0;
/// Highest fundamental frequency (Hz) considered voiced by [`autocorrelation_f0`].
const PITCH_MAX_HZ: f32 = 400.0;
/// Minimum normalized autocorrelation for a frame to be treated as voiced.
const PITCH_VOICING_THRESHOLD: f64 = 0.3;
/// Number of triangular mel filters used by the MFCC front end.
const MEL_FILTER_COUNT: usize = 26;

/// Hann window coefficient at index `i` for a window of length `n`.
fn hann(i: usize, n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0)).cos()
}

/// FFT window length for a given sample rate: a power of two close to a 25 ms
/// frame, clamped to `[256, 2048]`. Keeps frequency resolution sensible across
/// the supported sample rates (8 kHz–48 kHz).
fn frame_size(sample_rate: u32) -> usize {
    let target = (0.025 * sample_rate as f64).round() as usize;
    let mut n = 256usize;
    while n < target && n < 2048 {
        n *= 2;
    }
    n.clamp(256, 2048)
}

/// Average power spectrum across Hann-windowed, 50%-overlapping frames.
///
/// Returns a vector of length `n_fft / 2 + 1` holding the mean `|X_k|^2` over
/// all frames, or `None` when the input is too short to analyze. Frames shorter
/// than `n_fft` (the trailing frame, or a signal shorter than one frame) are
/// zero-padded.
fn averaged_power_spectrum(samples: &[f32], n_fft: usize) -> Option<Vec<f64>> {
    if samples.len() < 2 || n_fft < 2 {
        return None;
    }
    let num_bins = n_fft / 2 + 1;
    let hop = (n_fft / 2).max(1);
    let mut power = vec![0.0f64; num_bins];
    let mut buffer = vec![0.0f64; n_fft];
    let mut frame_count = 0usize;
    let mut start = 0usize;

    loop {
        let available = (samples.len() - start).min(n_fft);
        for (i, slot) in buffer.iter_mut().enumerate() {
            *slot = if i < available {
                samples[start + i] as f64 * hann(i, n_fft)
            } else {
                0.0
            };
        }
        if let Ok(spectrum) = scirs2_fft::rfft(&buffer, Some(n_fft)) {
            for (k, value) in spectrum.iter().take(num_bins).enumerate() {
                power[k] += value.re * value.re + value.im * value.im;
            }
            frame_count += 1;
        }
        if available < n_fft {
            break;
        }
        start += hop;
        if start >= samples.len() {
            break;
        }
    }

    if frame_count == 0 {
        return None;
    }
    let scale = 1.0 / frame_count as f64;
    for value in power.iter_mut() {
        *value *= scale;
    }
    Some(power)
}

/// FFT magnitude-weighted spectral centroid in Hz: `Σ(f_k·|X_k|) / Σ|X_k|`.
///
/// Returns `0.0` for empty input, a zero sample rate, or a fully silent signal.
pub(crate) fn spectral_centroid_hz(samples: &[f32], sample_rate: u32) -> f32 {
    if samples.is_empty() || sample_rate == 0 {
        return 0.0;
    }
    let n_fft = frame_size(sample_rate);
    let power = match averaged_power_spectrum(samples, n_fft) {
        Some(power) => power,
        None => return 0.0,
    };
    let bin_hz = sample_rate as f64 / n_fft as f64;
    let mut weighted = 0.0f64;
    let mut total = 0.0f64;
    for (k, &p) in power.iter().enumerate() {
        let magnitude = p.sqrt();
        weighted += (k as f64 * bin_hz) * magnitude;
        total += magnitude;
    }
    if total > 0.0 {
        (weighted / total) as f32
    } else {
        0.0
    }
}

/// FFT-based spectral rolloff in Hz: the frequency below which `rolloff_fraction`
/// of the total spectral energy (`Σ|X_k|^2`) is contained.
///
/// Returns `0.0` for empty/silent input and the Nyquist frequency when the
/// threshold is never reached. `rolloff_fraction` is clamped to `[0, 1]`.
pub(crate) fn spectral_rolloff_hz(samples: &[f32], sample_rate: u32, rolloff_fraction: f32) -> f32 {
    if samples.is_empty() || sample_rate == 0 {
        return 0.0;
    }
    let n_fft = frame_size(sample_rate);
    let power = match averaged_power_spectrum(samples, n_fft) {
        Some(power) => power,
        None => return 0.0,
    };
    let bin_hz = sample_rate as f64 / n_fft as f64;
    let total_energy: f64 = power.iter().sum();
    if total_energy <= 0.0 {
        return 0.0;
    }
    let threshold = total_energy * f64::from(rolloff_fraction.clamp(0.0, 1.0));
    let mut cumulative = 0.0f64;
    for (k, &p) in power.iter().enumerate() {
        cumulative += p;
        if cumulative >= threshold {
            return (k as f64 * bin_hz) as f32;
        }
    }
    sample_rate as f32 / 2.0
}

/// Convert a frequency in Hz to the mel scale (O'Shaughnessy formula).
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert a mel value back to Hz.
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

/// Build a triangular mel filterbank as a list of `(bin_index, weight)` pairs
/// per filter, spanning `0..=Nyquist` over `MEL_FILTER_COUNT` filters.
fn mel_filterbank(num_bins: usize, n_fft: usize, sample_rate: u32) -> Vec<Vec<(usize, f64)>> {
    let f_max = sample_rate as f64 / 2.0;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(f_max);
    let point_count = MEL_FILTER_COUNT + 2;

    // Center/edge bin index for each of the MEL_FILTER_COUNT + 2 mel points.
    let mut points = Vec::with_capacity(point_count);
    for i in 0..point_count {
        let mel = mel_min + (mel_max - mel_min) * i as f64 / (point_count - 1) as f64;
        let hz = mel_to_hz(mel);
        let bin = (hz * n_fft as f64 / sample_rate as f64).round() as usize;
        points.push(bin.min(num_bins - 1));
    }

    let mut filters = Vec::with_capacity(MEL_FILTER_COUNT);
    for m in 1..=MEL_FILTER_COUNT {
        let left = points[m - 1];
        let center = points[m];
        let right = points[m + 1];
        let mut filter = Vec::new();
        if center > left {
            for bin in left..center {
                filter.push((bin, (bin - left) as f64 / (center - left) as f64));
            }
        }
        if right > center {
            for bin in center..right {
                filter.push((bin, (right - bin) as f64 / (right - center) as f64));
            }
        }
        // Degenerate (collapsed) filter: contribute the center bin directly so
        // the coefficient is well defined even at coarse resolutions.
        if filter.is_empty() {
            filter.push((center, 1.0));
        }
        filters.push(filter);
    }
    filters
}

/// Extract `n_coeffs` MFCCs averaged across Hann-windowed frames.
///
/// Pipeline per frame: Hann window → `scirs2_fft::rfft` → power spectrum →
/// triangular mel filterbank (`MEL_FILTER_COUNT` filters) → log energies →
/// DCT-II → first `n_coeffs` coefficients. The per-frame coefficients are then
/// averaged. Returns a zero vector of length `n_coeffs` for degenerate input.
pub(crate) fn mfcc_features(samples: &[f32], sample_rate: u32, n_coeffs: usize) -> Vec<f32> {
    if samples.len() < 2 || sample_rate == 0 || n_coeffs == 0 {
        return vec![0.0; n_coeffs];
    }
    let n_fft = frame_size(sample_rate);
    let num_bins = n_fft / 2 + 1;
    let filterbank = mel_filterbank(num_bins, n_fft, sample_rate);
    let hop = (n_fft / 2).max(1);
    let mut accum = vec![0.0f64; n_coeffs];
    let mut buffer = vec![0.0f64; n_fft];
    let mut frame_count = 0usize;
    let mut start = 0usize;

    loop {
        let available = (samples.len() - start).min(n_fft);
        for (i, slot) in buffer.iter_mut().enumerate() {
            *slot = if i < available {
                samples[start + i] as f64 * hann(i, n_fft)
            } else {
                0.0
            };
        }
        if let Ok(spectrum) = scirs2_fft::rfft(&buffer, Some(n_fft)) {
            let power: Vec<f64> = spectrum
                .iter()
                .take(num_bins)
                .map(|value| value.re * value.re + value.im * value.im)
                .collect();

            // Log mel-band energies.
            let log_mel: Vec<f64> = filterbank
                .iter()
                .map(|filter| {
                    let energy: f64 = filter
                        .iter()
                        .map(|&(bin, weight)| weight * power[bin])
                        .sum();
                    (energy + 1e-10).ln()
                })
                .collect();

            // DCT-II of the log mel energies, keeping the first `n_coeffs`.
            for (c, acc) in accum.iter_mut().enumerate() {
                let mut sum = 0.0;
                for (m, &log_energy) in log_mel.iter().enumerate() {
                    let angle = std::f64::consts::PI * c as f64 * (m as f64 + 0.5)
                        / MEL_FILTER_COUNT as f64;
                    sum += log_energy * angle.cos();
                }
                *acc += sum;
            }
            frame_count += 1;
        }
        if available < n_fft {
            break;
        }
        start += hop;
        if start >= samples.len() {
            break;
        }
    }

    if frame_count == 0 {
        return vec![0.0; n_coeffs];
    }
    let scale = 1.0 / frame_count as f64;
    accum.iter().map(|&value| (value * scale) as f32).collect()
}

/// Estimate the fundamental frequency (Hz) using a mean-removed, normalized
/// autocorrelation over the 80–400 Hz lag range with parabolic interpolation
/// around the peak.
///
/// Returns `0.0` when the signal is unvoiced (the peak normalized
/// autocorrelation is below [`PITCH_VOICING_THRESHOLD`]), silent, or too short.
pub(crate) fn autocorrelation_f0(samples: &[f32], sample_rate: u32) -> f32 {
    if sample_rate == 0 || samples.len() < 2 {
        return 0.0;
    }
    let min_lag = (sample_rate as f32 / PITCH_MAX_HZ).floor() as usize;
    let max_lag_raw = (sample_rate as f32 / PITCH_MIN_HZ).ceil() as usize;
    let n = samples.len();
    if min_lag < 1 || max_lag_raw <= min_lag || n <= max_lag_raw + 1 {
        return 0.0;
    }
    let max_lag = max_lag_raw.min(n - 1);

    // Mean removal (DC offset rejection).
    let mean = samples.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let signal: Vec<f64> = samples.iter().map(|&x| x as f64 - mean).collect();

    // Zero-lag energy normalizes the autocorrelation into a voicing strength and
    // inherently down-weights longer lags, suppressing octave-down errors.
    let energy0: f64 = signal.iter().map(|&x| x * x).sum();
    if energy0 <= 0.0 {
        return 0.0;
    }

    let mut correlations = vec![0.0f64; max_lag + 1];
    let mut best_lag = 0usize;
    let mut best_corr = 0.0f64;
    for lag in min_lag..=max_lag {
        let mut cross = 0.0;
        for i in lag..n {
            cross += signal[i] * signal[i - lag];
        }
        let corr = cross / energy0;
        correlations[lag] = corr;
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    if best_lag == 0 || best_corr < PITCH_VOICING_THRESHOLD {
        return 0.0;
    }

    // Parabolic interpolation around the discrete autocorrelation peak for a
    // sub-sample lag estimate.
    let refined_lag = if best_lag > min_lag && best_lag < max_lag {
        let prev = correlations[best_lag - 1];
        let curr = correlations[best_lag];
        let next = correlations[best_lag + 1];
        let denom = prev - 2.0 * curr + next;
        if denom.abs() > 1e-12 {
            let delta = 0.5 * (prev - next) / denom;
            best_lag as f64 + delta.clamp(-1.0, 1.0)
        } else {
            best_lag as f64
        }
    } else {
        best_lag as f64
    };

    if refined_lag <= 0.0 {
        return 0.0;
    }
    (sample_rate as f64 / refined_lag) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: u32 = 16_000;

    /// Generate a pure sine wave of `freq` Hz.
    fn sine(freq: f64, len: usize, sample_rate: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate as f64).sin() as f32
            })
            .collect()
    }

    /// Deterministic zero-mean pseudo-random noise in `[-1, 1)` via a linear
    /// congruential generator (avoids the `rand` crate, per the SciRS2 policy).
    fn lcg_noise(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let unit = (state >> 33) as f32 / (1u64 << 31) as f32;
                2.0 * unit - 1.0
            })
            .collect()
    }

    #[test]
    fn test_autocorrelation_f0_sine_220hz() {
        let signal = sine(220.0, SAMPLE_RATE as usize, SAMPLE_RATE);
        let f0 = autocorrelation_f0(&signal, SAMPLE_RATE);
        assert!((f0 - 220.0).abs() < 5.0, "expected ~220 Hz, got {f0} Hz");
    }

    #[test]
    fn test_autocorrelation_f0_unvoiced_returns_zero() {
        // Silence -> unvoiced.
        let silence = vec![0.0f32; SAMPLE_RATE as usize];
        assert_eq!(autocorrelation_f0(&silence, SAMPLE_RATE), 0.0);

        // Broadband noise -> unvoiced (no strong periodicity).
        let noise = lcg_noise(0x1234_5678, SAMPLE_RATE as usize);
        assert_eq!(
            autocorrelation_f0(&noise, SAMPLE_RATE),
            0.0,
            "white noise should be classified unvoiced"
        );
    }

    #[test]
    fn test_mfcc_returns_thirteen_finite_coefficients() {
        let signal = sine(150.0, SAMPLE_RATE as usize, SAMPLE_RATE);
        let mfcc = mfcc_features(&signal, SAMPLE_RATE, 13);
        assert_eq!(mfcc.len(), 13, "expected exactly 13 MFCC coefficients");
        assert!(
            mfcc.iter().all(|c| c.is_finite()),
            "all MFCC coefficients must be finite: {mfcc:?}"
        );
    }

    #[test]
    fn test_spectral_rolloff_lowpass_below_broadband() {
        // Low-pass: a low-frequency tone concentrates energy near DC.
        let lowpass = sine(200.0, SAMPLE_RATE as usize, SAMPLE_RATE);
        // Broadband: white noise spreads energy up to Nyquist.
        let broadband = lcg_noise(0xDEAD_BEEF, SAMPLE_RATE as usize);

        let rolloff_low = spectral_rolloff_hz(&lowpass, SAMPLE_RATE, 0.85);
        let rolloff_broad = spectral_rolloff_hz(&broadband, SAMPLE_RATE, 0.85);

        assert!(
            rolloff_low < rolloff_broad,
            "low-pass rolloff ({rolloff_low} Hz) should be below broadband rolloff ({rolloff_broad} Hz)"
        );
    }

    #[test]
    fn test_spectral_centroid_tracks_tone_frequency() {
        let low = sine(300.0, SAMPLE_RATE as usize, SAMPLE_RATE);
        let high = sine(3000.0, SAMPLE_RATE as usize, SAMPLE_RATE);
        let centroid_low = spectral_centroid_hz(&low, SAMPLE_RATE);
        let centroid_high = spectral_centroid_hz(&high, SAMPLE_RATE);
        assert!(
            centroid_low < centroid_high,
            "centroid of 300 Hz tone ({centroid_low}) should be below 3 kHz tone ({centroid_high})"
        );
    }
}
