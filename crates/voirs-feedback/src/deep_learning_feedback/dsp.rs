//! Real audio DSP primitives for [`super::RealFeatureExtractor`].
//!
//! These free functions implement genuine frequency-domain analysis (MFCC via
//! a mel filterbank + DCT-II) and a time-domain autocorrelation pitch
//! estimator directly from a `&[f32]` signal plus its sample rate -- the same
//! well-tested algorithms already used by `voirs-evaluation::audio_dsp` (that
//! module is `pub(crate)` there and not reusable across the crate boundary,
//! so the pipeline is duplicated here rather than faked).
//!
//! All FFTs use the SciRS2 abstractions (`scirs2_fft::rfft`); no external
//! FFT, `ndarray`, or `rand` crate is used directly, per the SciRS2 policy.
//! Nothing here calls `scirs2_core::random` -- every output is a
//! deterministic function of the real input samples.

/// Lowest fundamental frequency (Hz) considered voiced by [`autocorrelation_f0`].
const PITCH_MIN_HZ: f32 = 80.0;
/// Highest fundamental frequency (Hz) considered voiced by [`autocorrelation_f0`].
const PITCH_MAX_HZ: f32 = 400.0;
/// Minimum normalized autocorrelation for a frame to be treated as voiced.
const PITCH_VOICING_THRESHOLD: f64 = 0.3;

/// Hann window coefficient at index `i` for a window of length `n`.
fn hann(i: usize, n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0)).cos()
}

/// FFT window length for a given sample rate: a power of two close to a
/// 25 ms frame, clamped to `[256, 2048]`.
fn frame_size(sample_rate: u32) -> usize {
    let target = (0.025 * sample_rate as f64).round() as usize;
    let mut n = 256usize;
    while n < target && n < 2048 {
        n *= 2;
    }
    n.clamp(256, 2048)
}

/// Convert a frequency in Hz to the mel scale (O'Shaughnessy formula).
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert a mel value back to Hz.
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

/// Build a triangular mel filterbank as a list of `(bin_index, weight)`
/// pairs per filter, spanning `0..=Nyquist` over `num_filters` filters.
fn mel_filterbank(
    num_filters: usize,
    num_bins: usize,
    n_fft: usize,
    sample_rate: u32,
) -> Vec<Vec<(usize, f64)>> {
    let f_max = sample_rate as f64 / 2.0;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(f_max);
    let point_count = num_filters + 2;

    let mut points = Vec::with_capacity(point_count);
    for i in 0..point_count {
        let mel = mel_min + (mel_max - mel_min) * i as f64 / (point_count - 1) as f64;
        let hz = mel_to_hz(mel);
        let bin = (hz * n_fft as f64 / sample_rate as f64).round() as usize;
        points.push(bin.min(num_bins.saturating_sub(1)));
    }

    let mut filters = Vec::with_capacity(num_filters);
    for m in 1..=num_filters {
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
        if filter.is_empty() {
            filter.push((center, 1.0));
        }
        filters.push(filter);
    }
    filters
}

/// Per-frame outcome of the shared STFT loop: the mel-filtered log energies
/// (used by [`mfcc_frames`]) and the raw power spectrum (used by spectral
/// measurements), one entry per analysis frame.
struct FrameAnalysis {
    log_mel: Vec<f64>,
    power: Vec<f64>,
}

/// Run the shared Hann-windowed, 50%-overlapping STFT + mel-filterbank
/// pipeline over `samples`, returning one [`FrameAnalysis`] per frame.
/// Returns an empty vector for input too short to analyze.
fn analyze_frames(samples: &[f32], sample_rate: u32, num_mel_filters: usize) -> Vec<FrameAnalysis> {
    if samples.len() < 2 || sample_rate == 0 {
        return Vec::new();
    }
    let n_fft = frame_size(sample_rate);
    let num_bins = n_fft / 2 + 1;
    let filterbank = mel_filterbank(num_mel_filters, num_bins, n_fft, sample_rate);
    let hop = (n_fft / 2).max(1);
    let mut frames = Vec::new();
    let mut buffer = vec![0.0f64; n_fft];
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
            frames.push(FrameAnalysis { log_mel, power });
        }
        if available < n_fft {
            break;
        }
        start += hop;
        if start >= samples.len() {
            break;
        }
    }
    frames
}

/// Extract `n_coeffs` MFCCs per analysis frame (Hann window ->
/// `scirs2_fft::rfft` -> power spectrum -> `num_mel_filters`-band
/// triangular mel filterbank -> log energies -> DCT-II, first `n_coeffs`
/// kept). Returns one `Vec<f32>` of length `n_coeffs` per frame; an empty
/// outer vector for input too short to analyze.
pub(super) fn mfcc_frames(
    samples: &[f32],
    sample_rate: u32,
    n_coeffs: usize,
    num_mel_filters: usize,
) -> Vec<Vec<f32>> {
    if n_coeffs == 0 {
        return Vec::new();
    }
    analyze_frames(samples, sample_rate, num_mel_filters)
        .iter()
        .map(|frame| {
            (0..n_coeffs)
                .map(|c| {
                    let mut sum = 0.0;
                    for (m, &log_energy) in frame.log_mel.iter().enumerate() {
                        let angle = std::f64::consts::PI * c as f64 * (m as f64 + 0.5)
                            / num_mel_filters as f64;
                        sum += log_energy * angle.cos();
                    }
                    sum as f32
                })
                .collect()
        })
        .collect()
}

/// Per-frame mel-scale log-energy spectrogram (before the DCT step of
/// [`mfcc_frames`]), one `Vec<f32>` of length `num_mel_filters` per frame.
pub(super) fn mel_spectrogram_frames(
    samples: &[f32],
    sample_rate: u32,
    num_mel_filters: usize,
) -> Vec<Vec<f32>> {
    analyze_frames(samples, sample_rate, num_mel_filters)
        .iter()
        .map(|frame| frame.log_mel.iter().map(|&v| v as f32).collect())
        .collect()
}

/// FFT magnitude-weighted spectral centroid in Hz for one frame's power
/// spectrum: `Σ(f_k·|X_k|) / Σ|X_k|`.
fn frame_spectral_centroid_hz(power: &[f64], sample_rate: u32, n_fft: usize) -> f32 {
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

/// Per-frame spectral centroid in Hz (see [`frame_spectral_centroid_hz`]).
/// Returns an empty vector for input too short to analyze.
pub(super) fn spectral_centroid_frames(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    if samples.len() < 2 || sample_rate == 0 {
        return Vec::new();
    }
    let n_fft = frame_size(sample_rate);
    analyze_frames(samples, sample_rate, 1) // filterbank unused here; num_mel_filters=1 keeps analyze_frames cheap
        .iter()
        .map(|frame| frame_spectral_centroid_hz(&frame.power, sample_rate, n_fft))
        .collect()
}

/// Estimate the fundamental frequency (Hz) using a mean-removed, normalized
/// autocorrelation over the 80-400 Hz lag range with parabolic
/// interpolation around the peak. Returns `0.0` when the signal is
/// unvoiced, silent, or too short.
pub(super) fn autocorrelation_f0(samples: &[f32], sample_rate: u32) -> f32 {
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

    let mean = samples.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let signal: Vec<f64> = samples.iter().map(|&x| x as f64 - mean).collect();

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

/// Root-mean-square amplitude of `samples`. Returns `0.0` for empty input.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Per-frame RMS energy envelope, using the same Hann-windowed 50%-overlap
/// frame grid as the spectral analysis above (energy contour for prosody).
pub(super) fn frame_rms_envelope(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }
    let n_fft = frame_size(sample_rate);
    let hop = (n_fft / 2).max(1);
    let mut envelope = Vec::new();
    let mut start = 0usize;
    loop {
        let available = (samples.len() - start).min(n_fft);
        envelope.push(rms(&samples[start..start + available]));
        if available < n_fft {
            break;
        }
        start += hop;
        if start >= samples.len() {
            break;
        }
    }
    envelope
}

/// Fraction of adjacent sample pairs that cross zero, in `[0, 1]`. A rough,
/// real proxy for spectral noisiness in the time domain.
pub(super) fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let crossings = samples
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    crossings as f32 / (samples.len() - 1) as f32
}

/// Fundamental frequency per analysis frame (200 ms frames, 50% overlap),
/// via [`autocorrelation_f0`] applied to each frame independently.
/// Unvoiced/silent frames report `0.0`.
pub(super) fn windowed_f0_track(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }
    let frame_len = ((0.2 * sample_rate as f64).round() as usize).max(64);
    let hop = (frame_len / 2).max(1);
    let mut track = Vec::new();
    let mut start = 0usize;
    loop {
        let end = (start + frame_len).min(samples.len());
        if end > start {
            track.push(autocorrelation_f0(&samples[start..end], sample_rate));
        }
        if end >= samples.len() {
            break;
        }
        start += hop;
    }
    track
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: u32 = 16_000;

    fn sine(freq: f64, len: usize, sample_rate: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate as f64).sin() as f32
            })
            .collect()
    }

    /// Deterministic zero-mean pseudo-random noise via a linear congruential
    /// generator (never `scirs2_core::random`/`rand`, per the SciRS2
    /// policy -- this is test-only signal generation, not a feature value).
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
        let silence = vec![0.0f32; SAMPLE_RATE as usize];
        assert_eq!(autocorrelation_f0(&silence, SAMPLE_RATE), 0.0);

        let noise = lcg_noise(0x1234_5678, SAMPLE_RATE as usize);
        assert_eq!(autocorrelation_f0(&noise, SAMPLE_RATE), 0.0);
    }

    #[test]
    fn test_autocorrelation_f0_varies_with_frequency() {
        // The defining property a fabricated/random F0 could never have:
        // real, distinct input frequencies must produce distinct outputs.
        let f0_low =
            autocorrelation_f0(&sine(150.0, SAMPLE_RATE as usize, SAMPLE_RATE), SAMPLE_RATE);
        let f0_high =
            autocorrelation_f0(&sine(300.0, SAMPLE_RATE as usize, SAMPLE_RATE), SAMPLE_RATE);
        assert!(
            (f0_low - 150.0).abs() < 5.0,
            "expected ~150 Hz, got {f0_low}"
        );
        assert!(
            (f0_high - 300.0).abs() < 5.0,
            "expected ~300 Hz, got {f0_high}"
        );
        assert!(f0_low < f0_high);
    }

    #[test]
    fn test_mfcc_frames_returns_finite_coefficients_and_varies_with_input() {
        let tone = sine(150.0, SAMPLE_RATE as usize, SAMPLE_RATE);
        let mfcc = mfcc_frames(&tone, SAMPLE_RATE, 13, 26);
        assert!(!mfcc.is_empty(), "expected at least one analysis frame");
        for frame in &mfcc {
            assert_eq!(frame.len(), 13);
            assert!(frame.iter().all(|c| c.is_finite()));
        }

        // A different real signal must produce genuinely different
        // coefficients -- the property a `scirs2_core::random`-based mock
        // could never guarantee to fail on repeat runs, but which real DSP
        // guarantees deterministically.
        let noise = lcg_noise(0xABCD_EF01, SAMPLE_RATE as usize);
        let mfcc_noise = mfcc_frames(&noise, SAMPLE_RATE, 13, 26);
        assert_ne!(mfcc[0], mfcc_noise[0]);
    }

    #[test]
    fn test_mfcc_frames_deterministic_across_calls() {
        let signal = sine(220.0, SAMPLE_RATE as usize / 4, SAMPLE_RATE);
        let a = mfcc_frames(&signal, SAMPLE_RATE, 13, 26);
        let b = mfcc_frames(&signal, SAMPLE_RATE, 13, 26);
        assert_eq!(a, b, "MFCC extraction must be a pure function of the input");
    }

    #[test]
    fn test_mel_spectrogram_frames_shape() {
        let signal = sine(200.0, SAMPLE_RATE as usize / 2, SAMPLE_RATE);
        let mel = mel_spectrogram_frames(&signal, SAMPLE_RATE, 80);
        assert!(!mel.is_empty());
        assert!(mel.iter().all(|frame| frame.len() == 80));
    }

    #[test]
    fn test_spectral_centroid_frames_tracks_tone_frequency() {
        let low = sine(300.0, SAMPLE_RATE as usize, SAMPLE_RATE);
        let high = sine(3000.0, SAMPLE_RATE as usize, SAMPLE_RATE);
        let centroid_low: f32 = {
            let frames = spectral_centroid_frames(&low, SAMPLE_RATE);
            frames.iter().sum::<f32>() / frames.len() as f32
        };
        let centroid_high: f32 = {
            let frames = spectral_centroid_frames(&high, SAMPLE_RATE);
            frames.iter().sum::<f32>() / frames.len() as f32
        };
        assert!(
            centroid_low < centroid_high,
            "centroid of 300 Hz tone ({centroid_low}) should be below 3 kHz tone ({centroid_high})"
        );
    }

    #[test]
    fn test_windowed_f0_track_detects_voiced_region() {
        let signal = sine(180.0, SAMPLE_RATE as usize, SAMPLE_RATE);
        let track = windowed_f0_track(&signal, SAMPLE_RATE);
        assert!(!track.is_empty());
        assert!(track.iter().any(|&f0| (f0 - 180.0).abs() < 10.0));
    }

    #[test]
    fn test_short_input_returns_empty_not_panic() {
        assert!(mfcc_frames(&[0.1], SAMPLE_RATE, 13, 26).is_empty());
        assert!(mel_spectrogram_frames(&[], SAMPLE_RATE, 80).is_empty());
        assert!(spectral_centroid_frames(&[0.1], SAMPLE_RATE).is_empty());
        assert!(windowed_f0_track(&[], SAMPLE_RATE).is_empty());
        assert!(frame_rms_envelope(&[], SAMPLE_RATE).is_empty());
    }

    #[test]
    fn test_rms_and_zcr() {
        assert_eq!(rms(&[]), 0.0);
        assert_eq!(rms(&[2.0, -2.0]), 2.0);
        assert_eq!(zero_crossing_rate(&[1.0, 1.0, 1.0]), 0.0);
        assert!(zero_crossing_rate(&[1.0, -1.0, 1.0, -1.0]) > 0.9);
    }

    #[test]
    fn test_frame_rms_envelope_tracks_amplitude_change() {
        let mut samples = sine(220.0, SAMPLE_RATE as usize / 2, SAMPLE_RATE);
        samples.extend(vec![0.0f32; SAMPLE_RATE as usize / 2]);
        let envelope = frame_rms_envelope(&samples, SAMPLE_RATE);
        assert!(!envelope.is_empty());
        assert!(envelope.first().copied().unwrap_or(0.0) > envelope.last().copied().unwrap_or(1.0));
    }
}
