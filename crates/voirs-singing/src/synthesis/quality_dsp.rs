//! Real signal-analysis primitives for synthesized-audio quality metrics.
//!
//! These free functions operate on a raw `&[f32]` waveform plus its sample rate
//! and provide the measured quantities used by
//! [`super::core::SynthesisEngine`] to populate
//! [`super::results::QualityMetrics`]. They replace the previous hard-coded
//! placeholder constants with genuine DSP:
//!
//! * Harmonic-to-noise ratio (HNR) via normalized autocorrelation
//!   ([`estimate_harmonicity`]) → harmonic quality and noise level.
//! * Spectral-flatness / centroid-plausibility ([`spectral_quality_score`]).
//! * LPC all-pole envelope prominence ([`formant_clarity_score`]).
//! * Autocorrelation F0-stability fallback ([`f0_stability_score`]).
//!
//! All frequency-domain analysis uses the SciRS2 FFT abstractions
//! (`scirs2_fft::rfft` / `scirs2_fft::fft`) and `scirs2_core::Complex`, per the
//! SciRS2 policy. No `rand`, `ndarray`, `rayon`, `num_complex`, or `nalgebra`
//! crates are used directly.

/// Lowest fundamental frequency (Hz) considered when searching for periodicity.
const PITCH_MIN_HZ: f32 = 70.0;
/// Highest fundamental frequency (Hz) considered when searching for periodicity.
const PITCH_MAX_HZ: f32 = 800.0;
/// Analysis frame length in seconds for time-domain (autocorrelation) measures.
const FRAME_SECONDS: f32 = 0.040;
/// Minimum analysis frame length in samples.
const MIN_FRAME: usize = 256;
/// Normalized-autocorrelation peak above which a frame is treated as voiced.
const VOICING_THRESHOLD: f32 = 0.5;
/// Upper clamp on the harmonic ratio, keeping `HNR = 10·log10(r/(1-r))` finite.
const MAX_RATIO: f32 = 0.999_999;
/// HNR (dB) that maps to a perfect harmonic-quality score of 1.0.
const HNR_FULLSCALE_DB: f32 = 20.0;
/// Coefficient of variation of voiced F0 that maps stability to 0.0.
const CV_FULLSCALE: f32 = 0.1;
/// LPC envelope dynamic range (dB) that maps formant clarity to 1.0.
const FORMANT_FULLSCALE_DB: f32 = 30.0;

/// Harmonicity estimate for a waveform.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Harmonicity {
    /// Mean normalized-autocorrelation peak over active frames, in `[0, 1]`.
    /// Interpreted as the fraction of frame energy that is periodic.
    pub harmonic_ratio: f32,
    /// Harmonic-to-noise ratio in decibels, `10·log10(r / (1 - r))`.
    pub hnr_db: f32,
}

/// Estimate harmonic-to-noise ratio across the waveform.
///
/// For each active (above-silence) frame the maximum normalized cross-correlation
/// over the pitch-period lag range is taken as that frame's harmonic ratio `r`.
/// The mean `r` over active frames is converted to HNR with
/// `10·log10(r / (1 - r))` (Boersma-style autocorrelation HNR).
pub(crate) fn estimate_harmonicity(audio: &[f32], sample_rate: f32) -> Harmonicity {
    let empty = Harmonicity {
        harmonic_ratio: 0.0,
        hnr_db: 0.0,
    };
    if audio.is_empty() || sample_rate <= 0.0 {
        return empty;
    }

    let frame = ((sample_rate * FRAME_SECONDS) as usize).max(MIN_FRAME);
    let hop = (frame / 2).max(1);
    let min_lag = (sample_rate / PITCH_MAX_HZ).floor().max(2.0) as usize;
    let max_lag = (sample_rate / PITCH_MIN_HZ).ceil() as usize;

    // Collect (ratio, energy) per analyzable frame.
    let mut frames: Vec<(f32, f32)> = Vec::new();
    let mut start = 0usize;
    while start + frame <= audio.len() {
        let slice = &audio[start..start + frame];
        if let Some((ratio, _lag)) = frame_autocorr_peak(slice, min_lag, max_lag) {
            let energy: f32 = slice.iter().map(|&x| x * x).sum();
            frames.push((ratio, energy));
        }
        start += hop;
    }
    // Short signals: analyze the whole buffer as a single frame.
    if frames.is_empty() {
        if let Some((ratio, _lag)) = frame_autocorr_peak(audio, min_lag, max_lag) {
            let energy: f32 = audio.iter().map(|&x| x * x).sum();
            frames.push((ratio, energy));
        }
    }
    if frames.is_empty() {
        return empty;
    }

    // Gate out near-silent frames so quiet tails do not depress the HNR.
    let max_energy = frames.iter().map(|&(_, e)| e).fold(0.0f32, f32::max);
    let gate = max_energy * 0.1;
    let active: Vec<f32> = frames
        .iter()
        .filter(|&&(_, e)| e > 0.0 && e >= gate)
        .map(|&(r, _)| r)
        .collect();
    let ratios = if active.is_empty() {
        frames.iter().map(|&(r, _)| r).collect::<Vec<f32>>()
    } else {
        active
    };

    let mean_ratio = (ratios.iter().sum::<f32>() / ratios.len() as f32).clamp(0.0, MAX_RATIO);
    let hnr_db = if mean_ratio <= 0.0 {
        0.0
    } else {
        10.0 * (mean_ratio / (1.0 - mean_ratio)).log10()
    };
    Harmonicity {
        harmonic_ratio: mean_ratio,
        hnr_db,
    }
}

/// Map an HNR value in dB to a harmonic-quality score in `[0, 1]`.
pub(crate) fn harmonic_quality_from_hnr(hnr_db: f32) -> f32 {
    (hnr_db / HNR_FULLSCALE_DB).clamp(0.0, 1.0)
}

/// Maximum normalized cross-correlation over the lag range and the lag at which
/// it occurs. Returns `None` for silent or too-short frames.
///
/// The correlation at each lag is normalized by the geometric mean of the two
/// overlapping windows' energies, making it amplitude-independent and bounded by
/// `1.0` for a perfectly periodic frame.
fn frame_autocorr_peak(frame: &[f32], min_lag: usize, max_lag: usize) -> Option<(f32, usize)> {
    let n = frame.len();
    if n < min_lag + 2 {
        return None;
    }
    let mean = frame.iter().sum::<f32>() / n as f32;
    let signal: Vec<f32> = frame.iter().map(|&x| x - mean).collect();
    let energy0: f32 = signal.iter().map(|&x| x * x).sum();
    if energy0 <= 1e-9 {
        return None;
    }
    let upper = max_lag.min(n - 1);
    if upper <= min_lag {
        return None;
    }

    let mut best_ratio = 0.0f32;
    let mut best_lag = 0usize;
    for lag in min_lag..=upper {
        let (cross, e1, e2) = signal[..n - lag]
            .iter()
            .zip(&signal[lag..])
            .fold((0.0f32, 0.0f32, 0.0f32), |(c, a1, a2), (&a, &b)| {
                (c + a * b, a1 + a * a, a2 + b * b)
            });
        let denom = (e1 * e2).sqrt();
        if denom > 1e-9 {
            let ratio = cross / denom;
            if ratio > best_ratio {
                best_ratio = ratio;
                best_lag = lag;
            }
        }
    }
    if best_lag == 0 {
        None
    } else {
        Some((best_ratio.clamp(0.0, 1.0), best_lag))
    }
}

/// Score F0 stability in `[0, 1]` from voiced-frame pitch consistency.
///
/// Used as the pitch-accuracy fallback when no target note pitches are available
/// (or when target matching fails). A high, consistently voiced F0 contour with
/// low coefficient of variation scores near `1.0`; sparse or erratic voicing
/// (e.g. broadband noise) scores near `0.0`.
pub(crate) fn f0_stability_score(audio: &[f32], sample_rate: f32) -> f32 {
    if audio.len() < MIN_FRAME || sample_rate <= 0.0 {
        return 0.0;
    }
    let frame = ((sample_rate * FRAME_SECONDS) as usize).max(MIN_FRAME);
    let hop = (frame / 2).max(1);
    let min_lag = (sample_rate / PITCH_MAX_HZ).floor().max(2.0) as usize;
    let max_lag = (sample_rate / PITCH_MIN_HZ).ceil() as usize;

    let mut total = 0usize;
    let mut f0s: Vec<f32> = Vec::new();
    let mut start = 0usize;
    while start + frame <= audio.len() {
        total += 1;
        if let Some((ratio, lag)) =
            frame_autocorr_peak(&audio[start..start + frame], min_lag, max_lag)
        {
            if ratio >= VOICING_THRESHOLD && lag > 0 {
                f0s.push(sample_rate / lag as f32);
            }
        }
        start += hop;
    }
    if total == 0 {
        return 0.0;
    }
    let voiced_fraction = f0s.len() as f32 / total as f32;
    if f0s.len() < 2 {
        return (voiced_fraction * 0.5).clamp(0.0, 1.0);
    }
    let mean = f0s.iter().sum::<f32>() / f0s.len() as f32;
    if mean <= 0.0 {
        return 0.0;
    }
    let variance = f0s.iter().map(|&f| (f - mean).powi(2)).sum::<f32>() / f0s.len() as f32;
    let cv = variance.sqrt() / mean;
    let stability = (1.0 - cv / CV_FULLSCALE).clamp(0.0, 1.0);
    (stability * voiced_fraction).clamp(0.0, 1.0)
}

/// Score spectral quality in `[0, 1]` from envelope tonality and centroid
/// plausibility.
///
/// Combines two cues equally: spectral tonality (`1 - spectral flatness`, high
/// for harmonic signals, low for noise) and the plausibility of the spectral
/// centroid for singing (a smooth plateau over roughly 200 Hz – 3.5 kHz).
pub(crate) fn spectral_quality_score(audio: &[f32], sample_rate: f32) -> f32 {
    if audio.is_empty() || sample_rate <= 0.0 {
        return 0.0;
    }
    let n_fft = analysis_fft_size(sample_rate);
    let power = match averaged_power_spectrum(audio, n_fft) {
        Some(power) => power,
        None => return 0.0,
    };
    let centroid = spectral_centroid_from_power(&power, sample_rate, n_fft);
    let plausibility = centroid_plausibility(centroid);
    let tonality = (1.0 - spectral_flatness(&power)).clamp(0.0, 1.0);
    (0.5 * plausibility + 0.5 * tonality).clamp(0.0, 1.0)
}

/// Score LPC-based formant clarity in `[0, 1]`.
///
/// Fits an all-pole (LPC) model to the highest-energy segment of the waveform
/// and measures the dynamic range of its spectral envelope across the formant
/// band (200 Hz – 5 kHz). Prominent, well-defined resonances produce a large
/// peak-to-valley range and a high score; a flat (noise-like) envelope scores
/// low.
pub(crate) fn formant_clarity_score(audio: &[f32], sample_rate: f32) -> f32 {
    if audio.is_empty() || sample_rate <= 0.0 {
        return 0.0;
    }
    let segment = highest_energy_segment(audio, sample_rate);
    let order = ((sample_rate as usize) / 1000 + 2).clamp(8, 30);
    let lpc = match lpc_coefficients(segment, order) {
        Some(lpc) => lpc,
        None => return 0.0,
    };
    lpc_envelope_clarity(&lpc, sample_rate, 200.0, 5000.0)
}

/// Hann window coefficient at index `i` for a window of length `n`.
fn hann(i: usize, n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0)).cos()
}

/// FFT length: a power of two near a 25 ms frame, clamped to `[256, 2048]`.
fn analysis_fft_size(sample_rate: f32) -> usize {
    let target = (0.025 * sample_rate as f64).round() as usize;
    let mut n = 256usize;
    while n < target && n < 2048 {
        n *= 2;
    }
    n.clamp(256, 2048)
}

/// Average power spectrum across Hann-windowed, 50%-overlapping frames.
///
/// Returns `n_fft / 2 + 1` mean `|X_k|^2` values, or `None` when the input is
/// too short. Uses `scirs2_fft::rfft` for each frame.
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

/// Magnitude-weighted spectral centroid (Hz) from a power spectrum.
fn spectral_centroid_from_power(power: &[f64], sample_rate: f32, n_fft: usize) -> f32 {
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

/// Spectral flatness (Wiener entropy) in `[0, 1]`: geometric mean over arithmetic
/// mean of the power bins (excluding DC). Near `1` for white noise, near `0` for
/// tonal signals.
fn spectral_flatness(power: &[f64]) -> f32 {
    let bins = &power[1.min(power.len())..];
    if bins.is_empty() {
        return 1.0;
    }
    let mut log_sum = 0.0f64;
    let mut arith_sum = 0.0f64;
    for &p in bins {
        let value = p.max(0.0) + 1e-12;
        log_sum += value.ln();
        arith_sum += value;
    }
    let count = bins.len() as f64;
    let geo_mean = (log_sum / count).exp();
    let arith_mean = arith_sum / count;
    if arith_mean <= 0.0 {
        return 1.0;
    }
    ((geo_mean / arith_mean) as f32).clamp(0.0, 1.0)
}

/// Plausibility in `[0, 1]` of a spectral centroid for a singing voice: a plateau
/// over 200 Hz – 3.5 kHz with linear ramps down to 50 Hz and up to 9 kHz.
fn centroid_plausibility(centroid_hz: f32) -> f32 {
    const LOW_ZERO: f32 = 50.0;
    const LOW_FULL: f32 = 200.0;
    const HIGH_FULL: f32 = 3500.0;
    const HIGH_ZERO: f32 = 9000.0;
    if centroid_hz < LOW_FULL {
        // Ramp up from 0 at LOW_ZERO to 1 at LOW_FULL.
        ((centroid_hz - LOW_ZERO) / (LOW_FULL - LOW_ZERO)).clamp(0.0, 1.0)
    } else if centroid_hz <= HIGH_FULL {
        1.0
    } else {
        // Ramp down from 1 at HIGH_FULL to 0 at HIGH_ZERO.
        ((HIGH_ZERO - centroid_hz) / (HIGH_ZERO - HIGH_FULL)).clamp(0.0, 1.0)
    }
}

/// Return the highest-energy ~80 ms window of the waveform (or the whole buffer
/// when shorter), focusing formant analysis on a sustained, voiced region.
fn highest_energy_segment(audio: &[f32], sample_rate: f32) -> &[f32] {
    let win = ((sample_rate * 0.080) as usize).max(MIN_FRAME);
    if audio.len() <= win {
        return audio;
    }
    let hop = (win / 2).max(1);
    let mut best_start = 0usize;
    let mut best_energy = -1.0f32;
    let mut start = 0usize;
    while start + win <= audio.len() {
        let energy: f32 = audio[start..start + win].iter().map(|&x| x * x).sum();
        if energy > best_energy {
            best_energy = energy;
            best_start = start;
        }
        start += hop;
    }
    &audio[best_start..best_start + win]
}

/// Compute LPC coefficients `[1, a_1, ..., a_order]` via the Levinson-Durbin
/// recursion over the autocorrelation of `samples`. Returns `None` for
/// degenerate (too short or zero-energy) input.
fn lpc_coefficients(samples: &[f32], order: usize) -> Option<Vec<f64>> {
    let n = samples.len();
    if order == 0 || n <= order + 1 {
        return None;
    }
    let mut autocorr = vec![0.0f64; order + 1];
    for (k, slot) in autocorr.iter_mut().enumerate() {
        *slot = samples[..n - k]
            .iter()
            .zip(&samples[k..])
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
    }
    if autocorr[0].abs() < 1e-12 {
        return None;
    }

    let mut a = vec![0.0f64; order + 1];
    let mut prev = vec![0.0f64; order + 1];
    let mut err = autocorr[0];
    for i in 1..=order {
        let mut acc = autocorr[i];
        for j in 1..i {
            acc -= a[j] * autocorr[i - j];
        }
        let reflection = acc / err;
        prev[..=order].clone_from_slice(&a[..=order]);
        for j in 1..i {
            a[j] = prev[j] - reflection * prev[i - j];
        }
        a[i] = reflection;
        err *= 1.0 - reflection * reflection;
        if err <= 0.0 {
            break;
        }
    }
    a[0] = 1.0;
    Some(a)
}

/// Dynamic range (mapped to `[0, 1]`) of the LPC all-pole envelope `1/|A(e^{jw})|`
/// across the `[lo_hz, hi_hz]` band. A peaky (formant-rich) envelope yields a
/// large peak-to-valley range; a flat envelope yields ~0.
fn lpc_envelope_clarity(lpc: &[f64], sample_rate: f32, lo_hz: f32, hi_hz: f32) -> f32 {
    const FFT_SIZE: usize = 512;
    let order = lpc.len().saturating_sub(1);
    if order == 0 || sample_rate <= 0.0 {
        return 0.0;
    }
    let mut buffer: Vec<scirs2_core::Complex<f64>> =
        vec![scirs2_core::Complex::new(0.0, 0.0); FFT_SIZE];
    buffer[0] = scirs2_core::Complex::new(1.0, 0.0);
    for k in 1..=order.min(FFT_SIZE - 1) {
        buffer[k] = scirs2_core::Complex::new(-lpc[k], 0.0);
    }
    let spectrum = match scirs2_fft::fft(&buffer, None) {
        Ok(spectrum) => spectrum,
        Err(_) => return 0.0,
    };

    let bin_hz = sample_rate / FFT_SIZE as f32;
    let lo_bin = (lo_hz / bin_hz).floor().max(1.0) as usize;
    let hi_bin = ((hi_hz / bin_hz).ceil() as usize).min(FFT_SIZE / 2 - 1);
    if lo_bin >= hi_bin {
        return 0.0;
    }

    let mut max_env = f32::MIN;
    let mut min_env = f32::MAX;
    let mut found = false;
    for value in spectrum.iter().take(hi_bin + 1).skip(lo_bin) {
        let mag = value.norm() as f32;
        if mag <= 1e-10 {
            continue;
        }
        let env = 1.0 / mag;
        max_env = max_env.max(env);
        min_env = min_env.min(env);
        found = true;
    }
    if !found || min_env <= 0.0 {
        return 0.0;
    }
    let range_db = 20.0 * (max_env / min_env).log10();
    (range_db / FORMANT_FULLSCALE_DB).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 22_050.0;
    const LEN: usize = 11_025; // 0.5 s

    /// Generate a pure sine tone.
    fn sine(freq: f32, amplitude: f32, len: usize, sample_rate: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate;
                amplitude * (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    /// Deterministic zero-mean pseudo-random noise via a linear congruential
    /// generator (no `rand` crate, per the SciRS2 policy).
    fn lcg_noise(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let unit = (state >> 33) as f32 / (1u64 << 31) as f32;
                0.8 * (2.0 * unit - 1.0)
            })
            .collect()
    }

    #[test]
    fn test_harmonicity_tone_vs_noise() {
        let tone = sine(440.0, 0.8, LEN, SAMPLE_RATE);
        let noise = lcg_noise(0x1234_5678, LEN);

        let tone_h = estimate_harmonicity(&tone, SAMPLE_RATE);
        let noise_h = estimate_harmonicity(&noise, SAMPLE_RATE);

        assert!(
            tone_h.harmonic_ratio > 0.9,
            "clean tone harmonic ratio should be high, got {}",
            tone_h.harmonic_ratio
        );
        assert!(
            tone_h.hnr_db > 12.0,
            "clean tone HNR should be high, got {} dB",
            tone_h.hnr_db
        );
        assert!(
            noise_h.harmonic_ratio < 0.5,
            "white noise harmonic ratio should be low, got {}",
            noise_h.harmonic_ratio
        );
        assert!(
            tone_h.harmonic_ratio > noise_h.harmonic_ratio,
            "tone must be more harmonic than noise"
        );
    }

    #[test]
    fn test_harmonic_quality_and_noise_level_mapping() {
        let tone = estimate_harmonicity(&sine(440.0, 0.8, LEN, SAMPLE_RATE), SAMPLE_RATE);
        let noise = estimate_harmonicity(&lcg_noise(0xABCD, LEN), SAMPLE_RATE);

        let tone_hq = harmonic_quality_from_hnr(tone.hnr_db);
        let noise_hq = harmonic_quality_from_hnr(noise.hnr_db);
        assert!(tone_hq > 0.7, "tone harmonic quality {tone_hq}");
        assert!(noise_hq < 0.3, "noise harmonic quality {noise_hq}");

        let tone_noise_level = (1.0 - tone.harmonic_ratio).clamp(0.0, 1.0);
        let noise_noise_level = (1.0 - noise.harmonic_ratio).clamp(0.0, 1.0);
        assert!(
            tone_noise_level < 0.2,
            "tone noise level {tone_noise_level}"
        );
        assert!(
            noise_noise_level > 0.5,
            "noise noise level {noise_noise_level}"
        );
    }

    #[test]
    fn test_spectral_quality_tone_high_noise_low() {
        let tone = spectral_quality_score(&sine(440.0, 0.8, LEN, SAMPLE_RATE), SAMPLE_RATE);
        let noise = spectral_quality_score(&lcg_noise(0x55AA, LEN), SAMPLE_RATE);
        assert!(tone > 0.6, "tone spectral quality {tone}");
        assert!(noise < 0.4, "noise spectral quality {noise}");
        assert!(tone > noise);
    }

    #[test]
    fn test_formant_clarity_tone_exceeds_noise() {
        let tone = formant_clarity_score(&sine(300.0, 0.8, LEN, SAMPLE_RATE), SAMPLE_RATE);
        let noise = formant_clarity_score(&lcg_noise(0x9E37, LEN), SAMPLE_RATE);
        assert!((0.0..=1.0).contains(&tone));
        assert!((0.0..=1.0).contains(&noise));
        assert!(
            tone > noise,
            "tone formant clarity ({tone}) should exceed noise ({noise})"
        );
    }

    #[test]
    fn test_f0_stability_tone_high_noise_low() {
        let tone = f0_stability_score(&sine(440.0, 0.8, LEN, SAMPLE_RATE), SAMPLE_RATE);
        let noise = f0_stability_score(&lcg_noise(0xFEED, LEN), SAMPLE_RATE);
        assert!(tone > 0.8, "tone F0 stability {tone}");
        assert!(noise < 0.4, "noise F0 stability {noise}");
    }

    #[test]
    fn test_empty_and_silent_inputs_are_safe() {
        assert_eq!(estimate_harmonicity(&[], SAMPLE_RATE).harmonic_ratio, 0.0);
        assert_eq!(spectral_quality_score(&[], SAMPLE_RATE), 0.0);
        assert_eq!(formant_clarity_score(&[], SAMPLE_RATE), 0.0);
        assert_eq!(f0_stability_score(&[], SAMPLE_RATE), 0.0);

        let silence = vec![0.0f32; LEN];
        // Silent input must not panic and must report no harmonic content.
        let h = estimate_harmonicity(&silence, SAMPLE_RATE);
        assert_eq!(h.harmonic_ratio, 0.0);
        assert_eq!(f0_stability_score(&silence, SAMPLE_RATE), 0.0);
    }
}
