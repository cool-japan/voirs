//! Sub-sample (fractional) delay line for HRIR / ITD processing.
//!
//! Inter-aural time differences (ITD) are almost never a whole number of
//! samples: at 44.1 kHz a single sample spans ~22.7 µs, yet the auditory system
//! resolves ITD localisation cues down to ~10 µs. Rounding the delay to the
//! nearest sample therefore discards a perceptually relevant binaural cue.
//!
//! This module implements a *true* fractional delay `y[n] = x[n - d]` for a
//! real-valued delay `d >= 0` using a **windowed-sinc** (band-limited)
//! interpolator. The ideal fractional-delay filter has the infinite impulse
//! response `h[k] = sinc(k - frac)`; we truncate it to a short, symmetric tap
//! window, apply a Blackman window to tame the truncation ripple, and normalise
//! the taps to unity DC gain. Compared with linear interpolation (a first-order
//! approximation that low-pass filters the signal and colours the high end), the
//! windowed-sinc keeps the magnitude response flat across the whole audio band,
//! which is exactly what an accurate, broadband ITD requires.

use scirs2_core::ndarray::Array1;

/// Half the number of interpolation taps. The filter uses `2 * HALF_TAPS`
/// (= 16) taps, which gives sub-sample accuracy across the full audio band
/// while staying cheap enough to run per HRIR.
const HALF_TAPS: isize = 8;

/// Normalised cardinal sine, `sinc(x) = sin(pi x) / (pi x)` with `sinc(0) = 1`.
///
/// For any non-zero integer `x` this is exactly zero, which is what makes a
/// whole-sample delay collapse to a plain shift.
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0
    } else {
        let pi_x = std::f32::consts::PI * x;
        pi_x.sin() / pi_x
    }
}

/// Blackman window centred at `t = 0` over the span `[-half, half]`.
///
/// Returns `1.0` at the centre and tapers smoothly to `0.0` at `±half`,
/// suppressing the side-lobes that truncating the sinc would otherwise add.
fn blackman_centered(t: f32, half: f32) -> f32 {
    if t.abs() >= half {
        return 0.0;
    }
    let phase = std::f32::consts::PI * (t + half) / half;
    0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos()
}

/// Apply a fractional delay of `delay` samples to `signal`, in place.
///
/// Computes the band-limited `y[n] = x[n - delay]` for any real `delay >= 0`.
/// The delay is split into an integer part `floor(delay)` (a plain shift) and a
/// fractional part `frac = delay - floor(delay)` realised by a 16-tap
/// windowed-sinc interpolator. Samples before the start of the signal are
/// treated as zero (zero-padding) and samples shifted past the end are dropped,
/// so the output length always equals the input length. A whole-sample `delay`
/// reproduces an exact integer shift; a `delay` of `0.0` (or anything `<= 0`, or
/// non-finite) leaves the signal untouched.
pub(crate) fn apply_fractional_delay(signal: &mut Array1<f32>, delay: f32) {
    let len = signal.len();
    if len == 0 || delay <= 0.0 || !delay.is_finite() {
        return;
    }

    let floor = delay.floor();
    let int_delay = floor as isize;
    let frac = delay - floor; // fractional part in [0, 1)
    let half = HALF_TAPS as f32;

    // Pre-compute the windowed-sinc tap weights for this fractional part. Tap
    // `offset` reads the input sample at index `out_idx - int_delay + offset`;
    // because `y[n] = x[n - int_delay - frac]`, the sinc is evaluated at
    // `frac + offset`. The weights depend only on `frac`, so they are computed
    // once and reused for every output sample (the delay is constant per call).
    let mut taps: Vec<(isize, f32)> = Vec::with_capacity((2 * HALF_TAPS) as usize);
    let mut tap_sum = 0.0f32;
    for offset in -HALF_TAPS..HALF_TAPS {
        let arg = frac + offset as f32;
        let weight = sinc(arg) * blackman_centered(arg, half);
        tap_sum += weight;
        taps.push((offset, weight));
    }
    if tap_sum.abs() < 1e-12 {
        return;
    }
    let inv_sum = 1.0 / tap_sum; // normalise to unity DC gain

    let input = signal.clone();
    for out_idx in 0..len {
        let base = out_idx as isize - int_delay;
        let mut acc = 0.0f32;
        for &(offset, weight) in &taps {
            let src = base + offset;
            if src >= 0 && (src as usize) < len {
                acc += weight * input[src as usize];
            }
        }
        signal[out_idx] = acc * inv_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_delay_matches_plain_shift() {
        // A whole-sample delay must reproduce an exact shift, with zeros padded
        // in at the start, just like a naive integer delay line would.
        let original: Array1<f32> = Array1::from_vec((1..=20).map(|v| v as f32).collect());
        let mut delayed = original.clone();
        apply_fractional_delay(&mut delayed, 3.0);

        assert_eq!(delayed.len(), original.len(), "length must be preserved");
        for idx in 0..delayed.len() {
            let expected = if idx >= 3 { original[idx - 3] } else { 0.0 };
            assert!(
                (delayed[idx] - expected).abs() < 1e-4,
                "idx={idx}: got {}, expected {expected}",
                delayed[idx]
            );
        }
    }

    #[test]
    fn zero_delay_is_identity() {
        let original = Array1::from_vec(vec![
            0.3_f32, -0.7, 1.2, 0.0, -2.5, 4.1, 0.9, -1.1, 3.3, -0.2,
        ]);
        let mut signal = original.clone();
        apply_fractional_delay(&mut signal, 0.0);

        assert_eq!(signal.len(), original.len(), "length must be preserved");
        for (got, want) in signal.iter().zip(original.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got}, expected {want}");
        }
    }

    #[test]
    fn half_sample_delay_interpolates_ramp_midpoints() {
        // A linear ramp delayed by half a sample should land on (close to) the
        // arithmetic midpoint of adjacent input samples, i.e. x[idx - 0.5].
        // A broken integer-truncation delay would instead return idx (error
        // ~0.5), so this comfortably discriminates a real fractional delay.
        let n = 64usize;
        let original: Array1<f32> = Array1::from_vec((0..n).map(|v| v as f32).collect());
        let mut delayed = original.clone();
        apply_fractional_delay(&mut delayed, 0.5);

        assert_eq!(delayed.len(), original.len(), "length must be preserved");
        for idx in 16..(n - 16) {
            let expected = idx as f32 - 0.5;
            assert!(
                (delayed[idx] - expected).abs() < 5e-2,
                "idx={idx}: got {}, expected ~{expected}",
                delayed[idx]
            );
        }
    }

    #[test]
    fn fractional_delay_matches_bandlimited_sine() {
        // For a band-limited sinusoid the windowed-sinc interpolator should
        // reproduce the analytically delayed signal in the interior, where the
        // zero-padded edges do not yet influence the result.
        let n = 256usize;
        let freq = 0.05_f32; // cycles/sample, well below Nyquist (0.5)
        let delay = 2.7_f32;
        let tone = |i: f32| (2.0 * std::f32::consts::PI * freq * i).sin();

        let original: Array1<f32> = Array1::from_vec((0..n).map(|v| tone(v as f32)).collect());
        let mut delayed = original.clone();
        apply_fractional_delay(&mut delayed, delay);

        assert_eq!(delayed.len(), original.len(), "length must be preserved");
        for idx in 32..(n - 32) {
            let expected = tone(idx as f32 - delay);
            assert!(
                (delayed[idx] - expected).abs() < 1e-2,
                "idx={idx}: got {}, expected {expected}",
                delayed[idx]
            );
        }
    }
}
