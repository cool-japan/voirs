//! Signal processing helper methods

use crate::{
    transforms::{PitchTransform, Transform},
    Error, Result,
};
use scirs2_core::Complex;
use scirs2_fft::RealFftPlanner;
use std::f32::consts::PI;

use super::converter::VoiceConverter;

/// STFT frame size used by the spectral-envelope formant warping routines.
const FORMANT_FRAME: usize = 1024;
/// STFT hop size (75 % overlap) used by the formant warping routines.
const FORMANT_HOP: usize = 256;

impl VoiceConverter {
    /// Adjust formants based on age
    pub(super) fn adjust_formants(
        &self,
        audio: &[f32],
        source_age: f32,
        target_age: f32,
    ) -> Result<Vec<f32>> {
        // Age affects vocal tract length: children have shorter vocal tracts
        let formant_factor = (source_age / target_age).sqrt();
        self.shift_formants(audio, (formant_factor - 1.0) * 0.5)
    }

    /// Adjust vocal tract length simulation.
    ///
    /// A longer vocal tract lowers the formant frequencies (and vice versa), which
    /// is a *warping of the spectral envelope* rather than a flat gain change. The
    /// envelope is compressed/expanded along the frequency axis by `length_factor`
    /// while the underlying harmonic fine structure (pitch) is preserved.
    ///
    /// `length_factor > 1.0` simulates a longer tract → formants shift **down**;
    /// `length_factor < 1.0` simulates a shorter tract → formants shift **up**.
    pub(super) fn adjust_vocal_tract_length(
        &self,
        audio: &[f32],
        length_factor: f32,
    ) -> Result<Vec<f32>> {
        if (length_factor - 1.0).abs() < f32::EPSILON {
            return Ok(audio.to_vec());
        }

        // Longer tract (length_factor > 1) lowers formants, i.e. the envelope is
        // scaled toward lower frequencies. The envelope-warp routine interprets its
        // `warp_factor` as a multiplier on the frequency axis of the *envelope*,
        // so we pass the reciprocal: scaling the envelope by 1/length_factor moves
        // a formant at frequency f to f/length_factor.
        self.warp_spectral_envelope(audio, 1.0 / length_factor)
    }

    /// Shift formant frequencies by warping the short-time spectral envelope.
    ///
    /// `shift_factor` is a fractional shift: the spectral envelope (and therefore
    /// every formant) is multiplied along the frequency axis by `1.0 + shift_factor`.
    /// Positive values brighten the timbre (formants move up); negative values
    /// darken it (formants move down). Pitch is preserved because only the spectral
    /// *envelope* is warped — the harmonic fine structure is divided out and
    /// re-applied unchanged.
    pub(super) fn shift_formants(&self, audio: &[f32], shift_factor: f32) -> Result<Vec<f32>> {
        if shift_factor.abs() < f32::EPSILON {
            return Ok(audio.to_vec());
        }

        self.warp_spectral_envelope(audio, 1.0 + shift_factor)
    }

    /// Shift fundamental frequency
    pub(super) fn shift_f0(&self, audio: &[f32], f0_factor: f32) -> Result<Vec<f32>> {
        if f0_factor == 1.0 {
            return Ok(audio.to_vec());
        }

        let pitch_transform = PitchTransform::new(f0_factor);
        pitch_transform.apply(audio)
    }

    /// Modulate pitch contour for emotional expression
    pub(super) fn modulate_pitch_contour(
        &self,
        audio: &[f32],
        variation_factor: f32,
    ) -> Result<Vec<f32>> {
        let mut result = audio.to_vec();

        // Apply sinusoidal modulation to simulate pitch contour changes
        for (i, sample) in result.iter_mut().enumerate() {
            let modulation = 1.0 + (i as f32 * 0.01).sin() * (variation_factor - 1.0) * 0.1;
            *sample *= modulation;
        }

        Ok(result)
    }

    /// Adjust spectral tilt with a first-order shelving filter.
    ///
    /// The previous implementation multiplied sample `i` by `1 + (i/len)·tilt`,
    /// which is a *time-position* ramp and has no relation to spectral tilt. This
    /// implementation applies a real first-order high-shelf: the signal is split
    /// into a one-pole low-passed component `lp` and its high-frequency residual
    /// `x - lp`, and the residual is scaled by a tilt-derived gain before being
    /// recombined:
    ///
    /// ```text
    /// lp[n] = a·x[n] + (1 - a)·lp[n-1]
    /// out[n] = lp[n] + (x[n] - lp[n]) · hf_gain
    /// ```
    ///
    /// `tilt_factor > 0` boosts the high band (brighter, positive spectral tilt);
    /// `tilt_factor < 0` cuts it (darker). The operation is stateful and entirely
    /// in the time domain (no FFT required).
    pub(super) fn adjust_spectral_tilt(&self, audio: &[f32], tilt_factor: f32) -> Result<Vec<f32>> {
        if tilt_factor.abs() < f32::EPSILON || audio.is_empty() {
            return Ok(audio.to_vec());
        }

        // One-pole low-pass coefficient. A moderate cutoff splits the spectrum into
        // a "low" and "high" shelf around roughly a quarter of the sample rate.
        let a = 0.25_f32;

        // Map the (unbounded) tilt factor to a strictly positive high-frequency gain.
        // tilt == 0 → unity; positive → >1 (boost); negative → <1 (cut). Using the
        // exponential keeps the gain positive for any tilt magnitude.
        let hf_gain = tilt_factor.exp();

        let mut result = Vec::with_capacity(audio.len());
        let mut lp = audio[0];
        for &x in audio {
            lp = a * x + (1.0 - a) * lp;
            let hf = x - lp;
            result.push(lp + hf * hf_gain);
        }
        Ok(result)
    }

    /// Warp the short-time spectral envelope by `warp_factor` along the frequency
    /// axis while preserving the harmonic fine structure (pitch).
    ///
    /// For each STFT frame:
    /// 1. Estimate the spectral envelope by smoothing the magnitude spectrum
    ///    (moving average over a small bin window).
    /// 2. Build the warped envelope by sampling the original envelope at
    ///    `bin / warp_factor` (linear interpolation). A formant originally at bin
    ///    `b` thus moves to bin `b · warp_factor`.
    /// 3. Multiply each complex bin by `warped_env / orig_env` so the fine
    ///    structure (the ratio of the raw spectrum to its own envelope) is left
    ///    intact and only the envelope is relocated.
    /// 4. Inverse FFT and overlap-add with a synthesis Hann window.
    ///
    /// Returns the resynthesised signal, length-matched to the input.
    fn warp_spectral_envelope(&self, audio: &[f32], warp_factor: f32) -> Result<Vec<f32>> {
        if audio.len() < FORMANT_FRAME || (warp_factor - 1.0).abs() < f32::EPSILON {
            return Ok(audio.to_vec());
        }

        let frame = FORMANT_FRAME;
        let hop = FORMANT_HOP;
        let num_bins = frame / 2 + 1;

        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(frame);
        let inv = planner.plan_fft_inverse(frame);

        let window: Vec<f32> = (0..frame)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / frame as f32).cos())
            .collect();

        let out_len = audio.len();
        let mut output = vec![0.0_f32; out_len + frame];
        let mut ola_sum = vec![0.0_f32; out_len + frame];

        let mut padded = audio.to_vec();
        padded.resize(audio.len() + frame, 0.0);

        // Per-frame working buffers.
        let mut windowed = vec![0.0_f32; frame];
        let mut spectrum = vec![Complex::new(0.0_f32, 0.0_f32); num_bins];
        let mut time_out = vec![0.0_f32; frame];

        let mut pos = 0_usize;
        while pos + frame <= padded.len() {
            for (w, (&s, &win)) in windowed
                .iter_mut()
                .zip(padded[pos..pos + frame].iter().zip(window.iter()))
            {
                *w = s * win;
            }

            fwd.process(&windowed, &mut spectrum)
                .map_err(|e| Error::processing(e.to_string()))?;

            // 1. Magnitude and smoothed spectral envelope.
            let magnitudes: Vec<f32> = spectrum.iter().map(|c| c.norm()).collect();
            let orig_env = Self::smooth_envelope(&magnitudes);

            // 2. Warped envelope: sample the original envelope at bin / warp_factor.
            let mut warped_env = vec![0.0_f32; num_bins];
            for (bin, warped) in warped_env.iter_mut().enumerate() {
                let src = bin as f32 / warp_factor;
                *warped = Self::interp_at(&orig_env, src);
            }

            // 3. Apply the envelope ratio to each complex bin (preserving fine structure).
            for (bin, c) in spectrum.iter_mut().enumerate() {
                let ratio = if orig_env[bin] > 1e-8 {
                    // Gentle per-bin gain bound: formant warping should relocate
                    // energy, not inflate it. A 4× ceiling keeps a single bin from
                    // dominating while still allowing real envelope motion.
                    (warped_env[bin] / orig_env[bin]).clamp(0.0, 4.0)
                } else {
                    0.0
                };
                let re = c.re * ratio;
                let im = if bin == 0 || bin == num_bins - 1 {
                    0.0 // DC and Nyquist must stay real-valued
                } else {
                    c.im * ratio
                };
                *c = Complex::new(re, im);
            }

            // 4. Inverse FFT + synthesis-windowed overlap-add.
            inv.process(&spectrum, &mut time_out)
                .map_err(|e| Error::processing(e.to_string()))?;

            for (i, (&s, &win)) in time_out.iter().zip(window.iter()).enumerate() {
                let idx = pos + i;
                if idx < output.len() {
                    output[idx] += s * win;
                    ola_sum[idx] += win * win;
                }
            }

            pos += hop;
        }

        // Normalise by the squared-window OLA sum, silencing under-overlapped edges.
        let max_ola = ola_sum.iter().cloned().fold(0.0_f32, f32::max);
        let threshold = (max_ola * 0.1).max(1e-8_f32);
        for (s, &n) in output.iter_mut().zip(ola_sum.iter()) {
            if n > threshold {
                *s /= n;
            } else {
                *s = 0.0;
            }
        }

        output.truncate(out_len);

        // Bound the output peak to the input peak. Formant shifting relocates
        // spectral energy along the frequency axis; it must never *increase* the
        // overall amplitude. The per-bin ratio clamp plus overlap-add can locally
        // amplify the waveform above the input range, so apply a single uniform
        // scale to pull the peak back. Because this is a uniform gain it leaves
        // every HF/LF (or any inter-bin) ratio unchanged — only the absolute level
        // is constrained.
        let in_peak = audio.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        let out_peak = output.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        if out_peak > in_peak && in_peak > 0.0 {
            let scale = in_peak / out_peak;
            for s in output.iter_mut() {
                *s *= scale;
            }
        }

        Ok(output)
    }

    /// Smooth a magnitude spectrum into a spectral-envelope estimate using a
    /// symmetric moving average. The window width scales with the spectrum size so
    /// that harmonic ripple is averaged out while broad formant peaks survive.
    fn smooth_envelope(magnitudes: &[f32]) -> Vec<f32> {
        let n = magnitudes.len();
        if n == 0 {
            return Vec::new();
        }
        // Half-width of the averaging window (~3 % of the spectrum, at least 1 bin).
        let half = (n / 32).max(1);
        let mut env = vec![0.0_f32; n];
        for (bin, e) in env.iter_mut().enumerate() {
            let start = bin.saturating_sub(half);
            let end = (bin + half + 1).min(n);
            let slice = &magnitudes[start..end];
            *e = slice.iter().sum::<f32>() / slice.len() as f32;
        }
        env
    }

    /// Linearly interpolate `data` at fractional index `x`, clamping to the array
    /// bounds. Returns 0.0 for an empty slice.
    fn interp_at(data: &[f32], x: f32) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        if x <= 0.0 {
            return data[0];
        }
        let max_idx = data.len() - 1;
        if x >= max_idx as f32 {
            return data[max_idx];
        }
        let i = x.floor() as usize;
        let frac = x - i as f32;
        data[i] * (1.0 - frac) + data[i + 1] * frac
    }
}
