//! Vocoder-free mel-spectrogram inversion via the Griffin-Lim algorithm.
//!
//! The streaming synthesiser's `mel_to_audio` previously faked a waveform with
//! an energy-modulated sine. This module replaces that with a real, fully
//! deterministic mel -> audio reconstruction that needs no neural vocoder:
//!
//! 1. **Mel filterbank inversion.** The same triangular mel filterbank `M`
//!    (`[n_mels x (n_fft/2 + 1)]`) used by the forward analysis path
//!    ([`crate::mel::create_mel_filterbank`]) is rebuilt. The log compression is
//!    undone (`exp`), then the linear *power* spectrogram is approximated with
//!    the transpose product `Mᵀ · mel`, normalised per frequency bin by the
//!    filterbank column sums to undo the energy bias of the (non-orthogonal)
//!    transpose. The power compression is finally undone (`x^{1/power}`,
//!    i.e. `sqrt` for a power spectrogram) to obtain a linear *magnitude*
//!    spectrogram.
//! 2. **Griffin-Lim phase recovery.** Starting from a deterministic phase
//!    (no RNG), the magnitude spectrogram is iteratively refined:
//!    ISTFT -> STFT -> keep our target magnitude / adopt the freshly estimated
//!    phase. After [`GRIFFIN_LIM_ITERS`] iterations a final ISTFT yields the
//!    time-domain waveform, normalised to a fixed peak.
//!
//! All FFTs use [`scirs2_fft::rfft`] / [`scirs2_fft::irfft`]; the analysis and
//! synthesis windows are the Hann window from [`crate::mel::WindowType`]. The
//! `rfft`/`irfft` pair is a consistent round trip (the forward transform is
//! unnormalised and the inverse carries the `1/N` factor), so the weighted
//! overlap-add ISTFT below divides by the sum of squared synthesis windows to
//! recover amplitude.

use crate::mel::{create_mel_filterbank, MelNormalization, WindowType};
use crate::{AcousticError, MelSpectrogram, Result};
use scirs2_core::numeric::Complex64;
use scirs2_fft::{irfft, rfft};

/// Number of Griffin-Lim refinement iterations (within the usual 32-60 range).
const GRIFFIN_LIM_ITERS: usize = 60;

/// Peak amplitude the reconstructed waveform is normalised to.
const TARGET_PEAK: f32 = 0.95;

/// Analysis / synthesis parameters used to invert a [`MelSpectrogram`].
#[derive(Debug, Clone)]
pub(crate) struct GriffinLimParams {
    /// Sample rate of the underlying audio (Hz).
    pub sample_rate: u32,
    /// FFT size.
    pub n_fft: usize,
    /// Hop length between frames in samples.
    pub hop_length: usize,
    /// Analysis / synthesis window length in samples.
    pub win_length: usize,
    /// Number of Griffin-Lim iterations.
    pub n_iters: usize,
    /// Power used by the forward spectrogram (`2.0` => power spectrogram).
    pub power: f32,
    /// Whether the mel values are natural-log compressed.
    pub log: bool,
    /// Offset added before the forward log to avoid `log(0)`.
    pub log_offset: f32,
    /// Minimum mel-filter frequency (Hz).
    pub fmin: f32,
    /// Maximum mel-filter frequency (Hz).
    pub fmax: f32,
    /// Mel-filter normalisation used by the forward path.
    pub norm: Option<MelNormalization>,
}

impl GriffinLimParams {
    /// Derive inversion parameters from a mel spectrogram.
    ///
    /// A [`MelSpectrogram`] only records `sample_rate`, `hop_length`, `n_mels`
    /// and `n_frames`, so the remaining analysis parameters are chosen to match
    /// the forward [`crate::mel::MelComputer`] defaults: a Hann window with
    /// `win_length == n_fft == 4 * hop_length` (75% overlap, which satisfies the
    /// constant-overlap-add condition for the Hann window), a power spectrogram
    /// (`power = 2`), natural-log compression, Slaney-normalised filters and a
    /// full `[0, Nyquist]` frequency range.
    pub fn for_mel(mel: &MelSpectrogram) -> Self {
        let sample_rate = if mel.sample_rate == 0 {
            22_050
        } else {
            mel.sample_rate
        };
        let hop_length = if mel.hop_length == 0 {
            256
        } else {
            mel.hop_length as usize
        };
        let n_fft = (4 * hop_length).max(2);

        Self {
            sample_rate,
            n_fft,
            hop_length,
            win_length: n_fft,
            n_iters: GRIFFIN_LIM_ITERS,
            power: 2.0,
            log: true,
            log_offset: 1e-10,
            fmin: 0.0,
            fmax: sample_rate as f32 / 2.0,
            norm: Some(MelNormalization::Slaney),
        }
    }
}

/// Reconstruct a waveform from a mel spectrogram without a neural vocoder.
///
/// Returns exactly `n_frames * hop_length` samples (the natural streaming
/// contract of one hop of audio per mel frame). The output is finite and
/// normalised to [`TARGET_PEAK`]; an empty mel yields an empty vector.
pub(crate) fn griffin_lim_reconstruct(
    mel: &MelSpectrogram,
    params: &GriffinLimParams,
) -> Result<Vec<f32>> {
    let n_frames = mel.n_frames;
    let n_mels = mel.n_mels;
    if n_frames == 0 || n_mels == 0 {
        return Ok(Vec::new());
    }

    let n_freqs = params.n_fft / 2 + 1;

    // (1) Rebuild the forward triangular mel filterbank: [n_mels][n_freqs].
    let filterbank = create_mel_filterbank(
        n_mels as u32,
        n_freqs as u32,
        params.sample_rate,
        params.fmin,
        params.fmax,
        params.norm,
    )?;

    // Column sums `Σ_m M[m][k]` used to debias the transpose pseudo-inverse.
    let mut col_sums = vec![0.0_f32; n_freqs];
    for filt in &filterbank {
        for (k, &weight) in filt.iter().enumerate() {
            col_sums[k] += weight;
        }
    }

    // (2) Recover a linear magnitude spectrogram, laid out as [n_freqs][n_frames].
    let inv_power = 1.0 / params.power;
    let mut magnitude = vec![vec![0.0_f32; n_frames]; n_freqs];
    for frame in 0..n_frames {
        // Undo the log compression once per (mel, frame): linear power-mel.
        let mel_lin: Vec<f32> = mel
            .data
            .iter()
            .map(|row| {
                let value = if frame < row.len() { row[frame] } else { 0.0 };
                if params.log {
                    (value.exp() - params.log_offset).max(0.0)
                } else {
                    value.max(0.0)
                }
            })
            .collect();

        // Transpose product `Mᵀ · mel`, debiased by column sums, then undo power.
        for (k, mag_row) in magnitude.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for (filt, &ml) in filterbank.iter().zip(mel_lin.iter()) {
                acc += filt[k] * ml;
            }
            let lin_power = if col_sums[k] > 1e-8 {
                acc / col_sums[k]
            } else {
                0.0
            };
            mag_row[frame] = lin_power.max(0.0).powf(inv_power);
        }
    }

    // (3) Griffin-Lim phase recovery.
    let window = WindowType::Hann.generate(params.win_length);

    // Initial complex spectrogram: target magnitude with a deterministic phase.
    let mut spectrum: Vec<Vec<Complex64>> = (0..n_frames)
        .map(|frame| {
            (0..n_freqs)
                .map(|k| {
                    let mag = f64::from(magnitude[k][frame]);
                    let phase = deterministic_phase(k, frame);
                    Complex64::new(mag * phase.cos(), mag * phase.sin())
                })
                .collect()
        })
        .collect();

    for _ in 0..params.n_iters {
        let audio = istft(&spectrum, &window, params.n_fft, params.hop_length)?;
        let estimate = stft(&audio, &window, params.n_fft, params.hop_length, n_frames)?;

        // Keep the target magnitude, adopt the freshly estimated phase.
        for frame in 0..n_frames {
            for k in 0..n_freqs {
                let est = estimate[frame][k];
                let est_mag = (est.re * est.re + est.im * est.im).sqrt();
                let mag = f64::from(magnitude[k][frame]);
                spectrum[frame][k] = if est_mag > 1e-12 {
                    Complex64::new(mag * est.re / est_mag, mag * est.im / est_mag)
                } else {
                    Complex64::new(mag, 0.0)
                };
            }
        }
    }

    let audio = istft(&spectrum, &window, params.n_fft, params.hop_length)?;

    // (4) Trim to the streaming contract and normalise to a sane peak.
    let target_len = n_frames * params.hop_length;
    let mut output = vec![0.0_f32; target_len];
    let mut peak = 0.0_f32;
    for (dst, &sample) in output.iter_mut().zip(audio.iter()) {
        let value = sample as f32;
        *dst = value;
        peak = peak.max(value.abs());
    }
    if peak > 1e-8 {
        let scale = TARGET_PEAK / peak;
        for sample in &mut output {
            *sample *= scale;
        }
    }

    Ok(output)
}

/// Deterministic, RNG-free phase in `[0, 2π)` seeded by the bin/frame indices.
///
/// A simple integer mix (no `rand`) decorrelates the initial phases across bins
/// and frames, which helps Griffin-Lim escape the degenerate zero-phase start
/// while remaining perfectly reproducible.
fn deterministic_phase(freq_bin: usize, frame: usize) -> f64 {
    let mixed = (freq_bin as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (frame as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    // Take the top 53 bits to form a fraction in [0, 1), then scale to [0, 2π).
    let frac = (mixed >> 11) as f64 / ((1_u64 << 53) as f64);
    frac * std::f64::consts::TAU
}

/// Short-time Fourier transform producing `n_frames` one-sided spectra.
///
/// Each frame windows `win_length` samples (zero-padded to `n_fft`) and
/// transforms with [`scirs2_fft::rfft`]; the result has `n_fft / 2 + 1` bins.
fn stft(
    audio: &[f64],
    window: &[f32],
    n_fft: usize,
    hop_length: usize,
    n_frames: usize,
) -> Result<Vec<Vec<Complex64>>> {
    let win_length = window.len();
    let mut frames = Vec::with_capacity(n_frames);
    let mut buffer = vec![0.0_f64; n_fft];

    for frame in 0..n_frames {
        let start = frame * hop_length;
        buffer.fill(0.0);
        for (i, &w) in window.iter().enumerate().take(win_length) {
            let idx = start + i;
            if idx < audio.len() {
                buffer[i] = audio[idx] * f64::from(w);
            }
        }

        let spec = rfft(&buffer, Some(n_fft)).map_err(|e| AcousticError::ProcessingError {
            message: format!("Griffin-Lim STFT rfft failed: {e}"),
        })?;
        frames.push(spec.iter().map(|c| Complex64::new(c.re, c.im)).collect());
    }

    Ok(frames)
}

/// Inverse short-time Fourier transform via weighted overlap-add.
///
/// Each one-sided spectrum is transformed with [`scirs2_fft::irfft`] (which
/// already applies the `1/N` normalisation), windowed with the synthesis Hann
/// window and overlap-added. The accumulated signal is divided by the running
/// sum of squared windows so that constant-overlap-add reconstruction recovers
/// amplitude regardless of the overlap factor.
fn istft(
    spectrum: &[Vec<Complex64>],
    window: &[f32],
    n_fft: usize,
    hop_length: usize,
) -> Result<Vec<f64>> {
    let n_frames = spectrum.len();
    if n_frames == 0 {
        return Ok(Vec::new());
    }

    let win_length = window.len();
    let output_len = (n_frames - 1) * hop_length + n_fft;
    let mut output = vec![0.0_f64; output_len];
    let mut window_sq_sum = vec![0.0_f64; output_len];

    for (frame, spec) in spectrum.iter().enumerate() {
        let time_frame =
            irfft(spec.as_slice(), Some(n_fft)).map_err(|e| AcousticError::ProcessingError {
                message: format!("Griffin-Lim ISTFT irfft failed: {e}"),
            })?;

        let start = frame * hop_length;
        for (i, &w) in window.iter().enumerate().take(win_length) {
            let weight = f64::from(w);
            output[start + i] += time_frame[i] * weight;
            window_sq_sum[start + i] += weight * weight;
        }
    }

    for (sample, &wsum) in output.iter_mut().zip(window_sq_sum.iter()) {
        if wsum > 1e-8 {
            *sample /= wsum;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mel::{MelComputer, MelParams};
    use std::f32::consts::PI;

    /// Build a mel spectrogram from a pure sine using analysis parameters that
    /// match [`GriffinLimParams::for_mel`], so the round trip is self-consistent.
    fn sine_mel(frequency: f32, sample_rate: u32, n_samples: usize) -> MelSpectrogram {
        let mut params = MelParams::new(sample_rate, 80);
        params.hop_length = 256;
        params.n_fft = 1024; // == 4 * hop_length, matching `for_mel`
        params.win_length = 1024; // == n_fft
        params.fmin = 0.0;
        params.fmax = None; // -> Nyquist
                            // power = 2.0, log = true, log_offset = 1e-10, norm = Slaney are defaults.

        let computer = MelComputer::new(params).expect("valid mel params");
        let audio: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * PI * frequency * i as f32 / sample_rate as f32).sin())
            .collect();
        computer.compute(&audio).expect("mel computation")
    }

    /// Dominant frequency (Hz) of a real signal via a single Hann-windowed rfft.
    fn dominant_frequency(signal: &[f32], sample_rate: u32) -> f32 {
        let n = signal.len();
        let window = WindowType::Hann.generate(n);
        let buffer: Vec<f64> = signal
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| f64::from(s) * f64::from(w))
            .collect();
        let spectrum = rfft(&buffer, Some(n)).expect("rfft");

        // Skip the DC bin and find the magnitude peak.
        let mut best_bin = 1_usize;
        let mut best_mag = 0.0_f64;
        for (k, c) in spectrum.iter().enumerate().skip(1) {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            if mag > best_mag {
                best_mag = mag;
                best_bin = k;
            }
        }
        best_bin as f32 * sample_rate as f32 / n as f32
    }

    #[test]
    fn test_griffin_lim_recovers_sine_frequency() {
        let sample_rate = 22_050;
        let frequency = 440.0;
        let mel = sine_mel(frequency, sample_rate, 11_025); // 0.5 s
        let params = GriffinLimParams::for_mel(&mel);
        let audio = griffin_lim_reconstruct(&mel, &params).expect("reconstruction");

        assert!(!audio.is_empty(), "reconstruction must produce audio");

        // Analyse the central portion to avoid overlap-add edge transients.
        let start = audio.len() / 4;
        let end = (audio.len() * 3 / 4).max(start + 2048).min(audio.len());
        let segment = &audio[start..end];
        let peak_hz = dominant_frequency(segment, sample_rate);

        assert!(
            (peak_hz - frequency).abs() < 80.0,
            "expected dominant frequency near {frequency} Hz, got {peak_hz} Hz"
        );
    }

    #[test]
    fn test_griffin_lim_output_length() {
        let mel = sine_mel(440.0, 22_050, 8_000);
        let params = GriffinLimParams::for_mel(&mel);
        let audio = griffin_lim_reconstruct(&mel, &params).expect("reconstruction");

        assert_eq!(
            audio.len(),
            mel.n_frames * params.hop_length,
            "output length must be n_frames * hop_length"
        );
    }

    #[test]
    fn test_griffin_lim_finite_and_nonsilent() {
        let mel = sine_mel(220.0, 22_050, 8_000);
        let params = GriffinLimParams::for_mel(&mel);
        let audio = griffin_lim_reconstruct(&mel, &params).expect("reconstruction");

        assert!(
            audio.iter().all(|s| s.is_finite()),
            "all reconstructed samples must be finite"
        );
        let peak = audio.iter().fold(0.0_f32, |a, &s| a.max(s.abs()));
        assert!(
            peak > 0.1,
            "non-zero mel must yield audible output, got peak {peak}"
        );
        assert!(
            peak <= TARGET_PEAK + 1e-4,
            "output must be peak-normalised, got {peak}"
        );
    }

    #[test]
    fn test_griffin_lim_empty_mel() {
        let empty = MelSpectrogram::new(Vec::new(), 22_050, 256);
        let params = GriffinLimParams::for_mel(&empty);
        let audio = griffin_lim_reconstruct(&empty, &params).expect("reconstruction");
        assert!(audio.is_empty(), "empty mel must yield empty audio");
    }
}
