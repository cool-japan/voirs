//! Audio processing helper functions for emotion effects

use crate::{
    types::{EmotionDimensions, EmotionParameters, EmotionState},
    Error, Result,
};
use scirs2_core::Complex;
use scirs2_fft::RealFftPlanner;
use std::collections::HashMap;
use std::f32::consts::PI;

use super::cache::BufferPool;
use super::simd;

/// Analysis/synthesis frame size for the phase-vocoder pitch shifter.
///
/// A power of two keeps the underlying real FFT efficient.
const PITCH_SHIFT_FRAME: usize = 1024;

/// Hop size for the phase-vocoder pitch shifter (75% overlap).
///
/// `FRAME / 4` satisfies the constant-overlap-add (COLA) constraint for the
/// periodic Hann window, enabling artifact-free overlap-add reconstruction.
const PITCH_SHIFT_HOP: usize = PITCH_SHIFT_FRAME / 4;

/// Apply voice quality effects like breathiness and roughness
pub(super) fn apply_voice_quality_effects(
    audio: &mut [f32],
    params: &EmotionParameters,
    strength: f32,
) -> Result<()> {
    // Apply breathiness effect (add controlled noise)
    if params.breathiness.abs() > 0.01 {
        let breathiness_level = params.breathiness * strength;
        for sample in audio.iter_mut() {
            let noise = (fastrand::f32() - 0.5) * 0.05 * breathiness_level;
            *sample = (*sample * (1.0 - breathiness_level * 0.3)) + noise;
        }
    }

    // Apply roughness effect (add harmonic distortion)
    if params.roughness.abs() > 0.01 {
        let roughness_level = params.roughness * strength;
        for sample in audio.iter_mut() {
            let distorted = (*sample).tanh() * roughness_level + *sample * (1.0 - roughness_level);
            *sample = distorted;
        }
    }

    Ok(())
}

/// Apply a duration-preserving pitch shift effect.
///
/// Unlike naive sample-rate conversion (reading the input at `i / shift`), which
/// changes *both* pitch and duration — the "chipmunk" effect — and aliases, this
/// uses a **phase vocoder**: the signal is analysed with overlapping
/// Hann-windowed STFT frames, each bin's instantaneous frequency is estimated
/// from the frame-to-frame phase advance and remapped by the pitch ratio, and the
/// result is reconstructed by overlap-add. Output length therefore always equals
/// input length. Buffers shorter than a single analysis frame fall back to a
/// length-preserving interpolating resampler (see
/// [`pitch_shift_resample_fallback`]).
///
/// `pitch_shift` is a pitch ratio (`> 1.0` raises pitch) and `strength` scales how
/// much of the requested shift is applied, via
/// `effective_shift = 1 + (pitch_shift - 1) * strength`. The shifted signal is
/// mixed back into `audio` with a short dry/wet crossfade at the edges, which both
/// avoids boundary clicks and masks the vocoder's overlap-add ramp at the
/// first/last frames.
///
/// To keep the real-time path allocation-light, the full-length shifted buffer is
/// borrowed from `buffer_pool`; only the inherent per-call FFT scratch is freshly
/// allocated.
pub(super) fn apply_pitch_shift_effect_optimized(
    audio: &mut [f32],
    pitch_shift: f32,
    strength: f32,
    buffer_pool: &BufferPool,
    use_simd: bool,
) -> Result<()> {
    if audio.len() < 2 {
        return Ok(());
    }

    let effective_shift = 1.0 + (pitch_shift - 1.0) * strength;

    // Negligible shift: leave the signal untouched (exact identity).
    if (effective_shift - 1.0).abs() <= 0.01 {
        return Ok(());
    }

    let audio_len = audio.len();
    let mut shifted_audio = buffer_pool.get_buffer(audio_len);

    // Render a pitch-shifted, duration-preserving copy into `shifted_audio`.
    if audio_len >= PITCH_SHIFT_FRAME {
        pitch_shift_phase_vocoder(audio, &mut shifted_audio, effective_shift, use_simd)?;
    } else {
        pitch_shift_resample_fallback(audio, &mut shifted_audio, effective_shift);
    }

    // Mix back, crossfading toward the dry signal at the edges to avoid clicks.
    let fade_samples = PITCH_SHIFT_HOP.min(audio_len / 4).max(1);
    for (i, sample) in audio.iter_mut().enumerate() {
        let wet = if i < fade_samples {
            i as f32 / fade_samples as f32
        } else if i >= audio_len - fade_samples {
            (audio_len - i) as f32 / fade_samples as f32
        } else {
            1.0
        };

        *sample = *sample * (1.0 - wet) + shifted_audio[i] * wet;
    }

    // Return buffer to pool
    buffer_pool.return_buffer(shifted_audio);

    Ok(())
}

/// Phase-vocoder pitch shift (duration-preserving).
///
/// Writes a pitch-scaled rendering of `input` into `output`; `output.len()` must
/// equal `input.len()` and `ratio > 1.0` raises the pitch.
///
/// Algorithm (75% overlap, periodic Hann analysis + synthesis windows):
///
/// 1. For each analysis frame: Hann-window it and take the forward real FFT.
/// 2. For each bin `k`, estimate the instantaneous frequency (in fractional bins)
///    from the phase deviation relative to the expected hop advance.
/// 3. Map source bin `k` to output bin `round(inst_bin * ratio)`, keeping the
///    maximum-magnitude source per output bin (prevents energy build-up when
///    several source bins collapse onto one output bin).
/// 4. Accumulate the pitch-scaled synthesis phase, build the output spectrum, take
///    the inverse real FFT, apply the synthesis window, and overlap-add.
/// 5. Normalise by the squared-window OLA sum, silencing the lightly-overlapped
///    edge samples.
fn pitch_shift_phase_vocoder(
    input: &[f32],
    output: &mut [f32],
    ratio: f32,
    use_simd: bool,
) -> Result<()> {
    const FRAME: usize = PITCH_SHIFT_FRAME;
    const HOP: usize = PITCH_SHIFT_HOP;
    let num_bins = FRAME / 2 + 1;
    let out_len = output.len();

    let mut planner = RealFftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(FRAME);
    let inv = planner.plan_fft_inverse(FRAME);

    let win = hann_window(FRAME);

    // Per-bin phase state across frames.
    let mut last_phase = vec![0.0_f32; num_bins];
    let mut synth_phase = vec![0.0_f32; num_bins];

    // Output / overlap-add accumulators with one frame of headroom.
    let mut acc = vec![0.0_f32; out_len + FRAME];
    let mut ola_sum = vec![0.0_f32; out_len + FRAME];

    // Zero-pad the tail so the final frame is fully covered.
    let mut padded = input.to_vec();
    padded.resize(input.len() + FRAME, 0.0);

    // Reusable per-frame working buffers.
    let mut frame = vec![0.0_f32; FRAME];
    let mut spectrum = vec![Complex::new(0.0_f32, 0.0_f32); num_bins];
    let mut out_spectrum = vec![Complex::new(0.0_f32, 0.0_f32); num_bins];
    let mut out_mag_best = vec![0.0_f32; num_bins];
    let mut time_out = vec![0.0_f32; FRAME];

    let mut pos = 0_usize;
    while pos + FRAME <= padded.len() {
        // ── Analysis: Hann-window the frame ──────────────────────────────
        let segment = &padded[pos..pos + FRAME];
        if use_simd && FRAME >= 16 {
            simd::apply_window_multiply_simd(&mut frame, segment, &win);
        } else {
            for (f, (&s, &w)) in frame.iter_mut().zip(segment.iter().zip(win.iter())) {
                *f = s * w;
            }
        }

        fwd.process(&frame, &mut spectrum)
            .map_err(|e| Error::Processing(e.to_string()))?;

        // ── Build the pitch-shifted output spectrum ──────────────────────
        out_spectrum.fill(Complex::new(0.0, 0.0));
        out_mag_best.fill(0.0);

        for k in 0..num_bins {
            let mag = spectrum[k].norm();
            let phase = spectrum[k].arg();

            // Instantaneous frequency, expressed in fractional bins.
            let expected = 2.0 * PI * k as f32 * HOP as f32 / FRAME as f32;
            let deviation = wrap_phase(phase - last_phase[k] - expected);
            let inst_bin = k as f32 + deviation * FRAME as f32 / (2.0 * PI * HOP as f32);
            last_phase[k] = phase;

            // Target output bin.
            let k_out = (inst_bin * ratio).round() as isize;
            if k_out < 0 || k_out as usize >= num_bins {
                continue;
            }
            let k_out = k_out as usize;

            // Advance the synthesis phase by the pitch-scaled instantaneous freq.
            synth_phase[k_out] += inst_bin * ratio * 2.0 * PI * HOP as f32 / FRAME as f32;

            // Keep the strongest source bin that maps to this output bin.
            if mag > out_mag_best[k_out] {
                out_mag_best[k_out] = mag;
                let re = mag * synth_phase[k_out].cos();
                let im = if k_out == 0 || k_out == num_bins - 1 {
                    0.0 // DC and Nyquist must be purely real.
                } else {
                    mag * synth_phase[k_out].sin()
                };
                out_spectrum[k_out] = Complex::new(re, im);
            }
        }

        // ── Synthesis: inverse FFT + synthesis window, overlap-add ───────
        inv.process(&out_spectrum, &mut time_out)
            .map_err(|e| Error::Processing(e.to_string()))?;

        for (i, (&s, &w)) in time_out.iter().zip(win.iter()).enumerate() {
            let idx = pos + i;
            if idx < acc.len() {
                acc[idx] += s * w;
                ola_sum[idx] += w * w;
            }
        }

        pos += HOP;
    }

    // ── Normalise by the OLA window-energy sum ────────────────────────────
    // Silence samples whose overlap is below 10% of the peak (the first/last
    // partially-covered frames) to avoid amplifying lightly-overlapped edges.
    let max_ola = ola_sum.iter().copied().fold(0.0_f32, f32::max);
    let threshold = (max_ola * 0.1).max(1e-8);
    for (dst, (&a, &n)) in output.iter_mut().zip(acc.iter().zip(ola_sum.iter())) {
        *dst = if n > threshold { a / n } else { 0.0 };
    }

    Ok(())
}

/// Length-preserving interpolating resampler for buffers too short for the phase
/// vocoder (`< PITCH_SHIFT_FRAME` samples).
///
/// Reads the input at `ratio` samples per output sample using linear
/// interpolation, which raises pitch for `ratio > 1.0`. The output is the same
/// length as the input, with the tail beyond the input zero-filled. For sub-frame
/// buffers (under ~23 ms at 44.1 kHz) the phase vocoder cannot resolve frequency
/// reliably, so this bounded-bandwidth fallback is used; it preserves length but
/// not content duration, which is acceptable at these very short lengths.
fn pitch_shift_resample_fallback(input: &[f32], output: &mut [f32], ratio: f32) {
    let in_len = input.len();
    for (i, dst) in output.iter_mut().enumerate() {
        let src = i as f32 * ratio;
        let idx = src as usize;
        *dst = if idx + 1 < in_len {
            let frac = src - idx as f32;
            input[idx] * (1.0 - frac) + input[idx + 1] * frac
        } else if idx < in_len {
            input[idx]
        } else {
            0.0
        };
    }
}

/// Periodic (DFT-even) Hann window of length `n`.
///
/// Satisfies the COLA constraint at a hop of `n / 4`, enabling artifact-free
/// overlap-add reconstruction in the phase vocoder.
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

/// Wrap a phase value into the interval `(-π, π]`.
#[inline]
fn wrap_phase(p: f32) -> f32 {
    // Branch-free symmetric modulo: p - 2π·round(p / 2π).
    p - (2.0 * PI) * (p / (2.0 * PI)).round()
}

/// Apply tempo effect by resampling (optimized with buffer pool)
pub(super) fn apply_tempo_effect_optimized(
    mut audio: Vec<f32>,
    tempo_scale: f32,
    strength: f32,
    buffer_pool: &BufferPool,
    use_simd: bool,
) -> Result<Vec<f32>> {
    let effective_tempo = 1.0 + (tempo_scale - 1.0) * strength;

    if (effective_tempo - 1.0).abs() < 0.01 {
        return Ok(audio);
    }

    let new_length = (audio.len() as f32 / effective_tempo) as usize;
    let mut resampled = buffer_pool.get_buffer(new_length);

    // SIMD-optimized linear interpolation resampling when possible
    if use_simd && new_length >= 16 {
        simd::apply_tempo_simd(&audio, &mut resampled, effective_tempo);
    } else {
        // Standard linear interpolation resampling
        #[allow(clippy::needless_range_loop)]
        for i in 0..new_length {
            let source_pos = i as f32 * effective_tempo;
            let source_idx = source_pos as usize;
            let frac = source_pos - source_idx as f32;

            if source_idx < audio.len() {
                resampled[i] = if source_idx + 1 < audio.len() {
                    audio[source_idx] * (1.0 - frac) + audio[source_idx + 1] * frac
                } else {
                    audio[source_idx]
                };
            }
        }
    }

    // Return the input buffer to pool and return the resampled one
    buffer_pool.return_buffer(audio);
    Ok(resampled)
}

/// Apply custom emotion-specific effects
pub(super) fn apply_custom_emotion_effects(
    audio: &mut [f32],
    params: &EmotionParameters,
    strength: f32,
    use_simd: bool,
) -> Result<()> {
    // Apply effects based on dominant emotion
    if let Some((emotion, intensity)) = params.emotion_vector.dominant_emotion() {
        let effect_strength = intensity.value() * strength;

        match emotion {
            crate::types::Emotion::Angry => {
                // Add slight distortion for anger
                for sample in audio.iter_mut() {
                    *sample = (*sample * 0.9).tanh() * effect_strength
                        + *sample * (1.0 - effect_strength);
                }
            }
            crate::types::Emotion::Sad => {
                // Reduce brightness for sadness
                apply_lowpass_filter(
                    audio,
                    0.7 * effect_strength + 1.0 * (1.0 - effect_strength),
                    use_simd,
                )?;
            }
            crate::types::Emotion::Happy | crate::types::Emotion::Excited => {
                // Enhance brightness for happiness/excitement
                apply_highpass_emphasis(audio, effect_strength)?;
            }
            crate::types::Emotion::Calm => {
                // Smooth the signal for calmness
                apply_smoothing_filter(audio, effect_strength)?;
            }
            _ => {
                // No specific effect for other emotions
            }
        }
    }

    // Apply custom parameter effects
    for (param_name, value) in &params.custom_params {
        match param_name.as_str() {
            "reverb" => {
                apply_simple_reverb(audio, *value)?;
            }
            "chorus" => {
                apply_simple_chorus(audio, *value)?;
            }
            _ => {
                // Unknown parameter, skip
            }
        }
    }

    Ok(())
}

/// Apply simple lowpass filter effect (SIMD optimized when possible)
pub(super) fn apply_lowpass_filter(audio: &mut [f32], cutoff: f32, use_simd: bool) -> Result<()> {
    let alpha = cutoff.clamp(0.1, 1.0);

    if use_simd && audio.len() >= 16 {
        simd::apply_lowpass_simd(audio, alpha);
    } else {
        let mut prev = 0.0;
        for sample in audio.iter_mut() {
            prev = alpha * *sample + (1.0 - alpha) * prev;
            *sample = prev;
        }
    }

    Ok(())
}

/// Apply highpass emphasis
pub(super) fn apply_highpass_emphasis(audio: &mut [f32], strength: f32) -> Result<()> {
    if audio.len() < 2 {
        return Ok(());
    }

    let mut prev = audio[0];
    #[allow(clippy::needless_range_loop)]
    for i in 1..audio.len() {
        let high_freq = audio[i] - prev;
        audio[i] += high_freq * strength * 0.3;
        prev = audio[i];
    }

    Ok(())
}

/// Apply smoothing filter
pub(super) fn apply_smoothing_filter(audio: &mut [f32], strength: f32) -> Result<()> {
    if audio.len() < 3 {
        return Ok(());
    }

    let mut smoothed = audio.to_vec();
    for i in 1..audio.len() - 1 {
        let average = (audio[i - 1] + audio[i] + audio[i + 1]) / 3.0;
        smoothed[i] = audio[i] * (1.0 - strength) + average * strength;
    }

    audio.copy_from_slice(&smoothed);
    Ok(())
}

/// Apply simple reverb effect
pub(super) fn apply_simple_reverb(audio: &mut [f32], strength: f32) -> Result<()> {
    if strength.abs() < 0.01 || audio.len() < 1000 {
        return Ok(());
    }

    let delay_samples = (audio.len() / 10).min(1000);
    let decay = 0.3 * strength;

    for i in delay_samples..audio.len() {
        audio[i] += audio[i - delay_samples] * decay;
    }

    Ok(())
}

/// Apply simple chorus effect
pub(super) fn apply_simple_chorus(audio: &mut [f32], strength: f32) -> Result<()> {
    if strength.abs() < 0.01 || audio.len() < 100 {
        return Ok(());
    }

    let delay_samples = 20;
    let mix = strength * 0.3;

    for i in delay_samples..audio.len() {
        audio[i] = audio[i] * (1.0 - mix) + audio[i - delay_samples] * mix;
    }

    Ok(())
}

/// Optimized interpolation computation with reduced allocations
pub(super) fn compute_optimized_interpolation(state: &EmotionState) -> EmotionParameters {
    if let Some(target) = &state.target {
        if state.transition_progress < 1.0 {
            let progress = state.transition_progress;

            // Pre-allocate with reasonable capacity
            let mut interpolated_emotions = HashMap::with_capacity(
                state
                    .current
                    .emotion_vector
                    .emotions
                    .len()
                    .max(target.emotion_vector.emotions.len()),
            );

            // Optimize emotion interpolation by avoiding HashSet allocation
            // First pass: interpolate emotions from current
            for (emotion, current_intensity) in &state.current.emotion_vector.emotions {
                let target_intensity = target
                    .emotion_vector
                    .emotions
                    .get(emotion)
                    .map(|i| i.value())
                    .unwrap_or(0.0);

                let interpolated_intensity = current_intensity.value()
                    + (target_intensity - current_intensity.value()) * progress;

                if interpolated_intensity > 0.01 {
                    interpolated_emotions.insert(
                        emotion.clone(),
                        crate::types::EmotionIntensity::new(interpolated_intensity),
                    );
                }
            }

            // Second pass: add target emotions not in current
            for (emotion, target_intensity) in &target.emotion_vector.emotions {
                if !interpolated_emotions.contains_key(emotion) {
                    let interpolated_intensity = target_intensity.value() * progress;
                    if interpolated_intensity > 0.01 {
                        interpolated_emotions.insert(
                            emotion.clone(),
                            crate::types::EmotionIntensity::new(interpolated_intensity),
                        );
                    }
                }
            }

            // Create interpolated emotion vector efficiently
            let mut emotion_vector = crate::types::EmotionVector::new();
            emotion_vector.emotions = interpolated_emotions;

            // Interpolate dimensions directly
            let current_dims = &state.current.emotion_vector.dimensions;
            let target_dims = &target.emotion_vector.dimensions;

            emotion_vector.dimensions = EmotionDimensions::new(
                current_dims.valence + (target_dims.valence - current_dims.valence) * progress,
                current_dims.arousal + (target_dims.arousal - current_dims.arousal) * progress,
                current_dims.dominance
                    + (target_dims.dominance - current_dims.dominance) * progress,
            );

            // Pre-allocate custom params map
            let mut interpolated_custom = HashMap::with_capacity(
                state
                    .current
                    .custom_params
                    .len()
                    .max(target.custom_params.len()),
            );

            // Efficient custom parameter interpolation
            for (param, current_value) in &state.current.custom_params {
                let target_value = target.custom_params.get(param).cloned().unwrap_or(0.0);
                let interpolated_value = current_value + (target_value - current_value) * progress;
                interpolated_custom.insert(param.clone(), interpolated_value);
            }

            for (param, target_value) in &target.custom_params {
                if !interpolated_custom.contains_key(param) {
                    let interpolated_value = target_value * progress;
                    interpolated_custom.insert(param.clone(), interpolated_value);
                }
            }

            // Build final parameters
            crate::types::EmotionParameters {
                emotion_vector,
                duration_ms: target.duration_ms.or(state.current.duration_ms),
                fade_in_ms: target.fade_in_ms.or(state.current.fade_in_ms),
                fade_out_ms: target.fade_out_ms.or(state.current.fade_out_ms),
                pitch_shift: state.current.pitch_shift
                    + (target.pitch_shift - state.current.pitch_shift) * progress,
                tempo_scale: state.current.tempo_scale
                    + (target.tempo_scale - state.current.tempo_scale) * progress,
                energy_scale: state.current.energy_scale
                    + (target.energy_scale - state.current.energy_scale) * progress,
                breathiness: state.current.breathiness
                    + (target.breathiness - state.current.breathiness) * progress,
                roughness: state.current.roughness
                    + (target.roughness - state.current.roughness) * progress,
                custom_params: interpolated_custom,
            }
        } else {
            state.current.clone()
        }
    } else {
        state.current.clone()
    }
}

/// Validate emotion parameters
pub(super) fn validate_emotion_parameters(
    params: &EmotionParameters,
    max_pitch: f32,
    max_tempo: f32,
    max_energy: f32,
) -> Result<()> {
    if params.pitch_shift < 0.1 || params.pitch_shift > max_pitch {
        return Err(Error::Validation(format!(
            "Pitch shift {} out of range [0.1, {}]",
            params.pitch_shift, max_pitch
        )));
    }

    if params.tempo_scale < 0.1 || params.tempo_scale > max_tempo {
        return Err(Error::Validation(format!(
            "Tempo scale {} out of range [0.1, {}]",
            params.tempo_scale, max_tempo
        )));
    }

    if params.energy_scale < 0.1 || params.energy_scale > max_energy {
        return Err(Error::Validation(format!(
            "Energy scale {} out of range [0.1, {}]",
            params.energy_scale, max_energy
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a pure sine wave with exactly `cycles` periods over `n` samples,
    /// so its energy lands on FFT bin `cycles` for an `n`-point transform.
    fn sine(n: usize, cycles: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * cycles * i as f32 / n as f32).sin())
            .collect()
    }

    /// Detect the dominant (highest-magnitude) FFT bin of `signal`, ignoring DC.
    fn dominant_bin(signal: &[f32]) -> usize {
        let n = signal.len();
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n);
        let mut spectrum = vec![Complex::new(0.0_f32, 0.0); n / 2 + 1];
        fwd.process(signal, &mut spectrum)
            .expect("forward FFT for fundamental detection should succeed");

        let mut best_bin = 1_usize;
        let mut best_mag = 0.0_f32;
        for (k, c) in spectrum.iter().enumerate().skip(1) {
            let mag = c.norm();
            if mag > best_mag {
                best_mag = mag;
                best_bin = k;
            }
        }
        best_bin
    }

    /// The shift must preserve length exactly and never produce NaN/inf.
    #[test]
    fn test_pitch_shift_preserves_length_and_finite() {
        let pool = BufferPool::new(4);
        let mut audio = sine(4096, 120.0);
        let original_len = audio.len();

        apply_pitch_shift_effect_optimized(&mut audio, 1.5, 1.0, &pool, true)
            .expect("pitch shift should succeed");

        assert_eq!(
            audio.len(),
            original_len,
            "phase-vocoder output length must equal input length"
        );
        assert!(
            audio.iter().all(|s| s.is_finite()),
            "all output samples must be finite"
        );
    }

    /// Shifting up by an octave (2x) must move the detected fundamental ~1 octave
    /// while keeping output length equal to input length.
    #[test]
    fn test_pitch_shift_octave_up_doubles_fundamental() {
        let pool = BufferPool::new(4);
        let n = 8192;
        let mut audio = sine(n, 100.0); // fundamental at FFT bin 100
        let in_bin = dominant_bin(&audio);

        apply_pitch_shift_effect_optimized(&mut audio, 2.0, 1.0, &pool, true)
            .expect("pitch shift should succeed");

        assert_eq!(audio.len(), n, "output length must equal input length");

        let out_bin = dominant_bin(&audio);
        let measured_ratio = out_bin as f32 / in_bin as f32;
        assert!(
            (1.7..=2.3).contains(&measured_ratio),
            "shifting up an octave (2x) should ~double the fundamental: \
             in_bin={in_bin}, out_bin={out_bin}, ratio={measured_ratio:.3}"
        );
    }

    /// A ratio of 1.0 leaves the signal untouched (exact identity).
    #[test]
    fn test_pitch_shift_identity_is_passthrough() {
        let pool = BufferPool::new(4);
        let original = sine(2048, 64.0);
        let mut audio = original.clone();

        apply_pitch_shift_effect_optimized(&mut audio, 1.0, 1.0, &pool, false)
            .expect("identity pitch shift should succeed");

        assert_eq!(audio.len(), original.len());
        for (a, b) in original.iter().zip(audio.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "ratio 1.0 must be identity, diff={:.3e}",
                (a - b).abs()
            );
        }
    }
}
