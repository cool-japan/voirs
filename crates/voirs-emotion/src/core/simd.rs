//! SIMD-optimized audio processing operations using the wide crate

use wide::f32x8;

/// SIMD-optimized energy scaling using wide crate
#[inline]
pub(super) fn apply_energy_scaling_simd(audio: &mut [f32], factor: f32) {
    let factor_vec = f32x8::splat(factor);
    let chunks = audio.len() / 8;
    let remainder = audio.len() % 8;

    // Process 8 samples at a time using SIMD
    for i in 0..chunks {
        let start_idx = i * 8;
        let chunk = &mut audio[start_idx..start_idx + 8];

        // Load 8 floats into SIMD register
        let samples = f32x8::from(&chunk[..8]);

        // Multiply by factor
        let result = samples * factor_vec;

        // Store back to memory
        let result_array: [f32; 8] = result.into();
        chunk.copy_from_slice(&result_array);
    }

    // Process remaining samples scalar
    for sample in &mut audio[chunks * 8..] {
        *sample *= factor;
    }
}

/// SIMD-accelerated element-wise window multiply: `dst[i] = src[i] * win[i]`.
///
/// This backs the per-frame analysis/synthesis windowing of the phase-vocoder
/// pitch shifter (`audio_processing::pitch_shift_phase_vocoder`), which is the
/// hottest per-sample loop in that transform. All three slices are processed up
/// to their shortest common length.
#[inline]
pub(super) fn apply_window_multiply_simd(dst: &mut [f32], src: &[f32], win: &[f32]) {
    let len = dst.len().min(src.len()).min(win.len());
    let chunks = len / 8;

    // Process 8 samples at a time using SIMD
    for i in 0..chunks {
        let start_idx = i * 8;
        let samples = f32x8::from(&src[start_idx..start_idx + 8]);
        let window = f32x8::from(&win[start_idx..start_idx + 8]);
        let product: [f32; 8] = (samples * window).into();
        dst[start_idx..start_idx + 8].copy_from_slice(&product);
    }

    // Process remaining samples scalar
    for i in chunks * 8..len {
        dst[i] = src[i] * win[i];
    }
}

/// SIMD-optimized tempo processing with linear interpolation
#[inline]
pub(super) fn apply_tempo_simd(input: &[f32], output: &mut [f32], tempo: f32) {
    let chunks = output.len() / 8;
    let remainder = output.len() % 8;

    // Process 8 samples at a time
    for i in 0..chunks {
        let start_idx = i * 8;
        let mut samples = [0.0f32; 8];

        // Calculate interpolated values for 8 output samples
        #[allow(clippy::needless_range_loop)]
        for j in 0..8 {
            let output_idx = start_idx + j;
            let source_pos = output_idx as f32 * tempo;
            let source_idx = source_pos as usize;
            let frac = source_pos - source_idx as f32;

            if source_idx < input.len() {
                samples[j] = if source_idx + 1 < input.len() {
                    input[source_idx] * (1.0 - frac) + input[source_idx + 1] * frac
                } else {
                    input[source_idx]
                };
            }
        }

        // Store to output
        output[start_idx..start_idx + 8].copy_from_slice(&samples);
    }

    // Process remaining samples scalar
    #[allow(clippy::needless_range_loop)]
    for i in chunks * 8..output.len() {
        let source_pos = i as f32 * tempo;
        let source_idx = source_pos as usize;
        let frac = source_pos - source_idx as f32;

        if source_idx < input.len() {
            output[i] = if source_idx + 1 < input.len() {
                input[source_idx] * (1.0 - frac) + input[source_idx + 1] * frac
            } else {
                input[source_idx]
            };
        }
    }
}

/// SIMD-optimized lowpass filter using wide crate
#[inline]
pub(super) fn apply_lowpass_simd(audio: &mut [f32], alpha: f32) {
    if audio.is_empty() {
        return;
    }

    let alpha_vec = f32x8::splat(alpha);
    let one_minus_alpha = f32x8::splat(1.0 - alpha);
    let chunks = audio.len() / 8;

    // Initialize with first sample for continuity
    let mut prev_vec = f32x8::splat(audio[0]);

    // Process first sample separately to establish initial state
    let mut prev_scalar = audio[0];

    // Process 8 samples at a time using SIMD
    for i in 0..chunks {
        let start_idx = i * 8;
        let chunk = &mut audio[start_idx..start_idx + 8];

        // Load 8 samples
        let samples = f32x8::from(&chunk[..8]);

        // Apply lowpass filter: output = alpha * input + (1-alpha) * prev
        let filtered = alpha_vec * samples + one_minus_alpha * prev_vec;

        // Store result
        let filtered_array: [f32; 8] = filtered.into();
        chunk.copy_from_slice(&filtered_array);

        // Update prev_vec for next iteration (use last value from filtered)
        prev_vec = f32x8::splat(chunk[7]);
    }

    // Process remaining samples scalar with continuity from SIMD processing
    let mut prev = if chunks > 0 {
        audio[chunks * 8 - 1]
    } else {
        prev_scalar
    };

    for sample in &mut audio[chunks * 8..] {
        prev = alpha * *sample + (1.0 - alpha) * prev;
        *sample = prev;
    }
}
