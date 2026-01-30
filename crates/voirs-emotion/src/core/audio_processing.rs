//! Audio processing helper functions for emotion effects

use crate::{
    types::{EmotionDimensions, EmotionParameters, EmotionState},
    Error, Result,
};
use std::collections::HashMap;

use super::cache::BufferPool;
use super::simd;

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

/// Apply simplified pitch shift effect (optimized to reduce allocations)
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

    // Simple pitch shift using interpolation with buffer pool
    if (effective_shift - 1.0).abs() > 0.01 {
        let mut shifted_audio = buffer_pool.get_buffer(audio.len());

        // Use SIMD-friendly loop if possible
        if use_simd && audio.len() >= 16 {
            simd::apply_pitch_shift_simd(audio, &mut shifted_audio, effective_shift);
        } else {
            #[allow(clippy::needless_range_loop)]
            for i in 0..audio.len() {
                let source_idx = (i as f32 / effective_shift) as usize;
                if source_idx < audio.len() {
                    shifted_audio[i] = audio[source_idx];
                }
            }
        }

        // Copy back with fade to avoid clicks (optimized)
        let audio_len = audio.len();
        let fade_samples = 64.min(audio_len / 4);

        for (i, sample) in audio.iter_mut().enumerate() {
            let fade_factor = if i < fade_samples {
                i as f32 / fade_samples as f32
            } else if i >= audio_len - fade_samples {
                (audio_len - i) as f32 / fade_samples as f32
            } else {
                1.0
            };

            *sample = *sample * (1.0 - fade_factor * strength)
                + shifted_audio[i] * fade_factor * strength;
        }

        // Return buffer to pool
        buffer_pool.return_buffer(shifted_audio);
    }

    Ok(())
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
