//! Audio operations module.
//!
//! This module provides fundamental audio processing operations including:
//! - Sample rate conversion
//! - Channel mixing and splitting
//! - Amplitude scaling and normalization
//! - Audio filtering operations

use crate::{AudioBuffer, Result, VocoderError};

/// Sample rate conversion using linear interpolation
pub fn resample_audio(audio: &AudioBuffer, target_sample_rate: u32) -> Result<AudioBuffer> {
    if audio.sample_rate() == target_sample_rate {
        return Ok(audio.clone());
    }

    let ratio = target_sample_rate as f64 / audio.sample_rate() as f64;
    let samples = audio.samples();

    let resampled_len = (samples.len() as f64 * ratio) as usize;
    let mut resampled_samples = Vec::with_capacity(resampled_len);

    // Simple linear interpolation
    for i in 0..resampled_len {
        let src_index = i as f64 / ratio;
        let src_index_floor = src_index.floor() as usize;
        let src_index_ceil = (src_index_floor + 1).min(samples.len() - 1);
        let fraction = src_index - src_index_floor as f64;

        if src_index_floor < samples.len() {
            let sample_low = samples[src_index_floor];
            let sample_high = samples[src_index_ceil];
            let interpolated = sample_low + (sample_high - sample_low) * fraction as f32;
            resampled_samples.push(interpolated);
        }
    }

    Ok(AudioBuffer::new(resampled_samples, target_sample_rate, 1))
}

/// Convert mono to stereo
pub fn mono_to_stereo(audio: &AudioBuffer) -> Result<AudioBuffer> {
    let samples = audio.samples();
    let mut stereo_samples = Vec::with_capacity(samples.len() * 2);

    // Duplicate mono samples to both channels
    for &sample in samples {
        stereo_samples.push(sample);
        stereo_samples.push(sample);
    }

    Ok(AudioBuffer::new(stereo_samples, audio.sample_rate(), 2))
}

/// Convert stereo to mono by averaging channels
pub fn stereo_to_mono(audio: &AudioBuffer) -> Result<AudioBuffer> {
    let samples = audio.samples();
    if samples.len() % 2 != 0 {
        return Err(VocoderError::InputError(
            "Invalid stereo audio length".to_string(),
        ));
    }

    let mut mono_samples = Vec::with_capacity(samples.len() / 2);

    // Average left and right channels
    for chunk in samples.chunks(2) {
        let left = chunk[0];
        let right = chunk[1];
        mono_samples.push((left + right) / 2.0);
    }

    Ok(AudioBuffer::new(mono_samples, audio.sample_rate(), 1))
}

/// Mix multiple audio buffers together
pub fn mix_audio_buffers(buffers: &[AudioBuffer]) -> Result<AudioBuffer> {
    if buffers.is_empty() {
        return Err(VocoderError::InputError(
            "No audio buffers to mix".to_string(),
        ));
    }

    let first = &buffers[0];
    let sample_rate = first.sample_rate();
    let max_len = buffers.iter().map(|b| b.samples().len()).max().unwrap_or(0);

    let mut mixed_samples = vec![0.0; max_len];

    // Mix all buffers
    for buffer in buffers {
        if buffer.sample_rate() != sample_rate {
            return Err(VocoderError::InputError("Sample rate mismatch".to_string()));
        }

        let samples = buffer.samples();
        for (i, &sample) in samples.iter().enumerate() {
            mixed_samples[i] += sample / buffers.len() as f32;
        }
    }

    Ok(AudioBuffer::new(mixed_samples, sample_rate, 1))
}

/// Apply gain to audio buffer
pub fn apply_gain(audio: &mut AudioBuffer, gain_db: f32) {
    let gain_linear = 10.0_f32.powf(gain_db / 20.0);
    let samples = audio.samples_mut();

    for sample in samples.iter_mut() {
        *sample *= gain_linear;
    }
}

/// Apply fade in to audio buffer
pub fn apply_fade_in(audio: &mut AudioBuffer, fade_duration_sec: f32) {
    let fade_samples = (fade_duration_sec * audio.sample_rate() as f32) as usize;
    let samples = audio.samples_mut();
    let fade_samples = fade_samples.min(samples.len());

    for (i, sample) in samples.iter_mut().enumerate().take(fade_samples) {
        let fade_factor = i as f32 / fade_samples as f32;
        *sample *= fade_factor;
    }
}

/// Apply fade out to audio buffer
pub fn apply_fade_out(audio: &mut AudioBuffer, fade_duration_sec: f32) {
    let fade_samples = (fade_duration_sec * audio.sample_rate() as f32) as usize;
    let samples = audio.samples_mut();
    let fade_samples = fade_samples.min(samples.len());
    let start_index = samples.len().saturating_sub(fade_samples);

    for i in 0..fade_samples {
        let fade_factor = 1.0 - (i as f32 / fade_samples as f32);
        samples[start_index + i] *= fade_factor;
    }
}

/// Trim silence from beginning and end of audio
pub fn trim_silence(audio: &AudioBuffer, threshold: f32) -> Result<AudioBuffer> {
    let samples = audio.samples();

    // Find first non-silent sample
    let start = samples
        .iter()
        .position(|&x| x.abs() > threshold)
        .unwrap_or(0);

    // Find last non-silent sample
    let end = samples
        .iter()
        .rposition(|&x| x.abs() > threshold)
        .map(|pos| pos + 1)
        .unwrap_or(samples.len());

    if start >= end {
        return Ok(AudioBuffer::silence(0.0, audio.sample_rate(), 1));
    }

    let trimmed_samples = samples[start..end].to_vec();
    Ok(AudioBuffer::new(trimmed_samples, audio.sample_rate(), 1))
}

/// Concatenate multiple audio buffers
pub fn concatenate_audio(buffers: &[AudioBuffer]) -> Result<AudioBuffer> {
    if buffers.is_empty() {
        return Err(VocoderError::InputError(
            "No audio buffers to concatenate".to_string(),
        ));
    }

    let first = &buffers[0];
    let sample_rate = first.sample_rate();
    let mut concatenated_samples = Vec::new();

    for buffer in buffers {
        if buffer.sample_rate() != sample_rate {
            return Err(VocoderError::InputError("Sample rate mismatch".to_string()));
        }
        concatenated_samples.extend_from_slice(buffer.samples());
    }

    Ok(AudioBuffer::new(concatenated_samples, sample_rate, 1))
}

/// Split audio buffer into chunks
pub fn split_audio_into_chunks(
    audio: &AudioBuffer,
    chunk_duration_sec: f32,
    overlap_sec: f32,
) -> Vec<AudioBuffer> {
    let chunk_samples = (chunk_duration_sec * audio.sample_rate() as f32) as usize;
    let overlap_samples = (overlap_sec * audio.sample_rate() as f32) as usize;
    let step_samples = chunk_samples.saturating_sub(overlap_samples);

    let samples = audio.samples();
    let mut chunks = Vec::new();

    let mut start = 0;
    while start < samples.len() {
        let end = (start + chunk_samples).min(samples.len());
        if end > start {
            let chunk_samples = samples[start..end].to_vec();
            chunks.push(AudioBuffer::new(chunk_samples, audio.sample_rate(), 1));
        }
        start += step_samples;

        // Break if we can't make a meaningful chunk
        if start + chunk_samples / 2 >= samples.len() {
            break;
        }
    }

    chunks
}

/// Apply simple low-pass filter
pub fn apply_low_pass_filter(audio: &mut AudioBuffer, cutoff_freq: f32) {
    let sample_rate = audio.sample_rate() as f32;
    let samples = audio.samples_mut();

    // Simple first-order low-pass filter
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_freq);
    let dt = 1.0 / sample_rate;
    let alpha = dt / (rc + dt);

    if !samples.is_empty() {
        for i in 1..samples.len() {
            samples[i] = samples[i - 1] + alpha * (samples[i] - samples[i - 1]);
        }
    }
}

/// Apply simple high-pass filter
pub fn apply_high_pass_filter(audio: &mut AudioBuffer, cutoff_freq: f32) {
    let sample_rate = audio.sample_rate() as f32;
    let samples = audio.samples_mut();

    // Simple first-order high-pass filter
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_freq);
    let dt = 1.0 / sample_rate;
    let alpha = rc / (rc + dt);

    if !samples.is_empty() {
        let mut prev_input = samples[0];
        let mut prev_output = 0.0;

        for sample in samples.iter_mut() {
            let current_input = *sample;
            let current_output = alpha * (prev_output + current_input - prev_input);
            *sample = current_output;

            prev_input = current_input;
            prev_output = current_output;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mono_to_stereo() {
        let mono = AudioBuffer::new(vec![0.1, 0.2, 0.3], 22050, 1);
        let stereo = mono_to_stereo(&mono).unwrap();

        let expected = vec![0.1, 0.1, 0.2, 0.2, 0.3, 0.3];
        assert_eq!(stereo.samples(), expected);
    }

    #[test]
    fn test_stereo_to_mono() {
        let stereo = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 22050, 2);
        let mono = stereo_to_mono(&stereo).unwrap();

        let expected = [0.15, 0.35]; // (0.1+0.2)/2, (0.3+0.4)/2
        assert_eq!(mono.samples().len(), expected.len());
        for (actual, expected) in mono.samples().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_gain() {
        let mut audio = AudioBuffer::new(vec![0.1, 0.2, 0.3], 22050, 1);
        apply_gain(&mut audio, 6.0); // +6dB gain

        let gain_linear = 10.0_f32.powf(6.0 / 20.0); // ~1.995
        for (original, &gained) in [0.1, 0.2, 0.3].iter().zip(audio.samples()) {
            assert!((gained - original * gain_linear).abs() < 1e-6);
        }
    }

    #[test]
    fn test_trim_silence() {
        let samples = vec![0.0, 0.001, 0.5, 0.3, 0.001, 0.0];
        let audio = AudioBuffer::new(samples, 22050, 1);
        let trimmed = trim_silence(&audio, 0.01).unwrap();

        assert_eq!(trimmed.samples(), &[0.5, 0.3]);
    }

    #[test]
    fn test_concatenate_audio() {
        let audio1 = AudioBuffer::new(vec![0.1, 0.2], 22050, 1);
        let audio2 = AudioBuffer::new(vec![0.3, 0.4], 22050, 1);

        let concatenated = concatenate_audio(&[audio1, audio2]).unwrap();
        assert_eq!(concatenated.samples(), &[0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn test_split_audio_into_chunks() {
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let audio = AudioBuffer::new(samples, 6, 1); // 6 Hz sample rate for easy testing

        let chunks = split_audio_into_chunks(&audio, 0.5, 0.0); // 0.5 sec chunks, no overlap
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].samples(), &[0.1, 0.2, 0.3]);
        assert_eq!(chunks[1].samples(), &[0.4, 0.5, 0.6]);
    }

    #[test]
    fn test_mix_audio_buffers() {
        let audio1 = AudioBuffer::new(vec![0.2, 0.4], 22050, 1);
        let audio2 = AudioBuffer::new(vec![0.6, 0.8], 22050, 1);

        let mixed = mix_audio_buffers(&[audio1, audio2]).unwrap();
        let expected = vec![0.4, 0.6]; // (0.2+0.6)/2, (0.4+0.8)/2
        assert_eq!(mixed.samples(), expected);
    }
}
