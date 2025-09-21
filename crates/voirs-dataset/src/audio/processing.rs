//! Audio processing pipeline
//!
//! This module provides audio processing operations including resampling,
//! normalization, and filtering with SIMD optimizations where available.

use super::simd::SimdAudioProcessor;
use crate::{AudioData, Result};

/// Audio processing pipeline configuration
#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    /// Target sample rate
    pub target_sample_rate: Option<u32>,
    /// Normalize amplitude
    pub normalize: bool,
    /// Trim silence
    pub trim_silence: bool,
    /// Silence threshold for trimming
    pub silence_threshold: f32,
    /// Convert to mono
    pub to_mono: bool,
    /// Apply fade in/out
    pub apply_fade: bool,
    /// Fade in duration in seconds
    pub fade_in_duration: f32,
    /// Fade out duration in seconds
    pub fade_out_duration: f32,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: None,
            normalize: true,
            trim_silence: true,
            silence_threshold: 0.01,
            to_mono: false,
            apply_fade: false,
            fade_in_duration: 0.1,
            fade_out_duration: 0.1,
        }
    }
}

/// Audio processing pipeline
pub struct AudioProcessingPipeline {
    config: ProcessingConfig,
}

impl AudioProcessingPipeline {
    /// Create new processing pipeline
    pub fn new(config: ProcessingConfig) -> Self {
        Self { config }
    }

    /// Process audio data according to configuration
    pub fn process(&self, audio: &AudioData) -> Result<AudioData> {
        let mut processed = audio.clone();

        // Resample if needed
        if let Some(target_sample_rate) = self.config.target_sample_rate {
            if processed.sample_rate() != target_sample_rate {
                processed = processed.resample(target_sample_rate)?;
            }
        }

        // Convert to mono if needed
        if self.config.to_mono && processed.channels() > 1 {
            processed = crate::audio::AudioProcessor::to_mono(&processed)?;
        }

        // Trim silence if needed
        if self.config.trim_silence {
            processed = crate::audio::AudioProcessor::trim_silence(
                &processed,
                self.config.silence_threshold,
            )?;
        }

        // Normalize if needed
        if self.config.normalize {
            processed.normalize()?;
        }

        // Apply fade if needed
        if self.config.apply_fade {
            crate::audio::AudioProcessor::apply_fade(
                &mut processed,
                self.config.fade_in_duration,
                self.config.fade_out_duration,
            )?;
        }

        Ok(processed)
    }
}

/// High-quality resampling using sinc interpolation
pub struct SincResampler;

impl SincResampler {
    /// Resample audio using sinc interpolation
    pub fn resample(audio: &AudioData, target_sample_rate: u32) -> Result<AudioData> {
        if audio.sample_rate() == target_sample_rate {
            return Ok(audio.clone());
        }

        let ratio = target_sample_rate as f64 / audio.sample_rate() as f64;
        let channels = audio.channels() as usize;
        let input_samples = audio.samples();
        let input_frames = input_samples.len() / channels;
        let output_frames = (input_frames as f64 * ratio) as usize;
        let mut output_samples = vec![0.0; output_frames * channels];

        // Sinc function parameters
        let cutoff = 0.95; // Anti-aliasing cutoff frequency
        let window_size = 64; // Kaiser window size
        let beta = 8.0; // Kaiser window parameter

        // Process each channel separately
        for ch in 0..channels {
            for out_frame in 0..output_frames {
                let in_pos = out_frame as f64 / ratio;
                let in_frame = in_pos as usize;
                let frac = in_pos - in_frame as f64;

                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                // Apply windowed sinc filter
                for i in 0..window_size {
                    let tap_pos = in_frame + i;
                    if tap_pos >= window_size / 2 && tap_pos < input_frames + window_size / 2 {
                        let sample_pos = tap_pos.saturating_sub(window_size / 2);

                        if sample_pos < input_frames {
                            let t = (i as f64 - window_size as f64 / 2.0 + frac) * cutoff;

                            // Sinc function with Kaiser window
                            let sinc = if t.abs() < 1e-10 {
                                cutoff
                            } else {
                                let pi_t = std::f64::consts::PI * t;
                                cutoff * pi_t.sin() / pi_t
                            };

                            // Kaiser window
                            let window_arg =
                                1.0 - (2.0 * i as f64 / window_size as f64 - 1.0).powi(2);
                            let window = if window_arg >= 0.0 {
                                Self::bessel_i0(beta * window_arg.sqrt()) / Self::bessel_i0(beta)
                            } else {
                                0.0
                            };

                            let weight = sinc * window;
                            sum += input_samples[sample_pos * channels + ch] as f64 * weight;
                            weight_sum += weight;
                        }
                    }
                }

                // Normalize and store result
                output_samples[out_frame * channels + ch] = if weight_sum != 0.0 {
                    (sum / weight_sum) as f32
                } else {
                    0.0
                };
            }
        }

        Ok(AudioData::new(
            output_samples,
            target_sample_rate,
            audio.channels(),
        ))
    }

    /// Modified Bessel function of the first kind (I0)
    fn bessel_i0(x: f64) -> f64 {
        let mut sum = 1.0;
        let mut term = 1.0;
        let x_half_squared = (x / 2.0).powi(2);

        for k in 1..=50 {
            term *= x_half_squared / (k as f64).powi(2);
            sum += term;
            if term < 1e-12 {
                break;
            }
        }

        sum
    }
}

/// Audio normalization utilities
pub struct AudioNormalizer;

impl AudioNormalizer {
    /// Peak normalization
    pub fn normalize_peak(audio: &mut AudioData) -> Result<()> {
        audio.normalize()
    }

    /// RMS normalization with SIMD optimization
    pub fn normalize_rms(audio: &mut AudioData, target_rms: f32) -> Result<()> {
        let samples = audio.samples();
        let rms = SimdAudioProcessor::calculate_rms(samples);

        if rms > 0.0 {
            let scale = target_rms / rms;
            SimdAudioProcessor::apply_gain(audio.samples_mut(), scale);
        }

        Ok(())
    }

    /// LUFS normalization according to ITU-R BS.1770-4 standard
    pub fn normalize_lufs(audio: &mut AudioData, target_lufs: f32) -> Result<()> {
        // Calculate current LUFS level
        let current_lufs = Self::calculate_lufs(audio)?;

        // Calculate gain needed to reach target LUFS
        let gain_db = target_lufs - current_lufs;
        let gain_linear = 10.0_f32.powf(gain_db / 20.0);

        // Apply gain to all samples
        SimdAudioProcessor::apply_gain(audio.samples_mut(), gain_linear);

        Ok(())
    }

    /// Calculate LUFS loudness according to ITU-R BS.1770-4
    pub fn calculate_lufs(audio: &AudioData) -> Result<f32> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();
        let channels = audio.channels() as usize;

        if samples.is_empty() {
            return Ok(-70.0); // Silence
        }

        // Pre-filter: High-pass filter at ~38Hz (Stage 1)
        let filtered_samples = Self::apply_highpass_filter(samples, sample_rate, channels)?;

        // Pre-filter: High-frequency shelving filter (Stage 2)
        let filtered_samples = Self::apply_shelf_filter(&filtered_samples, sample_rate, channels)?;

        // Calculate mean square for each channel
        let block_size = (sample_rate as f32 * 0.4) as usize; // 400ms blocks
        let overlap = block_size / 2;
        let mut loudness_blocks = Vec::new();

        let frames = filtered_samples.len() / channels;
        let mut start = 0;

        while start + block_size <= frames {
            let mut channel_powers = Vec::new();

            // Calculate mean square for each channel in this block
            for ch in 0..channels {
                let mut sum_squares = 0.0;
                for frame in start..start + block_size {
                    let sample = filtered_samples[frame * channels + ch];
                    sum_squares += sample * sample;
                }
                let mean_square = sum_squares / block_size as f32;
                channel_powers.push(mean_square);
            }

            // Apply channel weighting and sum
            let weighted_power = match channels {
                1 => channel_powers[0],                     // Mono
                2 => channel_powers[0] + channel_powers[1], // Stereo (L+R)
                _ => {
                    // For multi-channel, use simplified weighting
                    channel_powers.iter().sum::<f32>()
                }
            };

            if weighted_power > 0.0 {
                loudness_blocks.push(weighted_power);
            }

            start += overlap;
        }

        if loudness_blocks.is_empty() {
            return Ok(-70.0); // Silence
        }

        // Calculate relative threshold (-10 LU below mean)
        let mean_power: f32 = loudness_blocks.iter().sum::<f32>() / loudness_blocks.len() as f32;
        let relative_threshold = mean_power * 0.1; // -10 LU = factor of 0.1

        // Calculate gated loudness (only blocks above relative threshold)
        let gated_blocks: Vec<f32> = loudness_blocks
            .into_iter()
            .filter(|&power| power >= relative_threshold)
            .collect();

        if gated_blocks.is_empty() {
            return Ok(-70.0); // Below threshold
        }

        let gated_mean_power: f32 = gated_blocks.iter().sum::<f32>() / gated_blocks.len() as f32;

        // Convert to LUFS: -0.691 + 10 * log10(sum of mean squares)
        let lufs = -0.691 + 10.0 * gated_mean_power.log10();

        Ok(lufs)
    }

    /// Apply high-pass filter (Stage 1 pre-filter)
    fn apply_highpass_filter(
        samples: &[f32],
        sample_rate: u32,
        channels: usize,
    ) -> Result<Vec<f32>> {
        let mut filtered = samples.to_vec();
        let frames = samples.len() / channels;

        // Simple IIR high-pass filter at ~38Hz
        let cutoff = 38.0;
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
        let dt = 1.0 / sample_rate as f32;
        let alpha = rc / (rc + dt);

        // Apply filter to each channel
        for ch in 0..channels {
            let mut prev_input = 0.0;
            let mut prev_output = 0.0;

            for frame in 0..frames {
                let idx = frame * channels + ch;
                let input = samples[idx];
                let output = alpha * (prev_output + input - prev_input);
                filtered[idx] = output;

                prev_input = input;
                prev_output = output;
            }
        }

        Ok(filtered)
    }

    /// Apply shelf filter (Stage 2 pre-filter)
    fn apply_shelf_filter(samples: &[f32], sample_rate: u32, channels: usize) -> Result<Vec<f32>> {
        let mut filtered = samples.to_vec();
        let frames = samples.len() / channels;

        // High-frequency shelving filter
        let fc = 1681.0; // Center frequency
        let gain_db = 3.99; // Gain in dB
        let q = 1.0 / std::f32::consts::SQRT_2;

        let gain = 10.0_f32.powf(gain_db / 20.0);
        let omega = 2.0 * std::f32::consts::PI * fc / sample_rate as f32;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let a0 = (gain + 1.0) + (gain - 1.0) * cos_omega + 2.0 * gain.sqrt() * alpha;
        let a1 = -2.0 * ((gain - 1.0) + (gain + 1.0) * cos_omega);
        let a2 = (gain + 1.0) + (gain - 1.0) * cos_omega - 2.0 * gain.sqrt() * alpha;
        let b0 = gain * ((gain + 1.0) - (gain - 1.0) * cos_omega + 2.0 * gain.sqrt() * alpha);
        let b1 = 2.0 * gain * ((gain - 1.0) - (gain + 1.0) * cos_omega);
        let b2 = gain * ((gain + 1.0) - (gain - 1.0) * cos_omega - 2.0 * gain.sqrt() * alpha);

        // Normalize coefficients
        let b0 = b0 / a0;
        let b1 = b1 / a0;
        let b2 = b2 / a0;
        let a1 = a1 / a0;
        let a2 = a2 / a0;

        // Apply biquad filter to each channel
        for ch in 0..channels {
            let mut x1 = 0.0;
            let mut x2 = 0.0;
            let mut y1 = 0.0;
            let mut y2 = 0.0;

            for frame in 0..frames {
                let idx = frame * channels + ch;
                let x0 = samples[idx];

                let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                filtered[idx] = y0;

                x2 = x1;
                x1 = x0;
                y2 = y1;
                y1 = y0;
            }
        }

        Ok(filtered)
    }
}

/// Silence detection utilities
pub struct SilenceDetector;

impl SilenceDetector {
    /// Detect silence regions in audio
    pub fn detect_silence(audio: &AudioData, threshold: f32) -> Vec<(usize, usize)> {
        let samples = audio.samples();
        let mut silence_regions = Vec::new();
        let mut in_silence = false;
        let mut silence_start = 0;

        for (i, &sample) in samples.iter().enumerate() {
            if sample.abs() <= threshold {
                if !in_silence {
                    silence_start = i;
                    in_silence = true;
                }
            } else if in_silence {
                silence_regions.push((silence_start, i));
                in_silence = false;
            }
        }

        // Add final silence region if audio ends with silence
        if in_silence {
            silence_regions.push((silence_start, samples.len()));
        }

        silence_regions
    }

    /// Check if audio contains significant silence with SIMD optimization
    pub fn has_excessive_silence(
        audio: &AudioData,
        threshold: f32,
        max_silence_ratio: f32,
    ) -> bool {
        let samples = audio.samples();
        let silence_samples =
            samples.len() - SimdAudioProcessor::count_above_threshold(samples, threshold);
        let silence_ratio = silence_samples as f32 / samples.len() as f32;

        silence_ratio > max_silence_ratio
    }

    /// Check if audio contains significant silence (legacy implementation)
    pub fn has_excessive_silence_legacy(
        audio: &AudioData,
        threshold: f32,
        max_silence_ratio: f32,
    ) -> bool {
        let silence_regions = Self::detect_silence(audio, threshold);
        let total_silence: usize = silence_regions.iter().map(|(start, end)| end - start).sum();
        let silence_ratio = total_silence as f32 / audio.samples().len() as f32;

        silence_ratio > max_silence_ratio
    }
}

/// Normalize audio using specified method
pub fn normalize_audio(audio: &AudioData, method: &str, target_level: f32) -> Result<AudioData> {
    let mut normalized = audio.clone();

    match method.to_lowercase().as_str() {
        "peak" => {
            normalized.normalize()?;
        }
        "rms" => {
            let target_rms = 10_f32.powf(target_level / 20.0); // Convert dB to linear
            AudioNormalizer::normalize_rms(&mut normalized, target_rms)?;
        }
        "lufs" => {
            AudioNormalizer::normalize_lufs(&mut normalized, target_level)?;
        }
        _ => {
            // Default to peak normalization
            normalized.normalize()?;
        }
    }

    Ok(normalized)
}

/// Resample audio to target sample rate
pub fn resample_audio(audio: &AudioData, target_sample_rate: u32) -> Result<AudioData> {
    if audio.sample_rate() == target_sample_rate {
        Ok(audio.clone())
    } else {
        SincResampler::resample(audio, target_sample_rate)
    }
}

/// Detect silence and return start/end positions
pub fn detect_silence(audio: &AudioData, threshold_db: f32) -> Result<(usize, usize)> {
    let threshold = 10_f32.powf(threshold_db / 20.0); // Convert dB to linear
    let silence_regions = SilenceDetector::detect_silence(audio, threshold);

    // Return the start of first silence and end of last silence
    if silence_regions.is_empty() {
        Ok((0, audio.samples().len()))
    } else {
        let start = silence_regions.first().unwrap().0;
        let end = silence_regions.last().unwrap().1;
        Ok((start, end))
    }
}

/// Mix audio channels
pub fn mix_channels(
    audio: &AudioData,
    target_channels: usize,
    mix_method: &str,
) -> Result<AudioData> {
    let current_channels = audio.channels() as usize;

    if current_channels == target_channels {
        return Ok(audio.clone());
    }

    let samples = audio.samples();
    let sample_rate = audio.sample_rate();
    let frames = samples.len() / current_channels;

    let mut output_samples = Vec::with_capacity(frames * target_channels);

    match (current_channels, target_channels) {
        // Mono to stereo
        (1, 2) => {
            for sample in samples.iter().take(frames) {
                output_samples.push(*sample); // Left
                output_samples.push(*sample); // Right
            }
        }
        // Stereo to mono
        (2, 1) => {
            for i in 0..frames {
                let left = samples[i * 2];
                let right = samples[i * 2 + 1];
                let mixed = match mix_method.to_lowercase().as_str() {
                    "average" => (left + right) / 2.0,
                    "left" => left,
                    "right" => right,
                    _ => (left + right) / 2.0, // Default to average
                };
                output_samples.push(mixed);
            }
        }
        // Multi-channel to mono (average all channels)
        (n, 1) if n > 2 => {
            for i in 0..frames {
                let mut sum = 0.0;
                for ch in 0..n {
                    sum += samples[i * n + ch];
                }
                output_samples.push(sum / n as f32);
            }
        }
        // Other combinations - just truncate or duplicate as needed
        _ => {
            for i in 0..frames {
                for ch in 0..target_channels {
                    let source_ch = ch.min(current_channels - 1);
                    output_samples.push(samples[i * current_channels + source_ch]);
                }
            }
        }
    }

    Ok(AudioData::new(
        output_samples,
        sample_rate,
        target_channels as u32,
    ))
}
