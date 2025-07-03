//! Audio processing pipeline
//!
//! This module provides audio processing operations including resampling,
//! normalization, and filtering.

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
            processed = crate::audio::AudioProcessor::trim_silence(&processed, self.config.silence_threshold)?;
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
                self.config.fade_out_duration
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
        // TODO: Implement high-quality sinc resampling
        // For now, use simple linear interpolation
        audio.resample(target_sample_rate)
    }
}

/// Audio normalization utilities
pub struct AudioNormalizer;

impl AudioNormalizer {
    /// Peak normalization
    pub fn normalize_peak(audio: &mut AudioData) -> Result<()> {
        audio.normalize()
    }
    
    /// RMS normalization
    pub fn normalize_rms(audio: &mut AudioData, target_rms: f32) -> Result<()> {
        let samples = audio.samples();
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        
        if rms > 0.0 {
            let scale = target_rms / rms;
            for sample in audio.samples_mut() {
                *sample *= scale;
            }
        }
        
        Ok(())
    }
    
    /// LUFS normalization (placeholder)
    pub fn normalize_lufs(audio: &mut AudioData, target_lufs: f32) -> Result<()> {
        // TODO: Implement LUFS normalization
        let _ = target_lufs;
        audio.normalize()
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
            } else {
                if in_silence {
                    silence_regions.push((silence_start, i));
                    in_silence = false;
                }
            }
        }
        
        // Add final silence region if audio ends with silence
        if in_silence {
            silence_regions.push((silence_start, samples.len()));
        }
        
        silence_regions
    }
    
    /// Check if audio contains significant silence
    pub fn has_excessive_silence(audio: &AudioData, threshold: f32, max_silence_ratio: f32) -> bool {
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
        },
        "rms" => {
            let target_rms = 10_f32.powf(target_level / 20.0); // Convert dB to linear
            AudioNormalizer::normalize_rms(&mut normalized, target_rms)?;
        },
        "lufs" => {
            AudioNormalizer::normalize_lufs(&mut normalized, target_level)?;
        },
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
pub fn mix_channels(audio: &AudioData, target_channels: usize, mix_method: &str) -> Result<AudioData> {
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
            for i in 0..frames {
                let sample = samples[i];
                output_samples.push(sample); // Left
                output_samples.push(sample); // Right
            }
        },
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
        },
        // Multi-channel to mono (average all channels)
        (n, 1) if n > 2 => {
            for i in 0..frames {
                let mut sum = 0.0;
                for ch in 0..n {
                    sum += samples[i * n + ch];
                }
                output_samples.push(sum / n as f32);
            }
        },
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
    
    Ok(AudioData::new(output_samples, sample_rate, target_channels as u32))
}
