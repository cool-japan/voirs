//! Audio processing and I/O operations
//!
//! This module provides audio loading, processing, and manipulation
//! capabilities for speech synthesis datasets.

pub mod advanced_analysis;
pub mod data;
pub mod io;
pub mod multimodal;
pub mod processing;
pub mod psychoacoustic;
pub mod realtime;
pub mod simd;

use crate::{AudioData, AudioFormat, DatasetError, Result};
use std::path::Path;

/// Audio I/O operations
pub struct AudioIO;

impl AudioIO {
    /// Load audio file from path
    pub fn load<P: AsRef<Path>>(path: P) -> Result<AudioData> {
        let path = path.as_ref();
        let format = Self::detect_format(path)?;

        match format {
            AudioFormat::Wav => Self::load_wav(path),
            AudioFormat::Flac => Self::load_flac(path),
            AudioFormat::Mp3 => Self::load_mp3(path),
            AudioFormat::Ogg => Self::load_ogg(path),
            AudioFormat::Opus => Self::load_opus(path),
        }
    }

    /// Save audio file to path
    pub fn save<P: AsRef<Path>>(audio: &AudioData, path: P) -> Result<()> {
        let path = path.as_ref();
        let format = Self::detect_format(path)?;

        match format {
            AudioFormat::Wav => Self::save_wav(audio, path),
            AudioFormat::Flac => Self::save_flac(audio, path),
            AudioFormat::Mp3 => Self::save_mp3(audio, path),
            AudioFormat::Ogg => Self::save_ogg(audio, path),
            AudioFormat::Opus => Self::save_opus(audio, path),
        }
    }

    /// Detect audio format from file extension
    pub fn detect_format<P: AsRef<Path>>(path: P) -> Result<AudioFormat> {
        let path = path.as_ref();

        match path.extension().and_then(|ext| ext.to_str()) {
            Some("wav") => Ok(AudioFormat::Wav),
            Some("flac") => Ok(AudioFormat::Flac),
            Some("mp3") => Ok(AudioFormat::Mp3),
            Some("ogg") => Ok(AudioFormat::Ogg),
            _ => Err(DatasetError::FormatError(format!(
                "Unsupported audio format: {:?}",
                path.extension()
            ))),
        }
    }

    /// Load WAV file
    fn load_wav<P: AsRef<Path>>(path: P) -> Result<AudioData> {
        let mut reader = hound::WavReader::open(path)
            .map_err(|e| DatasetError::AudioError(format!("Failed to open WAV file: {e}")))?;

        let spec = reader.spec();
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / i16::MAX as f32))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| DatasetError::AudioError(format!("Failed to read WAV samples: {e}")))?;

        Ok(AudioData::new(
            samples,
            spec.sample_rate,
            spec.channels as u32,
        ))
    }

    /// Save WAV file
    fn save_wav<P: AsRef<Path>>(audio: &AudioData, path: P) -> Result<()> {
        let spec = hound::WavSpec {
            channels: audio.channels() as u16,
            sample_rate: audio.sample_rate(),
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path, spec)
            .map_err(|e| DatasetError::AudioError(format!("Failed to create WAV writer: {e}")))?;

        for &sample in audio.samples() {
            let sample_i16 = (sample * i16::MAX as f32) as i16;
            writer.write_sample(sample_i16).map_err(|e| {
                DatasetError::AudioError(format!("Failed to write WAV sample: {e}"))
            })?;
        }

        writer
            .finalize()
            .map_err(|e| DatasetError::AudioError(format!("Failed to finalize WAV file: {e}")))?;

        Ok(())
    }

    /// Load FLAC file
    fn load_flac<P: AsRef<Path>>(path: P) -> Result<AudioData> {
        let file = std::fs::File::open(path)
            .map_err(|e| DatasetError::AudioError(format!("Failed to open FLAC file: {e}")))?;

        let mut reader = claxon::FlacReader::new(file)
            .map_err(|e| DatasetError::AudioError(format!("Failed to create FLAC reader: {e}")))?;

        let streaminfo = reader.streaminfo();
        let sample_rate = streaminfo.sample_rate;
        let channels = streaminfo.channels;
        let bits_per_sample = streaminfo.bits_per_sample;

        let mut samples = Vec::new();

        // Read samples and convert to f32
        for sample in reader.samples() {
            let sample = sample.map_err(|e| {
                DatasetError::AudioError(format!("Failed to read FLAC sample: {e}"))
            })?;

            // Convert to f32 based on bit depth
            let normalized = match bits_per_sample {
                16 => sample as f32 / i16::MAX as f32,
                24 => sample as f32 / ((1 << 23) - 1) as f32,
                32 => sample as f32 / i32::MAX as f32,
                _ => {
                    return Err(DatasetError::FormatError(format!(
                        "Unsupported FLAC bit depth: {bits_per_sample}"
                    )))
                }
            };

            samples.push(normalized);
        }

        Ok(AudioData::new(samples, sample_rate, channels))
    }

    /// Save FLAC file
    fn save_flac<P: AsRef<Path>>(audio: &AudioData, path: P) -> Result<()> {
        // Delegate to the specialized io module implementation
        crate::audio::io::save_flac(audio, path)
    }

    /// Load MP3 file
    fn load_mp3<P: AsRef<Path>>(path: P) -> Result<AudioData> {
        // Delegate to the specialized io module implementation
        crate::audio::io::load_mp3(path)
    }

    /// Save MP3 file
    fn save_mp3<P: AsRef<Path>>(audio: &AudioData, path: P) -> Result<()> {
        // Delegate to the specialized io module implementation
        crate::audio::io::save_mp3(audio, path)
    }

    /// Load OGG file
    fn load_ogg<P: AsRef<Path>>(path: P) -> Result<AudioData> {
        let file = std::fs::File::open(path)
            .map_err(|e| DatasetError::AudioError(format!("Failed to open OGG file: {e}")))?;

        let mut reader = lewton::inside_ogg::OggStreamReader::new(file)
            .map_err(|e| DatasetError::AudioError(format!("Failed to create OGG reader: {e}")))?;

        let sample_rate = reader.ident_hdr.audio_sample_rate;
        let channels = reader.ident_hdr.audio_channels as u32;

        let mut samples = Vec::new();

        // Read OGG packets and decode
        while let Some(pck_samples) = reader
            .read_dec_packet_itl()
            .map_err(|e| DatasetError::AudioError(format!("Failed to read OGG packet: {e}")))?
        {
            // Convert samples to f32
            for sample in pck_samples {
                samples.push(sample as f32 / i16::MAX as f32);
            }
        }

        if samples.is_empty() {
            return Err(DatasetError::FormatError(
                "No valid OGG data found".to_string(),
            ));
        }

        Ok(AudioData::new(samples, sample_rate, channels))
    }

    fn load_opus<P: AsRef<Path>>(path: P) -> Result<AudioData> {
        let file = std::fs::File::open(path)
            .map_err(|e| DatasetError::AudioError(format!("Failed to open OPUS file: {e}")))?;

        let mut reader = lewton::inside_ogg::OggStreamReader::new(file)
            .map_err(|e| DatasetError::AudioError(format!("Failed to create OPUS reader: {e}")))?;

        let sample_rate = reader.ident_hdr.audio_sample_rate;
        let channels = reader.ident_hdr.audio_channels as u32;

        let mut samples = Vec::new();

        // Read OPUS packets and decode (using OGG container)
        while let Some(pck_samples) = reader
            .read_dec_packet_itl()
            .map_err(|e| DatasetError::AudioError(format!("Failed to read OPUS packet: {e}")))?
        {
            // Convert samples to f32
            for sample in pck_samples {
                samples.push(sample as f32 / i16::MAX as f32);
            }
        }

        if samples.is_empty() {
            return Err(DatasetError::FormatError(
                "No valid OPUS data found".to_string(),
            ));
        }

        Ok(AudioData::new(samples, sample_rate, channels))
    }

    /// Save OGG file
    fn save_ogg<P: AsRef<Path>>(audio: &AudioData, path: P) -> Result<()> {
        // Delegate to the specialized io module implementation
        crate::audio::io::save_ogg(audio, path)
    }

    /// Save OPUS file
    fn save_opus<P: AsRef<Path>>(_audio: &AudioData, _path: P) -> Result<()> {
        // OPUS encoding is complex and requires libopus bindings
        // For now, we recommend using FLAC or WAV for lossless encoding
        // or converting to OPUS using external tools like FFmpeg
        Err(DatasetError::FormatError(
            "OPUS encoding not yet implemented. Consider using FLAC or WAV formats for lossless audio, or use external tools like FFmpeg for OPUS conversion.".to_string(),
        ))
    }
}

/// Audio processing utilities
pub struct AudioProcessor;

impl AudioProcessor {
    /// Resample audio to target sample rate
    pub fn resample(audio: &AudioData, target_sample_rate: u32) -> Result<AudioData> {
        audio.resample(target_sample_rate)
    }

    /// Normalize audio amplitude
    pub fn normalize(audio: &mut AudioData) -> Result<()> {
        audio.normalize()
    }

    /// Trim silence from beginning and end
    pub fn trim_silence(audio: &AudioData, threshold: f32) -> Result<AudioData> {
        let samples = audio.samples();
        let mut start = 0;
        let mut end = samples.len();

        // Find start of audio content
        for (i, &sample) in samples.iter().enumerate() {
            if sample.abs() > threshold {
                start = i;
                break;
            }
        }

        // Find end of audio content
        for (i, &sample) in samples.iter().enumerate().rev() {
            if sample.abs() > threshold {
                end = i + 1;
                break;
            }
        }

        if start >= end {
            // All samples are below threshold, return silence
            return Ok(AudioData::silence(
                0.1,
                audio.sample_rate(),
                audio.channels(),
            ));
        }

        let trimmed_samples = samples[start..end].to_vec();
        Ok(AudioData::new(
            trimmed_samples,
            audio.sample_rate(),
            audio.channels(),
        ))
    }

    /// Convert stereo to mono by averaging channels
    pub fn to_mono(audio: &AudioData) -> Result<AudioData> {
        if audio.channels() == 1 {
            return Ok(audio.clone());
        }

        let samples = audio.samples();
        let channels = audio.channels() as usize;
        let mono_samples: Vec<f32> = samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect();

        Ok(AudioData::new(mono_samples, audio.sample_rate(), 1))
    }

    /// Apply fade in/out to audio
    pub fn apply_fade(
        audio: &mut AudioData,
        fade_in_duration: f32,
        fade_out_duration: f32,
    ) -> Result<()> {
        let sample_rate = audio.sample_rate() as f32;
        let samples = audio.samples_mut();
        let fade_in_samples = (fade_in_duration * sample_rate) as usize;
        let fade_out_samples = (fade_out_duration * sample_rate) as usize;

        // Apply fade in
        for i in 0..fade_in_samples.min(samples.len()) {
            let fade_factor = i as f32 / fade_in_samples as f32;
            samples[i] *= fade_factor;
        }

        // Apply fade out
        let start_fade_out = samples.len().saturating_sub(fade_out_samples);
        for i in start_fade_out..samples.len() {
            let fade_factor = (samples.len() - i) as f32 / fade_out_samples as f32;
            samples[i] *= fade_factor;
        }

        Ok(())
    }
}
