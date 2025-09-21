//! Audio format support for various file types

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

/// Supported audio format types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioFormatType {
    /// WAV format (uncompressed PCM)
    Wav,
    /// FLAC format (lossless compression)
    Flac,
    /// MP3 format (lossy compression)
    Mp3,
    /// AAC format (lossy compression)
    Aac,
    /// Opus format (lossy compression)
    Opus,
    /// OGG Vorbis format (lossy compression)
    Ogg,
    /// AIFF format (Apple Audio Interchange File Format)
    Aiff,
    /// Raw PCM format
    Raw,
    /// 24-bit WAV
    Wav24,
    /// 32-bit float WAV
    Wav32f,
}

impl AudioFormatType {
    /// Get file extensions for this format
    pub fn extensions(&self) -> &[&str] {
        match self {
            AudioFormatType::Wav => &["wav"],
            AudioFormatType::Flac => &["flac"],
            AudioFormatType::Mp3 => &["mp3"],
            AudioFormatType::Aac => &["aac", "m4a"],
            AudioFormatType::Opus => &["opus"],
            AudioFormatType::Ogg => &["ogg"],
            AudioFormatType::Aiff => &["aiff", "aif"],
            AudioFormatType::Raw => &["raw", "pcm"],
            AudioFormatType::Wav24 => &["wav"],
            AudioFormatType::Wav32f => &["wav"],
        }
    }

    /// Get MIME type for this format
    pub fn mime_type(&self) -> &str {
        match self {
            AudioFormatType::Wav | AudioFormatType::Wav24 | AudioFormatType::Wav32f => "audio/wav",
            AudioFormatType::Flac => "audio/flac",
            AudioFormatType::Mp3 => "audio/mpeg",
            AudioFormatType::Aac => "audio/aac",
            AudioFormatType::Opus => "audio/opus",
            AudioFormatType::Ogg => "audio/ogg",
            AudioFormatType::Aiff => "audio/aiff",
            AudioFormatType::Raw => "application/octet-stream",
        }
    }

    /// Check if format is lossy
    pub fn is_lossy(&self) -> bool {
        matches!(
            self,
            AudioFormatType::Mp3
                | AudioFormatType::Aac
                | AudioFormatType::Opus
                | AudioFormatType::Ogg
        )
    }

    /// Check if format is lossless
    pub fn is_lossless(&self) -> bool {
        !self.is_lossy()
    }

    /// Get typical bit rates for lossy formats (in kbps)
    pub fn typical_bitrates(&self) -> Option<&[u32]> {
        match self {
            AudioFormatType::Mp3 => Some(&[128, 192, 256, 320]),
            AudioFormatType::Aac => Some(&[128, 192, 256]),
            AudioFormatType::Opus => Some(&[64, 96, 128, 192]),
            AudioFormatType::Ogg => Some(&[112, 160, 192, 256]),
            _ => None,
        }
    }
}

/// Audio format specification with detailed parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioFormat {
    /// Format type
    pub format_type: AudioFormatType,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Bits per sample (for uncompressed formats)
    pub bits_per_sample: Option<u16>,
    /// Bit rate (for compressed formats)
    pub bit_rate: Option<u32>,
    /// Duration in seconds
    pub duration: Option<f64>,
    /// Additional format-specific metadata
    pub metadata: HashMap<String, String>,
}

impl AudioFormat {
    /// Create new audio format
    pub fn new(format_type: AudioFormatType, sample_rate: u32, channels: u16) -> Self {
        Self {
            format_type,
            sample_rate,
            channels,
            bits_per_sample: Some(16),
            bit_rate: None,
            duration: None,
            metadata: HashMap::new(),
        }
    }

    /// Set bits per sample
    pub fn with_bits_per_sample(mut self, bits: u16) -> Self {
        self.bits_per_sample = Some(bits);
        self
    }

    /// Set bit rate for compressed formats
    pub fn with_bit_rate(mut self, rate: u32) -> Self {
        self.bit_rate = Some(rate);
        self
    }

    /// Set duration
    pub fn with_duration(mut self, duration: f64) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Calculate estimated file size for audio
    pub fn estimated_file_size(&self, duration_seconds: f64) -> u64 {
        match self.format_type {
            // Uncompressed formats
            AudioFormatType::Wav | AudioFormatType::Aiff | AudioFormatType::Raw => {
                let bytes_per_second = self.sample_rate as u64
                    * self.channels as u64
                    * (self.bits_per_sample.unwrap_or(16) as u64 / 8);
                (bytes_per_second as f64 * duration_seconds) as u64
            }
            AudioFormatType::Wav24 => {
                let bytes_per_second = self.sample_rate as u64 * self.channels as u64 * 3; // 24 bits = 3 bytes
                (bytes_per_second as f64 * duration_seconds) as u64
            }
            AudioFormatType::Wav32f => {
                let bytes_per_second = self.sample_rate as u64 * self.channels as u64 * 4; // 32-bit float = 4 bytes
                (bytes_per_second as f64 * duration_seconds) as u64
            }
            // Compressed formats
            _ => {
                let bit_rate = self.bit_rate.unwrap_or(128); // Default 128 kbps
                ((bit_rate as f64 * 1000.0) / 8.0 * duration_seconds) as u64
            }
        }
    }

    /// Check if format supports the given sample rate
    pub fn supports_sample_rate(&self, sample_rate: u32) -> bool {
        match self.format_type {
            AudioFormatType::Opus => matches!(sample_rate, 8000 | 12000 | 16000 | 24000 | 48000),
            AudioFormatType::Mp3 => sample_rate <= 48000,
            _ => sample_rate <= 192000, // Most formats support up to 192kHz
        }
    }
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self::new(AudioFormatType::Wav, 44100, 2) // Standard CD quality
    }
}

/// Audio data with format information
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Raw PCM audio samples (normalized -1.0 to 1.0)
    pub samples: Vec<f32>,
    /// Audio format information
    pub format: AudioFormat,
}

impl AudioData {
    /// Create new audio data
    pub fn new(samples: Vec<f32>, format: AudioFormat) -> Self {
        Self { samples, format }
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f64 {
        self.samples.len() as f64 / (self.format.sample_rate as f64 * self.format.channels as f64)
    }

    /// Get number of frames (samples per channel)
    pub fn frames(&self) -> usize {
        self.samples.len() / self.format.channels as usize
    }

    /// Split into channels
    pub fn split_channels(&self) -> Vec<Vec<f32>> {
        let channels = self.format.channels as usize;
        let frames = self.frames();
        let mut channel_data = vec![Vec::with_capacity(frames); channels];

        for (i, &sample) in self.samples.iter().enumerate() {
            let channel = i % channels;
            channel_data[channel].push(sample);
        }

        channel_data
    }

    /// Combine channels into interleaved samples
    pub fn from_channels(channels: Vec<Vec<f32>>, format: AudioFormat) -> Result<Self> {
        if channels.is_empty() {
            return Err(Error::audio("No channels provided".to_string()));
        }

        let frames = channels[0].len();
        let num_channels = channels.len();

        // Verify all channels have same length
        for (i, channel) in channels.iter().enumerate() {
            if channel.len() != frames {
                return Err(Error::audio(format!(
                    "Channel {} has different length: {} vs {}",
                    i,
                    channel.len(),
                    frames
                )));
            }
        }

        let mut samples = Vec::with_capacity(frames * num_channels);
        for frame in 0..frames {
            for channel in &channels {
                samples.push(channel[frame]);
            }
        }

        Ok(AudioData::new(samples, format))
    }

    /// Convert to mono by averaging channels
    pub fn to_mono(&self) -> AudioData {
        if self.format.channels == 1 {
            return self.clone();
        }

        let channels = self.format.channels as usize;
        let frames = self.frames();
        let mut mono_samples = Vec::with_capacity(frames);

        for frame in 0..frames {
            let mut sum = 0.0;
            for channel in 0..channels {
                sum += self.samples[frame * channels + channel];
            }
            mono_samples.push(sum / channels as f32);
        }

        let mut mono_format = self.format.clone();
        mono_format.channels = 1;

        AudioData::new(mono_samples, mono_format)
    }

    /// Resample to target sample rate (basic linear interpolation)
    pub fn resample(&self, target_sample_rate: u32) -> AudioData {
        if self.format.sample_rate == target_sample_rate {
            return self.clone();
        }

        let ratio = target_sample_rate as f64 / self.format.sample_rate as f64;
        let channels = self.format.channels as usize;
        let input_frames = self.frames();
        let output_frames = (input_frames as f64 * ratio).round() as usize;
        let mut output_samples = vec![0.0f32; output_frames * channels];

        for output_frame in 0..output_frames {
            let input_position = output_frame as f64 / ratio;
            let input_frame = input_position.floor() as usize;
            let fraction = input_position - input_frame as f64;

            if input_frame + 1 < input_frames {
                // Linear interpolation between two frames
                for channel in 0..channels {
                    let sample1 = self.samples[input_frame * channels + channel];
                    let sample2 = self.samples[(input_frame + 1) * channels + channel];
                    let interpolated = sample1 + (sample2 - sample1) * fraction as f32;
                    output_samples[output_frame * channels + channel] = interpolated;
                }
            } else if input_frame < input_frames {
                // Use last frame
                for channel in 0..channels {
                    output_samples[output_frame * channels + channel] =
                        self.samples[input_frame * channels + channel];
                }
            }
        }

        let mut new_format = self.format.clone();
        new_format.sample_rate = target_sample_rate;

        AudioData::new(output_samples, new_format)
    }
}

/// Format detector for identifying audio file types
pub struct FormatDetector;

impl FormatDetector {
    /// Detect format from file extension
    pub fn detect_from_extension<P: AsRef<Path>>(path: P) -> Option<AudioFormatType> {
        let extension = path.as_ref().extension()?.to_str()?.to_lowercase();

        match extension.as_str() {
            "wav" => Some(AudioFormatType::Wav),
            "flac" => Some(AudioFormatType::Flac),
            "mp3" => Some(AudioFormatType::Mp3),
            "aac" | "m4a" => Some(AudioFormatType::Aac),
            "opus" => Some(AudioFormatType::Opus),
            "ogg" => Some(AudioFormatType::Ogg),
            "aiff" | "aif" => Some(AudioFormatType::Aiff),
            "raw" | "pcm" => Some(AudioFormatType::Raw),
            _ => None,
        }
    }

    /// Detect format from file header/magic bytes
    pub fn detect_from_header(data: &[u8]) -> Option<AudioFormatType> {
        if data.len() < 12 {
            return None;
        }

        // WAV format: "RIFF....WAVE"
        if &data[0..4] == b"RIFF" && &data[8..12] == b"WAVE" {
            return Some(AudioFormatType::Wav);
        }

        // FLAC format: "fLaC"
        if &data[0..4] == b"fLaC" {
            return Some(AudioFormatType::Flac);
        }

        // MP3 format: Check for ID3 tag or sync frame
        if &data[0..3] == b"ID3" {
            return Some(AudioFormatType::Mp3);
        }
        if data.len() >= 2 && data[0] == 0xFF && (data[1] & 0xE0) == 0xE0 {
            return Some(AudioFormatType::Mp3);
        }

        // OGG format: "OggS"
        if &data[0..4] == b"OggS" {
            return Some(AudioFormatType::Ogg);
        }

        // AIFF format: "FORM....AIFF"
        if data.len() >= 12 && &data[0..4] == b"FORM" && &data[8..12] == b"AIFF" {
            return Some(AudioFormatType::Aiff);
        }

        None
    }

    /// Detect format from MIME type
    pub fn detect_from_mime_type(mime_type: &str) -> Option<AudioFormatType> {
        match mime_type {
            "audio/wav" | "audio/wave" | "audio/x-wav" => Some(AudioFormatType::Wav),
            "audio/flac" => Some(AudioFormatType::Flac),
            "audio/mpeg" | "audio/mp3" => Some(AudioFormatType::Mp3),
            "audio/aac" | "audio/mp4" => Some(AudioFormatType::Aac),
            "audio/opus" => Some(AudioFormatType::Opus),
            "audio/ogg" => Some(AudioFormatType::Ogg),
            "audio/aiff" | "audio/x-aiff" => Some(AudioFormatType::Aiff),
            _ => None,
        }
    }
}

/// Audio format converter
pub struct FormatConverter;

impl FormatConverter {
    /// Convert audio data to target format specification
    pub fn convert(audio: &AudioData, target_format: &AudioFormat) -> Result<AudioData> {
        let mut result = audio.clone();

        // Convert sample rate if needed
        if audio.format.sample_rate != target_format.sample_rate {
            result = result.resample(target_format.sample_rate);
        }

        // Convert channels if needed
        if audio.format.channels != target_format.channels {
            if target_format.channels == 1 && audio.format.channels > 1 {
                // Convert to mono
                result = result.to_mono();
            } else if target_format.channels > 1 && audio.format.channels == 1 {
                // Convert mono to multi-channel (duplicate channels)
                let channels_needed = target_format.channels as usize;
                let frames = result.frames();
                let mut multi_channel_samples = Vec::with_capacity(frames * channels_needed);

                for frame in 0..frames {
                    let mono_sample = result.samples[frame];
                    for _ in 0..channels_needed {
                        multi_channel_samples.push(mono_sample);
                    }
                }

                result.samples = multi_channel_samples;
                result.format.channels = target_format.channels;
            }
            // Complex channel conversions (surround sound to stereo, etc.)
            else {
                result.samples = Self::convert_complex_channels(
                    &result.samples,
                    audio.format.channels,
                    target_format.channels,
                    result.frames(),
                )?;
                result.format.channels = target_format.channels;
            }
        }

        // Update format metadata
        result.format.format_type = target_format.format_type;
        result.format.bits_per_sample = target_format.bits_per_sample;
        result.format.bit_rate = target_format.bit_rate;

        Ok(result)
    }

    /// Get optimal format for conversion target
    pub fn get_optimal_format(
        source_format: &AudioFormat,
        target_type: AudioFormatType,
        quality_preference: FormatQuality,
    ) -> AudioFormat {
        let sample_rate = match quality_preference {
            FormatQuality::Low => 22050,
            FormatQuality::Medium => 44100,
            FormatQuality::High => source_format.sample_rate.max(44100),
            FormatQuality::Highest => source_format.sample_rate.max(48000),
        };

        let channels = source_format.channels;

        let mut format = AudioFormat::new(target_type, sample_rate, channels);

        // Set format-specific parameters
        match target_type {
            AudioFormatType::Wav => {
                format = format.with_bits_per_sample(match quality_preference {
                    FormatQuality::Low => 16,
                    FormatQuality::Medium => 16,
                    FormatQuality::High => 24,
                    FormatQuality::Highest => 24,
                });
            }
            AudioFormatType::Wav24 => {
                format = format.with_bits_per_sample(24);
            }
            AudioFormatType::Wav32f => {
                format = format.with_bits_per_sample(32);
            }
            AudioFormatType::Mp3 => {
                format = format.with_bit_rate(match quality_preference {
                    FormatQuality::Low => 128,
                    FormatQuality::Medium => 192,
                    FormatQuality::High => 256,
                    FormatQuality::Highest => 320,
                });
            }
            AudioFormatType::Aac => {
                format = format.with_bit_rate(match quality_preference {
                    FormatQuality::Low => 96,
                    FormatQuality::Medium => 128,
                    FormatQuality::High => 192,
                    FormatQuality::Highest => 256,
                });
            }
            AudioFormatType::Opus => {
                format = format.with_bit_rate(match quality_preference {
                    FormatQuality::Low => 64,
                    FormatQuality::Medium => 96,
                    FormatQuality::High => 128,
                    FormatQuality::Highest => 192,
                });
            }
            _ => {} // Use defaults
        }

        format
    }

    /// Convert between complex channel configurations (5.1, 7.1, stereo, etc.)
    fn convert_complex_channels(
        samples: &[f32],
        source_channels: u16,
        target_channels: u16,
        frames: usize,
    ) -> Result<Vec<f32>> {
        let source_ch = source_channels as usize;
        let target_ch = target_channels as usize;

        // Common channel layouts
        // Stereo: Left, Right
        // 5.1: Front Left, Front Right, Center, LFE, Rear Left, Rear Right
        // 7.1: Front Left, Front Right, Center, LFE, Side Left, Side Right, Rear Left, Rear Right

        match (source_ch, target_ch) {
            // 5.1 to stereo downmix
            (6, 2) => {
                let mut result = Vec::with_capacity(frames * 2);
                for frame in 0..frames {
                    let base_idx = frame * 6;
                    let fl = samples[base_idx]; // Front Left
                    let fr = samples[base_idx + 1]; // Front Right
                    let center = samples[base_idx + 2]; // Center
                    let _lfe = samples[base_idx + 3]; // LFE (Low Frequency Effects)
                    let rl = samples[base_idx + 4]; // Rear Left
                    let rr = samples[base_idx + 5]; // Rear Right

                    // Standard 5.1 to stereo downmix formula
                    let left = fl + (center * 0.707) + (rl * 0.707);
                    let right = fr + (center * 0.707) + (rr * 0.707);

                    result.push(left);
                    result.push(right);
                }
                Ok(result)
            }

            // 7.1 to stereo downmix
            (8, 2) => {
                let mut result = Vec::with_capacity(frames * 2);
                for frame in 0..frames {
                    let base_idx = frame * 8;
                    let fl = samples[base_idx]; // Front Left
                    let fr = samples[base_idx + 1]; // Front Right
                    let center = samples[base_idx + 2]; // Center
                    let _lfe = samples[base_idx + 3]; // LFE
                    let sl = samples[base_idx + 4]; // Side Left
                    let sr = samples[base_idx + 5]; // Side Right
                    let rl = samples[base_idx + 6]; // Rear Left
                    let rr = samples[base_idx + 7]; // Rear Right

                    // 7.1 to stereo downmix formula
                    let left = fl + (center * 0.707) + (sl * 0.5) + (rl * 0.5);
                    let right = fr + (center * 0.707) + (sr * 0.5) + (rr * 0.5);

                    result.push(left);
                    result.push(right);
                }
                Ok(result)
            }

            // 7.1 to 5.1 downmix
            (8, 6) => {
                let mut result = Vec::with_capacity(frames * 6);
                for frame in 0..frames {
                    let base_idx = frame * 8;
                    let fl = samples[base_idx]; // Front Left
                    let fr = samples[base_idx + 1]; // Front Right
                    let center = samples[base_idx + 2]; // Center
                    let lfe = samples[base_idx + 3]; // LFE
                    let sl = samples[base_idx + 4]; // Side Left
                    let sr = samples[base_idx + 5]; // Side Right
                    let rl = samples[base_idx + 6]; // Rear Left
                    let rr = samples[base_idx + 7]; // Rear Right

                    // Mix side and rear channels for 5.1 rear channels
                    let mixed_rl = (sl + rl) * 0.707; // Mix side and rear left
                    let mixed_rr = (sr + rr) * 0.707; // Mix side and rear right

                    result.push(fl);
                    result.push(fr);
                    result.push(center);
                    result.push(lfe);
                    result.push(mixed_rl);
                    result.push(mixed_rr);
                }
                Ok(result)
            }

            // Stereo to 5.1 upmix (basic)
            (2, 6) => {
                let mut result = Vec::with_capacity(frames * 6);
                for frame in 0..frames {
                    let base_idx = frame * 2;
                    let left = samples[base_idx];
                    let right = samples[base_idx + 1];

                    // Basic stereo to 5.1 upmix
                    result.push(left); // Front Left
                    result.push(right); // Front Right
                    result.push((left + right) * 0.5); // Center (mixed)
                    result.push(0.0); // LFE (silent)
                    result.push(left * 0.5); // Rear Left (attenuated)
                    result.push(right * 0.5); // Rear Right (attenuated)
                }
                Ok(result)
            }

            // General case: distribute channels evenly or truncate
            _ => {
                let mut result = Vec::with_capacity(frames * target_ch);
                for frame in 0..frames {
                    for target_ch_idx in 0..target_ch {
                        let source_ch_idx = if source_ch > target_ch {
                            // Downmix: map multiple source channels to fewer target channels
                            (target_ch_idx * source_ch) / target_ch
                        } else {
                            // Upmix: repeat source channels for target channels
                            target_ch_idx % source_ch
                        };

                        let sample_idx = frame * source_ch + source_ch_idx;
                        let sample = if sample_idx < samples.len() {
                            samples[sample_idx]
                        } else {
                            0.0 // Pad with silence if out of bounds
                        };
                        result.push(sample);
                    }
                }
                Ok(result)
            }
        }
    }
}

/// Quality preference for format conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatQuality {
    /// Low quality, small file size
    Low,
    /// Medium quality, balanced
    Medium,
    /// High quality, larger file size
    High,
    /// Highest quality, largest file size
    Highest,
}

/// Audio format reader (placeholder for future implementation)
pub struct AudioReader;

impl AudioReader {
    /// Read audio from file (basic WAV support)
    pub fn read_file<P: AsRef<Path>>(path: P) -> Result<AudioData> {
        // This is a placeholder implementation
        // In a real implementation, you would:
        // 1. Detect the format
        // 2. Use appropriate decoder (hound for WAV, symphonia for others)
        // 3. Convert to our internal format

        let format_type = FormatDetector::detect_from_extension(&path)
            .ok_or_else(|| Error::audio("Unsupported file format".to_string()))?;

        match format_type {
            AudioFormatType::Wav | AudioFormatType::Wav24 | AudioFormatType::Wav32f => {
                Self::read_wav_placeholder(path)
            }
            _ => Err(Error::audio(format!(
                "Format {format_type:?} not yet implemented - requires additional dependencies"
            ))),
        }
    }

    /// Basic WAV file reading (placeholder)
    fn read_wav_placeholder<P: AsRef<Path>>(path: P) -> Result<AudioData> {
        // Placeholder implementation - in real code would use hound crate
        let format = AudioFormat::new(AudioFormatType::Wav, 44100, 2);
        let samples = vec![0.0f32; 44100]; // 1 second of silence
        Ok(AudioData::new(samples, format))
    }

    /// Read from memory buffer
    pub fn read_buffer(buffer: &[u8]) -> Result<AudioData> {
        let format_type = FormatDetector::detect_from_header(buffer)
            .ok_or_else(|| Error::audio("Unknown audio format in buffer".to_string()))?;

        match format_type {
            AudioFormatType::Wav => Self::read_wav_buffer(buffer),
            _ => Err(Error::audio(format!(
                "Buffer format {format_type:?} not yet implemented"
            ))),
        }
    }

    fn read_wav_buffer(buffer: &[u8]) -> Result<AudioData> {
        // Placeholder - would parse WAV header and data
        let format = AudioFormat::new(AudioFormatType::Wav, 44100, 2);
        let samples = vec![0.0f32; 1024]; // Placeholder data
        Ok(AudioData::new(samples, format))
    }
}

/// Audio format writer (placeholder for future implementation)
pub struct AudioWriter;

impl AudioWriter {
    /// Write audio to file
    pub fn write_file<P: AsRef<Path>>(
        audio: &AudioData,
        path: P,
        target_format: Option<AudioFormatType>,
    ) -> Result<()> {
        let format_type = target_format
            .or_else(|| FormatDetector::detect_from_extension(&path))
            .unwrap_or(AudioFormatType::Wav);

        match format_type {
            AudioFormatType::Wav | AudioFormatType::Wav24 | AudioFormatType::Wav32f => {
                Self::write_wav_placeholder(audio, path)
            }
            _ => Err(Error::audio(format!(
                "Writing format {format_type:?} not yet implemented - requires additional dependencies"
            ))),
        }
    }

    fn write_wav_placeholder<P: AsRef<Path>>(audio: &AudioData, path: P) -> Result<()> {
        // Placeholder - would use hound crate to write WAV file
        Ok(())
    }

    /// Write to memory buffer
    pub fn write_buffer(audio: &AudioData, format_type: AudioFormatType) -> Result<Vec<u8>> {
        match format_type {
            AudioFormatType::Wav => Self::write_wav_buffer(audio),
            _ => Err(Error::audio(format!(
                "Buffer format {format_type:?} not yet implemented"
            ))),
        }
    }

    fn write_wav_buffer(audio: &AudioData) -> Result<Vec<u8>> {
        // Placeholder - would create WAV file in memory
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_format_type_properties() {
        assert_eq!(AudioFormatType::Wav.extensions(), &["wav"]);
        assert_eq!(AudioFormatType::Mp3.mime_type(), "audio/mpeg");
        assert!(AudioFormatType::Mp3.is_lossy());
        assert!(AudioFormatType::Wav.is_lossless());
        assert!(AudioFormatType::Mp3.typical_bitrates().is_some());
        assert!(AudioFormatType::Wav.typical_bitrates().is_none());
    }

    #[test]
    fn test_audio_format_creation() {
        let format = AudioFormat::new(AudioFormatType::Wav, 44100, 2)
            .with_bits_per_sample(24)
            .with_metadata("title".to_string(), "Test Audio".to_string());

        assert_eq!(format.format_type, AudioFormatType::Wav);
        assert_eq!(format.sample_rate, 44100);
        assert_eq!(format.channels, 2);
        assert_eq!(format.bits_per_sample, Some(24));
        assert_eq!(
            format.metadata.get("title"),
            Some(&"Test Audio".to_string())
        );
    }

    #[test]
    fn test_file_size_estimation() {
        let format = AudioFormat::new(AudioFormatType::Wav, 44100, 2).with_bits_per_sample(16);

        // 1 second of 44.1kHz stereo 16-bit should be about 176,400 bytes
        let size = format.estimated_file_size(1.0);
        assert_eq!(size, 176400);

        let mp3_format = AudioFormat::new(AudioFormatType::Mp3, 44100, 2).with_bit_rate(128);

        // 1 second of 128kbps MP3 should be 16,000 bytes
        let mp3_size = mp3_format.estimated_file_size(1.0);
        assert_eq!(mp3_size, 16000);
    }

    #[test]
    fn test_sample_rate_support() {
        let opus_format = AudioFormat::new(AudioFormatType::Opus, 48000, 2);
        assert!(opus_format.supports_sample_rate(48000));
        assert!(!opus_format.supports_sample_rate(44100));

        let wav_format = AudioFormat::new(AudioFormatType::Wav, 44100, 2);
        assert!(wav_format.supports_sample_rate(44100));
        assert!(wav_format.supports_sample_rate(96000));
    }

    #[test]
    fn test_audio_data_operations() {
        // Create stereo audio data
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // 3 frames, 2 channels
        let format = AudioFormat::new(AudioFormatType::Wav, 44100, 2);
        let audio = AudioData::new(samples, format);

        assert_eq!(audio.frames(), 3);
        assert_eq!(audio.duration(), 3.0 / 44100.0);

        // Test channel splitting
        let channels = audio.split_channels();
        assert_eq!(channels.len(), 2);
        assert_eq!(channels[0], vec![0.1, 0.3, 0.5]);
        assert_eq!(channels[1], vec![0.2, 0.4, 0.6]);

        // Test mono conversion
        let mono = audio.to_mono();
        assert_eq!(mono.format.channels, 1);
        // Use approximate comparison for floating point precision
        let expected = vec![0.15, 0.35, 0.55];
        for (i, (&actual, &expected)) in mono.samples.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Sample {} mismatch: {} vs {}",
                i,
                actual,
                expected
            );
        }
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(
            FormatDetector::detect_from_extension("test.wav"),
            Some(AudioFormatType::Wav)
        );
        assert_eq!(
            FormatDetector::detect_from_extension("test.mp3"),
            Some(AudioFormatType::Mp3)
        );
        assert_eq!(FormatDetector::detect_from_extension("test.unknown"), None);

        // Test header detection
        let wav_header = b"RIFF\x00\x00\x00\x00WAVE";
        assert_eq!(
            FormatDetector::detect_from_header(wav_header),
            Some(AudioFormatType::Wav)
        );

        let flac_header = b"fLaC\x00\x00\x00\x22\x10\x00\x10\x00";
        assert_eq!(
            FormatDetector::detect_from_header(flac_header),
            Some(AudioFormatType::Flac)
        );
    }

    #[test]
    fn test_format_converter() {
        let source_format = AudioFormat::new(AudioFormatType::Wav, 22050, 1);
        let target_format = AudioFormat::new(AudioFormatType::Mp3, 44100, 2);

        let optimal = FormatConverter::get_optimal_format(
            &source_format,
            AudioFormatType::Mp3,
            FormatQuality::High,
        );

        assert_eq!(optimal.format_type, AudioFormatType::Mp3);
        assert_eq!(optimal.sample_rate, 44100);
        assert_eq!(optimal.channels, 1); // Preserves source channel count
        assert_eq!(optimal.bit_rate, Some(256)); // High quality MP3
    }

    #[test]
    fn test_audio_resampling() {
        // Create 1 second of 22kHz mono sine wave-like data
        let samples: Vec<f32> = (0..22050).map(|i| (i as f32 * 0.01).sin()).collect();
        let format = AudioFormat::new(AudioFormatType::Wav, 22050, 1);
        let audio = AudioData::new(samples, format);

        // Resample to 44kHz
        let resampled = audio.resample(44100);
        assert_eq!(resampled.format.sample_rate, 44100);
        assert_eq!(resampled.samples.len(), 44100); // Should be ~2x length
    }

    #[test]
    fn test_channel_conversion() {
        // Test mono to stereo conversion via format converter
        let mono_samples = vec![0.1, 0.2, 0.3];
        let mono_format = AudioFormat::new(AudioFormatType::Wav, 44100, 1);
        let mono_audio = AudioData::new(mono_samples, mono_format);

        let stereo_format = AudioFormat::new(AudioFormatType::Wav, 44100, 2);
        let stereo_audio = FormatConverter::convert(&mono_audio, &stereo_format).unwrap();

        assert_eq!(stereo_audio.format.channels, 2);
        assert_eq!(stereo_audio.samples.len(), 6); // 3 frames * 2 channels
        assert_eq!(stereo_audio.samples, vec![0.1, 0.1, 0.2, 0.2, 0.3, 0.3]);
    }
}
