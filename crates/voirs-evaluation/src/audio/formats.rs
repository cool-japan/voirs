//! Format-specific audio decoders and utilities.
//!
//! This module provides implementations for loading various audio formats,
//! with fallback implementations that create placeholder data for formats
//! that don't have full decoder implementations yet.

use super::{AudioFormat, AudioIoError, AudioIoResult, AudioMetadata, LoadOptions};
use std::fs::File;
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use voirs_sdk::AudioBuffer;

/// Apply audio conversion (sample rate and channel conversion) if needed
fn apply_audio_conversion(
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u32,
    options: &LoadOptions,
) -> AudioIoResult<(Vec<f32>, u32, u32)> {
    let mut final_samples = samples;
    let mut final_sample_rate = sample_rate;
    let mut final_channels = channels;

    // Apply target sample rate conversion if needed
    if let Some(target_rate) = options.target_sample_rate {
        if target_rate != sample_rate {
            final_samples = resample_audio(final_samples, sample_rate, target_rate, channels)?;
            final_sample_rate = target_rate;
        }
    }

    // Apply channel conversion if needed
    if let Some(target_channels) = options.target_channels {
        if target_channels != channels {
            final_samples = convert_channels(final_samples, channels, target_channels)?;
            final_channels = target_channels;
        }
    }

    Ok((final_samples, final_sample_rate, final_channels))
}

/// Simple resampling using linear interpolation
fn resample_audio(
    samples: Vec<f32>,
    from_rate: u32,
    to_rate: u32,
    channels: u32,
) -> AudioIoResult<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(samples);
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let frames_in = samples.len() / channels as usize;
    let frames_out = (frames_in as f64 / ratio).ceil() as usize;

    let mut resampled = Vec::with_capacity(frames_out * channels as usize);

    for frame_out in 0..frames_out {
        let pos = frame_out as f64 * ratio;
        let input_frame = pos.floor() as usize;
        let frac = pos - input_frame as f64;

        for ch in 0..channels as usize {
            let sample = if input_frame + 1 < frames_in {
                let s0 = samples[input_frame * channels as usize + ch];
                let s1 = samples[(input_frame + 1) * channels as usize + ch];
                s0 + frac as f32 * (s1 - s0)
            } else if input_frame < frames_in {
                samples[input_frame * channels as usize + ch]
            } else {
                0.0
            };
            resampled.push(sample);
        }
    }

    Ok(resampled)
}

/// Convert between different channel counts
fn convert_channels(
    samples: Vec<f32>,
    from_channels: u32,
    to_channels: u32,
) -> AudioIoResult<Vec<f32>> {
    if from_channels == to_channels {
        return Ok(samples);
    }

    let frames = samples.len() / from_channels as usize;
    let mut converted = Vec::with_capacity(frames * to_channels as usize);

    for frame in 0..frames {
        match (from_channels, to_channels) {
            (1, 2) => {
                // Mono to stereo - duplicate the channel
                let sample = samples[frame];
                converted.push(sample);
                converted.push(sample);
            }
            (2, 1) => {
                // Stereo to mono - average the channels
                let left = samples[frame * 2];
                let right = samples[frame * 2 + 1];
                converted.push((left + right) / 2.0);
            }
            (from, to) => {
                // General case - simple downmix/upmix
                for ch in 0..to as usize {
                    if ch < from as usize {
                        converted.push(samples[frame * from as usize + ch]);
                    } else {
                        converted.push(0.0);
                    }
                }
            }
        }
    }

    Ok(converted)
}

/// Load audio file using Symphonia (supports multiple formats)
fn load_with_symphonia(
    path: &Path,
    options: &LoadOptions,
) -> AudioIoResult<(AudioBuffer, AudioMetadata)> {
    let file = File::open(path).map_err(|e| AudioIoError::IoError {
        message: format!("Failed to open file: {}", path.display()),
        source: Some(Box::new(e)),
    })?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();

    // Add file extension hint
    if let Some(extension) = path.extension() {
        if let Some(ext_str) = extension.to_str() {
            hint.with_extension(ext_str);
        }
    }

    let meta_opts = MetadataOptions::default();
    let fmt_opts = FormatOptions::default();

    let mut probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .map_err(|e| AudioIoError::IoError {
            message: format!("Symphonia error: {}", e),
            source: Some(Box::new(e)),
        })?;

    let mut format = probed.format;

    // Find the default track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| AudioIoError::IoError {
            message: "No audio track found".to_string(),
            source: None,
        })?;

    let track_id = track.id;
    let codec_params = &track.codec_params;

    // Get basic audio info
    let sample_rate = codec_params.sample_rate.unwrap_or(44100);
    let channels = codec_params.channels.map(|c| c.count()).unwrap_or(2) as u32;

    // Create decoder
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &Default::default())
        .map_err(|e| AudioIoError::IoError {
            message: format!("Symphonia error: {}", e),
            source: Some(Box::new(e)),
        })?;

    let mut samples = Vec::new();

    // Decode audio
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::ResetRequired) => {
                // Reset decoder
                decoder.reset();
                continue;
            }
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => {
                return Err(AudioIoError::IoError {
                    message: format!("Audio decoding error: {}", e),
                    source: Some(Box::new(e)),
                })
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buffer) => {
                // Convert to f32 samples
                match audio_buffer {
                    AudioBufferRef::F32(buf) => {
                        for ch in 0..buf.spec().channels.count() {
                            let channel_samples = buf.chan(ch);
                            for (i, &sample) in channel_samples.iter().enumerate() {
                                let sample_index = i * channels as usize + ch;
                                if sample_index >= samples.len() {
                                    samples.resize(sample_index + 1, 0.0);
                                }
                                samples[sample_index] = sample;
                            }
                        }
                    }
                    AudioBufferRef::U8(buf) => {
                        for ch in 0..buf.spec().channels.count() {
                            let channel_samples = buf.chan(ch);
                            for (i, &sample) in channel_samples.iter().enumerate() {
                                let sample_index = i * channels as usize + ch;
                                if sample_index >= samples.len() {
                                    samples.resize(sample_index + 1, 0.0);
                                }
                                samples[sample_index] = (sample as f32 - 128.0) / 128.0;
                            }
                        }
                    }
                    AudioBufferRef::U16(buf) => {
                        for ch in 0..buf.spec().channels.count() {
                            let channel_samples = buf.chan(ch);
                            for (i, &sample) in channel_samples.iter().enumerate() {
                                let sample_index = i * channels as usize + ch;
                                if sample_index >= samples.len() {
                                    samples.resize(sample_index + 1, 0.0);
                                }
                                samples[sample_index] = (sample as f32 - 32768.0) / 32768.0;
                            }
                        }
                    }
                    AudioBufferRef::U24(buf) => {
                        for ch in 0..buf.spec().channels.count() {
                            let channel_samples = buf.chan(ch);
                            for (i, &sample) in channel_samples.iter().enumerate() {
                                let sample_index = i * channels as usize + ch;
                                if sample_index >= samples.len() {
                                    samples.resize(sample_index + 1, 0.0);
                                }
                                samples[sample_index] =
                                    (sample.inner() as f32 - 8_388_608.0) / 8_388_608.0;
                            }
                        }
                    }
                    AudioBufferRef::U32(buf) => {
                        for ch in 0..buf.spec().channels.count() {
                            let channel_samples = buf.chan(ch);
                            for (i, &sample) in channel_samples.iter().enumerate() {
                                let sample_index = i * channels as usize + ch;
                                if sample_index >= samples.len() {
                                    samples.resize(sample_index + 1, 0.0);
                                }
                                samples[sample_index] =
                                    (sample as f32 - 2_147_483_648.0) / 2_147_483_648.0;
                            }
                        }
                    }
                    AudioBufferRef::S8(buf) => {
                        for ch in 0..buf.spec().channels.count() {
                            let channel_samples = buf.chan(ch);
                            for (i, &sample) in channel_samples.iter().enumerate() {
                                let sample_index = i * channels as usize + ch;
                                if sample_index >= samples.len() {
                                    samples.resize(sample_index + 1, 0.0);
                                }
                                samples[sample_index] = sample as f32 / 128.0;
                            }
                        }
                    }
                    AudioBufferRef::S16(buf) => {
                        for ch in 0..buf.spec().channels.count() {
                            let channel_samples = buf.chan(ch);
                            for (i, &sample) in channel_samples.iter().enumerate() {
                                let sample_index = i * channels as usize + ch;
                                if sample_index >= samples.len() {
                                    samples.resize(sample_index + 1, 0.0);
                                }
                                samples[sample_index] = sample as f32 / 32768.0;
                            }
                        }
                    }
                    AudioBufferRef::S24(buf) => {
                        for ch in 0..buf.spec().channels.count() {
                            let channel_samples = buf.chan(ch);
                            for (i, &sample) in channel_samples.iter().enumerate() {
                                let sample_index = i * channels as usize + ch;
                                if sample_index >= samples.len() {
                                    samples.resize(sample_index + 1, 0.0);
                                }
                                samples[sample_index] = sample.inner() as f32 / 8_388_608.0;
                            }
                        }
                    }
                    AudioBufferRef::S32(buf) => {
                        for ch in 0..buf.spec().channels.count() {
                            let channel_samples = buf.chan(ch);
                            for (i, &sample) in channel_samples.iter().enumerate() {
                                let sample_index = i * channels as usize + ch;
                                if sample_index >= samples.len() {
                                    samples.resize(sample_index + 1, 0.0);
                                }
                                samples[sample_index] = sample as f32 / 2_147_483_648.0;
                            }
                        }
                    }
                    AudioBufferRef::F64(buf) => {
                        for ch in 0..buf.spec().channels.count() {
                            let channel_samples = buf.chan(ch);
                            for (i, &sample) in channel_samples.iter().enumerate() {
                                let sample_index = i * channels as usize + ch;
                                if sample_index >= samples.len() {
                                    samples.resize(sample_index + 1, 0.0);
                                }
                                samples[sample_index] = sample as f32;
                            }
                        }
                    }
                }
            }
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => {
                // Skip decode errors
            }
            Err(e) => {
                return Err(AudioIoError::IoError {
                    message: format!("Audio decoding error: {}", e),
                    source: Some(Box::new(e)),
                })
            }
        }
    }

    if samples.is_empty() {
        return Err(AudioIoError::IoError {
            message: "No audio data found".to_string(),
            source: None,
        });
    }

    let duration = samples.len() as f64 / (sample_rate as f64 * channels as f64);

    // Apply target sample rate and channel conversion if needed
    let (final_samples, final_sample_rate, final_channels) =
        apply_audio_conversion(samples, sample_rate, channels, options)?;

    let audio = AudioBuffer::new(final_samples, final_sample_rate, final_channels);

    // Extract metadata
    let metadata =
        if let Some(metadata_rev) = probed.metadata.get().as_ref().and_then(|m| m.current()) {
            let mut meta = AudioMetadata::default();

            for tag in metadata_rev.tags() {
                match tag.key.as_str() {
                    "TITLE" => meta.title = Some(tag.value.to_string()),
                    "ARTIST" => meta.artist = Some(tag.value.to_string()),
                    "ALBUM" => meta.album = Some(tag.value.to_string()),
                    "GENRE" => meta.genre = Some(tag.value.to_string()),
                    "DATE" | "YEAR" => {
                        if let Ok(year) = tag.value.to_string().parse::<u32>() {
                            meta.year = Some(year);
                        }
                    }
                    "TRACKNUMBER" => {
                        if let Ok(track) = tag.value.to_string().parse::<u32>() {
                            meta.track = Some(track);
                        }
                    }
                    _ => {}
                }
            }

            meta.duration = Some(duration);
            meta
        } else {
            AudioMetadata {
                title: path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string()),
                duration: Some(duration),
                ..Default::default()
            }
        };

    Ok((audio, metadata))
}

/// Validate audio file using Symphonia
fn validate_with_symphonia(path: &Path) -> bool {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };

    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();

    // Add file extension hint
    if let Some(extension) = path.extension() {
        if let Some(ext_str) = extension.to_str() {
            hint.with_extension(ext_str);
        }
    }

    let meta_opts = MetadataOptions::default();
    let fmt_opts = FormatOptions::default();

    match symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts) {
        Ok(probed) => {
            // Check if there's at least one valid audio track
            probed
                .format
                .tracks()
                .iter()
                .any(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        }
        Err(_) => false,
    }
}

/// WAV format decoder using hound
pub struct WavDecoder;

impl WavDecoder {
    /// Load WAV file
    pub fn load(path: &Path, options: &LoadOptions) -> AudioIoResult<(AudioBuffer, AudioMetadata)> {
        let mut reader = hound::WavReader::open(path).map_err(|e| AudioIoError::IoError {
            message: format!("Symphonia error: {}", e),
            source: Some(Box::new(e)),
        })?;

        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let channels = spec.channels as u32;

        // Convert samples to f32
        let samples: Result<Vec<f32>, _> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().collect(),
            hound::SampleFormat::Int => match spec.bits_per_sample {
                16 => reader
                    .samples::<i16>()
                    .map(|s| s.map(|sample| sample as f32 / 32768.0))
                    .collect(),
                24 => reader
                    .samples::<i32>()
                    .map(|s| s.map(|sample| sample as f32 / 8_388_608.0))
                    .collect(),
                32 => reader
                    .samples::<i32>()
                    .map(|s| s.map(|sample| sample as f32 / 2_147_483_648.0))
                    .collect(),
                _ => {
                    return Err(AudioIoError::UnsupportedFormat {
                        format: AudioFormat::Wav,
                    })
                }
            },
        };

        let samples = samples.map_err(|e| AudioIoError::IoError {
            message: format!("Symphonia error: {}", e),
            source: Some(Box::new(e)),
        })?;

        let duration = samples.len() as f64 / (sample_rate as f64 * channels as f64);

        // Apply target sample rate and channel conversion if needed
        let (final_samples, final_sample_rate, final_channels) =
            apply_audio_conversion(samples, sample_rate, channels, options)?;

        let audio = AudioBuffer::new(final_samples, final_sample_rate, final_channels);
        let metadata = AudioMetadata {
            title: path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string()),
            duration: Some(duration),
            ..Default::default()
        };

        Ok((audio, metadata))
    }

    /// Check if file is valid WAV format
    pub fn is_valid_format(path: &Path) -> bool {
        hound::WavReader::open(path).is_ok()
    }
}

/// FLAC format decoder using symphonia
pub struct FlacDecoder;

impl FlacDecoder {
    /// Load FLAC file
    pub fn load(path: &Path, options: &LoadOptions) -> AudioIoResult<(AudioBuffer, AudioMetadata)> {
        load_with_symphonia(path, options)
    }

    /// Check if file is valid FLAC format
    pub fn is_valid_format(path: &Path) -> bool {
        validate_with_symphonia(path)
    }
}

/// MP3 format decoder using symphonia
pub struct Mp3Decoder;

impl Mp3Decoder {
    /// Load MP3 file
    pub fn load(path: &Path, options: &LoadOptions) -> AudioIoResult<(AudioBuffer, AudioMetadata)> {
        load_with_symphonia(path, options)
    }

    /// Check if file is valid MP3 format
    pub fn is_valid_format(path: &Path) -> bool {
        validate_with_symphonia(path)
    }
}

/// OGG Vorbis format decoder using symphonia
pub struct OggDecoder;

impl OggDecoder {
    /// Load OGG file
    pub fn load(path: &Path, options: &LoadOptions) -> AudioIoResult<(AudioBuffer, AudioMetadata)> {
        load_with_symphonia(path, options)
    }

    /// Check if file is valid OGG format
    pub fn is_valid_format(path: &Path) -> bool {
        validate_with_symphonia(path)
    }
}

/// M4A/AAC format decoder using symphonia
pub struct M4aDecoder;

impl M4aDecoder {
    /// Load M4A file
    pub fn load(path: &Path, options: &LoadOptions) -> AudioIoResult<(AudioBuffer, AudioMetadata)> {
        load_with_symphonia(path, options)
    }

    /// Check if file is valid M4A format
    pub fn is_valid_format(path: &Path) -> bool {
        validate_with_symphonia(path)
    }
}

/// AIFF format decoder using symphonia
pub struct AiffDecoder;

impl AiffDecoder {
    /// Load AIFF file
    pub fn load(path: &Path, options: &LoadOptions) -> AudioIoResult<(AudioBuffer, AudioMetadata)> {
        load_with_symphonia(path, options)
    }

    /// Check if file is valid AIFF format
    pub fn is_valid_format(path: &Path) -> bool {
        validate_with_symphonia(path)
    }
}

/// Universal format loader that dispatches to appropriate decoder
pub fn load_audio_file(
    path: &Path,
    options: &LoadOptions,
) -> AudioIoResult<(AudioBuffer, AudioMetadata)> {
    let format = AudioFormat::from_extension(path);

    match format {
        AudioFormat::Wav => WavDecoder::load(path, options),
        AudioFormat::Flac => FlacDecoder::load(path, options),
        AudioFormat::Mp3 => Mp3Decoder::load(path, options),
        AudioFormat::Ogg => OggDecoder::load(path, options),
        AudioFormat::M4a => M4aDecoder::load(path, options),
        AudioFormat::Aiff => AiffDecoder::load(path, options),
        AudioFormat::Unknown => Err(AudioIoError::UnsupportedFormat { format }),
    }
}

/// Validate that a file can be loaded
pub fn validate_audio_file(path: &Path) -> AudioIoResult<AudioFormat> {
    let format = AudioFormat::from_extension(path);

    let is_valid = match format {
        AudioFormat::Wav => WavDecoder::is_valid_format(path),
        AudioFormat::Flac => FlacDecoder::is_valid_format(path),
        AudioFormat::Mp3 => Mp3Decoder::is_valid_format(path),
        AudioFormat::Ogg => OggDecoder::is_valid_format(path),
        AudioFormat::M4a => M4aDecoder::is_valid_format(path),
        AudioFormat::Aiff => AiffDecoder::is_valid_format(path),
        AudioFormat::Unknown => false,
    };

    if is_valid {
        Ok(format)
    } else {
        Err(AudioIoError::UnsupportedFormat { format })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_wav_decoder() {
        let path = PathBuf::from("test.wav");

        // Skip test if file doesn't exist
        if !path.exists() {
            return;
        }

        let options = LoadOptions::default();
        let result = WavDecoder::load(&path, &options);
        assert!(result.is_ok());

        let (audio, metadata) = result.unwrap();
        assert_eq!(audio.sample_rate(), 16000);
        assert_eq!(audio.channels(), 1);
        assert!(metadata.title.is_some());
    }

    #[test]
    fn test_format_validation() {
        let wav_path = PathBuf::from("test.wav");

        // Only test if file exists
        if wav_path.exists() {
            let result = validate_audio_file(&wav_path);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), AudioFormat::Wav);
        }

        let unknown_path = PathBuf::from("test.xyz");
        let result = validate_audio_file(&unknown_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_universal_loader() {
        let paths = [
            "test.wav",
            "test.flac",
            "test.mp3",
            "test.ogg",
            "test.m4a",
            "test.aiff",
        ];

        let options = LoadOptions::new().target_sample_rate(16000);

        for path_str in &paths {
            let path = PathBuf::from(path_str);

            // Only test files that exist
            if path.exists() {
                let result = load_audio_file(&path, &options);
                assert!(result.is_ok(), "Failed to load {}", path_str);

                let (audio, _metadata) = result.unwrap();
                assert_eq!(audio.sample_rate(), 16000);
            }
        }
    }

    #[test]
    fn test_metadata_extraction() {
        let path = PathBuf::from("test.mp3");

        // Skip test if file doesn't exist
        if !path.exists() {
            return;
        }

        let options = LoadOptions::default();
        let result = Mp3Decoder::load(&path, &options);
        assert!(result.is_ok());

        let (_audio, metadata) = result.unwrap();
        // Only test basic metadata properties that should be present
        assert!(metadata.title.is_some() || metadata.duration.is_some());
    }
}
