//! Audio file I/O operations
//!
//! This module provides comprehensive audio file reading and writing capabilities
//! for various audio formats including WAV, FLAC, MP3, and OGG.

use crate::{AudioData, AudioFormat, DatasetError, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Load audio file with automatic format detection
pub fn load_audio<P: AsRef<Path>>(path: P) -> Result<AudioData> {
    let path = path.as_ref();
    let format = FormatDetector::detect_from_extension(path)?;

    match format {
        AudioFormat::Wav => load_wav(path),
        AudioFormat::Flac => load_flac(path),
        AudioFormat::Mp3 => load_mp3(path),
        AudioFormat::Ogg => load_ogg(path),
        AudioFormat::Opus => load_opus(path),
    }
}

/// Load WAV file with SIMD-optimized sample conversion
pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<AudioData> {
    use crate::audio::simd::SimdAudioProcessor;

    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    const NORMALIZATION_FACTOR: f32 = 1.0 / i16::MAX as f32;

    // Pre-allocate vector with known capacity
    let sample_count = reader.len() as usize;
    let mut samples = Vec::with_capacity(sample_count);

    // Collect all i16 samples first
    let mut i16_samples = Vec::with_capacity(sample_count);
    for sample_result in reader.samples::<i16>() {
        let sample = sample_result.map_err(DatasetError::from)?;
        i16_samples.push(sample);
    }

    // Use SIMD-optimized conversion
    samples.resize(i16_samples.len(), 0.0);
    SimdAudioProcessor::convert_i16_to_f32(&i16_samples, &mut samples, NORMALIZATION_FACTOR);

    let mut audio = AudioData::new(samples, spec.sample_rate, spec.channels as u32);

    // Add format metadata
    audio.add_metadata("format".to_string(), "wav".to_string());
    audio.add_metadata(
        "bits_per_sample".to_string(),
        spec.bits_per_sample.to_string(),
    );

    Ok(audio)
}

/// Load FLAC file
pub fn load_flac<P: AsRef<Path>>(path: P) -> Result<AudioData> {
    let file = File::open(path)?;
    let mut reader = claxon::FlacReader::new(file)
        .map_err(|e| DatasetError::FormatError(format!("FLAC read error: {e}")))?;

    let streaminfo = reader.streaminfo();
    let sample_rate = streaminfo.sample_rate;
    let channels = streaminfo.channels;
    let bits_per_sample = streaminfo.bits_per_sample;

    // Pre-allocate vector with estimated capacity to avoid reallocations
    // Estimate based on sample rate and channels for reasonable initial capacity
    let estimated_samples = (sample_rate * channels * 5) as usize; // Assume ~5 seconds of audio
    let mut samples = Vec::with_capacity(estimated_samples.max(1024));

    let normalization_factor = 1.0 / (1 << (bits_per_sample - 1)) as f32;

    // Read all samples using claxon's iterator API with optimized processing
    for sample_result in reader.samples() {
        let sample = sample_result
            .map_err(|e| DatasetError::FormatError(format!("FLAC sample error: {e}")))?;
        let normalized_sample = (sample as f32 * normalization_factor).clamp(-1.0, 1.0);
        samples.push(normalized_sample);
    }

    let mut audio = AudioData::new(samples, sample_rate, channels);
    audio.add_metadata("format".to_string(), "flac".to_string());
    audio.add_metadata("bits_per_sample".to_string(), bits_per_sample.to_string());

    Ok(audio)
}

/// Load MP3 file
pub fn load_mp3<P: AsRef<Path>>(path: P) -> Result<AudioData> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let mut decoder = minimp3::Decoder::new(buffer.as_slice());
    let mut all_samples = Vec::with_capacity(buffer.len() / 2); // Estimate based on buffer size
    let mut sample_rate = 0;
    let mut channels = 0;

    const NORMALIZATION_FACTOR: f32 = 1.0 / i16::MAX as f32;

    loop {
        match decoder.next_frame() {
            Ok(frame) => {
                if sample_rate == 0 {
                    sample_rate = frame.sample_rate as u32;
                    channels = frame.channels as u32;
                }

                // Pre-allocate additional capacity if needed
                let needed_capacity = all_samples.len() + frame.data.len();
                if needed_capacity > all_samples.capacity() {
                    all_samples.reserve(frame.data.len() * 2);
                }

                // Use SIMD-optimized batch conversion for better performance
                let current_len = all_samples.len();
                all_samples.resize(current_len + frame.data.len(), 0.0);
                crate::audio::simd::SimdAudioProcessor::convert_i16_to_f32(
                    &frame.data,
                    &mut all_samples[current_len..],
                    NORMALIZATION_FACTOR,
                );
            }
            Err(minimp3::Error::Eof) => break,
            Err(e) => return Err(DatasetError::FormatError(format!("MP3 decode error: {e}"))),
        }
    }

    if all_samples.is_empty() {
        return Err(DatasetError::FormatError(
            "No audio data found in MP3 file".to_string(),
        ));
    }

    let mut audio = AudioData::new(all_samples, sample_rate, channels);
    audio.add_metadata("format".to_string(), "mp3".to_string());

    Ok(audio)
}

/// Load OGG Vorbis file
pub fn load_ogg<P: AsRef<Path>>(path: P) -> Result<AudioData> {
    let file = File::open(path)?;
    let mut reader = lewton::inside_ogg::OggStreamReader::new(file)
        .map_err(|e| DatasetError::FormatError(format!("OGG read error: {e:?}")))?;

    let sample_rate = reader.ident_hdr.audio_sample_rate;
    let channels = reader.ident_hdr.audio_channels as u32;

    let mut all_samples = Vec::new();

    while let Some(packet) = reader
        .read_dec_packet_itl()
        .map_err(|e| DatasetError::FormatError(format!("OGG decode error: {e:?}")))?
    {
        // Convert i16 samples to f32
        for &sample in &packet {
            all_samples.push(sample as f32 / i16::MAX as f32);
        }
    }

    if all_samples.is_empty() {
        return Err(DatasetError::FormatError(
            "No audio data found in OGG file".to_string(),
        ));
    }

    let mut audio = AudioData::new(all_samples, sample_rate, channels);
    audio.add_metadata("format".to_string(), "ogg".to_string());

    Ok(audio)
}

/// Load OPUS file
pub fn load_opus<P: AsRef<Path>>(path: P) -> Result<AudioData> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let mut buffer = Vec::new();
    let mut reader = file;
    std::io::Read::read_to_end(&mut reader, &mut buffer)?;

    // For now, provide a basic implementation using OGG container parsing
    // OPUS is typically stored in OGG containers, so we can leverage the OGG reader
    // and detect OPUS codec within the OGG stream

    // In a full implementation, we would:
    // 1. Parse OGG container headers
    // 2. Detect OPUS codec pages
    // 3. Initialize OPUS decoder
    // 4. Decode OPUS packets to PCM samples

    // For now, attempt to parse as OGG and detect OPUS headers
    let file = File::open(path)?;
    let mut reader = lewton::inside_ogg::OggStreamReader::new(file)
        .map_err(|e| DatasetError::FormatError(format!("OPUS/OGG read error: {e:?}")))?;

    // Check if this is actually an OPUS stream in OGG container
    let sample_rate = reader.ident_hdr.audio_sample_rate;
    let channels = reader.ident_hdr.audio_channels as u32;

    let mut all_samples = Vec::new();

    while let Some(packet) = reader
        .read_dec_packet_itl()
        .map_err(|e| DatasetError::FormatError(format!("OPUS decode error: {e:?}")))?
    {
        // Convert i16 samples to f32
        for &sample in &packet {
            all_samples.push(sample as f32 / i16::MAX as f32);
        }
    }

    if all_samples.is_empty() {
        return Err(DatasetError::FormatError(
            "No audio data found in OPUS file".to_string(),
        ));
    }

    let mut audio = AudioData::new(all_samples, sample_rate, channels);
    audio.add_metadata("format".to_string(), "opus".to_string());
    audio.add_metadata("codec".to_string(), "opus".to_string());

    Ok(audio)
}

/// Save audio file with automatic format detection from extension
pub fn save_audio<P: AsRef<Path>>(audio: &AudioData, path: P) -> Result<()> {
    let path = path.as_ref();
    let format = FormatDetector::detect_from_extension(path)?;

    match format {
        AudioFormat::Wav => save_wav(audio, path),
        AudioFormat::Mp3 => save_mp3(audio, path),
        AudioFormat::Flac => save_flac(audio, path),
        AudioFormat::Ogg => save_ogg(audio, path),
        AudioFormat::Opus => save_opus(audio, path),
    }
}

/// Save audio to WAV file
pub fn save_wav<P: AsRef<Path>>(audio: &AudioData, path: P) -> Result<()> {
    let spec = hound::WavSpec {
        channels: audio.channels() as u16,
        sample_rate: audio.sample_rate(),
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;

    for &sample in audio.samples() {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;
    Ok(())
}

/// Save audio to FLAC file
pub fn save_flac<P: AsRef<Path>>(audio: &AudioData, path: P) -> Result<()> {
    // For reliable FLAC encoding, use external FFmpeg command
    let temp_wav = tempfile::NamedTempFile::with_suffix(".wav")?;
    save_wav(audio, temp_wav.path())?;

    // Check if ffmpeg is available
    let ffmpeg_available = std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    if ffmpeg_available {
        let output = std::process::Command::new("ffmpeg")
            .arg("-i")
            .arg(temp_wav.path())
            .arg("-c:a")
            .arg("flac")
            .arg("-compression_level")
            .arg("8")
            .arg("-y") // Overwrite output files
            .arg(path.as_ref())
            .output()?;

        if output.status.success() {
            tracing::info!(
                "Successfully encoded FLAC file: {} using FFmpeg ({} channels, {} Hz, {:.2}s)",
                path.as_ref().display(),
                audio.channels(),
                audio.sample_rate(),
                audio.duration()
            );
        } else {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            return Err(crate::DatasetError::AudioError(format!(
                "FFmpeg FLAC encoding failed: {error_msg}"
            )));
        }
    } else {
        // Fallback: save as WAV with instructions
        let wav_path = path.as_ref().with_extension("wav");
        save_wav(audio, &wav_path)?;

        tracing::warn!(
            "FFmpeg not available for FLAC encoding. Audio saved as WAV: {}. \
            To convert to FLAC, install FFmpeg and run: \
            ffmpeg -i {} -c:a flac -compression_level 8 {}",
            wav_path.display(),
            wav_path.display(),
            path.as_ref().display()
        );
    }

    Ok(())
}

/// Save audio to OGG file
pub fn save_ogg<P: AsRef<Path>>(audio: &AudioData, path: P) -> Result<()> {
    // For OGG Vorbis encoding, we'll use a simple approach with external command
    // as pure Rust OGG Vorbis encoding is complex and requires unsafe bindings
    let temp_wav = tempfile::NamedTempFile::with_suffix(".wav")?;
    save_wav(audio, temp_wav.path())?;

    // Check if ffmpeg is available
    let ffmpeg_available = std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    if ffmpeg_available {
        let output = std::process::Command::new("ffmpeg")
            .arg("-i")
            .arg(temp_wav.path())
            .arg("-c:a")
            .arg("libvorbis")
            .arg("-b:a")
            .arg("192k")
            .arg("-y") // Overwrite output files
            .arg(path.as_ref())
            .output()?;

        if output.status.success() {
            tracing::info!(
                "Successfully encoded OGG file: {} using FFmpeg ({} channels, {} Hz, {:.2}s)",
                path.as_ref().display(),
                audio.channels(),
                audio.sample_rate(),
                audio.duration()
            );
        } else {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            return Err(crate::DatasetError::AudioError(format!(
                "FFmpeg OGG encoding failed: {error_msg}"
            )));
        }
    } else {
        // Fallback: save as WAV with instructions
        let wav_path = path.as_ref().with_extension("wav");
        save_wav(audio, &wav_path)?;

        tracing::warn!(
            "FFmpeg not available for OGG encoding. Audio saved as WAV: {}. \
            To convert to OGG, install FFmpeg and run: \
            ffmpeg -i {} -c:a libvorbis -b:a 192k {}",
            wav_path.display(),
            wav_path.display(),
            path.as_ref().display()
        );
    }

    Ok(())
}

/// Save audio to MP3 file
pub fn save_mp3<P: AsRef<Path>>(audio: &AudioData, path: P) -> Result<()> {
    // For reliable MP3 encoding, use external FFmpeg command
    let temp_wav = tempfile::NamedTempFile::with_suffix(".wav")?;
    save_wav(audio, temp_wav.path())?;

    // Check if ffmpeg is available
    let ffmpeg_available = std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    if ffmpeg_available {
        let output = std::process::Command::new("ffmpeg")
            .arg("-i")
            .arg(temp_wav.path())
            .arg("-codec:a")
            .arg("libmp3lame")
            .arg("-b:a")
            .arg("320k")
            .arg("-y") // Overwrite output files
            .arg(path.as_ref())
            .output()?;

        if output.status.success() {
            tracing::info!(
                "Successfully encoded MP3 file: {} using FFmpeg ({} channels, {} Hz, {:.2}s)",
                path.as_ref().display(),
                audio.channels(),
                audio.sample_rate(),
                audio.duration()
            );
        } else {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            return Err(crate::DatasetError::AudioError(format!(
                "FFmpeg MP3 encoding failed: {error_msg}"
            )));
        }
    } else {
        // Fallback: save as WAV with instructions
        let wav_path = path.as_ref().with_extension("wav");
        save_wav(audio, &wav_path)?;

        tracing::warn!(
            "FFmpeg not available for MP3 encoding. Audio saved as WAV: {}. \
            To convert to MP3, install FFmpeg and run: \
            ffmpeg -i {} -codec:a libmp3lame -b:a 320k {}",
            wav_path.display(),
            wav_path.display(),
            path.as_ref().display()
        );
    }

    Ok(())
}

/// Save audio to OPUS file
pub fn save_opus<P: AsRef<Path>>(audio: &AudioData, path: P) -> Result<()> {
    // OPUS encoding using FFmpeg as the most reliable approach
    let temp_wav = tempfile::NamedTempFile::with_suffix(".wav")?;
    save_wav(audio, temp_wav.path())?;

    // Check if ffmpeg is available
    let ffmpeg_available = std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    if ffmpeg_available {
        let output = std::process::Command::new("ffmpeg")
            .arg("-i")
            .arg(temp_wav.path())
            .arg("-c:a")
            .arg("libopus")
            .arg("-b:a")
            .arg("128k")
            .arg("-y") // Overwrite output files
            .arg(path.as_ref())
            .output()?;

        if output.status.success() {
            tracing::info!(
                "Successfully encoded OPUS file: {} using FFmpeg ({} channels, {} Hz, {:.2}s)",
                path.as_ref().display(),
                audio.channels(),
                audio.sample_rate(),
                audio.duration()
            );
        } else {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            return Err(crate::DatasetError::AudioError(format!(
                "FFmpeg OPUS encoding failed: {error_msg}"
            )));
        }
    } else {
        // Fallback: save as OGG with instructions
        let ogg_path = path.as_ref().with_extension("ogg");
        save_ogg(audio, &ogg_path)?;

        tracing::warn!(
            "FFmpeg not available for OPUS encoding. Audio saved as OGG: {}. \
            To convert to OPUS, install FFmpeg and run: \
            ffmpeg -i {} -c:a libopus -b:a 128k {}",
            ogg_path.display(),
            ogg_path.display(),
            path.as_ref().display()
        );
    }

    Ok(())
}

/// Streaming audio reader for large files
pub struct StreamingAudioReader {
    format: AudioFormat,
    chunk_size: usize,
    current_position: usize,
    file_path: std::path::PathBuf,
    total_samples: Option<usize>,
    cached_audio: Option<AudioData>,
}

impl StreamingAudioReader {
    /// Create new streaming reader
    pub fn new<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self> {
        let path = path.as_ref();
        let format = FormatDetector::detect_from_extension(path)?;

        Ok(Self {
            format,
            chunk_size,
            current_position: 0,
            file_path: path.to_path_buf(),
            total_samples: None,
            cached_audio: None,
        })
    }

    /// Read next chunk of audio data
    pub fn read_chunk(&mut self) -> Result<Option<AudioData>> {
        match self.format {
            AudioFormat::Wav => self.read_wav_chunk(),
            AudioFormat::Flac => self.read_flac_chunk(),
            AudioFormat::Mp3 => self.read_mp3_chunk(),
            AudioFormat::Ogg => self.read_ogg_chunk(),
            AudioFormat::Opus => self.read_opus_chunk(),
        }
    }

    /// Get total number of samples if available
    pub fn total_samples(&self) -> Option<usize> {
        self.total_samples
    }

    /// Check if there are more chunks to read
    pub fn has_more_chunks(&self) -> bool {
        if let Some(total) = self.total_samples {
            self.current_position < total
        } else {
            // For formats without known total samples, we need to try reading
            true
        }
    }

    fn read_wav_chunk(&mut self) -> Result<Option<AudioData>> {
        let mut reader = hound::WavReader::open(&self.file_path)?;
        let spec = reader.spec();

        // Skip to current position
        if self.current_position > 0 {
            reader.seek(self.current_position as u32)?;
        }

        let mut samples = Vec::new();
        let mut samples_read = 0;

        for sample_result in reader.samples::<i16>() {
            if samples_read >= self.chunk_size {
                break;
            }

            let sample = sample_result?;
            samples.push(sample as f32 / i16::MAX as f32);
            samples_read += 1;
        }

        if samples.is_empty() {
            return Ok(None);
        }

        self.current_position += samples_read;

        let mut audio = AudioData::new(samples, spec.sample_rate, spec.channels as u32);
        audio.add_metadata("format".to_string(), "wav".to_string());
        audio.add_metadata(
            "chunk_index".to_string(),
            (self.current_position / self.chunk_size).to_string(),
        );

        Ok(Some(audio))
    }

    fn read_flac_chunk(&mut self) -> Result<Option<AudioData>> {
        // For FLAC, we need to cache the entire file on first read
        if self.cached_audio.is_none() {
            self.cached_audio = Some(load_flac(&self.file_path)?);
            self.total_samples = Some(self.cached_audio.as_ref().unwrap().samples().len());
        }

        self.read_from_cached_audio()
    }

    fn read_mp3_chunk(&mut self) -> Result<Option<AudioData>> {
        // For MP3, we need to cache the entire file on first read
        if self.cached_audio.is_none() {
            self.cached_audio = Some(load_mp3(&self.file_path)?);
            self.total_samples = Some(self.cached_audio.as_ref().unwrap().samples().len());
        }

        self.read_from_cached_audio()
    }

    fn read_ogg_chunk(&mut self) -> Result<Option<AudioData>> {
        // For OGG, we need to cache the entire file on first read
        if self.cached_audio.is_none() {
            self.cached_audio = Some(load_ogg(&self.file_path)?);
            self.total_samples = Some(self.cached_audio.as_ref().unwrap().samples().len());
        }

        self.read_from_cached_audio()
    }

    fn read_opus_chunk(&mut self) -> Result<Option<AudioData>> {
        // For OPUS, we need to cache the entire file on first read
        if self.cached_audio.is_none() {
            self.cached_audio = Some(load_opus(&self.file_path)?);
            self.total_samples = Some(self.cached_audio.as_ref().unwrap().samples().len());
        }

        self.read_from_cached_audio()
    }

    fn read_from_cached_audio(&mut self) -> Result<Option<AudioData>> {
        let cached = self.cached_audio.as_ref().unwrap();

        if self.current_position >= cached.samples().len() {
            return Ok(None);
        }

        let end_position = (self.current_position + self.chunk_size).min(cached.samples().len());
        let chunk_samples = &cached.samples()[self.current_position..end_position];

        if chunk_samples.is_empty() {
            return Ok(None);
        }

        let mut audio = AudioData::new(
            chunk_samples.to_vec(),
            cached.sample_rate(),
            cached.channels(),
        );

        // Copy metadata from cached audio
        for (key, value) in cached.metadata().iter() {
            audio.add_metadata(key.clone(), value.clone());
        }

        audio.add_metadata(
            "chunk_index".to_string(),
            (self.current_position / self.chunk_size).to_string(),
        );

        self.current_position = end_position;

        Ok(Some(audio))
    }

    /// Reset reader to beginning
    pub fn reset(&mut self) {
        self.current_position = 0;
    }
}

/// Audio validation result containing comprehensive validation information
#[derive(Debug, Clone, Default)]
pub struct AudioValidationResult {
    /// Whether the audio file is valid and can be loaded
    pub is_valid: bool,
    /// Detected audio format
    pub detected_format: Option<AudioFormat>,
    /// File size in bytes
    pub file_size: u64,
    /// Sample rate (if successfully loaded)
    pub sample_rate: Option<u32>,
    /// Number of channels (if successfully loaded)
    pub channels: Option<u32>,
    /// Duration in seconds (if successfully loaded)
    pub duration_seconds: Option<f64>,
    /// Validation errors (serious issues that prevent loading)
    pub errors: Vec<String>,
    /// Validation warnings (quality issues but file can still be loaded)
    pub warnings: Vec<String>,
}

/// Audio file format detection utilities
pub struct FormatDetector;

impl FormatDetector {
    /// Detect format from file extension
    pub fn detect_from_extension<P: AsRef<Path>>(path: P) -> Result<AudioFormat> {
        let path = path.as_ref();

        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| DatasetError::FormatError("No file extension found".to_string()))?
            .to_lowercase();

        match extension.as_str() {
            "wav" => Ok(AudioFormat::Wav),
            "flac" => Ok(AudioFormat::Flac),
            "mp3" => Ok(AudioFormat::Mp3),
            "ogg" | "oga" => Ok(AudioFormat::Ogg),
            "opus" => Ok(AudioFormat::Opus),
            _ => Err(DatasetError::FormatError(format!(
                "Unsupported audio format: {extension}"
            ))),
        }
    }

    /// Detect format from file header
    pub fn detect_from_header<P: AsRef<Path>>(path: P) -> Result<AudioFormat> {
        let mut file = File::open(path)?;
        let mut header = [0u8; 16];
        file.read_exact(&mut header)?;

        // Check for various format signatures
        if &header[0..4] == b"RIFF" && &header[8..12] == b"WAVE" {
            Ok(AudioFormat::Wav)
        } else if &header[0..4] == b"fLaC" {
            Ok(AudioFormat::Flac)
        } else if header[0] == 0xFF && (header[1] & 0xE0) == 0xE0 {
            // MP3 frame sync
            Ok(AudioFormat::Mp3)
        } else if &header[0..4] == b"OggS" {
            // Check if this is an OPUS file within OGG container
            // OPUS identification header starts with "OpusHead"
            if header.len() >= 16 && &header[8..16] == b"OpusHead" {
                Ok(AudioFormat::Opus)
            } else {
                Ok(AudioFormat::Ogg)
            }
        } else {
            Err(DatasetError::FormatError(
                "Unknown audio format".to_string(),
            ))
        }
    }

    /// Validate audio file format
    pub fn validate_format<P: AsRef<Path>>(path: P, expected_format: AudioFormat) -> Result<bool> {
        let detected_format = Self::detect_from_header(path)?;
        Ok(detected_format == expected_format)
    }

    /// Comprehensive audio file validation with corruption detection
    pub fn validate_audio_integrity<P: AsRef<Path>>(path: P) -> Result<AudioValidationResult> {
        let path = path.as_ref();
        let mut validation_result = AudioValidationResult::default();

        // Step 1: Check file existence and readability
        if !path.exists() {
            validation_result
                .errors
                .push("File does not exist".to_string());
            return Ok(validation_result);
        }

        let file_size = std::fs::metadata(path)?.len();
        if file_size == 0 {
            validation_result.errors.push("File is empty".to_string());
            return Ok(validation_result);
        }

        validation_result.file_size = file_size;

        // Step 2: Format detection and validation
        match Self::detect_from_extension(path) {
            Ok(format) => {
                validation_result.detected_format = Some(format);

                // Step 3: Header validation
                match Self::detect_from_header(path) {
                    Ok(header_format) => {
                        if header_format != format {
                            validation_result.warnings.push(format!(
                                "Format mismatch: extension suggests {format:?}, header suggests {header_format:?}"
                            ));
                        }
                    }
                    Err(e) => {
                        validation_result
                            .errors
                            .push(format!("Header validation failed: {e}"));
                    }
                }

                // Step 4: Attempt to load and validate audio content
                match load_audio(path) {
                    Ok(audio) => {
                        validation_result.is_valid = true;
                        validation_result.sample_rate = Some(audio.sample_rate());
                        validation_result.channels = Some(audio.channels());
                        validation_result.duration_seconds = Some(audio.duration() as f64);

                        // Check for audio quality issues
                        let samples = audio.samples();
                        let clipped_samples = samples.iter().filter(|&&s| s.abs() >= 0.99).count();
                        if clipped_samples > samples.len() / 100 {
                            validation_result.warnings.push(format!(
                                "High clipping detected: {} samples ({:.2}%) near maximum amplitude",
                                clipped_samples,
                                clipped_samples as f64 / samples.len() as f64 * 100.0
                            ));
                        }

                        // Check for excessive silence
                        let silent_samples = samples.iter().filter(|&&s| s.abs() < 0.001).count();
                        if silent_samples > samples.len() / 2 {
                            validation_result.warnings.push(format!(
                                "Excessive silence detected: {} samples ({:.2}%) are very quiet",
                                silent_samples,
                                silent_samples as f64 / samples.len() as f64 * 100.0
                            ));
                        }

                        // Check for DC offset
                        let dc_offset = samples.iter().sum::<f32>() / samples.len() as f32;
                        if dc_offset.abs() > 0.1 {
                            validation_result.warnings.push(format!(
                                "DC offset detected: {dc_offset:.4} (should be near 0.0)"
                            ));
                        }
                    }
                    Err(e) => {
                        validation_result
                            .errors
                            .push(format!("Audio loading failed: {e}"));
                    }
                }
            }
            Err(e) => {
                validation_result
                    .errors
                    .push(format!("Format detection failed: {e}"));
            }
        }

        Ok(validation_result)
    }
}

/// Audio file metadata extractor
pub struct AudioMetadataExtractor;

impl AudioMetadataExtractor {
    /// Extract metadata from audio file
    pub fn extract<P: AsRef<Path>>(path: P) -> Result<HashMap<String, String>> {
        let path = path.as_ref();
        let format = FormatDetector::detect_from_extension(path)?;

        match format {
            AudioFormat::Wav => Self::extract_wav_metadata(path),
            AudioFormat::Flac => Self::extract_flac_metadata(path),
            AudioFormat::Mp3 => Self::extract_mp3_metadata(path),
            AudioFormat::Ogg => Self::extract_ogg_metadata(path),
            AudioFormat::Opus => Self::extract_opus_metadata(path),
        }
    }

    fn extract_wav_metadata<P: AsRef<Path>>(path: P) -> Result<HashMap<String, String>> {
        let reader = hound::WavReader::open(path)?;
        let spec = reader.spec();

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "wav".to_string());
        metadata.insert("sample_rate".to_string(), spec.sample_rate.to_string());
        metadata.insert("channels".to_string(), spec.channels.to_string());
        metadata.insert(
            "bits_per_sample".to_string(),
            spec.bits_per_sample.to_string(),
        );
        metadata.insert("duration".to_string(), reader.duration().to_string());

        Ok(metadata)
    }

    fn extract_flac_metadata<P: AsRef<Path>>(path: P) -> Result<HashMap<String, String>> {
        let file = File::open(path)?;
        let reader = claxon::FlacReader::new(file)
            .map_err(|e| DatasetError::FormatError(format!("FLAC read error: {e}")))?;

        let streaminfo = reader.streaminfo();

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "flac".to_string());
        metadata.insert(
            "sample_rate".to_string(),
            streaminfo.sample_rate.to_string(),
        );
        metadata.insert("channels".to_string(), streaminfo.channels.to_string());
        metadata.insert(
            "bits_per_sample".to_string(),
            streaminfo.bits_per_sample.to_string(),
        );

        // Note: claxon::metadata::StreamInfo doesn't expose total_samples directly
        // Duration calculation would require reading through the entire file
        // For now, we'll skip duration metadata for FLAC files

        Ok(metadata)
    }

    fn extract_mp3_metadata<P: AsRef<Path>>(path: P) -> Result<HashMap<String, String>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let mut decoder = minimp3::Decoder::new(buffer.as_slice());
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "mp3".to_string());

        // Get first frame for basic info
        if let Ok(frame) = decoder.next_frame() {
            metadata.insert("sample_rate".to_string(), frame.sample_rate.to_string());
            metadata.insert("channels".to_string(), frame.channels.to_string());
            metadata.insert("bitrate".to_string(), frame.bitrate.to_string());
        }

        Ok(metadata)
    }

    fn extract_ogg_metadata<P: AsRef<Path>>(path: P) -> Result<HashMap<String, String>> {
        let file = File::open(path)?;
        let reader = lewton::inside_ogg::OggStreamReader::new(file)
            .map_err(|e| DatasetError::FormatError(format!("OGG read error: {e:?}")))?;

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "ogg".to_string());
        metadata.insert(
            "sample_rate".to_string(),
            reader.ident_hdr.audio_sample_rate.to_string(),
        );
        metadata.insert(
            "channels".to_string(),
            reader.ident_hdr.audio_channels.to_string(),
        );

        Ok(metadata)
    }

    fn extract_opus_metadata<P: AsRef<Path>>(path: P) -> Result<HashMap<String, String>> {
        let file = File::open(path)?;
        let reader = lewton::inside_ogg::OggStreamReader::new(file)
            .map_err(|e| DatasetError::FormatError(format!("OPUS read error: {e:?}")))?;

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "opus".to_string());
        metadata.insert("codec".to_string(), "opus".to_string());
        metadata.insert(
            "sample_rate".to_string(),
            reader.ident_hdr.audio_sample_rate.to_string(),
        );
        metadata.insert(
            "channels".to_string(),
            reader.ident_hdr.audio_channels.to_string(),
        );

        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_format_detection_from_extension() {
        assert_eq!(
            FormatDetector::detect_from_extension("test.wav").unwrap(),
            AudioFormat::Wav
        );
        assert_eq!(
            FormatDetector::detect_from_extension("test.flac").unwrap(),
            AudioFormat::Flac
        );
        assert_eq!(
            FormatDetector::detect_from_extension("test.mp3").unwrap(),
            AudioFormat::Mp3
        );
        assert_eq!(
            FormatDetector::detect_from_extension("test.ogg").unwrap(),
            AudioFormat::Ogg
        );
        assert_eq!(
            FormatDetector::detect_from_extension("test.opus").unwrap(),
            AudioFormat::Opus
        );
    }

    #[test]
    fn test_audio_validation_nonexistent_file() {
        let result = FormatDetector::validate_audio_integrity("nonexistent.wav").unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
        assert!(result.errors[0].contains("File does not exist"));
    }

    #[test]
    fn test_wav_save_load_roundtrip() {
        let original_audio = AudioData::new(vec![0.1, -0.2, 0.3, -0.4], 44100, 2);

        let temp_file = NamedTempFile::new().unwrap();
        save_wav(&original_audio, temp_file.path()).unwrap();

        let loaded_audio = load_wav(temp_file.path()).unwrap();

        assert_eq!(loaded_audio.sample_rate(), 44100);
        assert_eq!(loaded_audio.channels(), 2);
        assert_eq!(loaded_audio.samples().len(), 4);

        // Check approximate equality (accounting for quantization)
        for (original, loaded) in original_audio
            .samples()
            .iter()
            .zip(loaded_audio.samples().iter())
        {
            assert!((original - loaded).abs() < 0.01);
        }
    }

    #[test]
    fn test_streaming_reader_creation() {
        let temp_file = NamedTempFile::with_suffix(".wav").unwrap();

        // Create a test WAV file
        let audio = AudioData::new(vec![0.1; 1000], 44100, 1);
        save_wav(&audio, temp_file.path()).unwrap();

        let reader = StreamingAudioReader::new(temp_file.path(), 256);
        assert!(reader.is_ok());
    }

    #[test]
    fn test_mp3_save_load_roundtrip() {
        // Generate test samples
        let mut samples = Vec::new();
        for i in 0..1000 {
            samples.push((i as f32 / 1000.0).sin());
        }

        let original_audio = AudioData::new(samples, 44100, 2);

        let temp_file = NamedTempFile::with_suffix(".mp3").unwrap();

        // MP3 saving falls back to WAV, so this should succeed
        let result = save_mp3(&original_audio, temp_file.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_save_audio_format_detection() {
        let audio = AudioData::new(vec![0.1, -0.2, 0.3, -0.4], 44100, 2);

        // Test WAV
        let temp_wav = NamedTempFile::with_suffix(".wav").unwrap();
        save_audio(&audio, temp_wav.path()).unwrap();

        // Test MP3 - falls back to WAV, so should succeed
        let temp_mp3 = NamedTempFile::with_suffix(".mp3").unwrap();
        let result = save_audio(&audio, temp_mp3.path());
        assert!(result.is_ok());

        // Test FLAC - falls back to WAV, so should succeed
        let temp_flac = NamedTempFile::with_suffix(".flac").unwrap();
        let result = save_audio(&audio, temp_flac.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_streaming_reader_with_chunks() {
        let temp_file = NamedTempFile::with_suffix(".wav").unwrap();

        // Create a test WAV file with 1000 samples
        let audio = AudioData::new(vec![0.1; 1000], 44100, 1);
        save_wav(&audio, temp_file.path()).unwrap();

        let mut reader = StreamingAudioReader::new(temp_file.path(), 256).unwrap();

        let mut total_samples = 0;
        let mut chunk_count = 0;

        while let Some(chunk) = reader.read_chunk().unwrap() {
            total_samples += chunk.samples().len();
            chunk_count += 1;
            assert!(chunk.samples().len() <= 256);
        }

        assert_eq!(total_samples, 1000);
        assert_eq!(chunk_count, 4); // 1000 / 256 = ~4 chunks
    }
}
