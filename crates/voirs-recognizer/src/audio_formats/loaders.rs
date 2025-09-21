//! Audio format loaders for various file types
//!
//! Provides specialized loaders for WAV, FLAC, MP3, and OGG formats
//! with unified configuration and error handling.

use super::resampling::{mix_to_mono, normalize_audio, remove_dc_offset};
use super::{AudioLoadConfig, AudioResampler, ResamplingQuality};
use crate::RecognitionError;
use std::io::{Cursor, Read};
use std::path::Path;
use symphonia::core::audio::AudioBufferRef;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use voirs_sdk::AudioBuffer;

/// WAV file loader using hound crate
pub struct WavLoader {
    config: AudioLoadConfig,
}

impl WavLoader {
    /// Create a new WAV loader with the given configuration
    #[must_use]
    pub fn new(config: AudioLoadConfig) -> Self {
        Self { config }
    }

    /// Load WAV audio from a file path
    pub fn load_from_path<P: AsRef<Path>>(&self, path: P) -> Result<AudioBuffer, RecognitionError> {
        let reader = hound::WavReader::open(path).map_err(|e| {
            RecognitionError::InvalidFormat(format!("Failed to open WAV file: {e}"))
        })?;

        self.load_from_reader(reader)
    }

    pub fn load_from_bytes(&self, data: &[u8]) -> Result<AudioBuffer, RecognitionError> {
        let cursor = Cursor::new(data);
        let reader = hound::WavReader::new(cursor).map_err(|e| {
            RecognitionError::InvalidFormat(format!("Failed to parse WAV data: {e}"))
        })?;

        self.load_from_reader(reader)
    }

    fn load_from_reader<R: Read>(
        &self,
        mut reader: hound::WavReader<R>,
    ) -> Result<AudioBuffer, RecognitionError> {
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let channels = u32::from(spec.channels);
        let bits_per_sample = spec.bits_per_sample;

        // Read samples based on bit depth
        let samples: Vec<f32> = match bits_per_sample {
            8 => reader
                .samples::<i8>()
                .map(|s| s.map(|sample| f32::from(sample) / 128.0))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    RecognitionError::InvalidFormat(format!("Error reading 8-bit samples: {e}"))
                })?,
            16 => reader
                .samples::<i16>()
                .map(|s| s.map(|sample| f32::from(sample) / 32768.0))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    RecognitionError::InvalidFormat(format!("Error reading 16-bit samples: {e}"))
                })?,
            24 => reader
                .samples::<i32>()
                .map(|s| s.map(|sample| sample as f32 / 8_388_608.0))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    RecognitionError::InvalidFormat(format!("Error reading 24-bit samples: {e}"))
                })?,
            32 => {
                if spec.sample_format == hound::SampleFormat::Float {
                    reader
                        .samples::<f32>()
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| {
                            RecognitionError::InvalidFormat(format!(
                                "Error reading 32-bit float samples: {e}"
                            ))
                        })?
                } else {
                    reader
                        .samples::<i32>()
                        .map(|s| s.map(|sample| sample as f32 / 2_147_483_648.0))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| {
                            RecognitionError::InvalidFormat(format!(
                                "Error reading 32-bit int samples: {e}"
                            ))
                        })?
                }
            }
            _ => {
                return Err(RecognitionError::UnsupportedFormat(format!(
                    "Unsupported bit depth: {bits_per_sample} bits"
                )));
            }
        };

        self.process_samples(samples, sample_rate, channels)
    }

    fn process_samples(
        &self,
        mut samples: Vec<f32>,
        sample_rate: u32,
        channels: u32,
    ) -> Result<AudioBuffer, RecognitionError> {
        // Apply DC offset removal if requested
        if self.config.remove_dc {
            remove_dc_offset(&mut samples);
        }

        // Convert to mono if requested
        if self.config.force_mono && channels > 1 {
            samples = mix_to_mono(&samples, channels);
        }

        // Normalize if requested
        if self.config.normalize {
            normalize_audio(&mut samples);
        }

        let final_channels = if self.config.force_mono { 1 } else { channels };
        let mut audio = AudioBuffer::new(samples, sample_rate, final_channels);

        // Resample if requested
        if let Some(target_rate) = self.config.target_sample_rate {
            if target_rate != sample_rate {
                let resampler = AudioResampler::new(ResamplingQuality::High);
                audio = resampler.resample(&audio, target_rate)?;
            }
        }

        // Apply duration limit if specified
        if let Some(max_duration) = self.config.max_duration_seconds {
            let max_samples =
                (audio.sample_rate() as f32 * max_duration * audio.channels() as f32) as usize;
            if audio.samples().len() > max_samples {
                let limited_samples = audio.samples()[..max_samples].to_vec();
                audio = AudioBuffer::new(limited_samples, audio.sample_rate(), audio.channels());
            }
        }

        Ok(audio)
    }
}

/// FLAC file loader using claxon crate
pub struct FlacLoader {
    config: AudioLoadConfig,
}

impl FlacLoader {
    #[must_use]
    pub fn new(config: AudioLoadConfig) -> Self {
        Self { config }
    }

    pub fn load_from_path<P: AsRef<Path>>(&self, path: P) -> Result<AudioBuffer, RecognitionError> {
        let reader = claxon::FlacReader::open(path).map_err(|e| {
            RecognitionError::InvalidFormat(format!("Failed to open FLAC file: {e}"))
        })?;

        self.load_from_reader(reader)
    }

    pub fn load_from_bytes(&self, data: &[u8]) -> Result<AudioBuffer, RecognitionError> {
        let cursor = Cursor::new(data);
        let reader = claxon::FlacReader::new(cursor).map_err(|e| {
            RecognitionError::InvalidFormat(format!("Failed to parse FLAC data: {e}"))
        })?;

        self.load_from_reader(reader)
    }

    fn load_from_reader<R: Read>(
        &self,
        mut reader: claxon::FlacReader<R>,
    ) -> Result<AudioBuffer, RecognitionError> {
        let info = reader.streaminfo();
        let sample_rate = info.sample_rate;
        let channels = info.channels;
        let bits_per_sample = info.bits_per_sample;

        // Read all samples
        let mut samples = Vec::new();
        for sample in reader.samples() {
            let sample = sample.map_err(|e| {
                RecognitionError::InvalidFormat(format!("Error reading FLAC samples: {e}"))
            })?;

            // Convert to f32 based on bit depth
            let normalized = match bits_per_sample {
                8 => (sample as f32 - 128.0) / 128.0,
                16 => sample as f32 / 32768.0,
                24 => sample as f32 / 8_388_608.0,
                32 => sample as f32 / 2_147_483_648.0,
                _ => {
                    return Err(RecognitionError::UnsupportedFormat(format!(
                        "Unsupported FLAC bit depth: {bits_per_sample} bits"
                    )));
                }
            };

            samples.push(normalized);
        }

        self.process_samples(samples, sample_rate, channels)
    }

    fn process_samples(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
        channels: u32,
    ) -> Result<AudioBuffer, RecognitionError> {
        let wav_loader = WavLoader::new(self.config.clone());
        wav_loader.process_samples(samples, sample_rate, channels)
    }
}

/// MP3 file loader using minimp3 crate
pub struct Mp3Loader {
    config: AudioLoadConfig,
}

impl Mp3Loader {
    #[must_use]
    pub fn new(config: AudioLoadConfig) -> Self {
        Self { config }
    }

    pub fn load_from_path<P: AsRef<Path>>(&self, path: P) -> Result<AudioBuffer, RecognitionError> {
        let data = std::fs::read(path).map_err(|e| {
            RecognitionError::InvalidFormat(format!("Failed to read MP3 file: {e}"))
        })?;

        self.load_from_bytes(&data)
    }

    pub fn load_from_bytes(&self, data: &[u8]) -> Result<AudioBuffer, RecognitionError> {
        let mut decoder = minimp3::Decoder::new(data);

        let mut all_samples = Vec::new();
        let mut sample_rate = 0;
        let mut channels = 0;

        loop {
            match decoder.next_frame() {
                Ok(frame) => {
                    sample_rate = frame.sample_rate as u32;
                    channels = frame.channels as u32;

                    // Convert i16 samples to f32
                    for &sample in &frame.data {
                        all_samples.push(f32::from(sample) / 32768.0);
                    }
                }
                Err(minimp3::Error::Eof) => break,
                Err(e) => {
                    return Err(RecognitionError::InvalidFormat(format!(
                        "Error decoding MP3: {e}"
                    )));
                }
            }
        }

        if all_samples.is_empty() {
            return Err(RecognitionError::InvalidFormat(
                "Empty MP3 file".to_string(),
            ));
        }

        self.process_samples(all_samples, sample_rate, channels)
    }

    fn process_samples(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
        channels: u32,
    ) -> Result<AudioBuffer, RecognitionError> {
        let wav_loader = WavLoader::new(self.config.clone());
        wav_loader.process_samples(samples, sample_rate, channels)
    }
}

/// OGG Vorbis file loader using lewton crate
pub struct OggLoader {
    config: AudioLoadConfig,
}

impl OggLoader {
    #[must_use]
    pub fn new(config: AudioLoadConfig) -> Self {
        Self { config }
    }

    pub fn load_from_path<P: AsRef<Path>>(&self, path: P) -> Result<AudioBuffer, RecognitionError> {
        let file = std::fs::File::open(path).map_err(|e| {
            RecognitionError::InvalidFormat(format!("Failed to open OGG file: {e}"))
        })?;

        let reader = lewton::inside_ogg::OggStreamReader::new(file).map_err(|e| {
            RecognitionError::InvalidFormat(format!("Failed to parse OGG file: {e}"))
        })?;

        self.load_from_reader(reader)
    }

    pub fn load_from_bytes(&self, data: &[u8]) -> Result<AudioBuffer, RecognitionError> {
        let cursor = Cursor::new(data);
        let reader = lewton::inside_ogg::OggStreamReader::new(cursor).map_err(|e| {
            RecognitionError::InvalidFormat(format!("Failed to parse OGG data: {e}"))
        })?;

        self.load_from_reader(reader)
    }

    fn load_from_reader<R: Read + std::io::Seek>(
        &self,
        mut reader: lewton::inside_ogg::OggStreamReader<R>,
    ) -> Result<AudioBuffer, RecognitionError> {
        let sample_rate = reader.ident_hdr.audio_sample_rate;
        let channels = u32::from(reader.ident_hdr.audio_channels);

        let mut all_samples = Vec::new();

        while let Some(packet) = reader
            .read_dec_packet_generic::<Vec<Vec<f32>>>()
            .map_err(|e| {
                RecognitionError::InvalidFormat(format!("Error reading OGG packet: {e}"))
            })?
        {
            // Interleave channels
            if packet.is_empty() {
                continue;
            }

            let samples_per_channel = packet[0].len();
            for i in 0..samples_per_channel {
                for channel in &packet {
                    if i < channel.len() {
                        all_samples.push(channel[i]);
                    }
                }
            }
        }

        if all_samples.is_empty() {
            return Err(RecognitionError::InvalidFormat(
                "Empty OGG file".to_string(),
            ));
        }

        self.process_samples(all_samples, sample_rate, channels)
    }

    fn process_samples(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
        channels: u32,
    ) -> Result<AudioBuffer, RecognitionError> {
        let wav_loader = WavLoader::new(self.config.clone());
        wav_loader.process_samples(samples, sample_rate, channels)
    }
}

/// M4A/AAC file loader using Symphonia audio decoding
pub struct M4aLoader {
    config: AudioLoadConfig,
}

impl M4aLoader {
    #[must_use]
    pub fn new(config: AudioLoadConfig) -> Self {
        Self { config }
    }

    pub fn load_from_path<P: AsRef<Path>>(&self, path: P) -> Result<AudioBuffer, RecognitionError> {
        let data = std::fs::read(path).map_err(|e| RecognitionError::AudioProcessingError {
            message: format!("Failed to read M4A file: {e}"),
            source: Some(Box::new(e)),
        })?;

        self.load_from_bytes(&data)
    }

    pub fn load_from_bytes(&self, data: &[u8]) -> Result<AudioBuffer, RecognitionError> {
        let data_vec = data.to_vec(); // Copy data to avoid lifetime issues
        let cursor = Cursor::new(data_vec);
        let media_source = MediaSourceStream::new(Box::new(cursor), Default::default());

        // Create a probe hint to help format detection
        let mut hint = Hint::new();
        hint.with_extension("m4a");

        // Probe the media source
        let format_opts = FormatOptions::default();
        let metadata_opts = MetadataOptions::default();
        let decoder_opts = DecoderOptions::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, media_source, &format_opts, &metadata_opts)
            .map_err(|e| {
                RecognitionError::InvalidFormat(format!("Failed to probe M4A file: {e}"))
            })?;

        let mut format = probed.format;

        // Find the default audio track
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| {
                RecognitionError::InvalidFormat("No audio track found in M4A file".to_string())
            })?;

        let track_id = track.id;

        // Create a decoder for the track
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &decoder_opts)
            .map_err(|e| {
                RecognitionError::InvalidFormat(format!("Failed to create M4A decoder: {e}"))
            })?;

        // Decode the audio samples
        let mut audio_samples = Vec::new();
        let mut sample_rate = 44100; // Default sample rate
        let mut channels = 1; // Default to mono

        // Decode audio packets
        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::ResetRequired) => {
                    // The track list has been changed. Re-examine it and create a new set of decoders,
                    // then restart the decode loop. This is an advanced feature, so for now just break.
                    break;
                }
                Err(SymphoniaError::IoError(ref err))
                    if err.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    // End of stream
                    break;
                }
                Err(err) => {
                    return Err(RecognitionError::InvalidFormat(format!(
                        "Error reading M4A packet: {err}"
                    )));
                }
            };

            // Only process packets for our selected track
            if packet.track_id() != track_id {
                continue;
            }

            // Decode the packet into audio samples
            match decoder.decode(&packet) {
                Ok(audio_buf) => {
                    // Update sample rate and channel info from the first decoded buffer
                    if audio_samples.is_empty() {
                        sample_rate = audio_buf.spec().rate;
                        channels = audio_buf.spec().channels.count() as u32;
                    }

                    // Convert the audio buffer to f32 samples
                    self.extract_samples_from_audio_buffer(audio_buf, &mut audio_samples)?;
                }
                Err(SymphoniaError::IoError(ref err))
                    if err.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    // End of stream
                    break;
                }
                Err(SymphoniaError::DecodeError(ref err)) => {
                    // Decode error, skip this packet
                    tracing::warn!("Decode error in M4A packet: {}", err);
                    {}
                }
                Err(err) => {
                    return Err(RecognitionError::InvalidFormat(format!(
                        "Error decoding M4A packet: {err}"
                    )));
                }
            }
        }

        if audio_samples.is_empty() {
            return Err(RecognitionError::InvalidFormat(
                "No audio data found in M4A file".to_string(),
            ));
        }

        self.process_samples(audio_samples, sample_rate, channels)
    }

    /// Extract samples from symphonia AudioBufferRef and convert to f32
    fn extract_samples_from_audio_buffer(
        &self,
        audio_buf: AudioBufferRef,
        samples: &mut Vec<f32>,
    ) -> Result<(), RecognitionError> {
        match audio_buf {
            AudioBufferRef::U8(buf) => {
                for plane in buf.planes().planes() {
                    for &sample in *plane {
                        samples.push((f32::from(sample) - 128.0) / 128.0);
                    }
                }
            }
            AudioBufferRef::U16(buf) => {
                for plane in buf.planes().planes() {
                    for &sample in *plane {
                        samples.push((f32::from(sample) - 32768.0) / 32768.0);
                    }
                }
            }
            AudioBufferRef::U24(buf) => {
                for plane in buf.planes().planes() {
                    for &sample in *plane {
                        samples.push((sample.inner() as f32 - 8_388_608.0) / 8_388_608.0);
                    }
                }
            }
            AudioBufferRef::U32(buf) => {
                for plane in buf.planes().planes() {
                    for &sample in *plane {
                        samples.push((sample as f64 - 2_147_483_648.0) as f32 / 2_147_483_648.0);
                    }
                }
            }
            AudioBufferRef::S8(buf) => {
                for plane in buf.planes().planes() {
                    for &sample in *plane {
                        samples.push(f32::from(sample) / 128.0);
                    }
                }
            }
            AudioBufferRef::S16(buf) => {
                for plane in buf.planes().planes() {
                    for &sample in *plane {
                        samples.push(f32::from(sample) / 32768.0);
                    }
                }
            }
            AudioBufferRef::S24(buf) => {
                for plane in buf.planes().planes() {
                    for &sample in *plane {
                        samples.push(sample.inner() as f32 / 8_388_608.0);
                    }
                }
            }
            AudioBufferRef::S32(buf) => {
                for plane in buf.planes().planes() {
                    for &sample in *plane {
                        samples.push(sample as f32 / 2_147_483_648.0);
                    }
                }
            }
            AudioBufferRef::F32(buf) => {
                for plane in buf.planes().planes() {
                    samples.extend_from_slice(*plane);
                }
            }
            AudioBufferRef::F64(buf) => {
                for plane in buf.planes().planes() {
                    for &sample in *plane {
                        samples.push(sample as f32);
                    }
                }
            }
        }
        Ok(())
    }

    fn process_samples(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
        channels: u32,
    ) -> Result<AudioBuffer, RecognitionError> {
        let wav_loader = WavLoader::new(self.config.clone());
        wav_loader.process_samples(samples, sample_rate, channels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_loader_creation() {
        let config = AudioLoadConfig::default();
        let loader = WavLoader::new(config);
        assert_eq!(loader.config.target_sample_rate, Some(16000));
    }

    #[test]
    fn test_flac_loader_creation() {
        let config = AudioLoadConfig {
            target_sample_rate: Some(22050),
            force_mono: false,
            ..Default::default()
        };
        let loader = FlacLoader::new(config);
        assert_eq!(loader.config.target_sample_rate, Some(22050));
        assert!(!loader.config.force_mono);
    }

    #[test]
    fn test_mp3_loader_creation() {
        let config = AudioLoadConfig::default();
        let loader = Mp3Loader::new(config);
        assert!(loader.config.normalize);
    }

    #[test]
    fn test_ogg_loader_creation() {
        let config = AudioLoadConfig::default();
        let loader = OggLoader::new(config);
        assert!(loader.config.remove_dc);
    }

    #[test]
    fn test_m4a_loader_creation() {
        let config = AudioLoadConfig::default();
        let loader = M4aLoader::new(config);
        assert_eq!(loader.config.target_sample_rate, Some(16000));
    }

    // Note: Integration tests with actual audio files would require test assets
    // These should be added in a separate test module with sample audio files
}
