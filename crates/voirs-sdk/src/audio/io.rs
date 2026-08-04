//! Audio I/O operations for saving, loading, and format conversion.

use super::buffer::AudioBuffer;
use crate::{error::Result, types::AudioFormat, VoirsError};
use std::path::Path;

impl AudioBuffer {
    /// Save audio as WAV file
    pub fn save_wav(&self, path: impl AsRef<Path>) -> Result<()> {
        use hound::{WavSpec, WavWriter};

        let spec = WavSpec {
            channels: self.channels as u16,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = WavWriter::create(path, spec)
            .map_err(|e| VoirsError::audio_error(format!("Failed to create WAV writer: {e}")))?;

        // Convert f32 samples to i16
        for &sample in &self.samples {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer
                .write_sample(sample_i16)
                .map_err(|e| VoirsError::audio_error(format!("Failed to write sample: {e}")))?;
        }

        writer
            .finalize()
            .map_err(|e| VoirsError::audio_error(format!("Failed to finalize WAV file: {e}")))?;

        Ok(())
    }

    /// Save audio as 32-bit float WAV file
    pub fn save_wav_f32(&self, path: impl AsRef<Path>) -> Result<()> {
        use hound::{WavSpec, WavWriter};

        let spec = WavSpec {
            channels: self.channels as u16,
            sample_rate: self.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = WavWriter::create(path, spec)
            .map_err(|e| VoirsError::audio_error(format!("Failed to create WAV writer: {e}")))?;

        // Write f32 samples directly
        for &sample in &self.samples {
            writer
                .write_sample(sample.clamp(-1.0, 1.0))
                .map_err(|e| VoirsError::audio_error(format!("Failed to write sample: {e}")))?;
        }

        writer
            .finalize()
            .map_err(|e| VoirsError::audio_error(format!("Failed to finalize WAV file: {e}")))?;

        Ok(())
    }

    /// Save audio in specified format
    pub fn save(&self, path: impl AsRef<Path>, format: AudioFormat) -> Result<()> {
        match format {
            AudioFormat::Wav => self.save_wav(path),
            AudioFormat::Flac => self.save_flac(path),
            AudioFormat::Mp3 => self.save_mp3(path),
            AudioFormat::Ogg => self.save_ogg(path),
            AudioFormat::Opus => self.save_opus(path),
        }
    }

    /// Save audio as FLAC file.
    ///
    /// With the `ffi-codecs` feature enabled, this writes a true compressed FLAC
    /// stream via the Pure-Rust [`oxiaudio-encode`] encoder. Without the feature,
    /// it falls back to writing a WAV file at the same path stem.
    #[cfg(feature = "ffi-codecs")]
    pub fn save_flac(&self, path: impl AsRef<Path>) -> Result<()> {
        let bytes = self.to_flac_bytes()?;
        std::fs::write(path.as_ref(), &bytes)
            .map_err(|e| VoirsError::audio_error(format!("Failed to write FLAC file: {e}")))
    }

    /// Save audio as FLAC file (Pure-Rust default build: `ffi-codecs` disabled).
    ///
    /// FLAC encoding requires the `ffi-codecs` feature; without it the audio is
    /// saved as WAV (at the same path stem) and a warning is logged.
    #[cfg(not(feature = "ffi-codecs"))]
    pub fn save_flac(&self, path: impl AsRef<Path>) -> Result<()> {
        tracing::warn!("FLAC encoding requires the 'ffi-codecs' feature; saving as WAV instead");
        self.save_wav(path.as_ref().with_extension("wav"))
    }

    /// Save audio as MP3 file using the LAME encoder.
    ///
    /// Requires the `ffi-codecs` feature (there is no mature pure-Rust MP3 encoder to
    /// depend on instead). Without the feature this returns a clear error rather than
    /// silently writing WAV data at a different path — use [`Self::save_wav`] or
    /// [`Self::save_flac`] (lossless) for a Pure-Rust alternative.
    #[cfg(feature = "ffi-codecs")]
    pub fn save_mp3(&self, path: impl AsRef<Path>) -> Result<()> {
        let bytes = self.to_mp3_bytes()?;
        std::fs::write(path.as_ref(), &bytes)
            .map_err(|e| VoirsError::audio_error(format!("Failed to write MP3 file: {e}")))
    }

    /// Save audio as MP3 file (Pure-Rust default build: `ffi-codecs` disabled).
    ///
    /// MP3 encoding requires the `ffi-codecs` feature (LAME C library); no pure-Rust MP3
    /// encoder currently exists to encode with instead, so this fails closed rather than
    /// silently substituting WAV data at a different path.
    #[cfg(not(feature = "ffi-codecs"))]
    pub fn save_mp3(&self, _path: impl AsRef<Path>) -> Result<()> {
        Err(VoirsError::audio_error(
            "MP3 encoding requires the 'ffi-codecs' feature (LAME C library); no pure-Rust \
             MP3 encoder is available. Use save_wav or save_flac (lossless; FLAC also needs \
             'ffi-codecs') instead.",
        ))
    }

    /// Save audio as an Ogg Vorbis file.
    ///
    /// No pure-Rust Ogg Vorbis **encoder** exists in this workspace's dependency tree
    /// (decoding uses `lewton`, which does not implement encoding). Previously this
    /// method wrote a homebrew byte layout — a bare `b"OggS"` tag followed by
    /// length-prefixed plaintext metadata and raw PCM — which is not a valid Ogg
    /// bitstream: [`Self::load_ogg`]/[`Self::get_ogg_info`] (real `lewton`-based Ogg
    /// Vorbis parsers) could not read it back, and any real Ogg tool would reject it too
    /// despite the genuine magic bytes at the front. Rather than keep producing a file
    /// that lies about its own format, this fails closed with a clear error. Use
    /// [`Self::save_wav`] or [`Self::save_flac`] (lossless) for a Pure-Rust alternative,
    /// or [`Self::save_opus`] (lossy, requires the `ffi-codecs` feature) for a real
    /// compressed format this crate can actually encode.
    pub fn save_ogg(&self, _path: impl AsRef<Path>) -> Result<()> {
        Err(VoirsError::audio_error(
            "OGG Vorbis encoding is not available: no pure-Rust Vorbis encoder exists in this \
             build (decoding uses 'lewton', which is decode-only). Use save_wav/save_flac \
             (lossless) or save_opus (lossy, requires the 'ffi-codecs' feature) instead.",
        ))
    }

    /// Save audio as Opus file
    ///
    /// Opus encoding relies on the `libopus` C library (via the `opus` crate) and is
    /// therefore gated behind the default-OFF `ffi-codecs` feature to keep the default
    /// build Pure Rust.
    #[cfg(feature = "ffi-codecs")]
    pub fn save_opus(&self, path: impl AsRef<Path>) -> Result<()> {
        use opus::{Application, Channels, Encoder};
        use std::fs::File;
        use std::io::Write;

        // Opus requires specific sample rates (8, 12, 16, 24, or 48 kHz)
        let opus_sample_rate = match self.sample_rate {
            8000 => 8000,
            12000 => 12000,
            16000 => 16000,
            24000 => 24000,
            48000 => 48000,
            _ => 48000, // Default to 48kHz and resample
        };

        let channels = match self.channels {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            _ => {
                return Err(VoirsError::audio_error(
                    "Opus only supports mono or stereo audio",
                ))
            }
        };

        let mut encoder =
            Encoder::new(opus_sample_rate, channels, Application::Audio).map_err(|e| {
                VoirsError::audio_error(format!("Failed to create Opus encoder: {e:?}"))
            })?;

        // Convert f32 samples to i16
        let mut samples_i16 = Vec::with_capacity(self.samples.len());
        for &sample in &self.samples {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            samples_i16.push(sample_i16);
        }

        // Opus encodes in frames - we'll use a simple approach for now
        let frame_size = 960; // 20ms at 48kHz
        let mut encoded_data = Vec::new();

        for chunk in samples_i16.chunks(frame_size * self.channels as usize) {
            let mut output = vec![0u8; 4000]; // Max Opus frame size
            let encoded_size = encoder.encode(chunk, &mut output).map_err(|e| {
                VoirsError::audio_error(format!("Failed to encode Opus frame: {e:?}"))
            })?;

            encoded_data.extend_from_slice(&output[..encoded_size]);
        }

        // Write to file (Note: This is raw Opus data, not in an Ogg container)
        let mut file = File::create(path)
            .map_err(|e| VoirsError::audio_error(format!("Failed to create Opus file: {e}")))?;
        file.write_all(&encoded_data)
            .map_err(|e| VoirsError::audio_error(format!("Failed to write Opus data: {e}")))?;

        Ok(())
    }

    /// Save audio as Opus file (stub when the `ffi-codecs` feature is disabled).
    ///
    /// Opus encoding requires the `libopus` C library and is only available with the
    /// `ffi-codecs` feature enabled. This stub keeps the public API stable and returns a
    /// descriptive error so callers get a clear message instead of a missing symbol.
    #[cfg(not(feature = "ffi-codecs"))]
    pub fn save_opus(&self, _path: impl AsRef<Path>) -> Result<()> {
        Err(VoirsError::audio_error(
            "Opus codec requires the 'ffi-codecs' feature (libopus C library)",
        ))
    }

    /// Play audio through system speakers
    pub fn play(&self) -> Result<()> {
        use cpal::{
            traits::{DeviceTrait, HostTrait, StreamTrait},
            Device, SampleFormat, StreamConfig,
        };
        use std::sync::{Arc, Mutex};

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| VoirsError::audio_error("No output device available"))?;

        let config = device
            .default_output_config()
            .map_err(|e| VoirsError::audio_error(format!("Failed to get output config: {e}")))?;

        let sample_format = config.sample_format();
        let stream_config: StreamConfig = config.into();

        // Convert our samples to the device's sample rate if needed
        let samples = if self.sample_rate == stream_config.sample_rate {
            self.samples.clone()
        } else {
            self.resample(stream_config.sample_rate)?.samples
        };

        let samples = Arc::new(Mutex::new(samples.into_iter()));
        let channels = self.channels;

        let build_stream = |device: &Device, config: &StreamConfig, format: SampleFormat| {
            let samples = samples.clone();
            match format {
                SampleFormat::F32 => device.build_output_stream(
                    *config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        let mut samples_lock = samples.lock().unwrap_or_else(|e| e.into_inner());
                        for frame in data.chunks_mut(channels as usize) {
                            let sample = samples_lock.next().unwrap_or(0.0);
                            for channel_sample in frame.iter_mut() {
                                *channel_sample = sample;
                            }
                        }
                    },
                    move |err| eprintln!("Audio stream error: {err}"),
                    None,
                ),
                SampleFormat::I16 => device.build_output_stream(
                    *config,
                    move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                        let mut samples_lock = samples.lock().unwrap_or_else(|e| e.into_inner());
                        for frame in data.chunks_mut(channels as usize) {
                            let sample = samples_lock.next().unwrap_or(0.0);
                            let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                            for channel_sample in frame.iter_mut() {
                                *channel_sample = sample_i16;
                            }
                        }
                    },
                    move |err| eprintln!("Audio stream error: {err}"),
                    None,
                ),
                SampleFormat::U16 => device.build_output_stream(
                    *config,
                    move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                        let mut samples_lock = samples.lock().unwrap_or_else(|e| e.into_inner());
                        for frame in data.chunks_mut(channels as usize) {
                            let sample = samples_lock.next().unwrap_or(0.0);
                            let sample_u16 =
                                ((sample.clamp(-1.0, 1.0) + 1.0) * u16::MAX as f32 / 2.0) as u16;
                            for channel_sample in frame.iter_mut() {
                                *channel_sample = sample_u16;
                            }
                        }
                    },
                    move |err| eprintln!("Audio stream error: {err}"),
                    None,
                ),
                _ => unreachable!("format pre-checked"),
            }
        };

        if !matches!(
            sample_format,
            SampleFormat::F32 | SampleFormat::I16 | SampleFormat::U16
        ) {
            return Err(VoirsError::audio_error(format!(
                "Unsupported sample format: {sample_format:?}"
            )));
        }
        let stream = build_stream(&device, &stream_config, sample_format)
            .map_err(|e| VoirsError::audio_error(format!("Failed to build audio stream: {e}")))?;

        stream
            .play()
            .map_err(|e| VoirsError::audio_error(format!("Failed to start audio stream: {e}")))?;

        // Wait for playback to complete
        let duration = self.duration();
        std::thread::sleep(std::time::Duration::from_secs_f32(duration));

        tracing::info!(
            "Audio playback completed: {:.2}s @ {}Hz",
            duration,
            self.sample_rate
        );
        Ok(())
    }

    /// Play audio with callback for progress updates
    pub fn play_with_callback<F>(&self, callback: F) -> Result<()>
    where
        F: FnMut(f32) + Send + 'static, // Progress callback (0.0 to 1.0)
    {
        use cpal::{
            traits::{DeviceTrait, HostTrait, StreamTrait},
            Device, SampleFormat, StreamConfig,
        };
        use std::sync::{Arc, Mutex};
        use std::time::{Duration, Instant};

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| VoirsError::audio_error("No output device available"))?;

        let config = device
            .default_output_config()
            .map_err(|e| VoirsError::audio_error(format!("Failed to get output config: {e}")))?;

        let sample_format = config.sample_format();
        let stream_config: StreamConfig = config.into();

        // Convert our samples to the device's sample rate if needed
        let samples = if self.sample_rate == stream_config.sample_rate {
            self.samples.clone()
        } else {
            self.resample(stream_config.sample_rate)?.samples
        };

        let total_samples = samples.len();
        let samples_iter = Arc::new(Mutex::new(samples.into_iter().enumerate()));
        let channels = self.channels;

        let progress_callback = Arc::new(Mutex::new(callback));
        let last_progress_update = Arc::new(Mutex::new(Instant::now()));

        let build_stream = |device: &Device, config: &StreamConfig, format: SampleFormat| {
            let samples = samples_iter.clone();
            let progress_callback = progress_callback.clone();
            let last_progress_update = last_progress_update.clone();

            match format {
                SampleFormat::F32 => device.build_output_stream(
                    *config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        let mut samples_lock = samples.lock().unwrap_or_else(|e| e.into_inner());
                        for frame in data.chunks_mut(channels as usize) {
                            if let Some((index, sample)) = samples_lock.next() {
                                for channel_sample in frame.iter_mut() {
                                    *channel_sample = sample;
                                }

                                // Update progress every 100ms
                                let mut last_update = last_progress_update
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner());
                                let now = Instant::now();
                                if now.duration_since(*last_update) >= Duration::from_millis(100) {
                                    let progress = (index as f32) / (total_samples as f32);
                                    if let Ok(mut callback) = progress_callback.lock() {
                                        callback(progress);
                                    }
                                    *last_update = now;
                                }
                            } else {
                                // End of samples, fill with silence
                                for channel_sample in frame.iter_mut() {
                                    *channel_sample = 0.0;
                                }
                            }
                        }
                    },
                    move |err| eprintln!("Audio stream error: {err}"),
                    None,
                ),
                SampleFormat::I16 => device.build_output_stream(
                    *config,
                    move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                        let mut samples_lock = samples.lock().unwrap_or_else(|e| e.into_inner());
                        for frame in data.chunks_mut(channels as usize) {
                            if let Some((index, sample)) = samples_lock.next() {
                                let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                                for channel_sample in frame.iter_mut() {
                                    *channel_sample = sample_i16;
                                }

                                // Update progress every 100ms
                                let mut last_update = last_progress_update
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner());
                                let now = Instant::now();
                                if now.duration_since(*last_update) >= Duration::from_millis(100) {
                                    let progress = (index as f32) / (total_samples as f32);
                                    if let Ok(mut callback) = progress_callback.lock() {
                                        callback(progress);
                                    }
                                    *last_update = now;
                                }
                            } else {
                                // End of samples, fill with silence
                                for channel_sample in frame.iter_mut() {
                                    *channel_sample = 0;
                                }
                            }
                        }
                    },
                    move |err| eprintln!("Audio stream error: {err}"),
                    None,
                ),
                SampleFormat::U16 => device.build_output_stream(
                    *config,
                    move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                        let mut samples_lock = samples.lock().unwrap_or_else(|e| e.into_inner());
                        for frame in data.chunks_mut(channels as usize) {
                            if let Some((index, sample)) = samples_lock.next() {
                                let sample_u16 = ((sample.clamp(-1.0, 1.0) + 1.0) * u16::MAX as f32
                                    / 2.0) as u16;
                                for channel_sample in frame.iter_mut() {
                                    *channel_sample = sample_u16;
                                }

                                // Update progress every 100ms
                                let mut last_update = last_progress_update
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner());
                                let now = Instant::now();
                                if now.duration_since(*last_update) >= Duration::from_millis(100) {
                                    let progress = (index as f32) / (total_samples as f32);
                                    if let Ok(mut callback) = progress_callback.lock() {
                                        callback(progress);
                                    }
                                    *last_update = now;
                                }
                            } else {
                                // End of samples, fill with silence
                                for channel_sample in frame.iter_mut() {
                                    *channel_sample = u16::MAX / 2; // Mid-point for silence
                                }
                            }
                        }
                    },
                    move |err| eprintln!("Audio stream error: {err}"),
                    None,
                ),
                _ => unreachable!("format pre-checked"),
            }
        };

        if !matches!(
            sample_format,
            SampleFormat::F32 | SampleFormat::I16 | SampleFormat::U16
        ) {
            return Err(VoirsError::audio_error(format!(
                "Unsupported sample format: {sample_format:?}"
            )));
        }
        let stream = build_stream(&device, &stream_config, sample_format)
            .map_err(|e| VoirsError::audio_error(format!("Failed to build audio stream: {e}")))?;

        stream
            .play()
            .map_err(|e| VoirsError::audio_error(format!("Failed to start audio stream: {e}")))?;

        // Wait for playback to complete
        let duration = self.duration();
        std::thread::sleep(std::time::Duration::from_secs_f32(duration));

        // Ensure final progress callback
        if let Ok(mut callback) = progress_callback.lock() {
            callback(1.0);
        }

        tracing::info!(
            "Audio playback with progress completed: {:.2}s @ {}Hz",
            duration,
            self.sample_rate
        );
        Ok(())
    }

    /// Convert to different format as bytes
    pub fn to_format(&self, format: AudioFormat) -> Result<Vec<u8>> {
        match format {
            AudioFormat::Wav => self.to_wav_bytes(),
            AudioFormat::Flac => self.to_flac_bytes(),
            AudioFormat::Mp3 => self.to_mp3_bytes(),
            AudioFormat::Ogg => self.to_ogg_bytes(),
            AudioFormat::Opus => self.to_opus_bytes(),
        }
    }

    /// Convert to WAV bytes
    pub fn to_wav_bytes(&self) -> Result<Vec<u8>> {
        use hound::{WavSpec, WavWriter};
        use std::io::Cursor;

        let spec = WavSpec {
            channels: self.channels as u16,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = WavWriter::new(&mut cursor, spec).map_err(|e| {
                VoirsError::audio_error(format!("Failed to create WAV writer: {e}"))
            })?;

            // Convert f32 samples to i16
            for &sample in &self.samples {
                let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                writer
                    .write_sample(sample_i16)
                    .map_err(|e| VoirsError::audio_error(format!("Failed to write sample: {e}")))?;
            }

            writer
                .finalize()
                .map_err(|e| VoirsError::audio_error(format!("Failed to finalize WAV: {e}")))?;
        }

        Ok(cursor.into_inner())
    }

    /// Convert to FLAC bytes using the Pure-Rust OxiAudio encoder.
    ///
    /// Requires the `ffi-codecs` feature. The `f32` samples are encoded to a
    /// 24-bit FLAC stream at the default compression level.
    #[cfg(feature = "ffi-codecs")]
    pub fn to_flac_bytes(&self) -> Result<Vec<u8>> {
        use oxiaudio_core::{
            AudioBuffer as OxiAudioBuffer, AudioEncoder, ChannelLayout, SampleFormat,
        };
        use oxiaudio_encode::FlacEncoder;

        let oxi_buffer = OxiAudioBuffer::<f32> {
            samples: self.samples.clone(),
            sample_rate: self.sample_rate,
            channels: ChannelLayout::from(self.channels as u16),
            format: SampleFormat::F32,
        };

        let mut encoder = FlacEncoder::default().with_bits_per_sample(24);
        let mut cursor = std::io::Cursor::new(Vec::new());
        encoder
            .encode(&oxi_buffer, &mut cursor)
            .map_err(|e| VoirsError::audio_error(format!("FLAC encoding failed: {e}")))?;

        Ok(cursor.into_inner())
    }

    /// Convert to FLAC bytes (Pure-Rust default build: `ffi-codecs` disabled).
    ///
    /// FLAC encoding requires the `ffi-codecs` feature; without it WAV bytes are
    /// returned and a warning is logged.
    #[cfg(not(feature = "ffi-codecs"))]
    pub fn to_flac_bytes(&self) -> Result<Vec<u8>> {
        tracing::warn!(
            "FLAC encoding requires the 'ffi-codecs' feature; returning WAV bytes instead"
        );
        self.to_wav_bytes()
    }

    /// Convert to MP3 bytes using the LAME encoder.
    ///
    /// Requires the `ffi-codecs` feature; see [`Self::save_mp3`] for details.
    #[cfg(feature = "ffi-codecs")]
    pub fn to_mp3_bytes(&self) -> Result<Vec<u8>> {
        use mp3lame_encoder::{Bitrate, Builder, FlushNoGap, InterleavedPcm, MonoPcm, Quality};

        if self.channels == 0 || self.channels > 2 {
            return Err(VoirsError::audio_error(
                "MP3 encoding only supports mono or stereo audio",
            ));
        }
        if self.sample_rate > 48_000 {
            return Err(VoirsError::audio_error(
                "MP3 does not support sample rates above 48kHz",
            ));
        }

        // Convert f32 samples to i16 PCM (interleaved for stereo, matching self.samples'
        // existing layout).
        let pcm: Vec<i16> = self
            .samples
            .iter()
            .map(|&sample| (sample.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect();

        let mut builder = Builder::new()
            .ok_or_else(|| VoirsError::audio_error("Failed to create MP3 encoder"))?;
        builder
            .set_sample_rate(self.sample_rate)
            .map_err(|e| VoirsError::audio_error(format!("Failed to set MP3 sample rate: {e}")))?;
        builder
            .set_num_channels(self.channels as u8)
            .map_err(|e| VoirsError::audio_error(format!("Failed to set MP3 channels: {e}")))?;
        builder
            .set_brate(Bitrate::Kbps128)
            .map_err(|e| VoirsError::audio_error(format!("Failed to set MP3 bitrate: {e}")))?;
        builder
            .set_quality(Quality::Good)
            .map_err(|e| VoirsError::audio_error(format!("Failed to set MP3 quality: {e}")))?;
        let mut encoder = builder
            .build()
            .map_err(|e| VoirsError::audio_error(format!("Failed to build MP3 encoder: {e}")))?;

        let samples_per_channel = pcm.len() / self.channels as usize;
        let required_size = mp3lame_encoder::max_required_buffer_size(samples_per_channel);
        let mut mp3_buffer: Vec<u8> = Vec::with_capacity(required_size);

        // Safety: `encode` writes at most `spare_capacity_mut().len()` initialized bytes
        // and returns exactly how many it wrote; `set_len` below only accounts for that
        // many newly-initialized bytes.
        let encoded_size = if self.channels == 1 {
            encoder
                .encode(MonoPcm(&pcm), mp3_buffer.spare_capacity_mut())
                .map_err(|e| VoirsError::audio_error(format!("MP3 encoding failed: {e}")))?
        } else {
            encoder
                .encode(InterleavedPcm(&pcm), mp3_buffer.spare_capacity_mut())
                .map_err(|e| VoirsError::audio_error(format!("MP3 encoding failed: {e}")))?
        };
        unsafe {
            mp3_buffer.set_len(mp3_buffer.len().wrapping_add(encoded_size));
        }

        // Reserve room for the trailing flush frame (LAME's documented max single-frame
        // size) regardless of how much of `required_size` the main encode pass consumed.
        mp3_buffer.reserve(7200);
        let flushed_size = encoder
            .flush::<FlushNoGap>(mp3_buffer.spare_capacity_mut())
            .map_err(|e| VoirsError::audio_error(format!("MP3 flush failed: {e}")))?;
        unsafe {
            mp3_buffer.set_len(mp3_buffer.len().wrapping_add(flushed_size));
        }

        Ok(mp3_buffer)
    }

    /// Convert to MP3 bytes (Pure-Rust default build: `ffi-codecs` disabled).
    ///
    /// MP3 encoding requires the `ffi-codecs` feature (LAME C library); no pure-Rust MP3
    /// encoder is available to encode with instead, so this fails closed rather than
    /// silently returning WAV bytes mislabeled as MP3.
    #[cfg(not(feature = "ffi-codecs"))]
    pub fn to_mp3_bytes(&self) -> Result<Vec<u8>> {
        Err(VoirsError::audio_error(
            "MP3 encoding requires the 'ffi-codecs' feature (LAME C library); no pure-Rust \
             MP3 encoder is available. Use to_wav_bytes or to_flac_bytes (lossless; FLAC also \
             needs 'ffi-codecs') instead.",
        ))
    }

    /// Convert to Ogg Vorbis bytes.
    ///
    /// See [`Self::save_ogg`] — no pure-Rust Vorbis encoder is available, so this fails
    /// closed rather than returning bytes that claim to be Ogg Vorbis but are not a valid
    /// Ogg bitstream.
    pub fn to_ogg_bytes(&self) -> Result<Vec<u8>> {
        Err(VoirsError::audio_error(
            "OGG Vorbis encoding is not available: no pure-Rust Vorbis encoder exists in this \
             build (decoding uses 'lewton', which is decode-only). Use to_wav_bytes/to_flac_bytes \
             (lossless) or to_opus_bytes (lossy, requires the 'ffi-codecs' feature) instead.",
        ))
    }

    /// Convert to Opus bytes
    ///
    /// Opus encoding relies on the `libopus` C library (via the `opus` crate) and is
    /// therefore gated behind the default-OFF `ffi-codecs` feature to keep the default
    /// build Pure Rust.
    #[cfg(feature = "ffi-codecs")]
    pub fn to_opus_bytes(&self) -> Result<Vec<u8>> {
        use opus::{Application, Channels, Encoder};

        // Opus requires specific sample rates (8, 12, 16, 24, or 48 kHz)
        let opus_sample_rate = match self.sample_rate {
            8000 => 8000,
            12000 => 12000,
            16000 => 16000,
            24000 => 24000,
            48000 => 48000,
            _ => 48000, // Default to 48kHz
        };

        let channels = match self.channels {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            _ => {
                return Err(VoirsError::audio_error(
                    "Opus only supports mono or stereo audio",
                ))
            }
        };

        let mut encoder =
            Encoder::new(opus_sample_rate, channels, Application::Audio).map_err(|e| {
                VoirsError::audio_error(format!("Failed to create Opus encoder: {e:?}"))
            })?;

        // Convert f32 samples to i16
        let mut samples_i16 = Vec::with_capacity(self.samples.len());
        for &sample in &self.samples {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            samples_i16.push(sample_i16);
        }

        // Opus encodes in frames
        let frame_size = 960; // 20ms at 48kHz
        let mut encoded_data = Vec::new();

        for chunk in samples_i16.chunks(frame_size * self.channels as usize) {
            let mut output = vec![0u8; 4000]; // Max Opus frame size
            let encoded_size = encoder.encode(chunk, &mut output).map_err(|e| {
                VoirsError::audio_error(format!("Failed to encode Opus frame: {e:?}"))
            })?;

            encoded_data.extend_from_slice(&output[..encoded_size]);
        }

        Ok(encoded_data)
    }

    /// Convert to Opus bytes (stub when the `ffi-codecs` feature is disabled).
    ///
    /// Opus encoding requires the `libopus` C library and is only available with the
    /// `ffi-codecs` feature enabled. This stub keeps the public API stable and returns a
    /// descriptive error so callers get a clear message instead of a missing symbol.
    #[cfg(not(feature = "ffi-codecs"))]
    pub fn to_opus_bytes(&self) -> Result<Vec<u8>> {
        Err(VoirsError::audio_error(
            "Opus codec requires the 'ffi-codecs' feature (libopus C library)",
        ))
    }

    /// Load audio from WAV file
    pub fn load_wav(path: impl AsRef<Path>) -> Result<AudioBuffer> {
        use hound::WavReader;

        let mut reader = WavReader::open(path)
            .map_err(|e| VoirsError::audio_error(format!("Failed to open WAV file: {e}")))?;

        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let channels = spec.channels as u32;

        let samples: Result<Vec<f32>> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| {
                    s.map_err(|e| VoirsError::audio_error(format!("Failed to read sample: {e}")))
                })
                .collect(),
            hound::SampleFormat::Int => match spec.bits_per_sample {
                16 => reader
                    .samples::<i16>()
                    .map(|s| {
                        s.map(|sample| sample as f32 / 32767.0).map_err(|e| {
                            VoirsError::audio_error(format!("Failed to read sample: {e}"))
                        })
                    })
                    .collect(),
                24 => reader
                    .samples::<i32>()
                    .map(|s| {
                        s.map(|sample| sample as f32 / 8388607.0).map_err(|e| {
                            VoirsError::audio_error(format!("Failed to read sample: {e}"))
                        })
                    })
                    .collect(),
                32 => reader
                    .samples::<i32>()
                    .map(|s| {
                        s.map(|sample| sample as f32 / 2147483647.0).map_err(|e| {
                            VoirsError::audio_error(format!("Failed to read sample: {e}"))
                        })
                    })
                    .collect(),
                _ => {
                    return Err(VoirsError::audio_error(format!(
                        "Unsupported bit depth: {}",
                        spec.bits_per_sample
                    )))
                }
            },
        };

        let samples = samples?;
        Ok(AudioBuffer::new(samples, sample_rate, channels))
    }

    /// Load audio from file (auto-detect format)
    pub fn load(path: impl AsRef<Path>) -> Result<AudioBuffer> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "wav" => Self::load_wav(path),
            "flac" => Self::load_flac(path),
            "mp3" => Self::load_mp3(path),
            "ogg" => Self::load_ogg(path),
            "opus" => Self::load_opus(path),
            _ => Err(VoirsError::audio_error(format!(
                "Unsupported audio format: {extension}"
            ))),
        }
    }

    /// Load audio from FLAC file
    pub fn load_flac(path: impl AsRef<Path>) -> Result<AudioBuffer> {
        use claxon::FlacReader;
        use std::fs::File;

        let file = File::open(path)
            .map_err(|e| VoirsError::audio_error(format!("Failed to open FLAC file: {e}")))?;

        let mut reader = FlacReader::new(file)
            .map_err(|e| VoirsError::audio_error(format!("Failed to create FLAC reader: {e:?}")))?;

        let info = reader.streaminfo();
        let sample_rate = info.sample_rate;
        let channels = info.channels;
        let bits_per_sample = info.bits_per_sample;

        let mut samples = Vec::new();

        // Read all samples from FLAC file
        for sample_result in reader.samples() {
            let sample = sample_result.map_err(|e| {
                VoirsError::audio_error(format!("Failed to read FLAC sample: {e:?}"))
            })?;

            // Convert to f32 based on bit depth
            let sample_f32 = match bits_per_sample {
                16 => sample as f32 / 32767.0,
                24 => sample as f32 / 8388607.0,
                32 => sample as f32 / 2147483647.0,
                _ => {
                    return Err(VoirsError::audio_error(format!(
                        "Unsupported FLAC bit depth: {bits_per_sample}"
                    )))
                }
            };
            samples.push(sample_f32);
        }

        Ok(AudioBuffer::new(samples, sample_rate, channels))
    }

    /// Load audio from MP3 file
    ///
    /// MP3 decoding relies on the `minimp3` C library (via the `minimp3` crate) and is
    /// therefore gated behind the default-OFF `ffi-codecs` feature to keep the default
    /// build Pure Rust.
    #[cfg(feature = "ffi-codecs")]
    pub fn load_mp3(path: impl AsRef<Path>) -> Result<AudioBuffer> {
        use minimp3::{Decoder, Frame};
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)
            .map_err(|e| VoirsError::audio_error(format!("Failed to open MP3 file: {e}")))?;

        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .map_err(|e| VoirsError::audio_error(format!("Failed to read MP3 file: {e}")))?;

        let mut decoder = Decoder::new(&buf[..]);
        let mut samples = Vec::new();
        let mut sample_rate = 0;
        let mut channels = 0;

        // Decode all frames
        loop {
            match decoder.next_frame() {
                Ok(Frame {
                    data,
                    sample_rate: sr,
                    channels: ch,
                    ..
                }) => {
                    if sample_rate == 0 {
                        sample_rate = sr as u32;
                        channels = ch as u32;
                    }

                    // Convert i16 samples to f32
                    for &sample in &data {
                        samples.push(sample as f32 / 32767.0);
                    }
                }
                Err(minimp3::Error::Eof) => break,
                Err(e) => {
                    return Err(VoirsError::audio_error(format!(
                        "Failed to decode MP3: {e:?}"
                    )))
                }
            }
        }

        if samples.is_empty() {
            return Err(VoirsError::audio_error("No audio data found in MP3 file"));
        }

        Ok(AudioBuffer::new(samples, sample_rate, channels))
    }

    /// Load audio from MP3 file (stub when the `ffi-codecs` feature is disabled).
    ///
    /// MP3 decoding requires the `minimp3` C library and is only available with the
    /// `ffi-codecs` feature enabled. This stub keeps the public API stable and returns a
    /// descriptive error so callers get a clear message instead of a missing symbol.
    #[cfg(not(feature = "ffi-codecs"))]
    pub fn load_mp3(_path: impl AsRef<Path>) -> Result<AudioBuffer> {
        Err(VoirsError::audio_error(
            "MP3 codec requires the 'ffi-codecs' feature (minimp3 C library)",
        ))
    }

    /// Load audio from OGG file
    pub fn load_ogg(path: impl AsRef<Path>) -> Result<AudioBuffer> {
        use lewton::inside_ogg::OggStreamReader;
        use std::fs::File;

        let file = File::open(path)
            .map_err(|e| VoirsError::audio_error(format!("Failed to open OGG file: {e}")))?;

        let mut stream_reader = OggStreamReader::new(file)
            .map_err(|e| VoirsError::audio_error(format!("Failed to create OGG reader: {e:?}")))?;

        let sample_rate = stream_reader.ident_hdr.audio_sample_rate;
        let channels = stream_reader.ident_hdr.audio_channels as u32;

        let mut samples = Vec::new();

        // Read all packets and decode them
        while let Some(packet) = stream_reader
            .read_dec_packet_itl()
            .map_err(|e| VoirsError::audio_error(format!("Failed to read OGG packet: {e:?}")))?
        {
            // Convert i16 samples to f32
            for sample in packet {
                samples.push(sample as f32 / 32767.0);
            }
        }

        if samples.is_empty() {
            return Err(VoirsError::audio_error("No audio data found in OGG file"));
        }

        Ok(AudioBuffer::new(samples, sample_rate, channels))
    }

    /// Load audio from an Ogg Opus file.
    ///
    /// Parses the Ogg container with the `ogg` crate and decodes each Opus packet with the
    /// `opus` crate.  The OpusHead identification header is read to determine the channel
    /// count and the original input sample rate stored by the encoder.  Opus always outputs
    /// PCM at 48 000 Hz internally; we advertise the original sample rate so callers can
    /// resample if needed, but the raw PCM is at 48 kHz.
    ///
    /// Opus decoding relies on the `libopus` C library (via the `opus` crate) and is
    /// therefore gated behind the default-OFF `ffi-codecs` feature to keep the default
    /// build Pure Rust.
    #[cfg(feature = "ffi-codecs")]
    pub fn load_opus(path: impl AsRef<Path>) -> Result<AudioBuffer> {
        use ogg::reading::PacketReader;
        use opus::{Channels, Decoder};
        use std::fs::File;

        let file = File::open(path.as_ref())
            .map_err(|e| VoirsError::audio_error(format!("Failed to open Opus file: {e}")))?;

        let mut reader = PacketReader::new(file);

        // --- OpusHead (first logical packet) ---
        let head = reader
            .read_packet_expected()
            .map_err(|e| VoirsError::audio_error(format!("Failed to read Opus header: {e:?}")))?;

        if head.data.len() < 19 || &head.data[..8] != b"OpusHead" {
            return Err(VoirsError::audio_error(
                "Not a valid Ogg Opus file — missing OpusHead magic",
            ));
        }
        let n_channels = head.data[9] as u32;
        let pre_skip = u16::from_le_bytes([head.data[10], head.data[11]]) as usize;
        let input_sample_rate =
            u32::from_le_bytes([head.data[12], head.data[13], head.data[14], head.data[15]]);
        // Advertise the original rate; Opus PCM is always at 48 kHz.
        let sample_rate = if input_sample_rate > 0 {
            input_sample_rate
        } else {
            48_000
        };

        let opus_channels = if n_channels == 1 {
            Channels::Mono
        } else {
            Channels::Stereo
        };
        let mut decoder = Decoder::new(48_000, opus_channels).map_err(|e| {
            VoirsError::audio_error(format!("Failed to create Opus decoder: {e:?}"))
        })?;

        // --- OpusTags (second logical packet — skip) ---
        reader
            .read_packet_expected()
            .map_err(|e| VoirsError::audio_error(format!("Failed to skip OpusTags: {e:?}")))?;

        // --- Audio packets ---
        // Maximum Opus frame size: 120 ms at 48 kHz = 5760 samples per channel.
        const MAX_FRAME_SAMPLES: usize = 5760;
        let mut decode_buf = vec![0.0_f32; MAX_FRAME_SAMPLES * n_channels as usize];
        let mut samples: Vec<f32> = Vec::new();

        loop {
            match reader.read_packet() {
                Ok(Some(pkt)) => {
                    let samples_per_channel = decoder
                        .decode_float(&pkt.data, &mut decode_buf, false)
                        .map_err(|e| {
                            VoirsError::audio_error(format!("Opus decode error: {e:?}"))
                        })?;
                    samples.extend_from_slice(
                        &decode_buf[..samples_per_channel * n_channels as usize],
                    );
                }
                Ok(None) => break,
                Err(e) => return Err(VoirsError::audio_error(format!("Ogg read error: {e:?}"))),
            }
        }

        // Remove encoder pre-skip padding from the start of the decoded PCM.
        let skip_samples = pre_skip * n_channels as usize;
        let samples = if skip_samples < samples.len() {
            samples[skip_samples..].to_vec()
        } else {
            Vec::new()
        };

        if samples.is_empty() {
            return Err(VoirsError::audio_error("No audio data found in Opus file"));
        }

        Ok(AudioBuffer::new(samples, sample_rate, n_channels))
    }

    /// Load audio from an Ogg Opus file (stub when the `ffi-codecs` feature is disabled).
    ///
    /// Opus decoding requires the `libopus` C library and is only available with the
    /// `ffi-codecs` feature enabled. This stub keeps the public API stable and returns a
    /// descriptive error so callers get a clear message instead of a missing symbol.
    #[cfg(not(feature = "ffi-codecs"))]
    pub fn load_opus(_path: impl AsRef<Path>) -> Result<AudioBuffer> {
        Err(VoirsError::audio_error(
            "Opus codec requires the 'ffi-codecs' feature (libopus C library)",
        ))
    }

    /// Get audio information without loading samples
    pub fn get_info(path: impl AsRef<Path>) -> Result<AudioInfo> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "wav" => Self::get_wav_info(path),
            "flac" => Self::get_flac_info(path),
            "mp3" => Self::get_mp3_info(path),
            "ogg" => Self::get_ogg_info(path),
            "opus" => Self::get_opus_info(path),
            _ => Err(VoirsError::audio_error(format!(
                "Unsupported audio format: {extension}"
            ))),
        }
    }

    /// Get WAV file information
    pub fn get_wav_info(path: impl AsRef<Path>) -> Result<AudioInfo> {
        use hound::WavReader;

        let reader = WavReader::open(path)
            .map_err(|e| VoirsError::audio_error(format!("Failed to open WAV file: {e}")))?;

        let spec = reader.spec();
        let sample_count = reader.len() as usize;
        let duration = sample_count as f32 / (spec.sample_rate * spec.channels as u32) as f32;

        Ok(AudioInfo {
            sample_rate: spec.sample_rate,
            channels: spec.channels as u32,
            duration,
            sample_count,
            format: AudioFormat::Wav,
        })
    }

    /// Get FLAC file information
    pub fn get_flac_info(path: impl AsRef<Path>) -> Result<AudioInfo> {
        use claxon::FlacReader;
        use std::fs::File;

        let file = File::open(path)
            .map_err(|e| VoirsError::audio_error(format!("Failed to open FLAC file: {e}")))?;

        let reader = FlacReader::new(file)
            .map_err(|e| VoirsError::audio_error(format!("Failed to create FLAC reader: {e:?}")))?;

        let info = reader.streaminfo();
        let sample_rate = info.sample_rate;
        let channels = info.channels;
        let sample_count = info.samples.unwrap_or(0) as usize;
        let duration = sample_count as f32 / (sample_rate * channels) as f32;

        Ok(AudioInfo {
            sample_rate,
            channels,
            duration,
            sample_count,
            format: AudioFormat::Flac,
        })
    }

    /// Get MP3 file information
    ///
    /// MP3 decoding relies on the `minimp3` C library (via the `minimp3` crate) and is
    /// therefore gated behind the default-OFF `ffi-codecs` feature to keep the default
    /// build Pure Rust.
    #[cfg(feature = "ffi-codecs")]
    #[allow(unused_assignments)] // False positive: variables are used after assignment in match block
    pub fn get_mp3_info(path: impl AsRef<Path>) -> Result<AudioInfo> {
        use minimp3::{Decoder, Frame};
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)
            .map_err(|e| VoirsError::audio_error(format!("Failed to open MP3 file: {e}")))?;

        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .map_err(|e| VoirsError::audio_error(format!("Failed to read MP3 file: {e}")))?;

        let mut decoder = Decoder::new(&buf[..]);
        let mut sample_count = 0;
        let mut sample_rate = 0;
        let mut channels = 0;

        // Decode just enough to get file info
        match decoder.next_frame() {
            Ok(Frame {
                sample_rate: sr,
                channels: ch,
                ..
            }) => {
                sample_rate = sr as u32;
                channels = ch as u32;

                // Count total samples by decoding all frames
                loop {
                    match decoder.next_frame() {
                        Ok(Frame { data, .. }) => {
                            sample_count += data.len();
                        }
                        Err(minimp3::Error::Eof) => break,
                        Err(e) => {
                            return Err(VoirsError::audio_error(format!(
                                "Failed to decode MP3: {e:?}"
                            )))
                        }
                    }
                }
            }
            Err(e) => {
                return Err(VoirsError::audio_error(format!(
                    "Failed to read MP3 header: {e:?}"
                )))
            }
        }

        let duration = sample_count as f32 / (sample_rate * channels) as f32;

        Ok(AudioInfo {
            sample_rate,
            channels,
            duration,
            sample_count,
            format: AudioFormat::Mp3,
        })
    }

    /// Get MP3 file information (stub when the `ffi-codecs` feature is disabled).
    ///
    /// MP3 decoding requires the `minimp3` C library and is only available with the
    /// `ffi-codecs` feature enabled. This stub keeps the public API stable and returns a
    /// descriptive error so callers get a clear message instead of a missing symbol.
    #[cfg(not(feature = "ffi-codecs"))]
    pub fn get_mp3_info(_path: impl AsRef<Path>) -> Result<AudioInfo> {
        Err(VoirsError::audio_error(
            "MP3 codec requires the 'ffi-codecs' feature (minimp3 C library)",
        ))
    }

    /// Get OGG file information
    pub fn get_ogg_info(path: impl AsRef<Path>) -> Result<AudioInfo> {
        use lewton::inside_ogg::OggStreamReader;
        use std::fs::File;

        let file = File::open(path)
            .map_err(|e| VoirsError::audio_error(format!("Failed to open OGG file: {e}")))?;

        let mut stream_reader = OggStreamReader::new(file)
            .map_err(|e| VoirsError::audio_error(format!("Failed to create OGG reader: {e:?}")))?;

        let sample_rate = stream_reader.ident_hdr.audio_sample_rate;
        let channels = stream_reader.ident_hdr.audio_channels as u32;

        // Count total samples
        let mut sample_count = 0;
        while let Some(packet) = stream_reader
            .read_dec_packet_itl()
            .map_err(|e| VoirsError::audio_error(format!("Failed to read OGG packet: {e:?}")))?
        {
            sample_count += packet.len();
        }

        let duration = sample_count as f32 / (sample_rate * channels) as f32;

        Ok(AudioInfo {
            sample_rate,
            channels,
            duration,
            sample_count,
            format: AudioFormat::Ogg,
        })
    }

    /// Get audio information from an Ogg Opus file without decoding the full stream.
    ///
    /// Reads only the OpusHead identification packet to determine channel count and the
    /// original input sample rate.  `sample_count` and `duration` are set to 0 because
    /// determining them precisely would require decoding all packets; callers that need
    /// accurate duration should call `load_opus` instead.
    pub fn get_opus_info(path: impl AsRef<Path>) -> Result<AudioInfo> {
        use ogg::reading::PacketReader;
        use std::fs::File;

        let file = File::open(path.as_ref())
            .map_err(|e| VoirsError::audio_error(format!("Failed to open Opus file: {e}")))?;

        let mut reader = PacketReader::new(file);
        let head = reader
            .read_packet_expected()
            .map_err(|e| VoirsError::audio_error(format!("Failed to read Opus header: {e:?}")))?;

        if head.data.len() < 19 || &head.data[..8] != b"OpusHead" {
            return Err(VoirsError::audio_error(
                "Not a valid Ogg Opus file — missing OpusHead magic",
            ));
        }
        let channels = head.data[9] as u32;
        let input_sample_rate =
            u32::from_le_bytes([head.data[12], head.data[13], head.data[14], head.data[15]]);
        let sample_rate = if input_sample_rate > 0 {
            input_sample_rate
        } else {
            48_000
        };

        Ok(AudioInfo {
            sample_rate,
            channels,
            duration: 0.0,
            sample_count: 0,
            format: AudioFormat::Opus,
        })
    }

    /// Stream audio to callback function (for real-time processing)
    pub fn stream_to_callback<F>(&self, chunk_size: usize, mut callback: F) -> Result<()>
    where
        F: FnMut(&[f32]) -> Result<()>,
    {
        if chunk_size == 0 {
            return Err(VoirsError::audio_error("Chunk size must be greater than 0"));
        }

        for chunk in self.samples.chunks(chunk_size) {
            callback(chunk)?;
        }

        Ok(())
    }

    /// Export audio metadata as JSON
    pub fn export_metadata(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| VoirsError::audio_error(format!("Failed to serialize metadata: {e}")))
    }

    /// Create audio buffer from raw bytes
    pub fn from_raw_bytes(
        bytes: &[u8],
        sample_rate: u32,
        channels: u32,
        format: RawFormat,
    ) -> Result<AudioBuffer> {
        let samples = match format {
            RawFormat::F32Le => {
                if !bytes.len().is_multiple_of(4) {
                    return Err(VoirsError::audio_error(
                        "Invalid byte length for F32 format",
                    ));
                }
                bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            }
            RawFormat::I16Le => {
                if !bytes.len().is_multiple_of(2) {
                    return Err(VoirsError::audio_error(
                        "Invalid byte length for I16 format",
                    ));
                }
                bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let val = i16::from_le_bytes([chunk[0], chunk[1]]);
                        val as f32 / 32767.0
                    })
                    .collect()
            }
            RawFormat::U8 => bytes
                .iter()
                .map(|&byte| (byte as f32 - 128.0) / 128.0)
                .collect(),
        };

        Ok(AudioBuffer::new(samples, sample_rate, channels))
    }
}

/// Audio file information
#[derive(Debug, Clone)]
pub struct AudioInfo {
    pub sample_rate: u32,
    pub channels: u32,
    pub duration: f32,
    pub sample_count: usize,
    pub format: AudioFormat,
}

/// Raw audio format for byte conversion
#[derive(Debug, Clone, Copy)]
pub enum RawFormat {
    F32Le, // 32-bit float little-endian
    I16Le, // 16-bit int little-endian
    U8,    // 8-bit unsigned
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::buffer::AudioBuffer;
    use tempfile::NamedTempFile;

    #[test]
    fn test_ogg_save_fails_closed_and_never_writes_a_corrupt_file() {
        let buffer = AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
        let temp_file = NamedTempFile::new().unwrap();
        // Use a path that does not exist yet so we can tell whether save_ogg created it.
        let target = temp_file.path().with_extension("ogg_target_does_not_exist");
        assert!(!target.exists());

        let result = buffer.save_ogg(&target);
        assert!(
            result.is_err(),
            "save_ogg must fail closed: no pure-Rust Vorbis encoder is available"
        );
        assert!(
            !target.exists(),
            "save_ogg must never write a file (corrupt or otherwise) when it cannot really encode"
        );
    }

    #[test]
    fn test_to_ogg_bytes_fails_closed() {
        let buffer = AudioBuffer::sine_wave(440.0, 0.1, 44100, 0.5);
        assert!(buffer.to_ogg_bytes().is_err());
    }

    #[test]
    fn test_load_ogg_rejects_non_ogg_data_with_real_parser() {
        // A real Ogg Vorbis parser (lewton) must honestly reject arbitrary bytes rather
        // than accept anything that merely starts with the "OggS" magic - this is what
        // makes the fail-closed `save_ogg` above safe (there is no way to construct a file
        // this crate will silently mis-parse as valid Ogg Vorbis).
        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), b"OggSnot a real ogg page at all, just noise").unwrap();

        assert!(AudioBuffer::load_ogg(temp_file.path()).is_err());
        assert!(AudioBuffer::get_ogg_info(temp_file.path()).is_err());
    }

    #[cfg(feature = "ffi-codecs")]
    #[test]
    fn test_mp3_round_trip_writes_real_data_at_the_exact_requested_path() {
        let buffer = AudioBuffer::sine_wave(440.0, 1.0, 22050, 0.5);
        let temp_file = NamedTempFile::new().unwrap();
        let target = temp_file.path().with_extension("mp3");

        buffer.save_mp3(&target).unwrap();

        // Must exist at the exact path requested - never silently redirected to a
        // differently-named .wav file.
        assert!(target.exists());
        let written = std::fs::read(&target).unwrap();
        // Real MP3 data starts with the frame sync word (0xFF Ex), not a RIFF/WAVE header.
        assert_eq!(written[0], 0xFF);
        assert_ne!(&written[0..4], b"RIFF");

        let _ = std::fs::remove_file(&target);
    }

    #[cfg(feature = "ffi-codecs")]
    #[test]
    fn test_mp3_bytes_length_varies_with_audio_duration() {
        // Direct regression test for the fabrication bug: the old implementation always
        // emitted a WAV byte-for-byte proportional only to sample count, mislabeled as
        // MP3; a real encoder must still produce output whose size scales with input.
        let short = AudioBuffer::sine_wave(440.0, 0.2, 22050, 0.5);
        let long = AudioBuffer::sine_wave(440.0, 2.0, 22050, 0.5);

        let short_bytes = short.to_mp3_bytes().unwrap();
        let long_bytes = long.to_mp3_bytes().unwrap();

        assert!(!short_bytes.is_empty());
        assert!(long_bytes.len() > short_bytes.len());
    }

    #[cfg(not(feature = "ffi-codecs"))]
    #[test]
    fn test_mp3_without_ffi_codecs_fails_closed_not_silent_wav() {
        let buffer = AudioBuffer::sine_wave(440.0, 0.1, 44100, 0.5);
        let temp_file = NamedTempFile::new().unwrap();
        let target = temp_file.path().with_extension("mp3_target_does_not_exist");
        assert!(!target.exists());

        assert!(buffer.save_mp3(&target).is_err());
        assert!(!target.exists());
        assert!(buffer.to_mp3_bytes().is_err());
    }

    #[test]
    fn test_wav_save_load() {
        let original = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
        let temp_file = NamedTempFile::new().unwrap();

        // Save as WAV
        original.save_wav(temp_file.path()).unwrap();

        // Load back
        let loaded = AudioBuffer::load_wav(temp_file.path()).unwrap();

        assert_eq!(loaded.sample_rate(), original.sample_rate());
        assert_eq!(loaded.channels(), original.channels());
        assert!((loaded.duration() - original.duration()).abs() < 0.01);
    }

    #[test]
    fn test_wav_bytes_conversion() {
        let buffer = AudioBuffer::sine_wave(440.0, 0.1, 44100, 0.5);

        let wav_bytes = buffer.to_wav_bytes().unwrap();

        // WAV file should have a header, so bytes should be larger than raw samples
        assert!(wav_bytes.len() > buffer.len() * 2); // 2 bytes per sample for 16-bit
    }

    #[test]
    fn test_wav_info() {
        let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
        let temp_file = NamedTempFile::new().unwrap();

        buffer.save_wav(temp_file.path()).unwrap();

        let info = AudioBuffer::get_wav_info(temp_file.path()).unwrap();

        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 1);
        assert!((info.duration - 1.0).abs() < 0.01);
        assert_eq!(info.format, AudioFormat::Wav);
    }

    #[test]
    fn test_stream_to_callback() {
        let buffer = AudioBuffer::sine_wave(440.0, 0.1, 44100, 0.5);
        let chunk_size = 1024;
        let mut total_samples = 0;

        buffer
            .stream_to_callback(chunk_size, |chunk| {
                total_samples += chunk.len();
                Ok(())
            })
            .unwrap();

        assert_eq!(total_samples, buffer.len());
    }

    #[test]
    fn test_metadata_export() {
        let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);

        let metadata_json = buffer.export_metadata().unwrap();

        // Should be valid JSON
        assert!(metadata_json.contains("duration"));
        assert!(metadata_json.contains("peak_amplitude"));
        assert!(!metadata_json.contains("sample_rate")); // metadata doesn't include sample_rate
    }

    #[test]
    fn test_raw_bytes_conversion() {
        // Create test data
        let samples = vec![0.0, 0.5, -0.5, 1.0];
        let original = AudioBuffer::mono(samples, 44100);

        // Convert to bytes and back
        let _bytes = original.to_wav_bytes().unwrap();

        // For a more direct test, let's test F32 raw format
        let f32_bytes: Vec<u8> = original
            .samples()
            .iter()
            .flat_map(|&sample| sample.to_le_bytes())
            .collect();

        let reconstructed =
            AudioBuffer::from_raw_bytes(&f32_bytes, 44100, 1, RawFormat::F32Le).unwrap();

        assert_eq!(reconstructed.sample_rate(), original.sample_rate());
        assert_eq!(reconstructed.channels(), original.channels());
        assert_eq!(reconstructed.samples().len(), original.samples().len());
    }

    #[test]
    fn test_f32_wav_save() {
        let buffer = AudioBuffer::sine_wave(440.0, 0.1, 44100, 0.5);
        let temp_file = NamedTempFile::new().unwrap();

        // Save as 32-bit float WAV
        buffer.save_wav_f32(temp_file.path()).unwrap();

        // Load back
        let loaded = AudioBuffer::load_wav(temp_file.path()).unwrap();

        assert_eq!(loaded.sample_rate(), buffer.sample_rate());
        assert_eq!(loaded.channels(), buffer.channels());
        assert!((loaded.duration() - buffer.duration()).abs() < 0.01);
    }

    #[test]
    fn test_play_with_callback() {
        use std::sync::{Arc, Mutex};

        let buffer = AudioBuffer::sine_wave(440.0, 0.1, 44100, 0.5);
        let progress_updates = Arc::new(Mutex::new(0));
        let progress_updates_clone = progress_updates.clone();

        buffer
            .play_with_callback(move |progress| {
                let mut count = progress_updates_clone
                    .lock()
                    .expect("lock should not be poisoned");
                *count += 1;
                assert!((0.0..=1.0).contains(&progress));
            })
            .unwrap();

        let final_count = *progress_updates
            .lock()
            .expect("lock should not be poisoned");
        assert!(final_count > 0); // Should have at least some progress updates
    }
}
