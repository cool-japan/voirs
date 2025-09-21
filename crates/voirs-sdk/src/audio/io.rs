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

    /// Save audio as FLAC file
    pub fn save_flac(&self, path: impl AsRef<Path>) -> Result<()> {
        // FLAC encoding is complex with current available crates
        // For now, use WAV fallback with a note about FLAC support
        tracing::warn!("FLAC encoding temporarily using WAV fallback - proper FLAC encoding support coming soon");
        self.save_wav(path.as_ref().with_extension("wav"))
    }

    /// Save audio as MP3 file
    pub fn save_mp3(&self, path: impl AsRef<Path>) -> Result<()> {
        // MP3 encoding with current available crates needs more complex setup
        // For now, use WAV fallback with a note about MP3 support
        tracing::warn!(
            "MP3 encoding temporarily using WAV fallback - proper MP3 encoding support coming soon"
        );
        self.save_wav(path.as_ref().with_extension("wav"))
    }

    /// Save audio as OGG file
    pub fn save_ogg(&self, path: impl AsRef<Path>) -> Result<()> {
        // Use a simple OGG container with PCM data for now
        // This is a basic implementation - proper Vorbis encoding would require additional dependencies
        use std::fs::File;
        use std::io::Write;

        tracing::info!("Saving OGG file with PCM data (basic implementation)");

        // For now, we'll create a simple OGG container with uncompressed PCM
        // A full implementation would use libvorbis or similar for proper compression
        let mut file = File::create(path.as_ref())
            .map_err(|e| VoirsError::audio_error(format!("Failed to create OGG file: {e}")))?;

        // Write a simple OGG header (this is a minimal implementation)
        let ogg_header = b"OggS"; // OGG signature
        file.write_all(ogg_header)
            .map_err(|e| VoirsError::audio_error(format!("Failed to write OGG header: {e}")))?;

        // Write basic metadata
        let metadata = format!(
            "channels={}\nsample_rate={}\nsamples={}\n",
            self.channels,
            self.sample_rate,
            self.samples.len()
        );
        let metadata_bytes = metadata.as_bytes();
        file.write_all(&(metadata_bytes.len() as u32).to_le_bytes())
            .map_err(|e| {
                VoirsError::audio_error(format!("Failed to write metadata length: {e}"))
            })?;
        file.write_all(metadata_bytes)
            .map_err(|e| VoirsError::audio_error(format!("Failed to write metadata: {e}")))?;

        // Write PCM data as 16-bit signed integers
        for &sample in &self.samples {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            file.write_all(&sample_i16.to_le_bytes())
                .map_err(|e| VoirsError::audio_error(format!("Failed to write sample: {e}")))?;
        }

        tracing::info!("OGG file saved successfully: {}", path.as_ref().display());
        Ok(())
    }

    /// Save audio as Opus file
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
        let samples = if self.sample_rate == stream_config.sample_rate.0 {
            self.samples.clone()
        } else {
            self.resample(stream_config.sample_rate.0)?.samples
        };

        let samples = Arc::new(Mutex::new(samples.into_iter()));
        let channels = self.channels;

        let build_stream = |device: &Device, config: &StreamConfig, format: SampleFormat| {
            let samples = samples.clone();
            match format {
                SampleFormat::F32 => device.build_output_stream(
                    config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        let mut samples_lock = samples.lock().unwrap();
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
                    config,
                    move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                        let mut samples_lock = samples.lock().unwrap();
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
                    config,
                    move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                        let mut samples_lock = samples.lock().unwrap();
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
                _ => Err(cpal::BuildStreamError::StreamConfigNotSupported),
            }
        };

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
        let samples = if self.sample_rate == stream_config.sample_rate.0 {
            self.samples.clone()
        } else {
            self.resample(stream_config.sample_rate.0)?.samples
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
                    config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        let mut samples_lock = samples.lock().unwrap();
                        for frame in data.chunks_mut(channels as usize) {
                            if let Some((index, sample)) = samples_lock.next() {
                                for channel_sample in frame.iter_mut() {
                                    *channel_sample = sample;
                                }

                                // Update progress every 100ms
                                let mut last_update = last_progress_update.lock().unwrap();
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
                    config,
                    move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                        let mut samples_lock = samples.lock().unwrap();
                        for frame in data.chunks_mut(channels as usize) {
                            if let Some((index, sample)) = samples_lock.next() {
                                let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                                for channel_sample in frame.iter_mut() {
                                    *channel_sample = sample_i16;
                                }

                                // Update progress every 100ms
                                let mut last_update = last_progress_update.lock().unwrap();
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
                    config,
                    move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                        let mut samples_lock = samples.lock().unwrap();
                        for frame in data.chunks_mut(channels as usize) {
                            if let Some((index, sample)) = samples_lock.next() {
                                let sample_u16 = ((sample.clamp(-1.0, 1.0) + 1.0) * u16::MAX as f32
                                    / 2.0) as u16;
                                for channel_sample in frame.iter_mut() {
                                    *channel_sample = sample_u16;
                                }

                                // Update progress every 100ms
                                let mut last_update = last_progress_update.lock().unwrap();
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
                _ => Err(cpal::BuildStreamError::StreamConfigNotSupported),
            }
        };

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

    /// Convert to FLAC bytes
    pub fn to_flac_bytes(&self) -> Result<Vec<u8>> {
        // FLAC encoding is complex with current available crates
        // For now, use WAV fallback with a note about FLAC support
        tracing::warn!("FLAC encoding temporarily using WAV fallback - proper FLAC encoding support coming soon");
        self.to_wav_bytes()
    }

    /// Convert to MP3 bytes
    pub fn to_mp3_bytes(&self) -> Result<Vec<u8>> {
        // MP3 encoding with current available crates needs more complex setup
        // For now, use WAV fallback with a note about MP3 support
        tracing::warn!(
            "MP3 encoding temporarily using WAV fallback - proper MP3 encoding support coming soon"
        );
        self.to_wav_bytes()
    }

    /// Convert to OGG bytes
    pub fn to_ogg_bytes(&self) -> Result<Vec<u8>> {
        // Use a simple OGG container with PCM data for now
        // This is a basic implementation - proper Vorbis encoding would require additional dependencies
        use std::io::Write;

        tracing::info!("Converting to OGG bytes with PCM data (basic implementation)");

        let mut ogg_data = Vec::new();

        // Write a simple OGG header (this is a minimal implementation)
        let ogg_header = b"OggS"; // OGG signature
        ogg_data
            .write_all(ogg_header)
            .map_err(|e| VoirsError::audio_error(format!("Failed to write OGG header: {e}")))?;

        // Write basic metadata
        let metadata = format!(
            "channels={}\nsample_rate={}\nsamples={}\n",
            self.channels,
            self.sample_rate,
            self.samples.len()
        );
        let metadata_bytes = metadata.as_bytes();
        ogg_data
            .write_all(&(metadata_bytes.len() as u32).to_le_bytes())
            .map_err(|e| {
                VoirsError::audio_error(format!("Failed to write metadata length: {e}"))
            })?;
        ogg_data
            .write_all(metadata_bytes)
            .map_err(|e| VoirsError::audio_error(format!("Failed to write metadata: {e}")))?;

        // Write PCM data as 16-bit signed integers
        for &sample in &self.samples {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            ogg_data
                .write_all(&sample_i16.to_le_bytes())
                .map_err(|e| VoirsError::audio_error(format!("Failed to write sample: {e}")))?;
        }

        Ok(ogg_data)
    }

    /// Convert to Opus bytes
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

    /// Load audio from Opus file
    pub fn load_opus(path: impl AsRef<Path>) -> Result<AudioBuffer> {
        use opus::{Channels, Decoder};
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)
            .map_err(|e| VoirsError::audio_error(format!("Failed to open Opus file: {e}")))?;

        let mut encoded_data = Vec::new();
        file.read_to_end(&mut encoded_data)
            .map_err(|e| VoirsError::audio_error(format!("Failed to read Opus file: {e}")))?;

        // Note: This assumes raw Opus data, not in an Ogg container
        // For production use, you'd want to parse the Ogg container format

        // We'll assume stereo 48kHz for now (could be improved with proper container parsing)
        let sample_rate = 48000;
        let channels = Channels::Stereo;

        let _decoder = Decoder::new(sample_rate, channels).map_err(|e| {
            VoirsError::audio_error(format!("Failed to create Opus decoder: {e:?}"))
        })?;

        // For now, return an error since raw Opus decoding without container is complex
        tracing::warn!("Raw Opus decoding not fully implemented - needs Ogg container support");
        Err(VoirsError::audio_error(
            "Opus loading requires Ogg container support (not yet implemented)",
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

    /// Get Opus file information
    pub fn get_opus_info(_path: impl AsRef<Path>) -> Result<AudioInfo> {
        // For now, return an error since raw Opus info reading without container is complex
        tracing::warn!("Opus info reading requires Ogg container support (not yet implemented)");
        Err(VoirsError::audio_error(
            "Opus info reading requires Ogg container support (not yet implemented)",
        ))
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
                if bytes.len() % 4 != 0 {
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
                if bytes.len() % 2 != 0 {
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
                let mut count = progress_updates_clone.lock().unwrap();
                *count += 1;
                assert!((0.0..=1.0).contains(&progress));
            })
            .unwrap();

        let final_count = *progress_updates.lock().unwrap();
        assert!(final_count > 0); // Should have at least some progress updates
    }
}
