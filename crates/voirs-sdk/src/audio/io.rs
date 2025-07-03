//! Audio I/O operations for saving, loading, and format conversion.

use crate::{error::Result, types::AudioFormat, VoirsError};
use super::buffer::AudioBuffer;
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

        let mut writer = WavWriter::create(path, spec).map_err(|e| {
            VoirsError::audio_error(format!("Failed to create WAV writer: {}", e))
        })?;

        // Convert f32 samples to i16
        for &sample in &self.samples {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer.write_sample(sample_i16).map_err(|e| {
                VoirsError::audio_error(format!("Failed to write sample: {}", e))
            })?;
        }

        writer.finalize().map_err(|e| {
            VoirsError::audio_error(format!("Failed to finalize WAV file: {}", e))
        })?;

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

        let mut writer = WavWriter::create(path, spec).map_err(|e| {
            VoirsError::audio_error(format!("Failed to create WAV writer: {}", e))
        })?;

        // Write f32 samples directly
        for &sample in &self.samples {
            writer.write_sample(sample.clamp(-1.0, 1.0)).map_err(|e| {
                VoirsError::audio_error(format!("Failed to write sample: {}", e))
            })?;
        }

        writer.finalize().map_err(|e| {
            VoirsError::audio_error(format!("Failed to finalize WAV file: {}", e))
        })?;

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

    /// Save audio as FLAC file (placeholder)
    pub fn save_flac(&self, path: impl AsRef<Path>) -> Result<()> {
        // TODO: Implement FLAC saving
        tracing::warn!("FLAC saving not yet implemented, falling back to WAV");
        self.save_wav(path)
    }

    /// Save audio as MP3 file (placeholder)
    pub fn save_mp3(&self, path: impl AsRef<Path>) -> Result<()> {
        // TODO: Implement MP3 saving using lame or similar
        tracing::warn!("MP3 saving not yet implemented, falling back to WAV");
        self.save_wav(path)
    }

    /// Save audio as OGG file (placeholder)
    pub fn save_ogg(&self, path: impl AsRef<Path>) -> Result<()> {
        // TODO: Implement OGG saving using vorbis encoder
        tracing::warn!("OGG saving not yet implemented, falling back to WAV");
        self.save_wav(path)
    }

    /// Save audio as Opus file (placeholder)
    pub fn save_opus(&self, path: impl AsRef<Path>) -> Result<()> {
        // TODO: Implement Opus saving
        tracing::warn!("Opus saving not yet implemented, falling back to WAV");
        self.save_wav(path)
    }

    /// Play audio through system speakers (placeholder)
    pub fn play(&self) -> Result<()> {
        // TODO: Implement audio playback
        // This would use cpal or similar for cross-platform audio output
        tracing::info!(
            "Playing audio: {:.2}s @ {}Hz (implementation pending)",
            self.duration(),
            self.sample_rate
        );
        Ok(())
    }

    /// Play audio with callback for progress updates
    pub fn play_with_callback<F>(&self, mut callback: F) -> Result<()>
    where
        F: FnMut(f32), // Progress callback (0.0 to 1.0)
    {
        // TODO: Implement audio playback with progress callback
        // Simulate progress for now
        let total_samples = self.samples.len();
        let chunk_size = total_samples / 10; // 10 progress updates
        
        for i in 0..10 {
            let progress = (i + 1) as f32 / 10.0;
            callback(progress);
            
            // Simulate playback delay
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        
        tracing::info!(
            "Simulated audio playback: {:.2}s @ {}Hz",
            self.duration(),
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
                VoirsError::audio_error(format!("Failed to create WAV writer: {}", e))
            })?;

            // Convert f32 samples to i16
            for &sample in &self.samples {
                let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                writer.write_sample(sample_i16).map_err(|e| {
                    VoirsError::audio_error(format!("Failed to write sample: {}", e))
                })?;
            }

            writer.finalize().map_err(|e| {
                VoirsError::audio_error(format!("Failed to finalize WAV: {}", e))
            })?;
        }

        Ok(cursor.into_inner())
    }

    /// Convert to FLAC bytes (placeholder)
    pub fn to_flac_bytes(&self) -> Result<Vec<u8>> {
        // TODO: Implement FLAC conversion
        tracing::warn!("FLAC conversion not yet implemented, falling back to WAV");
        self.to_wav_bytes()
    }

    /// Convert to MP3 bytes (placeholder)
    pub fn to_mp3_bytes(&self) -> Result<Vec<u8>> {
        // TODO: Implement MP3 conversion using lame or similar
        tracing::warn!("MP3 conversion not yet implemented, falling back to WAV");
        self.to_wav_bytes()
    }

    /// Convert to OGG bytes (placeholder)
    pub fn to_ogg_bytes(&self) -> Result<Vec<u8>> {
        // TODO: Implement OGG conversion using vorbis encoder
        tracing::warn!("OGG conversion not yet implemented, falling back to WAV");
        self.to_wav_bytes()
    }

    /// Convert to Opus bytes (placeholder)
    pub fn to_opus_bytes(&self) -> Result<Vec<u8>> {
        // TODO: Implement Opus conversion
        tracing::warn!("Opus conversion not yet implemented, falling back to WAV");
        self.to_wav_bytes()
    }

    /// Load audio from WAV file
    pub fn load_wav(path: impl AsRef<Path>) -> Result<AudioBuffer> {
        use hound::WavReader;

        let mut reader = WavReader::open(path).map_err(|e| {
            VoirsError::audio_error(format!("Failed to open WAV file: {}", e))
        })?;

        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let channels = spec.channels as u32;

        let samples: Result<Vec<f32>> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.samples::<f32>()
                    .map(|s| s.map_err(|e| VoirsError::audio_error(format!("Failed to read sample: {}", e))))
                    .collect()
            }
            hound::SampleFormat::Int => {
                match spec.bits_per_sample {
                    16 => {
                        reader.samples::<i16>()
                            .map(|s| s.map(|sample| sample as f32 / 32767.0)
                                .map_err(|e| VoirsError::audio_error(format!("Failed to read sample: {}", e))))
                            .collect()
                    }
                    24 => {
                        reader.samples::<i32>()
                            .map(|s| s.map(|sample| sample as f32 / 8388607.0)
                                .map_err(|e| VoirsError::audio_error(format!("Failed to read sample: {}", e))))
                            .collect()
                    }
                    32 => {
                        reader.samples::<i32>()
                            .map(|s| s.map(|sample| sample as f32 / 2147483647.0)
                                .map_err(|e| VoirsError::audio_error(format!("Failed to read sample: {}", e))))
                            .collect()
                    }
                    _ => return Err(VoirsError::audio_error(format!(
                        "Unsupported bit depth: {}", spec.bits_per_sample
                    ))),
                }
            }
        };

        let samples = samples?;
        Ok(AudioBuffer::new(samples, sample_rate, channels))
    }

    /// Load audio from file (auto-detect format)
    pub fn load(path: impl AsRef<Path>) -> Result<AudioBuffer> {
        let path = path.as_ref();
        let extension = path.extension()
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
                "Unsupported audio format: {}", extension
            ))),
        }
    }

    /// Load audio from FLAC file (placeholder)
    pub fn load_flac(path: impl AsRef<Path>) -> Result<AudioBuffer> {
        // TODO: Implement FLAC loading
        tracing::warn!("FLAC loading not yet implemented");
        Err(VoirsError::audio_error("FLAC loading not implemented"))
    }

    /// Load audio from MP3 file (placeholder)
    pub fn load_mp3(path: impl AsRef<Path>) -> Result<AudioBuffer> {
        // TODO: Implement MP3 loading using minimp3 or similar
        tracing::warn!("MP3 loading not yet implemented");
        Err(VoirsError::audio_error("MP3 loading not implemented"))
    }

    /// Load audio from OGG file (placeholder)
    pub fn load_ogg(path: impl AsRef<Path>) -> Result<AudioBuffer> {
        // TODO: Implement OGG loading using vorbis decoder
        tracing::warn!("OGG loading not yet implemented");
        Err(VoirsError::audio_error("OGG loading not implemented"))
    }

    /// Load audio from Opus file (placeholder)
    pub fn load_opus(path: impl AsRef<Path>) -> Result<AudioBuffer> {
        // TODO: Implement Opus loading
        tracing::warn!("Opus loading not yet implemented");
        Err(VoirsError::audio_error("Opus loading not implemented"))
    }

    /// Get audio information without loading samples
    pub fn get_info(path: impl AsRef<Path>) -> Result<AudioInfo> {
        let path = path.as_ref();
        let extension = path.extension()
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
                "Unsupported audio format: {}", extension
            ))),
        }
    }

    /// Get WAV file information
    pub fn get_wav_info(path: impl AsRef<Path>) -> Result<AudioInfo> {
        use hound::WavReader;

        let reader = WavReader::open(path).map_err(|e| {
            VoirsError::audio_error(format!("Failed to open WAV file: {}", e))
        })?;

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

    /// Get FLAC file information (placeholder)
    pub fn get_flac_info(path: impl AsRef<Path>) -> Result<AudioInfo> {
        // TODO: Implement FLAC info reading
        tracing::warn!("FLAC info reading not yet implemented");
        Err(VoirsError::audio_error("FLAC info reading not implemented"))
    }

    /// Get MP3 file information (placeholder)
    pub fn get_mp3_info(path: impl AsRef<Path>) -> Result<AudioInfo> {
        // TODO: Implement MP3 info reading
        tracing::warn!("MP3 info reading not yet implemented");
        Err(VoirsError::audio_error("MP3 info reading not implemented"))
    }

    /// Get OGG file information (placeholder)
    pub fn get_ogg_info(path: impl AsRef<Path>) -> Result<AudioInfo> {
        // TODO: Implement OGG info reading
        tracing::warn!("OGG info reading not yet implemented");
        Err(VoirsError::audio_error("OGG info reading not implemented"))
    }

    /// Get Opus file information (placeholder)
    pub fn get_opus_info(path: impl AsRef<Path>) -> Result<AudioInfo> {
        // TODO: Implement Opus info reading
        tracing::warn!("Opus info reading not yet implemented");
        Err(VoirsError::audio_error("Opus info reading not implemented"))
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
            .map_err(|e| VoirsError::audio_error(format!("Failed to serialize metadata: {}", e)))
    }

    /// Create audio buffer from raw bytes
    pub fn from_raw_bytes(bytes: &[u8], sample_rate: u32, channels: u32, format: RawFormat) -> Result<AudioBuffer> {
        let samples = match format {
            RawFormat::F32Le => {
                if bytes.len() % 4 != 0 {
                    return Err(VoirsError::audio_error("Invalid byte length for F32 format"));
                }
                bytes.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            }
            RawFormat::I16Le => {
                if bytes.len() % 2 != 0 {
                    return Err(VoirsError::audio_error("Invalid byte length for I16 format"));
                }
                bytes.chunks_exact(2)
                    .map(|chunk| {
                        let val = i16::from_le_bytes([chunk[0], chunk[1]]);
                        val as f32 / 32767.0
                    })
                    .collect()
            }
            RawFormat::U8 => {
                bytes.iter()
                    .map(|&byte| (byte as f32 - 128.0) / 128.0)
                    .collect()
            }
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
    F32Le,  // 32-bit float little-endian
    I16Le,  // 16-bit int little-endian
    U8,     // 8-bit unsigned
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
        
        buffer.stream_to_callback(chunk_size, |chunk| {
            total_samples += chunk.len();
            Ok(())
        }).unwrap();
        
        assert_eq!(total_samples, buffer.len());
    }

    #[test]
    fn test_metadata_export() {
        let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
        
        let metadata_json = buffer.export_metadata().unwrap();
        
        // Should be valid JSON
        assert!(metadata_json.contains("duration"));
        assert!(metadata_json.contains("peak_amplitude"));
        assert!(metadata_json.contains("sample_rate") == false); // metadata doesn't include sample_rate
    }

    #[test]
    fn test_raw_bytes_conversion() {
        // Create test data
        let samples = vec![0.0, 0.5, -0.5, 1.0];
        let original = AudioBuffer::mono(samples, 44100);
        
        // Convert to bytes and back
        let bytes = original.to_wav_bytes().unwrap();
        
        // For a more direct test, let's test F32 raw format
        let f32_bytes: Vec<u8> = original.samples().iter()
            .flat_map(|&sample| sample.to_le_bytes())
            .collect();
        
        let reconstructed = AudioBuffer::from_raw_bytes(&f32_bytes, 44100, 1, RawFormat::F32Le).unwrap();
        
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
        let buffer = AudioBuffer::sine_wave(440.0, 0.1, 44100, 0.5);
        let mut progress_updates = 0;
        
        buffer.play_with_callback(|progress| {
            progress_updates += 1;
            assert!(progress >= 0.0 && progress <= 1.0);
        }).unwrap();
        
        assert_eq!(progress_updates, 10); // Should have 10 progress updates
    }
}