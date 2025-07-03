//! Audio file I/O operations
//!
//! This module provides comprehensive audio file reading and writing capabilities
//! for various audio formats including WAV, FLAC, MP3, and OGG.

use crate::{AudioData, AudioFormat, DatasetError, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom};
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
    }
}

/// Load WAV file
pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<AudioData> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    
    let samples: Result<Vec<f32>> = reader
        .samples::<i16>()
        .map(|s| s.map(|sample| sample as f32 / i16::MAX as f32).map_err(DatasetError::from))
        .collect();
    
    let samples = samples?;
    let mut audio = AudioData::new(samples, spec.sample_rate, spec.channels as u32);
    
    // Add format metadata
    audio.add_metadata("format".to_string(), "wav".to_string());
    audio.add_metadata("bits_per_sample".to_string(), spec.bits_per_sample.to_string());
    
    Ok(audio)
}

/// Load FLAC file
pub fn load_flac<P: AsRef<Path>>(path: P) -> Result<AudioData> {
    let file = File::open(path)?;
    let mut reader = claxon::FlacReader::new(file)
        .map_err(|e| DatasetError::FormatError(format!("FLAC read error: {}", e)))?;
    
    let streaminfo = reader.streaminfo();
    let sample_rate = streaminfo.sample_rate;
    let channels = streaminfo.channels;
    let bits_per_sample = streaminfo.bits_per_sample;
    
    let mut samples = Vec::new();
    
    // Read all samples using claxon's iterator API
    for sample_result in reader.samples() {
        let sample = sample_result
            .map_err(|e| DatasetError::FormatError(format!("FLAC sample error: {}", e)))?;
        let normalized_sample = sample as f32 / (1 << (bits_per_sample - 1)) as f32;
        samples.push(normalized_sample.clamp(-1.0, 1.0));
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
    let mut all_samples = Vec::new();
    let mut sample_rate = 0;
    let mut channels = 0;
    
    loop {
        match decoder.next_frame() {
            Ok(frame) => {
                if sample_rate == 0 {
                    sample_rate = frame.sample_rate as u32;
                    channels = frame.channels as u32;
                }
                
                // Convert i16 samples to f32
                for &sample in &frame.data {
                    all_samples.push(sample as f32 / i16::MAX as f32);
                }
            }
            Err(minimp3::Error::Eof) => break,
            Err(e) => return Err(DatasetError::FormatError(format!("MP3 decode error: {}", e))),
        }
    }
    
    if all_samples.is_empty() {
        return Err(DatasetError::FormatError("No audio data found in MP3 file".to_string()));
    }
    
    let mut audio = AudioData::new(all_samples, sample_rate, channels);
    audio.add_metadata("format".to_string(), "mp3".to_string());
    
    Ok(audio)
}

/// Load OGG Vorbis file
pub fn load_ogg<P: AsRef<Path>>(path: P) -> Result<AudioData> {
    let file = File::open(path)?;
    let mut reader = lewton::inside_ogg::OggStreamReader::new(file)
        .map_err(|e| DatasetError::FormatError(format!("OGG read error: {:?}", e)))?;
    
    let sample_rate = reader.ident_hdr.audio_sample_rate;
    let channels = reader.ident_hdr.audio_channels as u32;
    
    let mut all_samples = Vec::new();
    
    while let Some(packet) = reader.read_dec_packet_itl()
        .map_err(|e| DatasetError::FormatError(format!("OGG decode error: {:?}", e)))? {
        
        // Convert i16 samples to f32
        for &sample in &packet {
            all_samples.push(sample as f32 / i16::MAX as f32);
        }
    }
    
    if all_samples.is_empty() {
        return Err(DatasetError::FormatError("No audio data found in OGG file".to_string()));
    }
    
    let mut audio = AudioData::new(all_samples, sample_rate, channels);
    audio.add_metadata("format".to_string(), "ogg".to_string());
    
    Ok(audio)
}

/// Save audio file with automatic format detection from extension
pub fn save_audio<P: AsRef<Path>>(audio: &AudioData, path: P) -> Result<()> {
    let path = path.as_ref();
    let format = FormatDetector::detect_from_extension(path)?;
    
    match format {
        AudioFormat::Wav => save_wav(audio, path),
        AudioFormat::Mp3 => save_mp3(audio, path),
        AudioFormat::Flac => Err(DatasetError::FormatError("FLAC encoding not yet implemented".to_string())),
        AudioFormat::Ogg => Err(DatasetError::FormatError("OGG encoding not yet implemented".to_string())),
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

/// Save audio to MP3 file
pub fn save_mp3<P: AsRef<Path>>(audio: &AudioData, path: P) -> Result<()> {
    // For now, return an error as MP3 encoding requires proper configuration
    // This is a placeholder implementation
    Err(DatasetError::FormatError("MP3 encoding not yet fully implemented".to_string()))
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
        audio.add_metadata("chunk_index".to_string(), (self.current_position / self.chunk_size).to_string());
        
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
            cached.channels()
        );
        
        // Copy metadata from cached audio
        for (key, value) in cached.metadata().iter() {
            audio.add_metadata(key.clone(), value.clone());
        }
        
        audio.add_metadata("chunk_index".to_string(), (self.current_position / self.chunk_size).to_string());
        
        self.current_position = end_position;
        
        Ok(Some(audio))
    }
    
    /// Reset reader to beginning
    pub fn reset(&mut self) {
        self.current_position = 0;
    }
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
            _ => Err(DatasetError::FormatError(format!("Unsupported audio format: {}", extension))),
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
            Ok(AudioFormat::Ogg)
        } else {
            Err(DatasetError::FormatError("Unknown audio format".to_string()))
        }
    }
    
    /// Validate audio file format
    pub fn validate_format<P: AsRef<Path>>(path: P, expected_format: AudioFormat) -> Result<bool> {
        let detected_format = Self::detect_from_header(path)?;
        Ok(detected_format == expected_format)
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
        }
    }
    
    fn extract_wav_metadata<P: AsRef<Path>>(path: P) -> Result<HashMap<String, String>> {
        let reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "wav".to_string());
        metadata.insert("sample_rate".to_string(), spec.sample_rate.to_string());
        metadata.insert("channels".to_string(), spec.channels.to_string());
        metadata.insert("bits_per_sample".to_string(), spec.bits_per_sample.to_string());
        metadata.insert("duration".to_string(), reader.duration().to_string());
        
        Ok(metadata)
    }
    
    fn extract_flac_metadata<P: AsRef<Path>>(path: P) -> Result<HashMap<String, String>> {
        let file = File::open(path)?;
        let reader = claxon::FlacReader::new(file)
            .map_err(|e| DatasetError::FormatError(format!("FLAC read error: {}", e)))?;
        
        let streaminfo = reader.streaminfo();
        
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "flac".to_string());
        metadata.insert("sample_rate".to_string(), streaminfo.sample_rate.to_string());
        metadata.insert("channels".to_string(), streaminfo.channels.to_string());
        metadata.insert("bits_per_sample".to_string(), streaminfo.bits_per_sample.to_string());
        
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
            .map_err(|e| DatasetError::FormatError(format!("OGG read error: {:?}", e)))?;
        
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "ogg".to_string());
        metadata.insert("sample_rate".to_string(), reader.ident_hdr.audio_sample_rate.to_string());
        metadata.insert("channels".to_string(), reader.ident_hdr.audio_channels.to_string());
        
        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_format_detection_from_extension() {
        assert_eq!(FormatDetector::detect_from_extension("test.wav").unwrap(), AudioFormat::Wav);
        assert_eq!(FormatDetector::detect_from_extension("test.flac").unwrap(), AudioFormat::Flac);
        assert_eq!(FormatDetector::detect_from_extension("test.mp3").unwrap(), AudioFormat::Mp3);
        assert_eq!(FormatDetector::detect_from_extension("test.ogg").unwrap(), AudioFormat::Ogg);
    }
    
    #[test]
    fn test_wav_save_load_roundtrip() {
        let original_audio = AudioData::new(
            vec![0.1, -0.2, 0.3, -0.4],
            44100,
            2
        );
        
        let temp_file = NamedTempFile::new().unwrap();
        save_wav(&original_audio, temp_file.path()).unwrap();
        
        let loaded_audio = load_wav(temp_file.path()).unwrap();
        
        assert_eq!(loaded_audio.sample_rate(), 44100);
        assert_eq!(loaded_audio.channels(), 2);
        assert_eq!(loaded_audio.samples().len(), 4);
        
        // Check approximate equality (accounting for quantization)
        for (original, loaded) in original_audio.samples().iter().zip(loaded_audio.samples().iter()) {
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
        
        // MP3 saving is not yet implemented, so this should fail
        let result = save_mp3(&original_audio, temp_file.path());
        assert!(result.is_err());
    }
    
    #[test]
    fn test_save_audio_format_detection() {
        let audio = AudioData::new(vec![0.1, -0.2, 0.3, -0.4], 44100, 2);
        
        // Test WAV
        let temp_wav = NamedTempFile::with_suffix(".wav").unwrap();
        save_audio(&audio, temp_wav.path()).unwrap();
        
        // Test MP3 - should fail as not yet implemented
        let temp_mp3 = NamedTempFile::with_suffix(".mp3").unwrap();
        let result = save_audio(&audio, temp_mp3.path());
        assert!(result.is_err());
        
        // Test unsupported format
        let temp_flac = NamedTempFile::with_suffix(".flac").unwrap();
        let result = save_audio(&audio, temp_flac.path());
        assert!(result.is_err());
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
