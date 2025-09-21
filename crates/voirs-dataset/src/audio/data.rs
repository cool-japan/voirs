//! Audio data structures and utilities
//!
//! This module provides enhanced audio data structures with efficient
//! memory management and processing capabilities.

use crate::{AudioData, DatasetError, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Memory-mapped audio data for large files
///
/// This provides efficient access to large audio files without loading
/// them entirely into memory. Supports raw PCM data in f32 format.
pub struct MemoryMappedAudio {
    /// Memory-mapped file
    mmap: Mmap,
    /// Sample rate of the audio
    sample_rate: u32,
    /// Number of channels
    channels: u32,
    /// Total number of samples
    total_samples: usize,
    /// File path for metadata
    file_path: std::path::PathBuf,
}

impl MemoryMappedAudio {
    /// Create memory-mapped audio from raw PCM file
    ///
    /// # Arguments
    /// * `path` - Path to the raw PCM file (f32 samples)
    /// * `sample_rate` - Sample rate of the audio
    /// * `channels` - Number of channels
    ///
    /// # Examples
    /// ```no_run
    /// use voirs_dataset::audio::data::MemoryMappedAudio;
    ///
    /// let mmap_audio = MemoryMappedAudio::from_file("audio.raw", 44100, 2)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P, sample_rate: u32, channels: u32) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Validate that file size is compatible with f32 samples
        if mmap.len() % std::mem::size_of::<f32>() != 0 {
            return Err(DatasetError::FormatError(
                "File size is not compatible with f32 samples".to_string(),
            ));
        }

        let total_samples = mmap.len() / std::mem::size_of::<f32>();

        // Validate that samples are compatible with channel count
        if total_samples % channels as usize != 0 {
            return Err(DatasetError::FormatError(format!(
                "Total samples ({total_samples}) not divisible by channel count ({channels})"
            )));
        }

        Ok(Self {
            mmap,
            sample_rate,
            channels,
            total_samples,
            file_path: path.to_path_buf(),
        })
    }

    /// Create memory-mapped audio from WAV file header analysis
    ///
    /// This analyzes a WAV file and creates a memory-mapped view of its data section.
    pub fn from_wav_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // First, read WAV header to get metadata
        let reader = hound::WavReader::open(path)?;
        let spec = reader.spec();

        if spec.sample_format != hound::SampleFormat::Float || spec.bits_per_sample != 32 {
            return Err(DatasetError::FormatError(
                "Only 32-bit float WAV files are supported for memory mapping".to_string(),
            ));
        }

        // Memory-map the entire file
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Find the data chunk (simplified approach - assumes standard WAV structure)
        let data_offset = Self::find_wav_data_offset(&mmap)?;
        let data_size = mmap.len() - data_offset;

        // Validate data size
        if data_size % std::mem::size_of::<f32>() != 0 {
            return Err(DatasetError::FormatError(
                "WAV data section size incompatible with f32 samples".to_string(),
            ));
        }

        let total_samples = data_size / std::mem::size_of::<f32>();

        Ok(Self {
            mmap,
            sample_rate: spec.sample_rate,
            channels: spec.channels as u32,
            total_samples,
            file_path: path.to_path_buf(),
        })
    }

    /// Get audio data slice as f32 samples
    ///
    /// # Arguments
    /// * `start` - Starting sample index
    /// * `length` - Number of samples to retrieve
    ///
    /// # Returns
    /// Slice of f32 audio samples
    pub fn get_slice(&self, start: usize, length: usize) -> Result<&[f32]> {
        if start >= self.total_samples {
            return Err(DatasetError::ValidationError(format!(
                "Start index {start} exceeds total samples {}",
                self.total_samples
            )));
        }

        let end = (start + length).min(self.total_samples);
        let actual_length = end - start;

        if actual_length == 0 {
            return Ok(&[]);
        }

        let byte_start = start * std::mem::size_of::<f32>();
        let byte_length = actual_length * std::mem::size_of::<f32>();

        let byte_slice = &self.mmap[byte_start..byte_start + byte_length];

        // Safe conversion from bytes to f32 slice
        let (_, f32_slice, _) = unsafe { byte_slice.align_to::<f32>() };

        if f32_slice.len() != actual_length {
            return Err(DatasetError::FormatError(
                "Memory alignment issue in audio data".to_string(),
            ));
        }

        Ok(f32_slice)
    }

    /// Get all audio samples
    pub fn get_all_samples(&self) -> Result<&[f32]> {
        self.get_slice(0, self.total_samples)
    }

    /// Get audio metadata
    pub fn metadata(&self) -> AudioMetadata {
        AudioMetadata {
            sample_rate: self.sample_rate,
            channels: self.channels,
            total_samples: self.total_samples,
            duration: self.total_samples as f64 / (self.sample_rate as f64 * self.channels as f64),
            file_path: self.file_path.clone(),
        }
    }

    /// Convert a portion to AudioData
    ///
    /// This creates an in-memory AudioData from a slice of the memory-mapped data.
    pub fn to_audio_data(&self, start: usize, length: usize) -> Result<AudioData> {
        let slice = self.get_slice(start, length)?;
        let mut audio = AudioData::new(slice.to_vec(), self.sample_rate, self.channels);

        // Add metadata about the source
        audio.add_metadata("source".to_string(), "memory_mapped".to_string());
        audio.add_metadata(
            "original_file".to_string(),
            self.file_path.to_string_lossy().to_string(),
        );
        audio.add_metadata("slice_start".to_string(), start.to_string());
        audio.add_metadata("slice_length".to_string(), length.to_string());

        Ok(audio)
    }

    /// Find the offset of the data chunk in a WAV file
    fn find_wav_data_offset(mmap: &Mmap) -> Result<usize> {
        // Simplified WAV parser - looks for "data" chunk
        // This assumes a standard WAV file structure

        if mmap.len() < 44 {
            return Err(DatasetError::FormatError(
                "File too small to be a valid WAV file".to_string(),
            ));
        }

        // Check RIFF header
        if &mmap[0..4] != b"RIFF" || &mmap[8..12] != b"WAVE" {
            return Err(DatasetError::FormatError(
                "Invalid WAV file header".to_string(),
            ));
        }

        let mut offset = 12; // Skip RIFF header

        while offset + 8 <= mmap.len() {
            let chunk_id = &mmap[offset..offset + 4];
            let chunk_size = u32::from_le_bytes([
                mmap[offset + 4],
                mmap[offset + 5],
                mmap[offset + 6],
                mmap[offset + 7],
            ]) as usize;

            if chunk_id == b"data" {
                return Ok(offset + 8); // Return start of data section
            }

            offset += 8 + chunk_size;

            // WAV chunks are word-aligned
            if chunk_size % 2 == 1 {
                offset += 1;
            }
        }

        Err(DatasetError::FormatError(
            "Could not find data chunk in WAV file".to_string(),
        ))
    }
}

/// Audio metadata information
#[derive(Debug, Clone)]
pub struct AudioMetadata {
    pub sample_rate: u32,
    pub channels: u32,
    pub total_samples: usize,
    pub duration: f64,
    pub file_path: std::path::PathBuf,
}

/// Audio data cache for efficient access
pub struct AudioCache {
    cache: HashMap<String, AudioData>,
    max_size: usize,
    current_size: usize,
}

impl AudioCache {
    /// Create new audio cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            current_size: 0,
        }
    }

    /// Add audio data to cache
    pub fn insert(&mut self, key: String, audio: AudioData) {
        let audio_size = std::mem::size_of_val(audio.samples());

        // Remove old entries if needed
        while self.current_size + audio_size > self.max_size && !self.cache.is_empty() {
            if let Some((old_key, _)) = self.cache.iter().next() {
                let old_key = old_key.clone();
                if let Some(old_audio) = self.cache.remove(&old_key) {
                    self.current_size -= std::mem::size_of_val(old_audio.samples());
                }
            }
        }

        self.cache.insert(key, audio);
        self.current_size += audio_size;
    }

    /// Get audio data from cache
    pub fn get(&self, key: &str) -> Option<&AudioData> {
        self.cache.get(key)
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.current_size = 0;
    }
}

/// Audio data statistics
#[derive(Debug, Clone)]
pub struct AudioStats {
    /// Peak amplitude
    pub peak: f32,
    /// RMS amplitude
    pub rms: f32,
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    /// Spectral centroid
    pub spectral_centroid: Option<f32>,
    /// Spectral rolloff
    pub spectral_rolloff: Option<f32>,
}

impl AudioStats {
    /// Calculate statistics for audio data
    pub fn calculate(audio: &AudioData) -> Self {
        let samples = audio.samples();

        // Calculate peak
        let peak = samples
            .iter()
            .fold(0.0f32, |max, &sample| max.max(sample.abs()));

        // Calculate RMS
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();

        // Calculate zero crossing rate
        let zero_crossings = samples
            .windows(2)
            .filter(|window| (window[0] >= 0.0) != (window[1] >= 0.0))
            .count();
        let zero_crossing_rate = zero_crossings as f32 / samples.len() as f32;

        // Calculate spectral features
        let (spectral_centroid, spectral_rolloff) =
            Self::calculate_spectral_features(samples, audio.sample_rate());

        Self {
            peak,
            rms,
            zero_crossing_rate,
            spectral_centroid,
            spectral_rolloff,
        }
    }

    /// Calculate spectral features using FFT
    fn calculate_spectral_features(
        samples: &[f32],
        sample_rate: u32,
    ) -> (Option<f32>, Option<f32>) {
        use rustfft::{num_complex::Complex, FftPlanner};

        if samples.len() < 1024 {
            // Not enough samples for meaningful spectral analysis
            return (None, None);
        }

        // Use a window size that's a power of 2
        let window_size = 1024.min(samples.len());
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(window_size);

        // Take the first window_size samples for analysis
        let mut buffer: Vec<Complex<f32>> = samples[..window_size]
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        // Apply Hanning window to reduce spectral leakage
        for (i, sample) in buffer.iter_mut().enumerate() {
            let window_val = 0.5
                * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (window_size - 1) as f32).cos());
            *sample *= window_val;
        }

        // Perform FFT
        fft.process(&mut buffer);

        // Calculate magnitude spectrum (only use positive frequencies)
        let nyquist = window_size / 2;
        let magnitude_spectrum: Vec<f32> = buffer[..nyquist]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        let total_power: f32 = magnitude_spectrum.iter().sum();

        if total_power == 0.0 {
            return (None, None);
        }

        // Calculate spectral centroid
        let freq_resolution = sample_rate as f32 / window_size as f32;
        let mut weighted_sum = 0.0;

        for (i, &magnitude) in magnitude_spectrum.iter().enumerate() {
            let frequency = i as f32 * freq_resolution;
            weighted_sum += frequency * magnitude;
        }

        let spectral_centroid = weighted_sum / total_power;

        // Calculate spectral rolloff (frequency below which 85% of energy lies)
        let rolloff_threshold = 0.85 * total_power;
        let mut cumulative_power = 0.0;
        let mut rolloff_frequency = 0.0;

        for (i, &magnitude) in magnitude_spectrum.iter().enumerate() {
            cumulative_power += magnitude;
            if cumulative_power >= rolloff_threshold {
                rolloff_frequency = i as f32 * freq_resolution;
                break;
            }
        }

        (Some(spectral_centroid), Some(rolloff_frequency))
    }
}

/// Audio data builder for incremental construction
pub struct AudioDataBuilder {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u32,
    metadata: HashMap<String, String>,
}

impl AudioDataBuilder {
    /// Create new audio data builder
    pub fn new(sample_rate: u32, channels: u32) -> Self {
        Self {
            samples: Vec::new(),
            sample_rate,
            channels,
            metadata: HashMap::new(),
        }
    }

    /// Add samples to the builder
    pub fn add_samples(&mut self, samples: &[f32]) -> &mut Self {
        self.samples.extend_from_slice(samples);
        self
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) -> &mut Self {
        self.metadata.insert(key, value);
        self
    }

    /// Build the audio data
    pub fn build(self) -> AudioData {
        let mut audio = AudioData::new(self.samples, self.sample_rate, self.channels);
        for (key, value) in self.metadata {
            audio.add_metadata(key, value);
        }
        audio
    }
}
