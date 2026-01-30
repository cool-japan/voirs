//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{AudioBuffer, MelSpectrogram, Result, SynthesisConfig, Vocoder};
use async_trait::async_trait;
use std::collections::VecDeque;

/// Streaming utilities for chunk-based processing
pub struct StreamingBuffer {
    /// Ring buffer for audio data
    buffer: std::collections::VecDeque<f32>,
    /// Maximum buffer size
    max_size: usize,
    /// Chunk size for processing
    chunk_size: usize,
    /// Overlap size between chunks
    overlap_size: usize,
    /// Sample rate
    sample_rate: u32,
}
impl StreamingBuffer {
    /// Create a new streaming buffer
    pub fn new(chunk_size: usize, overlap_size: usize, sample_rate: u32) -> Self {
        let max_size = chunk_size * 4;
        Self {
            buffer: std::collections::VecDeque::with_capacity(max_size),
            max_size,
            chunk_size,
            overlap_size,
            sample_rate,
        }
    }
    /// Add new audio data to the buffer
    pub fn push_audio(&mut self, audio: &[f32]) {
        for &sample in audio {
            if self.buffer.len() >= self.max_size {
                self.buffer.pop_front();
            }
            self.buffer.push_back(sample);
        }
    }
    /// Get the next chunk for processing (returns None if not enough data)
    pub fn get_chunk(&mut self) -> Option<Vec<f32>> {
        if self.buffer.len() >= self.chunk_size {
            let chunk: Vec<f32> = self.buffer.iter().take(self.chunk_size).cloned().collect();
            let advance_size = self.chunk_size - self.overlap_size;
            for _ in 0..advance_size {
                self.buffer.pop_front();
            }
            Some(chunk)
        } else {
            None
        }
    }
    /// Check if there's enough data for a chunk
    pub fn has_chunk(&self) -> bool {
        self.buffer.len() >= self.chunk_size
    }
    /// Get the current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
    /// Get chunk size
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }
    /// Get overlap size
    pub fn overlap_size(&self) -> usize {
        self.overlap_size
    }
    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
/// Audio spectral statistics
#[derive(Debug, Clone)]
pub struct SpectralStatistics {
    /// Peak amplitude
    pub peak: f32,
    /// RMS energy
    pub rms: f32,
    /// Spectral centroid (center of mass of spectrum)
    pub spectral_centroid: f32,
    /// Spectral bandwidth (spread around centroid)
    pub spectral_bandwidth: f32,
    /// Spectral flatness (measure of noisiness)
    pub spectral_flatness: f32,
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    /// Dynamic range in dB
    pub dynamic_range_db: f32,
}
/// Crossfade curve types for smooth audio transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossfadeType {
    /// Linear crossfade (constant power)
    Linear,
    /// Exponential crossfade (smooth start/end)
    Exponential,
    /// Sine-based crossfade (musical transitions)
    Sine,
    /// Cosine-based crossfade (broadcast quality)
    Cosine,
}
/// Calculate advanced audio quality metrics
#[derive(Debug, Clone)]
pub struct AudioQualityMetrics {
    /// Total Harmonic Distortion (THD)
    pub thd_percent: f32,
    /// Signal-to-Noise Ratio in dB
    pub snr_db: f32,
    /// Crest factor (peak-to-RMS ratio)
    pub crest_factor: f32,
    /// Loudness estimate (ITU-R BS.1770)
    pub loudness_lufs: f32,
    /// Dynamic range
    pub dynamic_range_db: f32,
}
/// Streaming mel processor for real-time vocoding
pub struct StreamingMelProcessor {
    buffer: StreamingBuffer,
    vocoder: Option<Box<dyn crate::Vocoder + Send + Sync>>,
}
impl StreamingMelProcessor {
    pub fn new(chunk_size: usize, overlap_size: usize, sample_rate: u32) -> Self {
        Self {
            buffer: StreamingBuffer::new(chunk_size, overlap_size, sample_rate),
            vocoder: None,
        }
    }
    /// Set the vocoder for mel processing
    pub fn set_vocoder(&mut self, vocoder: Box<dyn crate::Vocoder + Send + Sync>) {
        self.vocoder = Some(vocoder);
    }
    /// Process mel chunk and return audio using the configured vocoder
    pub fn process_chunk(&mut self, mel_chunk: &MelSpectrogram) -> Option<AudioBuffer> {
        if let Some(vocoder) = &self.vocoder {
            match tokio::runtime::Runtime::new() {
                Ok(rt) => match rt.block_on(vocoder.vocode(mel_chunk, None)) {
                    Ok(audio) => {
                        self.buffer.push_audio(audio.samples());
                        Some(audio)
                    }
                    Err(e) => {
                        tracing::warn!("Vocoding failed: {e}");
                        None
                    }
                },
                Err(e) => {
                    tracing::warn!("Failed to create async runtime: {e}");
                    None
                }
            }
        } else {
            tracing::warn!("No vocoder configured for streaming mel processor");
            None
        }
    }
    pub fn add_audio(&mut self, audio: &[f32]) {
        self.buffer.push_audio(audio);
    }
    pub fn get_processed_chunk(&mut self) -> Option<Vec<f32>> {
        self.buffer.get_chunk()
    }
}
