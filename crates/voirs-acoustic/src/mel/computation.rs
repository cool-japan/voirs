//! Mel spectrogram computation engine
//!
//! This module provides efficient mel spectrogram computation from audio signals
//! using optimized STFT and mel filter bank operations.

use std::f32::consts::PI;
use crate::{Result, AcousticError, MelSpectrogram};
use super::{MelParams, MelNormalization, WindowType, create_mel_filterbank, MelMetadata, MelStats};

/// Mel spectrogram computation engine
pub struct MelComputer {
    /// Mel parameters
    params: MelParams,
    /// Mel filter bank
    mel_filterbank: Vec<Vec<f32>>,
    /// Window function
    window: Vec<f32>,
    /// FFT plan (placeholder for actual FFT implementation)
    fft_size: usize,
}

impl MelComputer {
    /// Create new mel computer with parameters
    pub fn new(params: MelParams) -> Result<Self> {
        params.validate()?;
        
        let n_freqs = params.n_freqs();
        let fmax = params.effective_fmax();
        
        let mel_filterbank = create_mel_filterbank(
            params.n_mels,
            n_freqs,
            params.sample_rate,
            params.fmin,
            fmax,
            params.norm,
        )?;
        
        let window = params.window.generate(params.win_length as usize);
        let fft_size = params.n_fft as usize;
        
        Ok(Self {
            params,
            mel_filterbank,
            window,
            fft_size,
        })
    }
    
    /// Compute mel spectrogram from audio
    pub fn compute(&self, audio: &[f32]) -> Result<MelSpectrogram> {
        if audio.is_empty() {
            return Err(AcousticError::InputError("Empty audio signal".to_string()));
        }
        
        // Compute STFT
        let stft = self.stft(audio)?;
        
        // Convert to power spectrogram
        let power_spec = self.compute_power_spectrogram(&stft)?;
        
        // Apply mel filter bank
        let mel_spec = self.apply_mel_filterbank(&power_spec)?;
        
        // Apply log scale if requested
        let mel_data = if self.params.log {
            self.apply_log_scale(&mel_spec)?
        } else {
            mel_spec
        };
        
        let mel = MelSpectrogram::new(mel_data, self.params.sample_rate, self.params.hop_length);
        
        Ok(mel)
    }
    
    /// Compute mel spectrogram with metadata
    pub fn compute_with_metadata(&self, audio: &[f32]) -> Result<(MelSpectrogram, MelMetadata)> {
        let mel = self.compute(audio)?;
        let stats = MelStats::compute(&mel)?;
        let metadata = MelMetadata::new(self.params.clone(), mel.n_frames as u32).with_stats(stats);
        Ok((mel, metadata))
    }
    
    /// Compute STFT (Short-Time Fourier Transform)
    fn stft(&self, audio: &[f32]) -> Result<Vec<Vec<Complex32>>> {
        let hop_length = self.params.hop_length as usize;
        let win_length = self.params.win_length as usize;
        let n_fft = self.params.n_fft as usize;
        
        // Calculate number of frames
        let n_frames = if audio.len() >= win_length {
            (audio.len() - win_length) / hop_length + 1
        } else {
            1
        };
        
        let mut stft_result = vec![vec![Complex32::new(0.0, 0.0); n_fft / 2 + 1]; n_frames];
        
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;
            let end = (start + win_length).min(audio.len());
            
            // Extract and window the frame
            let mut frame = vec![0.0; n_fft];
            for i in 0..(end - start) {
                if i < self.window.len() {
                    frame[i] = audio[start + i] * self.window[i];
                }
            }
            
            // Compute FFT (simplified implementation)
            let fft_result = self.simple_fft(&frame)?;
            
            // Take only the positive frequencies
            for (i, &value) in fft_result.iter().take(n_fft / 2 + 1).enumerate() {
                stft_result[frame_idx][i] = value;
            }
        }
        
        Ok(stft_result)
    }
    
    /// Simple FFT implementation (placeholder - in production use a proper FFT library)
    fn simple_fft(&self, signal: &[f32]) -> Result<Vec<Complex32>> {
        let n = signal.len();
        if n == 0 {
            return Ok(vec![]);
        }
        
        // This is a simplified DFT implementation for demonstration
        // In production, use a proper FFT library like rustfft
        let mut result = vec![Complex32::new(0.0, 0.0); n];
        
        for k in 0..n {
            let mut sum = Complex32::new(0.0, 0.0);
            for j in 0..n {
                let angle = -2.0 * PI * (k * j) as f32 / n as f32;
                let complex_exp = Complex32::new(angle.cos(), angle.sin());
                sum = sum + Complex32::new(signal[j], 0.0) * complex_exp;
            }
            result[k] = sum;
        }
        
        Ok(result)
    }
    
    /// Compute power spectrogram from STFT
    fn compute_power_spectrogram(&self, stft: &[Vec<Complex32>]) -> Result<Vec<Vec<f32>>> {
        let n_frames = stft.len();
        let n_freqs = stft.first().map_or(0, |frame| frame.len());
        
        let mut power_spec = vec![vec![0.0; n_frames]; n_freqs];
        
        for (frame_idx, frame) in stft.iter().enumerate() {
            for (freq_idx, &complex_val) in frame.iter().enumerate() {
                let magnitude = complex_val.norm();
                power_spec[freq_idx][frame_idx] = magnitude.powf(self.params.power);
            }
        }
        
        Ok(power_spec)
    }
    
    /// Apply mel filter bank to power spectrogram
    fn apply_mel_filterbank(&self, power_spec: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n_frames = power_spec.first().map_or(0, |freq| freq.len());
        let n_mels = self.mel_filterbank.len();
        
        let mut mel_spec = vec![vec![0.0; n_frames]; n_mels];
        
        for (mel_idx, mel_filter) in self.mel_filterbank.iter().enumerate() {
            for frame_idx in 0..n_frames {
                let mut mel_value = 0.0;
                for (freq_idx, &filter_weight) in mel_filter.iter().enumerate() {
                    if freq_idx < power_spec.len() && filter_weight > 0.0 {
                        mel_value += power_spec[freq_idx][frame_idx] * filter_weight;
                    }
                }
                mel_spec[mel_idx][frame_idx] = mel_value;
            }
        }
        
        Ok(mel_spec)
    }
    
    /// Apply log scale to mel spectrogram
    fn apply_log_scale(&self, mel_spec: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut log_mel_spec = mel_spec.to_vec();
        
        for mel_channel in &mut log_mel_spec {
            for value in mel_channel {
                *value = (*value + self.params.log_offset).ln();
            }
        }
        
        Ok(log_mel_spec)
    }
    
    /// Get mel parameters
    pub fn params(&self) -> &MelParams {
        &self.params
    }
    
    /// Get mel filter bank
    pub fn filterbank(&self) -> &[Vec<f32>] {
        &self.mel_filterbank
    }
    
    /// Inverse mel spectrogram (placeholder for vocoder functionality)
    pub fn inverse_mel(&self, _mel: &MelSpectrogram) -> Result<Vec<f32>> {
        // TODO: Implement inverse mel spectrogram computation
        // This would typically involve a vocoder like Griffin-Lim or neural vocoder
        Err(AcousticError::ModelError("Inverse mel computation not yet implemented".to_string()))
    }
}

/// Complex number type for FFT
#[derive(Debug, Clone, Copy)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
    
    pub fn norm(&self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

impl std::ops::Add for Complex32 {
    type Output = Self;
    
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl std::ops::Mul for Complex32 {
    type Output = Self;
    
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

/// Batch mel computation for multiple audio signals
pub struct BatchMelComputer {
    computer: MelComputer,
}

impl BatchMelComputer {
    /// Create new batch mel computer
    pub fn new(params: MelParams) -> Result<Self> {
        let computer = MelComputer::new(params)?;
        Ok(Self { computer })
    }
    
    /// Compute mel spectrograms for multiple audio signals
    pub fn compute_batch(&self, audio_batch: &[&[f32]]) -> Result<Vec<MelSpectrogram>> {
        let mut results = Vec::with_capacity(audio_batch.len());
        
        for audio in audio_batch {
            let mel = self.computer.compute(audio)?;
            results.push(mel);
        }
        
        Ok(results)
    }
    
    /// Compute mel spectrograms with metadata for multiple audio signals
    pub fn compute_batch_with_metadata(&self, audio_batch: &[&[f32]]) -> Result<Vec<(MelSpectrogram, MelMetadata)>> {
        let mut results = Vec::with_capacity(audio_batch.len());
        
        for audio in audio_batch {
            let (mel, metadata) = self.computer.compute_with_metadata(audio)?;
            results.push((mel, metadata));
        }
        
        Ok(results)
    }
}

/// Streaming mel computation for real-time applications
pub struct StreamingMelComputer {
    computer: MelComputer,
    buffer: Vec<f32>,
    overlap: usize,
}

impl StreamingMelComputer {
    /// Create new streaming mel computer
    pub fn new(params: MelParams, overlap_factor: f32) -> Result<Self> {
        let overlap = (params.win_length as f32 * overlap_factor) as usize;
        let computer = MelComputer::new(params)?;
        
        Ok(Self {
            computer,
            buffer: Vec::new(),
            overlap,
        })
    }
    
    /// Process audio chunk and return mel spectrogram frames
    pub fn process_chunk(&mut self, audio_chunk: &[f32]) -> Result<Option<MelSpectrogram>> {
        // Add new audio to buffer
        self.buffer.extend_from_slice(audio_chunk);
        
        // Check if we have enough data to process
        let min_length = self.computer.params.win_length as usize + self.computer.params.hop_length as usize;
        if self.buffer.len() < min_length {
            return Ok(None);
        }
        
        // Process available data
        let process_length = self.buffer.len() - self.overlap;
        let mel = self.computer.compute(&self.buffer[..process_length])?;
        
        // Keep overlap for next chunk
        if self.overlap > 0 && self.buffer.len() > self.overlap {
            self.buffer.drain(..self.buffer.len() - self.overlap);
        } else {
            self.buffer.clear();
        }
        
        Ok(Some(mel))
    }
    
    /// Flush remaining audio in buffer
    pub fn flush(&mut self) -> Result<Option<MelSpectrogram>> {
        if self.buffer.is_empty() {
            return Ok(None);
        }
        
        let mel = self.computer.compute(&self.buffer)?;
        self.buffer.clear();
        Ok(Some(mel))
    }
    
    /// Reset the streaming computer
    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_computer_creation() {
        let params = MelParams::standard_22khz();
        let computer = MelComputer::new(params).unwrap();
        
        assert_eq!(computer.params().sample_rate, 22050);
        assert_eq!(computer.params().n_mels, 80);
        assert_eq!(computer.filterbank().len(), 80);
    }

    #[test]
    fn test_mel_computation() {
        let params = MelParams::standard_22khz();
        let computer = MelComputer::new(params).unwrap();
        
        // Generate test audio (sine wave)
        let duration = 1.0; // 1 second
        let frequency = 440.0; // A4
        let sample_rate = 22050.0;
        let samples = (duration * sample_rate) as usize;
        
        let audio: Vec<f32> = (0..samples)
            .map(|i| (2.0 * PI * frequency * i as f32 / sample_rate).sin())
            .collect();
        
        let mel = computer.compute(&audio).unwrap();
        
        assert!(mel.n_mels > 0);
        assert!(mel.n_frames > 0);
        assert_eq!(mel.sample_rate, 22050);
        assert_eq!(mel.hop_length, 256);
    }

    #[test]
    fn test_mel_computation_with_metadata() {
        let params = MelParams::standard_22khz();
        let computer = MelComputer::new(params).unwrap();
        
        let audio = vec![0.1, 0.2, 0.3, 0.4, 0.5]; // Short test audio
        let (mel, metadata) = computer.compute_with_metadata(&audio).unwrap();
        
        assert!(mel.n_mels > 0);
        assert!(metadata.stats.is_some());
        assert_eq!(metadata.params.sample_rate, 22050);
    }

    #[test]
    fn test_complex32_operations() {
        let a = Complex32::new(1.0, 2.0);
        let b = Complex32::new(3.0, 4.0);
        
        let sum = a + b;
        assert_eq!(sum.re, 4.0);
        assert_eq!(sum.im, 6.0);
        
        let product = a * b;
        assert_eq!(product.re, -5.0); // 1*3 - 2*4
        assert_eq!(product.im, 10.0); // 1*4 + 2*3
        
        let norm = a.norm();
        assert!((norm - (1.0 + 4.0_f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_batch_mel_computer() {
        let params = MelParams::standard_22khz();
        let computer = BatchMelComputer::new(params).unwrap();
        
        let audio1 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let audio2 = vec![0.5, 0.4, 0.3, 0.2, 0.1];
        let batch = vec![audio1.as_slice(), audio2.as_slice()];
        
        let results = computer.compute_batch(&batch).unwrap();
        assert_eq!(results.len(), 2);
        
        for mel in &results {
            assert!(mel.n_mels > 0);
            assert!(mel.n_frames > 0);
        }
    }

    #[test]
    fn test_streaming_mel_computer() {
        let params = MelParams::standard_22khz();
        let mut computer = StreamingMelComputer::new(params, 0.5).unwrap();
        
        // Process small chunks
        let chunk1 = vec![0.1; 1000];
        let chunk2 = vec![0.2; 1000];
        let chunk3 = vec![0.3; 1000];
        
        let result1 = computer.process_chunk(&chunk1).unwrap();
        let result2 = computer.process_chunk(&chunk2).unwrap();
        let result3 = computer.process_chunk(&chunk3).unwrap();
        
        // At least one result should be Some
        assert!(result1.is_some() || result2.is_some() || result3.is_some());
        
        // Flush remaining data
        let final_result = computer.flush().unwrap();
        if let Some(mel) = final_result {
            assert!(mel.n_mels > 0);
        }
    }

    #[test]
    fn test_empty_audio_error() {
        let params = MelParams::standard_22khz();
        let computer = MelComputer::new(params).unwrap();
        
        let empty_audio: Vec<f32> = vec![];
        let result = computer.compute(&empty_audio);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_computer_reset() {
        let params = MelParams::standard_22khz();
        let mut computer = StreamingMelComputer::new(params, 0.5).unwrap();
        
        let chunk = vec![0.1; 1000];
        let _result = computer.process_chunk(&chunk).unwrap();
        
        computer.reset();
        
        // Buffer should be empty after reset
        let result = computer.flush().unwrap();
        assert!(result.is_none());
    }
}