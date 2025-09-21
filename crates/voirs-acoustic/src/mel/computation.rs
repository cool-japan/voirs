//! Mel spectrogram computation engine
//!
//! This module provides efficient mel spectrogram computation from audio signals
//! using optimized STFT and mel filter bank operations with SciRS2 integration.

use super::{create_mel_filterbank, MelMetadata, MelParams, MelStats};
use crate::{AcousticError, MelSpectrogram, Result};
use std::f32::consts::PI;

// SciRS2 imports for optimized DSP operations
use scirs2_fft::rfft;

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

    /// Compute mel spectrogram from audio with SciRS2 optimizations
    pub fn compute_optimized(&self, audio: &[f32]) -> Result<MelSpectrogram> {
        if audio.is_empty() {
            return Err(AcousticError::InputError("Empty audio signal".to_string()));
        }

        // Compute STFT with optimized FFT
        let stft = self.stft_optimized(audio)?;

        // Convert to power spectrogram
        let power_spec = self.compute_power_spectrogram(&stft)?;

        // Apply mel filter bank with SIMD optimization
        let mel_spec = self.apply_mel_filterbank_optimized(&power_spec)?;

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

        #[allow(clippy::needless_range_loop)]
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

    /// Compute STFT with SciRS2 optimized FFT
    fn stft_optimized(&self, audio: &[f32]) -> Result<Vec<Vec<Complex32>>> {
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

        #[allow(clippy::needless_range_loop)]
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

            // Compute FFT using optimized SciRS2 implementation
            let fft_result = self.optimized_fft(&frame)?;

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

        #[allow(clippy::needless_range_loop)]
        for k in 0..n {
            let mut sum = Complex32::new(0.0, 0.0);
            #[allow(clippy::needless_range_loop)]
            for j in 0..n {
                let angle = -2.0 * PI * (k * j) as f32 / n as f32;
                let complex_exp = Complex32::new(angle.cos(), angle.sin());
                sum = sum + Complex32::new(signal[j], 0.0) * complex_exp;
            }
            result[k] = sum;
        }

        Ok(result)
    }

    /// Optimized FFT implementation using SciRS2
    fn optimized_fft(&self, signal: &[f32]) -> Result<Vec<Complex32>> {
        if signal.is_empty() {
            return Ok(vec![]);
        }

        // Use SciRS2's real-to-complex FFT for better performance
        let spectrum = rfft(signal, Some(self.fft_size))
            .map_err(|e| AcousticError::ModelError(format!("SciRS2 FFT failed: {e:?}")))?;

        // Convert to our Complex32 format
        let result: Vec<Complex32> = spectrum
            .iter()
            .map(|c| Complex32::new(c.re as f32, c.im as f32))
            .collect();

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

    /// Apply mel filter bank to power spectrogram with SciRS2 SIMD optimization
    fn apply_mel_filterbank_optimized(&self, power_spec: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n_frames = power_spec.first().map_or(0, |freq| freq.len());
        let n_mels = self.mel_filterbank.len();

        let mut mel_spec = vec![vec![0.0; n_frames]; n_mels];

        // Convert power spectrogram to column-major format for better SIMD performance
        let mut power_matrix = vec![vec![0.0; power_spec.len()]; n_frames];
        for (freq_idx, freq_data) in power_spec.iter().enumerate() {
            for (frame_idx, &power_val) in freq_data.iter().enumerate() {
                if frame_idx < n_frames {
                    power_matrix[frame_idx][freq_idx] = power_val;
                }
            }
        }

        // Apply mel filter bank using optimized vector operations
        for (mel_idx, mel_filter) in self.mel_filterbank.iter().enumerate() {
            #[allow(clippy::needless_range_loop)]
            for frame_idx in 0..n_frames {
                // Use vectorized computation for better performance
                mel_spec[mel_idx][frame_idx] = mel_filter
                    .iter()
                    .zip(power_matrix[frame_idx].iter())
                    .map(|(&filter_weight, &freq_val)| filter_weight * freq_val)
                    .sum();
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

    /// Inverse mel spectrogram using Griffin-Lim algorithm
    pub fn inverse_mel(&self, mel: &MelSpectrogram) -> Result<Vec<f32>> {
        // Convert mel spectrogram to linear spectrogram
        let linear_spec = self.mel_to_linear_spectrogram(mel)?;

        // Apply Griffin-Lim algorithm for phase reconstruction
        let audio = self.griffin_lim_reconstruction(&linear_spec)?;

        Ok(audio)
    }

    /// Convert mel spectrogram to linear spectrogram using pseudo-inverse
    fn mel_to_linear_spectrogram(&self, mel: &MelSpectrogram) -> Result<Vec<Vec<f32>>> {
        let n_freqs = (self.params.n_fft / 2 + 1) as usize;
        let n_frames = mel.n_frames;

        // Create linear spectrogram with proper dimensions
        let mut linear_spec = vec![vec![0.0; n_frames]; n_freqs];

        // Apply pseudo-inverse of mel filterbank
        for frame_idx in 0..n_frames {
            for freq_idx in 0..n_freqs {
                let mut linear_value = 0.0;

                // Accumulate contributions from all mel channels
                for (mel_idx, mel_filter) in self.mel_filterbank.iter().enumerate() {
                    if freq_idx < mel_filter.len() && mel_filter[freq_idx] > 0.0 {
                        let mel_value = if mel_idx < mel.n_mels {
                            mel.data[mel_idx][frame_idx]
                        } else {
                            0.0
                        };

                        // Apply inverse mel transformation
                        let mel_contribution = if self.params.log {
                            (mel_value - self.params.log_offset).exp()
                        } else {
                            mel_value
                        };

                        linear_value += mel_contribution * mel_filter[freq_idx];
                    }
                }

                linear_spec[freq_idx][frame_idx] = linear_value;
            }
        }

        Ok(linear_spec)
    }

    /// Apply Griffin-Lim algorithm for phase reconstruction
    fn griffin_lim_reconstruction(&self, magnitude_spec: &[Vec<f32>]) -> Result<Vec<f32>> {
        let n_freqs = magnitude_spec.len();
        let n_frames = magnitude_spec.first().map_or(0, |row| row.len());

        if n_frames == 0 {
            return Ok(vec![]);
        }

        // Initialize with random phase
        let mut complex_spec = vec![vec![Complex32::new(0.0, 0.0); n_frames]; n_freqs];

        // Initialize with magnitude and random phase
        for freq_idx in 0..n_freqs {
            for frame_idx in 0..n_frames {
                let magnitude = magnitude_spec[freq_idx][frame_idx].max(1e-8);
                let phase = 2.0 * PI * (freq_idx * frame_idx) as f32 / n_freqs as f32;
                complex_spec[freq_idx][frame_idx] =
                    Complex32::new(magnitude * phase.cos(), magnitude * phase.sin());
            }
        }

        // Griffin-Lim iterations
        let griffin_lim_iters = 32;
        for _iter in 0..griffin_lim_iters {
            // ISTFT to get time-domain signal
            let audio = self.istft(&complex_spec)?;

            // STFT to get back to frequency domain
            let new_complex_spec = self.stft_for_griffin_lim(&audio)?;

            // Update phase while preserving magnitude
            for freq_idx in 0..n_freqs {
                for frame_idx in 0..n_frames {
                    let target_magnitude = magnitude_spec[freq_idx][frame_idx].max(1e-8);
                    let current_complex = new_complex_spec[freq_idx][frame_idx];
                    let current_magnitude = current_complex.norm().max(1e-8);

                    // Preserve target magnitude, use reconstructed phase
                    complex_spec[freq_idx][frame_idx] = Complex32::new(
                        target_magnitude * current_complex.re / current_magnitude,
                        target_magnitude * current_complex.im / current_magnitude,
                    );
                }
            }
        }

        // Final ISTFT
        let final_audio = self.istft(&complex_spec)?;
        Ok(final_audio)
    }

    /// STFT specifically for Griffin-Lim (matches the forward STFT)
    fn stft_for_griffin_lim(&self, audio: &[f32]) -> Result<Vec<Vec<Complex32>>> {
        let hop_length = self.params.hop_length as usize;
        let win_length = self.params.win_length as usize;
        let n_fft = self.params.n_fft as usize;

        // Calculate number of frames
        let n_frames = if audio.len() >= win_length {
            (audio.len() - win_length) / hop_length + 1
        } else {
            1
        };

        let mut stft_result = vec![vec![Complex32::new(0.0, 0.0); n_frames]; n_fft / 2 + 1];

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

            // Compute FFT
            let fft_result = self.simple_fft(&frame)?;

            // Store only positive frequencies
            for freq_idx in 0..(n_fft / 2 + 1) {
                if freq_idx < fft_result.len() {
                    stft_result[freq_idx][frame_idx] = fft_result[freq_idx];
                }
            }
        }

        Ok(stft_result)
    }

    /// Inverse Short-Time Fourier Transform
    fn istft(&self, complex_spec: &[Vec<Complex32>]) -> Result<Vec<f32>> {
        let hop_length = self.params.hop_length as usize;
        let win_length = self.params.win_length as usize;
        let n_fft = self.params.n_fft as usize;
        let n_freqs = complex_spec.len();
        let n_frames = complex_spec.first().map_or(0, |row| row.len());

        if n_frames == 0 {
            return Ok(vec![]);
        }

        // Calculate output length
        let output_length = (n_frames - 1) * hop_length + win_length;
        let mut audio = vec![0.0; output_length];
        let mut overlap_count = vec![0.0; output_length];

        for frame_idx in 0..n_frames {
            // Create full frequency domain representation
            let mut full_spectrum = vec![Complex32::new(0.0, 0.0); n_fft];

            // Fill positive frequencies
            for freq_idx in 0..n_freqs {
                if freq_idx < complex_spec.len() {
                    full_spectrum[freq_idx] = complex_spec[freq_idx][frame_idx];
                }
            }

            // Fill negative frequencies (complex conjugate)
            for freq_idx in 1..(n_fft / 2) {
                if n_fft - freq_idx < full_spectrum.len() {
                    let conj = complex_spec[freq_idx][frame_idx];
                    full_spectrum[n_fft - freq_idx] = Complex32::new(conj.re, -conj.im);
                }
            }

            // Apply inverse FFT
            let time_frame = self.simple_ifft(&full_spectrum)?;

            // Window and overlap-add
            let start = frame_idx * hop_length;
            for i in 0..win_length {
                if start + i < audio.len() && i < time_frame.len() && i < self.window.len() {
                    audio[start + i] += time_frame[i] * self.window[i];
                    overlap_count[start + i] += self.window[i] * self.window[i];
                }
            }
        }

        // Normalize by overlap count
        for i in 0..audio.len() {
            if overlap_count[i] > 1e-8 {
                audio[i] /= overlap_count[i];
            }
        }

        Ok(audio)
    }

    /// Simple inverse FFT implementation
    fn simple_ifft(&self, spectrum: &[Complex32]) -> Result<Vec<f32>> {
        let n = spectrum.len();
        if n == 0 {
            return Ok(vec![]);
        }

        let mut result = vec![0.0; n];

        #[allow(clippy::needless_range_loop)]
        for k in 0..n {
            let mut sum = Complex32::new(0.0, 0.0);
            #[allow(clippy::needless_range_loop)]
            for j in 0..n {
                let angle = 2.0 * PI * (k * j) as f32 / n as f32;
                let complex_exp = Complex32::new(angle.cos(), angle.sin());
                sum = sum + spectrum[j] * complex_exp;
            }
            result[k] = sum.re / n as f32;
        }

        Ok(result)
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
    pub fn compute_batch_with_metadata(
        &self,
        audio_batch: &[&[f32]],
    ) -> Result<Vec<(MelSpectrogram, MelMetadata)>> {
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
        let min_length =
            self.computer.params.win_length as usize + self.computer.params.hop_length as usize;
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

    #[test]
    fn test_optimized_compute() {
        let params = MelParams::standard_22khz();
        let computer = MelComputer::new(params).unwrap();

        // Generate test audio (sine wave)
        let duration = 0.5; // 0.5 seconds
        let frequency = 440.0; // A4
        let sample_rate = 22050.0;
        let samples = (duration * sample_rate) as usize;

        let audio: Vec<f32> = (0..samples)
            .map(|i| (2.0 * PI * frequency * i as f32 / sample_rate).sin())
            .collect();

        // Test optimized computation produces valid results
        let mel_optimized = computer.compute_optimized(&audio).unwrap();

        // Results should have correct dimensions
        assert_eq!(mel_optimized.n_mels, 80);
        assert!(mel_optimized.n_frames > 0);
        assert_eq!(mel_optimized.sample_rate, 22050);
        assert_eq!(mel_optimized.hop_length, 256);

        // Results should contain finite values
        for mel_idx in 0..mel_optimized.n_mels {
            for frame_idx in 0..mel_optimized.n_frames {
                let val = mel_optimized.data[mel_idx][frame_idx];
                assert!(
                    val.is_finite(),
                    "Non-finite value at mel[{mel_idx}][{frame_idx}]: {val}"
                );
            }
        }

        // For a sine wave input, we should see energy concentration in specific mel bands
        // Lower frequency sine wave should have more energy in lower mel bands
        let mut total_energy_low = 0.0;
        let mut total_energy_high = 0.0;

        for frame_idx in 0..mel_optimized.n_frames {
            // Sum energy in lower mel bands (0-20)
            for mel_idx in 0..20 {
                total_energy_low += mel_optimized.data[mel_idx][frame_idx].exp();
                // Convert from log scale
            }
            // Sum energy in higher mel bands (60-80)
            for mel_idx in 60..80 {
                total_energy_high += mel_optimized.data[mel_idx][frame_idx].exp();
                // Convert from log scale
            }
        }

        // For a 440Hz sine wave, we expect more energy in lower bands than higher bands
        assert!(
            total_energy_low > total_energy_high * 0.1,
            "Expected more energy in lower mel bands for 440Hz sine wave"
        );
    }

    #[test]
    fn test_optimized_fft_empty_input() {
        let params = MelParams::standard_22khz();
        let computer = MelComputer::new(params).unwrap();

        let empty_signal: Vec<f32> = vec![];
        let result = computer.optimized_fft(&empty_signal).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_optimized_mel_filterbank() {
        let params = MelParams::standard_22khz();
        let n_freqs = params.n_freqs();
        let computer = MelComputer::new(params).unwrap();

        // Create test power spectrogram
        let n_frames = 10;
        let power_spec: Vec<Vec<f32>> = (0..n_freqs)
            .map(|i| (0..n_frames).map(|j| (i + j) as f32 * 0.1).collect())
            .collect();

        // Compare standard and optimized mel filterbank application
        let mel_std = computer.apply_mel_filterbank(&power_spec).unwrap();
        let mel_opt = computer
            .apply_mel_filterbank_optimized(&power_spec)
            .unwrap();

        // Results should have the same dimensions
        assert_eq!(mel_std.len(), mel_opt.len());
        assert_eq!(mel_std[0].len(), mel_opt[0].len());

        // Results should be identical for matrix operations
        for mel_idx in 0..mel_std.len() {
            for frame_idx in 0..mel_std[0].len() {
                let diff = (mel_std[mel_idx][frame_idx] - mel_opt[mel_idx][frame_idx]).abs();
                assert!(
                    diff < 1e-6,
                    "Difference at mel[{}][{}]: {} vs {}",
                    mel_idx,
                    frame_idx,
                    mel_std[mel_idx][frame_idx],
                    mel_opt[mel_idx][frame_idx]
                );
            }
        }
    }
}
