//! Whisper audio preprocessing with STFT and mel spectrogram computation
//!
//! This module provides audio preprocessing for Whisper models including
//! STFT computation, mel filter bank application, and normalization.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use super::encoder::WhisperConfig;
use crate::RecognitionError;
use candle_core::{Device, Tensor};
use scirs2_core::Complex;
use scirs2_fft::{FftPlanner, RealFftPlanner};
use voirs_sdk::AudioBuffer;

/// Audio processor for Whisper
pub struct WhisperAudioProcessor {
    /// Mel filter bank
    mel_filters: Tensor,
    /// Window function
    window: Tensor,
    /// Configuration
    config: WhisperConfig,
}

impl WhisperAudioProcessor {
    /// Creates a new audio processor for Whisper
    ///
    /// # Errors
    ///
    /// Returns `RecognitionError` if mel filter bank or window function creation fails
    pub fn new(config: &WhisperConfig, device: &Device) -> Result<Self, RecognitionError> {
        // Create mel filter bank
        let mel_filters = Self::create_mel_filters(config, device)?;

        // Create window function (Hann window)
        let window = Self::create_hann_window(config, device)?;

        Ok(Self {
            mel_filters,
            window,
            config: config.clone(),
        })
    }

    fn create_mel_filters(
        config: &WhisperConfig,
        device: &Device,
    ) -> Result<Tensor, RecognitionError> {
        // Create mel filter bank for Whisper
        // FFT size = 1024, hop length = 160, n_mels = 80
        let n_fft = 1024;
        let n_freqs = n_fft / 2 + 1; // 513 frequency bins
        let sample_rate = config.sample_rate as f32;

        // Convert Hz to mel scale
        let mel_low = Self::hz_to_mel(0.0);
        let mel_high = Self::hz_to_mel(sample_rate / 2.0);

        // Create mel frequency points
        let n_mels = config.n_mels;
        let mel_points = (0..=n_mels + 1)
            .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (n_mels + 1) as f32)
            .collect::<Vec<f32>>();

        // Convert mel points back to Hz
        let hz_points = mel_points
            .iter()
            .map(|&mel| Self::mel_to_hz(mel))
            .collect::<Vec<f32>>();

        // Convert Hz to FFT bin indices
        let bin_points = hz_points
            .iter()
            .map(|&hz| (hz * n_fft as f32 / sample_rate).floor() as usize)
            .collect::<Vec<usize>>();

        // Create filter bank
        let mut filters = vec![0.0f32; n_mels * n_freqs];

        for m in 1..=n_mels {
            let left = bin_points[m - 1];
            let center = bin_points[m];
            let right = bin_points[m + 1];

            for k in left..=right {
                if k < n_freqs {
                    let filter_idx = (m - 1) * n_freqs + k;

                    if k <= center {
                        // Rising slope
                        if center > left {
                            filters[filter_idx] = (k - left) as f32 / (center - left) as f32;
                        }
                    } else {
                        // Falling slope
                        if right > center {
                            filters[filter_idx] = (right - k) as f32 / (right - center) as f32;
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&filters, (n_mels, n_freqs), device).map_err(|e| {
            RecognitionError::AudioProcessingError {
                message: format!("Failed to create mel filter bank: {e}"),
                source: Some(Box::new(e)),
            }
        })
    }

    fn create_hann_window(
        _config: &WhisperConfig,
        device: &Device,
    ) -> Result<Tensor, RecognitionError> {
        // Create Hann window for STFT
        let n_fft = 1024;
        let mut window = Vec::with_capacity(n_fft);

        for i in 0..n_fft {
            let val =
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n_fft - 1) as f32).cos());
            window.push(val);
        }

        Tensor::from_slice(&window, n_fft, device).map_err(|e| {
            RecognitionError::AudioProcessingError {
                message: format!("Failed to create Hann window: {e}"),
                source: Some(Box::new(e)),
            }
        })
    }

    /// Process audio buffer and extract mel spectrogram features.
    ///
    /// # Errors
    ///
    /// Returns `RecognitionError` if:
    /// - Audio preprocessing fails (resampling or channel conversion)
    /// - STFT computation encounters mathematical errors
    /// - Mel filter application fails
    /// - Log compression or normalization operations fail
    pub fn process_audio(&self, audio: &AudioBuffer) -> Result<Tensor, RecognitionError> {
        // Ensure audio is 16kHz mono
        let preprocessed = self.preprocess_audio(audio);

        // Compute STFT
        let stft = self.compute_stft(&preprocessed)?;

        // Convert to mel spectrogram
        let mel_spec = self.apply_mel_filters(&stft)?;

        // Apply log compression
        let log_mel = self.log_compress(&mel_spec)?;

        // Normalize
        let normalized = self.normalize_mel(&log_mel)?;

        Ok(normalized)
    }

    /// Extract mel features (alias for `process_audio` for language detection)
    ///
    /// # Errors
    ///
    /// Returns `RecognitionError` if audio processing fails.
    /// See [`process_audio`] for detailed error conditions.
    pub async fn extract_mel_features(
        &self,
        audio: &AudioBuffer,
    ) -> Result<Tensor, RecognitionError> {
        self.process_audio(audio)
    }

    /// Process audio with streaming support for real-time applications
    ///
    /// # Errors
    ///
    /// Returns `RecognitionError` if:
    /// - Audio preprocessing fails for any chunk
    /// - Individual chunk processing fails
    /// - Buffer operations or memory allocation fails
    pub async fn process_audio_streaming(
        &self,
        audio: &AudioBuffer,
        chunk_duration_ms: u32,
    ) -> Result<Vec<Tensor>, RecognitionError> {
        let chunk_samples =
            (self.config.sample_rate as f32 * chunk_duration_ms as f32 / 1000.0) as usize;
        let audio_samples = self.preprocess_audio(audio);

        if audio_samples.len() <= chunk_samples {
            return Ok(vec![self.process_audio(audio)?]);
        }

        let mut chunks = Vec::new();
        let overlap_samples = chunk_samples / 4; // 25% overlap
        let step_size = chunk_samples - overlap_samples;

        for start in (0..audio_samples.len()).step_by(step_size) {
            let end = (start + chunk_samples).min(audio_samples.len());
            let mut chunk_data = audio_samples[start..end].to_vec();

            // Pad if needed
            if chunk_data.len() < chunk_samples {
                chunk_data.resize(chunk_samples, 0.0);
            }

            // Create audio buffer for chunk
            let chunk_buffer = AudioBuffer::new(chunk_data, self.config.sample_rate, 1);

            let mel_tensor = self.process_audio(&chunk_buffer)?;
            chunks.push(mel_tensor);

            if end >= audio_samples.len() {
                break;
            }
        }

        Ok(chunks)
    }

    fn preprocess_audio(&self, audio: &AudioBuffer) -> Vec<f32> {
        let mut samples = audio.samples().to_vec();

        // Ensure correct sample rate (16kHz)
        if audio.sample_rate() != self.config.sample_rate {
            samples = Self::resample(&samples, audio.sample_rate(), self.config.sample_rate);
        }

        // Convert to mono if stereo
        if audio.channels() > 1 {
            let mono_samples = samples
                .chunks(audio.channels() as usize)
                .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
                .collect();
            samples = mono_samples;
        }

        // Pad or trim to 30 seconds (480,000 samples at 16kHz)
        let target_length = self.config.sample_rate as usize * 30;
        samples.resize(target_length, 0.0);

        samples
    }

    fn compute_stft(&self, samples: &[f32]) -> Result<Tensor, RecognitionError> {
        let n_fft = 1024;
        let hop_length = 160;
        let n_frames = (samples.len() - n_fft) / hop_length + 1;
        let n_freqs = n_fft / 2 + 1;

        // Use scirs2_fft for FFT computation
        let mut stft_real = vec![0.0f32; n_frames * n_freqs];
        let mut stft_imag = vec![0.0f32; n_frames * n_freqs];

        let window_data =
            self.window
                .to_vec1::<f32>()
                .map_err(|e| RecognitionError::AudioProcessingError {
                    message: format!("Failed to extract window data: {e}"),
                    source: Some(Box::new(e)),
                })?;

        for (frame_idx, frame_start) in (0..=(samples.len() - n_fft))
            .step_by(hop_length)
            .enumerate()
        {
            if frame_idx >= n_frames {
                break;
            }

            // Apply window and prepare for FFT
            let windowed_frame: Vec<f64> = (0..n_fft)
                .map(|i| {
                    let sample = if frame_start + i < samples.len() {
                        samples[frame_start + i] * window_data[i]
                    } else {
                        0.0
                    };
                    sample as f64
                })
                .collect();

            // Compute real FFT using scirs2_fft functional API
            let spectrum = scirs2_fft::rfft(&windowed_frame, None).map_err(|e| {
                RecognitionError::AudioProcessingError {
                    message: format!("FFT computation failed: {e}"),
                    source: None,
                }
            })?;

            // Store magnitude spectrum (only positive frequencies)
            for (freq_idx, &fft_val) in spectrum.iter().take(n_freqs).enumerate() {
                let magnitude = (fft_val.re * fft_val.re + fft_val.im * fft_val.im).sqrt() as f32;
                stft_real[frame_idx * n_freqs + freq_idx] = magnitude;
                stft_imag[frame_idx * n_freqs + freq_idx] = 0.0; // We only need magnitude
            }
        }

        // Return magnitude spectrogram
        Tensor::from_slice(&stft_real, (n_frames, n_freqs), self.mel_filters.device()).map_err(
            |e| RecognitionError::AudioProcessingError {
                message: format!("Failed to create STFT tensor: {e}"),
                source: Some(Box::new(e)),
            },
        )
    }

    fn apply_mel_filters(&self, stft: &Tensor) -> Result<Tensor, RecognitionError> {
        // stft: [n_frames, n_freqs], mel_filters: [n_mels, n_freqs]
        // output: [n_frames, n_mels]
        stft.matmul(
            &self
                .mel_filters
                .t()
                .map_err(|e| RecognitionError::AudioProcessingError {
                    message: format!("Failed to transpose mel filters: {e}"),
                    source: Some(Box::new(e)),
                })?,
        )
        .map_err(|e| RecognitionError::AudioProcessingError {
            message: format!("Failed to apply mel filters: {e}"),
            source: Some(Box::new(e)),
        })
    }

    #[allow(clippy::unused_self)]
    fn log_compress(&self, mel_spec: &Tensor) -> Result<Tensor, RecognitionError> {
        // Apply log compression: log(max(mel_spec, 1e-10))
        let epsilon = 1e-10f32;
        let clamped = mel_spec.clamp(epsilon, f32::MAX).map_err(|e| {
            RecognitionError::AudioProcessingError {
                message: format!("Failed to clamp mel spectrogram: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        clamped
            .log()
            .map_err(|e| RecognitionError::AudioProcessingError {
                message: format!("Failed to apply log compression: {e}"),
                source: Some(Box::new(e)),
            })
    }

    #[allow(clippy::unused_self)]
    fn normalize_mel(&self, log_mel: &Tensor) -> Result<Tensor, RecognitionError> {
        // Whisper normalization: subtract global mean, divide by global std
        // These are the values used by OpenAI Whisper
        let mean = -4.267_739_3_f32;
        let std = 4.568_997_4_f32;

        let mean_tensor = Tensor::new(&[mean], log_mel.device()).map_err(|e| {
            RecognitionError::AudioProcessingError {
                message: format!("Failed to create mean tensor: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        let std_tensor = Tensor::new(&[std], log_mel.device()).map_err(|e| {
            RecognitionError::AudioProcessingError {
                message: format!("Failed to create std tensor: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        let normalized = log_mel.broadcast_sub(&mean_tensor).map_err(|e| {
            RecognitionError::AudioProcessingError {
                message: format!("Failed to subtract mean: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        normalized
            .broadcast_div(&std_tensor)
            .map_err(|e| RecognitionError::AudioProcessingError {
                message: format!("Failed to divide by std: {e}"),
                source: Some(Box::new(e)),
            })
    }

    fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        if from_rate == to_rate {
            return samples.to_vec();
        }

        // Simple linear interpolation resampling
        let ratio = from_rate as f32 / to_rate as f32;
        let new_length = (samples.len() as f32 / ratio) as usize;
        let mut resampled = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let src_index = i as f32 * ratio;
            let src_index_floor = src_index.floor() as usize;
            let src_index_ceil = (src_index_floor + 1).min(samples.len() - 1);

            let frac = src_index - src_index_floor as f32;
            let sample = samples[src_index_floor] * (1.0 - frac) + samples[src_index_ceil] * frac;
            resampled.push(sample);
        }

        resampled
    }

    /// Apply voice activity detection to identify speech segments
    ///
    /// # Errors
    ///
    /// Returns an error if audio preprocessing fails.
    pub fn detect_voice_activity(
        &self,
        audio: &AudioBuffer,
        threshold: f32,
    ) -> Result<Vec<(f32, f32)>, RecognitionError> {
        let samples = self.preprocess_audio(audio);
        let frame_size = 1024;
        let hop_size = 160;
        let mut segments = Vec::new();
        let mut current_start = None;

        for (i, frame_start) in (0..=(samples.len() - frame_size))
            .step_by(hop_size)
            .enumerate()
        {
            let frame = &samples[frame_start..frame_start + frame_size];
            let energy = frame.iter().map(|&x| x * x).sum::<f32>() / frame.len() as f32;
            let rms = energy.sqrt();

            let time = i as f32 * hop_size as f32 / self.config.sample_rate as f32;

            if rms > threshold {
                if current_start.is_none() {
                    current_start = Some(time);
                }
            } else if let Some(start) = current_start {
                segments.push((start, time));
                current_start = None;
            }
        }

        // Handle case where speech continues to the end
        if let Some(start) = current_start {
            let end_time = samples.len() as f32 / self.config.sample_rate as f32;
            segments.push((start, end_time));
        }

        Ok(segments)
    }

    // Helper functions for mel scale conversion
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Get audio processing configuration
    #[must_use]
    /// config
    pub fn config(&self) -> &WhisperConfig {
        &self.config
    }

    /// Get mel filter bank
    #[must_use]
    /// mel filters
    pub fn mel_filters(&self) -> &Tensor {
        &self.mel_filters
    }

    /// Get window function
    #[must_use]
    /// window
    pub fn window(&self) -> &Tensor {
        &self.window
    }
}
