//! Audio buffer utility methods for common operations.
//!
//! This module provides convenient utility methods for AudioBuffer manipulation,
//! including concatenation, mixing, normalization, and effects.

use super::AudioBuffer;
use crate::error::Result;
use crate::VoirsError;

/// Audio buffer utilities
impl AudioBuffer {
    /// Concatenate multiple audio buffers
    ///
    /// Buffers must have the same sample rate and channel configuration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buf1 = AudioBuffer::mono(vec![1.0, 2.0, 3.0], 22050);
    /// let buf2 = AudioBuffer::mono(vec![4.0, 5.0, 6.0], 22050);
    /// let buf3 = AudioBuffer::mono(vec![7.0, 8.0, 9.0], 22050);
    ///
    /// let concatenated = AudioBuffer::concatenate(&[buf1, buf2, buf3]).expect("value should be present");
    /// assert_eq!(concatenated.len(), 9);
    /// ```
    pub fn concatenate(buffers: &[AudioBuffer]) -> Result<Self> {
        if buffers.is_empty() {
            return Err(VoirsError::AudioError {
                buffer_info: None,
                message: "Cannot concatenate empty buffer list".to_string(),
            });
        }

        let first = &buffers[0];
        let sample_rate = first.sample_rate;
        let channels = first.channels;

        // Validate all buffers have same format
        for buffer in buffers.iter().skip(1) {
            if buffer.sample_rate != sample_rate {
                return Err(VoirsError::AudioError {
                    buffer_info: None,
                    message: format!(
                        "Sample rate mismatch: {} != {}",
                        buffer.sample_rate, sample_rate
                    ),
                });
            }
            if buffer.channels != channels {
                return Err(VoirsError::AudioError {
                    buffer_info: None,
                    message: format!(
                        "Channel count mismatch: {} != {}",
                        buffer.channels, channels
                    ),
                });
            }
        }

        // Concatenate samples
        let total_samples: usize = buffers.iter().map(|b| b.samples.len()).sum();
        let mut samples = Vec::with_capacity(total_samples);

        for buffer in buffers {
            samples.extend_from_slice(&buffer.samples);
        }

        Ok(Self::new(samples, sample_rate, channels))
    }

    /// Pad buffer with silence
    ///
    /// Adds silence before and/or after the audio.
    ///
    /// # Arguments
    ///
    /// * `before_seconds` - Silence duration before audio
    /// * `after_seconds` - Silence duration after audio
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let mut buffer = AudioBuffer::mono(vec![1.0; 100], 22050);
    /// buffer.pad(0.1, 0.2); // Add 0.1s before, 0.2s after
    /// ```
    pub fn pad(&mut self, before_seconds: f32, after_seconds: f32) {
        let before_samples =
            (before_seconds * self.sample_rate as f32 * self.channels as f32) as usize;
        let after_samples =
            (after_seconds * self.sample_rate as f32 * self.channels as f32) as usize;

        let mut new_samples =
            Vec::with_capacity(before_samples + self.samples.len() + after_samples);
        new_samples.resize(before_samples, 0.0);
        new_samples.extend_from_slice(&self.samples);
        new_samples.resize(new_samples.len() + after_samples, 0.0);

        self.samples = new_samples;
        self.update_metadata();
    }

    /// Check if buffer contains clipping (samples outside [-1.0, 1.0])
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5, 1.5, 0.3], 22050);
    /// assert!(buffer.has_clipping());
    /// ```
    pub fn has_clipping(&self) -> bool {
        self.samples.iter().any(|&s| s.abs() > 1.0)
    }

    /// Count number of clipped samples
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5, 1.5, -1.2, 0.3], 22050);
    /// assert_eq!(buffer.count_clipped_samples(), 2);
    /// ```
    pub fn count_clipped_samples(&self) -> usize {
        self.samples.iter().filter(|&&s| s.abs() > 1.0).count()
    }

    /// Get RMS (Root Mean Square) level in dB
    ///
    /// Returns -∞ for silence.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 1000], 22050);
    /// let rms_db = buffer.rms_db();
    /// ```
    pub fn rms_db(&self) -> f32 {
        if self.metadata.rms_amplitude > 0.0 {
            20.0 * self.metadata.rms_amplitude.log10()
        } else {
            f32::NEG_INFINITY
        }
    }

    /// Get peak level in dB
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 1000], 22050);
    /// let peak_db = buffer.peak_db();
    /// ```
    pub fn peak_db(&self) -> f32 {
        if self.metadata.peak_amplitude > 0.0 {
            20.0 * self.metadata.peak_amplitude.log10()
        } else {
            f32::NEG_INFINITY
        }
    }

    /// Calculate zero-crossing rate (ZCR)
    ///
    /// ZCR is the rate at which the signal changes sign, useful for
    /// voice activity detection and audio classification.
    ///
    /// # Returns
    ///
    /// The zero-crossing rate as a fraction of the total samples.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5, -0.3, 0.2, -0.1, 0.4], 22050);
    /// let zcr = buffer.zero_crossing_rate();
    /// println!("Zero-crossing rate: {:.4}", zcr);
    /// ```
    pub fn zero_crossing_rate(&self) -> f32 {
        if self.samples.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 1..self.samples.len() {
            if (self.samples[i] >= 0.0 && self.samples[i - 1] < 0.0)
                || (self.samples[i] < 0.0 && self.samples[i - 1] >= 0.0)
            {
                crossings += 1;
            }
        }

        crossings as f32 / (self.samples.len() - 1) as f32
    }

    /// Calculate spectral centroid
    ///
    /// The spectral centroid is the "center of mass" of the spectrum,
    /// indicating where the majority of the signal's energy is concentrated.
    /// Higher values indicate brighter sounds.
    ///
    /// # Returns
    ///
    /// The spectral centroid in Hz, or 0.0 if calculation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 1024], 22050);
    /// let centroid = buffer.spectral_centroid();
    /// println!("Spectral centroid: {:.2} Hz", centroid);
    /// ```
    pub fn spectral_centroid(&self) -> f32 {
        // Need at least enough samples for meaningful analysis
        if self.samples.len() < 64 {
            return 0.0;
        }

        // Use next power of 2 for FFT efficiency
        let fft_size = self.samples.len().next_power_of_two().min(2048);
        let samples_to_analyze = self.samples.len().min(fft_size);

        // Prepare complex buffer for FFT (as f64 for scirs2-fft)
        let input: Vec<f64> = self.samples[..samples_to_analyze]
            .iter()
            .map(|&s| s as f64)
            .chain(std::iter::repeat(0.0))
            .take(fft_size)
            .collect();

        // Perform FFT
        let spectrum = match scirs2_fft::fft(&input, Some(fft_size)) {
            Ok(result) => result,
            Err(_) => return 0.0,
        };

        // Calculate magnitude spectrum
        let magnitudes: Vec<f32> = spectrum
            .iter()
            .take(fft_size / 2)
            .map(|c| c.norm() as f32)
            .collect();

        // Calculate weighted sum for centroid
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &mag) in magnitudes.iter().enumerate() {
            let freq = i as f32 * self.sample_rate as f32 / fft_size as f32;
            weighted_sum += freq * mag;
            magnitude_sum += mag;
        }

        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }

    /// Calculate spectral rolloff
    ///
    /// The spectral rolloff is the frequency below which a specified percentage
    /// (typically 85%) of the total spectral energy is contained.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The energy threshold (0.0 to 1.0), typically 0.85
    ///
    /// # Returns
    ///
    /// The rolloff frequency in Hz, or 0.0 if calculation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 1024], 22050);
    /// let rolloff = buffer.spectral_rolloff(0.85);
    /// println!("Spectral rolloff: {:.2} Hz", rolloff);
    /// ```
    pub fn spectral_rolloff(&self, threshold: f32) -> f32 {
        if self.samples.len() < 64 {
            return 0.0;
        }

        let fft_size = self.samples.len().next_power_of_two().min(2048);
        let samples_to_analyze = self.samples.len().min(fft_size);

        let input: Vec<f64> = self.samples[..samples_to_analyze]
            .iter()
            .map(|&s| s as f64)
            .chain(std::iter::repeat(0.0))
            .take(fft_size)
            .collect();

        let spectrum = match scirs2_fft::fft(&input, Some(fft_size)) {
            Ok(result) => result,
            Err(_) => return 0.0,
        };

        let magnitudes: Vec<f32> = spectrum
            .iter()
            .take(fft_size / 2)
            .map(|c| c.norm() as f32)
            .collect();

        // Calculate total energy
        let total_energy: f32 = magnitudes.iter().sum();
        if total_energy == 0.0 {
            return 0.0;
        }

        // Find rolloff frequency
        let target_energy = total_energy * threshold;
        let mut cumulative_energy = 0.0;

        for (i, &mag) in magnitudes.iter().enumerate() {
            cumulative_energy += mag;
            if cumulative_energy >= target_energy {
                return i as f32 * self.sample_rate as f32 / fft_size as f32;
            }
        }

        // If we didn't reach threshold, return nyquist frequency
        self.sample_rate as f32 / 2.0
    }

    /// Calculate signal-to-noise ratio (SNR) in dB
    ///
    /// Estimates SNR by comparing signal power to noise floor power.
    /// Uses a simple heuristic: assumes the quietest 10% of frames represent noise.
    ///
    /// # Returns
    ///
    /// SNR in dB, or 0.0 if calculation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 1000], 22050);
    /// let snr = buffer.signal_to_noise_ratio();
    /// println!("SNR: {:.2} dB", snr);
    /// ```
    pub fn signal_to_noise_ratio(&self) -> f32 {
        if self.samples.len() < 100 {
            return 0.0;
        }

        // Split audio into frames
        let frame_size = 512;
        let hop_size = frame_size / 2;
        let num_frames = (self.samples.len() - frame_size) / hop_size + 1;

        if num_frames < 10 {
            return 0.0;
        }

        // Calculate RMS for each frame
        let mut frame_rms: Vec<f32> = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            let end = (start + frame_size).min(self.samples.len());

            let rms: f32 =
                self.samples[start..end].iter().map(|&s| s * s).sum::<f32>() / (end - start) as f32;

            frame_rms.push(rms.sqrt());
        }

        // Sort to find noise floor (bottom 10%), treating NaN as max value
        frame_rms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let noise_threshold_idx = (frame_rms.len() as f32 * 0.1) as usize;
        let noise_floor = frame_rms[..noise_threshold_idx.max(1)].iter().sum::<f32>()
            / noise_threshold_idx.max(1) as f32;

        // Signal power is mean of all frames
        let signal_power = frame_rms.iter().sum::<f32>() / frame_rms.len() as f32;

        if noise_floor > 0.0 && signal_power > 0.0 {
            20.0 * (signal_power / noise_floor).log10()
        } else {
            0.0
        }
    }

    /// Calculate crest factor (peak-to-RMS ratio) in dB
    ///
    /// Crest factor indicates the dynamic range of the audio.
    /// Higher values indicate more dynamic content with prominent peaks.
    ///
    /// # Returns
    ///
    /// Crest factor in dB.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 1000], 22050);
    /// let crest = buffer.crest_factor();
    /// println!("Crest factor: {:.2} dB", crest);
    /// ```
    pub fn crest_factor(&self) -> f32 {
        let peak = self.metadata.peak_amplitude;
        let rms = self.metadata.rms_amplitude;

        if rms > 0.0 {
            20.0 * (peak / rms).log10()
        } else {
            f32::INFINITY
        }
    }

    /// Detect silence segments in the audio
    ///
    /// Returns a list of (start_time, end_time) tuples in seconds representing
    /// detected silence segments.
    ///
    /// # Arguments
    ///
    /// * `threshold_db` - Silence threshold in dB (e.g., -40.0)
    /// * `min_duration` - Minimum silence duration in seconds
    ///
    /// # Returns
    ///
    /// Vector of (start, end) time pairs in seconds.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 1000], 22050);
    /// let silences = buffer.detect_silence(-40.0, 0.1);
    /// for (start, end) in silences {
    ///     println!("Silence: {:.2}s to {:.2}s", start, end);
    /// }
    /// ```
    pub fn detect_silence(&self, threshold_db: f32, min_duration: f32) -> Vec<(f32, f32)> {
        let threshold_amplitude = 10.0_f32.powf(threshold_db / 20.0);
        let min_samples = (min_duration * self.sample_rate as f32) as usize;

        let frame_size = 512;
        let hop_size = frame_size / 2;

        let mut silence_segments = Vec::new();
        let mut silence_start: Option<usize> = None;

        let mut frame_idx = 0;
        while frame_idx * hop_size < self.samples.len() {
            let start = frame_idx * hop_size;
            let end = (start + frame_size).min(self.samples.len());

            // Calculate frame RMS
            let rms = (self.samples[start..end].iter().map(|&s| s * s).sum::<f32>()
                / (end - start) as f32)
                .sqrt();

            if rms < threshold_amplitude {
                // Silent frame
                if silence_start.is_none() {
                    silence_start = Some(start);
                }
            } else {
                // Non-silent frame
                if let Some(start_sample) = silence_start {
                    let duration_samples = start - start_sample;
                    if duration_samples >= min_samples {
                        let start_time = start_sample as f32 / self.sample_rate as f32;
                        let end_time = start as f32 / self.sample_rate as f32;
                        silence_segments.push((start_time, end_time));
                    }
                    silence_start = None;
                }
            }

            frame_idx += 1;
        }

        // Handle trailing silence
        if let Some(start_sample) = silence_start {
            let duration_samples = self.samples.len() - start_sample;
            if duration_samples >= min_samples {
                let start_time = start_sample as f32 / self.sample_rate as f32;
                let end_time = self.samples.len() as f32 / self.sample_rate as f32;
                silence_segments.push((start_time, end_time));
            }
        }

        silence_segments
    }

    /// Calculate Mel-Frequency Cepstral Coefficients (MFCCs)
    ///
    /// MFCCs are widely used in speech and audio processing for feature extraction.
    /// They represent the short-term power spectrum of a sound on a mel scale.
    ///
    /// # Arguments
    ///
    /// * `num_coeffs` - Number of MFCC coefficients to extract (typically 13)
    /// * `num_filters` - Number of mel-scale filters (typically 26-40)
    /// * `fft_size` - FFT size (power of 2, typically 512-2048)
    ///
    /// # Returns
    ///
    /// Vector of MFCC coefficients, or empty vector if calculation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 1024], 22050);
    /// let mfccs = buffer.mfcc(13, 26, 512);
    /// println!("MFCC coefficients: {:?}", mfccs);
    /// ```
    pub fn mfcc(&self, num_coeffs: usize, num_filters: usize, fft_size: usize) -> Vec<f32> {
        if self.samples.len() < fft_size || !fft_size.is_power_of_two() {
            return Vec::new();
        }

        // Prepare input with windowing (Hamming window)
        let window: Vec<f64> = (0..fft_size)
            .map(|i| {
                0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (fft_size - 1) as f64).cos()
            })
            .collect();

        let samples_to_analyze = self.samples.len().min(fft_size);
        let input: Vec<f64> = self.samples[..samples_to_analyze]
            .iter()
            .enumerate()
            .map(|(i, &s)| s as f64 * window[i])
            .chain(std::iter::repeat(0.0))
            .take(fft_size)
            .collect();

        // Perform FFT
        let spectrum = match scirs2_fft::fft(&input, Some(fft_size)) {
            Ok(result) => result,
            Err(_) => return Vec::new(),
        };

        // Calculate power spectrum
        let power_spectrum: Vec<f32> = spectrum
            .iter()
            .take(fft_size / 2 + 1)
            .map(|c| (c.norm() * c.norm()) as f32)
            .collect();

        // Create mel filterbank
        let mel_filters =
            Self::create_mel_filterbank(num_filters, fft_size, self.sample_rate as f32);

        // Apply mel filterbank
        let mut mel_energies = vec![0.0f32; num_filters];
        for (filter_idx, filter) in mel_filters.iter().enumerate() {
            for (bin_idx, &power) in power_spectrum.iter().enumerate() {
                if bin_idx < filter.len() {
                    mel_energies[filter_idx] += power * filter[bin_idx];
                }
            }
        }

        // Apply log and handle zeros
        let log_mel: Vec<f32> = mel_energies
            .iter()
            .map(|&e| if e > 1e-10 { e.ln() } else { -23.0 }) // ln(1e-10) ≈ -23
            .collect();

        // Apply DCT (Discrete Cosine Transform) Type-II
        let mut mfcc_coeffs = vec![0.0f32; num_coeffs];
        // Note: We need the index i for DCT calculation, not for direct indexing
        #[allow(clippy::needless_range_loop)]
        for i in 0..num_coeffs {
            let mut sum = 0.0;
            for (k, &log_e) in log_mel.iter().enumerate() {
                sum += log_e
                    * ((std::f32::consts::PI * i as f32 * (k as f32 + 0.5)) / num_filters as f32)
                        .cos();
            }
            mfcc_coeffs[i] = sum * (2.0 / num_filters as f32).sqrt();
        }

        mfcc_coeffs
    }

    /// Create mel-scale filterbank
    ///
    /// Helper function for MFCC calculation.
    fn create_mel_filterbank(
        num_filters: usize,
        fft_size: usize,
        sample_rate: f32,
    ) -> Vec<Vec<f32>> {
        // Mel scale conversion functions
        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

        let nyquist = sample_rate / 2.0;
        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(nyquist);

        // Create mel-spaced frequency points
        let mel_points: Vec<f32> = (0..=num_filters + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (num_filters + 1) as f32)
            .map(mel_to_hz)
            .collect();

        // Convert to FFT bin indices
        let bin_points: Vec<usize> = mel_points
            .iter()
            .map(|&freq| ((fft_size + 1) as f32 * freq / sample_rate).floor() as usize)
            .collect();

        // Create triangular filters
        let mut filterbank = Vec::with_capacity(num_filters);
        for i in 0..num_filters {
            let mut filter = vec![0.0f32; fft_size / 2 + 1];

            let left = bin_points[i];
            let center = bin_points[i + 1];
            let right = bin_points[i + 2];

            // Rising slope
            // Note: We need k for arithmetic, not just indexing
            #[allow(clippy::needless_range_loop)]
            for k in left..center {
                if center != left {
                    filter[k] = (k - left) as f32 / (center - left) as f32;
                }
            }

            // Falling slope
            // Note: We need k for arithmetic, not just indexing
            #[allow(clippy::needless_range_loop)]
            for k in center..right {
                if right != center {
                    filter[k] = (right - k) as f32 / (right - center) as f32;
                }
            }

            filterbank.push(filter);
        }

        filterbank
    }

    /// Detect pitch using autocorrelation
    ///
    /// Uses the autocorrelation method to estimate the fundamental frequency (F0)
    /// of the audio signal. This is more robust than simple zero-crossing rate.
    ///
    /// # Arguments
    ///
    /// * `min_freq` - Minimum expected frequency in Hz (e.g., 80 for male voice)
    /// * `max_freq` - Maximum expected frequency in Hz (e.g., 400 for female voice)
    ///
    /// # Returns
    ///
    /// Estimated pitch in Hz, or 0.0 if no pitch detected.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 1024], 22050);
    /// let pitch = buffer.detect_pitch_autocorr(80.0, 400.0);
    /// println!("Detected pitch: {:.2} Hz", pitch);
    /// ```
    pub fn detect_pitch_autocorr(&self, min_freq: f32, max_freq: f32) -> f32 {
        if self.samples.len() < 1024 {
            return 0.0;
        }

        let sample_rate = self.sample_rate as f32;
        let min_lag = (sample_rate / max_freq) as usize;
        let max_lag = (sample_rate / min_freq) as usize;

        // Use a reasonable window for analysis
        let window_size = (max_lag * 3).min(self.samples.len());
        let samples = &self.samples[..window_size];

        // Compute autocorrelation using normalized method
        let mut max_corr = 0.0;
        let mut best_lag = 0;

        for lag in min_lag..=max_lag.min(window_size / 2) {
            let mut sum = 0.0;
            let mut energy1 = 0.0;
            let mut energy2 = 0.0;

            for i in 0..(window_size - lag) {
                let s1 = samples[i];
                let s2 = samples[i + lag];
                sum += s1 * s2;
                energy1 += s1 * s1;
                energy2 += s2 * s2;
            }

            // Normalized autocorrelation
            let corr = if energy1 > 0.0 && energy2 > 0.0 {
                sum / (energy1 * energy2).sqrt()
            } else {
                0.0
            };

            if corr > max_corr {
                max_corr = corr;
                best_lag = lag;
            }
        }

        // Confidence threshold for pitch detection
        if max_corr > 0.5 && best_lag > 0 {
            sample_rate / best_lag as f32
        } else {
            0.0
        }
    }

    /// Calculate spectral flux
    ///
    /// Spectral flux measures the rate of change in the power spectrum,
    /// useful for detecting onsets and transients in audio.
    ///
    /// # Arguments
    ///
    /// * `prev_buffer` - Previous audio buffer for comparison (optional)
    /// * `fft_size` - FFT size for spectral analysis
    ///
    /// # Returns
    ///
    /// Spectral flux value, or 0.0 if calculation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer1 = AudioBuffer::mono(vec![0.5; 1024], 22050);
    /// let buffer2 = AudioBuffer::mono(vec![0.6; 1024], 22050);
    /// let flux = buffer2.spectral_flux(Some(&buffer1), 512);
    /// println!("Spectral flux: {:.4}", flux);
    /// ```
    pub fn spectral_flux(&self, prev_buffer: Option<&AudioBuffer>, fft_size: usize) -> f32 {
        if self.samples.len() < fft_size || !fft_size.is_power_of_two() {
            return 0.0;
        }

        // Get current spectrum
        let current_spectrum = self.get_magnitude_spectrum(fft_size);

        // If no previous buffer, return 0
        let prev_spectrum = match prev_buffer {
            Some(buf) => buf.get_magnitude_spectrum(fft_size),
            None => return 0.0,
        };

        if current_spectrum.is_empty() || prev_spectrum.is_empty() {
            return 0.0;
        }

        // Calculate flux as sum of squared differences (positive only)
        let mut flux = 0.0;
        let len = current_spectrum.len().min(prev_spectrum.len());

        for i in 0..len {
            let diff = current_spectrum[i] - prev_spectrum[i];
            if diff > 0.0 {
                flux += diff * diff;
            }
        }

        flux.sqrt()
    }

    /// Get magnitude spectrum (helper for spectral flux)
    fn get_magnitude_spectrum(&self, fft_size: usize) -> Vec<f32> {
        let samples_to_analyze = self.samples.len().min(fft_size);
        let input: Vec<f64> = self.samples[..samples_to_analyze]
            .iter()
            .map(|&s| s as f64)
            .chain(std::iter::repeat(0.0))
            .take(fft_size)
            .collect();

        match scirs2_fft::fft(&input, Some(fft_size)) {
            Ok(spectrum) => spectrum
                .iter()
                .take(fft_size / 2)
                .map(|c| c.norm() as f32)
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Estimate formant frequencies
    ///
    /// Formants are resonant frequencies of the vocal tract, crucial for
    /// vowel identification and speaker characteristics. This uses LPC
    /// (Linear Predictive Coding) analysis to estimate the first 4 formants.
    ///
    /// # Arguments
    ///
    /// * `num_formants` - Number of formants to estimate (typically 3-4)
    ///
    /// # Returns
    ///
    /// Vector of estimated formant frequencies in Hz.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 1024], 22050);
    /// let formants = buffer.estimate_formants(4);
    /// println!("Formants: {:?} Hz", formants);
    /// ```
    pub fn estimate_formants(&self, num_formants: usize) -> Vec<f32> {
        if self.samples.len() < 512 {
            return Vec::new();
        }

        // Use LPC order = 2 + sample_rate/1000 (common rule of thumb)
        let lpc_order = 2 + (self.sample_rate / 1000) as usize;

        // Window the signal (use up to 1024 samples)
        let window_size = self.samples.len().min(1024);
        let samples = &self.samples[..window_size];

        // Apply pre-emphasis filter (high-pass)
        let pre_emphasis = 0.97;
        let mut emphasized = vec![samples[0]];
        for i in 1..samples.len() {
            emphasized.push(samples[i] - pre_emphasis * samples[i - 1]);
        }

        // Compute autocorrelation
        let mut autocorr = vec![0.0; lpc_order + 1];
        for lag in 0..=lpc_order {
            let mut sum = 0.0;
            for i in 0..(emphasized.len() - lag) {
                sum += emphasized[i] * emphasized[i + lag];
            }
            autocorr[lag] = sum;
        }

        // Solve for LPC coefficients using Levinson-Durbin algorithm
        let lpc_coeffs = self.levinson_durbin(&autocorr, lpc_order);

        // Find roots of LPC polynomial to get formants
        // For simplicity, we use peak-picking in the frequency response
        let fft_size = 1024;
        let num_bins = fft_size / 2;
        let mut freq_response = vec![0.0; num_bins];

        // Note: We need bin index for omega calculation, not just for indexing
        #[allow(clippy::needless_range_loop)]
        for bin in 0..num_bins {
            let omega = 2.0 * std::f32::consts::PI * bin as f32 / fft_size as f32;
            let mut real_part = 1.0;
            let mut imag_part = 0.0;

            for (k, &coeff) in lpc_coeffs.iter().enumerate() {
                let angle = omega * (k + 1) as f32;
                real_part -= coeff * angle.cos();
                imag_part += coeff * angle.sin();
            }

            freq_response[bin] = 1.0 / (real_part * real_part + imag_part * imag_part).sqrt();
        }

        // Find peaks in frequency response and refine each peak to sub-bin
        // precision via parabolic (quadratic) interpolation.
        //
        // A raw local maximum only resolves the formant to the nearest FFT
        // bin. The true resonance peak usually lies between bins, so we fit a
        // parabola through the three magnitudes surrounding the discrete peak
        // (`y(k-1)`, `y(k)`, `y(k+1)`) and take the parabola's vertex as the
        // refined location:
        //
        //   delta = 0.5 * (y(k-1) - y(k+1)) / (y(k-1) - 2*y(k) + y(k+1))
        //
        // `delta` is the fractional-bin offset of the vertex from bin `k`. The
        // denominator is the discrete second difference (curvature) and is
        // guarded against ~0 (a flat/degenerate triple) to avoid division
        // blow-up. `delta` is clamped to [-0.5, +0.5] because the vertex of a
        // genuine local maximum cannot fall outside the central bin's
        // half-bin neighbourhood. The refined center frequency then becomes
        // `(k + delta) * bin_spacing`. For a symmetric peak (y(k-1) == y(k+1))
        // the offset is 0 and the result equals the plain bin-center
        // frequency; for an asymmetric peak the frequency shifts toward the
        // larger-magnitude neighbour.
        let bin_spacing = self.sample_rate as f32 / fft_size as f32;
        let mut formants = Vec::new();
        for k in 1..(num_bins - 1) {
            if freq_response[k] > freq_response[k - 1] && freq_response[k] > freq_response[k + 1] {
                let y_prev = freq_response[k - 1];
                let y_peak = freq_response[k];
                let y_next = freq_response[k + 1];

                // Parabolic interpolation for sub-bin accuracy.
                let delta = Self::parabolic_peak_offset(y_prev, y_peak, y_next);
                let freq = (k as f32 + delta) * bin_spacing;
                // Typical formant range: 200 Hz to 4000 Hz
                if freq >= 200.0 && freq <= 4000.0 {
                    formants.push(freq);
                    if formants.len() >= num_formants {
                        break;
                    }
                }
            }
        }

        formants
    }

    /// Refine a discrete spectral peak to sub-bin precision via parabolic
    /// (quadratic) interpolation.
    ///
    /// Given the magnitudes at a local maximum bin `k` and its two immediate
    /// neighbours — `y_prev = y(k-1)`, `y_peak = y(k)`, `y_next = y(k+1)` —
    /// this fits a parabola through the three points and returns the
    /// fractional-bin offset `delta` of the parabola's vertex from bin `k`:
    ///
    /// ```text
    /// delta = 0.5 * (y(k-1) - y(k+1)) / (y(k-1) - 2*y(k) + y(k+1))
    /// ```
    ///
    /// The refined peak location is therefore `(k + delta)` bins. The
    /// denominator is the discrete curvature (second difference); when it is
    /// ~0 (a flat or degenerate triple) the offset is forced to `0.0` to avoid
    /// a division blow-up. The result is clamped to `[-0.5, +0.5]` since the
    /// vertex of a genuine local maximum lies within the central bin's
    /// half-bin neighbourhood.
    ///
    /// Behaviour:
    /// - Symmetric peak (`y_prev == y_next`) → `delta == 0.0`.
    /// - Larger left neighbour (`y_prev > y_next`) → `delta < 0.0` (shift down).
    /// - Larger right neighbour (`y_next > y_prev`) → `delta > 0.0` (shift up).
    #[inline]
    fn parabolic_peak_offset(y_prev: f32, y_peak: f32, y_next: f32) -> f32 {
        let denom = y_prev - 2.0 * y_peak + y_next;
        if denom.abs() > 1e-12 {
            (0.5 * (y_prev - y_next) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        }
    }

    /// Levinson-Durbin algorithm for LPC coefficient estimation
    ///
    /// Solves the Yule-Walker equations efficiently.
    fn levinson_durbin(&self, autocorr: &[f32], order: usize) -> Vec<f32> {
        let mut lpc = vec![0.0; order];
        let mut error = autocorr[0];

        for i in 0..order {
            let mut lambda = 0.0;
            for j in 0..i {
                lambda -= lpc[j] * autocorr[i - j];
            }
            lambda -= autocorr[i + 1];
            lambda /= error;

            // Update LPC coefficients
            let mut new_lpc = vec![0.0; order];
            new_lpc[i] = lambda;
            for j in 0..i {
                new_lpc[j] = lpc[j] + lambda * lpc[i - 1 - j];
            }
            lpc = new_lpc;

            error *= 1.0 - lambda * lambda;
        }

        lpc
    }

    /// Calculate Jitter (pitch period irregularity)
    ///
    /// Jitter measures the cycle-to-cycle variation in fundamental frequency (F0),
    /// expressed as a percentage. It's a crucial indicator of voice quality and
    /// pathology. Higher jitter indicates more irregular vocal fold vibration.
    ///
    /// This implements the Jitter (local) metric, which is the average absolute
    /// difference between consecutive periods, divided by the average period.
    ///
    /// # Arguments
    ///
    /// * `min_freq` - Minimum expected pitch (e.g., 75 Hz for male voice)
    /// * `max_freq` - Maximum expected pitch (e.g., 500 Hz for female voice)
    ///
    /// # Returns
    ///
    /// Jitter percentage (0-100), or 0.0 if calculation fails. Typical values:
    /// - < 1.0%: Normal voice quality
    /// - 1.0-2.0%: Mild irregularity
    /// - > 2.0%: Potential voice pathology
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 4096], 22050);
    /// let jitter = buffer.calculate_jitter(75.0, 500.0);
    /// println!("Jitter: {:.2}%", jitter);
    /// ```
    pub fn calculate_jitter(&self, min_freq: f32, max_freq: f32) -> f32 {
        if self.samples.len() < 2048 {
            return 0.0;
        }

        let sample_rate = self.sample_rate as f32;
        let min_period = (sample_rate / max_freq) as usize;
        let max_period = (sample_rate / min_freq) as usize;

        // Find pitch periods using autocorrelation
        let window_size = (max_period * 4).min(self.samples.len());
        let mut periods = Vec::new();

        // Slide through the signal to find multiple periods
        let hop_size = max_period / 2;
        for start_idx in (0..self.samples.len() - window_size).step_by(hop_size) {
            let window = &self.samples[start_idx..start_idx + window_size];

            // Find period using autocorrelation for this window
            let mut max_corr = 0.0;
            let mut best_period = 0;

            for period in min_period..=max_period.min(window_size / 2) {
                let mut sum = 0.0;
                let mut energy1 = 0.0;
                let mut energy2 = 0.0;

                for i in 0..(window_size - period) {
                    let s1 = window[i];
                    let s2 = window[i + period];
                    sum += s1 * s2;
                    energy1 += s1 * s1;
                    energy2 += s2 * s2;
                }

                let corr = if energy1 > 0.0 && energy2 > 0.0 {
                    sum / (energy1 * energy2).sqrt()
                } else {
                    0.0
                };

                if corr > max_corr {
                    max_corr = corr;
                    best_period = period;
                }
            }

            // Only include periods with high confidence
            if max_corr > 0.6 && best_period > 0 {
                periods.push(best_period as f32);
            }
        }

        // Need at least 3 periods to calculate jitter
        if periods.len() < 3 {
            return 0.0;
        }

        // Calculate jitter (local): mean absolute difference between consecutive periods
        let mut sum_diff = 0.0;
        for i in 1..periods.len() {
            sum_diff += (periods[i] - periods[i - 1]).abs();
        }

        let mean_diff = sum_diff / (periods.len() - 1) as f32;
        let mean_period: f32 = periods.iter().sum::<f32>() / periods.len() as f32;

        if mean_period > 0.0 {
            (mean_diff / mean_period) * 100.0 // Convert to percentage
        } else {
            0.0
        }
    }

    /// Calculate Shimmer (amplitude variation)
    ///
    /// Shimmer measures the cycle-to-cycle variation in amplitude, expressed
    /// as a percentage. It's an important indicator of voice quality and vocal
    /// fold irregularities. Higher shimmer indicates unstable voice production.
    ///
    /// This implements the Shimmer (local) metric, which is the average absolute
    /// difference between consecutive peak amplitudes, divided by the average amplitude.
    ///
    /// # Arguments
    ///
    /// * `min_freq` - Minimum expected pitch for period detection
    /// * `max_freq` - Maximum expected pitch for period detection
    ///
    /// # Returns
    ///
    /// Shimmer percentage (0-100), or 0.0 if calculation fails. Typical values:
    /// - < 3.0%: Normal voice quality
    /// - 3.0-6.0%: Mild amplitude variation
    /// - > 6.0%: Potential voice pathology
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 4096], 22050);
    /// let shimmer = buffer.calculate_shimmer(75.0, 500.0);
    /// println!("Shimmer: {:.2}%", shimmer);
    /// ```
    pub fn calculate_shimmer(&self, min_freq: f32, max_freq: f32) -> f32 {
        if self.samples.len() < 2048 {
            return 0.0;
        }

        let sample_rate = self.sample_rate as f32;
        let min_period = (sample_rate / max_freq) as usize;
        let max_period = (sample_rate / min_freq) as usize;

        // Find peak amplitudes for each period
        let window_size = (max_period * 4).min(self.samples.len());
        let mut peak_amplitudes = Vec::new();

        let hop_size = max_period / 2;
        for start_idx in (0..self.samples.len() - window_size).step_by(hop_size) {
            let window = &self.samples[start_idx..start_idx + window_size];

            // Find period using autocorrelation
            let mut max_corr = 0.0;
            let mut best_period = 0;

            for period in min_period..=max_period.min(window_size / 2) {
                let mut sum = 0.0;
                let mut energy1 = 0.0;
                let mut energy2 = 0.0;

                for i in 0..(window_size - period) {
                    let s1 = window[i];
                    let s2 = window[i + period];
                    sum += s1 * s2;
                    energy1 += s1 * s1;
                    energy2 += s2 * s2;
                }

                let corr = if energy1 > 0.0 && energy2 > 0.0 {
                    sum / (energy1 * energy2).sqrt()
                } else {
                    0.0
                };

                if corr > max_corr {
                    max_corr = corr;
                    best_period = period;
                }
            }

            // Find peak amplitude in this period
            if max_corr > 0.6 && best_period > 0 {
                let period_samples = &window[..best_period.min(window.len())];
                let peak = period_samples
                    .iter()
                    .map(|&s| s.abs())
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);
                peak_amplitudes.push(peak);
            }
        }

        // Need at least 3 peaks to calculate shimmer
        if peak_amplitudes.len() < 3 {
            return 0.0;
        }

        // Calculate shimmer: mean absolute difference between consecutive peaks
        let mut sum_diff = 0.0;
        for i in 1..peak_amplitudes.len() {
            sum_diff += (peak_amplitudes[i] - peak_amplitudes[i - 1]).abs();
        }

        let mean_diff = sum_diff / (peak_amplitudes.len() - 1) as f32;
        let mean_amplitude: f32 =
            peak_amplitudes.iter().sum::<f32>() / peak_amplitudes.len() as f32;

        if mean_amplitude > 0.0 {
            (mean_diff / mean_amplitude) * 100.0 // Convert to percentage
        } else {
            0.0
        }
    }

    /// Calculate Harmonic-to-Noise Ratio (HNR)
    ///
    /// HNR measures the ratio of harmonic (periodic) to noise (aperiodic) energy
    /// in the voice signal. It's a fundamental measure of voice quality.
    /// Higher HNR indicates clearer, more periodic voice production.
    ///
    /// This uses autocorrelation-based method to separate harmonic and noise components.
    ///
    /// # Arguments
    ///
    /// * `min_freq` - Minimum expected pitch
    /// * `max_freq` - Maximum expected pitch
    ///
    /// # Returns
    ///
    /// HNR in decibels (dB). Typical values:
    /// - > 20 dB: Excellent voice quality
    /// - 10-20 dB: Good voice quality
    /// - 5-10 dB: Fair voice quality
    /// - < 5 dB: Poor voice quality or pathology
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 4096], 22050);
    /// let hnr = buffer.calculate_hnr(75.0, 500.0);
    /// println!("HNR: {:.2} dB", hnr);
    /// ```
    pub fn calculate_hnr(&self, min_freq: f32, max_freq: f32) -> f32 {
        if self.samples.len() < 2048 {
            return 0.0;
        }

        let sample_rate = self.sample_rate as f32;
        let min_period = (sample_rate / max_freq) as usize;
        let max_period = (sample_rate / min_freq) as usize;

        // Use autocorrelation to find the fundamental period
        let window_size = (max_period * 3).min(self.samples.len());
        let samples = &self.samples[..window_size];

        let mut max_corr = 0.0;

        for period in min_period..=max_period.min(window_size / 2) {
            let mut sum = 0.0;
            let mut energy1 = 0.0;
            let mut energy2 = 0.0;

            for i in 0..(window_size - period) {
                let s1 = samples[i];
                let s2 = samples[i + period];
                sum += s1 * s2;
                energy1 += s1 * s1;
                energy2 += s2 * s2;
            }

            let corr = if energy1 > 0.0 && energy2 > 0.0 {
                sum / (energy1 * energy2).sqrt()
            } else {
                0.0
            };

            if corr > max_corr {
                max_corr = corr;
                // Note: We only need max_corr for HNR calculation, not the period itself
            }
        }

        // If no strong periodicity found, voice is mostly noise
        if max_corr < 0.3 {
            return 0.0;
        }

        // Calculate HNR using the autocorrelation peak
        // HNR = 10 * log10(max_corr / (1 - max_corr))
        // This formula relates autocorrelation to harmonic/noise ratio
        if max_corr >= 0.99 {
            // Avoid division by very small numbers
            return 30.0; // Very high HNR
        }

        let hnr_linear = max_corr / (1.0 - max_corr);
        10.0 * hnr_linear.log10()
    }

    /// Calculate Delta MFCCs (first-order temporal derivatives)
    ///
    /// Delta coefficients represent the rate of change of MFCCs over time,
    /// capturing dynamic spectral information. These are essential for
    /// improving speech recognition accuracy.
    ///
    /// # Arguments
    ///
    /// * `mfcc_frames` - Vector of MFCC coefficient vectors from consecutive frames
    /// * `delta_window` - Number of frames to use for delta calculation (typically 2)
    ///
    /// # Returns
    ///
    /// Vector of delta MFCC vectors, one per input frame.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 8192], 22050);
    /// // Extract MFCCs from multiple frames...
    /// let mfcc_frames = vec![
    ///     buffer.mfcc(13, 26, 512),
    ///     buffer.mfcc(13, 26, 512),
    ///     buffer.mfcc(13, 26, 512),
    /// ];
    /// let delta_mfccs = AudioBuffer::calculate_delta_mfcc(&mfcc_frames, 2);
    /// ```
    pub fn calculate_delta_mfcc(mfcc_frames: &[Vec<f32>], delta_window: usize) -> Vec<Vec<f32>> {
        if mfcc_frames.is_empty() {
            return Vec::new();
        }

        let num_frames = mfcc_frames.len();
        let num_coeffs = mfcc_frames[0].len();
        let mut delta_mfccs = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let mut delta_coeffs = vec![0.0; num_coeffs];

            // Calculate delta using regression formula
            // delta[t] = (sum(n * mfcc[t+n]) - sum(n * mfcc[t-n])) / (2 * sum(n²))
            let mut numerator = vec![0.0; num_coeffs];
            let mut denominator = 0.0;

            for n in 1..=delta_window {
                let n_f32 = n as f32;

                // Future frame (t+n)
                let future_idx = (frame_idx + n).min(num_frames - 1);
                // Past frame (t-n)
                let past_idx = frame_idx.saturating_sub(n);

                // Note: We need coeff_idx for indexing both numerator and mfcc_frames
                #[allow(clippy::needless_range_loop)]
                for coeff_idx in 0..num_coeffs {
                    numerator[coeff_idx] += n_f32
                        * (mfcc_frames[future_idx][coeff_idx] - mfcc_frames[past_idx][coeff_idx]);
                }

                denominator += n_f32 * n_f32;
            }

            // Normalize by denominator
            if denominator > 0.0 {
                // Note: We need coeff_idx for indexing both delta_coeffs and numerator
                #[allow(clippy::needless_range_loop)]
                for coeff_idx in 0..num_coeffs {
                    delta_coeffs[coeff_idx] = numerator[coeff_idx] / (2.0 * denominator);
                }
            }

            delta_mfccs.push(delta_coeffs);
        }

        delta_mfccs
    }

    /// Calculate Delta-Delta MFCCs (second-order temporal derivatives)
    ///
    /// Delta-Delta (acceleration) coefficients represent the rate of change of
    /// Delta coefficients, capturing the dynamics of spectral dynamics.
    /// Combined with MFCCs and Deltas, they form a powerful feature set for ASR.
    ///
    /// # Arguments
    ///
    /// * `delta_mfccs` - Vector of delta MFCC coefficient vectors
    /// * `delta_window` - Number of frames to use (typically 2)
    ///
    /// # Returns
    ///
    /// Vector of delta-delta MFCC vectors.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 8192], 22050);
    /// let mfcc_frames = vec![
    ///     buffer.mfcc(13, 26, 512),
    ///     buffer.mfcc(13, 26, 512),
    ///     buffer.mfcc(13, 26, 512),
    /// ];
    /// let delta_mfccs = AudioBuffer::calculate_delta_mfcc(&mfcc_frames, 2);
    /// let delta_delta_mfccs = AudioBuffer::calculate_delta_delta_mfcc(&delta_mfccs, 2);
    /// ```
    pub fn calculate_delta_delta_mfcc(
        delta_mfccs: &[Vec<f32>],
        delta_window: usize,
    ) -> Vec<Vec<f32>> {
        // Delta-delta is just the delta of delta coefficients
        Self::calculate_delta_mfcc(delta_mfccs, delta_window)
    }

    /// Calculate Chroma features (pitch class representation)
    ///
    /// Chroma features, also known as pitch class profiles or chromagrams, represent
    /// the intensity of the 12 pitch classes (C, C#, D, ..., B) regardless of octave.
    /// This is particularly useful for music information retrieval, chord recognition,
    /// and key detection.
    ///
    /// The algorithm maps each frequency bin to one of 12 pitch classes using:
    /// pitch_class = 12 * log2(freq / ref_freq) mod 12
    ///
    /// # Arguments
    ///
    /// * `fft_size` - FFT size (must be power of 2, typically 2048 or 4096 for music)
    /// * `ref_freq` - Reference frequency for A4 (default: 440.0 Hz)
    ///
    /// # Returns
    ///
    /// 12-element vector representing energy in each pitch class (C=0, C#=1, ..., B=11).
    /// Returns empty vector if insufficient samples.
    ///
    /// # Applications
    ///
    /// - Music information retrieval
    /// - Chord recognition and key detection
    /// - Cover song identification
    /// - Music similarity analysis
    /// - Tonal music analysis
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 8192], 22050);
    /// let chroma = buffer.chroma_features(2048, 440.0);
    /// assert_eq!(chroma.len(), 12);
    /// ```
    pub fn chroma_features(&self, fft_size: usize, ref_freq: f32) -> Vec<f32> {
        if self.samples.len() < fft_size || !fft_size.is_power_of_two() {
            return Vec::new();
        }

        // Get magnitude spectrum
        let magnitudes = self.get_magnitude_spectrum(fft_size);

        // Initialize 12 pitch class bins
        let mut chroma = vec![0.0_f32; 12];

        let sample_rate = self.sample_rate as f32;
        let freq_resolution = sample_rate / fft_size as f32;

        // Map each frequency bin to a pitch class
        for (bin, &magnitude) in magnitudes.iter().enumerate().skip(1) {
            let freq = bin as f32 * freq_resolution;

            // Skip very low frequencies (below 20 Hz)
            if freq < 20.0 {
                continue;
            }

            // Calculate pitch class: 12 * log2(freq / ref_freq) mod 12
            let pitch_class_float = 12.0 * (freq / ref_freq).log2();
            let pitch_class = pitch_class_float.rem_euclid(12.0) as usize % 12;

            // Accumulate magnitude in corresponding pitch class
            chroma[pitch_class] += magnitude;
        }

        // Normalize chroma vector
        let max_chroma = chroma.iter().cloned().fold(0.0_f32, f32::max);
        if max_chroma > 0.0 {
            for c in &mut chroma {
                *c /= max_chroma;
            }
        }

        chroma
    }

    /// Calculate Spectral Contrast
    ///
    /// Spectral contrast measures the difference between peaks and valleys in the
    /// spectrum across multiple frequency bands. This provides a robust timbre
    /// representation, particularly useful for music genre classification and
    /// audio texture analysis.
    ///
    /// The spectrum is divided into sub-bands, and for each band:
    /// - Peak: Mean of top 20% of magnitudes
    /// - Valley: Mean of bottom 20% of magnitudes
    /// - Contrast: Peak - Valley (in dB)
    ///
    /// # Arguments
    ///
    /// * `fft_size` - FFT size (must be power of 2, typically 2048)
    /// * `num_bands` - Number of frequency bands (typically 6-8)
    ///
    /// # Returns
    ///
    /// Vector of contrast values (in dB) for each frequency band.
    /// Returns empty vector if insufficient samples.
    ///
    /// # Applications
    ///
    /// - Music genre classification
    /// - Audio texture analysis
    /// - Instrument recognition
    /// - Timbre characterization
    /// - Sound quality assessment
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 8192], 22050);
    /// let contrast = buffer.spectral_contrast(2048, 6);
    /// assert_eq!(contrast.len(), 6);
    /// ```
    pub fn spectral_contrast(&self, fft_size: usize, num_bands: usize) -> Vec<f32> {
        if self.samples.len() < fft_size || !fft_size.is_power_of_two() || num_bands == 0 {
            return Vec::new();
        }

        // Get magnitude spectrum
        let magnitudes = self.get_magnitude_spectrum(fft_size);

        // Divide spectrum into logarithmically-spaced frequency bands
        let mut contrasts = Vec::with_capacity(num_bands);

        let num_bins = magnitudes.len();
        let band_edges = (0..=num_bands)
            .map(|i| {
                let ratio = i as f32 / num_bands as f32;
                // Logarithmic spacing
                (ratio * (num_bins as f32).ln()).exp() as usize
            })
            .collect::<Vec<_>>();

        // Calculate contrast for each band
        for band_idx in 0..num_bands {
            let start = band_edges[band_idx];
            let end = band_edges[band_idx + 1].min(num_bins);

            if start >= end {
                contrasts.push(0.0);
                continue;
            }

            // Extract band magnitudes
            let mut band_mags: Vec<f32> = magnitudes[start..end].to_vec();
            band_mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let band_len = band_mags.len();
            if band_len < 5 {
                contrasts.push(0.0);
                continue;
            }

            // Calculate peak (top 20%) and valley (bottom 20%)
            let percentile_idx = (band_len as f32 * 0.2) as usize;

            // Valley: mean of bottom 20%
            let valley_sum: f32 = band_mags[..percentile_idx].iter().sum();
            let valley = valley_sum / percentile_idx as f32;

            // Peak: mean of top 20%
            let peak_sum: f32 = band_mags[band_len - percentile_idx..].iter().sum();
            let peak = peak_sum / percentile_idx as f32;

            // Calculate contrast in dB
            let contrast = if valley > 1e-10 {
                20.0 * (peak / valley).log10()
            } else if peak > 1e-10 {
                60.0 // Maximum contrast when valley is near zero
            } else {
                0.0
            };

            contrasts.push(contrast);
        }

        contrasts
    }

    /// Detect pitch using the YIN algorithm
    ///
    /// The YIN algorithm is a robust pitch detection method that improves upon
    /// autocorrelation by using a cumulative mean normalized difference function.
    /// It provides more accurate pitch detection, especially for noisy signals.
    ///
    /// YIN steps:
    /// 1. Calculate difference function
    /// 2. Apply cumulative mean normalization
    /// 3. Find first minimum below threshold
    /// 4. Apply parabolic interpolation for sub-sample accuracy
    ///
    /// Reference: "YIN, a fundamental frequency estimator for speech and music"
    /// by Alain de Cheveigné and Hideki Kawahara (2002)
    ///
    /// # Arguments
    ///
    /// * `min_freq` - Minimum frequency to search (Hz)
    /// * `max_freq` - Maximum frequency to search (Hz)
    /// * `threshold` - Threshold for minimum detection (typically 0.1-0.2)
    ///
    /// # Returns
    ///
    /// Detected fundamental frequency in Hz, or 0.0 if no pitch detected.
    ///
    /// # Applications
    ///
    /// - High-accuracy pitch tracking
    /// - Music transcription
    /// - Singing voice analysis
    /// - Instrument tuning
    /// - Prosody analysis in speech
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::audio::AudioBuffer;
    ///
    /// let buffer = AudioBuffer::mono(vec![0.5; 8192], 22050);
    /// let pitch = buffer.detect_pitch_yin(80.0, 400.0, 0.15);
    /// if pitch > 0.0 {
    ///     println!("Detected pitch: {:.1} Hz", pitch);
    /// }
    /// ```
    pub fn detect_pitch_yin(&self, min_freq: f32, max_freq: f32, threshold: f32) -> f32 {
        let sample_rate = self.sample_rate as f32;
        let min_period = (sample_rate / max_freq) as usize;
        let max_period = (sample_rate / min_freq) as usize;

        // Need sufficient samples
        let buffer_size = (max_period * 2).min(self.samples.len());
        if buffer_size < max_period {
            return 0.0;
        }

        // Step 1: Calculate difference function
        let mut diff = vec![0.0_f32; max_period + 1];
        diff[0] = 1.0; // By definition

        // Note: tau is used for both arithmetic (buffer_size - tau) and indexing (diff[tau], samples[i + tau])
        #[allow(clippy::needless_range_loop)]
        for tau in 1..=max_period {
            let mut sum = 0.0;
            for i in 0..(buffer_size - tau) {
                let delta = self.samples[i] - self.samples[i + tau];
                sum += delta * delta;
            }
            diff[tau] = sum;
        }

        // Step 2: Cumulative mean normalized difference function
        let mut cmnd = vec![0.0_f32; max_period + 1];
        cmnd[0] = 1.0;

        let mut running_sum = 0.0;
        for tau in 1..=max_period {
            running_sum += diff[tau];
            cmnd[tau] = if running_sum > 0.0 {
                diff[tau] * tau as f32 / running_sum
            } else {
                1.0
            };
        }

        // Step 3: Find first minimum below threshold
        let mut tau_estimate = 0;
        for tau in min_period..=max_period {
            if cmnd[tau] < threshold {
                // Look for local minimum
                if tau > 0
                    && tau < max_period
                    && cmnd[tau] < cmnd[tau - 1]
                    && cmnd[tau] < cmnd[tau + 1]
                {
                    tau_estimate = tau;
                    break;
                }
            }
        }

        if tau_estimate == 0 {
            // No pitch found below threshold
            return 0.0;
        }

        // Step 4: Parabolic interpolation for sub-sample accuracy
        let tau = tau_estimate;
        let refined_tau = if tau > 0 && tau < max_period {
            let alpha = cmnd[tau - 1];
            let beta = cmnd[tau];
            let gamma = cmnd[tau + 1];

            // Parabolic interpolation
            let adjustment = if (alpha - 2.0 * beta + gamma).abs() > 1e-10 {
                0.5 * (alpha - gamma) / (alpha - 2.0 * beta + gamma)
            } else {
                0.0
            };

            tau as f32 + adjustment
        } else {
            tau as f32
        };

        // Convert period to frequency
        sample_rate / refined_tau
    }
}

#[cfg(test)]
#[path = "utilities_tests.rs"]
mod tests;
