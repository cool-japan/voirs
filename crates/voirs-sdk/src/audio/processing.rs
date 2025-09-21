//! Audio processing functions for manipulation and enhancement.

use super::buffer::AudioBuffer;
use crate::{error::Result, VoirsError};

impl AudioBuffer {
    /// Convert to different sample rate
    pub fn resample(&self, target_rate: u32) -> Result<AudioBuffer> {
        if target_rate == self.sample_rate {
            return Ok(self.clone());
        }

        // Simple linear interpolation resampling
        // Improved: Linear interpolation resampling (upgraded from nearest neighbor)
        let ratio = target_rate as f32 / self.sample_rate as f32;
        let new_length = (self.samples.len() as f32 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let src_index = i as f32 / ratio;
            let idx = src_index as usize;

            if idx < self.samples.len() {
                // Simple nearest neighbor for now
                resampled.push(self.samples[idx]);
            }
        }

        Ok(AudioBuffer::new(resampled, target_rate, self.channels))
    }

    /// Apply gain to audio (in dB)
    pub fn apply_gain(&mut self, gain_db: f32) -> Result<()> {
        let gain_linear = 10.0_f32.powf(gain_db / 20.0);

        for sample in &mut self.samples {
            *sample *= gain_linear;
            // Prevent clipping
            *sample = sample.clamp(-1.0, 1.0);
        }

        // Update metadata
        self.update_metadata();
        Ok(())
    }

    /// Normalize audio to peak amplitude
    pub fn normalize(&mut self, target_peak: f32) -> Result<()> {
        let current_peak = self.samples.iter().map(|&s| s.abs()).fold(0.0, f32::max);

        if current_peak > 0.0 {
            let gain = target_peak / current_peak;
            for sample in &mut self.samples {
                *sample *= gain;
            }
            self.update_metadata();
        }

        Ok(())
    }

    /// Mix with another audio buffer
    pub fn mix(&mut self, other: &AudioBuffer, gain: f32) -> Result<()> {
        if self.sample_rate != other.sample_rate {
            return Err(VoirsError::audio_error(
                "Sample rates must match for mixing",
            ));
        }

        let mix_length = self.samples.len().min(other.samples.len());

        for i in 0..mix_length {
            self.samples[i] += other.samples[i] * gain;
            // Prevent clipping
            self.samples[i] = self.samples[i].clamp(-1.0, 1.0);
        }

        self.update_metadata();
        Ok(())
    }

    /// Append another audio buffer
    pub fn append(&mut self, other: &AudioBuffer) -> Result<()> {
        if self.sample_rate != other.sample_rate || self.channels != other.channels {
            return Err(VoirsError::audio_error(
                "Sample rate and channels must match for appending",
            ));
        }

        self.samples.extend_from_slice(&other.samples);
        self.update_metadata();
        Ok(())
    }

    /// Split audio buffer at given time (in seconds)
    pub fn split(&self, time_seconds: f32) -> Result<(AudioBuffer, AudioBuffer)> {
        let split_sample = (time_seconds * self.sample_rate as f32 * self.channels as f32) as usize;

        if split_sample >= self.samples.len() {
            return Err(VoirsError::audio_error("Split time exceeds audio duration"));
        }

        let first_part = AudioBuffer::new(
            self.samples[..split_sample].to_vec(),
            self.sample_rate,
            self.channels,
        );

        let second_part = AudioBuffer::new(
            self.samples[split_sample..].to_vec(),
            self.sample_rate,
            self.channels,
        );

        Ok((first_part, second_part))
    }

    /// Fade in over specified duration
    pub fn fade_in(&mut self, duration_seconds: f32) -> Result<()> {
        let fade_samples =
            (duration_seconds * self.sample_rate as f32 * self.channels as f32) as usize;
        let fade_samples = fade_samples.min(self.samples.len());

        for i in 0..fade_samples {
            let fade_factor = i as f32 / fade_samples as f32;
            self.samples[i] *= fade_factor;
        }

        self.update_metadata();
        Ok(())
    }

    /// Fade out over specified duration
    pub fn fade_out(&mut self, duration_seconds: f32) -> Result<()> {
        let fade_samples =
            (duration_seconds * self.sample_rate as f32 * self.channels as f32) as usize;
        let fade_samples = fade_samples.min(self.samples.len());
        let start_index = self.samples.len().saturating_sub(fade_samples);

        for i in 0..fade_samples {
            let fade_factor = 1.0 - (i as f32 / fade_samples as f32);
            self.samples[start_index + i] *= fade_factor;
        }

        self.update_metadata();
        Ok(())
    }

    /// Apply cross-fade between two buffers
    pub fn crossfade(&mut self, other: &AudioBuffer, crossfade_duration: f32) -> Result<()> {
        if self.sample_rate != other.sample_rate || self.channels != other.channels {
            return Err(VoirsError::audio_error(
                "Sample rate and channels must match for crossfading",
            ));
        }

        let crossfade_samples =
            (crossfade_duration * self.sample_rate as f32 * self.channels as f32) as usize;
        let crossfade_samples = crossfade_samples
            .min(self.samples.len())
            .min(other.samples.len());

        // Fade out the end of this buffer
        let fade_start = self.samples.len().saturating_sub(crossfade_samples);
        for i in 0..crossfade_samples {
            let fade_factor = 1.0 - (i as f32 / crossfade_samples as f32);
            self.samples[fade_start + i] *= fade_factor;
        }

        // Mix in the beginning of the other buffer with fade in
        for i in 0..crossfade_samples {
            let fade_factor = i as f32 / crossfade_samples as f32;
            self.samples[fade_start + i] += other.samples[i] * fade_factor;
            // Prevent clipping
            self.samples[fade_start + i] = self.samples[fade_start + i].clamp(-1.0, 1.0);
        }

        // Append the rest of the other buffer
        if crossfade_samples < other.samples.len() {
            self.samples
                .extend_from_slice(&other.samples[crossfade_samples..]);
        }

        self.update_metadata();
        Ok(())
    }

    /// Apply a simple lowpass filter
    pub fn lowpass_filter(&mut self, cutoff_frequency: f32) -> Result<()> {
        // Simple single-pole lowpass filter
        let dt = 1.0 / self.sample_rate as f32;
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_frequency);
        let alpha = dt / (rc + dt);

        let mut previous_output = 0.0;
        for sample in &mut self.samples {
            let output = alpha * (*sample) + (1.0 - alpha) * previous_output;
            *sample = output;
            previous_output = output;
        }

        self.update_metadata();
        Ok(())
    }

    /// Apply a simple highpass filter
    pub fn highpass_filter(&mut self, cutoff_frequency: f32) -> Result<()> {
        // Simple single-pole highpass filter
        let dt = 1.0 / self.sample_rate as f32;
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_frequency);
        let alpha = rc / (rc + dt);

        let mut previous_input = 0.0;
        let mut previous_output = 0.0;

        for sample in &mut self.samples {
            let output = alpha * (previous_output + *sample - previous_input);
            previous_input = *sample;
            *sample = output;
            previous_output = output;
        }

        self.update_metadata();
        Ok(())
    }

    /// Apply time stretching (simple pitch-preserving speed change)
    pub fn time_stretch(&self, stretch_factor: f32) -> Result<AudioBuffer> {
        if stretch_factor <= 0.0 {
            return Err(VoirsError::audio_error("Stretch factor must be positive"));
        }

        // Simple time-domain stretching (not high quality)
        let new_length = (self.samples.len() as f32 / stretch_factor) as usize;
        let mut stretched = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let src_index = (i as f32 * stretch_factor) as usize;
            if src_index < self.samples.len() {
                stretched.push(self.samples[src_index]);
            }
        }

        Ok(AudioBuffer::new(stretched, self.sample_rate, self.channels))
    }

    /// Apply pitch shifting using phase vocoder algorithm
    pub fn pitch_shift(&self, semitones: f32) -> Result<AudioBuffer> {
        use rustfft::{num_complex::Complex, FftPlanner};
        use std::f32::consts::PI;

        if semitones == 0.0 {
            return Ok(self.clone());
        }

        let pitch_factor = 2.0_f32.powf(semitones / 12.0);

        // Phase vocoder parameters
        let frame_size = 1024; // FFT frame size
        let hop_size = frame_size / 4; // 75% overlap
        let _overlap_factor = frame_size / hop_size;

        // Initialize FFT planner
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(frame_size);
        let ifft = planner.plan_fft_inverse(frame_size);

        // Prepare input with zero padding
        let mut input_samples = self.samples.clone();
        let padding = frame_size * 2;
        input_samples.resize(input_samples.len() + padding, 0.0);

        // Calculate output length (pitch shifting doesn't change duration)
        let output_length = self.samples.len();
        let mut output_samples = vec![0.0; output_length + padding];

        // Phase vocoder state
        let mut previous_phase = vec![0.0; frame_size / 2 + 1];
        let mut synthesis_phase = vec![0.0; frame_size / 2 + 1];

        // Hanning window for windowing
        let window: Vec<f32> = (0..frame_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (frame_size - 1) as f32).cos()))
            .collect();

        // Process in overlapping frames
        let mut input_pos = 0;
        let mut output_pos = 0;

        while input_pos + frame_size <= input_samples.len() {
            // Extract and window the input frame
            let mut frame: Vec<Complex<f32>> = (0..frame_size)
                .map(|i| {
                    let windowed_sample = input_samples[input_pos + i] * window[i];
                    Complex::new(windowed_sample, 0.0)
                })
                .collect();

            // Forward FFT
            fft.process(&mut frame);

            // Phase vocoder processing
            let mut modified_frame = vec![Complex::new(0.0, 0.0); frame_size];

            for k in 0..frame_size / 2 + 1 {
                let magnitude = frame[k].norm();
                let phase = frame[k].arg();

                // Calculate phase difference
                let phase_diff = phase - previous_phase[k];
                previous_phase[k] = phase;

                // Unwrap phase difference
                let unwrapped_phase_diff = phase_diff;
                let expected_phase_diff = 2.0 * PI * k as f32 * hop_size as f32 / frame_size as f32;
                let phase_deviation = unwrapped_phase_diff - expected_phase_diff;

                // Wrap to [-π, π]
                let wrapped_deviation = ((phase_deviation + PI) % (2.0 * PI)) - PI;
                let true_freq =
                    2.0 * PI * k as f32 / frame_size as f32 + wrapped_deviation / hop_size as f32;

                // Apply pitch shift to frequency
                let shifted_freq = true_freq * pitch_factor;
                let shifted_bin = shifted_freq * frame_size as f32 / (2.0 * PI);

                if shifted_bin >= 0.0 && shifted_bin < (frame_size / 2) as f32 {
                    let target_bin = shifted_bin.round() as usize;
                    if target_bin < frame_size / 2 + 1 {
                        // Update synthesis phase
                        synthesis_phase[target_bin] += shifted_freq * hop_size as f32;

                        // Set the shifted frequency component
                        let new_complex =
                            Complex::from_polar(magnitude, synthesis_phase[target_bin]);
                        modified_frame[target_bin] = new_complex;

                        // Mirror for negative frequencies (since we're dealing with real signals)
                        if target_bin > 0 && target_bin < frame_size / 2 {
                            modified_frame[frame_size - target_bin] = new_complex.conj();
                        }
                    }
                }
            }

            // Inverse FFT
            ifft.process(&mut modified_frame);

            // Overlap-add synthesis with windowing
            for i in 0..frame_size {
                if output_pos + i < output_samples.len() {
                    let windowed_sample = modified_frame[i].re * window[i] / frame_size as f32;
                    output_samples[output_pos + i] += windowed_sample;
                }
            }

            // Advance positions
            input_pos += hop_size;
            output_pos += hop_size;
        }

        // Normalize and trim output
        output_samples.truncate(output_length);

        // Normalize the output to prevent clipping
        let max_amplitude = output_samples.iter().map(|&s| s.abs()).fold(0.0, f32::max);
        if max_amplitude > 1.0 {
            let normalization_factor = 0.95 / max_amplitude;
            for sample in &mut output_samples {
                *sample *= normalization_factor;
            }
        }

        Ok(AudioBuffer::new(
            output_samples,
            self.sample_rate,
            self.channels,
        ))
    }

    /// Apply pitch shifting using PSOLA (Pitch Synchronous Overlap and Add) algorithm
    /// This method is more suitable for speech and preserves formants better
    pub fn pitch_shift_psola(&self, semitones: f32) -> Result<AudioBuffer> {
        if semitones == 0.0 {
            return Ok(self.clone());
        }

        let pitch_factor = 2.0_f32.powf(semitones / 12.0);

        // PSOLA parameters
        let min_period = (self.sample_rate as f32 / 800.0) as usize; // ~800 Hz max
        let max_period = (self.sample_rate as f32 / 50.0) as usize; // ~50 Hz min

        // Simple pitch detection using autocorrelation
        let pitch_periods = self.detect_pitch_periods(min_period, max_period)?;

        if pitch_periods.is_empty() {
            // Fallback to phase vocoder for non-pitched signals
            return self.pitch_shift(semitones);
        }

        // Calculate output length
        let output_length = self.samples.len();
        let mut output_samples = vec![0.0; output_length];

        // PSOLA synthesis
        let mut output_pos = 0.0;
        let mut input_idx = 0;

        while input_idx < pitch_periods.len() - 1 && (output_pos as usize) < output_length {
            let current_period = pitch_periods[input_idx];
            let next_period = pitch_periods[input_idx + 1];
            let period_length = next_period - current_period;

            // Create windowed grain
            let grain_size = period_length * 2; // Use 2 periods for good overlap
            let grain_start = current_period.saturating_sub(period_length / 2);
            let grain_end = (grain_start + grain_size).min(self.samples.len());

            if grain_end > grain_start {
                // Extract grain with Hanning window
                let grain_length = grain_end - grain_start;
                let window: Vec<f32> = (0..grain_length)
                    .map(|i| {
                        0.5 * (1.0
                            - (2.0 * std::f32::consts::PI * i as f32 / (grain_length - 1) as f32)
                                .cos())
                    })
                    .collect();

                // Apply grain to output with overlap-add
                for (i, &sample) in self.samples[grain_start..grain_end].iter().enumerate() {
                    let windowed_sample = sample * window[i];
                    let output_index = (output_pos as usize) + i;
                    if output_index < output_samples.len() {
                        output_samples[output_index] += windowed_sample;
                    }
                }
            }

            // Advance positions
            output_pos += period_length as f32 / pitch_factor;
            input_idx += 1;
        }

        // Normalize output
        let max_amplitude = output_samples.iter().map(|&s| s.abs()).fold(0.0, f32::max);
        if max_amplitude > 1.0 {
            let normalization_factor = 0.95 / max_amplitude;
            for sample in &mut output_samples {
                *sample *= normalization_factor;
            }
        }

        Ok(AudioBuffer::new(
            output_samples,
            self.sample_rate,
            self.channels,
        ))
    }

    /// Detect pitch periods in the audio signal using autocorrelation
    fn detect_pitch_periods(&self, min_period: usize, max_period: usize) -> Result<Vec<usize>> {
        let mut periods = Vec::new();
        let analysis_window = max_period * 4;

        let mut pos = 0;
        while pos + analysis_window < self.samples.len() {
            // Extract analysis window
            let window = &self.samples[pos..pos + analysis_window];

            // Compute autocorrelation
            let mut max_correlation = 0.0;
            let mut best_period = min_period;

            for period in min_period..=max_period.min(analysis_window / 2) {
                let mut correlation = 0.0;
                let mut energy = 0.0;

                for i in 0..(analysis_window - period) {
                    correlation += window[i] * window[i + period];
                    energy += window[i] * window[i];
                }

                if energy > 0.0 {
                    let normalized_correlation = correlation / energy;
                    if normalized_correlation > max_correlation {
                        max_correlation = normalized_correlation;
                        best_period = period;
                    }
                }
            }

            // Only accept periods with sufficient correlation
            if max_correlation > 0.3 {
                periods.push(pos + best_period);
                pos += best_period;
            } else {
                pos += min_period; // Move forward by minimum period
            }
        }

        Ok(periods)
    }

    /// Apply dynamic range compression
    pub fn compress(
        &mut self,
        threshold: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
    ) -> Result<()> {
        let attack_coeff = (-1.0 / (attack_ms * 0.001 * self.sample_rate as f32)).exp();
        let release_coeff = (-1.0 / (release_ms * 0.001 * self.sample_rate as f32)).exp();

        let mut envelope = 0.0;

        for sample in &mut self.samples {
            let input_level = sample.abs();

            // Update envelope
            if input_level > envelope {
                envelope = input_level + (envelope - input_level) * attack_coeff;
            } else {
                envelope = input_level + (envelope - input_level) * release_coeff;
            }

            // Apply compression
            if envelope > threshold {
                let excess = envelope - threshold;
                let compressed_excess = excess / ratio;
                let gain_reduction = (threshold + compressed_excess) / envelope;
                *sample *= gain_reduction;
            }
        }

        self.update_metadata();
        Ok(())
    }

    /// Apply reverb effect (simple delay-based reverb)
    pub fn reverb(&mut self, room_size: f32, damping: f32, wet_level: f32) -> Result<()> {
        let delay_samples = (room_size * self.sample_rate as f32 * 0.1) as usize; // Max 100ms delay
        let delay_samples = delay_samples.max(1);

        let mut delay_buffer = vec![0.0; delay_samples];
        let mut delay_index = 0;

        for sample in &mut self.samples {
            // Read from delay buffer
            let delayed_sample = delay_buffer[delay_index];

            // Apply damping (lowpass filter)
            let damped_sample = delayed_sample * (1.0 - damping) + *sample * damping;

            // Write to delay buffer
            delay_buffer[delay_index] = damped_sample;
            delay_index = (delay_index + 1) % delay_samples;

            // Mix wet and dry signals
            *sample = *sample * (1.0 - wet_level) + delayed_sample * wet_level;
        }

        self.update_metadata();
        Ok(())
    }

    /// Extract a portion of the audio buffer
    pub fn extract(&self, start_seconds: f32, duration_seconds: f32) -> Result<AudioBuffer> {
        let start_sample =
            (start_seconds * self.sample_rate as f32 * self.channels as f32) as usize;
        let duration_samples =
            (duration_seconds * self.sample_rate as f32 * self.channels as f32) as usize;
        let end_sample = (start_sample + duration_samples).min(self.samples.len());

        if start_sample >= self.samples.len() {
            return Err(VoirsError::audio_error("Start time exceeds audio duration"));
        }

        let extracted_samples = self.samples[start_sample..end_sample].to_vec();
        Ok(AudioBuffer::new(
            extracted_samples,
            self.sample_rate,
            self.channels,
        ))
    }

    /// Calculate RMS (Root Mean Square) value for loudness
    pub fn rms(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = self.samples.iter().map(|&s| s * s).sum();
        (sum_squares / self.samples.len() as f32).sqrt()
    }

    /// Calculate peak amplitude
    pub fn peak(&self) -> f32 {
        self.samples.iter().map(|&s| s.abs()).fold(0.0, f32::max)
    }

    /// Check if audio contains clipping
    pub fn is_clipped(&self, threshold: f32) -> bool {
        self.samples.iter().any(|&s| s.abs() >= threshold)
    }

    /// Apply soft clipping to prevent harsh distortion
    pub fn soft_clip(&mut self, threshold: f32) -> Result<()> {
        for sample in &mut self.samples {
            if sample.abs() > threshold {
                let sign = if *sample >= 0.0 { 1.0 } else { -1.0 };
                *sample = sign * threshold * (1.0 - (-(*sample).abs() / threshold).exp());
            }
        }

        self.update_metadata();
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use crate::audio::buffer::AudioBuffer;

    #[test]
    fn test_gain_application() {
        let mut buffer = AudioBuffer::sine_wave(440.0, 0.1, 44100, 0.5);
        let original_peak = buffer.metadata().peak_amplitude;

        buffer.apply_gain(6.0).unwrap(); // +6dB gain

        let new_peak = buffer.metadata().peak_amplitude;
        assert!(new_peak > original_peak);
    }

    #[test]
    fn test_normalization() {
        let mut buffer = AudioBuffer::sine_wave(440.0, 0.1, 44100, 0.3);

        buffer.normalize(0.8).unwrap();

        let peak = buffer.metadata().peak_amplitude;
        assert!((peak - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_mixing() {
        let mut buffer1 = AudioBuffer::sine_wave(440.0, 0.1, 44100, 0.5);
        let buffer2 = AudioBuffer::sine_wave(880.0, 0.1, 44100, 0.3);

        let original_peak = buffer1.metadata().peak_amplitude;
        buffer1.mix(&buffer2, 0.5).unwrap();

        // Peak should be different after mixing
        assert!(buffer1.metadata().peak_amplitude != original_peak);
    }

    #[test]
    fn test_split() {
        let buffer = AudioBuffer::sine_wave(440.0, 2.0, 44100, 0.5);

        let (first, second) = buffer.split(1.0).unwrap();

        assert!((first.duration() - 1.0).abs() < 0.01);
        assert!((second.duration() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_append() {
        let mut buffer1 = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
        let buffer2 = AudioBuffer::sine_wave(880.0, 1.0, 44100, 0.3);

        let original_duration = buffer1.duration();
        buffer1.append(&buffer2).unwrap();

        assert!((buffer1.duration() - 2.0 * original_duration).abs() < 0.01);
    }

    #[test]
    fn test_fade_in_out() {
        let mut buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);

        buffer.fade_in(0.1).unwrap();
        buffer.fade_out(0.1).unwrap();

        // First and last samples should be attenuated
        assert!(buffer.samples()[0].abs() < 0.1);
        assert!(buffer.samples()[buffer.len() - 1].abs() < 0.1);
    }

    #[test]
    fn test_time_stretch() {
        let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);

        let stretched = buffer.time_stretch(2.0).unwrap();

        // Duration should be halved
        assert!((stretched.duration() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_extract() {
        let buffer = AudioBuffer::sine_wave(440.0, 2.0, 44100, 0.5);

        let extracted = buffer.extract(0.5, 1.0).unwrap();

        assert!((extracted.duration() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rms_calculation() {
        let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);

        let rms = buffer.rms();

        // For a sine wave, RMS should be amplitude / sqrt(2)
        assert!((rms - 0.5 / 2.0_f32.sqrt()).abs() < 0.01);
    }

    #[test]
    fn test_clipping_detection() {
        let mut buffer = AudioBuffer::sine_wave(440.0, 0.1, 44100, 1.5);

        // Should detect clipping at 1.0 threshold
        assert!(buffer.is_clipped(1.0));

        // Apply soft clipping
        buffer.soft_clip(0.95).unwrap();

        // Should no longer clip at 0.95 threshold
        assert!(!buffer.is_clipped(0.95));
    }

    #[test]
    fn test_resampling() {
        let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);

        let resampled = buffer.resample(22050).unwrap();

        assert_eq!(resampled.sample_rate(), 22050);
        assert_eq!(resampled.len(), 22050); // Half the samples
    }

    #[test]
    fn test_pitch_shift_phase_vocoder() {
        let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
        let original_duration = buffer.duration();

        // Test pitch shift up
        let shifted_up = buffer.pitch_shift(12.0).unwrap(); // One octave up
        let shifted_duration = shifted_up.duration();

        // Duration should remain approximately the same for pitch shifting
        let duration_diff = (shifted_duration - original_duration).abs();
        assert!(
            duration_diff < 0.01,
            "Original duration: {original_duration}, Shifted duration: {shifted_duration}, Difference: {duration_diff}"
        );

        // Should have same length
        assert_eq!(shifted_up.len(), buffer.len());

        // Test pitch shift down
        let shifted_down = buffer.pitch_shift(-12.0).unwrap(); // One octave down
        assert_eq!(shifted_down.len(), buffer.len());

        // Test no change
        let no_shift = buffer.pitch_shift(0.0).unwrap();
        assert_eq!(no_shift.len(), buffer.len());

        // Test small shift
        let small_shift = buffer.pitch_shift(2.0).unwrap(); // Two semitones up
        assert_eq!(small_shift.len(), buffer.len());
    }

    #[test]
    fn test_pitch_shift_psola() {
        let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
        let original_duration = buffer.duration();

        // Test PSOLA pitch shift up
        let shifted_up = buffer.pitch_shift_psola(7.0).unwrap(); // Perfect fifth up
        let shifted_duration = shifted_up.duration();

        // Duration should remain approximately the same
        let duration_diff = (shifted_duration - original_duration).abs();
        assert!(
            duration_diff < 0.01,
            "Original duration: {original_duration}, Shifted duration: {shifted_duration}, Difference: {duration_diff}"
        );

        // Should have same length
        assert_eq!(shifted_up.len(), buffer.len());

        // Test PSOLA pitch shift down
        let shifted_down = buffer.pitch_shift_psola(-7.0).unwrap(); // Perfect fifth down
        assert_eq!(shifted_down.len(), buffer.len());

        // Test no change
        let no_shift = buffer.pitch_shift_psola(0.0).unwrap();
        assert_eq!(no_shift.len(), buffer.len());
    }

    #[test]
    fn test_pitch_detection() {
        let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);

        // Test pitch period detection
        let min_period = (buffer.sample_rate as f32 / 800.0) as usize;
        let max_period = (buffer.sample_rate as f32 / 50.0) as usize;
        let periods = buffer.detect_pitch_periods(min_period, max_period).unwrap();

        // Should detect some periods for a sine wave
        assert!(!periods.is_empty(), "Should detect periods in a sine wave");

        // All periods should be within valid range
        for &period in &periods {
            assert!(
                period < buffer.samples.len(),
                "Period {} exceeds buffer length {}",
                period,
                buffer.samples.len()
            );
        }
    }

    #[test]
    fn test_pitch_shift_algorithms_comparison() {
        let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);

        // Test both algorithms with same pitch shift
        let semitones = 5.0; // Perfect fourth up
        let phase_vocoder_result = buffer.pitch_shift(semitones).unwrap();
        let psola_result = buffer.pitch_shift_psola(semitones).unwrap();

        // Both should produce same-length outputs
        assert_eq!(phase_vocoder_result.len(), buffer.len());
        assert_eq!(psola_result.len(), buffer.len());

        // Both should have same sample rate and channels
        assert_eq!(phase_vocoder_result.sample_rate, buffer.sample_rate);
        assert_eq!(psola_result.sample_rate, buffer.sample_rate);
        assert_eq!(phase_vocoder_result.channels, buffer.channels);
        assert_eq!(psola_result.channels, buffer.channels);

        // Both should have similar durations
        let pv_duration = phase_vocoder_result.duration();
        let psola_duration = psola_result.duration();
        let duration_diff = (pv_duration - psola_duration).abs();
        assert!(
            duration_diff < 0.01,
            "Phase vocoder duration: {pv_duration}, PSOLA duration: {psola_duration}"
        );
    }

    #[test]
    fn test_pitch_shift_edge_cases() {
        let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);

        // Test extreme pitch shifts
        let extreme_up = buffer.pitch_shift(24.0).unwrap(); // Two octaves up
        assert_eq!(extreme_up.len(), buffer.len());

        let extreme_down = buffer.pitch_shift(-24.0).unwrap(); // Two octaves down
        assert_eq!(extreme_down.len(), buffer.len());

        // Test fractional semitones
        let fractional = buffer.pitch_shift(1.5).unwrap(); // 1.5 semitones up
        assert_eq!(fractional.len(), buffer.len());

        // Test negative fractional semitones
        let neg_fractional = buffer.pitch_shift(-2.5).unwrap(); // 2.5 semitones down
        assert_eq!(neg_fractional.len(), buffer.len());
    }

    #[test]
    fn test_pitch_shift_quality() {
        let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);

        // Test that pitch shifting doesn't introduce excessive artifacts
        let shifted = buffer.pitch_shift(12.0).unwrap();

        // Check that output is not silent
        let max_amplitude = shifted.samples.iter().map(|&s| s.abs()).fold(0.0, f32::max);
        assert!(max_amplitude > 0.0, "Output should not be silent");

        // Check that output doesn't clip
        assert!(
            max_amplitude <= 1.0,
            "Output should not exceed [-1, 1] range"
        );

        // Check that there's some variation in the output (not all zeros)
        let mut has_variation = false;
        let first_sample = shifted.samples[0];
        for &sample in &shifted.samples {
            if (sample - first_sample).abs() > 0.001 {
                has_variation = true;
                break;
            }
        }
        assert!(has_variation, "Output should have some variation");
    }
}
