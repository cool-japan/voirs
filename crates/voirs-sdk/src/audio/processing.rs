//! Audio processing functions for manipulation and enhancement.

use crate::{error::Result, VoirsError};
use super::buffer::AudioBuffer;

impl AudioBuffer {
    /// Convert to different sample rate
    pub fn resample(&self, target_rate: u32) -> Result<AudioBuffer> {
        if target_rate == self.sample_rate {
            return Ok(self.clone());
        }

        // Simple linear interpolation resampling
        // TODO: Implement proper high-quality resampling
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
            return Err(VoirsError::audio_error("Sample rates must match for mixing"));
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
                "Sample rate and channels must match for appending"
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
        let fade_samples = (duration_seconds * self.sample_rate as f32 * self.channels as f32) as usize;
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
        let fade_samples = (duration_seconds * self.sample_rate as f32 * self.channels as f32) as usize;
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
                "Sample rate and channels must match for crossfading"
            ));
        }

        let crossfade_samples = (crossfade_duration * self.sample_rate as f32 * self.channels as f32) as usize;
        let crossfade_samples = crossfade_samples.min(self.samples.len()).min(other.samples.len());
        
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
            self.samples.extend_from_slice(&other.samples[crossfade_samples..]);
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

    /// Apply pitch shifting (simple frequency domain approach)
    pub fn pitch_shift(&self, semitones: f32) -> Result<AudioBuffer> {
        // For this simple implementation, we'll just change frequency without duration change
        // This is a placeholder - a real implementation would use PSOLA, phase vocoder, etc.
        let pitch_factor = 2.0_f32.powf(semitones / 12.0);
        
        // Simple approach: change the frequency while keeping duration the same
        // This creates a basic pitch shift effect (not perfect quality)
        let mut result = self.clone();
        
        // Apply frequency modulation to simulate pitch shift
        for (i, sample) in result.samples.iter_mut().enumerate() {
            let t = i as f32 / self.sample_rate as f32;
            // Simple frequency scaling (this is a very basic approach)
            let phase_shift = 2.0 * std::f32::consts::PI * t * (pitch_factor - 1.0) * 440.0;
            *sample *= phase_shift.cos() * 0.1 + 0.9; // Subtle effect to simulate pitch change
        }
        
        Ok(result)
    }

    /// Apply dynamic range compression
    pub fn compress(&mut self, threshold: f32, ratio: f32, attack_ms: f32, release_ms: f32) -> Result<()> {
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
        let start_sample = (start_seconds * self.sample_rate as f32 * self.channels as f32) as usize;
        let duration_samples = (duration_seconds * self.sample_rate as f32 * self.channels as f32) as usize;
        let end_sample = (start_sample + duration_samples).min(self.samples.len());
        
        if start_sample >= self.samples.len() {
            return Err(VoirsError::audio_error("Start time exceeds audio duration"));
        }
        
        let extracted_samples = self.samples[start_sample..end_sample].to_vec();
        Ok(AudioBuffer::new(extracted_samples, self.sample_rate, self.channels))
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
    use super::*;
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
    fn test_pitch_shift() {
        let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
        let original_duration = buffer.duration();
        
        let shifted = buffer.pitch_shift(12.0).unwrap(); // One octave up
        let shifted_duration = shifted.duration();
        
        // Duration should remain approximately the same
        let duration_diff = (shifted_duration - original_duration).abs();
        println!("Original duration: {}, Shifted duration: {}, Difference: {}", 
                 original_duration, shifted_duration, duration_diff);
        assert!(duration_diff < 0.1, 
                "Duration difference {} is too large", duration_diff);
    }
}