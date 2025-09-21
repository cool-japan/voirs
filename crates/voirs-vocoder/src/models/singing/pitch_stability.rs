//! Pitch stability processor for singing voice vocoder.

use crate::models::singing::config::PitchStabilityConfig;
use anyhow::Result;
use ndarray::{Array1, Array2};
use realfft::RealFftPlanner;
use std::collections::VecDeque;

/// Processor for pitch stability analysis and correction
pub struct PitchStabilityProcessor {
    /// Configuration
    config: PitchStabilityConfig,
    /// Pitch history buffer
    pitch_history: VecDeque<f32>,
    /// FFT planner for pitch analysis
    fft_planner: RealFftPlanner<f32>,
    /// Window size for pitch analysis
    window_size: usize,
    /// Hop size for analysis
    #[allow(dead_code)]
    hop_size: usize,
    /// Sample rate
    sample_rate: u32,
}

impl PitchStabilityProcessor {
    /// Create new pitch stability processor
    pub fn new(config: &PitchStabilityConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            pitch_history: VecDeque::with_capacity(100),
            fft_planner: RealFftPlanner::new(),
            window_size: 2048,
            hop_size: 512,
            sample_rate: 22050,
        })
    }

    /// Process mel spectrogram for pitch stability
    pub fn process(&mut self, mel_spectrogram: &Array2<f32>) -> Result<Array2<f32>> {
        if !self.config.enable_correction {
            return Ok(mel_spectrogram.clone());
        }

        let mut processed = mel_spectrogram.clone();
        let frames = mel_spectrogram.shape()[1];

        // Process each frame
        for frame_idx in 0..frames {
            let frame = mel_spectrogram.column(frame_idx);
            let pitch = self.estimate_pitch(&frame)?;

            // Apply pitch stability correction
            let corrected_pitch = self.apply_pitch_correction(pitch)?;

            // Update mel spectrogram with corrected pitch
            self.apply_pitch_correction_to_frame(
                &mut processed,
                frame_idx,
                pitch,
                corrected_pitch,
            )?;
        }

        Ok(processed)
    }

    /// Estimate pitch from mel spectrogram frame
    fn estimate_pitch(&self, frame: &ndarray::ArrayView1<f32>) -> Result<f32> {
        // Use both mel-based and FFT-based pitch estimation for better accuracy
        let mel_pitch = self.estimate_pitch_from_mel(frame)?;

        // For now, primarily use mel-based estimation
        // FFT-based estimation could be added for audio signal processing
        Ok(mel_pitch)
    }

    /// Estimate pitch from mel spectrogram frame (mel-based approach)
    fn estimate_pitch_from_mel(&self, frame: &ndarray::ArrayView1<f32>) -> Result<f32> {
        // Convert mel spectrogram to frequency domain for pitch estimation
        let mel_bins = frame.len();
        let mut frequencies = Vec::new();
        let mut magnitudes = Vec::new();

        // Convert mel scale to frequency scale
        for (bin_idx, &magnitude) in frame.iter().enumerate() {
            let mel_freq = bin_idx as f32 * (self.sample_rate as f32 / 2.0) / mel_bins as f32;
            let freq = self.mel_to_hz(mel_freq);
            frequencies.push(freq);
            magnitudes.push(magnitude);
        }

        // Find peak frequency as pitch estimate
        let mut max_magnitude = 0.0;
        let mut peak_freq = 0.0;

        for (i, &magnitude) in magnitudes.iter().enumerate() {
            if magnitude > max_magnitude && frequencies[i] > 80.0 && frequencies[i] < 800.0 {
                max_magnitude = magnitude;
                peak_freq = frequencies[i];
            }
        }

        Ok(peak_freq)
    }

    /// Advanced pitch estimation from raw audio using FFT analysis
    /// This method uses the configured FFT planner, window size, and hop size
    pub fn estimate_pitch_from_audio(&mut self, audio: &[f32]) -> Result<f32> {
        if audio.len() < self.window_size {
            return Ok(0.0);
        }

        // Create FFT for the configured window size
        let fft = self.fft_planner.plan_fft_forward(self.window_size);
        let mut spectrum =
            vec![realfft::num_complex::Complex::new(0.0, 0.0); self.window_size / 2 + 1];
        let mut input = audio[..self.window_size].to_vec();

        // Apply window function (Hann window)
        for (i, sample) in input.iter_mut().enumerate() {
            let window_val = 0.5
                * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (self.window_size - 1) as f32)
                        .cos());
            *sample *= window_val;
        }

        // Perform FFT
        fft.process(&mut input, &mut spectrum)
            .map_err(|e| anyhow::anyhow!("FFT error: {}", e))?;

        // Find fundamental frequency using autocorrelation in frequency domain
        let mut max_magnitude = 0.0;
        let mut peak_bin = 0;

        // Search in typical vocal range (80Hz - 800Hz)
        let min_bin = (80.0 * self.window_size as f32 / self.sample_rate as f32) as usize;
        let max_bin = (800.0 * self.window_size as f32 / self.sample_rate as f32) as usize;

        for (bin, spectrum_val) in spectrum
            .iter()
            .enumerate()
            .take(max_bin.min(spectrum.len()))
            .skip(min_bin)
        {
            let magnitude = spectrum_val.norm();
            if magnitude > max_magnitude {
                max_magnitude = magnitude;
                peak_bin = bin;
            }
        }

        // Convert bin to frequency
        let frequency = peak_bin as f32 * self.sample_rate as f32 / self.window_size as f32;
        Ok(frequency)
    }

    /// Apply pitch correction based on stability analysis
    fn apply_pitch_correction(&mut self, current_pitch: f32) -> Result<f32> {
        // Add to pitch history
        self.pitch_history.push_back(current_pitch);
        if self.pitch_history.len() > 20 {
            self.pitch_history.pop_front();
        }

        // Calculate pitch stability
        let stability = self.calculate_pitch_stability()?;

        // Apply correction if instability detected
        if stability < self.config.stability_threshold {
            let target_pitch = self.calculate_target_pitch()?;
            let correction_amount =
                (target_pitch - current_pitch) * self.config.correction_strength;

            // Limit correction to prevent overcorrection
            let max_correction = current_pitch * self.config.max_pitch_deviation;
            let corrected_pitch =
                current_pitch + correction_amount.clamp(-max_correction, max_correction);

            Ok(corrected_pitch)
        } else {
            Ok(current_pitch)
        }
    }

    /// Calculate pitch stability metric
    fn calculate_pitch_stability(&self) -> Result<f32> {
        if self.pitch_history.len() < 3 {
            return Ok(1.0);
        }

        let pitches: Vec<f32> = self.pitch_history.iter().cloned().collect();
        let mean_pitch = pitches.iter().sum::<f32>() / pitches.len() as f32;

        // Calculate standard deviation
        let variance = pitches
            .iter()
            .map(|p| (p - mean_pitch).powi(2))
            .sum::<f32>()
            / pitches.len() as f32;

        let std_dev = variance.sqrt();

        // Normalize stability metric (higher std_dev = lower stability)
        let stability = std_dev / mean_pitch;

        Ok(stability)
    }

    /// Calculate target pitch for correction
    fn calculate_target_pitch(&self) -> Result<f32> {
        if self.pitch_history.is_empty() {
            return Ok(0.0);
        }

        // Use smoothed average of recent pitches
        let recent_pitches: Vec<f32> = self.pitch_history.iter().rev().take(5).cloned().collect();

        let mut smoothed_pitch = 0.0;
        let mut weight_sum = 0.0;

        for (i, &pitch) in recent_pitches.iter().enumerate() {
            let weight = self.config.smoothing_factor.powi(i as i32);
            smoothed_pitch += pitch * weight;
            weight_sum += weight;
        }

        Ok(smoothed_pitch / weight_sum)
    }

    /// Apply pitch correction to mel spectrogram frame
    fn apply_pitch_correction_to_frame(
        &self,
        mel_spectrogram: &mut Array2<f32>,
        frame_idx: usize,
        original_pitch: f32,
        corrected_pitch: f32,
    ) -> Result<()> {
        if original_pitch == 0.0 || corrected_pitch == 0.0 {
            return Ok(());
        }

        let pitch_ratio = corrected_pitch / original_pitch;
        let mel_bins = mel_spectrogram.shape()[0];

        // Apply pitch shift by frequency scaling
        let mut corrected_frame = Array1::zeros(mel_bins);

        for bin_idx in 0..mel_bins {
            let original_freq = self.mel_bin_to_hz(bin_idx, mel_bins);
            let corrected_freq = original_freq * pitch_ratio;
            let corrected_bin = self.hz_to_mel_bin(corrected_freq, mel_bins);

            if corrected_bin < mel_bins {
                corrected_frame[bin_idx] = mel_spectrogram[[corrected_bin, frame_idx]];
            }
        }

        // Update the frame
        for bin_idx in 0..mel_bins {
            mel_spectrogram[[bin_idx, frame_idx]] = corrected_frame[bin_idx];
        }

        Ok(())
    }

    /// Convert mel frequency to Hz
    fn mel_to_hz(&self, mel: f32) -> f32 {
        700.0 * (mel / 1127.0).exp() - 700.0
    }

    /// Convert Hz to mel frequency
    fn hz_to_mel(&self, hz: f32) -> f32 {
        1127.0 * (1.0 + hz / 700.0).ln()
    }

    /// Convert mel bin index to Hz
    fn mel_bin_to_hz(&self, bin_idx: usize, mel_bins: usize) -> f32 {
        let mel_freq =
            (bin_idx as f32 / mel_bins as f32) * self.hz_to_mel(self.sample_rate as f32 / 2.0);
        self.mel_to_hz(mel_freq)
    }

    /// Convert Hz to mel bin index
    fn hz_to_mel_bin(&self, hz: f32, mel_bins: usize) -> usize {
        let mel_freq = self.hz_to_mel(hz);
        let max_mel = self.hz_to_mel(self.sample_rate as f32 / 2.0);
        let bin_idx = (mel_freq / max_mel * mel_bins as f32) as usize;
        bin_idx.min(mel_bins - 1)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &PitchStabilityConfig) -> Result<()> {
        self.config = config.clone();
        Ok(())
    }

    /// Get current pitch stability
    pub fn get_current_stability(&self) -> Result<f32> {
        self.calculate_pitch_stability()
    }

    /// Reset pitch history
    pub fn reset(&mut self) {
        self.pitch_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pitch_stability_processor_creation() {
        let config = PitchStabilityConfig::default();
        let processor = PitchStabilityProcessor::new(&config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_pitch_estimation() {
        let config = PitchStabilityConfig::default();
        let processor = PitchStabilityProcessor::new(&config).unwrap();

        // Create sample mel frame
        let frame = Array1::from_vec(vec![0.1, 0.5, 0.8, 0.3, 0.1]);
        let frame_view = frame.view();

        let pitch = processor.estimate_pitch(&frame_view);
        assert!(pitch.is_ok());
    }

    #[test]
    fn test_pitch_correction() {
        let config = PitchStabilityConfig::default();
        let mut processor = PitchStabilityProcessor::new(&config).unwrap();

        // Apply correction multiple times to build history
        for _ in 0..5 {
            let result = processor.apply_pitch_correction(440.0);
            assert!(result.is_ok());
        }

        let corrected = processor.apply_pitch_correction(480.0);
        assert!(corrected.is_ok());
        assert!(corrected.unwrap() != 480.0); // Should be corrected
    }

    #[test]
    fn test_mel_frequency_conversion() {
        let config = PitchStabilityConfig::default();
        let processor = PitchStabilityProcessor::new(&config).unwrap();

        let hz = 440.0;
        let mel = processor.hz_to_mel(hz);
        let hz_back = processor.mel_to_hz(mel);

        assert!((hz - hz_back).abs() < 0.1);
    }

    #[test]
    fn test_process_mel_spectrogram() {
        let config = PitchStabilityConfig::default();
        let mut processor = PitchStabilityProcessor::new(&config).unwrap();

        // Create sample mel spectrogram
        let mel = Array2::ones((80, 100));
        let result = processor.process(&mel);
        assert!(result.is_ok());

        let processed = result.unwrap();
        assert_eq!(processed.shape(), mel.shape());
    }

    #[test]
    fn test_stability_calculation() {
        let config = PitchStabilityConfig::default();
        let mut processor = PitchStabilityProcessor::new(&config).unwrap();

        // Add stable pitches
        processor.pitch_history.push_back(440.0);
        processor.pitch_history.push_back(441.0);
        processor.pitch_history.push_back(440.5);

        let stability = processor.calculate_pitch_stability();
        assert!(stability.is_ok());
        assert!(stability.unwrap() < 0.01); // Lower values indicate higher stability
    }

    #[test]
    fn test_config_update() {
        let config = PitchStabilityConfig::default();
        let mut processor = PitchStabilityProcessor::new(&config).unwrap();

        let new_config = PitchStabilityConfig {
            stability_threshold: 0.1,
            ..Default::default()
        };

        let result = processor.update_config(&new_config);
        assert!(result.is_ok());
        assert_eq!(processor.config.stability_threshold, 0.1);
    }

    #[test]
    fn test_processor_reset() {
        let config = PitchStabilityConfig::default();
        let mut processor = PitchStabilityProcessor::new(&config).unwrap();

        processor.pitch_history.push_back(440.0);
        processor.pitch_history.push_back(441.0);

        processor.reset();
        assert!(processor.pitch_history.is_empty());
    }
}
