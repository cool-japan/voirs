//! Frequency enhancement module for high-frequency brightness and presence enhancement

use super::{db_to_linear, EnhancementConfig};
use crate::{AudioBuffer, Result};
use std::f32::consts::PI;

/// High-frequency enhancement processor
#[derive(Debug)]
pub struct FrequencyEnhancer {
    config: EnhancementConfig,
    sample_rate: f32,

    // Brightness filter state (high-pass filter)
    brightness_prev_input: f32,
    brightness_prev_output: f32,

    // Presence filter state (peaking filter)
    presence_prev_input: [f32; 2],
    presence_prev_output: [f32; 2],

    // Coefficients
    brightness_coeffs: HighPassCoeffs,
    presence_coeffs: PeakingCoeffs,

    // Statistics
    total_gain_applied: f32,
    processed_samples: u64,
}

#[derive(Debug, Clone)]
struct HighPassCoeffs {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

#[derive(Debug, Clone)]
struct PeakingCoeffs {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl FrequencyEnhancer {
    /// Create new frequency enhancer
    pub fn new(config: &EnhancementConfig, sample_rate: f32) -> Result<Self> {
        let brightness_coeffs =
            calculate_highpass_coeffs(config.high_freq_cutoff, config.q_factor, sample_rate)?;

        let presence_coeffs = calculate_peaking_coeffs(
            config.presence_freq,
            config.q_factor,
            config.presence_gain,
            sample_rate,
        )?;

        Ok(Self {
            config: config.clone(),
            sample_rate,
            brightness_prev_input: 0.0,
            brightness_prev_output: 0.0,
            presence_prev_input: [0.0; 2],
            presence_prev_output: [0.0; 2],
            brightness_coeffs,
            presence_coeffs,
            total_gain_applied: 0.0,
            processed_samples: 0,
        })
    }

    /// Process audio through frequency enhancement
    pub fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let samples = audio.samples_mut();
        let brightness_gain = db_to_linear(self.config.brightness_gain);

        for sample in samples.iter_mut() {
            // Apply brightness enhancement (high-frequency boost)
            let brightness_output = self.process_brightness_filter(*sample);
            let enhanced_brightness = *sample + brightness_output * brightness_gain;

            // Apply presence enhancement (mid-frequency boost)
            let presence_output = self.process_presence_filter(enhanced_brightness);

            *sample = presence_output;
            self.processed_samples += 1;
        }

        // Update total gain applied
        self.total_gain_applied = self.config.brightness_gain + self.config.presence_gain;

        Ok(())
    }

    /// Process single sample through brightness filter (high-pass)
    fn process_brightness_filter(&mut self, input: f32) -> f32 {
        let output = self.brightness_coeffs.b0 * input
                   + self.brightness_coeffs.b1 * self.brightness_prev_input
                   + self.brightness_coeffs.b2 * 0.0  // For high-pass, b2 is typically 0
                   - self.brightness_coeffs.a1 * self.brightness_prev_output
                   - self.brightness_coeffs.a2 * 0.0; // For high-pass, a2 is typically 0

        self.brightness_prev_input = input;
        self.brightness_prev_output = output;

        output
    }

    /// Process single sample through presence filter (peaking)
    fn process_presence_filter(&mut self, input: f32) -> f32 {
        let output = self.presence_coeffs.b0 * input
            + self.presence_coeffs.b1 * self.presence_prev_input[0]
            + self.presence_coeffs.b2 * self.presence_prev_input[1]
            - self.presence_coeffs.a1 * self.presence_prev_output[0]
            - self.presence_coeffs.a2 * self.presence_prev_output[1];

        // Shift delay line
        self.presence_prev_input[1] = self.presence_prev_input[0];
        self.presence_prev_input[0] = input;
        self.presence_prev_output[1] = self.presence_prev_output[0];
        self.presence_prev_output[0] = output;

        output
    }

    /// Get total gain applied
    pub fn total_gain(&self) -> f32 {
        self.total_gain_applied
    }

    /// Get number of processed samples
    pub fn processed_samples(&self) -> u64 {
        self.processed_samples
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &EnhancementConfig) -> Result<()> {
        self.config = config.clone();

        self.brightness_coeffs =
            calculate_highpass_coeffs(config.high_freq_cutoff, config.q_factor, self.sample_rate)?;

        self.presence_coeffs = calculate_peaking_coeffs(
            config.presence_freq,
            config.q_factor,
            config.presence_gain,
            self.sample_rate,
        )?;

        Ok(())
    }
}

/// Calculate high-pass filter coefficients
fn calculate_highpass_coeffs(cutoff_freq: f32, q: f32, sample_rate: f32) -> Result<HighPassCoeffs> {
    let omega = 2.0 * PI * cutoff_freq / sample_rate;
    let sin_omega = omega.sin();
    let cos_omega = omega.cos();
    let alpha = sin_omega / (2.0 * q);

    let b0 = (1.0 + cos_omega) / 2.0;
    let b1 = -(1.0 + cos_omega);
    let b2 = (1.0 + cos_omega) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha;

    Ok(HighPassCoeffs {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
    })
}

/// Calculate peaking filter coefficients
fn calculate_peaking_coeffs(
    center_freq: f32,
    q: f32,
    gain_db: f32,
    sample_rate: f32,
) -> Result<PeakingCoeffs> {
    let omega = 2.0 * PI * center_freq / sample_rate;
    let sin_omega = omega.sin();
    let cos_omega = omega.cos();
    let a = 10.0_f32.powf(gain_db / 40.0); // Gain factor
    let alpha = sin_omega / (2.0 * q);

    let b0 = 1.0 + alpha * a;
    let b1 = -2.0 * cos_omega;
    let b2 = 1.0 - alpha * a;
    let a0 = 1.0 + alpha / a;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha / a;

    Ok(PeakingCoeffs {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_enhancer_creation() {
        let config = EnhancementConfig::default();
        let enhancer = FrequencyEnhancer::new(&config, 44100.0);
        assert!(enhancer.is_ok());
    }

    #[test]
    fn test_frequency_enhancement_processing() {
        let config = EnhancementConfig::default();
        let mut enhancer = FrequencyEnhancer::new(&config, 44100.0).unwrap();

        let samples = vec![0.1, 0.2, 0.3, -0.1, -0.2];
        let mut audio = AudioBuffer::new(samples.clone(), 44100, 1);

        let result = enhancer.process(&mut audio);
        assert!(result.is_ok());

        // Verify samples were processed
        assert_eq!(enhancer.processed_samples(), 5);
        assert!(enhancer.total_gain() > 0.0);
    }

    #[test]
    fn test_disabled_enhancement() {
        let config = EnhancementConfig {
            enabled: false,
            ..Default::default()
        };

        let mut enhancer = FrequencyEnhancer::new(&config, 44100.0).unwrap();

        let samples = vec![0.1, 0.2, 0.3];
        let original_samples = samples.clone();
        let mut audio = AudioBuffer::new(samples, 44100, 1);

        enhancer.process(&mut audio).unwrap();

        // Samples should be unchanged when disabled
        assert_eq!(audio.samples(), &original_samples);
    }

    #[test]
    fn test_highpass_coeffs_calculation() {
        let coeffs = calculate_highpass_coeffs(1000.0, 0.707, 44100.0);
        assert!(coeffs.is_ok());

        let coeffs = coeffs.unwrap();
        assert!(coeffs.b0 > 0.0);
        assert!(coeffs.b1 < 0.0); // High-pass characteristic
    }

    #[test]
    fn test_peaking_coeffs_calculation() {
        let coeffs = calculate_peaking_coeffs(1000.0, 1.0, 3.0, 44100.0);
        assert!(coeffs.is_ok());

        let coeffs = coeffs.unwrap();
        assert!(coeffs.b0 > 0.0);
    }
}
