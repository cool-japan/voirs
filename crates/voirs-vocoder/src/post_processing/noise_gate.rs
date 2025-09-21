//! Noise gate and spectral subtraction for audio cleanup
//!
//! Implements a noise gate with configurable attack/release times and
//! spectral subtraction for advanced noise reduction.

use super::{db_to_linear, linear_to_db, NoiseGateConfig};
use crate::{AudioBuffer, Result};
use std::collections::VecDeque;

/// Noise gate with envelope following and spectral subtraction
#[derive(Debug)]
pub struct NoiseGate {
    config: NoiseGateConfig,
    sample_rate: f32,

    // Gate state
    gate_state: GateState,
    envelope: f32,
    hold_counter: u32,
    hold_samples: u32,

    // Attack/release coefficients
    attack_coeff: f32,
    release_coeff: f32,

    // Spectral subtraction
    spectral_processor: Option<SpectralSubtractor>,

    // Statistics
    current_attenuation: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum GateState {
    Open,
    Closing,
    Closed,
    Opening,
    Hold,
}

impl NoiseGate {
    /// Create a new noise gate with the given configuration
    pub fn new(config: &NoiseGateConfig, sample_rate: f32) -> Result<Self> {
        let attack_coeff = (-1.0 / (config.attack_ms * 0.001 * sample_rate)).exp();
        let release_coeff = (-1.0 / (config.release_ms * 0.001 * sample_rate)).exp();
        let hold_samples = (config.hold_ms * 0.001 * sample_rate) as u32;

        let spectral_processor = if config.spectral_subtraction {
            Some(SpectralSubtractor::new(
                sample_rate,
                config.subtraction_factor,
            )?)
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            sample_rate,
            gate_state: GateState::Closed,
            envelope: 0.0,
            hold_counter: 0,
            hold_samples,
            attack_coeff,
            release_coeff,
            spectral_processor,
            current_attenuation: 0.0,
        })
    }

    /// Process audio buffer with noise gating
    pub fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // First apply spectral subtraction if enabled
        if let Some(ref mut spectral_processor) = self.spectral_processor {
            spectral_processor.process(audio)?;
        }

        // Then apply the noise gate
        self.apply_gate(audio)?;

        Ok(())
    }

    fn apply_gate(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        let samples = audio.samples_mut();
        let threshold_linear = db_to_linear(self.config.threshold);

        for sample in samples.iter_mut() {
            // Calculate input level
            let input_level = sample.abs();

            // Update envelope
            let target_envelope = input_level;
            if target_envelope > self.envelope {
                self.envelope =
                    target_envelope + (self.envelope - target_envelope) * self.attack_coeff;
            } else {
                self.envelope =
                    target_envelope + (self.envelope - target_envelope) * self.release_coeff;
            }

            // Determine gate state based on envelope
            let should_open = self.envelope > threshold_linear;

            // State machine for gate behavior
            match self.gate_state {
                GateState::Closed => {
                    if should_open {
                        self.gate_state = GateState::Opening;
                    }
                }
                GateState::Opening => {
                    if !should_open {
                        self.gate_state = GateState::Closing;
                    } else {
                        self.gate_state = GateState::Open;
                    }
                }
                GateState::Open => {
                    if !should_open {
                        self.gate_state = GateState::Hold;
                        self.hold_counter = 0;
                    }
                }
                GateState::Hold => {
                    if should_open {
                        self.gate_state = GateState::Open;
                    } else {
                        self.hold_counter += 1;
                        if self.hold_counter >= self.hold_samples {
                            self.gate_state = GateState::Closing;
                        }
                    }
                }
                GateState::Closing => {
                    if should_open {
                        self.gate_state = GateState::Opening;
                    } else {
                        self.gate_state = GateState::Closed;
                    }
                }
            }

            // Calculate gate gain based on state
            let gate_gain = match self.gate_state {
                GateState::Open | GateState::Hold => 1.0,
                GateState::Closed => 0.0,
                GateState::Opening => {
                    // Smooth fade in
                    let fade_ratio = self.envelope / threshold_linear;
                    fade_ratio.min(1.0)
                }
                GateState::Closing => {
                    // Smooth fade out
                    let fade_ratio = self.envelope / threshold_linear;
                    fade_ratio.min(1.0)
                }
            };

            // Apply gate gain
            *sample *= gate_gain;
            self.current_attenuation = linear_to_db(gate_gain);
        }

        Ok(())
    }

    /// Get current attenuation in dB
    pub fn attenuation(&self) -> f32 {
        self.current_attenuation
    }

    /// Reset internal state
    pub fn reset(&mut self) {
        self.gate_state = GateState::Closed;
        self.envelope = 0.0;
        self.hold_counter = 0;
        self.current_attenuation = 0.0;

        if let Some(ref mut spectral_processor) = self.spectral_processor {
            spectral_processor.reset();
        }
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &NoiseGateConfig) -> Result<()> {
        let old_spectral_enabled = self.config.spectral_subtraction;
        self.config = config.clone();

        // Recalculate coefficients
        self.attack_coeff = (-1.0 / (config.attack_ms * 0.001 * self.sample_rate)).exp();
        self.release_coeff = (-1.0 / (config.release_ms * 0.001 * self.sample_rate)).exp();
        self.hold_samples = (config.hold_ms * 0.001 * self.sample_rate) as u32;

        // Update spectral processor
        if config.spectral_subtraction && !old_spectral_enabled {
            self.spectral_processor = Some(SpectralSubtractor::new(
                self.sample_rate,
                config.subtraction_factor,
            )?);
        } else if !config.spectral_subtraction {
            self.spectral_processor = None;
        } else if let Some(ref mut spectral_processor) = self.spectral_processor {
            spectral_processor.update_factor(config.subtraction_factor);
        }

        Ok(())
    }
}

/// Spectral subtraction processor for advanced noise reduction
#[derive(Debug)]
struct SpectralSubtractor {
    #[allow(dead_code)]
    sample_rate: f32,
    fft_size: usize,
    overlap: usize,
    window: Vec<f32>,
    input_buffer: VecDeque<f32>,
    output_buffer: VecDeque<f32>,
    noise_spectrum: Vec<f32>,
    subtraction_factor: f32,
    noise_learning: bool,
    noise_samples_collected: usize,
    min_noise_samples: usize,
}

impl SpectralSubtractor {
    fn new(sample_rate: f32, subtraction_factor: f32) -> Result<Self> {
        let fft_size = 1024; // Fixed FFT size for now
        let overlap = fft_size / 2;

        // Create Hann window
        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * i as f32 / (fft_size - 1) as f32;
                0.5 * (1.0 - phase.cos())
            })
            .collect();

        let min_noise_samples = (sample_rate * 0.5) as usize; // 0.5 seconds of noise learning

        Ok(Self {
            sample_rate,
            fft_size,
            overlap,
            window,
            input_buffer: VecDeque::new(),
            output_buffer: VecDeque::new(),
            noise_spectrum: vec![0.0; fft_size / 2 + 1],
            subtraction_factor,
            noise_learning: true,
            noise_samples_collected: 0,
            min_noise_samples,
        })
    }

    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        let samples = audio.samples_mut();

        // Add samples to input buffer
        for &sample in samples.iter() {
            self.input_buffer.push_back(sample);
        }

        // Process frames when we have enough samples
        while self.input_buffer.len() >= self.fft_size {
            self.process_frame()?;
        }

        // Replace samples with processed output
        for sample in samples.iter_mut() {
            if let Some(output_sample) = self.output_buffer.pop_front() {
                *sample = output_sample;
            }
        }

        Ok(())
    }

    fn process_frame(&mut self) -> Result<()> {
        // Extract frame from input buffer
        let frame: Vec<f32> = self.input_buffer.range(0..self.fft_size).copied().collect();

        // Apply window
        let windowed: Vec<f32> = frame
            .iter()
            .zip(self.window.iter())
            .map(|(s, w)| s * w)
            .collect();

        // Simple spectral subtraction (simplified for demonstration)
        // In a real implementation, you would use a proper FFT library
        let processed_frame = if self.noise_learning {
            // During noise learning phase, just copy the input
            self.noise_samples_collected += self.overlap;
            if self.noise_samples_collected >= self.min_noise_samples {
                self.noise_learning = false;
            }
            windowed
        } else {
            // Apply spectral subtraction
            self.apply_spectral_subtraction(&windowed)
        };

        // Add processed samples to output buffer
        for &sample in processed_frame.iter() {
            self.output_buffer.push_back(sample);
        }

        // Remove processed samples from input buffer
        for _ in 0..self.overlap {
            self.input_buffer.pop_front();
        }

        Ok(())
    }

    fn apply_spectral_subtraction(&self, frame: &[f32]) -> Vec<f32> {
        // Simplified spectral subtraction - just apply a factor
        // In a real implementation, this would involve:
        // 1. FFT of the input frame
        // 2. Magnitude calculation
        // 3. Noise spectrum estimation and subtraction
        // 4. Inverse FFT

        frame
            .iter()
            .map(|&sample| sample * (1.0 - self.subtraction_factor * 0.5))
            .collect()
    }

    fn update_factor(&mut self, factor: f32) {
        self.subtraction_factor = factor;
    }

    fn reset(&mut self) {
        self.input_buffer.clear();
        self.output_buffer.clear();
        self.noise_learning = true;
        self.noise_samples_collected = 0;
        self.noise_spectrum.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_gate_creation() {
        let config = NoiseGateConfig::default();
        let gate = NoiseGate::new(&config, 48000.0);
        assert!(gate.is_ok());
    }

    #[test]
    fn test_noise_gate_process() {
        let config = NoiseGateConfig {
            threshold: -40.0,
            attack_ms: 1.0,
            release_ms: 100.0,
            hold_ms: 10.0,
            spectral_subtraction: false,
            subtraction_factor: 0.5,
            enabled: true,
        };

        let mut gate = NoiseGate::new(&config, 48000.0).unwrap();

        // Test with quiet signal (should be gated)
        let quiet_samples = vec![0.001; 100];
        let mut quiet_audio = AudioBuffer::from_samples(quiet_samples, 48000.0);

        let result = gate.process(&mut quiet_audio);
        assert!(result.is_ok());

        // Check that quiet signal was attenuated
        let max_amplitude = quiet_audio
            .samples()
            .iter()
            .map(|s| s.abs())
            .fold(0.0, f32::max);
        assert!(max_amplitude < 0.001);
    }

    #[test]
    fn test_noise_gate_with_spectral_subtraction() {
        let config = NoiseGateConfig {
            spectral_subtraction: true,
            subtraction_factor: 0.3,
            ..NoiseGateConfig::default()
        };

        let gate = NoiseGate::new(&config, 48000.0);
        assert!(gate.is_ok());

        let mut gate = gate.unwrap();
        assert!(gate.spectral_processor.is_some());

        // Test processing
        let samples = vec![0.1; 2048]; // Enough samples for spectral processing
        let mut audio = AudioBuffer::from_samples(samples, 48000.0);

        let result = gate.process(&mut audio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_spectral_subtractor() {
        let processor = SpectralSubtractor::new(48000.0, 0.5);
        assert!(processor.is_ok());

        let mut processor = processor.unwrap();

        // Test with sufficient samples
        let samples = vec![0.1; 2048];
        let mut audio = AudioBuffer::from_samples(samples, 48000.0);

        let result = processor.process(&mut audio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gate_state_transitions() {
        let config = NoiseGateConfig::default();
        let mut gate = NoiseGate::new(&config, 48000.0).unwrap();

        // Initially should be closed
        assert_eq!(gate.gate_state, GateState::Closed);

        // Test with loud signal
        let loud_samples = vec![0.5; 100];
        let mut loud_audio = AudioBuffer::from_samples(loud_samples, 48000.0);

        gate.process(&mut loud_audio).unwrap();

        // Should have opened or be opening
        assert!(matches!(
            gate.gate_state,
            GateState::Open | GateState::Opening
        ));
    }
}
