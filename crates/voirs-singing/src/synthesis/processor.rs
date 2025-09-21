//! Synthesis processor for real-time processing

use super::harmonic::HarmonicProcessor;
use super::noise::NoiseProcessor;
use super::spectral::SpectralProcessor;
use ndarray::Array1;

/// Synthesis processor for real-time processing
pub struct SynthesisProcessor {
    /// Frame size
    frame_size: usize,
    /// Hop size
    hop_size: usize,
    /// Sample rate
    sample_rate: f32,
    /// Processing buffer
    buffer: Array1<f32>,
    /// Overlap buffer
    overlap_buffer: Array1<f32>,
    /// Window function
    window: Array1<f32>,
    /// Spectral processing
    spectral_processor: SpectralProcessor,
    /// Harmonic processor
    harmonic_processor: HarmonicProcessor,
    /// Noise processor
    noise_processor: NoiseProcessor,
}

impl SynthesisProcessor {
    /// Create a new synthesis processor
    pub fn new(frame_size: usize, hop_size: usize, sample_rate: f32) -> Self {
        let buffer = Array1::zeros(frame_size);
        let overlap_buffer = Array1::zeros(frame_size);
        let window = Self::create_window(frame_size);

        Self {
            frame_size,
            hop_size,
            sample_rate,
            buffer,
            overlap_buffer,
            window,
            spectral_processor: SpectralProcessor::new(frame_size),
            harmonic_processor: HarmonicProcessor::new(sample_rate),
            noise_processor: NoiseProcessor::new(),
        }
    }

    /// Process a frame of audio
    pub fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> crate::Result<()> {
        if input.len() != self.hop_size || output.len() != self.hop_size {
            return Err(crate::Error::Processing(
                "Input/output frame size mismatch".to_string(),
            ));
        }

        // Shift buffer and add new input
        self.shift_buffer(input);

        // Apply window
        let windowed = &self.buffer * &self.window;

        // Process through spectral processor
        let spectral_output = self
            .spectral_processor
            .process(&windowed, self.sample_rate)?;

        // Process through harmonic processor
        let harmonic_output = self
            .harmonic_processor
            .process(&spectral_output, self.sample_rate)?;

        // Process through noise processor
        let final_output = self.noise_processor.process(&harmonic_output)?;

        // Overlap-add reconstruction
        self.overlap_add(&final_output, output);

        Ok(())
    }

    /// Shift buffer for overlap processing
    fn shift_buffer(&mut self, input: &[f32]) {
        // Shift existing samples
        for i in 0..(self.frame_size - self.hop_size) {
            self.buffer[i] = self.buffer[i + self.hop_size];
        }

        // Add new samples
        for (i, &sample) in input.iter().enumerate() {
            self.buffer[self.frame_size - self.hop_size + i] = sample;
        }
    }

    /// Overlap-add reconstruction
    fn overlap_add(&mut self, processed: &Array1<f32>, output: &mut [f32]) {
        // Add overlap from previous frame
        for i in 0..self.hop_size {
            output[i] = self.overlap_buffer[i] + processed[i];
        }

        // Store overlap for next frame
        for i in 0..(self.frame_size - self.hop_size) {
            self.overlap_buffer[i] = processed[self.hop_size + i];
        }

        // Clear remaining overlap buffer
        for i in (self.frame_size - self.hop_size)..self.frame_size {
            if i < self.overlap_buffer.len() {
                self.overlap_buffer[i] = 0.0;
            }
        }
    }

    /// Create window function (Hann window)
    fn create_window(size: usize) -> Array1<f32> {
        let mut window = Array1::zeros(size);
        for i in 0..size {
            let phase = 2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32;
            window[i] = 0.5 * (1.0 - phase.cos());
        }
        window
    }

    /// Get frame size
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Get hop size
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.overlap_buffer.fill(0.0);
        self.spectral_processor.reset();
        self.harmonic_processor.reset();
        self.noise_processor.reset();
    }

    /// Set spectral envelope
    pub fn set_spectral_envelope(&mut self, envelope: &[f32]) -> crate::Result<()> {
        self.spectral_processor.set_envelope(envelope)
    }

    /// Set formant parameters
    pub fn set_formants(&mut self, freqs: &[f32], bws: &[f32], gains: &[f32]) -> crate::Result<()> {
        self.spectral_processor.set_formants(freqs, bws, gains)
    }

    /// Set fundamental frequency
    pub fn set_fundamental(&mut self, f0: f32) {
        self.harmonic_processor.set_fundamental(f0);
    }

    /// Set harmonic amplitudes
    pub fn set_harmonic_amplitudes(&mut self, amplitudes: &[f32]) {
        self.harmonic_processor.set_amplitudes(amplitudes);
    }

    /// Set noise level
    pub fn set_noise_level(&mut self, level: f32) {
        self.noise_processor.set_level(level);
    }

    /// Get processing latency in samples
    pub fn latency_samples(&self) -> usize {
        self.frame_size
    }

    /// Get processing latency in seconds
    pub fn latency_seconds(&self) -> f32 {
        self.frame_size as f32 / self.sample_rate
    }
}

impl Default for SynthesisProcessor {
    fn default() -> Self {
        Self::new(1024, 256, 44100.0)
    }
}
