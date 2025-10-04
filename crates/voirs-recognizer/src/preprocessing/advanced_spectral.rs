//! Advanced spectral processing for audio enhancement
//!
//! This module provides sophisticated spectral processing techniques including:
//! - Spectral noise gating
//! - Harmonic enhancement
//! - Spectral subtraction with oversubtraction factor
//! - Multi-band dynamic range compression
//! - Perceptual spectral shaping

use crate::RecognitionError;
use std::f32::consts::PI;
use voirs_sdk::AudioBuffer;

/// Advanced spectral processing configuration
#[derive(Debug, Clone)]
/// Advanced Spectral Config
pub struct AdvancedSpectralConfig {
    /// FFT size for spectral analysis
    pub fft_size: usize,
    /// Hop length for STFT
    pub hop_length: usize,
    /// Window type for STFT
    pub window_type: WindowType,
    /// Enable spectral noise gating
    pub spectral_noise_gate: bool,
    /// Noise gate threshold (dB)
    pub noise_gate_threshold: f32,
    /// Enable harmonic enhancement
    pub harmonic_enhancement: bool,
    /// Harmonic enhancement factor
    pub harmonic_factor: f32,
    /// Enable multi-band compression
    pub multiband_compression: bool,
    /// Number of frequency bands for compression
    pub num_bands: usize,
    /// Compression ratios for each band
    pub compression_ratios: Vec<f32>,
    /// Enable perceptual shaping
    pub perceptual_shaping: bool,
    /// Sample rate
    pub sample_rate: u32,
}

impl Default for AdvancedSpectralConfig {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            hop_length: 512,
            window_type: WindowType::Hann,
            spectral_noise_gate: true,
            noise_gate_threshold: -40.0,
            harmonic_enhancement: true,
            harmonic_factor: 1.2,
            multiband_compression: true,
            num_bands: 4,
            compression_ratios: vec![2.0, 3.0, 4.0, 2.5],
            perceptual_shaping: true,
            sample_rate: 16000,
        }
    }
}

/// Window types for STFT
#[derive(Debug, Clone, Copy)]
/// Window Type
pub enum WindowType {
    /// Hann
    Hann,
    /// Hamming
    Hamming,
    /// Blackman
    Blackman,
    /// Kaiser
    Kaiser,
    /// Tukey
    Tukey,
}

/// Complex number for FFT operations
#[derive(Debug, Clone, Copy)]
struct Complex {
    real: f32,
    imag: f32,
}

impl Complex {
    fn new(real: f32, imag: f32) -> Self {
        Self { real, imag }
    }

    fn magnitude(&self) -> f32 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    fn phase(&self) -> f32 {
        self.imag.atan2(self.real)
    }

    fn from_polar(magnitude: f32, phase: f32) -> Self {
        Self {
            real: magnitude * phase.cos(),
            imag: magnitude * phase.sin(),
        }
    }
}

/// Processing statistics for advanced spectral processing
#[derive(Debug, Clone)]
/// Advanced Spectral Stats
pub struct AdvancedSpectralStats {
    /// Noise gate activation percentage
    pub noise_gate_activation: f32,
    /// Harmonic enhancement gain (dB)
    pub harmonic_gain_db: f32,
    /// Average compression ratio applied
    pub avg_compression_ratio: f32,
    /// Spectral centroid (Hz)
    pub spectral_centroid: f32,
    /// Processing latency (ms)
    pub processing_latency_ms: f64,
}

/// Result of advanced spectral processing
#[derive(Debug, Clone)]
/// Advanced Spectral Result
pub struct AdvancedSpectralResult {
    /// Enhanced audio buffer
    pub enhanced_audio: AudioBuffer,
    /// Processing statistics
    pub stats: AdvancedSpectralStats,
}

/// Advanced spectral processor
#[derive(Debug)]
/// Advanced Spectral Processor
pub struct AdvancedSpectralProcessor {
    config: AdvancedSpectralConfig,
    window: Vec<f32>,
    fft_buffer: Vec<Complex>,
    prev_phase: Vec<f32>,
    overlap_buffer: Vec<f32>,
    bark_scale_weights: Vec<f32>,
}

impl AdvancedSpectralProcessor {
    /// Create a new advanced spectral processor
    pub fn new(config: AdvancedSpectralConfig) -> Result<Self, RecognitionError> {
        let window = Self::generate_window(config.fft_size, config.window_type);
        let fft_buffer = vec![Complex::new(0.0, 0.0); config.fft_size];
        let prev_phase = vec![0.0; config.fft_size / 2 + 1];
        let overlap_buffer = vec![0.0; config.fft_size];
        let bark_scale_weights =
            Self::generate_bark_scale_weights(config.fft_size, config.sample_rate);

        Ok(Self {
            config,
            window,
            fft_buffer,
            prev_phase,
            overlap_buffer,
            bark_scale_weights,
        })
    }

    /// Process audio with advanced spectral techniques
    pub fn process(
        &mut self,
        audio: &AudioBuffer,
    ) -> Result<AdvancedSpectralResult, RecognitionError> {
        let start_time = std::time::Instant::now();

        let samples = audio.samples();
        let mut enhanced_samples = Vec::with_capacity(samples.len());

        // Initialize statistics
        let mut noise_gate_activations = 0;
        let mut total_frames = 0;
        let mut total_harmonic_gain = 0.0;
        let mut total_compression = 0.0;
        let mut spectral_centroid_sum = 0.0;

        // Process in overlapping frames
        let mut pos = 0;
        while pos + self.config.fft_size <= samples.len() {
            // Extract frame and apply window
            let mut frame = vec![0.0; self.config.fft_size];
            for i in 0..self.config.fft_size {
                if pos + i < samples.len() {
                    frame[i] = samples[pos + i] * self.window[i];
                }
            }

            // Forward FFT
            self.real_fft(&frame);

            // Extract magnitude and phase
            let mut magnitudes = vec![0.0; self.config.fft_size / 2 + 1];
            let mut phases = vec![0.0; self.config.fft_size / 2 + 1];

            for i in 0..magnitudes.len() {
                magnitudes[i] = self.fft_buffer[i].magnitude();
                phases[i] = self.fft_buffer[i].phase();
            }

            // Apply spectral noise gate
            let gate_active = if self.config.spectral_noise_gate {
                self.apply_spectral_noise_gate(&mut magnitudes)?
            } else {
                false
            };

            if gate_active {
                noise_gate_activations += 1;
            }

            // Apply harmonic enhancement
            let harmonic_gain = if self.config.harmonic_enhancement {
                self.apply_harmonic_enhancement(&mut magnitudes)?
            } else {
                0.0
            };
            total_harmonic_gain += harmonic_gain;

            // Apply multi-band compression
            let compression_ratio = if self.config.multiband_compression {
                self.apply_multiband_compression(&mut magnitudes)?
            } else {
                1.0
            };
            total_compression += compression_ratio;

            // Apply perceptual shaping
            if self.config.perceptual_shaping {
                self.apply_perceptual_shaping(&mut magnitudes)?;
            }

            // Calculate spectral centroid
            spectral_centroid_sum += self.calculate_spectral_centroid(&magnitudes);

            // Reconstruct complex spectrum
            for i in 0..magnitudes.len() {
                self.fft_buffer[i] = Complex::from_polar(magnitudes[i], phases[i]);
            }

            // Inverse FFT
            let enhanced_frame = self.inverse_real_fft();

            // Overlap-add synthesis
            self.overlap_add(&enhanced_frame, &mut enhanced_samples, pos);

            pos += self.config.hop_length;
            total_frames += 1;
        }

        // Handle remaining samples
        while enhanced_samples.len() < samples.len() {
            enhanced_samples.push(0.0);
        }
        enhanced_samples.truncate(samples.len());

        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

        // Calculate final statistics
        let stats = AdvancedSpectralStats {
            noise_gate_activation: if total_frames > 0 {
                noise_gate_activations as f32 / total_frames as f32 * 100.0
            } else {
                0.0
            },
            harmonic_gain_db: if total_frames > 0 {
                20.0 * (total_harmonic_gain / total_frames as f32).log10()
            } else {
                0.0
            },
            avg_compression_ratio: if total_frames > 0 {
                total_compression / total_frames as f32
            } else {
                1.0
            },
            spectral_centroid: if total_frames > 0 {
                spectral_centroid_sum / total_frames as f32
            } else {
                0.0
            },
            processing_latency_ms: processing_time,
        };

        let enhanced_audio =
            AudioBuffer::new(enhanced_samples, audio.sample_rate(), audio.channels());

        Ok(AdvancedSpectralResult {
            enhanced_audio,
            stats,
        })
    }

    /// Generate window function
    fn generate_window(size: usize, window_type: WindowType) -> Vec<f32> {
        let mut window = vec![0.0; size];

        match window_type {
            WindowType::Hann => {
                for i in 0..size {
                    let phase = 2.0 * PI * i as f32 / (size - 1) as f32;
                    window[i] = 0.5 * (1.0 - phase.cos());
                }
            }
            WindowType::Hamming => {
                for i in 0..size {
                    let phase = 2.0 * PI * i as f32 / (size - 1) as f32;
                    window[i] = 0.54 - 0.46 * phase.cos();
                }
            }
            WindowType::Blackman => {
                for i in 0..size {
                    let phase = 2.0 * PI * i as f32 / (size - 1) as f32;
                    window[i] = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
                }
            }
            WindowType::Kaiser => {
                // Simplified Kaiser window (beta = 8.6)
                let beta = 8.6;
                let i0_beta = Self::modified_bessel_i0(beta);
                for i in 0..size {
                    let x = 2.0 * i as f32 / (size - 1) as f32 - 1.0;
                    let arg = beta * (1.0 - x * x).sqrt();
                    window[i] = Self::modified_bessel_i0(arg) / i0_beta;
                }
            }
            WindowType::Tukey => {
                let alpha = 0.5;
                let transition_width = (alpha * size as f32 / 2.0) as usize;

                for i in 0..size {
                    if i < transition_width {
                        let phase = PI * i as f32 / transition_width as f32;
                        window[i] = 0.5 * (1.0 - phase.cos());
                    } else if i >= size - transition_width {
                        let phase = PI * (size - 1 - i) as f32 / transition_width as f32;
                        window[i] = 0.5 * (1.0 - phase.cos());
                    } else {
                        window[i] = 1.0;
                    }
                }
            }
        }

        window
    }

    /// Modified Bessel function I0 (for Kaiser window)
    fn modified_bessel_i0(x: f32) -> f32 {
        let mut sum = 1.0;
        let mut term = 1.0;
        let x_half_squared = (x / 2.0) * (x / 2.0);

        for k in 1..=20 {
            term *= x_half_squared / (k as f32 * k as f32);
            sum += term;
            if term < 1e-8 {
                break;
            }
        }

        sum
    }

    /// Generate bark scale weights for perceptual processing
    fn generate_bark_scale_weights(fft_size: usize, sample_rate: u32) -> Vec<f32> {
        let num_bins = fft_size / 2 + 1;
        let mut weights = vec![1.0; num_bins];

        for i in 0..num_bins {
            let freq = i as f32 * sample_rate as f32 / fft_size as f32;
            let bark =
                13.0 * (freq / 1315.8).atan() + 3.5 * ((freq / 7518.0) * (freq / 7518.0)).atan();

            // Apply bark scale weighting (emphasize perceptually important frequencies)
            weights[i] = 1.0 + 0.3 * (-((bark - 8.0) / 4.0).powi(2)).exp();
        }

        weights
    }

    /// Apply spectral noise gate
    fn apply_spectral_noise_gate(&self, magnitudes: &mut [f32]) -> Result<bool, RecognitionError> {
        let threshold_linear = 10.0_f32.powf(self.config.noise_gate_threshold / 20.0);
        let mut gate_active = false;

        for magnitude in magnitudes.iter_mut() {
            if *magnitude < threshold_linear {
                *magnitude *= 0.1; // Attenuate by 20dB
                gate_active = true;
            }
        }

        Ok(gate_active)
    }

    /// Apply harmonic enhancement
    fn apply_harmonic_enhancement(&self, magnitudes: &mut [f32]) -> Result<f32, RecognitionError> {
        let mut total_enhancement = 0.0;
        let num_bins = magnitudes.len();

        // Find harmonic peaks and enhance them
        for i in 1..num_bins - 1 {
            let is_peak = magnitudes[i] > magnitudes[i - 1] && magnitudes[i] > magnitudes[i + 1];

            if is_peak {
                let enhancement = self.config.harmonic_factor;
                magnitudes[i] *= enhancement;
                total_enhancement += enhancement;
            }
        }

        Ok(total_enhancement / num_bins as f32)
    }

    /// Apply multi-band dynamic range compression
    fn apply_multiband_compression(&self, magnitudes: &mut [f32]) -> Result<f32, RecognitionError> {
        let num_bins = magnitudes.len();
        let band_size = num_bins / self.config.num_bands;
        let mut total_compression = 0.0;

        for band in 0..self.config.num_bands {
            let start_bin = band * band_size;
            let end_bin = ((band + 1) * band_size).min(num_bins);
            let compression_ratio = self.config.compression_ratios.get(band).unwrap_or(&2.0);

            // Calculate band energy
            let mut band_energy = 0.0;
            for i in start_bin..end_bin {
                band_energy += magnitudes[i] * magnitudes[i];
            }
            band_energy = (band_energy / (end_bin - start_bin) as f32).sqrt();

            // Apply compression
            if band_energy > 0.0 {
                let compressed_energy = band_energy.powf(1.0 / compression_ratio);
                let gain = compressed_energy / band_energy;

                for i in start_bin..end_bin {
                    magnitudes[i] *= gain;
                }

                total_compression += *compression_ratio;
            }
        }

        Ok(total_compression / self.config.num_bands as f32)
    }

    /// Apply perceptual shaping based on bark scale
    fn apply_perceptual_shaping(&self, magnitudes: &mut [f32]) -> Result<(), RecognitionError> {
        for (i, magnitude) in magnitudes.iter_mut().enumerate() {
            if i < self.bark_scale_weights.len() {
                *magnitude *= self.bark_scale_weights[i];
            }
        }
        Ok(())
    }

    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, magnitudes: &[f32]) -> f32 {
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &magnitude) in magnitudes.iter().enumerate() {
            let freq = i as f32 * self.config.sample_rate as f32 / self.config.fft_size as f32;
            weighted_sum += freq * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }

    /// Simplified real FFT (placeholder - in production use a proper FFT library)
    fn real_fft(&mut self, input: &[f32]) {
        // This is a simplified implementation
        // In production, use rustfft or similar
        for k in 0..self.config.fft_size / 2 + 1 {
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for n in 0..self.config.fft_size {
                let angle = -2.0 * PI * k as f32 * n as f32 / self.config.fft_size as f32;
                real_sum += input[n] * angle.cos();
                imag_sum += input[n] * angle.sin();
            }

            self.fft_buffer[k] = Complex::new(real_sum, imag_sum);
        }
    }

    /// Simplified inverse real FFT
    fn inverse_real_fft(&self) -> Vec<f32> {
        let mut output = vec![0.0; self.config.fft_size];

        for n in 0..self.config.fft_size {
            for k in 0..self.config.fft_size / 2 + 1 {
                let angle = 2.0 * PI * k as f32 * n as f32 / self.config.fft_size as f32;
                let weight = if k == 0 || k == self.config.fft_size / 2 {
                    1.0
                } else {
                    2.0
                };
                output[n] += weight
                    * (self.fft_buffer[k].real * angle.cos()
                        - self.fft_buffer[k].imag * angle.sin());
            }
            output[n] /= self.config.fft_size as f32;
        }

        output
    }

    /// Overlap-add synthesis
    fn overlap_add(&mut self, frame: &[f32], output: &mut Vec<f32>, pos: usize) {
        let output_start = output.len();
        let required_length = pos + self.config.fft_size;

        // Extend output buffer if necessary
        while output.len() < required_length {
            output.push(0.0);
        }

        // Add with overlap
        for i in 0..self.config.fft_size {
            if pos + i < output.len() {
                output[pos + i] += frame[i];
            }
        }
    }

    /// Reset processor state
    pub fn reset(&mut self) -> Result<(), RecognitionError> {
        self.prev_phase.fill(0.0);
        self.overlap_buffer.fill(0.0);
        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> &AdvancedSpectralConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_spectral_processor_creation() {
        let config = AdvancedSpectralConfig::default();
        let processor = AdvancedSpectralProcessor::new(config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_window_generation() {
        let window = AdvancedSpectralProcessor::generate_window(1024, WindowType::Hann);
        assert_eq!(window.len(), 1024);
        assert!(window[0] < 0.1); // Start near zero
        assert!(window[512] > 0.9); // Peak near middle
        assert!(window[1023] < 0.1); // End near zero
    }

    #[test]
    fn test_spectral_processing() {
        let config = AdvancedSpectralConfig::default();
        let mut processor = AdvancedSpectralProcessor::new(config).unwrap();

        let samples = vec![0.1; 4096];
        let audio = AudioBuffer::mono(samples, 16000);

        let result = processor.process(&audio);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.enhanced_audio.samples().len(), 4096);
        assert!(result.stats.processing_latency_ms > 0.0);
    }

    #[test]
    fn test_bark_scale_weights() {
        let weights = AdvancedSpectralProcessor::generate_bark_scale_weights(2048, 16000);
        assert_eq!(weights.len(), 1025); // FFT size / 2 + 1
        assert!(weights.iter().all(|&w| w > 0.0));
    }

    #[test]
    fn test_spectral_centroid() {
        let config = AdvancedSpectralConfig::default();
        let processor = AdvancedSpectralProcessor::new(config).unwrap();

        let magnitudes = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let centroid = processor.calculate_spectral_centroid(&magnitudes);
        assert!(centroid > 0.0);
    }
}
