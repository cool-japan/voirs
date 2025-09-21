//! Noise injection augmentation
//!
//! This module provides noise injection capabilities for audio augmentation
//! with various noise types and SNR control. Features SIMD optimizations for performance.

use crate::audio::simd::SimdAudioProcessor;
use crate::{AudioData, Result};
use std::f32::consts::PI;

/// Noise type enumeration
#[derive(Debug, Clone, Copy)]
pub enum NoiseType {
    /// White noise (flat spectrum)
    White,
    /// Pink noise (1/f spectrum)
    Pink,
    /// Brown noise (1/f² spectrum)
    Brown,
    /// Blue noise (f spectrum)
    Blue,
    /// Gaussian noise
    Gaussian,
    /// Environmental noise (simulated)
    Environmental,
}

/// Noise injection configuration
#[derive(Debug, Clone)]
pub struct NoiseConfig {
    /// Types of noise to inject
    pub noise_types: Vec<NoiseType>,
    /// SNR levels to apply (in dB)
    pub snr_levels: Vec<f32>,
    /// Noise color for colored noise
    pub noise_color: f32,
    /// Use dynamic SNR (varies over time)
    pub dynamic_snr: bool,
    /// SNR variation range for dynamic SNR
    pub snr_variation: f32,
    /// Preserve original signal statistics
    pub preserve_statistics: bool,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            noise_types: vec![NoiseType::White, NoiseType::Pink, NoiseType::Gaussian],
            snr_levels: vec![5.0, 10.0, 15.0, 20.0, 25.0],
            noise_color: 1.0,
            dynamic_snr: false,
            snr_variation: 3.0,
            preserve_statistics: true,
        }
    }
}

/// Noise injection augmentor
pub struct NoiseAugmentor {
    config: NoiseConfig,
    rng_state: u64,
    spare_normal: Option<f32>,
}

impl NoiseAugmentor {
    /// Create new noise augmentor with configuration
    pub fn new(config: NoiseConfig) -> Self {
        Self {
            config,
            rng_state: 1234567890,
            spare_normal: None,
        }
    }

    /// Create noise augmentor with default configuration
    pub fn with_default_config() -> Self {
        Self::new(NoiseConfig::default())
    }

    /// Apply noise injection to audio
    pub fn apply_noise_injection(
        &mut self,
        audio: &AudioData,
        noise_type: NoiseType,
        snr_db: f32,
    ) -> Result<AudioData> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();
        let channels = audio.channels();

        // Generate noise
        let noise = self.generate_noise(noise_type, samples.len(), sample_rate)?;

        // Calculate noise scaling factor from SNR
        let signal_rms = self.calculate_rms(samples);
        let noise_rms = self.calculate_rms(&noise);
        let snr_linear = 10.0_f32.powf(snr_db / 20.0);
        let noise_scale = if noise_rms > 0.0 {
            signal_rms / (noise_rms * snr_linear)
        } else {
            0.0
        };

        // Mix signal with noise
        let mut output_samples = Vec::with_capacity(samples.len());

        if self.config.dynamic_snr {
            // Apply dynamic SNR variation
            for (i, (&sample, &noise_sample)) in samples.iter().zip(noise.iter()).enumerate() {
                let variation = self.snr_variation(i, samples.len()) * self.config.snr_variation;
                let dynamic_snr = snr_db + variation;
                let dynamic_scale = signal_rms / (noise_rms * 10.0_f32.powf(dynamic_snr / 20.0));

                let mixed = sample + noise_sample * dynamic_scale;
                output_samples.push(mixed);
            }
        } else {
            // Apply constant SNR
            for (&sample, &noise_sample) in samples.iter().zip(noise.iter()) {
                let mixed = sample + noise_sample * noise_scale;
                output_samples.push(mixed);
            }
        }

        // Preserve statistics if requested
        if self.config.preserve_statistics {
            self.preserve_signal_statistics(&mut output_samples, samples);
        }

        Ok(AudioData::new(output_samples, sample_rate, channels))
    }

    /// Generate all noise variants for given audio
    pub fn generate_variants(&mut self, audio: &AudioData) -> Result<Vec<AudioData>> {
        let mut variants = Vec::new();

        // Clone the config vectors to avoid borrow checker issues
        let noise_types = self.config.noise_types.clone();
        let snr_levels = self.config.snr_levels.clone();

        for noise_type in noise_types {
            for snr_level in snr_levels.iter() {
                let noisy = self.apply_noise_injection(audio, noise_type, *snr_level)?;
                variants.push(noisy);
            }
        }

        Ok(variants)
    }

    /// Generate noise of specified type
    fn generate_noise(
        &mut self,
        noise_type: NoiseType,
        length: usize,
        sample_rate: u32,
    ) -> Result<Vec<f32>> {
        match noise_type {
            NoiseType::White => self.generate_white_noise(length),
            NoiseType::Pink => self.generate_pink_noise(length),
            NoiseType::Brown => self.generate_brown_noise(length),
            NoiseType::Blue => self.generate_blue_noise(length),
            NoiseType::Gaussian => self.generate_gaussian_noise(length),
            NoiseType::Environmental => self.generate_environmental_noise(length, sample_rate),
        }
    }

    /// Generate white noise
    fn generate_white_noise(&mut self, length: usize) -> Result<Vec<f32>> {
        let mut noise = Vec::with_capacity(length);

        for _ in 0..length {
            noise.push(self.random_normal());
        }

        Ok(noise)
    }

    /// Generate pink noise (1/f spectrum)
    fn generate_pink_noise(&mut self, length: usize) -> Result<Vec<f32>> {
        let mut noise = Vec::with_capacity(length);

        // Simple pink noise generation using running sum
        let mut b = [0.0; 7];

        for _ in 0..length {
            let white = self.random_normal();

            b[0] = 0.99886 * b[0] + white * 0.0555179;
            b[1] = 0.99332 * b[1] + white * 0.0750759;
            b[2] = 0.96900 * b[2] + white * 0.153852;
            b[3] = 0.86650 * b[3] + white * 0.3104856;
            b[4] = 0.55000 * b[4] + white * 0.5329522;
            b[5] = -0.7616 * b[5] - white * 0.0168980;

            let pink = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + white * 0.5362;
            b[6] = white * 0.115926;

            noise.push(pink * 0.11);
        }

        Ok(noise)
    }

    /// Generate brown noise (1/f² spectrum)
    fn generate_brown_noise(&mut self, length: usize) -> Result<Vec<f32>> {
        let mut noise = Vec::with_capacity(length);
        let mut state = 0.0;

        for _ in 0..length {
            let white = self.random_normal();
            state += white;

            // High-pass filter to prevent DC buildup
            state = state.clamp(-10.0, 10.0);

            noise.push(state * 0.1);
        }

        Ok(noise)
    }

    /// Generate blue noise (f spectrum)
    fn generate_blue_noise(&mut self, length: usize) -> Result<Vec<f32>> {
        let mut noise = Vec::with_capacity(length);
        let mut prev_sample = 0.0;

        for _ in 0..length {
            let white = self.random_normal();
            let blue = white - prev_sample;
            prev_sample = white;
            noise.push(blue * 0.5);
        }

        Ok(noise)
    }

    /// Generate Gaussian noise
    fn generate_gaussian_noise(&mut self, length: usize) -> Result<Vec<f32>> {
        let mut noise = Vec::with_capacity(length);

        for _ in 0..length {
            noise.push(self.random_normal());
        }

        Ok(noise)
    }

    /// Generate environmental noise (simulated)
    fn generate_environmental_noise(
        &mut self,
        length: usize,
        sample_rate: u32,
    ) -> Result<Vec<f32>> {
        let mut noise = Vec::with_capacity(length);

        // Simulate environmental noise with multiple frequency components
        let frequencies = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0];
        let amplitudes = [0.1, 0.15, 0.2, 0.25, 0.2, 0.1];

        for i in 0..length {
            let time = i as f32 / sample_rate as f32;
            let mut sample = 0.0;

            for (freq, amp) in frequencies.iter().zip(amplitudes.iter()) {
                let phase = 2.0 * PI * freq * time;
                sample += amp * phase.sin();
            }

            // Add some random variation
            sample += self.random_normal() * 0.05;

            noise.push(sample);
        }

        Ok(noise)
    }

    /// Calculate RMS of samples using SIMD optimization
    fn calculate_rms(&self, samples: &[f32]) -> f32 {
        SimdAudioProcessor::calculate_rms(samples)
    }

    /// Generate SNR variation over time
    fn snr_variation(&self, position: usize, total_length: usize) -> f32 {
        let normalized_pos = position as f32 / total_length as f32;
        let variation = (2.0 * PI * normalized_pos * 3.0).sin(); // 3 cycles over the audio
        variation * 0.5 // Scale to ±0.5
    }

    /// Preserve signal statistics using SIMD optimization
    fn preserve_signal_statistics(&self, output: &mut [f32], original: &[f32]) {
        if output.is_empty() || original.is_empty() {
            return;
        }

        let orig_rms = self.calculate_rms(original);
        let output_rms = self.calculate_rms(output);

        if output_rms > 0.0 {
            let scale = orig_rms / output_rms;
            SimdAudioProcessor::apply_gain(output, scale);
        }
    }

    /// Generate random normal distribution sample (Box-Muller transform)
    fn random_normal(&mut self) -> f32 {
        if let Some(spare) = self.spare_normal.take() {
            return spare;
        }

        let u1 = self.random_uniform();
        let u2 = self.random_uniform();

        let mag = 0.5 * (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * PI * u2).cos();
        let z1 = mag * (2.0 * PI * u2).sin();

        self.spare_normal = Some(z1);

        z0
    }

    /// Generate random uniform distribution sample
    fn random_uniform(&mut self) -> f32 {
        // Simple linear congruential generator
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.rng_state >> 16) & 0x7fff) as f32 / 32768.0
    }
}

/// Noise injection statistics
#[derive(Debug, Clone)]
pub struct NoiseStats {
    /// Number of variants generated
    pub variants_generated: usize,
    /// Noise types applied
    pub noise_types: Vec<NoiseType>,
    /// SNR levels applied
    pub snr_levels: Vec<f32>,
    /// Processing time
    pub processing_time: std::time::Duration,
    /// Quality metrics
    pub quality_metrics: Vec<f32>,
    /// Actual SNR measurements
    pub measured_snr: Vec<f32>,
}

impl Default for NoiseStats {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self {
            variants_generated: 0,
            noise_types: Vec::new(),
            snr_levels: Vec::new(),
            processing_time: std::time::Duration::from_secs(0),
            quality_metrics: Vec::new(),
            measured_snr: Vec::new(),
        }
    }

    /// Add variant statistics
    pub fn add_variant(
        &mut self,
        noise_type: NoiseType,
        snr_level: f32,
        quality: f32,
        measured_snr: f32,
    ) {
        self.variants_generated += 1;
        self.noise_types.push(noise_type);
        self.snr_levels.push(snr_level);
        self.quality_metrics.push(quality);
        self.measured_snr.push(measured_snr);
    }

    /// Set processing time
    pub fn set_processing_time(&mut self, duration: std::time::Duration) {
        self.processing_time = duration;
    }

    /// Get average quality
    pub fn average_quality(&self) -> f32 {
        if self.quality_metrics.is_empty() {
            0.0
        } else {
            self.quality_metrics.iter().sum::<f32>() / self.quality_metrics.len() as f32
        }
    }

    /// Get average measured SNR
    pub fn average_measured_snr(&self) -> f32 {
        if self.measured_snr.is_empty() {
            0.0
        } else {
            self.measured_snr.iter().sum::<f32>() / self.measured_snr.len() as f32
        }
    }

    /// Get SNR accuracy (how close measured SNR is to target)
    pub fn snr_accuracy(&self) -> f32 {
        if self.snr_levels.is_empty() || self.measured_snr.is_empty() {
            return 0.0;
        }

        let mut total_error = 0.0;
        for (target, measured) in self.snr_levels.iter().zip(self.measured_snr.iter()) {
            total_error += (target - measured).abs();
        }

        let avg_error = total_error / self.snr_levels.len() as f32;
        (10.0 - avg_error).max(0.0) / 10.0 * 100.0 // Convert to percentage
    }
}

/// Batch noise injection processor
pub struct BatchNoiseProcessor {
    augmentor: NoiseAugmentor,
}

impl BatchNoiseProcessor {
    /// Create new batch processor
    pub fn new(config: NoiseConfig) -> Self {
        Self {
            augmentor: NoiseAugmentor::new(config),
        }
    }

    /// Process multiple audio files with noise injection
    pub fn process_batch(
        &mut self,
        audio_files: &[AudioData],
    ) -> Result<(Vec<Vec<AudioData>>, NoiseStats)> {
        let start_time = std::time::Instant::now();
        let mut all_variants = Vec::new();
        let mut stats = NoiseStats::new();

        for audio in audio_files {
            let variants = self.augmentor.generate_variants(audio)?;

            // Calculate quality metrics for each variant
            let mut variant_idx = 0;
            for &noise_type in &self.augmentor.config.noise_types {
                for &snr_level in &self.augmentor.config.snr_levels {
                    if variant_idx < variants.len() {
                        let variant = &variants[variant_idx];
                        let quality = calculate_audio_quality(variant);
                        let measured_snr = measure_snr(audio, variant);
                        stats.add_variant(noise_type, snr_level, quality, measured_snr);
                        variant_idx += 1;
                    }
                }
            }

            all_variants.push(variants);
        }

        let processing_time = start_time.elapsed();
        stats.set_processing_time(processing_time);

        Ok((all_variants, stats))
    }
}

/// Calculate basic audio quality metric
fn calculate_audio_quality(audio: &AudioData) -> f32 {
    let samples = audio.samples();
    if samples.is_empty() {
        return 0.0;
    }

    // Calculate signal-to-noise ratio approximation
    let energy = samples.iter().map(|&x| x * x).sum::<f32>();
    let rms = (energy / samples.len() as f32).sqrt();

    // Simple quality metric based on RMS
    (rms * 100.0).min(100.0)
}

/// Measure actual SNR between original and noisy audio
fn measure_snr(original: &AudioData, noisy: &AudioData) -> f32 {
    let orig_samples = original.samples();
    let noisy_samples = noisy.samples();

    if orig_samples.is_empty() || noisy_samples.is_empty() {
        return 0.0;
    }

    let min_len = orig_samples.len().min(noisy_samples.len());

    // Calculate signal power
    let signal_power: f32 = orig_samples[..min_len].iter().map(|&x| x * x).sum();

    // Calculate noise power
    let mut noise_power = 0.0;
    for i in 0..min_len {
        let noise = noisy_samples[i] - orig_samples[i];
        noise_power += noise * noise;
    }

    if noise_power > 0.0 {
        let snr_linear = signal_power / noise_power;
        20.0 * snr_linear.log10()
    } else {
        100.0 // Very high SNR if no noise detected
    }
}

/// Adaptive noise injection
pub struct AdaptiveNoiseInjector {
    config: NoiseConfig,
    #[allow(dead_code)]
    rng_state: u64,
}

impl AdaptiveNoiseInjector {
    /// Create new adaptive noise injector
    pub fn new(config: NoiseConfig) -> Self {
        Self {
            config,
            rng_state: 987654321,
        }
    }

    /// Apply adaptive noise injection based on signal characteristics
    pub fn apply_adaptive_noise(&mut self, audio: &AudioData) -> Result<AudioData> {
        let samples = audio.samples();

        // Analyze signal characteristics
        let signal_energy = self.calculate_energy(samples);
        let spectral_centroid = self.calculate_spectral_centroid(samples, audio.sample_rate());

        // Choose noise type based on signal characteristics
        let noise_type = if spectral_centroid > 2000.0 {
            NoiseType::Pink // Use pink noise for high-frequency signals
        } else if signal_energy > 0.5 {
            NoiseType::White // Use white noise for high-energy signals
        } else {
            NoiseType::Gaussian // Use Gaussian noise for low-energy signals
        };

        // Choose SNR based on signal energy
        let snr_db = if signal_energy > 0.8 {
            20.0 // High SNR for high-energy signals
        } else if signal_energy > 0.3 {
            15.0 // Medium SNR for medium-energy signals
        } else {
            10.0 // Low SNR for low-energy signals
        };

        let mut augmentor = NoiseAugmentor::new(self.config.clone());
        augmentor.apply_noise_injection(audio, noise_type, snr_db)
    }

    /// Calculate signal energy
    fn calculate_energy(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let energy: f32 = samples.iter().map(|&x| x * x).sum();
        energy / samples.len() as f32
    }

    /// Calculate spectral centroid (simplified)
    fn calculate_spectral_centroid(&self, samples: &[f32], sample_rate: u32) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // Simple spectral centroid calculation
        // In a real implementation, this would use FFT
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &sample) in samples.iter().enumerate() {
            let frequency = i as f32 * sample_rate as f32 / samples.len() as f32;
            let magnitude = sample.abs();
            weighted_sum += frequency * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }
}
