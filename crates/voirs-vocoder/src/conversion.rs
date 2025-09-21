//! Voice conversion functionality for real-time voice transformation.

use crate::{AudioBuffer, Result};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Voice conversion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConversionConfig {
    /// Target age adjustment (-1.0 to 1.0, where -1 is child-like, 1 is elderly)
    pub age_shift: f32,
    /// Target gender adjustment (-1.0 to 1.0, where -1 is feminine, 1 is masculine)
    pub gender_shift: f32,
    /// Pitch shift in semitones
    pub pitch_shift: f32,
    /// Formant frequency scaling (0.5 to 2.0)
    pub formant_scaling: f32,
    /// Voice breathiness (0.0 to 1.0)
    pub breathiness: f32,
    /// Voice roughness (0.0 to 1.0)
    pub roughness: f32,
    /// Voice brightness adjustment (-1.0 to 1.0)
    pub brightness: f32,
    /// Voice warmth adjustment (-1.0 to 1.0)
    pub warmth: f32,
    /// Intensity of conversion (0.0 to 1.0)
    pub conversion_strength: f32,
}

impl Default for VoiceConversionConfig {
    fn default() -> Self {
        Self {
            age_shift: 0.0,
            gender_shift: 0.0,
            pitch_shift: 0.0,
            formant_scaling: 1.0,
            breathiness: 0.0,
            roughness: 0.0,
            brightness: 0.0,
            warmth: 0.0,
            conversion_strength: 1.0,
        }
    }
}

impl VoiceConversionConfig {
    /// Create configuration for making voice younger
    pub fn younger(intensity: f32) -> Self {
        Self {
            age_shift: -intensity.clamp(0.0, 1.0),
            pitch_shift: intensity * 2.0,           // Higher pitch
            formant_scaling: 1.0 + intensity * 0.1, // Slightly higher formants
            brightness: intensity * 0.3,
            conversion_strength: intensity,
            ..Default::default()
        }
    }

    /// Create configuration for making voice older
    pub fn older(intensity: f32) -> Self {
        Self {
            age_shift: intensity.clamp(0.0, 1.0),
            pitch_shift: -intensity * 1.5,          // Lower pitch
            formant_scaling: 1.0 - intensity * 0.1, // Slightly lower formants
            roughness: intensity * 0.2,             // Add some roughness
            warmth: intensity * 0.2,
            conversion_strength: intensity,
            ..Default::default()
        }
    }

    /// Create configuration for more feminine voice
    pub fn feminine(intensity: f32) -> Self {
        Self {
            gender_shift: -intensity.clamp(0.0, 1.0),
            pitch_shift: intensity * 4.0,            // Higher pitch
            formant_scaling: 1.0 + intensity * 0.15, // Higher formants
            brightness: intensity * 0.4,
            breathiness: intensity * 0.1,
            conversion_strength: intensity,
            ..Default::default()
        }
    }

    /// Create configuration for more masculine voice
    pub fn masculine(intensity: f32) -> Self {
        Self {
            gender_shift: intensity.clamp(0.0, 1.0),
            pitch_shift: -intensity * 3.0,           // Lower pitch
            formant_scaling: 1.0 - intensity * 0.15, // Lower formants
            warmth: intensity * 0.3,
            roughness: intensity * 0.1,
            conversion_strength: intensity,
            ..Default::default()
        }
    }

    /// Create configuration for child-like voice
    pub fn child_like(intensity: f32) -> Self {
        Self {
            age_shift: -0.8 * intensity.clamp(0.0, 1.0),
            pitch_shift: intensity * 6.0, // Much higher pitch
            formant_scaling: 1.0 + intensity * 0.2,
            brightness: intensity * 0.5,
            breathiness: intensity * 0.05,
            conversion_strength: intensity,
            ..Default::default()
        }
    }

    /// Create configuration for robotic voice effect
    pub fn robotic(intensity: f32) -> Self {
        Self {
            pitch_shift: 0.0,
            formant_scaling: 1.0,
            brightness: intensity * 0.6,
            roughness: intensity * 0.4,
            warmth: -intensity * 0.3,
            conversion_strength: intensity,
            ..Default::default()
        }
    }
}

/// Real-time voice conversion processor
pub struct VoiceConverter {
    config: VoiceConversionConfig,
    sample_rate: u32,
    // Pitch shifting state
    pitch_buffer: Vec<f32>,
    pitch_buffer_size: usize,
    pitch_write_pos: usize,
    pitch_read_pos: f32,
    // Formant processing state
    formant_filter_state: FormantFilterState,
    // Processing buffers
    prev_samples: Vec<f32>,
    // Performance optimization
    buffer_size: usize,
}

/// State for formant filtering
#[derive(Debug, Clone)]
struct FormantFilterState {
    // Simplified formant filter coefficients
    a1: f32,
    a2: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    // Filter memory
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl FormantFilterState {
    fn new() -> Self {
        Self {
            a1: 0.0,
            a2: 0.0,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn update_coefficients(&mut self, sample_rate: f32, formant_scale: f32) {
        // Simplified formant filter design
        // In a full implementation, this would use more sophisticated formant modeling
        let base_freq = 1000.0 * formant_scale;
        let q = 2.0;

        let omega = 2.0 * PI * base_freq / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        // Peaking EQ coefficients
        self.b0 = 1.0 + alpha;
        self.b1 = -2.0 * cos_omega;
        self.b2 = 1.0 - alpha;
        self.a1 = self.b1;
        self.a2 = self.b2;

        // Normalize
        let norm = self.b0;
        self.b0 /= norm;
        self.b1 /= norm;
        self.b2 /= norm;
        self.a1 /= norm;
        self.a2 /= norm;
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        // Update delay line
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }
}

impl VoiceConverter {
    /// Create a new voice converter
    pub fn new(config: VoiceConversionConfig, sample_rate: u32) -> Self {
        let buffer_size = (sample_rate as f32 * 0.05) as usize; // 50ms buffer
        let pitch_buffer_size = buffer_size * 2; // Allow for pitch shifting

        let mut formant_filter_state = FormantFilterState::new();
        formant_filter_state.update_coefficients(sample_rate as f32, config.formant_scaling);

        Self {
            config,
            sample_rate,
            pitch_buffer: vec![0.0; pitch_buffer_size],
            pitch_buffer_size,
            pitch_write_pos: 0,
            pitch_read_pos: 0.0,
            formant_filter_state,
            prev_samples: vec![0.0; 4], // For filtering operations
            buffer_size,
        }
    }

    /// Update voice conversion configuration
    pub fn update_config(&mut self, config: VoiceConversionConfig) {
        self.config = config;
        self.formant_filter_state
            .update_coefficients(self.sample_rate as f32, self.config.formant_scaling);
    }

    /// Get current configuration
    pub fn config(&self) -> &VoiceConversionConfig {
        &self.config
    }

    /// Process audio buffer with voice conversion
    pub fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if audio.is_empty() {
            return Ok(());
        }

        // Apply voice conversion processing step by step
        // We call each method separately to avoid borrow checker issues
        self.apply_pitch_shift_to_buffer(audio)?;
        self.apply_formant_transformation_to_buffer(audio)?;
        self.apply_spectral_modifications_to_buffer(audio)?;
        self.apply_voice_characteristics_to_buffer(audio)?;

        Ok(())
    }

    /// Apply pitch shifting to audio buffer
    fn apply_pitch_shift_to_buffer(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if self.config.pitch_shift.abs() < 0.01 {
            return Ok(());
        }

        let samples = audio.samples_mut();
        self.apply_pitch_shift(samples)
    }

    /// Apply formant transformation to audio buffer
    fn apply_formant_transformation_to_buffer(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if (self.config.formant_scaling - 1.0).abs() < 0.01 {
            return Ok(());
        }

        let samples = audio.samples_mut();
        self.apply_formant_transformation(samples)
    }

    /// Apply spectral modifications to audio buffer
    fn apply_spectral_modifications_to_buffer(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        let samples = audio.samples_mut();
        self.apply_spectral_modifications(samples)
    }

    /// Apply voice characteristics to audio buffer
    fn apply_voice_characteristics_to_buffer(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        let samples = audio.samples_mut();
        self.apply_voice_characteristics(samples)
    }

    /// Apply pitch shifting using simple time-domain method
    fn apply_pitch_shift(&mut self, samples: &mut [f32]) -> Result<()> {
        if self.config.pitch_shift.abs() < 0.01 {
            return Ok(());
        }

        let pitch_ratio = 2.0_f32.powf(self.config.pitch_shift / 12.0);
        let intensity = self.config.conversion_strength;

        for sample in samples.iter_mut() {
            // Write to pitch buffer
            self.pitch_buffer[self.pitch_write_pos] = *sample;

            // Read from pitch buffer with interpolation
            let read_pos_int = self.pitch_read_pos as usize;
            let read_pos_frac = self.pitch_read_pos - read_pos_int as f32;

            let sample1 = self.pitch_buffer[read_pos_int % self.pitch_buffer_size];
            let sample2 = self.pitch_buffer[(read_pos_int + 1) % self.pitch_buffer_size];

            // Linear interpolation
            let pitched_sample = sample1 + read_pos_frac * (sample2 - sample1);

            // Mix with original based on intensity
            *sample = sample.mul_add(1.0 - intensity, pitched_sample * intensity);

            // Update positions
            self.pitch_write_pos = (self.pitch_write_pos + 1) % self.pitch_buffer_size;
            self.pitch_read_pos =
                (self.pitch_read_pos + pitch_ratio) % self.pitch_buffer_size as f32;
        }

        Ok(())
    }

    /// Apply formant transformation
    fn apply_formant_transformation(&mut self, samples: &mut [f32]) -> Result<()> {
        if (self.config.formant_scaling - 1.0).abs() < 0.01 {
            return Ok(());
        }

        let intensity = self.config.conversion_strength;

        for sample in samples.iter_mut() {
            let processed = self.formant_filter_state.process(*sample);
            *sample = sample.mul_add(1.0 - intensity, processed * intensity);
        }

        Ok(())
    }

    /// Apply spectral modifications for age and gender transformation
    fn apply_spectral_modifications(&mut self, samples: &mut [f32]) -> Result<()> {
        // Age-related spectral changes
        if self.config.age_shift.abs() > 0.01 {
            self.apply_age_spectral_changes(samples)?;
        }

        // Gender-related spectral changes
        if self.config.gender_shift.abs() > 0.01 {
            self.apply_gender_spectral_changes(samples)?;
        }

        Ok(())
    }

    /// Apply age-related spectral changes
    fn apply_age_spectral_changes(&mut self, samples: &mut [f32]) -> Result<()> {
        let age_shift = self.config.age_shift;
        let intensity = self.config.conversion_strength;

        // Younger voices: brighten spectrum
        // Older voices: dampen high frequencies
        let spectral_tilt = -age_shift * 0.2;

        for i in 1..samples.len() {
            let high_freq_emphasis = samples[i] - self.prev_samples[0];
            let modified = samples[i] + spectral_tilt * high_freq_emphasis * intensity;
            samples[i] = modified.clamp(-1.0, 1.0);

            // Update history
            self.prev_samples[0] = samples[i - 1];
        }

        Ok(())
    }

    /// Apply gender-related spectral changes
    fn apply_gender_spectral_changes(&mut self, samples: &mut [f32]) -> Result<()> {
        let gender_shift = self.config.gender_shift;
        let intensity = self.config.conversion_strength;

        // Feminine: emphasize higher frequencies
        // Masculine: emphasize lower frequencies
        let spectral_emphasis = gender_shift * 0.15;

        for i in 2..samples.len() {
            let spectral_diff = samples[i] - 0.5 * (samples[i - 1] + samples[i - 2]);
            let modified = samples[i] + spectral_emphasis * spectral_diff * intensity;
            samples[i] = modified.clamp(-1.0, 1.0);
        }

        Ok(())
    }

    /// Apply voice characteristics (breathiness, roughness, etc.)
    fn apply_voice_characteristics(&mut self, samples: &mut [f32]) -> Result<()> {
        self.apply_breathiness(samples)?;
        self.apply_roughness(samples)?;
        self.apply_brightness(samples)?;
        self.apply_warmth(samples)?;
        Ok(())
    }

    /// Apply breathiness effect
    fn apply_breathiness(&self, samples: &mut [f32]) -> Result<()> {
        if self.config.breathiness < 0.01 {
            return Ok(());
        }

        let breathiness = self.config.breathiness * self.config.conversion_strength;
        let noise_level = breathiness * 0.03;

        for (i, sample) in samples.iter_mut().enumerate() {
            // Generate deterministic noise
            let t = i as f32 * 0.001;
            let noise = (t * 1847.0).sin() * 0.3 + (t * 3271.0).sin() * 0.2;
            let breath_noise = noise * noise_level * sample.abs().sqrt();
            *sample = (*sample + breath_noise).clamp(-1.0, 1.0);
        }

        Ok(())
    }

    /// Apply roughness effect
    fn apply_roughness(&self, samples: &mut [f32]) -> Result<()> {
        if self.config.roughness < 0.01 {
            return Ok(());
        }

        let roughness = self.config.roughness * self.config.conversion_strength;

        for sample in samples.iter_mut() {
            if sample.abs() > 0.1 {
                let sign = sample.signum();
                let distortion_amount = 1.0 - roughness * 0.3;
                let distorted = sample.abs().powf(distortion_amount);
                *sample = (sign * distorted).clamp(-1.0, 1.0);
            }
        }

        Ok(())
    }

    /// Apply brightness adjustment
    fn apply_brightness(&self, samples: &mut [f32]) -> Result<()> {
        if self.config.brightness.abs() < 0.01 {
            return Ok(());
        }

        let brightness = self.config.brightness * self.config.conversion_strength;

        for i in 1..samples.len() {
            let high_freq = samples[i] - samples[i - 1];
            samples[i] = (samples[i] + brightness * 0.2 * high_freq).clamp(-1.0, 1.0);
        }

        Ok(())
    }

    /// Apply warmth adjustment
    fn apply_warmth(&self, samples: &mut [f32]) -> Result<()> {
        if self.config.warmth.abs() < 0.01 {
            return Ok(());
        }

        let warmth = self.config.warmth * self.config.conversion_strength;

        // Simple low-pass filtering for warmth
        let alpha = (1.0 - warmth * 0.3).clamp(0.1, 0.9);

        for i in 1..samples.len() {
            samples[i] = alpha * samples[i] + (1.0 - alpha) * samples[i - 1];
        }

        Ok(())
    }

    /// Reset internal state
    pub fn reset(&mut self) {
        self.pitch_buffer.fill(0.0);
        self.pitch_write_pos = 0;
        self.pitch_read_pos = 0.0;
        self.prev_samples.fill(0.0);

        // Reset formant filter state
        self.formant_filter_state.x1 = 0.0;
        self.formant_filter_state.x2 = 0.0;
        self.formant_filter_state.y1 = 0.0;
        self.formant_filter_state.y2 = 0.0;
    }

    /// Get processing latency in milliseconds
    pub fn get_latency_ms(&self) -> f32 {
        (self.buffer_size as f32 / self.sample_rate as f32) * 1000.0
    }
}

/// Voice morphing for interpolating between voice characteristics
pub struct VoiceMorpher {
    converter_a: VoiceConverter,
    converter_b: VoiceConverter,
    morph_factor: f32, // 0.0 = full A, 1.0 = full B
}

impl VoiceMorpher {
    /// Create a new voice morpher
    pub fn new(
        config_a: VoiceConversionConfig,
        config_b: VoiceConversionConfig,
        sample_rate: u32,
    ) -> Self {
        Self {
            converter_a: VoiceConverter::new(config_a, sample_rate),
            converter_b: VoiceConverter::new(config_b, sample_rate),
            morph_factor: 0.0,
        }
    }

    /// Set morph factor (0.0 to 1.0)
    pub fn set_morph_factor(&mut self, factor: f32) {
        self.morph_factor = factor.clamp(0.0, 1.0);
    }

    /// Get current morph factor
    pub fn morph_factor(&self) -> f32 {
        self.morph_factor
    }

    /// Process audio with morphing between two voice configurations
    pub fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if self.morph_factor < 0.01 {
            // Pure A
            self.converter_a.process(audio)
        } else if self.morph_factor > 0.99 {
            // Pure B
            self.converter_b.process(audio)
        } else {
            // Create interpolated config
            let config_a = self.converter_a.config().clone();
            let config_b = self.converter_b.config().clone();
            let interpolated_config = interpolate_configs(&config_a, &config_b, self.morph_factor);

            // Use converter A with interpolated config
            self.converter_a.update_config(interpolated_config);
            self.converter_a.process(audio)
        }
    }

    /// Reset both converters
    pub fn reset(&mut self) {
        self.converter_a.reset();
        self.converter_b.reset();
    }
}

/// Interpolate between two voice conversion configurations
fn interpolate_configs(
    config_a: &VoiceConversionConfig,
    config_b: &VoiceConversionConfig,
    factor: f32,
) -> VoiceConversionConfig {
    VoiceConversionConfig {
        age_shift: config_a.age_shift + factor * (config_b.age_shift - config_a.age_shift),
        gender_shift: config_a.gender_shift
            + factor * (config_b.gender_shift - config_a.gender_shift),
        pitch_shift: config_a.pitch_shift + factor * (config_b.pitch_shift - config_a.pitch_shift),
        formant_scaling: config_a.formant_scaling
            + factor * (config_b.formant_scaling - config_a.formant_scaling),
        breathiness: config_a.breathiness + factor * (config_b.breathiness - config_a.breathiness),
        roughness: config_a.roughness + factor * (config_b.roughness - config_a.roughness),
        brightness: config_a.brightness + factor * (config_b.brightness - config_a.brightness),
        warmth: config_a.warmth + factor * (config_b.warmth - config_a.warmth),
        conversion_strength: config_a.conversion_strength
            + factor * (config_b.conversion_strength - config_a.conversion_strength),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_conversion_config_presets() {
        let younger = VoiceConversionConfig::younger(0.5);
        assert_eq!(younger.age_shift, -0.5);
        assert!(younger.pitch_shift > 0.0);

        let older = VoiceConversionConfig::older(0.7);
        assert_eq!(older.age_shift, 0.7);
        assert!(older.pitch_shift < 0.0);

        let feminine = VoiceConversionConfig::feminine(0.8);
        assert_eq!(feminine.gender_shift, -0.8);
        assert!(feminine.formant_scaling > 1.0);

        let masculine = VoiceConversionConfig::masculine(0.6);
        assert_eq!(masculine.gender_shift, 0.6);
        assert!(masculine.formant_scaling < 1.0);
    }

    #[test]
    fn test_voice_converter_creation() {
        let config = VoiceConversionConfig::default();
        let converter = VoiceConverter::new(config, 22050);
        assert_eq!(converter.sample_rate, 22050);
        assert_eq!(converter.config().conversion_strength, 1.0);
    }

    #[test]
    fn test_voice_morpher() {
        let config_a = VoiceConversionConfig::feminine(0.5);
        let config_b = VoiceConversionConfig::masculine(0.5);
        let mut morpher = VoiceMorpher::new(config_a, config_b, 22050);

        morpher.set_morph_factor(0.5);
        assert_eq!(morpher.morph_factor(), 0.5);

        morpher.set_morph_factor(1.5); // Should clamp to 1.0
        assert_eq!(morpher.morph_factor(), 1.0);
    }

    #[test]
    fn test_config_interpolation() {
        let config_a = VoiceConversionConfig::feminine(1.0);
        let config_b = VoiceConversionConfig::masculine(1.0);
        let interpolated = interpolate_configs(&config_a, &config_b, 0.5);

        // Should be halfway between feminine and masculine
        assert!((interpolated.gender_shift).abs() < 0.01); // Should be close to 0
        assert!(interpolated.pitch_shift < config_a.pitch_shift);
        assert!(interpolated.pitch_shift > config_b.pitch_shift);
    }
}
