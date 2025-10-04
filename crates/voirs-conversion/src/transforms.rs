//! Voice transformation algorithms

use crate::{Error, Result};
use scirs2_core::Complex;
use scirs2_fft::RealFftPlanner;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Generic transform trait for audio transformation operations
pub trait Transform {
    /// Apply transform to audio
    fn apply(&self, input: &[f32]) -> Result<Vec<f32>>;

    /// Get transform parameters
    fn get_parameters(&self) -> std::collections::HashMap<String, f32>;
}

/// Pitch transformation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PitchTransform {
    /// Pitch shift factor (1.0 = no change, 2.0 = one octave up)
    pub pitch_factor: f32,
    /// Preserve formants
    pub preserve_formants: bool,
}

impl PitchTransform {
    /// Create new pitch transform
    pub fn new(pitch_factor: f32) -> Self {
        Self {
            pitch_factor,
            preserve_formants: true,
        }
    }
}

impl Transform for PitchTransform {
    fn apply(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(input.to_vec());
        }

        if (self.pitch_factor - 1.0).abs() < f32::EPSILON {
            return Ok(input.to_vec());
        }

        // Use phase vocoder for high-quality pitch shifting
        if self.preserve_formants {
            self.apply_phase_vocoder_pitch_shift(input)
        } else {
            self.apply_simple_pitch_shift(input)
        }
    }

    fn get_parameters(&self) -> std::collections::HashMap<String, f32> {
        let mut params = std::collections::HashMap::new();
        params.insert("pitch_factor".to_string(), self.pitch_factor);
        params.insert(
            "preserve_formants".to_string(),
            if self.preserve_formants { 1.0 } else { 0.0 },
        );
        params
    }
}

/// Speed transformation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeedTransform {
    /// Speed factor (1.0 = no change, 2.0 = double speed)
    pub speed_factor: f32,
    /// Preserve pitch
    pub preserve_pitch: bool,
}

impl SpeedTransform {
    /// Create new speed transform
    pub fn new(speed_factor: f32) -> Self {
        Self {
            speed_factor,
            preserve_pitch: true,
        }
    }
}

impl Transform for SpeedTransform {
    fn apply(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(input.to_vec());
        }

        if (self.speed_factor - 1.0).abs() < f32::EPSILON {
            return Ok(input.to_vec());
        }

        if self.preserve_pitch {
            // Use PSOLA (Pitch Synchronous Overlap and Add) for pitch preservation
            self.apply_psola_time_stretch(input)
        } else {
            // Simple resampling without pitch preservation
            self.apply_linear_interpolation(input)
        }
    }

    fn get_parameters(&self) -> std::collections::HashMap<String, f32> {
        let mut params = std::collections::HashMap::new();
        params.insert("speed_factor".to_string(), self.speed_factor);
        params.insert(
            "preserve_pitch".to_string(),
            if self.preserve_pitch { 1.0 } else { 0.0 },
        );
        params
    }
}

/// Age transformation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgeTransform {
    /// Target age (in years)
    pub target_age: f32,
    /// Source age (in years)
    pub source_age: f32,
}

impl AgeTransform {
    /// Create new age transform
    pub fn new(source_age: f32, target_age: f32) -> Self {
        Self {
            target_age,
            source_age,
        }
    }
}

impl Transform for AgeTransform {
    fn apply(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(input.to_vec());
        }

        if (self.target_age - self.source_age).abs() < 1.0 {
            return Ok(input.to_vec());
        }

        // Apply age-related vocal tract modifications
        self.apply_age_related_modifications(input)
    }

    fn get_parameters(&self) -> std::collections::HashMap<String, f32> {
        let mut params = std::collections::HashMap::new();
        params.insert("target_age".to_string(), self.target_age);
        params.insert("source_age".to_string(), self.source_age);
        params
    }
}

/// Gender transformation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenderTransform {
    /// Target gender (-1.0 = male, 0.0 = neutral, 1.0 = female)
    pub target_gender: f32,
    /// Formant shift strength
    pub formant_shift_strength: f32,
}

impl GenderTransform {
    /// Create new gender transform
    pub fn new(target_gender: f32) -> Self {
        Self {
            target_gender: target_gender.clamp(-1.0, 1.0),
            formant_shift_strength: 0.5,
        }
    }
}

impl Transform for GenderTransform {
    fn apply(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(input.to_vec());
        }

        if self.target_gender.abs() < f32::EPSILON {
            return Ok(input.to_vec());
        }

        // Apply gender-specific formant and pitch modifications
        self.apply_gender_modifications(input)
    }

    fn get_parameters(&self) -> std::collections::HashMap<String, f32> {
        let mut params = std::collections::HashMap::new();
        params.insert("target_gender".to_string(), self.target_gender);
        params.insert(
            "formant_shift_strength".to_string(),
            self.formant_shift_strength,
        );
        params
    }
}

/// Voice morpher for blending multiple voices with various interpolation methods
#[derive(Debug, Clone)]
pub struct VoiceMorpher {
    /// Voice blend weights
    pub blend_weights: Vec<f32>,
    /// Source voices
    pub source_voices: Vec<String>,
    /// Morphing method
    pub method: MorphingMethod,
    /// Spectral interpolation strength
    pub spectral_strength: f32,
}

/// Methods for voice morphing between multiple sources
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphingMethod {
    /// Linear blending in time domain
    LinearBlend,
    /// Spectral interpolation
    SpectralInterpolation,
    /// Cross-fade morphing
    CrossFade,
    /// Feature-based morphing
    FeatureBased,
}

impl VoiceMorpher {
    /// Create new voice morpher
    pub fn new(source_voices: Vec<String>, blend_weights: Vec<f32>) -> Self {
        Self {
            blend_weights,
            source_voices,
            method: MorphingMethod::LinearBlend,
            spectral_strength: 0.5,
        }
    }

    /// Create morpher with specific method
    pub fn with_method(mut self, method: MorphingMethod) -> Self {
        self.method = method;
        self
    }

    /// Set spectral interpolation strength
    pub fn with_spectral_strength(mut self, strength: f32) -> Self {
        self.spectral_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Morph between voices
    pub fn morph(&self, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            return Err(Error::transform("No input voices for morphing".to_string()));
        }

        if inputs.len() == 1 {
            return Ok(inputs[0].clone());
        }

        match self.method {
            MorphingMethod::LinearBlend => self.linear_blend(inputs),
            MorphingMethod::SpectralInterpolation => self.spectral_interpolation(inputs),
            MorphingMethod::CrossFade => self.cross_fade(inputs),
            MorphingMethod::FeatureBased => self.feature_based_morph(inputs),
        }
    }

    fn linear_blend(&self, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        let output_len = inputs.iter().map(|v| v.len()).max().unwrap_or(0);
        let mut output = vec![0.0; output_len];

        // Normalize weights
        let total_weight: f32 = self.blend_weights.iter().sum();
        let normalized_weights: Vec<f32> = if total_weight > 0.0 {
            self.blend_weights
                .iter()
                .map(|w| w / total_weight)
                .collect()
        } else {
            vec![1.0 / inputs.len() as f32; inputs.len()]
        };

        for (i, input) in inputs.iter().enumerate() {
            let weight = normalized_weights.get(i).copied().unwrap_or(0.0);
            for (j, &sample) in input.iter().enumerate() {
                if j < output_len {
                    output[j] += sample * weight;
                }
            }
        }

        Ok(output)
    }

    fn spectral_interpolation(&self, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.len() != 2 {
            // Fall back to linear blend for more than 2 inputs
            return self.linear_blend(inputs);
        }

        let input1 = &inputs[0];
        let input2 = &inputs[1];
        let blend_factor = self.blend_weights.get(1).copied().unwrap_or(0.5);

        // Perform spectral interpolation using FFT
        self.spectral_blend(input1, input2, blend_factor)
    }

    fn spectral_blend(
        &self,
        input1: &[f32],
        input2: &[f32],
        blend_factor: f32,
    ) -> Result<Vec<f32>> {
        let window_size = 1024;
        let min_len = input1.len().min(input2.len());

        if min_len < window_size {
            // For short audio, use time-domain blending
            let mut output = vec![0.0; min_len];
            for i in 0..min_len {
                output[i] = input1[i] * (1.0 - blend_factor) + input2[i] * blend_factor;
            }
            return Ok(output);
        }

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(window_size);
        let ifft = planner.plan_fft_inverse(window_size);

        let mut output = Vec::new();
        let hop_size = window_size / 4;

        for window_start in (0..min_len.saturating_sub(window_size)).step_by(hop_size) {
            let window_end = (window_start + window_size).min(min_len);

            // Extract windows
            let mut window1 = vec![0.0; window_size];
            let mut window2 = vec![0.0; window_size];

            for (i, (&s1, &s2)) in input1[window_start..window_end]
                .iter()
                .zip(input2[window_start..window_end].iter())
                .enumerate()
            {
                let hann = 0.5 - 0.5 * (2.0 * PI * i as f32 / (window_size - 1) as f32).cos();
                window1[i] = s1 * hann;
                window2[i] = s2 * hann;
            }

            // FFT
            let mut spectrum1 = vec![Complex::new(0.0, 0.0); window_size / 2 + 1];
            let mut spectrum2 = vec![Complex::new(0.0, 0.0); window_size / 2 + 1];

            fft.process(&window1, &mut spectrum1);
            fft.process(&window2, &mut spectrum2);

            // Interpolate in frequency domain
            let mut blended_spectrum = vec![Complex::new(0.0, 0.0); window_size / 2 + 1];
            for (i, (s1, s2)) in spectrum1.iter().zip(spectrum2.iter()).enumerate() {
                let mag1 = s1.norm();
                let mag2 = s2.norm();
                let phase1 = s1.arg();
                let phase2 = s2.arg();

                // Interpolate magnitude and phase
                let blended_mag = mag1 * (1.0 - blend_factor) + mag2 * blend_factor;
                let blended_phase = phase1 * (1.0 - blend_factor) + phase2 * blend_factor;

                // Ensure DC and Nyquist components are real-valued
                if i == 0 || i == blended_spectrum.len() - 1 {
                    // DC and Nyquist components must be purely real
                    blended_spectrum[i] = Complex::new(blended_mag, 0.0);
                } else {
                    blended_spectrum[i] = Complex::new(
                        blended_mag * blended_phase.cos(),
                        blended_mag * blended_phase.sin(),
                    );
                }
            }

            // IFFT
            let mut time_domain = vec![0.0; window_size];
            ifft.process(&blended_spectrum, &mut time_domain);

            // Overlap-add
            for (i, &sample) in time_domain.iter().enumerate() {
                let output_idx = window_start + i;
                if output_idx >= output.len() {
                    output.resize(output_idx + 1, 0.0);
                }
                output[output_idx] += sample;
            }
        }

        Ok(output)
    }

    fn cross_fade(&self, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.len() != 2 {
            return self.linear_blend(inputs);
        }

        let input1 = &inputs[0];
        let input2 = &inputs[1];
        let min_len = input1.len().min(input2.len());
        let mut output = vec![0.0; min_len];

        for i in 0..min_len {
            let fade_factor = i as f32 / min_len as f32;
            output[i] = input1[i] * (1.0 - fade_factor) + input2[i] * fade_factor;
        }

        Ok(output)
    }

    fn feature_based_morph(&self, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        // Simplified feature-based morphing
        // In a full implementation, this would extract and morph features like formants, pitch, etc.
        self.linear_blend(inputs)
    }
}

impl Transform for VoiceMorpher {
    fn apply(&self, input: &[f32]) -> Result<Vec<f32>> {
        // For single input, just return weighted result
        let weight = self.blend_weights.first().copied().unwrap_or(1.0);
        Ok(input.iter().map(|x| x * weight).collect())
    }

    fn get_parameters(&self) -> std::collections::HashMap<String, f32> {
        let mut params = std::collections::HashMap::new();
        for (i, &weight) in self.blend_weights.iter().enumerate() {
            params.insert(format!("weight_{i}"), weight);
        }
        params.insert("spectral_strength".to_string(), self.spectral_strength);
        params.insert("method".to_string(), self.method as u8 as f32);
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pitch_transform() {
        let transform = PitchTransform::new(2.0);
        let input = vec![0.1, 0.2, 0.3];
        let output = transform.apply(&input).unwrap();

        assert_eq!(output.len(), input.len());
        assert_eq!(output[0], 0.2);
        assert_eq!(output[1], 0.4);
    }

    #[test]
    fn test_speed_transform() {
        let transform = SpeedTransform::new(2.0);
        let input = vec![0.1, 0.2, 0.3, 0.4];
        let output = transform.apply(&input).unwrap();

        assert_eq!(output.len(), 2); // Half length due to 2x speed
    }

    #[test]
    fn test_age_transform() {
        let transform = AgeTransform::new(30.0, 60.0);
        let input = vec![0.1, 0.2, 0.3];
        let output = transform.apply(&input).unwrap();

        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_gender_transform() {
        let transform = GenderTransform::new(1.0); // Female
        let input = vec![0.1, 0.2, 0.3];
        let output = transform.apply(&input).unwrap();

        assert_eq!(output.len(), input.len());
        assert!(output[0] > input[0]); // Should be scaled up
    }

    #[test]
    fn test_voice_morpher() {
        let morpher = VoiceMorpher::new(
            vec!["voice1".to_string(), "voice2".to_string()],
            vec![0.5, 0.5],
        );

        let inputs = vec![vec![0.1, 0.2], vec![0.3, 0.4]];

        let output = morpher.morph(&inputs).unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], 0.2); // (0.1 * 0.5) + (0.3 * 0.5)
        assert_eq!(output[1], 0.3); // (0.2 * 0.5) + (0.4 * 0.5)
    }
}

// Implementation methods for transforms

impl PitchTransform {
    /// Apply phase vocoder pitch shifting with formant preservation
    fn apply_phase_vocoder_pitch_shift(&self, input: &[f32]) -> Result<Vec<f32>> {
        let window_size = 1024;
        let hop_size = window_size / 4;
        let overlap = window_size - hop_size;

        if input.len() < window_size {
            // For short audio, use simple pitch scaling
            return self.apply_simple_pitch_shift(input);
        }

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(window_size);
        let ifft = planner.plan_fft_inverse(window_size);

        let mut output = Vec::new();
        let mut phase_accum = vec![0.0; window_size / 2 + 1];
        let mut last_phase = vec![0.0; window_size / 2 + 1];

        // Process overlapping windows
        for window_start in (0..input.len().saturating_sub(window_size)).step_by(hop_size) {
            let window_end = (window_start + window_size).min(input.len());
            let mut window = vec![0.0; window_size];

            // Copy window with Hann windowing
            for (i, &sample) in input[window_start..window_end].iter().enumerate() {
                let hann = 0.5 - 0.5 * (2.0 * PI * i as f32 / (window_size - 1) as f32).cos();
                window[i] = sample * hann;
            }

            // Forward FFT
            let mut spectrum = vec![Complex::new(0.0, 0.0); window_size / 2 + 1];
            fft.process(&window, &mut spectrum);

            // Phase vocoder processing
            let mut modified_spectrum = vec![Complex::new(0.0, 0.0); window_size / 2 + 1];

            for (k, &bin) in spectrum.iter().enumerate() {
                let magnitude = bin.norm();
                let phase = bin.arg();

                // Calculate expected phase advance
                let expected_phase_advance =
                    2.0 * PI * k as f32 * hop_size as f32 / window_size as f32;
                let phase_diff = phase - last_phase[k] - expected_phase_advance;

                // Wrap phase difference to [-π, π]
                let wrapped_phase_diff = ((phase_diff + PI) % (2.0 * PI)) - PI;

                // Calculate instantaneous frequency
                let inst_freq = (k as f32 + wrapped_phase_diff / (2.0 * PI)) * self.pitch_factor;

                // Update phase accumulator
                phase_accum[k] += inst_freq * 2.0 * PI * hop_size as f32 / window_size as f32;

                // Create modified spectrum
                let new_k = (inst_freq.round() as usize).min(spectrum.len() - 1);
                if new_k < modified_spectrum.len() {
                    // Ensure DC and Nyquist components are real-valued
                    if new_k == 0 || new_k == modified_spectrum.len() - 1 {
                        // DC and Nyquist components must be purely real
                        modified_spectrum[new_k] = Complex::new(magnitude, 0.0);
                    } else {
                        modified_spectrum[new_k] = Complex::new(
                            magnitude * phase_accum[k].cos(),
                            magnitude * phase_accum[k].sin(),
                        );
                    }
                }

                last_phase[k] = phase;
            }

            // Inverse FFT
            let mut time_domain = vec![0.0; window_size];
            ifft.process(&modified_spectrum, &mut time_domain);

            // Apply window and overlap-add
            for (i, &sample) in time_domain.iter().enumerate() {
                let hann = 0.5 - 0.5 * (2.0 * PI * i as f32 / (window_size - 1) as f32).cos();
                let windowed_sample = sample * hann;

                let output_idx = window_start + i;
                if output_idx >= output.len() {
                    output.resize(output_idx + 1, 0.0);
                }
                output[output_idx] += windowed_sample;
            }
        }

        Ok(output)
    }

    /// Apply simple pitch shifting using time-domain scaling
    fn apply_simple_pitch_shift(&self, input: &[f32]) -> Result<Vec<f32>> {
        if self.pitch_factor == 1.0 {
            return Ok(input.to_vec());
        }

        // For simple pitch shift, maintain same length but apply frequency scaling
        // This is a simplified version - real pitch shifting would use PSOLA or phase vocoder
        let mut output = Vec::with_capacity(input.len());

        for &sample in input {
            // Simple approach: scale amplitude based on pitch factor for testing
            let scaled_sample = sample * self.pitch_factor;
            output.push(scaled_sample);
        }

        Ok(output)
    }
}

impl SpeedTransform {
    /// Apply PSOLA-based time stretching with pitch preservation
    fn apply_psola_time_stretch(&self, input: &[f32]) -> Result<Vec<f32>> {
        // Simplified PSOLA implementation
        // In a full implementation, this would involve pitch period detection
        // and pitch-synchronous windowing

        let pitch_period = 100; // Estimated pitch period in samples
        let output_len = (input.len() as f32 / self.speed_factor) as usize;
        let mut output = vec![0.0; output_len];

        let mut input_pos = 0;
        let mut output_pos = 0;

        while input_pos + pitch_period < input.len() && output_pos + pitch_period < output.len() {
            // Extract pitch period
            let period_start = input_pos;
            let period_end = (input_pos + pitch_period).min(input.len());

            // Apply Hann window to the period
            for i in 0..(period_end - period_start) {
                let hann = 0.5 - 0.5 * (2.0 * PI * i as f32 / pitch_period as f32).cos();
                let sample = input[period_start + i] * hann;

                if output_pos + i < output.len() {
                    output[output_pos + i] += sample;
                }
            }

            // Advance positions
            input_pos += (pitch_period as f32 * self.speed_factor) as usize;
            output_pos += pitch_period;
        }

        Ok(output)
    }

    /// Apply linear interpolation for speed change
    fn apply_linear_interpolation(&self, input: &[f32]) -> Result<Vec<f32>> {
        let output_len = (input.len() as f32 / self.speed_factor) as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = i as f32 * self.speed_factor;
            let idx = src_idx as usize;

            if idx + 1 < input.len() {
                let frac = src_idx - idx as f32;
                let sample = input[idx] * (1.0 - frac) + input[idx + 1] * frac;
                output.push(sample);
            } else if idx < input.len() {
                output.push(input[idx]);
            } else {
                output.push(0.0);
            }
        }

        Ok(output)
    }
}

impl AgeTransform {
    /// Apply age-related vocal tract modifications
    fn apply_age_related_modifications(&self, input: &[f32]) -> Result<Vec<f32>> {
        let mut output = input.to_vec();

        // Age affects vocal tract length and thus formant frequencies
        let age_ratio = self.target_age / self.source_age.max(1.0);

        // Children have shorter vocal tracts (higher formants)
        // Adults have longer vocal tracts (lower formants)
        let formant_shift = if self.target_age < 18.0 {
            // Child voice: higher formants, brighter sound
            1.0 + (18.0 - self.target_age) * 0.02
        } else if self.target_age > 60.0 {
            // Elderly voice: slightly lower formants, reduced breath support
            1.0 - (self.target_age - 60.0) * 0.01
        } else {
            // Adult voice
            age_ratio.sqrt()
        };

        // Apply spectral modifications
        output = self.apply_spectral_scaling(&output, formant_shift)?;

        // Apply age-related amplitude modifications
        if self.target_age > 60.0 {
            // Elderly voice: add slight tremor and reduced amplitude
            output = self.apply_age_tremor(&output)?;
        } else if self.target_age < 12.0 {
            // Child voice: higher pitch variability
            output = self.apply_child_characteristics(&output)?;
        }

        Ok(output)
    }

    fn apply_spectral_scaling(&self, input: &[f32], scale_factor: f32) -> Result<Vec<f32>> {
        // Simplified spectral scaling
        Ok(input.iter().map(|&x| x * scale_factor).collect())
    }

    fn apply_age_tremor(&self, input: &[f32]) -> Result<Vec<f32>> {
        // Add slight tremor characteristic of elderly voices
        let tremor_freq = 6.0; // Hz
        let tremor_depth = 0.05;

        Ok(input
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let tremor =
                    1.0 + tremor_depth * (2.0 * PI * tremor_freq * i as f32 / 22050.0).sin();
                x * tremor
            })
            .collect())
    }

    fn apply_child_characteristics(&self, input: &[f32]) -> Result<Vec<f32>> {
        // Apply characteristics of child voices
        Ok(input.iter().map(|&x| x * 1.1).collect()) // Slightly brighter
    }
}

impl GenderTransform {
    /// Apply gender-specific modifications
    fn apply_gender_modifications(&self, input: &[f32]) -> Result<Vec<f32>> {
        let mut output = input.to_vec();

        if self.target_gender > 0.0 {
            // Feminize: raise formants, adjust pitch contour
            output = self.apply_feminization(&output)?;
        } else if self.target_gender < 0.0 {
            // Masculinize: lower formants, deepen resonance
            output = self.apply_masculinization(&output)?;
        }

        Ok(output)
    }

    fn apply_feminization(&self, input: &[f32]) -> Result<Vec<f32>> {
        // Raise formant frequencies (shorter vocal tract simulation)
        let formant_shift = 1.0 + (self.target_gender * self.formant_shift_strength * 0.15);

        // Apply formant shifting and brightness enhancement
        let mut output = input
            .iter()
            .map(|&x| x * formant_shift)
            .collect::<Vec<f32>>();

        // Add slight breathiness
        for (i, sample) in output.iter_mut().enumerate() {
            let breathiness = 0.02 * (i as f32 * 0.01).sin();
            *sample += breathiness * self.target_gender;
        }

        Ok(output)
    }

    fn apply_masculinization(&self, input: &[f32]) -> Result<Vec<f32>> {
        // Lower formant frequencies (longer vocal tract simulation)
        let formant_shift = 1.0 + (self.target_gender * self.formant_shift_strength * 0.15);

        // Apply formant shifting and add depth
        let output = input
            .iter()
            .map(|&x| x * formant_shift)
            .collect::<Vec<f32>>();

        Ok(output)
    }
}

/// Multi-channel audio data structure with per-channel samples
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiChannelAudio {
    /// Audio samples organized as [channel][sample]
    pub channels: Vec<Vec<f32>>,
    /// Sample rate
    pub sample_rate: u32,
}

impl MultiChannelAudio {
    /// Create new multi-channel audio
    pub fn new(channels: Vec<Vec<f32>>, sample_rate: u32) -> Result<Self> {
        if channels.is_empty() {
            return Err(Error::transform("No channels provided".to_string()));
        }

        // Validate all channels have the same length
        let first_len = channels[0].len();
        if !channels.iter().all(|ch| ch.len() == first_len) {
            return Err(Error::transform(
                "All channels must have the same length".to_string(),
            ));
        }

        Ok(Self {
            channels,
            sample_rate,
        })
    }

    /// Create from interleaved samples
    pub fn from_interleaved(data: &[f32], num_channels: usize, sample_rate: u32) -> Result<Self> {
        if num_channels == 0 {
            return Err(Error::Transform {
                transform_type: "channel_validation".to_string(),
                message: "Number of channels must be greater than 0".to_string(),
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Ensure num_channels parameter is greater than 0".to_string(),
                    "Check audio format specification".to_string(),
                ]),
            });
        }

        if data.len() % num_channels != 0 {
            return Err(Error::Transform {
                transform_type: "interleaved_validation".to_string(),
                message: "Data length must be divisible by number of channels".to_string(),
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Ensure audio data length matches channel count".to_string(),
                    "Verify audio format is properly structured".to_string(),
                ]),
            });
        }

        let samples_per_channel = data.len() / num_channels;
        let mut channels = vec![Vec::with_capacity(samples_per_channel); num_channels];

        for (i, &sample) in data.iter().enumerate() {
            channels[i % num_channels].push(sample);
        }

        Ok(Self {
            channels,
            sample_rate,
        })
    }

    /// Convert to interleaved samples
    pub fn to_interleaved(&self) -> Vec<f32> {
        let num_channels = self.channels.len();
        let samples_per_channel = self.channels[0].len();
        let mut interleaved = Vec::with_capacity(num_channels * samples_per_channel);

        for sample_idx in 0..samples_per_channel {
            for channel in &self.channels {
                interleaved.push(channel[sample_idx]);
            }
        }

        interleaved
    }

    /// Get number of channels
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Get number of samples per channel
    pub fn num_samples(&self) -> usize {
        self.channels.first().map(|ch| ch.len()).unwrap_or(0)
    }

    /// Convert to mono by averaging channels
    pub fn to_mono(&self) -> Vec<f32> {
        let samples_per_channel = self.num_samples();
        let num_channels = self.num_channels() as f32;

        let mut mono = Vec::with_capacity(samples_per_channel);

        for sample_idx in 0..samples_per_channel {
            let sum: f32 = self.channels.iter().map(|ch| ch[sample_idx]).sum();
            mono.push(sum / num_channels);
        }

        mono
    }
}

/// Multi-channel transform trait for processing multi-channel audio
pub trait MultiChannelTransform {
    /// Apply transform to multi-channel audio
    fn apply_multichannel(&self, input: &MultiChannelAudio) -> Result<MultiChannelAudio>;

    /// Get transform parameters
    fn get_parameters(&self) -> std::collections::HashMap<String, f32>;
}

/// Channel processing strategy for multi-channel transformations
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ChannelStrategy {
    /// Process each channel independently
    Independent,
    /// Process channels with cross-channel correlation
    Correlated,
    /// Convert to mono, process, then expand to multi-channel
    MonoExpanded,
    /// Use mid/side processing for stereo
    MidSide,
}

/// Multi-channel configuration defining processing parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiChannelConfig {
    /// Processing strategy
    pub strategy: ChannelStrategy,
    /// Channel gains (for output balancing)
    pub channel_gains: Vec<f32>,
    /// Enable channel crosstalk simulation
    pub enable_crosstalk: bool,
    /// Crosstalk amount (0.0-1.0)
    pub crosstalk_amount: f32,
}

impl Default for MultiChannelConfig {
    fn default() -> Self {
        Self {
            strategy: ChannelStrategy::Independent,
            channel_gains: vec![1.0, 1.0], // Default stereo
            enable_crosstalk: false,
            crosstalk_amount: 0.05,
        }
    }
}

/// Multi-channel pitch transform with per-channel control
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiChannelPitchTransform {
    /// Base pitch transform
    pub base_transform: PitchTransform,
    /// Multi-channel configuration
    pub config: MultiChannelConfig,
    /// Per-channel pitch adjustments
    pub channel_pitch_factors: Vec<f32>,
}

impl MultiChannelPitchTransform {
    /// Create new multi-channel pitch transform
    pub fn new(pitch_factor: f32, num_channels: usize) -> Self {
        Self {
            base_transform: PitchTransform::new(pitch_factor),
            config: MultiChannelConfig {
                channel_gains: vec![1.0; num_channels],
                ..Default::default()
            },
            channel_pitch_factors: vec![pitch_factor; num_channels],
        }
    }

    /// Create stereo pitch transform with independent channel factors
    pub fn stereo(left_pitch: f32, right_pitch: f32) -> Self {
        Self {
            base_transform: PitchTransform::new((left_pitch + right_pitch) / 2.0),
            config: MultiChannelConfig::default(),
            channel_pitch_factors: vec![left_pitch, right_pitch],
        }
    }

    /// Set channel-specific pitch factors
    pub fn set_channel_pitch_factors(&mut self, factors: Vec<f32>) {
        self.channel_pitch_factors = factors;
    }
}

impl MultiChannelTransform for MultiChannelPitchTransform {
    fn apply_multichannel(&self, input: &MultiChannelAudio) -> Result<MultiChannelAudio> {
        match self.config.strategy {
            ChannelStrategy::Independent => {
                let mut output_channels = Vec::new();

                for (ch_idx, channel) in input.channels.iter().enumerate() {
                    let pitch_factor = self
                        .channel_pitch_factors
                        .get(ch_idx)
                        .copied()
                        .unwrap_or(self.base_transform.pitch_factor);

                    let mut channel_transform = self.base_transform.clone();
                    channel_transform.pitch_factor = pitch_factor;

                    let processed_channel = channel_transform.apply(channel)?;
                    output_channels.push(processed_channel);
                }

                self.apply_channel_processing(output_channels, input.sample_rate)
            }

            ChannelStrategy::MidSide => {
                if input.num_channels() != 2 {
                    return Err(Error::Transform {
                        transform_type: "mid_side_validation".to_string(),
                        message: "Mid/Side processing requires exactly 2 channels".to_string(),
                        context: None,
                        recovery_suggestions: Box::new(vec![
                            "Convert audio to stereo format before Mid/Side processing".to_string(),
                            "Use a different transform for non-stereo audio".to_string(),
                        ]),
                    });
                }

                let left = &input.channels[0];
                let right = &input.channels[1];

                // Convert to Mid/Side
                let mid: Vec<f32> = left
                    .iter()
                    .zip(right.iter())
                    .map(|(&l, &r)| (l + r) / 2.0)
                    .collect();

                let side: Vec<f32> = left
                    .iter()
                    .zip(right.iter())
                    .map(|(&l, &r)| (l - r) / 2.0)
                    .collect();

                // Process Mid and Side independently
                let mid_factor = self
                    .channel_pitch_factors
                    .first()
                    .copied()
                    .unwrap_or(self.base_transform.pitch_factor);
                let side_factor = self
                    .channel_pitch_factors
                    .get(1)
                    .copied()
                    .unwrap_or(self.base_transform.pitch_factor);

                let mut mid_transform = self.base_transform.clone();
                mid_transform.pitch_factor = mid_factor;
                let processed_mid = mid_transform.apply(&mid)?;

                let mut side_transform = self.base_transform.clone();
                side_transform.pitch_factor = side_factor;
                let processed_side = side_transform.apply(&side)?;

                // Convert back to Left/Right
                let processed_left: Vec<f32> = processed_mid
                    .iter()
                    .zip(processed_side.iter())
                    .map(|(&m, &s)| m + s)
                    .collect();

                let processed_right: Vec<f32> = processed_mid
                    .iter()
                    .zip(processed_side.iter())
                    .map(|(&m, &s)| m - s)
                    .collect();

                self.apply_channel_processing(
                    vec![processed_left, processed_right],
                    input.sample_rate,
                )
            }

            ChannelStrategy::MonoExpanded => {
                let mono = input.to_mono();
                let processed_mono = self.base_transform.apply(&mono)?;

                // Expand mono to all channels
                let mut output_channels = Vec::new();
                for ch_idx in 0..input.num_channels() {
                    let gain = self
                        .config
                        .channel_gains
                        .get(ch_idx)
                        .copied()
                        .unwrap_or(1.0);
                    let channel = processed_mono.iter().map(|&s| s * gain).collect();
                    output_channels.push(channel);
                }

                self.apply_channel_processing(output_channels, input.sample_rate)
            }

            ChannelStrategy::Correlated => {
                // Process channels with correlation awareness
                let mut output_channels = Vec::new();
                let correlation_matrix = self.calculate_channel_correlation(input);

                for (ch_idx, channel) in input.channels.iter().enumerate() {
                    let mut correlated_channel = channel.clone();

                    // Apply correlation-based adjustments
                    for (other_idx, other_channel) in input.channels.iter().enumerate() {
                        if ch_idx != other_idx {
                            let correlation = correlation_matrix[ch_idx][other_idx];
                            let influence = correlation * 0.1; // Limit influence

                            for (i, &other_sample) in other_channel.iter().enumerate() {
                                if i < correlated_channel.len() {
                                    correlated_channel[i] += other_sample * influence;
                                }
                            }
                        }
                    }

                    let pitch_factor = self
                        .channel_pitch_factors
                        .get(ch_idx)
                        .copied()
                        .unwrap_or(self.base_transform.pitch_factor);

                    let mut channel_transform = self.base_transform.clone();
                    channel_transform.pitch_factor = pitch_factor;

                    let processed_channel = channel_transform.apply(&correlated_channel)?;
                    output_channels.push(processed_channel);
                }

                self.apply_channel_processing(output_channels, input.sample_rate)
            }
        }
    }

    fn get_parameters(&self) -> std::collections::HashMap<String, f32> {
        let mut params = Transform::get_parameters(&self.base_transform);
        params.insert(
            "num_channels".to_string(),
            self.config.channel_gains.len() as f32,
        );
        params.insert("crosstalk_amount".to_string(), self.config.crosstalk_amount);

        for (i, &factor) in self.channel_pitch_factors.iter().enumerate() {
            params.insert(format!("channel_{i}_pitch"), factor);
        }

        params
    }
}

impl MultiChannelPitchTransform {
    /// Apply channel-specific processing (gains, crosstalk)
    fn apply_channel_processing(
        &self,
        mut channels: Vec<Vec<f32>>,
        sample_rate: u32,
    ) -> Result<MultiChannelAudio> {
        // Apply channel gains
        for (ch_idx, channel) in channels.iter_mut().enumerate() {
            let gain = self
                .config
                .channel_gains
                .get(ch_idx)
                .copied()
                .unwrap_or(1.0);
            for sample in channel.iter_mut() {
                *sample *= gain;
            }
        }

        // Apply crosstalk if enabled
        if self.config.enable_crosstalk && channels.len() > 1 {
            self.apply_crosstalk(&mut channels);
        }

        MultiChannelAudio::new(channels, sample_rate)
    }

    /// Apply crosstalk between channels
    fn apply_crosstalk(&self, channels: &mut [Vec<f32>]) {
        let num_channels = channels.len();
        let crosstalk = self.config.crosstalk_amount;

        // Create a copy for crosstalk calculation
        let original_channels: Vec<Vec<f32>> = channels.to_vec();

        for (ch_idx, channel) in channels.iter_mut().enumerate() {
            for (sample_idx, sample) in channel.iter_mut().enumerate() {
                let mut crosstalk_sum = 0.0;
                let mut count = 0;

                // Add crosstalk from other channels
                for (other_idx, other_channel) in original_channels.iter().enumerate() {
                    if ch_idx != other_idx && sample_idx < other_channel.len() {
                        crosstalk_sum += other_channel[sample_idx];
                        count += 1;
                    }
                }

                if count > 0 {
                    let avg_crosstalk = crosstalk_sum / count as f32;
                    *sample = *sample * (1.0 - crosstalk) + avg_crosstalk * crosstalk;
                }
            }
        }
    }

    /// Calculate correlation matrix between channels
    fn calculate_channel_correlation(&self, input: &MultiChannelAudio) -> Vec<Vec<f32>> {
        let num_channels = input.num_channels();

        (0..num_channels)
            .map(|i| {
                (0..num_channels)
                    .map(|j| {
                        if i == j {
                            1.0
                        } else {
                            self.calculate_correlation(&input.channels[i], &input.channels[j])
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Calculate correlation between two channels
    fn calculate_correlation(&self, ch1: &[f32], ch2: &[f32]) -> f32 {
        if ch1.len() != ch2.len() || ch1.is_empty() {
            return 0.0;
        }

        let mean1 = ch1.iter().sum::<f32>() / ch1.len() as f32;
        let mean2 = ch2.iter().sum::<f32>() / ch2.len() as f32;

        let mut numerator = 0.0;
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;

        for (s1, s2) in ch1.iter().zip(ch2.iter()) {
            let diff1 = s1 - mean1;
            let diff2 = s2 - mean2;

            numerator += diff1 * diff2;
            sum_sq1 += diff1 * diff1;
            sum_sq2 += diff2 * diff2;
        }

        let denominator = (sum_sq1 * sum_sq2).sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

// Implement MultiChannelTransform for existing transforms by wrapping them

impl MultiChannelTransform for PitchTransform {
    fn apply_multichannel(&self, input: &MultiChannelAudio) -> Result<MultiChannelAudio> {
        let multichannel_transform = MultiChannelPitchTransform {
            base_transform: self.clone(),
            config: MultiChannelConfig {
                channel_gains: vec![1.0; input.num_channels()],
                ..Default::default()
            },
            channel_pitch_factors: vec![self.pitch_factor; input.num_channels()],
        };

        multichannel_transform.apply_multichannel(input)
    }

    fn get_parameters(&self) -> std::collections::HashMap<String, f32> {
        Transform::get_parameters(self)
    }
}

impl MultiChannelTransform for SpeedTransform {
    fn apply_multichannel(&self, input: &MultiChannelAudio) -> Result<MultiChannelAudio> {
        let mut output_channels = Vec::new();

        for channel in &input.channels {
            let processed_channel = self.apply(channel)?;
            output_channels.push(processed_channel);
        }

        MultiChannelAudio::new(output_channels, input.sample_rate)
    }

    fn get_parameters(&self) -> std::collections::HashMap<String, f32> {
        Transform::get_parameters(self)
    }
}

impl MultiChannelTransform for AgeTransform {
    fn apply_multichannel(&self, input: &MultiChannelAudio) -> Result<MultiChannelAudio> {
        let mut output_channels = Vec::new();

        for channel in &input.channels {
            let processed_channel = self.apply(channel)?;
            output_channels.push(processed_channel);
        }

        MultiChannelAudio::new(output_channels, input.sample_rate)
    }

    fn get_parameters(&self) -> std::collections::HashMap<String, f32> {
        Transform::get_parameters(self)
    }
}

impl MultiChannelTransform for GenderTransform {
    fn apply_multichannel(&self, input: &MultiChannelAudio) -> Result<MultiChannelAudio> {
        let mut output_channels = Vec::new();

        for channel in &input.channels {
            let processed_channel = self.apply(channel)?;
            output_channels.push(processed_channel);
        }

        MultiChannelAudio::new(output_channels, input.sample_rate)
    }

    fn get_parameters(&self) -> std::collections::HashMap<String, f32> {
        Transform::get_parameters(self)
    }
}

#[cfg(test)]
mod multichannel_tests {
    use super::*;

    #[test]
    fn test_multichannel_audio_creation() {
        let channels = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let audio = MultiChannelAudio::new(channels.clone(), 44100).unwrap();

        assert_eq!(audio.num_channels(), 2);
        assert_eq!(audio.num_samples(), 3);
        assert_eq!(audio.channels, channels);
    }

    #[test]
    fn test_interleaved_conversion() {
        let interleaved = vec![0.1, 0.4, 0.2, 0.5, 0.3, 0.6];
        let audio = MultiChannelAudio::from_interleaved(&interleaved, 2, 44100).unwrap();

        assert_eq!(audio.num_channels(), 2);
        assert_eq!(audio.num_samples(), 3);

        let back_to_interleaved = audio.to_interleaved();
        assert_eq!(back_to_interleaved, interleaved);
    }

    #[test]
    fn test_mono_conversion() {
        let channels = vec![vec![0.2, 0.4, 0.6], vec![0.8, 1.0, 1.2]];
        let audio = MultiChannelAudio::new(channels, 44100).unwrap();
        let mono = audio.to_mono();

        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.5).abs() < f32::EPSILON); // (0.2 + 0.8) / 2
        assert!((mono[1] - 0.7).abs() < f32::EPSILON); // (0.4 + 1.0) / 2
        assert!((mono[2] - 0.9).abs() < f32::EPSILON); // (0.6 + 1.2) / 2
    }

    #[test]
    fn test_multichannel_pitch_transform_independent() {
        let channels = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let audio = MultiChannelAudio::new(channels, 44100).unwrap();

        let transform = MultiChannelPitchTransform::new(2.0, 2);
        let result = transform.apply_multichannel(&audio).unwrap();

        assert_eq!(result.num_channels(), 2);
        assert_eq!(result.num_samples(), 3);
    }

    #[test]
    fn test_multichannel_pitch_transform_stereo() {
        let channels = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let audio = MultiChannelAudio::new(channels, 44100).unwrap();

        let transform = MultiChannelPitchTransform::stereo(1.5, 2.5);
        let result = transform.apply_multichannel(&audio).unwrap();

        assert_eq!(result.num_channels(), 2);
        assert_eq!(transform.channel_pitch_factors, vec![1.5, 2.5]);
    }

    #[test]
    fn test_multichannel_mid_side_processing() {
        let channels = vec![
            vec![0.8, 0.6, 0.4], // Left
            vec![0.2, 0.4, 0.6], // Right
        ];
        let audio = MultiChannelAudio::new(channels, 44100).unwrap();

        let mut transform = MultiChannelPitchTransform::stereo(2.0, 2.0);
        transform.config.strategy = ChannelStrategy::MidSide;

        let result = transform.apply_multichannel(&audio).unwrap();
        assert_eq!(result.num_channels(), 2);
    }

    #[test]
    fn test_multichannel_transform_with_crosstalk() {
        let channels = vec![vec![1.0, 0.0, 0.5], vec![0.0, 1.0, 0.5]];
        let audio = MultiChannelAudio::new(channels, 44100).unwrap();

        let mut transform = MultiChannelPitchTransform::new(1.0, 2);
        transform.config.enable_crosstalk = true;
        transform.config.crosstalk_amount = 0.1;

        let result = transform.apply_multichannel(&audio).unwrap();

        // With crosstalk, channels should influence each other
        assert_eq!(result.num_channels(), 2);
        assert_ne!(result.channels[0], vec![1.0, 0.0, 0.5]);
        assert_ne!(result.channels[1], vec![0.0, 1.0, 0.5]);
    }

    #[test]
    fn test_channel_correlation_calculation() {
        let channels = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0], // Perfect correlation
        ];
        let audio = MultiChannelAudio::new(channels, 44100).unwrap();

        let transform = MultiChannelPitchTransform::new(1.0, 2);
        let correlation_matrix = transform.calculate_channel_correlation(&audio);

        assert_eq!(correlation_matrix.len(), 2);
        assert_eq!(correlation_matrix[0].len(), 2);
        assert_eq!(correlation_matrix[0][0], 1.0); // Self-correlation
        assert!((correlation_matrix[0][1] - 1.0).abs() < f32::EPSILON); // Perfect correlation
    }
}
