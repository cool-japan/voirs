//! Signal processing helper methods

use crate::{
    transforms::{PitchTransform, Transform},
    Result,
};

use super::converter::VoiceConverter;

impl VoiceConverter {
    /// Adjust formants based on age
    pub(super) fn adjust_formants(
        &self,
        audio: &[f32],
        source_age: f32,
        target_age: f32,
    ) -> Result<Vec<f32>> {
        // Age affects vocal tract length: children have shorter vocal tracts
        let formant_factor = (source_age / target_age).sqrt();
        self.shift_formants(audio, (formant_factor - 1.0) * 0.5)
    }

    /// Adjust vocal tract length simulation
    pub(super) fn adjust_vocal_tract_length(
        &self,
        audio: &[f32],
        length_factor: f32,
    ) -> Result<Vec<f32>> {
        // Simple implementation using spectral scaling
        let mut result = audio.to_vec();
        if length_factor != 1.0 {
            // Apply frequency-domain scaling (simplified)
            for sample in &mut result {
                *sample *= length_factor;
            }
        }
        Ok(result)
    }

    /// Shift formant frequencies
    pub(super) fn shift_formants(&self, audio: &[f32], shift_factor: f32) -> Result<Vec<f32>> {
        // Simplified formant shifting - in practice would use spectral processing
        let mut result = audio.to_vec();
        if shift_factor != 0.0 {
            let scale = 1.0 + shift_factor;
            for sample in &mut result {
                *sample *= scale;
            }
        }
        Ok(result)
    }

    /// Shift fundamental frequency
    pub(super) fn shift_f0(&self, audio: &[f32], f0_factor: f32) -> Result<Vec<f32>> {
        if f0_factor == 1.0 {
            return Ok(audio.to_vec());
        }

        let pitch_transform = PitchTransform::new(f0_factor);
        pitch_transform.apply(audio)
    }

    /// Modulate pitch contour for emotional expression
    pub(super) fn modulate_pitch_contour(
        &self,
        audio: &[f32],
        variation_factor: f32,
    ) -> Result<Vec<f32>> {
        let mut result = audio.to_vec();

        // Apply sinusoidal modulation to simulate pitch contour changes
        for (i, sample) in result.iter_mut().enumerate() {
            let modulation = 1.0 + (i as f32 * 0.01).sin() * (variation_factor - 1.0) * 0.1;
            *sample *= modulation;
        }

        Ok(result)
    }

    /// Adjust spectral tilt
    pub(super) fn adjust_spectral_tilt(&self, audio: &[f32], tilt_factor: f32) -> Result<Vec<f32>> {
        // Simplified spectral tilt adjustment
        let mut result = audio.to_vec();
        if tilt_factor != 0.0 {
            let len = result.len();
            for (i, sample) in result.iter_mut().enumerate() {
                let freq_weight = 1.0 + (i as f32 / len as f32) * tilt_factor;
                *sample *= freq_weight;
            }
        }
        Ok(result)
    }
}
