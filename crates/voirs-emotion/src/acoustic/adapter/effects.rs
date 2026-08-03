//! Effect processing methods for acoustic emotion adaptation
//!
//! This module contains real (input-audio-transforming, not from-scratch
//! generating) effect processing methods:
//! - Vocoder effects (pitch shift, formant shift, breathiness, roughness)
//! - Speaker characteristic adjustments
//!
//! These all operate on *existing* audio passed in by the caller (see
//! [`super::synthesis::AcousticEmotionAdapter::vocode_with_emotion`] and
//! [`super::synthesis::AcousticEmotionAdapter::transfer_emotion_between_speakers`]).
//! Nothing in this module synthesizes audio from nothing (that fabricated
//! from-scratch tone generator was removed - see
//! [`super::synthesis::AcousticEmotionAdapter::synthesize_with_emotion`] for
//! why real text-to-speech synthesis fails closed instead).

use crate::{types::EmotionParameters, Result};

impl super::core::AcousticEmotionAdapter {
    /// Apply basic vocoder-style effects (fallback implementation)
    pub(super) fn apply_basic_vocoder_effects(
        &self,
        audio: &mut [f32],
        emotion_params: &EmotionParameters,
    ) -> Result<()> {
        // Apply pitch shifting
        if (emotion_params.pitch_shift - 1.0).abs() > 0.01 {
            self.apply_pitch_shift_effect(audio, emotion_params.pitch_shift)?;
        }

        // Apply formant shifting for emotion
        let formant_shift = 1.0 + emotion_params.emotion_vector.dimensions.arousal * 0.1;
        if (formant_shift - 1.0f32).abs() > 0.01 {
            self.apply_formant_shift_effect(audio, formant_shift)?;
        }

        // Apply voice quality effects
        if emotion_params.breathiness > 0.1 {
            self.apply_breathiness_effect(audio, emotion_params.breathiness)?;
        }

        if emotion_params.roughness > 0.1 {
            self.apply_roughness_effect(audio, emotion_params.roughness)?;
        }

        Ok(())
    }

    /// Apply basic pitch shifting effect
    pub(super) fn apply_pitch_shift_effect(
        &self,
        audio: &mut [f32],
        pitch_shift: f32,
    ) -> Result<()> {
        // Simple time-domain pitch shifting (not ideal but works as fallback)
        if (pitch_shift - 1.0).abs() < 0.01 {
            return Ok(());
        }

        let len = audio.len();
        let mut shifted_audio = vec![0.0; len];

        #[allow(clippy::needless_range_loop)]
        for i in 0..len {
            let source_index = (i as f32 / pitch_shift) as usize;
            if source_index < len {
                shifted_audio[i] = audio[source_index];
            }
        }

        audio.copy_from_slice(&shifted_audio);
        Ok(())
    }

    /// Apply basic formant shifting effect
    pub(super) fn apply_formant_shift_effect(
        &self,
        audio: &mut [f32],
        formant_shift: f32,
    ) -> Result<()> {
        // Apply a simple spectral shift approximation
        if (formant_shift - 1.0).abs() < 0.01 {
            return Ok(());
        }

        // This is a very basic approximation - real formant shifting requires complex DSP
        let shift_factor = formant_shift.clamp(0.5, 2.0);

        for sample in audio.iter_mut() {
            *sample *= shift_factor.sqrt(); // Basic amplitude compensation
        }

        Ok(())
    }

    /// Apply breathiness effect
    pub(super) fn apply_breathiness_effect(
        &self,
        audio: &mut [f32],
        breathiness: f32,
    ) -> Result<()> {
        if breathiness <= 0.0 {
            return Ok(());
        }

        let noise_level = breathiness * 0.1;

        for sample in audio.iter_mut() {
            let noise = (scirs2_core::random::random::<f32>() - 0.5) * noise_level;
            *sample = *sample * (1.0 - breathiness * 0.3) + noise;
        }

        Ok(())
    }

    /// Apply roughness effect
    pub(super) fn apply_roughness_effect(&self, audio: &mut [f32], roughness: f32) -> Result<()> {
        if roughness <= 0.0 {
            return Ok(());
        }

        // Add harmonic distortion for roughness
        for sample in audio.iter_mut() {
            if sample.abs() > 0.01 {
                let distorted = sample.signum() * (sample.abs().powf(1.0 - roughness * 0.3));
                *sample = *sample * (1.0 - roughness * 0.5) + distorted * roughness * 0.5;
            }
        }

        Ok(())
    }

    /// Apply basic speaker emotion transfer effects (fallback)
    pub(super) fn apply_speaker_emotion_transfer_effects(
        &self,
        audio: &mut [f32],
        emotion_params: &EmotionParameters,
    ) -> Result<()> {
        // Apply combined effects for emotion transfer
        self.apply_basic_vocoder_effects(audio, emotion_params)?;

        // Add speaker-specific emotion adaptations
        self.apply_speaker_characteristic_adjustments(audio, emotion_params)?;

        Ok(())
    }

    /// Apply speaker characteristic adjustments
    pub(super) fn apply_speaker_characteristic_adjustments(
        &self,
        audio: &mut [f32],
        emotion_params: &EmotionParameters,
    ) -> Result<()> {
        // Adjust formant characteristics based on emotion
        let formant_shift = 1.0 + emotion_params.emotion_vector.dimensions.dominance * 0.15;
        self.apply_formant_shift_effect(audio, formant_shift)?;

        // Adjust voice quality for emotion transfer
        if emotion_params.emotion_vector.dimensions.valence < -0.3 {
            // Add dampening for negative emotions
            for sample in audio.iter_mut() {
                *sample *= 0.9;
            }
        } else if emotion_params.emotion_vector.dimensions.valence > 0.3 {
            // Add brightness for positive emotions
            for sample in audio.iter_mut() {
                *sample = sample.tanh(); // Soft saturation for warmth
            }
        }

        Ok(())
    }
}
