//! Type-specific conversion implementations

use crate::{
    transforms::{
        AgeTransform, GenderTransform, PitchTransform, SpeedTransform, Transform, VoiceMorpher,
    },
    types::ConversionType,
    Error, Result,
};
use tracing::debug;

use super::{converter::VoiceConverter, types::AudioFeatures};

impl VoiceConverter {
    /// Convert age characteristics
    pub(super) async fn convert_age(
        &self,
        audio: &[f32],
        target: &crate::types::ConversionTarget,
    ) -> Result<Vec<f32>> {
        debug!("Performing age conversion");

        let source_age = target
            .characteristics
            .age_group
            .map(|age| match age {
                crate::types::AgeGroup::Child => 8.0,
                crate::types::AgeGroup::Teen => 16.0,
                crate::types::AgeGroup::YoungAdult => 25.0,
                crate::types::AgeGroup::Adult => 35.0,
                crate::types::AgeGroup::MiddleAged => 45.0,
                crate::types::AgeGroup::Senior => 65.0,
                crate::types::AgeGroup::Unknown => 30.0,
            })
            .unwrap_or(30.0);

        let target_age = 25.0; // Default young adult
        let transform = AgeTransform::new(source_age, target_age);

        // Apply age transformation with additional acoustic modifications
        let mut result = transform.apply(audio)?;

        // Adjust formants based on age
        result = self.adjust_formants(&result, source_age, target_age)?;

        // Adjust vocal tract length simulation
        result = self.adjust_vocal_tract_length(&result, source_age / target_age)?;

        Ok(result)
    }

    /// Convert gender characteristics
    pub(super) async fn convert_gender(
        &self,
        audio: &[f32],
        target: &crate::types::ConversionTarget,
    ) -> Result<Vec<f32>> {
        debug!("Performing gender conversion");

        let target_gender_value = match target.characteristics.gender {
            Some(crate::types::Gender::Male) => -1.0,
            Some(crate::types::Gender::Female) => 1.0,
            Some(crate::types::Gender::Other) => 0.0,
            _ => 0.0,
        };

        let transform = GenderTransform::new(target_gender_value);
        let mut result = transform.apply(audio)?;

        // Apply formant shifting for gender conversion
        let formant_shift = target.characteristics.spectral.formant_shift;
        result = self.shift_formants(&result, formant_shift)?;

        // Adjust fundamental frequency
        let f0_shift = target.characteristics.pitch.mean_f0 / 150.0; // Normalize to default
        result = self.shift_f0(&result, f0_shift)?;

        Ok(result)
    }

    /// Convert pitch characteristics
    pub(super) async fn convert_pitch(
        &self,
        audio: &[f32],
        target: &crate::types::ConversionTarget,
    ) -> Result<Vec<f32>> {
        debug!("Performing pitch conversion");

        let pitch_factor = target.characteristics.pitch.mean_f0 / 150.0; // Normalize to 150 Hz baseline
        let transform = PitchTransform::new(pitch_factor);

        transform.apply(audio)
    }

    /// Convert speed characteristics
    pub(super) async fn convert_speed(
        &self,
        audio: &[f32],
        target: &crate::types::ConversionTarget,
    ) -> Result<Vec<f32>> {
        debug!("Performing speed conversion");

        let speed_factor = target.characteristics.timing.speaking_rate;
        let transform = SpeedTransform::new(speed_factor);

        transform.apply(audio)
    }

    /// Convert using voice morphing
    pub(super) async fn convert_morph(
        &self,
        audio: &[f32],
        target: &crate::types::ConversionTarget,
    ) -> Result<Vec<f32>> {
        debug!("Performing voice morphing");

        if target.reference_samples.is_empty() {
            return Err(Error::processing(
                "Voice morphing requires reference samples".to_string(),
            ));
        }

        // Extract features from reference samples
        let mut reference_audio = Vec::new();
        for sample in &target.reference_samples {
            reference_audio.push(sample.audio.clone());
        }

        // Create equal weight blending by default
        let blend_weights = vec![1.0 / reference_audio.len() as f32; reference_audio.len()];
        let voice_ids: Vec<String> = (0..reference_audio.len())
            .map(|i| format!("ref_{i}"))
            .collect();

        let morpher = VoiceMorpher::new(voice_ids, blend_weights);
        let mut all_inputs = vec![audio.to_vec()];
        all_inputs.extend(reference_audio);

        morpher.morph(&all_inputs)
    }

    /// Convert emotional characteristics
    pub(super) async fn convert_emotion(
        &self,
        audio: &[f32],
        target: &crate::types::ConversionTarget,
    ) -> Result<Vec<f32>> {
        debug!("Performing emotional conversion");

        // Extract emotional parameters from custom_params
        let valence = target
            .characteristics
            .custom_params
            .get("valence")
            .copied()
            .unwrap_or(0.0);
        let arousal = target
            .characteristics
            .custom_params
            .get("arousal")
            .copied()
            .unwrap_or(0.0);

        let mut result = audio.to_vec();

        // Adjust pitch for emotional content (higher arousal = higher pitch variation)
        let pitch_variation = 1.0 + (arousal * 0.2);
        result = self.modulate_pitch_contour(&result, pitch_variation)?;

        // Adjust timing for emotional content (higher arousal = faster speech)
        let timing_factor = 1.0 + (arousal * 0.1);
        if timing_factor != 1.0 {
            let speed_transform = SpeedTransform::new(timing_factor);
            result = speed_transform.apply(&result)?;
        }

        // Adjust spectral characteristics for valence
        if valence != 0.0 {
            result = self.adjust_spectral_tilt(&result, valence * 0.1)?;
        }

        Ok(result)
    }

    /// Convert using zero-shot learning to unseen target voices
    pub(super) async fn convert_zero_shot(
        &self,
        audio: &[f32],
        target: &crate::types::ConversionTarget,
        features: Option<&AudioFeatures>,
    ) -> Result<Vec<f32>> {
        debug!("Performing zero-shot conversion to unseen target voice");

        // Zero-shot conversion combines reference samples with learned representations
        if !target.reference_samples.is_empty() {
            // Use reference samples for few-shot learning approach
            self.convert_using_reference_samples(audio, target, features)
                .await
        } else {
            // Fall back to characteristic-based conversion for zero-shot scenarios
            self.convert_using_characteristics(audio, target, features)
                .await
        }
    }

    /// Convert using custom transformation
    pub(super) async fn convert_custom(
        &self,
        audio: &[f32],
        name: &str,
        _target: &crate::types::ConversionTarget,
    ) -> Result<Vec<f32>> {
        debug!("Performing custom conversion: {}", name);

        // Load custom model or transformation
        if let Some(model) = self
            .models
            .read()
            .await
            .get(&ConversionType::Custom(name.to_string()))
        {
            let input_tensor = self.audio_to_tensor(audio)?;
            let output_tensor = model.process_tensor(&input_tensor).await?;
            self.tensor_to_audio(&output_tensor)
        } else {
            tracing::warn!(
                "Custom conversion '{}' not found, applying identity transform",
                name
            );
            Ok(audio.to_vec())
        }
    }
}
