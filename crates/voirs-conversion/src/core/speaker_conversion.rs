//! Speaker conversion implementation

use crate::{
    models::ConversionModel,
    transforms::{SpeedTransform, Transform},
    types::{ConversionType, VoiceCharacteristics},
    Result,
};
use tracing::{debug, warn};

use super::{converter::VoiceConverter, types::AudioFeatures};

impl VoiceConverter {
    /// Convert speaker characteristics
    pub(super) async fn convert_speaker(
        &self,
        audio: &[f32],
        target: &crate::types::ConversionTarget,
        features: Option<&AudioFeatures>,
    ) -> Result<Vec<f32>> {
        debug!(
            "Performing speaker conversion to target: {:?}",
            target.speaker_id
        );

        // Enhanced speaker conversion with multiple approaches
        let conversion_result = if let Some(speaker_id) = &target.speaker_id {
            // Named speaker conversion using learned embeddings
            self.convert_to_named_speaker(audio, speaker_id, target, features)
                .await?
        } else if !target.reference_samples.is_empty() {
            // Few-shot conversion using reference samples
            self.convert_using_reference_samples(audio, target, features)
                .await?
        } else {
            // Characteristic-based conversion
            self.convert_using_characteristics(audio, target, features)
                .await?
        };

        // Apply post-processing for speaker conversion
        self.apply_speaker_post_processing(&conversion_result, target)
            .await
    }

    /// Convert to a named speaker using learned embeddings
    pub(super) async fn convert_to_named_speaker(
        &self,
        audio: &[f32],
        speaker_id: &str,
        target: &crate::types::ConversionTarget,
        features: Option<&AudioFeatures>,
    ) -> Result<Vec<f32>> {
        debug!("Converting to named speaker: {}", speaker_id);

        // Try neural model first for named speaker conversion
        if let Some(model) = self.get_model(ConversionType::SpeakerConversion).await? {
            match self
                .neural_speaker_conversion(audio, speaker_id, &model, features)
                .await
            {
                Ok(result) => return Ok(result),
                Err(e) => {
                    warn!(
                        "Neural conversion failed for speaker {}: {}, falling back",
                        speaker_id, e
                    );
                }
            }
        }

        // Fallback to characteristic-based conversion
        self.convert_using_characteristics(audio, target, features)
            .await
    }

    /// Convert using reference samples (few-shot learning)
    pub(super) async fn convert_using_reference_samples(
        &self,
        audio: &[f32],
        target: &crate::types::ConversionTarget,
        features: Option<&AudioFeatures>,
    ) -> Result<Vec<f32>> {
        debug!(
            "Converting using {} reference samples",
            target.reference_samples.len()
        );

        // Extract target speaker characteristics from reference samples
        let target_characteristics = self
            .extract_speaker_characteristics_from_samples(&target.reference_samples)
            .await?;

        // Combine with provided characteristics
        let combined_characteristics = self.combine_characteristics(
            &target.characteristics,
            &target_characteristics,
            0.7, // Weight towards reference samples
        );

        // Apply conversion using combined characteristics
        let mut result = self
            .apply_advanced_speaker_transform(audio, &combined_characteristics, features)
            .await?;

        // Apply reference-guided fine-tuning
        result = self
            .apply_reference_guided_refinement(&result, &target.reference_samples, target.strength)
            .await?;

        Ok(result)
    }

    /// Convert using voice characteristics only
    pub(super) async fn convert_using_characteristics(
        &self,
        audio: &[f32],
        target: &crate::types::ConversionTarget,
        features: Option<&AudioFeatures>,
    ) -> Result<Vec<f32>> {
        debug!("Converting using voice characteristics");

        self.apply_advanced_speaker_transform(audio, &target.characteristics, features)
            .await
    }

    /// Neural speaker conversion using learned embeddings
    pub(super) async fn neural_speaker_conversion(
        &self,
        audio: &[f32],
        speaker_id: &str,
        model: &ConversionModel,
        features: Option<&AudioFeatures>,
    ) -> Result<Vec<f32>> {
        // Create input tensor with speaker embedding
        let mut input_tensor = self.audio_to_tensor(audio)?;

        // Add speaker embedding if available
        if let Some(speaker_embedding) = self.get_speaker_embedding(speaker_id).await? {
            input_tensor =
                self.combine_audio_and_speaker_embedding(input_tensor, speaker_embedding)?;
        }

        // Process through neural model
        let output_tensor = model.process_tensor(&input_tensor).await?;
        self.tensor_to_audio(&output_tensor)
    }

    /// Extract speaker characteristics from reference samples
    pub(super) async fn extract_speaker_characteristics_from_samples(
        &self,
        samples: &[crate::types::AudioSample],
    ) -> Result<VoiceCharacteristics> {
        if samples.is_empty() {
            return Ok(VoiceCharacteristics::default());
        }

        debug!("Extracting characteristics from {} samples", samples.len());

        let mut combined_characteristics = VoiceCharacteristics::default();
        let mut total_weight = 0.0;

        for sample in samples {
            let weight = sample.duration.min(10.0) / 10.0; // Weight by duration, max 10s
            let sample_chars = self
                .analyze_audio_characteristics(&sample.audio, sample.sample_rate)
                .await?;

            // Weighted combination
            combined_characteristics = self.combine_characteristics(
                &combined_characteristics,
                &sample_chars,
                weight / (total_weight + weight),
            );
            total_weight += weight;
        }

        Ok(combined_characteristics)
    }

    /// Analyze audio to extract voice characteristics
    pub(super) async fn analyze_audio_characteristics(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<VoiceCharacteristics> {
        // Extract features from audio
        let features = self
            .feature_extractor
            .extract_features(audio, sample_rate)
            .await?;

        let mut characteristics = VoiceCharacteristics::default();

        // Estimate fundamental frequency from features
        if !features.prosodic.is_empty() {
            characteristics.pitch.mean_f0 = features.prosodic[0].clamp(50.0, 500.0);
            if features.prosodic.len() > 1 {
                characteristics.pitch.range = features.prosodic[1].clamp(1.0, 48.0);
            }
        }

        // Estimate spectral characteristics
        if !features.spectral.is_empty() {
            characteristics.spectral.brightness = (features.spectral[0] - 0.5).clamp(-1.0, 1.0);
            if features.spectral.len() > 1 {
                characteristics.spectral.formant_shift =
                    (features.spectral[1] - 0.5).clamp(-0.5, 0.5);
            }
        }

        // Estimate voice quality from temporal features
        if !features.temporal.is_empty() {
            characteristics.quality.stability = features.temporal[0].clamp(0.0, 1.0);
            if features.temporal.len() > 1 {
                characteristics.quality.breathiness = features.temporal[1].clamp(0.0, 1.0);
            }
        }

        debug!(
            "Analyzed characteristics: F0={:.1}Hz, brightness={:.2}",
            characteristics.pitch.mean_f0, characteristics.spectral.brightness
        );

        Ok(characteristics)
    }

    /// Combine two sets of voice characteristics
    pub(super) fn combine_characteristics(
        &self,
        chars1: &VoiceCharacteristics,
        chars2: &VoiceCharacteristics,
        weight2: f32,
    ) -> VoiceCharacteristics {
        chars1.interpolate(chars2, weight2.clamp(0.0, 1.0))
    }

    /// Apply advanced speaker transformation with features
    pub(super) async fn apply_advanced_speaker_transform(
        &self,
        audio: &[f32],
        characteristics: &VoiceCharacteristics,
        features: Option<&AudioFeatures>,
    ) -> Result<Vec<f32>> {
        let mut result = audio.to_vec();

        // Apply transformations in optimal order for speaker conversion

        // 1. Fundamental frequency transformation
        let f0_ratio = characteristics.pitch.mean_f0 / 150.0; // Normalize to neutral
        if (f0_ratio - 1.0).abs() > 0.05 {
            // Only if significant change
            result = self
                .apply_advanced_pitch_shift(&result, f0_ratio, features)
                .await?;
        }

        // 2. Formant transformation for vocal tract characteristics
        if characteristics.spectral.formant_shift.abs() > 0.01 {
            result = self
                .apply_formant_transformation(
                    &result,
                    characteristics.spectral.formant_shift,
                    characteristics.gender,
                )
                .await?;
        }

        // 3. Voice quality transformation
        result = self
            .apply_voice_quality_transformation(&result, &characteristics.quality)
            .await?;

        // 4. Spectral envelope modification
        if characteristics.spectral.brightness.abs() > 0.01 {
            result = self
                .apply_spectral_brightness(&result, characteristics.spectral.brightness)
                .await?;
        }

        // 5. Apply speaker-specific prosodic patterns
        result = self
            .apply_prosodic_transformation(&result, &characteristics.timing)
            .await?;

        Ok(result)
    }

    /// Apply advanced pitch shifting with better quality
    pub(super) async fn apply_advanced_pitch_shift(
        &self,
        audio: &[f32],
        ratio: f32,
        _features: Option<&AudioFeatures>,
    ) -> Result<Vec<f32>> {
        if (ratio - 1.0).abs() < f32::EPSILON {
            return Ok(audio.to_vec());
        }

        // Use the enhanced pitch transform from transforms module
        let pitch_transform = crate::transforms::PitchTransform::new(ratio);
        pitch_transform.apply(audio)
    }

    /// Apply formant transformation for vocal tract simulation
    pub(super) async fn apply_formant_transformation(
        &self,
        audio: &[f32],
        formant_shift: f32,
        gender: Option<crate::types::Gender>,
    ) -> Result<Vec<f32>> {
        let mut result = audio.to_vec();

        // Apply formant shifting based on gender and characteristics
        let formant_factor = match gender {
            Some(crate::types::Gender::Male) => 1.0 + formant_shift * 0.8,
            Some(crate::types::Gender::Female) => 1.0 + formant_shift * 1.2,
            _ => 1.0 + formant_shift,
        };

        // Simple formant shifting (in practice would use more sophisticated methods)
        if formant_factor != 1.0 {
            for sample in &mut result {
                *sample *= formant_factor;
            }
        }

        Ok(result)
    }

    /// Apply voice quality transformation
    pub(super) async fn apply_voice_quality_transformation(
        &self,
        audio: &[f32],
        quality: &crate::types::QualityCharacteristics,
    ) -> Result<Vec<f32>> {
        let mut result = audio.to_vec();

        // Apply breathiness
        if quality.breathiness > 0.1 {
            for (i, sample) in result.iter_mut().enumerate() {
                let noise = (i as f32 * 0.01).sin() * quality.breathiness * 0.05;
                *sample += noise;
            }
        }

        // Apply roughness
        if quality.roughness > 0.1 {
            for (i, sample) in result.iter_mut().enumerate() {
                let modulation = 1.0 + (i as f32 * 0.02).sin() * quality.roughness * 0.1;
                *sample *= modulation;
            }
        }

        // Apply stability (inverse of jitter)
        if quality.stability < 0.8 {
            let jitter_amount = 1.0 - quality.stability;
            for (i, sample) in result.iter_mut().enumerate() {
                let jitter = (i as f32 * 0.1).sin() * jitter_amount * 0.02;
                *sample *= 1.0 + jitter;
            }
        }

        Ok(result)
    }

    /// Apply spectral brightness adjustment
    pub(super) async fn apply_spectral_brightness(
        &self,
        audio: &[f32],
        brightness: f32,
    ) -> Result<Vec<f32>> {
        // Simple high-frequency emphasis/de-emphasis
        let mut result = audio.to_vec();

        if brightness.abs() > 0.01 {
            // Apply frequency-dependent scaling (simplified)
            let brightness_factor = 1.0 + brightness * 0.3;
            for sample in &mut result {
                *sample *= brightness_factor;
            }
        }

        Ok(result)
    }

    /// Apply prosodic transformation
    pub(super) async fn apply_prosodic_transformation(
        &self,
        audio: &[f32],
        timing: &crate::types::TimingCharacteristics,
    ) -> Result<Vec<f32>> {
        let mut result = audio.to_vec();

        // Apply speaking rate transformation
        if (timing.speaking_rate - 1.0).abs() > 0.05 {
            let speed_transform = SpeedTransform::new(timing.speaking_rate);
            result = speed_transform.apply(&result)?;
        }

        Ok(result)
    }

    /// Apply reference-guided refinement
    pub(super) async fn apply_reference_guided_refinement(
        &self,
        audio: &[f32],
        reference_samples: &[crate::types::AudioSample],
        strength: f32,
    ) -> Result<Vec<f32>> {
        if reference_samples.is_empty() || strength < 0.1 {
            return Ok(audio.to_vec());
        }

        // Apply style transfer from reference samples
        let mut result = audio.to_vec();

        // Calculate average characteristics from reference
        let avg_energy = reference_samples
            .iter()
            .map(|sample| {
                sample.audio.iter().map(|x| x * x).sum::<f32>() / sample.audio.len() as f32
            })
            .sum::<f32>()
            / reference_samples.len() as f32;

        // Adjust energy to match reference
        let current_energy = result.iter().map(|x| x * x).sum::<f32>() / result.len() as f32;
        if current_energy > 0.0 {
            let energy_ratio = (avg_energy / current_energy).sqrt();
            let adjusted_ratio = 1.0 + (energy_ratio - 1.0) * strength;

            for sample in &mut result {
                *sample *= adjusted_ratio;
            }
        }

        Ok(result)
    }

    /// Apply speaker-specific post-processing
    pub(super) async fn apply_speaker_post_processing(
        &self,
        audio: &[f32],
        target: &crate::types::ConversionTarget,
    ) -> Result<Vec<f32>> {
        let mut result = audio.to_vec();

        // Apply conversion strength
        if target.strength < 1.0 {
            // Blend with original (would need original audio for real implementation)
            for sample in &mut result {
                *sample *= target.strength;
            }
        }

        // Apply preservation of original characteristics
        if target.preserve_original > 0.0 {
            // In practice, would blend with original audio characteristics
            let preservation_factor = 1.0 - target.preserve_original * 0.3;
            for sample in &mut result {
                *sample *= preservation_factor;
            }
        }

        Ok(result)
    }

    /// Apply speaker-specific acoustic transformation (legacy method)
    pub(super) async fn apply_speaker_transform(
        &self,
        audio: &[f32],
        characteristics: &VoiceCharacteristics,
    ) -> Result<Vec<f32>> {
        // Delegate to the enhanced version
        self.apply_advanced_speaker_transform(audio, characteristics, None)
            .await
    }
}
