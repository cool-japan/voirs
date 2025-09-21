//! Acoustic adapter for integrating voirs-acoustic with the SDK.

use crate::traits::{
    AcousticModel as SdkAcousticModel, AcousticModelFeature as SdkAcousticModelFeature,
    AcousticModelMetadata as SdkAcousticModelMetadata,
};
use crate::types::{
    LanguageCode as SdkLanguageCode, MelSpectrogram as SdkMelSpectrogram, Phoneme as SdkPhoneme,
};
use crate::{Result, VoirsError};
use async_trait::async_trait;
use std::{collections::HashMap, sync::Arc};
use voirs_acoustic::speaker::{EmotionIntensity, SpeakerId, VoiceStyleControl};

/// Adapter that bridges voirs-acoustic components to SDK AcousticModel trait
pub struct AcousticAdapter {
    inner: Arc<dyn voirs_acoustic::traits::AcousticModel>,
}

impl AcousticAdapter {
    /// Create new acoustic adapter wrapping a voirs-acoustic implementation
    pub fn new(acoustic: Arc<dyn voirs_acoustic::traits::AcousticModel>) -> Self {
        Self { inner: acoustic }
    }

    /// Convert SDK Phoneme to voirs-acoustic Phoneme
    fn convert_phoneme_to_acoustic(phoneme: &SdkPhoneme) -> voirs_acoustic::Phoneme {
        let mut features = std::collections::HashMap::new();

        // Store IPA symbol as feature
        if phoneme.ipa_symbol != phoneme.symbol {
            features.insert("ipa_symbol".to_string(), phoneme.ipa_symbol.clone());
        }

        // Store stress as feature
        if phoneme.stress > 0 {
            features.insert("stress".to_string(), phoneme.stress.to_string());
        }

        // Store syllable position as feature
        let syllable_pos = match phoneme.syllable_position {
            crate::types::SyllablePosition::Onset => "onset",
            crate::types::SyllablePosition::Nucleus => "nucleus",
            crate::types::SyllablePosition::Coda => "coda",
            crate::types::SyllablePosition::Unknown => "unknown",
        };
        features.insert("syllable_position".to_string(), syllable_pos.to_string());

        // Store confidence as feature
        if phoneme.confidence < 1.0 {
            features.insert("confidence".to_string(), phoneme.confidence.to_string());
        }

        voirs_acoustic::Phoneme {
            symbol: phoneme.symbol.clone(),
            features: if features.is_empty() {
                None
            } else {
                Some(features)
            },
            duration: phoneme.duration_ms.map(|d| d / 1000.0), // Convert ms to seconds
        }
    }

    /// Convert voirs-acoustic MelSpectrogram to SDK MelSpectrogram
    fn convert_mel_spectrogram_from_acoustic(
        mel: voirs_acoustic::MelSpectrogram,
    ) -> SdkMelSpectrogram {
        SdkMelSpectrogram {
            data: mel.data,
            sample_rate: mel.sample_rate,
            hop_length: mel.hop_length,
            n_mels: mel.n_mels as u32,
            n_frames: mel.n_frames as u32,
        }
    }

    /// Convert SDK SynthesisConfig to voirs-acoustic SynthesisConfig
    fn convert_synthesis_config_to_acoustic(
        config: &Option<&crate::types::SynthesisConfig>,
    ) -> Option<voirs_acoustic::SynthesisConfig> {
        config.map(|cfg| {
            voirs_acoustic::SynthesisConfig {
                speed: cfg.speaking_rate,
                pitch_shift: cfg.pitch_shift,
                energy: 10.0_f32.powf(cfg.volume_gain / 20.0), // Convert dB to linear
                speaker_id: None, // SDK doesn't have speaker_id in SynthesisConfig
                seed: cfg.seed,
                emotion: Self::convert_emotion_config(cfg),
                voice_style: Self::convert_voice_style_config(cfg),
            }
        })
    }

    /// Convert SDK emotion configuration to voirs-acoustic emotion
    fn convert_emotion_config(
        config: &crate::types::SynthesisConfig,
    ) -> Option<voirs_acoustic::EmotionConfig> {
        if !config.enable_emotion {
            return None;
        }

        let emotion_type = config.emotion_type.as_ref()?;
        let intensity = config.emotion_intensity;

        // Map SDK emotion types to voirs-acoustic emotion types
        let acoustic_emotion_type = match emotion_type.as_str() {
            "happy" => voirs_acoustic::EmotionType::Happy,
            "sad" => voirs_acoustic::EmotionType::Sad,
            "angry" => voirs_acoustic::EmotionType::Angry,
            "excited" => voirs_acoustic::EmotionType::Excited,
            "calm" => voirs_acoustic::EmotionType::Calm,
            "neutral" => voirs_acoustic::EmotionType::Neutral,
            "fear" => voirs_acoustic::EmotionType::Fear,
            "surprise" => voirs_acoustic::EmotionType::Surprise,
            "disgust" => voirs_acoustic::EmotionType::Disgust,
            _ => voirs_acoustic::EmotionType::Neutral, // Default fallback
        };

        Some(voirs_acoustic::EmotionConfig {
            emotion_type: acoustic_emotion_type,
            intensity: EmotionIntensity::Custom(intensity.clamp(0.0, 1.0)),
            secondary_emotions: Vec::new(), // No secondary emotions for now
            custom_params: HashMap::new(),  // No custom parameters for now
        })
    }

    /// Convert SDK voice style configuration to voirs-acoustic voice style
    fn convert_voice_style_config(
        config: &crate::types::SynthesisConfig,
    ) -> Option<VoiceStyleControl> {
        // Check if we have any style-related configuration
        let has_style_config = config.enable_emotion
            || (config.speaking_rate - 1.0).abs() > 0.1
            || config.pitch_shift.abs() > 0.1
            || config.volume_gain.abs() > 0.1;

        if !has_style_config {
            return None;
        }

        // Determine the primary style name
        let style_name = if config.enable_emotion && config.emotion_type.is_some() {
            let emotion = config.emotion_type.as_ref().unwrap();
            format!("emotional_{}", emotion)
        } else if config.speaking_rate > 1.3 {
            "fast_speaking".to_string()
        } else if config.speaking_rate < 0.7 {
            "slow_speaking".to_string()
        } else if config.pitch_shift > 2.0 {
            "high_pitch".to_string()
        } else if config.pitch_shift < -2.0 {
            "low_pitch".to_string()
        } else {
            "natural".to_string()
        };

        // Calculate overall intensity based on emotion or deviation from defaults
        let intensity = if config.enable_emotion {
            config.emotion_intensity
        } else {
            // Calculate intensity based on how much we deviate from defaults
            let rate_deviation = (config.speaking_rate - 1.0).abs();
            let pitch_deviation = config.pitch_shift.abs() / 12.0; // Normalize to 0-1
            let volume_deviation = config.volume_gain.abs() / 20.0; // Normalize to 0-1
            (rate_deviation + pitch_deviation + volume_deviation).min(1.0)
        };

        Some(VoiceStyleControl {
            base_speaker: SpeakerId::new(0), // Use default base speaker (ID 0)
            emotion: if config.enable_emotion {
                Self::convert_emotion_config(config).unwrap_or_default()
            } else {
                voirs_acoustic::EmotionConfig::default()
            },
            age_factor: 1.0,    // Default age
            gender_factor: 1.0, // Default gender
            speed_factor: config.speaking_rate,
            pitch_shift: config.pitch_shift, // Use actual field name
            energy_factor: 10.0_f32.powf(config.volume_gain / 20.0), // Convert dB to linear
            breathiness: 0.0,                // Default breathiness
            roughness: 0.0,                  // Default roughness
        })
    }

    /// Convert SDK LanguageCode to voirs-acoustic LanguageCode
    #[allow(dead_code)]
    fn convert_language_code_to_acoustic(lang: SdkLanguageCode) -> voirs_acoustic::LanguageCode {
        match lang {
            SdkLanguageCode::EnUs => voirs_acoustic::LanguageCode::EnUs,
            SdkLanguageCode::EnGb => voirs_acoustic::LanguageCode::EnGb,
            SdkLanguageCode::De | SdkLanguageCode::DeDe => voirs_acoustic::LanguageCode::DeDe,
            SdkLanguageCode::Fr | SdkLanguageCode::FrFr => voirs_acoustic::LanguageCode::FrFr,
            SdkLanguageCode::Es | SdkLanguageCode::EsEs | SdkLanguageCode::EsMx => {
                voirs_acoustic::LanguageCode::EsEs
            }
            SdkLanguageCode::Ja | SdkLanguageCode::JaJp => voirs_acoustic::LanguageCode::JaJp,
            SdkLanguageCode::ZhCn => voirs_acoustic::LanguageCode::ZhCn,
            SdkLanguageCode::Ko | SdkLanguageCode::KoKr => voirs_acoustic::LanguageCode::KoKr,
            // Default to English for unsupported languages
            _ => voirs_acoustic::LanguageCode::EnUs,
        }
    }

    /// Convert voirs-acoustic AcousticModelMetadata to SDK AcousticModelMetadata
    fn convert_metadata_from_acoustic(
        metadata: voirs_acoustic::AcousticModelMetadata,
    ) -> SdkAcousticModelMetadata {
        SdkAcousticModelMetadata {
            name: metadata.name,
            version: metadata.version,
            architecture: metadata.architecture,
            supported_languages: metadata
                .supported_languages
                .into_iter()
                .map(Self::convert_language_code_from_acoustic)
                .collect(),
            sample_rate: metadata.sample_rate,
            mel_channels: metadata.mel_channels,
            is_multi_speaker: metadata.is_multi_speaker,
            speaker_count: metadata.speaker_count,
        }
    }

    /// Convert voirs-acoustic LanguageCode to SDK LanguageCode
    fn convert_language_code_from_acoustic(lang: voirs_acoustic::LanguageCode) -> SdkLanguageCode {
        match lang {
            voirs_acoustic::LanguageCode::EnUs => SdkLanguageCode::EnUs,
            voirs_acoustic::LanguageCode::EnGb => SdkLanguageCode::EnGb,
            voirs_acoustic::LanguageCode::DeDe => SdkLanguageCode::De,
            voirs_acoustic::LanguageCode::FrFr => SdkLanguageCode::Fr,
            voirs_acoustic::LanguageCode::EsEs => SdkLanguageCode::Es,
            voirs_acoustic::LanguageCode::JaJp => SdkLanguageCode::Ja,
            voirs_acoustic::LanguageCode::ZhCn => SdkLanguageCode::ZhCn,
            voirs_acoustic::LanguageCode::KoKr => SdkLanguageCode::Ko,
            voirs_acoustic::LanguageCode::ItIt => SdkLanguageCode::It,
        }
    }
}

#[async_trait]
impl SdkAcousticModel for AcousticAdapter {
    async fn synthesize(
        &self,
        phonemes: &[SdkPhoneme],
        config: Option<&crate::types::SynthesisConfig>,
    ) -> Result<SdkMelSpectrogram> {
        let acoustic_phonemes: Vec<_> = phonemes
            .iter()
            .map(Self::convert_phoneme_to_acoustic)
            .collect();

        let acoustic_config = Self::convert_synthesis_config_to_acoustic(&config);

        match self
            .inner
            .synthesize(&acoustic_phonemes, acoustic_config.as_ref())
            .await
        {
            Ok(mel) => Ok(Self::convert_mel_spectrogram_from_acoustic(mel)),
            Err(err) => Err(VoirsError::ModelError {
                model_type: crate::error::types::ModelType::Acoustic,
                message: format!("Acoustic synthesis failed: {err}"),
                source: Some(Box::new(err)),
            }),
        }
    }

    async fn synthesize_batch(
        &self,
        inputs: &[&[SdkPhoneme]],
        configs: Option<&[crate::types::SynthesisConfig]>,
    ) -> Result<Vec<SdkMelSpectrogram>> {
        let acoustic_inputs: Vec<Vec<_>> = inputs
            .iter()
            .map(|phonemes| {
                phonemes
                    .iter()
                    .map(Self::convert_phoneme_to_acoustic)
                    .collect()
            })
            .collect();

        let acoustic_inputs_refs: Vec<&[_]> = acoustic_inputs
            .iter()
            .map(|phonemes| phonemes.as_slice())
            .collect();

        let acoustic_configs: Option<Vec<_>> = configs.map(|cfgs| {
            cfgs.iter()
                .map(|cfg| Self::convert_synthesis_config_to_acoustic(&Some(cfg)).unwrap())
                .collect()
        });

        match self
            .inner
            .synthesize_batch(&acoustic_inputs_refs, acoustic_configs.as_deref())
            .await
        {
            Ok(mels) => Ok(mels
                .into_iter()
                .map(Self::convert_mel_spectrogram_from_acoustic)
                .collect()),
            Err(err) => Err(VoirsError::ModelError {
                model_type: crate::error::types::ModelType::Acoustic,
                message: format!("Acoustic batch synthesis failed: {err}"),
                source: Some(Box::new(err)),
            }),
        }
    }

    fn metadata(&self) -> SdkAcousticModelMetadata {
        Self::convert_metadata_from_acoustic(self.inner.metadata())
    }

    fn supports(&self, feature: SdkAcousticModelFeature) -> bool {
        let acoustic_feature = match feature {
            SdkAcousticModelFeature::MultiSpeaker => {
                voirs_acoustic::AcousticModelFeature::MultiSpeaker
            }
            SdkAcousticModelFeature::EmotionControl => {
                voirs_acoustic::AcousticModelFeature::EmotionControl
            }
            SdkAcousticModelFeature::ProsodyControl => {
                voirs_acoustic::AcousticModelFeature::ProsodyControl
            }
            SdkAcousticModelFeature::StreamingInference => {
                voirs_acoustic::AcousticModelFeature::StreamingInference
            }
            SdkAcousticModelFeature::StreamingSynthesis => {
                voirs_acoustic::AcousticModelFeature::StreamingSynthesis
            }
            SdkAcousticModelFeature::BatchProcessing => {
                voirs_acoustic::AcousticModelFeature::BatchProcessing
            }
            SdkAcousticModelFeature::StyleTransfer => {
                voirs_acoustic::AcousticModelFeature::StyleTransfer
            }
            SdkAcousticModelFeature::GpuAcceleration => {
                voirs_acoustic::AcousticModelFeature::GpuAcceleration
            }
            SdkAcousticModelFeature::VoiceCloning => {
                voirs_acoustic::AcousticModelFeature::VoiceCloning
            }
            SdkAcousticModelFeature::RealTimeInference => {
                voirs_acoustic::AcousticModelFeature::RealTimeInference
            }
        };

        self.inner.supports(acoustic_feature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_acoustic::DummyAcousticModel;

    #[tokio::test]
    async fn test_acoustic_adapter_creation() {
        // Use dummy acoustic implementation for testing
        let dummy_acoustic = Arc::new(DummyAcousticModel::new());
        let adapter = AcousticAdapter::new(dummy_acoustic);

        // Test that adapter can be created and basic functionality works
        let metadata = adapter.metadata();
        assert!(!metadata.name.is_empty());
        assert!(!metadata.supported_languages.is_empty());
    }

    #[tokio::test]
    async fn test_acoustic_adapter_synthesis() {
        // Use dummy acoustic implementation for testing
        let dummy_acoustic = Arc::new(DummyAcousticModel::new());
        let adapter = AcousticAdapter::new(dummy_acoustic);

        // Test phoneme synthesis
        let phonemes = vec![
            SdkPhoneme {
                symbol: "h".to_string(),
                ipa_symbol: "h".to_string(),
                stress: 0,
                syllable_position: crate::types::SyllablePosition::Onset,
                duration_ms: Some(50.0),
                confidence: 0.9,
            },
            SdkPhoneme {
                symbol: "ɛ".to_string(),
                ipa_symbol: "ɛ".to_string(),
                stress: 1,
                syllable_position: crate::types::SyllablePosition::Nucleus,
                duration_ms: Some(120.0),
                confidence: 0.95,
            },
        ];

        let result = adapter.synthesize(&phonemes, None).await;
        assert!(result.is_ok());

        let mel = result.unwrap();
        assert!(mel.n_mels > 0);
        assert!(mel.n_frames > 0);
        assert!(!mel.data.is_empty());
    }
}
