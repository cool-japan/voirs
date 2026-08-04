//! Voice information wrapper for Python bindings

use super::common::*;

/// Python VoiceInfo wrapper
#[pyclass]
#[derive(Clone)]
pub struct PyVoiceInfo {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub language: String,
    #[pyo3(get)]
    pub quality: String,
    #[pyo3(get)]
    pub is_available: bool,
}

impl From<VoiceInfo> for PyVoiceInfo {
    fn from(voice: VoiceInfo) -> Self {
        // `VoiceConfig.metadata["status"]` is set by
        // `VoirsPipeline::list_voices()` (voirs-sdk) to "available" or
        // "downloadable" based on a real filesystem check of whether this
        // voice's model files exist locally
        // (`VoiceSwitchingManager::is_voice_available`, which `.exists()`s
        // each `ModelFileInfo.path` under the pipeline's models directory).
        // If this `VoiceInfo` was constructed via a path that never ran that
        // check, `metadata` has no "status" entry -- default to `false`
        // (not available) rather than fabricating a positive claim we can't
        // back up.
        let is_available = voice
            .config
            .metadata
            .get("status")
            .is_some_and(|status| status == "available");
        Self {
            id: voice.config.id,
            name: voice.config.name,
            language: voice.config.language.to_string(),
            quality: format!("{:?}", voice.config.characteristics.quality),
            is_available,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_sdk::types::{
        AgeRange, DeviceRequirements, Gender, LanguageCode, ModelConfig, ModelFormat,
        QualityLevel, SpeakingStyle, VoiceCharacteristics, VoiceConfig,
    };

    fn test_voice_config(status: Option<&str>) -> VoiceConfig {
        let mut metadata = std::collections::HashMap::new();
        if let Some(status) = status {
            metadata.insert("status".to_string(), status.to_string());
        }
        VoiceConfig {
            id: "test-voice".to_string(),
            name: "Test Voice".to_string(),
            language: LanguageCode::EnUs,
            characteristics: VoiceCharacteristics {
                gender: Some(Gender::Female),
                age: Some(AgeRange::Adult),
                style: SpeakingStyle::Neutral,
                emotion_support: false,
                quality: QualityLevel::High,
            },
            model_config: ModelConfig {
                g2p_model: None,
                acoustic_model: "acoustic.bin".to_string(),
                vocoder_model: "vocoder.bin".to_string(),
                format: ModelFormat::Candle,
                device_requirements: DeviceRequirements {
                    min_memory_mb: 512,
                    gpu_support: false,
                    compute_capabilities: vec!["cpu".to_string()],
                },
            },
            metadata,
        }
    }

    /// `is_available` must reflect the real `metadata["status"]` signal
    /// computed by voirs-sdk's filesystem-backed availability check, not a
    /// hardcoded `true`.
    #[test]
    fn test_is_available_reflects_real_metadata_status() {
        let available = VoiceInfo::from_config(test_voice_config(Some("available")));
        assert!(PyVoiceInfo::from(available).is_available);

        let downloadable = VoiceInfo::from_config(test_voice_config(Some("downloadable")));
        assert!(!PyVoiceInfo::from(downloadable).is_available);

        // No status metadata at all (e.g. a VoiceInfo constructed without
        // going through VoirsPipeline::list_voices()'s availability check)
        // must default to "not available" rather than fabricating `true`.
        let unknown = VoiceInfo::from_config(test_voice_config(None));
        assert!(!PyVoiceInfo::from(unknown).is_available);
    }
}
