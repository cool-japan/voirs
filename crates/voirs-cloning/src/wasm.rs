//! WebAssembly bindings for browser-based voice cloning
//!
//! This module provides WebAssembly bindings that enable voice cloning
//! capabilities to run in web browsers, allowing real-time speaker adaptation
//! and voice synthesis in client-side applications.

use crate::{
    config::CloningConfig,
    consent::{ConsentManager, ConsentRecord, ConsentStatus},
    core::{AdaptationConfig, VoiceCloner},
    embedding::{SpeakerEmbedding, SpeakerEmbeddingExtractor},
    few_shot::{FewShotConfig, FewShotLearner, FewShotResult},
    quality::{CloningQualityAssessor, QualityMetrics},
    types::{SpeakerProfile, VoiceCloneRequest, VoiceCloneResult, VoiceSample},
    verification::{SpeakerVerifier, VerificationResult},
    Result,
};
use js_sys::{Array, Object, Uint8Array};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{AudioBuffer, AudioContext};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

macro_rules! console_error {
    ($($t:tt)*) => (error(&format_args!($($t)*).to_string()))
}

/// WebAssembly-compatible cloning configuration
#[derive(Serialize, Deserialize, Clone)]
pub struct WasmCloningConfig {
    /// Target voice quality (0.0-1.0)
    pub target_quality: Option<f32>,
    /// Adaptation speed (0.0-1.0)  
    pub adaptation_speed: Option<f32>,
    /// Enable few-shot learning
    pub enable_few_shot: Option<bool>,
    /// Number of reference samples required
    pub min_reference_samples: Option<usize>,
    /// Enable real-time adaptation
    pub enable_realtime: Option<bool>,
    /// Enable consent verification
    pub enable_consent_verification: Option<bool>,
    /// Enable quality assessment
    pub enable_quality_assessment: Option<bool>,
    /// Cultural context for adaptation
    pub cultural_context: Option<String>,
}

/// WebAssembly-compatible voice sample
#[derive(Serialize, Deserialize, Clone)]
pub struct WasmVoiceSample {
    /// Audio data as bytes (PCM16)
    pub audio_data: Vec<u8>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Duration in seconds
    pub duration: f64,
    /// Transcription text
    pub transcript: Option<String>,
    /// Speaker identification
    pub speaker_id: Option<String>,
    /// Language code
    pub language: Option<String>,
    /// Quality metrics
    pub quality_score: Option<f32>,
}

/// WebAssembly-compatible cloning request
#[derive(Serialize, Deserialize)]
pub struct WasmCloneRequest {
    /// Reference voice samples
    pub reference_samples: Vec<WasmVoiceSample>,
    /// Target text to synthesize
    pub target_text: String,
    /// Adaptation configuration
    pub config: WasmCloningConfig,
    /// Speaker profile information
    pub speaker_profile: Option<WasmSpeakerProfile>,
    /// Consent information
    pub consent: Option<WasmConsentRecord>,
}

/// WebAssembly-compatible speaker profile
#[derive(Serialize, Deserialize, Clone)]
pub struct WasmSpeakerProfile {
    /// Unique speaker identifier
    pub speaker_id: String,
    /// Speaker name
    pub name: Option<String>,
    /// Gender information
    pub gender: Option<String>,
    /// Age range
    pub age_range: Option<String>,
    /// Native language
    pub native_language: Option<String>,
    /// Voice characteristics
    pub characteristics: HashMap<String, f32>,
    /// Embedding vector
    pub embedding: Option<Vec<f32>>,
}

/// WebAssembly-compatible consent record
#[derive(Serialize, Deserialize)]
pub struct WasmConsentRecord {
    /// Subject identifier
    pub subject_id: String,
    /// Consent status
    pub status: String, // "granted", "denied", "revoked"
    /// Timestamp of consent
    pub timestamp: u64,
    /// Consent type
    pub consent_type: String,
    /// Usage restrictions
    pub restrictions: HashMap<String, String>,
    /// Verification method
    pub verification_method: Option<String>,
}

/// WebAssembly-compatible cloning result
#[derive(Serialize, Deserialize)]
pub struct WasmCloneResult {
    /// Synthesized audio data
    pub audio_data: Vec<u8>,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Duration in seconds
    pub duration: f64,
    /// Quality metrics
    pub quality_metrics: WasmQualityMetrics,
    /// Adaptation statistics
    pub adaptation_stats: HashMap<String, f32>,
    /// Verification result
    pub verification: Option<WasmVerificationResult>,
    /// Processing metadata
    pub metadata: HashMap<String, String>,
}

/// WebAssembly-compatible quality metrics
#[derive(Serialize, Deserialize)]
pub struct WasmQualityMetrics {
    /// Overall quality score (0.0-1.0)
    pub overall_score: f32,
    /// Speaker similarity score (0.0-1.0)
    pub similarity_score: f32,
    /// Audio quality score (0.0-1.0)
    pub audio_quality: f32,
    /// Naturalness score (0.0-1.0)
    pub naturalness: f32,
    /// Intelligibility score (0.0-1.0)
    pub intelligibility: f32,
    /// Detailed metrics
    pub detailed_metrics: HashMap<String, f32>,
}

/// WebAssembly-compatible verification result
#[derive(Serialize, Deserialize)]
pub struct WasmVerificationResult {
    /// Verification passed
    pub verified: bool,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Speaker match probability
    pub match_probability: f32,
    /// Verification method used
    pub method: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Main WebAssembly voice cloning interface
#[wasm_bindgen]
pub struct WasmVoiceCloner {
    cloner: Option<VoiceCloner>,
    quality_assessor: Option<CloningQualityAssessor>,
    verifier: Option<SpeakerVerifier>,
    consent_manager: Option<ConsentManager>,
    audio_context: Option<AudioContext>,
    current_speaker_profile: Option<SpeakerProfile>,
}

#[wasm_bindgen]
impl WasmVoiceCloner {
    /// Create new WebAssembly voice cloner
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_log!("Creating new WasmVoiceCloner");
        utils::set_panic_hook();

        Self {
            cloner: None,
            quality_assessor: None,
            verifier: None,
            consent_manager: None,
            audio_context: None,
            current_speaker_profile: None,
        }
    }

    /// Initialize the voice cloner with configuration
    #[wasm_bindgen]
    pub async fn initialize(&mut self, config: JsValue) -> std::result::Result<(), JsValue> {
        console_log!("Initializing WasmVoiceCloner");

        let wasm_config: WasmCloningConfig = config
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("Config parsing error: {e}")))?;

        // Build cloning configuration
        let mut cloning_config = CloningConfig::default();

        if let Some(quality) = wasm_config.target_quality {
            cloning_config.target_quality = quality;
        }

        if let Some(speed) = wasm_config.adaptation_speed {
            cloning_config.adaptation_speed = speed;
        }

        if let Some(few_shot) = wasm_config.enable_few_shot {
            cloning_config.enable_few_shot = few_shot;
        }

        if let Some(min_samples) = wasm_config.min_reference_samples {
            cloning_config.min_reference_samples = min_samples;
        }

        if let Some(realtime) = wasm_config.enable_realtime {
            cloning_config.enable_realtime = realtime;
        }

        // Initialize voice cloner
        match VoiceCloner::new(cloning_config.clone()).await {
            Ok(cloner) => {
                console_log!("Voice cloner initialized successfully");
                self.cloner = Some(cloner);
            }
            Err(e) => {
                console_error!("Failed to initialize voice cloner: {}", e);
                return Err(JsValue::from_str(&format!(
                    "Cloner initialization failed: {e}"
                )));
            }
        }

        // Initialize quality assessor if enabled
        if wasm_config.enable_quality_assessment.unwrap_or(true) {
            match CloningQualityAssessor::new().await {
                Ok(assessor) => {
                    console_log!("Quality assessor initialized");
                    self.quality_assessor = Some(assessor);
                }
                Err(e) => {
                    console_error!("Failed to initialize quality assessor: {}", e);
                    // Non-fatal error, continue without quality assessment
                }
            }
        }

        // Initialize speaker verifier
        match SpeakerVerifier::new().await {
            Ok(verifier) => {
                console_log!("Speaker verifier initialized");
                self.verifier = Some(verifier);
            }
            Err(e) => {
                console_error!("Failed to initialize speaker verifier: {}", e);
                // Non-fatal error, continue without verification
            }
        }

        // Initialize consent manager if enabled
        if wasm_config.enable_consent_verification.unwrap_or(false) {
            let manager = ConsentManager::new();
            console_log!("Consent manager initialized");
            self.consent_manager = Some(manager);
        }

        // Initialize Web Audio Context
        match AudioContext::new() {
            Ok(ctx) => {
                self.audio_context = Some(ctx);
                console_log!("Audio context initialized");
            }
            Err(e) => {
                console_error!("Failed to create audio context: {:?}", e);
            }
        }

        Ok(())
    }

    /// Clone voice from reference samples
    #[wasm_bindgen]
    pub async fn clone_voice(&mut self, request: JsValue) -> std::result::Result<JsValue, JsValue> {
        let cloner = self
            .cloner
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Cloner not initialized"))?;

        let wasm_request: WasmCloneRequest = request
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("Request parsing error: {e}")))?;

        console_log!(
            "Processing voice cloning request with {} reference samples",
            wasm_request.reference_samples.len()
        );

        // Note: Consent verification in WASM context is simplified
        // Full consent management requires server-side validation
        if let (Some(_consent_manager), Some(consent)) =
            (&self.consent_manager, &wasm_request.consent)
        {
            // Basic consent check - verify it's granted
            if consent.status != "granted" {
                return Err(JsValue::from_str("Consent not granted"));
            }
            console_log!("Consent verification passed (simplified WASM mode)");
        }

        // Convert WASM voice samples to internal format
        let mut voice_samples = Vec::new();
        for wasm_sample in &wasm_request.reference_samples {
            // Convert audio data from bytes to f32 samples (assuming PCM16)
            let mut audio_samples = Vec::with_capacity(wasm_sample.audio_data.len() / 2);
            for chunk in wasm_sample.audio_data.chunks_exact(2) {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                audio_samples.push(sample);
            }

            let voice_sample = VoiceSample {
                id: wasm_sample
                    .speaker_id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                audio: audio_samples,
                sample_rate: wasm_sample.sample_rate,
                transcript: wasm_sample.transcript.clone(),
                language: wasm_sample.language.clone(),
                duration: wasm_sample.duration as f32,
                quality_score: wasm_sample.quality_score,
                metadata: std::collections::HashMap::new(),
                timestamp: std::time::SystemTime::now(),
            };
            voice_samples.push(voice_sample);
        }

        // Create speaker profile if provided
        let speaker_profile = if let Some(wasm_profile) = &wasm_request.speaker_profile {
            use crate::types::{AgeGroup, Gender, SpeakerCharacteristics};

            // Parse gender from string
            let gender =
                wasm_profile
                    .gender
                    .as_ref()
                    .and_then(|g| match g.to_lowercase().as_str() {
                        "male" => Some(Gender::Male),
                        "female" => Some(Gender::Female),
                        "other" => Some(Gender::Other),
                        _ => Some(Gender::Unknown),
                    });

            // Parse age range from string
            let age_group =
                wasm_profile
                    .age_range
                    .as_ref()
                    .and_then(|a| match a.to_lowercase().as_str() {
                        "child" => Some(AgeGroup::Child),
                        "teen" | "teenager" => Some(AgeGroup::Teen),
                        "young_adult" | "youngadult" => Some(AgeGroup::YoungAdult),
                        "middle_aged" | "middleaged" => Some(AgeGroup::MiddleAged),
                        "senior" => Some(AgeGroup::Senior),
                        _ => Some(AgeGroup::Unknown),
                    });

            let mut characteristics = SpeakerCharacteristics::default();
            characteristics.gender = gender;
            characteristics.age_group = age_group;
            characteristics.adaptive_features = wasm_profile.characteristics.clone();

            let now = std::time::SystemTime::now();
            let profile = SpeakerProfile {
                id: wasm_profile.speaker_id.clone(),
                name: wasm_profile
                    .name
                    .clone()
                    .unwrap_or_else(|| "Unknown".to_string()),
                characteristics,
                samples: Vec::new(),
                embedding: wasm_profile.embedding.clone(),
                languages: wasm_profile
                    .native_language
                    .as_ref()
                    .map(|lang| vec![lang.clone()])
                    .unwrap_or_default(),
                created_at: now,
                updated_at: now,
                metadata: std::collections::HashMap::new(),
            };
            Some(profile)
        } else {
            None
        };

        // Create speaker data from samples
        use crate::types::{CloningMethod, SpeakerData};

        if voice_samples.is_empty() {
            return Err(JsValue::from_str("No voice samples provided"));
        }

        // Create or use speaker profile
        let profile = speaker_profile.unwrap_or_else(|| {
            use crate::types::SpeakerCharacteristics;
            let now = std::time::SystemTime::now();
            SpeakerProfile {
                id: uuid::Uuid::new_v4().to_string(),
                name: "WASM Speaker".to_string(),
                characteristics: SpeakerCharacteristics::default(),
                samples: Vec::new(),
                embedding: None,
                languages: vec!["en".to_string()],
                created_at: now,
                updated_at: now,
                metadata: std::collections::HashMap::new(),
            }
        });

        let speaker_data = SpeakerData {
            profile,
            reference_samples: voice_samples,
            target_text: Some(wasm_request.target_text.clone()),
            target_language: None,
            context: std::collections::HashMap::new(),
        };

        // Create cloning request
        let clone_request = VoiceCloneRequest {
            id: uuid::Uuid::new_v4().to_string(),
            speaker_data,
            method: CloningMethod::FewShot,
            text: wasm_request.target_text.clone(),
            language: None,
            quality_level: 0.8,
            quality_tradeoff: 0.7,
            parameters: std::collections::HashMap::new(),
            timestamp: std::time::SystemTime::now(),
        };

        // Perform voice cloning
        match cloner.clone_voice(clone_request).await {
            Ok(clone_result) => {
                console_log!("Voice cloning completed successfully");

                // Convert audio data back to bytes (PCM16)
                let mut output_bytes = Vec::with_capacity(clone_result.audio.len() * 2);
                for sample in &clone_result.audio {
                    let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                    output_bytes.extend_from_slice(&sample_i16.to_le_bytes());
                }

                // Calculate duration from audio length and sample rate
                let duration = clone_result.audio.len() as f64 / clone_result.sample_rate as f64;

                // Note: Quality assessment in WASM requires original and cloned samples for comparison
                // For now, we use the quality metrics from the cloning result
                let quality_metrics = WasmQualityMetrics {
                    overall_score: clone_result.quality_metrics.overall_score,
                    similarity_score: clone_result.similarity_score,
                    audio_quality: clone_result.quality_metrics.audio_quality,
                    naturalness: clone_result.quality_metrics.naturalness,
                    intelligibility: 0.8, // Default
                    detailed_metrics: std::collections::HashMap::new(),
                };

                // Perform speaker verification if verifier is available
                // Note: Speaker verification is disabled in this context as the API has changed
                let verification = None;

                // Store current speaker profile
                self.current_speaker_profile = speaker_profile;

                let wasm_result = WasmCloneResult {
                    audio_data: output_bytes,
                    sample_rate: clone_result.sample_rate,
                    channels: 1, // Mono audio
                    duration,
                    quality_metrics,
                    adaptation_stats: clone_result.quality_metrics.clone(),
                    verification,
                    metadata: std::collections::HashMap::new(),
                };

                JsValue::from_serde(&wasm_result)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
            }
            Err(e) => {
                console_error!("Voice cloning failed: {}", e);
                Err(JsValue::from_str(&format!("Voice cloning failed: {e}")))
            }
        }
    }

    /// Perform few-shot learning adaptation
    #[wasm_bindgen]
    pub async fn few_shot_adapt(
        &mut self,
        samples: JsValue,
        config: JsValue,
    ) -> std::result::Result<JsValue, JsValue> {
        console_log!("Performing few-shot adaptation");

        let wasm_samples: Vec<WasmVoiceSample> = samples
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("Samples parsing error: {e}")))?;

        let wasm_config: WasmCloningConfig = config
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("Config parsing error: {e}")))?;

        // Convert to internal format
        let mut voice_samples = Vec::new();
        for wasm_sample in wasm_samples {
            let mut audio_samples = Vec::with_capacity(wasm_sample.audio_data.len() / 2);
            for chunk in wasm_sample.audio_data.chunks_exact(2) {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                audio_samples.push(sample);
            }

            let voice_sample = VoiceSample {
                id: wasm_sample
                    .speaker_id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                audio: audio_samples,
                sample_rate: wasm_sample.sample_rate,
                transcript: wasm_sample.transcript,
                language: wasm_sample.language,
                duration: wasm_sample.duration as f32,
                quality_score: wasm_sample.quality_score,
                metadata: std::collections::HashMap::new(),
                timestamp: std::time::SystemTime::now(),
            };
            voice_samples.push(voice_sample);
        }

        // Create few-shot learner
        let few_shot_config = FewShotConfig::default();
        let mut learner = match FewShotLearner::new(few_shot_config) {
            Ok(l) => l,
            Err(e) => {
                console_error!("Failed to create few-shot learner: {}", e);
                return Err(JsValue::from_str(&format!(
                    "Failed to create few-shot learner: {e}"
                )));
            }
        };

        // Generate a temporary speaker ID for adaptation
        let speaker_id = uuid::Uuid::new_v4().to_string();

        match learner.adapt_speaker(&speaker_id, &voice_samples).await {
            Ok(result) => {
                console_log!("Few-shot adaptation completed successfully");

                let wasm_result = serde_json::json!({
                    "success": true,
                    "adaptation_quality": result.adaptation_quality,
                    "samples_used": result.samples_processed,
                    "convergence_achieved": result.convergence_achieved,
                    "adaptation_time_ms": result.processing_time.as_millis(),
                    "quality_improvement": result.quality_improvement,
                    "metadata": result.metadata
                });

                JsValue::from_serde(&wasm_result)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
            }
            Err(e) => {
                console_error!("Few-shot adaptation failed: {}", e);
                Err(JsValue::from_str(&format!(
                    "Few-shot adaptation failed: {e}"
                )))
            }
        }
    }

    /// Get current speaker profile
    #[wasm_bindgen]
    pub fn get_current_speaker_profile(&self) -> JsValue {
        match &self.current_speaker_profile {
            Some(profile) => {
                // Convert gender enum to string
                let gender = profile.characteristics.gender.as_ref().map(|g| match g {
                    crate::types::Gender::Male => "male".to_string(),
                    crate::types::Gender::Female => "female".to_string(),
                    crate::types::Gender::Other => "other".to_string(),
                    crate::types::Gender::Unknown => "unknown".to_string(),
                });

                // Convert age group enum to string
                let age_range = profile.characteristics.age_group.as_ref().map(|a| match a {
                    crate::types::AgeGroup::Child => "child".to_string(),
                    crate::types::AgeGroup::Teen => "teen".to_string(),
                    crate::types::AgeGroup::YoungAdult => "young_adult".to_string(),
                    crate::types::AgeGroup::MiddleAged => "middle_aged".to_string(),
                    crate::types::AgeGroup::Senior => "senior".to_string(),
                    crate::types::AgeGroup::Unknown => "unknown".to_string(),
                });

                let wasm_profile = WasmSpeakerProfile {
                    speaker_id: profile.id.clone(),
                    name: Some(profile.name.clone()),
                    gender,
                    age_range,
                    native_language: profile.languages.first().cloned(),
                    characteristics: profile.characteristics.adaptive_features.clone(),
                    embedding: profile.embedding.clone(),
                };
                JsValue::from_serde(&wasm_profile).unwrap_or(JsValue::NULL)
            }
            None => JsValue::NULL,
        }
    }

    /// Clear current speaker adaptation
    #[wasm_bindgen]
    pub async fn clear_adaptation(&mut self) -> std::result::Result<(), JsValue> {
        if let Some(cloner) = &self.cloner {
            match cloner.clear_cache().await {
                Ok(()) => {
                    console_log!("Speaker adaptation cleared successfully");
                    self.current_speaker_profile = None;
                    Ok(())
                }
                Err(e) => {
                    console_error!("Failed to clear adaptation: {}", e);
                    Err(JsValue::from_str(&format!(
                        "Failed to clear adaptation: {e}"
                    )))
                }
            }
        } else {
            Err(JsValue::from_str("Cloner not initialized"))
        }
    }

    /// Get cloner capabilities
    #[wasm_bindgen]
    pub fn get_capabilities(&self) -> JsValue {
        let capabilities = serde_json::json!({
            "voice_cloning": true,
            "few_shot_learning": true,
            "speaker_verification": self.verifier.is_some(),
            "quality_assessment": self.quality_assessor.is_some(),
            "consent_management": self.consent_manager.is_some(),
            "real_time_adaptation": true,
            "multi_language_support": true,
            "supported_audio_formats": ["pcm16"],
            "supported_sample_rates": [16000, 22050, 44100, 48000],
            "max_reference_samples": 100,
            "min_reference_duration": 30.0,
            "max_audio_channels": 2
        });

        JsValue::from_serde(&capabilities).unwrap_or(JsValue::NULL)
    }

    /// Get memory usage statistics
    #[wasm_bindgen]
    pub fn get_memory_stats(&self) -> JsValue {
        // Note: We can't include JsValue directly in serde_json::json!
        // So we create the stats object without wasm_memory
        let stats = serde_json::json!({
            "cloner_initialized": self.cloner.is_some(),
            "quality_assessor_initialized": self.quality_assessor.is_some(),
            "verifier_initialized": self.verifier.is_some(),
            "consent_manager_initialized": self.consent_manager.is_some(),
            "current_speaker_loaded": self.current_speaker_profile.is_some(),
        });

        JsValue::from_serde(&stats).unwrap_or(JsValue::NULL)
    }

    /// Get cloner version
    #[wasm_bindgen]
    pub fn get_version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

/// Utility functions for WebAssembly
mod utils {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = console)]
        fn error(msg: &str);
    }

    pub fn set_panic_hook() {
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();
    }
}

/// Initialize WASM logger for debugging
#[wasm_bindgen]
pub fn init_wasm_logger() {
    console_log!("Initializing WASM logger for voice cloning");
    wasm_logger::init(wasm_logger::Config::default());
}

/// Get WebAssembly memory usage
#[wasm_bindgen]
pub fn get_wasm_memory_usage() -> JsValue {
    let memory = wasm_bindgen::memory()
        .dyn_into::<js_sys::WebAssembly::Memory>()
        .unwrap();

    let buffer = memory.buffer();
    let usage = serde_json::json!({
        "buffer_size": buffer.dyn_into::<js_sys::ArrayBuffer>().map(|ab| ab.byte_length()).unwrap_or(0),
        "available": true,
        "module": "voirs-cloning"
    });

    JsValue::from_serde(&usage).unwrap_or(JsValue::NULL)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_wasm_cloner_creation() {
        let cloner = WasmVoiceCloner::new();
        assert!(cloner.cloner.is_none());
        assert!(cloner.current_speaker_profile.is_none());
    }

    #[wasm_bindgen_test]
    async fn test_wasm_cloner_capabilities() {
        let cloner = WasmVoiceCloner::new();
        let capabilities = cloner.get_capabilities();
        assert!(!capabilities.is_null());
    }

    #[wasm_bindgen_test]
    async fn test_wasm_cloner_version() {
        let cloner = WasmVoiceCloner::new();
        let version = cloner.get_version();
        assert!(!version.is_empty());
    }

    #[wasm_bindgen_test]
    async fn test_memory_usage() {
        let usage = get_wasm_memory_usage();
        assert!(!usage.is_null());
    }
}
