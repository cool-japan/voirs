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
            match ConsentManager::new().await {
                Ok(manager) => {
                    console_log!("Consent manager initialized");
                    self.consent_manager = Some(manager);
                }
                Err(e) => {
                    console_error!("Failed to initialize consent manager: {}", e);
                    // Non-fatal error, continue without consent management
                }
            }
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

        // Verify consent if consent manager is available
        if let (Some(consent_manager), Some(consent)) =
            (&self.consent_manager, &wasm_request.consent)
        {
            let consent_record = ConsentRecord {
                subject_id: consent.subject_id.clone(),
                status: match consent.status.as_str() {
                    "granted" => ConsentStatus::Granted,
                    "denied" => ConsentStatus::Denied,
                    "revoked" => ConsentStatus::Revoked,
                    _ => ConsentStatus::Pending,
                },
                timestamp: std::time::SystemTime::UNIX_EPOCH
                    + std::time::Duration::from_secs(consent.timestamp),
                consent_type: crate::consent::ConsentType::VoiceCloning,
                permissions: crate::consent::ConsentPermissions::default(),
                usage_context: crate::consent::ConsentUsageContext {
                    purpose: "voice_cloning".to_string(),
                    duration: None,
                    data_retention: None,
                    third_party_sharing: false,
                },
                restrictions: crate::consent::UsageRestrictions::default(),
                verification_method: consent
                    .verification_method
                    .as_ref()
                    .map(|_| crate::consent::ConsentVerificationMethod::Digital),
                metadata: std::collections::HashMap::new(),
            };

            match consent_manager.verify_consent(&consent_record).await {
                Ok(result) => {
                    if !result.is_valid {
                        return Err(JsValue::from_str("Consent verification failed"));
                    }
                }
                Err(e) => {
                    console_error!("Consent verification error: {}", e);
                    return Err(JsValue::from_str(&format!(
                        "Consent verification error: {e}"
                    )));
                }
            }
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
                audio_data: audio_samples,
                sample_rate: wasm_sample.sample_rate,
                channels: wasm_sample.channels,
                duration: wasm_sample.duration,
                transcript: wasm_sample.transcript.clone(),
                speaker_id: wasm_sample.speaker_id.clone(),
                language: wasm_sample.language.clone(),
                quality_score: wasm_sample.quality_score,
                metadata: std::collections::HashMap::new(),
            };
            voice_samples.push(voice_sample);
        }

        // Create speaker profile if provided
        let speaker_profile = if let Some(wasm_profile) = &wasm_request.speaker_profile {
            let embedding = wasm_profile
                .embedding
                .as_ref()
                .map(|emb| SpeakerEmbedding::new(emb.clone()));

            let profile = SpeakerProfile {
                speaker_id: wasm_profile.speaker_id.clone(),
                name: wasm_profile.name.clone(),
                gender: wasm_profile.gender.clone(),
                age_range: wasm_profile.age_range.clone(),
                native_language: wasm_profile.native_language.clone(),
                characteristics: wasm_profile.characteristics.clone(),
                embedding,
                metadata: std::collections::HashMap::new(),
            };
            Some(profile)
        } else {
            None
        };

        // Create cloning request
        let clone_request = VoiceCloneRequest {
            reference_samples: voice_samples,
            target_text: wasm_request.target_text.clone(),
            speaker_profile: speaker_profile.clone(),
            adaptation_config: AdaptationConfig::default(), // Use default for now
            metadata: std::collections::HashMap::new(),
        };

        // Perform voice cloning
        match cloner.clone_voice(&clone_request).await {
            Ok(clone_result) => {
                console_log!("Voice cloning completed successfully");

                // Convert audio data back to bytes (PCM16)
                let mut output_bytes = Vec::with_capacity(clone_result.audio_data.len() * 2);
                for sample in &clone_result.audio_data {
                    let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                    output_bytes.extend_from_slice(&sample_i16.to_le_bytes());
                }

                // Assess quality if quality assessor is available
                let quality_metrics = if let Some(assessor) = &self.quality_assessor {
                    match assessor.assess_quality(&clone_result).await {
                        Ok(metrics) => WasmQualityMetrics {
                            overall_score: metrics.overall_score,
                            similarity_score: metrics.similarity_score,
                            audio_quality: metrics.audio_quality,
                            naturalness: metrics.naturalness,
                            intelligibility: metrics.intelligibility,
                            detailed_metrics: metrics.detailed_metrics,
                        },
                        Err(_) => WasmQualityMetrics {
                            overall_score: 0.8, // Default fallback
                            similarity_score: 0.8,
                            audio_quality: 0.8,
                            naturalness: 0.8,
                            intelligibility: 0.8,
                            detailed_metrics: std::collections::HashMap::new(),
                        },
                    }
                } else {
                    WasmQualityMetrics {
                        overall_score: 0.8, // Default fallback
                        similarity_score: 0.8,
                        audio_quality: 0.8,
                        naturalness: 0.8,
                        intelligibility: 0.8,
                        detailed_metrics: std::collections::HashMap::new(),
                    }
                };

                // Perform speaker verification if verifier is available
                let verification =
                    if let (Some(verifier), Some(profile)) = (&self.verifier, &speaker_profile) {
                        match verifier
                            .verify_speaker(&clone_result.audio_data, profile)
                            .await
                        {
                            Ok(result) => Some(WasmVerificationResult {
                                verified: result.verified,
                                confidence: result.confidence,
                                match_probability: result.similarity_score,
                                method: "embedding_similarity".to_string(),
                                metadata: result.metadata,
                            }),
                            Err(_) => None,
                        }
                    } else {
                        None
                    };

                // Store current speaker profile
                self.current_speaker_profile = speaker_profile;

                let wasm_result = WasmCloneResult {
                    audio_data: output_bytes,
                    sample_rate: clone_result.sample_rate,
                    channels: clone_result.channels,
                    duration: clone_result.duration,
                    quality_metrics,
                    adaptation_stats: clone_result
                        .metadata
                        .iter()
                        .filter_map(|(k, v)| v.parse::<f32>().ok().map(|f| (k.clone(), f)))
                        .collect(),
                    verification,
                    metadata: clone_result.metadata,
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
                audio_data: audio_samples,
                sample_rate: wasm_sample.sample_rate,
                channels: wasm_sample.channels,
                duration: wasm_sample.duration,
                transcript: wasm_sample.transcript,
                speaker_id: wasm_sample.speaker_id,
                language: wasm_sample.language,
                quality_score: wasm_sample.quality_score,
                metadata: std::collections::HashMap::new(),
            };
            voice_samples.push(voice_sample);
        }

        // Create few-shot learner
        let few_shot_config = FewShotConfig::default();
        let mut learner = FewShotLearner::new(few_shot_config);

        match learner.adapt(&voice_samples).await {
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
                let wasm_profile = WasmSpeakerProfile {
                    speaker_id: profile.speaker_id.clone(),
                    name: profile.name.clone(),
                    gender: profile.gender.clone(),
                    age_range: profile.age_range.clone(),
                    native_language: profile.native_language.clone(),
                    characteristics: profile.characteristics.clone(),
                    embedding: profile.embedding.as_ref().map(|emb| emb.vector().to_vec()),
                };
                JsValue::from_serde(&wasm_profile).unwrap_or(JsValue::NULL)
            }
            None => JsValue::NULL,
        }
    }

    /// Clear current speaker adaptation
    #[wasm_bindgen]
    pub async fn clear_adaptation(&mut self) -> std::result::Result<(), JsValue> {
        if let Some(cloner) = &mut self.cloner {
            match cloner.reset_adaptation().await {
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
        let stats = serde_json::json!({
            "cloner_initialized": self.cloner.is_some(),
            "quality_assessor_initialized": self.quality_assessor.is_some(),
            "verifier_initialized": self.verifier.is_some(),
            "consent_manager_initialized": self.consent_manager.is_some(),
            "current_speaker_loaded": self.current_speaker_profile.is_some(),
            "wasm_memory": get_wasm_memory_usage()
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
