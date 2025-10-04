use crate::integration::PipelineBuilder;
use crate::prelude::*;
use crate::{ASRConfig, RecognitionResult};
use js_sys::{Array, Uint8Array};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{console, AudioBuffer, AudioContext};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    fn warn(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

macro_rules! console_error {
    ($($t:tt)*) => (error(&format_args!($($t)*).to_string()))
}

macro_rules! console_warn {
    ($($t:tt)*) => (warn(&format_args!($($t)*).to_string()))
}

/// WASM recognition configuration
#[derive(Serialize, Deserialize, Clone)]
pub struct WasmRecognitionConfig {
    /// Model name to use for recognition
    pub model_name: Option<String>,
    /// Language code for recognition
    pub language: Option<String>,
    /// Sample rate of audio input
    pub sample_rate: Option<u32>,
    /// Size of audio chunks for processing
    pub chunk_size: Option<usize>,
    /// Enable voice activity detection
    pub enable_vad: Option<bool>,
    /// Confidence threshold for results
    pub confidence_threshold: Option<f32>,
    /// Beam search size
    pub beam_size: Option<usize>,
    /// Temperature for sampling
    pub temperature: Option<f32>,
    /// Suppress blank outputs
    pub suppress_blank: Option<bool>,
    /// Token IDs to suppress
    pub suppress_tokens: Option<Vec<u32>>,
}

/// WASM recognition result
#[derive(Serialize, Deserialize)]
pub struct WasmRecognitionResult {
    /// Recognized text
    pub text: String,
    /// Confidence score
    pub confidence: f32,
    /// Detected language
    pub language: Option<String>,
    /// Recognition segments
    pub segments: Vec<WasmSegment>,
    /// Processing time in milliseconds
    pub processing_time: f64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// WASM recognition segment
#[derive(Serialize, Deserialize)]
pub struct WasmSegment {
    /// Segment start time in seconds
    pub start_time: f64,
    /// Segment end time in seconds
    pub end_time: f64,
    /// Segment text
    pub text: String,
    /// Segment confidence score
    pub confidence: f32,
    /// No speech probability
    pub no_speech_prob: f32,
}

/// WASM streaming configuration
#[derive(Serialize, Deserialize)]
pub struct WasmStreamingConfig {
    /// Duration of each audio chunk in seconds
    pub chunk_duration: Option<f32>,
    /// Overlap duration between chunks in seconds
    pub overlap_duration: Option<f32>,
    /// Voice activity detection threshold
    pub vad_threshold: Option<f32>,
    /// Silence duration threshold in seconds
    pub silence_duration: Option<f32>,
    /// Maximum chunk size in bytes
    pub max_chunk_size: Option<usize>,
}

/// WASM speech recognizer
#[wasm_bindgen]
pub struct WasmVoirsRecognizer {
    pipeline: Option<UnifiedVoirsPipeline>,
    audio_context: Option<AudioContext>,
    streaming_active: bool,
    current_config: Option<WasmRecognitionConfig>,
}

#[wasm_bindgen]
impl WasmVoirsRecognizer {
    /// Create a new WASM recognizer
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_log!("Creating new WasmVoirsRecognizer");
        crate::wasm::utils::set_panic_hook();

        Self {
            pipeline: None,
            audio_context: None,
            streaming_active: false,
            current_config: None,
        }
    }

    /// Initialize the recognizer with configuration
    #[wasm_bindgen]
    pub async fn initialize(&mut self, config: JsValue) -> Result<(), JsValue> {
        console_log!("Initializing WasmVoirsRecognizer");

        let config: WasmRecognitionConfig = config
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("Config parsing error: {e}")))?;

        let mut builder = PipelineBuilder::new();

        if let Some(model_name) = &config.model_name {
            builder = builder.with_model(model_name);
        }

        if let Some(language_str) = &config.language {
            let language = match language_str.as_str() {
                "en" | "en-us" | "en_us" => Some(LanguageCode::EnUs),
                "en-gb" | "en_gb" => Some(LanguageCode::EnGb),
                "ja" | "ja-jp" | "ja_jp" => Some(LanguageCode::JaJp),
                "es" | "es-es" | "es_es" => Some(LanguageCode::EsEs),
                "es-mx" | "es_mx" => Some(LanguageCode::EsMx),
                "fr" | "fr-fr" | "fr_fr" => Some(LanguageCode::FrFr),
                "de" | "de-de" | "de_de" => Some(LanguageCode::DeDe),
                "zh" | "zh-cn" | "zh_cn" => Some(LanguageCode::ZhCn),
                "pt" | "pt-br" | "pt_br" => Some(LanguageCode::PtBr),
                "ru" | "ru-ru" | "ru_ru" => Some(LanguageCode::RuRu),
                "it" | "it-it" | "it_it" => Some(LanguageCode::ItIt),
                "ko" | "ko-kr" | "ko_kr" => Some(LanguageCode::KoKr),
                "nl" | "nl-nl" | "nl_nl" => Some(LanguageCode::NlNl),
                "sv" | "sv-se" | "sv_se" => Some(LanguageCode::SvSe),
                "no" | "no-no" | "no_no" => Some(LanguageCode::NoNo),
                "da" | "da-dk" | "da_dk" => Some(LanguageCode::DaDk),
                _ => None,
            };
            if let Some(lang) = language {
                builder = builder.with_language(lang);
            }
        }

        if let Some(sample_rate) = config.sample_rate {
            builder = builder.with_sample_rate(sample_rate);
        }

        // ASRConfig parameters would need to be handled by the model implementation
        // For now, we just store them for later use
        let mut asr_config = ASRConfig::default();

        if let Some(enable_vad) = config.enable_vad {
            asr_config.enable_voice_activity_detection = enable_vad;
        }

        if let Some(confidence_threshold) = config.confidence_threshold {
            asr_config.confidence_threshold = confidence_threshold;
        }

        // Store ASR config for later use when creating models
        // builder = builder.with_asr_config(asr_config); // This would need to be implemented

        match builder.build().await {
            Ok(pipeline) => {
                console_log!("Recognition pipeline initialized successfully");
                self.pipeline = Some(pipeline);
                self.current_config = Some(config);

                // Initialize Web Audio Context
                match AudioContext::new() {
                    Ok(ctx) => {
                        self.audio_context = Some(ctx);
                        console_log!("Audio context initialized");
                    }
                    Err(e) => {
                        console_warn!("Failed to create audio context: {:?}", e);
                    }
                }

                Ok(())
            }
            Err(e) => {
                console_error!("Pipeline initialization failed: {}", e);
                Err(JsValue::from_str(&format!(
                    "Pipeline initialization failed: {e}"
                )))
            }
        }
    }

    /// Recognize audio from raw bytes
    #[wasm_bindgen]
    pub async fn recognize_audio(&self, audio_data: &[u8]) -> Result<JsValue, JsValue> {
        console_log!("Recognizing audio data ({} bytes)", audio_data.len());

        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Pipeline not initialized"))?;

        let start_time = js_sys::Date::now();

        match pipeline.recognize_bytes(audio_data).await {
            Ok(result) => {
                let processing_time = js_sys::Date::now() - start_time;
                console_log!("Recognition completed in {:.2}ms", processing_time);

                let wasm_result = if let Some(transcript) = result.transcription {
                    WasmRecognitionResult {
                        text: transcript.text.clone(),
                        confidence: transcript.confidence,
                        language: Some(transcript.language.to_string()),
                        segments: transcript
                            .word_timestamps
                            .iter()
                            .map(|word| WasmSegment {
                                start_time: word.start_time as f64,
                                end_time: word.end_time as f64,
                                text: word.word.clone(),
                                confidence: word.confidence,
                                no_speech_prob: 0.0, // Not available in word timestamps
                            })
                            .collect(),
                        processing_time,
                        metadata: {
                            let mut map = HashMap::new();
                            map.insert("model".to_string(), "pipeline".to_string());
                            map.insert(
                                "detected_language".to_string(),
                                transcript.language.to_string(),
                            );
                            if let Some(duration) = transcript.processing_duration {
                                map.insert(
                                    "audio_duration".to_string(),
                                    duration.as_secs_f32().to_string(),
                                );
                            }
                            map.insert(
                                "processing_time_ms".to_string(),
                                processing_time.to_string(),
                            );
                            map
                        },
                    }
                } else {
                    // No transcription available
                    WasmRecognitionResult {
                        text: "".to_string(),
                        confidence: 0.0,
                        language: None,
                        segments: vec![],
                        processing_time,
                        metadata: HashMap::new(),
                    }
                };

                JsValue::from_serde(&wasm_result)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
            }
            Err(e) => {
                console_error!("Recognition failed: {}", e);
                Err(JsValue::from_str(&format!("Recognition failed: {e}")))
            }
        }
    }

    /// Recognize audio from microphone
    #[wasm_bindgen]
    pub async fn recognize_from_microphone(&self) -> Result<JsValue, JsValue> {
        console_log!("Starting microphone recognition");

        // This would require getUserMedia integration
        // For now, return an error indicating it needs to be implemented
        Err(JsValue::from_str(
            "Microphone recognition requires getUserMedia integration - implement in JavaScript",
        ))
    }

    /// Start streaming recognition
    #[wasm_bindgen]
    pub async fn start_streaming(&mut self, config: JsValue) -> Result<(), JsValue> {
        console_log!("Starting streaming recognition");

        let _streaming_config: WasmStreamingConfig = config
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("Config parsing error: {e}")))?;

        self.streaming_active = true;
        console_log!("Streaming recognition started");

        Ok(())
    }

    /// Stream audio chunk for recognition
    #[wasm_bindgen]
    pub async fn stream_audio(&self, audio_chunk: &[u8]) -> Result<JsValue, JsValue> {
        if !self.streaming_active {
            return Err(JsValue::from_str("Streaming not active"));
        }

        console_log!("Processing audio chunk ({} bytes)", audio_chunk.len());

        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Pipeline not initialized"))?;

        match pipeline.recognize_bytes(audio_chunk).await {
            Ok(result) => {
                console_log!("Chunk recognition completed");

                let wasm_result = if let Some(transcript) = result.transcription {
                    WasmRecognitionResult {
                        text: transcript.text.clone(),
                        confidence: transcript.confidence,
                        language: Some(transcript.language.to_string()),
                        segments: transcript
                            .word_timestamps
                            .iter()
                            .map(|word| WasmSegment {
                                start_time: word.start_time as f64,
                                end_time: word.end_time as f64,
                                text: word.word.clone(),
                                confidence: word.confidence,
                                no_speech_prob: 0.0, // Not available in word timestamps
                            })
                            .collect(),
                        processing_time: 0.0, // Not measured for streaming chunks
                        metadata: HashMap::new(),
                    }
                } else {
                    // No transcription available
                    WasmRecognitionResult {
                        text: "".to_string(),
                        confidence: 0.0,
                        language: None,
                        segments: vec![],
                        processing_time: 0.0,
                        metadata: HashMap::new(),
                    }
                };

                JsValue::from_serde(&wasm_result)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
            }
            Err(e) => {
                console_error!("Chunk recognition failed: {}", e);
                Err(JsValue::from_str(&format!("Chunk recognition failed: {e}")))
            }
        }
    }

    /// Stop streaming recognition
    #[wasm_bindgen]
    pub fn stop_streaming(&mut self) {
        console_log!("Stopping streaming recognition");
        self.streaming_active = false;
    }

    /// Get list of supported models
    #[wasm_bindgen]
    pub async fn get_supported_models(&self) -> Result<JsValue, JsValue> {
        console_log!("Getting supported models");

        let models = vec![
            "whisper-tiny",
            "whisper-base",
            "whisper-small",
            "whisper-medium",
            "whisper-large-v3",
        ];

        JsValue::from_serde(&models)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
    }

    /// Get list of supported languages
    #[wasm_bindgen]
    pub async fn get_supported_languages(&self) -> Result<JsValue, JsValue> {
        console_log!("Getting supported languages");

        let languages = vec![
            "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "nl", "sv", "no", "da",
            "fi", "pl", "tr", "ar", "hi", "th",
        ];

        JsValue::from_serde(&languages)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
    }

    /// Get library version
    #[wasm_bindgen]
    pub fn get_version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Get recognizer capabilities
    #[wasm_bindgen]
    pub fn get_capabilities(&self) -> JsValue {
        let capabilities = serde_json::json!({
            "recognition": true,
            "streaming": true,
            "real_time": true,
            "web_workers": true,
            "microphone": false, // Requires JavaScript integration
            "formats": ["wav", "mp3", "flac", "ogg", "m4a"],
            "models": ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large-v3"],
            "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
            "features": ["vad", "confidence_scoring", "segment_timestamps", "language_detection"]
        });

        JsValue::from_serde(&capabilities).unwrap_or(JsValue::NULL)
    }

    /// Check if streaming is active
    #[wasm_bindgen]
    pub fn is_streaming_active(&self) -> bool {
        self.streaming_active
    }

    /// Get current configuration
    #[wasm_bindgen]
    pub fn get_current_config(&self) -> JsValue {
        match &self.current_config {
            Some(config) => JsValue::from_serde(config).unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }

    /// Switch to a different model
    #[wasm_bindgen]
    pub async fn switch_model(&mut self, model_name: &str) -> Result<(), JsValue> {
        console_log!("Switching to model: {}", model_name);

        if let Some(mut config) = self.current_config.clone() {
            config.model_name = Some(model_name.to_string());

            // Reinitialize with new model
            let js_config = JsValue::from_serde(&config)
                .map_err(|e| JsValue::from_str(&format!("Config serialization error: {e}")))?;

            self.initialize(js_config).await
        } else {
            Err(JsValue::from_str("No current configuration found"))
        }
    }
}

impl Default for WasmVoirsRecognizer {
    fn default() -> Self {
        Self::new()
    }
}
