//! WebAssembly bindings for browser-based emotion processing
//!
//! This module provides WebAssembly bindings that allow emotion processing
//! to run in web browsers, enabling real-time emotion control and analysis
//! in client-side applications.

use crate::{
    core::EmotionProcessor,
    types::{EmotionDimensions, EmotionIntensity, EmotionParameters, EmotionVector},
    Result,
};
use js_sys::{Array, Object, Uint8Array};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use web_sys::AudioContext;

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

/// WebAssembly-compatible emotion configuration
#[derive(Serialize, Deserialize, Clone)]
pub struct WasmEmotionConfig {
    /// Enable real-time processing
    pub enable_realtime: Option<bool>,
    /// Cache size for emotion parameters
    pub cache_size: Option<usize>,
    /// Buffer size for audio processing
    pub buffer_size: Option<usize>,
}

/// WebAssembly-compatible emotion parameters
#[derive(Serialize, Deserialize, Clone)]
pub struct WasmEmotionParameters {
    /// Arousal level (0.0-1.0)
    pub arousal: f32,
    /// Valence level (-1.0 to 1.0)
    pub valence: f32,
    /// Dominance level (0.0-1.0)
    pub dominance: f32,
    /// Intensity (0.0-1.0)
    pub intensity: f32,
    /// Pitch shift factor
    pub pitch_shift: f32,
    /// Speaking rate multiplier
    pub speaking_rate: f32,
}

/// WebAssembly emotion recognition result
#[derive(Serialize, Deserialize)]
pub struct WasmEmotionRecognitionResult {
    /// Detected emotion label
    pub emotion: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Emotion dimensions
    pub dimensions: WasmEmotionParameters,
    /// Analysis metadata
    pub metadata: HashMap<String, String>,
}

/// Main WebAssembly emotion processor
#[wasm_bindgen]
pub struct WasmEmotionProcessor {
    processor: Option<EmotionProcessor>,
    audio_context: Option<AudioContext>,
    current_emotion: Option<EmotionParameters>,
}

#[wasm_bindgen]
impl WasmEmotionProcessor {
    /// Create new WebAssembly emotion processor
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_log!("Creating new WasmEmotionProcessor");
        utils::set_panic_hook();

        Self {
            processor: None,
            audio_context: None,
            current_emotion: None,
        }
    }

    /// Initialize the emotion processor with configuration
    #[wasm_bindgen]
    pub async fn initialize(&mut self, config: JsValue) -> std::result::Result<(), JsValue> {
        console_log!("Initializing WasmEmotionProcessor");

        let _wasm_config: WasmEmotionConfig = config
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("Config parsing error: {e}")))?;

        // Initialize emotion processor
        match EmotionProcessor::new() {
            Ok(processor) => {
                console_log!("Emotion processor initialized successfully");
                self.processor = Some(processor);
            }
            Err(e) => {
                console_error!("Failed to initialize emotion processor: {}", e);
                return Err(JsValue::from_str(&format!(
                    "Processor initialization failed: {e}"
                )));
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

    /// Process audio with emotion parameters (simplified implementation)
    #[wasm_bindgen]
    pub async fn process_audio(
        &mut self,
        audio_data: &[u8],
        emotion_params: JsValue,
    ) -> std::result::Result<JsValue, JsValue> {
        let _processor = self
            .processor
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Processor not initialized"))?;

        let wasm_params: WasmEmotionParameters = emotion_params
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("Parameters parsing error: {e}")))?;

        console_log!("Processing audio with emotion parameters");

        // Convert audio data to f32 samples (assuming 16-bit PCM input)
        let mut audio_samples = Vec::with_capacity(audio_data.len() / 2);
        for chunk in audio_data.chunks_exact(2) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
            audio_samples.push(sample);
        }

        // Create emotion parameters (simplified)
        let mut emotions_map = std::collections::HashMap::new();
        emotions_map.insert(
            crate::types::Emotion::Neutral,
            EmotionIntensity::new(wasm_params.intensity),
        );

        let emotion_vector = EmotionVector {
            emotions: emotions_map,
            dimensions: EmotionDimensions {
                arousal: wasm_params.arousal,
                valence: wasm_params.valence,
                dominance: wasm_params.dominance,
            },
        };

        let emotion_params = EmotionParameters {
            emotion_vector,
            duration_ms: Some(1000),
            fade_in_ms: Some(50),
            fade_out_ms: Some(50),
            pitch_shift: wasm_params.pitch_shift,
            tempo_scale: wasm_params.speaking_rate,
            energy_scale: 1.0,
            breathiness: 0.0,
            roughness: 0.0,
            custom_params: std::collections::HashMap::new(),
        };

        // For now, apply simple pitch shift to audio samples as demonstration
        let mut processed_audio = audio_samples.clone();
        if (wasm_params.pitch_shift - 1.0).abs() > 0.01 {
            for sample in &mut processed_audio {
                *sample *= wasm_params.pitch_shift;
                *sample = sample.clamp(-1.0, 1.0);
            }
        }

        // Apply simple volume scaling based on arousal
        let volume_factor = 0.5 + (wasm_params.arousal * 0.5);
        for sample in &mut processed_audio {
            *sample *= volume_factor;
            *sample = sample.clamp(-1.0, 1.0);
        }

        console_log!("Audio processed successfully with emotion parameters");

        // Convert back to 16-bit PCM
        let mut output_bytes = Vec::with_capacity(processed_audio.len() * 2);
        for sample in processed_audio {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            output_bytes.extend_from_slice(&sample_i16.to_le_bytes());
        }

        // Store current emotion for reference
        self.current_emotion = Some(emotion_params);

        JsValue::from_serde(&output_bytes)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
    }

    /// Recognize emotion from text (simplified implementation)
    #[wasm_bindgen]
    pub async fn recognize_emotion_from_text(
        &self,
        text: &str,
    ) -> std::result::Result<JsValue, JsValue> {
        console_log!("Recognizing emotion from text: {}", text);

        // Simple heuristic-based emotion recognition
        let text_lower = text.to_lowercase();

        let (emotion, arousal, valence, dominance) = if text_lower.contains("happy")
            || text_lower.contains("joy")
            || text_lower.contains("excited")
        {
            ("happy", 0.8, 0.9, 0.7)
        } else if text_lower.contains("sad")
            || text_lower.contains("cry")
            || text_lower.contains("depressed")
        {
            ("sad", 0.3, -0.8, 0.2)
        } else if text_lower.contains("angry")
            || text_lower.contains("mad")
            || text_lower.contains("furious")
        {
            ("angry", 0.9, -0.7, 0.9)
        } else if text_lower.contains("calm")
            || text_lower.contains("peaceful")
            || text_lower.contains("relaxed")
        {
            ("calm", 0.2, 0.5, 0.6)
        } else if text_lower.contains("fear")
            || text_lower.contains("scared")
            || text_lower.contains("afraid")
        {
            ("fear", 0.8, -0.6, 0.1)
        } else {
            ("neutral", 0.5, 0.0, 0.5)
        };

        let wasm_result = WasmEmotionRecognitionResult {
            emotion: emotion.to_string(),
            confidence: 0.75, // Fixed confidence for demo
            dimensions: WasmEmotionParameters {
                arousal,
                valence,
                dominance,
                intensity: arousal, // Use arousal as intensity
                pitch_shift: 1.0 + (arousal - 0.5) * 0.2,
                speaking_rate: 1.0 + (arousal - 0.5) * 0.3,
            },
            metadata: {
                let mut map = HashMap::new();
                map.insert("text_length".to_string(), text.len().to_string());
                map.insert("method".to_string(), "heuristic".to_string());
                map
            },
        };

        JsValue::from_serde(&wasm_result)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
    }

    /// Interpolate between two emotions
    #[wasm_bindgen]
    pub fn interpolate_emotions(
        &self,
        from_emotion: JsValue,
        to_emotion: JsValue,
        progress: f32,
    ) -> std::result::Result<JsValue, JsValue> {
        let from_params: WasmEmotionParameters = from_emotion
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("From emotion parsing error: {e}")))?;

        let to_params: WasmEmotionParameters = to_emotion
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("To emotion parsing error: {e}")))?;

        let progress = progress.clamp(0.0, 1.0);

        // Linear interpolation between emotion parameters
        let interpolated = WasmEmotionParameters {
            arousal: from_params.arousal + (to_params.arousal - from_params.arousal) * progress,
            valence: from_params.valence + (to_params.valence - from_params.valence) * progress,
            dominance: from_params.dominance
                + (to_params.dominance - from_params.dominance) * progress,
            intensity: from_params.intensity
                + (to_params.intensity - from_params.intensity) * progress,
            pitch_shift: from_params.pitch_shift
                + (to_params.pitch_shift - from_params.pitch_shift) * progress,
            speaking_rate: from_params.speaking_rate
                + (to_params.speaking_rate - from_params.speaking_rate) * progress,
        };

        console_log!("Interpolated emotions with progress: {}", progress);

        JsValue::from_serde(&interpolated)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
    }

    /// Get current emotion state
    #[wasm_bindgen]
    pub fn get_current_emotion(&self) -> JsValue {
        match &self.current_emotion {
            Some(emotion) => {
                // Get the first emotion's intensity (simplified)
                let intensity = emotion
                    .emotion_vector
                    .emotions
                    .values()
                    .next()
                    .map(|i| i.value())
                    .unwrap_or(0.5);

                let wasm_emotion = WasmEmotionParameters {
                    arousal: emotion.emotion_vector.dimensions.arousal,
                    valence: emotion.emotion_vector.dimensions.valence,
                    dominance: emotion.emotion_vector.dimensions.dominance,
                    intensity,
                    pitch_shift: emotion.pitch_shift,
                    speaking_rate: emotion.tempo_scale,
                };
                JsValue::from_serde(&wasm_emotion).unwrap_or(JsValue::NULL)
            }
            None => JsValue::NULL,
        }
    }

    /// Get processor capabilities
    #[wasm_bindgen]
    pub fn get_capabilities(&self) -> JsValue {
        let capabilities = serde_json::json!({
            "emotion_processing": true,
            "emotion_recognition": true,
            "interpolation": true,
            "real_time": true,
            "audio_processing": true,
            "supported_audio_formats": ["pcm16"],
            "supported_sample_rates": [16000, 22050, 44100, 48000],
            "max_audio_channels": 2
        });

        JsValue::from_serde(&capabilities).unwrap_or(JsValue::NULL)
    }

    /// Get processor version
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
    console_log!("Initializing WASM logger for emotion processing");
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
        "module": "voirs-emotion"
    });

    JsValue::from_serde(&usage).unwrap_or(JsValue::NULL)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_wasm_emotion_processor_creation() {
        let processor = WasmEmotionProcessor::new();
        assert!(processor.processor.is_none());
        assert!(processor.current_emotion.is_none());
    }

    #[wasm_bindgen_test]
    async fn test_wasm_emotion_capabilities() {
        let processor = WasmEmotionProcessor::new();
        let capabilities = processor.get_capabilities();
        assert!(!capabilities.is_null());
    }

    #[wasm_bindgen_test]
    async fn test_wasm_emotion_version() {
        let processor = WasmEmotionProcessor::new();
        let version = processor.get_version();
        assert!(!version.is_empty());
    }

    #[wasm_bindgen_test]
    async fn test_emotion_interpolation() {
        let processor = WasmEmotionProcessor::new();

        let from_emotion = serde_json::json!({
            "arousal": 0.2,
            "valence": -0.5,
            "dominance": 0.3,
            "intensity": 0.4,
            "pitch_shift": 0.9,
            "speaking_rate": 0.8
        });

        let to_emotion = serde_json::json!({
            "arousal": 0.8,
            "valence": 0.5,
            "dominance": 0.7,
            "intensity": 0.9,
            "pitch_shift": 1.2,
            "speaking_rate": 1.1
        });

        let result = processor.interpolate_emotions(
            JsValue::from_serde(&from_emotion).unwrap(),
            JsValue::from_serde(&to_emotion).unwrap(),
            0.5,
        );

        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    async fn test_memory_usage() {
        let usage = get_wasm_memory_usage();
        assert!(!usage.is_null());
    }
}
