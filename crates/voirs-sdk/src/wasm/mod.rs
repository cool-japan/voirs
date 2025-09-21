#[allow(deprecated, dead_code)]
use crate::prelude::*;
use crate::{VoirsPipeline, VoirsPipelineBuilder};
use js_sys::Uint8Array;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
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

#[derive(Serialize, Deserialize, Clone)]
pub struct WasmSynthesisConfig {
    pub voice_id: Option<String>,
    pub language: Option<String>,
    pub sample_rate: Option<u32>,
    pub quality: Option<String>,
    pub speed: Option<f32>,
    pub pitch: Option<f32>,
    pub volume: Option<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct WasmSynthesisResult {
    pub audio_data: Vec<u8>,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration: f64,
    pub metadata: HashMap<String, String>,
}

#[derive(Serialize, Deserialize)]
pub struct WasmStreamingConfig {
    pub chunk_size: Option<usize>,
    pub buffer_size: Option<usize>,
    pub latency_target: Option<f32>,
    pub quality_adaptive: Option<bool>,
}

#[wasm_bindgen]
pub struct WasmVoirsPipeline {
    pipeline: Option<VoirsPipeline>,
    audio_context: Option<AudioContext>,
    streaming_active: bool,
}

#[wasm_bindgen]
impl WasmVoirsPipeline {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_log!("Creating new WasmVoirsPipeline");
        utils::set_panic_hook();

        Self {
            pipeline: None,
            audio_context: None,
            streaming_active: false,
        }
    }

    #[wasm_bindgen]
    pub async fn initialize(&mut self, config: JsValue) -> std::result::Result<(), JsValue> {
        console_log!("Initializing WasmVoirsPipeline");

        let config: WasmSynthesisConfig = config
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("Config parsing error: {e}")))?;

        let mut builder = VoirsPipelineBuilder::new();

        if let Some(voice_id) = config.voice_id {
            builder = builder.with_voice(&voice_id);
        }

        if let Some(language_str) = config.language {
            let language = match language_str.as_str() {
                "en-us" | "en_us" => Some(LanguageCode::EnUs),
                "en-gb" | "en_gb" => Some(LanguageCode::EnGb),
                "ja-jp" | "ja_jp" => Some(LanguageCode::JaJp),
                "es-es" | "es_es" => Some(LanguageCode::EsEs),
                "es-mx" | "es_mx" => Some(LanguageCode::EsMx),
                "fr-fr" | "fr_fr" => Some(LanguageCode::FrFr),
                "de-de" | "de_de" => Some(LanguageCode::DeDe),
                "zh-cn" | "zh_cn" => Some(LanguageCode::ZhCn),
                "pt-br" | "pt_br" => Some(LanguageCode::PtBr),
                "ru-ru" | "ru_ru" => Some(LanguageCode::RuRu),
                "it-it" | "it_it" => Some(LanguageCode::ItIt),
                "ko-kr" | "ko_kr" => Some(LanguageCode::KoKr),
                "nl-nl" | "nl_nl" => Some(LanguageCode::NlNl),
                "sv-se" | "sv_se" => Some(LanguageCode::SvSe),
                "no-no" | "no_no" => Some(LanguageCode::NoNo),
                "da-dk" | "da_dk" => Some(LanguageCode::DaDk),
                _ => None,
            };
            if let Some(lang) = language {
                builder = builder.with_language(lang);
            }
        }

        if let Some(sample_rate) = config.sample_rate {
            builder = builder.with_sample_rate(sample_rate);
        }

        if let Some(quality_str) = config.quality {
            let quality = match quality_str.to_lowercase().as_str() {
                "low" => Some(QualityLevel::Low),
                "medium" => Some(QualityLevel::Medium),
                "high" => Some(QualityLevel::High),
                "ultra" => Some(QualityLevel::Ultra),
                _ => None,
            };
            if let Some(qual) = quality {
                builder = builder.with_quality(qual);
            }
        }

        match builder.build().await {
            Ok(pipeline) => {
                console_log!("Pipeline initialized successfully");
                self.pipeline = Some(pipeline);

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
            Err(e) => {
                console_error!("Pipeline initialization failed: {}", e);
                Err(JsValue::from_str(&format!(
                    "Pipeline initialization failed: {e}"
                )))
            }
        }
    }

    #[wasm_bindgen]
    pub async fn synthesize(
        &self,
        text: &str,
        config: JsValue,
    ) -> std::result::Result<JsValue, JsValue> {
        console_log!("Synthesizing text: {}", text);

        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Pipeline not initialized"))?;

        let wasm_config: WasmSynthesisConfig = config
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("Config parsing error: {e}")))?;

        let mut synthesis_config = SynthesisConfig::default();

        if let Some(speed) = wasm_config.speed {
            synthesis_config.speaking_rate = speed;
        }

        if let Some(pitch) = wasm_config.pitch {
            synthesis_config.pitch_shift = pitch;
        }

        if let Some(volume) = wasm_config.volume {
            synthesis_config.volume_gain = volume;
        }

        match pipeline
            .synthesize_with_config(text, &synthesis_config)
            .await
        {
            Ok(audio_buffer) => {
                console_log!("Synthesis completed successfully");

                let audio_data = audio_buffer
                    .to_wav_bytes()
                    .map_err(|e| JsValue::from_str(&format!("WAV conversion error: {e}")))?;

                let result = WasmSynthesisResult {
                    audio_data,
                    sample_rate: audio_buffer.sample_rate(),
                    channels: audio_buffer.channels() as u16,
                    duration: audio_buffer.duration() as f64,
                    metadata: {
                        let meta = audio_buffer.metadata();
                        let mut map = HashMap::new();
                        map.insert("duration".to_string(), meta.duration.to_string());
                        map.insert(
                            "peak_amplitude".to_string(),
                            meta.peak_amplitude.to_string(),
                        );
                        map.insert("rms_amplitude".to_string(), meta.rms_amplitude.to_string());
                        map.insert("dynamic_range".to_string(), meta.dynamic_range.to_string());
                        map.insert("format".to_string(), format!("{:?}", meta.format));
                        map
                    },
                };

                JsValue::from_serde(&result)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
            }
            Err(e) => {
                console_error!("Synthesis failed: {}", e);
                Err(JsValue::from_str(&format!("Synthesis failed: {e}")))
            }
        }
    }

    #[wasm_bindgen]
    pub async fn play_audio(&self, audio_data: &[u8]) -> std::result::Result<(), JsValue> {
        console_log!("Playing audio data");

        let audio_context = self
            .audio_context
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Audio context not initialized"))?;

        let uint8_array = Uint8Array::from(audio_data);
        let array_buffer = uint8_array.buffer();

        let promise = audio_context.decode_audio_data(&array_buffer)?;
        let audio_buffer = JsFuture::from(promise).await?;

        let source = audio_context
            .create_buffer_source()
            .map_err(|_| JsValue::from_str("Failed to create buffer source"))?;
        let web_audio_buffer = audio_buffer.dyn_into::<web_sys::AudioBuffer>()?;
        source.set_buffer(Some(&web_audio_buffer));
        let destination = audio_context.destination();
        source.connect_with_audio_node(&destination)?;

        source.start()?;
        console_log!("Audio playback started");

        Ok(())
    }

    #[wasm_bindgen]
    pub async fn start_streaming(&mut self, config: JsValue) -> std::result::Result<(), JsValue> {
        console_log!("Starting streaming synthesis");

        let _streaming_config: WasmStreamingConfig = config
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("Config parsing error: {e}")))?;

        self.streaming_active = true;
        console_log!("Streaming synthesis started");

        Ok(())
    }

    #[wasm_bindgen]
    pub async fn stream_text(&self, text: &str) -> std::result::Result<JsValue, JsValue> {
        if !self.streaming_active {
            return Err(JsValue::from_str("Streaming not active"));
        }

        console_log!("Streaming text chunk: {}", text);

        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Pipeline not initialized"))?;

        match pipeline.synthesize(text).await {
            Ok(audio_buffer) => {
                console_log!("Text synthesis completed");

                let audio_data = audio_buffer
                    .to_wav_bytes()
                    .map_err(|e| JsValue::from_str(&format!("WAV conversion error: {e}")))?;

                // Return as single chunk for compatibility
                let chunks: Vec<Vec<u8>> = vec![audio_data];

                JsValue::from_serde(&chunks)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
            }
            Err(e) => {
                console_error!("Text streaming failed: {}", e);
                Err(JsValue::from_str(&format!("Text streaming failed: {e}")))
            }
        }
    }

    #[wasm_bindgen]
    pub fn stop_streaming(&mut self) {
        console_log!("Stopping streaming synthesis");
        self.streaming_active = false;
    }

    #[wasm_bindgen]
    pub async fn get_voices(&self) -> std::result::Result<JsValue, JsValue> {
        console_log!("Getting available voices");

        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Pipeline not initialized"))?;

        match pipeline.list_voices().await {
            Ok(voices) => {
                console_log!("Retrieved {} voices", voices.len());
                JsValue::from_serde(&voices)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
            }
            Err(e) => {
                console_error!("Failed to get voices: {}", e);
                Err(JsValue::from_str(&format!("Failed to get voices: {e}")))
            }
        }
    }

    #[wasm_bindgen]
    pub async fn switch_voice(&mut self, voice_id: &str) -> std::result::Result<(), JsValue> {
        console_log!("Switching to voice: {}", voice_id);

        let pipeline = self
            .pipeline
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Pipeline not initialized"))?;

        match pipeline.set_voice(voice_id).await {
            Ok(()) => {
                console_log!("Voice switched successfully");
                Ok(())
            }
            Err(e) => {
                console_error!("Voice switch failed: {}", e);
                Err(JsValue::from_str(&format!("Voice switch failed: {e}")))
            }
        }
    }

    #[wasm_bindgen]
    pub fn get_version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    #[wasm_bindgen]
    pub fn get_capabilities(&self) -> JsValue {
        let capabilities = serde_json::json!({
            "synthesis": true,
            "streaming": true,
            "voice_switching": true,
            "audio_playback": true,
            "real_time": true,
            "web_workers": true,
            "formats": ["wav", "raw"],
            "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
        });

        JsValue::from_serde(&capabilities).unwrap_or(JsValue::NULL)
    }
}

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

#[wasm_bindgen]
pub fn init_wasm_logger() {
    console_log!("Initializing WASM logger");
    wasm_logger::init(wasm_logger::Config::default());
}

#[wasm_bindgen]
pub fn get_wasm_memory_usage() -> JsValue {
    let memory = wasm_bindgen::memory()
        .dyn_into::<js_sys::WebAssembly::Memory>()
        .unwrap();

    let buffer = memory.buffer();
    let usage = serde_json::json!({
        "buffer_size": buffer.dyn_into::<js_sys::ArrayBuffer>().map(|ab| ab.byte_length()).unwrap_or(0),
        "available": true
    });

    JsValue::from_serde(&usage).unwrap_or(JsValue::NULL)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_wasm_pipeline_creation() {
        let pipeline = WasmVoirsPipeline::new();
        assert!(!pipeline.streaming_active);
        assert!(pipeline.pipeline.is_none());
    }

    #[wasm_bindgen_test]
    async fn test_wasm_capabilities() {
        let pipeline = WasmVoirsPipeline::new();
        let capabilities = pipeline.get_capabilities();
        assert!(!capabilities.is_null());
    }

    #[wasm_bindgen_test]
    async fn test_wasm_version() {
        let pipeline = WasmVoirsPipeline::new();
        let version = pipeline.get_version();
        assert!(!version.is_empty());
    }

    #[wasm_bindgen_test]
    async fn test_memory_usage() {
        let usage = get_wasm_memory_usage();
        assert!(!usage.is_null());
    }
}
