use crate::config::{SingingConfig, SingingConfigBuilder};
use crate::core::SingingEngine;
use crate::score::MusicalScore;
use crate::techniques::SingingTechnique;
use crate::types::{
    NoteEvent, QualitySettings, SingingRequest, SingingResponse, VoiceCharacteristics, VoiceType,
};
use js_sys::{Array, Float32Array, Promise};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Mutex;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{console, AudioBuffer, AudioContext};

#[derive(Debug, Error)]
pub enum WasmError {
    #[error("Engine initialization failed: {0}")]
    EngineInitFailed(String),
    #[error("Synthesis failed: {0}")]
    SynthesisFailed(String),
    #[error("Audio context error: {0}")]
    AudioContextError(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("JavaScript error: {0}")]
    JsError(String),
}

impl From<WasmError> for JsValue {
    fn from(error: WasmError) -> Self {
        JsValue::from_str(&error.to_string())
    }
}

// WASM bindings for the singing engine
#[wasm_bindgen]
pub struct WasmSingingEngine {
    engine: Arc<Mutex<Option<SingingEngine>>>,
    config: SingingConfig,
}

#[wasm_bindgen]
impl WasmSingingEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmSingingEngine, JsValue> {
        console_error_panic_hook::set_once();

        let config = SingingConfig::default();

        // We'll create the engine lazily since async constructors aren't supported in WASM
        Ok(Self {
            engine: Arc::new(Mutex::new(None)),
            config,
        })
    }

    #[wasm_bindgen(js_name = "initializeAsync")]
    pub async fn initialize_async(&mut self) -> Result<(), JsValue> {
        let engine = SingingEngine::new(self.config.clone())
            .await
            .map_err(|e| WasmError::EngineInitFailed(e.to_string()))?;

        *self.engine.lock().await = Some(engine);

        web_sys::console::log_1(&"VoiRS Singing Engine initialized successfully".into());
        Ok(())
    }

    #[wasm_bindgen(js_name = "synthesizeNote")]
    pub async fn synthesize_note(
        &self,
        midi_note: u8,
        duration: f32,
        voice_type: &str,
    ) -> Result<Float32Array, JsValue> {
        let voice_type = parse_voice_type(voice_type)?;

        // Create a simple single-note score
        let mut score = MusicalScore::new("WASM Generated".to_string(), "VoiRS".to_string());

        // Create note event
        let note_event = NoteEvent {
            note: midi_to_note_name(midi_note),
            octave: (midi_note / 12).saturating_sub(1),
            frequency: midi_to_frequency(midi_note),
            duration,
            velocity: 1.0,
            vibrato: 0.0,
            lyric: Some("la".to_string()),
            phonemes: vec!["l".to_string(), "a".to_string()],
            expression: crate::types::Expression::Neutral,
            timing_offset: 0.0,
            breath_before: 0.0,
            legato: false,
            articulation: crate::types::Articulation::Normal,
        };

        let musical_note = crate::score::MusicalNote::new(note_event, 0.0, duration);
        score.add_note(musical_note);

        let voice_characteristics = VoiceCharacteristics::for_voice_type(voice_type);
        let technique = crate::techniques::SingingTechnique::default();
        let quality = QualitySettings {
            quality_level: 6,
            high_quality_pitch: true,
            advanced_vibrato: true,
            breath_modeling: true,
            formant_modeling: false,
            fft_size: 2048,
            hop_size: 512,
        };

        let request = SingingRequest {
            score,
            voice: voice_characteristics,
            technique,
            effects: vec![],
            sample_rate: 44100,
            target_duration: None,
            quality,
        };

        let engine_guard = self.engine.lock().await;
        let engine = engine_guard
            .as_ref()
            .ok_or_else(|| WasmError::EngineInitFailed("Engine not initialized".to_string()))?;
        let response = engine
            .synthesize(request)
            .await
            .map_err(|e| WasmError::SynthesisFailed(e.to_string()))?;

        Ok(Float32Array::from(&response.audio[..]))
    }

    #[wasm_bindgen(js_name = "synthesizeScore")]
    pub async fn synthesize_score(
        &self,
        score_json: &str,
        voice_type: &str,
    ) -> Result<Float32Array, JsValue> {
        let voice_type = parse_voice_type(voice_type)?;

        let score_data: WasmMusicalScore = serde_json::from_str(score_json)
            .map_err(|e| WasmError::InvalidInput(format!("Invalid score JSON: {e}")))?;

        let score = score_data.to_musical_score()?;
        let request = SingingRequest {
            score,
            voice: VoiceCharacteristics::for_voice_type(voice_type),
            technique: SingingTechnique::default(),
            effects: vec![],
            sample_rate: 44100,
            target_duration: None,
            quality: QualitySettings::default(),
        };

        let engine_guard = self.engine.lock().await;
        let engine = engine_guard
            .as_ref()
            .ok_or_else(|| WasmError::EngineInitFailed("Engine not initialized".to_string()))?;
        let response = engine
            .synthesize(request)
            .await
            .map_err(|e| WasmError::SynthesisFailed(e.to_string()))?;

        Ok(Float32Array::from(&response.audio[..]))
    }

    #[wasm_bindgen(js_name = "createAudioBuffer")]
    pub fn create_audio_buffer(
        &self,
        audio_context: &AudioContext,
        audio_data: &Float32Array,
        sample_rate: f32,
    ) -> Result<AudioBuffer, JsValue> {
        let length = audio_data.length();
        let buffer = audio_context.create_buffer(1, length, sample_rate)?;

        let mut channel_data = buffer.get_channel_data(0)?;
        let audio_slice: Vec<f32> = audio_data.to_vec();
        channel_data.copy_from_slice(&audio_slice);

        Ok(buffer)
    }

    #[wasm_bindgen(js_name = "getVoiceTypes")]
    pub fn get_voice_types() -> Array {
        let voice_types = Array::new();
        voice_types.push(&JsValue::from_str("soprano"));
        voice_types.push(&JsValue::from_str("alto"));
        voice_types.push(&JsValue::from_str("tenor"));
        voice_types.push(&JsValue::from_str("bass"));
        voice_types.push(&JsValue::from_str("mezzo_soprano"));
        voice_types.push(&JsValue::from_str("baritone"));
        voice_types
    }

    #[wasm_bindgen(js_name = "setConfig")]
    pub async fn set_config(&mut self, config_json: &str) -> Result<(), JsValue> {
        let config: WasmSingingConfig = serde_json::from_str(config_json)
            .map_err(|e| WasmError::InvalidInput(format!("Invalid config JSON: {e}")))?;

        self.config = config.to_singing_config()?;

        // Recreate engine with new config
        let engine = SingingEngine::new(self.config.clone())
            .await
            .map_err(|e| WasmError::EngineInitFailed(e.to_string()))?;

        *self.engine.lock().await = Some(engine);

        Ok(())
    }

    #[wasm_bindgen(js_name = "getVersion")]
    pub fn get_version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

// Helper functions for WebAssembly
#[wasm_bindgen]
pub struct WasmAudioPlayer {
    context: AudioContext,
    sample_rate: f32,
}

#[wasm_bindgen]
impl WasmAudioPlayer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmAudioPlayer, JsValue> {
        let context = AudioContext::new()?;
        let sample_rate = context.sample_rate();

        Ok(Self {
            context,
            sample_rate,
        })
    }

    #[wasm_bindgen(js_name = "playAudio")]
    pub fn play_audio(&self, audio_data: &Float32Array) -> Result<(), JsValue> {
        let buffer = self
            .context
            .create_buffer(1, audio_data.length(), self.sample_rate)?;
        let mut channel_data = buffer.get_channel_data(0)?;
        let audio_slice: Vec<f32> = audio_data.to_vec();
        channel_data.copy_from_slice(&audio_slice);

        let source = self.context.create_buffer_source()?;
        source.set_buffer(Some(&buffer));
        source.connect_with_audio_node(&self.context.destination())?;
        source.start()?;

        Ok(())
    }

    #[wasm_bindgen(js_name = "getSampleRate")]
    pub fn get_sample_rate(&self) -> f32 {
        self.sample_rate
    }

    #[wasm_bindgen(js_name = "getContext")]
    pub fn get_context(&self) -> AudioContext {
        self.context.clone()
    }
}

// Real-time synthesis for web applications
#[wasm_bindgen]
pub struct WasmRealtimeSynthesizer {
    engine: Arc<Mutex<Option<SingingEngine>>>,
    audio_context: AudioContext,
    is_playing: bool,
}

#[wasm_bindgen]
impl WasmRealtimeSynthesizer {
    #[wasm_bindgen(constructor)]
    pub fn new(audio_context: AudioContext) -> Result<WasmRealtimeSynthesizer, JsValue> {
        Ok(Self {
            engine: Arc::new(Mutex::new(None)),
            audio_context,
            is_playing: false,
        })
    }

    #[wasm_bindgen(js_name = "startRealtime")]
    pub async fn start_realtime(&mut self) -> Result<(), JsValue> {
        let config = SingingConfigBuilder::new().build();

        let engine = SingingEngine::new(config)
            .await
            .map_err(|e| WasmError::EngineInitFailed(e.to_string()))?;

        *self.engine.lock().await = Some(engine);
        self.is_playing = true;

        Ok(())
    }

    #[wasm_bindgen(js_name = "playNote")]
    pub async fn play_note(
        &self,
        midi_note: u8,
        duration: f32,
        voice_type: &str,
    ) -> Result<(), JsValue> {
        if !self.is_playing {
            return Err(
                WasmError::InvalidInput("Realtime synthesizer not started".to_string()).into(),
            );
        }

        let voice_type = parse_voice_type(voice_type)?;

        // Create and synthesize note
        let mut score = MusicalScore::new("Realtime".to_string(), "VoiRS".to_string());

        let note_event = NoteEvent {
            note: midi_to_note_name(midi_note),
            octave: (midi_note / 12).saturating_sub(1),
            frequency: midi_to_frequency(midi_note),
            duration,
            velocity: 1.0,
            vibrato: 0.0,
            lyric: Some("la".to_string()),
            phonemes: vec!["l".to_string(), "a".to_string()],
            expression: crate::types::Expression::Neutral,
            timing_offset: 0.0,
            breath_before: 0.0,
            legato: false,
            articulation: crate::types::Articulation::Normal,
        };

        let musical_note = crate::score::MusicalNote::new(note_event, 0.0, duration);
        score.add_note(musical_note);

        let voice_characteristics = VoiceCharacteristics::for_voice_type(voice_type);
        let technique = crate::techniques::SingingTechnique::default();
        let quality = QualitySettings {
            quality_level: 6,
            high_quality_pitch: true,
            advanced_vibrato: true,
            breath_modeling: true,
            formant_modeling: false,
            fft_size: 2048,
            hop_size: 512,
        };

        let request = SingingRequest {
            score,
            voice: voice_characteristics,
            technique,
            effects: vec![],
            sample_rate: 44100,
            target_duration: None,
            quality,
        };

        let engine_guard = self.engine.lock().await;
        let engine = engine_guard
            .as_ref()
            .ok_or_else(|| WasmError::EngineInitFailed("Engine not initialized".to_string()))?;
        let response = engine
            .synthesize(request)
            .await
            .map_err(|e| WasmError::SynthesisFailed(e.to_string()))?;

        // Play immediately
        let audio_data = Float32Array::from(&response.audio[..]);
        let buffer = self.audio_context.create_buffer(
            1,
            audio_data.length(),
            self.audio_context.sample_rate(),
        )?;
        let mut channel_data = buffer.get_channel_data(0)?;
        let audio_slice: Vec<f32> = audio_data.to_vec();
        channel_data.copy_from_slice(&audio_slice);

        let source = self.audio_context.create_buffer_source()?;
        source.set_buffer(Some(&buffer));
        source.connect_with_audio_node(&self.audio_context.destination())?;
        source.start()?;

        Ok(())
    }

    #[wasm_bindgen(js_name = "stop")]
    pub fn stop(&mut self) {
        self.is_playing = false;
    }
}

// JavaScript-compatible types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WasmMusicalScore {
    notes: Vec<WasmMusicalNote>,
    tempo_bpm: f32,
    time_signature: [u32; 2],
    key_signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WasmMusicalNote {
    midi_note: u8,
    duration: f32,
    velocity: f32,
    start_time: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WasmSingingConfig {
    sample_rate: u32,
    buffer_size: usize,
    quality_level: String,
    enable_effects: bool,
    max_voices: usize,
}

impl WasmMusicalScore {
    fn to_musical_score(&self) -> Result<MusicalScore, WasmError> {
        let mut score = MusicalScore::new("WASM Score".to_string(), "User".to_string());

        // Add notes
        for note in &self.notes {
            let note_event = NoteEvent {
                note: midi_to_note_name(note.midi_note),
                octave: (note.midi_note / 12).saturating_sub(1),
                frequency: midi_to_frequency(note.midi_note),
                duration: note.duration,
                velocity: note.velocity,
                vibrato: 0.0,
                lyric: Some("la".to_string()),
                phonemes: vec!["l".to_string(), "a".to_string()],
                expression: crate::types::Expression::Neutral,
                timing_offset: 0.0,
                breath_before: 0.0,
                legato: false,
                articulation: crate::types::Articulation::Normal,
            };

            let musical_note =
                crate::score::MusicalNote::new(note_event, note.start_time, note.duration);
            score.add_note(musical_note);
        }

        Ok(score)
    }
}

impl WasmSingingConfig {
    fn to_singing_config(&self) -> Result<SingingConfig, WasmError> {
        let quality = match self.quality_level.as_str() {
            "low" => QualitySettings {
                quality_level: 3,
                high_quality_pitch: false,
                advanced_vibrato: false,
                breath_modeling: false,
                formant_modeling: false,
                fft_size: 1024,
                hop_size: 256,
            },
            "medium" => QualitySettings {
                quality_level: 6,
                high_quality_pitch: true,
                advanced_vibrato: true,
                breath_modeling: true,
                formant_modeling: false,
                fft_size: 2048,
                hop_size: 512,
            },
            "high" => QualitySettings {
                quality_level: 10,
                high_quality_pitch: true,
                advanced_vibrato: true,
                breath_modeling: true,
                formant_modeling: true,
                fft_size: 4096,
                hop_size: 1024,
            },
            _ => return Err(WasmError::InvalidInput("Invalid quality level".to_string())),
        };

        let config = SingingConfigBuilder::new().build();

        Ok(config)
    }
}

// Utility functions
fn midi_to_note_name(midi_note: u8) -> String {
    let note_names = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let note = (midi_note % 12) as usize;
    note_names[note].to_string()
}

fn midi_to_frequency(midi_note: u8) -> f32 {
    440.0 * 2_f32.powf((midi_note as f32 - 69.0) / 12.0)
}

fn parse_voice_type(voice_type_str: &str) -> Result<VoiceType, WasmError> {
    match voice_type_str.to_lowercase().as_str() {
        "soprano" => Ok(VoiceType::Soprano),
        "alto" => Ok(VoiceType::Alto),
        "tenor" => Ok(VoiceType::Tenor),
        "bass" => Ok(VoiceType::Bass),
        "mezzo_soprano" => Ok(VoiceType::MezzoSoprano),
        "baritone" => Ok(VoiceType::Baritone),
        _ => Err(WasmError::InvalidInput(format!(
            "Unknown voice type: {}",
            voice_type_str
        ))),
    }
}

// JavaScript utilities for easier integration
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn init_logging() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"VoiRS Singing WASM module initialized".into());
}

// Performance monitoring for web applications
#[wasm_bindgen]
pub struct WasmPerformanceMonitor {
    start_time: f64,
    synthesis_times: Vec<f64>,
}

#[wasm_bindgen]
impl WasmPerformanceMonitor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            start_time: js_sys::Date::now(),
            synthesis_times: Vec::new(),
        }
    }

    #[wasm_bindgen(js_name = "recordSynthesisTime")]
    pub fn record_synthesis_time(&mut self, time_ms: f64) {
        self.synthesis_times.push(time_ms);
    }

    #[wasm_bindgen(js_name = "getAverageSynthesisTime")]
    pub fn get_average_synthesis_time(&self) -> f64 {
        if self.synthesis_times.is_empty() {
            0.0
        } else {
            self.synthesis_times.iter().sum::<f64>() / self.synthesis_times.len() as f64
        }
    }

    #[wasm_bindgen(js_name = "getTotalUptime")]
    pub fn get_total_uptime(&self) -> f64 {
        js_sys::Date::now() - self.start_time
    }

    #[wasm_bindgen(js_name = "reset")]
    pub fn reset(&mut self) {
        self.start_time = js_sys::Date::now();
        self.synthesis_times.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_wasm_engine_creation() {
        let engine = WasmSingingEngine::new();
        assert!(engine.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_voice_type_parsing() {
        assert!(parse_voice_type("soprano").is_ok());
        assert!(parse_voice_type("bass").is_ok());
        assert!(parse_voice_type("invalid").is_err());
    }

    #[wasm_bindgen_test]
    fn test_performance_monitor() {
        let mut monitor = WasmPerformanceMonitor::new();
        monitor.record_synthesis_time(100.0);
        monitor.record_synthesis_time(200.0);

        assert_eq!(monitor.get_average_synthesis_time(), 150.0);
    }

    #[wasm_bindgen_test]
    fn test_wasm_config_serialization() {
        let config = WasmSingingConfig {
            sample_rate: 44100,
            buffer_size: 1024,
            quality_level: "high".to_string(),
            enable_effects: true,
            max_voices: 4,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: WasmSingingConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.sample_rate, deserialized.sample_rate);
        assert_eq!(config.quality_level, deserialized.quality_level);
    }

    #[wasm_bindgen_test]
    fn test_musical_score_conversion() {
        let wasm_score = WasmMusicalScore {
            notes: vec![WasmMusicalNote {
                midi_note: 60,
                duration: 1.0,
                velocity: 0.8,
                start_time: 0.0,
            }],
            tempo_bpm: 120.0,
            time_signature: [4, 4],
            key_signature: "C".to_string(),
        };

        let musical_score = wasm_score.to_musical_score();
        assert!(musical_score.is_ok());
    }
}
