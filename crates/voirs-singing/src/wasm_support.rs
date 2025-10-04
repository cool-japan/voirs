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

/// Error types for WebAssembly singing synthesis operations.
///
/// These errors are converted to JavaScript values for interoperability with web applications.
#[derive(Debug, Error)]
pub enum WasmError {
    /// Engine initialization failed with the given error message
    #[error("Engine initialization failed: {0}")]
    EngineInitFailed(String),
    /// Synthesis operation failed with the given error message
    #[error("Synthesis failed: {0}")]
    SynthesisFailed(String),
    /// Web Audio API context error
    #[error("Audio context error: {0}")]
    AudioContextError(String),
    /// Invalid input parameters or data provided from JavaScript
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    /// Generic JavaScript interop error
    #[error("JavaScript error: {0}")]
    JsError(String),
}

impl From<WasmError> for JsValue {
    fn from(error: WasmError) -> Self {
        JsValue::from_str(&error.to_string())
    }
}

/// WebAssembly bindings for the VoiRS singing synthesis engine.
///
/// This struct provides JavaScript-compatible API for neural singing synthesis
/// in web browsers. It wraps the core `SingingEngine` with async-compatible
/// initialization and synthesis methods.
///
/// # Example (JavaScript)
/// ```javascript
/// const engine = new WasmSingingEngine();
/// await engine.initializeAsync();
/// const audio = await engine.synthesizeNote(60, 1.0, "soprano");
/// ```
#[wasm_bindgen]
pub struct WasmSingingEngine {
    engine: Arc<Mutex<Option<SingingEngine>>>,
    config: SingingConfig,
}

#[wasm_bindgen]
impl WasmSingingEngine {
    /// Creates a new WebAssembly singing engine instance.
    ///
    /// The engine is created in an uninitialized state. Call `initializeAsync()`
    /// to fully initialize the engine before synthesis operations.
    ///
    /// # Returns
    /// A new `WasmSingingEngine` instance or JavaScript error on failure
    ///
    /// # Errors
    /// Returns `JsValue` error if engine creation fails
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

    /// Asynchronously initializes the singing synthesis engine.
    ///
    /// This method must be called before any synthesis operations can be performed.
    /// It loads neural models and prepares the engine for synthesis.
    ///
    /// # Returns
    /// `Ok(())` on successful initialization
    ///
    /// # Errors
    /// Returns `JsValue` error if engine initialization fails (e.g., model loading errors)
    #[wasm_bindgen(js_name = "initializeAsync")]
    pub async fn initialize_async(&mut self) -> Result<(), JsValue> {
        let engine = SingingEngine::new(self.config.clone())
            .await
            .map_err(|e| WasmError::EngineInitFailed(e.to_string()))?;

        *self.engine.lock().await = Some(engine);

        web_sys::console::log_1(&"VoiRS Singing Engine initialized successfully".into());
        Ok(())
    }

    /// Synthesizes a single musical note with singing voice.
    ///
    /// Creates a simple single-note score and synthesizes it with the specified
    /// voice type. The synthesized audio is returned as a Float32Array suitable
    /// for Web Audio API playback.
    ///
    /// # Arguments
    /// * `midi_note` - MIDI note number (0-127, where 60 is middle C)
    /// * `duration` - Note duration in seconds
    /// * `voice_type` - Voice type string ("soprano", "alto", "tenor", "bass", "mezzo_soprano", "baritone")
    ///
    /// # Returns
    /// `Float32Array` containing synthesized audio samples at 44.1kHz sample rate
    ///
    /// # Errors
    /// Returns `JsValue` error if synthesis fails or engine not initialized
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

    /// Synthesizes a complete musical score from JSON representation.
    ///
    /// Takes a JSON-encoded musical score containing multiple notes and synthesizes
    /// it as a complete singing performance.
    ///
    /// # Arguments
    /// * `score_json` - JSON string representing the musical score with notes, tempo, and key
    /// * `voice_type` - Voice type string ("soprano", "alto", "tenor", "bass", "mezzo_soprano", "baritone")
    ///
    /// # Returns
    /// `Float32Array` containing synthesized audio samples at 44.1kHz sample rate
    ///
    /// # Errors
    /// Returns `JsValue` error if JSON parsing fails, synthesis fails, or engine not initialized
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

    /// Creates a Web Audio API AudioBuffer from synthesized audio data.
    ///
    /// Converts Float32Array audio samples into an AudioBuffer that can be played
    /// through the Web Audio API.
    ///
    /// # Arguments
    /// * `audio_context` - Web Audio API AudioContext
    /// * `audio_data` - Float32Array containing audio samples
    /// * `sample_rate` - Sample rate in Hz (typically 44100)
    ///
    /// # Returns
    /// `AudioBuffer` ready for playback via Web Audio API
    ///
    /// # Errors
    /// Returns `JsValue` error if buffer creation fails
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

    /// Returns an array of all supported voice types.
    ///
    /// This static method provides a list of voice type strings that can be used
    /// with synthesis methods.
    ///
    /// # Returns
    /// JavaScript `Array` containing voice type strings: "soprano", "alto", "tenor",
    /// "bass", "mezzo_soprano", "baritone"
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

    /// Updates the singing engine configuration from JSON.
    ///
    /// Reinitializes the engine with new configuration settings. The engine
    /// will be recreated with the updated configuration.
    ///
    /// # Arguments
    /// * `config_json` - JSON string containing configuration parameters
    ///
    /// # Returns
    /// `Ok(())` on successful configuration update
    ///
    /// # Errors
    /// Returns `JsValue` error if JSON parsing fails or engine reinitialization fails
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

    /// Returns the current version of the VoiRS singing WASM module.
    ///
    /// # Returns
    /// Version string from package metadata (e.g., "0.1.0-alpha.2")
    #[wasm_bindgen(js_name = "getVersion")]
    pub fn get_version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

/// Simple audio player for WebAssembly using Web Audio API.
///
/// Provides a convenient wrapper around Web Audio API AudioContext for
/// immediate playback of synthesized audio data.
///
/// # Example (JavaScript)
/// ```javascript
/// const player = new WasmAudioPlayer();
/// player.playAudio(audioData);
/// ```
#[wasm_bindgen]
pub struct WasmAudioPlayer {
    context: AudioContext,
    sample_rate: f32,
}

#[wasm_bindgen]
impl WasmAudioPlayer {
    /// Creates a new WebAssembly audio player.
    ///
    /// Initializes a Web Audio API AudioContext with default settings.
    ///
    /// # Returns
    /// New `WasmAudioPlayer` instance
    ///
    /// # Errors
    /// Returns `JsValue` error if AudioContext creation fails
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmAudioPlayer, JsValue> {
        let context = AudioContext::new()?;
        let sample_rate = context.sample_rate();

        Ok(Self {
            context,
            sample_rate,
        })
    }

    /// Plays audio data immediately through the Web Audio API.
    ///
    /// Creates an AudioBuffer from the provided samples and plays it through
    /// the default audio output device.
    ///
    /// # Arguments
    /// * `audio_data` - Float32Array containing audio samples
    ///
    /// # Returns
    /// `Ok(())` on successful playback start
    ///
    /// # Errors
    /// Returns `JsValue` error if buffer creation or playback fails
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

    /// Returns the sample rate of the audio context.
    ///
    /// # Returns
    /// Sample rate in Hz (typically 44100 or 48000 depending on system)
    #[wasm_bindgen(js_name = "getSampleRate")]
    pub fn get_sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Returns the underlying Web Audio API AudioContext.
    ///
    /// Provides access to the AudioContext for advanced audio processing operations.
    ///
    /// # Returns
    /// Clone of the internal `AudioContext`
    #[wasm_bindgen(js_name = "getContext")]
    pub fn get_context(&self) -> AudioContext {
        self.context.clone()
    }
}

/// Real-time singing synthesizer for interactive web applications.
///
/// Provides low-latency note synthesis with immediate playback through Web Audio API.
/// Suitable for interactive applications like virtual instruments and music games.
///
/// # Example (JavaScript)
/// ```javascript
/// const context = new AudioContext();
/// const synth = new WasmRealtimeSynthesizer(context);
/// await synth.startRealtime();
/// await synth.playNote(60, 1.0, "soprano");
/// ```
#[wasm_bindgen]
pub struct WasmRealtimeSynthesizer {
    engine: Arc<Mutex<Option<SingingEngine>>>,
    audio_context: AudioContext,
    is_playing: bool,
}

#[wasm_bindgen]
impl WasmRealtimeSynthesizer {
    /// Creates a new real-time synthesizer with the given audio context.
    ///
    /// The synthesizer is created in an inactive state. Call `startRealtime()`
    /// to initialize the engine before playing notes.
    ///
    /// # Arguments
    /// * `audio_context` - Web Audio API AudioContext for audio playback
    ///
    /// # Returns
    /// New `WasmRealtimeSynthesizer` instance
    ///
    /// # Errors
    /// Returns `JsValue` error if initialization fails
    #[wasm_bindgen(constructor)]
    pub fn new(audio_context: AudioContext) -> Result<WasmRealtimeSynthesizer, JsValue> {
        Ok(Self {
            engine: Arc::new(Mutex::new(None)),
            audio_context,
            is_playing: false,
        })
    }

    /// Starts the real-time synthesis engine.
    ///
    /// Initializes the singing engine with default configuration and enables
    /// real-time note playback. Must be called before `playNote()`.
    ///
    /// # Returns
    /// `Ok(())` on successful initialization
    ///
    /// # Errors
    /// Returns `JsValue` error if engine initialization fails
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

    /// Synthesizes and immediately plays a musical note.
    ///
    /// Creates a note with the specified parameters, synthesizes it, and plays
    /// the audio through the Web Audio API in a single operation. Suitable for
    /// interactive, low-latency applications.
    ///
    /// # Arguments
    /// * `midi_note` - MIDI note number (0-127, where 60 is middle C)
    /// * `duration` - Note duration in seconds
    /// * `voice_type` - Voice type string ("soprano", "alto", "tenor", "bass", "mezzo_soprano", "baritone")
    ///
    /// # Returns
    /// `Ok(())` when playback starts successfully
    ///
    /// # Errors
    /// Returns `JsValue` error if synthesizer not started, synthesis fails, or playback fails
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

    /// Stops the real-time synthesizer.
    ///
    /// Disables real-time synthesis mode. The engine can be restarted with
    /// `startRealtime()`.
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

/// Initializes logging and error handling for WASM environment.
///
/// Sets up console error panic hook for better error messages in browser console.
/// Should be called once at module initialization.
///
/// # Example (JavaScript)
/// ```javascript
/// import { init_logging } from './voirs_singing';
/// init_logging();
/// ```
#[wasm_bindgen]
pub fn init_logging() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"VoiRS Singing WASM module initialized".into());
}

/// Performance monitoring tool for web applications.
///
/// Tracks synthesis times and uptime statistics for performance analysis
/// and optimization of WASM-based singing synthesis.
///
/// # Example (JavaScript)
/// ```javascript
/// const monitor = new WasmPerformanceMonitor();
/// monitor.recordSynthesisTime(42.5);
/// console.log(monitor.getAverageSynthesisTime());
/// ```
#[wasm_bindgen]
pub struct WasmPerformanceMonitor {
    start_time: f64,
    synthesis_times: Vec<f64>,
}

impl Default for WasmPerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmPerformanceMonitor {
    /// Creates a new performance monitor.
    ///
    /// Initializes the monitor with the current timestamp and empty metrics.
    ///
    /// # Returns
    /// New `WasmPerformanceMonitor` instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            start_time: js_sys::Date::now(),
            synthesis_times: Vec::new(),
        }
    }

    /// Records a synthesis operation time.
    ///
    /// Adds a synthesis time measurement to the performance history for
    /// averaging and analysis.
    ///
    /// # Arguments
    /// * `time_ms` - Synthesis duration in milliseconds
    #[wasm_bindgen(js_name = "recordSynthesisTime")]
    pub fn record_synthesis_time(&mut self, time_ms: f64) {
        self.synthesis_times.push(time_ms);
    }

    /// Calculates the average synthesis time across all recorded operations.
    ///
    /// Computes the mean of all recorded synthesis times. Returns 0.0 if no
    /// times have been recorded.
    ///
    /// # Returns
    /// Average synthesis time in milliseconds, or 0.0 if no data
    #[wasm_bindgen(js_name = "getAverageSynthesisTime")]
    pub fn get_average_synthesis_time(&self) -> f64 {
        if self.synthesis_times.is_empty() {
            0.0
        } else {
            self.synthesis_times.iter().sum::<f64>() / self.synthesis_times.len() as f64
        }
    }

    /// Returns the total uptime since monitor creation or last reset.
    ///
    /// Calculates the elapsed time from monitor initialization to current time.
    ///
    /// # Returns
    /// Total uptime in milliseconds
    #[wasm_bindgen(js_name = "getTotalUptime")]
    pub fn get_total_uptime(&self) -> f64 {
        js_sys::Date::now() - self.start_time
    }

    /// Resets all performance metrics.
    ///
    /// Clears all recorded synthesis times and resets the start time to now.
    /// Useful for starting fresh performance measurements.
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
