//! Text-to-Speech (TTS) Integration for Accessibility
//!
//! This module provides text-to-speech functionality for the `VoiRS` feedback system,
//! enabling accessibility features for users with visual impairments or those who
//! prefer auditory feedback. Integrates with platform-specific TTS engines and
//! provides a unified interface for speaking feedback messages.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{OnceCell, RwLock};

/// Text-to-speech errors
#[derive(Error, Debug, Clone)]
pub enum TtsError {
    /// TTS engine not available
    #[error("TTS engine '{engine}' is not available")]
    EngineNotAvailable {
        /// Name of the TTS engine that is unavailable
        engine: String,
    },

    /// Voice not found
    #[error("Voice '{voice}' not found for language '{language}'")]
    VoiceNotFound {
        /// Voice identifier
        voice: String,
        /// Language code
        language: String,
    },

    /// Speech synthesis failed
    #[error("Speech synthesis failed: {message}")]
    SynthesisFailed {
        /// Error message describing the failure
        message: String,
    },

    /// Audio playback error
    #[error("Audio playback error: {message}")]
    PlaybackError {
        /// Error message describing playback failure
        message: String,
    },

    /// Invalid speech parameters
    #[error("Invalid speech parameters: {message}")]
    InvalidParameters {
        /// Details about invalid parameters
        message: String,
    },

    /// TTS engine initialization failed
    #[error("Failed to initialize TTS engine: {message}")]
    InitializationFailed {
        /// Details about initialization failure
        message: String,
    },
}

/// Result type for TTS operations
pub type TtsResult<T> = Result<T, TtsError>;

/// Speech rate (words per minute)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SpeechRate {
    /// Very slow (100 WPM)
    VerySlow,
    /// Slow (125 WPM)
    Slow,
    /// Normal (175 WPM)
    Normal,
    /// Fast (225 WPM)
    Fast,
    /// Very fast (300 WPM)
    VeryFast,
    /// Custom rate (WPM)
    Custom(f32),
}

impl SpeechRate {
    /// Get words per minute
    #[must_use]
    pub fn wpm(&self) -> f32 {
        match self {
            SpeechRate::VerySlow => 100.0,
            SpeechRate::Slow => 125.0,
            SpeechRate::Normal => 175.0,
            SpeechRate::Fast => 225.0,
            SpeechRate::VeryFast => 300.0,
            SpeechRate::Custom(rate) => *rate,
        }
    }

    /// Convert to platform-specific rate (0.0-1.0 scale)
    #[must_use]
    pub fn to_normalized(&self) -> f32 {
        // Map 100-300 WPM to 0.0-1.0
        (self.wpm() - 100.0) / 200.0
    }
}

/// Voice pitch
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VoicePitch {
    /// Very low pitch (-0.5)
    VeryLow,
    /// Low pitch (-0.25)
    Low,
    /// Normal pitch (0.0)
    Normal,
    /// High pitch (0.25)
    High,
    /// Very high pitch (0.5)
    VeryHigh,
    /// Custom pitch (-1.0 to 1.0)
    Custom(f32),
}

impl VoicePitch {
    /// Get pitch value (-1.0 to 1.0)
    #[must_use]
    pub fn value(&self) -> f32 {
        match self {
            VoicePitch::VeryLow => -0.5,
            VoicePitch::Low => -0.25,
            VoicePitch::Normal => 0.0,
            VoicePitch::High => 0.25,
            VoicePitch::VeryHigh => 0.5,
            VoicePitch::Custom(pitch) => pitch.clamp(-1.0, 1.0),
        }
    }
}

/// Voice volume
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VoiceVolume {
    /// Muted (0%)
    Muted,
    /// Very quiet (25%)
    VeryQuiet,
    /// Quiet (50%)
    Quiet,
    /// Normal (75%)
    Normal,
    /// Loud (100%)
    Loud,
    /// Custom volume (0.0-1.0)
    Custom(f32),
}

impl VoiceVolume {
    /// Get volume value (0.0-1.0)
    #[must_use]
    pub fn value(&self) -> f32 {
        match self {
            VoiceVolume::Muted => 0.0,
            VoiceVolume::VeryQuiet => 0.25,
            VoiceVolume::Quiet => 0.5,
            VoiceVolume::Normal => 0.75,
            VoiceVolume::Loud => 1.0,
            VoiceVolume::Custom(volume) => volume.clamp(0.0, 1.0),
        }
    }
}

/// Voice gender preference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoiceGender {
    /// Male voice
    Male,
    /// Female voice
    Female,
    /// Gender-neutral voice
    Neutral,
    /// No preference
    Any,
}

/// TTS engine type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TtsEngine {
    /// Platform native TTS (Speech Synthesis API on web, SAPI on Windows, etc.)
    Native,
    /// `VoiRS` integrated TTS engine
    VoiRS,
    /// Google Cloud Text-to-Speech
    GoogleCloud,
    /// Amazon Polly
    AmazonPolly,
    /// Microsoft Azure Speech
    AzureSpeech,
    /// Mozilla TTS
    MozillaTts,
    /// Espeak NG (open source)
    EspeakNg,
}

impl TtsEngine {
    /// Get engine name
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            TtsEngine::Native => "Native",
            TtsEngine::VoiRS => "VoiRS",
            TtsEngine::GoogleCloud => "Google Cloud TTS",
            TtsEngine::AmazonPolly => "Amazon Polly",
            TtsEngine::AzureSpeech => "Azure Speech",
            TtsEngine::MozillaTts => "Mozilla TTS",
            TtsEngine::EspeakNg => "Espeak NG",
        }
    }
}

/// Voice information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceInfo {
    /// Voice identifier
    pub id: String,
    /// Voice display name
    pub name: String,
    /// Language code (ISO 639-1)
    pub language: String,
    /// Voice gender
    pub gender: VoiceGender,
    /// Whether this is a neural voice
    pub is_neural: bool,
    /// TTS engine providing this voice
    pub engine: TtsEngine,
    /// Quality rating (0.0-1.0)
    pub quality: f32,
}

/// Speech parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechParameters {
    /// Speech rate
    pub rate: SpeechRate,
    /// Voice pitch
    pub pitch: VoicePitch,
    /// Voice volume
    pub volume: VoiceVolume,
    /// Voice ID to use (optional, will auto-select if None)
    pub voice_id: Option<String>,
    /// Language code (ISO 639-1)
    pub language: String,
    /// Whether to emphasize important words
    pub emphasize: bool,
    /// Whether to add pauses at punctuation
    pub add_pauses: bool,
    /// Whether to use SSML markup if available
    pub use_ssml: bool,
}

impl Default for SpeechParameters {
    fn default() -> Self {
        Self {
            rate: SpeechRate::Normal,
            pitch: VoicePitch::Normal,
            volume: VoiceVolume::Normal,
            voice_id: None,
            language: "en".to_string(),
            emphasize: true,
            add_pauses: true,
            use_ssml: false,
        }
    }
}

/// Speech utterance (text to be spoken)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechUtterance {
    /// Text to speak
    pub text: String,
    /// Speech parameters
    pub parameters: SpeechParameters,
    /// Priority (higher = more important)
    pub priority: i32,
    /// Whether to interrupt current speech
    pub interrupt: bool,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl SpeechUtterance {
    /// Create a new speech utterance with default parameters
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            parameters: SpeechParameters::default(),
            priority: 0,
            interrupt: false,
            metadata: HashMap::new(),
        }
    }

    /// Set speech rate
    #[must_use]
    pub fn with_rate(mut self, rate: SpeechRate) -> Self {
        self.parameters.rate = rate;
        self
    }

    /// Set voice pitch
    #[must_use]
    pub fn with_pitch(mut self, pitch: VoicePitch) -> Self {
        self.parameters.pitch = pitch;
        self
    }

    /// Set volume
    #[must_use]
    pub fn with_volume(mut self, volume: VoiceVolume) -> Self {
        self.parameters.volume = volume;
        self
    }

    /// Set language
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.parameters.language = language.into();
        self
    }

    /// Set priority
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Mark as interrupting
    #[must_use]
    pub fn interrupting(mut self) -> Self {
        self.interrupt = true;
        self
    }
}

/// Speech synthesis result
#[derive(Debug, Clone)]
pub struct SpeechResult {
    /// Synthesized audio data (PCM samples)
    pub audio_data: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Duration in seconds
    pub duration: f32,
}

/// Text-to-speech engine trait
#[async_trait]
pub trait TtsEngineBackend: Send + Sync {
    /// Get engine type
    fn engine_type(&self) -> TtsEngine;

    /// List available voices
    async fn list_voices(&self) -> TtsResult<Vec<VoiceInfo>>;

    /// Synthesize speech to audio
    async fn synthesize(&self, utterance: &SpeechUtterance) -> TtsResult<SpeechResult>;

    /// Speak text directly (play audio)
    async fn speak(&self, utterance: &SpeechUtterance) -> TtsResult<()>;

    /// Stop current speech
    async fn stop(&self) -> TtsResult<()>;

    /// Pause current speech
    async fn pause(&self) -> TtsResult<()>;

    /// Resume paused speech
    async fn resume(&self) -> TtsResult<()>;

    /// Check if engine is speaking
    async fn is_speaking(&self) -> bool;
}

/// Real TTS engine backed by `voirs-sdk`'s synthesis pipeline
/// ([`voirs_sdk::VoirsPipeline`]).
///
/// The pipeline is expensive to build (real G2P + real acoustic model +
/// real vocoder, possibly downloading model weights over the network on
/// first use), so it is constructed lazily on first call and cached for
/// the engine's lifetime. If construction fails — no network, no cached
/// model weights, an unsupported device, or it simply takes too long —
/// every method honestly returns [`TtsError::InitializationFailed`] or
/// [`TtsError::SynthesisFailed`]; there is no silent fallback to silence.
pub struct VoirsTtsEngine {
    pipeline: OnceCell<Result<Arc<voirs_sdk::VoirsPipeline>, String>>,
    /// When `true`, the underlying pipeline is built with
    /// `voirs_sdk`'s explicit test mode (offline, deterministic-per-input
    /// dummy components) instead of attempting to load real model weights.
    /// Only ever set by [`Self::new_test_mode`], used by this crate's own
    /// tests — production callers get [`Self::new`], which always attempts
    /// real synthesis.
    test_mode: bool,
    /// Upper bound on how long pipeline construction (including any model
    /// download) may take before this engine gives up and reports
    /// [`TtsError::InitializationFailed`].
    build_timeout: Duration,
    speaking: Arc<RwLock<bool>>,
}

impl std::fmt::Debug for VoirsTtsEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoirsTtsEngine")
            .field("test_mode", &self.test_mode)
            .field("build_timeout", &self.build_timeout)
            .field("pipeline_built", &self.pipeline.initialized())
            .finish()
    }
}

impl VoirsTtsEngine {
    /// Create a new engine that attempts real synthesis via `voirs-sdk`
    /// (real G2P, real acoustic model, real vocoder) on first use.
    #[must_use]
    pub fn new() -> Self {
        Self::with_options(false, Duration::from_secs(30))
    }

    /// Create an engine whose pipeline is built in `voirs-sdk`'s explicit
    /// test mode: fast, fully offline, deterministic-per-input dummy
    /// components (see `voirs_sdk::pipeline::{DummyG2p, DummyAcoustic,
    /// DummyVocoder}`) rather than real model weights. Intended for this
    /// crate's own integration tests; never used by [`TtsManager::new`].
    #[must_use]
    pub fn new_test_mode() -> Self {
        Self::with_options(true, Duration::from_secs(10))
    }

    fn with_options(test_mode: bool, build_timeout: Duration) -> Self {
        Self {
            pipeline: OnceCell::new(),
            test_mode,
            build_timeout,
            speaking: Arc::new(RwLock::new(false)),
        }
    }

    /// Get (building on first call) the underlying real pipeline.
    async fn pipeline(&self) -> TtsResult<Arc<voirs_sdk::VoirsPipeline>> {
        let test_mode = self.test_mode;
        let build_timeout = self.build_timeout;

        let result = self
            .pipeline
            .get_or_init(move || async move {
                let build = voirs_sdk::VoirsPipeline::builder()
                    .with_test_mode(test_mode)
                    .build();

                match tokio::time::timeout(build_timeout, build).await {
                    Ok(Ok(pipeline)) => Ok(Arc::new(pipeline)),
                    Ok(Err(e)) => Err(format!("failed to build VoiRS pipeline: {e}")),
                    Err(_) => Err(format!(
                        "timed out after {build_timeout:?} building the VoiRS pipeline \
                         (no network access or cached model weights?)"
                    )),
                }
            })
            .await;

        result
            .clone()
            .map_err(|message| TtsError::InitializationFailed { message })
    }
}

impl Default for VoirsTtsEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TtsEngineBackend for VoirsTtsEngine {
    fn engine_type(&self) -> TtsEngine {
        TtsEngine::VoiRS
    }

    async fn list_voices(&self) -> TtsResult<Vec<VoiceInfo>> {
        let pipeline = self.pipeline().await?;
        let voice_configs =
            pipeline
                .list_voices()
                .await
                .map_err(|e| TtsError::SynthesisFailed {
                    message: format!("failed to list VoiRS voices: {e}"),
                })?;

        Ok(voice_configs
            .into_iter()
            .map(|voice| {
                let gender = match voice.characteristics.gender {
                    Some(voirs_sdk::types::Gender::Male) => VoiceGender::Male,
                    Some(voirs_sdk::types::Gender::Female) => VoiceGender::Female,
                    Some(voirs_sdk::types::Gender::NonBinary) => VoiceGender::Neutral,
                    None => VoiceGender::Any,
                };
                VoiceInfo {
                    id: voice.id,
                    name: voice.name,
                    language: voice.language.as_str().to_string(),
                    gender,
                    is_neural: true,
                    engine: TtsEngine::VoiRS,
                    quality: 1.0,
                }
            })
            .collect())
    }

    async fn synthesize(&self, utterance: &SpeechUtterance) -> TtsResult<SpeechResult> {
        let pipeline = self.pipeline().await?;
        let audio =
            pipeline
                .synthesize(&utterance.text)
                .await
                .map_err(|e| TtsError::SynthesisFailed {
                    message: e.to_string(),
                })?;

        Ok(SpeechResult {
            audio_data: audio.samples().to_vec(),
            sample_rate: audio.sample_rate(),
            channels: audio.channels() as u16,
            duration: audio.duration(),
        })
    }

    async fn speak(&self, utterance: &SpeechUtterance) -> TtsResult<()> {
        let speech = self.synthesize(utterance).await?;

        *self.speaking.write().await = true;
        let audio = voirs_sdk::AudioBuffer::new(
            speech.audio_data,
            speech.sample_rate,
            u32::from(speech.channels),
        );
        // `AudioBuffer::play` is a blocking call (opens a real cpal output
        // stream and sleeps for the audio's duration), so it must run on a
        // blocking-friendly thread rather than the async executor.
        let play_result = tokio::task::spawn_blocking(move || audio.play()).await;
        *self.speaking.write().await = false;

        match play_result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(TtsError::PlaybackError {
                message: e.to_string(),
            }),
            Err(join_err) => Err(TtsError::PlaybackError {
                message: format!("playback task panicked: {join_err}"),
            }),
        }
    }

    async fn stop(&self) -> TtsResult<()> {
        // The underlying `cpal` stream created by `AudioBuffer::play` runs
        // to completion inside its own blocking task with no external
        // handle exposed to cancel it early; only the "are we speaking"
        // flag can be reset from here.
        *self.speaking.write().await = false;
        Ok(())
    }

    async fn pause(&self) -> TtsResult<()> {
        Err(TtsError::InvalidParameters {
            message: "VoirsTtsEngine does not support pausing in-progress playback".to_string(),
        })
    }

    async fn resume(&self) -> TtsResult<()> {
        Err(TtsError::InvalidParameters {
            message: "VoirsTtsEngine does not support resuming paused playback".to_string(),
        })
    }

    async fn is_speaking(&self) -> bool {
        *self.speaking.read().await
    }
}

/// Mock TTS engine: produces silent, purely length-derived audio and never
/// touches any real synthesis backend.
///
/// This exists solely for this crate's own unit tests and for downstream
/// consumers who explicitly want a deterministic stand-in in their own
/// tests (via [`TtsManager::register_engine`]). It is **never** registered
/// automatically — [`TtsManager::new`] registers the real
/// [`VoirsTtsEngine`] instead, so silence is never a default output.
#[derive(Debug)]
pub struct MockTtsEngine {
    voices: Vec<VoiceInfo>,
    speaking: Arc<RwLock<bool>>,
}

impl MockTtsEngine {
    /// Create a new mock TTS engine
    #[must_use]
    pub fn new() -> Self {
        let voices = vec![
            VoiceInfo {
                id: "en-US-mock-female".to_string(),
                name: "Mock Female Voice".to_string(),
                language: "en".to_string(),
                gender: VoiceGender::Female,
                is_neural: true,
                engine: TtsEngine::Native,
                quality: 0.8,
            },
            VoiceInfo {
                id: "en-US-mock-male".to_string(),
                name: "Mock Male Voice".to_string(),
                language: "en".to_string(),
                gender: VoiceGender::Male,
                is_neural: true,
                engine: TtsEngine::Native,
                quality: 0.8,
            },
        ];

        Self {
            voices,
            speaking: Arc::new(RwLock::new(false)),
        }
    }
}

impl Default for MockTtsEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TtsEngineBackend for MockTtsEngine {
    fn engine_type(&self) -> TtsEngine {
        TtsEngine::Native
    }

    async fn list_voices(&self) -> TtsResult<Vec<VoiceInfo>> {
        Ok(self.voices.clone())
    }

    async fn synthesize(&self, utterance: &SpeechUtterance) -> TtsResult<SpeechResult> {
        // Generate mock audio (silence)
        let sample_rate = 16000;
        let duration = (utterance.text.len() as f32) * 0.05; // ~50ms per character
        let num_samples = (duration * sample_rate as f32) as usize;

        Ok(SpeechResult {
            audio_data: vec![0.0; num_samples],
            sample_rate,
            channels: 1,
            duration,
        })
    }

    async fn speak(&self, utterance: &SpeechUtterance) -> TtsResult<()> {
        *self.speaking.write().await = true;

        // Simulate speech duration
        let duration = (utterance.text.len() as f32) * 0.05;
        tokio::time::sleep(tokio::time::Duration::from_secs_f32(duration)).await;

        *self.speaking.write().await = false;
        Ok(())
    }

    async fn stop(&self) -> TtsResult<()> {
        *self.speaking.write().await = false;
        Ok(())
    }

    async fn pause(&self) -> TtsResult<()> {
        Ok(())
    }

    async fn resume(&self) -> TtsResult<()> {
        Ok(())
    }

    async fn is_speaking(&self) -> bool {
        *self.speaking.read().await
    }
}

/// TTS manager for handling multiple engines and voice selection
pub struct TtsManager {
    engines: HashMap<TtsEngine, Arc<dyn TtsEngineBackend>>,
    active_engine: Arc<RwLock<TtsEngine>>,
    speech_queue: Arc<RwLock<Vec<SpeechUtterance>>>,
    preferences: Arc<RwLock<SpeechParameters>>,
}

impl TtsManager {
    /// Create a new TTS manager backed by the real [`VoirsTtsEngine`].
    ///
    /// The real engine is registered under both [`TtsEngine::Native`] (the
    /// manager's default active engine) and [`TtsEngine::VoiRS`], so a
    /// freshly constructed manager attempts genuine synthesis via
    /// `voirs-sdk` out of the box. Model loading is lazy and can fail (no
    /// network, no cached weights, ...); those failures surface as typed
    /// [`TtsError`]s from `synthesize`/`speak`, never as silent digital
    /// silence. Use [`MockTtsEngine`] explicitly (via
    /// [`Self::register_engine`]) if a deterministic offline stand-in is
    /// needed for testing.
    #[must_use]
    pub fn new() -> Self {
        let mut engines: HashMap<TtsEngine, Arc<dyn TtsEngineBackend>> = HashMap::new();
        let voirs_engine: Arc<dyn TtsEngineBackend> = Arc::new(VoirsTtsEngine::new());
        engines.insert(TtsEngine::Native, voirs_engine.clone());
        engines.insert(TtsEngine::VoiRS, voirs_engine);

        Self {
            engines,
            active_engine: Arc::new(RwLock::new(TtsEngine::Native)),
            speech_queue: Arc::new(RwLock::new(Vec::new())),
            preferences: Arc::new(RwLock::new(SpeechParameters::default())),
        }
    }

    /// Register a TTS engine backend
    pub async fn register_engine(&mut self, engine: Arc<dyn TtsEngineBackend>) {
        let engine_type = engine.engine_type();
        self.engines.insert(engine_type, engine);
    }

    /// Set active TTS engine
    pub async fn set_active_engine(&self, engine: TtsEngine) -> TtsResult<()> {
        if !self.engines.contains_key(&engine) {
            return Err(TtsError::EngineNotAvailable {
                engine: engine.name().to_string(),
            });
        }
        *self.active_engine.write().await = engine;
        Ok(())
    }

    /// Get active engine
    pub async fn get_active_engine(&self) -> TtsEngine {
        *self.active_engine.read().await
    }

    /// List available voices from active engine
    pub async fn list_voices(&self) -> TtsResult<Vec<VoiceInfo>> {
        let engine_type = *self.active_engine.read().await;
        let engine =
            self.engines
                .get(&engine_type)
                .ok_or_else(|| TtsError::EngineNotAvailable {
                    engine: engine_type.name().to_string(),
                })?;

        engine.list_voices().await
    }

    /// Find best matching voice for language and preferences
    pub async fn find_voice(
        &self,
        language: &str,
        gender: VoiceGender,
    ) -> TtsResult<Option<VoiceInfo>> {
        let voices = self.list_voices().await?;

        // First try exact language match with preferred gender
        if let Some(voice) = voices
            .iter()
            .find(|v| v.language == language && (gender == VoiceGender::Any || v.gender == gender))
        {
            return Ok(Some(voice.clone()));
        }

        // Try language prefix match (e.g., "en" for "en-US")
        let lang_prefix = language.split('-').next().unwrap_or(language);
        if let Some(voice) = voices.iter().find(|v| {
            v.language.starts_with(lang_prefix)
                && (gender == VoiceGender::Any || v.gender == gender)
        }) {
            return Ok(Some(voice.clone()));
        }

        // Fallback to any voice for the language
        if let Some(voice) = voices.iter().find(|v| v.language.starts_with(lang_prefix)) {
            return Ok(Some(voice.clone()));
        }

        Ok(None)
    }

    /// Set default speech parameters
    pub async fn set_preferences(&self, preferences: SpeechParameters) {
        *self.preferences.write().await = preferences;
    }

    /// Get default speech parameters
    pub async fn get_preferences(&self) -> SpeechParameters {
        self.preferences.read().await.clone()
    }

    /// Speak text immediately
    pub async fn speak(&self, text: impl Into<String>) -> TtsResult<()> {
        let preferences = self.get_preferences().await;
        let utterance = SpeechUtterance {
            text: text.into(),
            parameters: preferences,
            priority: 0,
            interrupt: false,
            metadata: HashMap::new(),
        };

        let engine_type = *self.active_engine.read().await;
        let engine =
            self.engines
                .get(&engine_type)
                .ok_or_else(|| TtsError::EngineNotAvailable {
                    engine: engine_type.name().to_string(),
                })?;

        engine.speak(&utterance).await
    }

    /// Speak utterance with custom parameters
    pub async fn speak_utterance(&self, utterance: &SpeechUtterance) -> TtsResult<()> {
        let engine_type = *self.active_engine.read().await;
        let engine =
            self.engines
                .get(&engine_type)
                .ok_or_else(|| TtsError::EngineNotAvailable {
                    engine: engine_type.name().to_string(),
                })?;

        if utterance.interrupt {
            engine.stop().await?;
        }

        engine.speak(utterance).await
    }

    /// Queue speech utterance
    pub async fn queue_speech(&self, utterance: SpeechUtterance) {
        let mut queue = self.speech_queue.write().await;
        queue.push(utterance);
        queue.sort_by_key(|b| std::cmp::Reverse(b.priority));
    }

    /// Process speech queue
    pub async fn process_queue(&self) -> TtsResult<()> {
        let mut queue = self.speech_queue.write().await;
        if queue.is_empty() {
            return Ok(());
        }

        let utterance = queue.remove(0);
        drop(queue); // Release lock before speaking

        self.speak_utterance(&utterance).await
    }

    /// Stop current speech
    pub async fn stop(&self) -> TtsResult<()> {
        let engine_type = *self.active_engine.read().await;
        let engine =
            self.engines
                .get(&engine_type)
                .ok_or_else(|| TtsError::EngineNotAvailable {
                    engine: engine_type.name().to_string(),
                })?;

        engine.stop().await
    }

    /// Pause current speech
    pub async fn pause(&self) -> TtsResult<()> {
        let engine_type = *self.active_engine.read().await;
        let engine =
            self.engines
                .get(&engine_type)
                .ok_or_else(|| TtsError::EngineNotAvailable {
                    engine: engine_type.name().to_string(),
                })?;

        engine.pause().await
    }

    /// Resume paused speech
    pub async fn resume(&self) -> TtsResult<()> {
        let engine_type = *self.active_engine.read().await;
        let engine =
            self.engines
                .get(&engine_type)
                .ok_or_else(|| TtsError::EngineNotAvailable {
                    engine: engine_type.name().to_string(),
                })?;

        engine.resume().await
    }

    /// Check if currently speaking
    pub async fn is_speaking(&self) -> bool {
        let engine_type = *self.active_engine.read().await;
        if let Some(engine) = self.engines.get(&engine_type) {
            engine.is_speaking().await
        } else {
            false
        }
    }

    /// Clear speech queue
    pub async fn clear_queue(&self) {
        self.speech_queue.write().await.clear();
    }
}

impl Default for TtsManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speech_rate() {
        assert_eq!(SpeechRate::VerySlow.wpm(), 100.0);
        assert_eq!(SpeechRate::Normal.wpm(), 175.0);
        assert_eq!(SpeechRate::VeryFast.wpm(), 300.0);

        let custom = SpeechRate::Custom(150.0);
        assert_eq!(custom.wpm(), 150.0);
        assert!((custom.to_normalized() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_voice_pitch() {
        assert_eq!(VoicePitch::VeryLow.value(), -0.5);
        assert_eq!(VoicePitch::Normal.value(), 0.0);
        assert_eq!(VoicePitch::VeryHigh.value(), 0.5);

        // Test clamping
        let custom = VoicePitch::Custom(2.0);
        assert_eq!(custom.value(), 1.0);
    }

    #[test]
    fn test_voice_volume() {
        assert_eq!(VoiceVolume::Muted.value(), 0.0);
        assert_eq!(VoiceVolume::Normal.value(), 0.75);
        assert_eq!(VoiceVolume::Loud.value(), 1.0);
    }

    #[test]
    fn test_speech_utterance_builder() {
        let utterance = SpeechUtterance::new("Hello world")
            .with_rate(SpeechRate::Fast)
            .with_pitch(VoicePitch::High)
            .with_volume(VoiceVolume::Loud)
            .with_language("en-US")
            .with_priority(10)
            .interrupting();

        assert_eq!(utterance.text, "Hello world");
        assert_eq!(utterance.parameters.rate, SpeechRate::Fast);
        assert_eq!(utterance.parameters.pitch, VoicePitch::High);
        assert_eq!(utterance.parameters.volume, VoiceVolume::Loud);
        assert_eq!(utterance.parameters.language, "en-US");
        assert_eq!(utterance.priority, 10);
        assert!(utterance.interrupt);
    }

    #[tokio::test]
    async fn test_mock_engine_list_voices() {
        let engine = MockTtsEngine::new();
        let voices = engine.list_voices().await.unwrap();

        assert_eq!(voices.len(), 2);
        assert!(voices.iter().any(|v| v.gender == VoiceGender::Female));
        assert!(voices.iter().any(|v| v.gender == VoiceGender::Male));
    }

    #[tokio::test]
    async fn test_mock_engine_synthesize() {
        let engine = MockTtsEngine::new();
        let utterance = SpeechUtterance::new("Test message");

        let result = engine.synthesize(&utterance).await.unwrap();

        assert_eq!(result.sample_rate, 16000);
        assert_eq!(result.channels, 1);
        assert!(result.duration > 0.0);
        assert!(!result.audio_data.is_empty());
    }

    #[tokio::test]
    async fn test_mock_engine_speak() {
        let engine = MockTtsEngine::new();
        let utterance = SpeechUtterance::new("Short");

        assert!(!engine.is_speaking().await);

        let speak_task = tokio::spawn(async move {
            engine.speak(&utterance).await.unwrap();
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        speak_task.await.unwrap();
    }

    #[tokio::test]
    async fn test_tts_manager_creation() {
        let manager = TtsManager::new();
        let engine = manager.get_active_engine().await;
        assert_eq!(engine, TtsEngine::Native);
    }

    #[tokio::test]
    async fn test_tts_manager_default_active_engine_is_real_voirs_not_mock() {
        let manager = TtsManager::new();
        assert_eq!(manager.get_active_engine().await, TtsEngine::Native);

        // The backend registered under the default-active `Native` key must
        // be the real VoiRS engine, not a silently-substituted mock (whose
        // `engine_type()` would also report `Native`, but is a distinct
        // type from `VoirsTtsEngine`).
        let native_backend = manager.engines.get(&TtsEngine::Native).unwrap();
        let voirs_backend = manager.engines.get(&TtsEngine::VoiRS).unwrap();
        assert_eq!(native_backend.engine_type(), TtsEngine::VoiRS);
        assert_eq!(voirs_backend.engine_type(), TtsEngine::VoiRS);
    }

    /// Registering a `MockTtsEngine` (which reports `TtsEngine::Native`)
    /// injects it into the `Native` slot, giving tests of manager
    /// *mechanics* (queueing, preferences, ...) a deterministic, offline
    /// backend independent of real model availability.
    async fn manager_with_mock_engine() -> TtsManager {
        let mut manager = TtsManager::new();
        manager
            .register_engine(Arc::new(MockTtsEngine::new()))
            .await;
        manager
    }

    #[tokio::test]
    async fn test_tts_manager_list_voices() {
        let manager = manager_with_mock_engine().await;
        let voices = manager.list_voices().await.unwrap();
        assert!(!voices.is_empty());
    }

    #[tokio::test]
    async fn test_tts_manager_find_voice() {
        let manager = manager_with_mock_engine().await;

        let voice = manager.find_voice("en", VoiceGender::Female).await.unwrap();
        assert!(voice.is_some());
        assert_eq!(voice.unwrap().gender, VoiceGender::Female);

        let voice = manager.find_voice("en", VoiceGender::Male).await.unwrap();
        assert!(voice.is_some());
        assert_eq!(voice.unwrap().gender, VoiceGender::Male);
    }

    #[tokio::test]
    async fn test_tts_manager_preferences() {
        let manager = TtsManager::new();

        let mut prefs = SpeechParameters::default();
        prefs.rate = SpeechRate::Fast;
        prefs.volume = VoiceVolume::Loud;

        manager.set_preferences(prefs.clone()).await;
        let retrieved = manager.get_preferences().await;

        assert_eq!(retrieved.rate, SpeechRate::Fast);
        assert_eq!(retrieved.volume, VoiceVolume::Loud);
    }

    #[tokio::test]
    async fn test_tts_manager_speak() {
        let manager = manager_with_mock_engine().await;
        let result = manager.speak("Hello TTS").await;
        assert!(result.is_ok());
    }

    /// Real end-to-end check of the default (non-mock) manager. A
    /// sandboxed CI environment with no network access and no cached model
    /// weights is expected to honestly fail pipeline construction; a
    /// developer machine with real models available is expected to
    /// actually synthesize. Silent digital silence (the old
    /// `MockTtsEngine`-by-default behavior) is the one outcome this test
    /// rules out.
    #[tokio::test]
    async fn test_voirs_tts_engine_synthesize_is_real_not_fake_silence() {
        let engine = VoirsTtsEngine::new();
        let utterance = SpeechUtterance::new("Testing real synthesis output.");

        match engine.synthesize(&utterance).await {
            Ok(result) => {
                assert!(result.sample_rate > 0);
                assert!(!result.audio_data.is_empty());
                assert!(
                    result.audio_data.iter().any(|&s| s != 0.0),
                    "real synthesis must not be all-zero silence"
                );
            }
            Err(TtsError::InitializationFailed { .. } | TtsError::SynthesisFailed { .. }) => {
                // Honest fail-closed: no network / no cached model weights
                // in this environment. Acceptable; fabricated success is not.
            }
            Err(other) => panic!("unexpected error variant from real engine: {other:?}"),
        }
    }

    #[test]
    fn test_voirs_tts_engine_type_is_not_mock() {
        let engine = VoirsTtsEngine::new();
        assert_eq!(engine.engine_type(), TtsEngine::VoiRS);
    }

    /// The strongest regression test for this fix: using `voirs-sdk`'s own
    /// explicit, fully-offline test mode (no network required), prove that
    /// (a) synthesis actually succeeds, (b) the output is not all-zero
    /// silence, and (c) different input text produces genuinely different
    /// output — the defining property the old
    /// `vec![0.0; utterance.text.len() * const]` mock never had.
    #[tokio::test]
    async fn test_voirs_tts_engine_test_mode_output_varies_with_input() {
        let engine = VoirsTtsEngine::new_test_mode();

        let short = SpeechUtterance::new("Hi");
        let long =
            SpeechUtterance::new("This is a considerably longer sentence than the short one.");

        let short_result = engine
            .synthesize(&short)
            .await
            .expect("offline test-mode synthesis should never require network access");
        let long_result = engine
            .synthesize(&long)
            .await
            .expect("offline test-mode synthesis should never require network access");

        assert!(!short_result.audio_data.is_empty());
        assert!(!long_result.audio_data.is_empty());
        assert!(short_result.audio_data.iter().any(|&s| s != 0.0));
        assert!(long_result.audio_data.iter().any(|&s| s != 0.0));
        assert_ne!(
            short_result.duration, long_result.duration,
            "audio duration must depend on the real input, not be a fixed/fake value"
        );

        // Calling twice with the same text must not spuriously fail: the
        // cached pipeline is reused, not rebuilt per call.
        let repeat_result = engine
            .synthesize(&short)
            .await
            .expect("second call should reuse the cached pipeline");
        assert_eq!(repeat_result.sample_rate, short_result.sample_rate);
    }

    #[tokio::test]
    async fn test_tts_manager_queue() {
        let manager = TtsManager::new();

        let utterance1 = SpeechUtterance::new("Low priority").with_priority(1);
        let utterance2 = SpeechUtterance::new("High priority").with_priority(10);
        let utterance3 = SpeechUtterance::new("Medium priority").with_priority(5);

        manager.queue_speech(utterance1).await;
        manager.queue_speech(utterance2).await;
        manager.queue_speech(utterance3).await;

        let queue = manager.speech_queue.read().await;
        assert_eq!(queue.len(), 3);
        assert_eq!(queue[0].priority, 10); // Highest priority first
        assert_eq!(queue[1].priority, 5);
        assert_eq!(queue[2].priority, 1);
    }

    #[tokio::test]
    async fn test_tts_manager_process_queue() {
        let manager = manager_with_mock_engine().await;

        let utterance = SpeechUtterance::new("Queued message");
        manager.queue_speech(utterance).await;

        let result = manager.process_queue().await;
        assert!(result.is_ok());

        let queue = manager.speech_queue.read().await;
        assert_eq!(queue.len(), 0); // Queue should be empty after processing
    }

    #[tokio::test]
    async fn test_tts_manager_clear_queue() {
        let manager = TtsManager::new();

        manager
            .queue_speech(SpeechUtterance::new("Message 1"))
            .await;
        manager
            .queue_speech(SpeechUtterance::new("Message 2"))
            .await;

        manager.clear_queue().await;

        let queue = manager.speech_queue.read().await;
        assert_eq!(queue.len(), 0);
    }
}
