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
use thiserror::Error;
use tokio::sync::RwLock;

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

/// Mock TTS engine for testing and fallback
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
    /// Create a new TTS manager with default mock engine
    #[must_use]
    pub fn new() -> Self {
        let mut engines: HashMap<TtsEngine, Arc<dyn TtsEngineBackend>> = HashMap::new();
        engines.insert(TtsEngine::Native, Arc::new(MockTtsEngine::new()));

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
        queue.sort_by(|a, b| b.priority.cmp(&a.priority));
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
    async fn test_tts_manager_list_voices() {
        let manager = TtsManager::new();
        let voices = manager.list_voices().await.unwrap();
        assert!(!voices.is_empty());
    }

    #[tokio::test]
    async fn test_tts_manager_find_voice() {
        let manager = TtsManager::new();

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
        let manager = TtsManager::new();
        let result = manager.speak("Hello TTS").await;
        assert!(result.is_ok());
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
        let manager = TtsManager::new();

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
