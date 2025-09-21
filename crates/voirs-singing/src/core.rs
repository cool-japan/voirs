//! Core singing engine implementation

#![allow(
    clippy::uninlined_format_args,
    clippy::get_first,
    clippy::upper_case_acronyms
)]

use crate::config::SingingConfig;
use crate::effects::EffectChain;
use crate::formats::FormatParser;
use crate::score::MusicalScore;
use crate::synthesis::SynthesisEngine;
use crate::techniques::SingingTechnique;
use crate::types::{SingingRequest, SingingResponse, SingingStats, VoiceCharacteristics};
use crate::voice::VoiceManager;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Main singing engine with comprehensive thread safety
pub struct SingingEngine {
    /// Configuration
    config: Arc<RwLock<SingingConfig>>,
    /// Synthesis engine
    synthesis_engine: Arc<RwLock<SynthesisEngine>>,
    /// Voice manager
    voice_manager: Arc<RwLock<VoiceManager>>,
    /// Effect chain
    effect_chain: Arc<RwLock<EffectChain>>,
    /// Format parsers (thread-safe)
    format_parsers: Arc<RwLock<HashMap<String, Box<dyn FormatParser>>>>,
    /// Performance statistics
    stats: Arc<RwLock<SingingStats>>,
    /// Enabled state (thread-safe)
    enabled: Arc<RwLock<bool>>,
}

/// Builder for singing engine
pub struct SingingEngineBuilder {
    config: Option<SingingConfig>,
    voice_characteristics: Option<VoiceCharacteristics>,
    technique: Option<SingingTechnique>,
    effects: Vec<String>,
    format_support: Vec<String>,
}

impl SingingEngine {
    /// Create new singing engine
    pub async fn new(config: SingingConfig) -> crate::Result<Self> {
        let synthesis_engine = SynthesisEngine::new(config.clone());
        let voice_manager = VoiceManager::new();
        let effect_chain = EffectChain::new();

        let mut format_parsers: HashMap<String, Box<dyn FormatParser>> = HashMap::new();

        // Register format parsers
        #[cfg(feature = "musicxml-support")]
        {
            format_parsers.insert(
                String::from("musicxml"),
                Box::new(crate::formats::MusicXmlParser::new()),
            );
        }

        #[cfg(feature = "midi-support")]
        {
            format_parsers.insert(
                String::from("midi"),
                Box::new(crate::formats::MidiParser::new()),
            );
        }

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            synthesis_engine: Arc::new(RwLock::new(synthesis_engine)),
            voice_manager: Arc::new(RwLock::new(voice_manager)),
            effect_chain: Arc::new(RwLock::new(effect_chain)),
            format_parsers: Arc::new(RwLock::new(format_parsers)),
            stats: Arc::new(RwLock::new(SingingStats::default())),
            enabled: Arc::new(RwLock::new(true)),
        })
    }

    /// Enable or disable singing engine (thread-safe)
    pub async fn set_enabled(&self, enabled: bool) {
        let mut enabled_guard = self.enabled.write().await;
        *enabled_guard = enabled;
    }

    /// Check if engine is enabled (thread-safe)
    pub async fn is_enabled(&self) -> bool {
        let enabled_guard = self.enabled.read().await;
        *enabled_guard
    }

    /// Set configuration
    pub async fn set_config(&self, config: SingingConfig) -> crate::Result<()> {
        let mut config_guard = self.config.write().await;
        *config_guard = config;
        Ok(())
    }

    /// Get configuration (clones current config)
    pub async fn config(&self) -> SingingConfig {
        self.config.read().await.clone()
    }

    /// Set voice characteristics
    pub async fn set_voice_characteristics(
        &self,
        voice: VoiceCharacteristics,
    ) -> crate::Result<()> {
        let mut synthesis_engine = self.synthesis_engine.write().await;
        synthesis_engine.set_voice_characteristics(voice);
        Ok(())
    }

    /// Set singing technique
    pub async fn set_technique(&self, technique: SingingTechnique) -> crate::Result<()> {
        let mut synthesis_engine = self.synthesis_engine.write().await;
        synthesis_engine.set_technique(technique);
        Ok(())
    }

    /// Synthesize singing from request
    pub async fn synthesize(&self, request: SingingRequest) -> crate::Result<SingingResponse> {
        let enabled = self.is_enabled().await;
        if !enabled {
            return Err(crate::Error::Config(String::from(
                "Singing engine is disabled",
            )));
        }

        let start_time = Instant::now();

        // Validate request
        self.validate_request(&request).await?;

        // Synthesize audio
        let mut synthesis_engine = self.synthesis_engine.write().await;
        let synthesis_result = synthesis_engine.synthesize(request.clone()).await?;

        // Apply effects
        let mut effect_chain = self.effect_chain.write().await;
        let processed_audio = effect_chain
            .process(synthesis_result.audio.clone(), request.sample_rate as f32)
            .await?;

        // Update statistics
        let processing_time = start_time.elapsed();
        let mut stats = self.stats.write().await;
        stats.update_synthesis_stats(processing_time, synthesis_result.stats);

        // Create response
        let response = SingingResponse {
            audio: processed_audio,
            sample_rate: request.sample_rate,
            duration: processing_time,
            voice: request.voice,
            technique: request.technique,
            stats: stats.clone(),
            metadata: HashMap::new(),
        };

        Ok(response)
    }

    /// Synthesize from score
    pub async fn synthesize_score(
        &self,
        score: MusicalScore,
        voice: VoiceCharacteristics,
        technique: SingingTechnique,
    ) -> crate::Result<SingingResponse> {
        let config = self.config.read().await;

        let request = SingingRequest {
            score,
            voice,
            technique,
            effects: config.effects.default_chain.clone(),
            sample_rate: config.audio.sample_rate,
            target_duration: None,
            quality: config.quality.clone(),
        };

        self.synthesize(request).await
    }

    /// Synthesize from file
    pub async fn synthesize_from_file(
        &self,
        file_path: &str,
        voice: VoiceCharacteristics,
        technique: SingingTechnique,
    ) -> crate::Result<SingingResponse> {
        // Determine format from file extension
        let format = self.detect_format(file_path)?;

        // Parse score (thread-safe access)
        let format_parsers = self.format_parsers.read().await;
        let parser = format_parsers
            .get(&format)
            .ok_or_else(|| crate::Error::Format(format!("Unsupported format: {format}")))?;

        let score = parser.parse_file(file_path).await?;

        // Synthesize
        self.synthesize_score(score, voice, technique).await
    }

    /// Synthesize from text with automatic melody generation
    pub async fn synthesize_from_text(
        &self,
        text: &str,
        key: &str,
        tempo: f32,
        voice: VoiceCharacteristics,
        technique: SingingTechnique,
    ) -> crate::Result<SingingResponse> {
        // Generate melody from text
        let score = self.generate_melody_from_text(text, key, tempo).await?;

        // Synthesize
        self.synthesize_score(score, voice, technique).await
    }

    /// Load voice from file
    pub async fn load_voice(&self, voice_path: &str) -> crate::Result<VoiceCharacteristics> {
        let voice_manager = self.voice_manager.read().await;
        voice_manager.load_voice(voice_path).await
    }

    /// Save voice to file
    pub async fn save_voice(
        &self,
        voice: &VoiceCharacteristics,
        voice_path: &str,
    ) -> crate::Result<()> {
        let voice_manager = self.voice_manager.read().await;
        voice_manager.save_voice(voice, voice_path).await
    }

    /// List available voices
    pub async fn list_voices(&self) -> crate::Result<Vec<String>> {
        let voice_manager = self.voice_manager.read().await;
        voice_manager.list_voices().await
    }

    /// Add effect to chain
    pub async fn add_effect(
        &self,
        effect_name: &str,
        parameters: &HashMap<String, f32>,
    ) -> crate::Result<()> {
        let mut effect_chain = self.effect_chain.write().await;
        effect_chain
            .add_effect(effect_name, parameters.clone())
            .await
    }

    /// Remove effect from chain
    pub async fn remove_effect(&self, effect_name: &str) -> crate::Result<()> {
        let mut effect_chain = self.effect_chain.write().await;
        effect_chain.remove_effect(effect_name).await
    }

    /// Clear effect chain
    pub async fn clear_effects(&self) -> crate::Result<()> {
        let mut effect_chain = self.effect_chain.write().await;
        effect_chain.clear();
        Ok(())
    }

    /// Get performance statistics (clones current stats)
    pub async fn stats(&self) -> SingingStats {
        self.stats.read().await.clone()
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        stats.reset();
    }

    /// Apply preset configuration
    pub async fn apply_preset(&self, preset_name: &str) -> crate::Result<()> {
        let config = self.config.read().await;

        if let Some(preset) = config.custom_presets.get(preset_name) {
            // Apply preset voice characteristics
            self.set_voice_characteristics(preset.voice.clone()).await?;

            // Apply preset technique
            self.set_technique(preset.technique.clone()).await?;

            // Apply preset effects
            self.clear_effects().await?;
            for effect in &preset.effects {
                let empty_params = HashMap::new();
                self.add_effect(effect, &empty_params).await?;
            }

            Ok(())
        } else {
            Err(crate::Error::Config(format!(
                "Preset '{}' not found",
                preset_name
            )))
        }
    }

    /// Get available presets
    pub async fn list_presets(&self) -> crate::Result<Vec<String>> {
        let config = self.config.read().await;
        Ok(config.custom_presets.keys().cloned().collect())
    }

    /// Validate singing request
    async fn validate_request(&self, request: &SingingRequest) -> crate::Result<()> {
        // Validate score
        request
            .score
            .validate()
            .map_err(|e| crate::Error::Validation(format!("Invalid score: {e}")))?;

        // Validate sample rate
        if request.sample_rate == 0 {
            return Err(crate::Error::Validation(String::from(
                "Invalid sample rate",
            )));
        }

        // Validate voice characteristics
        if request.voice.range.0 >= request.voice.range.1 {
            return Err(crate::Error::Validation(String::from(
                "Invalid voice range",
            )));
        }

        // Validate quality settings
        if request.quality.quality_level > 10 {
            return Err(crate::Error::Validation(String::from(
                "Quality level must be 0-10",
            )));
        }

        Ok(())
    }

    /// Detect format from file extension
    fn detect_format(&self, file_path: &str) -> crate::Result<String> {
        let extension = std::path::Path::new(file_path)
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| crate::Error::Format(String::from("No file extension")))?
            .to_lowercase();

        match extension.as_str() {
            "xml" | "musicxml" => Ok(String::from("musicxml")),
            "mid" | "midi" => Ok(String::from("midi")),
            "json" => Ok(String::from("json")),
            _ => Err(crate::Error::Format(format!(
                "Unsupported format: {}",
                extension
            ))),
        }
    }

    /// Generate melody from text (simplified)
    async fn generate_melody_from_text(
        &self,
        text: &str,
        key: &str,
        tempo: f32,
    ) -> crate::Result<MusicalScore> {
        let mut score = MusicalScore::new("Generated Melody".to_string(), "VoiRS".to_string());
        score.tempo = tempo;

        // Parse key signature
        let key_signature = self.parse_key_signature(key)?;
        score.key_signature = key_signature;

        // Generate notes from text
        let words: Vec<&str> = text.split_whitespace().collect();
        let scale_degrees = [0, 2, 4, 5, 7, 9, 11]; // Major scale

        for (i, word) in words.iter().enumerate() {
            let degree = scale_degrees[i % scale_degrees.len()];
            let note_name = self.degree_to_note_name(degree, &key_signature);

            let event =
                crate::types::NoteEvent::new(note_name, 4, 0.5, 0.8).with_lyric(word.to_string());

            let musical_note = crate::score::MusicalNote::new(event, i as f32 * 0.5, 0.5);
            score.add_note(musical_note);
        }

        Ok(score)
    }

    /// Parse key signature from string
    fn parse_key_signature(&self, key: &str) -> crate::Result<crate::score::KeySignature> {
        let parts: Vec<&str> = key.split_whitespace().collect();
        let root_str = parts
            .get(0)
            .ok_or_else(|| crate::Error::Format("Invalid key signature".to_string()))?;
        let mode_str = parts.get(1).unwrap_or(&"major");

        let root = match *root_str {
            "C" => crate::score::Note::C,
            "D" => crate::score::Note::D,
            "E" => crate::score::Note::E,
            "F" => crate::score::Note::F,
            "G" => crate::score::Note::G,
            "A" => crate::score::Note::A,
            "B" => crate::score::Note::B,
            _ => return Err(crate::Error::Format("Invalid root note".to_string())),
        };

        let mode = match mode_str.to_lowercase().as_str() {
            "major" => crate::score::Mode::Major,
            "minor" => crate::score::Mode::Minor,
            _ => return Err(crate::Error::Format("Invalid mode".to_string())),
        };

        Ok(crate::score::KeySignature {
            root,
            mode,
            accidentals: 0, // Simplified
        })
    }

    /// Convert scale degree to note name
    fn degree_to_note_name(
        &self,
        degree: i32,
        key_signature: &crate::score::KeySignature,
    ) -> String {
        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        let root_offset = match key_signature.root {
            crate::score::Note::C => 0,
            crate::score::Note::D => 2,
            crate::score::Note::E => 4,
            crate::score::Note::F => 5,
            crate::score::Note::G => 7,
            crate::score::Note::A => 9,
            crate::score::Note::B => 11,
        };

        let note_index = (root_offset + degree) % 12;
        note_names[note_index as usize].to_string()
    }
}

impl SingingEngineBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: None,
            voice_characteristics: None,
            technique: None,
            effects: Vec::new(),
            format_support: Vec::new(),
        }
    }

    /// Set configuration
    pub fn config(mut self, config: SingingConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set voice characteristics
    pub fn voice_characteristics(mut self, voice: VoiceCharacteristics) -> Self {
        self.voice_characteristics = Some(voice);
        self
    }

    /// Set singing technique
    pub fn technique(mut self, technique: SingingTechnique) -> Self {
        self.technique = Some(technique);
        self
    }

    /// Add effect to default chain
    pub fn add_effect(mut self, effect: String) -> Self {
        self.effects.push(effect);
        self
    }

    /// Add format support
    pub fn add_format_support(mut self, format: String) -> Self {
        self.format_support.push(format);
        self
    }

    /// Build the singing engine
    pub async fn build(self) -> crate::Result<SingingEngine> {
        let mut config = self.config.unwrap_or_default();

        // Apply builder settings
        if let Some(voice) = self.voice_characteristics {
            config.default_voice = voice;
        }

        if !self.effects.is_empty() {
            config.effects.default_chain = self.effects;
        }

        let engine = SingingEngine::new(config).await?;

        // Apply technique if specified
        if let Some(technique) = self.technique {
            engine.set_technique(technique).await?;
        }

        Ok(engine)
    }
}

impl Default for SingingEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SingingStats {
    /// Update synthesis statistics
    pub fn update_synthesis_stats(
        &mut self,
        processing_time: std::time::Duration,
        synthesis_stats: crate::synthesis::SynthesisStats,
    ) {
        self.total_notes += 1;
        self.processing_time += processing_time;

        // Update running averages
        let alpha = 0.1; // Smoothing factor
        self.pitch_accuracy = self.pitch_accuracy * (1.0 - alpha) + synthesis_stats.quality * alpha;
        self.overall_quality =
            self.overall_quality * (1.0 - alpha) + synthesis_stats.quality * alpha;
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Async-compatible singing engine wrapper
pub struct AsyncSingingEngine {
    engine: Arc<SingingEngine>,
}

impl AsyncSingingEngine {
    /// Create new async singing engine
    pub async fn new(config: SingingConfig) -> crate::Result<Self> {
        let engine = SingingEngine::new(config).await?;
        Ok(Self {
            engine: Arc::new(engine),
        })
    }

    /// Synthesize with async support
    pub async fn synthesize(&self, request: SingingRequest) -> crate::Result<SingingResponse> {
        let engine = self.engine.clone();
        tokio::task::spawn_blocking(move || {
            tokio::runtime::Handle::current().block_on(engine.synthesize(request))
        })
        .await
        .map_err(|e| crate::Error::Processing(e.to_string()))?
    }

    /// Get underlying engine
    pub fn engine(&self) -> &Arc<SingingEngine> {
        &self.engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VoiceType;

    #[tokio::test]
    async fn test_singing_engine_creation() {
        let config = SingingConfig::default();
        let engine = SingingEngine::new(config).await.unwrap();
        assert!(engine.is_enabled().await);
    }

    #[tokio::test]
    async fn test_singing_engine_builder() {
        let config = SingingConfig::default();
        let voice = VoiceCharacteristics {
            voice_type: VoiceType::Soprano,
            ..Default::default()
        };
        let technique = SingingTechnique::classical();

        let engine = SingingEngineBuilder::new()
            .config(config)
            .voice_characteristics(voice)
            .technique(technique)
            .add_effect("vibrato".to_string())
            .add_effect("reverb".to_string())
            .build()
            .await
            .unwrap();

        assert!(engine.is_enabled().await);
    }

    #[tokio::test]
    async fn test_melody_generation() {
        let config = SingingConfig::default();
        let engine = SingingEngine::new(config).await.unwrap();

        let score = engine
            .generate_melody_from_text("Hello world", "C major", 120.0)
            .await
            .unwrap();

        assert_eq!(score.notes.len(), 2);
        assert_eq!(score.tempo, 120.0);
        assert_eq!(score.key_signature.root, crate::score::Note::C);
    }

    #[tokio::test]
    async fn test_key_signature_parsing() {
        let config = SingingConfig::default();
        let engine = SingingEngine::new(config).await.unwrap();

        let key_sig = engine.parse_key_signature("G major").unwrap();
        assert_eq!(key_sig.root, crate::score::Note::G);
        assert_eq!(key_sig.mode, crate::score::Mode::Major);
    }

    #[tokio::test]
    async fn test_format_detection() {
        let config = SingingConfig::default();
        let engine = SingingEngine::new(config).await.unwrap();

        assert_eq!(engine.detect_format("song.musicxml").unwrap(), "musicxml");
        assert_eq!(engine.detect_format("song.mid").unwrap(), "midi");
        assert!(engine.detect_format("song.unknown").is_err());
    }

    #[tokio::test]
    async fn test_async_singing_engine() {
        let config = SingingConfig::default();
        let async_engine = AsyncSingingEngine::new(config).await.unwrap();

        // Test that we can access the underlying engine
        let engine = async_engine.engine();
        assert!(engine.is_enabled().await);
    }

    #[tokio::test]
    async fn test_engine_state_management() {
        let config = SingingConfig::default();
        let mut engine = SingingEngine::new(config).await.unwrap();

        // Test enable/disable
        assert!(engine.is_enabled().await);
        engine.set_enabled(false).await;
        assert!(!engine.is_enabled().await);
        engine.set_enabled(true).await;
        assert!(engine.is_enabled().await);
    }

    #[tokio::test]
    async fn test_stats_management() {
        let config = SingingConfig::default();
        let engine = SingingEngine::new(config).await.unwrap();

        let stats = engine.stats().await;
        assert_eq!(stats.total_notes, 0);

        engine.reset_stats().await;
        let stats = engine.stats().await;
        assert_eq!(stats.total_notes, 0);
    }
}
