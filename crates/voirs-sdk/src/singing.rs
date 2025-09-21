//! Singing voice synthesis integration for VoiRS SDK

use crate::{Result, VoirsError};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Musical note representation
#[derive(Debug, Clone, PartialEq)]
pub struct MusicalNote {
    /// Note name (C, D, E, F, G, A, B)
    pub note: String,
    /// Octave (0-8)
    pub octave: u8,
    /// Pitch in Hz
    pub frequency: f32,
    /// Duration in beats
    pub duration: f32,
    /// Velocity (0.0-1.0)
    pub velocity: f32,
    /// Vibrato intensity (0.0-1.0)
    pub vibrato: f32,
}

/// Singing technique parameters
#[derive(Debug, Clone)]
pub struct SingingTechnique {
    /// Breath control intensity (0.0-1.0)
    pub breath_control: f32,
    /// Vocal fry amount (0.0-1.0)
    pub vocal_fry: f32,
    /// Head voice ratio (0.0-1.0, vs chest voice)
    pub head_voice_ratio: f32,
    /// Vibrato speed (Hz)
    pub vibrato_speed: f32,
    /// Vibrato depth (0.0-1.0)
    pub vibrato_depth: f32,
    /// Pitch bend sensitivity (0.0-1.0)
    pub pitch_bend: f32,
    /// Legato vs staccato (0.0-1.0)
    pub legato: f32,
}

/// Musical score containing notes and timing
#[derive(Debug, Clone)]
pub struct MusicalScore {
    /// List of notes with timing
    pub notes: Vec<MusicalNote>,
    /// Beats per minute
    pub tempo: f32,
    /// Time signature numerator
    pub time_signature_num: u8,
    /// Time signature denominator
    pub time_signature_den: u8,
    /// Key signature
    pub key_signature: String,
}

/// Singing synthesis configuration
#[derive(Debug, Clone)]
pub struct SingingConfig {
    /// Enable singing mode
    pub enabled: bool,
    /// Voice type (soprano, alto, tenor, bass)
    pub voice_type: VoiceType,
    /// Default singing technique
    pub technique: SingingTechnique,
    /// Auto-detect notes from text
    pub auto_pitch_detection: bool,
    /// Cache musical scores
    pub cache_scores: bool,
}

/// Voice type for singing
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceType {
    Soprano,
    Alto,
    Tenor,
    Bass,
}

/// Singing synthesis result
#[derive(Debug, Clone)]
pub struct SingingResult {
    /// Synthesized singing audio
    pub audio: crate::audio::AudioBuffer,
    /// Applied musical score
    pub score: MusicalScore,
    /// Singing technique used
    pub technique: SingingTechnique,
    /// Performance statistics
    pub stats: SingingStats,
}

/// Performance statistics for singing
#[derive(Debug, Clone)]
pub struct SingingStats {
    /// Total notes sung
    pub total_notes: usize,
    /// Average pitch accuracy
    pub pitch_accuracy: f32,
    /// Vibrato consistency
    pub vibrato_consistency: f32,
    /// Breath control quality
    pub breath_quality: f32,
}

/// SDK-integrated singing controller
#[derive(Debug, Clone)]
pub struct SingingController {
    /// Internal singing processor (mock for now)
    config: Arc<RwLock<SingingConfig>>,
    /// Cached musical scores
    score_cache: Arc<RwLock<HashMap<String, MusicalScore>>>,
}

impl SingingController {
    /// Create new singing controller
    pub async fn new() -> Result<Self> {
        Ok(Self {
            config: Arc::new(RwLock::new(SingingConfig::default())),
            score_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Create with custom configuration
    pub async fn with_config(config: SingingConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            score_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Set singing technique
    pub async fn set_technique(&self, technique: SingingTechnique) -> Result<()> {
        let mut config = self.config.write().await;
        config.technique = technique;
        Ok(())
    }

    /// Set voice type
    pub async fn set_voice_type(&self, voice_type: VoiceType) -> Result<()> {
        let mut config = self.config.write().await;
        config.voice_type = voice_type;
        Ok(())
    }

    /// Synthesize singing from musical score
    pub async fn synthesize_score(&self, score: MusicalScore, text: &str) -> Result<SingingResult> {
        let config = self.config.read().await;
        if !config.enabled {
            return Err(VoirsError::ConfigError {
                field: "singing".to_string(),
                message: "Singing synthesis is disabled".to_string(),
            });
        }

        // Cache the score
        {
            let mut cache = self.score_cache.write().await;
            cache.insert(text.to_string(), score.clone());
        }

        // Mock implementation - in reality would use advanced singing synthesis
        let audio = self
            .synthesize_notes(&score.notes, &config.technique)
            .await?;

        Ok(SingingResult {
            audio,
            score: score.clone(),
            technique: config.technique.clone(),
            stats: SingingStats {
                total_notes: score.notes.len(),
                pitch_accuracy: 0.95,
                vibrato_consistency: 0.88,
                breath_quality: 0.92,
            },
        })
    }

    /// Synthesize from text with automatic pitch detection
    pub async fn synthesize_from_text(
        &self,
        text: &str,
        key: &str,
        tempo: f32,
    ) -> Result<SingingResult> {
        let config = self.config.read().await;
        if !config.enabled {
            return Err(VoirsError::ConfigError {
                field: "singing".to_string(),
                message: "Singing synthesis is disabled".to_string(),
            });
        }

        // Auto-generate musical score from text
        let score = self.generate_score_from_text(text, key, tempo).await?;
        self.synthesize_score(score, text).await
    }

    /// Apply singing preset
    pub async fn apply_preset(&self, preset_name: &str) -> Result<()> {
        let technique = match preset_name {
            "classical" => SingingTechnique {
                breath_control: 0.9,
                vocal_fry: 0.1,
                head_voice_ratio: 0.7,
                vibrato_speed: 6.0,
                vibrato_depth: 0.8,
                pitch_bend: 0.3,
                legato: 0.9,
            },
            "pop" => SingingTechnique {
                breath_control: 0.7,
                vocal_fry: 0.3,
                head_voice_ratio: 0.5,
                vibrato_speed: 4.5,
                vibrato_depth: 0.5,
                pitch_bend: 0.6,
                legato: 0.6,
            },
            "jazz" => SingingTechnique {
                breath_control: 0.8,
                vocal_fry: 0.4,
                head_voice_ratio: 0.6,
                vibrato_speed: 5.5,
                vibrato_depth: 0.7,
                pitch_bend: 0.8,
                legato: 0.5,
            },
            "opera" => SingingTechnique {
                breath_control: 1.0,
                vocal_fry: 0.0,
                head_voice_ratio: 0.8,
                vibrato_speed: 7.0,
                vibrato_depth: 0.9,
                pitch_bend: 0.2,
                legato: 1.0,
            },
            _ => {
                return Err(VoirsError::ConfigError {
                    field: "preset".to_string(),
                    message: format!("Unknown singing preset: {}", preset_name),
                })
            }
        };

        self.set_technique(technique).await
    }

    /// Get current singing configuration
    pub async fn get_config(&self) -> SingingConfig {
        self.config.read().await.clone()
    }

    /// Enable or disable singing synthesis
    pub async fn set_enabled(&self, enabled: bool) -> Result<()> {
        let mut config = self.config.write().await;
        config.enabled = enabled;
        Ok(())
    }

    /// Check if singing synthesis is enabled
    pub async fn is_enabled(&self) -> bool {
        let config = self.config.read().await;
        config.enabled
    }

    /// List available singing presets
    pub fn list_presets(&self) -> Vec<String> {
        vec![
            "classical".to_string(),
            "pop".to_string(),
            "jazz".to_string(),
            "opera".to_string(),
        ]
    }

    /// Parse musical score from text format
    pub async fn parse_score(&self, score_text: &str) -> Result<MusicalScore> {
        // Mock implementation - in reality would parse formats like MusicXML, MIDI, etc.
        let lines: Vec<&str> = score_text.lines().collect();
        let mut notes = Vec::new();
        let mut tempo = 120.0;
        let mut key_signature = "C".to_string();

        for line in lines {
            if line.starts_with("TEMPO:") {
                if let Some(tempo_str) = line.split(':').nth(1) {
                    tempo = tempo_str.trim().parse().unwrap_or(120.0);
                }
            } else if line.starts_with("KEY:") {
                if let Some(key_str) = line.split(':').nth(1) {
                    key_signature = key_str.trim().to_string();
                }
            } else if line.starts_with("NOTE:") {
                // Parse note format: NOTE: C4 0.5 0.8 0.3
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    let note_name = parts[1].chars().next().unwrap_or('C').to_string();
                    let octave = parts[1]
                        .chars()
                        .nth(1)
                        .and_then(|c| c.to_digit(10))
                        .unwrap_or(4) as u8;
                    let duration = parts[2].parse().unwrap_or(0.5);
                    let velocity = parts[3].parse().unwrap_or(0.8);
                    let vibrato = parts[4].parse().unwrap_or(0.3);

                    let frequency = self.note_to_frequency(&note_name, octave);
                    notes.push(MusicalNote {
                        note: note_name,
                        octave,
                        frequency,
                        duration,
                        velocity,
                        vibrato,
                    });
                }
            }
        }

        Ok(MusicalScore {
            notes,
            tempo,
            time_signature_num: 4,
            time_signature_den: 4,
            key_signature,
        })
    }

    /// Get cached musical score
    pub async fn get_cached_score(&self, text: &str) -> Option<MusicalScore> {
        let cache = self.score_cache.read().await;
        cache.get(text).cloned()
    }

    /// Clear score cache
    pub async fn clear_cache(&self) -> Result<()> {
        let mut cache = self.score_cache.write().await;
        cache.clear();
        Ok(())
    }

    // Private helper methods

    /// Convert note name and octave to frequency
    fn note_to_frequency(&self, note: &str, octave: u8) -> f32 {
        let base_frequencies = HashMap::from([
            ("C", 261.63),
            ("D", 293.66),
            ("E", 329.63),
            ("F", 349.23),
            ("G", 392.00),
            ("A", 440.00),
            ("B", 493.88),
        ]);

        let base_freq = base_frequencies.get(note).copied().unwrap_or(440.0);
        base_freq * 2.0_f32.powi(octave as i32 - 4)
    }

    /// Generate musical score from text
    async fn generate_score_from_text(
        &self,
        text: &str,
        key: &str,
        tempo: f32,
    ) -> Result<MusicalScore> {
        // Mock implementation - in reality would use advanced text-to-melody algorithms
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut notes = Vec::new();
        let note_names = ["C", "D", "E", "F", "G", "A", "B"];

        for (i, word) in words.iter().enumerate() {
            let note_name = note_names[i % note_names.len()];
            let octave = 4 + (i / note_names.len()) as u8;
            let duration = 0.5 + (word.len() as f32 * 0.1);
            let frequency = self.note_to_frequency(note_name, octave);

            notes.push(MusicalNote {
                note: note_name.to_string(),
                octave,
                frequency,
                duration,
                velocity: 0.8,
                vibrato: 0.4,
            });
        }

        Ok(MusicalScore {
            notes,
            tempo,
            time_signature_num: 4,
            time_signature_den: 4,
            key_signature: key.to_string(),
        })
    }

    /// Synthesize notes into audio
    async fn synthesize_notes(
        &self,
        notes: &[MusicalNote],
        technique: &SingingTechnique,
    ) -> Result<crate::audio::AudioBuffer> {
        // Mock implementation - in reality would use advanced singing synthesis
        let sample_rate = 44100;
        let mut audio_samples = Vec::new();

        for note in notes {
            let note_duration = note.duration;
            let samples_per_note = (note_duration * sample_rate as f32) as usize;

            for i in 0..samples_per_note {
                let t = i as f32 / sample_rate as f32;

                // Basic sine wave with vibrato
                let vibrato_freq = technique.vibrato_speed;
                let vibrato_depth = technique.vibrato_depth * note.vibrato;
                let vibrato_mod =
                    1.0 + vibrato_depth * (2.0 * std::f32::consts::PI * vibrato_freq * t).sin();

                let frequency = note.frequency * vibrato_mod;
                let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * note.velocity;

                // Apply singing technique modifications
                let processed_sample = sample * technique.breath_control;
                audio_samples.push(processed_sample);
            }
        }

        Ok(crate::audio::AudioBuffer::mono(audio_samples, sample_rate))
    }
}

/// Builder for singing controller configuration
#[derive(Debug, Clone)]
pub struct SingingControllerBuilder {
    config: SingingConfig,
}

impl SingingControllerBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: SingingConfig::default(),
        }
    }

    /// Enable or disable singing synthesis
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set voice type
    pub fn voice_type(mut self, voice_type: VoiceType) -> Self {
        self.config.voice_type = voice_type;
        self
    }

    /// Set singing technique
    pub fn technique(mut self, technique: SingingTechnique) -> Self {
        self.config.technique = technique;
        self
    }

    /// Enable auto pitch detection
    pub fn auto_pitch_detection(mut self, enabled: bool) -> Self {
        self.config.auto_pitch_detection = enabled;
        self
    }

    /// Enable score caching
    pub fn cache_scores(mut self, enabled: bool) -> Self {
        self.config.cache_scores = enabled;
        self
    }

    /// Build the singing controller
    pub async fn build(self) -> Result<SingingController> {
        let controller = SingingController::with_config(self.config).await?;
        Ok(controller)
    }
}

impl Default for SingingControllerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SingingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            voice_type: VoiceType::Alto,
            technique: SingingTechnique::default(),
            auto_pitch_detection: false,
            cache_scores: true,
        }
    }
}

impl Default for SingingTechnique {
    fn default() -> Self {
        Self {
            breath_control: 0.8,
            vocal_fry: 0.2,
            head_voice_ratio: 0.5,
            vibrato_speed: 5.0,
            vibrato_depth: 0.6,
            pitch_bend: 0.4,
            legato: 0.7,
        }
    }
}

impl MusicalNote {
    /// Create a new musical note
    pub fn new(note: String, octave: u8, duration: f32, velocity: f32) -> Self {
        let frequency = Self::calculate_frequency(&note, octave);
        Self {
            note,
            octave,
            frequency,
            duration,
            velocity,
            vibrato: 0.5,
        }
    }

    /// Calculate frequency from note name and octave
    fn calculate_frequency(note: &str, octave: u8) -> f32 {
        let base_frequencies = HashMap::from([
            ("C", 261.63),
            ("D", 293.66),
            ("E", 329.63),
            ("F", 349.23),
            ("G", 392.00),
            ("A", 440.00),
            ("B", 493.88),
        ]);

        let base_freq = base_frequencies.get(note).copied().unwrap_or(440.0);
        base_freq * 2.0_f32.powi(octave as i32 - 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_singing_controller_creation() {
        let controller = SingingController::new().await.unwrap();
        assert!(controller.is_enabled().await);
    }

    #[tokio::test]
    async fn test_singing_technique_setting() {
        let controller = SingingController::new().await.unwrap();
        let technique = SingingTechnique {
            breath_control: 0.9,
            vocal_fry: 0.1,
            head_voice_ratio: 0.8,
            vibrato_speed: 6.0,
            vibrato_depth: 0.7,
            pitch_bend: 0.3,
            legato: 0.9,
        };

        controller.set_technique(technique.clone()).await.unwrap();
        let config = controller.get_config().await;
        assert_eq!(config.technique.breath_control, 0.9);
    }

    #[tokio::test]
    async fn test_preset_application() {
        let controller = SingingController::new().await.unwrap();
        controller.apply_preset("classical").await.unwrap();

        let config = controller.get_config().await;
        assert_eq!(config.technique.breath_control, 0.9);
    }

    #[tokio::test]
    async fn test_singing_builder() {
        let controller = SingingControllerBuilder::new()
            .enabled(true)
            .voice_type(VoiceType::Soprano)
            .auto_pitch_detection(true)
            .build()
            .await
            .unwrap();

        assert!(controller.is_enabled().await);
        let config = controller.get_config().await;
        assert_eq!(config.voice_type, VoiceType::Soprano);
    }

    #[tokio::test]
    async fn test_musical_note_creation() {
        let note = MusicalNote::new("A".to_string(), 4, 0.5, 0.8);
        assert_eq!(note.note, "A");
        assert_eq!(note.octave, 4);
        assert!((note.frequency - 440.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_score_parsing() {
        let controller = SingingController::new().await.unwrap();
        let score_text = "TEMPO: 120\nKEY: C\nNOTE: C4 0.5 0.8 0.3\nNOTE: D4 0.5 0.8 0.3";

        let score = controller.parse_score(score_text).await.unwrap();
        assert_eq!(score.tempo, 120.0);
        assert_eq!(score.key_signature, "C");
        assert_eq!(score.notes.len(), 2);
    }

    #[tokio::test]
    async fn test_text_to_melody_generation() {
        let controller = SingingController::new().await.unwrap();
        let result = controller
            .synthesize_from_text("Hello world", "C", 120.0)
            .await
            .unwrap();

        assert_eq!(result.score.notes.len(), 2); // Two words
        assert!(result.audio.duration() > 0.0);
    }

    #[tokio::test]
    async fn test_preset_listing() {
        let controller = SingingController::new().await.unwrap();
        let presets = controller.list_presets();
        assert!(presets.contains(&"classical".to_string()));
        assert!(presets.contains(&"pop".to_string()));
        assert!(presets.contains(&"jazz".to_string()));
        assert!(presets.contains(&"opera".to_string()));
    }
}
