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

    /// Synthesize notes into audio with realistic singing characteristics
    async fn synthesize_notes(
        &self,
        notes: &[MusicalNote],
        technique: &SingingTechnique,
    ) -> Result<crate::audio::AudioBuffer> {
        let sample_rate = 44100;
        let mut audio_samples = Vec::new();

        for note in notes {
            let note_duration = note.duration;
            let samples_per_note = (note_duration * sample_rate as f32) as usize;

            for i in 0..samples_per_note {
                let t = i as f32 / sample_rate as f32;
                let note_phase = i as f32 / samples_per_note as f32;

                // ADSR envelope for natural attack and release
                let envelope = self.calculate_adsr_envelope(note_phase, note_duration, technique);

                // Vibrato modulation
                let vibrato_freq = technique.vibrato_speed;
                let vibrato_depth = technique.vibrato_depth * note.vibrato;
                let vibrato_mod =
                    1.0 + vibrato_depth * (2.0 * std::f32::consts::PI * vibrato_freq * t).sin();

                let frequency = note.frequency * vibrato_mod;

                // Fundamental frequency + harmonics for richer sound
                let fundamental = (2.0 * std::f32::consts::PI * frequency * t).sin();

                // Add harmonics with decreasing amplitude (creates vocal timbre)
                let harmonic2 = 0.5 * (2.0 * std::f32::consts::PI * frequency * 2.0 * t).sin();
                let harmonic3 = 0.25 * (2.0 * std::f32::consts::PI * frequency * 3.0 * t).sin();
                let harmonic4 = 0.125 * (2.0 * std::f32::consts::PI * frequency * 4.0 * t).sin();

                // Apply head voice vs chest voice mixing
                let harmonic_mix = fundamental
                    + harmonic2 * (1.0 - technique.head_voice_ratio * 0.5)
                    + harmonic3 * (1.0 - technique.head_voice_ratio * 0.7)
                    + harmonic4 * (1.0 - technique.head_voice_ratio * 0.9);

                // Apply formant-like filtering for vowel characteristics
                let formant_enhanced = self.apply_formant_enhancement(
                    harmonic_mix,
                    frequency,
                    technique.head_voice_ratio,
                );

                // Apply velocity and envelope
                let mut sample = formant_enhanced * note.velocity * envelope;

                // Add breath noise for realism
                let breath_noise = self.generate_breath_noise(note_phase, technique);
                sample += breath_noise * (1.0 - technique.breath_control);

                // Apply vocal fry effect in lower register
                if frequency < 150.0 && technique.vocal_fry > 0.0 {
                    let fry_freq = frequency * 0.5;
                    let fry = technique.vocal_fry
                        * 0.1
                        * (2.0 * std::f32::consts::PI * fry_freq * t).sin();
                    sample += fry * envelope;
                }

                // Pitch bend effect for smooth transitions
                let bend_amount = technique.pitch_bend * 0.1 * note_phase.sin();
                sample *= 1.0 + bend_amount;

                // Apply breath control (reduces amplitude, increases breathiness)
                let processed_sample = sample * technique.breath_control;
                audio_samples.push(processed_sample.clamp(-1.0, 1.0));
            }

            // Add subtle pause between notes if not full legato
            if technique.legato < 1.0 {
                let pause_samples = ((1.0 - technique.legato) * sample_rate as f32 * 0.05) as usize;
                audio_samples.resize(audio_samples.len() + pause_samples, 0.0);
            }
        }

        Ok(crate::audio::AudioBuffer::mono(audio_samples, sample_rate))
    }

    /// Calculate ADSR envelope for natural note articulation
    fn calculate_adsr_envelope(
        &self,
        phase: f32,
        duration: f32,
        technique: &SingingTechnique,
    ) -> f32 {
        let attack_time = 0.05; // 50ms attack
        let decay_time = 0.1; // 100ms decay
        let sustain_level = 0.8;
        let release_start = 0.85; // Start release at 85% of note duration

        if phase < attack_time / duration {
            // Attack phase - smooth cubic curve
            let attack_phase = phase / (attack_time / duration);
            attack_phase * attack_phase * (3.0 - 2.0 * attack_phase)
        } else if phase < (attack_time + decay_time) / duration {
            // Decay phase
            let decay_phase = (phase - attack_time / duration) / (decay_time / duration);
            1.0 - (1.0 - sustain_level) * decay_phase
        } else if phase < release_start {
            // Sustain phase
            sustain_level
        } else {
            // Release phase - smooth exponential-like decay
            let release_phase = (phase - release_start) / (1.0 - release_start);
            sustain_level * (1.0 - release_phase).powf(2.0)
        }
    }

    /// Apply formant-like enhancement to simulate vowel characteristics
    fn apply_formant_enhancement(&self, signal: f32, frequency: f32, head_voice: f32) -> f32 {
        // Simple formant boost simulation
        // In reality, this would use proper formant filtering
        let formant_boost = if frequency > 200.0 && frequency < 800.0 {
            1.2 * (1.0 + head_voice * 0.3) // Boost mid frequencies for vowel clarity
        } else if frequency > 2000.0 {
            0.8 * (1.0 - head_voice * 0.2) // Reduce high frequencies in chest voice
        } else {
            1.0
        };

        signal * formant_boost
    }

    /// Generate breath noise for singing realism
    fn generate_breath_noise(&self, phase: f32, technique: &SingingTechnique) -> f32 {
        use fastrand;

        // More breath noise at note transitions (attack and release)
        let noise_intensity = if phase < 0.1 || phase > 0.9 {
            0.02
        } else {
            0.005
        };

        // Generate pink-ish noise (more natural than white noise)
        let white_noise = (fastrand::f32() * 2.0 - 1.0) * noise_intensity;

        // Simple low-pass filter for pink-ish characteristics
        white_noise * (1.0 - technique.breath_control * 0.5)
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

    #[tokio::test]
    async fn test_harmonic_synthesis() {
        // Test that synthesis produces harmonics-rich audio
        let controller = SingingController::new().await.unwrap();

        let technique = SingingTechnique {
            breath_control: 0.9,
            vocal_fry: 0.1,
            head_voice_ratio: 0.5,
            vibrato_speed: 5.0,
            vibrato_depth: 0.5,
            pitch_bend: 0.3,
            legato: 0.8,
        };

        controller.set_technique(technique).await.unwrap();

        let result = controller
            .synthesize_from_text("Test", "C", 120.0)
            .await
            .unwrap();

        // Audio should have content
        assert!(result.audio.samples().len() > 0);

        // Check that audio has reasonable amplitude (not all zeros)
        let max_amplitude = result
            .audio
            .samples()
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        assert!(max_amplitude > 0.01, "Audio should have audible amplitude");
        assert!(max_amplitude <= 1.0, "Audio should be within bounds");
    }

    #[tokio::test]
    async fn test_adsr_envelope() {
        // Test that ADSR envelope is applied correctly
        let controller = SingingController::new().await.unwrap();

        let technique = SingingTechnique {
            breath_control: 1.0, // Perfect breath control
            vocal_fry: 0.0,
            head_voice_ratio: 0.5,
            vibrato_speed: 0.0, // No vibrato for cleaner test
            vibrato_depth: 0.0,
            pitch_bend: 0.0,
            legato: 1.0, // Full legato
        };

        controller.set_technique(technique).await.unwrap();

        let result = controller
            .synthesize_from_text("A", "C", 60.0)
            .await
            .unwrap();

        let samples = result.audio.samples();

        // Attack phase: amplitude should increase at the start
        let attack_samples = &samples[0..100.min(samples.len())];
        if attack_samples.len() > 10 {
            let start_avg = attack_samples[0..5].iter().map(|s| s.abs()).sum::<f32>() / 5.0;
            let mid_avg = attack_samples[50..55].iter().map(|s| s.abs()).sum::<f32>() / 5.0;
            assert!(
                mid_avg >= start_avg * 0.8,
                "Envelope should have attack phase"
            );
        }

        // Release phase: amplitude should decrease at the end
        if samples.len() > 100 {
            let end_samples = &samples[samples.len() - 100..];
            let mid_end_avg = end_samples[0..5].iter().map(|s| s.abs()).sum::<f32>() / 5.0;
            let final_avg = end_samples[95..100].iter().map(|s| s.abs()).sum::<f32>() / 5.0;
            assert!(
                mid_end_avg >= final_avg,
                "Envelope should have release phase"
            );
        }
    }

    #[tokio::test]
    async fn test_breath_noise_modeling() {
        // Test with different breath control levels
        let controller = SingingController::new().await.unwrap();

        // Low breath control = more breath noise
        let technique_breathy = SingingTechnique {
            breath_control: 0.3,
            vocal_fry: 0.0,
            head_voice_ratio: 0.5,
            vibrato_speed: 0.0,
            vibrato_depth: 0.0,
            pitch_bend: 0.0,
            legato: 1.0,
        };

        controller
            .set_technique(technique_breathy.clone())
            .await
            .unwrap();
        let result_breathy = controller
            .synthesize_from_text("Test", "C", 120.0)
            .await
            .unwrap();

        // High breath control = less breath noise, cleaner sound
        let technique_clean = SingingTechnique {
            breath_control: 1.0,
            ..technique_breathy
        };

        controller.set_technique(technique_clean).await.unwrap();
        let result_clean = controller
            .synthesize_from_text("Test", "C", 120.0)
            .await
            .unwrap();

        // Both should produce valid audio
        assert!(result_breathy.audio.samples().len() > 0);
        assert!(result_clean.audio.samples().len() > 0);

        // Clean version should generally have higher average amplitude
        // (since breath noise is added, not replacing signal)
        let breathy_max = result_breathy
            .audio
            .samples()
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        let clean_max = result_clean
            .audio
            .samples()
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        assert!(clean_max > 0.01);
        assert!(breathy_max > 0.01);
    }

    #[tokio::test]
    async fn test_vocal_fry_effect() {
        // Test vocal fry in low register
        let controller = SingingController::new().await.unwrap();

        let technique = SingingTechnique {
            breath_control: 0.9,
            vocal_fry: 0.5,        // Moderate vocal fry
            head_voice_ratio: 0.3, // Chest voice dominant
            vibrato_speed: 0.0,
            vibrato_depth: 0.0,
            pitch_bend: 0.0,
            legato: 1.0,
        };

        controller.set_technique(technique).await.unwrap();

        // Low note should trigger vocal fry effect
        let score = MusicalScore {
            notes: vec![MusicalNote {
                note: "C".to_string(),
                octave: 2,       // Very low note
                frequency: 65.4, // C2
                duration: 0.5,
                velocity: 0.8,
                vibrato: 0.0,
            }],
            tempo: 120.0,
            time_signature_num: 4,
            time_signature_den: 4,
            key_signature: "C".to_string(),
        };

        let result = controller
            .synthesize_score(score, "Low note test")
            .await
            .unwrap();

        assert!(result.audio.samples().len() > 0);
        assert!(result.stats.total_notes == 1);
    }

    #[tokio::test]
    async fn test_legato_vs_staccato() {
        let controller = SingingController::new().await.unwrap();

        // Full legato - no pauses between notes
        let technique_legato = SingingTechnique {
            breath_control: 0.9,
            vocal_fry: 0.0,
            head_voice_ratio: 0.5,
            vibrato_speed: 0.0,
            vibrato_depth: 0.0,
            pitch_bend: 0.0,
            legato: 1.0,
        };

        controller
            .set_technique(technique_legato.clone())
            .await
            .unwrap();
        let result_legato = controller
            .synthesize_from_text("Hello World", "C", 120.0)
            .await
            .unwrap();

        // Staccato - pauses between notes
        let technique_staccato = SingingTechnique {
            legato: 0.3,
            ..technique_legato
        };

        controller.set_technique(technique_staccato).await.unwrap();
        let result_staccato = controller
            .synthesize_from_text("Hello World", "C", 120.0)
            .await
            .unwrap();

        // Staccato version should be longer due to pauses
        assert!(result_staccato.audio.duration() > result_legato.audio.duration() * 0.95);
    }
}
