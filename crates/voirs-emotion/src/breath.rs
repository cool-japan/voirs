//! Breath Control and Pause Modeling for Natural Speech
//!
//! This module provides sophisticated breath control and pause insertion
//! for natural-sounding emotional speech synthesis.
//!
//! ## Features
//!
//! - **Breath Noise Generation**: Realistic breath sounds with emotion-specific characteristics
//! - **Pause Detection**: Intelligent pause placement based on linguistic and emotional context
//! - **Breath Timing**: Emotion-aware breath placement and duration
//! - **Micro-pauses**: Subtle hesitations and timing variations

use crate::{types::Emotion, Result};
use serde::{Deserialize, Serialize};

/// Breath control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathConfig {
    /// Enable breath insertion
    pub enabled: bool,
    /// Breath intensity (0.0-1.0)
    pub intensity: f32,
    /// Breath frequency (breaths per minute)
    pub frequency: f32,
    /// Breath duration (seconds)
    pub duration: f32,
    /// Pause sensitivity (0.0-1.0)
    pub pause_sensitivity: f32,
}

impl Default for BreathConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            intensity: 0.3,
            frequency: 15.0, // Normal resting rate
            duration: 0.15,
            pause_sensitivity: 0.5,
        }
    }
}

impl BreathConfig {
    /// Create config for calm breathing
    pub fn calm() -> Self {
        Self {
            enabled: true,
            intensity: 0.25,
            frequency: 12.0, // Slower
            duration: 0.2,   // Longer
            pause_sensitivity: 0.6,
        }
    }

    /// Create config for excited/anxious breathing
    pub fn excited() -> Self {
        Self {
            enabled: true,
            intensity: 0.4,
            frequency: 20.0,        // Faster
            duration: 0.1,          // Shorter
            pause_sensitivity: 0.3, // Fewer pauses
        }
    }

    /// Create config for tired/sad breathing
    pub fn tired() -> Self {
        Self {
            enabled: true,
            intensity: 0.35,
            frequency: 14.0,
            duration: 0.18,
            pause_sensitivity: 0.7, // More pauses
        }
    }

    /// Create config from emotion
    pub fn from_emotion(emotion: Emotion) -> Self {
        match emotion {
            Emotion::Calm => Self::calm(),
            Emotion::Excited | Emotion::Fear | Emotion::Angry => Self::excited(),
            Emotion::Sad | Emotion::Melancholic => Self::tired(),
            _ => Self::default(),
        }
    }
}

/// Pause type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PauseType {
    /// Brief pause at phrase boundary
    Phrase,
    /// Longer pause at sentence boundary
    Sentence,
    /// Micro-pause for hesitation
    Hesitation,
    /// Breath pause
    Breath,
}

/// Pause insertion point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pause {
    /// Type of pause
    pub pause_type: PauseType,
    /// Position in audio (samples)
    pub position: usize,
    /// Duration (samples)
    pub duration: usize,
    /// Whether to insert breath sound
    pub insert_breath: bool,
}

/// Breath generator for natural speech
pub struct BreathGenerator {
    /// Configuration
    config: BreathConfig,
    /// Sample rate
    sample_rate: f32,
    /// Time since last breath (samples)
    samples_since_breath: usize,
}

impl BreathGenerator {
    /// Create a new breath generator
    pub fn new(config: BreathConfig, sample_rate: f32) -> Self {
        Self {
            config,
            sample_rate,
            samples_since_breath: 0,
        }
    }

    /// Update configuration
    pub fn set_config(&mut self, config: BreathConfig) {
        self.config = config;
    }

    /// Generate breath sound
    pub fn generate_breath(&self, duration_samples: usize) -> Vec<f32> {
        let mut breath = vec![0.0; duration_samples];

        if !self.config.enabled {
            return breath;
        }

        // Generate breath noise (filtered white noise)
        for sample in breath.iter_mut() {
            *sample = (fastrand::f32() * 2.0 - 1.0) * self.config.intensity;
        }

        // Apply envelope (fade in/out)
        let fade_samples = (duration_samples as f32 * 0.3) as usize;

        for i in 0..fade_samples {
            let fade_in = i as f32 / fade_samples as f32;
            breath[i] *= fade_in;

            let fade_out_idx = duration_samples - 1 - i;
            if fade_out_idx < duration_samples {
                breath[fade_out_idx] *= fade_in;
            }
        }

        // Apply bandpass filter (breath is 100-1000 Hz)
        self.apply_breath_filter(&mut breath);

        breath
    }

    /// Apply spectral shaping for breath sound
    fn apply_breath_filter(&self, audio: &mut [f32]) {
        if audio.is_empty() {
            return;
        }

        // Simple lowpass filter for breath-like sound
        let cutoff = 1500.0; // Hz
        let alpha = 1.0 - (-2.0 * std::f32::consts::PI * cutoff / self.sample_rate).exp();

        let mut prev = audio[0];
        for sample in audio.iter_mut() {
            let filtered = *sample * alpha + prev * (1.0 - alpha);
            prev = filtered;
            *sample = filtered;
        }
    }

    /// Check if breath should be inserted
    pub fn should_insert_breath(&mut self, current_position: usize, pause_duration: usize) -> bool {
        if !self.config.enabled {
            return false;
        }

        let breath_interval = (60.0 * self.sample_rate / self.config.frequency) as usize;

        // Check if enough time has passed and pause is long enough
        let min_pause_for_breath = (0.3 * self.sample_rate) as usize;

        if self.samples_since_breath >= breath_interval && pause_duration >= min_pause_for_breath {
            self.samples_since_breath = 0;
            true
        } else {
            self.samples_since_breath += current_position;
            false
        }
    }

    /// Reset breath timing
    pub fn reset(&mut self) {
        self.samples_since_breath = 0;
    }
}

/// Pause analyzer for linguistic context
pub struct PauseAnalyzer {
    /// Configuration
    config: BreathConfig,
    /// Sample rate
    sample_rate: f32,
}

impl PauseAnalyzer {
    /// Create a new pause analyzer
    pub fn new(config: BreathConfig, sample_rate: f32) -> Self {
        Self {
            config,
            sample_rate,
        }
    }

    /// Determine pause duration based on context
    pub fn calculate_pause_duration(&self, pause_type: PauseType, emotion: &Emotion) -> usize {
        let base_duration = match pause_type {
            PauseType::Phrase => 0.15,     // 150ms
            PauseType::Sentence => 0.4,    // 400ms
            PauseType::Hesitation => 0.08, // 80ms
            PauseType::Breath => self.config.duration,
        };

        // Adjust for emotion
        let emotion_factor = match emotion {
            Emotion::Excited | Emotion::Angry => 0.7, // Shorter pauses
            Emotion::Sad | Emotion::Calm => 1.3,      // Longer pauses
            Emotion::Fear => 0.9,                     // Slightly shorter
            _ => 1.0,
        };

        // Apply sensitivity
        let adjusted_duration = base_duration * emotion_factor * self.config.pause_sensitivity;

        (adjusted_duration * self.sample_rate) as usize
    }

    /// Detect pause positions in text
    ///
    /// This is a simplified implementation. Production code would use
    /// linguistic analysis for better pause placement.
    pub fn detect_pauses(&self, text: &str, emotion: &Emotion) -> Vec<(usize, PauseType)> {
        let mut pauses = Vec::new();
        let mut char_position = 0;

        for (i, ch) in text.chars().enumerate() {
            match ch {
                '.' | '!' | '?' => {
                    pauses.push((char_position, PauseType::Sentence));
                }
                ',' | ';' | ':' => {
                    pauses.push((char_position, PauseType::Phrase));
                }
                _ => {}
            }

            char_position += 1;

            // Add hesitations for certain emotions
            if matches!(*emotion, Emotion::Fear | Emotion::Sad) && fastrand::f32() < 0.05 {
                pauses.push((char_position, PauseType::Hesitation));
            }
        }

        pauses
    }

    /// Create pause with breath if needed
    pub fn create_pause(
        &self,
        pause_type: PauseType,
        position: usize,
        emotion: &Emotion,
        insert_breath: bool,
    ) -> Pause {
        let duration = self.calculate_pause_duration(pause_type, emotion);

        Pause {
            pause_type,
            position,
            duration,
            insert_breath: insert_breath && pause_type != PauseType::Hesitation,
        }
    }
}

/// Manages breath and pause insertion for natural speech
pub struct BreathPauseController {
    /// Breath generator
    breath_generator: BreathGenerator,
    /// Pause analyzer
    pause_analyzer: PauseAnalyzer,
    /// Sample rate
    sample_rate: f32,
}

impl BreathPauseController {
    /// Create a new controller
    pub fn new(config: BreathConfig, sample_rate: f32) -> Self {
        Self {
            breath_generator: BreathGenerator::new(config.clone(), sample_rate),
            pause_analyzer: PauseAnalyzer::new(config, sample_rate),
            sample_rate,
        }
    }

    /// Process text and generate pauses with breaths
    pub fn process_text(&mut self, text: &str, emotion: &Emotion) -> Vec<Pause> {
        let detected_pauses = self.pause_analyzer.detect_pauses(text, emotion);
        let mut pauses = Vec::new();

        for (position, pause_type) in detected_pauses {
            let duration = self
                .pause_analyzer
                .calculate_pause_duration(pause_type, emotion);
            let should_breathe = self
                .breath_generator
                .should_insert_breath(position, duration);

            let pause =
                self.pause_analyzer
                    .create_pause(pause_type, position, emotion, should_breathe);

            pauses.push(pause);
        }

        pauses
    }

    /// Insert pauses and breaths into audio
    pub fn insert_pauses(&mut self, audio: &[f32], pauses: &[Pause]) -> Vec<f32> {
        let mut result = Vec::new();
        let mut last_position = 0;

        for pause in pauses {
            // Copy audio up to pause point
            if pause.position < audio.len() {
                result.extend_from_slice(&audio[last_position..pause.position]);
            }

            // Insert silence or breath
            if pause.insert_breath {
                let breath = self.breath_generator.generate_breath(pause.duration);
                result.extend_from_slice(&breath);
            } else {
                result.extend(vec![0.0; pause.duration]);
            }

            last_position = pause.position;
        }

        // Copy remaining audio
        if last_position < audio.len() {
            result.extend_from_slice(&audio[last_position..]);
        }

        result
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.breath_generator.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_breath_config_creation() {
        let config = BreathConfig::default();
        assert!(config.enabled);
        assert!(config.intensity > 0.0);
    }

    #[test]
    fn test_emotion_breath_configs() {
        let calm = BreathConfig::calm();
        let excited = BreathConfig::excited();

        // Calm should be slower
        assert!(calm.frequency < excited.frequency);
        // Calm should have longer breaths
        assert!(calm.duration > excited.duration);
    }

    #[test]
    fn test_breath_generator() {
        let config = BreathConfig::default();
        let generator = BreathGenerator::new(config, 44100.0);

        let breath = generator.generate_breath(4410); // 100ms at 44.1kHz

        // Breath should be generated
        assert_eq!(breath.len(), 4410);
        // Should have non-zero content
        assert!(breath.iter().any(|&x| x.abs() > 0.01));
    }

    #[test]
    fn test_breath_envelope() {
        let config = BreathConfig::default();
        let generator = BreathGenerator::new(config, 44100.0);

        let breath = generator.generate_breath(1000);

        // Test envelope shape by averaging multiple samples in each region
        // This is more robust than comparing single samples which are random
        let start_avg: f32 = breath[5..15].iter().map(|x| x.abs()).sum::<f32>() / 10.0;
        let mid_avg: f32 = breath[495..505].iter().map(|x| x.abs()).sum::<f32>() / 10.0;
        let end_avg: f32 = breath[985..995].iter().map(|x| x.abs()).sum::<f32>() / 10.0;

        // Middle section should be louder than start/end due to envelope
        assert!(
            mid_avg > start_avg,
            "Middle average ({}) should be > start average ({})",
            mid_avg,
            start_avg
        );
        assert!(
            mid_avg > end_avg,
            "Middle average ({}) should be > end average ({})",
            mid_avg,
            end_avg
        );
    }

    #[test]
    fn test_pause_analyzer() {
        let config = BreathConfig::default();
        let analyzer = PauseAnalyzer::new(config, 44100.0);

        let text = "Hello, world. How are you?";
        let pauses = analyzer.detect_pauses(text, &Emotion::Neutral);

        // Should detect pauses at comma, period, and question mark
        assert!(pauses.len() >= 3);
    }

    #[test]
    fn test_pause_duration_calculation() {
        let config = BreathConfig::default();
        let analyzer = PauseAnalyzer::new(config, 44100.0);

        let sentence_duration =
            analyzer.calculate_pause_duration(PauseType::Sentence, &Emotion::Neutral);
        let phrase_duration =
            analyzer.calculate_pause_duration(PauseType::Phrase, &Emotion::Neutral);

        // Sentence pauses should be longer
        assert!(sentence_duration > phrase_duration);
    }

    #[test]
    fn test_emotion_pause_adjustment() {
        let config = BreathConfig::default();
        let analyzer = PauseAnalyzer::new(config, 44100.0);

        let normal_duration =
            analyzer.calculate_pause_duration(PauseType::Sentence, &Emotion::Neutral);
        let excited_duration =
            analyzer.calculate_pause_duration(PauseType::Sentence, &Emotion::Excited);
        let sad_duration = analyzer.calculate_pause_duration(PauseType::Sentence, &Emotion::Sad);

        // Excited should have shorter pauses
        assert!(excited_duration < normal_duration);
        // Sad should have longer pauses
        assert!(sad_duration > normal_duration);
    }

    #[test]
    fn test_breath_pause_controller() {
        let config = BreathConfig::default();
        let mut controller = BreathPauseController::new(config, 44100.0);

        let text = "Hello, world.";
        let pauses = controller.process_text(text, &Emotion::Neutral);

        assert!(!pauses.is_empty());
    }

    #[test]
    fn test_pause_insertion() {
        let config = BreathConfig::default();
        let mut controller = BreathPauseController::new(config, 44100.0);

        let audio = vec![1.0; 44100]; // 1 second
        let pauses = vec![Pause {
            pause_type: PauseType::Phrase,
            position: 22050,
            duration: 4410,
            insert_breath: false,
        }];

        let result = controller.insert_pauses(&audio, &pauses);

        // Result should be longer due to pause
        assert!(result.len() > audio.len());
    }

    #[test]
    fn test_breath_insertion() {
        let config = BreathConfig::default();
        let mut controller = BreathPauseController::new(config, 44100.0);

        let audio = vec![1.0; 44100];
        let pauses = vec![Pause {
            pause_type: PauseType::Breath,
            position: 22050,
            duration: 6615,
            insert_breath: true,
        }];

        let result = controller.insert_pauses(&audio, &pauses);

        // Should have breath sound inserted
        assert!(result.len() > audio.len());
    }
}
