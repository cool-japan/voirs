//! Signal types for real-time emotion adaptation
//!
//! This module provides signal and audio characteristic types used in real-time
//! emotion adaptation:
//! - `EmotionSignal`: External emotion input signal with priority and duration
//! - `AudioCharacteristics`: Audio features for emotion adaptation

use crate::types::{EmotionDimensions, EmotionParameters};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// External emotion input signal
#[derive(Debug, Clone, PartialEq)]
pub struct EmotionSignal {
    /// Target emotion parameters
    pub target: EmotionParameters,
    /// Signal strength (0.0 to 1.0)
    pub strength: f32,
    /// Duration to maintain this emotion
    pub duration: Option<Duration>,
    /// Priority level (higher values override lower)
    pub priority: u8,
    /// Timestamp when signal was created
    pub timestamp: Instant,
}

impl EmotionSignal {
    /// Create new emotion signal
    pub fn new(target: EmotionParameters, strength: f32) -> Self {
        Self {
            target,
            strength: strength.clamp(0.0, 1.0),
            duration: None,
            priority: 0,
            timestamp: Instant::now(),
        }
    }

    /// Set signal duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set signal priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Check if signal has expired
    pub fn is_expired(&self) -> bool {
        if let Some(duration) = self.duration {
            self.timestamp.elapsed() > duration
        } else {
            false
        }
    }
}

/// Audio characteristics for emotion adaptation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioCharacteristics {
    /// RMS energy level (0.0 to 1.0)
    pub energy: f32,
    /// Spectral centroid (brightness measure)
    pub spectral_centroid: f32,
    /// Zero crossing rate (measure of noisiness)
    pub zero_crossing_rate: f32,
    /// Fundamental frequency (pitch)
    pub fundamental_frequency: Option<f32>,
    /// Spectral rolloff
    pub spectral_rolloff: f32,
    /// Tempo/rhythm strength
    pub tempo_strength: f32,
}

impl AudioCharacteristics {
    /// Create from audio samples
    pub fn from_audio(samples: &[f32], sample_rate: f32) -> Self {
        let energy = super::calculate_rms(samples);
        let zcr = super::calculate_zero_crossing_rate(samples);
        let spectral_centroid = super::calculate_spectral_centroid(samples, sample_rate);
        let spectral_rolloff = super::calculate_spectral_rolloff(samples, sample_rate);
        let tempo_strength = super::calculate_tempo_strength(samples, sample_rate);

        Self {
            energy,
            spectral_centroid,
            zero_crossing_rate: zcr,
            fundamental_frequency: None, // Would need pitch detection
            spectral_rolloff,
            tempo_strength,
        }
    }

    /// Map audio characteristics to emotion dimensions
    pub fn to_emotion_dimensions(&self) -> EmotionDimensions {
        // High energy and spectral centroid suggest high arousal
        let arousal = (self.energy * 0.7 + self.spectral_centroid * 0.3).clamp(-1.0, 1.0);

        // Bright sounds (high spectral centroid) tend to be more positive
        let valence =
            (self.spectral_centroid * 0.6 - self.zero_crossing_rate * 0.4).clamp(-1.0, 1.0);

        // Energy and fundamental frequency contribute to dominance
        let dominance = (self.energy * 0.8 + self.tempo_strength * 0.2).clamp(-1.0, 1.0);

        EmotionDimensions::new(valence, arousal, dominance)
    }
}
