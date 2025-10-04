//! Voice-related types and characteristics for singing synthesis

use super::core_types::VoiceType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Voice characteristics for singing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCharacteristics {
    /// Voice type
    pub voice_type: VoiceType,
    /// Vocal range in Hz (min, max)
    pub range: (f32, f32),
    /// Average fundamental frequency
    pub f0_mean: f32,
    /// F0 standard deviation
    pub f0_std: f32,
    /// Vibrato frequency in Hz
    pub vibrato_frequency: f32,
    /// Vibrato depth (0.0-1.0)
    pub vibrato_depth: f32,
    /// Breath capacity in seconds
    pub breath_capacity: f32,
    /// Vocal power (0.0-1.0)
    pub vocal_power: f32,
    /// Resonance characteristics
    pub resonance: HashMap<String, f32>,
    /// Timbre characteristics
    pub timbre: HashMap<String, f32>,
}

impl Default for VoiceCharacteristics {
    /// Create default voice characteristics
    ///
    /// # Returns
    ///
    /// Alto voice with range G3-G5 (196-784 Hz), F0 mean 440 Hz, 6 Hz vibrato at 0.3 depth,
    /// 8 second breath capacity, and 0.8 vocal power
    fn default() -> Self {
        Self {
            voice_type: VoiceType::Alto,
            range: (196.0, 784.0), // G3 to G5
            f0_mean: 440.0,
            f0_std: 50.0,
            vibrato_frequency: 6.0,
            vibrato_depth: 0.3,
            breath_capacity: 8.0,
            vocal_power: 0.8,
            resonance: HashMap::new(),
            timbre: HashMap::new(),
        }
    }
}
