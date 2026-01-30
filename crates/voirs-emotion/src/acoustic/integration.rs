//! VoiRS acoustic integration types
//!
//! This module provides types for integrating with the voirs-acoustic crate,
//! enabling seamless emotion control in acoustic synthesis.

use std::collections::HashMap;

/// Configuration compatible with voirs-acoustic emotion processing
#[derive(Debug, Clone, PartialEq)]
pub struct VoirsAcousticEmotionConfig {
    // Basic prosody parameters
    /// Pitch shift multiplier (1.0 = no change, >1.0 = higher, <1.0 = lower)
    pub pitch_shift: f32,
    /// Tempo scaling factor (1.0 = normal, <1.0 = slower, >1.0 = faster)
    pub tempo_scale: f32,
    /// Energy level scaling (1.0 = normal, <1.0 = quieter, >1.0 = louder)
    pub energy_scale: f32,

    // Voice quality parameters
    /// Breathiness level (0.0 = clear, 1.0 = very breathy)
    pub breathiness: f32,
    /// Roughness level (0.0 = smooth, 1.0 = very rough)
    pub roughness: f32,
    /// Brightness of voice timbre (0.0 = dark, 1.0 = bright)
    pub brightness: f32,
    /// Resonance strength (0.0 = minimal, 1.0 = maximum)
    pub resonance: f32,

    // Emotion dimensions
    /// Emotional valence from -1.0 (negative) to +1.0 (positive)
    pub valence: f32,
    /// Arousal level from 0.0 (calm) to 1.0 (excited)
    pub arousal: f32,
    /// Dominance level from 0.0 (submissive) to 1.0 (dominant)
    pub dominance: f32,

    // Acoustic conditioning parameters
    /// Energy boost applied to signal (0.0 = none, 1.0 = maximum)
    pub energy_boost: f32,
    /// Spectral brightness enhancement (0.0 = none, 1.0 = maximum)
    pub spectral_brightness: f32,
    /// Harmonic richness enhancement (0.0 = none, 1.0 = maximum)
    pub harmonic_richness: f32,
    /// Temporal dynamics modification strength (0.0 = none, 1.0 = maximum)
    pub temporal_dynamics: f32,

    // Speaker adaptation parameters
    /// Pitch range expansion factor (1.0 = normal, >1.0 = wider range)
    pub pitch_range_expansion: f32,
    /// Formant frequency shift in semitones
    pub formant_shift: f32,
    /// Voice quality adjustment strength (0.0 = none, 1.0 = maximum)
    pub voice_quality_adjustment: f32,

    // Prosody modification parameters
    /// Pitch contour variation strength (0.0 = flat, 1.0 = highly varied)
    pub pitch_contour_variation: f32,
    /// Rhythm modification strength (0.0 = none, 1.0 = maximum)
    pub rhythm_modification: f32,
    /// Stress pattern enhancement level (0.0 = none, 1.0 = maximum)
    pub stress_pattern_enhancement: f32,
}

impl Default for VoirsAcousticEmotionConfig {
    fn default() -> Self {
        Self {
            // Basic prosody parameters
            pitch_shift: 1.0,
            tempo_scale: 1.0,
            energy_scale: 1.0,

            // Voice quality parameters
            breathiness: 0.0,
            roughness: 0.0,
            brightness: 0.0,
            resonance: 0.0,

            // Emotion dimensions
            valence: 0.0,
            arousal: 0.0,
            dominance: 0.0,

            // Acoustic conditioning parameters
            energy_boost: 1.0,
            spectral_brightness: 0.0,
            harmonic_richness: 1.0,
            temporal_dynamics: 1.0,

            // Speaker adaptation parameters
            pitch_range_expansion: 1.0,
            formant_shift: 1.0,
            voice_quality_adjustment: 0.0,

            // Prosody modification parameters
            pitch_contour_variation: 1.0,
            rhythm_modification: 1.0,
            stress_pattern_enhancement: 1.0,
        }
    }
}

impl VoirsAcousticEmotionConfig {
    /// Convert to voirs-acoustic EmotionConfig format
    pub fn to_voirs_acoustic_format(&self) -> HashMap<String, f32> {
        let mut params = HashMap::new();

        // Add all parameters as key-value pairs for flexibility
        params.insert("pitch_shift".to_string(), self.pitch_shift);
        params.insert("tempo_scale".to_string(), self.tempo_scale);
        params.insert("energy_scale".to_string(), self.energy_scale);
        params.insert("breathiness".to_string(), self.breathiness);
        params.insert("roughness".to_string(), self.roughness);
        params.insert("brightness".to_string(), self.brightness);
        params.insert("resonance".to_string(), self.resonance);
        params.insert("valence".to_string(), self.valence);
        params.insert("arousal".to_string(), self.arousal);
        params.insert("dominance".to_string(), self.dominance);
        params.insert("energy_boost".to_string(), self.energy_boost);
        params.insert("spectral_brightness".to_string(), self.spectral_brightness);
        params.insert("harmonic_richness".to_string(), self.harmonic_richness);
        params.insert("temporal_dynamics".to_string(), self.temporal_dynamics);
        params.insert(
            "pitch_range_expansion".to_string(),
            self.pitch_range_expansion,
        );
        params.insert("formant_shift".to_string(), self.formant_shift);
        params.insert(
            "voice_quality_adjustment".to_string(),
            self.voice_quality_adjustment,
        );
        params.insert(
            "pitch_contour_variation".to_string(),
            self.pitch_contour_variation,
        );
        params.insert("rhythm_modification".to_string(), self.rhythm_modification);
        params.insert(
            "stress_pattern_enhancement".to_string(),
            self.stress_pattern_enhancement,
        );

        params
    }

    /// Create from voirs-acoustic EmotionConfig
    pub fn from_voirs_acoustic_format(params: &HashMap<String, f32>) -> Self {
        let mut config = Self::default();

        // Extract parameters with defaults
        config.pitch_shift = params.get("pitch_shift").copied().unwrap_or(1.0);
        config.tempo_scale = params.get("tempo_scale").copied().unwrap_or(1.0);
        config.energy_scale = params.get("energy_scale").copied().unwrap_or(1.0);
        config.breathiness = params.get("breathiness").copied().unwrap_or(0.0);
        config.roughness = params.get("roughness").copied().unwrap_or(0.0);
        config.brightness = params.get("brightness").copied().unwrap_or(0.0);
        config.resonance = params.get("resonance").copied().unwrap_or(0.0);
        config.valence = params.get("valence").copied().unwrap_or(0.0);
        config.arousal = params.get("arousal").copied().unwrap_or(0.0);
        config.dominance = params.get("dominance").copied().unwrap_or(0.0);
        config.energy_boost = params.get("energy_boost").copied().unwrap_or(1.0);
        config.spectral_brightness = params.get("spectral_brightness").copied().unwrap_or(0.0);
        config.harmonic_richness = params.get("harmonic_richness").copied().unwrap_or(1.0);
        config.temporal_dynamics = params.get("temporal_dynamics").copied().unwrap_or(1.0);
        config.pitch_range_expansion = params.get("pitch_range_expansion").copied().unwrap_or(1.0);
        config.formant_shift = params.get("formant_shift").copied().unwrap_or(1.0);
        config.voice_quality_adjustment = params
            .get("voice_quality_adjustment")
            .copied()
            .unwrap_or(0.0);
        config.pitch_contour_variation = params
            .get("pitch_contour_variation")
            .copied()
            .unwrap_or(1.0);
        config.rhythm_modification = params.get("rhythm_modification").copied().unwrap_or(1.0);
        config.stress_pattern_enhancement = params
            .get("stress_pattern_enhancement")
            .copied()
            .unwrap_or(1.0);

        config
    }
}

/// Mapping between emotion and speaker characteristics
#[derive(Debug, Clone, PartialEq)]
pub struct EmotionSpeakerMapping {
    /// Speaker ID to use for this emotion
    pub speaker_id: Option<String>,
    /// Speaker-specific parameters
    pub speaker_params: HashMap<String, f32>,
    /// Voice quality adjustments
    pub voice_quality: VoiceQualityMapping,
}

#[cfg(feature = "acoustic-integration")]
impl EmotionSpeakerMapping {
    /// Create new speaker mapping
    pub fn new() -> Self {
        Self {
            speaker_id: None,
            speaker_params: HashMap::new(),
            voice_quality: VoiceQualityMapping::default(),
        }
    }

    /// Set speaker ID
    pub fn with_speaker_id(mut self, speaker_id: String) -> Self {
        self.speaker_id = Some(speaker_id);
        self
    }

    /// Add speaker parameter
    pub fn with_param(mut self, name: String, value: f32) -> Self {
        self.speaker_params.insert(name, value);
        self
    }

    /// Set voice quality mapping
    pub fn with_voice_quality(mut self, voice_quality: VoiceQualityMapping) -> Self {
        self.voice_quality = voice_quality;
        self
    }
}

#[cfg(feature = "acoustic-integration")]
impl Default for EmotionSpeakerMapping {
    fn default() -> Self {
        Self::new()
    }
}

/// Vocoder emotion configuration (placeholder)
#[derive(Debug, Clone, PartialEq)]
pub struct VocoderEmotionConfig {
    /// Pitch shifting factor
    pub pitch_shift: f32,
    /// Formant shifting factor
    pub formant_shift: f32,
    /// Spectral tilt adjustment
    pub spectral_tilt: f32,
    /// Roughness processing factor
    pub roughness_factor: f32,
    /// Breathiness processing factor
    pub breathiness_factor: f32,
    /// Energy scaling factor
    pub energy_scale: f32,
}

/// Voice quality parameter mapping
#[cfg(feature = "acoustic-integration")]
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceQualityMapping {
    /// Breathiness adjustment
    pub breathiness: f32,
    /// Roughness adjustment
    pub roughness: f32,
    /// Tension adjustment
    pub tension: f32,
    /// Brightness adjustment
    pub brightness: f32,
    /// Custom quality parameters
    pub custom_params: HashMap<String, f32>,
}

#[cfg(feature = "acoustic-integration")]
impl VoiceQualityMapping {
    /// Create neutral voice quality mapping
    pub fn neutral() -> Self {
        Self {
            breathiness: 0.0,
            roughness: 0.0,
            tension: 0.0,
            brightness: 0.0,
            custom_params: HashMap::new(),
        }
    }

    /// Set breathiness
    pub fn with_breathiness(mut self, breathiness: f32) -> Self {
        self.breathiness = breathiness;
        self
    }

    /// Set roughness
    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.roughness = roughness;
        self
    }

    /// Set tension
    pub fn with_tension(mut self, tension: f32) -> Self {
        self.tension = tension;
        self
    }

    /// Set brightness
    pub fn with_brightness(mut self, brightness: f32) -> Self {
        self.brightness = brightness;
        self
    }

    /// Add custom parameter
    pub fn with_custom_param(mut self, name: String, value: f32) -> Self {
        self.custom_params.insert(name, value);
        self
    }
}

#[cfg(feature = "acoustic-integration")]
impl Default for VoiceQualityMapping {
    fn default() -> Self {
        Self::neutral()
    }
}

// Stub implementations when acoustic integration is disabled
#[cfg(not(feature = "acoustic-integration"))]
#[derive(Debug, Clone)]
pub struct EmotionSpeakerMapping;

#[cfg(not(feature = "acoustic-integration"))]
#[derive(Debug, Clone)]
pub struct VoiceQualityMapping;
