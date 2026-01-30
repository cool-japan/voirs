//! Feature extraction types for emotion analysis
//!
//! This module provides types for representing extracted acoustic features
//! and prosody patterns from audio signals.

use crate::types::EmotionVector;

/// Baseline speaker characteristics
#[derive(Debug, Clone)]
pub struct BaselineCharacteristics {
    /// Average fundamental frequency (pitch) in Hz
    pub average_f0: f32,
    /// F0 range (max - min) in Hz
    pub f0_range: f32,
    /// Speaking rate in syllables per second
    pub speaking_rate: f32,
    /// Average energy level (RMS)
    pub average_energy: f32,
    /// Energy dynamic range
    pub energy_range: f32,
    /// Typical pause duration in seconds
    pub typical_pause_duration: f32,
    /// Articulation rate (speech without pauses)
    pub articulation_rate: f32,
    /// Voice quality baseline
    pub voice_quality_baseline: VoiceQualityBaseline,
}

impl BaselineCharacteristics {
    /// Create default baseline characteristics
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self {
            average_f0: 150.0,           // Typical adult average
            f0_range: 100.0,             // Moderate range
            speaking_rate: 4.5,          // ~4.5 syllables/second
            average_energy: 0.3,         // Moderate energy
            energy_range: 0.5,           // Moderate dynamic range
            typical_pause_duration: 0.3, // 300ms pauses
            articulation_rate: 5.5,      // Faster without pauses
            voice_quality_baseline: VoiceQualityBaseline::default(),
        }
    }

    /// Create from audio analysis
    pub fn from_prosody_and_voice_quality(
        prosody: &ProsodyPatterns,
        voice_quality: &VoiceQualityProfile,
    ) -> Self {
        let average_f0 = prosody.average_pitch();
        let f0_range = if prosody.pitch_contour.len() > 1 {
            let max_pitch = prosody
                .pitch_contour
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let min_pitch = prosody
                .pitch_contour
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            max_pitch - min_pitch
        } else {
            0.0
        };

        let average_energy = prosody.average_energy();
        let energy_range = if prosody.energy_contour.len() > 1 {
            let max_energy = prosody
                .energy_contour
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let min_energy = prosody
                .energy_contour
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            max_energy - min_energy
        } else {
            0.0
        };

        // Estimate speaking rate from rhythm complexity
        let rhythm_complexity = prosody.rhythm_complexity();
        let speaking_rate = 3.0 + rhythm_complexity * 3.0; // 3-6 syllables/second
        let articulation_rate = speaking_rate * 1.2; // Typically 20% faster

        Self {
            average_f0,
            f0_range,
            speaking_rate,
            average_energy,
            energy_range,
            typical_pause_duration: 0.3, // Default
            articulation_rate,
            voice_quality_baseline: VoiceQualityBaseline::from_profile(voice_quality),
        }
    }

    /// Check if characteristics are within normal ranges
    pub fn is_valid(&self) -> bool {
        self.average_f0 > 50.0
            && self.average_f0 < 500.0
            && self.f0_range >= 0.0
            && self.speaking_rate > 0.0
            && self.speaking_rate < 10.0
            && self.average_energy > 0.0
            && self.average_energy <= 1.0
    }
}

/// Baseline voice quality characteristics
#[derive(Debug, Clone)]
pub struct VoiceQualityBaseline {
    /// Typical spectral balance
    pub spectral_balance: f32,
    /// Typical harmonic richness
    pub harmonic_richness: f32,
    /// Baseline breathiness level
    pub baseline_breathiness: f32,
    /// Baseline roughness level
    pub baseline_roughness: f32,
}

impl VoiceQualityBaseline {
    /// Create default voice quality baseline
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self {
            spectral_balance: 0.5,
            harmonic_richness: 0.7,
            baseline_breathiness: 0.1,
            baseline_roughness: 0.1,
        }
    }

    /// Create from voice quality profile
    pub fn from_profile(profile: &VoiceQualityProfile) -> Self {
        Self {
            spectral_balance: (profile.spectral_tilt + 1.0) / 2.0, // Normalize to 0-1
            harmonic_richness: (profile.harmonic_noise_ratio / 20.0).clamp(0.0, 1.0),
            baseline_breathiness: profile.breathiness_measure,
            baseline_roughness: profile.roughness_measure,
        }
    }
}

/// Speaker emotion features for voice cloning
#[derive(Debug, Clone)]
pub struct SpeakerEmotionFeatures {
    /// Speaker identifier
    pub speaker_id: String,
    /// Baseline speaker characteristics
    pub baseline_characteristics: BaselineCharacteristics,
    /// Emotion features in context of speaker
    pub emotion_features: EmotionVector,
    /// Prosody patterns extracted from audio
    pub prosody_patterns: ProsodyPatterns,
    /// Voice quality profile
    pub voice_quality_profile: VoiceQualityProfile,
}

/// Prosody patterns extracted from audio
#[derive(Debug, Clone)]
pub struct ProsodyPatterns {
    /// Pitch contour over time
    pub pitch_contour: Vec<f32>,
    /// Energy contour over time
    pub energy_contour: Vec<f32>,
    /// Rhythm pattern (binary peaks)
    pub rhythm_pattern: Vec<f32>,
    /// Tempo variations over time
    pub tempo_variations: Vec<f32>,
}

impl ProsodyPatterns {
    /// Create default prosody patterns
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self {
            pitch_contour: vec![220.0; 10],
            energy_contour: vec![0.1; 10],
            rhythm_pattern: vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            tempo_variations: vec![1.0; 10],
        }
    }

    /// Get average pitch
    pub fn average_pitch(&self) -> f32 {
        if self.pitch_contour.is_empty() {
            return 220.0;
        }
        self.pitch_contour.iter().sum::<f32>() / self.pitch_contour.len() as f32
    }

    /// Get average energy
    pub fn average_energy(&self) -> f32 {
        if self.energy_contour.is_empty() {
            return 0.1;
        }
        self.energy_contour.iter().sum::<f32>() / self.energy_contour.len() as f32
    }

    /// Get rhythm complexity (variance in rhythm pattern)
    pub fn rhythm_complexity(&self) -> f32 {
        if self.rhythm_pattern.len() < 2 {
            return 0.0;
        }

        let mean = self.rhythm_pattern.iter().sum::<f32>() / self.rhythm_pattern.len() as f32;
        let variance = self
            .rhythm_pattern
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / self.rhythm_pattern.len() as f32;

        variance.sqrt()
    }
}

/// Voice quality profile extracted from audio
#[derive(Debug, Clone)]
pub struct VoiceQualityProfile {
    /// Spectral tilt measure
    pub spectral_tilt: f32,
    /// Harmonic-to-noise ratio
    pub harmonic_noise_ratio: f32,
    /// Formant frequencies (F1, F2, F3, ...)
    pub formant_frequencies: Vec<f32>,
    /// Breathiness measure (0.0-1.0)
    pub breathiness_measure: f32,
    /// Roughness measure (0.0-1.0)
    pub roughness_measure: f32,
}

impl VoiceQualityProfile {
    /// Create default voice quality profile
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self {
            spectral_tilt: 0.0,
            harmonic_noise_ratio: 15.0, // Typical value
            formant_frequencies: vec![700.0, 1220.0, 2600.0], // Typical F1, F2, F3
            breathiness_measure: 0.1,
            roughness_measure: 0.1,
        }
    }

    /// Get voice brightness (based on spectral tilt and formants)
    pub fn brightness(&self) -> f32 {
        let high_formant_energy = self
            .formant_frequencies
            .iter()
            .skip(1) // Skip F1
            .sum::<f32>()
            / (self.formant_frequencies.len() - 1).max(1) as f32;

        (high_formant_energy / 2000.0 - self.spectral_tilt).clamp(0.0, 1.0)
    }

    /// Get voice clarity (based on HNR and roughness)
    pub fn clarity(&self) -> f32 {
        (self.harmonic_noise_ratio / 20.0 * (1.0 - self.roughness_measure)).clamp(0.0, 1.0)
    }

    /// Get voice naturalness (inverse of breathiness and roughness)
    pub fn naturalness(&self) -> f32 {
        (1.0 - (self.breathiness_measure + self.roughness_measure) / 2.0).clamp(0.0, 1.0)
    }
}
