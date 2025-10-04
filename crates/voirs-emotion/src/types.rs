//! Core emotion types and data structures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents different emotion types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Emotion {
    /// Neutral emotion (baseline)
    Neutral,
    /// Happy emotion
    Happy,
    /// Sad emotion
    Sad,
    /// Angry emotion
    Angry,
    /// Fear emotion
    Fear,
    /// Surprise emotion
    Surprise,
    /// Disgust emotion
    Disgust,
    /// Calm emotion
    Calm,
    /// Excited emotion
    Excited,
    /// Tender emotion
    Tender,
    /// Confident emotion
    Confident,
    /// Melancholic emotion
    Melancholic,
    /// Custom emotion with name
    Custom(String),
}

impl Emotion {
    /// Returns the string representation of the emotion
    pub fn as_str(&self) -> &str {
        match self {
            Emotion::Neutral => "neutral",
            Emotion::Happy => "happy",
            Emotion::Sad => "sad",
            Emotion::Angry => "angry",
            Emotion::Fear => "fear",
            Emotion::Surprise => "surprise",
            Emotion::Disgust => "disgust",
            Emotion::Calm => "calm",
            Emotion::Excited => "excited",
            Emotion::Tender => "tender",
            Emotion::Confident => "confident",
            Emotion::Melancholic => "melancholic",
            Emotion::Custom(name) => name,
        }
    }

    /// Parse emotion from string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "neutral" => Emotion::Neutral,
            "happy" => Emotion::Happy,
            "sad" => Emotion::Sad,
            "angry" => Emotion::Angry,
            "fear" => Emotion::Fear,
            "surprise" => Emotion::Surprise,
            "disgust" => Emotion::Disgust,
            "calm" => Emotion::Calm,
            "excited" => Emotion::Excited,
            "tender" => Emotion::Tender,
            "confident" => Emotion::Confident,
            "melancholic" => Emotion::Melancholic,
            _ => Emotion::Custom(s.to_string()),
        }
    }
}

/// Emotion intensity levels
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct EmotionIntensity(f32);

impl EmotionIntensity {
    /// Create new emotion intensity (clamped to 0.0-1.0)
    pub fn new(intensity: f32) -> Self {
        Self(intensity.clamp(0.0, 1.0))
    }

    /// Get the intensity value
    pub fn value(&self) -> f32 {
        self.0
    }

    /// Very low intensity (0.1)
    pub const VERY_LOW: Self = Self(0.1);
    /// Low intensity (0.3)
    pub const LOW: Self = Self(0.3);
    /// Medium intensity (0.5)
    pub const MEDIUM: Self = Self(0.5);
    /// High intensity (0.7)
    pub const HIGH: Self = Self(0.7);
    /// Very high intensity (0.9)
    pub const VERY_HIGH: Self = Self(0.9);
    /// Maximum intensity (1.0)
    pub const MAX: Self = Self(1.0);
}

impl Default for EmotionIntensity {
    fn default() -> Self {
        Self::MEDIUM
    }
}

impl From<f32> for EmotionIntensity {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

/// Emotion dimensions using the circumplex model
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EmotionDimensions {
    /// Valence: pleasure/displeasure (-1.0 to 1.0)
    pub valence: f32,
    /// Arousal: activation/deactivation (-1.0 to 1.0)
    pub arousal: f32,
    /// Dominance: control/submissiveness (-1.0 to 1.0)
    pub dominance: f32,
}

impl EmotionDimensions {
    /// Create new emotion dimensions
    pub fn new(valence: f32, arousal: f32, dominance: f32) -> Self {
        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(-1.0, 1.0),
            dominance: dominance.clamp(-1.0, 1.0),
        }
    }

    /// Neutral dimensions
    pub fn neutral() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Calculate distance to another emotion
    pub fn distance(&self, other: &Self) -> f32 {
        let dv = self.valence - other.valence;
        let da = self.arousal - other.arousal;
        let dd = self.dominance - other.dominance;
        (dv * dv + da * da + dd * dd).sqrt()
    }
}

impl Default for EmotionDimensions {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Multi-dimensional emotion vector for representing complex emotional states.
///
/// An `EmotionVector` can contain multiple emotions simultaneously, allowing for
/// nuanced expressions like "bittersweet" (happy + sad) or "triumphant" (happy + confident).
/// The vector automatically computes dimensional coordinates in the VAD
/// (Valence-Arousal-Dominance) space based on the constituent emotions.
///
/// # Design Philosophy
///
/// Real human emotions are rarely pure - we often feel mixtures of different emotions
/// with varying intensities. This vector representation allows for:
///
/// - **Complex emotions**: Combinations like anxiety (fear + sadness)
/// - **Cultural variations**: Different intensity patterns for the same base emotion
/// - **Temporal blending**: Smooth transitions between different emotional states
/// - **Personal expression**: Individual differences in emotional expression patterns
///
/// # Coordinate System
///
/// The system uses the Circumplex Model of emotions with three dimensions:
/// - **Valence**: Pleasant/unpleasant (-1.0 to 1.0)
/// - **Arousal**: Activated/deactivated (-1.0 to 1.0)  
/// - **Dominance**: Controlled/submissive (-1.0 to 1.0)
///
/// # Examples
///
/// ```rust
/// # use voirs_emotion::*;
/// let mut vector = EmotionVector::new();
///
/// // Simple emotion
/// vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);
///
/// // Complex emotion (bittersweet)
/// vector.add_emotion(Emotion::Sad, EmotionIntensity::MEDIUM);
///
/// // The dimensions are automatically calculated
/// println!("Valence: {}", vector.dimensions.valence);
/// println!("Arousal: {}", vector.dimensions.arousal);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionVector {
    /// Individual emotion components with their intensities.
    ///
    /// Multiple emotions can be active simultaneously, allowing complex
    /// emotional expressions. Intensities should typically sum to ≤ 1.0
    /// for realistic emotional states.
    pub emotions: HashMap<Emotion, EmotionIntensity>,

    /// Computed dimensional coordinates in VAD space.
    ///
    /// These are automatically updated when emotions are added/removed
    /// using weighted averages of individual emotion coordinates.
    pub dimensions: EmotionDimensions,
}

impl EmotionVector {
    /// Create new emotion vector
    pub fn new() -> Self {
        Self {
            emotions: HashMap::new(),
            dimensions: EmotionDimensions::default(),
        }
    }

    /// Add emotion component
    pub fn add_emotion(&mut self, emotion: Emotion, intensity: EmotionIntensity) {
        self.emotions.insert(emotion, intensity);
        self.update_dimensions();
    }

    /// Remove emotion component
    pub fn remove_emotion(&mut self, emotion: &Emotion) {
        self.emotions.remove(emotion);
        self.update_dimensions();
    }

    /// Get dominant emotion
    pub fn dominant_emotion(&self) -> Option<(Emotion, EmotionIntensity)> {
        self.emotions
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(e, i)| (e.clone(), *i))
    }

    /// Normalize emotion intensities to sum to 1.0
    pub fn normalize(&mut self) {
        let total: f32 = self.emotions.values().map(|i| i.value()).sum();
        if total > 0.0 {
            for intensity in self.emotions.values_mut() {
                *intensity = EmotionIntensity::new(intensity.value() / total);
            }
        }
    }

    /// Update VAD dimensions based on constituent emotion components.
    ///
    /// This method computes the overall emotional dimensions using a weighted average
    /// of individual emotion coordinates. Each emotion has predefined coordinates in
    /// the Valence-Arousal-Dominance space based on psychological research.
    ///
    /// # Algorithm
    ///
    /// 1. For each active emotion, multiply its VAD coordinates by its intensity
    /// 2. Sum all weighted coordinates and divide by total weight
    /// 3. Clamp results to valid range [-1.0, 1.0]
    ///
    /// # Emotion Mappings
    ///
    /// The coordinate mappings are based on the Circumplex Model:
    /// - **Happy**: (0.8, 0.5, 0.3) - High valence, moderate arousal/dominance
    /// - **Sad**: (-0.7, -0.3, -0.5) - Low valence, low arousal, submissive
    /// - **Angry**: (-0.5, 0.8, 0.7) - Negative valence, high arousal/dominance
    /// - **Fear**: (-0.6, 0.7, -0.8) - Negative valence, high arousal, very submissive
    ///
    /// Custom emotions default to (0.0, 0.0, 0.0) unless specifically configured.
    fn update_dimensions(&mut self) {
        // Compute weighted average of emotion coordinates
        let mut valence = 0.0;
        let mut arousal = 0.0;
        let mut dominance = 0.0;
        let mut total_weight = 0.0;

        for (emotion, intensity) in &self.emotions {
            let weight = intensity.value();
            let (v, a, d) = match emotion {
                Emotion::Happy => (0.8, 0.5, 0.3),
                Emotion::Sad => (-0.7, -0.3, -0.5),
                Emotion::Angry => (-0.5, 0.8, 0.7),
                Emotion::Fear => (-0.6, 0.7, -0.8),
                Emotion::Surprise => (0.2, 0.8, 0.0),
                Emotion::Disgust => (-0.7, 0.3, 0.2),
                Emotion::Calm => (0.3, -0.7, 0.2),
                Emotion::Excited => (0.7, 0.9, 0.5),
                Emotion::Tender => (0.6, -0.2, -0.1),
                Emotion::Confident => (0.5, 0.3, 0.8),
                Emotion::Melancholic => (-0.4, -0.5, -0.3),
                Emotion::Neutral | Emotion::Custom(_) => (0.0, 0.0, 0.0),
            };

            valence += v * weight;
            arousal += a * weight;
            dominance += d * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            self.dimensions = EmotionDimensions::new(
                valence / total_weight,
                arousal / total_weight,
                dominance / total_weight,
            );
        }
    }
}

impl Default for EmotionVector {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete emotion parameters for synthesis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionParameters {
    /// Base emotion vector
    pub emotion_vector: EmotionVector,
    /// Temporal dynamics
    pub duration_ms: Option<u64>,
    /// Transition parameters
    pub fade_in_ms: Option<u64>,
    /// Fade-out duration in milliseconds for emotion transition
    pub fade_out_ms: Option<u64>,
    /// Prosody adjustments
    pub pitch_shift: f32,
    /// Tempo scaling factor (1.0 = normal, <1.0 = slower, >1.0 = faster)
    pub tempo_scale: f32,
    /// Energy level scaling factor (1.0 = normal, <1.0 = quieter, >1.0 = louder)
    pub energy_scale: f32,
    /// Voice quality adjustments
    pub breathiness: f32,
    /// Voice roughness level (0.0 = smooth, 1.0 = very rough)
    pub roughness: f32,
    /// Custom parameters
    pub custom_params: HashMap<String, f32>,
}

impl EmotionParameters {
    /// Create new emotion parameters
    pub fn new(emotion_vector: EmotionVector) -> Self {
        Self {
            emotion_vector,
            duration_ms: None,
            fade_in_ms: None,
            fade_out_ms: None,
            pitch_shift: 1.0,
            tempo_scale: 1.0,
            energy_scale: 1.0,
            breathiness: 0.0,
            roughness: 0.0,
            custom_params: HashMap::new(),
        }
    }

    /// Create neutral parameters
    pub fn neutral() -> Self {
        Self::new(EmotionVector::default())
    }

    /// Set prosody adjustments
    pub fn with_prosody(mut self, pitch_shift: f32, tempo_scale: f32, energy_scale: f32) -> Self {
        self.pitch_shift = pitch_shift;
        self.tempo_scale = tempo_scale;
        self.energy_scale = energy_scale;
        self
    }

    /// Set temporal parameters
    pub fn with_timing(mut self, duration_ms: u64, fade_in_ms: u64, fade_out_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self.fade_in_ms = Some(fade_in_ms);
        self.fade_out_ms = Some(fade_out_ms);
        self
    }

    /// Add custom parameter
    pub fn with_custom_param(mut self, name: String, value: f32) -> Self {
        self.custom_params.insert(name, value);
        self
    }
}

impl Default for EmotionParameters {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Emotion state for tracking changes over time
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionState {
    /// Current emotion parameters
    pub current: EmotionParameters,
    /// Target emotion parameters (for transitions)
    pub target: Option<EmotionParameters>,
    /// Transition progress (0.0 to 1.0)
    pub transition_progress: f32,
    /// Timestamp of last update
    pub timestamp: std::time::SystemTime,
}

impl EmotionState {
    /// Create new emotion state
    pub fn new(emotion_params: EmotionParameters) -> Self {
        Self {
            current: emotion_params,
            target: None,
            transition_progress: 1.0,
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Start transition to new emotion
    pub fn transition_to(&mut self, target: EmotionParameters) {
        self.target = Some(target);
        self.transition_progress = 0.0;
        self.timestamp = std::time::SystemTime::now();
    }

    /// Update transition progress
    pub fn update_transition(&mut self, delta_progress: f32) {
        self.transition_progress = (self.transition_progress + delta_progress).clamp(0.0, 1.0);

        if self.transition_progress >= 1.0 {
            if let Some(target) = self.target.take() {
                self.current = target;
            }
        }

        self.timestamp = std::time::SystemTime::now();
    }

    /// Check if currently transitioning
    pub fn is_transitioning(&self) -> bool {
        self.target.is_some() && self.transition_progress < 1.0
    }

    /// Get interpolated parameters during transition
    pub fn get_interpolated(&self) -> EmotionParameters {
        if let Some(target) = &self.target {
            if self.transition_progress < 1.0 {
                // Perform linear interpolation between current and target
                let progress = self.transition_progress;

                // Interpolate emotion vector
                let mut interpolated_emotions = std::collections::HashMap::new();

                // Collect all unique emotions from both current and target
                let mut all_emotions = std::collections::HashSet::new();
                for emotion in self.current.emotion_vector.emotions.keys() {
                    all_emotions.insert(emotion.clone());
                }
                for emotion in target.emotion_vector.emotions.keys() {
                    all_emotions.insert(emotion.clone());
                }

                // Interpolate each emotion's intensity
                for emotion in all_emotions {
                    let current_intensity = self
                        .current
                        .emotion_vector
                        .emotions
                        .get(&emotion)
                        .map(|i| i.value())
                        .unwrap_or(0.0);
                    let target_intensity = target
                        .emotion_vector
                        .emotions
                        .get(&emotion)
                        .map(|i| i.value())
                        .unwrap_or(0.0);

                    let interpolated_intensity =
                        current_intensity + (target_intensity - current_intensity) * progress;

                    if interpolated_intensity > 0.01 {
                        interpolated_emotions
                            .insert(emotion, EmotionIntensity::new(interpolated_intensity));
                    }
                }

                // Create interpolated emotion vector
                let mut emotion_vector = EmotionVector::new();
                emotion_vector.emotions = interpolated_emotions;

                // Interpolate dimensions
                let current_dims = &self.current.emotion_vector.dimensions;
                let target_dims = &target.emotion_vector.dimensions;

                emotion_vector.dimensions = EmotionDimensions::new(
                    current_dims.valence + (target_dims.valence - current_dims.valence) * progress,
                    current_dims.arousal + (target_dims.arousal - current_dims.arousal) * progress,
                    current_dims.dominance
                        + (target_dims.dominance - current_dims.dominance) * progress,
                );

                // Interpolate other parameters
                let interpolated_params = EmotionParameters {
                    emotion_vector,
                    duration_ms: target.duration_ms.or(self.current.duration_ms),
                    fade_in_ms: target.fade_in_ms.or(self.current.fade_in_ms),
                    fade_out_ms: target.fade_out_ms.or(self.current.fade_out_ms),
                    pitch_shift: self.current.pitch_shift
                        + (target.pitch_shift - self.current.pitch_shift) * progress,
                    tempo_scale: self.current.tempo_scale
                        + (target.tempo_scale - self.current.tempo_scale) * progress,
                    energy_scale: self.current.energy_scale
                        + (target.energy_scale - self.current.energy_scale) * progress,
                    breathiness: self.current.breathiness
                        + (target.breathiness - self.current.breathiness) * progress,
                    roughness: self.current.roughness
                        + (target.roughness - self.current.roughness) * progress,
                    custom_params: {
                        let mut interpolated_custom = std::collections::HashMap::new();

                        // Collect all custom parameter names
                        let mut all_params = std::collections::HashSet::new();
                        for param in self.current.custom_params.keys() {
                            all_params.insert(param);
                        }
                        for param in target.custom_params.keys() {
                            all_params.insert(param);
                        }

                        // Interpolate each custom parameter
                        for param in all_params {
                            let current_value = self
                                .current
                                .custom_params
                                .get(param)
                                .copied()
                                .unwrap_or(0.0);
                            let target_value =
                                target.custom_params.get(param).copied().unwrap_or(0.0);
                            let interpolated_value =
                                current_value + (target_value - current_value) * progress;
                            interpolated_custom.insert(param.clone(), interpolated_value);
                        }

                        interpolated_custom
                    },
                };

                interpolated_params
            } else {
                self.current.clone()
            }
        } else {
            self.current.clone()
        }
    }
}

impl Default for EmotionState {
    fn default() -> Self {
        Self::new(EmotionParameters::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_intensity() {
        let intensity = EmotionIntensity::new(1.5); // Should be clamped to 1.0
        assert_eq!(intensity.value(), 1.0);

        let intensity = EmotionIntensity::new(-0.5); // Should be clamped to 0.0
        assert_eq!(intensity.value(), 0.0);

        let intensity = EmotionIntensity::new(0.7);
        assert_eq!(intensity.value(), 0.7);
    }

    #[test]
    fn test_emotion_dimensions() {
        let dims = EmotionDimensions::new(0.5, -0.8, 1.2);
        assert_eq!(dims.valence, 0.5);
        assert_eq!(dims.arousal, -0.8);
        assert_eq!(dims.dominance, 1.0); // Should be clamped

        let neutral = EmotionDimensions::neutral();
        assert_eq!(neutral.valence, 0.0);
        assert_eq!(neutral.arousal, 0.0);
        assert_eq!(neutral.dominance, 0.0);

        let distance = dims.distance(&neutral);
        assert!(distance > 0.0);
    }

    #[test]
    fn test_emotion_vector() {
        let mut vector = EmotionVector::new();
        vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);
        vector.add_emotion(Emotion::Excited, EmotionIntensity::MEDIUM);

        assert_eq!(vector.emotions.len(), 2);
        assert_eq!(
            vector.dominant_emotion(),
            Some((Emotion::Happy, EmotionIntensity::HIGH))
        );

        vector.normalize();
        let total: f32 = vector.emotions.values().map(|i| i.value()).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_emotion_state_interpolation() {
        let neutral_params = EmotionParameters::neutral();
        let mut state = EmotionState::new(neutral_params);

        // Create target emotion with different parameters
        let mut happy_vector = EmotionVector::new();
        happy_vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);
        let happy_params = EmotionParameters::new(happy_vector).with_prosody(1.5, 1.2, 1.3);

        // Start transition
        state.transition_to(happy_params);
        state.update_transition(0.5); // 50% progress

        // Get interpolated parameters
        let interpolated = state.get_interpolated();

        // Should be halfway between neutral and target
        assert!(interpolated.pitch_shift > 1.0 && interpolated.pitch_shift < 1.5);
        assert!(interpolated.tempo_scale > 1.0 && interpolated.tempo_scale < 1.2);
        assert!(interpolated.energy_scale > 1.0 && interpolated.energy_scale < 1.3);

        // Should have some happy emotion but less than full intensity
        let happy_intensity = interpolated
            .emotion_vector
            .emotions
            .get(&Emotion::Happy)
            .map(|i| i.value())
            .unwrap_or(0.0);
        assert!(happy_intensity > 0.0 && happy_intensity < EmotionIntensity::HIGH.value());
    }

    #[test]
    fn test_emotion_from_string() {
        assert_eq!(Emotion::from_str("happy"), Emotion::Happy);
        assert_eq!(Emotion::from_str("ANGRY"), Emotion::Angry);
        assert_eq!(
            Emotion::from_str("unknown"),
            Emotion::Custom("unknown".to_string())
        );
    }

    #[test]
    fn test_emotion_vector_update_dimensions() {
        let mut vector = EmotionVector::new();

        // Add happy emotion
        vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);

        // Dimensions should reflect happiness (positive valence, some arousal)
        assert!(vector.dimensions.valence > 0.0);
        assert!(vector.dimensions.arousal > 0.0);

        // Add sad emotion
        vector.add_emotion(Emotion::Sad, EmotionIntensity::MEDIUM);

        // Valence should decrease due to sadness
        assert!(vector.dimensions.valence < 0.8); // Less positive than pure happiness
    }
}
