//! Real-Time Emotion Transfer System
//!
//! This module provides advanced emotion transfer capabilities for singing synthesis,
//! enabling dynamic emotion manipulation during real-time performance.

use crate::types::{Articulation, Dynamics, Expression};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Real-time emotion transfer engine
///
/// Enables dynamic emotion transfer from reference audio to synthesized singing
#[derive(Debug)]
pub struct EmotionTransferEngine {
    /// Configuration
    config: EmotionTransferConfig,
    /// Emotion detector
    detector: EmotionDetector,
    /// Emotion interpolator
    interpolator: EmotionInterpolator,
    /// Emotion cache for performance
    emotion_cache: HashMap<String, EmotionVector>,
}

/// Emotion transfer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionTransferConfig {
    /// Enable real-time processing
    pub realtime: bool,
    /// Interpolation smoothness (0.0-1.0)
    pub smoothness: f32,
    /// Emotion intensity scaling (0.0-2.0)
    pub intensity_scale: f32,
    /// Transition duration in seconds
    pub transition_duration: f32,
    /// Enable emotion caching
    pub enable_caching: bool,
}

impl Default for EmotionTransferConfig {
    fn default() -> Self {
        Self {
            realtime: true,
            smoothness: 0.7,
            intensity_scale: 1.0,
            transition_duration: 0.5,
            enable_caching: true,
        }
    }
}

/// Multi-dimensional emotion vector
///
/// Represents emotion in a 3D space: valence (positive/negative),
/// arousal (calm/excited), and dominance (submissive/dominant)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EmotionVector {
    /// Valence: -1.0 (negative) to 1.0 (positive)
    pub valence: f32,
    /// Arousal: -1.0 (calm) to 1.0 (excited)
    pub arousal: f32,
    /// Dominance: -1.0 (submissive) to 1.0 (dominant)
    pub dominance: f32,
    /// Intensity: 0.0 (none) to 1.0 (maximum)
    pub intensity: f32,
}

impl EmotionVector {
    /// Create a new emotion vector
    pub fn new(valence: f32, arousal: f32, dominance: f32, intensity: f32) -> Self {
        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(-1.0, 1.0),
            dominance: dominance.clamp(-1.0, 1.0),
            intensity: intensity.clamp(0.0, 1.0),
        }
    }

    /// Create a neutral emotion
    pub fn neutral() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.0,
            dominance: 0.0,
            intensity: 0.0,
        }
    }

    /// Create happy emotion
    pub fn happy() -> Self {
        Self::new(0.8, 0.6, 0.5, 0.8)
    }

    /// Create sad emotion
    pub fn sad() -> Self {
        Self::new(-0.7, -0.5, -0.3, 0.7)
    }

    /// Create angry emotion
    pub fn angry() -> Self {
        Self::new(-0.6, 0.8, 0.7, 0.9)
    }

    /// Create fearful emotion
    pub fn fearful() -> Self {
        Self::new(-0.5, 0.7, -0.6, 0.8)
    }

    /// Interpolate between two emotions
    pub fn interpolate(&self, other: &EmotionVector, alpha: f32) -> EmotionVector {
        let alpha = alpha.clamp(0.0, 1.0);
        EmotionVector {
            valence: self.valence + (other.valence - self.valence) * alpha,
            arousal: self.arousal + (other.arousal - self.arousal) * alpha,
            dominance: self.dominance + (other.dominance - self.dominance) * alpha,
            intensity: self.intensity + (other.intensity - self.intensity) * alpha,
        }
    }

    /// Calculate Euclidean distance to another emotion
    pub fn distance(&self, other: &EmotionVector) -> f32 {
        let dv = self.valence - other.valence;
        let da = self.arousal - other.arousal;
        let dd = self.dominance - other.dominance;
        (dv * dv + da * da + dd * dd).sqrt()
    }
}

impl Default for EmotionVector {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Emotion detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionDetectionResult {
    /// Detected emotion vector
    pub emotion: EmotionVector,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Emotion label (e.g., "happy", "sad")
    pub label: String,
    /// Per-frame emotions for temporal analysis
    pub frame_emotions: Vec<EmotionVector>,
}

/// Emotion detector for audio analysis
#[derive(Debug)]
pub struct EmotionDetector {
    /// Feature extractor configuration
    feature_config: FeatureConfig,
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Frame size in samples
    pub frame_size: usize,
    /// Hop size in samples
    pub hop_size: usize,
    /// Enable spectral features
    pub enable_spectral: bool,
    /// Enable prosodic features
    pub enable_prosodic: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            frame_size: 2048,
            hop_size: 512,
            enable_spectral: true,
            enable_prosodic: true,
        }
    }
}

impl EmotionDetector {
    /// Create new emotion detector
    pub fn new(config: FeatureConfig) -> Self {
        Self {
            feature_config: config,
        }
    }

    /// Detect emotion from audio samples
    ///
    /// # Arguments
    /// * `audio` - Audio samples (mono, float)
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Emotion detection result with confidence
    pub fn detect(&self, audio: &[f32], sample_rate: u32) -> EmotionDetectionResult {
        // Extract spectral features
        let spectral_features = self.extract_spectral_features(audio);

        // Extract prosodic features
        let prosodic_features = self.extract_prosodic_features(audio, sample_rate);

        // Combine features and classify emotion
        let emotion = self.classify_emotion(&spectral_features, &prosodic_features);

        // Analyze per-frame emotions
        let frame_emotions = self.analyze_temporal_emotions(audio);

        EmotionDetectionResult {
            emotion,
            confidence: 0.85, // Simplified confidence
            label: self.emotion_to_label(&emotion),
            frame_emotions,
        }
    }

    /// Extract spectral features from audio
    fn extract_spectral_features(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified spectral feature extraction
        // In production, this would use FFT and compute MFCC, spectral centroid, etc.
        let energy = audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32;
        let mean = audio.iter().sum::<f32>() / audio.len() as f32;
        vec![energy, mean, energy.sqrt()]
    }

    /// Extract prosodic features from audio
    fn extract_prosodic_features(&self, audio: &[f32], _sample_rate: u32) -> Vec<f32> {
        // Simplified prosodic feature extraction
        // In production, this would extract F0, intensity contour, speech rate, etc.
        let max_amplitude = audio.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let variance = audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32;
        vec![max_amplitude, variance]
    }

    /// Classify emotion from features
    fn classify_emotion(&self, spectral: &[f32], prosodic: &[f32]) -> EmotionVector {
        // Simplified emotion classification
        // In production, this would use a trained neural network
        let energy = spectral[0];
        let amplitude = prosodic[0];

        let arousal = (energy * 2.0 - 1.0).clamp(-1.0, 1.0);
        let valence = (amplitude - 0.5).clamp(-1.0, 1.0);
        let dominance = 0.0;
        let intensity = energy.min(1.0);

        EmotionVector::new(valence, arousal, dominance, intensity)
    }

    /// Analyze temporal emotion patterns
    fn analyze_temporal_emotions(&self, audio: &[f32]) -> Vec<EmotionVector> {
        let num_frames = audio.len() / self.feature_config.hop_size;
        let mut frame_emotions = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            let start = i * self.feature_config.hop_size;
            let end = (start + self.feature_config.frame_size).min(audio.len());
            let frame = &audio[start..end];

            let spectral = self.extract_spectral_features(frame);
            let prosodic = self.extract_prosodic_features(frame, 44100);
            let emotion = self.classify_emotion(&spectral, &prosodic);

            frame_emotions.push(emotion);
        }

        frame_emotions
    }

    /// Convert emotion vector to label
    fn emotion_to_label(&self, emotion: &EmotionVector) -> String {
        if emotion.valence > 0.5 && emotion.arousal > 0.3 {
            "happy".to_string()
        } else if emotion.valence < -0.5 && emotion.arousal < 0.0 {
            "sad".to_string()
        } else if emotion.valence < -0.3 && emotion.arousal > 0.5 {
            "angry".to_string()
        } else if emotion.arousal > 0.5 && emotion.dominance < -0.3 {
            "fearful".to_string()
        } else {
            "neutral".to_string()
        }
    }
}

impl Default for EmotionDetector {
    fn default() -> Self {
        Self::new(FeatureConfig::default())
    }
}

/// Emotion interpolator for smooth transitions
#[derive(Debug)]
pub struct EmotionInterpolator {
    /// Smoothing window size
    window_size: usize,
    /// Previous emotions for smoothing
    emotion_history: Vec<EmotionVector>,
}

impl EmotionInterpolator {
    /// Create new emotion interpolator
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            emotion_history: Vec::new(),
        }
    }

    /// Smooth emotion transition
    ///
    /// # Arguments
    /// * `current` - Current emotion
    /// * `target` - Target emotion
    /// * `transition_progress` - Progress (0.0-1.0)
    ///
    /// # Returns
    /// Smoothed emotion vector
    pub fn smooth_transition(
        &mut self,
        current: &EmotionVector,
        target: &EmotionVector,
        transition_progress: f32,
    ) -> EmotionVector {
        // Apply smoothing curve (ease-in-out)
        let smoothed_progress = self.ease_in_out(transition_progress);

        // Interpolate
        let interpolated = current.interpolate(target, smoothed_progress);

        // Add to history
        self.emotion_history.push(interpolated);
        if self.emotion_history.len() > self.window_size {
            self.emotion_history.remove(0);
        }

        // Return moving average
        self.moving_average()
    }

    /// Apply ease-in-out curve
    fn ease_in_out(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        if t < 0.5 {
            2.0 * t * t
        } else {
            1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
        }
    }

    /// Calculate moving average of emotion history
    fn moving_average(&self) -> EmotionVector {
        if self.emotion_history.is_empty() {
            return EmotionVector::neutral();
        }

        let len = self.emotion_history.len() as f32;
        let avg_valence = self.emotion_history.iter().map(|e| e.valence).sum::<f32>() / len;
        let avg_arousal = self.emotion_history.iter().map(|e| e.arousal).sum::<f32>() / len;
        let avg_dominance = self
            .emotion_history
            .iter()
            .map(|e| e.dominance)
            .sum::<f32>()
            / len;
        let avg_intensity = self
            .emotion_history
            .iter()
            .map(|e| e.intensity)
            .sum::<f32>()
            / len;

        EmotionVector::new(avg_valence, avg_arousal, avg_dominance, avg_intensity)
    }

    /// Clear emotion history
    pub fn reset(&mut self) {
        self.emotion_history.clear();
    }
}

impl Default for EmotionInterpolator {
    fn default() -> Self {
        Self::new(5)
    }
}

impl EmotionTransferEngine {
    /// Create new emotion transfer engine
    pub fn new(config: EmotionTransferConfig) -> Self {
        Self {
            config,
            detector: EmotionDetector::default(),
            interpolator: EmotionInterpolator::default(),
            emotion_cache: HashMap::new(),
        }
    }

    /// Transfer emotion from reference audio to synthesis parameters
    ///
    /// # Arguments
    /// * `reference_audio` - Reference audio samples
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Detected emotion vector
    pub fn detect_reference_emotion(
        &mut self,
        reference_audio: &[f32],
        sample_rate: u32,
    ) -> crate::Result<EmotionVector> {
        // Check cache if enabled
        let cache_key = format!("{:?}", reference_audio.len());
        if self.config.enable_caching {
            if let Some(cached_emotion) = self.emotion_cache.get(&cache_key) {
                return Ok(*cached_emotion);
            }
        }

        // Detect emotion
        let detection = self.detector.detect(reference_audio, sample_rate);
        let mut emotion = detection.emotion;

        // Apply intensity scaling
        emotion.intensity *= self.config.intensity_scale;
        emotion.intensity = emotion.intensity.clamp(0.0, 1.0);

        // Cache result
        if self.config.enable_caching {
            self.emotion_cache.insert(cache_key, emotion);
        }

        Ok(emotion)
    }

    /// Apply emotion to synthesis parameters
    ///
    /// # Arguments
    /// * `emotion` - Emotion vector to apply
    ///
    /// # Returns
    /// Modified synthesis parameters
    pub fn apply_emotion(&self, emotion: &EmotionVector) -> EmotionSynthesisParams {
        // Convert emotion to synthesis parameters
        EmotionSynthesisParams {
            tempo_scale: 1.0 + emotion.arousal * 0.2,
            pitch_shift: emotion.valence * 2.0,
            dynamics: self.emotion_to_dynamics(emotion),
            articulation: self.emotion_to_articulation(emotion),
            vibrato_rate: (1.0 + emotion.arousal * 0.3).max(0.5),
            vibrato_depth: emotion.intensity * 0.05,
        }
    }

    /// Convert emotion to dynamics
    fn emotion_to_dynamics(&self, emotion: &EmotionVector) -> Dynamics {
        let intensity = emotion.intensity;

        if intensity > 0.8 {
            Dynamics::Forte
        } else if intensity > 0.6 {
            Dynamics::MezzoForte
        } else if intensity > 0.4 {
            Dynamics::MezzoPiano
        } else {
            Dynamics::Piano
        }
    }

    /// Convert emotion to articulation
    fn emotion_to_articulation(&self, emotion: &EmotionVector) -> Articulation {
        if emotion.arousal > 0.5 {
            Articulation::Staccato
        } else if emotion.valence < -0.5 {
            Articulation::Tenuto
        } else {
            Articulation::Normal
        }
    }

    /// Smooth emotion transition
    pub fn smooth_transition(
        &mut self,
        current: &EmotionVector,
        target: &EmotionVector,
        progress: f32,
    ) -> EmotionVector {
        self.interpolator
            .smooth_transition(current, target, progress)
    }

    /// Reset the emotion transfer engine
    pub fn reset(&mut self) {
        self.interpolator.reset();
        self.emotion_cache.clear();
    }
}

impl Default for EmotionTransferEngine {
    fn default() -> Self {
        Self::new(EmotionTransferConfig::default())
    }
}

/// Synthesis parameters derived from emotion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionSynthesisParams {
    /// Tempo scaling factor
    pub tempo_scale: f32,
    /// Pitch shift in semitones
    pub pitch_shift: f32,
    /// Dynamics level
    pub dynamics: Dynamics,
    /// Articulation style
    pub articulation: Articulation,
    /// Vibrato rate in Hz
    pub vibrato_rate: f32,
    /// Vibrato depth (0.0-1.0)
    pub vibrato_depth: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_vector_creation() {
        let emotion = EmotionVector::new(0.8, 0.6, 0.5, 0.9);
        assert_eq!(emotion.valence, 0.8);
        assert_eq!(emotion.arousal, 0.6);
        assert_eq!(emotion.dominance, 0.5);
        assert_eq!(emotion.intensity, 0.9);
    }

    #[test]
    fn test_emotion_vector_clamping() {
        let emotion = EmotionVector::new(2.0, -2.0, 1.5, -0.5);
        assert_eq!(emotion.valence, 1.0);
        assert_eq!(emotion.arousal, -1.0);
        assert_eq!(emotion.dominance, 1.0);
        assert_eq!(emotion.intensity, 0.0);
    }

    #[test]
    fn test_predefined_emotions() {
        let happy = EmotionVector::happy();
        assert!(happy.valence > 0.0);
        assert!(happy.intensity > 0.0);

        let sad = EmotionVector::sad();
        assert!(sad.valence < 0.0);

        let angry = EmotionVector::angry();
        assert!(angry.arousal > 0.0);
    }

    #[test]
    fn test_emotion_interpolation() {
        let happy = EmotionVector::happy();
        let sad = EmotionVector::sad();

        let mid = happy.interpolate(&sad, 0.5);
        assert!(mid.valence < happy.valence);
        assert!(mid.valence > sad.valence);
    }

    #[test]
    fn test_emotion_distance() {
        let happy = EmotionVector::happy();
        let sad = EmotionVector::sad();

        let distance = happy.distance(&sad);
        assert!(distance > 1.0);

        let same_distance = happy.distance(&happy);
        assert_eq!(same_distance, 0.0);
    }

    #[test]
    fn test_emotion_detector() {
        let detector = EmotionDetector::default();
        let audio = vec![0.5; 44100]; // 1 second of audio

        let result = detector.detect(&audio, 44100);
        assert!(result.confidence > 0.0);
        assert!(!result.label.is_empty());
        assert!(!result.frame_emotions.is_empty());
    }

    #[test]
    fn test_emotion_interpolator() {
        let mut interpolator = EmotionInterpolator::default();
        let current = EmotionVector::neutral();
        let target = EmotionVector::happy();

        let smoothed = interpolator.smooth_transition(&current, &target, 0.5);
        assert!(smoothed.valence >= 0.0);
    }

    #[test]
    fn test_emotion_transfer_engine() {
        let mut engine = EmotionTransferEngine::default();
        let audio = vec![0.5; 44100];

        let emotion = engine.detect_reference_emotion(&audio, 44100).unwrap();
        assert!(emotion.intensity >= 0.0);

        let params = engine.apply_emotion(&emotion);
        assert!(params.tempo_scale > 0.0);
        assert!(params.vibrato_rate > 0.0);
    }

    #[test]
    fn test_emotion_caching() {
        let config = EmotionTransferConfig {
            enable_caching: true,
            ..Default::default()
        };
        let mut engine = EmotionTransferEngine::new(config);

        let audio = vec![0.5; 44100];
        let emotion1 = engine.detect_reference_emotion(&audio, 44100).unwrap();
        let emotion2 = engine.detect_reference_emotion(&audio, 44100).unwrap();

        // Should return same result from cache
        assert_eq!(emotion1.valence, emotion2.valence);
    }

    #[test]
    fn test_smooth_transition() {
        let mut engine = EmotionTransferEngine::default();
        let current = EmotionVector::neutral();
        let target = EmotionVector::happy();

        let smoothed = engine.smooth_transition(&current, &target, 0.5);
        assert!(smoothed.valence > current.valence);
        assert!(smoothed.valence < target.valence);
    }

    #[test]
    fn test_emotion_to_label() {
        let detector = EmotionDetector::default();

        let happy = EmotionVector::happy();
        assert_eq!(detector.emotion_to_label(&happy), "happy");

        let sad = EmotionVector::sad();
        assert_eq!(detector.emotion_to_label(&sad), "sad");

        let angry = EmotionVector::angry();
        assert_eq!(detector.emotion_to_label(&angry), "angry");
    }

    #[test]
    fn test_config_defaults() {
        let config = EmotionTransferConfig::default();
        assert!(config.realtime);
        assert_eq!(config.smoothness, 0.7);
        assert_eq!(config.intensity_scale, 1.0);
    }
}
