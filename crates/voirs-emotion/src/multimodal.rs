//! # Multimodal Emotion Integration
//!
//! This module provides comprehensive integration with facial expression and gesture data
//! for enhanced emotion recognition and control. It combines voice-based emotion processing
//! with visual and sensor inputs for more accurate and natural emotional expressions.
//!
//! ## Features
//!
//! - **Facial Expression Analysis**: Computer vision-based facial emotion recognition
//! - **Body Gesture Recognition**: Full-body posture and gesture analysis
//! - **Eye Tracking Integration**: Gaze-based emotion indicators
//! - **Physiological Sensors**: Heart rate, skin conductance, and other biometric inputs
//! - **Multi-Modal Fusion**: Intelligent combination of multiple input modalities
//! - **Real-time Processing**: Low-latency multimodal emotion processing
//!
//! ## Example Usage
//!
//! ```rust
//! use voirs_emotion::multimodal::{MultimodalEmotionProcessor, FacialExpression, BodyPose};
//! use voirs_emotion::types::{Emotion, EmotionIntensity};
//!
//! // Create processor and data
//! let processor = MultimodalEmotionProcessor::new()?;
//!
//! // Create facial expression data (ensure high arousal for Happy emotion)
//! let facial_data = FacialExpression::new()
//!     .with_smile_intensity(0.8)
//!     .with_eyebrow_position(0.7)  // Higher eyebrow for more arousal
//!     .with_eye_openness(0.9)
//!     .with_mouth_openness(0.5)    // Add mouth openness for arousal
//!     .with_confidence(0.9);
//!
//! // Create body gesture data
//! let body_pose = BodyPose::new()
//!     .with_posture_confidence(0.7)
//!     .with_gesture_type("thumbs_up");
//!
//! // Test emotion inference from facial data
//! let (emotion, confidence) = facial_data.infer_emotion();
//! assert_eq!(emotion, Emotion::Happy);
//! assert_eq!(confidence, 0.9);
//!
//! // Test emotion inference from body pose
//! let (emotion, confidence) = body_pose.infer_emotion();
//! assert_eq!(emotion, Emotion::Happy);
//! assert_eq!(confidence, 0.7);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::types::{
    Emotion, EmotionDimensions, EmotionIntensity, EmotionParameters, EmotionVector,
};
use crate::{Error, Result};

/// Facial expression data from computer vision analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FacialExpression {
    /// Smile intensity (0.0 to 1.0)
    pub smile_intensity: f32,
    /// Frown intensity (0.0 to 1.0)
    pub frown_intensity: f32,
    /// Eyebrow position (-1.0: lowered, 0.0: neutral, 1.0: raised)
    pub eyebrow_position: f32,
    /// Eye openness (0.0: closed, 1.0: wide open)
    pub eye_openness: f32,
    /// Mouth openness (0.0: closed, 1.0: wide open)
    pub mouth_openness: f32,
    /// Jaw tension (0.0: relaxed, 1.0: tense)
    pub jaw_tension: f32,
    /// Nostril flare (0.0: normal, 1.0: flared)
    pub nostril_flare: f32,
    /// Cheek puff (0.0: normal, 1.0: puffed)
    pub cheek_puff: f32,
    /// Overall expression confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Timestamp of data capture
    #[serde(skip)]
    pub timestamp: Instant,
}

impl FacialExpression {
    /// Create new facial expression data
    pub fn new() -> Self {
        FacialExpression {
            smile_intensity: 0.0,
            frown_intensity: 0.0,
            eyebrow_position: 0.0,
            eye_openness: 0.7, // Default to normal eye openness
            mouth_openness: 0.0,
            jaw_tension: 0.0,
            nostril_flare: 0.0,
            cheek_puff: 0.0,
            confidence: 0.0,
            timestamp: Instant::now(),
        }
    }

    /// Set smile intensity
    pub fn with_smile_intensity(mut self, intensity: f32) -> Self {
        self.smile_intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set frown intensity
    pub fn with_frown_intensity(mut self, intensity: f32) -> Self {
        self.frown_intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set eyebrow position
    pub fn with_eyebrow_position(mut self, position: f32) -> Self {
        self.eyebrow_position = position.clamp(-1.0, 1.0);
        self
    }

    /// Set eye openness
    pub fn with_eye_openness(mut self, openness: f32) -> Self {
        self.eye_openness = openness.clamp(0.0, 1.0);
        self
    }

    /// Set mouth openness
    pub fn with_mouth_openness(mut self, openness: f32) -> Self {
        self.mouth_openness = openness.clamp(0.0, 1.0);
        self
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Convert facial expression to emotion dimensions
    pub fn to_emotion_dimensions(&self) -> EmotionDimensions {
        // Map facial features to VAD dimensions
        let valence = (self.smile_intensity - self.frown_intensity).clamp(-1.0, 1.0);
        let arousal = (self.eye_openness + self.eyebrow_position.abs() + self.mouth_openness) / 3.0;
        let arousal = (arousal * 2.0 - 1.0).clamp(-1.0, 1.0); // Scale to [-1, 1]
        let dominance = (self.eyebrow_position + self.jaw_tension - 0.5).clamp(-1.0, 1.0);

        EmotionDimensions {
            valence,
            arousal,
            dominance,
        }
    }

    /// Infer primary emotion from facial expression
    pub fn infer_emotion(&self) -> (Emotion, f32) {
        let dims = self.to_emotion_dimensions();

        // Simple emotion classification based on dimensional values
        if dims.valence > 0.3 && dims.arousal > 0.3 {
            if dims.arousal > 0.7 {
                (Emotion::Excited, self.confidence)
            } else {
                (Emotion::Happy, self.confidence)
            }
        } else if dims.valence < -0.3 && dims.arousal > 0.3 {
            if dims.dominance > 0.3 {
                (Emotion::Angry, self.confidence)
            } else {
                (Emotion::Fear, self.confidence)
            }
        } else if dims.valence < -0.3 && dims.arousal < -0.3 {
            (Emotion::Sad, self.confidence)
        } else if dims.arousal < -0.5 {
            (Emotion::Calm, self.confidence)
        } else {
            (Emotion::Neutral, self.confidence * 0.5) // Lower confidence for neutral
        }
    }
}

impl Default for FacialExpression {
    fn default() -> Self {
        Self::new()
    }
}

/// Body pose and gesture data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BodyPose {
    /// Head position (x, y, z) relative to body center
    pub head_position: [f32; 3],
    /// Head rotation (pitch, yaw, roll) in radians
    pub head_rotation: [f32; 3],
    /// Shoulder position and tension
    pub shoulder_position: f32, // -1.0: slouched, 0.0: normal, 1.0: raised
    /// Spine straightness (0.0: hunched, 1.0: straight)
    pub spine_straightness: f32,
    /// Arm positions (left, right) - gesture indicators
    pub arm_positions: [f32; 2],
    /// Hand gesture type (if detected)
    pub gesture_type: Option<String>,
    /// Overall posture confidence
    pub posture_confidence: f32,
    /// Movement energy level (0.0: still, 1.0: highly active)
    pub movement_energy: f32,
    /// Timestamp of data capture
    #[serde(skip)]
    pub timestamp: Instant,
}

impl BodyPose {
    /// Create new body pose data
    pub fn new() -> Self {
        BodyPose {
            head_position: [0.0, 0.0, 0.0],
            head_rotation: [0.0, 0.0, 0.0],
            shoulder_position: 0.0,
            spine_straightness: 0.7, // Default to reasonably straight
            arm_positions: [0.0, 0.0],
            gesture_type: None,
            posture_confidence: 0.0,
            movement_energy: 0.0,
            timestamp: Instant::now(),
        }
    }

    /// Set posture confidence
    pub fn with_posture_confidence(mut self, confidence: f32) -> Self {
        self.posture_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set gesture type
    pub fn with_gesture_type(mut self, gesture: &str) -> Self {
        self.gesture_type = Some(gesture.to_string());
        self
    }

    /// Set movement energy
    pub fn with_movement_energy(mut self, energy: f32) -> Self {
        self.movement_energy = energy.clamp(0.0, 1.0);
        self
    }

    /// Set spine straightness
    pub fn with_spine_straightness(mut self, straightness: f32) -> Self {
        self.spine_straightness = straightness.clamp(0.0, 1.0);
        self
    }

    /// Convert body pose to emotion dimensions
    pub fn to_emotion_dimensions(&self) -> EmotionDimensions {
        // Map body language to emotional dimensions
        let dominance = (self.spine_straightness + (1.0 + self.shoulder_position) / 2.0) / 2.0;
        let dominance = (dominance * 2.0 - 1.0).clamp(-1.0, 1.0); // Scale to [-1, 1]

        let arousal = self.movement_energy * 2.0 - 1.0; // Scale to [-1, 1]

        // Valence is harder to determine from posture alone, use gesture hints
        let valence = match &self.gesture_type {
            Some(gesture) => match gesture.as_str() {
                "thumbs_up" | "applause" | "wave" => 0.7,
                "thumbs_down" | "dismissive_wave" => -0.7,
                "pointing" | "stop_gesture" => 0.0,
                _ => 0.0,
            },
            None => 0.0,
        };

        EmotionDimensions {
            valence,
            arousal,
            dominance,
        }
    }

    /// Infer emotion from body pose
    pub fn infer_emotion(&self) -> (Emotion, f32) {
        let dims = self.to_emotion_dimensions();

        // Gesture-based emotion inference
        if let Some(gesture) = &self.gesture_type {
            let confidence = self.posture_confidence;
            return match gesture.as_str() {
                "thumbs_up" | "applause" => (Emotion::Happy, confidence),
                "thumbs_down" => (Emotion::Sad, confidence),
                "fist" | "aggressive_point" => (Emotion::Angry, confidence),
                "wave" => (Emotion::Excited, confidence * 0.8),
                "defensive_posture" => (Emotion::Fear, confidence),
                _ => (Emotion::Neutral, confidence * 0.5),
            };
        }

        // Posture-based emotion inference
        if dims.dominance > 0.5 && dims.arousal > 0.3 {
            (Emotion::Confident, self.posture_confidence)
        } else if dims.dominance < -0.3 && dims.arousal < 0.0 {
            (Emotion::Sad, self.posture_confidence)
        } else if dims.arousal > 0.7 {
            (Emotion::Excited, self.posture_confidence)
        } else if dims.arousal < -0.5 {
            (Emotion::Calm, self.posture_confidence)
        } else {
            (Emotion::Neutral, self.posture_confidence * 0.6)
        }
    }
}

impl Default for BodyPose {
    fn default() -> Self {
        Self::new()
    }
}

/// Eye tracking data for emotion analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EyeTrackingData {
    /// Gaze direction (x, y, z) normalized vector
    pub gaze_direction: [f32; 3],
    /// Pupil dilation (0.0: constricted, 1.0: dilated)
    pub pupil_dilation: f32,
    /// Blink rate (blinks per minute)
    pub blink_rate: f32,
    /// Fixation duration (seconds)
    pub fixation_duration: f32,
    /// Saccade velocity (degrees per second)
    pub saccade_velocity: f32,
    /// Eye tracking confidence
    pub tracking_confidence: f32,
    /// Timestamp of data capture
    #[serde(skip)]
    pub timestamp: Instant,
}

impl EyeTrackingData {
    /// Create new eye tracking data
    pub fn new() -> Self {
        EyeTrackingData {
            gaze_direction: [0.0, 0.0, -1.0], // Looking forward
            pupil_dilation: 0.5,              // Normal dilation
            blink_rate: 15.0,                 // Normal blink rate
            fixation_duration: 0.3,           // Normal fixation
            saccade_velocity: 300.0,          // Normal saccade speed
            tracking_confidence: 0.0,
            timestamp: Instant::now(),
        }
    }

    /// Set pupil dilation
    pub fn with_pupil_dilation(mut self, dilation: f32) -> Self {
        self.pupil_dilation = dilation.clamp(0.0, 1.0);
        self
    }

    /// Set blink rate
    pub fn with_blink_rate(mut self, rate: f32) -> Self {
        self.blink_rate = rate.max(0.0);
        self
    }

    /// Set tracking confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.tracking_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Convert eye tracking data to emotion indicators
    pub fn to_emotion_dimensions(&self) -> EmotionDimensions {
        // Map eye tracking metrics to emotional dimensions

        // High pupil dilation often indicates arousal/stress
        let arousal = (self.pupil_dilation - 0.5) * 2.0; // Scale to [-1, 1]

        // Rapid blinking can indicate stress or excitement
        let blink_arousal = if self.blink_rate > 20.0 {
            ((self.blink_rate - 20.0) / 20.0).min(1.0)
        } else if self.blink_rate < 10.0 {
            -((10.0 - self.blink_rate) / 10.0).min(1.0)
        } else {
            0.0
        };

        let combined_arousal = ((arousal + blink_arousal) / 2.0).clamp(-1.0, 1.0);

        // Valence is harder to determine from eye tracking alone
        let valence = 0.0;

        // Dominance can be inferred from gaze patterns (simplified)
        let dominance = if self.fixation_duration > 0.5 {
            0.3 // Sustained gaze indicates some dominance
        } else {
            -0.1 // Wandering gaze indicates less dominance
        };

        EmotionDimensions {
            valence,
            arousal: combined_arousal,
            dominance,
        }
    }

    /// Infer emotion from eye tracking
    pub fn infer_emotion(&self) -> (Emotion, f32) {
        let dims = self.to_emotion_dimensions();

        if dims.arousal > 0.5 {
            if self.blink_rate > 25.0 {
                (Emotion::Angry, self.tracking_confidence * 0.7)
            } else {
                (Emotion::Excited, self.tracking_confidence * 0.6)
            }
        } else if dims.arousal < -0.5 {
            (Emotion::Calm, self.tracking_confidence * 0.6)
        } else {
            (Emotion::Neutral, self.tracking_confidence * 0.4)
        }
    }
}

impl Default for EyeTrackingData {
    fn default() -> Self {
        Self::new()
    }
}

/// Physiological sensor data for emotion analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhysiologicalData {
    /// Heart rate (BPM)
    pub heart_rate: f32,
    /// Heart rate variability (RMSSD in ms)
    pub heart_rate_variability: f32,
    /// Skin conductance (microsiemens)
    pub skin_conductance: f32,
    /// Skin temperature (Celsius)
    pub skin_temperature: f32,
    /// Respiration rate (breaths per minute)
    pub respiration_rate: f32,
    /// Blood pressure systolic
    pub blood_pressure_systolic: Option<f32>,
    /// Blood pressure diastolic
    pub blood_pressure_diastolic: Option<f32>,
    /// Sensor data confidence
    pub sensor_confidence: f32,
    /// Timestamp of data capture
    #[serde(skip)]
    pub timestamp: Instant,
}

impl PhysiologicalData {
    /// Create new physiological data
    pub fn new() -> Self {
        PhysiologicalData {
            heart_rate: 70.0,             // Normal resting heart rate
            heart_rate_variability: 50.0, // Normal HRV
            skin_conductance: 5.0,        // Normal skin conductance
            skin_temperature: 32.0,       // Normal skin temperature
            respiration_rate: 16.0,       // Normal respiration rate
            blood_pressure_systolic: None,
            blood_pressure_diastolic: None,
            sensor_confidence: 0.0,
            timestamp: Instant::now(),
        }
    }

    /// Set heart rate
    pub fn with_heart_rate(mut self, bpm: f32) -> Self {
        self.heart_rate = bpm.max(30.0).min(200.0); // Reasonable bounds
        self
    }

    /// Set skin conductance
    pub fn with_skin_conductance(mut self, conductance: f32) -> Self {
        self.skin_conductance = conductance.max(0.0);
        self
    }

    /// Set sensor confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.sensor_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Convert physiological data to emotion dimensions
    pub fn to_emotion_dimensions(&self) -> EmotionDimensions {
        // Map physiological metrics to emotional dimensions

        // Heart rate and skin conductance indicate arousal
        let hr_arousal = (self.heart_rate - 70.0) / 30.0; // Normalize around 70 BPM
        let sc_arousal = (self.skin_conductance - 5.0) / 5.0; // Normalize around 5 μS
        let resp_arousal = (self.respiration_rate - 16.0) / 8.0; // Normalize around 16 BPM

        let arousal = ((hr_arousal + sc_arousal + resp_arousal) / 3.0).clamp(-1.0, 1.0);

        // Valence is difficult to determine from physiology alone
        let valence = 0.0;

        // HRV can indicate dominance/control
        let dominance = ((self.heart_rate_variability - 50.0) / 25.0).clamp(-1.0, 1.0);

        EmotionDimensions {
            valence,
            arousal,
            dominance,
        }
    }

    /// Infer emotion from physiological data
    pub fn infer_emotion(&self) -> (Emotion, f32) {
        let dims = self.to_emotion_dimensions();

        if dims.arousal > 0.6 {
            if self.skin_conductance > 8.0 {
                (Emotion::Angry, self.sensor_confidence * 0.8)
            } else {
                (Emotion::Excited, self.sensor_confidence * 0.7)
            }
        } else if dims.arousal < -0.3 {
            (Emotion::Calm, self.sensor_confidence * 0.8)
        } else if dims.dominance < -0.5 {
            (Emotion::Fear, self.sensor_confidence * 0.6)
        } else {
            (Emotion::Neutral, self.sensor_confidence * 0.5)
        }
    }
}

impl Default for PhysiologicalData {
    fn default() -> Self {
        Self::new()
    }
}

/// Multimodal fusion configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalConfig {
    /// Weight for facial expression input (0.0 to 1.0)
    pub facial_weight: f32,
    /// Weight for body pose input (0.0 to 1.0)
    pub body_weight: f32,
    /// Weight for eye tracking input (0.0 to 1.0)
    pub eye_tracking_weight: f32,
    /// Weight for physiological input (0.0 to 1.0)
    pub physiological_weight: f32,
    /// Weight for voice/audio input (0.0 to 1.0)
    pub voice_weight: f32,
    /// Minimum confidence threshold for input acceptance
    pub confidence_threshold: f32,
    /// Data freshness timeout (seconds)
    pub data_timeout_secs: f32,
    /// Enable temporal smoothing
    pub enable_temporal_smoothing: bool,
    /// Smoothing factor for temporal data (0.0 to 1.0)
    pub smoothing_factor: f32,
}

impl Default for MultimodalConfig {
    fn default() -> Self {
        MultimodalConfig {
            facial_weight: 0.3,
            body_weight: 0.2,
            eye_tracking_weight: 0.1,
            physiological_weight: 0.2,
            voice_weight: 0.2,
            confidence_threshold: 0.3,
            data_timeout_secs: 5.0,
            enable_temporal_smoothing: true,
            smoothing_factor: 0.7,
        }
    }
}

impl MultimodalConfig {
    /// Create new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set facial expression weight
    pub fn with_facial_weight(mut self, weight: f32) -> Self {
        self.facial_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set body pose weight
    pub fn with_body_weight(mut self, weight: f32) -> Self {
        self.body_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set voice weight
    pub fn with_voice_weight(mut self, weight: f32) -> Self {
        self.voice_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Enable/disable temporal smoothing
    pub fn with_temporal_smoothing(mut self, enabled: bool) -> Self {
        self.enable_temporal_smoothing = enabled;
        self
    }
}

/// Multimodal emotion analysis result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MultimodalEmotionResult {
    /// Primary detected emotion
    pub primary_emotion: Emotion,
    /// Primary emotion intensity
    pub primary_intensity: EmotionIntensity,
    /// Combined emotion dimensions
    pub dimensions: EmotionDimensions,
    /// Individual modality contributions
    pub modality_contributions: HashMap<String, f32>,
    /// Overall confidence score
    pub overall_confidence: f32,
    /// Processing timestamp
    #[serde(skip)]
    pub timestamp: Instant,
}

impl MultimodalEmotionResult {
    /// Convert to emotion parameters
    pub fn to_emotion_parameters(&self) -> EmotionParameters {
        let mut emotion_vector = EmotionVector::new();
        emotion_vector.dimensions = self.dimensions;
        emotion_vector.add_emotion(self.primary_emotion.clone(), self.primary_intensity);

        EmotionParameters::new(emotion_vector)
    }
}

impl Default for MultimodalEmotionResult {
    fn default() -> Self {
        MultimodalEmotionResult {
            primary_emotion: Emotion::Neutral,
            primary_intensity: EmotionIntensity::VERY_LOW,
            dimensions: EmotionDimensions::default(),
            modality_contributions: HashMap::new(),
            overall_confidence: 0.0,
            timestamp: Instant::now(),
        }
    }
}

/// Main multimodal emotion processor
pub struct MultimodalEmotionProcessor {
    /// Configuration
    config: MultimodalConfig,
    /// Latest facial expression data
    facial_data: RwLock<Option<FacialExpression>>,
    /// Latest body pose data
    body_data: RwLock<Option<BodyPose>>,
    /// Latest eye tracking data
    eye_data: RwLock<Option<EyeTrackingData>>,
    /// Latest physiological data
    physiological_data: RwLock<Option<PhysiologicalData>>,
    /// Latest voice emotion data
    voice_emotion: RwLock<Option<EmotionParameters>>,
    /// Previous emotion result for temporal smoothing
    previous_result: RwLock<Option<MultimodalEmotionResult>>,
    /// Processing statistics
    processing_stats: RwLock<HashMap<String, f32>>,
}

impl MultimodalEmotionProcessor {
    /// Create new multimodal emotion processor
    pub fn new() -> Result<Self> {
        Ok(MultimodalEmotionProcessor {
            config: MultimodalConfig::default(),
            facial_data: RwLock::new(None),
            body_data: RwLock::new(None),
            eye_data: RwLock::new(None),
            physiological_data: RwLock::new(None),
            voice_emotion: RwLock::new(None),
            previous_result: RwLock::new(None),
            processing_stats: RwLock::new(HashMap::new()),
        })
    }

    /// Create processor with custom configuration
    pub fn with_config(config: MultimodalConfig) -> Result<Self> {
        let mut processor = Self::new()?;
        processor.config = config;
        Ok(processor)
    }

    /// Update facial expression data
    pub async fn update_facial_expression(&self, data: FacialExpression) -> Result<()> {
        if data.confidence >= self.config.confidence_threshold {
            let mut facial_data = self.facial_data.write().await;
            *facial_data = Some(data);
        }
        Ok(())
    }

    /// Update body pose data
    pub async fn update_body_pose(&self, data: BodyPose) -> Result<()> {
        if data.posture_confidence >= self.config.confidence_threshold {
            let mut body_data = self.body_data.write().await;
            *body_data = Some(data);
        }
        Ok(())
    }

    /// Update eye tracking data
    pub async fn update_eye_tracking(&self, data: EyeTrackingData) -> Result<()> {
        if data.tracking_confidence >= self.config.confidence_threshold {
            let mut eye_data = self.eye_data.write().await;
            *eye_data = Some(data);
        }
        Ok(())
    }

    /// Update physiological data
    pub async fn update_physiological_data(&self, data: PhysiologicalData) -> Result<()> {
        if data.sensor_confidence >= self.config.confidence_threshold {
            let mut phys_data = self.physiological_data.write().await;
            *phys_data = Some(data);
        }
        Ok(())
    }

    /// Update voice emotion data
    pub async fn update_voice_emotion(&self, emotion: EmotionParameters) -> Result<()> {
        let mut voice_data = self.voice_emotion.write().await;
        *voice_data = Some(emotion);
        Ok(())
    }

    /// Process multimodal emotion fusion
    pub async fn process_multimodal_emotion(&self) -> Result<MultimodalEmotionResult> {
        let now = Instant::now();
        let timeout = Duration::from_secs_f32(self.config.data_timeout_secs);

        // Collect valid modality data
        let mut total_weight = 0.0;
        let mut combined_dimensions = EmotionDimensions::default();
        let mut emotion_votes: HashMap<Emotion, f32> = HashMap::new();
        let mut modality_contributions: HashMap<String, f32> = HashMap::new();
        let mut overall_confidence = 0.0;

        // Process facial data
        if let Some(facial) = self.facial_data.read().await.as_ref() {
            if now.duration_since(facial.timestamp) <= timeout {
                let weight = self.config.facial_weight * facial.confidence;
                let dims = facial.to_emotion_dimensions();
                let (emotion, confidence) = facial.infer_emotion();

                combined_dimensions.valence += dims.valence * weight;
                combined_dimensions.arousal += dims.arousal * weight;
                combined_dimensions.dominance += dims.dominance * weight;

                *emotion_votes.entry(emotion).or_insert(0.0) += weight * confidence;
                modality_contributions.insert("facial".to_string(), weight);
                total_weight += weight;
                overall_confidence += confidence * weight;
            }
        }

        // Process body data
        if let Some(body) = self.body_data.read().await.as_ref() {
            if now.duration_since(body.timestamp) <= timeout {
                let weight = self.config.body_weight * body.posture_confidence;
                let dims = body.to_emotion_dimensions();
                let (emotion, confidence) = body.infer_emotion();

                combined_dimensions.valence += dims.valence * weight;
                combined_dimensions.arousal += dims.arousal * weight;
                combined_dimensions.dominance += dims.dominance * weight;

                *emotion_votes.entry(emotion).or_insert(0.0) += weight * confidence;
                modality_contributions.insert("body".to_string(), weight);
                total_weight += weight;
                overall_confidence += confidence * weight;
            }
        }

        // Process eye tracking data
        if let Some(eye) = self.eye_data.read().await.as_ref() {
            if now.duration_since(eye.timestamp) <= timeout {
                let weight = self.config.eye_tracking_weight * eye.tracking_confidence;
                let dims = eye.to_emotion_dimensions();
                let (emotion, confidence) = eye.infer_emotion();

                combined_dimensions.valence += dims.valence * weight;
                combined_dimensions.arousal += dims.arousal * weight;
                combined_dimensions.dominance += dims.dominance * weight;

                *emotion_votes.entry(emotion).or_insert(0.0) += weight * confidence;
                modality_contributions.insert("eye_tracking".to_string(), weight);
                total_weight += weight;
                overall_confidence += confidence * weight;
            }
        }

        // Process physiological data
        if let Some(phys) = self.physiological_data.read().await.as_ref() {
            if now.duration_since(phys.timestamp) <= timeout {
                let weight = self.config.physiological_weight * phys.sensor_confidence;
                let dims = phys.to_emotion_dimensions();
                let (emotion, confidence) = phys.infer_emotion();

                combined_dimensions.valence += dims.valence * weight;
                combined_dimensions.arousal += dims.arousal * weight;
                combined_dimensions.dominance += dims.dominance * weight;

                *emotion_votes.entry(emotion).or_insert(0.0) += weight * confidence;
                modality_contributions.insert("physiological".to_string(), weight);
                total_weight += weight;
                overall_confidence += confidence * weight;
            }
        }

        // Process voice emotion data
        if let Some(voice) = self.voice_emotion.read().await.as_ref() {
            let weight = self.config.voice_weight;
            let dims = &voice.emotion_vector.dimensions;

            combined_dimensions.valence += dims.valence * weight;
            combined_dimensions.arousal += dims.arousal * weight;
            combined_dimensions.dominance += dims.dominance * weight;

            // Get dominant emotion from voice
            if let Some((emotion, intensity)) = voice.emotion_vector.dominant_emotion() {
                *emotion_votes.entry(emotion).or_insert(0.0) += weight * intensity.value();
            }

            modality_contributions.insert("voice".to_string(), weight);
            total_weight += weight;
            overall_confidence += weight; // Voice always has confidence of 1.0
        }

        // Normalize results
        if total_weight > 0.0 {
            combined_dimensions.valence /= total_weight;
            combined_dimensions.arousal /= total_weight;
            combined_dimensions.dominance /= total_weight;
            overall_confidence /= total_weight;
        }

        // Clamp dimensions to valid range
        combined_dimensions.valence = combined_dimensions.valence.clamp(-1.0, 1.0);
        combined_dimensions.arousal = combined_dimensions.arousal.clamp(-1.0, 1.0);
        combined_dimensions.dominance = combined_dimensions.dominance.clamp(-1.0, 1.0);

        // Determine primary emotion from votes
        let (primary_emotion, vote_strength) = emotion_votes
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((Emotion::Neutral, 0.0));

        let primary_intensity = EmotionIntensity::new(vote_strength.min(1.0));

        let mut result = MultimodalEmotionResult {
            primary_emotion,
            primary_intensity,
            dimensions: combined_dimensions,
            modality_contributions,
            overall_confidence,
            timestamp: now,
        };

        // Apply temporal smoothing if enabled
        if self.config.enable_temporal_smoothing {
            result = self.apply_temporal_smoothing(result).await;
        }

        // Update previous result
        let mut prev_result = self.previous_result.write().await;
        *prev_result = Some(result.clone());

        // Update processing statistics
        let mut stats = self.processing_stats.write().await;
        stats.insert(
            "last_processing_time_ms".to_string(),
            now.elapsed().as_millis() as f32,
        );
        stats.insert(
            "active_modalities".to_string(),
            result.modality_contributions.len() as f32,
        );
        stats.insert("total_weight".to_string(), total_weight);

        Ok(result)
    }

    /// Apply temporal smoothing to emotion result
    async fn apply_temporal_smoothing(
        &self,
        mut current: MultimodalEmotionResult,
    ) -> MultimodalEmotionResult {
        if let Some(previous) = self.previous_result.read().await.as_ref() {
            let smoothing = self.config.smoothing_factor;

            // Smooth dimensions
            current.dimensions.valence = previous.dimensions.valence * smoothing
                + current.dimensions.valence * (1.0 - smoothing);
            current.dimensions.arousal = previous.dimensions.arousal * smoothing
                + current.dimensions.arousal * (1.0 - smoothing);
            current.dimensions.dominance = previous.dimensions.dominance * smoothing
                + current.dimensions.dominance * (1.0 - smoothing);

            // Smooth confidence
            current.overall_confidence = previous.overall_confidence * smoothing
                + current.overall_confidence * (1.0 - smoothing);
        }

        current
    }

    /// Get processing statistics  
    pub async fn get_processing_stats(&self) -> HashMap<String, f32> {
        self.processing_stats.read().await.clone()
    }

    /// Clear all modality data
    pub async fn clear_all_data(&self) -> Result<()> {
        *self.facial_data.write().await = None;
        *self.body_data.write().await = None;
        *self.eye_data.write().await = None;
        *self.physiological_data.write().await = None;
        *self.voice_emotion.write().await = None;
        *self.previous_result.write().await = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_facial_expression_creation() {
        let facial = FacialExpression::new()
            .with_smile_intensity(0.8)
            .with_eyebrow_position(0.7) // More arousal
            .with_mouth_openness(0.5) // Additional arousal
            .with_eye_openness(1.0) // Maximum eye openness
            .with_confidence(0.9);

        assert_eq!(facial.smile_intensity, 0.8);
        assert_eq!(facial.confidence, 0.9);

        let (emotion, confidence) = facial.infer_emotion();
        assert_eq!(emotion, Emotion::Happy);
        assert_eq!(confidence, 0.9);
    }

    #[test]
    fn test_body_pose_creation() {
        let body = BodyPose::new()
            .with_gesture_type("thumbs_up")
            .with_posture_confidence(0.8);

        assert_eq!(body.gesture_type, Some("thumbs_up".to_string()));
        assert_eq!(body.posture_confidence, 0.8);

        let (emotion, confidence) = body.infer_emotion();
        assert_eq!(emotion, Emotion::Happy);
        assert_eq!(confidence, 0.8);
    }

    #[test]
    fn test_eye_tracking_data() {
        let eye_data = EyeTrackingData::new()
            .with_pupil_dilation(0.9)
            .with_blink_rate(26.0) // > 25.0 for Angry
            .with_confidence(0.7);

        assert_eq!(eye_data.pupil_dilation, 0.9);
        assert_eq!(eye_data.blink_rate, 26.0);

        let (emotion, _) = eye_data.infer_emotion();
        assert_eq!(emotion, Emotion::Angry); // High blink rate + high arousal
    }

    #[test]
    fn test_physiological_data() {
        let phys_data = PhysiologicalData::new()
            .with_heart_rate(110.0) // Higher heart rate for more arousal
            .with_skin_conductance(10.0) // > 8.0 for Angry
            .with_confidence(0.8);

        assert_eq!(phys_data.heart_rate, 110.0);
        assert_eq!(phys_data.skin_conductance, 10.0);

        let (emotion, _) = phys_data.infer_emotion();
        assert_eq!(emotion, Emotion::Angry); // High arousal + high skin conductance
    }

    #[test]
    fn test_multimodal_config() {
        let config = MultimodalConfig::new()
            .with_facial_weight(0.5)
            .with_voice_weight(0.3)
            .with_confidence_threshold(0.6)
            .with_temporal_smoothing(true);

        assert_eq!(config.facial_weight, 0.5);
        assert_eq!(config.voice_weight, 0.3);
        assert_eq!(config.confidence_threshold, 0.6);
        assert!(config.enable_temporal_smoothing);
    }

    #[tokio::test]
    async fn test_multimodal_processor_creation() {
        let processor = MultimodalEmotionProcessor::new();
        assert!(processor.is_ok());
    }

    #[tokio::test]
    async fn test_facial_expression_update() {
        let processor = MultimodalEmotionProcessor::new().unwrap();

        let facial_data = FacialExpression::new()
            .with_smile_intensity(0.8)
            .with_confidence(0.9);

        let result = processor.update_facial_expression(facial_data).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_body_pose_update() {
        let processor = MultimodalEmotionProcessor::new().unwrap();

        let body_data = BodyPose::new()
            .with_gesture_type("wave")
            .with_posture_confidence(0.8);

        let result = processor.update_body_pose(body_data).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_multimodal_emotion_processing() {
        let processor = MultimodalEmotionProcessor::new().unwrap();

        // Add facial expression data
        let facial_data = FacialExpression::new()
            .with_smile_intensity(0.8)
            .with_eyebrow_position(0.7) // More arousal
            .with_mouth_openness(0.5) // Additional arousal
            .with_eye_openness(1.0) // Maximum eye openness
            .with_confidence(0.9);
        processor
            .update_facial_expression(facial_data)
            .await
            .unwrap();

        // Add body pose data
        let body_data = BodyPose::new()
            .with_gesture_type("thumbs_up")
            .with_posture_confidence(0.8);
        processor.update_body_pose(body_data).await.unwrap();

        // Process multimodal emotion
        let result = processor.process_multimodal_emotion().await;
        assert!(result.is_ok());

        let emotion_result = result.unwrap();
        assert_eq!(emotion_result.primary_emotion, Emotion::Happy);
        assert!(emotion_result.overall_confidence > 0.0);
        assert!(emotion_result.modality_contributions.contains_key("facial"));
        assert!(emotion_result.modality_contributions.contains_key("body"));
    }

    #[tokio::test]
    async fn test_data_timeout() {
        let mut config = MultimodalConfig::new();
        config.data_timeout_secs = 0.001; // Very short timeout

        let processor = MultimodalEmotionProcessor::with_config(config).unwrap();

        // Add data
        let facial_data = FacialExpression::new()
            .with_smile_intensity(0.8)
            .with_confidence(0.9);
        processor
            .update_facial_expression(facial_data)
            .await
            .unwrap();

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Process - should have no valid data due to timeout
        let result = processor.process_multimodal_emotion().await.unwrap();
        assert_eq!(result.primary_emotion, Emotion::Neutral);
        assert!(result.modality_contributions.is_empty());
    }

    #[test]
    fn test_emotion_dimensions_conversion() {
        let facial = FacialExpression::new()
            .with_smile_intensity(0.9)
            .with_frown_intensity(0.1);

        let dims = facial.to_emotion_dimensions();
        assert!(dims.valence > 0.5); // Should be positive due to smile

        let body = BodyPose::new()
            .with_spine_straightness(0.9)
            .with_movement_energy(0.8);

        let dims = body.to_emotion_dimensions();
        assert!(dims.dominance > 0.0); // Should indicate dominance
        assert!(dims.arousal > 0.0); // Should indicate high arousal
    }

    #[tokio::test]
    async fn test_clear_all_data() {
        let processor = MultimodalEmotionProcessor::new().unwrap();

        // Add some data
        let facial_data = FacialExpression::new().with_confidence(0.9);
        processor
            .update_facial_expression(facial_data)
            .await
            .unwrap();

        // Clear all data
        processor.clear_all_data().await.unwrap();

        // Process should return neutral emotion
        let result = processor.process_multimodal_emotion().await.unwrap();
        assert_eq!(result.primary_emotion, Emotion::Neutral);
    }
}
