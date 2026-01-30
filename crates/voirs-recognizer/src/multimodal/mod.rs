//! Multi-modal processing for `VoiRS` recognizer.
//!
//! This module provides infrastructure for combining multiple modalities (audio, visual, gesture, context)
//! to improve speech recognition accuracy and robustness.

pub mod audio_visual;
pub mod context_aware;
pub mod fusion;
pub mod gesture;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Multi-modal processing errors
#[derive(Debug, Error)]
pub enum MultiModalError {
    /// Audio processing failed
    #[error("Audio processing error: {0}")]
    AudioError(String),

    /// Visual processing failed
    #[error("Visual processing error: {0}")]
    VisualError(String),

    /// Gesture recognition failed
    #[error("Gesture recognition error: {0}")]
    GestureError(String),

    /// Context analysis failed
    #[error("Context analysis error: {0}")]
    ContextError(String),

    /// Fusion strategy failed
    #[error("Fusion error: {0}")]
    FusionError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Modality synchronization error
    #[error("Synchronization error: {0}")]
    SyncError(String),
}

/// Multi-modal configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalConfig {
    /// Enable audio-visual fusion
    pub enable_audio_visual: bool,

    /// Enable gesture recognition
    pub enable_gesture: bool,

    /// Enable context-aware processing
    pub enable_context: bool,

    /// Fusion strategy
    pub fusion_strategy: FusionStrategy,

    /// Synchronization window in milliseconds
    pub sync_window_ms: u32,

    /// Visual feature extraction settings
    pub visual_config: VisualConfig,

    /// Gesture recognition settings
    pub gesture_config: GestureConfig,

    /// Context-aware settings
    pub context_config: ContextConfig,
}

impl Default for MultiModalConfig {
    fn default() -> Self {
        Self {
            enable_audio_visual: true,
            enable_gesture: false,
            enable_context: true,
            fusion_strategy: FusionStrategy::LateFusion,
            sync_window_ms: 100,
            visual_config: VisualConfig::default(),
            gesture_config: GestureConfig::default(),
            context_config: ContextConfig::default(),
        }
    }
}

/// Fusion strategies for combining modalities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Early fusion: concatenate features before processing
    EarlyFusion,
    /// Late fusion: combine predictions from each modality
    LateFusion,
    /// Hybrid fusion: both early and late fusion
    HybridFusion,
    /// Attention-based fusion: learned attention weights
    AttentionFusion,
    /// Hierarchical fusion: multi-level fusion
    HierarchicalFusion,
}

/// Visual processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualConfig {
    /// Enable lip reading
    pub enable_lip_reading: bool,

    /// Enable facial expression analysis
    pub enable_facial_expression: bool,

    /// Enable head pose estimation
    pub enable_head_pose: bool,

    /// Visual feature extraction method
    pub feature_method: VisualFeatureMethod,

    /// Target video FPS for processing
    pub target_fps: u32,

    /// Face detection confidence threshold
    pub face_detection_threshold: f32,

    /// Lip region extraction method
    pub lip_extraction_method: LipExtractionMethod,
}

impl Default for VisualConfig {
    fn default() -> Self {
        Self {
            enable_lip_reading: true,
            enable_facial_expression: true,
            enable_head_pose: true,
            feature_method: VisualFeatureMethod::DeepLearning,
            target_fps: 25,
            face_detection_threshold: 0.8,
            lip_extraction_method: LipExtractionMethod::DlibLandmarks,
        }
    }
}

/// Visual feature extraction methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualFeatureMethod {
    /// Deep learning-based features (CNNs)
    DeepLearning,
    /// Traditional computer vision features
    Traditional,
    /// Hybrid approach
    Hybrid,
}

/// Lip extraction methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LipExtractionMethod {
    /// Dlib facial landmarks
    DlibLandmarks,
    /// `MediaPipe` face mesh
    MediaPipe,
    /// Custom neural network
    CustomNN,
}

/// Gesture recognition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GestureConfig {
    /// Enable hand gesture recognition
    pub enable_hand_gestures: bool,

    /// Enable body pose estimation
    pub enable_body_pose: bool,

    /// Enable pointing gesture detection
    pub enable_pointing: bool,

    /// Gesture detection confidence threshold
    pub confidence_threshold: f32,

    /// Temporal smoothing window size
    pub smoothing_window: usize,
}

impl Default for GestureConfig {
    fn default() -> Self {
        Self {
            enable_hand_gestures: true,
            enable_body_pose: true,
            enable_pointing: true,
            confidence_threshold: 0.7,
            smoothing_window: 5,
        }
    }
}

/// Context-aware processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Enable environmental context analysis
    pub enable_environment: bool,

    /// Enable conversation history tracking
    pub enable_conversation_history: bool,

    /// Enable user profile adaptation
    pub enable_user_profile: bool,

    /// Enable temporal context (time of day, etc.)
    pub enable_temporal_context: bool,

    /// Maximum conversation history length
    pub max_history_length: usize,

    /// Context decay factor (how much old context matters)
    pub context_decay: f32,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            enable_environment: true,
            enable_conversation_history: true,
            enable_user_profile: true,
            enable_temporal_context: true,
            max_history_length: 10,
            context_decay: 0.9,
        }
    }
}

/// Multi-modal input data
#[derive(Debug, Clone)]
pub struct MultiModalInput {
    /// Audio samples
    pub audio: Vec<f32>,

    /// Audio sample rate
    pub sample_rate: u32,

    /// Video frames (optional)
    pub video_frames: Option<Vec<VideoFrame>>,

    /// Gesture data (optional)
    pub gesture_data: Option<GestureData>,

    /// Context information (optional)
    pub context: Option<ContextInformation>,

    /// Timestamp for synchronization
    pub timestamp_ms: u64,
}

/// Video frame data
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Frame data (RGB or grayscale)
    pub data: Vec<u8>,

    /// Frame width
    pub width: usize,

    /// Frame height
    pub height: usize,

    /// Number of channels (3 for RGB, 1 for grayscale)
    pub channels: usize,

    /// Frame timestamp
    pub timestamp_ms: u64,
}

/// Gesture data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GestureData {
    /// Hand keypoints (if available)
    pub hand_keypoints: Option<Vec<Keypoint>>,

    /// Body keypoints (if available)
    pub body_keypoints: Option<Vec<Keypoint>>,

    /// Detected gesture type
    pub gesture_type: Option<GestureType>,

    /// Gesture confidence score
    pub confidence: f32,

    /// Timestamp
    pub timestamp_ms: u64,
}

/// 2D/3D keypoint
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Keypoint {
    /// X coordinate
    pub x: f32,

    /// Y coordinate
    pub y: f32,

    /// Z coordinate (optional, for 3D)
    pub z: Option<f32>,

    /// Confidence score
    pub confidence: f32,
}

/// Gesture types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GestureType {
    /// Pointing gesture
    Pointing,
    /// Wave gesture
    Wave,
    /// Thumbs up
    ThumbsUp,
    /// Thumbs down
    ThumbsDown,
    /// OK sign
    OkSign,
    /// Open palm
    OpenPalm,
    /// Fist
    Fist,
    /// Custom gesture
    Custom,
}

/// Context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextInformation {
    /// Environmental context
    pub environment: EnvironmentContext,

    /// Conversation history
    pub conversation_history: Vec<String>,

    /// User profile data
    pub user_profile: Option<UserProfile>,

    /// Temporal context
    pub temporal: TemporalContext,
}

/// Environmental context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentContext {
    /// Noise level (dB)
    pub noise_level: Option<f32>,

    /// Number of speakers
    pub speaker_count: Option<usize>,

    /// Location type
    pub location: Option<LocationType>,

    /// Background activity level
    pub activity_level: Option<ActivityLevel>,
}

/// Location types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LocationType {
    /// Indoor environment
    Indoor,
    /// Outdoor environment
    Outdoor,
    /// Vehicle
    Vehicle,
    /// Office
    Office,
    /// Home
    Home,
    /// Public space
    PublicSpace,
}

/// Activity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivityLevel {
    /// Quiet environment
    Quiet,
    /// Normal activity
    Normal,
    /// Busy environment
    Busy,
    /// Very busy/noisy
    VeryBusy,
}

/// User profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// User ID
    pub user_id: String,

    /// Preferred language
    pub language: Option<String>,

    /// Speech patterns
    pub speech_patterns: Option<SpeechPatterns>,

    /// Common topics
    pub common_topics: Vec<String>,

    /// Accent type
    pub accent: Option<String>,
}

/// Speech pattern characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechPatterns {
    /// Average speaking rate (words per minute)
    pub speaking_rate: f32,

    /// Common filler words
    pub filler_words: Vec<String>,

    /// Pronunciation variants
    pub pronunciation_variants: Vec<(String, String)>,
}

/// Temporal context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    /// Current time of day
    pub time_of_day: Option<TimeOfDay>,

    /// Day of week
    pub day_of_week: Option<DayOfWeek>,

    /// Session duration (seconds)
    pub session_duration: Option<u64>,
}

/// Time of day categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeOfDay {
    /// Morning (6am-12pm)
    Morning,
    /// Afternoon (12pm-6pm)
    Afternoon,
    /// Evening (6pm-10pm)
    Evening,
    /// Night (10pm-6am)
    Night,
}

/// Day of week
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DayOfWeek {
    /// Monday
    Monday,
    /// Tuesday
    Tuesday,
    /// Wednesday
    Wednesday,
    /// Thursday
    Thursday,
    /// Friday
    Friday,
    /// Saturday
    Saturday,
    /// Sunday
    Sunday,
}

/// Multi-modal processing result
#[derive(Debug, Clone)]
pub struct MultiModalResult {
    /// Transcribed text
    pub text: String,

    /// Overall confidence score
    pub confidence: f32,

    /// Per-modality scores
    pub modality_scores: ModalityScores,

    /// Detected language
    pub language: Option<String>,

    /// Word-level timestamps
    pub word_timestamps: Option<Vec<WordTimestamp>>,

    /// Visual cues detected
    pub visual_cues: Option<Vec<VisualCue>>,

    /// Gesture cues detected
    pub gesture_cues: Option<Vec<GestureCue>>,

    /// Context influence score
    pub context_influence: f32,
}

/// Per-modality confidence scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityScores {
    /// Audio confidence
    pub audio: f32,

    /// Visual confidence (if available)
    pub visual: Option<f32>,

    /// Gesture confidence (if available)
    pub gesture: Option<f32>,

    /// Context confidence (if available)
    pub context: Option<f32>,

    /// Fused confidence
    pub fused: f32,
}

/// Word timestamp with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTimestamp {
    /// Word text
    pub word: String,

    /// Start time in seconds
    pub start: f32,

    /// End time in seconds
    pub end: f32,

    /// Confidence score
    pub confidence: f32,
}

/// Visual cue detected from video
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualCue {
    /// Type of visual cue
    pub cue_type: VisualCueType,

    /// Timestamp
    pub timestamp: f32,

    /// Confidence
    pub confidence: f32,

    /// Additional information
    pub metadata: Option<String>,
}

/// Types of visual cues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualCueType {
    /// Lip movement detected
    LipMovement,
    /// Facial expression
    FacialExpression,
    /// Head movement
    HeadMovement,
    /// Eye contact
    EyeContact,
    /// Mouth opening
    MouthOpening,
}

/// Gesture cue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GestureCue {
    /// Type of gesture
    pub gesture_type: GestureType,

    /// Timestamp
    pub timestamp: f32,

    /// Confidence
    pub confidence: f32,

    /// Spatial information
    pub spatial_info: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_config_default() {
        let config = MultiModalConfig::default();
        assert!(config.enable_audio_visual);
        assert!(config.enable_context);
        assert_eq!(config.fusion_strategy, FusionStrategy::LateFusion);
    }

    #[test]
    fn test_visual_config_default() {
        let config = VisualConfig::default();
        assert!(config.enable_lip_reading);
        assert_eq!(config.target_fps, 25);
        assert_eq!(config.face_detection_threshold, 0.8);
    }

    #[test]
    fn test_gesture_config_default() {
        let config = GestureConfig::default();
        assert!(config.enable_hand_gestures);
        assert_eq!(config.confidence_threshold, 0.7);
        assert_eq!(config.smoothing_window, 5);
    }

    #[test]
    fn test_context_config_default() {
        let config = ContextConfig::default();
        assert!(config.enable_environment);
        assert!(config.enable_conversation_history);
        assert_eq!(config.max_history_length, 10);
        assert_eq!(config.context_decay, 0.9);
    }

    #[test]
    fn test_keypoint_creation() {
        let kp = Keypoint {
            x: 100.0,
            y: 200.0,
            z: Some(50.0),
            confidence: 0.95,
        };
        assert_eq!(kp.x, 100.0);
        assert_eq!(kp.y, 200.0);
        assert_eq!(kp.z, Some(50.0));
        assert_eq!(kp.confidence, 0.95);
    }

    #[test]
    fn test_gesture_types() {
        let gestures = [
            GestureType::Pointing,
            GestureType::Wave,
            GestureType::ThumbsUp,
            GestureType::OkSign,
        ];
        assert_eq!(gestures.len(), 4);
        assert!(gestures.contains(&GestureType::Pointing));
    }

    #[test]
    fn test_location_types() {
        let locations = [
            LocationType::Indoor,
            LocationType::Outdoor,
            LocationType::Office,
            LocationType::Home,
        ];
        assert_eq!(locations.len(), 4);
        assert!(locations.contains(&LocationType::Office));
    }
}
