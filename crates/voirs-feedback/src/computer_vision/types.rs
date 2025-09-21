use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoFrame {
    pub timestamp: SystemTime,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub format: VideoFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VideoFormat {
    RGB24,
    BGR24,
    RGBA32,
    BGRA32,
    YUV420,
    Grayscale,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacialLandmarks {
    pub points: Vec<Point2D>,
    pub confidence: f32,
    pub face_box: BoundingBox,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LipMovementAnalysis {
    pub lip_opening: f32,
    pub lip_width: f32,
    pub lip_roundness: f32,
    pub vertical_movement: f32,
    pub horizontal_movement: f32,
    pub movement_velocity: f32,
    pub articulation_clarity: f32,
    pub synchronization_score: f32,
    pub confidence: f32,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacialExpression {
    pub expression_type: ExpressionType,
    pub intensity: f32,
    pub confidence: f32,
    pub secondary_expressions: Vec<(ExpressionType, f32)>,
    pub emotion_indicators: EmotionIndicators,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ExpressionType {
    Happy,
    Sad,
    Angry,
    Surprised,
    Fearful,
    Disgusted,
    Contemptuous,
    Neutral,
    Concentrated,
    Confused,
    Frustrated,
    Confident,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionIndicators {
    pub eyebrow_position: f32,
    pub eye_openness: f32,
    pub mouth_curvature: f32,
    pub cheek_tension: f32,
    pub forehead_lines: f32,
    pub overall_tension: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GesturePattern {
    pub gesture_type: GestureType,
    pub confidence: f32,
    pub duration: Duration,
    pub hand_positions: Vec<HandPosition>,
    pub movement_trajectory: Vec<Point2D>,
    pub velocity_profile: Vec<f32>,
    pub acceleration_pattern: Vec<f32>,
    pub gesture_quality: f32,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum GestureType {
    Pointing,
    Waving,
    ThumbsUp,
    ThumbsDown,
    OpenPalm,
    Fist,
    PeaceSign,
    OkSign,
    Stop,
    Applause,
    Gesture,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandPosition {
    pub landmarks: Vec<Point2D>,
    pub palm_center: Point2D,
    pub finger_positions: FingerPositions,
    pub hand_orientation: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerPositions {
    pub thumb: Vec<Point2D>,
    pub index: Vec<Point2D>,
    pub middle: Vec<Point2D>,
    pub ring: Vec<Point2D>,
    pub pinky: Vec<Point2D>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostureAnalysis {
    pub head_pose: HeadPose,
    pub shoulder_alignment: f32,
    pub spine_curvature: f32,
    pub body_symmetry: f32,
    pub stance_stability: f32,
    pub confidence_posture: f32,
    pub engagement_level: f32,
    pub attention_direction: f32,
    pub overall_posture_score: f32,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadPose {
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EyeGazeTracking {
    pub gaze_direction: Point2D,
    pub gaze_target: GazeTarget,
    pub attention_focus: f32,
    pub blink_rate: f32,
    pub eye_contact_duration: Duration,
    pub pupil_dilation: f32,
    pub gaze_stability: f32,
    pub tracking_quality: f32,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GazeTarget {
    Camera,
    Screen,
    OffScreen,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalAnalysis {
    pub lip_movement: LipMovementAnalysis,
    pub facial_expression: FacialExpression,
    pub gesture_pattern: Option<GesturePattern>,
    pub posture: PostureAnalysis,
    pub eye_gaze: EyeGazeTracking,
    pub coordination_score: f32,
    pub overall_engagement: f32,
    pub communication_effectiveness: f32,
    pub improvement_suggestions: Vec<String>,
    pub timestamp: SystemTime,
}

#[async_trait]
pub trait ComputerVisionAnalyzer: Send + Sync {
    async fn analyze_lip_movement(
        &self,
        frame: &VideoFrame,
        facial_landmarks: &FacialLandmarks,
    ) -> Result<LipMovementAnalysis>;

    async fn recognize_facial_expression(
        &self,
        frame: &VideoFrame,
        facial_landmarks: &FacialLandmarks,
    ) -> Result<FacialExpression>;

    async fn analyze_gesture_pattern(
        &self,
        frame: &VideoFrame,
        hand_landmarks: &[HandPosition],
    ) -> Result<Option<GesturePattern>>;

    async fn assess_posture(
        &self,
        frame: &VideoFrame,
        body_landmarks: &[Point2D],
    ) -> Result<PostureAnalysis>;

    async fn track_eye_gaze(
        &self,
        frame: &VideoFrame,
        facial_landmarks: &FacialLandmarks,
    ) -> Result<EyeGazeTracking>;
}

#[async_trait]
pub trait LandmarkDetector: Send + Sync {
    async fn detect_facial_landmarks(&self, frame: &VideoFrame) -> Result<FacialLandmarks>;
    async fn detect_hand_landmarks(&self, frame: &VideoFrame) -> Result<Vec<HandPosition>>;
    async fn detect_body_landmarks(&self, frame: &VideoFrame) -> Result<Vec<Point2D>>;
}

pub struct CombinedLandmarks {
    pub facial: FacialLandmarks,
    pub hands: Vec<HandPosition>,
    pub body: Vec<Point2D>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementTrends {
    pub engagement_trend: f32,
    pub communication_trend: f32,
    pub lip_movement_trend: f32,
    pub posture_trend: f32,
    pub total_analyses: usize,
}

impl Default for ImprovementTrends {
    fn default() -> Self {
        Self {
            engagement_trend: 0.0,
            communication_trend: 0.0,
            lip_movement_trend: 0.0,
            posture_trend: 0.0,
            total_analyses: 0,
        }
    }
}
