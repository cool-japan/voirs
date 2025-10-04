use super::eye_tracking::EyeTrackingAnalyzer;
use super::facial::FacialAnalyzer;
use super::gesture::GesturePostureAnalyzer;
use super::types::*;
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;

/// Description
pub struct MLComputerVisionAnalyzer {
    facial_analyzer: Arc<RwLock<FacialAnalyzer>>,
    gesture_posture_analyzer: Arc<RwLock<GesturePostureAnalyzer>>,
    eye_tracking_analyzer: Arc<RwLock<EyeTrackingAnalyzer>>,
    analysis_history: Arc<RwLock<Vec<MultiModalAnalysis>>>,
}

impl MLComputerVisionAnalyzer {
    /// Description
    pub fn new() -> Self {
        Self {
            facial_analyzer: Arc::new(RwLock::new(FacialAnalyzer::new())),
            gesture_posture_analyzer: Arc::new(RwLock::new(GesturePostureAnalyzer::new())),
            eye_tracking_analyzer: Arc::new(RwLock::new(EyeTrackingAnalyzer::new())),
            analysis_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Description
    pub async fn perform_multimodal_analysis(
        &self,
        frame: &VideoFrame,
        landmarks: &CombinedLandmarks,
    ) -> Result<MultiModalAnalysis> {
        let lip_movement = self.analyze_lip_movement(frame, &landmarks.facial).await?;
        let facial_expression = self
            .recognize_facial_expression(frame, &landmarks.facial)
            .await?;
        let gesture_pattern = self
            .analyze_gesture_pattern(frame, &landmarks.hands)
            .await?;
        let posture = self.assess_posture(frame, &landmarks.body).await?;
        let eye_gaze = self.track_eye_gaze(frame, &landmarks.facial).await?;

        let coordination_score = self
            .calculate_coordination_score(
                &lip_movement,
                &facial_expression,
                &gesture_pattern,
                &posture,
                &eye_gaze,
            )
            .await;

        let overall_engagement = self
            .calculate_engagement_score(&facial_expression, &posture, &eye_gaze)
            .await;

        let communication_effectiveness = self
            .calculate_communication_effectiveness(
                &lip_movement,
                &facial_expression,
                &gesture_pattern,
                &posture,
            )
            .await;

        let improvement_suggestions = self
            .generate_improvement_suggestions(
                &lip_movement,
                &facial_expression,
                &posture,
                &eye_gaze,
            )
            .await;

        let analysis = MultiModalAnalysis {
            lip_movement,
            facial_expression,
            gesture_pattern,
            posture,
            eye_gaze,
            coordination_score,
            overall_engagement,
            communication_effectiveness,
            improvement_suggestions,
            timestamp: SystemTime::now(),
        };

        let mut history = self.analysis_history.write().await;
        history.push(analysis.clone());
        if history.len() > 100 {
            history.remove(0);
        }

        Ok(analysis)
    }

    async fn calculate_coordination_score(
        &self,
        lip_movement: &LipMovementAnalysis,
        expression: &FacialExpression,
        gesture: &Option<GesturePattern>,
        posture: &PostureAnalysis,
        gaze: &EyeGazeTracking,
    ) -> f32 {
        let mut coordination_factors = Vec::new();

        coordination_factors.push(lip_movement.synchronization_score * 0.3);
        coordination_factors.push(expression.confidence * 0.2);

        if let Some(gesture_pattern) = gesture {
            coordination_factors.push(gesture_pattern.gesture_quality * 0.2);
        } else {
            coordination_factors.push(0.5);
        }

        coordination_factors.push(posture.overall_posture_score * 0.2);
        coordination_factors.push(gaze.tracking_quality * 0.1);

        coordination_factors.iter().sum::<f32>() / coordination_factors.len() as f32
    }

    async fn calculate_engagement_score(
        &self,
        expression: &FacialExpression,
        posture: &PostureAnalysis,
        gaze: &EyeGazeTracking,
    ) -> f32 {
        let expression_engagement = match expression.expression_type {
            ExpressionType::Happy | ExpressionType::Concentrated | ExpressionType::Confident => 0.8,
            ExpressionType::Surprised => 0.7,
            ExpressionType::Neutral => 0.5,
            ExpressionType::Confused => 0.4,
            ExpressionType::Frustrated => 0.3,
            _ => 0.2,
        };

        let posture_engagement = posture.engagement_level;
        let gaze_engagement = gaze.attention_focus;

        (expression_engagement * 0.4 + posture_engagement * 0.3 + gaze_engagement * 0.3).min(1.0)
    }

    async fn calculate_communication_effectiveness(
        &self,
        lip_movement: &LipMovementAnalysis,
        expression: &FacialExpression,
        gesture: &Option<GesturePattern>,
        posture: &PostureAnalysis,
    ) -> f32 {
        let articulation_score = lip_movement.articulation_clarity;
        let expression_clarity = expression.confidence;
        let posture_confidence = posture.confidence_posture;

        let gesture_support = if let Some(gesture_pattern) = gesture {
            gesture_pattern.gesture_quality
        } else {
            0.6
        };

        (articulation_score * 0.4
            + expression_clarity * 0.25
            + posture_confidence * 0.25
            + gesture_support * 0.1)
            .min(1.0)
    }

    async fn generate_improvement_suggestions(
        &self,
        lip_movement: &LipMovementAnalysis,
        expression: &FacialExpression,
        posture: &PostureAnalysis,
        gaze: &EyeGazeTracking,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        if lip_movement.articulation_clarity < 0.7 {
            suggestions.push("Focus on clearer lip movements and mouth opening".to_string());
        }

        if lip_movement.synchronization_score < 0.6 {
            suggestions.push("Work on synchronizing lip movements with speech".to_string());
        }

        if expression.confidence < 0.6 {
            suggestions.push("Try to be more expressive to convey emotions clearly".to_string());
        }

        if posture.overall_posture_score < 0.6 {
            suggestions.push(
                "Improve posture by keeping your back straight and shoulders relaxed".to_string(),
            );
        }

        if posture.confidence_posture < 0.5 {
            suggestions.push("Stand or sit more confidently to project authority".to_string());
        }

        if gaze.attention_focus < 0.6 {
            suggestions.push("Maintain better eye contact to show engagement".to_string());
        }

        if gaze.gaze_stability < 0.5 {
            suggestions
                .push("Keep your gaze more stable and avoid looking around too much".to_string());
        }

        if suggestions.is_empty() {
            suggestions.push("Excellent performance! Keep up the good work".to_string());
        }

        suggestions
    }
}

#[async_trait]
impl ComputerVisionAnalyzer for MLComputerVisionAnalyzer {
    async fn analyze_lip_movement(
        &self,
        frame: &VideoFrame,
        facial_landmarks: &FacialLandmarks,
    ) -> Result<LipMovementAnalysis> {
        let mut analyzer = self.facial_analyzer.write().await;
        analyzer.analyze_lip_movement(frame, facial_landmarks).await
    }

    async fn recognize_facial_expression(
        &self,
        frame: &VideoFrame,
        facial_landmarks: &FacialLandmarks,
    ) -> Result<FacialExpression> {
        let analyzer = self.facial_analyzer.read().await;
        analyzer
            .recognize_facial_expression(frame, facial_landmarks)
            .await
    }

    async fn analyze_gesture_pattern(
        &self,
        frame: &VideoFrame,
        hand_landmarks: &[HandPosition],
    ) -> Result<Option<GesturePattern>> {
        let analyzer = self.gesture_posture_analyzer.read().await;
        analyzer
            .analyze_gesture_pattern(frame, hand_landmarks)
            .await
    }

    async fn assess_posture(
        &self,
        frame: &VideoFrame,
        body_landmarks: &[Point2D],
    ) -> Result<PostureAnalysis> {
        let mut analyzer = self.gesture_posture_analyzer.write().await;
        analyzer.assess_posture(frame, body_landmarks).await
    }

    async fn track_eye_gaze(
        &self,
        frame: &VideoFrame,
        facial_landmarks: &FacialLandmarks,
    ) -> Result<EyeGazeTracking> {
        let mut analyzer = self.eye_tracking_analyzer.write().await;
        analyzer.track_eye_gaze(frame, facial_landmarks).await
    }
}

/// Description
pub struct SimpleLandmarkDetector {
    detection_confidence: f32,
}

impl SimpleLandmarkDetector {
    /// Description
    pub fn new() -> Self {
        Self {
            detection_confidence: 0.8,
        }
    }
}

#[async_trait]
impl LandmarkDetector for SimpleLandmarkDetector {
    async fn detect_facial_landmarks(&self, frame: &VideoFrame) -> Result<FacialLandmarks> {
        let landmarks = FacialLandmarks {
            points: (0..68)
                .map(|i| Point2D {
                    x: (i as f32 * 10.0) % frame.width as f32,
                    y: (i as f32 * 5.0) % frame.height as f32,
                })
                .collect(),
            confidence: self.detection_confidence,
            face_box: BoundingBox {
                x: frame.width as f32 * 0.2,
                y: frame.height as f32 * 0.2,
                width: frame.width as f32 * 0.6,
                height: frame.height as f32 * 0.6,
            },
            timestamp: SystemTime::now(),
        };

        Ok(landmarks)
    }

    async fn detect_hand_landmarks(&self, frame: &VideoFrame) -> Result<Vec<HandPosition>> {
        let hand_landmarks = vec![HandPosition {
            landmarks: (0..21)
                .map(|i| Point2D {
                    x: (i as f32 * 15.0) % frame.width as f32,
                    y: (i as f32 * 10.0) % frame.height as f32,
                })
                .collect(),
            palm_center: Point2D {
                x: frame.width as f32 * 0.5,
                y: frame.height as f32 * 0.5,
            },
            finger_positions: FingerPositions {
                thumb: vec![Point2D { x: 100.0, y: 100.0 }],
                index: vec![Point2D { x: 120.0, y: 80.0 }],
                middle: vec![Point2D { x: 130.0, y: 70.0 }],
                ring: vec![Point2D { x: 125.0, y: 80.0 }],
                pinky: vec![Point2D { x: 115.0, y: 90.0 }],
            },
            hand_orientation: 0.0,
            confidence: self.detection_confidence,
        }];

        Ok(hand_landmarks)
    }

    async fn detect_body_landmarks(&self, frame: &VideoFrame) -> Result<Vec<Point2D>> {
        let body_landmarks = (0..25)
            .map(|i| Point2D {
                x: (i as f32 * 20.0) % frame.width as f32,
                y: (i as f32 * 15.0) % frame.height as f32,
            })
            .collect();

        Ok(body_landmarks)
    }
}

/// Description
pub struct ComputerVisionSystem {
    analyzer: Arc<dyn ComputerVisionAnalyzer>,
    landmark_detector: Arc<dyn LandmarkDetector>,
    analysis_history: Arc<RwLock<Vec<MultiModalAnalysis>>>,
}

impl ComputerVisionSystem {
    /// Description
    pub fn new() -> Self {
        Self {
            analyzer: Arc::new(MLComputerVisionAnalyzer::new()),
            landmark_detector: Arc::new(SimpleLandmarkDetector::new()),
            analysis_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    async fn calculate_coordination_score(
        &self,
        lip_movement: &LipMovementAnalysis,
        expression: &FacialExpression,
        gesture: &Option<GesturePattern>,
        posture: &PostureAnalysis,
        gaze: &EyeGazeTracking,
    ) -> f32 {
        let mut coordination_factors = Vec::new();

        coordination_factors.push(lip_movement.synchronization_score * 0.3);
        coordination_factors.push(expression.confidence * 0.2);

        if let Some(gesture_pattern) = gesture {
            coordination_factors.push(gesture_pattern.gesture_quality * 0.2);
        } else {
            coordination_factors.push(0.5);
        }

        coordination_factors.push(posture.overall_posture_score * 0.2);
        coordination_factors.push(gaze.tracking_quality * 0.1);

        coordination_factors.iter().sum::<f32>() / coordination_factors.len() as f32
    }

    async fn calculate_engagement_score(
        &self,
        expression: &FacialExpression,
        posture: &PostureAnalysis,
        gaze: &EyeGazeTracking,
    ) -> f32 {
        let expression_engagement = match expression.expression_type {
            ExpressionType::Happy | ExpressionType::Concentrated | ExpressionType::Confident => 0.8,
            ExpressionType::Surprised => 0.7,
            ExpressionType::Neutral => 0.5,
            ExpressionType::Confused => 0.4,
            ExpressionType::Frustrated => 0.3,
            _ => 0.2,
        };

        let posture_engagement = posture.engagement_level;
        let gaze_engagement = gaze.attention_focus;

        (expression_engagement * 0.4 + posture_engagement * 0.3 + gaze_engagement * 0.3).min(1.0)
    }

    async fn calculate_communication_effectiveness(
        &self,
        lip_movement: &LipMovementAnalysis,
        expression: &FacialExpression,
        gesture: &Option<GesturePattern>,
        posture: &PostureAnalysis,
    ) -> f32 {
        let articulation_score = lip_movement.articulation_clarity;
        let expression_clarity = expression.confidence;
        let posture_confidence = posture.confidence_posture;

        let gesture_support = if let Some(gesture_pattern) = gesture {
            gesture_pattern.gesture_quality
        } else {
            0.6
        };

        (articulation_score * 0.4
            + expression_clarity * 0.25
            + posture_confidence * 0.25
            + gesture_support * 0.1)
            .min(1.0)
    }

    async fn generate_improvement_suggestions(
        &self,
        lip_movement: &LipMovementAnalysis,
        expression: &FacialExpression,
        posture: &PostureAnalysis,
        gaze: &EyeGazeTracking,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        if lip_movement.articulation_clarity < 0.7 {
            suggestions.push("Focus on clearer lip movements and mouth opening".to_string());
        }

        if lip_movement.synchronization_score < 0.6 {
            suggestions.push("Work on synchronizing lip movements with speech".to_string());
        }

        if expression.confidence < 0.6 {
            suggestions.push("Try to be more expressive to convey emotions clearly".to_string());
        }

        if posture.overall_posture_score < 0.6 {
            suggestions.push(
                "Improve posture by keeping your back straight and shoulders relaxed".to_string(),
            );
        }

        if posture.confidence_posture < 0.5 {
            suggestions.push("Stand or sit more confidently to project authority".to_string());
        }

        if gaze.attention_focus < 0.6 {
            suggestions.push("Maintain better eye contact to show engagement".to_string());
        }

        if gaze.gaze_stability < 0.5 {
            suggestions
                .push("Keep your gaze more stable and avoid looking around too much".to_string());
        }

        if suggestions.is_empty() {
            suggestions.push("Excellent performance! Keep up the good work".to_string());
        }

        suggestions
    }

    /// Description
    pub async fn analyze_frame(&self, frame: &VideoFrame) -> Result<MultiModalAnalysis> {
        let facial_landmarks = self
            .landmark_detector
            .detect_facial_landmarks(frame)
            .await?;
        let hand_landmarks = self.landmark_detector.detect_hand_landmarks(frame).await?;
        let body_landmarks = self.landmark_detector.detect_body_landmarks(frame).await?;

        let combined_landmarks = CombinedLandmarks {
            facial: facial_landmarks,
            hands: hand_landmarks,
            body: body_landmarks,
        };

        let lip_movement = self
            .analyzer
            .analyze_lip_movement(frame, &combined_landmarks.facial)
            .await?;
        let facial_expression = self
            .analyzer
            .recognize_facial_expression(frame, &combined_landmarks.facial)
            .await?;
        let gesture_pattern = self
            .analyzer
            .analyze_gesture_pattern(frame, &combined_landmarks.hands)
            .await?;
        let posture = self
            .analyzer
            .assess_posture(frame, &combined_landmarks.body)
            .await?;
        let eye_gaze = self
            .analyzer
            .track_eye_gaze(frame, &combined_landmarks.facial)
            .await?;

        let coordination_score = self
            .calculate_coordination_score(
                &lip_movement,
                &facial_expression,
                &gesture_pattern,
                &posture,
                &eye_gaze,
            )
            .await;

        let overall_engagement = self
            .calculate_engagement_score(&facial_expression, &posture, &eye_gaze)
            .await;

        let communication_effectiveness = self
            .calculate_communication_effectiveness(
                &lip_movement,
                &facial_expression,
                &gesture_pattern,
                &posture,
            )
            .await;

        let improvement_suggestions = self
            .generate_improvement_suggestions(
                &lip_movement,
                &facial_expression,
                &posture,
                &eye_gaze,
            )
            .await;

        let analysis = MultiModalAnalysis {
            lip_movement,
            facial_expression,
            gesture_pattern,
            posture,
            eye_gaze,
            coordination_score,
            overall_engagement,
            communication_effectiveness,
            improvement_suggestions,
            timestamp: SystemTime::now(),
        };

        let mut history = self.analysis_history.write().await;
        history.push(analysis.clone());
        if history.len() > 50 {
            history.remove(0);
        }

        Ok(analysis)
    }

    /// Description
    pub async fn get_analysis_history(&self) -> Vec<MultiModalAnalysis> {
        let history = self.analysis_history.read().await;
        history.clone()
    }

    /// Description
    pub async fn get_improvement_trends(&self) -> Result<ImprovementTrends> {
        let history = self.analysis_history.read().await;

        if history.len() < 5 {
            return Ok(ImprovementTrends::default());
        }

        let recent_analyses = &history[history.len() - 5..];
        let early_analyses = &history[..5.min(history.len())];

        let recent_avg_engagement = recent_analyses
            .iter()
            .map(|a| a.overall_engagement)
            .sum::<f32>()
            / recent_analyses.len() as f32;
        let early_avg_engagement = early_analyses
            .iter()
            .map(|a| a.overall_engagement)
            .sum::<f32>()
            / early_analyses.len() as f32;

        let recent_avg_communication = recent_analyses
            .iter()
            .map(|a| a.communication_effectiveness)
            .sum::<f32>()
            / recent_analyses.len() as f32;
        let early_avg_communication = early_analyses
            .iter()
            .map(|a| a.communication_effectiveness)
            .sum::<f32>()
            / early_analyses.len() as f32;

        Ok(ImprovementTrends {
            engagement_trend: recent_avg_engagement - early_avg_engagement,
            communication_trend: recent_avg_communication - early_avg_communication,
            lip_movement_trend: recent_analyses
                .iter()
                .map(|a| a.lip_movement.articulation_clarity)
                .sum::<f32>()
                / recent_analyses.len() as f32
                - early_analyses
                    .iter()
                    .map(|a| a.lip_movement.articulation_clarity)
                    .sum::<f32>()
                    / early_analyses.len() as f32,
            posture_trend: recent_analyses
                .iter()
                .map(|a| a.posture.overall_posture_score)
                .sum::<f32>()
                / recent_analyses.len() as f32
                - early_analyses
                    .iter()
                    .map(|a| a.posture.overall_posture_score)
                    .sum::<f32>()
                    / early_analyses.len() as f32,
            total_analyses: history.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_facial_landmark_detection() {
        let detector = SimpleLandmarkDetector::new();
        let frame = VideoFrame {
            timestamp: SystemTime::now(),
            width: 640,
            height: 480,
            data: vec![0; 640 * 480 * 3],
            format: VideoFormat::RGB24,
        };

        let landmarks = detector.detect_facial_landmarks(&frame).await.unwrap();
        assert_eq!(landmarks.points.len(), 68);
        assert!(landmarks.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_lip_movement_analysis() {
        let analyzer = MLComputerVisionAnalyzer::new();
        let frame = VideoFrame {
            timestamp: SystemTime::now(),
            width: 640,
            height: 480,
            data: vec![0; 640 * 480 * 3],
            format: VideoFormat::RGB24,
        };

        let facial_landmarks = FacialLandmarks {
            points: (0..68)
                .map(|i| Point2D {
                    x: i as f32 * 5.0,
                    y: i as f32 * 3.0,
                })
                .collect(),
            confidence: 0.9,
            face_box: BoundingBox {
                x: 100.0,
                y: 100.0,
                width: 200.0,
                height: 200.0,
            },
            timestamp: SystemTime::now(),
        };

        let analysis = analyzer
            .analyze_lip_movement(&frame, &facial_landmarks)
            .await
            .unwrap();
        assert!(analysis.confidence > 0.0);
        assert!(analysis.articulation_clarity >= 0.0 && analysis.articulation_clarity <= 1.0);
    }

    #[tokio::test]
    async fn test_facial_expression_recognition() {
        let analyzer = MLComputerVisionAnalyzer::new();
        let frame = VideoFrame {
            timestamp: SystemTime::now(),
            width: 640,
            height: 480,
            data: vec![0; 640 * 480 * 3],
            format: VideoFormat::RGB24,
        };

        let facial_landmarks = FacialLandmarks {
            points: (0..68)
                .map(|i| Point2D {
                    x: i as f32 * 5.0,
                    y: i as f32 * 3.0,
                })
                .collect(),
            confidence: 0.9,
            face_box: BoundingBox {
                x: 100.0,
                y: 100.0,
                width: 200.0,
                height: 200.0,
            },
            timestamp: SystemTime::now(),
        };

        let expression = analyzer
            .recognize_facial_expression(&frame, &facial_landmarks)
            .await
            .unwrap();
        assert!(expression.confidence > 0.0);
        assert!(expression.intensity >= 0.0 && expression.intensity <= 1.0);
    }

    #[tokio::test]
    async fn test_posture_analysis() {
        let analyzer = MLComputerVisionAnalyzer::new();
        let frame = VideoFrame {
            timestamp: SystemTime::now(),
            width: 640,
            height: 480,
            data: vec![0; 640 * 480 * 3],
            format: VideoFormat::RGB24,
        };

        let body_landmarks: Vec<Point2D> = (0..25)
            .map(|i| Point2D {
                x: i as f32 * 20.0,
                y: i as f32 * 15.0,
            })
            .collect();

        let posture = analyzer
            .assess_posture(&frame, &body_landmarks)
            .await
            .unwrap();
        assert!(posture.overall_posture_score >= 0.0 && posture.overall_posture_score <= 1.0);
        assert!(posture.confidence_posture >= 0.0 && posture.confidence_posture <= 1.0);
    }

    #[tokio::test]
    async fn test_eye_gaze_tracking() {
        let analyzer = MLComputerVisionAnalyzer::new();
        let frame = VideoFrame {
            timestamp: SystemTime::now(),
            width: 640,
            height: 480,
            data: vec![0; 640 * 480 * 3],
            format: VideoFormat::RGB24,
        };

        let facial_landmarks = FacialLandmarks {
            points: (0..68)
                .map(|i| Point2D {
                    x: i as f32 * 5.0,
                    y: i as f32 * 3.0,
                })
                .collect(),
            confidence: 0.9,
            face_box: BoundingBox {
                x: 100.0,
                y: 100.0,
                width: 200.0,
                height: 200.0,
            },
            timestamp: SystemTime::now(),
        };

        let gaze = analyzer
            .track_eye_gaze(&frame, &facial_landmarks)
            .await
            .unwrap();
        assert!(gaze.tracking_quality >= 0.0 && gaze.tracking_quality <= 1.0);
        assert!(gaze.attention_focus >= 0.0 && gaze.attention_focus <= 1.0);
    }

    #[tokio::test]
    async fn test_multimodal_analysis() {
        let system = ComputerVisionSystem::new();
        let frame = VideoFrame {
            timestamp: SystemTime::now(),
            width: 640,
            height: 480,
            data: vec![0; 640 * 480 * 3],
            format: VideoFormat::RGB24,
        };

        let analysis = system.analyze_frame(&frame).await.unwrap();
        assert!(analysis.overall_engagement >= 0.0 && analysis.overall_engagement <= 1.0);
        assert!(
            analysis.communication_effectiveness >= 0.0
                && analysis.communication_effectiveness <= 1.0
        );
        assert!(analysis.coordination_score >= 0.0 && analysis.coordination_score <= 1.0);
        assert!(!analysis.improvement_suggestions.is_empty());
    }

    #[tokio::test]
    async fn test_improvement_trends() {
        let system = ComputerVisionSystem::new();
        let frame = VideoFrame {
            timestamp: SystemTime::now(),
            width: 640,
            height: 480,
            data: vec![0; 640 * 480 * 3],
            format: VideoFormat::RGB24,
        };

        for _ in 0..10 {
            let _ = system.analyze_frame(&frame).await.unwrap();
        }

        let trends = system.get_improvement_trends().await.unwrap();
        assert_eq!(trends.total_analyses, 10);
    }
}
