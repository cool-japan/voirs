use super::types::*;
use anyhow::Result;
use std::collections::HashMap;
use std::time::SystemTime;

/// Description
pub struct LipMovementModel {
    previous_lip_state: Option<Vec<Point2D>>,
    movement_history: Vec<MovementData>,
}

#[derive(Clone)]
/// Description
pub struct MovementData {
    /// Description
    pub vertical: f32,
    /// Description
    pub horizontal: f32,
    /// Description
    pub velocity: f32,
    /// Description
    pub clarity: f32,
    /// Description
    pub sync_score: f32,
}

impl LipMovementModel {
    /// Description
    pub fn new() -> Self {
        Self {
            previous_lip_state: None,
            movement_history: Vec::new(),
        }
    }

    /// Description
    pub async fn analyze_movement(&mut self, lip_points: &[Point2D]) -> MovementData {
        let movement_data = if let Some(ref previous) = self.previous_lip_state {
            self.calculate_movement(previous, lip_points)
        } else {
            MovementData {
                vertical: 0.0,
                horizontal: 0.0,
                velocity: 0.0,
                clarity: 0.7,
                sync_score: 0.7,
            }
        };

        self.previous_lip_state = Some(lip_points.to_vec());
        self.movement_history.push(movement_data.clone());

        if self.movement_history.len() > 30 {
            self.movement_history.remove(0);
        }

        movement_data
    }

    fn calculate_movement(&self, previous: &[Point2D], current: &[Point2D]) -> MovementData {
        if previous.len() != current.len() || previous.is_empty() {
            return MovementData {
                vertical: 0.0,
                horizontal: 0.0,
                velocity: 0.0,
                clarity: 0.5,
                sync_score: 0.5,
            };
        }

        let mut vertical_movement = 0.0;
        let mut horizontal_movement = 0.0;
        let mut total_distance = 0.0;

        for (prev, curr) in previous.iter().zip(current.iter()) {
            let dx = curr.x - prev.x;
            let dy = curr.y - prev.y;

            vertical_movement += dy.abs();
            horizontal_movement += dx.abs();
            total_distance += (dx.powi(2) + dy.powi(2)).sqrt();
        }

        let point_count = previous.len() as f32;
        vertical_movement /= point_count;
        horizontal_movement /= point_count;
        let velocity = total_distance / point_count;

        let clarity = self.calculate_clarity(velocity);
        let sync_score = self.calculate_sync_score(&self.movement_history);

        MovementData {
            vertical: vertical_movement,
            horizontal: horizontal_movement,
            velocity,
            clarity,
            sync_score,
        }
    }

    fn calculate_clarity(&self, velocity: f32) -> f32 {
        let optimal_velocity = 2.0;
        let velocity_ratio = (velocity / optimal_velocity).min(2.0);

        if velocity_ratio < 0.5 {
            velocity_ratio * 2.0
        } else if velocity_ratio > 1.5 {
            2.0 - velocity_ratio
        } else {
            1.0
        }
    }

    fn calculate_sync_score(&self, history: &[MovementData]) -> f32 {
        if history.len() < 3 {
            return 0.7;
        }

        let recent_velocities: Vec<f32> =
            history.iter().rev().take(5).map(|d| d.velocity).collect();
        let mean_velocity = recent_velocities.iter().sum::<f32>() / recent_velocities.len() as f32;

        let variance = recent_velocities
            .iter()
            .map(|v| (v - mean_velocity).powi(2))
            .sum::<f32>()
            / recent_velocities.len() as f32;

        let consistency = 1.0 - (variance.sqrt() / (mean_velocity + 0.1)).min(1.0);
        consistency
    }
}

/// Description
pub struct ExpressionClassifier {
    expression_models: HashMap<ExpressionType, Vec<f32>>,
}

impl ExpressionClassifier {
    /// Description
    pub fn new() -> Self {
        Self {
            expression_models: Self::initialize_models(),
        }
    }

    fn initialize_models() -> HashMap<ExpressionType, Vec<f32>> {
        let mut models = HashMap::new();
        models.insert(ExpressionType::Happy, vec![0.1; 136]);
        models.insert(ExpressionType::Sad, vec![0.1; 136]);
        models.insert(ExpressionType::Angry, vec![0.1; 136]);
        models.insert(ExpressionType::Surprised, vec![0.1; 136]);
        models.insert(ExpressionType::Neutral, vec![0.1; 136]);
        models.insert(ExpressionType::Concentrated, vec![0.1; 136]);
        models.insert(ExpressionType::Confident, vec![0.1; 136]);
        models
    }

    /// Description
    pub async fn classify(&self, features: &[f32]) -> (ExpressionType, f32, f32) {
        let mut best_score = 0.0;
        let mut best_expression = ExpressionType::Neutral;

        for (expression_type, model) in &self.expression_models {
            let score = self.calculate_similarity(features, model);
            if score > best_score {
                best_score = score;
                best_expression = expression_type.clone();
            }
        }

        (best_expression, best_score, best_score)
    }

    /// Description
    pub async fn get_secondary_expressions(&self, features: &[f32]) -> Vec<(ExpressionType, f32)> {
        let mut scores: Vec<(ExpressionType, f32)> = self
            .expression_models
            .iter()
            .map(|(expr, model)| (expr.clone(), self.calculate_similarity(features, model)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.into_iter().skip(1).take(3).collect()
    }

    fn calculate_similarity(&self, features1: &[f32], features2: &[f32]) -> f32 {
        if features1.len() != features2.len() {
            return 0.0;
        }

        let dot_product: f32 = features1
            .iter()
            .zip(features2.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm1: f32 = features1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = features2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            (dot_product / (norm1 * norm2) + 1.0) / 2.0
        } else {
            0.0
        }
    }
}

/// Description
pub struct FacialAnalyzer {
    lip_movement_model: LipMovementModel,
    expression_classifier: ExpressionClassifier,
}

impl FacialAnalyzer {
    /// Description
    pub fn new() -> Self {
        Self {
            lip_movement_model: LipMovementModel::new(),
            expression_classifier: ExpressionClassifier::new(),
        }
    }

    /// Description
    pub async fn analyze_lip_movement(
        &mut self,
        _frame: &VideoFrame,
        facial_landmarks: &FacialLandmarks,
    ) -> Result<LipMovementAnalysis> {
        let lip_points = Self::extract_lip_landmarks(facial_landmarks);

        let lip_opening = Self::calculate_lip_opening(&lip_points);
        let lip_width = Self::calculate_lip_width(&lip_points);
        let lip_roundness = Self::calculate_lip_roundness(&lip_points);

        let movement_data = self.lip_movement_model.analyze_movement(&lip_points).await;

        Ok(LipMovementAnalysis {
            lip_opening,
            lip_width,
            lip_roundness,
            vertical_movement: movement_data.vertical,
            horizontal_movement: movement_data.horizontal,
            movement_velocity: movement_data.velocity,
            articulation_clarity: movement_data.clarity,
            synchronization_score: movement_data.sync_score,
            confidence: facial_landmarks.confidence,
            timestamp: SystemTime::now(),
        })
    }

    /// Description
    pub async fn recognize_facial_expression(
        &self,
        _frame: &VideoFrame,
        facial_landmarks: &FacialLandmarks,
    ) -> Result<FacialExpression> {
        let features = Self::extract_expression_features(facial_landmarks);

        let (expression_type, intensity, confidence) =
            self.expression_classifier.classify(&features).await;
        let secondary_expressions = self
            .expression_classifier
            .get_secondary_expressions(&features)
            .await;
        let emotion_indicators = Self::calculate_emotion_indicators(facial_landmarks);

        Ok(FacialExpression {
            expression_type,
            intensity,
            confidence,
            secondary_expressions,
            emotion_indicators,
            timestamp: SystemTime::now(),
        })
    }

    fn extract_lip_landmarks(facial_landmarks: &FacialLandmarks) -> Vec<Point2D> {
        if facial_landmarks.points.len() >= 68 {
            facial_landmarks.points[48..68].to_vec()
        } else {
            Vec::new()
        }
    }

    fn extract_expression_features(facial_landmarks: &FacialLandmarks) -> Vec<f32> {
        let mut features = Vec::new();

        if facial_landmarks.points.len() >= 68 {
            for point in &facial_landmarks.points {
                features.push(point.x);
                features.push(point.y);
            }
        }

        while features.len() < 136 {
            features.push(0.0);
        }

        features
    }

    fn calculate_lip_opening(lip_points: &[Point2D]) -> f32 {
        if lip_points.len() < 20 {
            return 0.0;
        }

        let upper_lip = &lip_points[12..16];
        let lower_lip = &lip_points[16..20];

        let upper_center_y = upper_lip.iter().map(|p| p.y).sum::<f32>() / upper_lip.len() as f32;
        let lower_center_y = lower_lip.iter().map(|p| p.y).sum::<f32>() / lower_lip.len() as f32;

        (lower_center_y - upper_center_y).abs()
    }

    fn calculate_lip_width(lip_points: &[Point2D]) -> f32 {
        if lip_points.len() < 20 {
            return 0.0;
        }

        let left_corner = &lip_points[0];
        let right_corner = &lip_points[6];

        ((right_corner.x - left_corner.x).powi(2) + (right_corner.y - left_corner.y).powi(2)).sqrt()
    }

    fn calculate_lip_roundness(lip_points: &[Point2D]) -> f32 {
        if lip_points.len() < 20 {
            return 0.0;
        }

        let center_x = lip_points.iter().map(|p| p.x).sum::<f32>() / lip_points.len() as f32;
        let center_y = lip_points.iter().map(|p| p.y).sum::<f32>() / lip_points.len() as f32;

        let distances: Vec<f32> = lip_points
            .iter()
            .map(|p| ((p.x - center_x).powi(2) + (p.y - center_y).powi(2)).sqrt())
            .collect();

        let mean_distance = distances.iter().sum::<f32>() / distances.len() as f32;
        let variance = distances
            .iter()
            .map(|d| (d - mean_distance).powi(2))
            .sum::<f32>()
            / distances.len() as f32;

        1.0 - (variance.sqrt() / mean_distance).min(1.0)
    }

    fn calculate_emotion_indicators(facial_landmarks: &FacialLandmarks) -> EmotionIndicators {
        let points = &facial_landmarks.points;

        if points.len() < 68 {
            return EmotionIndicators {
                eyebrow_position: 0.5,
                eye_openness: 0.5,
                mouth_curvature: 0.5,
                cheek_tension: 0.5,
                forehead_lines: 0.5,
                overall_tension: 0.5,
            };
        }

        let eyebrow_position = Self::calculate_eyebrow_position(points);
        let eye_openness = Self::calculate_eye_openness(points);
        let mouth_curvature = Self::calculate_mouth_curvature(points);
        let cheek_tension = Self::calculate_cheek_tension(points);
        let forehead_lines = Self::calculate_forehead_lines(points);

        let overall_tension =
            (eyebrow_position + eye_openness + mouth_curvature + cheek_tension + forehead_lines)
                / 5.0;

        EmotionIndicators {
            eyebrow_position,
            eye_openness,
            mouth_curvature,
            cheek_tension,
            forehead_lines,
            overall_tension,
        }
    }

    fn calculate_eyebrow_position(points: &[Point2D]) -> f32 {
        let left_eyebrow = &points[17..22];
        let right_eyebrow = &points[22..27];
        let left_eye = &points[36..42];
        let right_eye = &points[42..48];

        let left_eyebrow_y =
            left_eyebrow.iter().map(|p| p.y).sum::<f32>() / left_eyebrow.len() as f32;
        let right_eyebrow_y =
            right_eyebrow.iter().map(|p| p.y).sum::<f32>() / right_eyebrow.len() as f32;
        let left_eye_y = left_eye.iter().map(|p| p.y).sum::<f32>() / left_eye.len() as f32;
        let right_eye_y = right_eye.iter().map(|p| p.y).sum::<f32>() / right_eye.len() as f32;

        let left_distance = left_eye_y - left_eyebrow_y;
        let right_distance = right_eye_y - right_eyebrow_y;

        ((left_distance + right_distance) / 2.0) / 50.0
    }

    fn calculate_eye_openness(points: &[Point2D]) -> f32 {
        let left_eye = &points[36..42];
        let right_eye = &points[42..48];

        let left_height =
            (left_eye[4].y - left_eye[1].y).abs() + (left_eye[5].y - left_eye[2].y).abs();
        let right_height =
            (right_eye[4].y - right_eye[1].y).abs() + (right_eye[5].y - right_eye[2].y).abs();

        ((left_height + right_height) / 2.0) / 20.0
    }

    fn calculate_mouth_curvature(points: &[Point2D]) -> f32 {
        let mouth_corners = [&points[48], &points[54]];
        let mouth_center = &points[51];

        let left_curve = mouth_corners[0].y - mouth_center.y;
        let right_curve = mouth_corners[1].y - mouth_center.y;

        ((left_curve + right_curve) / 2.0) / 10.0 + 0.5
    }

    fn calculate_cheek_tension(points: &[Point2D]) -> f32 {
        let nose_base = &points[33];
        let left_mouth_corner = &points[48];
        let right_mouth_corner = &points[54];

        let left_cheek_distance = ((nose_base.x - left_mouth_corner.x).powi(2)
            + (nose_base.y - left_mouth_corner.y).powi(2))
        .sqrt();
        let right_cheek_distance = ((nose_base.x - right_mouth_corner.x).powi(2)
            + (nose_base.y - right_mouth_corner.y).powi(2))
        .sqrt();

        ((left_cheek_distance + right_cheek_distance) / 2.0) / 100.0
    }

    fn calculate_forehead_lines(points: &[Point2D]) -> f32 {
        let forehead_points = &points[0..17];
        let forehead_center_y =
            forehead_points.iter().map(|p| p.y).sum::<f32>() / forehead_points.len() as f32;

        let variation = forehead_points
            .iter()
            .map(|p| (p.y - forehead_center_y).abs())
            .sum::<f32>()
            / forehead_points.len() as f32;

        variation / 10.0
    }
}
