use super::types::*;
use anyhow::Result;
use std::time::{Duration, SystemTime};

pub struct GazeTracker {
    gaze_history: Vec<GazePoint>,
    calibration_data: Option<CalibrationData>,
}

struct GazePoint {
    direction: Point2D,
    timestamp: SystemTime,
    confidence: f32,
}

struct CalibrationData {
    screen_bounds: BoundingBox,
    camera_position: Point2D,
}

pub struct GazeData {
    pub direction: Point2D,
    pub target: GazeTarget,
    pub attention_focus: f32,
    pub blink_rate: f32,
    pub eye_contact_duration: Duration,
    pub pupil_dilation: f32,
    pub stability: f32,
    pub quality: f32,
}

impl GazeTracker {
    pub fn new() -> Self {
        Self {
            gaze_history: Vec::new(),
            calibration_data: None,
        }
    }

    pub async fn track_gaze(&mut self, eye_landmarks: &[Point2D]) -> GazeData {
        if eye_landmarks.len() < 12 {
            return GazeData {
                direction: Point2D { x: 0.0, y: 0.0 },
                target: GazeTarget::Unknown,
                attention_focus: 0.5,
                blink_rate: 15.0,
                eye_contact_duration: Duration::from_secs(0),
                pupil_dilation: 0.5,
                stability: 0.5,
                quality: 0.3,
            };
        }

        let gaze_direction = self.calculate_gaze_direction(eye_landmarks);
        let gaze_target = self.determine_gaze_target(&gaze_direction);
        let attention_focus = self.calculate_attention_focus(&gaze_direction);
        let blink_rate = self.calculate_blink_rate(eye_landmarks);
        let eye_contact_duration = self.calculate_eye_contact_duration(&gaze_target);
        let pupil_dilation = self.calculate_pupil_dilation(eye_landmarks);
        let stability = self.calculate_gaze_stability();

        let gaze_point = GazePoint {
            direction: gaze_direction.clone(),
            timestamp: SystemTime::now(),
            confidence: 0.8,
        };

        self.gaze_history.push(gaze_point);
        if self.gaze_history.len() > 30 {
            self.gaze_history.remove(0);
        }

        GazeData {
            direction: gaze_direction,
            target: gaze_target,
            attention_focus,
            blink_rate,
            eye_contact_duration,
            pupil_dilation,
            stability,
            quality: 0.8,
        }
    }

    fn calculate_gaze_direction(&self, eye_landmarks: &[Point2D]) -> Point2D {
        let left_eye_center = self.calculate_eye_center(&eye_landmarks[0..6]);
        let right_eye_center = self.calculate_eye_center(&eye_landmarks[6..12]);

        Point2D {
            x: (left_eye_center.x + right_eye_center.x) / 2.0,
            y: (left_eye_center.y + right_eye_center.y) / 2.0,
        }
    }

    fn calculate_eye_center(&self, eye_landmarks: &[Point2D]) -> Point2D {
        let sum_x = eye_landmarks.iter().map(|p| p.x).sum::<f32>();
        let sum_y = eye_landmarks.iter().map(|p| p.y).sum::<f32>();

        Point2D {
            x: sum_x / eye_landmarks.len() as f32,
            y: sum_y / eye_landmarks.len() as f32,
        }
    }

    fn determine_gaze_target(&self, gaze_direction: &Point2D) -> GazeTarget {
        if gaze_direction.x.abs() < 50.0 && gaze_direction.y.abs() < 50.0 {
            GazeTarget::Camera
        } else if gaze_direction.x.abs() < 200.0 && gaze_direction.y.abs() < 200.0 {
            GazeTarget::Screen
        } else {
            GazeTarget::OffScreen
        }
    }

    fn calculate_attention_focus(&self, gaze_direction: &Point2D) -> f32 {
        let distance_from_center = (gaze_direction.x.powi(2) + gaze_direction.y.powi(2)).sqrt();
        (1.0 - (distance_from_center / 300.0)).max(0.0)
    }

    fn calculate_blink_rate(&self, eye_landmarks: &[Point2D]) -> f32 {
        let left_eye_openness = self.calculate_eye_openness(&eye_landmarks[0..6]);
        let right_eye_openness = self.calculate_eye_openness(&eye_landmarks[6..12]);

        let avg_openness = (left_eye_openness + right_eye_openness) / 2.0;

        if avg_openness < 0.3 {
            25.0
        } else {
            15.0
        }
    }

    fn calculate_eye_openness(&self, eye_landmarks: &[Point2D]) -> f32 {
        if eye_landmarks.len() >= 6 {
            let top = &eye_landmarks[1];
            let bottom = &eye_landmarks[4];
            let height = (top.y - bottom.y).abs();
            height / 20.0
        } else {
            0.5
        }
    }

    fn calculate_eye_contact_duration(&self, gaze_target: &GazeTarget) -> Duration {
        if *gaze_target == GazeTarget::Camera {
            Duration::from_secs(2)
        } else {
            Duration::from_millis(500)
        }
    }

    fn calculate_pupil_dilation(&self, eye_landmarks: &[Point2D]) -> f32 {
        if eye_landmarks.len() >= 12 {
            let left_eye_width = (eye_landmarks[3].x - eye_landmarks[0].x).abs();
            let right_eye_width = (eye_landmarks[9].x - eye_landmarks[6].x).abs();

            (left_eye_width + right_eye_width) / 2.0 / 30.0
        } else {
            0.5
        }
    }

    fn calculate_gaze_stability(&self) -> f32 {
        if self.gaze_history.len() < 5 {
            return 0.5;
        }

        let recent_points: Vec<&Point2D> = self
            .gaze_history
            .iter()
            .rev()
            .take(5)
            .map(|gp| &gp.direction)
            .collect();

        let center_x = recent_points.iter().map(|p| p.x).sum::<f32>() / recent_points.len() as f32;
        let center_y = recent_points.iter().map(|p| p.y).sum::<f32>() / recent_points.len() as f32;

        let variance = recent_points
            .iter()
            .map(|p| ((p.x - center_x).powi(2) + (p.y - center_y).powi(2)).sqrt())
            .sum::<f32>()
            / recent_points.len() as f32;

        (1.0 - (variance / 100.0)).max(0.0)
    }
}

pub struct EyeTrackingAnalyzer {
    gaze_tracker: GazeTracker,
}

impl EyeTrackingAnalyzer {
    pub fn new() -> Self {
        Self {
            gaze_tracker: GazeTracker::new(),
        }
    }

    pub async fn track_eye_gaze(
        &mut self,
        _frame: &VideoFrame,
        facial_landmarks: &FacialLandmarks,
    ) -> Result<EyeGazeTracking> {
        let eye_landmarks = Self::extract_eye_landmarks(facial_landmarks);
        let gaze_data = self.gaze_tracker.track_gaze(&eye_landmarks).await;

        Ok(EyeGazeTracking {
            gaze_direction: gaze_data.direction,
            gaze_target: gaze_data.target,
            attention_focus: gaze_data.attention_focus,
            blink_rate: gaze_data.blink_rate,
            eye_contact_duration: gaze_data.eye_contact_duration,
            pupil_dilation: gaze_data.pupil_dilation,
            gaze_stability: gaze_data.stability,
            tracking_quality: gaze_data.quality,
            timestamp: SystemTime::now(),
        })
    }

    fn extract_eye_landmarks(facial_landmarks: &FacialLandmarks) -> Vec<Point2D> {
        if facial_landmarks.points.len() >= 68 {
            let mut eye_points = Vec::new();
            eye_points.extend_from_slice(&facial_landmarks.points[36..48]);
            eye_points
        } else {
            Vec::new()
        }
    }
}
