//! Motion prediction for sources and listeners

use crate::types::Position3D;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Duration;

/// Motion snapshot for tracking and prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionSnapshot {
    /// Position at this time
    pub position: Position3D,
    /// Velocity at this time
    pub velocity: Position3D,
    /// Acceleration at this time
    pub acceleration: Position3D,
    /// Timestamp
    pub timestamp: f64,
}

/// Motion predictor for anticipating source movement
#[derive(Debug, Clone)]
pub struct MotionPredictor {
    /// Prediction time horizon (seconds)
    prediction_horizon: f32,
    /// Minimum samples needed for prediction
    min_samples: usize,
    /// Maximum history size
    max_history_size: usize,
}

impl MotionPredictor {
    /// Create new motion predictor
    pub fn new() -> Self {
        Self {
            prediction_horizon: 0.1, // 100ms prediction
            min_samples: 3,
            max_history_size: 50,
        }
    }

    /// Predict position based on motion history
    pub fn predict_position(
        &self,
        history: &VecDeque<MotionSnapshot>,
        prediction_time: Duration,
    ) -> Option<Position3D> {
        if history.len() < self.min_samples {
            return None;
        }

        // Get latest motion data
        let latest = history.back()?;
        let dt = prediction_time.as_secs_f32();

        // Simple kinematic prediction
        Some(Position3D::new(
            latest.position.x + latest.velocity.x * dt + 0.5 * latest.acceleration.x * dt * dt,
            latest.position.y + latest.velocity.y * dt + 0.5 * latest.acceleration.y * dt * dt,
            latest.position.z + latest.velocity.z * dt + 0.5 * latest.acceleration.z * dt * dt,
        ))
    }

    /// Set prediction horizon
    pub fn set_prediction_horizon(&mut self, horizon: f32) {
        self.prediction_horizon = horizon;
    }

    /// Get prediction horizon
    pub fn prediction_horizon(&self) -> f32 {
        self.prediction_horizon
    }
}

impl Default for MotionPredictor {
    fn default() -> Self {
        Self::new()
    }
}
