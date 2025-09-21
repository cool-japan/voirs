//! Progress metrics calculation and measurement utilities
//!
//! This module provides comprehensive metrics calculation for user progress tracking,
//! including session analytics, consistency metrics, trend analysis, and system statistics.
//! It contains the core measurement and calculation functions used throughout the progress system.

use crate::traits::{FocusArea, ProgressSnapshot, UserProgress};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export analytics types for convenience
pub use crate::progress::analytics::TrendDirection;

/// Session analytics for tracking user session performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionAnalytics {
    /// Total sessions
    pub total_sessions: usize,
    /// Average score
    pub average_score: f32,
    /// Score variance
    pub score_variance: f32,
    /// Best score achieved
    pub best_score: f32,
    /// Consistency rating
    pub consistency_rating: f32,
}

/// Consistency metrics for measuring user performance stability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyMetrics {
    /// Score stability [0.0, 1.0]
    pub score_stability: f32,
    /// Improvement consistency [0.0, 1.0]
    pub improvement_consistency: f32,
    /// Session regularity [0.0, 1.0]
    pub session_regularity: f32,
}

/// Internal progress system metrics (not exposed publicly)
#[derive(Debug, Default)]
pub(crate) struct ProgressMetrics {
    /// Total users
    pub total_users: usize,
    /// Total sessions tracked
    pub total_sessions: usize,
    /// Total progress snapshots
    pub total_snapshots: usize,
}

/// Progress system statistics for public consumption
#[derive(Debug, Clone)]
pub struct ProgressSystemStats {
    /// Total users
    pub total_users: usize,
    /// Active users
    pub active_users: usize,
    /// Total sessions
    pub total_sessions: usize,
    /// Total achievements unlocked
    pub total_achievements_unlocked: usize,
    /// Average skill level
    pub average_skill_level: f32,
    /// Total snapshots
    pub total_snapshots: usize,
}

/// Metrics calculation utilities
pub struct MetricsCalculator;

impl MetricsCalculator {
    /// Calculate session analytics from progress history
    pub fn calculate_session_analytics(history: &[&ProgressSnapshot]) -> SessionAnalytics {
        let session_count = history.len();
        let total_score: f32 = history.iter().map(|s| s.overall_score).sum();
        let average_score = if session_count > 0 {
            total_score / session_count as f32
        } else {
            0.0
        };

        let scores: Vec<f32> = history.iter().map(|s| s.overall_score).collect();
        let score_variance = if scores.len() > 1 {
            let mean = if scores.is_empty() {
                0.0
            } else {
                scores.iter().sum::<f32>() / scores.len() as f32
            };
            scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32
        } else {
            0.0
        };

        SessionAnalytics {
            total_sessions: session_count,
            average_score,
            score_variance,
            best_score: scores.iter().fold(0.0f32, |a, &b| a.max(b)),
            consistency_rating: 1.0 / (1.0 + score_variance), // Higher consistency = lower variance
        }
    }

    /// Calculate consistency metrics from progress history
    pub fn calculate_consistency_metrics(history: &[&ProgressSnapshot]) -> ConsistencyMetrics {
        let scores: Vec<f32> = history.iter().map(|s| s.overall_score).collect();

        if scores.len() < 2 {
            return ConsistencyMetrics {
                score_stability: 0.0,
                improvement_consistency: 0.0,
                session_regularity: 0.0,
            };
        }

        // Score stability (inverse of coefficient of variation)
        let mean_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        };
        let score_cv = if mean_score > 0.0 && scores.len() > 1 {
            let variance =
                scores.iter().map(|s| (s - mean_score).powi(2)).sum::<f32>() / scores.len() as f32;
            let std_dev = variance.sqrt();
            std_dev / mean_score
        } else {
            0.0
        };
        let score_stability = (1.0 / (1.0 + score_cv)).min(1.0);

        // Improvement consistency (how consistent the improvement trend is)
        let improvements: Vec<f32> = scores
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect();
        let improvement_consistency = if improvements.is_empty() {
            0.0
        } else {
            let improvements_std_dev = if improvements.len() < 2 {
                0.0
            } else {
                let improvements_mean = if improvements.is_empty() {
                    0.0
                } else {
                    improvements.iter().sum::<f32>() / improvements.len() as f32
                };
                let variance = improvements
                    .iter()
                    .map(|&x| (x - improvements_mean).powi(2))
                    .sum::<f32>()
                    / (improvements.len() - 1) as f32;
                variance.sqrt()
            };
            1.0 - (improvements_std_dev / 0.1).min(1.0) // Normalized by expected variation
        };

        // Session regularity based on time gaps between sessions
        let session_regularity = if history.len() < 2 {
            0.0
        } else {
            // Calculate time gaps between consecutive sessions
            let time_gaps: Vec<i64> = history
                .windows(2)
                .map(|window| {
                    let gap = window[1].timestamp - window[0].timestamp;
                    gap.num_hours().abs()
                })
                .collect();

            if time_gaps.is_empty() {
                0.0
            } else {
                // Calculate coefficient of variation for time gaps
                let mean_gap = time_gaps.iter().sum::<i64>() as f64 / time_gaps.len() as f64;

                if mean_gap == 0.0 {
                    0.0
                } else {
                    let variance = time_gaps
                        .iter()
                        .map(|&gap| {
                            let diff = gap as f64 - mean_gap;
                            diff * diff
                        })
                        .sum::<f64>()
                        / time_gaps.len() as f64;

                    let std_dev = variance.sqrt();
                    let cv = std_dev / mean_gap;

                    // Convert to regularity score (inverse of coefficient of variation)
                    // Lower CV means more regular sessions
                    let regularity = 1.0 / (1.0 + cv);

                    // Bonus for sessions within reasonable intervals (1-7 days)
                    let ideal_gap_bonus = if mean_gap >= 24.0 && mean_gap <= 168.0 {
                        1.2 // 20% bonus for sessions between 1-7 days apart
                    } else if mean_gap < 24.0 {
                        0.9 // Slight penalty for too frequent sessions
                    } else {
                        0.8 // Penalty for sessions too far apart
                    };

                    (regularity * ideal_gap_bonus).min(1.0) as f32
                }
            }
        };

        ConsistencyMetrics {
            score_stability,
            improvement_consistency,
            session_regularity,
        }
    }

    /// Calculate overall improvement from progress history
    pub fn calculate_overall_improvement(history: &[&ProgressSnapshot]) -> f32 {
        if history.len() < 2 {
            return 0.0;
        }

        let first_score = history.first().unwrap().overall_score;
        let last_score = history.last().unwrap().overall_score;

        last_score - first_score
    }

    /// Calculate improvements per focus area
    pub fn calculate_area_improvements(history: &[&ProgressSnapshot]) -> HashMap<FocusArea, f32> {
        let mut improvements = HashMap::new();

        if history.len() < 2 {
            return improvements;
        }

        let first_snapshot = history.first().unwrap();
        let last_snapshot = history.last().unwrap();

        for (area, &last_score) in &last_snapshot.area_scores {
            if let Some(&first_score) = first_snapshot.area_scores.get(area) {
                improvements.insert(area.clone(), last_score - first_score);
            }
        }

        improvements
    }

    /// Calculate skill trends for different focus areas
    pub fn calculate_skill_trends(
        history: &[&ProgressSnapshot],
    ) -> HashMap<FocusArea, TrendDirection> {
        let mut trends = HashMap::new();

        if history.len() < 3 {
            return trends;
        }

        // Calculate trends for each focus area
        for area in [
            FocusArea::Pronunciation,
            FocusArea::Quality,
            FocusArea::Naturalness,
        ] {
            let scores: Vec<f32> = history
                .iter()
                .filter_map(|snapshot| snapshot.area_scores.get(&area))
                .copied()
                .collect();

            if scores.len() >= 3 {
                let trend = Self::calculate_linear_trend(&scores);
                trends.insert(area, trend);
            }
        }

        trends
    }

    /// Calculate linear trend direction from a series of scores
    pub fn calculate_linear_trend(scores: &[f32]) -> TrendDirection {
        if scores.len() < 2 {
            return TrendDirection::Stable;
        }

        let n = scores.len() as f32;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        };

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &score) in scores.iter().enumerate() {
            let x_diff = i as f32 - x_mean;
            numerator += x_diff * (score - y_mean);
            denominator += x_diff * x_diff;
        }

        let slope = if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        };

        if slope > 0.02 {
            TrendDirection::Improving
        } else if slope < -0.02 {
            TrendDirection::Declining
        } else {
            TrendDirection::Stable
        }
    }

    /// Calculate learning velocity (rate of improvement)
    pub fn calculate_learning_velocity(history: &[ProgressSnapshot]) -> f32 {
        if history.len() < 2 {
            return 0.0;
        }

        let scores: Vec<f32> = history.iter().map(|s| s.overall_score).collect();
        let improvements: Vec<f32> = scores
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect();

        if improvements.is_empty() {
            0.0
        } else {
            (improvements.iter().sum::<f32>() / improvements.len() as f32).max(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{FocusArea, ProgressSnapshot};
    use chrono::Utc;
    use std::collections::HashMap;

    fn create_test_snapshot(overall_score: f32, timestamp: DateTime<Utc>) -> ProgressSnapshot {
        let mut area_scores = HashMap::new();
        area_scores.insert(FocusArea::Pronunciation, overall_score);
        area_scores.insert(FocusArea::Quality, overall_score + 0.1);
        area_scores.insert(FocusArea::Naturalness, overall_score - 0.1);

        ProgressSnapshot {
            timestamp,
            overall_score,
            area_scores,
            session_count: 1,
            events: vec!["Training session completed".to_string()],
        }
    }

    #[test]
    fn test_session_analytics_calculation() {
        let now = Utc::now();
        let snapshots = vec![
            create_test_snapshot(0.7, now),
            create_test_snapshot(0.8, now),
            create_test_snapshot(0.9, now),
        ];
        let refs: Vec<&ProgressSnapshot> = snapshots.iter().collect();

        let analytics = MetricsCalculator::calculate_session_analytics(&refs);

        assert_eq!(analytics.total_sessions, 3);
        assert!((analytics.average_score - 0.8).abs() < 0.01);
        assert_eq!(analytics.best_score, 0.9);
        assert!(analytics.consistency_rating > 0.0);
    }

    #[test]
    fn test_consistency_metrics_calculation() {
        let now = Utc::now();
        let snapshots = vec![
            create_test_snapshot(0.7, now),
            create_test_snapshot(0.75, now + chrono::Duration::days(1)),
            create_test_snapshot(0.8, now + chrono::Duration::days(2)),
        ];
        let refs: Vec<&ProgressSnapshot> = snapshots.iter().collect();

        let metrics = MetricsCalculator::calculate_consistency_metrics(&refs);

        assert!(metrics.score_stability > 0.0);
        assert!(metrics.improvement_consistency > 0.0);
        assert!(metrics.session_regularity > 0.0);
    }

    #[test]
    fn test_overall_improvement_calculation() {
        let now = Utc::now();
        let snapshots = vec![
            create_test_snapshot(0.7, now),
            create_test_snapshot(0.8, now),
            create_test_snapshot(0.9, now),
        ];
        let refs: Vec<&ProgressSnapshot> = snapshots.iter().collect();

        let improvement = MetricsCalculator::calculate_overall_improvement(&refs);
        assert!((improvement - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_linear_trend_calculation() {
        let improving_scores = vec![0.5, 0.6, 0.7, 0.8];
        let stable_scores = vec![0.7, 0.71, 0.69, 0.7];
        let declining_scores = vec![0.8, 0.7, 0.6, 0.5];

        assert_eq!(
            MetricsCalculator::calculate_linear_trend(&improving_scores),
            TrendDirection::Improving
        );
        assert_eq!(
            MetricsCalculator::calculate_linear_trend(&stable_scores),
            TrendDirection::Stable
        );
        assert_eq!(
            MetricsCalculator::calculate_linear_trend(&declining_scores),
            TrendDirection::Declining
        );
    }

    #[test]
    fn test_learning_velocity_calculation() {
        let now = Utc::now();
        let snapshots = vec![
            create_test_snapshot(0.5, now),
            create_test_snapshot(0.6, now),
            create_test_snapshot(0.7, now),
            create_test_snapshot(0.8, now),
        ];

        let velocity = MetricsCalculator::calculate_learning_velocity(&snapshots);
        assert!(velocity > 0.0);
        assert!((velocity - 0.1).abs() < 0.01);
    }
}
