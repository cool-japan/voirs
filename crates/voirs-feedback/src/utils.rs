//! Utility functions for `VoiRS` Feedback system
//!
//! This module provides helper functions for common operations including:
//! - Score calculations and normalization
//! - Time and duration utilities
//! - Data validation and sanitization
//! - Format conversion helpers

use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;

/// Calculate weighted average score
///
/// # Arguments
///
/// * `scores` - Slice of scores (0.0 to 1.0)
/// * `weights` - Slice of weights (must sum to 1.0 or will be normalized)
///
/// # Returns
///
/// Weighted average score, or simple average if weights don't match
///
/// # Examples
///
/// ```
/// use voirs_feedback::utils::weighted_average;
///
/// let scores = vec![0.8, 0.9, 0.7];
/// let weights = vec![0.5, 0.3, 0.2];
/// let avg = weighted_average(&scores, &weights);
/// assert!((avg - 0.8).abs() < 0.01);
/// ```
#[must_use]
pub fn weighted_average(scores: &[f32], weights: &[f32]) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }

    if scores.len() != weights.len() {
        // Fallback to simple average if weights don't match
        return scores.iter().sum::<f32>() / scores.len() as f32;
    }

    let weight_sum: f32 = weights.iter().sum();
    if weight_sum == 0.0 {
        return scores.iter().sum::<f32>() / scores.len() as f32;
    }

    let weighted_sum: f32 = scores
        .iter()
        .zip(weights.iter())
        .map(|(score, weight)| score * weight)
        .sum();

    weighted_sum / weight_sum
}

/// Normalize scores to 0.0-1.0 range
///
/// # Arguments
///
/// * `scores` - Slice of scores to normalize
///
/// # Returns
///
/// Vector of normalized scores
///
/// # Examples
///
/// ```
/// use voirs_feedback::utils::normalize_scores;
///
/// let scores = vec![10.0, 20.0, 30.0];
/// let normalized = normalize_scores(&scores);
/// assert_eq!(normalized[0], 0.0);
/// assert_eq!(normalized[2], 1.0);
/// ```
pub fn normalize_scores(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    let min = scores.iter().copied().fold(f32::INFINITY, f32::min);
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if (max - min).abs() < f32::EPSILON {
        // All scores are the same
        return vec![0.5; scores.len()];
    }

    scores
        .iter()
        .map(|&score| (score - min) / (max - min))
        .collect()
}

/// Clamp score to valid range [0.0, 1.0]
///
/// # Arguments
///
/// * `score` - Score to clamp
///
/// # Returns
///
/// Clamped score
///
/// # Examples
///
/// ```
/// use voirs_feedback::utils::clamp_score;
///
/// assert_eq!(clamp_score(1.5), 1.0);
/// assert_eq!(clamp_score(-0.5), 0.0);
/// assert_eq!(clamp_score(0.7), 0.7);
/// ```
#[must_use]
pub fn clamp_score(score: f32) -> f32 {
    score.clamp(0.0, 1.0)
}

/// Calculate percentage from score
///
/// # Arguments
///
/// * `score` - Score (0.0 to 1.0)
///
/// # Returns
///
/// Percentage (0.0 to 100.0)
///
/// # Examples
///
/// ```
/// use voirs_feedback::utils::score_to_percentage;
///
/// assert_eq!(score_to_percentage(0.85), 85.0);
/// assert_eq!(score_to_percentage(1.0), 100.0);
/// ```
#[must_use]
pub fn score_to_percentage(score: f32) -> f32 {
    clamp_score(score) * 100.0
}

/// Calculate score from percentage
///
/// # Arguments
///
/// * `percentage` - Percentage (0.0 to 100.0)
///
/// # Returns
///
/// Score (0.0 to 1.0)
///
/// # Examples
///
/// ```
/// use voirs_feedback::utils::percentage_to_score;
///
/// assert_eq!(percentage_to_score(85.0), 0.85);
/// assert_eq!(percentage_to_score(100.0), 1.0);
/// ```
#[must_use]
pub fn percentage_to_score(percentage: f32) -> f32 {
    (percentage / 100.0).clamp(0.0, 1.0)
}

/// Calculate improvement rate between two scores
///
/// # Arguments
///
/// * `old_score` - Previous score (0.0 to 1.0)
/// * `new_score` - Current score (0.0 to 1.0)
///
/// # Returns
///
/// Improvement rate (-1.0 to 1.0, positive = improvement)
///
/// # Examples
///
/// ```
/// use voirs_feedback::utils::improvement_rate;
///
/// let rate = improvement_rate(0.7, 0.8);
/// assert!((rate - 0.142857).abs() < 0.001); // ~14.3% improvement
/// ```
#[must_use]
pub fn improvement_rate(old_score: f32, new_score: f32) -> f32 {
    if old_score == 0.0 {
        return if new_score > 0.0 { 1.0 } else { 0.0 };
    }

    ((new_score - old_score) / old_score).clamp(-1.0, 1.0)
}

/// Format duration for human-readable display
///
/// # Arguments
///
/// * `duration` - Duration to format
///
/// # Returns
///
/// Human-readable duration string
///
/// # Examples
///
/// ```
/// use voirs_feedback::utils::format_duration;
/// use std::time::Duration;
///
/// let duration = Duration::from_secs(3665);
/// assert_eq!(format_duration(&duration), "1h 1m 5s");
/// ```
#[must_use]
pub fn format_duration(duration: &std::time::Duration) -> String {
    let secs = duration.as_secs();

    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;

    if hours > 0 {
        format!("{hours}h {minutes}m {seconds}s")
    } else if minutes > 0 {
        format!("{minutes}m {seconds}s")
    } else {
        format!("{seconds}s")
    }
}

/// Calculate time since timestamp
///
/// # Arguments
///
/// * `timestamp` - Past timestamp
///
/// # Returns
///
/// Duration since timestamp
///
/// # Examples
///
/// ```
/// use voirs_feedback::utils::time_since;
/// use chrono::Utc;
///
/// let past = Utc::now() - chrono::Duration::seconds(60);
/// let duration = time_since(&past);
/// assert!(duration.as_secs() >= 60);
/// ```
#[must_use]
pub fn time_since(timestamp: &DateTime<Utc>) -> std::time::Duration {
    let now = Utc::now();
    let duration = now.signed_duration_since(*timestamp);

    std::time::Duration::from_secs(duration.num_seconds().max(0) as u64)
}

/// Check if timestamp is recent (within specified duration)
///
/// # Arguments
///
/// * `timestamp` - Timestamp to check
/// * `threshold` - Maximum age
///
/// # Returns
///
/// True if timestamp is recent
///
/// # Examples
///
/// ```
/// use voirs_feedback::utils::is_recent;
/// use chrono::Utc;
/// use std::time::Duration;
///
/// let recent = Utc::now() - chrono::Duration::seconds(30);
/// assert!(is_recent(&recent, &Duration::from_secs(60)));
/// ```
#[must_use]
pub fn is_recent(timestamp: &DateTime<Utc>, threshold: &std::time::Duration) -> bool {
    time_since(timestamp) <= *threshold
}

/// Calculate moving average
///
/// # Arguments
///
/// * `values` - Slice of values
/// * `window_size` - Window size for moving average
///
/// # Returns
///
/// Vector of moving averages
///
/// # Examples
///
/// ```
/// use voirs_feedback::utils::moving_average;
///
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let ma = moving_average(&values, 3);
/// assert_eq!(ma.len(), 3); // 5 - 3 + 1
/// ```
#[must_use]
pub fn moving_average(values: &[f32], window_size: usize) -> Vec<f32> {
    if values.len() < window_size || window_size == 0 {
        return Vec::new();
    }

    values
        .windows(window_size)
        .map(|window| window.iter().sum::<f32>() / window_size as f32)
        .collect()
}

/// Calculate exponential moving average
///
/// # Arguments
///
/// * `values` - Slice of values
/// * `alpha` - Smoothing factor (0.0 to 1.0)
///
/// # Returns
///
/// Vector of exponential moving averages
///
/// # Examples
///
/// ```
/// use voirs_feedback::utils::exponential_moving_average;
///
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let ema = exponential_moving_average(&values, 0.3);
/// assert_eq!(ema.len(), values.len());
/// ```
#[must_use]
pub fn exponential_moving_average(values: &[f32], alpha: f32) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }

    let alpha = alpha.clamp(0.0, 1.0);
    let mut ema = Vec::with_capacity(values.len());
    ema.push(values[0]);

    for &value in &values[1..] {
        let prev_ema = *ema.last().expect("collection should not be empty");
        ema.push(alpha * value + (1.0 - alpha) * prev_ema);
    }

    ema
}

/// Calculate standard deviation
///
/// # Arguments
///
/// * `values` - Slice of values
///
/// # Returns
///
/// Standard deviation
///
/// # Examples
///
/// ```
/// use voirs_feedback::utils::standard_deviation;
///
/// let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let std = standard_deviation(&values);
/// assert!((std - 2.0).abs() < 0.1);
/// ```
#[must_use]
pub fn standard_deviation(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|&value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f32>()
        / values.len() as f32;

    variance.sqrt()
}

/// Calculate confidence interval
///
/// # Arguments
///
/// * `values` - Slice of values
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95%)
///
/// # Returns
///
/// (`lower_bound`, `upper_bound`) tuple
///
/// # Examples
///
/// ```
/// use voirs_feedback::utils::confidence_interval;
///
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let (lower, upper) = confidence_interval(&values, 0.95);
/// assert!(lower < upper);
/// ```
#[must_use]
pub fn confidence_interval(values: &[f32], confidence_level: f32) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let std = standard_deviation(values);

    // Using z-score for 95% confidence (1.96)
    let z_score = if confidence_level >= 0.95 {
        1.96
    } else if confidence_level >= 0.90 {
        1.645
    } else {
        1.282 // 80%
    };

    let margin = z_score * (std / (values.len() as f32).sqrt());

    (mean - margin, mean + margin)
}

/// Sanitize user input string
///
/// # Arguments
///
/// * `input` - User input string
///
/// # Returns
///
/// Sanitized string
///
/// # Examples
///
/// ```
/// use voirs_feedback::utils::sanitize_input;
///
/// let sanitized = sanitize_input("  Hello\nWorld\t  ");
/// assert_eq!(sanitized, "Hello World");
/// ```
#[must_use]
pub fn sanitize_input(input: &str) -> String {
    input
        .chars()
        .map(|c| {
            if c.is_whitespace() {
                ' '
            } else if c.is_control() {
                ' '
            } else {
                c
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Truncate string to maximum length with ellipsis
///
/// # Arguments
///
/// * `s` - String to truncate
/// * `max_length` - Maximum length
///
/// # Returns
///
/// Truncated string
///
/// # Examples
///
/// ```
/// use voirs_feedback::utils::truncate_string;
///
/// let truncated = truncate_string("This is a long string", 10);
/// assert_eq!(truncated, "This is...");
/// ```
#[must_use]
pub fn truncate_string(s: &str, max_length: usize) -> String {
    if s.len() <= max_length {
        s.to_string()
    } else if max_length <= 3 {
        "...".to_string()
    } else {
        format!("{}...", &s[..max_length - 3])
    }
}

/// Merge two hashmaps, preferring values from the second map
///
/// # Arguments
///
/// * `a` - First hashmap
/// * `b` - Second hashmap
///
/// # Returns
///
/// Merged hashmap
///
/// # Examples
///
/// ```
/// use voirs_feedback::utils::merge_hashmaps;
/// use std::collections::HashMap;
///
/// let mut a = HashMap::new();
/// a.insert("key1", 1);
/// let mut b = HashMap::new();
/// b.insert("key2", 2);
/// let merged = merge_hashmaps(a, b);
/// assert_eq!(merged.len(), 2);
/// ```
#[must_use]
pub fn merge_hashmaps<K, V>(mut a: HashMap<K, V>, b: HashMap<K, V>) -> HashMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    for (key, value) in b {
        a.insert(key, value);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_average() {
        let scores = vec![0.8, 0.9, 0.7];
        let weights = vec![0.5, 0.3, 0.2];
        let avg = weighted_average(&scores, &weights);
        assert!((avg - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_normalize_scores() {
        let scores = vec![10.0, 20.0, 30.0];
        let normalized = normalize_scores(&scores);
        assert_eq!(normalized[0], 0.0);
        assert_eq!(normalized[2], 1.0);
        assert_eq!(normalized[1], 0.5);
    }

    #[test]
    fn test_clamp_score() {
        assert_eq!(clamp_score(1.5), 1.0);
        assert_eq!(clamp_score(-0.5), 0.0);
        assert_eq!(clamp_score(0.7), 0.7);
    }

    #[test]
    fn test_score_percentage_conversion() {
        assert_eq!(score_to_percentage(0.85), 85.0);
        assert_eq!(percentage_to_score(85.0), 0.85);
    }

    #[test]
    fn test_improvement_rate() {
        let rate = improvement_rate(0.7, 0.8);
        assert!((rate - 0.142857).abs() < 0.001);
    }

    #[test]
    fn test_format_duration() {
        let duration = std::time::Duration::from_secs(3665);
        assert_eq!(format_duration(&duration), "1h 1m 5s");

        let duration = std::time::Duration::from_secs(125);
        assert_eq!(format_duration(&duration), "2m 5s");

        let duration = std::time::Duration::from_secs(45);
        assert_eq!(format_duration(&duration), "45s");
    }

    #[test]
    fn test_moving_average() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = moving_average(&values, 3);
        assert_eq!(ma.len(), 3);
        assert_eq!(ma[0], 2.0); // (1+2+3)/3
        assert_eq!(ma[1], 3.0); // (2+3+4)/3
        assert_eq!(ma[2], 4.0); // (3+4+5)/3
    }

    #[test]
    fn test_exponential_moving_average() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = exponential_moving_average(&values, 0.3);
        assert_eq!(ema.len(), values.len());
        assert_eq!(ema[0], 1.0);
    }

    #[test]
    fn test_standard_deviation() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = standard_deviation(&values);
        assert!((std - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_sanitize_input() {
        let sanitized = sanitize_input("  Hello\nWorld\t  ");
        assert_eq!(sanitized, "Hello World");

        let sanitized = sanitize_input("Test\x00Control");
        assert_eq!(sanitized, "Test Control");
    }

    #[test]
    fn test_truncate_string() {
        let truncated = truncate_string("This is a long string", 10);
        assert_eq!(truncated, "This is...");

        let truncated = truncate_string("Short", 10);
        assert_eq!(truncated, "Short");
    }
}
