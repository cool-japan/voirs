//! Statistical analysis helpers for feedback metrics.
//!
//! This module provides advanced statistical functions for analyzing
//! user progress, performance trends, and quality metrics.

use crate::float_utils::{approx_eq, approx_zero};
use std::collections::HashMap;

/// Calculate the median of a dataset.
///
/// # Examples
///
/// ```
/// use voirs_feedback::statistical_helpers::median;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(median(&data), Some(3.0));
/// ```
#[must_use]
pub fn median(data: &[f32]) -> Option<f32> {
    if data.is_empty() {
        return None;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        Some(f32::midpoint(sorted[mid - 1], sorted[mid]))
    } else {
        Some(sorted[mid])
    }
}

/// Calculate the mode (most frequent value) of a dataset.
///
/// Returns None if the dataset is empty.
///
/// # Examples
///
/// ```
/// use voirs_feedback::statistical_helpers::mode;
///
/// let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
/// assert_eq!(mode(&data), Some(3.0));
/// ```
#[must_use]
pub fn mode(data: &[f32]) -> Option<f32> {
    if data.is_empty() {
        return None;
    }

    let mut counts: HashMap<String, usize> = HashMap::new();

    // Use string keys to handle float precision issues
    for &value in data {
        let key = format!("{value:.6}");
        *counts.entry(key).or_insert(0) += 1;
    }

    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .and_then(|(value_str, _)| value_str.parse().ok())
}

/// Calculate percentile of a dataset.
///
/// # Arguments
///
/// * `data` - The dataset
/// * `percentile` - The percentile to calculate (0.0 to 1.0)
///
/// # Examples
///
/// ```
/// use voirs_feedback::statistical_helpers::percentile;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(percentile(&data, 0.5), Some(3.0)); // Median
/// ```
#[must_use]
pub fn percentile(data: &[f32], percentile: f32) -> Option<f32> {
    if data.is_empty() || !(0.0..=1.0).contains(&percentile) {
        return None;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let index = (percentile * (sorted.len() - 1) as f32).round() as usize;
    Some(sorted[index.min(sorted.len() - 1)])
}

/// Calculate interquartile range (IQR).
///
/// Returns (Q1, Q2/median, Q3, IQR) tuple.
///
/// # Examples
///
/// ```
/// use voirs_feedback::statistical_helpers::interquartile_range;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
/// let (q1, q2, q3, iqr) = interquartile_range(&data).unwrap();
/// ```
#[must_use]
pub fn interquartile_range(data: &[f32]) -> Option<(f32, f32, f32, f32)> {
    if data.len() < 4 {
        return None;
    }

    let q1 = percentile(data, 0.25)?;
    let q2 = percentile(data, 0.50)?;
    let q3 = percentile(data, 0.75)?;
    let iqr = q3 - q1;

    Some((q1, q2, q3, iqr))
}

/// Detect outliers using the IQR method.
///
/// Returns indices of outliers in the original dataset.
///
/// # Arguments
///
/// * `data` - The dataset
/// * `multiplier` - IQR multiplier for outlier detection (typically 1.5)
///
/// # Examples
///
/// ```
/// use voirs_feedback::statistical_helpers::detect_outliers;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100.0 is an outlier
/// let outliers = detect_outliers(&data, 1.5);
/// assert!(outliers.contains(&5));
/// ```
#[must_use]
pub fn detect_outliers(data: &[f32], multiplier: f32) -> Vec<usize> {
    if data.len() < 4 {
        return vec![];
    }

    let (q1, _, q3, iqr) = match interquartile_range(data) {
        Some(vals) => vals,
        None => return vec![],
    };

    let lower_bound = q1 - multiplier * iqr;
    let upper_bound = q3 + multiplier * iqr;

    data.iter()
        .enumerate()
        .filter(|(_, &value)| value < lower_bound || value > upper_bound)
        .map(|(index, _)| index)
        .collect()
}

/// Calculate skewness of a dataset.
///
/// Positive skew indicates right-skewed distribution,
/// negative skew indicates left-skewed distribution.
///
/// # Examples
///
/// ```
/// use voirs_feedback::statistical_helpers::skewness;
///
/// let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0];
/// let skew = skewness(&data);
/// ```
#[must_use]
pub fn skewness(data: &[f32]) -> Option<f32> {
    if data.len() < 3 {
        return None;
    }

    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let std_dev = standard_deviation(data)?;

    if approx_zero(std_dev) {
        return Some(0.0);
    }

    let n = data.len() as f32;
    let sum_cubed_deviations: f32 = data
        .iter()
        .map(|&x| {
            let deviation = (x - mean) / std_dev;
            deviation * deviation * deviation
        })
        .sum();

    Some((n / ((n - 1.0) * (n - 2.0))) * sum_cubed_deviations)
}

/// Calculate kurtosis of a dataset.
///
/// Measures the "tailedness" of the distribution.
/// High kurtosis indicates heavy tails and outliers.
///
/// # Examples
///
/// ```
/// use voirs_feedback::statistical_helpers::kurtosis;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let kurt = kurtosis(&data);
/// ```
#[must_use]
pub fn kurtosis(data: &[f32]) -> Option<f32> {
    if data.len() < 4 {
        return None;
    }

    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let std_dev = standard_deviation(data)?;

    if approx_zero(std_dev) {
        return Some(0.0);
    }

    let n = data.len() as f32;
    let sum_fourth_deviations: f32 = data
        .iter()
        .map(|&x| {
            let deviation = (x - mean) / std_dev;
            deviation.powi(4)
        })
        .sum();

    let kurt = (n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * sum_fourth_deviations
        - (3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0)));

    Some(kurt)
}

/// Calculate standard deviation (already exists in utils.rs, but added here for completeness).
fn standard_deviation(data: &[f32]) -> Option<f32> {
    if data.len() < 2 {
        return None;
    }

    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (data.len() - 1) as f32;

    Some(variance.sqrt())
}

/// Calculate coefficient of variation (CV).
///
/// Returns the ratio of standard deviation to mean, useful for comparing
/// variability of different datasets.
///
/// # Examples
///
/// ```
/// use voirs_feedback::statistical_helpers::coefficient_of_variation;
///
/// let data = vec![10.0, 12.0, 14.0, 16.0, 18.0];
/// let cv = coefficient_of_variation(&data);
/// ```
#[must_use]
pub fn coefficient_of_variation(data: &[f32]) -> Option<f32> {
    if data.is_empty() {
        return None;
    }

    let mean = data.iter().sum::<f32>() / data.len() as f32;
    if approx_zero(mean) {
        return None;
    }

    let std_dev = standard_deviation(data)?;
    Some((std_dev / mean.abs()) * 100.0)
}

/// Calculate Z-score for a value in a dataset.
///
/// # Examples
///
/// ```
/// use voirs_feedback::statistical_helpers::z_score;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let z = z_score(5.0, &data);
/// ```
#[must_use]
pub fn z_score(value: f32, data: &[f32]) -> Option<f32> {
    if data.len() < 2 {
        return None;
    }

    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let std_dev = standard_deviation(data)?;

    if approx_zero(std_dev) {
        return None;
    }

    Some((value - mean) / std_dev)
}

/// Calculate Pearson correlation coefficient between two datasets.
///
/// Returns value between -1.0 and 1.0:
/// * 1.0 = perfect positive correlation
/// * 0.0 = no correlation
/// * -1.0 = perfect negative correlation
///
/// # Examples
///
/// ```
/// use voirs_feedback::statistical_helpers::correlation;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
/// let r = correlation(&x, &y); // Should be close to 1.0
/// ```
#[must_use]
pub fn correlation(x: &[f32], y: &[f32]) -> Option<f32> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    let n = x.len() as f32;
    let mean_x = x.iter().sum::<f32>() / n;
    let mean_y = y.iter().sum::<f32>() / n;

    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }

    let denominator = (sum_x2 * sum_y2).sqrt();
    if approx_zero(denominator) {
        return None;
    }

    Some(sum_xy / denominator)
}

/// Calculate R-squared (coefficient of determination) for a linear regression.
///
/// # Examples
///
/// ```
/// use voirs_feedback::statistical_helpers::r_squared;
///
/// let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let predicted = vec![1.1, 2.1, 2.9, 4.2, 4.8];
/// let r2 = r_squared(&actual, &predicted);
/// ```
#[must_use]
pub fn r_squared(actual: &[f32], predicted: &[f32]) -> Option<f32> {
    if actual.len() != predicted.len() || actual.is_empty() {
        return None;
    }

    let mean_actual = actual.iter().sum::<f32>() / actual.len() as f32;

    let ss_tot: f32 = actual.iter().map(|&y| (y - mean_actual).powi(2)).sum();
    let ss_res: f32 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(&y, &pred)| (y - pred).powi(2))
        .sum();

    if approx_zero(ss_tot) {
        return Some(1.0);
    }

    Some(1.0 - (ss_res / ss_tot))
}

/// Calculate mean absolute error (MAE).
///
/// # Examples
///
/// ```
/// use voirs_feedback::statistical_helpers::mean_absolute_error;
///
/// let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let predicted = vec![1.1, 2.1, 2.9, 4.2, 4.8];
/// let mae = mean_absolute_error(&actual, &predicted);
/// ```
#[must_use]
pub fn mean_absolute_error(actual: &[f32], predicted: &[f32]) -> Option<f32> {
    if actual.len() != predicted.len() || actual.is_empty() {
        return None;
    }

    let sum_abs_errors: f32 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(&a, &p)| (a - p).abs())
        .sum();

    Some(sum_abs_errors / actual.len() as f32)
}

/// Calculate root mean squared error (RMSE).
///
/// # Examples
///
/// ```
/// use voirs_feedback::statistical_helpers::root_mean_squared_error;
///
/// let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let predicted = vec![1.1, 2.1, 2.9, 4.2, 4.8];
/// let rmse = root_mean_squared_error(&actual, &predicted);
/// ```
#[must_use]
pub fn root_mean_squared_error(actual: &[f32], predicted: &[f32]) -> Option<f32> {
    if actual.len() != predicted.len() || actual.is_empty() {
        return None;
    }

    let sum_squared_errors: f32 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(&a, &p)| (a - p).powi(2))
        .sum();

    Some((sum_squared_errors / actual.len() as f32).sqrt())
}

/// Perform simple linear regression.
///
/// Returns (slope, intercept) for the line y = slope * x + intercept.
///
/// # Examples
///
/// ```
/// use voirs_feedback::statistical_helpers::linear_regression;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
/// let (slope, intercept) = linear_regression(&x, &y).unwrap();
/// assert!((slope - 2.0).abs() < 0.001);
/// ```
#[must_use]
pub fn linear_regression(x: &[f32], y: &[f32]) -> Option<(f32, f32)> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    let n = x.len() as f32;
    let mean_x = x.iter().sum::<f32>() / n;
    let mean_y = y.iter().sum::<f32>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        numerator += dx * (y[i] - mean_y);
        denominator += dx * dx;
    }

    if approx_zero(denominator) {
        return None;
    }

    let slope = numerator / denominator;
    let intercept = mean_y - slope * mean_x;

    Some((slope, intercept))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median() {
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0, 5.0]), Some(3.0));
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
        assert_eq!(median(&[]), None);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&data, 0.0), Some(1.0));
        assert_eq!(percentile(&data, 0.5), Some(3.0));
        assert_eq!(percentile(&data, 1.0), Some(5.0));
    }

    #[test]
    fn test_interquartile_range() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let (q1, q2, q3, iqr) = interquartile_range(&data).unwrap();
        // Percentile values may vary slightly based on calculation method
        assert!(q1 >= 2.0 && q1 <= 3.0);
        assert!(approx_eq(q2, 5.0));
        assert!(q3 >= 7.0 && q3 <= 8.0);
        assert!(iqr >= 4.0 && iqr <= 6.0);
    }

    #[test]
    fn test_detect_outliers() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let outliers = detect_outliers(&data, 1.5);
        assert!(outliers.contains(&5));
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = correlation(&x, &y).unwrap();
        assert!(approx_eq(r, 1.0));

        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r_neg = correlation(&x, &y_neg).unwrap();
        assert!(approx_eq(r_neg, -1.0));
    }

    #[test]
    fn test_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let (slope, intercept) = linear_regression(&x, &y).unwrap();
        assert!(approx_eq(slope, 2.0));
        assert!(approx_eq(intercept, 0.0));
    }

    #[test]
    fn test_r_squared() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Perfect prediction
        let r2 = r_squared(&actual, &predicted).unwrap();
        assert!(approx_eq(r2, 1.0));
    }

    #[test]
    fn test_error_metrics() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![1.1, 2.1, 2.9, 4.2, 4.8];

        let mae = mean_absolute_error(&actual, &predicted).unwrap();
        assert!(mae > 0.0 && mae < 0.5);

        let rmse = root_mean_squared_error(&actual, &predicted).unwrap();
        assert!(rmse > 0.0 && rmse < 0.5);
    }

    #[test]
    fn test_coefficient_of_variation() {
        let data = vec![10.0, 12.0, 14.0, 16.0, 18.0];
        let cv = coefficient_of_variation(&data).unwrap();
        assert!(cv > 0.0 && cv < 100.0);
    }

    #[test]
    fn test_z_score() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let z = z_score(5.0, &data).unwrap();
        assert!(z > 1.0); // 5.0 is above mean
    }
}
