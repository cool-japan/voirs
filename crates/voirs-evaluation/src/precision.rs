/// Numerical precision utilities for improved calculation accuracy
///
/// This module provides high-precision mathematical operations and utilities
/// to address numerical precision issues in metric calculations.
use std::ops::{Add, Sub};

/// Implements Kahan summation algorithm for compensated summation
/// This reduces numerical error in the total when summing a sequence
/// of finite-precision floating-point numbers
#[derive(Debug, Clone)]
pub struct KahanSum {
    sum: f64,
    compensation: f64,
}

impl KahanSum {
    /// Create a new Kahan summation accumulator
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            compensation: 0.0,
        }
    }

    /// Add a value using compensated summation
    pub fn add(&mut self, value: f64) {
        // Compensate for low-order bits lost in previous operations
        let y = value - self.compensation;
        // Add to high-order sum
        let t = self.sum + y;
        // Subtract the high-order sum to recover low-order bits
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }

    /// Get the current sum
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Reset the accumulator
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.compensation = 0.0;
    }
}

impl Default for KahanSum {
    fn default() -> Self {
        Self::new()
    }
}

/// High-precision Euclidean distance calculation
/// Uses double precision internally and Kahan summation for better accuracy
pub fn precise_euclidean_distance(vec1: &[f32], vec2: &[f32]) -> f64 {
    assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length");

    let mut kahan = KahanSum::new();

    for (&a, &b) in vec1.iter().zip(vec2.iter()) {
        let diff = a as f64 - b as f64;
        kahan.add(diff * diff);
    }

    kahan.sum().max(0.0).sqrt()
}

/// High-precision mean calculation using Kahan summation
pub fn precise_mean(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut kahan = KahanSum::new();
    for &value in values {
        kahan.add(value as f64);
    }

    kahan.sum() / values.len() as f64
}

/// High-precision variance calculation using two-pass algorithm
pub fn precise_variance(values: &[f32]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }

    // First pass: calculate mean
    let mean = precise_mean(values);

    // Second pass: calculate variance
    let mut kahan = KahanSum::new();
    for &value in values {
        let diff = value as f64 - mean;
        kahan.add(diff * diff);
    }

    kahan.sum() / (values.len() - 1) as f64
}

/// High-precision standard deviation
pub fn precise_std_dev(values: &[f32]) -> f64 {
    precise_variance(values).sqrt()
}

/// High-precision percentile calculation with linear interpolation
pub fn precise_percentile(mut values: Vec<f32>, percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let rank = percentile / 100.0 * (values.len() - 1) as f64;
    let lower_index = rank.floor() as usize;
    let upper_index = rank.ceil() as usize;

    if lower_index == upper_index {
        values[lower_index] as f64
    } else {
        let weight = rank - lower_index as f64;
        let lower_value = values[lower_index] as f64;
        let upper_value = values[upper_index] as f64;
        lower_value + weight * (upper_value - lower_value)
    }
}

/// Numerically stable calculation of log-sum-exp
/// Prevents overflow and underflow in calculations involving exponentials
pub fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if max_val.is_infinite() {
        return max_val;
    }

    let mut kahan = KahanSum::new();
    for &value in values {
        kahan.add((value - max_val).exp());
    }

    max_val + kahan.sum().ln()
}

/// Numerically stable softmax calculation
pub fn stable_softmax(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }

    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let shifted: Vec<f64> = values.iter().map(|&x| x - max_val).collect();

    let sum_exp = shifted.iter().map(|&x| x.exp()).sum::<f64>();

    shifted.iter().map(|&x| x.exp() / sum_exp).collect()
}

/// High-precision correlation coefficient calculation
pub fn precise_correlation(x: &[f32], y: &[f32]) -> f64 {
    assert_eq!(x.len(), y.len(), "Arrays must have the same length");

    if x.len() <= 1 {
        return 0.0;
    }

    let mean_x = precise_mean(x);
    let mean_y = precise_mean(y);

    let mut sum_xy = KahanSum::new();
    let mut sum_x2 = KahanSum::new();
    let mut sum_y2 = KahanSum::new();

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi as f64 - mean_x;
        let dy = yi as f64 - mean_y;

        sum_xy.add(dx * dy);
        sum_x2.add(dx * dx);
        sum_y2.add(dy * dy);
    }

    let denominator = (sum_x2.sum() * sum_y2.sum()).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        sum_xy.sum() / denominator
    }
}

/// Numerical stability constants
pub mod constants {
    /// Minimum value to prevent division by zero
    pub const EPSILON: f64 = 1e-12;

    /// Maximum safe value for exponential calculations
    pub const MAX_EXP: f64 = 700.0;

    /// Minimum safe value for logarithmic calculations
    pub const MIN_LOG: f64 = 1e-15;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kahan_sum_basic() {
        let mut kahan = KahanSum::new();

        // Add some values
        kahan.add(1.0);
        kahan.add(2.0);
        kahan.add(3.0);

        assert_eq!(kahan.sum(), 6.0);
    }

    #[test]
    fn test_kahan_sum_precision() {
        let mut basic_sum = 0.0f64;
        let mut kahan = KahanSum::new();

        // Add many small values that would normally lose precision
        for _ in 0..1000000 {
            let value = 1e-10;
            basic_sum += value;
            kahan.add(value);
        }

        // Kahan sum should be more accurate
        let expected = 1000000.0 * 1e-10;
        assert!((kahan.sum() - expected).abs() < (basic_sum - expected).abs());
    }

    #[test]
    fn test_precise_euclidean_distance() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![1.0, 2.0, 3.0];

        let distance = precise_euclidean_distance(&vec1, &vec2);
        assert!((distance - 0.0).abs() < 1e-10);

        let vec3 = vec![2.0, 3.0, 4.0];
        let distance2 = precise_euclidean_distance(&vec1, &vec3);
        assert!((distance2 - (3.0f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_precise_percentile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(precise_percentile(values.clone(), 0.0), 1.0);
        assert_eq!(precise_percentile(values.clone(), 50.0), 3.0);
        assert_eq!(precise_percentile(values.clone(), 100.0), 5.0);

        // Test interpolation
        let percentile_25 = precise_percentile(values.clone(), 25.0);
        assert!((percentile_25 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp() {
        let values = vec![1.0, 2.0, 3.0];
        let result = log_sum_exp(&values);
        let expected = (1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_precise_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let correlation = precise_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 1e-10); // Perfect positive correlation

        let y_neg = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let correlation_neg = precise_correlation(&x, &y_neg);
        assert!((correlation_neg - (-1.0)).abs() < 1e-10); // Perfect negative correlation
    }
}
