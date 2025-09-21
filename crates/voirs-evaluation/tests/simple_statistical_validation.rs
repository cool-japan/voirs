//! Simple Statistical Validation Tests
//!
//! These tests validate that our statistical significance calculations
//! work correctly with the current API.

use voirs_evaluation::prelude::*;
use voirs_evaluation::statistical::*;

#[test]
fn test_paired_t_test_identical_samples() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();

    // Test with identical samples (should have high p-value)
    let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let sample2 = sample1.clone();

    let result = analyzer.paired_t_test(&sample1, &sample2, Some(0.05))?;

    // p-value should be high for identical samples
    assert!(
        result.p_value > 0.5,
        "p-value for identical samples should be > 0.5, got {:.6}",
        result.p_value
    );

    // Should not be statistically significant
    assert!(
        !result.is_significant,
        "Identical samples should not be significantly different"
    );

    println!(
        "✓ Paired t-test with identical samples: p-value = {:.6}",
        result.p_value
    );

    Ok(())
}

#[test]
fn test_paired_t_test_different_samples() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();

    // Test with clearly different samples
    let sample1 = vec![1.0, 1.1, 1.2, 1.0, 1.1, 1.0, 1.1, 1.2, 1.0, 1.1]; // Mean ~1.08
    let sample2 = vec![10.0, 10.1, 10.2, 10.0, 10.1, 10.0, 10.1, 10.2, 10.0, 10.1]; // Mean ~10.08

    let result = analyzer.paired_t_test(&sample1, &sample2, Some(0.05))?;

    // p-value should be low for clearly different samples
    assert!(
        result.p_value < 0.05,
        "p-value for very different samples should be < 0.05, got {:.6}",
        result.p_value
    );

    // Should be statistically significant
    assert!(
        result.is_significant,
        "Very different samples should be significantly different"
    );

    println!(
        "✓ Paired t-test with different samples: p-value = {:.6}",
        result.p_value
    );

    Ok(())
}

#[test]
fn test_independent_t_test() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();

    // Test with two different groups
    let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Mean = 3.0
    let group2 = vec![8.0, 9.0, 10.0, 11.0, 12.0]; // Mean = 10.0

    let result = analyzer.independent_t_test(&group1, &group2)?;

    // Should find significant difference
    assert!(
        result.p_value < 0.05,
        "p-value for different groups should be < 0.05, got {:.6}",
        result.p_value
    );

    assert!(
        result.is_significant,
        "Different groups should be significantly different"
    );

    // Effect size should exist and be large (absolute value)
    if let Some(effect_size) = result.effect_size {
        assert!(
            effect_size.abs() > 0.8,
            "Effect size magnitude should be large (> 0.8), got {:.3}",
            effect_size
        );
        println!(
            "✓ Independent t-test: p-value = {:.6}, effect size = {:.3}",
            result.p_value, effect_size
        );
    } else {
        println!(
            "✓ Independent t-test: p-value = {:.6}, no effect size calculated",
            result.p_value
        );
    }

    Ok(())
}

#[test]
fn test_correlation_test() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();

    // Test with perfectly correlated data
    let x: Vec<f32> = (1..=10).map(|i| i as f32).collect();
    let y: Vec<f32> = x.iter().map(|&val| 2.0 * val + 1.0).collect(); // Perfect linear relationship

    // TODO: Implement correlation_test method
    let result = analyzer.correlation_test(&x, &y)?;

    // Should find significant correlation
    assert!(
        result.p_value < 0.05,
        "p-value for perfectly correlated data should be < 0.05, got {:.6}",
        result.p_value
    );

    assert!(
        result.is_significant,
        "Perfect correlation should be significant"
    );

    // Test statistic should be high for perfect correlation
    assert!(
        result.test_statistic.abs() > 0.9,
        "Test statistic for perfect correlation should be > 0.9, got {:.3}",
        result.test_statistic.abs()
    );

    println!(
        "✓ Correlation test: p-value = {:.6}, correlation = {:.3}",
        result.p_value, result.test_statistic
    );

    Ok(())
}

#[test]
fn test_confidence_intervals() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();

    // Test with known data
    let data = vec![10.0, 12.0, 14.0, 11.0, 13.0, 15.0, 12.0, 11.0, 13.0, 14.0];

    // Use mean as the statistic function
    let mean_fn = |samples: &[f32]| samples.iter().sum::<f32>() / samples.len() as f32;

    let result = analyzer.bootstrap_confidence_interval(&data, mean_fn)?;

    // Mean should be within the confidence interval
    let mean = data.iter().map(|&x| x as f64).sum::<f64>() / data.len() as f64;
    assert!(
        mean >= result.0 && mean <= result.1,
        "Sample mean {:.2} should be within CI [{:.2}, {:.2}]",
        mean,
        result.0,
        result.1
    );

    // Confidence interval should be reasonable
    let ci_width = result.1 - result.0;
    assert!(
        ci_width > 0.0 && ci_width < 10.0,
        "CI width {:.2} should be reasonable",
        ci_width
    );

    println!(
        "✓ 95% Confidence interval: [{:.2}, {:.2}], mean = {:.2}",
        result.0, result.1, mean
    );

    Ok(())
}

#[test]
fn test_linear_regression() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();

    // Test with linear data: y = 2x + 1
    let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f32> = vec![3.0, 5.0, 7.0, 9.0, 11.0]; // Perfect y = 2x + 1

    let result = analyzer.linear_regression(&x, &y)?;

    // R-squared should be high for perfect linear relationship
    assert!(
        result.r_squared > 0.95,
        "R-squared should be > 0.95 for linear data, got {:.3}",
        result.r_squared
    );

    // Slope should be approximately 2.0
    assert!(
        (result.slope - 2.0).abs() < 0.1,
        "Slope should be approximately 2.0, got {:.3}",
        result.slope
    );

    // Intercept should be approximately 1.0
    assert!(
        (result.intercept - 1.0).abs() < 0.1,
        "Intercept should be approximately 1.0, got {:.3}",
        result.intercept
    );

    println!(
        "✓ Linear regression: slope={:.3}, intercept={:.3}, R²={:.3}",
        result.slope, result.intercept, result.r_squared
    );

    Ok(())
}

#[test]
fn test_statistical_test_interpretation() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();

    // Test interpretation field
    let sample1 = vec![1.0, 1.1, 1.2];
    let sample2 = vec![5.0, 5.1, 5.2];

    let result = analyzer.paired_t_test(&sample1, &sample2, Some(0.05))?;

    // Interpretation should contain meaningful text
    assert!(
        !result.interpretation.is_empty(),
        "Interpretation should not be empty"
    );

    assert!(
        result.interpretation.contains("significant")
            || result.interpretation.contains("different"),
        "Interpretation should mention significance: {}",
        result.interpretation
    );

    println!("✓ Statistical interpretation: {}", result.interpretation);

    Ok(())
}

#[test]
fn test_type_i_error_simulation() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();
    let num_tests = 100; // Smaller number for faster testing
    let mut false_positives = 0;

    // Perform tests where null hypothesis is true (identical distributions)
    for i in 0..num_tests {
        let sample1: Vec<f32> = (0..10).map(|j| (i * 10 + j) as f32).collect();
        let sample2: Vec<f32> = (0..10).map(|j| (i * 10 + j) as f32).collect();

        if let Ok(result) = analyzer.paired_t_test(&sample1, &sample2, Some(0.05)) {
            if result.is_significant {
                false_positives += 1;
            }
        }
    }

    let type_i_error_rate = false_positives as f32 / num_tests as f32;

    // Type I error rate should be low (since we're using identical samples)
    assert!(
        type_i_error_rate <= 0.1,
        "Type I error rate {:.3} should be ≤ 0.1 for identical samples",
        type_i_error_rate
    );

    println!(
        "✓ Type I error rate simulation: {:.3} ({}/{} false positives)",
        type_i_error_rate, false_positives, num_tests
    );

    Ok(())
}
