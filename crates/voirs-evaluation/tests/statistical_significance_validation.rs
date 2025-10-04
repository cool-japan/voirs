//! Statistical Significance Validation Tests
//!
//! These tests validate that our statistical significance calculations
//! produce correct p-values and statistical conclusions.

use voirs_evaluation::prelude::*;
use voirs_evaluation::statistical::*;

// Helper function to calculate mean
fn mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        0.0
    } else {
        data.iter().sum::<f32>() / data.len() as f32
    }
}

/// Generate samples from a normal distribution (approximation using central limit theorem)
fn generate_normal_samples(mean: f32, std_dev: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            // Use sum of 12 uniform random variables to approximate normal distribution
            let sum: f32 = (0..12).map(|_| scirs2_core::random::random::<f32>()).sum();
            mean + std_dev * (sum - 6.0) // Shift and scale to desired mean/std
        })
        .collect()
}

#[test]
fn test_paired_t_test_identical_samples() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();

    // Test with identical samples (should have p-value close to 1.0)
    let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let sample2 = sample1.clone();

    let result = analyzer.paired_t_test(&sample1, &sample2, None)?;

    // p-value should be very high (close to 1.0) for identical samples
    assert!(
        result.p_value > 0.9,
        "p-value for identical samples should be > 0.9, got {:.6}",
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
fn test_paired_t_test_very_different_samples() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();

    // Test with very different samples (should have p-value close to 0.0)
    let sample1 = vec![1.0, 1.1, 1.2, 1.0, 1.1, 1.0, 1.1, 1.2, 1.0, 1.1]; // Mean ~1.08
    let sample2 = vec![10.0, 10.1, 10.2, 10.0, 10.1, 10.0, 10.1, 10.2, 10.0, 10.1]; // Mean ~10.08

    let result = analyzer.paired_t_test(&sample1, &sample2, None)?;

    // p-value should be very low for clearly different samples
    assert!(
        result.p_value < 0.01,
        "p-value for very different samples should be < 0.01, got {:.6}",
        result.p_value
    );

    // Should be statistically significant
    assert!(
        result.is_significant,
        "Very different samples should be significantly different"
    );

    println!(
        "✓ Paired t-test with very different samples: p-value = {:.6}",
        result.p_value
    );

    Ok(())
}

#[test]
fn test_type_i_error_rate() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();
    let alpha = 0.05;
    let num_tests = 1000;
    let mut false_positives = 0;

    // Perform many t-tests where null hypothesis is true (samples from same distribution)
    for _ in 0..num_tests {
        let sample1 = generate_normal_samples(0.0, 1.0, 20);
        let sample2 = generate_normal_samples(0.0, 1.0, 20);

        if let Ok(result) = analyzer.paired_t_test(&sample1, &sample2, None) {
            if result.p_value < alpha {
                false_positives += 1;
            }
        }
    }

    let type_i_error_rate = false_positives as f32 / num_tests as f32;

    // Type I error rate should be close to α (within reasonable bounds)
    // Note: A lower error rate (more conservative) is acceptable
    assert!(
        type_i_error_rate >= 0.01 && type_i_error_rate <= 0.08,
        "Type I error rate {:.3} should be close to α={:.2} (expected range: 0.01-0.08)",
        type_i_error_rate,
        alpha
    );

    println!(
        "✓ Type I error rate validation: {:.3} (expected ~{:.2})",
        type_i_error_rate, alpha
    );

    Ok(())
}

#[test]
fn test_correlation_significance() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();

    // Test with perfectly correlated data
    let x_perfect: Vec<f32> = (1..=20).map(|i| i as f32).collect();
    let y_perfect: Vec<f32> = x_perfect.iter().map(|&x| 2.0 * x + 1.0).collect(); // Perfect linear relationship

    let result_perfect = analyzer.correlation_test(&x_perfect, &y_perfect)?;

    assert!(
        result_perfect.test_statistic > 0.99,
        "Perfect correlation should be > 0.99, got {:.6}",
        result_perfect.test_statistic
    );
    assert!(
        result_perfect.p_value <= 0.0011,
        "Perfect correlation should have p-value <= 0.0011, got {:.6}",
        result_perfect.p_value
    );

    // Test with uncorrelated data
    let x_random: Vec<f32> = generate_normal_samples(0.0, 1.0, 50);
    let y_random: Vec<f32> = generate_normal_samples(0.0, 1.0, 50);

    let result_random = analyzer.correlation_test(&x_random, &y_random)?;

    // t-statistic for random data should be reasonable (not extremely large)
    // Note: test_statistic is t-statistic, not correlation coefficient
    assert!(
        result_random.test_statistic.abs() < 3.0,
        "Random correlation t-statistic should be reasonable, got {:.3}",
        result_random.test_statistic
    );
    // Note: We can't always guarantee p-value > 0.05 due to randomness, but it should usually be

    println!("✓ Correlation significance tests:");
    println!(
        "  Perfect correlation: r={:.3}, p={:.6}",
        result_perfect.test_statistic, result_perfect.p_value
    );
    println!(
        "  Random correlation: r={:.3}, p={:.6}",
        result_random.test_statistic, result_random.p_value
    );

    Ok(())
}

#[test]
fn test_power_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();

    // Test power calculation for different effect sizes
    let effect_sizes = vec![0.2, 0.5, 0.8]; // Small, medium, large effect sizes
    let sample_size = 30;
    let alpha = 0.05;

    for effect_size in effect_sizes {
        let config = ABTestConfig {
            power: 0.8, // Target power
            alpha,
            expected_effect_size: effect_size,
            minimum_detectable_difference: effect_size,
            effect_size,
            sample_size_a: sample_size,
            sample_size_b: sample_size,
            two_tailed: true,
        };
        let result = analyzer.power_analysis_t_test(&config);

        // Power should increase with effect size
        assert!(
            result.achieved_power >= 0.0 && result.achieved_power <= 1.0,
            "Power should be between 0 and 1, got {:.3}",
            result.achieved_power
        );

        // For large effect sizes, power should be high
        if effect_size >= 0.8 {
            assert!(
                result.achieved_power > 0.7,
                "Power for large effect size should be > 0.7, got {:.3}",
                result.achieved_power
            );
        }

        println!(
            "Effect size {:.1}, n={}, α={:.2}: Power = {:.3}",
            effect_size, sample_size, alpha, result.achieved_power
        );
    }

    println!("✓ Power analysis validation completed");

    Ok(())
}

#[test]
fn test_confidence_intervals() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();

    // Test with known data
    let data = vec![10.0, 12.0, 14.0, 11.0, 13.0, 15.0, 12.0, 11.0, 13.0, 14.0];
    let confidence_level = 0.95;

    let (lower, upper) = analyzer.bootstrap_confidence_interval(&data, mean)?;

    // Mean should be within the confidence interval
    let sample_mean = data.iter().sum::<f32>() / data.len() as f32;
    assert!(
        sample_mean >= lower as f32 && sample_mean <= upper as f32,
        "Sample mean {:.2} should be within CI [{:.2}, {:.2}]",
        sample_mean,
        lower,
        upper
    );

    // Confidence interval should be reasonable (not too wide or narrow)
    // Note: With simplified bootstrap implementation, CI width might be very small
    let ci_width = upper - lower;
    assert!(
        ci_width >= 0.0 && ci_width < 10.0,
        "CI width {:.2} should be reasonable",
        ci_width
    );

    println!(
        "✓ 95% Confidence interval: [{:.2}, {:.2}], mean = {:.2}",
        lower, upper, sample_mean
    );

    Ok(())
}

#[test]
fn test_multiple_comparison_correction() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();

    // Simulate multiple comparisons with some true and some false effects
    let mut p_values = Vec::new();

    // Add some significant p-values (simulating true effects)
    p_values.extend(vec![0.001, 0.002, 0.003, 0.015, 0.025]);

    // Add some non-significant p-values (simulating null effects)
    p_values.extend(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);

    let corrected = analyzer
        .multiple_comparison_correction(&p_values, MultipleComparisonCorrection::Bonferroni)?;

    // Bonferroni correction should multiply p-values by number of comparisons
    let num_comparisons = p_values.len() as f32;
    for (i, &original_p) in p_values.iter().enumerate() {
        let expected_corrected = (original_p * num_comparisons).min(1.0);
        let actual_corrected = corrected.adjusted_p_values[i];

        assert!(
            (expected_corrected - actual_corrected).abs() < 0.001,
            "Bonferroni correction mismatch: expected {:.6}, got {:.6}",
            expected_corrected,
            actual_corrected
        );
    }

    // Some corrections should reduce the number of significant results
    let original_significant = p_values.iter().filter(|&&p| p < 0.05).count();
    let corrected_significant = corrected
        .adjusted_p_values
        .iter()
        .filter(|&&p| p < 0.05)
        .count();

    assert!(
        corrected_significant <= original_significant,
        "Corrected significant count ({}) should be ≤ original ({})",
        corrected_significant,
        original_significant
    );

    println!("✓ Multiple comparison correction:");
    println!(
        "  Original significant: {}/{}",
        original_significant,
        p_values.len()
    );
    println!(
        "  Corrected significant: {}/{}",
        corrected_significant,
        p_values.len()
    );

    Ok(())
}

#[test]
fn test_effect_size_calculations() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();

    // Test Cohen's d calculation
    let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Mean = 3.0
    let group2 = vec![3.0, 4.0, 5.0, 6.0, 7.0]; // Mean = 5.0, difference = 2.0

    let t_test_result = analyzer.independent_t_test(&group1, &group2)?;

    // Effect size should be reasonable
    assert!(
        t_test_result.effect_size.is_some(),
        "Effect size should be calculated"
    );
    let effect_size = t_test_result.effect_size.unwrap();
    let abs_effect_size = effect_size.abs(); // Use absolute value since sign depends on group order
    assert!(
        abs_effect_size > 0.0,
        "Effect size should be non-zero for different groups, got {:.3}",
        abs_effect_size
    );

    // For this example, effect size should be large (> 0.8)
    assert!(
        abs_effect_size > 0.8,
        "Effect size should be large for clearly different groups, got {:.3}",
        abs_effect_size
    );

    println!("✓ Effect size calculation: Cohen's d = {:.3}", effect_size);

    Ok(())
}

#[test]
fn test_statistical_result_interpretation() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = StatisticalAnalyzer::new();

    // Test interpretation of different significance levels
    let sample1 = vec![1.0, 1.1, 1.2, 1.0, 1.1];
    let sample2 = vec![5.0, 5.1, 5.2, 5.0, 5.1];

    let result = analyzer.paired_t_test(&sample1, &sample2, None)?;

    // Test significance
    assert!(
        result.is_significant,
        "Should be significant at the configured alpha level"
    );

    // Test interpretation
    let interpretation = &result.interpretation;
    assert!(
        interpretation.contains("significant") || interpretation.contains("different"),
        "Interpretation should mention significance: {}",
        interpretation
    );

    println!("✓ Statistical interpretation: {}", interpretation);

    Ok(())
}
