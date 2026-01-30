//! Example demonstrating quality gate validation system
//!
//! This example shows how to use the quality gate validation system to
//! enforce quality standards and performance requirements for TTS systems.

use voirs_evaluation::quality_gates::{
    EnforcementLevel, EvaluationMetrics, GateStatus, QualityGateConfig, QualityGateValidator, Trend,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VoiRS Quality Gate Validation Example ===\n");

    // Create a quality gate validator with default configuration
    let mut validator = QualityGateValidator::default();
    println!("✓ Created quality gate validator with default configuration");
    println!("  - PESQ correlation threshold: ≥ 0.9");
    println!("  - STOI accuracy threshold: ≥ 0.95");
    println!("  - MCD precision threshold: < 0.01 dB");
    println!("  - Max RTF: < 0.1x");
    println!("  - Max memory: < 1 GB");
    println!("  - Enforcement: Strict\n");

    // Example 1: High-quality system that passes all gates
    println!("Example 1: High-Quality TTS System");
    println!("-----------------------------------");
    let excellent_metrics = EvaluationMetrics {
        pesq_correlation: Some(0.95),
        stoi_accuracy: Some(0.97),
        mcd_variance: Some(0.005),
        type_i_error: Some(0.03),
        mos_score: Some(4.5),
        intelligibility: Some(0.98),
        real_time_factor: Some(0.05),
        memory_usage_gb: Some(0.5),
        latency_ms: Some(50),
        streaming_latency_ms: Some(80),
        reference_agreement: Some(0.995),
        test_coverage: Some(0.95),
        ..Default::default()
    };

    let result = validator.validate(&excellent_metrics)?;
    println!("Status: {:?}", result.status);
    println!("Overall Score: {:.2}%", result.overall_score * 100.0);
    println!(
        "Checks: {} passed, {} failed",
        result.report.passed_checks, result.report.failed_checks
    );
    println!("Execution Time: {:?}", result.report.execution_time);

    if result.status == GateStatus::Passed {
        println!("✓ All quality gates passed! System is production-ready.\n");
    }

    // Example 2: System with performance issues
    println!("Example 2: System with Performance Issues");
    println!("------------------------------------------");
    let performance_issues = EvaluationMetrics {
        pesq_correlation: Some(0.92),
        stoi_accuracy: Some(0.96),
        mos_score: Some(4.2),
        real_time_factor: Some(0.15), // Too slow!
        memory_usage_gb: Some(1.5),   // Too much memory!
        latency_ms: Some(150),        // Too high latency!
        ..Default::default()
    };

    let result = validator.validate(&performance_issues)?;
    println!("Status: {:?}", result.status);
    println!("Overall Score: {:.2}%", result.overall_score * 100.0);

    if !result.report.errors.is_empty() {
        println!("\nErrors detected:");
        for error in &result.report.errors {
            println!("  ✗ {}", error);
        }
    }

    if !result.report.recommendations.is_empty() {
        println!("\nRecommendations:");
        for rec in result.report.recommendations.iter().take(3) {
            println!("  → {}", rec);
        }
    }
    println!();

    // Example 3: System with quality issues
    println!("Example 3: System with Quality Issues");
    println!("--------------------------------------");
    let quality_issues = EvaluationMetrics {
        pesq_correlation: Some(0.85), // Below threshold
        stoi_accuracy: Some(0.92),    // Below threshold
        mos_score: Some(3.8),         // Below threshold
        mcd_variance: Some(0.02),     // Above threshold
        real_time_factor: Some(0.08), // Good
        memory_usage_gb: Some(0.6),   // Good
        ..Default::default()
    };

    let result = validator.validate(&quality_issues)?;
    println!("Status: {:?}", result.status);
    println!("Overall Score: {:.2}%", result.overall_score * 100.0);

    if result.status == GateStatus::Failed {
        println!("\n⚠ Quality gate failed! System requires improvements.");
        println!("\nCritical issues:");
        for error in result.report.errors.iter().take(5) {
            println!("  ✗ {}", error);
        }
    }
    println!();

    // Example 4: Using advisory mode
    println!("Example 4: Advisory Mode (Non-blocking)");
    println!("----------------------------------------");
    let mut advisory_config = QualityGateConfig::default();
    advisory_config.enforcement_level = EnforcementLevel::Advisory;
    let mut advisory_validator = QualityGateValidator::new(advisory_config);

    let marginal_metrics = EvaluationMetrics {
        mos_score: Some(3.9), // Slightly below threshold
        real_time_factor: Some(0.12),
        ..Default::default()
    };

    let result = advisory_validator.validate(&marginal_metrics)?;
    println!("Status: {:?}", result.status);
    println!("Overall Score: {:.2}%", result.overall_score * 100.0);

    if result.status == GateStatus::Warning {
        println!("⚠ Some checks failed, but advisory mode allows deployment");
        println!("  Consider improvements before production release\n");
    }

    // Example 5: Historical trend analysis
    println!("Example 5: Historical Trend Analysis");
    println!("-------------------------------------");
    let mut trend_validator = QualityGateValidator::default();

    // Simulate improving quality over time
    for i in 1..=10 {
        let mos = 3.8 + (i as f64 * 0.08);
        let rtf = 0.12 - (i as f64 * 0.008);

        let metrics = EvaluationMetrics {
            mos_score: Some(mos),
            real_time_factor: Some(rtf),
            pesq_correlation: Some(0.88 + (i as f64 * 0.01)),
            ..Default::default()
        };

        trend_validator.validate(&metrics)?;
    }

    let summary = trend_validator.generate_summary();
    println!("Total Validations: {}", summary.total_validations);
    println!("Pass Rate: {:.1}%", summary.pass_rate * 100.0);
    println!("Average Score: {:.2}%", summary.average_score * 100.0);
    println!("Trend: {:?}", summary.recent_trend);

    match summary.recent_trend {
        Trend::Improving => println!("✓ Quality is improving over time!"),
        Trend::Stable => println!("= Quality is stable"),
        Trend::Degrading => println!("⚠ Quality is degrading - investigate!"),
        Trend::Insufficient => println!("? Not enough data to determine trend"),
    }
    println!();

    // Example 6: Custom enforcement rules
    println!("Example 6: Custom Enforcement Rules");
    println!("------------------------------------");
    let mut custom_config = QualityGateConfig::default();
    custom_config.enforcement_level = EnforcementLevel::Custom;
    let mut custom_validator = QualityGateValidator::new(custom_config);

    let custom_metrics = EvaluationMetrics {
        mos_score: Some(4.3),
        pesq_correlation: Some(0.93),
        stoi_accuracy: Some(0.96),
        real_time_factor: Some(0.08),
        memory_usage_gb: Some(0.7),
        reference_agreement: Some(0.992),
        ..Default::default()
    };

    let result = custom_validator.validate(&custom_metrics)?;
    println!("Status: {:?}", result.status);
    println!("Overall Score: {:.2}%", result.overall_score * 100.0);
    println!("Custom enforcement: Pass if overall score ≥ 80%");

    if result.overall_score >= 0.8 {
        println!("✓ System meets custom quality threshold\n");
    }

    // Example 7: Detailed metric breakdown
    println!("Example 7: Detailed Metric Breakdown");
    println!("-------------------------------------");
    let detailed_metrics = EvaluationMetrics {
        pesq_correlation: Some(0.94),
        stoi_accuracy: Some(0.96),
        mcd_variance: Some(0.007),
        mos_score: Some(4.4),
        real_time_factor: Some(0.06),
        memory_usage_gb: Some(0.8),
        latency_ms: Some(60),
        ..Default::default()
    };

    let result = validator.validate(&detailed_metrics)?;

    println!("Metric Requirements:");
    for (name, check) in &result.metric_results {
        let status = if check.passed { "✓" } else { "✗" };
        println!(
            "  {} {}: {:.3} (threshold: {:.3})",
            status, name, check.actual_value, check.threshold
        );
    }

    println!("\nPerformance Requirements:");
    for (name, check) in &result.performance_results {
        let status = if check.passed { "✓" } else { "✗" };
        println!(
            "  {} {}: {:.3} (threshold: {:.3})",
            status, name, check.actual_value, check.threshold
        );
    }
    println!();

    // Example 8: Production deployment decision
    println!("Example 8: Production Deployment Decision");
    println!("-----------------------------------------");
    let production_metrics = EvaluationMetrics {
        // Excellent quality metrics
        mos_score: Some(4.6),
        pesq_correlation: Some(0.96),
        stoi_accuracy: Some(0.98),
        intelligibility: Some(0.99),

        // Good performance
        real_time_factor: Some(0.04),
        memory_usage_gb: Some(0.4),
        latency_ms: Some(45),

        // High validation standards
        reference_agreement: Some(0.998),
        test_coverage: Some(0.97),
        cross_platform_consistency: Some(0.995),

        ..Default::default()
    };

    let result = validator.validate(&production_metrics)?;

    println!("=== DEPLOYMENT DECISION ===");
    println!("Status: {:?}", result.status);
    println!("Overall Score: {:.2}%", result.overall_score * 100.0);
    println!(
        "Quality Checks: {}/{} passed",
        result.report.passed_checks, result.report.total_checks
    );

    if result.status == GateStatus::Passed && result.overall_score >= 0.95 {
        println!("\n🎉 APPROVED FOR PRODUCTION DEPLOYMENT");
        println!("   System meets all quality standards and performance requirements.");
        println!("   Ready for release to production environment.");
    } else {
        println!("\n⛔ NOT APPROVED FOR PRODUCTION");
        println!("   System requires improvements before deployment.");
    }

    Ok(())
}
