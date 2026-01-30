//! Comprehensive example demonstrating the quality gate system
//!
//! This example shows how to:
//! - Create quality gates with different strictness levels
//! - Define custom quality rules
//! - Validate datasets against quality criteria
//! - Generate detailed validation reports
//! - Filter datasets to only passing samples
//! - Use quality gates in CI/CD pipelines
//!
//! Run with: cargo run --example quality_gate

use voirs_dataset::{
    processing::quality_gate::{QualityGate, QualityRule, RuleType, Severity},
    AudioData, DatasetSample, LanguageCode, QualityMetrics, SpeakerInfo,
};

fn create_diverse_samples() -> Vec<DatasetSample> {
    let mut samples = Vec::new();

    // Sample 1: High quality, good duration
    let audio1 = AudioData::silence(3.0, 22050, 1);
    let sample1 = DatasetSample::new(
        "sample_001".to_string(),
        "This is a high quality sample with perfect metrics.".to_string(),
        audio1,
        LanguageCode::EnUs,
    )
    .with_quality(QualityMetrics {
        overall_quality: Some(0.95),
        snr: Some(35.0),
        clipping: Some(0.001),
        dynamic_range: Some(55.0),
        spectral_quality: Some(0.93),
    })
    .with_speaker(SpeakerInfo {
        id: "speaker_001".to_string(),
        name: Some("Professional Speaker".to_string()),
        gender: Some("female".to_string()),
        age: Some(30),
        accent: None,
        metadata: Default::default(),
    });

    // Sample 2: Too short
    let audio2 = AudioData::silence(0.3, 22050, 1);
    let sample2 = DatasetSample::new(
        "sample_002".to_string(),
        "Short".to_string(),
        audio2,
        LanguageCode::EnUs,
    )
    .with_quality(QualityMetrics {
        overall_quality: Some(0.85),
        snr: Some(28.0),
        clipping: Some(0.01),
        dynamic_range: Some(45.0),
        spectral_quality: Some(0.88),
    });

    // Sample 3: Low quality
    let audio3 = AudioData::silence(2.5, 22050, 1);
    let sample3 = DatasetSample::new(
        "sample_003".to_string(),
        "This sample has low quality metrics despite good duration.".to_string(),
        audio3,
        LanguageCode::EnUs,
    )
    .with_quality(QualityMetrics {
        overall_quality: Some(0.45),
        snr: Some(12.0),
        clipping: Some(0.15),
        dynamic_range: Some(20.0),
        spectral_quality: Some(0.50),
    });

    // Sample 4: High clipping
    let audio4 = AudioData::silence(2.0, 22050, 1);
    let sample4 = DatasetSample::new(
        "sample_004".to_string(),
        "This sample has excessive clipping.".to_string(),
        audio4,
        LanguageCode::EnUs,
    )
    .with_quality(QualityMetrics {
        overall_quality: Some(0.70),
        snr: Some(25.0),
        clipping: Some(0.25),
        dynamic_range: Some(35.0),
        spectral_quality: Some(0.75),
    });

    // Sample 5: No quality metrics
    let audio5 = AudioData::silence(2.5, 22050, 1);
    let sample5 = DatasetSample::new(
        "sample_005".to_string(),
        "This sample has no quality metrics.".to_string(),
        audio5,
        LanguageCode::EnUs,
    );

    // Sample 6: Empty text
    let audio6 = AudioData::silence(2.0, 22050, 1);
    let sample6 = DatasetSample::new(
        "sample_006".to_string(),
        "".to_string(),
        audio6,
        LanguageCode::EnUs,
    )
    .with_quality(QualityMetrics {
        overall_quality: Some(0.90),
        snr: Some(30.0),
        clipping: Some(0.01),
        dynamic_range: Some(50.0),
        spectral_quality: Some(0.88),
    });

    // Sample 7: Too long
    let audio7 = AudioData::silence(35.0, 22050, 1);
    let sample7 = DatasetSample::new(
        "sample_007".to_string(),
        "This is a very long sample that exceeds the maximum duration limit for training purposes."
            .to_string(),
        audio7,
        LanguageCode::EnUs,
    )
    .with_quality(QualityMetrics {
        overall_quality: Some(0.85),
        snr: Some(27.0),
        clipping: Some(0.02),
        dynamic_range: Some(48.0),
        spectral_quality: Some(0.86),
    });

    // Sample 8: Good quality, no speaker info
    let audio8 = AudioData::silence(2.2, 22050, 1);
    let sample8 = DatasetSample::new(
        "sample_008".to_string(),
        "Good quality but missing speaker information.".to_string(),
        audio8,
        LanguageCode::EnUs,
    )
    .with_quality(QualityMetrics {
        overall_quality: Some(0.88),
        snr: Some(29.0),
        clipping: Some(0.015),
        dynamic_range: Some(46.0),
        spectral_quality: Some(0.87),
    });

    samples.push(sample1);
    samples.push(sample2);
    samples.push(sample3);
    samples.push(sample4);
    samples.push(sample5);
    samples.push(sample6);
    samples.push(sample7);
    samples.push(sample8);

    samples
}

fn main() {
    println!("=== VoiRS Quality Gate System Example ===\n");

    let samples = create_diverse_samples();
    println!("Created {} test samples\n", samples.len());

    // Example 1: Permissive quality gate (warnings only)
    println!("Example 1: Permissive quality gate");
    println!("───────────────────────────────────\n");
    {
        let gate = QualityGate::permissive();
        let report = gate.validate(&samples).unwrap();

        println!("Permissive gate results:");
        println!(
            "  Passed: {}/{}",
            report.stats.passed_samples, report.stats.total_samples
        );
        println!("  Failed: {}", report.stats.failed_samples);
        println!("  Warnings: {}", report.warning_count);
        println!(
            "  Gate status: {}",
            if report.passed() {
                "✓ PASS"
            } else {
                "✗ FAIL"
            }
        );
    }

    // Example 2: Standard quality gate
    println!("\n\nExample 2: Standard quality gate");
    println!("─────────────────────────────────\n");
    {
        let gate = QualityGate::standard();
        let report = gate.validate(&samples).unwrap();

        report.print();
        println!(
            "\nGate status: {}",
            if report.passed() {
                "✓ PASS"
            } else {
                "✗ FAIL"
            }
        );
    }

    // Example 3: Strict quality gate
    println!("\n\nExample 3: Strict quality gate (production-ready)");
    println!("─────────────────────────────────────────────────\n");
    {
        let gate = QualityGate::strict();
        let report = gate.validate(&samples).unwrap();

        println!("Strict gate results:");
        println!("  Errors: {}", report.error_count);
        println!("  Warnings: {}", report.warning_count);
        println!("  Pass rate: {:.1}%", report.stats.pass_rate * 100.0);
        println!("\nTop violations:");

        let mut rule_counts: Vec<_> = report.stats.issues_by_rule.iter().collect();
        rule_counts.sort_by_key(|(_, count)| std::cmp::Reverse(**count));

        for (rule_id, count) in rule_counts.iter().take(5) {
            println!("  • {}: {} violations", rule_id, count);
        }
    }

    // Example 4: Custom quality gate with specific rules
    println!("\n\nExample 4: Custom quality gate with custom rules");
    println!("─────────────────────────────────────────────────\n");
    {
        let mut gate = QualityGate::new();

        // Add custom rules for our specific use case
        gate.add_rule(QualityRule::new(
            "optimal_duration",
            RuleType::MinDuration(1.5),
            "Samples should be at least 1.5 seconds for optimal training",
        ));

        gate.add_rule(QualityRule::new(
            "optimal_duration_max",
            RuleType::MaxDuration(10.0),
            "Samples should not exceed 10 seconds",
        ));

        gate.add_rule(QualityRule::new(
            "production_quality",
            RuleType::MinQuality(0.85),
            "Production samples require quality >= 0.85",
        ));

        gate.add_rule(QualityRule::new(
            "text_content",
            RuleType::NonEmptyText,
            "All samples must have text content",
        ));

        gate.add_rule(
            QualityRule::new(
                "snr_warning",
                RuleType::MinSnr(25.0),
                "SNR should be at least 25 dB for best results",
            )
            .with_severity(Severity::Warning),
        );

        let report = gate.validate(&samples).unwrap();

        println!("Custom gate with {} rules:", gate.rules.len());
        for rule in &gate.rules {
            println!("  • {} ({})", rule.description, rule.id);
        }

        println!("\nResults:");
        println!("  Total issues: {}", report.issues.len());
        println!("  Errors: {}", report.error_count);
        println!("  Warnings: {}", report.warning_count);
    }

    // Example 5: Examining specific issues
    println!("\n\nExample 5: Detailed issue examination");
    println!("──────────────────────────────────────\n");
    {
        let gate = QualityGate::strict();
        let report = gate.validate(&samples).unwrap();

        // Show issues for a specific sample
        let sample_id = "sample_003";
        let issues = report.issues_for_sample(sample_id);

        println!("Issues for sample '{}':", sample_id);
        for issue in &issues {
            println!(
                "  • [{}] {}: {}",
                match issue.severity {
                    Severity::Error => "ERROR",
                    Severity::Warning => "WARN",
                    Severity::Info => "INFO",
                },
                issue.rule_id,
                issue.context
            );
        }

        // Show all error issues
        println!("\nAll error issues:");
        let errors = report.issues_by_severity(Severity::Error);
        for issue in errors.iter().take(5) {
            println!(
                "  • [{}] {}: {}",
                issue.sample_id, issue.rule_id, issue.context
            );
        }
    }

    // Example 6: Filtering to passing samples only
    println!("\n\nExample 6: Filter dataset to passing samples");
    println!("────────────────────────────────────────────\n");
    {
        let mut gate = QualityGate::new();

        gate.add_rule(QualityRule::new(
            "min_duration",
            RuleType::MinDuration(1.0),
            "Minimum 1 second duration",
        ));

        gate.add_rule(QualityRule::new(
            "min_quality",
            RuleType::MinQuality(0.7),
            "Minimum 0.7 quality score",
        ));

        println!("Original dataset: {} samples", samples.len());

        let passing = gate.filter_passing(samples.clone()).unwrap();

        println!("After filtering: {} samples", passing.len());
        println!("Removed: {} samples", samples.len() - passing.len());

        println!("\nPassing sample IDs:");
        for sample in &passing {
            println!("  • {}", sample.id);
        }
    }

    // Example 7: Rule management
    println!("\n\nExample 7: Dynamic rule management");
    println!("───────────────────────────────────\n");
    {
        let mut gate = QualityGate::standard();

        println!("Standard gate rules: {}", gate.rules.len());

        // Add a new rule
        gate.add_rule(QualityRule::new(
            "custom_rule",
            RuleType::MinTextLength(20),
            "Text must be at least 20 characters",
        ));

        println!("After adding rule: {}", gate.rules.len());

        // Disable a rule
        gate.set_rule_enabled("min_text_length", false);

        // Remove a rule
        let removed = gate.remove_rule("custom_rule");
        if removed.is_some() {
            println!("Removed custom rule");
        }

        println!("Final rules: {}", gate.rules.len());
    }

    // Example 8: Using severity levels
    println!("\n\nExample 8: Understanding severity levels");
    println!("─────────────────────────────────────────\n");
    {
        let mut gate = QualityGate::new();

        // Error: Blocks the gate
        gate.add_rule(QualityRule::new(
            "critical_duration",
            RuleType::MinDuration(0.5),
            "Critical: Duration too short",
        ));

        // Warning: Doesn't block the gate
        gate.add_rule(
            QualityRule::new(
                "recommended_snr",
                RuleType::MinSnr(30.0),
                "Recommended: SNR could be higher",
            )
            .with_severity(Severity::Warning),
        );

        // Info: Just informational
        gate.add_rule(
            QualityRule::new(
                "speaker_info",
                RuleType::RequireSpeaker,
                "Info: Speaker info helps with training",
            )
            .with_severity(Severity::Info),
        );

        let report = gate.validate(&samples).unwrap();

        println!("Severity breakdown:");
        println!("  Errors (block gate): {}", report.error_count);
        println!("  Warnings (don't block): {}", report.warning_count);
        println!("  Info (informational): {}", report.info_count);
        println!("\nGate passed: {} (errors = 0)", report.passed());
    }

    // Example 9: Export report to JSON
    println!("\n\nExample 9: Export validation report");
    println!("────────────────────────────────────\n");
    {
        let gate = QualityGate::standard();
        let report = gate.validate(&samples).unwrap();

        match report.to_json() {
            Ok(json) => {
                println!("✓ Report exported to JSON");
                println!("JSON size: {} bytes", json.len());
                println!("\nFirst 200 characters:");
                println!("{}", &json[..200.min(json.len())]);
                println!("...");
            }
            Err(e) => println!("✗ Failed to export: {}", e),
        }
    }

    // Example 10: CI/CD integration pattern
    println!("\n\nExample 10: CI/CD integration pattern");
    println!("──────────────────────────────────────\n");
    {
        let gate = QualityGate::strict();
        let report = gate.validate(&samples).unwrap();

        // Simulate CI/CD workflow
        println!("Quality Gate Check:");
        println!("  Total samples: {}", report.stats.total_samples);
        println!("  Pass rate: {:.1}%", report.stats.pass_rate * 100.0);
        println!("  Errors: {}", report.error_count);
        println!("  Warnings: {}", report.warning_count);

        if report.passed() {
            println!("\n✓ Quality gate PASSED - proceeding with deployment");
            std::process::exit(0);
        } else {
            println!("\n✗ Quality gate FAILED - blocking deployment");
            println!("\nFix the following issues before deployment:");

            for (i, issue) in report
                .issues_by_severity(Severity::Error)
                .iter()
                .take(5)
                .enumerate()
            {
                println!(
                    "  {}. [{}] {}: {}",
                    i + 1,
                    issue.sample_id,
                    issue.rule_description,
                    issue.context
                );
            }

            // In real CI/CD, would exit with non-zero code
            // std::process::exit(1);
        }
    }

    println!("\n=== All examples completed successfully! ===");
}
