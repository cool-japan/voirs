//! # Accuracy Validation Example
//!
//! This example demonstrates how to use the VoiRS accuracy validation framework
//! to validate ASR model performance against standard benchmarks.

use voirs_recognizer::prelude::*;
use voirs_recognizer::{
    asr::{AccuracyRequirement, BenchmarkingConfig},
    RecognitionError,
};

#[tokio::main]
async fn main() -> Result<(), RecognitionError> {
    println!("🚀 VoiRS Accuracy Validation Example");
    println!("=====================================");

    // Create a benchmarking suite with default configuration
    println!("📊 Setting up benchmarking suite...");
    let benchmark_config = BenchmarkingConfig::default();
    let benchmark_suite = ASRBenchmarkingSuite::new(benchmark_config).await?;

    // Create an accuracy validator with standard VoiRS requirements
    println!("🎯 Creating accuracy validator with standard requirements...");
    let accuracy_validator = AccuracyValidator::new_standard();

    println!("📋 Standard accuracy requirements:");
    for (i, requirement) in accuracy_validator.requirements.iter().enumerate() {
        println!("  {}. {}", i + 1, requirement.description);
        println!("     Dataset: {:?}", requirement.dataset);
        println!("     Language: {:?}", requirement.language);
        println!("     Max WER: {:.1}%", requirement.max_wer * 100.0);
        println!("     Max CER: {:.1}%", requirement.max_cer * 100.0);
        if let Some(min_phoneme_acc) = requirement.min_phoneme_accuracy {
            println!("     Min Phoneme Accuracy: {:.1}%", min_phoneme_acc * 100.0);
        }
        println!();
    }

    // Validate accuracy against standard benchmarks
    println!("🔍 Running accuracy validation...");
    let validation_report = accuracy_validator
        .validate_accuracy(&benchmark_suite)
        .await?;

    // Generate and display validation report
    println!("📈 Generating validation report...");
    let summary = accuracy_validator.generate_summary_report(&validation_report);
    println!("{}", summary);

    // Check if all requirements passed
    if validation_report.overall_passed {
        println!("✅ All accuracy requirements passed!");
        println!("🎉 The system meets production quality standards.");
    } else {
        println!("❌ Some accuracy requirements failed.");
        println!(
            "📊 Passed: {}/{}",
            validation_report.passed_requirements, validation_report.total_requirements
        );

        // Show detailed failure information
        for result in &validation_report.results {
            if !result.passed {
                println!("🔴 Failed: {}", result.requirement.description);
                if let Some(failure_reason) = &result.failure_reason {
                    println!("   Reason: {}", failure_reason);
                }
            }
        }
    }

    // Example of custom accuracy validator
    println!("\n🛠️  Custom Accuracy Validator Example");
    println!("=====================================");

    let custom_requirements = vec![
        AccuracyRequirement {
            id: "production_english".to_string(),
            description: "Production English ASR - WER < 3%".to_string(),
            dataset: voirs_recognizer::asr::benchmarking_suite::Dataset::LibriSpeech,
            language: LanguageCode::EnUs,
            max_wer: 0.03,
            max_cer: 0.015,
            min_phoneme_accuracy: Some(0.95),
            model_id: "whisper_large".to_string(),
        },
        AccuracyRequirement {
            id: "multilingual_support".to_string(),
            description: "Multilingual support validation".to_string(),
            dataset: voirs_recognizer::asr::benchmarking_suite::Dataset::CommonVoice,
            language: LanguageCode::DeDe,
            max_wer: 0.08,
            max_cer: 0.04,
            min_phoneme_accuracy: Some(0.88),
            model_id: "whisper_base".to_string(),
        },
    ];

    let custom_validator = AccuracyValidator::new_custom(custom_requirements);
    println!("🎯 Custom requirements created:");
    for requirement in &custom_validator.requirements {
        println!("  - {}", requirement.description);
    }

    // Validate custom requirements
    let custom_report = custom_validator.validate_accuracy(&benchmark_suite).await?;
    let custom_summary = custom_validator.generate_summary_report(&custom_report);
    println!("\n📊 Custom Validation Results:");
    println!("{}", custom_summary);

    // Performance tips
    println!("\n💡 Performance Tips:");
    println!("=====================");
    println!("1. Use caching for repeated validations");
    println!("2. Run validations in parallel for multiple models");
    println!("3. Consider using smaller datasets for quick validation");
    println!("4. Monitor validation performance over time");
    println!("5. Set up automated validation in CI/CD pipelines");

    Ok(())
}
