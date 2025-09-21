//! Quality Metrics Automation Example
//!
//! This example demonstrates the automated quality metrics system
//! for comprehensive emotion processing quality analysis and monitoring.

use voirs_emotion::prelude::*;
use voirs_emotion::quality::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🎯 VoiRS Emotion Quality Metrics Example");
    println!("{}", "=".repeat(45));

    // Create quality analyzer with default production targets
    println!("\n📊 Creating quality analyzer with production targets...");
    let analyzer = QualityAnalyzer::new()?;

    // Create sample emotion vectors for testing
    let mut happy_emotion = EmotionVector::new();
    happy_emotion.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);

    let mut sad_emotion = EmotionVector::new();
    sad_emotion.add_emotion(Emotion::Sad, EmotionIntensity::MEDIUM);

    let mut excited_emotion = EmotionVector::new();
    excited_emotion.add_emotion(Emotion::Excited, EmotionIntensity::VERY_HIGH);

    // Generate sample audio data (in real use, this would be actual audio)
    let sample_audio = generate_sample_audio(44100); // 1 second of audio

    println!("\n🔍 Analyzing emotion quality for different emotions...");

    // Analyze Happy emotion quality
    println!("\n--- Happy Emotion Analysis ---");
    let happy_quality = analyzer
        .analyze_emotion_quality(&happy_emotion, &sample_audio)
        .await?;
    println!("{}", happy_quality.detailed_report());

    // Analyze Sad emotion quality
    println!("\n--- Sad Emotion Analysis ---");
    let sad_quality = analyzer
        .analyze_emotion_quality(&sad_emotion, &sample_audio)
        .await?;
    println!("{}", sad_quality.summary());

    // Analyze Excited emotion quality
    println!("\n--- Excited Emotion Analysis ---");
    let excited_quality = analyzer
        .analyze_emotion_quality(&excited_emotion, &sample_audio)
        .await?;
    println!("{}", excited_quality.summary());

    // Demonstrate custom quality targets
    println!("\n🎯 Testing with custom quality targets...");
    let strict_targets = QualityTargets {
        min_naturalness_score: 4.5,          // Stricter: 4.5 instead of 4.2
        min_emotion_accuracy_percent: 95.0,  // Stricter: 95% instead of 90%
        min_consistency_score_percent: 98.0, // Stricter: 98% instead of 95%
        min_user_satisfaction_percent: 90.0, // Stricter: 90% instead of 85%
        min_audio_quality_score: 4.3,        // Stricter audio quality
        max_distortion_percent: 0.5,         // Lower distortion tolerance
    };

    let strict_analyzer = QualityAnalyzer::with_targets(strict_targets)?;
    let strict_analysis = strict_analyzer
        .analyze_emotion_quality(&happy_emotion, &sample_audio)
        .await?;

    println!("Strict Quality Analysis:");
    println!("{}", strict_analysis.summary());

    // Demonstrate regression testing
    println!("\n🔄 Demonstrating quality regression testing...");
    let mut regression_tester = QualityRegressionTester::new()?;

    // Set baseline measurements
    let baseline_measurements = vec![happy_quality.clone(), sad_quality.clone()];
    regression_tester.set_baseline(baseline_measurements);

    // Test for regression with modified audio (simulating a change)
    let mut modified_audio = sample_audio.clone();
    // Add some distortion to simulate quality degradation
    for sample in &mut modified_audio {
        *sample *= 0.9; // Slight amplitude reduction
    }

    let regression_result = regression_tester
        .test_regression(&happy_emotion, &modified_audio)
        .await?;
    println!("Regression Test Result: {}", regression_result.summary);

    if regression_result.regression_detected {
        println!("⚠️ Quality regression detected!");
        println!("Degradation: {:.1}%", regression_result.degradation_percent);
    } else {
        println!("✅ No quality regression detected");
    }

    // Quality metrics summary
    println!("\n📈 Quality Metrics Summary");
    println!("{}", "=".repeat(30));

    let all_measurements = vec![&happy_quality, &sad_quality, &excited_quality];
    let avg_naturalness: f64 = all_measurements
        .iter()
        .map(|m| m.naturalness_score)
        .sum::<f64>()
        / all_measurements.len() as f64;
    let avg_accuracy: f64 = all_measurements
        .iter()
        .map(|m| m.emotion_accuracy_percent)
        .sum::<f64>()
        / all_measurements.len() as f64;
    let avg_consistency: f64 = all_measurements
        .iter()
        .map(|m| m.consistency_score_percent)
        .sum::<f64>()
        / all_measurements.len() as f64;
    let avg_satisfaction: f64 = all_measurements
        .iter()
        .map(|m| m.user_satisfaction_percent)
        .sum::<f64>()
        / all_measurements.len() as f64;

    println!(
        "Average Naturalness Score: {:.2}/5.0 {}",
        avg_naturalness,
        if avg_naturalness >= 4.2 { "✅" } else { "❌" }
    );
    println!(
        "Average Emotion Accuracy: {:.1}% {}",
        avg_accuracy,
        if avg_accuracy >= 90.0 { "✅" } else { "❌" }
    );
    println!(
        "Average Consistency Score: {:.1}% {}",
        avg_consistency,
        if avg_consistency >= 95.0 {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "Average User Satisfaction: {:.1}% {}",
        avg_satisfaction,
        if avg_satisfaction >= 85.0 {
            "✅"
        } else {
            "❌"
        }
    );

    // Production readiness assessment
    let production_ready = all_measurements
        .iter()
        .all(|m| m.meets_production_standards());
    println!(
        "\n🚀 Production Readiness: {}",
        if production_ready {
            "READY ✅"
        } else {
            "NOT READY ❌"
        }
    );

    if !production_ready {
        println!("\nFailed Quality Targets:");
        for measurement in all_measurements {
            if !measurement.meets_production_standards() {
                for (metric, &passed) in &measurement.metric_status {
                    if !passed {
                        println!("  • {} in {} emotion", metric, measurement.metadata.emotion);
                    }
                }
            }
        }
    }

    println!("\n✨ Quality metrics analysis completed!");
    println!("\nKey Quality Targets Validated:");
    println!("  🎭 Naturalness Score: Target MOS 4.2+");
    println!("  🎯 Emotion Accuracy: Target 90%+");
    println!("  📏 Consistency Score: Target 95%+");
    println!("  😊 User Satisfaction: Target 85%+");
    println!("  🎵 Audio Quality: Target MOS 4.0+");
    println!("  📉 Distortion Level: Target <1%");

    Ok(())
}

/// Generate sample audio data for testing
fn generate_sample_audio(sample_rate: usize) -> Vec<f32> {
    let duration_seconds = 1.0;
    let num_samples = (sample_rate as f64 * duration_seconds) as usize;
    let mut audio = Vec::with_capacity(num_samples);

    // Generate a simple sine wave with some harmonics
    let fundamental_freq = 440.0; // A4 note
    let sample_rate_f64 = sample_rate as f64;

    for i in 0..num_samples {
        let t = i as f64 / sample_rate_f64;
        let sample = 0.3 * (2.0 * std::f64::consts::PI * fundamental_freq * t).sin()
            + 0.1 * (2.0 * std::f64::consts::PI * fundamental_freq * 2.0 * t).sin()
            + 0.05 * (2.0 * std::f64::consts::PI * fundamental_freq * 3.0 * t).sin();

        // Add some gentle amplitude modulation to make it more interesting
        let modulation = 1.0 + 0.1 * (2.0 * std::f64::consts::PI * 5.0 * t).sin();

        audio.push((sample * modulation) as f32);
    }

    audio
}
