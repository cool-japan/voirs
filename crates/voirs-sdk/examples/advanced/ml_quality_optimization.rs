//! ML-Based Quality Optimization Example
//!
//! Demonstrates the use of machine learning-based quality prediction
//! for adaptive synthesis optimization.
//!
//! This example shows:
//! - ML-based quality predictor with online learning
//! - Text complexity analysis for prediction
//! - Adaptive quality controller with ML integration
//! - Performance tracking and statistics

use voirs_sdk::adaptive::{
    AdaptationStats, AdaptiveConfig, AdaptiveController, PredictionInput, QualityPredictor,
    QualityTarget, SystemMetrics, TextComplexityAnalyzer, TrainingSample,
};
use voirs_sdk::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== ML-Based Quality Optimization Demo ===\n");

    // Demo 1: Text Complexity Analysis
    demo_text_complexity_analysis()?;

    // Demo 2: ML Quality Predictor
    demo_ml_quality_predictor().await?;

    // Demo 3: Adaptive Controller with ML
    demo_adaptive_controller_with_ml().await?;

    // Demo 4: Online Learning Simulation
    demo_online_learning().await?;

    Ok(())
}

/// Demo 1: Text Complexity Analysis
fn demo_text_complexity_analysis() -> Result<()> {
    println!("--- Demo 1: Text Complexity Analysis ---");

    let test_texts = vec![
        ("Simple text.", "Simple"),
        (
            "The quick brown fox jumps over the lazy dog.",
            "Medium complexity",
        ),
        (
            "The sophisticated implementation of machine learning algorithms \
             requires comprehensive understanding of mathematical foundations, \
             statistical methodologies, and computational efficiency considerations.",
            "Complex",
        ),
        (
            "Pneumonoultramicroscopicsilicovolcanoconiosis is an incredibly \
             long word that represents a lung disease caused by inhaling very \
             fine silica particles.",
            "Very complex",
        ),
    ];

    for (text, description) in test_texts {
        let complexity = TextComplexityAnalyzer::analyze(text);
        println!(
            "{}: {:.3} - \"{}...\"",
            description,
            complexity,
            text.chars().take(50).collect::<String>()
        );
    }

    println!();
    Ok(())
}

/// Demo 2: ML Quality Predictor
async fn demo_ml_quality_predictor() -> Result<()> {
    println!("--- Demo 2: ML Quality Predictor ---");

    let mut predictor = QualityPredictor::new().with_history_size(100, 5);

    // Simulate training with various scenarios
    println!("Training predictor with 20 samples...");

    for i in 0..20 {
        let cpu_usage = 0.3 + (i as f32 * 0.03);
        let memory_usage = 0.4 + (i as f32 * 0.02);
        let quality = if cpu_usage < 0.5 {
            QualityTarget::VeryHigh
        } else if cpu_usage < 0.7 {
            QualityTarget::High
        } else {
            QualityTarget::Medium
        };

        let sample = TrainingSample {
            input: PredictionInput {
                cpu_usage,
                memory_usage,
                text_complexity: 0.5,
                time_of_day: 12,
                recent_rtf: cpu_usage * 0.8,
            },
            quality,
            synthesis_time_ms: (100.0 + cpu_usage * 500.0) as u64,
            success: true,
            measured_rtf: cpu_usage * 0.8,
        };

        predictor.train(sample).await?;
    }

    // Test predictions with different scenarios
    let test_scenarios = vec![
        (0.3, 0.4, "Low load"),
        (0.6, 0.5, "Medium load"),
        (0.9, 0.8, "High load"),
    ];

    for (cpu, mem, scenario) in test_scenarios {
        let input = PredictionInput {
            cpu_usage: cpu,
            memory_usage: mem,
            text_complexity: 0.5,
            time_of_day: 12,
            recent_rtf: cpu * 0.8,
        };

        let prediction = predictor.predict(&input).await?;
        println!(
            "{}: Predicted quality={:?} (confidence={:.2}), expected_time={}ms",
            scenario, prediction.quality, prediction.confidence, prediction.expected_time_ms
        );
    }

    // Show predictor statistics
    let stats = predictor.get_stats();
    println!("\nPredictor Statistics:");
    println!("  Total samples: {}", stats.total_samples);
    println!(
        "  Success rate: {:.1}%",
        stats.successful_samples as f64 / stats.total_samples as f64 * 100.0
    );
    println!("  Avg synthesis time: {:.0}ms", stats.avg_synthesis_time_ms);
    println!("  Overall confidence: {:.2}", stats.confidence);

    println!();
    Ok(())
}

/// Demo 3: Adaptive Controller with ML
async fn demo_adaptive_controller_with_ml() -> Result<()> {
    println!("--- Demo 3: Adaptive Controller with ML ---");

    let config = AdaptiveConfig::default()
        .with_target_latency(100)
        .with_min_quality(QualityTarget::Low)
        .with_max_quality(QualityTarget::VeryHigh)
        .with_adaptation_speed(0.8)
        .with_prediction(true); // Enable ML prediction

    let controller = AdaptiveController::new(config);

    // Simulate various system states
    let scenarios = vec![
        (0.3, 0.4, 0.4, "Optimal conditions"),
        (0.7, 0.6, 0.7, "Moderate load"),
        (0.9, 0.8, 1.1, "High stress"),
        (0.4, 0.5, 0.5, "Recovery"),
    ];

    println!("Simulating system state changes:\n");

    for (cpu, mem, rtf, description) in scenarios {
        let metrics = SystemMetrics {
            cpu_usage: cpu,
            memory_usage: mem,
            current_rtf: rtf,
            recent_latency_ms: (rtf * 100.0) as u64,
            timestamp: std::time::Instant::now(),
        };

        // Update controller with metrics
        if let Some(new_quality) = controller.update_metrics(metrics).await? {
            println!("{}: Quality adjusted to {:?}", description, new_quality);
        } else {
            let current_quality = controller.get_recommended_quality().await?;
            println!("{}: Quality stable at {:?}", description, current_quality);
        }

        // Record performance for ML training
        let quality = controller.get_recommended_quality().await?;
        controller
            .record_performance(quality, (rtf * 100.0) as u64, true)
            .await?;

        // Wait for min_change_interval
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    // Show adaptation statistics
    let stats = controller.get_adaptation_stats().await?;
    println!("\nAdaptation Statistics:");
    println!("  Total adjustments: {}", stats.total_adjustments);
    println!("  Current quality: {:?}", stats.current_quality);
    println!("  Success rate: {:.1}%", stats.success_rate * 100.0);
    println!("  Avg synthesis time: {:.0}ms", stats.avg_synthesis_time_ms);

    // Show ML predictor statistics if available
    if let Some(pred_stats) = controller.get_predictor_stats().await? {
        println!("\nML Predictor Statistics:");
        println!("  Training samples: {}", pred_stats.total_samples);
        println!("  Model confidence: {:.2}", pred_stats.confidence);
    }

    println!();
    Ok(())
}

/// Demo 4: Online Learning Simulation
async fn demo_online_learning() -> Result<()> {
    println!("--- Demo 4: Online Learning Simulation ---");

    let config = AdaptiveConfig::default()
        .with_target_latency(100)
        .with_prediction(true);

    let controller = AdaptiveController::new(config);

    let texts = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence and machine learning are revolutionizing technology.",
        "Natural language processing enables computers to understand human speech.",
    ];

    println!("Training ML predictor with real text examples:\n");

    for (i, text) in texts.iter().enumerate() {
        let complexity = TextComplexityAnalyzer::analyze(text);

        // Simulate system metrics varying with text complexity
        let base_load = 0.4;
        let load_factor = complexity * 0.3;

        let metrics = SystemMetrics {
            cpu_usage: base_load + load_factor,
            memory_usage: base_load + load_factor * 0.8,
            current_rtf: base_load + load_factor,
            recent_latency_ms: ((base_load + load_factor) * 100.0) as u64,
            timestamp: std::time::Instant::now(),
        };

        // Update metrics
        controller.update_metrics(metrics).await?;

        // Get predicted quality for this text
        if let Some(predicted_quality) = controller.get_predicted_quality(text).await? {
            println!(
                "Text {}: complexity={:.3}, predicted_quality={:?}",
                i + 1,
                complexity,
                predicted_quality
            );
        } else {
            println!(
                "Text {}: complexity={:.3} (not enough training data yet)",
                i + 1,
                complexity
            );
        }

        // Record performance with text context
        let quality = controller.get_recommended_quality().await?;
        let synthesis_time = ((base_load + load_factor) * 200.0) as u64;

        controller
            .record_performance_with_text(quality, synthesis_time, true, text)
            .await?;

        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    }

    // Final statistics
    if let Some(stats) = controller.get_predictor_stats().await? {
        println!("\nFinal ML Predictor State:");
        println!("  Training samples: {}", stats.total_samples);
        println!(
            "  Success rate: {:.1}%",
            stats.successful_samples as f64 / stats.total_samples as f64 * 100.0
        );
        println!("  Avg synthesis time: {:.0}ms", stats.avg_synthesis_time_ms);
        println!("  Model confidence: {:.2}", stats.confidence);
    }

    println!("\n=== ML-Based Quality Optimization Demo Complete ===");

    Ok(())
}
