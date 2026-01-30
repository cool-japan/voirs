//! # Adaptive Learning System Demo
//!
//! This example demonstrates the adaptive learning system that learns from user feedback
//! to continuously improve synthesis quality and personalize voice characteristics.

use voirs_singing::prelude::*;
use voirs_singing::QualityRatings;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== VoiRS Adaptive Learning System Demo ===\n");

    // 1. Create adaptive learning system
    println!("1. Creating adaptive learning system...");
    let config = AdaptiveLearningConfig {
        learning_rate: 0.02,
        min_samples: 5,
        decay_factor: 0.95,
        auto_finetune: true,
        confidence_threshold: 0.6,
    };
    let system = AdaptiveLearningSystem::new(config);
    println!("   ✓ System created with learning rate: 0.02\n");

    // 2. Simulate user feedback collection
    println!("2. Collecting user feedback...");
    for i in 0..8 {
        let rating = 3.5 + (i as f32 * 0.2);
        let audio = vec![0.0; 16000]; // Simulated audio

        let mut feedback = UserFeedback::new(
            "demo_user",
            audio,
            rating,
            Some(format!("Feedback #{}: Quality improving", i + 1)),
        );

        // Add quality ratings
        feedback.quality_ratings = QualityRatings {
            pitch_accuracy: 3.0 + (i as f32 * 0.2),
            timing_precision: 3.5 + (i as f32 * 0.15),
            naturalness: 3.2 + (i as f32 * 0.18),
            expression: 3.8 + (i as f32 * 0.1),
            voice_quality: 3.6 + (i as f32 * 0.12),
        };

        system.add_feedback(feedback).await?;
        println!("   ✓ Feedback #{} added (rating: {:.1})", i + 1, rating);
    }
    println!();

    // 3. Get learning statistics
    println!("3. Learning Statistics:");
    let stats = system.get_statistics().await;
    println!("   Total Users: {}", stats.total_users);
    println!("   Total Feedback: {}", stats.total_feedback);
    println!("   Average Rating: {:.2}", stats.average_rating);
    println!("   Total Improvements: {}", stats.total_improvements);
    println!(
        "   Cumulative Improvement: {:.2}%\n",
        stats.cumulative_improvement * 100.0
    );

    // 4. Get user preferences
    println!("4. User Preferences:");
    if let Some(prefs) = system.get_user_preferences("demo_user").await {
        println!("   Sample Count: {}", prefs.sample_count);
        println!("   Confidence: {:.1}%", prefs.confidence * 100.0);
        println!("   Quality Weights:");
        println!("     - Pitch: {:.2}", prefs.quality_weights.pitch_weight);
        println!("     - Timing: {:.2}", prefs.quality_weights.timing_weight);
        println!(
            "     - Naturalness: {:.2}",
            prefs.quality_weights.naturalness_weight
        );
        println!(
            "     - Expression: {:.2}",
            prefs.quality_weights.expression_weight
        );
        println!(
            "     - Voice Quality: {:.2}\n",
            prefs.quality_weights.voice_quality_weight
        );
    }

    // 5. Get personalized recommendations
    println!("5. Personalized Recommendations:");
    match system.get_recommendations("demo_user").await {
        Ok(recs) => {
            println!("   Confidence: {:.1}%", recs.confidence * 100.0);
            println!("   Recommended Parameters:");
            for (key, value) in &recs.parameter_adjustments {
                println!("     - {}: {:.2}", key, value);
            }
            println!("   Recommended Techniques:");
            for technique in &recs.techniques {
                println!("     - {}", technique);
            }
        }
        Err(e) => println!("   ⚠ Not enough data yet: {}", e),
    }
    println!();

    // 6. Get style adaptation
    println!("6. Style Adaptation:");
    let style_id = "user_demo_user_style";
    if let Some(adaptation) = system.get_style_adaptation(style_id).await {
        println!("   Style ID: {}", adaptation.style_id);
        println!("   Example Count: {}", adaptation.example_count);
        println!("   Confidence: {:.1}%", adaptation.confidence * 100.0);
        println!("   Vibrato Parameters:");
        println!("     - Rate: {:.2} Hz", adaptation.vibrato_params.rate);
        println!("     - Depth: {:.2}", adaptation.vibrato_params.depth);
        println!(
            "     - Onset Delay: {:.2}s",
            adaptation.vibrato_params.onset_delay
        );
    }
    println!();

    // 7. Get improvement history
    println!("7. Model Improvement History:");
    let history = system.get_improvement_history().await;
    for improvement in history.iter().take(3) {
        println!(
            "   Iteration {}: Quality Δ = {:.3}, Training Samples = {}",
            improvement.iteration, improvement.quality_delta, improvement.training_samples
        );
    }

    println!("\n=== Demo Complete ===");
    println!("The adaptive learning system continuously improves synthesis quality");
    println!("based on user feedback, preferences, and style adaptations.");

    Ok(())
}
