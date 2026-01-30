//! Simple Voice Cloning Example
//!
//! This example demonstrates basic voice cloning using few-shot learning.
//! Run with: `cargo run --example simple_voice_cloning --features acoustic-integration`

use voirs_cloning::{FewShotConfig, FewShotLearner, MetaLearningAlgorithm, VoiceSample};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VoiRS Voice Cloning Example ===\n");

    // Step 1: Configure few-shot learner
    println!("1. Configuring few-shot learner...");
    let config = FewShotConfig {
        num_shots: 3,
        quality_threshold: 0.3,
        meta_algorithm: MetaLearningAlgorithm::ProtoNet,
        ..Default::default()
    };

    let mut learner = FewShotLearner::new(config)?;
    println!("   ✓ Learner configured with 3-shot learning\n");

    // Step 2: Create sample voice data
    // In a real application, you would load actual audio files here
    println!("2. Preparing voice samples...");
    let sample_rate = 16000;
    let duration_seconds = 3.0;
    let num_samples = (sample_rate as f32 * duration_seconds) as usize;

    // Create synthetic audio samples (in practice, use real audio)
    let samples = vec![
        VoiceSample::new(
            "sample_1".to_string(),
            (0..num_samples)
                .map(|i| (i as f32 * 0.001).sin() * 0.1)
                .collect(),
            sample_rate,
        ),
        VoiceSample::new(
            "sample_2".to_string(),
            (0..num_samples)
                .map(|i| (i as f32 * 0.002).sin() * 0.1)
                .collect(),
            sample_rate,
        ),
        VoiceSample::new(
            "sample_3".to_string(),
            (0..num_samples)
                .map(|i| (i as f32 * 0.003).sin() * 0.1)
                .collect(),
            sample_rate,
        ),
    ];

    println!(
        "   ✓ Prepared {} samples ({} seconds each)\n",
        samples.len(),
        duration_seconds
    );

    // Step 3: Perform voice cloning adaptation
    println!("3. Adapting speaker with few-shot learning...");
    let result = learner.adapt_speaker("demo_speaker", &samples).await?;

    println!("   ✓ Adaptation complete!");
    println!("   • Speaker ID: demo_speaker");
    println!("   • Confidence: {:.2}%", result.confidence * 100.0);
    println!("   • Quality Score: {:.2}%", result.quality_score * 100.0);
    println!("   • Samples Used: {}", result.samples_used);
    println!("   • Embedding Dim: {}", result.speaker_embedding.len());
    println!("   • Algorithm: {:?}", result.algorithm);
    println!("   • Adaptation Time: {:.3?}\n", result.adaptation_time);

    // Step 5: Display cross-lingual info if available
    if let Some(cross_lingual) = result.cross_lingual_info {
        println!("5. Cross-lingual Information:");
        println!("   • Source Language: {}", cross_lingual.source_language);
        println!("   • Target Language: {}", cross_lingual.target_language);
        println!(
            "   • Phonetic Similarity: {:.2}%",
            cross_lingual.phonetic_similarity * 100.0
        );
        println!(
            "   • Adaptation Applied: {}",
            cross_lingual.language_adaptation_applied
        );
        println!();
    }

    println!("=== Voice Cloning Complete ===");
    println!("\nNext steps:");
    println!("• Use the speaker embedding for voice synthesis");
    println!("• Integrate with voirs-acoustic for TTS");
    println!("• Apply quality assessment to verify cloning quality");
    println!("• Store the speaker profile for future use");

    Ok(())
}
