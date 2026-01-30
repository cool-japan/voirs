//! Speaker Verification Example
//!
//! This example demonstrates speaker verification and identity validation.
//! Run with: `cargo run --example speaker_verification --features acoustic-integration`

use voirs_cloning::{
    verification::{SpeakerVerifier, VerificationConfig},
    VoiceSample,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VoiRS Speaker Verification Example ===\n");

    // Step 1: Configure verifier
    println!("1. Configuring speaker verifier...");
    let config = VerificationConfig::default();

    let mut verifier = SpeakerVerifier::new(config)?;
    println!("   ✓ Verifier configured\n");

    // Step 2: Create sample voice data
    println!("2. Creating sample voice data...");
    let sample_rate = 16000;
    let duration_seconds = 5.0;
    let num_samples = (sample_rate as f32 * duration_seconds) as usize;

    // Create reference sample (Alice's voice)
    let reference_sample = VoiceSample::new(
        "alice_reference".to_string(),
        (0..num_samples)
            .map(|i| (i as f32 * 0.001).sin() * 0.15)
            .collect(),
        sample_rate,
    );

    // Create test sample (genuine - slightly different but same speaker)
    let genuine_sample = VoiceSample::new(
        "alice_test".to_string(),
        (0..num_samples)
            .map(|i| (i as f32 * 0.00105).sin() * 0.15)
            .collect(),
        sample_rate,
    );

    // Create impostor sample (different speaker)
    let impostor_sample = VoiceSample::new(
        "bob_test".to_string(),
        (0..num_samples)
            .map(|i| (i as f32 * 0.005).sin() * 0.2)
            .collect(),
        sample_rate,
    );

    println!("   ✓ Created reference, genuine, and impostor samples\n");

    // Step 3: Verify genuine speaker
    println!("3. Verifying genuine speaker...");
    let genuine_result = verifier
        .verify_samples(&reference_sample, &genuine_sample)
        .await?;

    println!("   Verification Result:");
    println!(
        "   • Verified: {}",
        if genuine_result.verified {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!("   • Confidence: {:.2}%", genuine_result.confidence * 100.0);
    println!(
        "   • Similarity Score: {:.2}%",
        genuine_result.metrics.embedding_similarity * 100.0
    );
    println!();

    // Step 4: Verify impostor
    println!("4. Verifying impostor...");
    let impostor_result = verifier
        .verify_samples(&reference_sample, &impostor_sample)
        .await?;

    println!("   Verification Result:");
    println!(
        "   • Verified: {}",
        if impostor_result.verified {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!(
        "   • Confidence: {:.2}%",
        impostor_result.confidence * 100.0
    );
    println!(
        "   • Similarity Score: {:.2}%",
        impostor_result.metrics.embedding_similarity * 100.0
    );
    println!();

    println!("=== Speaker Verification Complete ===");
    println!("\nSecurity Notes:");
    println!("• Use multi-factor authentication for critical applications");
    println!("• Regularly update enrollment samples to handle voice changes");
    println!("• Enable anti-spoofing for production deployments");
    println!("• Monitor FAR/FRR rates and adjust thresholds as needed");

    Ok(())
}
