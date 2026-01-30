//! Quality Assessment Example
//!
//! This example demonstrates comprehensive quality assessment for voice cloning.
//! Run with: `cargo run --example quality_assessment --features acoustic-integration`

use voirs_cloning::{quality::CloningQualityAssessor, VoiceSample};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VoiRS Quality Assessment Example ===\n");

    // Step 1: Create quality assessor
    println!("1. Configuring quality assessor...");
    let mut assessor = CloningQualityAssessor::new()?;
    println!("   ✓ Assessor configured with comprehensive analysis\n");

    // Step 2: Create original and cloned samples
    println!("2. Preparing audio samples...");
    let sample_rate = 16000;
    let duration_seconds = 3.0;
    let num_samples = (sample_rate as f32 * duration_seconds) as usize;

    // Original voice sample
    let original = VoiceSample::new(
        "original_speaker".to_string(),
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.2
            })
            .collect(),
        sample_rate,
    );

    // Cloned voice sample (slightly different to simulate cloning)
    let cloned = VoiceSample::new(
        "cloned_speaker".to_string(),
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (t * 442.0 * 2.0 * std::f32::consts::PI).sin() * 0.19 // Slight differences
            })
            .collect(),
        sample_rate,
    );

    println!("   ✓ Prepared original and cloned samples\n");

    // Step 3: Assess cloning quality
    println!("3. Assessing cloning quality...");
    let metrics = assessor.assess_quality(&original, &cloned).await?;

    println!("\n=== Quality Metrics ===");
    println!("\n📊 Overall Scores:");
    println!(
        "   • Overall Quality: {:.2}% {}",
        metrics.overall_score * 100.0,
        quality_emoji(metrics.overall_score)
    );
    println!(
        "   • Speaker Similarity: {:.2}% {}",
        metrics.speaker_similarity * 100.0,
        quality_emoji(metrics.speaker_similarity)
    );
    println!(
        "   • Audio Quality: {:.2}% {}",
        metrics.audio_quality * 100.0,
        quality_emoji(metrics.audio_quality)
    );
    println!(
        "   • Naturalness: {:.2}% {}",
        metrics.naturalness * 100.0,
        quality_emoji(metrics.naturalness)
    );
    println!(
        "   • Content Preservation: {:.2}% {}",
        metrics.content_preservation * 100.0,
        quality_emoji(metrics.content_preservation)
    );

    // Step 4: Display detailed analysis
    println!("\n🔍 Signal Analysis:");
    let snr = &metrics.analysis.snr_analysis;
    println!("   • Original SNR: {:.2} dB", snr.original_snr);
    println!("   • Cloned SNR: {:.2} dB", snr.cloned_snr);
    println!("   • SNR Degradation: {:.2} dB", snr.snr_degradation);
    println!("   • Dynamic Range: {:.2} dB", snr.dynamic_range);

    println!("\n🎵 Spectral Analysis:");
    let spectral = &metrics.analysis.spectral_analysis;
    println!(
        "   • Spectral Centroid Similarity: {:.2}%",
        spectral.spectral_centroid_similarity * 100.0
    );
    println!(
        "   • Harmonic Similarity: {:.2}%",
        spectral.harmonic_similarity * 100.0
    );
    println!(
        "   • Formant Similarity: {:.2}%",
        spectral.formant_similarity * 100.0
    );

    println!("\n⏱️  Temporal Analysis:");
    let temporal = &metrics.analysis.temporal_analysis;
    println!(
        "   • Duration Similarity: {:.2}%",
        temporal.duration_similarity * 100.0
    );
    println!(
        "   • Rhythm Similarity: {:.2}%",
        temporal.rhythm_similarity * 100.0
    );
    println!(
        "   • Speech Rate Similarity: {:.2}%",
        temporal.speech_rate_similarity * 100.0
    );

    println!("\n🎭 Perceptual Analysis:");
    let perceptual = &metrics.analysis.perceptual_analysis;
    println!(
        "   • Loudness Similarity: {:.2}%",
        perceptual.loudness_similarity * 100.0
    );
    println!(
        "   • Pitch Similarity: {:.2}%",
        perceptual.pitch_similarity * 100.0
    );
    println!(
        "   • Timber Similarity: {:.2}%",
        perceptual.timber_similarity * 100.0
    );

    println!("\n⚠️  Artifact Detection:");
    let artifacts = &metrics.analysis.artifact_analysis;
    println!(
        "   • Click Detection: {:.2}%",
        artifacts.click_detection * 100.0
    );
    println!(
        "   • Discontinuity Detection: {:.2}%",
        artifacts.discontinuity_detection * 100.0
    );
    println!(
        "   • Robotic Artifacts: {:.2}%",
        artifacts.robotic_artifacts * 100.0
    );
    println!(
        "   • Overall Artifact Score: {:.2}%",
        artifacts.overall_artifact_score * 100.0
    );

    // Step 5: Assessment metadata
    println!("\n📝 Assessment Metadata:");
    let metadata = &metrics.metadata;
    println!("   • Assessment Time: {:.2}s", metadata.assessment_time);
    println!(
        "   • Assessment Duration: {:.2}ms",
        metadata.assessment_duration
    );
    println!("   • Sample Rate: {} Hz", metadata.sample_rate);
    println!("   • Method: {}", metadata.assessment_method);
    println!("   • Version: {}", metadata.quality_version);

    // Step 6: Quality recommendation
    println!("\n💡 Recommendations:");
    if metrics.overall_score >= 0.9 {
        println!("   ✅ Excellent cloning quality! Ready for production use.");
    } else if metrics.overall_score >= 0.7 {
        println!("   ✓ Good cloning quality. Minor improvements possible.");
        if metrics.audio_quality < 0.7 {
            println!("   • Consider improving audio quality (noise reduction, better recording)");
        }
        if metrics.speaker_similarity < 0.7 {
            println!("   • Consider using more training samples or better quality samples");
        }
    } else if metrics.overall_score >= 0.5 {
        println!("   ⚠ Moderate cloning quality. Improvements recommended:");
        if metrics.audio_quality < 0.5 {
            println!("   • Improve audio quality (higher SNR, less artifacts)");
        }
        if metrics.speaker_similarity < 0.5 {
            println!("   • Use more diverse training samples");
        }
        if artifacts.overall_artifact_score > 0.3 {
            println!("   • Reduce synthesis artifacts");
        }
    } else {
        println!("   ❌ Low cloning quality. Significant improvements needed:");
        println!("   • Review training data quality");
        println!("   • Consider different cloning method");
        println!("   • Check model configuration");
    }

    println!("\n=== Quality Assessment Complete ===");

    Ok(())
}

fn quality_emoji(score: f32) -> &'static str {
    match score {
        s if s >= 0.9 => "🌟",
        s if s >= 0.7 => "✅",
        s if s >= 0.5 => "⚠️",
        _ => "❌",
    }
}
