//! Audio Workflows Example
//!
//! Demonstrates the high-level audio processing workflows for common use cases.
//!
//! Run with: cargo run --example audio_workflows

use voirs_sdk::audio::workflows;
use voirs_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== VoiRS Audio Workflows Examples ===\n");

    // Create pipeline
    println!("Creating synthesis pipeline...");
    let pipeline = VoirsPipelineBuilder::new().build().await?;
    println!("✓ Pipeline ready\n");

    // Example 1: Podcast Quality Processing
    println!("1. Podcast Quality Processing");
    println!("   Synthesizing podcast intro...");
    let podcast_text = "Welcome to our podcast! Today we'll discuss neural speech synthesis.";
    let audio = pipeline.synthesize(podcast_text).await?;
    println!("   Applying podcast-quality processing...");
    let podcast_audio = workflows::podcast_quality(&audio)?;
    println!(
        "   ✓ Podcast audio: {} samples, {:.2}s duration",
        podcast_audio.len(),
        podcast_audio.duration()
    );
    println!(
        "   ✓ RMS: {:.2} dB, Peak: {:.2} dB\n",
        podcast_audio.rms_db(),
        podcast_audio.peak_db()
    );

    // Example 2: Voice Feature Analysis
    println!("2. Voice Feature Extraction");
    println!("   Synthesizing analysis sample...");
    let analysis_text = "This audio will be analyzed for voice characteristics.";
    let analysis_audio = pipeline.synthesize(analysis_text).await?;
    println!("   Extracting voice features...");
    let features = workflows::voice_feature_extraction(&analysis_audio)?;
    println!("   ✓ Fundamental Frequency (F0): {:.1} Hz", features.f0);
    println!(
        "   ✓ Formants: F1={:.0}Hz, F2={:.0}Hz, F3={:.0}Hz",
        features.formants.first().unwrap_or(&0.0),
        features.formants.get(1).unwrap_or(&0.0),
        features.formants.get(2).unwrap_or(&0.0)
    );
    println!("   ✓ Voice Quality:");
    println!("     - Jitter: {:.2}%", features.jitter);
    println!("     - Shimmer: {:.2}%", features.shimmer);
    println!("     - HNR: {:.2} dB", features.hnr);
    println!("   ✓ MFCC coefficients: {} features\n", features.mfcc.len());

    // Example 3: Audio Quality Metrics
    println!("3. Audio Quality Analysis");
    let quality_audio = pipeline.synthesize("Quality check sample").await?;
    let metrics = workflows::analyze_quality(&quality_audio)?;
    println!("   ✓ Level Metrics:");
    println!("     - RMS: {:.2} dB", metrics.rms_db);
    println!("     - Peak: {:.2} dB", metrics.peak_db);
    println!("     - Crest Factor: {:.2}", metrics.crest_factor);
    println!("   ✓ Quality Metrics:");
    println!("     - SNR: {:.2} dB", metrics.snr);
    println!(
        "     - Clipping: {}",
        if metrics.has_clipping { "Yes" } else { "No" }
    );
    println!("     - Clipped Samples: {}\n", metrics.clipped_samples);

    // Example 4: Telephone Quality Simulation
    println!("4. Telephone Quality Simulation");
    let phone_text = "This simulates telephone bandwidth.";
    let phone_audio = pipeline.synthesize(phone_text).await?;
    println!("   Original: {} Hz bandwidth", phone_audio.sample_rate());
    let telephone = workflows::telephone_quality(&phone_audio)?;
    println!("   ✓ Limited to 300-3400 Hz (telephone bandwidth)");
    println!("   ✓ Duration: {:.2}s\n", telephone.duration());

    // Example 5: Broadcast Quality Processing
    println!("5. Broadcast Quality Processing");
    let broadcast_text = "This is a professional broadcast announcement.";
    let broadcast_audio = pipeline.synthesize(broadcast_text).await?;
    let broadcast = workflows::broadcast_quality(&broadcast_audio)?;
    println!("   ✓ Broadcast-ready audio:");
    println!(
        "     - Peak level: {:.2} dB (target: -1 dB)",
        broadcast.peak_db()
    );
    println!("     - RMS level: {:.2} dB", broadcast.rms_db());
    println!("     - Duration: {:.2}s\n", broadcast.duration());

    // Example 6: Low-Bitrate Optimization
    println!("6. Low-Bitrate Codec Optimization");
    let optimize_text = "Optimizing for low-bitrate encoding.";
    let optimize_audio = pipeline.synthesize(optimize_text).await?;
    let optimized = workflows::low_bitrate_optimize(&optimize_audio)?;
    println!("   ✓ Optimized for low-bitrate codecs:");
    println!("     - High-pass filtered at 60 Hz");
    println!("     - Low-pass filtered at 16 kHz");
    println!("     - Normalized for maximum dynamic range\n");

    // Example 7: Silence Removal
    println!("7. Silence Detection and Removal");
    let silence_text = "Text with... long pauses... between words.";
    let silence_audio = pipeline.synthesize(silence_text).await?;
    println!(
        "   Original: {} samples ({:.2}s)",
        silence_audio.len(),
        silence_audio.duration()
    );

    // Detect silence first
    let silence_regions = silence_audio.detect_silence(-50.0, 0.3);
    println!("   Detected {} silence regions", silence_regions.len());

    if !silence_regions.is_empty() {
        let trimmed = workflows::remove_silence(&silence_audio, -50.0, 0.3)?;
        println!(
            "   ✓ After removal: {} samples ({:.2}s)",
            trimmed.len(),
            trimmed.duration()
        );
        let reduction =
            ((silence_audio.len() - trimmed.len()) as f32 / silence_audio.len() as f32) * 100.0;
        println!("   ✓ Reduced by: {:.1}%\n", reduction);
    } else {
        println!("   (No significant silence detected)\n");
    }

    // Example 8: Comparison of Different Workflows
    println!("8. Workflow Comparison");
    let compare_text = "Comparing different audio processing workflows.";
    let original = pipeline.synthesize(compare_text).await?;

    println!("   Original Audio:");
    println!("     - Peak: {:.2} dB", original.peak_db());
    println!("     - RMS: {:.2} dB", original.rms_db());

    let podcast = workflows::podcast_quality(&original)?;
    println!("   Podcast Quality:");
    println!("     - Peak: {:.2} dB (target: -6 dB)", podcast.peak_db());
    println!("     - RMS: {:.2} dB", podcast.rms_db());

    let broadcast = workflows::broadcast_quality(&original)?;
    println!("   Broadcast Quality:");
    println!("     - Peak: {:.2} dB (target: -1 dB)", broadcast.peak_db());
    println!("     - RMS: {:.2} dB", broadcast.rms_db());

    let telephone = workflows::telephone_quality(&original)?;
    println!("   Telephone Quality:");
    println!("     - Bandwidth limited to 300-3400 Hz");
    println!("     - Duration: {:.2}s\n", telephone.duration());

    // Summary
    println!("=== Summary ===");
    println!("✓ Demonstrated 8 audio workflow scenarios:");
    println!("  1. Podcast quality processing");
    println!("  2. Voice feature extraction (F0, formants, MFCC, jitter, shimmer, HNR)");
    println!("  3. Audio quality metrics (RMS, peak, SNR, clipping)");
    println!("  4. Telephone quality simulation");
    println!("  5. Broadcast quality processing");
    println!("  6. Low-bitrate codec optimization");
    println!("  7. Silence detection and removal");
    println!("  8. Workflow comparison");
    println!("\n✓ All workflows ready for production use!");

    Ok(())
}
