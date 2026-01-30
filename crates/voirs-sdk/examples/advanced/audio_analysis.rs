//! Comprehensive Audio Analysis Example
//!
//! This example demonstrates advanced audio analysis features in VoiRS SDK including:
//! - Zero-crossing rate (ZCR) analysis for voice activity detection
//! - Spectral centroid calculation for brightness assessment
//! - Spectral rolloff for frequency content analysis
//! - Signal-to-noise ratio (SNR) estimation
//! - Crest factor for dynamic range measurement
//! - Silence detection with configurable parameters
//! - Audio quality metrics for synthesis evaluation
//!
//! These features are useful for:
//! - Quality assessment of synthesized speech
//! - Voice activity detection in audio streams
//! - Audio fingerprinting and classification
//! - Real-time audio processing and effects
//! - Research and development of TTS systems

use voirs_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== VoiRS SDK Advanced Audio Analysis Example ===\n");

    // Create a pipeline for synthesis
    let pipeline = VoirsPipelineBuilder::new().build().await?;

    // Synthesize test audio
    println!("1. Synthesizing test audio...");
    let text = "This is a test of advanced audio analysis features. \
                The quick brown fox jumps over the lazy dog. \
                We can analyze various acoustic properties of the synthesized speech.";

    let audio = pipeline.synthesize(text).await?;

    println!(
        "   Synthesized {} samples at {} Hz",
        audio.len(),
        audio.sample_rate()
    );
    println!("   Duration: {:.2} seconds\n", audio.duration());

    // Basic audio metrics
    println!("2. Basic Audio Metrics:");
    println!(
        "   Peak amplitude: {:.4} ({:.2} dB)",
        audio.metadata().peak_amplitude,
        audio.peak_db()
    );
    println!(
        "   RMS amplitude: {:.4} ({:.2} dB)",
        audio.metadata().rms_amplitude,
        audio.rms_db()
    );
    println!("   Channel count: {}", audio.channels());
    println!();

    // Zero-Crossing Rate Analysis
    println!("3. Zero-Crossing Rate (ZCR) Analysis:");
    let zcr = audio.zero_crossing_rate();
    println!("   ZCR: {:.6}", zcr);
    println!(
        "   Interpretation: {} content",
        if zcr > 0.5 {
            "High frequency"
        } else {
            "Low frequency"
        }
    );
    println!("   Use case: Voice activity detection, speech/music classification");
    println!();

    // Spectral Analysis
    println!("4. Spectral Analysis:");

    let centroid = audio.spectral_centroid();
    println!("   Spectral centroid: {:.2} Hz", centroid);
    println!(
        "   Interpretation: {} voice",
        if centroid > 2000.0 { "Bright" } else { "Dark" }
    );
    println!("   Use case: Timbre characterization, speaker identification");
    println!();

    let rolloff_85 = audio.spectral_rolloff(0.85);
    let rolloff_95 = audio.spectral_rolloff(0.95);
    println!("   Spectral rolloff (85%): {:.2} Hz", rolloff_85);
    println!("   Spectral rolloff (95%): {:.2} Hz", rolloff_95);
    println!("   Bandwidth: {:.2} Hz", rolloff_95 - rolloff_85);
    println!("   Use case: Frequency content analysis, audio classification");
    println!();

    // Signal Quality Metrics
    println!("5. Signal Quality Metrics:");

    let snr = audio.signal_to_noise_ratio();
    println!("   Signal-to-Noise Ratio (SNR): {:.2} dB", snr);
    println!(
        "   Quality: {}",
        match snr {
            s if s > 40.0 => "Excellent (Studio quality)",
            s if s > 30.0 => "Very Good (Broadcast quality)",
            s if s > 20.0 => "Good (Acceptable quality)",
            s if s > 10.0 => "Fair (Noticeable noise)",
            _ => "Poor (High noise floor)",
        }
    );
    println!("   Use case: Quality assessment, noise floor measurement");
    println!();

    let crest = audio.crest_factor();
    println!("   Crest factor: {:.2} dB", crest);
    println!(
        "   Dynamic range: {}",
        match crest {
            c if c > 15.0 => "High (Very dynamic)",
            c if c > 10.0 => "Medium (Normal speech)",
            c if c > 5.0 => "Low (Compressed/limited)",
            _ => "Very low (Heavy processing)",
        }
    );
    println!("   Use case: Dynamic range assessment, compression detection");
    println!();

    // Clipping Detection
    println!("6. Clipping Detection:");
    let has_clipping = audio.has_clipping();
    let clipped_samples = audio.count_clipped_samples();

    if has_clipping {
        let clip_percentage = (clipped_samples as f32 / audio.len() as f32) * 100.0;
        println!("   ⚠ Warning: Audio contains clipping!");
        println!(
            "   Clipped samples: {} ({:.2}%)",
            clipped_samples, clip_percentage
        );
        println!("   Recommendation: Reduce synthesis volume or apply normalization");
    } else {
        println!("   ✓ No clipping detected (audio within [-1.0, 1.0] range)");
        println!(
            "   Headroom: {:.2} dB",
            20.0 * (1.0 / audio.metadata().peak_amplitude).log10()
        );
    }
    println!();

    // Silence Detection
    println!("7. Silence Detection:");
    let silences = audio.detect_silence(-40.0, 0.1);

    if silences.is_empty() {
        println!("   No silence segments detected (threshold: -40 dB, min duration: 0.1s)");
    } else {
        println!("   Detected {} silence segment(s):", silences.len());
        for (i, (start, end)) in silences.iter().enumerate() {
            let duration = end - start;
            println!(
                "     {}. {:.3}s to {:.3}s (duration: {:.3}s)",
                i + 1,
                start,
                end,
                duration
            );
        }

        let total_silence: f32 = silences.iter().map(|(s, e)| e - s).sum();
        let silence_percentage = (total_silence / audio.duration()) * 100.0;
        println!(
            "   Total silence: {:.3}s ({:.1}% of audio)",
            total_silence, silence_percentage
        );
    }
    println!();

    // Batch Analysis Example
    println!("8. Batch Analysis Example (Multiple Texts):");
    let test_texts = vec![
        "Hello world.",
        "This is a longer sentence with more complexity and varied intonation patterns.",
        "Short.",
        "The quick brown fox jumps over the lazy dog multiple times continuously without stopping.",
    ];

    println!(
        "   Analyzing {} different text samples:\n",
        test_texts.len()
    );
    println!(
        "   {:50} | {:8} | {:12} | {:10} | {:10}",
        "Text", "Duration", "Centroid(Hz)", "SNR(dB)", "ZCR"
    );
    println!(
        "   {:-<50}-+-{:-<8}-+-{:-<12}-+-{:-<10}-+-{:-<10}",
        "", "", "", "", ""
    );

    for text in test_texts {
        let audio = pipeline.synthesize(text).await?;
        let centroid = audio.spectral_centroid();
        let snr = audio.signal_to_noise_ratio();
        let zcr = audio.zero_crossing_rate();

        let display_text = if text.len() > 47 {
            format!("{}...", &text[..47])
        } else {
            text.to_string()
        };

        println!(
            "   {:50} | {:6.2}s | {:10.1} | {:8.2} | {:8.6}",
            display_text,
            audio.duration(),
            centroid,
            snr,
            zcr
        );
    }
    println!();

    // Quality Assessment Summary
    println!("9. Quality Assessment Summary:");
    println!(
        "   Overall audio quality: {}",
        if snr > 30.0 && !has_clipping && crest > 8.0 {
            "✓ EXCELLENT - Ready for production use"
        } else if snr > 20.0 && !has_clipping {
            "✓ GOOD - Suitable for most applications"
        } else if has_clipping {
            "⚠ NEEDS ATTENTION - Clipping detected"
        } else {
            "⚠ FAIR - May benefit from enhancement"
        }
    );
    println!();

    // Save analyzed audio for verification
    println!("10. Saving Analyzed Audio:");
    let output_file = std::env::temp_dir().join("voirs_analysis_test.wav");
    audio.save_wav(&output_file)?;
    println!("   Saved to: {}", output_file.display());
    println!("   You can verify the analysis by listening to the audio file.");
    println!();

    println!("=== Analysis Complete ===");
    println!("\nKey Takeaways:");
    println!("- Zero-crossing rate helps detect voiced/unvoiced segments");
    println!("- Spectral centroid indicates voice brightness and timbre");
    println!("- SNR measures overall signal quality vs noise floor");
    println!("- Crest factor reveals dynamic range characteristics");
    println!("- Silence detection helps identify pauses and non-speech");
    println!("\nThese metrics are invaluable for:");
    println!("- Quality assurance in TTS systems");
    println!("- Automatic voice quality evaluation");
    println!("- Research and development");
    println!("- Production monitoring and debugging");

    Ok(())
}
