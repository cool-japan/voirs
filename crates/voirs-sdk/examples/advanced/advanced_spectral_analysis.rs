//! Advanced Spectral Analysis Example
//!
//! This example demonstrates the advanced audio analysis capabilities of the VoiRS SDK,
//! including MFCC extraction, pitch detection, spectral flux, and formant estimation.
//!
//! These features are essential for:
//! - Speech analysis and recognition
//! - Voice quality assessment
//! - Speaker identification
//! - Emotion detection
//! - Audio forensics
//!
//! Run with:
//! ```bash
//! cargo run --example advanced_spectral_analysis --all-features
//! ```

use voirs_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== VoiRS SDK - Advanced Spectral Analysis Example ===\n");

    // Create a pipeline for generating test audio
    let pipeline = VoirsPipelineBuilder::new()
        .with_quality(QualityLevel::High)
        .build()
        .await?;

    println!("1. MFCC (Mel-Frequency Cepstral Coefficients) Analysis");
    println!("--------------------------------------------------------");

    // Synthesize a short phrase
    let text = "Hello world, this is a test of advanced audio analysis.";
    let audio = pipeline.synthesize(text).await?;

    println!("Synthesized text: \"{}\"", text);
    println!("Audio duration: {:.2} seconds", audio.duration());
    println!("Sample rate: {} Hz", audio.sample_rate());

    // Extract MFCC features
    let num_coeffs = 13;
    let num_filters = 26;
    let fft_size = 512;

    let mfccs = audio.mfcc(num_coeffs, num_filters, fft_size);

    if !mfccs.is_empty() {
        println!("\nMFCC Coefficients (first {} coefficients):", num_coeffs);
        for (i, &coeff) in mfccs.iter().enumerate() {
            println!("  C{}: {:.4}", i, coeff);
        }

        println!("\nMFCC Analysis:");
        println!("  C0 (Energy): {:.4}", mfccs[0]);
        println!("  C1-C2 (Spectral tilt): {:.4}, {:.4}", mfccs[1], mfccs[2]);
        println!("  Higher coefficients represent fine spectral details");
    } else {
        println!("Warning: MFCC extraction failed (insufficient samples or invalid FFT size)");
    }

    println!("\n2. Pitch Detection (Autocorrelation Method)");
    println!("--------------------------------------------");

    // Detect pitch using autocorrelation
    let min_freq = 80.0; // Typical male voice lower bound
    let max_freq = 400.0; // Typical female voice upper bound

    let detected_pitch = audio.detect_pitch_autocorr(min_freq, max_freq);

    if detected_pitch > 0.0 {
        println!("Detected pitch: {:.2} Hz", detected_pitch);
        println!("Note: This represents the fundamental frequency (F0)");

        // Classify voice type based on pitch
        if detected_pitch < 120.0 {
            println!("Voice type: Likely male voice");
        } else if detected_pitch < 200.0 {
            println!("Voice type: Likely female voice or high male voice");
        } else {
            println!("Voice type: Likely female voice or child voice");
        }
    } else {
        println!("No pitch detected (may be noise or insufficient signal)");
    }

    println!("\n3. Spectral Flux Analysis");
    println!("-------------------------");

    // For spectral flux, we need two buffers to compare
    let text1 = "A stable tone.";
    let text2 = "A sudden change!";

    let audio1 = pipeline.synthesize(text1).await?;
    let audio2 = pipeline.synthesize(text2).await?;

    let flux = audio2.spectral_flux(Some(&audio1), 512);

    println!("Spectral flux between two phrases: {:.6}", flux);
    println!("Interpretation:");
    if flux > 0.1 {
        println!("  High flux - significant spectral change detected");
        println!("  Use case: Onset detection, event segmentation");
    } else {
        println!("  Low flux - minimal spectral change");
        println!("  Use case: Detecting stable regions in audio");
    }

    // Self-comparison (should be near zero)
    let self_flux = audio1.spectral_flux(Some(&audio1), 512);
    println!("\nSelf-flux (same audio): {:.6}", self_flux);
    println!("(Should be near zero for identical audio)");

    println!("\n4. Formant Estimation");
    println!("---------------------");

    // Generate vowel-like sounds for formant analysis
    let vowel_texts = vec![
        ("ah", "father"), // /ɑ/ - low, back vowel
        ("ee", "see"),    // /i/ - high, front vowel
        ("oo", "food"),   // /u/ - high, back vowel
    ];

    for (vowel, word) in vowel_texts {
        let vowel_audio = pipeline.synthesize(word).await?;
        let formants = vowel_audio.estimate_formants(4);

        println!("\nVowel sound '{}' (from \"{}\"):", vowel, word);
        if formants.is_empty() {
            println!("  No formants detected");
        } else {
            for (i, &freq) in formants.iter().enumerate() {
                println!("  F{}: {:.1} Hz", i + 1, freq);
            }

            // Provide linguistic interpretation
            if formants.len() >= 2 {
                println!("  Linguistic interpretation:");
                if formants[0] < 500.0 && formants[1] > 2000.0 {
                    println!("    High front vowel (like 'ee')");
                } else if formants[0] > 700.0 && formants[1] < 1500.0 {
                    println!("    Low back vowel (like 'ah' or 'oo')");
                } else {
                    println!("    Mid or mixed vowel");
                }
            }
        }
    }

    println!("\n5. Combined Analysis - Comprehensive Voice Profile");
    println!("---------------------------------------------------");

    let analysis_text = "The quick brown fox jumps over the lazy dog.";
    let analysis_audio = pipeline.synthesize(analysis_text).await?;

    println!("Analyzing: \"{}\"", analysis_text);
    println!("\nComprehensive Audio Profile:");

    // Basic metrics
    println!("  Duration: {:.2}s", analysis_audio.duration());
    println!("  Peak amplitude: {:.3}", analysis_audio.peak());
    println!("  RMS amplitude: {:.3}", analysis_audio.rms());
    println!("  Peak dB: {:.2}", analysis_audio.peak_db());
    println!("  RMS dB: {:.2}", analysis_audio.rms_db());

    // Advanced spectral metrics
    let zcr = analysis_audio.zero_crossing_rate();
    println!("\n  Zero-crossing rate: {:.4}", zcr);
    println!("    (Higher ZCR indicates brighter, noisier sound)");

    let centroid = analysis_audio.spectral_centroid();
    println!("\n  Spectral centroid: {:.1} Hz", centroid);
    println!("    (Center of mass of spectrum - brightness indicator)");

    let rolloff = analysis_audio.spectral_rolloff(0.85);
    println!("\n  Spectral rolloff (85%): {:.1} Hz", rolloff);
    println!("    (Frequency below which 85% of energy is contained)");

    let snr = analysis_audio.signal_to_noise_ratio();
    println!("\n  Signal-to-Noise Ratio: {:.2} dB", snr);

    let crest = analysis_audio.crest_factor();
    println!("  Crest factor: {:.2} dB", crest);
    println!("    (Peak-to-RMS ratio - dynamic range indicator)");

    // Advanced features
    let pitch = analysis_audio.detect_pitch_autocorr(80.0, 400.0);
    if pitch > 0.0 {
        println!("\n  Detected pitch (F0): {:.1} Hz", pitch);
    }

    let formants = analysis_audio.estimate_formants(3);
    if !formants.is_empty() {
        println!("\n  Formants:");
        for (i, &f) in formants.iter().enumerate() {
            println!("    F{}: {:.1} Hz", i + 1, f);
        }
    }

    let mfccs = analysis_audio.mfcc(13, 26, 1024);
    if !mfccs.is_empty() {
        println!("\n  MFCC Summary:");
        println!("    Energy (C0): {:.4}", mfccs[0]);
        println!(
            "    Spectral shape (C1-C3): {:.4}, {:.4}, {:.4}",
            mfccs[1],
            mfccs[2],
            mfccs.get(3).unwrap_or(&0.0)
        );
    }

    println!("\n6. Quality Assessment");
    println!("---------------------");

    // Quality categorization based on multiple metrics
    let quality_score = if snr > 40.0 && !analysis_audio.has_clipping() {
        "Excellent"
    } else if snr > 30.0 && analysis_audio.count_clipped_samples() < 10 {
        "Good"
    } else if snr > 20.0 {
        "Fair"
    } else {
        "Poor"
    };

    println!("Overall quality: {}", quality_score);
    println!("  SNR: {:.2} dB", snr);
    println!(
        "  Clipping: {}",
        if analysis_audio.has_clipping() {
            "Yes"
        } else {
            "No"
        }
    );
    println!(
        "  Clipped samples: {}",
        analysis_audio.count_clipped_samples()
    );

    // Silence detection
    let silences = analysis_audio.detect_silence(-40.0, 0.1);
    if !silences.is_empty() {
        println!("\n  Silence segments detected: {}", silences.len());
        for (i, (start, end)) in silences.iter().enumerate().take(5) {
            println!(
                "    Silence {}: {:.2}s - {:.2}s ({:.2}s duration)",
                i + 1,
                start,
                end,
                end - start
            );
        }

        let total_silence: f32 = silences.iter().map(|(start, end)| end - start).sum();
        println!(
            "  Total silence: {:.2}s ({:.1}% of audio)",
            total_silence,
            (total_silence / analysis_audio.duration()) * 100.0
        );
    } else {
        println!("\n  No significant silence segments detected");
    }

    println!("\n=== Analysis Complete ===");
    println!("\nThese advanced features enable:");
    println!("  • Speech recognition and speaker identification");
    println!("  • Emotion detection from voice characteristics");
    println!("  • Voice quality assessment and monitoring");
    println!("  • Audio forensics and authentication");
    println!("  • Automatic content classification");
    println!("  • Real-time voice activity detection");

    Ok(())
}
