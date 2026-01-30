//! Music Information Retrieval Example
//!
//! This example demonstrates advanced music and audio analysis capabilities of the VoiRS SDK,
//! including chroma features, spectral contrast, and YIN pitch detection.
//!
//! These features are essential for:
//! - Music information retrieval
//! - Chord recognition and key detection
//! - Genre classification
//! - Instrument recognition
//! - High-accuracy pitch tracking
//! - Music similarity analysis
//!
//! Run with:
//! ```bash
//! cargo run --example music_information_retrieval --all-features
//! ```

use voirs_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== VoiRS SDK - Music Information Retrieval Example ===\n");

    // Create a pipeline for generating test audio
    let pipeline = VoirsPipelineBuilder::new()
        .with_quality(QualityLevel::High)
        .build()
        .await?;

    println!("1. Chroma Features - Pitch Class Profiles");
    println!("------------------------------------------");

    // Synthesize a musical phrase
    let text = "Music has twelve pitch classes in Western music theory.";
    let audio = pipeline.synthesize(text).await?;

    println!("Synthesized text: \"{}\"", text);
    println!("Audio duration: {:.2} seconds", audio.duration());

    // Extract chroma features (12 pitch classes)
    let chroma = audio.chroma_features(4096, 440.0);

    if !chroma.is_empty() {
        println!("\nChroma Features (12 pitch classes):");
        let pitch_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];

        for (i, &energy) in chroma.iter().enumerate() {
            let bar_length = (energy * 50.0) as usize;
            let bar = "█".repeat(bar_length);
            println!(
                "  {:>2} {:<3}: {:<50} {:.3}",
                i, pitch_names[i], bar, energy
            );
        }

        // Find dominant pitch classes
        let mut indexed_chroma: Vec<(usize, f32)> =
            chroma.iter().enumerate().map(|(i, &c)| (i, c)).collect();
        indexed_chroma.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("\nTop 3 Dominant Pitch Classes:");
        for (i, (idx, energy)) in indexed_chroma.iter().take(3).enumerate() {
            println!("  {}. {} ({:.3})", i + 1, pitch_names[*idx], energy);
        }

        println!("\nInterpretation:");
        println!("  Chroma features represent the distribution of energy across");
        println!("  the 12 pitch classes, useful for:");
        println!("  • Chord recognition");
        println!("  • Key detection");
        println!("  • Cover song identification");
        println!("  • Music similarity comparison");
    }

    println!("\n2. Spectral Contrast - Timbre Analysis");
    println!("---------------------------------------");

    let text2 = "Spectral contrast measures the difference between spectral peaks and valleys.";
    let audio2 = pipeline.synthesize(text2).await?;

    println!("Synthesized text: \"{}\"", text2);

    // Extract spectral contrast across 8 frequency bands
    let num_bands = 8;
    let contrast = audio2.spectral_contrast(4096, num_bands);

    if !contrast.is_empty() {
        println!("\nSpectral Contrast (8 frequency bands):");
        println!("  Band  Frequency Range (approx)  Contrast (dB)");
        println!("  ----  -----------------------  -------------");

        let sample_rate = audio2.sample_rate() as f32;
        let nyquist = sample_rate / 2.0;

        for (i, &contrast_db) in contrast.iter().enumerate() {
            // Approximate frequency ranges for each band (logarithmic)
            let band_start = (i as f32 / num_bands as f32) * nyquist;
            let band_end = ((i + 1) as f32 / num_bands as f32) * nyquist;

            let bar_length = ((contrast_db / 60.0) * 40.0).min(40.0) as usize;
            let bar = "▓".repeat(bar_length);

            println!(
                "  {:>4}  {:>6.0} - {:>6.0} Hz      {:>5.1} {}",
                i + 1,
                band_start,
                band_end,
                contrast_db,
                bar
            );
        }

        let avg_contrast: f32 = contrast.iter().sum::<f32>() / contrast.len() as f32;
        println!("\nAverage Spectral Contrast: {:.2} dB", avg_contrast);

        println!("\nInterpretation:");
        if avg_contrast > 20.0 {
            println!("  High contrast - Rich harmonic structure");
            println!("  Suggests: Musical instruments or complex tones");
        } else if avg_contrast > 10.0 {
            println!("  Moderate contrast - Varied spectral content");
            println!("  Suggests: Speech or mixed audio");
        } else {
            println!("  Low contrast - Smooth spectrum");
            println!("  Suggests: Noise or simple waveforms");
        }

        println!("\nUse cases:");
        println!("  • Music genre classification");
        println!("  • Instrument recognition");
        println!("  • Audio texture analysis");
        println!("  • Timbre characterization");
    }

    println!("\n3. YIN Algorithm - Advanced Pitch Detection");
    println!("-------------------------------------------");

    // Test with multiple pitch ranges
    let pitch_tests = vec![
        ("Low male voice", 80.0, 200.0),
        ("High female voice", 150.0, 400.0),
        ("Child voice", 200.0, 500.0),
    ];

    for (voice_type, min_freq, max_freq) in pitch_tests {
        let test_text = format!("Testing {} pitch detection.", voice_type.to_lowercase());
        let test_audio = pipeline.synthesize(&test_text).await?;

        println!(
            "\n{} ({:.0}-{:.0} Hz range):",
            voice_type, min_freq, max_freq
        );

        // Compare YIN with autocorrelation
        let pitch_yin = test_audio.detect_pitch_yin(min_freq, max_freq, 0.15);
        let pitch_autocorr = test_audio.detect_pitch_autocorr(min_freq, max_freq);

        println!("  Text: \"{}\"", test_text);

        if pitch_yin > 0.0 {
            println!("  YIN Algorithm:      {:.2} Hz", pitch_yin);
        } else {
            println!("  YIN Algorithm:      No pitch detected");
        }

        if pitch_autocorr > 0.0 {
            println!("  Autocorrelation:    {:.2} Hz", pitch_autocorr);
        } else {
            println!("  Autocorrelation:    No pitch detected");
        }

        if pitch_yin > 0.0 && pitch_autocorr > 0.0 {
            let diff = (pitch_yin - pitch_autocorr).abs();
            let diff_pct = (diff / pitch_autocorr) * 100.0;
            println!("  Difference:         {:.2} Hz ({:.1}%)", diff, diff_pct);

            if diff_pct < 2.0 {
                println!("  Agreement:          Excellent (within 2%)");
            } else if diff_pct < 5.0 {
                println!("  Agreement:          Good (within 5%)");
            } else {
                println!("  Agreement:          Moderate (>5% difference)");
            }
        }
    }

    println!("\nYIN Algorithm Advantages:");
    println!("  • More robust to noise than autocorrelation");
    println!("  • Better handling of complex harmonic structures");
    println!("  • Sub-sample accuracy via parabolic interpolation");
    println!("  • Lower octave errors");
    println!("  • Configurable threshold for sensitivity control");

    println!("\n4. Combined Music Analysis - Comprehensive Profile");
    println!("--------------------------------------------------");

    let music_text = "Music combines rhythm, melody, harmony, and timbre.";
    let music_audio = pipeline.synthesize(music_text).await?;

    println!("Analyzing: \"{}\"", music_text);
    println!("\nComprehensive Music Profile:");

    // Basic metrics
    println!("\n  Duration: {:.2}s", music_audio.duration());
    println!("  Sample rate: {} Hz", music_audio.sample_rate());
    println!("  Peak amplitude: {:.3}", music_audio.peak());
    println!("  RMS amplitude: {:.3}", music_audio.rms());

    // Advanced spectral metrics
    let zcr = music_audio.zero_crossing_rate();
    println!("\n  Zero-crossing rate: {:.4}", zcr);

    let centroid = music_audio.spectral_centroid();
    println!("  Spectral centroid: {:.1} Hz", centroid);

    let rolloff = music_audio.spectral_rolloff(0.85);
    println!("  Spectral rolloff (85%): {:.1} Hz", rolloff);

    // Pitch analysis
    let pitch_yin = music_audio.detect_pitch_yin(80.0, 400.0, 0.15);
    if pitch_yin > 0.0 {
        println!("\n  Detected pitch (YIN): {:.1} Hz", pitch_yin);

        // Classify pitch range
        if pitch_yin < 120.0 {
            println!("  Pitch classification: Low (bass/male)");
        } else if pitch_yin < 200.0 {
            println!("  Pitch classification: Medium (alto/high male)");
        } else if pitch_yin < 300.0 {
            println!("  Pitch classification: High (soprano/female)");
        } else {
            println!("  Pitch classification: Very high (child/soprano)");
        }
    }

    // Chroma analysis
    let chroma = music_audio.chroma_features(4096, 440.0);
    if !chroma.is_empty() {
        let chroma_energy: f32 = chroma.iter().sum();
        let max_chroma = chroma.iter().cloned().fold(0.0_f32, f32::max);
        let active_classes = chroma.iter().filter(|&&c| c > 0.2).count();

        println!("\n  Chroma Features:");
        println!("    Total energy: {:.3}", chroma_energy);
        println!("    Peak energy: {:.3}", max_chroma);
        println!("    Active pitch classes: {}", active_classes);
    }

    // Spectral contrast
    let contrast = music_audio.spectral_contrast(4096, 6);
    if !contrast.is_empty() {
        let avg_contrast: f32 = contrast.iter().sum::<f32>() / contrast.len() as f32;
        let max_contrast = contrast.iter().cloned().fold(0.0_f32, f32::max);

        println!("\n  Spectral Contrast:");
        println!("    Average contrast: {:.2} dB", avg_contrast);
        println!("    Maximum contrast: {:.2} dB", max_contrast);

        if avg_contrast > 15.0 {
            println!("    Texture: Rich and complex");
        } else if avg_contrast > 8.0 {
            println!("    Texture: Moderate complexity");
        } else {
            println!("    Texture: Smooth and simple");
        }
    }

    println!("\n5. Application Examples");
    println!("-----------------------");

    println!("\nMusic Genre Classification:");
    println!("  1. Extract chroma features for harmonic content");
    println!("  2. Extract spectral contrast for timbre");
    println!("  3. Extract tempo and rhythm features");
    println!("  4. Train classifier on labeled dataset");

    println!("\nChord Recognition:");
    println!("  1. Compute chroma features in sliding windows");
    println!("  2. Match against known chord templates");
    println!("  3. Use temporal smoothing for stability");
    println!("  4. Output chord progression");

    println!("\nCover Song Detection:");
    println!("  1. Extract chroma features (octave-invariant)");
    println!("  2. Compute chroma similarity matrix");
    println!("  3. Use dynamic time warping for alignment");
    println!("  4. Calculate similarity score");

    println!("\nPitch Tracking for Music Transcription:");
    println!("  1. Use YIN algorithm for robust F0 estimation");
    println!("  2. Convert pitch to MIDI note numbers");
    println!("  3. Detect note onsets with spectral flux");
    println!("  4. Generate musical score");

    println!("\n=== Analysis Complete ===");
    println!("\nThese advanced features enable:");
    println!("  • Music information retrieval systems");
    println!("  • Automatic music transcription");
    println!("  • Genre and mood classification");
    println!("  • Instrument and timbre recognition");
    println!("  • Cover song identification");
    println!("  • Music similarity search");
    println!("  • Automatic chord recognition");
    println!("  • High-accuracy pitch tracking");

    Ok(())
}
