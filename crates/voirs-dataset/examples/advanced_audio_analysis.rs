//! Advanced Audio Analysis Example
//!
//! This example demonstrates the advanced audio analysis features
//! including perceptual loudness, spectral features, and harmonic analysis.

use std::f32::consts::PI;
use voirs_dataset::audio::advanced_analysis::{AdvancedAnalysisConfig, AdvancedAudioAnalyzer};
use voirs_dataset::AudioData;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 VoiRS Advanced Audio Analysis Example");
    println!("========================================");

    // Create test audio signals
    println!("\n1. Creating test audio signals...");
    let signals = create_test_signals();

    // Initialize advanced analyzer
    let config = AdvancedAnalysisConfig {
        enable_loudness: true,
        enable_bark_scale: true,
        enable_mel_scale: true,
        enable_chroma: true,
        window_size: 2048,
        hop_size: 512,
        min_frequency: 20.0,
        max_frequency: 8000.0,
    };

    let mut analyzer = AdvancedAudioAnalyzer::new(config)?;
    println!("✅ Advanced analyzer initialized");

    // Analyze each test signal
    println!("\n2. Analyzing test signals...");
    for (name, audio) in signals {
        println!("\n📊 Analyzing: {name}");
        let features = analyzer.analyze(&audio)?;

        // Display perceptual loudness features
        println!("   Loudness:");
        println!("     LUFS: {:.2} dB", features.loudness_lufs);
        println!("     Range: {:.2} LU", features.loudness_range);
        println!("     True Peak: {:.2} dBTP", features.true_peak_dbtp);

        // Display spectral features summary
        if !features.bark_features.is_empty() {
            let bark_energy: f32 = features.bark_features.iter().sum();
            println!(
                "   Bark Features: {} bands, total energy: {:.4}",
                features.bark_features.len(),
                bark_energy
            );
        }

        if !features.mel_features.is_empty() {
            let mel_mean: f32 =
                features.mel_features.iter().sum::<f32>() / features.mel_features.len() as f32;
            println!(
                "   Mel Features: {} bands, mean: {:.4}",
                features.mel_features.len(),
                mel_mean
            );
        }

        // Display harmonic features
        if !features.chroma_features.is_empty() {
            let dominant_chroma = features
                .chroma_features
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            let note_names = [
                "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
            ];
            println!(
                "   Chroma Features: dominant note: {}",
                note_names[dominant_chroma]
            );
        }

        // Display spectral contrast
        if !features.spectral_contrast.is_empty() {
            let avg_contrast: f32 = features.spectral_contrast.iter().sum::<f32>()
                / features.spectral_contrast.len() as f32;
            println!(
                "   Spectral Contrast: {:.4} (avg across {} bands)",
                avg_contrast,
                features.spectral_contrast.len()
            );
        }

        // Display temporal features
        println!("   Temporal Features:");
        println!(
            "     Tempo: {:.1} BPM",
            features.temporal_features.tempo_bpm
        );
        println!(
            "     Onset Density: {:.2} onsets/sec",
            features.temporal_features.onset_density
        );

        // Display overall quality
        println!(
            "   Overall Quality: {:.2} ({}/10)",
            features.perceptual_quality,
            (features.perceptual_quality * 10.0).round() as i32
        );

        // Quality interpretation
        let quality_desc = match features.perceptual_quality {
            q if q >= 0.8 => "Excellent",
            q if q >= 0.6 => "Good",
            q if q >= 0.4 => "Fair",
            q if q >= 0.2 => "Poor",
            _ => "Very Poor",
        };
        println!("     Quality Rating: {quality_desc}");
    }

    // Demonstrate feature comparison
    println!("\n3. Feature Comparison Analysis...");
    demonstrate_feature_comparison(&mut analyzer).await?;

    // Demonstrate real-time analysis
    println!("\n4. Simulated Real-time Analysis...");
    demonstrate_realtime_analysis(&mut analyzer).await?;

    println!("\n✅ Advanced audio analysis complete!");
    Ok(())
}

/// Create various test audio signals for analysis
fn create_test_signals() -> Vec<(String, AudioData)> {
    let sample_rate = 22050;
    let duration = 2.0; // 2 seconds
    let n_samples = (sample_rate as f32 * duration) as usize;
    let mut signals = Vec::new();

    // 1. Pure sine wave (440 Hz - A4)
    let sine_wave: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * PI * 440.0 * t).sin() * 0.5
        })
        .collect();
    signals.push((
        "Pure Sine Wave (440 Hz)".to_string(),
        AudioData::new(sine_wave, sample_rate, 1),
    ));

    // 2. Harmonic complex tone (fundamental + harmonics)
    let harmonic_tone: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let fundamental = 220.0;
            let mut sample = 0.0;
            // Add first 5 harmonics with decreasing amplitude
            for harmonic in 1..=5 {
                let freq = fundamental * harmonic as f32;
                let amplitude = 0.5 / harmonic as f32;
                sample += (2.0 * PI * freq * t).sin() * amplitude;
            }
            sample * 0.3 // Scale down overall amplitude
        })
        .collect();
    signals.push((
        "Harmonic Complex Tone".to_string(),
        AudioData::new(harmonic_tone, sample_rate, 1),
    ));

    // 3. Major chord (C-E-G)
    let chord: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let c4 = 261.63; // C4
            let e4 = 329.63; // E4
            let g4 = 392.00; // G4

            let c_wave = (2.0 * PI * c4 * t).sin() * 0.33;
            let e_wave = (2.0 * PI * e4 * t).sin() * 0.33;
            let g_wave = (2.0 * PI * g4 * t).sin() * 0.33;

            (c_wave + e_wave + g_wave) * 0.5
        })
        .collect();
    signals.push((
        "Major Chord (C-E-G)".to_string(),
        AudioData::new(chord, sample_rate, 1),
    ));

    // 4. White noise
    let white_noise: Vec<f32> = (0..n_samples)
        .map(|_| (rand::random::<f32>() - 0.5) * 0.3)
        .collect();
    signals.push((
        "White Noise".to_string(),
        AudioData::new(white_noise, sample_rate, 1),
    ));

    // 5. Chirp (frequency sweep)
    let chirp: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let start_freq = 200.0;
            let end_freq = 2000.0;
            let freq = start_freq + (end_freq - start_freq) * t / duration;
            (2.0 * PI * freq * t).sin() * 0.5
        })
        .collect();
    signals.push((
        "Frequency Sweep (200-2000 Hz)".to_string(),
        AudioData::new(chirp, sample_rate, 1),
    ));

    // 6. Simulated speech-like signal (formants)
    let speech_like: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let f0 = 120.0; // Fundamental frequency

            // Add formants (simplified)
            let f1 = 800.0; // First formant
            let f2 = 1200.0; // Second formant
            let f3 = 2400.0; // Third formant

            let fundamental = (2.0 * PI * f0 * t).sin() * 0.4;
            let formant1 = (2.0 * PI * f1 * t).sin() * 0.2;
            let formant2 = (2.0 * PI * f2 * t).sin() * 0.15;
            let formant3 = (2.0 * PI * f3 * t).sin() * 0.1;

            // Add some amplitude modulation for speech-like characteristics
            let am = (2.0 * PI * 5.0 * t).sin() * 0.1 + 1.0;

            (fundamental + formant1 + formant2 + formant3) * am * 0.3
        })
        .collect();
    signals.push((
        "Speech-like Signal".to_string(),
        AudioData::new(speech_like, sample_rate, 1),
    ));

    signals
}

/// Demonstrate feature comparison between different signals
async fn demonstrate_feature_comparison(
    analyzer: &mut AdvancedAudioAnalyzer,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing harmonic content between signals...");

    // Create a pure tone and a chord
    let sample_rate = 22050;
    let duration = 1.0;
    let n_samples = (sample_rate as f32 * duration) as usize;

    // Pure tone
    let pure_tone: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * PI * 440.0 * t).sin() * 0.5
        })
        .collect();
    let pure_audio = AudioData::new(pure_tone, sample_rate, 1);

    // Complex chord
    let chord: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let freqs = [261.63, 329.63, 392.00, 523.25]; // C major chord
            freqs
                .iter()
                .map(|&freq| (2.0 * PI * freq * t).sin() * 0.25)
                .sum()
        })
        .collect();
    let chord_audio = AudioData::new(chord, sample_rate, 1);

    let pure_features = analyzer.analyze(&pure_audio)?;
    let chord_features = analyzer.analyze(&chord_audio)?;

    println!("   Pure Tone vs Chord:");
    println!(
        "     Pure Tone Quality: {:.3}",
        pure_features.perceptual_quality
    );
    println!(
        "     Chord Quality: {:.3}",
        chord_features.perceptual_quality
    );

    if !pure_features.chroma_features.is_empty() && !chord_features.chroma_features.is_empty() {
        let pure_chroma_energy: f32 = pure_features.chroma_features.iter().sum();
        let chord_chroma_energy: f32 = chord_features.chroma_features.iter().sum();
        println!("     Pure Tone Harmonic Energy: {pure_chroma_energy:.3}");
        println!("     Chord Harmonic Energy: {chord_chroma_energy:.3}");

        if chord_chroma_energy > pure_chroma_energy {
            println!("     ✅ Chord shows richer harmonic content as expected");
        }
    }

    Ok(())
}

/// Demonstrate real-time style analysis on streaming audio
async fn demonstrate_realtime_analysis(
    analyzer: &mut AdvancedAudioAnalyzer,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Simulating real-time audio analysis...");

    let sample_rate = 22050;
    let chunk_duration = 0.5; // 500ms chunks
    let chunk_size = (sample_rate as f32 * chunk_duration) as usize;
    let total_chunks = 6; // 3 seconds total

    for chunk_idx in 0..total_chunks {
        // Generate different signal types for each chunk
        let chunk_audio = match chunk_idx % 3 {
            0 => {
                // Low frequency content
                let samples: Vec<f32> = (0..chunk_size)
                    .map(|i| {
                        let t = i as f32 / sample_rate as f32;
                        (2.0 * PI * 200.0 * t).sin() * 0.5
                    })
                    .collect();
                AudioData::new(samples, sample_rate, 1)
            }
            1 => {
                // Mid frequency content
                let samples: Vec<f32> = (0..chunk_size)
                    .map(|i| {
                        let t = i as f32 / sample_rate as f32;
                        (2.0 * PI * 800.0 * t).sin() * 0.5
                    })
                    .collect();
                AudioData::new(samples, sample_rate, 1)
            }
            _ => {
                // High frequency content
                let samples: Vec<f32> = (0..chunk_size)
                    .map(|i| {
                        let t = i as f32 / sample_rate as f32;
                        (2.0 * PI * 2000.0 * t).sin() * 0.5
                    })
                    .collect();
                AudioData::new(samples, sample_rate, 1)
            }
        };

        let features = analyzer.analyze(&chunk_audio)?;

        println!(
            "   Chunk {} ({:.1}s): LUFS={:.1}, Quality={:.2}, Tempo={:.0} BPM",
            chunk_idx + 1,
            (chunk_idx as f32 + 1.0) * chunk_duration,
            features.loudness_lufs,
            features.perceptual_quality,
            features.temporal_features.tempo_bpm
        );

        // Simulate real-time delay
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    println!("   ✅ Real-time analysis simulation complete");
    Ok(())
}
