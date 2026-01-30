//! Example demonstrating advanced perceptual quality metrics
//!
//! This example shows how to use the advanced perceptual quality metrics
//! to evaluate speech synthesis quality using industry-standard metrics.

use voirs_dataset::quality::perceptual::{
    PerceptualQualityAnalyzer, PerceptualQualityConfig, WindowType,
};
use voirs_dataset::AudioData;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Perceptual Quality Metrics Example ===\n");

    // Create configuration for perceptual analysis
    let config = PerceptualQualityConfig {
        fft_size: 2048,
        hop_size: 512,
        num_mfcc: 13,
        num_mel_bins: 40,
        min_freq: 80.0,
        max_freq: 8000.0,
        pitch_min_f0: 60.0,
        pitch_max_f0: 400.0,
        window_type: WindowType::Hann,
        enable_all: true,
    };

    let analyzer = PerceptualQualityAnalyzer::new(config);

    // Generate reference audio (clean sine wave representing voiced speech)
    println!("1. Generating reference audio (simulated clean speech)...");
    let sample_rate = 22050;
    let duration_secs = 2.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;

    // Simulate voiced speech with fundamental frequency around 150 Hz
    let f0 = 150.0;
    let reference_samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Fundamental + harmonics (more realistic voice simulation)
            let fundamental = (2.0 * std::f32::consts::PI * f0 * t).sin();
            let harmonic2 = 0.5 * (2.0 * std::f32::consts::PI * 2.0 * f0 * t).sin();
            let harmonic3 = 0.3 * (2.0 * std::f32::consts::PI * 3.0 * f0 * t).sin();
            0.5 * (fundamental + harmonic2 + harmonic3)
        })
        .collect();

    let reference = AudioData::new(reference_samples.clone(), sample_rate, 1);

    // Generate synthesized audio (with slight degradation)
    println!("2. Generating synthesized audio (with slight quality degradation)...");
    let synthesized_samples: Vec<f32> = reference_samples
        .iter()
        .enumerate()
        .map(|(i, &sample)| {
            // Add slight pitch jitter
            let jitter = 0.02 * ((i as f32 * 0.01).sin());
            // Add slight amplitude shimmer
            let shimmer = 1.0 + 0.03 * ((i as f32 * 0.02).cos());
            // Add small amount of noise
            let noise = 0.01 * ((i as f32 * 0.1).sin());
            sample * shimmer * (1.0 + jitter) + noise
        })
        .collect();

    let synthesized = AudioData::new(synthesized_samples, sample_rate, 1);

    // Analyze reference audio quality
    println!("\n3. Analyzing reference audio...");
    let ref_metrics = analyzer.analyze(&reference)?;

    println!("   Reference Audio Metrics:");
    if let Some(jitter) = ref_metrics.jitter_percent {
        println!("   - Jitter: {:.3}% (target: <1%)", jitter);
    }
    if let Some(shimmer) = ref_metrics.shimmer_percent {
        println!("   - Shimmer: {:.3}% (target: <3%)", shimmer);
    }
    if let Some(hnr) = ref_metrics.hnr {
        println!("   - HNR: {:.2} dB (target: >15 dB)", hnr);
    }
    if let Some(cpp) = ref_metrics.cpp {
        println!("   - CPP: {:.2} dB (target: >10 dB)", cpp);
    }
    println!(
        "   - Overall Score: {:.1}/100",
        ref_metrics.overall_perceptual_score
    );

    // Analyze synthesized audio quality
    println!("\n4. Analyzing synthesized audio...");
    let syn_metrics = analyzer.analyze(&synthesized)?;

    println!("   Synthesized Audio Metrics:");
    if let Some(jitter) = syn_metrics.jitter_percent {
        println!("   - Jitter: {:.3}% (target: <1%)", jitter);
    }
    if let Some(shimmer) = syn_metrics.shimmer_percent {
        println!("   - Shimmer: {:.3}% (target: <3%)", shimmer);
    }
    if let Some(hnr) = syn_metrics.hnr {
        println!("   - HNR: {:.2} dB (target: >15 dB)", hnr);
    }
    if let Some(cpp) = syn_metrics.cpp {
        println!("   - CPP: {:.2} dB (target: >10 dB)", cpp);
    }
    println!(
        "   - Overall Score: {:.1}/100",
        syn_metrics.overall_perceptual_score
    );

    // Compare reference vs synthesized
    println!("\n5. Comparing reference vs synthesized audio...");
    let comparison_metrics = analyzer.analyze_pair(&reference, &synthesized)?;

    println!("   Comparative Metrics:");
    if let Some(mcd) = comparison_metrics.mcd {
        println!("   - MCD: {:.2} dB (target: <6 dB for good quality)", mcd);
        println!(
            "     Quality: {}",
            match mcd {
                m if m < 4.0 => "Excellent",
                m if m < 6.0 => "Good",
                m if m < 8.0 => "Fair",
                _ => "Poor",
            }
        );
    }
    if let Some(lsd) = comparison_metrics.lsd {
        println!("   - LSD: {:.3} dB (lower is better)", lsd);
    }
    if let Some(bsd) = comparison_metrics.bsd {
        println!("   - BSD: {:.3} (lower is better)", bsd);
    }
    if let Some(sc) = comparison_metrics.spectral_convergence {
        println!("   - Spectral Convergence: {:.3} (lower is better)", sc);
    }
    println!(
        "   - Overall Perceptual Score: {:.1}/100",
        comparison_metrics.overall_perceptual_score
    );

    // Quality interpretation
    println!("\n=== Quality Interpretation ===");
    println!("\nMetric Guide:");
    println!("  MCD (Mel-Cepstral Distortion):");
    println!("    - < 4.0 dB: Excellent quality");
    println!("    - 4-6 dB: Good quality (natural sounding)");
    println!("    - 6-8 dB: Fair quality (some artifacts)");
    println!("    - > 8 dB: Poor quality (noticeable degradation)");
    println!();
    println!("  Jitter (Pitch Period Perturbation):");
    println!("    - < 0.5%: Excellent voice stability");
    println!("    - 0.5-1%: Good voice stability");
    println!("    - 1-2%: Fair (slight roughness)");
    println!("    - > 2%: Poor (perceived roughness)");
    println!();
    println!("  Shimmer (Amplitude Perturbation):");
    println!("    - < 2%: Excellent amplitude stability");
    println!("    - 2-3%: Good amplitude stability");
    println!("    - 3-5%: Fair (slight breathiness)");
    println!("    - > 5%: Poor (breathy or hoarse)");
    println!();
    println!("  HNR (Harmonic-to-Noise Ratio):");
    println!("    - > 20 dB: Excellent harmonicity");
    println!("    - 15-20 dB: Good harmonicity");
    println!("    - 10-15 dB: Fair (some noise)");
    println!("    - < 10 dB: Poor (noisy or breathy)");
    println!();
    println!("  CPP (Cepstral Peak Prominence):");
    println!("    - > 15 dB: Excellent voice quality");
    println!("    - 10-15 dB: Good voice quality");
    println!("    - 5-10 dB: Fair voice quality");
    println!("    - < 5 dB: Poor voice quality");

    println!("\n=== Analysis Complete ===");

    Ok(())
}
