// ! Vocoder Model Comparison Example
//!
//! This example demonstrates:
//! - Comparing different vocoder models (HiFi-GAN variants, DiffWave, etc.)
//! - Performance benchmarking (RTF, latency, throughput)
//! - Quality assessment (basic audio metrics)
//! - Model selection guidance
//!
//! Run with: cargo run --example vocoder_comparison --features candle

use std::time::Instant;
use voirs_vocoder::{
    audio::io::{AudioEncodeConfig, AudioEncoder, AudioFileFormat},
    models::hifigan::{HiFiGanConfig, HiFiGanVariant},
    DummyVocoder, MelSpectrogram, SynthesisConfig, Vocoder,
};

#[cfg(feature = "candle")]
use candle_core::Device;

/// Vocoder benchmark results
#[derive(Debug, Clone)]
struct BenchmarkResult {
    name: String,
    variant: String,
    rtf: f32,
    latency_ms: f32,
    peak_level: f32,
    rms_level: f32,
    sample_count: usize,
    success: bool,
    error: Option<String>,
}

impl BenchmarkResult {
    fn print_summary(&self) {
        println!("\n┌─────────────────────────────────────────────");
        println!("│ {} - {}", self.name, self.variant);
        println!("├─────────────────────────────────────────────");

        if self.success {
            println!("│ ✅ Status: Success");
            println!(
                "│ ⏱️  RTF: {:.4}x {}",
                self.rtf,
                if self.rtf < 0.1 {
                    "🚀 Excellent"
                } else if self.rtf < 0.5 {
                    "✨ Good"
                } else {
                    "⚠️  Fair"
                }
            );
            println!("│ 📊 Latency: {:.2} ms", self.latency_ms);
            println!("│ 🔊 Peak: {:.4}", self.peak_level);
            println!("│ 📈 RMS: {:.4}", self.rms_level);
            println!("│ 🎵 Samples: {}", self.sample_count);
        } else {
            println!("│ ❌ Status: Failed");
            if let Some(err) = &self.error {
                println!("│ Error: {}", err);
            }
        }
        println!("└─────────────────────────────────────────────");
    }
}

/// Generate a realistic test mel spectrogram with harmonic content
fn generate_harmonic_mel(
    n_mels: usize,
    n_frames: usize,
    sample_rate: u32,
    fundamental_freq: f32,
) -> MelSpectrogram {
    let mut data = Vec::with_capacity(n_mels);
    let hop_length = 256;

    for mel_idx in 0..n_mels {
        let mut frame = Vec::with_capacity(n_frames);
        // Approximate mel to frequency conversion
        let mel_freq = 700.0 * ((mel_idx as f32 / n_mels as f32 * 45.0).exp2() - 1.0);

        for frame_idx in 0..n_frames {
            let time = frame_idx as f32 * hop_length as f32 / sample_rate as f32;

            // Generate harmonic content (fundamental + harmonics)
            let mut magnitude = 0.0;
            for harmonic in 1..=5 {
                let harmonic_freq = fundamental_freq * harmonic as f32;
                let harmonic_amplitude = 1.0 / harmonic as f32; // Decrease with harmonic number

                // Check if this mel bin is close to the harmonic frequency
                let freq_diff = (mel_freq - harmonic_freq).abs();
                if freq_diff < 200.0 {
                    let phase = 2.0 * std::f32::consts::PI * harmonic_freq * time;
                    magnitude += harmonic_amplitude * phase.sin().abs();
                }
            }

            // Convert to log scale (dB)
            let db_magnitude = if magnitude > 1e-8 {
                20.0 * magnitude.log10()
            } else {
                -80.0
            };

            // Normalize to typical mel range [-20, 20] dB
            frame.push(db_magnitude.clamp(-20.0, 20.0));
        }
        data.push(frame);
    }

    MelSpectrogram::new(data, sample_rate, hop_length)
}

/// Benchmark a vocoder model
async fn benchmark_vocoder<V: Vocoder>(
    name: &str,
    variant: &str,
    vocoder: &V,
    mel: &MelSpectrogram,
    config: &SynthesisConfig,
) -> BenchmarkResult {
    let start = Instant::now();

    match vocoder.vocode(mel, Some(config)).await {
        Ok(audio) => {
            let latency_ms = start.elapsed().as_micros() as f32 / 1000.0;
            let duration = audio.duration();
            let rtf = start.elapsed().as_secs_f32() / duration;

            // Calculate audio metrics
            let samples = audio.samples();
            let peak_level = samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
            let rms_level =
                (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

            BenchmarkResult {
                name: name.to_string(),
                variant: variant.to_string(),
                rtf,
                latency_ms,
                peak_level,
                rms_level,
                sample_count: samples.len(),
                success: true,
                error: None,
            }
        }
        Err(e) => BenchmarkResult {
            name: name.to_string(),
            variant: variant.to_string(),
            rtf: 0.0,
            latency_ms: 0.0,
            peak_level: 0.0,
            rms_level: 0.0,
            sample_count: 0,
            success: false,
            error: Some(e.to_string()),
        },
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 VoiRS Vocoder Model Comparison");
    println!("═══════════════════════════════════════════════");
    println!();

    // Configuration
    let sample_rate = 22050;
    let duration_secs = 3.0;
    let n_frames = ((duration_secs * sample_rate as f32) / 256.0) as usize;
    let n_mels = 80;
    let fundamental_freq = 200.0; // ~G3 musical note

    println!("📋 Test Configuration:");
    println!("  • Sample rate: {} Hz", sample_rate);
    println!("  • Duration: {:.1} seconds", duration_secs);
    println!("  • Mel bins: {}", n_mels);
    println!("  • Time frames: {}", n_frames);
    println!("  • Fundamental frequency: {:.0} Hz", fundamental_freq);

    // Generate test mel spectrogram
    println!("\n🎵 Generating harmonic test signal...");
    let mel = generate_harmonic_mel(n_mels, n_frames, sample_rate, fundamental_freq);
    println!("  ✓ Mel spectrogram generated");

    // Synthesis configuration
    let synthesis_config = SynthesisConfig::default();

    // Collect results
    let mut results = Vec::new();

    // Benchmark DummyVocoder (baseline)
    println!("\n🔄 Benchmarking models...");
    println!("\n[1/4] Testing DummyVocoder (Baseline)...");
    let dummy_vocoder = DummyVocoder::new();
    let result = benchmark_vocoder(
        "DummyVocoder",
        "Baseline",
        &dummy_vocoder,
        &mel,
        &synthesis_config,
    )
    .await;
    result.print_summary();
    results.push(result);

    // Note: HiFi-GAN requires model weights which may not be available
    // This example shows the structure for comparison
    #[cfg(feature = "candle")]
    {
        println!("\n[2/4] HiFi-GAN models require pre-trained weights");
        println!("  ℹ️  HiFi-GAN V1 - Highest quality (MOS 4.5+)");
        println!("  ℹ️  HiFi-GAN V2 - Balanced quality/speed");
        println!("  ℹ️  HiFi-GAN V3 - Fastest inference");
        println!("  💡 Load weights with: vocoder.load_weights(\"path/to/model\")");

        // Example of how you would benchmark HiFi-GAN if weights were available:
        /*
        use voirs_vocoder::models::hifigan::HiFiGanInference;

        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        println!("  Device: {:?}", device);

        // HiFi-GAN V1
        let config_v1 = HiFiGanConfig::v1(sample_rate);
        let hifigan_v1 = HiFiGanInference::new(config_v1, device.clone())?;
        // hifigan_v1.load_weights("path/to/hifigan_v1.safetensors")?;

        let result_v1 = benchmark_vocoder(
            "HiFi-GAN",
            "V1 (Quality)",
            &hifigan_v1,
            &mel,
            &synthesis_config,
        ).await;
        result_v1.print_summary();
        results.push(result_v1);
        */
    }

    // Print comparison summary
    println!("\n\n📊 COMPARISON SUMMARY");
    println!("═══════════════════════════════════════════════");

    // Sort by RTF (lower is better)
    results.sort_by(|a, b| a.rtf.partial_cmp(&b.rtf).unwrap());

    println!("\n┌──────────────────┬──────────┬────────────┬──────────┐");
    println!("│ Model            │ Variant  │ RTF        │ Latency  │");
    println!("├──────────────────┼──────────┼────────────┼──────────┤");

    for result in &results {
        if result.success {
            println!(
                "│ {:<16} │ {:<8} │ {:>8.4}x │ {:>6.1}ms │",
                result.name, result.variant, result.rtf, result.latency_ms
            );
        }
    }

    println!("└──────────────────┴──────────┴────────────┴──────────┘");

    // Performance tiers
    println!("\n🏆 Performance Tiers:");
    println!("  🚀 Excellent: RTF < 0.1× (10× faster than real-time)");
    println!("  ✨ Good:      RTF < 0.5× (2× faster than real-time)");
    println!("  ⚠️  Fair:      RTF < 1.0× (faster than real-time)");
    println!("  ❌ Slow:      RTF ≥ 1.0× (slower than real-time)");

    // Quality assessment
    println!("\n🎧 Audio Quality Metrics:");
    println!("┌──────────────────┬──────────┬──────────┬──────────┐");
    println!("│ Model            │ Variant  │ Peak     │ RMS      │");
    println!("├──────────────────┼──────────┼──────────┼──────────┤");

    for result in &results {
        if result.success {
            println!(
                "│ {:<16} │ {:<8} │ {:>7.4} │ {:>7.4} │",
                result.name, result.variant, result.peak_level, result.rms_level
            );
        }
    }

    println!("└──────────────────┴──────────┴──────────┴──────────┘");

    // Model recommendations
    println!("\n💡 Model Selection Guide:");
    println!("─────────────────────────────────────────────────────");
    println!("  🎯 Real-time applications → HiFi-GAN V3 (fastest)");
    println!("  🎵 High-quality synthesis → HiFi-GAN V1 or BigVGAN");
    println!("  ⚖️  Balanced use cases     → HiFi-GAN V2 or UnivNet");
    println!("  🌊 Maximum quality        → DiffWave or BigVGAN Large");
    println!("  📱 Mobile deployment      → HiFi-GAN V3 (quantized)");
    println!("  🔬 Research/development   → DummyVocoder (fast iteration)");

    // Save sample output
    if let Some(best_result) = results.iter().find(|r| r.success) {
        println!("\n💾 Saving sample audio from {}...", best_result.name);
        let dummy_vocoder = DummyVocoder::new();
        let audio = dummy_vocoder.vocode(&mel, Some(&synthesis_config)).await?;

        let encode_config = AudioEncodeConfig {
            format: AudioFileFormat::Wav,
            sample_rate: audio.sample_rate(),
            channels: audio.channels() as u16,
            bits_per_sample: 16,
            bit_rate: None,
            quality: Some(0.8),
            compression_level: Some(5),
        };

        let encoder = AudioEncoder::new(encode_config);
        let output_path = std::env::temp_dir().join("vocoder_comparison_sample.wav");
        encoder.write_to_file(&audio, &output_path)?;
        println!("  ✓ Saved to: {}", output_path.display());
    }

    println!("\n✨ Comparison complete!");
    println!("═══════════════════════════════════════════════\n");

    Ok(())
}
