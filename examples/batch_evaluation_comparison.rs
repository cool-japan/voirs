//! Batch Evaluation and Model Comparison Example
//!
//! This example demonstrates:
//! 1. Basic text-to-speech synthesis
//! 2. Simple evaluation metrics
//! 3. Basic performance comparison
//! 4. Report generation

use anyhow::Result;
use std::collections::HashMap;
use std::time::Instant;
use voirs_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("📊 Simple Batch Evaluation Example");
    println!("==================================");

    // Create a simple pipeline
    let pipeline = VoirsPipelineBuilder::new().build().await?;

    // Test scenarios
    let test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck?",
    ];

    let mut results = HashMap::new();

    for (i, text) in test_texts.iter().enumerate() {
        println!("\n🎯 Processing text {}: {}", i + 1, text);

        let start_time = Instant::now();

        // Synthesize the text
        let audio = pipeline.synthesize(text).await?;

        let generation_time = start_time.elapsed();
        let audio_duration = audio.duration();
        let rtf = generation_time.as_secs_f32() / audio_duration;

        println!("   ✅ Generated audio");
        println!("      Duration: {audio_duration:.2}s");
        println!(
            "      Generation time: {:.2}s",
            generation_time.as_secs_f32()
        );
        println!("      RTF: {rtf:.3}");

        // Save the results
        results.insert(
            text.to_string(),
            EvaluationMetrics {
                audio_duration,
                generation_time: generation_time.as_secs_f32(),
                rtf,
                audio_length: audio.len(),
                sample_rate: audio.sample_rate(),
            },
        );
    }

    // Generate simple report
    println!("\n📋 Evaluation Summary");
    println!("====================");

    let total_audio_duration: f32 = results.values().map(|m| m.audio_duration).sum();
    let total_generation_time: f32 = results.values().map(|m| m.generation_time).sum();
    let average_rtf: f32 = results.values().map(|m| m.rtf).sum::<f32>() / results.len() as f32;

    println!("Total texts processed: {}", results.len());
    println!("Total audio duration: {total_audio_duration:.2}s");
    println!("Total generation time: {total_generation_time:.2}s");
    println!("Average RTF: {average_rtf:.3}");

    if average_rtf < 1.0 {
        println!("✅ Real-time performance achieved!");
    } else {
        println!("⚠️  Slower than real-time");
    }

    // Show best and worst performing texts
    let best = results
        .iter()
        .min_by(|a, b| a.1.rtf.partial_cmp(&b.1.rtf).unwrap());
    let worst = results
        .iter()
        .max_by(|a, b| a.1.rtf.partial_cmp(&b.1.rtf).unwrap());

    if let Some((text, metrics)) = best {
        println!("\n🏆 Best performance:");
        println!("   Text: \"{}...\"", &text[..text.len().min(30)]);
        println!("   RTF: {:.3}", metrics.rtf);
    }

    if let Some((text, metrics)) = worst {
        println!("\n🐌 Slowest performance:");
        println!("   Text: \"{}...\"", &text[..text.len().min(30)]);
        println!("   RTF: {:.3}", metrics.rtf);
    }

    println!("\n✅ Batch evaluation completed successfully!");
    Ok(())
}

#[derive(Debug, Clone)]
struct EvaluationMetrics {
    audio_duration: f32,
    generation_time: f32,
    rtf: f32,
    audio_length: usize,
    sample_rate: u32,
}
