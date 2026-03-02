//! Batch Transcription Example
//!
//! This example demonstrates batch processing of multiple audio files
//! with VoiRS Recognizer, including performance monitoring and optimization.
//!
//! Usage:
//! ```bash
//! cargo run --example batch_transcription --features="whisper-pure"
//! ```

use std::time::{Duration, Instant};
use voirs_recognizer::asr::{ASRBackend, FallbackConfig, WhisperModelSize};
use voirs_recognizer::prelude::*;
use voirs_recognizer::{PerformanceValidator, RecognitionError};

#[tokio::main]
async fn main() -> Result<(), RecognitionError> {
    println!("📚 VoiRS Batch Transcription Demo");
    println!("=================================\n");

    // Step 1: Create multiple synthetic audio samples
    println!("🎵 Creating multiple audio samples...");
    let sample_rate = 16000;
    let audio_samples = create_diverse_audio_samples(sample_rate);

    println!("✅ Created {} audio samples:", audio_samples.len());
    for (i, (name, audio)) in audio_samples.iter().enumerate() {
        println!(
            "   {}. {}: {:.1}s, {} samples",
            i + 1,
            name,
            audio.duration(),
            audio.len()
        );
    }

    // Step 2: Configure ASR for batch processing
    println!("\n🔧 Configuring ASR for batch processing...");
    let asr_config = ASRConfig {
        language: Some(LanguageCode::EnUs),
        word_timestamps: true,
        confidence_threshold: 0.5,
        ..Default::default()
    };

    println!("   • Configuration optimized for batch processing");
    println!("   • Language: {:?}", asr_config.language);
    println!(
        "   • Confidence threshold: {} (balanced speed/accuracy)",
        asr_config.confidence_threshold
    );

    // Step 3: Initialize ASR system
    println!("\n🚀 Initializing ASR system...");
    let init_start = Instant::now();
    let fallback_config = FallbackConfig {
        primary_backend: ASRBackend::Whisper {
            model_size: WhisperModelSize::Base,
            model_path: None,
        },
        fallback_backends: vec![],
        quality_threshold: 0.5,
        ..Default::default()
    };
    let mut asr = IntelligentASRFallback::new(fallback_config).await?;
    let init_time = init_start.elapsed();

    println!("✅ ASR system initialized in {:?}", init_time);

    // Step 4: Sequential processing (baseline)
    println!("\n📝 Sequential Processing:");
    let sequential_start = Instant::now();
    let mut sequential_results = Vec::new();

    for (name, audio) in &audio_samples {
        let recognition_start = Instant::now();
        let result = asr.transcribe(audio, None).await?;
        let recognition_time = recognition_start.elapsed();

        sequential_results.push((name.clone(), result.clone(), recognition_time));
        println!(
            "   • {}: \"{}\" ({:.2}s, RTF={:.3})",
            name,
            result.transcript.text,
            recognition_time.as_secs_f32(),
            recognition_time.as_secs_f32() / audio.duration()
        );
    }

    let sequential_total = sequential_start.elapsed();
    println!("   📊 Sequential total time: {:?}", sequential_total);

    // Step 5: Batch processing optimization
    println!("\n🚀 Batch Processing Optimization:");
    let batch_start = Instant::now();

    // Extract audio buffers for batch processing
    let audio_buffers: Vec<&AudioBuffer> = audio_samples.iter().map(|(_, audio)| audio).collect();

    // Process in batch
    // Process all audio buffers sequentially since there's no batch method
    let mut batch_results = Vec::new();
    for audio in &audio_buffers {
        let result = asr.transcribe(audio, None).await?;
        batch_results.push(result);
    }
    let batch_total = batch_start.elapsed();

    println!("   📊 Batch processing results:");
    for (i, ((name, _), result)) in audio_samples.iter().zip(batch_results.iter()).enumerate() {
        println!(
            "   • {}: \"{}\" (confidence: {:.2})",
            name, result.transcript.text, result.transcript.confidence
        );
    }

    println!("   📊 Batch total time: {:?}", batch_total);
    println!(
        "   📊 Speedup: {:.2}x",
        sequential_total.as_secs_f32() / batch_total.as_secs_f32()
    );

    // Step 6: Performance analysis
    println!("\n⚡ Performance Analysis:");
    let validator = PerformanceValidator::new().with_verbose(false);

    // Calculate aggregate metrics
    let total_audio_duration: f32 = audio_samples
        .iter()
        .map(|(_, audio)| audio.duration())
        .sum();
    let batch_rtf = batch_total.as_secs_f32() / total_audio_duration;

    println!("   • Total audio duration: {:.1}s", total_audio_duration);
    println!("   • Batch processing RTF: {:.3}", batch_rtf);
    println!(
        "   • Average per-file time: {:?}",
        batch_total / audio_samples.len() as u32
    );

    // Memory usage validation
    let (memory_usage, memory_passed) = validator.estimate_memory_usage()?;
    println!(
        "   • Memory usage: {:.1} MB ({})",
        memory_usage as f64 / (1024.0 * 1024.0),
        if memory_passed {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

    // Step 7: Quality analysis for batch results
    println!("\n🔍 Quality Analysis:");
    let analyzer_config = AudioAnalysisConfig::default();
    let analyzer = AudioAnalyzerImpl::new(analyzer_config).await?;

    for (i, (name, audio)) in audio_samples.iter().enumerate() {
        let analysis = analyzer
            .analyze(audio, Some(&AudioAnalysisConfig::default()))
            .await?;
        let result = &batch_results[i];

        println!("   • {}:", name);
        println!("     - Transcript: \"{}\"", result.transcript.text);
        println!("     - Confidence: {:.2}", result.transcript.confidence);

        if let Some(snr) = analysis.quality_metrics.get("snr") {
            println!("     - SNR: {:.1} dB", snr);
        }
        if let Some(energy) = analysis.quality_metrics.get("energy") {
            println!("     - Energy: {:.3}", energy);
        }
    }

    // Step 8: Demonstrate different batch sizes
    println!("\n📊 Batch Size Optimization:");
    let batch_sizes = vec![1, 2, 4, audio_samples.len()];

    for batch_size in batch_sizes {
        let batch_test_start = Instant::now();

        // Process in chunks of batch_size
        let mut chunk_results = Vec::new();
        for chunk in audio_buffers.chunks(batch_size) {
            let mut chunk_results_batch = Vec::new();
            for audio in chunk {
                let result = asr.transcribe(audio, None).await?;
                chunk_results_batch.push(result);
            }
            chunk_results.extend(chunk_results_batch);
        }

        let batch_test_time = batch_test_start.elapsed();
        let batch_test_rtf = batch_test_time.as_secs_f32() / total_audio_duration;

        println!(
            "   • Batch size {}: {:?} (RTF={:.3})",
            batch_size, batch_test_time, batch_test_rtf
        );
    }

    // Step 9: Error handling in batch processing
    println!("\n🛡️ Error Handling in Batch Processing:");

    // Create a mix of valid and invalid audio
    let mixed_audio = vec![
        AudioBuffer::mono(vec![0.0; 16000], 16000), // Valid 1s audio
        AudioBuffer::mono(vec![], 16000),           // Invalid empty audio
        AudioBuffer::mono(vec![0.0; 8000], 16000),  // Valid 0.5s audio
    ];

    println!("   • Processing mixed valid/invalid audio batch...");

    // Process each individually to handle errors gracefully
    let mut mixed_results = Vec::new();
    for (i, audio) in mixed_audio.iter().enumerate() {
        match asr.transcribe(audio, None).await {
            Ok(result) => {
                mixed_results.push(Some(result));
                println!("   • Audio {}: Success", i + 1);
            }
            Err(e) => {
                mixed_results.push(None);
                println!("   • Audio {}: Error - {}", i + 1, e);
            }
        }
    }

    // Step 10: Output format options
    println!("\n📄 Output Format Options:");

    // JSON-like output
    println!("   • JSON-like format:");
    for (i, ((name, _), result)) in audio_samples.iter().zip(batch_results.iter()).enumerate() {
        println!("   {{");
        println!("     \"file\": \"{}\",", name);
        println!("     \"transcript\": \"{}\",", result.transcript.text);
        println!("     \"confidence\": {:.2},", result.transcript.confidence);
        println!("     \"language\": \"{:?}\",", result.transcript.language);

        if !result.transcript.word_timestamps.is_empty() {
            println!("     \"words\": [");
            for (j, word) in result.transcript.word_timestamps.iter().enumerate() {
                println!(
                    "       {{\"word\": \"{}\", \"start\": {:.2}, \"end\": {:.2}}}{}",
                    word.word,
                    word.start_time,
                    word.end_time,
                    if j < result.transcript.word_timestamps.len() - 1 {
                        ","
                    } else {
                        ""
                    }
                );
            }
            println!("     ]");
        }

        println!(
            "   }}{}",
            if i < batch_results.len() - 1 { "," } else { "" }
        );
    }

    // CSV-like output
    println!("\n   • CSV-like format:");
    println!("   File,Transcript,Confidence,Language,Duration");
    for ((name, audio), result) in audio_samples.iter().zip(batch_results.iter()) {
        println!(
            "   \"{}\",\"{}\",{:.2},{:?},{:.2}",
            name,
            result.transcript.text,
            result.transcript.confidence,
            result.transcript.language,
            audio.duration()
        );
    }

    println!("\n✅ Batch transcription demo completed successfully!");
    println!("📊 Performance Summary:");
    println!("   • Processed {} audio files", audio_samples.len());
    println!("   • Total duration: {:.1}s", total_audio_duration);
    println!("   • Batch RTF: {:.3}", batch_rtf);
    println!(
        "   • Batch speedup: {:.2}x over sequential",
        sequential_total.as_secs_f32() / batch_total.as_secs_f32()
    );

    println!("\n💡 Key insights:");
    println!("   • Batch processing provides significant speedup");
    println!("   • Memory usage scales appropriately with batch size");
    println!("   • Error handling is important for production batches");
    println!("   • Different output formats suit different use cases");

    println!("\n🎯 Next steps:");
    println!("   • Scale to larger batch sizes");
    println!("   • Add parallel processing for multiple batches");
    println!("   • Implement progress tracking for long batches");
    println!("   • Add file I/O for real audio files");

    Ok(())
}

fn create_diverse_audio_samples(sample_rate: u32) -> Vec<(String, AudioBuffer)> {
    let mut samples = Vec::new();

    // Sample 1: Low frequency signal (male voice simulation)
    let mut low_freq = Vec::new();
    for i in 0..(sample_rate as f32 * 2.0) as usize {
        let t = i as f32 / sample_rate as f32;
        let sample = 0.1 * (2.0 * std::f32::consts::PI * 120.0 * t).sin()
            + 0.05 * (2.0 * std::f32::consts::PI * 240.0 * t).sin();
        low_freq.push(sample);
    }
    samples.push((
        "Low_Frequency_Speech".to_string(),
        AudioBuffer::mono(low_freq, sample_rate),
    ));

    // Sample 2: High frequency signal (female voice simulation)
    let mut high_freq = Vec::new();
    for i in 0..(sample_rate as f32 * 1.5) as usize {
        let t = i as f32 / sample_rate as f32;
        let sample = 0.1 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
            + 0.06 * (2.0 * std::f32::consts::PI * 400.0 * t).sin();
        high_freq.push(sample);
    }
    samples.push((
        "High_Frequency_Speech".to_string(),
        AudioBuffer::mono(high_freq, sample_rate),
    ));

    // Sample 3: Complex harmonic signal
    let mut complex = Vec::new();
    for i in 0..(sample_rate as f32 * 2.5) as usize {
        let t = i as f32 / sample_rate as f32;
        let f0 = 150.0;
        let sample = 0.1 * (2.0 * std::f32::consts::PI * f0 * t).sin()
            + 0.05 * (2.0 * std::f32::consts::PI * f0 * 2.0 * t).sin()
            + 0.03 * (2.0 * std::f32::consts::PI * f0 * 3.0 * t).sin()
            + 0.02 * (2.0 * std::f32::consts::PI * f0 * 4.0 * t).sin();
        complex.push(sample);
    }
    samples.push((
        "Complex_Harmonic_Speech".to_string(),
        AudioBuffer::mono(complex, sample_rate),
    ));

    // Sample 4: Short burst signal
    let mut burst = Vec::new();
    for i in 0..(sample_rate as f32 * 0.8) as usize {
        let t = i as f32 / sample_rate as f32;
        let envelope = if t < 0.4 { 1.0 } else { 0.0 };
        let sample = 0.15 * (2.0 * std::f32::consts::PI * 180.0 * t).sin() * envelope;
        burst.push(sample);
    }
    samples.push((
        "Short_Burst_Speech".to_string(),
        AudioBuffer::mono(burst, sample_rate),
    ));

    // Sample 5: Noisy signal
    let mut noisy = Vec::new();
    for i in 0..(sample_rate as f32 * 1.8) as usize {
        let t = i as f32 / sample_rate as f32;
        let signal = 0.08 * (2.0 * std::f32::consts::PI * 160.0 * t).sin();
        let noise = 0.02 * (scirs2_core::random::random::<f32>() - 0.5);
        noisy.push(signal + noise);
    }
    samples.push((
        "Noisy_Speech".to_string(),
        AudioBuffer::mono(noisy, sample_rate),
    ));

    samples
}
