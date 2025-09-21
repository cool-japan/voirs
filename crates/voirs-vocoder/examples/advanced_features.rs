//! Advanced features example showing streaming and performance monitoring
//!
//! This example demonstrates:
//! - Streaming audio processing  
//! - Performance monitoring across different configurations
//! - Batch processing with different parameters

use futures::{stream, StreamExt};
use voirs_vocoder::{
    audio::io::{AudioEncodeConfig, AudioEncoder, AudioFileFormat},
    DummyVocoder, MelSpectrogram, SynthesisConfig, Vocoder,
};

/// Generate test mel spectrogram with configurable characteristics
fn generate_enhanced_mel(
    n_mels: usize,
    n_frames: usize,
    sample_rate: u32,
    complexity: f32,
) -> MelSpectrogram {
    let mut data = Vec::with_capacity(n_mels);
    for mel_idx in 0..n_mels {
        let mut frame = Vec::with_capacity(n_frames);
        for frame_idx in 0..n_frames {
            let base_freq = (mel_idx as f32 / n_mels as f32) * 4000.0 + 80.0;
            let time = frame_idx as f32 / (sample_rate as f32 / 256.0);

            // Add complexity with multiple harmonics
            let fundamental = (2.0 * std::f32::consts::PI * base_freq * time / 8000.0).sin();
            let harmonic2 = (4.0 * std::f32::consts::PI * base_freq * time / 8000.0).sin() * 0.3;
            let harmonic3 = (6.0 * std::f32::consts::PI * base_freq * time / 8000.0).sin() * 0.1;

            let magnitude = -20.0
                + 15.0 * (fundamental + harmonic2 * complexity + harmonic3 * complexity).abs();
            frame.push(magnitude);
        }
        data.push(frame);
    }

    MelSpectrogram::new(data, sample_rate, 256)
}

/// Helper function to save audio with proper encoder
fn save_audio(
    audio: &voirs_vocoder::AudioBuffer,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = AudioEncodeConfig {
        sample_rate: audio.sample_rate(),
        channels: audio.channels() as u16,
        bits_per_sample: 16,
        format: AudioFileFormat::Wav,
        bit_rate: None,
        quality: None,
        compression_level: None,
    };
    let encoder = AudioEncoder::new(config);
    encoder.write_to_file(audio, path)?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 voirs-vocoder Advanced Features Example");
    println!("==========================================");

    let vocoder = DummyVocoder::new();
    let sample_rate = 22050;
    let duration_secs = 1.0;
    let n_frames = ((duration_secs * sample_rate as f32) / 256.0) as usize;

    // Demo different synthesis configurations
    println!("\n1. Testing different synthesis configurations...");

    let test_configs = [
        (
            "Normal",
            SynthesisConfig {
                speed: 1.0,
                pitch_shift: 0.0,
                energy: 1.0,
                speaker_id: None,
                seed: None,
            },
        ),
        (
            "Fast",
            SynthesisConfig {
                speed: 1.5,
                pitch_shift: 0.0,
                energy: 1.0,
                speaker_id: None,
                seed: None,
            },
        ),
        (
            "Slow",
            SynthesisConfig {
                speed: 0.7,
                pitch_shift: 0.0,
                energy: 1.0,
                speaker_id: None,
                seed: None,
            },
        ),
        (
            "High Pitch",
            SynthesisConfig {
                speed: 1.0,
                pitch_shift: 5.0,
                energy: 1.0,
                speaker_id: None,
                seed: None,
            },
        ),
        (
            "Low Pitch",
            SynthesisConfig {
                speed: 1.0,
                pitch_shift: -5.0,
                energy: 1.0,
                speaker_id: None,
                seed: None,
            },
        ),
        (
            "Quiet",
            SynthesisConfig {
                speed: 1.0,
                pitch_shift: 0.0,
                energy: 0.5,
                speaker_id: None,
                seed: None,
            },
        ),
        (
            "Loud",
            SynthesisConfig {
                speed: 1.0,
                pitch_shift: 0.0,
                energy: 1.5,
                speaker_id: None,
                seed: None,
            },
        ),
    ];

    let base_mel = generate_enhanced_mel(80, n_frames, sample_rate, 0.5);

    for (name, config) in &test_configs {
        let start = std::time::Instant::now();
        let audio = vocoder.vocode(&base_mel, Some(config)).await?;
        let elapsed = start.elapsed();
        let rtf = elapsed.as_secs_f32() / audio.duration();

        println!(
            "   {}: RTF = {:.3}x, Duration = {:.2}s, Samples = {}",
            name,
            rtf,
            audio.duration(),
            audio.samples().len()
        );

        // Save each variant
        let filename = format!(
            "output_config_{}.wav",
            name.to_lowercase().replace(' ', "_")
        );
        save_audio(&audio, &filename)?;
    }

    // Demo streaming processing
    println!("\n2. Streaming audio processing...");
    let stream_mels: Vec<MelSpectrogram> = (0..5)
        .map(|i| {
            let complexity = (i as f32 + 1.0) * 0.2; // Increasing complexity
            generate_enhanced_mel(80, n_frames / 2, sample_rate, complexity)
        })
        .collect();

    let mel_stream = stream::iter(stream_mels);
    let audio_stream = vocoder
        .vocode_stream(Box::new(mel_stream), Some(&SynthesisConfig::default()))
        .await?;

    let mut chunk_count = 0;
    let mut total_samples = 0;
    let stream_start = std::time::Instant::now();

    tokio::pin!(audio_stream);
    while let Some(result) = audio_stream.next().await {
        let audio_chunk = result?;
        chunk_count += 1;
        total_samples += audio_chunk.samples().len();
        println!(
            "   Processed chunk {}: {} samples, {:.2}s",
            chunk_count,
            audio_chunk.samples().len(),
            audio_chunk.duration()
        );

        // Save each chunk
        let chunk_path = format!("output_stream_chunk_{chunk_count}.wav");
        save_audio(&audio_chunk, &chunk_path)?;
    }

    let stream_time = stream_start.elapsed();
    let total_duration = total_samples as f32 / sample_rate as f32;
    let stream_rtf = stream_time.as_secs_f32() / total_duration;

    println!("   Streaming completed: {chunk_count} chunks, {total_duration:.1}s total audio");
    println!("   Streaming RTF: {stream_rtf:.3}x");

    // Demo batch processing with different sizes
    println!("\n3. Batch processing performance...");

    for &batch_size in &[1, 2, 4, 8, 16] {
        let batch_mels: Vec<MelSpectrogram> = (0..batch_size)
            .map(|_| generate_enhanced_mel(80, n_frames / 4, sample_rate, 0.5))
            .collect();

        let batch_configs: Vec<SynthesisConfig> = (0..batch_size)
            .map(|i| SynthesisConfig {
                speed: 1.0,
                pitch_shift: (i as f32 - batch_size as f32 / 2.0) * 2.0, // Vary pitch
                energy: 1.0,
                speaker_id: None,
                seed: Some(i as u64),
            })
            .collect();

        let batch_start = std::time::Instant::now();
        let audio_batch = vocoder
            .vocode_batch(&batch_mels, Some(&batch_configs))
            .await?;
        let batch_time = batch_start.elapsed();

        let total_samples: usize = audio_batch.iter().map(|a| a.samples().len()).sum();
        let total_duration: f32 = audio_batch.iter().map(|a| a.duration()).sum();
        let batch_rtf = batch_time.as_secs_f32() / total_duration;

        println!(
            "   Batch size {}: {} samples, {:.2}s audio, RTF = {:.3}x, Time = {:.1}ms",
            batch_size,
            total_samples,
            total_duration,
            batch_rtf,
            batch_time.as_millis()
        );

        // Save first item from each batch
        if !audio_batch.is_empty() {
            let path = format!("output_batch_size_{batch_size}.wav");
            save_audio(&audio_batch[0], &path)?;
        }
    }

    // Performance comparison with different complexities
    println!("\n4. Performance vs complexity analysis...");

    let complexity_tests = [
        ("Simple", 0.1),
        ("Medium", 0.5),
        ("Complex", 1.0),
        ("Very Complex", 1.5),
    ];

    for (name, complexity) in &complexity_tests {
        let complex_mel = generate_enhanced_mel(80, n_frames, sample_rate, *complexity);

        let start = std::time::Instant::now();
        let audio = vocoder
            .vocode(&complex_mel, Some(&SynthesisConfig::default()))
            .await?;
        let elapsed = start.elapsed();

        let rtf = elapsed.as_secs_f32() / audio.duration();
        let memory_mb = (audio.samples().len() * 4) as f32 / 1_000_000.0;

        println!(
            "   {}: RTF = {:.3}x, Memory = {:.1} MB, Samples = {}",
            name,
            rtf,
            memory_mb,
            audio.samples().len()
        );

        let filename = format!(
            "output_complexity_{}.wav",
            name.to_lowercase().replace(' ', "_")
        );
        save_audio(&audio, &filename)?;
    }

    // Memory usage analysis
    println!("\n5. Memory usage analysis...");

    let size_tests = [
        ("Small", 40, n_frames / 4),
        ("Medium", 80, n_frames / 2),
        ("Large", 80, n_frames),
        ("Extra Large", 128, n_frames),
    ];

    for (name, mels, frames) in &size_tests {
        let test_mel = generate_enhanced_mel(*mels, *frames, sample_rate, 0.5);

        let start = std::time::Instant::now();
        let audio = vocoder
            .vocode(&test_mel, Some(&SynthesisConfig::default()))
            .await?;
        let elapsed = start.elapsed();

        let input_size_mb = (mels * frames * 4) as f32 / 1_000_000.0;
        let output_size_mb = (audio.samples().len() * 4) as f32 / 1_000_000.0;
        let rtf = elapsed.as_secs_f32() / audio.duration();

        println!("   {name}: Input {input_size_mb:.1} MB -> Output {output_size_mb:.1} MB, RTF = {rtf:.3}x");
    }

    println!("\n✅ Advanced features demo completed!");
    println!("\nGenerated files include:");
    println!("  - Configuration variants (output_config_*.wav)");
    println!("  - Streaming chunks (output_stream_chunk_*.wav)");
    println!("  - Batch size tests (output_batch_size_*.wav)");
    println!("  - Complexity tests (output_complexity_*.wav)");

    Ok(())
}
