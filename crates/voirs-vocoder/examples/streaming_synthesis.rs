//! Streaming Synthesis Example
//!
//! This example demonstrates:
//! - Chunk-based mel spectrogram processing
//! - Real-time audio streaming
//! - Latency optimization techniques
//! - Buffer management for low-latency synthesis
//! - Progressive audio generation
//!
//! Run with: cargo run --example streaming_synthesis --features candle

use futures::StreamExt;
use std::time::Instant;
use tokio::sync::mpsc;
use voirs_vocoder::{
    audio::io::{AudioEncodeConfig, AudioEncoder, AudioFileFormat},
    AudioBuffer, DummyVocoder, MelSpectrogram, SynthesisConfig, Vocoder,
};

/// Configuration for streaming synthesis
#[derive(Debug, Clone)]
struct StreamConfig {
    /// Chunk size in frames
    chunk_size: usize,
    /// Overlap between chunks (for smooth transitions)
    overlap: usize,
    /// Sample rate
    sample_rate: u32,
    /// Number of mel bins
    n_mels: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 50, // 50 frames per chunk (~0.6s at 22kHz with hop=256)
            overlap: 10,    // 10 frame overlap for crossfading
            sample_rate: 22050,
            n_mels: 80,
        }
    }
}

/// Statistics for streaming synthesis
#[derive(Debug, Default)]
struct StreamStats {
    total_chunks: usize,
    total_frames: usize,
    total_samples: usize,
    total_latency_ms: f32,
    min_chunk_latency_ms: f32,
    max_chunk_latency_ms: f32,
    avg_chunk_latency_ms: f32,
}

impl StreamStats {
    fn new() -> Self {
        Self {
            min_chunk_latency_ms: f32::MAX,
            ..Default::default()
        }
    }

    fn update(&mut self, chunk_frames: usize, chunk_samples: usize, latency_ms: f32) {
        self.total_chunks += 1;
        self.total_frames += chunk_frames;
        self.total_samples += chunk_samples;
        self.total_latency_ms += latency_ms;
        self.min_chunk_latency_ms = self.min_chunk_latency_ms.min(latency_ms);
        self.max_chunk_latency_ms = self.max_chunk_latency_ms.max(latency_ms);
        self.avg_chunk_latency_ms = self.total_latency_ms / self.total_chunks as f32;
    }

    fn print_summary(&self, total_duration: f32) {
        println!("\n📊 Streaming Statistics:");
        println!("├─────────────────────────────────────────");
        println!("│ Total chunks processed: {}", self.total_chunks);
        println!("│ Total frames: {}", self.total_frames);
        println!("│ Total samples: {}", self.total_samples);
        println!("│ Audio duration: {:.2}s", total_duration);
        println!("├─────────────────────────────────────────");
        println!("│ Latency (min): {:.2}ms", self.min_chunk_latency_ms);
        println!("│ Latency (avg): {:.2}ms", self.avg_chunk_latency_ms);
        println!("│ Latency (max): {:.2}ms", self.max_chunk_latency_ms);
        println!("├─────────────────────────────────────────");
        let total_processing_time = self.total_latency_ms / 1000.0;
        let rtf = total_processing_time / total_duration;
        println!("│ Total processing: {:.3}s", total_processing_time);
        println!("│ Real-Time Factor: {:.4}x", rtf);
        if rtf < 1.0 {
            println!("│ ✅ Can process in real-time!");
        } else {
            println!("│ ⚠️  Cannot maintain real-time");
        }
        println!("└─────────────────────────────────────────");
    }
}

/// Generate a continuous mel spectrogram for streaming
fn generate_streaming_mel(
    total_frames: usize,
    n_mels: usize,
    sample_rate: u32,
) -> Vec<Vec<Vec<f32>>> {
    let chunk_size = 50;
    let num_chunks = (total_frames + chunk_size - 1) / chunk_size;

    let mut chunks = Vec::new();

    for chunk_idx in 0..num_chunks {
        let start_frame = chunk_idx * chunk_size;
        let end_frame = ((chunk_idx + 1) * chunk_size).min(total_frames);
        let frames_in_chunk = end_frame - start_frame;

        let mut chunk_data = Vec::with_capacity(n_mels);

        for mel_idx in 0..n_mels {
            let mut mel_frames = Vec::with_capacity(frames_in_chunk);

            // Create a evolving frequency pattern across chunks
            let base_freq = 100.0 + (mel_idx as f32 / n_mels as f32) * 2000.0;
            let chunk_phase = chunk_idx as f32 / num_chunks as f32 * 2.0 * std::f32::consts::PI;

            for local_frame in 0..frames_in_chunk {
                let global_frame = start_frame + local_frame;
                let time = global_frame as f32 * 256.0 / sample_rate as f32; // hop_length = 256

                // Evolving sinusoidal pattern with chunk-based modulation
                let base_value = (2.0 * std::f32::consts::PI * base_freq * time / 1000.0).sin();
                let modulation = (chunk_phase + time * 0.5).sin() * 0.5 + 0.5;
                let magnitude = base_value * modulation;

                // Convert to dB scale
                let db_value = 20.0 * magnitude.abs().log10();
                mel_frames.push(db_value.clamp(-20.0, 20.0));
            }

            chunk_data.push(mel_frames);
        }

        chunks.push(chunk_data);
    }

    chunks
}

/// Process mel chunks in streaming fashion
async fn process_streaming<V: Vocoder>(
    vocoder: &V,
    mel_chunks: Vec<Vec<Vec<f32>>>,
    config: &StreamConfig,
    synthesis_config: &SynthesisConfig,
) -> Result<(Vec<AudioBuffer>, StreamStats), Box<dyn std::error::Error>> {
    let mut audio_chunks = Vec::new();
    let mut stats = StreamStats::new();

    println!("\n🔄 Processing {} chunks...", mel_chunks.len());

    for (idx, chunk_data) in mel_chunks.iter().enumerate() {
        let start = Instant::now();

        // Create mel spectrogram for this chunk
        let chunk_mel = MelSpectrogram::new(
            chunk_data.clone(),
            config.sample_rate,
            256, // hop_length
        );

        // Process chunk
        let audio = vocoder.vocode(&chunk_mel, Some(synthesis_config)).await?;

        let latency_ms = start.elapsed().as_micros() as f32 / 1000.0;
        stats.update(chunk_mel.n_frames, audio.samples().len(), latency_ms);

        print!(
            "\r  Chunk {}/{}: {:.2}ms ",
            idx + 1,
            mel_chunks.len(),
            latency_ms
        );
        std::io::Write::flush(&mut std::io::stdout())?;

        audio_chunks.push(audio);
    }

    println!("\n  ✓ All chunks processed");

    Ok((audio_chunks, stats))
}

/// Concatenate audio chunks with crossfading
fn concatenate_chunks(
    chunks: Vec<AudioBuffer>,
    overlap_samples: usize,
) -> Result<AudioBuffer, Box<dyn std::error::Error>> {
    if chunks.is_empty() {
        return Err("No chunks to concatenate".into());
    }

    let sample_rate = chunks[0].sample_rate();
    let channels = chunks[0].channels();

    // Calculate total length accounting for overlaps
    let total_samples: usize = chunks.iter().map(|c| c.samples().len()).sum::<usize>()
        - (overlap_samples * (chunks.len() - 1));

    let mut concatenated = vec![0.0f32; total_samples];
    let mut write_pos = 0;

    for (idx, chunk) in chunks.iter().enumerate() {
        let chunk_samples = chunk.samples();

        if idx == 0 {
            // First chunk: copy entirely
            concatenated[..chunk_samples.len()].copy_from_slice(chunk_samples);
            write_pos = chunk_samples.len();
        } else {
            // Subsequent chunks: crossfade overlap region
            let overlap_start = write_pos - overlap_samples;

            // Crossfade
            for i in 0..overlap_samples.min(chunk_samples.len()) {
                let fade_factor = i as f32 / overlap_samples as f32;
                let existing = concatenated[overlap_start + i];
                let new = chunk_samples[i];
                concatenated[overlap_start + i] =
                    existing * (1.0 - fade_factor) + new * fade_factor;
            }

            // Copy remaining part
            if chunk_samples.len() > overlap_samples {
                let copy_start = overlap_samples;
                let copy_len = chunk_samples.len() - overlap_samples;
                concatenated[write_pos..write_pos + copy_len]
                    .copy_from_slice(&chunk_samples[copy_start..]);
                write_pos += copy_len;
            }
        }
    }

    Ok(AudioBuffer::new(concatenated, sample_rate, channels))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌊 VoiRS Streaming Synthesis Example");
    println!("═══════════════════════════════════════════════");

    // Configuration
    let stream_config = StreamConfig::default();
    let total_duration = 5.0; // seconds
    let total_frames = ((total_duration * stream_config.sample_rate as f32) / 256.0) as usize;

    println!("\n📋 Configuration:");
    println!("  • Sample rate: {} Hz", stream_config.sample_rate);
    println!("  • Total duration: {:.1}s", total_duration);
    println!("  • Chunk size: {} frames", stream_config.chunk_size);
    println!("  • Overlap: {} frames", stream_config.overlap);
    println!("  • Total frames: {}", total_frames);

    // Generate streaming mel data
    println!("\n🎵 Generating mel spectrogram chunks...");
    let mel_chunks = generate_streaming_mel(
        total_frames,
        stream_config.n_mels,
        stream_config.sample_rate,
    );
    println!("  ✓ Generated {} chunks", mel_chunks.len());

    // Create vocoder
    println!("\n🔧 Initializing vocoder...");
    let vocoder = DummyVocoder::new();
    println!("  ✓ Vocoder ready");

    // Synthesis configuration
    let synthesis_config = SynthesisConfig::default();

    // Process in streaming fashion
    let (audio_chunks, stats) =
        process_streaming(&vocoder, mel_chunks, &stream_config, &synthesis_config).await?;

    // Print statistics
    stats.print_summary(total_duration);

    // Concatenate chunks
    println!("\n🔗 Concatenating audio chunks...");
    let overlap_samples = stream_config.overlap * 256; // Convert frames to samples
    let final_audio = concatenate_chunks(audio_chunks, overlap_samples)?;

    println!("  ✓ Final audio:");
    println!("    • Samples: {}", final_audio.samples().len());
    println!("    • Duration: {:.2}s", final_audio.duration());
    println!("    • Sample rate: {} Hz", final_audio.sample_rate());
    println!("    • Channels: {}", final_audio.channels());

    // Save output
    println!("\n💾 Saving audio...");
    let encode_config = AudioEncodeConfig {
        format: AudioFileFormat::Wav,
        sample_rate: final_audio.sample_rate(),
        channels: final_audio.channels() as u16,
        bits_per_sample: 16,
        bit_rate: None,
        quality: Some(0.8),
        compression_level: Some(5),
    };

    let encoder = AudioEncoder::new(encode_config);
    let output_path = std::env::temp_dir().join("streaming_synthesis_output.wav");
    encoder.write_to_file(&final_audio, &output_path)?;
    println!("  ✓ Saved to: {}", output_path.display());

    // Performance insights
    println!("\n💡 Performance Insights:");
    println!("─────────────────────────────────────────────");
    if stats.avg_chunk_latency_ms < 50.0 {
        println!("  🚀 Excellent latency for real-time applications");
    } else if stats.avg_chunk_latency_ms < 100.0 {
        println!("  ✨ Good latency, suitable for interactive use");
    } else {
        println!("  ⚠️  Consider optimizing for real-time use");
    }

    let chunk_duration = stream_config.chunk_size as f32 * 256.0 / stream_config.sample_rate as f32;
    println!("  • Chunk duration: {:.0}ms", chunk_duration * 1000.0);
    println!(
        "  • Processing time: {:.0}ms avg",
        stats.avg_chunk_latency_ms
    );

    if stats.avg_chunk_latency_ms < chunk_duration * 1000.0 {
        let headroom = (chunk_duration * 1000.0 - stats.avg_chunk_latency_ms)
            / (chunk_duration * 1000.0)
            * 100.0;
        println!("  • Headroom: {:.1}% (can maintain real-time)", headroom);
    }

    println!("\n✨ Streaming synthesis complete!");
    println!("═══════════════════════════════════════════════\n");

    Ok(())
}
