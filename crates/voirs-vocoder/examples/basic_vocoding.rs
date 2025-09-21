//! Basic vocoding example showing how to use voirs-vocoder
//!
//! This example demonstrates:
//! - Creating a vocoder instance
//! - Generating test mel spectrograms
//! - Converting mel to audio
//! - Saving audio to WAV file

use voirs_vocoder::{
    audio::io::{AudioEncodeConfig, AudioEncoder, AudioFileFormat},
    DummyVocoder, MelSpectrogram, SynthesisConfig, Vocoder,
};

/// Generate a simple test mel spectrogram with sine wave pattern
fn generate_test_mel(n_mels: usize, n_frames: usize, sample_rate: u32) -> MelSpectrogram {
    println!("Generating test mel spectrogram: {n_mels} mels x {n_frames} frames");

    let mut data = Vec::with_capacity(n_mels);
    for mel_idx in 0..n_mels {
        let mut frame = Vec::with_capacity(n_frames);
        for frame_idx in 0..n_frames {
            // Generate realistic mel values (log magnitude in dB range)
            let base_freq = (mel_idx as f32 / n_mels as f32) * 4000.0 + 80.0;
            let time = frame_idx as f32 / (sample_rate as f32 / 256.0); // Assume hop_length = 256
            let magnitude = -20.0
                + 15.0
                    * (2.0 * std::f32::consts::PI * base_freq * time / 8000.0)
                        .sin()
                        .abs();
            frame.push(magnitude);
        }
        data.push(frame);
    }

    MelSpectrogram::new(data, sample_rate, 256)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 voirs-vocoder Basic Example");
    println!("==============================");

    // Create a vocoder instance
    println!("\n1. Creating vocoder instance...");
    let vocoder = DummyVocoder::new();
    let metadata = vocoder.metadata();
    println!("   Vocoder: {}", metadata.name);
    println!("   Version: {}", metadata.version);
    println!("   Sample rate: {} Hz", metadata.sample_rate);

    // Generate test mel spectrogram
    println!("\n2. Generating test mel spectrogram...");
    let sample_rate = 22050;
    let duration_secs = 2.0;
    let n_frames = ((duration_secs * sample_rate as f32) / 256.0) as usize; // hop_length = 256
    let n_mels = 80;

    let mel = generate_test_mel(n_mels, n_frames, sample_rate);
    println!("   Generated mel: {n_mels} mels x {n_frames} frames");
    println!("   Duration: {duration_secs:.1} seconds");

    // Configure synthesis parameters
    println!("\n3. Configuring synthesis...");
    let synthesis_config = SynthesisConfig {
        speed: 1.0,       // Normal speed
        pitch_shift: 0.0, // No pitch shift
        energy: 1.0,      // Normal energy
        ..Default::default()
    };
    println!("   Speed: {:.1}x", synthesis_config.speed);
    println!(
        "   Pitch shift: {:.1} semitones",
        synthesis_config.pitch_shift
    );
    println!("   Energy scale: {:.1}x", synthesis_config.energy);

    // Convert mel to audio
    println!("\n4. Converting mel to audio...");
    let start_time = std::time::Instant::now();
    let audio_buffer = vocoder.vocode(&mel, Some(&synthesis_config)).await?;
    let conversion_time = start_time.elapsed();

    println!(
        "   Audio generated: {} samples",
        audio_buffer.samples().len()
    );
    println!("   Sample rate: {} Hz", audio_buffer.sample_rate());
    println!("   Channels: {}", audio_buffer.channels());
    println!("   Duration: {:.2} seconds", audio_buffer.duration());
    println!("   Conversion time: {:.2} ms", conversion_time.as_millis());

    // Calculate Real-Time Factor (RTF)
    let rtf = conversion_time.as_secs_f32() / audio_buffer.duration();
    println!("   Real-Time Factor: {rtf:.3}x");
    if rtf < 1.0 {
        println!("   ✅ Faster than real-time!");
    } else {
        println!("   ⚠️  Slower than real-time");
    }

    // Save audio to file
    println!("\n5. Saving audio to file...");
    let output_path = "output_basic_example.wav";
    let config = AudioEncodeConfig {
        sample_rate: audio_buffer.sample_rate(),
        channels: audio_buffer.channels() as u16,
        bits_per_sample: 16,
        format: AudioFileFormat::Wav,
        bit_rate: None,
        quality: None,
        compression_level: None,
    };
    let encoder = AudioEncoder::new(config);
    encoder.write_to_file(&audio_buffer, output_path)?;
    println!("   Saved to: {output_path}");

    // Demo batch processing
    println!("\n6. Demo batch processing...");
    let batch_size = 3;
    let mels: Vec<MelSpectrogram> = (0..batch_size)
        .map(|i| {
            println!("   Generating mel {}/{}", i + 1, batch_size);
            generate_test_mel(n_mels, n_frames / 2, sample_rate) // Shorter for batch demo
        })
        .collect();

    let batch_start = std::time::Instant::now();
    let configs = vec![synthesis_config.clone(); batch_size];
    let audio_batch = vocoder.vocode_batch(&mels, Some(&configs)).await?;
    let batch_time = batch_start.elapsed();

    println!("   Batch processed: {} audio buffers", audio_batch.len());
    println!("   Batch time: {:.2} ms", batch_time.as_millis());
    println!(
        "   Average per item: {:.2} ms",
        batch_time.as_millis() as f32 / batch_size as f32
    );

    // Save batch results
    for (i, audio) in audio_batch.iter().enumerate() {
        let path = format!("output_batch_{}.wav", i + 1);
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
        encoder.write_to_file(audio, &path)?;
        println!("   Saved batch item {} to: {}", i + 1, path);
    }

    println!("\n✅ Example completed successfully!");
    println!("\nGenerated files:");
    println!("  - output_basic_example.wav");
    for i in 1..=batch_size {
        println!("  - output_batch_{i}.wav");
    }

    Ok(())
}
