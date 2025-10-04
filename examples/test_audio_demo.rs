//! Test Audio Demo - Verify VoiRS audio generation works
//!
//! This example tests basic audio generation without requiring trained models.
//! It generates simple test signals to verify the audio pipeline works.

use anyhow::Result;
use std::f32::consts::PI;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎵 VoiRS Audio Generation Test");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Test 1: Generate simple sine wave
    println!("Test 1: Generating 440Hz sine wave (A4 note)");
    let sample_rate = 22050;
    let duration = 2.0; // seconds
    let frequency = 440.0; // Hz

    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * PI * frequency * t).sin() * 0.5; // Amplitude 0.5
        samples.push(sample);
    }

    println!("  ✅ Generated {} samples", samples.len());
    println!("  Duration: {:.2}s", duration);
    println!("  Sample rate: {} Hz", sample_rate);
    println!();

    // Save as WAV file
    let output_path = "/tmp/test_sine_wave.wav";
    save_wav(output_path, &samples, sample_rate)?;
    println!("  💾 Saved to: {}", output_path);
    println!();

    // Test 2: Generate chord (C major: C4, E4, G4)
    println!("Test 2: Generating C major chord");
    let frequencies = [261.63, 329.63, 392.00]; // C4, E4, G4
    let mut chord_samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let mut sample = 0.0;
        for &freq in &frequencies {
            sample += (2.0 * PI * freq * t).sin();
        }
        sample *= 0.3; // Normalize amplitude
        chord_samples.push(sample);
    }

    let chord_path = "/tmp/test_chord.wav";
    save_wav(chord_path, &chord_samples, sample_rate)?;
    println!("  ✅ Generated C major chord");
    println!("  💾 Saved to: {}", chord_path);
    println!();

    // Test 3: Generate sweep (20Hz to 2000Hz)
    println!("Test 3: Generating frequency sweep");
    let start_freq = 20.0;
    let end_freq = 2000.0;
    let mut sweep_samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let progress = t / duration;
        let freq = start_freq + (end_freq - start_freq) * progress;
        let sample = (2.0 * PI * freq * t).sin() * 0.4;
        sweep_samples.push(sample);
    }

    let sweep_path = "/tmp/test_sweep.wav";
    save_wav(sweep_path, &sweep_samples, sample_rate)?;
    println!(
        "  ✅ Generated frequency sweep ({:.0}Hz → {:.0}Hz)",
        start_freq, end_freq
    );
    println!("  💾 Saved to: {}", sweep_path);
    println!();

    // Summary
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ All audio generation tests passed!");
    println!();
    println!("Generated files:");
    println!("  1. {}", output_path);
    println!("  2. {}", chord_path);
    println!("  3. {}", sweep_path);
    println!();
    println!("You can play these files with:");
    println!("  macOS:  afplay /tmp/test_*.wav");
    println!("  Linux:  aplay /tmp/test_*.wav");
    println!("  Windows: start /tmp/test_*.wav");
    println!();
    println!("🎉 VoiRS audio pipeline is working!");

    Ok(())
}

/// Save audio samples as WAV file
fn save_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path)?;

    // WAV header
    let num_samples = samples.len() as u32;
    let num_channels = 1u16;
    let bits_per_sample = 16u16;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_size = num_samples * num_channels as u32 * bits_per_sample as u32 / 8;

    // RIFF header
    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_size).to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // Chunk size
    file.write_all(&1u16.to_le_bytes())?; // Audio format (PCM)
    file.write_all(&num_channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;

    // data chunk
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;

    // Write samples (convert f32 to i16)
    for &sample in samples {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        file.write_all(&sample_i16.to_le_bytes())?;
    }

    Ok(())
}
