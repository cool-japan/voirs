//! Kokoro Japanese TTS Demo - Pure Rust Implementation
//!
//! This example demonstrates Japanese text-to-speech using Kokoro multilingual model
//! with ONNX Runtime. Input is IPA phonemes (from misaki library).
//!
//! To generate IPA phonemes, use Python:
//! ```python
//! import misaki.ja as ja
//! g2p = ja.JAG2P()
//! phonemes, _ = g2p("こんにちは")
//! print(phonemes)  # Output: koɲɲiʨiβa
//! ```

use std::collections::HashMap;

#[cfg(feature = "onnx")]
use voirs_acoustic::vits::onnx_kokoro::KokoroOnnxInference;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🎙️  VoiRS Kokoro Japanese TTS - Pure Rust");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Input: IPA phonemes for "こんにちは、今日の世界も美しいです"
    // Generated using: misaki.ja.JAG2P()
    let ipa_phonemes = "koɲɲiʨiβa kʲoː no sekai mo ɯʦɨkɯɕiː desɨ";

    println!("📝 Input (Japanese): \"こんにちは、今日の世界も美しいです\"");
    println!("   IPA Phonemes: \"{}\"", ipa_phonemes);
    println!();

    #[cfg(feature = "onnx")]
    {
        // Load Kokoro model
        let temp_dir = std::env::temp_dir();
        let model_dir = temp_dir.join("voirs_models/kokoro-zh");

        println!("📥 Loading Kokoro multilingual model...");
        println!("   Model dir: {}", model_dir.display());

        let mut model = KokoroOnnxInference::from_kokoro_files(&model_dir)?;
        println!("   ✅ Model loaded");
        println!();

        // Voice selection (from voices_averaged.bin, alphabetically sorted)
        // Japanese female voices: 38-41 (jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro)
        // Japanese male voice: 42 (jm_kumo)
        // See voice_names.txt for full mapping
        let voice_idx = 38; // jf_alpha (Japanese female)
        let speed = 1.0;

        println!("🎵 Synthesizing speech...");
        println!("   Voice index: {} (Japanese female)", voice_idx);
        println!("   Speed: {}", speed);
        println!();

        // Use synthesize_trim_end to trim only trailing silence
        let audio_samples = model.synthesize_trim_end(ipa_phonemes, voice_idx, speed)?;

        let sample_rate = model.sample_rate();
        println!("   ✅ Generated {} samples", audio_samples.len());
        println!("   Sample rate: {} Hz", sample_rate);
        println!(
            "   Duration: {:.2}s",
            audio_samples.len() as f32 / sample_rate as f32
        );
        println!();

        // Save audio file
        let output_path = temp_dir.join("kokoro_japanese_rust.wav");
        save_wav(output_path.to_str().unwrap(), &audio_samples, sample_rate)?;

        println!("💾 Saved to: {}", output_path.display());
        println!();

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("✅ Pure Rust Japanese TTS complete!");
        println!();
        println!("🎧 Play the audio:");
        println!("   macOS:  afplay {}", output_path.display());
        println!("   Linux:  aplay {}", output_path.display());
        println!();

        return Ok(());
    }

    #[cfg(not(feature = "onnx"))]
    {
        eprintln!("❌ ONNX feature not enabled!");
        eprintln!("   Run with: cargo run --example kokoro_japanese_demo --features onnx");
        return Err("ONNX feature required".into());
    }
}

/// Save audio samples as WAV file
fn save_wav(
    path: &str,
    samples: &[f32],
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path)?;

    // Add 100ms of silence padding at the beginning
    let padding_samples = (sample_rate as f32 * 0.1) as usize; // 100ms
    let total_samples = padding_samples + samples.len();

    // WAV header
    let num_samples = total_samples as u32;
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
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&num_channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;

    // data chunk
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;

    // Write padding silence first
    for _ in 0..padding_samples {
        file.write_all(&0i16.to_le_bytes())?;
    }

    // Write actual samples
    for &sample in samples {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        file.write_all(&sample_i16.to_le_bytes())?;
    }

    Ok(())
}
