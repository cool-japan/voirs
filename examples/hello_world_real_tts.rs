//! Hello World Real TTS - Pure Rust Implementation
//!
//! This example demonstrates real text-to-speech synthesis using VoiRS
//! with SafeTensors model loading (no Python dependency).

use std::collections::HashMap;

#[cfg(feature = "onnx")]
use voirs_acoustic::vits::onnx_loader::VitsOnnxInference;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🎙️  VoiRS Real TTS - Pure Rust");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Input text
    let text = "Hello World";
    println!("📝 Input: \"{}\"", text);
    println!();

    // Step 1: Text to Phonemes (G2P)
    let phonemes = text_to_phonemes(text);
    println!("🔤 Phonemes: {:?}", phonemes);
    println!();

    // Step 2: Phonemes to Token IDs
    let token_map = create_token_map();
    let token_ids: Vec<i64> = phonemes
        .iter()
        .filter_map(|p| token_map.get(p.as_str()).copied())
        .collect();

    println!("🔢 Token IDs: {:?}", token_ids);
    println!("   Length: {}", token_ids.len());
    println!();

    // Step 3: Load VITS model from ONNX
    #[cfg(feature = "onnx")]
    {
        println!("📥 Loading VITS model from ONNX...");

        // Try temp_dir first, fallback to /tmp
        let temp_dir = std::env::temp_dir();
        let mut model_path = temp_dir.join("voirs_models/ljspeech-vits-onnx/model.onnx");

        if !model_path.exists() {
            model_path =
                std::path::PathBuf::from("/tmp/voirs_models/ljspeech-vits-onnx/model.onnx");
        }

        println!("   Model: {}", model_path.display());
        println!();

        let mut model = VitsOnnxInference::from_file(&model_path)?;
        println!("   ✅ Model loaded");
        println!();

        // Step 4: Run inference
        println!("🎵 Generating speech...");
        let audio_samples = model.synthesize(&token_ids)?;

        println!("   ✅ Generated {} samples", audio_samples.len());
        println!("   Duration: {:.2}s", audio_samples.len() as f32 / 22050.0);
        println!();

        // Step 5: Save audio file
        let output_path = temp_dir.join("hello_world_rust.wav");
        save_wav(output_path.to_str().unwrap(), &audio_samples, 22050)?;

        println!("💾 Saved to: {}", output_path.display());
        println!();

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("✅ Pure Rust TTS complete!");
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
        eprintln!("   Run with: cargo run --example hello_world_real_tts --features onnx");
        return Err("ONNX feature required".into());
    }
}

/// Simple English G2P using CMUdict-style phonemes
fn text_to_phonemes(text: &str) -> Vec<String> {
    let text_lower = text.to_lowercase();
    let mut phonemes = Vec::new();

    // Word-level phoneme dictionary
    let word_phonemes: HashMap<&str, Vec<&str>> = [
        ("hello", vec!["HH", "EH1", "L", "OW1"]),
        ("world", vec!["W", "ER1", "L", "D"]),
    ]
    .iter()
    .cloned()
    .collect();

    for (i, word) in text_lower.split_whitespace().enumerate() {
        // Add pause between words
        if i > 0 {
            phonemes.push(",".to_string());
        }

        // Get phonemes for word
        if let Some(word_phones) = word_phonemes.get(word) {
            phonemes.extend(word_phones.iter().map(|s| s.to_string()));
        } else {
            // Fallback
            for ch in word.chars() {
                phonemes.push(ch.to_uppercase().to_string());
            }
        }
    }

    phonemes
}

/// Create token mapping from config
fn create_token_map() -> HashMap<String, i64> {
    let tokens = vec![
        ("<blank>", 0),
        ("<unk>", 1),
        ("AH0", 2),
        ("N", 3),
        ("T", 4),
        ("D", 5),
        ("S", 6),
        ("R", 7),
        ("L", 8),
        ("DH", 9),
        ("K", 10),
        ("Z", 11),
        ("IH1", 12),
        ("IH0", 13),
        ("M", 14),
        ("EH1", 15),
        ("W", 16),
        ("P", 17),
        ("AE1", 18),
        ("AH1", 19),
        ("V", 20),
        ("ER0", 21),
        ("F", 22),
        (",", 23),
        ("AA1", 24),
        ("B", 25),
        ("HH", 26),
        ("IY1", 27),
        ("UW1", 28),
        ("IY0", 29),
        ("AO1", 30),
        ("EY1", 31),
        ("AY1", 32),
        (".", 33),
        ("OW1", 34),
        ("SH", 35),
        ("NG", 36),
        ("G", 37),
        ("ER1", 38),
        ("CH", 39),
        ("JH", 40),
        ("Y", 41),
        ("AW1", 42),
        ("TH", 43),
        ("UH1", 44),
        ("EH2", 45),
        ("OW0", 46),
        ("EY2", 47),
        ("AO0", 48),
        ("IH2", 49),
        ("AE2", 50),
        ("AY2", 51),
        ("AA2", 52),
        ("UW0", 53),
        ("EH0", 54),
        ("OY1", 55),
        ("EY0", 56),
        ("AO2", 57),
        ("ZH", 58),
        ("OW2", 59),
        ("AE0", 60),
        ("UW2", 61),
        ("AH2", 62),
        ("AY0", 63),
        ("IY2", 64),
        ("AW2", 65),
        ("AA0", 66),
        ("'", 67),
        ("ER2", 68),
        ("UH2", 69),
        ("?", 70),
        ("OY2", 71),
        ("!", 72),
        ("AW0", 73),
        ("UH0", 74),
        ("OY0", 75),
    ];

    tokens
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
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

    // Write samples
    for &sample in samples {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        file.write_all(&sample_i16.to_le_bytes())?;
    }

    Ok(())
}
