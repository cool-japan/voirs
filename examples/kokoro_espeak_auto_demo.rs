//! Kokoro TTS with Automatic IPA Generation using eSpeak NG
//!
//! This example demonstrates automatic IPA phoneme generation using eSpeak NG
//! for supported Kokoro languages. Users only need to provide raw text.
//!
//! Supported languages:
//! - English (American & British)
//! - Spanish
//! - French
//! - Hindi
//! - Italian
//! - Portuguese (Brazilian)
//!
//! Note: Japanese and Chinese use pre-generated IPA (misaki library recommended)

use std::process::Command;

#[cfg(feature = "onnx")]
use voirs_acoustic::vits::onnx_kokoro::KokoroOnnxInference;

/// Language configuration with eSpeak NG support
struct Language {
    name: &'static str,
    espeak_voice: &'static str,
    voice_idx: usize,
    voice_name: &'static str,
}

/// Generate IPA phonemes using eSpeak NG
fn generate_ipa(text: &str, espeak_voice: &str) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("espeak-ng")
        .arg("-v")
        .arg(espeak_voice)
        .arg("-q")
        .arg("--ipa")
        .arg(text)
        .output()?;

    if !output.status.success() {
        return Err(format!(
            "eSpeak NG failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    let ipa = String::from_utf8(output.stdout)?
        .lines()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();

    Ok(ipa)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🌍 VoiRS Kokoro TTS with Automatic IPA Generation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Check if eSpeak NG is available
    if Command::new("espeak-ng").arg("--version").output().is_err() {
        eprintln!("❌ eSpeak NG not found!");
        eprintln!("   Please install: brew install espeak-ng (macOS)");
        eprintln!("                   sudo apt install espeak-ng (Linux)");
        return Err("eSpeak NG not installed".into());
    }

    println!("✅ eSpeak NG detected");
    println!();

    // Language configurations
    let languages = vec![
        Language {
            name: "English (American)",
            espeak_voice: "en-us",
            voice_idx: 4,
            voice_name: "af_jessica",
        },
        Language {
            name: "English (British)",
            espeak_voice: "en-gb",
            voice_idx: 20,
            voice_name: "bf_alice",
        },
        Language {
            name: "Spanish",
            espeak_voice: "es",
            voice_idx: 28,
            voice_name: "ef_dora",
        },
        Language {
            name: "French",
            espeak_voice: "fr-fr",
            voice_idx: 31,
            voice_name: "ff_siwis",
        },
        Language {
            name: "Hindi",
            espeak_voice: "hi",
            voice_idx: 32,
            voice_name: "hf_alpha",
        },
        Language {
            name: "Italian",
            espeak_voice: "it",
            voice_idx: 36,
            voice_name: "if_sara",
        },
        Language {
            name: "Portuguese (Brazilian)",
            espeak_voice: "pt-br",
            voice_idx: 43,
            voice_name: "pf_dora",
        },
    ];

    // Sample texts for each language
    let texts = vec![
        "Hello, the world is beautiful today",    // English (US)
        "Hello, the world is beautiful today",    // English (UK)
        "Hola, el mundo es hermoso hoy",          // Spanish
        "Bonjour, le monde est beau aujourd'hui", // French
        "नमस्ते, दुनिया आज सुंदर है",                   // Hindi
        "Ciao, il mondo è bello oggi",            // Italian
        "Olá, o mundo está lindo hoje",           // Portuguese
    ];

    #[cfg(feature = "onnx")]
    {
        // Load Kokoro model
        let temp_dir = std::env::temp_dir();
        let model_dir = temp_dir.join("voirs_models/kokoro-zh");

        println!("📥 Loading Kokoro multilingual model...");
        let mut model = KokoroOnnxInference::from_kokoro_files(&model_dir)?;
        println!("   ✅ Model loaded");
        println!();

        // Synthesize for each language
        for (idx, (lang, text)) in languages.iter().zip(texts.iter()).enumerate() {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("🌐 {} ({}/{})", lang.name, idx + 1, languages.len());
            println!();
            println!("📝 Input text: \"{}\"", text);
            println!("   eSpeak voice: {}", lang.espeak_voice);
            println!();

            // Generate IPA using eSpeak NG
            print!("🔊 Generating IPA phonemes... ");
            std::io::Write::flush(&mut std::io::stdout())?;
            let ipa = generate_ipa(text, lang.espeak_voice)?;
            println!("✓");
            println!("   IPA: \"{}\"", ipa);
            println!();

            // Synthesize
            print!("🎵 Synthesizing speech... ");
            std::io::Write::flush(&mut std::io::stdout())?;
            let speed = 1.0;
            let audio_samples = model.synthesize_trim_end(&ipa, lang.voice_idx, speed)?;
            println!("✓");

            let sample_rate = model.sample_rate();
            let duration = audio_samples.len() as f32 / sample_rate as f32;

            println!("   Voice: {} (index {})", lang.voice_name, lang.voice_idx);
            println!(
                "   Generated {} samples ({:.2}s)",
                audio_samples.len(),
                duration
            );
            println!();

            // Save audio file
            let filename = format!(
                "kokoro_auto_{}_{}.wav",
                lang.name
                    .to_lowercase()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", ""),
                lang.voice_name
            );
            let output_path = temp_dir.join(&filename);
            save_wav(output_path.to_str().unwrap(), &audio_samples, sample_rate)?;

            println!("💾 Saved to: {}", output_path.display());
            println!();
        }

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "✅ All {} languages synthesized with automatic IPA generation!",
            languages.len()
        );
        println!();
        println!("🎧 To play audio files:");
        println!("   macOS:  afplay <file.wav>");
        println!("   Linux:  aplay <file.wav>");
        println!();

        return Ok(());
    }

    #[cfg(not(feature = "onnx"))]
    {
        eprintln!("❌ ONNX feature not enabled!");
        eprintln!("   Run with: cargo run --example kokoro_espeak_auto_demo --features onnx");
        return Err("ONNX feature required".into());
    }
}

/// Save audio samples as WAV file with 100ms padding
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
