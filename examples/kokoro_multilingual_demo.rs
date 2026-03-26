//! Kokoro Multilingual TTS Demo - All 9 Supported Languages
//!
//! This example demonstrates text-to-speech for all languages supported by Kokoro-82M:
//! - English (American & British)
//! - Spanish
//! - French
//! - Hindi
//! - Italian
//! - Portuguese (Brazilian)
//! - Japanese
//! - Chinese
//!
//! IPA phoneme generation:
//! - Japanese & Chinese: misaki library (Python)
//! - Other languages: eSpeak NG (via command-line)

use std::collections::HashMap;
use std::process::Command;

#[cfg(feature = "onnx")]
use voirs_acoustic::vits::onnx_kokoro::KokoroOnnxInference;

/// Language configuration
struct LanguageConfig {
    name: &'static str,
    text: &'static str,
    espeak_voice: Option<&'static str>,
    ipa_phonemes: &'static str,
    voice_idx: usize,
    voice_name: &'static str,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🌍 VoiRS Kokoro Multilingual TTS - Pure Rust");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Language configurations
    let languages = vec![
        LanguageConfig {
            name: "English (American)",
            text: "Hello, the world is beautiful today",
            espeak_voice: Some("en-us"),
            ipa_phonemes: "həlˈoʊ ðə wˈɜːld ɪz bjˈuːɾifəl tədˈeɪ",
            voice_idx: 4, // af_jessica (American female)
            voice_name: "af_jessica",
        },
        LanguageConfig {
            name: "English (British)",
            text: "Hello, the world is beautiful today",
            espeak_voice: Some("en-gb"),
            ipa_phonemes: "həlˈəʊ ðə wˈɜːld ɪz bjˈuːtifəl tədˈeɪ",
            voice_idx: 20, // bf_alice (British female)
            voice_name: "bf_alice",
        },
        LanguageConfig {
            name: "Spanish",
            text: "Hola, el mundo es hermoso hoy",
            espeak_voice: Some("es"),
            ipa_phonemes: "ˈola el mˈundo ˈes eɾmˈoso ˈoɪ",
            voice_idx: 28, // ef_dora (Spanish female)
            voice_name: "ef_dora",
        },
        LanguageConfig {
            name: "French",
            text: "Bonjour, le monde est beau aujourd'hui",
            espeak_voice: Some("fr-fr"),
            ipa_phonemes: "bɔ̃ʒˈuʁ lə mˈɔ̃d ɛ bˈo oʒuʁdˈɥi",
            voice_idx: 31, // ff_siwis (French female)
            voice_name: "ff_siwis",
        },
        LanguageConfig {
            name: "Hindi",
            text: "नमस्ते, दुनिया आज सुंदर है",
            espeak_voice: Some("hi"),
            ipa_phonemes: "nəməsteː dʊnɪjaː aːɟ sʊ̃dər hɛː",
            voice_idx: 32, // hf_alpha (Hindi female)
            voice_name: "hf_alpha",
        },
        LanguageConfig {
            name: "Italian",
            text: "Ciao, il mondo è bello oggi",
            espeak_voice: Some("it"),
            ipa_phonemes: "tʃˈao il mˈondo ɛ bˈɛllo ˈɔdːʒi",
            voice_idx: 36, // if_sara (Italian female)
            voice_name: "if_sara",
        },
        LanguageConfig {
            name: "Portuguese (Brazilian)",
            text: "Olá, o mundo está lindo hoje",
            espeak_voice: Some("pt-br"),
            ipa_phonemes: "olˈa o mˈũdʊ esˈta lˈĩdʊ ˈoʒi",
            voice_idx: 43, // pf_dora (Portuguese female)
            voice_name: "pf_dora",
        },
        LanguageConfig {
            name: "Japanese",
            text: "こんにちは、今日の世界も美しいです",
            espeak_voice: None, // Use pre-generated IPA from misaki
            ipa_phonemes: "koɲɲiʨiβa kʲoː no sekai mo ɯʦɨkɯɕiː desɨ",
            voice_idx: 38, // jf_alpha (Japanese female)
            voice_name: "jf_alpha",
        },
        LanguageConfig {
            name: "Chinese (Mandarin)",
            text: "你好，今天的世界也很美丽",
            espeak_voice: None, // Use pre-generated IPA from misaki
            ipa_phonemes: "↓ni ↓xau, →ʨi →ntʰjɛn tɤ ↘ʂɨ ↘ʨje ↓je ↓xən ↓mei ↘li",
            voice_idx: 46, // zf_xiaobei (Chinese female)
            voice_name: "zf_xiaobei",
        },
    ];

    #[cfg(feature = "onnx")]
    {
        // Load Kokoro model
        let temp_dir = std::env::temp_dir();
        let model_dir = temp_dir.join("voirs_models/kokoro-zh");

        println!("📥 Loading Kokoro multilingual model...");
        println!("   Model dir: {}", model_dir.display());

        let model = KokoroOnnxInference::from_kokoro_files(&model_dir)?;
        println!("   ✅ Model loaded");
        println!();

        // Synthesize for each language
        for (idx, lang) in languages.iter().enumerate() {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("🌐 {} ({}/{})", lang.name, idx + 1, languages.len());
            println!();
            println!("📝 Input: \"{}\"", lang.text);
            println!("   IPA Phonemes: \"{}\"", lang.ipa_phonemes);
            println!("   Voice: {} (index {})", lang.voice_name, lang.voice_idx);
            println!();

            // Synthesize
            let speed = 1.0;
            let audio_samples =
                model.synthesize_trim_end(lang.ipa_phonemes, lang.voice_idx, speed)?;

            let sample_rate = model.sample_rate();
            let duration = audio_samples.len() as f32 / sample_rate as f32;

            println!(
                "   ✅ Generated {} samples ({:.2}s)",
                audio_samples.len(),
                duration
            );
            println!();

            // Save audio file
            let filename = format!(
                "kokoro_{}_{}.wav",
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
            "✅ All {} languages synthesized successfully!",
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
        eprintln!("   Run with: cargo run --example kokoro_multilingual_demo --features onnx");
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
