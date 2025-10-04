//! Chinese TTS Demo - Pure Rust Implementation
//!
//! This example demonstrates Chinese text-to-speech synthesis using VoiRS
//! with ONNX Runtime. Input is pinyin (e.g., "ni3 hao3").

use std::collections::HashMap;

#[cfg(feature = "onnx")]
use voirs_acoustic::vits::onnx_chinese::ChineseVitsOnnxInference;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🎙️  VoiRS Chinese TTS - Pure Rust");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Input: Pinyin with tones
    // "你好，今天的世界也很美丽" = "ni3 hao3 jin1 tian1 de5 shi4 jie4 ye3 hen3 mei3 li4"
    let pinyin_input = "ni3 hao3 jin1 tian1 de5 shi4 jie4 ye3 hen3 mei3 li4";
    println!("📝 Input (Pinyin): \"{}\"", pinyin_input);
    println!("   中文: \"你好，今天的世界也很美丽\"");
    println!();

    // Step 1: Pinyin to Phonemes
    let phonemes = pinyin_to_phonemes(pinyin_input);
    println!("🔤 Phonemes: {:?}", phonemes);
    println!();

    // Step 2: Load token mapping from model file
    let temp_dir = std::env::temp_dir();
    let mut token_file = temp_dir.join("voirs_models/vits-zh-aishell3/tokens.txt");
    if !token_file.exists() {
        token_file = std::path::PathBuf::from("/tmp/voirs_models/vits-zh-aishell3/tokens.txt");
    }
    let token_map = load_token_map_from_file(&token_file)?;

    // Step 3: Phonemes to Token IDs
    let mut token_ids: Vec<i64> = Vec::new();
    token_ids.push(*token_map.get("sil").unwrap_or(&0)); // Start silence

    for phoneme in &phonemes {
        if let Some(&id) = token_map.get(phoneme.as_str()) {
            token_ids.push(id);
        } else {
            eprintln!("⚠️  Unknown phoneme: {}", phoneme);
        }
    }

    token_ids.push(*token_map.get("eos").unwrap_or(&1)); // End

    println!("🔢 Token IDs: {:?}", token_ids);
    println!("   Length: {}", token_ids.len());
    println!();

    // Step 4: Load VITS model from ONNX
    #[cfg(feature = "onnx")]
    {
        println!("📥 Loading Chinese VITS model from ONNX...");

        let temp_dir = std::env::temp_dir();
        let mut model_path = temp_dir.join("voirs_models/vits-zh-aishell3/vits-aishell3.onnx");

        if !model_path.exists() {
            model_path =
                std::path::PathBuf::from("/tmp/voirs_models/vits-zh-aishell3/vits-aishell3.onnx");
        }

        println!("   Model: {}", model_path.display());
        println!();

        let mut model = ChineseVitsOnnxInference::from_file(&model_path)?;
        println!("   ✅ Model loaded");
        println!();

        // Step 5: Run inference
        println!("🎵 Generating speech...");
        let audio_samples = model.synthesize(&token_ids)?;

        println!("   ✅ Generated {} samples", audio_samples.len());
        println!("   Duration: {:.2}s", audio_samples.len() as f32 / 22050.0);
        println!();

        // Step 6: Save audio file
        let output_path = temp_dir.join("chinese_tts_rust.wav");
        save_wav(output_path.to_str().unwrap(), &audio_samples, 22050)?;

        println!("💾 Saved to: {}", output_path.display());
        println!();

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("✅ Pure Rust Chinese TTS complete!");
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
        eprintln!("   Run with: cargo run --example chinese_tts_demo --features onnx");
        return Err("ONNX feature required".into());
    }
}

/// Convert pinyin syllables to phonemes (initial + final)
fn pinyin_to_phonemes(pinyin: &str) -> Vec<String> {
    let syllables: Vec<&str> = pinyin.split_whitespace().collect();
    let mut phonemes = Vec::new();

    // Initials (声母) - order matters! Longer first
    let initials = vec![
        "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x",
        "r", "z", "c", "s",
    ];

    for syllable in syllables {
        let mut initial = "";
        let mut final_part = syllable;

        // Special handling for y/w (not true initials in token map)
        // ye3 → e3, yi3 → i3, wu3 → u3, wa3 → ua3, etc.
        if syllable.starts_with('y') || syllable.starts_with('w') {
            // Remove y/w and keep the rest
            final_part = &syllable[1..];
        } else {
            // Extract initial (声母) for other syllables
            for init in &initials {
                if syllable.starts_with(init) {
                    initial = init;
                    final_part = &syllable[init.len()..];
                    break;
                }
            }
        }

        // Add phonemes
        if !initial.is_empty() {
            phonemes.push(initial.to_string());
        }

        if !final_part.is_empty() {
            phonemes.push(final_part.to_string());
        }
    }

    phonemes
}

/// Load token mapping from file
fn load_token_map_from_file(
    path: &std::path::Path,
) -> Result<HashMap<String, i64>, Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut map = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 2 {
            let token = parts[0].to_string();
            let id: i64 = parts[1].parse()?;
            map.insert(token, id);
        }
    }

    Ok(map)
}

/// Load Chinese token mapping (AISHELL3 VITS model) - Deprecated, use load_token_map_from_file
#[allow(dead_code)]
fn load_chinese_token_map() -> HashMap<String, i64> {
    let tokens = vec![
        ("sil", 0),
        ("eos", 1),
        ("sp", 2),
        ("#0", 3),
        ("#1", 4),
        ("#2", 5),
        ("#3", 6),
        ("^", 7),
        ("b", 8),
        ("c", 9),
        ("ch", 10),
        ("d", 11),
        ("f", 12),
        ("g", 13),
        ("h", 14),
        ("j", 15),
        ("k", 16),
        ("l", 17),
        ("m", 18),
        ("n", 19),
        ("p", 20),
        ("q", 21),
        ("r", 22),
        ("s", 23),
        ("sh", 24),
        ("t", 25),
        ("x", 26),
        ("z", 27),
        ("zh", 28),
        // Finals with tones
        ("a1", 29),
        ("a2", 30),
        ("a3", 31),
        ("a4", 32),
        ("a5", 33),
        ("ai1", 34),
        ("ai2", 35),
        ("ai3", 36),
        ("ai4", 37),
        ("ai5", 38),
        ("an1", 39),
        ("an2", 40),
        ("an3", 41),
        ("an4", 42),
        ("an5", 43),
        ("ang1", 44),
        ("ang2", 45),
        ("ang3", 46),
        ("ang4", 47),
        ("ang5", 48),
        ("ao1", 49),
        ("ao2", 50),
        ("ao3", 51),
        ("ao4", 52),
        ("ao5", 53),
        ("e1", 54),
        ("e2", 55),
        ("e3", 56),
        ("e4", 57),
        ("e5", 58),
        ("ei1", 59),
        ("ei2", 60),
        ("ei3", 61),
        ("ei4", 62),
        ("ei5", 63),
        ("en1", 64),
        ("en2", 65),
        ("en3", 66),
        ("en4", 67),
        ("en5", 68),
        ("eng1", 69),
        ("eng2", 70),
        ("eng3", 71),
        ("eng4", 72),
        ("eng5", 73),
        ("er1", 74),
        ("er2", 75),
        ("er3", 76),
        ("er4", 77),
        ("er5", 78),
        ("i1", 79),
        ("i2", 80),
        ("i3", 81),
        ("i4", 82),
        ("i5", 83),
        ("ia1", 84),
        ("ia2", 85),
        ("ia3", 86),
        ("ia4", 87),
        ("ia5", 88),
        ("ian1", 89),
        ("ian2", 90),
        ("ian3", 91),
        ("ian4", 92),
        ("ian5", 93),
        ("iang1", 94),
        ("iang2", 95),
        ("iang3", 96),
        ("iang4", 97),
        ("iang5", 98),
        ("iao1", 99),
        ("iao2", 100),
        ("iao3", 101),
        ("iao4", 102),
        ("iao5", 103),
        ("ie1", 104),
        ("ie2", 105),
        ("ie3", 106),
        ("ie4", 107),
        ("ie5", 108),
        ("in1", 109),
        ("in2", 110),
        ("in3", 111),
        ("in4", 112),
        ("in5", 113),
        ("ing1", 114),
        ("ing2", 115),
        ("ing3", 116),
        ("ing4", 117),
        ("ing5", 118),
        ("iong1", 119),
        ("iong2", 120),
        ("iong3", 121),
        ("iong4", 122),
        ("iong5", 123),
        ("iu1", 124),
        ("iu2", 125),
        ("iu3", 126),
        ("iu4", 127),
        ("iu5", 128),
        ("o1", 129),
        ("o2", 130),
        ("o3", 131),
        ("o4", 132),
        ("o5", 133),
        ("ong1", 134),
        ("ong2", 135),
        ("ong3", 136),
        ("ong4", 137),
        ("ong5", 138),
        ("ou1", 139),
        ("ou2", 140),
        ("ou3", 141),
        ("ou4", 142),
        ("ou5", 143),
        ("u1", 144),
        ("u2", 145),
        ("u3", 146),
        ("u4", 147),
        ("u5", 148),
        ("ua1", 149),
        ("ua2", 150),
        ("ua3", 151),
        ("ua4", 152),
        ("ua5", 153),
        ("uai1", 154),
        ("uai2", 155),
        ("uai3", 156),
        ("uai4", 157),
        ("uai5", 158),
        ("uan1", 159),
        ("uan2", 160),
        ("uan3", 161),
        ("uan4", 162),
        ("uan5", 163),
        ("uang1", 164),
        ("uang2", 165),
        ("uang3", 166),
        ("uang4", 167),
        ("uang5", 168),
        ("ui1", 169),
        ("ui2", 170),
        ("ui3", 171),
        ("ui4", 172),
        ("ui5", 173),
        ("un1", 174),
        ("un2", 175),
        ("un3", 176),
        ("un4", 177),
        ("un5", 178),
        ("uo1", 179),
        ("uo2", 180),
        ("uo3", 181),
        ("uo4", 182),
        ("uo5", 183),
        ("v1", 184),
        ("v2", 185),
        ("v3", 186),
        ("v4", 187),
        ("v5", 188),
        ("van1", 189),
        ("van2", 190),
        ("van3", 191),
        ("van4", 192),
        ("van5", 193),
        ("ve1", 194),
        ("ve2", 195),
        ("ve3", 196),
        ("ve4", 197),
        ("ve5", 198),
        ("vn1", 199),
        ("vn2", 200),
        ("vn3", 201),
        ("vn4", 202),
        ("vn5", 203),
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
