//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use clap::Subcommand;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;
use voirs_sdk::config::AppConfig;
use voirs_sdk::Result;
use crate::GlobalOptions;
/// Get standard Kokoro config file paths in order of preference
fn get_kokoro_config_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        paths.push(cwd.join("kokoro.toml"));
        paths.push(cwd.join("kokoro.json"));
        paths.push(cwd.join("kokoro.yaml"));
    }
    if let Some(config_dir) = dirs::config_dir() {
        let voirs_config = config_dir.join("voirs");
        paths.push(voirs_config.join("kokoro.toml"));
        paths.push(voirs_config.join("kokoro.json"));
        paths.push(voirs_config.join("kokoro.yaml"));
    }
    if let Some(home_dir) = dirs::home_dir() {
        paths.push(home_dir.join(".kokoro.toml"));
        paths.push(home_dir.join(".kokororc"));
    }
    paths
}
/// Execute Kokoro command
pub async fn execute_kokoro_command(
    command: &KokoroCommands,
    _config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    let kokoro_config = KokoroConfig::load();
    match command {
        KokoroCommands::Synth {
            text,
            output,
            lang,
            voice_index,
            voice_name,
            speed,
            ipa,
            play,
            model_dir,
        } => {
            let effective_lang = lang;
            let effective_speed = *speed;
            let effective_model_dir = model_dir
                .as_deref()
                .or(kokoro_config.model_dir.as_deref());
            execute_synth(
                    text,
                    output,
                    effective_lang,
                    *voice_index,
                    voice_name.as_deref(),
                    effective_speed,
                    ipa.as_deref(),
                    *play,
                    effective_model_dir,
                    global,
                )
                .await
        }
        KokoroCommands::Voices { lang, detailed, format } => {
            execute_voices(lang.as_deref(), *detailed, format, global).await
        }
        KokoroCommands::Languages { show_ipa_method } => {
            execute_languages(*show_ipa_method, global).await
        }
        KokoroCommands::Test { model_dir, lang } => {
            execute_test(model_dir.as_deref(), lang.as_deref(), global).await
        }
        KokoroCommands::Download { output, force } => {
            execute_download(output.as_deref(), *force, global).await
        }
        KokoroCommands::TextToIpa { text, lang, method, output } => {
            execute_text_to_ipa(text, lang, method, output.as_deref(), global).await
        }
        KokoroCommands::Batch { input, output_dir, jobs, manual_ipa } => {
            execute_batch(input, output_dir, *jobs, *manual_ipa, global).await
        }
        KokoroCommands::Config { show, init, path } => {
            execute_config(*show, *init, path.as_deref(), &kokoro_config, global).await
        }
    }
}
/// Execute synth command
async fn execute_synth(
    text: &str,
    output: &PathBuf,
    lang: &str,
    voice_index: Option<usize>,
    voice_name: Option<&str>,
    speed: f32,
    ipa: Option<&str>,
    play: bool,
    model_dir: Option<&std::path::Path>,
    global: &GlobalOptions,
) -> Result<()> {
    use voirs_acoustic::vits::onnx_kokoro::KokoroOnnxInference;
    let output_is_stdout = output.to_str() == Some("-");
    let quiet = global.quiet || output_is_stdout;
    if !quiet {
        println!("🎙️  Kokoro Multilingual TTS");
        println!(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        );
        println!();
        println!("📝 Text: \"{}\"", text);
        println!("   Language: {}", lang);
    }
    let model_path = if let Some(dir) = model_dir {
        dir.to_path_buf()
    } else {
        let temp_dir = std::env::temp_dir();
        temp_dir.join("voirs_models/kokoro-zh")
    };
    if !quiet {
        println!("   Model: {}", model_path.display());
    }
    if !quiet {
        print!("📥 Loading model... ");
        std::io::Write::flush(&mut std::io::stdout())
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(
                    format!("Failed to flush stdout: {}", e),
                )
            })?;
    }
    let mut model = KokoroOnnxInference::from_kokoro_files(&model_path)?;
    if !quiet {
        println!("✓");
    }
    let phonemes = if let Some(manual_ipa) = ipa {
        if !quiet {
            println!("   Using manual IPA: \"{}\"", manual_ipa);
        }
        manual_ipa.to_string()
    } else {
        if !quiet {
            print!("🔊 Generating IPA phonemes... ");
            std::io::Write::flush(&mut std::io::stdout())
                .map_err(|e| {
                    voirs_sdk::VoirsError::config_error(
                        format!("Failed to flush stdout: {}", e),
                    )
                })?;
        }
        let espeak_voice = match lang {
            "en-us" => "en-us",
            "en-gb" => "en-gb",
            "es" => "es",
            "fr" => "fr-fr",
            "hi" => "hi",
            "it" => "it",
            "pt-br" => "pt-br",
            _ => {
                return Err(
                    voirs_sdk::VoirsError::config_error(
                        format!(
                            "Language '{}' does not support automatic IPA generation.\nPlease use --ipa to provide manual phonemes.\n\nFor Japanese/Chinese, you can use the misaki Python library to generate IPA.",
                            lang
                        ),
                    ),
                );
            }
        };
        let output = Command::new("espeak-ng")
            .arg("-v")
            .arg(espeak_voice)
            .arg("-q")
            .arg("--ipa")
            .arg(text)
            .output()
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(
                    format!(
                        "Failed to run eSpeak NG. Is it installed?\nError: {}\n\nInstall eSpeak NG:\n  macOS: brew install espeak-ng\n  Linux: sudo apt install espeak-ng\n\nOr use --ipa to provide manual phonemes.",
                        e
                    ),
                )
            })?;
        if !output.status.success() {
            return Err(
                voirs_sdk::VoirsError::config_error(
                    format!(
                        "eSpeak NG failed: {}", String::from_utf8_lossy(& output.stderr)
                    ),
                ),
            );
        }
        let ipa_result = String::from_utf8(output.stdout)
            .map_err(|e| voirs_sdk::VoirsError::config_error(
                format!("Invalid UTF-8: {}", e),
            ))?
            .lines()
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string();
        if !quiet {
            println!("✓");
            println!("   IPA: \"{}\"", ipa_result);
        }
        ipa_result
    };
    let voice_idx = if let Some(idx) = voice_index {
        idx
    } else if let Some(name) = voice_name {
        get_voice_index_by_name(name)?
    } else {
        get_default_voice_for_language(lang)?
    };
    if !quiet {
        println!();
        print!("🎵 Synthesizing... ");
        std::io::Write::flush(&mut std::io::stdout())
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(
                    format!("Failed to flush stdout: {}", e),
                )
            })?;
    }
    let audio_samples = model.synthesize_trim_end(&phonemes, voice_idx, speed)?;
    if !quiet {
        println!("✓");
    }
    let sample_rate = model.sample_rate();
    if output_is_stdout {
        save_wav_to_stdout(&audio_samples, sample_rate)?;
    } else {
        save_wav(output.to_str().unwrap(), &audio_samples, sample_rate)?;
        if !quiet {
            let duration = audio_samples.len() as f32 / sample_rate as f32;
            println!();
            println!("✅ Success!");
            println!("   Samples: {}", audio_samples.len());
            println!("   Duration: {:.2}s", duration);
            println!("   Output: {}", output.display());
        }
        if play {
            if !quiet {
                println!();
                print!("▶️  Playing audio... ");
                std::io::Write::flush(&mut std::io::stdout())
                    .map_err(|e| {
                        voirs_sdk::VoirsError::config_error(
                            format!("Failed to flush stdout: {}", e),
                        )
                    })?;
            }
            play_audio_file(output)?;
            if !quiet {
                println!("✓");
            }
        }
    }
    Ok(())
}
/// Save audio as WAV file
fn save_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    use std::fs::File;
    use std::io::Write;
    let mut file = File::create(path)
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to create output file: {}", e),
            )
        })?;
    let padding_samples = (sample_rate as f32 * 0.1) as usize;
    let total_samples = padding_samples + samples.len();
    let num_samples = total_samples as u32;
    let num_channels = 1u16;
    let bits_per_sample = 16u16;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_size = num_samples * num_channels as u32 * bits_per_sample as u32 / 8;
    file.write_all(b"RIFF")
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    file.write_all(&(36 + data_size).to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    file.write_all(b"WAVE")
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    file.write_all(b"fmt ")
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    file.write_all(&16u32.to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    file.write_all(&1u16.to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    file.write_all(&num_channels.to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    file.write_all(&sample_rate.to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    file.write_all(&byte_rate.to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    file.write_all(&block_align.to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    file.write_all(&bits_per_sample.to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    file.write_all(b"data")
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV data: {}", e),
            )
        })?;
    file.write_all(&data_size.to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV data: {}", e),
            )
        })?;
    for _ in 0..padding_samples {
        file.write_all(&0i16.to_le_bytes())
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(
                    format!("Failed to write WAV data: {}", e),
                )
            })?;
    }
    for &sample in samples {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        file.write_all(&sample_i16.to_le_bytes())
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(
                    format!("Failed to write WAV data: {}", e),
                )
            })?;
    }
    Ok(())
}
/// Save audio as WAV to stdout
fn save_wav_to_stdout(samples: &[f32], sample_rate: u32) -> Result<()> {
    use std::io::Write;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    let padding_samples = (sample_rate as f32 * 0.1) as usize;
    let total_samples = padding_samples + samples.len();
    let num_samples = total_samples as u32;
    let num_channels = 1u16;
    let bits_per_sample = 16u16;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_size = num_samples * num_channels as u32 * bits_per_sample as u32 / 8;
    handle
        .write_all(b"RIFF")
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    handle
        .write_all(&(36 + data_size).to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    handle
        .write_all(b"WAVE")
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    handle
        .write_all(b"fmt ")
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    handle
        .write_all(&16u32.to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    handle
        .write_all(&1u16.to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    handle
        .write_all(&num_channels.to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    handle
        .write_all(&sample_rate.to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    handle
        .write_all(&byte_rate.to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    handle
        .write_all(&block_align.to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    handle
        .write_all(&bits_per_sample.to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV header: {}", e),
            )
        })?;
    handle
        .write_all(b"data")
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV data: {}", e),
            )
        })?;
    handle
        .write_all(&data_size.to_le_bytes())
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to write WAV data: {}", e),
            )
        })?;
    for _ in 0..padding_samples {
        handle
            .write_all(&0i16.to_le_bytes())
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(
                    format!("Failed to write WAV data: {}", e),
                )
            })?;
    }
    for &sample in samples {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        handle
            .write_all(&sample_i16.to_le_bytes())
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(
                    format!("Failed to write WAV data: {}", e),
                )
            })?;
    }
    handle
        .flush()
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!("Failed to flush stdout: {}", e))
        })?;
    Ok(())
}
/// Get voice index by name
fn get_voice_index_by_name(name: &str) -> Result<usize> {
    let voice_map: std::collections::HashMap<&str, usize> = [
        ("af_alloy", 0),
        ("af_aoede", 1),
        ("af_bella", 2),
        ("af_heart", 3),
        ("af_jessica", 4),
        ("af_kore", 5),
        ("af_nicole", 6),
        ("af_nova", 7),
        ("af_river", 8),
        ("af_sarah", 9),
        ("af_sky", 10),
        ("am_adam", 11),
        ("am_echo", 12),
        ("am_eric", 13),
        ("am_fenrir", 14),
        ("am_liam", 15),
        ("am_michael", 16),
        ("am_onyx", 17),
        ("am_puck", 18),
        ("am_santa", 19),
        ("bf_alice", 20),
        ("bf_emma", 21),
        ("bf_isabella", 22),
        ("bf_lily", 23),
        ("bm_daniel", 24),
        ("bm_fable", 25),
        ("bm_george", 26),
        ("bm_lewis", 27),
        ("ef_dora", 28),
        ("em_alex", 29),
        ("em_santa", 30),
        ("ff_siwis", 31),
        ("hf_alpha", 32),
        ("hf_beta", 33),
        ("hm_omega", 34),
        ("hm_psi", 35),
        ("if_sara", 36),
        ("im_nicola", 37),
        ("jf_alpha", 38),
        ("jf_gongitsune", 39),
        ("jf_nezumi", 40),
        ("jf_tebukuro", 41),
        ("jm_kumo", 42),
        ("pf_dora", 43),
        ("pm_alex", 44),
        ("pm_santa", 45),
        ("zf_xiaobei", 46),
        ("zf_xiaoni", 47),
        ("zf_xiaoxiao", 48),
        ("zf_xiaoyi", 49),
        ("zm_yunjian", 50),
        ("zm_yunxi", 51),
        ("zm_yunxia", 52),
        ("zm_yunyang", 53),
    ]
        .iter()
        .copied()
        .collect();
    voice_map
        .get(name)
        .copied()
        .ok_or_else(|| voirs_sdk::VoirsError::config_error(
            format!("Unknown voice: {}", name),
        ))
}
/// Get default voice for language
fn get_default_voice_for_language(lang: &str) -> Result<usize> {
    Ok(
        match lang {
            "en-us" => 4,
            "en-gb" => 20,
            "es" => 28,
            "fr" => 31,
            "hi" => 32,
            "it" => 36,
            "pt-br" => 43,
            "ja" => 38,
            "zh" => 46,
            _ => {
                return Err(
                    voirs_sdk::VoirsError::config_error(
                        format!("Unknown language: {}", lang),
                    ),
                );
            }
        },
    )
}
/// Execute voices command
async fn execute_voices(
    lang: Option<&str>,
    detailed: bool,
    format: &str,
    global: &GlobalOptions,
) -> Result<()> {
    let voices = get_all_voices();
    let filtered_voices: Vec<_> = if let Some(lang_filter) = lang {
        voices
            .into_iter()
            .filter(|v| {
                v.language.to_lowercase().contains(&lang_filter.to_lowercase())
                    || v.language_short == lang_filter
            })
            .collect()
    } else {
        voices
    };
    if !global.quiet {
        match format {
            "table" => {
                println!("🎙️  Kokoro Voices");
                println!(
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                );
                println!();
                if detailed {
                    println!(
                        "{:<4} {:<20} {:<25} {:<10}", "Idx", "Voice Name", "Language",
                        "Gender"
                    );
                    println!("{:-<4} {:-<20} {:-<25} {:-<10}", "", "", "", "");
                    for voice in &filtered_voices {
                        println!(
                            "{:<4} {:<20} {:<25} {:<10}", voice.index, voice.name, voice
                            .language, voice.gender
                        );
                    }
                } else {
                    println!("{:<4} {:<20} {:<10}", "Idx", "Voice Name", "Language");
                    println!("{:-<4} {:-<20} {:-<10}", "", "", "");
                    for voice in &filtered_voices {
                        println!(
                            "{:<4} {:<20} {:<10}", voice.index, voice.name, voice
                            .language_short
                        );
                    }
                }
                println!();
                println!("Total: {} voices", filtered_voices.len());
            }
            "json" => {
                println!("[");
                for (i, voice) in filtered_voices.iter().enumerate() {
                    println!("  {{");
                    println!("    \"index\": {},", voice.index);
                    println!("    \"name\": \"{}\",", voice.name);
                    println!("    \"language\": \"{}\",", voice.language);
                    println!("    \"language_code\": \"{}\",", voice.language_short);
                    println!("    \"gender\": \"{}\"", voice.gender);
                    if i < filtered_voices.len() - 1 {
                        println!("  }},");
                    } else {
                        println!("  }}");
                    }
                }
                println!("]");
            }
            "csv" => {
                println!("index,name,language,language_code,gender");
                for voice in &filtered_voices {
                    println!(
                        "{},{},{},{},{}", voice.index, voice.name, voice.language, voice
                        .language_short, voice.gender
                    );
                }
            }
            _ => {
                return Err(
                    voirs_sdk::VoirsError::config_error(
                        format!("Unknown format: {}. Use table, json, or csv", format),
                    ),
                );
            }
        }
    }
    Ok(())
}
/// Get all 54 voices with metadata
fn get_all_voices() -> Vec<VoiceInfo> {
    vec![
        VoiceInfo { index : 0, name : "af_alloy", language : "English (American)",
        language_short : "en-us", gender : "Female", }, VoiceInfo { index : 1, name :
        "af_aoede", language : "English (American)", language_short : "en-us", gender :
        "Female", }, VoiceInfo { index : 2, name : "af_bella", language :
        "English (American)", language_short : "en-us", gender : "Female", }, VoiceInfo {
        index : 3, name : "af_heart", language : "English (American)", language_short :
        "en-us", gender : "Female", }, VoiceInfo { index : 4, name : "af_jessica",
        language : "English (American)", language_short : "en-us", gender : "Female", },
        VoiceInfo { index : 5, name : "af_kore", language : "English (American)",
        language_short : "en-us", gender : "Female", }, VoiceInfo { index : 6, name :
        "af_nicole", language : "English (American)", language_short : "en-us", gender :
        "Female", }, VoiceInfo { index : 7, name : "af_nova", language :
        "English (American)", language_short : "en-us", gender : "Female", }, VoiceInfo {
        index : 8, name : "af_river", language : "English (American)", language_short :
        "en-us", gender : "Female", }, VoiceInfo { index : 9, name : "af_sarah", language
        : "English (American)", language_short : "en-us", gender : "Female", }, VoiceInfo
        { index : 10, name : "af_sky", language : "English (American)", language_short :
        "en-us", gender : "Female", }, VoiceInfo { index : 11, name : "am_adam", language
        : "English (American)", language_short : "en-us", gender : "Male", }, VoiceInfo {
        index : 12, name : "am_echo", language : "English (American)", language_short :
        "en-us", gender : "Male", }, VoiceInfo { index : 13, name : "am_eric", language :
        "English (American)", language_short : "en-us", gender : "Male", }, VoiceInfo {
        index : 14, name : "am_fenrir", language : "English (American)", language_short :
        "en-us", gender : "Male", }, VoiceInfo { index : 15, name : "am_liam", language :
        "English (American)", language_short : "en-us", gender : "Male", }, VoiceInfo {
        index : 16, name : "am_michael", language : "English (American)", language_short
        : "en-us", gender : "Male", }, VoiceInfo { index : 17, name : "am_onyx", language
        : "English (American)", language_short : "en-us", gender : "Male", }, VoiceInfo {
        index : 18, name : "am_puck", language : "English (American)", language_short :
        "en-us", gender : "Male", }, VoiceInfo { index : 19, name : "am_santa", language
        : "English (American)", language_short : "en-us", gender : "Male", }, VoiceInfo {
        index : 20, name : "bf_alice", language : "English (British)", language_short :
        "en-gb", gender : "Female", }, VoiceInfo { index : 21, name : "bf_emma", language
        : "English (British)", language_short : "en-gb", gender : "Female", }, VoiceInfo
        { index : 22, name : "bf_isabella", language : "English (British)",
        language_short : "en-gb", gender : "Female", }, VoiceInfo { index : 23, name :
        "bf_lily", language : "English (British)", language_short : "en-gb", gender :
        "Female", }, VoiceInfo { index : 24, name : "bm_daniel", language :
        "English (British)", language_short : "en-gb", gender : "Male", }, VoiceInfo {
        index : 25, name : "bm_fable", language : "English (British)", language_short :
        "en-gb", gender : "Male", }, VoiceInfo { index : 26, name : "bm_george", language
        : "English (British)", language_short : "en-gb", gender : "Male", }, VoiceInfo {
        index : 27, name : "bm_lewis", language : "English (British)", language_short :
        "en-gb", gender : "Male", }, VoiceInfo { index : 28, name : "ef_dora", language :
        "Spanish", language_short : "es", gender : "Female", }, VoiceInfo { index : 29,
        name : "em_alex", language : "Spanish", language_short : "es", gender : "Male",
        }, VoiceInfo { index : 30, name : "em_santa", language : "Spanish",
        language_short : "es", gender : "Male", }, VoiceInfo { index : 31, name :
        "ff_siwis", language : "French", language_short : "fr", gender : "Female", },
        VoiceInfo { index : 32, name : "hf_alpha", language : "Hindi", language_short :
        "hi", gender : "Female", }, VoiceInfo { index : 33, name : "hf_beta", language :
        "Hindi", language_short : "hi", gender : "Female", }, VoiceInfo { index : 34,
        name : "hm_omega", language : "Hindi", language_short : "hi", gender : "Male", },
        VoiceInfo { index : 35, name : "hm_psi", language : "Hindi", language_short :
        "hi", gender : "Male", }, VoiceInfo { index : 36, name : "if_sara", language :
        "Italian", language_short : "it", gender : "Female", }, VoiceInfo { index : 37,
        name : "im_nicola", language : "Italian", language_short : "it", gender : "Male",
        }, VoiceInfo { index : 38, name : "jf_alpha", language : "Japanese",
        language_short : "ja", gender : "Female", }, VoiceInfo { index : 39, name :
        "jf_gongitsune", language : "Japanese", language_short : "ja", gender : "Female",
        }, VoiceInfo { index : 40, name : "jf_nezumi", language : "Japanese",
        language_short : "ja", gender : "Female", }, VoiceInfo { index : 41, name :
        "jf_tebukuro", language : "Japanese", language_short : "ja", gender : "Female",
        }, VoiceInfo { index : 42, name : "jm_kumo", language : "Japanese",
        language_short : "ja", gender : "Male", }, VoiceInfo { index : 43, name :
        "pf_dora", language : "Portuguese (Brazilian)", language_short : "pt-br", gender
        : "Female", }, VoiceInfo { index : 44, name : "pm_alex", language :
        "Portuguese (Brazilian)", language_short : "pt-br", gender : "Male", }, VoiceInfo
        { index : 45, name : "pm_santa", language : "Portuguese (Brazilian)",
        language_short : "pt-br", gender : "Male", }, VoiceInfo { index : 46, name :
        "zf_xiaobei", language : "Chinese (Mandarin)", language_short : "zh", gender :
        "Female", }, VoiceInfo { index : 47, name : "zf_xiaoni", language :
        "Chinese (Mandarin)", language_short : "zh", gender : "Female", }, VoiceInfo {
        index : 48, name : "zf_xiaoxiao", language : "Chinese (Mandarin)", language_short
        : "zh", gender : "Female", }, VoiceInfo { index : 49, name : "zf_xiaoyi",
        language : "Chinese (Mandarin)", language_short : "zh", gender : "Female", },
        VoiceInfo { index : 50, name : "zm_yunjian", language : "Chinese (Mandarin)",
        language_short : "zh", gender : "Male", }, VoiceInfo { index : 51, name :
        "zm_yunxi", language : "Chinese (Mandarin)", language_short : "zh", gender :
        "Male", }, VoiceInfo { index : 52, name : "zm_yunxia", language :
        "Chinese (Mandarin)", language_short : "zh", gender : "Male", }, VoiceInfo {
        index : 53, name : "zm_yunyang", language : "Chinese (Mandarin)", language_short
        : "zh", gender : "Male", },
    ]
}
/// Execute languages command
async fn execute_languages(show_ipa_method: bool, global: &GlobalOptions) -> Result<()> {
    if !global.quiet {
        println!("🌍 Kokoro Supported Languages");
        println!(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        );
        println!();
        let languages = vec![
            ("en-us", "English (American)", "eSpeak NG", "11 female, 9 male",), ("en-gb",
            "English (British)", "eSpeak NG", "4 female, 4 male",), ("es", "Spanish",
            "eSpeak NG", "1 female, 2 male"), ("fr", "French", "eSpeak NG", "1 female"),
            ("hi", "Hindi", "eSpeak NG", "2 female, 2 male"), ("it", "Italian",
            "eSpeak NG", "1 female, 1 male"), ("pt-br", "Portuguese (Brazilian)",
            "eSpeak NG", "1 female, 2 male",), ("ja", "Japanese", "misaki library",
            "4 female, 1 male"), ("zh", "Chinese (Mandarin)", "misaki library",
            "4 female, 4 male",),
        ];
        if show_ipa_method {
            println!(
                "{:<8} {:<25} {:<20} {:<20}", "Code", "Language", "IPA Method", "Voices"
            );
            println!("{:-<8} {:-<25} {:-<20} {:-<20}", "", "", "", "");
            for (code, name, ipa_method, voices) in &languages {
                println!("{:<8} {:<25} {:<20} {:<20}", code, name, ipa_method, voices);
            }
        } else {
            println!("{:<8} {:<25} {:<20}", "Code", "Language", "Voices");
            println!("{:-<8} {:-<25} {:-<20}", "", "", "");
            for (code, name, _, voices) in &languages {
                println!("{:<8} {:<25} {:<20}", code, name, voices);
            }
        }
        println!();
        println!("Total: {} languages, 54 voices", languages.len());
        println!();
        println!(
            "💡 Tip: IPA phonemes are auto-generated by default (use --ipa to provide manual phonemes)"
        );
        println!("   (Supported: en-us, en-gb, es, fr, hi, it, pt-br)");
        println!();
        println!(
            "   For Japanese and Chinese, use misaki library or provide manual IPA"
        );
    }
    Ok(())
}
/// Execute test command
async fn execute_test(
    model_dir: Option<&std::path::Path>,
    lang: Option<&str>,
    global: &GlobalOptions,
) -> Result<()> {
    use voirs_acoustic::vits::onnx_kokoro::KokoroOnnxInference;
    if !global.quiet {
        println!("🧪 Testing Kokoro Model Installation");
        println!(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        );
        println!();
    }
    let model_path = if let Some(dir) = model_dir {
        dir.to_path_buf()
    } else {
        let temp_dir = std::env::temp_dir();
        temp_dir.join("voirs_models/kokoro-zh")
    };
    if !global.quiet {
        println!("📁 Model directory: {}", model_path.display());
        println!();
    }
    let onnx_file = model_path.join("kokoro-v1.0.onnx");
    let config_file = model_path.join("config.json");
    let voices_npz = model_path.join("voices-v1.0.bin");
    let voices_avg = model_path.join("voices_averaged.bin");
    if !global.quiet {
        print!("📋 Checking model files... ");
        std::io::Write::flush(&mut std::io::stdout())
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(
                    format!("Failed to flush stdout: {}", e),
                )
            })?;
    }
    let mut missing_files = Vec::new();
    if !onnx_file.exists() {
        missing_files.push("kokoro-v1.0.onnx");
    }
    if !config_file.exists() {
        missing_files.push("config.json");
    }
    if !voices_npz.exists() && !voices_avg.exists() {
        missing_files.push("voices-v1.0.bin or voices_averaged.bin");
    }
    if !missing_files.is_empty() {
        if !global.quiet {
            println!("❌");
            println!();
            println!("Missing files:");
            for file in &missing_files {
                println!("  - {}", file);
            }
            println!();
            println!("💡 Run 'voirs kokoro download' to download model files");
        }
        return Err(
            voirs_sdk::VoirsError::config_error("Missing Kokoro model files".to_string()),
        );
    }
    if !global.quiet {
        println!("✓");
        println!("  - kokoro-v1.0.onnx: ✓");
        println!("  - config.json: ✓");
        if voices_avg.exists() {
            println!("  - voices_averaged.bin: ✓");
        } else {
            println!("  - voices-v1.0.bin: ✓");
        }
        println!();
    }
    if !global.quiet {
        print!("📥 Loading model... ");
        std::io::Write::flush(&mut std::io::stdout())
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(
                    format!("Failed to flush stdout: {}", e),
                )
            })?;
    }
    let mut model = KokoroOnnxInference::from_kokoro_files(&model_path)?;
    if !global.quiet {
        println!("✓");
        println!();
    }
    let test_cases = if let Some(lang_code) = lang {
        vec![(lang_code, get_test_text(lang_code) ?)]
    } else {
        vec![
            ("en-us", "Hello world"), ("en-gb", "Hello world"), ("es", "Hola mundo"),
            ("fr", "Bonjour le monde"), ("hi", "नमस्ते दुनिया"),
            ("it", "Ciao mondo"), ("pt-br", "Olá mundo"),
        ]
    };
    if !global.quiet {
        println!("🎵 Testing synthesis...");
        println!();
    }
    for (test_lang, test_text) in test_cases {
        if !global.quiet {
            print!("  {} ({})... ", test_text, test_lang);
            std::io::Write::flush(&mut std::io::stdout())
                .map_err(|e| {
                    voirs_sdk::VoirsError::config_error(
                        format!("Failed to flush stdout: {}", e),
                    )
                })?;
        }
        let espeak_voice = match test_lang {
            "en-us" => "en-us",
            "en-gb" => "en-gb",
            "es" => "es",
            "fr" => "fr-fr",
            "hi" => "hi",
            "it" => "it",
            "pt-br" => "pt-br",
            _ => {
                if !global.quiet {
                    println!("⊘ (manual IPA needed)");
                }
                continue;
            }
        };
        let output = Command::new("espeak-ng")
            .arg("-v")
            .arg(espeak_voice)
            .arg("-q")
            .arg("--ipa")
            .arg(test_text)
            .output();
        let phonemes = match output {
            Ok(out) if out.status.success() => {
                String::from_utf8(out.stdout)
                    .map_err(|e| voirs_sdk::VoirsError::config_error(
                        format!("Invalid UTF-8: {}", e),
                    ))?
                    .lines()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .trim()
                    .to_string()
            }
            _ => {
                if !global.quiet {
                    println!("⊘ (eSpeak NG not available)");
                }
                continue;
            }
        };
        let voice_idx = get_default_voice_for_language(test_lang)?;
        let result = model.synthesize_trim_end(&phonemes, voice_idx, 1.0);
        if result.is_ok() {
            if !global.quiet {
                println!("✓");
            }
        } else {
            if !global.quiet {
                println!("❌");
            }
            return Err(
                voirs_sdk::VoirsError::config_error(
                    format!("Synthesis failed for {}: {:?}", test_lang, result),
                ),
            );
        }
    }
    if !global.quiet {
        println!();
        println!("✅ All tests passed!");
        println!();
        println!("💡 Model is ready to use. Try:");
        println!("   voirs kokoro synth \"Hello world\" output.wav --lang en-us");
    }
    Ok(())
}
/// Get test text for language
fn get_test_text(lang: &str) -> Result<&'static str> {
    Ok(
        match lang {
            "en-us" | "en-gb" => "Hello world",
            "es" => "Hola mundo",
            "fr" => "Bonjour le monde",
            "hi" => "नमस्ते दुनिया",
            "it" => "Ciao mondo",
            "pt-br" => "Olá mundo",
            "ja" => "こんにちは",
            "zh" => "你好",
            _ => {
                return Err(
                    voirs_sdk::VoirsError::config_error(
                        format!("Unknown language: {}", lang),
                    ),
                );
            }
        },
    )
}
/// Execute download command
async fn execute_download(
    output: Option<&std::path::Path>,
    force: bool,
    global: &GlobalOptions,
) -> Result<()> {
    use futures_util::StreamExt;
    use indicatif::{ProgressBar, ProgressStyle};
    if !global.quiet {
        println!("📥 Downloading Kokoro Model Files");
        println!(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        );
        println!();
    }
    let output_dir = if let Some(dir) = output {
        dir.to_path_buf()
    } else {
        let temp_dir = std::env::temp_dir();
        temp_dir.join("voirs_models/kokoro-zh")
    };
    if !global.quiet {
        println!("📁 Output directory: {}", output_dir.display());
        println!();
    }
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to create directory: {}", e),
            )
        })?;
    let files = vec![
        ("model.onnx",
        "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_q8f16.onnx",),
        ("config.json",
        "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json",),
        ("af.bin",
        "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af.bin",),
    ];
    let client = reqwest::Client::builder()
        .user_agent("voirs-cli")
        .build()
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to create HTTP client: {}", e),
            )
        })?;
    for (filename, url) in &files {
        let output_path = output_dir.join(filename);
        if output_path.exists() && !force {
            if !global.quiet {
                println!("⊘ {} already exists (use --force to re-download)", filename);
            }
            continue;
        }
        if !global.quiet {
            println!("📥 Downloading {}...", filename);
        }
        let response = client
            .get(*url)
            .send()
            .await
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(
                    format!("Failed to download {}: {}", filename, e),
                )
            })?;
        if !response.status().is_success() {
            return Err(
                voirs_sdk::VoirsError::config_error(
                    format!(
                        "Failed to download {}: HTTP {}", filename, response.status()
                    ),
                ),
            );
        }
        let total_size = response.content_length().unwrap_or(0);
        let pb = if !global.quiet && total_size > 0 {
            let pb = ProgressBar::new(total_size);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("   [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };
        let mut file = std::fs::File::create(&output_path)
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(
                    format!("Failed to create file {}: {}", filename, e),
                )
            })?;
        let mut stream = response.bytes_stream();
        let mut downloaded = 0u64;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk
                .map_err(|e| {
                    voirs_sdk::VoirsError::config_error(
                        format!("Failed to download chunk: {}", e),
                    )
                })?;
            std::io::Write::write_all(&mut file, &chunk)
                .map_err(|e| {
                    voirs_sdk::VoirsError::config_error(
                        format!("Failed to write to file: {}", e),
                    )
                })?;
            downloaded += chunk.len() as u64;
            if let Some(ref pb) = pb {
                pb.set_position(downloaded);
            }
        }
        if let Some(pb) = pb {
            pb.finish_and_clear();
        }
        if !global.quiet {
            println!("   ✓ Downloaded {} ({} bytes)", filename, downloaded);
        }
    }
    if !global.quiet {
        println!();
        println!("✅ Basic model files downloaded!");
        println!();
        println!("📦 Downloaded:");
        println!("   - model.onnx (quantized 8-bit, 86MB)");
        println!("   - config.json");
        println!("   - af.bin (sample voice file)");
        println!();
        println!("⚠️  Note:");
        println!(
            "   The individual .bin voice files from the ONNX repository are not yet"
        );
        println!(
            "   compatible with the VoiRS loader. For full functionality, please use"
        );
        println!("   the Rust examples which directly load from the ONNX model.");
        println!();
        println!("💡 To use Kokoro with VoiRS:");
        println!(
            "   1. Run examples: cargo run --example kokoro_espeak_auto_demo --features onnx"
        );
        println!("   2. See examples/KOKORO_EXAMPLES.md for detailed usage");
        println!();
        println!("📚 Additional resources:");
        println!(
            "   - ONNX Model: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX"
        );
        println!(
            "   - All voices: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main/voices"
        );
    }
    Ok(())
}
/// Execute text-to-ipa command
async fn execute_text_to_ipa(
    text: &str,
    lang: &str,
    method: &str,
    output: Option<&std::path::Path>,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("🔊 Converting Text to IPA Phonemes");
        println!(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        );
        println!();
        println!("📝 Text: \"{}\"", text);
        println!("   Language: {}", lang);
        println!("   Method: {}", method);
        println!();
    }
    let actual_method = if method == "auto" {
        match lang {
            "ja" | "zh" => "misaki",
            _ => "espeak",
        }
    } else {
        method
    };
    let phonemes = match actual_method {
        "espeak" => {
            let espeak_voice = match lang {
                "en-us" => "en-us",
                "en-gb" => "en-gb",
                "es" => "es",
                "fr" => "fr-fr",
                "hi" => "hi",
                "it" => "it",
                "pt-br" => "pt-br",
                _ => {
                    return Err(
                        voirs_sdk::VoirsError::config_error(
                            format!(
                                "Language '{}' is not supported by eSpeak NG. Use 'misaki' for Japanese/Chinese.",
                                lang
                            ),
                        ),
                    );
                }
            };
            if !global.quiet {
                print!("🔊 Generating IPA with eSpeak NG... ");
                std::io::Write::flush(&mut std::io::stdout())
                    .map_err(|e| {
                        voirs_sdk::VoirsError::config_error(
                            format!("Failed to flush stdout: {}", e),
                        )
                    })?;
            }
            let output = Command::new("espeak-ng")
                .arg("-v")
                .arg(espeak_voice)
                .arg("-q")
                .arg("--ipa")
                .arg(text)
                .output()
                .map_err(|e| {
                    voirs_sdk::VoirsError::config_error(
                        format!("Failed to run eSpeak NG. Is it installed? Error: {}", e),
                    )
                })?;
            if !output.status.success() {
                return Err(
                    voirs_sdk::VoirsError::config_error(
                        format!(
                            "eSpeak NG failed: {}", String::from_utf8_lossy(& output
                            .stderr)
                        ),
                    ),
                );
            }
            let ipa = String::from_utf8(output.stdout)
                .map_err(|e| voirs_sdk::VoirsError::config_error(
                    format!("Invalid UTF-8: {}", e),
                ))?
                .lines()
                .collect::<Vec<_>>()
                .join(" ")
                .trim()
                .to_string();
            if !global.quiet {
                println!("✓");
            }
            ipa
        }
        "misaki" => {
            return Err(
                voirs_sdk::VoirsError::config_error(
                    "misaki method requires Python library. Please use Python script or provide manual IPA."
                        .to_string(),
                ),
            );
        }
        _ => {
            return Err(
                voirs_sdk::VoirsError::config_error(
                    format!(
                        "Unknown method: {}. Use 'auto', 'espeak', or 'misaki'", method
                    ),
                ),
            );
        }
    };
    if !global.quiet {
        println!();
        println!("📤 IPA Phonemes:");
        println!("   {}", phonemes);
    }
    if let Some(output_path) = output {
        std::fs::write(output_path, &phonemes)
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(
                    format!("Failed to write output file: {}", e),
                )
            })?;
        if !global.quiet {
            println!();
            println!("✅ IPA saved to: {}", output_path.display());
        }
    } else if global.quiet {
        println!("{}", phonemes);
    }
    Ok(())
}
/// Execute batch command
async fn execute_batch(
    input: &PathBuf,
    output_dir: &PathBuf,
    jobs: usize,
    manual_ipa: bool,
    global: &GlobalOptions,
) -> Result<()> {
    use std::io::BufRead;
    use voirs_acoustic::vits::onnx_kokoro::KokoroOnnxInference;
    if !global.quiet {
        println!("📦 Batch Synthesis from CSV");
        println!(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        );
        println!();
        println!("📄 Input: {}", input.display());
        println!("   Output directory: {}", output_dir.display());
        println!("   Parallel jobs: {}", jobs);
        println!();
    }
    std::fs::create_dir_all(output_dir)
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to create output directory: {}", e),
            )
        })?;
    let file = std::fs::File::open(input)
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to open input file: {}", e),
            )
        })?;
    let reader = std::io::BufReader::new(file);
    let mut lines = reader.lines();
    lines.next();
    #[derive(Debug)]
    struct BatchItem {
        text: String,
        language: String,
        voice: String,
        output_file: String,
    }
    let mut items = Vec::new();
    for (line_num, line) in lines.enumerate() {
        let line = line
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(
                    format!("Failed to read line {}: {}", line_num + 2, e),
                )
            })?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 4 {
            if !global.quiet {
                println!(
                    "⚠️  Skipping line {} (insufficient columns)", line_num + 2
                );
            }
            continue;
        }
        items
            .push(BatchItem {
                text: parts[0].trim().to_string(),
                language: parts[1].trim().to_string(),
                voice: parts[2].trim().to_string(),
                output_file: parts[3].trim().to_string(),
            });
    }
    if items.is_empty() {
        return Err(
            voirs_sdk::VoirsError::config_error(
                "No valid items found in CSV file".to_string(),
            ),
        );
    }
    if !global.quiet {
        println!("📋 Found {} items to process", items.len());
        println!();
    }
    if !global.quiet {
        print!("📥 Loading Kokoro model... ");
        std::io::Write::flush(&mut std::io::stdout())
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(
                    format!("Failed to flush stdout: {}", e),
                )
            })?;
    }
    let temp_dir = std::env::temp_dir();
    let model_path = temp_dir.join("voirs_models/kokoro-zh");
    let mut model = KokoroOnnxInference::from_kokoro_files(&model_path)?;
    if !global.quiet {
        println!("✓");
        println!();
        println!("🎵 Processing items...");
        println!();
    }
    let mut success_count = 0;
    let mut error_count = 0;
    for (idx, item) in items.iter().enumerate() {
        if !global.quiet {
            print!(
                "  [{}/{}] \"{}\" ({})... ", idx + 1, items.len(), item.text, item
                .language
            );
            std::io::Write::flush(&mut std::io::stdout())
                .map_err(|e| {
                    voirs_sdk::VoirsError::config_error(
                        format!("Failed to flush stdout: {}", e),
                    )
                })?;
        }
        let phonemes = if manual_ipa {
            item.text.clone()
        } else {
            let espeak_voice = match item.language.as_str() {
                "en-us" => "en-us",
                "en-gb" => "en-gb",
                "es" => "es",
                "fr" => "fr-fr",
                "hi" => "hi",
                "it" => "it",
                "pt-br" => "pt-br",
                _ => {
                    if !global.quiet {
                        println!(
                            "❌ (language not supported for auto-IPA, use --manual-ipa)"
                        );
                    }
                    error_count += 1;
                    continue;
                }
            };
            let output = Command::new("espeak-ng")
                .arg("-v")
                .arg(espeak_voice)
                .arg("-q")
                .arg("--ipa")
                .arg(&item.text)
                .output();
            match output {
                Ok(out) if out.status.success() => {
                    String::from_utf8(out.stdout)
                        .unwrap_or_default()
                        .lines()
                        .collect::<Vec<_>>()
                        .join(" ")
                        .trim()
                        .to_string()
                }
                _ => {
                    if !global.quiet {
                        println!("❌ (eSpeak NG failed)");
                    }
                    error_count += 1;
                    continue;
                }
            }
        };
        let voice_idx = if let Ok(idx) = item.voice.parse::<usize>() {
            idx
        } else if let Ok(idx) = get_voice_index_by_name(&item.voice) {
            idx
        } else if let Ok(idx) = get_default_voice_for_language(&item.language) {
            idx
        } else {
            if !global.quiet {
                println!("❌ (invalid voice)");
            }
            error_count += 1;
            continue;
        };
        let result = model.synthesize_trim_end(&phonemes, voice_idx, 1.0);
        match result {
            Ok(audio_samples) => {
                let output_path = output_dir.join(&item.output_file);
                let sample_rate = model.sample_rate();
                if let Err(e) = save_wav(
                    output_path.to_str().unwrap(),
                    &audio_samples,
                    sample_rate,
                ) {
                    if !global.quiet {
                        println!("❌ (failed to save: {})", e);
                    }
                    error_count += 1;
                } else {
                    if !global.quiet {
                        println!("✓");
                    }
                    success_count += 1;
                }
            }
            Err(e) => {
                if !global.quiet {
                    println!("❌ (synthesis failed: {})", e);
                }
                error_count += 1;
            }
        }
    }
    if !global.quiet {
        println!();
        println!(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        );
        println!("✅ Batch processing complete!");
        println!("   Success: {}", success_count);
        println!("   Errors: {}", error_count);
        println!("   Total: {}", items.len());
    }
    if error_count > 0 {
        return Err(
            voirs_sdk::VoirsError::config_error(
                format!("Batch processing completed with {} errors", error_count),
            ),
        );
    }
    Ok(())
}
/// Play audio file using platform-specific command
fn play_audio_file(path: &PathBuf) -> Result<()> {
    #[cfg(target_os = "macos")]
    let (command, args) = ("afplay", vec![path.to_str().unwrap()]);
    #[cfg(target_os = "linux")]
    let (command, args) = {
        if Command::new("aplay").arg("--version").output().is_ok() {
            ("aplay", vec![path.to_str().unwrap()])
        } else if Command::new("paplay").arg("--version").output().is_ok() {
            ("paplay", vec![path.to_str().unwrap()])
        } else {
            return Err(
                voirs_sdk::VoirsError::config_error(
                    "No audio player found. Install 'alsa-utils' (aplay) or 'pulseaudio-utils' (paplay)."
                        .to_string(),
                ),
            );
        }
    };
    #[cfg(target_os = "windows")]
    let (command, args) = (
        "powershell",
        vec![
            "-c", & format!("(New-Object Media.SoundPlayer '{}').PlaySync()", path
            .display()),
        ],
    );
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        return Err(
            voirs_sdk::VoirsError::config_error(
                "Audio playback not supported on this platform".to_string(),
            ),
        );
    }
    let status = Command::new(command)
        .args(&args)
        .status()
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(
                format!("Failed to play audio with '{}': {}", command, e),
            )
        })?;
    if !status.success() {
        return Err(
            voirs_sdk::VoirsError::config_error(
                format!("Audio player '{}' exited with error", command),
            ),
        );
    }
    Ok(())
}
/// Execute config command
async fn execute_config(
    show: bool,
    init: bool,
    path: Option<&std::path::Path>,
    current_config: &KokoroConfig,
    global: &GlobalOptions,
) -> Result<()> {
    if show {
        let config_str = toml::to_string_pretty(current_config)
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(
                    format!("Failed to serialize config: {}", e),
                )
            })?;
        if !global.quiet {
            println!("# Kokoro Configuration");
            println!("#");
            println!("# Config file search paths (in order of preference):");
            for path in get_kokoro_config_paths() {
                if path.exists() {
                    println!("# [✓] {}", path.display());
                } else {
                    println!("# [ ] {}", path.display());
                }
            }
            println!();
        }
        println!("{}", config_str);
        return Ok(());
    }
    if init {
        let config_path = if let Some(p) = path {
            p.to_path_buf()
        } else {
            let config_dir = dirs::config_dir()
                .ok_or_else(|| {
                    voirs_sdk::VoirsError::config_error(
                        "Could not find config directory".to_string(),
                    )
                })?;
            let voirs_config = config_dir.join("voirs");
            std::fs::create_dir_all(&voirs_config)
                .map_err(|e| voirs_sdk::VoirsError::IoError {
                    path: voirs_config.clone(),
                    operation: voirs_sdk::error::IoOperation::Write,
                    source: e,
                })?;
            voirs_config.join("kokoro.toml")
        };
        if config_path.exists() {
            return Err(
                voirs_sdk::VoirsError::config_error(
                    format!(
                        "Config file already exists: {}\nUse --path to specify a different location",
                        config_path.display()
                    ),
                ),
            );
        }
        let config_content = r#"# Kokoro TTS Configuration
# This file contains default settings for Kokoro multilingual TTS commands

# Default language code (en-us, en-gb, es, fr, hi, it, pt-br, ja, zh)
default_lang = "en-us"

# Default voice name (optional, e.g., "af_jessica", "bf_alice")
# default_voice = "af_jessica"

# Default speaking speed (0.5 - 2.0)
default_speed = 1.0

# Path to Kokoro model directory (optional, auto-detected if not specified)
# model_dir = "/path/to/kokoro/models"

# Path to eSpeak NG binary (optional, auto-detected if not specified)
# espeak_path = "/usr/bin/espeak-ng"
"#;
        std::fs::write(&config_path, config_content)
            .map_err(|e| {
                voirs_sdk::VoirsError::IoError {
                    path: config_path.clone(),
                    operation: voirs_sdk::error::IoOperation::Write,
                    source: e,
                }
            })?;
        if !global.quiet {
            println!("✓ Created config file: {}", config_path.display());
            println!(
                "\nYou can now edit this file to customize your Kokoro TTS settings."
            );
        }
        return Ok(());
    }
    if !global.quiet {
        println!("Usage:");
        println!("  voirs kokoro config --show          Show current configuration");
        println!("  voirs kokoro config --init          Initialize default config file");
        println!(
            "  voirs kokoro config --init --path <path>   Initialize at specific path"
        );
    }
    Ok(())
}
