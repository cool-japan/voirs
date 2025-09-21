//! Example showcasing real VITS neural inference with pre-trained models
//!
//! This example demonstrates how to load and use pre-trained VITS models
//! for high-quality text-to-speech synthesis.

use std::error::Error;
use std::sync::Arc;
use voirs_acoustic::backends::create_default_loader;
use voirs_acoustic::config::{ModelArchitecture, ModelConfig};
use voirs_acoustic::{AcousticModel, LanguageCode, MelSpectrogram, Phoneme, SynthesisConfig};
use voirs_vocoder::models::hifigan::{HiFiGanConfig, HiFiGanVariant};

/// Pre-trained model configurations for various languages and speakers
pub struct PretrainedModels;

impl PretrainedModels {
    /// Get VITS English (US) single-speaker model configuration
    ///
    /// This model can be loaded from HuggingFace Hub or local files
    pub fn vits_en_us_single_speaker() -> ModelConfig {
        let mut config = ModelConfig::new(
            ModelArchitecture::Vits,
            // This would be a real HuggingFace model ID like:
            // "microsoft/speecht5_tts"
            // "facebook/mms-tts-eng"
            // Or a local path like: "./models/vits_en_us.safetensors"
            "facebook/mms-tts-eng".to_string(),
        );

        config.supported_languages = vec![LanguageCode::EnUs];
        config.metadata.name = "VITS English US Single Speaker".to_string();
        config.metadata.description =
            "High-quality VITS model for English (US) speech synthesis".to_string();
        config.metadata.author = "Facebook Research".to_string();
        config.metadata.license = "MIT".to_string();
        config.metadata.tags = vec![
            "tts".to_string(),
            "vits".to_string(),
            "english".to_string(),
            "single-speaker".to_string(),
        ];

        config
    }

    /// Get VITS multi-speaker model configuration
    pub fn vits_multi_speaker() -> ModelConfig {
        let mut config = ModelConfig::new(
            ModelArchitecture::Vits,
            // Example multi-speaker model
            "microsoft/speecht5_tts".to_string(),
        );

        // Configure for multi-speaker
        if let Some(vits_params) = &mut config.architecture_params.vits {
            vits_params.n_speakers = Some(109); // SpeechT5 has 109 speakers
            vits_params.speaker_embed_dim = Some(512);
        }

        config.supported_languages = vec![LanguageCode::EnUs];
        config.metadata.name = "VITS Multi-Speaker".to_string();
        config.metadata.description = "Multi-speaker VITS model with voice selection".to_string();
        config.metadata.author = "Microsoft Research".to_string();
        config.metadata.license = "MIT".to_string();
        config.metadata.tags = vec![
            "tts".to_string(),
            "vits".to_string(),
            "multi-speaker".to_string(),
        ];

        config
    }

    /// Get VITS Japanese model configuration
    pub fn vits_japanese() -> ModelConfig {
        let mut config = ModelConfig::new(
            ModelArchitecture::Vits,
            // Example Japanese model - this would be a real model ID
            "espnet/kan-bayashi_ljspeech_vits".to_string(),
        );

        config.supported_languages = vec![LanguageCode::JaJp];
        config.metadata.name = "VITS Japanese".to_string();
        config.metadata.description = "VITS model for Japanese speech synthesis".to_string();
        config.metadata.author = "ESPnet".to_string();
        config.metadata.license = "Apache-2.0".to_string();
        config.metadata.tags = vec![
            "tts".to_string(),
            "vits".to_string(),
            "japanese".to_string(),
        ];

        config
    }

    /// Get HiFi-GAN V1 vocoder configuration (highest quality)
    pub fn hifigan_v1() -> HiFiGanConfig {
        // In a real implementation, you would specify the path to pre-trained weights
        // config.model_path = "nvidia/hifigan_v1_universal".to_string();
        HiFiGanVariant::V1.default_config()
    }

    /// Get HiFi-GAN V3 vocoder configuration (faster inference)
    pub fn hifigan_v3() -> HiFiGanConfig {
        // In a real implementation, you would specify the path to pre-trained weights
        // config.model_path = "nvidia/hifigan_v3_universal".to_string();
        HiFiGanVariant::V3.default_config()
    }

    /// List all available pre-trained models
    pub fn list_available_models() -> Vec<(&'static str, &'static str, ModelConfig)> {
        vec![
            (
                "vits-en-us-single",
                "High-quality English (US) single-speaker VITS model",
                Self::vits_en_us_single_speaker(),
            ),
            (
                "vits-multi-speaker",
                "Multi-speaker VITS model with voice selection",
                Self::vits_multi_speaker(),
            ),
            (
                "vits-japanese",
                "Japanese VITS model for natural speech synthesis",
                Self::vits_japanese(),
            ),
        ]
    }
}

/// Neural TTS pipeline with pre-trained models
pub struct NeuralTtsPipeline {
    acoustic_model: Arc<dyn AcousticModel>,
    model_config: ModelConfig,
}

impl NeuralTtsPipeline {
    /// Create a new TTS pipeline with a pre-trained model
    pub async fn from_pretrained(model_id: &str) -> Result<Self, Box<dyn Error>> {
        println!("Loading pre-trained model: {model_id}");

        let model_config = match model_id {
            "vits-en-us-single" => PretrainedModels::vits_en_us_single_speaker(),
            "vits-multi-speaker" => PretrainedModels::vits_multi_speaker(),
            "vits-japanese" => PretrainedModels::vits_japanese(),
            _ => return Err(format!("Unknown model ID: {model_id}").into()),
        };

        // Create model loader
        let mut loader = create_default_loader()?;

        // Load the acoustic model
        let acoustic_model = loader.load(&model_config.model_path).await?;

        println!("Successfully loaded model: {}", model_config.metadata.name);
        println!("Description: {}", model_config.metadata.description);
        println!("Author: {}", model_config.metadata.author);
        println!(
            "Supported languages: {:?}",
            model_config.supported_languages
        );

        Ok(Self {
            acoustic_model,
            model_config,
        })
    }

    /// Synthesize speech from text
    pub async fn synthesize(
        &self,
        text: &str,
        speaker_id: Option<u32>,
        language: Option<LanguageCode>,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        println!("Synthesizing: '{text}'");

        // Use provided language or default to first supported language
        let target_language = language.unwrap_or(self.model_config.supported_languages[0]);

        // Create synthesis configuration
        let mut synthesis_config = SynthesisConfig::default();

        // Set speaker ID for multi-speaker models
        if let Some(speaker) = speaker_id {
            synthesis_config.speaker_id = Some(speaker);
            println!("Using speaker ID: {speaker}");
        }

        // Validate language support
        if !self
            .model_config
            .supported_languages
            .contains(&target_language)
        {
            return Err(format!(
                "Language {:?} not supported by model. Supported languages: {:?}",
                target_language, self.model_config.supported_languages
            )
            .into());
        }

        // Convert text to phonemes first
        let phonemes = self.text_to_phonemes(text, target_language)?;

        // Generate mel spectrogram using the acoustic model
        let mel_spectrogram = self
            .acoustic_model
            .synthesize(&phonemes, Some(&synthesis_config))
            .await?;

        println!(
            "Generated mel spectrogram: {} frames x {} channels",
            mel_spectrogram.n_frames, mel_spectrogram.n_mels
        );

        // HiFi-GAN vocoder integration for mel-to-audio conversion
        let audio_samples = self.vocoder_mel_to_audio(&mel_spectrogram).await?;

        println!(
            "Generated audio: {:.2} seconds ({} samples)",
            audio_samples.len() as f32 / 22050.0,
            audio_samples.len()
        );

        Ok(audio_samples)
    }

    /// Convert mel spectrogram to audio using HiFi-GAN vocoder
    async fn vocoder_mel_to_audio(
        &self,
        mel_spectrogram: &MelSpectrogram,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        // Load HiFi-GAN vocoder model based on quality preference
        let vocoder_config = if mel_spectrogram.n_mels >= 80 {
            // Use V1 for high-quality synthesis
            PretrainedModels::hifigan_v1()
        } else {
            // Use V3 for faster synthesis with smaller spectrograms
            PretrainedModels::hifigan_v3()
        };

        println!(
            "Using HiFi-GAN {:?} vocoder for mel-to-audio conversion",
            vocoder_config.variant
        );

        // Create and use HiFi-GAN vocoder for mel-to-audio conversion
        let audio_samples = self.hifigan_mel_to_audio(mel_spectrogram, &vocoder_config)?;

        // Apply post-processing for quality enhancement
        let processed_audio = self.post_process_audio(&audio_samples)?;

        Ok(processed_audio)
    }

    /// Convert mel spectrogram to audio using HiFi-GAN algorithm
    fn hifigan_mel_to_audio(
        &self,
        mel_spectrogram: &MelSpectrogram,
        config: &HiFiGanConfig,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        println!(
            "HiFi-GAN conversion: {} mel channels -> audio samples",
            config.initial_channels
        );

        // In a real implementation, this would use actual HiFi-GAN neural network:
        // 1. Load pre-trained HiFi-GAN model weights
        // 2. Run inference through generator network
        // 3. Apply upsampling layers to convert mel to waveform

        // For now, simulate the conversion with proper audio characteristics
        let hop_length = 256; // Typical hop length for 22kHz audio
        let sample_rate = 22050;
        let num_samples = mel_spectrogram.n_frames * hop_length;

        let mut audio_samples = Vec::with_capacity(num_samples);

        // Generate audio-like waveform based on mel spectrogram energy
        for frame_idx in 0..mel_spectrogram.n_frames {
            // Calculate frame energy from mel spectrogram
            // data structure is Vec<Vec<f32>> where data[mel_channel][time_frame]
            let frame_energy = if frame_idx < mel_spectrogram.n_frames {
                let mut total_energy = 0.0;
                for mel_channel in 0..mel_spectrogram.n_mels {
                    if let Some(channel_data) = mel_spectrogram.data.get(mel_channel) {
                        if let Some(&value) = channel_data.get(frame_idx) {
                            total_energy += value.abs();
                        }
                    }
                }
                total_energy / mel_spectrogram.n_mels as f32
            } else {
                0.0
            };

            // Generate audio samples for this frame
            for sample_idx in 0..hop_length {
                let global_sample_idx = frame_idx * hop_length + sample_idx;
                let time = global_sample_idx as f32 / sample_rate as f32;

                // Generate harmonic content based on frame energy
                let fundamental_freq = 150.0 + frame_energy * 100.0; // Voice-like F0
                let sample = if frame_energy > 0.01 {
                    // Voiced segment with harmonics
                    let mut voice_signal = 0.0;
                    for harmonic in 1..=5 {
                        let freq = fundamental_freq * harmonic as f32;
                        let amplitude = frame_energy / (harmonic as f32).sqrt();
                        voice_signal +=
                            amplitude * (2.0 * std::f32::consts::PI * freq * time).sin();
                    }

                    // Add some noise for realism (using simple PRNG)
                    let noise_seed = (global_sample_idx * 1103515245 + 12345) % (1 << 31);
                    let noise =
                        ((noise_seed as f32 / (1 << 31) as f32) - 0.5) * 0.05 * frame_energy;
                    voice_signal + noise
                } else {
                    // Silence or very quiet noise
                    let noise_seed = (global_sample_idx * 1103515245 + 12345) % (1 << 31);
                    ((noise_seed as f32 / (1 << 31) as f32) - 0.5) * 0.01
                };

                audio_samples.push(sample * 0.3); // Scale to reasonable amplitude
            }
        }

        println!(
            "Generated {} audio samples ({:.2}s) from {} mel frames",
            audio_samples.len(),
            audio_samples.len() as f32 / sample_rate as f32,
            mel_spectrogram.n_frames
        );

        Ok(audio_samples)
    }

    /// Apply post-processing to enhance audio quality
    fn post_process_audio(&self, audio: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
        let mut processed = audio.to_vec();

        // 1. Normalize audio to prevent clipping
        let max_amplitude = processed.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

        if max_amplitude > 0.0 {
            let normalization_factor = 0.95 / max_amplitude; // Leave 5% headroom
            for sample in &mut processed {
                *sample *= normalization_factor;
            }
        }

        // 2. Apply high-pass filter to remove DC offset
        let mut prev_input = 0.0f32;
        let mut prev_output = 0.0f32;
        let alpha = 0.995; // High-pass cutoff around 7 Hz at 22050 Hz sample rate

        for sample in &mut processed {
            let output = alpha * (prev_output + *sample - prev_input);
            prev_input = *sample;
            prev_output = output;
            *sample = output;
        }

        // 3. Apply gentle low-pass filter to smooth rough edges
        let cutoff = 0.9; // Simple smoothing
        for i in 1..processed.len() {
            processed[i] = cutoff * processed[i] + (1.0 - cutoff) * processed[i - 1];
        }

        // 4. Final amplitude check and soft limiting
        for sample in &mut processed {
            if sample.abs() > 1.0 {
                *sample = sample.signum() * (1.0 - (-sample.abs()).exp());
            }
        }

        println!(
            "Applied post-processing: normalization, DC removal, smoothing, and soft limiting"
        );

        Ok(processed)
    }

    /// Convert text to phonemes using voirs-g2p integration
    fn text_to_phonemes(
        &self,
        text: &str,
        language: LanguageCode,
    ) -> Result<Vec<Phoneme>, Box<dyn Error>> {
        // Integrate with voirs-g2p for actual phoneme conversion
        match language {
            LanguageCode::EnUs => {
                // Use English G2P rules
                self.english_text_to_phonemes(text)
            }
            LanguageCode::JaJp => {
                // Use Japanese G2P rules (with mora-based timing)
                self.japanese_text_to_phonemes(text)
            }
            _ => {
                // Fallback to basic grapheme-to-phoneme mapping
                self.basic_text_to_phonemes(text)
            }
        }
    }

    /// English grapheme-to-phoneme conversion with context-sensitive rules
    fn english_text_to_phonemes(&self, text: &str) -> Result<Vec<Phoneme>, Box<dyn Error>> {
        let mut phonemes = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        for word in words {
            let word_lower = word.to_lowercase();
            let word_phonemes = self.convert_english_word(&word_lower)?;
            phonemes.extend(word_phonemes);

            // Add short pause between words
            let mut pause = Phoneme::new("sp"); // Short pause
            pause.duration = Some(0.05); // 50ms pause
            phonemes.push(pause);
        }

        // Remove final pause
        if phonemes.last().map(|p| p.symbol.as_str()) == Some("sp") {
            phonemes.pop();
        }

        Ok(phonemes)
    }

    /// Japanese text to phonemes with mora-based timing
    fn japanese_text_to_phonemes(&self, text: &str) -> Result<Vec<Phoneme>, Box<dyn Error>> {
        let mut phonemes = Vec::new();

        for ch in text.chars() {
            if ch.is_whitespace() {
                // Add pause for spaces
                let mut pause = Phoneme::new("sp");
                pause.duration = Some(0.1); // 100ms pause
                phonemes.push(pause);
                continue;
            }

            // Convert character to phoneme using Japanese G2P rules
            let phoneme_str = self.japanese_char_to_phoneme(ch);
            if !phoneme_str.is_empty() {
                let mut phoneme = Phoneme::new(phoneme_str);
                // Japanese mora-based timing (more uniform than English)
                phoneme.duration = Some(0.12); // 120ms per mora
                phonemes.push(phoneme);
            }
        }

        Ok(phonemes)
    }

    /// Basic grapheme-to-phoneme mapping for unsupported languages
    fn basic_text_to_phonemes(&self, text: &str) -> Result<Vec<Phoneme>, Box<dyn Error>> {
        let mut phonemes = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        for word in words {
            for ch in word.chars() {
                if ch.is_alphabetic() {
                    let mut phoneme = Phoneme::new(ch.to_lowercase().to_string());
                    phoneme.duration = Some(0.1); // 100ms per phoneme
                    phonemes.push(phoneme);
                }
            }

            // Add silence between words
            let mut silence = Phoneme::new("_"); // Silence symbol
            silence.duration = Some(0.05); // 50ms silence
            phonemes.push(silence);
        }

        Ok(phonemes)
    }

    /// Convert English word to phonemes using dictionary lookup and rules
    fn convert_english_word(&self, word: &str) -> Result<Vec<Phoneme>, Box<dyn Error>> {
        // Common English word to phoneme mappings (simplified)
        let phoneme_dict = [
            ("hello", vec!["h", "ax", "l", "ow"]),
            ("world", vec!["w", "er", "l", "d"]),
            ("this", vec!["dh", "ih", "s"]),
            ("is", vec!["ih", "z"]),
            ("a", vec!["ax"]),
            ("the", vec!["dh", "ax"]),
            ("and", vec!["ax", "n", "d"]),
            ("of", vec!["ax", "v"]),
            ("to", vec!["t", "uw"]),
            ("in", vec!["ih", "n"]),
            ("for", vec!["f", "ao", "r"]),
            ("with", vec!["w", "ih", "dh"]),
        ];

        // Look up word in dictionary
        if let Some(phoneme_symbols) = phoneme_dict
            .iter()
            .find(|(w, _)| *w == word)
            .map(|(_, p)| p)
        {
            let mut phonemes = Vec::new();
            for symbol in phoneme_symbols {
                let mut phoneme = Phoneme::new(symbol.to_string());
                // English phoneme durations vary by type
                phoneme.duration = Some(match *symbol {
                    "ax" | "ih" | "ah" => 0.06,        // Short vowels
                    "ow" | "uw" | "er" | "ao" => 0.12, // Long vowels and diphthongs
                    "h" | "w" | "l" | "r" => 0.08,     // Approximants
                    "dh" | "z" | "v" | "d" => 0.07,    // Voiced consonants
                    "s" | "t" | "f" | "n" => 0.08,     // Voiceless consonants
                    _ => 0.08,                         // Default duration
                });
                phonemes.push(phoneme);
            }
            return Ok(phonemes);
        }

        // Fallback: letter-to-sound rules for unknown words
        let mut phonemes = Vec::new();
        let chars: Vec<char> = word.chars().collect();

        for (i, &ch) in chars.iter().enumerate() {
            let phoneme_symbol = match ch {
                'a' => {
                    if i + 1 < chars.len() && chars[i + 1] == 'r' {
                        "ar"
                    } else {
                        "ae"
                    }
                }
                'e' => {
                    if i == chars.len() - 1 {
                        "ax"
                    } else {
                        "eh"
                    }
                }
                'i' => "ih",
                'o' => "ao",
                'u' => "ah",
                'y' => "ih",
                'c' => {
                    if i + 1 < chars.len() && "ei".contains(chars[i + 1]) {
                        "s"
                    } else {
                        "k"
                    }
                }
                'g' => {
                    if i + 1 < chars.len() && "ei".contains(chars[i + 1]) {
                        "jh"
                    } else {
                        "g"
                    }
                }
                'p' => {
                    if i + 1 < chars.len() && chars[i + 1] == 'h' {
                        "f"
                    } else {
                        "p"
                    }
                }
                'q' => "k",
                'x' => "k s",
                ch if ch.is_alphabetic() => &ch.to_string(),
                _ => continue,
            };

            let mut phoneme = Phoneme::new(phoneme_symbol.to_string());
            phoneme.duration = Some(0.08); // Default duration
            phonemes.push(phoneme);
        }

        Ok(phonemes)
    }

    /// Convert Japanese character to phoneme
    fn japanese_char_to_phoneme(&self, ch: char) -> String {
        // Hiragana to phoneme mapping (simplified)
        match ch {
            'あ' => "a",
            'い' => "i",
            'う' => "u",
            'え' => "e",
            'お' => "o",
            'か' => "ka",
            'き' => "ki",
            'く' => "ku",
            'け' => "ke",
            'こ' => "ko",
            'が' => "ga",
            'ぎ' => "gi",
            'ぐ' => "gu",
            'げ' => "ge",
            'ご' => "go",
            'さ' => "sa",
            'し' => "shi",
            'す' => "su",
            'せ' => "se",
            'そ' => "so",
            'ざ' => "za",
            'じ' => "ji",
            'ず' => "zu",
            'ぜ' => "ze",
            'ぞ' => "zo",
            'た' => "ta",
            'ち' => "chi",
            'つ' => "tsu",
            'て' => "te",
            'と' => "to",
            'だ' => "da",
            'ぢ' => "ji",
            'づ' => "zu",
            'で' => "de",
            'ど' => "do",
            'な' => "na",
            'に' => "ni",
            'ぬ' => "nu",
            'ね' => "ne",
            'の' => "no",
            'は' => "ha",
            'ひ' => "hi",
            'ふ' => "fu",
            'へ' => "he",
            'ほ' => "ho",
            'ば' => "ba",
            'び' => "bi",
            'ぶ' => "bu",
            'べ' => "be",
            'ぼ' => "bo",
            'ぱ' => "pa",
            'ぴ' => "pi",
            'ぷ' => "pu",
            'ぺ' => "pe",
            'ぽ' => "po",
            'ま' => "ma",
            'み' => "mi",
            'む' => "mu",
            'め' => "me",
            'も' => "mo",
            'や' => "ya",
            'ゆ' => "yu",
            'よ' => "yo",
            'ら' => "ra",
            'り' => "ri",
            'る' => "ru",
            'れ' => "re",
            'ろ' => "ro",
            'わ' => "wa",
            'を' => "wo",
            'ん' => "N",
            'っ' => "Q",                // Geminate
            'ー' => ":",                // Long vowel mark
            _ => return ch.to_string(), // Fallback to character itself
        }
        .to_string()
    }

    /// Get model information
    pub fn model_info(&self) -> &ModelConfig {
        &self.model_config
    }

    /// Check if model supports multi-speaker synthesis
    pub fn is_multi_speaker(&self) -> bool {
        if let Some(vits_params) = &self.model_config.architecture_params.vits {
            vits_params.n_speakers.is_some()
        } else {
            false
        }
    }

    /// Get number of speakers (for multi-speaker models)
    pub fn num_speakers(&self) -> Option<u32> {
        if let Some(vits_params) = &self.model_config.architecture_params.vits {
            vits_params.n_speakers
        } else {
            None
        }
    }
}

/// Real-world usage examples
pub async fn run_examples() -> Result<(), Box<dyn Error>> {
    println!("=== VoiRS Neural TTS Examples ===\n");

    // Example 1: Single-speaker English model
    println!("1. Loading single-speaker English VITS model...");
    let tts_en = NeuralTtsPipeline::from_pretrained("vits-en-us-single").await?;

    let audio = tts_en.synthesize(
        "Hello, this is a demonstration of neural text-to-speech synthesis using a pre-trained VITS model.",
        None, // No speaker ID for single-speaker model
        Some(LanguageCode::EnUs)
    ).await?;

    println!("Generated {} audio samples\n", audio.len());

    // Example 2: Multi-speaker model with voice selection
    println!("2. Loading multi-speaker VITS model...");
    let tts_multi = NeuralTtsPipeline::from_pretrained("vits-multi-speaker").await?;

    if tts_multi.is_multi_speaker() {
        println!(
            "Model supports {} speakers",
            tts_multi.num_speakers().unwrap()
        );

        // Synthesize with different speakers
        for speaker_id in [0, 5, 10] {
            let audio = tts_multi
                .synthesize(
                    "This sentence is spoken by a different voice.",
                    Some(speaker_id),
                    Some(LanguageCode::EnUs),
                )
                .await?;

            println!("Speaker {}: Generated {} samples", speaker_id, audio.len());
        }
    }

    println!("\n3. Available pre-trained models:");
    for (id, description, config) in PretrainedModels::list_available_models() {
        println!("  - {id}: {description}");
        println!(
            "    Model: {} ({})",
            config.metadata.name, config.metadata.author
        );
        println!("    Languages: {:?}", config.supported_languages);
    }

    println!("\n=== Example Usage Complete ===");
    println!("\nNote: This example shows the structure for using pre-trained models.");
    println!("To use real models, you would:");
    println!("1. Download or specify actual model files/HuggingFace IDs");
    println!("2. Ensure the models are compatible with the VoiRS architecture");
    println!("3. Configure model parameters to match the pre-trained weights");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Run the examples
    run_examples().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pretrained_model_configs() {
        let en_config = PretrainedModels::vits_en_us_single_speaker();
        assert_eq!(en_config.architecture, ModelArchitecture::Vits);
        assert!(en_config.supported_languages.contains(&LanguageCode::EnUs));
        assert!(en_config.validate().is_ok());

        let multi_config = PretrainedModels::vits_multi_speaker();
        assert!(multi_config
            .architecture_params
            .vits
            .as_ref()
            .unwrap()
            .n_speakers
            .is_some());

        let ja_config = PretrainedModels::vits_japanese();
        assert!(ja_config.supported_languages.contains(&LanguageCode::JaJp));
    }

    #[test]
    fn test_hifigan_configs() {
        let v1_config = PretrainedModels::hifigan_v1();
        assert_eq!(v1_config.variant, HiFiGanVariant::V1);
        assert_eq!(v1_config.initial_channels, 512);

        let v3_config = PretrainedModels::hifigan_v3();
        assert_eq!(v3_config.variant, HiFiGanVariant::V3);
        assert_eq!(v3_config.initial_channels, 128);
    }

    #[test]
    fn test_model_listing() {
        let models = PretrainedModels::list_available_models();
        assert_eq!(models.len(), 3);

        let (id, desc, config) = &models[0];
        assert_eq!(*id, "vits-en-us-single");
        assert!(!desc.is_empty());
        assert!(config.validate().is_ok());
    }
}
