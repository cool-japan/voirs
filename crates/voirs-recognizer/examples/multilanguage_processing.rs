//! Multi-language Processing Example
//!
//! This example demonstrates VoiRS Recognizer's multi-language capabilities including
//! automatic language detection, cross-linguistic phoneme mapping, and language-specific
//! audio analysis.
//!
//! Usage:
//! ```bash
//! cargo run --example multilanguage_processing --features="whisper-pure"
//! ```

use std::collections::HashMap;
use voirs_recognizer::phoneme::phoneme_sets::{
    calculate_phoneme_similarity, get_phoneme_inventory, PhonemeNotation, PhonemeSet,
};
use voirs_recognizer::prelude::*;
use voirs_recognizer::RecognitionError;

#[tokio::main]
async fn main() -> Result<(), RecognitionError> {
    println!("🌍 VoiRS Multi-language Processing Example");
    println!("==========================================\n");

    // Step 1: Demonstrate supported languages
    println!("🗣️ Supported Languages:");
    let supported_languages = vec![
        (LanguageCode::EnUs, "English (US)", "Hello, how are you?"),
        (LanguageCode::DeDe, "German", "Hallo, wie geht es dir?"),
        (LanguageCode::EsEs, "Spanish", "Hola, ¿cómo estás?"),
        (LanguageCode::FrFr, "French", "Bonjour, comment allez-vous?"),
        (LanguageCode::JaJp, "Japanese", "こんにちは、元気ですか？"),
        (LanguageCode::KoKr, "Korean", "안녕하세요, 어떻게 지내세요?"),
        (LanguageCode::ZhCn, "Chinese (Simplified)", "你好，你好吗？"),
    ];

    for (lang_code, lang_name, sample_text) in &supported_languages {
        println!(
            "   • {} ({}): \"{}\"",
            lang_name,
            format!("{:?}", lang_code),
            sample_text
        );
    }

    // Step 2: Create language-specific audio samples
    println!("\n🎵 Creating language-specific audio samples...");
    let mut audio_samples = HashMap::new();

    for (lang_code, lang_name, _) in &supported_languages {
        // Create synthetic audio with language-specific characteristics
        let audio = create_language_specific_audio(*lang_code);
        audio_samples.insert(*lang_code, audio);
        println!("   ✅ Generated audio sample for {}", lang_name);
    }

    // Step 3: Demonstrate language-specific phoneme systems
    println!("\n🔤 Language-specific Phoneme Systems:");

    for (lang_code, lang_name, _) in &supported_languages {
        let inventory = get_phoneme_inventory(*lang_code);

        println!("   • {} phoneme inventory:", lang_name);
        println!("     - Vowels: {}", inventory.vowels.len());
        println!("     - Consonants: {}", inventory.consonants.len());
        println!(
            "     - Total phonemes: {}",
            inventory.vowels.len() + inventory.consonants.len()
        );

        // Show a few example phonemes
        let sample_phonemes: Vec<_> = inventory
            .vowels
            .iter()
            .chain(inventory.consonants.iter())
            .take(5)
            .collect();
        println!("     - Examples: {:?}", sample_phonemes);
    }

    // Step 4: Demonstrate cross-linguistic phoneme mapping
    println!("\n🔄 Cross-linguistic Phoneme Mapping:");

    let source_lang = LanguageCode::EnUs;
    let target_languages = vec![LanguageCode::DeDe, LanguageCode::EsEs, LanguageCode::FrFr];

    println!("   Source language: English (US)");
    for target_lang in &target_languages {
        println!("   • Mapping to {:?}:", target_lang);

        // Demonstrate phoneme similarity mapping
        let source_inventory = get_phoneme_inventory(source_lang);
        let target_inventory = get_phoneme_inventory(*target_lang);

        // Example English phonemes to map
        let english_phonemes = vec!["æ", "θ", "ð", "ɹ"];

        for en_phoneme in &english_phonemes {
            // Use the similarity calculation function from the phoneme_sets module
            let similarity = calculate_phoneme_similarity(en_phoneme, "a", *target_lang);
            println!(
                "     - English '{}' → closest similarity: {:.3}",
                en_phoneme, similarity
            );
        }
    }

    // Step 5: Language-specific audio analysis
    println!("\n🔍 Language-specific Audio Analysis:");

    for (lang_code, lang_name, _) in supported_languages.iter().take(3) {
        // Analyze first 3 for brevity
        println!("\n   📊 Analyzing {} audio:", lang_name);

        if let Some(audio) = audio_samples.get(lang_code) {
            // Create language-specific analysis configuration
            let analysis_config = AudioAnalysisConfig {
                quality_metrics: true,
                prosody_analysis: true,
                speaker_analysis: true,
                ..Default::default()
            };

            let analyzer = AudioAnalyzerImpl::new(analysis_config.clone()).await?;
            let analysis = analyzer.analyze(audio, Some(&analysis_config)).await?;

            // Display language-specific metrics
            let prosody = &analysis.prosody;
            {
                println!(
                    "     • F0 (fundamental frequency): {:.2} Hz",
                    prosody.pitch.mean_f0
                );
                println!(
                    "     • Speaking rate: {:.2} syllables/sec",
                    prosody.rhythm.speaking_rate
                );
                println!(
                    "     • Intonation pattern: {:?}",
                    prosody.intonation.pattern_type
                );

                // Language-specific prosody insights
                match lang_code {
                    LanguageCode::EnUs => {
                        println!("     • English prosody: Stress-timed rhythm");
                        if prosody.pitch.mean_f0 > 200.0 {
                            println!("     • Higher pitch typical for English questions");
                        }
                    }
                    LanguageCode::JaJp => {
                        println!("     • Japanese prosody: Mora-timed rhythm");
                        println!("     • Pitch accent language characteristics");
                    }
                    LanguageCode::ZhCn => {
                        println!("     • Chinese prosody: Tonal language characteristics");
                        println!("     • Lexical tone affects F0 patterns");
                    }
                    _ => {}
                }
            }

            // Quality analysis with language context
            if let Some(snr) = analysis.quality_metrics.get("snr") {
                println!("     • Signal-to-noise ratio: {:.2} dB", snr);
            }
            if let Some(zcr) = analysis.quality_metrics.get("zcr") {
                println!("     • Zero crossing rate: {:.3}", zcr);

                // Language-specific ZCR interpretation
                match lang_code {
                    LanguageCode::DeDe => {
                        if *zcr > 0.1 {
                            println!("     • High ZCR consistent with German fricatives");
                        }
                    }
                    LanguageCode::EsEs => {
                        println!("     • ZCR pattern typical for Spanish phonology");
                    }
                    _ => {}
                }
            }
        }
    }

    // Step 6: Demonstrate multi-notation phoneme support
    println!("\n📝 Multi-notation Phoneme Support:");

    let phoneme_inventory = get_phoneme_inventory(LanguageCode::EnUs);
    let sample_phonemes = vec!["æ", "θ", "ɹ", "ŋ"];

    println!("   English phonemes in different notations:");
    for phoneme in &sample_phonemes {
        println!("     • IPA: {}", phoneme);

        if let Some(arpabet_set) = phoneme_inventory
            .notation_sets
            .iter()
            .find(|set| set.notation == PhonemeNotation::ARPABET)
        {
            if let Some(arpabet) = arpabet_set.from_ipa_map.get(*phoneme) {
                println!("       ARPABET: {}", arpabet);
            }
        }

        if let Some(sampa_set) = phoneme_inventory
            .notation_sets
            .iter()
            .find(|set| set.notation == PhonemeNotation::SAMPA)
        {
            if let Some(sampa) = sampa_set.from_ipa_map.get(*phoneme) {
                println!("       SAMPA: {}", sampa);
            }
        }
    }

    // Step 7: Language detection simulation
    println!("\n🔍 Language Detection Capabilities:");

    println!("   Simulating automatic language detection:");
    for (lang_code, lang_name, _) in supported_languages.iter().take(4) {
        if let Some(audio) = audio_samples.get(lang_code) {
            // Create language-agnostic config for detection
            let detection_config = voirs_recognizer::default_asr_config(*lang_code);
            println!(
                "     • Audio sample → Detected: {} (confidence: simulated)",
                lang_name
            );
            println!("       - Language code: {:?}", detection_config.language);
            println!(
                "       - Word timestamps: {}",
                detection_config.word_timestamps
            );
        }
    }

    // Step 8: Best practices for multi-language processing
    println!("\n💡 Multi-language Processing Best Practices:");
    println!("   • Use automatic language detection for unknown inputs");
    println!("   • Configure language-specific phoneme sets");
    println!("   • Adapt prosody analysis for language characteristics");
    println!("   • Consider cross-linguistic phoneme mapping for accents");
    println!("   • Use appropriate notation systems (IPA, ARPABET, SAMPA)");
    println!("   • Account for language-specific audio characteristics");

    println!("\n✅ Multi-language processing example completed!");
    println!("🌍 Demonstrated capabilities:");
    println!("   • {} language support", supported_languages.len());
    println!("   • Cross-linguistic phoneme mapping");
    println!("   • Language-specific audio analysis");
    println!("   • Multi-notation phoneme support");
    println!("   • Language detection simulation");

    Ok(())
}

/// Create synthetic audio with language-specific acoustic characteristics
fn create_language_specific_audio(language: LanguageCode) -> AudioBuffer {
    let sample_rate = 16000;
    let duration = 1.0; // 1 second
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    // Language-specific acoustic parameters
    let (base_f0, f0_variation, formant_pattern, rhythm_pattern) = match language {
        LanguageCode::EnUs => (150.0, 50.0, vec![730.0, 1090.0, 2440.0], 1.2), // English vowel formants
        LanguageCode::DeDe => (140.0, 40.0, vec![700.0, 1100.0, 2300.0], 1.1), // German characteristics
        LanguageCode::EsEs => (160.0, 45.0, vec![750.0, 1200.0, 2400.0], 1.0), // Spanish characteristics
        LanguageCode::FrFr => (155.0, 35.0, vec![720.0, 1150.0, 2350.0], 1.15), // French characteristics
        LanguageCode::JaJp => (170.0, 60.0, vec![800.0, 1300.0, 2500.0], 0.9), // Japanese pitch accent
        LanguageCode::KoKr => (165.0, 55.0, vec![780.0, 1250.0, 2450.0], 0.95), // Korean characteristics
        LanguageCode::ZhCn => (180.0, 80.0, vec![760.0, 1180.0, 2480.0], 1.3),  // Chinese tonal
        _ => (150.0, 50.0, vec![730.0, 1090.0, 2440.0], 1.0), // Default to English
    };

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;

        // Create fundamental frequency with language-specific modulation
        let f0_modulation = match language {
            LanguageCode::ZhCn => (8.0 * t).sin() * 0.3, // Tonal modulation
            LanguageCode::JaJp => (6.0 * t).sin() * 0.2, // Pitch accent
            _ => (4.0 * t).sin() * 0.1,                  // General intonation
        };

        let f0 = base_f0 + f0_variation * f0_modulation;

        // Create formant-based signal
        let mut signal = 0.0;
        for (j, &formant_freq) in formant_pattern.iter().enumerate() {
            let amplitude = 1.0 / (j + 1) as f32; // Decreasing amplitude
            signal += amplitude * (2.0 * std::f32::consts::PI * formant_freq * t).sin();
        }

        // Add fundamental frequency
        signal += 0.5 * (2.0 * std::f32::consts::PI * f0 * t).sin();

        // Apply rhythm pattern
        let rhythm_envelope = (rhythm_pattern * 10.0 * t).sin().abs();
        signal *= rhythm_envelope;

        // Add language-specific noise characteristics
        let noise_level = match language {
            LanguageCode::DeDe => 0.02, // Slightly more noise for fricatives
            LanguageCode::EsEs => 0.01, // Clear pronunciation
            _ => 0.015,
        };

        let noise = (i as f32 * 0.001).sin() * noise_level;
        signal += noise;

        // Normalize and add to samples
        samples.push(signal * 0.1);
    }

    AudioBuffer::mono(samples, sample_rate)
}
