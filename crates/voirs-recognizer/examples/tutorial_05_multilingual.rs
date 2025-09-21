//! Tutorial 05: Multi-language Support
//!
//! This tutorial covers VoiRS Recognizer's multi-language capabilities,
//! including automatic language detection, language-specific models,
//! and cross-linguistic phoneme analysis.
//!
//! Learning Objectives:
//! - Understand supported languages and their characteristics
//! - Configure language-specific recognition
//! - Implement automatic language detection
//! - Handle multilingual audio streams
//! - Understand phoneme mapping across languages
//! - Optimize for specific language families
//!
//! Prerequisites: Complete Tutorials 01-04
//!
//! Usage:
//! ```bash
//! cargo run --example tutorial_05_multilingual --features="whisper-pure"
//! ```

use std::collections::HashMap;
use std::error::Error;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use voirs_recognizer::asr::{ASRBackend, WhisperModelSize};
use voirs_recognizer::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🎓 Tutorial 05: Multi-language Support");
    println!("======================================\n");

    // Step 1: Introduction to multilingual processing
    println!("🌍 Learning Goal: Master multilingual speech recognition");
    println!("   • Understand supported languages");
    println!("   • Configure language-specific recognition");
    println!("   • Implement automatic language detection");
    println!("   • Handle multilingual audio streams");
    println!("   • Understand phoneme mapping across languages\n");

    // Step 2: Language support overview
    println!("📊 Step 1: Language Support Overview");
    demonstrate_language_support();

    // Step 3: Language-specific configuration
    println!(
        "
🔧 Step 2: Language-Specific Configuration"
    );
    demonstrate_language_configs().await?;

    // Step 4: Automatic language detection
    println!(
        "
🔍 Step 3: Automatic Language Detection"
    );
    demonstrate_language_detection().await?;

    // Step 5: Multilingual audio processing
    println!(
        "
🎤 Step 4: Multilingual Audio Processing"
    );
    demonstrate_multilingual_processing().await?;

    // Step 6: Phoneme analysis across languages
    println!(
        "
🔤 Step 5: Cross-linguistic Phoneme Analysis"
    );
    demonstrate_phoneme_analysis().await?;

    // Step 7: Performance optimization by language
    println!(
        "
⚡ Step 6: Language-Specific Optimization"
    );
    demonstrate_language_optimization().await?;

    // Step 8: Conclusion
    println!(
        "
🎉 Congratulations! You've completed Tutorial 05!"
    );
    println!(
        "
📖 What you learned:"
    );
    println!("   • VoiRS supports 99+ languages with varying degrees of support");
    println!("   • Language-specific configuration improves accuracy");
    println!("   • Automatic language detection enables flexible applications");
    println!("   • Multilingual processing requires careful buffer management");
    println!("   • Phoneme analysis reveals linguistic relationships");
    println!("   • Language-specific optimization improves performance");

    println!(
        "
🚀 Next Steps:"
    );
    println!("   • Tutorial 06: Performance optimization");
    println!("   • Tutorial 07: Integration examples");
    println!("   • Advanced: Custom model fine-tuning");

    Ok(())
}

fn demonstrate_language_support() {
    println!("   VoiRS Recognizer supports multiple languages through different models:");

    let language_support = vec![
        (
            "🌟 Tier 1 (Excellent Support)",
            vec![
                ("English", "en-US", "Whisper + DeepSpeech + Wav2Vec2 + MFA"),
                ("Spanish", "es-ES", "Whisper + MFA"),
                ("French", "fr-FR", "Whisper + MFA"),
                ("German", "de-DE", "Whisper + MFA"),
            ],
        ),
        (
            "🔥 Tier 2 (Great Support)",
            vec![
                ("Japanese", "ja-JP", "Whisper"),
                ("Chinese", "zh-CN", "Whisper"),
                ("Korean", "ko-KR", "Whisper"),
                ("Portuguese", "pt-BR", "Whisper"),
                ("Italian", "it-IT", "Whisper"),
                ("Russian", "ru-RU", "Whisper"),
            ],
        ),
        (
            "⚡ Tier 3 (Good Support)",
            vec![
                ("Hindi", "hi-IN", "Whisper"),
                ("Arabic", "ar-SA", "Whisper"),
                ("Dutch", "nl-NL", "Whisper"),
                ("Polish", "pl-PL", "Whisper"),
                ("Turkish", "tr-TR", "Whisper"),
            ],
        ),
    ];

    for (tier, languages) in language_support {
        println!(
            "   
   {}:",
            tier
        );
        for (lang, code, support) in languages {
            println!("   • {} ({}): {}", lang, code, support);
        }
    }

    println!(
        "   
   📋 Model Capabilities:"
    );
    println!("   • Whisper: 99+ languages, multilingual, robust");
    println!("   • DeepSpeech: English only, fast, privacy-focused");
    println!("   • Wav2Vec2: English + some multilingual models");
    println!("   • MFA: Phoneme alignment for supported languages");
}

async fn demonstrate_language_configs() -> Result<(), Box<dyn Error>> {
    println!("   Different languages may require different configurations:");

    let language_configs = vec![
        (
            "English (US)",
            LanguageCode::EnUs,
            "General-purpose, works with all models",
            ASRConfig {
                preferred_models: vec!["whisper".to_string()],
                whisper_model_size: Some("base".to_string()),
                language: Some(LanguageCode::EnUs),
                enable_voice_activity_detection: true,
                chunk_duration_ms: 30000,
                ..Default::default()
            },
        ),
        (
            "Japanese",
            LanguageCode::JaJp,
            "Requires larger model for better accuracy",
            ASRConfig {
                preferred_models: vec!["whisper".to_string()],
                whisper_model_size: Some("small".to_string()), // Larger model for Japanese
                language: Some(LanguageCode::JaJp),
                enable_voice_activity_detection: true,
                chunk_duration_ms: 20000, // Shorter chunks for Japanese
                ..Default::default()
            },
        ),
        (
            "German",
            LanguageCode::DeDe,
            "Benefits from longer context for compound words",
            ASRConfig {
                preferred_models: vec!["whisper".to_string()],
                whisper_model_size: Some("base".to_string()),
                language: Some(LanguageCode::DeDe),
                enable_voice_activity_detection: true,
                chunk_duration_ms: 35000, // Longer chunks for German compounds
                ..Default::default()
            },
        ),
    ];

    for (lang_name, lang_code, description, config) in language_configs {
        println!(
            "   
   🌐 {} Configuration:",
            lang_name
        );
        println!("   • Description: {}", description);
        println!("   • Language code: {:?}", lang_code);
        println!("   • Model size: {:?}", config.whisper_model_size);
        println!("   • Chunk duration: {}ms", config.chunk_duration_ms);
        println!(
            "   • Voice activity detection: {}",
            config.enable_voice_activity_detection
        );
        println!("   • Word timestamps: {}", config.word_timestamps);

        // Simulate recognition with this config
        let start_time = Instant::now();
        let result = simulate_language_recognition(lang_name, &config).await;
        let elapsed = start_time.elapsed();

        println!("   • Sample result: \"{}\"", result);
        println!("   • Processing time: {}ms", elapsed.as_millis());
    }

    Ok(())
}

async fn simulate_language_recognition(language: &str, _config: &ASRConfig) -> String {
    // Simulate processing time
    sleep(Duration::from_millis(150)).await;

    // Return language-appropriate sample text
    match language {
        "English (US)" => "Hello, how are you today?",
        "Japanese" => "こんにちは、今日はどうですか？",
        "German" => "Hallo, wie geht es Ihnen heute?",
        _ => "Hello world",
    }
    .to_string()
}

async fn demonstrate_language_detection() -> Result<(), Box<dyn Error>> {
    println!("   Automatic language detection enables flexible applications:");

    println!(
        "   
   🔍 Language Detection Process:"
    );
    println!("   1. Analyze audio spectral characteristics");
    println!("   2. Run inference on multiple language models");
    println!("   3. Compare confidence scores across languages");
    println!("   4. Select best language and continue processing");

    // Simulate language detection on different audio samples
    let audio_samples = vec![
        (
            "English audio",
            vec![
                (LanguageCode::EnUs, 0.95),
                (LanguageCode::DeDe, 0.23),
                (LanguageCode::FrFr, 0.15),
                (LanguageCode::EsEs, 0.18),
            ],
        ),
        (
            "Spanish audio",
            vec![
                (LanguageCode::EsEs, 0.87),
                (LanguageCode::EnUs, 0.32),
                (LanguageCode::FrFr, 0.45),
                (LanguageCode::DeDe, 0.12),
            ],
        ),
        (
            "Mixed language audio",
            vec![
                (LanguageCode::EnUs, 0.65),
                (LanguageCode::FrFr, 0.62),
                (LanguageCode::DeDe, 0.58),
                (LanguageCode::EsEs, 0.45),
            ],
        ),
    ];

    for (sample_name, detections) in audio_samples {
        println!(
            "   
   🎵 Processing {}:",
            sample_name
        );

        // Simulate detection processing
        sleep(Duration::from_millis(200)).await;

        // Find best language
        let best_detection = detections
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let confidence_threshold = 0.7;

        println!("   Detection results:");
        for (lang, confidence) in &detections {
            let status = if confidence > &confidence_threshold {
                "✅"
            } else {
                "❌"
            };
            println!("   • {:?}: {:.1}% {}", lang, confidence * 100.0, status);
        }

        if let Some((best_lang, best_confidence)) = best_detection {
            if *best_confidence > confidence_threshold {
                println!(
                    "   🎯 Selected language: {:?} (confidence: {:.1}%)",
                    best_lang,
                    best_confidence * 100.0
                );
            } else {
                println!("   ⚠️ Low confidence detection - using fallback to English");
            }
        }
    }

    println!(
        "   
   📊 Detection Strategies:"
    );
    println!("   • Confidence threshold: Set minimum confidence for auto-detection");
    println!("   • Fallback language: Use when detection confidence is low");
    println!("   • Language hints: Bias detection toward expected languages");
    println!("   • Continuous detection: Re-evaluate language during long audio");

    Ok(())
}

async fn demonstrate_multilingual_processing() -> Result<(), Box<dyn Error>> {
    println!("   Processing multilingual audio streams requires special handling:");

    println!(
        "   
   🌍 Multilingual Processing Challenges:"
    );
    println!("   • Language switching within audio");
    println!("   • Code-switching (mixing languages in sentences)");
    println!("   • Accent variations");
    println!("   • Cultural context differences");

    // Simulate multilingual audio stream
    let audio_stream = vec![
        (
            "English segment",
            LanguageCode::EnUs,
            "Hello, my name is John",
        ),
        ("Spanish segment", LanguageCode::EsEs, "Hola, me llamo Juan"),
        (
            "French segment",
            LanguageCode::FrFr,
            "Bonjour, je m'appelle Jean",
        ),
        (
            "Code-switching",
            LanguageCode::EnUs,
            "I speak English and español",
        ),
    ];

    println!(
        "   
   🎤 Processing Multilingual Stream:"
    );

    let mut current_language = LanguageCode::EnUs;
    let mut language_switches = 0;

    for (segment_name, detected_lang, transcript) in &audio_stream {
        println!(
            "   
   Segment: {}",
            segment_name
        );

        // Simulate language detection
        sleep(Duration::from_millis(100)).await;

        if *detected_lang != current_language {
            println!(
                "   🔄 Language switch detected: {:?} → {:?}",
                current_language, detected_lang
            );
            current_language = *detected_lang;
            language_switches += 1;

            // Simulate model reconfiguration
            sleep(Duration::from_millis(200)).await;
            println!("   🔧 Reconfigured ASR for {:?}", detected_lang);
        }

        // Simulate recognition
        sleep(Duration::from_millis(300)).await;
        println!("   📝 Transcript: \"{}\"", transcript);
        println!("   🎯 Language: {:?}", detected_lang);
    }

    println!(
        "   
   📊 Stream Processing Summary:"
    );
    println!("   • Total segments: {}", audio_stream.len());
    println!("   • Language switches: {}", language_switches);
    println!("   • Final language: {:?}", current_language);

    println!(
        "   
   🛠️ Multilingual Best Practices:"
    );
    println!("   • Use continuous language detection");
    println!("   • Implement smooth model switching");
    println!("   • Buffer context across language boundaries");
    println!("   • Handle code-switching gracefully");
    println!("   • Maintain language-specific vocabularies");

    Ok(())
}

async fn demonstrate_phoneme_analysis() -> Result<(), Box<dyn Error>> {
    println!("   Phoneme analysis reveals linguistic relationships across languages:");

    println!(
        "   
   🔤 Phoneme Inventory Comparison:"
    );

    // Simulate phoneme inventories for different languages
    let phoneme_inventories = vec![
        (
            "English",
            vec![
                "p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "m",
                "n", "ŋ", "l", "r", "j", "w",
            ],
        ),
        (
            "Spanish",
            vec![
                "p", "b", "t", "d", "k", "g", "f", "β", "s", "x", "m", "n", "ɲ", "ŋ", "l", "ʎ",
                "r", "rr", "j", "w",
            ],
        ),
        (
            "German",
            vec![
                "p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "x", "ç", "h", "m",
                "n", "ŋ", "l", "r", "j", "w",
            ],
        ),
        (
            "Japanese",
            vec![
                "p", "b", "t", "d", "k", "g", "f", "s", "z", "ʃ", "ʒ", "h", "m", "n", "ŋ", "r",
                "j", "w",
            ],
        ),
    ];

    for (language, phonemes) in &phoneme_inventories {
        println!(
            "   
   🗣️ {} Phonemes ({} total):",
            language,
            phonemes.len()
        );
        let phoneme_str = phonemes.join(", ");
        println!("   {}", phoneme_str);
    }

    println!(
        "   
   🔍 Cross-linguistic Phoneme Analysis:"
    );

    // Analyze phoneme similarities
    let sample_word = "hello";
    let phoneme_mappings = vec![
        ("English", "/hɛloʊ/"),
        ("Spanish", "/ˈe.lo/"),
        ("German", "/haˈloː/"),
        ("Japanese", "/he.ro/"),
    ];

    println!(
        "   
   Word: \"{}\"",
        sample_word
    );
    for (language, phonemes) in phoneme_mappings {
        println!("   • {}: {}", language, phonemes);
    }

    println!(
        "   
   🔄 Phoneme Mapping Applications:"
    );
    println!("   • Cross-language pronunciation models");
    println!("   • Accent adaptation systems");
    println!("   • Language learning applications");
    println!("   • Speech synthesis voice conversion");

    Ok(())
}

async fn demonstrate_language_optimization() -> Result<(), Box<dyn Error>> {
    println!("   Language-specific optimization improves performance:");

    println!(
        "   
   ⚡ Optimization Strategies by Language Family:"
    );

    let optimization_strategies = vec![
        (
            "Germanic Languages (English, German, Dutch)",
            vec![
                "Longer context windows for compound words",
                "Stress-based syllable detection",
                "Consonant cluster handling",
                "Modal particle recognition",
            ],
        ),
        (
            "Romance Languages (Spanish, French, Italian)",
            vec![
                "Vowel-centric processing",
                "Syllable-timed rhythm detection",
                "Liaison and elision handling",
                "Gender agreement patterns",
            ],
        ),
        (
            "East Asian Languages (Japanese, Chinese, Korean)",
            vec![
                "Tone recognition and processing",
                "Character-based text processing",
                "Pitch accent detection",
                "Morphological analysis",
            ],
        ),
    ];

    for (family, strategies) in optimization_strategies {
        println!(
            "   
   🌐 {}:",
            family
        );
        for strategy in strategies {
            println!("   • {}", strategy);
        }
    }

    println!(
        "   
   📊 Performance Comparison:"
    );

    // Simulate performance metrics for different languages
    let performance_metrics = vec![
        ("English", 0.25, 1.2, 95.2),
        ("Spanish", 0.28, 1.1, 92.8),
        ("German", 0.32, 1.4, 91.5),
        ("Japanese", 0.45, 1.8, 88.3),
        ("Chinese", 0.42, 1.6, 89.1),
    ];

    println!(
        "   
   Language        RTF    Memory(GB)   Accuracy(%)"
    );
    println!("   -----------------------------------------------");
    for (language, rtf, memory, accuracy) in performance_metrics {
        println!(
            "   {:12}   {:.2}      {:.1}        {:.1}",
            language, rtf, memory, accuracy
        );
    }

    println!(
        "   
   💡 Optimization Tips:"
    );
    println!("   • Use language-specific models when available");
    println!("   • Adjust chunk sizes based on language characteristics");
    println!("   • Implement language-aware voice activity detection");
    println!("   • Cache frequently used language models");
    println!("   • Use language-specific text normalization");

    Ok(())
}
