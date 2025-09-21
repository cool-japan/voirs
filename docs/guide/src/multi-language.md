# Multi-language Support

VoiRS Recognizer provides comprehensive multi-language support with automatic language detection, language-specific optimizations, and cross-lingual capabilities.

## Supported Languages

VoiRS supports 99+ languages through the Whisper backend. Here are the most commonly used languages:

### Major Languages

| Language | Code | Whisper Support | Notes |
|----------|------|-----------------|-------|
| English (US) | `EnUs` | ✅ | Best supported |
| English (UK) | `EnGb` | ✅ | British English |
| Spanish | `EsEs` | ✅ | Spain Spanish |
| French | `FrFr` | ✅ | French |
| German | `DeDE` | ✅ | German |
| Italian | `ItIt` | ✅ | Italian |
| Portuguese | `PtBr` | ✅ | Brazilian Portuguese |
| Russian | `RuRu` | ✅ | Russian |
| Japanese | `JaJp` | ✅ | Japanese |
| Korean | `KoKr` | ✅ | Korean |
| Chinese | `ZhCn` | ✅ | Mandarin Chinese |
| Arabic | `ArAr` | ✅ | Modern Standard Arabic |
| Hindi | `HiIn` | ✅ | Hindi |

### Regional Variants

```rust
use voirs_recognizer::prelude::*;

// English variants
let us_english = LanguageCode::EnUs;
let uk_english = LanguageCode::EnGb;
let au_english = LanguageCode::EnAu;

// Spanish variants
let spain_spanish = LanguageCode::EsEs;
let mexico_spanish = LanguageCode::EsMx;
let argentina_spanish = LanguageCode::EsAr;

// Portuguese variants
let brazil_portuguese = LanguageCode::PtBr;
let portugal_portuguese = LanguageCode::PtPt;
```

## Automatic Language Detection

### Basic Language Detection

```rust
use voirs_recognizer::prelude::*;

#[tokio::main]
async fn main() -> Result<(), RecognitionError> {
    let audio = load_audio("multilingual_audio.wav", &AudioLoadConfig::default()).await?;
    
    // Enable automatic language detection
    let config = ASRConfig {
        language: None,  // Auto-detect
        detect_language: true,
        language_detection_threshold: 0.8,
        ..Default::default()
    };
    
    let mut asr = ASRBackend::new_whisper(config).await?;
    let result = asr.recognize(&audio, None).await?;
    
    println!("Detected language: {:?}", result.language);
    println!("Language confidence: {:.2}", result.language_confidence);
    println!("Transcript: {}", result.text);
    
    Ok(())
}
```

### Language Detection with Candidates

```rust
// Limit detection to specific languages
let config = ASRConfig {
    language: None,
    detect_language: true,
    language_candidates: Some(vec![
        LanguageCode::EnUs,
        LanguageCode::EsEs,
        LanguageCode::FrFr,
    ]),
    ..Default::default()
};

let mut asr = ASRBackend::new_whisper(config).await?;
let result = asr.recognize(&audio, None).await?;

// Get language probabilities
if let Some(probs) = result.language_probabilities {
    for (lang, prob) in probs {
        println!("{:?}: {:.3}", lang, prob);
    }
}
```

## Language-Specific Configuration

### Optimized Configurations

```rust
use voirs_recognizer::prelude::*;

// English - optimized for natural speech
let english_config = ASRConfig {
    language: Some(LanguageCode::EnUs),
    punctuation: true,
    sentence_segmentation: true,
    word_timestamps: true,
    ..Default::default()
};

// Japanese - optimized for character-based writing
let japanese_config = ASRConfig {
    language: Some(LanguageCode::JaJp),
    punctuation: false,  // Less punctuation in Japanese
    sentence_segmentation: false,
    character_level: true,
    ..Default::default()
};

// Arabic - optimized for RTL text
let arabic_config = ASRConfig {
    language: Some(LanguageCode::ArAr),
    text_direction: TextDirection::RightToLeft,
    punctuation: true,
    normalize_text: true,
    ..Default::default()
};
```

### Language-Specific Preprocessing

```rust
// Chinese - handle tone and character conversion
let chinese_config = ASRConfig {
    language: Some(LanguageCode::ZhCn),
    text_normalization: TextNormalization::Traditional,
    tone_marks: true,
    character_conversion: CharacterConversion::Simplified,
    ..Default::default()
};

// German - handle compound words
let german_config = ASRConfig {
    language: Some(LanguageCode::DeDE),
    compound_word_splitting: true,
    case_normalization: true,
    ..Default::default()
};
```

## Multi-language Processing

### Sequential Processing

```rust
use voirs_recognizer::prelude::*;

async fn process_multilingual_content(
    audio_files: Vec<&str>,
    languages: Vec<LanguageCode>
) -> Result<Vec<String>, RecognitionError> {
    let mut results = Vec::new();
    
    for (file, lang) in audio_files.iter().zip(languages.iter()) {
        let audio = load_audio(file, &AudioLoadConfig::default()).await?;
        
        let config = ASRConfig {
            language: Some(*lang),
            ..Default::default()
        };
        
        let mut asr = ASRBackend::new_whisper(config).await?;
        let result = asr.recognize(&audio, None).await?;
        
        results.push(result.text);
    }
    
    Ok(results)
}
```

### Parallel Processing

```rust
use tokio::task;
use futures::future::join_all;

async fn parallel_multilingual_processing(
    audio_files: Vec<&str>,
    languages: Vec<LanguageCode>
) -> Result<Vec<String>, RecognitionError> {
    let tasks = audio_files.into_iter()
        .zip(languages.into_iter())
        .map(|(file, lang)| {
            task::spawn(async move {
                let audio = load_audio(file, &AudioLoadConfig::default()).await?;
                
                let config = ASRConfig {
                    language: Some(lang),
                    ..Default::default()
                };
                
                let mut asr = ASRBackend::new_whisper(config).await?;
                asr.recognize(&audio, None).await.map(|r| r.text)
            })
        })
        .collect::<Vec<_>>();
    
    let results = join_all(tasks).await;
    
    results.into_iter()
        .map(|r| r.unwrap())
        .collect::<Result<Vec<_>, _>>()
}
```

## Code-switching Support

### Mixed Language Audio

```rust
// Handle audio with multiple languages
let mixed_config = ASRConfig {
    language: None,  // Auto-detect
    detect_language: true,
    code_switching: true,
    segment_languages: true,
    ..Default::default()
};

let mut asr = ASRBackend::new_whisper(mixed_config).await?;
let result = asr.recognize(&audio, None).await?;

// Get language segments
if let Some(segments) = result.language_segments {
    for segment in segments {
        println!("{}s-{}s: {:?} - \"{}\"", 
                 segment.start_time, 
                 segment.end_time, 
                 segment.language, 
                 segment.text);
    }
}
```

## Language-Specific Features

### Phoneme Systems

```rust
use voirs_recognizer::phoneme::*;

// English phonemes (ARPABET)
let english_phonemes = PhonemeRecognitionConfig {
    language: LanguageCode::EnUs,
    phoneme_set: PhonemeSet::ARPABET,
    ..Default::default()
};

// Japanese phonemes (Hiragana-based)
let japanese_phonemes = PhonemeRecognitionConfig {
    language: LanguageCode::JaJp,
    phoneme_set: PhonemeSet::Japanese,
    ..Default::default()
};

// International phonemes (IPA)
let ipa_phonemes = PhonemeRecognitionConfig {
    language: LanguageCode::EnUs,
    phoneme_set: PhonemeSet::IPA,
    ..Default::default()
};
```

### Cross-lingual Phoneme Mapping

```rust
// Map phonemes between languages
let cross_lingual_config = PhonemeRecognitionConfig {
    language: LanguageCode::EnUs,
    target_language: Some(LanguageCode::EsEs),
    cross_lingual_mapping: true,
    ..Default::default()
};
```

## Regional Accent Support

### Accent Detection

```rust
let config = ASRConfig {
    language: Some(LanguageCode::EnUs),
    accent_detection: true,
    accent_adaptation: true,
    ..Default::default()
};

let mut asr = ASRBackend::new_whisper(config).await?;
let result = asr.recognize(&audio, None).await?;

if let Some(accent) = result.detected_accent {
    println!("Detected accent: {:?}", accent);
}
```

### Accent-Specific Models

```rust
// Use accent-specific models
let southern_us_config = ASRConfig {
    language: Some(LanguageCode::EnUs),
    accent: Some(AccentCode::EnUsSouthern),
    ..Default::default()
};

let british_config = ASRConfig {
    language: Some(LanguageCode::EnGb),
    accent: Some(AccentCode::EnGbRP),  // Received Pronunciation
    ..Default::default()
};
```

## Text Normalization

### Language-Specific Normalization

```rust
// English text normalization
let english_norm = TextNormalization {
    language: LanguageCode::EnUs,
    expand_contractions: true,
    normalize_numbers: true,
    normalize_dates: true,
    normalize_times: true,
    ..Default::default()
};

// Japanese text normalization
let japanese_norm = TextNormalization {
    language: LanguageCode::JaJp,
    hiragana_katakana_conversion: true,
    kanji_conversion: false,
    normalize_punctuation: true,
    ..Default::default()
};

// Arabic text normalization
let arabic_norm = TextNormalization {
    language: LanguageCode::ArAr,
    normalize_diacritics: true,
    normalize_alef: true,
    normalize_teh: true,
    ..Default::default()
};
```

## Performance Optimization by Language

### Language-Specific Model Selection

```rust
// Optimize model size based on language complexity
fn get_optimal_model_size(language: LanguageCode) -> WhisperModelSize {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => WhisperModelSize::Base,
        LanguageCode::ZhCn | LanguageCode::JaJp | LanguageCode::ArAr => WhisperModelSize::Large,
        _ => WhisperModelSize::Small,
    }
}

let config = ASRConfig {
    language: Some(LanguageCode::ZhCn),
    model_size: Some(get_optimal_model_size(LanguageCode::ZhCn)),
    ..Default::default()
};
```

### Language-Specific Preprocessing

```rust
// Different preprocessing for different languages
let preprocessing_config = match language {
    LanguageCode::EnUs => PreprocessingConfig {
        noise_suppression: true,
        auto_gain_control: true,
        echo_cancellation: true,
        ..Default::default()
    },
    LanguageCode::ZhCn => PreprocessingConfig {
        noise_suppression: true,
        auto_gain_control: false,  // Preserve tonal information
        echo_cancellation: true,
        ..Default::default()
    },
    LanguageCode::ArAr => PreprocessingConfig {
        noise_suppression: false,  // Preserve emphatic consonants
        auto_gain_control: true,
        echo_cancellation: true,
        ..Default::default()
    },
    _ => PreprocessingConfig::default(),
};
```

## Translation Integration

### Recognition + Translation

```rust
// Recognize and translate in one pipeline
let config = ASRConfig {
    language: Some(LanguageCode::EsEs),
    translate_to: Some(LanguageCode::EnUs),
    ..Default::default()
};

let mut asr = ASRBackend::new_whisper(config).await?;
let result = asr.recognize(&audio, None).await?;

println!("Original (Spanish): {}", result.text);
if let Some(translation) = result.translation {
    println!("Translation (English): {}", translation);
}
```

## Best Practices

### 1. Language Detection Strategy

```rust
// Use language detection for unknown content
async fn smart_language_detection(audio: &AudioBuffer) -> Result<LanguageCode, RecognitionError> {
    // First, try auto-detection
    let detection_config = ASRConfig {
        language: None,
        detect_language: true,
        language_detection_threshold: 0.8,
        ..Default::default()
    };
    
    let mut detector = ASRBackend::new_whisper(detection_config).await?;
    let result = detector.recognize(audio, None).await?;
    
    if let Some(lang) = result.language {
        if result.language_confidence > 0.8 {
            return Ok(lang);
        }
    }
    
    // Fallback to English if detection fails
    Ok(LanguageCode::EnUs)
}
```

### 2. Language-Specific Quality Checks

```rust
// Validate results based on language characteristics
fn validate_language_result(result: &RecognitionResult, language: LanguageCode) -> bool {
    match language {
        LanguageCode::EnUs => {
            // English should have spaces between words
            result.text.contains(' ') && result.confidence > 0.5
        },
        LanguageCode::ZhCn => {
            // Chinese may not have spaces, check character count
            result.text.chars().count() > 0 && result.confidence > 0.6
        },
        LanguageCode::JaJp => {
            // Japanese mixed scripts validation
            result.text.chars().any(|c| c.is_ascii_alphanumeric() || 
                                   (c >= '\u{3040}' && c <= '\u{309f}') || // Hiragana
                                   (c >= '\u{30a0}' && c <= '\u{30ff}'))   // Katakana
        },
        _ => result.confidence > 0.5,
    }
}
```

### 3. Efficient Multi-language Processing

```rust
// Reuse models when possible
struct MultiLanguageProcessor {
    models: HashMap<LanguageCode, ASRBackend>,
}

impl MultiLanguageProcessor {
    async fn new(languages: Vec<LanguageCode>) -> Result<Self, RecognitionError> {
        let mut models = HashMap::new();
        
        for lang in languages {
            let config = ASRConfig {
                language: Some(lang),
                ..Default::default()
            };
            let asr = ASRBackend::new_whisper(config).await?;
            models.insert(lang, asr);
        }
        
        Ok(Self { models })
    }
    
    async fn recognize(&mut self, audio: &AudioBuffer, language: LanguageCode) -> Result<RecognitionResult, RecognitionError> {
        if let Some(asr) = self.models.get_mut(&language) {
            asr.recognize(audio, None).await
        } else {
            Err(RecognitionError::UnsupportedLanguage(language))
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Low accuracy for non-English languages**: Use larger models (Small/Large)
2. **Incorrect language detection**: Set language candidates or use manual language selection
3. **Poor accent recognition**: Enable accent adaptation or use accent-specific models
4. **Text normalization issues**: Configure language-specific normalization

### Debug Language Detection

```rust
// Debug language detection
let debug_config = ASRConfig {
    language: None,
    detect_language: true,
    language_detection_debug: true,
    ..Default::default()
};

let mut asr = ASRBackend::new_whisper(debug_config).await?;
let result = asr.recognize(&audio, None).await?;

if let Some(debug_info) = result.language_debug_info {
    println!("Language detection debug:");
    for (lang, score) in debug_info {
        println!("  {:?}: {:.3}", lang, score);
    }
}
```

## Next Steps

- Explore [Phoneme Alignment](./phoneme-alignment.md) for detailed linguistic analysis
- Learn about [Audio Analysis](./audio-analysis.md) for quality assessment
- Check out [Real-time Streaming](./streaming.md) for live multilingual processing
- Review [Performance Optimization](./performance.md) for production deployment