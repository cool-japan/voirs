# voirs-g2p

[![Crates.io](https://img.shields.io/crates/v/voirs-g2p.svg)](https://crates.io/crates/voirs-g2p)
[![Documentation](https://docs.rs/voirs-g2p/badge.svg)](https://docs.rs/voirs-g2p)

**Grapheme-to-Phoneme (G2P) conversion for VoiRS speech synthesis framework.**

This crate provides high-quality text-to-phoneme conversion with support for multiple languages and backends. It serves as the first stage in the VoiRS speech synthesis pipeline, converting input text into phonetic representations that can be processed by acoustic models.

## Features

- **Multi-backend Support**: Phonetisaurus (FST), OpenJTalk (Japanese), Neural G2P (LSTM)
- **Multi-language**: 20+ languages with extensible language pack system
- **High Accuracy**: >95% phoneme accuracy on standard benchmarks
- **Performance**: <1ms latency for typical sentences, >1000 sentences/second batch processing
- **Flexible Input**: Raw text, SSML markup, mixed languages
- **Rich Output**: IPA phonemes, stress markers, syllable boundaries, timing information

## Quick Start

```rust
use voirs_g2p::{G2p, PhoneticusG2p, Phoneme};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize English G2P with Phonetisaurus backend
    let g2p = PhoneticusG2p::new("en-US").await?;
    
    // Convert text to phonemes
    let phonemes: Vec<Phoneme> = g2p.to_phonemes("Hello world!", None).await?;
    
    // Print phonetic representation
    for phoneme in phonemes {
        println!("{}", phoneme.symbol());
    }
    
    Ok(())
}
```

## Supported Languages

| Language | Backend | Accuracy | Status |
|----------|---------|----------|--------|
| English (US) | Phonetisaurus | 95.2% | ✅ Stable |
| English (UK) | Phonetisaurus | 94.8% | ✅ Stable |
| Japanese | OpenJTalk | 92.1% | ✅ Stable |
| Spanish | Neural G2P | 89.3% | 🚧 Beta |
| French | Neural G2P | 88.7% | 🚧 Beta |
| German | Neural G2P | 88.1% | 🚧 Beta |
| Mandarin | Neural G2P | 85.9% | 🚧 Beta |

## Backends

### Phonetisaurus (FST-based)
- **Best for**: English and well-resourced languages
- **Pros**: Very fast, high accuracy, deterministic
- **Cons**: Requires pre-built FST models
- **Memory**: ~50MB per language model

### OpenJTalk (Japanese)
- **Best for**: Japanese text processing
- **Pros**: Handles Kanji→Kana conversion, pitch accent
- **Cons**: Japanese-specific, requires C library
- **Memory**: ~100MB for full Japanese model

### Neural G2P (LSTM-based)
- **Best for**: Under-resourced languages, fallback
- **Pros**: Trainable, handles unseen words well
- **Cons**: Slower inference, requires training data
- **Memory**: ~20MB per language model

### Modern Transformer G2P ⚡ NEW
- **State-of-the-art neural architecture** with cutting-edge improvements
- **Rotary Position Embeddings (RoPE)**: Superior position encoding from LLaMA/PaLM
- **SwiGLU Activation**: Advanced gating mechanism from PaLM 540B
- **ALiBi Position Bias**: Excellent extrapolation to longer sequences
- **Multi-Head Attention**: Parallel attention mechanisms for richer representations
- **Benefits**:
  - Better accuracy on complex pronunciation patterns
  - Improved handling of rare words and proper nouns
  - Excellent zero-shot generalization
  - Scales efficiently to longer sequences
  - Production-ready with comprehensive benchmarking

## Architecture

```
Text Input → Preprocessing → Language Detection → Backend Selection → Phonemes
     ↓              ↓               ↓                    ↓              ↓
  "Hello"      "hello"          "en-US"          Phonetisaurus    [HH, AH, L, OW]
```

### Core Components

1. **Text Preprocessing**
   - Unicode normalization (NFC, NFD)
   - Number expansion ("123" → "one hundred twenty three")
   - Abbreviation expansion ("Dr." → "Doctor")
   - Currency/date parsing

2. **Language Detection**
   - Rule-based for ASCII text
   - Statistical models for Unicode scripts
   - Confidence scoring and fallback

3. **Backend Routing**
   - Language-specific backend selection
   - Fallback chain (primary → neural → default)
   - Load balancing for high throughput

4. **Phoneme Generation**
   - IPA standardization
   - Stress and syllable marking
   - Duration prediction
   - Quality scoring

5. **Phonological Processing** (19 total processes)
   - **Place Assimilation**: Nasals adapt to following consonant place (e.g., "input" → /ɪmpʊt/)
   - **Voicing Assimilation**: Consonants match voicing of neighbors (German, Russian)
   - **Vowel Reduction**: Unstressed vowels reduce to schwa in casual speech
   - **Elision**: Sound deletion in rapid/casual speech (e.g., schwa deletion)
   - **Liaison**: Linking sounds between words (French)
   - **Nasalization**: Vowels nasalize before nasal consonants (French, Portuguese)
   - **Palatalization**: Consonants palatalize before front vowels (Japanese, Russian)
   - **Lenition**: Intervocalic consonant weakening (Spanish /b/ → [β])
   - **Fortition**: Consonant gemination after stressed vowels (Italian, Japanese)
   - **R-dropping**: Non-rhotic pronunciation in British RP, New England
   - **T-flapping**: American English /t/ → [ɾ] between vowels
   - **Final Devoicing**: German Auslautverhärtung (word-final devoicing)
   - **Vowel Devoicing**: Japanese high vowel devoicing between voiceless consonants
   - **H-dropping**: British English dialects (Cockney, Yorkshire) 🆕
   - **TH-fronting**: London/Southern US English (/θ/ → /f/, /ð/ → /v/) 🆕
   - **L-vocalization**: London English, Portuguese (/l/ → /w/ in coda) 🆕
   - **G-dropping**: Casual English -ing endings (/ŋ/ → /n/) 🆕
   - **T-glottaling**: British English (/t/ → /ʔ/ in coda) 🆕
   - **Yod-coalescence**: American English (/tj/ → /tʃ/, /dj/ → /dʒ/) 🆕
   - Context-aware coarticulation effects

   🆕 = New dialect-specific processes added in v0.1.0-alpha.3

## Advanced Phonological Processing Examples

The phonological processes enable natural and dialect-specific pronunciation across major world languages:

### Dialect-Specific Processes (English)

```rust
use voirs_g2p::phonology::{PhonologicalProcessor, ProcessConfig};
use voirs_g2p::{Phoneme, LanguageCode};

// British English: H-dropping (Cockney)
let config = ProcessConfig {
    enable_h_dropping: true,
    language: LanguageCode::EnGb,
    aggressiveness: 0.5,
    ..Default::default()
};
let processor = PhonologicalProcessor::with_config(config);
let input = vec![Phoneme::new("h".into()), Phoneme::new("aʊ".into()), Phoneme::new("s".into())];
let output = processor.apply_all_processes(&input)?;
// "house" /haʊs/ → /aʊs/

// British English: TH-fronting (London)
let config = ProcessConfig {
    enable_th_fronting: true,
    language: LanguageCode::EnGb,
    aggressiveness: 0.6,
    ..Default::default()
};
let processor = PhonologicalProcessor::with_config(config);
let input = vec![Phoneme::new("θ".into()), Phoneme::new("ɪ".into()), Phoneme::new("ŋ".into())];
let output = processor.apply_all_processes(&input)?;
// "think" /θɪŋ/ → /fɪŋ/

// British English: L-vocalization (Estuary)
let config = ProcessConfig {
    enable_l_vocalization: true,
    language: LanguageCode::EnGb,
    aggressiveness: 0.6,
    ..Default::default()
};
let processor = PhonologicalProcessor::with_config(config);
let input = vec![Phoneme::new("m".into()), Phoneme::new("ɪ".into()),
                 Phoneme::new("l".into()), Phoneme::new("k".into())];
let output = processor.apply_all_processes(&input)?;
// "milk" /mɪlk/ → /mɪwk/

// American English: Yod-coalescence
let config = ProcessConfig {
    enable_yod_coalescence: true,
    language: LanguageCode::EnUs,
    aggressiveness: 0.5,
    ..Default::default()
};
let processor = PhonologicalProcessor::with_config(config);
let input = vec![Phoneme::new("t".into()), Phoneme::new("j".into()),
                 Phoneme::new("uː".into()), Phoneme::new("n".into())];
let output = processor.apply_all_processes(&input)?;
// "tune" /tjuːn/ → /tʃuːn/
```

### Language-Specific Processes

```rust
// French Nasalization
let config = ProcessConfig {
    enable_nasalization: true,
    language: LanguageCode::Fr,
    aggressiveness: 0.5,
    ..Default::default()
};
let processor = PhonologicalProcessor::with_config(config);
let input = vec![Phoneme::new("a".into()), Phoneme::new("n".into())];
let output = processor.apply_all_processes(&input)?;
// "a" becomes nasalized: "ã"

// Spanish Lenition
let config = ProcessConfig {
    enable_lenition: true,
    language: LanguageCode::Es,
    aggressiveness: 0.6,
    ..Default::default()
};
let processor = PhonologicalProcessor::with_config(config);
let input = vec![
    Phoneme::new("a".into()),
    Phoneme::new("b".into()),  // Will weaken to [β] between vowels
    Phoneme::new("o".into()),
];
let output = processor.apply_all_processes(&input)?;

// Japanese Palatalization
let config = ProcessConfig {
    enable_palatalization: true,
    language: LanguageCode::Ja,
    aggressiveness: 0.7,
    ..Default::default()
};
let processor = PhonologicalProcessor::with_config(config);
let input = vec![Phoneme::new("t".into()), Phoneme::new("i".into())];
let output = processor.apply_all_processes(&input)?;
// "t" palatalizes before "i": "tʲ"

// Italian Fortition (Gemination)
let config = ProcessConfig {
    enable_fortition: true,
    language: LanguageCode::It,
    aggressiveness: 0.7,
    ..Default::default()
};
let processor = PhonologicalProcessor::with_config(config);
let mut input = vec![
    {
        let mut p = Phoneme::new("a".into());
        p.stress = 1;  // Stressed vowel
        p
    },
    Phoneme::new("t".into()),  // Will geminate: "tt"
    Phoneme::new("o".into()),
];
let output = processor.apply_all_processes(&input)?;
```

## API Reference

### Core Trait

```rust
#[async_trait]
pub trait G2p: Send + Sync {
    /// Convert text to phonemes for given language
    async fn to_phonemes(&self, text: &str, lang: Option<&str>) -> Result<Vec<Phoneme>>;

    /// Get list of supported language codes
    fn supported_languages(&self) -> Vec<LanguageCode>;

    /// Get backend metadata and capabilities
    fn metadata(&self) -> G2pMetadata;

    /// Preprocess text before phoneme conversion
    async fn preprocess(&self, text: &str, lang: Option<&str>) -> Result<String>;

    /// Detect language of input text
    async fn detect_language(&self, text: &str) -> Result<LanguageCode>;
}
```

### Phoneme Representation

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct Phoneme {
    /// IPA symbol (e.g., "æ", "t̪", "d͡ʒ")
    pub symbol: String,
    
    /// Stress level (0=none, 1=primary, 2=secondary)
    pub stress: u8,
    
    /// Position within syllable
    pub syllable_position: SyllablePosition,
    
    /// Predicted duration in milliseconds
    pub duration_ms: Option<f32>,
    
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}
```

### Language Support

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LanguageCode {
    EnUs,   // English (US)
    EnGb,   // English (UK)
    JaJp,   // Japanese
    EsEs,   // Spanish (Spain)
    EsMx,   // Spanish (Mexico)
    FrFr,   // French (France)
    DeDE,   // German (Germany)
    ZhCn,   // Chinese (Simplified)
    // ... more languages
}
```

## Usage Examples

### Basic Text-to-Phoneme Conversion

```rust
use voirs_g2p::{PhoneticusG2p, G2p};

let g2p = PhoneticusG2p::new("en-US").await?;
let phonemes = g2p.to_phonemes("The quick brown fox.", None).await?;

// Convert to IPA string
let ipa: String = phonemes.iter()
    .map(|p| p.symbol.as_str())
    .collect::<Vec<_>>()
    .join(" ");
println!("IPA: {}", ipa);
```

### Multi-language Processing

```rust
use voirs_g2p::{MultilingualG2p, G2p};

let g2p = MultilingualG2p::builder()
    .add_backend("en", PhoneticusG2p::new("en-US").await?)
    .add_backend("ja", OpenJTalkG2p::new().await?)
    .build();

// Automatic language detection
let text = "Hello world! こんにちは世界！";
let phonemes = g2p.to_phonemes(text, None).await?;
```

### SSML Processing

```rust
use voirs_g2p::{SsmlG2p, G2p};

let g2p = SsmlG2p::new(PhoneticusG2p::new("en-US").await?);

let ssml = r#"
<speak>
    <phoneme alphabet="ipa" ph="təˈmeɪtoʊ">tomato</phoneme>
    versus
    <phoneme alphabet="ipa" ph="təˈmɑːtoʊ">tomato</phoneme>
</speak>
"#;

let phonemes = g2p.to_phonemes(ssml, Some("en-US")).await?;
```

### Batch Processing

```rust
use voirs_g2p::{BatchG2p, G2p};

let g2p = PhoneticusG2p::new("en-US").await?;
let batch_g2p = BatchG2p::new(g2p, 32); // batch size of 32

let texts = vec![
    "First sentence.",
    "Second sentence.",
    "Third sentence.",
];

let results = batch_g2p.to_phonemes_batch(&texts, None).await?;
```

### Phonological Processing

Apply natural phonological processes to enhance pronunciation quality:

```rust
use voirs_g2p::phonology::{PhonologicalProcessor, ProcessConfig};
use voirs_g2p::{Phoneme, LanguageCode};

// Create a processor for English
let processor = PhonologicalProcessor::new(LanguageCode::EnUs);

// Input phonemes (e.g., from G2P conversion of "input")
let phonemes = vec![
    Phoneme::new("ɪ".to_string()),
    Phoneme::new("n".to_string()),
    Phoneme::new("p".to_string()),
    Phoneme::new("ʊ".to_string()),
    Phoneme::new("t".to_string()),
];

// Apply phonological processes
let processed = processor.apply_all_processes(&phonemes)?;
// Result: [ɪ, m, p, ʊ, t] - /n/ assimilated to /m/ before /p/

// Custom configuration for more aggressive processing
let config = ProcessConfig {
    enable_place_assimilation: true,
    enable_vowel_reduction: true,
    aggressiveness: 0.8,
    ..Default::default()
};

let processor = PhonologicalProcessor::with_config(config);
```

### Custom Preprocessing

```rust
use voirs_g2p::{G2p, TextPreprocessor};

let mut preprocessor = TextPreprocessor::new("en-US");
preprocessor.add_rule(r"\$(\d+)", |caps| {
    format!("{} dollars", caps[1].parse::<i32>().unwrap())
});

let g2p = PhoneticusG2p::with_preprocessor("en-US", preprocessor).await?;
let phonemes = g2p.to_phonemes("It costs $5.99", None).await?;
```

## Language-Specific Phoneme Inventories ✨ NEW

voirs-g2p now includes comprehensive phoneme inventories for major world languages with detailed phonetic information. These inventories provide:

- Complete phoneme sets with IPA symbols
- Phonetic features (place, manner, voicing, vowel height/frontness/rounding)
- Grapheme-to-phoneme mappings
- Language-specific processing rules

### Supported Language Inventories

#### Chinese (Mandarin) 🇨🇳
- **21 initial consonants** (声母 shēngmǔ): Including retroflex (zh, ch, sh, r), palatal (j, q, x), and dental sibilants (z, c, s)
- **23 finals** (韵母 yùnmǔ): Monophthongs, diphthongs, and nasal finals
- **5 tones** (声调 shēngdiào): High level (55), Rising (35), Falling-rising (214), Falling (51), Neutral
- **Unique features**: Retroflex consonants, fronted rounded vowels (ü), comprehensive tone system

#### Japanese 🇯🇵
- **19 consonants** (子音 shiin): Including palatalized variants (tɕ, dʑ, ɕ) and special sounds (bilabial fricative ɸ, tap ɾ)
- **5 vowels** (母音 boin): Pure vowel system (a, i, ɯ, e, o) with no diphthongs
- **Special morae**: Moraic nasal /ɴ/ (ん), geminate marker /Q/ (っ)
- **Unique features**: Mora-timed rhythm, pitch accent system, simple (C)V(N) syllable structure, back unrounded /ɯ/

#### Korean (Hangul) 🇰🇷
- **19 initial consonants** (초성 choseong): Three-way contrast (plain/aspirated/tense)
- **21 medial vowels** (중성 jungseong): 7 monophthongs + 13 diphthongs
- **27 final consonants** (종성 jongseong): Including clusters
- **Unique features**: Three-way fortis/lenis/aspirated distinction, syllable-block structure

#### Russian 🇷🇺
- **Comprehensive consonant system**: Palatalized pairs, voiced/voiceless contrasts
- **Vowel reduction patterns**: Unstressed vowel centralization
- **Phonological processes**: Voicing assimilation, final devoicing

#### Arabic 🇸🇦
- **Emphatic consonants**: Pharyngealized stops and fricatives
- **Uvular and pharyngeal sounds**: Extensive back-of-throat articulations
- **Tri-consonantal roots**: Support for morphological patterns

### Usage Example

```rust
use voirs_g2p::languages::{get_inventory, PhonemeInventory};
use voirs_g2p::LanguageCode;

// Get Chinese (Mandarin) phoneme inventory
let inventory = get_inventory(LanguageCode::ZhCn).unwrap();

// Check phoneme validity
assert!(inventory.is_valid_phoneme("ʈʂ"));  // Retroflex affricate (zh)
assert!(inventory.is_valid_phoneme("ɕ"));   // Palatal fricative (x)
assert!(inventory.is_valid_phoneme("ü"));   // Front rounded vowel

// Get all phonemes for the language
let all_phonemes = inventory.all_phonemes();
println!("Chinese has {} phonemes", all_phonemes.len());

// Look up possible phonemes for a grapheme (Pinyin)
if let Some(phonemes) = inventory.phonemes_for_grapheme("zh") {
    println!("'zh' maps to: {:?}", phonemes);  // ["ʈʂ"]
}

// Japanese mora structure example
let japanese_inventory = get_inventory(LanguageCode::Ja).unwrap();
assert!(japanese_inventory.is_valid_phoneme("tɕ"));  // Palatalized 'chi'
assert!(japanese_inventory.is_valid_phoneme("ɴ"));   // Moraic nasal 'n'
assert!(japanese_inventory.is_valid_phoneme("ɸ"));   // Bilabial fricative 'fu'

// Korean three-way contrast example
let korean_inventory = get_inventory(LanguageCode::Ko).unwrap();
assert!(korean_inventory.is_valid_phoneme("k"));   // Plain velar
assert!(korean_inventory.is_valid_phoneme("kʰ"));  // Aspirated velar
assert!(korean_inventory.is_valid_phoneme("k͈"));   // Tense velar
```

### Tone System Support (Chinese)

The Chinese module includes comprehensive tone support:

```rust
use voirs_g2p::languages::chinese::MandarinTone;

let tone1 = MandarinTone::Tone1;
assert_eq!(tone1.number(), 1);
assert_eq!(tone1.contour(), "55");  // High level
assert_eq!(tone1.diacritic_example(), "ā");

let tone3 = MandarinTone::Tone3;
assert_eq!(tone3.contour(), "214");  // Falling-rising
```

## Performance

### Benchmarks (Intel i7-12700K)

| Backend | Latency (1 sentence) | Throughput (batch) | Memory Usage |
|---------|---------------------|-------------------|--------------|
| Phonetisaurus | 0.3ms | 2,500 sent/s | 50MB |
| OpenJTalk | 0.8ms | 1,200 sent/s | 100MB |
| Neural G2P | 2.1ms | 800 sent/s | 20MB |

### Memory Usage
- **Phonetisaurus**: 50MB per language model
- **OpenJTalk**: 100MB for full Japanese model
- **Neural G2P**: 20MB per language model
- **Runtime overhead**: 5-10MB per backend instance

### Parallel Processing

voirs-g2p includes high-performance parallel processing capabilities for batch operations using SciRS2-Core's parallel abstractions:

**Automatic Parallelization**: Batch operations automatically leverage all available CPU cores for large batches:
- **`parallel_batch_process`**: Processes ≥100 sequences in parallel (sequential for <100)
- **`batch_distance_matrix`**: Parallel distance matrix computation for ≥20 sequences
- **Performance**: Speedup proportional to CPU core count for CPU-bound tasks

```rust
use voirs_g2p::utils::phoneme_simd::{parallel_batch_process, batch_distance_matrix};
use voirs_g2p::Phoneme;

// Parallel processing of large batches
let sequences: Vec<Vec<Phoneme>> = /* ... 500 sequences ... */;

// Automatically uses parallel processing (≥100 threshold)
let lengths = parallel_batch_process(&sequences, |seq| seq.len());

// Parallel distance matrix for clustering/similarity search
let matrix = batch_distance_matrix(&sequences);  // Parallel for ≥20 sequences
```

**Benchmark Results** (12-core CPU):
- **Batch Processing (500 sequences)**: ~10× speedup vs sequential
- **Distance Matrix (50 sequences)**: ~8× speedup vs sequential
- **Threshold Overhead**: <5% for boundary cases (99-101 sequences)

Run benchmarks: `cargo bench --bench parallel_processing_benchmarks`

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
voirs-g2p = "0.1"

# Optional backends
[dependencies.voirs-g2p]
version = "0.1"
features = ["phonetisaurus", "openjtalk", "neural"]
```

### Feature Flags

- `phonetisaurus`: Enable Phonetisaurus FST backend
- `openjtalk`: Enable OpenJTalk Japanese backend  
- `neural`: Enable neural LSTM backend
- `all-backends`: Enable all available backends
- `cli`: Enable command-line binary

### System Dependencies

**Phonetisaurus backend:**
```bash
# Ubuntu/Debian
sudo apt-get install libfst-dev

# macOS
brew install openfst
```

**OpenJTalk backend:**
```bash
# Ubuntu/Debian
sudo apt-get install libopenjtalk-dev

# macOS  
brew install open-jtalk
```

## Configuration

Create `~/.voirs/g2p.toml`:

```toml
[default]
language = "en-US"
backend = "phonetisaurus"

[preprocessing]
expand_numbers = true
expand_abbreviations = true
normalize_unicode = true

[phonetisaurus]
model_path = "~/.voirs/models/g2p/"
cache_size = 10000

[openjtalk]
dictionary_path = "/usr/share/open-jtalk/dic"
voice_path = "/usr/share/open-jtalk/voice"

[neural]
model_path = "~/.voirs/models/neural-g2p/"
device = "cpu"  # or "cuda:0"
```

## Error Handling

```rust
use voirs_g2p::{G2pError, ErrorKind};

match g2p.to_phonemes("text", None).await {
    Ok(phonemes) => println!("Success: {} phonemes", phonemes.len()),
    Err(G2pError { kind, context, .. }) => match kind {
        ErrorKind::UnsupportedLanguage => {
            eprintln!("Language not supported: {}", context);
        }
        ErrorKind::ModelNotFound => {
            eprintln!("Model files missing: {}", context);
        }
        ErrorKind::ParseError => {
            eprintln!("Failed to parse input: {}", context);
        }
        _ => eprintln!("Other error: {}", context),
    }
}
```

## Contributing

We welcome contributions! Please see the [main repository](https://github.com/cool-japan/voirs) for contribution guidelines.

### Development Setup

```bash
git clone https://github.com/cool-japan/voirs.git
cd voirs/crates/voirs-g2p

# Install development dependencies
cargo install cargo-nextest

# Run tests
cargo nextest run

# Run benchmarks
cargo bench

# Check code quality
cargo clippy -- -D warnings
cargo fmt --check
```

### Adding New Languages

1. Implement the `G2p` trait for your language
2. Add language code to `LanguageCode` enum
3. Create test cases with reference phoneme data
4. Add documentation and examples
5. Submit a pull request

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.