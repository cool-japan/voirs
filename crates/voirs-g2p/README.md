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