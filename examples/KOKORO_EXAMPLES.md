# Kokoro Multilingual TTS Examples

This directory contains examples demonstrating **pure Rust multilingual text-to-speech** using the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) ONNX model.

## Overview

VoiRS supports **9 languages** through the Kokoro-82M model:
- 🇺🇸 🇬🇧 **English** (American & British)
- 🇪🇸 **Spanish**
- 🇫🇷 **French**
- 🇮🇳 **Hindi**
- 🇮🇹 **Italian**
- 🇧🇷 **Portuguese** (Brazilian)
- 🇯🇵 **Japanese**
- 🇨🇳 **Chinese** (Mandarin)

**Key Features:**
- ✅ **Pure Rust** - No Python dependencies at runtime
- ✅ **NumPy .npz Support** - Direct loading via `numrs2` (no conversion scripts needed)
- ✅ **54 Voices** - Multiple voices per language
- ✅ **ONNX Runtime** - Cross-platform inference
- ✅ **IPA Phoneme Input** - International Phonetic Alphabet for accurate pronunciation

## Prerequisites

### 1. Download Kokoro Model

**Option A: Using VoiRS CLI (Recommended)**

The easiest way to download the model with progress bars:

```bash
voirs kokoro download
```

This downloads the quantized ONNX model (86MB), config, and a sample voice to `$TMPDIR/voirs_models/kokoro-zh`.

**Option B: Manual Download**

Download the Kokoro-82M ONNX model files manually:

```bash
# Model directory
TEMP_DIR=$(python3 -c 'import tempfile; print(tempfile.gettempdir())')
MODEL_DIR="$TEMP_DIR/voirs_models/kokoro-zh"
mkdir -p "$MODEL_DIR"

cd "$MODEL_DIR"

# Download model files from ONNX Community repository
wget https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_q8f16.onnx -O model.onnx
wget https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/config.json
wget https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af.bin
```

**Note:**
- The quantized model (`model_q8f16.onnx`) is 86MB vs 326MB for the full precision model
- Voice files are now individual `.bin` files per voice (512KB each)
- 50 voices available at: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main/voices

### 2. Install ONNX Runtime

VoiRS uses the `ort` crate which requires ONNX Runtime to be installed:

**macOS:**
```bash
brew install onnxruntime
```

**Linux (Ubuntu/Debian):**
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz
tar -xzf onnxruntime-linux-x64-1.17.0.tgz
sudo cp -r onnxruntime-linux-x64-1.17.0/lib/* /usr/local/lib/
sudo cp -r onnxruntime-linux-x64-1.17.0/include/* /usr/local/include/
sudo ldconfig
```

**Windows:**
```powershell
# Download from https://github.com/microsoft/onnxruntime/releases
# Add to PATH
```

## CLI Quick Start

VoiRS includes a powerful CLI for Kokoro TTS with advanced features:

### Basic Synthesis

```bash
# Simple synthesis (auto-generates IPA phonemes)
voirs kokoro synth "Hello world" output.wav --lang en-us

# With voice selection
voirs kokoro synth "Hello world" output.wav --lang en-us --voice-name af_jessica

# With playback after synthesis
voirs kokoro synth "Hello world" output.wav --lang en-us --play

# Stream to stdout (pipe to other tools)
voirs kokoro synth "Hello world" - --lang en-us > output.wav
```

### Configuration

```bash
# Show current configuration
voirs kokoro config --show

# Initialize config file at ~/.config/voirs/kokoro.toml
voirs kokoro config --init

# Set defaults (in config file)
default_lang = "en-us"
default_voice = "af_jessica"
default_speed = 1.0
```

### Voice Management

```bash
# List available voices
voirs kokoro voices

# List voices for specific language
voirs kokoro voices --lang ja

# Detailed voice information
voirs kokoro voices --detailed
```

### Available Languages

| Code | Language | Voices |
|------|----------|--------|
| `en-us` | American English | af_*, am_* (20 voices) |
| `en-gb` | British English | bf_*, bm_* (8 voices) |
| `es` | Spanish | ef_*, em_* (3 voices) |
| `fr` | French | ff_* (1 voice) |
| `hi` | Hindi | hf_*, hm_* (3 voices) |
| `it` | Italian | if_*, im_* (2 voices) |
| `pt-br` | Portuguese (Brazil) | pf_*, pm_* (3 voices) |
| `ja` | Japanese | jf_*, jm_* (5 voices) |
| `zh` | Chinese (Mandarin) | zf_* (3 voices) |

## Examples

### 1. Japanese TTS Demo (`kokoro_japanese_demo.rs`)

Demonstrates Japanese text-to-speech synthesis.

**Run:**
```bash
cargo run --example kokoro_japanese_demo --features onnx --release
```

**Output:**
- Text: "こんにちは、今日の世界も美しいです"
- Voice: `jf_alpha` (Japanese female)
- Duration: ~2.3 seconds
- File: `$TMPDIR/kokoro_japanese_rust.wav`

**Features:**
- IPA phonemes from misaki library
- Trailing silence trimming
- 100ms padding at start

### 2. Chinese TTS Demo (`kokoro_chinese_demo.rs`)

Demonstrates Mandarin Chinese text-to-speech synthesis.

**Run:**
```bash
cargo run --example kokoro_chinese_demo --features onnx --release
```

**Output:**
- Text: "你好，今天的世界也很美丽"
- Voice: `zf_xiaobei` (Chinese female)
- Duration: ~3.3 seconds
- File: `$TMPDIR/kokoro_chinese_rust.wav`

**Features:**
- IPA phonemes with tone marks (↓↘→) placed **before** syllables
- No silence trimming (preserves natural beginning)
- 100ms padding at start

### 3. Multilingual Demo (`kokoro_multilingual_demo.rs`)

Demonstrates **all 9 supported languages** in a single run with pre-generated IPA phonemes.

**Run:**
```bash
cargo run --example kokoro_multilingual_demo --features onnx --release
```

**Output:**
Generates 9 WAV files in `$TMPDIR`:
- `kokoro_english_american_af_jessica.wav` (2.24s)
- `kokoro_english_british_bf_alice.wav` (2.31s)
- `kokoro_spanish_ef_dora.wav` (1.68s)
- `kokoro_french_ff_siwis.wav` (1.90s)
- `kokoro_hindi_hf_alpha.wav` (2.78s)
- `kokoro_italian_if_sara.wav` (1.38s)
- `kokoro_portuguese_brazilian_pf_dora.wav` (1.63s)
- `kokoro_japanese_jf_alpha.wav` (2.34s)
- `kokoro_chinese_mandarin_zf_xiaobei.wav` (3.31s)

**Play audio:**
```bash
# macOS
afplay $TMPDIR/kokoro_english_american_af_jessica.wav

# Linux
aplay $TMPDIR/kokoro_english_american_af_jessica.wav
```

### 4. 🆕 Automatic IPA Generation Demo (`kokoro_espeak_auto_demo.rs`)

**New!** Demonstrates **automatic IPA phoneme generation** using eSpeak NG. Just provide text - no manual IPA needed!

**Supported Languages:**
- 🇺🇸 🇬🇧 English (American & British)
- 🇪🇸 Spanish
- 🇫🇷 French
- 🇮🇳 Hindi
- 🇮🇹 Italian
- 🇧🇷 Portuguese (Brazilian)

**Prerequisites:**
```bash
# Install eSpeak NG
brew install espeak-ng  # macOS
sudo apt install espeak-ng  # Linux
```

**Run:**
```bash
cargo run --example kokoro_espeak_auto_demo --features onnx --release
```

**Output:**
Automatically generates IPA phonemes and synthesizes speech for 7 languages:
- `kokoro_auto_english_american_af_jessica.wav`
- `kokoro_auto_english_british_bf_alice.wav`
- `kokoro_auto_spanish_ef_dora.wav`
- `kokoro_auto_french_ff_siwis.wav`
- `kokoro_auto_hindi_hf_alpha.wav`
- `kokoro_auto_italian_if_sara.wav`
- `kokoro_auto_portuguese_brazilian_pf_dora.wav`

**Features:**
- ✅ Automatic IPA generation - no manual phoneme creation needed
- ✅ Pure Rust with eSpeak NG subprocess calls
- ✅ Text-to-speech in one command
- ✅ Error handling for missing eSpeak NG

**Note:** Japanese and Chinese are not included because:
- Japanese: eSpeak NG quality is poor compared to misaki library
- Chinese: Tone mark handling requires special formatting (misaki recommended)

## CLI Usage

VoiRS provides a comprehensive command-line interface for Kokoro multilingual TTS. All commands require the `onnx` feature.

### Installation

First, install the `voirs` CLI tool:

```bash
cargo install --path crates/voirs-cli --features onnx
```

Or from crates.io (when published):

```bash
cargo install voirs-cli --features onnx
```

### Download Model Files

```bash
# Download to default location ($TMPDIR/voirs_models/kokoro-zh)
voirs kokoro download

# Download to custom directory
voirs kokoro download --output /path/to/models
```

### List Available Voices

```bash
# List all voices (54 total)
voirs kokoro voices

# Filter by language
voirs kokoro voices --lang ja

# Detailed output with gender info
voirs kokoro voices --lang zh --detailed

# Output as JSON
voirs kokoro voices --format json

# Output as CSV
voirs kokoro voices --format csv
```

### List Supported Languages

```bash
# List all languages
voirs kokoro languages

# Show IPA generation method for each language
voirs kokoro languages --show-ipa-method
```

### Synthesize Speech

**Automatic IPA generation (default for en-us, en-gb, es, fr, hi, it, pt-br):**

```bash
# Basic synthesis (auto-IPA enabled by default)
voirs kokoro synth "Hello world" output.wav --lang en-us

# Spanish synthesis
voirs kokoro synth "Hola mundo" spanish.wav --lang es

# Specify voice by name
voirs kokoro synth "Bonjour le monde" french.wav --lang fr --voice-name ff_siwis

# Specify voice by index
voirs kokoro synth "Hello" hello.wav --lang en-gb --voice-index 20

# Adjust speaking speed (0.5 - 2.0)
voirs kokoro synth "Fast speech" fast.wav --lang en-us --speed 1.5
```

**Manual IPA input (required for Japanese/Chinese):**

```bash
# Japanese with manual IPA
voirs kokoro synth "こんにちは" japanese.wav --lang ja --ipa "koɲɲiʨiβa"

# Chinese with manual IPA (tone marks before syllables)
voirs kokoro synth "你好" chinese.wav --lang zh --ipa "↓ni ↓xau"
```

### Convert Text to IPA

```bash
# Convert English text to IPA
voirs kokoro text-to-ipa "Hello world" --lang en-us

# Save IPA to file
voirs kokoro text-to-ipa "Bonjour" --lang fr --output phonemes.txt
```

### Test Model Installation

```bash
# Test all languages
voirs kokoro test

# Test specific language
voirs kokoro test --lang es

# Test with custom model directory
voirs kokoro test --model-dir /path/to/kokoro-zh
```

### Batch Synthesis

Create a CSV file with columns: `text,language,voice,output_file`

**Example `batch.csv`:**
```csv
text,language,voice,output_file
Hello world,en-us,af_jessica,hello.wav
Hola mundo,es,ef_dora,hola.wav
Bonjour,fr,ff_siwis,bonjour.wav
Ciao,it,if_sara,ciao.wav
```

**Run batch synthesis:**
```bash
# Auto-IPA generation (default)
voirs kokoro batch batch.csv --output-dir ./output --jobs 4

# Manual IPA (if CSV contains IPA phonemes instead of text)
voirs kokoro batch batch_ipa.csv --output-dir ./output --manual-ipa --jobs 4
```

### Quick Reference

| Command | Description |
|---------|-------------|
| `kokoro download` | Download Kokoro model files from HuggingFace |
| `kokoro voices` | List all 54 available voices with filtering |
| `kokoro languages` | List 9 supported languages and voice counts |
| `kokoro synth` | Synthesize text to speech (main command) |
| `kokoro text-to-ipa` | Convert text to IPA phonemes |
| `kokoro test` | Test model installation and synthesis |
| `kokoro batch` | Batch process multiple texts from CSV |

## Supported Languages and Voices

### Voice Index Mapping

Voices are loaded from `voices-v1.0.bin` in **alphabetical order**:

| Index | Voice Name | Language | Gender |
|-------|-----------|----------|--------|
| 0-10 | af_* | American English | Female |
| 11-19 | am_* | American English | Male |
| 20-23 | bf_* | British English | Female |
| 24-27 | bm_* | British English | Male |
| 28 | ef_dora | Spanish | Female |
| 29-30 | em_* | Spanish | Male |
| 31 | ff_siwis | French | Female |
| 32-33 | hf_* | Hindi | Female |
| 34-35 | hm_* | Hindi | Male |
| 36 | if_sara | Italian | Female |
| 37 | im_nicola | Italian | Male |
| 38-41 | jf_* | Japanese | Female |
| 42 | jm_kumo | Japanese | Male |
| 43 | pf_dora | Portuguese | Female |
| 44-45 | pm_* | Portuguese | Male |
| 46-49 | zf_* | Chinese | Female |
| 50-53 | zm_* | Chinese | Male |

Full voice names are available in `voice_names.txt` (generated during NPZ loading).

## IPA Phoneme Generation

### Option 1: Automatic with eSpeak NG (Recommended for Most Languages)

For supported Kokoro languages (English, Spanish, French, Hindi, Italian, Portuguese), use **eSpeak NG** for automatic IPA generation:

```bash
# Install eSpeak NG
brew install espeak-ng  # macOS
sudo apt install espeak-ng  # Linux

# Generate IPA for any supported language
espeak-ng -v en-us -q --ipa "Your text here"
espeak-ng -v es -q --ipa "Tu texto aquí"
espeak-ng -v fr-fr -q --ipa "Votre texte ici"
```

**Advantages:**
- ✅ No Python dependencies
- ✅ Works offline
- ✅ Supports 100+ languages (though Kokoro only supports 9)
- ✅ Easy to integrate into Rust code

See [`kokoro_espeak_auto_demo.rs`](kokoro_espeak_auto_demo.rs) for a complete implementation.

### Option 2: Manual Generation (Japanese & Chinese)

For Japanese and Chinese, the **misaki** Python library provides better quality than eSpeak NG:

```python
# Japanese
import misaki.ja as ja
g2p = ja.JAG2P()
phonemes, _ = g2p("こんにちは")
print(phonemes)  # Output: koɲɲiʨiβa

# Chinese (with tone marks)
import misaki.zh as zh
g2p = zh.ZHG2P(version="1.0")
phonemes, _ = g2p("你好")
print(phonemes)  # Output: ni↓xau↓
```

**Important for Chinese:** Move tone marks (↓↘→) **before** syllables for better pronunciation:
- ❌ Wrong: `ni↓xau↓`
- ✅ Correct: `↓ni ↓xau`

### Southeast Asian Languages

**⚠️ Important Limitation:** While eSpeak NG supports many Southeast Asian languages, **Kokoro-82M only supports 9 languages** (listed above).

eSpeak NG can generate IPA for:
- 🇹🇭 Thai (`th`)
- 🇻🇳 Vietnamese (`vi`, `vi-vn-x-central`, `vi-vn-x-south`)
- 🇮🇩 Indonesian (`id`)
- 🇲🇾 Malay (`ms`)
- 🇲🇲 Myanmar/Burmese (`my`)

However, **these languages cannot be synthesized** with the current Kokoro model. To add support for these languages, you would need:

1. **Alternative TTS Models:**
   - VITS multilingual models trained on ASEAN languages
   - XTTSv2 (supports more languages)
   - Coqui TTS models

2. **Future VoiRS Updates:**
   - Integration with other ONNX TTS models
   - Custom voice training for Southeast Asian languages

**Current Status:**
- ✅ IPA generation: Supported via eSpeak NG
- ❌ Speech synthesis: **Not supported** (Kokoro limitation)

If you need Southeast Asian language TTS, consider:
- Using a different TTS model/service
- Contributing to VoiRS to add alternative model support
- Training your own VITS model for these languages

## Implementation Details

### Pure Rust NPZ Loading

VoiRS uses **numrs2** to load NumPy `.npz` files directly in Rust:

```rust
use numrs2::io::{list_npz_arrays, load_npz_array};

// List all voice arrays
let array_names = list_npz_arrays(file)?;

// Load specific voice
let voice_array = load_npz_array::<f32, _>(file, "af_jessica")?;

// Voice shape: (510, 1, 256) -> average to (256,)
let averaged_embedding = average_over_first_dimension(&voice_array);
```

**No Python scripts needed!** The original `voices-v1.0.bin` file is used directly.

### File Format Support

The `from_kokoro_files()` function automatically detects and loads from:

1. **`voices_averaged.bin`** (preferred) - Flat binary format, fastest
2. **`voices-v1.0.bin`** (fallback) - NumPy .npz format, pure Rust loading

### Silence Trimming

Different trimming strategies for different languages:

**Japanese:**
```rust
// Trim only trailing silence, keep natural beginning
model.synthesize_trim_end(phonemes, voice_idx, speed)?
```

**Chinese:**
```rust
// Disable trimming entirely, preserve model output
model.synthesize_with_options(phonemes, voice_idx, speed, false)?
```

**Other languages:**
```rust
// Default: trim both leading and trailing silence
model.synthesize(phonemes, voice_idx, speed)?
```

All outputs include **100ms padding** at the start when saved to WAV.

## Troubleshooting

### Model Loading Issues

**Error: "Failed to open NPZ file"**
- Ensure `voices-v1.0.bin` is downloaded to `$TMPDIR/voirs_models/kokoro-zh/`
- Check file size: should be ~27MB

**Error: "ONNX Runtime not found"**
- Install ONNX Runtime (see Prerequisites)
- On macOS: `brew install onnxruntime`

### Voice Quality Issues

**Chinese tone marks not working:**
- Move tone marks BEFORE syllables: `↓ni ↓xau` not `ni↓xau↓`

**Missing beginning of speech:**
- Disable silence trimming: `synthesize_with_options(..., false)`
- Add padding: 100ms is included automatically

**Unknown phoneme warnings:**
- Some IPA characters may not be in Kokoro's vocabulary
- Check `config.json` for supported phonemes
- Use eSpeak NG with `-q --ipa` to get compatible IPA

### Performance

**Slow first run:**
- ONNX model compilation happens on first load
- Subsequent runs are much faster (~0.5s synthesis time)

**Build time too long:**
- Use `--release` flag for optimized builds
- Consider using `sccache` for faster recompilation

## References

- [Kokoro-82M Model](https://huggingface.co/hexgrad/Kokoro-82M)
- [ONNX Runtime](https://onnxruntime.ai/)
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng)
- [misaki G2P library](https://github.com/reazon-research/misaki)
- [numrs2 - NumPy for Rust](https://github.com/cool-japan/numrs)

## License

These examples are part of the VoiRS project and are licensed under MIT OR Apache-2.0.

The Kokoro-82M model has its own license terms - please refer to the [model card](https://huggingface.co/hexgrad/Kokoro-82M) for details.
