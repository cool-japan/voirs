# voirs-dataset

[![Crates.io](https://img.shields.io/crates/v/voirs-dataset.svg)](https://crates.io/crates/voirs-dataset)
[![Documentation](https://docs.rs/voirs-dataset/badge.svg)](https://docs.rs/voirs-dataset)

**Dataset utilities and processing for VoiRS speech synthesis training and evaluation.**

This crate provides comprehensive tools for loading, processing, and managing speech datasets used in VoiRS model training and evaluation. It supports popular datasets like LJSpeech and JVS, as well as custom dataset creation and preprocessing pipelines.

## Features

- **Multi-dataset Support**: LJSpeech, JVS, VCTK, LibriTTS, and custom datasets
- **Audio Processing**: Normalization, resampling, trimming, and quality validation
- **Data Augmentation**: Speed perturbation, pitch shifting, noise injection
- **Parallel Processing**: Multi-threaded audio processing with Rayon
- **Manifest Generation**: JSON, CSV, and Parquet metadata formats
- **Quality Control**: Automatic filtering of low-quality samples
- **Streaming**: Memory-efficient processing of large datasets

## Quick Start

```rust
use voirs_dataset::{LJSpeechDataset, Dataset, DatasetLoader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Download and load LJSpeech dataset
    let dataset = LJSpeechDataset::download_and_load("./data/ljspeech").await?;
    
    // Get dataset statistics
    println!("Dataset: {} samples", dataset.len());
    println!("Total duration: {:.1} hours", dataset.total_duration() / 3600.0);
    
    // Iterate over samples
    for sample in dataset.iter().take(5) {
        println!("Text: {}", sample.text);
        println!("Audio: {:.2}s @ {}Hz", sample.audio.duration(), sample.audio.sample_rate());
    }
    
    Ok(())
}
```

## Supported Datasets

| Dataset | Language | Speakers | Hours | Domain | Status |
|---------|----------|----------|-------|--------|--------|
| LJSpeech | English | 1 | 24h | Audiobooks | ✅ Stable |
| JVS | Japanese | 100 | 30h | Various | ✅ Stable |
| VCTK | English | 110 | 44h | News | 🚧 Beta |
| LibriTTS | English | 2,456 | 585h | Audiobooks | 🚧 Beta |
| Common Voice | Multi | 50,000+ | 20,000h+ | Read speech | 📋 Planned |
| JSUT | Japanese | 1 | 10h | News | 📋 Planned |

## Architecture

```
Raw Dataset → Download → Extract → Validate → Process → Augment → Export
     ↓           ↓        ↓        ↓        ↓        ↓        ↓
  Archive     Metadata   Audio    Quality  Enhance  Variants  Manifest
```

### Core Components

1. **Dataset Loaders**
   - Automatic download and extraction
   - Metadata parsing and validation
   - Audio file discovery and loading

2. **Audio Processing**
   - Format conversion and normalization
   - Resampling and channel management
   - Silence detection and trimming

3. **Data Augmentation**
   - Speed and pitch perturbation
   - Noise injection and mixing
   - Room impulse response simulation

4. **Quality Control**
   - SNR and clipping detection
   - Duration and frequency validation
   - Manual review and annotation tools

## API Reference

### Core Trait

```rust
#[async_trait]
pub trait Dataset: Send + Sync {
    type Sample: DatasetSample;
    
    /// Get number of samples in dataset
    fn len(&self) -> usize;
    
    /// Get sample by index
    fn get(&self, index: usize) -> Result<Self::Sample>;
    
    /// Iterate over all samples
    fn iter(&self) -> impl Iterator<Item = Self::Sample>;
    
    /// Get dataset metadata
    fn metadata(&self) -> DatasetMetadata;
    
    /// Split dataset into train/validation/test
    fn split(&self, config: SplitConfig) -> Result<DatasetSplit<Self>>;
}
```

### Dataset Sample

```rust
pub struct DatasetSample {
    /// Unique sample identifier
    pub id: String,
    
    /// Text transcript
    pub text: String,
    
    /// Audio data
    pub audio: AudioData,
    
    /// Speaker information
    pub speaker: Option<SpeakerInfo>,
    
    /// Language and locale
    pub language: LanguageCode,
    
    /// Quality metrics
    pub quality: QualityMetrics,
    
    /// Additional metadata
    pub metadata: HashMap<String, Value>,
}
```

### Audio Data

```rust
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Audio samples as f32 values
    samples: Vec<f32>,
    
    /// Sample rate in Hz
    sample_rate: u32,
    
    /// Number of channels
    channels: u32,
    
    /// Original file path
    path: Option<PathBuf>,
}

impl AudioData {
    /// Get audio duration in seconds
    pub fn duration(&self) -> f32;
    
    /// Resample to target sample rate
    pub fn resample(&self, target_rate: u32) -> Result<AudioData>;
    
    /// Normalize audio amplitude
    pub fn normalize(&self, method: NormalizationMethod) -> AudioData;
    
    /// Trim silence from beginning and end
    pub fn trim_silence(&self, threshold: f32) -> AudioData;
    
    /// Convert to mel spectrogram
    pub fn to_mel(&self, config: &MelConfig) -> Result<MelSpectrogram>;
}
```

## Usage Examples

### Loading Standard Datasets

```rust
use voirs_dataset::{LJSpeechDataset, JVSDataset, VCTKDataset};

// Load LJSpeech dataset
let ljspeech = LJSpeechDataset::from_path("./data/ljspeech")?;

// Load JVS dataset
let jvs = JVSDataset::from_path("./data/jvs")?;

// Load VCTK dataset
let vctk = VCTKDataset::from_path("./data/vctk")?;
```

### Custom Dataset Creation

```rust
use voirs_dataset::{CustomDataset, DatasetBuilder, AudioFormat};

let dataset = DatasetBuilder::new()
    .name("MyDataset")
    .audio_dir("./audio/")
    .transcript_file("./transcripts.txt")
    .audio_format(AudioFormat::Wav)
    .sample_rate(22050)
    .build()?;

// Save dataset manifest
dataset.save_manifest("./my_dataset.json")?;
```

### Data Processing Pipeline

```rust
use voirs_dataset::{ProcessingPipeline, AudioProcessor, AugmentationConfig};

let processor = ProcessingPipeline::builder()
    .add_step(AudioProcessor::Resample { target_rate: 22050 })
    .add_step(AudioProcessor::Normalize { method: NormalizationMethod::RMS })
    .add_step(AudioProcessor::TrimSilence { threshold: -40.0 })
    .add_step(AudioProcessor::ValidateQuality { min_snr: 20.0 })
    .build();

let processed_dataset = processor.process(&dataset).await?;
```

### Data Augmentation

```rust
use voirs_dataset::{AugmentationPipeline, Augmentation};

let augmenter = AugmentationPipeline::builder()
    .add_augmentation(Augmentation::SpeedPerturbation {
        rates: vec![0.9, 1.0, 1.1],
        probability: 0.3,
    })
    .add_augmentation(Augmentation::PitchShift {
        semitones: vec![-1.0, 0.0, 1.0],
        probability: 0.2,
    })
    .add_augmentation(Augmentation::NoiseInjection {
        snr_range: (20.0, 40.0),
        noise_types: vec![NoiseType::White, NoiseType::Pink],
        probability: 0.1,
    })
    .build();

let augmented_dataset = augmenter.augment(&dataset).await?;
```

### Dataset Splitting

```rust
use voirs_dataset::{SplitConfig, SplitStrategy};

let split_config = SplitConfig {
    train_ratio: 0.8,
    val_ratio: 0.1,
    test_ratio: 0.1,
    strategy: SplitStrategy::Random { seed: 42 },
    stratify_by: Some("speaker".to_string()),
};

let DatasetSplit { train, val, test } = dataset.split(split_config)?;

println!("Train: {} samples", train.len());
println!("Val: {} samples", val.len());
println!("Test: {} samples", test.len());
```

### Parallel Processing

```rust
use voirs_dataset::{ParallelProcessor, ProcessingConfig};

let config = ProcessingConfig {
    num_workers: 8,
    chunk_size: 100,
    memory_limit: 4 * 1024 * 1024 * 1024, // 4GB
    progress_callback: Some(Box::new(|progress| {
        println!("Progress: {:.1}%", progress * 100.0);
    })),
};

let processor = ParallelProcessor::new(config);
let results = processor.process_dataset(&dataset).await?;
```

### Quality Analysis

```rust
use voirs_dataset::{QualityAnalyzer, QualityReport};

let analyzer = QualityAnalyzer::new();
let report = analyzer.analyze_dataset(&dataset).await?;

println!("Quality Report:");
println!("  Average SNR: {:.1} dB", report.avg_snr);
println!("  Clipped samples: {}", report.clipped_count);
println!("  Duration range: {:.1}s - {:.1}s", report.min_duration, report.max_duration);
println!("  Recommended filters: {:?}", report.recommended_filters);
```

### Streaming Large Datasets

```rust
use voirs_dataset::{StreamingDataset, StreamingConfig};
use futures::StreamExt;

let config = StreamingConfig {
    chunk_size: 1000,
    prefetch_count: 3,
    shuffle: true,
    shuffle_buffer_size: 10000,
};

let streaming_dataset = StreamingDataset::new(&dataset, config);
let mut stream = streaming_dataset.stream();

while let Some(batch) = stream.next().await {
    let samples = batch?;
    // Process batch of samples
    process_batch(samples).await?;
}
```

### Export to Different Formats

```rust
use voirs_dataset::{DatasetExporter, ExportFormat, ExportConfig};

let exporter = DatasetExporter::new();

// Export to HuggingFace Datasets format
exporter.export(&dataset, ExportConfig {
    format: ExportFormat::HuggingFace,
    output_path: "./hf_dataset/".into(),
    include_audio: true,
    audio_format: AudioFormat::Flac,
}).await?;

// Export to PyTorch format
exporter.export(&dataset, ExportConfig {
    format: ExportFormat::PyTorch,
    output_path: "./torch_dataset/".into(),
    include_audio: false, // Just metadata
    manifest_format: ManifestFormat::JSON,
}).await?;
```

## Performance

### Processing Speed (Intel i7-12700K, 8 workers)

| Operation | Speed | Memory Usage | Notes |
|-----------|-------|--------------|-------|
| Audio loading | 450 files/s | 2GB | WAV files, 22kHz |
| Resampling | 280 files/s | 1.5GB | 48kHz → 22kHz |
| Mel extraction | 220 files/s | 2.5GB | 80 mel bins |
| Augmentation | 180 files/s | 3GB | 3x augmentation |
| Quality analysis | 320 files/s | 1GB | SNR, clipping, etc. |

### Memory Efficiency
- **Streaming**: Constant memory usage regardless of dataset size
- **Chunked processing**: Configurable memory limits
- **Lazy loading**: Audio files loaded on demand
- **Memory pooling**: Reuse of audio buffers

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
voirs-dataset = "0.1"

# Enable specific features
[dependencies.voirs-dataset]
version = "0.1"
features = ["augmentation", "streaming", "export"]
```

### Feature Flags

- `augmentation`: Enable data augmentation pipeline
- `streaming`: Enable streaming dataset support
- `export`: Enable dataset export to various formats
- `analysis`: Enable quality analysis tools
- `visualization`: Enable dataset visualization
- `pandrs`: Integration with PandRS for ETL operations

### System Dependencies

**Audio processing:**
```bash
# Ubuntu/Debian
sudo apt-get install libsndfile1-dev

# macOS
brew install libsndfile
```

**Optional dependencies:**
```bash
# For advanced audio processing
sudo apt-get install libfftw3-dev

# For dataset downloading
sudo apt-get install curl wget
```

## Configuration

Create `~/.voirs/dataset.toml`:

```toml
[default]
cache_dir = "~/.voirs/cache/datasets"
num_workers = 8
memory_limit = "4GB"

[download]
auto_download = true
verify_checksums = true
mirror_urls = [
    "https://github.com/keithito/tacotron/releases/download/v0.3/",
    "https://mirror.voirs.org/datasets/"
]

[audio]
default_sample_rate = 22050
default_channels = 1
normalization_method = "rms"
trim_silence = true
silence_threshold = -40.0

[augmentation]
speed_perturbation_rates = [0.9, 1.0, 1.1]
pitch_shift_range = [-2.0, 2.0]
noise_snr_range = [20.0, 40.0]
room_impulse_responses = "~/.voirs/rir/"

[quality]
min_snr = 15.0
max_clipping_ratio = 0.01
min_duration = 0.5
max_duration = 20.0

[export]
default_format = "json"
include_checksums = true
compression = "gzip"
```

## Dataset Creation Guide

### Preparing Custom Dataset

```rust
use voirs_dataset::{DatasetCreator, CreationConfig, Validator};

let config = CreationConfig {
    name: "MyCustomDataset".to_string(),
    language: LanguageCode::EnUs,
    description: "Custom speech dataset for training".to_string(),
    audio_dir: "./audio/".into(),
    transcript_file: "./transcripts.txt".into(),
    speaker_info: Some("./speakers.json".into()),
    license: "CC-BY-4.0".to_string(),
};

let creator = DatasetCreator::new(config);

// Validate input data
let validation = creator.validate().await?;
if !validation.is_valid() {
    println!("Validation errors: {:?}", validation.errors);
    return Err("Dataset validation failed".into());
}

// Create dataset
let dataset = creator.create().await?;

// Generate quality report
let report = dataset.quality_report().await?;
println!("Dataset created successfully!");
println!("Quality score: {:.2}/10", report.overall_score);
```

### Audio File Organization

```
dataset/
├── audio/
│   ├── speaker1/
│   │   ├── 001.wav
│   │   ├── 002.wav
│   │   └── ...
│   └── speaker2/
│       ├── 001.wav
│       └── ...
├── transcripts.txt
├── speakers.json
└── metadata.json
```

### Transcript Format

```
# transcripts.txt
speaker1/001.wav|Hello, this is the first sentence.
speaker1/002.wav|This is the second sentence with punctuation!
speaker2/001.wav|Different speaker saying something else.
```

### Speaker Information

```json
{
  "speaker1": {
    "name": "John Doe",
    "gender": "male",
    "age": 35,
    "accent": "General American",
    "native_language": "en-US"
  },
  "speaker2": {
    "name": "Jane Smith", 
    "gender": "female",
    "age": 28,
    "accent": "British RP",
    "native_language": "en-GB"
  }
}
```

## Error Handling

```rust
use voirs_dataset::{DatasetError, ErrorKind};

match dataset.get(index) {
    Ok(sample) => process_sample(sample),
    Err(DatasetError { kind, context, .. }) => match kind {
        ErrorKind::FileNotFound => {
            eprintln!("Audio file missing: {}", context);
        }
        ErrorKind::InvalidAudio => {
            eprintln!("Corrupted audio file: {}", context);
        }
        ErrorKind::TranscriptMismatch => {
            eprintln!("Transcript doesn't match audio: {}", context);
        }
        ErrorKind::QualityTooLow => {
            eprintln!("Audio quality below threshold: {}", context);
        }
        _ => eprintln!("Other error: {}", context),
    }
}
```

## Advanced Features

### Custom Data Loaders

```rust
use voirs_dataset::{Dataset, DatasetSample, DatasetMetadata};

pub struct MyCustomDataset {
    samples: Vec<DatasetSample>,
    metadata: DatasetMetadata,
}

#[async_trait]
impl Dataset for MyCustomDataset {
    type Sample = DatasetSample;
    
    fn len(&self) -> usize {
        self.samples.len()
    }
    
    fn get(&self, index: usize) -> Result<Self::Sample> {
        self.samples.get(index)
            .cloned()
            .ok_or_else(|| DatasetError::IndexOutOfBounds(index))
    }
    
    fn iter(&self) -> impl Iterator<Item = Self::Sample> {
        self.samples.iter().cloned()
    }
    
    fn metadata(&self) -> DatasetMetadata {
        self.metadata.clone()
    }
    
    fn split(&self, config: SplitConfig) -> Result<DatasetSplit<Self>> {
        // Custom splitting logic
        todo!()
    }
}
```

### Integration with ML Frameworks

```rust
#[cfg(feature = "torch")]
use voirs_dataset::{TorchDataset, TorchDataLoader};

let torch_dataset = TorchDataset::from_dataset(&dataset)?;
let dataloader = TorchDataLoader::new(torch_dataset)
    .batch_size(32)
    .shuffle(true)
    .num_workers(4)
    .build()?;

for batch in dataloader {
    let (audio, text, lengths) = batch?;
    // Train model with batch
}
```

## Contributing

We welcome contributions! Please see the [main repository](https://github.com/cool-japan/voirs) for contribution guidelines.

### Development Setup

```bash
git clone https://github.com/cool-japan/voirs.git
cd voirs/crates/voirs-dataset

# Install development dependencies
cargo install cargo-nextest

# Download test datasets
./scripts/download_test_data.sh

# Run tests
cargo nextest run

# Run benchmarks
cargo bench

# Check code quality
cargo clippy -- -D warnings
cargo fmt --check
```

### Adding New Datasets

1. Implement the `Dataset` trait for your dataset
2. Add dataset-specific loading and processing logic
3. Create comprehensive tests with sample data
4. Add documentation and usage examples
5. Update the supported datasets table

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.