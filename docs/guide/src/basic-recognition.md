# Basic Recognition

This guide covers the fundamental concepts of speech recognition with VoiRS Recognizer, from loading audio files to processing results.

## Overview

VoiRS Recognizer provides a powerful and flexible speech recognition engine with support for multiple ASR backends, real-time processing, and comprehensive audio analysis. This guide will walk you through the basic usage patterns.

## Loading Audio Files

VoiRS supports various audio formats including WAV, FLAC, MP3, and OGG. The universal audio loader handles format detection and conversion automatically.

```rust
use voirs_recognizer::prelude::*;

// Load audio with default settings
let audio = load_audio("speech.wav", &AudioLoadConfig::default()).await?;

// Load with specific configuration
let config = AudioLoadConfig {
    target_sample_rate: Some(16000),
    normalize_volume: true,
    remove_dc_offset: true,
    ..Default::default()
};
let audio = load_audio("speech.mp3", &config).await?;
```

## Basic Recognition

### Simple Recognition

The simplest way to recognize speech:

```rust
use voirs_recognizer::prelude::*;

#[tokio::main]
async fn main() -> Result<(), RecognitionError> {
    // Load audio file
    let audio = load_audio("speech.wav", &AudioLoadConfig::default()).await?;
    
    // Create ASR configuration
    let config = ASRConfig {
        language: Some(LanguageCode::EnUs),
        ..Default::default()
    };
    
    // Initialize ASR backend
    let mut asr = ASRBackend::new_whisper(config).await?;
    
    // Perform recognition
    let result = asr.recognize(&audio, None).await?;
    
    println!("Transcript: {}", result.text);
    println!("Confidence: {:.2}", result.confidence);
    
    Ok(())
}
```

### Recognition with Timestamps

Enable word-level timestamps for precise timing information:

```rust
let config = ASRConfig {
    language: Some(LanguageCode::EnUs),
    word_timestamps: true,
    ..Default::default()
};

let mut asr = ASRBackend::new_whisper(config).await?;
let result = asr.recognize(&audio, None).await?;

// Display word-level timestamps
for word in &result.word_timestamps {
    println!("{}: {:.2}s - {:.2}s", word.word, word.start_time, word.end_time);
}
```

### Confidence Scoring

Enable confidence scores for quality assessment:

```rust
let config = ASRConfig {
    language: Some(LanguageCode::EnUs),
    confidence_scores: true,
    confidence_threshold: 0.5,
    ..Default::default()
};

let mut asr = ASRBackend::new_whisper(config).await?;
let result = asr.recognize(&audio, None).await?;

println!("Overall confidence: {:.2}", result.confidence);

// Per-word confidence if available
for word in &result.word_timestamps {
    println!("{}: {:.2}", word.word, word.confidence);
}
```

## Multiple ASR Backends

VoiRS supports multiple ASR backends for different use cases:

### Whisper Backend

```rust
use voirs_recognizer::asr::{ASRBackend, WhisperModelSize};

// Different model sizes for speed/accuracy trade-off
let tiny_config = ASRConfig {
    model_size: Some(WhisperModelSize::Tiny),  // Fastest
    ..Default::default()
};

let base_config = ASRConfig {
    model_size: Some(WhisperModelSize::Base),  // Balanced
    ..Default::default()
};

let large_config = ASRConfig {
    model_size: Some(WhisperModelSize::Large), // Most accurate
    ..Default::default()
};
```

### Intelligent Fallback

Use multiple backends with intelligent fallback:

```rust
use voirs_recognizer::asr::{FallbackConfig, IntelligentASRFallback};

let fallback_config = FallbackConfig {
    primary_backend: ASRBackend::Whisper {
        model_size: WhisperModelSize::Base,
        model_path: None,
    },
    fallback_backends: vec![
        ASRBackend::Whisper {
            model_size: WhisperModelSize::Tiny,
            model_path: None,
        },
    ],
    quality_threshold: 0.7,
    adaptive_selection: true,
    ..Default::default()
};

let mut intelligent_asr = IntelligentASRFallback::new(fallback_config).await?;
let result = intelligent_asr.transcribe(&audio, None).await?;
```

## Language Support

### Automatic Language Detection

```rust
let config = ASRConfig {
    language: None,  // Auto-detect
    detect_language: true,
    ..Default::default()
};

let mut asr = ASRBackend::new_whisper(config).await?;
let result = asr.recognize(&audio, None).await?;

println!("Detected language: {:?}", result.language);
```

### Specific Language

```rust
let config = ASRConfig {
    language: Some(LanguageCode::EsEs),  // Spanish
    ..Default::default()
};
```

### Multi-language Processing

```rust
let languages = vec![
    LanguageCode::EnUs,
    LanguageCode::EsEs,
    LanguageCode::FrFr,
    LanguageCode::DeDE,
];

for lang in languages {
    let config = ASRConfig {
        language: Some(lang),
        ..Default::default()
    };
    
    let mut asr = ASRBackend::new_whisper(config).await?;
    let result = asr.recognize(&audio, None).await?;
    
    println!("{:?}: {}", lang, result.text);
}
```

## Error Handling

Robust error handling for production applications:

```rust
use voirs_recognizer::prelude::*;

async fn recognize_with_error_handling(audio_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let audio = match load_audio(audio_path, &AudioLoadConfig::default()).await {
        Ok(audio) => audio,
        Err(e) => {
            eprintln!("Failed to load audio: {}", e);
            return Err(e.into());
        }
    };
    
    let config = ASRConfig {
        language: Some(LanguageCode::EnUs),
        ..Default::default()
    };
    
    let mut asr = ASRBackend::new_whisper(config).await?;
    
    match asr.recognize(&audio, None).await {
        Ok(result) => {
            if result.confidence < 0.5 {
                eprintln!("Low confidence result: {:.2}", result.confidence);
            }
            Ok(result.text)
        }
        Err(RecognitionError::AudioError(e)) => {
            eprintln!("Audio processing error: {}", e);
            Err(e.into())
        }
        Err(RecognitionError::ModelError(e)) => {
            eprintln!("Model error: {}", e);
            Err(e.into())
        }
        Err(e) => {
            eprintln!("Recognition error: {}", e);
            Err(e.into())
        }
    }
}
```

## Performance Considerations

### Memory Management

```rust
// Configure memory limits
let config = ASRConfig {
    memory_limit_mb: Some(1024),  // 1GB limit
    ..Default::default()
};
```

### Processing Optimization

```rust
// Optimize for speed
let fast_config = ASRConfig {
    model_size: Some(WhisperModelSize::Tiny),
    beam_size: 1,  // Greedy decoding
    enable_preprocessing: false,
    ..Default::default()
};

// Optimize for accuracy
let accurate_config = ASRConfig {
    model_size: Some(WhisperModelSize::Large),
    beam_size: 5,  // Beam search
    enable_preprocessing: true,
    temperature: 0.0,
    ..Default::default()
};
```

## Best Practices

### 1. Choose the Right Model Size

```rust
// For real-time applications
let realtime_config = ASRConfig {
    model_size: Some(WhisperModelSize::Tiny),
    ..Default::default()
};

// For batch processing where accuracy is important
let batch_config = ASRConfig {
    model_size: Some(WhisperModelSize::Large),
    ..Default::default()
};
```

### 2. Handle Different Audio Qualities

```rust
// For high-quality studio recordings
let hq_config = ASRConfig {
    enable_preprocessing: false,
    noise_suppression: false,
    ..Default::default()
};

// For noisy environments
let noisy_config = ASRConfig {
    enable_preprocessing: true,
    noise_suppression: true,
    vad_enabled: true,
    ..Default::default()
};
```

### 3. Validate Results

```rust
let result = asr.recognize(&audio, None).await?;

// Check confidence
if result.confidence < 0.5 {
    println!("Warning: Low confidence result");
}

// Check for empty results
if result.text.trim().is_empty() {
    println!("Warning: Empty transcription");
}

// Check duration vs audio length
if let Some(duration) = result.processing_duration {
    let rtf = duration.as_secs_f32() / audio.duration();
    if rtf > 1.0 {
        println!("Warning: Processing too slow (RTF: {:.2})", rtf);
    }
}
```

## Common Use Cases

### Voice Commands

```rust
let config = ASRConfig {
    language: Some(LanguageCode::EnUs),
    confidence_threshold: 0.8,  // High confidence for commands
    sentence_segmentation: false,
    ..Default::default()
};
```

### Transcription Services

```rust
let config = ASRConfig {
    language: Some(LanguageCode::EnUs),
    word_timestamps: true,
    speaker_diarization: true,
    punctuation: true,
    ..Default::default()
};
```

### Real-time Captioning

```rust
let config = ASRConfig {
    language: Some(LanguageCode::EnUs),
    streaming: true,
    partial_results: true,
    low_latency: true,
    ..Default::default()
};
```

## Next Steps

- Learn about [Real-time Streaming](./streaming.md) for live audio processing
- Explore [Audio Analysis](./audio-analysis.md) for quality assessment
- Check out [Multi-language Support](./multi-language.md) for international applications
- Review [Performance Optimization](./performance.md) for production deployment