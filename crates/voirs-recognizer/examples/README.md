# VoiRS Recognizer Examples

This directory contains comprehensive examples demonstrating the full capabilities of VoiRS Recognizer, from basic audio analysis to advanced multilingual speech recognition.

## 🚀 Quick Start

### Zero-Config Quick Start
The fastest way to get started with speech recognition:

```bash
cargo run --example zero_config_quickstart
```

This example demonstrates:
- Zero configuration setup
- Default settings that "just work"
- Basic audio analysis
- Sample audio generation for testing
- Support for your own audio files

### Usage with Your Audio File
```bash
cargo run --example zero_config_quickstart -- your_audio.wav
```

## 📚 Learning Path: Step-by-Step Tutorials

**New users should start with the tutorial series for a structured learning experience:**

### 🎓 Tutorial Series (Recommended Learning Path)

1. **[Tutorial 01: Hello World](tutorial_01_hello_world.rs)** - Your first speech recognition
   ```bash
   cargo run --example tutorial_01_hello_world
   ```
   - Basic VoiRS setup and configuration
   - Understanding AudioBuffer and audio analysis
   - Interpreting analysis results
   - **Prerequisites**: None - start here!

2. **[Tutorial 02: Real Audio Files](tutorial_02_real_audio.rs)** - Working with actual audio
   ```bash
   cargo run --example tutorial_02_real_audio
   # Or with your own file:
   cargo run --example tutorial_02_real_audio -- /path/to/your/audio.wav
   ```
   - Loading different audio formats (WAV, MP3, FLAC, OGG)
   - Audio preprocessing and optimization
   - Quality analysis and recommendations
   - **Prerequisites**: Tutorial 01

3. **[Tutorial 03: Speech Recognition](tutorial_03_speech_recognition.rs)** - Basic ASR
   ```bash
   cargo run --example tutorial_03_speech_recognition --features="whisper-pure"
   ```
   - Configuring ASR models (Whisper, DeepSpeech, Wav2Vec2)
   - Performing speech recognition
   - Understanding confidence scores and results
   - **Prerequisites**: Tutorials 01-02

4. **[Tutorial 04: Real-time Processing](tutorial_04_realtime_processing.rs)** - Streaming ASR
   ```bash
   cargo run --example tutorial_04_realtime_processing --features="whisper-pure"
   ```
   - Streaming vs batch processing
   - Latency optimization techniques
   - Voice activity detection
   - Error handling and recovery
   - **Prerequisites**: Tutorials 01-03

5. **[Tutorial 05: Multi-language Support](tutorial_05_multilingual.rs)** - Global ASR
   ```bash
   cargo run --example tutorial_05_multilingual --features="whisper-pure"
   ```
   - 99+ language support
   - Automatic language detection
   - Cross-linguistic phoneme analysis
   - Language-specific optimizations
   - **Prerequisites**: Tutorials 01-04

## 🔧 Feature Examples

### Core Functionality
- **[Simple ASR Demo](simple_asr_demo.rs)** - Basic automatic speech recognition
  ```bash
  cargo run --example simple_asr_demo --features="whisper-pure"
  ```

- **[Basic Speech Recognition](basic_speech_recognition.rs)** - Recognition with configuration
  ```bash
  cargo run --example basic_speech_recognition --features="whisper-pure"
  ```

### Real-time Processing
- **[Streaming ASR](streaming_asr.rs)** - Real-time streaming recognition
  ```bash
  cargo run --example streaming_asr --features="whisper-pure"
  ```

- **[Advanced Real-time](advanced_realtime.rs)** - Advanced real-time processing
  ```bash
  cargo run --example advanced_realtime --features="whisper-pure"
  ```

- **[Real-time Processing](realtime_processing.rs)** - Real-time processing patterns
  ```bash
  cargo run --example realtime_processing --features="whisper-pure"
  ```

### Batch Processing
- **[Batch Transcription](batch_transcription.rs)** - Process multiple files efficiently
  ```bash
  cargo run --example batch_transcription --features="whisper-pure"
  ```

### Multi-language Support
- **[Multi-language Processing](multilanguage_processing.rs)** - Multi-language recognition
  ```bash
  cargo run --example multilanguage_processing --features="whisper-pure"
  ```

### Specialized Use Cases
- **[Emotion & Sentiment Recognition](emotion_sentiment_recognition.rs)** - Emotion and sentiment analysis
  ```bash
  cargo run --example emotion_sentiment_recognition --features="whisper-pure"
  ```

- **[Wake Word Training](wake_word_training.rs)** - Custom wake word detection
  ```bash
  cargo run --example wake_word_training --features="whisper-pure"
  ```

- **[Custom Model Integration](custom_model_integration.rs)** - Using custom models
  ```bash
  cargo run --example custom_model_integration --features="whisper-pure"
  ```

## 📊 Performance & Testing

### Benchmarking
- **[Accuracy Benchmarking](accuracy_benchmarking.rs)** - Performance benchmarking
  ```bash
  cargo run --example accuracy_benchmarking --features="whisper-pure"
  ```

- **[Accuracy Validation](accuracy_validation.rs)** - Accuracy validation
  ```bash
  cargo run --example accuracy_validation --features="whisper-pure"
  ```

### Stress Testing
- **[Stress Testing & Reliability](stress_testing_reliability.rs)** - System stress testing
  ```bash
  cargo run --example stress_testing_reliability --features="whisper-pure"
  ```

## 🎯 Feature Flags

Enable specific functionality through feature flags:

```toml
[dependencies]
voirs-recognizer = { 
    version = "0.1.0", 
    features = [
        "whisper-pure",   # Pure Rust Whisper implementation
        "whisper",        # OpenAI Whisper support
        "deepspeech",     # Mozilla DeepSpeech support  
        "wav2vec2",       # Facebook Wav2Vec2 support
        "forced-align",   # Basic forced alignment
        "mfa",            # Montreal Forced Alignment
        "all-models",     # Enable all ASR models
        "gpu",            # GPU acceleration support
    ]
}
```

## 🎵 Supported Audio Formats

VoiRS Recognizer supports multiple audio formats with automatic conversion:

| Format | Support | Quality | Use Case |
|--------|---------|---------|----------|
| **WAV** | ✅ Full | Highest | Production, development |
| **FLAC** | ✅ Full | Lossless | High-quality archival |
| **MP3** | ✅ Full | Lossy | Common web/mobile use |
| **OGG** | ✅ Full | Variable | Open-source preference |

### Recommended Settings
- **Sample Rate**: 16kHz (auto-converted if different)
- **Channels**: Mono (auto-converted from stereo)
- **Bit Depth**: 16-bit
- **Duration**: Up to 30 seconds per chunk for optimal memory usage

## 📋 Example Categories

### By Complexity Level
- **🟢 Beginner**: `zero_config_quickstart`, `tutorial_01_hello_world`, `tutorial_02_real_audio`
- **🟡 Intermediate**: `tutorial_03_speech_recognition`, `tutorial_04_realtime_processing`, `simple_asr_demo`
- **🔴 Advanced**: `tutorial_05_multilingual`, `advanced_realtime`, `emotion_sentiment_recognition`

### By Use Case
- **🎤 Basic Recognition**: `simple_asr_demo`, `basic_speech_recognition`
- **⚡ Real-time**: `streaming_asr`, `advanced_realtime`, `realtime_processing`
- **🌍 Multi-language**: `multilanguage_processing`, `tutorial_05_multilingual`
- **🎯 Specialized**: `emotion_sentiment_recognition`, `wake_word_training`, `custom_model_integration`
- **📊 Analysis**: `accuracy_benchmarking`, `accuracy_validation`, `stress_testing_reliability`

### By Feature
- **Audio Analysis**: `tutorial_01_hello_world`, `tutorial_02_real_audio`
- **Speech Recognition**: `tutorial_03_speech_recognition`, `simple_asr_demo`
- **Streaming**: `tutorial_04_realtime_processing`, `streaming_asr`
- **Batch Processing**: `batch_transcription`
- **Performance**: `accuracy_benchmarking`, `stress_testing_reliability`

## 🛠️ Development Tips

### Performance Optimization
- Use appropriate model sizes for your use case
- Enable GPU acceleration when available
- Configure chunk sizes based on latency requirements
- Implement voice activity detection for real-time applications

### Error Handling
- Always handle audio loading errors gracefully
- Implement fallback strategies for model failures
- Monitor memory usage in long-running applications
- Use appropriate timeouts for recognition operations

### Testing
- Test with various audio qualities and formats
- Validate performance on target hardware
- Test multilingual scenarios if applicable
- Benchmark against your accuracy requirements

## 🔗 Getting Help

- 📖 **Documentation**: [VoiRS Recognizer Docs](https://docs.voirs.ai)
- 🐛 **Issues**: [GitHub Issues](https://github.com/cool-japan/voirs/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/cool-japan/voirs/discussions)
- 📚 **Examples**: Start with the tutorial series above
- 🤝 **Community**: Join our Discord server for real-time help

## 🎓 Next Steps

After completing the examples:

1. **Integrate into your application**: Use the patterns learned in tutorials
2. **Optimize for your use case**: Adjust configurations based on your requirements
3. **Contribute**: Help improve VoiRS by contributing examples or fixes
4. **Explore advanced features**: Custom models, fine-tuning, specialized domains

## 💡 Pro Tips

1. **Start with tutorials**: Follow the numbered tutorial series for best learning experience
2. **Audio quality matters**: Use clear, high-quality recordings for best results
3. **Language specification**: Always specify the correct language for better accuracy
4. **Memory management**: Use streaming mode for large files to reduce memory usage
5. **GPU acceleration**: Enable GPU support for significant performance improvements
6. **Error handling**: Implement robust error handling for production applications
7. **Performance monitoring**: Use built-in performance validation tools
8. **Feature flags**: Only enable features you need to reduce binary size