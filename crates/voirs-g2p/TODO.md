# voirs-g2p Implementation TODO

> **Last Updated**: 2025-07-03  
> **Priority**: Critical Path Component  
> **Target**: Q3 2025 MVP

## 🎉 Recent Achievements (2025-07-03)

### ✅ Major Components Completed:
1. **Foundation Setup** - Complete lib.rs structure with all core types and traits
2. **Text Preprocessing Engine** - Full Unicode normalization, number expansion, and text processing
3. **Language Detection Framework** - Rule-based, statistical, and mixed-language detection
4. **Advanced Rule-based G2P Backend** - Context-aware phonological rules for 5+ languages
5. **Enhanced Phonetisaurus Backend** - FST-based G2P with improved pronunciation algorithms
6. **OpenJTalk Backend** - Japanese G2P with FFI bindings, mora timing, and pitch accent support
7. **Hybrid G2P Backend** - Multi-backend approach with confidence scoring and selection strategies
8. **Backend Registry System** - Dynamic backend loading, priority-based selection, and load balancing
9. **Configuration System** - TOML config files, environment variables, and runtime updates
10. **Comprehensive Test Suite** - 128 tests covering all implemented features
11. **Enhanced Phoneme Representation** - Complete phoneme struct with stress, syllable position, confidence, and duration
12. **Performance Optimization Module** - LRU caching, batch processing, SIMD acceleration framework
13. **CLI Tool** - Full-featured command-line interface with convert, file, batch, config, list, and benchmark commands

### 📊 Current Status:
- **Total Tests**: 128 (all passing)
- **Languages Supported**: English, German, French, Spanish, Japanese (basic), Chinese, Korean
- **Text Processing**: Unicode normalization, number expansion, abbreviations, currency, dates
- **G2P Backends**: Rule-based (advanced), Phonetisaurus (FST-based), OpenJTalk (Japanese), Hybrid (multi-backend), Registry (multi-backend), Dummy (testing)
- **Language Detection**: 3 detection methods with confidence scoring
- **Backend Management**: Dynamic registration, priority-based selection, load balancing
- **Configuration**: TOML files, environment variables, runtime updates
- **Performance**: Caching system, batch processing, SIMD acceleration framework
- **CLI Tool**: Full-featured command-line interface with multiple output formats

## 🎯 Critical Path (Week 1-4)

### Foundation Setup
- [x] **Create basic lib.rs structure** ✅ COMPLETED
  ```rust
  pub mod backends;
  pub mod models;
  pub mod rules;
  pub mod utils;
  pub mod preprocessing;
  pub mod detection;
  ```
- [x] **Define core types and traits** ✅ COMPLETED
  - [x] `Phoneme` struct with symbol, features, duration
  - [x] `LanguageCode` enum for supported languages (8 languages)
  - [x] `G2p` trait with async methods
  - [x] `G2pError` hierarchy with context
- [x] **Implement dummy backend for testing** ✅ COMPLETED
  - [x] `DummyG2p` that returns mock phonemes
  - [x] Enable end-to-end pipeline testing
  - [x] Basic error handling and logging

### Core Trait Implementation ✅ COMPLETED
- [x] **G2p trait definition** (src/lib.rs) ✅ COMPLETED
  ```rust
  #[async_trait]
  pub trait G2p: Send + Sync {
      async fn to_phonemes(&self, text: &str, lang: Option<LanguageCode>) -> Result<Vec<Phoneme>>;
      fn supported_languages(&self) -> Vec<LanguageCode>;
      fn metadata(&self) -> G2pMetadata;
  }
  ```
- [x] **Enhanced Phoneme representation** (src/lib.rs) ✅ COMPLETED
  ```rust
  #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
  pub struct Phoneme {
      pub symbol: String,           // IPA symbol
      pub stress: u8,               // 0=none, 1=primary, 2=secondary
      pub syllable_position: SyllablePosition,
      pub duration_ms: Option<f32>,
      pub confidence: f32,
      pub features: Option<HashMap<String, String>>,
  }
  ```

---

## 📋 Phase 1: Core Implementation (Weeks 5-12)

### Text Preprocessing Engine ✅ COMPLETED
- [x] **Unicode normalization** (src/preprocessing/unicode.rs) ✅ COMPLETED
  - [x] NFC/NFD normalization for consistent processing
  - [x] Script detection (Latin, Cyrillic, CJK, Arabic, Hiragana, Katakana, Hangul, etc.)
  - [x] Character filtering and cleaning
  - [x] Emoji and symbol handling
  - [x] RTL text detection
- [x] **Number expansion** (src/preprocessing/numbers.rs) ✅ COMPLETED
  - [x] Cardinal numbers: "123" → "one hundred twenty three"
  - [x] Ordinal numbers: "1st" → "first" (simplified)
  - [x] Decimal numbers: "3.14" → "three point one four"
  - [x] Negative numbers: "-5" → "negative five"
  - [x] Multi-language support (English, German, French, Spanish)
- [x] **Text normalization** (src/preprocessing/text.rs) ✅ COMPLETED
  - [x] Abbreviation expansion: "Dr." → "Doctor", "U.S.A." → "United States"
  - [x] Currency parsing: "$5.99" → "5.99 dollar"
  - [x] Date/time parsing: "Jan 1st, 2024" → "January 1st, 2024"
  - [x] URL handling: "https://example.com" → "H T T P S example dot com"
  - [x] Time formats: "3:30 PM" → "3 30 P M"
- [x] **Language-specific rules** ✅ COMPLETED
  - [x] English, German, French, Spanish abbreviations
  - [x] Multi-language month/day names
  - [x] Language-specific currency terms
  - [x] Configurable preprocessing pipeline

### Language Detection ✅ COMPLETED
- [x] **Rule-based detection** (src/detection/rules.rs) ✅ COMPLETED
  - [x] Character frequency analysis for multiple languages
  - [x] Common word detection with language indicators
  - [x] Unicode range-based script detection
  - [x] Function word pattern matching
- [x] **Statistical models** (src/detection/statistical.rs) ✅ COMPLETED
  - [x] Trigram language models for Latin scripts
  - [x] Character-based classification for CJK languages
  - [x] Confidence scoring and thresholds
  - [x] Multi-language support (EN, DE, FR, ES, JA, ZH, KO)
- [x] **Mixed language handling** (src/detection/mixed.rs) ✅ COMPLETED
  - [x] Sentence-level language switching
  - [x] Script change detection and segmentation
  - [x] Language distribution analysis
  - [x] Primary language determination

### Backend Infrastructure ✅ COMPLETED
- [x] **Backend registry** (src/backends/registry.rs) ✅ COMPLETED
  - [x] Dynamic backend loading and registration
  - [x] Priority-based backend selection
  - [x] Fallback chain configuration
  - [x] Load balancing for high throughput
- [x] **Configuration system** (src/config.rs) ✅ COMPLETED
  - [x] TOML configuration file parsing
  - [x] Environment variable overrides
  - [x] Runtime configuration updates
  - [x] Validation and error reporting

---

## 🔧 Backend Implementations

### Rule-based Backend ✅ COMPLETED
- [x] **Advanced phonological rules** (src/backends/rule_based.rs) ✅ COMPLETED
  - [x] Context-aware phonological rules with priority system
  - [x] Multi-language support (EN, DE, FR, ES, JA basic)
  - [x] Pattern matching with left/right context
  - [x] Comprehensive English phoneme mappings
  - [x] German umlaut and consonant cluster handling
  - [x] French accent and liaison rules
  - [x] Spanish regular pronunciation rules
- [x] **Stress assignment** ✅ COMPLETED
  - [x] Language-specific stress rules
  - [x] Pattern-based stress assignment
  - [x] Position-based stress calculation
- [x] **Integration with preprocessing** ✅ COMPLETED
  - [x] Text preprocessing pipeline integration
  - [x] Language detection integration
  - [x] Error handling and logging

### Phonetisaurus Backend ✅ COMPLETED
- [x] **FST model loading** (src/backends/neural.rs::phonetisaurus) ✅ COMPLETED
  - [x] OpenFST integration for model loading
  - [x] Lazy loading and model caching
  - [x] Memory mapping for large models
  - [x] Model validation and integrity checking
- [x] **Pronunciation generation** ✅ COMPLETED
  - [x] Enhanced FST-like traversal for phoneme generation
  - [x] Multiple pronunciation variants support
  - [x] Context-aware phoneme generation
  - [x] OOV (out-of-vocabulary) handling with fallbacks
- [x] **Model management** ✅ COMPLETED
  - [x] Model downloading from URLs
  - [x] Dictionary-based model building
  - [x] Compression and decompression support
  - [x] Language-specific model variants

### OpenJTalk Backend ✅ COMPLETED
- [x] **C library integration** (src/backends/openjtalk.rs) ✅ COMPLETED
  - [x] Safe FFI bindings to OpenJTalk with mock support
  - [x] Memory management and cleanup
  - [x] Error handling and translation
  - [x] Thread safety considerations
- [x] **Japanese text processing** ✅ COMPLETED
  - [x] Japanese phoneme normalization to IPA
  - [x] Katakana/Hiragana phoneme mapping
  - [x] Pitch accent feature extraction
  - [x] Mora timing calculation
- [x] **Dictionary management** ✅ COMPLETED
  - [x] Dictionary path configuration
  - [x] Pronunciation caching system
  - [x] Configurable cache size management
  - [x] Pronunciation customization options

### Hybrid G2P Backend ✅ COMPLETED
- [x] **Multi-backend integration** (src/backends/hybrid.rs) ✅ COMPLETED
  - [x] Rule-based and FST backend combination
  - [x] Configurable backend weights and thresholds
  - [x] Multiple selection strategies (FirstSuccess, HighestConfidence, WeightedEnsemble, MajorityVoting)
  - [x] Fallback chain configuration
- [x] **Confidence scoring and selection** ✅ COMPLETED
  - [x] Backend-specific confidence estimation
  - [x] Phoneme pattern analysis for quality assessment
  - [x] Weighted ensemble scoring
  - [x] Dynamic backend enabling/disabling
- [x] **Performance optimization** ✅ COMPLETED
  - [x] Pronunciation caching system
  - [x] Backend statistics and monitoring
  - [x] Configurable cache management
  - [x] Load balancing support

### Neural G2P Backend (Priority: Medium) - Future Work
- [ ] **LSTM model implementation** (src/backends/neural.rs)
  - [ ] Candle-based neural network inference
  - [ ] Encoder-decoder architecture
  - [ ] Attention mechanism for long sequences
  - [ ] Beam search for multiple candidates
- [ ] **Training infrastructure**
  - [ ] Dataset loading and preprocessing
  - [ ] Model training with Candle
  - [ ] Validation and checkpointing
  - [ ] Hyperparameter optimization
- [ ] **Multi-language support**
  - [ ] Language-specific embedding layers
  - [ ] Transfer learning between languages
  - [ ] Zero-shot pronunciation for unseen languages
  - [ ] Multilingual training strategies

---

## 🧪 Quality Assurance

### Testing Framework
- [ ] **Unit tests** (tests/unit/)
  - [ ] Phoneme representation correctness
  - [ ] Preprocessing accuracy (numbers, abbreviations)
  - [ ] Language detection precision/recall
  - [ ] Backend-specific functionality
- [ ] **Integration tests** (tests/integration/)
  - [ ] End-to-end text-to-phoneme conversion
  - [ ] Multi-backend fallback behavior
  - [ ] SSML processing accuracy
  - [ ] Configuration loading and validation
- [ ] **Benchmark suite** (benches/)
  - [ ] Latency measurements per backend
  - [ ] Throughput testing with batches
  - [ ] Memory usage profiling
  - [ ] Scalability testing

### Accuracy Validation
- [ ] **Reference datasets** (tests/data/)
  - [ ] CMU Pronunciation Dictionary test set
  - [ ] JVS corpus phoneme annotations
  - [ ] Common Voice pronunciation samples
  - [ ] Custom multi-language test cases
- [ ] **Accuracy metrics**
  - [ ] Phoneme-level accuracy (exact match)
  - [ ] Edit distance (Levenshtein) scoring
  - [ ] Stress pattern accuracy
  - [ ] Word-level accuracy aggregation
- [ ] **Regression testing**
  - [ ] Golden phoneme outputs for test sentences
  - [ ] Performance regression detection
  - [ ] Quality degradation alerts
  - [ ] Cross-platform consistency validation

### Performance Optimization ✅ COMPLETED
- [x] **Memory optimization** ✅ COMPLETED
  - [x] LRU cache with configurable size and TTL
  - [x] Lazy loading strategies in backend selection
  - [x] Memory-efficient phoneme representation
  - [x] Cache statistics and monitoring
- [x] **Speed optimization** ✅ COMPLETED
  - [x] SIMD acceleration framework for text processing
  - [x] Parallel processing for batch requests with async/await
  - [x] High-performance caching system for frequently used pronunciations
  - [x] Optimized batch processing utilities

---

## 📦 CLI Tool ✅ COMPLETED

### Command Interface ✅ COMPLETED
- [x] **Complete command set** (src/bin/voirs-g2p.rs) ✅ COMPLETED
  ```bash
  voirs-g2p convert "Hello world"           # Basic conversion
  voirs-g2p file input.txt --output phonemes.json  # File processing
  voirs-g2p batch --input texts.csv --output phonemes.json
  voirs-g2p benchmark "test text" --iterations 100
  voirs-g2p list backends|languages|models
  ```
- [x] **Configuration management** ✅ COMPLETED
  ```bash
  voirs-g2p config show                     # Show current config
  voirs-g2p config generate --output config.toml
  voirs-g2p config set key value
  voirs-g2p config get key
  ```

### Output Formats ✅ COMPLETED
- [x] **Format options** ✅ COMPLETED
  - [x] Plain text IPA symbols with optional metadata
  - [x] JSON with full phoneme metadata
  - [x] CSV for spreadsheet processing
  - [x] SSML phoneme annotations
- [x] **Advanced features** ✅ COMPLETED
  - [x] Stress pattern display (--show-stress)
  - [x] Syllable position information (--show-syllables)
  - [x] Confidence score display (--show-confidence)
  - [x] Multiple backend selection
  - [x] Batch processing with concurrency control
  - [x] Performance benchmarking

---

## 🔄 Advanced Features (Future)

### SSML Integration
- [ ] **SSML parsing** (src/ssml/)
  - [ ] XML parsing with proper error handling
  - [ ] Phoneme override support: `<phoneme alphabet="ipa" ph="təˈmeɪtoʊ">tomato</phoneme>`
  - [ ] Language switching: `<lang xml:lang="ja">こんにちは</lang>`
  - [ ] Emphasis and stress hints
- [ ] **Pronunciation control**
  - [ ] Custom pronunciation dictionaries
  - [ ] User-defined phoneme mappings
  - [ ] Context-sensitive pronunciations
  - [ ] Regional accent modifications

### Model Training Support
- [ ] **Training pipeline**
  - [ ] Dataset preparation tools
  - [ ] Model architecture configuration
  - [ ] Training progress monitoring
  - [ ] Model evaluation and validation
- [ ] **Transfer learning**
  - [ ] Pre-trained model adaptation
  - [ ] Few-shot learning for new languages
  - [ ] Domain adaptation techniques
  - [ ] Pronunciation customization

### Advanced Preprocessing
- [ ] **Context awareness**
  - [ ] Part-of-speech tagging integration
  - [ ] Named entity recognition
  - [ ] Semantic context for disambiguation
  - [ ] Pronunciation variant selection
- [ ] **Quality filtering**
  - [ ] Input validation and sanitization
  - [ ] Noise detection and removal
  - [ ] Encoding detection and conversion
  - [ ] Malformed input handling

---

## 📊 Performance Targets

### Latency Requirements
- **Single sentence**: <1ms (target: 0.5ms)
- **Batch processing**: >1000 sentences/second
- **Cold start**: <100ms model loading
- **Memory footprint**: <100MB per backend

### Accuracy Targets
- **English (CMU dict)**: >95% phoneme accuracy
- **Japanese (JVS)**: >90% mora accuracy  
- **Multilingual**: >85% phoneme accuracy
- **OOV handling**: >80% pronunciation quality

### Resource Usage
- **CPU utilization**: <50% single core for real-time
- **Memory growth**: <1MB per 1000 processed sentences
- **Model size**: <100MB compressed per language
- **Startup time**: <200ms including model loading

---

## 🚀 Implementation Schedule

### Week 1-2: Foundation
- [ ] Project structure and basic types
- [ ] Dummy backend for testing
- [ ] Core trait definitions
- [ ] Basic preprocessing functions

### Week 3-4: Text Processing
- [ ] Unicode normalization
- [ ] Number and abbreviation expansion
- [ ] Language detection framework
- [ ] Configuration system

### Week 5-8: Phonetisaurus Backend
- [ ] FST model loading
- [ ] Pronunciation generation
- [ ] Model management
- [ ] English language support

### Week 9-12: OpenJTalk Backend
- [ ] C library integration
- [ ] Japanese text processing
- [ ] Dictionary management
- [ ] Accuracy validation

### Week 13-16: Neural Backend
- [ ] LSTM implementation
- [ ] Training infrastructure
- [ ] Multi-language support
- [ ] Performance optimization

---

## 📝 Development Notes

### Critical Dependencies
- `unicode-normalization` for text preprocessing
- `tokio` for async operations
- `serde` for serialization
- `thiserror` for error handling
- `tracing` for structured logging

### Architecture Decisions
- Async-first design for non-blocking I/O
- Plugin-based backend architecture
- Language-agnostic core with language-specific plugins
- Configuration-driven behavior with sensible defaults

### Quality Gates
- All code must compile without warnings
- >90% test coverage for critical paths
- Performance benchmarks must pass regression tests
- Documentation for all public APIs

This TODO list should be updated as implementation progresses and priorities shift based on user feedback and integration requirements.