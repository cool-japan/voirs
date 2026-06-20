# VoiRS Examples Changelog

All notable changes to the VoiRS examples are documented in this file.

Examples track the VoiRS main library version. Each examples release is aligned with a VoiRS library release, and the version numbers here mirror the library's [Semantic Versioning](https://semver.org/) scheme.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

_Placeholder for changes targeting the next release._

---

## [0.1.0] - 2026-06-20

### Added

#### Getting Started
- `hello_world.rs` - Minimal first synthesis example validating the environment
- `hello_world_tts.rs` - Hello-world variant using the TTS pipeline directly
- `hello_world_real_tts.rs` - Hello-world variant exercising the real TTS backend
- `basic_configuration.rs` - Essential configuration patterns and validation
- `simple_synthesis.rs` - Single-voice text-to-speech with file output and error handling

#### Core Synthesis
- `ssml_synthesis.rs` - Speech Synthesis Markup Language (SSML) prosody and voice control
- `batch_synthesis.rs` - Parallel batch processing with progress reporting
- `streaming_synthesis.rs` - Chunk-based streaming synthesis for real-time audio output
- `streaming_synthesis_optimization.rs` - Low-latency tuning for streaming pipelines
- `complete_pipeline.rs` - End-to-end synthesis pipeline with quality control
- `complete_voice_pipeline.rs` - Full multi-stage voice processing pipeline with monitoring

#### Emotion and Prosody
- `emotion_control_example.rs` - Emotion state control and prosody modification
- `emotion_control_example_fixed.rs` - Improved emotion control with real-time adaptation

#### Voice Cloning and Conversion
- `voice_cloning_example.rs` - Few-shot voice cloning with ethical safeguards
- `voice_cloning_example_fixed.rs` - Enhanced cloning with cross-lingual support
- `voice_conversion_example.rs` - Real-time voice-to-voice conversion and style transfer

#### Specialised Synthesis
- `singing_synthesis_example.rs` - Singing voice synthesis with MIDI and vibrato support
- `spatial_audio_example.rs` - 3D spatial audio with HRTF and VR/AR integration

#### Multilingual / Kokoro Demos
- `chinese_tts_demo.rs` - Chinese language TTS demonstration
- `kokoro_chinese_demo.rs` - Kokoro model Chinese demo
- `kokoro_japanese_demo.rs` - Kokoro model Japanese demo
- `kokoro_multilingual_demo.rs` - Kokoro multilingual synthesis across languages
- `kokoro_espeak_auto_demo.rs` - Kokoro with automatic eSpeak G2P selection

#### Speech Recognition
- `production_whisper_example.rs` - Production-ready Whisper ASR with forced alignment

#### Real-time and Interactive
- `realtime_voice_coach.rs` - Interactive real-time voice coaching with adaptive feedback

#### Performance and Optimisation
- `performance_benchmarking.rs` - RTF measurement, CPU utilisation, and throughput analysis
- `performance_optimization_techniques.rs` - CPU / memory / GPU / model / pipeline optimisations
- `low_latency_optimization.rs` - Sub-10 ms techniques for conversational and gaming scenarios
- `comprehensive_benchmark_suite.rs` - Multi-category benchmarking with statistical analysis
- `memory_profiling_analysis.rs` - Real-time memory leak detection and fragmentation analysis
- `profiling_optimization_example.rs` - Profiling-driven performance tuning workflow

#### Quality Assessment
- `audio_quality_assessment.rs` - SNR, PESQ, STOI, and MCD objective quality metrics
- `ab_testing_quality_comparison.rs` - Systematic A/B testing with statistical significance
- `batch_evaluation_comparison.rs` - Batch quality comparison across multiple synthesis runs

#### Production and Monitoring
- `production_pipeline_example.rs` - Enterprise deployment with load balancing and observability
- `production_monitoring_example.rs` - Real-time metrics, health checks, and alerting
- `debug_troubleshooting_example.rs` - Systematic debugging across pipeline stages
- `robust_error_handling_patterns.rs` - Circuit breakers, retry logic, and graceful degradation

#### Platform Integration
- `desktop_integration_example.rs` - Desktop GUI integration with worker pools and GPU acceleration
- `mobile_integration_example.rs` - iOS and Android native integration patterns
- `wasm_integration_example.rs` - WebAssembly / Web Audio API synthesis in the browser
- `cloud_deployment_example.rs` - Multi-cloud deployment with Kubernetes and auto-scaling
- `iot_edge_synthesis_example.rs` - Resource-constrained synthesis for edge and IoT devices
- `game_integration_example.rs` - Unity and Unreal Engine real-time voice integration
- `vr_ar_immersive_example.rs` - 6DOF spatial audio for VR/AR with HRTF and haptic sync

#### AI and Multimodal
- `ai_integration_example.rs` - LLM-powered conversational voice with personality simulation
- `multimodal_integration_example.rs` - Synchronised text, audio, and visual content generation
- `educational_tools_example.rs` - Language learning with pronunciation assessment and gamification
- `creative_applications_example.rs` - Music composition and artistic voice applications

#### Testing and Validation
- `comprehensive_testing_framework.rs` - Property-based testing and memory leak detection
- `examples_testing_framework.rs` - Multi-language example testing with JUnit XML output
- `testing_framework_example.rs` - Unit, integration, property, and regression test strategies
- `documentation_testing_example.rs` - Link checking, code-sample compilation, and version validation
- `test_audio_demo.rs` - Audio output validation for CI pipelines

#### Community and Reference
- `community_contributions_gallery.rs` - User-contributed example submission and rating system
- `use_case_gallery.rs` - Real-world application showcase with business metrics
- `best_practices_guide.rs` - Curated best practices with code examples and performance impact
- `faq_examples.rs` - Common questions and answers with runnable code solutions

---

[Unreleased]: https://github.com/cool-japan/voirs/compare/0.1.0...HEAD
[0.1.0]: https://github.com/cool-japan/voirs/releases/tag/0.1.0
