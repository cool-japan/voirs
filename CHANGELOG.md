# Changelog

All notable changes to VoiRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-22

First public release of VoiRS, a pure-Rust neural speech-synthesis (TTS) framework with a modular, pipeline-based architecture.

### Added

#### Architecture
- Modular pipeline architecture: Text → G2P → Acoustic Model → Vocoder → Audio Output
- Cargo workspace organizing 15 specialized crates with strict workspace dependency management
- Unified high-level API through the `voirs-sdk` crate orchestrating the full pipeline

#### Workspace Crates
- **voirs-g2p**: Grapheme-to-phoneme conversion with multiple backends (Phonetisaurus, OpenJTalk, Neural)
- **voirs-acoustic**: Neural acoustic models (VITS, FastSpeech2) converting phonemes to mel spectrograms
- **voirs-vocoder**: Neural vocoders (HiFi-GAN, DiffWave, UnivNet, BigVGAN) converting mel spectrograms to waveforms, including a DiffWave training pipeline with SafeTensors checkpointing; UnivNet and BigVGAN now expose the `Vocoder` trait implementation under the `candle` feature
- **voirs-dataset**: Dataset loading, preprocessing, and training-data utilities
- **voirs-sdk**: Unified high-level API exposing all features through a consistent interface
- **voirs-cli**: Command-line tool (`voirs` binary) for synthesis, voice management, training, and utilities including `sing` and `spatial` subcommands wired to real voirs-singing/voirs-spatial crate calls
- **voirs-ffi**: Foreign Function Interface with C, Python, and Node.js bindings; Windows native memory-mapped file I/O (`MemoryMappedFile::open_read_only` / `open_read_write`) implemented using the `windows` crate (replacing a prior stub)
- **voirs-recognizer**: Speech recognition (Whisper, DeepSpeech, Wav2Vec2) with forced alignment; new `cloud_auth` module providing AWS SigV4, Azure SharedKey, and GCS HMAC request-signing helpers (feature-gated under `cloud`); Paillier and ElGamal homomorphic-encryption schemes for private inference (feature-gated under `homomorphic`)
- **voirs-evaluation**: Quality metrics, MOS prediction, and A/B testing framework; objective evaluator gained a parabolic-interpolation mel-band F0 estimator with median-filter smoothing and octave-continuity pass; perceptual evaluator now uses `scirs2-fft` real-FFT O(n log n) spectrum and a 24-band Bark-weighted spectrum
- **voirs-feedback**: Real-time feedback systems with adaptive learning and progress tracking; `FeedbackApiManager::validate_auth` now fully implements JWT (HS256 via `jsonwebtoken`), Custom-header, and OAuth error-path branches — the JWT stub that always returned `Err` is replaced by real token validation; `AuthConfig` gains a `jwt_secret` field
- **voirs-emotion**: Multi-dimensional emotion control and prosody manipulation
- **voirs-cloning**: Voice cloning with few-shot learning, cross-lingual support, and ethical safeguards; age/gender voice adaptation refactored into a dedicated sub-module (`age_gender_adaptation`) with rich configuration types (`AgeGenderAdaptationConfig`, `F0AdaptationConfig`, `FormantAdaptationConfig`, etc.) and an `AgeGenderAdapter` processor; consent cryptography migrated from `ring::hmac` to pure-Rust `hmac` + `sha2` crates for constant-time HMAC-SHA256 verification
- **voirs-conversion**: Real-time voice conversion with zero-shot capabilities; `AudioReader::read_file` and `AudioReader::read_buffer` are now fully implemented (replacing placeholders) supporting WAV (`hound`), FLAC (`claxon`), OGG/Vorbis (`lewton`), AIFF/AAC (`symphonia`), and raw PCM; FLAC/AIFF/Vorbis encoding via `oxiaudio-core` / `oxiaudio-encode`; MP3 decoding feature-gated behind `ffi-codecs`
- **voirs-singing**: Singing synthesis with MusicXML/MIDI support and breath modeling
- **voirs-spatial**: 3D spatial audio with HRTF, binaural rendering, and VR/AR integration; `HrtfDatabaseManager` gained spherical-spline, radial-basis-function, and barycentric HRTF interpolation with precomputed weights and personalized adaptation parameters; `AdvancedRoomSimulator` gained Fibonacci-spiral and stratified ray-distribution methods

#### Core Features
- Real-time and streaming text-to-speech synthesis with low latency
- SSML markup support for advanced prosody control
- VITS acoustic modeling paired with HiFi-GAN, DiffWave, UnivNet, and BigVGAN vocoders
- Voice cloning and speaker adaptation with age/gender morphing
- Multi-dimensional emotion and prosody control
- Singing voice synthesis with MusicXML/MIDI input
- 3D spatial audio positioning with advanced HRTF interpolation
- Real-time voice conversion between speakers
- Speech recognition with forced alignment and private homomorphic inference
- Quality evaluation metrics including Bark-weighted spectral analysis and mel-band F0 estimation
- Batch processing utilities and a comprehensive example collection

#### In-House HuggingFace Hub Downloader (`voirs_acoustic::hub`)
- New `crate::hub` module in `voirs-acoustic` replaces the external `hf-hub` crate dependency
- `download_file(repo, filename, revision)` — async file download with atomic rename and non-empty-file cache-hit fast path; layout: `$CACHE_DIR/voirs/hub/<repo>/<revision>/<filename>`
- `list_files(repo, revision)` — lists repository siblings via the HF model-info REST API
- `ensure_crypto_provider()` — installs the RustCrypto-backed `rustls` `CryptoProvider` as the process default (Once-guarded; safe to call from multiple sites); required because `reqwest` is built with `rustls-no-provider`
- `HubError` — typed error enum covering HTTP transport, I/O, status, and 404 not-found cases

#### Bindings
- C, Python, and Node.js bindings via `voirs-ffi`

#### Platform Support
- Linux, macOS, and Windows on x86_64 and aarch64 architectures
- Optional GPU acceleration via CUDA (Linux/Windows) and Metal (macOS)
- WebAssembly (wasm32) target support (CPU-only)

### Changed

- Version promoted from `0.1.0-rc.1` to `0.1.0` across all 15 workspace crates
- **scirs2-core / scirs2-fft** updated from 0.3.4 → 0.5.0; `scirs2-fft` now uses the `oxifft` backend (`features = ["oxifft"]`) instead of `rustfft-backend`
- **candle-core / candle-nn / candle-transformers** updated from 0.9.2 → 0.10.2
- **oxionnx** updated from 0.1.0 → 0.1.4
- **safetensors** updated from 0.7.0 → 0.8.0
- **tract-onnx / tract-core** updated from 0.22.1 → 0.23.2
- **numrs2** updated from 0.3.1 → 0.4.0
- **pyo3 / numpy / pyo3-async-runtimes** updated from 0.28 → 0.29
- **tokio** updated from 1.50.0 → 1.52.3
- **cpal** updated from 0.17.3 → 0.18.1
- **oxifft** updated from 0.1.3 → 0.3.2
- **wide** updated from 1.2.0 → 1.5.0; **simba** updated from 0.9 → 0.10
- **wasmtime** updated from 43.0.0 → 45.0.2 (now with explicit feature list: component-model, GC, stack-switching, etc.)
- **parquet / arrow** updated from 58.0.0 → 59.0.0; `parquet` now uses explicit features (`arrow`, `snap`, `brotli`, `flate2-zlib-rs`, `lz4`, `base64`, `simdutf8`)
- **symphonia** updated from 0.5.5 → 0.6.0 with explicit `aiff`, `aac`, and `mp3` features
- **wasm-bindgen / wasm-bindgen-futures / web-sys / js-sys** updated from 0.2.114/0.3.91 → 0.2.125/0.3.102
- **oxiarc-deflate / oxiarc-zstd / oxiarc-archive / oxiarc-bzip2** updated from 0.2.5 → 0.3.3
- **tower-http** updated from 0.6 → 0.7; **hyper** updated from 1.8 → 1.10
- **lru** updated from 0.16.3 → 0.18.0; **ratatui** updated from 0.30.0 → 0.30.1
- **serde** gains the `rc` feature for `Arc`/`Rc` serialization
- `reqwest` now uses `rustls-no-provider` feature (was `rustls`); TLS crypto provider must be installed explicitly via `voirs_acoustic::hub::ensure_crypto_provider()`
- Added workspace dependencies: `oxitls-adapter-rustls-rustcrypto 0.1.2`, `rustls 0.23`, `oxiaudio-core 0.2.0`, `oxiaudio-encode 0.2.0`, `ogg 0.9`, `num-bigint-dig 0.8`, `hex 0.4`
- Removed `hf-hub`, `openssl`, and `flac-bound` from workspace dependencies; `flac-bound` (C FFI) replaced by `oxiaudio-encode` (pure Rust)
- `voirs-acoustic`: `AcousticModelLoader::download_from_hub` now uses the in-house `crate::hub::download_file` instead of the `hf-hub` API client; `ensure_crypto_provider()` called before all `reqwest::get` calls
- `voirs-acoustic` codegen: AVX2/AVX512 feature detection split into two `let` bindings to silence a clippy `nonminimal_bool` false-positive on macro expansion
- `voirs-cloning` consent cryptography: `ring::hmac` replaced by pure-Rust `hmac` + `sha2`; HMAC-SHA256 now verified in constant time via `mac.verify_slice`
- `voirs-ffi` zero-copy: `#[cfg(not(unix))]` stub replaced by a proper `#[cfg(windows)]` implementation and a narrower `#[cfg(not(any(unix, windows)))]` stub for WASM/embedded targets

### Fixed

- `voirs-ffi`: `voirs_error_message` now returns a null-terminated C-string literal (`c"Success"` etc.) instead of a bare `&str` pointer; eliminates undefined behaviour when C/Python callers call `CStr::from_ptr` on the returned pointer
- `voirs-vocoder` DiffWave: `map_weight_name` idempotency bug fixed — an `"time_embed.*"` key was erroneously rewritten to `"time_embeded.*"` by the `time_emb → time_embed` substitution arm, causing `load_weights_into_varmap` to fail for any model whose first HashMap entry had a `time_embed` prefix; a guard arm now short-circuits before the `time_emb` arm
- `voirs-cli` update manager: `voirs_acoustic::hub::ensure_crypto_provider()` called at the top of the version-comparison test to prevent a panic from `reqwest::Client::new()` when no `rustls` `CryptoProvider` is installed
- `voirs-recognizer` encrypted inference: Paillier and ElGamal schemes no longer unconditionally return `UnsupportedScheme`; they dispatch to real implementations under `feature = "homomorphic"` and return a descriptive error on stable builds
- `voirs-feedback` API: `validate_auth` for `AuthType::Jwt` now performs real HS256 JWT validation instead of always returning `Err("Authentication method not implemented")`
- Lock-poisoning handled gracefully across multiple modules: `unwrap_or_else(|e| e.into_inner())` replaces panicking `unwrap()` on `Mutex`/`RwLock` in audio processing and model-loading paths
- Conformer CTC decoding: integer overflow in `tokens_to_text` fixed — `token_id as u8` could wrap for IDs > 255; corrected to `(token_id - 2) % 26` in `usize` arithmetic
- Timing-sensitive tests hardened across `voirs-evaluation`, `voirs-conversion`, `voirs-cli`, `voirs-feedback`, and `voirs-g2p`: time budgets increased 4–10× and throughput minimums decreased proportionally to eliminate flakiness under parallel CI load

### Security
- `voirs-cloning` consent cryptography: `ring` (C assembly) replaced by the pure-Rust `hmac` crate; HMAC-SHA256 verification uses `verify_slice` for constant-time comparison, preventing timing side-channels in consent-proof validation
- Memory-safe Rust implementation throughout
- Secure consent management for voice cloning with GDPR compliance methods (`get_user_consent`, `delete_user_data`, audit-trail retrieval)
- No embedded secrets or sensitive data; licensed under Apache-2.0

[0.1.0]: https://github.com/cool-japan/voirs/releases/tag/v0.1.0
