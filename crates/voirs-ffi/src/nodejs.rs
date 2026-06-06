//! Modern Node.js bindings for VoiRS using NAPI-RS 3.x
//!
//! This module provides Node.js bindings with async/await support
//! for JavaScript/TypeScript applications.
//!
//! ## Key Changes from Previous Version
//! - Uses async/await with Promises instead of callbacks
//! - Compatible with napi-rs 3.x API
//! - Simplified method signatures
//! - Better error handling

#[cfg(feature = "nodejs")]
pub mod napi_bindings {
    use crate::{VoirsAudioFormat, VoirsQualityLevel};
    use napi::bindgen_prelude::*;
    use napi::{Error, Result as NapiResult, Status};
    use napi_derive::napi;
    use std::sync::Arc;
    use tokio::runtime::Runtime;
    use voirs_sdk::{
        audio::AudioBuffer as SdkAudioBuffer,
        error::VoirsError,
        types::{AudioFormat, LanguageCode, QualityLevel, SynthesisConfig},
        VoirsPipeline as SdkPipeline,
    };

    /// Audio buffer result for Node.js
    #[napi(object)]
    pub struct AudioBufferResult {
        pub samples: Buffer,
        pub sample_rate: u32,
        pub channels: u32,
        pub duration: f64,
    }

    /// Synthesis configuration options
    #[napi(object)]
    #[derive(Clone, Default)]
    pub struct SynthesisOptions {
        pub speaking_rate: Option<f64>,
        pub pitch_shift: Option<f64>,
        pub volume_gain: Option<f64>,
        pub sample_rate: Option<u32>,
        pub quality: Option<String>,
    }

    /// Pipeline configuration options
    #[napi(object)]
    pub struct PipelineOptions {
        pub use_gpu: Option<bool>,
        pub num_threads: Option<u32>,
        pub cache_dir: Option<String>,
    }

    /// Voice information
    #[napi(object)]
    pub struct VoiceInfo {
        pub id: String,
        pub name: String,
        pub language: String,
        pub gender: String,
    }

    /// Main VoiRS Pipeline for Node.js
    #[napi]
    pub struct VoirsPipeline {
        inner: Arc<SdkPipeline>,
        rt: Runtime,
    }

    #[napi]
    impl VoirsPipeline {
        /// Create a new VoiRS pipeline
        #[napi(constructor)]
        pub fn new(options: Option<PipelineOptions>) -> Self {
            // Install the pure-Rust rustls CryptoProvider before any TLS handshake
            // (reqwest is built with `rustls-no-provider`). Once-guarded; safe to repeat.
            voirs_acoustic::hub::ensure_crypto_provider();

            let rt = Runtime::new().expect("Failed to create runtime");

            let mut builder = SdkPipeline::builder();

            if let Some(opts) = options {
                if let Some(gpu) = opts.use_gpu {
                    builder = builder.with_gpu(gpu);
                }
                if let Some(threads) = opts.num_threads {
                    builder = builder.with_threads(threads as usize);
                }
            }

            let inner = rt
                .block_on(async { builder.build().await })
                .expect("Failed to build pipeline");

            Self {
                inner: Arc::new(inner),
                rt,
            }
        }

        /// Synthesize text to audio (async with Promise)
        #[napi]
        pub async fn synthesize(
            &self,
            text: String,
            options: Option<SynthesisOptions>,
        ) -> NapiResult<AudioBufferResult> {
            let pipeline = self.inner.clone();

            let audio = if let Some(opts) = options {
                let config = build_synthesis_config(opts)?;
                pipeline
                    .synthesize_with_config(&text, &config)
                    .await
                    .map_err(|e| {
                        Error::new(Status::GenericFailure, format!("Synthesis failed: {}", e))
                    })?
            } else {
                pipeline.synthesize(&text).await.map_err(|e| {
                    Error::new(Status::GenericFailure, format!("Synthesis failed: {}", e))
                })?
            };

            Ok(audio_buffer_to_result(audio))
        }

        /// Synthesize SSML to audio (async with Promise)
        #[napi]
        pub async fn synthesize_ssml(
            &self,
            ssml: String,
            options: Option<SynthesisOptions>,
        ) -> NapiResult<AudioBufferResult> {
            let pipeline = self.inner.clone();

            let audio = if let Some(opts) = options {
                let config = build_synthesis_config(opts)?;
                pipeline
                    .synthesize_with_config(&ssml, &config)
                    .await
                    .map_err(|e| {
                        Error::new(
                            Status::GenericFailure,
                            format!("SSML synthesis failed: {}", e),
                        )
                    })?
            } else {
                pipeline.synthesize(&ssml).await.map_err(|e| {
                    Error::new(
                        Status::GenericFailure,
                        format!("SSML synthesis failed: {}", e),
                    )
                })?
            };

            Ok(audio_buffer_to_result(audio))
        }

        /// Set the active voice
        #[napi]
        pub async fn set_voice(&self, voice_id: String) -> NapiResult<()> {
            self.inner.set_voice(&voice_id).await.map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Failed to set voice: {}", e),
                )
            })
        }

        /// Get available voices
        #[napi]
        pub async fn list_voices(&self) -> NapiResult<Vec<VoiceInfo>> {
            let voices = self.inner.list_voices().await.map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Failed to list voices: {}", e),
                )
            })?;

            Ok(voices
                .into_iter()
                .map(|v| VoiceInfo {
                    id: v.id.clone(),
                    name: v.name.clone(),
                    language: v.language.to_string(),
                    gender: format!("{:?}", v.characteristics.gender),
                })
                .collect())
        }

        /// Get current voice ID
        #[napi]
        pub async fn get_current_voice(&self) -> NapiResult<String> {
            self.inner
                .current_voice()
                .await
                .map(|v| v.id.clone())
                .ok_or_else(|| Error::new(Status::GenericFailure, "No voice selected"))
        }

        /// Check if GPU is available
        #[napi]
        pub fn is_gpu_available() -> bool {
            cfg!(feature = "gpu")
        }

        /// Get version information
        #[napi]
        pub fn version() -> String {
            env!("CARGO_PKG_VERSION").to_string()
        }
    }

    // Helper functions

    fn audio_buffer_to_result(audio: SdkAudioBuffer) -> AudioBufferResult {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();
        let channels = audio.channels();
        let duration = samples.len() as f64 / (sample_rate as f64 * channels as f64);

        // Convert f32 samples to i16 bytes
        let bytes: Vec<u8> = samples
            .iter()
            .flat_map(|&sample| {
                let scaled = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                scaled.to_le_bytes()
            })
            .collect();

        AudioBufferResult {
            samples: Buffer::from(bytes),
            sample_rate,
            channels,
            duration,
        }
    }

    fn build_synthesis_config(opts: SynthesisOptions) -> NapiResult<SynthesisConfig> {
        let mut config = SynthesisConfig::default();

        if let Some(rate) = opts.speaking_rate {
            config.speaking_rate = rate as f32;
        }

        if let Some(pitch) = opts.pitch_shift {
            config.pitch_shift = pitch as f32;
        }

        if let Some(volume) = opts.volume_gain {
            config.volume_gain = volume as f32;
        }

        if let Some(sr) = opts.sample_rate {
            config.sample_rate = sr;
        }

        if let Some(quality_str) = opts.quality {
            config.quality = match quality_str.as_str() {
                "low" => QualityLevel::Low,
                "medium" => QualityLevel::Medium,
                "high" => QualityLevel::High,
                "ultra" => QualityLevel::Ultra,
                _ => {
                    return Err(Error::new(
                        Status::InvalidArg,
                        format!("Invalid quality level: {}", quality_str),
                    ))
                }
            };
        }

        Ok(config)
    }

    // Recognition support (if feature enabled)
    #[cfg(feature = "recognition")]
    pub mod recognition {
        use super::*;
        use voirs_recognizer::{
            asr::WhisperModelSize,
            traits::{ASRModel as ASRModelTrait, AudioAnalyzer as AudioAnalyzerTrait},
        };

        /// Recognition result
        #[napi(object)]
        pub struct RecognitionResult {
            pub text: String,
            pub confidence: f64,
            pub language: String,
        }

        /// ASR Model for speech recognition
        #[napi]
        pub struct ASRModel {
            inner: Box<dyn ASRModelTrait>,
            rt: Runtime,
        }

        #[napi]
        impl ASRModel {
            /// Create a new Whisper-based ASR model
            ///
            /// Note: This method requires the 'whisper-pure' feature to be enabled.
            /// If the feature is not enabled, this will fail at compile time.
            #[napi(factory, js_name = "whisper")]
            #[cfg(feature = "whisper-pure")]
            pub fn new_whisper(model_size: Option<String>) -> Self {
                let rt = Runtime::new().expect("Failed to create runtime");

                let size_str = model_size.as_deref().unwrap_or("base");
                let size = match size_str {
                    "tiny" => WhisperModelSize::Tiny,
                    "base" => WhisperModelSize::Base,
                    "small" => WhisperModelSize::Small,
                    "medium" => WhisperModelSize::Medium,
                    "large" => WhisperModelSize::Large,
                    _ => WhisperModelSize::Base,
                };

                let model = rt
                    .block_on(async {
                        voirs_recognizer::asr::PureRustWhisper::new_from_model_size(size).await
                    })
                    .expect("Failed to load Whisper model");

                Self {
                    inner: Box::new(model),
                    rt,
                }
            }

            /// Recognize speech from audio buffer (async)
            #[napi]
            pub async fn recognize(
                &self,
                audio_samples: Buffer,
                sample_rate: u32,
            ) -> NapiResult<RecognitionResult> {
                // Convert bytes to f32 samples
                let bytes = audio_samples.as_ref();
                let mut f32_samples = Vec::with_capacity(bytes.len() / 2);

                for chunk in bytes.chunks(2) {
                    if chunk.len() == 2 {
                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                        f32_samples.push(sample);
                    }
                }

                let audio_buffer = SdkAudioBuffer::new(f32_samples, sample_rate, 1);
                let config = voirs_recognizer::traits::ASRConfig::default();

                let result = self
                    .inner
                    .transcribe(&audio_buffer, Some(&config))
                    .await
                    .map_err(|e| {
                        Error::new(Status::GenericFailure, format!("Recognition failed: {}", e))
                    })?;

                Ok(RecognitionResult {
                    text: result.text,
                    confidence: result.confidence as f64,
                    language: result.language.to_string(),
                })
            }
        }
    }
}
