//! Node.js bindings for VoiRS using NAPI-RS.
//!
//! This module provides Node.js bindings that expose VoiRS functionality
//! to JavaScript/TypeScript applications with full async support.

#[cfg(feature = "nodejs")]
pub mod napi_bindings {
    use napi::{
        bindgen_prelude::*,
        threadsafe_function::{
            ThreadsafeFunction, ThreadsafeFunctionCallMode, 
        },
        JsFunction, Result as NapiResult,
    };
    use napi_derive::napi;
    use std::sync::Arc;
    use tokio::runtime::Runtime;
    use voirs::{VoirsPipeline as SdkPipeline, AudioBuffer, error::VoirsError};
    use crate::{VoirsQualityLevel, VoirsAudioFormat};

    /// Node.js VoiRS Pipeline wrapper
    #[napi]
    pub struct VoirsPipeline {
        inner: Arc<SdkPipeline>,
        rt: Runtime,
    }

    /// Configuration options for creating a VoiRS pipeline
    #[napi(object)]
    pub struct PipelineOptions {
        pub use_gpu: Option<bool>,
        pub num_threads: Option<u32>,
        pub cache_dir: Option<String>,
        pub device: Option<String>,
    }

    /// Synthesis configuration options
    #[napi(object)]
    pub struct SynthesisOptions {
        pub speaking_rate: Option<f64>,
        pub pitch_shift: Option<f64>,
        pub volume_gain: Option<f64>,
        pub enable_enhancement: Option<bool>,
        pub output_format: Option<String>,
        pub sample_rate: Option<u32>,
        pub quality: Option<String>,
    }

    /// Audio buffer result
    #[napi(object)]
    pub struct AudioBufferResult {
        pub samples: Buffer,
        pub sample_rate: u32,
        pub channels: u32,
        pub duration: f64,
    }

    /// Voice information
    #[napi(object)]
    pub struct VoiceInfo {
        pub id: String,
        pub name: String,
        pub language: String,
        pub quality: String,
        pub is_available: bool,
    }

    /// Progress callback function type
    pub type ProgressCallback = ThreadsafeFunction<f64, ErrorStrategy::CalleeHandled>;

    /// Error callback function type
    pub type ErrorCallback = ThreadsafeFunction<String, ErrorStrategy::CalleeHandled>;

    /// Completion callback function type
    pub type CompletionCallback = ThreadsafeFunction<AudioBufferResult, ErrorStrategy::CalleeHandled>;

    #[napi]
    impl VoirsPipeline {
        /// Create a new VoiRS pipeline
        #[napi(constructor)]
        pub fn new(options: Option<PipelineOptions>) -> NapiResult<Self> {
            let rt = Runtime::new()
                .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to create runtime: {}", e)))?;
            
            let mut builder = SdkPipeline::builder();
            
            if let Some(opts) = options {
                if let Some(gpu) = opts.use_gpu {
                    builder = builder.with_gpu(gpu);
                }
                
                if let Some(threads) = opts.num_threads {
                    builder = builder.with_threads(threads as usize);
                }
                
                if let Some(cache) = opts.cache_dir {
                    builder = builder.with_cache_dir(&cache);
                }
                
                if let Some(device) = opts.device {
                    builder = builder.with_device(device);
                }
            }
            
            let inner = rt.block_on(builder.build())
                .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to create pipeline: {}", e)))?;
            
            Ok(Self {
                inner: Arc::new(inner),
                rt,
            })
        }

        /// Synthesize text to audio synchronously
        #[napi]
        pub fn synthesize_sync(&self, text: String, options: Option<SynthesisOptions>) -> NapiResult<AudioBufferResult> {
            let audio = if let Some(opts) = options {
                let config = build_synthesis_config(opts)?;
                self.rt.block_on(self.inner.synthesize_with_config(&text, &config))
            } else {
                self.rt.block_on(self.inner.synthesize(&text))
            }.map_err(|e| Error::new(Status::GenericFailure, format!("Synthesis failed: {}", e)))?;
            
            Ok(audio_buffer_to_result(audio))
        }

        /// Synthesize text to audio asynchronously
        #[napi]
        pub async fn synthesize(&self, text: String, options: Option<SynthesisOptions>) -> NapiResult<AudioBufferResult> {
            let audio = if let Some(opts) = options {
                let config = build_synthesis_config(opts)?;
                self.inner.synthesize_with_config(&text, &config).await
            } else {
                self.inner.synthesize(&text).await
            }.map_err(|e| Error::new(Status::GenericFailure, format!("Synthesis failed: {}", e)))?;
            
            Ok(audio_buffer_to_result(audio))
        }

        /// Synthesize SSML to audio asynchronously
        #[napi]
        pub async fn synthesize_ssml(&self, ssml: String) -> NapiResult<AudioBufferResult> {
            let audio = self.inner.synthesize_ssml(&ssml).await
                .map_err(|e| Error::new(Status::GenericFailure, format!("SSML synthesis failed: {}", e)))?;
            
            Ok(audio_buffer_to_result(audio))
        }

        /// Synthesize with callbacks for progress updates
        #[napi]
        pub async fn synthesize_with_callbacks(
            &self,
            text: String,
            options: Option<SynthesisOptions>,
            progress_callback: Option<JsFunction>,
            error_callback: Option<JsFunction>,
        ) -> NapiResult<AudioBufferResult> {
            // Convert JS functions to thread-safe functions
            let progress_cb: Option<ProgressCallback> = if let Some(cb) = progress_callback {
                Some(cb.create_threadsafe_function(0, |ctx| {
                    ctx.env.create_double(ctx.value).map(|v| vec![v])
                })?)
            } else {
                None
            };

            let error_cb: Option<ErrorCallback> = if let Some(cb) = error_callback {
                Some(cb.create_threadsafe_function(0, |ctx| {
                    ctx.env.create_string(&ctx.value).map(|v| vec![v])
                })?)
            } else {
                None
            };

            // Simulate progress updates
            if let Some(ref progress) = progress_cb {
                progress.call(Ok(0.0), ThreadsafeFunctionCallMode::NonBlocking);
            }

            // Perform synthesis
            let result = if let Some(opts) = options {
                let config = build_synthesis_config(opts)?;
                self.inner.synthesize_with_config(&text, &config).await
            } else {
                self.inner.synthesize(&text).await
            };

            match result {
                Ok(audio) => {
                    if let Some(ref progress) = progress_cb {
                        progress.call(Ok(1.0), ThreadsafeFunctionCallMode::NonBlocking);
                    }
                    Ok(audio_buffer_to_result(audio))
                }
                Err(e) => {
                    let error_msg = format!("Synthesis failed: {}", e);
                    if let Some(ref error) = error_cb {
                        error.call(Ok(error_msg.clone()), ThreadsafeFunctionCallMode::NonBlocking);
                    }
                    Err(Error::new(Status::GenericFailure, error_msg))
                }
            }
        }

        /// Set the voice for synthesis
        #[napi]
        pub async fn set_voice(&self, voice_id: String) -> NapiResult<()> {
            self.inner.set_voice(&voice_id).await
                .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to set voice: {}", e)))?;
            Ok(())
        }

        /// Get the current voice
        #[napi]
        pub async fn get_voice(&self) -> NapiResult<Option<String>> {
            match self.inner.current_voice().await {
                Some(voice) => Ok(Some(voice.id)),
                None => Ok(None),
            }
        }

        /// List available voices
        #[napi]
        pub async fn list_voices(&self) -> NapiResult<Vec<VoiceInfo>> {
            let voices = self.inner.list_voices().await
                .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to list voices: {}", e)))?;
            
            let voice_list = voices.into_iter().map(|voice| VoiceInfo {
                id: voice.id,
                name: voice.name,
                language: voice.language.to_string(),
                quality: format!("{:?}", voice.characteristics.quality),
                is_available: true, // Always available for now
            }).collect();
            
            Ok(voice_list)
        }

        /// Get pipeline information
        #[napi]
        pub fn get_info(&self) -> NapiResult<serde_json::Value> {
            let info = serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "features": {
                    "gpu_support": cfg!(feature = "gpu"),
                    "python_bindings": cfg!(feature = "python"),
                    "nodejs_bindings": true,
                },
                "runtime_info": {
                    "worker_threads": self.rt.metrics().num_workers(),
                }
            });
            Ok(info)
        }
    }

    /// Streaming synthesis with real-time callbacks
    #[napi]
    pub async fn synthesize_streaming(
        pipeline: &VoirsPipeline,
        text: String,
        chunk_callback: JsFunction,
        progress_callback: Option<JsFunction>,
        options: Option<SynthesisOptions>,
    ) -> NapiResult<()> {
        // Create thread-safe callback for audio chunks
        let chunk_cb: ThreadsafeFunction<Buffer, ErrorStrategy::CalleeHandled> = 
            chunk_callback.create_threadsafe_function(0, |ctx| {
                Ok(vec![ctx.value])
            })?;

        let progress_cb: Option<ProgressCallback> = if let Some(cb) = progress_callback {
            Some(cb.create_threadsafe_function(0, |ctx| {
                ctx.env.create_double(ctx.value).map(|v| vec![v])
            })?)
        } else {
            None
        };

        // Perform synthesis
        let audio = if let Some(opts) = options {
            let config = build_synthesis_config(opts)?;
            pipeline.inner.synthesize_with_config(&text, &config).await
        } else {
            pipeline.inner.synthesize(&text).await
        }.map_err(|e| Error::new(Status::GenericFailure, format!("Synthesis failed: {}", e)))?;

        // Stream audio in chunks
        let chunk_size = 1024; // samples per chunk
        let samples = audio.samples();
        let total_chunks = (samples.len() + chunk_size - 1) / chunk_size;

        for (i, chunk) in samples.chunks(chunk_size).enumerate() {
            // Send progress update
            if let Some(ref progress) = progress_cb {
                let progress_value = (i as f64) / (total_chunks as f64);
                progress.call(Ok(progress_value), ThreadsafeFunctionCallMode::NonBlocking);
            }

            // Convert chunk to bytes and send
            let chunk_bytes: Vec<u8> = chunk.iter()
                .flat_map(|&sample| sample.to_le_bytes())
                .collect();
            
            let buffer = Buffer::from(chunk_bytes);
            chunk_cb.call(Ok(buffer), ThreadsafeFunctionCallMode::Blocking);
        }

        // Final progress update
        if let Some(ref progress) = progress_cb {
            progress.call(Ok(1.0), ThreadsafeFunctionCallMode::NonBlocking);
        }

        Ok(())
    }

    /// Utility functions
    fn build_synthesis_config(options: SynthesisOptions) -> NapiResult<voirs::types::SynthesisConfig> {
        let mut config = voirs::types::SynthesisConfig::default();
        
        if let Some(rate) = options.speaking_rate {
            config.speaking_rate = rate as f32;
        }
        
        if let Some(pitch) = options.pitch_shift {
            config.pitch_shift = pitch as f32;
        }
        
        if let Some(gain) = options.volume_gain {
            config.volume_gain = gain as f32;
        }
        
        if let Some(enhancement) = options.enable_enhancement {
            config.enable_enhancement = enhancement;
        }
        
        if let Some(format) = options.output_format {
            config.output_format = match format.as_str() {
                "wav" => voirs::types::AudioFormat::Wav,
                "flac" => voirs::types::AudioFormat::Flac,
                "mp3" => voirs::types::AudioFormat::Mp3,
                "opus" => voirs::types::AudioFormat::Opus,
                "ogg" => voirs::types::AudioFormat::Ogg,
                _ => return Err(Error::new(Status::InvalidArg, "Invalid audio format")),
            };
        }
        
        if let Some(rate) = options.sample_rate {
            config.sample_rate = rate;
        }
        
        if let Some(quality) = options.quality {
            config.quality = match quality.as_str() {
                "low" => voirs::types::QualityLevel::Low,
                "medium" => voirs::types::QualityLevel::Medium,
                "high" => voirs::types::QualityLevel::High,
                "ultra" => voirs::types::QualityLevel::Ultra,
                _ => return Err(Error::new(Status::InvalidArg, "Invalid quality level")),
            };
        }
        
        Ok(config)
    }

    fn audio_buffer_to_result(audio: AudioBuffer) -> AudioBufferResult {
        let samples = audio.samples();
        let sample_bytes: Vec<u8> = samples.iter()
            .flat_map(|&sample| sample.to_le_bytes())
            .collect();
        
        AudioBufferResult {
            samples: Buffer::from(sample_bytes),
            sample_rate: audio.sample_rate(),
            channels: audio.channels(),
            duration: audio.duration(),
        }
    }
}

#[cfg(not(feature = "nodejs"))]
pub mod napi_bindings {
    //! Stub module when Node.js feature is not enabled
    pub fn not_available() -> &'static str {
        "Node.js bindings not available. Enable the 'nodejs' feature to use these bindings."
    }
}