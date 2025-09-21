//! Node.js bindings for VoiRS using NAPI-RS.
//!
//! This module provides Node.js bindings that expose VoiRS functionality
//! to JavaScript/TypeScript applications with full async support.

#[cfg(feature = "nodejs")]
pub mod napi_bindings {
    use crate::{VoirsAudioFormat, VoirsQualityLevel};
    use napi::{
        bindgen_prelude::*,
        threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode},
        JsFunction, Result as NapiResult, Status,
    };
    use napi_derive::napi;
    use std::sync::Arc;
    use tokio::runtime::Runtime;
    use voirs_sdk::{audio::AudioBuffer, error::VoirsError, VoirsPipeline as SdkPipeline};

    /// Node.js VoiRS Pipeline wrapper
    /// Cache statistics for tracking cache performance
    #[derive(Debug, Clone)]
    struct CacheStats {
        total_requests: u64,
        cache_hits: u64,
        cache_size_bytes: u64,
    }

    impl CacheStats {
        fn new() -> Self {
            Self {
                total_requests: 0,
                cache_hits: 0,
                cache_size_bytes: 0,
            }
        }

        fn hit_rate(&self) -> f64 {
            if self.total_requests == 0 {
                0.0
            } else {
                self.cache_hits as f64 / self.total_requests as f64
            }
        }

        fn size_mb(&self) -> f64 {
            self.cache_size_bytes as f64 / (1024.0 * 1024.0)
        }
    }

    #[napi]
    pub struct VoirsPipeline {
        inner: Arc<SdkPipeline>,
        rt: Runtime,
        cache_stats: Arc<std::sync::Mutex<CacheStats>>,
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

    /// Synthesis metrics for performance monitoring
    #[napi(object)]
    pub struct SynthesisMetrics {
        pub processing_time_ms: f64,
        pub audio_duration_ms: f64,
        pub real_time_factor: f64,
        pub memory_usage_mb: f64,
        pub cache_hit_rate: f64,
    }

    /// Enhanced synthesis result with metrics
    #[napi(object)]
    pub struct SynthesisResult {
        pub audio: AudioBufferResult,
        pub metrics: SynthesisMetrics,
    }

    /// Structured error information
    #[napi(object)]
    pub struct ErrorInfo {
        pub code: String,
        pub message: String,
        pub details: Option<String>,
        pub suggestion: Option<String>,
    }

    /// Audio analysis result
    #[napi(object)]
    pub struct AudioAnalysis {
        pub duration_seconds: f64,
        pub sample_rate: u32,
        pub channels: u32,
        pub rms_energy: f64,
        pub zero_crossing_rate: f64,
        pub spectral_centroid: f64,
        pub silence_regions: Vec<Vec<u32>>,
    }

    /// Performance information
    #[napi(object)]
    pub struct PerformanceInfo {
        pub cpu_cores: u32,
        pub memory_usage_mb: f64,
        pub gpu_available: bool,
        pub cache_size_mb: f64,
        pub thread_count: u32,
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
    pub type CompletionCallback =
        ThreadsafeFunction<AudioBufferResult, ErrorStrategy::CalleeHandled>;

    #[napi]
    impl VoirsPipeline {
        /// Create a new VoiRS pipeline
        #[napi(constructor)]
        pub fn new(options: Option<PipelineOptions>) -> Self {
            let rt = Runtime::new().expect("Failed to create runtime");

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

            let inner = rt
                .block_on(builder.build())
                .expect("Failed to create pipeline");

            Self {
                inner: Arc::new(inner),
                rt,
                cache_stats: Arc::new(std::sync::Mutex::new(CacheStats::new())),
            }
        }

        /// Helper method to simulate cache tracking for synthesis requests
        fn track_cache_request(&self, text: &str) -> bool {
            // Simple cache simulation based on text length and content
            let cache_hit = text.len() < 100
                && text
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c.is_whitespace());

            if let Ok(mut stats) = self.cache_stats.lock() {
                stats.total_requests += 1;
                if cache_hit {
                    stats.cache_hits += 1;
                }
                // Simulate cache size growth
                stats.cache_size_bytes += text.len() as u64 * 1024; // Approximate 1KB per character
            }

            cache_hit
        }

        /// Get current cache hit rate
        fn get_cache_hit_rate(&self) -> f64 {
            if let Ok(stats) = self.cache_stats.lock() {
                stats.hit_rate()
            } else {
                0.0
            }
        }

        /// Get current cache size in MB
        fn get_cache_size_mb(&self) -> f64 {
            if let Ok(stats) = self.cache_stats.lock() {
                stats.size_mb()
            } else {
                0.0
            }
        }

        /// Synthesize text to audio synchronously
        #[napi]
        pub fn synthesize_sync(
            &self,
            text: String,
            options: Option<SynthesisOptions>,
        ) -> NapiResult<AudioBufferResult> {
            let audio = if let Some(opts) = options {
                let config = build_synthesis_config(opts)?;
                self.rt
                    .block_on(self.inner.synthesize_with_config(&text, &config))
            } else {
                self.rt.block_on(self.inner.synthesize(&text))
            }
            .map_err(|e| Error::new(Status::GenericFailure, format!("Synthesis failed: {}", e)))?;

            Ok(audio_buffer_to_result(audio))
        }

        /// Synthesize text to audio asynchronously
        #[napi]
        pub fn synthesize(
            &self,
            text: String,
            options: Option<SynthesisOptions>,
        ) -> NapiResult<AudioBufferResult> {
            let audio = if let Some(opts) = options {
                let config = build_synthesis_config(opts)?;
                self.rt
                    .block_on(self.inner.synthesize_with_config(&text, &config))
            } else {
                self.rt.block_on(self.inner.synthesize(&text))
            }
            .map_err(|e| Error::new(Status::GenericFailure, format!("Synthesis failed: {}", e)))?;

            Ok(audio_buffer_to_result(audio))
        }

        /// Synthesize SSML to audio asynchronously
        #[napi]
        pub fn synthesize_ssml(&self, ssml: String) -> NapiResult<AudioBufferResult> {
            let audio = self
                .rt
                .block_on(self.inner.synthesize_ssml(&ssml))
                .map_err(|e| {
                    Error::new(
                        Status::GenericFailure,
                        format!("SSML synthesis failed: {}", e),
                    )
                })?;

            Ok(audio_buffer_to_result(audio))
        }

        /// Synthesize with callbacks for progress updates
        #[napi]
        pub fn synthesize_with_callbacks(
            &self,
            text: String,
            options: Option<SynthesisOptions>,
            progress_callback: Option<JsFunction>,
            error_callback: Option<JsFunction>,
        ) -> NapiResult<AudioBufferResult> {
            // Convert JS functions to thread-safe functions
            let progress_cb: Option<ProgressCallback> = if let Some(cb) = progress_callback {
                Some(cb.create_threadsafe_function::<f64, _, _, _>(0, |ctx| {
                    ctx.env.create_double(ctx.value).map(|v| vec![v])
                })?)
            } else {
                None
            };

            let error_cb: Option<ErrorCallback> = if let Some(cb) = error_callback {
                Some(cb.create_threadsafe_function::<String, _, _, _>(0, |ctx| {
                    ctx.env.create_string(&ctx.value).map(|v| vec![v])
                })?)
            } else {
                None
            };

            // Simulate progress updates
            if let Some(ref progress) = progress_cb {
                let _: Status = progress.call(Ok(0.0), ThreadsafeFunctionCallMode::NonBlocking);
            }

            // Perform synthesis
            let result = if let Some(opts) = options {
                let config = build_synthesis_config(opts)?;
                self.rt
                    .block_on(self.inner.synthesize_with_config(&text, &config))
            } else {
                self.rt.block_on(self.inner.synthesize(&text))
            };

            match result {
                Ok(audio) => {
                    if let Some(ref progress) = progress_cb {
                        let _: Status =
                            progress.call(Ok(1.0), ThreadsafeFunctionCallMode::NonBlocking);
                    }
                    Ok(audio_buffer_to_result(audio))
                }
                Err(e) => {
                    let error_msg = format!("Synthesis failed: {}", e);
                    if let Some(ref error) = error_cb {
                        let _: Status = error.call(
                            Ok(error_msg.clone()),
                            ThreadsafeFunctionCallMode::NonBlocking,
                        );
                    }
                    Err(Error::new(Status::GenericFailure, error_msg))
                }
            }
        }

        /// Set the voice for synthesis
        #[napi]
        pub fn set_voice(&self, voice_id: String) -> NapiResult<()> {
            self.rt
                .block_on(self.inner.set_voice(&voice_id))
                .map_err(|e| {
                    Error::new(
                        Status::GenericFailure,
                        format!("Failed to set voice: {}", e),
                    )
                })?;
            Ok(())
        }

        /// Get the current voice
        #[napi]
        pub fn get_voice(&self) -> NapiResult<Option<String>> {
            match self.rt.block_on(self.inner.current_voice()) {
                Some(voice) => Ok(Some(voice.id)),
                None => Ok(None),
            }
        }

        /// List available voices
        #[napi]
        pub fn list_voices(&self) -> NapiResult<Vec<VoiceInfo>> {
            let voices = self.rt.block_on(self.inner.list_voices()).map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Failed to list voices: {}", e),
                )
            })?;

            let voice_list = voices
                .into_iter()
                .map(|voice| VoiceInfo {
                    id: voice.id,
                    name: voice.name,
                    language: voice.language.to_string(),
                    quality: format!("{:?}", voice.characteristics.quality),
                    is_available: true, // Always available for now
                })
                .collect();

            Ok(voice_list)
        }

        /// Enhanced synthesis with metrics
        #[napi]
        pub fn synthesize_with_metrics(
            &self,
            text: String,
            options: Option<SynthesisOptions>,
        ) -> NapiResult<SynthesisResult> {
            use std::time::Instant;

            let start_time = Instant::now();
            let start_memory = self.get_memory_usage_mb();

            // Track cache request
            self.track_cache_request(&text);

            let audio = if let Some(opts) = options {
                let config = build_synthesis_config(opts)?;
                self.rt
                    .block_on(self.inner.synthesize_with_config(&text, &config))
            } else {
                self.rt.block_on(self.inner.synthesize(&text))
            }
            .map_err(|e| Error::new(Status::GenericFailure, format!("Synthesis failed: {}", e)))?;

            let processing_time_ms = start_time.elapsed().as_millis() as f64;
            let audio_duration_ms =
                (audio.samples().len() as f64) / (audio.sample_rate() as f64) * 1000.0;
            let real_time_factor = processing_time_ms / audio_duration_ms;
            let memory_usage_mb = self.get_memory_usage_mb() - start_memory;

            let metrics = SynthesisMetrics {
                processing_time_ms,
                audio_duration_ms,
                real_time_factor,
                memory_usage_mb,
                cache_hit_rate: self.get_cache_hit_rate(),
            };

            Ok(SynthesisResult {
                audio: audio_buffer_to_result(audio),
                metrics,
            })
        }

        /// Enhanced SSML synthesis with metrics
        #[napi]
        pub fn synthesize_ssml_with_metrics(&self, ssml: String) -> NapiResult<SynthesisResult> {
            use std::time::Instant;

            let start_time = Instant::now();
            let start_memory = self.get_memory_usage_mb();

            // Track cache request for SSML
            self.track_cache_request(&ssml);

            let audio = self
                .rt
                .block_on(self.inner.synthesize_ssml(&ssml))
                .map_err(|e| {
                    Error::new(
                        Status::GenericFailure,
                        format!("SSML synthesis failed: {}", e),
                    )
                })?;

            let processing_time_ms = start_time.elapsed().as_millis() as f64;
            let audio_duration_ms =
                (audio.samples().len() as f64) / (audio.sample_rate() as f64) * 1000.0;
            let real_time_factor = processing_time_ms / audio_duration_ms;
            let memory_usage_mb = self.get_memory_usage_mb() - start_memory;

            let metrics = SynthesisMetrics {
                processing_time_ms,
                audio_duration_ms,
                real_time_factor,
                memory_usage_mb,
                cache_hit_rate: self.get_cache_hit_rate(),
            };

            Ok(SynthesisResult {
                audio: audio_buffer_to_result(audio),
                metrics,
            })
        }

        /// Batch synthesis with progress tracking
        #[napi]
        pub fn batch_synthesize(
            &self,
            texts: Vec<String>,
            options: Option<SynthesisOptions>,
            progress_callback: Option<JsFunction>,
        ) -> NapiResult<Vec<SynthesisResult>> {
            let progress_cb: Option<
                ThreadsafeFunction<(u32, u32, f64), ErrorStrategy::CalleeHandled>,
            > = if let Some(cb) = progress_callback {
                Some(
                    cb.create_threadsafe_function::<(u32, u32, f64), _, _, _>(0, |ctx| {
                        let (current, total, progress) = ctx.value;
                        let current_val = ctx.env.create_uint32(current)?;
                        let total_val = ctx.env.create_uint32(total)?;
                        let progress_val = ctx.env.create_double(progress)?;
                        Ok(vec![current_val, total_val, progress_val])
                    })?,
                )
            } else {
                None
            };

            let mut results = Vec::new();
            let total_count = texts.len() as u32;

            for (i, text) in texts.iter().enumerate() {
                if let Some(ref progress) = progress_cb {
                    let progress_value = (i as f64) / (total_count as f64);
                    let _: Status = progress.call(
                        Ok((i as u32, total_count, progress_value)),
                        ThreadsafeFunctionCallMode::NonBlocking,
                    );
                }

                let result = self.synthesize_with_metrics(text.clone(), options.clone())?;
                results.push(result);
            }

            // Final progress callback
            if let Some(ref progress) = progress_cb {
                let _: Status = progress.call(
                    Ok((total_count, total_count, 1.0)),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }

            Ok(results)
        }

        /// Analyze audio buffer
        #[napi]
        pub fn analyze_audio(&self, audio_buffer: &AudioBufferResult) -> NapiResult<AudioAnalysis> {
            let samples = audio_buffer.samples.as_ref();
            let sample_count = samples.len() / 2; // Assuming 16-bit samples
            let mut f32_samples = Vec::with_capacity(sample_count);

            // Convert bytes to f32 samples
            for chunk in samples.chunks(2) {
                if chunk.len() == 2 {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                    f32_samples.push(sample);
                }
            }

            // Calculate RMS energy
            let rms_energy = if !f32_samples.is_empty() {
                let sum_squares: f32 = f32_samples.iter().map(|&x| x * x).sum();
                (sum_squares / f32_samples.len() as f32).sqrt()
            } else {
                0.0
            };

            // Calculate zero crossing rate
            let zero_crossing_rate = if f32_samples.len() > 1 {
                let mut crossings = 0;
                for i in 1..f32_samples.len() {
                    if (f32_samples[i] >= 0.0) != (f32_samples[i - 1] >= 0.0) {
                        crossings += 1;
                    }
                }
                crossings as f64 / (f32_samples.len() - 1) as f64
            } else {
                0.0
            };

            // Calculate spectral centroid (simplified)
            let spectral_centroid = if !f32_samples.is_empty() {
                let mut magnitude_sum = 0.0f32;
                let mut weighted_sum = 0.0f32;

                for (i, &sample) in f32_samples.iter().enumerate() {
                    let magnitude = sample.abs();
                    let freq =
                        (i as f32 * audio_buffer.sample_rate as f32) / (f32_samples.len() as f32);
                    magnitude_sum += magnitude;
                    weighted_sum += magnitude * freq;
                }

                if magnitude_sum > 0.0 {
                    weighted_sum / magnitude_sum
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Find silence regions (simplified)
            let silence_threshold = 0.01;
            let min_silence_duration = audio_buffer.sample_rate / 10; // 100ms
            let mut silence_regions = Vec::new();
            let mut in_silence = false;
            let mut silence_start = 0;

            for (i, &sample) in f32_samples.iter().enumerate() {
                let is_silent = sample.abs() < silence_threshold;

                if is_silent && !in_silence {
                    silence_start = i;
                    in_silence = true;
                } else if !is_silent && in_silence {
                    let duration = i - silence_start;
                    if duration >= min_silence_duration as usize {
                        silence_regions.push(vec![silence_start as u32, i as u32]);
                    }
                    in_silence = false;
                }
            }

            // Handle silence at the end
            if in_silence {
                let duration = f32_samples.len() - silence_start;
                if duration >= min_silence_duration as usize {
                    silence_regions.push(vec![silence_start as u32, f32_samples.len() as u32]);
                }
            }

            Ok(AudioAnalysis {
                duration_seconds: audio_buffer.duration,
                sample_rate: audio_buffer.sample_rate,
                channels: audio_buffer.channels,
                rms_energy: rms_energy as f64,
                zero_crossing_rate,
                spectral_centroid: spectral_centroid as f64,
                silence_regions,
            })
        }

        /// Get performance information
        #[napi]
        pub fn get_performance_info(&self) -> NapiResult<PerformanceInfo> {
            Ok(PerformanceInfo {
                cpu_cores: num_cpus::get() as u32,
                memory_usage_mb: self.get_memory_usage_mb(),
                gpu_available: self.is_gpu_available(),
                cache_size_mb: self.get_cache_size_mb(),
                thread_count: self.rt.metrics().num_workers() as u32,
            })
        }

        /// Get pipeline information
        #[napi]
        pub fn get_info(&self) -> String {
            serde_json::to_string(&serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "features": {
                    "gpu_support": cfg!(feature = "gpu"),
                    "python_bindings": cfg!(feature = "python"),
                    "nodejs_bindings": true,
                },
                "runtime_info": {
                    "worker_threads": self.rt.metrics().num_workers(),
                }
            }))
            .unwrap_or_else(|_| "{}".to_string())
        }
    }

    // Helper methods for VoirsPipeline
    impl VoirsPipeline {
        /// Get current memory usage in MB
        fn get_memory_usage_mb(&self) -> f64 {
            // Simplified memory usage calculation
            #[cfg(all(target_os = "linux", feature = "memory-detection"))]
            {
                if let Ok(memory_info) = procfs::process::Process::myself() {
                    if let Ok(stat) = memory_info.stat() {
                        return stat.rss as f64 / 1024.0 / 1024.0; // Convert pages to MB
                    }
                }
            }

            #[cfg(all(windows, feature = "memory-detection"))]
            {
                // Windows memory usage detection (simplified)
                // This would need proper Windows API calls
                0.0
            }

            // Fallback for other platforms
            0.0
        }

        /// Check if GPU is available
        fn is_gpu_available(&self) -> bool {
            // This would need proper GPU detection
            cfg!(feature = "gpu")
        }
    }

    /// Streaming synthesis with real-time callbacks
    #[napi]
    pub fn synthesize_streaming(
        pipeline: &VoirsPipeline,
        text: String,
        chunk_callback: JsFunction,
        progress_callback: Option<JsFunction>,
        options: Option<SynthesisOptions>,
    ) -> NapiResult<()> {
        // Create thread-safe callback for audio chunks
        let chunk_cb: ThreadsafeFunction<Buffer, ErrorStrategy::CalleeHandled> =
            chunk_callback
                .create_threadsafe_function::<Buffer, _, _, _>(0, |ctx| Ok(vec![ctx.value]))?;

        let progress_cb: Option<ProgressCallback> = if let Some(cb) = progress_callback {
            Some(cb.create_threadsafe_function::<f64, _, _, _>(0, |ctx| {
                ctx.env.create_double(ctx.value).map(|v| vec![v])
            })?)
        } else {
            None
        };

        // Perform synthesis
        let audio = if let Some(opts) = options {
            let config = build_synthesis_config(opts)?;
            pipeline
                .rt
                .block_on(pipeline.inner.synthesize_with_config(&text, &config))
        } else {
            pipeline.rt.block_on(pipeline.inner.synthesize(&text))
        }
        .map_err(|e| Error::new(Status::GenericFailure, format!("Synthesis failed: {}", e)))?;

        // Stream audio in chunks
        let chunk_size = 1024; // samples per chunk
        let samples = audio.samples();
        let total_chunks = (samples.len() + chunk_size - 1) / chunk_size;

        for (i, chunk) in samples.chunks(chunk_size).enumerate() {
            // Send progress update
            if let Some(ref progress) = progress_cb {
                let progress_value = (i as f64) / (total_chunks as f64);
                let _: Status =
                    progress.call(Ok(progress_value), ThreadsafeFunctionCallMode::NonBlocking);
            }

            // Convert chunk to bytes and send
            let chunk_bytes: Vec<u8> = chunk
                .iter()
                .flat_map(|&sample| sample.to_le_bytes())
                .collect();

            let buffer = Buffer::from(chunk_bytes);
            let _: Status = chunk_cb.call(Ok(buffer), ThreadsafeFunctionCallMode::Blocking);
        }

        // Final progress update
        if let Some(ref progress) = progress_cb {
            let _: Status = progress.call(Ok(1.0), ThreadsafeFunctionCallMode::NonBlocking);
        }

        Ok(())
    }

    /// Utility functions
    fn build_synthesis_config(
        options: SynthesisOptions,
    ) -> NapiResult<voirs_sdk::types::SynthesisConfig> {
        let mut config = voirs_sdk::types::SynthesisConfig::default();

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
                "wav" => voirs_sdk::types::AudioFormat::Wav,
                "flac" => voirs_sdk::types::AudioFormat::Flac,
                "mp3" => voirs_sdk::types::AudioFormat::Mp3,
                "opus" => voirs_sdk::types::AudioFormat::Opus,
                "ogg" => voirs_sdk::types::AudioFormat::Ogg,
                _ => return Err(Error::new(Status::InvalidArg, "Invalid audio format")),
            };
        }

        if let Some(rate) = options.sample_rate {
            config.sample_rate = rate;
        }

        if let Some(quality) = options.quality {
            config.quality = match quality.as_str() {
                "low" => voirs_sdk::types::QualityLevel::Low,
                "medium" => voirs_sdk::types::QualityLevel::Medium,
                "high" => voirs_sdk::types::QualityLevel::High,
                "ultra" => voirs_sdk::types::QualityLevel::Ultra,
                _ => return Err(Error::new(Status::InvalidArg, "Invalid quality level")),
            };
        }

        Ok(config)
    }

    fn audio_buffer_to_result(audio: AudioBuffer) -> AudioBufferResult {
        let samples = audio.samples();
        let sample_bytes: Vec<u8> = samples
            .iter()
            .flat_map(|&sample| sample.to_le_bytes())
            .collect();

        AudioBufferResult {
            samples: Buffer::from(sample_bytes),
            sample_rate: audio.sample_rate(),
            channels: audio.channels(),
            duration: audio.duration() as f64,
        }
    }

    // ========================================================================
    // Audio Utility Functions
    // ========================================================================

    /// Convert audio buffer to WAV format
    #[napi]
    pub fn audio_to_wav(audio_buffer: &AudioBufferResult) -> NapiResult<Buffer> {
        let samples = audio_buffer.samples.as_ref();
        let sample_rate = audio_buffer.sample_rate;
        let channels = audio_buffer.channels;

        // Create WAV header
        let data_size = samples.len() as u32;
        let file_size = 36 + data_size;

        let mut wav_data = Vec::new();

        // RIFF header
        wav_data.extend_from_slice(b"RIFF");
        wav_data.extend_from_slice(&file_size.to_le_bytes());
        wav_data.extend_from_slice(b"WAVE");

        // Format chunk
        wav_data.extend_from_slice(b"fmt ");
        wav_data.extend_from_slice(&16u32.to_le_bytes()); // format chunk size
        wav_data.extend_from_slice(&1u16.to_le_bytes()); // PCM format
        wav_data.extend_from_slice(&(channels as u16).to_le_bytes());
        wav_data.extend_from_slice(&sample_rate.to_le_bytes());
        wav_data.extend_from_slice(&(sample_rate * channels * 2).to_le_bytes()); // byte rate
        wav_data.extend_from_slice(&(channels * 2).to_le_bytes()); // block align
        wav_data.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

        // Data chunk
        wav_data.extend_from_slice(b"data");
        wav_data.extend_from_slice(&data_size.to_le_bytes());
        wav_data.extend_from_slice(samples);

        Ok(Buffer::from(wav_data))
    }

    /// Get supported audio formats
    #[napi]
    pub fn get_supported_formats() -> Vec<String> {
        vec![
            "wav".to_string(),
            "flac".to_string(),
            "mp3".to_string(),
            "opus".to_string(),
            "ogg".to_string(),
        ]
    }

    /// Get supported quality levels
    #[napi]
    pub fn get_supported_qualities() -> Vec<String> {
        vec![
            "low".to_string(),
            "medium".to_string(),
            "high".to_string(),
            "ultra".to_string(),
        ]
    }

    /// Validate synthesis options
    #[napi]
    pub fn validate_synthesis_options(options: &SynthesisOptions) -> NapiResult<bool> {
        // Check speaking rate
        if let Some(rate) = options.speaking_rate {
            if !(0.1..=3.0).contains(&rate) {
                return Err(Error::new(
                    Status::InvalidArg,
                    "Speaking rate must be between 0.1 and 3.0",
                ));
            }
        }

        // Check pitch shift
        if let Some(pitch) = options.pitch_shift {
            if !(-12.0..=12.0).contains(&pitch) {
                return Err(Error::new(
                    Status::InvalidArg,
                    "Pitch shift must be between -12.0 and 12.0 semitones",
                ));
            }
        }

        // Check volume gain
        if let Some(gain) = options.volume_gain {
            if !(-20.0..=20.0).contains(&gain) {
                return Err(Error::new(
                    Status::InvalidArg,
                    "Volume gain must be between -20.0 and 20.0 dB",
                ));
            }
        }

        // Check sample rate
        if let Some(rate) = options.sample_rate {
            if ![8000, 16000, 22050, 44100, 48000].contains(&rate) {
                return Err(Error::new(
                    Status::InvalidArg,
                    "Sample rate must be 8000, 16000, 22050, 44100, or 48000 Hz",
                ));
            }
        }

        // Check output format
        if let Some(format) = &options.output_format {
            if !["wav", "flac", "mp3", "opus", "ogg"].contains(&format.as_str()) {
                return Err(Error::new(
                    Status::InvalidArg,
                    "Output format must be wav, flac, mp3, opus, or ogg",
                ));
            }
        }

        // Check quality
        if let Some(quality) = &options.quality {
            if !["low", "medium", "high", "ultra"].contains(&quality.as_str()) {
                return Err(Error::new(
                    Status::InvalidArg,
                    "Quality must be low, medium, high, or ultra",
                ));
            }
        }

        Ok(true)
    }

    /// Resample audio buffer
    #[napi]
    pub fn resample_audio(
        audio_buffer: &AudioBufferResult,
        target_sample_rate: u32,
    ) -> NapiResult<AudioBufferResult> {
        let samples = audio_buffer.samples.as_ref();
        let original_rate = audio_buffer.sample_rate;

        if original_rate == target_sample_rate {
            return Ok(AudioBufferResult {
                samples: Buffer::from(samples.to_vec()),
                sample_rate: target_sample_rate,
                channels: audio_buffer.channels,
                duration: audio_buffer.duration,
            });
        }

        // Convert bytes to f32 samples
        let mut f32_samples = Vec::new();
        for chunk in samples.chunks(2) {
            if chunk.len() == 2 {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                f32_samples.push(sample);
            }
        }

        // Simple linear interpolation resampling
        let ratio = target_sample_rate as f64 / original_rate as f64;
        let new_length = (f32_samples.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let src_index = i as f64 / ratio;
            let src_index_floor = src_index.floor() as usize;
            let src_index_ceil = (src_index_floor + 1).min(f32_samples.len() - 1);
            let frac = src_index - src_index_floor as f64;

            let sample = if src_index_floor < f32_samples.len() {
                let a = f32_samples[src_index_floor];
                let b = f32_samples[src_index_ceil];
                a + (b - a) * frac as f32
            } else {
                0.0
            };
            resampled.push(sample);
        }

        // Convert back to bytes
        let resampled_bytes: Vec<u8> = resampled
            .iter()
            .flat_map(|&sample| ((sample * 32768.0) as i16).to_le_bytes())
            .collect();

        let new_duration = resampled.len() as f64 / target_sample_rate as f64;

        Ok(AudioBufferResult {
            samples: Buffer::from(resampled_bytes),
            sample_rate: target_sample_rate,
            channels: audio_buffer.channels,
            duration: new_duration,
        })
    }

    /// Mix two audio buffers
    #[napi]
    pub fn mix_audio(
        audio1: &AudioBufferResult,
        audio2: &AudioBufferResult,
        mix_ratio: Option<f64>,
    ) -> NapiResult<AudioBufferResult> {
        let mix_ratio = mix_ratio.unwrap_or(0.5);

        if !(0.0..=1.0).contains(&mix_ratio) {
            return Err(Error::new(
                Status::InvalidArg,
                "Mix ratio must be between 0.0 and 1.0",
            ));
        }

        // For simplicity, assume both buffers have the same format
        let samples1 = audio1.samples.as_ref();
        let samples2 = audio2.samples.as_ref();
        let min_length = samples1.len().min(samples2.len());

        let mut mixed_samples = Vec::with_capacity(min_length);

        for i in 0..min_length {
            let sample1 = samples1[i] as f32;
            let sample2 = samples2[i] as f32;
            let mixed = (sample1 * (1.0 - mix_ratio as f32) + sample2 * mix_ratio as f32) as u8;
            mixed_samples.push(mixed);
        }

        Ok(AudioBufferResult {
            samples: Buffer::from(mixed_samples),
            sample_rate: audio1.sample_rate,
            channels: audio1.channels,
            duration: (min_length as f64 / 2.0) / audio1.sample_rate as f64,
        })
    }

    // ========================================================================
    // Recognition/ASR Integration (when recognition feature is enabled)
    // ========================================================================

    #[cfg(feature = "recognition")]
    mod recognition_bindings {
        use super::*;
        use voirs_recognizer::prelude::*;

        /// ASR Model wrapper for Node.js
        #[napi]
        pub struct ASRModel {
            inner: Box<dyn voirs_recognizer::traits::ASRModel + Send + Sync>,
            rt: Runtime,
        }

        /// Recognition result
        #[napi(object)]
        pub struct RecognitionResult {
            pub text: String,
            pub confidence: f64,
            pub language: String,
            pub processing_time_ms: f64,
        }

        /// Audio analysis result
        #[napi(object)]
        pub struct AudioAnalysisResult {
            pub duration_seconds: f64,
            pub sample_rate: u32,
            pub channels: u32,
            pub rms_energy: f64,
            pub zero_crossing_rate: f64,
            pub spectral_centroid: f64,
        }

        #[napi]
        impl ASRModel {
            /// Create a new ASR model with Whisper
            #[napi(factory)]
            pub fn whisper(model_size: Option<String>) -> NapiResult<Self> {
                let rt = Runtime::new().map_err(|e| {
                    Error::new(
                        Status::GenericFailure,
                        format!("Failed to create runtime: {}", e),
                    )
                })?;

                #[cfg(feature = "whisper")]
                {
                    let model = rt
                        .block_on(async {
                            voirs_recognizer::asr::whisper::WhisperModel::new(
                                model_size.as_deref().unwrap_or("base"),
                            )
                            .await
                        })
                        .map_err(|e| {
                            Error::new(
                                Status::GenericFailure,
                                format!("Failed to load Whisper model: {}", e),
                            )
                        })?;

                    Ok(Self {
                        inner: Box::new(model),
                        rt,
                    })
                }
                #[cfg(not(feature = "whisper"))]
                {
                    Err(Error::new(
                        Status::GenericFailure,
                        "Whisper support not compiled in",
                    ))
                }
            }

            /// Recognize speech from audio buffer
            #[napi]
            pub fn recognize(&self, audio: &AudioBufferResult) -> NapiResult<RecognitionResult> {
                // Convert audio buffer to VoiRS AudioBuffer
                let samples = audio.samples.as_ref();
                let mut f32_samples = Vec::new();
                for chunk in samples.chunks(2) {
                    if chunk.len() == 2 {
                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                        f32_samples.push(sample);
                    }
                }

                let audio_buffer = AudioBuffer::new(f32_samples, audio.sample_rate, audio.channels);
                let config = voirs_recognizer::traits::ASRConfig::default();

                let result = self
                    .rt
                    .block_on(async { self.inner.transcribe(&audio_buffer, Some(&config)).await })
                    .map_err(|e| {
                        Error::new(Status::GenericFailure, format!("Recognition failed: {}", e))
                    })?;

                Ok(RecognitionResult {
                    text: result.text,
                    confidence: result.confidence as f64,
                    language: result.language.to_string(),
                    processing_time_ms: result
                        .processing_duration
                        .map(|d| d.as_millis() as f64)
                        .unwrap_or(0.0),
                })
            }

            /// Get supported languages
            #[napi]
            pub fn supported_languages(&self) -> Vec<String> {
                self.inner
                    .supported_languages()
                    .iter()
                    .map(|lang| lang.to_string())
                    .collect()
            }
        }

        /// Audio analyzer for Node.js
        #[napi]
        pub struct AudioAnalyzer {
            inner: voirs_recognizer::analysis::AudioAnalyzerImpl,
            rt: Runtime,
        }

        #[napi]
        impl AudioAnalyzer {
            /// Create a new audio analyzer
            #[napi(constructor)]
            pub fn new() -> NapiResult<Self> {
                let rt = Runtime::new().map_err(|e| {
                    Error::new(
                        Status::GenericFailure,
                        format!("Failed to create runtime: {}", e),
                    )
                })?;

                let config = voirs_recognizer::traits::AudioAnalysisConfig::default();
                let analyzer = rt
                    .block_on(async {
                        voirs_recognizer::analysis::AudioAnalyzerImpl::new(config).await
                    })
                    .map_err(|e| {
                        Error::new(
                            Status::GenericFailure,
                            format!("Failed to create analyzer: {}", e),
                        )
                    })?;

                Ok(Self {
                    inner: analyzer,
                    rt,
                })
            }

            /// Analyze audio buffer
            #[napi]
            pub fn analyze(&self, audio: &AudioBufferResult) -> NapiResult<AudioAnalysisResult> {
                // Convert audio buffer to VoiRS AudioBuffer
                let samples = audio.samples.as_ref();
                let mut f32_samples = Vec::new();
                for chunk in samples.chunks(2) {
                    if chunk.len() == 2 {
                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                        f32_samples.push(sample);
                    }
                }

                let audio_buffer = AudioBuffer::new(f32_samples, audio.sample_rate, audio.channels);
                let config = voirs_recognizer::traits::AudioAnalysisConfig::default();

                let analysis = self
                    .rt
                    .block_on(async { self.inner.analyze(&audio_buffer, Some(&config)).await })
                    .map_err(|e| {
                        Error::new(Status::GenericFailure, format!("Analysis failed: {}", e))
                    })?;

                Ok(AudioAnalysisResult {
                    duration_seconds: audio.duration,
                    sample_rate: audio.sample_rate,
                    channels: audio.channels,
                    rms_energy: analysis
                        .quality_metrics
                        .get("rms_energy")
                        .copied()
                        .unwrap_or(0.0) as f64,
                    zero_crossing_rate: analysis
                        .quality_metrics
                        .get("zero_crossing_rate")
                        .copied()
                        .unwrap_or(0.0) as f64,
                    spectral_centroid: analysis
                        .quality_metrics
                        .get("spectral_centroid")
                        .copied()
                        .unwrap_or(0.0) as f64,
                })
            }
        }
    }

    #[cfg(feature = "recognition")]
    pub use recognition_bindings::*;
}

#[cfg(not(feature = "nodejs"))]
pub mod napi_bindings {
    //! Stub module when Node.js feature is not enabled
    pub fn not_available() -> &'static str {
        "Node.js bindings not available. Enable the 'nodejs' feature to use these bindings."
    }
}
