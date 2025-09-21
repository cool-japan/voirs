//! WebAssembly bindings for VoiRS using wasm-bindgen.
//!
//! This module provides WebAssembly bindings that expose VoiRS functionality
//! to web browsers and other WASM runtime environments.

#[cfg(feature = "wasm")]
pub mod wasm_bindings {
    use crate::{VoirsAudioFormat, VoirsQualityLevel};
    use js_sys::{Array, Promise, Uint8Array};
    use std::sync::Arc;
    use voirs_sdk::{audio::AudioBuffer, error::VoirsError, VoirsPipeline as SdkPipeline};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{console, AudioBuffer as WebAudioBuffer, AudioContext};

    /// WASM-compatible error type
    #[wasm_bindgen]
    #[derive(Debug, Clone)]
    pub struct WasmError {
        message: String,
    }

    #[wasm_bindgen]
    impl WasmError {
        #[wasm_bindgen(getter)]
        pub fn message(&self) -> String {
            self.message.clone()
        }
    }

    impl From<VoirsError> for WasmError {
        fn from(error: VoirsError) -> Self {
            Self {
                message: error.to_string(),
            }
        }
    }

    /// WASM VoiRS Pipeline wrapper
    #[wasm_bindgen]
    pub struct VoirsPipeline {
        inner: Arc<SdkPipeline>,
    }

    /// Configuration options for creating a VoiRS pipeline
    #[wasm_bindgen]
    #[derive(Default)]
    pub struct PipelineConfig {
        use_gpu: bool,
        num_threads: u32,
        enable_streaming: bool,
        buffer_size: u32,
        cache_dir: Option<String>,
        device: Option<String>,
    }

    #[wasm_bindgen]
    impl PipelineConfig {
        #[wasm_bindgen(constructor)]
        pub fn new() -> Self {
            Self {
                use_gpu: false,
                num_threads: 4,
                enable_streaming: false,
                buffer_size: 4096,
                cache_dir: None,
                device: None,
            }
        }

        #[wasm_bindgen(setter)]
        pub fn set_use_gpu(&mut self, use_gpu: bool) {
            self.use_gpu = use_gpu;
        }

        #[wasm_bindgen(getter)]
        pub fn use_gpu(&self) -> bool {
            self.use_gpu
        }

        #[wasm_bindgen(setter)]
        pub fn set_num_threads(&mut self, num_threads: u32) {
            self.num_threads = num_threads;
        }

        #[wasm_bindgen(getter)]
        pub fn num_threads(&self) -> u32 {
            self.num_threads
        }

        #[wasm_bindgen(setter)]
        pub fn set_enable_streaming(&mut self, enable_streaming: bool) {
            self.enable_streaming = enable_streaming;
        }

        #[wasm_bindgen(getter)]
        pub fn enable_streaming(&self) -> bool {
            self.enable_streaming
        }

        #[wasm_bindgen(setter)]
        pub fn set_buffer_size(&mut self, buffer_size: u32) {
            self.buffer_size = buffer_size;
        }

        #[wasm_bindgen(getter)]
        pub fn buffer_size(&self) -> u32 {
            self.buffer_size
        }

        #[wasm_bindgen(setter)]
        pub fn set_cache_dir(&mut self, cache_dir: Option<String>) {
            self.cache_dir = cache_dir;
        }

        #[wasm_bindgen(getter)]
        pub fn cache_dir(&self) -> Option<String> {
            self.cache_dir.clone()
        }

        #[wasm_bindgen(setter)]
        pub fn set_device(&mut self, device: Option<String>) {
            self.device = device;
        }

        #[wasm_bindgen(getter)]
        pub fn device(&self) -> Option<String> {
            self.device.clone()
        }
    }

    /// Enhanced synthesis result with Web Audio API integration
    #[wasm_bindgen]
    pub struct WasmSynthesisResult {
        audio_buffer: WasmAudioBuffer,
        processing_time_ms: f64,
        real_time_factor: f64,
        audio_context: Option<AudioContext>,
    }

    #[wasm_bindgen]
    impl WasmSynthesisResult {
        #[wasm_bindgen(getter)]
        pub fn audio_buffer(&self) -> WasmAudioBuffer {
            self.audio_buffer.clone()
        }

        #[wasm_bindgen(getter)]
        pub fn processing_time_ms(&self) -> f64 {
            self.processing_time_ms
        }

        #[wasm_bindgen(getter)]
        pub fn real_time_factor(&self) -> f64 {
            self.real_time_factor
        }

        /// Convert to Web Audio API AudioBuffer for playback
        #[wasm_bindgen]
        pub fn to_web_audio_buffer(&self) -> Result<WebAudioBuffer, JsValue> {
            if let Some(ref ctx) = self.audio_context {
                let buffer = ctx.create_buffer(
                    self.audio_buffer.channels(),
                    self.audio_buffer.length(),
                    self.audio_buffer.sample_rate() as f32,
                )?;

                // Copy audio data to Web Audio buffer
                let channel_data = self.audio_buffer.get_channel_data(0);
                buffer.copy_to_channel(&channel_data, 0)?;

                Ok(buffer)
            } else {
                Err(JsValue::from_str("AudioContext not available"))
            }
        }
    }

    /// Web Workers support for background processing
    #[wasm_bindgen]
    pub struct WasmWorkerMessage {
        message_type: String,
        data: String,
        id: u32,
    }

    #[wasm_bindgen]
    impl WasmWorkerMessage {
        #[wasm_bindgen(constructor)]
        pub fn new(message_type: String, data: String, id: u32) -> Self {
            Self {
                message_type,
                data,
                id,
            }
        }

        #[wasm_bindgen(getter)]
        pub fn message_type(&self) -> String {
            self.message_type.clone()
        }

        #[wasm_bindgen(getter)]
        pub fn data(&self) -> String {
            self.data.clone()
        }

        #[wasm_bindgen(getter)]
        pub fn id(&self) -> u32 {
            self.id
        }
    }

    /// Streaming audio processor for real-time synthesis
    #[wasm_bindgen]
    pub struct WasmStreamingProcessor {
        pipeline: Arc<SdkPipeline>,
        buffer_size: usize,
        sample_rate: f32,
        channels: u32,
        current_buffer: Vec<f32>,
        audio_context: Option<AudioContext>,
    }

    #[wasm_bindgen]
    impl WasmStreamingProcessor {
        #[wasm_bindgen(constructor)]
        pub fn new(pipeline: &VoirsPipeline, buffer_size: usize, sample_rate: f32) -> Self {
            Self {
                pipeline: pipeline.inner.clone(),
                buffer_size,
                sample_rate,
                channels: 1,
                current_buffer: Vec::new(),
                audio_context: None,
            }
        }

        /// Set audio context for Web Audio API integration
        #[wasm_bindgen]
        pub fn set_audio_context(&mut self, audio_context: AudioContext) {
            self.audio_context = Some(audio_context);
        }

        /// Process a chunk of text and return audio data
        #[wasm_bindgen]
        pub async fn process_text_chunk(
            &mut self,
            text: String,
        ) -> Result<js_sys::Float32Array, JsValue> {
            let audio = self
                .pipeline
                .synthesize(&text)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let samples = audio.samples();
            let float_array = js_sys::Float32Array::new_with_length(samples.len() as u32);

            for (i, &sample) in samples.iter().enumerate() {
                float_array.set_index(i as u32, sample);
            }

            Ok(float_array)
        }

        /// Create a ScriptProcessorNode for real-time processing
        #[wasm_bindgen]
        pub fn create_processor_node(&self) -> Result<JsValue, JsValue> {
            if let Some(ref ctx) = self.audio_context {
                let processor = ctx.create_script_processor_with_buffer_size_and_number_of_input_channels_and_number_of_output_channels(
                    self.buffer_size as u32,
                    0, // No input channels
                    self.channels,
                )?;

                Ok(processor.into())
            } else {
                Err(JsValue::from_str("AudioContext not set"))
            }
        }

        /// Get optimal buffer size for current context
        #[wasm_bindgen]
        pub fn get_optimal_buffer_size(&self) -> u32 {
            if let Some(ref ctx) = self.audio_context {
                // Use AudioContext's baseLatency for optimal buffer size calculation
                let base_latency = ctx.base_latency();
                let optimal_size = (base_latency * self.sample_rate * 2.0) as u32;
                optimal_size.max(256).min(16384) // Clamp to reasonable range
            } else {
                4096 // Default buffer size
            }
        }
    }

    /// Synthesis configuration options
    #[wasm_bindgen]
    #[derive(Clone)]
    pub struct SynthesisConfig {
        speaking_rate: f32,
        pitch_shift: f32,
        volume_gain: f32,
        enable_enhancement: bool,
        output_format: String,
        sample_rate: u32,
        quality: String,
    }

    #[wasm_bindgen]
    impl SynthesisConfig {
        #[wasm_bindgen(constructor)]
        pub fn new() -> Self {
            Self {
                speaking_rate: 1.0,
                pitch_shift: 0.0,
                volume_gain: 1.0,
                enable_enhancement: true,
                output_format: "wav".to_string(),
                sample_rate: 44100,
                quality: "high".to_string(),
            }
        }

        #[wasm_bindgen(setter)]
        pub fn set_speaking_rate(&mut self, value: f32) {
            self.speaking_rate = value;
        }

        #[wasm_bindgen(getter)]
        pub fn speaking_rate(&self) -> f32 {
            self.speaking_rate
        }

        #[wasm_bindgen(setter)]
        pub fn set_pitch_shift(&mut self, value: f32) {
            self.pitch_shift = value;
        }

        #[wasm_bindgen(getter)]
        pub fn pitch_shift(&self) -> f32 {
            self.pitch_shift
        }

        #[wasm_bindgen(setter)]
        pub fn set_volume_gain(&mut self, value: f32) {
            self.volume_gain = value;
        }

        #[wasm_bindgen(getter)]
        pub fn volume_gain(&self) -> f32 {
            self.volume_gain
        }

        #[wasm_bindgen(setter)]
        pub fn set_enable_enhancement(&mut self, value: bool) {
            self.enable_enhancement = value;
        }

        #[wasm_bindgen(getter)]
        pub fn enable_enhancement(&self) -> bool {
            self.enable_enhancement
        }

        #[wasm_bindgen(setter)]
        pub fn set_sample_rate(&mut self, value: u32) {
            self.sample_rate = value;
        }

        #[wasm_bindgen(getter)]
        pub fn sample_rate(&self) -> u32 {
            self.sample_rate
        }
    }

    /// Audio buffer result for WASM
    #[wasm_bindgen]
    pub struct WasmAudioBuffer {
        samples: Vec<f32>,
        sample_rate: u32,
        channels: u32,
        duration: f32,
    }

    #[wasm_bindgen]
    impl WasmAudioBuffer {
        #[wasm_bindgen(getter)]
        pub fn sample_rate(&self) -> u32 {
            self.sample_rate
        }

        #[wasm_bindgen(getter)]
        pub fn channels(&self) -> u32 {
            self.channels
        }

        #[wasm_bindgen(getter)]
        pub fn duration(&self) -> f32 {
            self.duration
        }

        #[wasm_bindgen(getter)]
        pub fn length(&self) -> u32 {
            self.samples.len() as u32
        }

        /// Get samples as Float32Array
        #[wasm_bindgen]
        pub fn get_samples(&self) -> js_sys::Float32Array {
            js_sys::Float32Array::from(&self.samples[..])
        }

        /// Convert to Web Audio API AudioBuffer
        #[wasm_bindgen]
        pub fn to_web_audio_buffer(
            &self,
            context: &AudioContext,
        ) -> Result<WebAudioBuffer, JsValue> {
            let web_buffer = context.create_buffer(
                self.channels,
                self.samples.len() as u32 / self.channels,
                self.sample_rate as f32,
            )?;

            for channel in 0..self.channels {
                let mut channel_data = web_buffer.get_channel_data(channel)?;
                let channel_samples = self.get_channel_samples(channel as usize);
                for (i, &sample) in channel_samples.iter().enumerate() {
                    channel_data[i] = sample;
                }
            }

            Ok(web_buffer)
        }

        /// Get audio data for a specific channel
        #[wasm_bindgen]
        pub fn get_channel_data(&self, channel: u32) -> js_sys::Float32Array {
            if channel >= self.channels {
                return js_sys::Float32Array::new_with_length(0);
            }

            let channel_samples = self.get_channel_samples(channel as usize);
            js_sys::Float32Array::from(&channel_samples[..])
        }

        /// Convert to Blob for download
        #[wasm_bindgen]
        pub fn to_blob(&self, mime_type: &str) -> Result<web_sys::Blob, JsValue> {
            let mut blob_parts = js_sys::Array::new();

            match mime_type {
                "audio/wav" => {
                    let wav_data = self.to_wav_bytes();
                    let uint8_array = js_sys::Uint8Array::from(&wav_data[..]);
                    blob_parts.push(&uint8_array);
                }
                "audio/raw" => {
                    let samples_bytes: Vec<u8> =
                        self.samples.iter().flat_map(|&f| f.to_le_bytes()).collect();
                    let uint8_array = js_sys::Uint8Array::from(&samples_bytes[..]);
                    blob_parts.push(&uint8_array);
                }
                _ => return Err(JsValue::from_str("Unsupported MIME type")),
            }

            let blob_options = web_sys::BlobPropertyBag::new();
            blob_options.set_type(mime_type);

            web_sys::Blob::new_with_u8_array_sequence_and_options(&blob_parts, &blob_options)
        }

        /// Calculate audio statistics (RMS, peak, etc.)
        #[wasm_bindgen]
        pub fn calculate_statistics(&self) -> js_sys::Object {
            let stats = js_sys::Object::new();

            // Calculate RMS
            let rms = (self.samples.iter().map(|&x| x * x).sum::<f32>()
                / self.samples.len() as f32)
                .sqrt();
            js_sys::Reflect::set(
                &stats,
                &JsValue::from_str("rms"),
                &JsValue::from_f64(rms as f64),
            )
            .unwrap();

            // Calculate peak
            let peak = self.samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
            js_sys::Reflect::set(
                &stats,
                &JsValue::from_str("peak"),
                &JsValue::from_f64(peak as f64),
            )
            .unwrap();

            // Calculate zero crossing rate
            let mut crossings = 0;
            for i in 1..self.samples.len() {
                if (self.samples[i] >= 0.0) != (self.samples[i - 1] >= 0.0) {
                    crossings += 1;
                }
            }
            let zcr = crossings as f64 / (self.samples.len() - 1) as f64;
            js_sys::Reflect::set(
                &stats,
                &JsValue::from_str("zeroCrossingRate"),
                &JsValue::from_f64(zcr),
            )
            .unwrap();

            // Calculate dynamic range
            let min_val = self.samples.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = self
                .samples
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let dynamic_range = max_val - min_val;
            js_sys::Reflect::set(
                &stats,
                &JsValue::from_str("dynamicRange"),
                &JsValue::from_f64(dynamic_range as f64),
            )
            .unwrap();

            stats
        }

        /// Apply simple audio effects
        #[wasm_bindgen]
        pub fn apply_effect(
            &mut self,
            effect_type: &str,
            parameters: &js_sys::Object,
        ) -> Result<(), JsValue> {
            match effect_type {
                "gain" => {
                    let gain = js_sys::Reflect::get(parameters, &JsValue::from_str("gain"))?
                        .as_f64()
                        .unwrap_or(1.0) as f32;
                    for sample in &mut self.samples {
                        *sample *= gain;
                    }
                }
                "fade_in" => {
                    let duration = js_sys::Reflect::get(parameters, &JsValue::from_str("duration"))?
                        .as_f64()
                        .unwrap_or(1.0) as f32;
                    let fade_samples = (duration * self.sample_rate as f32) as usize;
                    let fade_samples = fade_samples.min(self.samples.len());

                    for (i, sample) in self.samples.iter_mut().enumerate().take(fade_samples) {
                        *sample *= i as f32 / fade_samples as f32;
                    }
                }
                "fade_out" => {
                    let duration = js_sys::Reflect::get(parameters, &JsValue::from_str("duration"))?
                        .as_f64()
                        .unwrap_or(1.0) as f32;
                    let fade_samples = (duration * self.sample_rate as f32) as usize;
                    let fade_samples = fade_samples.min(self.samples.len());
                    let start_fade = self.samples.len() - fade_samples;

                    for (i, sample) in self.samples.iter_mut().enumerate().skip(start_fade) {
                        *sample *= (self.samples.len() - i) as f32 / fade_samples as f32;
                    }
                }
                "normalize" => {
                    let target_level =
                        js_sys::Reflect::get(parameters, &JsValue::from_str("targetLevel"))?
                            .as_f64()
                            .unwrap_or(0.95) as f32;
                    let peak = self.samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
                    if peak > 0.0 {
                        let gain = target_level / peak;
                        for sample in &mut self.samples {
                            *sample *= gain;
                        }
                    }
                }
                _ => return Err(JsValue::from_str("Unknown effect type")),
            }
            Ok(())
        }

        /// Get samples for a specific channel
        fn get_channel_samples(&self, channel: usize) -> Vec<f32> {
            let mut channel_samples = Vec::new();
            let channels = self.channels as usize;

            for i in (channel..self.samples.len()).step_by(channels) {
                channel_samples.push(self.samples[i]);
            }

            channel_samples
        }
    }

    impl From<AudioBuffer> for WasmAudioBuffer {
        fn from(audio: AudioBuffer) -> Self {
            Self {
                samples: audio.samples().to_vec(),
                sample_rate: audio.sample_rate(),
                channels: audio.channels(),
                duration: audio.duration(),
            }
        }
    }

    /// Voice information for WASM
    #[wasm_bindgen]
    #[derive(Clone)]
    pub struct VoiceInfo {
        id: String,
        name: String,
        language: String,
        quality: String,
        is_available: bool,
    }

    #[wasm_bindgen]
    impl VoiceInfo {
        #[wasm_bindgen(getter)]
        pub fn id(&self) -> String {
            self.id.clone()
        }

        #[wasm_bindgen(getter)]
        pub fn name(&self) -> String {
            self.name.clone()
        }

        #[wasm_bindgen(getter)]
        pub fn language(&self) -> String {
            self.language.clone()
        }

        #[wasm_bindgen(getter)]
        pub fn quality(&self) -> String {
            self.quality.clone()
        }

        #[wasm_bindgen(getter)]
        pub fn is_available(&self) -> bool {
            self.is_available
        }
    }

    #[wasm_bindgen]
    impl VoirsPipeline {
        /// Create a new VoiRS pipeline
        #[wasm_bindgen(constructor)]
        pub async fn new(config: Option<PipelineConfig>) -> Result<VoirsPipeline, WasmError> {
            console::log_1(&"Creating VoiRS pipeline for WASM...".into());

            let mut builder = SdkPipeline::builder();

            if let Some(cfg) = config {
                if cfg.use_gpu {
                    builder = builder.with_gpu(true);
                }
                if cfg.num_threads > 0 {
                    builder = builder.with_threads(cfg.num_threads as usize);
                }
                if let Some(cache_dir) = cfg.cache_dir {
                    builder = builder.with_cache_dir(&cache_dir);
                }
                if let Some(device) = cfg.device {
                    builder = builder.with_device(device);
                }
            }

            let pipeline = builder.build().await.map_err(WasmError::from)?;

            Ok(Self {
                inner: Arc::new(pipeline),
            })
        }

        /// Synthesize text to audio
        #[wasm_bindgen]
        pub async fn synthesize(&self, text: &str) -> Result<WasmAudioBuffer, WasmError> {
            console::log_1(&format!("Synthesizing text: {}", text).into());

            let audio = self.inner.synthesize(text).await.map_err(WasmError::from)?;

            Ok(WasmAudioBuffer::from(audio))
        }

        /// Synthesize text with custom configuration
        #[wasm_bindgen]
        pub async fn synthesize_with_config(
            &self,
            text: &str,
            config: &SynthesisConfig,
        ) -> Result<WasmAudioBuffer, WasmError> {
            console::log_1(&format!("Synthesizing with config: {}", text).into());

            // Convert WASM config to SDK config
            let sdk_config = voirs_sdk::types::SynthesisConfig {
                speaking_rate: config.speaking_rate,
                pitch_shift: config.pitch_shift,
                volume_gain: config.volume_gain,
                enable_enhancement: config.enable_enhancement,
                output_format: match config.output_format.as_str() {
                    "wav" => voirs_sdk::types::AudioFormat::Wav,
                    "flac" => voirs_sdk::types::AudioFormat::Flac,
                    "mp3" => voirs_sdk::types::AudioFormat::Mp3,
                    _ => voirs_sdk::types::AudioFormat::Wav,
                },
                sample_rate: config.sample_rate,
                quality: match config.quality.as_str() {
                    "low" => voirs_sdk::types::QualityLevel::Low,
                    "medium" => voirs_sdk::types::QualityLevel::Medium,
                    "high" => voirs_sdk::types::QualityLevel::High,
                    "ultra" => voirs_sdk::types::QualityLevel::Ultra,
                    _ => voirs_sdk::types::QualityLevel::High,
                },
                language: voirs_sdk::types::LanguageCode::EnUs, // Default
                effects: Vec::new(),
                streaming_chunk_size: None,
                seed: None,
            };

            let audio = self
                .inner
                .synthesize_with_config(text, &sdk_config)
                .await
                .map_err(WasmError::from)?;

            Ok(WasmAudioBuffer::from(audio))
        }

        /// Set the voice for synthesis
        #[wasm_bindgen]
        pub async fn set_voice(&self, voice_id: &str) -> Result<(), WasmError> {
            self.inner
                .set_voice(voice_id)
                .await
                .map_err(WasmError::from)?;
            Ok(())
        }

        /// Get the current voice
        #[wasm_bindgen]
        pub async fn get_voice(&self) -> Option<String> {
            self.inner.current_voice().await.map(|v| v.id)
        }

        /// List available voices
        #[wasm_bindgen]
        pub async fn list_voices(&self) -> Result<Array, WasmError> {
            let voices = self.inner.list_voices().await.map_err(WasmError::from)?;

            let js_voices = Array::new();
            for voice in voices {
                let voice_info = VoiceInfo {
                    id: voice.id,
                    name: voice.name,
                    language: voice.language.to_string(),
                    quality: format!("{:?}", voice.characteristics.quality),
                    is_available: true, // Always available for now
                };
                js_voices.push(&JsValue::from(voice_info));
            }

            Ok(js_voices)
        }

        /// Get pipeline information
        #[wasm_bindgen]
        pub fn get_info(&self) -> js_sys::Object {
            let info = js_sys::Object::new();
            js_sys::Reflect::set(&info, &"version".into(), &env!("CARGO_PKG_VERSION").into())
                .unwrap();
            js_sys::Reflect::set(&info, &"platform".into(), &"wasm".into()).unwrap();
            js_sys::Reflect::set(&info, &"features".into(), &{
                let features = js_sys::Object::new();
                js_sys::Reflect::set(
                    &features,
                    &"gpu_support".into(),
                    &cfg!(feature = "gpu").into(),
                )
                .unwrap();
                js_sys::Reflect::set(
                    &features,
                    &"python_bindings".into(),
                    &cfg!(feature = "python").into(),
                )
                .unwrap();
                js_sys::Reflect::set(
                    &features,
                    &"nodejs_bindings".into(),
                    &cfg!(feature = "nodejs").into(),
                )
                .unwrap();
                js_sys::Reflect::set(&features, &"wasm_bindings".into(), &true.into()).unwrap();
                features.into()
            })
            .unwrap();
            info
        }
    }

    /// Initialize VoiRS for WASM
    #[wasm_bindgen(start)]
    pub fn init() {
        console::log_1(&"VoiRS WASM module initialized".into());

        // Set panic hook for better error messages in browser console
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    }

    /// Get the library version
    #[wasm_bindgen]
    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Check if feature is available
    #[wasm_bindgen]
    pub fn has_feature(feature: &str) -> bool {
        match feature {
            "gpu" => cfg!(feature = "gpu"),
            "python" => cfg!(feature = "python"),
            "nodejs" => cfg!(feature = "nodejs"),
            "wasm" => true,
            _ => false,
        }
    }
}

#[cfg(not(feature = "wasm"))]
pub mod wasm_bindings {
    //! Stub module when WASM feature is not enabled
    pub fn not_available() -> &'static str {
        "WebAssembly bindings not available. Enable the 'wasm' feature to use these bindings."
    }
}
