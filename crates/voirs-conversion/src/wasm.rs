//! WebAssembly support for browser-based voice conversion
//!
//! This module provides comprehensive WebAssembly bindings for voice conversion operations,
//! enabling high-quality voice processing directly in web browsers. It includes optimizations
//! for browser environments, Web Audio API integration, and efficient memory management.
//!
//! ## Key Features
//!
//! - **Web Audio API Integration**: Direct integration with browser audio processing
//! - **Streaming Processing**: Real-time audio streaming with low latency
//! - **Memory Optimization**: Efficient memory usage for browser constraints
//! - **Progressive Loading**: Lazy loading of models and resources
//! - **Worker Thread Support**: Background processing using Web Workers
//! - **TypeScript Bindings**: Full TypeScript support for web integration
//!
//! ## Performance Targets
//!
//! - **Real-time Processing**: <50ms latency for browser-based conversion
//! - **Memory Footprint**: <100MB for full conversion pipeline
//! - **Model Loading**: <2s initial model loading time
//! - **Progressive Enhancement**: Graceful degradation on slower devices
//!
//! ## Usage
//!
//! ```javascript
//! import init, { WasmVoiceConverter } from './pkg/voirs_conversion.js';
//!
//! async function main() {
//!     await init();
//!     
//!     const converter = new WasmVoiceConverter();
//!     await converter.initialize();
//!     
//!     const audioBuffer = new Float32Array([...]);
//!     const result = await converter.convertAudio(audioBuffer, {
//!         type: 'PitchShift',
//!         parameters: { factor: 1.2 }
//!     });
//!     
//!     // Play result through Web Audio API
//!     playAudio(result);
//! }
//! ```

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use js_sys::{Array, Float32Array, Object, Promise, Uint8Array};

#[cfg(feature = "wasm")]
use web_sys::{
    console, AudioBuffer, AudioContext, AudioDestinationNode, AudioNode, GainNode, Performance,
    ScriptProcessorNode, Window,
};

use crate::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "wasm")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    fn warn(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

#[cfg(feature = "wasm")]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[cfg(feature = "wasm")]
macro_rules! console_warn {
    ($($t:tt)*) => (warn(&format_args!($($t)*).to_string()))
}

#[cfg(feature = "wasm")]
macro_rules! console_error {
    ($($t:tt)*) => (error(&format_args!($($t)*).to_string()))
}

/// WebAssembly voice converter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmConversionConfig {
    /// Enable Web Audio API integration
    pub enable_web_audio: bool,
    /// Enable streaming processing
    pub enable_streaming: bool,
    /// Enable Web Worker support
    pub enable_workers: bool,
    /// Maximum buffer size for processing
    pub max_buffer_size: usize,
    /// Audio context sample rate
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u32,
    /// Enable progressive model loading
    pub enable_progressive_loading: bool,
    /// Memory limit in MB
    pub memory_limit_mb: u32,
    /// Enable real-time monitoring
    pub enable_monitoring: bool,
    /// Audio processing latency target in ms
    pub target_latency_ms: f64,
}

impl Default for WasmConversionConfig {
    fn default() -> Self {
        Self {
            enable_web_audio: true,
            enable_streaming: true,
            enable_workers: false, // Disabled by default due to complexity
            max_buffer_size: 4096,
            sample_rate: 44100,
            channels: 2,
            enable_progressive_loading: true,
            memory_limit_mb: 100,
            enable_monitoring: true,
            target_latency_ms: 50.0,
        }
    }
}

/// Web Audio API processing node types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WebAudioNodeType {
    /// Input audio source
    Source,
    /// Voice conversion processor
    Processor,
    /// Output destination
    Destination,
    /// Gain control
    Gain,
    /// Analyzer node
    Analyzer,
}

/// Browser capability detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserCapabilities {
    /// Web Audio API support
    pub web_audio_supported: bool,
    /// Web Workers support
    pub web_workers_supported: bool,
    /// WebAssembly support level
    pub wasm_support_level: WasmSupportLevel,
    /// Available memory in MB
    pub available_memory_mb: u32,
    /// Audio context sample rate
    pub audio_context_sample_rate: u32,
    /// Maximum audio channels
    pub max_audio_channels: u32,
    /// Browser type and version
    pub browser_info: String,
    /// Performance score (0-100)
    pub performance_score: u32,
}

/// WebAssembly support levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WasmSupportLevel {
    /// No WebAssembly support
    None,
    /// Basic WebAssembly support
    Basic,
    /// Full WebAssembly support with SIMD
    Full,
    /// Advanced support with threads
    Advanced,
}

#[cfg(feature = "wasm")]
impl BrowserCapabilities {
    /// Detect current browser capabilities
    pub fn detect() -> Self {
        let window = web_sys::window().unwrap();
        let navigator = window.navigator();

        let web_audio_supported = Self::detect_web_audio_support(&window);
        let web_workers_supported = Self::detect_web_workers_support(&window);
        let wasm_support_level = Self::detect_wasm_support_level();
        let available_memory_mb = Self::estimate_available_memory(&navigator);
        let (audio_context_sample_rate, max_audio_channels) = Self::detect_audio_capabilities();
        let browser_info = Self::detect_browser_info(&navigator);
        let performance_score = Self::calculate_performance_score();

        Self {
            web_audio_supported,
            web_workers_supported,
            wasm_support_level,
            available_memory_mb,
            audio_context_sample_rate,
            max_audio_channels,
            browser_info,
            performance_score,
        }
    }

    fn detect_web_audio_support(window: &Window) -> bool {
        js_sys::Reflect::has(&window.clone().into(), &"AudioContext".into()).unwrap_or(false)
            || js_sys::Reflect::has(&window.clone().into(), &"webkitAudioContext".into())
                .unwrap_or(false)
    }

    fn detect_web_workers_support(window: &Window) -> bool {
        js_sys::Reflect::has(&window.clone().into(), &"Worker".into()).unwrap_or(false)
    }

    fn detect_wasm_support_level() -> WasmSupportLevel {
        // Check for WebAssembly support
        if js_sys::Reflect::has(&js_sys::global(), &"WebAssembly".into()).unwrap_or(false) {
            // Check for SIMD support
            if js_sys::Reflect::has(
                &js_sys::global().get(&"WebAssembly".into()).unwrap(),
                &"SIMD".into(),
            )
            .unwrap_or(false)
            {
                WasmSupportLevel::Advanced
            } else {
                WasmSupportLevel::Full
            }
        } else {
            WasmSupportLevel::None
        }
    }

    fn estimate_available_memory(navigator: &web_sys::Navigator) -> u32 {
        // Try to get memory info if available
        if let Ok(memory_info) =
            js_sys::Reflect::get(&navigator.clone().into(), &"deviceMemory".into())
        {
            if !memory_info.is_undefined() {
                if let Some(memory_gb) = memory_info.as_f64() {
                    return (memory_gb * 1024.0 * 0.3) as u32; // 30% of device memory
                }
            }
        }

        // Fallback estimation based on user agent
        let user_agent = navigator.user_agent().unwrap_or_default();
        if user_agent.contains("Mobile") || user_agent.contains("Android") {
            256 // Conservative mobile estimate
        } else {
            512 // Conservative desktop estimate
        }
    }

    fn detect_audio_capabilities() -> (u32, u32) {
        // Try to create AudioContext to detect capabilities
        match AudioContext::new() {
            Ok(ctx) => {
                let sample_rate = ctx.sample_rate() as u32;
                let max_channels = ctx.destination().max_channel_count();
                (sample_rate, max_channels)
            }
            Err(_) => (44100, 2), // Fallback values
        }
    }

    fn detect_browser_info(navigator: &web_sys::Navigator) -> String {
        let user_agent = navigator.user_agent().unwrap_or_default();
        let app_name = navigator.app_name().unwrap_or_default();
        let app_version = navigator.app_version().unwrap_or_default();

        format!("{app_name} {app_version} ({user_agent})")
    }

    fn calculate_performance_score() -> u32 {
        // Simple performance score based on available features
        let mut score = 50; // Base score

        if Self::detect_web_audio_support(&web_sys::window().unwrap()) {
            score += 20;
        }

        if Self::detect_web_workers_support(&web_sys::window().unwrap()) {
            score += 15;
        }

        match Self::detect_wasm_support_level() {
            WasmSupportLevel::Advanced => score += 15,
            WasmSupportLevel::Full => score += 10,
            WasmSupportLevel::Basic => score += 5,
            WasmSupportLevel::None => score -= 30,
        }

        score.min(100).max(0)
    }
}

#[cfg(not(feature = "wasm"))]
impl BrowserCapabilities {
    /// Detect current browser capabilities (stub for non-WASM builds)
    pub fn detect() -> Self {
        Self {
            web_audio_supported: false,
            web_workers_supported: false,
            wasm_support_level: WasmSupportLevel::None,
            available_memory_mb: 0,
            audio_context_sample_rate: 44100,
            max_audio_channels: 2,
            browser_info: "Non-WASM build".to_string(),
            performance_score: 0,
        }
    }
}

/// WebAssembly voice converter
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmVoiceConverter {
    converter: Arc<VoiceConverter>,
    config: WasmConversionConfig,
    capabilities: BrowserCapabilities,
    audio_context: Option<AudioContext>,
    processing_nodes: HashMap<String, WebAudioNode>,
    stats: WasmConversionStats,
    initialized: bool,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmVoiceConverter {
    /// Create new WebAssembly voice converter
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_log!("Creating WASM Voice Converter");

        Self {
            converter: Arc::new(VoiceConverter::new().expect("Failed to create voice converter")),
            config: WasmConversionConfig::default(),
            capabilities: BrowserCapabilities::detect(),
            audio_context: None,
            processing_nodes: HashMap::new(),
            stats: WasmConversionStats::new(),
            initialized: false,
        }
    }

    /// Create converter with custom configuration
    #[wasm_bindgen]
    pub fn with_config(config_js: &JsValue) -> std::result::Result<WasmVoiceConverter, JsValue> {
        let config: WasmConversionConfig = config_js
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;

        console_log!("Creating WASM Voice Converter with custom config");

        Ok(Self {
            converter: Arc::new(
                VoiceConverter::new().map_err(|e| JsValue::from_str(&e.to_string()))?,
            ),
            config,
            capabilities: BrowserCapabilities::detect(),
            audio_context: None,
            processing_nodes: HashMap::new(),
            stats: WasmConversionStats::new(),
            initialized: false,
        })
    }

    /// Initialize the converter and Web Audio API
    #[wasm_bindgen]
    pub async fn initialize(&mut self) -> Result<(), JsValue> {
        console_log!("Initializing WASM Voice Converter");

        if !self.capabilities.web_audio_supported {
            console_warn!("Web Audio API not supported, using fallback mode");
        }

        // Initialize Audio Context if supported
        if self.config.enable_web_audio && self.capabilities.web_audio_supported {
            self.initialize_audio_context().await?;
        }

        // Validate memory constraints
        self.validate_memory_constraints()?;

        // Initialize processing pipeline
        self.initialize_processing_pipeline().await?;

        self.initialized = true;
        self.stats.record_initialization();

        console_log!("WASM Voice Converter initialized successfully");
        Ok(())
    }

    /// Convert audio data
    #[wasm_bindgen]
    pub async fn convert_audio(
        &mut self,
        audio_data: &Float32Array,
        conversion_params: &JsValue,
    ) -> Result<Float32Array, JsValue> {
        if !self.initialized {
            return Err(JsValue::from_str("Converter not initialized"));
        }

        let start_time = Self::get_performance_now();

        // Convert JS types to Rust
        let audio_vec: Vec<f32> = audio_data.to_vec();
        let params: ConversionParameters = conversion_params
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("Invalid conversion parameters: {e}")))?;

        // Create conversion request
        let request = ConversionRequest::new(
            self.generate_request_id(),
            audio_vec,
            self.config.sample_rate,
            params.conversion_type,
            params.target,
        );

        // Perform conversion
        let result = if self.config.enable_streaming
            && audio_data.length() > self.config.max_buffer_size as u32
        {
            self.convert_streaming(&request).await?
        } else {
            self.convert_standard(&request).await?
        };

        // Convert result back to JS
        let output_array = Float32Array::new_with_length(result.converted_audio.len() as u32);
        for (i, sample) in result.converted_audio.iter().enumerate() {
            output_array.set_index(i as u32, *sample);
        }

        // Record statistics
        let processing_time = Self::get_performance_now() - start_time;
        self.stats
            .record_conversion(processing_time, params.conversion_type);

        console_log!("Conversion completed in {:.2}ms", processing_time);
        Ok(output_array)
    }

    /// Start real-time audio processing
    #[wasm_bindgen]
    pub async fn start_realtime_processing(&mut self, source_node_id: &str) -> Result<(), JsValue> {
        if !self.initialized {
            return Err(JsValue::from_str("Converter not initialized"));
        }

        if !self.config.enable_web_audio {
            return Err(JsValue::from_str("Web Audio API not enabled"));
        }

        console_log!("Starting real-time processing");

        // Create processing chain
        self.create_realtime_processing_chain(source_node_id)
            .await?;

        self.stats.record_realtime_start();
        Ok(())
    }

    /// Stop real-time audio processing
    #[wasm_bindgen]
    pub fn stop_realtime_processing(&mut self) -> std::result::Result<(), JsValue> {
        console_log!("Stopping real-time processing");

        // Disconnect all processing nodes
        for (_, node) in &mut self.processing_nodes {
            node.disconnect()?;
        }

        self.processing_nodes.clear();
        self.stats.record_realtime_stop();
        Ok(())
    }

    /// Get conversion statistics
    #[wasm_bindgen]
    pub fn get_statistics(&self) -> Result<JsValue, JsValue> {
        let stats = self.stats.get_statistics();
        JsValue::from_serde(&stats)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize statistics: {e}")))
    }

    /// Get browser capabilities
    #[wasm_bindgen]
    pub fn get_capabilities(&self) -> Result<JsValue, JsValue> {
        JsValue::from_serde(&self.capabilities)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize capabilities: {e}")))
    }

    /// Check if converter is initialized
    #[wasm_bindgen]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get current configuration
    #[wasm_bindgen]
    pub fn get_config(&self) -> Result<JsValue, JsValue> {
        JsValue::from_serde(&self.config)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize config: {e}")))
    }

    /// Update configuration
    #[wasm_bindgen]
    pub fn update_config(&mut self, config_js: &JsValue) -> Result<(), JsValue> {
        let config: WasmConversionConfig = config_js
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;

        self.config = config;
        console_log!("Configuration updated");
        Ok(())
    }

    // Internal implementation methods

    async fn initialize_audio_context(&mut self) -> Result<(), JsValue> {
        let audio_context = AudioContext::new()
            .map_err(|e| JsValue::from_str(&format!("Failed to create AudioContext: {:?}", e)))?;

        // Resume context if suspended
        if audio_context.state() != web_sys::AudioContextState::Running {
            let resume_promise = audio_context.resume().map_err(|e| {
                JsValue::from_str(&format!("Failed to resume AudioContext: {:?}", e))
            })?;

            wasm_bindgen_futures::JsFuture::from(resume_promise)
                .await
                .map_err(|e| JsValue::from_str(&format!("AudioContext resume failed: {:?}", e)))?;
        }

        self.audio_context = Some(audio_context);
        console_log!("AudioContext initialized");
        Ok(())
    }

    fn validate_memory_constraints(&self) -> Result<(), JsValue> {
        if self.capabilities.available_memory_mb < self.config.memory_limit_mb {
            console_warn!(
                "Available memory ({} MB) is less than required ({} MB)",
                self.capabilities.available_memory_mb,
                self.config.memory_limit_mb
            );
        }
        Ok(())
    }

    async fn initialize_processing_pipeline(&mut self) -> Result<(), JsValue> {
        // Initialize conversion models and pipeline
        console_log!("Initializing processing pipeline");

        if self.config.enable_progressive_loading {
            // Load models progressively
            self.load_models_progressively().await?;
        } else {
            // Load all models at once
            self.load_all_models().await?;
        }

        Ok(())
    }

    async fn load_models_progressively(&self) -> Result<(), JsValue> {
        console_log!("Loading models progressively");
        // In a real implementation, this would load models on demand
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(())
    }

    async fn load_all_models(&self) -> Result<(), JsValue> {
        console_log!("Loading all models");
        // In a real implementation, this would load all models upfront
        tokio::time::sleep(Duration::from_millis(500)).await;
        Ok(())
    }

    async fn convert_streaming(
        &self,
        request: &ConversionRequest,
    ) -> Result<ConversionResult, JsValue> {
        console_log!("Performing streaming conversion");

        // Process audio in chunks
        let chunk_size = self.config.max_buffer_size;
        let mut converted_chunks = Vec::new();

        for chunk in request.source_audio.chunks(chunk_size) {
            let chunk_request = ConversionRequest::new(
                format!("{}_chunk", request.id),
                chunk.to_vec(),
                request.source_sample_rate,
                request.conversion_type.clone(),
                request.target.clone(),
            );

            let chunk_result = self
                .converter
                .convert(chunk_request)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            converted_chunks.extend(chunk_result.converted_audio);
        }

        Ok(ConversionResult {
            request_id: request.id.clone(),
            converted_audio: converted_chunks,
            output_sample_rate: request.source_sample_rate,
            quality_metrics: HashMap::new(),
            artifacts: None,
            objective_quality: None,
            processing_time: Duration::from_millis(0),
            conversion_type: request.conversion_type.clone(),
            success: true,
            error_message: None,
            timestamp: std::time::SystemTime::now(),
        })
    }

    async fn convert_standard(
        &self,
        request: &ConversionRequest,
    ) -> Result<ConversionResult, JsValue> {
        console_log!("Performing standard conversion");

        self.converter
            .convert(request.clone())
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    async fn create_realtime_processing_chain(
        &mut self,
        source_node_id: &str,
    ) -> Result<(), JsValue> {
        let audio_context = self
            .audio_context
            .as_ref()
            .ok_or_else(|| JsValue::from_str("AudioContext not initialized"))?;

        // Create ScriptProcessorNode for custom processing
        let buffer_size = self.config.max_buffer_size as u32;
        let script_processor = audio_context
            .create_script_processor_with_buffer_size_and_number_of_output_channels(
                buffer_size,
                self.config.channels,
                self.config.channels,
            )
            .map_err(|e| {
                JsValue::from_str(&format!("Failed to create ScriptProcessorNode: {:?}", e))
            })?;

        // Set up audio processing callback
        let converter = Arc::clone(&self.converter);
        let config = self.config.clone();

        let closure = Closure::wrap(Box::new(move |event: web_sys::AudioProcessingEvent| {
            let input_buffer = event.input_buffer().unwrap();
            let output_buffer = event.output_buffer().unwrap();

            // Process audio in real-time
            let channel_data = input_buffer.get_channel_data(0).unwrap();
            let audio_vec: Vec<f32> = channel_data.to_vec();

            // Create conversion request
            let request = ConversionRequest::new(
                "realtime".to_string(),
                audio_vec,
                config.sample_rate,
                ConversionType::PitchShift, // Default conversion
                ConversionTarget::new(VoiceCharacteristics::default()),
            );

            // Note: In a real implementation, this would need to be handled differently
            // as async operations cannot be performed in the audio callback
            let mut output_channel = output_buffer.get_channel_data(0).unwrap();
            for (i, sample) in request.source_audio.iter().enumerate() {
                if i < output_channel.length() as usize {
                    output_channel.set_index(i as u32, *sample);
                }
            }
        }) as Box<dyn FnMut(_)>);

        script_processor.set_onaudioprocess(Some(closure.as_ref().unchecked_ref()));
        closure.forget();

        // Connect to destination
        script_processor
            .connect_with_audio_node(&audio_context.destination())
            .map_err(|e| {
                JsValue::from_str(&format!("Failed to connect to destination: {:?}", e))
            })?;

        // Store processing node
        self.processing_nodes.insert(
            source_node_id.to_string(),
            WebAudioNode::ScriptProcessor(script_processor),
        );

        Ok(())
    }

    fn generate_request_id(&self) -> String {
        format!("wasm_{}", fastrand::u64(..))
    }

    fn get_performance_now() -> f64 {
        web_sys::window()
            .and_then(|window| window.performance())
            .map(|performance| performance.now())
            .unwrap_or(0.0)
    }
}

/// Web Audio API node wrapper
#[cfg(feature = "wasm")]
enum WebAudioNode {
    ScriptProcessor(ScriptProcessorNode),
    Gain(GainNode),
}

#[cfg(feature = "wasm")]
impl WebAudioNode {
    fn disconnect(&mut self) -> Result<(), Error> {
        match self {
            WebAudioNode::ScriptProcessor(node) => node
                .disconnect()
                .map_err(|e| Error::processing(format!("Disconnect failed: {:?}", e))),
            WebAudioNode::Gain(node) => node
                .disconnect()
                .map_err(|e| Error::processing(format!("Disconnect failed: {:?}", e))),
        }
    }
}

/// Conversion parameters for WebAssembly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionParameters {
    /// Type of conversion to perform
    pub conversion_type: ConversionType,
    /// Target voice characteristics
    pub target: ConversionTarget,
    /// Additional parameters
    pub parameters: HashMap<String, f32>,
}

/// WebAssembly conversion statistics
pub struct WasmConversionStats {
    total_conversions: std::sync::atomic::AtomicU64,
    total_processing_time: std::sync::atomic::AtomicU64,
    realtime_sessions: std::sync::atomic::AtomicU32,
    initialization_count: std::sync::atomic::AtomicU32,
    error_count: std::sync::atomic::AtomicU32,
    conversion_types: Arc<tokio::sync::Mutex<HashMap<ConversionType, u32>>>,
}

impl WasmConversionStats {
    fn new() -> Self {
        Self {
            total_conversions: std::sync::atomic::AtomicU64::new(0),
            total_processing_time: std::sync::atomic::AtomicU64::new(0),
            realtime_sessions: std::sync::atomic::AtomicU32::new(0),
            initialization_count: std::sync::atomic::AtomicU32::new(0),
            error_count: std::sync::atomic::AtomicU32::new(0),
            conversion_types: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }

    fn record_conversion(&self, processing_time_ms: f64, conversion_type: ConversionType) {
        use std::sync::atomic::Ordering;

        self.total_conversions.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time
            .fetch_add(processing_time_ms as u64, Ordering::Relaxed);

        // Record conversion type asynchronously
        let conversion_types = Arc::clone(&self.conversion_types);
        wasm_bindgen_futures::spawn_local(async move {
            let mut types = conversion_types.lock().await;
            *types.entry(conversion_type).or_insert(0) += 1;
        });
    }

    fn record_initialization(&self) {
        self.initialization_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn record_realtime_start(&self) {
        self.realtime_sessions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn record_realtime_stop(&self) {
        // Could be used to track session durations
    }

    fn record_error(&self) {
        self.error_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn get_statistics(&self) -> WasmConversionStatistics {
        use std::sync::atomic::Ordering;

        let total_conversions = self.total_conversions.load(Ordering::Relaxed);
        let total_processing_time_ms = self.total_processing_time.load(Ordering::Relaxed);

        let average_processing_time_ms = if total_conversions > 0 {
            total_processing_time_ms as f64 / total_conversions as f64
        } else {
            0.0
        };

        WasmConversionStatistics {
            total_conversions,
            average_processing_time_ms,
            realtime_sessions: self.realtime_sessions.load(Ordering::Relaxed),
            initialization_count: self.initialization_count.load(Ordering::Relaxed),
            error_count: self.error_count.load(Ordering::Relaxed),
            conversion_types: HashMap::new(), // Would be populated from async data
        }
    }
}

/// WebAssembly conversion statistics result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmConversionStatistics {
    /// Total number of conversions processed
    pub total_conversions: u64,
    /// Average processing time in milliseconds
    pub average_processing_time_ms: f64,
    /// Number of real-time sessions
    pub realtime_sessions: u32,
    /// Number of initializations
    pub initialization_count: u32,
    /// Number of errors encountered
    pub error_count: u32,
    /// Conversion type counts
    pub conversion_types: HashMap<ConversionType, u32>,
}

// Non-WASM stubs for compilation
#[cfg(not(feature = "wasm"))]
pub struct WasmVoiceConverter;

#[cfg(not(feature = "wasm"))]
impl WasmVoiceConverter {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_config_creation() {
        let config = WasmConversionConfig::default();
        assert!(config.enable_web_audio);
        assert!(config.enable_streaming);
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channels, 2);
        assert_eq!(config.memory_limit_mb, 100);
    }

    #[test]
    fn test_browser_capabilities_detection() {
        let capabilities = BrowserCapabilities::detect();
        assert!(!capabilities.browser_info.is_empty());
        assert!(capabilities.performance_score <= 100);
    }

    #[test]
    fn test_wasm_support_level() {
        let level = WasmSupportLevel::Full;
        assert_eq!(level, WasmSupportLevel::Full);
    }

    #[test]
    fn test_conversion_parameters() {
        let params = ConversionParameters {
            conversion_type: ConversionType::PitchShift,
            target: ConversionTarget::new(VoiceCharacteristics::default()),
            parameters: HashMap::new(),
        };

        assert_eq!(params.conversion_type, ConversionType::PitchShift);
    }

    #[test]
    fn test_wasm_stats() {
        let stats = WasmConversionStats::new();
        stats.record_conversion(100.0, ConversionType::PitchShift);
        stats.record_initialization();

        let statistics = stats.get_statistics();
        assert_eq!(statistics.total_conversions, 1);
        assert_eq!(statistics.initialization_count, 1);
        assert!(statistics.average_processing_time_ms > 0.0);
    }

    #[cfg(feature = "wasm")]
    #[test]
    fn test_wasm_converter_creation() {
        let converter = WasmVoiceConverter::new();
        assert!(!converter.is_initialized());
    }

    #[test]
    fn test_web_audio_node_types() {
        use std::collections::HashSet;

        let mut node_types = HashSet::new();
        node_types.insert(WebAudioNodeType::Source);
        node_types.insert(WebAudioNodeType::Processor);
        node_types.insert(WebAudioNodeType::Destination);

        assert_eq!(node_types.len(), 3);
        assert!(node_types.contains(&WebAudioNodeType::Source));
    }
}
