//! WebXR Integration for VoiRS Spatial Audio
//!
//! This module provides browser-based immersive audio experiences for VR/AR applications
//! running in web browsers through WebXR APIs. It includes WASM bindings, Web Audio API
//! integration, and browser-specific optimizations.

use crate::config::SpatialConfig;
use crate::types::{AudioChannel, Position3D};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// WebXR session types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WebXRSessionType {
    /// Immersive VR session
    ImmersiveVr,
    /// Immersive AR session
    ImmersiveAr,
    /// Inline session (fallback)
    Inline,
}

/// Web browser types for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BrowserType {
    /// Chrome/Chromium-based browsers
    Chrome,
    /// Firefox
    Firefox,
    /// Safari
    Safari,
    /// Edge
    Edge,
    /// Other browsers
    Other,
}

/// WebXR configuration for spatial audio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebXRConfig {
    /// Session type preference
    pub session_type: WebXRSessionType,
    /// Target browser
    pub browser: BrowserType,
    /// Use Web Audio API worklets
    pub use_worklets: bool,
    /// Use SharedArrayBuffer if available
    pub use_shared_buffer: bool,
    /// Target latency for web processing (ms)
    pub target_latency_ms: f32,
    /// Buffer size for web audio
    pub buffer_size: usize,
    /// Enable browser-specific optimizations
    pub browser_optimizations: bool,
    /// Maximum concurrent audio sources
    pub max_sources: usize,
    /// Use offscreen canvas for processing
    pub use_offscreen_canvas: bool,
    /// Enable spatial audio in WebXR
    pub enable_spatial_audio: bool,
    /// Quality level for web processing
    pub quality_level: f32,
}

impl Default for WebXRConfig {
    fn default() -> Self {
        Self {
            session_type: WebXRSessionType::ImmersiveVr,
            browser: BrowserType::Chrome,
            use_worklets: true,
            use_shared_buffer: false, // Often not available due to security
            target_latency_ms: 128.0, // Higher latency for web
            buffer_size: 4096,        // Larger buffer for web
            browser_optimizations: true,
            max_sources: 16, // Limited for web performance
            use_offscreen_canvas: true,
            enable_spatial_audio: true,
            quality_level: 0.6, // Moderate quality for web
        }
    }
}

/// WebXR device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebXRCapabilities {
    /// Supports immersive VR
    pub supports_vr: bool,
    /// Supports immersive AR
    pub supports_ar: bool,
    /// Has WebGL support
    pub has_webgl: bool,
    /// Has WebGL2 support
    pub has_webgl2: bool,
    /// Has WebGPU support
    pub has_webgpu: bool,
    /// Supports Web Audio API
    pub has_web_audio: bool,
    /// Supports AudioWorklet
    pub has_audio_worklet: bool,
    /// Supports SharedArrayBuffer
    pub has_shared_array_buffer: bool,
    /// Maximum audio context sample rate
    pub max_sample_rate: f32,
    /// Available audio context states
    pub audio_context_states: Vec<String>,
}

impl Default for WebXRCapabilities {
    fn default() -> Self {
        Self {
            supports_vr: false,
            supports_ar: false,
            has_webgl: true,
            has_webgl2: false,
            has_webgpu: false,
            has_web_audio: true,
            has_audio_worklet: false,
            has_shared_array_buffer: false,
            max_sample_rate: 48000.0,
            audio_context_states: vec!["suspended".to_string()],
        }
    }
}

/// WebXR pose data from browser
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebXRPose {
    /// Position in 3D space
    pub position: Position3D,
    /// Orientation quaternion (x, y, z, w)
    pub orientation: (f32, f32, f32, f32),
    /// Linear velocity (if available)
    pub linear_velocity: Option<Position3D>,
    /// Angular velocity (if available)
    pub angular_velocity: Option<Position3D>,
    /// Tracking confidence
    pub confidence: f32,
    /// Timestamp from browser
    pub timestamp: f64,
}

/// WebXR audio source for web processing
#[derive(Debug, Clone)]
pub struct WebXRAudioSource {
    /// Unique source ID
    pub id: String,
    /// Source position
    pub position: Position3D,
    /// Audio buffer
    pub buffer: Vec<f32>,
    /// Source type
    pub source_type: WebXRSourceType,
    /// Volume level
    pub volume: f32,
    /// Is currently playing
    pub playing: bool,
    /// Loop the audio
    pub looping: bool,
}

/// WebXR audio source types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebXRSourceType {
    /// Point source
    Point,
    /// Directional source
    Directional,
    /// Area source
    Area,
    /// Ambient source
    Ambient,
}

/// WebXR spatial audio processor
pub struct WebXRProcessor {
    config: WebXRConfig,
    capabilities: WebXRCapabilities,
    sources: HashMap<String, WebXRAudioSource>,
    listener_pose: WebXRPose,
    audio_context_ready: bool,
    performance_metrics: WebXRMetrics,
    frame_counter: u64,
    last_frame_time: Instant,
}

/// WebXR performance metrics
#[derive(Debug, Clone, Default)]
pub struct WebXRMetrics {
    /// Processing latency (ms)
    pub processing_latency: f32,
    /// Frame rate (FPS)
    pub frame_rate: f32,
    /// JavaScript execution time (ms)
    pub js_execution_time: f32,
    /// Web Audio buffer underruns
    pub buffer_underruns: u32,
    /// Active audio sources
    pub active_sources: u32,
    /// Browser memory usage (MB)
    pub memory_usage: f32,
    /// WebGL/GPU usage (if available)
    pub gpu_usage: Option<f32>,
}

impl WebXRProcessor {
    /// Create a new WebXR spatial audio processor
    pub fn new(config: WebXRConfig) -> Self {
        Self {
            config,
            capabilities: WebXRCapabilities::default(),
            sources: HashMap::new(),
            listener_pose: WebXRPose {
                position: Position3D::new(0.0, 1.7, 0.0),
                orientation: (0.0, 0.0, 0.0, 1.0),
                linear_velocity: None,
                angular_velocity: None,
                confidence: 1.0,
                timestamp: 0.0,
            },
            audio_context_ready: false,
            performance_metrics: WebXRMetrics::default(),
            frame_counter: 0,
            last_frame_time: Instant::now(),
        }
    }

    /// Initialize WebXR session and audio context
    pub async fn initialize(&mut self) -> Result<()> {
        // Detect browser capabilities
        self.capabilities = self.detect_capabilities().await?;

        // Browser-specific initialization
        match self.config.browser {
            BrowserType::Chrome => self.initialize_chrome().await?,
            BrowserType::Firefox => self.initialize_firefox().await?,
            BrowserType::Safari => self.initialize_safari().await?,
            BrowserType::Edge => self.initialize_edge().await?,
            BrowserType::Other => self.initialize_generic().await?,
        }

        self.audio_context_ready = true;
        Ok(())
    }

    /// Detect browser capabilities
    async fn detect_capabilities(&self) -> Result<WebXRCapabilities> {
        // In a real implementation, this would call JavaScript to detect capabilities
        // For now, return default capabilities
        Ok(WebXRCapabilities::default())
    }

    /// Update listener pose from WebXR session
    pub fn update_listener_pose(&mut self, pose: WebXRPose) {
        self.listener_pose = pose;
    }

    /// Add an audio source
    pub fn add_source(&mut self, source: WebXRAudioSource) -> Result<()> {
        if self.sources.len() >= self.config.max_sources {
            return Err(Error::LegacyAudio("Maximum sources exceeded".to_string()));
        }

        self.sources.insert(source.id.clone(), source);
        Ok(())
    }

    /// Remove an audio source
    pub fn remove_source(&mut self, source_id: &str) -> Result<()> {
        self.sources.remove(source_id);
        Ok(())
    }

    /// Update source position
    pub fn update_source_position(&mut self, source_id: &str, position: Position3D) -> Result<()> {
        if let Some(source) = self.sources.get_mut(source_id) {
            source.position = position;
            Ok(())
        } else {
            Err(Error::LegacyAudio(format!("Source {source_id} not found")))
        }
    }

    /// Process frame of spatial audio for WebXR
    pub fn process_frame(&mut self, output_buffer: &mut [f32]) -> Result<()> {
        if !self.audio_context_ready {
            return Err(Error::LegacyAudio("Audio context not ready".to_string()));
        }

        let frame_start = Instant::now();
        self.frame_counter += 1;

        // Clear output buffer
        output_buffer.fill(0.0);

        // Process each active source
        let mut active_count = 0;
        for source in self.sources.values() {
            if source.playing && !source.buffer.is_empty() {
                active_count += 1;
                self.process_source(source, output_buffer)?;
            }
        }

        // Update performance metrics
        let frame_time = frame_start.elapsed();
        self.performance_metrics.processing_latency = frame_time.as_secs_f32() * 1000.0;
        self.performance_metrics.active_sources = active_count;

        // Update frame rate
        let time_since_last = frame_start.duration_since(self.last_frame_time);
        if time_since_last.as_millis() > 0 {
            self.performance_metrics.frame_rate = 1000.0 / time_since_last.as_millis() as f32;
        }
        self.last_frame_time = frame_start;

        Ok(())
    }

    /// Process individual audio source
    fn process_source(&self, source: &WebXRAudioSource, output_buffer: &mut [f32]) -> Result<()> {
        // Calculate distance and attenuation
        let distance = self.listener_pose.position.distance_to(&source.position);
        let attenuation = self.calculate_distance_attenuation(distance);

        // Simple spatial processing (placeholder for full implementation)
        let samples_to_process = output_buffer.len().min(source.buffer.len());
        for (i, output_sample) in output_buffer
            .iter_mut()
            .enumerate()
            .take(samples_to_process)
        {
            let sample = source.buffer[i] * source.volume * attenuation;
            *output_sample += sample;
        }

        Ok(())
    }

    /// Calculate distance-based attenuation
    fn calculate_distance_attenuation(&self, distance: f32) -> f32 {
        // Simple inverse square law with minimum distance
        let min_distance = 1.0;
        let max_distance = 100.0;

        if distance <= min_distance {
            1.0
        } else if distance >= max_distance {
            0.01
        } else {
            min_distance / distance
        }
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> WebXRMetrics {
        self.performance_metrics.clone()
    }

    /// Set source playback state
    pub fn set_source_playing(&mut self, source_id: &str, playing: bool) -> Result<()> {
        if let Some(source) = self.sources.get_mut(source_id) {
            source.playing = playing;
            Ok(())
        } else {
            Err(Error::LegacyAudio(format!("Source {source_id} not found")))
        }
    }

    /// Get optimized spatial config for WebXR
    pub fn get_spatial_config(&self) -> SpatialConfig {
        let mut config = SpatialConfig::default();

        // Web-specific optimizations
        config.sample_rate = self.capabilities.max_sample_rate.min(48000.0) as u32;
        config.buffer_size = self.config.buffer_size;
        config.quality_level = self.config.quality_level;
        config.max_sources = self.config.max_sources;

        // Disable GPU acceleration for web (use WebGL/WebGPU separately)
        config.use_gpu = false;

        // Browser-specific optimizations
        match self.config.browser {
            BrowserType::Chrome => {
                // Chrome has good Web Audio performance
                config.buffer_size = 2048;
            }
            BrowserType::Firefox => {
                // Firefox might need larger buffers
                config.buffer_size = 4096;
            }
            BrowserType::Safari => {
                // Safari has different performance characteristics
                config.buffer_size = 4096;
                config.quality_level *= 0.8; // Reduce quality for Safari
            }
            _ => {
                // Conservative settings for other browsers
                config.buffer_size = 4096;
                config.quality_level *= 0.7;
            }
        }

        config
    }

    // Browser-specific initialization methods
    async fn initialize_chrome(&self) -> Result<()> {
        // Chrome-specific optimizations
        Ok(())
    }

    async fn initialize_firefox(&self) -> Result<()> {
        // Firefox-specific optimizations
        Ok(())
    }

    async fn initialize_safari(&self) -> Result<()> {
        // Safari-specific optimizations
        Ok(())
    }

    async fn initialize_edge(&self) -> Result<()> {
        // Edge-specific optimizations
        Ok(())
    }

    async fn initialize_generic(&self) -> Result<()> {
        // Generic browser initialization
        Ok(())
    }
}

/// WebXR utility functions
pub mod utils {
    use super::*;

    /// Convert WebXR pose to VoiRS position
    pub fn webxr_pose_to_position(pose: &WebXRPose) -> Position3D {
        pose.position
    }

    /// Create WebXR audio source from position and buffer
    pub fn create_point_source(
        id: String,
        position: Position3D,
        buffer: Vec<f32>,
    ) -> WebXRAudioSource {
        WebXRAudioSource {
            id,
            position,
            buffer,
            source_type: WebXRSourceType::Point,
            volume: 1.0,
            playing: false,
            looping: false,
        }
    }

    /// Check if WebXR is supported in current browser
    pub async fn is_webxr_supported() -> bool {
        // In real implementation, this would check navigator.xr
        false // Conservative default
    }

    /// Get supported WebXR session types
    pub async fn get_supported_session_types() -> Vec<WebXRSessionType> {
        // In real implementation, this would query browser capabilities
        vec![WebXRSessionType::Inline] // Conservative default
    }
}

/// JavaScript interop functions (for WASM builds)
#[cfg(target_arch = "wasm32")]
pub mod js_interop {
    use super::*;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = console)]
        fn log(s: &str);

        #[wasm_bindgen(js_namespace = navigator, js_name = xr)]
        static XR: Option<XRSystem>;

        type XRSystem;

        #[wasm_bindgen(method, js_name = isSessionSupported)]
        fn is_session_supported(this: &XRSystem, session_type: &str) -> js_sys::Promise;
    }

    /// Initialize WebXR from JavaScript
    #[wasm_bindgen]
    pub async fn init_webxr_session(session_type: &str) -> Result<(), JsValue> {
        // Implementation would create WebXR session
        Ok(())
    }

    /// Process audio frame in JavaScript/WASM
    #[wasm_bindgen]
    pub fn process_webxr_frame(input: &[f32], output: &mut [f32]) {
        // Simple passthrough for now
        let len = input.len().min(output.len());
        output[..len].copy_from_slice(&input[..len]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webxr_config_creation() {
        let config = WebXRConfig::default();
        assert_eq!(config.session_type, WebXRSessionType::ImmersiveVr);
        assert!(config.enable_spatial_audio);
    }

    #[test]
    fn test_webxr_processor_creation() {
        let config = WebXRConfig::default();
        let processor = WebXRProcessor::new(config);

        assert!(!processor.audio_context_ready);
        assert_eq!(processor.sources.len(), 0);
    }

    #[test]
    fn test_source_management() {
        let config = WebXRConfig::default();
        let mut processor = WebXRProcessor::new(config);

        let source = WebXRAudioSource {
            id: "test_source".to_string(),
            position: Position3D::new(1.0, 0.0, 0.0),
            buffer: vec![0.5; 1024],
            source_type: WebXRSourceType::Point,
            volume: 1.0,
            playing: false,
            looping: false,
        };

        assert!(processor.add_source(source).is_ok());
        assert_eq!(processor.sources.len(), 1);

        assert!(processor.remove_source("test_source").is_ok());
        assert_eq!(processor.sources.len(), 0);
    }

    #[test]
    fn test_pose_updates() {
        let config = WebXRConfig::default();
        let mut processor = WebXRProcessor::new(config);

        let new_pose = WebXRPose {
            position: Position3D::new(1.0, 2.0, 3.0),
            orientation: (0.0, 0.0, 0.0, 1.0),
            linear_velocity: None,
            angular_velocity: None,
            confidence: 0.95,
            timestamp: 1234.5,
        };

        processor.update_listener_pose(new_pose.clone());
        assert_eq!(processor.listener_pose.position.x, 1.0);
        assert_eq!(processor.listener_pose.position.y, 2.0);
        assert_eq!(processor.listener_pose.position.z, 3.0);
    }

    #[test]
    fn test_distance_attenuation() {
        let config = WebXRConfig::default();
        let processor = WebXRProcessor::new(config);

        // Test minimum distance
        let attenuation_close = processor.calculate_distance_attenuation(0.5);
        assert_eq!(attenuation_close, 1.0);

        // Test normal distance
        let attenuation_normal = processor.calculate_distance_attenuation(5.0);
        assert!(attenuation_normal > 0.0 && attenuation_normal < 1.0);

        // Test maximum distance
        let attenuation_far = processor.calculate_distance_attenuation(150.0);
        assert_eq!(attenuation_far, 0.01);
    }

    #[test]
    fn test_spatial_config_generation() {
        let config = WebXRConfig {
            browser: BrowserType::Chrome,
            quality_level: 0.8,
            max_sources: 12,
            buffer_size: 2048,
            ..Default::default()
        };
        let processor = WebXRProcessor::new(config);

        let spatial_config = processor.get_spatial_config();
        assert_eq!(spatial_config.buffer_size, 2048);
        assert_eq!(spatial_config.max_sources, 12);
        assert_eq!(spatial_config.quality_level, 0.8);
    }

    #[test]
    fn test_browser_specific_optimization() {
        let safari_config = WebXRConfig {
            browser: BrowserType::Safari,
            quality_level: 1.0,
            ..Default::default()
        };
        let safari_processor = WebXRProcessor::new(safari_config);
        let safari_spatial_config = safari_processor.get_spatial_config();

        // Safari should have reduced quality
        assert!(safari_spatial_config.quality_level < 1.0);
    }

    #[test]
    fn test_webxr_utils() {
        let pose = WebXRPose {
            position: Position3D::new(1.0, 2.0, 3.0),
            orientation: (0.0, 0.0, 0.0, 1.0),
            linear_velocity: None,
            angular_velocity: None,
            confidence: 1.0,
            timestamp: 0.0,
        };

        let position = utils::webxr_pose_to_position(&pose);
        assert_eq!(position.x, 1.0);
        assert_eq!(position.y, 2.0);
        assert_eq!(position.z, 3.0);

        let source = utils::create_point_source(
            "test".to_string(),
            Position3D::new(0.0, 0.0, 0.0),
            vec![0.5; 100],
        );
        assert_eq!(source.source_type, WebXRSourceType::Point);
        assert_eq!(source.buffer.len(), 100);
    }

    #[tokio::test]
    async fn test_webxr_initialization() {
        let config = WebXRConfig::default();
        let mut processor = WebXRProcessor::new(config);

        // Test initialization
        let result = processor.initialize().await;
        assert!(result.is_ok());
        assert!(processor.audio_context_ready);
    }
}
