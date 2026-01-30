//! Core VitsModel implementation - constructors and accessors
//!
//! This module contains the core VitsModel implementation including:
//! - Model construction and initialization
//! - Configuration and component accessors
//! - Device management
//! - Memory and performance monitoring

use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    AcousticError, MemoryOptimizer, PerformanceMonitor, ProsodyConfig, ProsodyController, Result,
    TensorMemoryPool,
};

use super::{
    Decoder, DurationPredictor, NormalizingFlows, PosteriorEncoder, StyleTransfer,
    StyleTransferConfig, TextEncoder, VitsConfig, VitsModel, VoiceCloning, VoiceCloningConfig,
};

impl VitsModel {
    /// Create new VITS model with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(VitsConfig::default())
    }

    /// Create VITS model with custom configuration
    pub fn with_config(config: VitsConfig) -> Result<Self> {
        let device = Self::select_optimal_device()?;

        let text_encoder = Arc::new(TextEncoder::new(
            config.text_encoder.clone(),
            device.clone(),
        )?);

        let posterior_encoder = Arc::new(PosteriorEncoder::new(
            config.posterior_encoder.clone(),
            device.clone(),
        )?);

        let duration_predictor = Arc::new(DurationPredictor::new(
            config.duration_predictor.clone(),
            device.clone(),
        )?);

        let flows = Arc::new(std::sync::Mutex::new(NormalizingFlows::new(
            config.flows.clone(),
            device.clone(),
        )?));

        let decoder = Arc::new(Decoder::new(config.decoder.clone(), device.clone())?);

        // Initialize default prosody controller
        let prosody_controller = Some(Arc::new(ProsodyController::new(ProsodyConfig::default())));

        // Initialize memory management
        let memory_pool = Arc::new(TensorMemoryPool::new());
        let performance_monitor = Arc::new(PerformanceMonitor::new());

        // Initialize style transfer module
        let style_transfer = Some(Arc::new(StyleTransfer::new(
            StyleTransferConfig::default(),
            device.clone(),
        )?));

        // Initialize voice cloning module
        let voice_cloning = Some(Arc::new(VoiceCloning::new(
            VoiceCloningConfig::default(),
            device.clone(),
        )?));

        // Initialize emotion embeddings if emotion is enabled
        let emotion_embeddings = if config.emotion_enabled {
            Some(Self::create_default_emotion_embeddings(
                &device,
                config.emotion_embedding_dim.unwrap_or(256),
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            text_encoder,
            posterior_encoder,
            duration_predictor,
            flows,
            decoder,
            device,
            prosody_controller,
            memory_pool,
            performance_monitor,
            optimization_enabled: true,
            style_transfer,
            voice_cloning,
            current_emotion: None,
            emotion_embeddings,
        })
    }

    /// Create default emotion embeddings for all basic emotion types
    pub(crate) fn create_default_emotion_embeddings(
        device: &Device,
        embedding_dim: usize,
    ) -> Result<HashMap<String, Tensor>> {
        use crate::speaker::EmotionType;

        let mut embeddings = HashMap::new();
        let emotion_types = EmotionType::all_basic();

        // Create unique embeddings for each emotion type
        for (i, emotion_type) in emotion_types.iter().enumerate() {
            let mut embedding_vec = vec![0.0f32; embedding_dim];

            // Create a unique pattern for each emotion
            match emotion_type {
                EmotionType::Neutral => {
                    // Neutral: low values across all dimensions
                    embedding_vec.fill(0.1);
                }
                EmotionType::Happy => {
                    // Happy: higher energy in first quarter of dimensions
                    for (j, val) in embedding_vec.iter_mut().enumerate() {
                        *val = if j < embedding_dim / 4 { 0.8 } else { 0.2 };
                    }
                }
                EmotionType::Sad => {
                    // Sad: lower energy, concentrated in middle dimensions
                    for (j, val) in embedding_vec.iter_mut().enumerate() {
                        *val = if j >= embedding_dim / 4 && j < 3 * embedding_dim / 4 {
                            0.6
                        } else {
                            0.1
                        };
                    }
                }
                EmotionType::Angry => {
                    // Angry: high energy in latter dimensions
                    for (j, val) in embedding_vec.iter_mut().enumerate() {
                        *val = if j >= 3 * embedding_dim / 4 { 0.9 } else { 0.3 };
                    }
                }
                _ => {
                    // For other emotions, create a unique pattern based on index
                    let phase =
                        (i as f32 * 2.0 * std::f32::consts::PI) / emotion_types.len() as f32;
                    for (j, val) in embedding_vec.iter_mut().enumerate() {
                        let sin_val = (phase + j as f32 * 0.1).sin();
                        *val = (sin_val * 0.5 + 0.5) * 0.7; // Scale to [0, 0.7]
                    }
                }
            }

            // Create tensor from embedding vector
            let tensor = Tensor::new(&embedding_vec[..], device)
                .map_err(|e| AcousticError::ModelError {
                    message: format!("Failed to create emotion embedding tensor: {e}"),
                })?
                .unsqueeze(0) // Add batch dimension [1, embedding_dim]
                .map_err(|e| AcousticError::ModelError {
                    message: format!("Failed to unsqueeze emotion embedding: {e}"),
                })?;

            embeddings.insert(emotion_type.as_str().to_string(), tensor);
        }

        tracing::info!(
            "Initialized {} default emotion embeddings with dimension {}",
            embeddings.len(),
            embedding_dim
        );
        Ok(embeddings)
    }

    /// Get model configuration
    pub fn config(&self) -> &VitsConfig {
        &self.config
    }

    /// Get text encoder
    pub fn text_encoder(&self) -> &TextEncoder {
        &self.text_encoder
    }

    /// Get posterior encoder
    pub fn posterior_encoder(&self) -> &PosteriorEncoder {
        &self.posterior_encoder
    }

    /// Get duration predictor
    pub fn duration_predictor(&self) -> &DurationPredictor {
        &self.duration_predictor
    }

    /// Get normalizing flows
    pub fn flows(&self) -> &Arc<std::sync::Mutex<NormalizingFlows>> {
        &self.flows
    }

    /// Get decoder
    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }

    /// Select optimal device for inference
    pub fn select_optimal_device() -> Result<Device> {
        // Try to use the best available device in order of preference

        // Fallback to CPU
        tracing::info!("Selected CPU device for VITS inference");
        Ok(Device::Cpu)
    }

    /// Get current device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Set prosody configuration
    pub fn with_prosody_config(mut self, prosody_config: ProsodyConfig) -> Self {
        self.prosody_controller = Some(Arc::new(ProsodyController::new(prosody_config)));
        self
    }

    /// Get prosody controller
    pub fn prosody_controller(&self) -> Option<&ProsodyController> {
        self.prosody_controller.as_ref().map(|c| c.as_ref())
    }

    /// Get memory pool for external use
    pub fn memory_pool(&self) -> &TensorMemoryPool {
        &self.memory_pool
    }

    /// Get performance monitor
    pub fn performance_monitor(&self) -> &PerformanceMonitor {
        &self.performance_monitor
    }

    /// Enable or disable performance optimizations
    pub fn set_optimization_enabled(&mut self, enabled: bool) {
        self.optimization_enabled = enabled;
    }

    /// Check if optimizations are enabled
    pub fn is_optimization_enabled(&self) -> bool {
        self.optimization_enabled
    }

    /// Get memory pool statistics
    pub fn memory_stats(&self) -> crate::memory::PoolStats {
        self.memory_pool.stats()
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> HashMap<String, std::time::Duration> {
        let mut stats = HashMap::new();

        // Common operations to report
        let operations = [
            "synthesize",
            "text_encoding",
            "duration_prediction",
            "flows",
            "decoding",
        ];

        for op in &operations {
            if let Some(avg_time) = self.performance_monitor.average_timing(op) {
                stats.insert(op.to_string(), avg_time);
            }
        }

        stats
    }

    /// Set device for inference
    pub fn with_device(mut self, device: Device) -> Result<Self> {
        self.device = device.clone();

        // Move all components to new device
        self.text_encoder = Arc::new(TextEncoder::new(
            self.config.text_encoder.clone(),
            device.clone(),
        )?);

        self.posterior_encoder = Arc::new(PosteriorEncoder::new(
            self.config.posterior_encoder.clone(),
            device.clone(),
        )?);

        self.duration_predictor = Arc::new(DurationPredictor::new(
            self.config.duration_predictor.clone(),
            device.clone(),
        )?);

        self.flows = Arc::new(std::sync::Mutex::new(NormalizingFlows::new(
            self.config.flows.clone(),
            device.clone(),
        )?));

        self.decoder = Arc::new(Decoder::new(self.config.decoder.clone(), device)?);

        // Keep the existing prosody controller and memory management (no device dependency)

        Ok(self)
    }
}

impl Default for VitsModel {
    fn default() -> Self {
        Self::new().expect("Failed to create default VITS model")
    }
}
