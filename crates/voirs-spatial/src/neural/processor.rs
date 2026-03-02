//! Neural spatial audio processor implementation

use super::models::*;
use super::quality::AdaptiveQualityController;
use super::training::{NeuralTrainer, NeuralTrainingResults};
use super::types::*;
use crate::{Error, Result};
use candle_core::Device;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Neural spatial audio processor
pub struct NeuralSpatialProcessor {
    /// Configuration
    config: NeuralSpatialConfig,
    /// Neural network model
    model: Box<dyn NeuralModel + Send + Sync>,
    /// Computing device (CPU/GPU)
    device: Device,
    /// Performance metrics
    metrics: Arc<RwLock<NeuralPerformanceMetrics>>,
    /// Input buffer for temporal context
    input_buffer: Arc<RwLock<Vec<NeuralInputFeatures>>>,
    /// Model cache for different configurations
    model_cache: Arc<RwLock<HashMap<String, Box<dyn NeuralModel + Send + Sync>>>>,
    /// Quality adaptation controller
    quality_controller: AdaptiveQualityController,
}

impl NeuralSpatialProcessor {
    /// Create a new neural spatial processor
    pub fn new(config: NeuralSpatialConfig) -> Result<Self> {
        let device = if config.use_gpu {
            // Use catch_unwind because Device::new_cuda can panic on systems without CUDA
            std::panic::catch_unwind(|| Device::new_cuda(0))
                .unwrap_or(Ok(Device::Cpu))
                .unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        let model = Self::create_model(&config, &device)?;
        let quality_controller =
            AdaptiveQualityController::new(config.realtime_constraints.max_latency_ms);

        Ok(Self {
            config,
            model,
            device,
            metrics: Arc::new(RwLock::new(NeuralPerformanceMetrics::default())),
            input_buffer: Arc::new(RwLock::new(Vec::new())),
            model_cache: Arc::new(RwLock::new(HashMap::new())),
            quality_controller,
        })
    }

    /// Create a model based on configuration
    fn create_model(
        config: &NeuralSpatialConfig,
        device: &Device,
    ) -> Result<Box<dyn NeuralModel + Send + Sync>> {
        match config.model_type {
            NeuralModelType::Feedforward => Ok(Box::new(FeedforwardModel::new(
                config.clone(),
                device.clone(),
            )?)),
            NeuralModelType::Convolutional => Ok(Box::new(ConvolutionalModel::new(
                config.clone(),
                device.clone(),
            )?)),
            NeuralModelType::Transformer => Ok(Box::new(TransformerModel::new(
                config.clone(),
                device.clone(),
            )?)),
            _ => Err(Error::LegacyProcessing(format!(
                "Neural model type {:?} not yet implemented",
                config.model_type
            ))),
        }
    }

    /// Process audio with neural spatial synthesis
    pub fn process(&mut self, input: &NeuralInputFeatures) -> Result<NeuralSpatialOutput> {
        let start_time = std::time::Instant::now();

        // Add to input buffer for temporal context
        {
            let mut buffer = self.input_buffer.write().map_err(|e| {
                Error::LegacyProcessing(format!(
                    "Failed to acquire write lock on input buffer: {}",
                    e
                ))
            })?;
            buffer.push(input.clone());

            // Keep only recent frames for temporal context
            if buffer.len() > 10 {
                buffer.remove(0);
            }
        }

        // Forward pass through the model
        let mut output = self.model.forward(input)?;

        // Calculate processing time
        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;
        output.latency_ms = processing_time;

        // Update metrics
        {
            let mut metrics = self.metrics.write().map_err(|e| {
                Error::LegacyProcessing(format!("Failed to acquire write lock on metrics: {}", e))
            })?;
            metrics.frames_processed += 1;
            metrics.avg_processing_time_ms = (metrics.avg_processing_time_ms
                * (metrics.frames_processed - 1) as f32
                + processing_time)
                / metrics.frames_processed as f32;
            metrics.peak_processing_time_ms = metrics.peak_processing_time_ms.max(processing_time);

            if processing_time > self.config.realtime_constraints.max_latency_ms {
                metrics.realtime_violations += 1;
            }
        }

        // Adaptive quality control
        if self.config.realtime_constraints.adaptive_quality {
            self.quality_controller.update(processing_time);
            let new_quality = self.quality_controller.get_quality();
            if (new_quality - self.quality_controller.current_quality).abs() > 0.05 {
                self.model.set_quality(new_quality)?;
                self.quality_controller.current_quality = new_quality;
            }
        }

        output.quality_score = self.quality_controller.current_quality;
        Ok(output)
    }

    /// Process batch of inputs for better efficiency
    pub fn process_batch(
        &mut self,
        inputs: &[NeuralInputFeatures],
    ) -> Result<Vec<NeuralSpatialOutput>> {
        let mut outputs = Vec::with_capacity(inputs.len());

        for input in inputs {
            outputs.push(self.process(input)?);
        }

        Ok(outputs)
    }

    /// Get current performance metrics
    pub fn metrics(&self) -> Result<NeuralPerformanceMetrics> {
        Ok(self
            .metrics
            .read()
            .map_err(|e| {
                Error::LegacyProcessing(format!("Failed to acquire read lock on metrics: {}", e))
            })?
            .clone())
    }

    /// Reset performance metrics
    pub fn reset_metrics(&self) -> Result<()> {
        let mut metrics = self.metrics.write().map_err(|e| {
            Error::LegacyProcessing(format!("Failed to acquire write lock on metrics: {}", e))
        })?;
        *metrics = NeuralPerformanceMetrics::default();
        Ok(())
    }

    /// Update configuration
    pub fn update_config(&mut self, new_config: NeuralSpatialConfig) -> Result<()> {
        // Check if model needs to be recreated
        if new_config.model_type != self.config.model_type
            || new_config.hidden_dims != self.config.hidden_dims
        {
            self.model = Self::create_model(&new_config, &self.device)?;
        }

        self.config = new_config;
        self.quality_controller.target_latency_ms = self.config.realtime_constraints.max_latency_ms;

        Ok(())
    }

    /// Train the neural model with provided data
    pub fn train(
        &mut self,
        training_data: &[(NeuralInputFeatures, Vec<Vec<f32>>)],
    ) -> Result<NeuralTrainingResults> {
        let config = self.config.training_config.as_ref().ok_or_else(|| {
            Error::LegacyConfig("Training configuration not provided".to_string())
        })?;

        let mut trainer = NeuralTrainer::new(config.clone());
        trainer.train(&mut *self.model, training_data)
    }

    /// Save the current model
    pub fn save_model(&self, path: &str) -> Result<()> {
        self.model.save(path)
    }

    /// Load a trained model
    pub fn load_model(&mut self, path: &str) -> Result<()> {
        self.model.load(path)
    }
}
