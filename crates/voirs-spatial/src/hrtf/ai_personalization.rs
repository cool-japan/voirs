//! AI-driven HRTF personalization using machine learning
//!
//! This module provides machine learning-based personalization of Head-Related Transfer Functions (HRTFs)
//! based on individual anthropometric measurements, listening preferences, and perceptual feedback.

use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::hrtf::{HrtfDatabase, HrtfMetadata};
use crate::{Error, Result};

/// Type alias for personalized HRTF responses
type PersonalizedHrtfResponses = HashMap<(i32, i32), (Array1<f32>, Array1<f32>)>;

/// AI-based HRTF personalization system
pub struct AiHrtfPersonalizer {
    /// Neural network model for HRTF adaptation
    model: PersonalizationModel,
    /// Training data cache
    training_data: Vec<PersonalizationSample>,
    /// Model configuration
    config: PersonalizationConfig,
    /// Compute device (CPU/GPU)
    device: Device,
    /// Variable map for model parameters
    var_map: VarMap,
}

/// Neural network model for HRTF personalization
pub struct PersonalizationModel {
    /// Input layer (anthropometric features → hidden)
    input_layer: Linear,
    /// Hidden layers
    hidden_layers: Vec<Linear>,
    /// Output layer (hidden → HRTF modifications)
    output_layer: Linear,
    /// Model dimensions
    config: ModelConfig,
}

/// Configuration for the personalization model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizationConfig {
    /// Model architecture configuration
    pub model_config: ModelConfig,
    /// Training configuration
    pub training_config: TrainingConfig,
    /// Adaptation strategy
    pub adaptation_strategy: AdaptationStrategy,
    /// Enable real-time adaptation
    pub enable_realtime_adaptation: bool,
    /// Minimum samples required for personalization
    pub min_samples_for_personalization: usize,
    /// Confidence threshold for applying adaptations
    pub confidence_threshold: f32,
}

/// Model architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Input feature dimension (anthropometric measurements)
    pub input_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Output dimension (HRTF modification parameters)
    pub output_dim: usize,
    /// Activation function
    pub activation: ActivationFunction,
    /// Dropout rate for regularization
    pub dropout_rate: f32,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Validation split ratio
    pub validation_split: f32,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// L2 regularization weight
    pub l2_regularization: f64,
}

/// HRTF adaptation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    /// Adapt existing HRTF based on measurements
    Measurement,
    /// Adapt based on perceptual feedback
    Perceptual,
    /// Hybrid approach combining measurements and feedback
    Hybrid,
    /// Transfer learning from similar subjects
    TransferLearning,
}

/// Activation functions for neural network
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationFunction {
    /// Rectified Linear Unit
    ReLU,
    /// Leaky ReLU
    LeakyReLU,
    /// Exponential Linear Unit
    ELU,
    /// Gaussian Error Linear Unit
    GELU,
    /// Hyperbolic tangent
    Tanh,
}

/// Anthropometric measurements for HRTF personalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropometricMeasurements {
    /// Head circumference (cm)
    pub head_circumference: f32,
    /// Head width (cm)
    pub head_width: f32,
    /// Head depth (cm)
    pub head_depth: f32,
    /// Pinna height (cm)
    pub pinna_height: f32,
    /// Pinna width (cm)
    pub pinna_width: f32,
    /// Concha depth (cm)
    pub concha_depth: f32,
    /// Interaural distance (cm)
    pub interaural_distance: f32,
    /// Shoulder width (cm)
    pub shoulder_width: f32,
    /// Torso depth (cm)
    pub torso_depth: f32,
    /// Height (cm)
    pub height: f32,
    /// Age (years)
    pub age: f32,
    /// Gender (0.0 = female, 1.0 = male, 0.5 = non-binary)
    pub gender: f32,
}

/// Perceptual feedback for HRTF personalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualFeedback {
    /// Sound source positions that were tested
    pub test_positions: Vec<(f32, f32, f32)>,
    /// Perceived positions (user feedback)
    pub perceived_positions: Vec<(f32, f32, f32)>,
    /// Confidence ratings (0.0-1.0)
    pub confidence_ratings: Vec<f32>,
    /// Localization accuracy scores
    pub accuracy_scores: Vec<f32>,
    /// Preference ratings for different HRTF variants
    pub preference_ratings: Vec<f32>,
    /// Comments or qualitative feedback
    pub comments: Vec<String>,
}

/// Training sample for personalization model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizationSample {
    /// Input features (anthropometric measurements)
    pub features: AnthropometricMeasurements,
    /// Target HRTF modifications
    pub target_modifications: HrtfModifications,
    /// Optional perceptual feedback
    pub perceptual_feedback: Option<PerceptualFeedback>,
    /// Sample weight for training
    pub weight: f32,
}

/// HRTF modifications produced by the AI model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrtfModifications {
    /// Frequency response adjustments (per frequency bin)
    pub frequency_adjustments: Vec<f32>,
    /// Time delay adjustments (ITD modifications)
    pub time_delay_adjustments: f32,
    /// Level adjustments (ILD modifications)
    pub level_adjustments: f32,
    /// Spectral shape modifications
    pub spectral_shape_mods: Vec<f32>,
    /// Directional pattern adjustments
    pub directional_adjustments: HashMap<(i32, i32), f32>,
    /// Confidence score for these modifications
    pub confidence: f32,
}

/// Personalized HRTF generated by AI
pub struct PersonalizedHrtf {
    /// Base HRTF database
    pub base_hrtf: HrtfDatabase,
    /// AI-generated modifications
    pub modifications: HrtfModifications,
    /// Personalization metadata
    pub metadata: PersonalizationMetadata,
    /// Modified impulse responses
    pub personalized_responses: HashMap<(i32, i32), (Array1<f32>, Array1<f32>)>,
}

/// Metadata for personalized HRTF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizationMetadata {
    /// User ID or identifier
    pub user_id: String,
    /// Anthropometric measurements used
    pub measurements: AnthropometricMeasurements,
    /// Model version used for personalization
    pub model_version: String,
    /// Personalization timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Validation accuracy
    pub validation_accuracy: f32,
    /// Number of training samples used
    pub training_samples: usize,
}

impl Default for PersonalizationConfig {
    fn default() -> Self {
        Self {
            model_config: ModelConfig {
                input_dim: 12, // Number of anthropometric features
                hidden_dims: vec![64, 32, 16],
                output_dim: 128, // HRTF modification parameters
                activation: ActivationFunction::ReLU,
                dropout_rate: 0.2,
            },
            training_config: TrainingConfig {
                learning_rate: 0.001,
                batch_size: 32,
                epochs: 100,
                validation_split: 0.2,
                early_stopping_patience: 10,
                l2_regularization: 0.001,
            },
            adaptation_strategy: AdaptationStrategy::Hybrid,
            enable_realtime_adaptation: true,
            min_samples_for_personalization: 10,
            confidence_threshold: 0.7,
        }
    }
}

impl Default for AnthropometricMeasurements {
    fn default() -> Self {
        Self {
            head_circumference: 56.0,
            head_width: 15.5,
            head_depth: 19.0,
            pinna_height: 6.5,
            pinna_width: 3.5,
            concha_depth: 1.5,
            interaural_distance: 14.5,
            shoulder_width: 40.0,
            torso_depth: 25.0,
            height: 170.0,
            age: 30.0,
            gender: 0.5,
        }
    }
}

impl AiHrtfPersonalizer {
    /// Create a new AI HRTF personalizer
    pub fn new(config: PersonalizationConfig) -> Result<Self> {
        let device = Device::Cpu; // Default to CPU, can be enhanced for GPU
        let var_map = VarMap::new();

        let model = PersonalizationModel::new(&config.model_config, &device, &var_map)?;

        Ok(Self {
            model,
            training_data: Vec::new(),
            config,
            device,
            var_map,
        })
    }

    /// Add training sample to the dataset
    pub fn add_training_sample(&mut self, sample: PersonalizationSample) {
        self.training_data.push(sample);
    }

    /// Train the personalization model
    pub fn train_model(&mut self) -> Result<TrainingResults> {
        if self.training_data.len() < self.config.min_samples_for_personalization {
            return Err(Error::processing(&format!(
                "Insufficient training data: {} samples, need at least {}",
                self.training_data.len(),
                self.config.min_samples_for_personalization
            )));
        }

        let (train_data, val_data) = self.split_training_data()?;
        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;
        let mut training_losses = Vec::new();
        let mut validation_losses = Vec::new();

        // Create optimizer
        let mut optimizer = candle_nn::AdamW::new(
            self.var_map.all_vars(),
            candle_nn::ParamsAdamW {
                lr: self.config.training_config.learning_rate,
                weight_decay: self.config.training_config.l2_regularization,
                ..Default::default()
            },
        )
        .map_err(|e| Error::processing(&format!("Failed to create optimizer: {e}")))?;

        // Training loop
        for epoch in 0..self.config.training_config.epochs {
            let train_loss = self.train_epoch(&train_data, &mut optimizer)?;
            let val_loss = self.validate_epoch(&val_data)?;

            training_losses.push(train_loss);
            validation_losses.push(val_loss);

            // Early stopping
            if val_loss < best_loss {
                best_loss = val_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.training_config.early_stopping_patience {
                    break;
                }
            }

            if epoch % 10 == 0 {
                println!("Epoch {epoch}: Train Loss = {train_loss:.4}, Val Loss = {val_loss:.4}");
            }
        }

        let epochs_completed = training_losses.len();
        Ok(TrainingResults {
            final_train_loss: *training_losses.last().unwrap_or(&0.0),
            final_validation_loss: *validation_losses.last().unwrap_or(&0.0),
            training_losses,
            validation_losses,
            epochs_completed,
        })
    }

    /// Generate personalized HRTF for given measurements
    pub fn generate_personalized_hrtf(
        &self,
        measurements: &AnthropometricMeasurements,
        base_hrtf: &HrtfDatabase,
    ) -> Result<PersonalizedHrtf> {
        let features = self.measurements_to_tensor(measurements)?;
        let modifications = self.model.forward(&features)?;
        let hrtf_mods = self.tensor_to_modifications(&modifications)?;

        if hrtf_mods.confidence < self.config.confidence_threshold {
            return Err(Error::processing(&format!(
                "Personalization confidence too low: {:.3} < {:.3}",
                hrtf_mods.confidence, self.config.confidence_threshold
            )));
        }

        let personalized_responses = self.apply_modifications(base_hrtf, &hrtf_mods)?;

        Ok(PersonalizedHrtf {
            base_hrtf: base_hrtf.clone(),
            modifications: hrtf_mods,
            metadata: PersonalizationMetadata {
                user_id: "anonymous".to_string(),
                measurements: measurements.clone(),
                model_version: "0.1.0".to_string(),
                created_at: chrono::Utc::now(),
                validation_accuracy: 0.85, // Placeholder
                training_samples: self.training_data.len(),
            },
            personalized_responses,
        })
    }

    /// Update model with new perceptual feedback
    pub fn update_with_feedback(
        &mut self,
        measurements: &AnthropometricMeasurements,
        feedback: &PerceptualFeedback,
    ) -> Result<()> {
        if !self.config.enable_realtime_adaptation {
            return Ok(());
        }

        // Convert feedback to training sample
        let modifications = self.feedback_to_modifications(feedback)?;
        let sample = PersonalizationSample {
            features: measurements.clone(),
            target_modifications: modifications,
            perceptual_feedback: Some(feedback.clone()),
            weight: 1.0,
        };

        self.add_training_sample(sample);

        // Retrain if we have enough new samples
        if self.training_data.len() % 50 == 0 {
            let _ = self.train_model()?;
        }

        Ok(())
    }

    // Private helper methods

    fn split_training_data(
        &self,
    ) -> Result<(Vec<PersonalizationSample>, Vec<PersonalizationSample>)> {
        use scirs2_core::random::seq::SliceRandom;
        let mut rng = scirs2_core::random::thread_rng();
        let mut data = self.training_data.clone();
        data.shuffle(&mut rng);

        let split_point =
            (data.len() as f32 * (1.0 - self.config.training_config.validation_split)) as usize;
        let (train_data, val_data) = data.split_at(split_point);

        Ok((train_data.to_vec(), val_data.to_vec()))
    }

    fn train_epoch(
        &mut self,
        train_data: &[PersonalizationSample],
        optimizer: &mut candle_nn::AdamW,
    ) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        // Process data in batches
        for batch in train_data.chunks(self.config.training_config.batch_size) {
            let (inputs, targets) = self.prepare_batch(batch)?;
            let predictions = self.model.forward(&inputs)?;
            let loss = self.compute_loss(&predictions, &targets)?;

            optimizer
                .backward_step(&loss)
                .map_err(|e| Error::processing(&format!("Backward step failed: {e}")))?;

            total_loss += loss
                .to_scalar::<f64>()
                .map_err(|e| Error::processing(&format!("Failed to extract loss: {e}")))?;
            batch_count += 1;
        }

        Ok(total_loss / batch_count as f64)
    }

    fn validate_epoch(&self, val_data: &[PersonalizationSample]) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for batch in val_data.chunks(self.config.training_config.batch_size) {
            let (inputs, targets) = self.prepare_batch(batch)?;
            let predictions = self.model.forward(&inputs)?;
            let loss = self.compute_loss(&predictions, &targets)?;

            total_loss += loss
                .to_scalar::<f64>()
                .map_err(|e| Error::processing(&format!("Failed to extract loss: {e}")))?;
            batch_count += 1;
        }

        Ok(total_loss / batch_count as f64)
    }

    fn prepare_batch(&self, batch: &[PersonalizationSample]) -> Result<(Tensor, Tensor)> {
        let batch_size = batch.len();
        let input_dim = self.config.model_config.input_dim;
        let output_dim = self.config.model_config.output_dim;

        let mut input_data = vec![0.0f32; batch_size * input_dim];
        let mut target_data = vec![0.0f32; batch_size * output_dim];

        for (i, sample) in batch.iter().enumerate() {
            let features = self.measurements_to_vec(&sample.features);
            let targets = self.modifications_to_vec(&sample.target_modifications);

            let start_idx = i * input_dim;
            input_data[start_idx..start_idx + input_dim].copy_from_slice(&features);

            let start_idx = i * output_dim;
            target_data[start_idx..start_idx + output_dim].copy_from_slice(&targets);
        }

        let inputs = Tensor::from_vec(input_data, (batch_size, input_dim), &self.device)
            .map_err(|e| Error::processing(&format!("Failed to create input tensor: {e}")))?;
        let targets = Tensor::from_vec(target_data, (batch_size, output_dim), &self.device)
            .map_err(|e| Error::processing(&format!("Failed to create target tensor: {e}")))?;

        Ok((inputs, targets))
    }

    fn compute_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let diff = (predictions - targets)
            .map_err(|e| Error::processing(&format!("Failed to compute difference: {e}")))?;
        let squared = diff
            .sqr()
            .map_err(|e| Error::processing(&format!("Failed to square: {e}")))?;
        let loss = squared
            .mean_all()
            .map_err(|e| Error::processing(&format!("Failed to compute mean: {e}")))?;
        Ok(loss)
    }

    fn measurements_to_tensor(&self, measurements: &AnthropometricMeasurements) -> Result<Tensor> {
        let features = self.measurements_to_vec(measurements);
        Tensor::from_vec(
            features,
            (1, self.config.model_config.input_dim),
            &self.device,
        )
        .map_err(|e| Error::processing(&format!("Failed to create tensor: {e}")))
    }

    fn measurements_to_vec(&self, measurements: &AnthropometricMeasurements) -> Vec<f32> {
        vec![
            measurements.head_circumference / 100.0, // Normalize to meters
            measurements.head_width / 100.0,
            measurements.head_depth / 100.0,
            measurements.pinna_height / 100.0,
            measurements.pinna_width / 100.0,
            measurements.concha_depth / 100.0,
            measurements.interaural_distance / 100.0,
            measurements.shoulder_width / 100.0,
            measurements.torso_depth / 100.0,
            measurements.height / 200.0, // Normalize to 0-1 range
            measurements.age / 100.0,    // Normalize to 0-1 range
            measurements.gender,         // Already 0-1
        ]
    }

    fn modifications_to_vec(&self, modifications: &HrtfModifications) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.config.model_config.output_dim);

        // Add frequency adjustments (truncate or pad to fit)
        for i in 0..64 {
            vec.push(
                modifications
                    .frequency_adjustments
                    .get(i)
                    .cloned()
                    .unwrap_or(0.0),
            );
        }

        // Add other modifications
        vec.push(modifications.time_delay_adjustments);
        vec.push(modifications.level_adjustments);

        // Add spectral shape modifications (truncate or pad to fit)
        for i in 0..32 {
            vec.push(
                modifications
                    .spectral_shape_mods
                    .get(i)
                    .cloned()
                    .unwrap_or(0.0),
            );
        }

        // Add directional adjustments (select key directions)
        let key_directions = [(0, 0), (90, 0), (180, 0), (270, 0), (0, 30), (0, -30)];
        for &dir in &key_directions {
            vec.push(
                modifications
                    .directional_adjustments
                    .get(&dir)
                    .cloned()
                    .unwrap_or(0.0),
            );
        }

        // Add confidence
        vec.push(modifications.confidence);

        // Pad to exact output dimension
        while vec.len() < self.config.model_config.output_dim {
            vec.push(0.0);
        }

        // Truncate if too long
        vec.truncate(self.config.model_config.output_dim);

        vec
    }

    fn tensor_to_modifications(&self, tensor: &Tensor) -> Result<HrtfModifications> {
        let data = tensor
            .flatten_all()
            .map_err(|e| Error::processing(&format!("Failed to flatten tensor: {e}")))?
            .to_vec1::<f32>()
            .map_err(|e| Error::processing(&format!("Failed to convert to vec: {e}")))?;

        let mut idx = 0;

        // Extract frequency adjustments
        let frequency_adjustments = data[idx..idx + 64].to_vec();
        idx += 64;

        let time_delay_adjustments = data[idx];
        idx += 1;

        let level_adjustments = data[idx];
        idx += 1;

        // Extract spectral shape modifications
        let spectral_shape_mods = data[idx..idx + 32].to_vec();
        idx += 32;

        // Extract directional adjustments
        let mut directional_adjustments = HashMap::new();
        let key_directions = [(0, 0), (90, 0), (180, 0), (270, 0), (0, 30), (0, -30)];
        for &dir in &key_directions {
            directional_adjustments.insert(dir, data[idx]);
            idx += 1;
        }

        let confidence = data[idx].clamp(0.0, 1.0); // Clamp to [0, 1]

        Ok(HrtfModifications {
            frequency_adjustments,
            time_delay_adjustments,
            level_adjustments,
            spectral_shape_mods,
            directional_adjustments,
            confidence,
        })
    }

    fn feedback_to_modifications(
        &self,
        feedback: &PerceptualFeedback,
    ) -> Result<HrtfModifications> {
        // Convert perceptual feedback to HRTF modifications
        // This is a simplified implementation - in practice, this would be more sophisticated

        let mut frequency_adjustments = vec![0.0; 64];
        let mut time_delay_adjustments = 0.0;
        let mut level_adjustments = 0.0;
        let mut spectral_shape_mods = vec![0.0; 32];
        let mut directional_adjustments = HashMap::new();

        // Analyze localization errors to infer needed modifications
        for (i, (&test_pos, &perceived_pos)) in feedback
            .test_positions
            .iter()
            .zip(feedback.perceived_positions.iter())
            .enumerate()
        {
            let error_x = perceived_pos.0 - test_pos.0;
            let error_y = perceived_pos.1 - test_pos.1;
            let error_z = perceived_pos.2 - test_pos.2;

            // Simple heuristic: front/back confusion suggests spectral cues need adjustment
            if error_y.abs() > 0.5 {
                for item in spectral_shape_mods.iter_mut().take(16).skip(8) {
                    // Mid-high frequencies
                    *item += error_y * 0.1;
                }
            }

            // Left/right errors suggest ITD/ILD adjustments
            if error_x.abs() > 0.3 {
                time_delay_adjustments += error_x * 0.01;
                level_adjustments += error_x * 0.02;
            }

            // Elevation errors suggest pinna cue adjustments
            if error_z.abs() > 0.3 {
                for item in spectral_shape_mods.iter_mut().take(32).skip(16) {
                    // High frequencies
                    *item += error_z * 0.05;
                }
            }
        }

        // Calculate confidence based on accuracy scores
        let confidence = if feedback.accuracy_scores.is_empty() {
            0.5
        } else {
            feedback.accuracy_scores.iter().sum::<f32>() / feedback.accuracy_scores.len() as f32
        };

        Ok(HrtfModifications {
            frequency_adjustments,
            time_delay_adjustments,
            level_adjustments,
            spectral_shape_mods,
            directional_adjustments,
            confidence,
        })
    }

    fn apply_modifications(
        &self,
        base_hrtf: &HrtfDatabase,
        modifications: &HrtfModifications,
    ) -> Result<PersonalizedHrtfResponses> {
        let mut personalized_responses = HashMap::new();

        // Apply modifications to each direction in the base HRTF
        for (&(azimuth, elevation), left_response) in &base_hrtf.left_responses {
            if let Some(right_response) = base_hrtf.right_responses.get(&(azimuth, elevation)) {
                let modified_left =
                    self.apply_modifications_to_response(left_response, modifications);
                let modified_right =
                    self.apply_modifications_to_response(right_response, modifications);

                personalized_responses
                    .insert((azimuth, elevation), (modified_left, modified_right));
            }
        }

        Ok(personalized_responses)
    }

    fn apply_modifications_to_response(
        &self,
        response: &Array1<f32>,
        modifications: &HrtfModifications,
    ) -> Array1<f32> {
        let mut modified = response.clone();

        // Apply simple time-domain modifications
        // In practice, this would be more sophisticated with FFT-based filtering

        // Apply level adjustment
        modified.mapv_inplace(|x| x * (1.0 + modifications.level_adjustments));

        // Apply spectral modifications (simplified)
        // This is a placeholder - real implementation would use proper filtering
        for (i, &adj) in modifications.spectral_shape_mods.iter().enumerate() {
            if i < modified.len() {
                modified[i] *= 1.0 + adj * 0.1;
            }
        }

        modified
    }
}

impl PersonalizationModel {
    /// Create a new personalization model
    pub fn new(config: &ModelConfig, device: &Device, var_map: &VarMap) -> Result<Self> {
        let vb = VarBuilder::from_varmap(var_map, candle_core::DType::F32, device);

        // Create input layer
        let input_layer = linear(config.input_dim, config.hidden_dims[0], vb.pp("input"))
            .map_err(|e| Error::processing(&format!("Failed to create input layer: {e}")))?;

        // Create hidden layers
        let mut hidden_layers = Vec::new();
        for i in 0..config.hidden_dims.len() - 1 {
            let layer = linear(
                config.hidden_dims[i],
                config.hidden_dims[i + 1],
                vb.pp(format!("hidden_{i}")),
            )
            .map_err(|e| Error::processing(&format!("Failed to create hidden layer {i}: {e}")))?;
            hidden_layers.push(layer);
        }

        // Create output layer
        let output_layer = linear(
            *config.hidden_dims.last().unwrap(),
            config.output_dim,
            vb.pp("output"),
        )
        .map_err(|e| Error::processing(&format!("Failed to create output layer: {e}")))?;

        Ok(Self {
            input_layer,
            hidden_layers,
            output_layer,
            config: config.clone(),
        })
    }

    /// Forward pass through the model
    pub fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        let mut x = self.input_layer.forward(input)?;
        x = self.apply_activation(&x)?;

        for layer in &self.hidden_layers {
            x = layer.forward(&x)?;
            x = self.apply_activation(&x)?;
        }

        self.output_layer.forward(&x)
    }

    fn apply_activation(&self, tensor: &Tensor) -> CandleResult<Tensor> {
        match self.config.activation {
            ActivationFunction::ReLU => {
                let zeros = tensor.zeros_like()?;
                tensor.maximum(&zeros)
            }
            ActivationFunction::LeakyReLU => {
                let zeros = tensor.zeros_like()?;
                let negative_part = (tensor * 0.01)?;
                let condition = tensor.gt(&zeros)?;
                condition.where_cond(tensor, &negative_part)
            }
            ActivationFunction::ELU => {
                let zeros = tensor.zeros_like()?;
                let positive_part = tensor.clone();
                let negative_part = (tensor.exp()? - 1.0)?;
                let condition = tensor.gt(&zeros)?;
                condition.where_cond(&positive_part, &negative_part)
            }
            ActivationFunction::GELU => {
                // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                let x_cubed = tensor.powf(3.0)?;
                let inner = (tensor + &(x_cubed * 0.044715)?)?;
                let scaled = (&inner * (2.0 / std::f64::consts::PI).sqrt())?;
                let tanh_part = scaled.tanh()?;
                let one_plus_tanh = (&tanh_part + 1.0)?;
                (tensor * 0.5)? * one_plus_tanh
            }
            ActivationFunction::Tanh => tensor.tanh(),
        }
    }
}

/// Results from model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResults {
    /// Final training loss
    pub final_train_loss: f64,
    /// Final validation loss
    pub final_validation_loss: f64,
    /// Training loss history
    pub training_losses: Vec<f64>,
    /// Validation loss history
    pub validation_losses: Vec<f64>,
    /// Number of epochs completed
    pub epochs_completed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_personalization_config_default() {
        let config = PersonalizationConfig::default();
        assert_eq!(config.model_config.input_dim, 12);
        assert_eq!(config.model_config.output_dim, 128);
        assert_eq!(config.adaptation_strategy, AdaptationStrategy::Hybrid);
    }

    #[test]
    fn test_anthropometric_measurements_default() {
        let measurements = AnthropometricMeasurements::default();
        assert_eq!(measurements.head_circumference, 56.0);
        assert_eq!(measurements.gender, 0.5);
    }

    #[test]
    fn test_personalizer_creation() {
        let config = PersonalizationConfig::default();
        let personalizer = AiHrtfPersonalizer::new(config);
        assert!(personalizer.is_ok());
    }

    #[test]
    fn test_measurements_to_vec() {
        let config = PersonalizationConfig::default();
        let personalizer = AiHrtfPersonalizer::new(config).unwrap();
        let measurements = AnthropometricMeasurements::default();

        let vec = personalizer.measurements_to_vec(&measurements);
        assert_eq!(vec.len(), 12);
        assert_eq!(vec[0], 0.56); // head_circumference / 100
        assert_eq!(vec[11], 0.5); // gender
    }

    #[test]
    fn test_feedback_to_modifications() {
        let config = PersonalizationConfig::default();
        let personalizer = AiHrtfPersonalizer::new(config).unwrap();

        let feedback = PerceptualFeedback {
            test_positions: vec![(1.0, 0.0, 0.0)],
            perceived_positions: vec![(0.5, 0.0, 0.0)],
            confidence_ratings: vec![0.8],
            accuracy_scores: vec![0.7],
            preference_ratings: vec![0.9],
            comments: vec!["Good localization".to_string()],
        };

        let modifications = personalizer.feedback_to_modifications(&feedback).unwrap();
        assert_eq!(modifications.confidence, 0.7);
        assert!(modifications.time_delay_adjustments.abs() > 0.0); // Should have some adjustment
    }

    #[test]
    fn test_training_sample_creation() {
        let measurements = AnthropometricMeasurements::default();
        let modifications = HrtfModifications {
            frequency_adjustments: vec![0.0; 64],
            time_delay_adjustments: 0.01,
            level_adjustments: 0.02,
            spectral_shape_mods: vec![0.0; 32],
            directional_adjustments: HashMap::new(),
            confidence: 0.8,
        };

        let sample = PersonalizationSample {
            features: measurements,
            target_modifications: modifications,
            perceptual_feedback: None,
            weight: 1.0,
        };

        assert_eq!(sample.weight, 1.0);
        assert_eq!(sample.target_modifications.confidence, 0.8);
    }
}
