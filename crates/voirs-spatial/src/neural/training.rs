//! Training pipeline and loss functions for neural models
//!
//! [`NeuralTrainer::train`] performs *real* gradient-based training: each
//! sample's forward pass is computed via [`super::models::NeuralModel::forward_tensor`]
//! (which preserves candle's autograd graph), a differentiable mean-squared-error
//! loss is built against the target audio, and `loss.backward()` followed by a
//! real optimizer step (`candle_nn::SGD`/`AdamW`) actually mutates the model's
//! weights (`candle_core::Var`s, shared storage with the model's layers - see
//! [`super::models::NeuralModel::trainable_vars`]).
//!
//! The *reported* training/validation loss curve uses the user-configured
//! [`LossFunction`] (MSE/MAE/spectral/perceptual/multi-scale/combined),
//! computed non-differentiably from the same forward pass purely for
//! diagnostics - the gradient itself always flows through the MSE surrogate
//! above, since the audio-domain losses are not (yet) implemented as
//! differentiable tensor operations in this crate.

use super::models::NeuralModel;
use super::types::*;
use crate::{Error, Result};
use candle_core::Tensor;
use candle_nn::{AdamW, Optimizer as CandleOptimizer, ParamsAdamW, SGD};

pub use super::types::NeuralTrainingResults;

/// Real gradient-based optimizer used by [`NeuralTrainer`].
///
/// Wraps the two optimizers `candle_nn` 0.11 ships. `OptimizerType::Adam` maps
/// onto `AdamW` with zero weight decay (mathematically equivalent to plain
/// Adam); `OptimizerType::AdamW` uses `candle_nn`'s default weight decay.
/// `OptimizerType::RMSprop` has no `candle_nn` implementation to delegate to
/// and is rejected with a clear error rather than silently substituted with a
/// different optimizer under the requested name.
enum SpatialOptimizer {
    Sgd(SGD),
    AdamW(AdamW),
}

impl SpatialOptimizer {
    fn new(vars: Vec<candle_core::Var>, config: &TrainingConfig) -> Result<Self> {
        match config.optimizer {
            OptimizerType::SGD => {
                let optimizer = SGD::new(vars, config.learning_rate).map_err(|e| {
                    Error::LegacyProcessing(format!("Failed to build SGD optimizer: {e}"))
                })?;
                Ok(Self::Sgd(optimizer))
            }
            OptimizerType::Adam => {
                let params = ParamsAdamW {
                    lr: config.learning_rate,
                    weight_decay: 0.0,
                    ..ParamsAdamW::default()
                };
                let optimizer = AdamW::new(vars, params).map_err(|e| {
                    Error::LegacyProcessing(format!("Failed to build Adam optimizer: {e}"))
                })?;
                Ok(Self::AdamW(optimizer))
            }
            OptimizerType::AdamW => {
                let params = ParamsAdamW {
                    lr: config.learning_rate,
                    ..ParamsAdamW::default()
                };
                let optimizer = AdamW::new(vars, params).map_err(|e| {
                    Error::LegacyProcessing(format!("Failed to build AdamW optimizer: {e}"))
                })?;
                Ok(Self::AdamW(optimizer))
            }
            OptimizerType::RMSprop => Err(Error::LegacyProcessing(
                "RMSprop optimizer is not implemented (candle_nn 0.11 only provides SGD and \
                 AdamW); select OptimizerType::Adam, OptimizerType::AdamW, or OptimizerType::SGD \
                 instead of silently training with a substitute optimizer"
                    .to_string(),
            )),
        }
    }

    /// Real backward pass + parameter update: computes gradients of `loss`
    /// with respect to every tracked `Var` and applies the optimizer's update
    /// rule to them in place. This is what actually changes the model's
    /// weights - the previous implementation never called anything like this.
    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        match self {
            Self::Sgd(optimizer) => optimizer.backward_step(loss),
            Self::AdamW(optimizer) => optimizer.backward_step(loss),
        }
        .map_err(|e| Error::LegacyProcessing(format!("Optimizer step failed: {e}")))
    }
}

/// Neural model trainer
pub struct NeuralTrainer {
    config: TrainingConfig,
}

impl NeuralTrainer {
    /// Create a new neural trainer
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    /// Train a neural model
    pub fn train(
        &mut self,
        model: &mut dyn NeuralModel,
        training_data: &[(NeuralInputFeatures, Vec<Vec<f32>>)],
    ) -> Result<NeuralTrainingResults> {
        let start_time = std::time::Instant::now();
        let mut training_loss_history = Vec::new();
        let mut validation_loss_history = Vec::new();
        let mut best_validation_loss = f32::INFINITY;
        let mut patience_counter = 0;
        let mut early_stopped = false;

        // Build a real optimizer over the model's actual trainable `Var`s.
        let trainable_vars = model.trainable_vars();
        if trainable_vars.is_empty() {
            return Err(Error::LegacyProcessing(
                "Model exposes no trainable parameters (trainable_vars() is empty); there is \
                 nothing for training to update"
                    .to_string(),
            ));
        }
        let mut optimizer = SpatialOptimizer::new(trainable_vars, &self.config)?;

        // Split data into training and validation sets
        let split_index =
            (training_data.len() as f32 * (1.0 - self.config.validation_split)) as usize;
        let (train_data, val_data) = training_data.split_at(split_index);

        println!(
            "Starting neural model training with {} training samples, {} validation samples",
            train_data.len(),
            val_data.len()
        );

        for epoch in 0..self.config.epochs {
            let epoch_start = std::time::Instant::now();

            // Training phase: real forward + backward + optimizer step per sample.
            let train_loss = self.train_epoch(model, train_data, &mut optimizer)?;
            training_loss_history.push(train_loss);

            // Validation phase: forward-only, no gradient computation.
            let val_loss = self.validate_epoch(model, val_data)?;
            validation_loss_history.push(val_loss);

            println!(
                "Epoch {}/{}: train_loss={:.6}, val_loss={:.6}, time={:.2}s",
                epoch + 1,
                self.config.epochs,
                train_loss,
                val_loss,
                epoch_start.elapsed().as_secs_f32()
            );

            // Early stopping check
            if val_loss < best_validation_loss {
                best_validation_loss = val_loss;
                patience_counter = 0;
                println!("New best validation loss: {val_loss:.6}");
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    println!(
                        "Early stopping triggered after {patience_counter} epochs of no improvement"
                    );
                    early_stopped = true;
                    break;
                }
            }
        }

        let training_duration = start_time.elapsed().as_secs_f32();
        let epochs_completed = training_loss_history.len();

        // Calculate final accuracy based on final validation loss
        let final_accuracy = if !validation_loss_history.is_empty() {
            let final_val_loss = validation_loss_history[validation_loss_history.len() - 1];
            // Convert loss to accuracy estimate (simple heuristic)
            (1.0 - final_val_loss.min(1.0)).max(0.0)
        } else {
            0.0
        };

        println!(
            "Training completed: {epochs_completed} epochs, {training_duration:.2}s total, final accuracy: {final_accuracy:.3}"
        );

        Ok(NeuralTrainingResults {
            training_loss: training_loss_history,
            validation_loss: validation_loss_history,
            final_accuracy,
            training_duration_secs: training_duration,
            epochs_completed,
            early_stopped,
        })
    }

    fn train_epoch(
        &self,
        model: &mut dyn NeuralModel,
        train_data: &[(NeuralInputFeatures, Vec<Vec<f32>>)],
        optimizer: &mut SpatialOptimizer,
    ) -> Result<f32> {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        // Process in batches
        for batch_start in (0..train_data.len()).step_by(self.config.batch_size.max(1)) {
            let batch_end = (batch_start + self.config.batch_size.max(1)).min(train_data.len());
            let batch = &train_data[batch_start..batch_end];

            let batch_loss = self.train_batch(model, batch, optimizer)?;
            total_loss += batch_loss;
            batch_count += 1;
        }

        Ok(if batch_count > 0 {
            total_loss / batch_count as f32
        } else {
            0.0
        })
    }

    fn validate_epoch(
        &self,
        model: &mut dyn NeuralModel,
        val_data: &[(NeuralInputFeatures, Vec<Vec<f32>>)],
    ) -> Result<f32> {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        // Process validation data in batches
        for batch_start in (0..val_data.len()).step_by(self.config.batch_size.max(1)) {
            let batch_end = (batch_start + self.config.batch_size.max(1)).min(val_data.len());
            let batch = &val_data[batch_start..batch_end];

            let batch_loss = self.validate_batch(model, batch)?;
            total_loss += batch_loss;
            batch_count += 1;
        }

        Ok(if batch_count > 0 {
            total_loss / batch_count as f32
        } else {
            0.0
        })
    }

    fn train_batch(
        &self,
        model: &mut dyn NeuralModel,
        batch: &[(NeuralInputFeatures, Vec<Vec<f32>>)],
        optimizer: &mut SpatialOptimizer,
    ) -> Result<f32> {
        let mut batch_loss = 0.0;
        let output_channels = model.config().output_channels;
        let buffer_size = model.config().buffer_size;

        for (input, target) in batch {
            // Real, differentiable forward pass: the autograd graph linking
            // `output_tensor` back to every trainable `Var` stays intact
            // (unlike `NeuralModel::forward`, which extracts to `Vec<f32>`).
            let input_tensor = model.input_tensor(input)?;
            let output_tensor = model.forward_tensor(&input_tensor)?;

            // Report the loss using the model's *configured* `LossFunction`,
            // computed from this same (pre-update) forward pass, purely for
            // the diagnostic training-loss curve.
            let predicted_flat = output_tensor
                .flatten_all()
                .and_then(|flat| flat.to_vec1::<f32>())
                .map_err(|e| {
                    Error::LegacyProcessing(format!("Failed to read model output: {e}"))
                })?;
            let predicted_channels =
                unflatten_channels(&predicted_flat, output_channels, buffer_size);
            batch_loss += self.compute_loss(&predicted_channels, target)?;

            // Real backward pass + real optimizer step: this is what actually
            // updates the model's weights, replacing the previous
            // "we simulate the training process" no-op.
            let target_flat = flatten_target(target, output_channels, buffer_size);
            let target_tensor = Tensor::from_vec(
                target_flat,
                output_tensor.shape().clone(),
                output_tensor.device(),
            )
            .map_err(|e| Error::LegacyProcessing(format!("Failed to build target tensor: {e}")))?;

            let diff = (&output_tensor - &target_tensor).map_err(|e| {
                Error::LegacyProcessing(format!("Loss tensor computation failed: {e}"))
            })?;
            let loss_tensor = diff.sqr().and_then(|sq| sq.mean_all()).map_err(|e| {
                Error::LegacyProcessing(format!("Loss tensor computation failed: {e}"))
            })?;

            optimizer.backward_step(&loss_tensor)?;

            // Apply data augmentation if enabled
            if self.config.augmentation.noise_injection {
                // Would apply noise injection here
            }
        }

        Ok(batch_loss / batch.len() as f32)
    }

    fn validate_batch(
        &self,
        model: &mut dyn NeuralModel,
        batch: &[(NeuralInputFeatures, Vec<Vec<f32>>)],
    ) -> Result<f32> {
        let mut batch_loss = 0.0;

        for (input, target) in batch {
            // Forward pass only (no gradient computation)
            let output = model.forward(input)?;

            // Compute loss
            let loss = self.compute_loss(&output.binaural_audio, target)?;
            batch_loss += loss;
        }

        Ok(batch_loss / batch.len() as f32)
    }

    fn compute_loss(&self, predicted: &[Vec<f32>], target: &[Vec<f32>]) -> Result<f32> {
        if predicted.len() != target.len() {
            return Err(Error::LegacyProcessing(
                "Predicted and target channel counts don't match".to_string(),
            ));
        }

        let mut total_loss = 0.0;
        let mut sample_count = 0;

        match self.config.loss_function {
            LossFunction::MSE => {
                for (pred_channel, target_channel) in predicted.iter().zip(target.iter()) {
                    let min_len = pred_channel.len().min(target_channel.len());
                    for i in 0..min_len {
                        let diff = pred_channel[i] - target_channel[i];
                        total_loss += diff * diff;
                        sample_count += 1;
                    }
                }
            }
            LossFunction::MAE => {
                for (pred_channel, target_channel) in predicted.iter().zip(target.iter()) {
                    let min_len = pred_channel.len().min(target_channel.len());
                    for i in 0..min_len {
                        let diff = (pred_channel[i] - target_channel[i]).abs();
                        total_loss += diff;
                        sample_count += 1;
                    }
                }
            }
            LossFunction::SpectralLoss => {
                // Simplified spectral loss - would implement FFT-based comparison
                for (pred_channel, target_channel) in predicted.iter().zip(target.iter()) {
                    let min_len = pred_channel.len().min(target_channel.len());
                    for i in 0..min_len {
                        let diff = pred_channel[i] - target_channel[i];
                        total_loss += diff * diff; // Simplified spectral approximation
                        sample_count += 1;
                    }
                }
                total_loss *= 1.2; // Weight spectral loss slightly higher
            }
            LossFunction::PerceptualLoss => {
                // Simplified perceptual loss based on psychoacoustic principles
                for (pred_channel, target_channel) in predicted.iter().zip(target.iter()) {
                    let min_len = pred_channel.len().min(target_channel.len());
                    for i in 0..min_len {
                        let diff = pred_channel[i] - target_channel[i];
                        // Apply perceptual weighting (simplified)
                        let perceptual_weight = 1.0 + 0.5 * (i as f32 / min_len as f32);
                        total_loss += diff * diff * perceptual_weight;
                        sample_count += 1;
                    }
                }
            }
            LossFunction::MultiScaleSpectralLoss => {
                // Multi-scale analysis at different time scales
                for (pred_channel, target_channel) in predicted.iter().zip(target.iter()) {
                    let min_len = pred_channel.len().min(target_channel.len());
                    // Multiple scales: full, half, quarter
                    for scale in [1, 2, 4] {
                        for i in (0..min_len).step_by(scale) {
                            let diff = pred_channel[i] - target_channel[i];
                            total_loss += diff * diff / (scale as f32);
                            sample_count += 1;
                        }
                    }
                }
            }
            LossFunction::Combined => {
                // Combination of MSE and spectral loss
                let mse_loss = self.compute_mse_loss(predicted, target)?;
                let spectral_loss = self.compute_spectral_loss(predicted, target)?;
                total_loss = 0.7 * mse_loss + 0.3 * spectral_loss;
                sample_count = 1; // Already normalized
            }
        }

        if sample_count > 0 {
            Ok(total_loss / sample_count as f32)
        } else {
            Ok(0.0)
        }
    }

    fn compute_mse_loss(&self, predicted: &[Vec<f32>], target: &[Vec<f32>]) -> Result<f32> {
        let mut total_loss = 0.0;
        let mut sample_count = 0;

        for (pred_channel, target_channel) in predicted.iter().zip(target.iter()) {
            let min_len = pred_channel.len().min(target_channel.len());
            for i in 0..min_len {
                let diff = pred_channel[i] - target_channel[i];
                total_loss += diff * diff;
                sample_count += 1;
            }
        }

        Ok(if sample_count > 0 {
            total_loss / sample_count as f32
        } else {
            0.0
        })
    }

    fn compute_spectral_loss(&self, predicted: &[Vec<f32>], target: &[Vec<f32>]) -> Result<f32> {
        // Simplified spectral loss computation
        // In a full implementation, this would use FFT to compare frequency domain representations
        let mut total_loss = 0.0;
        let mut sample_count = 0;

        for (pred_channel, target_channel) in predicted.iter().zip(target.iter()) {
            let min_len = pred_channel.len().min(target_channel.len());

            // Simple approximation: compare signal energy at different scales
            for window_size in [16, 32, 64, 128] {
                for start in (0..min_len).step_by(window_size / 2) {
                    let end = (start + window_size).min(min_len);
                    if end > start {
                        let pred_energy: f32 = pred_channel[start..end].iter().map(|x| x * x).sum();
                        let target_energy: f32 =
                            target_channel[start..end].iter().map(|x| x * x).sum();
                        let diff = pred_energy - target_energy;
                        total_loss += diff * diff;
                        sample_count += 1;
                    }
                }
            }
        }

        Ok(if sample_count > 0 {
            total_loss / sample_count as f32
        } else {
            0.0
        })
    }
}

/// Flatten per-channel target audio (`target[channel][frame]`) into the same
/// interleaved layout every [`NeuralModel`] emits (`flat[i]` corresponds to
/// `channel = i % output_channels`, `frame = i / output_channels`), zero-
/// padding any channel/frame the target data doesn't cover.
fn flatten_target(target: &[Vec<f32>], output_channels: usize, buffer_size: usize) -> Vec<f32> {
    let mut flat = vec![0.0f32; output_channels * buffer_size];
    for (i, sample) in flat.iter_mut().enumerate() {
        let channel = i % output_channels;
        let frame = i / output_channels;
        if let Some(value) = target.get(channel).and_then(|c| c.get(frame)) {
            *sample = *value;
        }
    }
    flat
}

/// Inverse of [`flatten_target`]: de-interleave a flat model output back into
/// per-channel audio, matching every [`NeuralModel`] implementation's
/// `tensor_to_binaural_audio` layout.
fn unflatten_channels(flat: &[f32], output_channels: usize, buffer_size: usize) -> Vec<Vec<f32>> {
    let mut channels = vec![Vec::with_capacity(buffer_size); output_channels];
    for (i, &sample) in flat.iter().enumerate() {
        let channel = i % output_channels;
        if channels[channel].len() < buffer_size {
            channels[channel].push(sample);
        }
    }
    channels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::models::FeedforwardModel;
    use crate::types::Position3D;
    use candle_core::Device;

    fn tiny_config() -> NeuralSpatialConfig {
        NeuralSpatialConfig {
            model_type: NeuralModelType::Feedforward,
            hidden_dims: vec![8],
            input_dim: 8,
            output_channels: 1,
            sample_rate: 48000,
            buffer_size: 4,
            use_gpu: false,
            quality: 0.8,
            realtime_constraints: RealtimeConstraints::default(),
            training_config: None,
        }
    }

    fn tiny_input() -> NeuralInputFeatures {
        NeuralInputFeatures {
            position: Position3D::new(0.5, -0.3, 0.2),
            listener_orientation: [1.0, 0.0, 0.0, 0.0],
            audio_features: Vec::new(),
            room_features: Vec::new(),
            hrtf_features: None,
            temporal_context: Vec::new(),
            user_features: None,
        }
    }

    /// Order-independent aggregate of every trainable parameter's values,
    /// used to prove weights really changed without depending on
    /// `VarMap::all_vars()` returning the same `HashMap` iteration order
    /// across two separate calls.
    fn total_weight_sum(vars: &[candle_core::Var]) -> f32 {
        vars.iter()
            .map(|v| {
                v.flatten_all()
                    .and_then(|t| t.sum_all())
                    .and_then(|t| t.to_scalar::<f32>())
                    .unwrap_or(0.0)
            })
            .sum()
    }

    #[test]
    fn test_training_actually_decreases_loss_and_changes_weights() {
        let config = tiny_config();
        let device = Device::Cpu;
        let mut model = FeedforwardModel::new(config, device).expect("model construction");

        let vars_before = model.trainable_vars();
        assert!(!vars_before.is_empty(), "model must expose trainable vars");
        let weight_sum_before = total_weight_sum(&vars_before);

        // A tiny, easily-fittable synthetic task: every sample has the same
        // input and the same small (tanh-representable) target, repeated
        // enough times to give both a training and a validation split.
        let target: Vec<Vec<f32>> = vec![vec![0.3, -0.2, 0.1, -0.1]];
        let training_data: Vec<(NeuralInputFeatures, Vec<Vec<f32>>)> =
            (0..10).map(|_| (tiny_input(), target.clone())).collect();

        let training_config = TrainingConfig {
            learning_rate: 1e-2,
            batch_size: 8,
            epochs: 200,
            validation_split: 0.2,
            loss_function: LossFunction::MSE,
            optimizer: OptimizerType::Adam,
            early_stopping_patience: 30,
            augmentation: AugmentationConfig {
                noise_injection: false,
                time_stretching: false,
                pitch_shifting: false,
                reverb_augmentation: false,
                gain_variation: 0.0,
            },
        };

        let mut trainer = NeuralTrainer::new(training_config);
        let results = trainer
            .train(&mut model, &training_data)
            .expect("training should succeed on a trivial fitting task");

        assert!(
            results.epochs_completed >= 2,
            "expected multiple epochs to run before early stopping, got {}",
            results.epochs_completed
        );

        let first_loss = *results
            .training_loss
            .first()
            .expect("training loss history should be non-empty");
        let last_loss = *results
            .training_loss
            .last()
            .expect("training loss history should be non-empty");
        assert!(
            last_loss < first_loss * 0.5,
            "training loss should drop substantially on this trivial fitting task: \
             first={first_loss}, last={last_loss}"
        );

        // Prove the weights really changed - the bug this replaces
        // (`update_parameters` only `println!`-ed "would update ...") left
        // every weight bit-for-bit identical to its construction-time value.
        let vars_after = model.trainable_vars();
        let weight_sum_after = total_weight_sum(&vars_after);
        assert_ne!(
            weight_sum_before, weight_sum_after,
            "training must actually mutate model weights, not just report fake progress"
        );
    }

    #[test]
    fn test_forward_is_deterministic_across_calls() {
        // Regression test for the sibling `Tensor::randn`-per-forward-call bug:
        // calling `forward` twice on an untrained model with the same input
        // must produce bit-identical output.
        let config = tiny_config();
        let model = FeedforwardModel::new(config, Device::Cpu).expect("model construction");
        let input = tiny_input();

        let first = model.forward(&input).expect("first forward pass");
        let second = model.forward(&input).expect("second forward pass");

        assert_eq!(
            first.binaural_audio, second.binaural_audio,
            "forward() must be deterministic for a fixed set of weights and input"
        );
    }

    #[test]
    fn test_rmsprop_is_honestly_rejected() {
        // RMSprop has no `candle_nn` implementation to delegate to; it must
        // fail closed with a clear error rather than silently train with a
        // different optimizer under the requested name.
        let config = TrainingConfig {
            optimizer: OptimizerType::RMSprop,
            ..TrainingConfig::default()
        };
        let result = SpatialOptimizer::new(vec![], &config);
        assert!(
            result.is_err(),
            "RMSprop must be rejected rather than silently substituted"
        );
    }
}
