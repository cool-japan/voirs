//! VITS Normalizing Flows implementation
//!
//! Implements invertible transformations for the latent space in VITS.
//! Uses coupling layers, invertible 1x1 convolutions, and activation normalization.

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};
use serde::{Deserialize, Serialize};

use crate::{AcousticError, Result};

/// Configuration for normalizing flows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowConfig {
    /// Number of flow layers
    pub n_flows: usize,
    /// Number of coupling layers per flow
    pub n_coupling_layers: usize,
    /// Hidden dimension for coupling networks
    pub hidden_dim: usize,
    /// Kernel size for convolutions
    pub kernel_size: usize,
    /// Number of channels
    pub n_channels: usize,
    /// Dropout probability
    pub dropout: f64,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            n_flows: 4,
            n_coupling_layers: 4,
            hidden_dim: 256,
            kernel_size: 5,
            n_channels: 80, // Should match mel_channels
            dropout: 0.0,
        }
    }
}

/// Activation Normalization layer
pub struct ActNorm {
    scale: Tensor,
    bias: Tensor,
    #[allow(dead_code)]
    initialized: bool,
}

impl ActNorm {
    pub fn new(n_channels: usize, device: &Device) -> CandleResult<Self> {
        let scale = Tensor::ones((n_channels, 1), DType::F32, device)?;
        let bias = Tensor::zeros((n_channels, 1), DType::F32, device)?;

        Ok(Self {
            scale,
            bias,
            initialized: false,
        })
    }

    /// Initialize parameters from first batch
    #[allow(dead_code)]
    fn initialize(&mut self, x: &Tensor) -> CandleResult<()> {
        if self.initialized {
            return Ok(());
        }

        // Compute mean and std across batch and time dimensions
        let dims = x.dims();
        if dims.len() != 3 {
            return Err(candle_core::Error::Msg(
                "ActNorm expects 3D input [batch, channels, time]".to_string(),
            ));
        }

        let (_, _n_channels, _) = x.dims3()?;

        // Compute statistics
        let mean = x.mean_keepdim(0)?.mean_keepdim(2)?; // [1, channels, 1]
        let mean = mean.squeeze(0)?.squeeze(1)?; // [channels, 1]

        // Manual variance computation: E[(x - mean)^2]
        let mean_broadcast = mean.unsqueeze(0)?.broadcast_as(x.dims())?;
        let diff = (x - mean_broadcast)?;
        let var = diff.sqr()?.mean_keepdim(0)?.mean_keepdim(2)?;
        let var = var.squeeze(0)?.squeeze(1)?; // [channels, 1]
        let std = (var + 1e-6)?.sqrt()?;

        // Initialize parameters to normalize to zero mean, unit variance
        self.bias = mean.neg()?;
        self.scale = std.recip()?;
        self.initialized = true;

        Ok(())
    }

    pub fn forward(&mut self, x: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        // Temporarily disable ActNorm to debug shape issues
        let (batch_size, _n_channels, _n_frames) = x.dims3()?;

        // Pass through input unchanged
        let y = x.clone();

        // Zero log determinant
        let log_det = Tensor::zeros((batch_size,), candle_core::DType::F32, x.device())?;

        Ok((y, log_det))
    }

    pub fn inverse(&self, y: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let (batch_size, _, n_frames) = y.dims3()?;

        // Inverse transformation: x = y / scale - bias
        let x = (y / &self.scale)? - &self.bias;

        // Compute negative log determinant
        let log_det_scale = self.scale.log()?.sum_all()?;
        let log_det = (log_det_scale * (n_frames as f64))?;
        let log_det = log_det.neg()?.broadcast_as((batch_size,))?;

        Ok((x?, log_det))
    }
}

/// Invertible 1x1 convolution
pub struct InvertibleConv1x1 {
    weight: Tensor,
    log_det_weight: Tensor,
}

impl InvertibleConv1x1 {
    pub fn new(n_channels: usize, device: &Device) -> CandleResult<Self> {
        // Initialize with random orthogonal matrix
        let mut weight_data = vec![0.0f32; n_channels * n_channels];

        // Create identity matrix as starting point
        for i in 0..n_channels {
            weight_data[i * n_channels + i] = 1.0;
        }

        // Add small random perturbations
        for weight in &mut weight_data {
            *weight += (fastrand::f32() - 0.5) * 0.1;
        }

        let weight = Tensor::from_vec(weight_data, (n_channels, n_channels), device)?;

        // For simplicity, assume determinant is 1 (log_det = 0)
        // In a full implementation, this would compute the actual determinant
        let log_det_weight = Tensor::zeros((), DType::F32, device)?;

        Ok(Self {
            weight,
            log_det_weight,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let (batch_size, n_channels, n_frames) = x.dims3()?;

        // Reshape for matrix multiplication: [batch*frames, channels]
        let x_reshaped = x
            .permute((0, 2, 1))?
            .reshape((batch_size * n_frames, n_channels))?;

        // Apply 1x1 convolution as matrix multiplication
        let y_reshaped = x_reshaped.matmul(&self.weight.t()?)?;

        // Reshape back: [batch, channels, frames]
        let y = y_reshaped
            .reshape((batch_size, n_frames, n_channels))?
            .permute((0, 2, 1))?;

        // Log determinant is constant for all time steps
        let log_det = (self.log_det_weight.broadcast_as((batch_size,))? * (n_frames as f64))?;

        Ok((y, log_det))
    }

    pub fn inverse(&self, y: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let (batch_size, n_channels, n_frames) = y.dims3()?;

        // For simplicity, use transpose as pseudo-inverse (assumes orthogonal matrix)
        // In a full implementation, this would compute the actual matrix inverse
        let weight_inv = self.weight.t()?;

        // Reshape for matrix multiplication
        let y_reshaped = y
            .permute((0, 2, 1))?
            .reshape((batch_size * n_frames, n_channels))?;

        // Apply inverse transformation
        let x_reshaped = y_reshaped.matmul(&weight_inv.t()?)?;

        // Reshape back
        let x = x_reshaped
            .reshape((batch_size, n_frames, n_channels))?
            .permute((0, 2, 1))?;

        // Negative log determinant
        let log_det =
            (self.log_det_weight.neg()?.broadcast_as((batch_size,))? * (n_frames as f64))?;

        Ok((x, log_det))
    }
}

/// Coupling layer using affine transformations
pub struct CouplingLayer {
    scale_net: WaveNet,
    translate_net: WaveNet,
    split_dim: usize,
}

impl CouplingLayer {
    pub fn new(
        n_channels: usize,
        hidden_dim: usize,
        kernel_size: usize,
        n_layers: usize,
        dropout: f64,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let split_dim = n_channels / 2;

        let scale_net = WaveNet::new(
            split_dim,
            split_dim,
            hidden_dim,
            kernel_size,
            n_layers,
            dropout,
            vb.pp("scale_net"),
        )?;

        let translate_net = WaveNet::new(
            split_dim,
            split_dim,
            hidden_dim,
            kernel_size,
            n_layers,
            dropout,
            vb.pp("translate_net"),
        )?;

        Ok(Self {
            scale_net,
            translate_net,
            split_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let (_batch_size, n_channels, _n_frames) = x.dims3()?;

        // Split input into two halves
        let x1 = x.narrow(1, 0, self.split_dim)?;
        let x2 = x.narrow(1, self.split_dim, n_channels - self.split_dim)?;

        // Compute scale and translation from first half
        let log_scale = self.scale_net.forward(&x1)?;
        let translation = self.translate_net.forward(&x1)?;

        // Apply affine transformation to second half
        let scale = log_scale.exp()?;
        let y2 = (&x2 * &scale)? + &translation;

        // Concatenate outputs
        let y = Tensor::cat(&[&x1, &y2?], 1)?;

        // Compute log determinant (sum of log scales)
        let log_det = log_scale.sum((1, 2))?; // Sum over channels and time

        Ok((y, log_det))
    }

    pub fn inverse(&self, y: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let (_batch_size, n_channels, _n_frames) = y.dims3()?;

        // Split input
        let y1 = y.narrow(1, 0, self.split_dim)?;
        let y2 = y.narrow(1, self.split_dim, n_channels - self.split_dim)?;

        // Compute scale and translation from first half
        let log_scale = self.scale_net.forward(&y1)?;
        let translation = self.translate_net.forward(&y1)?;

        // Apply inverse transformation to second half
        let scale = log_scale.exp()?;
        let x2 = (&y2 - &translation)? / &scale;

        // Concatenate outputs
        let x = Tensor::cat(&[&y1, &x2?], 1)?;

        // Negative log determinant
        let log_det = log_scale.sum((1, 2))?.neg()?;

        Ok((x, log_det))
    }
}

/// WaveNet-style network for coupling transformations
#[allow(dead_code)]
pub struct WaveNet {
    layers: Vec<Conv1d>,
    residual_layers: Vec<Conv1d>,
    skip_layers: Vec<Conv1d>,
    output_layer: Conv1d,
    n_layers: usize,
    dropout: f64,
}

impl WaveNet {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        hidden_dim: usize,
        kernel_size: usize,
        n_layers: usize,
        dropout: f64,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let mut layers = Vec::new();
        let mut residual_layers = Vec::new();
        let mut skip_layers = Vec::new();

        for i in 0..n_layers {
            let dilation = 2_usize.pow(i as u32);
            let padding = (kernel_size - 1) * dilation / 2;

            let conv_config = Conv1dConfig {
                padding,
                stride: 1,
                dilation,
                ..Default::default()
            };

            let in_dim = if i == 0 { in_channels } else { hidden_dim };

            // Main convolution layer
            let layer = candle_nn::conv1d(
                in_dim,
                hidden_dim,
                kernel_size,
                conv_config,
                vb.pp(format!("layer_{i}")),
            )?;
            layers.push(layer);

            // Residual connection
            let residual = candle_nn::conv1d(
                hidden_dim,
                hidden_dim,
                1,
                Default::default(),
                vb.pp(format!("residual_{i}")),
            )?;
            residual_layers.push(residual);

            // Skip connection
            let skip = candle_nn::conv1d(
                hidden_dim,
                hidden_dim,
                1,
                Default::default(),
                vb.pp(format!("skip_{i}")),
            )?;
            skip_layers.push(skip);
        }

        // Output layer
        let output_layer = candle_nn::conv1d(
            hidden_dim,
            out_channels,
            1,
            Default::default(),
            vb.pp("output"),
        )?;

        Ok(Self {
            layers,
            residual_layers,
            skip_layers,
            output_layer,
            n_layers,
            dropout,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Simplified WaveNet for debugging - just use a single linear transformation
        let (batch_size, in_channels, n_frames) = x.dims3()?;

        // Create a simple 1x1 convolution to transform input to output channels
        let weight = Tensor::randn(0.0f32, 0.1f32, (in_channels, in_channels), x.device())?;

        // Apply to each frame
        let x_reshaped = x
            .permute((0, 2, 1))?
            .reshape((batch_size * n_frames, in_channels))?;
        let output_reshaped = x_reshaped.matmul(&weight)?;
        let output = output_reshaped
            .reshape((batch_size, n_frames, in_channels))?
            .permute((0, 2, 1))?;

        Ok(output)
    }
}

/// Flow step combining all transformations
pub struct FlowStep {
    actnorm: ActNorm,
    inv_conv: InvertibleConv1x1,
    coupling: CouplingLayer,
}

impl FlowStep {
    pub fn new(
        n_channels: usize,
        hidden_dim: usize,
        kernel_size: usize,
        n_coupling_layers: usize,
        dropout: f64,
        device: &Device,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let actnorm = ActNorm::new(n_channels, device)?;
        let inv_conv = InvertibleConv1x1::new(n_channels, device)?;
        let coupling = CouplingLayer::new(
            n_channels,
            hidden_dim,
            kernel_size,
            n_coupling_layers,
            dropout,
            vb.pp("coupling"),
        )?;

        Ok(Self {
            actnorm,
            inv_conv,
            coupling,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        // Step 1: ActNorm
        let (z, log_det1) = self.actnorm.forward(x)?;

        // Step 2: Invertible 1x1 convolution
        let (z, log_det2) = self.inv_conv.forward(&z)?;

        // Step 3: Coupling layer
        let (z, log_det3) = self.coupling.forward(&z)?;

        // Total log determinant
        let log_det = ((&log_det1 + &log_det2)? + log_det3)?;

        Ok((z, log_det))
    }

    pub fn inverse(&self, z: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        // Inverse step 3: Coupling layer
        let (y, log_det3) = self.coupling.inverse(z)?;

        // Inverse step 2: Invertible 1x1 convolution
        let (y, log_det2) = self.inv_conv.inverse(&y)?;

        // Inverse step 1: ActNorm
        let (x, log_det1) = self.actnorm.inverse(&y)?;

        // Total log determinant
        let log_det = ((&log_det1 + &log_det2)? + log_det3)?;

        Ok((x, log_det))
    }
}

/// VITS Normalizing Flows
pub struct NormalizingFlows {
    #[allow(dead_code)]
    config: FlowConfig,
    device: Device,
    flow_steps: Vec<FlowStep>,
}

impl NormalizingFlows {
    pub fn new(config: FlowConfig, device: Device) -> Result<Self> {
        let vs = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vs, DType::F32, &device);

        Self::load_with_varbuilder(config, device, vb)
    }

    pub fn load_with_varbuilder(
        config: FlowConfig,
        device: Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut flow_steps = Vec::new();

        for i in 0..config.n_flows {
            let step = FlowStep::new(
                config.n_channels,
                config.hidden_dim,
                config.kernel_size,
                config.n_coupling_layers,
                config.dropout,
                &device,
                vb.pp(format!("flow_{i}")),
            )
            .map_err(|e| {
                AcousticError::ModelError(format!("Failed to create flow step {i}: {e}"))
            })?;

            flow_steps.push(step);
        }

        Ok(Self {
            config,
            device,
            flow_steps,
        })
    }

    /// Forward flow transformation: z_0 -> z_K
    pub fn forward(&mut self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut current_z = z.clone();
        let mut total_log_det =
            Tensor::zeros((z.dims()[0],), DType::F32, &self.device).map_err(|e| {
                AcousticError::ModelError(format!("Failed to create log_det tensor: {e}"))
            })?;

        tracing::debug!("NormalizingFlows forward: input shape {:?}", z.dims());

        for (i, step) in self.flow_steps.iter_mut().enumerate() {
            let (new_z, log_det) = step.forward(&current_z).map_err(|e| {
                AcousticError::ModelError(format!("Flow step {i} forward failed: {e}"))
            })?;

            current_z = new_z;
            total_log_det = (&total_log_det + log_det).map_err(|e| {
                AcousticError::ModelError(format!("Log det accumulation failed at step {i}: {e}"))
            })?;

            tracing::debug!("Flow step {}: output shape {:?}", i, current_z.dims());
        }

        Ok((current_z, total_log_det))
    }

    /// Inverse flow transformation: z_K -> z_0
    pub fn inverse(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut current_z = z.clone();
        let mut total_log_det =
            Tensor::zeros((z.dims()[0],), DType::F32, &self.device).map_err(|e| {
                AcousticError::ModelError(format!("Failed to create log_det tensor: {e}"))
            })?;

        tracing::debug!("NormalizingFlows inverse: input shape {:?}", z.dims());

        // Apply steps in reverse order
        for (i, step) in self.flow_steps.iter().enumerate().rev() {
            let (new_z, log_det) = step.inverse(&current_z).map_err(|e| {
                AcousticError::ModelError(format!("Flow step {i} inverse failed: {e}"))
            })?;

            current_z = new_z;
            total_log_det = (&total_log_det + log_det).map_err(|e| {
                AcousticError::ModelError(format!("Log det accumulation failed at step {i}: {e}"))
            })?;

            tracing::debug!(
                "Flow step {} inverse: output shape {:?}",
                i,
                current_z.dims()
            );
        }

        Ok((current_z, total_log_det))
    }
}
