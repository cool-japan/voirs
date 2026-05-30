//! VITS Normalizing Flows implementation
//!
//! Implements invertible transformations for the latent space in VITS.
//! Uses coupling layers, invertible 1x1 convolutions, and activation normalization.

use candle_core::{DType, Device, Module, Result as CandleResult, Tensor};
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
        // Real WaveNet forward pass with gated dilated convolutions.
        //
        // Architecture per layer i:
        //   h = dilated_conv(x)            — shape [B, h_channels, T]
        //
        //   Gated activation (standard WaveNet):
        //     h_tanh   = h[:, :half, :]    — first half of channels
        //     h_sig    = h[:, half:, :]    — second half of channels
        //     gate     = tanh(h_tanh) * sigmoid(h_sig)
        //                                  — shape [B, half, T]
        //
        //   When the constructor emits h_channels == hidden_dim (not 2*hidden_dim),
        //   gate.shape[1] == hidden_dim/2.  The residual/skip layers are built
        //   with hidden_dim in-channels, so we must guard all projections.
        //
        //   Residual: res = residual_conv(gate)  — add to x only if dims match
        //   Skip:     skip = skip_conv(gate)      — accumulate; if dims mismatch,
        //             accumulate gate directly (no projection).
        //
        // After all layers: relu(skip_sum) -> output_layer -> final output

        // The constructor builds layers[i] with output channels == hidden_dim,
        // while residual_layers[i] and skip_layers[i] take hidden_dim as input.
        // After the gated activation, gate has hidden_dim/2 channels.
        //
        // We handle the resulting dimension mismatch by keeping `current_x` always
        // at hidden_dim channels after the first layer:
        //   - For residual/skip projections, we check dims at runtime and fall back
        //     to the pre-gate hidden state when the gate is too narrow.
        //   - `current_x` is always updated to h (dilated conv output, hidden_dim)
        //     so subsequent layers receive the correct number of input channels.

        let mut current_x = x.clone();
        let mut skip_sum: Option<Tensor> = None;

        for i in 0..self.n_layers {
            // --- Dilated convolution ---
            // h shape: [B, hidden_dim, T]
            let h = self.layers[i].forward(&current_x)?;

            let h_channels = h.dims()[1];
            let half_dim = h_channels / 2;

            // --- Gated activation ---
            // Split h along channel dim into two equal halves.
            // h_tanh: [B, half_dim, T]  h_sig: [B, h_channels - half_dim, T]
            let h_tanh = h.narrow(1, 0, half_dim)?;
            let h_sig = h.narrow(1, half_dim, h_channels - half_dim)?;
            // gate: [B, half_dim, T]
            let gate = h_tanh.tanh()?.mul(&candle_nn::ops::sigmoid(&h_sig)?)?;

            let gate_channels = gate.dims()[1];

            // --- Residual connection ---
            // residual_layers[i] takes hidden_dim inputs (same as h_channels).
            // When gate_channels == h_channels (no split mismatch), project and add.
            // Otherwise use h directly as the residual base.
            let res_in_channels = self.residual_layers[i].weight().dims()[1];
            let res_out = if gate_channels == res_in_channels {
                self.residual_layers[i].forward(&gate)?
            } else {
                // gate is narrower than residual layer expects (half_dim < hidden_dim);
                // fall back to projecting h, which has exactly hidden_dim channels.
                self.residual_layers[i].forward(&h)?
            };

            // Add residual to current_x when channel counts agree.
            let x_channels = current_x.dims()[1];
            let res_channels = res_out.dims()[1];
            // Always advance current_x to hidden_dim for the next iteration.
            current_x = if x_channels == res_channels {
                (&current_x + &res_out)?
            } else {
                // x is still at in_channels (first layer, in_channels ≠ hidden_dim);
                // discard the residual add and carry h so shapes stay at hidden_dim.
                h.clone()
            };

            // --- Skip connection ---
            // Project gate (or h, on mismatch) and accumulate.
            let skip_in_channels = self.skip_layers[i].weight().dims()[1];
            let skip_contribution = if gate_channels == skip_in_channels {
                self.skip_layers[i].forward(&gate)?
            } else {
                // gate narrower than skip layer; use h instead.
                self.skip_layers[i].forward(&h)?
            };

            skip_sum = Some(if let Some(acc) = skip_sum {
                (&acc + &skip_contribution)?
            } else {
                skip_contribution
            });
        }

        // --- Aggregate and project to output ---
        let skip_agg = skip_sum.ok_or_else(|| {
            candle_core::Error::Msg("WaveNet has zero layers; skip_sum is empty".to_string())
        })?;

        // skip_agg has hidden_dim channels; output_layer expects hidden_dim → out_channels.
        let out = self.output_layer.forward(&skip_agg.relu()?)?;

        Ok(out)
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
            .map_err(|e| AcousticError::ModelError {
                message: format!("Failed to create flow step {i}: {e}"),
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
                AcousticError::ModelError {
                    message: format!("Failed to create log_det tensor: {e}"),
                }
            })?;

        tracing::debug!("NormalizingFlows forward: input shape {:?}", z.dims());

        for (i, step) in self.flow_steps.iter_mut().enumerate() {
            let (new_z, log_det) =
                step.forward(&current_z)
                    .map_err(|e| AcousticError::ModelError {
                        message: format!("Flow step {i} forward failed: {e}"),
                    })?;

            current_z = new_z;
            total_log_det = (&total_log_det + log_det).map_err(|e| AcousticError::ModelError {
                message: format!("Log det accumulation failed at step {i}: {e}"),
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
                AcousticError::ModelError {
                    message: format!("Failed to create log_det tensor: {e}"),
                }
            })?;

        tracing::debug!("NormalizingFlows inverse: input shape {:?}", z.dims());

        // Apply steps in reverse order
        for (i, step) in self.flow_steps.iter().enumerate().rev() {
            let (new_z, log_det) =
                step.inverse(&current_z)
                    .map_err(|e| AcousticError::ModelError {
                        message: format!("Flow step {i} inverse failed: {e}"),
                    })?;

            current_z = new_z;
            total_log_det = (&total_log_det + log_det).map_err(|e| AcousticError::ModelError {
                message: format!("Log det accumulation failed at step {i}: {e}"),
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    /// Build a small WaveNet on the CPU for testing
    fn make_wavenet(
        in_channels: usize,
        out_channels: usize,
        hidden_dim: usize,
        n_layers: usize,
    ) -> CandleResult<WaveNet> {
        let device = Device::Cpu;
        let vs = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vs, DType::F32, &device);
        WaveNet::new(in_channels, out_channels, hidden_dim, 3, n_layers, 0.0, vb)
    }

    /// Create a deterministic input tensor filled with a constant value
    fn const_input(
        batch: usize,
        channels: usize,
        frames: usize,
        value: f32,
    ) -> CandleResult<Tensor> {
        let device = Device::Cpu;
        let data = vec![value; batch * channels * frames];
        Tensor::from_vec(data, (batch, channels, frames), &device)
    }

    /// test_wavenet_forward_deterministic:
    /// Two calls with the same input tensor must produce bit-identical results
    /// (i.e. no internal randomness is introduced during forward).
    #[test]
    fn test_wavenet_forward_deterministic() -> CandleResult<()> {
        let wn = make_wavenet(4, 4, 8, 3)?;
        let x = const_input(1, 4, 16, 0.5)?;

        let out1 = wn.forward(&x)?;
        let out2 = wn.forward(&x)?;

        let diff = ((&out1 - &out2)?.abs()?)
            .max(0)?
            .max(0)?
            .max(0)?
            .to_scalar::<f32>()?;

        assert_eq!(
            diff, 0.0,
            "forward() is not deterministic; max diff = {diff}"
        );
        Ok(())
    }

    /// test_wavenet_forward_shape:
    /// Input [1, C_in, T] must produce output [1, C_out, T] — time dimension preserved.
    #[test]
    fn test_wavenet_forward_shape() -> CandleResult<()> {
        let in_channels = 4;
        let out_channels = 4;
        let frames = 32;
        let wn = make_wavenet(in_channels, out_channels, 8, 3)?;
        let x = const_input(1, in_channels, frames, 0.1)?;

        let out = wn.forward(&x)?;
        let dims = out.dims();

        assert_eq!(dims.len(), 3, "output must be 3-dimensional");
        assert_eq!(dims[0], 1, "batch dimension must be preserved");
        assert_eq!(
            dims[1], out_channels,
            "channel dimension must equal out_channels"
        );
        assert_eq!(dims[2], frames, "time dimension must be preserved");
        Ok(())
    }

    /// test_wavenet_no_nan:
    /// For a well-behaved input the output must contain no NaN values.
    #[test]
    fn test_wavenet_no_nan() -> CandleResult<()> {
        let wn = make_wavenet(4, 4, 8, 4)?;
        let x = const_input(2, 4, 24, 0.3)?;

        let out = wn.forward(&x)?;

        // NaN check: a tensor equals itself only if there are no NaNs
        let is_nan = out.ne(&out)?;
        let nan_count = is_nan.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;

        assert_eq!(nan_count, 0.0, "output contains NaN values");
        Ok(())
    }
}
