//! Advanced sampling algorithms for DiffWave diffusion.
//!
//! This module implements DDPM, DDIM, and fast sampling techniques
//! for high-quality audio generation from mel spectrograms.

use candle_core::{Device, Result as CandleResult, Tensor};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

use super::{EnhancedUNet, NoiseScheduler, NoiseSchedulerConfig};
use crate::Result;

/// Sampling algorithm types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SamplingAlgorithm {
    /// DDPM (Denoising Diffusion Probabilistic Models) - Original algorithm
    DDPM,
    /// DDIM (Denoising Diffusion Implicit Models) - Deterministic and faster
    DDIM,
    /// Fast DDIM with reduced steps
    FastDDIM,
    /// Adaptive sampling based on noise level
    Adaptive,
    /// DPM-Solver++ - Fast high-order solver (2-5 steps for good quality)
    DPMSolverPlusPlus,
    /// UniPC - Unified Predictor-Corrector (5-10 steps for excellent quality)
    UniPC,
}

/// Configuration for diffusion sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Sampling algorithm to use
    pub algorithm: SamplingAlgorithm,
    /// Number of sampling steps (fewer = faster, more = better quality)
    pub num_steps: u32,
    /// DDIM eta parameter (0.0 = deterministic, 1.0 = stochastic like DDPM)
    pub eta: f32,
    /// Temperature for sampling (higher = more random)
    pub temperature: f32,
    /// Guidance scale for classifier-free guidance
    pub guidance_scale: f32,
    /// Whether to use classifier-free guidance
    pub use_guidance: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            algorithm: SamplingAlgorithm::DDIM,
            num_steps: 50,
            eta: 0.0,
            temperature: 1.0,
            guidance_scale: 7.5,
            use_guidance: false,
            seed: None,
        }
    }
}

/// Statistics about the sampling process
#[derive(Debug, Clone)]
pub struct SamplingStats {
    pub algorithm_used: SamplingAlgorithm,
    pub total_steps: u32,
    pub actual_steps: u32,
    pub avg_step_time_ms: f32,
    pub total_time_ms: f32,
    pub convergence_score: f32,
}

/// Advanced diffusion sampler with multiple algorithms
#[derive(Debug, Clone)]
pub struct DiffusionSampler {
    config: SamplingConfig,
    scheduler: NoiseScheduler,
    device: Device,
}

impl DiffusionSampler {
    /// Create a new diffusion sampler
    pub fn new(config: SamplingConfig, scheduler: NoiseScheduler, device: Device) -> Result<Self> {
        Ok(Self {
            config,
            scheduler,
            device,
        })
    }

    /// Generate audio using the configured sampling algorithm
    pub fn sample(
        &self,
        unet: &EnhancedUNet,
        shape: &[usize],
        mel_condition: &Tensor,
    ) -> CandleResult<(Tensor, SamplingStats)> {
        let start_time = std::time::Instant::now();

        let result = match self.config.algorithm {
            SamplingAlgorithm::DDPM => self.ddpm_sample(unet, shape, mel_condition),
            SamplingAlgorithm::DDIM => self.ddim_sample(unet, shape, mel_condition),
            SamplingAlgorithm::FastDDIM => self.fast_ddim_sample(unet, shape, mel_condition),
            SamplingAlgorithm::Adaptive => self.adaptive_sample(unet, shape, mel_condition),
            SamplingAlgorithm::DPMSolverPlusPlus => {
                self.dpm_solver_plusplus_sample(unet, shape, mel_condition)
            }
            SamplingAlgorithm::UniPC => self.unipc_sample(unet, shape, mel_condition),
        };

        let total_time = start_time.elapsed().as_millis() as f32;

        match result {
            Ok(audio) => {
                // Compute convergence score based on audio quality metrics
                let convergence_score = self.compute_convergence_score(&audio)?;

                let stats = SamplingStats {
                    algorithm_used: self.config.algorithm,
                    total_steps: self.config.num_steps,
                    actual_steps: self.config.num_steps,
                    avg_step_time_ms: total_time / self.config.num_steps as f32,
                    total_time_ms: total_time,
                    convergence_score,
                };
                Ok((audio, stats))
            }
            Err(e) => Err(e),
        }
    }

    /// DDPM sampling - original stochastic algorithm
    fn ddpm_sample(
        &self,
        unet: &EnhancedUNet,
        shape: &[usize],
        mel_condition: &Tensor,
    ) -> CandleResult<Tensor> {
        // Start from random noise
        let mut x = self.sample_noise(shape)?;

        let num_steps = self.config.num_steps as usize;
        let total_timesteps = self.scheduler.config().num_steps as usize;

        // Create timestep schedule
        let step_size = total_timesteps / num_steps;
        let timesteps: Vec<usize> = (0..num_steps)
            .map(|i| total_timesteps - 1 - i * step_size)
            .collect();

        for &t in &timesteps {
            // Create timestep tensor
            let t_tensor = Tensor::new(&[t as f32], &self.device)?;

            // Predict noise
            let predicted_noise = unet.forward(&x, &t_tensor, mel_condition)?;

            // DDPM reverse step
            x = self.ddpm_step(&x, &predicted_noise, t)?;

            // Apply temperature scaling
            if self.config.temperature != 1.0 {
                x = x.affine(self.config.temperature as f64, 0.0)?;
            }
        }

        Ok(x)
    }

    /// DDIM sampling - deterministic and faster
    fn ddim_sample(
        &self,
        unet: &EnhancedUNet,
        shape: &[usize],
        mel_condition: &Tensor,
    ) -> CandleResult<Tensor> {
        // Start from random noise
        let mut x = self.sample_noise(shape)?;

        let num_steps = self.config.num_steps as usize;
        let total_timesteps = self.scheduler.config().num_steps as usize;

        // Create DDIM timestep schedule
        let timesteps: Vec<usize> = (0..num_steps)
            .map(|i| total_timesteps * i / num_steps)
            .rev()
            .collect();

        for i in 0..timesteps.len() {
            let t = timesteps[i];
            let prev_t = if i == timesteps.len() - 1 {
                0
            } else {
                timesteps[i + 1]
            };

            // Create timestep tensor
            let t_tensor = Tensor::new(&[t as f32], &self.device)?;

            // Predict noise
            let predicted_noise = unet.forward(&x, &t_tensor, mel_condition)?;

            // DDIM reverse step
            x = self.ddim_step(&x, &predicted_noise, t, prev_t)?;
        }

        Ok(x)
    }

    /// Fast DDIM with significantly reduced steps
    fn fast_ddim_sample(
        &self,
        unet: &EnhancedUNet,
        shape: &[usize],
        mel_condition: &Tensor,
    ) -> CandleResult<Tensor> {
        // Use fewer steps for speed
        let fast_steps = (self.config.num_steps / 4).max(10);

        let mut fast_config = self.config.clone();
        fast_config.num_steps = fast_steps;

        // Create fast sampler directly without error conversion
        let fast_scheduler = self.scheduler.clone();
        let fast_sampler = DiffusionSampler {
            config: fast_config,
            scheduler: fast_scheduler,
            device: self.device.clone(),
        };
        fast_sampler.ddim_sample(unet, shape, mel_condition)
    }

    /// Adaptive sampling that adjusts steps based on convergence
    fn adaptive_sample(
        &self,
        unet: &EnhancedUNet,
        shape: &[usize],
        mel_condition: &Tensor,
    ) -> CandleResult<Tensor> {
        // Start with DDIM but monitor convergence
        let mut x = self.sample_noise(shape)?;
        let mut prev_x = x.clone();

        let num_steps = self.config.num_steps as usize;
        let total_timesteps = self.scheduler.config().num_steps as usize;

        let timesteps: Vec<usize> = (0..num_steps)
            .map(|i| total_timesteps * i / num_steps)
            .rev()
            .collect();

        for i in 0..timesteps.len() {
            let t = timesteps[i];
            let prev_t = if i == timesteps.len() - 1 {
                0
            } else {
                timesteps[i + 1]
            };

            let t_tensor = Tensor::new(&[t as f32], &self.device)?;
            let predicted_noise = unet.forward(&x, &t_tensor, mel_condition)?;

            x = self.ddim_step(&x, &predicted_noise, t, prev_t)?;

            // Check convergence (simplified)
            if i > 5 {
                let diff = x.sub(&prev_x)?.abs()?.mean_all()?;
                let diff_value: f32 = diff.to_vec0()?;
                if diff_value < 0.001 {
                    // Early convergence - can stop sampling
                    break;
                }
            }

            prev_x = x.clone();
        }

        Ok(x)
    }

    /// DPM-Solver++ sampling - Fast high-order ODE solver for diffusion models
    ///
    /// DPM-Solver++ is a second-order solver that achieves excellent quality with very few steps (2-20).
    /// It uses semi-linear structure of diffusion ODEs for efficient integration.
    ///
    /// Reference: "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models"
    /// Lu et al., 2022 (NeurIPS)
    fn dpm_solver_plusplus_sample(
        &self,
        unet: &EnhancedUNet,
        shape: &[usize],
        mel_condition: &Tensor,
    ) -> CandleResult<Tensor> {
        // Start from random noise
        let mut x = self.sample_noise(shape)?;

        let num_steps = self.config.num_steps.max(2) as usize;
        let total_timesteps = self.scheduler.config().num_steps as usize;

        // Create timestep schedule (uniform spacing in noise level)
        let timesteps: Vec<usize> = (0..num_steps)
            .map(|i| total_timesteps * i / num_steps)
            .rev()
            .collect();

        // Store previous model outputs for higher-order steps
        let mut model_outputs: Vec<Tensor> = Vec::new();

        for i in 0..timesteps.len() {
            let t = timesteps[i];
            let prev_t = if i == timesteps.len() - 1 {
                0
            } else {
                timesteps[i + 1]
            };

            let t_tensor = Tensor::new(&[t as f32], &self.device)?;

            // Predict noise (model output)
            let model_output = unet.forward(&x, &t_tensor, mel_condition)?;

            // Use second-order solver if we have previous step
            if i > 0 && i < timesteps.len() - 1 && !model_outputs.is_empty() {
                // Second-order multistep DPM-Solver++
                x = self.dpm_solver_second_order_step(
                    &x,
                    &model_output,
                    &model_outputs[model_outputs.len() - 1],
                    t,
                    prev_t,
                    timesteps.get(i + 2).copied().unwrap_or(0),
                )?;
            } else {
                // First-order step
                x = self.dpm_solver_first_order_step(&x, &model_output, t, prev_t)?;
            }

            // Store model output for next iteration
            model_outputs.push(model_output);

            // Keep only last 2 outputs for memory efficiency
            if model_outputs.len() > 2 {
                model_outputs.remove(0);
            }
        }

        Ok(x)
    }

    /// UniPC sampling - Unified Predictor-Corrector for fast high-quality sampling
    ///
    /// UniPC combines predictor and corrector steps with unified formulation.
    /// Achieves excellent quality in 5-10 steps.
    ///
    /// Reference: "UniPC: A Unified Predictor-Corrector Framework for Fast Sampling"
    /// Zhao et al., 2023 (NeurIPS)
    fn unipc_sample(
        &self,
        unet: &EnhancedUNet,
        shape: &[usize],
        mel_condition: &Tensor,
    ) -> CandleResult<Tensor> {
        // Start from random noise
        let mut x = self.sample_noise(shape)?;

        let num_steps = self.config.num_steps.max(5) as usize;
        let total_timesteps = self.scheduler.config().num_steps as usize;

        // Create timestep schedule
        let timesteps: Vec<usize> = (0..num_steps)
            .map(|i| total_timesteps * i / num_steps)
            .rev()
            .collect();

        // Model output cache for predictor-corrector
        let mut model_outputs: Vec<Tensor> = Vec::new();

        for i in 0..timesteps.len() {
            let t = timesteps[i];
            let prev_t = if i == timesteps.len() - 1 {
                0
            } else {
                timesteps[i + 1]
            };

            let t_tensor = Tensor::new(&[t as f32], &self.device)?;

            // Predictor step: Predict next state
            let model_output = unet.forward(&x, &t_tensor, mel_condition)?;

            // First step or when we don't have enough history: use first-order
            if i == 0 || model_outputs.is_empty() {
                x = self.dpm_solver_first_order_step(&x, &model_output, t, prev_t)?;
                model_outputs.push(model_output);
                continue;
            }

            // Predictor: Use multi-step prediction
            let x_pred = if model_outputs.len() >= 2 {
                // Second-order predictor
                self.dpm_solver_second_order_step(
                    &x,
                    &model_output,
                    &model_outputs[model_outputs.len() - 1],
                    t,
                    prev_t,
                    timesteps.get(i + 2).copied().unwrap_or(0),
                )?
            } else {
                // First-order predictor
                self.dpm_solver_first_order_step(&x, &model_output, t, prev_t)?
            };

            // Corrector step: Refine the prediction
            if i < timesteps.len() - 1 {
                let prev_t_tensor = Tensor::new(&[prev_t as f32], &self.device)?;
                let corrector_output = unet.forward(&x_pred, &prev_t_tensor, mel_condition)?;

                // Combine predictor and corrector
                let alpha = 0.5_f32; // Balance between predictor and corrector
                let corrected = x_pred.affine(alpha as f64, 0.0)?;
                let correction_term = self
                    .dpm_solver_first_order_step(&x_pred, &corrector_output, prev_t, 0)?
                    .affine((1.0 - alpha) as f64, 0.0)?;
                x = corrected.add(&correction_term)?;
            } else {
                x = x_pred;
            }

            model_outputs.push(model_output);
            if model_outputs.len() > 3 {
                model_outputs.remove(0);
            }
        }

        Ok(x)
    }

    /// DPM-Solver first-order step
    fn dpm_solver_first_order_step(
        &self,
        x: &Tensor,
        model_output: &Tensor,
        t: usize,
        prev_t: usize,
    ) -> CandleResult<Tensor> {
        // Get noise schedule parameters
        let alpha_t = self.scheduler.alphas_cumprod()[t];
        let alpha_prev = if prev_t == 0 {
            1.0_f32
        } else {
            self.scheduler.alphas_cumprod()[prev_t]
        };

        let lambda_t = ((alpha_t / (1.0 - alpha_t)).ln()) as f64;
        let lambda_prev = ((alpha_prev / (1.0 - alpha_prev)).ln()) as f64;
        let h = lambda_prev - lambda_t;

        // Predict x_0
        let alpha_t_sqrt = (alpha_t.sqrt()) as f64;
        let sigma_t = ((1.0 - alpha_t).sqrt()) as f64;

        let x0_pred = x
            .affine(1.0 / alpha_t_sqrt, 0.0)?
            .sub(&model_output.affine(sigma_t / alpha_t_sqrt, 0.0)?)?;

        // First-order exponential integrator
        let alpha_prev_sqrt = (alpha_prev.sqrt()) as f64;
        let sigma_prev = ((1.0 - alpha_prev).sqrt()) as f64;

        let x_prev = x0_pred
            .affine(alpha_prev_sqrt, 0.0)?
            .add(&model_output.affine(sigma_prev * (-h).exp(), 0.0)?)?;

        Ok(x_prev)
    }

    /// DPM-Solver++ second-order step using multistep method
    fn dpm_solver_second_order_step(
        &self,
        x: &Tensor,
        model_output: &Tensor,
        model_output_prev: &Tensor,
        t: usize,
        prev_t: usize,
        _prev_prev_t: usize,
    ) -> CandleResult<Tensor> {
        // Get noise schedule parameters
        let alpha_t = self.scheduler.alphas_cumprod()[t];
        let alpha_prev = if prev_t == 0 {
            1.0_f32
        } else {
            self.scheduler.alphas_cumprod()[prev_t]
        };

        let lambda_t = ((alpha_t / (1.0 - alpha_t)).ln()) as f64;
        let lambda_prev = ((alpha_prev / (1.0 - alpha_prev)).ln()) as f64;
        let h = lambda_prev - lambda_t;

        // Predict x_0
        let alpha_t_sqrt = (alpha_t.sqrt()) as f64;
        let sigma_t = ((1.0 - alpha_t).sqrt()) as f64;

        let x0_pred = x
            .affine(1.0 / alpha_t_sqrt, 0.0)?
            .sub(&model_output.affine(sigma_t / alpha_t_sqrt, 0.0)?)?;

        // Second-order correction using previous model output
        let model_diff = model_output.sub(model_output_prev)?;

        // Second-order exponential integrator
        let alpha_prev_sqrt = (alpha_prev.sqrt()) as f64;
        let sigma_prev = ((1.0 - alpha_prev).sqrt()) as f64;

        // Linear multistep coefficient
        let r = 0.5; // Second-order coefficient

        let x_prev = x0_pred
            .affine(alpha_prev_sqrt, 0.0)?
            .add(&model_output.affine(sigma_prev * (-h).exp(), 0.0)?)?
            .add(&model_diff.affine(r * sigma_prev * (-h).exp(), 0.0)?)?;

        Ok(x_prev)
    }

    /// DDPM reverse diffusion step
    fn ddpm_step(
        &self,
        x: &Tensor,
        predicted_noise: &Tensor,
        timestep: usize,
    ) -> CandleResult<Tensor> {
        let alpha_t = self.scheduler.alphas_cumprod()[timestep];
        let beta_t = self.scheduler.betas()[timestep];

        let alpha_t_sqrt = alpha_t.sqrt();
        let one_minus_alpha_t_sqrt = (1.0_f32 - alpha_t).sqrt();

        // Predict x0 from noise
        let _x0_pred = (x.affine(1.0 / alpha_t_sqrt as f64, 0.0)?
            - predicted_noise.affine(one_minus_alpha_t_sqrt as f64 / alpha_t_sqrt as f64, 0.0)?)?;

        // Compute direction pointing to x_t
        let direction = predicted_noise.affine(beta_t.sqrt() as f64, 0.0)?;

        // Compute x_{t-1}
        let x_prev = x.sub(&direction)?;

        // Add noise for stochasticity (DDPM)
        if timestep > 0 {
            let noise = self.sample_noise(x.dims())?;
            let sigma = beta_t.sqrt();
            let x_prev = x_prev.add(&noise.affine(sigma as f64, 0.0)?)?;
            Ok(x_prev)
        } else {
            Ok(x_prev)
        }
    }

    /// DDIM reverse diffusion step
    fn ddim_step(
        &self,
        x: &Tensor,
        predicted_noise: &Tensor,
        timestep: usize,
        prev_timestep: usize,
    ) -> CandleResult<Tensor> {
        let alpha_t = self.scheduler.alphas_cumprod()[timestep];
        let alpha_t_prev = if prev_timestep == 0 {
            1.0_f32
        } else {
            self.scheduler.alphas_cumprod()[prev_timestep]
        };

        let alpha_t_sqrt = alpha_t.sqrt();
        let one_minus_alpha_t_sqrt = (1.0_f32 - alpha_t).sqrt();

        // Predict x0
        let x0_pred = (x.affine(1.0 / alpha_t_sqrt as f64, 0.0)?
            - predicted_noise.affine(one_minus_alpha_t_sqrt as f64 / alpha_t_sqrt as f64, 0.0)?)?;

        // Compute direction for x_t
        let alpha_t_prev_sqrt = alpha_t_prev.sqrt();
        let one_minus_alpha_t_prev: f32 = 1.0 - alpha_t_prev;

        // DDIM deterministic step
        let x_prev =
            x0_pred
                .affine(alpha_t_prev_sqrt as f64, 0.0)?
                .add(&predicted_noise.affine(
                    one_minus_alpha_t_prev.sqrt() as f64 * self.config.eta as f64,
                    0.0,
                )?)?;

        Ok(x_prev)
    }

    /// Sample random noise tensor
    fn sample_noise(&self, shape: &[usize]) -> CandleResult<Tensor> {
        let device = &self.device;

        // Generate random noise
        let total_elements: usize = shape.iter().product();
        let mut noise_data = Vec::with_capacity(total_elements);

        // Use Box-Muller transform for Gaussian noise
        for _ in 0..total_elements {
            let u1: f32 = fastrand::f32();
            let u2: f32 = fastrand::f32();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            noise_data.push(noise * self.config.temperature);
        }

        Tensor::from_vec(noise_data, shape, device)
    }

    /// Compute convergence score based on audio quality metrics
    ///
    /// This method evaluates the quality of the generated audio to estimate
    /// how well the diffusion process has converged. The score ranges from 0.0 to 1.0,
    /// with higher values indicating better convergence.
    ///
    /// Metrics considered:
    /// 1. Signal energy distribution (should be neither too concentrated nor too sparse)
    /// 2. Dynamic range utilization (good audio uses reasonable dynamic range)
    /// 3. Variance consistency (stable variance indicates good convergence)
    /// 4. Peak-to-RMS ratio (indicates healthy signal structure)
    fn compute_convergence_score(&self, audio: &Tensor) -> CandleResult<f32> {
        // Convert tensor to flat vector for analysis
        let audio_vec = audio.flatten_all()?.to_vec1::<f32>()?;

        if audio_vec.is_empty() {
            return Ok(0.0);
        }

        let n = audio_vec.len() as f32;

        // 1. Compute signal energy (RMS)
        let energy: f32 = audio_vec.iter().map(|&x| x * x).sum::<f32>() / n;
        let rms = energy.sqrt();

        // 2. Compute peak amplitude
        let peak = audio_vec.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

        // 3. Peak-to-RMS ratio (healthy audio typically has ratio between 3-10)
        let peak_to_rms = if rms > 1e-10 { peak / rms } else { 0.0 };
        let peak_score = if peak_to_rms >= 3.0 && peak_to_rms <= 10.0 {
            1.0
        } else if peak_to_rms > 10.0 {
            // Too spiky, likely not well converged
            (15.0 - peak_to_rms).max(0.0) / 5.0
        } else {
            // Too flat, also not ideal
            peak_to_rms / 3.0
        };

        // 4. Energy distribution score (audio should have reasonable energy)
        // Ideal RMS is around 0.1-0.5 for normalized audio
        let energy_score = if rms >= 0.05 && rms <= 0.7 {
            1.0
        } else if rms < 0.05 {
            (rms / 0.05).min(1.0)
        } else {
            (1.0 - (rms - 0.7) / 0.3).max(0.0)
        };

        // 5. Dynamic range utilization (good audio uses reasonable range)
        // Peak should be between 0.3 and 1.0 for well-converged audio
        let dynamic_score = if peak >= 0.3 && peak <= 1.0 {
            1.0
        } else if peak < 0.3 {
            (peak / 0.3).min(1.0)
        } else {
            // Clipping indicator
            0.7
        };

        // 6. Variance consistency (measure local variance stability)
        let chunk_size = (audio_vec.len() / 10).max(1);
        let mut chunk_variances = Vec::new();
        for chunk_start in (0..audio_vec.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(audio_vec.len());
            let chunk = &audio_vec[chunk_start..chunk_end];
            let chunk_mean = chunk.iter().sum::<f32>() / chunk.len() as f32;
            let chunk_var =
                chunk.iter().map(|&x| (x - chunk_mean).powi(2)).sum::<f32>() / chunk.len() as f32;
            chunk_variances.push(chunk_var);
        }

        let variance_mean = chunk_variances.iter().sum::<f32>() / chunk_variances.len() as f32;
        let variance_std = (chunk_variances
            .iter()
            .map(|&v| (v - variance_mean).powi(2))
            .sum::<f32>()
            / chunk_variances.len() as f32)
            .sqrt();

        let variance_stability = if variance_mean > 1e-10 {
            let cv = variance_std / variance_mean; // Coefficient of variation
                                                   // Lower CV means more stable variance (better convergence)
            (1.0 - cv.min(1.0)).max(0.0)
        } else {
            0.5 // Neutral score if signal is too quiet
        };

        // Combine scores with weights
        let convergence_score = 0.25 * peak_score
            + 0.25 * energy_score
            + 0.25 * dynamic_score
            + 0.25 * variance_stability;

        Ok(convergence_score.clamp(0.0, 1.0))
    }

    /// Get sampling configuration
    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }

    /// Update sampling configuration
    pub fn set_config(&mut self, config: SamplingConfig) {
        self.config = config;
    }
}

// Note: Clone implementation is in schedule.rs

#[cfg(test)]
mod tests {
    use super::*;
    // use candle_core::Device;

    #[test]
    fn test_sampling_config() {
        let config = SamplingConfig::default();
        assert_eq!(config.num_steps, 50);
        assert_eq!(config.eta, 0.0);
        assert!(matches!(config.algorithm, SamplingAlgorithm::DDIM));
    }

    #[test]
    fn test_sampling_algorithms() {
        // Test that all algorithm variants can be created
        let algorithms = [
            SamplingAlgorithm::DDPM,
            SamplingAlgorithm::DDIM,
            SamplingAlgorithm::FastDDIM,
            SamplingAlgorithm::Adaptive,
            SamplingAlgorithm::DPMSolverPlusPlus,
            SamplingAlgorithm::UniPC,
        ];

        for algorithm in &algorithms {
            let config = SamplingConfig {
                algorithm: *algorithm,
                ..Default::default()
            };
            assert_eq!(config.algorithm as u8, *algorithm as u8);
        }
    }

    #[test]
    fn test_sampling_stats() {
        let stats = SamplingStats {
            algorithm_used: SamplingAlgorithm::DDIM,
            total_steps: 50,
            actual_steps: 50,
            avg_step_time_ms: 10.0,
            total_time_ms: 500.0,
            convergence_score: 0.95,
        };

        assert_eq!(stats.total_steps, 50);
        assert_eq!(stats.total_time_ms, 500.0);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_convergence_score_healthy_signal() {
        use candle_core::{Device, Tensor};

        let device = Device::Cpu;
        let config = SamplingConfig::default();
        let scheduler_config = NoiseSchedulerConfig::default();
        let scheduler = NoiseScheduler::new(scheduler_config, &device).unwrap();
        let sampler = DiffusionSampler::new(config, scheduler, device.clone()).unwrap();

        // Create a healthy audio signal with good dynamics
        let audio_data: Vec<f32> = (0..44100)
            .map(|i| {
                let t = i as f32 / 44100.0;
                // Mix of frequencies with good dynamic range
                0.3 * (2.0 * PI * 440.0 * t).sin() + 0.2 * (2.0 * PI * 880.0 * t).sin()
            })
            .collect();

        let audio = Tensor::from_vec(audio_data, &[1, 44100], &device).unwrap();
        let score = sampler.compute_convergence_score(&audio).unwrap();

        // Healthy signal should have high convergence score
        assert!(
            score > 0.7,
            "Healthy signal should have high convergence score, got: {}",
            score
        );
        assert!(score <= 1.0, "Score should not exceed 1.0");
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_convergence_score_low_energy_signal() {
        use candle_core::{Device, Tensor};

        let device = Device::Cpu;
        let config = SamplingConfig::default();
        let scheduler_config = NoiseSchedulerConfig::default();
        let scheduler = NoiseScheduler::new(scheduler_config, &device).unwrap();
        let sampler = DiffusionSampler::new(config, scheduler, device.clone()).unwrap();

        // Create a very quiet signal (poor convergence indicator)
        let audio_data: Vec<f32> = (0..44100)
            .map(|i| {
                let t = i as f32 / 44100.0;
                0.001 * (2.0 * PI * 440.0 * t).sin() // Very quiet
            })
            .collect();

        let audio = Tensor::from_vec(audio_data, &[1, 44100], &device).unwrap();
        let score = sampler.compute_convergence_score(&audio).unwrap();

        // Low energy signal should have lower convergence score
        assert!(
            score < 0.8,
            "Low energy signal should have lower convergence score, got: {}",
            score
        );
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_convergence_score_clipped_signal() {
        use candle_core::{Device, Tensor};

        let device = Device::Cpu;
        let config = SamplingConfig::default();
        let scheduler_config = NoiseSchedulerConfig::default();
        let scheduler = NoiseScheduler::new(scheduler_config, &device).unwrap();
        let sampler = DiffusionSampler::new(config, scheduler, device.clone()).unwrap();

        // Create a heavily clipped signal (poor convergence)
        let audio_data: Vec<f32> = (0..44100)
            .map(|i| {
                let t = i as f32 / 44100.0;
                let raw = 2.0 * (2.0 * PI * 440.0 * t).sin();
                raw.clamp(-1.0, 1.0) // Heavily clipped
            })
            .collect();

        let audio = Tensor::from_vec(audio_data, &[1, 44100], &device).unwrap();
        let score = sampler.compute_convergence_score(&audio).unwrap();

        // Clipped signal should have moderate convergence score
        assert!(
            score >= 0.0 && score <= 1.0,
            "Score should be in valid range"
        );
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_convergence_score_noise_signal() {
        use candle_core::{Device, Tensor};

        let device = Device::Cpu;
        let config = SamplingConfig::default();
        let scheduler_config = NoiseSchedulerConfig::default();
        let scheduler = NoiseScheduler::new(scheduler_config, &device).unwrap();
        let sampler = DiffusionSampler::new(config, scheduler, device.clone()).unwrap();

        // Create pure noise (unconverged signal)
        let audio_data: Vec<f32> = (0..44100)
            .map(|_| {
                // Random noise using fastrand
                (fastrand::f32() - 0.5) * 0.5
            })
            .collect();

        let audio = Tensor::from_vec(audio_data, &[1, 44100], &device).unwrap();
        let score = sampler.compute_convergence_score(&audio).unwrap();

        // Pure noise should have variable but valid convergence score
        assert!(
            score >= 0.0 && score <= 1.0,
            "Noise signal should have valid convergence score, got: {}",
            score
        );
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_convergence_score_empty_signal() {
        use candle_core::{Device, Tensor};

        let device = Device::Cpu;
        let config = SamplingConfig::default();
        let scheduler_config = NoiseSchedulerConfig::default();
        let scheduler = NoiseScheduler::new(scheduler_config, &device).unwrap();
        let sampler = DiffusionSampler::new(config, scheduler, device.clone()).unwrap();

        // Empty signal
        let audio_data: Vec<f32> = vec![];
        let audio = Tensor::from_vec(audio_data, &[1, 0], &device).unwrap();
        let score = sampler.compute_convergence_score(&audio).unwrap();

        // Empty signal should return 0.0
        assert_eq!(
            score, 0.0,
            "Empty signal should have zero convergence score"
        );
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_convergence_score_silence() {
        use candle_core::{Device, Tensor};

        let device = Device::Cpu;
        let config = SamplingConfig::default();
        let scheduler_config = NoiseSchedulerConfig::default();
        let scheduler = NoiseScheduler::new(scheduler_config, &device).unwrap();
        let sampler = DiffusionSampler::new(config, scheduler, device.clone()).unwrap();

        // Complete silence
        let audio_data: Vec<f32> = vec![0.0; 44100];
        let audio = Tensor::from_vec(audio_data, &[1, 44100], &device).unwrap();
        let score = sampler.compute_convergence_score(&audio).unwrap();

        // Silence should have low convergence score
        assert!(
            score < 0.5,
            "Silence should have low convergence score, got: {}",
            score
        );
    }

    #[test]
    fn test_dpm_solver_plusplus_config() {
        let config = SamplingConfig {
            algorithm: SamplingAlgorithm::DPMSolverPlusPlus,
            num_steps: 10, // DPM-Solver++ works well with very few steps
            ..Default::default()
        };
        assert!(matches!(
            config.algorithm,
            SamplingAlgorithm::DPMSolverPlusPlus
        ));
        assert_eq!(config.num_steps, 10);
    }

    #[test]
    fn test_unipc_config() {
        let config = SamplingConfig {
            algorithm: SamplingAlgorithm::UniPC,
            num_steps: 8, // UniPC achieves excellent quality in 5-10 steps
            ..Default::default()
        };
        assert!(matches!(config.algorithm, SamplingAlgorithm::UniPC));
        assert_eq!(config.num_steps, 8);
    }

    #[test]
    fn test_fast_sampling_configs() {
        // Test that fast samplers use appropriate step counts
        let dpm_config = SamplingConfig {
            algorithm: SamplingAlgorithm::DPMSolverPlusPlus,
            num_steps: 5,
            ..Default::default()
        };
        assert!(
            dpm_config.num_steps < 10,
            "DPM-Solver++ should use few steps"
        );

        let unipc_config = SamplingConfig {
            algorithm: SamplingAlgorithm::UniPC,
            num_steps: 7,
            ..Default::default()
        };
        assert!(
            unipc_config.num_steps < 15,
            "UniPC should use few steps for fast inference"
        );
    }

    #[test]
    fn test_all_sampling_algorithms_serialization() {
        // Ensure all algorithms can be serialized/deserialized
        use serde_json;

        let algorithms = vec![
            SamplingAlgorithm::DDPM,
            SamplingAlgorithm::DDIM,
            SamplingAlgorithm::FastDDIM,
            SamplingAlgorithm::Adaptive,
            SamplingAlgorithm::DPMSolverPlusPlus,
            SamplingAlgorithm::UniPC,
        ];

        for algorithm in algorithms {
            let config = SamplingConfig {
                algorithm,
                ..Default::default()
            };

            // Serialize
            let json = serde_json::to_string(&config).unwrap();

            // Deserialize
            let deserialized: SamplingConfig = serde_json::from_str(&json).unwrap();

            // Verify
            assert_eq!(config.algorithm as u8, deserialized.algorithm as u8);
            assert_eq!(config.num_steps, deserialized.num_steps);
        }
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_convergence_score_range() {
        use candle_core::{Device, Tensor};

        let device = Device::Cpu;
        let config = SamplingConfig::default();
        let scheduler_config = NoiseSchedulerConfig::default();
        let scheduler = NoiseScheduler::new(scheduler_config, &device).unwrap();
        let sampler = DiffusionSampler::new(config, scheduler, device.clone()).unwrap();

        // Test various signal types to ensure score is always in [0, 1]
        let test_signals = vec![
            // Quiet signal
            (0..1024)
                .map(|i| 0.01 * (i as f32 / 100.0).sin())
                .collect::<Vec<f32>>(),
            // Normal signal
            (0..1024)
                .map(|i| 0.5 * (i as f32 / 100.0).sin())
                .collect::<Vec<f32>>(),
            // Loud signal
            (0..1024)
                .map(|i| 0.9 * (i as f32 / 100.0).sin())
                .collect::<Vec<f32>>(),
        ];

        for signal_data in test_signals {
            let audio = Tensor::from_vec(signal_data, &[1, 1024], &device).unwrap();
            let score = sampler.compute_convergence_score(&audio).unwrap();

            assert!(
                score >= 0.0 && score <= 1.0,
                "Convergence score must be in [0, 1] range, got: {}",
                score
            );
        }
    }
}
