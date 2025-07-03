//! Advanced sampling algorithms for DiffWave diffusion.
//!
//! This module implements DDPM, DDIM, and fast sampling techniques
//! for high-quality audio generation from mel spectrograms.

use candle_core::{Result as CandleResult, Tensor, Device};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

use crate::{Result, VocoderError};
use super::{EnhancedUNet, NoiseScheduler};

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
#[derive(Debug)]
pub struct DiffusionSampler {
    config: SamplingConfig,
    scheduler: NoiseScheduler,
    device: Device,
}

impl DiffusionSampler {
    /// Create a new diffusion sampler
    pub fn new(
        config: SamplingConfig,
        scheduler: NoiseScheduler,
        device: Device,
    ) -> Result<Self> {
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
        };
        
        let total_time = start_time.elapsed().as_millis() as f32;
        
        match result {
            Ok(audio) => {
                let stats = SamplingStats {
                    algorithm_used: self.config.algorithm,
                    total_steps: self.config.num_steps,
                    actual_steps: self.config.num_steps,
                    avg_step_time_ms: total_time / self.config.num_steps as f32,
                    total_time_ms: total_time,
                    convergence_score: 0.95, // Placeholder
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
        let x0_pred = (x.affine(1.0 / alpha_t_sqrt as f64, 0.0)?
            - predicted_noise.affine(one_minus_alpha_t_sqrt as f64 / alpha_t_sqrt as f64, 0.0)?)?;
        
        // Compute direction pointing to x_t
        let direction = predicted_noise.affine(beta_t.sqrt() as f64, 0.0)?;
        
        // Compute x_{t-1}
        let x_prev = x.sub(&direction)?;
        
        // Add noise for stochasticity (DDPM)
        if timestep > 0 {
            let noise = self.sample_noise(&x.dims())?;
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
        let x_prev = x0_pred.affine(alpha_t_prev_sqrt as f64, 0.0)?
            .add(&predicted_noise.affine(one_minus_alpha_t_prev.sqrt() as f64 * self.config.eta as f64, 0.0)?)?;
        
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
    use candle_core::Device;
    
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
}