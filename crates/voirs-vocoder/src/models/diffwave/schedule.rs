//! Advanced noise scheduling algorithms for DiffWave.
//!
//! This module implements various noise schedules including linear, cosine,
//! sigmoid, and custom schedules for optimal diffusion training and inference.

use candle_core::{Result as CandleResult, Tensor, Device};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

use crate::{Result, VocoderError};

/// Types of noise schedules for diffusion
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NoiseSchedule {
    /// Linear beta schedule - simple and widely used
    Linear,
    /// Cosine beta schedule - recommended for better quality
    Cosine,
    /// Sigmoid beta schedule - smooth transitions
    Sigmoid,
    /// Quadratic beta schedule - faster noise increase
    Quadratic,
    /// Custom beta schedule from provided values
    Custom,
}

/// Advanced noise scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseSchedulerConfig {
    /// Type of noise schedule to use
    pub schedule_type: NoiseSchedule,
    /// Total number of diffusion timesteps
    pub num_steps: u32,
    /// Beta start value (for linear/quadratic schedules)
    pub beta_start: f32,
    /// Beta end value (for linear/quadratic schedules)
    pub beta_end: f32,
    /// Cosine schedule offset parameter
    pub cosine_s: f32,
    /// Maximum beta value (clipping threshold)
    pub max_beta: f32,
    /// Minimum beta value (clipping threshold)
    pub min_beta: f32,
    /// Sigmoid schedule center point (0-1)
    pub sigmoid_center: f32,
    /// Sigmoid schedule steepness
    pub sigmoid_steepness: f32,
    /// Custom beta values (only used for Custom schedule)
    pub custom_betas: Vec<f32>,
}

impl Default for NoiseSchedulerConfig {
    fn default() -> Self {
        Self {
            schedule_type: NoiseSchedule::Cosine,
            num_steps: 1000,
            beta_start: 0.0001,
            beta_end: 0.02,
            cosine_s: 0.008,
            max_beta: 0.999,
            min_beta: 0.0001,
            sigmoid_center: 0.5,
            sigmoid_steepness: 10.0,
            custom_betas: Vec::new(),
        }
    }
}

/// Noise scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub schedule_type: NoiseSchedule,
    pub num_steps: usize,
    pub min_beta: f32,
    pub max_beta: f32,
    pub avg_beta: f32,
    pub min_alpha_cumprod: f32,
    pub max_alpha_cumprod: f32,
}

/// Advanced noise scheduler with multiple algorithms
#[derive(Debug)]
pub struct NoiseScheduler {
    config: NoiseSchedulerConfig,
    betas: Vec<f32>,
    alphas: Vec<f32>,
    alphas_cumprod: Vec<f32>,
    alphas_cumprod_prev: Vec<f32>,
    sqrt_alphas_cumprod: Vec<f32>,
    sqrt_one_minus_alphas_cumprod: Vec<f32>,
    log_one_minus_alphas_cumprod: Vec<f32>,
    sqrt_recip_alphas_cumprod: Vec<f32>,
    sqrt_recipm1_alphas_cumprod: Vec<f32>,
    posterior_variance: Vec<f32>,
    posterior_log_variance_clipped: Vec<f32>,
    posterior_mean_coef1: Vec<f32>,
    posterior_mean_coef2: Vec<f32>,
}

impl NoiseScheduler {
    /// Create a new noise scheduler with the given configuration
    pub fn new(config: NoiseSchedulerConfig, _device: &Device) -> Result<Self> {
        let betas = Self::compute_betas(&config)?;
        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();
        
        // Compute cumulative products
        let mut alphas_cumprod = Vec::new();
        let mut cumprod = 1.0;
        for alpha in &alphas {
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }
        
        // Compute previous cumulative products
        let mut alphas_cumprod_prev = vec![1.0]; // α_0 = 1
        alphas_cumprod_prev.extend(&alphas_cumprod[..alphas_cumprod.len() - 1]);
        
        // Precompute commonly used values for efficiency
        let sqrt_alphas_cumprod: Vec<f32> = alphas_cumprod.iter().map(|a| a.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod: Vec<f32> = alphas_cumprod.iter().map(|a| (1.0 - a).sqrt()).collect();
        let log_one_minus_alphas_cumprod: Vec<f32> = alphas_cumprod.iter().map(|a| (1.0 - a).ln()).collect();
        let sqrt_recip_alphas_cumprod: Vec<f32> = alphas_cumprod.iter().map(|a| 1.0 / a.sqrt()).collect();
        let sqrt_recipm1_alphas_cumprod: Vec<f32> = alphas_cumprod.iter().map(|a| (1.0 / a - 1.0).sqrt()).collect();
        
        // Compute posterior variance β̃_t = (1 - α_{t-1}) / (1 - α_t) * β_t
        let mut posterior_variance = Vec::new();
        let mut posterior_log_variance_clipped = Vec::new();
        for i in 0..betas.len() {
            let variance = if i == 0 {
                0.0
            } else {
                betas[i] * (1.0 - alphas_cumprod_prev[i]) / (1.0 - alphas_cumprod[i])
            };
            posterior_variance.push(variance);
            posterior_log_variance_clipped.push(variance.max(1e-20).ln());
        }
        
        // Compute posterior mean coefficients
        let posterior_mean_coef1: Vec<f32> = betas.iter().zip(&alphas_cumprod_prev)
            .map(|(beta, alpha_prev)| beta.sqrt() * alpha_prev.sqrt() / (1.0 - alphas_cumprod[0]))
            .collect();
        let posterior_mean_coef2: Vec<f32> = alphas.iter().zip(&alphas_cumprod)
            .map(|(alpha, alpha_cumprod)| (1.0 - alpha).sqrt() * alpha.sqrt() / (1.0 - alpha_cumprod))
            .collect();
        
        Ok(Self {
            config,
            betas,
            alphas,
            alphas_cumprod,
            alphas_cumprod_prev,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            log_one_minus_alphas_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            posterior_variance,
            posterior_log_variance_clipped,
            posterior_mean_coef1,
            posterior_mean_coef2,
        })
    }
    
    /// Compute beta values according to the configured schedule
    fn compute_betas(config: &NoiseSchedulerConfig) -> Result<Vec<f32>> {
        let steps = config.num_steps as usize;
        
        let betas = match config.schedule_type {
            NoiseSchedule::Linear => {
                Self::linear_beta_schedule(config.beta_start, config.beta_end, steps)
            }
            NoiseSchedule::Cosine => {
                Self::cosine_beta_schedule(config.cosine_s, steps)
            }
            NoiseSchedule::Sigmoid => {
                Self::sigmoid_beta_schedule(
                    config.beta_start,
                    config.beta_end,
                    config.sigmoid_center,
                    config.sigmoid_steepness,
                    steps,
                )
            }
            NoiseSchedule::Quadratic => {
                Self::quadratic_beta_schedule(config.beta_start, config.beta_end, steps)
            }
            NoiseSchedule::Custom => {
                if config.custom_betas.len() != steps {
                    return Err(VocoderError::ConfigError(
                        format!("Custom betas length {} doesn't match num_steps {}", 
                               config.custom_betas.len(), steps)
                    ));
                }
                config.custom_betas.clone()
            }
        };
        
        // Apply clipping
        let clipped_betas: Vec<f32> = betas.iter()
            .map(|&b| b.clamp(config.min_beta, config.max_beta))
            .collect();
        
        Ok(clipped_betas)
    }
    
    /// Linear beta schedule: β_t = β_start + (β_end - β_start) * t / T
    fn linear_beta_schedule(beta_start: f32, beta_end: f32, steps: usize) -> Vec<f32> {
        let step_size = (beta_end - beta_start) / (steps - 1) as f32;
        (0..steps)
            .map(|i| beta_start + step_size * i as f32)
            .collect()
    }
    
    /// Cosine beta schedule for better perceptual quality
    fn cosine_beta_schedule(s: f32, steps: usize) -> Vec<f32> {
        let mut alphas_cumprod = Vec::new();
        for i in 0..steps {
            let t = i as f32 / steps as f32;
            let alpha_cumprod = Self::cosine_alpha_cumprod(t, s);
            alphas_cumprod.push(alpha_cumprod);
        }
        
        let mut betas = Vec::new();
        betas.push(1.0 - alphas_cumprod[0]);
        
        for i in 1..steps {
            let beta = 1.0 - alphas_cumprod[i] / alphas_cumprod[i - 1];
            betas.push(beta.min(0.999));
        }
        
        betas
    }
    
    /// Sigmoid beta schedule for smooth transitions
    fn sigmoid_beta_schedule(
        beta_start: f32,
        beta_end: f32,
        center: f32,
        steepness: f32,
        steps: usize,
    ) -> Vec<f32> {
        (0..steps)
            .map(|i| {
                let t = i as f32 / (steps - 1) as f32;
                let sigmoid = 1.0 / (1.0 + (-steepness * (t - center)).exp());
                beta_start + (beta_end - beta_start) * sigmoid
            })
            .collect()
    }
    
    /// Quadratic beta schedule: β_t = β_start + (β_end - β_start) * (t / T)²
    fn quadratic_beta_schedule(beta_start: f32, beta_end: f32, steps: usize) -> Vec<f32> {
        (0..steps)
            .map(|i| {
                let t = i as f32 / (steps - 1) as f32;
                beta_start + (beta_end - beta_start) * t * t
            })
            .collect()
    }
    
    /// Helper function for cosine schedule
    fn cosine_alpha_cumprod(t: f32, s: f32) -> f32 {
        let cos_val = ((t + s) / (1.0 + s) * PI / 2.0).cos();
        cos_val * cos_val
    }
    
    /// Add noise to clean audio: q(x_t | x_0) = N(x_t; √α̅_t x_0, (1 - α̅_t)I)
    pub fn add_noise(&self, x0: &Tensor, noise: &Tensor, timestep: usize) -> CandleResult<Tensor> {
        if timestep >= self.sqrt_alphas_cumprod.len() {
            return Err(candle_core::Error::Msg("Timestep out of range".to_string()));
        }
        
        let sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timestep];
        let sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timestep];
        
        let x0_scaled = x0.affine(sqrt_alpha_cumprod as f64, 0.0)?;
        let noise_scaled = noise.affine(sqrt_one_minus_alpha_cumprod as f64, 0.0)?;
        let result = x0_scaled.add(&noise_scaled)?;
        Ok(result)
    }
    
    /// Get the mean and variance of q(x_{t-1} | x_t, x_0)
    pub fn get_posterior_mean_variance(
        &self,
        x_start: &Tensor,
        x_t: &Tensor,
        timestep: usize,
    ) -> CandleResult<(Tensor, f32, f32)> {
        if timestep >= self.posterior_mean_coef1.len() {
            return Err(candle_core::Error::Msg("Timestep out of range".to_string()));
        }
        
        let coef1 = self.posterior_mean_coef1[timestep];
        let coef2 = self.posterior_mean_coef2[timestep];
        
        let mean = x_start.affine(coef1 as f64, 0.0)?
            .add(&x_t.affine(coef2 as f64, 0.0)?)?;
        
        let variance = self.posterior_variance[timestep];
        let log_variance = self.posterior_log_variance_clipped[timestep];
        
        Ok((mean, variance, log_variance))
    }
    
    /// Predict x_0 from x_t and predicted noise
    pub fn predict_start_from_noise(
        &self,
        x_t: &Tensor,
        t: usize,
        noise: &Tensor,
    ) -> CandleResult<Tensor> {
        if t >= self.sqrt_recip_alphas_cumprod.len() {
            return Err(candle_core::Error::Msg("Timestep out of range".to_string()));
        }
        
        let sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod[t];
        let sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod[t];
        
        let x_start = x_t.affine(sqrt_recip_alphas_cumprod as f64, 0.0)?
            .sub(&noise.affine(sqrt_recipm1_alphas_cumprod as f64, 0.0)?)?;
        
        Ok(x_start)
    }
    
    /// Get noise level at timestep t
    pub fn get_noise_level(&self, timestep: usize) -> Result<f32> {
        if timestep >= self.alphas_cumprod.len() {
            return Err(VocoderError::ConfigError("Timestep out of range".to_string()));
        }
        Ok(1.0 - self.alphas_cumprod[timestep])
    }
    
    /// Get signal level at timestep t
    pub fn get_signal_level(&self, timestep: usize) -> Result<f32> {
        if timestep >= self.alphas_cumprod.len() {
            return Err(VocoderError::ConfigError("Timestep out of range".to_string()));
        }
        Ok(self.alphas_cumprod[timestep])
    }
    
    /// Get scheduler statistics
    pub fn stats(&self) -> SchedulerStats {
        let min_beta = self.betas.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_beta = self.betas.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let avg_beta = self.betas.iter().sum::<f32>() / self.betas.len() as f32;
        
        let min_alpha_cumprod = self.alphas_cumprod.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_alpha_cumprod = self.alphas_cumprod.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        SchedulerStats {
            schedule_type: self.config.schedule_type,
            num_steps: self.betas.len(),
            min_beta,
            max_beta,
            avg_beta,
            min_alpha_cumprod,
            max_alpha_cumprod,
        }
    }
    
    /// Get configuration
    pub fn config(&self) -> &NoiseSchedulerConfig {
        &self.config
    }
    
    /// Public accessors for scheduler values
    pub fn betas(&self) -> &[f32] {
        &self.betas
    }
    
    pub fn alphas(&self) -> &[f32] {
        &self.alphas
    }
    
    pub fn alphas_cumprod(&self) -> &[f32] {
        &self.alphas_cumprod
    }
    
    pub fn sqrt_alphas_cumprod(&self) -> &[f32] {
        &self.sqrt_alphas_cumprod
    }
    
    pub fn sqrt_one_minus_alphas_cumprod(&self) -> &[f32] {
        &self.sqrt_one_minus_alphas_cumprod
    }
}

impl Clone for NoiseScheduler {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            betas: self.betas.clone(),
            alphas: self.alphas.clone(),
            alphas_cumprod: self.alphas_cumprod.clone(),
            alphas_cumprod_prev: self.alphas_cumprod_prev.clone(),
            sqrt_alphas_cumprod: self.sqrt_alphas_cumprod.clone(),
            sqrt_one_minus_alphas_cumprod: self.sqrt_one_minus_alphas_cumprod.clone(),
            log_one_minus_alphas_cumprod: self.log_one_minus_alphas_cumprod.clone(),
            sqrt_recip_alphas_cumprod: self.sqrt_recip_alphas_cumprod.clone(),
            sqrt_recipm1_alphas_cumprod: self.sqrt_recipm1_alphas_cumprod.clone(),
            posterior_variance: self.posterior_variance.clone(),
            posterior_log_variance_clipped: self.posterior_log_variance_clipped.clone(),
            posterior_mean_coef1: self.posterior_mean_coef1.clone(),
            posterior_mean_coef2: self.posterior_mean_coef2.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_noise_scheduler_config() {
        let config = NoiseSchedulerConfig::default();
        assert_eq!(config.num_steps, 1000);
        assert!(matches!(config.schedule_type, NoiseSchedule::Cosine));
        assert_eq!(config.beta_start, 0.0001);
        assert_eq!(config.beta_end, 0.02);
    }
    
    #[test]
    fn test_linear_beta_schedule() {
        let betas = NoiseScheduler::linear_beta_schedule(0.0001, 0.02, 1000);
        assert_eq!(betas.len(), 1000);
        assert!((betas[0] - 0.0001).abs() < 1e-6);
        assert!((betas[999] - 0.02).abs() < 1e-6);
        
        // Should be monotonically increasing
        for i in 1..betas.len() {
            assert!(betas[i] >= betas[i-1]);
        }
    }
    
    #[test]
    fn test_cosine_beta_schedule() {
        let betas = NoiseScheduler::cosine_beta_schedule(0.008, 1000);
        assert_eq!(betas.len(), 1000);
        
        // All betas should be positive and less than max_beta
        for beta in &betas {
            assert!(*beta > 0.0);
            assert!(*beta <= 0.999);
        }
    }
    
    #[test]
    fn test_noise_scheduler_creation() {
        let device = Device::Cpu;
        let config = NoiseSchedulerConfig::default();
        let scheduler = NoiseScheduler::new(config, &device);
        assert!(scheduler.is_ok());
        
        let scheduler = scheduler.unwrap();
        assert_eq!(scheduler.betas().len(), 1000);
        assert_eq!(scheduler.alphas().len(), 1000);
        assert_eq!(scheduler.alphas_cumprod().len(), 1000);
    }
    
    #[test]
    fn test_noise_scheduler_stats() {
        let device = Device::Cpu;
        let config = NoiseSchedulerConfig::default();
        let scheduler = NoiseScheduler::new(config, &device).unwrap();
        let stats = scheduler.stats();
        
        assert_eq!(stats.num_steps, 1000);
        assert!(stats.min_beta > 0.0);
        assert!(stats.max_beta < 1.0);
        assert!(stats.avg_beta > 0.0);
    }
    
    #[test]
    fn test_different_schedules() {
        let device = Device::Cpu;
        let schedules = [
            NoiseSchedule::Linear,
            NoiseSchedule::Cosine,
            NoiseSchedule::Sigmoid,
            NoiseSchedule::Quadratic,
        ];
        
        for schedule in &schedules {
            let config = NoiseSchedulerConfig {
                schedule_type: *schedule,
                num_steps: 100, // Smaller for testing
                ..Default::default()
            };
            
            let scheduler = NoiseScheduler::new(config, &device);
            assert!(scheduler.is_ok(), "Failed to create scheduler for {:?}", schedule);
            
            let scheduler = scheduler.unwrap();
            assert_eq!(scheduler.betas().len(), 100);
        }
    }
}