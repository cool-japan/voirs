//! GAN loss functions for neural vocoder training.
//!
//! This module provides adversarial loss functions used in GAN-based vocoders
//! like HiFi-GAN, BigVGAN, and UnivNet.
//!
//! # Loss Components
//!
//! 1. **Adversarial Loss**: Standard GAN loss for generator and discriminator
//! 2. **Feature Matching Loss**: Matches intermediate discriminator features
//! 3. **Multi-Scale Discriminator Loss**: Loss from discriminators at multiple scales
//! 4. **Multi-Period Discriminator Loss**: Loss from discriminators with different periods
//!
//! # Usage
//!
//! ```rust,ignore
//! use voirs_vocoder::loss::gan::{AdversarialLoss, FeatureMatchingLoss};
//! use scirs2_core::ndarray::Array2;
//!
//! // Adversarial loss for generator
//! let adv_loss = AdversarialLoss::new();
//! let disc_outputs = vec![/* discriminator outputs */];
//! let gen_loss = adv_loss.generator_loss(&disc_outputs)?;
//!
//! // Feature matching loss
//! let fm_loss = FeatureMatchingLoss::new();
//! let real_features = vec![/* features from real audio */];
//! let fake_features = vec![/* features from generated audio */];
//! let loss = fm_loss.compute(&real_features, &fake_features)?;
//! ```

use crate::{Result, VocoderError};
use scirs2_core::ndarray::prelude::*;
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};

/// Loss function type for adversarial training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdversarialLossType {
    /// Hinge loss (used in BigVGAN)
    Hinge,
    /// Least squares loss (used in HiFi-GAN)
    LeastSquares,
    /// Binary cross entropy
    BCE,
}

impl Default for AdversarialLossType {
    fn default() -> Self {
        Self::LeastSquares
    }
}

/// Configuration for adversarial loss
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialLossConfig {
    /// Loss function type
    pub loss_type: AdversarialLossType,
    /// Weight for adversarial loss
    pub weight: f32,
}

impl Default for AdversarialLossConfig {
    fn default() -> Self {
        Self {
            loss_type: AdversarialLossType::LeastSquares,
            weight: 1.0,
        }
    }
}

/// Adversarial loss calculator for GAN training
pub struct AdversarialLoss {
    config: AdversarialLossConfig,
}

impl AdversarialLoss {
    /// Create a new adversarial loss calculator
    pub fn new(config: AdversarialLossConfig) -> Self {
        Self { config }
    }

    /// Compute generator adversarial loss
    ///
    /// Encourages discriminator to output 1 (real) for generated samples
    ///
    /// # Arguments
    /// * `disc_fake_outputs` - Discriminator outputs for generated (fake) audio
    ///
    /// # Returns
    /// * Generator adversarial loss value
    pub fn generator_loss(&self, disc_fake_outputs: &[Array1<f32>]) -> Result<f32> {
        if disc_fake_outputs.is_empty() {
            return Err(VocoderError::InputError(
                "Discriminator outputs cannot be empty".to_string(),
            ));
        }

        let mut total_loss = 0.0;
        let mut count = 0;

        for disc_output in disc_fake_outputs {
            let loss = match self.config.loss_type {
                AdversarialLossType::LeastSquares => {
                    // L_G = mean((D(G(z)) - 1)^2)
                    disc_output.iter().map(|&x| (x - 1.0).powi(2)).sum::<f32>()
                        / disc_output.len() as f32
                }
                AdversarialLossType::Hinge => {
                    // L_G = -mean(D(G(z)))
                    -disc_output.mean().unwrap_or(0.0)
                }
                AdversarialLossType::BCE => {
                    // L_G = -mean(log(D(G(z))))
                    let eps = 1e-7;
                    -disc_output.iter().map(|&x| (x + eps).ln()).sum::<f32>()
                        / disc_output.len() as f32
                }
            };

            total_loss += loss;
            count += 1;
        }

        Ok(self.config.weight * total_loss / count as f32)
    }

    /// Compute discriminator adversarial loss
    ///
    /// Encourages discriminator to output 1 for real, 0 for fake
    ///
    /// # Arguments
    /// * `disc_real_outputs` - Discriminator outputs for real audio
    /// * `disc_fake_outputs` - Discriminator outputs for generated audio
    ///
    /// # Returns
    /// * Discriminator adversarial loss value
    pub fn discriminator_loss(
        &self,
        disc_real_outputs: &[Array1<f32>],
        disc_fake_outputs: &[Array1<f32>],
    ) -> Result<f32> {
        if disc_real_outputs.len() != disc_fake_outputs.len() {
            return Err(VocoderError::InputError(
                "Real and fake outputs must have the same length".to_string(),
            ));
        }

        if disc_real_outputs.is_empty() {
            return Err(VocoderError::InputError(
                "Discriminator outputs cannot be empty".to_string(),
            ));
        }

        let mut total_loss = 0.0;
        let mut count = 0;

        for (disc_real, disc_fake) in disc_real_outputs.iter().zip(disc_fake_outputs.iter()) {
            let real_loss = match self.config.loss_type {
                AdversarialLossType::LeastSquares => {
                    // L_D_real = mean((D(x) - 1)^2)
                    disc_real.iter().map(|&x| (x - 1.0).powi(2)).sum::<f32>()
                        / disc_real.len() as f32
                }
                AdversarialLossType::Hinge => {
                    // L_D_real = mean(max(0, 1 - D(x)))
                    disc_real.iter().map(|&x| (1.0 - x).max(0.0)).sum::<f32>()
                        / disc_real.len() as f32
                }
                AdversarialLossType::BCE => {
                    // L_D_real = -mean(log(D(x)))
                    let eps = 1e-7;
                    -disc_real.iter().map(|&x| (x + eps).ln()).sum::<f32>() / disc_real.len() as f32
                }
            };

            let fake_loss = match self.config.loss_type {
                AdversarialLossType::LeastSquares => {
                    // L_D_fake = mean(D(G(z))^2)
                    disc_fake.iter().map(|&x| x.powi(2)).sum::<f32>() / disc_fake.len() as f32
                }
                AdversarialLossType::Hinge => {
                    // L_D_fake = mean(max(0, 1 + D(G(z))))
                    disc_fake.iter().map(|&x| (1.0 + x).max(0.0)).sum::<f32>()
                        / disc_fake.len() as f32
                }
                AdversarialLossType::BCE => {
                    // L_D_fake = -mean(log(1 - D(G(z))))
                    let eps = 1e-7;
                    -disc_fake.iter().map(|&x| (1.0 - x + eps).ln()).sum::<f32>()
                        / disc_fake.len() as f32
                }
            };

            total_loss += real_loss + fake_loss;
            count += 1;
        }

        Ok(self.config.weight * total_loss / count as f32)
    }
}

impl Default for AdversarialLoss {
    fn default() -> Self {
        Self::new(AdversarialLossConfig::default())
    }
}

/// Configuration for feature matching loss
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMatchingLossConfig {
    /// Weight for feature matching loss
    pub weight: f32,
    /// Number of discriminator layers to use for feature matching
    pub num_layers: usize,
}

impl Default for FeatureMatchingLossConfig {
    fn default() -> Self {
        Self {
            weight: 10.0, // Typically weighted higher than adversarial loss
            num_layers: 4,
        }
    }
}

/// Feature matching loss calculator
///
/// Matches intermediate features from discriminator to encourage generator
/// to produce similar feature representations to real audio
pub struct FeatureMatchingLoss {
    config: FeatureMatchingLossConfig,
}

impl FeatureMatchingLoss {
    /// Create a new feature matching loss calculator
    pub fn new(config: FeatureMatchingLossConfig) -> Self {
        Self { config }
    }

    /// Compute feature matching loss
    ///
    /// # Arguments
    /// * `real_features` - Intermediate features from discriminator on real audio
    /// * `fake_features` - Intermediate features from discriminator on generated audio
    ///
    /// # Returns
    /// * Feature matching loss value
    pub fn compute<F: Float>(
        &self,
        real_features: &[Array2<F>],
        fake_features: &[Array2<F>],
    ) -> Result<f32> {
        if real_features.len() != fake_features.len() {
            return Err(VocoderError::InputError(format!(
                "Real and fake features must have the same number of layers: {} vs {}",
                real_features.len(),
                fake_features.len()
            )));
        }

        if real_features.is_empty() {
            return Err(VocoderError::InputError(
                "Feature lists cannot be empty".to_string(),
            ));
        }

        let mut total_loss = 0.0;
        let num_layers = self.config.num_layers.min(real_features.len());

        for i in 0..num_layers {
            let real_feat = &real_features[i];
            let fake_feat = &fake_features[i];

            if real_feat.shape() != fake_feat.shape() {
                return Err(VocoderError::InputError(format!(
                    "Feature shapes must match at layer {}: {:?} vs {:?}",
                    i,
                    real_feat.shape(),
                    fake_feat.shape()
                )));
            }

            // Compute L1 distance between features
            let mut layer_loss = 0.0;
            for (r, f) in real_feat.iter().zip(fake_feat.iter()) {
                let r_f32 = r.to_f32().unwrap_or(0.0);
                let f_f32 = f.to_f32().unwrap_or(0.0);
                layer_loss += (r_f32 - f_f32).abs();
            }

            layer_loss /= real_feat.len() as f32;
            total_loss += layer_loss;
        }

        Ok(self.config.weight * total_loss / num_layers as f32)
    }

    /// Compute feature matching loss with per-layer weights
    ///
    /// # Arguments
    /// * `real_features` - Intermediate features from discriminator on real audio
    /// * `fake_features` - Intermediate features from discriminator on generated audio
    /// * `layer_weights` - Weight for each layer
    ///
    /// # Returns
    /// * Weighted feature matching loss value
    pub fn compute_weighted<F: Float>(
        &self,
        real_features: &[Array2<F>],
        fake_features: &[Array2<F>],
        layer_weights: &[f32],
    ) -> Result<f32> {
        if real_features.len() != fake_features.len() {
            return Err(VocoderError::InputError(
                "Real and fake features must have the same number of layers".to_string(),
            ));
        }

        if real_features.len() != layer_weights.len() {
            return Err(VocoderError::InputError(
                "Number of layer weights must match number of feature layers".to_string(),
            ));
        }

        let mut total_loss = 0.0;
        let mut total_weight = 0.0;

        for i in 0..real_features.len() {
            let real_feat = &real_features[i];
            let fake_feat = &fake_features[i];

            if real_feat.shape() != fake_feat.shape() {
                return Err(VocoderError::InputError(format!(
                    "Feature shapes must match at layer {i}: {:?} vs {:?}",
                    real_feat.shape(),
                    fake_feat.shape()
                )));
            }

            // Compute L1 distance between features
            let mut layer_loss = 0.0;
            for (r, f) in real_feat.iter().zip(fake_feat.iter()) {
                let r_f32 = r.to_f32().unwrap_or(0.0);
                let f_f32 = f.to_f32().unwrap_or(0.0);
                layer_loss += (r_f32 - f_f32).abs();
            }

            layer_loss /= real_feat.len() as f32;
            total_loss += layer_weights[i] * layer_loss;
            total_weight += layer_weights[i];
        }

        if total_weight > 0.0 {
            total_loss /= total_weight;
        }

        Ok(self.config.weight * total_loss)
    }
}

impl Default for FeatureMatchingLoss {
    fn default() -> Self {
        Self::new(FeatureMatchingLossConfig::default())
    }
}

/// Multi-scale discriminator loss configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleDiscriminatorLossConfig {
    /// Number of discriminator scales
    pub num_scales: usize,
    /// Adversarial loss configuration
    pub adversarial_config: AdversarialLossConfig,
    /// Feature matching loss configuration
    pub feature_matching_config: Option<FeatureMatchingLossConfig>,
}

impl Default for MultiScaleDiscriminatorLossConfig {
    fn default() -> Self {
        Self {
            num_scales: 3,
            adversarial_config: AdversarialLossConfig::default(),
            feature_matching_config: Some(FeatureMatchingLossConfig::default()),
        }
    }
}

/// Multi-scale discriminator loss
///
/// Combines adversarial and feature matching losses across multiple scales
pub struct MultiScaleDiscriminatorLoss {
    adversarial_loss: AdversarialLoss,
    feature_matching_loss: Option<FeatureMatchingLoss>,
}

impl MultiScaleDiscriminatorLoss {
    /// Create a new multi-scale discriminator loss
    pub fn new(config: MultiScaleDiscriminatorLossConfig) -> Self {
        let adversarial_loss = AdversarialLoss::new(config.adversarial_config);
        let feature_matching_loss = config.feature_matching_config.map(FeatureMatchingLoss::new);

        Self {
            adversarial_loss,
            feature_matching_loss,
        }
    }

    /// Compute total generator loss across all scales
    pub fn generator_loss(
        &self,
        disc_fake_outputs: &[Vec<Array1<f32>>],
        real_features: Option<&[Vec<Array2<f32>>]>,
        fake_features: Option<&[Vec<Array2<f32>>]>,
    ) -> Result<GANLossBreakdown> {
        let mut total_adv_loss = 0.0;
        let mut total_fm_loss = 0.0;

        // Adversarial loss for each scale
        for scale_outputs in disc_fake_outputs {
            let scale_loss = self.adversarial_loss.generator_loss(scale_outputs)?;
            total_adv_loss += scale_loss;
        }
        total_adv_loss /= disc_fake_outputs.len() as f32;

        // Feature matching loss if enabled
        if let Some(fm_loss_calc) = &self.feature_matching_loss {
            if let (Some(real_feats), Some(fake_feats)) = (real_features, fake_features) {
                if real_feats.len() != fake_feats.len() {
                    return Err(VocoderError::InputError(
                        "Real and fake features must have same number of scales".to_string(),
                    ));
                }

                for (real_scale, fake_scale) in real_feats.iter().zip(fake_feats.iter()) {
                    let scale_fm_loss = fm_loss_calc.compute(real_scale, fake_scale)?;
                    total_fm_loss += scale_fm_loss;
                }
                total_fm_loss /= real_feats.len() as f32;
            }
        }

        Ok(GANLossBreakdown {
            adversarial_loss: total_adv_loss,
            feature_matching_loss: if total_fm_loss > 0.0 {
                Some(total_fm_loss)
            } else {
                None
            },
            total_loss: total_adv_loss + total_fm_loss,
        })
    }

    /// Compute total discriminator loss across all scales
    pub fn discriminator_loss(
        &self,
        disc_real_outputs: &[Vec<Array1<f32>>],
        disc_fake_outputs: &[Vec<Array1<f32>>],
    ) -> Result<f32> {
        if disc_real_outputs.len() != disc_fake_outputs.len() {
            return Err(VocoderError::InputError(
                "Real and fake outputs must have same number of scales".to_string(),
            ));
        }

        let mut total_loss = 0.0;

        for (real_scale, fake_scale) in disc_real_outputs.iter().zip(disc_fake_outputs.iter()) {
            let scale_loss = self
                .adversarial_loss
                .discriminator_loss(real_scale, fake_scale)?;
            total_loss += scale_loss;
        }

        Ok(total_loss / disc_real_outputs.len() as f32)
    }
}

impl Default for MultiScaleDiscriminatorLoss {
    fn default() -> Self {
        Self::new(MultiScaleDiscriminatorLossConfig::default())
    }
}

/// Breakdown of GAN loss components
#[derive(Debug, Clone)]
pub struct GANLossBreakdown {
    /// Adversarial loss component
    pub adversarial_loss: f32,
    /// Feature matching loss component (if used)
    pub feature_matching_loss: Option<f32>,
    /// Total combined loss
    pub total_loss: f32,
}

impl GANLossBreakdown {
    /// Get a summary string
    pub fn summary(&self) -> String {
        let mut lines = vec![
            format!("Total GAN Loss: {:.6}", self.total_loss),
            format!("  Adversarial: {:.6}", self.adversarial_loss),
        ];

        if let Some(fm_loss) = self.feature_matching_loss {
            lines.push(format!("  Feature Matching: {:.6}", fm_loss));
        }

        lines.join("\n")
    }
}

/// Multi-period discriminator loss configuration
///
/// Used in UnivNet to capture periodic patterns at different periods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPeriodDiscriminatorLossConfig {
    /// Periods to use for discriminators (e.g., [2, 3, 5, 7, 11])
    pub periods: Vec<usize>,
    /// Adversarial loss configuration
    pub adversarial_config: AdversarialLossConfig,
    /// Feature matching loss configuration
    pub feature_matching_config: Option<FeatureMatchingLossConfig>,
}

impl Default for MultiPeriodDiscriminatorLossConfig {
    fn default() -> Self {
        Self {
            periods: vec![2, 3, 5, 7, 11], // Standard UnivNet periods
            adversarial_config: AdversarialLossConfig::default(),
            feature_matching_config: Some(FeatureMatchingLossConfig::default()),
        }
    }
}

/// Multi-period discriminator loss
///
/// Processes audio with discriminators at different periods to capture
/// periodic patterns. Used in UnivNet architecture.
///
/// # Example
///
/// ```rust,ignore
/// use voirs_vocoder::loss::gan::{
///     MultiPeriodDiscriminatorLoss,
///     MultiPeriodDiscriminatorLossConfig,
/// };
///
/// let config = MultiPeriodDiscriminatorLossConfig {
///     periods: vec![2, 3, 5, 7, 11],
///     ..Default::default()
/// };
///
/// let mpd_loss = MultiPeriodDiscriminatorLoss::new(config);
///
/// // Compute generator loss across all periods
/// let breakdown = mpd_loss.generator_loss(
///     &disc_fake_outputs,
///     Some(&real_features),
///     Some(&fake_features),
/// )?;
/// ```
pub struct MultiPeriodDiscriminatorLoss {
    adversarial_loss: AdversarialLoss,
    feature_matching_loss: Option<FeatureMatchingLoss>,
}

impl MultiPeriodDiscriminatorLoss {
    /// Create a new multi-period discriminator loss
    pub fn new(config: MultiPeriodDiscriminatorLossConfig) -> Self {
        let adversarial_loss = AdversarialLoss::new(config.adversarial_config);
        let feature_matching_loss = config.feature_matching_config.map(FeatureMatchingLoss::new);

        Self {
            adversarial_loss,
            feature_matching_loss,
        }
    }

    /// Compute total generator loss across all periods
    ///
    /// # Arguments
    /// * `disc_fake_outputs` - Discriminator outputs for each period
    /// * `real_features` - Optional real audio features for each period
    /// * `fake_features` - Optional generated audio features for each period
    ///
    /// # Returns
    /// * GAN loss breakdown with adversarial and optional feature matching components
    pub fn generator_loss(
        &self,
        disc_fake_outputs: &[Vec<Array1<f32>>],
        real_features: Option<&[Vec<Array2<f32>>]>,
        fake_features: Option<&[Vec<Array2<f32>>]>,
    ) -> Result<GANLossBreakdown> {
        let mut total_adv_loss = 0.0;
        let mut total_fm_loss = 0.0;

        // Adversarial loss for each period
        for period_outputs in disc_fake_outputs {
            let period_loss = self.adversarial_loss.generator_loss(period_outputs)?;
            total_adv_loss += period_loss;
        }
        total_adv_loss /= disc_fake_outputs.len() as f32;

        // Feature matching loss if enabled
        if let Some(fm_loss_calc) = &self.feature_matching_loss {
            if let (Some(real_feats), Some(fake_feats)) = (real_features, fake_features) {
                if real_feats.len() != fake_feats.len() {
                    return Err(VocoderError::InputError(
                        "Real and fake features must have same number of periods".to_string(),
                    ));
                }

                for (real_period, fake_period) in real_feats.iter().zip(fake_feats.iter()) {
                    let period_fm_loss = fm_loss_calc.compute(real_period, fake_period)?;
                    total_fm_loss += period_fm_loss;
                }
                total_fm_loss /= real_feats.len() as f32;
            }
        }

        Ok(GANLossBreakdown {
            adversarial_loss: total_adv_loss,
            feature_matching_loss: if total_fm_loss > 0.0 {
                Some(total_fm_loss)
            } else {
                None
            },
            total_loss: total_adv_loss + total_fm_loss,
        })
    }

    /// Compute total discriminator loss across all periods
    ///
    /// # Arguments
    /// * `disc_real_outputs` - Discriminator outputs for real audio at each period
    /// * `disc_fake_outputs` - Discriminator outputs for generated audio at each period
    ///
    /// # Returns
    /// * Average discriminator loss across all periods
    pub fn discriminator_loss(
        &self,
        disc_real_outputs: &[Vec<Array1<f32>>],
        disc_fake_outputs: &[Vec<Array1<f32>>],
    ) -> Result<f32> {
        if disc_real_outputs.len() != disc_fake_outputs.len() {
            return Err(VocoderError::InputError(
                "Real and fake outputs must have same number of periods".to_string(),
            ));
        }

        let mut total_loss = 0.0;

        for (real_period, fake_period) in disc_real_outputs.iter().zip(disc_fake_outputs.iter()) {
            let period_loss = self
                .adversarial_loss
                .discriminator_loss(real_period, fake_period)?;
            total_loss += period_loss;
        }

        Ok(total_loss / disc_real_outputs.len() as f32)
    }
}

impl Default for MultiPeriodDiscriminatorLoss {
    fn default() -> Self {
        Self::new(MultiPeriodDiscriminatorLossConfig::default())
    }
}

/// Combined multi-scale and multi-period discriminator loss
///
/// Combines both multi-scale and multi-period discriminators for comprehensive
/// audio quality assessment. Used in advanced vocoder architectures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedDiscriminatorLossConfig {
    /// Multi-scale discriminator configuration
    pub multi_scale_config: MultiScaleDiscriminatorLossConfig,
    /// Multi-period discriminator configuration
    pub multi_period_config: MultiPeriodDiscriminatorLossConfig,
    /// Weight for multi-scale loss
    pub scale_weight: f32,
    /// Weight for multi-period loss
    pub period_weight: f32,
}

impl Default for CombinedDiscriminatorLossConfig {
    fn default() -> Self {
        Self {
            multi_scale_config: MultiScaleDiscriminatorLossConfig::default(),
            multi_period_config: MultiPeriodDiscriminatorLossConfig::default(),
            scale_weight: 1.0,
            period_weight: 1.0,
        }
    }
}

/// Combined multi-scale and multi-period discriminator loss calculator
pub struct CombinedDiscriminatorLoss {
    multi_scale_loss: MultiScaleDiscriminatorLoss,
    multi_period_loss: MultiPeriodDiscriminatorLoss,
    config: CombinedDiscriminatorLossConfig,
}

impl CombinedDiscriminatorLoss {
    /// Create a new combined discriminator loss
    pub fn new(config: CombinedDiscriminatorLossConfig) -> Self {
        let multi_scale_loss = MultiScaleDiscriminatorLoss::new(config.multi_scale_config.clone());
        let multi_period_loss =
            MultiPeriodDiscriminatorLoss::new(config.multi_period_config.clone());

        Self {
            multi_scale_loss,
            multi_period_loss,
            config,
        }
    }

    /// Compute combined generator loss from both multi-scale and multi-period discriminators
    ///
    /// # Arguments
    /// * `scale_fake_outputs` - Multi-scale discriminator outputs for generated audio
    /// * `period_fake_outputs` - Multi-period discriminator outputs for generated audio
    /// * `scale_real_features` - Optional real audio features for multi-scale discriminator
    /// * `scale_fake_features` - Optional generated audio features for multi-scale discriminator
    /// * `period_real_features` - Optional real audio features for multi-period discriminator
    /// * `period_fake_features` - Optional generated audio features for multi-period discriminator
    ///
    /// # Returns
    /// * Combined GAN loss breakdown
    pub fn generator_loss(
        &self,
        scale_fake_outputs: &[Vec<Array1<f32>>],
        period_fake_outputs: &[Vec<Array1<f32>>],
        scale_real_features: Option<&[Vec<Array2<f32>>]>,
        scale_fake_features: Option<&[Vec<Array2<f32>>]>,
        period_real_features: Option<&[Vec<Array2<f32>>]>,
        period_fake_features: Option<&[Vec<Array2<f32>>]>,
    ) -> Result<CombinedGANLossBreakdown> {
        let scale_breakdown = self.multi_scale_loss.generator_loss(
            scale_fake_outputs,
            scale_real_features,
            scale_fake_features,
        )?;

        let period_breakdown = self.multi_period_loss.generator_loss(
            period_fake_outputs,
            period_real_features,
            period_fake_features,
        )?;

        let total_loss = self.config.scale_weight * scale_breakdown.total_loss
            + self.config.period_weight * period_breakdown.total_loss;

        Ok(CombinedGANLossBreakdown {
            scale_breakdown,
            period_breakdown,
            total_loss,
            scale_weight: self.config.scale_weight,
            period_weight: self.config.period_weight,
        })
    }

    /// Compute combined discriminator loss
    pub fn discriminator_loss(
        &self,
        scale_real_outputs: &[Vec<Array1<f32>>],
        scale_fake_outputs: &[Vec<Array1<f32>>],
        period_real_outputs: &[Vec<Array1<f32>>],
        period_fake_outputs: &[Vec<Array1<f32>>],
    ) -> Result<f32> {
        let scale_loss = self
            .multi_scale_loss
            .discriminator_loss(scale_real_outputs, scale_fake_outputs)?;
        let period_loss = self
            .multi_period_loss
            .discriminator_loss(period_real_outputs, period_fake_outputs)?;

        Ok(self.config.scale_weight * scale_loss + self.config.period_weight * period_loss)
    }
}

impl Default for CombinedDiscriminatorLoss {
    fn default() -> Self {
        Self::new(CombinedDiscriminatorLossConfig::default())
    }
}

/// Breakdown of combined multi-scale and multi-period GAN losses
#[derive(Debug, Clone)]
pub struct CombinedGANLossBreakdown {
    /// Multi-scale discriminator loss breakdown
    pub scale_breakdown: GANLossBreakdown,
    /// Multi-period discriminator loss breakdown
    pub period_breakdown: GANLossBreakdown,
    /// Total combined loss
    pub total_loss: f32,
    /// Weight applied to scale loss
    pub scale_weight: f32,
    /// Weight applied to period loss
    pub period_weight: f32,
}

impl CombinedGANLossBreakdown {
    /// Get a comprehensive summary string
    pub fn summary(&self) -> String {
        let mut lines = vec![
            format!("Total Combined Loss: {:.6}", self.total_loss),
            String::new(),
            format!("Multi-Scale Loss (weight: {:.2}):", self.scale_weight),
            format!("  Total: {:.6}", self.scale_breakdown.total_loss),
            format!(
                "  Adversarial: {:.6}",
                self.scale_breakdown.adversarial_loss
            ),
        ];

        if let Some(fm) = self.scale_breakdown.feature_matching_loss {
            lines.push(format!("  Feature Matching: {:.6}", fm));
        }

        lines.push(String::new());
        lines.push(format!(
            "Multi-Period Loss (weight: {:.2}):",
            self.period_weight
        ));
        lines.push(format!("  Total: {:.6}", self.period_breakdown.total_loss));
        lines.push(format!(
            "  Adversarial: {:.6}",
            self.period_breakdown.adversarial_loss
        ));

        if let Some(fm) = self.period_breakdown.feature_matching_loss {
            lines.push(format!("  Feature Matching: {:.6}", fm));
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adversarial_loss_config() {
        let config = AdversarialLossConfig::default();
        assert_eq!(config.loss_type, AdversarialLossType::LeastSquares);
        assert_eq!(config.weight, 1.0);
    }

    #[test]
    fn test_adversarial_loss_generator_least_squares() {
        let config = AdversarialLossConfig {
            loss_type: AdversarialLossType::LeastSquares,
            weight: 1.0,
        };
        let loss = AdversarialLoss::new(config);

        // Perfect discriminator output (all 1s means discriminator thinks it's real)
        let perfect_output = vec![Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0])];
        let perfect_loss = loss.generator_loss(&perfect_output).unwrap();
        assert!(
            perfect_loss < 1e-6,
            "Perfect output should have near-zero loss"
        );

        // Bad discriminator output (all 0s means discriminator thinks it's fake)
        let bad_output = vec![Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0])];
        let bad_loss = loss.generator_loss(&bad_output).unwrap();
        assert!(bad_loss > 0.9, "Bad output should have high loss");
    }

    #[test]
    fn test_adversarial_loss_generator_hinge() {
        let config = AdversarialLossConfig {
            loss_type: AdversarialLossType::Hinge,
            weight: 1.0,
        };
        let loss = AdversarialLoss::new(config);

        let output = vec![Array1::from_vec(vec![0.5, 0.6, 0.7])];
        let loss_value = loss.generator_loss(&output).unwrap();
        assert!(loss_value < 0.0); // Hinge loss for generator is negative
    }

    #[test]
    fn test_adversarial_loss_discriminator() {
        let config = AdversarialLossConfig::default();
        let loss = AdversarialLoss::new(config);

        let real_outputs = vec![Array1::from_vec(vec![0.9, 0.95, 0.85])];
        let fake_outputs = vec![Array1::from_vec(vec![0.1, 0.15, 0.05])];

        let disc_loss = loss
            .discriminator_loss(&real_outputs, &fake_outputs)
            .unwrap();
        assert!(disc_loss > 0.0, "Discriminator loss should be positive");
    }

    #[test]
    fn test_feature_matching_loss() {
        let config = FeatureMatchingLossConfig::default();
        let fm_loss = FeatureMatchingLoss::new(config);

        // Create dummy features
        let real_features = vec![
            Array2::<f32>::from_shape_vec((4, 8), (0..32).map(|x| x as f32).collect()).unwrap(),
            Array2::<f32>::from_shape_vec((4, 16), (0..64).map(|x| x as f32 * 0.5).collect())
                .unwrap(),
        ];

        let fake_features = vec![
            Array2::<f32>::from_shape_vec((4, 8), (0..32).map(|x| x as f32 * 0.9).collect())
                .unwrap(),
            Array2::<f32>::from_shape_vec((4, 16), (0..64).map(|x| x as f32 * 0.45).collect())
                .unwrap(),
        ];

        let loss = fm_loss.compute(&real_features, &fake_features).unwrap();
        assert!(loss > 0.0, "Feature matching loss should be positive");
    }

    #[test]
    fn test_feature_matching_loss_identical_features() {
        let config = FeatureMatchingLossConfig::default();
        let fm_loss = FeatureMatchingLoss::new(config);

        let features =
            vec![
                Array2::<f32>::from_shape_vec((4, 8), (0..32).map(|x| x as f32).collect()).unwrap(),
            ];

        let loss = fm_loss.compute(&features, &features).unwrap();
        assert!(
            loss < 1e-5,
            "Feature matching loss should be near zero for identical features"
        );
    }

    #[test]
    fn test_feature_matching_loss_weighted() {
        let config = FeatureMatchingLossConfig::default();
        let fm_loss = FeatureMatchingLoss::new(config);

        let real_features = vec![
            Array2::<f32>::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                .unwrap(),
            Array2::<f32>::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                .unwrap(),
        ];

        let fake_features = vec![
            Array2::<f32>::from_shape_vec((2, 4), vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1])
                .unwrap(),
            Array2::<f32>::from_shape_vec((2, 4), vec![0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9])
                .unwrap(),
        ];

        let weights = vec![2.0, 1.0];
        let loss = fm_loss
            .compute_weighted(&real_features, &fake_features, &weights)
            .unwrap();
        assert!(loss > 0.0);
    }

    #[test]
    fn test_multi_scale_discriminator_loss() {
        let config = MultiScaleDiscriminatorLossConfig::default();
        let msd_loss = MultiScaleDiscriminatorLoss::new(config);

        // 3 scales
        let disc_fake_outputs = vec![
            vec![Array1::from_vec(vec![0.8, 0.7, 0.9])],
            vec![Array1::from_vec(vec![0.75, 0.85])],
            vec![Array1::from_vec(vec![0.9])],
        ];

        let breakdown = msd_loss
            .generator_loss(&disc_fake_outputs, None, None)
            .unwrap();
        assert!(breakdown.adversarial_loss > 0.0);
        assert!(breakdown.total_loss > 0.0);
        assert!(breakdown.feature_matching_loss.is_none());
    }

    #[test]
    fn test_multi_scale_discriminator_loss_with_features() {
        let config = MultiScaleDiscriminatorLossConfig::default();
        let msd_loss = MultiScaleDiscriminatorLoss::new(config);

        let disc_fake_outputs = vec![vec![Array1::from_vec(vec![0.8, 0.7])]];

        let real_features = vec![vec![Array2::<f32>::from_shape_vec(
            (2, 4),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap()]];

        let fake_features = vec![vec![Array2::<f32>::from_shape_vec(
            (2, 4),
            vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1],
        )
        .unwrap()]];

        let breakdown = msd_loss
            .generator_loss(
                &disc_fake_outputs,
                Some(&real_features),
                Some(&fake_features),
            )
            .unwrap();

        assert!(breakdown.adversarial_loss > 0.0);
        assert!(breakdown.feature_matching_loss.is_some());
        assert!(breakdown.total_loss > 0.0);
    }

    #[test]
    fn test_gan_loss_breakdown_summary() {
        let breakdown = GANLossBreakdown {
            adversarial_loss: 0.5,
            feature_matching_loss: Some(1.2),
            total_loss: 1.7,
        };

        let summary = breakdown.summary();
        assert!(summary.contains("Total GAN Loss: 1.7"));
        assert!(summary.contains("Adversarial: 0.5"));
        assert!(summary.contains("Feature Matching: 1.2"));
    }

    #[test]
    fn test_adversarial_loss_empty_input() {
        let loss = AdversarialLoss::default();
        let empty_outputs: Vec<Array1<f32>> = vec![];
        let result = loss.generator_loss(&empty_outputs);
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_matching_loss_mismatched_shapes() {
        let fm_loss = FeatureMatchingLoss::default();

        let real_features =
            vec![
                Array2::<f32>::from_shape_vec((4, 8), (0..32).map(|x| x as f32).collect()).unwrap(),
            ];

        let fake_features =
            vec![
                Array2::<f32>::from_shape_vec((4, 16), (0..64).map(|x| x as f32).collect())
                    .unwrap(),
            ];

        let result = fm_loss.compute(&real_features, &fake_features);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_period_discriminator_loss_config() {
        let config = MultiPeriodDiscriminatorLossConfig::default();
        assert_eq!(config.periods, vec![2, 3, 5, 7, 11]);
        assert!(config.feature_matching_config.is_some());
    }

    #[test]
    fn test_multi_period_discriminator_loss() {
        let config = MultiPeriodDiscriminatorLossConfig::default();
        let mpd_loss = MultiPeriodDiscriminatorLoss::new(config);

        // 5 periods
        let disc_fake_outputs = vec![
            vec![Array1::from_vec(vec![0.75, 0.72])], // Period 2
            vec![Array1::from_vec(vec![0.78, 0.76])], // Period 3
            vec![Array1::from_vec(vec![0.80, 0.77])], // Period 5
            vec![Array1::from_vec(vec![0.73, 0.79])], // Period 7
            vec![Array1::from_vec(vec![0.74, 0.81])], // Period 11
        ];

        let breakdown = mpd_loss
            .generator_loss(&disc_fake_outputs, None, None)
            .unwrap();
        assert!(breakdown.adversarial_loss > 0.0);
        assert!(breakdown.total_loss > 0.0);
        assert!(breakdown.feature_matching_loss.is_none());
    }

    #[test]
    fn test_multi_period_discriminator_loss_with_features() {
        let config = MultiPeriodDiscriminatorLossConfig::default();
        let mpd_loss = MultiPeriodDiscriminatorLoss::new(config);

        let disc_fake_outputs = vec![
            vec![Array1::from_vec(vec![0.75, 0.72])],
            vec![Array1::from_vec(vec![0.78, 0.76])],
        ];

        let real_features = vec![
            vec![Array2::<f32>::from_shape_vec(
                (2, 4),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            )
            .unwrap()],
            vec![Array2::<f32>::from_shape_vec(
                (2, 4),
                vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1],
            )
            .unwrap()],
        ];

        let fake_features = vec![
            vec![Array2::<f32>::from_shape_vec(
                (2, 4),
                vec![1.05, 2.05, 3.05, 4.05, 5.05, 6.05, 7.05, 8.05],
            )
            .unwrap()],
            vec![Array2::<f32>::from_shape_vec(
                (2, 4),
                vec![1.15, 2.15, 3.15, 4.15, 5.15, 6.15, 7.15, 8.15],
            )
            .unwrap()],
        ];

        let breakdown = mpd_loss
            .generator_loss(
                &disc_fake_outputs,
                Some(&real_features),
                Some(&fake_features),
            )
            .unwrap();

        assert!(breakdown.adversarial_loss > 0.0);
        assert!(breakdown.feature_matching_loss.is_some());
        assert!(breakdown.total_loss > 0.0);
    }

    #[test]
    fn test_multi_period_discriminator_loss_discriminator() {
        let config = MultiPeriodDiscriminatorLossConfig::default();
        let mpd_loss = MultiPeriodDiscriminatorLoss::new(config);

        let disc_real_outputs = vec![
            vec![Array1::from_vec(vec![0.9, 0.92])],
            vec![Array1::from_vec(vec![0.88, 0.91])],
        ];

        let disc_fake_outputs = vec![
            vec![Array1::from_vec(vec![0.2, 0.18])],
            vec![Array1::from_vec(vec![0.15, 0.22])],
        ];

        let disc_loss = mpd_loss
            .discriminator_loss(&disc_real_outputs, &disc_fake_outputs)
            .unwrap();

        assert!(disc_loss > 0.0);
    }

    #[test]
    fn test_combined_discriminator_loss_config() {
        let config = CombinedDiscriminatorLossConfig::default();
        assert_eq!(config.scale_weight, 1.0);
        assert_eq!(config.period_weight, 1.0);
    }

    #[test]
    fn test_combined_discriminator_loss() {
        let config = CombinedDiscriminatorLossConfig::default();
        let combined_loss = CombinedDiscriminatorLoss::new(config);

        // Multi-scale outputs (3 scales)
        let scale_fake_outputs = vec![
            vec![Array1::from_vec(vec![0.8, 0.75])],
            vec![Array1::from_vec(vec![0.77, 0.82])],
            vec![Array1::from_vec(vec![0.79])],
        ];

        // Multi-period outputs (2 periods)
        let period_fake_outputs = vec![
            vec![Array1::from_vec(vec![0.76, 0.73])],
            vec![Array1::from_vec(vec![0.81, 0.78])],
        ];

        let breakdown = combined_loss
            .generator_loss(
                &scale_fake_outputs,
                &period_fake_outputs,
                None,
                None,
                None,
                None,
            )
            .unwrap();

        assert!(breakdown.total_loss > 0.0);
        assert!(breakdown.scale_breakdown.adversarial_loss > 0.0);
        assert!(breakdown.period_breakdown.adversarial_loss > 0.0);
    }

    #[test]
    fn test_combined_discriminator_loss_with_features() {
        let config = CombinedDiscriminatorLossConfig::default();
        let combined_loss = CombinedDiscriminatorLoss::new(config);

        let scale_fake_outputs = vec![vec![Array1::from_vec(vec![0.8, 0.75])]];
        let period_fake_outputs = vec![vec![Array1::from_vec(vec![0.76, 0.73])]];

        let features = vec![vec![Array2::<f32>::from_shape_vec(
            (2, 4),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap()]];

        let fake_features = vec![vec![Array2::<f32>::from_shape_vec(
            (2, 4),
            vec![1.05, 2.05, 3.05, 4.05, 5.05, 6.05, 7.05, 8.05],
        )
        .unwrap()]];

        let breakdown = combined_loss
            .generator_loss(
                &scale_fake_outputs,
                &period_fake_outputs,
                Some(&features),
                Some(&fake_features),
                Some(&features),
                Some(&fake_features),
            )
            .unwrap();

        assert!(breakdown.total_loss > 0.0);
        assert!(breakdown.scale_breakdown.feature_matching_loss.is_some());
        assert!(breakdown.period_breakdown.feature_matching_loss.is_some());
    }

    #[test]
    fn test_combined_discriminator_loss_discriminator() {
        let config = CombinedDiscriminatorLossConfig::default();
        let combined_loss = CombinedDiscriminatorLoss::new(config);

        let scale_real = vec![vec![Array1::from_vec(vec![0.9, 0.92])]];
        let scale_fake = vec![vec![Array1::from_vec(vec![0.2, 0.18])]];
        let period_real = vec![vec![Array1::from_vec(vec![0.88, 0.91])]];
        let period_fake = vec![vec![Array1::from_vec(vec![0.15, 0.22])]];

        let disc_loss = combined_loss
            .discriminator_loss(&scale_real, &scale_fake, &period_real, &period_fake)
            .unwrap();

        assert!(disc_loss > 0.0);
    }

    #[test]
    fn test_combined_breakdown_summary() {
        let breakdown = CombinedGANLossBreakdown {
            scale_breakdown: GANLossBreakdown {
                adversarial_loss: 0.5,
                feature_matching_loss: Some(1.2),
                total_loss: 1.7,
            },
            period_breakdown: GANLossBreakdown {
                adversarial_loss: 0.6,
                feature_matching_loss: Some(1.1),
                total_loss: 1.7,
            },
            total_loss: 3.4,
            scale_weight: 1.0,
            period_weight: 1.0,
        };

        let summary = breakdown.summary();
        assert!(summary.contains("Total Combined Loss: 3.4"));
        assert!(summary.contains("Multi-Scale Loss"));
        assert!(summary.contains("Multi-Period Loss"));
    }
}
