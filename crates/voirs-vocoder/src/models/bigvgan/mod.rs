//! BigVGAN: A Universal Neural Vocoder with Large-Scale Training
//!
//! BigVGAN is a state-of-the-art neural vocoder that improves upon HiFi-GAN
//! with anti-aliased periodic activations and multi-periodicity composition.
//!
//! # Features
//!
//! - **Anti-Aliased Snake Activation**: Periodic activation with built-in anti-aliasing
//! - **Multi-Periodicity Composition (AMP)**: Rich harmonic structure through multiple periods
//! - **Multi-Receptive Field Fusion (MRF)**: Multi-scale feature extraction
//! - **High-Quality Audio**: Superior to HiFi-GAN in quality metrics
//! - **Multiple Variants**: Base, Large, Fast, and Ultra models
//!
//! # Architecture Highlights
//!
//! 1. **Snake Activation**: `snake(x) = x + (1/α) * sin²(αx)`
//!    - Periodic nonlinearity introduces harmonic richness
//!    - Learnable frequency parameter α
//!    - Maintains good gradient flow
//!
//! 2. **Anti-Aliasing Filter**: Low-pass filtering after periodic activation
//!    - Prevents aliasing artifacts from nonlinearity
//!    - Kaiser window FIR filter design
//!    - Configurable filter order and cutoff
//!
//! 3. **AMP Block**: Multiple Snake activations with different periods
//!    - Captures multi-scale periodicities
//!    - Fused with 1x1 convolution
//!    - Optional for speed vs quality trade-off
//!
//! # Usage
//!
//! ```rust,ignore
//! use voirs_vocoder::models::bigvgan::{BigVGANInference, BigVGANVariant};
//! use candle_core::Device;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Load pretrained model
//!     let device = Device::cuda_if_available(0)?;
//!     let mut inference = BigVGANInference::from_pretrained(
//!         BigVGANVariant::Base,
//!         device
//!     )?;
//!
//!     // Load weights
//!     inference.load_weights("path/to/bigvgan_base.safetensors")?;
//!
//!     // Generate audio from mel spectrogram
//!     let mel = /* your mel spectrogram tensor */;
//!     let waveform = inference.generate(&mel)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! # Model Variants
//!
//! - **Base** (24kHz): Balanced quality and speed (~14M parameters, MOS 4.5)
//! - **Large** (24kHz): Highest quality at 24kHz (~112M parameters, MOS 4.7)
//! - **Fast** (24kHz): Optimized for speed (~4M parameters, MOS 4.3)
//! - **Ultra** (48kHz): Ultra-high quality at 48kHz (~150M parameters, MOS 4.8)
//!
//! # Performance
//!
//! BigVGAN achieves superior quality compared to HiFi-GAN while maintaining
//! competitive inference speed:
//!
//! - **Base**: RTF ~0.08 on GPU, MOS 4.5
//! - **Large**: RTF ~0.12 on GPU, MOS 4.7
//! - **Fast**: RTF ~0.05 on GPU, MOS 4.3
//! - **Ultra**: RTF ~0.15 on GPU, MOS 4.8
//!
//! # References
//!
//! - Lee, S. G., et al. (2023). "BigVGAN: A Universal Neural Vocoder with
//!   Large-Scale Training." arXiv preprint arXiv:2206.04658.

pub mod activation;
pub mod config;
pub mod generator;
pub mod inference;

// Re-export main types
pub use activation::{ActivationConfig, SnakeActivationCpu};
pub use config::{BigVGANConfig, BigVGANMetadata, BigVGANVariant};

#[cfg(feature = "candle")]
pub use activation::{AntiAliasedSnakeActivation, SnakeActivation};

#[cfg(feature = "candle")]
pub use generator::{AMPBlock, BigVGANGenerator, ResBlock, MRF};

#[cfg(feature = "candle")]
pub use inference::BigVGANInference;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Test that all main types are accessible
        let _ = BigVGANVariant::Base;
        let _ = BigVGANConfig::default();
        let _ = ActivationConfig::default();
    }

    #[test]
    fn test_variant_configs() {
        for variant in [
            BigVGANVariant::Base,
            BigVGANVariant::Large,
            BigVGANVariant::Fast,
            BigVGANVariant::Ultra,
        ] {
            let config = variant.config();
            assert!(config.validate().is_ok());

            let metadata = BigVGANMetadata::for_variant(variant);
            assert_eq!(metadata.variant, variant);
        }
    }

    #[test]
    fn test_snake_activation_cpu() {
        use scirs2_core::ndarray::prelude::*;

        let config = ActivationConfig::default();
        let snake = SnakeActivationCpu::new(config);

        let x = Array1::<f32>::from_vec(vec![0.0, 1.0, -1.0]);
        let y = snake.forward(&x);

        assert_eq!(y.len(), x.len());
        // Snake(0) should be 0
        assert!(y[0].abs() < 1e-6);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_bigvgan_inference_api() {
        use candle_core::Device;

        let device = Device::Cpu;
        let variant = BigVGANVariant::Fast;

        let inference = BigVGANInference::from_pretrained(variant, device);
        assert!(inference.is_ok());
    }
}
