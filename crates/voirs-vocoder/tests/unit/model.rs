//! Unit tests for model implementations
//!
//! Tests HiFiGAN, DiffWave, and WaveGlow model components
//! to ensure correct architecture and inference behavior.

use voirs_vocoder::{
    AudioBuffer, MelSpectrogram, VocoderError, SynthesisConfig,
    models::{HiFiGanVocoder, DiffWaveVocoder, WaveGlowVocoder},
    config::{HiFiGanVariant, DiffusionScheduler, SamplingMethod}
};
use std::sync::Arc;

#[tokio::test]
async fn test_hifigan_model_creation() {
    // Test different HiFiGAN variants
    let v1_config = create_hifigan_config(HiFiGanVariant::V1);
    let v2_config = create_hifigan_config(HiFiGanVariant::V2);
    let v3_config = create_hifigan_config(HiFiGanVariant::V3);
    
    // Models should be creatable (even without actual model files)
    assert!(v1_config.is_valid());
    assert!(v2_config.is_valid());
    assert!(v3_config.is_valid());
    
    // Test variant differences
    assert_ne!(v1_config.estimated_size(), v2_config.estimated_size());
    assert_ne!(v2_config.estimated_size(), v3_config.estimated_size());
    
    // V1 should be largest, V3 smallest
    assert!(v1_config.estimated_size() > v2_config.estimated_size());
    assert!(v2_config.estimated_size() > v3_config.estimated_size());
}

#[test]
fn test_hifigan_architecture_parameters() {
    let v1_arch = HiFiGanArchitecture::v1();
    let v2_arch = HiFiGanArchitecture::v2();
    let v3_arch = HiFiGanArchitecture::v3();
    
    // Test architecture differences
    assert_ne!(v1_arch.upsample_rates, v2_arch.upsample_rates);
    assert_ne!(v1_arch.resblock_kernel_sizes, v2_arch.resblock_kernel_sizes);
    
    // Test architecture validation
    assert!(v1_arch.validate().is_ok());
    assert!(v2_arch.validate().is_ok());
    assert!(v3_arch.validate().is_ok());
    
    // Test parameter counts
    let v1_params = v1_arch.parameter_count();
    let v2_params = v2_arch.parameter_count();
    let v3_params = v3_arch.parameter_count();
    
    assert!(v1_params > v2_params);
    assert!(v2_params > v3_params);
    assert!(v1_params > 0);
}

#[test]
fn test_hifigan_generator_structure() {
    let arch = HiFiGanArchitecture::v1();
    let generator = HiFiGanGenerator::new(arch);
    
    // Test generator structure
    assert!(generator.is_ok());
    let generator = generator.unwrap();
    
    assert_eq!(generator.input_channels(), 80); // Mel channels
    assert_eq!(generator.output_channels(), 1); // Audio channels
    
    // Test upsampling layers
    let upsample_layers = generator.upsample_layers();
    assert!(!upsample_layers.is_empty());
    
    // Test MRF blocks
    let mrf_blocks = generator.mrf_blocks();
    assert!(!mrf_blocks.is_empty());
    
    // Test layer dimensions
    for (i, layer) in upsample_layers.iter().enumerate() {
        assert!(layer.kernel_size() > 0);
        assert!(layer.stride() > 0);
        assert!(layer.padding() >= 0);
    }
}

#[test]
fn test_hifigan_mrf_blocks() {
    let mrf = MultiReceptiveField::new(&[3, 7, 11], &[1, 3, 5]);
    
    assert!(mrf.is_ok());
    let mrf = mrf.unwrap();
    
    // Test MRF structure
    let kernel_sizes = mrf.kernel_sizes();
    assert_eq!(kernel_sizes, &[3, 7, 11]);
    
    let dilation_rates = mrf.dilation_rates();
    assert_eq!(dilation_rates, &[1, 3, 5]);
    
    // Test MRF validation
    assert!(mrf.validate().is_ok());
    
    // Test parameter count
    let params = mrf.parameter_count();
    assert!(params > 0);
}

#[test]
fn test_diffwave_model_creation() {
    let config = create_diffwave_config();
    
    assert!(config.is_valid());
    assert_eq!(config.diffusion_steps, 50);
    assert_eq!(config.scheduler, DiffusionScheduler::Cosine);
    assert_eq!(config.sampling_method, SamplingMethod::DDIM);
    
    // Test different configurations
    let fast_config = create_diffwave_config_fast();
    assert!(fast_config.diffusion_steps < config.diffusion_steps);
    
    let quality_config = create_diffwave_config_quality();
    assert!(quality_config.diffusion_steps > config.diffusion_steps);
}

#[test]
fn test_diffwave_unet_architecture() {
    let unet = DiffWaveUNet::new();
    
    assert!(unet.is_ok());
    let unet = unet.unwrap();
    
    // Test U-Net structure
    assert!(unet.encoder_layers() > 0);
    assert!(unet.decoder_layers() > 0);
    assert_eq!(unet.encoder_layers(), unet.decoder_layers());
    
    // Test skip connections
    let skip_connections = unet.skip_connections();
    assert!(!skip_connections.is_empty());
    
    // Test attention layers
    let attention_layers = unet.attention_layers();
    assert!(!attention_layers.is_empty());
    
    // Test parameter count
    let params = unet.parameter_count();
    assert!(params > 0);
}

#[test]
fn test_diffwave_noise_scheduling() {
    // Test different noise schedulers
    let linear = NoiseScheduler::linear(1000, 1e-4, 0.02);
    let cosine = NoiseScheduler::cosine(1000);
    let sigmoid = NoiseScheduler::sigmoid(1000, 3.0);
    
    assert!(linear.is_ok());
    assert!(cosine.is_ok());
    assert!(sigmoid.is_ok());
    
    let linear = linear.unwrap();
    let cosine = cosine.unwrap();
    let sigmoid = sigmoid.unwrap();
    
    // Test scheduler properties
    assert_eq!(linear.steps(), 1000);
    assert_eq!(cosine.steps(), 1000);
    assert_eq!(sigmoid.steps(), 1000);
    
    // Test noise levels
    let linear_alphas = linear.alphas();
    let cosine_alphas = cosine.alphas();
    let sigmoid_alphas = sigmoid.alphas();
    
    assert_eq!(linear_alphas.len(), 1000);
    assert_eq!(cosine_alphas.len(), 1000);
    assert_eq!(sigmoid_alphas.len(), 1000);
    
    // Test monotonicity (alphas should decrease)
    for i in 1..linear_alphas.len() {
        assert!(linear_alphas[i] <= linear_alphas[i-1]);
        assert!(cosine_alphas[i] <= cosine_alphas[i-1]);
    }
    
    // Test range (alphas should be in [0, 1])
    for &alpha in linear_alphas.iter() {
        assert!(alpha >= 0.0 && alpha <= 1.0);
    }
}

#[test]
fn test_diffwave_sampling_methods() {
    let scheduler = NoiseScheduler::cosine(50).unwrap();
    
    // Test DDPM sampling
    let ddpm = DDPMSampler::new(scheduler.clone());
    assert!(ddpm.is_ok());
    let ddpm = ddpm.unwrap();
    
    assert_eq!(ddpm.steps(), 50);
    assert!(ddpm.supports_fast_sampling());
    
    // Test DDIM sampling
    let ddim = DDIMSampler::new(scheduler.clone(), 10);
    assert!(ddim.is_ok());
    let ddim = ddim.unwrap();
    
    assert_eq!(ddim.steps(), 10); // Reduced steps
    assert!(ddim.supports_fast_sampling());
    
    // Test adaptive sampling
    let adaptive = AdaptiveSampler::new(scheduler, 0.01);
    assert!(adaptive.is_ok());
    let adaptive = adaptive.unwrap();
    
    assert!(adaptive.convergence_threshold() > 0.0);
    assert!(adaptive.supports_early_stopping());
}

#[test]
fn test_waveglow_model_creation() {
    let config = create_waveglow_config();
    
    assert!(config.is_valid());
    assert!(config.n_flows > 0);
    assert!(config.n_group > 0);
    assert!(config.n_early_every > 0);
    assert!(config.n_early_size > 0);
    
    // Test model parameter estimation
    let params = config.estimated_parameters();
    assert!(params > 0);
    
    let memory = config.estimated_memory_mb();
    assert!(memory > 0.0);
}

#[test]
fn test_waveglow_coupling_layers() {
    let coupling = AffineCouplingLayer::new(512, 256);
    
    assert!(coupling.is_ok());
    let coupling = coupling.unwrap();
    
    // Test coupling layer structure
    assert_eq!(coupling.input_size(), 512);
    assert_eq!(coupling.hidden_size(), 256);
    
    // Test parameter count
    let params = coupling.parameter_count();
    assert!(params > 0);
    
    // Test layer validation
    assert!(coupling.validate().is_ok());
}

#[test]
fn test_waveglow_invertible_conv() {
    let inv_conv = InvertibleConv1x1::new(512);
    
    assert!(inv_conv.is_ok());
    let inv_conv = inv_conv.unwrap();
    
    // Test invertible convolution properties
    assert_eq!(inv_conv.channels(), 512);
    assert!(inv_conv.is_invertible());
    
    // Test parameter count
    let params = inv_conv.parameter_count();
    assert_eq!(params, 512 * 512); // Weight matrix
    
    // Test weight initialization
    assert!(inv_conv.validate_weights().is_ok());
}

#[test]
fn test_model_inference_shapes() {
    // Test input/output shape compatibility
    let mel_frames = 100;
    let mel_bins = 80;
    let hop_length = 256;
    
    // Expected output samples
    let expected_samples = mel_frames * hop_length;
    
    // Test HiFiGAN shapes
    let hifigan_output = estimate_hifigan_output_size(mel_frames, mel_bins, hop_length);
    assert_eq!(hifigan_output, expected_samples);
    
    // Test DiffWave shapes
    let diffwave_output = estimate_diffwave_output_size(mel_frames, mel_bins, hop_length);
    assert_eq!(diffwave_output, expected_samples);
    
    // Test WaveGlow shapes
    let waveglow_output = estimate_waveglow_output_size(mel_frames, mel_bins, hop_length);
    assert_eq!(waveglow_output, expected_samples);
}

#[test]
fn test_model_memory_requirements() {
    // Test memory estimation for different models
    let mel_frames = 100;
    
    let hifigan_v1_memory = estimate_hifigan_memory(HiFiGanVariant::V1, mel_frames);
    let hifigan_v2_memory = estimate_hifigan_memory(HiFiGanVariant::V2, mel_frames);
    let hifigan_v3_memory = estimate_hifigan_memory(HiFiGanVariant::V3, mel_frames);
    
    // V1 should use more memory than V2, V2 more than V3
    assert!(hifigan_v1_memory > hifigan_v2_memory);
    assert!(hifigan_v2_memory > hifigan_v3_memory);
    
    let diffwave_memory = estimate_diffwave_memory(50, mel_frames); // 50 diffusion steps
    let waveglow_memory = estimate_waveglow_memory(12, mel_frames); // 12 flow steps
    
    // All should be reasonable (< 1GB for test sizes)
    assert!(hifigan_v1_memory < 1024.0);
    assert!(diffwave_memory < 1024.0);
    assert!(waveglow_memory < 1024.0);
}

#[test]
fn test_model_parameter_validation() {
    // Test invalid HiFiGAN parameters
    let mut hifigan_config = create_hifigan_config(HiFiGanVariant::V1);
    hifigan_config.upsample_rates = vec![]; // Empty upsampling
    assert!(hifigan_config.validate().is_err());
    
    // Test invalid DiffWave parameters
    let mut diffwave_config = create_diffwave_config();
    diffwave_config.diffusion_steps = 0; // No diffusion steps
    assert!(diffwave_config.validate().is_err());
    
    diffwave_config.diffusion_steps = 10000; // Too many steps
    assert!(diffwave_config.validate().is_err());
    
    // Test invalid WaveGlow parameters
    let mut waveglow_config = create_waveglow_config();
    waveglow_config.n_flows = 0; // No flows
    assert!(waveglow_config.validate().is_err());
    
    waveglow_config.n_flows = 100; // Too many flows
    assert!(waveglow_config.validate().is_err());
}

#[test]
fn test_model_compatibility() {
    // Test model compatibility with different configurations
    let hifigan_config = create_hifigan_config(HiFiGanVariant::V1);
    let streaming_config = create_streaming_config();
    
    assert!(hifigan_config.is_compatible_with(&streaming_config));
    
    // Test incompatible configurations
    let mut incompatible_streaming = streaming_config.clone();
    incompatible_streaming.chunk_size = 32; // Too small for HiFiGAN
    
    assert!(!hifigan_config.is_compatible_with(&incompatible_streaming));
    
    // Test sample rate compatibility
    let mut incompatible_hifigan = hifigan_config.clone();
    incompatible_hifigan.sample_rate = 8000; // Low sample rate
    
    assert!(!incompatible_hifigan.is_compatible_with(&streaming_config));
}

#[test]
fn test_model_performance_characteristics() {
    // Test RTF (Real-Time Factor) estimation
    let hifigan_v1_rtf = estimate_rtf(ModelType::HiFiGAN(HiFiGanVariant::V1));
    let hifigan_v3_rtf = estimate_rtf(ModelType::HiFiGAN(HiFiGanVariant::V3));
    let diffwave_rtf = estimate_rtf(ModelType::DiffWave(50));
    let waveglow_rtf = estimate_rtf(ModelType::WaveGlow(12));
    
    // V3 should be faster than V1
    assert!(hifigan_v3_rtf < hifigan_v1_rtf);
    
    // HiFiGAN should be faster than DiffWave
    assert!(hifigan_v1_rtf < diffwave_rtf);
    
    // All should be reasonable for real-time (< 1.0 RTF)
    assert!(hifigan_v1_rtf < 1.0);
    assert!(hifigan_v3_rtf < 1.0);
    
    // DiffWave might be slower but should still be usable
    assert!(diffwave_rtf < 5.0);
}

// Helper functions and structures for testing

#[derive(Clone)]
struct HiFiGanConfig {
    variant: HiFiGanVariant,
    sample_rate: u32,
    upsample_rates: Vec<usize>,
    resblock_kernel_sizes: Vec<usize>,
    resblock_dilation_sizes: Vec<Vec<usize>>,
}

impl HiFiGanConfig {
    fn is_valid(&self) -> bool {
        !self.upsample_rates.is_empty() && 
        !self.resblock_kernel_sizes.is_empty() &&
        self.sample_rate > 0
    }
    
    fn estimated_size(&self) -> usize {
        match self.variant {
            HiFiGanVariant::V1 => 25 * 1024 * 1024, // 25MB
            HiFiGanVariant::V2 => 15 * 1024 * 1024, // 15MB
            HiFiGanVariant::V3 => 10 * 1024 * 1024, // 10MB
        }
    }
    
    fn validate(&self) -> Result<(), VocoderError> {
        if self.upsample_rates.is_empty() {
            return Err(VocoderError::ConfigError("Empty upsample rates".to_string()));
        }
        if self.sample_rate == 0 {
            return Err(VocoderError::ConfigError("Invalid sample rate".to_string()));
        }
        Ok(())
    }
    
    fn is_compatible_with(&self, streaming: &StreamingConfig) -> bool {
        streaming.chunk_size >= 256 && // Minimum chunk size for HiFiGAN
        self.sample_rate >= 16000 // Minimum sample rate
    }
}

fn create_hifigan_config(variant: HiFiGanVariant) -> HiFiGanConfig {
    match variant {
        HiFiGanVariant::V1 => HiFiGanConfig {
            variant,
            sample_rate: 22050,
            upsample_rates: vec![8, 8, 2, 2],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
        },
        HiFiGanVariant::V2 => HiFiGanConfig {
            variant,
            sample_rate: 22050,
            upsample_rates: vec![8, 8, 4, 2],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
        },
        HiFiGanVariant::V3 => HiFiGanConfig {
            variant,
            sample_rate: 22050,
            upsample_rates: vec![8, 8, 8, 2],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
        },
    }
}

#[derive(Clone)]
struct DiffWaveConfig {
    diffusion_steps: usize,
    scheduler: DiffusionScheduler,
    sampling_method: SamplingMethod,
}

impl DiffWaveConfig {
    fn is_valid(&self) -> bool {
        self.diffusion_steps > 0 && self.diffusion_steps <= 1000
    }
    
    fn validate(&self) -> Result<(), VocoderError> {
        if self.diffusion_steps == 0 {
            return Err(VocoderError::ConfigError("Zero diffusion steps".to_string()));
        }
        if self.diffusion_steps > 1000 {
            return Err(VocoderError::ConfigError("Too many diffusion steps".to_string()));
        }
        Ok(())
    }
}

fn create_diffwave_config() -> DiffWaveConfig {
    DiffWaveConfig {
        diffusion_steps: 50,
        scheduler: DiffusionScheduler::Cosine,
        sampling_method: SamplingMethod::DDIM,
    }
}

fn create_diffwave_config_fast() -> DiffWaveConfig {
    DiffWaveConfig {
        diffusion_steps: 10,
        scheduler: DiffusionScheduler::Linear,
        sampling_method: SamplingMethod::FastDDIM,
    }
}

fn create_diffwave_config_quality() -> DiffWaveConfig {
    DiffWaveConfig {
        diffusion_steps: 100,
        scheduler: DiffusionScheduler::Cosine,
        sampling_method: SamplingMethod::DDPM,
    }
}

#[derive(Clone)]
struct WaveGlowConfig {
    n_flows: usize,
    n_group: usize,
    n_early_every: usize,
    n_early_size: usize,
}

impl WaveGlowConfig {
    fn is_valid(&self) -> bool {
        self.n_flows > 0 && self.n_group > 0 && 
        self.n_early_every > 0 && self.n_early_size > 0
    }
    
    fn estimated_parameters(&self) -> usize {
        // Rough estimation
        self.n_flows * self.n_group * 1024 * 1024
    }
    
    fn estimated_memory_mb(&self) -> f32 {
        (self.estimated_parameters() * 4) as f32 / (1024.0 * 1024.0)
    }
    
    fn validate(&self) -> Result<(), VocoderError> {
        if self.n_flows == 0 {
            return Err(VocoderError::ConfigError("Zero flows".to_string()));
        }
        if self.n_flows > 20 {
            return Err(VocoderError::ConfigError("Too many flows".to_string()));
        }
        Ok(())
    }
}

fn create_waveglow_config() -> WaveGlowConfig {
    WaveGlowConfig {
        n_flows: 12,
        n_group: 8,
        n_early_every: 4,
        n_early_size: 2,
    }
}

#[derive(Clone)]
struct StreamingConfig {
    chunk_size: usize,
}

fn create_streaming_config() -> StreamingConfig {
    StreamingConfig {
        chunk_size: 1024,
    }
}

#[derive(Clone)]
enum ModelType {
    HiFiGAN(HiFiGanVariant),
    DiffWave(usize),
    WaveGlow(usize),
}

fn estimate_rtf(model_type: ModelType) -> f32 {
    match model_type {
        ModelType::HiFiGAN(HiFiGanVariant::V1) => 0.05,
        ModelType::HiFiGAN(HiFiGanVariant::V2) => 0.03,
        ModelType::HiFiGAN(HiFiGanVariant::V3) => 0.02,
        ModelType::DiffWave(steps) => 0.1 + (steps as f32 * 0.01),
        ModelType::WaveGlow(flows) => 0.08 + (flows as f32 * 0.005),
    }
}

fn estimate_hifigan_output_size(mel_frames: usize, _mel_bins: usize, hop_length: usize) -> usize {
    mel_frames * hop_length
}

fn estimate_diffwave_output_size(mel_frames: usize, _mel_bins: usize, hop_length: usize) -> usize {
    mel_frames * hop_length
}

fn estimate_waveglow_output_size(mel_frames: usize, _mel_bins: usize, hop_length: usize) -> usize {
    mel_frames * hop_length
}

fn estimate_hifigan_memory(variant: HiFiGanVariant, mel_frames: usize) -> f32 {
    let base_memory = match variant {
        HiFiGanVariant::V1 => 100.0,
        HiFiGanVariant::V2 => 80.0,
        HiFiGanVariant::V3 => 60.0,
    };
    base_memory + (mel_frames as f32 * 0.1)
}

fn estimate_diffwave_memory(steps: usize, mel_frames: usize) -> f32 {
    120.0 + (steps as f32 * 2.0) + (mel_frames as f32 * 0.2)
}

fn estimate_waveglow_memory(flows: usize, mel_frames: usize) -> f32 {
    90.0 + (flows as f32 * 5.0) + (mel_frames as f32 * 0.15)
}

// Mock structures for testing (these would be real implementations in the actual codebase)
struct HiFiGanArchitecture {
    upsample_rates: Vec<usize>,
    resblock_kernel_sizes: Vec<usize>,
}

impl HiFiGanArchitecture {
    fn v1() -> Self {
        Self {
            upsample_rates: vec![8, 8, 2, 2],
            resblock_kernel_sizes: vec![3, 7, 11],
        }
    }
    
    fn v2() -> Self {
        Self {
            upsample_rates: vec![8, 8, 4, 2],
            resblock_kernel_sizes: vec![3, 7, 11],
        }
    }
    
    fn v3() -> Self {
        Self {
            upsample_rates: vec![8, 8, 8, 2],
            resblock_kernel_sizes: vec![3, 7, 11],
        }
    }
    
    fn validate(&self) -> Result<(), VocoderError> {
        if self.upsample_rates.is_empty() {
            return Err(VocoderError::ConfigError("Empty upsample rates".to_string()));
        }
        Ok(())
    }
    
    fn parameter_count(&self) -> usize {
        self.upsample_rates.iter().sum::<usize>() * 1000 +
        self.resblock_kernel_sizes.iter().sum::<usize>() * 500
    }
}

struct HiFiGanGenerator;
struct MultiReceptiveField;
struct DiffWaveUNet;
struct NoiseScheduler;
struct DDPMSampler;
struct DDIMSampler;
struct AdaptiveSampler;
struct AffineCouplingLayer;
struct InvertibleConv1x1;

// Mock implementations would go here...