//! Unit tests for configuration system
//!
//! Tests configuration validation, serialization, and default values
//! to ensure robust configuration management.

use voirs_vocoder::config::{
    VocodingConfig, StreamingConfig, ModelConfig, QualityLevel, 
    PerformanceProfile, VocodingMode, DeviceType
};
use serde_json;

#[test]
fn test_vocoding_config_defaults() {
    let config = VocodingConfig::default();
    
    assert_eq!(config.quality, QualityLevel::Medium);
    assert_eq!(config.mode, VocodingMode::Balanced);
    assert_eq!(config.profile, PerformanceProfile::Balanced);
    assert_eq!(config.sample_rate, 22050);
    assert_eq!(config.bit_depth, 16);
    assert_eq!(config.mel_channels, 80);
    assert_eq!(config.temperature, 1.0);
    assert_eq!(config.guidance_scale, 1.0);
}

#[test]
fn test_vocoding_config_quality_levels() {
    let mut config = VocodingConfig::default();
    
    // Test different quality levels
    config.quality = QualityLevel::Low;
    assert_eq!(config.estimated_rtf(), 0.01);
    
    config.quality = QualityLevel::Medium;
    assert_eq!(config.estimated_rtf(), 0.02);
    
    config.quality = QualityLevel::High;
    assert_eq!(config.estimated_rtf(), 0.05);
    
    config.quality = QualityLevel::Ultra;
    assert_eq!(config.estimated_rtf(), 0.1);
}

#[test]
fn test_vocoding_config_memory_estimation() {
    let mut config = VocodingConfig::default();
    
    // Test memory estimation for different configurations
    config.sample_rate = 22050;
    config.channels = 1;
    let mono_22k = config.estimated_memory_mb();
    
    config.sample_rate = 44100;
    let mono_44k = config.estimated_memory_mb();
    assert!(mono_44k > mono_22k);
    
    config.channels = 2;
    let stereo_44k = config.estimated_memory_mb();
    assert!(stereo_44k > mono_44k);
    
    config.quality = QualityLevel::Ultra;
    let ultra_stereo = config.estimated_memory_mb();
    assert!(ultra_stereo > stereo_44k);
}

#[test]
fn test_vocoding_config_validation() {
    let mut config = VocodingConfig::default();
    
    // Test valid configurations
    assert!(config.validate().is_ok());
    
    // Test invalid sample rates
    config.sample_rate = 0;
    assert!(config.validate().is_err());
    
    config.sample_rate = 1000; // Too low
    assert!(config.validate().is_err());
    
    config.sample_rate = 200000; // Too high
    assert!(config.validate().is_err());
    
    // Reset to valid
    config.sample_rate = 22050;
    assert!(config.validate().is_ok());
    
    // Test invalid bit depths
    config.bit_depth = 12; // Not supported
    assert!(config.validate().is_err());
    
    config.bit_depth = 16;
    assert!(config.validate().is_ok());
    
    // Test invalid channels
    config.channels = 0;
    assert!(config.validate().is_err());
    
    config.channels = 10; // Too many
    assert!(config.validate().is_err());
    
    config.channels = 2;
    assert!(config.validate().is_ok());
    
    // Test invalid temperature
    config.temperature = -1.0;
    assert!(config.validate().is_err());
    
    config.temperature = 0.0;
    assert!(config.validate().is_err());
    
    config.temperature = 5.0; // Too high
    assert!(config.validate().is_err());
    
    config.temperature = 1.0;
    assert!(config.validate().is_ok());
}

#[test]
fn test_streaming_config_defaults() {
    let config = StreamingConfig::default();
    
    assert_eq!(config.chunk_size, 1024);
    assert_eq!(config.overlap_size, 256);
    assert_eq!(config.buffer_size, 8192);
    assert_eq!(config.latency_mode, LatencyMode::Balanced);
    assert_eq!(config.max_latency_ms, 100.0);
    assert_eq!(config.enable_prediction, false);
    assert_eq!(config.enable_lookahead, false);
    assert_eq!(config.lookahead_samples, 512);
    assert_eq!(config.adaptive_chunking, true);
    assert_eq!(config.max_concurrent_chunks, 4);
}

#[test]
fn test_streaming_config_latency_modes() {
    let mut config = StreamingConfig::default();
    
    // Test different latency modes
    config.latency_mode = LatencyMode::UltraLow;
    assert_eq!(config.estimated_latency_ms(), 25.0);
    
    config.latency_mode = LatencyMode::Low;
    assert_eq!(config.estimated_latency_ms(), 50.0);
    
    config.latency_mode = LatencyMode::Balanced;
    assert_eq!(config.estimated_latency_ms(), 100.0);
    
    config.latency_mode = LatencyMode::Quality;
    assert_eq!(config.estimated_latency_ms(), 200.0);
}

#[test]
fn test_streaming_config_buffer_calculations() {
    let mut config = StreamingConfig::default();
    
    // Test buffer calculations
    config.chunk_size = 1024;
    config.overlap_size = 256;
    
    let effective_chunk = config.effective_chunk_size();
    assert_eq!(effective_chunk, 1024 - 256);
    
    let chunks_per_buffer = config.chunks_per_buffer();
    assert_eq!(chunks_per_buffer, config.buffer_size / effective_chunk);
    
    // Test memory estimation
    let memory_mb = config.estimated_memory_mb();
    assert!(memory_mb > 0.0);
    
    // Larger buffers should use more memory
    config.buffer_size = 16384;
    let larger_memory = config.estimated_memory_mb();
    assert!(larger_memory > memory_mb);
}

#[test]
fn test_streaming_config_validation() {
    let mut config = StreamingConfig::default();
    
    // Test valid configuration
    assert!(config.validate().is_ok());
    
    // Test invalid chunk sizes
    config.chunk_size = 0;
    assert!(config.validate().is_err());
    
    config.chunk_size = 64; // Too small
    assert!(config.validate().is_err());
    
    config.chunk_size = 1024;
    assert!(config.validate().is_ok());
    
    // Test invalid overlap
    config.overlap_size = config.chunk_size; // Overlap >= chunk size
    assert!(config.validate().is_err());
    
    config.overlap_size = 256;
    assert!(config.validate().is_ok());
    
    // Test invalid buffer size
    config.buffer_size = config.chunk_size - 1; // Buffer < chunk
    assert!(config.validate().is_err());
    
    config.buffer_size = 8192;
    assert!(config.validate().is_ok());
    
    // Test invalid latency
    config.max_latency_ms = 0.0;
    assert!(config.validate().is_err());
    
    config.max_latency_ms = 100.0;
    assert!(config.validate().is_ok());
    
    // Test invalid concurrent chunks
    config.max_concurrent_chunks = 0;
    assert!(config.validate().is_err());
    
    config.max_concurrent_chunks = 100; // Too many
    assert!(config.validate().is_err());
    
    config.max_concurrent_chunks = 4;
    assert!(config.validate().is_ok());
}

#[test]
fn test_model_config_defaults() {
    let config = ModelConfig::default();
    
    assert_eq!(config.model_type, "hifigan");
    assert_eq!(config.hifigan_variant, HiFiGanVariant::V1);
    assert_eq!(config.device, "auto");
    assert_eq!(config.enable_optimization, true);
    assert_eq!(config.enable_quantization, false);
    assert_eq!(config.cache_models, true);
    assert_eq!(config.diffusion_steps, 50);
    assert_eq!(config.scheduler, DiffusionScheduler::Cosine);
    assert_eq!(config.sampling_method, SamplingMethod::DDIM);
}

#[test]
fn test_model_config_hifigan_variants() {
    let mut config = ModelConfig::default();
    
    // Test different HiFiGAN variants
    config.hifigan_variant = HiFiGanVariant::V1;
    assert_eq!(config.estimated_model_size_mb(), 25.0);
    
    config.hifigan_variant = HiFiGanVariant::V2;
    assert_eq!(config.estimated_model_size_mb(), 15.0);
    
    config.hifigan_variant = HiFiGanVariant::V3;
    assert_eq!(config.estimated_model_size_mb(), 10.0);
}

#[test]
fn test_model_config_diffusion_settings() {
    let mut config = ModelConfig::default();
    
    // Test different diffusion steps
    config.diffusion_steps = 10;
    assert_eq!(config.estimated_inference_time_ms(), 100.0);
    
    config.diffusion_steps = 50;
    assert_eq!(config.estimated_inference_time_ms(), 500.0);
    
    config.diffusion_steps = 100;
    assert_eq!(config.estimated_inference_time_ms(), 1000.0);
    
    // Test different schedulers
    config.scheduler = DiffusionScheduler::Linear;
    assert_eq!(config.scheduler, DiffusionScheduler::Linear);
    
    config.scheduler = DiffusionScheduler::Cosine;
    assert_eq!(config.scheduler, DiffusionScheduler::Cosine);
    
    config.scheduler = DiffusionScheduler::Sigmoid;
    assert_eq!(config.scheduler, DiffusionScheduler::Sigmoid);
}

#[test]
fn test_model_config_validation() {
    let mut config = ModelConfig::default();
    
    // Test valid configuration
    assert!(config.validate().is_ok());
    
    // Test invalid model type
    config.model_type = "invalid_model".to_string();
    assert!(config.validate().is_err());
    
    config.model_type = "hifigan".to_string();
    assert!(config.validate().is_ok());
    
    // Test invalid device
    config.device = "".to_string();
    assert!(config.validate().is_err());
    
    config.device = "auto".to_string();
    assert!(config.validate().is_ok());
    
    // Test invalid diffusion steps
    config.diffusion_steps = 0;
    assert!(config.validate().is_err());
    
    config.diffusion_steps = 1000; // Too many
    assert!(config.validate().is_err());
    
    config.diffusion_steps = 50;
    assert!(config.validate().is_ok());
    
    // Test invalid model path
    config.model_path = Some("".to_string());
    assert!(config.validate().is_err());
    
    config.model_path = Some("/valid/path/model.safetensors".to_string());
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_serialization() {
    let vocoding_config = VocodingConfig::default();
    let streaming_config = StreamingConfig::default();
    let model_config = ModelConfig::default();
    
    // Test JSON serialization
    let vocoding_json = serde_json::to_string(&vocoding_config).unwrap();
    let streaming_json = serde_json::to_string(&streaming_config).unwrap();
    let model_json = serde_json::to_string(&model_config).unwrap();
    
    assert!(vocoding_json.contains("quality"));
    assert!(streaming_json.contains("chunk_size"));
    assert!(model_json.contains("model_type"));
    
    // Test deserialization
    let vocoding_deserialized: VocodingConfig = serde_json::from_str(&vocoding_json).unwrap();
    let streaming_deserialized: StreamingConfig = serde_json::from_str(&streaming_json).unwrap();
    let model_deserialized: ModelConfig = serde_json::from_str(&model_json).unwrap();
    
    assert_eq!(vocoding_config.quality, vocoding_deserialized.quality);
    assert_eq!(streaming_config.chunk_size, streaming_deserialized.chunk_size);
    assert_eq!(model_config.model_type, model_deserialized.model_type);
}

#[test]
fn test_config_cloning() {
    let original = VocodingConfig::default();
    let cloned = original.clone();
    
    assert_eq!(original.quality, cloned.quality);
    assert_eq!(original.sample_rate, cloned.sample_rate);
    assert_eq!(original.temperature, cloned.temperature);
    
    // Test that modifications don't affect original
    let mut modified = original.clone();
    modified.sample_rate = 44100;
    
    assert_eq!(original.sample_rate, 22050);
    assert_eq!(modified.sample_rate, 44100);
}

#[test]
fn test_config_debug_display() {
    let config = VocodingConfig::default();
    let debug_str = format!("{:?}", config);
    
    assert!(debug_str.contains("VocodingConfig"));
    assert!(debug_str.contains("quality"));
    assert!(debug_str.contains("sample_rate"));
}

#[test]
fn test_config_compatibility() {
    let vocoding = VocodingConfig::default();
    let streaming = StreamingConfig::default();
    let model = ModelConfig::default();
    
    // Test that configs are compatible
    assert!(vocoding.is_compatible_with(&streaming));
    assert!(streaming.is_compatible_with(&model));
    
    // Test incompatible configurations
    let mut incompatible_streaming = streaming.clone();
    incompatible_streaming.chunk_size = 64; // Too small for quality
    
    assert!(!vocoding.is_compatible_with(&incompatible_streaming));
    
    let mut incompatible_model = model.clone();
    incompatible_model.model_type = "diffwave".to_string();
    
    // DiffWave might have different requirements
    assert!(streaming.is_compatible_with(&incompatible_model));
}

#[test]
fn test_config_optimization_suggestions() {
    let mut config = VocodingConfig::default();
    
    // Test optimization suggestions for different scenarios
    config.performance_mode = PerformanceMode::Speed;
    config.quality = QualityLevel::Ultra;
    
    let suggestions = config.optimization_suggestions();
    assert!(suggestions.contains("quality") || suggestions.contains("speed"));
    
    config.performance_mode = PerformanceMode::Quality;
    config.quality = QualityLevel::Low;
    
    let suggestions = config.optimization_suggestions();
    assert!(suggestions.contains("quality") || suggestions.contains("performance"));
    
    config.performance_mode = PerformanceMode::Balanced;
    config.quality = QualityLevel::High;
    
    let suggestions = config.optimization_suggestions();
    assert!(suggestions.is_empty() || suggestions.contains("optimal"));
}

#[test]
fn test_config_presets() {
    // Test predefined presets
    let low_latency = VocodingConfig::low_latency_preset();
    assert_eq!(low_latency.performance_mode, PerformanceMode::Speed);
    assert_eq!(low_latency.quality, QualityLevel::Medium);
    
    let high_quality = VocodingConfig::high_quality_preset();
    assert_eq!(high_quality.performance_mode, PerformanceMode::Quality);
    assert_eq!(high_quality.quality, QualityLevel::Ultra);
    
    let balanced = VocodingConfig::balanced_preset();
    assert_eq!(balanced.performance_mode, PerformanceMode::Balanced);
    assert_eq!(balanced.quality, QualityLevel::High);
    
    // Test streaming presets
    let realtime = StreamingConfig::realtime_preset();
    assert_eq!(realtime.latency_mode, LatencyMode::UltraLow);
    assert!(realtime.enable_prediction);
    
    let quality_streaming = StreamingConfig::quality_preset();
    assert_eq!(quality_streaming.latency_mode, LatencyMode::Quality);
    assert!(!quality_streaming.enable_prediction);
}

#[test]
fn test_config_edge_cases() {
    let mut config = VocodingConfig::default();
    
    // Test edge cases for numeric values
    config.temperature = 0.001; // Very low
    assert!(config.validate().is_err());
    
    config.temperature = 3.999; // Just under limit
    assert!(config.validate().is_ok());
    
    config.guidance_scale = 0.0;
    assert!(config.validate().is_err());
    
    config.guidance_scale = 10.0; // At limit
    assert!(config.validate().is_ok());
    
    // Test boundary values
    config.sample_rate = 8000; // Minimum
    assert!(config.validate().is_ok());
    
    config.sample_rate = 96000; // Maximum
    assert!(config.validate().is_ok());
}