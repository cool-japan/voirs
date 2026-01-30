//! Comprehensive integration tests for model optimization features
//!
//! Tests for quantization, pruning, distillation, and hardware-specific optimizations.

use voirs_acoustic::{
    optimization::{
        DistillationConfig, DistillationMethod, HardwareOptimization, OptimizationConfig,
        OptimizationTargets, PruningConfig, PruningStrategy, PruningType, QuantizationConfig,
        QuantizationMethod, QuantizationPrecision, TargetDevice,
    },
    Result,
};

/// Test optimization configuration defaults
#[tokio::test]
async fn test_optimization_config_defaults() -> Result<()> {
    let config = OptimizationConfig::default();

    // Verify quantization defaults
    assert!(config.quantization.enabled);
    assert!(matches!(
        config.quantization.precision,
        QuantizationPrecision::Float16
    ));
    assert!(config.quantization.calibration_samples > 0);
    assert!(matches!(
        config.quantization.quantization_method,
        QuantizationMethod::PostTraining
    ));

    // Verify pruning defaults
    assert!(config.pruning.enabled);
    assert!(matches!(
        config.pruning.strategy,
        PruningStrategy::Magnitude
    ));
    assert!(config.pruning.target_sparsity >= 0.0 && config.pruning.target_sparsity <= 1.0);

    // Verify distillation defaults
    assert!(!config.distillation.enabled);
    assert!(matches!(
        config.distillation.method,
        DistillationMethod::Standard
    ));

    println!("✅ Optimization config defaults validated");
    Ok(())
}

/// Test quantization configuration options
#[tokio::test]
async fn test_quantization_config_options() -> Result<()> {
    // Test INT8 quantization
    let int8_config = QuantizationConfig {
        enabled: true,
        precision: QuantizationPrecision::Int8,
        calibration_samples: 1000,
        excluded_layers: vec!["attention".to_string(), "output".to_string()],
        quantization_method: QuantizationMethod::QuantizationAware,
        dynamic_quantization: false,
    };

    assert!(int8_config.enabled);
    assert!(matches!(int8_config.precision, QuantizationPrecision::Int8));
    assert_eq!(int8_config.calibration_samples, 1000);
    assert_eq!(int8_config.excluded_layers.len(), 2);
    assert!(matches!(
        int8_config.quantization_method,
        QuantizationMethod::QuantizationAware
    ));

    // Test Float16 quantization
    let fp16_config = QuantizationConfig {
        enabled: true,
        precision: QuantizationPrecision::Float16,
        calibration_samples: 500,
        excluded_layers: vec![],
        quantization_method: QuantizationMethod::PostTraining,
        dynamic_quantization: true,
    };

    assert!(matches!(
        fp16_config.precision,
        QuantizationPrecision::Float16
    ));
    assert!(fp16_config.dynamic_quantization);
    assert!(fp16_config.excluded_layers.is_empty());

    // Test Mixed precision
    let mixed_config = QuantizationConfig {
        enabled: true,
        precision: QuantizationPrecision::Mixed,
        calibration_samples: 2000,
        excluded_layers: vec!["encoder".to_string()],
        quantization_method: QuantizationMethod::Gradual,
        dynamic_quantization: false,
    };

    assert!(matches!(
        mixed_config.precision,
        QuantizationPrecision::Mixed
    ));
    assert!(matches!(
        mixed_config.quantization_method,
        QuantizationMethod::Gradual
    ));

    println!("✅ Quantization config options validated");
    Ok(())
}

/// Test pruning configuration options
#[tokio::test]
async fn test_pruning_config_options() -> Result<()> {
    // Test magnitude-based pruning
    let magnitude_config = PruningConfig {
        enabled: true,
        strategy: PruningStrategy::Magnitude,
        target_sparsity: 0.5,
        gradual_pruning: true,
        pruning_type: PruningType::Unstructured,
        excluded_layers: vec!["final_layer".to_string()],
    };

    assert!(magnitude_config.enabled);
    assert!(matches!(
        magnitude_config.strategy,
        PruningStrategy::Magnitude
    ));
    assert_eq!(magnitude_config.target_sparsity, 0.5);
    assert!(magnitude_config.gradual_pruning);
    assert!(matches!(
        magnitude_config.pruning_type,
        PruningType::Unstructured
    ));

    // Test gradient-based pruning
    let gradient_config = PruningConfig {
        enabled: true,
        strategy: PruningStrategy::Gradient,
        target_sparsity: 0.3,
        gradual_pruning: false,
        pruning_type: PruningType::Structured,
        excluded_layers: vec![],
    };

    assert!(matches!(
        gradient_config.strategy,
        PruningStrategy::Gradient
    ));
    assert_eq!(gradient_config.target_sparsity, 0.3);
    assert!(!gradient_config.gradual_pruning);
    assert!(matches!(
        gradient_config.pruning_type,
        PruningType::Structured
    ));

    // Test Fisher information pruning
    let fisher_config = PruningConfig {
        enabled: true,
        strategy: PruningStrategy::Fisher,
        target_sparsity: 0.7,
        gradual_pruning: true,
        pruning_type: PruningType::Unstructured,
        excluded_layers: vec!["attention".to_string(), "norm".to_string()],
    };

    assert!(matches!(fisher_config.strategy, PruningStrategy::Fisher));
    assert_eq!(fisher_config.target_sparsity, 0.7);
    assert_eq!(fisher_config.excluded_layers.len(), 2);

    println!("✅ Pruning config options validated");
    Ok(())
}

/// Test distillation configuration options
#[tokio::test]
async fn test_distillation_config_options() -> Result<()> {
    use voirs_acoustic::optimization::StudentModelConfig;

    let distillation_config = DistillationConfig {
        enabled: true,
        method: DistillationMethod::Standard,
        teacher_model_path: Some("teacher_model.safetensors".to_string()),
        student_config: StudentModelConfig {
            hidden_reduction_factor: 0.5,
            layer_reduction_factor: 0.5,
            num_heads: 4,
            shared_parameters: false,
        },
        temperature: 4.0,
        distillation_weight: 0.7,
    };

    assert!(distillation_config.enabled);
    assert!(matches!(
        distillation_config.method,
        DistillationMethod::Standard
    ));
    assert_eq!(
        distillation_config.teacher_model_path,
        Some("teacher_model.safetensors".to_string())
    );
    assert_eq!(distillation_config.temperature, 4.0);
    assert_eq!(distillation_config.distillation_weight, 0.7);

    // Test feature-based distillation
    let feature_distillation = DistillationConfig {
        enabled: true,
        method: DistillationMethod::FeatureBased,
        teacher_model_path: Some("teacher.safetensors".to_string()),
        student_config: StudentModelConfig {
            hidden_reduction_factor: 0.75,
            layer_reduction_factor: 0.5,
            num_heads: 6,
            shared_parameters: true,
        },
        temperature: 3.0,
        distillation_weight: 0.5,
    };

    assert!(matches!(
        feature_distillation.method,
        DistillationMethod::FeatureBased
    ));

    println!("✅ Distillation config options validated");
    Ok(())
}

/// Test hardware optimization options
#[tokio::test]
async fn test_hardware_optimization_options() -> Result<()> {
    // Test mobile optimization
    let mobile_optimization = HardwareOptimization {
        target_device: TargetDevice::Mobile,
        enable_simd: true,
        enable_gpu: false,
        memory_limit_mb: Some(512),
        cpu_cores: Some(4),
    };

    assert!(matches!(
        mobile_optimization.target_device,
        TargetDevice::Mobile
    ));
    assert!(mobile_optimization.enable_simd);
    assert!(!mobile_optimization.enable_gpu);
    assert_eq!(mobile_optimization.memory_limit_mb, Some(512));
    assert_eq!(mobile_optimization.cpu_cores, Some(4));

    // Test desktop optimization
    let desktop_optimization = HardwareOptimization {
        target_device: TargetDevice::Desktop,
        enable_simd: true,
        enable_gpu: true,
        memory_limit_mb: Some(8192),
        cpu_cores: Some(8),
    };

    assert!(matches!(
        desktop_optimization.target_device,
        TargetDevice::Desktop
    ));
    assert!(desktop_optimization.enable_gpu);
    assert_eq!(desktop_optimization.memory_limit_mb, Some(8192));

    // Test server optimization
    let server_optimization = HardwareOptimization {
        target_device: TargetDevice::Server,
        enable_simd: true,
        enable_gpu: true,
        memory_limit_mb: None,
        cpu_cores: Some(16),
    };

    assert!(matches!(
        server_optimization.target_device,
        TargetDevice::Server
    ));
    assert!(server_optimization.memory_limit_mb.is_none());
    assert_eq!(server_optimization.cpu_cores, Some(16));

    println!("✅ Hardware optimization options validated");
    Ok(())
}

/// Test optimization targets configuration
#[tokio::test]
async fn test_optimization_targets() -> Result<()> {
    let targets = OptimizationTargets {
        max_quality_loss: 0.05,        // 5% max quality loss
        memory_reduction_target: 0.5,  // 50% memory reduction
        speed_improvement_target: 2.0, // 2x speedup
        max_model_size_mb: Some(256),
        target_latency_ms: Some(100.0),
    };

    assert_eq!(targets.max_quality_loss, 0.05);
    assert_eq!(targets.memory_reduction_target, 0.5);
    assert_eq!(targets.speed_improvement_target, 2.0);
    assert_eq!(targets.max_model_size_mb, Some(256));
    assert_eq!(targets.target_latency_ms, Some(100.0));

    // Verify constraints are reasonable
    assert!(targets.max_quality_loss >= 0.0 && targets.max_quality_loss <= 1.0);
    assert!(targets.memory_reduction_target >= 0.0);
    assert!(targets.speed_improvement_target >= 1.0);

    println!("✅ Optimization targets validated");
    Ok(())
}

/// Test quantization precision variants
#[tokio::test]
async fn test_quantization_precision_variants() -> Result<()> {
    let precisions = vec![
        QuantizationPrecision::Int8,
        QuantizationPrecision::Float16,
        QuantizationPrecision::Mixed,
        QuantizationPrecision::Dynamic,
    ];

    for precision in precisions {
        let config = QuantizationConfig {
            enabled: true,
            precision: precision.clone(),
            calibration_samples: 1000,
            excluded_layers: vec![],
            quantization_method: QuantizationMethod::PostTraining,
            dynamic_quantization: false,
        };

        // Verify precision is set correctly
        match precision {
            QuantizationPrecision::Int8 => {
                assert!(matches!(config.precision, QuantizationPrecision::Int8));
            }
            QuantizationPrecision::Float16 => {
                assert!(matches!(config.precision, QuantizationPrecision::Float16));
            }
            QuantizationPrecision::Mixed => {
                assert!(matches!(config.precision, QuantizationPrecision::Mixed));
            }
            QuantizationPrecision::Dynamic => {
                assert!(matches!(config.precision, QuantizationPrecision::Dynamic));
            }
        }
    }

    println!("✅ Quantization precision variants validated");
    Ok(())
}

/// Test pruning strategy variants
#[tokio::test]
async fn test_pruning_strategy_variants() -> Result<()> {
    let strategies = vec![
        PruningStrategy::Magnitude,
        PruningStrategy::Gradient,
        PruningStrategy::Fisher,
        PruningStrategy::Adaptive,
    ];

    for strategy in strategies {
        let config = PruningConfig {
            enabled: true,
            strategy: strategy.clone(),
            target_sparsity: 0.5,
            gradual_pruning: false,
            pruning_type: PruningType::Unstructured,
            excluded_layers: vec![],
        };

        // Verify strategy is set correctly
        match strategy {
            PruningStrategy::Magnitude => {
                assert!(matches!(config.strategy, PruningStrategy::Magnitude));
            }
            PruningStrategy::Gradient => {
                assert!(matches!(config.strategy, PruningStrategy::Gradient));
            }
            PruningStrategy::Fisher => {
                assert!(matches!(config.strategy, PruningStrategy::Fisher));
            }
            PruningStrategy::Adaptive => {
                assert!(matches!(config.strategy, PruningStrategy::Adaptive));
            }
        }
    }

    println!("✅ Pruning strategy variants validated");
    Ok(())
}

/// Test optimization configuration validation
#[tokio::test]
async fn test_optimization_config_validation() -> Result<()> {
    use voirs_acoustic::optimization::StudentModelConfig;

    let valid_config = OptimizationConfig {
        quantization: QuantizationConfig {
            enabled: true,
            precision: QuantizationPrecision::Int8,
            calibration_samples: 1000,
            excluded_layers: vec!["sensitive_layer".to_string()],
            quantization_method: QuantizationMethod::PostTraining,
            dynamic_quantization: false,
        },
        pruning: PruningConfig {
            enabled: true,
            strategy: PruningStrategy::Magnitude,
            target_sparsity: 0.3,
            gradual_pruning: true,
            pruning_type: PruningType::Unstructured,
            excluded_layers: vec![],
        },
        distillation: DistillationConfig {
            enabled: false,
            method: DistillationMethod::Standard,
            teacher_model_path: None,
            student_config: StudentModelConfig {
                hidden_reduction_factor: 0.5,
                layer_reduction_factor: 0.5,
                num_heads: 4,
                shared_parameters: false,
            },
            temperature: 3.0,
            distillation_weight: 0.7,
        },
        hardware_optimization: HardwareOptimization {
            target_device: TargetDevice::Desktop,
            enable_simd: true,
            enable_gpu: true,
            memory_limit_mb: Some(2048),
            cpu_cores: Some(8),
        },
        optimization_targets: OptimizationTargets {
            max_quality_loss: 0.05,
            memory_reduction_target: 0.5,
            speed_improvement_target: 1.5,
            max_model_size_mb: Some(128),
            target_latency_ms: Some(50.0),
        },
    };

    // Verify configuration constraints
    assert!(
        valid_config.pruning.target_sparsity >= 0.0 && valid_config.pruning.target_sparsity <= 1.0
    );
    assert!(valid_config.quantization.calibration_samples > 0);
    assert!(valid_config.distillation.temperature > 0.0);
    assert!(
        valid_config.distillation.distillation_weight >= 0.0
            && valid_config.distillation.distillation_weight <= 1.0
    );

    if let Some(latency) = valid_config.optimization_targets.target_latency_ms {
        assert!(latency > 0.0);
    }

    if let Some(memory) = valid_config.hardware_optimization.memory_limit_mb {
        assert!(memory > 0);
    }

    println!("✅ Optimization config validation passed");
    Ok(())
}

/// Test target device enumeration
#[tokio::test]
async fn test_target_device_enumeration() -> Result<()> {
    let devices = vec![
        TargetDevice::Mobile,
        TargetDevice::Desktop,
        TargetDevice::Server,
        TargetDevice::Edge,
    ];

    for device in devices {
        let optimization = HardwareOptimization {
            target_device: device.clone(),
            enable_simd: true,
            enable_gpu: false,
            memory_limit_mb: Some(1024),
            cpu_cores: Some(4),
        };

        // Verify device is set correctly
        match device {
            TargetDevice::Mobile => {
                assert!(matches!(optimization.target_device, TargetDevice::Mobile));
            }
            TargetDevice::Desktop => {
                assert!(matches!(optimization.target_device, TargetDevice::Desktop));
            }
            TargetDevice::Server => {
                assert!(matches!(optimization.target_device, TargetDevice::Server));
            }
            TargetDevice::Edge => {
                assert!(matches!(optimization.target_device, TargetDevice::Edge));
            }
        }
    }

    println!("✅ Target device enumeration validated");
    Ok(())
}

/// Test distillation method enumeration
#[tokio::test]
async fn test_distillation_method_enumeration() -> Result<()> {
    use voirs_acoustic::optimization::StudentModelConfig;

    let methods = vec![
        DistillationMethod::Standard,
        DistillationMethod::FeatureBased,
        DistillationMethod::AttentionBased,
        DistillationMethod::Progressive,
    ];

    for method in methods {
        let config = DistillationConfig {
            enabled: true,
            method: method.clone(),
            teacher_model_path: Some("teacher.model".to_string()),
            student_config: StudentModelConfig {
                hidden_reduction_factor: 0.5,
                layer_reduction_factor: 0.5,
                num_heads: 4,
                shared_parameters: false,
            },
            temperature: 3.0,
            distillation_weight: 0.5,
        };

        // Verify method is set correctly
        match method {
            DistillationMethod::Standard => {
                assert!(matches!(config.method, DistillationMethod::Standard));
            }
            DistillationMethod::FeatureBased => {
                assert!(matches!(config.method, DistillationMethod::FeatureBased));
            }
            DistillationMethod::AttentionBased => {
                assert!(matches!(config.method, DistillationMethod::AttentionBased));
            }
            DistillationMethod::Progressive => {
                assert!(matches!(config.method, DistillationMethod::Progressive));
            }
        }
    }

    println!("✅ Distillation method enumeration validated");
    Ok(())
}

/// Test optimization configuration serialization/deserialization
#[tokio::test]
async fn test_optimization_config_serialization() -> Result<()> {
    let mut config = OptimizationConfig::default();

    // Customize some values for testing
    config.quantization.precision = QuantizationPrecision::Mixed;
    config.quantization.calibration_samples = 2000;
    config.quantization.excluded_layers = vec!["layer1".to_string(), "layer2".to_string()];
    config.quantization.quantization_method = QuantizationMethod::QuantizationAware;
    config.quantization.dynamic_quantization = true;

    config.pruning.enabled = false;
    config.pruning.strategy = PruningStrategy::Fisher;
    config.pruning.target_sparsity = 0.6;
    config.pruning.pruning_type = PruningType::Structured;
    config.pruning.excluded_layers = vec!["critical_layer".to_string()];

    // Test JSON serialization
    let serialized = serde_json::to_string(&config).expect("Should serialize");
    assert!(!serialized.is_empty());

    // Test JSON deserialization
    let deserialized: OptimizationConfig =
        serde_json::from_str(&serialized).expect("Should deserialize");

    // Verify deserialized config matches original
    assert_eq!(
        config.quantization.enabled,
        deserialized.quantization.enabled
    );
    assert_eq!(
        config.quantization.calibration_samples,
        deserialized.quantization.calibration_samples
    );
    assert_eq!(
        config.quantization.excluded_layers,
        deserialized.quantization.excluded_layers
    );
    assert_eq!(config.pruning.enabled, deserialized.pruning.enabled);
    assert_eq!(
        config.pruning.target_sparsity,
        deserialized.pruning.target_sparsity
    );

    println!("✅ Optimization config serialization validated");
    Ok(())
}
