//! Integration tests for end-to-end spatial audio pipelines
//!
//! These tests validate complete workflows from audio input to binaural output.

use std::sync::Arc;
use voirs_spatial::{
    Position3D, SpatialConfig, SpatialConfigBuilder, SpatialEffect, SpatialProcessor,
    SpatialRequest,
};

/// Test basic spatial processing pipeline
#[tokio::test]
async fn test_basic_spatial_pipeline() {
    // Create configuration
    let config = SpatialConfig::default();

    // Create processor
    let mut processor = SpatialProcessor::new(config).await.unwrap();

    // Generate test audio
    let audio: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.01).sin()).collect();

    // Create request
    let request = SpatialRequest {
        id: "test_source".to_string(),
        audio: audio.clone(),
        source_position: Position3D::new(1.0, 0.0, 0.0),
        listener_position: Position3D::new(0.0, 0.0, 0.0),
        listener_orientation: (0.0, 0.0, 0.0),
        sample_rate: 48000,
        effects: vec![SpatialEffect::Hrtf, SpatialEffect::DistanceAttenuation],
        parameters: std::collections::HashMap::new(),
    };

    // Process
    let result = processor.process_request(request).await.unwrap();

    // Validate result
    assert!(result.success);
    assert_eq!(result.audio.left.len(), audio.len());
    assert_eq!(result.audio.right.len(), audio.len());
    assert_eq!(result.request_id, "test_source");
    assert_eq!(result.applied_effects.len(), 2);
}

/// Test spatial processing with multiple positions
#[tokio::test]
async fn test_multiple_positions() {
    let config = SpatialConfigBuilder::new()
        .sample_rate(48000)
        .buffer_size(512)
        .build()
        .unwrap();

    let mut processor = SpatialProcessor::new(config).await.unwrap();

    let audio: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.02).sin()).collect();

    let positions = vec![
        Position3D::new(0.0, 1.0, 0.0),  // Front
        Position3D::new(1.0, 0.0, 0.0),  // Right
        Position3D::new(-1.0, 0.0, 0.0), // Left
        Position3D::new(0.0, 0.0, 1.0),  // Above
    ];

    for (i, pos) in positions.iter().enumerate() {
        let request = SpatialRequest {
            id: format!("source_{}", i),
            audio: audio.clone(),
            source_position: *pos,
            listener_position: Position3D::new(0.0, 0.0, 0.0),
            listener_orientation: (0.0, 0.0, 0.0),
            sample_rate: 48000,
            effects: vec![SpatialEffect::Hrtf, SpatialEffect::DistanceAttenuation],
            parameters: std::collections::HashMap::new(),
        };

        let result = processor.process_request(request).await.unwrap();
        assert!(result.success);
        assert_eq!(result.audio.left.len(), 512);
    }
}

/// Test spatial processing with all effects
#[tokio::test]
async fn test_all_effects() {
    let config = SpatialConfig::default();
    let mut processor = SpatialProcessor::new(config).await.unwrap();

    let audio: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.01).sin() * 0.5).collect();

    let request = SpatialRequest {
        id: "all_effects_test".to_string(),
        audio: audio.clone(),
        source_position: Position3D::new(2.0, 1.0, 0.5),
        listener_position: Position3D::new(0.0, 0.0, 0.0),
        listener_orientation: (0.0, 0.0, 0.0),
        sample_rate: 48000,
        effects: vec![
            SpatialEffect::Hrtf,
            SpatialEffect::DistanceAttenuation,
            SpatialEffect::Reverb,
            SpatialEffect::Doppler,
            SpatialEffect::AirAbsorption,
        ],
        parameters: std::collections::HashMap::new(),
    };

    let result = processor.process_request(request).await.unwrap();

    assert!(result.success);
    assert_eq!(result.applied_effects.len(), 5);
    assert_eq!(result.audio.left.len(), audio.len());
    assert_eq!(result.audio.right.len(), audio.len());
}

/// Test processing with varying distances
#[tokio::test]
async fn test_distance_attenuation_pipeline() {
    let config = SpatialConfig::default();
    let mut processor = SpatialProcessor::new(config).await.unwrap();

    let audio: Vec<f32> = (0..500).map(|_| 0.5).collect();

    let distances = vec![0.5, 1.0, 2.0, 5.0, 10.0];
    let mut peak_amplitudes = Vec::new();

    for (i, &distance) in distances.iter().enumerate() {
        let request = SpatialRequest {
            id: format!("distance_{}", i),
            audio: audio.clone(),
            source_position: Position3D::new(distance, 0.0, 0.0),
            listener_position: Position3D::new(0.0, 0.0, 0.0),
            listener_orientation: (0.0, 0.0, 0.0),
            sample_rate: 48000,
            effects: vec![SpatialEffect::DistanceAttenuation],
            parameters: std::collections::HashMap::new(),
        };

        let result = processor.process_request(request).await.unwrap();

        // Calculate peak amplitude
        let peak = result
            .audio
            .left
            .iter()
            .map(|&x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        peak_amplitudes.push(peak);
    }

    // Verify that amplitude decreases with distance
    for i in 1..peak_amplitudes.len() {
        assert!(
            peak_amplitudes[i] <= peak_amplitudes[i - 1],
            "Amplitude should decrease with distance"
        );
    }
}

/// Test configuration validation
#[tokio::test]
async fn test_config_validation() {
    // Valid configuration should work
    let config = SpatialConfigBuilder::new()
        .sample_rate(48000)
        .buffer_size(1024)
        .build();

    assert!(config.is_ok());

    // Create processor with valid config
    let result = SpatialProcessor::new(config.unwrap()).await;
    assert!(result.is_ok());
}

/// Test request validation
#[tokio::test]
async fn test_request_validation() {
    let config = SpatialConfig::default();
    let mut processor = SpatialProcessor::new(config).await.unwrap();

    // Empty audio should be rejected
    let request = SpatialRequest {
        id: "empty_audio".to_string(),
        audio: vec![],
        source_position: Position3D::new(1.0, 0.0, 0.0),
        listener_position: Position3D::new(0.0, 0.0, 0.0),
        listener_orientation: (0.0, 0.0, 0.0),
        sample_rate: 48000,
        effects: vec![SpatialEffect::Hrtf],
        parameters: std::collections::HashMap::new(),
    };

    let result = processor.process_request(request).await;
    assert!(result.is_err(), "Empty audio should be rejected");
}

/// Test position updates
#[tokio::test]
async fn test_moving_source() {
    let config = SpatialConfig::default();
    let mut processor = SpatialProcessor::new(config).await.unwrap();

    let audio: Vec<f32> = (0..500).map(|i| ((i as f32) * 0.02).sin()).collect();

    // Simulate a source moving from left to right
    let positions = vec![
        Position3D::new(-2.0, 1.0, 0.0),
        Position3D::new(-1.0, 1.0, 0.0),
        Position3D::new(0.0, 1.0, 0.0),
        Position3D::new(1.0, 1.0, 0.0),
        Position3D::new(2.0, 1.0, 0.0),
    ];

    for (i, pos) in positions.iter().enumerate() {
        let request = SpatialRequest {
            id: "moving_source".to_string(),
            audio: audio.clone(),
            source_position: *pos,
            listener_position: Position3D::new(0.0, 0.0, 0.0),
            listener_orientation: (0.0, 0.0, 0.0),
            sample_rate: 48000,
            effects: vec![SpatialEffect::Hrtf, SpatialEffect::DistanceAttenuation],
            parameters: std::collections::HashMap::new(),
        };

        let result = processor.process_request(request).await.unwrap();
        assert!(result.success, "Frame {} should process successfully", i);
    }
}

/// Test stereo output generation
#[tokio::test]
async fn test_stereo_output() {
    let config = SpatialConfig::default();
    let mut processor = SpatialProcessor::new(config).await.unwrap();

    let audio: Vec<f32> = vec![0.5; 1000];

    // Test processing with different positions
    let positions = vec![
        Position3D::new(2.0, 0.0, 0.0),  // Right
        Position3D::new(-2.0, 0.0, 0.0), // Left
        Position3D::new(0.0, 2.0, 0.0),  // Front
    ];

    for (i, pos) in positions.iter().enumerate() {
        let request = SpatialRequest {
            id: format!("stereo_test_{}", i),
            audio: audio.clone(),
            source_position: *pos,
            listener_position: Position3D::new(0.0, 0.0, 0.0),
            listener_orientation: (0.0, 0.0, 0.0),
            sample_rate: 48000,
            effects: vec![SpatialEffect::Hrtf, SpatialEffect::DistanceAttenuation],
            parameters: std::collections::HashMap::new(),
        };

        let result = processor.process_request(request).await.unwrap();

        // Verify we get stereo output
        assert_eq!(
            result.audio.left.len(),
            1000,
            "Left channel should have correct length"
        );
        assert_eq!(
            result.audio.right.len(),
            1000,
            "Right channel should have correct length"
        );

        // Verify output is not completely silent
        let left_energy: f32 = result.audio.left.iter().map(|&x| x * x).sum();
        let right_energy: f32 = result.audio.right.iter().map(|&x| x * x).sum();
        let total_energy = left_energy + right_energy;

        assert!(
            total_energy > 0.0,
            "Stereo output should not be completely silent"
        );
    }
}

/// Test processing performance
#[tokio::test]
async fn test_processing_performance() {
    let config = SpatialConfig::default();
    let mut processor = SpatialProcessor::new(config).await.unwrap();

    let audio: Vec<f32> = (0..48000).map(|i| ((i as f32) * 0.001).sin()).collect();

    let request = SpatialRequest {
        id: "performance_test".to_string(),
        audio,
        source_position: Position3D::new(1.0, 1.0, 0.0),
        listener_position: Position3D::new(0.0, 0.0, 0.0),
        listener_orientation: (0.0, 0.0, 0.0),
        sample_rate: 48000,
        effects: vec![
            SpatialEffect::Hrtf,
            SpatialEffect::DistanceAttenuation,
            SpatialEffect::Reverb,
        ],
        parameters: std::collections::HashMap::new(),
    };

    let result = processor.process_request(request).await.unwrap();

    // Processing 1 second of audio should be reasonably fast
    assert!(
        result.processing_time.as_millis() < 1000,
        "Processing should complete in less than 1 second for 1 second of audio"
    );
}

/// Test concurrent processing
#[tokio::test]
async fn test_concurrent_requests() {
    let config = SpatialConfig::default();
    let processor = Arc::new(tokio::sync::RwLock::new(
        SpatialProcessor::new(config).await.unwrap(),
    ));

    let audio: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.01).sin()).collect();

    let mut handles = vec![];

    for i in 0..5 {
        let proc = Arc::clone(&processor);
        let audio_clone = audio.clone();

        let handle = tokio::spawn(async move {
            let request = SpatialRequest {
                id: format!("concurrent_{}", i),
                audio: audio_clone,
                source_position: Position3D::new((i as f32) - 2.0, 1.0, 0.0),
                listener_position: Position3D::new(0.0, 0.0, 0.0),
                listener_orientation: (0.0, 0.0, 0.0),
                sample_rate: 48000,
                effects: vec![SpatialEffect::Hrtf, SpatialEffect::DistanceAttenuation],
                parameters: std::collections::HashMap::new(),
            };

            let mut proc_write = proc.write().await;
            proc_write.process_request(request).await
        });

        handles.push(handle);
    }

    // Wait for all requests to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
        assert!(result.unwrap().success);
    }
}
