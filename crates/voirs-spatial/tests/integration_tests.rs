//! Integration tests for VoiRS Spatial Audio
//!
//! These tests verify that different modules work together correctly
//! in realistic usage scenarios.

use scirs2_core::ndarray::Array1;
use std::sync::Arc;
use voirs_spatial::{
    AmbisonicsDecoder, AmbisonicsEncoder, BinauralConfig, BinauralRenderer, ChannelOrdering,
    HeadTracker, HrtfDatabase, NormalizationScheme, Position3D, SoundSource, SpeakerConfiguration,
};

/// Test that HRTF database can be loaded and used with binaural renderer
#[tokio::test]
async fn test_hrtf_binaural_integration() {
    let hrtf_db = HrtfDatabase::load_default()
        .await
        .expect("Failed to load HRTF database");

    let config = BinauralConfig {
        sample_rate: 48000,
        buffer_size: 512,
        hrir_length: 200,
        max_sources: 8,
        use_gpu: false,
        crossfade_duration: 0.05,
        quality_level: 0.8,
        enable_distance_modeling: true,
        enable_air_absorption: true,
        near_field_distance: 0.2,
        far_field_distance: 10.0,
        optimize_for_latency: false,
    };

    let _renderer = BinauralRenderer::new(config, Arc::new(hrtf_db))
        .await
        .expect("Failed to create binaural renderer");

    // Renderer created successfully
}

/// Test that head tracking integrates correctly with position tracking
#[test]
fn test_head_tracking_integration() {
    // Create head tracker
    let mut head_tracker = HeadTracker::new();
    head_tracker.update_position_with_time(Position3D::new(0.0, 1.7, 0.0), 0.0);
    head_tracker.update_orientation_with_time((0.0, 0.0, 0.0), 0.0);

    // Update head orientation
    head_tracker.update_orientation_with_time((0.785, 0.0, 0.0), 0.1); // 45 degrees

    let (yaw, pitch, roll) = head_tracker
        .current_orientation()
        .unwrap_or((0.0, 0.0, 0.0));

    // Verify orientation was updated
    assert!((yaw - 0.785).abs() < 0.001);
    assert!((pitch - 0.0).abs() < 0.001);
    assert!((roll - 0.0).abs() < 0.001);
}

/// Test ambisonics encoding and decoding pipeline
#[test]
fn test_ambisonics_pipeline() {
    // Create 2nd-order encoder
    let encoder = AmbisonicsEncoder::new(2, NormalizationScheme::N3D, ChannelOrdering::ACN);

    // Create decoder for stereo output
    let decoder = AmbisonicsDecoder::for_speaker_config(2, SpeakerConfiguration::Stereo)
        .expect("Failed to create decoder");

    // Create test audio
    let audio = Array1::from_vec(vec![1.0f32; 1000]);
    let position = Position3D::new(1.0, 0.0, 0.0);

    // Encode
    let encoded = encoder
        .encode_mono(&audio, &position)
        .expect("Failed to encode");

    // Verify channel count (2nd order = 9 channels)
    assert_eq!(encoded.shape()[0], 9);
    assert_eq!(encoded.shape()[1], 1000);

    // Decode
    let stereo = decoder.decode(&encoded).expect("Failed to decode");

    // Verify stereo output
    assert_eq!(stereo.shape()[0], 2);
    assert_eq!(stereo.shape()[1], 1000);
}

/// Test position tracking with sound sources
#[test]
fn test_position_tracking_integration() {
    // Test source and listener positions
    let source = SoundSource::new_point("source1".to_string(), Position3D::new(2.0, 1.5, 3.0));
    let listener = Position3D::new(5.0, 1.7, 4.0);

    let distance = source.position().distance_to(&listener);
    assert!(distance > 0.0);

    // Verify position is correct
    assert!((source.position().x - 2.0).abs() < 0.001);
    assert!((source.position().y - 1.5).abs() < 0.001);
    assert!((source.position().z - 3.0).abs() < 0.001);
}

/// Test multiple sound sources
#[test]
fn test_multiple_sources() {
    // Create different sound sources
    let source1 = SoundSource::new_point("static1".to_string(), Position3D::new(2.0, 0.0, 0.0));
    let source2 = SoundSource::new_point("static2".to_string(), Position3D::new(-2.0, 0.0, 0.0));
    let source3 = SoundSource::new_point("static3".to_string(), Position3D::new(0.0, 2.0, 0.0));

    // Verify each source has correct position
    assert!((source1.position().x - 2.0).abs() < 0.001);
    assert!((source2.position().x - (-2.0)).abs() < 0.001);
    assert!((source3.position().y - 2.0).abs() < 0.001);

    // Test that all sources can be created successfully
    assert_eq!(source1.id, "static1");
    assert_eq!(source2.id, "static2");
    assert_eq!(source3.id, "static3");
}

/// Test position prediction with velocity
#[test]
fn test_position_prediction() {
    let mut tracker = HeadTracker::new();

    // Add position history with movement
    tracker.update_position_with_time(Position3D::new(0.0, 0.0, 0.0), 0.0);
    tracker.update_position_with_time(Position3D::new(1.0, 0.0, 0.0), 0.1);
    tracker.update_position_with_time(Position3D::new(2.0, 0.0, 0.0), 0.2);

    // Predict future position
    let predicted = tracker.predict_position(std::time::Duration::from_millis(100));

    assert!(predicted.is_some());
    let pred_pos = predicted.unwrap();

    // Should be ahead of current position
    assert!(pred_pos.x > 2.0);
}

/// Test SIMD batch operations
#[test]
fn test_simd_batch_operations() {
    use voirs_spatial::SIMDSpatialOps;

    // Create test positions
    let positions: Vec<Position3D> = (0..1000)
        .map(|i| {
            let angle = (i as f32) * 0.01;
            Position3D::new(angle.cos() * 5.0, angle.sin() * 5.0, 1.0)
        })
        .collect();

    let listener = Position3D::new(0.0, 0.0, 0.0);

    // Calculate distances using SIMD
    let distances = SIMDSpatialOps::distances(listener, &positions);

    assert_eq!(distances.len(), 1000);

    // Verify distances are reasonable
    for &dist in &distances {
        assert!(dist > 0.0 && dist < 10.0);
    }
}

/// Test memory management configuration
#[test]
fn test_memory_configuration() {
    use voirs_spatial::{CachePolicy, MemoryConfig};

    let memory_config = MemoryConfig {
        max_buffer_pool_size: 100,
        max_cache_size: 50,
        enable_monitoring: true,
        memory_pressure_threshold: 0.8,
        cache_policy: CachePolicy::LRU,
        buffer_alignment: 64,
    };

    assert_eq!(memory_config.max_buffer_pool_size, 100);
    assert_eq!(memory_config.max_cache_size, 50);
    assert!(memory_config.enable_monitoring);
    assert!((memory_config.memory_pressure_threshold - 0.8).abs() < 0.001);
}
