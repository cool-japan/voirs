//! Tests for position tracking and spatial audio systems

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use super::super::occlusion::{OcclusionDetector, OcclusionMaterial};
    use super::super::prediction::{MotionPredictor, MotionSnapshot};
    use super::super::source_manager::{DopplerProcessor, DynamicSource, DynamicSourceManager};
    use super::super::spatial_grid::SpatialGrid;
    use super::super::tracking::{HeadTracker, ListenerMovementSystem, PlatformIntegration};
    use super::super::types::*;
    use crate::types::Position3D;
    use std::collections::VecDeque;
    use std::time::Duration;

    #[test]
    fn test_listener_creation() {
        let listener = Listener::new();
        assert_eq!(listener.position(), Position3D::default());
        assert_eq!(listener.orientation(), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_listener_position_update() {
        let mut listener = Listener::new();
        let new_pos = Position3D::new(1.0, 2.0, 3.0);
        listener.set_position(new_pos);
        assert_eq!(listener.position(), new_pos);
    }

    #[test]
    fn test_sound_source_creation() {
        let source = SoundSource::new_point("test".to_string(), Position3D::new(5.0, 0.0, 0.0));
        assert_eq!(source.position(), Position3D::new(5.0, 0.0, 0.0));
        assert_eq!(source.source_type(), SourceType::Point);
        assert!(source.is_active());
    }

    #[test]
    fn test_ear_positions() {
        let mut listener = Listener::new();
        listener.set_interaural_distance(0.2);

        let left_ear = listener.left_ear_position();
        let right_ear = listener.right_ear_position();

        // With zero orientation, ears should be on x-axis
        assert_eq!(left_ear.x, -0.1);
        assert_eq!(right_ear.x, 0.1);
    }

    #[test]
    fn test_directivity_patterns() {
        let omni = DirectivityPattern::omnidirectional();
        assert_eq!(omni.front_gain, 1.0);
        assert_eq!(omni.back_gain, 1.0);

        let cardioid = DirectivityPattern::cardioid();
        assert_eq!(cardioid.front_gain, 1.0);
        assert_eq!(cardioid.back_gain, 0.0);
    }

    #[test]
    fn test_position_prediction() {
        let mut source = SoundSource::new_point("test".to_string(), Position3D::default());

        // Simulate movement
        source.set_position(Position3D::new(1.0, 0.0, 0.0));
        std::thread::sleep(Duration::from_millis(10));
        source.set_position(Position3D::new(2.0, 0.0, 0.0));

        // Should have positive velocity in x direction
        assert!(source.velocity().x > 0.0);

        // Predicted position should be further along x axis
        let predicted = source.predict_position(Duration::from_secs(1));
        assert!(predicted.x > 2.0);
    }

    #[test]
    fn test_head_tracker() {
        let mut tracker = HeadTracker::new();

        // Add position updates with explicit timestamps for predictable testing
        tracker.update_position_with_time(Position3D::new(0.0, 0.0, 0.0), 0.0);
        tracker.update_position_with_time(Position3D::new(1.0, 0.0, 0.0), 0.1); // 0.1s later
        tracker.update_position_with_time(Position3D::new(2.0, 0.0, 0.0), 0.2); // 0.2s later

        // Test prediction
        let predicted = tracker.predict_position(Duration::from_millis(100));
        assert!(predicted.is_some());

        let predicted_pos = predicted.unwrap();
        assert!(predicted_pos.x > 2.0); // Should be ahead of current position
    }

    #[test]
    fn test_head_tracker_orientation() {
        let mut tracker = HeadTracker::new();

        // Add orientation updates with explicit timestamps
        tracker.update_orientation_with_time((0.0, 0.0, 0.0), 0.0);
        tracker.update_orientation_with_time((0.1, 0.0, 0.0), 0.1); // 0.1s later
        tracker.update_orientation_with_time((0.2, 0.0, 0.0), 0.2); // 0.2s later

        // Test orientation prediction
        let predicted = tracker.predict_orientation(Duration::from_millis(100));
        assert!(predicted.is_some());

        let predicted_orient = predicted.unwrap();
        assert!(predicted_orient.0 > 0.2); // Should be ahead of current orientation
    }

    #[test]
    fn test_spatial_source_manager() {
        use super::super::source_manager::SpatialSourceManager;
        let bounds = (
            Position3D::new(-10.0, -10.0, -10.0),
            Position3D::new(10.0, 10.0, 10.0),
        );
        let mut manager = SpatialSourceManager::new(bounds, 2.0);

        // Add sources
        let source1 = SoundSource::new_point("source1".to_string(), Position3D::new(1.0, 0.0, 0.0));
        let source2 = SoundSource::new_point("source2".to_string(), Position3D::new(5.0, 0.0, 0.0));

        assert!(manager.add_source(source1).is_ok());
        assert!(manager.add_source(source2).is_ok());

        // Test nearby sources query
        let listener_pos = Position3D::new(0.0, 0.0, 0.0);
        let nearby = manager.get_nearby_sources(listener_pos, 3.0);
        assert!(!nearby.is_empty());

        // Test source removal
        let removed = manager.remove_source("source1");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, "source1");
    }

    #[test]
    fn test_spatial_grid() {
        let bounds = (
            Position3D::new(-10.0, -10.0, -10.0),
            Position3D::new(10.0, 10.0, 10.0),
        );
        let mut grid = SpatialGrid::new(bounds, 1.0); // Smaller cell size

        // Add sources to grid with more separation
        grid.add_source("source1", Position3D::new(1.0, 1.0, 1.0));
        grid.add_source("source2", Position3D::new(8.0, 8.0, 8.0));

        // Query nearby sources with smaller radius
        let nearby = grid.query_sphere(Position3D::new(0.0, 0.0, 0.0), 2.5);
        assert!(nearby.contains(&"source1".to_string()));
        assert!(!nearby.contains(&"source2".to_string()));

        // Test source movement
        grid.move_source(
            "source1",
            Position3D::new(1.0, 1.0, 1.0),
            Position3D::new(9.0, 9.0, 9.0),
        );

        let nearby_after_move = grid.query_sphere(Position3D::new(0.0, 0.0, 0.0), 2.5);
        assert!(!nearby_after_move.contains(&"source1".to_string()));
    }

    #[test]
    fn test_occlusion_detector() {
        let mut detector = OcclusionDetector::new();

        // Add obstacle
        let obstacle = Box3D {
            min: Position3D::new(-1.0, -1.0, -1.0),
            max: Position3D::new(1.0, 1.0, 1.0),
            material_id: "wall".to_string(),
        };
        detector.add_obstacle(obstacle);

        // Add material
        let material = OcclusionMaterial {
            name: "wall".to_string(),
            transmission: 0.2,
            high_freq_absorption: 0.8,
            low_freq_absorption: 0.4,
            scattering: 0.3,
        };
        detector.add_material(material);

        // Test occlusion (line passes through obstacle)
        let source = Position3D::new(-5.0, 0.0, 0.0);
        let listener = Position3D::new(5.0, 0.0, 0.0);
        let result = detector.check_occlusion(source, listener);

        assert!(result.is_occluded);
        assert_eq!(result.transmission_factor, 0.2);

        // Test no occlusion (line doesn't pass through obstacle)
        let source_clear = Position3D::new(-5.0, 5.0, 0.0);
        let listener_clear = Position3D::new(5.0, 5.0, 0.0);
        let result_clear = detector.check_occlusion(source_clear, listener_clear);

        assert!(!result_clear.is_occluded);
        assert_eq!(result_clear.transmission_factor, 1.0);
    }

    #[test]
    fn test_box3d_intersection() {
        let detector = OcclusionDetector::new();

        let box3d = Box3D {
            min: Position3D::new(-1.0, -1.0, -1.0),
            max: Position3D::new(1.0, 1.0, 1.0),
            material_id: "test".to_string(),
        };

        // Line passes through box
        let start = Position3D::new(-2.0, 0.0, 0.0);
        let end = Position3D::new(2.0, 0.0, 0.0);
        assert!(detector.line_intersects_box(start, end, &box3d));

        // Line doesn't intersect box
        let start_miss = Position3D::new(-2.0, 2.0, 0.0);
        let end_miss = Position3D::new(2.0, 2.0, 0.0);
        assert!(!detector.line_intersects_box(start_miss, end_miss, &box3d));
    }

    #[test]
    fn test_angle_difference() {
        let tracker = HeadTracker::new();

        // Test normal angle difference
        let diff1 = tracker.angle_difference(0.5, 0.0);
        assert!((diff1 - 0.5).abs() < 0.001);

        // Test wrap-around with PI values
        let diff2 = tracker.angle_difference(0.1, 2.0 * std::f32::consts::PI - 0.1);
        assert!(diff2.abs() < 0.5); // Should be small after wrap-around

        // Test wrap-around (crossing zero)
        let diff3 = tracker.angle_difference(-0.1, 0.1);
        assert!((diff3 + 0.2).abs() < 0.001);

        // Test wrap-around near PI
        let diff4 =
            tracker.angle_difference(std::f32::consts::PI - 0.1, -std::f32::consts::PI + 0.1);
        assert!(diff4.abs() < 0.5); // Should wrap around to small difference
    }

    #[test]
    fn test_source_manager_culling() {
        use super::super::source_manager::SpatialSourceManager;
        let bounds = (
            Position3D::new(-50.0, -50.0, -50.0),
            Position3D::new(50.0, 50.0, 50.0),
        );
        let mut manager = SpatialSourceManager::new(bounds, 5.0);

        // Set a small culling distance for testing
        manager.culling_distance = 10.0;

        // Add sources at different distances
        let near_source =
            SoundSource::new_point("near".to_string(), Position3D::new(2.0, 0.0, 0.0));
        let far_source = SoundSource::new_point("far".to_string(), Position3D::new(20.0, 0.0, 0.0));

        manager.add_source(near_source).unwrap();
        manager.add_source(far_source).unwrap();

        assert_eq!(manager.sources.len(), 2);

        // Cull distant sources
        let listener_pos = Position3D::new(0.0, 0.0, 0.0);
        manager.cull_distant_sources(listener_pos);

        // Far source should be culled
        assert_eq!(manager.sources.len(), 1);
        assert!(manager.sources.contains_key("near"));
        assert!(!manager.sources.contains_key("far"));
    }

    #[test]
    fn test_doppler_processor_creation() {
        let doppler = DopplerProcessor::new(44100.0);
        assert_eq!(doppler.speed_of_sound(), 343.0);
    }

    #[test]
    fn test_doppler_factor_calculation() {
        let doppler = DopplerProcessor::new(44100.0);

        // Stationary source and listener
        let source_pos = Position3D::new(0.0, 0.0, 0.0);
        let source_vel = Position3D::new(0.0, 0.0, 0.0);
        let listener_pos = Position3D::new(10.0, 0.0, 0.0);
        let listener_vel = Position3D::new(0.0, 0.0, 0.0);

        let factor =
            doppler.calculate_doppler_factor(source_pos, source_vel, listener_pos, listener_vel);
        assert!((factor - 1.0).abs() < 0.001); // Should be no Doppler effect

        // Source approaching listener
        let approaching_vel = Position3D::new(10.0, 0.0, 0.0); // 10 m/s towards listener
        let factor_approaching = doppler.calculate_doppler_factor(
            source_pos,
            approaching_vel,
            listener_pos,
            listener_vel,
        );
        assert!(factor_approaching > 1.0); // Higher frequency when approaching

        // Source receding from listener
        let receding_vel = Position3D::new(-10.0, 0.0, 0.0); // 10 m/s away from listener
        let factor_receding =
            doppler.calculate_doppler_factor(source_pos, receding_vel, listener_pos, listener_vel);
        assert!(factor_receding < 1.0); // Lower frequency when receding
    }

    #[test]
    fn test_dynamic_source_manager() {
        let mut manager = DynamicSourceManager::new(44100.0);

        // Create a sound source
        let source =
            SoundSource::new_point("test_source".to_string(), Position3D::new(0.0, 0.0, 0.0));

        // Add to manager
        assert!(manager.add_source(source).is_ok());
        assert_eq!(manager.sources().len(), 1);

        // Update motion
        let position = Position3D::new(1.0, 0.0, 0.0);
        let velocity = Position3D::new(5.0, 0.0, 0.0);
        let acceleration = Position3D::new(0.0, 0.0, 0.0);

        assert!(manager
            .update_source_motion("test_source", position, velocity, acceleration)
            .is_ok());

        // Check motion was updated
        let source = manager.get_source("test_source").unwrap();
        assert_eq!(source.velocity, velocity);
    }

    #[tokio::test]
    async fn test_dynamic_source_processing() {
        let mut manager = DynamicSourceManager::new(44100.0);

        // Add a moving source
        let source =
            SoundSource::new_point("moving_source".to_string(), Position3D::new(0.0, 0.0, 0.0));
        manager.add_source(source).unwrap();

        // Update with motion
        manager
            .update_source_motion(
                "moving_source",
                Position3D::new(1.0, 0.0, 0.0),
                Position3D::new(10.0, 0.0, 0.0), // Moving at 10 m/s
                Position3D::new(0.0, 0.0, 0.0),
            )
            .unwrap();

        // Process with listener
        let listener_pos = Position3D::new(10.0, 0.0, 0.0);
        let listener_vel = Position3D::new(0.0, 0.0, 0.0);

        assert!(manager
            .process_dynamic_sources(listener_pos, listener_vel)
            .await
            .is_ok());

        // Check Doppler factor was calculated
        let source = manager.get_source("moving_source").unwrap();
        assert!(source.doppler_factor > 1.0); // Should be higher frequency (approaching)
    }

    #[test]
    fn test_doppler_audio_processing() {
        let doppler = DopplerProcessor::new(44100.0);

        // Create test audio (sine wave)
        let input: Vec<f32> = (0..1000)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI / 44.1).sin())
            .collect();
        let mut output = vec![0.0; 1000];

        // Apply no Doppler effect
        assert!(doppler
            .process_doppler_effect(&input, &mut output, 1.0)
            .is_ok());

        // Should be identical
        for i in 0..input.len() {
            assert!((input[i] - output[i]).abs() < 0.001);
        }

        // Apply Doppler effect
        assert!(doppler
            .process_doppler_effect(&input, &mut output, 1.1)
            .is_ok());

        // Output should be different (higher frequency)
        let mut differences = 0;
        for i in 0..input.len() {
            if (input[i] - output[i]).abs() > 0.001 {
                differences += 1;
            }
        }
        assert!(differences > 100); // Should have many differences
    }

    #[test]
    fn test_motion_prediction() {
        let predictor = MotionPredictor::new();
        let mut history = VecDeque::new();

        // Add motion snapshots
        for i in 0..5 {
            let snapshot = MotionSnapshot {
                position: Position3D::new(i as f32, 0.0, 0.0),
                velocity: Position3D::new(1.0, 0.0, 0.0),
                acceleration: Position3D::new(0.0, 0.0, 0.0),
                timestamp: i as f64 * 0.1,
            };
            history.push_back(snapshot);
        }

        // Predict position
        let prediction_time = Duration::from_millis(100);
        let predicted = predictor
            .predict_position(&history, prediction_time)
            .unwrap();

        // Should predict ahead
        assert!(predicted.x > 4.0); // Last position was 4.0, should be ahead
    }

    #[test]
    fn test_dynamic_source_motion_history() {
        let source = SoundSource::new_point("test".to_string(), Position3D::default());
        let mut dynamic_source = DynamicSource::new(source);

        // Update motion several times
        for i in 0..10 {
            let pos = Position3D::new(i as f32, 0.0, 0.0);
            let vel = Position3D::new(1.0, 0.0, 0.0);
            let acc = Position3D::new(0.0, 0.0, 0.0);
            dynamic_source.update_motion(pos, vel, acc);
        }

        // Should have motion history
        assert_eq!(dynamic_source.motion_history.len(), 10);

        // Test prediction
        let predicted = dynamic_source.predict_position(Duration::from_millis(100));
        assert!(predicted.x > 9.0); // Should predict ahead of last position
        assert!(dynamic_source.is_moving()); // Should detect movement
    }

    #[test]
    fn test_enhanced_head_tracker_configuration() {
        let mut tracker = HeadTracker::new();

        // Test configuration
        tracker.configure(30, 50, 0.5, 0.6);
        tracker.set_prediction_enabled(true);
        tracker.set_latency_compensation(20.0);

        // Add some position data
        for i in 0..5 {
            let pos = Position3D::new(i as f32 * 0.1, 0.0, 0.0);
            tracker.update_position_with_time(pos, i as f64 * 0.1);
        }

        // Test current position
        assert!(tracker.current_position().is_some());
        assert!(tracker.current_velocity().is_some());

        // Test prediction quality
        let quality = tracker.prediction_quality();
        assert!(quality >= 0.0 && quality <= 1.0);

        // Test reset
        tracker.reset();
        assert!(tracker.current_position().is_none());
    }

    #[test]
    fn test_listener_movement_system() {
        let mut system = ListenerMovementSystem::new();

        // Test initial state
        assert_eq!(system.navigation_mode, NavigationMode::FreeFlight);
        assert!(system.enable_movement_prediction);

        // Test position update
        let position = Position3D::new(1.0, 0.0, 0.0);
        let result = system.update_position_from_platform(position, None);
        assert!(result.is_ok());

        assert_eq!(system.listener().position(), position);

        // Test prediction
        let predicted = system.predict_position(Duration::from_millis(100));
        assert!(predicted.is_some());

        // Test metrics
        let metrics = system.movement_metrics();
        assert!(metrics.update_count > 0);
    }

    #[test]
    fn test_navigation_modes() {
        // Test seated mode
        let seated_system = ListenerMovementSystem::with_navigation_mode(NavigationMode::Seated);
        assert_eq!(seated_system.movement_constraints.max_speed, 0.0);
        assert!(!seated_system.enable_movement_prediction);

        // Test walking mode
        let walking_system = ListenerMovementSystem::with_navigation_mode(NavigationMode::Walking);
        assert_eq!(walking_system.movement_constraints.max_speed, 5.0);
        assert!(walking_system.comfort_settings.ground_reference);

        // Test vehicle mode
        let vehicle_system = ListenerMovementSystem::with_navigation_mode(NavigationMode::Vehicle);
        assert_eq!(vehicle_system.movement_constraints.max_speed, 50.0);
        assert_eq!(
            vehicle_system.comfort_settings.motion_sickness_reduction,
            0.7
        );
    }

    #[test]
    fn test_movement_constraints() {
        let mut system = ListenerMovementSystem::new();

        // Set boundary constraints
        let boundary = Box3D {
            min: Position3D::new(-5.0, 0.0, -5.0),
            max: Position3D::new(5.0, 3.0, 5.0),
            material_id: "boundary".to_string(),
        };

        let constraints = MovementConstraints {
            boundary: Some(boundary),
            max_speed: 2.0,
            max_acceleration: 5.0,
            ground_height: Some(0.0),
            ceiling_height: Some(3.0),
        };

        system.set_movement_constraints(constraints);

        // Test position constraint
        let out_of_bounds = Position3D::new(10.0, -1.0, 0.0);
        let result = system.update_position_from_platform(out_of_bounds, None);
        assert!(result.is_ok());

        // Position should be constrained
        let actual_pos = system.listener().position();
        assert!(actual_pos.x <= 5.0);
        assert!(actual_pos.y >= 0.0);
    }

    #[test]
    fn test_comfort_settings() {
        let mut system = ListenerMovementSystem::new();

        let comfort = ComfortSettings {
            motion_sickness_reduction: 0.8,
            snap_turn: true,
            snap_turn_degrees: 45.0,
            movement_vignetting: true,
            ground_reference: true,
            speed_multiplier: 0.8,
        };

        system.set_comfort_settings(comfort.clone());

        // Test orientation snap
        let orientation = (0.1, 0.0, 0.0); // Small rotation
        let result = system.update_orientation_from_platform(orientation, None);
        assert!(result.is_ok());

        // Should be snapped to nearest 45-degree increment
        let actual_orientation = system.listener().orientation();
        assert!(actual_orientation.0 % 45.0f32.to_radians() < 0.01);
    }

    #[test]
    fn test_platform_integration() {
        let mut integration = PlatformIntegration::new(PlatformType::Oculus);
        assert_eq!(integration.platform_type(), PlatformType::Oculus);

        // Test platform data update
        let data = PlatformData {
            device_id: "Oculus Quest 2".to_string(),
            pose_data: vec![1.0, 0.0, 0.0, 0.0], // Quaternion
            tracking_confidence: 0.95,
            platform_timestamp: 123456,
            properties: std::collections::HashMap::new(),
        };

        integration.update_tracking_data(data.clone());
        assert_eq!(integration.tracking_confidence(), 0.95);

        // Test calibration
        let calibration = CalibrationData {
            head_circumference: Some(58.5),
            ipd: Some(63.0),
            height_offset: 1.75,
            forward_offset: 0.0,
            custom_hrtf_profile: None,
        };

        integration.set_calibration(calibration);
    }

    #[test]
    fn test_movement_metrics() {
        let mut system = ListenerMovementSystem::new();

        // Move in a pattern
        for i in 0..10 {
            let pos = Position3D::new(i as f32 * 0.5, 0.0, 0.0);
            system.update_position_from_platform(pos, None).unwrap();
        }

        let metrics = system.movement_metrics();
        assert!(metrics.total_distance > 0.0);
        assert!(metrics.update_count == 10);

        // Reset and verify
        system.reset_metrics();
        let new_metrics = system.movement_metrics();
        assert_eq!(new_metrics.total_distance, 0.0);
        assert_eq!(new_metrics.update_count, 0);
    }

    #[test]
    fn test_platform_types() {
        // Test all platform types can be created
        let platforms = [
            PlatformType::Generic,
            PlatformType::Oculus,
            PlatformType::SteamVR,
            PlatformType::ARKit,
            PlatformType::ARCore,
            PlatformType::WMR,
            PlatformType::Custom,
        ];

        for platform in platforms.iter() {
            let integration = PlatformIntegration::new(*platform);
            assert_eq!(integration.platform_type(), *platform);
        }
    }
}
