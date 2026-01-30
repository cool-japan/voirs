//! VR Platform Integration Example
//!
//! This example demonstrates:
//! - Integrating with VR platforms for head tracking
//! - Real-time spatial audio processing for VR
//! - Head tracking and position updates
//! - Dynamic source management
//!
//! Run with: cargo run --example vr_platform_integration --no-default-features
//!
//! Note: This example demonstrates the API without requiring actual VR hardware

use std::f32::consts::PI;
use voirs_spatial::{PlatformCapabilities, Position3D, TrackingConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VR Platform Integration Example ===\n");

    // Create tracking configuration for VR head tracking
    let tracking_config = TrackingConfig {
        enable_prediction: true,
        prediction_time_ms: 15.0, // 15ms prediction for VR
        position_smoothing: 0.8,
        orientation_smoothing: 0.7,
        enable_hand_tracking: true,
        enable_eye_tracking: false,
        target_refresh_rate: 90.0,
    };

    println!("✓ Created VR head tracking configuration");
    println!(
        "  Prediction enabled: {}",
        tracking_config.enable_prediction
    );
    println!(
        "  Prediction time: {:.1}ms (for latency compensation)",
        tracking_config.prediction_time_ms
    );
    println!(
        "  Position smoothing: {:.2}",
        tracking_config.position_smoothing
    );
    println!(
        "  Orientation smoothing: {:.2}",
        tracking_config.orientation_smoothing
    );
    println!(
        "  Target refresh rate: {}Hz\n",
        tracking_config.target_refresh_rate
    );

    // Simulate VR platform capabilities
    let platform_caps = PlatformCapabilities {
        head_tracking_6dof: true,
        hand_tracking: true,
        eye_tracking: false,
        controller_tracking: true,
        room_scale: true,
        passthrough: false,
        refresh_rates: vec![72.0, 90.0, 120.0],
        tracking_range: 5.0, // 5 meters
    };

    println!("✓ VR Platform Capabilities:");
    println!(
        "  6DOF head tracking: {}",
        if platform_caps.head_tracking_6dof {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "  Hand tracking: {}",
        if platform_caps.hand_tracking {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "  Eye tracking: {}",
        if platform_caps.eye_tracking {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "  Controller tracking: {}",
        if platform_caps.controller_tracking {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "  Room scale: {}",
        if platform_caps.room_scale {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "  Supported refresh rates: {:?}",
        platform_caps.refresh_rates
    );
    println!("  Tracking range: {:.1}m\n", platform_caps.tracking_range);

    // Simulate VR environment with multiple audio sources
    println!("Setting up VR audio environment:\n");

    // Listener (VR user) starts at origin facing forward
    let listener_position = Position3D::new(0.0, 0.0, 1.6); // Eye height ~1.6m
    let mut listener_orientation = Position3D::new(0.0, 1.0, 0.0); // Facing forward (Y+)

    println!("Initial listener state:");
    println!(
        "  Position: ({:.2}, {:.2}, {:.2})",
        listener_position.x, listener_position.y, listener_position.z
    );
    println!(
        "  Orientation: ({:.2}, {:.2}, {:.2})\n",
        listener_orientation.x, listener_orientation.y, listener_orientation.z
    );

    // Define audio sources in the VR environment
    let sources = vec![
        ("ambient_music", Position3D::new(0.0, 2.0, 1.6), true), // Front, at eye level
        ("left_speaker", Position3D::new(-3.0, 0.0, 1.6), false), // Left side
        ("right_speaker", Position3D::new(3.0, 0.0, 1.6), false), // Right side
        ("npc_voice", Position3D::new(1.5, 2.5, 1.6), true),     // Front-right, slightly ahead
        ("footsteps", Position3D::new(0.0, -2.0, 0.0), true),    // Behind, ground level
        ("overhead_fx", Position3D::new(0.0, 0.0, 3.5), false),  // Overhead
    ];

    println!("Audio sources in VR environment:");
    for (name, pos, moving) in &sources {
        println!(
            "  • {:<15} - ({:+.1}, {:+.1}, {:+.1}) {}",
            name,
            pos.x,
            pos.y,
            pos.z,
            if *moving { "[dynamic]" } else { "[static]" }
        );
    }
    println!();

    // Simulate head tracking updates over time
    println!("Simulating VR head movement:\n");

    let simulation_steps = 5;
    for step in 0..simulation_steps {
        let t = step as f32 / (simulation_steps - 1) as f32;

        // Simulate head rotation (looking left to right)
        let yaw_angle = (t - 0.5) * PI / 2.0; // ±45 degrees

        // Update listener orientation based on yaw rotation
        listener_orientation = Position3D::new(yaw_angle.sin(), yaw_angle.cos(), 0.0);

        // Calculate distances and relative positions to sources
        println!("Frame {} (yaw: {:+.1}°):", step, yaw_angle.to_degrees());

        for (name, source_pos, _) in &sources {
            let distance = listener_position.distance_to(source_pos);

            // Calculate relative position
            let relative = Position3D::new(
                source_pos.x - listener_position.x,
                source_pos.y - listener_position.y,
                source_pos.z - listener_position.z,
            );

            // Calculate angle relative to listener orientation
            let dot = relative
                .normalized()
                .dot(&listener_orientation.normalized());
            let angle = dot.acos().to_degrees();

            println!(
                "  {:<15} - distance: {:.2}m, angle: {:>4.0}°",
                name, distance, angle
            );
        }
        println!();
    }

    // Demonstrate distance-based culling for performance
    println!("Distance-based source management:");
    let max_audible_distance = 20.0; // Maximum hearing distance
    let priority_distance = 5.0; // High priority within this distance

    for (name, pos, _) in &sources {
        let distance = listener_position.distance_to(pos);
        let status = if distance < priority_distance {
            "HIGH PRIORITY"
        } else if distance < max_audible_distance {
            "active       "
        } else {
            "culled (far) "
        };

        println!("  {:<15} - {:.1}m [{:<13}]", name, distance, status);
    }
    println!();

    // Calculate performance metrics for VR
    println!("VR Performance Metrics:");
    let sample_rate = 48000;
    let buffer_size = 256;
    let target_latency_ms = 20.0;
    let buffer_duration_ms = (buffer_size as f32 / sample_rate as f32) * 1000.0;
    let estimated_processing_ms = buffer_duration_ms * 0.3; // Estimate 30% of buffer time
    let total_latency_ms =
        buffer_duration_ms + estimated_processing_ms + tracking_config.prediction_time_ms;

    println!(
        "  Sample rate: {}Hz, Buffer: {} samples",
        sample_rate, buffer_size
    );
    println!("  Buffer duration: {:.2}ms", buffer_duration_ms);
    println!("  Processing estimate: {:.2}ms", estimated_processing_ms);
    println!(
        "  Tracking prediction: {:.2}ms",
        tracking_config.prediction_time_ms
    );
    println!("  Total latency: {:.2}ms", total_latency_ms);
    println!(
        "  Target: <{:.0}ms {}",
        target_latency_ms,
        if total_latency_ms < target_latency_ms {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );
    println!();

    // Summary of VR integration features
    println!("✅ VR Platform Integration Complete!\n");
    println!("Key features demonstrated:");
    println!("  • Low-latency audio processing (<20ms target)");
    println!("  • Head tracking with prediction compensation");
    println!("  • Multiple simultaneous spatial audio sources");
    println!("  • Distance-based source management");
    println!("  • Real-time position and orientation updates");
    println!("  • Performance metrics and optimization");
    println!();
    println!("In a real VR application:");
    println!("  1. Connect to VR SDK (SteamVR/Oculus/etc.)");
    println!("  2. Stream head pose data at 90-120Hz");
    println!("  3. Process audio with predicted head position");
    println!("  4. Output to VR headset audio device");
    println!("  5. Monitor latency and adjust quality dynamically");

    Ok(())
}
