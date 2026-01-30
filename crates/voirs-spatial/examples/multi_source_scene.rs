//! Multi-Source Spatial Audio Scene Example
//!
//! This example demonstrates handling multiple simultaneous audio sources
//! with proper position management and performance optimization.
//!
//! Run with: cargo run --example multi_source_scene

use scirs2_core::ndarray::Array1;
use voirs_spatial::{Position3D, Result, SIMDSpatialOps};

/// Audio source in the scene
struct AudioSource {
    id: usize,
    position: Position3D,
    velocity: Position3D,
    active: bool,
}

impl AudioSource {
    fn new(id: usize, position: Position3D) -> Self {
        Self {
            id,
            position,
            velocity: Position3D::new(0.0, 0.0, 0.0),
            active: true,
        }
    }

    fn update(&mut self, delta_time: f32) {
        // Update position based on velocity
        let displacement = self.velocity.scale(delta_time);
        self.position = self.position.add(&displacement);

        // Bounce off boundaries (-10 to 10 in each dimension)
        if self.position.x.abs() > 10.0 {
            self.velocity.x *= -1.0;
            self.position.x = self.position.x.clamp(-10.0, 10.0);
        }
        if self.position.y.abs() > 10.0 {
            self.velocity.y *= -1.0;
            self.position.y = self.position.y.clamp(-10.0, 10.0);
        }
        if self.position.z.abs() > 10.0 {
            self.velocity.z *= -1.0;
            self.position.z = self.position.z.clamp(-10.0, 10.0);
        }
    }
}

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Multi-Source Spatial Audio Scene Demo               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Create listener at origin
    let listener = Position3D::new(0.0, 0.0, 0.0);

    // Create multiple audio sources in a sphere around listener
    let mut sources = Vec::new();
    let num_sources = 16;

    println!("🎵 Creating {} audio sources...", num_sources);
    for i in 0..num_sources {
        // Position sources on a sphere
        let theta = (i as f32) * 2.0 * std::f32::consts::PI / (num_sources as f32);
        let phi = ((i % 4) as f32) * std::f32::consts::PI / 4.0;

        let radius = 5.0;
        let position = Position3D::new(
            radius * phi.sin() * theta.cos(),
            radius * phi.cos(),
            radius * phi.sin() * theta.sin(),
        );

        let mut source = AudioSource::new(i, position);

        // Assign random velocities
        source.velocity = Position3D::new(
            fastrand::f32() * 2.0 - 1.0,
            fastrand::f32() * 2.0 - 1.0,
            fastrand::f32() * 2.0 - 1.0,
        );

        sources.push(source);
    }
    println!("✅ Sources created in spherical arrangement");
    println!();

    // Simulate 5 seconds
    let delta_time = 0.1; // 100ms updates
    let num_updates = 50;

    println!("🎬 Simulating spatial audio scene...");
    println!();

    for update_idx in 0..num_updates {
        // Update all source positions
        for source in &mut sources {
            source.update(delta_time);
        }

        // Collect positions for SIMD processing
        let positions: Vec<Position3D> = sources.iter().map(|s| s.position).collect();

        // Calculate all distances in one SIMD operation
        let distances = SIMDSpatialOps::distances(listener, &positions);

        // Print status every 10 updates (1 second)
        if update_idx % 10 == 0 {
            let time = update_idx as f32 * delta_time;
            println!("⏱️  Time: {:.1}s", time);
            println!(
                "   Active sources: {}",
                sources.iter().filter(|s| s.active).count()
            );

            // Show distances for first 4 sources
            println!("   Distances:");
            for (i, &distance) in distances.iter().enumerate().take(4) {
                println!("     Source {}: {:.2}m", i, distance);
            }

            // Calculate average distance
            let avg_distance: f32 = distances.iter().sum::<f32>() / distances.len() as f32;
            println!("   Average distance: {:.2}m", avg_distance);
            println!();
        }
    }

    println!("✅ Simulation complete!");
    println!();

    // Final statistics
    println!("📊 Final Statistics:");
    println!("   Total sources: {}", sources.len());
    println!("   Updates processed: {}", num_updates);
    println!("   Total time: {:.1}s", num_updates as f32 * delta_time);
    println!();

    println!("💡 This example demonstrates:");
    println!("   ✓ Managing multiple simultaneous audio sources");
    println!("   ✓ Efficient distance calculations with SIMD");
    println!("   ✓ Dynamic source position updates");
    println!("   ✓ Boundary handling for contained environments");

    Ok(())
}
