//! Simple 3D positioning example
//!
//! This example demonstrates:
//! - Creating Position3D objects
//! - Distance calculations
//! - Vector operations
//! - Coordinate transformations
//!
//! Run with: cargo run --example simple_positioning --no-default-features

use voirs_spatial::Position3D;

fn main() {
    println!("=== Simple 3D Positioning Example ===\n");

    // Create positions
    let listener = Position3D::new(0.0, 0.0, 0.0);
    let source1 = Position3D::new(2.0, 0.0, 0.0); // 2m to the right
    let source2 = Position3D::new(0.0, 3.0, 0.0); // 3m forward
    let source3 = Position3D::new(-1.0, 1.0, 1.0); // left, forward, and up

    println!("✓ Created positions:");
    println!(
        "  Listener: ({:.1}, {:.1}, {:.1})",
        listener.x, listener.y, listener.z
    );
    println!(
        "  Source 1: ({:.1}, {:.1}, {:.1})",
        source1.x, source1.y, source1.z
    );
    println!(
        "  Source 2: ({:.1}, {:.1}, {:.1})",
        source2.x, source2.y, source2.z
    );
    println!(
        "  Source 3: ({:.1}, {:.1}, {:.1})\n",
        source3.x, source3.y, source3.z
    );

    // Calculate distances
    println!("Distance calculations:");
    println!(
        "  Source 1 distance: {:.2}m",
        source1.distance_to(&listener)
    );
    println!(
        "  Source 2 distance: {:.2}m",
        source2.distance_to(&listener)
    );
    println!(
        "  Source 3 distance: {:.2}m\n",
        source3.distance_to(&listener)
    );

    // Vector operations
    println!("Vector operations:");

    let mag1 = source1.magnitude();
    println!("  Source 1 magnitude: {:.2}", mag1);

    let norm1 = source1.normalized();
    println!(
        "  Source 1 normalized: ({:.2}, {:.2}, {:.2})",
        norm1.x, norm1.y, norm1.z
    );
    println!(
        "  Normalized magnitude: {:.2} (should be 1.0)",
        norm1.magnitude()
    );

    let dot = source1.dot(&source2);
    println!("\n  Dot product (source1 · source2): {:.2}", dot);

    let cross = source1.cross(&source2);
    println!(
        "  Cross product (source1 × source2): ({:.2}, {:.2}, {:.2})\n",
        cross.x, cross.y, cross.z
    );

    // Interpolation
    println!("Linear interpolation (LERP):");
    for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let interp = source1.lerp(&source2, t);
        println!(
            "  t={:.2}: ({:.2}, {:.2}, {:.2})",
            t, interp.x, interp.y, interp.z
        );
    }

    // Simulate a moving source
    println!("\nSimulating moving source (10 steps):");
    let start_pos = Position3D::new(-5.0, 5.0, 0.0);
    let end_pos = Position3D::new(5.0, 5.0, 0.0);

    for step in 0..=10 {
        let t = step as f32 / 10.0;
        let current_pos = start_pos.lerp(&end_pos, t);
        let distance = current_pos.distance_to(&listener);
        let angle = current_pos.y.atan2(current_pos.x).to_degrees();

        println!(
            "  Step {}: pos=({:+.1}, {:+.1}, {:+.1}) dist={:.2}m angle={:+.0}°",
            step, current_pos.x, current_pos.y, current_pos.z, distance, angle
        );
    }

    println!("\n✅ Example completed!");
    println!("\nKey takeaways:");
    println!("  - Position3D provides basic 3D vector operations");
    println!("  - Distance calculations are straightforward");
    println!("  - LERP enables smooth position transitions");
    println!("  - All operations are optimized for audio processing");
}
