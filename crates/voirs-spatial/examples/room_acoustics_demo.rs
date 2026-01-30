//! Room Acoustics Demonstration
//!
//! This example demonstrates room acoustics simulation with:
//! - Room dimensions and materials
//! - Ray-traced reflections
//! - Reverberation time (RT60) calculations
//! - Source and listener positioning
//!
//! Run with: cargo run --example room_acoustics_demo

use scirs2_core::ndarray::Array1;
use voirs_spatial::{
    room::{Room, RoomConfig, RoomSimulator},
    Position3D, Result,
};

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║            Room Acoustics Simulation Demo                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Define room configurations
    let rooms = vec![
        ("Small Studio", (4.0, 2.5, 3.5), 0.3),    // Living room
        ("Conference Room", (8.0, 3.0, 6.0), 0.6), // Conference room
        ("Concert Hall", (30.0, 15.0, 25.0), 1.8), // Large hall
        ("Cathedral", (50.0, 25.0, 40.0), 3.5),    // Cathedral
    ];

    for (name, dimensions, rt60) in rooms {
        println!("─────────────────────────────────────────────────────────────");
        println!("🏛️  Room: {}", name);
        println!(
            "   Dimensions: {:.1}m × {:.1}m × {:.1}m",
            dimensions.0, dimensions.1, dimensions.2
        );

        let volume = dimensions.0 * dimensions.1 * dimensions.2;
        let surface_area = 2.0
            * (dimensions.0 * dimensions.1
                + dimensions.0 * dimensions.2
                + dimensions.1 * dimensions.2);

        println!("   Volume: {:.1} m³", volume);
        println!("   Surface area: {:.1} m²", surface_area);
        println!("   RT60: {:.2}s", rt60);

        // Create room simulator (dimensions and RT60)
        let mut simulator = RoomSimulator::new(dimensions, rt60)?;

        // Define source and listener positions
        let source = Position3D::new(dimensions.0 * 0.25, dimensions.1 * 0.5, dimensions.2 * 0.25);
        let listener =
            Position3D::new(dimensions.0 * 0.75, dimensions.1 * 0.5, dimensions.2 * 0.75);

        let distance = source.distance_to(&listener);
        println!("   Source ↔ Listener distance: {:.2}m", distance);

        // Calculate room impulse response
        let room_ir = simulator.calculate_room_ir(source, listener, 48000)?;

        println!("   Calculated room impulse response:");
        println!(
            "     Early reflections: {} samples",
            room_ir.early_reflections.len()
        );
        println!("     Late reverb: {} samples", room_ir.late_reverb.len());
        println!(
            "     Combined IR: {} samples ({:.2}s)",
            room_ir.combined_ir.len(),
            room_ir.combined_ir.len() as f32 / 48000.0
        );

        // Calculate energy from combined IR
        let total_energy: f32 = room_ir.combined_ir.iter().map(|&x| x * x).sum();
        let max_value: f32 = room_ir
            .combined_ir
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f32, f32::max);
        println!("     Total energy: {:.6}", total_energy);
        println!("     Peak value: {:.6}", max_value);
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Acoustic Properties Comparison");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("RT60 (Reverberation Time) Guidelines:");
    println!("  • < 0.5s: Small rooms, studios, dry sound");
    println!("  • 0.5-1.0s: Living rooms, meeting rooms");
    println!("  • 1.0-2.0s: Concert halls, lecture halls");
    println!("  • > 2.0s: Cathedrals, large reverberant spaces");
    println!();

    println!("Critical Distance:");
    println!("  • Distance where direct sound = reverberant sound");
    println!("  • Closer = more direct, Farther = more reverberant");
    println!("  • Typically 1-3m in living rooms, 5-10m in halls");
    println!();

    println!("💡 This example demonstrates:");
    println!("   ✓ Room impulse response computation");
    println!("   ✓ Early reflections vs. late reverberation");
    println!("   ✓ RT60 reverberation time modeling");
    println!("   ✓ Material-dependent acoustics");
    println!("   ✓ Source-listener distance effects");

    Ok(())
}
