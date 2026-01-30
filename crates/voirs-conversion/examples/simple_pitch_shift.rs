//! Simple Pitch Shift Example
//!
//! Demonstrates basic pitch shifting transformation.
//!
//! Run with:
//! ```bash
//! cargo run --example simple_pitch_shift
//! ```

use voirs_conversion::transforms::{PitchTransform, Transform};
use voirs_conversion::Result;

fn main() -> Result<()> {
    println!("🎵 Simple Pitch Shift Example\n");

    // Generate sample audio (1 second at 22050 Hz)
    let sample_rate = 22050;
    let duration = 1.0;
    let audio = generate_sine_wave(sample_rate, duration, 200.0);

    println!(
        "✓ Generated audio: {:.1}s at {}Hz ({} samples)",
        duration,
        sample_rate,
        audio.len()
    );
    println!("  Base frequency: 200 Hz\n");

    // Test different pitch shift factors
    let pitch_factors = vec![0.5, 0.75, 1.0, 1.25, 1.5, 2.0];

    for &factor in &pitch_factors {
        let transform = PitchTransform::new(factor);
        let result = transform.apply(&audio)?;

        let expected_freq = 200.0 * factor;
        println!(
            "Pitch factor {:.2}x: {} → {} samples (expected freq: {:.0} Hz)",
            factor,
            audio.len(),
            result.len(),
            expected_freq
        );
    }

    println!("\n✅ All pitch shifts completed successfully!");

    Ok(())
}

fn generate_sine_wave(sample_rate: u32, duration: f32, frequency: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration) as usize;

    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.3
        })
        .collect()
}
