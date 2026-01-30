//! Age Transformation Example
//!
//! Demonstrates voice aging and de-aging transformations.
//!
//! Run with:
//! ```bash
//! cargo run --example age_transformation
//! ```

use voirs_conversion::transforms::{AgeTransform, Transform};
use voirs_conversion::Result;

fn main() -> Result<()> {
    println!("👴 Age Transformation Example\n");

    // Generate sample audio representing a 30-year-old voice
    let sample_rate = 22050;
    let audio = generate_voice_audio(sample_rate, 2.0, 30.0);

    println!("✓ Generated voice sample: 30 years old");
    println!("  Duration: 2.0s ({} samples)\n", audio.len());

    // Transformation 1: Young adult (25 years)
    println!("--- Transformation 1: De-aging to 25 years ---");
    let transform1 = AgeTransform::new(30.0, 25.0);
    let result1 = transform1.apply(&audio)?;
    println!("  ✓ Transformed: 30 → 25 years");
    println!("  Output: {} samples", result1.len());
    println!("  Changes: Slightly brighter voice, more energy\n");

    // Transformation 2: Middle-aged (50 years)
    println!("--- Transformation 2: Aging to 50 years ---");
    let transform2 = AgeTransform::new(30.0, 50.0);
    let result2 = transform2.apply(&audio)?;
    println!("  ✓ Transformed: 30 → 50 years");
    println!("  Output: {} samples", result2.len());
    println!("  Changes: Lower formants, slightly deeper voice\n");

    // Transformation 3: Elderly (70 years)
    println!("--- Transformation 3: Aging to 70 years ---");
    let transform3 = AgeTransform::new(30.0, 70.0);
    let result3 = transform3.apply(&audio)?;
    println!("  ✓ Transformed: 30 → 70 years");
    println!("  Output: {} samples", result3.len());
    println!("  Changes: Significant formant shift, voice tremor simulation\n");

    // Transformation 4: Very young (20 years)
    println!("--- Transformation 4: De-aging to 20 years ---");
    let transform4 = AgeTransform::new(30.0, 20.0);
    let result4 = transform4.apply(&audio)?;
    println!("  ✓ Transformed: 30 → 20 years");
    println!("  Output: {} samples", result4.len());
    println!("  Changes: Higher formants, more resonance\n");

    println!("✅ All age transformations completed!");

    // Display age transformation characteristics
    println!("\n📊 Age Transformation Characteristics:");
    println!("  Younger voices (< 30):");
    println!("    • Higher formant frequencies");
    println!("    • More spectral energy in high frequencies");
    println!("    • Clearer articulation");
    println!("\n  Older voices (> 50):");
    println!("    • Lower formant frequencies");
    println!("    • Reduced high-frequency content");
    println!("    • Possible voice tremor effects");

    Ok(())
}

fn generate_voice_audio(sample_rate: u32, duration: f32, age: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration) as usize;

    // Age-dependent parameters
    let age_factor = (age - 20.0) / 50.0; // 0.0 for 20yo, 1.0 for 70yo
    let f0 = 150.0 - age_factor * 20.0; // Fundamental frequency decreases with age

    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;

            // Fundamental + harmonics
            let h1 = (2.0 * std::f32::consts::PI * f0 * t).sin();
            let h2 = (1.0 - age_factor * 0.3) * (2.0 * std::f32::consts::PI * f0 * 2.0 * t).sin();
            let h3 = (1.0 - age_factor * 0.5) * (2.0 * std::f32::consts::PI * f0 * 3.0 * t).sin();

            // Syllable envelope
            let envelope = ((2.0 * std::f32::consts::PI * 3.5 * t).sin() * 0.5 + 0.5).powf(1.5);

            (h1 + h2 + h3) * envelope * 0.3
        })
        .collect()
}
