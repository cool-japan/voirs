//! Speed and Pitch Transformation Example
//!
//! Demonstrates independent control of speed and pitch.
//!
//! Run with:
//! ```bash
//! cargo run --example speed_and_pitch
//! ```

use voirs_conversion::transforms::{PitchTransform, SpeedTransform, Transform};
use voirs_conversion::Result;

fn main() -> Result<()> {
    println!("⚡ Speed and Pitch Transformation Example\n");

    // Generate sample audio with varying frequency (simulating speech)
    let sample_rate = 22050;
    let audio = generate_speech_like_audio(sample_rate, 2.0);

    println!(
        "✓ Generated speech-like audio: 2.0s ({} samples)\n",
        audio.len()
    );

    // Scenario 1: Slow down without changing pitch
    println!("--- Scenario 1: Slow down (0.8x) without pitch change ---");
    let speed_transform = SpeedTransform::new(0.8);
    let slowed = speed_transform.apply(&audio)?;
    println!("  Original: {} samples (2.0s)", audio.len());
    println!("  Slowed:   {} samples (~2.5s)", slowed.len());
    println!("  Pitch: unchanged\n");

    // Scenario 2: Speed up without changing pitch
    println!("--- Scenario 2: Speed up (1.3x) without pitch change ---");
    let speed_transform = SpeedTransform::new(1.3);
    let sped_up = speed_transform.apply(&audio)?;
    println!("  Original: {} samples (2.0s)", audio.len());
    println!("  Sped up:  {} samples (~1.5s)", sped_up.len());
    println!("  Pitch: unchanged\n");

    // Scenario 3: Raise pitch without changing speed
    println!("--- Scenario 3: Raise pitch (1.5x) without speed change ---");
    let pitch_transform = PitchTransform::new(1.5);
    let pitched = pitch_transform.apply(&audio)?;
    println!("  Original: {} samples, base pitch ~200Hz", audio.len());
    println!("  Pitched:  {} samples, pitch ~300Hz", pitched.len());
    println!("  Duration: ~2.0s (unchanged)\n");

    // Scenario 4: Combined transformation
    println!("--- Scenario 4: Combined (faster + higher pitch) ---");
    let speed_transform = SpeedTransform::new(1.2);
    let pitch_transform = PitchTransform::new(1.4);

    let step1 = speed_transform.apply(&audio)?;
    let step2 = pitch_transform.apply(&step1)?;

    println!("  Original:    {} samples (2.0s at 200Hz)", audio.len());
    println!("  After speed: {} samples (~1.67s at 200Hz)", step1.len());
    println!("  After pitch: {} samples (~1.67s at 280Hz)", step2.len());

    println!("\n✅ All transformations completed!");
    println!("\n💡 Key insight: Speed and pitch can be controlled independently");

    Ok(())
}

fn generate_speech_like_audio(sample_rate: u32, duration: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration) as usize;

    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;

            // Varying fundamental frequency (simulating intonation)
            let f0 = 200.0 + 30.0 * (2.0 * std::f32::consts::PI * 3.0 * t).sin();

            // Harmonics
            let h1 = (2.0 * std::f32::consts::PI * f0 * t).sin();
            let h2 = 0.5 * (2.0 * std::f32::consts::PI * f0 * 2.0 * t).sin();
            let h3 = 0.25 * (2.0 * std::f32::consts::PI * f0 * 3.0 * t).sin();

            // Amplitude envelope (syllables)
            let envelope = ((2.0 * std::f32::consts::PI * 4.0 * t).sin() * 0.5 + 0.5).powf(2.0);

            (h1 + h2 + h3) * envelope * 0.3
        })
        .collect()
}
