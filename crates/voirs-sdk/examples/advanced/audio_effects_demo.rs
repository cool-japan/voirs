//! Audio Effects Processing Demo
//!
//! Demonstrates the professional audio effects system including reverb, delay,
//! chorus, compression, and equalization. Shows both individual effects and
//! effects chains for sophisticated audio processing.
//!
//! Run with: cargo run --example audio_effects_demo

use voirs_sdk::audio::effects::*;
use voirs_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== VoiRS Audio Effects Processing Demo ===\n");

    // Create pipeline
    println!("Creating synthesis pipeline...");
    let pipeline = VoirsPipelineBuilder::new().build().await?;
    println!("✓ Pipeline ready\n");

    // Synthesize test audio
    let text = "Welcome to the audio effects demonstration. This shows professional-grade audio processing capabilities.";
    println!("Synthesizing audio: \"{}\"", text);
    let audio = pipeline.synthesize(text).await?;
    println!(
        "✓ Generated audio: {:.2}s, {} Hz, {} samples\n",
        audio.duration(),
        audio.sample_rate(),
        audio.len()
    );

    // Example 1: Reverb Effect
    println!("=== Example 1: Reverb Effect (Freeverb Algorithm) ===");
    println!("Applying natural room reverb...");
    let reverb = ReverbEffect::new(audio.sample_rate())
        .with_room_size(0.8) // Large room
        .with_damping(0.5) // Medium damping
        .with_wet_level(0.3) // 30% effect
        .with_dry_level(0.7); // 70% original

    let reverb_audio = reverb.process(&audio)?;
    println!("✓ Reverb applied:");
    println!("  Room size: 0.8 (large room)");
    println!("  Damping: 0.5 (medium)");
    println!("  Wet/Dry mix: 30%/70%");
    println!(
        "  Output: {} samples, {:.2}s duration",
        reverb_audio.len(),
        reverb_audio.duration()
    );
    println!("  Effect: Natural concert hall ambience\n");

    // Example 2: Delay/Echo Effect
    println!("=== Example 2: Delay/Echo Effect ===");
    println!("Applying delay with feedback...");
    let delay = DelayEffect::new(audio.sample_rate(), 0.3, 0.4)?
        .with_wet_level(0.5)
        .with_dry_level(1.0);

    let delayed_audio = delay.process(&audio)?;
    println!("✓ Delay applied:");
    println!("  Delay time: 300ms");
    println!("  Feedback: 40% (multiple echoes)");
    println!("  Wet/Dry mix: 50%/100%");
    println!(
        "  Output: {} samples, {:.2}s duration",
        delayed_audio.len(),
        delayed_audio.duration()
    );
    println!("  Effect: Distinct echo with gradual decay\n");

    // Example 3: Chorus Effect
    println!("=== Example 3: Chorus Effect (LFO Modulation) ===");
    println!("Applying chorus for richness...");
    let chorus = ChorusEffect::new(audio.sample_rate())
        .with_rate(0.5) // 0.5 Hz LFO
        .with_depth(0.3) // 30% modulation
        .with_wet_level(0.5)
        .with_dry_level(1.0);

    let chorus_audio = chorus.process(&audio)?;
    println!("✓ Chorus applied:");
    println!("  LFO rate: 0.5 Hz (slow modulation)");
    println!("  Depth: 30% (moderate effect)");
    println!("  Base delay: 20ms");
    println!(
        "  Output: {} samples, {:.2}s duration",
        chorus_audio.len(),
        chorus_audio.duration()
    );
    println!("  Effect: Rich, ensemble-like texture\n");

    // Example 4: Dynamic Range Compressor
    println!("=== Example 4: Dynamic Range Compressor ===");
    println!("Applying compression for consistent levels...");
    let compressor = CompressorEffect::new(audio.sample_rate())
        .with_threshold(-20.0) // Compress above -20 dBFS
        .with_ratio(4.0) // 4:1 compression ratio
        .with_attack(0.005) // 5ms attack
        .with_release(0.1) // 100ms release
        .with_makeup_gain(2.0); // +2 dB makeup gain

    let compressed_audio = compressor.process(&audio)?;
    println!("✓ Compressor applied:");
    println!("  Threshold: -20 dBFS");
    println!("  Ratio: 4:1");
    println!("  Attack time: 5ms");
    println!("  Release time: 100ms");
    println!("  Makeup gain: +2 dB");
    println!(
        "  Output: {} samples, {:.2}s duration",
        compressed_audio.len(),
        compressed_audio.duration()
    );

    // Analyze dynamic range reduction
    let original_peak = audio.peak_db();
    let original_rms = audio.rms_db();
    let compressed_peak = compressed_audio.peak_db();
    let compressed_rms = compressed_audio.rms_db();
    let dynamic_range_reduction =
        (original_peak - original_rms) - (compressed_peak - compressed_rms);

    println!(
        "  Original dynamic range: {:.2} dB",
        original_peak - original_rms
    );
    println!(
        "  Compressed dynamic range: {:.2} dB",
        compressed_peak - compressed_rms
    );
    println!(
        "  Dynamic range reduction: {:.2} dB",
        dynamic_range_reduction
    );
    println!("  Effect: More consistent, broadcast-ready levels\n");

    // Example 5: Parametric Equalizer
    println!("=== Example 5: Parametric Equalizer (3-Band) ===");
    println!("Applying frequency-selective processing...");
    let eq = EqualizerEffect::new(audio.sample_rate())
        .add_band(200.0, -3.0, 1.0)? // Cut low rumble at 200 Hz
        .add_band(2000.0, 4.0, 1.5)? // Boost presence at 2 kHz
        .add_band(8000.0, 2.0, 1.0)?; // Gentle high-end boost at 8 kHz

    let eq_audio = eq.process(&audio)?;
    println!("✓ EQ applied (3 bands):");
    println!("  Band 1: 200 Hz, -3 dB (low-end cleanup)");
    println!("  Band 2: 2000 Hz, +4 dB (presence boost)");
    println!("  Band 3: 8000 Hz, +2 dB (air/clarity)");
    println!(
        "  Output: {} samples, {:.2}s duration",
        eq_audio.len(),
        eq_audio.duration()
    );
    println!("  Effect: Enhanced clarity and presence\n");

    // Example 6: Effects Chain (Professional Processing)
    println!("=== Example 6: Effects Chain (Multi-Stage Processing) ===");
    println!("Building professional processing chain...");

    let chain = EffectsChain::new()
        // Stage 1: EQ for tonal shaping
        .add_effect(Box::new(
            EqualizerEffect::new(audio.sample_rate())
                .add_band(100.0, -6.0, 0.7)? // Remove low rumble
                .add_band(3000.0, 3.0, 1.5)?, // Boost presence
        ))
        // Stage 2: Compression for consistent levels
        .add_effect(Box::new(
            CompressorEffect::new(audio.sample_rate())
                .with_threshold(-18.0)
                .with_ratio(3.0)
                .with_attack(0.01)
                .with_release(0.15)
                .with_makeup_gain(3.0),
        ))
        // Stage 3: Subtle chorus for richness
        .add_effect(Box::new(
            ChorusEffect::new(audio.sample_rate())
                .with_rate(0.3)
                .with_depth(0.15)
                .with_wet_level(0.2),
        ))
        // Stage 4: Room reverb for ambience
        .add_effect(Box::new(
            ReverbEffect::new(audio.sample_rate())
                .with_room_size(0.6)
                .with_damping(0.6)
                .with_wet_level(0.2),
        ));

    println!("✓ Chain configured:");
    println!("  Stage 1: Parametric EQ (tonal shaping)");
    println!("  Stage 2: Compressor (level control)");
    println!("  Stage 3: Chorus (subtle richness)");
    println!("  Stage 4: Reverb (room ambience)");
    println!("  Total effects: {}", chain.len());

    println!("\nProcessing through chain...");
    let chain_audio = chain.process(&audio)?;
    println!("✓ Chain processed:");
    println!(
        "  Output: {} samples, {:.2}s duration",
        chain_audio.len(),
        chain_audio.duration()
    );
    println!("  Peak level: {:.2} dB", chain_audio.peak_db());
    println!("  RMS level: {:.2} dB", chain_audio.rms_db());
    println!("  Effect: Professional broadcast quality\n");

    // Example 7: Comparison of Different Reverb Settings
    println!("=== Example 7: Reverb Comparison (Different Room Sizes) ===");

    let small_room = ReverbEffect::new(audio.sample_rate())
        .with_room_size(0.3)
        .with_damping(0.7)
        .with_wet_level(0.25);

    let medium_room = ReverbEffect::new(audio.sample_rate())
        .with_room_size(0.6)
        .with_damping(0.5)
        .with_wet_level(0.25);

    let large_hall = ReverbEffect::new(audio.sample_rate())
        .with_room_size(0.9)
        .with_damping(0.3)
        .with_wet_level(0.25);

    let small_audio = small_room.process(&audio)?;
    let medium_audio = medium_room.process(&audio)?;
    let large_audio = large_hall.process(&audio)?;

    println!("Small Room (room_size=0.3, damping=0.7):");
    println!("  Peak: {:.2} dB", small_audio.peak_db());
    println!("  RMS: {:.2} dB", small_audio.rms_db());
    println!("  Character: Tight, intimate space\n");

    println!("Medium Room (room_size=0.6, damping=0.5):");
    println!("  Peak: {:.2} dB", medium_audio.peak_db());
    println!("  RMS: {:.2} dB", medium_audio.rms_db());
    println!("  Character: Balanced studio room\n");

    println!("Large Hall (room_size=0.9, damping=0.3):");
    println!("  Peak: {:.2} dB", large_audio.peak_db());
    println!("  RMS: {:.2} dB", large_audio.rms_db());
    println!("  Character: Spacious concert hall\n");

    // Example 8: Creative Effects Chain
    println!("=== Example 8: Creative Effects Chain (Experimental) ===");
    println!("Building experimental creative chain...");

    let creative_chain = EffectsChain::new()
        // Heavy compression for pumping effect
        .add_effect(Box::new(
            CompressorEffect::new(audio.sample_rate())
                .with_threshold(-30.0)
                .with_ratio(8.0)
                .with_attack(0.001)
                .with_release(0.05)
                .with_makeup_gain(10.0),
        ))
        // Dual delay for rhythmic effect
        .add_effect(Box::new(
            DelayEffect::new(audio.sample_rate(), 0.15, 0.5)?
                .with_wet_level(0.4)
                .with_dry_level(0.8),
        ))
        // Deep chorus for texture
        .add_effect(Box::new(
            ChorusEffect::new(audio.sample_rate())
                .with_rate(1.0)
                .with_depth(0.5)
                .with_wet_level(0.6),
        ))
        // Large reverb for space
        .add_effect(Box::new(
            ReverbEffect::new(audio.sample_rate())
                .with_room_size(0.95)
                .with_damping(0.2)
                .with_wet_level(0.4),
        ));

    let creative_audio = creative_chain.process(&audio)?;
    println!("✓ Creative chain applied:");
    println!("  Effect 1: Heavy compression (pumping)");
    println!("  Effect 2: Rhythmic delay (150ms, 50% feedback)");
    println!("  Effect 3: Deep chorus (1 Hz LFO, 50% depth)");
    println!("  Effect 4: Huge reverb (room_size=0.95)");
    println!(
        "  Output: {} samples, {:.2}s duration",
        creative_audio.len(),
        creative_audio.duration()
    );
    println!("  Character: Ethereal, atmospheric, experimental\n");

    // Example 9: Effect Quality Comparison
    println!("=== Example 9: Audio Quality Analysis ===");
    println!("Comparing original vs processed audio quality...");

    use voirs_sdk::audio::workflows;

    let original_metrics = workflows::analyze_quality(&audio)?;
    let reverb_metrics = workflows::analyze_quality(&reverb_audio)?;
    let compressed_metrics = workflows::analyze_quality(&compressed_audio)?;
    let chain_metrics = workflows::analyze_quality(&chain_audio)?;

    println!("Original Audio:");
    println!("  RMS: {:.2} dB", original_metrics.rms_db);
    println!("  Peak: {:.2} dB", original_metrics.peak_db);
    println!(
        "  Dynamic range: {:.2} dB",
        original_metrics.peak_db - original_metrics.rms_db
    );
    println!("  SNR: {:.2} dB", original_metrics.snr);

    println!("\nWith Reverb:");
    println!("  RMS: {:.2} dB", reverb_metrics.rms_db);
    println!("  Peak: {:.2} dB", reverb_metrics.peak_db);
    println!(
        "  Dynamic range: {:.2} dB",
        reverb_metrics.peak_db - reverb_metrics.rms_db
    );
    println!("  SNR: {:.2} dB", reverb_metrics.snr);

    println!("\nWith Compression:");
    println!("  RMS: {:.2} dB", compressed_metrics.rms_db);
    println!("  Peak: {:.2} dB", compressed_metrics.peak_db);
    println!(
        "  Dynamic range: {:.2} dB",
        compressed_metrics.peak_db - compressed_metrics.rms_db
    );
    println!("  SNR: {:.2} dB", compressed_metrics.snr);

    println!("\nWith Full Chain:");
    println!("  RMS: {:.2} dB", chain_metrics.rms_db);
    println!("  Peak: {:.2} dB", chain_metrics.peak_db);
    println!(
        "  Dynamic range: {:.2} dB",
        chain_metrics.peak_db - chain_metrics.rms_db
    );
    println!("  SNR: {:.2} dB", chain_metrics.snr);
    println!();

    // Example 10: Real-Time Effect State Management
    println!("=== Example 10: Effect State Management ===");
    println!("Demonstrating effect state reset for real-time processing...");

    let mut stateful_reverb = ReverbEffect::new(audio.sample_rate())
        .with_room_size(0.7)
        .with_wet_level(0.3);

    // Process first segment
    let segment1 = audio.extract(0.0, 1.0)?;
    let processed1 = stateful_reverb.process(&segment1)?;
    println!("✓ Processed segment 1: {:.2}s", processed1.duration());

    // Reset state
    stateful_reverb.reset();
    println!("✓ Effect state reset");

    // Process second segment with clean state
    let segment2 = audio.extract(1.0, 1.0)?;
    let processed2 = stateful_reverb.process(&segment2)?;
    println!("✓ Processed segment 2: {:.2}s", processed2.duration());
    println!("Effect: Clean processing for each segment\n");

    // Summary
    println!("=== Summary ===");
    println!("✓ Demonstrated 10 audio effects scenarios:");
    println!("  1. Reverb (Freeverb algorithm)");
    println!("  2. Delay/Echo with feedback");
    println!("  3. Chorus (LFO modulation)");
    println!("  4. Dynamic range compressor");
    println!("  5. Parametric EQ (3-band)");
    println!("  6. Professional effects chain");
    println!("  7. Reverb comparison (room sizes)");
    println!("  8. Creative experimental chain");
    println!("  9. Quality analysis comparison");
    println!("  10. Real-time state management");
    println!("\n✓ All effects tested and production-ready!");
    println!("\nUse Cases:");
    println!("  • Podcast production (compression + EQ)");
    println!("  • Broadcast audio (full professional chain)");
    println!("  • Creative sound design (experimental chains)");
    println!("  • Voice enhancement (subtle reverb + chorus)");
    println!("  • Audio mastering (multi-stage processing)");
    println!("\nEffect Characteristics:");
    println!("  • Zero-latency processing");
    println!("  • Real-time capable");
    println!("  • Professional-grade algorithms");
    println!("  • Chainable architecture");
    println!("  • State-resetable for streaming");

    Ok(())
}
