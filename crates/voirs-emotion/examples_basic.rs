//! # Basic VoiRS Emotion Control Examples
//!
//! This file contains foundational examples for getting started with the VoiRS
//! emotion control system, covering basic emotion processing, interpolation,
//! and common use cases.

use voirs_emotion::prelude::*;
use tokio::time::Duration;
use std::collections::HashMap;

/// Example 1: Basic Emotion Processing
/// 
/// This example shows the simplest way to create an emotion processor
/// and apply basic emotions.
#[tokio::main]
async fn example_basic_emotion_processing() -> Result<()> {
    println!("=== Basic Emotion Processing ===");
    
    // Create a basic emotion processor with default configuration
    let processor = EmotionProcessor::new()?;
    
    // Set a simple emotion
    processor.set_emotion(Emotion::Happy, Some(0.8)).await?;
    
    // Get the current emotion state
    let state = processor.get_current_state().await;
    println!("Current emotion state: {:?}", state.current.emotion_vector);
    
    // Check if the processor is transitioning between emotions
    if state.is_transitioning() {
        println!("Processor is currently transitioning between emotions");
        
        // You can get the target state as well
        if let Some(target) = &state.target {
            println!("Target emotion: {:?}", target.emotion_vector);
        }
    }
    
    // Process some audio with the current emotion
    let input_audio = vec![0.1, -0.2, 0.3, -0.1, 0.2, 0.4, -0.3, 0.1];
    let processed_audio = processor.process_audio(&input_audio).await?;
    
    println!("Input audio length: {}", input_audio.len());
    println!("Processed audio length: {}", processed_audio.len());
    println!("Audio was modified by emotion processing: {}", 
             input_audio != processed_audio);
    
    Ok(())
}

/// Example 2: Emotion Intensity and Multiple Emotions
///
/// Shows how to work with different emotion intensities and mix multiple emotions.
#[tokio::main]
async fn example_emotion_intensity_and_mixing() -> Result<()> {
    println!("=== Emotion Intensity and Mixing ===");
    
    let processor = EmotionProcessor::new()?;
    
    // Test different intensity levels for the same emotion
    let intensities = [0.2, 0.5, 0.8, 1.0];
    
    for intensity in &intensities {
        processor.set_emotion(Emotion::Sad, Some(*intensity)).await?;
        let state = processor.get_current_state().await;
        
        if let Some(sad_value) = state.current.emotion_vector.emotions.get(&Emotion::Sad) {
            println!("Set sadness to {:.1}, actual value: {:.3}", intensity, sad_value);
        }
    }
    
    // Mix multiple emotions
    let mut emotion_mix = HashMap::new();
    emotion_mix.insert(Emotion::Happy, 0.6);
    emotion_mix.insert(Emotion::Excited, 0.4);
    emotion_mix.insert(Emotion::Confident, 0.3);
    
    processor.set_emotion_mix(emotion_mix).await?;
    
    let mixed_state = processor.get_current_state().await;
    println!("\nMixed emotion state:");
    for (emotion, intensity) in &mixed_state.current.emotion_vector.emotions {
        println!("  {:?}: {:.3}", emotion, intensity);
    }
    
    // Get the dominant emotion
    let dominant = mixed_state.current.emotion_vector.dominant_emotion();
    if let Some((emotion, intensity)) = dominant {
        println!("Dominant emotion: {:?} at {:.3} intensity", emotion, intensity);
    }
    
    Ok(())
}

/// Example 3: Emotion Interpolation and Transitions
///
/// Demonstrates smooth transitions between different emotional states.
#[tokio::main]
async fn example_emotion_interpolation() -> Result<()> {
    println!("=== Emotion Interpolation and Transitions ===");
    
    // Create processor with specific transition smoothing
    let config = EmotionConfig::builder()
        .transition_smoothing(0.8) // Smooth transitions
        .build()?;
    
    let processor = EmotionProcessor::with_config(config)?;
    
    // Start with one emotion
    processor.set_emotion(Emotion::Calm, Some(0.7)).await?;
    println!("Starting with calm emotion");
    
    // Transition to a different emotion
    processor.set_emotion(Emotion::Excited, Some(0.9)).await?;
    println!("Transitioning to excited emotion");
    
    // Monitor the transition over time
    for step in 0..10 {
        // Simulate 100ms time steps
        processor.update_transition(100.0).await?;
        
        let state = processor.get_current_state().await;
        let calm_intensity = state.current.emotion_vector.emotions
            .get(&Emotion::Calm).unwrap_or(&0.0);
        let excited_intensity = state.current.emotion_vector.emotions
            .get(&Emotion::Excited).unwrap_or(&0.0);
        
        println!("Step {}: Calm={:.3}, Excited={:.3}, Transitioning={}",
                 step, calm_intensity, excited_intensity, state.is_transitioning());
        
        if !state.is_transitioning() {
            println!("Transition completed at step {}", step);
            break;
        }
    }
    
    // Manual interpolation between emotions
    let interpolator = EmotionInterpolator::new(InterpolationMethod::Linear);
    
    // Create two emotion states to interpolate between
    let happy_state = EmotionState {
        emotion_vector: {
            let mut ev = EmotionVector::new();
            ev.set_emotion(Emotion::Happy, 0.8)?;
            ev
        },
        emotion_parameters: EmotionParameters {
            arousal: 0.8,
            valence: 0.9,
            dominance: 0.7,
            intensity: EmotionIntensity::new(0.8),
            pitch_shift: 1.2,
            tempo_scale: 1.1,
            energy_scale: 1.3,
            breathiness: 0.0,
            roughness: 0.0,
            brightness: 0.4,
            resonance: 0.3,
        },
        timestamp: std::time::Instant::now(),
        confidence: 1.0,
        context: Some("happy_example".to_string()),
    };
    
    let sad_state = EmotionState {
        emotion_vector: {
            let mut ev = EmotionVector::new();
            ev.set_emotion(Emotion::Sad, 0.6)?;
            ev
        },
        emotion_parameters: EmotionParameters {
            arousal: 0.3,
            valence: 0.2,
            dominance: 0.4,
            intensity: EmotionIntensity::new(0.6),
            pitch_shift: 0.8,
            tempo_scale: 0.9,
            energy_scale: 0.7,
            breathiness: 0.2,
            roughness: 0.1,
            brightness: -0.2,
            resonance: 0.1,
        },
        timestamp: std::time::Instant::now(),
        confidence: 1.0,
        context: Some("sad_example".to_string()),
    };
    
    // Interpolate at different points
    println!("\nManual interpolation between happy and sad:");
    for i in 0..=5 {
        let t = i as f32 / 5.0; // 0.0 to 1.0
        let interpolated = interpolator.interpolate(&happy_state, &sad_state, t)?;
        
        println!("t={:.1}: pitch_shift={:.2}, valence={:.2}, energy_scale={:.2}",
                 t, 
                 interpolated.emotion_parameters.pitch_shift,
                 interpolated.emotion_parameters.valence,
                 interpolated.emotion_parameters.energy_scale);
    }
    
    Ok(())
}

/// Example 4: Working with Emotion Presets
///
/// Shows how to use predefined emotion presets for common emotional expressions.
#[tokio::main]
async fn example_emotion_presets() -> Result<()> {
    println!("=== Working with Emotion Presets ===");
    
    // Get the default preset library
    let preset_library = EmotionPresetLibrary::default();
    
    // List available presets
    println!("Available emotion presets:");
    for preset_name in preset_library.list_presets() {
        if let Some(preset) = preset_library.get_preset(&preset_name) {
            println!("  {}: {} (intensity: {:.1})", 
                     preset_name, 
                     preset.description, 
                     preset.base_intensity.value());
        }
    }
    
    // Use a preset with an emotion processor
    let processor = EmotionProcessor::new()?;
    
    // Apply different presets
    let preset_names = ["happy", "sad", "angry", "calm", "excited"];
    
    for preset_name in &preset_names {
        if let Some(preset) = preset_library.get_preset(preset_name) {
            // Apply the preset
            processor.apply_emotion_parameters(preset.parameters.clone()).await?;
            
            let state = processor.get_current_state().await;
            println!("\nApplied '{}' preset:", preset_name);
            println!("  Description: {}", preset.description);
            println!("  Arousal: {:.2}", state.current.emotion_parameters.arousal);
            println!("  Valence: {:.2}", state.current.emotion_parameters.valence);
            println!("  Pitch shift: {:.2}", state.current.emotion_parameters.pitch_shift);
            println!("  Tempo scale: {:.2}", state.current.emotion_parameters.tempo_scale);
        }
    }
    
    // Create and add a custom preset
    let custom_preset = EmotionPreset {
        name: "mysterious".to_string(),
        description: "A secretive, intriguing emotional tone".to_string(),
        base_intensity: EmotionIntensity::new(0.6),
        parameters: EmotionParameters {
            arousal: 0.4,
            valence: 0.0, // Neutral valence
            dominance: 0.7,
            intensity: EmotionIntensity::new(0.6),
            pitch_shift: 0.9,
            tempo_scale: 0.85,
            energy_scale: 0.8,
            breathiness: 0.3,
            roughness: 0.1,
            brightness: -0.3,
            resonance: 0.4,
        },
        tags: vec!["mysterious".to_string(), "intriguing".to_string()],
        cultural_context: Some("universal".to_string()),
    };
    
    let mut custom_library = preset_library.clone();
    custom_library.add_preset(custom_preset)?;
    
    // Use the custom preset
    if let Some(mysterious_preset) = custom_library.get_preset("mysterious") {
        processor.apply_emotion_parameters(mysterious_preset.parameters.clone()).await?;
        println!("\nApplied custom 'mysterious' preset");
        
        let state = processor.get_current_state().await;
        println!("  Valence: {:.2} (neutral)", state.current.emotion_parameters.valence);
        println!("  Breathiness: {:.2} (whispery)", state.current.emotion_parameters.breathiness);
        println!("  Brightness: {:.2} (darker tone)", state.current.emotion_parameters.brightness);
    }
    
    Ok(())
}

/// Example 5: Basic Configuration and Customization
///
/// Demonstrates how to configure the emotion processor for different use cases.
#[tokio::main]
async fn example_basic_configuration() -> Result<()> {
    println!("=== Basic Configuration and Customization ===");
    
    // Create different configurations for different scenarios
    
    // 1. Real-time configuration (fast transitions, minimal latency)
    let realtime_config = EmotionConfig::builder()
        .enabled(true)
        .transition_smoothing(0.3) // Fast transitions
        .prosody_strength(0.8)     // Strong prosodic effects
        .voice_quality_strength(0.6) // Moderate voice quality changes
        .max_emotions(3)           // Limit complexity for performance
        .build()?;
    
    let realtime_processor = EmotionProcessor::with_config(realtime_config)?;
    
    // 2. High-quality configuration (smooth transitions, full effects)
    let quality_config = EmotionConfig::builder()
        .enabled(true)
        .transition_smoothing(0.9) // Very smooth transitions
        .prosody_strength(1.0)     // Full prosodic effects
        .voice_quality_strength(0.8) // Strong voice quality changes
        .max_emotions(5)           // Allow complex emotion mixing
        .build()?;
    
    let quality_processor = EmotionProcessor::with_config(quality_config)?;
    
    // 3. Subtle configuration (gentle effects, natural-sounding)
    let subtle_config = EmotionConfig::builder()
        .enabled(true)
        .transition_smoothing(0.7)
        .prosody_strength(0.4)     // Gentle prosodic effects
        .voice_quality_strength(0.3) // Minimal voice quality changes
        .max_emotions(2)           // Simple emotion mixing
        .build()?;
    
    let subtle_processor = EmotionProcessor::with_config(subtle_config)?;
    
    // Test the same emotion with different configurations
    let test_emotion = Emotion::Happy;
    let test_intensity = Some(0.8);
    
    println!("Testing {} emotion with intensity {:.1} across configurations:\n", 
             format!("{:?}", test_emotion), test_intensity.unwrap());
    
    // Apply to all processors
    realtime_processor.set_emotion(test_emotion.clone(), test_intensity).await?;
    quality_processor.set_emotion(test_emotion.clone(), test_intensity).await?;
    subtle_processor.set_emotion(test_emotion.clone(), test_intensity).await?;
    
    // Compare the results
    let configs = [
        ("Real-time", &realtime_processor),
        ("High-quality", &quality_processor), 
        ("Subtle", &subtle_processor),
    ];
    
    for (name, processor) in &configs {
        let state = processor.get_current_state().await;
        let params = &state.current.emotion_parameters;
        
        println!("{} configuration:", name);
        println!("  Pitch shift: {:.3}", params.pitch_shift);
        println!("  Tempo scale: {:.3}", params.tempo_scale);
        println!("  Energy scale: {:.3}", params.energy_scale);
        println!("  Breathiness: {:.3}", params.breathiness);
        println!("  Brightness: {:.3}", params.brightness);
        println!();
    }
    
    // Process audio with different configurations
    let test_audio = vec![0.1; 1000]; // 1000 samples of test audio
    
    println!("Audio processing comparison:");
    for (name, processor) in &configs {
        let start = std::time::Instant::now();
        let processed = processor.process_audio(&test_audio).await?;
        let duration = start.elapsed();
        
        // Calculate a simple energy measure
        let energy: f32 = processed.iter().map(|x| x * x).sum();
        let energy_change = (energy / test_audio.len() as f32) / 
                           (test_audio.iter().map(|x| x * x).sum::<f32>() / test_audio.len() as f32);
        
        println!("  {}: {:.2}ms processing, energy change: {:.2}x", 
                 name, duration.as_secs_f64() * 1000.0, energy_change);
    }
    
    Ok(())
}

/// Example 6: Error Handling and Validation
///
/// Shows proper error handling patterns when working with the emotion system.
#[tokio::main]
async fn example_error_handling() -> Result<()> {
    println!("=== Error Handling and Validation ===");
    
    // Test configuration validation
    println!("Testing configuration validation:");
    
    // This should work fine
    match EmotionConfig::builder()
        .transition_smoothing(0.8)
        .prosody_strength(0.7)
        .voice_quality_strength(0.6)
        .build() {
        Ok(_config) => println!("✓ Valid configuration created successfully"),
        Err(e) => println!("✗ Configuration error: {}", e),
    }
    
    // Test invalid configuration values
    match EmotionConfig::builder()
        .transition_smoothing(1.5) // Invalid: > 1.0
        .build() {
        Ok(_config) => println!("✗ Invalid configuration was accepted (this shouldn't happen)"),
        Err(e) => println!("✓ Invalid configuration properly rejected: {}", e),
    }
    
    // Test emotion intensity validation
    println!("\nTesting emotion intensity validation:");
    
    // Valid intensity
    match EmotionIntensity::new(0.8) {
        Ok(intensity) => println!("✓ Valid intensity {:.1} created", intensity.value()),
        Err(e) => println!("✗ Valid intensity rejected: {}", e),
    }
    
    // Invalid intensity
    match EmotionIntensity::new(1.5) {
        Ok(intensity) => println!("✗ Invalid intensity {:.1} was accepted", intensity.value()),
        Err(e) => println!("✓ Invalid intensity properly rejected: {}", e),
    }
    
    // Test processor operations with error handling
    println!("\nTesting processor error handling:");
    
    let processor = match EmotionProcessor::new() {
        Ok(p) => {
            println!("✓ Processor created successfully");
            p
        },
        Err(e) => {
            println!("✗ Failed to create processor: {}", e);
            return Err(e);
        }
    };
    
    // Test valid emotion setting
    match processor.set_emotion(Emotion::Happy, Some(0.7)).await {
        Ok(_) => println!("✓ Emotion set successfully"),
        Err(e) => println!("✗ Failed to set emotion: {}", e),
    }
    
    // Test invalid emotion mixing (empty map)
    match processor.set_emotion_mix(HashMap::new()).await {
        Ok(_) => println!("✗ Empty emotion mix was accepted"),
        Err(e) => println!("✓ Empty emotion mix properly rejected: {}", e),
    }
    
    // Test audio processing with invalid input
    let empty_audio: Vec<f32> = vec![];
    match processor.process_audio(&empty_audio).await {
        Ok(result) => println!("✓ Empty audio processed, output length: {}", result.len()),
        Err(e) => println!("Note: Empty audio processing error: {}", e),
    }
    
    // Test audio processing with normal input
    let normal_audio = vec![0.1, -0.2, 0.3, -0.1, 0.2];
    match processor.process_audio(&normal_audio).await {
        Ok(result) => println!("✓ Normal audio processed, output length: {}", result.len()),
        Err(e) => println!("✗ Normal audio processing failed: {}", e),
    }
    
    // Demonstrate graceful degradation
    println!("\nDemonstrating graceful degradation:");
    
    // Try to get state even if something goes wrong
    let state = processor.get_current_state().await;
    println!("Current state retrieved: confidence={:.2}, transitioning={}", 
             state.current.confidence, state.is_transitioning());
    
    // Reset to neutral as a safe fallback
    match processor.reset_to_neutral().await {
        Ok(_) => println!("✓ Successfully reset to neutral state"),
        Err(e) => println!("✗ Failed to reset to neutral: {}", e),
    }
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎭 VoiRS Emotion Control System - Basic Examples\n");
    
    // Run all basic examples
    example_basic_emotion_processing().await?;
    println!("\n{}\n", "=".repeat(50));
    
    example_emotion_intensity_and_mixing().await?;
    println!("\n{}\n", "=".repeat(50));
    
    example_emotion_interpolation().await?;
    println!("\n{}\n", "=".repeat(50));
    
    example_emotion_presets().await?;
    println!("\n{}\n", "=".repeat(50));
    
    example_basic_configuration().await?;
    println!("\n{}\n", "=".repeat(50));
    
    example_error_handling().await?;
    
    println!("\n🎉 All basic examples completed successfully!");
    println!("\nNext steps:");
    println!("- Try the advanced examples in examples_advanced.rs");
    println!("- Check out the comprehensive API documentation");
    println!("- Run the test suite with: cargo test");
    println!("- Run benchmarks with: cargo bench");
    
    Ok(())
}