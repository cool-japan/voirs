//! # Voice Management Example
//!
//! This example demonstrates how to work with different voices,
//! including listing available voices, selecting voices, and
//! switching between voices at runtime.

use voirs_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<(), VoirsError> {
    voirs_sdk::logging::init_logging("info")?;
    
    println!("=== Voice Management Demo ===\n");
    
    // Create a pipeline with default voice
    let mut pipeline = VoirsPipelineBuilder::new()
        .build()
        .await?;
    
    // List all available voices
    println!("Available voices:");
    let voices = pipeline.list_voices()?;
    
    for (i, voice) in voices.iter().enumerate() {
        println!("  {}. {} ({})", i + 1, voice.name, voice.language);
        println!("     Quality: {:.1}/5, Gender: {:?}", 
                 voice.quality_score, voice.gender);
        println!("     Features: {:?}", voice.features);
        println!();
    }
    
    if voices.is_empty() {
        println!("No voices available. Using default synthesis.");
        let audio = pipeline.synthesize("No specific voice selected.")?;
        audio.save_wav("default_voice.wav")?;
        return Ok(());
    }
    
    // Demonstrate synthesis with different voices
    for (i, voice) in voices.iter().enumerate().take(3) {
        println!("Switching to voice: {} ({})", voice.name, voice.language);
        
        // Set the voice
        pipeline.set_voice(&voice.name)?;
        
        // Synthesize some text
        let text = format!("Hello! I am speaking with the {} voice.", voice.name);
        let audio = pipeline.synthesize(&text)?;
        
        // Save with voice-specific filename
        let filename = format!("voice_{}_{}.wav", i + 1, voice.name.replace(" ", "_"));
        audio.save_wav(&filename)?;
        
        println!("  Audio saved to: {}", filename);
        println!("  Duration: {:.2}s, Samples: {}", audio.duration(), audio.len());
        println!();
    }
    
    // Demonstrate voice comparison
    demonstrate_voice_comparison(&mut pipeline, &voices).await?;
    
    // Demonstrate voice features
    demonstrate_voice_features(&voices);
    
    Ok(())
}

async fn demonstrate_voice_comparison(
    pipeline: &mut VoirsPipeline, 
    voices: &[VoiceInfo]
) -> Result<(), VoirsError> {
    println!("=== Voice Comparison Demo ===\n");
    
    let sample_text = "The quick brown fox jumps over the lazy dog.";
    
    for voice in voices.iter().take(2) {
        pipeline.set_voice(&voice.name)?;
        let audio = pipeline.synthesize(sample_text)?;
        
        // Analyze audio quality
        let quality = audio.analyze_quality()?;
        
        println!("Voice: {}", voice.name);
        println!("  Text: '{}'", sample_text);
        println!("  Quality metrics:");
        println!("    SNR: {:.1} dB", quality.snr_db);
        println!("    Dynamic Range: {:.1} dB", quality.dynamic_range_db);
        println!("    Quality Grade: {:?}", quality.quality_grade());
        println!();
    }
}

fn demonstrate_voice_features(voices: &[VoiceInfo]) {
    println!("=== Voice Features Analysis ===\n");
    
    // Group voices by language
    let mut by_language: std::collections::HashMap<String, Vec<&VoiceInfo>> = 
        std::collections::HashMap::new();
    
    for voice in voices {
        by_language.entry(voice.language.clone())
            .or_insert_with(Vec::new)
            .push(voice);
    }
    
    for (language, lang_voices) in by_language {
        println!("Language: {}", language);
        for voice in lang_voices {
            println!("  {}: Quality {:.1}, Speed {:.1}x", 
                     voice.name, voice.quality_score, voice.speed_factor);
        }
        println!();
    }
    
    // Find best quality voice
    if let Some(best_voice) = voices.iter().max_by(|a, b| 
        a.quality_score.partial_cmp(&b.quality_score).unwrap()
    ) {
        println!("Highest quality voice: {} ({:.1}/5)", 
                 best_voice.name, best_voice.quality_score);
    }
    
    // Find fastest voice
    if let Some(fastest_voice) = voices.iter().max_by(|a, b| 
        a.speed_factor.partial_cmp(&b.speed_factor).unwrap()
    ) {
        println!("Fastest voice: {} ({:.1}x speed)", 
                 fastest_voice.name, fastest_voice.speed_factor);
    }
}

/// Example of voice selection based on criteria
#[tokio::main]
async fn voice_selection_example() -> Result<(), VoirsError> {
    let pipeline = VoirsPipelineBuilder::new().build().await?;
    let voices = pipeline.list_voices()?;
    
    // Select voice by language
    let english_voices: Vec<_> = voices.iter()
        .filter(|v| v.language.starts_with("en"))
        .collect();
    
    println!("English voices: {}", english_voices.len());
    
    // Select voice by quality
    let high_quality_voices: Vec<_> = voices.iter()
        .filter(|v| v.quality_score >= 4.0)
        .collect();
    
    println!("High quality voices (4.0+): {}", high_quality_voices.len());
    
    // Select voice by gender
    let female_voices: Vec<_> = voices.iter()
        .filter(|v| matches!(v.gender, Some(Gender::Female)))
        .collect();
    
    println!("Female voices: {}", female_voices.len());
    
    Ok(())
}

/// Example of voice caching and performance
#[tokio::main]
async fn voice_performance_example() -> Result<(), VoirsError> {
    let mut pipeline = VoirsPipelineBuilder::new()
        .with_cache_size(512) // 512MB cache
        .build()
        .await?;
    
    let voices = pipeline.list_voices()?;
    if voices.is_empty() {
        return Ok(());
    }
    
    let test_voice = &voices[0];
    let test_text = "Performance test synthesis.";
    
    // First synthesis (cold)
    let start = std::time::Instant::now();
    pipeline.set_voice(&test_voice.name)?;
    let _audio1 = pipeline.synthesize(test_text)?;
    let cold_time = start.elapsed();
    
    // Second synthesis (warm)
    let start = std::time::Instant::now();
    let _audio2 = pipeline.synthesize(test_text)?;
    let warm_time = start.elapsed();
    
    println!("Voice performance for '{}':", test_voice.name);
    println!("  Cold synthesis: {:?}", cold_time);
    println!("  Warm synthesis: {:?}", warm_time);
    println!("  Speedup: {:.2}x", cold_time.as_secs_f64() / warm_time.as_secs_f64());
    
    Ok(())
}